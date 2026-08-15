//! C++ (Itanium ABI) symbol demangling — turning a compiler-mangled C++
//! symbol back into the declaration a human wrote.
//!
//! The Itanium C++ ABI encodes a linkable name as `_Z <encoding>`, where
//! the encoding is a compact grammar over ASCII:
//!
//! - **Names.** A `<nested-name>` (`N ... E`) is a chain of length-prefixed
//!   source names joined with `::`, optionally carrying CV-qualifiers and a
//!   ref-qualifier that belong to the *function*, not the class. Class and
//!   namespace components can also be constructors (`C1`/`C2`/`C3`),
//!   destructors (`D0`/`D1`/`D2`), operator codes (`pl` → `operator+`), or
//!   template-ids (`I ... E`).
//! - **Types.** A single-letter table for the builtins (`i` → `int`), plus
//!   type constructors for pointers, references, arrays, function types,
//!   pointer-to-member, and CV-qualification. Function and array types need
//!   declarator-style printing, so a type is carried here as a *left* and a
//!   *right* half with the declarator slot in between: `void (` + `*` + `)()`.
//! - **Compression.** Every non-builtin type and every prefix of a nested
//!   name becomes a numbered substitution (`S_`, `S0_`, … base-36), and a
//!   handful of `std` entities have fixed abbreviations (`St`, `Sa`, `Ss`, …).
//!   Getting the table's population order right is most of the work.
//! - **Special names.** `_ZTV`/`_ZTI`/`_ZTS`/`_ZTT` name a class's vtable,
//!   typeinfo, typeinfo name, and VTT; `_ZGV` names a static-init guard;
//!   `_ZTh`/`_ZTv` name thunks wrapping another whole encoding.
//!
//! **Printing convention.** Qualifiers print *east*: `char const*`, not
//! `const char*`; `int const&`, not `const int&`. Template arguments join
//! with `, ` and close without a padding space (`vector<int, allocator<int>>`).
//!
//! **Best-effort, no-panic contract.** [`demangle`] never fails and never
//! panics: an input that is not a recognized Itanium mangling — or one that
//! is truncated, malformed, or adversarial — is returned unchanged. Every
//! length in the stream is treated as attacker-controlled, so the parser
//! bounds input size, recursion depth, substitution-table size, rendered
//! output, and total copying work, and bails to `None` past any cap.
//!
//! Deliberately *out of scope*, and refused rather than guessed at:
//! expressions in template arguments (`X ... E`), lambdas and unnamed types
//! (`Ul`, `Ut`), local names (`Z ... E`), and `decltype`. ABI tags
//! (`B <source-name>`) are parsed and dropped.
//!
//! Rust's legacy mangling is also valid Itanium, so this module will happily
//! demangle a `_ZN4core..E` symbol; callers wanting Rust spelling should try
//! [`crate::demangle`] first.
//!
//! This is a clean-room implementation written from the public Itanium C++
//! ABI grammar, in the spirit of `c++filt`, not a port of any demangler.

// ---------------------------------------------------------------------------
// Resource caps
// ---------------------------------------------------------------------------

/// Longest symbol accepted. Real manglings run to a few hundred bytes; the
/// cap keeps a pathological input from reaching the parser at all.
const MAX_INPUT: usize = 4096;

/// Maximum recursion depth across the name/type/template productions.
const MAX_DEPTH: u32 = 64;

/// Maximum substitution-table size, which also bounds any single seq-id.
const MAX_SUBS: usize = 512;

/// Maximum rendered length of any single fragment, and of the result.
const MAX_OUTPUT: usize = 1 << 14;

/// Total byte-copying budget. Substitutions let a short input describe a
/// large tree; charging every constructed fragment its own length makes the
/// total work linear in this budget however the references are shaped.
const MAX_BUDGET: usize = 1 << 20;

/// Maximum parameters in one function type, bounding the join loop.
const MAX_PARAMS: usize = 1024;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// True if `symbol` carries a prefix this module recognizes.
///
/// A cheap syntactic check only — a `true` result does not guarantee the
/// body parses. Used to decide quickly whether demangling is worth trying.
pub fn is_mangled(symbol: &str) -> bool {
    strip_prefix(symbol).is_some_and(|rest| !rest.is_empty())
}

/// Demangle `symbol`, or return it unchanged if it is not recognized.
///
/// Total and panic-free by contract: any input maps to some string, and a
/// non-mangled or malformed input maps to itself. This is the wrapper most
/// callers want; [`try_demangle`] exposes the recognized/not-recognized
/// distinction.
pub fn demangle(symbol: &str) -> String {
    try_demangle(symbol).unwrap_or_else(|| symbol.to_string())
}

/// Demangle `symbol`, returning `None` when it is not a recognized Itanium
/// mangling (or is malformed, truncated, or past a resource cap).
///
/// The whole body must parse and be consumed; a partially-understood symbol
/// is refused rather than rendered half-right.
pub fn try_demangle(symbol: &str) -> Option<String> {
    if symbol.len() > MAX_INPUT || !symbol.is_ascii() {
        return None;
    }
    let body = strip_prefix(symbol)?;

    // A trailing `.`-suffix (`.cold`, `.part.0`, `.llvm.1234`) is a linker
    // artifact, not part of the grammar, so split it off and note it.
    let (body, suffix) = match body.find('.') {
        Some(i) => (&body[..i], Some(&body[i..])),
        None => (body, None),
    };
    if body.is_empty() {
        return None;
    }

    let mut parser = Cxx::new(body.as_bytes());
    let mut out = parser.encoding()?;
    if parser.pos != parser.input.len() {
        return None;
    }
    if let Some(sfx) = suffix {
        out.push_str(" [clone ");
        out.push_str(sfx);
        out.push(']');
    }
    if out.len() > MAX_OUTPUT {
        return None;
    }
    Some(out)
}

/// Strip the Itanium prefix, accounting for the extra leading underscore a
/// Mach-O assembler prepends (`__Z`).
fn strip_prefix(s: &str) -> Option<&str> {
    s.strip_prefix("_Z").or_else(|| s.strip_prefix("__Z"))
}

// ---------------------------------------------------------------------------
// Rendered types
// ---------------------------------------------------------------------------

/// A type split at its declarator slot.
///
/// C declarator syntax puts the thing being declared *inside* the type:
/// a pointer-to-function is `void (*x)()`. Carrying the rendering as a left
/// half (`void (*`) and a right half (`)()`) lets each type constructor wrap
/// the previous one without re-parsing, and the bare type is just the two
/// halves concatenated. For ordinary types the right half is empty.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Ty {
    left: String,
    right: String,
}

impl Ty {
    fn plain(text: impl Into<String>) -> Ty {
        Ty {
            left: text.into(),
            right: String::new(),
        }
    }

    /// The type with nothing in the declarator slot.
    fn flat(&self) -> String {
        let mut s = String::with_capacity(self.left.len() + self.right.len());
        s.push_str(&self.left);
        s.push_str(&self.right);
        s
    }

    /// True if the declarator slot is at the end, so suffixes may be
    /// appended directly rather than parenthesized.
    fn is_simple(&self) -> bool {
        self.right.is_empty()
    }
}

/// Wrap `t` in an indirection (`*`, `&`, `&&`), parenthesizing when the
/// type's declarator slot is interior (function and array types).
fn indirect(t: &Ty, sym: &str) -> Ty {
    if t.is_simple() {
        Ty::plain(format!("{}{}", t.left, sym))
    } else {
        Ty {
            left: format!("{}({}", t.left, sym),
            right: format!("){}", t.right),
        }
    }
}

/// CV-qualifiers, in the order the grammar spells them (`r`, `V`, `K`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Cv {
    const_: bool,
    volatile_: bool,
    restrict_: bool,
}

impl Cv {
    fn any(self) -> bool {
        self.const_ || self.volatile_ || self.restrict_
    }

    /// The trailing qualifier text, east-const style, with a leading space.
    fn text(self) -> &'static str {
        match (self.const_, self.volatile_, self.restrict_) {
            (false, false, false) => "",
            (true, false, false) => " const",
            (false, true, false) => " volatile",
            (true, true, false) => " const volatile",
            (false, false, true) => " restrict",
            (true, false, true) => " const restrict",
            (false, true, true) => " volatile restrict",
            (true, true, true) => " const volatile restrict",
        }
    }
}

/// Apply CV-qualifiers to a rendered type, east-const.
fn apply_cv(t: &Ty, cv: Cv) -> Ty {
    let q = cv.text();
    if q.is_empty() {
        return t.clone();
    }
    if t.is_simple() {
        Ty::plain(format!("{}{}", t.left, q))
    } else {
        // A cv-qualified function type: the qualifiers trail the parameter
        // list, as in `void (C::*)() const`.
        Ty {
            left: t.left.clone(),
            right: format!("{}{}", t.right, q),
        }
    }
}

// ---------------------------------------------------------------------------
// Parser state
// ---------------------------------------------------------------------------

/// One parsed `<name>`, plus the facts the enclosing `<encoding>` needs.
struct NameInfo {
    text: String,
    /// The name ended in a `<template-args>` list, so the bare-function-type
    /// that follows leads with an encoded return type.
    templated: bool,
    /// Constructors, destructors, and conversion operators never encode a
    /// return type, even when templated.
    suppress_ret: bool,
    /// The name was exactly a substitution reference, so it must not be
    /// entered into the table a second time.
    plain_sub: bool,
    /// CV-qualifiers and ref-qualifier from a `<nested-name>`; these belong
    /// to the member function and print after its parameter list.
    cv: Cv,
    ref_q: &'static str,
}

/// One parsed `<unqualified-name>`.
struct Uqn {
    text: String,
    suppress_ret: bool,
}

/// One parsed `<substitution>`.
struct SubRef {
    ty: Ty,
    /// The `St` abbreviation, the only substitution a bare
    /// `<unqualified-name>` may directly follow.
    std_prefix: bool,
}

/// Recursive-descent parser for one Itanium symbol body.
///
/// Holds the input, a byte cursor, the substitution table, the template
/// arguments currently in scope (so `T_` can render as the argument it names
/// rather than a placeholder), and the budgets that enforce the
/// no-panic/no-hang contract.
struct Cxx<'a> {
    input: &'a [u8],
    pos: usize,
    depth: u32,
    budget: usize,
    subs: Vec<Ty>,
    /// Template arguments of the name being encoded, for `<template-param>`
    /// resolution. Frozen once the bare-function-type starts so a parameter
    /// type's own arguments cannot shadow them.
    targs: Vec<String>,
    targs_frozen: bool,
}

impl<'a> Cxx<'a> {
    fn new(input: &'a [u8]) -> Cxx<'a> {
        Cxx {
            input,
            pos: 0,
            depth: 0,
            budget: MAX_BUDGET,
            subs: Vec::new(),
            targs: Vec::new(),
            targs_frozen: false,
        }
    }

    // -- cursor primitives ------------------------------------------------

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.input.get(self.pos.checked_add(offset)?).copied()
    }

    fn take(&mut self) -> Option<u8> {
        let b = self.input.get(self.pos).copied()?;
        self.pos += 1;
        Some(b)
    }

    /// Consume the next byte iff it equals `b`; report whether it did.
    fn eat(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    // -- budgets ----------------------------------------------------------

    /// Charge `n` units of the copying budget, failing when it is spent.
    fn spend(&mut self, n: usize) -> Option<()> {
        self.budget = self.budget.checked_sub(n)?;
        Some(())
    }

    /// Charge a freshly built fragment and reject it if it blew the cap.
    fn build(&mut self, s: String) -> Option<String> {
        self.spend(s.len())?;
        if s.len() > MAX_OUTPUT {
            return None;
        }
        Some(s)
    }

    /// The [`build`](Self::build) equivalent for a two-halved type.
    fn build_ty(&mut self, t: Ty) -> Option<Ty> {
        let n = t.left.len().checked_add(t.right.len())?;
        self.spend(n)?;
        if n > MAX_OUTPUT {
            return None;
        }
        Some(t)
    }

    /// Enter a component into the substitution table.
    fn add_sub(&mut self, t: Ty) -> Option<()> {
        if self.subs.len() >= MAX_SUBS {
            return None;
        }
        self.spend(t.left.len() + t.right.len())?;
        self.subs.push(t);
        Some(())
    }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

impl Cxx<'_> {
    /// `<encoding> ::= <name> <bare-function-type> | <name> | <special-name>`
    fn encoding(&mut self) -> Option<String> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let r = self.encoding_body();
        self.depth -= 1;
        r
    }

    fn encoding_body(&mut self) -> Option<String> {
        self.spend(1)?;
        if matches!(self.peek()?, b'T' | b'G') {
            return self.special_name();
        }

        let name = self.name()?;

        // Nothing follows: a data name. `E` can follow when this encoding is
        // nested inside an `L <mangled-name> E` template argument.
        if self.pos >= self.input.len() || self.peek() == Some(b'E') {
            if name.cv.any() || !name.ref_q.is_empty() {
                return None;
            }
            return Some(name.text);
        }

        // Everything from here is the function's own type, so the template
        // arguments in scope are fixed at the name's.
        self.targs_frozen = true;
        let has_ret = name.templated && !name.suppress_ret;
        let (ret, params) = self.bare_function_type(has_ret)?;

        let decl = self.build(format!(
            "{}{}{}{}",
            name.text,
            params,
            name.cv.text(),
            name.ref_q
        ))?;
        match ret {
            None => Some(decl),
            // The return type has a declarator slot of its own: a function
            // returning a function pointer reads `void (*f(int))()`.
            Some(r) if r.is_simple() => self.build(format!("{} {}", r.left, decl)),
            Some(r) => self.build(format!("{}{}{}", r.left, decl, r.right)),
        }
    }

    /// `<bare-function-type> ::= <type>+`, the first being the return type
    /// only where the grammar puts one (template functions).
    fn bare_function_type(&mut self, has_ret: bool) -> Option<(Option<Ty>, String)> {
        let ret = if has_ret { Some(self.ty()?) } else { None };
        let params = self.parameter_list()?;
        Some((ret, params))
    }

    /// Parse types up to end-of-input or `E` and render them as `(a, b)`,
    /// with a lone `void` collapsing to `()`.
    fn parameter_list(&mut self) -> Option<String> {
        let mut params: Vec<String> = Vec::new();
        while self.pos < self.input.len() && self.peek() != Some(b'E') {
            params.push(self.ty()?.flat());
            if params.len() > MAX_PARAMS {
                return None;
            }
        }
        if params.is_empty() {
            return None;
        }
        if params.len() == 1 && params[0] == "void" {
            return Some("()".to_string());
        }
        self.build(format!("({})", params.join(", ")))
    }

    // -- special names ----------------------------------------------------

    /// `<special-name>`: vtables, typeinfo, guard variables, and thunks.
    fn special_name(&mut self) -> Option<String> {
        match self.take()? {
            b'T' => match self.peek()? {
                b'V' => {
                    self.pos += 1;
                    let t = self.ty()?.flat();
                    self.build(format!("vtable for {t}"))
                }
                b'T' => {
                    self.pos += 1;
                    let t = self.ty()?.flat();
                    self.build(format!("VTT for {t}"))
                }
                b'I' => {
                    self.pos += 1;
                    let t = self.ty()?.flat();
                    self.build(format!("typeinfo for {t}"))
                }
                b'S' => {
                    self.pos += 1;
                    let t = self.ty()?.flat();
                    self.build(format!("typeinfo name for {t}"))
                }
                b'H' => {
                    self.pos += 1;
                    let n = self.name()?.text;
                    self.build(format!("thread-local initialization routine for {n}"))
                }
                b'W' => {
                    self.pos += 1;
                    let n = self.name()?.text;
                    self.build(format!("thread-local wrapper routine for {n}"))
                }
                b'c' => {
                    self.pos += 1;
                    self.call_offset()?;
                    self.call_offset()?;
                    let inner = self.encoding()?;
                    self.build(format!("covariant return thunk to {inner}"))
                }
                // `T <call-offset> <encoding>`: the call-offset's own tag
                // (`h` non-virtual, `v` virtual) is the byte we are looking
                // at, so it is left for `call_offset` to consume.
                tag @ (b'h' | b'v') => {
                    self.call_offset()?;
                    let inner = self.encoding()?;
                    let kind = if tag == b'v' {
                        "virtual"
                    } else {
                        "non-virtual"
                    };
                    self.build(format!("{kind} thunk to {inner}"))
                }
                _ => None,
            },
            b'G' => match self.take()? {
                b'V' => {
                    let n = self.name()?.text;
                    self.build(format!("guard variable for {n}"))
                }
                b'R' => {
                    let n = self.name()?.text;
                    // An optional discriminating seq-id, then `_`.
                    while let Some(c) = self.peek() {
                        if c == b'_' {
                            break;
                        }
                        if !(c.is_ascii_digit() || c.is_ascii_uppercase()) {
                            return None;
                        }
                        self.pos += 1;
                    }
                    if !self.eat(b'_') {
                        return None;
                    }
                    self.build(format!("reference temporary for {n}"))
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// `<call-offset> ::= h <nv-offset> _ | v <v-offset> _`. Consumed for
    /// its length only — the offsets themselves are not printed.
    fn call_offset(&mut self) -> Option<()> {
        match self.take()? {
            b'h' => self.skip_number()?,
            b'v' => {
                self.skip_number()?;
                if !self.eat(b'_') {
                    return None;
                }
                self.skip_number()?;
            }
            _ => return None,
        }
        if !self.eat(b'_') {
            return None;
        }
        Some(())
    }

    /// `<number> ::= [n] <digit>+`, consumed without being interpreted.
    fn skip_number(&mut self) -> Option<()> {
        self.eat(b'n');
        let start = self.pos;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == start { None } else { Some(()) }
    }
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

impl Cxx<'_> {
    /// `<name> ::= <nested-name> | <unscoped-name>`
    /// `        | <unscoped-template-name> <template-args> | <local-name>`
    ///
    /// Local names (`Z ... E`) are out of scope and refused.
    fn name(&mut self) -> Option<NameInfo> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let r = self.name_body();
        self.depth -= 1;
        r
    }

    fn name_body(&mut self) -> Option<NameInfo> {
        self.spend(1)?;
        match self.peek()? {
            b'N' => self.nested_name(),
            b'Z' => None,
            b'S' => {
                let sub = self.substitution()?;
                let mut text = sub.ty.flat();
                let mut plain_sub = true;
                let mut suppress = false;
                // Only the `St` abbreviation may be followed directly by an
                // unqualified name: `<unscoped-name> ::= St <unqualified-name>`.
                if sub.std_prefix && self.starts_unqualified() {
                    let u = self.unqualified_name(Some(&text))?;
                    text = self.build(format!("{}::{}", text, u.text))?;
                    suppress = u.suppress_ret;
                    plain_sub = false;
                }
                let templated = self.maybe_template_args(&mut text, plain_sub)?;
                Some(NameInfo {
                    text,
                    templated,
                    suppress_ret: suppress && !templated,
                    plain_sub: plain_sub && !templated,
                    cv: Cv::default(),
                    ref_q: "",
                })
            }
            _ => {
                let u = self.unqualified_name(None)?;
                let mut text = u.text;
                let templated = self.maybe_template_args(&mut text, false)?;
                Some(NameInfo {
                    text,
                    templated,
                    suppress_ret: u.suppress_ret,
                    plain_sub: false,
                    cv: Cv::default(),
                    ref_q: "",
                })
            }
        }
    }

    /// If a `<template-args>` list follows, apply it to `text` and report
    /// `true`. The template *name* is itself a substitution candidate, but
    /// only when it was not already written as one (`base_is_sub`).
    fn maybe_template_args(&mut self, text: &mut String, base_is_sub: bool) -> Option<bool> {
        if self.peek() != Some(b'I') {
            return Some(false);
        }
        if !base_is_sub {
            self.add_sub(Ty::plain(text.clone()))?;
        }
        let args = self.template_args()?;
        *text = self.build(format!("{}<{}>", text, args.join(", ")))?;
        Some(true)
    }

    /// `<nested-name> ::= N [<CV-qualifiers>] [<ref-qualifier>] <prefix>* E`
    ///
    /// The prefix chain is walked iteratively. Every completed component
    /// except the last is a substitution candidate, so registration is done
    /// lazily at the top of the following turn — which is also exactly the
    /// point in the byte stream where the ABI says the entry appears.
    fn nested_name(&mut self) -> Option<NameInfo> {
        if !self.eat(b'N') {
            return None;
        }
        let cv = self.cv_qualifiers();
        let ref_q = if self.eat(b'R') {
            " &"
        } else if self.eat(b'O') {
            " &&"
        } else {
            ""
        };

        let mut cur: Option<String> = None;
        // A component written as a substitution is already in the table.
        let mut cur_is_sub = false;
        let mut templated = false;
        let mut suppress = false;

        loop {
            self.spend(1)?;
            let b = self.peek()?;
            if b == b'E' {
                self.pos += 1;
                break;
            }
            if let Some(prev) = &cur
                && !cur_is_sub
            {
                let prev = prev.clone();
                self.add_sub(Ty::plain(prev))?;
            }
            cur_is_sub = false;

            match b {
                // `<template-prefix> <template-args>`.
                b'I' => {
                    let base = cur.take()?;
                    let args = self.template_args()?;
                    cur = Some(self.build(format!("{}<{}>", base, args.join(", ")))?);
                    templated = true;
                    suppress = false;
                }
                b'S' => {
                    if cur.is_some() {
                        return None;
                    }
                    let sub = self.substitution()?;
                    cur = Some(sub.ty.flat());
                    cur_is_sub = true;
                    templated = false;
                }
                b'T' => {
                    if cur.is_some() {
                        return None;
                    }
                    cur = Some(self.template_param()?);
                    templated = false;
                }
                _ => {
                    let u = self.unqualified_name(cur.as_deref())?;
                    let joined = match cur.take() {
                        Some(prefix) => self.build(format!("{}::{}", prefix, u.text))?,
                        None => u.text,
                    };
                    cur = Some(joined);
                    templated = false;
                    suppress = u.suppress_ret;
                }
            }
        }

        Some(NameInfo {
            text: cur?,
            templated,
            suppress_ret: suppress,
            plain_sub: false,
            cv,
            ref_q,
        })
    }

    /// `<CV-qualifiers> ::= [r] [V] [K]`, consumed in grammar order.
    fn cv_qualifiers(&mut self) -> Cv {
        // Order matters: the tags must be consumed as the grammar spells
        // them, so bind each before building the value.
        let restrict_ = self.eat(b'r');
        let volatile_ = self.eat(b'V');
        let const_ = self.eat(b'K');
        Cv {
            const_,
            volatile_,
            restrict_,
        }
    }

    /// True if the cursor could begin an `<unqualified-name>`.
    fn starts_unqualified(&self) -> bool {
        match self.peek() {
            Some(b) => {
                b.is_ascii_digit() || b.is_ascii_lowercase() || matches!(b, b'L' | b'C' | b'D')
            }
            None => false,
        }
    }

    /// `<unqualified-name>`: a source name, a constructor or destructor, or
    /// an operator. `enclosing` supplies the class for a ctor/dtor.
    ///
    /// GCC's internal-linkage marker (`L` before a source name) and ABI tags
    /// (`B <source-name>`) are consumed and dropped.
    fn unqualified_name(&mut self, enclosing: Option<&str>) -> Option<Uqn> {
        if self.peek() == Some(b'L') && self.peek_at(1).is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
        let b = self.peek()?;
        let uqn = if b.is_ascii_digit() {
            Uqn {
                text: self.source_name()?,
                suppress_ret: false,
            }
        } else if b == b'C' {
            if !self.peek_at(1)?.is_ascii_digit() {
                return None; // inheriting constructors are out of scope
            }
            self.pos += 2;
            Uqn {
                text: base_component(enclosing?)?,
                suppress_ret: true,
            }
        } else if b == b'D' {
            if !self.peek_at(1)?.is_ascii_digit() {
                return None;
            }
            self.pos += 2;
            let base = base_component(enclosing?)?;
            Uqn {
                text: self.build(format!("~{base}"))?,
                suppress_ret: true,
            }
        } else if b == b'U' {
            return None; // lambdas (`Ul`) and unnamed types (`Ut`)
        } else {
            self.operator_name()?
        };

        while self.peek() == Some(b'B') {
            self.pos += 1;
            self.source_name()?;
        }
        Some(uqn)
    }

    /// `<source-name> ::= <positive length> <identifier>`.
    fn source_name(&mut self) -> Option<String> {
        let start = self.pos;
        let mut len = 0usize;
        while let Some(b) = self.peek() {
            if !b.is_ascii_digit() {
                break;
            }
            len = len.checked_mul(10)?.checked_add((b - b'0') as usize)?;
            self.pos += 1;
        }
        if self.pos == start || len == 0 {
            return None;
        }
        let end = self.pos.checked_add(len)?;
        if end > self.input.len() {
            return None;
        }
        let raw = self.input.get(self.pos..end)?;
        self.pos = end;
        let text = std::str::from_utf8(raw).ok()?.to_string();
        self.build(text)
    }
}

/// The class name a constructor or destructor is named for: the last
/// component of a qualified prefix, with any template arguments removed.
fn base_component(qualified: &str) -> Option<String> {
    let bytes = qualified.as_bytes();
    let mut angle = 0i32;
    let mut last = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => angle += 1,
            b'>' => angle -= 1,
            b':' if angle == 0 && bytes.get(i + 1) == Some(&b':') => {
                last = i + 2;
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    let tail = qualified.get(last..)?;
    let cut = tail.find('<').unwrap_or(tail.len());
    let base = tail.get(..cut)?;
    if base.is_empty() {
        None
    } else {
        Some(base.to_string())
    }
}

// ---------------------------------------------------------------------------
// Operator names
// ---------------------------------------------------------------------------

/// The `<operator-name>` table, keyed by its two-letter code. `cv`, `li`,
/// and the vendor form `v <digit>` take an operand and are handled
/// separately in [`Cxx::operator_name`].
const OPERATORS: &[(&[u8; 2], &str)] = &[
    (b"aN", "operator&="),
    (b"aS", "operator="),
    (b"aa", "operator&&"),
    (b"ad", "operator&"),
    (b"an", "operator&"),
    (b"aw", "operator co_await"),
    (b"cl", "operator()"),
    (b"cm", "operator,"),
    (b"co", "operator~"),
    (b"dV", "operator/="),
    (b"da", "operator delete[]"),
    (b"de", "operator*"),
    (b"dl", "operator delete"),
    (b"ds", "operator.*"),
    (b"dt", "operator."),
    (b"dv", "operator/"),
    (b"eO", "operator^="),
    (b"eo", "operator^"),
    (b"eq", "operator=="),
    (b"ge", "operator>="),
    (b"gt", "operator>"),
    (b"ix", "operator[]"),
    (b"lS", "operator<<="),
    (b"le", "operator<="),
    (b"ls", "operator<<"),
    (b"lt", "operator<"),
    (b"mI", "operator-="),
    (b"mL", "operator*="),
    (b"mi", "operator-"),
    (b"ml", "operator*"),
    (b"mm", "operator--"),
    (b"na", "operator new[]"),
    (b"ne", "operator!="),
    (b"ng", "operator-"),
    (b"nt", "operator!"),
    (b"nw", "operator new"),
    (b"oR", "operator|="),
    (b"oo", "operator||"),
    (b"or", "operator|"),
    (b"pL", "operator+="),
    (b"pl", "operator+"),
    (b"pm", "operator->*"),
    (b"pp", "operator++"),
    (b"ps", "operator+"),
    (b"pt", "operator->"),
    (b"qu", "operator?:"),
    (b"rM", "operator%="),
    (b"rS", "operator>>="),
    (b"rm", "operator%"),
    (b"rs", "operator>>"),
    (b"ss", "operator<=>"),
];

impl Cxx<'_> {
    /// `<operator-name>`: a two-letter code, a conversion operator
    /// (`cv <type>`), a literal operator (`li <source-name>`), or a vendor
    /// extension (`v <digit> <source-name>`).
    fn operator_name(&mut self) -> Option<Uqn> {
        let a = self.peek()?;
        let b = self.peek_at(1)?;
        if a == b'c' && b == b'v' {
            self.pos += 2;
            let t = self.ty()?.flat();
            // A conversion function never encodes a return type.
            return Some(Uqn {
                text: self.build(format!("operator {t}"))?,
                suppress_ret: true,
            });
        }
        if a == b'l' && b == b'i' {
            self.pos += 2;
            let n = self.source_name()?;
            return Some(Uqn {
                text: self.build(format!("operator\"\"{n}"))?,
                suppress_ret: false,
            });
        }
        if a == b'v' && b.is_ascii_digit() {
            self.pos += 2;
            let n = self.source_name()?;
            return Some(Uqn {
                text: self.build(format!("operator {n}"))?,
                suppress_ret: false,
            });
        }
        let code = [a, b];
        for (op, rendered) in OPERATORS {
            if **op == code {
                self.pos += 2;
                return Some(Uqn {
                    text: (*rendered).to_string(),
                    suppress_ret: false,
                });
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Substitutions and template parameters
// ---------------------------------------------------------------------------

impl Cxx<'_> {
    /// `<substitution> ::= S <seq-id> _ | S_ | St | Sa | Sb | Ss | Si | So | Sd`
    ///
    /// Seq-ids are base-36 (`0`-`9`, `A`-`Z`), with `S_` naming entry 0 and
    /// `S0_` naming entry 1.
    fn substitution(&mut self) -> Option<SubRef> {
        if !self.eat(b'S') {
            return None;
        }
        let b = self.peek()?;
        if b == b'_' {
            self.pos += 1;
            return Some(SubRef {
                ty: self.subs.first()?.clone(),
                std_prefix: false,
            });
        }
        if b.is_ascii_digit() || b.is_ascii_uppercase() {
            let mut id = 0usize;
            loop {
                let c = self.take()?;
                if c == b'_' {
                    break;
                }
                let digit = if c.is_ascii_digit() {
                    (c - b'0') as usize
                } else if c.is_ascii_uppercase() {
                    (c - b'A') as usize + 10
                } else {
                    return None;
                };
                id = id.checked_mul(36)?.checked_add(digit)?;
                if id > MAX_SUBS {
                    return None;
                }
            }
            return Some(SubRef {
                ty: self.subs.get(id.checked_add(1)?)?.clone(),
                std_prefix: false,
            });
        }
        // The fixed `std` abbreviations. `Ss`/`Si`/`So`/`Sd` stand for
        // long `basic_*` instantiations and are printed by their common
        // typedef spelling.
        let (text, is_st) = match b {
            b't' => ("std", true),
            b'a' => ("std::allocator", false),
            b'b' => ("std::basic_string", false),
            b's' => ("std::string", false),
            b'i' => ("std::istream", false),
            b'o' => ("std::ostream", false),
            b'd' => ("std::iostream", false),
            _ => return None,
        };
        self.pos += 1;
        Some(SubRef {
            ty: Ty::plain(text),
            std_prefix: is_st,
        })
    }

    /// `<template-param> ::= T_ | T <seq-id> _`.
    ///
    /// Resolved against the template arguments in scope where possible, so
    /// `_Z3fooIiEvT_` prints `void foo<int>(int)` rather than a placeholder;
    /// an unresolvable index falls back to `T`, `T1`, `T2`, …
    fn template_param(&mut self) -> Option<String> {
        if !self.eat(b'T') {
            return None;
        }
        let idx = if self.eat(b'_') {
            0usize
        } else {
            let mut id = 0usize;
            loop {
                let c = self.take()?;
                if c == b'_' {
                    break;
                }
                let digit = if c.is_ascii_digit() {
                    (c - b'0') as usize
                } else if c.is_ascii_uppercase() {
                    (c - b'A') as usize + 10
                } else {
                    return None;
                };
                id = id.checked_mul(36)?.checked_add(digit)?;
                if id > MAX_SUBS {
                    return None;
                }
            }
            id.checked_add(1)?
        };
        let text = match self.targs.get(idx) {
            Some(arg) => arg.clone(),
            None if idx == 0 => "T".to_string(),
            None => format!("T{idx}"),
        };
        self.build(text)
    }

    /// `<template-args> ::= I <template-arg>+ E`.
    fn template_args(&mut self) -> Option<Vec<String>> {
        if !self.eat(b'I') {
            return None;
        }
        let mut args: Vec<String> = Vec::new();
        loop {
            self.spend(1)?;
            if self.peek()? == b'E' {
                self.pos += 1;
                break;
            }
            args.push(self.template_arg()?);
            if args.len() > MAX_PARAMS {
                return None;
            }
        }
        if args.is_empty() {
            return None;
        }
        // The outermost list of the name completes last, so plain assignment
        // leaves the name's own arguments in scope for `T_` resolution.
        if !self.targs_frozen {
            self.targs = args.clone();
        }
        Some(args)
    }

    /// `<template-arg> ::= <type> | <expr-primary> | J <template-arg>* E`.
    /// General expressions (`X ... E`) are out of scope and refused.
    fn template_arg(&mut self) -> Option<String> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let r = self.template_arg_body();
        self.depth -= 1;
        r
    }

    fn template_arg_body(&mut self) -> Option<String> {
        match self.peek()? {
            b'X' => None,
            b'L' => self.expr_primary(),
            b'J' => {
                self.pos += 1;
                let mut parts: Vec<String> = Vec::new();
                loop {
                    self.spend(1)?;
                    if self.peek()? == b'E' {
                        self.pos += 1;
                        break;
                    }
                    parts.push(self.template_arg()?);
                    if parts.len() > MAX_PARAMS {
                        return None;
                    }
                }
                self.build(parts.join(", "))
            }
            _ => Some(self.ty()?.flat()),
        }
    }

    /// `<expr-primary> ::= L <type> <value number> E | L <mangled-name> E`.
    ///
    /// Only literals whose value is unambiguously renderable are accepted:
    /// booleans, `nullptr`, and integers (with the usual C++ suffix for the
    /// wider types). Floating-point literals are encoded as raw hex and are
    /// refused rather than mis-rendered.
    fn expr_primary(&mut self) -> Option<String> {
        if !self.eat(b'L') {
            return None;
        }
        if self.peek() == Some(b'_') && self.peek_at(1) == Some(b'Z') {
            self.pos += 2;
            let inner = self.encoding()?;
            if !self.eat(b'E') {
                return None;
            }
            return Some(inner);
        }

        let ty = self.ty()?.flat();
        let negative = self.eat(b'n');
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == b'E' {
                break;
            }
            if !c.is_ascii_digit() {
                return None;
            }
            self.pos += 1;
        }
        let digits = std::str::from_utf8(self.input.get(start..self.pos)?).ok()?;
        if !self.eat(b'E') {
            return None;
        }

        match ty.as_str() {
            "bool" => match digits {
                "0" => Some("false".to_string()),
                "1" => Some("true".to_string()),
                _ => None,
            },
            "std::nullptr_t" => Some("nullptr".to_string()),
            "float" | "double" | "long double" | "__float128" => None,
            _ if digits.is_empty() => None,
            _ => {
                let suffix = match ty.as_str() {
                    "unsigned int" => "u",
                    "long" => "l",
                    "unsigned long" => "ul",
                    "long long" => "ll",
                    "unsigned long long" => "ull",
                    _ => "",
                };
                let sign = if negative { "-" } else { "" };
                self.build(format!("{sign}{digits}{suffix}"))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The single-letter `<builtin-type>` table, or `None` if `tag` is not one.
fn builtin_type(tag: u8) -> Option<&'static str> {
    Some(match tag {
        b'v' => "void",
        b'w' => "wchar_t",
        b'b' => "bool",
        b'c' => "char",
        b'a' => "signed char",
        b'h' => "unsigned char",
        b's' => "short",
        b't' => "unsigned short",
        b'i' => "int",
        b'j' => "unsigned int",
        b'l' => "long",
        b'm' => "unsigned long",
        b'x' => "long long",
        b'y' => "unsigned long long",
        b'n' => "__int128",
        b'o' => "unsigned __int128",
        b'f' => "float",
        b'd' => "double",
        b'e' => "long double",
        b'g' => "__float128",
        b'z' => "...",
        _ => return None,
    })
}

/// The two-letter `D`-prefixed builtin types.
fn d_builtin_type(tag: u8) -> Option<&'static str> {
    Some(match tag {
        b'd' => "decimal64",
        b'e' => "decimal128",
        b'f' => "decimal32",
        b'h' => "half",
        b'i' => "char32_t",
        b's' => "char16_t",
        b'u' => "char8_t",
        b'a' => "auto",
        b'c' => "decltype(auto)",
        b'n' => "std::nullptr_t",
        _ => return None,
    })
}

impl Cxx<'_> {
    /// `<type>`, in all the forms this module supports.
    ///
    /// Every type that is not a builtin becomes a substitution candidate,
    /// registered as it completes — innermost first, which is the order the
    /// ABI's compression scheme expects.
    fn ty(&mut self) -> Option<Ty> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let r = self.ty_body();
        self.depth -= 1;
        r
    }

    fn ty_body(&mut self) -> Option<Ty> {
        self.spend(1)?;
        let tag = self.peek()?;

        // `<CV-qualifiers> <type>` — printed east: `int const`.
        if matches!(tag, b'r' | b'V' | b'K') {
            let cv = self.cv_qualifiers();
            let inner = self.ty()?;
            let t = self.build_ty(apply_cv(&inner, cv))?;
            self.add_sub(t.clone())?;
            return Some(t);
        }

        match tag {
            b'P' | b'R' | b'O' => {
                self.pos += 1;
                let inner = self.ty()?;
                let sym = match tag {
                    b'P' => "*",
                    b'R' => "&",
                    _ => "&&",
                };
                let t = self.build_ty(indirect(&inner, sym))?;
                self.add_sub(t.clone())?;
                Some(t)
            }
            b'C' | b'G' => {
                self.pos += 1;
                let inner = self.ty()?.flat();
                let word = if tag == b'C' {
                    "_Complex"
                } else {
                    "_Imaginary"
                };
                let t = Ty::plain(self.build(format!("{inner} {word}"))?);
                self.add_sub(t.clone())?;
                Some(t)
            }
            b'A' => {
                self.pos += 1;
                let t = self.array_type()?;
                self.add_sub(t.clone())?;
                Some(t)
            }
            b'M' => {
                self.pos += 1;
                let class = self.ty()?.flat();
                let member = self.ty()?;
                let t = if member.is_simple() {
                    Ty::plain(format!("{} {}::*", member.left, class))
                } else {
                    Ty {
                        left: format!("{}({}::*", member.left, class),
                        right: format!("){}", member.right),
                    }
                };
                let t = self.build_ty(t)?;
                self.add_sub(t.clone())?;
                Some(t)
            }
            b'F' => {
                let t = self.function_type()?;
                self.add_sub(t.clone())?;
                Some(t)
            }
            b'T' => {
                let mut text = self.template_param()?;
                let t = Ty::plain(text.clone());
                self.add_sub(t.clone())?;
                // `<template-template-param> <template-args>`.
                if self.maybe_template_args(&mut text, true)? {
                    let t = Ty::plain(text);
                    self.add_sub(t.clone())?;
                    return Some(t);
                }
                Some(t)
            }
            b'u' => {
                // A vendor extended type is a builtin, so it is not a
                // substitution candidate.
                self.pos += 1;
                Some(Ty::plain(self.source_name()?))
            }
            b'U' => {
                // `U <source-name> <type>`: a vendor extended qualifier.
                self.pos += 1;
                let name = self.source_name()?;
                if self.peek() == Some(b'I') {
                    return None; // qualifier template arguments: out of scope
                }
                let inner = self.ty()?.flat();
                let t = Ty::plain(self.build(format!("{inner} {name}"))?);
                self.add_sub(t.clone())?;
                Some(t)
            }
            b'D' => {
                let kind = self.peek_at(1)?;
                if kind == b'p' {
                    // `Dp <type>`: a pack expansion.
                    self.pos += 2;
                    let inner = self.ty()?;
                    let t = if inner.is_simple() {
                        Ty::plain(format!("{}...", inner.left))
                    } else {
                        Ty {
                            left: inner.left.clone(),
                            right: format!("{}...", inner.right),
                        }
                    };
                    let t = self.build_ty(t)?;
                    self.add_sub(t.clone())?;
                    return Some(t);
                }
                // `Dt`/`DT` are decltype and stay out of scope.
                let name = d_builtin_type(kind)?;
                self.pos += 2;
                Some(Ty::plain(name))
            }
            // `<class-enum-type> ::= <name>`, including substitutions.
            b'N' | b'S' | b'Z' => self.named_type(),
            _ if tag.is_ascii_digit() => self.named_type(),
            _ => {
                let name = builtin_type(tag)?;
                self.pos += 1;
                Some(Ty::plain(name))
            }
        }
    }

    /// A `<name>` used as a type. The finished name is a substitution
    /// candidate unless it was written as a bare substitution already.
    fn named_type(&mut self) -> Option<Ty> {
        let name = self.name()?;
        let t = Ty::plain(name.text);
        if !name.plain_sub {
            self.add_sub(t.clone())?;
        }
        Some(t)
    }

    /// `<array-type> ::= A <dimension number> _ <element type>`.
    /// A dimension given as an expression is out of scope.
    fn array_type(&mut self) -> Option<Ty> {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
        let dim = std::str::from_utf8(self.input.get(start..self.pos)?).ok()?;
        if !self.eat(b'_') {
            return None;
        }
        let dim = dim.to_string();
        let inner = self.ty()?;
        // The outer dimension prints leftmost: `int [2][3]`.
        let t = if inner.is_simple() {
            Ty {
                left: format!("{} ", inner.left),
                right: format!("[{dim}]"),
            }
        } else {
            Ty {
                left: inner.left.clone(),
                right: format!("[{dim}]{}", inner.right),
            }
        };
        self.build_ty(t)
    }

    /// `<function-type> ::= F [Y] <bare-function-type> [<ref-qualifier>] E`.
    ///
    /// The first type is the return type; the rest are parameters. A `R`/`O`
    /// immediately before the closing `E` is a ref-qualifier, not a type.
    fn function_type(&mut self) -> Option<Ty> {
        if !self.eat(b'F') {
            return None;
        }
        self.eat(b'Y'); // extern "C"
        let ret = self.ty()?;

        let mut params: Vec<String> = Vec::new();
        let mut ref_q = "";
        loop {
            self.spend(1)?;
            let c = self.peek()?;
            if c == b'E' {
                self.pos += 1;
                break;
            }
            if matches!(c, b'R' | b'O') && self.peek_at(1) == Some(b'E') {
                ref_q = if c == b'R' { " &" } else { " &&" };
                self.pos += 1;
                continue;
            }
            params.push(self.ty()?.flat());
            if params.len() > MAX_PARAMS {
                return None;
            }
        }
        if params.is_empty() {
            return None;
        }
        let plist = if params.len() == 1 && params[0] == "void" {
            format!("(){ref_q}")
        } else {
            format!("({}){}", params.join(", "), ref_q)
        };
        let t = if ret.is_simple() {
            Ty {
                left: format!("{} ", ret.left),
                right: plist,
            }
        } else {
            Ty {
                left: ret.left.clone(),
                right: format!("{}{}", ret.right, plist),
            }
        };
        self.build_ty(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Names and functions ----------------------------------------------

    #[test]
    fn a_mangled_symbol_name_is_demangled() {
        assert_eq!(demangle("_ZN3foo3barEi"), "foo::bar(int)");
    }

    #[test]
    fn macho_leading_underscore_is_stripped() {
        assert_eq!(demangle("__ZN3foo3barEi"), "foo::bar(int)");
    }

    #[test]
    fn unscoped_name_with_builtin_parameters() {
        assert_eq!(demangle("_Z3fooiPKc"), "foo(int, char const*)");
        assert_eq!(demangle("_Z3barv"), "bar()");
        assert_eq!(demangle("_Z3foobcahstijlmxynofdez"), {
            "foo(bool, char, signed char, unsigned char, short, unsigned short, int, \
             unsigned int, long, unsigned long, long long, unsigned long long, __int128, \
             unsigned __int128, float, double, long double, ...)"
        });
    }

    #[test]
    fn nested_data_name_has_no_parameter_list() {
        assert_eq!(demangle("_ZN3foo3barE"), "foo::bar");
    }

    #[test]
    fn internal_linkage_marker_is_dropped() {
        assert_eq!(demangle("_ZN3fooL3barEv"), "foo::bar()");
    }

    #[test]
    fn abi_tag_is_parsed_and_dropped() {
        assert_eq!(demangle("_ZN3fooB5cxx113barEv"), "foo::bar()");
    }

    #[test]
    fn constructors_and_destructors_take_the_class_name() {
        assert_eq!(demangle("_ZN3FooC1Ev"), "Foo::Foo()");
        assert_eq!(demangle("_ZN3FooC2Ei"), "Foo::Foo(int)");
        assert_eq!(demangle("_ZN3FooD1Ev"), "Foo::~Foo()");
        assert_eq!(demangle("_ZN3FooD0Ev"), "Foo::~Foo()");
        // A template class's constructor drops the argument list.
        assert_eq!(demangle("_ZN3FooIiEC1Ev"), "Foo<int>::Foo()");
    }

    #[test]
    fn cv_and_ref_qualifiers_trail_the_parameter_list() {
        assert_eq!(demangle("_ZNK3Foo3getEv"), "Foo::get() const");
        assert_eq!(demangle("_ZNV3Foo3getEv"), "Foo::get() volatile");
        assert_eq!(demangle("_ZNVK3Foo3getEv"), "Foo::get() const volatile");
        assert_eq!(demangle("_ZNO3Foo3getEv"), "Foo::get() &&");
        assert_eq!(demangle("_ZNKR3Foo3getEv"), "Foo::get() const &");
    }

    // -- Substitutions ----------------------------------------------------

    #[test]
    fn substitutions_replay_earlier_components() {
        // S_ is `Hello`, S0_ is `World`, in order of first appearance.
        assert_eq!(
            demangle("_Z3foo5Hello5WorldS0_S_"),
            "foo(Hello, World, World, Hello)"
        );
    }

    #[test]
    fn nested_prefixes_are_substitution_candidates() {
        // S_ = N, S0_ = N::T, S1_ = N::T<int, int>.
        assert_eq!(
            demangle("_ZN1N1TIiiE2mfES0_IddE"),
            "N::T<int, int>::mf(N::T<double, double>)"
        );
    }

    #[test]
    fn std_abbreviations_expand() {
        assert_eq!(demangle("_ZSt5state"), "std::state");
        assert_eq!(demangle("_Z1fSs"), "f(std::string)");
        assert_eq!(demangle("_Z1fRSi"), "f(std::istream&)");
        assert_eq!(demangle("_Z1fRSo"), "f(std::ostream&)");
        assert_eq!(demangle("_Z1fRSd"), "f(std::iostream&)");
        assert_eq!(demangle("_Z1fSaIcE"), "f(std::allocator<char>)");
        assert_eq!(demangle("_Z1fSbIcE"), "f(std::basic_string<char>)");
        assert_eq!(demangle("_ZNSt3_In4wardE"), "std::_In::ward");
    }

    #[test]
    fn abbreviation_and_substitution_together() {
        assert_eq!(
            demangle("_ZNSt6vectorIiSaIiEE9push_backERKi"),
            "std::vector<int, std::allocator<int>>::push_back(int const&)"
        );
    }

    #[test]
    fn base_36_sequence_ids_are_accepted() {
        // Twelve registered types; `S_` is entry 0 and `SA_` is entry 11.
        let sym = "_Z3foo1a1b1c1d1e1f1g1h1i1j1k1lSA_S_";
        assert_eq!(
            demangle(sym),
            "foo(a, b, c, d, e, f, g, h, i, j, k, l, l, a)"
        );
    }

    // -- Types ------------------------------------------------------------

    #[test]
    fn indirection_and_qualifiers_print_east() {
        assert_eq!(demangle("_Z3fooPi"), "foo(int*)");
        assert_eq!(demangle("_Z3fooRi"), "foo(int&)");
        assert_eq!(demangle("_Z3fooOi"), "foo(int&&)");
        assert_eq!(demangle("_Z3fooPKPi"), "foo(int* const*)");
        assert_eq!(demangle("_Z3fooVKi"), "foo(int const volatile)");
    }

    #[test]
    fn function_pointer_and_array_types_use_declarator_syntax() {
        assert_eq!(demangle("_Z3fooPFvvE"), "foo(void (*)())");
        assert_eq!(demangle("_Z3fooPFiiE"), "foo(int (*)(int))");
        assert_eq!(demangle("_Z3fooA3_i"), "foo(int [3])");
        assert_eq!(demangle("_Z3fooPA3_i"), "foo(int (*)[3])");
        assert_eq!(demangle("_Z3fooPA2_A3_i"), "foo(int (*)[2][3])");
    }

    #[test]
    fn pointer_to_member_types() {
        assert_eq!(demangle("_Z3fooM1Ci"), "foo(int C::*)");
        assert_eq!(demangle("_Z3fooM1CFvvE"), "foo(void (C::*)())");
        assert_eq!(demangle("_Z3fooM1CKFvvE"), "foo(void (C::*)() const)");
    }

    #[test]
    fn d_prefixed_builtin_types() {
        assert_eq!(demangle("_Z3fooDn"), "foo(std::nullptr_t)");
        assert_eq!(demangle("_Z3fooDi"), "foo(char32_t)");
        assert_eq!(demangle("_Z3fooDs"), "foo(char16_t)");
        assert_eq!(demangle("_Z3fooDu"), "foo(char8_t)");
        assert_eq!(demangle("_Z3fooDa"), "foo(auto)");
        assert_eq!(demangle("_Z3fooDc"), "foo(decltype(auto))");
    }

    #[test]
    fn vendor_extended_type() {
        assert_eq!(demangle("_Z3foou6__bf16"), "foo(__bf16)");
    }

    #[test]
    fn pack_expansion_prints_ellipsis() {
        assert_eq!(demangle("_Z3fooIJiEEvDpT_"), "void foo<int>(int...)");
    }

    // -- Templates --------------------------------------------------------

    #[test]
    fn template_function_prints_its_return_type() {
        assert_eq!(demangle("_Z3fooIiEvT_"), "void foo<int>(int)");
        assert_eq!(demangle("_Z3fooIiEiT_"), "int foo<int>(int)");
    }

    #[test]
    fn template_method_on_a_nested_name() {
        assert_eq!(demangle("_ZN3foo3barIiEEvv"), "void foo::bar<int>()");
    }

    #[test]
    fn literal_template_arguments() {
        assert_eq!(demangle("_Z3fooILb1EEvv"), "void foo<true>()");
        assert_eq!(demangle("_Z3fooILb0EEvv"), "void foo<false>()");
        assert_eq!(demangle("_Z3fooILi42EEvv"), "void foo<42>()");
        assert_eq!(demangle("_Z3fooILin7EEvv"), "void foo<-7>()");
        assert_eq!(demangle("_Z3fooILj3EEvv"), "void foo<3u>()");
    }

    #[test]
    fn unresolvable_template_parameter_falls_back_to_a_placeholder() {
        // No enclosing argument list, so `T_` prints as the placeholder.
        assert_eq!(demangle("_Z3fooT_"), "foo(T)");
        assert_eq!(demangle("_Z3fooT0_"), "foo(T1)");
    }

    // -- Operators --------------------------------------------------------

    #[test]
    fn operator_names_render_as_declarations() {
        assert_eq!(demangle("_ZN3FooplERKS_"), "Foo::operator+(Foo const&)");
        assert_eq!(demangle("_ZN3FoomiEi"), "Foo::operator-(int)");
        assert_eq!(demangle("_ZN3FooaSERKS_"), "Foo::operator=(Foo const&)");
        assert_eq!(demangle("_ZN3FooixEi"), "Foo::operator[](int)");
        assert_eq!(demangle("_ZN3FooclEv"), "Foo::operator()()");
        assert_eq!(demangle("_ZN3FooptEv"), "Foo::operator->()");
        assert_eq!(demangle("_ZN3FooeqERKS_"), "Foo::operator==(Foo const&)");
        assert_eq!(demangle("_ZN3FoolsEi"), "Foo::operator<<(int)");
        assert_eq!(demangle("_ZN3FoorSEi"), "Foo::operator>>=(int)");
        assert_eq!(demangle("_ZN3FooppEv"), "Foo::operator++()");
        assert_eq!(demangle("_ZN3FoocoEv"), "Foo::operator~()");
        assert_eq!(demangle("_ZN3FoontEv"), "Foo::operator!()");
        assert_eq!(demangle("_ZN3FoocmEi"), "Foo::operator,(int)");
        assert_eq!(demangle("_ZN3FoopmEi"), "Foo::operator->*(int)");
        assert_eq!(demangle("_ZN3FooquEi"), "Foo::operator?:(int)");
    }

    #[test]
    fn global_allocation_operators() {
        assert_eq!(demangle("_Znwm"), "operator new(unsigned long)");
        assert_eq!(demangle("_Znam"), "operator new[](unsigned long)");
        assert_eq!(demangle("_ZdlPv"), "operator delete(void*)");
        assert_eq!(demangle("_ZdaPv"), "operator delete[](void*)");
    }

    #[test]
    fn conversion_operator_names_its_target_type() {
        assert_eq!(demangle("_ZN3FoocviEv"), "Foo::operator int()");
        assert_eq!(
            demangle("_ZNK3FoocvPKcEv"),
            "Foo::operator char const*() const"
        );
    }

    // -- Special names ----------------------------------------------------

    #[test]
    fn special_names_are_described_in_words() {
        assert_eq!(demangle("_ZTV3Foo"), "vtable for Foo");
        assert_eq!(demangle("_ZTI3Foo"), "typeinfo for Foo");
        assert_eq!(demangle("_ZTS3Foo"), "typeinfo name for Foo");
        assert_eq!(demangle("_ZTT3Foo"), "VTT for Foo");
        assert_eq!(demangle("_ZGVN3foo3barE"), "guard variable for foo::bar");
    }

    #[test]
    fn thunks_wrap_the_encoding_they_forward_to() {
        assert_eq!(
            demangle("_ZThn8_N3Foo3barEv"),
            "non-virtual thunk to Foo::bar()"
        );
        assert_eq!(
            demangle("_ZTv0_n24_N3FooD1Ev"),
            "virtual thunk to Foo::~Foo()"
        );
    }

    #[test]
    fn clone_suffix_is_preserved() {
        assert_eq!(demangle("_ZN3foo3barEv.cold"), "foo::bar() [clone .cold]");
    }

    // -- Rust legacy overlap ----------------------------------------------

    #[test]
    fn rust_legacy_symbols_are_valid_itanium_too() {
        assert_eq!(
            demangle("_ZN4core3fmt9Formatter3pad17h0123456789abcdefE"),
            "core::fmt::Formatter::pad::h0123456789abcdef"
        );
    }

    // -- Passthrough ------------------------------------------------------

    #[test]
    fn passthrough_non_mangled() {
        for s in [
            "main",
            "_start",
            "sub_401000",
            "",
            "printf",
            "_RNvC7mycrate3foo",
        ] {
            assert_eq!(demangle(s), s, "should pass {s:?} through unchanged");
        }
    }

    #[test]
    fn is_mangled_prefix_check() {
        assert!(is_mangled("_Z3foov"));
        assert!(is_mangled("__Z3foov"));
        assert!(is_mangled("_ZN3foo3barEi"));
        assert!(!is_mangled("_Z"));
        assert!(!is_mangled("main"));
        assert!(!is_mangled("_RNvC7mycrate3foo"));
        assert!(!is_mangled(""));
    }

    // -- No panic / no hang / refuse-don't-guess ---------------------------

    #[test]
    fn out_of_scope_productions_are_refused() {
        // Local names, lambdas, unnamed types, decltype, and expressions.
        for sym in [
            "_ZZ4mainE3foo",
            "_ZN3fooUlvE_clEv",
            "_ZN3fooUt_E",
            "_Z3fooDTadL_Z1fvEE",
            "_Z3fooIXadL_Z1fvEEEvv",
        ] {
            assert_eq!(demangle(sym), sym, "{sym:?} should not be guessed at");
        }
    }

    #[test]
    fn malformed_inputs_pass_through() {
        for sym in [
            "_Z",
            "_ZN",
            "_ZN3foo",
            "_ZN99999999999foo",
            "_ZN0fooE",
            "_Z3fooS_",     // substitution with an empty table
            "_Z3fooS9_",    // seq-id past the end of the table
            "_Z3fooS_____", // nonsense seq-id
            "_ZN3fooEE",    // trailing junk
            "_Z3fooIiEv",   // template function with no parameter list
            "_ZTV",         // truncated special name
            "_ZThn8_",      // thunk with no target encoding
            "_ZC1Ev",       // constructor with no enclosing class
        ] {
            let _ = demangle(sym);
        }
        assert_eq!(demangle("_ZN99999999999foo"), "_ZN99999999999foo");
        assert_eq!(demangle("_Z3fooS_"), "_Z3fooS_");
        assert_eq!(demangle(""), "");
    }

    #[test]
    fn deep_nesting_hits_the_depth_cap_and_bails() {
        let sym = format!("_Z3foo{}i", "P".repeat(200));
        assert_eq!(demangle(&sym), sym);
        let nested = format!("_Z3foo{}{}", "A1_".repeat(100), "i");
        assert_eq!(demangle(&nested), nested);
    }

    #[test]
    fn oversized_input_is_refused_without_work() {
        let sym = format!("_ZN3foo{}E", "3bar".repeat(2000));
        assert!(sym.len() > MAX_INPUT);
        assert_eq!(demangle(&sym), sym);
    }

    #[test]
    fn truncated_prefixes_never_panic() {
        let samples = [
            "_ZNSt6vectorIiSaIiEE9push_backERKi",
            "_ZN1N1TIiiE2mfES0_IddE",
            "_ZTv0_n24_N3FooD1Ev",
            "_Z3fooPA2_A3_i",
            "_ZNK3FoocvPKcEv",
            "_Z3fooILi42EEvv",
            "_ZN4core3fmt9Formatter3pad17h0123456789abcdefE",
        ];
        for sym in samples {
            for i in 0..=sym.len() {
                // Every prefix must return *something* without panicking.
                let out = demangle(&sym[..i]);
                assert!(out.len() <= MAX_OUTPUT + 64);
            }
        }
    }

    #[test]
    fn self_referential_substitutions_terminate() {
        // A substitution can only name an *earlier* entry, so no input can
        // build a cycle; these must all terminate within the caps.
        for sym in ["_ZS_", "_ZNS_E", "_Z1fS_S_S_", "_ZNS0_S0_E", "_Z1fIS_ES_S_"] {
            let out = demangle(sym);
            assert!(out.len() <= MAX_OUTPUT + 64);
        }
    }

    #[test]
    fn deterministic_fuzz_sweep_never_panics_or_hangs() {
        // A fixed-seed LCG drives a few thousand `_Z` + pseudo-random-ASCII
        // inputs through the parser. The assertion is weak by design — the
        // real property under test is that every call *returns* within the
        // caps (no panic, no infinite loop, bounded memory).
        const ALPHABET: &[u8] =
            b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_.NSITEKPROMFAUDCGLBJXZ";
        let mut state: u64 = 0x0bad_c0de_dead_1234;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };

        for _ in 0..6000 {
            let len = (next() % 56) as usize;
            let mut body = String::with_capacity(len + 2);
            body.push_str("_Z");
            for _ in 0..len {
                let idx = (next() as usize) % ALPHABET.len();
                body.push(ALPHABET[idx] as char);
            }
            let out = demangle(&body);
            assert!(
                out.len() <= MAX_OUTPUT + 64,
                "output exceeded cap for {body:?}"
            );
        }
    }

    #[test]
    fn real_libstdcxx_symbols() {
        // Whole-symbol regressions taken from a GCC-built binary. The
        // `map::find` case pins the substitution indices exactly: `S5_` is
        // the sixth entry, `std::allocator<std::pair<int const, int>>`.
        assert_eq!(demangle("_ZSt4cout"), "std::cout");
        assert_eq!(
            demangle("_ZNKSt3mapIiiSt4lessIiESaISt4pairIKiiEEE4findERS5_"),
            "std::map<int, int, std::less<int>, \
             std::allocator<std::pair<int const, int>>>::find(\
             std::allocator<std::pair<int const, int>>&) const"
        );
        assert_eq!(
            demangle("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc"),
            "std::__cxx11::basic_string<char, std::char_traits<char>, \
             std::allocator<char>>::append(char const*)"
        );
        assert_eq!(
            demangle("_ZNSt8ios_base4InitC1Ev"),
            "std::ios_base::Init::Init()"
        );
        assert_eq!(
            demangle("_ZTVN10__cxxabiv117__class_type_infoE"),
            "vtable for __cxxabiv1::__class_type_info"
        );
        assert_eq!(
            demangle("_ZN5boost6system15system_categoryEv"),
            "boost::system::system_category()"
        );
        // `St` directly in front of an operator name, a template argument
        // list on the operator, and a `T_` resolved from it.
        assert_eq!(
            demangle("_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc"),
            "std::basic_ostream<char, std::char_traits<char>>& \
             std::operator<<<std::char_traits<char>>(\
             std::basic_ostream<char, std::char_traits<char>>&, char const*)"
        );
    }

    #[test]
    fn already_demangled_text_is_idempotent() {
        let text = "std::vector<int, std::allocator<int>>::push_back(int const&)";
        assert_eq!(demangle(text), text);
        assert_eq!(demangle(&demangle(text)), text);
    }
}
