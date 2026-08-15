//! Rust symbol demangling — turning a compiler-mangled symbol back into
//! the readable path a human wrote.
//!
//! The Rust compiler has emitted two mangling schemes over its lifetime,
//! and a real binary can carry either (or both), so this module handles
//! both:
//!
//! - **Legacy** (`_ZN..E`, pre-2018) — an Itanium-C++-style nested name.
//!   Each component is length-prefixed decimal, the components join with
//!   `::`, and a trailing `17h<16 hex>` disambiguator hash is stripped.
//!   Characters illegal in a linker symbol are escaped as `$..$` sequences
//!   (`$LT$`→`<`, `$u20$`→space, the general `$u{hex}$`→a Unicode scalar),
//!   `..` stands for `::`, and a leading `_` guarding a digit is dropped.
//! - **v0** (`_R..`, current) — the RFC 2603 scheme. A compact grammar of
//!   length-prefixed identifiers, nested paths, impl blocks, generic
//!   argument lists, a single-letter basic-type table, type constructors,
//!   const generics, lifetimes, and *backreferences* that point at an
//!   earlier byte to reuse a previously-encoded fragment. Identifiers may
//!   be punycode-encoded to carry non-ASCII. Parsed here by recursive
//!   descent over the byte stream.
//!
//! **Best-effort, no-panic contract.** [`demangle`] never fails and never
//! panics: an input that is not a recognized Rust mangling — or one that is
//! truncated, malformed, or adversarial — is returned unchanged. The v0
//! parser treats every count in the stream as attacker-controlled, so it
//! bounds recursion depth, total output size, and total parse work, and it
//! only follows a backreference that points strictly *backward*. A cyclic
//! or fan-out-exponential symbol therefore terminates within the caps and
//! yields the original string rather than hanging or exhausting memory.
//!
//! This is a clean-room implementation written from the public v0 grammar,
//! in the spirit of `c++filt`/`rustfilt`, not a port of any crate.

/// Maximum recursion depth across path/type/const productions. Real
/// symbols nest a few dozen levels at most; the cap only fires on
/// pathological input, where it forces a clean bail-out.
const MAX_DEPTH: u32 = 256;

/// Maximum rendered length. A valid Rust symbol demangles to far less; the
/// cap bounds memory against an input engineered to expand without limit.
const MAX_OUTPUT: usize = 1 << 16;

/// Total parse-step budget. Backreferences let a linear input describe an
/// exponentially large tree; charging one unit per production and per
/// emitted fragment makes total work linear in the budget regardless of
/// how the backref DAG is shaped.
const MAX_BUDGET: u32 = 1_000_000;

/// Cap on the lifetimes a single `for<...>` binder may introduce, so a
/// huge binder count cannot drive a long print loop.
const MAX_BINDER_LTS: u64 = 64;

/// Cap on legacy component count, guarding the element scan.
const MAX_LEGACY_ELEMENTS: usize = 1 << 16;

/// True if `symbol` carries a prefix this module recognizes.
///
/// A cheap syntactic check only — a `true` result does not guarantee the
/// body parses. Used to decide quickly whether demangling is worth trying.
pub fn is_mangled(symbol: &str) -> bool {
    strip_v0_prefix(symbol).is_some() || strip_legacy_prefix(symbol).is_some()
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

/// Demangle `symbol`, returning `None` when it is not a recognized Rust
/// mangling (or is malformed past repair).
///
/// v0 is tried first because its prefix (`_R`) cannot collide with the
/// legacy one (`_ZN`); a successful parse of either scheme wins.
pub fn try_demangle(symbol: &str) -> Option<String> {
    if let Some(inner) = strip_v0_prefix(symbol)
        && let Some(text) = demangle_v0(inner)
    {
        return Some(text);
    }
    if strip_legacy_prefix(symbol).is_some()
        && let Some(text) = demangle_legacy(symbol)
    {
        return Some(text);
    }
    None
}

/// Strip the v0 prefix, accounting for the extra leading underscore a
/// Mach-O assembler prepends (`__R`).
fn strip_v0_prefix(s: &str) -> Option<&str> {
    s.strip_prefix("_R").or_else(|| s.strip_prefix("__R"))
}

/// Strip the legacy prefix. Accepts the bare (`ZN`), ELF (`_ZN`), and
/// Mach-O (`__ZN`) spellings.
fn strip_legacy_prefix(s: &str) -> Option<&str> {
    s.strip_prefix("_ZN")
        .or_else(|| s.strip_prefix("__ZN"))
        .or_else(|| s.strip_prefix("ZN"))
}

// ---------------------------------------------------------------------------
// Legacy scheme
// ---------------------------------------------------------------------------

/// Demangle a legacy `_ZN..E` symbol, or `None` if it does not parse.
fn demangle_legacy(symbol: &str) -> Option<String> {
    let inner = strip_legacy_prefix(symbol)?;
    if !inner.is_ascii() {
        return None;
    }
    let bytes = inner.as_bytes();

    // Split the length-prefixed components up to the terminating `E`.
    let mut elements: Vec<&str> = Vec::new();
    let mut i = 0usize;
    loop {
        let b = *bytes.get(i)?;
        if b == b'E' {
            break;
        }
        if !b.is_ascii_digit() {
            return None;
        }
        let mut len = 0usize;
        while let Some(&d) = bytes.get(i) {
            if !d.is_ascii_digit() {
                break;
            }
            len = len.checked_mul(10)?.checked_add((d - b'0') as usize)?;
            i += 1;
        }
        let start = i;
        let end = start.checked_add(len)?;
        if end > bytes.len() {
            return None;
        }
        elements.push(&inner[start..end]);
        i = end;
        if elements.len() > MAX_LEGACY_ELEMENTS {
            return None;
        }
    }
    if elements.is_empty() {
        return None;
    }
    // The `E` must end the symbol: legacy manglings put nothing after it,
    // so trailing bytes mean this is some other Itanium mangling (a C++
    // function's parameter list starts right after its name's `E`). The
    // one exception is a linker/LLVM dot-suffix (`.llvm.…`, `.cold`),
    // which no Itanium encoding can begin with.
    match bytes.get(i + 1) {
        None | Some(b'.') => {}
        Some(_) => return None,
    }

    // Drop the trailing compiler hash (`h` followed by hex), but never the
    // sole component of a one-element symbol.
    if elements.len() > 1
        && let Some(last) = elements.last()
        && is_rust_hash(last)
    {
        elements.pop();
    }

    let mut out = String::new();
    for (idx, element) in elements.iter().enumerate() {
        if idx != 0 {
            out.push_str("::");
        }
        // A component escaped to avoid a leading digit carries a guard `_`.
        let mut element = *element;
        if let Some(rest) = element.strip_prefix('_')
            && rest.starts_with(|c: char| c.is_ascii_digit())
        {
            element = rest;
        }
        legacy_decode(element, &mut out);
    }
    Some(out)
}

/// True if `s` is a legacy disambiguator hash: `h` then all hex digits.
fn is_rust_hash(s: &str) -> bool {
    match s.strip_prefix('h') {
        Some(rest) => !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_hexdigit()),
        None => false,
    }
}

/// Decode one legacy component's `$..$` escapes and `.`/`..` runs into
/// `out`.
fn legacy_decode(component: &str, out: &mut String) {
    let mut rest = component;
    while !rest.is_empty() {
        if let Some(after_dollar) = rest.strip_prefix('$') {
            match after_dollar.find('$') {
                Some(end) => {
                    let escape = &after_dollar[..end];
                    match legacy_escape(escape) {
                        Some(ch) => out.push(ch),
                        None => {
                            // Not a known escape — emit it verbatim so the
                            // text is never silently lost.
                            out.push('$');
                            out.push_str(escape);
                            out.push('$');
                        }
                    }
                    rest = &after_dollar[end + 1..];
                }
                None => {
                    // Unterminated `$` run: nothing more to decode.
                    out.push_str(rest);
                    break;
                }
            }
        } else {
            match rest.find(['$', '.']) {
                None => {
                    out.push_str(rest);
                    break;
                }
                Some(pos) => {
                    out.push_str(&rest[..pos]);
                    rest = &rest[pos..];
                    if let Some(after) = rest.strip_prefix("..") {
                        out.push_str("::");
                        rest = after;
                    } else if let Some(after) = rest.strip_prefix('.') {
                        out.push('.');
                        rest = after;
                    }
                    // A `$` at `pos` is handled by the next loop turn.
                }
            }
        }
    }
}

/// Map a legacy `$..$` escape body to its character, or `None` if unknown.
///
/// The named forms are shorthands; anything of the form `u<hex>` is the
/// general Unicode-scalar escape (which also covers `$u20$`, `$u7e$`, …).
fn legacy_escape(escape: &str) -> Option<char> {
    Some(match escape {
        "SP" => '@',
        "BP" => '*',
        "RF" => '&',
        "LT" => '<',
        "GT" => '>',
        "LP" => '(',
        "RP" => ')',
        "C" => ',',
        _ => {
            let hex = escape.strip_prefix('u')?;
            if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return None;
            }
            let cp = u32::from_str_radix(hex, 16).ok()?;
            char::from_u32(cp)?
        }
    })
}

// ---------------------------------------------------------------------------
// v0 scheme
// ---------------------------------------------------------------------------

/// Demangle a v0 body (the bytes after the `_R`/`__R` prefix).
fn demangle_v0(inner: &str) -> Option<String> {
    // A trailing `.`-suffix (`.llvm.NNNN`, `.42`) is not part of the v0
    // grammar, whose body contains no `.`, so cut it off first.
    let inner = match inner.find('.') {
        Some(i) => &inner[..i],
        None => inner,
    };

    // A leading decimal is the encoding-version field; only version 0
    // (encoded as *absent*) exists, and a path never starts with a digit,
    // so any digit here is an unsupported future version.
    if inner.as_bytes().first().is_some_and(u8::is_ascii_digit) {
        return None;
    }

    let mut parser = V0 {
        input: inner.as_bytes(),
        pos: 0,
        out: String::new(),
        depth: 0,
        budget: MAX_BUDGET,
        bound_lifetimes: 0,
        silent: false,
    };
    parser.print_path(true)?;
    Some(parser.out)
}

/// Recursive-descent parser and printer for one v0 symbol.
///
/// Holds the input, a byte cursor, the growing output, and the three
/// budgets that enforce the no-panic/no-hang contract. `silent` suppresses
/// output while a subtree is parsed purely to advance the cursor (used for
/// the impl-path of an impl block, which is consumed but not printed).
struct V0<'a> {
    input: &'a [u8],
    pos: usize,
    out: String,
    depth: u32,
    budget: u32,
    /// Count of in-scope `for<...>`-bound lifetimes, for De Bruijn naming.
    bound_lifetimes: u64,
    silent: bool,
}

impl V0<'_> {
    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn take_byte(&mut self) -> Option<u8> {
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

    /// Charge one unit of the parse budget, failing when it is spent.
    fn spend(&mut self) -> Option<()> {
        self.budget = self.budget.checked_sub(1)?;
        Some(())
    }

    /// Append `s`, unless suppressed, respecting the output cap.
    fn emit(&mut self, s: &str) -> Option<()> {
        self.spend()?;
        if !self.silent {
            if self.out.len().checked_add(s.len())? > MAX_OUTPUT {
                return None;
            }
            self.out.push_str(s);
        }
        Some(())
    }

    /// Append a single character, unless suppressed, respecting the cap.
    fn emit_char(&mut self, c: char) -> Option<()> {
        self.spend()?;
        if !self.silent {
            if self.out.len().checked_add(c.len_utf8())? > MAX_OUTPUT {
                return None;
            }
            self.out.push(c);
        }
        Some(())
    }

    /// A `<decimal-number>`: `0`, or a non-zero run of digits.
    fn decimal(&mut self) -> Option<usize> {
        let first = self.peek()?;
        if first == b'0' {
            self.pos += 1;
            return Some(0);
        }
        if !first.is_ascii_digit() {
            return None;
        }
        let mut n = 0usize;
        while let Some(b) = self.peek() {
            if !b.is_ascii_digit() {
                break;
            }
            n = n.checked_mul(10)?.checked_add((b - b'0') as usize)?;
            self.pos += 1;
        }
        Some(n)
    }

    /// A `<base-62-number>`: `_` is 0, otherwise base-62 digits then `_`
    /// encode the value plus one.
    fn base62(&mut self) -> Option<u64> {
        if self.eat(b'_') {
            return Some(0);
        }
        let mut num = 0u64;
        loop {
            let b = self.take_byte()?;
            if b == b'_' {
                return num.checked_add(1);
            }
            let digit = match b {
                b'0'..=b'9' => (b - b'0') as u64,
                b'a'..=b'z' => (b - b'a') as u64 + 10,
                b'A'..=b'Z' => (b - b'A') as u64 + 36,
                _ => return None,
            };
            num = num.checked_mul(62)?.checked_add(digit)?;
        }
    }

    /// An optional `s`-disambiguator; absent reads as 0. The stored value
    /// is offset by one so an explicit `s_` is 1, never colliding with the
    /// absent-0 case.
    fn disambiguator(&mut self) -> Option<u64> {
        if self.eat(b's') {
            self.base62()?.checked_add(1)
        } else {
            Some(0)
        }
    }

    /// An `<undisambiguated-identifier>`: optional punycode flag, a length,
    /// an optional `_` separator, then that many bytes.
    fn ident(&mut self) -> Option<String> {
        self.spend()?;
        let punycode = self.eat(b'u');
        let len = self.decimal()?;
        self.eat(b'_'); // optional separator guarding a leading digit/`_`
        let end = self.pos.checked_add(len)?;
        if end > self.input.len() {
            return None;
        }
        let raw = &self.input[self.pos..end];
        self.pos = end;
        let text = std::str::from_utf8(raw).ok()?;
        if punycode {
            decode_punycode(text)
        } else {
            Some(text.to_string())
        }
    }

    /// A backreference: read the target offset, validate it points strictly
    /// backward (the invariant that makes cycles impossible), then run
    /// `print` at that offset with the cursor restored afterward. `b_pos`
    /// is the offset of the `B` tag itself.
    fn follow_backref(&mut self, b_pos: usize, print: fn(&mut Self) -> Option<()>) -> Option<()> {
        let target = usize::try_from(self.base62()?).ok()?;
        if target >= b_pos {
            return None;
        }
        let saved = self.pos;
        self.pos = target;
        let r = print(self);
        self.pos = saved;
        r
    }

    /// Print a `<path>` in value (`true`) or type (`false`) context; the
    /// context selects turbofish (`::<...>`) versus bare (`<...>`) generic
    /// argument syntax.
    fn print_path(&mut self, in_value: bool) -> Option<()> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let r = self.print_path_body(in_value);
        self.depth -= 1;
        r
    }

    fn print_path_body(&mut self, in_value: bool) -> Option<()> {
        self.spend()?;
        let b_pos = self.pos;
        let tag = self.take_byte()?;
        match tag {
            b'C' => {
                let _dis = self.disambiguator()?;
                let name = self.ident()?;
                self.emit(&name)?;
            }
            b'N' => {
                let ns = self.take_byte()?;
                self.print_path(in_value)?;
                let dis = self.disambiguator()?;
                let name = self.ident()?;
                if ns.is_ascii_uppercase() {
                    // A compiler-internal name (closure, shim, …): braces.
                    self.emit("::{")?;
                    match ns {
                        b'C' => self.emit("closure")?,
                        b'S' => self.emit("shim")?,
                        _ => self.emit_char(ns as char)?,
                    }
                    if !name.is_empty() {
                        self.emit(":")?;
                        self.emit(&name)?;
                    }
                    self.emit("#")?;
                    self.emit(&dis.to_string())?;
                    self.emit("}")?;
                } else if !name.is_empty() {
                    self.emit("::")?;
                    self.emit(&name)?;
                }
            }
            b'M' => {
                // Inherent impl: `<Type>`. The impl-path (where the block is
                // written) is parsed but not printed.
                self.disambiguator()?;
                self.print_path_silently()?;
                self.emit("<")?;
                self.print_type()?;
                self.emit(">")?;
            }
            b'X' => {
                // Trait impl: `<Type as Trait>`.
                self.disambiguator()?;
                self.print_path_silently()?;
                self.emit("<")?;
                self.print_type()?;
                self.emit(" as ")?;
                self.print_path(false)?;
                self.emit(">")?;
            }
            b'Y' => {
                // Qualified `<Type as Trait>` with no impl-path.
                self.emit("<")?;
                self.print_type()?;
                self.emit(" as ")?;
                self.print_path(false)?;
                self.emit(">")?;
            }
            b'I' => {
                self.print_path(in_value)?;
                if in_value {
                    self.emit("::")?;
                }
                self.emit("<")?;
                let mut first = true;
                while self.peek()? != b'E' {
                    if !first {
                        self.emit(", ")?;
                    }
                    first = false;
                    self.print_generic_arg()?;
                }
                self.take_byte()?; // 'E'
                self.emit(">")?;
            }
            b'B' => {
                // Path backref: same shape as `follow_backref`, but the
                // continuation needs the `in_value` context, which a plain
                // `fn` pointer cannot carry.
                let target = usize::try_from(self.base62()?).ok()?;
                if target >= b_pos {
                    return None;
                }
                let saved = self.pos;
                self.pos = target;
                let r = self.print_path(in_value);
                self.pos = saved;
                r?;
            }
            _ => return None,
        }
        Some(())
    }

    /// Parse a path only to advance the cursor, printing nothing.
    fn print_path_silently(&mut self) -> Option<()> {
        let was_silent = self.silent;
        self.silent = true;
        let r = self.print_path(false);
        self.silent = was_silent;
        r
    }

    /// A `<generic-arg>`: a lifetime, a `K`-tagged const, or a type.
    fn print_generic_arg(&mut self) -> Option<()> {
        match self.peek()? {
            b'L' => {
                self.take_byte()?;
                let lt = self.base62()?;
                self.print_lifetime(lt)?;
            }
            b'K' => {
                self.take_byte()?;
                self.print_const()?;
            }
            _ => self.print_type()?,
        }
        Some(())
    }

    /// Print a lifetime by De Bruijn index: 0 is the anonymous `'_`, and a
    /// bound index maps to `'a`, `'b`, … by scope depth.
    fn print_lifetime(&mut self, lt: u64) -> Option<()> {
        self.emit("'")?;
        if lt == 0 {
            return self.emit("_");
        }
        match self.bound_lifetimes.checked_sub(lt) {
            Some(depth) => {
                self.emit_char((b'a' + (depth % 26) as u8) as char)?;
                if depth >= 26 {
                    self.emit(&(depth / 26).to_string())?;
                }
            }
            None => self.emit(&lt.to_string())?,
        }
        Some(())
    }

    fn print_type(&mut self) -> Option<()> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let r = self.print_type_body();
        self.depth -= 1;
        r
    }

    fn print_type_body(&mut self) -> Option<()> {
        self.spend()?;
        let tag = self.peek()?;
        if let Some(basic) = basic_type(tag) {
            self.take_byte()?;
            return self.emit(basic);
        }
        if matches!(tag, b'C' | b'M' | b'X' | b'Y' | b'N' | b'I') {
            // A named type is just a path.
            return self.print_path(false);
        }
        let b_pos = self.pos;
        let tag = self.take_byte()?;
        match tag {
            b'A' => {
                self.emit("[")?;
                self.print_type()?;
                self.emit("; ")?;
                self.print_const()?;
                self.emit("]")?;
            }
            b'S' => {
                self.emit("[")?;
                self.print_type()?;
                self.emit("]")?;
            }
            b'T' => {
                self.emit("(")?;
                let mut count = 0u32;
                while self.peek()? != b'E' {
                    if count > 0 {
                        self.emit(", ")?;
                    }
                    self.print_type()?;
                    count += 1;
                }
                self.take_byte()?; // 'E'
                if count == 1 {
                    // A one-tuple needs the trailing comma to stay a tuple.
                    self.emit(",")?;
                }
                self.emit(")")?;
            }
            b'R' | b'Q' => {
                self.emit("&")?;
                if self.peek() == Some(b'L') {
                    self.take_byte()?;
                    let lt = self.base62()?;
                    if lt != 0 {
                        self.print_lifetime(lt)?;
                        self.emit(" ")?;
                    }
                }
                if tag == b'Q' {
                    self.emit("mut ")?;
                }
                self.print_type()?;
            }
            b'P' => {
                self.emit("*const ")?;
                self.print_type()?;
            }
            b'O' => {
                self.emit("*mut ")?;
                self.print_type()?;
            }
            b'F' => self.print_fn_sig()?,
            b'D' => self.print_dyn()?,
            b'B' => self.follow_backref(b_pos, Self::print_type)?,
            _ => return None,
        }
        Some(())
    }

    /// A function-pointer signature:
    /// `[binder] [U] [K <abi>] {type} E <ret>`.
    fn print_fn_sig(&mut self) -> Option<()> {
        let restore = self.print_binder()?;
        if self.eat(b'U') {
            self.emit("unsafe ")?;
        }
        if self.eat(b'K') {
            self.emit("extern \"")?;
            if self.eat(b'C') {
                self.emit("C")?;
            } else {
                let abi = self.ident()?;
                if abi.is_empty() {
                    return None;
                }
                self.emit(&abi.replace('_', "-"))?;
            }
            self.emit("\" ")?;
        }
        self.emit("fn(")?;
        let mut count = 0u32;
        while self.peek()? != b'E' {
            if count > 0 {
                self.emit(", ")?;
            }
            self.print_type()?;
            count += 1;
        }
        self.take_byte()?; // 'E'
        self.emit(")")?;
        if self.peek()? == b'u' {
            // A unit return is left implicit, matching rendered Rust.
            self.take_byte()?;
        } else {
            self.emit(" -> ")?;
            self.print_type()?;
        }
        self.bound_lifetimes = restore;
        Some(())
    }

    /// A trait object: `dyn Trait + ... + 'lt`.
    fn print_dyn(&mut self) -> Option<()> {
        self.emit("dyn ")?;
        let restore = self.print_binder()?;
        let mut first = true;
        while self.peek()? != b'E' {
            if !first {
                self.emit(" + ")?;
            }
            first = false;
            self.print_dyn_trait()?;
        }
        self.take_byte()?; // 'E'
        self.bound_lifetimes = restore;
        if self.eat(b'L') {
            let lt = self.base62()?;
            if lt != 0 {
                self.emit(" + ")?;
                self.print_lifetime(lt)?;
            }
        }
        Some(())
    }

    /// One `dyn` bound: a trait path plus any `p`-tagged associated-type
    /// bindings, rendered as `Trait<Name = Type, ...>`.
    fn print_dyn_trait(&mut self) -> Option<()> {
        self.print_path(false)?;
        if self.peek() == Some(b'p') {
            self.emit("<")?;
            let mut first = true;
            while self.peek() == Some(b'p') {
                self.take_byte()?;
                if !first {
                    self.emit(", ")?;
                }
                first = false;
                let name = self.ident()?;
                self.emit(&name)?;
                self.emit(" = ")?;
                self.print_type()?;
            }
            self.emit(">")?;
        }
        Some(())
    }

    /// An optional `G`-binder; prints `for<'a, ...>` and returns the prior
    /// bound-lifetime count so the caller can restore scope.
    fn print_binder(&mut self) -> Option<u64> {
        let saved = self.bound_lifetimes;
        if self.eat(b'G') {
            let count = self.base62()?.checked_add(1)?.min(MAX_BINDER_LTS);
            self.emit("for<")?;
            for i in 0..count {
                if i > 0 {
                    self.emit(", ")?;
                }
                let depth = self.bound_lifetimes + i;
                self.emit("'")?;
                self.emit_char((b'a' + (depth % 26) as u8) as char)?;
                if depth >= 26 {
                    self.emit(&(depth / 26).to_string())?;
                }
            }
            self.bound_lifetimes = self.bound_lifetimes.checked_add(count)?;
            self.emit("> ")?;
        }
        Some(saved)
    }

    fn print_const(&mut self) -> Option<()> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        let r = self.print_const_body();
        self.depth -= 1;
        r
    }

    fn print_const_body(&mut self) -> Option<()> {
        self.spend()?;
        let b_pos = self.pos;
        let tag = self.take_byte()?;
        match tag {
            b'B' => self.follow_backref(b_pos, Self::print_const)?,
            b'p' => self.emit("_")?,
            b'b' => {
                let v = self.hex_nibbles()?;
                self.emit(if v == 0 { "false" } else { "true" })?;
            }
            b'c' => {
                let v = self.hex_nibbles()?;
                match u32::try_from(v).ok().and_then(char::from_u32) {
                    Some(ch) => {
                        self.emit("'")?;
                        self.emit_char(ch)?;
                        self.emit("'")?;
                    }
                    None => self.emit("'?'")?,
                }
            }
            // Signed integers carry an optional `n` sign before the digits.
            b'a' | b's' | b'l' | b'x' | b'n' | b'i' => {
                if self.eat(b'n') {
                    self.emit("-")?;
                }
                let v = self.hex_nibbles()?;
                self.emit(&v.to_string())?;
            }
            // Unsigned integers.
            b'h' | b't' | b'm' | b'y' | b'o' | b'j' => {
                let v = self.hex_nibbles()?;
                self.emit(&v.to_string())?;
            }
            _ => return None,
        }
        Some(())
    }

    /// Read a `_`-terminated run of lowercase-hex nibbles into a value; an
    /// empty run (a bare `_`) is zero.
    fn hex_nibbles(&mut self) -> Option<u64> {
        let mut val = 0u64;
        loop {
            let b = self.take_byte()?;
            if b == b'_' {
                return Some(val);
            }
            let digit = match b {
                b'0'..=b'9' => (b - b'0') as u64,
                b'a'..=b'f' => (b - b'a') as u64 + 10,
                _ => return None,
            };
            val = val.checked_mul(16)?.checked_add(digit)?;
        }
    }
}

/// The v0 single-letter basic-type table, or `None` if `tag` is not one.
fn basic_type(tag: u8) -> Option<&'static str> {
    Some(match tag {
        b'a' => "i8",
        b's' => "i16",
        b'l' => "i32",
        b'x' => "i64",
        b'n' => "i128",
        b'i' => "isize",
        b'h' => "u8",
        b't' => "u16",
        b'm' => "u32",
        b'y' => "u64",
        b'o' => "u128",
        b'j' => "usize",
        b'f' => "f32",
        b'd' => "f64",
        b'b' => "bool",
        b'c' => "char",
        b'e' => "str",
        b'u' => "()",
        b'z' => "!",
        b'p' => "_",
        b'v' => "...",
        _ => return None,
    })
}

/// Decode a v0 punycode identifier (RFC 3492 bootstring, with `_` as the
/// delimiter in place of `-`). Returns `None` on malformed input rather
/// than partial garbage.
fn decode_punycode(input: &str) -> Option<String> {
    // Bootstring parameters (RFC 3492, §5).
    const BASE: u32 = 36;
    const TMIN: u32 = 1;
    const TMAX: u32 = 26;
    const SKEW: u32 = 38;
    const DAMP: u32 = 700;
    const INITIAL_BIAS: u32 = 72;
    const INITIAL_N: u32 = 128;

    // Split off the basic (already-ASCII) code points at the last `_`.
    let (basic, deltas) = match input.rfind('_') {
        Some(i) => (&input[..i], &input[i + 1..]),
        None => ("", input),
    };

    let mut output: Vec<u32> = basic.chars().map(|c| c as u32).collect();

    let deltas = deltas.as_bytes();
    let mut n = INITIAL_N;
    let mut i = 0u32;
    let mut bias = INITIAL_BIAS;
    let mut in_pos = 0usize;

    while in_pos < deltas.len() {
        let oldi = i;
        let mut w = 1u32;
        let mut k = BASE;
        loop {
            let byte = *deltas.get(in_pos)?;
            in_pos += 1;
            let digit = match byte {
                b'a'..=b'z' => (byte - b'a') as u32,
                b'0'..=b'9' => (byte - b'0') as u32 + 26,
                _ => return None,
            };
            i = i.checked_add(digit.checked_mul(w)?)?;
            let t = if k <= bias {
                TMIN
            } else if k >= bias + TMAX {
                TMAX
            } else {
                k - bias
            };
            if digit < t {
                break;
            }
            w = w.checked_mul(BASE - t)?;
            k += BASE;
        }

        let out_len = output.len() as u32 + 1;
        bias = adapt(i - oldi, out_len, oldi == 0, BASE, TMIN, TMAX, SKEW, DAMP);
        n = n.checked_add(i / out_len)?;
        i %= out_len;
        // Bound the reconstructed identifier so a crafted delta stream
        // cannot grow it without limit.
        if output.len() >= MAX_OUTPUT {
            return None;
        }
        output.insert(i as usize, n);
        i += 1;
    }

    let mut s = String::with_capacity(output.len());
    for cp in output {
        s.push(char::from_u32(cp)?);
    }
    Some(s)
}

/// The bootstring bias adaptation (RFC 3492, §6.1).
#[allow(clippy::too_many_arguments)]
fn adapt(
    delta: u32,
    numpoints: u32,
    firsttime: bool,
    base: u32,
    tmin: u32,
    tmax: u32,
    skew: u32,
    damp: u32,
) -> u32 {
    let mut delta = if firsttime { delta / damp } else { delta / 2 };
    delta += delta / numpoints;
    let mut k = 0u32;
    while delta > ((base - tmin) * tmax) / 2 {
        delta /= base - tmin;
        k += base;
    }
    k + (((base - tmin + 1) * delta) / (delta + skew))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Legacy scheme ----------------------------------------------------

    #[test]
    fn legacy_basic_path_strips_hash() {
        // `4core 3fmt 9Formatter 3pad` then the 17-char hash `h`+16 hex.
        assert_eq!(
            demangle("_ZN4core3fmt9Formatter3pad17h0123456789abcdefE"),
            "core::fmt::Formatter::pad"
        );
    }

    #[test]
    fn legacy_angle_and_space_escapes() {
        // Component `$LT$impl$u20$Foo$GT$` (len 20) -> `<impl Foo>`, then
        // `3bar`, then the hash.
        assert_eq!(
            demangle("_ZN20$LT$impl$u20$Foo$GT$3bar17hffffffffffffffffE"),
            "<impl Foo>::bar"
        );
    }

    #[test]
    fn legacy_ref_and_unicode_space() {
        // `$RF$mut$u20$T` (len 13) -> `&mut T`.
        assert_eq!(demangle("_ZN13$RF$mut$u20$T17h0000000000000000E"), "&mut T");
    }

    #[test]
    fn legacy_comma_and_general_unicode() {
        // `a$C$$u7e$b` (len 10) -> `a,~b` ($C$ comma, $u7e$ tilde).
        assert_eq!(demangle("_ZN10a$C$$u7e$b17h1111111111111111E"), "a,~b");
    }

    #[test]
    fn legacy_leading_underscore_before_digit_dropped() {
        // Component `_0` (len 2) drops its guard `_` -> `0`.
        assert_eq!(demangle("_ZN2_03bar17h2222222222222222E"), "0::bar");
    }

    #[test]
    fn legacy_macho_double_underscore() {
        assert_eq!(demangle("__ZN3foo3barE"), "foo::bar");
    }

    #[test]
    fn legacy_double_dot_is_path_separator() {
        // `foo..bar` (len 8) -> `foo::bar`.
        assert_eq!(demangle("_ZN8foo..bar17h3333333333333333E"), "foo::bar");
    }

    #[test]
    fn legacy_trailing_bytes_after_e_are_refused() {
        // `_ZN6Widget3addEi` is a C++ function (`Widget::add(int)`): the
        // `i` after the `E` is its parameter list, so the legacy parser
        // must refuse it rather than swallow the name and drop the rest.
        assert_eq!(try_demangle("_ZN6Widget3addEi"), None);
    }

    #[test]
    fn legacy_dot_suffix_after_e_is_tolerated() {
        assert_eq!(demangle("_ZN3foo3barE.llvm.1234567890"), "foo::bar");
    }

    // -- v0 scheme --------------------------------------------------------

    #[test]
    fn v0_plain_path() {
        // Nv C7mycrate 3foo  ->  mycrate::foo
        assert_eq!(demangle("_RNvC7mycrate3foo"), "mycrate::foo");
    }

    #[test]
    fn v0_nested_modules() {
        // Nv Nt Nt C4core 3fmt 9Formatter 3pad
        assert_eq!(
            demangle("_RNvNtNtC4core3fmt9Formatter3pad"),
            "core::fmt::Formatter::pad"
        );
    }

    #[test]
    fn v0_generic_type_arg() {
        // I (Nv C1a 3foo) h E  ->  a::foo::<u8>
        assert_eq!(demangle("_RINvC1a3foohE"), "a::foo::<u8>");
    }

    #[test]
    fn v0_trait_impl_method() {
        // Nv X C1c (NtC1c3Foo) (NtC1c3Bar) 3baz
        assert_eq!(
            demangle("_RNvXC1cNtC1c3FooNtC1c3Bar3baz"),
            "<c::Foo as c::Bar>::baz"
        );
    }

    #[test]
    fn v0_inherent_impl_method() {
        // Nv M C1c (NtC1c3Foo) 3new
        assert_eq!(demangle("_RNvMC1cNtC1c3Foo3new"), "<c::Foo>::new");
    }

    #[test]
    fn v0_const_generic_array() {
        // I (Nv C1a 3foo) A h (j 4 _) E  ->  a::foo::<[u8; 4]>
        assert_eq!(demangle("_RINvC1a3fooAhj4_E"), "a::foo::<[u8; 4]>");
    }

    #[test]
    fn v0_backreference_reuse() {
        // I (NvC1a3foo) T (NtC1c3Foo) B<11> E E — the second tuple element
        // is a backref to the byte offset (11) of the first `Foo` path.
        assert_eq!(
            demangle("_RINvC1a3fooTNtC1c3FooBa_EE"),
            "a::foo::<(c::Foo, c::Foo)>"
        );
    }

    #[test]
    fn v0_reference_type_arg() {
        // I (NvC1a3foo) R h E  ->  a::foo::<&u8>
        assert_eq!(demangle("_RINvC1a3fooRhE"), "a::foo::<&u8>");
    }

    #[test]
    fn v0_strips_llvm_suffix() {
        assert_eq!(demangle("_RNvC7mycrate3foo.llvm.9012345"), "mycrate::foo");
    }

    // -- Passthrough ------------------------------------------------------

    #[test]
    fn passthrough_non_mangled() {
        for s in ["main", "_start", "sub_401000", "", "_Z3foov", "printf"] {
            assert_eq!(demangle(s), s, "should pass {s:?} through unchanged");
        }
    }

    #[test]
    fn passthrough_already_demangled_is_idempotent() {
        let text = "core::fmt::Formatter::pad";
        assert_eq!(demangle(text), text);
        assert_eq!(demangle(&demangle(text)), text);
    }

    #[test]
    fn is_mangled_prefix_check() {
        assert!(is_mangled("_ZN3foo3barE"));
        assert!(is_mangled("_RNvC7mycrate3foo"));
        assert!(is_mangled("__ZN3fooE"));
        assert!(!is_mangled("main"));
        assert!(!is_mangled("_Z3foov"));
    }

    // -- No panic / no hang ----------------------------------------------

    #[test]
    fn truncated_prefixes_never_panic() {
        let samples = [
            "_ZN4core3fmt9Formatter3pad17h0123456789abcdefE",
            "_RNvNtNtC4core3fmt9Formatter3pad",
            "_RINvC1a3fooTNtC1c3FooBa_EE",
            "_RNvXC1cNtC1c3FooNtC1c3Bar3baz",
            "_ZN20$LT$impl$u20$Foo$GT$3bar17hffffffffffffffffE",
        ];
        for sym in samples {
            for i in 0..=sym.len() {
                // Every prefix must return *something* without panicking.
                let _ = demangle(&sym[..i]);
            }
        }
    }

    #[test]
    fn cyclic_backref_terminates() {
        // A backref that points at itself / forward is rejected by the
        // strictly-backward rule, so these terminate and pass through.
        for sym in ["_RB_", "_RB0_", "_RNvB_1a", "_RINvC1a3fooB0_E"] {
            let out = demangle(sym);
            // Bounded output and no hang (reaching here proves termination).
            assert!(out.len() <= MAX_OUTPUT + 32);
        }
    }

    #[test]
    fn self_referential_generic_backref_terminates() {
        // `_RIB_...` — a generic whose own path is a backref to offset 0
        // (the `I` itself). Backward-only + budget guarantee termination.
        let out = demangle("_RIB_hE");
        assert!(out.len() <= MAX_OUTPUT + 32);
    }

    #[test]
    fn deterministic_fuzz_sweep_never_panics_or_hangs() {
        // A fixed-seed LCG drives a few thousand `_R` + pseudo-random-ASCII
        // inputs through the parser. The assertion is weak by design — the
        // real property under test is that every call *returns* within the
        // caps (no panic, no infinite loop, bounded memory).
        const ALPHABET: &[u8] =
            b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_.$EBILNXYMCTRQK";
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };

        for _ in 0..4000 {
            let len = (next() % 48) as usize;
            let mut body = String::with_capacity(len + 2);
            body.push_str("_R");
            for _ in 0..len {
                let idx = (next() as usize) % ALPHABET.len();
                body.push(ALPHABET[idx] as char);
            }
            let out = demangle(&body);
            assert!(
                out.len() <= MAX_OUTPUT + 32,
                "output exceeded cap for {body:?}"
            );
        }
    }

    #[test]
    fn fuzz_sweep_legacy_shape_never_panics() {
        const ALPHABET: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz$_.E";
        let mut state: u64 = 0xdead_beef_cafe_babe;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for _ in 0..4000 {
            let len = (next() % 48) as usize;
            let mut body = String::with_capacity(len + 3);
            body.push_str("_ZN");
            for _ in 0..len {
                let idx = (next() as usize) % ALPHABET.len();
                body.push(ALPHABET[idx] as char);
            }
            let _ = demangle(&body);
        }
    }

    #[test]
    fn punycode_identifier_roundtrip_shape() {
        // A `u`-flagged identifier decodes via punycode. An all-basic
        // identifier keeps its trailing `_` delimiter, so `abc_` (basic
        // "abc", no deltas) decodes to "abc".
        assert_eq!(decode_punycode("abc_").as_deref(), Some("abc"));
        // A real mixed case: standard punycode "a_c" would decode with a
        // delta stream; here we only require it not panic and stay bounded.
        let _ = decode_punycode("Foo_bar");
        // Known vector: standard punycode encodes "bücher" as "bcher-kva";
        // the v0 variant swaps the `-` delimiter for `_`.
        assert_eq!(decode_punycode("bcher_kva").as_deref(), Some("bücher"));
        // And end-to-end through a v0 identifier: `Cu9bcher_kva` is a
        // crate whose punycode-flagged 9-byte name is "bücher".
        assert_eq!(demangle("_RNvCu9bcher_kva3foo"), "bücher::foo");
        // Pure-delta with empty basic set must not panic and stays bounded.
        let _ = decode_punycode("99999999");
    }
}
