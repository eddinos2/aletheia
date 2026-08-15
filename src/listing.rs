//! The disassembly listing — every other module's work, rendered for a
//! human.
//!
//! Analysis produces facts; a listing is where an analyst *reads* them.
//! That makes this module a joiner, not an analyzer: it computes nothing
//! about the program that [`cfg`], [`xref`], [`strings`], and
//! [`annotate`] have not already established, and it never guesses. Its
//! whole job is to put those separately recovered views on the same page
//! at the same address.
//!
//! # Why the output is a `String`, and why it is deterministic
//!
//! A listing is a *diffable artifact*. Analysts commit them, mail them,
//! and compare two builds of the same binary by running `diff` over two
//! renders. That only works if the same inputs always produce the same
//! bytes, so every collection walked here is a [`BTreeMap`] or
//! [`BTreeSet`] — never a hash map — functions are emitted in entry-VA
//! order, and every tie is broken by address. Rendering the same program
//! twice is byte-identical by construction, and rendering the *rebased*
//! same program differs only where addresses genuinely differ.
//!
//! # Under-approximation is visible, never silent
//!
//! [`crate::asmtext`] formatters are best-effort by contract: `None`
//! means "this encoding has no text yet". A listing must not paper over
//! that, because an analyst who reads invented mnemonics is worse off
//! than one who reads raw bytes. So the fallback prints `db <bytes>` —
//! *but still symbolizes control flow*, because the branch and call
//! targets come from [`Flow`], which the decoder always knows even when
//! the text renderer does not:
//!
//! ```text
//!   0x0000000000401004  e8 07 00 00 00           db e8 07 00 00 00   ; call -> 0x0000000000401010 (inflate)
//! ```
//!
//! The same principle governs annotations. [`annotate::Db`] entries are
//! anchored content-relatively (see [`crate::anchor`]), so after the
//! binary changes an annotation may reattach by *shape* or by bare
//! *address* rather than by an exact byte match. A weak reattachment is
//! marked `?` and explained on its own note line: the analyst must be
//! able to see that a name is a best-effort carry-over, not a fact about
//! these bytes.
//!
//! # Resource caps
//!
//! Hostile input reaches this renderer through the same path as anything
//! else, and text is the one output that a caller cannot bound after the
//! fact. [`Options`] therefore caps both the number of functions and the
//! total number of lines; hitting either cap ends the listing with an
//! explicit `; ...` footer, so a truncated listing is never mistaken for
//! a complete one.

use std::collections::{BTreeMap, BTreeSet};

use crate::anchor::{AnchorIndex, Resolution};
use crate::annotate::{self, Field};
use crate::asmtext::{AsmFormatter, formatter_for};
use crate::cfg::{self, BasicBlock, Function, Terminator};
use crate::cxxdemangle;
use crate::demangle;
use crate::model::{Decoder, Flow, Image, decoder_for};
use crate::strings;
use crate::xref::{self, Xrefs};

/// Hex digits in every printed virtual address. Fixed-width and
/// uniform across the whole listing so addresses line up in a column and
/// two renders diff cleanly.
const VA_HEX: usize = 16;

/// Raw bytes shown per instruction before eliding. x86-64 encodings run
/// to 15 bytes; showing all of them would push the text column past any
/// reasonable width for the sake of a handful of instructions.
const BYTES_SHOWN: usize = 8;

/// Width of the raw-byte column: `BYTES_SHOWN` two-digit bytes with
/// single-space separators. An elided run prints seven bytes plus `..`,
/// which is exactly the same width.
const BYTES_COL: usize = BYTES_SHOWN * 3 - 1;

/// Column at which a trailing `;` comment starts, when the line is short
/// enough to reach it.
const COMMENT_COL: usize = 74;

/// Cross-references listed inline on a label before the count elides.
const XREFS_SHOWN: usize = 4;

/// Characters of a referenced string shown inline.
const STRING_PREVIEW: usize = 40;

/// Instructions rendered per basic block before the body is cut short.
/// A block is a straight-line run, so this is only ever reached on a
/// pathological image; it exists so a single block cannot monopolize the
/// line budget.
const MAX_BLOCK_INSNS: usize = 65_536;

/// What to render and how much of it.
///
/// The caps are not tuning knobs but a safety contract: see the module
/// docs. Both default to values that comfortably cover a real binary
/// while still bounding output on a hostile one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// Maximum functions rendered, in entry-VA order. Remaining
    /// functions are reported in the footer.
    pub max_functions: usize,
    /// Maximum lines emitted across the whole listing, footers excluded.
    pub max_lines: usize,
    /// Show the raw-byte column between the address and the text.
    pub bytes_column: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            max_functions: 4096,
            max_lines: 262_144,
            bytes_column: true,
        }
    }
}

/// Render every recovered function of `program` as a listing.
///
/// `program` must be the [`cfg::Program`] recovered from `image`;
/// passing a mismatched pair yields a listing of whatever the addresses
/// happen to hold, never a panic. `db`, when given, supplies analyst
/// names, types, and comments, reattached to this program through
/// [`AnchorIndex`].
///
/// The result always ends with a newline (and is never empty: a program
/// with no functions renders as a single explanatory comment). Never
/// panics on any input, including blocks whose bytes fall outside the
/// file.
pub fn render(
    image: &dyn Image,
    program: &cfg::Program,
    db: Option<&annotate::Db>,
    opts: &Options,
) -> String {
    let ctx = Ctx::build(image, program, db, opts);
    let total = program.functions.len();
    let mut out = Out::new(opts.max_lines);

    for func in program.functions.values().take(opts.max_functions) {
        if out.exhausted() {
            break;
        }
        if !out.lines.is_empty() {
            out.push(String::new());
        }
        ctx.function(func, &mut out);
    }

    ctx.finish(out, total.saturating_sub(opts.max_functions))
}

/// Render the single function entered at `entry`, with the same
/// conventions as [`render`].
///
/// Returns an explanatory comment (not an empty string) when `entry`
/// does not name a recovered function, so the output is always a valid
/// listing fragment. The joined views — cross-references, strings, and
/// annotations — are still computed against the whole program, because a
/// call target's name and an incoming xref both live outside the
/// function being printed.
pub fn render_function(
    image: &dyn Image,
    program: &cfg::Program,
    entry: u64,
    db: Option<&annotate::Db>,
    opts: &Options,
) -> String {
    let Some(func) = program.functions.get(&entry) else {
        return format!("; (no function recovered at {})\n", hex_va(entry));
    };
    let ctx = Ctx::build(image, program, db, opts);
    let mut out = Out::new(opts.max_lines);
    ctx.function(func, &mut out);
    ctx.finish(out, 0)
}

// ---------------------------------------------------------------------------
// Line buffer
// ---------------------------------------------------------------------------

/// A line-capped output buffer.
///
/// Every line in the listing goes through [`Out::push`], so the cap is
/// enforced in one place and cannot be bypassed by a code path that
/// forgets to check it.
struct Out {
    lines: Vec<String>,
    max: usize,
    capped: bool,
}

impl Out {
    fn new(max: usize) -> Self {
        Out {
            lines: Vec::new(),
            max,
            capped: false,
        }
    }

    fn push(&mut self, line: String) {
        if self.lines.len() >= self.max {
            self.capped = true;
            return;
        }
        self.lines.push(line);
    }

    /// Whether the budget is spent. Called only where content still
    /// remains to be rendered, so it records the truncation: a caller
    /// that bails out here must not leave the footer unwritten.
    fn exhausted(&mut self) -> bool {
        if self.lines.len() >= self.max {
            self.capped = true;
            return true;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Rendering context
// ---------------------------------------------------------------------------

/// Everything the renderer joins onto the instruction stream, gathered
/// once so a function's cost is proportional to its own size.
struct Ctx<'a> {
    image: &'a dyn Image,
    opts: &'a Options,
    decoder: Option<&'static dyn Decoder>,
    formatter: Option<&'static dyn AsmFormatter>,
    /// Cross-references, when they could be recovered. Absent only when
    /// the architecture has no decoder, in which case nothing decodes
    /// either and the listing is raw bytes throughout.
    xrefs: Option<Xrefs>,
    /// Display name per recovered function entry VA, name precedence
    /// already applied.
    names: BTreeMap<u64, String>,
    /// Import slot VA to imported name, for indirect calls and jumps
    /// through a statically addressed cell.
    slots: BTreeMap<u64, String>,
    /// String start VA to `(byte length, text)`, for data-reference
    /// previews.
    strings: BTreeMap<u64, (u64, String)>,
    /// Annotations placed onto this program: `(entry VA, field)` to the
    /// value and, when the anchor reattached by weak evidence, the word
    /// naming that evidence.
    ann: BTreeMap<(u64, Field), (String, Option<&'static str>)>,
}

impl<'a> Ctx<'a> {
    fn build(
        image: &'a dyn Image,
        program: &'a cfg::Program,
        db: Option<&annotate::Db>,
        opts: &'a Options,
    ) -> Ctx<'a> {
        let arch = image.arch();

        // Annotations first: a Db name outranks every other source, so
        // the name table below has to see them.
        let mut ann = BTreeMap::new();
        if let Some(db) = db {
            let index = AnchorIndex::build(image, program);
            for placed in db.resolve_onto(&index) {
                let weak = match placed.resolution {
                    Resolution::Exact(_) => None,
                    Resolution::Shape(_) => Some("shape"),
                    Resolution::Absolute(_) => Some("address"),
                    // `resolve_onto` drops these; matched for totality.
                    Resolution::Ambiguous(_) | Resolution::Unresolved => continue,
                };
                // First writer wins, so two anchors landing on one VA
                // resolve deterministically (`resolve_onto` sorts).
                ann.entry((placed.va, placed.field))
                    .or_insert_with(|| (placed.value.to_string(), weak));
            }
        }

        let mut slots = BTreeMap::new();
        for s in image.import_slots() {
            slots.entry(s.slot_va).or_insert(s.name);
        }

        // Name precedence, strongest first: an analyst's Db name, then the
        // name analysis recovered for the function (a loader symbol or a
        // Go pclntab entry, both supplied by `cfg`), then an import-thunk
        // target, then a synthesized `sub_`. A recovered name may be a raw
        // linker symbol, so it is run through the demanglers for display; a
        // Db name is the analyst's own text and needs no demangling.
        // Rust is tried before C++ because Rust's legacy scheme is itself
        // a valid Itanium mangling — the reverse order would render Rust
        // symbols as C++. Both refuse anything unmangled, so an ordinary
        // name passes through untouched.
        let names = program
            .functions
            .iter()
            .map(|(&entry, func)| {
                let name = ann
                    .get(&(entry, Field::Name))
                    .map(|(v, weak)| mark(v, *weak))
                    .or_else(|| {
                        func.name
                            .clone()
                            .filter(|n| !n.is_empty())
                            .map(|n| {
                                demangle::try_demangle(&n)
                                    .or_else(|| cxxdemangle::try_demangle(&n))
                                    .unwrap_or(n)
                            })
                    })
                    .or_else(|| thunk_import(func).map(str::to_string))
                    .unwrap_or_else(|| format!("sub_{entry:x}"));
                (entry, name)
            })
            .collect();

        Ctx {
            image,
            opts,
            decoder: decoder_for(arch),
            formatter: formatter_for(arch),
            xrefs: xref::compute(image, program).ok(),
            names,
            slots,
            strings: string_map(image),
            ann,
        }
    }

    /// Join the buffered lines into the final text, appending the
    /// truncation footers that make an incomplete listing self-evident.
    fn finish(&self, out: Out, omitted_functions: usize) -> String {
        let mut text = String::new();
        for line in &out.lines {
            text.push_str(line);
            text.push('\n');
        }
        if out.capped {
            text.push_str(&format!(
                "; ... listing truncated at {} lines (max_lines)\n",
                self.opts.max_lines
            ));
        }
        if omitted_functions > 0 {
            text.push_str(&format!(
                "; ... {omitted_functions} more function(s) not shown (max_functions = {})\n",
                self.opts.max_functions
            ));
        }
        if text.is_empty() {
            text.push_str("; (no functions recovered)\n");
        }
        text
    }

    // -- one function --------------------------------------------------

    fn function(&self, func: &Function, out: &mut Out) {
        let name = self.name_of(func.entry);
        let bytes: u64 = func
            .blocks
            .values()
            .map(|b| b.end.saturating_sub(b.start))
            .sum();
        let ty = self
            .ann
            .get(&(func.entry, Field::Type))
            .map(|(v, weak)| format!(" : {}", mark(v, *weak)))
            .unwrap_or_default();

        out.push(format!(
            "; ======== {name}{ty} @ {}  {}, {} ========",
            hex_va(func.entry),
            plural(func.blocks.len() as u64, "block"),
            plural(bytes, "byte"),
        ));
        // Blocks are emitted in address order, and a function's entry is
        // not always its lowest address — a compiler may lay error paths
        // out below the prologue. The name label therefore goes where
        // the entry *is*, not at the top of the body, so no label ever
        // sits above code it does not name.
        let labels = block_labels(func);
        if !func.blocks.contains_key(&func.entry) {
            self.label(&name, func.entry, out);
            self.notes(func.entry, out);
        }
        for blk in func.blocks.values() {
            if out.exhausted() {
                return;
            }
            if blk.start == func.entry {
                self.label(&name, func.entry, out);
                self.notes(func.entry, out);
            } else if labels.contains(&blk.start) {
                self.label(&format!("loc_{:x}", blk.start), blk.start, out);
                self.notes(blk.start, out);
            }
            self.block(func, blk, &labels, out);
        }
    }

    /// Emit a `name:` line, carrying the incoming cross-references that
    /// answer "who reaches this address?".
    fn label(&self, name: &str, va: u64, out: &mut Out) {
        let mut line = format!("{name}:");
        if let Some(xrefs) = self.xrefs.as_ref() {
            // `refs_to` is sorted by `(from, kind)`; two kinds from one
            // instruction are one source for display purposes.
            let mut sources: Vec<u64> = xrefs.refs_to(va).iter().map(|x| x.from).collect();
            sources.dedup();
            if !sources.is_empty() {
                let shown: Vec<String> =
                    sources.iter().take(XREFS_SHOWN).map(|v| hex_va(*v)).collect();
                let mut note = format!("xref: {}", shown.join(", "));
                let more = sources.len().saturating_sub(XREFS_SHOWN);
                if more > 0 {
                    note.push_str(&format!(", +{more} more"));
                }
                pad_to(&mut line, COMMENT_COL);
                line.push_str("; ");
                line.push_str(&note);
            }
        }
        out.push(line);
    }

    /// Emit the analyst's comment and any weak-reattachment warnings for
    /// the location at `va`.
    fn notes(&self, va: u64, out: &mut Out) {
        if let Some((text, weak)) = self.ann.get(&(va, Field::Comment)) {
            for line in text.lines() {
                out.push(format!("; note: {}", mark(line, *weak)));
            }
        }
        for field in [Field::Name, Field::Type, Field::Comment] {
            if let Some((_, Some(how))) = self.ann.get(&(va, field)) {
                out.push(format!(
                    "; ? {} reattached by {how} match, not an exact byte match",
                    field_word(field)
                ));
            }
        }
    }

    // -- one basic block -----------------------------------------------

    fn block(&self, func: &Function, blk: &BasicBlock, labels: &BTreeSet<u64>, out: &mut Out) {
        let image_bytes = self.image.bytes();
        let mut va = blk.start;

        for _ in 0..MAX_BLOCK_INSNS {
            if va >= blk.end {
                break;
            }
            if out.exhausted() {
                return;
            }
            // Resolved per instruction rather than once per block: a
            // block can straddle the end of its region's file backing,
            // and the mapping is only linear within one region.
            let Some(off) = self.image.va_to_offset(va) else {
                out.push(self.line(va, &[], "db ?", Some("(no file backing)")));
                return;
            };
            let avail = image_bytes.get(off..).unwrap_or(&[]);
            if avail.is_empty() {
                out.push(self.line(va, &[], "db ?", Some("(past end of file)")));
                return;
            }

            let decoded = self.decoder.and_then(|d| d.decode_flow(avail, va).ok());
            let Some(decoded) = decoded.filter(|d| d.length > 0) else {
                let raw = &avail[..1];
                out.push(self.line(va, raw, &db_text(raw), Some("(undecodable)")));
                return;
            };

            // Clamp to the block and to the file: a decoder length is a
            // claim about bytes that may not all be present.
            let span = (decoded.length as u64).min(blk.end.saturating_sub(va)) as usize;
            let raw = &avail[..span.min(avail.len())];

            let text = self.formatter.and_then(|f| f.format(raw, va));
            // Only the block's last instruction may borrow the
            // terminator's import: that is the instruction `cfg`
            // classified when it built the terminator.
            let last = va.wrapping_add(decoded.length as u64) >= blk.end;
            let mut comments = self.flow_comment(
                func,
                labels,
                decoded.flow,
                text.is_some(),
                decoded.mem_target,
                last.then_some(&blk.terminator),
            );
            if let Some(s) = self.string_comment(va) {
                comments.push(s);
            }
            let comment = (!comments.is_empty()).then(|| comments.join("  "));

            let body = text.unwrap_or_else(|| db_text(raw));
            out.push(self.line(va, raw, &body, comment.as_deref()));

            let next = va.wrapping_add(decoded.length as u64);
            if next <= va {
                return; // zero-length or wrapped: never loop
            }
            va = next;
        }
        self.dead_end(blk, out);
    }

    /// Report a block that ended because analysis gave up rather than
    /// because control flow left it.
    ///
    /// [`Terminator::Undecodable`] and [`Terminator::Truncated`] both
    /// mean the instruction *after* the block was never analyzed, so the
    /// block itself has nothing to show for them. Without this line an
    /// analyst would see a function simply stop, with no way to tell an
    /// honest `ret` from an under-approximation.
    fn dead_end(&self, blk: &BasicBlock, out: &mut Out) {
        let reason = match blk.terminator {
            Terminator::Undecodable => "undecodable here: analysis stopped, nothing guessed",
            Terminator::Truncated => "a recovery cap was hit here: analysis stopped",
            _ => return,
        };
        let raw = self
            .image
            .va_to_offset(blk.end)
            .and_then(|off| self.image.bytes().get(off..off.saturating_add(1)))
            .unwrap_or(&[]);
        out.push(self.line(blk.end, raw, &db_text(raw), Some(reason)));
    }

    /// The control-flow fragment of an instruction's comment.
    ///
    /// With instruction text present, only the *name* is added — the
    /// text already carries the address. Without it, the fallback must
    /// carry both the flow class and the target, because `db e8 ...`
    /// says nothing on its own.
    fn flow_comment(
        &self,
        func: &Function,
        labels: &BTreeSet<u64>,
        flow: Flow,
        has_text: bool,
        mem_target: Option<u64>,
        terminator: Option<&Terminator>,
    ) -> Vec<String> {
        let word = flow_word(flow);
        match flow {
            Flow::Sequential => Vec::new(),
            Flow::Jump(t) | Flow::CondJump(t) | Flow::Call(t) => {
                let name = self.target_name(t, func, labels);
                match (has_text, name) {
                    (true, Some(n)) => vec![format!("-> {n}")],
                    (true, None) => Vec::new(),
                    (false, Some(n)) => vec![format!("{word} -> {} ({n})", hex_va(t))],
                    (false, None) => vec![format!("{word} -> {}", hex_va(t))],
                }
            }
            Flow::IndirectCall | Flow::IndirectJump => {
                // The slot table resolves any instruction that goes
                // through a statically addressed cell; the terminator
                // covers the block-ending case the same way `cfg` saw it.
                let import = mem_target
                    .and_then(|slot| self.slots.get(&slot))
                    .map(String::as_str)
                    .or_else(|| terminator.and_then(terminator_import));
                match (has_text, import) {
                    (true, Some(n)) => vec![format!("-> {n}")],
                    (true, None) => Vec::new(),
                    (false, Some(n)) => vec![format!("{word} -> {n}")],
                    (false, None) => vec![format!("{word} -> ?")],
                }
            }
            Flow::Return | Flow::Interrupt | Flow::Halt => {
                if has_text {
                    Vec::new()
                } else {
                    vec![word.to_string()]
                }
            }
        }
    }

    /// The preview of a string this instruction references, if any.
    ///
    /// Only *data* references are considered: a code reference lands on
    /// an instruction, and bytes that happen to decode as text there are
    /// not a string.
    fn string_comment(&self, va: u64) -> Option<String> {
        let xrefs = self.xrefs.as_ref()?;
        for xref in xrefs.refs_from(va) {
            if !xref.kind.is_data() {
                continue;
            }
            if let Some((start, (len, text))) = self.strings.range(..=xref.to).next_back()
                && xref.to < start.saturating_add(*len)
            {
                return Some(format!("\"{}\"", escape(text, STRING_PREVIEW)));
            }
        }
        None
    }

    // -- naming --------------------------------------------------------

    /// The display name of a recovered function entry.
    fn name_of(&self, entry: u64) -> String {
        self.names
            .get(&entry)
            .cloned()
            .unwrap_or_else(|| format!("sub_{entry:x}"))
    }

    /// The best name for a branch or call target, or `None` when the
    /// address is all that is known about it.
    ///
    /// A local block label only applies inside the function being
    /// printed: `loc_...` names are function-scoped in this listing.
    fn target_name(&self, va: u64, func: &Function, labels: &BTreeSet<u64>) -> Option<String> {
        if let Some(name) = self.names.get(&va) {
            return Some(name.clone());
        }
        if labels.contains(&va) {
            return Some(format!("loc_{va:x}"));
        }
        if va == func.entry {
            return Some(self.name_of(va));
        }
        self.xrefs
            .as_ref()
            .and_then(|x| x.symbol_at(va))
            .map(str::to_string)
    }

    // -- line layout ---------------------------------------------------

    /// Lay out one instruction line: address, optional raw bytes, text,
    /// and an optional right-hand comment.
    fn line(&self, va: u64, raw: &[u8], text: &str, comment: Option<&str>) -> String {
        let mut line = format!("  {}  ", hex_va(va));
        if self.opts.bytes_column {
            let hex = hex_bytes(raw);
            line.push_str(&hex);
            pad_by(&mut line, BYTES_COL.saturating_sub(hex.chars().count()));
            line.push_str("  ");
        }
        line.push_str(text);
        if let Some(comment) = comment {
            pad_to(&mut line, COMMENT_COL);
            line.push_str("; ");
            line.push_str(comment);
        }
        line
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Every VA in a listing, in one fixed-width lowercase form.
fn hex_va(va: u64) -> String {
    format!("0x{va:0VA_HEX$x}")
}

/// Raw bytes for the byte column, eliding past [`BYTES_SHOWN`] so the
/// column keeps its width for a 15-byte x86 encoding.
fn hex_bytes(raw: &[u8]) -> String {
    let elide = raw.len() > BYTES_SHOWN;
    let take = if elide { BYTES_SHOWN - 1 } else { raw.len() };
    let mut out: Vec<String> = raw[..take].iter().map(|b| format!("{b:02x}")).collect();
    if elide {
        out.push("..".to_string());
    }
    out.join(" ")
}

/// The raw-bytes fallback for an instruction with no rendered text.
fn db_text(raw: &[u8]) -> String {
    if raw.is_empty() {
        return "db ?".to_string();
    }
    format!("db {}", hex_bytes(raw))
}

/// The short word naming a flow class in a fallback comment.
fn flow_word(flow: Flow) -> &'static str {
    match flow {
        Flow::Sequential => "",
        Flow::Jump(_) => "jmp",
        Flow::CondJump(_) => "jcc",
        Flow::IndirectJump => "jmp *",
        Flow::Call(_) => "call",
        Flow::IndirectCall => "call *",
        Flow::Return => "ret",
        Flow::Interrupt => "int",
        Flow::Halt => "hlt",
    }
}

/// The word naming an annotation field in a warning line.
fn field_word(field: Field) -> &'static str {
    match field {
        Field::Name => "name",
        Field::Type => "type",
        Field::Comment => "comment",
    }
}

/// Append the weak-reattachment marker when the value came from an
/// anchor that did not match exactly.
fn mark(value: &str, weak: Option<&'static str>) -> String {
    match weak {
        Some(_) => format!("{value}?"),
        None => value.to_string(),
    }
}

/// The import a function forwards to when it is a jump thunk: its entry
/// block ends immediately in a jump through an import slot.
fn thunk_import(func: &Function) -> Option<&str> {
    terminator_import(&func.blocks.get(&func.entry)?.terminator)
}

/// The import name a terminator resolved to, if it resolved one.
fn terminator_import(term: &Terminator) -> Option<&str> {
    match term {
        Terminator::IndirectJump { import } | Terminator::IndirectCall { import, .. } => {
            import.as_deref()
        }
        _ => None,
    }
}

/// The block starts inside `func` that deserve a `loc_...` label.
///
/// Two cases earn one: a block that something *branches* to (a
/// fall-through edge needs no label, since the previous line already
/// leads there), and a block that is not contiguous with the block
/// before it — without a label, an analyst reading down the listing
/// would have no marker that the address stream jumped.
fn block_labels(func: &Function) -> BTreeSet<u64> {
    let mut labels = BTreeSet::new();
    for blk in func.blocks.values() {
        match &blk.terminator {
            Terminator::Jump(t) if func.blocks.contains_key(t) => {
                labels.insert(*t);
            }
            Terminator::CondJump { taken, .. } if func.blocks.contains_key(taken) => {
                labels.insert(*taken);
            }
            _ => {}
        }
    }
    let mut prev_end: Option<u64> = None;
    for blk in func.blocks.values() {
        if prev_end.is_some_and(|end| end != blk.start) {
            labels.insert(blk.start);
        }
        prev_end = Some(blk.end);
    }
    labels.remove(&func.entry);
    labels
}

/// Recovered strings keyed by start VA, for data-reference previews.
///
/// Only mapped strings are useful here — an unmapped run has no address
/// a code reference could point at — and the cap keeps a file that is
/// mostly text from dominating the render's cost.
fn string_map(image: &dyn Image) -> BTreeMap<u64, (u64, String)> {
    let config = strings::Config {
        min_len: 4,
        max_strings: 65_536,
        max_len: 4096,
        scan_utf16: true,
        require_mapped: true,
    };
    let mut out = BTreeMap::new();
    for found in strings::extract(image, &config) {
        let Some(va) = found.va else {
            continue;
        };
        let width = match found.encoding {
            strings::Encoding::Ascii => 1u64,
            strings::Encoding::Utf16Le => 2u64,
        };
        let len = (found.text.chars().count() as u64).saturating_mul(width);
        out.entry(va).or_insert((len, found.text));
    }
    out
}

/// Escape `text` for inline display and truncate it to `max` characters,
/// marking a truncation with a trailing `...`.
///
/// Control characters and quotes become escapes: a listing is read by
/// humans and diffed by tools, and neither survives a raw newline or an
/// unbalanced quote in the middle of a line.
fn escape(text: &str, max: usize) -> String {
    let mut out = String::new();
    for (i, ch) in text.chars().enumerate() {
        if i >= max {
            out.push_str("...");
            break;
        }
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7F => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Pad `line` with spaces until it is at least `col` characters wide,
/// leaving a two-space gap when it already overflows the column.
fn pad_to(line: &mut String, col: usize) {
    let width = line.chars().count();
    pad_by(line, col.saturating_sub(width).max(2));
}

/// `"1 block"` / `"3 blocks"` — a header an analyst reads should not
/// read like a machine wrote it.
fn plural(n: u64, word: &str) -> String {
    if n == 1 {
        format!("{n} {word}")
    } else {
        format!("{n} {word}s")
    }
}

/// Append `n` spaces.
fn pad_by(line: &mut String, n: usize) {
    for _ in 0..n {
        line.push(' ');
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anchor;
    use crate::cfg::recover;
    use crate::model::load;
    use crate::pe::tests::synthetic_pe64;

    /// Image base of the synthetic PE fixture.
    const PE_BASE: u64 = 0x1_4000_0000;
    /// Entry VA of the synthetic PE fixture (RVA 0x1000, file 0x200).
    const PE_ENTRY: u64 = PE_BASE + 0x1000;

    /// A two-function x86-64 body placed at the fixture's entry:
    ///
    /// ```text
    /// 140001000  xor eax, eax
    /// 140001002  je  140001009
    /// 140001004  call 140001010
    /// 140001009  ret
    /// 140001010  nop
    /// 140001011  ret
    /// ```
    const SAMPLE: &[u8] = &[
        0x31, 0xC0, // xor eax, eax
        0x74, 0x05, // je +5
        0xE8, 0x07, 0x00, 0x00, 0x00, // call +7
        0xC3, // ret
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad to 0x1010
        0x90, // nop
        0xC3, // ret
    ];

    /// The fixture with `code` at the entry point.
    ///
    /// The base fixture points its import directory at RVA 0x1010 —
    /// inside `.text` — so that its own import tests have something to
    /// parse. A listing fixture wants that range to be *code*, so the
    /// import data directory (index 1, at `dirs + 8`) is cleared.
    fn pe_with(code: &[u8]) -> Vec<u8> {
        let mut img = synthetic_pe64();
        let dirs = 0x80 + 4 + 20 + 112;
        img[dirs + 8..dirs + 16].fill(0);
        img[0x200..0x200 + code.len()].copy_from_slice(code);
        img
    }

    /// Render a synthetic PE holding `code`, with `opts` and no database.
    fn render_pe(code: &[u8], opts: &Options) -> String {
        render_pe_with(code, None, opts)
    }

    fn render_pe_with(code: &[u8], db: Option<&annotate::Db>, opts: &Options) -> String {
        let img = pe_with(code);
        let image = load(&img).expect("synthetic PE loads");
        let program = recover(image.as_ref()).expect("x86-64 recovers");
        render(image.as_ref(), &program, db, opts)
    }

    /// A database naming (and typing, and commenting) the entry function
    /// of a program recovered from `code`.
    fn db_naming_entry(code: &[u8]) -> annotate::Db {
        let img = pe_with(code);
        let image = load(&img).expect("synthetic PE loads");
        let program = recover(image.as_ref()).expect("x86-64 recovers");
        let target = anchor::of_function(image.as_ref(), &program.functions[&PE_ENTRY]);
        let mut db = annotate::Db::new();
        db.set_name(target, "decode_frame");
        db
    }

    // -- the golden listing --------------------------------------------

    /// The whole rendered listing of a known program, byte for byte.
    ///
    /// A golden-in-source assertion is the point: every future change to
    /// spacing, symbolization, or comment placement has to be made
    /// deliberately, because it shows up here as a diff an analyst can
    /// read. Instruction text comes from the [`crate::asmtext`]
    /// formatter; with the operand text present, the flow comment carries
    /// only the target's name (`-> loc_...`), since the mnemonic already
    /// prints the address.
    #[test]
    fn golden_listing_of_a_known_program() {
        let expected = "\
; ======== sub_140001000 @ 0x0000000140001000  3 blocks, 10 bytes ========
sub_140001000:
  0x0000000140001000  31 c0                    xor eax, eax
  0x0000000140001002  74 05                    je 0x140001009             ; -> loc_140001009
  0x0000000140001004  e8 07 00 00 00           call 0x140001010           ; -> sub_140001010
loc_140001009:                                                            ; xref: 0x0000000140001002
  0x0000000140001009  c3                       ret

; ======== sub_140001010 @ 0x0000000140001010  1 block, 2 bytes ========
sub_140001010:                                                            ; xref: 0x0000000140001004
  0x0000000140001010  90                       nop
  0x0000000140001011  c3                       ret
";
        assert_eq!(render_pe(SAMPLE, &Options::default()), expected);
    }

    #[test]
    fn output_always_ends_with_a_newline() {
        for opts in [
            Options::default(),
            Options {
                bytes_column: false,
                ..Options::default()
            },
        ] {
            let text = render_pe(SAMPLE, &opts);
            assert!(text.ends_with('\n'), "{text:?}");
        }
    }

    #[test]
    fn bytes_column_can_be_dropped() {
        let text = render_pe(
            SAMPLE,
            &Options {
                bytes_column: false,
                ..Options::default()
            },
        );
        assert!(
            text.contains("  0x0000000140001000  xor eax, eax\n"),
            "no byte column expected:\n{text}"
        );
    }

    // -- name precedence -----------------------------------------------

    #[test]
    fn db_name_outranks_the_synthesized_one() {
        let plain = render_pe(SAMPLE, &Options::default());
        assert!(plain.contains("sub_140001000:"), "{plain}");
        assert!(!plain.contains("decode_frame"), "{plain}");

        let db = db_naming_entry(SAMPLE);
        let named = render_pe_with(SAMPLE, Some(&db), &Options::default());
        assert!(
            named.contains("; ======== decode_frame @ 0x0000000140001000"),
            "{named}"
        );
        assert!(named.contains("decode_frame:"), "{named}");
        // The unannotated function keeps its synthesized name...
        assert!(named.contains("sub_140001010:"), "{named}");
        // ...and the call to the annotated one is symbolized through it.
        assert!(!named.contains("(sub_140001000)"), "{named}");
    }

    #[test]
    fn db_type_and_comment_reach_the_header_and_a_note() {
        let img = pe_with(SAMPLE);
        let image = load(&img).unwrap();
        let program = recover(image.as_ref()).unwrap();
        let target = anchor::of_function(image.as_ref(), &program.functions[&PE_ENTRY]);
        let mut db = annotate::Db::new();
        db.set_name(target, "decode_frame");
        db.set_type(target, "int(const char *)");
        db.set_comment(target, "clears the accumulator first");

        let text = render(image.as_ref(), &program, Some(&db), &Options::default());
        assert!(
            text.contains("; ======== decode_frame : int(const char *) @ 0x0000000140001000"),
            "{text}"
        );
        assert!(
            text.contains("; note: clears the accumulator first\n"),
            "{text}"
        );
        // An exact byte match must not be flagged as weak.
        assert!(!text.contains("; ?"), "{text}");
    }

    #[test]
    fn a_weak_reattachment_is_marked_and_explained() {
        // Capture the anchor against one build, then render another that
        // differs only in the `call` displacement: the byte fingerprint
        // misses and the annotation reattaches by shape alone.
        let db = db_naming_entry(SAMPLE);
        let mut shifted = SAMPLE.to_vec();
        shifted[5] = 0x08; // call target 0x140001011 instead of 0x140001010

        let text = render_pe_with(&shifted, Some(&db), &Options::default());
        assert!(text.contains("decode_frame?"), "{text}");
        assert!(
            text.contains("; ? name reattached by shape match, not an exact byte match\n"),
            "{text}"
        );
    }

    // -- determinism ---------------------------------------------------

    #[test]
    fn rendering_is_byte_identical_across_runs() {
        let db = db_naming_entry(SAMPLE);
        let once = render_pe_with(SAMPLE, Some(&db), &Options::default());
        let twice = render_pe_with(SAMPLE, Some(&db), &Options::default());
        assert_eq!(once, twice);
    }

    /// The function map is a `BTreeMap`, so discovery order cannot reach
    /// the output — but assert it, because the guarantee is the point.
    #[test]
    fn discovery_order_cannot_reach_the_output() {
        let img = pe_with(SAMPLE);
        let image = load(&img).unwrap();
        let forward = recover(image.as_ref()).unwrap();

        let mut reversed = forward.clone();
        reversed.functions = forward.functions.iter().rev().map(|(k, v)| (*k, v.clone())).collect();
        reversed.call_graph = forward
            .call_graph
            .iter()
            .rev()
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        assert_eq!(
            render(image.as_ref(), &forward, None, &Options::default()),
            render(image.as_ref(), &reversed, None, &Options::default()),
        );
    }

    // -- resource caps -------------------------------------------------

    #[test]
    fn max_functions_caps_the_listing_and_says_so() {
        let text = render_pe(
            SAMPLE,
            &Options {
                max_functions: 1,
                ..Options::default()
            },
        );
        assert!(text.contains("sub_140001000:"), "{text}");
        assert!(!text.contains("sub_140001010:"), "{text}");
        assert!(
            text.contains("; ... 1 more function(s) not shown (max_functions = 1)\n"),
            "{text}"
        );
    }

    #[test]
    fn max_lines_caps_the_listing_and_says_so() {
        let opts = Options {
            max_lines: 3,
            ..Options::default()
        };
        let text = render_pe(SAMPLE, &opts);
        let body: Vec<&str> = text.lines().filter(|l| !l.starts_with("; ...")).collect();
        assert_eq!(body.len(), 3, "{text}");
        assert!(
            text.contains("; ... listing truncated at 3 lines (max_lines)\n"),
            "{text}"
        );
    }

    #[test]
    fn zero_caps_still_produce_a_well_formed_listing() {
        for opts in [
            Options {
                max_functions: 0,
                ..Options::default()
            },
            Options {
                max_lines: 0,
                ..Options::default()
            },
        ] {
            let text = render_pe(SAMPLE, &opts);
            assert!(text.ends_with('\n'), "{text:?}");
            assert!(text.starts_with("; ..."), "{text:?}");
        }
    }

    // -- the per-function variant --------------------------------------

    #[test]
    fn render_function_renders_exactly_one_function() {
        let img = pe_with(SAMPLE);
        let image = load(&img).unwrap();
        let program = recover(image.as_ref()).unwrap();
        let opts = Options::default();

        let one = render_function(image.as_ref(), &program, PE_ENTRY, None, &opts);
        assert!(one.contains("sub_140001000:"), "{one}");
        assert!(!one.contains("sub_140001010:"), "{one}");
        // Naming still joins across the whole program.
        assert!(one.contains("-> sub_140001010"), "{one}");

        let missing = render_function(image.as_ref(), &program, 0xdead_beef, None, &opts);
        assert_eq!(missing, "; (no function recovered at 0x00000000deadbeef)\n");
    }

    // -- symbolization -------------------------------------------------

    #[test]
    fn an_image_symbol_names_a_call_target() {
        // The fixture's export directory is not populated, so drive the
        // symbol join through a target the loader does name: a function
        // recovered from a symbol keeps its `cfg` name, which the
        // listing must prefer over `sub_...`.
        let img = pe_with(SAMPLE);
        let image = load(&img).unwrap();
        let mut program = recover(image.as_ref()).unwrap();
        program
            .functions
            .get_mut(&(PE_BASE + 0x1010))
            .unwrap()
            .name = Some("inflate".to_string());

        let text = render(image.as_ref(), &program, None, &Options::default());
        assert!(text.contains("inflate:"), "{text}");
        // With operand text present, the call line names its target as
        // `-> inflate`; the mnemonic itself carries the address.
        assert!(text.contains("call 0x140001010"), "{text}");
        assert!(text.contains("-> inflate"), "{text}");
    }

    /// A loader symbol that is a mangled Rust name is demangled for
    /// display: the raw `_ZN..` form must not reach the listing, and the
    /// readable path must.
    #[test]
    fn a_mangled_symbol_name_is_demangled() {
        let img = pe_with(SAMPLE);
        let image = load(&img).unwrap();
        let mut program = recover(image.as_ref()).unwrap();
        program.functions.get_mut(&PE_ENTRY).unwrap().name =
            Some("_ZN4core3fmt9Formatter3pad17h0123456789abcdefE".to_string());

        let text = render(image.as_ref(), &program, None, &Options::default());
        assert!(text.contains("core::fmt::Formatter::pad"), "{text}");
        assert!(!text.contains("_ZN4core3fmt"), "{text}");
    }

    /// A loader symbol that is an Itanium-mangled C++ name is demangled
    /// for display; the Rust demangler refuses it first, so the C++
    /// rendering (with its parameter list) is what appears.
    #[test]
    fn a_cxx_mangled_symbol_name_is_demangled() {
        let img = pe_with(SAMPLE);
        let image = load(&img).unwrap();
        let mut program = recover(image.as_ref()).unwrap();
        program.functions.get_mut(&PE_ENTRY).unwrap().name =
            Some("_ZNSt6vectorIiSaIiEE9push_backERKi".to_string());

        let text = render(image.as_ref(), &program, None, &Options::default());
        assert!(text.contains("std::vector<int, std::allocator<int>>::push_back(int const&)"), "{text}");
        assert!(!text.contains("_ZNSt6vector"), "{text}");
    }

    #[test]
    fn a_data_reference_into_a_string_shows_the_string() {
        // lea rax, [rip+0x19]  ; -> 0x140001020
        // ret
        let mut img = pe_with(&[0x48, 0x8D, 0x05, 0x19, 0x00, 0x00, 0x00, 0xC3]);
        img[0x220..0x220 + 13].copy_from_slice(b"hello, world\0");

        let image = load(&img).unwrap();
        let program = recover(image.as_ref()).unwrap();
        let text = render(image.as_ref(), &program, None, &Options::default());
        assert!(text.contains("\"hello, world\""), "{text}");
    }

    #[test]
    fn a_string_preview_is_escaped_in_place() {
        let mut img = pe_with(&[0x48, 0x8D, 0x05, 0x19, 0x00, 0x00, 0x00, 0xC3]);
        img[0x220..0x220 + 10].copy_from_slice(b"a\"b\\c/d\te\0");

        let image = load(&img).unwrap();
        let program = recover(image.as_ref()).unwrap();
        let text = render(image.as_ref(), &program, None, &Options::default());
        // The tab is not printable, so extraction ends the run there.
        assert!(text.contains("\"a\\\"b\\\\c/d\""), "{text}");
    }

    /// The renderer names no container format and no instruction set, so
    /// an ELF of AArch64 code must come out in the same shape as the PE
    /// of x86-64 code above — with the ELF symbol table supplying the
    /// names that the PE fixture had to synthesize.
    #[test]
    fn an_aarch64_elf_renders_through_the_same_path() {
        let mut img = crate::elf::tests::synthetic_elf64();
        img[18..20].copy_from_slice(&183u16.to_le_bytes()); // EM_AARCH64
        // .text at file 0x100 == VA 0x40_1000; `main` at +0, `helper`
        // at +0x20 are both function symbols in the fixture.
        for (off, word) in [
            (0x00usize, 0x5400_0040u32), // b.eq +8      -> 0x401008
            (0x04, 0x9400_0007),         // bl  +0x1c    -> 0x401020 (helper)
            (0x08, 0xD65F_03C0),         // ret
            (0x20, 0xD65F_03C0),         // helper: ret
        ] {
            img[0x100 + off..0x100 + off + 4].copy_from_slice(&word.to_le_bytes());
        }

        let image = load(&img).expect("synthetic ELF loads");
        let program = recover(image.as_ref()).expect("aarch64 recovers");
        let text = render(image.as_ref(), &program, None, &Options::default());

        assert!(text.contains("; ======== main @ 0x0000000000401000"), "{text}");
        assert!(text.contains("helper:"), "{text}");
        // Flow symbolization names the target on any ISA; the A64 text
        // renderer supplies the mnemonic and address.
        assert!(text.contains("bl 0x401020"), "{text}");
        assert!(text.contains("-> helper"), "{text}");
        assert!(text.contains("-> loc_401008"), "{text}");
        // A64 is fixed-width, so every byte column holds four bytes.
        assert!(text.contains("  40 00 00 54             "), "{text}");
    }

    /// A function whose entry is not its lowest address: the fixture's
    /// `jmp` backwards makes the entry block the *second* one in address
    /// order, and the name label must follow the entry, not the header.
    #[test]
    fn the_name_label_sits_at_the_entry_not_at_the_top() {
        // 0x140001000  ret               (the out-of-line block)
        // 0x140001001  jmp 0x140001000   (the entry, seeded below)
        let mut img = pe_with(&[0xC3, 0xEB, 0xFD]);
        // Give the image an export naming 0x140001001, so `cfg` seeds a
        // function there rather than only at the entry point.
        img[0x98 + 16..0x98 + 20].copy_from_slice(&0x1001u32.to_le_bytes());

        let image = load(&img).unwrap();
        let program = recover(image.as_ref()).unwrap();
        let text = render(image.as_ref(), &program, None, &Options::default());

        let entry_line = text
            .lines()
            .position(|l| l.starts_with("sub_140001001:"))
            .expect("entry label present");
        let low_line = text
            .lines()
            .position(|l| l.contains("0x0000000140001000"))
            .expect("the lower block is rendered");
        assert!(low_line < entry_line, "label must follow the lower block:\n{text}");
    }

    #[test]
    fn an_import_thunk_is_named_for_its_import() {
        let mut func = Function {
            entry: PE_ENTRY,
            name: None,
            blocks: BTreeMap::new(),
        };
        func.blocks.insert(
            PE_ENTRY,
            BasicBlock {
                start: PE_ENTRY,
                end: PE_ENTRY + 6,
                terminator: Terminator::IndirectJump {
                    import: Some("KERNEL32.dll!ExitProcess".to_string()),
                },
                successors: Vec::new(),
            },
        );
        assert_eq!(thunk_import(&func), Some("KERNEL32.dll!ExitProcess"));
    }

    // -- hostile input -------------------------------------------------

    /// A block whose bytes are not in the file at all. `cfg` will not
    /// build one, but a caller can hand us any `Program`, and a renderer
    /// that panics on one is a renderer that panics on a crafted input.
    #[test]
    fn a_block_outside_the_file_renders_without_panicking() {
        let img = pe_with(SAMPLE);
        let image = load(&img).unwrap();
        let mut program = recover(image.as_ref()).unwrap();

        let entry = 0xdead_0000_u64;
        let mut blocks = BTreeMap::new();
        blocks.insert(
            entry,
            BasicBlock {
                start: entry,
                end: entry + 0x10,
                terminator: Terminator::Undecodable,
                successors: Vec::new(),
            },
        );
        // ...and one that starts just before the end of the file, so the
        // decoder runs off the buffer rather than off the address space.
        let edge = PE_BASE + 0x10FF;
        let mut edge_blocks = BTreeMap::new();
        edge_blocks.insert(
            edge,
            BasicBlock {
                start: edge,
                end: edge + 0x40,
                terminator: Terminator::Truncated,
                successors: Vec::new(),
            },
        );
        program.functions.insert(
            entry,
            Function {
                entry,
                name: None,
                blocks,
            },
        );
        program.functions.insert(
            edge,
            Function {
                entry: edge,
                name: None,
                blocks: edge_blocks,
            },
        );

        let text = render(image.as_ref(), &program, None, &Options::default());
        assert!(text.contains("(no file backing)"), "{text}");
        assert!(text.ends_with('\n'), "{text:?}");
    }

    #[test]
    fn a_wrapping_block_terminates() {
        let img = pe_with(SAMPLE);
        let image = load(&img).unwrap();
        let mut program = recover(image.as_ref()).unwrap();

        let entry = u64::MAX - 3;
        let mut blocks = BTreeMap::new();
        blocks.insert(
            entry,
            BasicBlock {
                start: entry,
                end: u64::MAX,
                terminator: Terminator::Undecodable,
                successors: Vec::new(),
            },
        );
        program.functions.insert(
            entry,
            Function {
                entry,
                name: None,
                blocks,
            },
        );
        let text = render(image.as_ref(), &program, None, &Options::default());
        assert!(text.contains("sub_fffffffffffffffc"), "{text}");
    }

    #[test]
    fn undecodable_bytes_are_reported_not_guessed() {
        // 0x06 is an invalid opcode in 64-bit mode, so `cfg` ends the
        // entry block on it with nothing decoded. The listing must say
        // so rather than showing an empty function.
        let text = render_pe(&[0x06, 0xC3], &Options::default());
        assert!(text.contains("db 06"), "{text}");
        assert!(
            text.contains("; undecodable here: analysis stopped, nothing guessed\n"),
            "{text}"
        );
    }

    // -- helpers -------------------------------------------------------

    #[test]
    fn long_encodings_elide_inside_a_fixed_width_column() {
        let short = hex_bytes(&[0xAA; BYTES_SHOWN]);
        let long = hex_bytes(&[0xAA; 15]);
        assert_eq!(short.chars().count(), BYTES_COL);
        assert_eq!(long.chars().count(), BYTES_COL);
        assert!(long.ends_with(".."), "{long}");
        assert_eq!(hex_bytes(&[]), "");
    }

    #[test]
    fn string_previews_are_escaped_and_truncated() {
        assert_eq!(escape("a\"b\\c", 40), "a\\\"b\\\\c");
        assert_eq!(escape("a\nb\tc\r", 40), "a\\nb\\tc\\r");
        assert_eq!(escape("\u{1}", 40), "\\x01");
        assert_eq!(escape("abcdef", 3), "abc...");
        assert_eq!(escape("", 40), "");
    }

    #[test]
    fn every_address_is_the_same_fixed_width() {
        assert_eq!(hex_va(0), "0x0000000000000000");
        assert_eq!(hex_va(u64::MAX), "0xffffffffffffffff");
        assert_eq!(hex_va(0x401000).len(), hex_va(0).len());
    }
}
