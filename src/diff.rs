//! Binary diffing — matching functions across two versions of a program.
//!
//! Built directly on [`crate::anchor`]: a function's content-relative
//! anchor (raw-bytes hash, rebase-invariant shape hash) already defines
//! identity across rebases, so diffing two binaries is resolving every
//! old function's anchor against the new binary's [`AnchorIndex`] and
//! bucketing the resolutions: identical, moved, modified-but-same-shape,
//! added, removed.
//!
//! # No new heuristics live here
//!
//! "Is this the same function?" is exactly the question the anchor layer
//! answers for the annotation database, and answering it twice — once
//! for annotations, differently for diffing — is how a tool starts
//! telling an analyst two incompatible stories about one binary. So this
//! module contributes **no identity heuristics of its own**. It is a
//! deterministic orchestration: build one [`AnchorIndex`] over the new
//! program, resolve each old function against it, and translate the
//! [`Resolution`] into a bucket. Every judgement about sameness is the
//! anchor layer's, and improving matching means improving *that* layer.
//!
//! # The buckets, and what each one is evidence of
//!
//! - [`MatchKind::Unchanged`] — an exact raw-byte match at the same
//!   virtual address. The function is byte-for-byte what it was.
//! - [`MatchKind::Moved`] — an exact raw-byte match at a *different*
//!   address. The code is identical; the linker put it elsewhere.
//! - [`MatchKind::Modified`] — a shape match without a byte match: the
//!   instruction stream has the same lengths and control-flow classes,
//!   but some bytes differ. That is what a patched constant, a retargeted
//!   call, or a rebase of referenced data looks like.
//! - [`MatchKind::Uncertain`] — the anchor layer resolved only by
//!   absolute address, or to several equally good candidates, or to a
//!   function an earlier old function already claimed. The tier is
//!   reported ([`Uncertainty`]) and nothing further is guessed: a wrong
//!   confident answer costs an analyst more than an honest "unsure".
//! - Removed and added — an old function nothing in the new program
//!   matched, and a new function no old function claimed.
//!
//! # One old function per new function
//!
//! Claims are exclusive: the first old function (in entry-VA order) to
//! resolve to a given new entry keeps the match, and any later old
//! function resolving there degrades to
//! [`Uncertainty::Contested`]. Without that rule, two copies of an
//! inlined helper in the old build would both "match" the single
//! surviving copy and the diff would double-count it. Because old
//! functions are walked in [`BTreeMap`] order, which claim wins is a
//! function of the addresses alone — never of iteration luck.
//!
//! # Names are carried, never written
//!
//! When a matched pair has a name on the old side and none on the new
//! side, the diff *records* the carry-over in [`Pair::carried_name`].
//! Neither input [`cfg::Program`] is mutated: a diff is an observation,
//! and the caller decides whether to act on it. Names are reported
//! exactly as stored — display-time demangling belongs to
//! [`crate::listing`], which owns presentation.
//!
//! # Determinism
//!
//! Every collection here is a [`BTreeMap`], candidate lists are sorted,
//! and [`render`] walks buckets in a fixed order. The same two binaries
//! always produce a byte-identical report, which is what makes the
//! report itself diffable and committable.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use crate::anchor::{self, AnchorIndex, Resolution};
use crate::cfg;
use crate::model::Image;

/// Width of a rendered virtual address: `0x` plus 16 hex digits, the
/// same fixed column the listing uses so the two outputs line up.
const VA_WIDTH: usize = 18;

/// Ambiguous candidates printed before eliding. A hostile pair of
/// binaries can produce thousands of same-shape functions; a report is
/// no use to a human if one line of it runs for a page.
const CANDIDATES_SHOWN: usize = 8;

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// How an old function relates to its counterpart in the new program.
///
/// Ordered by confidence, and derived purely from the
/// [`Resolution`] the anchor index returned — see the module docs for
/// what each bucket is evidence of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchKind {
    /// Exact raw-byte match at the same entry VA: byte-for-byte the same
    /// function in the same place.
    Unchanged,
    /// Exact raw-byte match at a different entry VA: identical code that
    /// the linker relocated.
    Moved,
    /// Shape match without a byte match: same instruction lengths and
    /// control-flow classes, different bytes.
    Modified,
    /// The anchor layer could not name one counterpart confidently. The
    /// payload reports which tier fell short.
    Uncertain(Uncertainty),
}

impl MatchKind {
    /// Stable lowercase label — the bucket heading used by [`render`]
    /// and a convenient key for a caller's own output.
    pub fn label(&self) -> &'static str {
        match self {
            MatchKind::Unchanged => "unchanged",
            MatchKind::Moved => "moved",
            MatchKind::Modified => "modified",
            MatchKind::Uncertain(_) => "uncertain",
        }
    }

    /// Whether this is a confident match — the pair the diff is willing
    /// to carry a name across. False for every [`MatchKind::Uncertain`].
    pub fn is_confident(&self) -> bool {
        !matches!(self, MatchKind::Uncertain(_))
    }
}

/// Why a match is uncertain.
///
/// Each variant corresponds to a weak or contested
/// [`Resolution`]; none of them is upgraded by guesswork here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Uncertainty {
    /// Neither the bytes nor the shape matched, but a function still
    /// lives at the old entry VA ([`Resolution::Absolute`]). The address
    /// is the only evidence, and addresses are the first thing a rebuild
    /// invalidates.
    AddressOnly,
    /// Several new functions share the shape and instruction count and
    /// the old entry VA picked none of them out
    /// ([`Resolution::Ambiguous`]). Candidate entry VAs, sorted.
    Ambiguous(Vec<u64>),
    /// The resolution named a new function that an earlier old function
    /// (carried here, by entry VA) already claimed.
    Contested(u64),
}

impl Uncertainty {
    /// A short explanation for the report, e.g. `address-only match`.
    /// Candidate lists are elided past [`CANDIDATES_SHOWN`] so one line
    /// of output can never run away.
    pub fn note(&self) -> String {
        match self {
            Uncertainty::AddressOnly => "address-only match".to_string(),
            Uncertainty::Ambiguous(candidates) => {
                let shown: Vec<String> = candidates
                    .iter()
                    .take(CANDIDATES_SHOWN)
                    .map(|&va| hex_va(va))
                    .collect();
                let rest = candidates.len().saturating_sub(shown.len());
                let mut note = format!("ambiguous: {}", shown.join(", "));
                if rest > 0 {
                    note.push_str(&format!(", ... ({rest} more)"));
                }
                note
            }
            Uncertainty::Contested(holder) => {
                format!("contested; claimed by {}", hex_va(*holder))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The diff
// ---------------------------------------------------------------------------

/// One old function and the new function it resolved to.
///
/// Present for every old function that resolved at all, uncertain
/// matches included; an old function that resolved to nothing is in
/// [`Diff::removed`] instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pair {
    /// Entry VA in the old program.
    pub old_entry: u64,
    /// Entry VA in the new program, when the resolution named exactly
    /// one function. `None` only for [`Uncertainty::Ambiguous`], which
    /// names a candidate set rather than a function.
    pub new_entry: Option<u64>,
    /// How the two were matched.
    pub kind: MatchKind,
    /// The old function's name, as stored.
    pub old_name: Option<String>,
    /// The new function's name, as stored.
    pub new_name: Option<String>,
    /// The old name, recorded here because the match is confident and
    /// the new function has no name of its own. A record of what a
    /// caller *may* apply — nothing is written to either program.
    pub carried_name: Option<String>,
}

impl Pair {
    /// The name to display for this pair: the new function's own name if
    /// it has one, otherwise the carried-over old name, otherwise the
    /// old name (which an uncertain match never carries).
    pub fn name(&self) -> Option<&str> {
        self.new_name
            .as_deref()
            .or(self.carried_name.as_deref())
            .or(self.old_name.as_deref())
    }
}

/// The result of diffing two programs: every old function bucketed, plus
/// the new functions no old function claimed.
///
/// All collections are keyed by entry VA and iterate in address order,
/// so consuming a `Diff` is deterministic however it is walked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diff {
    /// Matched old functions keyed by old entry VA — confident and
    /// uncertain alike; [`Pair::kind`] distinguishes them.
    pub pairs: BTreeMap<u64, Pair>,
    /// New functions claimed by no old function, keyed by entry VA, with
    /// the name each carries in the new program.
    pub added: BTreeMap<u64, Option<String>>,
    /// Old functions nothing in the new program matched, keyed by entry
    /// VA, with the name each carried in the old program.
    pub removed: BTreeMap<u64, Option<String>>,
    /// Functions in the old program.
    old_functions: usize,
    /// Functions in the new program.
    new_functions: usize,
}

/// Per-bucket totals, for a caller that wants the summary without
/// walking the pairs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Exact matches at the same address.
    pub unchanged: usize,
    /// Exact matches at a different address.
    pub moved: usize,
    /// Shape matches without a byte match.
    pub modified: usize,
    /// Matches the anchor layer could not make confidently.
    pub uncertain: usize,
    /// New functions no old function claimed.
    pub added: usize,
    /// Old functions nothing matched.
    pub removed: usize,
}

impl Diff {
    /// Functions in the old program.
    pub fn old_functions(&self) -> usize {
        self.old_functions
    }

    /// Functions in the new program.
    pub fn new_functions(&self) -> usize {
        self.new_functions
    }

    /// The pair for an old function, if it resolved to anything.
    pub fn pair(&self, old_entry: u64) -> Option<&Pair> {
        self.pairs.get(&old_entry)
    }

    /// Every pair whose [`MatchKind::label`] is `label`, in old-entry
    /// order.
    pub fn of_kind(&self, label: &str) -> Vec<&Pair> {
        self.pairs
            .values()
            .filter(|pair| pair.kind.label() == label)
            .collect()
    }

    /// Per-bucket totals.
    pub fn counts(&self) -> Counts {
        let mut counts = Counts {
            added: self.added.len(),
            removed: self.removed.len(),
            ..Counts::default()
        };
        for pair in self.pairs.values() {
            let slot = match pair.kind {
                MatchKind::Unchanged => &mut counts.unchanged,
                MatchKind::Moved => &mut counts.moved,
                MatchKind::Modified => &mut counts.modified,
                MatchKind::Uncertain(_) => &mut counts.uncertain,
            };
            *slot = slot.saturating_add(1);
        }
        counts
    }

    /// Whether the two programs are the same at function granularity:
    /// nothing added, nothing removed, and every function matched
    /// byte-for-byte at its old address.
    ///
    /// Two empty programs are trivially identical.
    pub fn is_identical(&self) -> bool {
        self.added.is_empty()
            && self.removed.is_empty()
            && self
                .pairs
                .values()
                .all(|pair| pair.kind == MatchKind::Unchanged)
    }
}

// ---------------------------------------------------------------------------
// Diffing
// ---------------------------------------------------------------------------

/// Diff the old program against the new one.
///
/// Walks `old_program.functions` in entry-VA order, resolving each
/// function's [`anchor::Anchor`] against an [`AnchorIndex`] built over
/// `new_program`, and buckets the outcome. Claims on new functions are
/// exclusive and first-come in that same order, so the result depends
/// only on the two programs.
///
/// Never panics: an anchor is always producible (see
/// [`anchor::of_function`]), an unresolvable function is simply
/// *removed*, and mismatched architectures or formats between the two
/// images just make everything resolve poorly.
pub fn diff(
    old_image: &dyn Image,
    old_program: &cfg::Program,
    new_image: &dyn Image,
    new_program: &cfg::Program,
) -> Diff {
    let index = AnchorIndex::build(new_image, new_program);

    // New entry VA -> the old entry VA that claimed it. Exclusive: an
    // entry is claimed by the first old function, in address order, that
    // resolves to it.
    let mut claims: BTreeMap<u64, u64> = BTreeMap::new();
    let mut pairs: BTreeMap<u64, Pair> = BTreeMap::new();
    let mut removed: BTreeMap<u64, Option<String>> = BTreeMap::new();

    for (&old_entry, old_func) in &old_program.functions {
        let anchor = anchor::of_function(old_image, old_func);
        let (new_entry, kind) = match index.resolve(&anchor) {
            Resolution::Exact(va) if va == old_entry => (Some(va), MatchKind::Unchanged),
            Resolution::Exact(va) => (Some(va), MatchKind::Moved),
            Resolution::Shape(va) => (Some(va), MatchKind::Modified),
            Resolution::Absolute(va) => (Some(va), MatchKind::Uncertain(Uncertainty::AddressOnly)),
            Resolution::Ambiguous(mut candidates) => {
                // Sorted and deduplicated so the rendered note is a
                // function of the candidate *set*, not of how the index
                // happened to accumulate it.
                candidates.sort_unstable();
                candidates.dedup();
                (
                    None,
                    MatchKind::Uncertain(Uncertainty::Ambiguous(candidates)),
                )
            }
            Resolution::Unresolved => {
                removed.insert(old_entry, old_func.name.clone());
                continue;
            }
        };

        // Enforce the exclusive claim. A resolution that names no single
        // function (ambiguous) claims nothing, so its candidates stay
        // available to other old functions and to the added bucket.
        let kind = match new_entry {
            Some(va) => match claims.entry(va) {
                Entry::Occupied(held) => MatchKind::Uncertain(Uncertainty::Contested(*held.get())),
                Entry::Vacant(slot) => {
                    slot.insert(old_entry);
                    kind
                }
            },
            None => kind,
        };

        let new_name = new_entry
            .and_then(|va| new_program.functions.get(&va))
            .and_then(|func| func.name.clone());
        // A name is carried only across a confident match, and only onto
        // a function that has no name of its own.
        let carried_name = match (&old_func.name, &new_name) {
            (Some(name), None) if kind.is_confident() => Some(name.clone()),
            _ => None,
        };

        pairs.insert(
            old_entry,
            Pair {
                old_entry,
                new_entry,
                kind,
                old_name: old_func.name.clone(),
                new_name,
                carried_name,
            },
        );
    }

    let added = new_program
        .functions
        .iter()
        .filter(|(va, _)| !claims.contains_key(va))
        .map(|(&va, func)| (va, func.name.clone()))
        .collect();

    Diff {
        pairs,
        added,
        removed,
        old_functions: old_program.functions.len(),
        new_functions: new_program.functions.len(),
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Render a [`Diff`] as a readable, diffable report.
///
/// A summary line, then one section per non-empty bucket in a fixed
/// order (unchanged, moved, modified, uncertain, added, removed), each
/// line `{old_va} -> {new_va} name`. Names print exactly as stored;
/// demangling for display is [`crate::listing`]'s job. Always ends with
/// a newline, so the output is safe to print verbatim or concatenate.
pub fn render(diff: &Diff) -> String {
    let counts = diff.counts();
    let mut lines: Vec<String> = vec![
        format!(
            "; ======== aletheia diff: {} old functions, {} new functions ========",
            diff.old_functions(),
            diff.new_functions()
        ),
        format!(
            "; unchanged {}  moved {}  modified {}  uncertain {}  added {}  removed {}",
            counts.unchanged,
            counts.moved,
            counts.modified,
            counts.uncertain,
            counts.added,
            counts.removed
        ),
    ];
    if diff.is_identical() {
        lines.push("; no differences".to_string());
    }

    for label in ["unchanged", "moved", "modified", "uncertain"] {
        let group = diff.of_kind(label);
        if group.is_empty() {
            continue;
        }
        lines.push(String::new());
        lines.push(format!("; -- {label} ({}) --", group.len()));
        lines.extend(group.into_iter().map(pair_line));
    }

    for (label, entries) in [("added", &diff.added), ("removed", &diff.removed)] {
        if entries.is_empty() {
            continue;
        }
        lines.push(String::new());
        lines.push(format!("; -- {label} ({}) --", entries.len()));
        lines.extend(entries.iter().map(|(&va, name)| entry_line(va, name)));
    }

    let mut out = lines.join("\n");
    out.push('\n');
    out
}

/// One `{old_va} -> {new_va} name` line, with the carry-over marker and
/// the uncertainty note when either applies.
fn pair_line(pair: &Pair) -> String {
    let target = match pair.new_entry {
        Some(va) => hex_va(va),
        // An ambiguous match names a set, not a function; the candidates
        // are in the trailing note.
        None => format!("{:>VA_WIDTH$}", "?"),
    };
    let mut line = format!("{} -> {}", hex_va(pair.old_entry), target);
    if let Some(name) = pair.name() {
        line.push_str("  ");
        line.push_str(name);
    }
    if pair.carried_name.is_some() {
        line.push_str(" (carried)");
    }
    if let MatchKind::Uncertain(why) = &pair.kind {
        line.push_str("  ; ");
        line.push_str(&why.note());
    }
    line
}

/// One added/removed line: an entry VA and its name, if it has one.
fn entry_line(va: u64, name: &Option<String>) -> String {
    match name {
        Some(name) => format!("{}  {}", hex_va(va), name),
        None => hex_va(va),
    }
}

/// A virtual address in the crate's fixed-width form.
fn hex_va(va: u64) -> String {
    format!("0x{va:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::recover;
    use crate::elf::tests::synthetic_elf64;
    use crate::model::load;
    use crate::pe::tests::synthetic_pe64;

    /// Image base of the synthetic PE fixture.
    const PE_BASE: u64 = 0x1_4000_0000;
    /// Entry VA of the synthetic PE fixture (RVA 0x1000, file 0x200).
    const PE_ENTRY: u64 = PE_BASE + 0x1000;

    /// The synthetic PE with `code` written at RVA `entry_rva` and the
    /// image's entry point pointed there.
    ///
    /// The base fixture aims its import directory at RVA 0x1010, inside
    /// `.text`, for its own import tests; a diff fixture wants that range
    /// to be code, so the directory (index 1) is cleared.
    fn pe_at(entry_rva: u32, code: &[u8]) -> Vec<u8> {
        let mut img = synthetic_pe64();
        let opt = 0x80 + 4 + 20;
        let dirs = opt + 112;
        img[dirs + 8..dirs + 16].fill(0);
        img[opt + 16..opt + 20].copy_from_slice(&entry_rva.to_le_bytes());
        let off = 0x200 + (entry_rva as usize - 0x1000);
        img[off..off + code.len()].copy_from_slice(code);
        img
    }

    /// The synthetic PE with `code` at the entry point (RVA 0x1000).
    fn pe_with(code: &[u8]) -> Vec<u8> {
        pe_at(0x1000, code)
    }

    /// Recover a program from an image, keeping the image alive for the
    /// caller (anchoring needs the bytes, not just the CFG).
    fn recovered(img: &[u8]) -> cfg::Program {
        let image = load(img).expect("synthetic image loads");
        recover(image.as_ref()).expect("x86-64 recovers")
    }

    /// Diff two raw images end to end.
    fn diff_images(old: &[u8], new: &[u8]) -> Diff {
        let old_image = load(old).expect("old image loads");
        let new_image = load(new).expect("new image loads");
        let old_program = recover(old_image.as_ref()).expect("old recovers");
        let new_program = recover(new_image.as_ref()).expect("new recovers");
        diff(
            old_image.as_ref(),
            &old_program,
            new_image.as_ref(),
            &new_program,
        )
    }

    /// A program with no functions at all.
    fn empty_program() -> cfg::Program {
        cfg::Program {
            functions: BTreeMap::new(),
            call_graph: BTreeMap::new(),
            stats: cfg::Stats::default(),
        }
    }

    /// `mov eax, 1; ret` — one 5-byte sequential instruction and a
    /// return, so a changed immediate keeps the shape.
    const MOV_1: &[u8] = &[0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3];
    /// The same function with a patched immediate.
    const MOV_2: &[u8] = &[0xB8, 0x02, 0x00, 0x00, 0x00, 0xC3];

    /// `call 0x1010; ret` at RVA 0x1000, with `nop; ret` at RVA 0x1010:
    /// two functions, the second discovered through the call.
    const CALLS_1010: &[u8] = &[
        0xE8, 0x0B, 0x00, 0x00, 0x00, // 0x1000: call 0x1010
        0xC3, // 0x1005: ret
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad
        0x90, // 0x1010: nop
        0xC3, // 0x1011: ret
    ];

    // -- the confident buckets -----------------------------------------

    #[test]
    fn an_identical_binary_diffs_as_all_unchanged() {
        let img = pe_with(CALLS_1010);
        let d = diff_images(&img, &img);

        assert!(d.is_identical());
        assert_eq!(d.old_functions(), 2);
        assert_eq!(d.new_functions(), 2);
        assert_eq!(
            d.counts(),
            Counts {
                unchanged: 2,
                ..Counts::default()
            }
        );
        for (&old_entry, pair) in &d.pairs {
            assert_eq!(pair.kind, MatchKind::Unchanged);
            assert_eq!(pair.new_entry, Some(old_entry));
        }
    }

    #[test]
    fn a_function_at_a_new_address_is_moved() {
        // The same single function, placed at RVA 0x1000 in one build and
        // at RVA 0x1010 in the other. Identical bytes, different entry.
        let d = diff_images(&pe_at(0x1000, MOV_1), &pe_at(0x1010, MOV_1));

        let pair = d.pair(PE_ENTRY).expect("the old entry resolved");
        assert_eq!(pair.kind, MatchKind::Moved);
        assert_eq!(pair.new_entry, Some(PE_ENTRY + 0x10));
        assert!(!d.is_identical());
        assert_eq!(
            d.counts(),
            Counts {
                moved: 1,
                ..Counts::default()
            }
        );
    }

    #[test]
    fn a_changed_immediate_is_modified_not_removed() {
        let d = diff_images(&pe_with(MOV_1), &pe_with(MOV_2));

        let pair = d.pair(PE_ENTRY).expect("the old entry resolved");
        assert_eq!(pair.kind, MatchKind::Modified);
        assert_eq!(pair.new_entry, Some(PE_ENTRY));
        assert_eq!(
            d.counts(),
            Counts {
                modified: 1,
                ..Counts::default()
            }
        );
    }

    // -- added and removed ---------------------------------------------

    /// The old build calls `nop; ret` at 0x1010; the new build calls a
    /// different helper, `xor eax, eax; ret`, at 0x1020. The callee at
    /// 0x1010 is gone, the one at 0x1020 is new, and the caller — same
    /// shape, different call target — is modified.
    const CALLS_1020: &[u8] = &[
        0xE8, 0x1B, 0x00, 0x00, 0x00, // 0x1000: call 0x1020
        0xC3, // 0x1005: ret
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad
        0x31, 0xC0, // 0x1020: xor eax, eax
        0xC3, // 0x1022: ret
    ];

    #[test]
    fn a_deleted_function_is_removed() {
        let d = diff_images(&pe_with(CALLS_1010), &pe_with(CALLS_1020));

        assert_eq!(
            d.removed.keys().copied().collect::<Vec<_>>(),
            [PE_ENTRY + 0x10]
        );
        assert!(d.pair(PE_ENTRY + 0x10).is_none());
        // The caller kept its shape, so it is modified rather than lost.
        assert_eq!(
            d.pair(PE_ENTRY).map(|p| p.kind.clone()),
            Some(MatchKind::Modified)
        );
    }

    #[test]
    fn a_function_no_old_function_claims_is_added() {
        let d = diff_images(&pe_with(CALLS_1010), &pe_with(CALLS_1020));

        assert_eq!(
            d.added.keys().copied().collect::<Vec<_>>(),
            [PE_ENTRY + 0x20]
        );
        assert_eq!(
            d.counts(),
            Counts {
                modified: 1,
                added: 1,
                removed: 1,
                ..Counts::default()
            }
        );
    }

    #[test]
    fn an_unchanged_function_is_never_also_added() {
        let img = pe_with(CALLS_1010);
        let d = diff_images(&img, &img);
        assert!(d.added.is_empty(), "a claimed function is not added");
    }

    // -- exclusive claims ----------------------------------------------

    /// Two byte-identical helpers, at 0x1010 and 0x1020, both called from
    /// the entry.
    const TWO_COPIES: &[u8] = &[
        0xE8, 0x0B, 0x00, 0x00, 0x00, // 0x1000: call 0x1010
        0xE8, 0x16, 0x00, 0x00, 0x00, // 0x1005: call 0x1020
        0xC3, // 0x100a: ret
        0x00, 0x00, 0x00, 0x00, 0x00, // pad
        0x90, 0xC3, // 0x1010: nop; ret
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad
        0x00, 0x00, 0x00, 0x00, // pad
        0x90, 0xC3, // 0x1020: nop; ret
    ];

    /// The same program with only the first copy of the helper left.
    const ONE_COPY: &[u8] = CALLS_1010;

    #[test]
    fn two_identical_bodies_let_the_lower_address_claim_the_match() {
        let d = diff_images(&pe_with(TWO_COPIES), &pe_with(ONE_COPY));

        // The lower old entry wins the single surviving copy...
        let first = d.pair(PE_ENTRY + 0x10).expect("first copy resolved");
        assert_eq!(first.kind, MatchKind::Unchanged);
        assert_eq!(first.new_entry, Some(PE_ENTRY + 0x10));

        // ...and the later one is reported as contested, not as a second
        // match on the same function.
        let second = d.pair(PE_ENTRY + 0x20).expect("second copy resolved");
        assert_eq!(
            second.kind,
            MatchKind::Uncertain(Uncertainty::Contested(PE_ENTRY + 0x10))
        );
        assert_eq!(second.new_entry, Some(PE_ENTRY + 0x10));
        assert!(d.added.is_empty());
    }

    #[test]
    fn a_contested_claim_is_stable_across_runs() {
        let old = pe_with(TWO_COPIES);
        let new = pe_with(ONE_COPY);
        assert_eq!(diff_images(&old, &new), diff_images(&old, &new));
        assert_eq!(
            render(&diff_images(&old, &new)),
            render(&diff_images(&old, &new))
        );
    }

    #[test]
    fn an_address_only_match_is_uncertain() {
        // Two unrelated bodies at the same entry: no byte match, no shape
        // match, but the address still names a function.
        let d = diff_images(&pe_with(MOV_1), &pe_with(&[0x90, 0x90, 0x90, 0xC3]));

        let pair = d.pair(PE_ENTRY).expect("the old entry resolved");
        assert_eq!(
            pair.kind,
            MatchKind::Uncertain(Uncertainty::AddressOnly),
            "address-only evidence must not be reported as a match"
        );
        assert!(pair.carried_name.is_none());
    }

    /// `mov eax, eax; ret` — a two-byte sequential instruction and a
    /// return, the same *shape* as `xor eax, eax; ret` and as
    /// `inc eax; ret`, but none of their bytes.
    const MOV_SELF: &[u8] = &[0x89, 0xC0, 0xC3];

    /// A build whose entry calls two helpers that share that shape:
    /// `xor eax, eax; ret` at 0x1010 and `inc eax; ret` at 0x1020.
    const TWO_SHAPES: &[u8] = &[
        0xE8, 0x0B, 0x00, 0x00, 0x00, // 0x1000: call 0x1010
        0xE8, 0x16, 0x00, 0x00, 0x00, // 0x1005: call 0x1020
        0xC3, // 0x100a: ret
        0x00, 0x00, 0x00, 0x00, 0x00, // pad
        0x31, 0xC0, 0xC3, // 0x1010: xor eax, eax; ret
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // pad
        0x00, 0x00, 0x00, // pad
        0xFF, 0xC0, 0xC3, // 0x1020: inc eax; ret
    ];

    #[test]
    fn a_shape_shared_by_several_new_functions_stays_ambiguous() {
        let d = diff_images(&pe_with(MOV_SELF), &pe_with(TWO_SHAPES));

        let pair = d.pair(PE_ENTRY).expect("the old entry resolved");
        assert_eq!(
            pair.kind,
            MatchKind::Uncertain(Uncertainty::Ambiguous(vec![
                PE_ENTRY + 0x10,
                PE_ENTRY + 0x20
            ]))
        );
        // An ambiguous resolution names no function, so it claims none:
        // both candidates remain available and are reported as added.
        assert_eq!(pair.new_entry, None);
        assert_eq!(
            d.added.keys().copied().collect::<Vec<_>>(),
            [PE_ENTRY, PE_ENTRY + 0x10, PE_ENTRY + 0x20]
        );

        let text = render(&d);
        assert!(
            text.contains(
                "0x0000000140001000 ->                  ?  \
                 ; ambiguous: 0x0000000140001010, 0x0000000140001020"
            ),
            "ambiguous line names its candidates:\n{text}"
        );
    }

    // -- name carry-over -----------------------------------------------

    /// Recover `img` and name the function at `entry`, standing in for a
    /// build whose symbols survived.
    fn named(img: &[u8], entry: u64, name: &str) -> cfg::Program {
        let mut program = recovered(img);
        if let Some(func) = program.functions.get_mut(&entry) {
            func.name = Some(name.to_string());
        }
        program
    }

    #[test]
    fn a_name_is_carried_onto_an_unnamed_new_function() {
        let old_img = pe_at(0x1000, MOV_1);
        let new_img = pe_at(0x1010, MOV_1);
        let old_image = load(&old_img).expect("old loads");
        let new_image = load(&new_img).expect("new loads");
        let old_program = named(&old_img, PE_ENTRY, "decode_frame");
        let new_program = recovered(&new_img);

        let d = diff(
            old_image.as_ref(),
            &old_program,
            new_image.as_ref(),
            &new_program,
        );
        let pair = d.pair(PE_ENTRY).expect("the old entry resolved");
        assert_eq!(pair.kind, MatchKind::Moved);
        assert_eq!(pair.old_name.as_deref(), Some("decode_frame"));
        assert_eq!(pair.new_name, None);
        assert_eq!(pair.carried_name.as_deref(), Some("decode_frame"));
        assert_eq!(pair.name(), Some("decode_frame"));
        // The inputs are observed, never rewritten.
        assert_eq!(new_program.functions[&(PE_ENTRY + 0x10)].name, None);
    }

    #[test]
    fn a_name_the_new_build_already_has_is_not_overwritten() {
        let img = pe_with(MOV_1);
        let image = load(&img).expect("image loads");
        let old_program = named(&img, PE_ENTRY, "old_name");
        let new_program = named(&img, PE_ENTRY, "new_name");

        let d = diff(image.as_ref(), &old_program, image.as_ref(), &new_program);
        let pair = d.pair(PE_ENTRY).expect("the old entry resolved");
        assert_eq!(pair.carried_name, None);
        assert_eq!(pair.name(), Some("new_name"));
    }

    #[test]
    fn a_removed_function_keeps_its_old_name() {
        let old_img = pe_with(CALLS_1010);
        let new_img = pe_with(CALLS_1020);
        let old_image = load(&old_img).expect("old loads");
        let new_image = load(&new_img).expect("new loads");
        let old_program = named(&old_img, PE_ENTRY + 0x10, "helper");
        let new_program = recovered(&new_img);

        let d = diff(
            old_image.as_ref(),
            &old_program,
            new_image.as_ref(),
            &new_program,
        );
        assert_eq!(
            d.removed.get(&(PE_ENTRY + 0x10)),
            Some(&Some("helper".to_string()))
        );
    }

    // -- degenerate inputs ---------------------------------------------

    #[test]
    fn an_empty_new_program_removes_everything() {
        let img = pe_with(CALLS_1010);
        let image = load(&img).expect("image loads");
        let old_program = recovered(&img);
        let empty = empty_program();

        let d = diff(image.as_ref(), &old_program, image.as_ref(), &empty);
        assert_eq!(d.removed.len(), 2);
        assert!(d.pairs.is_empty() && d.added.is_empty());
        assert!(!d.is_identical());
        assert_eq!(d.new_functions(), 0);
    }

    #[test]
    fn an_empty_old_program_adds_everything() {
        let img = pe_with(CALLS_1010);
        let image = load(&img).expect("image loads");
        let new_program = recovered(&img);
        let empty = empty_program();

        let d = diff(image.as_ref(), &empty, image.as_ref(), &new_program);
        assert_eq!(d.added.len(), 2);
        assert!(d.pairs.is_empty() && d.removed.is_empty());
        assert_eq!(d.old_functions(), 0);
    }

    #[test]
    fn two_empty_programs_are_trivially_identical() {
        let img = pe_with(MOV_1);
        let image = load(&img).expect("image loads");
        let empty = empty_program();

        let d = diff(image.as_ref(), &empty, image.as_ref(), &empty);
        assert!(d.is_identical());
        assert_eq!(d.counts(), Counts::default());
        assert!(render(&d).ends_with("; no differences\n"));
    }

    #[test]
    fn unrelated_binaries_of_different_formats_never_panic() {
        // A PE against an ELF: different loader, different base, nothing
        // in common. Everything should fall out as removed and added
        // without a panic anywhere on the path.
        let pe = pe_with(CALLS_1010);
        let elf = synthetic_elf64();
        let d = diff_images(&pe, &elf);
        assert!(!d.removed.is_empty());
        assert!(!render(&d).is_empty());

        let back = diff_images(&elf, &pe);
        assert!(!render(&back).is_empty());
    }

    // -- the report ----------------------------------------------------

    /// The whole report of a known pair of builds, byte for byte.
    ///
    /// A golden-in-source assertion is the point: the report is meant to
    /// be committed and `diff`ed, so any change to its spacing, ordering,
    /// or notes has to be made deliberately, because it shows up here.
    #[test]
    fn the_report_lists_every_bucket_in_address_order() {
        // The caller keeps its shape but calls a different helper; the
        // old helper is gone and a new one appears.
        let expected = "\
; ======== aletheia diff: 2 old functions, 2 new functions ========
; unchanged 0  moved 0  modified 1  uncertain 0  added 1  removed 1

; -- modified (1) --
0x0000000140001000 -> 0x0000000140001000

; -- added (1) --
0x0000000140001020

; -- removed (1) --
0x0000000140001010
";
        let text = render(&diff_images(&pe_with(CALLS_1010), &pe_with(CALLS_1020)));
        assert_eq!(text, expected);
    }

    #[test]
    fn the_report_explains_a_contested_claim() {
        // The entry function loses a call, so it resolves by address
        // alone; the two identical helpers contest the surviving copy.
        let expected = "\
; ======== aletheia diff: 3 old functions, 2 new functions ========
; unchanged 1  moved 0  modified 0  uncertain 2  added 0  removed 0

; -- unchanged (1) --
0x0000000140001010 -> 0x0000000140001010

; -- uncertain (2) --
0x0000000140001000 -> 0x0000000140001000  ; address-only match
0x0000000140001020 -> 0x0000000140001010  ; contested; claimed by 0x0000000140001010
";
        let text = render(&diff_images(&pe_with(TWO_COPIES), &pe_with(ONE_COPY)));
        assert_eq!(text, expected);
    }

    #[test]
    fn the_report_marks_a_carried_name() {
        let old_img = pe_at(0x1000, MOV_1);
        let new_img = pe_at(0x1010, MOV_1);
        let old_image = load(&old_img).expect("old loads");
        let new_image = load(&new_img).expect("new loads");
        let old_program = named(&old_img, PE_ENTRY, "decode_frame");
        let new_program = recovered(&new_img);

        let text = render(&diff(
            old_image.as_ref(),
            &old_program,
            new_image.as_ref(),
            &new_program,
        ));
        assert!(
            text.contains("0x0000000140001000 -> 0x0000000140001010  decode_frame (carried)"),
            "carried names are marked as such:\n{text}"
        );
    }

    #[test]
    fn the_report_always_ends_with_a_newline() {
        for (old, new) in [
            (pe_with(MOV_1), pe_with(MOV_1)),
            (pe_with(CALLS_1010), pe_with(CALLS_1020)),
            (pe_with(TWO_COPIES), pe_with(ONE_COPY)),
        ] {
            let text = render(&diff_images(&old, &new));
            assert!(text.ends_with('\n'), "report must end with a newline");
            assert!(!text.contains("\n\n\n"), "no runs of blank lines");
        }
    }

    #[test]
    fn an_ambiguous_note_elides_a_runaway_candidate_list() {
        let many: Vec<u64> = (0..40).map(|i| 0x1000 + i * 0x10).collect();
        let note = Uncertainty::Ambiguous(many).note();
        assert!(note.starts_with("ambiguous: 0x0000000000001000, "));
        assert!(note.ends_with(", ... (32 more)"), "note was: {note}");
    }
}
