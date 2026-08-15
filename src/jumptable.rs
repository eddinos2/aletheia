//! Switch/jump-table recovery: turning indirect jumps back into concrete
//! target sets.
//!
//! [`crate::cfg`] deliberately gives an indirect jump no successors — it
//! never guesses. Most indirect jumps in compiled code are not really
//! unknown, though: they are `switch` dispatches through a table the
//! compiler emitted right next to the code, reached by a fixed
//! instruction idiom. This module recognizes a small, closed set of those
//! idioms *exactly*, reads the table out of the mapped image, and hands
//! back the resolved edges. Everything it does not recognize is left
//! unresolved.
//!
//! # ISA use
//!
//! Jump-table recognition is inherently instruction- and operand-level
//! work, so unlike [`crate::cfg`] this module reaches past the
//! [`crate::model::Decoder`] flow abstraction into the rich per-ISA
//! decoders [`crate::x86`] and [`crate::aarch64`]. It stays
//! *format*-neutral: memory is read only through [`Image::regions`],
//! [`Image::va_to_offset`], and [`Image::bytes`], never through a loader.
//!
//! # Recognized idioms
//!
//! Each is matched by walking *backwards* from the indirect jump over the
//! instructions of its basic block, following the last definition of each
//! register involved. An unrelated instruction that redefines a register
//! in the chain breaks the match (see "False positives", below).
//!
//! The walk is not confined to the dispatch block: compilers split the
//! idiom across blocks (hoisting the `lea`/`adrp` above the range check,
//! or above a whole dispatch loop), so when a block's top is reached the
//! walk continues into its predecessor — along a straight-line chain
//! only, at most [`Config::max_walk_blocks`] blocks deep (see "The
//! split-block chain", below). All idiom and validation requirements are
//! unchanged; only *where the instructions may sit* widens.
//!
//! ### x86-64
//!
//! 1. **PIC table of 4-byte self-relative offsets**
//!    ([`Idiom::X86RipRelativeOffsetTable`]) — clang and gcc `-fPIC`
//!    (and the MSVC x64 `switch` dispatch, modulo its image-base `lea`):
//!
//!    ```text
//!    cmp    edi, 3                     ; bounds check (previous block)
//!    ja     default
//!    lea    rcx, [rip + table]         ; table base
//!    movsxd rax, dword [rcx + rdi*4]   ; (or `mov eax, [rcx + rdi*4]`)
//!    add    rax, rcx                   ; entry is an offset from the base
//!    jmp    rax
//!    ```
//!
//!    Entries are `i32` offsets from the `lea`'d base
//!    ([`TableKind::SelfRelativeOffsets`]).
//!
//!    The split-block variant hoists the `lea` out of the dispatch
//!    block — /bin/ls's option loop is the type specimen, its table
//!    base parked in callee-saved `r13` above the whole `getopt` loop:
//!
//!    ```text
//!    lea    r13, [rip + table]         ; loop preheader
//!    head:                             ; loop head (join: preheader + latch)
//!    call   _getopt_long               ; r13 is callee-saved
//!    lea    ecx, [rax - 0x25]
//!    cmp    ecx, 0x5b                  ; bounds check
//!    ja     default
//!    movsxd rax, dword [r13 + rcx*4]   ; dispatch block
//!    add    rax, r13
//!    jmp    rax
//!    ```
//!
//! 2. **PIC table of 8-byte absolute pointers**
//!    ([`Idiom::X86RipRelativePointerTable`]):
//!
//!    ```text
//!    lea rcx, [rip + table]
//!    jmp qword [rcx + rdi*8]           ; (or `mov rax, [rcx+rdi*8]; jmp rax`)
//!    ```
//!
//! 3. **Non-PIC table of 8-byte absolute pointers**
//!    ([`Idiom::X86AbsolutePointerTable`]) — clang/gcc without `-fPIC`,
//!    where the table address is a `disp32` in the jump itself:
//!
//!    ```text
//!    jmp qword [rdi*8 + 0x401040]
//!    ```
//!
//! ### AArch64
//!
//! 4. **Compressed byte/halfword offset table**
//!    ([`Idiom::A64CompressedOffsetTable`]) — the LLVM
//!    `JumpTableDest8`/`JumpTableDest16` expansion, whose table holds
//!    `(target - anchor) / 4` in one or two bytes:
//!
//!    ```text
//!    cmp   w0, #3                      ; bounds check (previous block)
//!    b.hi  default
//!    adr   x8, .LJTI                   ; anchor == table base
//!    ldrb  w9, [x8, x0]                ; (or `ldrh w9, [x8, x0, lsl #1]`)
//!    add   x10, x8, x9, lsl #2         ; (or an extended-register `add`)
//!    br    x10
//!    ```
//!
//!    Entries are unsigned, scaled by 4 and added to the anchor
//!    ([`TableKind::ByteOffsetShifted`]). The anchor and the table base
//!    are resolved independently (each may come from `ADR`, or from an
//!    `ADRP`+`ADD #lo12` pair), so LLVM's two-register form — where the
//!    `ADR` anchor register differs from the `ADRP`+`ADD` table register
//!    — is recognized as well.
//!
//! 5. **Table of 4-byte self-relative offsets**
//!    ([`Idiom::A64SelfRelativeWordTable`]) — the LLVM
//!    `JumpTableDest32` expansion:
//!
//!    ```text
//!    adrp  x8, .LJTI
//!    add   x8, x8, :lo12:.LJTI
//!    ldrsw x9, [x8, x0, lsl #2]
//!    add   x9, x8, x9
//!    br    x9
//!    ```
//!
//!    Entries are `i32` offsets from the table base, which here doubles
//!    as the anchor ([`TableKind::SelfRelativeOffsets`]).
//!
//! Pointer-authenticated dispatch (`BRAA`), Windows ARM64 tables, gcc's
//! `casesi` thumb-style byte tables, and interpreter-style computed
//! `goto` are **not** recognized and are left unresolved.
//!
//! # The split-block chain
//!
//! Register definitions are resolved over the *straight-line chain of
//! blocks* ending at the dispatch block, built deterministically and
//! refused rather than guessed:
//!
//! - A block with exactly one predecessor extends the chain into it —
//!   the only path there is, so a definition found in it is the
//!   definition that executed.
//! - A block with several predecessors extends the chain only into the
//!   predecessor that *dominates* it (at most one predecessor can — the
//!   immediate dominators come from the standard iterative RPO
//!   computation, deterministic and function-local), and only when every
//!   skipped predecessor is a loop edge: a block the join itself
//!   reaches. Both halves matter. Dominance guarantees the chained
//!   definition executed before the dispatch; the loop-edge test
//!   guarantees the only paths the chain does not see are loop
//!   iterations. This is how the walk climbs out of a dispatch loop to
//!   the preheader holding the `lea`, whether the back edges re-enter at
//!   the join itself or at its dominating header. It assumes the loop
//!   body preserves the tracked register — the one deliberate leap past
//!   pure last-definition reasoning (compilers park split table bases in
//!   callee-saved registers for exactly this reason), and the table
//!   validation below still has to pass. Any other join shape stops the
//!   chain.
//! - The chain crosses at most [`Config::max_walk_blocks`] predecessor
//!   blocks (`0` confines matching to the dispatch block), never revisits
//!   a block, and stops at any block whose decoded window does not reach
//!   back to its first instruction — a hidden instruction could be a
//!   hidden definition.
//!
//! The chain's blocks are decoded into one instruction stream and the
//! idiom matchers run over it unchanged, so every requirement below —
//! last-definition following, clobber breaking, exact shapes — applies
//! across blocks exactly as within one. The index-register rule keeps
//! its original scope (no redefinition inside the dispatch block), and
//! bounds-check discovery is untouched: it already reads the guarding
//! predecessor.
//!
//! A `call` in the chain clobbers the ABI's caller-saved registers, not
//! literally everything: rbx/rbp/r12-r15 survive on x86-64 (callee-saved
//! in both the SysV and Windows ABIs; rsi/rdi are treated as clobbered,
//! over-approximating Windows), x19-x29 on AArch64 (AAPCS64). The
//! split-block idiom exists *because* compilers keep the table base in a
//! callee-saved register across the dispatch loop's calls; assuming a
//! callee honors the ABI is the price of seeing it, and a lying callee
//! still has to forge a table that passes validation.
//!
//! # Bounds
//!
//! The entry count comes from the `switch` range check when one can be
//! recovered: the predecessor block whose conditional branch guards this
//! one must end in `cmp <index>, N` + an unsigned `ja`/`jae` (x86) or
//! `cmp <index>, #N` + `b.hi`/`b.hs` (A64), on the *same* register the
//! table is indexed by. That yields [`Bound::FromCompare`]. With no
//! recoverable check the count falls back to [`Config::max_entries`] and
//! the table is marked [`Bound::Assumed`] — or dropped entirely when
//! [`Config::require_bounds_check`] is set.
//!
//! # Reading the table, and truncation
//!
//! Nothing is read without a bound. `entry_count` is the number of
//! entries this pass was willing to consider: the compare-derived (or
//! assumed) count, clamped to [`Config::max_entries`] *and* to the number
//! of whole entries between the table and the end of the region
//! containing it. So a crafted `jmp [rax*8 + table]` can never read
//! beyond one region, let alone 4 GB.
//!
//! `targets` is then the prefix of those entries that actually decoded.
//! The table is **truncated** — not merely filtered — at the first entry
//! that:
//!
//! - cannot be read (no file backing, or a non-contiguous mapping), or
//! - does not land in an executable region, or
//! - points inside the table bytes read so far.
//!
//! Truncation rather than dropping is the honest choice: a jump table is
//! a contiguous array, so the first nonsensical entry marks where the
//! table really ends, and everything past it is somebody else's data. The
//! last rule is what terminates an [`Bound::Assumed`] table that has run
//! off the end of the real one into zero padding (a zero self-relative
//! offset points at the table's own base). `targets.len()` is therefore
//! `<= entry_count`; the two are equal for a clean, fully validated
//! table.
//!
//! # False positives
//!
//! Recognizing a table that is not there is worse than recognizing
//! nothing, so every step is a conjunction of exact requirements:
//!
//! - The idiom must match instruction-for-instruction: opcode, operand
//!   shape, register widths, scale factor, and shift amount.
//! - Register chains are followed by *last definition*, not by "the
//!   nearest instruction that looks right". Anything that redefines a
//!   register in the chain between its definition and its use breaks the
//!   match — including a `call` for every caller-saved register (see
//!   "The split-block chain" for the callee-saved contract) and, on
//!   A64, any instruction outside the decoder's modeled subset.
//! - The index register must not be redefined between the bounds check
//!   and the table load, and the bounds check must compare *that*
//!   register.
//! - The table base must resolve to a concrete VA inside a mapped
//!   region; an unmapped `lea`/`adr` target yields no table.
//! - A resolved table must produce at least [`Config::min_targets`]
//!   validated targets (default 2). A "table" with one target is a
//!   coincidence, not a `switch`.
//! - An indirect jump through an import slot (a tail call, already named
//!   by [`crate::cfg`]) is skipped outright.
//!
//! # Determinism and panics
//!
//! Functions, blocks, and results are all walked and keyed in address
//! order, so the same image always yields the same `Vec<JumpTable>`. All
//! address arithmetic is checked or wrapping, and every table read is
//! bounds-checked against both the region and the file buffer: no input
//! panics.
//!
//! # Feeding the edges back
//!
//! This module does not mutate [`crate::cfg`]'s types.
//! [`JumpTable::resolved_successors`] and [`successor_map`] expose
//! `jump_site -> targets` as plain data, and
//! [`cfg::recover_with_tables`] folds such a map into block successors.
//! The two are chicken-and-egg — a table's case bodies may be code
//! recovery only reaches *through* the folded edges, and that new code
//! can hold further tables — so [`resolve_folded`] is the entry point
//! that runs recover → resolve rounds to the joint fixpoint. Callers
//! that want real dispatch edges use it; plain [`resolve`] stays the
//! single-pass proof over an existing program.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::cfg;
use crate::error::{ParseError, Result};
use crate::model::{Arch, Image, Region};
use crate::{aarch64, x86};

/// Where a table's entry count came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    /// Recovered from the `switch` range check guarding the dispatch.
    FromCompare,
    /// No range check was recovered; [`Config::max_entries`] was assumed
    /// and the table was terminated by entry validation (see the module
    /// docs on truncation).
    Assumed,
}

/// How a table entry encodes its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableKind {
    /// Each entry is the target VA itself (8 bytes, little-endian).
    AbsolutePointers,
    /// Each entry is a signed offset from `base` (the VA the compiler
    /// added the loaded value to), stored in `element_size` bytes.
    SelfRelativeOffsets {
        /// VA the offsets are relative to.
        base: u64,
    },
    /// Each entry is an *unsigned* value scaled by `1 << shift` and added
    /// to `anchor` — the compressed A64 form.
    ByteOffsetShifted {
        /// VA the scaled offsets are relative to.
        anchor: u64,
        /// Left-shift applied to each entry (2 for A64: instructions are
        /// 4-byte aligned).
        shift: u8,
    },
}

/// Which compiler idiom matched. Recorded so a caller can tell *why* a
/// table was believed, and so an unexpected mix is visible in a report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Idiom {
    /// x86-64 `lea rB,[rip+T]; movsxd rD,[rB+rI*4]; add rD,rB; jmp rD`.
    X86RipRelativeOffsetTable,
    /// x86-64 `lea rB,[rip+T]; jmp [rB+rI*8]` (or via a `mov` + `jmp rD`).
    X86RipRelativePointerTable,
    /// x86-64 `jmp [rI*8 + T]` with a static `disp32` table base.
    X86AbsolutePointerTable,
    /// A64 `adr xB,T; ldrb wS,[xT,xI]; add xD,xB,xS,lsl #2; br xD`.
    A64CompressedOffsetTable,
    /// A64 `adrp+add xT,T; ldrsw xS,[xT,xI,lsl #2]; add xD,xT,xS; br xD`.
    A64SelfRelativeWordTable,
}

/// One recovered jump table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JumpTable {
    /// VA of the indirect jump this table dispatches.
    pub jump_site: u64,
    /// VA of the table's first entry.
    pub table_va: u64,
    /// Size of one entry in bytes.
    pub element_size: u8,
    /// How entries encode their targets.
    pub kind: TableKind,
    /// Entries considered: the bound (see [`Bound`]), clamped to
    /// [`Config::max_entries`] and to the end of the table's region.
    /// Always `>= targets.len()`; see the module docs on truncation.
    pub entry_count: usize,
    /// Validated targets, in table order, truncated at the first entry
    /// that failed to decode or validate.
    pub targets: Vec<u64>,
    /// Provenance of `entry_count`.
    pub bound: Bound,
    /// The compiler idiom that matched.
    pub idiom: Idiom,
}

impl JumpTable {
    /// The table's targets as a control-flow successor set: sorted and
    /// deduplicated, ready to fold into a basic block's edges.
    pub fn resolved_successors(&self) -> Vec<u64> {
        let mut s = self.targets.clone();
        s.sort_unstable();
        s.dedup();
        s
    }
}

/// `jump_site -> resolved successors`, for a future CFG pass. Tables are
/// keyed by jump site, so the map is deterministic and one entry per
/// indirect jump.
pub fn successor_map(tables: &[JumpTable]) -> BTreeMap<u64, Vec<u64>> {
    tables
        .iter()
        .map(|t| (t.jump_site, t.resolved_successors()))
        .collect()
}

/// Caps and policy for [`resolve_with`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Hard cap on entries read from any one table, whatever the
    /// recovered or assumed bound says.
    pub max_entries: usize,
    /// Minimum validated targets for a match to be believed. Below this
    /// the candidate is discarded (see "False positives").
    pub min_targets: usize,
    /// Reject candidates with no recoverable range check instead of
    /// falling back to [`Config::max_entries`] / [`Bound::Assumed`].
    pub require_bounds_check: bool,
    /// How many trailing instructions of a basic block the backward walk
    /// keeps. Every recognized idiom is a handful of instructions long;
    /// this bounds the memory a pathological block can cost.
    pub scan_window: usize,
    /// How many predecessor blocks the backward def-walk may cross when
    /// an idiom's instructions are split across blocks (see "The
    /// split-block chain" in the module docs). `0` confines matching to
    /// the dispatch block. The chain is straight-line and refused at any
    /// ambiguous join, so the bound is depth, not fan-out.
    pub max_walk_blocks: usize,
    /// Upper bound on the recover → resolve rounds [`resolve_folded_with`]
    /// runs (clamped to at least one). Each fold can reach new code that
    /// holds new tables, so the loop iterates; real images settle in two
    /// or three rounds, and the cap is defense in depth against an image
    /// crafted to keep unveiling tables. Hitting it sets
    /// [`Folded::capped`] — visible, never silent.
    pub max_fold_rounds: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_entries: 1024,
            min_targets: 2,
            require_bounds_check: false,
            scan_window: 32,
            max_walk_blocks: 8,
            max_fold_rounds: 8,
        }
    }
}

/// Recover jump tables for every unresolved indirect jump in `program`,
/// with the default [`Config`].
///
/// Fails with [`ParseError::Unsupported`] when the image's architecture
/// has no jump-table patterns in this pass ([`Arch::Other`] — which
/// [`cfg::recover`] rejects too). Never panics on any input.
pub fn resolve(image: &dyn Image, program: &cfg::Program) -> Result<Vec<JumpTable>> {
    resolve_with(image, program, &Config::default())
}

/// [`resolve`] with caller-supplied caps and policy.
pub fn resolve_with(
    image: &dyn Image,
    program: &cfg::Program,
    config: &Config,
) -> Result<Vec<JumpTable>> {
    let arch = image.arch();
    if arch == Arch::Other {
        return Err(ParseError::Unsupported(format!(
            "jump-table recovery: no patterns for architecture {arch:?}"
        )));
    }

    let regions = image.regions();
    let mut exec: Vec<(u64, u64)> = regions
        .iter()
        .filter(|r| r.perms.x && r.size > 0)
        .map(|r| (r.va, r.va.saturating_add(r.size)))
        .collect();
    exec.sort_unstable();
    exec.dedup();

    let res = Resolver {
        image,
        regions,
        exec,
        config,
    };

    // Keyed by jump site: blocks shared between functions resolve once,
    // and the result is emitted in address order.
    let mut out: BTreeMap<u64, JumpTable> = BTreeMap::new();
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    for func in program.functions.values() {
        for block in func.blocks.values() {
            let cfg::Terminator::IndirectJump { import } = &block.terminator else {
                continue;
            };
            // Already resolved by cfg as a tail call to an import.
            if import.is_some() {
                continue;
            }
            if !seen.insert(block.start) {
                continue;
            }
            let table = match arch {
                Arch::X86_64 => res.x86_table(func, block),
                Arch::Aarch64 => res.a64_table(func, block),
                Arch::Other => None,
            };
            if let Some(t) = table {
                out.entry(t.jump_site).or_insert(t);
            }
        }
    }
    Ok(out.into_values().collect())
}

/// A recovered program and its proven jump tables, run to a joint
/// fixpoint by [`resolve_folded`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Folded {
    /// The recovered program with every proven table's targets folded
    /// into its dispatch block's successors
    /// ([`cfg::Program::stats`] carries the fold counters).
    pub program: cfg::Program,
    /// The final round's proven tables, in jump-site order — exactly
    /// what [`resolve`] returns over `program`.
    pub tables: Vec<JumpTable>,
    /// recover → resolve rounds run, `>= 1`. One round means no table
    /// was proven; a table needs a second round to fold; each round
    /// after that folded a table found only in the previous round's
    /// newly reached code.
    pub rounds: usize,
    /// [`Config::max_fold_rounds`] stopped the loop before the
    /// fixpoint: `tables` holds proofs `program` has not folded.
    pub capped: bool,
}

/// Recover `image` and resolve its jump tables to a joint fixpoint,
/// with the default [`cfg::Config`] and [`Config`].
///
/// Round one is plain [`cfg::recover`] + [`resolve`]. Each further
/// round hands the proven `jump_site -> targets` map to
/// [`cfg::recover_with_tables`], whose fold walks the case bodies into
/// blocks — code only reachable through the proven edges — and
/// re-resolves, since that new code can hold further tables. The loop
/// stops when a round proves nothing new (the map reaches a fixpoint)
/// or at [`Config::max_fold_rounds`] ([`Folded::capped`]). Every round
/// recovers from scratch, so the result is a pure function of the
/// image: deterministic, and identical to [`cfg::recover`] wherever no
/// table proved anything.
///
/// Fails with [`ParseError::Unsupported`] exactly when [`cfg::recover`]
/// does. Never panics on any input.
pub fn resolve_folded(image: &dyn Image) -> Result<Folded> {
    resolve_folded_with(image, &cfg::Config::default(), &Config::default())
}

/// [`resolve_folded`] with caller-supplied recovery caps and resolution
/// policy.
pub fn resolve_folded_with(
    image: &dyn Image,
    cfg_config: &cfg::Config,
    config: &Config,
) -> Result<Folded> {
    let cap = config.max_fold_rounds.max(1);
    let mut map: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for round in 1..=cap {
        let program = cfg::recover_with_tables(image, cfg_config, &map)?;
        let tables = resolve_with(image, &program, config)?;
        let next = successor_map(&tables);
        if next == map || round == cap {
            // `next == map`: `program` folded exactly the tables
            // returned. On the cap the two can disagree; `capped` says so.
            return Ok(Folded {
                program,
                tables,
                rounds: round,
                capped: next != map,
            });
        }
        map = next;
    }
    unreachable!("the loop returns on its final round");
}

/// A matched idiom, before the table bytes have been read.
struct Candidate {
    jump_site: u64,
    table_va: u64,
    element_size: u8,
    kind: TableKind,
    /// Entry count from a recovered range check, if any.
    bound: Option<usize>,
    idiom: Idiom,
}

/// The indirect jump being resolved, with the function it belongs to —
/// needed to find the predecessor block carrying the range check.
struct Site<'a> {
    func: &'a cfg::Function,
    block: &'a cfg::BasicBlock,
    /// VA of the indirect jump itself.
    jump_site: u64,
}

/// Whole-image resolution state.
struct Resolver<'a> {
    image: &'a dyn Image,
    regions: Vec<Region>,
    /// Executable `[start, end)` ranges, sorted.
    exec: Vec<(u64, u64)>,
    config: &'a Config,
}

impl Resolver<'_> {
    /// The mapped region containing `va` (first match in region order).
    fn region_of(&self, va: u64) -> Option<&Region> {
        self.regions
            .iter()
            .find(|r| r.size > 0 && va >= r.va && va - r.va < r.size)
    }

    fn is_exec(&self, va: u64) -> bool {
        self.exec.iter().any(|&(s, e)| va >= s && va < e)
    }

    /// `len` bytes at `va`, or `None` unless they lie wholly inside one
    /// region *and* map to a contiguous, in-bounds run of file bytes.
    fn read(&self, va: u64, len: usize) -> Option<&[u8]> {
        let last = va.checked_add(len.checked_sub(1)? as u64)?;
        let region = self.region_of(va)?;
        if last >= region.va.saturating_add(region.size) {
            return None;
        }
        let off = self.image.va_to_offset(va)?;
        let end = off.checked_add(len)?;
        let bytes = self.image.bytes();
        if end > bytes.len() {
            return None;
        }
        // The trait maps addresses one at a time; require the last byte
        // to land exactly where a contiguous mapping would put it.
        if self.image.va_to_offset(last)? != end - 1 {
            return None;
        }
        Some(&bytes[off..end])
    }

    /// Decode entry `index` of a table to its target VA.
    fn entry_target(&self, kind: TableKind, table_va: u64, esz: u8, index: usize) -> Option<u64> {
        let va = table_va.checked_add((index as u64).checked_mul(u64::from(esz))?)?;
        let b = self.read(va, usize::from(esz))?;
        Some(match kind {
            TableKind::AbsolutePointers => {
                let raw: [u8; 8] = b.try_into().ok()?;
                u64::from_le_bytes(raw)
            }
            TableKind::SelfRelativeOffsets { base } => {
                let raw: [u8; 4] = b.try_into().ok()?;
                base.wrapping_add(i32::from_le_bytes(raw) as i64 as u64)
            }
            TableKind::ByteOffsetShifted { anchor, shift } => {
                let scaled = match b.len() {
                    1 => u64::from(b[0]),
                    2 => u64::from(u16::from_le_bytes([b[0], b[1]])),
                    _ => return None,
                }
                .checked_shl(u32::from(shift))?;
                anchor.checked_add(scaled)?
            }
        })
    }

    /// Read and validate a matched candidate's table. See the module docs
    /// for the bound clamping and truncation rules.
    fn build(&self, c: Candidate) -> Option<JumpTable> {
        if c.bound.is_none() && self.config.require_bounds_check {
            return None;
        }
        let bound = if c.bound.is_some() {
            Bound::FromCompare
        } else {
            Bound::Assumed
        };
        let requested = c.bound.unwrap_or(self.config.max_entries);
        let region = self.region_of(c.table_va)?;
        let room = region.va.saturating_add(region.size).checked_sub(c.table_va)?;
        let fits = (room / u64::from(c.element_size)) as usize;
        let entry_count = requested.min(fits).min(self.config.max_entries);
        if entry_count == 0 {
            return None;
        }

        let mut targets = Vec::new();
        for i in 0..entry_count {
            let Some(target) = self.entry_target(c.kind, c.table_va, c.element_size, i) else {
                break; // unreadable: the table ends here
            };
            if !self.is_exec(target) {
                break;
            }
            // A target inside the table bytes read so far is not a target.
            let read_so_far = (i as u64 + 1) * u64::from(c.element_size);
            if target >= c.table_va && target - c.table_va < read_so_far {
                break;
            }
            targets.push(target);
        }
        if targets.len() < self.config.min_targets {
            return None;
        }
        Some(JumpTable {
            jump_site: c.jump_site,
            table_va: c.table_va,
            element_size: c.element_size,
            kind: c.kind,
            entry_count,
            targets,
            bound,
            idiom: c.idiom,
        })
    }
}

// ---------------------------------------------------------------------
// The split-block chain
// ---------------------------------------------------------------------

/// The blocks of `func` that transfer control to `start` — by branch,
/// fall-through, call return, or a previously folded table edge.
fn preds_of(func: &cfg::Function, start: u64) -> Vec<&cfg::BasicBlock> {
    func.blocks
        .values()
        .filter(|b| b.successors.contains(&start))
        .collect()
}

/// Every block start reachable from `start` along successor edges,
/// `start` itself included. Bounded by the function's block count (which
/// [`cfg::Config::max_blocks`] already caps) and deterministic: the
/// result is a set, so visit order cannot matter.
fn reachable_from(func: &cfg::Function, start: u64) -> BTreeSet<u64> {
    let mut seen = BTreeSet::from([start]);
    let mut work = vec![start];
    while let Some(va) = work.pop() {
        let Some(block) = func.blocks.get(&va) else {
            continue;
        };
        for &succ in &block.successors {
            if seen.insert(succ) {
                work.push(succ);
            }
        }
    }
    seen
}

/// Immediate dominators for `func`'s blocks, keyed by block start (the
/// entry maps to itself). The standard iterative RPO algorithm
/// (Cooper–Harvey–Kennedy): deterministic — the RPO comes from a DFS
/// over the blocks' fixed successor order — and terminating in a
/// handful of passes on real CFGs, bounded by the function's block
/// count. Blocks not reachable from the entry get no entry here, which
/// [`dominates`] reads as "dominates nothing".
fn dominators(func: &cfg::Function) -> BTreeMap<u64, u64> {
    // Reverse postorder over the blocks reachable from the entry.
    let mut post: Vec<u64> = Vec::new();
    let mut seen = BTreeSet::from([func.entry]);
    let mut stack: Vec<(u64, usize)> = vec![(func.entry, 0)];
    while let Some((va, i)) = stack.last_mut() {
        let succs = func
            .blocks
            .get(va)
            .map(|b| b.successors.as_slice())
            .unwrap_or(&[]);
        if let Some(&s) = succs.get(*i) {
            *i += 1;
            if func.blocks.contains_key(&s) && seen.insert(s) {
                stack.push((s, 0));
            }
        } else {
            post.push(*va);
            stack.pop();
        }
    }
    let order: Vec<u64> = post.into_iter().rev().collect();
    let index: BTreeMap<u64, usize> = order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let preds: BTreeMap<u64, Vec<u64>> = order
        .iter()
        .map(|&b| {
            let ps = func
                .blocks
                .values()
                .filter(|p| index.contains_key(&p.start) && p.successors.contains(&b))
                .map(|p| p.start)
                .collect();
            (b, ps)
        })
        .collect();

    fn intersect(idom: &BTreeMap<u64, u64>, index: &BTreeMap<u64, usize>, a: u64, b: u64) -> u64 {
        let (mut a, mut b) = (a, b);
        while a != b {
            while index[&a] > index[&b] {
                a = idom[&a];
            }
            while index[&b] > index[&a] {
                b = idom[&b];
            }
        }
        a
    }

    let mut idom: BTreeMap<u64, u64> = BTreeMap::from([(func.entry, func.entry)]);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().skip(1) {
            let mut new = None;
            for &p in &preds[&b] {
                if !idom.contains_key(&p) {
                    continue; // back edge from a not-yet-processed block
                }
                new = Some(match new {
                    None => p,
                    Some(n) => intersect(&idom, &index, p, n),
                });
            }
            let Some(new) = new else { continue };
            if idom.get(&b) != Some(&new) {
                idom.insert(b, new);
                changed = true;
            }
        }
    }
    idom
}

/// Does block `a` dominate block `b`, per a [`dominators`] map? Walks
/// `b`'s immediate-dominator chain up to the entry (whose idom is
/// itself), so the walk always terminates.
fn dominates(idom: &BTreeMap<u64, u64>, a: u64, b: u64) -> bool {
    let mut b = b;
    loop {
        if a == b {
            return true;
        }
        let Some(&up) = idom.get(&b) else {
            return false;
        };
        if up == b {
            return false; // reached the entry without meeting `a`
        }
        b = up;
    }
}

impl Resolver<'_> {
    /// The straight-line chain of blocks ending at `block` (which is the
    /// chain's last element), per the rules in "The split-block chain":
    /// single-predecessor edges extend it; a join extends it only through
    /// the predecessor that dominates it, and only when every skipped
    /// predecessor is a loop edge; anything else — no predecessor, an
    /// ambiguous join, a revisit, [`Config::max_walk_blocks`] — stops it.
    /// Deterministic: blocks and predecessors are walked in the
    /// function's fixed address order.
    fn def_walk_chain<'f>(
        &self,
        func: &'f cfg::Function,
        block: &'f cfg::BasicBlock,
    ) -> Vec<&'f cfg::BasicBlock> {
        let mut chain = vec![block];
        let mut visited = BTreeSet::from([block.start]);
        let mut idom = None; // dominators, computed at the first join
        for _ in 0..self.config.max_walk_blocks {
            let head = chain[0];
            let preds = preds_of(func, head.start);
            let next = match preds.as_slice() {
                [] => break,
                [only] => *only,
                _ => {
                    // A join: continue only through the predecessor that
                    // dominates it (at most one exists), and only when
                    // every skipped predecessor is a loop edge — a block
                    // the join itself reaches — so the paths the chain
                    // does not see are loop iterations, covered by the
                    // documented preserves-the-register leap. Any other
                    // join shape could hide a clobber and stops the walk.
                    let idom = idom.get_or_insert_with(|| dominators(func));
                    let Some(p) = preds
                        .iter()
                        .find(|p| p.start != head.start && dominates(idom, p.start, head.start))
                    else {
                        break;
                    };
                    let body = reachable_from(func, head.start);
                    if preds
                        .iter()
                        .any(|q| q.start != p.start && !body.contains(&q.start))
                    {
                        break;
                    }
                    *p
                }
            };
            if !visited.insert(next.start) {
                break; // a cycle proves nothing twice
            }
            chain.insert(0, next);
        }
        chain
    }
}

// ---------------------------------------------------------------------
// x86-64
// ---------------------------------------------------------------------

/// One decoded x86-64 instruction of a basic block.
struct X86Insn {
    va: u64,
    ins: x86::Instruction,
}

/// Bit for a register operand, or 0 for anything else.
fn x86_reg_bit(op: Option<&x86::Operand>) -> u16 {
    match op {
        Some(x86::Operand::Reg(r)) => 1u16 << (r.num & 0xF),
        _ => 0,
    }
}

/// The set of general-purpose registers an instruction may write, as a
/// bitmask over register numbers 0-15.
///
/// Deliberately over-approximate — over-approximating here can only
/// *break* an idiom match, never invent one — with one measured
/// exception: `call` clobbers the caller-saved set rather than
/// everything. rbx/rbp/r12-r15 are callee-saved in both the SysV and
/// Windows x64 ABIs (rsi/rdi, callee-saved only on Windows, stay
/// clobbered), and the split-block idiom keeps its table base live in
/// exactly those registers across the dispatch loop's calls — /bin/ls
/// parks it in r13 across `_getopt_long`. See "The split-block chain"
/// in the module docs.
fn x86_defs(ins: &x86::Instruction) -> u16 {
    use x86::Opcode as O;
    const RAX: u16 = 0x0001;
    const RCX: u16 = 0x0002;
    const RDX: u16 = 0x0004;
    const RBX: u16 = 0x0008;
    const RSP: u16 = 0x0010;
    const RBP: u16 = 0x0020;
    const RSI: u16 = 0x0040;
    const RDI: u16 = 0x0080;
    const R8: u16 = 0x0100;
    const R9: u16 = 0x0200;
    const R10: u16 = 0x0400;
    const R11: u16 = 0x0800;
    /// What a `call` may write: every GPR minus the common callee-saved
    /// set of the SysV and Windows x64 ABIs (rbx, rbp, r12-r15).
    const CALL_CLOBBERS: u16 = RAX | RCX | RDX | RSP | RSI | RDI | R8 | R9 | R10 | R11;

    let d0 = x86_reg_bit(ins.operands.first());
    let d1 = x86_reg_bit(ins.operands.get(1));
    match ins.opcode {
        O::Add
        | O::Or
        | O::Adc
        | O::Sbb
        | O::And
        | O::Sub
        | O::Xor
        | O::Mov
        | O::Movsx
        | O::Movzx
        | O::Movsxd
        | O::Lea
        | O::Inc
        | O::Dec
        | O::Not
        | O::Neg
        | O::Bts
        | O::Btr
        | O::Btc
        | O::Setcc(_)
        | O::Cmov(_) => d0,
        O::Pop => d0 | RSP,
        O::Push => RSP,
        O::Xchg | O::Xadd => d0 | d1,
        O::Cmpxchg => d0 | d1 | RAX,
        // One-operand `imul`/`mul`/`div`/`idiv` write rDX:rAX.
        O::Imul => {
            if ins.operands.len() == 1 {
                RAX | RDX
            } else {
                d0
            }
        }
        O::Mul | O::Div | O::Idiv => RAX | RDX,
        O::Cwde => RAX,
        O::Cdq => RAX | RDX,
        O::Ret => RSP,
        O::Leave => RSP | RBP,
        O::Syscall => RAX | RCX | R11,
        O::Cpuid => RAX | RBX | RCX | RDX,
        O::Rdtsc => RAX | RDX,
        O::Call => CALL_CLOBBERS,
        O::Cmp
        | O::Test
        | O::Bt
        | O::Nop
        | O::Jmp
        | O::Jcc(_)
        | O::Int3
        | O::Int
        | O::Ud2
        | O::Hlt
        | O::Endbr64
        | O::Endbr32 => 0,
        // Only the GPR-writing SSE forms define a register, and it is the
        // (first) destination operand; `d0` is exactly that set.
        O::Sse { .. } => d0,
    }
}

/// Index of the last instruction before `before` that writes register
/// `num`.
fn x86_last_def(insns: &[X86Insn], before: usize, num: u8) -> Option<usize> {
    let mask = 1u16 << (num & 0xF);
    insns
        .get(..before)?
        .iter()
        .rposition(|i| x86_defs(&i.ins) & mask != 0)
}

/// A 64-bit register operand's number.
fn x86_gpr64(op: Option<&x86::Operand>) -> Option<u8> {
    match op {
        Some(&x86::Operand::Reg(r)) if r.width == x86::Width::W64 && !r.high_byte => Some(r.num),
        _ => None,
    }
}

/// A memory operand, destructured.
struct X86Mem {
    base: Option<u8>,
    index: Option<u8>,
    scale: u8,
    disp: i64,
    rip_relative: bool,
}

fn x86_mem(op: Option<&x86::Operand>) -> Option<X86Mem> {
    match op {
        Some(&x86::Operand::Mem {
            base,
            index,
            scale,
            disp,
            rip_relative,
        }) => Some(X86Mem {
            base: base.map(|r| r.num),
            index: index.map(|r| r.num),
            scale,
            disp,
            rip_relative,
        }),
        _ => None,
    }
}

impl Resolver<'_> {
    /// Decode a block's instructions, keeping the last
    /// [`Config::scan_window`] of them.
    fn x86_block_insns(&self, block: &cfg::BasicBlock) -> Vec<X86Insn> {
        let mut window: VecDeque<X86Insn> = VecDeque::new();
        let bytes = self.image.bytes();
        let mut va = block.start;
        while va < block.end {
            let Some(off) = self.image.va_to_offset(va) else {
                break;
            };
            let avail = bytes.len().saturating_sub(off);
            let limit = (block.end - va).min(x86::MAX_INSTRUCTION_LEN as u64) as usize;
            let Ok(ins) = x86::decode(&bytes[off..off + limit.min(avail)], va) else {
                break;
            };
            let Some(next) = va.checked_add(u64::from(ins.length)) else {
                break;
            };
            if ins.length == 0 {
                break;
            }
            window.push_back(X86Insn { va, ins });
            if window.len() > self.config.scan_window.max(1) {
                window.pop_front();
            }
            va = next;
        }
        window.into()
    }

    /// Decode the def-walk chain ending at `block` into one instruction
    /// stream, oldest block first, and return it with the stream index of
    /// the dispatch block's first instruction. The idiom matchers run
    /// over the stream unchanged; the index is what keeps the
    /// index-register rule scoped to the dispatch block.
    ///
    /// A window that does not reach back to its block's first instruction
    /// ends the stream: an instruction it dropped could be a definition
    /// the match must see. A predecessor window that does not reach its
    /// block's *end* is excluded outright for the same reason — its
    /// missing tail would sit right where the stream splices.
    fn x86_chain_insns(
        &self,
        func: &cfg::Function,
        block: &cfg::BasicBlock,
    ) -> (Vec<X86Insn>, usize) {
        let chain = self.def_walk_chain(func, block);
        let mut segments: Vec<Vec<X86Insn>> = vec![self.x86_block_insns(block)];
        let mut extendable = segments[0].first().is_some_and(|i| i.va == block.start);
        for b in chain.iter().rev().skip(1) {
            if !extendable {
                break;
            }
            let insns = self.x86_block_insns(b);
            let reaches_end = insns
                .last()
                .is_some_and(|i| i.va.checked_add(u64::from(i.ins.length)) == Some(b.end));
            if !reaches_end {
                break;
            }
            extendable = insns.first().is_some_and(|i| i.va == b.start);
            segments.push(insns);
        }
        let dispatch_start = segments[1..].iter().map(Vec::len).sum();
        let mut stream = Vec::new();
        for seg in segments.into_iter().rev() {
            stream.extend(seg);
        }
        (stream, dispatch_start)
    }

    /// The VA a register was loaded with by a preceding
    /// `lea r64, [rip + disp]`, with that `lea`'s index.
    fn x86_lea_target(&self, insns: &[X86Insn], before: usize, num: u8) -> Option<(usize, u64)> {
        let j = x86_last_def(insns, before, num)?;
        let ins = &insns[j].ins;
        if ins.opcode != x86::Opcode::Lea || x86_gpr64(ins.operands.first()) != Some(num) {
            return None;
        }
        let mem = x86_mem(ins.operands.get(1))?;
        if !mem.rip_relative || mem.base.is_some() || mem.index.is_some() {
            return None;
        }
        // `[rip + disp]` is relative to the *next* instruction (SDM §2.2.1.6).
        let next = insns[j].va.checked_add(u64::from(ins.length))?;
        Some((j, next.checked_add_signed(mem.disp)?))
    }

    /// Match an x86-64 jump-table idiom on a block ending in an indirect
    /// jump. The instructions come from the block's def-walk chain, so
    /// the idiom may sit split across the chain's blocks.
    fn x86_table(&self, func: &cfg::Function, block: &cfg::BasicBlock) -> Option<JumpTable> {
        let (insns, dispatch) = self.x86_chain_insns(func, block);
        let last = insns.len().checked_sub(1)?;
        let term = &insns[last].ins;
        if term.opcode != x86::Opcode::Jmp {
            return None;
        }
        let site = Site {
            func,
            block,
            jump_site: insns[last].va,
        };

        // `jmp qword [... + idx*8]`: a table of absolute pointers.
        if let Some(mem) = x86_mem(term.operands.first()) {
            return self.x86_pointer_table(&site, &insns, last, &mem, last, dispatch);
        }

        // `jmp reg`: the target was computed into a register.
        let dest = x86_gpr64(term.operands.first())?;
        let di = x86_last_def(&insns, last, dest)?;
        match insns[di].ins.opcode {
            // `add rD, rB` — the offset-table form.
            x86::Opcode::Add => {
                let def = &insns[di].ins;
                if x86_gpr64(def.operands.first()) != Some(dest) {
                    return None;
                }
                let src = x86_gpr64(def.operands.get(1))?;
                // One of {previous def of rD, def of rB} is the table
                // load and the other is the `lea`; try both assignments.
                let a = x86_last_def(&insns, di, dest);
                let b = x86_last_def(&insns, di, src);
                a.and_then(|load| self.x86_offset_table(&site, &insns, di, load, src, dispatch))
                    .or_else(|| {
                        b.and_then(|load| {
                            self.x86_offset_table(&site, &insns, di, load, dest, dispatch)
                        })
                    })
            }
            // `mov rD, qword [rB + idx*8]; jmp rD` — pointer table.
            x86::Opcode::Mov => {
                let def = &insns[di].ins;
                if x86_gpr64(def.operands.first()) != Some(dest) {
                    return None;
                }
                let mem = x86_mem(def.operands.get(1))?;
                self.x86_pointer_table(&site, &insns, di, &mem, di, dispatch)
            }
            _ => None,
        }
    }

    /// Idioms 2 and 3: an 8-byte absolute-pointer table addressed by
    /// `[base + idx*8]` (base from a rip-relative `lea`) or by
    /// `[idx*8 + disp32]`.
    fn x86_pointer_table(
        &self,
        site: &Site,
        insns: &[X86Insn],
        at: usize,
        mem: &X86Mem,
        index_use: usize,
        dispatch: usize,
    ) -> Option<JumpTable> {
        if mem.scale != 8 || mem.rip_relative {
            return None;
        }
        let index = mem.index?;
        let (table_va, idiom) = match mem.base {
            Some(base) => {
                let (_, base_va) = self.x86_lea_target(insns, at, base)?;
                (
                    base_va.checked_add_signed(mem.disp)?,
                    Idiom::X86RipRelativePointerTable,
                )
            }
            None => {
                if mem.disp <= 0 {
                    return None;
                }
                (mem.disp as u64, Idiom::X86AbsolutePointerTable)
            }
        };
        if x86_last_def(insns, index_use, index).is_some_and(|d| d >= dispatch) {
            return None; // index recomputed in the dispatch block
        }
        self.build(Candidate {
            jump_site: site.jump_site,
            table_va,
            element_size: 8,
            kind: TableKind::AbsolutePointers,
            bound: self.x86_bound(site, index),
            idiom,
        })
    }

    /// Idiom 1: `movsxd`/`mov` of a 4-byte self-relative offset from
    /// `[base + idx*4]`, added to the same `lea`'d base.
    fn x86_offset_table(
        &self,
        site: &Site,
        insns: &[X86Insn],
        add: usize,
        load: usize,
        base: u8,
        dispatch: usize,
    ) -> Option<JumpTable> {
        let ins = &insns[load].ins;
        let dst = match ins.operands.first() {
            Some(&x86::Operand::Reg(r)) if !r.high_byte => r,
            _ => return None,
        };
        // `movsxd r64, m32` sign-extends; `mov r32, m32` zero-extends.
        // Both leave a 64-bit value the `add` can use.
        match (ins.opcode, dst.width) {
            (x86::Opcode::Movsxd, x86::Width::W64) | (x86::Opcode::Mov, x86::Width::W32) => {}
            _ => return None,
        }
        let mem = x86_mem(ins.operands.get(1))?;
        if mem.scale != 4 || mem.rip_relative || mem.base != Some(base) {
            return None;
        }
        let index = mem.index?;
        // The base register must be the same `lea` result at the load and
        // at the add — nothing may have redefined it in between.
        let (lea, base_va) = self.x86_lea_target(insns, load, base)?;
        if self.x86_lea_target(insns, add, base)? != (lea, base_va) {
            return None;
        }
        if x86_last_def(insns, load, index).is_some_and(|d| d >= dispatch) {
            return None; // index recomputed in the dispatch block
        }
        self.build(Candidate {
            jump_site: site.jump_site,
            table_va: base_va.checked_add_signed(mem.disp)?,
            element_size: 4,
            kind: TableKind::SelfRelativeOffsets { base: base_va },
            bound: self.x86_bound(site, index),
            idiom: Idiom::X86RipRelativeOffsetTable,
        })
    }

    /// Entry count from the `switch` range check guarding the dispatch
    /// block: `cmp <index>, N` immediately followed by an unsigned
    /// `ja`/`jae` (or `jbe`/`jb` when the dispatch is on the taken edge).
    fn x86_bound(&self, site: &Site, index: u8) -> Option<usize> {
        for pred in site.func.blocks.values() {
            let cfg::Terminator::CondJump {
                taken, fallthrough, ..
            } = pred.terminator
            else {
                continue;
            };
            if taken == fallthrough || !pred.successors.contains(&site.block.start) {
                continue;
            }
            let insns = self.x86_block_insns(pred);
            let Some(last) = insns.len().checked_sub(1) else {
                continue;
            };
            let x86::Opcode::Jcc(cond) = insns[last].ins.opcode else {
                continue;
            };
            let Some(cmp) = last.checked_sub(1).map(|i| &insns[i].ins) else {
                continue;
            };
            if cmp.opcode != x86::Opcode::Cmp {
                continue;
            }
            let (Some(&x86::Operand::Reg(r)), Some(&x86::Operand::Imm(n))) =
                (cmp.operands.first(), cmp.operands.get(1))
            else {
                continue;
            };
            if r.num != index || r.high_byte || n < 0 {
                continue;
            }
            let n = n as usize;
            let count = if fallthrough == site.block.start {
                match cond {
                    x86::Cond::A => n.checked_add(1)?,  // idx <= n
                    x86::Cond::Ae => n,                 // idx <  n
                    _ => continue,
                }
            } else {
                match cond {
                    x86::Cond::Be => n.checked_add(1)?, // idx <= n
                    x86::Cond::B => n,                  // idx <  n
                    _ => continue,
                }
            };
            if count > 0 {
                return Some(count);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------
// AArch64
// ---------------------------------------------------------------------

/// One decoded A64 instruction of a basic block.
struct A64Insn {
    va: u64,
    ins: aarch64::Instruction,
}

/// An `ADD <Xd>, <Xn>, <Xm>{, LSL #amount}` (shifted register) or
/// `ADD <Xd>, <Xn>, <Wm|Xm>{, <extend> #amount}` (extended register),
/// read from the decoded opcode. The extend option is deliberately
/// ignored, exactly as the raw-word matcher this replaced ignored it: the
/// offset register was itself just loaded (zero- or sign-extended) from
/// the table, so its W and X views agree.
struct A64AddReg {
    rd: u8,
    rn: u8,
    rm: u8,
    amount: u8,
}

fn a64_add_reg(op: &aarch64::Opcode) -> Option<A64AddReg> {
    match *op {
        aarch64::Opcode::AddReg {
            sf: true,
            set_flags: false,
            rd,
            rn,
            rm,
            shift: aarch64::Shift::Lsl,
            amount,
        }
        | aarch64::Opcode::AddExt {
            sf: true,
            set_flags: false,
            rd,
            rn,
            rm,
            option: _,
            amount,
        } => Some(A64AddReg { rd, rn, rm, amount }),
        _ => None,
    }
}

/// The set of X registers an instruction may write, as a bitmask over
/// x0-x30 (writes to XZR/SP are not tracked).
///
/// Anything the decoder leaves as [`aarch64::Opcode::Unknown`] — beyond
/// the two register forms this module decodes itself — is treated as
/// writing every register. Over-approximating can only break a match.
fn a64_defs(ins: &aarch64::Instruction) -> u32 {
    use aarch64::AddrMode as M;
    use aarch64::Opcode as O;
    fn bit(r: u8) -> u32 {
        if r < 31 { 1u32 << r } else { 0 }
    }
    fn writeback(mode: M, rn: u8) -> u32 {
        match mode {
            M::Offset(_) => 0,
            M::PreIndex(_) | M::PostIndex(_) => bit(rn),
        }
    }
    match ins.opcode {
        O::Adr { rd, .. } | O::Adrp { rd, .. } => bit(rd),
        O::AddImm { rd, .. } | O::SubImm { rd, .. } => bit(rd),
        O::Movn { rd, .. } | O::Movz { rd, .. } | O::Movk { rd, .. } => bit(rd),
        O::Ldr { rt, rn, mode, .. } => bit(rt) | writeback(mode, rn),
        O::Str { rn, mode, .. } => writeback(mode, rn),
        O::Ldrs { rt, rn, mode, .. } => bit(rt) | writeback(mode, rn),
        // Register-offset and unscaled forms have no writeback: only Rt
        // is defined (loads), and a store defines nothing.
        O::LdrReg { rt, .. } | O::LdrsReg { rt, .. } | O::Ldur { rt, .. } | O::Ldurs { rt, .. } => {
            bit(rt)
        }
        O::StrReg { .. } | O::Stur { .. } => 0,
        O::Csel { rd, .. } | O::Csinc { rd, .. } | O::Csinv { rd, .. } | O::Csneg { rd, .. } => {
            bit(rd)
        }
        // Integer data-processing: exactly Rd (flag-only forms encode
        // Rd = 31, which `bit` already drops, as it does SP destinations).
        O::AddReg { rd, .. }
        | O::SubReg { rd, .. }
        | O::AddExt { rd, .. }
        | O::SubExt { rd, .. }
        | O::LogReg { rd, .. }
        | O::LogImm { rd, .. }
        | O::Sbfm { rd, .. }
        | O::Bfm { rd, .. }
        | O::Ubfm { rd, .. }
        | O::ShiftReg { rd, .. }
        | O::Udiv { rd, .. }
        | O::Sdiv { rd, .. }
        | O::Madd { rd, .. }
        | O::Msub { rd, .. }
        | O::Maddl { rd, .. }
        | O::Mulh { rd, .. }
        | O::Adc { rd, .. }
        | O::Sbc { rd, .. } => bit(rd),
        // The conditional compares write flags only, never a register.
        O::CcmpReg { .. } | O::CcmpImm { .. } => 0,
        // SIMD&FP loads/stores touch no X register beyond a writeback
        // base, and of the SIMD moves only the FP→general FMOV writes
        // one.
        O::FLdr { rn, mode, .. }
        | O::FStr { rn, mode, .. }
        | O::FLdp { rn, mode, .. }
        | O::FStp { rn, mode, .. } => writeback(mode, rn),
        O::FLdur { .. }
        | O::FStur { .. }
        | O::FLdrReg { .. }
        | O::FStrReg { .. }
        | O::FLdrLit { .. }
        | O::FmovReg { .. }
        | O::FmovFromGp { .. }
        | O::FmovImm { .. }
        | O::FmovVecImm { .. }
        | O::Movi { .. } => 0,
        O::FmovToGp { rd, .. } => bit(rd),
        O::LdrLit { rt, .. } => bit(rt),
        O::Ldp {
            rt, rt2, rn, mode, ..
        } => bit(rt) | bit(rt2) | writeback(mode, rn),
        O::Stp { rn, mode, .. } => writeback(mode, rn),
        O::B { .. }
        | O::BCond { .. }
        | O::Cbz { .. }
        | O::Cbnz { .. }
        | O::Tbz { .. }
        | O::Tbnz { .. }
        | O::Br { .. }
        | O::Ret { .. }
        | O::Svc { .. }
        | O::Hvc { .. }
        | O::Smc { .. }
        | O::Brk { .. }
        | O::Hlt { .. }
        | O::Nop
        | O::Yield
        | O::Wfe
        | O::Wfi
        | O::Sev
        | O::Sevl
        | O::Hint { .. } => 0,
        // Scalar FP arithmetic, compares, selects, and the whole-register
        // SIMD element writes touch no X register; the FP→general
        // conversions and element extractions define exactly Rd/Rt.
        O::FArith2 { .. }
        | O::FArith3 { .. }
        | O::FArith1 { .. }
        | O::FCvtPrec { .. }
        | O::Fcmp { .. }
        | O::Fccmp { .. }
        | O::Fcsel { .. }
        | O::FcvtToFp { .. }
        | O::FcvtIntScalar { .. }
        | O::DupGp { .. }
        | O::DupElemScalar { .. }
        | O::DupElemVec { .. }
        | O::InsGp { .. }
        | O::InsElem { .. } => 0,
        O::FcvtFromFp { rd, .. } | O::Umov { rd, .. } | O::Smov { rd, .. } => bit(rd),
        // Exclusives address with no writeback; the exclusive store also
        // defines its status register.
        O::Ldar { rt, .. } | O::Ldxr { rt, .. } => bit(rt),
        O::Stlr { .. } => 0,
        O::Stxr { ws, .. } => bit(ws),
        // Pointer authentication rewrites its one target register; the
        // authenticated branches mirror their plain forms. UDF traps.
        O::PacGpr { rd, .. } | O::XPac { rd, .. } => bit(rd),
        O::PacHint { .. } => bit(30),
        O::RetA { .. } | O::BrAuth { link: false, .. } | O::Udf { .. } => 0,
        O::Bits1 { rd, .. } | O::Extr { rd, .. } => bit(rd),
        O::LdpSw {
            rt, rt2, rn, mode, ..
        } => bit(rt) | bit(rt2) | writeback(mode, rn),
        // A call clobbers the AAPCS64 caller-saved set — x0-x18 and the
        // link register — and preserves callee-saved x19-x28 and the
        // frame pointer, mirroring the x86 `call` contract above.
        O::Bl { .. } | O::Blr { .. } | O::BrAuth { link: true, .. } => 0x4007_FFFF,
        // The data-processing forms this module used to re-parse out of
        // the raw word now decode directly and are handled above; what
        // remains `Unknown` really is unknown.
        O::Unknown(_) => u32::MAX,
    }
}

/// Index of the last instruction before `before` that writes `num`.
fn a64_last_def(insns: &[A64Insn], before: usize, num: u8) -> Option<usize> {
    if num >= 31 {
        return None;
    }
    let mask = 1u32 << num;
    insns
        .get(..before)?
        .iter()
        .rposition(|i| a64_defs(&i.ins) & mask != 0)
}

impl Resolver<'_> {
    /// Decode a block's instruction words, keeping the last
    /// [`Config::scan_window`] of them.
    fn a64_block_insns(&self, block: &cfg::BasicBlock) -> Vec<A64Insn> {
        let mut window: VecDeque<A64Insn> = VecDeque::new();
        let bytes = self.image.bytes();
        let size = aarch64::Instruction::SIZE;
        let mut va = block.start;
        while va < block.end {
            let Some(off) = self.image.va_to_offset(va) else {
                break;
            };
            if bytes.len().saturating_sub(off) < size || block.end - va < size as u64 {
                break;
            }
            let Ok(ins) = aarch64::decode(&bytes[off..off + size], va) else {
                break;
            };
            let Some(next) = va.checked_add(size as u64) else {
                break;
            };
            window.push_back(A64Insn { va, ins });
            if window.len() > self.config.scan_window.max(1) {
                window.pop_front();
            }
            va = next;
        }
        window.into()
    }

    /// [`Resolver::x86_chain_insns`], for A64: the def-walk chain as one
    /// instruction stream, oldest block first, with the stream index of
    /// the dispatch block's first instruction. The same window-coverage
    /// rules apply (a fixed 4-byte instruction size makes them exact).
    fn a64_chain_insns(
        &self,
        func: &cfg::Function,
        block: &cfg::BasicBlock,
    ) -> (Vec<A64Insn>, usize) {
        let size = aarch64::Instruction::SIZE as u64;
        let chain = self.def_walk_chain(func, block);
        let mut segments: Vec<Vec<A64Insn>> = vec![self.a64_block_insns(block)];
        let mut extendable = segments[0].first().is_some_and(|i| i.va == block.start);
        for b in chain.iter().rev().skip(1) {
            if !extendable {
                break;
            }
            let insns = self.a64_block_insns(b);
            let reaches_end = insns
                .last()
                .is_some_and(|i| i.va.checked_add(size) == Some(b.end));
            if !reaches_end {
                break;
            }
            extendable = insns.first().is_some_and(|i| i.va == b.start);
            segments.push(insns);
        }
        let dispatch = segments[1..].iter().map(Vec::len).sum();
        let mut stream = Vec::new();
        for seg in segments.into_iter().rev() {
            stream.extend(seg);
        }
        (stream, dispatch)
    }

    /// The VA a register holds from a preceding `ADR`, or an
    /// `ADRP` + `ADD #lo12` pair.
    fn a64_addr_value(&self, insns: &[A64Insn], before: usize, num: u8) -> Option<u64> {
        use aarch64::Opcode as O;
        let j = a64_last_def(insns, before, num)?;
        match insns[j].ins.opcode {
            O::Adr { rd, target } | O::Adrp { rd, target } if rd == num => Some(target),
            O::AddImm {
                sf: true,
                set_flags: false,
                rd,
                rn,
                imm,
            } if rd == num => {
                let k = a64_last_def(insns, j, rn)?;
                match insns[k].ins.opcode {
                    O::Adrp { rd: page_rd, target } if page_rd == rn => {
                        target.checked_add(u64::from(imm))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Match an AArch64 jump-table idiom on a block ending in `BR`. The
    /// instructions come from the block's def-walk chain, so the idiom
    /// may sit split across the chain's blocks.
    fn a64_table(&self, func: &cfg::Function, block: &cfg::BasicBlock) -> Option<JumpTable> {
        let (insns, dispatch) = self.a64_chain_insns(func, block);
        let last = insns.len().checked_sub(1)?;
        let aarch64::Opcode::Br { rn: dest } = insns[last].ins.opcode else {
            return None;
        };
        let site = Site {
            func,
            block,
            jump_site: insns[last].va,
        };

        // `add xD, xAnchor, xOff{, lsl #n}` — now decoded directly by the
        // aarch64 decoder (shifted- or extended-register form) rather than
        // left as `Unknown` for this module to re-parse.
        let ai = a64_last_def(&insns, last, dest)?;
        let add = a64_add_reg(&insns[ai].ins.opcode)?;
        if add.rd != dest {
            return None;
        }

        // `ldr{b,h,sw} wOff, [xTable, xIdx{, lsl #n}]` — now decoded directly
        // by the aarch64 decoder as a register-offset load rather than left
        // as `Unknown` for this module to re-parse.
        let li = a64_last_def(&insns, ai, add.rm)?;
        let (load_size, load_signed, load_rt, load_rn, load_off) = match insns[li].ins.opcode {
            aarch64::Opcode::LdrReg {
                size, rt, rn, off, ..
            } => (size, false, rt, rn, off),
            aarch64::Opcode::LdrsReg {
                size, rt, rn, off, ..
            } => (size, true, rt, rn, off),
            _ => return None,
        };
        // The index must be scaled by the element size. The effective left
        // shift is the access size when the `S` bit is set, else 0 — so a
        // byte table (size 0) matches whether or not `S` is set, exactly as
        // the raw-word check this replaced did.
        let load_shift = if load_off.scaled { load_size } else { 0 };
        if load_rt != add.rm || load_shift != load_size {
            return None;
        }
        if a64_last_def(&insns, li, load_off.rm).is_some_and(|d| d >= dispatch) {
            return None; // index recomputed in the dispatch block
        }

        let table_va = self.a64_addr_value(&insns, li, load_rn)?;
        let anchor = self.a64_addr_value(&insns, ai, add.rn)?;
        let element_size = 1u8 << load_size;
        let bound = self.a64_bound(&site, load_off.rm);

        let (kind, idiom) = match (element_size, load_signed, add.amount) {
            // Compressed table: unsigned byte/halfword entries, scaled by
            // 4 into an ADR-relative anchor.
            (1 | 2, false, 2) => (
                TableKind::ByteOffsetShifted { anchor, shift: 2 },
                Idiom::A64CompressedOffsetTable,
            ),
            // Word table: signed 4-byte offsets from the table base,
            // which is also the anchor.
            (4, true, 0) if anchor == table_va => (
                TableKind::SelfRelativeOffsets { base: table_va },
                Idiom::A64SelfRelativeWordTable,
            ),
            _ => return None,
        };
        self.build(Candidate {
            jump_site: site.jump_site,
            table_va,
            element_size,
            kind,
            bound,
            idiom,
        })
    }

    /// Entry count from the A64 range check guarding the dispatch block:
    /// `cmp <index>, #N` (`SUBS XZR, ...`) immediately followed by
    /// `b.hi`/`b.hs` (or `b.ls`/`b.lo` when the dispatch is on the taken
    /// edge).
    fn a64_bound(&self, site: &Site, index: u8) -> Option<usize> {
        for pred in site.func.blocks.values() {
            let cfg::Terminator::CondJump {
                taken, fallthrough, ..
            } = pred.terminator
            else {
                continue;
            };
            if taken == fallthrough || !pred.successors.contains(&site.block.start) {
                continue;
            }
            let insns = self.a64_block_insns(pred);
            let Some(last) = insns.len().checked_sub(1) else {
                continue;
            };
            let aarch64::Opcode::BCond { cond, .. } = insns[last].ins.opcode else {
                continue;
            };
            let Some(cmp) = last.checked_sub(1).map(|i| insns[i].ins.opcode) else {
                continue;
            };
            let aarch64::Opcode::SubImm {
                set_flags: true,
                rd: 31,
                rn,
                imm,
                ..
            } = cmp
            else {
                continue;
            };
            if rn != index {
                continue;
            }
            let n = imm as usize;
            let count = if fallthrough == site.block.start {
                match cond {
                    aarch64::Cond::Hi => n.checked_add(1)?, // idx <= n
                    aarch64::Cond::Cs => n,                 // idx <  n
                    _ => continue,
                }
            } else {
                match cond {
                    aarch64::Cond::Ls => n.checked_add(1)?, // idx <= n
                    aarch64::Cond::Cc => n,                 // idx <  n
                    _ => continue,
                }
            };
            if count > 0 {
                return Some(count);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::load;

    /// Image base of the synthetic PE fixture.
    const PE_BASE: u64 = 0x1_4000_0000;
    /// VA of the PE fixture's .text (RVA 0x1000, file 0x200, size 0x100).
    const PE_TEXT: u64 = PE_BASE + 0x1000;
    /// VA of the ELF fixture's .text (file 0x100, size 0x80).
    const ELF_TEXT: u64 = 0x40_1000;

    /// `synthetic_pe64` with `.text` painted from `(offset, bytes)`
    /// pairs relative to the section start.
    ///
    /// The base fixture points its import data directory into `.text`,
    /// which every fixture here overwrites; clear it so the eager import
    /// parse sees "no imports" instead of garbage.
    fn pe_fixture(patches: &[(usize, &[u8])]) -> Vec<u8> {
        let mut img = crate::pe::tests::synthetic_pe64();
        let dirs = 0x80 + 4 + 20 + 112;
        img[dirs + 8..dirs + 16].fill(0);
        for &(off, bytes) in patches {
            img[0x200 + off..0x200 + off + bytes.len()].copy_from_slice(bytes);
        }
        img
    }

    /// `synthetic_elf64` painted the same way, optionally retargeted to
    /// AArch64 (`e_machine` = EM_AARCH64).
    fn elf_fixture(aarch64_arch: bool, patches: &[(usize, &[u8])]) -> Vec<u8> {
        let mut img = crate::elf::tests::synthetic_elf64();
        if aarch64_arch {
            img[18..20].copy_from_slice(&183u16.to_le_bytes());
        }
        for &(off, bytes) in patches {
            img[0x100 + off..0x100 + off + bytes.len()].copy_from_slice(bytes);
        }
        img
    }

    /// A64 words at 4-byte offsets, as a `.text` patch list source.
    fn a64_words(words: &[(usize, u32)]) -> Vec<(usize, [u8; 4])> {
        words.iter().map(|&(o, w)| (o, w.to_le_bytes())).collect()
    }

    fn patches(owned: &[(usize, [u8; 4])]) -> Vec<(usize, &[u8])> {
        owned.iter().map(|(o, b)| (*o, &b[..])).collect()
    }

    /// Full pipeline: load -> cfg -> jump-table resolution.
    fn tables_of(img: &[u8]) -> Vec<JumpTable> {
        tables_with(img, &Config::default())
    }

    fn tables_with(img: &[u8], config: &Config) -> Vec<JumpTable> {
        let image = load(img).unwrap();
        let program = cfg::recover(image.as_ref()).unwrap();
        resolve_with(image.as_ref(), &program, config).unwrap()
    }

    fn i32b(v: i32) -> [u8; 4] {
        v.to_le_bytes()
    }

    // -----------------------------------------------------------------
    // x86-64
    // -----------------------------------------------------------------

    /// Idiom 1, with a `cmp`/`ja` range check bounding it at 4 entries.
    ///
    /// ```text
    /// +0x00 cmp    edi, 3
    /// +0x03 ja     +0x30                    (default case)
    /// +0x05 lea    rcx, [rip + 0x34]        -> table at +0x40
    /// +0x0c movsxd rax, dword [rcx + rdi*4]
    /// +0x10 add    rax, rcx
    /// +0x13 jmp    rax
    /// ```
    fn x86_self_relative_fixture() -> Vec<u8> {
        let table = [i32b(0x10), i32b(0x14), i32b(0x18), i32b(0x1C)].concat();
        pe_fixture(&[
            (0x00, &[0x83, 0xFF, 0x03]),
            (0x03, &[0x77, 0x2B]),
            (0x05, &[0x48, 0x8D, 0x0D, 0x34, 0x00, 0x00, 0x00]),
            (0x0C, &[0x48, 0x63, 0x04, 0xB9]),
            (0x10, &[0x48, 0x01, 0xC8]),
            (0x13, &[0xFF, 0xE0]),
            (0x30, &[0xC3]), // default
            (0x40, &table),
            (0x50, &[0xC3]),
            (0x54, &[0xC3]),
            (0x58, &[0xC3]),
            (0x5C, &[0xC3]),
        ])
    }

    #[test]
    fn x86_self_relative_offset_table_resolves() {
        let t = tables_of(&x86_self_relative_fixture());
        assert_eq!(t.len(), 1, "{t:#x?}");
        assert_eq!(
            t[0],
            JumpTable {
                jump_site: PE_TEXT + 0x13,
                table_va: PE_TEXT + 0x40,
                element_size: 4,
                kind: TableKind::SelfRelativeOffsets {
                    base: PE_TEXT + 0x40
                },
                entry_count: 4,
                targets: vec![
                    PE_TEXT + 0x50,
                    PE_TEXT + 0x54,
                    PE_TEXT + 0x58,
                    PE_TEXT + 0x5C
                ],
                bound: Bound::FromCompare,
                idiom: Idiom::X86RipRelativeOffsetTable,
            }
        );
        assert_eq!(
            successor_map(&t),
            BTreeMap::from([(PE_TEXT + 0x13, t[0].targets.clone())])
        );
    }

    /// Idiom 2: `lea` base plus `jmp qword [rcx + rdi*8]`.
    ///
    /// ```text
    /// +0x00 cmp edi, 3
    /// +0x03 ja  +0x20
    /// +0x05 lea rcx, [rip + 0x34]      -> table at +0x40
    /// +0x0c jmp qword [rcx + rdi*8]
    /// ```
    #[test]
    fn x86_absolute_pointer_table_resolves() {
        let table = [
            (PE_TEXT + 0x24).to_le_bytes(),
            (PE_TEXT + 0x25).to_le_bytes(),
            (PE_TEXT + 0x26).to_le_bytes(),
            (PE_TEXT + 0x27).to_le_bytes(),
        ]
        .concat();
        let img = pe_fixture(&[
            (0x00, &[0x83, 0xFF, 0x03]),
            (0x03, &[0x77, 0x1B]),
            (0x05, &[0x48, 0x8D, 0x0D, 0x34, 0x00, 0x00, 0x00]),
            (0x0C, &[0xFF, 0x24, 0xF9]),
            (0x20, &[0xC3]),
            (0x24, &[0xC3, 0xC3, 0xC3, 0xC3]),
            (0x40, &table),
        ]);

        let t = tables_of(&img);
        assert_eq!(t.len(), 1, "{t:#x?}");
        assert_eq!(
            t[0],
            JumpTable {
                jump_site: PE_TEXT + 0x0C,
                table_va: PE_TEXT + 0x40,
                element_size: 8,
                kind: TableKind::AbsolutePointers,
                entry_count: 4,
                targets: vec![
                    PE_TEXT + 0x24,
                    PE_TEXT + 0x25,
                    PE_TEXT + 0x26,
                    PE_TEXT + 0x27
                ],
                bound: Bound::FromCompare,
                idiom: Idiom::X86RipRelativePointerTable,
            }
        );
    }

    /// Idiom 3: non-PIC `jmp qword [rdi*8 + 0x401040]`, on ELF (whose
    /// low load address leaves the table VA encodable as a `disp32`).
    #[test]
    fn x86_static_base_pointer_table_resolves() {
        let table = [
            (ELF_TEXT + 0x24).to_le_bytes(),
            (ELF_TEXT + 0x25).to_le_bytes(),
            (ELF_TEXT + 0x26).to_le_bytes(),
            (ELF_TEXT + 0x27).to_le_bytes(),
        ]
        .concat();
        let img = elf_fixture(
            false,
            &[
                (0x00, &[0x83, 0xFF, 0x03]),
                (0x03, &[0x77, 0x1B]),
                (0x05, &[0xFF, 0x24, 0xFD, 0x40, 0x10, 0x40, 0x00]),
                (0x20, &[0xC3]),
                (0x24, &[0xC3, 0xC3, 0xC3, 0xC3]),
                (0x40, &table),
            ],
        );

        let t = tables_of(&img);
        assert_eq!(t.len(), 1, "{t:#x?}");
        assert_eq!(t[0].jump_site, ELF_TEXT + 0x05);
        assert_eq!(t[0].table_va, ELF_TEXT + 0x40);
        assert_eq!(t[0].kind, TableKind::AbsolutePointers);
        assert_eq!(t[0].idiom, Idiom::X86AbsolutePointerTable);
        assert_eq!(t[0].bound, Bound::FromCompare);
        assert_eq!(
            t[0].targets,
            [
                ELF_TEXT + 0x24,
                ELF_TEXT + 0x25,
                ELF_TEXT + 0x26,
                ELF_TEXT + 0x27
            ]
        );
    }

    // -----------------------------------------------------------------
    // AArch64
    // -----------------------------------------------------------------

    /// The clang compressed byte-table idiom (idiom 4).
    ///
    /// ```text
    /// +0x00 cmp  w0, #3
    /// +0x04 b.hi +0x20                  (default case)
    /// +0x08 adr  x8, +0x24              (anchor == table base)
    /// +0x0c ldrb w9, [x8, x0]
    /// +0x10 add  x10, x8, x9, lsl #2
    /// +0x14 br   x10
    /// +0x24 .byte 4, 5, 6, 7            -> +0x34, +0x38, +0x3c, +0x40
    /// ```
    fn a64_byte_table_fixture() -> Vec<u8> {
        let words = a64_words(&[
            (0x00, 0x7100_0C1F), // cmp w0, #3
            (0x04, 0x5400_00E8), // b.hi +0x1c
            (0x08, 0x1000_00E8), // adr x8, +0x1c
            (0x0C, 0x3860_6909), // ldrb w9, [x8, x0]
            (0x10, 0x8B09_090A), // add x10, x8, x9, lsl #2
            (0x14, 0xD61F_0140), // br x10
            (0x20, 0xD65F_03C0), // ret (default)
            (0x24, 0x0706_0504), // table: 4, 5, 6, 7
            (0x34, 0xD65F_03C0), // case 0
            (0x38, 0xD65F_03C0), // case 1
            (0x3C, 0xD65F_03C0), // case 2
            (0x40, 0xD65F_03C0), // case 3
        ]);
        elf_fixture(true, &patches(&words))
    }

    #[test]
    fn aarch64_compressed_byte_table_resolves() {
        let t = tables_of(&a64_byte_table_fixture());
        assert_eq!(t.len(), 1, "{t:#x?}");
        assert_eq!(
            t[0],
            JumpTable {
                jump_site: ELF_TEXT + 0x14,
                table_va: ELF_TEXT + 0x24,
                element_size: 1,
                kind: TableKind::ByteOffsetShifted {
                    anchor: ELF_TEXT + 0x24,
                    shift: 2,
                },
                entry_count: 4,
                targets: vec![
                    ELF_TEXT + 0x34,
                    ELF_TEXT + 0x38,
                    ELF_TEXT + 0x3C,
                    ELF_TEXT + 0x40
                ],
                bound: Bound::FromCompare,
                idiom: Idiom::A64CompressedOffsetTable,
            }
        );
        assert_eq!(
            t[0].resolved_successors(),
            [
                ELF_TEXT + 0x34,
                ELF_TEXT + 0x38,
                ELF_TEXT + 0x3C,
                ELF_TEXT + 0x40
            ]
        );
    }

    /// The LLVM `JumpTableDest32` idiom (idiom 5), with `ADRP`+`ADD`
    /// forming the table base and negative self-relative entries.
    ///
    /// ```text
    /// +0x00 cmp   w0, #3
    /// +0x04 b.hi  +0x20
    /// +0x08 adrp  x8, 0x401000
    /// +0x0c add   x8, x8, #0x40          -> table at +0x40
    /// +0x10 ldrsw x9, [x8, x0, lsl #2]
    /// +0x14 add   x9, x8, x9
    /// +0x18 br    x9
    /// ```
    #[test]
    fn aarch64_self_relative_word_table_resolves() {
        let words = a64_words(&[
            (0x00, 0x7100_0C1F), // cmp w0, #3
            (0x04, 0x5400_00E8), // b.hi +0x1c
            (0x08, 0x9000_0008), // adrp x8, 0x401000
            (0x0C, 0x9101_0108), // add x8, x8, #0x40
            (0x10, 0xB8A0_7909), // ldrsw x9, [x8, x0, lsl #2]
            (0x14, 0x8B09_0109), // add x9, x8, x9
            (0x18, 0xD61F_0120), // br x9
            (0x20, 0xD65F_03C0), // ret (default)
            (0x24, 0xD65F_03C0), // case 0
            (0x28, 0xD65F_03C0), // case 1
            (0x2C, 0xD65F_03C0), // case 2
            (0x30, 0xD65F_03C0), // case 3
            // Table at +0x40: offsets from +0x40 to +0x24..+0x30.
            (0x40, u32::from_le_bytes(i32b(-0x1C))),
            (0x44, u32::from_le_bytes(i32b(-0x18))),
            (0x48, u32::from_le_bytes(i32b(-0x14))),
            (0x4C, u32::from_le_bytes(i32b(-0x10))),
        ]);
        let t = tables_of(&elf_fixture(true, &patches(&words)));

        assert_eq!(t.len(), 1, "{t:#x?}");
        assert_eq!(
            t[0],
            JumpTable {
                jump_site: ELF_TEXT + 0x18,
                table_va: ELF_TEXT + 0x40,
                element_size: 4,
                kind: TableKind::SelfRelativeOffsets {
                    base: ELF_TEXT + 0x40
                },
                entry_count: 4,
                targets: vec![
                    ELF_TEXT + 0x24,
                    ELF_TEXT + 0x28,
                    ELF_TEXT + 0x2C,
                    ELF_TEXT + 0x30
                ],
                bound: Bound::FromCompare,
                idiom: Idiom::A64SelfRelativeWordTable,
            }
        );
    }

    // -----------------------------------------------------------------
    // Bounds safety, near misses, determinism
    // -----------------------------------------------------------------

    /// A compare-derived bound of 101 entries against a table with room
    /// for 4 before the end of `.text`, whose third entry points out of
    /// the image: clamped to the region, then truncated at the bad entry.
    #[test]
    fn oversized_bound_is_clamped_to_the_region_and_truncated() {
        let table = [i32b(-0xBC), i32b(-0xB8), i32b(0x1000), i32b(0x2000)].concat();
        let img = pe_fixture(&[
            (0x00, &[0x83, 0xFF, 0x64]), // cmp edi, 100  -> 101 entries
            (0x03, &[0x77, 0x2B]),       // ja +0x30
            (0x05, &[0x48, 0x8D, 0x0D, 0xE4, 0x00, 0x00, 0x00]), // lea rcx,[rip+0xE4] -> +0xF0
            (0x0C, &[0x48, 0x63, 0x04, 0xB9]),
            (0x10, &[0x48, 0x01, 0xC8]),
            (0x13, &[0xFF, 0xE0]),
            (0x30, &[0xC3]),
            (0x34, &[0xC3]),
            (0x38, &[0xC3]),
            (0xF0, &table),
        ]);

        let t = tables_of(&img);
        assert_eq!(t.len(), 1, "{t:#x?}");
        // .text ends at +0x100, so only 4 of the 101 entries fit.
        assert_eq!(t[0].entry_count, 4);
        assert_eq!(t[0].bound, Bound::FromCompare);
        // Entry 2 leaves the image: the table is truncated, not filtered.
        assert_eq!(t[0].targets, [PE_TEXT + 0x34, PE_TEXT + 0x38]);
    }

    /// With no recoverable range check the count is assumed, and the
    /// table is terminated by entry validation instead. Setting
    /// `require_bounds_check` rejects the candidate outright.
    #[test]
    fn missing_bounds_check_is_assumed_or_rejected() {
        // The self-relative fixture with `cmp`/`ja` replaced by nops, so
        // there is no guarded predecessor block at all.
        let mut img = x86_self_relative_fixture();
        img[0x200..0x205].fill(0x90);

        let t = tables_of(&img);
        assert_eq!(t.len(), 1, "{t:#x?}");
        assert_eq!(t[0].bound, Bound::Assumed);
        // 48 whole entries fit between the table and the end of .text...
        assert_eq!(t[0].entry_count, 48);
        // ...but validation stops at the first entry past the real table.
        assert_eq!(
            t[0].targets,
            [
                PE_TEXT + 0x50,
                PE_TEXT + 0x54,
                PE_TEXT + 0x58,
                PE_TEXT + 0x5C
            ]
        );

        let strict = Config {
            require_bounds_check: true,
            ..Config::default()
        };
        assert!(tables_with(&img, &strict).is_empty());
    }

    /// Idioms that almost match must yield nothing rather than a wrong
    /// table.
    #[test]
    fn almost_matching_idioms_yield_no_table() {
        // (a) The `lea` target is not in any mapped region.
        let mut img = x86_self_relative_fixture();
        img[0x200 + 0x08..0x200 + 0x0C].copy_from_slice(&0x7FFF_FFF0u32.to_le_bytes());
        assert!(tables_of(&img).is_empty(), "unmapped lea target");

        // (b) The register chain is broken: `add rax, rdx` where rdx was
        //     never the table base.
        let mut img = x86_self_relative_fixture();
        img[0x200 + 0x10..0x200 + 0x13].copy_from_slice(&[0x48, 0x01, 0xD0]);
        assert!(tables_of(&img).is_empty(), "broken register chain");

        // (c) The base register is clobbered between the `lea` and the
        //     load, so the base no longer resolves to the table.
        let img = pe_fixture(&[
            (0x00, &[0x83, 0xFF, 0x03]),
            (0x03, &[0x77, 0x2B]),
            (0x05, &[0x48, 0x8D, 0x0D, 0x34, 0x00, 0x00, 0x00]), // lea rcx,[rip+0x34]
            (0x0C, &[0x31, 0xC9]),                               // xor ecx, ecx
            (0x0E, &[0x48, 0x63, 0x04, 0xB9]),                   // movsxd rax,[rcx+rdi*4]
            (0x12, &[0x48, 0x01, 0xC8]),                         // add rax, rcx
            (0x15, &[0xFF, 0xE0]),                               // jmp rax
            (0x30, &[0xC3]),
            (0x40, &[i32b(0x10), i32b(0x14), i32b(0x18), i32b(0x1C)].concat()),
            (0x50, &[0xC3]),
            (0x54, &[0xC3]),
            (0x58, &[0xC3]),
            (0x5C, &[0xC3]),
        ]);
        assert!(tables_of(&img).is_empty(), "clobbered table base");

        // (d) A single validated target is a coincidence, not a switch:
        //     only entry 0 of the table survives validation.
        let mut img = x86_self_relative_fixture();
        img[0x200 + 0x44..0x200 + 0x48].copy_from_slice(&i32b(0x4000));
        assert!(tables_of(&img).is_empty(), "one target is not a table");

        // (e) An A64 `br` whose register comes from a plain `movz`, with
        //     no table arithmetic at all.
        let words = a64_words(&[
            (0x00, 0xD280_0020), // movz x0, #1
            (0x04, 0xD61F_0000), // br x0
        ]);
        assert!(tables_of(&elf_fixture(true, &patches(&words))).is_empty());

        // (f) The A64 byte-table idiom with the `lsl #2` scaling dropped
        //     (`add x10, x8, x9`): the entries would not be instruction
        //     offsets, so the idiom is not the one we recognize.
        let mut words = a64_words(&[
            (0x00, 0x7100_0C1F),
            (0x04, 0x5400_00E8),
            (0x08, 0x1000_00E8),
            (0x0C, 0x3860_6909),
            (0x10, 0x8B09_010A), // add x10, x8, x9   (no shift)
            (0x14, 0xD61F_0140),
            (0x20, 0xD65F_03C0),
            (0x24, 0x0706_0504),
        ]);
        words.extend(a64_words(&[
            (0x34, 0xD65F_03C0),
            (0x38, 0xD65F_03C0),
            (0x3C, 0xD65F_03C0),
            (0x40, 0xD65F_03C0),
        ]));
        assert!(tables_of(&elf_fixture(true, &patches(&words))).is_empty());

        // (g) The index register is redefined between the range check
        //     and the table load, so the check no longer bounds it.
        let img = pe_fixture(&[
            (0x00, &[0x83, 0xFF, 0x03]),                         // cmp edi, 3
            (0x03, &[0x77, 0x2B]),                               // ja +0x30
            (0x05, &[0x89, 0xCF]),                               // mov edi, ecx
            (0x07, &[0x48, 0x8D, 0x0D, 0x32, 0x00, 0x00, 0x00]), // lea rcx,[rip+0x32]
            (0x0E, &[0x48, 0x63, 0x04, 0xB9]),                   // movsxd rax,[rcx+rdi*4]
            (0x12, &[0x48, 0x01, 0xC8]),                         // add rax, rcx
            (0x15, &[0xFF, 0xE0]),                               // jmp rax
            (0x30, &[0xC3]),
            (0x40, &[i32b(0x10), i32b(0x14), i32b(0x18), i32b(0x1C)].concat()),
            (0x50, &[0xC3]),
            (0x54, &[0xC3]),
            (0x58, &[0xC3]),
            (0x5C, &[0xC3]),
        ]);
        assert!(tables_of(&img).is_empty(), "index clobbered after the check");

        // (h) The A64 word-table idiom whose `ldrsw` does not scale the
        //     index by the element size (`[x8, x0]`, not `[x8, x0, lsl
        //     #2]`): the index would not select whole entries.
        let words = a64_words(&[
            (0x00, 0x7100_0C1F),
            (0x04, 0x5400_00E8),
            (0x08, 0x9000_0008),
            (0x0C, 0x9101_0108),
            (0x10, 0xB8A0_6909), // ldrsw x9, [x8, x0]   (no lsl #2)
            (0x14, 0x8B09_0109),
            (0x18, 0xD61F_0120),
            (0x20, 0xD65F_03C0),
            (0x24, 0xD65F_03C0),
            (0x28, 0xD65F_03C0),
            (0x40, u32::from_le_bytes(i32b(-0x1C))),
            (0x44, u32::from_le_bytes(i32b(-0x18))),
        ]);
        assert!(tables_of(&elf_fixture(true, &patches(&words))).is_empty());

        // (i) The x86 offset-table load indexes by 1, not by the 4-byte
        //     element size (`[rcx + rdi*1]`).
        let mut img = x86_self_relative_fixture();
        img[0x200 + 0x0F] = 0x39; // SIB scale 4 -> 1
        assert!(tables_of(&img).is_empty(), "index not scaled by the element");

        // (j) An A64 word table whose `add` anchors on a *different*
        //     register than the table base: entries would be relative to
        //     something this pass has not established, so it is rejected
        //     rather than resolved against the wrong base.
        let words = a64_words(&[
            (0x00, 0x7100_0C1F), // cmp w0, #3
            (0x04, 0x5400_00E8), // b.hi +0x1c
            (0x08, 0x9000_0008), // adrp x8, 0x401000
            (0x0C, 0x9101_0108), // add x8, x8, #0x40   -> table
            (0x10, 0x1000_00A9), // adr x9, +0x14       -> other anchor
            (0x14, 0xB8A0_790A), // ldrsw x10, [x8, x0, lsl #2]
            (0x18, 0x8B0A_012A), // add x10, x9, x10
            (0x1C, 0xD61F_0140), // br x10
            (0x20, 0xD65F_03C0),
            (0x24, 0xD65F_03C0),
            (0x28, 0xD65F_03C0),
            (0x40, u32::from_le_bytes(i32b(-0x1C))),
            (0x44, u32::from_le_bytes(i32b(-0x18))),
        ]);
        assert!(tables_of(&elf_fixture(true, &patches(&words))).is_empty());
    }

    /// An assumed-bound table that runs off the end of the real one into
    /// zero padding: a zero self-relative offset points at the table's own
    /// base, which terminates the table even though that address is
    /// executable.
    #[test]
    fn self_referential_entries_terminate_an_assumed_table() {
        // Targets sit *before* the table, so the bytes after the last
        // real entry are plain zero padding.
        let img = pe_fixture(&[
            (0x00, &[0x90, 0x90, 0x90, 0x90, 0x90]),             // no range check
            (0x05, &[0x48, 0x8D, 0x0D, 0x34, 0x00, 0x00, 0x00]), // lea rcx,[rip+0x34]
            (0x0C, &[0x48, 0x63, 0x04, 0xB9]),
            (0x10, &[0x48, 0x01, 0xC8]),
            (0x13, &[0xFF, 0xE0]),
            (0x20, &[0xC3]),
            (0x24, &[0xC3]),
            (0x28, &[0xC3]),
            (0x40, &[i32b(-0x20), i32b(-0x1C), i32b(-0x18)].concat()),
        ]);

        let t = tables_of(&img);
        assert_eq!(t.len(), 1, "{t:#x?}");
        assert_eq!(t[0].bound, Bound::Assumed);
        assert_eq!(t[0].entry_count, 48); // room to the end of .text
        assert_eq!(
            t[0].targets,
            [PE_TEXT + 0x20, PE_TEXT + 0x24, PE_TEXT + 0x28]
        );
    }

    /// Every fixture, twice: same image, same tables, same order.
    #[test]
    fn resolution_is_deterministic() {
        let mut relaxed = x86_self_relative_fixture();
        relaxed[0x200..0x205].fill(0x90);
        for img in [
            x86_self_relative_fixture(),
            a64_byte_table_fixture(),
            relaxed,
            pe_fixture(&[]),
        ] {
            assert_eq!(tables_of(&img), tables_of(&img));
        }
    }

    /// Hostile bytes reach the pattern matcher: single-byte corruptions
    /// of a real idiom must never panic, and a table whose entry count is
    /// capped at zero must simply resolve nothing.
    #[test]
    fn hostile_input_never_panics() {
        for b in 0..=0xFFu8 {
            for off in [0x00usize, 0x05, 0x0C, 0x10, 0x13, 0x40] {
                let mut img = x86_self_relative_fixture();
                img[0x200 + off] = b;
                let _ = tables_of(&img);
            }
        }
        // A zero entry cap resolves nothing but stays well-defined.
        let capped = Config {
            max_entries: 0,
            ..Config::default()
        };
        assert!(tables_with(&x86_self_relative_fixture(), &capped).is_empty());
    }

    /// An architecture with no patterns is a typed error, matching
    /// [`cfg::recover`]'s behavior.
    #[test]
    fn unsupported_arch_is_a_typed_error() {
        let mut img = crate::elf::tests::synthetic_elf64();
        img[18..20].copy_from_slice(&0xF00u16.to_le_bytes());
        let image = load(&img).unwrap();
        let empty = cfg::Program {
            functions: BTreeMap::new(),
            call_graph: BTreeMap::new(),
            stats: cfg::Stats::default(),
        };
        assert!(matches!(
            resolve(image.as_ref(), &empty),
            Err(ParseError::Unsupported(_))
        ));
    }

    /// An import tail call (`jmp [IAT]`) is already resolved by cfg and
    /// must not be mistaken for a table dispatch.
    #[test]
    fn import_tail_calls_are_skipped() {
        let mut img = crate::pe::tests::with_imports();
        let opt = 0x80 + 4 + 20;
        img[opt + 16..opt + 20].copy_from_slice(&0x10A0u32.to_le_bytes());
        // 0x10A0: jmp [rip - 0x3E] -> IAT slot RVA 0x1068.
        img[0x2A0..0x2A6].copy_from_slice(&[0xFF, 0x25, 0xC2, 0xFF, 0xFF, 0xFF]);

        let image = load(&img).unwrap();
        let program = cfg::recover(image.as_ref()).unwrap();
        // Precondition: cfg really did name this an import tail call, so
        // the skip in `resolve_with` is the thing under test.
        let entry = PE_BASE + 0x10A0;
        assert_eq!(
            program.functions[&entry].blocks[&entry].terminator,
            cfg::Terminator::IndirectJump {
                import: Some("KERNEL32.dll!#7".into())
            }
        );
        assert!(resolve(image.as_ref(), &program).unwrap().is_empty());
    }

    // -----------------------------------------------------------------
    // resolve_folded: the recover → resolve fixpoint
    // -----------------------------------------------------------------

    fn folded_of(img: &[u8]) -> Folded {
        let image = load(img).unwrap();
        resolve_folded(image.as_ref()).unwrap()
    }

    /// The proven table folds end to end: the dispatch block's successors
    /// are exactly the table's targets, every target is walked into a
    /// block, and the returned tables are [`resolve`]'s over the folded
    /// program. Two rounds: one to prove, one to fold.
    #[test]
    fn resolve_folded_folds_a_proven_table_and_walks_its_cases() {
        let img = x86_self_relative_fixture();
        let f = folded_of(&img);
        assert_eq!((f.rounds, f.capped), (2, false));

        let func = &f.program.functions[&PE_TEXT];
        let cases = [PE_TEXT + 0x50, PE_TEXT + 0x54, PE_TEXT + 0x58, PE_TEXT + 0x5C];
        let dispatch = &func.blocks[&(PE_TEXT + 0x05)];
        assert_eq!(
            dispatch.terminator,
            cfg::Terminator::IndirectJump { import: None }
        );
        assert_eq!(dispatch.successors, cases);
        for va in cases {
            assert!(func.blocks.contains_key(&va), "case {va:#x} has no block");
        }
        assert_eq!(f.program.stats.tables_folded, 1);
        assert_eq!(f.program.stats.table_targets_dropped, 0);

        // The proof returned is the proof over the folded program.
        let image = load(&img).unwrap();
        assert_eq!(f.tables, resolve(image.as_ref(), &f.program).unwrap());
    }

    /// Two chained dispatches: the entry's table targets a case body that
    /// is itself a second dispatch with its own table — code recovery only
    /// reaches through the first fold, so the second table needs the
    /// fixpoint. Both fold, three rounds.
    ///
    /// ```text
    /// +0x00 cmp edi, 1 ; ja +0x30            table 1 at +0x48
    /// +0x05 lea/movsxd/add ; jmp rax         -> +0x60, +0x94
    /// +0x60 cmp esi, 1 ; ja +0x90            table 2 at +0xA0
    /// +0x65 lea/movsxd/add ; jmp rax         -> +0xB0, +0xB4
    /// ```
    fn nested_dispatch_fixture() -> Vec<u8> {
        pe_fixture(&[
            (0x00, &[0x83, 0xFF, 0x01]),                         // cmp edi, 1
            (0x03, &[0x77, 0x2B]),                               // ja +0x30
            (0x05, &[0x48, 0x8D, 0x0D, 0x3C, 0x00, 0x00, 0x00]), // lea rcx,[rip+0x3C]
            (0x0C, &[0x48, 0x63, 0x04, 0xB9]),                   // movsxd rax,[rcx+rdi*4]
            (0x10, &[0x48, 0x01, 0xC8]),                         // add rax, rcx
            (0x13, &[0xFF, 0xE0]),                               // jmp rax
            (0x30, &[0xC3]),                                     // default 1
            (0x48, &[i32b(0x18), i32b(0x4C)].concat()),          // table 1
            (0x60, &[0x83, 0xFE, 0x01]),                         // cmp esi, 1
            (0x63, &[0x77, 0x2B]),                               // ja +0x90
            (0x65, &[0x48, 0x8D, 0x15, 0x34, 0x00, 0x00, 0x00]), // lea rdx,[rip+0x34]
            (0x6C, &[0x48, 0x63, 0x04, 0xB2]),                   // movsxd rax,[rdx+rsi*4]
            (0x70, &[0x48, 0x01, 0xD0]),                         // add rax, rdx
            (0x73, &[0xFF, 0xE0]),                               // jmp rax
            (0x90, &[0xC3]),                                     // default 2
            (0x94, &[0xC3]),                                     // case 1.1
            (0xA0, &[i32b(0x10), i32b(0x14)].concat()),          // table 2
            (0xB0, &[0xC3]),                                     // case 2.0
            (0xB4, &[0xC3]),                                     // case 2.1
        ])
    }

    #[test]
    fn nested_tables_fold_to_a_fixpoint() {
        let f = folded_of(&nested_dispatch_fixture());
        assert_eq!((f.rounds, f.capped), (3, false));
        assert_eq!(f.tables.len(), 2, "{:#x?}", f.tables);

        let func = &f.program.functions[&PE_TEXT];
        assert_eq!(
            func.blocks[&(PE_TEXT + 0x05)].successors,
            [PE_TEXT + 0x60, PE_TEXT + 0x94]
        );
        assert_eq!(
            func.blocks[&(PE_TEXT + 0x65)].successors,
            [PE_TEXT + 0xB0, PE_TEXT + 0xB4]
        );
        for off in [0x60u64, 0x65, 0x90, 0x94, 0xB0, 0xB4] {
            assert!(
                func.blocks.contains_key(&(PE_TEXT + off)),
                "block +{off:#x} missing"
            );
        }
        assert_eq!(f.program.stats.tables_folded, 2);
    }

    /// An indirect jump no table proves stays successor-less through the
    /// fixpoint, and costs it nothing: one round, no tables.
    #[test]
    fn unproven_indirect_jumps_stay_successor_less() {
        let words = a64_words(&[
            (0x00, 0xD280_0020), // movz x0, #1
            (0x04, 0xD61F_0000), // br x0
        ]);
        let f = folded_of(&elf_fixture(true, &patches(&words)));
        assert_eq!((f.rounds, f.capped), (1, false));
        assert!(f.tables.is_empty());
        let block = &f.program.functions[&ELF_TEXT].blocks[&ELF_TEXT];
        assert_eq!(
            block.terminator,
            cfg::Terminator::IndirectJump { import: None }
        );
        assert!(block.successors.is_empty());
        assert_eq!(f.program.stats.tables_folded, 0);
    }

    /// The round cap stops the loop honestly: with two rounds allowed the
    /// nested fixture folds the first table, proves the second, and
    /// reports `capped` — the proofs the program has not folded are
    /// returned, never dropped.
    #[test]
    fn fold_round_cap_is_respected_and_visible() {
        let img = nested_dispatch_fixture();
        let image = load(&img).unwrap();
        let capped = Config {
            max_fold_rounds: 2,
            ..Config::default()
        };
        let f = resolve_folded_with(image.as_ref(), &cfg::Config::default(), &capped).unwrap();
        assert_eq!((f.rounds, f.capped), (2, true));
        assert_eq!(f.tables.len(), 2);
        let func = &f.program.functions[&PE_TEXT];
        // Table 1 folded; table 2 was only proven this round.
        assert!(!func.blocks[&(PE_TEXT + 0x05)].successors.is_empty());
        assert!(func.blocks[&(PE_TEXT + 0x65)].successors.is_empty());
        assert_eq!(f.program.stats.tables_folded, 1);

        // A zero cap is clamped to one round: plain recover + resolve.
        let zero = Config {
            max_fold_rounds: 0,
            ..Config::default()
        };
        let f = resolve_folded_with(image.as_ref(), &cfg::Config::default(), &zero).unwrap();
        assert_eq!((f.rounds, f.capped), (1, true));
        assert_eq!(f.program.stats.tables_folded, 0);
    }

    /// The fixpoint is deterministic: same image, same `Folded`, twice.
    #[test]
    fn resolve_folded_is_deterministic() {
        for img in [
            x86_self_relative_fixture(),
            nested_dispatch_fixture(),
            a64_byte_table_fixture(),
            pe_fixture(&[]),
        ] {
            assert_eq!(folded_of(&img), folded_of(&img));
        }
    }

    // -----------------------------------------------------------------
    // The split-block chain
    // -----------------------------------------------------------------

    /// Idiom 1 with the `lea` hoisted one block up, above the range
    /// check whose `ja` ends the predecessor block.
    ///
    /// ```text
    /// +0x00 lea    rcx, [rip + 0x39]        -> table at +0x40
    /// +0x07 cmp    edi, 3
    /// +0x0a ja     +0x30                    (default case)
    /// +0x0c movsxd rax, dword [rcx + rdi*4] (dispatch block)
    /// +0x10 add    rax, rcx
    /// +0x13 jmp    rax
    /// ```
    fn split_lea_offset_fixture() -> Vec<u8> {
        let table = [i32b(0x10), i32b(0x14), i32b(0x18), i32b(0x1C)].concat();
        pe_fixture(&[
            (0x00, &[0x48, 0x8D, 0x0D, 0x39, 0x00, 0x00, 0x00]),
            (0x07, &[0x83, 0xFF, 0x03]),
            (0x0A, &[0x77, 0x24]),
            (0x0C, &[0x48, 0x63, 0x04, 0xB9]),
            (0x10, &[0x48, 0x01, 0xC8]),
            (0x13, &[0xFF, 0xE0]),
            (0x30, &[0xC3]),
            (0x40, &table),
            (0x50, &[0xC3]),
            (0x54, &[0xC3]),
            (0x58, &[0xC3]),
            (0x5C, &[0xC3]),
        ])
    }

    #[test]
    fn split_lea_offset_table_resolves() {
        let t = tables_of(&split_lea_offset_fixture());
        assert_eq!(t.len(), 1, "{t:#x?}");
        assert_eq!(
            t[0],
            JumpTable {
                jump_site: PE_TEXT + 0x13,
                table_va: PE_TEXT + 0x40,
                element_size: 4,
                kind: TableKind::SelfRelativeOffsets {
                    base: PE_TEXT + 0x40
                },
                entry_count: 4,
                targets: vec![
                    PE_TEXT + 0x50,
                    PE_TEXT + 0x54,
                    PE_TEXT + 0x58,
                    PE_TEXT + 0x5C
                ],
                bound: Bound::FromCompare,
                idiom: Idiom::X86RipRelativeOffsetTable,
            }
        );
    }

    /// Idiom 2 with the `lea` one block up: `jmp qword [rcx + rdi*8]`
    /// alone in the dispatch block.
    #[test]
    fn split_lea_pointer_table_resolves() {
        let table = [
            (PE_TEXT + 0x24).to_le_bytes(),
            (PE_TEXT + 0x25).to_le_bytes(),
            (PE_TEXT + 0x26).to_le_bytes(),
            (PE_TEXT + 0x27).to_le_bytes(),
        ]
        .concat();
        let img = pe_fixture(&[
            (0x00, &[0x48, 0x8D, 0x0D, 0x39, 0x00, 0x00, 0x00]), // lea rcx,[rip+0x39]
            (0x07, &[0x83, 0xFF, 0x03]),                         // cmp edi, 3
            (0x0A, &[0x77, 0x14]),                               // ja +0x20
            (0x0C, &[0xFF, 0x24, 0xF9]),                         // jmp [rcx+rdi*8]
            (0x20, &[0xC3]),
            (0x24, &[0xC3, 0xC3, 0xC3, 0xC3]),
            (0x40, &table),
        ]);
        let t = tables_of(&img);
        assert_eq!(t.len(), 1, "{t:#x?}");
        assert_eq!(t[0].jump_site, PE_TEXT + 0x0C);
        assert_eq!(t[0].idiom, Idiom::X86RipRelativePointerTable);
        assert_eq!(t[0].bound, Bound::FromCompare);
        assert_eq!(
            t[0].targets,
            [PE_TEXT + 0x24, PE_TEXT + 0x25, PE_TEXT + 0x26, PE_TEXT + 0x27]
        );
    }

    /// The `lea` two single-predecessor blocks up.
    ///
    /// ```text
    /// +0x00 lea rcx, [rip + 0x59]  ; jmp +0x10   -> table at +0x60
    /// +0x10 cmp edi, 3             ; ja +0x30
    /// +0x15 movsxd/add/jmp rax                   (dispatch block)
    /// ```
    fn split_lea_two_blocks_fixture() -> Vec<u8> {
        let table = [i32b(0x10), i32b(0x14), i32b(0x18), i32b(0x1C)].concat();
        pe_fixture(&[
            (0x00, &[0x48, 0x8D, 0x0D, 0x59, 0x00, 0x00, 0x00]),
            (0x07, &[0xEB, 0x07]),
            (0x10, &[0x83, 0xFF, 0x03]),
            (0x13, &[0x77, 0x1B]),
            (0x15, &[0x48, 0x63, 0x04, 0xB9]),
            (0x19, &[0x48, 0x01, 0xC8]),
            (0x1C, &[0xFF, 0xE0]),
            (0x30, &[0xC3]),
            (0x60, &table),
            (0x70, &[0xC3]),
            (0x74, &[0xC3]),
            (0x78, &[0xC3]),
            (0x7C, &[0xC3]),
        ])
    }

    /// Two blocks up proves; a one-block budget refuses it; a zero
    /// budget refuses even the one-block split. All-or-nothing at the
    /// documented cap, with the same validated table whenever it proves.
    #[test]
    fn walk_depth_cap_is_respected() {
        let img = split_lea_two_blocks_fixture();
        let t = tables_of(&img);
        assert_eq!(t.len(), 1, "{t:#x?}");
        assert_eq!(t[0].jump_site, PE_TEXT + 0x1C);
        assert_eq!(t[0].table_va, PE_TEXT + 0x60);
        assert_eq!(t[0].bound, Bound::FromCompare);
        assert_eq!(
            t[0].targets,
            [PE_TEXT + 0x70, PE_TEXT + 0x74, PE_TEXT + 0x78, PE_TEXT + 0x7C]
        );

        let one = Config {
            max_walk_blocks: 1,
            ..Config::default()
        };
        assert!(tables_with(&img, &one).is_empty(), "past the bound");

        let zero = Config {
            max_walk_blocks: 0,
            ..Config::default()
        };
        assert!(
            tables_with(&split_lea_offset_fixture(), &zero).is_empty(),
            "walking disabled"
        );
    }

    /// A clobber of the base register in a block between the `lea` and
    /// the dispatch breaks the chain: no table.
    #[test]
    fn split_base_clobber_between_blocks_refuses() {
        let img = pe_fixture(&[
            (0x00, &[0x48, 0x8D, 0x0D, 0x59, 0x00, 0x00, 0x00]), // lea rcx,[rip+0x59]
            (0x07, &[0xEB, 0x07]),                               // jmp +0x10
            (0x10, &[0x31, 0xC9]),                               // xor ecx, ecx
            (0x12, &[0x83, 0xFF, 0x03]),                         // cmp edi, 3
            (0x15, &[0x77, 0x19]),                               // ja +0x30
            (0x17, &[0x48, 0x63, 0x04, 0xB9]),                   // movsxd rax,[rcx+rdi*4]
            (0x1B, &[0x48, 0x01, 0xC8]),                         // add rax, rcx
            (0x1E, &[0xFF, 0xE0]),                               // jmp rax
            (0x30, &[0xC3]),
            (0x60, &[i32b(0x10), i32b(0x14), i32b(0x18), i32b(0x1C)].concat()),
            (0x70, &[0xC3]),
            (0x74, &[0xC3]),
            (0x78, &[0xC3]),
            (0x7C, &[0xC3]),
        ]);
        assert!(tables_of(&img).is_empty(), "clobbered across blocks");
    }

    /// A join whose predecessors both come from the branch of a diamond:
    /// neither dominates the dispatch, so the chain refuses — even
    /// though both arms load the same table base.
    #[test]
    fn split_ambiguous_join_refuses() {
        let img = pe_fixture(&[
            (0x00, &[0x83, 0xFF, 0x03]),                         // cmp edi, 3
            (0x03, &[0x77, 0x2B]),                               // ja +0x30
            (0x05, &[0x83, 0xFF, 0x01]),                         // cmp edi, 1
            (0x08, &[0x74, 0x0E]),                               // je +0x18
            (0x0A, &[0x48, 0x8D, 0x0D, 0x2F, 0x00, 0x00, 0x00]), // lea rcx,[rip+0x2F]
            (0x11, &[0xEB, 0x0C]),                               // jmp +0x1f
            (0x18, &[0x48, 0x8D, 0x0D, 0x21, 0x00, 0x00, 0x00]), // lea rcx,[rip+0x21]
            (0x1F, &[0x48, 0x63, 0x04, 0xB9]),                   // movsxd rax,[rcx+rdi*4]
            (0x23, &[0x48, 0x01, 0xC8]),                         // add rax, rcx
            (0x26, &[0xFF, 0xE0]),                               // jmp rax
            (0x30, &[0xC3]),
            (0x40, &[i32b(0x20), i32b(0x24), i32b(0x28), i32b(0x2C)].concat()),
            (0x60, &[0xC3]),
            (0x64, &[0xC3]),
            (0x68, &[0xC3]),
            (0x6C, &[0xC3]),
        ]);
        assert!(tables_of(&img).is_empty(), "no dominating predecessor");
    }

    /// The /bin/ls shape: the `lea` in a preheader above a dispatch
    /// loop, the loop head a join of preheader and latch, and a `call`
    /// between them. A callee-saved base (rbx) survives the call and
    /// proves; the identical shape on caller-saved rcx refuses.
    ///
    /// ```text
    /// +0x00 lea    rbx, [rip + 0x79]        -> table at +0x80
    /// +0x07 call   +0x70                    (loop head; latch at +0x30)
    /// +0x0c cmp    eax, 3
    /// +0x0f ja     +0x30
    /// +0x11 movsxd rcx, [rbx + rax*4]       (dispatch block)
    /// +0x15 add    rcx, rbx
    /// +0x18 jmp    rcx
    /// +0x30 jmp    +0x07                    (latch)
    /// ```
    fn split_loop_call_fixture() -> Vec<u8> {
        let table = [i32b(-0x40), i32b(-0x3C), i32b(-0x38), i32b(-0x34)].concat();
        pe_fixture(&[
            (0x00, &[0x48, 0x8D, 0x1D, 0x79, 0x00, 0x00, 0x00]),
            (0x07, &[0xE8, 0x64, 0x00, 0x00, 0x00]),
            (0x0C, &[0x83, 0xF8, 0x03]),
            (0x0F, &[0x77, 0x1F]),
            (0x11, &[0x48, 0x63, 0x0C, 0x83]),
            (0x15, &[0x48, 0x01, 0xD9]),
            (0x18, &[0xFF, 0xE1]),
            (0x30, &[0xEB, 0xD5]),
            (0x40, &[0xC3]),
            (0x44, &[0xC3]),
            (0x48, &[0xC3]),
            (0x4C, &[0xC3]),
            (0x70, &[0xC3]),
            (0x80, &table),
        ])
    }

    #[test]
    fn split_loop_head_join_with_callee_saved_base_resolves() {
        let t = tables_of(&split_loop_call_fixture());
        assert_eq!(t.len(), 1, "{t:#x?}");
        assert_eq!(
            t[0],
            JumpTable {
                jump_site: PE_TEXT + 0x18,
                table_va: PE_TEXT + 0x80,
                element_size: 4,
                kind: TableKind::SelfRelativeOffsets {
                    base: PE_TEXT + 0x80
                },
                entry_count: 4,
                targets: vec![
                    PE_TEXT + 0x40,
                    PE_TEXT + 0x44,
                    PE_TEXT + 0x48,
                    PE_TEXT + 0x4C
                ],
                bound: Bound::FromCompare,
                idiom: Idiom::X86RipRelativeOffsetTable,
            }
        );
    }

    /// The same loop-and-call shape with the base in caller-saved rcx:
    /// the call clobbers it, so the chain finds no `lea` and refuses.
    #[test]
    fn split_caller_saved_base_across_call_refuses() {
        let img = pe_fixture(&[
            (0x00, &[0x48, 0x8D, 0x0D, 0x79, 0x00, 0x00, 0x00]), // lea rcx,[rip+0x79]
            (0x07, &[0xE8, 0x64, 0x00, 0x00, 0x00]),             // call +0x70
            (0x0C, &[0x83, 0xFF, 0x03]),                         // cmp edi, 3
            (0x0F, &[0x77, 0x1F]),                               // ja +0x30
            (0x11, &[0x48, 0x63, 0x04, 0xB9]),                   // movsxd rax,[rcx+rdi*4]
            (0x15, &[0x48, 0x01, 0xC8]),                         // add rax, rcx
            (0x18, &[0xFF, 0xE0]),                               // jmp rax
            (0x30, &[0xEB, 0xD5]),                               // jmp +0x07
            (0x40, &[0xC3]),
            (0x44, &[0xC3]),
            (0x48, &[0xC3]),
            (0x4C, &[0xC3]),
            (0x70, &[0xC3]),
            (0x80, &[i32b(-0x40), i32b(-0x3C), i32b(-0x38), i32b(-0x34)].concat()),
        ]);
        assert!(tables_of(&img).is_empty(), "call clobbers a caller-saved base");
    }

    /// The A64 word-table idiom with the whole `adrp`+`add` address
    /// formation one block up, above the range check.
    ///
    /// ```text
    /// +0x00 adrp  x8, 0x401000
    /// +0x04 add   x8, x8, #0x40            -> table at +0x40
    /// +0x08 cmp   w0, #3
    /// +0x0c b.hi  +0x18                    (default case)
    /// +0x10 ldrsw x9, [x8, x0, lsl #2]     (dispatch block)
    /// +0x14 add   x9, x8, x9
    /// +0x18 br    x9
    /// ```
    #[test]
    fn split_a64_word_table_resolves() {
        let words = a64_words(&[
            (0x00, 0x9000_0008), // adrp x8, 0x401000
            (0x04, 0x9101_0108), // add x8, x8, #0x40
            (0x08, 0x7100_0C1F), // cmp w0, #3
            (0x0C, 0x5400_00C8), // b.hi +0x18
            (0x10, 0xB8A0_7909), // ldrsw x9, [x8, x0, lsl #2]
            (0x14, 0x8B09_0109), // add x9, x8, x9
            (0x18, 0xD61F_0120), // br x9
            (0x24, 0xD65F_03C0), // ret (default)
            (0x28, 0xD65F_03C0), // case 0
            (0x2C, 0xD65F_03C0), // case 1
            (0x30, 0xD65F_03C0), // case 2
            (0x34, 0xD65F_03C0), // case 3
            (0x40, u32::from_le_bytes(i32b(-0x18))),
            (0x44, u32::from_le_bytes(i32b(-0x14))),
            (0x48, u32::from_le_bytes(i32b(-0x10))),
            (0x4C, u32::from_le_bytes(i32b(-0x0C))),
        ]);
        let t = tables_of(&elf_fixture(true, &patches(&words)));
        assert_eq!(t.len(), 1, "{t:#x?}");
        assert_eq!(
            t[0],
            JumpTable {
                jump_site: ELF_TEXT + 0x18,
                table_va: ELF_TEXT + 0x40,
                element_size: 4,
                kind: TableKind::SelfRelativeOffsets {
                    base: ELF_TEXT + 0x40
                },
                entry_count: 4,
                targets: vec![
                    ELF_TEXT + 0x28,
                    ELF_TEXT + 0x2C,
                    ELF_TEXT + 0x30,
                    ELF_TEXT + 0x34
                ],
                bound: Bound::FromCompare,
                idiom: Idiom::A64SelfRelativeWordTable,
            }
        );
    }

    /// The widening is strictly additive: every single-block fixture
    /// resolves identically with the walk enabled (default) and disabled
    /// (`max_walk_blocks: 0`).
    #[test]
    fn split_walk_is_strictly_additive_on_single_block_idioms() {
        let mut relaxed = x86_self_relative_fixture();
        relaxed[0x200..0x205].fill(0x90);
        let zero = Config {
            max_walk_blocks: 0,
            ..Config::default()
        };
        for img in [
            x86_self_relative_fixture(),
            nested_dispatch_fixture(),
            a64_byte_table_fixture(),
            relaxed,
            pe_fixture(&[]),
        ] {
            let with_walk = tables_of(&img);
            assert_eq!(with_walk, tables_with(&img, &zero));
            assert!(!with_walk.is_empty() || img == pe_fixture(&[]));
        }
    }

    /// Split-block proofs feed the recover → resolve fixpoint exactly
    /// like single-block ones: the loop fixture folds in two rounds,
    /// deterministically, and its dispatch block gains the case edges.
    #[test]
    fn split_tables_fold_to_a_fixpoint() {
        let img = split_loop_call_fixture();
        let f = folded_of(&img);
        assert_eq!((f.rounds, f.capped), (2, false));
        assert_eq!(f.tables.len(), 1);
        let func = &f.program.functions[&PE_TEXT];
        assert_eq!(
            func.blocks[&(PE_TEXT + 0x11)].successors,
            [PE_TEXT + 0x40, PE_TEXT + 0x44, PE_TEXT + 0x48, PE_TEXT + 0x4C]
        );
        assert_eq!(f.program.stats.tables_folded, 1);
        assert_eq!(f, folded_of(&img));

        // The split fixtures are deterministic under the plain resolver
        // too.
        for img in [
            split_lea_offset_fixture(),
            split_lea_two_blocks_fixture(),
            split_loop_call_fixture(),
        ] {
            assert_eq!(tables_of(&img), tables_of(&img));
        }
    }
}
