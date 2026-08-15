//! Cross-reference (xref) recovery: code→code and code→data.
//!
//! [`compute`] consumes a recovered [`cfg::Program`] and the [`Image`] it
//! came from, walks every instruction of every basic block, and records
//! each reference the instruction makes as an [`Xref`] — a `(from, to,
//! kind)` triple. The result is indexed both ways, so the two questions
//! an analyst actually asks are single lookups:
//!
//! - *what does this instruction reference?* — [`Xrefs::refs_from`]
//! - *who references this address?* — [`Xrefs::refs_to`]
//!
//! A `to` address is joined against [`Image::symbols`] and
//! [`Image::import_slots`] at compute time, so a reference can be
//! rendered as `→ printf` without carrying the image around
//! ([`Xrefs::to_label`]).
//!
//! # What is exact and what is a heuristic
//!
//! Exactness is a property of the *reference form*, not of the target, and
//! every emitted [`Xref`] falls in one of two tiers.
//!
//! **Exact** — the target is computed from the encoding alone, with no
//! assumption about run-time state, and is emitted whether or not it
//! lands in a mapped region:
//!
//! - Direct calls and direct (conditional) branches, from the decoder's
//!   [`Flow`]: [`XrefKind::Call`], [`XrefKind::Branch`],
//!   [`XrefKind::CondBranch`].
//! - x86-64 `[rip + disp32]` memory operands (including `lea`) and
//!   absolute `[disp32]` operands with neither base nor index.
//! - AArch64 `ADR`, `ADRP` (the page base), and `LDR` literal.
//!
//! **Heuristic** — the target is only *plausibly* a reference, so it is
//! emitted solely when it lands inside a region the image maps. Each is
//! documented at its emission site below:
//!
//! - x86-64 immediates that look like data pointers
//!   ([`XrefKind::ImmediatePointer`]): required to land in a mapped,
//!   **non-executable** region, so a loop count or a small constant is
//!   never mistaken for a reference. Immediates of control-transfer
//!   instructions (branch displacements) are never scanned.
//! - x86-64 `[index*scale + disp32]` operands with no base — a table
//!   base, e.g. a switch jump table: emitted only when `disp32` is
//!   mapped.
//! - AArch64 `ADRP`+`ADD`/`LDR`/`STR`/`LDP`/`STP` page-pair
//!   reconstruction (see below).
//!
//! # AArch64 ADRP pairing, and its precision limits
//!
//! A64 materializes an address in two instructions: `ADRP Xd, page`
//! computes a 4 KiB-aligned page base, and a following `ADD Xd, Xn, #lo12`
//! (or a `LDR`/`STR`/`LDP`/`STP` with `#lo12` as its offset) completes it.
//! Recovery tracks, per basic block, which register holds which page base
//! and reconstructs the absolute target when the pair is completed. The
//! reference is attributed to the *completing* instruction. An `ADRP`
//! whose page base is never consumed is reported on its own, as an
//! [`XrefKind::AddressOf`] to the page base.
//!
//! Known limits, all deliberate under- or over-approximations:
//!
//! - Pairing never crosses a basic-block boundary and never spans more
//!   than [`Config::adrp_window`] instructions, which bounds how far a
//!   stale page base can survive.
//! - A page base is dropped when its register is overwritten by an
//!   instruction this crate *models* (`ADR`/`ADRP`, `ADD`/`SUB`
//!   immediate, `MOV[NZK]`, the loads, and — per AAPCS64 — the
//!   caller-saved registers at a `BL`/`BLR`). [`aarch64::Opcode::Unknown`]
//!   encodings (a large part of the A64 space, including register-form
//!   `MOV`) are *not* treated as clobbers: doing so would suppress most
//!   real pairings. A false pairing is therefore possible when an
//!   unmodeled instruction rewrites the page register within the window.
//! - `x31` is never paired: it is `XZR` for `ADRP`/`ADD` and `SP` for the
//!   loads, and the two must not be confused.
//! - There is no [`XrefKind::ImmediatePointer`] on AArch64: A64 has no
//!   large immediates, and `MOVZ`/`MOVK` address materialization is not
//!   reassembled.
//!
//! # ISA layering
//!
//! Unlike [`crate::cfg`], this module names the two rich ISA decoders
//! ([`crate::x86`], [`crate::aarch64`]) directly and deliberately: an
//! xref *is* an operand-level fact, and the trait layer's
//! [`crate::model::Decoded::mem_target`] only exposes the memory operand
//! of an indirect branch. Container formats stay out entirely — the image
//! is only ever seen through [`Image`].
//!
//! # Determinism and resource caps
//!
//! Every container is a B-tree keyed by address, blocks are walked in
//! (function, block) address order, and blocks shared between functions
//! are walked once, so the same `(image, program)` always yields the same
//! [`Xrefs`]. [`Config`] caps the total xrefs retained and the total
//! instructions decoded; hitting either stops the walk cleanly and sets
//! its flag in [`Stats`], so a truncated result is always identifiable.
//! Nothing here panics on any input: address arithmetic is checked or
//! wrapping, and an undecodable instruction simply ends the walk of its
//! block.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{ParseError, Result};
use crate::model::{Arch, Flow, Image, decoder_for};
use crate::{aarch64, cfg, x86};

/// Enough bytes for any single instruction on the supported ISAs
/// (x86-64 caps at 15; A64 is always 4).
const MAX_INSN_BYTES: u64 = 16;

/// Resource caps for [`compute_with`], and the knobs on the heuristic
/// reference forms. See the module docs for the semantics of each cap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Maximum number of distinct xrefs retained. Reaching it stops the
    /// walk and sets [`Stats::xref_cap_hit`].
    pub max_xrefs: usize,
    /// Maximum instructions decoded across the whole walk. Reaching it
    /// stops the walk and sets [`Stats::instruction_cap_hit`].
    pub max_instructions: usize,
    /// Scan immediate operands for values that look like data pointers
    /// (x86-64 only; see the module docs). Enabled by default.
    pub scan_immediates: bool,
    /// AArch64 only: how many instructions an `ADRP` page base may
    /// survive before it is retired unpaired. Also bounded by the end of
    /// the basic block.
    pub adrp_window: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_xrefs: 1_000_000,
            max_instructions: 2_000_000,
            scan_immediates: true,
            adrp_window: 8,
        }
    }
}

/// What one instruction does with the address it references.
///
/// The first three are code→code; the rest are code→data. Read/write
/// classification follows the operand's role in the instruction, not a
/// dataflow analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum XrefKind {
    /// A call: `to` is the callee's entry VA for a direct call, or the
    /// VA of the *slot* for a call through a statically addressed memory
    /// cell (x86 `call [rip+disp]`), which [`Xrefs::to_import`] resolves
    /// to the imported name.
    Call,
    /// An unconditional direct branch, including a tail call through an
    /// import slot (x86 `jmp [rip+disp]`), whose `to` is the slot VA.
    Branch,
    /// A conditional direct branch; `to` is the taken edge. The
    /// fall-through edge is control flow, not a reference, and is not
    /// recorded.
    CondBranch,
    /// The instruction loads from `to`.
    Read,
    /// The instruction stores to `to`. Read-modify-write forms
    /// (`add [m], r`, `xchg`, `cmpxchg`, `xadd`, `bts`/`btr`/`btc`) are
    /// reported as writes.
    Write,
    /// The instruction computes the address `to` without accessing it:
    /// x86 `lea`, AArch64 `ADR`/`ADRP` and a completed `ADRP`+`ADD` pair.
    AddressOf,
    /// An immediate operand whose value lands in a mapped, non-executable
    /// region — a probable pointer to data. The most speculative kind;
    /// see the module docs.
    ImmediatePointer,
}

impl XrefKind {
    /// Whether this kind is a control-flow (code→code) reference.
    pub fn is_code(self) -> bool {
        matches!(
            self,
            XrefKind::Call | XrefKind::Branch | XrefKind::CondBranch
        )
    }

    /// Whether this kind is a data (code→data) reference.
    pub fn is_data(self) -> bool {
        !self.is_code()
    }

    /// Short label for listings (`call`, `read`, `addr`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            XrefKind::Call => "call",
            XrefKind::Branch => "branch",
            XrefKind::CondBranch => "cbranch",
            XrefKind::Read => "read",
            XrefKind::Write => "write",
            XrefKind::AddressOf => "addr",
            XrefKind::ImmediatePointer => "imm",
        }
    }
}

/// One cross-reference: instruction at `from` references address `to`.
///
/// Ordered by `(from, to, kind)`, which is the order both indexes present
/// their entries in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Xref {
    /// VA of the referencing instruction.
    pub from: u64,
    /// Referenced VA.
    pub to: u64,
    pub kind: XrefKind,
}

/// Recovery statistics and truncation flags. Either flag means the
/// [`Xrefs`] is a documented under-approximation, not a complete result.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stats {
    /// Distinct xrefs retained.
    pub xrefs: usize,
    /// Instructions decoded (each shared block walked once).
    pub instructions: usize,
    /// Basic blocks walked.
    pub blocks: usize,
    /// [`Config::max_xrefs`] was hit; at least one xref was dropped and
    /// the walk stopped there.
    pub xref_cap_hit: bool,
    /// [`Config::max_instructions`] was hit; the walk stopped there.
    pub instruction_cap_hit: bool,
}

/// The recovered cross-reference view of an image: the same set of
/// [`Xref`]s indexed by source and by target, plus the name join needed
/// to display a target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xrefs {
    /// Outgoing references keyed by the referencing instruction's VA;
    /// each value is sorted by `(to, kind)` and deduplicated.
    pub by_from: BTreeMap<u64, Vec<Xref>>,
    /// Incoming references keyed by the referenced VA — the reverse
    /// index that answers "who references this address?". Each value is
    /// sorted by `(from, kind)` and deduplicated.
    pub by_to: BTreeMap<u64, Vec<Xref>>,
    pub stats: Stats,
    /// VA → symbol name, from [`Image::symbols`].
    symbols: BTreeMap<u64, String>,
    /// Import slot VA → imported name, from [`Image::import_slots`].
    imports: BTreeMap<u64, String>,
}

impl Xrefs {
    /// Everything the instruction at `va` references, sorted by
    /// `(to, kind)`. Empty when the address references nothing.
    pub fn refs_from(&self, va: u64) -> &[Xref] {
        self.by_from.get(&va).map_or(&[], Vec::as_slice)
    }

    /// Everything that references `va`, sorted by `(from, kind)` — the
    /// "who points here?" query. Empty when nothing references it.
    pub fn refs_to(&self, va: u64) -> &[Xref] {
        self.by_to.get(&va).map_or(&[], Vec::as_slice)
    }

    /// Every xref, in `(from, to, kind)` order.
    pub fn iter(&self) -> impl Iterator<Item = &Xref> {
        self.by_from.values().flatten()
    }

    /// Number of distinct xrefs.
    pub fn len(&self) -> usize {
        self.stats.xrefs
    }

    /// Whether no xref was recovered.
    pub fn is_empty(&self) -> bool {
        self.stats.xrefs == 0
    }

    /// The symbol named at `va`, if any. When several symbols share a VA
    /// the first in [`Image::symbols`] order (by VA, then name) wins.
    pub fn symbol_at(&self, va: u64) -> Option<&str> {
        self.symbols.get(&va).map(String::as_str)
    }

    /// The import whose slot is at `va`, if any.
    pub fn import_at(&self, va: u64) -> Option<&str> {
        self.imports.get(&va).map(String::as_str)
    }

    /// [`Xrefs::symbol_at`] for a reference's target.
    pub fn to_symbol(&self, xref: &Xref) -> Option<&str> {
        self.symbol_at(xref.to)
    }

    /// [`Xrefs::import_at`] for a reference's target — the join that
    /// turns `call [rip+disp]` into `→ printf`.
    pub fn to_import(&self, xref: &Xref) -> Option<&str> {
        self.import_at(xref.to)
    }

    /// The best display name for a reference's target: the imported name
    /// when the target is an import slot, else the symbol name, else
    /// `None`.
    pub fn to_label(&self, xref: &Xref) -> Option<&str> {
        self.to_import(xref).or_else(|| self.to_symbol(xref))
    }
}

/// Recover cross-references from `program`'s instructions with the
/// default [`Config`].
///
/// `program` must be the [`cfg::Program`] recovered from `image`;
/// blocks that do not map into the image are skipped rather than
/// reported.
///
/// Fails with [`ParseError::Unsupported`] when the image's architecture
/// has no decoder ([`Arch::Other`]) — the same contract as
/// [`cfg::recover`]. Never panics on any input.
pub fn compute(image: &dyn Image, program: &cfg::Program) -> Result<Xrefs> {
    compute_with(image, program, &Config::default())
}

/// [`compute`] with caller-supplied caps and heuristic settings.
pub fn compute_with(image: &dyn Image, program: &cfg::Program, config: &Config) -> Result<Xrefs> {
    let arch = image.arch();
    if decoder_for(arch).is_none() {
        return Err(ParseError::Unsupported(format!(
            "cross-reference recovery: no decoder for architecture {arch:?}"
        )));
    }

    // Mapped ranges as (start, end, executable), in address order.
    let mut regions: Vec<(u64, u64, bool)> = image
        .regions()
        .iter()
        .filter(|r| r.size > 0)
        .map(|r| (r.va, r.va.saturating_add(r.size), r.perms.x))
        .collect();
    regions.sort_unstable();
    regions.dedup();

    let mut walk = Walk {
        image,
        bytes: image.bytes(),
        arch,
        config,
        regions,
        xrefs: BTreeSet::new(),
        stats: Stats::default(),
        capped: false,
    };

    // Blocks shared between functions (recovery copies them per function)
    // are walked once, keyed by their exact extent.
    let mut seen: BTreeSet<(u64, u64)> = BTreeSet::new();
    'walk: for func in program.functions.values() {
        for block in func.blocks.values() {
            if walk.capped {
                break 'walk;
            }
            if seen.insert((block.start, block.end)) {
                walk.walk_block(block);
            }
        }
    }

    let mut by_from: BTreeMap<u64, Vec<Xref>> = BTreeMap::new();
    let mut by_to: BTreeMap<u64, Vec<Xref>> = BTreeMap::new();
    for x in &walk.xrefs {
        by_from.entry(x.from).or_default().push(*x);
        by_to.entry(x.to).or_default().push(*x);
    }
    let mut stats = walk.stats;
    stats.xrefs = walk.xrefs.len();

    let mut symbols: BTreeMap<u64, String> = BTreeMap::new();
    for s in image.symbols() {
        symbols.entry(s.va).or_insert(s.name);
    }
    let mut imports: BTreeMap<u64, String> = BTreeMap::new();
    for s in image.import_slots() {
        imports.entry(s.slot_va).or_insert(s.name);
    }

    Ok(Xrefs {
        by_from,
        by_to,
        stats,
        symbols,
        imports,
    })
}

/// Whole-image walk state.
struct Walk<'a> {
    image: &'a dyn Image,
    bytes: &'a [u8],
    arch: Arch,
    config: &'a Config,
    /// Mapped `[start, end)` ranges with their executable bit, sorted.
    regions: Vec<(u64, u64, bool)>,
    xrefs: BTreeSet<Xref>,
    stats: Stats,
    /// A cap was hit: stop the whole walk.
    capped: bool,
}

/// A live `ADRP` page base awaiting the instruction that completes it.
struct PageBase {
    /// The 4 KiB-aligned page address the `ADRP` computed.
    page: u64,
    /// VA of the `ADRP` itself.
    va: u64,
    /// Index of the `ADRP` within its block, for window expiry.
    index: usize,
    /// Whether some later instruction has already completed this base.
    used: bool,
}

/// Per-block AArch64 pairing state: register number → live page base.
#[derive(Default)]
struct A64State {
    pages: BTreeMap<u8, PageBase>,
}

impl Walk<'_> {
    /// Whether `va` lies in any mapped region.
    fn is_mapped(&self, va: u64) -> bool {
        self.regions.iter().any(|&(s, e, _)| va >= s && va < e)
    }

    /// Whether `va` lies in a mapped, non-executable region — the
    /// conservative test the immediate heuristic uses.
    fn is_data(&self, va: u64) -> bool {
        self.regions
            .iter()
            .any(|&(s, e, x)| !x && va >= s && va < e)
    }

    /// Record one reference, honoring [`Config::max_xrefs`]. Dropping a
    /// genuinely new xref sets the flag and stops the whole walk, so the
    /// truncated result stays a prefix of the full one in walk order.
    fn push(&mut self, from: u64, to: u64, kind: XrefKind) {
        if self.capped {
            return;
        }
        let xref = Xref { from, to, kind };
        if self.xrefs.contains(&xref) {
            return;
        }
        if self.xrefs.len() >= self.config.max_xrefs {
            self.stats.xref_cap_hit = true;
            self.capped = true;
            return;
        }
        self.xrefs.insert(xref);
    }

    /// Record the code→code reference of a decoded instruction's flow,
    /// if it has one. Fall-through and indirect edges are not references.
    fn push_flow(&mut self, va: u64, flow: Flow) {
        match flow {
            Flow::Call(t) => self.push(va, t, XrefKind::Call),
            Flow::Jump(t) => self.push(va, t, XrefKind::Branch),
            Flow::CondJump(t) => self.push(va, t, XrefKind::CondBranch),
            _ => {}
        }
    }

    /// Decode `[block.start, block.end)` linearly and emit every xref its
    /// instructions produce.
    ///
    /// The window handed to the decoder is clamped to the block, the file
    /// buffer, and the longest supported encoding, so a decode can never
    /// read past the block. Any decode failure, unmapped byte, or VA
    /// wraparound ends the block's walk without a guess.
    fn walk_block(&mut self, block: &cfg::BasicBlock) {
        self.stats.blocks += 1;
        let all = self.bytes;
        let mut state = A64State::default();
        let mut cur = block.start;
        let mut index = 0usize;
        while cur < block.end {
            if self.capped {
                break;
            }
            if self.stats.instructions >= self.config.max_instructions {
                self.stats.instruction_cap_hit = true;
                self.capped = true;
                break;
            }
            let Some(off) = self.image.va_to_offset(cur) else {
                break;
            };
            let window = (block.end - cur)
                .min(MAX_INSN_BYTES)
                .min(all.len().saturating_sub(off) as u64) as usize;
            if window == 0 {
                break;
            }
            let code = &all[off..off + window];
            let length = match self.arch {
                Arch::X86_64 => self.x86_insn(code, cur),
                Arch::Aarch64 => self.a64_insn(code, cur, &mut state, index),
                // Unreachable: `compute_with` rejects undecodable arches.
                Arch::Other => None,
            };
            let Some(length) = length else {
                break;
            };
            self.stats.instructions += 1;
            let Some(next) = cur.checked_add(u64::from(length)) else {
                break;
            };
            if next <= cur {
                break; // a zero-length decode would not terminate
            }
            cur = next;
            index = index.saturating_add(1);
        }
        // Page bases still live at the end of the block never got a
        // completing instruction: report them on the ADRP itself.
        let live: Vec<u8> = state.pages.keys().copied().collect();
        for reg in live {
            self.retire(&mut state, reg);
        }
    }

    /// One x86-64 instruction: its flow reference plus every operand
    /// reference. Returns the instruction length, or `None` if it does
    /// not decode.
    fn x86_insn(&mut self, code: &[u8], va: u64) -> Option<u8> {
        let ins = x86::decode(code, va).ok()?;
        self.push_flow(va, ins.flow);
        // `[rip + disp]` resolves against the *next* instruction's VA
        // (SDM Vol 2 §2.2.1.6).
        let next = va.wrapping_add(u64::from(ins.length));
        for (i, operand) in ins.operands.iter().enumerate() {
            match *operand {
                x86::Operand::Mem {
                    base,
                    index,
                    disp,
                    rip_relative,
                    ..
                } => {
                    // A base register makes the address run-time state:
                    // nothing static to reference.
                    if base.is_some() {
                        continue;
                    }
                    let target = if rip_relative {
                        next.wrapping_add(disp as u64)
                    } else {
                        disp as u64
                    };
                    // `[index*scale + disp]`: `disp` is a table base only
                    // if it is a real address — heuristic, hence gated.
                    if index.is_some() && !self.is_mapped(target) {
                        continue;
                    }
                    let kind = x86_mem_kind(&ins, i);
                    self.push(va, target, kind);
                }
                x86::Operand::Imm(value) => {
                    // Branch displacements are flow, not data, and are
                    // already covered by `push_flow`.
                    if !self.config.scan_immediates || ins.flow != Flow::Sequential {
                        continue;
                    }
                    // Both the sign-extended value and its low 32 bits:
                    // a 32-bit address encoded as a negative imm32 is
                    // still an address. Duplicates collapse in the set.
                    for candidate in [value as u64, (value as u64) & 0xFFFF_FFFF] {
                        if candidate != 0 && self.is_data(candidate) {
                            self.push(va, candidate, XrefKind::ImmediatePointer);
                        }
                    }
                }
                x86::Operand::Reg(_) | x86::Operand::Xmm(_) => {}
            }
        }
        Some(ins.length)
    }

    /// One AArch64 instruction: its flow reference, its own pc-relative
    /// references, and the `ADRP` page-pair bookkeeping.
    fn a64_insn(&mut self, code: &[u8], va: u64, st: &mut A64State, index: usize) -> Option<u8> {
        let ins = aarch64::decode(code, va).ok()?;
        self.push_flow(va, ins.flow);

        // Retire page bases that have outlived the pairing window.
        let window = self.config.adrp_window;
        let expired: Vec<u8> = st
            .pages
            .iter()
            .filter(|(_, p)| index.saturating_sub(p.index) > window)
            .map(|(&reg, _)| reg)
            .collect();
        for reg in expired {
            self.retire(st, reg);
        }

        match ins.opcode {
            aarch64::Opcode::Adrp { rd, target } => {
                self.retire(st, rd);
                if rd == 31 {
                    // XZR: the page base is discarded, so it can never be
                    // completed — report it as computed.
                    self.push(va, target, XrefKind::AddressOf);
                } else {
                    st.pages.insert(
                        rd,
                        PageBase {
                            page: target,
                            va,
                            index,
                            used: false,
                        },
                    );
                }
            }
            aarch64::Opcode::Adr { rd, target } => {
                self.push(va, target, XrefKind::AddressOf);
                self.retire(st, rd);
            }
            aarch64::Opcode::LdrLit { rt, target, .. } => {
                self.push(va, target, XrefKind::Read);
                self.retire(st, rt);
            }
            aarch64::Opcode::AddImm {
                sf, rd, rn, imm, ..
            } => {
                if sf {
                    self.complete(st, va, rn, i64::from(imm), XrefKind::AddressOf);
                }
                self.retire(st, rd);
            }
            aarch64::Opcode::SubImm {
                sf, rd, rn, imm, ..
            } => {
                if sf {
                    self.complete(st, va, rn, -i64::from(imm), XrefKind::AddressOf);
                }
                self.retire(st, rd);
            }
            aarch64::Opcode::Ldr { rt, rn, mode, .. } => {
                self.complete(st, va, rn, access_offset(mode), XrefKind::Read);
                self.retire_base(st, rn, mode);
                self.retire(st, rt);
            }
            aarch64::Opcode::Str {
                rt: _, rn, mode, ..
            } => {
                self.complete(st, va, rn, access_offset(mode), XrefKind::Write);
                self.retire_base(st, rn, mode);
            }
            aarch64::Opcode::Ldp {
                rt, rt2, rn, mode, ..
            } => {
                self.complete(st, va, rn, access_offset(mode), XrefKind::Read);
                self.retire_base(st, rn, mode);
                self.retire(st, rt);
                self.retire(st, rt2);
            }
            aarch64::Opcode::Stp { rn, mode, .. } => {
                self.complete(st, va, rn, access_offset(mode), XrefKind::Write);
                self.retire_base(st, rn, mode);
            }
            aarch64::Opcode::Movn { rd, .. }
            | aarch64::Opcode::Movz { rd, .. }
            | aarch64::Opcode::Movk { rd, .. } => self.retire(st, rd),
            // AAPCS64: a call clobbers x0-x18 and the link register.
            aarch64::Opcode::Bl { .. } | aarch64::Opcode::Blr { .. } => {
                for reg in (0..=18u8).chain(std::iter::once(30)) {
                    self.retire(st, reg);
                }
            }
            // Everything else either writes no register or is unmodeled;
            // see the module docs on `Opcode::Unknown`.
            _ => {}
        }
        Some(aarch64::Instruction::SIZE as u8)
    }

    /// Complete an `ADRP` page base held in `rn` with `offset`, emitting
    /// the reconstructed absolute reference from the completing
    /// instruction at `va`. A no-op when `rn` holds no page base, and
    /// never applied to `x31` (`XZR`/`SP`).
    fn complete(&mut self, st: &mut A64State, va: u64, rn: u8, offset: i64, kind: XrefKind) {
        if rn == 31 {
            return;
        }
        let Some(page) = st.pages.get_mut(&rn) else {
            return;
        };
        page.used = true;
        let target = page.page.wrapping_add(offset as u64);
        self.push(va, target, kind);
    }

    /// Drop the page base held in `reg`, reporting it as an
    /// [`XrefKind::AddressOf`] on its `ADRP` if nothing ever completed it.
    fn retire(&mut self, st: &mut A64State, reg: u8) {
        if let Some(page) = st.pages.remove(&reg)
            && !page.used
        {
            self.push(page.va, page.page, XrefKind::AddressOf);
        }
    }

    /// Writeback addressing modes update the base register.
    fn retire_base(&mut self, st: &mut A64State, rn: u8, mode: aarch64::AddrMode) {
        if !matches!(mode, aarch64::AddrMode::Offset(_)) {
            self.retire(st, rn);
        }
    }
}

/// Byte offset applied to the base register by a load/store's addressing
/// mode. Post-indexed accesses use the base unmodified.
fn access_offset(mode: aarch64::AddrMode) -> i64 {
    match mode {
        aarch64::AddrMode::Offset(off) | aarch64::AddrMode::PreIndex(off) => off,
        aarch64::AddrMode::PostIndex(_) => 0,
    }
}

/// Classify the memory operand at position `i` of an x86 instruction.
///
/// Indirect branches through a memory cell reference the *slot*, so they
/// keep their control-flow kind; `lea` computes without accessing; and
/// the Intel-order destination operand of a writing mnemonic is a write.
fn x86_mem_kind(ins: &x86::Instruction, i: usize) -> XrefKind {
    match ins.flow {
        Flow::IndirectCall => XrefKind::Call,
        Flow::IndirectJump => XrefKind::Branch,
        _ if ins.opcode == x86::Opcode::Lea => XrefKind::AddressOf,
        _ if i == 0 && x86_writes_destination(ins.opcode) => XrefKind::Write,
        _ => XrefKind::Read,
    }
}

/// Whether the mnemonic writes its first (Intel-order destination)
/// operand. Read-modify-write mnemonics count as writes; `cmp`, `test`,
/// `bt` and `push` do not write theirs at all.
fn x86_writes_destination(opcode: x86::Opcode) -> bool {
    use x86::Opcode::*;
    matches!(
        opcode,
        Add | Or
            | Adc
            | Sbb
            | And
            | Sub
            | Xor
            | Mov
            | Movsx
            | Movzx
            | Movsxd
            | Xchg
            | Inc
            | Dec
            | Not
            | Neg
            | Pop
            | Setcc(_)
            | Cmov(_)
            | Bts
            | Btr
            | Btc
            | Cmpxchg
            | Xadd
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::tests::synthetic_elf64;
    use crate::model::load;
    use crate::pe::tests::{synthetic_pe64, with_imports};

    /// Image base of the synthetic PE fixtures.
    const PE_BASE: u64 = 0x1_4000_0000;
    /// Entry VA of the synthetic PE fixtures (RVA 0x1000, file 0x200).
    const PE_ENTRY: u64 = PE_BASE + 0x1000;
    /// VA of the `.data` region [`pe_with_data`] adds (RVA 0x2000).
    const PE_DATA: u64 = PE_BASE + 0x2000;
    /// VA of the synthetic ELF `.text` (== its entry point, file 0x100).
    const ELF_TEXT: u64 = 0x40_1000;

    /// Recover control flow, then cross-references, with default caps.
    fn xrefs_of(img: &[u8]) -> Xrefs {
        xrefs_with(img, &Config::default())
    }

    fn xrefs_with(img: &[u8], config: &Config) -> Xrefs {
        let image = load(img).unwrap();
        let program = cfg::recover(image.as_ref()).unwrap();
        compute_with(image.as_ref(), &program, config).unwrap()
    }

    fn xref(from: u64, to: u64, kind: XrefKind) -> Xref {
        Xref { from, to, kind }
    }

    /// `synthetic_pe64` plus a second, read/write `.data` section at RVA
    /// 0x2000 (file 0x400). The base fixture's import directory points
    /// into `.text`, which these tests fill with code, so it is dropped.
    fn pe_with_data() -> Vec<u8> {
        let mut img = synthetic_pe64();
        let coff = 0x80 + 4;
        let opt = coff + 20;
        let dirs = opt + 112;
        img[dirs + 8..dirs + 16].fill(0); // no import directory
        img[coff + 2..coff + 4].copy_from_slice(&2u16.to_le_bytes()); // 2 sections
        img[opt + 56..opt + 60].copy_from_slice(&0x3000u32.to_le_bytes()); // size of image

        let sec = opt + 0xF0 + 40; // second section header
        img[sec..sec + 5].copy_from_slice(b".data");
        img[sec + 8..sec + 12].copy_from_slice(&0x100u32.to_le_bytes()); // virtual size
        img[sec + 12..sec + 16].copy_from_slice(&0x2000u32.to_le_bytes()); // RVA
        img[sec + 16..sec + 20].copy_from_slice(&0x200u32.to_le_bytes()); // raw size
        img[sec + 20..sec + 24].copy_from_slice(&0x400u32.to_le_bytes()); // raw pointer
        // initialized data | read | write
        img[sec + 36..sec + 40].copy_from_slice(&0xC000_0040u32.to_le_bytes());
        img.resize(0x600, 0);
        img
    }

    /// [`pe_with_data`] with hand-assembled code at the entry point:
    /// every code→data form this pass recognizes on x86-64, then a
    /// direct call to a second function.
    fn pe_data_ref_fixture() -> Vec<u8> {
        let mut img = pe_with_data();
        let code: &[u8] = &[
            // +0x00 (7): lea rax, [rip+0xFF9]      -> PE_DATA
            0x48, 0x8D, 0x05, 0xF9, 0x0F, 0x00, 0x00,
            // +0x07 (6): mov ecx, [rip+0x1003]     -> PE_DATA+0x10
            0x8B, 0x0D, 0x03, 0x10, 0x00, 0x00,
            // +0x0D (6): mov [rip+0x100D], edx     -> PE_DATA+0x20
            0x89, 0x15, 0x0D, 0x10, 0x00, 0x00,
            // +0x13 (10): mov rdi, 0x140002030     -> PE_DATA+0x30
            0x48, 0xBF, 0x30, 0x20, 0x00, 0x40, 0x01, 0x00, 0x00, 0x00,
            // +0x1D (5): call +0xE                 -> PE_ENTRY+0x30
            0xE8, 0x0E, 0x00, 0x00, 0x00, //
            // +0x22 (1): ret
            0xC3,
        ];
        img[0x200..0x200 + code.len()].copy_from_slice(code);
        img[0x230] = 0xC3; // RVA 0x1030: the callee, a lone `ret`
        img
    }

    #[test]
    fn x86_memory_operands_and_immediates_become_data_xrefs() {
        let x = xrefs_of(&pe_data_ref_fixture());
        let e = PE_ENTRY;

        // Exact forms: rip-relative lea / load / store.
        assert_eq!(x.refs_from(e), [xref(e, PE_DATA, XrefKind::AddressOf)]);
        assert_eq!(
            x.refs_from(e + 0x07),
            [xref(e + 0x07, PE_DATA + 0x10, XrefKind::Read)]
        );
        assert_eq!(
            x.refs_from(e + 0x0D),
            [xref(e + 0x0D, PE_DATA + 0x20, XrefKind::Write)]
        );
        // Heuristic form: an immediate landing in the .data region.
        assert_eq!(
            x.refs_from(e + 0x13),
            [xref(e + 0x13, PE_DATA + 0x30, XrefKind::ImmediatePointer)]
        );
        // code->code: the direct call, and nothing from the `ret`.
        assert_eq!(
            x.refs_from(e + 0x1D),
            [xref(e + 0x1D, e + 0x30, XrefKind::Call)]
        );
        assert!(x.refs_from(e + 0x22).is_empty());

        assert_eq!(x.len(), 5);
        assert!(!x.is_empty());
        assert_eq!(x.iter().filter(|r| r.kind.is_data()).count(), 4);
        assert_eq!(x.iter().filter(|r| r.kind.is_code()).count(), 1);
        assert!(!x.stats.xref_cap_hit && !x.stats.instruction_cap_hit);
    }

    #[test]
    fn reverse_index_answers_who_references_this_address() {
        let mut img = pe_data_ref_fixture();
        // A second reader of PE_DATA+0x10 at RVA 0x1030 (where the callee
        // was): mov esi, [rip+0xFDA] then ret.
        img[0x230..0x237].copy_from_slice(&[0x8B, 0x35, 0xDA, 0x0F, 0x00, 0x00, 0xC3]);
        let x = xrefs_of(&img);
        let e = PE_ENTRY;

        // Both readers of the same datum, in source-address order.
        assert_eq!(
            x.refs_to(PE_DATA + 0x10),
            [
                xref(e + 0x07, PE_DATA + 0x10, XrefKind::Read),
                xref(e + 0x30, PE_DATA + 0x10, XrefKind::Read),
            ]
        );
        // The callee's call site is found by the same query.
        assert_eq!(
            x.refs_to(e + 0x30),
            [xref(e + 0x1D, e + 0x30, XrefKind::Call)]
        );
        // Nothing references the entry point or an unrelated address.
        assert!(x.refs_to(e).is_empty());
        assert!(x.refs_to(0xDEAD_BEEF).is_empty());
    }

    #[test]
    fn x86_table_base_is_only_taken_when_mapped() {
        // An absolute `disp32` cannot reach the PE fixture's image base,
        // so this one uses the low-address ELF fixture.
        let mut img = synthetic_elf64();
        let code: &[u8] = &[
            // mov eax, [rcx*4 + 0x401060]: no base, mapped disp.
            0x8B, 0x04, 0x8D, 0x60, 0x10, 0x40, 0x00,
            // mov eax, [rcx*4 + 0x11223344]: no base, unmapped disp.
            0x8B, 0x04, 0x8D, 0x44, 0x33, 0x22, 0x11, //
            0xC3,
        ];
        img[0x100..0x100 + code.len()].copy_from_slice(code);
        let x = xrefs_of(&img);

        assert_eq!(
            x.refs_from(ELF_TEXT),
            [xref(ELF_TEXT, ELF_TEXT + 0x60, XrefKind::Read)]
        );
        assert!(x.refs_from(ELF_TEXT + 7).is_empty());
    }

    /// `with_imports` plus code that calls an IAT slot directly (the same
    /// shape [`crate::cfg`] uses to resolve import calls).
    fn import_call_fixture() -> Vec<u8> {
        let mut img = with_imports();
        // Repoint the entry RVA from 0x1000 (import data) to 0x10A0.
        let opt = 0x80 + 4 + 20;
        img[opt + 16..opt + 20].copy_from_slice(&0x10A0u32.to_le_bytes());
        let code: &[u8] = &[
            // 0x10A0: call [rip - 0x46] -> IAT slot RVA 0x1060
            0xFF, 0x15, 0xBA, 0xFF, 0xFF, 0xFF, //
            // 0x10A6: ret
            0xC3,
        ];
        img[0x2A0..0x2A0 + code.len()].copy_from_slice(code); // RVA 0x10A0
        img
    }

    #[test]
    fn import_slot_reference_resolves_to_the_imported_name() {
        let x = xrefs_of(&import_call_fixture());
        let from = PE_BASE + 0x10A0;
        let slot = PE_BASE + 0x1060;

        // An indirect call through a statically addressed cell references
        // the slot, and keeps its call kind.
        let refs = x.refs_from(from);
        assert_eq!(refs, [xref(from, slot, XrefKind::Call)]);
        assert_eq!(x.refs_to(slot), refs);
        // The join that renders it as "-> KERNEL32.dll!ExitProcess".
        assert_eq!(x.to_import(&refs[0]), Some("KERNEL32.dll!ExitProcess"));
        assert_eq!(x.to_label(&refs[0]), Some("KERNEL32.dll!ExitProcess"));
        assert_eq!(x.to_symbol(&refs[0]), None); // PE exports only, none here
        assert_eq!(x.import_at(slot), Some("KERNEL32.dll!ExitProcess"));
        assert_eq!(x.import_at(slot + 1), None);
    }

    /// The x86-64 ELF fixture with a direct call from `main` to `helper`.
    fn elf_call_fixture() -> Vec<u8> {
        let mut img = synthetic_elf64();
        // 0x401000: call +0x1B -> 0x401020 (`helper`); 0x401005: ret.
        img[0x100..0x106].copy_from_slice(&[0xE8, 0x1B, 0x00, 0x00, 0x00, 0xC3]);
        img
    }

    #[test]
    fn call_target_resolves_to_a_symbol_name() {
        let x = xrefs_of(&elf_call_fixture());
        let refs = x.refs_from(ELF_TEXT);

        assert_eq!(refs, [xref(ELF_TEXT, ELF_TEXT + 0x20, XrefKind::Call)]);
        assert_eq!(x.to_symbol(&refs[0]), Some("helper"));
        assert_eq!(x.to_label(&refs[0]), Some("helper"));
        assert_eq!(x.symbol_at(ELF_TEXT), Some("main"));
        assert_eq!(x.symbol_at(ELF_TEXT + 1), None);
    }

    /// The ELF fixture rebuilt as AArch64 with `ADRP` pairs: a completed
    /// `ADRP`+`ADD`, a completed `ADRP`+`LDR`, and an unpaired `ADRP`.
    fn aarch64_adrp_fixture() -> Vec<u8> {
        let mut img = synthetic_elf64();
        img[18..20].copy_from_slice(&183u16.to_le_bytes()); // EM_AARCH64
        let words: &[(usize, u32)] = &[
            (0x00, 0xB000_0000), // adrp x0, +1 page   -> 0x40_2000
            (0x04, 0x9100_C000), // add  x0, x0, #0x30 -> 0x40_2030
            (0x08, 0xB000_0001), // adrp x1, +1 page   -> 0x40_2000
            (0x0C, 0xF940_0C22), // ldr  x2, [x1,#0x18]-> 0x40_2018
            (0x10, 0xD000_0003), // adrp x3, +2 pages  -> 0x40_3000 (unpaired)
            (0x14, 0x9400_0007), // bl   +0x1c         -> 0x40_1030
            (0x18, 0xD65F_03C0), // ret
            (0x20, 0xD65F_03C0), // helper: ret
            (0x30, 0xD65F_03C0), // callee: ret
        ];
        for &(off, word) in words {
            img[0x100 + off..0x100 + off + 4].copy_from_slice(&word.to_le_bytes());
        }
        img
    }

    #[test]
    fn aarch64_adrp_pairs_reconstruct_absolute_targets() {
        let x = xrefs_of(&aarch64_adrp_fixture());
        let t = ELF_TEXT;

        // The completing instruction owns the reconstructed reference;
        // the ADRP itself contributes nothing once it is consumed.
        assert!(x.refs_from(t).is_empty());
        assert_eq!(
            x.refs_from(t + 4),
            [xref(t + 4, 0x40_2030, XrefKind::AddressOf)]
        );
        assert!(x.refs_from(t + 8).is_empty());
        assert_eq!(
            x.refs_from(t + 0x0C),
            [xref(t + 0x0C, 0x40_2018, XrefKind::Read)]
        );
        // The unpaired ADRP is reported on itself, at its page base.
        assert_eq!(
            x.refs_from(t + 0x10),
            [xref(t + 0x10, 0x40_3000, XrefKind::AddressOf)]
        );
        // And A64 code->code still comes from the flow layer.
        assert_eq!(
            x.refs_from(t + 0x14),
            [xref(t + 0x14, t + 0x30, XrefKind::Call)]
        );

        // The reverse index reaches the completing instruction.
        assert_eq!(
            x.refs_to(0x40_2030),
            [xref(t + 4, 0x40_2030, XrefKind::AddressOf)]
        );
    }

    #[test]
    fn aarch64_page_base_does_not_pair_across_a_clobber() {
        let mut img = synthetic_elf64();
        img[18..20].copy_from_slice(&183u16.to_le_bytes()); // EM_AARCH64
        let words: &[(usize, u32)] = &[
            (0x00, 0xB000_0000), // adrp x0, +1 page -> 0x40_2000
            (0x04, 0xD280_0020), // movz x0, #1      (clobbers x0)
            (0x08, 0x9100_C000), // add  x0, x0, #0x30 (no live page base)
            (0x0C, 0xD65F_03C0), // ret
        ];
        for &(off, word) in words {
            img[0x100 + off..0x100 + off + 4].copy_from_slice(&word.to_le_bytes());
        }
        let x = xrefs_of(&img);

        // The clobber retires the base, which is then reported unpaired
        // on the ADRP; the later ADD reconstructs nothing.
        assert_eq!(
            x.refs_from(ELF_TEXT),
            [xref(ELF_TEXT, 0x40_2000, XrefKind::AddressOf)]
        );
        assert!(x.refs_from(ELF_TEXT + 8).is_empty());
    }

    #[test]
    fn recovery_is_deterministic() {
        for img in [
            pe_data_ref_fixture(),
            import_call_fixture(),
            elf_call_fixture(),
            aarch64_adrp_fixture(),
        ] {
            assert_eq!(xrefs_of(&img), xrefs_of(&img));
        }
    }

    #[test]
    fn caps_truncate_cleanly() {
        let img = pe_data_ref_fixture();
        let full = xrefs_of(&img);
        assert_eq!(full.len(), 5);

        // Xref cap: exactly the cap is retained, and the flag is set.
        let capped = xrefs_with(
            &img,
            &Config {
                max_xrefs: 2,
                ..Config::default()
            },
        );
        assert!(capped.stats.xref_cap_hit);
        assert!(!capped.stats.instruction_cap_hit);
        assert_eq!(capped.len(), 2);
        assert_eq!(capped.by_from.values().flatten().count(), 2);
        assert_eq!(capped.by_to.values().flatten().count(), 2);

        // Instruction cap: only the first instruction is examined.
        let capped = xrefs_with(
            &img,
            &Config {
                max_instructions: 1,
                ..Config::default()
            },
        );
        assert!(capped.stats.instruction_cap_hit);
        assert_eq!(capped.stats.instructions, 1);
        assert_eq!(
            capped.iter().copied().collect::<Vec<_>>(),
            [xref(PE_ENTRY, PE_DATA, XrefKind::AddressOf)]
        );

        // A zero cap yields an empty, still well-formed result.
        let none = xrefs_with(
            &img,
            &Config {
                max_xrefs: 0,
                ..Config::default()
            },
        );
        assert!(none.is_empty() && none.stats.xref_cap_hit);
    }

    #[test]
    fn immediate_scanning_is_optional_and_conservative() {
        let img = pe_data_ref_fixture();
        let off = xrefs_with(
            &img,
            &Config {
                scan_immediates: false,
                ..Config::default()
            },
        );
        assert!(off.refs_from(PE_ENTRY + 0x13).is_empty());
        assert_eq!(off.len(), 4);

        // An immediate that is not a mapped data address is never a ref:
        // mov rdi, 0x140001000 (the entry point, executable) then ret.
        let mut img = pe_with_data();
        img[0x200..0x20B].copy_from_slice(&[
            0x48, 0xBF, 0x00, 0x10, 0x00, 0x40, 0x01, 0x00, 0x00, 0x00, 0xC3,
        ]);
        assert!(xrefs_of(&img).is_empty());
    }

    #[test]
    fn hostile_bytes_never_panic() {
        // A .text of nothing but 0xFF (the byte with the most decode
        // paths, several of them indirect branches).
        let mut img = pe_with_data();
        img[0x200..0x400].fill(0xFF);
        let _ = xrefs_of(&img);

        // Every single-byte corruption of the entry window.
        for byte in 0..=0xFFu8 {
            let mut img = pe_data_ref_fixture();
            img[0x200] = byte;
            let _ = xrefs_of(&img);
        }

        // A64: every 4-byte word made of one repeated byte.
        for byte in 0..=0xFFu8 {
            let mut img = aarch64_adrp_fixture();
            img[0x100..0x180].fill(byte);
            let _ = xrefs_of(&img);
        }
    }

    #[test]
    fn images_with_no_code_and_unsupported_arches() {
        // Nothing recovered means nothing referenced.
        let img = crate::elf::tests::synthetic_dynamic_elf64();
        let x = xrefs_of(&img);
        assert!(x.is_empty());
        assert_eq!(x.stats, Stats::default());

        // An architecture with no decoder is a typed error, matching
        // `cfg::recover`'s contract.
        let mut img = synthetic_elf64();
        img[18..20].copy_from_slice(&0xF00u16.to_le_bytes()); // unknown e_machine
        let image = load(&img).unwrap();
        let program = cfg::Program {
            functions: BTreeMap::new(),
            call_graph: BTreeMap::new(),
            stats: cfg::Stats::default(),
        };
        assert!(matches!(
            compute(image.as_ref(), &program),
            Err(ParseError::Unsupported(_))
        ));
    }

    #[test]
    fn kinds_classify_and_render() {
        assert!(XrefKind::Call.is_code() && !XrefKind::Call.is_data());
        assert!(XrefKind::Branch.is_code());
        assert!(XrefKind::CondBranch.is_code());
        for kind in [
            XrefKind::Read,
            XrefKind::Write,
            XrefKind::AddressOf,
            XrefKind::ImmediatePointer,
        ] {
            assert!(kind.is_data() && !kind.is_code(), "{kind:?}");
        }
        assert_eq!(XrefKind::AddressOf.as_str(), "addr");
    }
}
