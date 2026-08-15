//! Virtual-call resolution — turning indirect calls into candidate targets.
//!
//! A C++ virtual dispatch or a Rust trait-object call compiles to an
//! indirect call through a vtable slot: `call [reg + off]`. On its own
//! that edge is opaque, but the recovered vtables ([`crate::vtable`]) name
//! exactly what sits at each slot offset of each class, so a load of a
//! known vtable followed by an indirect call through offset `off` resolves
//! to the concrete function in that slot. This module recovers those
//! (call site -> candidate targets) edges, enriching the call graph past
//! the indirect boundary. Best-effort by contract: an unresolvable
//! indirect call is left as-is, and no input ever panics.
//!
//! # Two tiers, with confidence
//!
//! 1. [`Confidence::SlotClass`] — **slot-offset enumeration**, always
//!    available. Every recovered class contributes its vtable's slot `i`
//!    to the byte offset `i * 8` measured from the vtable's *address
//!    point* (the VA an object's vptr holds, which is the header VA plus
//!    two words — see [`crate::vtable::Vtable`]). The displacement of a
//!    `call [reg + off]` is exactly that byte offset, so the candidate set
//!    of `call [rax + 0x10]` is slot 2 of *every* class. This is
//!    deliberately **over-approximate**: it is the sound superset that
//!    honestly describes "the class is unknown, so it could be any of
//!    these", never a guess at one of them.
//!
//! 2. [`Confidence::Exact`] — **intra-procedural class refinement**, when
//!    the class is established in the open. If the last instruction to
//!    define `reg` before the call is `lea reg, [rip + disp]` or
//!    `mov reg, imm` whose value *is* a known vtable's address point, the
//!    call is a dispatch on that one class and resolves to that one slot.
//!    This is the precise case.
//!
//! The refinement walks backwards over the call's own basic block only
//! (bounded by [`MAX_SCAN_WINDOW`] instructions) and follows the *last
//! definition* of the base register, so an intervening redefinition —
//! including the `mov reg, [reg2]` that loads an object's vptr, whose
//! value is not statically known — costs the site its exact answer and
//! leaves the tier-1 superset. A definition in an earlier block is not
//! followed: the result is a broader candidate list, never a wrong one.
//!
//! # ISA use
//!
//! Recognizing a dispatch is operand-level work, so like
//! [`crate::jumptable`] this module reaches past the
//! [`crate::model::Decoder`] flow abstraction into [`crate::x86`]. It
//! stays *format*-neutral: memory is read only through [`Image`]. Only
//! x86-64 is implemented; the model (a slot offset, a class, a slot index)
//! is ISA-neutral, and an image of any other architecture resolves to an
//! empty result rather than an error.
//!
//! # Bounds and determinism
//!
//! Attacker-controlled bytes reach every step. A displacement that is
//! negative, not a multiple of the pointer size, or past any plausible
//! vtable extent is skipped rather than converted; the backward scan is
//! capped at [`MAX_SCAN_WINDOW`] instructions, the candidates of one site
//! at [`MAX_CANDIDATES`], the whole result at [`MAX_SITES`] sites, and the
//! slot index at [`MAX_INDEX_ENTRIES`] entries. Sites are keyed by call VA
//! in a [`BTreeMap`] and candidate lists are sorted and deduplicated, so
//! the same inputs always yield the same `Vec<CallSite>`. All arithmetic
//! is checked: no input panics, and [`resolve`] never errors.

use std::collections::{BTreeMap, VecDeque};

use crate::cfg;
use crate::model::{Arch, Flow, Image};
use crate::vtable::{self, CxxClass};
use crate::x86;

// ---------------------------------------------------------------------------
// Resource caps
// ---------------------------------------------------------------------------

/// Pointer width of the images this module resolves against, and hence the
/// stride between vtable slots.
const PTR: u64 = 8;

/// Distance from a sub-vtable's header VA to its address point — the VA an
/// object's vptr holds, and the base a dispatch's displacement is measured
/// from (Itanium C++ ABI §2.9: `[offset-to-top][typeinfo]` precede it).
const ADDRESS_POINT: u64 = 2 * PTR;

/// Maximum instructions of the call's basic block the backward scan keeps.
/// Every recognized establishing idiom is a handful of instructions long.
pub const MAX_SCAN_WINDOW: usize = 64;

/// Maximum candidate targets recorded for one call site.
pub const MAX_CANDIDATES: usize = 4096;

/// Maximum call sites reported from one program.
pub const MAX_SITES: usize = 1 << 20;

/// Maximum (slot offset, target) pairs and vtable references the slot
/// index holds, whatever the recovered class list claims.
pub const MAX_INDEX_ENTRIES: usize = 1 << 20;

/// Largest slot index a dispatch may select, matching the cap
/// [`crate::vtable`] puts on one sub-vtable's slots. A displacement past
/// `MAX_SLOT_INDEX * 8` is not addressing any vtable this crate can
/// recover, so it is not a dispatch.
pub const MAX_SLOT_INDEX: usize = vtable::MAX_SLOTS;

/// Candidates [`render`] prints per site before eliding the rest.
const RENDER_CANDIDATES: usize = 16;

// ---------------------------------------------------------------------------
// Public model
// ---------------------------------------------------------------------------

/// How precisely a call site's candidate list describes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    /// The receiver's class was established in the calling function, so
    /// the list is the one function that slot dispatches to.
    Exact,
    /// The class is unknown: the list is every class's function at this
    /// slot offset — a sound superset, not a guess.
    SlotClass,
}

impl Confidence {
    /// Short label for reports.
    pub fn label(self) -> &'static str {
        match self {
            Confidence::Exact => "exact",
            Confidence::SlotClass => "slot-class",
        }
    }
}

/// One candidate target of a virtual call: a function, and the class and
/// slot it was found in.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Target {
    /// Virtual address of the candidate function.
    pub func_va: u64,
    /// Name of the class whose vtable holds it.
    pub class_name: String,
    /// Index of the slot it occupies, counted from the address point.
    pub slot_index: usize,
}

/// One resolved virtual call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSite {
    /// Virtual address of the indirect call instruction.
    pub va: u64,
    /// Entry VA of the function containing it (the first such function in
    /// address order, for a block shared between functions).
    pub func_entry: u64,
    /// Displacement of the call's memory operand: the byte offset of the
    /// dispatched slot from the vtable address point. Never negative and
    /// always a multiple of the pointer size.
    pub slot_offset: i64,
    /// `slot_offset / 8` — the slot's index.
    pub slot_index: usize,
    /// Candidate targets, sorted and deduplicated, capped at
    /// [`MAX_CANDIDATES`]. Never empty: a site with no candidate is not
    /// reported at all.
    pub candidates: Vec<Target>,
    /// How precisely `candidates` describes this site.
    pub confidence: Confidence,
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Resolve every vtable-shaped indirect call in `program` against the
/// classes [`crate::vtable::recover`] finds in `image`.
///
/// Returns the sites sorted by call VA — deterministic for given inputs —
/// or an empty vector when the image carries no vtables, the architecture
/// is not x86-64, or nothing resolves. Never errors, never panics.
pub fn resolve(image: &dyn Image, program: &cfg::Program) -> Vec<CallSite> {
    let classes = vtable::recover(image);
    resolve_with(image, program, &classes)
}

/// [`resolve`] against a caller-supplied class list, for a pipeline that
/// has already run [`crate::vtable::recover`] (or that resolves against a
/// filtered subset of the hierarchy).
pub fn resolve_with(
    image: &dyn Image,
    program: &cfg::Program,
    classes: &[CxxClass],
) -> Vec<CallSite> {
    if image.arch() != Arch::X86_64 {
        // The model is ISA-neutral; only x86-64 dispatch is recognized.
        return Vec::new();
    }
    let index = SlotIndex::build(classes);
    if index.is_empty() {
        return Vec::new();
    }

    // Keyed by call site: a block shared between functions resolves once,
    // to the first function that reaches it in address order.
    let mut out: BTreeMap<u64, CallSite> = BTreeMap::new();
    for func in program.functions.values() {
        for block in func.blocks.values() {
            let cfg::Terminator::IndirectCall { import, .. } = &block.terminator else {
                continue;
            };
            // Already resolved by cfg as a call through an import slot.
            if import.is_some() {
                continue;
            }
            if out.len() >= MAX_SITES {
                return out.into_values().collect();
            }
            if let Some(site) = x86_site(image, &index, func, block) {
                out.entry(site.va).or_insert(site);
            }
        }
    }
    out.into_values().collect()
}

/// Render resolved call sites as a readable report.
///
/// Deterministic for a given input and always newline-terminated, so a CLI
/// can print it verbatim. An empty slice renders as a single explanatory
/// line rather than an empty string. Long candidate lists are elided after
/// the first 16 entries, with a count of the remainder.
pub fn render(sites: &[CallSite]) -> String {
    if sites.is_empty() {
        return "no virtual call sites resolved\n".to_string();
    }
    let exact = sites
        .iter()
        .filter(|s| s.confidence == Confidence::Exact)
        .count();
    let broad = sites.len().saturating_sub(exact);
    let mut out = format!(
        "virtual call sites: {} ({exact} exact, {broad} slot-class)\n",
        sites.len()
    );
    for site in sites {
        out.push_str(&format!(
            "\n{}  in {}  [+0x{:x}]  -> {} candidate(s) {}\n",
            hex_va(site.va),
            hex_va(site.func_entry),
            site.slot_offset.max(0),
            site.candidates.len(),
            site.confidence.label()
        ));
        for target in site.candidates.iter().take(RENDER_CANDIDATES) {
            out.push_str(&format!(
                "    {}  {}::slot{}\n",
                hex_va(target.func_va),
                target.class_name,
                target.slot_index
            ));
        }
        if let Some(elided) = site.candidates.len().checked_sub(RENDER_CANDIDATES)
            && elided > 0
        {
            out.push_str(&format!("    ... {elided} more\n"));
        }
    }
    out
}

/// Canonical VA spelling for the report.
fn hex_va(va: u64) -> String {
    format!("0x{va:016x}")
}

// ---------------------------------------------------------------------------
// The slot index
// ---------------------------------------------------------------------------

/// One class's sub-vtable, borrowed for the lifetime of a resolution.
struct VtableRef<'a> {
    class_name: &'a str,
    slots: &'a [u64],
}

/// The recovered hierarchy, indexed the two ways resolution asks about it.
struct SlotIndex<'a> {
    /// Slot byte offset (from the address point) -> every class's function
    /// at that offset. This is the tier-1 answer, and it is an
    /// over-approximation by construction.
    by_offset: BTreeMap<i64, Vec<Target>>,
    /// Vtable address point VA -> the sub-vtables published at it. This is
    /// what makes tier-2 refinement exact.
    by_point: BTreeMap<u64, Vec<VtableRef<'a>>>,
}

impl<'a> SlotIndex<'a> {
    /// Build both indexes, bounded by [`MAX_INDEX_ENTRIES`] entries and
    /// [`MAX_CANDIDATES`] targets per offset. Null slots (pure-virtual and
    /// offset entries) are not candidates, but they keep their index: the
    /// offsets of the slots after them are unaffected.
    fn build(classes: &'a [CxxClass]) -> SlotIndex<'a> {
        let mut by_offset: BTreeMap<i64, Vec<Target>> = BTreeMap::new();
        let mut by_point: BTreeMap<u64, Vec<VtableRef<'a>>> = BTreeMap::new();
        let mut entries = 0usize;

        'classes: for class in classes {
            for vt in &class.vtables {
                if entries >= MAX_INDEX_ENTRIES {
                    break 'classes;
                }
                if let Some(point) = vt.va.checked_add(ADDRESS_POINT) {
                    by_point.entry(point).or_default().push(VtableRef {
                        class_name: class.name.as_str(),
                        slots: vt.slots.as_slice(),
                    });
                    entries += 1;
                }
                for (slot_index, &func_va) in vt.slots.iter().enumerate() {
                    if slot_index >= MAX_SLOT_INDEX {
                        break;
                    }
                    if entries >= MAX_INDEX_ENTRIES {
                        break 'classes;
                    }
                    if func_va == 0 {
                        continue;
                    }
                    let Some(offset) = slot_offset(slot_index) else {
                        break;
                    };
                    let bucket = by_offset.entry(offset).or_default();
                    if bucket.len() >= MAX_CANDIDATES {
                        continue;
                    }
                    bucket.push(Target {
                        func_va,
                        class_name: class.name.clone(),
                        slot_index,
                    });
                    entries += 1;
                }
            }
        }
        for bucket in by_offset.values_mut() {
            bucket.sort();
            bucket.dedup();
        }
        SlotIndex {
            by_offset,
            by_point,
        }
    }

    /// Whether nothing can resolve against this index.
    fn is_empty(&self) -> bool {
        self.by_offset.is_empty()
    }

    /// Tier 1: every class's function at `offset`.
    fn at_offset(&self, offset: i64) -> Vec<Target> {
        self.by_offset.get(&offset).cloned().unwrap_or_default()
    }

    /// Tier 2: the function at `slot_index` of the vtable(s) whose address
    /// point is `point`. Empty when nothing publishes that address point,
    /// or when the vtable there has no such slot.
    fn at_point(&self, point: u64, slot_index: usize) -> Vec<Target> {
        let mut out = Vec::new();
        for vt in self.by_point.get(&point).into_iter().flatten() {
            match vt.slots.get(slot_index) {
                Some(&func_va) if func_va != 0 => out.push(Target {
                    func_va,
                    class_name: vt.class_name.to_string(),
                    slot_index,
                }),
                _ => {}
            }
        }
        out.sort();
        out.dedup();
        out.truncate(MAX_CANDIDATES);
        out
    }
}

/// Byte offset of slot `index` from the address point.
fn slot_offset(index: usize) -> Option<i64> {
    i64::try_from(index).ok()?.checked_mul(PTR as i64)
}

/// The slot a displacement selects: `disp / 8`, and only when `disp` is
/// non-negative, pointer-aligned, and inside a plausible vtable extent.
/// Anything else is not a vtable dispatch — skipped, never converted.
fn slot_index_of(disp: i64) -> Option<usize> {
    if disp < 0 || disp % PTR as i64 != 0 {
        return None;
    }
    let index = usize::try_from(disp / PTR as i64).ok()?;
    (index < MAX_SLOT_INDEX).then_some(index)
}

// ---------------------------------------------------------------------------
// x86-64 recognition
// ---------------------------------------------------------------------------

/// One decoded instruction of a basic block.
struct Insn {
    va: u64,
    ins: x86::Instruction,
}

/// Decode a block's instructions, keeping the last [`MAX_SCAN_WINDOW`] —
/// which always ends with the block's terminator, the call being resolved.
/// A decode error ends the walk: the block is under-approximated, never
/// resynchronized by guessing a length.
fn block_insns(image: &dyn Image, block: &cfg::BasicBlock) -> Vec<Insn> {
    let bytes = image.bytes();
    let mut window: VecDeque<Insn> = VecDeque::new();
    let mut va = block.start;
    while va < block.end {
        let Some(off) = image.va_to_offset(va) else {
            break;
        };
        let avail = bytes.len().saturating_sub(off);
        let limit = (block.end - va).min(x86::MAX_INSTRUCTION_LEN as u64) as usize;
        let Some(chunk) = bytes.get(off..off.saturating_add(limit.min(avail))) else {
            break;
        };
        let Ok(ins) = x86::decode(chunk, va) else {
            break;
        };
        if ins.length == 0 {
            break;
        }
        let Some(next) = va.checked_add(u64::from(ins.length)) else {
            break;
        };
        window.push_back(Insn { va, ins });
        if window.len() > MAX_SCAN_WINDOW {
            window.pop_front();
        }
        va = next;
    }
    window.into()
}

/// A 64-bit general-purpose register operand's number.
fn gpr64(op: Option<&x86::Operand>) -> Option<u8> {
    match op {
        Some(&x86::Operand::Reg(r)) if r.width == x86::Width::W64 && !r.high_byte => Some(r.num),
        _ => None,
    }
}

/// Bit for a register operand, or 0 for anything else.
fn reg_bit(op: Option<&x86::Operand>) -> u16 {
    match op {
        Some(x86::Operand::Reg(r)) => 1u16 << (r.num & 0xF),
        _ => 0,
    }
}

/// The set of general-purpose registers an instruction may write, as a
/// bitmask over register numbers 0-15.
///
/// Deliberately over-approximate: `call` is treated as clobbering
/// everything, since this pass models no ABI. Over-approximating can only
/// *lose* an exact answer (leaving the sound tier-1 superset), never
/// invent one.
fn defs(ins: &x86::Instruction) -> u16 {
    use x86::Opcode as O;
    const RAX: u16 = 0x0001;
    const RCX: u16 = 0x0002;
    const RDX: u16 = 0x0004;
    const RBX: u16 = 0x0008;
    const RSP: u16 = 0x0010;
    const RBP: u16 = 0x0020;
    const R11: u16 = 0x0800;

    let d0 = reg_bit(ins.operands.first());
    let d1 = reg_bit(ins.operands.get(1));
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
        // One-operand `imul` writes rDX:rAX; the two- and three-operand
        // forms write their destination register.
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
        O::Call => u16::MAX, // unknown callee: assume everything
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
        // SSE: only the `cvt*2si` converts and the `movd`/`movq` stores write
        // a GPR, and those put it in the (first) destination operand, so `d0`
        // is exactly the GPR-write set — zero for the XMM-destination forms.
        O::Sse { .. } => d0,
    }
}

/// Index of the last instruction before `before` that writes register
/// `num`.
fn last_def(insns: &[Insn], before: usize, num: u8) -> Option<usize> {
    let mask = 1u16 << (num & 0xF);
    insns
        .get(..before)?
        .iter()
        .rposition(|i| defs(&i.ins) & mask != 0)
}

/// The constant address an instruction establishes in register `num`, if
/// it establishes one at all:
///
/// - `lea num, [rip + disp]` — resolved against the *next* instruction's
///   VA (SDM Vol 2 §2.2.1.6), which is how a PIC build names a vtable.
/// - `mov num, imm` — a non-PIC build's absolute address point (a 32-bit
///   destination zero-extends, per the SDM's 64-bit-mode rules).
///
/// Anything else — most importantly the `mov num, [reg]` that loads an
/// object's vptr, whose value is a runtime property of the object — yields
/// `None`, and the site keeps its tier-1 candidate set.
fn established_address(at: &Insn, num: u8) -> Option<u64> {
    let ins = &at.ins;
    match ins.opcode {
        x86::Opcode::Lea => {
            if gpr64(ins.operands.first()) != Some(num) {
                return None;
            }
            let Some(&x86::Operand::Mem {
                base: None,
                index: None,
                disp,
                rip_relative: true,
                ..
            }) = ins.operands.get(1)
            else {
                return None;
            };
            at.va
                .checked_add(u64::from(ins.length))?
                .checked_add_signed(disp)
        }
        x86::Opcode::Mov => {
            let Some(&x86::Operand::Reg(dst)) = ins.operands.first() else {
                return None;
            };
            let Some(&x86::Operand::Imm(imm)) = ins.operands.get(1) else {
                return None;
            };
            if dst.num != num || dst.high_byte {
                return None;
            }
            match dst.width {
                x86::Width::W64 => Some(imm as u64),
                x86::Width::W32 => Some(u64::from(imm as u32)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Resolve one block whose terminator is an indirect call, if that call is
/// a vtable dispatch this pass recognizes.
fn x86_site(
    image: &dyn Image,
    index: &SlotIndex,
    func: &cfg::Function,
    block: &cfg::BasicBlock,
) -> Option<CallSite> {
    let insns = block_insns(image, block);
    let at = insns.len().checked_sub(1)?;
    let call = &insns[at].ins;
    if call.opcode != x86::Opcode::Call || call.flow != Flow::IndirectCall {
        return None;
    }
    // `call [reg + disp]`: a base register holding the address point and a
    // displacement selecting the slot. An index register means the slot is
    // computed, not fixed by the displacement, and `[rip + disp]` is a
    // static cell (an import or a function pointer), not a dispatch.
    let Some(&x86::Operand::Mem {
        base: Some(base),
        index: None,
        disp,
        rip_relative: false,
        ..
    }) = call.operands.first()
    else {
        return None;
    };
    if base.width != x86::Width::W64 || base.high_byte {
        return None;
    }
    let slot_index = slot_index_of(disp)?;

    // Tier 2 first: an established address point answers exactly. When the
    // class is known but its recovered vtable has no such slot, fall back
    // to the tier-1 superset rather than dropping the site — vtable
    // recovery truncates a slot walk it cannot validate, so a short slot
    // list is a limit of recovery, not proof the call has no target.
    let exact = last_def(&insns, at, base.num)
        .and_then(|def| established_address(&insns[def], base.num))
        .map(|point| index.at_point(point, slot_index))
        .filter(|targets| !targets.is_empty());

    let (mut candidates, confidence) = match exact {
        Some(targets) => (targets, Confidence::Exact),
        None => (index.at_offset(disp), Confidence::SlotClass),
    };
    candidates.truncate(MAX_CANDIDATES);
    if candidates.is_empty() {
        return None; // nothing resolved: the call is left as-is
    }
    Some(CallSite {
        va: insns[at].va,
        func_entry: func.entry,
        slot_offset: disp,
        slot_index,
        candidates,
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImportSlot, Perms, Region, Symbol, SymbolKind};
    use crate::vtable::{Provenance, TypeInfoKind, Vtable};

    // ---- Synthetic images ------------------------------------------------

    /// Size of every synthetic image's buffer.
    const IMG_SIZE: usize = 0x1_0000;
    /// Readable, non-executable region (typeinfo objects and vtables).
    const DATA_VA: u64 = 0x1000;
    const DATA_END: u64 = 0x8000;
    /// Executable region (calling code and virtual functions).
    const TEXT_VA: u64 = 0x8000;
    const TEXT_END: u64 = 0x9000;

    /// Identity-mapped image (VA == file offset) with one data and one
    /// text region — the [`crate::vtable`] fixture, plus entry points so
    /// [`cfg::recover`] has somewhere to start decoding.
    struct FakeImage {
        data: Vec<u8>,
        regions: Vec<Region>,
        symbols: Vec<Symbol>,
        entries: Vec<u64>,
        arch: Arch,
    }

    impl Image for FakeImage {
        fn arch(&self) -> Arch {
            self.arch
        }
        fn entry_points(&self) -> Vec<u64> {
            self.entries.clone()
        }
        fn regions(&self) -> Vec<Region> {
            self.regions.clone()
        }
        fn symbols(&self) -> Vec<Symbol> {
            self.symbols.clone()
        }
        fn import_slots(&self) -> Vec<ImportSlot> {
            Vec::new()
        }
        fn va_to_offset(&self, va: u64) -> Option<usize> {
            let off = usize::try_from(va).ok()?;
            (off < self.data.len()).then_some(off)
        }
        fn bytes(&self) -> &[u8] {
            &self.data
        }
    }

    impl FakeImage {
        fn new() -> FakeImage {
            FakeImage {
                data: vec![0u8; IMG_SIZE],
                regions: vec![
                    Region {
                        name: ".data.rel.ro".into(),
                        va: DATA_VA,
                        size: DATA_END - DATA_VA,
                        perms: Perms {
                            r: true,
                            w: false,
                            x: false,
                        },
                    },
                    Region {
                        name: ".text".into(),
                        va: TEXT_VA,
                        size: TEXT_END - TEXT_VA,
                        perms: Perms {
                            r: true,
                            w: false,
                            x: true,
                        },
                    },
                ],
                symbols: Vec::new(),
                entries: Vec::new(),
                arch: Arch::X86_64,
            }
        }

        fn u64(&mut self, va: u64, value: u64) -> &mut Self {
            let off = va as usize;
            self.data[off..off + 8].copy_from_slice(&value.to_le_bytes());
            self
        }

        fn str(&mut self, va: u64, s: &str) -> &mut Self {
            let off = va as usize;
            self.data[off..off + s.len()].copy_from_slice(s.as_bytes());
            self.data[off + s.len()] = 0;
            self
        }

        fn sym(&mut self, name: &str, va: u64) -> &mut Self {
            self.symbols.push(Symbol {
                name: name.into(),
                va,
                kind: SymbolKind::Data,
            });
            self
        }

        /// A value that is neither zero nor a code pointer, which ends a
        /// vtable's slot walk the way the next object's bytes would.
        fn terminator(&mut self, va: u64) -> &mut Self {
            self.u64(va, 0x0bad_0bad_0bad_0bad)
        }

        /// A class: a typeinfo object (`_ZTI`), its `_ZTS` payload string,
        /// and a `_ZTV` vtable object holding `slots` after the
        /// `[offset-to-top][typeinfo]` header.
        fn class(
            &mut self,
            payload: &str,
            typeinfo_va: u64,
            name_va: u64,
            vtable_va: u64,
            slots: &[u64],
        ) -> &mut Self {
            self.u64(typeinfo_va, 0)
                .u64(typeinfo_va + 8, name_va)
                .str(name_va, payload)
                .u64(vtable_va, 0)
                .u64(vtable_va + 8, typeinfo_va);
            for (i, &slot) in slots.iter().enumerate() {
                self.u64(vtable_va + ADDRESS_POINT + i as u64 * PTR, slot);
            }
            self.terminator(vtable_va + ADDRESS_POINT + slots.len() as u64 * PTR)
                .sym(&format!("_ZTI{payload}"), typeinfo_va)
                .sym(&format!("_ZTV{payload}"), vtable_va)
        }

        /// Machine code at `va`, and `va` as a function entry point.
        fn code(&mut self, va: u64, bytes: &[u8]) -> &mut Self {
            let off = va as usize;
            self.data[off..off + bytes.len()].copy_from_slice(bytes);
            self
        }

        fn entry(&mut self, va: u64) -> &mut Self {
            self.entries.push(va);
            self
        }
    }

    // ---- x86-64 encodings the fixtures emit ------------------------------
    //
    // Written by hand from the Intel SDM Vol 2 opcode maps, so the tests
    // exercise the real decoder rather than a mock:
    //
    //   FF /2                  call r/m64  (ModRM reg field = 010)
    //     mod=01 rm=base       `call [base + disp8]`   -> FF, 0x50|base, d8
    //     mod=10 rm=base       `call [base + disp32]`  -> FF, 0x90|base, d32
    //     mod=11 rm=reg        `call reg`              -> FF, 0xD0|reg
    //     mod=01 rm=100 + SIB  `call [base + idx*8 + d8]`
    //   48 8D /r with mod=00 rm=101   `lea r64, [rip + disp32]`
    //   48 B8+r imm64                 `mov r64, imm64`
    //   48 8B /r with mod=00          `mov r64, [base]`
    //   C3                            `ret`

    /// `call qword [base + disp8]`.
    fn call_disp8(base: u8, disp: i8) -> Vec<u8> {
        vec![0xFF, 0x50 | base, disp as u8]
    }

    /// `call qword [base + disp32]`.
    fn call_disp32(base: u8, disp: i32) -> Vec<u8> {
        let mut out = vec![0xFF, 0x90 | base];
        out.extend_from_slice(&disp.to_le_bytes());
        out
    }

    /// `call base` — register-indirect, no memory operand.
    fn call_reg(base: u8) -> Vec<u8> {
        vec![0xFF, 0xD0 | base]
    }

    /// `call qword [base + index*8 + disp8]`.
    fn call_indexed(base: u8, index: u8, disp: i8) -> Vec<u8> {
        vec![0xFF, 0x54, 0xC0 | (index << 3) | base, disp as u8]
    }

    /// `lea dst, [rip + disp32]` placed at `va`, addressing `target`.
    fn lea_rip(dst: u8, va: u64, target: u64) -> Vec<u8> {
        let disp = target.wrapping_sub(va.wrapping_add(7)) as i64 as i32;
        let mut out = vec![0x48, 0x8D, 0x05 | (dst << 3)];
        out.extend_from_slice(&disp.to_le_bytes());
        out
    }

    /// `mov dst, imm64`.
    fn mov_imm64(dst: u8, imm: u64) -> Vec<u8> {
        let mut out = vec![0x48, 0xB8 | dst];
        out.extend_from_slice(&imm.to_le_bytes());
        out
    }

    /// `mov dst, qword [base]` — the object-to-vptr load.
    fn mov_load(dst: u8, base: u8) -> Vec<u8> {
        vec![0x48, 0x8B, (dst << 3) | base]
    }

    const RAX: u8 = 0;
    const RCX: u8 = 1;

    /// Concatenate encodings into one straight-line run.
    fn run(parts: &[Vec<u8>]) -> Vec<u8> {
        parts.concat()
    }

    fn ret() -> Vec<u8> {
        vec![0xC3]
    }

    // ---- Fixtures --------------------------------------------------------

    /// Widget's vtable, at 0x3000: address point 0x3010, three slots.
    const WIDGET_TI: u64 = 0x2000;
    const WIDGET_VT: u64 = 0x3000;
    const WIDGET_POINT: u64 = WIDGET_VT + ADDRESS_POINT;
    /// Gadget's vtable, at 0x3100: address point 0x3110, three slots.
    const GADGET_TI: u64 = 0x2200;
    const GADGET_VT: u64 = 0x3100;
    const GADGET_POINT: u64 = GADGET_VT + ADDRESS_POINT;

    /// Virtual functions, one per class per slot.
    const W0: u64 = TEXT_VA + 0x100;
    const W1: u64 = TEXT_VA + 0x110;
    const W2: u64 = TEXT_VA + 0x120;
    const G0: u64 = TEXT_VA + 0x200;
    const G1: u64 = TEXT_VA + 0x210;
    const G2: u64 = TEXT_VA + 0x220;

    /// VA of the caller the fixtures assemble.
    const CALLER: u64 = TEXT_VA;

    fn with_widget(img: &mut FakeImage) {
        img.class("6Widget", WIDGET_TI, 0x2100, WIDGET_VT, &[W0, W1, W2]);
    }

    fn with_gadget(img: &mut FakeImage) {
        img.class("6Gadget", GADGET_TI, 0x2300, GADGET_VT, &[G0, G1, G2]);
    }

    /// A caller function at [`CALLER`] made of `body` plus a `ret`.
    fn with_caller(img: &mut FakeImage, body: &[u8]) {
        let mut code = body.to_vec();
        code.extend_from_slice(&ret());
        img.code(CALLER, &code).entry(CALLER);
    }

    /// Full pipeline: cfg recovery, then resolution against the image's
    /// own recovered vtables.
    fn sites_of(img: &FakeImage) -> Vec<CallSite> {
        let program = cfg::recover(img).expect("x86-64 image must recover");
        resolve(img, &program)
    }

    /// Resolution against a caller-supplied class list.
    fn sites_with(img: &FakeImage, classes: &[CxxClass]) -> Vec<CallSite> {
        let program = cfg::recover(img).expect("x86-64 image must recover");
        resolve_with(img, &program, classes)
    }

    /// A hand-built class, for the cases where planting thousands of
    /// typeinfo objects in a fixture would test the recovery pass rather
    /// than this one.
    fn synthetic_class(name: &str, vtable_va: u64, slots: &[u64]) -> CxxClass {
        CxxClass {
            name: name.to_string(),
            typeinfo_va: vtable_va.wrapping_sub(0x1000),
            kind: TypeInfoKind::Class,
            vtables: vec![Vtable {
                va: vtable_va,
                offset_to_top: 0,
                typeinfo_va: vtable_va.wrapping_sub(0x1000),
                slots: slots.to_vec(),
            }],
            bases: Vec::new(),
            provenance: Provenance::Symbol,
        }
    }

    fn target(func_va: u64, class_name: &str, slot_index: usize) -> Target {
        Target {
            func_va,
            class_name: class_name.to_string(),
            slot_index,
        }
    }

    // ---- Tier 1: slot-offset enumeration ---------------------------------

    #[test]
    fn a_single_class_gives_a_call_its_slot_as_the_candidate() {
        let mut img = FakeImage::new();
        with_widget(&mut img);
        // `call [rax + 0x10]` with rax established nowhere in the block.
        with_caller(&mut img, &call_disp8(RAX, 0x10));

        let sites = sites_of(&img);
        assert_eq!(sites.len(), 1, "{sites:#x?}");
        let site = &sites[0];
        assert_eq!(site.va, CALLER);
        assert_eq!(site.func_entry, CALLER);
        assert_eq!(site.slot_offset, 0x10);
        assert_eq!(site.slot_index, 2);
        assert_eq!(site.confidence, Confidence::SlotClass);
        assert_eq!(site.candidates, vec![target(W2, "Widget", 2)]);
    }

    #[test]
    fn two_classes_sharing_a_slot_offset_are_both_candidates() {
        let mut img = FakeImage::new();
        with_widget(&mut img);
        with_gadget(&mut img);
        with_caller(&mut img, &call_disp8(RAX, 0x10));

        let sites = sites_of(&img);
        assert_eq!(sites.len(), 1, "{sites:#x?}");
        // The whole set is the honest answer when the class is unknown.
        assert_eq!(sites[0].confidence, Confidence::SlotClass);
        assert_eq!(
            sites[0].candidates,
            vec![target(W2, "Widget", 2), target(G2, "Gadget", 2)]
        );
    }

    #[test]
    fn a_slot_offset_of_zero_dispatches_the_first_slot() {
        let mut img = FakeImage::new();
        with_widget(&mut img);
        with_gadget(&mut img);
        // `call [rcx + 0]`: ModRM mod=01 with a zero disp8.
        with_caller(&mut img, &call_disp8(RCX, 0));

        let sites = sites_of(&img);
        assert_eq!(sites.len(), 1, "{sites:#x?}");
        assert_eq!(sites[0].slot_index, 0);
        assert_eq!(
            sites[0].candidates,
            vec![target(W0, "Widget", 0), target(G0, "Gadget", 0)]
        );
    }

    #[test]
    fn null_slots_are_not_candidates() {
        let mut img = FakeImage::new();
        // Slot 1 is a pure-virtual entry: it keeps its index (slot 2 is
        // still at +0x10) but is nobody's call target.
        img.class("6Widget", WIDGET_TI, 0x2100, WIDGET_VT, &[W0, 0, W2]);
        with_caller(&mut img, &call_disp8(RAX, 0x08));

        assert!(sites_of(&img).is_empty());
    }

    // ---- Tier 2: class refinement ----------------------------------------

    #[test]
    fn a_lea_of_a_vtable_address_point_resolves_exactly() {
        let mut img = FakeImage::new();
        with_widget(&mut img);
        with_gadget(&mut img);
        let lea = lea_rip(RAX, CALLER, WIDGET_POINT);
        let call_va = CALLER + lea.len() as u64;
        with_caller(&mut img, &run(&[lea, call_disp8(RAX, 0x10)]));

        let sites = sites_of(&img);
        assert_eq!(sites.len(), 1, "{sites:#x?}");
        assert_eq!(sites[0].va, call_va);
        assert_eq!(sites[0].confidence, Confidence::Exact);
        // Gadget also has a slot 2, but this call is not on a Gadget.
        assert_eq!(sites[0].candidates, vec![target(W2, "Widget", 2)]);
    }

    #[test]
    fn a_mov_immediate_address_point_resolves_exactly() {
        let mut img = FakeImage::new();
        with_widget(&mut img);
        with_gadget(&mut img);
        with_caller(
            &mut img,
            &run(&[mov_imm64(RCX, GADGET_POINT), call_disp8(RCX, 0x08)]),
        );

        let sites = sites_of(&img);
        assert_eq!(sites.len(), 1, "{sites:#x?}");
        assert_eq!(sites[0].confidence, Confidence::Exact);
        assert_eq!(sites[0].candidates, vec![target(G1, "Gadget", 1)]);
    }

    #[test]
    fn a_load_through_an_object_pointer_keeps_the_broad_set() {
        // `lea rax, [rip -> Widget]` then `mov rax, [rcx]`: the last
        // definition is the object's vptr load, whose value is a runtime
        // property, so the class is not established.
        let mut img = FakeImage::new();
        with_widget(&mut img);
        with_gadget(&mut img);
        let lea = lea_rip(RAX, CALLER, WIDGET_POINT);
        with_caller(
            &mut img,
            &run(&[lea, mov_load(RAX, RCX), call_disp8(RAX, 0x10)]),
        );

        let sites = sites_of(&img);
        assert_eq!(sites.len(), 1, "{sites:#x?}");
        assert_eq!(sites[0].confidence, Confidence::SlotClass);
        assert_eq!(
            sites[0].candidates,
            vec![target(W2, "Widget", 2), target(G2, "Gadget", 2)]
        );
    }

    #[test]
    fn a_lea_of_something_that_is_not_an_address_point_keeps_the_broad_set() {
        // The `lea` lands on the vtable's *header*, not its address point,
        // so no class is established.
        let mut img = FakeImage::new();
        with_widget(&mut img);
        let lea = lea_rip(RAX, CALLER, WIDGET_VT);
        with_caller(&mut img, &run(&[lea, call_disp8(RAX, 0x10)]));

        let sites = sites_of(&img);
        assert_eq!(sites.len(), 1, "{sites:#x?}");
        assert_eq!(sites[0].confidence, Confidence::SlotClass);
    }

    #[test]
    fn an_exact_class_without_that_slot_falls_back_to_the_broad_set() {
        // Widget's recovered vtable has three slots; the call dispatches
        // slot 4. Rather than drop the site, the sound superset stands.
        let mut img = FakeImage::new();
        with_widget(&mut img);
        let slots = [G0, G1, G2, G2, G1];
        img.class("6Gadget", GADGET_TI, 0x2300, GADGET_VT, &slots);
        let lea = lea_rip(RAX, CALLER, WIDGET_POINT);
        with_caller(&mut img, &run(&[lea, call_disp8(RAX, 0x20)]));

        let sites = sites_of(&img);
        assert_eq!(sites.len(), 1, "{sites:#x?}");
        assert_eq!(sites[0].slot_index, 4);
        assert_eq!(sites[0].confidence, Confidence::SlotClass);
        assert_eq!(sites[0].candidates, vec![target(G1, "Gadget", 4)]);
    }

    // ---- Shapes that are not a dispatch ----------------------------------

    #[test]
    fn a_register_indirect_call_is_ignored() {
        let mut img = FakeImage::new();
        with_widget(&mut img);
        with_caller(&mut img, &call_reg(RAX));

        assert!(sites_of(&img).is_empty());
    }

    #[test]
    fn an_indexed_call_is_not_a_fixed_slot_dispatch() {
        let mut img = FakeImage::new();
        with_widget(&mut img);
        with_caller(&mut img, &call_indexed(RAX, RCX, 0x10));

        assert!(sites_of(&img).is_empty());
    }

    #[test]
    fn misaligned_and_negative_displacements_are_skipped() {
        for disp in [0x14i8, 0x01, -0x10, -0x08] {
            let mut img = FakeImage::new();
            with_widget(&mut img);
            with_gadget(&mut img);
            with_caller(&mut img, &call_disp8(RAX, disp));
            assert!(
                sites_of(&img).is_empty(),
                "disp {disp:#x} must not convert to a slot"
            );
        }
        assert_eq!(slot_index_of(-8), None);
        assert_eq!(slot_index_of(4), None);
        assert_eq!(slot_index_of(i64::MIN), None);
        assert_eq!(slot_index_of(0), Some(0));
        assert_eq!(slot_index_of(0x10), Some(2));
    }

    #[test]
    fn a_displacement_past_every_vtable_yields_no_candidates() {
        let mut img = FakeImage::new();
        with_widget(&mut img);
        with_caller(&mut img, &call_disp32(RAX, 0x7FFF_FFF8));

        assert!(sites_of(&img).is_empty());
        // And past the slot-index cap it is not even a slot.
        assert_eq!(slot_index_of(i64::MAX & !7), None);
        assert_eq!(slot_offset(MAX_SLOT_INDEX - 1), Some(32760));
    }

    #[test]
    fn an_image_without_vtables_resolves_nothing() {
        let mut img = FakeImage::new();
        with_caller(&mut img, &call_disp8(RAX, 0x10));

        assert!(sites_of(&img).is_empty());
        assert_eq!(render(&[]), "no virtual call sites resolved\n");
    }

    #[test]
    fn a_non_x86_image_resolves_nothing() {
        let mut img = FakeImage::new();
        with_widget(&mut img);
        with_caller(&mut img, &call_disp8(RAX, 0x10));
        let program = cfg::recover(&img).unwrap();
        img.arch = Arch::Aarch64;
        assert!(resolve(&img, &program).is_empty());
        img.arch = Arch::Other;
        assert!(resolve(&img, &program).is_empty());
    }

    #[test]
    fn a_program_with_no_functions_resolves_nothing() {
        let mut img = FakeImage::new();
        with_widget(&mut img);
        img.code(CALLER, &call_disp8(RAX, 0x10)); // no entry point
        assert!(sites_of(&img).is_empty());
    }

    // ---- Caps, determinism, and reporting --------------------------------

    #[test]
    fn candidates_per_site_are_capped() {
        let mut img = FakeImage::new();
        with_caller(&mut img, &call_disp8(RAX, 0x10));
        let classes: Vec<CxxClass> = (0..MAX_CANDIDATES + 500)
            .map(|i| {
                let va = 0x10_0000 + i as u64 * 0x40;
                synthetic_class(&format!("C{i}"), va, &[0, 0, TEXT_VA + i as u64])
            })
            .collect();

        let sites = sites_with(&img, &classes);
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].candidates.len(), MAX_CANDIDATES);
        assert!(render(&sites).lines().count() <= RENDER_CANDIDATES + 4);
    }

    #[test]
    fn an_exact_hit_is_capped_too() {
        // Many classes publishing the same address point: the exact answer
        // is still bounded, and still deterministic.
        let mut img = FakeImage::new();
        let lea = lea_rip(RAX, CALLER, 0x4010);
        with_caller(&mut img, &run(&[lea, call_disp8(RAX, 0x08)]));
        let classes: Vec<CxxClass> = (0..MAX_CANDIDATES + 10)
            .map(|i| synthetic_class(&format!("C{i}"), 0x4000, &[0, TEXT_VA + i as u64]))
            .collect();

        let sites = sites_with(&img, &classes);
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].confidence, Confidence::Exact);
        assert_eq!(sites[0].candidates.len(), MAX_CANDIDATES);
    }

    #[test]
    fn resolution_is_deterministic_and_sorted_by_call_va() {
        let mut img = FakeImage::new();
        with_widget(&mut img);
        with_gadget(&mut img);
        // Two dispatches in one function, plus a second function.
        let first = call_disp8(RAX, 0x10);
        let second = call_disp8(RCX, 0x08);
        let second_va = CALLER + first.len() as u64;
        with_caller(&mut img, &run(&[first, second]));
        let other = TEXT_VA + 0x40;
        img.code(other, &run(&[call_disp8(RAX, 0), ret()]));
        img.entry(other);

        let sites = sites_of(&img);
        assert_eq!(
            sites.iter().map(|s| s.va).collect::<Vec<_>>(),
            [CALLER, second_va, other]
        );
        assert_eq!(sites, sites_of(&img));
        assert_eq!(render(&sites), render(&sites_of(&img)));
    }

    #[test]
    fn render_reports_sites_candidates_and_elision() {
        let mut img = FakeImage::new();
        with_widget(&mut img);
        with_gadget(&mut img);
        let lea = lea_rip(RAX, CALLER, WIDGET_POINT);
        with_caller(&mut img, &run(&[lea, call_disp8(RAX, 0x10)]));

        let text = render(&sites_of(&img));
        assert!(text.ends_with('\n'));
        assert!(text.contains("virtual call sites: 1 (1 exact, 0 slot-class)"));
        assert!(text.contains("[+0x10]  -> 1 candidate(s) exact"));
        assert!(text.contains("0x0000000000008120  Widget::slot2"));
        assert_eq!(Confidence::Exact.label(), "exact");
        assert_eq!(Confidence::SlotClass.label(), "slot-class");

        // A list longer than the render cap is elided with a count.
        let mut img = FakeImage::new();
        with_caller(&mut img, &call_disp8(RAX, 0x10));
        let classes: Vec<CxxClass> = (0..RENDER_CANDIDATES + 4)
            .map(|i| {
                let va = 0x10_0000 + i as u64 * 0x40;
                synthetic_class(&format!("C{i}"), va, &[0, 0, TEXT_VA + i as u64])
            })
            .collect();
        let text = render(&sites_with(&img, &classes));
        assert!(text.contains("-> 20 candidate(s) slot-class"));
        assert!(text.contains("    ... 4 more\n"), "{text}");
    }

    // ---- Hostile and degenerate inputs -----------------------------------

    #[test]
    fn adversarial_code_bytes_stay_bounded_and_never_panic() {
        let mut state: u64 = 0x5eed_de71_2345_0001;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for _ in 0..32 {
            let mut img = FakeImage::new();
            with_widget(&mut img);
            with_gadget(&mut img);
            // Pseudo-garbage over the whole text region, entered at
            // several addresses at once.
            for off in 0..(TEXT_END - TEXT_VA) as usize {
                img.data[TEXT_VA as usize + off] = next() as u8;
            }
            for i in 0..8 {
                img.entry(TEXT_VA + (next() as u64 % 0x800) + i);
            }
            let sites = sites_of(&img);
            assert!(sites.len() <= MAX_SITES);
            for site in &sites {
                assert!(site.slot_offset >= 0 && site.slot_offset % PTR as i64 == 0);
                assert_eq!(site.slot_index, (site.slot_offset / PTR as i64) as usize);
                assert!(!site.candidates.is_empty());
                assert!(site.candidates.len() <= MAX_CANDIDATES);
            }
            let _ = render(&sites);
        }
    }

    #[test]
    fn hostile_class_lists_are_bounded() {
        let mut img = FakeImage::new();
        with_caller(&mut img, &call_disp8(RAX, 0x10));
        // A class whose vtable claims a slot list far past the cap, and a
        // vtable VA that overflows on its way to an address point.
        let mut huge = Vec::new();
        for i in 0..MAX_SLOT_INDEX + 64 {
            huge.push(TEXT_VA + i as u64);
        }
        let classes = vec![
            synthetic_class("Huge", 0x10_0000, &huge),
            synthetic_class("Overflow", u64::MAX - 4, &[TEXT_VA, TEXT_VA]),
        ];

        let sites = sites_with(&img, &classes);
        assert_eq!(sites.len(), 1);
        assert!(sites[0].candidates.len() <= MAX_CANDIDATES);
        let _ = render(&sites);
    }

    #[test]
    fn a_truncated_block_at_the_end_of_the_region_never_panics() {
        // The call's bytes run off the end of the text region: cfg cannot
        // decode it, and neither can this pass.
        let mut img = FakeImage::new();
        with_widget(&mut img);
        let at = TEXT_END - 2;
        img.code(at, &[0xFF, 0x90]).entry(at);
        let _ = sites_of(&img);
    }
}
