//! Affine stack-pointer tracking and stack-slot partition (DESIGN 9–10).
//!
//! After SSA optimization, every SSA name of the stack-pointer cell is
//! classified in a tiny abstract domain `Affine(c) | NotSp | Unknown`.
//! Load/store addresses that resolve to `sp0 + c` become
//! [`AddrClass::StackOff`]; those facts partition into evidence-backed
//! [`StackSlot`]s that [`slot_namer`] turns into `local_N` strings for
//! [`crate::pseudo`]'s [`VarNamer`] hook.
//!
//! Full VSA is deliberately not used — see `PLAN_IRSTACK.md` /
//! `research/decompiler/DESIGN.md` slice 9.

use std::collections::BTreeMap;

use crate::ir::{BinOp, Expr, Space, Stmt, Width};
use crate::irssa::SsaFunction;
use crate::model::Arch;

/// Cap on distinct affine constants tracked per function.
pub const MAX_AFFINE: usize = 4096;
/// Cap on stack slots emitted per function.
pub const MAX_SLOTS: usize = 512;

/// Abstract value of an SSA name relative to entry SP (`sp0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpFact {
    /// Proven `sp0 + offset` (offset may be negative).
    Affine(i64),
    /// Proven not derived from SP.
    NotSp,
    /// Insufficient evidence.
    Unknown,
}

/// Classification of a memory address expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddrClass {
    /// Address is `sp0 + offset`.
    StackOff(i64),
    /// Address proven independent of SP.
    NonStack,
    /// Could not prove either way.
    Unproven,
}

/// One evidence-backed stack slot: byte range relative to `sp0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackSlot {
    /// Lowest byte offset relative to entry SP (often negative).
    pub base: i64,
    /// Size in bytes justified by observed accesses.
    pub size: u64,
    /// Access sites that justify this slot: (block VA, stmt index, offset, width).
    pub evidence: Vec<(u64, usize, i64, Width)>,
}

/// Per-function stack analysis result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackFacts {
    pub entry: u64,
    /// SP cell number (x86: 4, aarch64: 31).
    pub sp_cell: u16,
    /// Facts per SSA name id.
    pub name_facts: BTreeMap<u16, SpFact>,
    /// Memory ops classified: (block, stmt_ix, class, access width).
    pub accesses: Vec<(u64, usize, AddrClass, Width)>,
    pub slots: Vec<StackSlot>,
    /// True when a non-affine SP write was seen (alloca-like).
    pub sp_escaped: bool,
    pub affine_capped: bool,
    pub slots_capped: bool,
}

impl StackFacts {
    /// Render a deterministic dump for CLI / tests.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "; stack facts entry={:#x} sp_cell={} escaped={} affine={} slots={}\n",
            self.entry,
            self.sp_cell,
            self.sp_escaped,
            self.name_facts
                .values()
                .filter(|f| matches!(f, SpFact::Affine(_)))
                .count(),
            self.slots.len()
        ));
        if self.affine_capped {
            out.push_str("; note: affine constant cap hit\n");
        }
        if self.slots_capped {
            out.push_str("; note: slot cap hit\n");
        }
        for (id, fact) in &self.name_facts {
            if let SpFact::Affine(c) = fact {
                out.push_str(&format!("  name#{id} = sp0{c:+}\n"));
            }
        }
        for slot in &self.slots {
            out.push_str(&format!(
                "  slot local_{} base={:+} size={} evidence={}\n",
                slot.base.unsigned_abs(),
                slot.base,
                slot.size,
                slot.evidence.len()
            ));
        }
        for &(b, i, class, w) in &self.accesses {
            let c = match class {
                AddrClass::StackOff(o) => format!("stack{o:+}"),
                AddrClass::NonStack => "nonstack".into(),
                AddrClass::Unproven => "unproven".into(),
            };
            out.push_str(&format!("  access {b:#x}:{i} {c} .{}\n", w.bits() / 8));
        }
        out
    }
}

/// Analyze `f` (post-SSA-opt). Total: never panics.
pub fn analyze(f: &SsaFunction) -> StackFacts {
    let sp_cell = sp_cell_num(f.arch);
    let mut name_facts: BTreeMap<u16, SpFact> = BTreeMap::new();
    let mut affine_capped = false;
    let mut distinct_affine: BTreeMap<i64, ()> = BTreeMap::new();

    // Entry: every live-in of the SP cell is Affine(0).
    for &id in &f.live_in {
        if let Some(n) = f.names.get(id as usize)
            && n.space == Space::Arch
            && n.cell == sp_cell
        {
            name_facts.insert(id, SpFact::Affine(0));
            distinct_affine.insert(0, ());
        }
    }

    let mut sp_escaped = false;
    let order = block_order(f);

    // Fixpoint over blocks (bounded iterations = block count).
    for _ in 0..order.len().saturating_add(1).max(1) {
        let mut changed = false;
        for &bva in &order {
            let Some(block) = f.blocks.get(&bva) else {
                continue;
            };
            // φ nodes: join args.
            for phi in &block.phis {
                let n = match f.names.get(phi.dst as usize) {
                    Some(n) => n,
                    None => continue,
                };
                if !(n.space == Space::Arch && n.cell == sp_cell) {
                    continue;
                }
                let mut joined = SpFact::Unknown;
                let mut first = true;
                for &(_, arg) in &phi.args {
                    let fact = name_facts.get(&arg).copied().unwrap_or(SpFact::Unknown);
                    if first {
                        joined = fact;
                        first = false;
                    } else {
                        joined = join(joined, fact);
                    }
                }
                changed |= set_fact(
                    &mut name_facts,
                    &mut distinct_affine,
                    &mut affine_capped,
                    phi.dst,
                    joined,
                );
            }
            for stmt in &block.stmts {
                if let Stmt::Assign { dst, value } = stmt {
                    let dst_id = dst.num;
                    let Some(n) = f.names.get(dst_id as usize) else {
                        continue;
                    };
                    let fact = eval_expr(value, &name_facts, f, sp_cell);
                    let is_sp = n.space == Space::Arch && n.cell == sp_cell;
                    if is_sp && !matches!(fact, SpFact::Affine(_)) {
                        sp_escaped = true;
                    }
                    // Record affine facts for SP and for any copy of SP into another cell.
                    if is_sp || matches!(fact, SpFact::Affine(_)) {
                        changed |= set_fact(
                            &mut name_facts,
                            &mut distinct_affine,
                            &mut affine_capped,
                            dst_id,
                            fact,
                        );
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Classify memory ops.
    let mut accesses = Vec::new();
    for &bva in &order {
        let Some(block) = f.blocks.get(&bva) else {
            continue;
        };
        for (i, stmt) in block.stmts.iter().enumerate() {
            match stmt {
                Stmt::Assign {
                    value: Expr::Load { addr, width },
                    ..
                } => {
                    let class = classify_addr(addr, &name_facts, f, sp_cell);
                    accesses.push((bva, i, class, *width));
                }
                Stmt::Store { addr, value } => {
                    let w = value.width_of().unwrap_or(Width::W64);
                    let class = classify_addr(addr, &name_facts, f, sp_cell);
                    accesses.push((bva, i, class, w));
                }
                _ => {}
            }
        }
    }

    let (slots, slots_capped) = partition_slots(&accesses);

    StackFacts {
        entry: f.entry,
        sp_cell,
        name_facts,
        accesses,
        slots,
        sp_escaped,
        affine_capped,
        slots_capped,
    }
}

/// Build a [`crate::pseudo::VarNamer`] closure mapping SSA var ids
/// (out-of-SSA) is separate — this maps **stack offsets** to names via
/// a lookup from SSA name id when that name is a proven stack address
/// temporary. For the first landing we expose slot labels by offset
/// for dumps; decompile wiring uses [`slot_label`].
pub fn slot_label(facts: &StackFacts, offset: i64) -> Option<String> {
    facts
        .slots
        .iter()
        .find(|s| offset >= s.base && offset < s.base + s.size as i64)
        .map(|s| format!("local_{}", s.base.unsigned_abs()))
}

/// Namer for out-of-SSA variable ids: if the underlying SSA name's cell
/// is never needed — instead, when rendering loads of stack slots the
/// caller can rewrite. Here we provide a helper that names an SSA name
/// id when that name is Affine (the address itself), returning
/// `local_<abs>` for the offset.
pub fn name_affine_local(facts: &StackFacts, ssa_name_id: u16) -> Option<String> {
    match facts.name_facts.get(&ssa_name_id)? {
        SpFact::Affine(c) => slot_label(facts, *c).or_else(|| Some(format!("sp0{c:+}"))),
        _ => None,
    }
}

fn sp_cell_num(arch: Arch) -> u16 {
    match arch {
        Arch::X86_64 => 4,
        Arch::Aarch64 => 31,
        Arch::Other => 4, // unused path; analysis refuses Other arches upstream
    }
}

fn block_order(f: &SsaFunction) -> Vec<u64> {
    f.blocks.keys().copied().collect()
}

fn join(a: SpFact, b: SpFact) -> SpFact {
    match (a, b) {
        (SpFact::Affine(x), SpFact::Affine(y)) if x == y => SpFact::Affine(x),
        (SpFact::NotSp, SpFact::NotSp) => SpFact::NotSp,
        (SpFact::Unknown, x) | (x, SpFact::Unknown) => x,
        _ => SpFact::Unknown,
    }
}

fn set_fact(
    map: &mut BTreeMap<u16, SpFact>,
    distinct: &mut BTreeMap<i64, ()>,
    capped: &mut bool,
    id: u16,
    fact: SpFact,
) -> bool {
    if let SpFact::Affine(c) = fact
        && !distinct.contains_key(&c)
    {
        if distinct.len() >= MAX_AFFINE {
            *capped = true;
            return false;
        }
        distinct.insert(c, ());
    }
    match map.get(&id) {
        Some(old) if *old == fact => false,
        _ => {
            map.insert(id, fact);
            true
        }
    }
}

fn eval_expr(
    e: &Expr,
    facts: &BTreeMap<u16, SpFact>,
    f: &SsaFunction,
    sp_cell: u16,
) -> SpFact {
    match e {
        Expr::Reg(r) => {
            if r.space == Space::Arch {
                // SSA: num is name id.
                if let Some(fact) = facts.get(&r.num) {
                    return *fact;
                }
                if let Some(n) = f.names.get(r.num as usize)
                    && n.space == Space::Arch
                    && n.cell == sp_cell
                    && n.version == 0
                {
                    return SpFact::Affine(0);
                }
            }
            SpFact::Unknown
        }
        Expr::Const { .. } => SpFact::NotSp,
        Expr::Binary {
            op: BinOp::Add,
            lhs,
            rhs,
        } => match (
            eval_expr(lhs, facts, f, sp_cell),
            as_i64_const(rhs),
            as_i64_const(lhs),
            eval_expr(rhs, facts, f, sp_cell),
        ) {
            (SpFact::Affine(c), Some(k), _, _) => SpFact::Affine(c.wrapping_add(k)),
            (_, _, Some(k), SpFact::Affine(c)) => SpFact::Affine(c.wrapping_add(k)),
            (SpFact::NotSp, _, _, SpFact::NotSp) => SpFact::NotSp,
            _ => SpFact::Unknown,
        },
        Expr::Binary {
            op: BinOp::Sub,
            lhs,
            rhs,
        } => match (eval_expr(lhs, facts, f, sp_cell), as_i64_const(rhs)) {
            (SpFact::Affine(c), Some(k)) => SpFact::Affine(c.wrapping_sub(k)),
            _ => SpFact::Unknown,
        },
        Expr::Load { .. } | Expr::Unary { .. } | Expr::Binary { .. } => SpFact::Unknown,
    }
}

fn as_i64_const(e: &Expr) -> Option<i64> {
    match e {
        Expr::Const { value, width } => {
            let bits = width.bits();
            let v = *value & width.mask();
            // Sign-extend for subtraction immediates commonly negative in two's complement.
            if bits < 64 {
                let sign = 1u64 << (bits - 1);
                let v = if v & sign != 0 {
                    v | (!0u64 << bits)
                } else {
                    v
                };
                Some(v as i64)
            } else {
                Some(v as i64)
            }
        }
        _ => None,
    }
}

fn classify_addr(
    addr: &Expr,
    facts: &BTreeMap<u16, SpFact>,
    f: &SsaFunction,
    sp_cell: u16,
) -> AddrClass {
    match eval_expr(addr, facts, f, sp_cell) {
        SpFact::Affine(c) => AddrClass::StackOff(c),
        SpFact::NotSp => AddrClass::NonStack,
        SpFact::Unknown => {
            // Try base+imm form even when eval_expr returned Unknown for complex trees.
            if let Expr::Binary {
                op: BinOp::Add,
                lhs,
                rhs,
            } = addr
            {
                if let (SpFact::Affine(c), Some(k)) =
                    (eval_expr(lhs, facts, f, sp_cell), as_i64_const(rhs))
                {
                    return AddrClass::StackOff(c.wrapping_add(k));
                }
                if let (Some(k), SpFact::Affine(c)) =
                    (as_i64_const(lhs), eval_expr(rhs, facts, f, sp_cell))
                {
                    return AddrClass::StackOff(c.wrapping_add(k));
                }
            }
            AddrClass::Unproven
        }
    }
}

fn partition_slots(accesses: &[(u64, usize, AddrClass, Width)]) -> (Vec<StackSlot>, bool) {
    type SlotAcc = (u64, Vec<(u64, usize, i64, Width)>);
    let mut by_off: BTreeMap<i64, SlotAcc> = BTreeMap::new();
    for &(b, i, class, w) in accesses {
        let AddrClass::StackOff(off) = class else {
            continue;
        };
        let size = (w.bits() / 8) as u64;
        let e = by_off.entry(off).or_insert((size, Vec::new()));
        e.0 = e.0.max(size);
        e.1.push((b, i, off, w));
    }
    let mut slots = Vec::new();
    let mut capped = false;
    for (off, (size, evidence)) in by_off {
        if slots.len() >= MAX_SLOTS {
            capped = true;
            break;
        }
        slots.push(StackSlot {
            base: off,
            size,
            evidence,
        });
    }
    (slots, capped)
}

/// From-scratch sanity check on facts vs the function.
pub fn check(f: &SsaFunction, facts: &StackFacts) -> Result<(), String> {
    if facts.entry != f.entry {
        return Err("entry mismatch".into());
    }
    if facts.sp_cell != sp_cell_num(f.arch) {
        return Err("sp_cell mismatch".into());
    }
    for &id in facts.name_facts.keys() {
        if f.names.get(id as usize).is_none() {
            return Err(format!("unknown name id {id}"));
        }
    }
    for slot in &facts.slots {
        if slot.evidence.is_empty() {
            return Err("slot without evidence".into());
        }
        if slot.size == 0 {
            return Err("zero-size slot".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, Expr, Reg, Space, Stmt, Width};
    use crate::irssa::{Name, SsaBlock, SsaFunction};
    use crate::model::Arch;
    use std::collections::BTreeMap;

    fn name(cell: u16, version: u32, w: Width) -> Name {
        Name {
            space: Space::Arch,
            cell,
            version,
            width: w,
        }
    }

    /// Hand-built: sp#0 live-in; sp#1 := sp#0 - 0x20; store [sp#1], rax#0
    fn mini_frame() -> SsaFunction {
        let mut names = vec![
            name(4, 0, Width::W64), // 0: rsp#0
            name(4, 1, Width::W64), // 1: rsp#1
            name(0, 0, Width::W64), // 2: rax#0
        ];
        let _ = &mut names;
        let stmts = vec![
            Stmt::Assign {
                dst: Reg {
                    space: Space::Arch,
                    num: 1,
                    width: Width::W64,
                },
                value: Expr::binary(
                    BinOp::Sub,
                    Expr::reg(Reg {
                        space: Space::Arch,
                        num: 0,
                        width: Width::W64,
                    }),
                    Expr::constant(0x20, Width::W64),
                ),
            },
            Stmt::Store {
                addr: Expr::reg(Reg {
                    space: Space::Arch,
                    num: 1,
                    width: Width::W64,
                }),
                value: Expr::reg(Reg {
                    space: Space::Arch,
                    num: 2,
                    width: Width::W64,
                }),
            },
        ];
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x1000,
            SsaBlock {
                start: 0x1000,
                end: 0x1010,
                phis: vec![],
                stmts,
                successors: vec![],
                truncated: false,
            },
        );
        SsaFunction {
            entry: 0x1000,
            name: Some("frame".into()),
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![0, 2],
            partial: vec![],
        }
    }

    #[test]
    fn affine_sub_and_stack_store() {
        let f = mini_frame();
        let facts = analyze(&f);
        assert!(check(&f, &facts).is_ok());
        assert_eq!(facts.name_facts.get(&0), Some(&SpFact::Affine(0)));
        assert_eq!(facts.name_facts.get(&1), Some(&SpFact::Affine(-0x20)));
        assert!(facts
            .accesses
            .iter()
            .any(|(_, _, c, _)| *c == AddrClass::StackOff(-0x20)));
        assert!(!facts.slots.is_empty());
        let dump = facts.render();
        assert!(dump.contains("local_"));
        assert_eq!(
            name_affine_local(&facts, 1).as_deref(),
            Some("local_32")
        );
    }

    #[test]
    fn aarch64_sp_cell_is_31() {
        let mut f = mini_frame();
        f.arch = Arch::Aarch64;
        // Remap names cell 4 → 31 for SP.
        f.names[0].cell = 31;
        f.names[1].cell = 31;
        let facts = analyze(&f);
        assert_eq!(facts.sp_cell, 31);
        assert_eq!(facts.name_facts.get(&1), Some(&SpFact::Affine(-0x20)));
    }
}
