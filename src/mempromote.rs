//! Stack-slot promotion candidates (DESIGN slice 11, helper half).
//!
//! Given [`crate::irstack::StackFacts`] and the SSA they came from, classify
//! each evidence-backed slot as:
//! - [`Decision::Promote`] — clearly a local affine SP access with no
//!   address escape and no memory clobber barrier; safe to treat as a
//!   named local in pseudocode.
//! - [`Decision::Candidate`] — still labeled `local_*` for rendering, but
//!   must stay as load/store (address-taken, call barrier, or unproven
//!   store).
//!
//! Full HSSA MEM versioning is deferred; this module only marks decisions
//! and names. Total: never panics. Caps truncate.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::ir::{Expr, Stmt, Width};
use crate::irssa::SsaFunction;
use crate::irstack::{AddrClass, SpFact, StackFacts};

/// Cap on promotion decisions emitted per function.
pub const MAX_PROMOTE: usize = 512;

/// Whether a stack slot may become an SSA-style local for pseudo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// No escape, no clobber barrier — promote to a named local.
    Promote,
    /// Keep as memory, but still expose `local_*` for naming.
    Candidate,
}

impl Decision {
    fn token(self) -> &'static str {
        match self {
            Decision::Promote => "promote",
            Decision::Candidate => "candidate",
        }
    }
}

/// One slot's promotion verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotDecision {
    pub base: i64,
    pub size: u64,
    /// Stable `local_<abs_base>` label for [`crate::pseudo`] hooks.
    pub name: String,
    pub decision: Decision,
    /// Short honesty tag for dumps (`ok`, `addr-taken`, `call-barrier`, …).
    pub reason: &'static str,
    /// Access sites copied from the justifying [`crate::irstack::StackSlot`].
    pub evidence: Vec<(u64, usize, i64, Width)>,
}

/// Per-function promotion table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromoteFacts {
    pub entry: u64,
    pub slots: Vec<SlotDecision>,
    pub has_call_barrier: bool,
    pub has_unproven_store: bool,
    pub promote_capped: bool,
}

impl PromoteFacts {
    /// Deterministic multi-line dump for CLI / tests.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let promoted = self
            .slots
            .iter()
            .filter(|s| s.decision == Decision::Promote)
            .count();
        let _ = writeln!(
            out,
            "; promote entry={:#x} slots={} promoted={} call_barrier={} unproven_store={}",
            self.entry,
            self.slots.len(),
            promoted,
            self.has_call_barrier,
            self.has_unproven_store
        );
        if self.promote_capped {
            out.push_str("; note: promote cap hit\n");
        }
        for s in &self.slots {
            let _ = writeln!(
                out,
                "  {} base={:+} size={} {} ({}) evidence={}",
                s.name,
                s.base,
                s.size,
                s.decision.token(),
                s.reason,
                s.evidence.len()
            );
        }
        out
    }

    /// Lookup by stack offset (same range rule as [`crate::irstack::slot_label`]).
    pub fn name_at(&self, offset: i64) -> Option<&str> {
        self.slots
            .iter()
            .find(|s| offset >= s.base && offset < s.base + s.size as i64)
            .map(|s| s.name.as_str())
    }

    /// Promoted locals only — ready for a future rewrite / namer.
    pub fn promoted_names(&self) -> BTreeMap<i64, String> {
        self.slots
            .iter()
            .filter(|s| s.decision == Decision::Promote)
            .map(|s| (s.base, s.name.clone()))
            .collect()
    }
}

/// Classify slots from `stack` against `f`. Total.
pub fn promote(f: &SsaFunction, stack: &StackFacts) -> PromoteFacts {
    let has_call_barrier = function_has_call(f);
    let has_unproven_store = has_unproven_store(f, stack);
    let escaped_offsets = address_taken_offsets(f, stack);

    let mut slots = Vec::new();
    let mut promote_capped = false;
    for slot in &stack.slots {
        if slots.len() >= MAX_PROMOTE {
            promote_capped = true;
            break;
        }
        let name = format!("local_{}", slot.base.unsigned_abs());
        let addr_taken = escaped_offsets.iter().any(|&off| {
            off >= slot.base && off < slot.base + slot.size as i64
        });

        let (decision, reason) = if has_call_barrier {
            (Decision::Candidate, "call-barrier")
        } else if has_unproven_store {
            (Decision::Candidate, "unproven-store")
        } else if stack.sp_escaped {
            (Decision::Candidate, "sp-escaped")
        } else if addr_taken {
            (Decision::Candidate, "addr-taken")
        } else {
            (Decision::Promote, "ok")
        };

        slots.push(SlotDecision {
            base: slot.base,
            size: slot.size,
            name,
            decision,
            reason,
            evidence: slot.evidence.clone(),
        });
    }

    PromoteFacts {
        entry: f.entry,
        slots,
        has_call_barrier,
        has_unproven_store,
        promote_capped,
    }
}

/// Unproven classification on a Store site is a conservative clobber.
fn has_unproven_store(f: &SsaFunction, stack: &StackFacts) -> bool {
    for &(bva, ix, class, _) in &stack.accesses {
        if !matches!(class, AddrClass::Unproven) {
            continue;
        }
        let Some(block) = f.blocks.get(&bva) else {
            continue;
        };
        if matches!(block.stmts.get(ix), Some(Stmt::Store { .. })) {
            return true;
        }
    }
    false
}

fn function_has_call(f: &SsaFunction) -> bool {
    use crate::ir::BranchKind;
    for block in f.blocks.values() {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Branch {
                    kind: BranchKind::Call,
                    ..
                } => return true,
                Stmt::Intrinsic { name, .. } if *name == "callfx" => return true,
                _ => {}
            }
        }
    }
    false
}

/// Offsets whose Affine SSA names appear outside load/store address position.
fn address_taken_offsets(f: &SsaFunction, stack: &StackFacts) -> BTreeSet<i64> {
    // name id → affine offset, excluding the live SP-cell lineage used only
    // for frame setup: we still track every Affine fact, but SP-cell names
    // used in SP updates are filtered at use sites.
    let sp_cell = stack.sp_cell;
    let mut affine: BTreeMap<u16, i64> = BTreeMap::new();
    for (&id, fact) in &stack.name_facts {
        if let SpFact::Affine(c) = *fact {
            affine.insert(id, c);
        }
    }

    let mut escaped = BTreeSet::new();
    for block in f.blocks.values() {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Assign { dst, value } => {
                    let dst_is_sp = f
                        .names
                        .get(dst.num as usize)
                        .map(|n| n.space == crate::ir::Space::Arch && n.cell == sp_cell)
                        .unwrap_or(false);
                    // Frame setup: assigning into the SP cell is not escape.
                    if dst_is_sp {
                        continue;
                    }
                    // Any other def that consumes an Affine name materializes
                    // the address as a first-class value (LEA / mov of &local).
                    collect_affine_uses(value, &affine, &mut escaped);
                }
                Stmt::Store { addr, value } => {
                    // Address position: not escape.
                    let _ = addr;
                    collect_affine_uses(value, &affine, &mut escaped);
                }
                Stmt::Branch { cond, target, .. } => {
                    if let Some(c) = cond {
                        collect_affine_uses(c, &affine, &mut escaped);
                    }
                    collect_affine_uses(target, &affine, &mut escaped);
                }
                Stmt::Intrinsic { reads, .. } => {
                    for r in reads {
                        collect_affine_uses(r, &affine, &mut escaped);
                    }
                }
            }
        }
        // Load/Store address trees are never walked here — address position
        // is not escape. φ joining of address temps is deferred.
    }
    let _ = sp_cell;
    escaped
}

fn collect_affine_uses(e: &Expr, affine: &BTreeMap<u16, i64>, escaped: &mut BTreeSet<i64>) {
    match e {
        Expr::Reg(r) => {
            if let Some(&off) = affine.get(&r.num) {
                escaped.insert(off);
            }
        }
        Expr::Const { .. } => {}
        Expr::Load { addr, .. } => collect_affine_uses(addr, affine, escaped),
        Expr::Unary { operand, .. } => collect_affine_uses(operand, affine, escaped),
        Expr::Binary { lhs, rhs, .. } => {
            collect_affine_uses(lhs, affine, escaped);
            collect_affine_uses(rhs, affine, escaped);
        }
    }
}

/// From-scratch sanity check. Total.
pub fn check(f: &SsaFunction, stack: &StackFacts, facts: &PromoteFacts) -> Result<(), String> {
    if facts.entry != f.entry {
        return Err("entry mismatch".into());
    }
    if facts.entry != stack.entry {
        return Err("stack entry mismatch".into());
    }
    if facts.slots.len() > MAX_PROMOTE {
        return Err("slots exceed MAX_PROMOTE".into());
    }
    if !facts.promote_capped && facts.slots.len() != stack.slots.len() {
        return Err(format!(
            "slot count {} != stack slots {}",
            facts.slots.len(),
            stack.slots.len()
        ));
    }
    for (i, s) in facts.slots.iter().enumerate() {
        if s.name.is_empty() {
            return Err(format!("empty name at slot {i}"));
        }
        if s.size == 0 {
            return Err(format!("zero-size slot {}", s.name));
        }
        if s.evidence.is_empty() {
            return Err(format!("no evidence for {}", s.name));
        }
        if s.decision == Decision::Promote {
            if facts.has_call_barrier {
                return Err(format!("{} promoted under call barrier", s.name));
            }
            if facts.has_unproven_store {
                return Err(format!("{} promoted under unproven store", s.name));
            }
            if s.reason != "ok" {
                return Err(format!("{} promote reason is not ok", s.name));
            }
        }
        // Base must match a stack slot when not capped mid-way.
        if i < stack.slots.len() && s.base != stack.slots[i].base {
            return Err(format!(
                "base mismatch at {i}: {} vs {}",
                s.base, stack.slots[i].base
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, BranchKind, Expr, Reg, Space, Stmt, Width};
    use crate::irssa::{Name, SsaBlock, SsaFunction};
    use crate::irstack;
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

    /// sp#0 live-in; sp#1 := sp#0 - 0x20; store [sp#1], rax#0
    fn mini_frame() -> SsaFunction {
        let names = vec![
            name(4, 0, Width::W64),
            name(4, 1, Width::W64),
            name(0, 0, Width::W64),
        ];
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

    /// Same frame, but `rax := sp#1` (address taken into a GP reg).
    fn addr_taken_frame() -> SsaFunction {
        let mut f = mini_frame();
        // names: 3 = rax#1 after lea
        f.names.push(name(0, 1, Width::W64));
        let block = f.blocks.get_mut(&0x1000).unwrap();
        block.stmts.push(Stmt::Assign {
            dst: Reg {
                space: Space::Arch,
                num: 3,
                width: Width::W64,
            },
            value: Expr::reg(Reg {
                space: Space::Arch,
                num: 1,
                width: Width::W64,
            }),
        });
        f
    }

    /// Frame plus a call.
    fn frame_with_call() -> SsaFunction {
        let mut f = mini_frame();
        let block = f.blocks.get_mut(&0x1000).unwrap();
        block.stmts.push(Stmt::Branch {
            kind: BranchKind::Call,
            cond: None,
            target: Expr::constant(0x2000, Width::W64),
        });
        f
    }

    #[test]
    fn leaf_spill_promotes() {
        let f = mini_frame();
        let stack = irstack::analyze(&f);
        let facts = promote(&f, &stack);
        assert!(check(&f, &stack, &facts).is_ok(), "{:?}", check(&f, &stack, &facts));
        assert!(!facts.has_call_barrier);
        assert!(!facts.slots.is_empty());
        assert!(
            facts.slots.iter().any(|s| s.decision == Decision::Promote),
            "{}",
            facts.render()
        );
        assert!(facts.render().contains("promote"), "{}", facts.render());
        assert_eq!(facts.name_at(-0x20), Some("local_32"));
    }

    #[test]
    fn address_taken_is_candidate() {
        let f = addr_taken_frame();
        let stack = irstack::analyze(&f);
        let facts = promote(&f, &stack);
        assert!(check(&f, &stack, &facts).is_ok());
        let slot = facts
            .slots
            .iter()
            .find(|s| s.base == -0x20)
            .expect("slot");
        assert_eq!(slot.decision, Decision::Candidate);
        assert_eq!(slot.reason, "addr-taken");
        assert_eq!(slot.name, "local_32");
    }

    #[test]
    fn call_barrier_blocks_promote() {
        let f = frame_with_call();
        let stack = irstack::analyze(&f);
        let facts = promote(&f, &stack);
        assert!(check(&f, &stack, &facts).is_ok());
        assert!(facts.has_call_barrier);
        assert!(facts
            .slots
            .iter()
            .all(|s| s.decision == Decision::Candidate));
        assert!(facts.render().contains("call-barrier"), "{}", facts.render());
    }
}
