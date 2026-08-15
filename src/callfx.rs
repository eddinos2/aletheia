//! ABI-aware call effects on the lifted IR.
//!
//! A lifted [`ir::BranchKind::Call`] records the transfer and nothing
//! else: the IR statements say what the *caller* does (push the return
//! address, branch), and stay silent about what the *callee* does to
//! machine state. Any dataflow drawn across a call site is therefore a
//! guess — a def-use link from a pre-call definition of `rax` to a
//! post-call read is simply wrong when the callee overwrote it. This
//! module makes the callee's effects explicit: [`apply`] inserts, after
//! every call, one [`ir::Stmt::Intrinsic`] named [`EFFECT_NAME`] whose
//! `writes` are the registers a conforming callee may clobber and whose
//! `reads` are the registers it may consume, followed (where the ABI's
//! return pops the stack, i.e. x86-64) by the explicit stack-pointer
//! restore. Downstream, [`crate::irssa`]'s renaming and
//! [`crate::irflow`]'s transfer already treat intrinsic writes as
//! definitions and its reads as uses, so the clobbers become fresh SSA
//! versions and the argument registers stay live into the call — with
//! **zero algorithm changes** anywhere else.
//!
//! # Statement-ordering convention
//!
//! The effects go *after* the `Branch { kind: Call }`. The CFG ends a
//! block at every call with the fallthrough as its only successor, so
//! statements after the branch read naturally as "on return": the
//! callee's whole execution is summarized by the one intrinsic, then
//! control reaches the fallthrough. Backward liveness does the right
//! thing under this order — a post-call read of a clobbered cell is cut
//! at the intrinsic, while the branch's own target expression (an
//! indirect `call rax`) still reads the *pre*-clobber value because it
//! precedes the intrinsic. [`ir::check`] imposes no branch-position
//! rule.
//!
//! # Soundness directions
//!
//! Both register lists are over-approximations, each in the direction
//! its consumer needs. Clobbers (`writes`) are over-approximated: an
//! extra clobber only kills more propagation facts and severs a def-use
//! link a preserved register would have kept — conservative, never a
//! wrong value — whereas an under-approximation would let a stale
//! constant survive a call. Uses (`reads`) are over-approximated: an
//! extra use only keeps more definitions alive under dead-code
//! elimination, whereas an under-approximation would delete a live
//! argument setup. The full argument-register superset is therefore read
//! at every call regardless of the callee's real arity, direct or
//! indirect. Memory needs no modeling here: [`crate::irflow`] never
//! deletes a store, a load-bearing assign, or an intrinsic, and neither
//! consumer reasons about memory contents.
//!
//! # Trust boundary
//!
//! This models a *conforming* callee. A callee that violates its ABI —
//! hand-written assembly clobbering `rbx`, say — is outside the model,
//! the same assumption every ABI-aware analysis makes. The lift itself
//! stays faithful: an ABI is a modeling assumption about the callee, not
//! instruction semantics, so the lifters and the `redump --lift` /
//! `--simplify` views are untouched; effects are opt-in via [`apply`],
//! and `redump --ssa` opts in.

use crate::ir::{self, BinOp, BranchKind, Expr, Flag, Reg, Stmt, Width};
use crate::irlift;
use crate::model::Arch;

/// The [`ir::Stmt::Intrinsic`] name [`apply`] inserts after each call.
pub const EFFECT_NAME: &str = "callfx";

// ---------------------------------------------------------------------------
// The ABI summary
// ---------------------------------------------------------------------------

/// One architecture's conservative call-effect summary: what a
/// conforming callee may write, what it may read, and what its return
/// does to the stack pointer. Built by [`x86_64`] / [`aarch64`];
/// consumed by [`apply`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallAbi {
    /// Cells a conforming callee may overwrite: caller-saved GPRs as
    /// full-width `W64` [`ir::Reg`]s (return-value registers included —
    /// a return def IS a clobber def) plus the `W1` flag cells the
    /// ISA's lifter models. These become the intrinsic's `writes`.
    pub clobbers: Vec<Reg>,
    /// Registers a conforming callee may read: the argument registers
    /// as a superset (argument counts are unknown), plus the stack
    /// pointer (stack arguments / return address). The intrinsic's
    /// `reads`, as [`ir::Expr::Reg`] at `W64`.
    pub uses: Vec<Reg>,
    /// The stack-pointer cell, at `W64`.
    pub sp: Reg,
    /// Bytes the return pops (the callee's `ret` consuming the pushed
    /// return address): 8 on x86-64, 0 on aarch64. Nonzero emits the
    /// `sp := sp + sp_pop` assign after the intrinsic.
    pub sp_pop: u64,
}

/// The x86-64 summary: deliberately the **union of SysV AMD64 and
/// Microsoft x64**, so one table is sound for ELF / Mach-O *and* PE
/// images (Win64's clobbers {rax rcx rdx r8–r11} and register args
/// {rcx rdx r8 r9} are both strict subsets of SysV's).
///
/// Clobbers: rax, rcx, rdx, rsi, rdi, r8–r11 at `W64`, then the five
/// flags the x86 lifter models. Callee-saved rbx, rsp, rbp, r12–r15 are
/// absent — the ABI preserves them, so their SSA version flows through
/// the call unbroken. Uses: the six SysV integer argument registers in
/// argument order, then rax (the SysV varargs vector count — `al` is a
/// real input compilers set with `xor eax,eax`), r10 (static chain),
/// and rsp (return address, stack arguments).
///
/// `sp_pop` is 8: composed with the lift's own `rsp -= 8; store [rsp],
/// retaddr`, the net rsp change across a call is zero, so rsp/rbp frame
/// tracking stays exact through calls — modeling rsp as a clobber
/// instead would destroy stack-slot reasoning.
pub fn x86_64() -> CallAbi {
    let gpr = |num: u16| Reg::arch(num, Width::W64);
    CallAbi {
        clobbers: vec![
            gpr(0),  // rax
            gpr(1),  // rcx
            gpr(2),  // rdx
            gpr(6),  // rsi
            gpr(7),  // rdi
            gpr(8),  // r8
            gpr(9),  // r9
            gpr(10), // r10
            gpr(11), // r11
            Reg::flag(Flag::Carry),
            Reg::flag(Flag::Zero),
            Reg::flag(Flag::Sign),
            Reg::flag(Flag::Overflow),
            Reg::flag(Flag::Parity),
        ],
        uses: vec![
            gpr(7),  // rdi: arg 1
            gpr(6),  // rsi: arg 2
            gpr(2),  // rdx: arg 3
            gpr(1),  // rcx: arg 4
            gpr(8),  // r8:  arg 5
            gpr(9),  // r9:  arg 6
            gpr(0),  // rax: SysV varargs vector count
            gpr(10), // r10: static chain
            gpr(4),  // rsp: return address, stack arguments
        ],
        sp: gpr(4),
        sp_pop: 8,
    }
}

/// The AAPCS64 summary (Apple arm64 agrees on the GPR sets; x18 is
/// platform-reserved and conservatively clobbered).
///
/// Clobbers: x0–x18 and x30 at `W64` (x30 because the callee's own
/// calls trash the link register; the `bl` lift's own `x30 := retaddr`
/// def precedes the branch and this def supersedes it after the call),
/// then exactly the four NZCV flags the aarch64 lifter models — no
/// Parity — and the vector file per AAPCS64: v0–v7 and v16–v31 whole
/// (both 64-bit half cells, numbering per the lifter: 32 + n low,
/// 64 + n high), plus the **high halves only** of v8–v15 — a callee
/// must preserve just the bottom 64 bits of v8–v15, so the low cells
/// 40–47 flow through the call unbroken while their high halves do
/// not. Callee-saved x19–x28, x29 (FP), sp (31), and d8–d15 are
/// absent.
///
/// Uses: x0–x7 (arguments), x8 (indirect-result pointer), sp, and
/// v0–v7 whole (FP/vector arguments pass in the full registers).
///
/// `sp_pop` is 0 — `bl`/`ret` do not touch sp, so [`apply`] emits no
/// adjust statement.
pub fn aarch64() -> CallAbi {
    let x = |num: u16| Reg::arch(num, Width::W64);
    let vlo = |n: u16| Reg::arch(32 + n, Width::W64);
    let vhi = |n: u16| Reg::arch(64 + n, Width::W64);
    let mut clobbers: Vec<Reg> = (0..=18).map(x).collect();
    clobbers.push(x(30));
    clobbers.extend((0..=7).map(vlo));
    clobbers.extend((16..=31).map(vlo));
    clobbers.extend((0..=31).map(vhi));
    clobbers.extend(
        [Flag::Carry, Flag::Zero, Flag::Sign, Flag::Overflow]
            .into_iter()
            .map(Reg::flag),
    );
    let mut uses: Vec<Reg> = (0..=8).map(x).collect();
    uses.push(x(31)); // sp: stack arguments
    uses.extend((0..=7).map(vlo));
    uses.extend((0..=7).map(vhi));
    CallAbi {
        clobbers,
        uses,
        sp: x(31),
        sp_pop: 0,
    }
}

/// The call-effect summary for `arch`, or `None` when the crate models
/// no ABI for it.
pub fn abi_for(arch: Arch) -> Option<CallAbi> {
    match arch {
        Arch::X86_64 => Some(x86_64()),
        Arch::Aarch64 => Some(aarch64()),
        Arch::Other => None,
    }
}

/// The registers whose value at a `ret` a *caller* may observe: the
/// return-value registers plus every callee-saved register the ABI
/// promises to hand back unchanged. `None` when the crate models no ABI
/// for `arch`.
///
/// This is the live-out root set [`crate::irssaopt::eliminate_dead`]
/// marks from, and its soundness direction is the mirror of
/// [`CallAbi::clobbers`]': live-out is an **over**-approximation. An
/// extra entry only keeps a dead definition alive (lost precision); a
/// missing one would delete a live definition (unsound), so both tables
/// are generous supersets — aarch64 lists x0–x8 rather than reasoning
/// about the callee's real return arity, and x86-64 lists the rdx of the
/// `(rax, rdx)` return pair whether or not the function returns 128 bits.
///
/// No flag cell appears in either table: no ABI returns a value in the
/// flags, so a flag definition that no branch reads is dead at the
/// return — which is exactly the lift noise this set must not pin.
/// x86-64's rsp and aarch64's sp are present (the caller's stack must
/// come back), and aarch64's x30 is absent on purpose: it is the `ret`
/// instruction's target, so the branch's own read marks it.
pub fn function_live_out(arch: Arch) -> Option<Vec<Reg>> {
    let gpr = |num: u16| Reg::arch(num, Width::W64);
    match arch {
        // rax, rdx: the SysV return pair. rbx, rsp, rbp, r12–r15: the
        // callee-saved set, identical in SysV and Win64 (Win64 also
        // preserves rsi/rdi, a subset condition that keeps this table
        // sound for PE images too).
        Arch::X86_64 => Some(vec![
            gpr(0),
            gpr(2),
            gpr(3),
            gpr(4),
            gpr(5),
            gpr(12),
            gpr(13),
            gpr(14),
            gpr(15),
        ]),
        // x0–x8: the return-value superset (x8 is the indirect-result
        // pointer, observable on return by convention). x19–x28, x29
        // (FP), sp: callee-saved. v0–v7 whole (cells 32–39 and 64–71):
        // the FP/vector return superset. d8–d15 (low cells 40–47): the
        // callee-saved bottom halves the caller may observe back — their
        // high halves are not preserved and stay out.
        Arch::Aarch64 => Some(
            (0..=8)
                .chain(19..=29)
                .chain([31])
                .chain(32..=47)
                .chain(64..=71)
                .map(gpr)
                .collect::<Vec<Reg>>(),
        ),
        Arch::Other => None,
    }
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

/// Insert `abi`'s call effects after every `Branch { kind: Call }` of
/// every block: the [`EFFECT_NAME`] intrinsic, then — when `sp_pop` is
/// nonzero — the stack-pointer restore. Everything else (entry, name,
/// block keys, bounds, successors, existing `truncated` flags) is
/// copied verbatim; blocks with no call are unchanged. Pure,
/// deterministic, total, no panics.
///
/// A block whose insertions would push it past [`ir::MAX_STMTS`] gets
/// nothing inserted and its `truncated` flag set instead — the existing
/// honest "this block's model is incomplete" marker (such a block is at
/// the irlift trim cap and already effectively truncated; reachable
/// only from hand-built input). Every block that passed [`ir::check`]
/// before still passes it after.
pub fn apply(func: &irlift::LiftedFunction, abi: &CallAbi) -> irlift::LiftedFunction {
    irlift::LiftedFunction {
        entry: func.entry,
        name: func.name.clone(),
        arch: func.arch,
        blocks: func
            .blocks
            .iter()
            .map(|(&start, block)| (start, apply_block(block, abi)))
            .collect(),
    }
}

/// [`apply`] for one block: count the calls, refuse (truncated) when the
/// effects cannot fit, otherwise splice them in after each call.
fn apply_block(block: &irlift::LiftedBlock, abi: &CallAbi) -> irlift::LiftedBlock {
    let is_call = |s: &Stmt| {
        matches!(
            s,
            Stmt::Branch {
                kind: BranchKind::Call,
                ..
            }
        )
    };
    let calls = block.stmts.iter().filter(|s| is_call(s)).count();
    if calls == 0 {
        return block.clone();
    }
    let per_call = 1 + usize::from(abi.sp_pop != 0);
    let needed = calls.saturating_mul(per_call);
    if block.stmts.len().saturating_add(needed) > ir::MAX_STMTS {
        let mut out = block.clone();
        out.truncated = true;
        return out;
    }
    let mut stmts = Vec::with_capacity(block.stmts.len() + needed);
    for stmt in &block.stmts {
        let call = is_call(stmt);
        stmts.push(stmt.clone());
        if call {
            stmts.push(Stmt::Intrinsic {
                name: EFFECT_NAME,
                writes: abi.clobbers.clone(),
                reads: abi.uses.iter().copied().map(Expr::reg).collect(),
            });
            if abi.sp_pop != 0 {
                stmts.push(Stmt::Assign {
                    dst: abi.sp,
                    value: Expr::binary(
                        BinOp::Add,
                        Expr::reg(abi.sp),
                        Expr::constant(abi.sp_pop, abi.sp.width),
                    ),
                });
            }
        }
    }
    irlift::LiftedBlock {
        start: block.start,
        end: block.end,
        stmts,
        successors: block.successors.clone(),
        truncated: block.truncated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Space;
    use crate::{aarch64_lift, x86, x86_lift};
    use std::collections::BTreeMap;

    // -- construction helpers ------------------------------------------------

    fn ra(num: u16, w: Width) -> Reg {
        Reg::arch(num, w)
    }

    fn block(start: u64, stmts: Vec<Stmt>, successors: Vec<u64>) -> irlift::LiftedBlock {
        irlift::LiftedBlock {
            start,
            end: start + 4,
            stmts,
            successors,
            truncated: false,
        }
    }

    fn func(entry: u64, blocks: Vec<irlift::LiftedBlock>) -> irlift::LiftedFunction {
        irlift::LiftedFunction {
            entry,
            name: None,
            arch: Arch::X86_64,
            blocks: blocks.into_iter().map(|b| (b.start, b)).collect(),
        }
    }

    fn call_to(target: u64) -> Stmt {
        Stmt::Branch {
            kind: BranchKind::Call,
            cond: None,
            target: Expr::constant(target, Width::W64),
        }
    }

    /// Decode and lift `bytes` at `va` as one straight-line block.
    fn lift_x86(bytes: &[u8], va: u64) -> Vec<Stmt> {
        let mut pairs = Vec::new();
        let mut cur = va;
        let mut off = 0usize;
        while off < bytes.len() {
            let insn = x86::decode(&bytes[off..], cur).expect("test bytes decode");
            let len = usize::from(insn.length);
            off += len;
            pairs.push((insn, cur));
            cur += len as u64;
        }
        x86_lift::lift_block(&pairs)
    }

    /// The spelled-out name of an arch cell at W64, per the given lifter
    /// namer — the cross-check oracle for the ABI tables.
    fn named(namer: fn(u16, Width) -> Option<String>, r: Reg) -> String {
        assert_eq!(r.space, Space::Arch, "only arch cells are named");
        namer(r.num, r.width).expect("table numbers are named by the lifter")
    }

    // -- the x86-64 table ----------------------------------------------------

    #[test]
    fn x86_64_clobbers_are_the_sysv_win64_union_verified_by_the_lifter_namer() {
        let abi = x86_64();
        let (gprs, flags): (Vec<Reg>, Vec<Reg>) =
            abi.clobbers.iter().partition(|r| r.space == Space::Arch);
        let names: Vec<String> = gprs.iter().map(|&r| named(x86_lift::reg_name, r)).collect();
        assert_eq!(
            names,
            ["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"]
        );
        assert!(gprs.iter().all(|r| r.width == Width::W64));
        assert_eq!(
            flags,
            [
                Reg::flag(Flag::Carry),
                Reg::flag(Flag::Zero),
                Reg::flag(Flag::Sign),
                Reg::flag(Flag::Overflow),
                Reg::flag(Flag::Parity),
            ]
        );
    }

    #[test]
    fn x86_64_uses_are_the_argument_superset_verified_by_the_lifter_namer() {
        let abi = x86_64();
        let names: Vec<String> = abi
            .uses
            .iter()
            .map(|&r| named(x86_lift::reg_name, r))
            .collect();
        assert_eq!(
            names,
            ["rdi", "rsi", "rdx", "rcx", "r8", "r9", "rax", "r10", "rsp"]
        );
        assert!(abi.uses.iter().all(|r| r.width == Width::W64));
    }

    #[test]
    fn x86_64_preserves_the_callee_saved_registers() {
        let abi = x86_64();
        // rbx(3), rsp(4), rbp(5), r12–r15: preserved, never clobbered.
        for num in [3u16, 4, 5, 12, 13, 14, 15] {
            assert!(
                !abi.clobbers
                    .iter()
                    .any(|r| r.space == Space::Arch && r.num == num),
                "{} must not be clobbered",
                named(x86_lift::reg_name, ra(num, Width::W64))
            );
        }
        assert_eq!(named(x86_lift::reg_name, abi.sp), "rsp");
        assert_eq!(abi.sp, ra(4, Width::W64));
        assert_eq!(abi.sp_pop, 8);
    }

    // -- the aarch64 table ---------------------------------------------------

    #[test]
    fn aarch64_clobbers_are_the_aapcs64_set_verified_by_the_lifter_namer() {
        let abi = aarch64();
        let (gprs, flags): (Vec<Reg>, Vec<Reg>) =
            abi.clobbers.iter().partition(|r| r.space == Space::Arch);
        let names: Vec<String> = gprs
            .iter()
            .map(|&r| named(aarch64_lift::reg_name, r))
            .collect();
        let expected: Vec<String> = (0..=18)
            .map(|n| format!("x{n}"))
            .chain(["x30".to_string()])
            // The vector file: v0–v7 and v16–v31 whole, v8–v15 high
            // halves only (their bottom 64 bits are callee-saved).
            .chain((0..=7).map(|n| format!("d{n}")))
            .chain((16..=31).map(|n| format!("d{n}")))
            .chain((0..=31).map(|n| format!("v{n}hi")))
            .collect();
        assert_eq!(names, expected);
        assert!(gprs.iter().all(|r| r.width == Width::W64));
        // d8–d15 (cells 40–47) are preserved: never clobbered.
        for num in 40u16..=47 {
            assert!(
                !abi.clobbers
                    .iter()
                    .any(|r| r.space == Space::Arch && r.num == num),
                "d{} is callee-saved, so it cannot be a clobber",
                num - 32
            );
        }
        // Exactly the four NZCV flags the aarch64 lifter models — no Parity.
        assert_eq!(
            flags,
            [
                Reg::flag(Flag::Carry),
                Reg::flag(Flag::Zero),
                Reg::flag(Flag::Sign),
                Reg::flag(Flag::Overflow),
            ]
        );
    }

    #[test]
    fn aarch64_preserves_the_callee_saved_registers_and_pops_nothing() {
        let abi = aarch64();
        // x19–x28, x29 (FP), and sp (31): preserved.
        for num in (19u16..=29).chain([31]) {
            assert!(
                !abi.clobbers
                    .iter()
                    .any(|r| r.space == Space::Arch && r.num == num),
                "{} must not be clobbered",
                named(aarch64_lift::reg_name, ra(num, Width::W64))
            );
        }
        assert_eq!(named(aarch64_lift::reg_name, abi.sp), "sp");
        assert_eq!(abi.sp, ra(31, Width::W64));
        assert_eq!(abi.sp_pop, 0);
    }

    #[test]
    fn aarch64_uses_are_the_argument_superset() {
        let abi = aarch64();
        let names: Vec<String> = abi
            .uses
            .iter()
            .map(|&r| named(aarch64_lift::reg_name, r))
            .collect();
        let expected: Vec<String> = (0..=8)
            .map(|n| format!("x{n}"))
            .chain(["sp".to_string()])
            // v0–v7 whole: the FP/vector argument registers.
            .chain((0..=7).map(|n| format!("d{n}")))
            .chain((0..=7).map(|n| format!("v{n}hi")))
            .collect();
        assert_eq!(names, expected);
    }

    // -- dispatch -------------------------------------------------------------

    #[test]
    fn abi_for_dispatches_per_architecture() {
        assert_eq!(abi_for(Arch::X86_64), Some(x86_64()));
        assert_eq!(abi_for(Arch::Aarch64), Some(aarch64()));
        assert_eq!(abi_for(Arch::Other), None);
    }

    // -- the live-out tables ---------------------------------------------------

    #[test]
    fn x86_64_live_out_is_the_return_pair_and_the_callee_saved_set() {
        let live = function_live_out(Arch::X86_64).expect("x86-64 is modeled");
        let names: Vec<String> = live.iter().map(|&r| named(x86_lift::reg_name, r)).collect();
        assert_eq!(
            names,
            [
                "rax", "rdx", "rbx", "rsp", "rbp", "r12", "r13", "r14", "r15"
            ]
        );
        assert!(live.iter().all(|r| r.width == Width::W64));
        // Every callee-saved register the clobber table preserves is
        // live-out: the two tables are complements, not independent lists.
        for r in &live {
            if [0u16, 2].contains(&r.num) {
                continue; // the return pair *is* clobbered
            }
            assert!(
                !x86_64().clobbers.contains(r),
                "{} is callee-saved, so it cannot be a clobber",
                named(x86_lift::reg_name, *r)
            );
        }
    }

    #[test]
    fn aarch64_live_out_is_the_return_superset_and_the_callee_saved_set() {
        let live = function_live_out(Arch::Aarch64).expect("aarch64 is modeled");
        let names: Vec<String> = live
            .iter()
            .map(|&r| named(aarch64_lift::reg_name, r))
            .collect();
        let expected: Vec<String> = (0..=8)
            .chain(19..=29)
            .map(|n| format!("x{n}"))
            .chain(["sp".to_string()])
            // v0–v7 whole (the return superset) and the callee-saved
            // d8–d15 bottom halves the caller may observe back.
            .chain((0..=15).map(|n| format!("d{n}")))
            .chain((0..=7).map(|n| format!("v{n}hi")))
            .collect();
        assert_eq!(names, expected);
        assert!(live.iter().all(|r| r.width == Width::W64));
        // x30 is the `ret` target, marked through the branch's own read.
        assert!(!live.iter().any(|r| r.num == 30));
        // The high halves of v8–v15 are preserved by no one: dead at a
        // return unless something reads them.
        assert!(!live.iter().any(|r| (72..=79).contains(&r.num)));
    }

    #[test]
    fn no_live_out_table_holds_a_flag() {
        for arch in [Arch::X86_64, Arch::Aarch64] {
            let live = function_live_out(arch).expect("modeled");
            assert!(
                live.iter().all(|r| r.space == Space::Arch),
                "no ABI returns a value in the flags: {live:?}"
            );
        }
    }

    #[test]
    fn function_live_out_dispatches_per_architecture_and_is_deterministic() {
        assert_eq!(function_live_out(Arch::Other), None);
        assert_eq!(
            function_live_out(Arch::X86_64),
            function_live_out(Arch::X86_64)
        );
        assert_eq!(
            function_live_out(Arch::Aarch64),
            function_live_out(Arch::Aarch64)
        );
    }

    // -- apply on lifted code --------------------------------------------------

    #[test]
    fn a_lifted_direct_call_gets_the_intrinsic_then_the_rsp_restore() {
        // call 0x2000 at 0x1000 (e8 fb 0f 00 00).
        let stmts = lift_x86(&[0xE8, 0xFB, 0x0F, 0x00, 0x00], 0x1000);
        let f = func(0x1000, vec![block(0x1000, stmts, vec![0x1005])]);
        let out = apply(&f, &x86_64());
        let b = &out.blocks[&0x1000];
        assert_eq!(ir::check(&b.stmts), Ok(()));
        // The lift's three statements, then the two inserted effects.
        assert_eq!(
            ir::render(&b.stmts, &x86_lift::reg_name),
            "rsp := (rsp - 0x8.q)\n\
             store.q [rsp], 0x1005.q\n\
             call 0x2000.q\n\
             rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11, CF, ZF, SF, OF, PF := \
             callfx(rdi, rsi, rdx, rcx, r8, r9, rax, r10, rsp)\n\
             rsp := (rsp + 0x8.q)\n"
        );
    }

    #[test]
    fn an_indirect_call_target_precedes_the_clobber_intrinsic() {
        // call rax (ff d0): the branch's target expression reads rax
        // *before* the intrinsic clobbers it, by statement order.
        let stmts = lift_x86(&[0xFF, 0xD0], 0x1000);
        let f = func(0x1000, vec![block(0x1000, stmts.clone(), vec![])]);
        let out = apply(&f, &x86_64());
        let b = &out.blocks[&0x1000];
        assert_eq!(ir::check(&b.stmts), Ok(()));
        // The lifted prefix — the branch included — is byte-identical.
        assert_eq!(&b.stmts[..stmts.len()], &stmts[..]);
        let call_at = b
            .stmts
            .iter()
            .position(|s| {
                matches!(
                    s,
                    Stmt::Branch {
                        kind: BranchKind::Call,
                        ..
                    }
                )
            })
            .expect("the call survives");
        assert!(
            matches!(&b.stmts[call_at + 1], Stmt::Intrinsic { name, .. } if *name == EFFECT_NAME),
            "the intrinsic follows the branch immediately"
        );
    }

    #[test]
    fn the_aarch64_abi_inserts_no_stack_adjust() {
        let f = func(0x1000, vec![block(0x1000, vec![call_to(0x2000)], vec![])]);
        let out = apply(&f, &aarch64());
        let stmts = &out.blocks[&0x1000].stmts;
        assert_eq!(stmts.len(), 2, "call, intrinsic, and nothing else");
        let Stmt::Intrinsic {
            name,
            writes,
            reads,
        } = &stmts[1]
        else {
            panic!("the intrinsic follows the call");
        };
        assert_eq!(*name, EFFECT_NAME);
        assert_eq!(*writes, aarch64().clobbers);
        assert_eq!(reads.len(), aarch64().uses.len());
        assert_eq!(ir::check(stmts), Ok(()));
    }

    // -- apply leaves non-calls alone -------------------------------------------

    #[test]
    fn blocks_without_a_call_are_byte_identical() {
        let jump = Stmt::Branch {
            kind: BranchKind::Jump,
            cond: None,
            target: Expr::constant(0x1010, Width::W64),
        };
        let ret = Stmt::Branch {
            kind: BranchKind::Return,
            cond: None,
            target: Expr::reg(Reg::temp(0, Width::W64)),
        };
        let tail = Stmt::Branch {
            kind: BranchKind::Jump,
            cond: None,
            target: Expr::constant(0x9000, Width::W64),
        };
        let mut truncated = block(0x1030, vec![], vec![]);
        truncated.truncated = true;
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![jump], vec![0x1010]),
                block(
                    0x1010,
                    vec![
                        Stmt::Assign {
                            dst: Reg::temp(0, Width::W64),
                            value: Expr::constant(0x1234, Width::W64),
                        },
                        ret,
                    ],
                    vec![],
                ),
                block(0x1020, vec![tail], vec![]),
                truncated,
            ],
        );
        assert_eq!(apply(&f, &x86_64()), f);
        assert_eq!(apply(&f, &aarch64()), f);
    }

    #[test]
    fn every_call_of_a_multi_call_block_gets_its_effects() {
        // Hand-built: lifted code has at most one call per block, but the
        // pass handles any input honestly.
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![call_to(0x2000), call_to(0x3000)],
                vec![],
            )],
        );
        let out = apply(&f, &x86_64());
        let stmts = &out.blocks[&0x1000].stmts;
        assert_eq!(stmts.len(), 6, "each call gains an intrinsic and a pop");
        for at in [0usize, 3] {
            assert!(matches!(
                &stmts[at],
                Stmt::Branch {
                    kind: BranchKind::Call,
                    ..
                }
            ));
            assert!(matches!(&stmts[at + 1], Stmt::Intrinsic { name, .. } if *name == EFFECT_NAME));
            assert!(matches!(&stmts[at + 2], Stmt::Assign { dst, .. } if *dst == x86_64().sp));
        }
        assert_eq!(ir::check(stmts), Ok(()));
    }

    // -- the overflow refusal ----------------------------------------------------

    #[test]
    fn a_block_the_effects_cannot_fit_is_marked_truncated_not_grown() {
        let filler = Stmt::Assign {
            dst: ra(0, Width::W64),
            value: Expr::constant(0, Width::W64),
        };
        let mut stmts = vec![filler; ir::MAX_STMTS - 1];
        stmts.push(call_to(0x2000));
        let f = func(0x1000, vec![block(0x1000, stmts.clone(), vec![])]);
        let out = apply(&f, &x86_64());
        let b = &out.blocks[&0x1000];
        assert_eq!(b.stmts, stmts, "nothing inserted");
        assert!(b.truncated, "the incompleteness is marked, never hidden");
        assert_eq!(ir::check(&b.stmts), Ok(()));
    }

    // -- structure preservation and determinism -----------------------------------

    #[test]
    fn apply_preserves_the_function_structure() {
        let mut f = func(
            0x1000,
            vec![
                block(0x1000, vec![call_to(0x9000)], vec![0x1010]),
                block(0x1010, vec![], vec![]),
            ],
        );
        f.name = Some("caller".into());
        f.blocks.get_mut(&0x1000).unwrap().end = 0x1005;
        f.blocks.get_mut(&0x1000).unwrap().truncated = true;
        let out = apply(&f, &x86_64());
        assert_eq!(out.entry, f.entry);
        assert_eq!(out.name, f.name);
        assert_eq!(
            out.blocks.keys().collect::<Vec<_>>(),
            f.blocks.keys().collect::<Vec<_>>()
        );
        for (va, b) in &f.blocks {
            let ob = &out.blocks[va];
            assert_eq!(ob.start, b.start);
            assert_eq!(ob.end, b.end);
            assert_eq!(ob.successors, b.successors);
            assert_eq!(ob.truncated, b.truncated);
        }
    }

    #[test]
    fn tables_and_apply_are_deterministic() {
        assert_eq!(x86_64(), x86_64());
        assert_eq!(aarch64(), aarch64());
        let f = func(
            0x1000,
            vec![block(0x1000, vec![call_to(0x2000)], vec![0x1005])],
        );
        assert_eq!(apply(&f, &x86_64()), apply(&f, &x86_64()));
        assert_eq!(apply(&f, &aarch64()), apply(&f, &aarch64()));
    }

    #[test]
    fn an_empty_function_applies_to_an_empty_function() {
        let f = irlift::LiftedFunction {
            entry: 0x1000,
            name: None,
            arch: Arch::X86_64,
            blocks: BTreeMap::new(),
        };
        assert_eq!(apply(&f, &x86_64()), f);
    }
}
