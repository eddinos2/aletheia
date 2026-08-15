//! Lifting x86-64 instructions into the [`crate::ir`] register-transfer IR.
//!
//! The x86-64 half of the lifter layer: it turns a decoded
//! [`crate::x86::Instruction`] into the `Vec<ir::Stmt>` that spells out its
//! effect on registers, flags, and memory — the ISA-specific knowledge the
//! IR core deliberately does not hold. Best-effort by contract: an
//! instruction whose semantics are not modeled lifts to a conservative
//! [`crate::ir::Stmt::Intrinsic`] naming the state it may touch rather than
//! to nothing, and no input ever panics. Every output passes
//! [`crate::ir::check`].
//!
//! # Register numbering
//!
//! An x86 GPR number `num` (0-15: rAX, rCX, rDX, rBX, rSP, rBP, rSI, rDI,
//! r8-r15) maps directly to [`ir::Reg::arch`]`(num, width)`; RSP is arch 4
//! and RBP arch 5, the standard convention. [`reg_name`] is the
//! [`ir::RegNamer`] hook that renders those numbers back to conventional
//! assembly names.
//!
//! # Sub-register write model (the crux)
//!
//! Architectural GPRs are modeled as **canonically 64-bit cells**. Every
//! sub-width write is lifted to a full-width `W64` assignment that computes
//! the correctly merged value, so [`ir::check`]'s width agreement holds
//! uniformly and the real x86 semantics are encoded rather than assumed:
//!
//! - A `W64` write is a plain assignment.
//! - A `W32` write **zero-extends to 64 bits** (`mov eax, X` clears bits
//!   63:32): `rax := zext.q(value.d)`.
//! - A `W16`/`W8` low write **merges**, preserving the upper bits:
//!   `rax := (rax & ~mask) | zext.q(value)`.
//! - A high-byte write (`ah`) merges into bits 8:15:
//!   `rax := (rax & ~0xFF00) | (zext.q(value.b) << 8)`.
//!
//! Reads of a sub-register are truncations of the full cell
//! (`trunc.d(rax.q)`); a high-byte read is `trunc.b(rax.q >>u 8)`. Because
//! every architectural reference is thus emitted at `W64` and narrowed/
//! merged explicitly, the [`reg_name`] hook is essentially only ever asked
//! for 64-bit names, and the lifted text stays readable.
//!
//! # Choices worth stating
//!
//! - **High-byte registers** (`ah`/`ch`/`dh`/`bh`) are modeled precisely
//!   with the shift-and-merge forms above, not punted to an intrinsic.
//! - **`cmov`** has no IR select, so it is lifted **branchless**:
//!   `dst := (src & m) | (dst & ~m)` where `m = sext(cond)` is all-ones or
//!   all-zeros.
//! - **Memory access width is not recorded by the decoder.** For the forms
//!   whose width is fixed by the opcode (`setcc` = byte, `push`/`pop` and
//!   indirect branches = 64-bit) the width is known; for a width-ambiguous
//!   memory destination with only an immediate (`add dword [mem], 1`,
//!   `mov byte [mem], 0`, `neg [mem]`, `movzx r, [mem]`) the width cannot be
//!   recovered from operands alone, so the instruction lifts to a sound
//!   [`ir::Stmt::Intrinsic`] rather than guessing a width. Register-operand
//!   forms — the analysis-critical majority — are always precise.
//! - **Parity (PF)** is not synthesized on arithmetic (it has no compact IR
//!   form); it is still a readable flag cell for `jp`/`jnp`. **Segment
//!   overrides** (`fs:`/`gs:`) are modeled as flat addressing.
//! - **Temporaries** are numbered per lifted instruction from 0.
//!   [`lift_block`] threads one monotonic counter across the block so a
//!   later instruction can never read an earlier one's temporary — required
//!   because [`ir::check`] treats a block as one temp namespace.

use crate::ir::{self, BinOp, Expr, Flag, Stmt, UnOp, Width};
use crate::x86::{self, Cond, Opcode, Operand};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Lift one decoded instruction at virtual address `va` into a statement
/// list. Always well-formed (passes [`ir::check`]); never panics.
pub fn lift(insn: &x86::Instruction, va: u64) -> Vec<Stmt> {
    let mut temp = 0u16;
    lift_with(insn, va, &mut temp)
}

/// Lift a straight-line run of instructions, concatenating their statement
/// lists. Temporary numbers are threaded across the block from a single
/// monotonic counter so no instruction can read another's temporary, which
/// [`ir::check`] would otherwise permit within its single-block temp
/// namespace. `insns` pairs each decoded instruction with its own VA.
pub fn lift_block(insns: &[(x86::Instruction, u64)]) -> Vec<Stmt> {
    let mut temp = 0u16;
    let mut out = Vec::new();
    for (insn, va) in insns {
        out.extend(lift_with(insn, *va, &mut temp));
    }
    out
}

/// The [`ir::RegNamer`] hook: the conventional assembly name of arch
/// register `num` at `width`, or `None` for an unknown number or the
/// flag-only [`Width::W1`]. Reuses [`x86::Reg::name`].
pub fn reg_name(num: u16, width: Width) -> Option<String> {
    if num > 15 {
        return None;
    }
    let xw = match width {
        Width::W8 => x86::Width::W8,
        Width::W16 => x86::Width::W16,
        Width::W32 => x86::Width::W32,
        Width::W64 => x86::Width::W64,
        Width::W1 => return None,
    };
    Some(x86::Reg::gpr(num as u8, xw).name().to_string())
}

// ---------------------------------------------------------------------------
// Small expression constructors
// ---------------------------------------------------------------------------

fn k(value: u64, width: Width) -> Expr {
    Expr::constant(value, width)
}

fn rd(reg: ir::Reg) -> Expr {
    Expr::reg(reg)
}

fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
    Expr::binary(op, a, b)
}

fn un(op: UnOp, e: Expr) -> Expr {
    Expr::unary(op, e)
}

fn assign(dst: ir::Reg, value: Expr) -> Stmt {
    Stmt::Assign { dst, value }
}

fn flag(f: Flag) -> ir::Reg {
    ir::Reg::flag(f)
}

/// The canonical 64-bit cell for x86 GPR number `num`.
fn cell(num: u8) -> ir::Reg {
    ir::Reg::arch(num as u16, Width::W64)
}

const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;
const RBX: u8 = 3;
const RSP: u8 = 4;
const RBP: u8 = 5;
const R11: u8 = 11;

/// Map the decoder's operand width onto the IR width.
fn iw(w: x86::Width) -> Width {
    match w {
        x86::Width::W8 => Width::W8,
        x86::Width::W16 => Width::W16,
        x86::Width::W32 => Width::W32,
        x86::Width::W64 => Width::W64,
    }
}

// ---------------------------------------------------------------------------
// Register read / write with the sub-register merge model
// ---------------------------------------------------------------------------

/// Read a register operand as a value at its own width (a truncation of the
/// 64-bit cell for a narrow reference, a shifted truncation for a high byte).
fn read_reg(r: &x86::Reg) -> Expr {
    let c = cell(r.num);
    if r.high_byte {
        // bits 8..=15 of the cell.
        return un(
            UnOp::Truncate(Width::W8),
            bin(BinOp::LShr, rd(c), k(8, Width::W64)),
        );
    }
    match iw(r.width) {
        Width::W64 => rd(c),
        w => un(UnOp::Truncate(w), rd(c)),
    }
}

/// Write `value` (already at the operand's width) into a register operand,
/// producing the single full-width assignment the merge model requires.
fn write_reg(r: &x86::Reg, value: Expr) -> Stmt {
    let c = cell(r.num);
    if r.high_byte {
        let masked = bin(BinOp::And, rd(c), k(!0xFF00u64, Width::W64));
        let placed = bin(
            BinOp::Shl,
            un(UnOp::ZeroExtend(Width::W64), value),
            k(8, Width::W64),
        );
        return assign(c, bin(BinOp::Or, masked, placed));
    }
    let merged = match iw(r.width) {
        Width::W64 => value,
        Width::W32 => un(UnOp::ZeroExtend(Width::W64), value),
        Width::W16 => bin(
            BinOp::Or,
            bin(BinOp::And, rd(c), k(!0xFFFFu64, Width::W64)),
            un(UnOp::ZeroExtend(Width::W64), value),
        ),
        _ => bin(
            BinOp::Or,
            bin(BinOp::And, rd(c), k(!0xFFu64, Width::W64)),
            un(UnOp::ZeroExtend(Width::W64), value),
        ),
    };
    assign(c, merged)
}

/// An address register contributes a 64-bit term; a 32-bit address register
/// (a `67`-prefixed address) is zero-extended.
fn addr_reg(r: &x86::Reg) -> Expr {
    match iw(r.width) {
        Width::W64 => rd(cell(r.num)),
        _ => un(UnOp::ZeroExtend(Width::W64), read_reg(r)),
    }
}

// ---------------------------------------------------------------------------
// Condition-code expressions
// ---------------------------------------------------------------------------

fn not1(e: Expr) -> Expr {
    un(UnOp::Not, e)
}

/// The `W1` guard expression for a condition code, built from flag reads.
fn cond_expr(c: Cond) -> Expr {
    let cf = || rd(flag(Flag::Carry));
    let zf = || rd(flag(Flag::Zero));
    let sf = || rd(flag(Flag::Sign));
    let of = || rd(flag(Flag::Overflow));
    let pf = || rd(flag(Flag::Parity));
    match c {
        Cond::O => of(),
        Cond::No => not1(of()),
        Cond::B => cf(),
        Cond::Ae => not1(cf()),
        Cond::E => zf(),
        Cond::Ne => not1(zf()),
        Cond::Be => bin(BinOp::Or, cf(), zf()),
        Cond::A => bin(BinOp::And, not1(cf()), not1(zf())),
        Cond::S => sf(),
        Cond::Ns => not1(sf()),
        Cond::P => pf(),
        Cond::Np => not1(pf()),
        Cond::L => bin(BinOp::Ne, sf(), of()),
        Cond::Ge => bin(BinOp::Eq, sf(), of()),
        Cond::Le => bin(BinOp::Or, zf(), bin(BinOp::Ne, sf(), of())),
        Cond::G => bin(BinOp::And, not1(zf()), bin(BinOp::Eq, sf(), of())),
    }
}

// ---------------------------------------------------------------------------
// The lifter context (VA, length, and a temporary counter)
// ---------------------------------------------------------------------------

/// Which flavour of flag update an ALU form performs.
#[derive(Clone, Copy)]
enum AluKind {
    Add,
    Sub,
    Logic,
}

/// Which unary read-modify-write form is being lifted.
#[derive(Clone, Copy)]
enum UnKind {
    Inc,
    Dec,
    Neg,
    Not,
}

struct Ctx {
    /// VA of the instruction being lifted.
    va: u64,
    /// Its encoded length in bytes.
    len: u64,
    /// Next free temporary number.
    temp: u16,
}

impl Ctx {
    fn fresh(&mut self, width: Width) -> ir::Reg {
        let r = ir::Reg::temp(self.temp, width);
        self.temp = self.temp.wrapping_add(1);
        r
    }

    /// The VA of the instruction that follows this one.
    fn next_va(&self) -> u64 {
        self.va.wrapping_add(self.len)
    }

    /// The 64-bit effective address of a memory operand.
    fn mem_addr(&self, op: &Operand) -> Expr {
        let Operand::Mem {
            base,
            index,
            scale,
            disp,
            rip_relative,
        } = op
        else {
            return k(0, Width::W64);
        };
        if *rip_relative {
            // [rip + disp] resolves against the next instruction's VA.
            let target = self.next_va().wrapping_add(*disp as u64);
            return k(target, Width::W64);
        }
        let mut acc: Option<Expr> = None;
        if let Some(b) = base {
            acc = Some(addr_reg(b));
        }
        if let Some(ix) = index {
            let mut term = addr_reg(ix);
            if *scale > 1 {
                term = bin(BinOp::Mul, term, k(*scale as u64, Width::W64));
            }
            acc = Some(match acc {
                Some(a) => bin(BinOp::Add, a, term),
                None => term,
            });
        }
        if *disp != 0 || acc.is_none() {
            let d = k(*disp as u64, Width::W64);
            acc = Some(match acc {
                Some(a) => bin(BinOp::Add, a, d),
                None => d,
            });
        }
        acc.unwrap_or_else(|| k(0, Width::W64))
    }

    /// Read an operand as a value of width `w`.
    fn read_operand(&self, op: &Operand, w: Width) -> Expr {
        match op {
            Operand::Reg(r) => read_reg(r),
            Operand::Mem { .. } => Expr::load(self.mem_addr(op), w),
            Operand::Imm(v) => k(*v as u64, w),
            // XMM state is not an IR value; the SSE opcodes carrying such
            // operands lift to intrinsics and never reach this precise path.
            Operand::Xmm(_) => k(0, w),
        }
    }

    /// Write `value` (at width `w`) to a destination operand.
    fn write_value(&self, dst: &Operand, value: Expr, _w: Width) -> Vec<Stmt> {
        match dst {
            Operand::Reg(r) => vec![write_reg(r, value)],
            Operand::Mem { .. } => vec![Stmt::Store {
                addr: self.mem_addr(dst),
                value,
            }],
            // An immediate destination is not encodable; nothing to do.
            Operand::Imm(_) => vec![],
            // XMM destinations are handled by the SSE intrinsic path, never here.
            Operand::Xmm(_) => vec![],
        }
    }

    // ---- arithmetic and logic ----

    fn arith(&mut self, ops: &[Operand], kind: AluKind, op: BinOp, write_back: bool) -> Vec<Stmt> {
        let w = match access_width(ops) {
            Some(w) => w,
            None => return self.generic_intrinsic(op_name_binary(kind), ops),
        };
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return Vec::new();
        };
        let tl = self.fresh(w);
        let tr = self.fresh(w);
        let tres = self.fresh(w);
        let zero = k(0, w);

        let mut out = vec![
            assign(tl, self.read_operand(dst, w)),
            assign(tr, self.read_operand(src, w)),
            assign(tres, bin(op, rd(tl), rd(tr))),
            assign(flag(Flag::Zero), bin(BinOp::Eq, rd(tres), zero.clone())),
            assign(flag(Flag::Sign), bin(BinOp::Slt, rd(tres), zero.clone())),
        ];
        match kind {
            AluKind::Add => {
                out.push(assign(flag(Flag::Carry), bin(BinOp::Ult, rd(tres), rd(tl))));
                out.push(assign(
                    flag(Flag::Overflow),
                    bin(
                        BinOp::Slt,
                        bin(
                            BinOp::And,
                            bin(BinOp::Xor, rd(tl), rd(tres)),
                            bin(BinOp::Xor, rd(tr), rd(tres)),
                        ),
                        zero,
                    ),
                ));
            }
            AluKind::Sub => {
                out.push(assign(flag(Flag::Carry), bin(BinOp::Ult, rd(tl), rd(tr))));
                out.push(assign(
                    flag(Flag::Overflow),
                    bin(
                        BinOp::Slt,
                        bin(
                            BinOp::And,
                            bin(BinOp::Xor, rd(tl), rd(tr)),
                            bin(BinOp::Xor, rd(tl), rd(tres)),
                        ),
                        zero,
                    ),
                ));
            }
            AluKind::Logic => {
                // x86 clears CF and OF for the logical group.
                out.push(assign(flag(Flag::Carry), k(0, Width::W1)));
                out.push(assign(flag(Flag::Overflow), k(0, Width::W1)));
            }
        }
        if write_back {
            out.extend(self.write_value(dst, rd(tres), w));
        }
        out
    }

    // ---- inc / dec / neg / not ----

    fn unary(&mut self, ops: &[Operand], kind: UnKind) -> Vec<Stmt> {
        let w = match access_width(ops) {
            Some(w) => w,
            None => return self.generic_intrinsic(un_name(kind), ops),
        };
        let Some(dst) = ops.first() else {
            return Vec::new();
        };
        let tx = self.fresh(w);
        let tres = self.fresh(w);
        let zero = k(0, w);
        let one = k(1, w);
        let mut out = vec![assign(tx, self.read_operand(dst, w))];

        match kind {
            UnKind::Not => {
                out.push(assign(tres, un(UnOp::Not, rd(tx))));
            }
            UnKind::Inc => {
                out.push(assign(tres, bin(BinOp::Add, rd(tx), one.clone())));
                out.push(assign(flag(Flag::Zero), bin(BinOp::Eq, rd(tres), zero.clone())));
                out.push(assign(flag(Flag::Sign), bin(BinOp::Slt, rd(tres), zero.clone())));
                // inc sets OF but leaves CF untouched (an x86 quirk).
                out.push(assign(
                    flag(Flag::Overflow),
                    bin(
                        BinOp::Slt,
                        bin(
                            BinOp::And,
                            bin(BinOp::Xor, rd(tx), rd(tres)),
                            bin(BinOp::Xor, one, rd(tres)),
                        ),
                        zero,
                    ),
                ));
            }
            UnKind::Dec => {
                out.push(assign(tres, bin(BinOp::Sub, rd(tx), one.clone())));
                out.push(assign(flag(Flag::Zero), bin(BinOp::Eq, rd(tres), zero.clone())));
                out.push(assign(flag(Flag::Sign), bin(BinOp::Slt, rd(tres), zero.clone())));
                out.push(assign(
                    flag(Flag::Overflow),
                    bin(
                        BinOp::Slt,
                        bin(
                            BinOp::And,
                            bin(BinOp::Xor, rd(tx), one),
                            bin(BinOp::Xor, rd(tx), rd(tres)),
                        ),
                        zero,
                    ),
                ));
            }
            UnKind::Neg => {
                // neg x == sub 0, x.
                out.push(assign(tres, bin(BinOp::Sub, zero.clone(), rd(tx))));
                out.push(assign(flag(Flag::Zero), bin(BinOp::Eq, rd(tres), zero.clone())));
                out.push(assign(flag(Flag::Sign), bin(BinOp::Slt, rd(tres), zero.clone())));
                out.push(assign(flag(Flag::Carry), bin(BinOp::Ult, zero.clone(), rd(tx))));
                out.push(assign(
                    flag(Flag::Overflow),
                    bin(
                        BinOp::Slt,
                        bin(
                            BinOp::And,
                            bin(BinOp::Xor, zero.clone(), rd(tx)),
                            bin(BinOp::Xor, zero, rd(tres)),
                        ),
                        k(0, w),
                    ),
                ));
            }
        }
        out.extend(self.write_value(dst, rd(tres), w));
        out
    }

    // ---- moves ----

    fn mov(&self, ops: &[Operand]) -> Vec<Stmt> {
        let w = match access_width(ops) {
            Some(w) => w,
            None => return self.generic_intrinsic("mov", ops),
        };
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return Vec::new();
        };
        self.write_value(dst, self.read_operand(src, w), w)
    }

    /// `movzx` / `movsx`: extend a narrow source into a wider destination.
    fn movx(&self, ops: &[Operand], signed: bool) -> Vec<Stmt> {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return Vec::new();
        };
        let Operand::Reg(dr) = dst else {
            return Vec::new();
        };
        let dw = iw(dr.width);
        let (val, sw) = match src {
            Operand::Reg(sr) => (read_reg(sr), iw(sr.width)),
            // A memory source's width is not recoverable from operands.
            _ => return self.generic_intrinsic(if signed { "movsx" } else { "movzx" }, ops),
        };
        let ext = if dw == sw {
            val
        } else if signed {
            un(UnOp::SignExtend(dw), val)
        } else {
            un(UnOp::ZeroExtend(dw), val)
        };
        self.write_value(dst, ext, dw)
    }

    /// `movsxd`: the doubleword source width is fixed by the opcode, so it
    /// is precise even from memory.
    fn movsxd(&self, ops: &[Operand]) -> Vec<Stmt> {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return Vec::new();
        };
        let Operand::Reg(dr) = dst else {
            return Vec::new();
        };
        let dw = iw(dr.width);
        let val = self.read_operand(src, Width::W32);
        let value = if dw == Width::W64 {
            un(UnOp::SignExtend(Width::W64), val)
        } else {
            val
        };
        self.write_value(dst, value, dw)
    }

    fn lea(&self, ops: &[Operand]) -> Vec<Stmt> {
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return Vec::new();
        };
        let Operand::Reg(dr) = dst else {
            return Vec::new();
        };
        let dw = iw(dr.width);
        let addr = self.mem_addr(src);
        let value = match dw {
            Width::W64 => addr,
            _ => un(UnOp::Truncate(dw), addr),
        };
        self.write_value(dst, value, dw)
    }

    // ---- stack ----

    fn push(&self, ops: &[Operand]) -> Vec<Stmt> {
        let Some(src) = ops.first() else {
            return Vec::new();
        };
        let w = match src {
            Operand::Reg(r) => iw(r.width),
            _ => Width::W64,
        };
        let bytes = w.bytes() as u64;
        let rsp = cell(RSP);
        vec![
            assign(rsp, bin(BinOp::Sub, rd(rsp), k(bytes, Width::W64))),
            Stmt::Store {
                addr: rd(rsp),
                value: self.read_operand(src, w),
            },
        ]
    }

    fn pop(&mut self, ops: &[Operand]) -> Vec<Stmt> {
        let Some(dst) = ops.first() else {
            return Vec::new();
        };
        let w = match dst {
            Operand::Reg(r) => iw(r.width),
            _ => Width::W64,
        };
        let bytes = w.bytes() as u64;
        let rsp = cell(RSP);
        let t = self.fresh(w);
        // Load from the old rsp, advance rsp, then write the destination —
        // this order makes `pop rsp` (dst == rsp) end with the popped value.
        let mut out = vec![
            assign(t, Expr::load(rd(rsp), w)),
            assign(rsp, bin(BinOp::Add, rd(rsp), k(bytes, Width::W64))),
        ];
        out.extend(self.write_value(dst, rd(t), w));
        out
    }

    fn leave(&mut self) -> Vec<Stmt> {
        // rsp := rbp ; rbp := pop.
        let rsp = cell(RSP);
        let rbp = cell(RBP);
        let t = self.fresh(Width::W64);
        vec![
            assign(rsp, rd(rbp)),
            assign(t, Expr::load(rd(rsp), Width::W64)),
            assign(rsp, bin(BinOp::Add, rd(rsp), k(8, Width::W64))),
            assign(rbp, rd(t)),
        ]
    }

    fn xchg(&mut self, ops: &[Operand]) -> Vec<Stmt> {
        let w = match access_width(ops) {
            Some(w) => w,
            None => return self.generic_intrinsic("xchg", ops),
        };
        let (Some(a), Some(b)) = (ops.first(), ops.get(1)) else {
            return Vec::new();
        };
        let t = self.fresh(w);
        let mut out = vec![assign(t, self.read_operand(a, w))];
        out.extend(self.write_value(a, self.read_operand(b, w), w));
        out.extend(self.write_value(b, rd(t), w));
        out
    }

    // ---- control flow ----

    fn ret(&mut self, ops: &[Operand]) -> Vec<Stmt> {
        let rsp = cell(RSP);
        let extra = match ops.first() {
            Some(Operand::Imm(v)) => *v as u64,
            _ => 0,
        };
        let t = self.fresh(Width::W64);
        vec![
            assign(t, Expr::load(rd(rsp), Width::W64)),
            assign(
                rsp,
                bin(BinOp::Add, rd(rsp), k(8u64.wrapping_add(extra), Width::W64)),
            ),
            Stmt::Branch {
                kind: ir::BranchKind::Return,
                cond: None,
                target: rd(t),
            },
        ]
    }

    fn call(&self, insn: &x86::Instruction) -> Vec<Stmt> {
        let target = match insn.flow {
            x86::Flow::Call(t) => k(t, Width::W64),
            x86::Flow::IndirectCall => match self.indirect_target(insn) {
                Some(e) => e,
                None => return self.generic_intrinsic("call", &insn.operands),
            },
            _ => return self.generic_intrinsic("call", &insn.operands),
        };
        let rsp = cell(RSP);
        vec![
            assign(rsp, bin(BinOp::Sub, rd(rsp), k(8, Width::W64))),
            Stmt::Store {
                addr: rd(rsp),
                value: k(self.next_va(), Width::W64),
            },
            Stmt::Branch {
                kind: ir::BranchKind::Call,
                cond: None,
                target,
            },
        ]
    }

    fn jmp(&self, insn: &x86::Instruction) -> Vec<Stmt> {
        let target = match insn.flow {
            x86::Flow::Jump(t) => k(t, Width::W64),
            x86::Flow::IndirectJump => match self.indirect_target(insn) {
                Some(e) => e,
                None => return self.generic_intrinsic("jmp", &insn.operands),
            },
            _ => return self.generic_intrinsic("jmp", &insn.operands),
        };
        vec![Stmt::Branch {
            kind: ir::BranchKind::Jump,
            cond: None,
            target,
        }]
    }

    fn jcc(&self, insn: &x86::Instruction, c: Cond) -> Vec<Stmt> {
        let target = match insn.flow {
            x86::Flow::CondJump(t) => k(t, Width::W64),
            _ => return self.generic_intrinsic("jcc", &insn.operands),
        };
        vec![Stmt::Branch {
            kind: ir::BranchKind::Jump,
            cond: Some(cond_expr(c)),
            target,
        }]
    }

    /// The 64-bit target expression of an indirect call/jump: the register
    /// value, or a load of the memory operand.
    fn indirect_target(&self, insn: &x86::Instruction) -> Option<Expr> {
        match insn.operands.first()? {
            Operand::Reg(r) => Some(rd(cell(r.num))),
            m @ Operand::Mem { .. } => Some(Expr::load(self.mem_addr(m), Width::W64)),
            Operand::Imm(_) | Operand::Xmm(_) => None,
        }
    }

    fn setcc(&self, ops: &[Operand], c: Cond) -> Vec<Stmt> {
        let Some(dst) = ops.first() else {
            return Vec::new();
        };
        let value = un(UnOp::ZeroExtend(Width::W8), cond_expr(c));
        self.write_value(dst, value, Width::W8)
    }

    /// `cmov`, lifted branchless: `dst := (src & m) | (dst & ~m)` where the
    /// mask `m = sext(cond)` is all-ones when the condition holds.
    fn cmov(&self, ops: &[Operand], c: Cond) -> Vec<Stmt> {
        let w = match access_width(ops) {
            Some(w) => w,
            None => return self.generic_intrinsic("cmov", ops),
        };
        let (Some(dst), Some(src)) = (ops.first(), ops.get(1)) else {
            return Vec::new();
        };
        let mask = un(UnOp::SignExtend(w), cond_expr(c));
        let value = bin(
            BinOp::Or,
            bin(BinOp::And, self.read_operand(src, w), mask.clone()),
            bin(BinOp::And, self.read_operand(dst, w), un(UnOp::Not, mask)),
        );
        self.write_value(dst, value, w)
    }

    // ---- intrinsics ----

    fn intr(&self, name: &'static str, writes: Vec<ir::Reg>, reads: Vec<Expr>) -> Vec<Stmt> {
        vec![Stmt::Intrinsic {
            name,
            writes,
            reads,
        }]
    }

    /// A conservative intrinsic for a width-ambiguous or genuinely unmodeled
    /// form: names the mnemonic, records every register operand as clobbered
    /// and every operand's value as read.
    fn generic_intrinsic(&self, name: &'static str, ops: &[Operand]) -> Vec<Stmt> {
        self.intr(name, operand_cells(ops), self.operand_reads(ops))
    }

    /// The read exprs of a list of operands, at approximate 64-bit width for
    /// memory (the exact width is unknown; the intrinsic is a sound
    /// over-approximation regardless).
    fn operand_reads(&self, ops: &[Operand]) -> Vec<Expr> {
        ops.iter()
            .filter_map(|op| match op {
                Operand::Reg(r) => Some(read_reg(r)),
                Operand::Mem { .. } => Some(Expr::load(self.mem_addr(op), Width::W64)),
                Operand::Imm(v) => Some(k(*v as u64, Width::W64)),
                // XMM operands carry no IR value; omit them from the read set.
                Operand::Xmm(_) => None,
            })
            .collect()
    }

    /// Lift the opcodes the IR does not model in detail into honest
    /// intrinsics: the state each writes and reads, never silently empty.
    fn modeled_intrinsic(&self, insn: &x86::Instruction) -> Vec<Stmt> {
        let ops = &insn.operands;
        let alu_flags = || {
            vec![
                flag(Flag::Carry),
                flag(Flag::Zero),
                flag(Flag::Sign),
                flag(Flag::Overflow),
                flag(Flag::Parity),
            ]
        };
        match insn.opcode {
            Opcode::Rdtsc => self.intr("rdtsc", vec![cell(RAX), cell(RDX)], vec![]),
            Opcode::Cpuid => self.intr(
                "cpuid",
                vec![cell(RAX), cell(RBX), cell(RCX), cell(RDX)],
                vec![rd(cell(RAX)), rd(cell(RCX))],
            ),
            Opcode::Syscall => self.intr(
                "syscall",
                vec![cell(RAX), cell(RCX), cell(R11)],
                vec![rd(cell(RAX))],
            ),
            Opcode::Cwde => self.intr("cwde", vec![cell(RAX)], vec![rd(cell(RAX))]),
            Opcode::Cdq => self.intr("cdq", vec![cell(RDX)], vec![rd(cell(RAX))]),
            Opcode::Hlt => self.intr("hlt", vec![], vec![]),
            Opcode::Ud2 => self.intr("ud2", vec![], vec![]),
            Opcode::Int3 => self.intr("int3", vec![], vec![]),
            Opcode::Int => self.intr("int", vec![], self.operand_reads(ops)),
            Opcode::Mul => self.intr(
                "mul",
                vec![cell(RAX), cell(RDX)],
                {
                    let mut r = self.operand_reads(ops);
                    r.push(rd(cell(RAX)));
                    r
                },
            ),
            Opcode::Imul => {
                if ops.len() <= 1 {
                    self.intr("imul", vec![cell(RAX), cell(RDX)], {
                        let mut r = self.operand_reads(ops);
                        r.push(rd(cell(RAX)));
                        r
                    })
                } else {
                    // Two/three-operand imul writes only the destination.
                    let writes = ops.first().map(operand_cell).unwrap_or_default();
                    self.intr("imul", writes, self.operand_reads(&ops[1..]))
                }
            }
            Opcode::Div => self.intr("div", vec![cell(RAX), cell(RDX)], {
                let mut r = self.operand_reads(ops);
                r.push(rd(cell(RAX)));
                r.push(rd(cell(RDX)));
                r
            }),
            Opcode::Idiv => self.intr("idiv", vec![cell(RAX), cell(RDX)], {
                let mut r = self.operand_reads(ops);
                r.push(rd(cell(RAX)));
                r.push(rd(cell(RDX)));
                r
            }),
            Opcode::Adc | Opcode::Sbb => {
                let name = if matches!(insn.opcode, Opcode::Adc) {
                    "adc"
                } else {
                    "sbb"
                };
                let mut writes = ops.first().map(operand_cell).unwrap_or_default();
                writes.extend(alu_flags());
                let mut reads = self.operand_reads(ops);
                reads.push(rd(flag(Flag::Carry)));
                self.intr(name, writes, reads)
            }
            Opcode::Bt => self.intr("bt", vec![flag(Flag::Carry)], self.operand_reads(ops)),
            Opcode::Bts | Opcode::Btr | Opcode::Btc => {
                let name = match insn.opcode {
                    Opcode::Bts => "bts",
                    Opcode::Btr => "btr",
                    _ => "btc",
                };
                let mut writes = ops.first().map(operand_cell).unwrap_or_default();
                writes.push(flag(Flag::Carry));
                self.intr(name, writes, self.operand_reads(ops))
            }
            Opcode::Xadd => {
                let mut writes = operand_cells(ops);
                writes.extend(alu_flags());
                self.intr("xadd", writes, self.operand_reads(ops))
            }
            Opcode::Cmpxchg => {
                let mut writes = vec![cell(RAX)];
                writes.extend(ops.first().map(operand_cell).unwrap_or_default());
                writes.extend(alu_flags());
                let mut reads = self.operand_reads(ops);
                reads.push(rd(cell(RAX)));
                self.intr("cmpxchg", writes, reads)
            }
            // The floating compares (comiss/comisd/ucomiss/ucomisd) clobber
            // ZF/PF/CF and clear OF/SF; mark the ALU flags written so a later
            // read (a jcc/setcc on the compare) never sees a stale flag.
            Opcode::Sse {
                mnem,
                writes_flags: true,
            } => {
                let mut writes = operand_cells(ops);
                writes.extend(alu_flags());
                self.intr(mnem, writes, self.operand_reads(ops))
            }
            // Anything else that reached here: a sound generic intrinsic.
            _ => self.generic_intrinsic(opcode_name(insn.opcode), ops),
        }
    }
}

// ---------------------------------------------------------------------------
// Operand helpers
// ---------------------------------------------------------------------------

/// The width implied by the first register operand, or `None` when no
/// operand fixes it (a width-ambiguous memory/immediate form).
fn access_width(ops: &[Operand]) -> Option<Width> {
    ops.iter().find_map(|op| match op {
        Operand::Reg(r) => Some(iw(r.width)),
        _ => None,
    })
}

/// The arch cell of a register operand, as a one-element vector (empty for a
/// non-register operand), for building intrinsic write lists.
fn operand_cell(op: &Operand) -> Vec<ir::Reg> {
    match op {
        Operand::Reg(r) => vec![cell(r.num)],
        _ => Vec::new(),
    }
}

/// The arch cells of every register operand in a list.
fn operand_cells(ops: &[Operand]) -> Vec<ir::Reg> {
    ops.iter().flat_map(operand_cell).collect()
}

fn op_name_binary(kind: AluKind) -> &'static str {
    match kind {
        AluKind::Add => "add",
        AluKind::Sub => "sub",
        AluKind::Logic => "logic",
    }
}

fn un_name(kind: UnKind) -> &'static str {
    match kind {
        UnKind::Inc => "inc",
        UnKind::Dec => "dec",
        UnKind::Neg => "neg",
        UnKind::Not => "not",
    }
}

/// A stable mnemonic string for any opcode, for intrinsic names.
fn opcode_name(op: Opcode) -> &'static str {
    match op {
        Opcode::Add => "add",
        Opcode::Or => "or",
        Opcode::Adc => "adc",
        Opcode::Sbb => "sbb",
        Opcode::And => "and",
        Opcode::Sub => "sub",
        Opcode::Xor => "xor",
        Opcode::Cmp => "cmp",
        Opcode::Test => "test",
        Opcode::Mov => "mov",
        Opcode::Movsx => "movsx",
        Opcode::Movzx => "movzx",
        Opcode::Movsxd => "movsxd",
        Opcode::Lea => "lea",
        Opcode::Push => "push",
        Opcode::Pop => "pop",
        Opcode::Xchg => "xchg",
        Opcode::Inc => "inc",
        Opcode::Dec => "dec",
        Opcode::Not => "not",
        Opcode::Neg => "neg",
        Opcode::Mul => "mul",
        Opcode::Imul => "imul",
        Opcode::Div => "div",
        Opcode::Idiv => "idiv",
        Opcode::Nop => "nop",
        Opcode::Cwde => "cwde",
        Opcode::Cdq => "cdq",
        Opcode::Ret => "ret",
        Opcode::Leave => "leave",
        Opcode::Int3 => "int3",
        Opcode::Int => "int",
        Opcode::Call => "call",
        Opcode::Jmp => "jmp",
        Opcode::Jcc(_) => "jcc",
        Opcode::Setcc(_) => "setcc",
        Opcode::Cmov(_) => "cmov",
        Opcode::Syscall => "syscall",
        Opcode::Ud2 => "ud2",
        Opcode::Cpuid => "cpuid",
        Opcode::Rdtsc => "rdtsc",
        Opcode::Bt => "bt",
        Opcode::Bts => "bts",
        Opcode::Btr => "btr",
        Opcode::Btc => "btc",
        Opcode::Cmpxchg => "cmpxchg",
        Opcode::Xadd => "xadd",
        Opcode::Hlt => "hlt",
        Opcode::Endbr64 => "endbr64",
        Opcode::Endbr32 => "endbr32",
        Opcode::Sse { mnem, .. } => mnem,
    }
}

// ---------------------------------------------------------------------------
// The top-level dispatch
// ---------------------------------------------------------------------------

fn lift_with(insn: &x86::Instruction, va: u64, temp: &mut u16) -> Vec<Stmt> {
    let mut ctx = Ctx {
        va,
        len: insn.length as u64,
        temp: *temp,
    };
    let ops = &insn.operands;
    let out = match insn.opcode {
        Opcode::Add => ctx.arith(ops, AluKind::Add, BinOp::Add, true),
        Opcode::Sub => ctx.arith(ops, AluKind::Sub, BinOp::Sub, true),
        Opcode::And => ctx.arith(ops, AluKind::Logic, BinOp::And, true),
        Opcode::Or => ctx.arith(ops, AluKind::Logic, BinOp::Or, true),
        Opcode::Xor => ctx.arith(ops, AluKind::Logic, BinOp::Xor, true),
        Opcode::Cmp => ctx.arith(ops, AluKind::Sub, BinOp::Sub, false),
        Opcode::Test => ctx.arith(ops, AluKind::Logic, BinOp::And, false),
        Opcode::Inc => ctx.unary(ops, UnKind::Inc),
        Opcode::Dec => ctx.unary(ops, UnKind::Dec),
        Opcode::Neg => ctx.unary(ops, UnKind::Neg),
        Opcode::Not => ctx.unary(ops, UnKind::Not),
        Opcode::Mov => ctx.mov(ops),
        Opcode::Movzx => ctx.movx(ops, false),
        Opcode::Movsx => ctx.movx(ops, true),
        Opcode::Movsxd => ctx.movsxd(ops),
        Opcode::Lea => ctx.lea(ops),
        Opcode::Push => ctx.push(ops),
        Opcode::Pop => ctx.pop(ops),
        Opcode::Xchg => ctx.xchg(ops),
        Opcode::Leave => ctx.leave(),
        Opcode::Ret => ctx.ret(ops),
        Opcode::Call => ctx.call(insn),
        Opcode::Jmp => ctx.jmp(insn),
        Opcode::Jcc(c) => ctx.jcc(insn, c),
        Opcode::Setcc(c) => ctx.setcc(ops, c),
        Opcode::Cmov(c) => ctx.cmov(ops, c),
        // No architectural effect worth modeling.
        Opcode::Nop | Opcode::Endbr64 | Opcode::Endbr32 => Vec::new(),
        // Everything else: an honest intrinsic.
        _ => ctx.modeled_intrinsic(insn),
    };
    *temp = ctx.temp;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::x86::decode;

    /// Decode at a fixed VA and lift; panics only if decoding fails (the
    /// test bytes are all valid encodings).
    fn lift_bytes(bytes: &[u8]) -> Vec<Stmt> {
        let insn = decode(bytes, 0x1000).expect("decode");
        lift(&insn, 0x1000)
    }

    /// Lift and render with the crate namer, trimming the trailing newline.
    fn text(bytes: &[u8]) -> String {
        let stmts = lift_bytes(bytes);
        assert_eq!(ir::check(&stmts), Ok(()), "check failed for {bytes:02x?}");
        ir::render(&stmts, &reg_name).trim_end().to_string()
    }

    /// Assert the lift of `bytes` is well-formed.
    fn ok(bytes: &[u8]) {
        assert_eq!(
            ir::check(&lift_bytes(bytes)),
            Ok(()),
            "check failed for {bytes:02x?}"
        );
    }

    fn has_flag_assign(stmts: &[Stmt], f: Flag) -> bool {
        stmts.iter().any(|s| match s {
            Stmt::Assign { dst, .. } => *dst == flag(f),
            _ => false,
        })
    }

    fn writes_cell(stmts: &[Stmt], num: u8) -> bool {
        stmts.iter().any(|s| match s {
            Stmt::Assign { dst, .. } => dst.space == ir::Space::Arch && dst.num == num as u16,
            _ => false,
        })
    }

    // ---- the register namer ----

    #[test]
    fn reg_name_maps_numbers_and_widths() {
        assert_eq!(reg_name(0, Width::W64).as_deref(), Some("rax"));
        assert_eq!(reg_name(4, Width::W64).as_deref(), Some("rsp"));
        assert_eq!(reg_name(5, Width::W64).as_deref(), Some("rbp"));
        assert_eq!(reg_name(0, Width::W32).as_deref(), Some("eax"));
        assert_eq!(reg_name(12, Width::W8).as_deref(), Some("r12b"));
        assert_eq!(reg_name(16, Width::W64), None);
        assert_eq!(reg_name(0, Width::W1), None);
    }

    // ---- moves and the sub-register write model ----

    #[test]
    fn mov_w32_write_zero_extends_the_full_cell() {
        // mov eax, eax must clear bits 63:32 via a zext of the truncation.
        assert_eq!(text(&[0x89, 0xC0]), "rax := zext.q(trunc.d(rax))");
    }

    #[test]
    fn mov_w64_write_is_a_plain_assignment() {
        // mov rax, rcx (REX.W 89 C8: dst rm=rax, src reg=rcx).
        assert_eq!(text(&[0x48, 0x89, 0xC8]), "rax := rcx");
    }

    #[test]
    fn mov_w8_low_write_merges_the_low_byte() {
        // mov al, cl (88 C8): dst al, src cl.
        assert_eq!(
            text(&[0x88, 0xC8]),
            "rax := ((rax & 0xffffffffffffff00.q) | zext.q(trunc.b(rcx)))"
        );
    }

    #[test]
    fn mov_w16_write_merges_the_low_word() {
        // mov ax, cx (66 89 C8).
        assert_eq!(
            text(&[0x66, 0x89, 0xC8]),
            "rax := ((rax & 0xffffffffffff0000.q) | zext.q(trunc.w(rcx)))"
        );
    }

    #[test]
    fn high_byte_write_merges_bits_eight_through_fifteen() {
        // mov ah, 1 (B4 01): no REX, encoding 4 selects ah.
        assert_eq!(
            text(&[0xB4, 0x01]),
            "rax := ((rax & 0xffffffffffff00ff.q) | (zext.q(0x1.b) << 0x8.q))"
        );
    }

    #[test]
    fn high_byte_read_shifts_then_truncates() {
        // mov cl, ah (8A CC): dst cl, src ah. The source read is the shift.
        let t = text(&[0x8A, 0xCC]);
        assert!(t.contains("trunc.b((rax >>u 0x8.q))"), "{t}");
    }

    #[test]
    fn mov_immediate_into_register() {
        // mov ecx, 1 (B9 01 00 00 00).
        assert_eq!(
            text(&[0xB9, 0x01, 0x00, 0x00, 0x00]),
            "rcx := zext.q(0x1.d)"
        );
        // mov rax, imm64 (REX.W B8).
        assert_eq!(
            text(&[0x48, 0xB8, 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01]),
            "rax := 0x123456789abcdef.q"
        );
    }

    #[test]
    fn movzx_and_movsx_extend() {
        // movzx eax, cl (0F B6 C1): read cl, zext to d, write zext.q.
        assert_eq!(
            text(&[0x0F, 0xB6, 0xC1]),
            "rax := zext.q(zext.d(trunc.b(rcx)))"
        );
        // movsx rax, bl (REX.W 0F BE C3).
        assert_eq!(text(&[0x48, 0x0F, 0xBE, 0xC3]), "rax := sext.q(trunc.b(rbx))");
    }

    #[test]
    fn movsxd_sign_extends_from_memory_precisely() {
        // movsxd rcx, eax (REX.W 63 C8).
        assert_eq!(text(&[0x48, 0x63, 0xC8]), "rcx := sext.q(trunc.d(rax))");
        // movsxd rax, dword [rdi] (REX.W 63 07): memory source, fixed width.
        ok(&[0x48, 0x63, 0x07]);
        let t = text(&[0x48, 0x63, 0x07]);
        assert!(t.contains("sext.q(load.d [rdi])"), "{t}");
    }

    // ---- lea ----

    #[test]
    fn lea_computes_the_address_expression_not_a_load() {
        // lea rax, [rdx + rcx*1 + 8] (48 8D 44 0A 08).
        assert_eq!(text(&[0x48, 0x8D, 0x44, 0x0A, 0x08]), "rax := ((rdx + rcx) + 0x8.q)");
    }

    #[test]
    fn lea_rip_relative_folds_to_an_absolute_constant() {
        // lea rdi, [rip + 0x4D2] at VA 0x1000, length 7 -> 0x1000+7+0x4D2.
        assert_eq!(text(&[0x48, 0x8D, 0x3D, 0xD2, 0x04, 0x00, 0x00]), "rdi := 0x14d9.q");
    }

    #[test]
    fn lea_scaled_index_multiplies() {
        // lea rax, [rax + rbx*4 + 0x100] (48 8D 84 98 00 01 00 00).
        let t = text(&[0x48, 0x8D, 0x84, 0x98, 0x00, 0x01, 0x00, 0x00]);
        assert!(t.contains("(rbx * 0x4.q)"), "{t}");
    }

    // ---- arithmetic and flags ----

    #[test]
    fn add_sets_all_five_arithmetic_flags_and_writes_dest() {
        // add eax, ecx (01 C8).
        let s = lift_bytes(&[0x01, 0xC8]);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(has_flag_assign(&s, Flag::Zero));
        assert!(has_flag_assign(&s, Flag::Sign));
        assert!(has_flag_assign(&s, Flag::Carry));
        assert!(has_flag_assign(&s, Flag::Overflow));
        assert!(writes_cell(&s, RAX));
    }

    #[test]
    fn sub_carry_is_lhs_below_rhs() {
        // sub eax, ecx (29 C8).
        let t = text(&[0x29, 0xC8]);
        assert!(t.contains("CF := (t0.d <u t1.d)"), "{t}");
    }

    #[test]
    fn logical_ops_clear_carry_and_overflow() {
        // and edi, esi (21 F7).
        let t = text(&[0x21, 0xF7]);
        assert!(t.contains("CF := 0x0.i1"), "{t}");
        assert!(t.contains("OF := 0x0.i1"), "{t}");
        ok(&[0x09, 0xC8]); // or
    }

    #[test]
    fn xor_self_zeroing_is_well_formed() {
        // xor eax, eax (31 C0).
        let s = lift_bytes(&[0x31, 0xC0]);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(writes_cell(&s, RAX));
    }

    #[test]
    fn cmp_sets_flags_without_writing_a_destination() {
        // cmp eax, ecx (39 C8).
        let s = lift_bytes(&[0x39, 0xC8]);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(has_flag_assign(&s, Flag::Zero));
        assert!(has_flag_assign(&s, Flag::Carry));
        // No architectural register is assigned.
        assert!(!writes_cell(&s, RAX));
    }

    #[test]
    fn test_sets_flags_without_writing_a_destination() {
        // test eax, eax (85 C0).
        let s = lift_bytes(&[0x85, 0xC0]);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(has_flag_assign(&s, Flag::Zero));
        assert!(!writes_cell(&s, RAX));
    }

    #[test]
    fn inc_and_dec_touch_overflow_but_not_carry() {
        // inc eax (FF C0).
        let s = lift_bytes(&[0xFF, 0xC0]);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(has_flag_assign(&s, Flag::Overflow));
        assert!(!has_flag_assign(&s, Flag::Carry));
        assert!(writes_cell(&s, RAX));
        // dec rcx (48 FF C9).
        let s = lift_bytes(&[0x48, 0xFF, 0xC9]);
        assert!(has_flag_assign(&s, Flag::Overflow));
        assert!(!has_flag_assign(&s, Flag::Carry));
    }

    #[test]
    fn neg_sets_flags_like_a_subtract_from_zero() {
        // neg rax (48 F7 D8).
        let s = lift_bytes(&[0x48, 0xF7, 0xD8]);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(has_flag_assign(&s, Flag::Carry));
        assert!(has_flag_assign(&s, Flag::Overflow));
        assert!(writes_cell(&s, RAX));
    }

    #[test]
    fn not_sets_no_flags() {
        // not eax (F7 D0).
        let s = lift_bytes(&[0xF7, 0xD0]);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(!has_flag_assign(&s, Flag::Zero));
        assert!(!has_flag_assign(&s, Flag::Carry));
        assert!(writes_cell(&s, RAX));
    }

    #[test]
    fn arith_with_a_memory_destination_and_register_is_precise() {
        // add [rbx], eax (01 03): dst mem, src reg fixes the width.
        let t = text(&[0x01, 0x03]);
        assert!(t.contains("store.d [rbx]"), "{t}");
    }

    #[test]
    fn width_ambiguous_memory_immediate_falls_back_to_intrinsic() {
        // mov dword [rax], 1 (C7 00 01 00 00 00): no register operand.
        let s = lift_bytes(&[0xC7, 0x00, 0x01, 0x00, 0x00, 0x00]);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(matches!(s.as_slice(), [Stmt::Intrinsic { name: "mov", .. }]));
        // add dword [rax], 1 (83 00 01) likewise.
        assert!(matches!(
            lift_bytes(&[0x83, 0x00, 0x01]).as_slice(),
            [Stmt::Intrinsic { name: "add", .. }]
        ));
    }

    // ---- stack ----

    #[test]
    fn push_decrements_rsp_then_stores() {
        // push rbp (55).
        assert_eq!(text(&[0x55]), "rsp := (rsp - 0x8.q)\nstore.q [rsp], rbp");
    }

    #[test]
    fn pop_loads_then_increments_rsp() {
        // pop rbp (5D).
        assert_eq!(
            text(&[0x5D]),
            "t0.q := load.q [rsp]\nrsp := (rsp + 0x8.q)\nrbp := t0.q"
        );
    }

    #[test]
    fn pop_rsp_ends_holding_the_popped_value() {
        // pop rsp (5C): the rsp adjust must precede the destination write.
        assert_eq!(
            text(&[0x5C]),
            "t0.q := load.q [rsp]\nrsp := (rsp + 0x8.q)\nrsp := t0.q"
        );
    }

    #[test]
    fn push_immediate_stores_a_sign_extended_constant() {
        // push -1 (6A FF).
        assert_eq!(
            text(&[0x6A, 0xFF]),
            "rsp := (rsp - 0x8.q)\nstore.q [rsp], 0xffffffffffffffff.q"
        );
    }

    #[test]
    fn push_word_uses_a_two_byte_slot() {
        // push ax (66 50).
        assert_eq!(
            text(&[0x66, 0x50]),
            "rsp := (rsp - 0x2.q)\nstore.w [rsp], trunc.w(rax)"
        );
    }

    #[test]
    fn leave_restores_rsp_from_rbp_then_pops() {
        assert_eq!(
            text(&[0xC9]),
            "rsp := rbp\nt0.q := load.q [rsp]\nrsp := (rsp + 0x8.q)\nrbp := t0.q"
        );
    }

    #[test]
    fn xchg_swaps_through_a_temporary() {
        // xchg eax, ebx (87 D8).
        assert_eq!(
            text(&[0x87, 0xD8]),
            "t0.d := trunc.d(rax)\nrax := zext.q(trunc.d(rbx))\nrbx := zext.q(t0.d)"
        );
    }

    // ---- control flow ----

    #[test]
    fn ret_reads_the_return_address_and_pops_it() {
        assert_eq!(
            text(&[0xC3]),
            "t0.q := load.q [rsp]\nrsp := (rsp + 0x8.q)\nreturn t0.q"
        );
    }

    #[test]
    fn ret_imm_adds_the_extra_to_rsp() {
        // ret 8 (C2 08 00).
        let t = text(&[0xC2, 0x08, 0x00]);
        assert!(t.contains("rsp := (rsp + 0x10.q)"), "{t}");
    }

    #[test]
    fn call_direct_pushes_the_return_address_then_branches() {
        // call rel32 at 0x1000, length 5 -> next VA 0x1005, target 0x100A.
        assert_eq!(
            text(&[0xE8, 0x05, 0x00, 0x00, 0x00]),
            "rsp := (rsp - 0x8.q)\nstore.q [rsp], 0x1005.q\ncall 0x100a.q"
        );
    }

    #[test]
    fn call_indirect_targets_the_register_value() {
        // call rax (FF D0).
        assert_eq!(
            text(&[0xFF, 0xD0]),
            "rsp := (rsp - 0x8.q)\nstore.q [rsp], 0x1002.q\ncall rax"
        );
    }

    #[test]
    fn call_indirect_through_memory_loads_the_target() {
        // call [rip+0x10] (FF 15 10 00 00 00): PLT/GOT shape.
        let t = text(&[0xFF, 0x15, 0x10, 0x00, 0x00, 0x00]);
        assert!(t.contains("call load.q [0x1016.q]"), "{t}");
    }

    #[test]
    fn jmp_direct_is_an_unconditional_branch() {
        // jmp rel8 self-loop (EB FE) at 0x1000.
        assert_eq!(text(&[0xEB, 0xFE]), "goto 0x1000.q");
    }

    #[test]
    fn jmp_indirect_through_memory_loads_the_target() {
        // jmp qword [rsp+8] (FF 64 24 08).
        let t = text(&[0xFF, 0x64, 0x24, 0x08]);
        assert!(t.contains("goto load.q [(rsp + 0x8.q)]"), "{t}");
    }

    #[test]
    fn jcc_carries_the_condition_expression() {
        // je rel8 (74 10) -> ZF condition, target 0x1012.
        assert_eq!(text(&[0x74, 0x10]), "goto if ZF 0x1012.q");
        // jne (75 10) -> ~ZF.
        assert_eq!(text(&[0x75, 0x10]), "goto if ~(ZF) 0x1012.q");
    }

    #[test]
    fn jcc_signed_less_is_sign_ne_overflow() {
        // jl rel8 (7C 10).
        assert_eq!(text(&[0x7C, 0x10]), "goto if (SF != OF) 0x1012.q");
        // jg (7F 10) -> ~ZF & (SF == OF).
        assert_eq!(text(&[0x7F, 0x10]), "goto if (~(ZF) & (SF == OF)) 0x1012.q");
    }

    #[test]
    fn jcc_above_is_not_carry_and_not_zero() {
        // ja rel8 (77 10).
        assert_eq!(text(&[0x77, 0x10]), "goto if (~(CF) & ~(ZF)) 0x1012.q");
    }

    #[test]
    fn setcc_zero_extends_the_condition_into_a_byte() {
        // sete al (0F 94 C0).
        assert_eq!(
            text(&[0x0F, 0x94, 0xC0]),
            "rax := ((rax & 0xffffffffffffff00.q) | zext.q(zext.b(ZF)))"
        );
    }

    #[test]
    fn cmov_is_a_branchless_masked_merge() {
        // cmove rax, rcx (48 0F 44 C1).
        let t = text(&[0x48, 0x0F, 0x44, 0xC1]);
        assert!(t.contains("sext.q(ZF)"), "{t}");
        assert!(t.contains("| (rax & ~(sext.q(ZF)))"), "{t}");
        assert!(writes_cell(&lift_bytes(&[0x48, 0x0F, 0x44, 0xC1]), RAX));
    }

    // ---- no-ops and intrinsics ----

    #[test]
    fn nop_and_endbr_lift_to_nothing() {
        assert!(lift_bytes(&[0x90]).is_empty());
        assert!(lift_bytes(&[0x0F, 0x1F, 0x00]).is_empty()); // multi-byte nop
        assert!(lift_bytes(&[0xF3, 0x0F, 0x1E, 0xFA]).is_empty()); // endbr64
    }

    #[test]
    fn rdtsc_is_an_intrinsic_writing_rax_and_rdx() {
        let s = lift_bytes(&[0x0F, 0x31]);
        assert_eq!(ir::check(&s), Ok(()));
        assert_eq!(
            s.as_slice(),
            [Stmt::Intrinsic {
                name: "rdtsc",
                writes: vec![cell(RAX), cell(RDX)],
                reads: vec![],
            }]
        );
        assert_eq!(
            ir::render(&s, &reg_name).trim_end(),
            "rax, rdx := rdtsc()"
        );
    }

    #[test]
    fn div_and_mul_name_their_implicit_operands() {
        // div esi (F7 F6): writes rax, rdx; reads the operand, rax, rdx.
        let s = lift_bytes(&[0xF7, 0xF6]);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(matches!(
            s.as_slice(),
            [Stmt::Intrinsic { name: "div", writes, .. }]
                if *writes == vec![cell(RAX), cell(RDX)]
        ));
        // mul rcx (48 F7 E1).
        assert!(matches!(
            lift_bytes(&[0x48, 0xF7, 0xE1]).as_slice(),
            [Stmt::Intrinsic { name: "mul", .. }]
        ));
    }

    #[test]
    fn syscall_and_cpuid_are_honest_intrinsics() {
        assert!(matches!(
            lift_bytes(&[0x0F, 0x05]).as_slice(),
            [Stmt::Intrinsic { name: "syscall", .. }]
        ));
        let s = lift_bytes(&[0x0F, 0xA2]);
        assert!(matches!(
            s.as_slice(),
            [Stmt::Intrinsic { name: "cpuid", writes, .. }]
                if writes.len() == 4
        ));
    }

    #[test]
    fn adc_and_bit_ops_are_intrinsics() {
        // adc eax, ecx (11 C8).
        assert!(matches!(
            lift_bytes(&[0x11, 0xC8]).as_slice(),
            [Stmt::Intrinsic { name: "adc", .. }]
        ));
        // bt eax, edx (0F A3 D0): writes CF.
        let s = lift_bytes(&[0x0F, 0xA3, 0xD0]);
        assert!(matches!(
            s.as_slice(),
            [Stmt::Intrinsic { name: "bt", writes, .. }] if *writes == vec![flag(Flag::Carry)]
        ));
        // cmpxchg (0F B1 C8).
        assert!(matches!(
            lift_bytes(&[0x0F, 0xB1, 0xC8]).as_slice(),
            [Stmt::Intrinsic { name: "cmpxchg", .. }]
        ));
    }

    // ---- robustness and determinism ----

    /// A broad corpus of valid encodings; every lift must be well-formed.
    const CORPUS: &[&[u8]] = &[
        &[0x55],
        &[0x5D],
        &[0x5C],
        &[0x48, 0x89, 0xE5],
        &[0x89, 0xC0],
        &[0x88, 0xC8],
        &[0x66, 0x89, 0xC8],
        &[0xB4, 0x01],
        &[0x8A, 0xCC],
        &[0x48, 0xB8, 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01],
        &[0x0F, 0xB6, 0xC1],
        &[0x0F, 0xB7, 0xC0],
        &[0x48, 0x0F, 0xBE, 0xC3],
        &[0x48, 0x63, 0xC8],
        &[0x48, 0x63, 0x07],
        &[0x48, 0x8D, 0x44, 0x0A, 0x08],
        &[0x48, 0x8D, 0x3D, 0xD2, 0x04, 0x00, 0x00],
        &[0x01, 0xC8],
        &[0x29, 0xC8],
        &[0x21, 0xF7],
        &[0x09, 0xC8],
        &[0x31, 0xC0],
        &[0x39, 0xC8],
        &[0x85, 0xC0],
        &[0xFF, 0xC0],
        &[0x48, 0xFF, 0xC9],
        &[0x48, 0xF7, 0xD8],
        &[0xF7, 0xD0],
        &[0x01, 0x03],
        &[0xC7, 0x00, 0x01, 0x00, 0x00, 0x00],
        &[0x83, 0x00, 0x01],
        &[0x6A, 0xFF],
        &[0x66, 0x50],
        &[0xC9],
        &[0x87, 0xD8],
        &[0xC3],
        &[0xC2, 0x08, 0x00],
        &[0xE8, 0x05, 0x00, 0x00, 0x00],
        &[0xFF, 0xD0],
        &[0xFF, 0x15, 0x10, 0x00, 0x00, 0x00],
        &[0xEB, 0xFE],
        &[0xE9, 0x00, 0x01, 0x00, 0x00],
        &[0xFF, 0xE0],
        &[0xFF, 0x64, 0x24, 0x08],
        &[0x74, 0x10],
        &[0x75, 0x10],
        &[0x7C, 0x10],
        &[0x7F, 0x10],
        &[0x77, 0x10],
        &[0x0F, 0x84, 0x00, 0x01, 0x00, 0x00],
        &[0x0F, 0x94, 0xC0],
        &[0x48, 0x0F, 0x44, 0xC1],
        &[0x0F, 0x4F, 0xCA],
        &[0x90],
        &[0x0F, 0x1F, 0x00],
        &[0xF3, 0x0F, 0x1E, 0xFA],
        &[0x0F, 0x31],
        &[0x0F, 0xA2],
        &[0x0F, 0x05],
        &[0x11, 0xC8],
        &[0x0F, 0xA3, 0xD0],
        &[0x48, 0x0F, 0xAB, 0xC8],
        &[0x0F, 0xB1, 0xC8],
        &[0x0F, 0xC1, 0xD8],
        &[0x98],
        &[0x99],
        &[0xCC],
        &[0xCD, 0x80],
        &[0x0F, 0x0B],
        &[0xF4],
        &[0x48, 0x6B, 0xC0, 0x0A],
        &[0x69, 0xC0, 0xE8, 0x03, 0x00, 0x00],
        &[0x0F, 0xAF, 0xC3],
        &[0x64, 0x48, 0x8B, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00],
    ];

    #[test]
    fn every_corpus_lift_is_well_formed() {
        for bytes in CORPUS {
            ok(bytes);
        }
    }

    #[test]
    fn lifting_is_deterministic() {
        for bytes in CORPUS {
            let insn = decode(bytes, 0x1000).unwrap();
            assert_eq!(lift(&insn, 0x1000), lift(&insn, 0x1000), "{bytes:02x?}");
        }
    }

    #[test]
    fn a_pseudo_exhaustive_sweep_never_panics_and_always_checks() {
        // Every one- and two-byte sequence that decodes must lift to a
        // well-formed list — no panic, no malformed IR.
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                if let Ok(insn) = decode(&[a, b], 0x1000) {
                    let stmts = lift(&insn, 0x1000);
                    assert_eq!(ir::check(&stmts), Ok(()), "[{a:02x} {b:02x}]");
                }
            }
        }
    }

    #[test]
    fn lift_block_threads_temporaries_so_they_stay_unique() {
        // Two pops in a row: the block's temporaries must not collide, so a
        // later instruction cannot read an earlier one's temp.
        let a = decode(&[0x5D], 0x1000).unwrap();
        let b = decode(&[0x5C], 0x1001).unwrap();
        let block = lift_block(&[(a, 0x1000), (b, 0x1001)]);
        assert_eq!(ir::check(&block), Ok(()));
        // Collect the temp numbers written; they must be distinct.
        let mut temps: Vec<u16> = block
            .iter()
            .filter_map(|s| match s {
                Stmt::Assign { dst, .. } if dst.space == ir::Space::Temp => Some(dst.num),
                _ => None,
            })
            .collect();
        let n = temps.len();
        temps.sort_unstable();
        temps.dedup();
        assert_eq!(temps.len(), n, "temporaries collided across the block");
    }

    #[test]
    fn garbage_and_odd_operand_shapes_do_not_panic() {
        // Feed a range of awkward encodings; each must lift and check.
        for bytes in [
            &[0xFF, 0x34, 0x25, 0x00, 0x00, 0x00, 0x00][..], // push [abs disp32]
            &[0x8F, 0xC0],                                   // pop r/m64
            &[0xFE, 0xC0],                                   // inc al
            &[0x48, 0x8D, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00], // lea rax, [0]
            &[0x00, 0x00],                                   // add [rax], al
        ] {
            ok(bytes);
        }
    }
}
