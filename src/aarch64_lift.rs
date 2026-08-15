//! Lifting AArch64 (A64) instructions into the [`crate::ir`]
//! register-transfer IR.
//!
//! The AArch64 half of the lifter layer: it turns a decoded
//! [`crate::aarch64::Instruction`] into the `Vec<ir::Stmt>` that spells out
//! its effect on registers, flags, and memory — the ISA-specific knowledge
//! the IR core deliberately does not hold. Best-effort by contract: an
//! instruction whose semantics are not modeled lifts to a conservative
//! [`crate::ir::Stmt::Intrinsic`] naming the state it may touch rather than
//! to nothing, and no input ever panics. Every output passes
//! [`crate::ir::check`].
//!
//! # Register numbering
//!
//! GPR numbers 0–30 map to `X0`–`X30` as [`ir::Reg::arch`]`(num, W64)`
//! cells; **number 31 is the stack pointer** (`SP`), its own cell. The zero
//! register `XZR`/`WZR` is *not* a cell: a read of register 31 in a
//! ZR-position operand lifts to a zero constant, and a write to it is
//! discarded (the rest of the lift — a load, a flag update — is kept).
//! Whether an encoded `31` means SP or ZR depends on the operand position:
//! the base of every load/store and the `rn`/`rd` of a non-flag-setting
//! `ADD`/`SUB` immediate are SP; every other position is ZR. [`reg_name`]
//! is the [`ir::RegNamer`] hook that renders the numbering back
//! (`x0`…`x30`, `sp`).
//!
//! # The SIMD&FP register file
//!
//! The IR's width model tops out at `W64` and grows no vector operators,
//! so each 128-bit `Vn` is **two 64-bit `Space::Arch` cells**: numbers
//! 32–63 are the low halves (rendered `d0`…`d31` — architecturally the
//! `Dn` view, which is what scalar FP code reads and writes) and 64–95
//! the high halves (rendered `v0hi`…`v31hi`). That is option (a) of the
//! plan's IR question, chosen after inventory (no vector cells existed;
//! SIMD registers previously lived only inside the `a64.unknown`
//! clobber): a `ldr q0, [x8]` that loads two named cells beats a named
//! intrinsic for every downstream pass, and the `SMULH` intrinsic
//! doctrine stays reserved for values the IR genuinely cannot express.
//! Consequences, all architectural: a scalar write (`ldr s0`, `fmov d1,
//! x2`, …) zero-extends into the low cell **and zeroes the high cell**
//! (writes to a SIMD&FP register clear the rest of its 128 bits); a
//! b/h/s read is a truncation of the low cell; the q forms load/store
//! the two cells at `addr` and `addr + 8` (little-endian: low half at
//! the lower address); only `FMOV Vd.D[1], Xn` writes a half in
//! isolation. Register 31 is `V31` — the file has no ZR/SP. The
//! `a64.unknown` clobber now covers all 64 vector cells too, so the
//! unmodeled remainder (remaining Advanced SIMD vector ALU beyond the
//! three-same integer slice, LSE atomics, SVE, …) stays sound.
//! Scalar FP *arithmetic* lifts to precise named intrinsics over these
//! cells — `a64.fadd` writes exactly `vlo(rd)` and reads its two
//! operand cells, `a64.fcmp` writes exactly the four NZCV flags — so
//! FP dataflow keeps real def-use chains even where the operation
//! itself is opaque; every scalar FP write is followed by the
//! architectural `vhi := 0`. The three-same integer ALU lifts bitwise
//! AND/ORR/EOR (and `.2d` ADD/SUB) exactly over the two 64-bit cells;
//! packed ADD/SUB of narrower lanes uses a precise named intrinsic
//! writing the destination halves. `callfx`'s AAPCS64 summary covers the
//! vector file both directions: v0–v7/v16–v31 and the high halves of
//! v8–v15 clobbered (only the bottom 64 bits of v8–v15 are
//! callee-saved), v0–v7 read as arguments and live-out as the return
//! superset alongside the callee-saved d8–d15.
//!
//! # The W-register write model
//!
//! Cells are canonically 64-bit, mirroring the x86 lifter's contract. A
//! 32-bit (`Wn`) result **zero-extends into the full X register** — the
//! architectural rule — so every sub-width write is a single full-width
//! assignment `xN := zext.q(value.d)`, and reads of a W register are
//! truncations of the cell. 32-bit (`sf == 0`) arithmetic and its flags are
//! computed at `W32` and the result widened on the way into the cell.
//!
//! # NZCV
//!
//! The A64 condition flags reuse the IR's common flag cells: N is
//! [`Flag::Sign`], Z [`Flag::Zero`], C [`Flag::Carry`], V
//! [`Flag::Overflow`]; [`Flag::Parity`] is never touched. **A64 subtraction
//! sets C to NOT-borrow** (the opposite of x86): `SUBS` emits
//! `CF := (rhs <=u lhs)` and `ADDS` emits `CF := (result <u lhs)`, and
//! [`cond_expr`] builds every condition (`HS` = `CF`, `LO` = `~CF`, …)
//! consistently with that convention. `AL`/`NV` are both "always".
//!
//! There is exactly one flag model — `nzcv_model` — and every
//! flag-setting arithmetic lift is built from its expressions:
//! `ADDS`/`SUBS` in all three operand forms assign them directly;
//! `ADCS`/`SBCS` extend the model's add half once with a carry-in (C
//! gains the wraparound term `c & (result == lhs)`, and `SBC` is
//! `lhs + NOT(rhs) + C`, the Arm ARM's `AddWithCarry` identity); and
//! `CCMP`/`CCMN` wrap each expression in a branchless select against
//! the corresponding literal `nzcv` bit — the compare's flag
//! expressions are the model's products, never re-derived.
//!
//! # Choices worth stating
//!
//! - **Loads snapshot first.** Every memory access begins by capturing the
//!   effective address in a temporary, then loads into fresh value
//!   temporaries, then applies base writeback, then writes destinations.
//!   That fixed order makes `ldr x0, [x0], #8`, the `rt == rn` writeback
//!   cases, and `ldr wzr, [x0]` (load kept, destination discarded) all
//!   deterministic and well-formed.
//! - **Conditional select is branchless**, like x86 `cmov`: a sign-extended
//!   condition mask merges `rn` with the alternative
//!   (`rm` / `rm + 1` / `~rm` / `-rm`).
//! - **`SVC`/`HVC`/`SMC`** lift to named intrinsics with honest AAPCS-ish
//!   read/write sets; `BRK`/`HLT` and the event hints (`WFE`, `WFI`, …) to
//!   named intrinsics with empty sets; `NOP` and unallocated hints (which
//!   execute as `NOP` by definition) to nothing.
//! - **[`Opcode::Unknown`]** lifts to a single clobber-everything intrinsic
//!   writing all 32 GPR/SP cells, all 64 SIMD&FP half cells, and all four
//!   NZCV flags — sound for the unmodeled remainder (remaining Advanced
//!   SIMD vector ALU, LSE atomics, SVE, …).
//! - **Temporaries** are numbered per lifted instruction from 0;
//!   [`lift_block`] threads one monotonic counter across a block so a later
//!   instruction can never read an earlier one's temporary.

use crate::aarch64::{self, AddrMode, Cond, LogOp, Opcode, RegOffset, Shift};
use crate::ir::{self, BinOp, Expr, Flag, Stmt, UnOp, Width};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Lift one decoded instruction at virtual address `va` into a statement
/// list. Always well-formed (passes [`ir::check`]); never panics.
pub fn lift(insn: &aarch64::Instruction, va: u64) -> Vec<Stmt> {
    let mut temp = 0u16;
    lift_with(insn, va, &mut temp)
}

/// Lift a straight-line run of instructions, concatenating their statement
/// lists. Temporary numbers are threaded across the block from a single
/// monotonic counter so no instruction can read another's temporary, which
/// [`ir::check`] would otherwise permit within its single-block temp
/// namespace. `insns` pairs each decoded instruction with its own VA.
pub fn lift_block(insns: &[(aarch64::Instruction, u64)]) -> Vec<Stmt> {
    let mut temp = 0u16;
    let mut out = Vec::new();
    for (insn, va) in insns {
        out.extend(lift_with(insn, *va, &mut temp));
    }
    out
}

/// The [`ir::RegNamer`] hook: the conventional assembly name of arch
/// register `num` at `width` (`x0`…`x30` and `sp` at [`Width::W64`],
/// `w0`…`w30` and `wsp` at [`Width::W32`]; the SIMD&FP cells 32–95 as
/// `d0`…`d31` / `s0`…`s31` for the low halves and `v0hi`…`v31hi` for
/// the high), or `None` for an unknown number or a width the numbering
/// does not name. In practice the lifter only emits 64-bit cell
/// references, so only the `x`/`sp`/`d`/`vhi` names appear.
pub fn reg_name(num: u16, width: Width) -> Option<String> {
    match (num, width) {
        (31, Width::W64) => Some("sp".to_string()),
        (31, Width::W32) => Some("wsp".to_string()),
        (0..=30, Width::W64) => Some(format!("x{num}")),
        (0..=30, Width::W32) => Some(format!("w{num}")),
        (32..=63, Width::W64) => Some(format!("d{}", num - 32)),
        (32..=63, Width::W32) => Some(format!("s{}", num - 32)),
        (64..=95, Width::W64) => Some(format!("v{}hi", num - 64)),
        _ => None,
    }
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

fn not1(e: Expr) -> Expr {
    un(UnOp::Not, e)
}

/// The canonical 64-bit cell for GPR number `num` (31 is the SP cell).
fn cell(num: u8) -> ir::Reg {
    ir::Reg::arch(num as u16, Width::W64)
}

/// First cell number of the SIMD&FP low halves (`d0`…`d31`).
const VLO_BASE: u16 = 32;

/// First cell number of the SIMD&FP high halves (`v0hi`…`v31hi`).
const VHI_BASE: u16 = 64;

/// The 64-bit cell holding the low half (bits 63:0 — the `Dn` view) of
/// SIMD&FP register `Vn`. Register 31 is `V31`: the file has no ZR/SP.
fn vlo(r: u8) -> ir::Reg {
    ir::Reg::arch(VLO_BASE + r as u16, Width::W64)
}

/// The 64-bit cell holding the high half (bits 127:64) of `Vn`.
fn vhi(r: u8) -> ir::Reg {
    ir::Reg::arch(VHI_BASE + r as u16, Width::W64)
}

/// X30, the link register.
const LR: u8 = 30;

/// The operand width selected by an `sf` bit.
fn sf_w(sf: bool) -> Width {
    if sf { Width::W64 } else { Width::W32 }
}

/// The access width of a load/store `size` field (log2 bytes).
fn size_w(size: u8) -> Width {
    match size {
        0 => Width::W8,
        1 => Width::W16,
        2 => Width::W32,
        _ => Width::W64,
    }
}

// ---------------------------------------------------------------------------
// Register read / write with the SP/ZR and W-zero-extend models
// ---------------------------------------------------------------------------

/// Read GPR `r` at width `w` where an encoded 31 means SP: always the cell,
/// truncated for a sub-64-bit reference.
fn read_sp(r: u8, w: Width) -> Expr {
    match w {
        Width::W64 => rd(cell(r)),
        w => un(UnOp::Truncate(w), rd(cell(r))),
    }
}

/// Read GPR `r` at width `w` where an encoded 31 means the zero register.
fn read_zr(r: u8, w: Width) -> Expr {
    if r == 31 { k(0, w) } else { read_sp(r, w) }
}

/// The single full-width assignment that writes `value` (at width `w`) into
/// the cell of GPR `r`: a plain assignment at `W64`, a zero-extension into
/// the full X register for anything narrower (the architectural W-register
/// rule, which also covers byte/halfword load destinations).
fn write_cell(r: u8, value: Expr, w: Width) -> Stmt {
    match w {
        Width::W64 => assign(cell(r), value),
        _ => assign(cell(r), un(UnOp::ZeroExtend(Width::W64), value)),
    }
}

/// Write to GPR `r` where an encoded 31 means the zero register: the write
/// is discarded (no statement) rather than touching any cell.
fn write_zr(r: u8, value: Expr, w: Width) -> Vec<Stmt> {
    if r == 31 {
        Vec::new()
    } else {
        vec![write_cell(r, value, w)]
    }
}

/// Write to GPR `r` where an encoded 31 means SP.
fn write_sp(r: u8, value: Expr, w: Width) -> Vec<Stmt> {
    vec![write_cell(r, value, w)]
}

/// Read the low `w` bits of SIMD&FP register `r`: the low cell,
/// truncated for a sub-64-bit access.
fn vread(r: u8, w: Width) -> Expr {
    match w {
        Width::W64 => rd(vlo(r)),
        w => un(UnOp::Truncate(w), rd(vlo(r))),
    }
}

/// The two statements of a scalar write to SIMD&FP register `r`: the
/// `w`-wide value zero-extends into the low cell and the high cell is
/// cleared — the architectural rule that every scalar write to a
/// SIMD&FP register zeroes the rest of its 128 bits.
fn vwrite_scalar(r: u8, value: Expr, w: Width) -> Vec<Stmt> {
    let low = match w {
        Width::W64 => assign(vlo(r), value),
        _ => assign(vlo(r), un(UnOp::ZeroExtend(Width::W64), value)),
    };
    vec![low, assign(vhi(r), k(0, Width::W64))]
}

/// The whole-register immediate write of the vector moves
/// (`MOVI`/`MVNI`/vector `FMOV`): `imm64` into the low cell, replicated
/// into the high cell on the 128-bit (`q`) form and zeroing it on the
/// 64-bit form — either way the write covers the whole register.
fn vmove_imm64(r: u8, imm64: u64, q: bool) -> Vec<Stmt> {
    let hi = if q { imm64 } else { 0 };
    vec![
        assign(vlo(r), k(imm64, Width::W64)),
        assign(vhi(r), k(hi, Width::W64)),
    ]
}

/// The IEEE bit pattern of an `FMOV` immediate at the given precision.
/// Every VFP immediate value is exact in both formats
/// ([`aarch64::fp_imm_value`]), so the `f32` cast never rounds.
fn fp_imm_bits(double: bool, imm: u8) -> u64 {
    let value = aarch64::fp_imm_value(imm);
    if double {
        value.to_bits()
    } else {
        (value as f32).to_bits() as u64
    }
}

// ---------------------------------------------------------------------------
// Condition-code expressions
// ---------------------------------------------------------------------------

/// The `W1` guard expression for an A64 condition, built from flag reads
/// and consistent with the flag definitions the arithmetic lifts emit
/// (`C` is NOT-borrow after a subtract). `AL`/`NV` are both a constant
/// true.
fn cond_expr(c: Cond) -> Expr {
    let n = || rd(flag(Flag::Sign));
    let z = || rd(flag(Flag::Zero));
    let cf = || rd(flag(Flag::Carry));
    let v = || rd(flag(Flag::Overflow));
    match c {
        Cond::Eq => z(),
        Cond::Ne => not1(z()),
        Cond::Cs => cf(),
        Cond::Cc => not1(cf()),
        Cond::Mi => n(),
        Cond::Pl => not1(n()),
        Cond::Vs => v(),
        Cond::Vc => not1(v()),
        Cond::Hi => bin(BinOp::And, cf(), not1(z())),
        Cond::Ls => bin(BinOp::Or, not1(cf()), z()),
        Cond::Ge => bin(BinOp::Eq, n(), v()),
        Cond::Lt => bin(BinOp::Ne, n(), v()),
        Cond::Gt => bin(BinOp::And, not1(z()), bin(BinOp::Eq, n(), v())),
        Cond::Le => bin(BinOp::Or, z(), bin(BinOp::Ne, n(), v())),
        Cond::Al | Cond::Nv => k(1, Width::W1),
    }
}

// ---------------------------------------------------------------------------
// The one NZCV flag model
// ---------------------------------------------------------------------------

/// The four NZCV expressions of the one add/subtract flag model, in
/// emission order (Z, N, C, V), for the result temporary `tres` of
/// `tl + tr`, `tl - tr`, or — with `carry` — `tl + tr + c`, where the
/// operands were snapshotted into the `tl`/`tr` temporaries first.
///
/// N and Z look only at the result. C and V carry the A64 conventions:
/// C is NOT-borrow after a subtract (`tr <=u tl`) and the carry-out of
/// an add (`tres <u tl`); V is the same-sign-operands/other-sign-result
/// test in its subtract and add spellings. `carry` is the ADC/SBC
/// extension — the `W1` temporary holding the carry-in — and adds the
/// wraparound term `c & (tres == tl)` to the add's carry-out (`tres ==
/// tl` with a set carry-in means `tr` was all-ones and the true sum
/// wrapped a full register); it is only meaningful on the add side, and
/// the V test is unchanged by a carry-in.
///
/// Every flag-setting arithmetic lift builds its flag writes from
/// exactly these expressions: `ADDS`/`SUBS` (all three operand forms)
/// and `ADCS`/`SBCS` assign them directly, `CCMP`/`CCMN` wraps each in
/// a select against its literal `nzcv` bit.
fn nzcv_model(
    tl: ir::Reg,
    tr: ir::Reg,
    tres: ir::Reg,
    w: Width,
    sub: bool,
    carry: Option<ir::Reg>,
) -> [(Flag, Expr); 4] {
    let zero = k(0, w);
    let cf = match (sub, carry) {
        (true, _) => bin(BinOp::Ule, rd(tr), rd(tl)),
        (false, None) => bin(BinOp::Ult, rd(tres), rd(tl)),
        (false, Some(c)) => bin(
            BinOp::Or,
            bin(BinOp::Ult, rd(tres), rd(tl)),
            bin(BinOp::And, rd(c), bin(BinOp::Eq, rd(tres), rd(tl))),
        ),
    };
    let of = if sub {
        bin(
            BinOp::Slt,
            bin(
                BinOp::And,
                bin(BinOp::Xor, rd(tl), rd(tr)),
                bin(BinOp::Xor, rd(tl), rd(tres)),
            ),
            zero.clone(),
        )
    } else {
        bin(
            BinOp::Slt,
            bin(
                BinOp::And,
                bin(BinOp::Xor, rd(tl), rd(tres)),
                bin(BinOp::Xor, rd(tr), rd(tres)),
            ),
            zero.clone(),
        )
    };
    [
        (Flag::Zero, bin(BinOp::Eq, rd(tres), zero.clone())),
        (Flag::Sign, bin(BinOp::Slt, rd(tres), zero)),
        (Flag::Carry, cf),
        (Flag::Overflow, of),
    ]
}

// ---------------------------------------------------------------------------
// Addressing helpers
// ---------------------------------------------------------------------------

/// The 64-bit address an immediate-mode access reads or writes: the base
/// cell plus the offset for the offset and pre-indexed forms, the bare base
/// for post-index. `rn == 31` is the SP cell.
fn addr_of(rn: u8, mode: AddrMode) -> Expr {
    let base = rd(cell(rn));
    match mode {
        AddrMode::PostIndex(_) => base,
        AddrMode::Offset(imm) | AddrMode::PreIndex(imm) => {
            if imm == 0 {
                base
            } else {
                bin(BinOp::Add, base, k(imm as u64, Width::W64))
            }
        }
    }
}

/// The base-writeback statements of an immediate-mode access, given the
/// temporary holding the pre-access address: none for a plain offset,
/// `rn := taddr` for pre-index, `rn := taddr + imm` for post-index.
fn writeback(rn: u8, mode: AddrMode, taddr: ir::Reg) -> Vec<Stmt> {
    match mode {
        AddrMode::Offset(_) => Vec::new(),
        AddrMode::PreIndex(_) => vec![assign(cell(rn), rd(taddr))],
        AddrMode::PostIndex(imm) => {
            let next = if imm == 0 {
                rd(taddr)
            } else {
                bin(BinOp::Add, rd(taddr), k(imm as u64, Width::W64))
            };
            vec![assign(cell(rn), next)]
        }
    }
}

/// The 64-bit address of a register-offset access:
/// `base + (extend(rm) << amount)`, with the extend derived from the raw
/// `option` field (UXTW / LSL / SXTW / SXTX) and `amount` the access size
/// when the `S` bit scales the index. `rm == 31` is the zero register, so
/// the index term vanishes. `None` for an `option` value outside the
/// allocated four (unreachable from the decoder, handled without panicking).
fn reg_offset_addr(rn: u8, off: RegOffset, size: u8) -> Option<Expr> {
    if !matches!(off.option, 0b010 | 0b011 | 0b110 | 0b111) {
        return None;
    }
    let base = rd(cell(rn));
    if off.rm == 31 {
        return Some(base);
    }
    let xm = rd(cell(off.rm));
    let index = match off.option {
        // UXTW: zero-extend the 32-bit index register.
        0b010 => un(UnOp::ZeroExtend(Width::W64), un(UnOp::Truncate(Width::W32), xm)),
        // SXTW: sign-extend the 32-bit index register.
        0b110 => un(UnOp::SignExtend(Width::W64), un(UnOp::Truncate(Width::W32), xm)),
        // LSL/UXTX and SXTX: the 64-bit index as-is.
        _ => xm,
    };
    let term = if off.scaled && size > 0 {
        bin(BinOp::Shl, index, k(size as u64, Width::W64))
    } else {
        index
    };
    Some(bin(BinOp::Add, base, term))
}

// ---------------------------------------------------------------------------
// Intrinsics
// ---------------------------------------------------------------------------

fn intr(name: &'static str, writes: Vec<ir::Reg>, reads: Vec<Expr>) -> Vec<Stmt> {
    vec![Stmt::Intrinsic {
        name,
        writes,
        reads,
    }]
}

/// The clobber-everything fallback for an unmodeled encoding: one intrinsic
/// writing all 32 GPR/SP cells, all 64 SIMD&FP half cells (unmodeled FP
/// arithmetic writes vector state), and all four NZCV flags, reading the
/// raw word so the lift is never silently empty and stays sound for
/// analysis.
fn unknown_intrinsic(raw: u32) -> Vec<Stmt> {
    let mut writes: Vec<ir::Reg> = (0..=95).map(|n| ir::Reg::arch(n, Width::W64)).collect();
    writes.extend([
        flag(Flag::Carry),
        flag(Flag::Zero),
        flag(Flag::Sign),
        flag(Flag::Overflow),
    ]);
    intr("a64.unknown", writes, vec![k(raw as u64, Width::W64)])
}

/// An exception-generating call (`svc`/`hvc`/`smc`) as a named intrinsic
/// with an honest AAPCS-ish surface: the immediate and the call-number and
/// argument registers (`x8`, `x0`–`x5`) read, the result register (`x0`)
/// written.
fn exception_call(name: &'static str, imm: u16) -> Vec<Stmt> {
    let mut reads = vec![k(imm as u64, Width::W64), rd(cell(8))];
    reads.extend((0u8..=5).map(|r| rd(cell(r))));
    intr(name, vec![cell(0)], reads)
}

/// A scalar FP operation the IR cannot express: a *precise* named
/// intrinsic writing exactly the destination's low cell — never the
/// 100-cell `a64.unknown` clobber — followed by the architectural
/// zeroing of the high cell.
fn fp_scalar_intr(name: &'static str, rdn: u8, reads: Vec<Expr>) -> Vec<Stmt> {
    vec![
        Stmt::Intrinsic {
            name,
            writes: vec![vlo(rdn)],
            reads,
        },
        assign(vhi(rdn), k(0, Width::W64)),
    ]
}

/// Advanced SIMD three-same integer ALU over the two 64-bit cells.
/// Bitwise AND/ORR/EOR and `.2d` ADD/SUB are exact; packed ADD/SUB of
/// narrower lanes is a precise named intrinsic. `Q = 0` zeroes the
/// high half architecturally.
fn simd_alu(op: aarch64::SimdAluOp, q: bool, size: u8, rdn: u8, rn: u8, rm: u8) -> Vec<Stmt> {
    use aarch64::SimdAluOp;
    let exact = match op {
        SimdAluOp::And => Some(BinOp::And),
        SimdAluOp::Orr => Some(BinOp::Or),
        SimdAluOp::Eor => Some(BinOp::Xor),
        SimdAluOp::Add if size == 3 => Some(BinOp::Add),
        SimdAluOp::Sub if size == 3 => Some(BinOp::Sub),
        _ => None,
    };
    if let Some(bop) = exact {
        let mut out = vec![assign(
            vlo(rdn),
            bin(bop, rd(vlo(rn)), rd(vlo(rm))),
        )];
        if q {
            out.push(assign(vhi(rdn), bin(bop, rd(vhi(rn)), rd(vhi(rm)))));
        } else {
            out.push(assign(vhi(rdn), k(0, Width::W64)));
        }
        return out;
    }
    let name = match op {
        SimdAluOp::Add => "a64.vadd",
        SimdAluOp::Sub => "a64.vsub",
        _ => unreachable!("logical and .2d are exact"),
    };
    let mut reads = vec![rd(vlo(rn)), rd(vlo(rm))];
    let mut writes = vec![vlo(rdn)];
    if q {
        reads.extend([rd(vhi(rn)), rd(vhi(rm))]);
        writes.push(vhi(rdn));
    }
    let mut out = vec![Stmt::Intrinsic { name, writes, reads }];
    if !q {
        out.push(assign(vhi(rdn), k(0, Width::W64)));
    }
    out
}

/// The intrinsic name of a scalar FP two-source operation.
fn f2_name(op: aarch64::F2Op) -> &'static str {
    use aarch64::F2Op;
    match op {
        F2Op::Mul => "a64.fmul",
        F2Op::Div => "a64.fdiv",
        F2Op::Add => "a64.fadd",
        F2Op::Sub => "a64.fsub",
        F2Op::Max => "a64.fmax",
        F2Op::Min => "a64.fmin",
        F2Op::MaxNm => "a64.fmaxnm",
        F2Op::MinNm => "a64.fminnm",
        F2Op::NMul => "a64.fnmul",
    }
}

/// The intrinsic name of an inexpressible scalar FP one-source
/// operation (`FABS`/`FNEG` are exact and never reach this).
fn f1_name(op: aarch64::F1Op) -> &'static str {
    use aarch64::{F1Op, FpRound};
    match op {
        F1Op::Sqrt => "a64.fsqrt",
        F1Op::Rint(FpRound::N) => "a64.frintn",
        F1Op::Rint(FpRound::P) => "a64.frintp",
        F1Op::Rint(FpRound::M) => "a64.frintm",
        F1Op::Rint(FpRound::Z) => "a64.frintz",
        F1Op::Rint(FpRound::A) => "a64.frinta",
        F1Op::RintX => "a64.frintx",
        F1Op::RintI => "a64.frinti",
        F1Op::Abs | F1Op::Neg => unreachable!("exact lifts"),
    }
}

/// The intrinsic name of an FP-to-integer conversion, by rounding
/// direction and signedness.
fn fcvt_name(round: aarch64::FpRound, unsigned: bool) -> &'static str {
    use aarch64::FpRound;
    match (round, unsigned) {
        (FpRound::N, false) => "a64.fcvtns",
        (FpRound::N, true) => "a64.fcvtnu",
        (FpRound::P, false) => "a64.fcvtps",
        (FpRound::P, true) => "a64.fcvtpu",
        (FpRound::M, false) => "a64.fcvtms",
        (FpRound::M, true) => "a64.fcvtmu",
        (FpRound::Z, false) => "a64.fcvtzs",
        (FpRound::Z, true) => "a64.fcvtzu",
        (FpRound::A, false) => "a64.fcvtas",
        (FpRound::A, true) => "a64.fcvtau",
    }
}

/// The intrinsic name of a one-source bit operation.
fn bit1_name(op: aarch64::Bit1Op) -> &'static str {
    use aarch64::Bit1Op;
    match op {
        Bit1Op::Rbit => "a64.rbit",
        Bit1Op::Rev16 => "a64.rev16",
        Bit1Op::Rev32 => "a64.rev32",
        Bit1Op::Rev => "a64.rev",
        Bit1Op::Clz => "a64.clz",
        Bit1Op::Cls => "a64.cls",
    }
}

/// The vector-element bit width named by a copy-group `size`.
fn elem_width(size: u8) -> Width {
    match size {
        0 => Width::W8,
        1 => Width::W16,
        2 => Width::W32,
        _ => Width::W64,
    }
}

/// Read element `index` of SIMD&FP register `rn` (element log2 width
/// `size`), zero-extended to a `W64` expression: the owning half cell,
/// shifted and masked.
fn velem(size: u8, index: u8, rn: u8) -> Expr {
    let bitpos = (index as u64) << (3 + size);
    let src = if bitpos >= 64 { vhi(rn) } else { vlo(rn) };
    let off = bitpos % 64;
    let mut e = rd(src);
    if off != 0 {
        e = bin(BinOp::LShr, e, k(off, Width::W64));
    }
    if size < 3 {
        e = bin(BinOp::And, e, k(ones(8u32 << size), Width::W64));
    }
    e
}

// ---------------------------------------------------------------------------
// The lifter context (VA and a temporary counter)
// ---------------------------------------------------------------------------

/// Which alternative the conditional-select family computes from `rm` when
/// the condition is false.
#[derive(Clone, Copy)]
enum SelAlt {
    /// `CSEL`: `rm` itself.
    Value,
    /// `CSINC`: `rm + 1`.
    Inc,
    /// `CSINV`: `~rm`.
    Inv,
    /// `CSNEG`: `-rm`.
    Neg,
}

struct Ctx {
    /// VA of the instruction being lifted.
    va: u64,
    /// Next free temporary number.
    temp: u16,
}

impl Ctx {
    fn fresh(&mut self, width: Width) -> ir::Reg {
        let r = ir::Reg::temp(self.temp, width);
        self.temp = self.temp.wrapping_add(1);
        r
    }

    /// The VA of the instruction that follows this one (A64 instructions
    /// are always 4 bytes).
    fn next_va(&self) -> u64 {
        self.va.wrapping_add(4)
    }

    // ---- add/subtract (immediate, shifted register, extended register) ----

    /// `ADD`/`SUB` immediate, with or without flags. `rn` (and, when not
    /// setting flags, `rd`) treat an encoded 31 as SP; the flag-setting
    /// forms treat `rd == 31` as ZR — the `CMP`/`CMN` aliases, flags only.
    fn arith_imm(
        &mut self,
        sf: bool,
        set_flags: bool,
        rdn: u8,
        rn: u8,
        imm: u32,
        sub: bool,
    ) -> Vec<Stmt> {
        let w = sf_w(sf);
        self.arith(sf, set_flags, rdn, read_sp(rn, w), k(imm as u64, w), sub, true)
    }

    /// The shared add/subtract body: `rd := lhs op rhs`, optionally with
    /// the NZCV writes. This is the *one* flag model — every flag-setting
    /// add/sub form (immediate, shifted register, extended register)
    /// funnels through here so they all write identical flag expressions.
    /// `sp_dest` selects the SP-position destination of the
    /// non-flag-setting immediate/extended forms; a flag-setting `rd`
    /// is always ZR-position (the `CMP`/`CMN` aliases).
    #[allow(clippy::too_many_arguments)]
    fn arith(
        &mut self,
        sf: bool,
        set_flags: bool,
        rdn: u8,
        lhs: Expr,
        rhs: Expr,
        sub: bool,
        sp_dest: bool,
    ) -> Vec<Stmt> {
        let w = sf_w(sf);
        let op = if sub { BinOp::Sub } else { BinOp::Add };
        if !set_flags {
            let value = bin(op, lhs, rhs);
            return if sp_dest {
                write_sp(rdn, value, w)
            } else {
                write_zr(rdn, value, w)
            };
        }
        let tl = self.fresh(w);
        let tr = self.fresh(w);
        let tres = self.fresh(w);
        let mut out = vec![
            assign(tl, lhs),
            assign(tr, rhs),
            assign(tres, bin(op, rd(tl), rd(tr))),
        ];
        for (f, e) in nzcv_model(tl, tr, tres, w, sub, None) {
            out.push(assign(flag(f), e));
        }
        out.extend(write_zr(rdn, rd(tres), w));
        out
    }

    /// `ADC{S}`/`SBC{S}`: add with carry through the one flag model.
    /// `SBC` is `rn + NOT(rm) + C` (the Arm ARM's `AddWithCarry`
    /// identity), so both route through the model's add half with
    /// `invert` complementing the second operand. The flag-setting forms
    /// snapshot the carry-in first — they overwrite CF — and take their
    /// flag writes from [`nzcv_model`]'s carry extension.
    fn adc(
        &mut self,
        sf: bool,
        set_flags: bool,
        rdn: u8,
        rn: u8,
        rm: u8,
        invert: bool,
    ) -> Vec<Stmt> {
        let w = sf_w(sf);
        let operand = read_zr(rm, w);
        let rhs = if invert { un(UnOp::Not, operand) } else { operand };
        if !set_flags {
            let value = bin(
                BinOp::Add,
                bin(BinOp::Add, read_zr(rn, w), rhs),
                un(UnOp::ZeroExtend(w), rd(flag(Flag::Carry))),
            );
            return write_zr(rdn, value, w);
        }
        let tc = self.fresh(Width::W1);
        let tl = self.fresh(w);
        let tr = self.fresh(w);
        let tres = self.fresh(w);
        let mut out = vec![
            assign(tc, rd(flag(Flag::Carry))),
            assign(tl, read_zr(rn, w)),
            assign(tr, rhs),
            assign(
                tres,
                bin(
                    BinOp::Add,
                    bin(BinOp::Add, rd(tl), rd(tr)),
                    un(UnOp::ZeroExtend(w), rd(tc)),
                ),
            ),
        ];
        for (f, e) in nzcv_model(tl, tr, tres, w, false, Some(tc)) {
            out.push(assign(flag(f), e));
        }
        out.extend(write_zr(rdn, rd(tres), w));
        out
    }

    /// `CCMP`/`CCMN`: if the condition holds, NZCV gets the flags of the
    /// compare (`rn - op2` for `CCMP`, `rn + op2` for `CCMN`); otherwise
    /// the literal `nzcv` immediate. Lifted branchless: the condition
    /// and the compare's operands and result run unconditionally into
    /// temporaries, and each flag cell selects between the one model's
    /// expression and its imm4 bit —
    /// `flag := (c & model) | (~c & bit)` — so the compare's flag
    /// expressions are textually [`nzcv_model`]'s products.
    fn ccmp(&mut self, sf: bool, sub: bool, rn: u8, op2: Expr, nzcv: u8, cond: Cond) -> Vec<Stmt> {
        let w = sf_w(sf);
        let tc = self.fresh(Width::W1);
        let tl = self.fresh(w);
        let tr = self.fresh(w);
        let tres = self.fresh(w);
        let op = if sub { BinOp::Sub } else { BinOp::Add };
        let mut out = vec![
            assign(tc, cond_expr(cond)),
            assign(tl, read_zr(rn, w)),
            assign(tr, op2),
            assign(tres, bin(op, rd(tl), rd(tr))),
        ];
        for (f, e) in nzcv_model(tl, tr, tres, w, sub, None) {
            // The imm4 bit that replaces this flag when the condition
            // fails: N:Z:C:V from bit 3 down.
            let bit = match f {
                Flag::Sign => nzcv >> 3,
                Flag::Zero => nzcv >> 2,
                Flag::Carry => nzcv >> 1,
                _ => nzcv,
            } & 1;
            let sel = bin(
                BinOp::Or,
                bin(BinOp::And, rd(tc), e),
                bin(BinOp::And, not1(rd(tc)), k(bit as u64, Width::W1)),
            );
            out.push(assign(flag(f), sel));
        }
        out
    }

    // ---- logical (shifted register / immediate) ----

    /// The shared logical body: `rd := rn op operand` (the operand
    /// complemented first for `BIC`/`ORN`/`EON`/`BICS`). The flag-setting
    /// forms (`ANDS`/`BICS`, and `TST` as their rd = zr alias) write N and
    /// Z from the result and architecturally *clear* C and V. `sp_dest`
    /// selects the SP-position destination of the non-flag-setting
    /// immediate forms; every register-form destination is ZR-position.
    #[allow(clippy::too_many_arguments)]
    fn logical(
        &mut self,
        sf: bool,
        op: LogOp,
        set_flags: bool,
        invert: bool,
        rdn: u8,
        rn: u8,
        operand: Expr,
        sp_dest: bool,
    ) -> Vec<Stmt> {
        let w = sf_w(sf);
        let rhs = if invert { un(UnOp::Not, operand) } else { operand };
        let bop = match op {
            LogOp::And => BinOp::And,
            LogOp::Orr => BinOp::Or,
            LogOp::Eor => BinOp::Xor,
        };
        let value = bin(bop, read_zr(rn, w), rhs);
        if !set_flags {
            return if sp_dest {
                write_sp(rdn, value, w)
            } else {
                write_zr(rdn, value, w)
            };
        }
        let tres = self.fresh(w);
        let zero = k(0, w);
        let mut out = vec![
            assign(tres, value),
            assign(flag(Flag::Zero), bin(BinOp::Eq, rd(tres), zero.clone())),
            assign(flag(Flag::Sign), bin(BinOp::Slt, rd(tres), zero)),
            assign(flag(Flag::Carry), k(0, Width::W1)),
            assign(flag(Flag::Overflow), k(0, Width::W1)),
        ];
        out.extend(write_zr(rdn, rd(tres), w));
        out
    }

    // ---- bitfield moves ----

    /// `UBFM`: the `imms >= immr` half extracts (`UBFX`/`LSR` — shift
    /// right, mask), the other half inserts at zero (`UBFIZ`/`LSL` —
    /// mask, shift left). Both shift amounts stay below the width, which
    /// the decoder's 32-bit `immr`/`imms` bound guarantees.
    fn ubfm(&mut self, sf: bool, rdn: u8, rn: u8, immr: u8, imms: u8) -> Vec<Stmt> {
        let w = sf_w(sf);
        let bits = w.bits();
        let src = read_zr(rn, w);
        let value = if imms >= immr {
            let mask = ones((imms - immr + 1) as u32);
            let shifted = if immr == 0 {
                src
            } else {
                bin(BinOp::LShr, src, k(immr as u64, w))
            };
            bin(BinOp::And, shifted, k(mask, w))
        } else {
            // imms < immr forces immr >= 1, so the left shift is < bits.
            let masked = bin(BinOp::And, src, k(ones(imms as u32 + 1), w));
            bin(BinOp::Shl, masked, k((bits - immr as u32) as u64, w))
        };
        write_zr(rdn, value, w)
    }

    /// `SBFM`: position the field's top bit at the register's top with a
    /// left shift, sign-fill down with an arithmetic right shift
    /// (`SBFX`/`ASR`/`SXT*`), and for the insert-at-lsb half (`SBFIZ`)
    /// shift the sign-extended field back up.
    fn sbfm(&mut self, sf: bool, rdn: u8, rn: u8, immr: u8, imms: u8) -> Vec<Stmt> {
        let w = sf_w(sf);
        let bits = w.bits();
        let src = read_zr(rn, w);
        let up = bits - 1 - imms as u32;
        let raised = if up == 0 {
            src
        } else {
            bin(BinOp::Shl, src, k(up as u64, w))
        };
        let value = if imms >= immr {
            let down = up + immr as u32;
            if down == 0 {
                raised
            } else {
                bin(BinOp::AShr, raised, k(down as u64, w))
            }
        } else {
            let field = if up == 0 {
                raised
            } else {
                bin(BinOp::AShr, raised, k(up as u64, w))
            };
            bin(BinOp::Shl, field, k((bits - immr as u32) as u64, w))
        };
        write_zr(rdn, value, w)
    }

    /// `BFM`: a read-modify-write merge — `BFXIL` replaces the low bits of
    /// `rd` with a field extracted from `rn`, `BFI`/`BFC` punches a hole
    /// at the insertion point and fills it from `rn`'s low bits.
    fn bfm(&mut self, sf: bool, rdn: u8, rn: u8, immr: u8, imms: u8) -> Vec<Stmt> {
        let w = sf_w(sf);
        let bits = w.bits();
        let dst = read_zr(rdn, w);
        let src = read_zr(rn, w);
        let value = if imms >= immr {
            let width = (imms - immr + 1) as u32;
            let shifted = if immr == 0 {
                src
            } else {
                bin(BinOp::LShr, src, k(immr as u64, w))
            };
            let field = bin(BinOp::And, shifted, k(ones(width), w));
            bin(BinOp::Or, bin(BinOp::And, dst, k(!ones(width), w)), field)
        } else {
            let width = imms as u32 + 1;
            let lsb = bits - immr as u32;
            let field = bin(
                BinOp::Shl,
                bin(BinOp::And, src, k(ones(width), w)),
                k(lsb as u64, w),
            );
            let hole = !(ones(width) << lsb);
            bin(BinOp::Or, bin(BinOp::And, dst, k(hole, w)), field)
        };
        write_zr(rdn, value, w)
    }

    // ---- two-source: variable shifts and divides ----

    /// `LSLV`/`LSRV`/`ASRV`/`RORV`: shift `rn` by `rm` modulo the width.
    /// The amount is snapshotted (masked) in a temporary; the rotate is
    /// `(x >>u s) | (x << (-s & (bits-1)))`, whose second shift is 0 when
    /// `s` is 0 — never a full-width shift.
    fn shift_var(&mut self, sf: bool, kind: Shift, rdn: u8, rn: u8, rm: u8) -> Vec<Stmt> {
        let w = sf_w(sf);
        let mask = (w.bits() - 1) as u64;
        let ts = self.fresh(w);
        let mut out = vec![assign(ts, bin(BinOp::And, read_zr(rm, w), k(mask, w)))];
        let x = read_zr(rn, w);
        let value = match kind {
            Shift::Lsl => bin(BinOp::Shl, x, rd(ts)),
            Shift::Lsr => bin(BinOp::LShr, x, rd(ts)),
            Shift::Asr => bin(BinOp::AShr, x, rd(ts)),
            Shift::Ror => bin(
                BinOp::Or,
                bin(BinOp::LShr, x.clone(), rd(ts)),
                bin(
                    BinOp::Shl,
                    x,
                    bin(BinOp::And, un(UnOp::Neg, rd(ts)), k(mask, w)),
                ),
            ),
        };
        out.extend(write_zr(rdn, value, w));
        out
    }

    /// `UDIV`/`SDIV` with the architectural zero-divisor rule: the result
    /// of dividing by zero is zero, not a trap. The IR treats division by
    /// zero as a lift-defined trap, so the guard is structural — the
    /// divisor is forced to 1 when `rm == 0` (no evaluator ever divides
    /// by zero) and the quotient is masked to 0 on that same condition.
    /// The remaining `SDIV` corner, `INT_MIN / -1`, wraps in both the
    /// architecture and the IR's folding, so it needs no guard.
    fn div(&mut self, sf: bool, signed: bool, rdn: u8, rn: u8, rm: u8) -> Vec<Stmt> {
        let w = sf_w(sf);
        let op = if signed { BinOp::SDiv } else { BinOp::UDiv };
        let tz = self.fresh(Width::W1);
        let tq = self.fresh(w);
        let divisor = bin(BinOp::Or, read_zr(rm, w), un(UnOp::ZeroExtend(w), rd(tz)));
        let mut out = vec![
            assign(tz, bin(BinOp::Eq, read_zr(rm, w), k(0, w))),
            assign(tq, bin(op, read_zr(rn, w), divisor)),
        ];
        let value = bin(
            BinOp::And,
            rd(tq),
            un(UnOp::Not, un(UnOp::SignExtend(w), rd(tz))),
        );
        out.extend(write_zr(rdn, value, w));
        out
    }

    // ---- conditional select ----

    /// The `CSEL` family, lifted branchless like x86 `cmov`: a sign-extended
    /// condition mask `m` merges `rn` (condition true) with the alternative
    /// computed from `rm` (condition false):
    /// `rd := (rn & m) | (alt & ~m)`.
    fn csel(&mut self, sf: bool, rdn: u8, rn: u8, rm: u8, cond: Cond, alt: SelAlt) -> Vec<Stmt> {
        let w = sf_w(sf);
        let tm = self.fresh(w);
        let talt = self.fresh(w);
        let alt_value = match alt {
            SelAlt::Value => read_zr(rm, w),
            SelAlt::Inc => bin(BinOp::Add, read_zr(rm, w), k(1, w)),
            SelAlt::Inv => un(UnOp::Not, read_zr(rm, w)),
            SelAlt::Neg => un(UnOp::Neg, read_zr(rm, w)),
        };
        let mut out = vec![
            assign(tm, un(UnOp::SignExtend(w), cond_expr(cond))),
            assign(talt, alt_value),
        ];
        let merged = bin(
            BinOp::Or,
            bin(BinOp::And, read_zr(rn, w), rd(tm)),
            bin(BinOp::And, rd(talt), un(UnOp::Not, rd(tm))),
        );
        out.extend(write_zr(rdn, merged, w));
        out
    }

    /// `FCSEL`: the branchless conditional-select merge of [`Ctx::csel`]
    /// over the two low cells, written through the scalar rule (high
    /// cell zeroed) — exact, no FP semantics involved.
    fn fcsel(&mut self, double: bool, rdn: u8, rn: u8, rm: u8, cond: Cond) -> Vec<Stmt> {
        let w = if double { Width::W64 } else { Width::W32 };
        let tc = self.fresh(w);
        let mut out = vec![assign(tc, un(UnOp::SignExtend(w), cond_expr(cond)))];
        let merged = bin(
            BinOp::Or,
            bin(BinOp::And, vread(rn, w), rd(tc)),
            bin(BinOp::And, vread(rm, w), un(UnOp::Not, rd(tc))),
        );
        out.extend(vwrite_scalar(rdn, merged, w));
        out
    }

    /// Replicate the `W64`-typed element expression `elem` (already
    /// masked to its low `8 << size` bits) across a 64-bit lane by
    /// doubling, and write the destination whole: the low cell always,
    /// the high cell a copy (`q`) or zero.
    fn dup_replicate(&mut self, rdn: u8, elem: Expr, size: u8, q: bool) -> Vec<Stmt> {
        let t = self.fresh(Width::W64);
        let mut out = vec![assign(t, elem)];
        let mut width = 8u64 << size;
        while width < 64 {
            out.push(assign(
                t,
                bin(
                    BinOp::Or,
                    rd(t),
                    bin(BinOp::Shl, rd(t), k(width, Width::W64)),
                ),
            ));
            width *= 2;
        }
        out.push(assign(vlo(rdn), rd(t)));
        out.push(assign(
            vhi(rdn),
            if q { rd(t) } else { k(0, Width::W64) },
        ));
        out
    }

    /// `INS` (both forms): read-modify-write of the one half cell that
    /// holds element `index` — the only SIMD&FP write besides
    /// `FMOV Vd.D[1]` that leaves the rest of the register intact.
    /// `elem` is the new value at `W64`, masked to its low bits.
    fn ins_elem(&mut self, rdn: u8, size: u8, index: u8, elem: Expr) -> Vec<Stmt> {
        let bitpos = (index as u64) << (3 + size);
        let dst = if bitpos >= 64 { vhi(rdn) } else { vlo(rdn) };
        let off = bitpos % 64;
        if size == 3 {
            return vec![assign(dst, elem)];
        }
        let t = self.fresh(Width::W64);
        let mask = ones(8u32 << size) << off;
        let mut shifted = rd(t);
        if off != 0 {
            shifted = bin(BinOp::Shl, shifted, k(off, Width::W64));
        }
        vec![
            assign(t, elem),
            assign(
                dst,
                bin(
                    BinOp::Or,
                    bin(BinOp::And, rd(dst), k(!mask, Width::W64)),
                    shifted,
                ),
            ),
        ]
    }

    /// `LDPSW`: the [`Ctx::ldp`] two-load body at word width, each half
    /// sign-extended into its X register.
    fn ldpsw(&mut self, rt: u8, rt2: u8, rn: u8, mode: AddrMode) -> Vec<Stmt> {
        let taddr = self.fresh(Width::W64);
        let t1 = self.fresh(Width::W32);
        let t2 = self.fresh(Width::W32);
        let mut out = vec![
            assign(taddr, addr_of(rn, mode)),
            assign(t1, Expr::load(rd(taddr), Width::W32)),
            assign(
                t2,
                Expr::load(bin(BinOp::Add, rd(taddr), k(4, Width::W64)), Width::W32),
            ),
        ];
        out.extend(writeback(rn, mode, taddr));
        out.extend(write_zr(rt, un(UnOp::SignExtend(Width::W64), rd(t1)), Width::W64));
        out.extend(write_zr(rt2, un(UnOp::SignExtend(Width::W64), rd(t2)), Width::W64));
        out
    }

    /// `STXR`/`STLXR`: the store itself (over-approximated as taken —
    /// source-level retry loops read naturally, and no consumer deletes
    /// a store), then the status write as a named intrinsic — success
    /// is unknowable statically, so `ws` gets a fresh opaque definition
    /// reading the address it hinges on.
    fn stxr(&mut self, size: u8, rt: u8, ws: u8, rn: u8) -> Vec<Stmt> {
        let lw = size_w(size);
        let taddr = self.fresh(Width::W64);
        let mut out = vec![
            assign(taddr, addr_of(rn, AddrMode::Offset(0))),
            Stmt::Store {
                addr: rd(taddr),
                value: read_zr(rt, lw),
            },
        ];
        if ws != 31 {
            out.extend(intr("a64.stxr", vec![cell(ws)], vec![rd(taddr)]));
        }
        out
    }

    // ---- loads and stores ----

    /// The shared body of every load: snapshot the address, load into a
    /// fresh value temporary, apply base writeback, then hand the loaded
    /// value (still at the access width) to `finish` for the destination
    /// write.
    fn load_common(
        &mut self,
        lw: Width,
        addr: Expr,
        wb: Option<(u8, AddrMode)>,
        finish: impl FnOnce(Expr) -> Vec<Stmt>,
    ) -> Vec<Stmt> {
        let taddr = self.fresh(Width::W64);
        let tval = self.fresh(lw);
        let mut out = vec![
            assign(taddr, addr),
            assign(tval, Expr::load(rd(taddr), lw)),
        ];
        if let Some((rn, mode)) = wb {
            out.extend(writeback(rn, mode, taddr));
        }
        out.extend(finish(rd(tval)));
        out
    }

    /// The shared body of every store — GPR and SIMD&FP alike: snapshot
    /// the address, store `value` (its width is the access size), then
    /// apply base writeback.
    fn store_common(&mut self, addr: Expr, wb: Option<(u8, AddrMode)>, value: Expr) -> Vec<Stmt> {
        let taddr = self.fresh(Width::W64);
        let mut out = vec![assign(taddr, addr)];
        out.push(Stmt::Store {
            addr: rd(taddr),
            value,
        });
        if let Some((rn, mode)) = wb {
            out.extend(writeback(rn, mode, taddr));
        }
        out
    }

    /// Zero-extending load (`LDR`/`LDRB`/`LDRH`), immediate addressing.
    fn ldr_imm(&mut self, size: u8, rt: u8, rn: u8, mode: AddrMode) -> Vec<Stmt> {
        let lw = size_w(size);
        self.load_common(lw, addr_of(rn, mode), Some((rn, mode)), |v| {
            write_zr(rt, v, lw)
        })
    }

    /// Store (`STR`/`STRB`/`STRH`), immediate addressing. An encoded 31
    /// stores zero.
    fn str_imm(&mut self, size: u8, rt: u8, rn: u8, mode: AddrMode) -> Vec<Stmt> {
        let value = read_zr(rt, size_w(size));
        self.store_common(addr_of(rn, mode), Some((rn, mode)), value)
    }

    /// Sign-extending load (`LDRSB`/`LDRSH`/`LDRSW`), immediate addressing.
    fn ldrs_imm(&mut self, size: u8, sf: bool, rt: u8, rn: u8, mode: AddrMode) -> Vec<Stmt> {
        let lw = size_w(size);
        let tw = sf_w(sf);
        self.load_common(lw, addr_of(rn, mode), Some((rn, mode)), |v| {
            write_zr(rt, extend_signed(v, lw, tw), tw)
        })
    }

    /// Zero-extending load, register-offset addressing.
    fn ldr_reg(&mut self, size: u8, rt: u8, rn: u8, off: RegOffset, raw: u32) -> Vec<Stmt> {
        let lw = size_w(size);
        match reg_offset_addr(rn, off, size) {
            Some(addr) => self.load_common(lw, addr, None, |v| write_zr(rt, v, lw)),
            None => unknown_intrinsic(raw),
        }
    }

    /// Store, register-offset addressing.
    fn str_reg(&mut self, size: u8, rt: u8, rn: u8, off: RegOffset, raw: u32) -> Vec<Stmt> {
        match reg_offset_addr(rn, off, size) {
            Some(addr) => {
                let value = read_zr(rt, size_w(size));
                self.store_common(addr, None, value)
            }
            None => unknown_intrinsic(raw),
        }
    }

    /// Sign-extending load, register-offset addressing.
    fn ldrs_reg(
        &mut self,
        size: u8,
        sf: bool,
        rt: u8,
        rn: u8,
        off: RegOffset,
        raw: u32,
    ) -> Vec<Stmt> {
        let lw = size_w(size);
        let tw = sf_w(sf);
        match reg_offset_addr(rn, off, size) {
            Some(addr) => self.load_common(lw, addr, None, |v| {
                write_zr(rt, extend_signed(v, lw, tw), tw)
            }),
            None => unknown_intrinsic(raw),
        }
    }

    /// `LDR` literal: the address is an absolute constant, so no snapshot
    /// temporary is needed — load straight into a value temp, then write.
    fn ldr_lit(&mut self, sf: bool, rt: u8, target: u64) -> Vec<Stmt> {
        let lw = sf_w(sf);
        let tval = self.fresh(lw);
        let mut out = vec![assign(tval, Expr::load(k(target, Width::W64), lw))];
        out.extend(write_zr(rt, rd(tval), lw));
        out
    }

    /// `LDP`: two loads at `taddr` and `taddr + step`, writeback, then both
    /// destination writes — so `ldp x0, x1, [x0]` and the epilogue
    /// `ldp x29, x30, [sp], #16` are deterministic.
    fn ldp(&mut self, sf: bool, rt: u8, rt2: u8, rn: u8, mode: AddrMode) -> Vec<Stmt> {
        let w = sf_w(sf);
        let step = w.bytes() as u64;
        let taddr = self.fresh(Width::W64);
        let t1 = self.fresh(w);
        let t2 = self.fresh(w);
        let mut out = vec![
            assign(taddr, addr_of(rn, mode)),
            assign(t1, Expr::load(rd(taddr), w)),
            assign(
                t2,
                Expr::load(bin(BinOp::Add, rd(taddr), k(step, Width::W64)), w),
            ),
        ];
        out.extend(writeback(rn, mode, taddr));
        out.extend(write_zr(rt, rd(t1), w));
        out.extend(write_zr(rt2, rd(t2), w));
        out
    }

    /// `STP`: two stores at `taddr` and `taddr + step`, then writeback.
    fn stp(&mut self, sf: bool, rt: u8, rt2: u8, rn: u8, mode: AddrMode) -> Vec<Stmt> {
        let w = sf_w(sf);
        let step = w.bytes() as u64;
        let taddr = self.fresh(Width::W64);
        let mut out = vec![
            assign(taddr, addr_of(rn, mode)),
            Stmt::Store {
                addr: rd(taddr),
                value: read_zr(rt, w),
            },
            Stmt::Store {
                addr: bin(BinOp::Add, rd(taddr), k(step, Width::W64)),
                value: read_zr(rt2, w),
            },
        ];
        out.extend(writeback(rn, mode, taddr));
        out
    }

    // ---- SIMD&FP loads and stores ----

    /// SIMD&FP load (`LDR b/h/s/d/q`), any immediate mode: the b–d sizes
    /// go through the shared scalar body ([`vwrite_scalar`] zeroes the
    /// high cell); the q form loads the two half cells from `taddr` and
    /// `taddr + 8` (low half at the lower address — little-endian).
    fn fldr(&mut self, size: u8, rt: u8, rn: u8, mode: AddrMode) -> Vec<Stmt> {
        let addr = addr_of(rn, mode);
        let wb = Some((rn, mode));
        if size < 4 {
            let lw = size_w(size);
            return self.load_common(lw, addr, wb, |v| vwrite_scalar(rt, v, lw));
        }
        self.fldr_q(rt, addr, wb)
    }

    /// The q-register load body shared by every addressing mode.
    fn fldr_q(&mut self, rt: u8, addr: Expr, wb: Option<(u8, AddrMode)>) -> Vec<Stmt> {
        let taddr = self.fresh(Width::W64);
        let tlo = self.fresh(Width::W64);
        let thi = self.fresh(Width::W64);
        let mut out = vec![
            assign(taddr, addr),
            assign(tlo, Expr::load(rd(taddr), Width::W64)),
            assign(
                thi,
                Expr::load(bin(BinOp::Add, rd(taddr), k(8, Width::W64)), Width::W64),
            ),
        ];
        if let Some((rn, mode)) = wb {
            out.extend(writeback(rn, mode, taddr));
        }
        out.push(assign(vlo(rt), rd(tlo)));
        out.push(assign(vhi(rt), rd(thi)));
        out
    }

    /// SIMD&FP store (`STR b/h/s/d/q`), any immediate mode.
    fn fstr(&mut self, size: u8, rt: u8, rn: u8, mode: AddrMode) -> Vec<Stmt> {
        let addr = addr_of(rn, mode);
        let wb = Some((rn, mode));
        if size < 4 {
            let value = vread(rt, size_w(size));
            return self.store_common(addr, wb, value);
        }
        self.fstr_q(rt, addr, wb)
    }

    /// The q-register store body shared by every addressing mode.
    fn fstr_q(&mut self, rt: u8, addr: Expr, wb: Option<(u8, AddrMode)>) -> Vec<Stmt> {
        let taddr = self.fresh(Width::W64);
        let mut out = vec![
            assign(taddr, addr),
            Stmt::Store {
                addr: rd(taddr),
                value: rd(vlo(rt)),
            },
            Stmt::Store {
                addr: bin(BinOp::Add, rd(taddr), k(8, Width::W64)),
                value: rd(vhi(rt)),
            },
        ];
        if let Some((rn, mode)) = wb {
            out.extend(writeback(rn, mode, taddr));
        }
        out
    }

    /// SIMD&FP load, register-offset addressing (no writeback).
    fn fldr_reg(&mut self, size: u8, rt: u8, rn: u8, off: RegOffset, raw: u32) -> Vec<Stmt> {
        match reg_offset_addr(rn, off, size) {
            Some(addr) if size < 4 => {
                let lw = size_w(size);
                self.load_common(lw, addr, None, |v| vwrite_scalar(rt, v, lw))
            }
            Some(addr) => self.fldr_q(rt, addr, None),
            None => unknown_intrinsic(raw),
        }
    }

    /// SIMD&FP store, register-offset addressing.
    fn fstr_reg(&mut self, size: u8, rt: u8, rn: u8, off: RegOffset, raw: u32) -> Vec<Stmt> {
        match reg_offset_addr(rn, off, size) {
            Some(addr) if size < 4 => {
                let value = vread(rt, size_w(size));
                self.store_common(addr, None, value)
            }
            Some(addr) => self.fstr_q(rt, addr, None),
            None => unknown_intrinsic(raw),
        }
    }

    /// SIMD&FP `LDR` literal: an absolute constant address (and, for q,
    /// its `+ 8` neighbor — also constant-folded).
    fn fldr_lit(&mut self, size: u8, rt: u8, target: u64) -> Vec<Stmt> {
        if size < 4 {
            let lw = size_w(size);
            let tval = self.fresh(lw);
            let mut out = vec![assign(tval, Expr::load(k(target, Width::W64), lw))];
            out.extend(vwrite_scalar(rt, rd(tval), lw));
            return out;
        }
        let tlo = self.fresh(Width::W64);
        let thi = self.fresh(Width::W64);
        vec![
            assign(tlo, Expr::load(k(target, Width::W64), Width::W64)),
            assign(
                thi,
                Expr::load(k(target.wrapping_add(8), Width::W64), Width::W64),
            ),
            assign(vlo(rt), rd(tlo)),
            assign(vhi(rt), rd(thi)),
        ]
    }

    /// SIMD&FP `LDP` (s/d/q): loads first, then writeback, then the
    /// destination writes — the same fixed order as the integer pair.
    fn fldp(&mut self, size: u8, rt: u8, rt2: u8, rn: u8, mode: AddrMode) -> Vec<Stmt> {
        let step = 1u64 << size;
        let taddr = self.fresh(Width::W64);
        let mut out = vec![assign(taddr, addr_of(rn, mode))];
        if size < 4 {
            let lw = size_w(size);
            let t1 = self.fresh(lw);
            let t2 = self.fresh(lw);
            out.push(assign(t1, Expr::load(rd(taddr), lw)));
            out.push(assign(
                t2,
                Expr::load(bin(BinOp::Add, rd(taddr), k(step, Width::W64)), lw),
            ));
            out.extend(writeback(rn, mode, taddr));
            out.extend(vwrite_scalar(rt, rd(t1), lw));
            out.extend(vwrite_scalar(rt2, rd(t2), lw));
            return out;
        }
        // q: four 64-bit loads at taddr, +8, +16, +24.
        let temps: Vec<ir::Reg> = (0..4).map(|_| self.fresh(Width::W64)).collect();
        for (i, &t) in temps.iter().enumerate() {
            let addr = if i == 0 {
                rd(taddr)
            } else {
                bin(BinOp::Add, rd(taddr), k(8 * i as u64, Width::W64))
            };
            out.push(assign(t, Expr::load(addr, Width::W64)));
        }
        out.extend(writeback(rn, mode, taddr));
        out.push(assign(vlo(rt), rd(temps[0])));
        out.push(assign(vhi(rt), rd(temps[1])));
        out.push(assign(vlo(rt2), rd(temps[2])));
        out.push(assign(vhi(rt2), rd(temps[3])));
        out
    }

    /// SIMD&FP `STP` (s/d/q): stores, then writeback.
    fn fstp(&mut self, size: u8, rt: u8, rt2: u8, rn: u8, mode: AddrMode) -> Vec<Stmt> {
        let step = 1u64 << size;
        let taddr = self.fresh(Width::W64);
        let mut out = vec![assign(taddr, addr_of(rn, mode))];
        let at = |i: u64| {
            if i == 0 {
                rd(taddr)
            } else {
                bin(BinOp::Add, rd(taddr), k(i, Width::W64))
            }
        };
        if size < 4 {
            let sw = size_w(size);
            out.push(Stmt::Store {
                addr: at(0),
                value: vread(rt, sw),
            });
            out.push(Stmt::Store {
                addr: at(step),
                value: vread(rt2, sw),
            });
        } else {
            for (i, value) in [vlo(rt), vhi(rt), vlo(rt2), vhi(rt2)].into_iter().enumerate() {
                out.push(Stmt::Store {
                    addr: at(8 * i as u64),
                    value: rd(value),
                });
            }
        }
        out.extend(writeback(rn, mode, taddr));
        out
    }

    // ---- branches ----

    /// A compare-and-branch guard (`CBZ`/`CBNZ`).
    fn cb(&self, sf: bool, rt: u8, target: u64, op: BinOp) -> Vec<Stmt> {
        let w = sf_w(sf);
        vec![Stmt::Branch {
            kind: ir::BranchKind::Jump,
            cond: Some(bin(op, read_zr(rt, w), k(0, w))),
            target: k(target, Width::W64),
        }]
    }

    /// A test-bit-and-branch guard (`TBZ`/`TBNZ`), computed at `W64`.
    fn tb(&self, rt: u8, bit: u8, target: u64, op: BinOp) -> Vec<Stmt> {
        let shifted = bin(BinOp::LShr, read_zr(rt, Width::W64), k(bit as u64, Width::W64));
        let picked = bin(BinOp::And, shifted, k(1, Width::W64));
        vec![Stmt::Branch {
            kind: ir::BranchKind::Jump,
            cond: Some(bin(op, picked, k(0, Width::W64))),
            target: k(target, Width::W64),
        }]
    }
}

/// A mask of the low `width` bits as a `u64` (total for `width >= 64`).
fn ones(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// The shifted-register operand `shift(rm, #amount)` at width `w`
/// (`rm == 31` is the zero register). A rotate — allocated only in the
/// logical group — is composed from the two logical shifts; amount 0 is
/// the operand itself for every kind.
fn shifted_reg(rm: u8, shift: Shift, amount: u8, w: Width) -> Expr {
    let x = read_zr(rm, w);
    if amount == 0 {
        return x;
    }
    let amt = k(amount as u64, w);
    match shift {
        Shift::Lsl => bin(BinOp::Shl, x, amt),
        Shift::Lsr => bin(BinOp::LShr, x, amt),
        Shift::Asr => bin(BinOp::AShr, x, amt),
        Shift::Ror => bin(
            BinOp::Or,
            bin(BinOp::LShr, x.clone(), amt),
            bin(BinOp::Shl, x, k(w.bits() as u64 - amount as u64, w)),
        ),
    }
}

/// The extended-register operand `extend(rm) << amount` at width `w`:
/// truncate the (ZR-position) index to the extend width, re-extend to
/// `w`, then apply the left shift. An extend at or above `w` (`UXTX`,
/// and the W-sized extends of the 32-bit form) is the register itself.
fn extended_reg(rm: u8, option: u8, amount: u8, w: Width) -> Expr {
    let ext_bits = 8u32 << (option & 0b011);
    let x = if ext_bits >= w.bits() {
        read_zr(rm, w)
    } else {
        let ew = match ext_bits {
            8 => Width::W8,
            16 => Width::W16,
            _ => Width::W32,
        };
        let narrow = un(UnOp::Truncate(ew), read_zr(rm, Width::W64));
        if option & 0b100 != 0 {
            un(UnOp::SignExtend(w), narrow)
        } else {
            un(UnOp::ZeroExtend(w), narrow)
        }
    };
    if amount == 0 {
        x
    } else {
        bin(BinOp::Shl, x, k(amount as u64, w))
    }
}

/// Widen (or defensively narrow) a loaded value from the access width to
/// the destination register width by sign extension. The decoder only
/// produces widening pairs; the equal and narrowing cases keep the lift
/// total without panicking.
fn extend_signed(e: Expr, from: Width, to: Width) -> Expr {
    if from.bits() < to.bits() {
        un(UnOp::SignExtend(to), e)
    } else if from.bits() > to.bits() {
        un(UnOp::Truncate(to), e)
    } else {
        e
    }
}

// ---------------------------------------------------------------------------
// The top-level dispatch
// ---------------------------------------------------------------------------

fn lift_with(insn: &aarch64::Instruction, va: u64, temp: &mut u16) -> Vec<Stmt> {
    let mut ctx = Ctx { va, temp: *temp };
    let out = match insn.opcode {
        // ---- data processing, immediate ----
        Opcode::Adr { rd: rdn, target } | Opcode::Adrp { rd: rdn, target } => {
            write_zr(rdn, k(target, Width::W64), Width::W64)
        }
        Opcode::AddImm {
            sf,
            set_flags,
            rd: rdn,
            rn,
            imm,
        } => ctx.arith_imm(sf, set_flags, rdn, rn, imm, false),
        Opcode::SubImm {
            sf,
            set_flags,
            rd: rdn,
            rn,
            imm,
        } => ctx.arith_imm(sf, set_flags, rdn, rn, imm, true),
        // ---- data processing, register ----
        Opcode::AddReg {
            sf,
            set_flags,
            rd: rdn,
            rn,
            rm,
            shift,
            amount,
        } => {
            let w = sf_w(sf);
            let rhs = shifted_reg(rm, shift, amount, w);
            ctx.arith(sf, set_flags, rdn, read_zr(rn, w), rhs, false, false)
        }
        Opcode::SubReg {
            sf,
            set_flags,
            rd: rdn,
            rn,
            rm,
            shift,
            amount,
        } => {
            let w = sf_w(sf);
            let rhs = shifted_reg(rm, shift, amount, w);
            ctx.arith(sf, set_flags, rdn, read_zr(rn, w), rhs, true, false)
        }
        Opcode::AddExt {
            sf,
            set_flags,
            rd: rdn,
            rn,
            rm,
            option,
            amount,
        } => {
            let w = sf_w(sf);
            let rhs = extended_reg(rm, option, amount, w);
            ctx.arith(sf, set_flags, rdn, read_sp(rn, w), rhs, false, true)
        }
        Opcode::SubExt {
            sf,
            set_flags,
            rd: rdn,
            rn,
            rm,
            option,
            amount,
        } => {
            let w = sf_w(sf);
            let rhs = extended_reg(rm, option, amount, w);
            ctx.arith(sf, set_flags, rdn, read_sp(rn, w), rhs, true, true)
        }
        Opcode::Adc {
            sf,
            set_flags,
            rd: rdn,
            rn,
            rm,
        } => ctx.adc(sf, set_flags, rdn, rn, rm, false),
        Opcode::Sbc {
            sf,
            set_flags,
            rd: rdn,
            rn,
            rm,
        } => ctx.adc(sf, set_flags, rdn, rn, rm, true),
        Opcode::CcmpReg {
            sf,
            sub,
            rn,
            rm,
            nzcv,
            cond,
        } => {
            let op2 = read_zr(rm, sf_w(sf));
            ctx.ccmp(sf, sub, rn, op2, nzcv, cond)
        }
        Opcode::CcmpImm {
            sf,
            sub,
            rn,
            imm,
            nzcv,
            cond,
        } => ctx.ccmp(sf, sub, rn, k(imm as u64, sf_w(sf)), nzcv, cond),
        Opcode::LogReg {
            sf,
            op,
            set_flags,
            invert,
            rd: rdn,
            rn,
            rm,
            shift,
            amount,
        } => {
            let operand = shifted_reg(rm, shift, amount, sf_w(sf));
            ctx.logical(sf, op, set_flags, invert, rdn, rn, operand, false)
        }
        Opcode::LogImm {
            sf,
            op,
            set_flags,
            rd: rdn,
            rn,
            imm,
        } => {
            let operand = k(imm, sf_w(sf));
            ctx.logical(sf, op, set_flags, false, rdn, rn, operand, true)
        }
        Opcode::Sbfm {
            sf,
            rd: rdn,
            rn,
            immr,
            imms,
        } => ctx.sbfm(sf, rdn, rn, immr, imms),
        Opcode::Bfm {
            sf,
            rd: rdn,
            rn,
            immr,
            imms,
        } => ctx.bfm(sf, rdn, rn, immr, imms),
        Opcode::Ubfm {
            sf,
            rd: rdn,
            rn,
            immr,
            imms,
        } => ctx.ubfm(sf, rdn, rn, immr, imms),
        Opcode::ShiftReg {
            sf,
            kind,
            rd: rdn,
            rn,
            rm,
        } => ctx.shift_var(sf, kind, rdn, rn, rm),
        Opcode::Udiv { sf, rd: rdn, rn, rm } => ctx.div(sf, false, rdn, rn, rm),
        Opcode::Sdiv { sf, rd: rdn, rn, rm } => ctx.div(sf, true, rdn, rn, rm),
        Opcode::Madd {
            sf,
            rd: rdn,
            rn,
            rm,
            ra,
        }
        | Opcode::Msub {
            sf,
            rd: rdn,
            rn,
            rm,
            ra,
        } => {
            let sub = matches!(insn.opcode, Opcode::Msub { .. });
            let w = sf_w(sf);
            let op = if sub { BinOp::Sub } else { BinOp::Add };
            let prod = bin(BinOp::Mul, read_zr(rn, w), read_zr(rm, w));
            write_zr(rdn, bin(op, read_zr(ra, w), prod), w)
        }
        Opcode::Maddl {
            signed,
            sub,
            rd: rdn,
            rn,
            rm,
            ra,
        } => {
            // 32x32 -> 64 widening multiply-accumulate: extend the W
            // sources to 64 bits, multiply and accumulate at W64.
            let ext = |r: u8| {
                let narrow = read_zr(r, Width::W32);
                if signed {
                    un(UnOp::SignExtend(Width::W64), narrow)
                } else {
                    un(UnOp::ZeroExtend(Width::W64), narrow)
                }
            };
            let op = if sub { BinOp::Sub } else { BinOp::Add };
            let prod = bin(BinOp::Mul, ext(rn), ext(rm));
            let value = bin(op, read_zr(ra, Width::W64), prod);
            write_zr(rdn, value, Width::W64)
        }
        Opcode::Mulh {
            signed,
            rd: rdn,
            rn,
            rm,
        } => {
            // The IR has no 128-bit multiply, so the product's high half
            // is a named intrinsic with exact read/write sets — precise
            // for dataflow, opaque in value.
            let name = if signed { "a64.smulh" } else { "a64.umulh" };
            let writes = if rdn == 31 { vec![] } else { vec![cell(rdn)] };
            intr(
                name,
                writes,
                vec![read_zr(rn, Width::W64), read_zr(rm, Width::W64)],
            )
        }

        Opcode::Movz { sf, rd: rdn, imm, shift } => {
            let w = sf_w(sf);
            write_zr(rdn, k((imm as u64) << shift, w), w)
        }
        Opcode::Movn { sf, rd: rdn, imm, shift } => {
            let w = sf_w(sf);
            write_zr(rdn, k(!((imm as u64) << shift), w), w)
        }
        Opcode::Movk { sf, rd: rdn, imm, shift } => {
            let w = sf_w(sf);
            let kept = bin(BinOp::And, read_zr(rdn, w), k(!(0xFFFFu64 << shift), w));
            let value = bin(BinOp::Or, kept, k((imm as u64) << shift, w));
            write_zr(rdn, value, w)
        }

        // ---- conditional select ----
        Opcode::Csel { sf, rd: rdn, rn, rm, cond } => {
            ctx.csel(sf, rdn, rn, rm, cond, SelAlt::Value)
        }
        Opcode::Csinc { sf, rd: rdn, rn, rm, cond } => {
            ctx.csel(sf, rdn, rn, rm, cond, SelAlt::Inc)
        }
        Opcode::Csinv { sf, rd: rdn, rn, rm, cond } => {
            ctx.csel(sf, rdn, rn, rm, cond, SelAlt::Inv)
        }
        Opcode::Csneg { sf, rd: rdn, rn, rm, cond } => {
            ctx.csel(sf, rdn, rn, rm, cond, SelAlt::Neg)
        }

        // ---- loads and stores ----
        Opcode::Ldr { size, rt, rn, mode } => ctx.ldr_imm(size, rt, rn, mode),
        Opcode::Str { size, rt, rn, mode } => ctx.str_imm(size, rt, rn, mode),
        Opcode::Ldrs {
            size,
            sf,
            rt,
            rn,
            mode,
        } => ctx.ldrs_imm(size, sf, rt, rn, mode),
        // The unscaled forms are the immediate forms with an offset mode
        // and no writeback — the shared bodies already lift that shape.
        Opcode::Ldur { size, rt, rn, imm } => ctx.ldr_imm(size, rt, rn, AddrMode::Offset(imm)),
        Opcode::Stur { size, rt, rn, imm } => ctx.str_imm(size, rt, rn, AddrMode::Offset(imm)),
        Opcode::Ldurs {
            size,
            sf,
            rt,
            rn,
            imm,
        } => ctx.ldrs_imm(size, sf, rt, rn, AddrMode::Offset(imm)),
        Opcode::LdrReg { size, rt, rn, off } => ctx.ldr_reg(size, rt, rn, off, insn.raw),
        Opcode::StrReg { size, rt, rn, off } => ctx.str_reg(size, rt, rn, off, insn.raw),
        Opcode::LdrsReg {
            size,
            sf,
            rt,
            rn,
            off,
        } => ctx.ldrs_reg(size, sf, rt, rn, off, insn.raw),
        Opcode::LdrLit { sf, rt, target } => ctx.ldr_lit(sf, rt, target),
        Opcode::Ldp {
            sf,
            rt,
            rt2,
            rn,
            mode,
        } => ctx.ldp(sf, rt, rt2, rn, mode),
        Opcode::Stp {
            sf,
            rt,
            rt2,
            rn,
            mode,
        } => ctx.stp(sf, rt, rt2, rn, mode),

        // ---- SIMD&FP loads and stores ----
        Opcode::FLdr { size, rt, rn, mode } => ctx.fldr(size, rt, rn, mode),
        Opcode::FStr { size, rt, rn, mode } => ctx.fstr(size, rt, rn, mode),
        // The unscaled forms are the immediate forms with an offset mode
        // and no writeback, exactly as on the integer side.
        Opcode::FLdur { size, rt, rn, imm } => ctx.fldr(size, rt, rn, AddrMode::Offset(imm)),
        Opcode::FStur { size, rt, rn, imm } => ctx.fstr(size, rt, rn, AddrMode::Offset(imm)),
        Opcode::FLdrReg { size, rt, rn, off } => ctx.fldr_reg(size, rt, rn, off, insn.raw),
        Opcode::FStrReg { size, rt, rn, off } => ctx.fstr_reg(size, rt, rn, off, insn.raw),
        Opcode::FLdrLit { size, rt, target } => ctx.fldr_lit(size, rt, target),
        Opcode::FLdp {
            size,
            rt,
            rt2,
            rn,
            mode,
        } => ctx.fldp(size, rt, rt2, rn, mode),
        Opcode::FStp {
            size,
            rt,
            rt2,
            rn,
            mode,
        } => ctx.fstp(size, rt, rt2, rn, mode),

        // ---- SIMD&FP moves ----
        Opcode::FmovReg { double, rd: rdn, rn } => {
            if double {
                vec![
                    assign(vlo(rdn), rd(vlo(rn))),
                    assign(vhi(rdn), k(0, Width::W64)),
                ]
            } else {
                vwrite_scalar(rdn, vread(rn, Width::W32), Width::W32)
            }
        }
        Opcode::FmovToGp { sf, hi, rd: rdn, rn } => {
            let w = sf_w(sf);
            let value = if hi { rd(vhi(rn)) } else { vread(rn, w) };
            write_zr(rdn, value, w)
        }
        Opcode::FmovFromGp { sf, hi, rd: rdn, rn } => {
            let src = read_zr(rn, sf_w(sf));
            if hi {
                // The one SIMD write that touches a half in isolation:
                // the D[1] lane insert keeps the low half.
                vec![assign(vhi(rdn), src)]
            } else {
                vwrite_scalar(rdn, src, sf_w(sf))
            }
        }
        Opcode::FmovImm { double, imm, rd: rdn } => {
            let w = if double { Width::W64 } else { Width::W32 };
            vwrite_scalar(rdn, k(fp_imm_bits(double, imm), w), w)
        }
        Opcode::FmovVecImm { q, double, imm, rd: rdn } => {
            let elem = fp_imm_bits(double, imm);
            let imm64 = if double { elem } else { elem << 32 | elem };
            vmove_imm64(rdn, imm64, q)
        }
        Opcode::Movi {
            q,
            invert,
            size,
            imm,
            shift,
            msl,
            rd: rdn,
        } => vmove_imm64(rdn, aarch64::movi_expand(size, imm, shift, msl, invert), q),

        // ---- scalar FP arithmetic: precise intrinsics over exact cells ----
        Opcode::FArith2 {
            op,
            double,
            rd: rdn,
            rn,
            rm,
        } => {
            let w = if double { Width::W64 } else { Width::W32 };
            fp_scalar_intr(f2_name(op), rdn, vec![vread(rn, w), vread(rm, w)])
        }
        Opcode::FArith3 {
            negate,
            sub,
            double,
            rd: rdn,
            rn,
            rm,
            ra,
        } => {
            let name = match (negate, sub) {
                (false, false) => "a64.fmadd",
                (false, true) => "a64.fmsub",
                (true, false) => "a64.fnmadd",
                (true, true) => "a64.fnmsub",
            };
            let w = if double { Width::W64 } else { Width::W32 };
            fp_scalar_intr(name, rdn, vec![vread(rn, w), vread(rm, w), vread(ra, w)])
        }
        Opcode::FArith1 {
            op,
            double,
            rd: rdn,
            rn,
        } => {
            let w = if double { Width::W64 } else { Width::W32 };
            match op {
                // FABS/FNEG are sign-bit masks — exact, no FP semantics.
                aarch64::F1Op::Abs => {
                    let mask = ones(w.bits() - 1);
                    vwrite_scalar(rdn, bin(BinOp::And, vread(rn, w), k(mask, w)), w)
                }
                aarch64::F1Op::Neg => {
                    let sign = 1u64 << (w.bits() - 1);
                    vwrite_scalar(rdn, bin(BinOp::Xor, vread(rn, w), k(sign, w)), w)
                }
                _ => fp_scalar_intr(f1_name(op), rdn, vec![vread(rn, w)]),
            }
        }
        Opcode::FCvtPrec {
            to_double,
            rd: rdn,
            rn,
        } => {
            let src = if to_double { Width::W32 } else { Width::W64 };
            fp_scalar_intr("a64.fcvt", rdn, vec![vread(rn, src)])
        }
        Opcode::Fcmp {
            double,
            signal,
            rn,
            rm,
        } => {
            let w = if double { Width::W64 } else { Width::W32 };
            let comparand = match rm {
                Some(rm) => vread(rm, w),
                None => k(0, w),
            };
            intr(
                if signal { "a64.fcmpe" } else { "a64.fcmp" },
                vec![
                    flag(Flag::Sign),
                    flag(Flag::Zero),
                    flag(Flag::Carry),
                    flag(Flag::Overflow),
                ],
                vec![vread(rn, w), comparand],
            )
        }
        Opcode::Fccmp {
            double,
            signal,
            rn,
            rm,
            nzcv,
            cond,
        } => {
            let w = if double { Width::W64 } else { Width::W32 };
            intr(
                if signal { "a64.fccmpe" } else { "a64.fccmp" },
                vec![
                    flag(Flag::Sign),
                    flag(Flag::Zero),
                    flag(Flag::Carry),
                    flag(Flag::Overflow),
                ],
                vec![
                    cond_expr(cond),
                    vread(rn, w),
                    vread(rm, w),
                    k(nzcv as u64, Width::W64),
                ],
            )
        }
        Opcode::Fcsel {
            double,
            rd: rdn,
            rn,
            rm,
            cond,
        } => ctx.fcsel(double, rdn, rn, rm, cond),
        Opcode::FcvtToFp {
            sf,
            double,
            unsigned,
            rd: rdn,
            rn,
        } => fp_scalar_intr(
            if unsigned { "a64.ucvtf" } else { "a64.scvtf" },
            rdn,
            vec![read_zr(rn, sf_w(sf)), k(double as u64, Width::W64)],
        ),
        Opcode::FcvtFromFp {
            sf,
            double,
            unsigned,
            round,
            rd: rdn,
            rn,
        } => {
            let w = if double { Width::W64 } else { Width::W32 };
            let reads = vec![vread(rn, w), k(sf as u64, Width::W64)];
            if rdn == 31 {
                // A convert to the zero register keeps its reads (the
                // signalling side effect) and defines nothing.
                intr(fcvt_name(round, unsigned), vec![], reads)
            } else {
                intr(fcvt_name(round, unsigned), vec![cell(rdn)], reads)
            }
        }
        Opcode::FcvtIntScalar {
            double,
            unsigned,
            rd: rdn,
            rn,
        } => {
            let w = if double { Width::W64 } else { Width::W32 };
            fp_scalar_intr(
                if unsigned { "a64.ucvtf" } else { "a64.scvtf" },
                rdn,
                vec![vread(rn, w)],
            )
        }

        // ---- element moves: exact shifts and masks over the half cells ----
        Opcode::DupGp {
            q,
            size,
            rd: rdn,
            rn,
        } => {
            let elem = match size {
                3 => read_zr(rn, Width::W64),
                _ => bin(
                    BinOp::And,
                    read_zr(rn, Width::W64),
                    k(ones(8u32 << size), Width::W64),
                ),
            };
            ctx.dup_replicate(rdn, elem, size, q)
        }
        Opcode::DupElemScalar {
            size,
            index,
            rd: rdn,
            rn,
        } => {
            // The scalar form zeroes everything above the element.
            vec![
                assign(vlo(rdn), velem(size, index, rn)),
                assign(vhi(rdn), k(0, Width::W64)),
            ]
        }
        Opcode::DupElemVec {
            q,
            size,
            index,
            rd: rdn,
            rn,
        } => {
            let elem = velem(size, index, rn);
            ctx.dup_replicate(rdn, elem, size, q)
        }
        Opcode::Umov {
            sf,
            size,
            index,
            rd: rdn,
            rn,
        } => {
            let w = sf_w(sf);
            let value = match w {
                Width::W64 => velem(size, index, rn),
                _ => un(UnOp::Truncate(w), velem(size, index, rn)),
            };
            write_zr(rdn, value, w)
        }
        Opcode::Smov {
            sf,
            size,
            index,
            rd: rdn,
            rn,
        } => {
            let w = sf_w(sf);
            let narrow = un(UnOp::Truncate(elem_width(size)), velem(size, index, rn));
            write_zr(rdn, un(UnOp::SignExtend(w), narrow), w)
        }
        Opcode::InsGp {
            size,
            index,
            rd: rdn,
            rn,
        } => {
            let elem = match size {
                3 => read_zr(rn, Width::W64),
                _ => bin(
                    BinOp::And,
                    read_zr(rn, Width::W64),
                    k(ones(8u32 << size), Width::W64),
                ),
            };
            ctx.ins_elem(rdn, size, index, elem)
        }
        Opcode::InsElem {
            size,
            dst,
            src,
            rd: rdn,
            rn,
        } => {
            let elem = velem(size, src, rn);
            ctx.ins_elem(rdn, size, dst, elem)
        }
        Opcode::SimdAlu {
            op,
            q,
            size,
            rd: rdn,
            rn,
            rm,
        } => simd_alu(op, q, size, rdn, rn, rm),

        // ---- exclusives and ordered accesses ----
        // Acquire/release ordering has no single-threaded content; the
        // exclusive monitor is not modeled. The loads are plain loads,
        // the stores plain stores, and the exclusive-store status an
        // opaque named definition — see `Ctx::stxr`.
        Opcode::Ldar { size, rt, rn } | Opcode::Ldxr { size, rt, rn, .. } => {
            let lw = size_w(size);
            ctx.load_common(lw, addr_of(rn, AddrMode::Offset(0)), None, |v| {
                write_zr(rt, v, lw)
            })
        }
        Opcode::Stlr { size, rt, rn } => {
            let lw = size_w(size);
            ctx.store_common(addr_of(rn, AddrMode::Offset(0)), None, read_zr(rt, lw))
        }
        Opcode::Stxr {
            size, ws, rt, rn, ..
        } => ctx.stxr(size, rt, ws, rn),

        // ---- pointer authentication ----
        // PAC bits live outside the value model: signing and stripping
        // are opaque rewrites of the one register they touch, and the
        // authenticated branches transfer control exactly like their
        // plain forms (an authentication trap is not control flow the
        // decompiler models).
        Opcode::PacGpr {
            auth,
            rd: rdn,
            rn,
            zero,
            ..
        } => {
            let modifier = if zero { k(0, Width::W64) } else { read_sp(rn, Width::W64) };
            intr(
                if auth { "a64.aut" } else { "a64.pac" },
                vec![cell(rdn)],
                vec![rd(cell(rdn)), modifier],
            )
        }
        Opcode::XPac { rd: rdn, .. } => {
            intr("a64.xpac", vec![cell(rdn)], vec![rd(cell(rdn))])
        }
        Opcode::PacHint { auth, .. } => intr(
            if auth { "a64.aut" } else { "a64.pac" },
            vec![cell(LR)],
            vec![rd(cell(LR)), rd(cell(31))],
        ),
        Opcode::RetA { .. } => vec![Stmt::Branch {
            kind: ir::BranchKind::Return,
            cond: None,
            target: read_zr(30, Width::W64),
        }],
        Opcode::BrAuth { link, rn, .. } => {
            if link {
                let t = ctx.fresh(Width::W64);
                vec![
                    assign(t, read_zr(rn, Width::W64)),
                    assign(cell(LR), k(ctx.next_va(), Width::W64)),
                    Stmt::Branch {
                        kind: ir::BranchKind::Call,
                        cond: None,
                        target: rd(t),
                    },
                ]
            } else {
                vec![Stmt::Branch {
                    kind: ir::BranchKind::Jump,
                    cond: None,
                    target: read_zr(rn, Width::W64),
                }]
            }
        }

        // ---- the integer one-source row, extract, LDPSW, UDF ----
        Opcode::Bits1 {
            op,
            sf,
            rd: rdn,
            rn,
        } => {
            if rdn == 31 {
                Vec::new()
            } else {
                intr(
                    bit1_name(op),
                    vec![cell(rdn)],
                    vec![read_zr(rn, sf_w(sf))],
                )
            }
        }
        Opcode::Extr {
            sf,
            rd: rdn,
            rn,
            rm,
            lsb,
        } => {
            let w = sf_w(sf);
            let value = if lsb == 0 {
                read_zr(rm, w)
            } else {
                bin(
                    BinOp::Or,
                    bin(BinOp::LShr, read_zr(rm, w), k(lsb as u64, w)),
                    bin(
                        BinOp::Shl,
                        read_zr(rn, w),
                        k(w.bits() as u64 - lsb as u64, w),
                    ),
                )
            };
            write_zr(rdn, value, w)
        }
        Opcode::LdpSw { rt, rt2, rn, mode } => ctx.ldpsw(rt, rt2, rn, mode),
        Opcode::Udf { .. } => intr("udf", vec![], vec![]),

        // ---- branches ----
        Opcode::B { target } => vec![Stmt::Branch {
            kind: ir::BranchKind::Jump,
            cond: None,
            target: k(target, Width::W64),
        }],
        Opcode::BCond { cond, target } => vec![Stmt::Branch {
            kind: ir::BranchKind::Jump,
            // AL/NV are architecturally "always": an unconditional jump.
            cond: if cond.is_al_nv() {
                None
            } else {
                Some(cond_expr(cond))
            },
            target: k(target, Width::W64),
        }],
        Opcode::Cbz { sf, rt, target } => ctx.cb(sf, rt, target, BinOp::Eq),
        Opcode::Cbnz { sf, rt, target } => ctx.cb(sf, rt, target, BinOp::Ne),
        Opcode::Tbz { rt, bit, target } => ctx.tb(rt, bit, target, BinOp::Eq),
        Opcode::Tbnz { rt, bit, target } => ctx.tb(rt, bit, target, BinOp::Ne),
        Opcode::Bl { target } => vec![
            assign(cell(LR), k(ctx.next_va(), Width::W64)),
            Stmt::Branch {
                kind: ir::BranchKind::Call,
                cond: None,
                target: k(target, Width::W64),
            },
        ],
        Opcode::Blr { rn } => {
            // Snapshot the target before writing the link register so
            // `blr x30` calls the old value.
            let t = ctx.fresh(Width::W64);
            vec![
                assign(t, read_zr(rn, Width::W64)),
                assign(cell(LR), k(ctx.next_va(), Width::W64)),
                Stmt::Branch {
                    kind: ir::BranchKind::Call,
                    cond: None,
                    target: rd(t),
                },
            ]
        }
        Opcode::Br { rn } => vec![Stmt::Branch {
            kind: ir::BranchKind::Jump,
            cond: None,
            target: read_zr(rn, Width::W64),
        }],
        Opcode::Ret { rn } => vec![Stmt::Branch {
            kind: ir::BranchKind::Return,
            cond: None,
            target: read_zr(rn, Width::W64),
        }],

        // ---- exceptions, hints, and the unmodeled remainder ----
        Opcode::Svc { imm } => exception_call("svc", imm),
        Opcode::Hvc { imm } => exception_call("hvc", imm),
        Opcode::Smc { imm } => exception_call("smc", imm),
        Opcode::Brk { .. } => intr("brk", vec![], vec![]),
        Opcode::Hlt { .. } => intr("hlt", vec![], vec![]),
        Opcode::Yield => intr("yield", vec![], vec![]),
        Opcode::Wfe => intr("wfe", vec![], vec![]),
        Opcode::Wfi => intr("wfi", vec![], vec![]),
        Opcode::Sev => intr("sev", vec![], vec![]),
        Opcode::Sevl => intr("sevl", vec![], vec![]),
        // No architectural effect: NOP, and unallocated hints execute as
        // NOP by definition.
        Opcode::Nop | Opcode::Hint { .. } => Vec::new(),
        Opcode::Unknown(raw) => unknown_intrinsic(raw),
    };
    *temp = ctx.temp;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aarch64::decode;

    const VA: u64 = 0x1000;

    /// Decode one instruction word at the fixed test VA and lift it.
    /// Decoding is total on 4 bytes, so this never fails.
    fn lift_word(w: u32) -> Vec<Stmt> {
        let insn = decode(&w.to_le_bytes(), VA).expect("decode");
        lift(&insn, VA)
    }

    /// Lift, assert well-formedness, and render with the crate namer.
    fn text(w: u32) -> String {
        let stmts = lift_word(w);
        assert_eq!(ir::check(&stmts), Ok(()), "check failed for {w:#010x}");
        ir::render(&stmts, &reg_name).trim_end().to_string()
    }

    /// Assert the lift of `w` is well-formed.
    fn ok(w: u32) {
        assert_eq!(
            ir::check(&lift_word(w)),
            Ok(()),
            "check failed for {w:#010x}"
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

    fn writes_any_arch(stmts: &[Stmt]) -> bool {
        (0u8..=31).any(|n| writes_cell(stmts, n))
    }

    // ---- the register namer ----

    #[test]
    fn reg_name_maps_numbers_and_widths() {
        assert_eq!(reg_name(0, Width::W64).as_deref(), Some("x0"));
        assert_eq!(reg_name(30, Width::W64).as_deref(), Some("x30"));
        assert_eq!(reg_name(31, Width::W64).as_deref(), Some("sp"));
        assert_eq!(reg_name(5, Width::W32).as_deref(), Some("w5"));
        assert_eq!(reg_name(31, Width::W32).as_deref(), Some("wsp"));
        // The SIMD&FP file: low halves as the D (and S) view, high
        // halves with the `hi` suffix; nothing past cell 95.
        assert_eq!(reg_name(32, Width::W64).as_deref(), Some("d0"));
        assert_eq!(reg_name(63, Width::W64).as_deref(), Some("d31"));
        assert_eq!(reg_name(33, Width::W32).as_deref(), Some("s1"));
        assert_eq!(reg_name(64, Width::W64).as_deref(), Some("v0hi"));
        assert_eq!(reg_name(95, Width::W64).as_deref(), Some("v31hi"));
        assert_eq!(reg_name(64, Width::W32), None);
        assert_eq!(reg_name(96, Width::W64), None);
        assert_eq!(reg_name(0, Width::W8), None);
        assert_eq!(reg_name(0, Width::W1), None);
    }

    // ---- add/subtract immediate and the W-register rule ----

    #[test]
    fn add_imm_64_is_a_plain_cell_op() {
        // add x0, x1, #1.
        assert_eq!(text(0x9100_0420), "x0 := (x1 + 0x1.q)");
    }

    #[test]
    fn add_imm_32_zero_extends_into_the_x_cell() {
        // add w0, w1, #1: computed at W32, widened into the cell.
        assert_eq!(text(0x1100_0420), "x0 := zext.q((trunc.d(x1) + 0x1.d))");
    }

    #[test]
    fn add_imm_carries_the_pre_shifted_immediate() {
        // add x2, x3, #1, lsl #12: the decoder stores imm fully shifted.
        assert_eq!(text(0x9140_0462), "x2 := (x3 + 0x1000.q)");
    }

    #[test]
    fn add_and_sub_treat_r31_operands_as_sp() {
        // sub sp, sp, #16.
        assert_eq!(text(0xD100_43FF), "sp := (sp - 0x10.q)");
        // mov x0, sp (add x0, sp, #0).
        assert_eq!(text(0x9100_03E0), "x0 := (sp + 0x0.q)");
        // mov sp, x0 (add sp, x0, #0): rd = 31 is SP, not a discard.
        assert_eq!(text(0x9100_001F), "sp := (x0 + 0x0.q)");
    }

    #[test]
    fn adds_sets_nzcv_with_the_a64_add_carry() {
        // adds x0, x1, #2.
        let t = text(0xB100_0820);
        assert!(t.contains("ZF := (t2.q == 0x0.q)"), "{t}");
        assert!(t.contains("SF := (t2.q <s 0x0.q)"), "{t}");
        assert!(t.contains("CF := (t2.q <u t0.q)"), "{t}");
        assert!(t.contains("OF := (((t0.q ^ t2.q) & (t1.q ^ t2.q)) <s 0x0.q)"), "{t}");
        assert!(t.contains("x0 := t2.q"), "{t}");
    }

    #[test]
    fn subs_carry_is_not_borrow() {
        // subs x0, x1, #1: A64 sets C when no borrow occurs (lhs >= rhs),
        // the opposite of x86's borrow flag.
        let t = text(0xF100_0420);
        assert!(t.contains("CF := (t1.q <=u t0.q)"), "{t}");
        assert!(t.contains("OF := (((t0.q ^ t1.q) & (t0.q ^ t2.q)) <s 0x0.q)"), "{t}");
    }

    #[test]
    fn cmp_writes_flags_only() {
        // cmp x2, #1 (subs xzr, x2, #1): rd = 31 is ZR — no register write.
        assert_eq!(
            text(0xF100_045F),
            "t0.q := x2\n\
             t1.q := 0x1.q\n\
             t2.q := (t0.q - t1.q)\n\
             ZF := (t2.q == 0x0.q)\n\
             SF := (t2.q <s 0x0.q)\n\
             CF := (t1.q <=u t0.q)\n\
             OF := (((t0.q ^ t1.q) & (t0.q ^ t2.q)) <s 0x0.q)"
        );
        assert!(!writes_any_arch(&lift_word(0xF100_045F)));
    }

    #[test]
    fn cmn_is_a_flags_only_add() {
        // cmn x0, #4 (adds xzr, x0, #4).
        let s = lift_word(0xB100_101F);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(has_flag_assign(&s, Flag::Carry));
        assert!(has_flag_assign(&s, Flag::Overflow));
        assert!(!writes_any_arch(&s));
    }

    #[test]
    fn thirty_two_bit_flags_are_computed_at_w32() {
        // subs w0, w1, #5: temps and flag exprs at W32, result widened.
        let t = text(0x7100_1420);
        assert!(t.contains("t0.d := trunc.d(x1)"), "{t}");
        assert!(t.contains("CF := (t1.d <=u t0.d)"), "{t}");
        assert!(t.contains("x0 := zext.q(t2.d)"), "{t}");
    }

    // ---- pc-relative addresses ----

    #[test]
    fn adr_and_adrp_are_absolute_constants() {
        // adr x0, 0x1010 (decoded at 0x1000).
        assert_eq!(text(0x1000_0080), "x0 := 0x1010.q");
        // adrp x1, +1 page: the pc's low 12 bits drop.
        assert_eq!(text(0xB000_0001), "x1 := 0x2000.q");
    }

    // ---- move wide ----

    #[test]
    fn movz_and_movn_build_constants() {
        // movz w1, #5.
        assert_eq!(text(0x5280_00A1), "x1 := zext.q(0x5.d)");
        // movz x0, #0x1234.
        assert_eq!(text(0xD282_4680), "x0 := 0x1234.q");
        // movn x2, #0 (mov x2, #-1).
        assert_eq!(text(0x9280_0002), "x2 := 0xffffffffffffffff.q");
        // movn w0, #0: the complement is masked to 32 bits.
        assert_eq!(text(0x1280_0000), "x0 := zext.q(0xffffffff.d)");
    }

    #[test]
    fn movk_merges_sixteen_bits_read_modify_write() {
        // movk x0, #0xbeef, lsl #16.
        assert_eq!(
            text(0xF2B7_DDE0),
            "x0 := ((x0 & 0xffffffff0000ffff.q) | 0xbeef0000.q)"
        );
        // movk w0, #1: mask and merge at W32, then widen.
        assert_eq!(
            text(0x7280_0020),
            "x0 := zext.q(((trunc.d(x0) & 0xffff0000.d) | 0x1.d))"
        );
    }

    #[test]
    fn a_move_to_the_zero_register_lifts_to_nothing() {
        // movz xzr, #1: the write is discarded and nothing else happens.
        assert!(lift_word(0xD280_003F).is_empty());
    }

    // ---- loads and stores, immediate ----

    #[test]
    fn ldr_from_sp_snapshots_the_address_first() {
        // ldr x0, [sp, #16].
        assert_eq!(
            text(0xF940_0BE0),
            "t0.q := (sp + 0x10.q)\n\
             t1.q := load.q [t0.q]\n\
             x0 := t1.q"
        );
    }

    #[test]
    fn ldr_w_zero_extends_the_cell() {
        // ldr w2, [x1].
        assert_eq!(
            text(0xB940_0022),
            "t0.q := x1\n\
             t1.d := load.d [t0.q]\n\
             x2 := zext.q(t1.d)"
        );
    }

    #[test]
    fn ldr_to_wzr_keeps_the_load_but_discards_the_write() {
        // ldr wzr, [x0]: the (possibly faulting) load stays; no cell write.
        assert_eq!(
            text(0xB940_001F),
            "t0.q := x0\n\
             t1.d := load.d [t0.q]"
        );
        assert!(!writes_any_arch(&lift_word(0xB940_001F)));
    }

    #[test]
    fn str_of_wzr_stores_zero() {
        // str wzr, [x0].
        assert_eq!(
            text(0xB900_001F),
            "t0.q := x0\n\
             store.d [t0.q], 0x0.d"
        );
    }

    #[test]
    fn byte_and_halfword_accesses_use_their_widths() {
        // ldrb w3, [x4, #1].
        assert_eq!(
            text(0x3940_0483),
            "t0.q := (x4 + 0x1.q)\n\
             t1.b := load.b [t0.q]\n\
             x3 := zext.q(t1.b)"
        );
        // strb w1, [x2].
        assert_eq!(
            text(0x3900_0041),
            "t0.q := x2\n\
             store.b [t0.q], trunc.b(x1)"
        );
        // ldrh w2, [x3, #2].
        let t = text(0x7940_0462);
        assert!(t.contains("load.w"), "{t}");
    }

    #[test]
    fn post_index_load_snapshots_then_writes_back_then_lands() {
        // ldr x0, [x0], #8: rt == rn — the snapshot makes it deterministic,
        // and the destination write lands last (x0 ends holding the load).
        assert_eq!(
            text(0xF840_8400),
            "t0.q := x0\n\
             t1.q := load.q [t0.q]\n\
             x0 := (t0.q + 0x8.q)\n\
             x0 := t1.q"
        );
    }

    #[test]
    fn pre_index_load_writes_back_the_new_base() {
        // ldr x0, [x1, #-8]!.
        assert_eq!(
            text(0xF85F_8C20),
            "t0.q := (x1 + 0xfffffffffffffff8.q)\n\
             t1.q := load.q [t0.q]\n\
             x1 := t0.q\n\
             x0 := t1.q"
        );
    }

    #[test]
    fn post_index_store_stores_before_the_writeback() {
        // str x0, [sp], #16.
        assert_eq!(
            text(0xF801_07E0),
            "t0.q := sp\n\
             store.q [t0.q], x0\n\
             sp := (t0.q + 0x10.q)"
        );
    }

    // ---- pairs ----

    #[test]
    fn ldp_epilogue_loads_both_then_adjusts_sp_then_lands() {
        // ldp x29, x30, [sp], #16 — the canonical epilogue word.
        assert_eq!(
            text(0xA8C1_7BFD),
            "t0.q := sp\n\
             t1.q := load.q [t0.q]\n\
             t2.q := load.q [(t0.q + 0x8.q)]\n\
             sp := (t0.q + 0x10.q)\n\
             x29 := t1.q\n\
             x30 := t2.q"
        );
    }

    #[test]
    fn stp_prologue_stores_both_then_writes_back() {
        // stp x29, x30, [sp, #-16]! — the canonical prologue word.
        assert_eq!(
            text(0xA9BF_7BFD),
            "t0.q := (sp + 0xfffffffffffffff0.q)\n\
             store.q [t0.q], x29\n\
             store.q [(t0.q + 0x8.q)], x30\n\
             sp := t0.q"
        );
    }

    #[test]
    fn thirty_two_bit_pairs_step_by_four() {
        // ldp w0, w1, [x2].
        assert_eq!(
            text(0x2940_0440),
            "t0.q := x2\n\
             t1.d := load.d [t0.q]\n\
             t2.d := load.d [(t0.q + 0x4.q)]\n\
             x0 := zext.q(t1.d)\n\
             x1 := zext.q(t2.d)"
        );
    }

    // ---- SIMD&FP loads, stores, and moves ----

    #[test]
    fn simd_scalar_load_zeroes_the_high_cell() {
        // ldr d6, [x7, #8]: the low cell takes the value, the high cell
        // is architecturally zeroed.
        assert_eq!(
            text(0xFD40_04E6),
            "t0.q := (x7 + 0x8.q)\n\
             t1.q := load.q [t0.q]\n\
             d6 := t1.q\n\
             v6hi := 0x0.q"
        );
        // ldr s4, [sp, #4]: sub-64 widths zero-extend into the low cell.
        assert_eq!(
            text(0xBD40_07E4),
            "t0.q := (sp + 0x4.q)\n\
             t1.d := load.d [t0.q]\n\
             d4 := zext.q(t1.d)\n\
             v4hi := 0x0.q"
        );
        // ldr b0, [x1, #1] and ldur h1, [x2, #5] use their widths.
        let t = text(0x3D40_0420);
        assert!(t.contains("load.b") && t.contains("d0 := zext.q(t1.b)"), "{t}");
        let t = text(0x7C40_5041);
        assert!(t.contains("load.w") && t.contains("d1 := zext.q(t1.w)"), "{t}");
    }

    #[test]
    fn simd_q_load_fills_both_cells() {
        // ldr q8, [x9, #0x10]: low half at the lower address.
        assert_eq!(
            text(0x3DC0_0528),
            "t0.q := (x9 + 0x10.q)\n\
             t1.q := load.q [t0.q]\n\
             t2.q := load.q [(t0.q + 0x8.q)]\n\
             d8 := t1.q\n\
             v8hi := t2.q"
        );
    }

    #[test]
    fn simd_store_reads_the_cells() {
        // str s5, [sp]: the sub-64 store truncates the low cell.
        assert_eq!(
            text(0xBD00_03E5),
            "t0.q := sp\n\
             store.d [t0.q], trunc.d(d5)"
        );
        // str q9, [x10, #0x20]: both cells, low first.
        assert_eq!(
            text(0x3D80_0949),
            "t0.q := (x10 + 0x20.q)\n\
             store.q [t0.q], d9\n\
             store.q [(t0.q + 0x8.q)], v9hi"
        );
    }

    #[test]
    fn simd_writeback_lands_between_loads_and_destinations() {
        // ldr q1, [x2], #16: snapshot, both loads, writeback, then the
        // destination cells — same fixed order as the integer loads.
        assert_eq!(
            text(0x3CC1_0441),
            "t0.q := x2\n\
             t1.q := load.q [t0.q]\n\
             t2.q := load.q [(t0.q + 0x8.q)]\n\
             x2 := (t0.q + 0x10.q)\n\
             d1 := t1.q\n\
             v1hi := t2.q"
        );
        // str s2, [x3, #-4]!: store before the base update.
        assert_eq!(
            text(0xBC1F_CC62),
            "t0.q := (x3 + 0xfffffffffffffffc.q)\n\
             store.d [t0.q], trunc.d(d2)\n\
             x3 := t0.q"
        );
    }

    #[test]
    fn simd_unscaled_forms_lift_like_the_scaled_forms() {
        // ldur d0, [x1] and ldr d0, [x1]: identical statement lists.
        assert_eq!(lift_word(0xFC40_0020), lift_word(0xFD40_0020));
        // ldur q2, [x3, #-0x10].
        let t = text(0x3CDF_0062);
        assert!(t.contains("t0.q := (x3 + 0xfffffffffffffff0.q)"), "{t}");
        assert!(t.contains("v2hi := t2.q"), "{t}");
    }

    #[test]
    fn simd_register_offset_addressing() {
        // ldr s2, [x4, w5, uxtw #2].
        assert_eq!(
            text(0xBC65_5882),
            "t0.q := (x4 + (zext.q(trunc.d(x5)) << 0x2.q))\n\
             t1.d := load.d [t0.q]\n\
             d2 := zext.q(t1.d)\n\
             v2hi := 0x0.q"
        );
        // ldr q1, [x2, x3, lsl #4]: the q form scales by 16.
        let t = text(0x3CE3_7841);
        assert!(t.contains("t0.q := (x2 + (x3 << 0x4.q))"), "{t}");
        // str q30, [x2, x3, sxtx].
        let t = text(0x3CA3_E85E);
        assert!(t.contains("store.q [t0.q], d30"), "{t}");
        assert!(t.contains("store.q [(t0.q + 0x8.q)], v30hi"), "{t}");
    }

    #[test]
    fn simd_pairs_load_and_store_every_half() {
        // stp d8, d9, [sp, #-0x10]! — the canonical FP prologue word.
        assert_eq!(
            text(0x6DBF_27E8),
            "t0.q := (sp + 0xfffffffffffffff0.q)\n\
             store.q [t0.q], d8\n\
             store.q [(t0.q + 0x8.q)], d9\n\
             sp := t0.q"
        );
        // ldp s0, s1, [x2]: 4-byte step, zeroed high cells.
        assert_eq!(
            text(0x2D40_0440),
            "t0.q := x2\n\
             t1.d := load.d [t0.q]\n\
             t2.d := load.d [(t0.q + 0x4.q)]\n\
             d0 := zext.q(t1.d)\n\
             v0hi := 0x0.q\n\
             d1 := zext.q(t2.d)\n\
             v1hi := 0x0.q"
        );
        // ldp q4, q5, [sp, #0x20]: four loads, then the four cells.
        assert_eq!(
            text(0xAD41_17E4),
            "t0.q := (sp + 0x20.q)\n\
             t1.q := load.q [t0.q]\n\
             t2.q := load.q [(t0.q + 0x8.q)]\n\
             t3.q := load.q [(t0.q + 0x10.q)]\n\
             t4.q := load.q [(t0.q + 0x18.q)]\n\
             d4 := t1.q\n\
             v4hi := t2.q\n\
             d5 := t3.q\n\
             v5hi := t4.q"
        );
        // ldp d6, d7, [x8], #0x10: writeback before the destinations.
        let t = text(0x6CC1_1D06);
        assert!(t.contains("x8 := (t0.q + 0x10.q)\nd6 := t1.q"), "{t}");
        // stp q0, q1, [x3]: four stores.
        assert_eq!(text(0xAD00_0460).matches("store.q").count(), 4);
    }

    #[test]
    fn simd_literal_loads_from_the_absolute_target() {
        // ldr d1, 0xffc (decoded at 0x1000).
        assert_eq!(
            text(0x5CFF_FFE1),
            "t0.q := load.q [0xffc.q]\n\
             d1 := t0.q\n\
             v1hi := 0x0.q"
        );
        // ldr q2, 0xff8: the high half's address is also a constant.
        assert_eq!(
            text(0x9CFF_FFC2),
            "t0.q := load.q [0xff8.q]\n\
             t1.q := load.q [0x1000.q]\n\
             d2 := t0.q\n\
             v2hi := t1.q"
        );
    }

    #[test]
    fn simd_load_then_store_round_trips_the_cells() {
        // ldr q0, [x0] ; str q0, [x1]: the stored values are reads of
        // exactly the cells the load wrote.
        let a = decode(&0x3DC0_0000u32.to_le_bytes(), VA).unwrap();
        let b = decode(&0x3D80_0020u32.to_le_bytes(), VA + 4).unwrap();
        let block = lift_block(&[(a, VA), (b, VA + 4)]);
        assert_eq!(ir::check(&block), Ok(()));
        let stored: Vec<&Expr> = block
            .iter()
            .filter_map(|s| match s {
                Stmt::Store { value, .. } => Some(value),
                _ => None,
            })
            .collect();
        assert_eq!(stored, [&rd(vlo(0)), &rd(vhi(0))]);
    }

    #[test]
    fn fmov_register_and_general_moves() {
        // fmov d2, d3: a scalar copy still zeroes the high cell.
        assert_eq!(text(0x1E60_4062), "d2 := d3\nv2hi := 0x0.q");
        // fmov s0, s1: the single form moves the low 32 bits.
        assert_eq!(
            text(0x1E20_4020),
            "d0 := zext.q(trunc.d(d1))\nv0hi := 0x0.q"
        );
        // fmov w0, s1 / fmov x2, d3: FP → general writes the X cell.
        assert_eq!(text(0x1E26_0020), "x0 := zext.q(trunc.d(d1))");
        assert_eq!(text(0x9E66_0062), "x2 := d3");
        // fmov s4, w5 / fmov d6, x7: general → FP zeroes the rest.
        assert_eq!(
            text(0x1E27_00A4),
            "d4 := zext.q(trunc.d(x5))\nv4hi := 0x0.q"
        );
        assert_eq!(text(0x9E67_00E6), "d6 := x7\nv6hi := 0x0.q");
        // The D[1] lane forms: the insert keeps the low half — the one
        // SIMD write that touches a half in isolation.
        assert_eq!(text(0x9EAE_0128), "x8 := v9hi");
        assert_eq!(text(0x9EAF_016A), "v10hi := x11");
        // fmov xzr, d3: the ZR destination discards the move.
        assert!(lift_word(0x9E66_007F).is_empty());
        // fmov s8, wzr: the ZR source is a zero.
        assert_eq!(text(0x1E27_03E8), "d8 := zext.q(0x0.d)\nv8hi := 0x0.q");
    }

    #[test]
    fn fmov_immediates_are_ieee_bit_constants() {
        // fmov s0, #1.0: the single-precision pattern, widened.
        assert_eq!(text(0x1E2E_1000), "d0 := zext.q(0x3f800000.d)\nv0hi := 0x0.q");
        // fmov d1, #-0.5.
        assert_eq!(
            text(0x1E7C_1001),
            "d1 := 0xbfe0000000000000.q\nv1hi := 0x0.q"
        );
        // fmov v14.2s, #1.0: the element replicated across the low cell,
        // high cell zeroed (a 64-bit vector write covers the register).
        assert_eq!(
            text(0x0F03_F60E),
            "d14 := 0x3f8000003f800000.q\nv14hi := 0x0.q"
        );
        // fmov v15.4s, #-2.5: the 128-bit form fills both cells.
        assert_eq!(
            text(0x4F04_F48F),
            "d15 := 0xc0200000c0200000.q\nv15hi := 0xc0200000c0200000.q"
        );
        // fmov v16.2d, #0.25.
        assert_eq!(
            text(0x6F02_F610),
            "d16 := 0x3fd0000000000000.q\nv16hi := 0x3fd0000000000000.q"
        );
    }

    #[test]
    fn movi_and_mvni_write_expanded_constants() {
        // movi v0.8b, #0x7f: replicated byte, 64-bit form.
        assert_eq!(
            text(0x0F03_E7E0),
            "d0 := 0x7f7f7f7f7f7f7f7f.q\nv0hi := 0x0.q"
        );
        // movi v1.16b, #0x80: the q form fills both cells.
        assert_eq!(
            text(0x4F04_E401),
            "d1 := 0x8080808080808080.q\nv1hi := 0x8080808080808080.q"
        );
        // movi v31.2d, #0 — the common vector-zero idiom.
        assert_eq!(text(0x6F00_E41F), "d31 := 0x0.q\nv31hi := 0x0.q");
        // movi d8, #0xff00ff00ff00ff00: the byte mask, D form.
        assert_eq!(
            text(0x2F05_E548),
            "d8 := 0xff00ff00ff00ff00.q\nv8hi := 0x0.q"
        );
        // mvni v11.4s, #0x2, lsl #16: the complemented element.
        assert_eq!(
            text(0x6F00_444B),
            "d11 := 0xfffdfffffffdffff.q\nv11hi := 0xfffdfffffffdffff.q"
        );
        // mvni v12.2s, #0x3, msl #8: shifting ones, then complemented.
        assert_eq!(
            text(0x2F00_C46C),
            "d12 := 0xfffffc00fffffc00.q\nv12hi := 0x0.q"
        );
        // movi v7.4s, #0x56, msl #16: the shifting-ones form.
        assert_eq!(
            text(0x4F02_D6C7),
            "d7 := 0x56ffff0056ffff.q\nv7hi := 0x56ffff0056ffff.q"
        );
    }

    // ---- register-offset addressing ----

    #[test]
    fn register_offset_extends_and_scales_the_index() {
        // str x0, [x1, x2]: bare 64-bit index.
        assert_eq!(
            text(0xF822_6820),
            "t0.q := (x1 + x2)\n\
             store.q [t0.q], x0"
        );
        // ldr x0, [x1, x2, lsl #3]: scaled by the access size.
        let t = text(0xF862_7820);
        assert!(t.contains("t0.q := (x1 + (x2 << 0x3.q))"), "{t}");
        // ldr x0, [x1, w2, uxtw #3]: 32-bit index, zero-extended, scaled.
        let t = text(0xF862_5820);
        assert!(t.contains("(x1 + (zext.q(trunc.d(x2)) << 0x3.q))"), "{t}");
        // ldrsw x0, [x1, w2, sxtw]: sign-extended index, unscaled.
        assert_eq!(
            text(0xB8A2_C820),
            "t0.q := (x1 + sext.q(trunc.d(x2)))\n\
             t1.d := load.d [t0.q]\n\
             x0 := sext.q(t1.d)"
        );
        // ldrb w10, [x8, x9].
        let t = text(0x3869_690A);
        assert!(t.contains("t0.q := (x8 + x9)"), "{t}");
        assert!(t.contains("load.b"), "{t}");
    }

    // ---- unscaled loads and stores ----

    #[test]
    fn unscaled_accesses_lift_like_the_scaled_forms() {
        // ldur x0, [x1, #-1]: an offset no scaled encoding can express.
        assert_eq!(
            text(0xF85F_F020),
            "t0.q := (x1 + 0xffffffffffffffff.q)\n\
             t1.q := load.q [t0.q]\n\
             x0 := t1.q"
        );
        // ldur x7, [x8] and ldr x7, [x8] are the same access: statement
        // lists identical, only the rendered mnemonic differs.
        assert_eq!(lift_word(0xF840_0107), lift_word(0xF940_0107));
        // stur w0, [x1, #3].
        assert_eq!(
            text(0xB800_3020),
            "t0.q := (x1 + 0x3.q)\n\
             store.d [t0.q], trunc.d(x0)"
        );
        // sturb w9, [x10, #1] stores at byte width.
        let t = text(0x3800_1149);
        assert!(t.contains("store.b"), "{t}");
    }

    #[test]
    fn unscaled_sign_extending_loads_extend_to_their_register_width() {
        // ldursb x11, [x12, #-3]: byte straight to 64 bits.
        assert_eq!(
            text(0x389F_D18B),
            "t0.q := (x12 + 0xfffffffffffffffd.q)\n\
             t1.b := load.b [t0.q]\n\
             x11 := sext.q(t1.b)"
        );
        // ldursh w13, [x14, #5]: sign-extend to 32, then the W-register
        // rule zero-extends the cell.
        assert_eq!(
            text(0x78C0_51CD),
            "t0.q := (x14 + 0x5.q)\n\
             t1.w := load.w [t0.q]\n\
             x13 := zext.q(sext.d(t1.w))"
        );
        // ldursw x15, [x16, #2].
        let t = text(0xB880_220F);
        assert!(t.contains("x15 := sext.q(t1.d)"), "{t}");
    }

    // ---- sign-extending loads ----

    #[test]
    fn sign_extending_loads_extend_to_their_register_width() {
        // ldrsw x0, [x1, #4]: 32-bit load, sign-extended to the X register.
        assert_eq!(
            text(0xB980_0420),
            "t0.q := (x1 + 0x4.q)\n\
             t1.d := load.d [t0.q]\n\
             x0 := sext.q(t1.d)"
        );
        // ldrsb w8, [sp, #0x2f]: byte to W — sign-extend to 32, then the
        // W-register rule zero-extends the cell.
        assert_eq!(
            text(0x39C0_BFE8),
            "t0.q := (sp + 0x2f.q)\n\
             t1.b := load.b [t0.q]\n\
             x8 := zext.q(sext.d(t1.b))"
        );
        // ldrsb x0, [x1]: byte straight to 64 bits.
        let t = text(0x3980_0020);
        assert!(t.contains("x0 := sext.q(t1.b)"), "{t}");
    }

    // ---- literal loads ----

    #[test]
    fn ldr_literal_loads_from_the_absolute_target() {
        // ldr x0, 0x1008 (decoded at 0x1000).
        assert_eq!(
            text(0x5800_0040),
            "t0.q := load.q [0x1008.q]\n\
             x0 := t0.q"
        );
        // ldr w1, 0x1008.
        assert_eq!(
            text(0x1800_0041),
            "t0.d := load.d [0x1008.q]\n\
             x1 := zext.q(t0.d)"
        );
    }

    // ---- branches ----

    #[test]
    fn b_is_an_unconditional_jump() {
        // b 0x1010.
        assert_eq!(text(0x1400_0004), "goto 0x1010.q");
        // b -0x10.
        assert_eq!(text(0x17FF_FFFC), "goto 0xff0.q");
    }

    #[test]
    fn bl_writes_the_link_register_then_calls() {
        // bl 0x1010 from 0x1000: x30 receives 0x1004.
        assert_eq!(text(0x9400_0004), "x30 := 0x1004.q\ncall 0x1010.q");
    }

    #[test]
    fn b_cond_carries_the_condition_expression() {
        // b.eq 0x1020.
        assert_eq!(text(0x5400_0100), "goto if ZF 0x1020.q");
        // b.hs: C set means no borrow — unsigned higher-or-same.
        assert_eq!(text(0x5400_0102), "goto if CF 0x1020.q");
        // b.lo: unsigned lower is a clear carry.
        assert_eq!(text(0x5400_0103), "goto if ~(CF) 0x1020.q");
        // b.hi and b.ls.
        assert_eq!(text(0x5400_0108), "goto if (CF & ~(ZF)) 0x1020.q");
        assert_eq!(text(0x5400_0109), "goto if (~(CF) | ZF) 0x1020.q");
        // b.ge / b.lt on the sign-overflow relation.
        assert_eq!(text(0x5400_010A), "goto if (SF == OF) 0x1020.q");
        assert_eq!(text(0x5400_010B), "goto if (SF != OF) 0x1020.q");
        // b.gt and b.le.
        assert_eq!(text(0x5400_010C), "goto if (~(ZF) & (SF == OF)) 0x1020.q");
        assert_eq!(text(0x5400_010D), "goto if (ZF | (SF != OF)) 0x1020.q");
    }

    #[test]
    fn b_al_and_b_nv_are_unconditional() {
        assert_eq!(text(0x5400_010E), "goto 0x1020.q");
        assert_eq!(text(0x5400_010F), "goto 0x1020.q");
    }

    #[test]
    fn cbz_and_cbnz_compare_against_zero_at_their_width() {
        // cbz w0, 0x1008.
        assert_eq!(text(0x3400_0040), "goto if (trunc.d(x0) == 0x0.d) 0x1008.q");
        // cbnz x5, 0xff8.
        assert_eq!(text(0xB5FF_FFC5), "goto if (x5 != 0x0.q) 0xff8.q");
        // cbz xzr, +8: the ZR read folds to a constant guard.
        assert_eq!(text(0xB400_005F), "goto if (0x0.q == 0x0.q) 0x1008.q");
    }

    #[test]
    fn tbz_and_tbnz_pick_the_bit_at_w64() {
        // tbz x3, #33, 0x1010.
        assert_eq!(
            text(0xB608_0083),
            "goto if (((x3 >>u 0x21.q) & 0x1.q) == 0x0.q) 0x1010.q"
        );
        // tbnz w2, #5, 0xff8.
        assert_eq!(
            text(0x372F_FFC2),
            "goto if (((x2 >>u 0x5.q) & 0x1.q) != 0x0.q) 0xff8.q"
        );
    }

    #[test]
    fn register_branches_target_the_cell() {
        assert_eq!(text(0xD61F_0200), "goto x16"); // br x16
        assert_eq!(text(0xD65F_03C0), "return x30"); // ret
        assert_eq!(text(0xD65F_0020), "return x1"); // ret x1
    }

    #[test]
    fn blr_snapshots_the_target_before_writing_the_link() {
        // blr x30: must call the old x30, not the fresh return address.
        assert_eq!(
            text(0xD63F_03C0),
            "t0.q := x30\n\
             x30 := 0x1004.q\n\
             call t0.q"
        );
        // blr x8.
        assert_eq!(
            text(0xD63F_0100),
            "t0.q := x8\n\
             x30 := 0x1004.q\n\
             call t0.q"
        );
    }

    // ---- conditional select ----

    #[test]
    fn csel_is_a_branchless_masked_merge() {
        // csel x0, x1, x2, eq.
        assert_eq!(
            text(0x9A82_0020),
            "t0.q := sext.q(ZF)\n\
             t1.q := x2\n\
             x0 := ((x1 & t0.q) | (t1.q & ~(t0.q)))"
        );
    }

    #[test]
    fn cset_reads_only_zeros_and_the_flags() {
        // cset w0, ne (csinc w0, wzr, wzr, eq).
        assert_eq!(
            text(0x1A9F_07E0),
            "t0.d := sext.d(ZF)\n\
             t1.d := (0x0.d + 0x1.d)\n\
             x0 := zext.q(((0x0.d & t0.d) | (t1.d & ~(t0.d))))"
        );
    }

    #[test]
    fn csinv_and_csneg_transform_the_alternative() {
        // csinv x0, x1, x2, lt.
        let t = text(0xDA82_B020);
        assert!(t.contains("t0.q := sext.q((SF != OF))"), "{t}");
        assert!(t.contains("t1.q := ~(x2)"), "{t}");
        // csneg x3, x4, x5, gt.
        let t = text(0xDA85_C483);
        assert!(t.contains("t0.q := sext.q((~(ZF) & (SF == OF)))"), "{t}");
        assert!(t.contains("t1.q := -(x5)"), "{t}");
        assert!(t.contains("x3 := ((x4 & t0.q) | (t1.q & ~(t0.q)))"), "{t}");
    }

    // ---- shifted-register arithmetic and logical ----

    #[test]
    fn add_reg_composes_the_shifted_operand() {
        // add x0, x1, x2.
        assert_eq!(text(0x8B02_0020), "x0 := (x1 + x2)");
        // add w0, w1, w2, lsl #3: computed at W32, widened into the cell.
        assert_eq!(
            text(0x0B02_0C20),
            "x0 := zext.q((trunc.d(x1) + (trunc.d(x2) << 0x3.d)))"
        );
        // sub w4, w5, w6, asr #7: the arithmetic shift.
        assert_eq!(
            text(0x4B86_1CA4),
            "x4 := zext.q((trunc.d(x5) - (trunc.d(x6) >>s 0x7.d)))"
        );
        // neg x0, x2 (sub from zr): the zero folds into the expression.
        assert_eq!(text(0xCB02_03E0), "x0 := (0x0.q - x2)");
    }

    #[test]
    fn adds_reg_reuses_the_immediate_flag_model() {
        // adds x0, x1, x2 and adds x0, x1, #2 must write textually
        // identical flag statements — one flag model, not two.
        let is_flag = |l: &&str| {
            l.starts_with("ZF") || l.starts_with("SF") || l.starts_with("CF") || l.starts_with("OF")
        };
        let reg = text(0xAB02_0020);
        let imm = text(0xB100_0820);
        assert_eq!(
            reg.lines().filter(is_flag).collect::<Vec<_>>(),
            imm.lines().filter(is_flag).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cmp_reg_writes_flags_only_with_not_borrow_carry() {
        // cmp x1, x2 (subs xzr, x1, x2).
        let s = lift_word(0xEB02_003F);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(!writes_any_arch(&s));
        let t = text(0xEB02_003F);
        assert!(t.contains("CF := (t1.q <=u t0.q)"), "{t}");
    }

    #[test]
    fn cmp_reg_then_b_cond_round_trips_the_condition() {
        // cmp w0, w1 ; b.lt — the flags written and the flags read must
        // meet in the same convention across the block.
        let a = decode(&0x6B01_001Fu32.to_le_bytes(), VA).unwrap();
        let b = decode(&0x5400_010Bu32.to_le_bytes(), VA + 4).unwrap();
        let block = lift_block(&[(a, VA), (b, VA + 4)]);
        assert_eq!(ir::check(&block), Ok(()));
        let t = ir::render(&block, &reg_name);
        assert!(t.contains("SF := (t2.d <s 0x0.d)"), "{t}");
        assert!(t.contains("goto if (SF != OF)"), "{t}");
    }

    #[test]
    fn logical_reg_ops_and_operand_inversion() {
        // and / orn / eor-with-rotate / mov (orr from zr).
        assert_eq!(text(0x8A02_0020), "x0 := (x1 & x2)");
        assert_eq!(text(0xAA22_0020), "x0 := (x1 | ~(x2))");
        assert_eq!(
            text(0xCAC5_3083),
            "x3 := (x4 ^ ((x5 >>u 0xc.q) | (x5 << 0x34.q)))"
        );
        assert_eq!(text(0xAA02_03E0), "x0 := (0x0.q | x2)");
    }

    #[test]
    fn ands_sets_nz_and_clears_cv() {
        // ands x0, x1, x2: N/Z from the result, C and V architecturally
        // cleared.
        assert_eq!(
            text(0xEA02_0020),
            "t0.q := (x1 & x2)\n\
             ZF := (t0.q == 0x0.q)\n\
             SF := (t0.q <s 0x0.q)\n\
             CF := 0x0.i1\n\
             OF := 0x0.i1\n\
             x0 := t0.q"
        );
        // tst x1, x2 (ands xzr, ...): flags only.
        let s = lift_word(0xEA02_003F);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(!writes_any_arch(&s));
        assert!(has_flag_assign(&s, Flag::Carry));
    }

    // ---- extended-register add/sub ----

    #[test]
    fn extended_register_extends_then_shifts() {
        // add x0, sp, w1, uxtw #2 — and rn = 31 is the SP cell.
        assert_eq!(
            text(0x8B21_4BE0),
            "x0 := (sp + (zext.q(trunc.d(x1)) << 0x2.q))"
        );
        // add x3, x4, w5, sxtb #1: a signed byte extend.
        assert_eq!(
            text(0x8B25_8483),
            "x3 := (x4 + (sext.q(trunc.b(x5)) << 0x1.q))"
        );
        // add sp, sp, x1: the identity extend, SP destination.
        assert_eq!(text(0x8B21_63FF), "sp := (sp + x1)");
        // cmp sp, x1: flag-setting form reads SP, writes no register.
        assert!(!writes_any_arch(&lift_word(0xEB21_63FF)));
    }

    // ---- logical immediate ----

    #[test]
    fn logical_immediate_is_a_constant_operand() {
        // and x0, x1, #0xff.
        assert_eq!(text(0x9240_1C20), "x0 := (x1 & 0xff.q)");
        // mov w1, #0x1010101 (orr from zr) at W32.
        assert_eq!(text(0x3200_C3E1), "x1 := zext.q((0x0.d | 0x1010101.d))");
        // and sp, x1, #0xfffffffffffffff0: rd = 31 is SP — the
        // frame-align idiom must write the SP cell.
        assert_eq!(
            text(0x927C_EC3F),
            "sp := (x1 & 0xfffffffffffffff0.q)"
        );
        // tst x1, #0x6: flags only.
        assert!(!writes_any_arch(&lift_word(0xF27F_043F)));
    }

    // ---- bitfield moves ----

    #[test]
    fn bitfield_extracts_and_inserts() {
        // lsr x4, x5, #16 (ubfm): shift right, mask the field.
        assert_eq!(
            text(0xD350_FCA4),
            "x4 := ((x5 >>u 0x10.q) & 0xffffffffffff.q)"
        );
        // lsl x0, x1, #8 (ubfm): mask, then shift left.
        assert_eq!(
            text(0xD378_DC20),
            "x0 := ((x1 & 0xffffffffffffff.q) << 0x8.q)"
        );
        // asr w6, w7, #31 (sbfm, field at the top): a single W32 ashr.
        assert_eq!(text(0x131F_7CE6), "x6 := zext.q((trunc.d(x7) >>s 0x1f.d))");
        // sxtb x2, w3 (sbfm): raise the byte to the top, sign-fill down.
        assert_eq!(text(0x9340_1C62), "x2 := ((x3 << 0x38.q) >>s 0x38.q)");
        // sbfiz x10, x11, #2, #5: sign-extend the field, then place it.
        assert_eq!(
            text(0x937E_116A),
            "x10 := (((x11 << 0x3b.q) >>s 0x3b.q) << 0x2.q)"
        );
        // bfi x0, x1, #8, #16: read-modify-write hole and fill.
        assert_eq!(
            text(0xB378_3C20),
            "x0 := ((x0 & 0xffffffffff0000ff.q) | ((x1 & 0xffff.q) << 0x8.q))"
        );
        // bfxil w2, w3, #4, #8: replace the low bits of the destination.
        assert_eq!(
            text(0x3304_2C62),
            "x2 := zext.q(((trunc.d(x2) & 0xffffff00.d) | ((trunc.d(x3) >>u 0x4.d) & 0xff.d)))"
        );
        // bfc x4, #16, #8: the zr source inserts a zero field.
        assert_eq!(
            text(0xB370_1FE4),
            "x4 := ((x4 & 0xffffffffff00ffff.q) | ((0x0.q & 0xff.q) << 0x10.q))"
        );
    }

    // ---- variable shifts and divides ----

    #[test]
    fn variable_shifts_mask_the_amount() {
        // lsl x0, x1, x2: the amount is rm modulo the width.
        assert_eq!(
            text(0x9AC2_2020),
            "t0.q := (x2 & 0x3f.q)\n\
             x0 := (x1 << t0.q)"
        );
        // lsr w3, w4, w5: the 32-bit form masks to 31.
        let t = text(0x1AC5_2483);
        assert!(t.contains("t0.d := (trunc.d(x5) & 0x1f.d)"), "{t}");
        // ror x9, x10, x11: (x >>u s) | (x << (-s & 63)) — the second
        // shift is 0 when s is, never a full-width shift.
        assert_eq!(
            text(0x9ACB_2D49),
            "t0.q := (x11 & 0x3f.q)\n\
             x9 := ((x10 >>u t0.q) | (x10 << (-(t0.q) & 0x3f.q)))"
        );
    }

    #[test]
    fn division_by_zero_yields_zero_not_a_trap() {
        // udiv x0, x1, x2: the divisor is forced to 1 when rm == 0 and
        // the quotient masked to 0 on the same condition.
        assert_eq!(
            text(0x9AC2_0820),
            "t0.i1 := (x2 == 0x0.q)\n\
             t1.q := (x1 /u (x2 | zext.q(t0.i1)))\n\
             x0 := (t1.q & ~(sext.q(t0.i1)))"
        );
        // sdiv x6, x7, x8 shares the guard around the signed operator.
        let t = text(0x9AC8_0CE6);
        assert!(t.contains("(x8 == 0x0.q)"), "{t}");
        assert!(t.contains("/s"), "{t}");
    }

    // ---- multiplies ----

    #[test]
    fn multiplies_and_the_widening_forms() {
        // madd x0, x1, x2, x3 and mul (ra = zr).
        assert_eq!(text(0x9B02_0C20), "x0 := (x3 + (x1 * x2))");
        assert_eq!(text(0x9B0A_7D28), "x8 := (0x0.q + (x9 * x10))");
        // msub w4, w5, w6, w7 at W32.
        assert_eq!(
            text(0x1B06_9CA4),
            "x4 := zext.q((trunc.d(x7) - (trunc.d(x5) * trunc.d(x6))))"
        );
        // smaddl x0, w1, w2, x3: W sources sign-extended, math at W64.
        assert_eq!(
            text(0x9B22_0C20),
            "x0 := (x3 + (sext.q(trunc.d(x1)) * sext.q(trunc.d(x2))))"
        );
        // umull x3, w4, w5: zero extends, zr accumulator.
        assert_eq!(
            text(0x9BA5_7C83),
            "x3 := (0x0.q + (zext.q(trunc.d(x4)) * zext.q(trunc.d(x5))))"
        );
    }

    #[test]
    fn mulh_is_a_precise_intrinsic() {
        // smulh x0, x1, x2: no 128-bit multiply in the IR, so the high
        // half is an intrinsic with exact read/write sets.
        let s = lift_word(0x9B42_7C20);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(matches!(
            s.as_slice(),
            [Stmt::Intrinsic { name: "a64.smulh", writes, reads }]
                if *writes == vec![cell(0)] && reads.len() == 2
        ));
        // umulh xzr, x4, x5: the discarded destination writes nothing.
        let s = lift_word(0x9BC5_7C9F);
        assert!(matches!(
            s.as_slice(),
            [Stmt::Intrinsic { name: "a64.umulh", writes, .. }] if writes.is_empty()
        ));
    }

    // ---- add/subtract with carry ----

    #[test]
    fn adc_folds_the_carry_in_to_the_sum() {
        // adc x0, x1, x2.
        assert_eq!(text(0x9A02_0020), "x0 := ((x1 + x2) + zext.q(CF))");
        // adc w3, w4, w5: computed at W32, widened into the cell.
        assert_eq!(
            text(0x1A05_0083),
            "x3 := zext.q(((trunc.d(x4) + trunc.d(x5)) + zext.d(CF)))"
        );
        // sbc x9, x10, x11 is rn + NOT(rm) + C, the Arm ARM identity.
        assert_eq!(text(0xDA0B_0149), "x9 := ((x10 + ~(x11)) + zext.q(CF))");
        // ngc x0, x1 (sbc from zr): the zero folds into the expression.
        assert_eq!(text(0xDA01_03E0), "x0 := ((0x0.q + ~(x1)) + zext.q(CF))");
    }

    #[test]
    fn adcs_snapshots_the_carry_and_writes_the_extended_model() {
        // adcs x6, x7, x8: CF is read into a temp before being
        // overwritten, and the carry-out gains the wraparound term.
        assert_eq!(
            text(0xBA08_00E6),
            "t0.i1 := CF\n\
             t1.q := x7\n\
             t2.q := x8\n\
             t3.q := ((t1.q + t2.q) + zext.q(t0.i1))\n\
             ZF := (t3.q == 0x0.q)\n\
             SF := (t3.q <s 0x0.q)\n\
             CF := ((t3.q <u t1.q) | (t0.i1 & (t3.q == t1.q)))\n\
             OF := (((t1.q ^ t3.q) & (t2.q ^ t3.q)) <s 0x0.q)\n\
             x6 := t3.q"
        );
        // sbcs w12, w13, w14: the complemented operand through the same
        // add half, at W32.
        assert_eq!(
            text(0x7A0E_01AC),
            "t0.i1 := CF\n\
             t1.d := trunc.d(x13)\n\
             t2.d := ~(trunc.d(x14))\n\
             t3.d := ((t1.d + t2.d) + zext.d(t0.i1))\n\
             ZF := (t3.d == 0x0.d)\n\
             SF := (t3.d <s 0x0.d)\n\
             CF := ((t3.d <u t1.d) | (t0.i1 & (t3.d == t1.d)))\n\
             OF := (((t1.d ^ t3.d) & (t2.d ^ t3.d)) <s 0x0.d)\n\
             x12 := zext.q(t3.d)"
        );
    }

    #[test]
    fn adcs_flags_are_the_shared_models_products() {
        // adcs x6, x7, x8: every flag assignment's expression must be
        // verbatim what nzcv_model emits for its temporaries — the one
        // model extended once, never re-derived.
        let s = lift_word(0xBA08_00E6);
        let tc = ir::Reg::temp(0, Width::W1);
        let tl = ir::Reg::temp(1, Width::W64);
        let tr = ir::Reg::temp(2, Width::W64);
        let tres = ir::Reg::temp(3, Width::W64);
        for (f, e) in nzcv_model(tl, tr, tres, Width::W64, false, Some(tc)) {
            assert!(
                s.contains(&assign(flag(f), e)),
                "{f:?} write is not the model's product"
            );
        }
        // And the model's N, Z, and V are untouched by the extension:
        // those three lines match ADDS's model output exactly.
        let plain = nzcv_model(tl, tr, tres, Width::W64, false, None);
        let extended = nzcv_model(tl, tr, tres, Width::W64, false, Some(tc));
        assert_eq!(plain[0], extended[0]);
        assert_eq!(plain[1], extended[1]);
        assert_eq!(plain[3], extended[3]);
    }

    // ---- conditional compare ----

    #[test]
    fn ccmp_selects_between_the_compare_and_the_literal_nzcv() {
        // ccmp x1, x2, #0, eq: each flag cell selects between the
        // compare's flag expression and its imm4 bit under the snapshotted
        // condition.
        assert_eq!(
            text(0xFA42_0020),
            "t0.i1 := ZF\n\
             t1.q := x1\n\
             t2.q := x2\n\
             t3.q := (t1.q - t2.q)\n\
             ZF := ((t0.i1 & (t3.q == 0x0.q)) | (~(t0.i1) & 0x0.i1))\n\
             SF := ((t0.i1 & (t3.q <s 0x0.q)) | (~(t0.i1) & 0x0.i1))\n\
             CF := ((t0.i1 & (t2.q <=u t1.q)) | (~(t0.i1) & 0x0.i1))\n\
             OF := ((t0.i1 & (((t1.q ^ t2.q) & (t1.q ^ t3.q)) <s 0x0.q)) | (~(t0.i1) & 0x0.i1))"
        );
        assert!(!writes_any_arch(&lift_word(0xFA42_0020)));
        // ccmn x3, #0, #1, mi: the add compare, an immediate operand, and
        // the V bit of the literal nzcv set.
        assert_eq!(
            text(0xBA40_4861),
            "t0.i1 := SF\n\
             t1.q := x3\n\
             t2.q := 0x0.q\n\
             t3.q := (t1.q + t2.q)\n\
             ZF := ((t0.i1 & (t3.q == 0x0.q)) | (~(t0.i1) & 0x0.i1))\n\
             SF := ((t0.i1 & (t3.q <s 0x0.q)) | (~(t0.i1) & 0x0.i1))\n\
             CF := ((t0.i1 & (t3.q <u t1.q)) | (~(t0.i1) & 0x0.i1))\n\
             OF := ((t0.i1 & (((t1.q ^ t3.q) & (t2.q ^ t3.q)) <s 0x0.q)) | (~(t0.i1) & 0x1.i1))"
        );
        // The W32 form computes the compare at W32, and nzcv = 4 sets
        // exactly the Z bit of the literal half.
        let t = text(0x7A44_1064); // ccmp w3, w4, #4, ne
        assert!(t.contains("t3.d := (t1.d - t2.d)"), "{t}");
        assert!(t.contains("ZF := ((t0.i1 & (t3.d == 0x0.d)) | (~(t0.i1) & 0x1.i1))"), "{t}");
        assert!(t.contains("CF := ((t0.i1 & (t2.d <=u t1.d)) | (~(t0.i1) & 0x0.i1))"), "{t}");
    }

    #[test]
    fn ccmp_flag_cells_select_over_the_shared_models_products() {
        // ccmp x1, x2, #0, eq: inside every select, the true arm is
        // verbatim nzcv_model's subtract expression for its temporaries.
        let s = lift_word(0xFA42_0020);
        let tc = ir::Reg::temp(0, Width::W1);
        let tl = ir::Reg::temp(1, Width::W64);
        let tr = ir::Reg::temp(2, Width::W64);
        let tres = ir::Reg::temp(3, Width::W64);
        for (f, e) in nzcv_model(tl, tr, tres, Width::W64, true, None) {
            let sel = bin(
                BinOp::Or,
                bin(BinOp::And, rd(tc), e),
                bin(BinOp::And, not1(rd(tc)), k(0, Width::W1)),
            );
            assert!(
                s.contains(&assign(flag(f), sel)),
                "{f:?} cell is not a select over the model's product"
            );
        }
    }

    #[test]
    fn ccmp_then_b_cond_round_trips_the_condition() {
        // ccmp x1, x2, #0, eq ; b.lt — the selected flags and the
        // condition read meet in the same convention across the block.
        let a = decode(&0xFA42_0020u32.to_le_bytes(), VA).unwrap();
        let b = decode(&0x5400_010Bu32.to_le_bytes(), VA + 4).unwrap();
        let block = lift_block(&[(a, VA), (b, VA + 4)]);
        assert_eq!(ir::check(&block), Ok(()));
        let t = ir::render(&block, &reg_name);
        assert!(t.contains("goto if (SF != OF)"), "{t}");
    }

    #[test]
    fn chained_condition_block_survives_ssa_and_forwarding() {
        use crate::model::Arch;
        use crate::{irlift, irssa, irssaopt};

        let block = |start: u64, words: &[u32], successors: Vec<u64>| {
            let insns: Vec<_> = words
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    let va = start + 4 * i as u64;
                    (decode(&w.to_le_bytes(), va).unwrap(), va)
                })
                .collect();
            irlift::LiftedBlock {
                start,
                end: start + 4 * words.len() as u64,
                stmts: lift_block(&insns),
                successors,
                truncated: false,
            }
        };
        // cmp w0, w1 ; b.eq → ccmp w2, w3, #0, ne ; b.lt → mov w0, #1 →
        // ret: the chained `&&` shape the conditional compare exists for,
        // through construct → optimize → forward.
        let build = || irlift::LiftedFunction {
            entry: 0x1000,
            name: None,
            arch: Arch::Aarch64,
            blocks: [
                block(0x1000, &[0x6B01_001F, 0x5400_0060], vec![0x1008, 0x1010]),
                block(0x1008, &[0x7A43_1040, 0x5400_006B], vec![0x1010, 0x1018]),
                block(0x1010, &[0x5280_0020, 0x1400_0001], vec![0x1018]),
                block(0x1018, &[0xD65F_03C0], vec![]),
            ]
            .into_iter()
            .map(|b| (b.start, b))
            .collect(),
        };
        let pipeline = || {
            let ssa = irssa::construct(&build()).expect("chained-condition block constructs");
            assert_eq!(irssa::check(&ssa), Ok(()));
            let (opt, _) = irssaopt::optimize(&ssa);
            assert_eq!(irssa::check(&opt), Ok(()));
            let (fwd, _) = irssaopt::forward(&opt);
            assert_eq!(irssa::check(&fwd), Ok(()));
            irssa::render(&fwd)
        };
        let t = pipeline();
        // Deterministic end to end.
        assert_eq!(t, pipeline());
        // Both guards survive as conditional branches. Observed on the
        // forwarded SSA (recorded, not promised): the first compare's
        // condition folds to the relational `w0 == w1`; the ccmp-fed
        // branch keeps its condition-masked flag pair — the
        // `(c & SF) != (c & OF)` shape (its `~c & 0` false arms fold
        // away) — and collapsing that to `(w0 != w1) & (w2 <s w3)` is
        // irflow pattern territory, not this lift's.
        assert_eq!(t.matches("goto if").count(), 2, "{t}");
    }

    // ---- flag definitions and condition uses agree ----

    #[test]
    fn subs_then_b_hs_use_the_same_no_borrow_carry() {
        // subs x0, x1, #1 ; b.hs — the flag written and the flag read must
        // encode the same convention (C = NOT borrow).
        let a = decode(&0xF100_0420u32.to_le_bytes(), VA).unwrap();
        let b = decode(&0x5400_0102u32.to_le_bytes(), VA + 4).unwrap();
        let block = lift_block(&[(a, VA), (b, VA + 4)]);
        assert_eq!(ir::check(&block), Ok(()));
        let t = ir::render(&block, &reg_name);
        assert!(t.contains("CF := (t1.q <=u t0.q)"), "{t}");
        assert!(t.contains("goto if CF"), "{t}");
    }

    // ---- intrinsics ----

    #[test]
    fn svc_is_an_honest_intrinsic() {
        let s = lift_word(0xD400_0001); // svc #0
        assert_eq!(ir::check(&s), Ok(()));
        assert!(matches!(
            s.as_slice(),
            [Stmt::Intrinsic { name: "svc", writes, reads }]
                if *writes == vec![cell(0)] && reads.len() == 8
        ));
        assert_eq!(
            ir::render(&s, &reg_name).trim_end(),
            "x0 := svc(0x0.q, x8, x0, x1, x2, x3, x4, x5)"
        );
        // hvc #1 and smc #2 share the shape.
        assert!(matches!(
            lift_word(0xD400_0022).as_slice(),
            [Stmt::Intrinsic { name: "hvc", .. }]
        ));
        assert!(matches!(
            lift_word(0xD400_0043).as_slice(),
            [Stmt::Intrinsic { name: "smc", .. }]
        ));
    }

    #[test]
    fn brk_and_hlt_are_empty_set_intrinsics() {
        assert_eq!(
            lift_word(0xD43E_0000).as_slice(), // brk #0xf000
            [Stmt::Intrinsic {
                name: "brk",
                writes: vec![],
                reads: vec![],
            }]
        );
        assert_eq!(
            lift_word(0xD440_0000).as_slice(), // hlt #0
            [Stmt::Intrinsic {
                name: "hlt",
                writes: vec![],
                reads: vec![],
            }]
        );
    }

    #[test]
    fn nop_and_unallocated_hints_lift_to_nothing() {
        assert!(lift_word(0xD503_201F).is_empty()); // nop
        assert!(lift_word(0xD503_20FF).is_empty()); // hint #7
    }

    #[test]
    fn event_hints_are_visible_intrinsics() {
        for (w, name) in [
            (0xD503_203Fu32, "yield"),
            (0xD503_205F, "wfe"),
            (0xD503_207F, "wfi"),
            (0xD503_209F, "sev"),
            (0xD503_20BF, "sevl"),
        ] {
            let s = lift_word(w);
            assert_eq!(ir::check(&s), Ok(()));
            assert!(
                matches!(s.as_slice(), [Stmt::Intrinsic { name: n, .. }] if *n == name),
                "{w:#010x} should lift to the {name} intrinsic"
            );
        }
    }

    #[test]
    fn unknown_clobbers_every_cell_and_flag() {
        // An LSE atomic (ldaddal — a documented gap) takes the
        // clobber-everything fallback, carrying its raw word.
        let s = lift_word(0xF8E9_0108);
        assert_eq!(ir::check(&s), Ok(()));
        let [Stmt::Intrinsic { name, writes, reads }] = s.as_slice() else {
            panic!("expected a single intrinsic");
        };
        assert_eq!(*name, "a64.unknown");
        // 32 GPR/SP cells + 64 SIMD&FP half cells + 4 NZCV flags.
        assert_eq!(writes.len(), 100);
        assert!(writes.contains(&vlo(0)) && writes.contains(&vhi(31)));
        assert_eq!(reads.as_slice(), [k(0xF8E9_0108, Width::W64)]);
    }

    #[test]
    fn simd_three_same_alu_lifts_exact_bitwise_and_precise_add() {
        // orr v1.16b, v1.16b, v1.16b — exact Or over both halves.
        let s = lift_word(0x4EA1_1C21);
        assert_eq!(ir::check(&s), Ok(()));
        assert_eq!(
            s.as_slice(),
            [
                assign(vlo(1), bin(BinOp::Or, rd(vlo(1)), rd(vlo(1)))),
                assign(vhi(1), bin(BinOp::Or, rd(vhi(1)), rd(vhi(1)))),
            ]
        );
        // and v0.8b, v1.8b, v2.8b — Q = 0 zeroes the high half.
        let s = lift_word(0x0E22_1C20);
        assert_eq!(ir::check(&s), Ok(()));
        assert_eq!(
            s.as_slice(),
            [
                assign(vlo(0), bin(BinOp::And, rd(vlo(1)), rd(vlo(2)))),
                assign(vhi(0), k(0, Width::W64)),
            ]
        );
        // eor v0.16b, v1.16b, v2.16b
        let s = lift_word(0x6E22_1C20);
        assert!(matches!(
            s.as_slice(),
            [
                Stmt::Assign { dst, .. },
                Stmt::Assign { .. }
            ] if *dst == vlo(0)
        ));
        assert!(s.iter().any(|st| matches!(
            st,
            Stmt::Assign { value: Expr::Binary { op: BinOp::Xor, .. }, .. }
        )));
        // sub v0.2d, v1.2d, v2.2d — exact Sub on each half.
        let s = lift_word(0x6EE2_8420);
        assert_eq!(ir::check(&s), Ok(()));
        assert_eq!(
            s.as_slice(),
            [
                assign(vlo(0), bin(BinOp::Sub, rd(vlo(1)), rd(vlo(2)))),
                assign(vhi(0), bin(BinOp::Sub, rd(vhi(1)), rd(vhi(2)))),
            ]
        );
        // add v0.4s, v1.4s, v2.4s — packed; precise intrinsic.
        let s = lift_word(0x4EA2_8420);
        assert_eq!(ir::check(&s), Ok(()));
        let [Stmt::Intrinsic { name, writes, reads }] = s.as_slice() else {
            panic!("expected a64.vadd intrinsic, got {s:?}");
        };
        assert_eq!(*name, "a64.vadd");
        assert_eq!(writes.as_slice(), [vlo(0), vhi(0)]);
        assert_eq!(
            reads.as_slice(),
            [rd(vlo(1)), rd(vlo(2)), rd(vhi(1)), rd(vhi(2))]
        );
    }

    // ---- scalar FP arithmetic: precise intrinsics, never the clobber ----

    #[test]
    fn fp_two_source_writes_exactly_the_low_cell_and_zeroes_the_high() {
        // fmul d0, d1, d2.
        let s = lift_word(0x1E62_0820);
        assert_eq!(ir::check(&s), Ok(()));
        let [Stmt::Intrinsic { name, writes, reads }, hi] = s.as_slice() else {
            panic!("intrinsic + high-cell zero, got {s:?}");
        };
        assert_eq!(*name, "a64.fmul");
        assert_eq!(writes.as_slice(), [vlo(0)]);
        assert_eq!(reads.as_slice(), [rd(vlo(1)), rd(vlo(2))]);
        assert_eq!(*hi, assign(vhi(0), k(0, Width::W64)));
        // The single form reads truncations, never whole cells.
        let s = lift_word(0x1E28_28E6); // fadd s6, s7, s8
        let [Stmt::Intrinsic { name, reads, .. }, _] = s.as_slice() else {
            panic!();
        };
        assert_eq!(*name, "a64.fadd");
        assert_eq!(
            reads.as_slice(),
            [
                un(UnOp::Truncate(Width::W32), rd(vlo(7))),
                un(UnOp::Truncate(Width::W32), rd(vlo(8)))
            ]
        );
    }

    #[test]
    fn fp_three_source_reads_all_three_operands() {
        // fnmsub s12, s13, s14, s15.
        let s = lift_word(0x1F2E_BDAC);
        assert_eq!(ir::check(&s), Ok(()));
        let [Stmt::Intrinsic { name, writes, reads }, _] = s.as_slice() else {
            panic!();
        };
        assert_eq!(*name, "a64.fnmsub");
        assert_eq!(writes.as_slice(), [vlo(12)]);
        assert_eq!(reads.len(), 3);
    }

    #[test]
    fn fabs_and_fneg_are_exact_sign_bit_masks() {
        // fabs d1, d2: the low cell masked, the high cell zeroed —
        // plain assignments, no intrinsic at all.
        let s = lift_word(0x1E60_C041);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(!s.iter().any(|st| matches!(st, Stmt::Intrinsic { .. })));
        assert_eq!(
            s[0],
            assign(
                vlo(1),
                bin(BinOp::And, rd(vlo(2)), k(0x7FFF_FFFF_FFFF_FFFF, Width::W64))
            )
        );
        assert_eq!(s[1], assign(vhi(1), k(0, Width::W64)));
        // fneg s3, s4: the 32-bit sign bit flipped, zero-extended.
        let s = lift_word(0x1E21_4083);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(!s.iter().any(|st| matches!(st, Stmt::Intrinsic { .. })));
        assert_eq!(
            s[0],
            assign(
                vlo(3),
                un(
                    UnOp::ZeroExtend(Width::W64),
                    bin(
                        BinOp::Xor,
                        un(UnOp::Truncate(Width::W32), rd(vlo(4))),
                        k(0x8000_0000, Width::W32)
                    )
                )
            )
        );
    }

    #[test]
    fn fcmp_writes_exactly_the_four_flags() {
        // fcmp d0, d1.
        let s = lift_word(0x1E61_2000);
        assert_eq!(ir::check(&s), Ok(()));
        let [Stmt::Intrinsic { name, writes, reads }] = s.as_slice() else {
            panic!();
        };
        assert_eq!(*name, "a64.fcmp");
        assert_eq!(
            writes.as_slice(),
            [
                flag(Flag::Sign),
                flag(Flag::Zero),
                flag(Flag::Carry),
                flag(Flag::Overflow)
            ]
        );
        assert_eq!(reads.as_slice(), [rd(vlo(0)), rd(vlo(1))]);
        assert!(!writes_any_arch(&s), "no register may be touched");
        // The zero form reads a literal zero comparand; FCMPE keeps its
        // own name.
        let s = lift_word(0x1E20_20B8); // fcmpe s5, #0.0
        let [Stmt::Intrinsic { name, reads, .. }] = s.as_slice() else {
            panic!();
        };
        assert_eq!(*name, "a64.fcmpe");
        assert_eq!(reads[1], k(0, Width::W32));
        // fccmp reads the condition first, so the incoming flags stay
        // live into the select.
        let s = lift_word(0x1E61_1404); // fccmp d0, d1, #0x4, ne
        let [Stmt::Intrinsic { name, reads, .. }] = s.as_slice() else {
            panic!();
        };
        assert_eq!(*name, "a64.fccmp");
        assert_eq!(reads.len(), 4);
        assert_eq!(ir::check(&s), Ok(()));
    }

    #[test]
    fn fcsel_is_the_branchless_select_over_the_low_cells() {
        // fcsel d3, d4, d5, pl: no intrinsic — the csel merge.
        let s = lift_word(0x1E65_5C83);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(!s.iter().any(|st| matches!(st, Stmt::Intrinsic { .. })));
        let last_two = &s[s.len() - 2..];
        assert!(matches!(
            last_two[0],
            Stmt::Assign { dst, .. } if dst == vlo(3)
        ));
        assert_eq!(last_two[1], assign(vhi(3), k(0, Width::W64)));
    }

    #[test]
    fn conversions_read_and_write_the_exact_cells() {
        // scvtf d2, w3: writes d2 (and zeroes the high half), reads w3.
        let s = lift_word(0x1E62_0062);
        assert_eq!(ir::check(&s), Ok(()));
        let [Stmt::Intrinsic { name, writes, reads }, hi] = s.as_slice() else {
            panic!();
        };
        assert_eq!(*name, "a64.scvtf");
        assert_eq!(writes.as_slice(), [vlo(2)]);
        assert_eq!(reads[0], un(UnOp::Truncate(Width::W32), rd(cell(3))));
        assert_eq!(*hi, assign(vhi(2), k(0, Width::W64)));
        // fcvtzs x6, d7: writes the GPR cell, reads d7.
        let s = lift_word(0x9E78_00E6);
        assert_eq!(ir::check(&s), Ok(()));
        let [Stmt::Intrinsic { name, writes, reads }] = s.as_slice() else {
            panic!();
        };
        assert_eq!(*name, "a64.fcvtzs");
        assert_eq!(writes.as_slice(), [cell(6)]);
        assert_eq!(reads[0], rd(vlo(7)));
        // A convert to the zero register defines nothing but keeps its
        // reads.
        let s = lift_word(0x9E78_00FF); // fcvtzs xzr, d7
        assert_eq!(ir::check(&s), Ok(()));
        let [Stmt::Intrinsic { writes, reads, .. }] = s.as_slice() else {
            panic!();
        };
        assert!(writes.is_empty());
        assert!(!reads.is_empty());
    }

    // ---- element moves: exact bit manipulation ----

    #[test]
    fn dup_general_replicates_by_doubling() {
        // dup v1.4s, w9: mask to 32 bits, one doubling, both halves
        // written with the same lane.
        let s = lift_word(0x4E04_0D21);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(!s.iter().any(|st| matches!(st, Stmt::Intrinsic { .. })));
        let n = s.len();
        assert!(matches!(s[n - 2], Stmt::Assign { dst, .. } if dst == vlo(1)));
        assert!(matches!(s[n - 1], Stmt::Assign { dst, .. } if dst == vhi(1)));
        // The 64-bit (non-q) form zeroes the high half instead.
        let s = lift_word(0x0E04_0D84); // dup v4.2s, w12
        assert_eq!(ir::check(&s), Ok(()));
        assert_eq!(s[s.len() - 1], assign(vhi(4), k(0, Width::W64)));
    }

    #[test]
    fn element_extractions_are_shifts_and_masks() {
        // mov d0, v1.d[1]: the scalar dup of the high half — exactly
        // two assignments.
        let s = lift_word(0x5E18_0420);
        assert_eq!(ir::check(&s), Ok(()));
        assert_eq!(
            s.as_slice(),
            [
                assign(vlo(0), rd(vhi(1))),
                assign(vhi(0), k(0, Width::W64))
            ]
        );
        // umov w0, v1.s[1]: bits 63:32 of the low cell, truncated and
        // zero-extended into x0.
        let s = lift_word(0x0E0C_3C20);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(writes_cell(&s, 0));
        // smov x4, v5.s[3]: the high cell's top word, sign-extended.
        let s = lift_word(0x4E1C_2CA4);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(writes_cell(&s, 4));
        let Stmt::Assign { value, .. } = &s[0] else {
            panic!();
        };
        assert!(
            format!("{:?}", value).contains("SignExtend"),
            "smov must sign-extend: {value:?}"
        );
    }

    #[test]
    fn ins_preserves_the_untouched_half_and_the_other_elements() {
        // ins v2.s[3], w3: element 3 lives in the high cell; the low
        // cell is never written, and the high cell is masked then
        // merged.
        let s = lift_word(0x4E1C_1C62);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(
            !s.iter().any(
                |st| matches!(st, Stmt::Assign { dst, .. } if *dst == vlo(2))
            ),
            "the low half must be preserved: {s:?}"
        );
        // ins v0.d[1], x1 writes the high cell alone, whole.
        let s = lift_word(0x4E18_1C20);
        assert_eq!(ir::check(&s), Ok(()));
        assert_eq!(s.as_slice(), [assign(vhi(0), rd(cell(1)))]);
        // ins v0.d[0], v1.d[1] is a plain cross-half copy.
        let s = lift_word(0x6E08_4420);
        assert_eq!(ir::check(&s), Ok(()));
        assert_eq!(s.as_slice(), [assign(vlo(0), rd(vhi(1)))]);
    }

    // ---- exclusives, PAC, UDF, and the integer row ----

    #[test]
    fn exclusive_and_ordered_accesses_are_plain_accesses() {
        // ldar x0, [x1] is a plain load; ldxr/ldaxr identical.
        for w in [0xC8DF_FC20u32, 0xC85F_7C20, 0xC85F_FD28] {
            let s = lift_word(w);
            assert_eq!(ir::check(&s), Ok(()), "{w:#010x}");
            assert!(
                s.iter().any(|st| matches!(
                    st,
                    Stmt::Assign { value: Expr::Load { .. }, .. }
                )),
                "{w:#010x} must load"
            );
            assert!(!s.iter().any(|st| matches!(st, Stmt::Intrinsic { .. })));
        }
        // stlr x8, [x9] is a plain store.
        let s = lift_word(0xC89F_FD28);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(s.iter().any(|st| matches!(st, Stmt::Store { .. })));
        assert!(!s.iter().any(|st| matches!(st, Stmt::Intrinsic { .. })));
        // stxr w0, x1, [x2]: the store plus the opaque status def.
        let s = lift_word(0xC800_7C41);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(s.iter().any(|st| matches!(st, Stmt::Store { .. })));
        assert!(s.iter().any(|st| matches!(
            st,
            Stmt::Intrinsic { name: "a64.stxr", writes, .. } if writes.as_slice() == [cell(0)]
        )));
        // A status write to wzr is discarded: store only.
        let s = lift_word(0xC81F_7C41); // stxr wzr, x1, [x2]
        assert_eq!(ir::check(&s), Ok(()));
        assert!(!s.iter().any(|st| matches!(st, Stmt::Intrinsic { .. })));
    }

    #[test]
    fn pac_rewrites_its_one_register_and_auth_branches_transfer() {
        // paciasp: x30 := pac(x30, sp).
        let s = lift_word(0xD503_233F);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(matches!(
            s.as_slice(),
            [Stmt::Intrinsic { name: "a64.pac", writes, reads }]
                if writes.as_slice() == [cell(30)] && reads.len() == 2
        ));
        // paciza x8: the zero-modifier form reads a literal zero.
        let s = lift_word(0xDAC1_23E8);
        let [Stmt::Intrinsic { writes, reads, .. }] = s.as_slice() else {
            panic!();
        };
        assert_eq!(writes.as_slice(), [cell(8)]);
        assert_eq!(reads[1], k(0, Width::W64));
        // retab returns through x30 exactly like ret.
        assert_eq!(lift_word(0xD65F_0FFF), lift_word(0xD65F_03C0));
        // blraa x6, x7 snapshots the target, links, calls.
        let s = lift_word(0xD73F_08C7);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(matches!(
            s.last(),
            Some(Stmt::Branch { kind: ir::BranchKind::Call, .. })
        ));
        assert!(writes_cell(&s, 30));
        // braaz x4 jumps.
        let s = lift_word(0xD61F_089F);
        assert!(matches!(
            s.last(),
            Some(Stmt::Branch { kind: ir::BranchKind::Jump, .. })
        ));
    }

    #[test]
    fn udf_lifts_like_a_trap() {
        let s = lift_word(0x0000_0000);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(matches!(
            s.as_slice(),
            [Stmt::Intrinsic { name: "udf", writes, reads }]
                if writes.is_empty() && reads.is_empty()
        ));
    }

    #[test]
    fn bits1_extr_and_ldpsw_lift_exactly() {
        // clz w0, w1: one named intrinsic on the exact cells.
        let s = lift_word(0x5AC0_1020);
        assert_eq!(ir::check(&s), Ok(()));
        assert!(matches!(
            s.as_slice(),
            [Stmt::Intrinsic { name: "a64.clz", writes, .. }]
                if writes.as_slice() == [cell(0)]
        ));
        // rbit to the zero register is nothing at all.
        assert!(lift_word(0x5AC0_013F).is_empty());
        // extr x3, x4, x5, #33: (x5 >>u 33) | (x4 << 31).
        let s = lift_word(0x93C5_8483);
        assert_eq!(ir::check(&s), Ok(()));
        assert_eq!(
            s.as_slice(),
            [assign(
                cell(3),
                bin(
                    BinOp::Or,
                    bin(BinOp::LShr, rd(cell(5)), k(33, Width::W64)),
                    bin(BinOp::Shl, rd(cell(4)), k(31, Width::W64))
                )
            )]
        );
        // ror x8, x9, #0 degenerates to the source itself — no shift
        // by the full width ever appears.
        let s = lift_word(0x93C9_0128);
        assert_eq!(ir::check(&s), Ok(()));
        assert_eq!(s.as_slice(), [assign(cell(8), rd(cell(9)))]);
        // ldpsw x0, x1, [x2, #8]: two word loads, each sign-extended.
        let s = lift_word(0x6941_0440);
        assert_eq!(ir::check(&s), Ok(()));
        let loads = s
            .iter()
            .filter(|st| matches!(st, Stmt::Assign { value: Expr::Load { .. }, .. }))
            .count();
        assert_eq!(loads, 2);
        assert!(writes_cell(&s, 0) && writes_cell(&s, 1));
        assert!(
            format!("{s:?}").contains("SignExtend"),
            "ldpsw sign-extends"
        );
    }

    // ---- robustness and determinism ----

    /// A broad corpus of real-shaped encodings; every lift must check.
    const CORPUS: &[u32] = &[
        // add/sub immediate, aliases, flag forms.
        0x9100_0420, 0x1100_0420, 0x9140_0462, 0x9100_03E0, 0x9100_001F,
        0xD100_43FF, 0xB100_0820, 0xF100_0420, 0x7100_1420, 0xF100_045F,
        0xB100_101F, 0x7100_001F,
        // move wide.
        0xD282_4680, 0x5280_00A1, 0x9280_0002, 0x1280_0000, 0xF2B7_DDE0,
        0x7280_0020, 0xD280_003F,
        // pc-relative addresses.
        0x1000_0080, 0x10FF_FFE1, 0xB000_0001, 0xF0FF_FFE2,
        // loads/stores, immediate.
        0xF940_0BE0, 0xF940_0420, 0xB940_0022, 0xB940_001F, 0xB900_03E2,
        0xB900_001F, 0x3940_0483, 0x3900_0041, 0x7940_0462, 0xF840_8400,
        0xF85F_8C20, 0xF801_07E0,
        // sign-extending loads.
        0x38C0_1420, 0x39C0_BFE8, 0x3980_0020, 0x79C0_0420, 0xB980_0420,
        // register offset.
        0xF822_6820, 0xF862_7820, 0xF862_5820, 0xB8A2_C820, 0x3869_690A,
        // literals and pairs.
        0x5800_0040, 0x1800_0041, 0xA8C1_7BFD, 0xA9BF_7BFD, 0x2940_0440,
        0xA940_0440,
        // branches.
        0x1400_0004, 0x17FF_FFFC, 0x9400_0004, 0x5400_0100, 0x5400_0102,
        0x5400_0103, 0x5400_010E, 0x3400_0040, 0xB5FF_FFC5, 0xB400_005F,
        0xB608_0083, 0x372F_FFC2, 0xD61F_0200, 0xD63F_0100, 0xD63F_03C0,
        0xD65F_03C0, 0xD65F_0020,
        // conditional select.
        0x9A82_0020, 0x9A88_B2B4, 0x1A9F_07E0, 0xDA82_B020, 0xDA85_C483,
        0x1A82_0420,
        // shifted-register add/sub and aliases.
        0x8B02_0020, 0x0B02_0C20, 0xAB42_1020, 0xEB02_0020, 0x4B86_1CA4,
        0xEB02_003F, 0xAB04_007F, 0xCB02_03E0, 0x6B02_07E0,
        // logical shifted register and aliases.
        0x8A02_0020, 0x0A22_0820, 0xAA42_2020, 0xAA22_0020, 0xCAC5_3083,
        0x4A23_0041, 0xEA02_0020, 0xEAA2_0C20, 0xEA02_003F, 0xAA02_03E0,
        0x2A21_03E0,
        // extended-register add/sub.
        0x8B21_4BE0, 0x8B21_63FF, 0x8B25_8483, 0xCB22_73FF, 0xEB22_2820,
        0xEB21_63FF, 0xAB23_C05F,
        // logical immediate.
        0x9240_1C20, 0x1200_1820, 0xB204_C462, 0x5203_C8A4, 0xF204_CCE6,
        0xF27F_043F, 0xB201_F3E0, 0x3200_C3E1, 0x927C_EC3F,
        // bitfield moves.
        0xD378_DC20, 0x531F_7862, 0xD350_FCA4, 0x131F_7CE6, 0x937F_FD28,
        0xD348_5C20, 0x531C_1C62, 0x5300_1CA4, 0x9344_3D28, 0x937E_116A,
        0x9340_1C62, 0x9340_7CE6, 0xB378_3C20, 0x3304_2C62, 0xB370_1FE4,
        // variable shifts and divides.
        0x9AC2_2020, 0x1AC5_2483, 0x9AC8_28E6, 0x9ACB_2D49, 0x9AC2_0820,
        0x1AC5_0883, 0x9AC8_0CE6,
        // multiplies.
        0x9B02_0C20, 0x1B06_9CA4, 0x9B0A_7D28, 0x9B0D_FD8B, 0x9B22_0C20,
        0x9BAA_2D28, 0x9B22_7C20, 0x9BA5_7C83, 0x9B42_7C20, 0x9BC5_7C83,
        // add/sub with carry and the ngc aliases.
        0x9A02_0020, 0x1A05_0083, 0xBA08_00E6, 0x3A02_0020, 0xDA0B_0149,
        0x5A03_0041, 0xFA02_0020, 0x7A0E_01AC, 0xDA01_03E0, 0x7A03_03E2,
        // conditional compare, register and immediate.
        0xFA42_0020, 0x7A44_1064, 0xBA46_B0A8, 0x3A48_20EF, 0xFA5F_2820,
        0x7A45_C84F, 0xBA40_4861, 0x3A4C_D923,
        // unscaled loads and stores.
        0xF85F_F020, 0xB84F_F3E2, 0x3850_0083, 0x7840_70C5, 0xF840_0107,
        0xF81F_8107, 0xB800_3020, 0x3800_1149, 0x7800_3041, 0x389F_D18B,
        0x38C0_5020, 0x78C0_51CD, 0x789F_9062, 0xB880_220F,
        // exceptions and hints.
        0xD400_0001, 0xD400_0022, 0xD400_0043, 0xD43E_0000, 0xD440_0000,
        0xD503_201F, 0xD503_203F, 0xD503_205F, 0xD503_207F, 0xD503_209F,
        0xD503_20BF, 0xD503_20FF,
        // SIMD&FP loads/stores, all five sizes and every addressing mode.
        0x3D40_0420, 0x7D40_0462, 0xBD40_07E4, 0xFD40_04E6, 0x3DC0_0528,
        0x3D00_0441, 0x7D00_0483, 0xBD00_03E5, 0xFD00_0D07, 0x3D80_0949,
        0x3DFF_FC1F, 0xFC5F_8C20, 0x3CC1_0441, 0xBC1F_CC62, 0x3C82_07E3,
        0x3C40_14A4, 0x7C00_2CC5, 0xFC5F_F020, 0x3CDF_0062, 0xBC00_30A4,
        0x3C1F_70E6, 0x7C40_5041, 0x3C9E_03E7, 0xFC62_6820, 0x3CE3_7841,
        0xBC65_5882, 0xFC27_D8C3, 0x3C29_6904, 0x3C69_7905, 0x7C61_D806,
        0x3CA3_E85E, 0x2D40_0440, 0x6D41_0C82, 0xAD41_17E4, 0x6CC1_1D06,
        0x6DBF_27E8, 0xAD00_0460, 0x2CBF_0C82, 0xADBE_7C1E, 0x1C00_0000,
        0x5CFF_FFE1, 0x9CFF_FFC2,
        // SIMD&FP moves: fmov register/general/immediate, movi/mvni.
        0x1E20_4020, 0x1E60_4062, 0x1E26_0020, 0x9E66_0062, 0x1E27_00A4,
        0x9E67_00E6, 0x9EAE_0128, 0x9EAF_016A, 0x9E66_007F, 0x1E27_03E8,
        0x1E2E_1000, 0x1E7C_1001, 0x1E67_F002, 0x1E28_1003, 0x0F03_E7E0,
        0x4F04_E401, 0x0F00_8642, 0x4F00_A643, 0x0F01_0684, 0x4F01_6685,
        0x0F02_C6C6, 0x4F02_D6C7, 0x2F05_E548, 0x6F07_E7E9, 0x6F00_E41F,
        0x2F00_842A, 0x6F00_444B, 0x2F00_C46C, 0x6F02_A48D, 0x0F03_F60E,
        0x4F04_F48F, 0x6F02_F610,
        // scalar FP arithmetic: two-source (every op, both precisions),
        // three-source, one-source with FRINT, precision converts.
        0x1E22_0820, 0x1E25_1883, 0x1E28_28E6, 0x1E2B_3949, 0x1E2E_49AC,
        0x1E31_5A0F, 0x1E34_6A72, 0x1E37_7AD5, 0x1E3A_8B38, 0x1E62_0820,
        0x1E65_1883, 0x1E68_28E6, 0x1E6B_3949, 0x1E6E_49AC, 0x1E71_5A0F,
        0x1E74_6A72, 0x1E77_7AD5, 0x1E7A_8B38, 0x1F02_0C20, 0x1F06_9CA4,
        0x1F2A_2D28, 0x1F2E_BDAC, 0x1F42_0C20, 0x1F46_9CA4, 0x1F6A_2D28,
        0x1F6E_BDAC, 0x1E20_C041, 0x1E21_4083, 0x1E21_C0C5, 0x1E60_C041,
        0x1E61_4083, 0x1E61_C0C5, 0x1E22_C107, 0x1E62_4149, 0x1E24_4020,
        0x1E24_C062, 0x1E25_40A4, 0x1E25_C0E6, 0x1E26_4128, 0x1E27_416A,
        0x1E27_C1AC, 0x1E64_4020, 0x1E64_C062, 0x1E65_40A4, 0x1E65_C0E6,
        0x1E66_4128, 0x1E67_416A, 0x1E67_C1AC,
        // FP compares, conditional compares, selects.
        0x1E21_2000, 0x1E20_2048, 0x1E24_2070, 0x1E20_20B8, 0x1E61_2000,
        0x1E60_2048, 0x1E64_2070, 0x1E60_20B8, 0x1E21_1404, 0x1E23_A45F,
        0x1E61_1404, 0x1E63_A45F, 0x1E22_4C20, 0x1E65_5C83,
        // conversions, GPR both directions plus scalar-integer.
        0x1E22_0020, 0x1E62_0062, 0x9E22_00A4, 0x9E62_00E6, 0x1E23_0128,
        0x1E63_016A, 0x9E23_01AC, 0x9E63_01EE, 0x1E38_0020, 0x9E38_0062,
        0x1E78_00A4, 0x9E78_00E6, 0x1E39_0128, 0x9E39_016A, 0x1E79_01AC,
        0x9E79_01EE, 0x1E20_0020, 0x1E21_0062, 0x1E68_00A4, 0x9E69_00E6,
        0x1E30_0128, 0x1E71_016A, 0x9E24_01AC, 0x1E65_01EE, 0x5E21_D820,
        0x5E61_D862, 0x7E21_D8A4, 0x7E61_D8E6,
        // element moves: dup/mov/umov/smov/ins in every arrangement.
        0x4E08_0D00, 0x4E04_0D21, 0x4E02_0D42, 0x4E01_0D63, 0x0E04_0D84,
        0x0E01_0DA5, 0x0E02_0DC6, 0x5E18_0420, 0x5E14_0462, 0x5E0E_04A4,
        0x5E13_04E6, 0x4E08_0420, 0x4E0C_0462, 0x0E0C_3C20, 0x0E12_3C62,
        0x0E17_3CA4, 0x4E18_3CE6, 0x0E0A_2C20, 0x0E0B_2C62, 0x4E1C_2CA4,
        0x4E1E_2CE6, 0x4E1F_2D28, 0x4E18_1C20, 0x4E1C_1C62, 0x4E16_1CA4,
        0x4E1B_1CE6, 0x6E08_4420, 0x6E0C_4462, 0x6E0A_64A4, 0x6E07_64E6,
        // exclusives and ordered accesses, every size.
        0xC8DF_FC20, 0x88DF_FC62, 0x08DF_FCA4, 0x48DF_FCE6, 0xC89F_FD28,
        0x889F_FD6A, 0x089F_FDAC, 0x489F_FDEE, 0xC85F_7C20, 0x885F_7C62,
        0x085F_7CA4, 0x485F_7CE6, 0xC85F_FD28, 0x885F_FD6A, 0x085F_FDAC,
        0x485F_FDEE, 0xC800_7C41, 0x8803_7CA4, 0x0806_7D07, 0x4809_7D6A,
        0xC80C_FDCD, 0x880F_FE30, 0x0812_FE93, 0x4815_FEF6,
        // PAC: returns, branches, the dp-1source row, the hints. UDF.
        0xD65F_0BFF, 0xD65F_0FFF, 0xD71F_0801, 0xD71F_0C43, 0xD61F_089F,
        0xD61F_0CBF, 0xD73F_08C7, 0xD73F_0D09, 0xD63F_095F, 0xD63F_0D7F,
        0xDAC1_0020, 0xDAC1_0462, 0xDAC1_10A4, 0xDAC1_14E6, 0xDAC1_23E8,
        0xDAC1_27E9, 0xDAC1_33EA, 0xDAC1_37EB, 0xDAC1_43EC, 0xDAC1_47ED,
        0xD503_233F, 0xD503_23BF, 0xD503_237F, 0xD503_23FF, 0x0000_0000,
        0x0000_04B8,
        // the one-source bit row, extr/ror, ldpsw.
        0x5AC0_1020, 0xDAC0_1062, 0x5AC0_14A4, 0xDAC0_14E6, 0x5AC0_0128,
        0xDAC0_016A, 0x5AC0_09AC, 0xDAC0_0DEE, 0x5AC0_0630, 0xDAC0_0672,
        0xDAC0_0AB4, 0x1382_1420, 0x93C5_8483, 0x1387_24E6, 0x93C9_8128,
        0x6941_0440, 0x69FF_90A3, 0x68C2_1D06,
        // three-same integer ALU (modeled) plus unmodeled LSE / SVE.
        0x4EA1_1C21, 0x4EA2_8420, 0x6E22_1C20, 0xF8E9_0108, 0x0420_0000,
    ];

    #[test]
    fn every_corpus_lift_is_well_formed() {
        for &w in CORPUS {
            ok(w);
        }
    }

    #[test]
    fn lifting_is_deterministic() {
        for &w in CORPUS {
            let insn = decode(&w.to_le_bytes(), VA).unwrap();
            assert_eq!(lift(&insn, VA), lift(&insn, VA), "{w:#010x}");
        }
    }

    #[test]
    fn lift_block_threads_temporaries_so_they_stay_unique() {
        // ldp x29, x30, [sp], #16 ; ldr x0, [x0], #8 — both instructions
        // use temporaries; across the block they must not collide.
        let a = decode(&0xA8C1_7BFDu32.to_le_bytes(), VA).unwrap();
        let b = decode(&0xF840_8400u32.to_le_bytes(), VA + 4).unwrap();
        let block = lift_block(&[(a, VA), (b, VA + 4)]);
        assert_eq!(ir::check(&block), Ok(()));
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

    /// Decode-lift-check one word; the sweeps' shared assertion.
    fn sweep_one(w: u32) {
        let insn = decode(&w.to_le_bytes(), VA).expect("4-byte decode is total");
        let stmts = lift(&insn, VA);
        assert_eq!(ir::check(&stmts), Ok(()), "check failed for {w:#010x}");
    }

    #[test]
    fn sweep_every_high_half_with_corner_low_halves() {
        // All 65 536 high halves — bits 25–28 select every encoding class —
        // crossed with boundary low halves (all-zero/all-one register and
        // immediate fields, r31 in every position).
        const LOWS: [u32; 8] = [
            0x0000, 0xFFFF, 0x03FF, 0x7C1F, 0xFFE0, 0x001F, 0x8000, 0x5555,
        ];
        for hi in 0..=0xFFFFu32 {
            for lo in LOWS {
                sweep_one((hi << 16) | lo);
            }
        }
    }

    #[test]
    fn sweep_every_low_half_with_representative_high_halves() {
        // One representative high half per decoder dispatch shape, so every
        // low half exercises full register/immediate fields of each family.
        const HIGHS: [u32; 22] = [
            0x9100, // add/sub immediate (64)
            0xF100, // subs immediate
            0x1280, // move wide (movn w)
            0xF2A0, // movk
            0x9000, // adrp
            0x5400, // b.cond
            0x3400, // cbz
            0xB6F8, // tbz, high bit numbers
            0xD400, // exception generation
            0xD503, // hints and system space
            0xD61F, // br/blr/ret and pointer-auth neighbors
            0xF940, // ldr unsigned offset (64)
            0xF840, // ldr pre/post/register-offset
            0xA8C1, // ldp post-index
            0x5800, // ldr literal
            0x9A82, // conditional select
            0x8B02, // add shifted register (64)
            0x8B21, // add extended register
            0xEA22, // logical shifted register (bics)
            0x9240, // logical immediate
            0xD350, // bitfield (ubfm 64)
            0x9AC2, // two-source and three-source neighborhoods
        ];
        for hi in HIGHS {
            for lo in 0..=0xFFFFu32 {
                sweep_one((hi << 16) | lo);
            }
        }
    }

    #[test]
    fn sweep_a_seeded_random_sample_of_the_word_space() {
        // xorshift64* with a fixed seed: deterministic, no wall clock.
        let mut s = 0x0DDB_1A5E_5BAD_5EEDu64;
        for _ in 0..1_000_000 {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let w = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32;
            sweep_one(w);
        }
    }
}
