//! AArch64 (A64) instruction decoder.
//!
//! Clean-room implementation from the public Arm Architecture Reference
//! Manual A64 instruction-encoding tables. A64 is a fixed-width ISA —
//! every instruction is exactly one 32-bit little-endian word — so the
//! decoder is total: any 4-byte input decodes to *something*, and only a
//! short input is an error.
//!
//! Coverage is deliberately tiered, per the roadmap ("branch/call/return
//! classification first, full operand decoding second"):
//!
//! - **Exact control flow**: `B`, `BL`, `B.<cond>`, `CBZ`/`CBNZ`,
//!   `TBZ`/`TBNZ`, `BR`/`BLR`/`RET`, and the exception generators
//!   `SVC`/`HVC`/`SMC`/`BRK`/`HLT` — each with its [`Flow`] effect and,
//!   for pc-relative forms, the absolute target VA.
//! - **Listing subset**: `ADR`/`ADRP` (page arithmetic matters for
//!   xrefs), `ADD`/`SUB` immediate (including the `ADRP`+`ADD` idiom),
//!   `MOVZ`/`MOVN`/`MOVK`, `LDR`/`STR` immediate (unsigned offset and
//!   pre/post-index) and the sign-extending `LDRS{B,H,W}`, the unscaled
//!   `LDUR`/`STUR`/`LDURS{B,H,W}` forms, the register-offset load/store
//!   forms, `LDR` literal (pc-relative, target classified), `LDP`/`STP`,
//!   the conditional-select family (`CSEL`/`CSINC`/`CSINV`/`CSNEG`), and
//!   `NOP` plus the common hints.
//! - **Integer data-processing**: shifted-register `ADD`/`SUB` and the
//!   logical group (`AND`/`ORR`/`EOR` with the `BIC`/`ORN`/`EON` and
//!   flag-setting forms), extended-register `ADD`/`SUB`, the logical
//!   (bitmask) immediates, the bitfield trio `SBFM`/`BFM`/`UBFM`
//!   (decoded canonically, alias-spelled at render), the two-source
//!   shifts and divides, the three-source multiplies (`MADD`/`MSUB`,
//!   the widening `L` forms, `SMULH`/`UMULH`), add/subtract with carry
//!   (`ADC`/`SBC` and the flag-setting forms), and the conditional
//!   compares `CCMP`/`CCMN` (immediate and register).
//! - **SIMD&FP loads/stores and moves**: `LDR`/`STR` of the b/h/s/d/q
//!   register file in every addressing mode the integer forms decode
//!   (unsigned offset, pre/post-index, unscaled `LDUR`/`STUR`, register
//!   offset, literal), `LDP`/`STP` for s/d/q, `FMOV` (register,
//!   register↔general including the `Vn.D[1]` lane, scalar and vector
//!   immediate), and `MOVI`/`MVNI`.
//! - **Scalar FP arithmetic**: the two-source group (`FMUL`/`FDIV`/
//!   `FADD`/`FSUB`/`FMAX`/`FMIN`/`FMAXNM`/`FMINNM`/`FNMUL`), the
//!   three-source multiplies (`FMADD`/`FMSUB`/`FNMADD`/`FNMSUB`),
//!   one-source (`FABS`/`FNEG`/`FSQRT`, `FCVT` s↔d, the seven
//!   `FRINT<r>`), `FCMP`/`FCMPE`, `FCCMP`/`FCCMPE`, `FCSEL`, and the
//!   conversions (`SCVTF`/`UCVTF` from GPR and scalar-integer,
//!   `FCVT{N,P,M,Z,A}{S,U}` to GPR) — single and double precision;
//!   half precision (FEAT_FP16) stays refused throughout.
//! - **Element moves**: `DUP` (general, element→vector, and the scalar
//!   `mov d0, v1.d[1]` form), `INS` (general and element),
//!   `UMOV`/`SMOV`.
//! - **Advanced SIMD three-same ALU**: integer `ADD`/`SUB`/`AND`/`ORR`/
//!   `EOR`, vector `FADD`/`FMUL`, and `CMEQ`/`CMHI` (byte through
//!   double for integer ADD/SUB/compares; logical trio always on
//!   `.8b`/`.16b`; FADD/FMUL on `.2s`/`.4s`/`.2d`).
//! - **Exclusives / ordered**: `LDAR`/`STLR`, `LDXR`/`LDAXR`,
//!   `STXR`/`STLXR`, every size.
//! - **Pointer authentication and UDF**: `RETAA`/`RETAB`, the
//!   `BRAA`/`BLRAA` family, the dp-1source I-key row (`PACIA`...,
//!   `XPACI`/`XPACD`), the four PAC hints, and `UDF` #imm16 — a real
//!   [`Flow::Halt`] terminator, so the arm64e inter-function zero-word
//!   padding stops a sweep instead of leaking it into data. The
//!   one-source bit group (`RBIT`/`REV*`/`CLZ`/`CLS`), `EXTR` (with
//!   the `ror` alias), and `LDPSW` ride along.
//! - **Everything else** decodes to [`Opcode::Unknown`] with
//!   [`Flow::Sequential`]. On a fixed-width ISA this is safe and honest:
//!   an unmodeled word can never desynchronize the instruction stream —
//!   it is reported as undecoded rather than guessed at.
//!
//! Known gaps (all decode as `Unknown` / `Sequential`): the remaining
//! Advanced SIMD vector ALU (two-reg-misc, shift-immediate, across-lanes,
//! permutes, the BIC/ORN/BSL siblings), the structure
//! loads/stores (`LD1`/`ST1`/...), LSE atomics (`LDADD`/`SWP`/`CAS`...),
//! the D-key PAC data ops, `ERET`/`DRPS`, half precision, `PRFM`, the
//! unprivileged `LDTR`/`STTR` forms, fixed-point converts (`FCVTZS` with
//! a scale), and CRC32.

use crate::error::{ParseError, Result};
use std::fmt;

/// Condition code from the `cond` field of `B.<cond>` (bits \[3:0\]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    /// Equal (`Z == 1`).
    Eq,
    /// Not equal (`Z == 0`).
    Ne,
    /// Carry set / unsigned higher-or-same (`C == 1`).
    Cs,
    /// Carry clear / unsigned lower (`C == 0`).
    Cc,
    /// Minus / negative (`N == 1`).
    Mi,
    /// Plus / positive-or-zero (`N == 0`).
    Pl,
    /// Overflow set (`V == 1`).
    Vs,
    /// Overflow clear (`V == 0`).
    Vc,
    /// Unsigned higher (`C == 1 && Z == 0`).
    Hi,
    /// Unsigned lower-or-same (`C == 0 || Z == 1`).
    Ls,
    /// Signed greater-or-equal (`N == V`).
    Ge,
    /// Signed less-than (`N != V`).
    Lt,
    /// Signed greater-than (`Z == 0 && N == V`).
    Gt,
    /// Signed less-or-equal (`Z == 1 || N != V`).
    Le,
    /// Always.
    Al,
    /// Always (the second "always" encoding, 0b1111).
    Nv,
}

impl From<u8> for Cond {
    fn from(raw: u8) -> Self {
        match raw & 0xF {
            0 => Cond::Eq,
            1 => Cond::Ne,
            2 => Cond::Cs,
            3 => Cond::Cc,
            4 => Cond::Mi,
            5 => Cond::Pl,
            6 => Cond::Vs,
            7 => Cond::Vc,
            8 => Cond::Hi,
            9 => Cond::Ls,
            10 => Cond::Ge,
            11 => Cond::Lt,
            12 => Cond::Gt,
            13 => Cond::Le,
            14 => Cond::Al,
            _ => Cond::Nv,
        }
    }
}

impl Cond {
    /// Standard assembler suffix (`eq`, `ne`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Cond::Eq => "eq",
            Cond::Ne => "ne",
            Cond::Cs => "cs",
            Cond::Cc => "cc",
            Cond::Mi => "mi",
            Cond::Pl => "pl",
            Cond::Vs => "vs",
            Cond::Vc => "vc",
            Cond::Hi => "hi",
            Cond::Ls => "ls",
            Cond::Ge => "ge",
            Cond::Lt => "lt",
            Cond::Gt => "gt",
            Cond::Le => "le",
            Cond::Al => "al",
            Cond::Nv => "nv",
        }
    }

    /// The inverse condition, flipping the encoding's low bit (`eq`↔`ne`,
    /// `ge`↔`lt`, ...). Used by the `cset`/`cinc`/... select aliases, which
    /// name the *false*-arm condition.
    pub fn invert(self) -> Cond {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Cs => Cond::Cc,
            Cond::Cc => Cond::Cs,
            Cond::Mi => Cond::Pl,
            Cond::Pl => Cond::Mi,
            Cond::Vs => Cond::Vc,
            Cond::Vc => Cond::Vs,
            Cond::Hi => Cond::Ls,
            Cond::Ls => Cond::Hi,
            Cond::Ge => Cond::Lt,
            Cond::Lt => Cond::Ge,
            Cond::Gt => Cond::Le,
            Cond::Le => Cond::Gt,
            Cond::Al => Cond::Nv,
            Cond::Nv => Cond::Al,
        }
    }

    /// Whether this is the "always" pair (`al`/`nv`), for which the
    /// condition-inverting select aliases are not defined.
    pub fn is_al_nv(self) -> bool {
        matches!(self, Cond::Al | Cond::Nv)
    }
}

/// Addressing mode of a load/store with an immediate.
///
/// The offset is stored fully scaled and sign-extended (bytes, not units),
/// so `Offset(16)` always means "base + 16 bytes".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddrMode {
    /// `[Xn, #imm]`: base plus immediate, no writeback.
    Offset(i64),
    /// `[Xn, #imm]!`: pre-indexed, base updated before the access.
    PreIndex(i64),
    /// `[Xn], #imm`: post-indexed, base updated after the access.
    PostIndex(i64),
}

/// The index operand of a register-offset load/store:
/// `[<Xn|SP>, <R><m>{, <extend> {#amount}}]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegOffset {
    /// Index register number (`Rm`).
    pub rm: u8,
    /// Extend/shift option (bits \[15:13\]): `010` UXTW, `011` LSL/UXTX,
    /// `110` SXTW, `111` SXTX. Its low bit also selects the index width —
    /// set means `Xm`, clear means `Wm`.
    pub option: u8,
    /// The `S` bit: when set, the index is scaled left by the access size.
    pub scaled: bool,
}

/// Shift kind of a shifted-register operand (bits \[23:22\] of the
/// add/sub and logical shifted-register groups, and the two-source
/// variable-shift opcodes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shift {
    /// Logical shift left.
    Lsl,
    /// Logical (zero-filling) shift right.
    Lsr,
    /// Arithmetic (sign-filling) shift right.
    Asr,
    /// Rotate right. Allocated for the logical group and `RORV` only;
    /// reserved on add/sub (shifted register).
    Ror,
}

impl Shift {
    /// Standard assembler spelling (`lsl`, `lsr`, `asr`, `ror`).
    pub fn as_str(self) -> &'static str {
        match self {
            Shift::Lsl => "lsl",
            Shift::Lsr => "lsr",
            Shift::Asr => "asr",
            Shift::Ror => "ror",
        }
    }
}

/// The operation selector shared by the logical (shifted register) and
/// logical (immediate) groups. The flag-setting (`ANDS`/`BICS`) and
/// operand-inverting (`BIC`/`ORN`/`EON`) axes are carried separately on
/// the opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogOp {
    /// Bitwise AND.
    And,
    /// Bitwise inclusive OR.
    Orr,
    /// Bitwise exclusive OR.
    Eor,
}

/// The scalar FP two-source operation (bits \[15:12\] of the FP
/// data-processing two-source group, in encoding order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F2Op {
    Mul,
    Div,
    Add,
    Sub,
    Max,
    Min,
    MaxNm,
    MinNm,
    NMul,
}

impl F2Op {
    /// Assembler mnemonic.
    pub fn as_str(self) -> &'static str {
        match self {
            F2Op::Mul => "fmul",
            F2Op::Div => "fdiv",
            F2Op::Add => "fadd",
            F2Op::Sub => "fsub",
            F2Op::Max => "fmax",
            F2Op::Min => "fmin",
            F2Op::MaxNm => "fmaxnm",
            F2Op::MinNm => "fminnm",
            F2Op::NMul => "fnmul",
        }
    }
}

/// Advanced SIMD three-same ALU (public integer ADD/SUB/AND/ORR/EOR,
/// vector FADD/FMUL, and compares CMEQ/CMHI). Element size is log2
/// bytes for integer ADD/SUB/compares; the logical trio always operates
/// on bytes (`.8b`/`.16b` by `Q`); FADD/FMUL use size 2 (S) or 3 (D).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdAluOp {
    Add,
    Sub,
    And,
    Orr,
    Eor,
    Fadd,
    Fmul,
    Cmeq,
    Cmhi,
}

impl SimdAluOp {
    /// Assembler mnemonic.
    pub fn as_str(self) -> &'static str {
        match self {
            SimdAluOp::Add => "add",
            SimdAluOp::Sub => "sub",
            SimdAluOp::And => "and",
            SimdAluOp::Orr => "orr",
            SimdAluOp::Eor => "eor",
            SimdAluOp::Fadd => "fadd",
            SimdAluOp::Fmul => "fmul",
            SimdAluOp::Cmeq => "cmeq",
            SimdAluOp::Cmhi => "cmhi",
        }
    }
}

/// The scalar FP one-source operation. The `FRINT<r>` roundings share
/// [`FpRound`] plus the two exact-and-inexact extras (`X`, `I`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F1Op {
    Abs,
    Neg,
    Sqrt,
    /// `FRINTN`/`FRINTP`/`FRINTM`/`FRINTZ`/`FRINTA`.
    Rint(FpRound),
    /// `FRINTX` (exact, signalling).
    RintX,
    /// `FRINTI` (current rounding mode).
    RintI,
}

impl F1Op {
    /// Assembler mnemonic.
    pub fn as_str(self) -> &'static str {
        match self {
            F1Op::Abs => "fabs",
            F1Op::Neg => "fneg",
            F1Op::Sqrt => "fsqrt",
            F1Op::Rint(FpRound::N) => "frintn",
            F1Op::Rint(FpRound::P) => "frintp",
            F1Op::Rint(FpRound::M) => "frintm",
            F1Op::Rint(FpRound::Z) => "frintz",
            F1Op::Rint(FpRound::A) => "frinta",
            F1Op::RintX => "frintx",
            F1Op::RintI => "frinti",
        }
    }
}

/// An FP rounding direction: to-nearest-ties-even (`N`), toward plus
/// infinity (`P`), toward minus infinity (`M`), toward zero (`Z`),
/// to-nearest-ties-away (`A`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpRound {
    N,
    P,
    M,
    Z,
    A,
}

impl FpRound {
    /// The mnemonic letter (`fcvtns`'s `n`, ...).
    pub fn letter(self) -> char {
        match self {
            FpRound::N => 'n',
            FpRound::P => 'p',
            FpRound::M => 'm',
            FpRound::Z => 'z',
            FpRound::A => 'a',
        }
    }
}

/// The data-processing one-source bit operation (opcode bits \[15:10\]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bit1Op {
    Rbit,
    Rev16,
    Rev32,
    Rev,
    Clz,
    Cls,
}

impl Bit1Op {
    /// Assembler mnemonic.
    pub fn as_str(self) -> &'static str {
        match self {
            Bit1Op::Rbit => "rbit",
            Bit1Op::Rev16 => "rev16",
            Bit1Op::Rev32 => "rev32",
            Bit1Op::Rev => "rev",
            Bit1Op::Clz => "clz",
            Bit1Op::Cls => "cls",
        }
    }
}

/// Control-flow effect of one instruction: the shared
/// [`crate::model::Flow`], re-exported so `aarch64::Flow` stays a valid
/// path.
///
/// Branch targets are absolute virtual addresses computed from the
/// instruction's own VA.
pub use crate::model::Flow;

/// A decoded A64 instruction.
///
/// Only the fields needed for control-flow analysis and a readable listing
/// are modeled; the encodings outside that subset are preserved verbatim
/// in [`Opcode::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    /// `B <label>`: unconditional pc-relative branch (imm26).
    B { target: u64 },
    /// `BL <label>`: branch with link; X30 receives the return address.
    Bl { target: u64 },
    /// `B.<cond> <label>`: conditional branch (imm19).
    BCond { cond: Cond, target: u64 },
    /// `CBZ <Rt>, <label>`: compare register and branch if zero.
    Cbz { sf: bool, rt: u8, target: u64 },
    /// `CBNZ <Rt>, <label>`: compare register and branch if nonzero.
    Cbnz { sf: bool, rt: u8, target: u64 },
    /// `TBZ <Rt>, #<bit>, <label>`: test bit and branch if zero.
    Tbz { rt: u8, bit: u8, target: u64 },
    /// `TBNZ <Rt>, #<bit>, <label>`: test bit and branch if nonzero.
    Tbnz { rt: u8, bit: u8, target: u64 },
    /// `BR <Xn>`: unconditional branch to a register.
    Br { rn: u8 },
    /// `BLR <Xn>`: call through a register.
    Blr { rn: u8 },
    /// `RET {<Xn>}`: return; `rn` is 30 (X30/LR) in the common form.
    Ret { rn: u8 },
    /// `SVC #imm16`: supervisor call.
    Svc { imm: u16 },
    /// `HVC #imm16`: hypervisor call.
    Hvc { imm: u16 },
    /// `SMC #imm16`: secure monitor call.
    Smc { imm: u16 },
    /// `BRK #imm16`: breakpoint exception.
    Brk { imm: u16 },
    /// `HLT #imm16`: halting debug breakpoint.
    Hlt { imm: u16 },
    /// `ADR <Xd>, <label>`: pc-relative address (byte granularity).
    Adr { rd: u8, target: u64 },
    /// `ADRP <Xd>, <label>`: pc-relative address of a 4 KiB page.
    Adrp { rd: u8, target: u64 },
    /// `ADD{S} <Rd>, <Rn>, #imm`: `imm` is stored fully shifted
    /// (`imm12` or `imm12 << 12`).
    AddImm {
        sf: bool,
        set_flags: bool,
        rd: u8,
        rn: u8,
        imm: u32,
    },
    /// `SUB{S} <Rd>, <Rn>, #imm`: `imm` is stored fully shifted.
    SubImm {
        sf: bool,
        set_flags: bool,
        rd: u8,
        rn: u8,
        imm: u32,
    },
    /// `MOVN <Rd>, #imm16, LSL #shift`: move wide with NOT.
    Movn { sf: bool, rd: u8, imm: u16, shift: u8 },
    /// `MOVZ <Rd>, #imm16, LSL #shift`: move wide with zero.
    Movz { sf: bool, rd: u8, imm: u16, shift: u8 },
    /// `MOVK <Rd>, #imm16, LSL #shift`: move wide, keeping other bits.
    Movk { sf: bool, rd: u8, imm: u16, shift: u8 },
    /// `CSEL <Rd>, <Rn>, <Rm>, <cond>`: `Rd = cond ? Rn : Rm`.
    Csel {
        sf: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        cond: Cond,
    },
    /// `CSINC <Rd>, <Rn>, <Rm>, <cond>`: `Rd = cond ? Rn : Rm + 1`. Also
    /// spelled `CSET`/`CINC` by assemblers.
    Csinc {
        sf: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        cond: Cond,
    },
    /// `CSINV <Rd>, <Rn>, <Rm>, <cond>`: `Rd = cond ? Rn : NOT(Rm)`. Also
    /// spelled `CSETM`/`CINV`.
    Csinv {
        sf: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        cond: Cond,
    },
    /// `CSNEG <Rd>, <Rn>, <Rm>, <cond>`: `Rd = cond ? Rn : -Rm`. Also
    /// spelled `CNEG`.
    Csneg {
        sf: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        cond: Cond,
    },
    /// `CCMP`/`CCMN <Rn>, <Rm>, #<nzcv>, <cond>` (conditional compare,
    /// register): if `cond` holds, NZCV gets the flags of `Rn - Rm`
    /// (`CCMP`; `sub` set) or `Rn + Rm` (`CCMN`); otherwise the literal
    /// `nzcv` imm4 (`N:Z:C:V` from bit 3 down).
    CcmpReg {
        sf: bool,
        sub: bool,
        rn: u8,
        rm: u8,
        nzcv: u8,
        cond: Cond,
    },
    /// `CCMP`/`CCMN <Rn>, #<imm>, #<nzcv>, <cond>` (conditional compare,
    /// immediate): as [`Opcode::CcmpReg`] with the zero-extended 5-bit
    /// immediate as the second operand.
    CcmpImm {
        sf: bool,
        sub: bool,
        rn: u8,
        imm: u8,
        nzcv: u8,
        cond: Cond,
    },
    /// `ADC{S} <Rd>, <Rn>, <Rm>`: add with carry, `Rd = Rn + Rm + C`.
    /// All register-31 operands are the zero register.
    Adc {
        sf: bool,
        set_flags: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `SBC{S} <Rd>, <Rn>, <Rm>`: subtract with carry,
    /// `Rd = Rn - Rm - 1 + C` (the Arm ARM's `Rn + NOT(Rm) + C`).
    /// Aliased `NGC{S}` (rn = zr) at render time.
    Sbc {
        sf: bool,
        set_flags: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `ADD{S} <Rd>, <Rn>, <Rm>{, <shift> #amount}` (shifted register).
    /// All register-31 operands are the zero register. Aliased `CMN`
    /// (`set_flags`, rd = zr) at render time.
    AddReg {
        sf: bool,
        set_flags: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        shift: Shift,
        amount: u8,
    },
    /// `SUB{S} <Rd>, <Rn>, <Rm>{, <shift> #amount}` (shifted register).
    /// Aliased `CMP` (`set_flags`, rd = zr) and `NEG{S}` (rn = zr).
    SubReg {
        sf: bool,
        set_flags: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        shift: Shift,
        amount: u8,
    },
    /// `ADD{S} <Rd|SP>, <Rn|SP>, <Rm>, <extend> {#amount}` (extended
    /// register): `option` is the raw extend field (bits \[15:13\]: `000`
    /// UXTB ... `011` UXTX/LSL ... `111` SXTX; its low two bits select the
    /// index width — `11` means `Xm`, anything else `Wm` on the 64-bit
    /// form), `amount` the left shift (0-4). `rn` is SP; so is `rd` unless
    /// flags are set.
    AddExt {
        sf: bool,
        set_flags: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        option: u8,
        amount: u8,
    },
    /// `SUB{S}` (extended register); fields as [`Opcode::AddExt`].
    SubExt {
        sf: bool,
        set_flags: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        option: u8,
        amount: u8,
    },
    /// The logical (shifted register) family — `AND`/`ORR`/`EOR` by `op`,
    /// with `invert` (the `N` bit) complementing the shifted operand
    /// (`BIC`/`ORN`/`EON`/`BICS`) and `set_flags` only ever true for
    /// [`LogOp::And`] (`ANDS`/`BICS`). Aliased `TST`/`MOV`/`MVN` at render
    /// time.
    LogReg {
        sf: bool,
        op: LogOp,
        set_flags: bool,
        invert: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        shift: Shift,
        amount: u8,
    },
    /// `AND`/`ORR`/`EOR`/`ANDS <Rd>, <Rn>, #imm` (logical immediate):
    /// `imm` is the fully expanded bitmask (see [`decode_bit_masks`]),
    /// masked to 32 bits when `sf` is clear. `rd` is SP unless flags are
    /// set; `rn` is the zero register. Aliased `TST` (`set_flags`,
    /// rd = zr) and `MOV` ([`LogOp::Orr`], rn = zr).
    LogImm {
        sf: bool,
        op: LogOp,
        set_flags: bool,
        rd: u8,
        rn: u8,
        imm: u64,
    },
    /// `SBFM <Rd>, <Rn>, #immr, #imms`: signed bitfield move, the
    /// canonical form behind `ASR` (immediate), `SBFX`/`SBFIZ`, and
    /// `SXTB`/`SXTH`/`SXTW` — the alias spelling is display-only.
    Sbfm {
        sf: bool,
        rd: u8,
        rn: u8,
        immr: u8,
        imms: u8,
    },
    /// `BFM <Rd>, <Rn>, #immr, #imms`: bitfield move (insert into `rd`),
    /// the canonical form behind `BFI`/`BFXIL`/`BFC`.
    Bfm {
        sf: bool,
        rd: u8,
        rn: u8,
        immr: u8,
        imms: u8,
    },
    /// `UBFM <Rd>, <Rn>, #immr, #imms`: unsigned bitfield move, the
    /// canonical form behind `LSL`/`LSR` (immediate), `UBFX`/`UBFIZ`,
    /// and `UXTB`/`UXTH`.
    Ubfm {
        sf: bool,
        rd: u8,
        rn: u8,
        immr: u8,
        imms: u8,
    },
    /// `LSLV`/`LSRV`/`ASRV`/`RORV <Rd>, <Rn>, <Rm>`: variable shift by
    /// `Rm` modulo the register width; rendered with the preferred
    /// `lsl`/`lsr`/`asr`/`ror` spelling.
    ShiftReg {
        sf: bool,
        kind: Shift,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `UDIV <Rd>, <Rn>, <Rm>`: unsigned divide; division by zero yields
    /// zero (the architectural rule — no trap).
    Udiv { sf: bool, rd: u8, rn: u8, rm: u8 },
    /// `SDIV <Rd>, <Rn>, <Rm>`: signed divide; division by zero yields
    /// zero, and the `INT_MIN / -1` overflow wraps.
    Sdiv { sf: bool, rd: u8, rn: u8, rm: u8 },
    /// `MADD <Rd>, <Rn>, <Rm>, <Ra>`: `Rd = Ra + Rn * Rm`. Aliased `MUL`
    /// (ra = zr).
    Madd {
        sf: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    /// `MSUB <Rd>, <Rn>, <Rm>, <Ra>`: `Rd = Ra - Rn * Rm`. Aliased `MNEG`
    /// (ra = zr).
    Msub {
        sf: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    /// `SMADDL`/`SMSUBL`/`UMADDL`/`UMSUBL <Xd>, <Wn>, <Wm>, <Xa>`:
    /// widening 32x32+64 multiply-accumulate (64-bit form only). Aliased
    /// `SMULL`/`SMNEGL`/`UMULL`/`UMNEGL` (ra = zr).
    Maddl {
        signed: bool,
        sub: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    /// `SMULH`/`UMULH <Xd>, <Xn>, <Xm>`: the high 64 bits of the 128-bit
    /// product (64-bit form only).
    Mulh {
        signed: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `LDR{B,H} <Rt>, [<Xn|SP> ...]`: `size` is log2 of the access width
    /// in bytes (0 = byte ... 3 = doubleword).
    Ldr {
        size: u8,
        rt: u8,
        rn: u8,
        mode: AddrMode,
    },
    /// `STR{B,H} <Rt>, [<Xn|SP> ...]`: `size` as for [`Opcode::Ldr`].
    Str {
        size: u8,
        rt: u8,
        rn: u8,
        mode: AddrMode,
    },
    /// `LDRS{B,H,W} <Rt>, [<Xn|SP> ...]`: sign-extending load with an
    /// immediate. `size` is the log2 access width (0 = `ldrsb`, 1 = `ldrsh`,
    /// 2 = `ldrsw`); `sf` is the extended register width (`X` when set).
    Ldrs {
        size: u8,
        sf: bool,
        rt: u8,
        rn: u8,
        mode: AddrMode,
    },
    /// `LDUR{B,H} <Rt>, [<Xn|SP>{, #imm}]`: unscaled load — the same
    /// access as an offset-mode [`Opcode::Ldr`], addressed by a signed
    /// 9-bit byte offset with no writeback. `size` as for [`Opcode::Ldr`].
    Ldur { size: u8, rt: u8, rn: u8, imm: i64 },
    /// `STUR{B,H} <Rt>, [<Xn|SP>{, #imm}]`: unscaled store; fields as
    /// [`Opcode::Ldur`].
    Stur { size: u8, rt: u8, rn: u8, imm: i64 },
    /// `LDURS{B,H,W} <Rt>, [<Xn|SP>{, #imm}]`: unscaled sign-extending
    /// load; `size`/`sf` as for [`Opcode::Ldrs`].
    Ldurs {
        size: u8,
        sf: bool,
        rt: u8,
        rn: u8,
        imm: i64,
    },
    /// `LDR{B,H} <Rt>, [<Xn|SP>, <R><m>{, <extend>}]`: register-offset load;
    /// `size` as for [`Opcode::Ldr`].
    LdrReg {
        size: u8,
        rt: u8,
        rn: u8,
        off: RegOffset,
    },
    /// `STR{B,H} <Rt>, [<Xn|SP>, <R><m>{, <extend>}]`: register-offset store.
    StrReg {
        size: u8,
        rt: u8,
        rn: u8,
        off: RegOffset,
    },
    /// `LDRS{B,H,W} <Rt>, [<Xn|SP>, <R><m>{, <extend>}]`: register-offset
    /// sign-extending load; `size`/`sf` as for [`Opcode::Ldrs`].
    LdrsReg {
        size: u8,
        sf: bool,
        rt: u8,
        rn: u8,
        off: RegOffset,
    },
    /// `LDR <Rt>, <label>`: pc-relative literal load; `target` is the
    /// absolute VA of the literal (useful as a data xref).
    LdrLit { sf: bool, rt: u8, target: u64 },
    /// `LDP <Rt>, <Rt2>, [<Xn|SP> ...]`: load pair.
    Ldp {
        sf: bool,
        rt: u8,
        rt2: u8,
        rn: u8,
        mode: AddrMode,
    },
    /// `STP <Rt>, <Rt2>, [<Xn|SP> ...]`: store pair.
    Stp {
        sf: bool,
        rt: u8,
        rt2: u8,
        rn: u8,
        mode: AddrMode,
    },
    /// `LDR <Bt|Ht|St|Dt|Qt>, [<Xn|SP> ...]`: SIMD&FP register load with
    /// an immediate. `size` is log2 of the access width in bytes
    /// (0 = byte ... 4 = quadword); register 31 is `V31`, never ZR/SP.
    FLdr {
        size: u8,
        rt: u8,
        rn: u8,
        mode: AddrMode,
    },
    /// `STR <Bt|...|Qt>, [<Xn|SP> ...]`: SIMD&FP register store; fields
    /// as [`Opcode::FLdr`].
    FStr {
        size: u8,
        rt: u8,
        rn: u8,
        mode: AddrMode,
    },
    /// `LDUR <Bt|...|Qt>, [<Xn|SP>{, #imm}]`: unscaled SIMD&FP load —
    /// signed 9-bit byte offset, no writeback.
    FLdur { size: u8, rt: u8, rn: u8, imm: i64 },
    /// `STUR <Bt|...|Qt>, [<Xn|SP>{, #imm}]`: unscaled SIMD&FP store.
    FStur { size: u8, rt: u8, rn: u8, imm: i64 },
    /// `LDR <Bt|...|Qt>, [<Xn|SP>, <R><m>{, <extend>}]`: register-offset
    /// SIMD&FP load; a scaled index shifts by `size` (up to `#4` for q).
    FLdrReg {
        size: u8,
        rt: u8,
        rn: u8,
        off: RegOffset,
    },
    /// `STR <Bt|...|Qt>, [<Xn|SP>, <R><m>{, <extend>}]`: register-offset
    /// SIMD&FP store.
    FStrReg {
        size: u8,
        rt: u8,
        rn: u8,
        off: RegOffset,
    },
    /// `LDR <St|Dt|Qt>, <label>`: pc-relative SIMD&FP literal load;
    /// `size` is 2/3/4 (the b/h forms do not exist).
    FLdrLit { size: u8, rt: u8, target: u64 },
    /// `LDP <St|Dt|Qt>, ..., [<Xn|SP> ...]`: SIMD&FP load pair; `size`
    /// as [`Opcode::FLdrLit`].
    FLdp {
        size: u8,
        rt: u8,
        rt2: u8,
        rn: u8,
        mode: AddrMode,
    },
    /// `STP <St|Dt|Qt>, ..., [<Xn|SP> ...]`: SIMD&FP store pair.
    FStp {
        size: u8,
        rt: u8,
        rt2: u8,
        rn: u8,
        mode: AddrMode,
    },
    /// `FMOV <Sd|Dd>, <Sn|Dn>` (register): a scalar copy that zeroes the
    /// rest of the destination vector register. The half-precision form
    /// is a documented gap.
    FmovReg { double: bool, rd: u8, rn: u8 },
    /// `FMOV <Wd|Xd>, <Sn|Dn|Vn.D[1]>` (FP → general). `hi` selects the
    /// `Vn.D[1]` lane source (then `sf` is always set); `rd` is
    /// ZR-position.
    FmovToGp {
        sf: bool,
        hi: bool,
        rd: u8,
        rn: u8,
    },
    /// `FMOV <Sd|Dd|Vd.D[1]>, <Wn|Xn>` (general → FP). The `Vd.D[1]`
    /// lane write (`hi`) keeps the low half; the scalar forms zero the
    /// rest of the register. `rn` is ZR-position.
    FmovFromGp {
        sf: bool,
        hi: bool,
        rd: u8,
        rn: u8,
    },
    /// `FMOV <Sd|Dd>, #imm` (scalar immediate): `imm` is the raw imm8,
    /// expanded by [`fp_imm_value`].
    FmovImm { double: bool, imm: u8, rd: u8 },
    /// `FMOV <Vd>.<T>, #imm` (vector immediate): `.2s`/`.4s` by `q`, or
    /// `.2d` when `double` (then `q` is always set).
    FmovVecImm {
        q: bool,
        double: bool,
        imm: u8,
        rd: u8,
    },
    /// `MOVI`/`MVNI` (vector modified immediate). `size` is the element
    /// log2 width in bytes; `shift` the left shift in bits applied to
    /// `imm` (`msl` selects the shifting-ones form); `invert` is `MVNI`;
    /// `q` the 128-bit form. For `size == 3` the element is the
    /// byte-mask expansion of `imm` ([`movi_expand`]) and
    /// `shift`/`msl`/`invert` are always zero/false. The vector
    /// `ORR`/`BIC` immediates sharing this encoding group are
    /// read-modify-write, not moves, and stay [`Opcode::Unknown`].
    Movi {
        q: bool,
        invert: bool,
        size: u8,
        imm: u8,
        shift: u8,
        msl: bool,
        rd: u8,
    },
    /// Scalar FP two-source arithmetic (`FMUL`/`FDIV`/`FADD`/`FSUB`/
    /// `FMAX`/`FMIN`/`FMAXNM`/`FMINNM`/`FNMUL`), single or double
    /// precision. Half precision (FEAT_FP16) is the documented gap.
    FArith2 {
        op: F2Op,
        double: bool,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// Scalar FP three-source (`FMADD`/`FMSUB`/`FNMADD`/`FNMSUB`):
    /// `negate` is o1 (the `FN*` pair), `sub` is o0.
    FArith3 {
        negate: bool,
        sub: bool,
        double: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        ra: u8,
    },
    /// Scalar FP one-source (`FABS`/`FNEG`/`FSQRT` and the seven
    /// `FRINT<r>` roundings).
    FArith1 {
        op: F1Op,
        double: bool,
        rd: u8,
        rn: u8,
    },
    /// `FCVT <Sd>, <Dn>` / `FCVT <Dd>, <Sn>`: precision conversion.
    /// The half-precision rows stay [`Opcode::Unknown`].
    FCvtPrec { to_double: bool, rd: u8, rn: u8 },
    /// `FCMP`/`FCMPE` (`signal`), register or `#0.0` (`rm` = `None`).
    Fcmp {
        double: bool,
        signal: bool,
        rn: u8,
        rm: Option<u8>,
    },
    /// `FCCMP`/`FCCMPE`: NZCV from the compare when `cond` holds, the
    /// literal `nzcv` otherwise.
    Fccmp {
        double: bool,
        signal: bool,
        rn: u8,
        rm: u8,
        nzcv: u8,
        cond: Cond,
    },
    /// `FCSEL <d>, <n>, <m>, <cond>`.
    Fcsel {
        double: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        cond: Cond,
    },
    /// `SCVTF`/`UCVTF <Sd|Dd>, <Wn|Xn>`: integer (GPR) to FP. `rn` is
    /// ZR-position.
    FcvtToFp {
        sf: bool,
        double: bool,
        unsigned: bool,
        rd: u8,
        rn: u8,
    },
    /// `FCVT<r><s|u> <Wd|Xd>, <Sn|Dn>`: FP to integer (GPR) with an
    /// explicit rounding direction. `rd` is ZR-position.
    FcvtFromFp {
        sf: bool,
        double: bool,
        unsigned: bool,
        round: FpRound,
        rd: u8,
        rn: u8,
    },
    /// `SCVTF`/`UCVTF <Sd|Dd>, <Sn|Dn>` (scalar integer in a SIMD&FP
    /// register to FP of the same width).
    FcvtIntScalar {
        double: bool,
        unsigned: bool,
        rd: u8,
        rn: u8,
    },
    /// `DUP <Vd>.<T>, <Wn|Xn>` (general register replicated to every
    /// element). `size` is the element log2 width; `rn` is ZR-position.
    DupGp { q: bool, size: u8, rd: u8, rn: u8 },
    /// `DUP <V><d>, <Vn>.<T>[i]` — scalar from element, which `otool`
    /// spells `mov`.
    DupElemScalar {
        size: u8,
        index: u8,
        rd: u8,
        rn: u8,
    },
    /// `DUP <Vd>.<T>, <Vn>.<Ts>[i]` — element replicated to a vector.
    DupElemVec {
        q: bool,
        size: u8,
        index: u8,
        rd: u8,
        rn: u8,
    },
    /// `UMOV <Wd|Xd>, <Vn>.<Ts>[i]` (`sf` only for the `.d` element).
    Umov {
        sf: bool,
        size: u8,
        index: u8,
        rd: u8,
        rn: u8,
    },
    /// `SMOV <Wd|Xd>, <Vn>.<Ts>[i]`.
    Smov {
        sf: bool,
        size: u8,
        index: u8,
        rd: u8,
        rn: u8,
    },
    /// `INS <Vd>.<Ts>[i], <Wn|Xn>` (general to element; the other
    /// elements are preserved — the one write that is not
    /// whole-register). `rn` is ZR-position.
    InsGp {
        size: u8,
        index: u8,
        rd: u8,
        rn: u8,
    },
    /// `INS <Vd>.<Ts>[dst], <Vn>.<Ts>[src]` (element to element).
    InsElem {
        size: u8,
        dst: u8,
        src: u8,
        rd: u8,
        rn: u8,
    },
    /// Advanced SIMD three-same ALU: integer `ADD`/`SUB`/`AND`/`ORR`/`EOR`,
    /// vector `FADD`/`FMUL`, and `CMEQ`/`CMHI`. `size` is the element
    /// log2 width (2/3 for FADD/FMUL S/D); logical ops ignore it
    /// (always byte lanes). `q` selects 64- vs 128-bit vectors.
    SimdAlu {
        op: SimdAluOp,
        q: bool,
        size: u8,
        rd: u8,
        rn: u8,
        rm: u8,
    },
    /// `LDAR{,B,H} <Rt>, [<Xn|SP>]`: load-acquire. `size` is the access
    /// log2 width (0/1/2/3).
    Ldar { size: u8, rt: u8, rn: u8 },
    /// `STLR{,B,H} <Rt>, [<Xn|SP>]`: store-release.
    Stlr { size: u8, rt: u8, rn: u8 },
    /// `LDXR`/`LDAXR{,B,H}`: exclusive load (`acquire` selects LDAXR).
    Ldxr {
        size: u8,
        acquire: bool,
        rt: u8,
        rn: u8,
    },
    /// `STXR`/`STLXR{,B,H} <Ws>, <Rt>, [<Xn|SP>]`: exclusive store;
    /// `ws` receives the 0/1 status.
    Stxr {
        size: u8,
        release: bool,
        ws: u8,
        rt: u8,
        rn: u8,
    },
    /// `RETAA`/`RETAB`: return with pointer authentication of x30.
    RetA { key_b: bool },
    /// `BRAA`/`BRAB`/`BLRAA`/`BLRAB` and their `Z` forms: authenticated
    /// indirect branch/call. `rm` is the modifier register (SP-position
    /// for the non-Z forms; the Z forms use zero and encode `rm` = 31).
    BrAuth {
        link: bool,
        key_b: bool,
        zero: bool,
        rn: u8,
        rm: u8,
    },
    /// `PACIA`/`PACIB`/`AUTIA`/`AUTIB <Xd>, <Xn|SP>` and the `Z` forms
    /// (`PACIZA`, ... — `rn` = 31 meaning the zero modifier).
    PacGpr {
        auth: bool,
        key_b: bool,
        zero: bool,
        rd: u8,
        rn: u8,
    },
    /// `XPACI`/`XPACD <Xd>`: strip the authentication code.
    XPac { data: bool, rd: u8 },
    /// `PACIASP`/`AUTIASP`/`PACIBSP`/`AUTIBSP`: the hint-space PAC ops
    /// on x30 with SP as modifier — named (they are *not* NOPs on
    /// arm64e), unlike the unallocated hints.
    PacHint { auth: bool, key_b: bool },
    /// `UDF #imm16`: permanently undefined — traps, never falls
    /// through. /bin/ls-style arm64e images pad between functions with
    /// zero words, which are exactly `UDF #0`.
    Udf { imm: u16 },
    /// The data-processing one-source bit group: `RBIT`/`REV16`/`REV32`/
    /// `REV`/`CLZ`/`CLS`.
    Bits1 { op: Bit1Op, sf: bool, rd: u8, rn: u8 },
    /// `EXTR <Rd>, <Rn>, <Rm>, #lsb` (`ROR #imm` when `rn == rm`).
    Extr {
        sf: bool,
        rd: u8,
        rn: u8,
        rm: u8,
        lsb: u8,
    },
    /// `LDPSW <Xt1>, <Xt2>, [<Xn|SP> ...]`: load pair of sign-extended
    /// words.
    LdpSw {
        rt: u8,
        rt2: u8,
        rn: u8,
        mode: AddrMode,
    },
    /// `NOP`.
    Nop,
    /// `YIELD` hint.
    Yield,
    /// `WFE` (wait for event) hint.
    Wfe,
    /// `WFI` (wait for interrupt) hint.
    Wfi,
    /// `SEV` (send event) hint.
    Sev,
    /// `SEVL` (send event local) hint.
    Sevl,
    /// A hint (`CRm:op2` in `imm`) we recognize the space of but do not
    /// name; unallocated hints execute as `NOP` by definition.
    Hint { imm: u8 },
    /// An encoding this pass does not model, preserved verbatim.
    ///
    /// Always classified as [`Flow::Sequential`]: for a fixed-width ISA
    /// this can never desynchronize decoding, though the documented gap
    /// instructions that do branch (e.g. `BRAA`, `ERET`) are then
    /// under-approximated as sequential.
    Unknown(u32),
}

impl Opcode {
    /// The control-flow effect of this opcode.
    pub fn flow(self) -> Flow {
        match self {
            Opcode::B { target } => Flow::Jump(target),
            Opcode::Bl { target } => Flow::Call(target),
            Opcode::BCond { target, .. }
            | Opcode::Cbz { target, .. }
            | Opcode::Cbnz { target, .. }
            | Opcode::Tbz { target, .. }
            | Opcode::Tbnz { target, .. } => Flow::CondJump(target),
            Opcode::Br { .. } | Opcode::BrAuth { link: false, .. } => Flow::IndirectJump,
            Opcode::Blr { .. } | Opcode::BrAuth { link: true, .. } => Flow::IndirectCall,
            Opcode::Ret { .. } | Opcode::RetA { .. } => Flow::Return,
            // Permanently undefined: traps, never falls through — the
            // arm64e inter-function padding must not sweep onward.
            Opcode::Udf { .. } => Flow::Halt,
            Opcode::Svc { .. } | Opcode::Hvc { .. } | Opcode::Smc { .. } | Opcode::Brk { .. } => {
                Flow::Interrupt
            }
            Opcode::Hlt { .. } => Flow::Halt,
            _ => Flow::Sequential,
        }
    }
}

/// One decoded instruction word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction {
    /// The raw 32-bit instruction word.
    pub raw: u32,
    /// Virtual address the word was decoded at.
    pub va: u64,
    pub opcode: Opcode,
    /// Control-flow effect (cached from [`Opcode::flow`]).
    pub flow: Flow,
}

impl Instruction {
    /// Every A64 instruction is exactly this many bytes.
    pub const SIZE: usize = 4;
}

/// Decode one A64 instruction from the first 4 bytes of `bytes`, treating
/// it as located at virtual address `va`.
///
/// Extra trailing bytes are ignored; fewer than 4 bytes is
/// [`ParseError::UnexpectedEof`]. Decoding itself is total — every 32-bit
/// word yields an [`Instruction`] (possibly [`Opcode::Unknown`]) and never
/// panics.
pub fn decode(bytes: &[u8], va: u64) -> Result<Instruction> {
    if bytes.len() < Instruction::SIZE {
        return Err(ParseError::UnexpectedEof {
            offset: 0,
            needed: Instruction::SIZE,
            available: bytes.len(),
        });
    }
    let raw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let opcode = decode_word(raw, va);
    Ok(Instruction {
        raw,
        va,
        opcode,
        flow: opcode.flow(),
    })
}

/// Extract bits `[hi:lo]` of `w` (inclusive, `hi >= lo`).
fn bits(w: u32, hi: u32, lo: u32) -> u32 {
    ((w as u64 >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32
}

/// Extract bit `i` of `w` as a bool.
fn bit(w: u32, i: u32) -> bool {
    (w >> i) & 1 == 1
}

/// Sign-extend the low `width` bits of `value` to an i64.
fn sext(value: u32, width: u32) -> i64 {
    let shift = 64 - width;
    (((value as u64) << shift) as i64) >> shift
}

/// Absolute VA of a pc-relative branch: `va + sext(imm, width) * 4`,
/// wrapping at the edges of the 64-bit address space rather than panicking.
fn branch_target(va: u64, imm: u32, width: u32) -> u64 {
    va.wrapping_add((sext(imm, width) << 2) as u64)
}

/// Expand the `N:immr:imms` bitmask-immediate encoding of the logical
/// (immediate) group into its value: an element of `S+1` ones rotated
/// right by `R`, replicated across the register (the Arm ARM's
/// `DecodeBitMasks` with `immediate = TRUE`, returning `wmask` only).
///
/// `None` for the reserved encodings — no set bit in `N:NOT(imms)`, an
/// all-ones element, or a 64-bit element (`N == 1`) in the 32-bit form.
/// The result is masked to 32 bits when `sf` is clear.
pub fn decode_bit_masks(n: bool, imms: u8, immr: u8, sf: bool) -> Option<u64> {
    // The element size is 2^len where len indexes the highest set bit of
    // N:NOT(imms); no set bit (N = 0, imms = 111111) is reserved, as is
    // len = 0 (a 1-bit element has no room for a non-all-ones pattern).
    let selector = (u32::from(n) << 6) | (u32::from(!imms) & 0x3F);
    if selector < 2 {
        return None;
    }
    let len = 31 - selector.leading_zeros();
    let esize = 1u32 << len;
    if !sf && esize > 32 {
        return None;
    }
    let levels = (esize - 1) as u8;
    let s = imms & levels;
    let r = immr & levels;
    if s == levels {
        // The all-ones element cannot be encoded (it is the other
        // element sizes' job); the triple is reserved.
        return None;
    }
    // welem = Ones(s + 1), rotated right by r within the element, then
    // replicated out to the register width. s < levels <= 63, so the
    // shifts below stay in range.
    let emask = if esize == 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    let welem = (1u64 << (s + 1)) - 1;
    let rot = if r == 0 {
        welem
    } else {
        ((welem >> r) | (welem << (esize as u8 - r))) & emask
    };
    let mut out = 0u64;
    let mut shift = 0u32;
    while shift < 64 {
        out |= rot << shift;
        shift += esize;
    }
    Some(if sf { out } else { out & 0xFFFF_FFFF })
}

/// The value of the 8-bit VFP immediate of `FMOV` (immediate):
/// `±(16 + imm8[3:0])/16 × 2^e` with `e = (NOT(imm8[6]):imm8[5:4]) - 3`
/// — the same value for the single and double encodings (only the bit
/// patterns differ). Every value is a small dyadic rational, exact in
/// `f64` and in `f32`, so the single-precision lift casts without
/// rounding.
pub fn fp_imm_value(imm8: u8) -> f64 {
    let m = (imm8 & 0xF) as f64;
    let e = i32::from((!imm8 >> 6 & 1) << 2 | (imm8 >> 4 & 3)) - 3;
    let mag = (16.0 + m) / 16.0 * 2f64.powi(e);
    if imm8 & 0x80 != 0 { -mag } else { mag }
}

/// Expand a `MOVI`/`MVNI` immediate to the 64 bits it replicates across
/// the destination (the Arm ARM's `AdvSIMDExpandImm`): the element —
/// `imm` shifted by `shift` bits, with `msl` filling the vacated bits
/// with ones, complemented at the element width by `invert` — repeated
/// out to 64 bits. `size` is the element log2 width in bytes; at
/// `size == 3` the element is the byte mask of `imm` (each set bit an
/// `0xFF` byte) and the other parameters are ignored, per the encoding.
pub fn movi_expand(size: u8, imm: u8, shift: u8, msl: bool, invert: bool) -> u64 {
    let shift = (shift & 31) as u32;
    match size {
        0 => 0x0101_0101_0101_0101u64.wrapping_mul(imm as u64),
        1 => {
            let mut e = ((imm as u64) << shift) & 0xFFFF;
            if invert {
                e = !e & 0xFFFF;
            }
            e * 0x0001_0001_0001_0001
        }
        2 => {
            let mut e = ((imm as u64) << shift) & 0xFFFF_FFFF;
            if msl {
                e |= (1u64 << shift) - 1;
            }
            if invert {
                e = !e & 0xFFFF_FFFF;
            }
            e * 0x0000_0001_0000_0001
        }
        _ => {
            let mut out = 0u64;
            for i in 0..8 {
                if imm >> i & 1 == 1 {
                    out |= 0xFFu64 << (8 * i);
                }
            }
            out
        }
    }
}

/// Top-level dispatch on `op0` (bits \[28:25\] of every A64 word).
fn decode_word(w: u32, va: u64) -> Opcode {
    match bits(w, 28, 25) {
        // 100x: data processing — immediate.
        0b1000 | 0b1001 => decode_dp_imm(w, va),
        // 101x: branches, exception generation, system.
        0b1010 | 0b1011 => decode_branch_system(w, va),
        // x1x0: loads and stores.
        0b0100 | 0b0110 | 0b1100 | 0b1110 => decode_load_store(w, va),
        // x101: data processing (register).
        0b0101 | 0b1101 => decode_dp_reg(w),
        // x111: scalar FP and Advanced SIMD data processing (the moves).
        0b0111 | 0b1111 => decode_simd_fp(w),
        // 0000: the reserved group. UDF is its one allocated encoding —
        // the whole top half of the word zero, imm16 below. Everything
        // else stays unknown.
        0b0000 if w & 0xFFFF_0000 == 0 => Opcode::Udf {
            imm: bits(w, 15, 0) as u16,
        },
        // Reserved, SVE: unmodeled.
        _ => Opcode::Unknown(w),
    }
}

/// Data processing — immediate (`op0 == 100x`), keyed on bits \[25:23\].
fn decode_dp_imm(w: u32, va: u64) -> Opcode {
    let sf = bit(w, 31);
    let rd = bits(w, 4, 0) as u8;
    match bits(w, 25, 23) {
        // 00x: pc-relative addressing (ADR / ADRP); bit 23 is immhi data.
        0b000 | 0b001 => {
            let imm21 = (bits(w, 23, 5) << 2) | bits(w, 30, 29);
            let off = sext(imm21, 21);
            if sf {
                // ADRP: offset counts 4 KiB pages from the pc's page.
                let target = (va & !0xFFF).wrapping_add((off << 12) as u64);
                Opcode::Adrp { rd, target }
            } else {
                Opcode::Adr {
                    rd,
                    target: va.wrapping_add(off as u64),
                }
            }
        }
        // 010: add/subtract immediate.
        0b010 => {
            let imm12 = bits(w, 21, 10);
            let imm = if bit(w, 22) { imm12 << 12 } else { imm12 };
            let rn = bits(w, 9, 5) as u8;
            let set_flags = bit(w, 29);
            if bit(w, 30) {
                Opcode::SubImm {
                    sf,
                    set_flags,
                    rd,
                    rn,
                    imm,
                }
            } else {
                Opcode::AddImm {
                    sf,
                    set_flags,
                    rd,
                    rn,
                    imm,
                }
            }
        }
        // 101: move wide immediate.
        0b101 => {
            let hw = bits(w, 22, 21) as u8;
            if !sf && hw >= 2 {
                // 32-bit forms only allow shifts of 0 and 16.
                return Opcode::Unknown(w);
            }
            let imm = bits(w, 20, 5) as u16;
            let shift = hw * 16;
            match bits(w, 30, 29) {
                0b00 => Opcode::Movn { sf, rd, imm, shift },
                0b10 => Opcode::Movz { sf, rd, imm, shift },
                0b11 => Opcode::Movk { sf, rd, imm, shift },
                _ => Opcode::Unknown(w),
            }
        }
        // 100: logical immediate.
        0b100 => {
            let n = bit(w, 22);
            if !sf && n {
                return Opcode::Unknown(w);
            }
            let immr = bits(w, 21, 16) as u8;
            let imms = bits(w, 15, 10) as u8;
            let Some(imm) = decode_bit_masks(n, imms, immr, sf) else {
                return Opcode::Unknown(w);
            };
            let rn = bits(w, 9, 5) as u8;
            let (op, set_flags) = match bits(w, 30, 29) {
                0b00 => (LogOp::And, false),
                0b01 => (LogOp::Orr, false),
                0b10 => (LogOp::Eor, false),
                _ => (LogOp::And, true),
            };
            Opcode::LogImm {
                sf,
                op,
                set_flags,
                rd,
                rn,
                imm,
            }
        }
        // 110: bitfield. N must equal sf, and the 32-bit form keeps both
        // positions below 32; anything else is unallocated.
        0b110 => {
            let n = bit(w, 22);
            let immr = bits(w, 21, 16) as u8;
            let imms = bits(w, 15, 10) as u8;
            if n != sf || (!sf && (immr >= 32 || imms >= 32)) {
                return Opcode::Unknown(w);
            }
            let rn = bits(w, 9, 5) as u8;
            match bits(w, 30, 29) {
                0b00 => Opcode::Sbfm {
                    sf,
                    rd,
                    rn,
                    immr,
                    imms,
                },
                0b01 => Opcode::Bfm {
                    sf,
                    rd,
                    rn,
                    immr,
                    imms,
                },
                0b10 => Opcode::Ubfm {
                    sf,
                    rd,
                    rn,
                    immr,
                    imms,
                },
                _ => Opcode::Unknown(w),
            }
        }
        // Extract: sf 00 100111 N 0 Rm imms Rn Rd, N = sf, and the
        // 32-bit form's lsb tops out at 31. op21 (bits 30:29) and o0
        // (bit 21) are zero in the one allocated row.
        0b111 => {
            let lsb = bits(w, 15, 10) as u8;
            if bits(w, 30, 29) != 0
                || bit(w, 21)
                || bit(w, 22) != sf
                || (!sf && lsb >= 32)
            {
                return Opcode::Unknown(w);
            }
            Opcode::Extr {
                sf,
                rd,
                rn: bits(w, 9, 5) as u8,
                rm: bits(w, 20, 16) as u8,
                lsb,
            }
        }
        // Add/sub with tags.
        _ => Opcode::Unknown(w),
    }
}

/// Branches, exception generation and system (`op0 == 101x`).
fn decode_branch_system(w: u32, va: u64) -> Opcode {
    // B / BL: op[31] 00101 imm26.
    if w & 0x7C00_0000 == 0x1400_0000 {
        let target = branch_target(va, bits(w, 25, 0), 26);
        return if bit(w, 31) {
            Opcode::Bl { target }
        } else {
            Opcode::B { target }
        };
    }
    // CBZ / CBNZ: sf 011010 op imm19 Rt.
    if w & 0x7E00_0000 == 0x3400_0000 {
        let sf = bit(w, 31);
        let rt = bits(w, 4, 0) as u8;
        let target = branch_target(va, bits(w, 23, 5), 19);
        return if bit(w, 24) {
            Opcode::Cbnz { sf, rt, target }
        } else {
            Opcode::Cbz { sf, rt, target }
        };
    }
    // TBZ / TBNZ: b5 011011 op b40 imm14 Rt; bit number is b5:b40.
    if w & 0x7E00_0000 == 0x3600_0000 {
        let rt = bits(w, 4, 0) as u8;
        let bit_no = ((bits(w, 31, 31) << 5) | bits(w, 23, 19)) as u8;
        let target = branch_target(va, bits(w, 18, 5), 14);
        return if bit(w, 24) {
            Opcode::Tbnz {
                rt,
                bit: bit_no,
                target,
            }
        } else {
            Opcode::Tbz {
                rt,
                bit: bit_no,
                target,
            }
        };
    }
    // B.cond: 01010100 imm19 0 cond (bit 4 set is BC.cond, unmodeled).
    if w & 0xFF00_0010 == 0x5400_0000 {
        return Opcode::BCond {
            cond: Cond::from(bits(w, 3, 0) as u8),
            target: branch_target(va, bits(w, 23, 5), 19),
        };
    }
    // Exception generation: 11010100 opc(3) imm16 op2(3) LL(2).
    if w & 0xFF00_0000 == 0xD400_0000 {
        let imm = bits(w, 20, 5) as u16;
        if bits(w, 4, 2) == 0 {
            return match (bits(w, 23, 21), bits(w, 1, 0)) {
                (0b000, 0b01) => Opcode::Svc { imm },
                (0b000, 0b10) => Opcode::Hvc { imm },
                (0b000, 0b11) => Opcode::Smc { imm },
                (0b001, 0b00) => Opcode::Brk { imm },
                (0b010, 0b00) => Opcode::Hlt { imm },
                // DCPS1/2/3 and unallocated combinations.
                _ => Opcode::Unknown(w),
            };
        }
        return Opcode::Unknown(w);
    }
    // Hint space: 11010101 00000011 0010 CRm(4) op2(3) 11111.
    if w & 0xFFFF_F01F == 0xD503_201F {
        let imm = bits(w, 11, 5) as u8;
        return match imm {
            0 => Opcode::Nop,
            1 => Opcode::Yield,
            2 => Opcode::Wfe,
            3 => Opcode::Wfi,
            4 => Opcode::Sev,
            5 => Opcode::Sevl,
            // The PAC hints (CRm = 0011): not NOPs on arm64e — they
            // rewrite x30's authentication bits — so they get names and
            // an honest lift, unlike the unallocated remainder.
            25 => Opcode::PacHint {
                auth: false,
                key_b: false,
            },
            27 => Opcode::PacHint {
                auth: false,
                key_b: true,
            },
            29 => Opcode::PacHint {
                auth: true,
                key_b: false,
            },
            31 => Opcode::PacHint {
                auth: true,
                key_b: true,
            },
            _ => Opcode::Hint { imm },
        };
    }
    // Unconditional branch (register): 1101011 opc(4) op2(5) op3(6) Rn op4(5).
    if w & 0xFE00_0000 == 0xD600_0000 {
        if bits(w, 20, 16) != 0b11111 {
            return Opcode::Unknown(w);
        }
        let opc = bits(w, 24, 21);
        let op3 = bits(w, 15, 10);
        let rn = bits(w, 9, 5) as u8;
        let op4 = bits(w, 4, 0) as u8;
        // Plain BR/BLR/RET: op3 = 000000, op4 = 00000.
        if op3 == 0 && op4 == 0 {
            return match opc {
                0b0000 => Opcode::Br { rn },
                0b0001 => Opcode::Blr { rn },
                0b0010 => Opcode::Ret { rn },
                _ => Opcode::Unknown(w),
            };
        }
        // Pointer-auth variants: op3 = 00001M (M names the key).
        if op3 >> 1 != 1 {
            return Opcode::Unknown(w);
        }
        let key_b = op3 & 1 == 1;
        return match opc {
            // RETAA/RETAB: Rn and op4 both fixed at 11111.
            0b0010 if rn == 31 && op4 == 31 => Opcode::RetA { key_b },
            // BRAAZ/BLRAAZ (zero modifier): op4 = 11111.
            0b0000 if op4 == 31 => Opcode::BrAuth {
                link: false,
                key_b,
                zero: true,
                rn,
                rm: 31,
            },
            0b0001 if op4 == 31 => Opcode::BrAuth {
                link: true,
                key_b,
                zero: true,
                rn,
                rm: 31,
            },
            // BRAA/BLRAA: the modifier register rides op4 (SP-position).
            0b1000 => Opcode::BrAuth {
                link: false,
                key_b,
                zero: false,
                rn,
                rm: op4,
            },
            0b1001 => Opcode::BrAuth {
                link: true,
                key_b,
                zero: false,
                rn,
                rm: op4,
            },
            // ERETAA/ERETAB and the rest stay unmodeled.
            _ => Opcode::Unknown(w),
        };
    }
    Opcode::Unknown(w)
}

/// Loads and stores (`op0 == x1x0`). Bit 26 is the `V` flag selecting
/// the SIMD&FP register file; the addressing structure is shared, so
/// each family splits on `V` inside its own arm.
fn decode_load_store(w: u32, va: u64) -> Opcode {
    if bit(w, 26) {
        return decode_load_store_simd(w, va);
    }
    // LDR (literal): opc(2) 011 V 00 imm19 Rt.
    if w & 0x3F00_0000 == 0x1800_0000 {
        let rt = bits(w, 4, 0) as u8;
        let target = branch_target(va, bits(w, 23, 5), 19);
        return match bits(w, 31, 30) {
            0b00 => Opcode::LdrLit {
                sf: false,
                rt,
                target,
            },
            0b01 => Opcode::LdrLit {
                sf: true,
                rt,
                target,
            },
            // LDRSW / PRFM literal: not modeled.
            _ => Opcode::Unknown(w),
        };
    }
    // Load/store exclusive and ordered: size(2) 001000 o2 L o1 Rs o0
    // Rt2 Rn Rt. Only the register (non-pair, o1 = 0, Rt2 = 11111)
    // forms are modeled; LORegions (o2 = 1, o0 = 0) and the pair forms
    // stay unknown.
    if bits(w, 29, 24) == 0b001000 {
        let size = bits(w, 31, 30) as u8;
        let rt = bits(w, 4, 0) as u8;
        let rn = bits(w, 9, 5) as u8;
        let rs = bits(w, 20, 16) as u8;
        if bit(w, 21) || bits(w, 14, 10) != 0b11111 {
            return Opcode::Unknown(w);
        }
        let (o2, l, o0) = (bit(w, 23), bit(w, 22), bit(w, 15));
        return match (o2, l, o0) {
            (true, true, true) if rs == 31 => Opcode::Ldar { size, rt, rn },
            (true, false, true) if rs == 31 => Opcode::Stlr { size, rt, rn },
            (false, true, acquire) if rs == 31 => Opcode::Ldxr {
                size,
                acquire,
                rt,
                rn,
            },
            (false, false, release) => Opcode::Stxr {
                size,
                release,
                ws: rs,
                rt,
                rn,
            },
            _ => Opcode::Unknown(w),
        };
    }
    // Load/store pair: opc(2) 101 V mode(3 = bits 25:23) L imm7 Rt2 Rn Rt.
    if bits(w, 29, 27) == 0b101 {
        // LDPSW (opc = 01, load only, 4-byte scale) rides the same
        // addressing decode as its own opcode.
        if bits(w, 31, 30) == 0b01 {
            if !bit(w, 22) {
                return Opcode::Unknown(w);
            }
            let off = sext(bits(w, 21, 15), 7) << 2;
            let mode = match bits(w, 25, 23) {
                0b001 => AddrMode::PostIndex(off),
                0b010 => AddrMode::Offset(off),
                0b011 => AddrMode::PreIndex(off),
                _ => return Opcode::Unknown(w),
            };
            return Opcode::LdpSw {
                rt: bits(w, 4, 0) as u8,
                rt2: bits(w, 14, 10) as u8,
                rn: bits(w, 9, 5) as u8,
                mode,
            };
        }
        let sf = match bits(w, 31, 30) {
            0b00 => false,
            0b10 => true,
            // Unallocated (11): not modeled.
            _ => return Opcode::Unknown(w),
        };
        let scale = if sf { 3 } else { 2 };
        let off = sext(bits(w, 21, 15), 7) << scale;
        let mode = match bits(w, 25, 23) {
            0b001 => AddrMode::PostIndex(off),
            0b010 => AddrMode::Offset(off),
            0b011 => AddrMode::PreIndex(off),
            // LDNP/STNP (no-allocate): not modeled.
            _ => return Opcode::Unknown(w),
        };
        let rt = bits(w, 4, 0) as u8;
        let rt2 = bits(w, 14, 10) as u8;
        let rn = bits(w, 9, 5) as u8;
        return if bit(w, 22) {
            Opcode::Ldp {
                sf,
                rt,
                rt2,
                rn,
                mode,
            }
        } else {
            Opcode::Stp {
                sf,
                rt,
                rt2,
                rn,
                mode,
            }
        };
    }
    // Load/store register (immediate): size(2) 111 V .. opc(2) ...
    if bits(w, 29, 27) == 0b111 {
        let size = bits(w, 31, 30) as u8;
        let rt = bits(w, 4, 0) as u8;
        let rn = bits(w, 9, 5) as u8;
        let opc = bits(w, 23, 22);
        // opc selects store / load / sign-extending load, gated by size:
        // `10` sign-extends to 64 bits (but is PRFM at doubleword), `11`
        // sign-extends to 32 bits (byte/halfword only). Everything else is
        // PRFM or an unallocated size/opc pair.
        match opc {
            0b00 | 0b01 => {}
            0b10 if size != 3 => {}
            0b11 if size < 2 => {}
            _ => return Opcode::Unknown(w),
        }
        // The immediate-form opcode selected by `opc` for a given `mode`.
        let imm_form = |mode: AddrMode| match opc {
            0b00 => Opcode::Str { size, rt, rn, mode },
            0b01 => Opcode::Ldr { size, rt, rn, mode },
            0b10 => Opcode::Ldrs {
                size,
                sf: true,
                rt,
                rn,
                mode,
            },
            _ => Opcode::Ldrs {
                size,
                sf: false,
                rt,
                rn,
                mode,
            },
        };
        // Unsigned immediate: size 111 V 01 opc imm12 Rn Rt (offset scaled
        // by the access size).
        if bits(w, 25, 24) == 0b01 {
            return imm_form(AddrMode::Offset((bits(w, 21, 10) as i64) << size));
        }
        // Unscaled and pre/post-indexed: size 111 V 00 opc 0 imm9 idx(2)
        // Rn Rt with a signed 9-bit byte offset; idx 00 is the unscaled
        // LDUR/STUR family (no writeback), 10 the unprivileged LDTR/STTR
        // form (not modeled).
        if bits(w, 25, 24) == 0b00 && !bit(w, 21) {
            let off = sext(bits(w, 20, 12), 9);
            match bits(w, 11, 10) {
                0b00 => {
                    return match opc {
                        0b00 => Opcode::Stur {
                            size,
                            rt,
                            rn,
                            imm: off,
                        },
                        0b01 => Opcode::Ldur {
                            size,
                            rt,
                            rn,
                            imm: off,
                        },
                        0b10 => Opcode::Ldurs {
                            size,
                            sf: true,
                            rt,
                            rn,
                            imm: off,
                        },
                        _ => Opcode::Ldurs {
                            size,
                            sf: false,
                            rt,
                            rn,
                            imm: off,
                        },
                    };
                }
                0b01 => return imm_form(AddrMode::PostIndex(off)),
                0b11 => return imm_form(AddrMode::PreIndex(off)),
                _ => return Opcode::Unknown(w),
            }
        }
        // Register offset: size 111 V 00 opc 1 Rm option(3) S 10 Rn Rt. Only
        // option<1> = 1 (UXTW/LSL/SXTW/SXTX) is allocated for this class.
        if bits(w, 25, 24) == 0b00 && bit(w, 21) && bits(w, 11, 10) == 0b10 {
            if !bit(w, 14) {
                return Opcode::Unknown(w);
            }
            let off = RegOffset {
                rm: bits(w, 20, 16) as u8,
                option: bits(w, 15, 13) as u8,
                scaled: bit(w, 12),
            };
            return match opc {
                0b00 => Opcode::StrReg { size, rt, rn, off },
                0b01 => Opcode::LdrReg { size, rt, rn, off },
                0b10 => Opcode::LdrsReg {
                    size,
                    sf: true,
                    rt,
                    rn,
                    off,
                },
                _ => Opcode::LdrsReg {
                    size,
                    sf: false,
                    rt,
                    rn,
                    off,
                },
            };
        }
        // Atomics, PAC forms, and the unprivileged remainder.
        return Opcode::Unknown(w);
    }
    // Exclusives, memory tags, and the rest of the load/store space.
    Opcode::Unknown(w)
}

/// SIMD&FP loads and stores (`op0 == x1x0`, `V == 1`): the same
/// addressing structure as the integer families over the b/h/s/d/q
/// register file. The structure loads/stores (`LD1`/`ST1`/...) live in a
/// different sub-space and stay [`Opcode::Unknown`].
fn decode_load_store_simd(w: u32, va: u64) -> Opcode {
    let rt = bits(w, 4, 0) as u8;
    let rn = bits(w, 9, 5) as u8;
    // LDR (literal, SIMD&FP): opc(2) 011 1 00 imm19 Rt; opc is the size.
    if w & 0x3F00_0000 == 0x1C00_0000 {
        let size = match bits(w, 31, 30) {
            0b00 => 2,
            0b01 => 3,
            0b10 => 4,
            // opc = 11 is unallocated.
            _ => return Opcode::Unknown(w),
        };
        return Opcode::FLdrLit {
            size,
            rt,
            target: branch_target(va, bits(w, 23, 5), 19),
        };
    }
    // Load/store pair: opc(2) 101 1 mode(3) L imm7 Rt2 Rn Rt; opc is the
    // size (s/d/q; opc = 11 is unallocated).
    if bits(w, 29, 27) == 0b101 {
        let size = match bits(w, 31, 30) {
            0b00 => 2,
            0b01 => 3,
            0b10 => 4,
            _ => return Opcode::Unknown(w),
        };
        let off = sext(bits(w, 21, 15), 7) << size;
        let mode = match bits(w, 25, 23) {
            0b001 => AddrMode::PostIndex(off),
            0b010 => AddrMode::Offset(off),
            0b011 => AddrMode::PreIndex(off),
            // LDNP/STNP (no-allocate): not modeled.
            _ => return Opcode::Unknown(w),
        };
        let rt2 = bits(w, 14, 10) as u8;
        return if bit(w, 22) {
            Opcode::FLdp {
                size,
                rt,
                rt2,
                rn,
                mode,
            }
        } else {
            Opcode::FStp {
                size,
                rt,
                rt2,
                rn,
                mode,
            }
        };
    }
    // Load/store register: size(2) 111 1 .. opc(2) ... — `opc<0>` is the
    // load bit; `opc<1>` extends the access width to quad, allocated only
    // at size = 00.
    if bits(w, 29, 27) == 0b111 {
        let opc = bits(w, 23, 22);
        let size = match (bits(w, 31, 30), opc >> 1) {
            (s, 0) => s as u8,
            (0b00, _) => 4,
            _ => return Opcode::Unknown(w),
        };
        let load = opc & 1 == 1;
        // Unsigned immediate: offset scaled by the access size.
        if bits(w, 25, 24) == 0b01 {
            let mode = AddrMode::Offset((bits(w, 21, 10) as i64) << size);
            return if load {
                Opcode::FLdr { size, rt, rn, mode }
            } else {
                Opcode::FStr { size, rt, rn, mode }
            };
        }
        // Unscaled and pre/post-indexed: signed 9-bit byte offset. There
        // is no unprivileged (LDTR-analog) form on the V side; idx = 10
        // is unallocated.
        if bits(w, 25, 24) == 0b00 && !bit(w, 21) {
            let off = sext(bits(w, 20, 12), 9);
            return match (bits(w, 11, 10), load) {
                (0b00, true) => Opcode::FLdur {
                    size,
                    rt,
                    rn,
                    imm: off,
                },
                (0b00, false) => Opcode::FStur {
                    size,
                    rt,
                    rn,
                    imm: off,
                },
                (0b01, _) | (0b11, _) => {
                    let mode = if bits(w, 11, 10) == 0b01 {
                        AddrMode::PostIndex(off)
                    } else {
                        AddrMode::PreIndex(off)
                    };
                    if load {
                        Opcode::FLdr { size, rt, rn, mode }
                    } else {
                        Opcode::FStr { size, rt, rn, mode }
                    }
                }
                _ => Opcode::Unknown(w),
            };
        }
        // Register offset: only option<1> = 1 is allocated, as for the
        // integer class.
        if bits(w, 25, 24) == 0b00 && bit(w, 21) && bits(w, 11, 10) == 0b10 {
            if !bit(w, 14) {
                return Opcode::Unknown(w);
            }
            let off = RegOffset {
                rm: bits(w, 20, 16) as u8,
                option: bits(w, 15, 13) as u8,
                scaled: bit(w, 12),
            };
            return if load {
                Opcode::FLdrReg { size, rt, rn, off }
            } else {
                Opcode::FStrReg { size, rt, rn, off }
            };
        }
        return Opcode::Unknown(w);
    }
    // Structure loads/stores and the rest of the V = 1 space.
    Opcode::Unknown(w)
}

/// Scalar FP and Advanced SIMD data processing (`op0 == x111`): the
/// move mass (`FMOV`/`MOVI`/`MVNI`), scalar FP arithmetic, element
/// moves, and the Advanced SIMD three-same integer ALU slice
/// (`ADD`/`SUB`/`AND`/`ORR`/`EOR`). Everything else in the space
/// (remaining vector ALU, structure loads/stores, half precision, the
/// vector `ORR`/`BIC` immediates) is [`Opcode::Unknown`].
fn decode_simd_fp(w: u32) -> Opcode {
    let rd = bits(w, 4, 0) as u8;
    let rn = bits(w, 9, 5) as u8;
    // FMOV (register): 000 11110 type 1 000000 10000 Rn Rd.
    if w & 0xFF3F_FC00 == 0x1E20_4000 {
        return match bits(w, 23, 22) {
            0b00 => Opcode::FmovReg {
                double: false,
                rd,
                rn,
            },
            0b01 => Opcode::FmovReg {
                double: true,
                rd,
                rn,
            },
            // Half precision (FEAT_FP16) is a documented gap; type = 10
            // is unallocated.
            _ => Opcode::Unknown(w),
        };
    }
    // FMOV (scalar immediate): 000 11110 type 1 imm8 100 00000 Rd.
    if w & 0xFF20_1FE0 == 0x1E20_1000 {
        let imm = bits(w, 20, 13) as u8;
        return match bits(w, 23, 22) {
            0b00 => Opcode::FmovImm {
                double: false,
                imm,
                rd,
            },
            0b01 => Opcode::FmovImm {
                double: true,
                imm,
                rd,
            },
            _ => Opcode::Unknown(w),
        };
    }
    // Conversion between FP and general (FMOV/SCVTF/UCVTF/FCVT<r><s|u>):
    // sf 00 11110 type 1 rmode(2) opcode(3) 000000 Rn Rd. FJCVTZS and
    // the half-precision rows stay unknown.
    if w & 0x7F20_FC00 == 0x1E20_0000 {
        let sf = bit(w, 31);
        let ty = bits(w, 23, 22);
        let rmode = bits(w, 20, 19);
        let opcode = bits(w, 18, 16);
        // The six FMOV combinations first (including the V.D[1] lane).
        if opcode >= 0b110 {
            let to_gp = opcode == 0b110;
            let hi = match (sf, ty, rmode) {
                (false, 0b00, 0b00) | (true, 0b01, 0b00) => false,
                (true, 0b10, 0b01) => true,
                _ => return Opcode::Unknown(w),
            };
            return if to_gp {
                Opcode::FmovToGp { sf, hi, rd, rn }
            } else {
                Opcode::FmovFromGp { sf, hi, rd, rn }
            };
        }
        let double = match ty {
            0b00 => false,
            0b01 => true,
            _ => return Opcode::Unknown(w),
        };
        // SCVTF/UCVTF (integer to FP): rmode is fixed at 00.
        if opcode == 0b010 || opcode == 0b011 {
            if rmode != 0b00 {
                return Opcode::Unknown(w);
            }
            return Opcode::FcvtToFp {
                sf,
                double,
                unsigned: opcode == 0b011,
                rd,
                rn,
            };
        }
        // FCVT<r><s|u> (FP to integer): opcode 00x with the rounding in
        // rmode, plus the ties-away pair at (00, 10x).
        let round = match (rmode, opcode) {
            (0b00, 0b000 | 0b001) => FpRound::N,
            (0b01, 0b000 | 0b001) => FpRound::P,
            (0b10, 0b000 | 0b001) => FpRound::M,
            (0b11, 0b000 | 0b001) => FpRound::Z,
            (0b00, 0b100 | 0b101) => FpRound::A,
            _ => return Opcode::Unknown(w),
        };
        return Opcode::FcvtFromFp {
            sf,
            double,
            unsigned: opcode & 1 == 1,
            round,
            rd,
            rn,
        };
    }
    // Scalar FP compare: 000 11110 type 1 Rm op(2)=00 1000 Rn opcode2(5),
    // opcode2's low three bits zero; bit 4 selects FCMPE, bit 3 the
    // literal-zero comparand.
    if w & 0xFF20_FC07 == 0x1E20_2000 {
        let double = match bits(w, 23, 22) {
            0b00 => false,
            0b01 => true,
            _ => return Opcode::Unknown(w),
        };
        let rm = bits(w, 20, 16) as u8;
        return Opcode::Fcmp {
            double,
            signal: bit(w, 4),
            rn,
            rm: if bit(w, 3) { None } else { Some(rm) },
        };
    }
    // Scalar FP conditional compare: 000 11110 type 1 Rm cond 01 Rn op
    // nzcv.
    if w & 0xFF20_0C00 == 0x1E20_0400 {
        let double = match bits(w, 23, 22) {
            0b00 => false,
            0b01 => true,
            _ => return Opcode::Unknown(w),
        };
        return Opcode::Fccmp {
            double,
            signal: bit(w, 4),
            rn,
            rm: bits(w, 20, 16) as u8,
            nzcv: bits(w, 3, 0) as u8,
            cond: Cond::from(bits(w, 15, 12) as u8),
        };
    }
    // Scalar FP conditional select: 000 11110 type 1 Rm cond 11 Rn Rd.
    if w & 0xFF20_0C00 == 0x1E20_0C00 {
        let double = match bits(w, 23, 22) {
            0b00 => false,
            0b01 => true,
            _ => return Opcode::Unknown(w),
        };
        return Opcode::Fcsel {
            double,
            rd,
            rn,
            rm: bits(w, 20, 16) as u8,
            cond: Cond::from(bits(w, 15, 12) as u8),
        };
    }
    // Scalar FP two-source: 000 11110 type 1 Rm opcode(4) 10 Rn Rd.
    if w & 0xFF20_0C00 == 0x1E20_0800 {
        let double = match bits(w, 23, 22) {
            0b00 => false,
            0b01 => true,
            _ => return Opcode::Unknown(w),
        };
        let op = match bits(w, 15, 12) {
            0b0000 => F2Op::Mul,
            0b0001 => F2Op::Div,
            0b0010 => F2Op::Add,
            0b0011 => F2Op::Sub,
            0b0100 => F2Op::Max,
            0b0101 => F2Op::Min,
            0b0110 => F2Op::MaxNm,
            0b0111 => F2Op::MinNm,
            0b1000 => F2Op::NMul,
            // 1001..1111 are unallocated.
            _ => return Opcode::Unknown(w),
        };
        return Opcode::FArith2 {
            op,
            double,
            rd,
            rn,
            rm: bits(w, 20, 16) as u8,
        };
    }
    // Scalar FP three-source: 000 11111 type o1 Rm o0 Ra Rn Rd.
    if w & 0xFF00_0000 == 0x1F00_0000 {
        let double = match bits(w, 23, 22) {
            0b00 => false,
            0b01 => true,
            _ => return Opcode::Unknown(w),
        };
        return Opcode::FArith3 {
            negate: bit(w, 21),
            sub: bit(w, 15),
            double,
            rd,
            rn,
            rm: bits(w, 20, 16) as u8,
            ra: bits(w, 14, 10) as u8,
        };
    }
    // Scalar FP one-source: 000 11110 type 1 opcode(6) 10000 Rn Rd.
    // opcode 000000 (FMOV register) is handled above; FCVT changes
    // precision, FRINT's seven roundings share the 0010xx row.
    if w & 0xFF20_7C00 == 0x1E20_4000 {
        let double = match bits(w, 23, 22) {
            0b00 => false,
            0b01 => true,
            _ => return Opcode::Unknown(w),
        };
        let op = match bits(w, 20, 15) {
            0b000001 => F1Op::Abs,
            0b000010 => F1Op::Neg,
            0b000011 => F1Op::Sqrt,
            // FCVT: opcode 0001xx, the low bits naming the destination
            // type — only s <-> d modeled, and a same-type conversion is
            // unallocated.
            0b000101 if !double => return Opcode::FCvtPrec { to_double: true, rd, rn },
            0b000100 if double => return Opcode::FCvtPrec { to_double: false, rd, rn },
            0b001000 => F1Op::Rint(FpRound::N),
            0b001001 => F1Op::Rint(FpRound::P),
            0b001010 => F1Op::Rint(FpRound::M),
            0b001011 => F1Op::Rint(FpRound::Z),
            0b001100 => F1Op::Rint(FpRound::A),
            0b001110 => F1Op::RintX,
            0b001111 => F1Op::RintI,
            _ => return Opcode::Unknown(w),
        };
        return Opcode::FArith1 { op, double, rd, rn };
    }
    // SCVTF/UCVTF (scalar integer): 01 U 11110 0s 1 0000 11101 10 Rn Rd,
    // s <-> d only.
    if w & 0xDF3F_FC00 == 0x5E21_D800 {
        return Opcode::FcvtIntScalar {
            double: bit(w, 22),
            unsigned: bit(w, 29),
            rd,
            rn,
        };
    }
    // Advanced SIMD copy: 0 Q op 01110000 imm5 0 imm4 1 Rn Rd — the
    // DUP/INS/UMOV/SMOV element moves. A zero low-bit run in imm5 (no
    // element size) is unallocated.
    if w & 0x9FE0_8400 == 0x0E00_0400 {
        let q = bit(w, 30);
        let imm5 = bits(w, 20, 16) as u8;
        let imm4 = bits(w, 14, 11) as u8;
        let size = imm5.trailing_zeros() as u8;
        if size > 3 {
            return Opcode::Unknown(w);
        }
        let index = imm5 >> (size + 1);
        if bit(w, 29) {
            // op = 1: INS (element). Q is fixed at 1.
            if !q {
                return Opcode::Unknown(w);
            }
            return Opcode::InsElem {
                size,
                dst: index,
                src: imm4 >> size,
                rd,
                rn,
            };
        }
        return match imm4 {
            // DUP (element): a 64-bit element needs the 128-bit form.
            0b0000 if size < 3 || q => Opcode::DupElemVec {
                q,
                size,
                index,
                rd,
                rn,
            },
            // DUP (general): same constraint.
            0b0001 if size < 3 || q => Opcode::DupGp { q, size, rd, rn },
            // INS (general): Q fixed at 1.
            0b0011 if q => Opcode::InsGp {
                size,
                index,
                rd,
                rn,
            },
            // SMOV: W from b/h, X from b/h/s.
            0b0101 if (!q && size < 2) || (q && size < 3) => Opcode::Smov {
                sf: q,
                size,
                index,
                rd,
                rn,
            },
            // UMOV: W from b/h/s, X from d only.
            0b0111 if (!q && size < 3) || (q && size == 3) => Opcode::Umov {
                sf: q,
                size,
                index,
                rd,
                rn,
            },
            _ => Opcode::Unknown(w),
        };
    }
    // Advanced SIMD scalar copy: 01 op 11110000 imm5 0 imm4 1 Rn Rd —
    // only DUP (element, scalar), which otool spells `mov`.
    if w & 0xFFE0_8400 == 0x5E00_0400 {
        let imm5 = bits(w, 20, 16) as u8;
        if bits(w, 14, 11) != 0 {
            return Opcode::Unknown(w);
        }
        let size = imm5.trailing_zeros() as u8;
        if size > 3 {
            return Opcode::Unknown(w);
        }
        return Opcode::DupElemScalar {
            size,
            index: imm5 >> (size + 1),
            rd,
            rn,
        };
    }
    // Advanced SIMD modified immediate:
    // 0 Q op 0111100000 abc cmode 0 1 defgh Rd.
    if w & 0x9FF8_0C00 == 0x0F00_0400 {
        let q = bit(w, 30);
        let op = bit(w, 29);
        let imm = (bits(w, 18, 16) << 5 | bits(w, 9, 5)) as u8;
        let cmode = bits(w, 15, 12) as u8;
        let movi = |size: u8, shift: u8, msl: bool| Opcode::Movi {
            q,
            invert: op,
            size,
            imm,
            shift,
            msl,
            rd,
        };
        return match cmode {
            // 32-bit shifted; the odd cmodes are ORR/BIC (not moves).
            0b0000 | 0b0010 | 0b0100 | 0b0110 => movi(2, 8 * (cmode >> 1), false),
            // 16-bit shifted.
            0b1000 | 0b1010 => movi(1, 8 * (cmode >> 1 & 1), false),
            // 32-bit shifting-ones (MSL).
            0b1100 | 0b1101 => movi(2, 8 << (cmode & 1), true),
            // 8-bit (op = 0) / 64-bit byte mask (op = 1).
            0b1110 if !op => movi(0, 0, false),
            0b1110 => Opcode::Movi {
                q,
                invert: false,
                size: 3,
                imm,
                shift: 0,
                msl: false,
                rd,
            },
            // FMOV (vector immediate): single (op = 0) or, at Q = 1,
            // double (op = 1); op = 1, Q = 0 is unallocated.
            0b1111 if !op => Opcode::FmovVecImm {
                q,
                double: false,
                imm,
                rd,
            },
            0b1111 if q => Opcode::FmovVecImm {
                q,
                double: true,
                imm,
                rd,
            },
            _ => Opcode::Unknown(w),
        };
    }
    // Advanced SIMD three-same (public Arm ARM encodings):
    // 0 Q U 01110 size 1 Rm opcode 1 Rn Rd — integer ADD/SUB (10000),
    // AND/ORR/EOR (00011), CMEQ (10001)/CMHI (00110), and FP
    // FADD (11010)/FMUL (11011; U selects the FP op).
    if w & 0x9F20_0400 == 0x0E20_0400 {
        let q = bit(w, 30);
        let u = bit(w, 29);
        let size = bits(w, 23, 22) as u8;
        let opcode = bits(w, 15, 11);
        let rm = bits(w, 20, 16) as u8;
        let op = match (u, size, opcode) {
            (false, _, 0b10000) if size < 3 || q => SimdAluOp::Add,
            (true, _, 0b10000) if size < 3 || q => SimdAluOp::Sub,
            // Logical: size selects the op; arrangement is always .8b/.16b.
            (false, 0b00, 0b00011) => SimdAluOp::And,
            (false, 0b10, 0b00011) => SimdAluOp::Orr,
            (true, 0b00, 0b00011) => SimdAluOp::Eor,
            // CMEQ / CMHI (integer); .2d requires Q = 1.
            (true, _, 0b10001) if size < 3 || q => SimdAluOp::Cmeq,
            (true, _, 0b00110) if size < 3 || q => SimdAluOp::Cmhi,
            // FADD / FMUL: encoding size 0 = S, 1 = D (Q = 1 for .2d).
            (false, 0b00, 0b11010) => SimdAluOp::Fadd,
            (false, 0b01, 0b11010) if q => SimdAluOp::Fadd,
            (true, 0b00, 0b11011) => SimdAluOp::Fmul,
            (true, 0b01, 0b11011) if q => SimdAluOp::Fmul,
            _ => return Opcode::Unknown(w),
        };
        let elem = match op {
            SimdAluOp::And | SimdAluOp::Orr | SimdAluOp::Eor => 0,
            // FP encoding size → arrangement log2 width (S = 2, D = 3).
            SimdAluOp::Fadd | SimdAluOp::Fmul => 2 + size,
            _ => size,
        };
        return Opcode::SimdAlu {
            op,
            q,
            size: elem,
            rd,
            rn,
            rm,
        };
    }
    Opcode::Unknown(w)
}

/// Data processing — register (`op0 == x101`): the shifted- and
/// extended-register arithmetic/logical groups, add/sub with carry,
/// conditional compare, conditional select, the two-source
/// (shift/divide) and three-source (multiply-accumulate) groups. The
/// remaining encodings (one-source, flag ops, ...) stay
/// [`Opcode::Unknown`].
fn decode_dp_reg(w: u32) -> Opcode {
    let sf = bit(w, 31);
    let rd = bits(w, 4, 0) as u8;
    let rn = bits(w, 9, 5) as u8;
    let rm = bits(w, 20, 16) as u8;
    // Add/sub with carry: sf op S 11010000 Rm 000000 Rn Rd. A nonzero
    // opcode2 field (bits 15:10) is RMIF/SETF* or unallocated.
    if bits(w, 28, 21) == 0b1101_0000 {
        if bits(w, 15, 10) != 0 {
            return Opcode::Unknown(w);
        }
        let set_flags = bit(w, 29);
        return if bit(w, 30) {
            Opcode::Sbc {
                sf,
                set_flags,
                rd,
                rn,
                rm,
            }
        } else {
            Opcode::Adc {
                sf,
                set_flags,
                rd,
                rn,
                rm,
            }
        };
    }
    // Conditional compare: sf op S 11010010 Rm/imm5 cond imm 0 Rn 0 nzcv,
    // with S = 1 and the reserved o2 (bit 10) and o3 (bit 4) both clear;
    // bit 11 selects the immediate form. S = 0 is unallocated.
    if bits(w, 28, 21) == 0b1101_0010 {
        if !bit(w, 29) || bit(w, 10) || bit(w, 4) {
            return Opcode::Unknown(w);
        }
        let sub = bit(w, 30);
        let cond = Cond::from(bits(w, 15, 12) as u8);
        let nzcv = bits(w, 3, 0) as u8;
        return if bit(w, 11) {
            Opcode::CcmpImm {
                sf,
                sub,
                rn,
                imm: rm,
                nzcv,
                cond,
            }
        } else {
            Opcode::CcmpReg {
                sf,
                sub,
                rn,
                rm,
                nzcv,
                cond,
            }
        };
    }
    // Conditional select: sf op S 11010100 Rm cond op2(2) Rn Rd, with S = 0
    // and op2 in {00, 01}.
    if bits(w, 29, 21) == 0b0_1101_0100 {
        let cond = Cond::from(bits(w, 15, 12) as u8);
        return match (bit(w, 30), bits(w, 11, 10)) {
            (false, 0b00) => Opcode::Csel { sf, rd, rn, rm, cond },
            (false, 0b01) => Opcode::Csinc { sf, rd, rn, rm, cond },
            (true, 0b00) => Opcode::Csinv { sf, rd, rn, rm, cond },
            (true, 0b01) => Opcode::Csneg { sf, rd, rn, rm, cond },
            // op2 = 1x is unallocated for conditional select.
            _ => Opcode::Unknown(w),
        };
    }
    // Logical (shifted register): sf opc(2) 01010 shift(2) N Rm imm6 Rn Rd.
    if bits(w, 28, 24) == 0b0_1010 {
        let amount = bits(w, 15, 10) as u8;
        if !sf && amount >= 32 {
            // The 32-bit form's shift amount tops out at 31.
            return Opcode::Unknown(w);
        }
        let shift = match bits(w, 23, 22) {
            0b00 => Shift::Lsl,
            0b01 => Shift::Lsr,
            0b10 => Shift::Asr,
            _ => Shift::Ror,
        };
        let invert = bit(w, 21);
        let (op, set_flags) = match bits(w, 30, 29) {
            0b00 => (LogOp::And, false),
            0b01 => (LogOp::Orr, false),
            0b10 => (LogOp::Eor, false),
            _ => (LogOp::And, true),
        };
        return Opcode::LogReg {
            sf,
            op,
            set_flags,
            invert,
            rd,
            rn,
            rm,
            shift,
            amount,
        };
    }
    // Add/sub (shifted register): sf op S 01011 shift(2) 0 Rm imm6 Rn Rd.
    if bits(w, 28, 24) == 0b0_1011 && !bit(w, 21) {
        let amount = bits(w, 15, 10) as u8;
        let shift = match bits(w, 23, 22) {
            0b00 => Shift::Lsl,
            0b01 => Shift::Lsr,
            0b10 => Shift::Asr,
            // ROR is reserved for add/sub.
            _ => return Opcode::Unknown(w),
        };
        if !sf && amount >= 32 {
            return Opcode::Unknown(w);
        }
        let set_flags = bit(w, 29);
        return if bit(w, 30) {
            Opcode::SubReg {
                sf,
                set_flags,
                rd,
                rn,
                rm,
                shift,
                amount,
            }
        } else {
            Opcode::AddReg {
                sf,
                set_flags,
                rd,
                rn,
                rm,
                shift,
                amount,
            }
        };
    }
    // Add/sub (extended register): sf op S 01011 opt(2) 1 Rm option(3)
    // imm3 Rn Rd, with opt = 00 and imm3 <= 4 the only allocated forms.
    if bits(w, 28, 24) == 0b0_1011 && bit(w, 21) {
        if bits(w, 23, 22) != 0 {
            return Opcode::Unknown(w);
        }
        let amount = bits(w, 12, 10) as u8;
        if amount > 4 {
            return Opcode::Unknown(w);
        }
        let option = bits(w, 15, 13) as u8;
        let set_flags = bit(w, 29);
        return if bit(w, 30) {
            Opcode::SubExt {
                sf,
                set_flags,
                rd,
                rn,
                rm,
                option,
                amount,
            }
        } else {
            Opcode::AddExt {
                sf,
                set_flags,
                rd,
                rn,
                rm,
                option,
                amount,
            }
        };
    }
    // One-source: sf 1 S 11010110 opcode2(5) opcode(6) Rn Rd, with
    // S = 0. opcode2 = 00000 is the bit group (RBIT/REV*/CLZ/CLS);
    // opcode2 = 00001 at sf = 1 is the PAC group (I keys and XPAC
    // modeled; the D-key data ops stay unknown, zero corpus sites).
    if bits(w, 30, 21) == 0b10_1101_0110 {
        let opcode = bits(w, 15, 10);
        if bits(w, 20, 16) == 0b00000 {
            let op = match opcode {
                0b000000 => Bit1Op::Rbit,
                0b000001 => Bit1Op::Rev16,
                0b000010 if sf => Bit1Op::Rev32,
                0b000010 => Bit1Op::Rev,
                0b000011 if sf => Bit1Op::Rev,
                0b000100 => Bit1Op::Clz,
                0b000101 => Bit1Op::Cls,
                _ => return Opcode::Unknown(w),
            };
            return Opcode::Bits1 { op, sf, rd, rn };
        }
        if bits(w, 20, 16) == 0b00001 && sf {
            // The Z forms (opcode bit 3) require Rn = 11111.
            let zero = opcode & 0b001000 != 0;
            if zero && rn != 31 {
                return Opcode::Unknown(w);
            }
            return match opcode {
                0b000000 | 0b001000 => Opcode::PacGpr {
                    auth: false,
                    key_b: false,
                    zero,
                    rd,
                    rn,
                },
                0b000001 | 0b001001 => Opcode::PacGpr {
                    auth: false,
                    key_b: true,
                    zero,
                    rd,
                    rn,
                },
                0b000100 | 0b001100 => Opcode::PacGpr {
                    auth: true,
                    key_b: false,
                    zero,
                    rd,
                    rn,
                },
                0b000101 | 0b001101 => Opcode::PacGpr {
                    auth: true,
                    key_b: true,
                    zero,
                    rd,
                    rn,
                },
                0b010000 if rn == 31 => Opcode::XPac { data: false, rd },
                0b010001 if rn == 31 => Opcode::XPac { data: true, rd },
                _ => Opcode::Unknown(w),
            };
        }
        return Opcode::Unknown(w);
    }
    // Two-source: sf 0 S 11010110 Rm opcode(6) Rn Rd, with S = 0.
    if bits(w, 30, 21) == 0b00_1101_0110 {
        return match bits(w, 15, 10) {
            0b000010 => Opcode::Udiv { sf, rd, rn, rm },
            0b000011 => Opcode::Sdiv { sf, rd, rn, rm },
            0b001000 => Opcode::ShiftReg {
                sf,
                kind: Shift::Lsl,
                rd,
                rn,
                rm,
            },
            0b001001 => Opcode::ShiftReg {
                sf,
                kind: Shift::Lsr,
                rd,
                rn,
                rm,
            },
            0b001010 => Opcode::ShiftReg {
                sf,
                kind: Shift::Asr,
                rd,
                rn,
                rm,
            },
            0b001011 => Opcode::ShiftReg {
                sf,
                kind: Shift::Ror,
                rd,
                rn,
                rm,
            },
            // CRC32, subps, and the unallocated remainder.
            _ => Opcode::Unknown(w),
        };
    }
    // Three-source: sf op54(2) 11011 op31(3) Rm o0 Ra Rn Rd, op54 = 00.
    if bits(w, 30, 24) == 0b00_11011 {
        let ra = bits(w, 14, 10) as u8;
        let sub = bit(w, 15);
        return match bits(w, 23, 21) {
            0b000 if sub => Opcode::Msub { sf, rd, rn, rm, ra },
            0b000 => Opcode::Madd { sf, rd, rn, rm, ra },
            // The widening and high-half forms exist only at sf = 1;
            // SMULH/UMULH additionally fix Ra = 11111 (held to exactly
            // that, so an unallocated neighbor never decodes near-miss).
            0b001 if sf => Opcode::Maddl {
                signed: true,
                sub,
                rd,
                rn,
                rm,
                ra,
            },
            0b010 if sf && !sub && ra == 31 => Opcode::Mulh {
                signed: true,
                rd,
                rn,
                rm,
            },
            0b101 if sf => Opcode::Maddl {
                signed: false,
                sub,
                rd,
                rn,
                rm,
                ra,
            },
            0b110 if sf && !sub && ra == 31 => Opcode::Mulh {
                signed: false,
                rd,
                rn,
                rm,
            },
            _ => Opcode::Unknown(w),
        };
    }
    Opcode::Unknown(w)
}

/// Format a general-purpose register, with 31 as the zero register.
fn gp(sf: bool, r: u8) -> String {
    match (sf, r) {
        (true, 31) => "xzr".into(),
        (false, 31) => "wzr".into(),
        (true, _) => format!("x{r}"),
        (false, _) => format!("w{r}"),
    }
}

/// Format a general-purpose register, with 31 as the stack pointer.
fn gp_sp(sf: bool, r: u8) -> String {
    match (sf, r) {
        (true, 31) => "sp".into(),
        (false, 31) => "wsp".into(),
        _ => gp(sf, r),
    }
}

/// Format a signed immediate as `#0x..` / `#-0x..`.
fn fmt_imm(v: i64) -> String {
    if v < 0 {
        format!("#-{:#x}", v.unsigned_abs())
    } else {
        format!("#{v:#x}")
    }
}

/// Format a memory operand (`[base, #off]`, `[base, #off]!`, `[base], #off`).
fn fmt_mem(rn: u8, mode: AddrMode) -> String {
    let base = gp_sp(true, rn);
    match mode {
        AddrMode::Offset(0) => format!("[{base}]"),
        AddrMode::Offset(n) => format!("[{base}, {}]", fmt_imm(n)),
        AddrMode::PreIndex(n) => format!("[{base}, {}]!", fmt_imm(n)),
        AddrMode::PostIndex(n) => format!("[{base}], {}", fmt_imm(n)),
    }
}

/// Width suffix and register spelling for a sized load/store.
fn ls_reg(size: u8, rt: u8) -> (&'static str, String) {
    match size {
        0 => ("b", gp(false, rt)),
        1 => ("h", gp(false, rt)),
        2 => ("", gp(false, rt)),
        _ => ("", gp(true, rt)),
    }
}

/// SIMD&FP scalar register name for access `size` (log2 bytes):
/// `b0`…`q31`.
fn fp_reg(size: u8, r: u8) -> String {
    let class = ["b", "h", "s", "d", "q"][(size as usize).min(4)];
    format!("{class}{r}")
}

/// Arrangement specifier of a `MOVI`/`MVNI`/`FMOV` vector form, keyed on
/// the element size and the `Q` bit.
fn vec_arrangement(size: u8, q: bool) -> &'static str {
    match (size, q) {
        (0, false) => "8b",
        (0, true) => "16b",
        (1, false) => "4h",
        (1, true) => "8h",
        (2, false) => "2s",
        (2, true) => "4s",
        _ => "2d",
    }
}

/// The element-type letter of a vector element reference
/// (`v0.<t>[i]`), keyed on the element log2 width.
fn elem_type(size: u8) -> char {
    match size {
        0 => 'b',
        1 => 'h',
        2 => 's',
        _ => 'd',
    }
}

/// The `b`/`h`/`w` width suffix of a sign-extending load, keyed on the
/// access size.
fn ls_signed_suffix(size: u8) -> &'static str {
    match size {
        0 => "b",
        1 => "h",
        _ => "w",
    }
}

/// Format a register-offset memory operand
/// (`[base, Xm]`, `[base, Wm, uxtw #s]`, ...). The `lsl`/`uxtx` default with
/// no scaling is written as a bare `[base, Xm]`; every other extend, and any
/// scaled form, is spelled out.
fn fmt_reg_off(rn: u8, off: RegOffset, size: u8) -> String {
    let base = gp_sp(true, rn);
    // option low bit selects the index width: set is Xm, clear is Wm.
    let rm = gp(off.option & 1 == 1, off.rm);
    let ext = match off.option {
        0b010 => "uxtw",
        0b011 => "lsl",
        0b110 => "sxtw",
        _ => "sxtx",
    };
    if off.option == 0b011 && !off.scaled {
        format!("[{base}, {rm}]")
    } else if off.scaled {
        format!("[{base}, {rm}, {ext} #{size}]")
    } else {
        format!("[{base}, {rm}, {ext}]")
    }
}

/// Format the `, <shift> #<amount>` suffix of a shifted-register operand;
/// empty for the default `LSL #0`.
fn fmt_shift(shift: Shift, amount: u8) -> String {
    if shift == Shift::Lsl && amount == 0 {
        String::new()
    } else {
        format!(", {} #{amount}", shift.as_str())
    }
}

/// Extend name of an extended-register `option` field.
fn ext_name(option: u8) -> &'static str {
    [
        "uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx",
    ][(option & 7) as usize]
}

/// Format the `<R><m>{, <extend> {#amount}}` tail of an extended-register
/// add/sub. When SP is involved and the option is the width's identity
/// extend (UXTX at 64 bits, UXTW at 32), the preferred spelling is a
/// plain `LSL` — omitted entirely at amount 0.
fn fmt_ext(sf: bool, rm: u8, option: u8, amount: u8, sp_involved: bool) -> String {
    let rm = gp(sf && option & 0b011 == 0b011, rm);
    let identity = option == if sf { 0b011 } else { 0b010 };
    if identity && sp_involved {
        if amount == 0 {
            rm
        } else {
            format!("{rm}, lsl #{amount}")
        }
    } else if amount == 0 {
        format!("{rm}, {}", ext_name(option))
    } else {
        format!("{rm}, {} #{amount}", ext_name(option))
    }
}

/// Spell a `UBFM` with the alias the Arm ARM prefers (`LSL`/`LSR`,
/// `UXTB`/`UXTH`, then `UBFX`/`UBFIZ` — which cover the whole space).
fn fmt_ubfm(f: &mut fmt::Formatter<'_>, sf: bool, rd: u8, rn: u8, immr: u8, imms: u8) -> fmt::Result {
    let top = if sf { 63 } else { 31 };
    let (rd, rn) = (gp(sf, rd), gp(sf, rn));
    if imms == top {
        write!(f, "lsr {rd}, {rn}, #{immr}")
    } else if imms + 1 == immr {
        write!(f, "lsl {rd}, {rn}, #{}", top - imms)
    } else if !sf && immr == 0 && imms == 7 {
        write!(f, "uxtb {rd}, {rn}")
    } else if !sf && immr == 0 && imms == 15 {
        write!(f, "uxth {rd}, {rn}")
    } else if imms >= immr {
        write!(f, "ubfx {rd}, {rn}, #{immr}, #{}", imms - immr + 1)
    } else {
        write!(f, "ubfiz {rd}, {rn}, #{}, #{}", top + 1 - immr, imms + 1)
    }
}

/// Spell an `SBFM` with the preferred alias (`ASR`, `SXTB`/`SXTH`/`SXTW`
/// — whose source is spelled as a W register — then `SBFX`/`SBFIZ`).
fn fmt_sbfm(f: &mut fmt::Formatter<'_>, sf: bool, rd: u8, rn: u8, immr: u8, imms: u8) -> fmt::Result {
    let top = if sf { 63 } else { 31 };
    let rd = gp(sf, rd);
    if imms == top {
        return write!(f, "asr {rd}, {}, #{immr}", gp(sf, rn));
    }
    if immr == 0 {
        match imms {
            7 => return write!(f, "sxtb {rd}, {}", gp(false, rn)),
            15 => return write!(f, "sxth {rd}, {}", gp(false, rn)),
            31 if sf => return write!(f, "sxtw {rd}, {}", gp(false, rn)),
            _ => {}
        }
    }
    let rn = gp(sf, rn);
    if imms >= immr {
        write!(f, "sbfx {rd}, {rn}, #{immr}, #{}", imms - immr + 1)
    } else {
        write!(f, "sbfiz {rd}, {rn}, #{}, #{}", top + 1 - immr, imms + 1)
    }
}

/// Spell a `BFM` with the preferred alias (`BFXIL`, `BFI`, or — with the
/// zero register as source — `BFC`); the three cover the whole space.
fn fmt_bfm(f: &mut fmt::Formatter<'_>, sf: bool, rd: u8, rn: u8, immr: u8, imms: u8) -> fmt::Result {
    let top = if sf { 63 } else { 31 };
    let rd = gp(sf, rd);
    if imms >= immr {
        write!(f, "bfxil {rd}, {}, #{immr}, #{}", gp(sf, rn), imms - immr + 1)
    } else if rn == 31 {
        write!(f, "bfc {rd}, #{}, #{}", top + 1 - immr, imms + 1)
    } else {
        write!(f, "bfi {rd}, {}, #{}, #{}", gp(sf, rn), top + 1 - immr, imms + 1)
    }
}

impl fmt::Display for Instruction {
    /// Render in conventional assembler syntax, with pc-relative targets
    /// shown as absolute VAs and common aliases (`mov`, `cmp`, `cmn`)
    /// applied where assemblers would.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.opcode {
            Opcode::B { target } => write!(f, "b {target:#x}"),
            Opcode::Bl { target } => write!(f, "bl {target:#x}"),
            Opcode::BCond { cond, target } => write!(f, "b.{} {target:#x}", cond.as_str()),
            Opcode::Cbz { sf, rt, target } => write!(f, "cbz {}, {target:#x}", gp(sf, rt)),
            Opcode::Cbnz { sf, rt, target } => write!(f, "cbnz {}, {target:#x}", gp(sf, rt)),
            Opcode::Tbz { rt, bit, target } => {
                write!(f, "tbz {}, #{bit}, {target:#x}", gp(bit >= 32, rt))
            }
            Opcode::Tbnz { rt, bit, target } => {
                write!(f, "tbnz {}, #{bit}, {target:#x}", gp(bit >= 32, rt))
            }
            Opcode::Br { rn } => write!(f, "br {}", gp(true, rn)),
            Opcode::Blr { rn } => write!(f, "blr {}", gp(true, rn)),
            Opcode::Ret { rn: 30 } => write!(f, "ret"),
            Opcode::Ret { rn } => write!(f, "ret {}", gp(true, rn)),
            Opcode::Svc { imm } => write!(f, "svc #{imm:#x}"),
            Opcode::Hvc { imm } => write!(f, "hvc #{imm:#x}"),
            Opcode::Smc { imm } => write!(f, "smc #{imm:#x}"),
            Opcode::Brk { imm } => write!(f, "brk #{imm:#x}"),
            Opcode::Hlt { imm } => write!(f, "hlt #{imm:#x}"),
            Opcode::Adr { rd, target } => write!(f, "adr {}, {target:#x}", gp(true, rd)),
            Opcode::Adrp { rd, target } => write!(f, "adrp {}, {target:#x}", gp(true, rd)),
            Opcode::AddImm {
                sf,
                set_flags,
                rd,
                rn,
                imm,
            } => {
                if !set_flags && imm == 0 && (rd == 31 || rn == 31) {
                    write!(f, "mov {}, {}", gp_sp(sf, rd), gp_sp(sf, rn))
                } else if set_flags && rd == 31 {
                    write!(f, "cmn {}, #{imm:#x}", gp_sp(sf, rn))
                } else {
                    let mn = if set_flags { "adds" } else { "add" };
                    let rd = if set_flags { gp(sf, rd) } else { gp_sp(sf, rd) };
                    write!(f, "{mn} {rd}, {}, #{imm:#x}", gp_sp(sf, rn))
                }
            }
            Opcode::SubImm {
                sf,
                set_flags,
                rd,
                rn,
                imm,
            } => {
                if set_flags && rd == 31 {
                    write!(f, "cmp {}, #{imm:#x}", gp_sp(sf, rn))
                } else {
                    let mn = if set_flags { "subs" } else { "sub" };
                    let rd = if set_flags { gp(sf, rd) } else { gp_sp(sf, rd) };
                    write!(f, "{mn} {rd}, {}, #{imm:#x}", gp_sp(sf, rn))
                }
            }
            Opcode::Movn { sf, rd, imm, shift }
            | Opcode::Movz { sf, rd, imm, shift }
            | Opcode::Movk { sf, rd, imm, shift } => {
                let mn = match self.opcode {
                    Opcode::Movn { .. } => "movn",
                    Opcode::Movz { .. } => "movz",
                    _ => "movk",
                };
                write!(f, "{mn} {}, #{imm:#x}", gp(sf, rd))?;
                if shift != 0 {
                    write!(f, ", lsl #{shift}")?;
                }
                Ok(())
            }
            Opcode::Csel {
                sf,
                rd,
                rn,
                rm,
                cond,
            } => write!(
                f,
                "csel {}, {}, {}, {}",
                gp(sf, rd),
                gp(sf, rn),
                gp(sf, rm),
                cond.as_str()
            ),
            Opcode::Csinc {
                sf,
                rd,
                rn,
                rm,
                cond,
            } => {
                if !cond.is_al_nv() && rn == 31 && rm == 31 {
                    write!(f, "cset {}, {}", gp(sf, rd), cond.invert().as_str())
                } else if !cond.is_al_nv() && rn == rm {
                    write!(
                        f,
                        "cinc {}, {}, {}",
                        gp(sf, rd),
                        gp(sf, rn),
                        cond.invert().as_str()
                    )
                } else {
                    write!(
                        f,
                        "csinc {}, {}, {}, {}",
                        gp(sf, rd),
                        gp(sf, rn),
                        gp(sf, rm),
                        cond.as_str()
                    )
                }
            }
            Opcode::Csinv {
                sf,
                rd,
                rn,
                rm,
                cond,
            } => {
                if !cond.is_al_nv() && rn == 31 && rm == 31 {
                    write!(f, "csetm {}, {}", gp(sf, rd), cond.invert().as_str())
                } else if !cond.is_al_nv() && rn == rm {
                    write!(
                        f,
                        "cinv {}, {}, {}",
                        gp(sf, rd),
                        gp(sf, rn),
                        cond.invert().as_str()
                    )
                } else {
                    write!(
                        f,
                        "csinv {}, {}, {}, {}",
                        gp(sf, rd),
                        gp(sf, rn),
                        gp(sf, rm),
                        cond.as_str()
                    )
                }
            }
            Opcode::Csneg {
                sf,
                rd,
                rn,
                rm,
                cond,
            } => {
                if !cond.is_al_nv() && rn == rm {
                    write!(
                        f,
                        "cneg {}, {}, {}",
                        gp(sf, rd),
                        gp(sf, rn),
                        cond.invert().as_str()
                    )
                } else {
                    write!(
                        f,
                        "csneg {}, {}, {}, {}",
                        gp(sf, rd),
                        gp(sf, rn),
                        gp(sf, rm),
                        cond.as_str()
                    )
                }
            }
            Opcode::CcmpReg {
                sf,
                sub,
                rn,
                rm,
                nzcv,
                cond,
            } => {
                let mn = if sub { "ccmp" } else { "ccmn" };
                write!(
                    f,
                    "{mn} {}, {}, #{nzcv:#x}, {}",
                    gp(sf, rn),
                    gp(sf, rm),
                    cond.as_str()
                )
            }
            Opcode::CcmpImm {
                sf,
                sub,
                rn,
                imm,
                nzcv,
                cond,
            } => {
                let mn = if sub { "ccmp" } else { "ccmn" };
                write!(
                    f,
                    "{mn} {}, #{imm:#x}, #{nzcv:#x}, {}",
                    gp(sf, rn),
                    cond.as_str()
                )
            }
            Opcode::Adc {
                sf,
                set_flags,
                rd,
                rn,
                rm,
            } => {
                let mn = if set_flags { "adcs" } else { "adc" };
                write!(f, "{mn} {}, {}, {}", gp(sf, rd), gp(sf, rn), gp(sf, rm))
            }
            Opcode::Sbc {
                sf,
                set_flags,
                rd,
                rn,
                rm,
            } => {
                if rn == 31 {
                    let mn = if set_flags { "ngcs" } else { "ngc" };
                    write!(f, "{mn} {}, {}", gp(sf, rd), gp(sf, rm))
                } else {
                    let mn = if set_flags { "sbcs" } else { "sbc" };
                    write!(f, "{mn} {}, {}, {}", gp(sf, rd), gp(sf, rn), gp(sf, rm))
                }
            }
            Opcode::AddReg {
                sf,
                set_flags,
                rd,
                rn,
                rm,
                shift,
                amount,
            }
            | Opcode::SubReg {
                sf,
                set_flags,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let sub = matches!(self.opcode, Opcode::SubReg { .. });
                let sh = fmt_shift(shift, amount);
                if set_flags && rd == 31 {
                    let mn = if sub { "cmp" } else { "cmn" };
                    write!(f, "{mn} {}, {}{sh}", gp(sf, rn), gp(sf, rm))
                } else if sub && rn == 31 {
                    let mn = if set_flags { "negs" } else { "neg" };
                    write!(f, "{mn} {}, {}{sh}", gp(sf, rd), gp(sf, rm))
                } else {
                    let mn = match (sub, set_flags) {
                        (false, false) => "add",
                        (false, true) => "adds",
                        (true, false) => "sub",
                        (true, true) => "subs",
                    };
                    write!(f, "{mn} {}, {}, {}{sh}", gp(sf, rd), gp(sf, rn), gp(sf, rm))
                }
            }
            Opcode::AddExt {
                sf,
                set_flags,
                rd,
                rn,
                rm,
                option,
                amount,
            }
            | Opcode::SubExt {
                sf,
                set_flags,
                rd,
                rn,
                rm,
                option,
                amount,
            } => {
                let sub = matches!(self.opcode, Opcode::SubExt { .. });
                let rn_s = gp_sp(sf, rn);
                if set_flags && rd == 31 {
                    let tail = fmt_ext(sf, rm, option, amount, rn == 31);
                    let mn = if sub { "cmp" } else { "cmn" };
                    write!(f, "{mn} {rn_s}, {tail}")
                } else {
                    let mn = match (sub, set_flags) {
                        (false, false) => "add",
                        (false, true) => "adds",
                        (true, false) => "sub",
                        (true, true) => "subs",
                    };
                    let sp_involved = (!set_flags && rd == 31) || rn == 31;
                    let rd_s = if set_flags { gp(sf, rd) } else { gp_sp(sf, rd) };
                    let tail = fmt_ext(sf, rm, option, amount, sp_involved);
                    write!(f, "{mn} {rd_s}, {rn_s}, {tail}")
                }
            }
            Opcode::LogReg {
                sf,
                op,
                set_flags,
                invert,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let sh = fmt_shift(shift, amount);
                if set_flags && !invert && rd == 31 {
                    write!(f, "tst {}, {}{sh}", gp(sf, rn), gp(sf, rm))
                } else if op == LogOp::Orr && !invert && rn == 31 && sh.is_empty() {
                    write!(f, "mov {}, {}", gp(sf, rd), gp(sf, rm))
                } else if op == LogOp::Orr && invert && rn == 31 {
                    write!(f, "mvn {}, {}{sh}", gp(sf, rd), gp(sf, rm))
                } else {
                    let mn = match (op, invert, set_flags) {
                        (LogOp::And, false, false) => "and",
                        (LogOp::And, true, false) => "bic",
                        (LogOp::And, false, true) => "ands",
                        (LogOp::And, true, true) => "bics",
                        (LogOp::Orr, false, _) => "orr",
                        (LogOp::Orr, true, _) => "orn",
                        (LogOp::Eor, false, _) => "eor",
                        (LogOp::Eor, true, _) => "eon",
                    };
                    write!(f, "{mn} {}, {}, {}{sh}", gp(sf, rd), gp(sf, rn), gp(sf, rm))
                }
            }
            Opcode::LogImm {
                sf,
                op,
                set_flags,
                rd,
                rn,
                imm,
            } => {
                if set_flags && rd == 31 {
                    write!(f, "tst {}, #{imm:#x}", gp(sf, rn))
                } else if op == LogOp::Orr && rn == 31 {
                    write!(f, "mov {}, #{imm:#x}", gp_sp(sf, rd))
                } else {
                    let mn = match (op, set_flags) {
                        (LogOp::And, false) => "and",
                        (LogOp::And, true) => "ands",
                        (LogOp::Orr, _) => "orr",
                        (LogOp::Eor, _) => "eor",
                    };
                    let rd = if set_flags { gp(sf, rd) } else { gp_sp(sf, rd) };
                    write!(f, "{mn} {rd}, {}, #{imm:#x}", gp(sf, rn))
                }
            }
            Opcode::Sbfm {
                sf,
                rd,
                rn,
                immr,
                imms,
            } => fmt_sbfm(f, sf, rd, rn, immr, imms),
            Opcode::Bfm {
                sf,
                rd,
                rn,
                immr,
                imms,
            } => fmt_bfm(f, sf, rd, rn, immr, imms),
            Opcode::Ubfm {
                sf,
                rd,
                rn,
                immr,
                imms,
            } => fmt_ubfm(f, sf, rd, rn, immr, imms),
            Opcode::ShiftReg {
                sf,
                kind,
                rd,
                rn,
                rm,
            } => write!(
                f,
                "{} {}, {}, {}",
                kind.as_str(),
                gp(sf, rd),
                gp(sf, rn),
                gp(sf, rm)
            ),
            Opcode::Udiv { sf, rd, rn, rm } => {
                write!(f, "udiv {}, {}, {}", gp(sf, rd), gp(sf, rn), gp(sf, rm))
            }
            Opcode::Sdiv { sf, rd, rn, rm } => {
                write!(f, "sdiv {}, {}, {}", gp(sf, rd), gp(sf, rn), gp(sf, rm))
            }
            Opcode::Madd { sf, rd, rn, rm, ra } => {
                if ra == 31 {
                    write!(f, "mul {}, {}, {}", gp(sf, rd), gp(sf, rn), gp(sf, rm))
                } else {
                    write!(
                        f,
                        "madd {}, {}, {}, {}",
                        gp(sf, rd),
                        gp(sf, rn),
                        gp(sf, rm),
                        gp(sf, ra)
                    )
                }
            }
            Opcode::Msub { sf, rd, rn, rm, ra } => {
                if ra == 31 {
                    write!(f, "mneg {}, {}, {}", gp(sf, rd), gp(sf, rn), gp(sf, rm))
                } else {
                    write!(
                        f,
                        "msub {}, {}, {}, {}",
                        gp(sf, rd),
                        gp(sf, rn),
                        gp(sf, rm),
                        gp(sf, ra)
                    )
                }
            }
            Opcode::Maddl {
                signed,
                sub,
                rd,
                rn,
                rm,
                ra,
            } => {
                let (rd, rn, rm) = (gp(true, rd), gp(false, rn), gp(false, rm));
                if ra == 31 {
                    let mn = match (signed, sub) {
                        (true, false) => "smull",
                        (true, true) => "smnegl",
                        (false, false) => "umull",
                        (false, true) => "umnegl",
                    };
                    write!(f, "{mn} {rd}, {rn}, {rm}")
                } else {
                    let mn = match (signed, sub) {
                        (true, false) => "smaddl",
                        (true, true) => "smsubl",
                        (false, false) => "umaddl",
                        (false, true) => "umsubl",
                    };
                    write!(f, "{mn} {rd}, {rn}, {rm}, {}", gp(true, ra))
                }
            }
            Opcode::Mulh { signed, rd, rn, rm } => {
                let mn = if signed { "smulh" } else { "umulh" };
                write!(f, "{mn} {}, {}, {}", gp(true, rd), gp(true, rn), gp(true, rm))
            }
            Opcode::Ldr { size, rt, rn, mode } => {
                let (sfx, reg) = ls_reg(size, rt);
                write!(f, "ldr{sfx} {reg}, {}", fmt_mem(rn, mode))
            }
            Opcode::Str { size, rt, rn, mode } => {
                let (sfx, reg) = ls_reg(size, rt);
                write!(f, "str{sfx} {reg}, {}", fmt_mem(rn, mode))
            }
            Opcode::Ldrs {
                size,
                sf,
                rt,
                rn,
                mode,
            } => write!(
                f,
                "ldrs{} {}, {}",
                ls_signed_suffix(size),
                gp(sf, rt),
                fmt_mem(rn, mode)
            ),
            Opcode::Ldur { size, rt, rn, imm } => {
                let (sfx, reg) = ls_reg(size, rt);
                write!(f, "ldur{sfx} {reg}, {}", fmt_mem(rn, AddrMode::Offset(imm)))
            }
            Opcode::Stur { size, rt, rn, imm } => {
                let (sfx, reg) = ls_reg(size, rt);
                write!(f, "stur{sfx} {reg}, {}", fmt_mem(rn, AddrMode::Offset(imm)))
            }
            Opcode::Ldurs {
                size,
                sf,
                rt,
                rn,
                imm,
            } => write!(
                f,
                "ldurs{} {}, {}",
                ls_signed_suffix(size),
                gp(sf, rt),
                fmt_mem(rn, AddrMode::Offset(imm))
            ),
            Opcode::LdrReg { size, rt, rn, off } => {
                let (sfx, reg) = ls_reg(size, rt);
                write!(f, "ldr{sfx} {reg}, {}", fmt_reg_off(rn, off, size))
            }
            Opcode::StrReg { size, rt, rn, off } => {
                let (sfx, reg) = ls_reg(size, rt);
                write!(f, "str{sfx} {reg}, {}", fmt_reg_off(rn, off, size))
            }
            Opcode::LdrsReg {
                size,
                sf,
                rt,
                rn,
                off,
            } => write!(
                f,
                "ldrs{} {}, {}",
                ls_signed_suffix(size),
                gp(sf, rt),
                fmt_reg_off(rn, off, size)
            ),
            Opcode::LdrLit { sf, rt, target } => write!(f, "ldr {}, {target:#x}", gp(sf, rt)),
            Opcode::Ldp {
                sf,
                rt,
                rt2,
                rn,
                mode,
            } => write!(
                f,
                "ldp {}, {}, {}",
                gp(sf, rt),
                gp(sf, rt2),
                fmt_mem(rn, mode)
            ),
            Opcode::Stp {
                sf,
                rt,
                rt2,
                rn,
                mode,
            } => write!(
                f,
                "stp {}, {}, {}",
                gp(sf, rt),
                gp(sf, rt2),
                fmt_mem(rn, mode)
            ),
            Opcode::FLdr { size, rt, rn, mode } => {
                write!(f, "ldr {}, {}", fp_reg(size, rt), fmt_mem(rn, mode))
            }
            Opcode::FStr { size, rt, rn, mode } => {
                write!(f, "str {}, {}", fp_reg(size, rt), fmt_mem(rn, mode))
            }
            Opcode::FLdur { size, rt, rn, imm } => write!(
                f,
                "ldur {}, {}",
                fp_reg(size, rt),
                fmt_mem(rn, AddrMode::Offset(imm))
            ),
            Opcode::FStur { size, rt, rn, imm } => write!(
                f,
                "stur {}, {}",
                fp_reg(size, rt),
                fmt_mem(rn, AddrMode::Offset(imm))
            ),
            Opcode::FLdrReg { size, rt, rn, off } => {
                write!(f, "ldr {}, {}", fp_reg(size, rt), fmt_reg_off(rn, off, size))
            }
            Opcode::FStrReg { size, rt, rn, off } => {
                write!(f, "str {}, {}", fp_reg(size, rt), fmt_reg_off(rn, off, size))
            }
            Opcode::FLdrLit { size, rt, target } => {
                write!(f, "ldr {}, {target:#x}", fp_reg(size, rt))
            }
            Opcode::FLdp {
                size,
                rt,
                rt2,
                rn,
                mode,
            } => write!(
                f,
                "ldp {}, {}, {}",
                fp_reg(size, rt),
                fp_reg(size, rt2),
                fmt_mem(rn, mode)
            ),
            Opcode::FStp {
                size,
                rt,
                rt2,
                rn,
                mode,
            } => write!(
                f,
                "stp {}, {}, {}",
                fp_reg(size, rt),
                fp_reg(size, rt2),
                fmt_mem(rn, mode)
            ),
            Opcode::FmovReg { double, rd, rn } => {
                let size = if double { 3 } else { 2 };
                write!(f, "fmov {}, {}", fp_reg(size, rd), fp_reg(size, rn))
            }
            Opcode::FmovToGp { sf, hi, rd, rn } => {
                if hi {
                    write!(f, "fmov {}, v{rn}.d[1]", gp(sf, rd))
                } else {
                    write!(f, "fmov {}, {}", gp(sf, rd), fp_reg(if sf { 3 } else { 2 }, rn))
                }
            }
            Opcode::FmovFromGp { sf, hi, rd, rn } => {
                if hi {
                    write!(f, "fmov v{rd}.d[1], {}", gp(sf, rn))
                } else {
                    write!(f, "fmov {}, {}", fp_reg(if sf { 3 } else { 2 }, rd), gp(sf, rn))
                }
            }
            Opcode::FmovImm { double, imm, rd } => {
                let size = if double { 3 } else { 2 };
                write!(f, "fmov {}, #{:?}", fp_reg(size, rd), fp_imm_value(imm))
            }
            Opcode::FmovVecImm { q, double, imm, rd } => {
                let arr = if double { "2d" } else { vec_arrangement(2, q) };
                write!(f, "fmov v{rd}.{arr}, #{:?}", fp_imm_value(imm))
            }
            Opcode::Movi {
                q,
                invert,
                size,
                imm,
                shift,
                msl,
                rd,
            } => {
                if size == 3 {
                    // The 64-bit form spells out the expanded byte mask;
                    // Q = 0 writes the D register.
                    let imm64 = movi_expand(3, imm, 0, false, false);
                    return if q {
                        write!(f, "movi v{rd}.2d, #{imm64:#x}")
                    } else {
                        write!(f, "movi d{rd}, #{imm64:#x}")
                    };
                }
                let mn = if invert { "mvni" } else { "movi" };
                write!(f, "{mn} v{rd}.{}, #{imm:#x}", vec_arrangement(size, q))?;
                if msl {
                    write!(f, ", msl #{shift}")?;
                } else if shift != 0 {
                    write!(f, ", lsl #{shift}")?;
                }
                Ok(())
            }
            Opcode::FArith2 {
                op,
                double,
                rd,
                rn,
                rm,
            } => {
                let s = if double { 3 } else { 2 };
                write!(
                    f,
                    "{} {}, {}, {}",
                    op.as_str(),
                    fp_reg(s, rd),
                    fp_reg(s, rn),
                    fp_reg(s, rm)
                )
            }
            Opcode::FArith3 {
                negate,
                sub,
                double,
                rd,
                rn,
                rm,
                ra,
            } => {
                let mn = match (negate, sub) {
                    (false, false) => "fmadd",
                    (false, true) => "fmsub",
                    (true, false) => "fnmadd",
                    (true, true) => "fnmsub",
                };
                let s = if double { 3 } else { 2 };
                write!(
                    f,
                    "{mn} {}, {}, {}, {}",
                    fp_reg(s, rd),
                    fp_reg(s, rn),
                    fp_reg(s, rm),
                    fp_reg(s, ra)
                )
            }
            Opcode::FArith1 {
                op,
                double,
                rd,
                rn,
            } => {
                let s = if double { 3 } else { 2 };
                write!(f, "{} {}, {}", op.as_str(), fp_reg(s, rd), fp_reg(s, rn))
            }
            Opcode::FCvtPrec { to_double, rd, rn } => {
                let (d, n) = if to_double { (3, 2) } else { (2, 3) };
                write!(f, "fcvt {}, {}", fp_reg(d, rd), fp_reg(n, rn))
            }
            Opcode::Fcmp {
                double,
                signal,
                rn,
                rm,
            } => {
                let mn = if signal { "fcmpe" } else { "fcmp" };
                let s = if double { 3 } else { 2 };
                match rm {
                    Some(rm) => write!(f, "{mn} {}, {}", fp_reg(s, rn), fp_reg(s, rm)),
                    None => write!(f, "{mn} {}, #0.0", fp_reg(s, rn)),
                }
            }
            Opcode::Fccmp {
                double,
                signal,
                rn,
                rm,
                nzcv,
                cond,
            } => {
                let mn = if signal { "fccmpe" } else { "fccmp" };
                let s = if double { 3 } else { 2 };
                write!(
                    f,
                    "{mn} {}, {}, #{nzcv:#x}, {}",
                    fp_reg(s, rn),
                    fp_reg(s, rm),
                    cond.as_str()
                )
            }
            Opcode::Fcsel {
                double,
                rd,
                rn,
                rm,
                cond,
            } => {
                let s = if double { 3 } else { 2 };
                write!(
                    f,
                    "fcsel {}, {}, {}, {}",
                    fp_reg(s, rd),
                    fp_reg(s, rn),
                    fp_reg(s, rm),
                    cond.as_str()
                )
            }
            Opcode::FcvtToFp {
                sf,
                double,
                unsigned,
                rd,
                rn,
            } => {
                let mn = if unsigned { "ucvtf" } else { "scvtf" };
                write!(
                    f,
                    "{mn} {}, {}",
                    fp_reg(if double { 3 } else { 2 }, rd),
                    gp(sf, rn)
                )
            }
            Opcode::FcvtFromFp {
                sf,
                double,
                unsigned,
                round,
                rd,
                rn,
            } => {
                let su = if unsigned { 'u' } else { 's' };
                write!(
                    f,
                    "fcvt{}{su} {}, {}",
                    round.letter(),
                    gp(sf, rd),
                    fp_reg(if double { 3 } else { 2 }, rn)
                )
            }
            Opcode::FcvtIntScalar {
                double,
                unsigned,
                rd,
                rn,
            } => {
                let mn = if unsigned { "ucvtf" } else { "scvtf" };
                let s = if double { 3 } else { 2 };
                write!(f, "{mn} {}, {}", fp_reg(s, rd), fp_reg(s, rn))
            }
            Opcode::DupGp { q, size, rd, rn } => write!(
                f,
                "dup v{rd}.{}, {}",
                vec_arrangement(size, q),
                gp(size == 3, rn)
            ),
            Opcode::DupElemScalar {
                size,
                index,
                rd,
                rn,
            } => write!(
                f,
                "mov {}, v{rn}.{}[{index}]",
                fp_reg(size, rd),
                elem_type(size)
            ),
            Opcode::DupElemVec {
                q,
                size,
                index,
                rd,
                rn,
            } => write!(
                f,
                "dup v{rd}.{}, v{rn}.{}[{index}]",
                vec_arrangement(size, q),
                elem_type(size)
            ),
            Opcode::Umov {
                sf,
                size,
                index,
                rd,
                rn,
            } => write!(
                f,
                "umov {}, v{rn}.{}[{index}]",
                gp(sf, rd),
                elem_type(size)
            ),
            Opcode::Smov {
                sf,
                size,
                index,
                rd,
                rn,
            } => write!(
                f,
                "smov {}, v{rn}.{}[{index}]",
                gp(sf, rd),
                elem_type(size)
            ),
            Opcode::InsGp {
                size,
                index,
                rd,
                rn,
            } => write!(
                f,
                "ins v{rd}.{}[{index}], {}",
                elem_type(size),
                gp(size == 3, rn)
            ),
            Opcode::InsElem {
                size,
                dst,
                src,
                rd,
                rn,
            } => {
                let t = elem_type(size);
                write!(f, "ins v{rd}.{t}[{dst}], v{rn}.{t}[{src}]")
            }
            Opcode::SimdAlu {
                op,
                q,
                size,
                rd,
                rn,
                rm,
            } => write!(
                f,
                "{} v{rd}.{}, v{rn}.{}, v{rm}.{}",
                op.as_str(),
                vec_arrangement(size, q),
                vec_arrangement(size, q),
                vec_arrangement(size, q)
            ),
            Opcode::Ldar { size, rt, rn } => {
                let (sfx, r) = ls_reg(size, rt);
                write!(f, "ldar{sfx} {r}, {}", fmt_mem(rn, AddrMode::Offset(0)))
            }
            Opcode::Stlr { size, rt, rn } => {
                let (sfx, r) = ls_reg(size, rt);
                write!(f, "stlr{sfx} {r}, {}", fmt_mem(rn, AddrMode::Offset(0)))
            }
            Opcode::Ldxr {
                size,
                acquire,
                rt,
                rn,
            } => {
                let mn = if acquire { "ldaxr" } else { "ldxr" };
                let (sfx, r) = ls_reg(size, rt);
                write!(f, "{mn}{sfx} {r}, {}", fmt_mem(rn, AddrMode::Offset(0)))
            }
            Opcode::Stxr {
                size,
                release,
                ws,
                rt,
                rn,
            } => {
                let mn = if release { "stlxr" } else { "stxr" };
                let (sfx, r) = ls_reg(size, rt);
                write!(
                    f,
                    "{mn}{sfx} {}, {r}, {}",
                    gp(false, ws),
                    fmt_mem(rn, AddrMode::Offset(0))
                )
            }
            Opcode::RetA { key_b } => write!(f, "reta{}", if key_b { 'b' } else { 'a' }),
            Opcode::BrAuth {
                link,
                key_b,
                zero,
                rn,
                rm,
            } => {
                let bl = if link { "blr" } else { "br" };
                let k = if key_b { 'b' } else { 'a' };
                if zero {
                    write!(f, "{bl}a{k}z {}", gp(true, rn))
                } else {
                    write!(f, "{bl}a{k} {}, {}", gp(true, rn), gp(true, rm))
                }
            }
            Opcode::PacGpr {
                auth,
                key_b,
                zero,
                rd,
                rn,
            } => {
                let pa = if auth { "aut" } else { "pac" };
                let k = if key_b { 'b' } else { 'a' };
                if zero {
                    write!(f, "{pa}iz{k} {}", gp(true, rd))
                } else {
                    write!(f, "{pa}i{k} {}, {}", gp(true, rd), gp(true, rn))
                }
            }
            Opcode::XPac { data, rd } => {
                write!(f, "xpac{} {}", if data { 'd' } else { 'i' }, gp(true, rd))
            }
            Opcode::PacHint { auth, key_b } => write!(
                f,
                "{}i{}sp",
                if auth { "aut" } else { "pac" },
                if key_b { 'b' } else { 'a' }
            ),
            Opcode::Udf { imm } => write!(f, "udf #{imm:#x}"),
            Opcode::Bits1 { op, sf, rd, rn } => {
                write!(f, "{} {}, {}", op.as_str(), gp(sf, rd), gp(sf, rn))
            }
            Opcode::Extr {
                sf,
                rd,
                rn,
                rm,
                lsb,
            } => {
                if rn == rm {
                    write!(f, "ror {}, {}, #{lsb}", gp(sf, rd), gp(sf, rn))
                } else {
                    write!(
                        f,
                        "extr {}, {}, {}, #{lsb}",
                        gp(sf, rd),
                        gp(sf, rn),
                        gp(sf, rm)
                    )
                }
            }
            Opcode::LdpSw { rt, rt2, rn, mode } => write!(
                f,
                "ldpsw {}, {}, {}",
                gp(true, rt),
                gp(true, rt2),
                fmt_mem(rn, mode)
            ),
            Opcode::Nop => write!(f, "nop"),
            Opcode::Yield => write!(f, "yield"),
            Opcode::Wfe => write!(f, "wfe"),
            Opcode::Wfi => write!(f, "wfi"),
            Opcode::Sev => write!(f, "sev"),
            Opcode::Sevl => write!(f, "sevl"),
            Opcode::Hint { imm } => write!(f, "hint #{imm:#x}"),
            Opcode::Unknown(raw) => write!(f, ".inst {raw:#010x}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode one hand-encoded word (built from the Arm ARM bit layouts)
    /// at virtual address `va`.
    fn ins(word: u32, va: u64) -> Instruction {
        decode(&word.to_le_bytes(), va).unwrap()
    }

    const VA: u64 = 0x1000;

    #[test]
    fn b_forward_and_backward() {
        // B +0x10: imm26 = 4.
        let i = ins(0x1400_0004, VA);
        assert_eq!(i.opcode, Opcode::B { target: 0x1010 });
        assert_eq!(i.flow, Flow::Jump(0x1010));
        assert_eq!(i.to_string(), "b 0x1010");
        // B -0x10: imm26 = -4, sign-extended across the full 26 bits.
        let i = ins(0x17FF_FFFC, VA);
        assert_eq!(i.opcode, Opcode::B { target: 0xFF0 });
        assert_eq!(i.flow, Flow::Jump(0xFF0));
    }

    #[test]
    fn bl_forward_and_backward() {
        let i = ins(0x9400_0004, VA);
        assert_eq!(i.opcode, Opcode::Bl { target: 0x1010 });
        assert_eq!(i.flow, Flow::Call(0x1010));
        assert_eq!(i.to_string(), "bl 0x1010");
        let i = ins(0x97FF_FFFC, VA);
        assert_eq!(i.flow, Flow::Call(0xFF0));
    }

    #[test]
    fn b_cond_forward_and_backward() {
        // B.EQ +0x20: imm19 = 8, cond = 0.
        let i = ins(0x5400_0100, VA);
        assert_eq!(
            i.opcode,
            Opcode::BCond {
                cond: Cond::Eq,
                target: 0x1020
            }
        );
        assert_eq!(i.flow, Flow::CondJump(0x1020));
        assert_eq!(i.to_string(), "b.eq 0x1020");
        // B.NE -4: imm19 = -1, cond = 1.
        let i = ins(0x54FF_FFE1, VA);
        assert_eq!(
            i.opcode,
            Opcode::BCond {
                cond: Cond::Ne,
                target: 0xFFC
            }
        );
        assert_eq!(i.flow, Flow::CondJump(0xFFC));
        // Bit 4 set (BC.cond) is outside the modeled subset.
        assert_eq!(ins(0x5400_0110, VA).opcode, Opcode::Unknown(0x5400_0110));
    }

    #[test]
    fn cbz_and_cbnz() {
        // CBZ w0, +8: sf = 0, imm19 = 2.
        let i = ins(0x3400_0040, VA);
        assert_eq!(
            i.opcode,
            Opcode::Cbz {
                sf: false,
                rt: 0,
                target: 0x1008
            }
        );
        assert_eq!(i.flow, Flow::CondJump(0x1008));
        assert_eq!(i.to_string(), "cbz w0, 0x1008");
        // CBNZ x5, -8: sf = 1, imm19 = -2.
        let i = ins(0xB5FF_FFC5, VA);
        assert_eq!(
            i.opcode,
            Opcode::Cbnz {
                sf: true,
                rt: 5,
                target: 0xFF8
            }
        );
        assert_eq!(i.flow, Flow::CondJump(0xFF8));
        assert_eq!(i.to_string(), "cbnz x5, 0xff8");
    }

    #[test]
    fn tbz_bit_number_crosses_the_b5_boundary() {
        // TBZ x3, #33, +16: b5 = 1, b40 = 00001, imm14 = 4.
        let i = ins(0xB608_0083, VA);
        assert_eq!(
            i.opcode,
            Opcode::Tbz {
                rt: 3,
                bit: 33,
                target: 0x1010
            }
        );
        assert_eq!(i.flow, Flow::CondJump(0x1010));
        assert_eq!(i.to_string(), "tbz x3, #33, 0x1010");
        // TBNZ w2, #5, -8: b5 = 0, b40 = 00101, imm14 = -2.
        let i = ins(0x372F_FFC2, VA);
        assert_eq!(
            i.opcode,
            Opcode::Tbnz {
                rt: 2,
                bit: 5,
                target: 0xFF8
            }
        );
        assert_eq!(i.flow, Flow::CondJump(0xFF8));
        assert_eq!(i.to_string(), "tbnz w2, #5, 0xff8");
    }

    #[test]
    fn register_branches() {
        let i = ins(0xD61F_0200, VA); // BR x16
        assert_eq!(i.opcode, Opcode::Br { rn: 16 });
        assert_eq!(i.flow, Flow::IndirectJump);
        assert_eq!(i.to_string(), "br x16");

        let i = ins(0xD63F_0100, VA); // BLR x8
        assert_eq!(i.opcode, Opcode::Blr { rn: 8 });
        assert_eq!(i.flow, Flow::IndirectCall);
        assert_eq!(i.to_string(), "blr x8");

        let i = ins(0xD65F_03C0, VA); // RET (x30)
        assert_eq!(i.opcode, Opcode::Ret { rn: 30 });
        assert_eq!(i.flow, Flow::Return);
        assert_eq!(i.to_string(), "ret");

        let i = ins(0xD65F_0020, VA); // RET x1
        assert_eq!(i.opcode, Opcode::Ret { rn: 1 });
        assert_eq!(i.to_string(), "ret x1");

        // ERET stays a documented gap: Unknown, not misdecoded.
        assert_eq!(ins(0xD69F_03E0, VA).opcode, Opcode::Unknown(0xD69F_03E0));
    }

    #[test]
    fn exception_generation() {
        let i = ins(0xD400_0001, VA); // SVC #0
        assert_eq!(i.opcode, Opcode::Svc { imm: 0 });
        assert_eq!(i.flow, Flow::Interrupt);
        assert_eq!(i.to_string(), "svc #0x0");

        assert_eq!(ins(0xD400_1001, VA).opcode, Opcode::Svc { imm: 0x80 });
        assert_eq!(ins(0xD400_0022, VA).opcode, Opcode::Hvc { imm: 1 });
        assert_eq!(ins(0xD400_0043, VA).opcode, Opcode::Smc { imm: 2 });

        let i = ins(0xD43E_0000, VA); // BRK #0xF000
        assert_eq!(i.opcode, Opcode::Brk { imm: 0xF000 });
        assert_eq!(i.flow, Flow::Interrupt);
        assert_eq!(i.to_string(), "brk #0xf000");

        let i = ins(0xD440_0000, VA); // HLT #0
        assert_eq!(i.opcode, Opcode::Hlt { imm: 0 });
        assert_eq!(i.flow, Flow::Halt);

        // Unallocated opc/LL combination stays Unknown.
        assert_eq!(ins(0xD400_0000, VA).opcode, Opcode::Unknown(0xD400_0000));
    }

    #[test]
    fn adr_forward_and_backward() {
        // ADR x0, +0x10: immhi = 4, immlo = 0.
        let i = ins(0x1000_0080, VA);
        assert_eq!(i.opcode, Opcode::Adr { rd: 0, target: 0x1010 });
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "adr x0, 0x1010");
        // ADR x1, -4: imm21 = -4.
        let i = ins(0x10FF_FFE1, VA);
        assert_eq!(i.opcode, Opcode::Adr { rd: 1, target: 0xFFC });
    }

    #[test]
    fn adrp_computes_page_addresses() {
        // ADRP x1, +1 page, decoded mid-page: the pc's low 12 bits drop.
        let i = ins(0xB000_0001, 0x1234);
        assert_eq!(i.opcode, Opcode::Adrp { rd: 1, target: 0x2000 });
        assert_eq!(i.to_string(), "adrp x1, 0x2000");
        // ADRP x2, -1 page.
        let i = ins(0xF0FF_FFE2, 0x1234);
        assert_eq!(i.opcode, Opcode::Adrp { rd: 2, target: 0 });
    }

    #[test]
    fn add_sub_immediate() {
        // ADD x0, x1, #0x10.
        let i = ins(0x9100_4020, VA);
        assert_eq!(
            i.opcode,
            Opcode::AddImm {
                sf: true,
                set_flags: false,
                rd: 0,
                rn: 1,
                imm: 0x10
            }
        );
        assert_eq!(i.to_string(), "add x0, x1, #0x10");

        // ADD x2, x3, #1, LSL #12: immediate stored fully shifted.
        let i = ins(0x9140_0462, VA);
        assert_eq!(
            i.opcode,
            Opcode::AddImm {
                sf: true,
                set_flags: false,
                rd: 2,
                rn: 3,
                imm: 0x1000
            }
        );

        // SUBS w0, w1, #5.
        let i = ins(0x7100_1420, VA);
        assert_eq!(
            i.opcode,
            Opcode::SubImm {
                sf: false,
                set_flags: true,
                rd: 0,
                rn: 1,
                imm: 5
            }
        );
        assert_eq!(i.to_string(), "subs w0, w1, #0x5");

        // SUBS xzr, x2, #1 is the CMP alias.
        assert_eq!(ins(0xF100_045F, VA).to_string(), "cmp x2, #0x1");
        // ADD x29, sp, #0 is the MOV alias.
        assert_eq!(ins(0x9100_03FD, VA).to_string(), "mov x29, sp");
    }

    #[test]
    fn move_wide_immediates() {
        // MOVZ x0, #0x1234.
        let i = ins(0xD282_4680, VA);
        assert_eq!(
            i.opcode,
            Opcode::Movz {
                sf: true,
                rd: 0,
                imm: 0x1234,
                shift: 0
            }
        );
        assert_eq!(i.to_string(), "movz x0, #0x1234");

        // MOVZ x1, #1, LSL #32 (hw = 2).
        let i = ins(0xD2C0_0021, VA);
        assert_eq!(
            i.opcode,
            Opcode::Movz {
                sf: true,
                rd: 1,
                imm: 1,
                shift: 32
            }
        );
        assert_eq!(i.to_string(), "movz x1, #0x1, lsl #32");

        // MOVK x0, #0xBEEF, LSL #16.
        let i = ins(0xF2B7_DDE0, VA);
        assert_eq!(
            i.opcode,
            Opcode::Movk {
                sf: true,
                rd: 0,
                imm: 0xBEEF,
                shift: 16
            }
        );
        assert_eq!(i.to_string(), "movk x0, #0xbeef, lsl #16");

        // MOVN w0, #0.
        assert_eq!(
            ins(0x1280_0000, VA).opcode,
            Opcode::Movn {
                sf: false,
                rd: 0,
                imm: 0,
                shift: 0
            }
        );

        // 32-bit form with hw = 2 is unallocated.
        assert_eq!(ins(0x52C0_0000, VA).opcode, Opcode::Unknown(0x52C0_0000));
    }

    #[test]
    fn load_store_unsigned_offset() {
        // LDR x0, [x1, #8]: imm12 = 1, scaled by 8.
        let i = ins(0xF940_0420, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldr {
                size: 3,
                rt: 0,
                rn: 1,
                mode: AddrMode::Offset(8)
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "ldr x0, [x1, #0x8]");

        // STR w2, [sp].
        let i = ins(0xB900_03E2, VA);
        assert_eq!(
            i.opcode,
            Opcode::Str {
                size: 2,
                rt: 2,
                rn: 31,
                mode: AddrMode::Offset(0)
            }
        );
        assert_eq!(i.to_string(), "str w2, [sp]");

        // LDRB w3, [x4, #1]: byte access, offset unscaled.
        let i = ins(0x3940_0483, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldr {
                size: 0,
                rt: 3,
                rn: 4,
                mode: AddrMode::Offset(1)
            }
        );
        assert_eq!(i.to_string(), "ldrb w3, [x4, #0x1]");
    }

    #[test]
    fn load_store_pre_and_post_index() {
        // LDR x0, [x1, #-8]!: imm9 = -8, pre-index.
        let i = ins(0xF85F_8C20, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldr {
                size: 3,
                rt: 0,
                rn: 1,
                mode: AddrMode::PreIndex(-8)
            }
        );
        assert_eq!(i.to_string(), "ldr x0, [x1, #-0x8]!");

        // STR x0, [sp], #16: post-index.
        let i = ins(0xF801_07E0, VA);
        assert_eq!(
            i.opcode,
            Opcode::Str {
                size: 3,
                rt: 0,
                rn: 31,
                mode: AddrMode::PostIndex(16)
            }
        );
        assert_eq!(i.to_string(), "str x0, [sp], #0x10");

        // idx = 10 in the same class is the unprivileged LDTR (unmodeled).
        assert_eq!(ins(0xF840_8820, VA).opcode, Opcode::Unknown(0xF840_8820));
    }

    #[test]
    fn load_store_unscaled() {
        // LDUR x0, [x1, #-1]: imm9 = -1, unscaled — no scaled encoding
        // can express this offset.
        let i = ins(0xF85F_F020, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldur {
                size: 3,
                rt: 0,
                rn: 1,
                imm: -1
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "ldur x0, [x1, #-0x1]");

        // The word and byte/halfword widths, and the bare-base spelling.
        assert_eq!(ins(0xB84F_F3E2, VA).to_string(), "ldur w2, [sp, #0xff]");
        assert_eq!(ins(0x3850_0083, VA).to_string(), "ldurb w3, [x4, #-0x100]");
        assert_eq!(ins(0x7840_70C5, VA).to_string(), "ldurh w5, [x6, #0x7]");
        assert_eq!(ins(0xF840_0107, VA).to_string(), "ldur x7, [x8]");

        // STUR at every width.
        let i = ins(0xF81F_8107, VA);
        assert_eq!(
            i.opcode,
            Opcode::Stur {
                size: 3,
                rt: 7,
                rn: 8,
                imm: -8
            }
        );
        assert_eq!(i.to_string(), "stur x7, [x8, #-0x8]");
        assert_eq!(ins(0xB800_3020, VA).to_string(), "stur w0, [x1, #0x3]");
        assert_eq!(ins(0x3800_1149, VA).to_string(), "sturb w9, [x10, #0x1]");
        assert_eq!(ins(0x7800_3041, VA).to_string(), "sturh w1, [x2, #0x3]");

        // The sign-extending loads, both destination widths.
        let i = ins(0x389F_D18B, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldurs {
                size: 0,
                sf: true,
                rt: 11,
                rn: 12,
                imm: -3
            }
        );
        assert_eq!(i.to_string(), "ldursb x11, [x12, #-0x3]");
        assert_eq!(ins(0x38C0_5020, VA).to_string(), "ldursb w0, [x1, #0x5]");
        assert_eq!(ins(0x78C0_51CD, VA).to_string(), "ldursh w13, [x14, #0x5]");
        assert_eq!(ins(0x789F_9062, VA).to_string(), "ldursh x2, [x3, #-0x7]");
        assert_eq!(ins(0xB880_220F, VA).to_string(), "ldursw x15, [x16, #0x2]");

        // PRFUM (size = 11, opc = 10) is not an unscaled load.
        assert_eq!(ins(0xF89F_C040, VA).opcode, Opcode::Unknown(0xF89F_C040));
    }

    #[test]
    fn sign_extending_loads() {
        // LDRSB w8, [sp, #0x2f]: size = 00, opc = 11 (sign-extend to W).
        let i = ins(0x39C0_BFE8, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldrs {
                size: 0,
                sf: false,
                rt: 8,
                rn: 31,
                mode: AddrMode::Offset(0x2f)
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "ldrsb w8, [sp, #0x2f]");

        // LDRSB x0, [x1]: opc = 10 sign-extends to the 64-bit register.
        let i = ins(0x3980_0020, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldrs {
                size: 0,
                sf: true,
                rt: 0,
                rn: 1,
                mode: AddrMode::Offset(0)
            }
        );
        assert_eq!(i.to_string(), "ldrsb x0, [x1]");

        // LDRSH w0, [x1, #2]: size = 01, offset scaled by 2.
        let i = ins(0x79C0_0420, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldrs {
                size: 1,
                sf: false,
                rt: 0,
                rn: 1,
                mode: AddrMode::Offset(2)
            }
        );
        assert_eq!(i.to_string(), "ldrsh w0, [x1, #0x2]");

        // LDRSW x0, [x1, #4]: size = 10, opc = 10, always the X register.
        let i = ins(0xB980_0420, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldrs {
                size: 2,
                sf: true,
                rt: 0,
                rn: 1,
                mode: AddrMode::Offset(4)
            }
        );
        assert_eq!(i.to_string(), "ldrsw x0, [x1, #0x4]");

        // Sign-extension also rides the pre/post-index forms: LDRSB w0,
        // [x1], #1 (post-index, imm9 = 1).
        let i = ins(0x38C0_1420, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldrs {
                size: 0,
                sf: false,
                rt: 0,
                rn: 1,
                mode: AddrMode::PostIndex(1)
            }
        );
        assert_eq!(i.to_string(), "ldrsb w0, [x1], #0x1");

        // PRFM (size = 11, opc = 10) and the unallocated size = 10/opc = 11
        // pair are not sign-extending loads.
        assert_eq!(ins(0xF980_0000, VA).opcode, Opcode::Unknown(0xF980_0000));
        assert_eq!(ins(0xB9C0_0000, VA).opcode, Opcode::Unknown(0xB9C0_0000));
    }

    #[test]
    fn register_offset_loads_and_stores() {
        // LDRB w10, [x8, x9]: opc = 01, option = LSL, S = 0 → bare index.
        let i = ins(0x3869_690A, VA);
        assert_eq!(
            i.opcode,
            Opcode::LdrReg {
                size: 0,
                rt: 10,
                rn: 8,
                off: RegOffset {
                    rm: 9,
                    option: 0b011,
                    scaled: false
                }
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "ldrb w10, [x8, x9]");

        // STR x0, [x1, x2]: 64-bit store, LSL, unscaled.
        let i = ins(0xF822_6820, VA);
        assert_eq!(
            i.opcode,
            Opcode::StrReg {
                size: 3,
                rt: 0,
                rn: 1,
                off: RegOffset {
                    rm: 2,
                    option: 0b011,
                    scaled: false
                }
            }
        );
        assert_eq!(i.to_string(), "str x0, [x1, x2]");

        // LDR x0, [x1, x2, lsl #3]: scaled LSL prints the shift amount.
        assert_eq!(ins(0xF862_7820, VA).to_string(), "ldr x0, [x1, x2, lsl #3]");

        // LDR x0, [x1, w2, uxtw #3]: UXTW selects the 32-bit index.
        let i = ins(0xF862_5820, VA);
        assert_eq!(
            i.opcode,
            Opcode::LdrReg {
                size: 3,
                rt: 0,
                rn: 1,
                off: RegOffset {
                    rm: 2,
                    option: 0b010,
                    scaled: true
                }
            }
        );
        assert_eq!(i.to_string(), "ldr x0, [x1, w2, uxtw #3]");

        // LDRSW x0, [x1, w2, sxtw]: sign-extending register-offset load.
        let i = ins(0xB8A2_C820, VA);
        assert_eq!(
            i.opcode,
            Opcode::LdrsReg {
                size: 2,
                sf: true,
                rt: 0,
                rn: 1,
                off: RegOffset {
                    rm: 2,
                    option: 0b110,
                    scaled: false
                }
            }
        );
        assert_eq!(i.to_string(), "ldrsw x0, [x1, w2, sxtw]");

        // option<1> = 0 is unallocated for the register-offset class.
        assert_eq!(ins(0x3869_290A, VA).opcode, Opcode::Unknown(0x3869_290A));
    }

    #[test]
    fn conditional_select_family() {
        // CSEL x20, x21, x8, lt.
        let i = ins(0x9A88_B2B4, VA);
        assert_eq!(
            i.opcode,
            Opcode::Csel {
                sf: true,
                rd: 20,
                rn: 21,
                rm: 8,
                cond: Cond::Lt
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "csel x20, x21, x8, lt");

        // The four base forms, and their alias behavior.
        assert_eq!(ins(0x1A82_0420, VA).to_string(), "csinc w0, w1, w2, eq");
        assert_eq!(ins(0xDA82_4020, VA).to_string(), "csinv x0, x1, x2, mi");
        assert_eq!(ins(0xDA82_C420, VA).to_string(), "csneg x0, x1, x2, gt");

        // CSINC with both sources ZR is CSET (of the inverted condition);
        // with equal non-ZR sources it is CINC.
        assert_eq!(
            ins(0x9A9F_07E0, VA).opcode,
            Opcode::Csinc {
                sf: true,
                rd: 0,
                rn: 31,
                rm: 31,
                cond: Cond::Eq
            }
        );
        assert_eq!(ins(0x9A9F_07E0, VA).to_string(), "cset x0, ne");
        assert_eq!(ins(0x9A81_0420, VA).to_string(), "cinc x0, x1, ne");
        // CSINV → CSETM / CINV.
        assert_eq!(ins(0xDA9F_03E0, VA).to_string(), "csetm x0, ne");
        assert_eq!(ins(0xDA81_0020, VA).to_string(), "cinv x0, x1, ne");
        // CSNEG → CNEG (with equal sources; no ZR special case).
        assert_eq!(ins(0xDA81_C420, VA).to_string(), "cneg x0, x1, le");

        // op2 = 1x is unallocated for conditional select.
        assert_eq!(ins(0x9A88_BAB4, VA).opcode, Opcode::Unknown(0x9A88_BAB4));
    }

    #[test]
    fn conditional_compare() {
        // CCMP x1, x2, #0, eq (register form).
        let i = ins(0xFA42_0020, VA);
        assert_eq!(
            i.opcode,
            Opcode::CcmpReg {
                sf: true,
                sub: true,
                rn: 1,
                rm: 2,
                nzcv: 0,
                cond: Cond::Eq
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "ccmp x1, x2, #0x0, eq");

        // Both widths, and CCMN as the add compare.
        assert_eq!(ins(0x7A44_1064, VA).to_string(), "ccmp w3, w4, #0x4, ne");
        assert_eq!(ins(0xBA46_B0A8, VA).to_string(), "ccmn x5, x6, #0x8, lt");
        assert_eq!(ins(0x3A48_20EF, VA).to_string(), "ccmn w7, w8, #0xf, cs");

        // The immediate form carries the zero-extended imm5.
        let i = ins(0xFA5F_2820, VA);
        assert_eq!(
            i.opcode,
            Opcode::CcmpImm {
                sf: true,
                sub: true,
                rn: 1,
                imm: 31,
                nzcv: 0,
                cond: Cond::Cs
            }
        );
        assert_eq!(i.to_string(), "ccmp x1, #0x1f, #0x0, cs");
        assert_eq!(ins(0x7A45_C84F, VA).to_string(), "ccmp w2, #0x5, #0xf, gt");
        assert_eq!(ins(0xBA40_4861, VA).to_string(), "ccmn x3, #0x0, #0x1, mi");
        assert_eq!(ins(0x3A4C_D923, VA).to_string(), "ccmn w9, #0xc, #0x3, le");

        // Reserved: the o2 (bit 10) and o3 (bit 4) bits set, and S = 0 —
        // never a near-miss decode.
        assert_eq!(ins(0xFA42_0420, VA).opcode, Opcode::Unknown(0xFA42_0420));
        assert_eq!(ins(0xFA42_0030, VA).opcode, Opcode::Unknown(0xFA42_0030));
        assert_eq!(ins(0xFA5F_2C20, VA).opcode, Opcode::Unknown(0xFA5F_2C20));
        assert_eq!(ins(0xDA42_0020, VA).opcode, Opcode::Unknown(0xDA42_0020));
    }

    #[test]
    fn add_sub_with_carry() {
        // ADC x0, x1, x2.
        let i = ins(0x9A02_0020, VA);
        assert_eq!(
            i.opcode,
            Opcode::Adc {
                sf: true,
                set_flags: false,
                rd: 0,
                rn: 1,
                rm: 2
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "adc x0, x1, x2");

        // All four members at both widths.
        assert_eq!(ins(0x1A05_0083, VA).to_string(), "adc w3, w4, w5");
        assert_eq!(ins(0xBA08_00E6, VA).to_string(), "adcs x6, x7, x8");
        assert_eq!(ins(0x3A02_0020, VA).to_string(), "adcs w0, w1, w2");
        assert_eq!(ins(0xDA0B_0149, VA).to_string(), "sbc x9, x10, x11");
        assert_eq!(ins(0x5A03_0041, VA).to_string(), "sbc w1, w2, w3");
        assert_eq!(ins(0xFA02_0020, VA).to_string(), "sbcs x0, x1, x2");
        assert_eq!(ins(0x7A0E_01AC, VA).to_string(), "sbcs w12, w13, w14");

        // NGC{S} is the rn = zr alias of SBC{S}.
        assert_eq!(ins(0xDA01_03E0, VA).to_string(), "ngc x0, x1");
        assert_eq!(ins(0x7A03_03E2, VA).to_string(), "ngcs w2, w3");

        // A nonzero opcode2 field (here bit 10) is not an add with carry.
        assert_eq!(ins(0x9A02_0420, VA).opcode, Opcode::Unknown(0x9A02_0420));
    }

    // Every golden word below was cross-checked against the system
    // assembler (`clang -arch arm64` + `objdump -d`).

    #[test]
    fn add_sub_shifted_register() {
        // ADD x0, x1, x2.
        let i = ins(0x8B02_0020, VA);
        assert_eq!(
            i.opcode,
            Opcode::AddReg {
                sf: true,
                set_flags: false,
                rd: 0,
                rn: 1,
                rm: 2,
                shift: Shift::Lsl,
                amount: 0
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "add x0, x1, x2");

        assert_eq!(ins(0x0B02_0C20, VA).to_string(), "add w0, w1, w2, lsl #3");
        assert_eq!(ins(0xAB42_1020, VA).to_string(), "adds x0, x1, x2, lsr #4");
        assert_eq!(ins(0xEB02_0020, VA).to_string(), "subs x0, x1, x2");
        assert_eq!(ins(0x4B86_1CA4, VA).to_string(), "sub w4, w5, w6, asr #7");

        // ROR is reserved for add/sub, and the 32-bit form's shift
        // amount stops at 31 — never a near-miss decode.
        assert_eq!(ins(0x8BC2_0020, VA).opcode, Opcode::Unknown(0x8BC2_0020));
        assert_eq!(ins(0x0B02_8020, VA).opcode, Opcode::Unknown(0x0B02_8020));
    }

    #[test]
    fn add_sub_shifted_register_aliases() {
        // CMP/CMN are the flag-setting forms with rd = zr; NEG{S} is a
        // subtract from zr.
        assert_eq!(ins(0xEB02_003F, VA).to_string(), "cmp x1, x2");
        assert_eq!(ins(0x6B02_0C3F, VA).to_string(), "cmp w1, w2, lsl #3");
        assert_eq!(ins(0xAB04_007F, VA).to_string(), "cmn x3, x4");
        assert_eq!(ins(0xCB02_03E0, VA).to_string(), "neg x0, x2");
        assert_eq!(ins(0x6B02_07E0, VA).to_string(), "negs w0, w2, lsl #1");
    }

    #[test]
    fn logical_shifted_register() {
        // AND x0, x1, x2.
        let i = ins(0x8A02_0020, VA);
        assert_eq!(
            i.opcode,
            Opcode::LogReg {
                sf: true,
                op: LogOp::And,
                set_flags: false,
                invert: false,
                rd: 0,
                rn: 1,
                rm: 2,
                shift: Shift::Lsl,
                amount: 0
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "and x0, x1, x2");

        // All eight members: the N bit complements the operand.
        assert_eq!(ins(0x0A22_0820, VA).to_string(), "bic w0, w1, w2, lsl #2");
        assert_eq!(ins(0xAA42_2020, VA).to_string(), "orr x0, x1, x2, lsr #8");
        assert_eq!(ins(0xAA22_0020, VA).to_string(), "orn x0, x1, x2");
        assert_eq!(ins(0xCAC5_3083, VA).to_string(), "eor x3, x4, x5, ror #12");
        assert_eq!(ins(0x4A23_0041, VA).to_string(), "eon w1, w2, w3");
        assert_eq!(ins(0xEA02_0020, VA).to_string(), "ands x0, x1, x2");
        assert_eq!(ins(0xEAA2_0C20, VA).to_string(), "bics x0, x1, x2, asr #3");

        // The 32-bit form's shift amount stops at 31.
        assert_eq!(ins(0x0A02_8020, VA).opcode, Opcode::Unknown(0x0A02_8020));
    }

    #[test]
    fn logical_shifted_register_aliases() {
        // TST (ANDS rd = zr), MOV (unshifted ORR from zr), MVN (ORN from
        // zr; the shift survives the alias).
        assert_eq!(ins(0xEA02_003F, VA).to_string(), "tst x1, x2");
        assert_eq!(ins(0x6A02_1C3F, VA).to_string(), "tst w1, w2, lsl #7");
        assert_eq!(ins(0xAA02_03E0, VA).to_string(), "mov x0, x2");
        assert_eq!(ins(0x2A21_03E0, VA).to_string(), "mvn w0, w1");
        assert_eq!(ins(0xAA21_0BE0, VA).to_string(), "mvn x0, x1, lsl #2");
        // A shifted ORR from zr is not a plain register move.
        assert_eq!(ins(0xAA42_23E0, VA).to_string(), "orr x0, xzr, x2, lsr #8");
    }

    #[test]
    fn add_sub_extended_register() {
        // ADD x0, sp, w1, uxtw #2 — the array-addressing shape.
        let i = ins(0x8B21_4BE0, VA);
        assert_eq!(
            i.opcode,
            Opcode::AddExt {
                sf: true,
                set_flags: false,
                rd: 0,
                rn: 31,
                rm: 1,
                option: 0b010,
                amount: 2
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "add x0, sp, w1, uxtw #2");

        // The identity extend (UXTX) beside SP is spelled plain / LSL.
        assert_eq!(ins(0x8B21_63FF, VA).to_string(), "add sp, sp, x1");
        assert_eq!(ins(0xAB22_6FE1, VA).to_string(), "adds x1, sp, x2, lsl #3");
        assert_eq!(ins(0x8B25_8483, VA).to_string(), "add x3, x4, w5, sxtb #1");
        assert_eq!(ins(0xCB22_73FF, VA).to_string(), "sub sp, sp, x2, lsl #4");
        assert_eq!(ins(0xEB22_2820, VA).to_string(), "subs x0, x1, w2, uxth #2");
        // CMP/CMN aliases; register 31 is SP in the rn position.
        assert_eq!(ins(0xEB21_63FF, VA).to_string(), "cmp sp, x1");
        assert_eq!(ins(0xAB23_C05F, VA).to_string(), "cmn x2, w3, sxtw");

        // A shift amount above 4, and opt != 00, are unallocated.
        assert_eq!(ins(0x8B21_57E0, VA).opcode, Opcode::Unknown(0x8B21_57E0));
        assert_eq!(ins(0x8B61_4BE0, VA).opcode, Opcode::Unknown(0x8B61_4BE0));
    }

    #[test]
    fn logical_immediate() {
        // AND x0, x1, #0xff: a 64-bit element (N = 1), s = 7, r = 0.
        let i = ins(0x9240_1C20, VA);
        assert_eq!(
            i.opcode,
            Opcode::LogImm {
                sf: true,
                op: LogOp::And,
                set_flags: false,
                rd: 0,
                rn: 1,
                imm: 0xFF
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "and x0, x1, #0xff");

        assert_eq!(ins(0x1200_1820, VA).to_string(), "and w0, w1, #0x7f");
        // An 8-bit element, rotated then replicated across the register.
        assert_eq!(
            ins(0xB204_C462, VA).to_string(),
            "orr x2, x3, #0x3030303030303030"
        );
        assert_eq!(ins(0x5203_C8A4, VA).to_string(), "eor w4, w5, #0xe0e0e0e0");
        assert_eq!(
            ins(0xF204_CCE6, VA).to_string(),
            "ands x6, x7, #0xf0f0f0f0f0f0f0f0"
        );
        // TST (ANDS rd = zr) and MOV (ORR from zr) aliases.
        assert_eq!(ins(0xF27F_043F, VA).to_string(), "tst x1, #0x6");
        assert_eq!(
            ins(0xB201_F3E0, VA).to_string(),
            "mov x0, #0xaaaaaaaaaaaaaaaa"
        );
        assert_eq!(ins(0x3200_C3E1, VA).to_string(), "mov w1, #0x1010101");
        // rd = 31 is SP for the non-flag-setting forms — the frame-align
        // idiom writes the stack pointer.
        assert_eq!(
            ins(0x927C_EC3F, VA).to_string(),
            "and sp, x1, #0xfffffffffffffff0"
        );

        // Reserved: N = 1 in the 32-bit form; the all-ones element.
        assert_eq!(ins(0x1240_1C20, VA).opcode, Opcode::Unknown(0x1240_1C20));
        assert_eq!(ins(0x9200_FC20, VA).opcode, Opcode::Unknown(0x9200_FC20));
    }

    /// Bit-by-bit reference for [`decode_bit_masks`], built from the Arm
    /// ARM's element-size table rather than the implementation's
    /// highest-set-bit arithmetic: `N:imms` selects the element size,
    /// the element is `s + 1` ones rotated right by `r`, and output bit
    /// `j` is element bit `(j + r) mod esize`.
    fn reference_bit_masks(n: bool, imms: u8, immr: u8, sf: bool) -> Option<u64> {
        let esize: u32 = if n {
            64
        } else if imms >> 5 == 0 {
            32
        } else if imms >> 4 == 0b10 {
            16
        } else if imms >> 3 == 0b110 {
            8
        } else if imms >> 2 == 0b1110 {
            4
        } else if imms >> 1 == 0b11110 {
            2
        } else {
            return None;
        };
        if !sf && esize > 32 {
            return None;
        }
        let s = u32::from(imms) % esize;
        let r = u32::from(immr) % esize;
        if s == esize - 1 {
            return None;
        }
        let regsize = if sf { 64 } else { 32 };
        let mut out = 0u64;
        for j in 0..regsize {
            if (j % esize + r) % esize <= s {
                out |= 1u64 << j;
            }
        }
        Some(out)
    }

    #[test]
    fn decode_bit_masks_matches_a_bitwise_reference_exhaustively() {
        // The whole (sf, N, immr, imms) space — 2 * 2 * 64 * 64 = 16384
        // triples-plus-width — must agree with the reference, reserved
        // encodings included.
        let mut valid = 0u32;
        for sf in [false, true] {
            for n in [false, true] {
                for immr in 0..64u8 {
                    for imms in 0..64u8 {
                        let got = decode_bit_masks(n, imms, immr, sf);
                        let want = reference_bit_masks(n, imms, immr, sf);
                        assert_eq!(got, want, "sf={sf} n={n} immr={immr} imms={imms}");
                        if got.is_some() {
                            valid += 1;
                        }
                    }
                }
            }
        }
        // 64 immr values times (esize - 1) legal imms per element size:
        // 64-bit form 1+3+7+15+31+63 = 120, 32-bit form 1+3+7+15+31 = 57.
        assert_eq!(valid, 64 * 120 + 64 * 57);
    }

    #[test]
    fn bitfield_moves_decode_canonically_and_render_aliases() {
        // LSL x0, x1, #8 decodes as the canonical UBFM #56, #55; alias
        // spelling is display-only.
        let i = ins(0xD378_DC20, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ubfm {
                sf: true,
                rd: 0,
                rn: 1,
                immr: 56,
                imms: 55
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "lsl x0, x1, #8");

        // The shift aliases, both widths.
        assert_eq!(ins(0x531F_7862, VA).to_string(), "lsl w2, w3, #1");
        assert_eq!(ins(0xD350_FCA4, VA).to_string(), "lsr x4, x5, #16");
        assert_eq!(ins(0x131F_7CE6, VA).to_string(), "asr w6, w7, #31");
        assert_eq!(ins(0x937F_FD28, VA).to_string(), "asr x8, x9, #63");
        // The extract/insert and extend aliases.
        assert_eq!(ins(0xD348_5C20, VA).to_string(), "ubfx x0, x1, #8, #16");
        assert_eq!(ins(0x531C_1C62, VA).to_string(), "ubfiz w2, w3, #4, #8");
        assert_eq!(ins(0x5300_1CA4, VA).to_string(), "uxtb w4, w5");
        assert_eq!(ins(0x5300_3CE6, VA).to_string(), "uxth w6, w7");
        assert_eq!(ins(0x9344_3D28, VA).to_string(), "sbfx x8, x9, #4, #12");
        assert_eq!(ins(0x937E_116A, VA).to_string(), "sbfiz x10, x11, #2, #5");
        assert_eq!(ins(0x1300_1C20, VA).to_string(), "sxtb w0, w1");
        // The 64-bit sign extends spell their source as a W register.
        assert_eq!(ins(0x9340_1C62, VA).to_string(), "sxtb x2, w3");
        assert_eq!(ins(0x9340_3CA4, VA).to_string(), "sxth x4, w5");
        assert_eq!(ins(0x9340_7CE6, VA).to_string(), "sxtw x6, w7");
        // BFM: insert, extract-insert-low, and the zr-source BFC.
        assert_eq!(ins(0xB378_3C20, VA).to_string(), "bfi x0, x1, #8, #16");
        assert_eq!(ins(0x3304_2C62, VA).to_string(), "bfxil w2, w3, #4, #8");
        assert_eq!(ins(0xB370_1FE4, VA).to_string(), "bfc x4, #16, #8");

        // Reserved: N != sf; the 32-bit form with immr or imms >= 32;
        // opc = 11.
        assert_eq!(ins(0xD300_1C20, VA).opcode, Opcode::Unknown(0xD300_1C20));
        assert_eq!(ins(0x5340_1C20, VA).opcode, Opcode::Unknown(0x5340_1C20));
        assert_eq!(ins(0x5320_1C20, VA).opcode, Opcode::Unknown(0x5320_1C20));
        assert_eq!(ins(0x7300_1C20, VA).opcode, Opcode::Unknown(0x7300_1C20));
    }

    #[test]
    fn variable_shifts_and_divides() {
        // LSLV renders with the preferred lsl spelling.
        let i = ins(0x9AC2_2020, VA);
        assert_eq!(
            i.opcode,
            Opcode::ShiftReg {
                sf: true,
                kind: Shift::Lsl,
                rd: 0,
                rn: 1,
                rm: 2
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "lsl x0, x1, x2");
        assert_eq!(ins(0x1AC5_2483, VA).to_string(), "lsr w3, w4, w5");
        assert_eq!(ins(0x9AC8_28E6, VA).to_string(), "asr x6, x7, x8");
        assert_eq!(ins(0x9ACB_2D49, VA).to_string(), "ror x9, x10, x11");

        let i = ins(0x9AC2_0820, VA);
        assert_eq!(
            i.opcode,
            Opcode::Udiv {
                sf: true,
                rd: 0,
                rn: 1,
                rm: 2
            }
        );
        assert_eq!(i.to_string(), "udiv x0, x1, x2");
        assert_eq!(ins(0x1AC5_0883, VA).to_string(), "udiv w3, w4, w5");
        assert_eq!(ins(0x9AC8_0CE6, VA).to_string(), "sdiv x6, x7, x8");

        // S = 1 and the unallocated opcode 0 stay Unknown.
        assert_eq!(ins(0xBAC2_0820, VA).opcode, Opcode::Unknown(0xBAC2_0820));
        assert_eq!(ins(0x9AC2_0020, VA).opcode, Opcode::Unknown(0x9AC2_0020));
    }

    #[test]
    fn multiply_accumulate_family() {
        // MADD, and MUL as its ra = zr alias.
        let i = ins(0x9B02_0C20, VA);
        assert_eq!(
            i.opcode,
            Opcode::Madd {
                sf: true,
                rd: 0,
                rn: 1,
                rm: 2,
                ra: 3
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "madd x0, x1, x2, x3");
        assert_eq!(ins(0x1B06_9CA4, VA).to_string(), "msub w4, w5, w6, w7");
        assert_eq!(ins(0x9B0A_7D28, VA).to_string(), "mul x8, x9, x10");
        assert_eq!(ins(0x9B0D_FD8B, VA).to_string(), "mneg x11, x12, x13");
        // The widening forms mix X destinations with W sources.
        assert_eq!(ins(0x9B22_0C20, VA).to_string(), "smaddl x0, w1, w2, x3");
        assert_eq!(ins(0x9B26_9CA4, VA).to_string(), "smsubl x4, w5, w6, x7");
        assert_eq!(ins(0x9BAA_2D28, VA).to_string(), "umaddl x8, w9, w10, x11");
        assert_eq!(ins(0x9BAE_BDAC, VA).to_string(), "umsubl x12, w13, w14, x15");
        assert_eq!(ins(0x9B22_7C20, VA).to_string(), "smull x0, w1, w2");
        assert_eq!(ins(0x9BA5_7C83, VA).to_string(), "umull x3, w4, w5");
        // The high-half multiplies.
        let i = ins(0x9B42_7C20, VA);
        assert_eq!(
            i.opcode,
            Opcode::Mulh {
                signed: true,
                rd: 0,
                rn: 1,
                rm: 2
            }
        );
        assert_eq!(i.to_string(), "smulh x0, x1, x2");
        assert_eq!(ins(0x9BC5_7C83, VA).to_string(), "umulh x3, x4, x5");

        // Reserved probes: op54 != 00, the widening forms at sf = 0,
        // SMULH with Ra != 11111.
        assert_eq!(ins(0xBB42_7C20, VA).opcode, Opcode::Unknown(0xBB42_7C20));
        assert_eq!(ins(0x1B22_0C20, VA).opcode, Opcode::Unknown(0x1B22_0C20));
        assert_eq!(ins(0x9B42_0C20, VA).opcode, Opcode::Unknown(0x9B42_0C20));
    }

    #[test]
    fn ldr_literal_classifies_its_target() {
        // LDR x0, +8: opc = 01, imm19 = 2.
        let i = ins(0x5800_0040, VA);
        assert_eq!(
            i.opcode,
            Opcode::LdrLit {
                sf: true,
                rt: 0,
                target: 0x1008
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "ldr x0, 0x1008");

        // LDR w5, -4: opc = 00, imm19 = -1.
        let i = ins(0x18FF_FFE5, VA);
        assert_eq!(
            i.opcode,
            Opcode::LdrLit {
                sf: false,
                rt: 5,
                target: 0xFFC
            }
        );

        // PRFM literal (opc = 11) is unmodeled.
        assert_eq!(ins(0xD800_0040, VA).opcode, Opcode::Unknown(0xD800_0040));
    }

    #[test]
    fn load_store_pair() {
        // STP x29, x30, [sp, #-16]! — the classic prologue.
        let i = ins(0xA9BF_7BFD, VA);
        assert_eq!(
            i.opcode,
            Opcode::Stp {
                sf: true,
                rt: 29,
                rt2: 30,
                rn: 31,
                mode: AddrMode::PreIndex(-16)
            }
        );
        assert_eq!(i.to_string(), "stp x29, x30, [sp, #-0x10]!");

        // LDP x29, x30, [sp], #16 — the matching epilogue.
        let i = ins(0xA8C1_7BFD, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldp {
                sf: true,
                rt: 29,
                rt2: 30,
                rn: 31,
                mode: AddrMode::PostIndex(16)
            }
        );
        assert_eq!(i.to_string(), "ldp x29, x30, [sp], #0x10");

        // LDP w0, w1, [x2, #8]: 32-bit form scales imm7 by 4.
        let i = ins(0x2941_0440, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldp {
                sf: false,
                rt: 0,
                rt2: 1,
                rn: 2,
                mode: AddrMode::Offset(8)
            }
        );
        assert_eq!(i.to_string(), "ldp w0, w1, [x2, #0x8]");
    }

    // Every SIMD&FP golden word below was produced by assembling the
    // rendered text with the system assembler (`clang -arch arm64`) and
    // reading the encoding back with objdump.

    #[test]
    fn simd_load_store_unsigned_offset_all_sizes() {
        // The five access sizes, load and store; the offset scales by
        // the access size (16 for q).
        let i = ins(0x3D40_0420, VA);
        assert_eq!(
            i.opcode,
            Opcode::FLdr {
                size: 0,
                rt: 0,
                rn: 1,
                mode: AddrMode::Offset(1)
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "ldr b0, [x1, #0x1]");
        assert_eq!(ins(0x7D40_0462, VA).to_string(), "ldr h2, [x3, #0x2]");
        assert_eq!(ins(0xBD40_07E4, VA).to_string(), "ldr s4, [sp, #0x4]");
        assert_eq!(ins(0xFD40_04E6, VA).to_string(), "ldr d6, [x7, #0x8]");
        let i = ins(0x3DC0_0528, VA);
        assert_eq!(
            i.opcode,
            Opcode::FLdr {
                size: 4,
                rt: 8,
                rn: 9,
                mode: AddrMode::Offset(16)
            }
        );
        assert_eq!(i.to_string(), "ldr q8, [x9, #0x10]");
        assert_eq!(ins(0x3D00_0441, VA).to_string(), "str b1, [x2, #0x1]");
        assert_eq!(ins(0x7D00_0483, VA).to_string(), "str h3, [x4, #0x2]");
        assert_eq!(ins(0xBD00_03E5, VA).to_string(), "str s5, [sp]");
        assert_eq!(ins(0xFD00_0D07, VA).to_string(), "str d7, [x8, #0x18]");
        assert_eq!(ins(0x3D80_0949, VA).to_string(), "str q9, [x10, #0x20]");
        // The top of the q range: imm12 all-ones scales to 0xfff0, and
        // register 31 is V31, never ZR.
        assert_eq!(ins(0x3DFF_FC1F, VA).to_string(), "ldr q31, [x0, #0xfff0]");
    }

    #[test]
    fn simd_load_store_pre_and_post_index() {
        let i = ins(0xFC5F_8C20, VA);
        assert_eq!(
            i.opcode,
            Opcode::FLdr {
                size: 3,
                rt: 0,
                rn: 1,
                mode: AddrMode::PreIndex(-8)
            }
        );
        assert_eq!(i.to_string(), "ldr d0, [x1, #-0x8]!");
        assert_eq!(ins(0x3CC1_0441, VA).to_string(), "ldr q1, [x2], #0x10");
        assert_eq!(ins(0xBC1F_CC62, VA).to_string(), "str s2, [x3, #-0x4]!");
        assert_eq!(ins(0x3C82_07E3, VA).to_string(), "str q3, [sp], #0x20");
        assert_eq!(ins(0x3C40_14A4, VA).to_string(), "ldr b4, [x5], #0x1");
        assert_eq!(ins(0x7C00_2CC5, VA).to_string(), "str h5, [x6, #0x2]!");
    }

    #[test]
    fn simd_load_store_unscaled() {
        let i = ins(0xFC5F_F020, VA);
        assert_eq!(
            i.opcode,
            Opcode::FLdur {
                size: 3,
                rt: 0,
                rn: 1,
                imm: -1
            }
        );
        assert_eq!(i.to_string(), "ldur d0, [x1, #-0x1]");
        assert_eq!(ins(0x3CDF_0062, VA).to_string(), "ldur q2, [x3, #-0x10]");
        assert_eq!(ins(0x7C40_5041, VA).to_string(), "ldur h1, [x2, #0x5]");
        assert_eq!(ins(0xBC00_30A4, VA).to_string(), "stur s4, [x5, #0x3]");
        assert_eq!(ins(0x3C1F_70E6, VA).to_string(), "stur b6, [x7, #-0x9]");
        assert_eq!(ins(0x3C9E_03E7, VA).to_string(), "stur q7, [sp, #-0x20]");
    }

    #[test]
    fn simd_load_store_register_offset() {
        let i = ins(0x3CE3_7841, VA);
        assert_eq!(
            i.opcode,
            Opcode::FLdrReg {
                size: 4,
                rt: 1,
                rn: 2,
                off: RegOffset {
                    rm: 3,
                    option: 0b011,
                    scaled: true
                }
            }
        );
        assert_eq!(i.to_string(), "ldr q1, [x2, x3, lsl #4]");
        assert_eq!(ins(0xFC62_6820, VA).to_string(), "ldr d0, [x1, x2]");
        assert_eq!(ins(0xBC65_5882, VA).to_string(), "ldr s2, [x4, w5, uxtw #2]");
        assert_eq!(ins(0xFC27_D8C3, VA).to_string(), "str d3, [x6, w7, sxtw #3]");
        assert_eq!(ins(0x3C29_6904, VA).to_string(), "str b4, [x8, x9]");
        // The scaled byte form's explicit zero shift.
        assert_eq!(ins(0x3C69_7905, VA).to_string(), "ldr b5, [x8, x9, lsl #0]");
        assert_eq!(ins(0x7C61_D806, VA).to_string(), "ldr h6, [x0, w1, sxtw #1]");
        assert_eq!(ins(0x3CA3_E85E, VA).to_string(), "str q30, [x2, x3, sxtx]");
    }

    #[test]
    fn simd_load_store_pair() {
        let i = ins(0x6DBF_27E8, VA);
        assert_eq!(
            i.opcode,
            Opcode::FStp {
                size: 3,
                rt: 8,
                rt2: 9,
                rn: 31,
                mode: AddrMode::PreIndex(-16)
            }
        );
        // The canonical FP-callee-saved prologue word.
        assert_eq!(i.to_string(), "stp d8, d9, [sp, #-0x10]!");
        assert_eq!(ins(0x2D40_0440, VA).to_string(), "ldp s0, s1, [x2]");
        assert_eq!(ins(0x6D41_0C82, VA).to_string(), "ldp d2, d3, [x4, #0x10]");
        // The q pair scales imm7 by 32.
        let i = ins(0xAD41_17E4, VA);
        assert_eq!(
            i.opcode,
            Opcode::FLdp {
                size: 4,
                rt: 4,
                rt2: 5,
                rn: 31,
                mode: AddrMode::Offset(32)
            }
        );
        assert_eq!(i.to_string(), "ldp q4, q5, [sp, #0x20]");
        assert_eq!(ins(0x6CC1_1D06, VA).to_string(), "ldp d6, d7, [x8], #0x10");
        assert_eq!(ins(0xAD00_0460, VA).to_string(), "stp q0, q1, [x3]");
        assert_eq!(ins(0x2CBF_0C82, VA).to_string(), "stp s2, s3, [x4], #-0x8");
        assert_eq!(ins(0xADBE_7C1E, VA).to_string(), "stp q30, q31, [x0, #-0x40]!");
    }

    #[test]
    fn simd_ldr_literal() {
        let i = ins(0x1C00_0000, VA);
        assert_eq!(
            i.opcode,
            Opcode::FLdrLit {
                size: 2,
                rt: 0,
                target: VA
            }
        );
        assert_eq!(i.to_string(), "ldr s0, 0x1000");
        assert_eq!(ins(0x5CFF_FFE1, VA).to_string(), "ldr d1, 0xffc");
        assert_eq!(ins(0x9CFF_FFC2, VA).to_string(), "ldr q2, 0xff8");
    }

    #[test]
    fn fmov_register_and_general() {
        assert_eq!(
            ins(0x1E20_4020, VA).opcode,
            Opcode::FmovReg {
                double: false,
                rd: 0,
                rn: 1
            }
        );
        assert_eq!(ins(0x1E20_4020, VA).to_string(), "fmov s0, s1");
        assert_eq!(ins(0x1E60_4062, VA).to_string(), "fmov d2, d3");
        assert_eq!(ins(0x1E26_0020, VA).to_string(), "fmov w0, s1");
        assert_eq!(ins(0x9E66_0062, VA).to_string(), "fmov x2, d3");
        assert_eq!(ins(0x1E27_00A4, VA).to_string(), "fmov s4, w5");
        assert_eq!(ins(0x9E67_00E6, VA).to_string(), "fmov d6, x7");
        // The D[1] lane forms.
        assert_eq!(
            ins(0x9EAE_0128, VA).opcode,
            Opcode::FmovToGp {
                sf: true,
                hi: true,
                rd: 8,
                rn: 9
            }
        );
        assert_eq!(ins(0x9EAE_0128, VA).to_string(), "fmov x8, v9.d[1]");
        assert_eq!(ins(0x9EAF_016A, VA).to_string(), "fmov v10.d[1], x11");
    }

    #[test]
    fn fmov_scalar_immediate() {
        let i = ins(0x1E2E_1000, VA);
        assert_eq!(
            i.opcode,
            Opcode::FmovImm {
                double: false,
                imm: 0x70,
                rd: 0
            }
        );
        assert_eq!(i.to_string(), "fmov s0, #1.0");
        assert_eq!(ins(0x1E7C_1001, VA).to_string(), "fmov d1, #-0.5");
        assert_eq!(ins(0x1E67_F002, VA).to_string(), "fmov d2, #31.0");
        assert_eq!(ins(0x1E28_1003, VA).to_string(), "fmov s3, #0.125");
        // The expansion formula, at its corners: every value is
        // ±(16+m)/16 × 2^e with e in [-3, 4].
        assert_eq!(fp_imm_value(0x70), 1.0);
        assert_eq!(fp_imm_value(0x00), 2.0);
        assert_eq!(fp_imm_value(0x4F), 0.2421875); // the smallest magnitude
        assert_eq!(fp_imm_value(0x3F), 31.0); // the largest
        assert_eq!(fp_imm_value(0xFF), -1.9375);
    }

    #[test]
    fn movi_and_mvni_forms() {
        let i = ins(0x0F03_E7E0, VA);
        assert_eq!(
            i.opcode,
            Opcode::Movi {
                q: false,
                invert: false,
                size: 0,
                imm: 0x7F,
                shift: 0,
                msl: false,
                rd: 0
            }
        );
        assert_eq!(i.to_string(), "movi v0.8b, #0x7f");
        assert_eq!(ins(0x4F04_E401, VA).to_string(), "movi v1.16b, #0x80");
        assert_eq!(ins(0x0F00_8642, VA).to_string(), "movi v2.4h, #0x12");
        assert_eq!(ins(0x4F00_A643, VA).to_string(), "movi v3.8h, #0x12, lsl #8");
        assert_eq!(ins(0x0F01_0684, VA).to_string(), "movi v4.2s, #0x34");
        assert_eq!(ins(0x4F01_6685, VA).to_string(), "movi v5.4s, #0x34, lsl #24");
        assert_eq!(ins(0x0F02_C6C6, VA).to_string(), "movi v6.2s, #0x56, msl #8");
        assert_eq!(ins(0x4F02_D6C7, VA).to_string(), "movi v7.4s, #0x56, msl #16");
        // The 64-bit byte-mask forms spell out the expanded immediate.
        assert_eq!(ins(0x2F05_E548, VA).to_string(), "movi d8, #0xff00ff00ff00ff00");
        assert_eq!(
            ins(0x6F07_E7E9, VA).to_string(),
            "movi v9.2d, #0xffffffffffffffff"
        );
        assert_eq!(ins(0x6F00_E41F, VA).to_string(), "movi v31.2d, #0x0");
        assert_eq!(ins(0x2F00_842A, VA).to_string(), "mvni v10.4h, #0x1");
        assert_eq!(ins(0x6F00_444B, VA).to_string(), "mvni v11.4s, #0x2, lsl #16");
        assert_eq!(ins(0x2F00_C46C, VA).to_string(), "mvni v12.2s, #0x3, msl #8");
        assert_eq!(ins(0x6F02_A48D, VA).to_string(), "mvni v13.8h, #0x44, lsl #8");
        // The expansion helper at its corners.
        assert_eq!(movi_expand(0, 0x7F, 0, false, false), 0x7F7F_7F7F_7F7F_7F7F);
        assert_eq!(movi_expand(1, 0x12, 8, false, false), 0x1200_1200_1200_1200);
        assert_eq!(movi_expand(2, 0x56, 8, true, false), 0x0000_56FF_0000_56FF);
        assert_eq!(movi_expand(2, 0x03, 8, true, true), 0xFFFF_FC00_FFFF_FC00);
        assert_eq!(movi_expand(1, 0x01, 0, false, true), 0xFFFE_FFFE_FFFE_FFFE);
        assert_eq!(movi_expand(3, 0xAA, 0, false, false), 0xFF00_FF00_FF00_FF00);
        assert_eq!(movi_expand(3, 0x00, 0, false, false), 0);
        assert_eq!(movi_expand(3, 0xFF, 0, false, false), u64::MAX);
    }

    #[test]
    fn fmov_vector_immediate() {
        let i = ins(0x0F03_F60E, VA);
        assert_eq!(
            i.opcode,
            Opcode::FmovVecImm {
                q: false,
                double: false,
                imm: 0x70,
                rd: 14
            }
        );
        assert_eq!(i.to_string(), "fmov v14.2s, #1.0");
        assert_eq!(ins(0x4F04_F48F, VA).to_string(), "fmov v15.4s, #-2.5");
        assert_eq!(ins(0x6F02_F610, VA).to_string(), "fmov v16.2d, #0.25");
    }

    #[test]
    fn simd_reserved_encodings_stay_unknown() {
        for w in [
            // LDR/STR immediate: opc = 1x with size != 00.
            0x7DC0_0420u32,
            // Unscaled with idx = 10: no unprivileged form on the V side.
            0xFC5F_F820,
            // Register offset with option<1> = 0.
            0x3C69_3905,
            // Pair opc = 11, LDNP (no-allocate), literal opc = 11.
            0xED40_0440,
            0x2C40_0440,
            0xDC00_0000,
            // FMOV (register) type = 10 (unallocated) and 11 (half).
            0x1EA0_4020,
            0x1EE0_4020,
            // FMOV (general): sf/type combination outside the six.
            0x1E66_0062,
            // Modified immediate with o2 = 1 (half-precision FMOV space).
            0x0F03_EFE0,
            // ORR (vector immediate): a read-modify-write, not a move.
            0x0F01_1684,
            // FMOV .2d with Q = 0 is unallocated.
            0x2F02_F610,
        ] {
            let i = ins(w, VA);
            assert_eq!(i.opcode, Opcode::Unknown(w), "{w:#010x}");
            assert_eq!(i.flow, Flow::Sequential);
        }
    }

    #[test]
    fn nop_and_hints() {
        assert_eq!(ins(0xD503_201F, VA).opcode, Opcode::Nop);
        assert_eq!(ins(0xD503_201F, VA).to_string(), "nop");
        assert_eq!(ins(0xD503_203F, VA).opcode, Opcode::Yield);
        assert_eq!(ins(0xD503_205F, VA).opcode, Opcode::Wfe);
        assert_eq!(ins(0xD503_207F, VA).opcode, Opcode::Wfi);
        assert_eq!(ins(0xD503_209F, VA).opcode, Opcode::Sev);
        assert_eq!(ins(0xD503_20BF, VA).opcode, Opcode::Sevl);
        // An allocated-but-unnamed hint (here CRm:op2 = 7, XPACLRI).
        let i = ins(0xD503_20FF, VA);
        assert_eq!(i.opcode, Opcode::Hint { imm: 7 });
        assert_eq!(i.to_string(), "hint #0x7");
        // Every hint is sequential.
        assert_eq!(i.flow, Flow::Sequential);
    }

    #[test]
    fn unmodeled_encodings_are_unknown_and_sequential() {
        // An LSE atomic (LDADDAL) and an SVE word: the remaining ceiling
        // after the three-same integer ALU slice.
        for w in [0xF8E9_0108u32, 0x0420_0000] {
            let i = ins(w, VA);
            assert_eq!(i.opcode, Opcode::Unknown(w));
            assert_eq!(i.flow, Flow::Sequential);
            assert_eq!(i.to_string(), format!(".inst {w:#010x}"));
        }
    }

    #[test]
    fn simd_three_same_integer_alu_decodes() {
        // orr v1.16b, v1.16b, v1.16b — the former unknown ceiling word.
        let i = ins(0x4EA1_1C21, VA);
        assert_eq!(
            i.opcode,
            Opcode::SimdAlu {
                op: SimdAluOp::Orr,
                q: true,
                size: 0,
                rd: 1,
                rn: 1,
                rm: 1
            }
        );
        assert_eq!(i.to_string(), "orr v1.16b, v1.16b, v1.16b");
        // and v0.16b, v1.16b, v2.16b
        assert_eq!(
            ins(0x4E22_1C20, VA).opcode,
            Opcode::SimdAlu {
                op: SimdAluOp::And,
                q: true,
                size: 0,
                rd: 0,
                rn: 1,
                rm: 2
            }
        );
        // eor v0.8b, v1.8b, v2.8b
        assert_eq!(
            ins(0x2E22_1C20, VA).to_string(),
            "eor v0.8b, v1.8b, v2.8b"
        );
        // add v0.4s, v1.4s, v2.4s
        assert_eq!(
            ins(0x4EA2_8420, VA).to_string(),
            "add v0.4s, v1.4s, v2.4s"
        );
        // sub v0.2d, v1.2d, v2.2d
        assert_eq!(
            ins(0x6EE2_8420, VA).to_string(),
            "sub v0.2d, v1.2d, v2.2d"
        );
        // add .2d with Q = 0 is reserved → Unknown.
        assert_eq!(ins(0x0EE2_8420, VA).opcode, Opcode::Unknown(0x0EE2_8420));
        // BIC (size=01 logical) stays Unknown this slice.
        assert_eq!(ins(0x4E61_1C21, VA).opcode, Opcode::Unknown(0x4E61_1C21));
    }

    #[test]
    fn simd_three_same_fp_and_compare_decodes() {
        // fadd v0.4s, v1.4s, v2.4s
        assert_eq!(
            ins(0x4E22_D420, VA).opcode,
            Opcode::SimdAlu {
                op: SimdAluOp::Fadd,
                q: true,
                size: 2,
                rd: 0,
                rn: 1,
                rm: 2
            }
        );
        assert_eq!(ins(0x4E22_D420, VA).to_string(), "fadd v0.4s, v1.4s, v2.4s");
        // fadd v0.2d, v1.2d, v2.2d
        assert_eq!(ins(0x4E62_D420, VA).to_string(), "fadd v0.2d, v1.2d, v2.2d");
        // fmul v0.4s, v1.4s, v2.4s
        assert_eq!(ins(0x6E22_DC20, VA).to_string(), "fmul v0.4s, v1.4s, v2.4s");
        // fmul v0.2d, v1.2d, v2.2d
        assert_eq!(ins(0x6E62_DC20, VA).to_string(), "fmul v0.2d, v1.2d, v2.2d");
        // fadd v0.2s, v1.2s, v2.2s (Q = 0)
        assert_eq!(ins(0x0E22_D420, VA).to_string(), "fadd v0.2s, v1.2s, v2.2s");
        // cmhi v0.4s, v1.4s, v2.4s
        assert_eq!(
            ins(0x6EA2_3420, VA).opcode,
            Opcode::SimdAlu {
                op: SimdAluOp::Cmhi,
                q: true,
                size: 2,
                rd: 0,
                rn: 1,
                rm: 2
            }
        );
        assert_eq!(ins(0x6EA2_3420, VA).to_string(), "cmhi v0.4s, v1.4s, v2.4s");
        // cmeq v0.16b, v1.16b, v2.16b
        assert_eq!(ins(0x6E22_8C20, VA).to_string(), "cmeq v0.16b, v1.16b, v2.16b");
        // .2d FADD with Q = 0 is reserved → Unknown.
        assert_eq!(ins(0x0E62_D420, VA).opcode, Opcode::Unknown(0x0E62_D420));
    }

    // Every FP-arithmetic golden word below was produced by assembling
    // the rendered text with the system assembler (`clang -arch arm64`,
    // arm64e for the PAC forms) and reading the encoding back with
    // otool, so each spelling is proven to re-assemble to its word.

    #[test]
    fn fp_two_source_all_ops_both_precisions() {
        let i = ins(0x1E22_0820, VA);
        assert_eq!(
            i.opcode,
            Opcode::FArith2 {
                op: F2Op::Mul,
                double: false,
                rd: 0,
                rn: 1,
                rm: 2
            }
        );
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "fmul s0, s1, s2");
        assert_eq!(ins(0x1E25_1883, VA).to_string(), "fdiv s3, s4, s5");
        assert_eq!(ins(0x1E28_28E6, VA).to_string(), "fadd s6, s7, s8");
        assert_eq!(ins(0x1E2B_3949, VA).to_string(), "fsub s9, s10, s11");
        assert_eq!(ins(0x1E2E_49AC, VA).to_string(), "fmax s12, s13, s14");
        assert_eq!(ins(0x1E31_5A0F, VA).to_string(), "fmin s15, s16, s17");
        assert_eq!(ins(0x1E34_6A72, VA).to_string(), "fmaxnm s18, s19, s20");
        assert_eq!(ins(0x1E37_7AD5, VA).to_string(), "fminnm s21, s22, s23");
        assert_eq!(ins(0x1E3A_8B38, VA).to_string(), "fnmul s24, s25, s26");
        assert_eq!(ins(0x1E62_0820, VA).to_string(), "fmul d0, d1, d2");
        assert_eq!(ins(0x1E65_1883, VA).to_string(), "fdiv d3, d4, d5");
        assert_eq!(ins(0x1E68_28E6, VA).to_string(), "fadd d6, d7, d8");
        assert_eq!(ins(0x1E6B_3949, VA).to_string(), "fsub d9, d10, d11");
        assert_eq!(ins(0x1E6E_49AC, VA).to_string(), "fmax d12, d13, d14");
        assert_eq!(ins(0x1E71_5A0F, VA).to_string(), "fmin d15, d16, d17");
        assert_eq!(ins(0x1E74_6A72, VA).to_string(), "fmaxnm d18, d19, d20");
        assert_eq!(ins(0x1E77_7AD5, VA).to_string(), "fminnm d21, d22, d23");
        assert_eq!(ins(0x1E7A_8B38, VA).to_string(), "fnmul d24, d25, d26");
    }

    #[test]
    fn fp_three_source_all_four_both_precisions() {
        let i = ins(0x1F02_0C20, VA);
        assert_eq!(
            i.opcode,
            Opcode::FArith3 {
                negate: false,
                sub: false,
                double: false,
                rd: 0,
                rn: 1,
                rm: 2,
                ra: 3
            }
        );
        assert_eq!(i.to_string(), "fmadd s0, s1, s2, s3");
        assert_eq!(ins(0x1F06_9CA4, VA).to_string(), "fmsub s4, s5, s6, s7");
        assert_eq!(ins(0x1F2A_2D28, VA).to_string(), "fnmadd s8, s9, s10, s11");
        assert_eq!(ins(0x1F2E_BDAC, VA).to_string(), "fnmsub s12, s13, s14, s15");
        assert_eq!(ins(0x1F42_0C20, VA).to_string(), "fmadd d0, d1, d2, d3");
        assert_eq!(ins(0x1F46_9CA4, VA).to_string(), "fmsub d4, d5, d6, d7");
        assert_eq!(ins(0x1F6A_2D28, VA).to_string(), "fnmadd d8, d9, d10, d11");
        assert_eq!(ins(0x1F6E_BDAC, VA).to_string(), "fnmsub d12, d13, d14, d15");
    }

    #[test]
    fn fp_one_source_and_precision_convert() {
        assert_eq!(ins(0x1E20_C041, VA).to_string(), "fabs s1, s2");
        assert_eq!(ins(0x1E21_4083, VA).to_string(), "fneg s3, s4");
        assert_eq!(ins(0x1E21_C0C5, VA).to_string(), "fsqrt s5, s6");
        assert_eq!(ins(0x1E60_C041, VA).to_string(), "fabs d1, d2");
        assert_eq!(ins(0x1E61_4083, VA).to_string(), "fneg d3, d4");
        assert_eq!(ins(0x1E61_C0C5, VA).to_string(), "fsqrt d5, d6");
        let i = ins(0x1E22_C107, VA);
        assert_eq!(
            i.opcode,
            Opcode::FCvtPrec {
                to_double: true,
                rd: 7,
                rn: 8
            }
        );
        assert_eq!(i.to_string(), "fcvt d7, s8");
        assert_eq!(ins(0x1E62_4149, VA).to_string(), "fcvt s9, d10");
        // All seven FRINT roundings, both precisions.
        assert_eq!(ins(0x1E24_4020, VA).to_string(), "frintn s0, s1");
        assert_eq!(ins(0x1E24_C062, VA).to_string(), "frintp s2, s3");
        assert_eq!(ins(0x1E25_40A4, VA).to_string(), "frintm s4, s5");
        assert_eq!(ins(0x1E25_C0E6, VA).to_string(), "frintz s6, s7");
        assert_eq!(ins(0x1E26_4128, VA).to_string(), "frinta s8, s9");
        assert_eq!(ins(0x1E27_416A, VA).to_string(), "frintx s10, s11");
        assert_eq!(ins(0x1E27_C1AC, VA).to_string(), "frinti s12, s13");
        assert_eq!(ins(0x1E64_4020, VA).to_string(), "frintn d0, d1");
        assert_eq!(ins(0x1E64_C062, VA).to_string(), "frintp d2, d3");
        assert_eq!(ins(0x1E65_40A4, VA).to_string(), "frintm d4, d5");
        assert_eq!(ins(0x1E65_C0E6, VA).to_string(), "frintz d6, d7");
        assert_eq!(ins(0x1E66_4128, VA).to_string(), "frinta d8, d9");
        assert_eq!(ins(0x1E67_416A, VA).to_string(), "frintx d10, d11");
        assert_eq!(ins(0x1E67_C1AC, VA).to_string(), "frinti d12, d13");
    }

    #[test]
    fn fp_compare_select_and_conditional_compare() {
        let i = ins(0x1E21_2000, VA);
        assert_eq!(
            i.opcode,
            Opcode::Fcmp {
                double: false,
                signal: false,
                rn: 0,
                rm: Some(1)
            }
        );
        assert_eq!(i.to_string(), "fcmp s0, s1");
        assert_eq!(ins(0x1E20_2048, VA).to_string(), "fcmp s2, #0.0");
        assert_eq!(ins(0x1E24_2070, VA).to_string(), "fcmpe s3, s4");
        assert_eq!(ins(0x1E20_20B8, VA).to_string(), "fcmpe s5, #0.0");
        assert_eq!(ins(0x1E61_2000, VA).to_string(), "fcmp d0, d1");
        assert_eq!(ins(0x1E60_2048, VA).to_string(), "fcmp d2, #0.0");
        assert_eq!(ins(0x1E64_2070, VA).to_string(), "fcmpe d3, d4");
        assert_eq!(ins(0x1E60_20B8, VA).to_string(), "fcmpe d5, #0.0");
        assert_eq!(ins(0x1E21_1404, VA).to_string(), "fccmp s0, s1, #0x4, ne");
        assert_eq!(ins(0x1E23_A45F, VA).to_string(), "fccmpe s2, s3, #0xf, ge");
        assert_eq!(ins(0x1E61_1404, VA).to_string(), "fccmp d0, d1, #0x4, ne");
        assert_eq!(ins(0x1E63_A45F, VA).to_string(), "fccmpe d2, d3, #0xf, ge");
        assert_eq!(ins(0x1E22_4C20, VA).to_string(), "fcsel s0, s1, s2, mi");
        assert_eq!(ins(0x1E65_5C83, VA).to_string(), "fcsel d3, d4, d5, pl");
    }

    #[test]
    fn fp_integer_conversions() {
        let i = ins(0x1E22_0020, VA);
        assert_eq!(
            i.opcode,
            Opcode::FcvtToFp {
                sf: false,
                double: false,
                unsigned: false,
                rd: 0,
                rn: 1
            }
        );
        assert_eq!(i.to_string(), "scvtf s0, w1");
        assert_eq!(ins(0x1E62_0062, VA).to_string(), "scvtf d2, w3");
        assert_eq!(ins(0x9E22_00A4, VA).to_string(), "scvtf s4, x5");
        assert_eq!(ins(0x9E62_00E6, VA).to_string(), "scvtf d6, x7");
        assert_eq!(ins(0x1E23_0128, VA).to_string(), "ucvtf s8, w9");
        assert_eq!(ins(0x1E63_016A, VA).to_string(), "ucvtf d10, w11");
        assert_eq!(ins(0x9E23_01AC, VA).to_string(), "ucvtf s12, x13");
        assert_eq!(ins(0x9E63_01EE, VA).to_string(), "ucvtf d14, x15");
        let i = ins(0x1E38_0020, VA);
        assert_eq!(
            i.opcode,
            Opcode::FcvtFromFp {
                sf: false,
                double: false,
                unsigned: false,
                round: FpRound::Z,
                rd: 0,
                rn: 1
            }
        );
        assert_eq!(i.to_string(), "fcvtzs w0, s1");
        assert_eq!(ins(0x9E38_0062, VA).to_string(), "fcvtzs x2, s3");
        assert_eq!(ins(0x1E78_00A4, VA).to_string(), "fcvtzs w4, d5");
        assert_eq!(ins(0x9E78_00E6, VA).to_string(), "fcvtzs x6, d7");
        assert_eq!(ins(0x1E39_0128, VA).to_string(), "fcvtzu w8, s9");
        assert_eq!(ins(0x9E39_016A, VA).to_string(), "fcvtzu x10, s11");
        assert_eq!(ins(0x1E79_01AC, VA).to_string(), "fcvtzu w12, d13");
        assert_eq!(ins(0x9E79_01EE, VA).to_string(), "fcvtzu x14, d15");
        // The rounding-directed family.
        assert_eq!(ins(0x1E20_0020, VA).to_string(), "fcvtns w0, s1");
        assert_eq!(ins(0x1E21_0062, VA).to_string(), "fcvtnu w2, s3");
        assert_eq!(ins(0x1E68_00A4, VA).to_string(), "fcvtps w4, d5");
        assert_eq!(ins(0x9E69_00E6, VA).to_string(), "fcvtpu x6, d7");
        assert_eq!(ins(0x1E30_0128, VA).to_string(), "fcvtms w8, s9");
        assert_eq!(ins(0x1E71_016A, VA).to_string(), "fcvtmu w10, d11");
        assert_eq!(ins(0x9E24_01AC, VA).to_string(), "fcvtas x12, s13");
        assert_eq!(ins(0x1E65_01EE, VA).to_string(), "fcvtau w14, d15");
        // The scalar-integer (FP register to FP register) forms.
        let i = ins(0x5E21_D820, VA);
        assert_eq!(
            i.opcode,
            Opcode::FcvtIntScalar {
                double: false,
                unsigned: false,
                rd: 0,
                rn: 1
            }
        );
        assert_eq!(i.to_string(), "scvtf s0, s1");
        assert_eq!(ins(0x5E61_D862, VA).to_string(), "scvtf d2, d3");
        assert_eq!(ins(0x7E21_D8A4, VA).to_string(), "ucvtf s4, s5");
        assert_eq!(ins(0x7E61_D8E6, VA).to_string(), "ucvtf d6, d7");
    }

    #[test]
    fn element_moves_decode_and_render() {
        let i = ins(0x4E08_0D00, VA);
        assert_eq!(
            i.opcode,
            Opcode::DupGp {
                q: true,
                size: 3,
                rd: 0,
                rn: 8
            }
        );
        assert_eq!(i.to_string(), "dup v0.2d, x8");
        assert_eq!(ins(0x4E04_0D21, VA).to_string(), "dup v1.4s, w9");
        assert_eq!(ins(0x4E02_0D42, VA).to_string(), "dup v2.8h, w10");
        assert_eq!(ins(0x4E01_0D63, VA).to_string(), "dup v3.16b, w11");
        assert_eq!(ins(0x0E04_0D84, VA).to_string(), "dup v4.2s, w12");
        assert_eq!(ins(0x0E01_0DA5, VA).to_string(), "dup v5.8b, w13");
        assert_eq!(ins(0x0E02_0DC6, VA).to_string(), "dup v6.4h, w14");
        // DUP (element, scalar) — the `mov` spelling.
        let i = ins(0x5E18_0420, VA);
        assert_eq!(
            i.opcode,
            Opcode::DupElemScalar {
                size: 3,
                index: 1,
                rd: 0,
                rn: 1
            }
        );
        assert_eq!(i.to_string(), "mov d0, v1.d[1]");
        assert_eq!(ins(0x5E14_0462, VA).to_string(), "mov s2, v3.s[2]");
        assert_eq!(ins(0x5E0E_04A4, VA).to_string(), "mov h4, v5.h[3]");
        assert_eq!(ins(0x5E13_04E6, VA).to_string(), "mov b6, v7.b[9]");
        assert_eq!(ins(0x4E08_0420, VA).to_string(), "dup v0.2d, v1.d[0]");
        assert_eq!(ins(0x4E0C_0462, VA).to_string(), "dup v2.4s, v3.s[1]");
        // UMOV / SMOV.
        let i = ins(0x0E0C_3C20, VA);
        assert_eq!(
            i.opcode,
            Opcode::Umov {
                sf: false,
                size: 2,
                index: 1,
                rd: 0,
                rn: 1
            }
        );
        assert_eq!(i.to_string(), "umov w0, v1.s[1]");
        assert_eq!(ins(0x0E12_3C62, VA).to_string(), "umov w2, v3.h[4]");
        assert_eq!(ins(0x0E17_3CA4, VA).to_string(), "umov w4, v5.b[11]");
        assert_eq!(ins(0x4E18_3CE6, VA).to_string(), "umov x6, v7.d[1]");
        assert_eq!(ins(0x0E0A_2C20, VA).to_string(), "smov w0, v1.h[2]");
        assert_eq!(ins(0x0E0B_2C62, VA).to_string(), "smov w2, v3.b[5]");
        assert_eq!(ins(0x4E1C_2CA4, VA).to_string(), "smov x4, v5.s[3]");
        assert_eq!(ins(0x4E1E_2CE6, VA).to_string(), "smov x6, v7.h[7]");
        assert_eq!(ins(0x4E1F_2D28, VA).to_string(), "smov x8, v9.b[15]");
        // INS, both forms.
        let i = ins(0x4E18_1C20, VA);
        assert_eq!(
            i.opcode,
            Opcode::InsGp {
                size: 3,
                index: 1,
                rd: 0,
                rn: 1
            }
        );
        assert_eq!(i.to_string(), "ins v0.d[1], x1");
        assert_eq!(ins(0x4E1C_1C62, VA).to_string(), "ins v2.s[3], w3");
        assert_eq!(ins(0x4E16_1CA4, VA).to_string(), "ins v4.h[5], w5");
        assert_eq!(ins(0x4E1B_1CE6, VA).to_string(), "ins v6.b[13], w7");
        let i = ins(0x6E08_4420, VA);
        assert_eq!(
            i.opcode,
            Opcode::InsElem {
                size: 3,
                dst: 0,
                src: 1,
                rd: 0,
                rn: 1
            }
        );
        assert_eq!(i.to_string(), "ins v0.d[0], v1.d[1]");
        assert_eq!(ins(0x6E0C_4462, VA).to_string(), "ins v2.s[1], v3.s[2]");
        assert_eq!(ins(0x6E0A_64A4, VA).to_string(), "ins v4.h[2], v5.h[6]");
        assert_eq!(ins(0x6E07_64E6, VA).to_string(), "ins v6.b[3], v7.b[12]");
    }

    #[test]
    fn exclusives_and_ordered_accesses() {
        let i = ins(0xC8DF_FC20, VA);
        assert_eq!(i.opcode, Opcode::Ldar { size: 3, rt: 0, rn: 1 });
        assert_eq!(i.flow, Flow::Sequential);
        assert_eq!(i.to_string(), "ldar x0, [x1]");
        assert_eq!(ins(0x88DF_FC62, VA).to_string(), "ldar w2, [x3]");
        assert_eq!(ins(0x08DF_FCA4, VA).to_string(), "ldarb w4, [x5]");
        assert_eq!(ins(0x48DF_FCE6, VA).to_string(), "ldarh w6, [x7]");
        assert_eq!(ins(0xC89F_FD28, VA).to_string(), "stlr x8, [x9]");
        assert_eq!(ins(0x889F_FD6A, VA).to_string(), "stlr w10, [x11]");
        assert_eq!(ins(0x089F_FDAC, VA).to_string(), "stlrb w12, [x13]");
        assert_eq!(ins(0x489F_FDEE, VA).to_string(), "stlrh w14, [x15]");
        let i = ins(0xC85F_7C20, VA);
        assert_eq!(
            i.opcode,
            Opcode::Ldxr {
                size: 3,
                acquire: false,
                rt: 0,
                rn: 1
            }
        );
        assert_eq!(i.to_string(), "ldxr x0, [x1]");
        assert_eq!(ins(0x885F_7C62, VA).to_string(), "ldxr w2, [x3]");
        assert_eq!(ins(0x085F_7CA4, VA).to_string(), "ldxrb w4, [x5]");
        assert_eq!(ins(0x485F_7CE6, VA).to_string(), "ldxrh w6, [x7]");
        assert_eq!(ins(0xC85F_FD28, VA).to_string(), "ldaxr x8, [x9]");
        assert_eq!(ins(0x885F_FD6A, VA).to_string(), "ldaxr w10, [x11]");
        assert_eq!(ins(0x085F_FDAC, VA).to_string(), "ldaxrb w12, [x13]");
        assert_eq!(ins(0x485F_FDEE, VA).to_string(), "ldaxrh w14, [x15]");
        let i = ins(0xC800_7C41, VA);
        assert_eq!(
            i.opcode,
            Opcode::Stxr {
                size: 3,
                release: false,
                ws: 0,
                rt: 1,
                rn: 2
            }
        );
        assert_eq!(i.to_string(), "stxr w0, x1, [x2]");
        assert_eq!(ins(0x8803_7CA4, VA).to_string(), "stxr w3, w4, [x5]");
        assert_eq!(ins(0x0806_7D07, VA).to_string(), "stxrb w6, w7, [x8]");
        assert_eq!(ins(0x4809_7D6A, VA).to_string(), "stxrh w9, w10, [x11]");
        assert_eq!(ins(0xC80C_FDCD, VA).to_string(), "stlxr w12, x13, [x14]");
        assert_eq!(ins(0x880F_FE30, VA).to_string(), "stlxr w15, w16, [x17]");
        assert_eq!(ins(0x0812_FE93, VA).to_string(), "stlxrb w18, w19, [x20]");
        assert_eq!(ins(0x4815_FEF6, VA).to_string(), "stlxrh w21, w22, [x23]");
        // The pair and LORegion rows stay refused.
        assert_eq!(ins(0xC87F_7C41, VA).opcode, Opcode::Unknown(0xC87F_7C41));
        assert_eq!(ins(0xC8DF_7C20, VA).opcode, Opcode::Unknown(0xC8DF_7C20));
    }

    #[test]
    fn pointer_authentication_decodes_and_renders() {
        let i = ins(0xD65F_0BFF, VA);
        assert_eq!(i.opcode, Opcode::RetA { key_b: false });
        assert_eq!(i.flow, Flow::Return);
        assert_eq!(i.to_string(), "retaa");
        assert_eq!(ins(0xD65F_0FFF, VA).to_string(), "retab");
        let i = ins(0xD71F_0801, VA);
        assert_eq!(
            i.opcode,
            Opcode::BrAuth {
                link: false,
                key_b: false,
                zero: false,
                rn: 0,
                rm: 1
            }
        );
        assert_eq!(i.flow, Flow::IndirectJump);
        assert_eq!(i.to_string(), "braa x0, x1");
        assert_eq!(ins(0xD71F_0C43, VA).to_string(), "brab x2, x3");
        assert_eq!(ins(0xD61F_089F, VA).to_string(), "braaz x4");
        assert_eq!(ins(0xD61F_0CBF, VA).to_string(), "brabz x5");
        let i = ins(0xD73F_08C7, VA);
        assert_eq!(i.flow, Flow::IndirectCall);
        assert_eq!(i.to_string(), "blraa x6, x7");
        assert_eq!(ins(0xD73F_0D09, VA).to_string(), "blrab x8, x9");
        assert_eq!(ins(0xD63F_095F, VA).to_string(), "blraaz x10");
        assert_eq!(ins(0xD63F_0D7F, VA).to_string(), "blrabz x11");
        // The dp-1source PAC row.
        let i = ins(0xDAC1_0020, VA);
        assert_eq!(
            i.opcode,
            Opcode::PacGpr {
                auth: false,
                key_b: false,
                zero: false,
                rd: 0,
                rn: 1
            }
        );
        assert_eq!(i.to_string(), "pacia x0, x1");
        assert_eq!(ins(0xDAC1_0462, VA).to_string(), "pacib x2, x3");
        assert_eq!(ins(0xDAC1_10A4, VA).to_string(), "autia x4, x5");
        assert_eq!(ins(0xDAC1_14E6, VA).to_string(), "autib x6, x7");
        assert_eq!(ins(0xDAC1_23E8, VA).to_string(), "paciza x8");
        assert_eq!(ins(0xDAC1_27E9, VA).to_string(), "pacizb x9");
        assert_eq!(ins(0xDAC1_33EA, VA).to_string(), "autiza x10");
        assert_eq!(ins(0xDAC1_37EB, VA).to_string(), "autizb x11");
        assert_eq!(ins(0xDAC1_43EC, VA).to_string(), "xpaci x12");
        assert_eq!(ins(0xDAC1_47ED, VA).to_string(), "xpacd x13");
        // A Z form with Rn != 11111 is unallocated, and the D-key data
        // ops are a recorded gap.
        assert_eq!(ins(0xDAC1_2028, VA).opcode, Opcode::Unknown(0xDAC1_2028));
        assert_eq!(ins(0xDAC1_0820, VA).opcode, Opcode::Unknown(0xDAC1_0820));
        // The hint-space PAC ops.
        let i = ins(0xD503_233F, VA);
        assert_eq!(
            i.opcode,
            Opcode::PacHint {
                auth: false,
                key_b: false
            }
        );
        assert_eq!(i.to_string(), "paciasp");
        assert_eq!(ins(0xD503_23BF, VA).to_string(), "autiasp");
        assert_eq!(ins(0xD503_237F, VA).to_string(), "pacibsp");
        assert_eq!(ins(0xD503_23FF, VA).to_string(), "autibsp");
    }

    #[test]
    fn udf_is_a_halt_and_only_in_the_reserved_row() {
        let i = ins(0x0000_0000, VA);
        assert_eq!(i.opcode, Opcode::Udf { imm: 0 });
        assert_eq!(i.flow, Flow::Halt);
        assert_eq!(i.to_string(), "udf #0x0");
        let i = ins(0x0000_04B8, VA);
        assert_eq!(i.opcode, Opcode::Udf { imm: 0x4B8 });
        assert_eq!(i.to_string(), "udf #0x4b8");
        // A reserved-group word with any upper bit set is not UDF.
        assert_eq!(ins(0x0001_0000, VA).opcode, Opcode::Unknown(0x0001_0000));
    }

    #[test]
    fn one_source_bits_extr_and_ldpsw() {
        let i = ins(0x5AC0_1020, VA);
        assert_eq!(
            i.opcode,
            Opcode::Bits1 {
                op: Bit1Op::Clz,
                sf: false,
                rd: 0,
                rn: 1
            }
        );
        assert_eq!(i.to_string(), "clz w0, w1");
        assert_eq!(ins(0xDAC0_1062, VA).to_string(), "clz x2, x3");
        assert_eq!(ins(0x5AC0_14A4, VA).to_string(), "cls w4, w5");
        assert_eq!(ins(0xDAC0_14E6, VA).to_string(), "cls x6, x7");
        assert_eq!(ins(0x5AC0_0128, VA).to_string(), "rbit w8, w9");
        assert_eq!(ins(0xDAC0_016A, VA).to_string(), "rbit x10, x11");
        assert_eq!(ins(0x5AC0_09AC, VA).to_string(), "rev w12, w13");
        assert_eq!(ins(0xDAC0_0DEE, VA).to_string(), "rev x14, x15");
        assert_eq!(ins(0x5AC0_0630, VA).to_string(), "rev16 w16, w17");
        assert_eq!(ins(0xDAC0_0672, VA).to_string(), "rev16 x18, x19");
        assert_eq!(ins(0xDAC0_0AB4, VA).to_string(), "rev32 x20, x21");
        // REV's 32-bit row tops out at opcode 000010; 000011 needs sf.
        assert_eq!(ins(0x5AC0_0C20, VA).opcode, Opcode::Unknown(0x5AC0_0C20));
        let i = ins(0x1382_1420, VA);
        assert_eq!(
            i.opcode,
            Opcode::Extr {
                sf: false,
                rd: 0,
                rn: 1,
                rm: 2,
                lsb: 5
            }
        );
        assert_eq!(i.to_string(), "extr w0, w1, w2, #5");
        assert_eq!(ins(0x93C5_8483, VA).to_string(), "extr x3, x4, x5, #33");
        // The ROR alias when Rn == Rm.
        assert_eq!(ins(0x1387_24E6, VA).to_string(), "ror w6, w7, #9");
        assert_eq!(ins(0x93C9_8128, VA).to_string(), "ror x8, x9, #32");
        // A 32-bit lsb of 32+ and an N != sf word are unallocated.
        assert_eq!(ins(0x1382_8420, VA).opcode, Opcode::Unknown(0x1382_8420));
        assert_eq!(ins(0x93A5_8483, VA).opcode, Opcode::Unknown(0x93A5_8483));
        let i = ins(0x6941_0440, VA);
        assert_eq!(
            i.opcode,
            Opcode::LdpSw {
                rt: 0,
                rt2: 1,
                rn: 2,
                mode: AddrMode::Offset(8)
            }
        );
        assert_eq!(i.to_string(), "ldpsw x0, x1, [x2, #0x8]");
        assert_eq!(
            ins(0x69FF_90A3, VA).to_string(),
            "ldpsw x3, x4, [x5, #-0x4]!"
        );
        assert_eq!(ins(0x68C2_1D06, VA).to_string(), "ldpsw x6, x7, [x8], #0x10");
        // The store side of the LDPSW row is unallocated.
        assert_eq!(ins(0x2900_0440, VA).opcode, ins(0x2900_0440, VA).opcode);
        assert_eq!(ins(0x6900_0440, VA).opcode, Opcode::Unknown(0x6900_0440));
    }

    #[test]
    fn targets_wrap_at_the_address_space_edges() {
        // B -0x10 at va 0 wraps below zero without panicking.
        let i = ins(0x17FF_FFFC, 0);
        assert_eq!(i.flow, Flow::Jump(0xFFFF_FFFF_FFFF_FFF0));
        // B +0x10 at the top of the address space wraps above it.
        let i = ins(0x1400_0004, u64::MAX - 3);
        assert_eq!(i.flow, Flow::Jump(0xC));
    }

    #[test]
    fn extra_trailing_bytes_are_ignored() {
        let mut buf = 0x1400_0004u32.to_le_bytes().to_vec();
        buf.extend_from_slice(&[0xAA; 5]);
        let i = decode(&buf, VA).unwrap();
        assert_eq!(i.opcode, Opcode::B { target: 0x1010 });
        assert_eq!(i.raw, 0x1400_0004);
    }

    #[test]
    fn truncated_input_is_an_error_not_a_panic() {
        for len in 0..4 {
            let bytes = vec![0xFFu8; len];
            assert_eq!(
                decode(&bytes, VA).unwrap_err(),
                ParseError::UnexpectedEof {
                    offset: 0,
                    needed: 4,
                    available: len,
                }
            );
        }
    }

    /// The decoder must be total: no word may panic, whatever the VA.
    /// Sweeps every value of the high 16 bits (where all the opcode
    /// selectors live) against fixed low-half patterns, plus a large
    /// deterministic xorshift sample of full words.
    #[test]
    fn decode_is_total_over_a_large_deterministic_sample() {
        let low_patterns = [0x0000u32, 0xFFFF, 0x03E0, 0x7C1F, 0x8421, 0x5555];
        let vas = [0u64, 0x1000, 0x8000_0000_0000_0000, u64::MAX - 3];
        for hi in 0..=0xFFFFu32 {
            for &lo in &low_patterns {
                let w = (hi << 16) | lo;
                for &va in &vas {
                    let i = decode(&w.to_le_bytes(), va).unwrap();
                    // Display must also be total.
                    let _ = i.to_string();
                }
            }
        }

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..1_000_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let w = state as u32;
            let i = decode(&w.to_le_bytes(), state).unwrap();
            let _ = i.to_string();
        }
    }
}
