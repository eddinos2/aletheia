//! AArch64 (A64) instruction-text rendering.
//!
//! The listing half of the A64 story: [`crate::aarch64`] answers "where
//! does control go from here", this module answers "what does the analyst
//! read on this line". Both are clean-room implementations of the same
//! public Arm Architecture Reference Manual encoding tables, but they stay
//! deliberately separate — as [`crate::asmtext`] puts it, a text renderer
//! must be free to grow coverage (here: the whole data-processing-register
//! group, which flow analysis never needs) without touching the decoder
//! that analysis correctness rests on.
//!
//! Rendering is best-effort by contract. An encoding outside the covered
//! subset returns `None` and the listing falls back to the raw word;
//! nothing here guesses, because an analyst trusts what a disassembler
//! prints. A64 is fixed-width, so a gap costs exactly one line and can
//! never desynchronize the instruction stream.
//!
//! Covered:
//!
//! - **Branches**: `b`, `bl`, `b.<cond>` (all 16 codes), `cbz`/`cbnz`,
//!   `tbz`/`tbnz`, `br`/`blr`/`ret`.
//! - **Pc-relative**: `adr`, `adrp`.
//! - **Moves**: `movz`/`movn`/`movk` with the shift operand, plus the
//!   `mov <Rd>, #imm` and `mov <Rd>, <Rm>` aliases.
//! - **Loads/stores**: `ldr`/`str` (+`b`/`h` widths) in unsigned-offset,
//!   unscaled (`ldur`/`stur`), pre/post-index and register-offset forms;
//!   the sign-extending `ldrs{b,h,w}` (and unscaled `ldurs{b,h,w}`) in the
//!   same addressing modes; `ldp`/`stp` signed-offset and pre/post-index;
//!   `ldr` literal.
//! - **ALU**: `add`/`sub`(`s`) immediate and shifted-register with the
//!   `cmp`/`cmn`/`mov` aliases; `and`/`orr`/`eor` shifted-register;
//!   `lsl`/`lsr`/`asr` immediate (the bitfield aliases); `mul`;
//!   `sdiv`/`udiv`; the conditional-select family
//!   `csel`/`csinc`/`csinv`/`csneg` with the `cset`/`csetm`/`cinc`/`cinv`/
//!   `cneg` aliases.
//! - **System**: `nop` and the named hints, `svc`/`hvc`/`smc`,
//!   `brk`/`hlt`.
//!
//! Not covered — every one renders `None` rather than a guess: SIMD/FP
//! and their loads/stores, `prfm`, exclusives, atomics, extended-register
//! addressing, logical-immediate and the general bitfield/extract forms,
//! `bic`/`orn`/`eon`/`ands`, shifted-register carry forms, pointer
//! authentication, `eret`, system-register access, and unnamed hints.
//!
//! Conventions:
//!
//! - Pc-relative operands (`b`, `adrp`, `ldr` literal, ...) print as
//!   absolute virtual addresses, computed with wrapping arithmetic and
//!   full sign extension so a target near either edge of the address
//!   space is rendered, not panicked on.
//! - Immediate *values* print in hex (`#0x10`, `#-0x10`); immediate
//!   *positions* — shift amounts and bit numbers — print in decimal
//!   (`lsl #12`, `tbz x0, #33, ...`), matching assembler convention.
//! - Aliases are applied where they are unconditional and local to one
//!   encoding (`cmp`, `cmn`, `mov`, `lsl`/`lsr`/`asr`, `mul`, bare `ret`).
//!   Where an assembler would go further — `neg` for a `sub` from the zero
//!   register, `mov #<64-bit constant>` for a shifted `movz` — the base
//!   instruction is printed instead: still exact, still reassembles to the
//!   same word, and one fewer rule to get subtly wrong.
//! - `sp`/`wsp` versus `xzr`/`wzr` follows the encoding class rather than
//!   the register number: register 31 is the stack pointer only for the
//!   base of a load/store, for `<Rn>` of add/sub immediate, and for
//!   `<Rd>` of add/sub immediate without flag setting. Everywhere else it
//!   reads as the zero register.

use crate::aarch64::{Cond, Instruction};
use crate::asmtext::AsmFormatter;

/// The AArch64 text formatter.
#[derive(Debug, Clone, Copy, Default)]
pub struct A64Text;

impl AsmFormatter for A64Text {
    /// Render the A64 word at `bytes[0..4]`, located at `va`.
    ///
    /// `None` when fewer than 4 bytes are available or the encoding is
    /// outside the covered subset. Never panics, for any input.
    fn format(&self, bytes: &[u8], va: u64) -> Option<String> {
        let word = bytes.get(..Instruction::SIZE)?;
        let w = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        render(w, va)
    }
}

/// Render one instruction word, dispatching on `op0` (bits \[28:25\]) the
/// same way the decoder's top-level table does.
fn render(w: u32, va: u64) -> Option<String> {
    match bits(w, 28, 25) {
        // 100x: data processing — immediate.
        0b1000 | 0b1001 => dp_imm(w, va),
        // 101x: branches, exception generation, system.
        0b1010 | 0b1011 => branch_system(w, va),
        // x101: data processing — register.
        0b0101 | 0b1101 => dp_reg(w),
        // x1x0: loads and stores.
        0b0100 | 0b0110 | 0b1100 | 0b1110 => load_store(w, va),
        // Reserved and SVE.
        _ => None,
    }
}

// ---------------------------------------------------------------------
// Bit-field and formatting primitives
// ---------------------------------------------------------------------

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

/// Absolute VA of a pc-relative branch or literal: `va + sext(imm) * 4`,
/// wrapping at the edges of the address space rather than panicking.
fn branch_target(va: u64, imm: u32, width: u32) -> u64 {
    va.wrapping_add((sext(imm, width) << 2) as u64)
}

/// A general-purpose register operand, with 31 as the zero register.
fn gp(sf: bool, r: u8) -> String {
    match (sf, r) {
        (true, 31) => "xzr".into(),
        (false, 31) => "wzr".into(),
        (true, _) => format!("x{r}"),
        (false, _) => format!("w{r}"),
    }
}

/// A general-purpose register operand, with 31 as the stack pointer.
fn gp_sp(sf: bool, r: u8) -> String {
    match (sf, r) {
        (true, 31) => "sp".into(),
        (false, 31) => "wsp".into(),
        _ => gp(sf, r),
    }
}

/// An unsigned immediate value: `#0x1f`.
fn uimm(v: u32) -> String {
    format!("#{v:#x}")
}

/// A signed immediate value: `#0x10` / `#-0x10`.
fn simm(v: i64) -> String {
    if v < 0 {
        format!("#-{:#x}", v.unsigned_abs())
    } else {
        format!("#{v:#x}")
    }
}

/// Which way a load/store writes back to its base register.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Index {
    /// `[Xn, #imm]`: no writeback.
    None,
    /// `[Xn, #imm]!`: base updated before the access.
    Pre,
    /// `[Xn], #imm`: base updated after the access.
    Post,
}

/// A memory operand. The base is the one place register 31 always reads
/// as `sp`.
fn mem(rn: u8, off: i64, index: Index) -> String {
    let base = gp_sp(true, rn);
    match index {
        Index::None if off == 0 => format!("[{base}]"),
        Index::None => format!("[{base}, {}]", simm(off)),
        Index::Pre => format!("[{base}, {}]!", simm(off)),
        Index::Post => format!("[{base}], {}", simm(off)),
    }
}

/// The optional shift operand of a shifted-register data-processing
/// instruction.
///
/// An `lsl #0` shift carries no information and assemblers omit it; every
/// other zero-amount shift is a distinct encoding, so it is printed to
/// keep the text a faithful reading of the word.
fn shift_operand(kind: u32, amount: u32) -> String {
    let name = match kind {
        0b00 if amount == 0 => return String::new(),
        0b00 => "lsl",
        0b01 => "lsr",
        0b10 => "asr",
        _ => "ror",
    };
    format!(", {name} #{amount}")
}

// ---------------------------------------------------------------------
// Data processing — immediate
// ---------------------------------------------------------------------

/// Data processing — immediate (`op0 == 100x`), keyed on bits \[25:23\].
fn dp_imm(w: u32, va: u64) -> Option<String> {
    match bits(w, 25, 23) {
        // 00x: pc-relative addressing; bit 23 is immhi data, not a selector.
        0b000 | 0b001 => Some(pc_rel(w, va)),
        0b010 => add_sub_imm(w),
        0b101 => move_wide(w),
        0b110 => bitfield(w),
        // Add/sub with tags, logical immediate, extract.
        _ => None,
    }
}

/// `ADR`/`ADRP`: `op immlo(2) 10000 immhi(19) Rd`, rendered as the
/// absolute address they compute.
fn pc_rel(w: u32, va: u64) -> String {
    let rd = gp(true, bits(w, 4, 0) as u8);
    let off = sext((bits(w, 23, 5) << 2) | bits(w, 30, 29), 21);
    if bit(w, 31) {
        // ADRP: the offset counts 4 KiB pages from the pc's own page.
        let target = (va & !0xFFF).wrapping_add((off << 12) as u64);
        format!("adrp {rd}, {target:#x}")
    } else {
        let target = va.wrapping_add(off as u64);
        format!("adr {rd}, {target:#x}")
    }
}

/// `ADD{S}`/`SUB{S}` immediate: `sf op S 100010 sh imm12 Rn Rd`.
///
/// The shift is rendered as the assembler writes it (`, lsl #12`) rather
/// than folded into the immediate, so the text round-trips to the same
/// encoding.
fn add_sub_imm(w: u32) -> Option<String> {
    let sf = bit(w, 31);
    let sub = bit(w, 30);
    let set_flags = bit(w, 29);
    let shifted = bit(w, 22);
    let imm12 = bits(w, 21, 10);
    let rd = bits(w, 4, 0) as u8;
    let rn = bits(w, 9, 5) as u8;
    let shift = if shifted { ", lsl #12" } else { "" };
    let imm = uimm(imm12);

    // MOV (to/from SP) is the preferred form of a zero-immediate ADD
    // naming the stack pointer on either side.
    if !sub && !set_flags && !shifted && imm12 == 0 && (rd == 31 || rn == 31) {
        return Some(format!("mov {}, {}", gp_sp(sf, rd), gp_sp(sf, rn)));
    }
    // Flag-setting forms discard their result into the zero register:
    // that is CMP/CMN.
    if set_flags && rd == 31 {
        let mn = if sub { "cmp" } else { "cmn" };
        return Some(format!("{mn} {}, {imm}{shift}", gp_sp(sf, rn)));
    }
    let mn = match (sub, set_flags) {
        (false, false) => "add",
        (false, true) => "adds",
        (true, false) => "sub",
        (true, true) => "subs",
    };
    // Rd is the stack pointer only in the non-flag-setting forms; ADDS/SUBS
    // write the zero register there.
    let rd = if set_flags { gp(sf, rd) } else { gp_sp(sf, rd) };
    Some(format!("{mn} {rd}, {}, {imm}{shift}", gp_sp(sf, rn)))
}

/// `MOVN`/`MOVZ`/`MOVK`: `sf opc 100101 hw imm16 Rd`.
fn move_wide(w: u32) -> Option<String> {
    let sf = bit(w, 31);
    let hw = bits(w, 22, 21);
    if !sf && hw >= 2 {
        // The 32-bit forms only allow shifts of 0 and 16.
        return None;
    }
    let rd = gp(sf, bits(w, 4, 0) as u8);
    let imm = uimm(bits(w, 20, 5));
    let shift = hw * 16;
    let mn = match bits(w, 30, 29) {
        0b00 => "movn",
        0b10 => "movz",
        0b11 => "movk",
        _ => return None,
    };
    // An unshifted MOVZ is written as MOV of the plain immediate. Shifted
    // forms would need the assembled 64-bit constant to stay faithful, so
    // they keep the MOVZ spelling.
    if mn == "movz" && shift == 0 {
        return Some(format!("mov {rd}, {imm}"));
    }
    if shift == 0 {
        Some(format!("{mn} {rd}, {imm}"))
    } else {
        Some(format!("{mn} {rd}, {imm}, lsl #{shift}"))
    }
}

/// The `LSL`/`LSR`/`ASR` immediate aliases of `SBFM`/`UBFM`:
/// `sf opc 100110 N immr imms Rn Rd`.
///
/// Only the three shift aliases are rendered; the bitfield-insert,
/// bitfield-extract and sign/zero-extend forms of the same encodings are
/// left to a later pass.
fn bitfield(w: u32) -> Option<String> {
    let sf = bit(w, 31);
    // N must track the operand size; other combinations are unallocated.
    if bit(w, 22) != sf {
        return None;
    }
    let size = if sf { 64 } else { 32 };
    let immr = bits(w, 21, 16);
    let imms = bits(w, 15, 10);
    if immr >= size || imms >= size {
        return None;
    }
    let rd = gp(sf, bits(w, 4, 0) as u8);
    let rn = gp(sf, bits(w, 9, 5) as u8);
    match bits(w, 30, 29) {
        // SBFM with imms == size-1 is ASR #immr.
        0b00 if imms == size - 1 => Some(format!("asr {rd}, {rn}, #{immr}")),
        0b10 => {
            if imms == size - 1 {
                // UBFM with imms == size-1 is LSR #immr.
                Some(format!("lsr {rd}, {rn}, #{immr}"))
            } else if imms + 1 == immr {
                // ... and immr == imms+1 is LSL #(size-1-imms).
                Some(format!("lsl {rd}, {rn}, #{}", size - 1 - imms))
            } else {
                None
            }
        }
        // BFM, and the extract/insert forms of SBFM and UBFM.
        _ => None,
    }
}

// ---------------------------------------------------------------------
// Branches, exception generation and system
// ---------------------------------------------------------------------

/// Branches, exception generation and system (`op0 == 101x`).
fn branch_system(w: u32, va: u64) -> Option<String> {
    // B / BL: op 00101 imm26.
    if w & 0x7C00_0000 == 0x1400_0000 {
        let target = branch_target(va, bits(w, 25, 0), 26);
        let mn = if bit(w, 31) { "bl" } else { "b" };
        return Some(format!("{mn} {target:#x}"));
    }
    // CBZ / CBNZ: sf 011010 op imm19 Rt.
    if w & 0x7E00_0000 == 0x3400_0000 {
        let rt = gp(bit(w, 31), bits(w, 4, 0) as u8);
        let target = branch_target(va, bits(w, 23, 5), 19);
        let mn = if bit(w, 24) { "cbnz" } else { "cbz" };
        return Some(format!("{mn} {rt}, {target:#x}"));
    }
    // TBZ / TBNZ: b5 011011 op b40 imm14 Rt; the bit number is b5:b40 and
    // its top bit also selects the register width.
    if w & 0x7E00_0000 == 0x3600_0000 {
        let bit_no = (bits(w, 31, 31) << 5) | bits(w, 23, 19);
        let rt = gp(bit_no >= 32, bits(w, 4, 0) as u8);
        let target = branch_target(va, bits(w, 18, 5), 14);
        let mn = if bit(w, 24) { "tbnz" } else { "tbz" };
        return Some(format!("{mn} {rt}, #{bit_no}, {target:#x}"));
    }
    // B.cond: 01010100 imm19 0 cond. Bit 4 set is BC.cond, not covered.
    if w & 0xFF00_0010 == 0x5400_0000 {
        let cond = Cond::from(bits(w, 3, 0) as u8).as_str();
        let target = branch_target(va, bits(w, 23, 5), 19);
        return Some(format!("b.{cond} {target:#x}"));
    }
    // Exception generation: 11010100 opc(3) imm16 op2(3) LL(2).
    if w & 0xFF00_0000 == 0xD400_0000 {
        if bits(w, 4, 2) != 0 {
            return None;
        }
        let imm = uimm(bits(w, 20, 5));
        let mn = match (bits(w, 23, 21), bits(w, 1, 0)) {
            (0b000, 0b01) => "svc",
            (0b000, 0b10) => "hvc",
            (0b000, 0b11) => "smc",
            (0b001, 0b00) => "brk",
            (0b010, 0b00) => "hlt",
            // DCPS1/2/3 and unallocated combinations.
            _ => return None,
        };
        return Some(format!("{mn} {imm}"));
    }
    // Hint space: 11010101 00000011 0010 CRm(4) op2(3) 11111. Hints we
    // cannot name are left to the raw-word fallback.
    if w & 0xFFFF_F01F == 0xD503_201F {
        return match bits(w, 11, 5) {
            0 => Some("nop".into()),
            1 => Some("yield".into()),
            2 => Some("wfe".into()),
            3 => Some("wfi".into()),
            4 => Some("sev".into()),
            5 => Some("sevl".into()),
            _ => None,
        };
    }
    // Unconditional branch (register): 1101011 opc(4) op2(5) op3(6) Rn op4(5).
    // Plain BR/BLR/RET require op2 = 11111, op3 = 000000, op4 = 00000; the
    // other combinations are pointer-auth variants or ERET.
    if w & 0xFE00_0000 == 0xD600_0000
        && bits(w, 20, 16) == 0b11111
        && bits(w, 15, 10) == 0
        && bits(w, 4, 0) == 0
    {
        let rn = bits(w, 9, 5) as u8;
        return match bits(w, 24, 21) {
            0b0000 => Some(format!("br {}", gp(true, rn))),
            0b0001 => Some(format!("blr {}", gp(true, rn))),
            // RET defaults to X30 and is then written bare.
            0b0010 if rn == 30 => Some("ret".into()),
            0b0010 => Some(format!("ret {}", gp(true, rn))),
            _ => None,
        };
    }
    None
}

// ---------------------------------------------------------------------
// Data processing — register
// ---------------------------------------------------------------------

/// Data processing — register (`op0 == x101`).
fn dp_reg(w: u32) -> Option<String> {
    // Logical (shifted register): sf opc 01010 shift N Rm imm6 Rn Rd.
    if bits(w, 28, 24) == 0b01010 {
        return logical_shifted(w);
    }
    // Add/subtract (shifted register): sf op S 01011 shift 0 Rm imm6 Rn Rd.
    // Bit 21 set is the extended-register form, not covered.
    if bits(w, 28, 24) == 0b01011 && !bit(w, 21) {
        return add_sub_shifted(w);
    }
    // Conditional select: sf op S 11010100 Rm cond op2(2) Rn Rd (S = 0).
    if bits(w, 29, 21) == 0b0_1101_0100 {
        return cond_select(w);
    }
    // Data processing (2 source): sf 0 S 11010110 Rm opcode(6) Rn Rd.
    if bits(w, 30, 21) == 0b00_1101_0110 {
        let mn = match bits(w, 15, 10) {
            0b000010 => "udiv",
            0b000011 => "sdiv",
            _ => return None,
        };
        return Some(three_reg(w, mn));
    }
    // Data processing (3 source): sf 00 11011 000 Rm o0 Ra Rn Rd. With
    // o0 = 0 (MADD) and Ra = ZR this is MUL; MADD proper is not covered.
    if bits(w, 30, 21) == 0b00_1101_1000 && !bit(w, 15) && bits(w, 14, 10) == 31 {
        return Some(three_reg(w, "mul"));
    }
    None
}

/// `<mn> <Rd>, <Rn>, <Rm>` for the register-only three-operand forms.
fn three_reg(w: u32, mn: &str) -> String {
    let sf = bit(w, 31);
    format!(
        "{mn} {}, {}, {}",
        gp(sf, bits(w, 4, 0) as u8),
        gp(sf, bits(w, 9, 5) as u8),
        gp(sf, bits(w, 20, 16) as u8)
    )
}

/// `AND`/`ORR`/`EOR` shifted register. The inverted-operand siblings
/// (`BIC`/`ORN`/`EON`) and the flag-setting `ANDS` are not covered.
fn logical_shifted(w: u32) -> Option<String> {
    let sf = bit(w, 31);
    let imm6 = bits(w, 15, 10);
    if bit(w, 21) || (!sf && imm6 >= 32) {
        return None;
    }
    let mn = match bits(w, 30, 29) {
        0b00 => "and",
        0b01 => "orr",
        0b10 => "eor",
        _ => return None,
    };
    let kind = bits(w, 23, 22);
    let rd = bits(w, 4, 0) as u8;
    let rn = bits(w, 9, 5) as u8;
    let rm = bits(w, 20, 16) as u8;
    // ORR from the zero register with no shift is the MOV (register) alias.
    if mn == "orr" && rn == 31 && kind == 0b00 && imm6 == 0 {
        return Some(format!("mov {}, {}", gp(sf, rd), gp(sf, rm)));
    }
    let shift = shift_operand(kind, imm6);
    Some(format!(
        "{mn} {}, {}, {}{shift}",
        gp(sf, rd),
        gp(sf, rn),
        gp(sf, rm)
    ))
}

/// `ADD{S}`/`SUB{S}` shifted register, with the `CMP`/`CMN` aliases.
/// Every operand is a zero-register-flavored name: this class cannot
/// address the stack pointer.
fn add_sub_shifted(w: u32) -> Option<String> {
    let sf = bit(w, 31);
    let kind = bits(w, 23, 22);
    let imm6 = bits(w, 15, 10);
    // ROR is unallocated here, and a 32-bit shift amount must fit in 5 bits.
    if kind == 0b11 || (!sf && imm6 >= 32) {
        return None;
    }
    let sub = bit(w, 30);
    let set_flags = bit(w, 29);
    let rd = bits(w, 4, 0) as u8;
    let rn = gp(sf, bits(w, 9, 5) as u8);
    let rm = gp(sf, bits(w, 20, 16) as u8);
    let shift = shift_operand(kind, imm6);
    if set_flags && rd == 31 {
        let mn = if sub { "cmp" } else { "cmn" };
        return Some(format!("{mn} {rn}, {rm}{shift}"));
    }
    let mn = match (sub, set_flags) {
        (false, false) => "add",
        (false, true) => "adds",
        (true, false) => "sub",
        (true, true) => "subs",
    };
    Some(format!("{mn} {}, {rn}, {rm}{shift}", gp(sf, rd)))
}

/// `CSEL`/`CSINC`/`CSINV`/`CSNEG`, with the `cset`/`csetm`/`cinc`/`cinv`/
/// `cneg` aliases. Each alias names the *false*-arm condition, so it prints
/// the inverse of the encoded one, and applies only when that condition is
/// not `al`/`nv`.
fn cond_select(w: u32) -> Option<String> {
    let op2 = bits(w, 11, 10);
    if op2 > 1 {
        // op2 = 1x is unallocated for conditional select.
        return None;
    }
    let sf = bit(w, 31);
    let rd = gp(sf, bits(w, 4, 0) as u8);
    let rn_n = bits(w, 9, 5) as u8;
    let rm_n = bits(w, 20, 16) as u8;
    let rn = gp(sf, rn_n);
    let rm = gp(sf, rm_n);
    let cond = Cond::from(bits(w, 15, 12) as u8);
    let c = cond.as_str();
    let inv = cond.invert().as_str();
    let aliasable = !cond.is_al_nv();
    let out = match (bit(w, 30), op2) {
        (false, 0) => format!("csel {rd}, {rn}, {rm}, {c}"),
        (false, _) => {
            if aliasable && rn_n == 31 && rm_n == 31 {
                format!("cset {rd}, {inv}")
            } else if aliasable && rn_n == rm_n {
                format!("cinc {rd}, {rn}, {inv}")
            } else {
                format!("csinc {rd}, {rn}, {rm}, {c}")
            }
        }
        (true, 0) => {
            if aliasable && rn_n == 31 && rm_n == 31 {
                format!("csetm {rd}, {inv}")
            } else if aliasable && rn_n == rm_n {
                format!("cinv {rd}, {rn}, {inv}")
            } else {
                format!("csinv {rd}, {rn}, {rm}, {c}")
            }
        }
        (true, _) => {
            if aliasable && rn_n == rm_n {
                format!("cneg {rd}, {rn}, {inv}")
            } else {
                format!("csneg {rd}, {rn}, {rm}, {c}")
            }
        }
    };
    Some(out)
}

// ---------------------------------------------------------------------
// Loads and stores
// ---------------------------------------------------------------------

/// Loads and stores (`op0 == x1x0`).
fn load_store(w: u32, va: u64) -> Option<String> {
    if bit(w, 26) {
        // SIMD & FP loads and stores.
        return None;
    }
    // LDR (literal): opc(2) 011 V 00 imm19 Rt.
    if w & 0x3F00_0000 == 0x1800_0000 {
        let sf = match bits(w, 31, 30) {
            0b00 => false,
            0b01 => true,
            // LDRSW / PRFM literal.
            _ => return None,
        };
        let rt = gp(sf, bits(w, 4, 0) as u8);
        let target = branch_target(va, bits(w, 23, 5), 19);
        return Some(format!("ldr {rt}, {target:#x}"));
    }
    // Load/store pair: opc(2) 101 V mode(3) L imm7 Rt2 Rn Rt.
    if bits(w, 29, 27) == 0b101 {
        return load_store_pair(w);
    }
    // Load/store register (immediate): size(2) 111 V ...
    if bits(w, 29, 27) == 0b111 {
        return load_store_imm(w);
    }
    // Exclusives, memory tags, and the rest of the space.
    None
}

/// `LDP`/`STP` in the signed-offset and pre/post-index forms. `LDPSW` and
/// the no-allocate `LDNP`/`STNP` pair are not covered.
fn load_store_pair(w: u32) -> Option<String> {
    let sf = match bits(w, 31, 30) {
        0b00 => false,
        0b10 => true,
        _ => return None,
    };
    let (index, scale) = match (bits(w, 25, 23), if sf { 3 } else { 2 }) {
        (0b001, s) => (Index::Post, s),
        (0b010, s) => (Index::None, s),
        (0b011, s) => (Index::Pre, s),
        _ => return None,
    };
    let off = sext(bits(w, 21, 15), 7) << scale;
    let mn = if bit(w, 22) { "ldp" } else { "stp" };
    Some(format!(
        "{mn} {}, {}, {}",
        gp(sf, bits(w, 4, 0) as u8),
        gp(sf, bits(w, 14, 10) as u8),
        mem(bits(w, 9, 5) as u8, off, index)
    ))
}

/// `LDR`/`STR` (and the `B`/`H` widths) and the sign-extending
/// `LDRS{B,H,W}` with an immediate: unsigned offset, unscaled
/// (`LDUR`/`STUR`/`LDURS*`), pre/post-index and register-offset. `PRFM` and
/// the unprivileged (`LDTR`/`STTR`) forms are not covered.
fn load_store_imm(w: u32) -> Option<String> {
    let size = bits(w, 31, 30);
    let opc = bits(w, 23, 22);
    let rn = bits(w, 9, 5) as u8;
    let rt_n = bits(w, 4, 0) as u8;
    // Resolve the stem, width suffix and transferred-register width from
    // opc, gated by the access size. `ldrs*` names its width by the access
    // size and its register by opc; plain load/store take the register
    // width from the access size. Anything else (PRFM, unallocated) is None.
    let (base, width, rt) = match opc {
        0b00 | 0b01 => {
            let base = if opc == 0b01 { "ldr" } else { "str" };
            let (width, rt) = match size {
                0 => ("b", gp(false, rt_n)),
                1 => ("h", gp(false, rt_n)),
                2 => ("", gp(false, rt_n)),
                _ => ("", gp(true, rt_n)),
            };
            (base, width, rt)
        }
        0b10 if size != 3 => {
            let width = match size {
                0 => "b",
                1 => "h",
                _ => "w",
            };
            ("ldrs", width, gp(true, rt_n))
        }
        0b11 if size < 2 => {
            let width = if size == 0 { "b" } else { "h" };
            ("ldrs", width, gp(false, rt_n))
        }
        _ => return None,
    };

    // Unsigned offset: size 111 V 01 opc imm12 Rn Rt, scaled by the
    // access width.
    if bits(w, 25, 24) == 0b01 {
        let off = (bits(w, 21, 10) as i64) << size;
        return Some(format!("{base}{width} {rt}, {}", mem(rn, off, Index::None)));
    }
    // Unscaled/indexed: size 111 V 00 opc 0 imm9 idx(2) Rn Rt.
    if bits(w, 25, 24) == 0b00 && !bit(w, 21) {
        let off = sext(bits(w, 20, 12), 9);
        // idx 00 is the unscaled form: `ldur`/`stur`/`ldurs*`.
        let (mn, index) = match bits(w, 11, 10) {
            0b00 => {
                let unscaled = match base {
                    "ldr" => "ldur",
                    "str" => "stur",
                    _ => "ldurs",
                };
                (unscaled, Index::None)
            }
            0b01 => (base, Index::Post),
            0b11 => (base, Index::Pre),
            // The unprivileged LDTR/STTR forms.
            _ => return None,
        };
        return Some(format!("{mn}{width} {rt}, {}", mem(rn, off, index)));
    }
    // Register offset: size 111 V 00 opc 1 Rm option(3) S 10 Rn Rt. Only
    // option<1> = 1 (UXTW/LSL/SXTW/SXTX) is allocated for this class.
    if bits(w, 25, 24) == 0b00 && bit(w, 21) && bits(w, 11, 10) == 0b10 {
        if !bit(w, 14) {
            return None;
        }
        return Some(format!("{base}{width} {rt}, {}", reg_off(w, rn, size)));
    }
    // Atomics, PAC forms, and the unscaled/unprivileged remainder.
    None
}

/// Format a register-offset memory operand from the raw word. The
/// `lsl`/`uxtx` default with no scaling prints as a bare `[base, Xm]`; every
/// other extend, and any scaled form, is spelled out with its amount (the
/// access `size`).
fn reg_off(w: u32, rn: u8, size: u32) -> String {
    let base = gp_sp(true, rn);
    let option = bits(w, 15, 13);
    // The option's low bit selects the index width: set is Xm, clear is Wm.
    let rm = gp(option & 1 == 1, bits(w, 20, 16) as u8);
    let scaled = bit(w, 12);
    let ext = match option {
        0b010 => "uxtw",
        0b011 => "lsl",
        0b110 => "sxtw",
        _ => "sxtx",
    };
    if option == 0b011 && !scaled {
        format!("[{base}, {rm}]")
    } else if scaled {
        format!("[{base}, {rm}, {ext} #{size}]")
    } else {
        format!("[{base}, {rm}, {ext}]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Aarch64Decoder, Decoder, Flow};

    const VA: u64 = 0x1000;

    /// Render one hand-encoded word (built from the Arm ARM bit layouts)
    /// at [`VA`].
    fn t(word: u32) -> Option<String> {
        A64Text.format(&word.to_le_bytes(), VA)
    }

    /// Assert that `word` renders exactly as `text` at [`VA`].
    fn eq(word: u32, text: &str) {
        assert_eq!(t(word).as_deref(), Some(text), "word {word:#010x}");
    }

    #[test]
    fn immediate_branches_render_absolute_targets() {
        eq(0x1400_0004, "b 0x1010"); // B  +0x10 (imm26 = 4)
        eq(0x17FF_FFFC, "b 0xff0"); // B  -0x10
        eq(0x9400_0004, "bl 0x1010"); // BL +0x10
        eq(0x97FF_FFFC, "bl 0xff0"); // BL -0x10
    }

    #[test]
    fn conditional_branch_covers_every_condition() {
        // B.<cond> +0x20 for cond = 0..15: 0x54000100 | cond.
        let names = [
            "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le",
            "al", "nv",
        ];
        for (cond, name) in names.iter().enumerate() {
            eq(0x5400_0100 | cond as u32, &format!("b.{name} 0x1020"));
        }
        eq(0x54FF_FFE1, "b.ne 0xffc"); // B.NE -4
        // Bit 4 set is BC.cond, outside the covered subset.
        assert_eq!(t(0x5400_0110), None);
    }

    #[test]
    fn compare_and_test_branches() {
        eq(0x3400_0040, "cbz w0, 0x1008"); // CBZ  w0, +8
        eq(0xB5FF_FFC5, "cbnz x5, 0xff8"); // CBNZ x5, -8
        eq(0xB608_0083, "tbz x3, #33, 0x1010"); // TBZ  x3, #33, +0x10
        eq(0x372F_FFC2, "tbnz w2, #5, 0xff8"); // TBNZ w2, #5, -8
    }

    #[test]
    fn register_branches() {
        eq(0xD61F_0200, "br x16");
        eq(0xD63F_0100, "blr x8");
        eq(0xD65F_03C0, "ret"); // RET with the default X30
        eq(0xD65F_0020, "ret x1");
        // ERET and RETAA are outside the covered subset.
        assert_eq!(t(0xD69F_03E0), None);
        assert_eq!(t(0xD65F_0BFF), None);
    }

    #[test]
    fn exceptions_and_hints() {
        eq(0xD400_0001, "svc #0x0");
        eq(0xD400_1001, "svc #0x80");
        eq(0xD400_0022, "hvc #0x1");
        eq(0xD400_0043, "smc #0x2");
        eq(0xD43E_0000, "brk #0xf000");
        eq(0xD440_0000, "hlt #0x0");
        eq(0xD503_201F, "nop");
        eq(0xD503_203F, "yield");
        eq(0xD503_205F, "wfe");
        eq(0xD503_207F, "wfi");
        eq(0xD503_209F, "sev");
        eq(0xD503_20BF, "sevl");
        // An unnamed hint (CRm:op2 = 7, XPACLRI) and an unallocated
        // exception opc/LL pair both fall back to the raw word.
        assert_eq!(t(0xD503_20FF), None);
        assert_eq!(t(0xD400_0000), None);
    }

    #[test]
    fn pc_relative_addressing() {
        eq(0x1000_0080, "adr x0, 0x1010"); // ADR x0, +0x10
        eq(0x10FF_FFE1, "adr x1, 0xffc"); // ADR x1, -4
        // ADRP drops the pc's low 12 bits before adding whole pages.
        assert_eq!(
            A64Text
                .format(&0xB000_0001u32.to_le_bytes(), 0x1234)
                .as_deref(),
            Some("adrp x1, 0x2000")
        );
        assert_eq!(
            A64Text
                .format(&0xF0FF_FFE2u32.to_le_bytes(), 0x1234)
                .as_deref(),
            Some("adrp x2, 0x0")
        );
    }

    #[test]
    fn move_wide_and_its_aliases() {
        eq(0xD282_4680, "mov x0, #0x1234"); // MOVZ x0, #0x1234
        eq(0xD2C0_0021, "movz x1, #0x1, lsl #32"); // shifted MOVZ keeps its name
        eq(0xF2B7_DDE0, "movk x0, #0xbeef, lsl #16");
        eq(0x1280_0000, "movn w0, #0x0");
        eq(0x9280_0000, "movn x0, #0x0");
        // A 32-bit move-wide with hw = 2 is unallocated.
        assert_eq!(t(0x52C0_0000), None);
    }

    #[test]
    fn add_sub_immediate_and_aliases() {
        eq(0x9100_4020, "add x0, x1, #0x10");
        // The shift stays an operand rather than being folded in.
        eq(0x9140_0462, "add x2, x3, #0x1, lsl #12");
        eq(0x7100_1420, "subs w0, w1, #0x5");
        eq(0xF100_045F, "cmp x2, #0x1"); // SUBS xzr, x2, #1
        eq(0xB100_041F, "cmn x0, #0x1"); // ADDS xzr, x0, #1
        eq(0x9100_03FD, "mov x29, sp"); // ADD x29, sp, #0
        eq(0xD100_4020, "sub x0, x1, #0x10");
        eq(0xB100_4020, "adds x0, x1, #0x10");
    }

    #[test]
    fn add_sub_shifted_register() {
        eq(0x8B02_0020, "add x0, x1, x2");
        eq(0x4B05_0883, "sub w3, w4, w5, lsl #2");
        eq(0xAB02_0020, "adds x0, x1, x2");
        eq(0xEB01_001F, "cmp x0, x1"); // SUBS xzr, x0, x1
        eq(0x2B01_001F, "cmn w0, w1"); // ADDS wzr, w0, w1
        // ROR is unallocated for add/sub, and a 32-bit form cannot shift
        // by 32 or more.
        assert_eq!(t(0x8BC2_0020), None);
        assert_eq!(t(0x0B02_8020), None);
        // Bit 21 set is the extended-register form.
        assert_eq!(t(0x8B22_0020), None);
    }

    #[test]
    fn logical_shifted_register() {
        eq(0x8A02_0020, "and x0, x1, x2");
        eq(0xAA01_03E0, "mov x0, x1"); // ORR x0, xzr, x1
        eq(0xAA02_0020, "orr x0, x1, x2");
        eq(0x4A42_0C20, "eor w0, w1, w2, lsr #3");
        eq(0x8A81_0C20, "and x0, x1, x1, asr #3");
        // ROR is allocated for the logical forms.
        eq(0x8AC2_0420, "and x0, x1, x2, ror #1");
        // BIC (N = 1) and ANDS (opc = 11) are not covered.
        assert_eq!(t(0x8A22_0020), None);
        assert_eq!(t(0xEA02_0020), None);
    }

    #[test]
    fn shift_immediate_aliases() {
        eq(0xD37C_EC20, "lsl x0, x1, #4"); // UBFM x0, x1, #60, #59
        eq(0x5303_7C20, "lsr w0, w1, #3"); // UBFM w0, w1, #3,  #31
        eq(0x9347_FC62, "asr x2, x3, #7"); // SBFM x2, x3, #7,  #63
        eq(0x1303_7C20, "asr w0, w1, #3"); // SBFM w0, w1, #3,  #31
        // A bitfield extract (UBFX x0, x1, #4, #8) is not a shift alias.
        assert_eq!(t(0xD344_2C20), None);
        // N must track sf: a 64-bit opcode with N = 0 is unallocated.
        assert_eq!(t(0xD30C_EC20), None);
    }

    #[test]
    fn multiply_and_divide() {
        eq(0x9B02_7C20, "mul x0, x1, x2"); // MADD x0, x1, x2, xzr
        eq(0x1B02_7C20, "mul w0, w1, w2");
        eq(0x1AC2_0820, "udiv w0, w1, w2");
        eq(0x9AC2_0C20, "sdiv x0, x1, x2");
        // MADD with a real addend, and MSUB, are not covered.
        assert_eq!(t(0x9B02_0C20), None);
        assert_eq!(t(0x9B02_FC20), None);
    }

    #[test]
    fn load_store_unsigned_offset() {
        eq(0xF940_0420, "ldr x0, [x1, #0x8]"); // imm12 = 1, scaled by 8
        eq(0xB900_03E2, "str w2, [sp]");
        eq(0x3940_0483, "ldrb w3, [x4, #0x1]");
        eq(0x7940_0483, "ldrh w3, [x4, #0x2]");
        eq(0x3900_0483, "strb w3, [x4, #0x1]");
        eq(0x7900_0483, "strh w3, [x4, #0x2]");
        // PRFM shares the class but is not covered.
        assert_eq!(t(0xF980_0483), None);
    }

    #[test]
    fn sign_extending_loads() {
        eq(0x39C0_BFE8, "ldrsb w8, [sp, #0x2f]"); // opc = 11 → W register
        eq(0x3980_0020, "ldrsb x0, [x1]"); // opc = 10 → X register
        eq(0x79C0_0420, "ldrsh w0, [x1, #0x2]");
        eq(0xB980_0420, "ldrsw x0, [x1, #0x4]"); // always the X register
        eq(0x38C0_1420, "ldrsb w0, [x1], #0x1"); // post-index
        eq(0x3880_43E0, "ldursb x0, [sp, #0x4]"); // unscaled LDURSB
        // PRFM (size = 11, opc = 10) and the unallocated size = 10/opc = 11.
        assert_eq!(t(0xF980_0000), None);
        assert_eq!(t(0xB9C0_0000), None);
    }

    #[test]
    fn register_offset_loads_and_stores() {
        eq(0x3869_690A, "ldrb w10, [x8, x9]"); // LSL, unscaled → bare index
        eq(0xF822_6820, "str x0, [x1, x2]");
        eq(0xF862_7820, "ldr x0, [x1, x2, lsl #3]"); // scaled LSL prints #size
        eq(0xF862_5820, "ldr x0, [x1, w2, uxtw #3]"); // UXTW → 32-bit index
        eq(0xB8A2_C820, "ldrsw x0, [x1, w2, sxtw]"); // sign-extending form
        // option<1> = 0 is unallocated for the register-offset class.
        assert_eq!(t(0x3869_290A), None);
    }

    #[test]
    fn conditional_select_and_aliases() {
        eq(0x9A88_B2B4, "csel x20, x21, x8, lt");
        eq(0x1A82_0420, "csinc w0, w1, w2, eq");
        eq(0xDA82_4020, "csinv x0, x1, x2, mi");
        eq(0xDA82_C420, "csneg x0, x1, x2, gt");
        // The condition-inverting aliases name the false-arm condition.
        eq(0x9A9F_07E0, "cset x0, ne"); // CSINC x0, xzr, xzr, eq
        eq(0x9A81_0420, "cinc x0, x1, ne"); // CSINC x0, x1, x1, eq
        eq(0xDA9F_03E0, "csetm x0, ne"); // CSINV x0, xzr, xzr, eq
        eq(0xDA81_0020, "cinv x0, x1, ne"); // CSINV x0, x1, x1, eq
        eq(0xDA81_C420, "cneg x0, x1, le"); // CSNEG x0, x1, x1, gt
        // op2 = 1x is unallocated.
        assert_eq!(t(0x9A88_BAB4), None);
    }

    #[test]
    fn load_store_unscaled_and_indexed() {
        eq(0xF85F_8020, "ldur x0, [x1, #-0x8]");
        eq(0xB81F_C3A0, "stur w0, [x29, #-0x4]");
        eq(0x381F_C3A0, "sturb w0, [x29, #-0x4]");
        eq(0xF85F_8C20, "ldr x0, [x1, #-0x8]!");
        eq(0xF801_07E0, "str x0, [sp], #0x10");
        // idx = 10 is the unprivileged LDTR/STTR form.
        assert_eq!(t(0xF85F_8820), None);
    }

    #[test]
    fn load_store_pairs() {
        eq(0xA9BF_7BFD, "stp x29, x30, [sp, #-0x10]!"); // the classic prologue
        eq(0xA8C1_7BFD, "ldp x29, x30, [sp], #0x10"); // and its epilogue
        eq(0x2941_0440, "ldp w0, w1, [x2, #0x8]"); // 32-bit scales by 4
        eq(0xA940_0440, "ldp x0, x1, [x2]");
        // LDPSW and the no-allocate pair forms are not covered.
        assert_eq!(t(0x6941_0440), None);
        assert_eq!(t(0xA841_0440), None);
    }

    #[test]
    fn literal_loads() {
        eq(0x5800_0040, "ldr x0, 0x1008"); // LDR x0, +8
        eq(0x18FF_FFE5, "ldr w5, 0xffc"); // LDR w5, -4
        // LDRSW and PRFM literal are not covered.
        assert_eq!(t(0x9800_0040), None);
        assert_eq!(t(0xD800_0040), None);
    }

    /// The classic A64 rendering bug: register 31 is `sp` in some operand
    /// positions and `xzr` in others, per encoding class.
    #[test]
    fn stack_pointer_versus_zero_register() {
        // Add/sub immediate: both Rd and Rn are SP-flavored...
        eq(0x9100_43FF, "add sp, sp, #0x10");
        eq(0x9100_03E0, "mov x0, sp");
        eq(0x9100_001F, "mov sp, x0");
        // ... but the flag-setting forms write the zero register, so Rd 31
        // is the CMP/CMN alias while Rn 31 is still the stack pointer.
        eq(0xB100_43E0, "adds x0, sp, #0x10");
        eq(0xF100_43FF, "cmp sp, #0x10");
        // Shifted-register and logical forms have no SP operand at all.
        eq(0x8B01_03E0, "add x0, xzr, x1");
        eq(0x8B1F_0020, "add x0, x1, xzr");
        eq(0x8A1F_03E0, "and x0, xzr, xzr");
        eq(0x9B1F_7FE0, "mul x0, xzr, xzr");
        // Loads and stores: the base is SP, the transferred register is ZR.
        eq(0xF940_03E0, "ldr x0, [sp]");
        eq(0xF900_07FF, "str xzr, [sp, #0x8]");
        eq(0xA900_7FFF, "stp xzr, xzr, [sp]");
        // Move-wide, literal loads and branches all name the zero register.
        eq(0xD280_001F, "mov xzr, #0x0");
        eq(0x5800_005F, "ldr xzr, 0x1008");
        eq(0x3400_005F, "cbz wzr, 0x1008");
        eq(0xD61F_03E0, "br xzr");
    }

    /// Pc-relative arithmetic must sign-extend correctly and wrap at the
    /// edges of the address space instead of panicking.
    #[test]
    fn targets_sign_extend_and_wrap() {
        let f = |w: u32, va: u64| A64Text.format(&w.to_le_bytes(), va);
        // B -0x10 at VA 0 wraps below zero.
        assert_eq!(f(0x17FF_FFFC, 0).as_deref(), Some("b 0xfffffffffffffff0"));
        // B +0x10 at the top of the space wraps above it.
        assert_eq!(f(0x1400_0004, u64::MAX - 3).as_deref(), Some("b 0xc"));
        // The largest reachable branch displacements, both ways.
        assert_eq!(
            f(0x1400_0000 | 0x01FF_FFFF, 0).as_deref(),
            Some("b 0x7fffffc")
        );
        assert_eq!(
            f(0x1400_0000 | 0x0200_0000, 0).as_deref(),
            Some("b 0xfffffffff8000000")
        );
        // ADRP one page below zero, and a literal load below zero.
        assert_eq!(
            f(0xF0FF_FFE2, 0).as_deref(),
            Some("adrp x2, 0xfffffffffffff000")
        );
        assert_eq!(
            f(0x18FF_FFE5, 0).as_deref(),
            Some("ldr w5, 0xfffffffffffffffc")
        );
        // Conditional and test branches wrap the same way.
        assert_eq!(
            f(0x54FF_FFE1, 0).as_deref(),
            Some("b.ne 0xfffffffffffffffc")
        );
        assert_eq!(
            f(0x372F_FFC2, 0).as_deref(),
            Some("tbnz w2, #5, 0xfffffffffffffff8")
        );
    }

    #[test]
    fn truncated_input_renders_nothing() {
        for len in 0..4 {
            for fill in [0x00u8, 0xFF, 0x94] {
                let bytes = vec![fill; len];
                assert_eq!(A64Text.format(&bytes, VA), None);
                assert_eq!(A64Text.format(&bytes, u64::MAX), None);
            }
        }
    }

    #[test]
    fn extra_trailing_bytes_are_ignored() {
        let mut buf = 0x1400_0004u32.to_le_bytes().to_vec();
        buf.extend_from_slice(&[0xAA; 5]);
        assert_eq!(A64Text.format(&buf, VA).as_deref(), Some("b 0x1010"));
    }

    #[test]
    fn unmodeled_encodings_render_nothing() {
        // UDF #0, a SIMD word, a logical immediate, an extract, a system
        // register write, and a load-exclusive.
        for w in [
            0x0000_0000u32,
            0x4EA1_1C21,
            0x9200_0400,
            0x93C1_0420,
            0xD51B_4200,
            0xC85F_7C20,
        ] {
            assert_eq!(t(w), None, "word {w:#010x}");
        }
    }

    /// The mnemonic we print must agree with how the flow decoder
    /// classifies the same word — a `bl` in the listing must be a
    /// [`Flow::Call`] in the graph, and the printed target must be the
    /// target analysis follows.
    fn assert_flow_agrees(w: u32, va: u64, text: &str) {
        let flow = Aarch64Decoder
            .decode_flow(&w.to_le_bytes(), va)
            .expect("a full word always decodes")
            .flow;
        let mnemonic = text.split(' ').next().unwrap_or_default();
        let expected = match mnemonic {
            "b" => Flow::Jump(0),
            "bl" => Flow::Call(0),
            "br" => Flow::IndirectJump,
            "blr" => Flow::IndirectCall,
            "ret" => Flow::Return,
            "svc" | "hvc" | "smc" | "brk" => Flow::Interrupt,
            "hlt" => Flow::Halt,
            "cbz" | "cbnz" | "tbz" | "tbnz" => Flow::CondJump(0),
            m if m.starts_with("b.") => Flow::CondJump(0),
            _ => Flow::Sequential,
        };
        // Compare the classification, then the target where there is one.
        let target = match (expected, flow) {
            (Flow::Jump(_), Flow::Jump(t))
            | (Flow::Call(_), Flow::Call(t))
            | (Flow::CondJump(_), Flow::CondJump(t)) => Some(t),
            (a, b) if a == b => None,
            _ => panic!("{w:#010x} rendered {text:?} but decoded as {flow:?}"),
        };
        if let Some(target) = target {
            let printed = text.rsplit(' ').next().unwrap_or_default();
            let printed = u64::from_str_radix(printed.trim_start_matches("0x"), 16)
                .unwrap_or_else(|_| panic!("{w:#010x}: {text:?} has no hex target"));
            assert_eq!(printed, target, "{w:#010x}: {text:?} vs {flow:?}");
        }
    }

    /// Rendering must be total (never panic) and never disagree with the
    /// flow decoder, over a broad deterministic sample: every value of the
    /// high 16 bits — where the opcode selectors live — against fixed low
    /// halves, plus a linear-congruential sweep of full words and VAs.
    #[test]
    fn rendering_is_total_and_flow_consistent() {
        let low = [0x0000u32, 0xFFFF, 0x03E0, 0x7C1F, 0x8421, 0x5555];
        let vas = [0u64, VA, 0x8000_0000_0000_0000, u64::MAX - 3];
        for hi in 0..=0xFFFFu32 {
            for &lo in &low {
                let w = (hi << 16) | lo;
                for &va in &vas {
                    if let Some(text) = A64Text.format(&w.to_le_bytes(), va) {
                        assert_flow_agrees(w, va, &text);
                    }
                }
            }
        }

        // LCG (the Knuth/MMIX constants); the high bits are the good ones,
        // so words and VAs are drawn from separate draws.
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        for _ in 0..20_000 {
            let w = (next() >> 32) as u32;
            let va = next();
            if let Some(text) = A64Text.format(&w.to_le_bytes(), va) {
                assert_flow_agrees(w, va, &text);
                // Rendered text is always a mnemonic plus operands.
                assert!(!text.is_empty() && !text.starts_with(' '));
            }
            // Unaligned and short slices must behave too.
            let bytes = w.to_le_bytes();
            for len in 0..4 {
                assert_eq!(A64Text.format(&bytes[..len], va), None);
            }
        }
    }
}
