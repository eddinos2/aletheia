//! x86-64 instruction-text rendering (Intel syntax).
//!
//! Clean-room implementation from the public Intel SDM Volume 2 encoding
//! rules (§2.1 legacy prefixes, §2.2.1 REX and the 64-bit ModRM/SIB
//! tables, and the one- and two-byte opcode maps in Appendix A).
//!
//! This is a *renderer*, not a second decoder for analysis to depend on:
//! [`crate::x86`] owns length and control-flow truth, and everything
//! downstream of [`crate::model::Decoder`] keeps using it. What a listing
//! needs on top of that is operand text, and operand text has a much
//! harsher failure mode than a length: an analyst reads `mov rax, [rbp-8]`
//! and believes it. So the rule here is *certainty or nothing* — every
//! encoding this module does not model exactly returns `None` and the
//! listing falls back to raw bytes. Coverage grows by adding forms that
//! have been checked against the SDM, never by extrapolating from a
//! neighbouring opcode.
//!
//! Deliberately not covered (all `None`, never guessed):
//! - the `67` address-size prefix, which would make addresses 32-bit and
//!   turn `[rip+disp]` into a truncated `[eip+disp]` — too easy to render
//!   a plausible-but-wrong absolute address, so the whole instruction is
//!   refused;
//! - `F2`/`F3` prefixes on anything but the CET `endbr` markers (the
//!   string instructions they modify are outside this module, and
//!   `F3 C3` is a `ret` idiom whose text would be misleading);
//! - a segment override on an instruction with no memory operand, and a
//!   `lock` prefix on an instruction whose destination is not memory:
//!   both are malformed, and inventing text for malformed bytes is the
//!   one thing a listing must not do;
//! - AVX and VEX/EVEX encodings, the `0F 38`/`0F 3A` maps, far
//!   transfers, string instructions, and the bit-test and
//!   `cmpxchg`/`xadd` groups. The legacy-encoded SSE/SSE2 subset that
//!   [`crate::x86`] models — scalar/packed moves, the arithmetic and
//!   bitwise operations, the ordered/unordered compares, the
//!   integer<->float converts, and the GPR<->XMM `movd`/`movq` — *is*
//!   rendered, mirroring that decoder's `(prefix, opcode)` decisions
//!   exactly; the MMX and prefixless forms it refuses are refused here
//!   too.
//!
//! Formatting conventions, fixed here so listings diff cleanly:
//! - every numeric literal is hex with an `0x` prefix, lower case, no
//!   zero padding;
//! - immediates that the CPU sign-extends are rendered *signed*
//!   (`add rsp, -0x8`), because that is the value the instruction
//!   actually applies;
//! - memory operands are compact IDA-style: `[rbp-0x8]`,
//!   `[rax+rcx*8+0x10]`, with the index scale always spelled out;
//! - `[rip+disp]` is resolved to the absolute address it names,
//!   `[0x402030]`, using wrapping arithmetic — that is what the analyst
//!   wants to see, and it is exactly what the CPU computes;
//! - relative branches render their absolute target, `jmp 0x401020`, also
//!   with wrapping arithmetic so an address near the top of the space
//!   cannot panic;
//! - a size keyword (`byte`/`word`/`dword`/`qword`) appears only when no
//!   register operand pins the width down: immediate-to-memory, the unary
//!   and shift groups, `push`/`pop` of memory, indirect `call`/`jmp`, and
//!   the multi-byte `nop`.
//!
//! Like every decoder in this crate the module is total: it never panics,
//! never reads past the input or past the architectural 15-byte
//! instruction limit, and never reports a length it did not consume.

use crate::asmtext::AsmFormatter;

/// Architectural maximum length of one instruction, in bytes. An encoding
/// that would need more is malformed (the CPU raises #GP), so decoding
/// stops rather than running off into the next instruction.
const MAX_LEN: usize = 15;

/// The x86-64 text formatter.
pub struct X86Text;

impl AsmFormatter for X86Text {
    fn format(&self, bytes: &[u8], va: u64) -> Option<String> {
        let insn = decode(bytes)?;
        Some(render(&insn, va))
    }
}

// ---------------------------------------------------------------------
// Operand model
// ---------------------------------------------------------------------

/// Operand width, which selects both the register name table and the size
/// keyword used for memory operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Size {
    B,
    W,
    D,
    Q,
}

impl Size {
    /// The Intel-syntax size keyword for a memory operand of this width.
    fn keyword(self) -> &'static str {
        match self {
            Size::B => "byte",
            Size::W => "word",
            Size::D => "dword",
            Size::Q => "qword",
        }
    }
}

/// One rendered-ready operand.
///
/// Address-dependent values ([`Op::Rel`] and the RIP-relative flavour of
/// [`Op::Mem`]) keep their raw displacement here and are resolved during
/// rendering, because both need the *total* instruction length — which is
/// not known until the trailing immediate has been consumed.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Op {
    /// A register, already named for its width and REX context.
    Reg(&'static str),
    Mem {
        /// Segment-override prefix, if any (`fs`, `gs`, ...).
        seg: Option<&'static str>,
        /// Size keyword, present only when no register operand fixes the
        /// operand width.
        size: Option<&'static str>,
        base: Option<&'static str>,
        index: Option<&'static str>,
        /// Index scale (1, 2, 4, 8); meaningless when `index` is `None`.
        scale: u8,
        disp: i64,
        /// `[rip + disp32]`: `disp` is relative to the next instruction.
        rip: bool,
    },
    /// An immediate the CPU sign-extends; rendered signed.
    Imm(i64),
    /// An immediate with no sign (`ret imm16`, `int imm8`, shift counts).
    UImm(u64),
    /// A relative branch displacement, rendered as its absolute target.
    Rel(i64),
}

/// One decoded instruction, ready to render.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Insn {
    mnemonic: &'static str,
    operands: Vec<Op>,
    /// Bytes consumed, prefixes included. Always `<= MAX_LEN`.
    len: usize,
    /// `F0` lock prefix, validated against a memory destination.
    lock: bool,
}

// ---------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------

/// Hex for a value the instruction treats as signed: `-0x8`, not the
/// two's-complement bit pattern. Uses [`i64::unsigned_abs`] so that
/// `i64::MIN` formats instead of overflowing.
fn hex_signed(v: i64) -> String {
    if v < 0 {
        format!("-0x{:x}", v.unsigned_abs())
    } else {
        format!("0x{v:x}")
    }
}

/// Render one operand into `out`.
///
/// `va` and `len` are the instruction's address and total length, needed
/// to resolve RIP-relative memory and relative branch targets. Both
/// computations wrap, matching the CPU's modulo-2^64 address arithmetic.
fn render_op(op: &Op, va: u64, len: usize, out: &mut String) {
    match *op {
        Op::Reg(name) => out.push_str(name),
        Op::Imm(v) => out.push_str(&hex_signed(v)),
        Op::UImm(v) => out.push_str(&format!("0x{v:x}")),
        Op::Rel(d) => {
            let target = va.wrapping_add(len as u64).wrapping_add(d as u64);
            out.push_str(&format!("0x{target:x}"));
        }
        Op::Mem {
            seg,
            size,
            base,
            index,
            scale,
            disp,
            rip,
        } => {
            if let Some(kw) = size {
                out.push_str(kw);
                out.push(' ');
            }
            if let Some(s) = seg {
                out.push_str(s);
                out.push(':');
            }
            out.push('[');
            if rip {
                // SDM Vol 2 §2.2.1.6: the displacement is relative to the
                // first byte of the *next* instruction.
                let abs = va.wrapping_add(len as u64).wrapping_add(disp as u64);
                out.push_str(&format!("0x{abs:x}"));
            } else {
                let mut have_reg = false;
                if let Some(b) = base {
                    out.push_str(b);
                    have_reg = true;
                }
                if let Some(i) = index {
                    if have_reg {
                        out.push('+');
                    }
                    out.push_str(i);
                    out.push_str(&format!("*{scale}"));
                    have_reg = true;
                }
                if !have_reg {
                    // No base and no index: the displacement is the whole
                    // address, so show it as one, unsigned.
                    out.push_str(&format!("0x{:x}", disp as u64));
                } else if disp < 0 {
                    out.push_str(&format!("-0x{:x}", disp.unsigned_abs()));
                } else if disp > 0 {
                    out.push_str(&format!("+0x{disp:x}"));
                }
            }
            out.push(']');
        }
    }
}

/// Render a decoded instruction as `"mnemonic op, op"`.
fn render(insn: &Insn, va: u64) -> String {
    let mut out = String::new();
    if insn.lock {
        out.push_str("lock ");
    }
    out.push_str(insn.mnemonic);
    for (i, op) in insn.operands.iter().enumerate() {
        out.push_str(if i == 0 { " " } else { ", " });
        render_op(op, va, insn.len, &mut out);
    }
    out
}

// ---------------------------------------------------------------------
// Byte cursor
// ---------------------------------------------------------------------

/// Bounds- and limit-checked cursor over the instruction bytes.
///
/// Every read returns `None` rather than panicking when it would pass the
/// end of the buffer or the 15-byte architectural limit, so truncated
/// input simply produces "no text".
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(count)?;
        if end > MAX_LEN || end > self.data.len() {
            return None;
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Some(slice)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    /// imm8 / rel8 / disp8, sign-extended.
    fn i8(&mut self) -> Option<i64> {
        Some(self.u8()? as i8 as i64)
    }

    /// imm16, sign-extended.
    fn i16(&mut self) -> Option<i64> {
        let b = self.take(2)?;
        Some(i16::from_le_bytes([b[0], b[1]]) as i64)
    }

    /// imm16 with no sign (`ret imm16`).
    fn u16(&mut self) -> Option<u64> {
        let b = self.take(2)?;
        Some(u16::from_le_bytes([b[0], b[1]]) as u64)
    }

    /// imm32 / rel32 / disp32, sign-extended.
    fn i32(&mut self) -> Option<i64> {
        let b = self.take(4)?;
        Some(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
    }

    /// imm64, the only 64-bit immediate (`REX.W B8+r`).
    fn i64(&mut self) -> Option<i64> {
        let b = self.take(8)?;
        Some(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
}

// ---------------------------------------------------------------------
// Prefixes
// ---------------------------------------------------------------------

/// Legacy and REX prefix state accumulated ahead of the opcode.
#[derive(Default, Clone, Copy)]
struct Prefixes {
    /// The REX byte (`40-4F`) when one immediately precedes the opcode.
    rex: Option<u8>,
    /// `66`, the operand-size override.
    opsize: bool,
    /// `67`, the address-size override (causes a blanket refusal).
    addrsize: bool,
    /// `F0`.
    lock: bool,
    /// `F2`/`F3`, kept as the raw byte.
    rep: Option<u8>,
    /// Segment override, already as its assembly name.
    seg: Option<&'static str>,
}

impl Prefixes {
    fn rex_present(&self) -> bool {
        self.rex.is_some()
    }

    fn rex_w(&self) -> bool {
        self.rex.unwrap_or(0) & 0x8 != 0
    }

    /// REX.R, pre-shifted into bit 3 of a register number.
    fn rex_r(&self) -> u8 {
        (self.rex.unwrap_or(0) & 0x4) << 1
    }

    /// REX.X, pre-shifted into bit 3.
    fn rex_x(&self) -> u8 {
        (self.rex.unwrap_or(0) & 0x2) << 2
    }

    /// REX.B, pre-shifted into bit 3.
    fn rex_b(&self) -> u8 {
        (self.rex.unwrap_or(0) & 0x1) << 3
    }

    /// Effective width for `v`-sized operands: REX.W beats `66`, which
    /// beats the 32-bit default.
    fn vsize(&self) -> Size {
        if self.rex_w() {
            Size::Q
        } else if self.opsize {
            Size::W
        } else {
            Size::D
        }
    }

    /// Effective width for `push`/`pop` and their memory forms, which
    /// default to 64 bits in long mode; `66` still shrinks them.
    fn stack_size(&self) -> Size {
        if self.opsize { Size::W } else { Size::Q }
    }
}

// ---------------------------------------------------------------------
// Registers
// ---------------------------------------------------------------------

/// Assembly name of GPR `num` (0-15) at `size`.
///
/// Without a REX prefix, byte-register numbers 4-7 name the legacy
/// high-byte aliases `ah`/`ch`/`dh`/`bh`; with any REX prefix present the
/// same encodings name `spl`/`bpl`/`sil`/`dil` instead (SDM Vol 2
/// Table 3-1). Getting this backwards silently mislabels operands, so it
/// is decided by `rex_present`, not by which REX bits are set.
fn reg_name(num: u8, size: Size, rex_present: bool) -> &'static str {
    const Q: [&str; 16] = [
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15",
    ];
    const D: [&str; 16] = [
        "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d",
        "r12d", "r13d", "r14d", "r15d",
    ];
    const W: [&str; 16] = [
        "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w",
        "r13w", "r14w", "r15w",
    ];
    const B: [&str; 16] = [
        "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b",
        "r13b", "r14b", "r15b",
    ];
    const H: [&str; 4] = ["ah", "ch", "dh", "bh"];
    let idx = (num & 0xF) as usize;
    if size == Size::B && !rex_present && (4..=7).contains(&idx) {
        return H[idx - 4];
    }
    match size {
        Size::Q => Q[idx],
        Size::D => D[idx],
        Size::W => W[idx],
        Size::B => B[idx],
    }
}

/// Assembly name of XMM register `num` (0-15). REX.R/REX.B have already
/// been folded into `num` by the ModRM decoder, exactly as for the GPRs.
fn xmm_name(num: u8) -> &'static str {
    const X: [&str; 16] = [
        "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10",
        "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
    ];
    X[(num & 0xF) as usize]
}

// ---------------------------------------------------------------------
// ModRM / SIB
// ---------------------------------------------------------------------

/// The r/m half of a ModRM byte, before an operand width is applied.
#[derive(Clone, Copy)]
enum Rm {
    /// mod=11: a register number 0-15, REX.B applied.
    Reg(u8),
    Mem {
        base: Option<u8>,
        index: Option<u8>,
        scale: u8,
        disp: i64,
        rip: bool,
    },
}

/// A decoded ModRM byte together with its SIB and displacement.
struct ModRm {
    /// The reg field with REX.R applied (0-15). Group opcodes use only
    /// `reg & 7`, the SDM's `/digit`.
    reg: u8,
    rm: Rm,
}

/// Decode ModRM + optional SIB + displacement (SDM Vol 2 Table 2-2, with
/// the 64-bit-mode substitutions of §2.2.1.6).
fn decode_modrm(cur: &mut Cursor, pfx: &Prefixes) -> Option<ModRm> {
    let byte = cur.u8()?;
    let mode = byte >> 6;
    let reg = ((byte >> 3) & 7) | pfx.rex_r();
    let rm_bits = byte & 7;

    if mode == 3 {
        return Some(ModRm {
            reg,
            rm: Rm::Reg(rm_bits | pfx.rex_b()),
        });
    }

    let mut base = None;
    let mut index = None;
    let mut scale = 1u8;
    let mut rip = false;
    let mut disp_size = match mode {
        0 => 0,
        1 => 1,
        _ => 4,
    };

    if rm_bits == 4 {
        let sib = cur.u8()?;
        let index_bits = (sib >> 3) & 7;
        let base_bits = sib & 7;
        // index=100 means "no index" — unless REX.X is set, which makes
        // it r12, a perfectly ordinary index register.
        if index_bits != 4 || pfx.rex_x() != 0 {
            index = Some(index_bits | pfx.rex_x());
            scale = 1 << (sib >> 6);
        }
        // base=101 with mod=00 is "disp32, no base", regardless of REX.B.
        if base_bits == 5 && mode == 0 {
            disp_size = 4;
        } else {
            base = Some(base_bits | pfx.rex_b());
        }
    } else if mode == 0 && rm_bits == 5 {
        rip = true;
        disp_size = 4;
    } else {
        base = Some(rm_bits | pfx.rex_b());
    }

    let disp = match disp_size {
        0 => 0,
        1 => cur.i8()?,
        _ => cur.i32()?,
    };

    Some(ModRm {
        reg,
        rm: Rm::Mem {
            base,
            index,
            scale,
            disp,
            rip,
        },
    })
}

/// The r/m half as an operand of width `size`.
///
/// `show_size` requests the size keyword; it is ignored for the register
/// form, where the register name already states the width.
fn rm_op(rm: &Rm, size: Size, pfx: &Prefixes, show_size: bool) -> Op {
    match *rm {
        Rm::Reg(num) => Op::Reg(reg_name(num, size, pfx.rex_present())),
        Rm::Mem {
            base,
            index,
            scale,
            disp,
            rip,
        } => Op::Mem {
            seg: pfx.seg,
            size: if show_size {
                Some(size.keyword())
            } else {
                None
            },
            // Addresses are 64-bit: the `67` prefix, which would make them
            // 32-bit, is refused outright in `decode`.
            base: base.map(|n| reg_name(n, Size::Q, true)),
            index: index.map(|n| reg_name(n, Size::Q, true)),
            scale,
            disp,
            rip,
        },
    }
}

/// The reg half as a register operand of width `size`.
fn reg_op(num: u8, size: Size, pfx: &Prefixes) -> Op {
    Op::Reg(reg_name(num, size, pfx.rex_present()))
}

/// Is this operand a memory reference?
fn is_mem(op: &Op) -> bool {
    matches!(op, Op::Mem { .. })
}

/// Read an "immz" immediate: 16 bits under a `66` prefix, otherwise 32,
/// sign-extended to the operand width (including 64-bit under REX.W).
fn immz(cur: &mut Cursor, size: Size) -> Option<i64> {
    match size {
        Size::W => cur.i16(),
        _ => cur.i32(),
    }
}

// ---------------------------------------------------------------------
// Mnemonic tables
// ---------------------------------------------------------------------

/// The eight arithmetic/logic operations, in opcode-map order. Indexes
/// both bits 3-5 of the `00-3D` block and the `/digit` of group 1.
const ARITH: [&str; 8] = ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"];

/// `Jcc` mnemonics, indexed by the opcode's low nibble.
const JCC: [&str; 16] = [
    "jo", "jno", "jb", "jae", "je", "jne", "jbe", "ja", "js", "jns", "jp", "jnp", "jl", "jge",
    "jle", "jg",
];

/// `SETcc` mnemonics, indexed by the opcode's low nibble.
const SETCC: [&str; 16] = [
    "seto", "setno", "setb", "setae", "sete", "setne", "setbe", "seta", "sets", "setns", "setp",
    "setnp", "setl", "setge", "setle", "setg",
];

/// `CMOVcc` mnemonics, indexed by the opcode's low nibble.
const CMOVCC: [&str; 16] = [
    "cmovo", "cmovno", "cmovb", "cmovae", "cmove", "cmovne", "cmovbe", "cmova", "cmovs", "cmovns",
    "cmovp", "cmovnp", "cmovl", "cmovge", "cmovle", "cmovg",
];

/// Group 2 (shift/rotate) by `/digit`. `/6` is an undocumented `shl`
/// alias on most parts; it is left `None` rather than guessed.
fn shift_mnemonic(digit: u8) -> Option<&'static str> {
    match digit & 7 {
        0 => Some("rol"),
        1 => Some("ror"),
        2 => Some("rcl"),
        3 => Some("rcr"),
        4 => Some("shl"),
        5 => Some("shr"),
        7 => Some("sar"),
        _ => None,
    }
}

// ---------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------

/// Decode one instruction from the start of `bytes`.
///
/// Returns `None` — never a partial or guessed result — for truncated
/// input, over-long encodings, and every encoding this module does not
/// model exactly. On success `len` is the number of bytes consumed;
/// `bytes` may extend past the instruction.
fn decode(bytes: &[u8]) -> Option<Insn> {
    let mut cur = Cursor::new(bytes);
    let mut pfx = Prefixes::default();

    // Prefix loop. REX only takes effect when it is the last byte before
    // the opcode, so any legacy prefix after one voids it (SDM Vol 2
    // §2.2.1).
    let opcode = loop {
        let b = cur.u8()?;
        match b {
            0xF0 => {
                pfx.lock = true;
                pfx.rex = None;
            }
            0xF2 | 0xF3 => {
                pfx.rep = Some(b);
                pfx.rex = None;
            }
            0x66 => {
                pfx.opsize = true;
                pfx.rex = None;
            }
            0x67 => {
                pfx.addrsize = true;
                pfx.rex = None;
            }
            0x26 => {
                pfx.seg = Some("es");
                pfx.rex = None;
            }
            0x2E => {
                pfx.seg = Some("cs");
                pfx.rex = None;
            }
            0x36 => {
                pfx.seg = Some("ss");
                pfx.rex = None;
            }
            0x3E => {
                pfx.seg = Some("ds");
                pfx.rex = None;
            }
            0x64 => {
                pfx.seg = Some("fs");
                pfx.rex = None;
            }
            0x65 => {
                pfx.seg = Some("gs");
                pfx.rex = None;
            }
            0x40..=0x4F => pfx.rex = Some(b),
            other => break other,
        }
    };

    // 32-bit addressing would change every memory operand and make the
    // RIP-relative absolute address wrong; refuse rather than mislead.
    if pfx.addrsize {
        return None;
    }

    let (mnemonic, operands, sse) = if opcode == 0x0F {
        dispatch_0f(&mut cur, &pfx)?
    } else {
        let (m, o) = dispatch(&mut cur, &pfx, opcode)?;
        (m, o, false)
    };

    // A repeat prefix changes the semantics of what follows; the only
    // forms modelled here are the CET markers and the SSE subset, whose
    // `F2`/`F3` is a mandatory variant selector (already consumed, never
    // rendered as `rep`/`repne`), not a string-instruction repeat.
    if pfx.rep.is_some() && !sse && !matches!(mnemonic, "endbr64" | "endbr32") {
        return None;
    }
    // A segment override with nothing to apply to, or `lock` without a
    // memory destination, is malformed input, not an instruction.
    if pfx.seg.is_some() && !operands.iter().any(is_mem) {
        return None;
    }
    if pfx.lock && !operands.first().is_some_and(is_mem) {
        return None;
    }

    Some(Insn {
        mnemonic,
        operands,
        len: cur.pos,
        lock: pfx.lock,
    })
}

/// Dispatch on a one-byte-map opcode.
fn dispatch(cur: &mut Cursor, pfx: &Prefixes, op: u8) -> Option<(&'static str, Vec<Op>)> {
    let vs = pfx.vsize();
    let ss = pfx.stack_size();

    match op {
        // 00-3D: add/or/adc/sbb/and/sub/xor/cmp, six encodings each.
        // Columns 6 and 7 of each row are invalid in 64-bit mode.
        0x00..=0x3D if op & 7 <= 5 => {
            let mnem = ARITH[(op >> 3) as usize & 7];
            let ops = match op & 7 {
                // Eb, Gb
                0 => {
                    let m = decode_modrm(cur, pfx)?;
                    vec![
                        rm_op(&m.rm, Size::B, pfx, false),
                        reg_op(m.reg, Size::B, pfx),
                    ]
                }
                // Ev, Gv
                1 => {
                    let m = decode_modrm(cur, pfx)?;
                    vec![rm_op(&m.rm, vs, pfx, false), reg_op(m.reg, vs, pfx)]
                }
                // Gb, Eb
                2 => {
                    let m = decode_modrm(cur, pfx)?;
                    vec![
                        reg_op(m.reg, Size::B, pfx),
                        rm_op(&m.rm, Size::B, pfx, false),
                    ]
                }
                // Gv, Ev
                3 => {
                    let m = decode_modrm(cur, pfx)?;
                    vec![reg_op(m.reg, vs, pfx), rm_op(&m.rm, vs, pfx, false)]
                }
                // AL, ib
                4 => vec![Op::Reg("al"), Op::Imm(cur.i8()?)],
                // rAX, iz
                _ => vec![
                    Op::Reg(reg_name(0, vs, pfx.rex_present())),
                    Op::Imm(immz(cur, vs)?),
                ],
            };
            Some((mnem, ops))
        }
        // push/pop r64 (r16 under `66`).
        0x50..=0x57 => Some((
            "push",
            vec![Op::Reg(reg_name((op & 7) | pfx.rex_b(), ss, true))],
        )),
        0x58..=0x5F => Some((
            "pop",
            vec![Op::Reg(reg_name((op & 7) | pfx.rex_b(), ss, true))],
        )),
        // movsxd Gv, Ed. The `66` form (16-bit destination) is rare and
        // ambiguous enough in practice that it is left unmodelled.
        0x63 if !pfx.opsize => {
            let m = decode_modrm(cur, pfx)?;
            let dst = if pfx.rex_w() { Size::Q } else { Size::D };
            Some((
                "movsxd",
                vec![
                    reg_op(m.reg, dst, pfx),
                    rm_op(&m.rm, Size::D, pfx, is_mem_rm(&m.rm)),
                ],
            ))
        }
        // push iz / ib, both sign-extended to 64 bits.
        0x68 => Some(("push", vec![Op::Imm(immz(cur, vs)?)])),
        0x6A => Some(("push", vec![Op::Imm(cur.i8()?)])),
        // imul Gv, Ev, iz/ib.
        0x69 | 0x6B => {
            let m = decode_modrm(cur, pfx)?;
            let imm = if op == 0x69 {
                immz(cur, vs)?
            } else {
                cur.i8()?
            };
            Some((
                "imul",
                vec![
                    reg_op(m.reg, vs, pfx),
                    rm_op(&m.rm, vs, pfx, false),
                    Op::Imm(imm),
                ],
            ))
        }
        // jcc rel8.
        0x70..=0x7F => Some((JCC[(op & 0xF) as usize], vec![Op::Rel(cur.i8()?)])),
        // Group 1: arithmetic with an immediate. `82` is the alias that
        // 64-bit mode removed.
        0x80 | 0x81 | 0x83 => {
            let m = decode_modrm(cur, pfx)?;
            let (size, imm) = match op {
                0x80 => (Size::B, cur.i8()?),
                0x81 => (vs, immz(cur, vs)?),
                _ => (vs, cur.i8()?),
            };
            Some((
                ARITH[(m.reg & 7) as usize],
                vec![rm_op(&m.rm, size, pfx, is_mem_rm(&m.rm)), Op::Imm(imm)],
            ))
        }
        // test/xchg/mov, Eb,Gb then Ev,Gv.
        0x84 | 0x86 | 0x88 => {
            let m = decode_modrm(cur, pfx)?;
            let mnem = match op {
                0x84 => "test",
                0x86 => "xchg",
                _ => "mov",
            };
            Some((
                mnem,
                vec![
                    rm_op(&m.rm, Size::B, pfx, false),
                    reg_op(m.reg, Size::B, pfx),
                ],
            ))
        }
        0x85 | 0x87 | 0x89 => {
            let m = decode_modrm(cur, pfx)?;
            let mnem = match op {
                0x85 => "test",
                0x87 => "xchg",
                _ => "mov",
            };
            Some((
                mnem,
                vec![rm_op(&m.rm, vs, pfx, false), reg_op(m.reg, vs, pfx)],
            ))
        }
        0x8A => {
            let m = decode_modrm(cur, pfx)?;
            Some((
                "mov",
                vec![
                    reg_op(m.reg, Size::B, pfx),
                    rm_op(&m.rm, Size::B, pfx, false),
                ],
            ))
        }
        0x8B => {
            let m = decode_modrm(cur, pfx)?;
            Some((
                "mov",
                vec![reg_op(m.reg, vs, pfx), rm_op(&m.rm, vs, pfx, false)],
            ))
        }
        // lea Gv, M. A register source (mod=11) is architecturally
        // invalid, so it gets no text.
        0x8D => {
            let m = decode_modrm(cur, pfx)?;
            if !is_mem_rm(&m.rm) {
                return None;
            }
            Some((
                "lea",
                vec![reg_op(m.reg, vs, pfx), rm_op(&m.rm, vs, pfx, false)],
            ))
        }
        // Group 1A: pop r/m (/0 only).
        0x8F => {
            let m = decode_modrm(cur, pfx)?;
            if m.reg & 7 != 0 {
                return None;
            }
            Some(("pop", vec![rm_op(&m.rm, ss, pfx, is_mem_rm(&m.rm))]))
        }
        // 90 with no REX.B and no `66` is the canonical one-byte nop;
        // otherwise the row is xchg rAX, r.
        0x90 if pfx.rex_b() == 0 && !pfx.opsize => Some(("nop", vec![])),
        0x90..=0x97 => Some((
            "xchg",
            vec![
                Op::Reg(reg_name(0, vs, pfx.rex_present())),
                Op::Reg(reg_name((op & 7) | pfx.rex_b(), vs, pfx.rex_present())),
            ],
        )),
        // Sign-extend accumulator: the mnemonic is the operand size.
        0x98 => Some((
            match vs {
                Size::Q => "cdqe",
                Size::W => "cbw",
                _ => "cwde",
            },
            vec![],
        )),
        0x99 => Some((
            match vs {
                Size::Q => "cqo",
                Size::W => "cwd",
                _ => "cdq",
            },
            vec![],
        )),
        // test AL, ib / rAX, iz.
        0xA8 => Some(("test", vec![Op::Reg("al"), Op::Imm(cur.i8()?)])),
        0xA9 => Some((
            "test",
            vec![
                Op::Reg(reg_name(0, vs, pfx.rex_present())),
                Op::Imm(immz(cur, vs)?),
            ],
        )),
        // mov r8, ib.
        0xB0..=0xB7 => Some((
            "mov",
            vec![
                Op::Reg(reg_name((op & 7) | pfx.rex_b(), Size::B, pfx.rex_present())),
                Op::Imm(cur.i8()?),
            ],
        )),
        // mov r, imm — the only imm64 form, under REX.W.
        0xB8..=0xBF => {
            let imm = match vs {
                Size::Q => cur.i64()?,
                Size::W => cur.i16()?,
                _ => cur.i32()?,
            };
            Some((
                "mov",
                vec![
                    Op::Reg(reg_name((op & 7) | pfx.rex_b(), vs, pfx.rex_present())),
                    Op::Imm(imm),
                ],
            ))
        }
        // Group 2: shift/rotate by an immediate count.
        0xC0 | 0xC1 => {
            let m = decode_modrm(cur, pfx)?;
            let mnem = shift_mnemonic(m.reg)?;
            let size = if op == 0xC0 { Size::B } else { vs };
            let count = cur.u8()?;
            Some((
                mnem,
                vec![
                    rm_op(&m.rm, size, pfx, is_mem_rm(&m.rm)),
                    Op::UImm(count as u64),
                ],
            ))
        }
        // ret imm16 / ret.
        0xC2 => Some(("ret", vec![Op::UImm(cur.u16()?)])),
        0xC3 => Some(("ret", vec![])),
        // Group 11: mov r/m, imm (/0 only).
        0xC6 | 0xC7 => {
            let m = decode_modrm(cur, pfx)?;
            if m.reg & 7 != 0 {
                return None;
            }
            let (size, imm) = if op == 0xC6 {
                (Size::B, cur.i8()?)
            } else {
                (vs, immz(cur, vs)?)
            };
            Some((
                "mov",
                vec![rm_op(&m.rm, size, pfx, is_mem_rm(&m.rm)), Op::Imm(imm)],
            ))
        }
        0xC9 => Some(("leave", vec![])),
        0xCC => Some(("int3", vec![])),
        0xCD => Some(("int", vec![Op::UImm(cur.u8()? as u64)])),
        // Group 2: shift/rotate by 1 and by cl.
        0xD0..=0xD3 => {
            let m = decode_modrm(cur, pfx)?;
            let mnem = shift_mnemonic(m.reg)?;
            let size = if op & 1 == 0 { Size::B } else { vs };
            let count = if op & 2 == 0 {
                Op::UImm(1)
            } else {
                Op::Reg("cl")
            };
            Some((mnem, vec![rm_op(&m.rm, size, pfx, is_mem_rm(&m.rm)), count]))
        }
        // call rel32 / jmp rel32 / jmp rel8.
        0xE8 => Some(("call", vec![Op::Rel(cur.i32()?)])),
        0xE9 => Some(("jmp", vec![Op::Rel(cur.i32()?)])),
        0xEB => Some(("jmp", vec![Op::Rel(cur.i8()?)])),
        0xF4 => Some(("hlt", vec![])),
        // Group 3: test/not/neg/mul/imul/div/idiv.
        0xF6 | 0xF7 => {
            let m = decode_modrm(cur, pfx)?;
            let size = if op == 0xF6 { Size::B } else { vs };
            let show = is_mem_rm(&m.rm);
            let rm = rm_op(&m.rm, size, pfx, show);
            match m.reg & 7 {
                // /0 and /1 are both `test r/m, imm`.
                0 | 1 => {
                    let imm = if op == 0xF6 {
                        cur.i8()?
                    } else {
                        immz(cur, vs)?
                    };
                    Some(("test", vec![rm, Op::Imm(imm)]))
                }
                2 => Some(("not", vec![rm])),
                3 => Some(("neg", vec![rm])),
                4 => Some(("mul", vec![rm])),
                5 => Some(("imul", vec![rm])),
                6 => Some(("div", vec![rm])),
                _ => Some(("idiv", vec![rm])),
            }
        }
        // Group 4: inc/dec r/m8.
        0xFE => {
            let m = decode_modrm(cur, pfx)?;
            let rm = rm_op(&m.rm, Size::B, pfx, is_mem_rm(&m.rm));
            match m.reg & 7 {
                0 => Some(("inc", vec![rm])),
                1 => Some(("dec", vec![rm])),
                _ => None,
            }
        }
        // Group 5: inc/dec/call/jmp/push r/m.
        0xFF => {
            let m = decode_modrm(cur, pfx)?;
            let show = is_mem_rm(&m.rm);
            match m.reg & 7 {
                0 => Some(("inc", vec![rm_op(&m.rm, vs, pfx, show)])),
                1 => Some(("dec", vec![rm_op(&m.rm, vs, pfx, show)])),
                // Near indirect call/jmp: always 64-bit operands in long
                // mode. A `66` prefix would name the 16-bit forms, which
                // 64-bit mode does not support.
                2 | 4 if !pfx.opsize => {
                    let mnem = if m.reg & 7 == 2 { "call" } else { "jmp" };
                    Some((mnem, vec![rm_op(&m.rm, Size::Q, pfx, show)]))
                }
                6 => Some(("push", vec![rm_op(&m.rm, ss, pfx, show)])),
                // /3 and /5 are far transfers; /7 is undefined.
                _ => None,
            }
        }
        _ => None,
    }
}

/// Dispatch on a two-byte-map (`0F xx`) opcode.
///
/// The trailing `bool` reports whether the instruction legitimately
/// consumed a mandatory `F2`/`F3` prefix (the SSE subset); [`decode`]
/// uses it to keep those prefixes from being rejected as a stray repeat.
fn dispatch_0f(cur: &mut Cursor, pfx: &Prefixes) -> Option<(&'static str, Vec<Op>, bool)> {
    let vs = pfx.vsize();
    let op = cur.u8()?;

    // The SSE/SSE2 subset (all `0F xx /r`, mandatory-prefix selected).
    // Intercepted before the general map so the mandatory prefix is
    // consumed as a variant selector rather than refused.
    if matches!(
        op,
        0x10 | 0x11
            | 0x28
            | 0x29
            | 0x2E
            | 0x2F
            | 0x54
            | 0x57
            | 0x58
            | 0x59
            | 0x5C
            | 0x5E
            | 0x2A
            | 0x2C
            | 0x2D
            | 0x6E
            | 0x7E
            | 0xD6
    ) {
        return decode_sse(cur, pfx, op).map(|(m, o)| (m, o, true));
    }

    let res = match op {
        0x05 => Some(("syscall", vec![])),
        0x0B => Some(("ud2", vec![])),
        // F3 0F 1E FA/FB: the CET branch-target markers. Every other
        // `0F 1E` form is reserved hint space and gets no text.
        0x1E if pfx.rep == Some(0xF3) => match cur.u8()? {
            0xFA => Some(("endbr64", vec![])),
            0xFB => Some(("endbr32", vec![])),
            _ => None,
        },
        // Multi-byte nop, 0F 1F /r: the operand is decoded and shown
        // because its length is what distinguishes the padding sizes.
        0x1F => {
            let m = decode_modrm(cur, pfx)?;
            Some(("nop", vec![rm_op(&m.rm, vs, pfx, is_mem_rm(&m.rm))]))
        }
        // cmovcc Gv, Ev.
        0x40..=0x4F => {
            let m = decode_modrm(cur, pfx)?;
            Some((
                CMOVCC[(op & 0xF) as usize],
                vec![reg_op(m.reg, vs, pfx), rm_op(&m.rm, vs, pfx, false)],
            ))
        }
        // jcc rel32.
        0x80..=0x8F => Some((JCC[(op & 0xF) as usize], vec![Op::Rel(cur.i32()?)])),
        // setcc r/m8; the reg field only carries the condition.
        0x90..=0x9F => {
            let m = decode_modrm(cur, pfx)?;
            Some((
                SETCC[(op & 0xF) as usize],
                vec![rm_op(&m.rm, Size::B, pfx, is_mem_rm(&m.rm))],
            ))
        }
        // imul Gv, Ev.
        0xAF => {
            let m = decode_modrm(cur, pfx)?;
            Some((
                "imul",
                vec![reg_op(m.reg, vs, pfx), rm_op(&m.rm, vs, pfx, false)],
            ))
        }
        // movzx/movsx Gv, Eb/Ew. The source is narrower than the
        // destination, so a memory source always needs its size spelled.
        0xB6 | 0xB7 | 0xBE | 0xBF => {
            let m = decode_modrm(cur, pfx)?;
            let mnem = if op & 0x08 == 0 { "movzx" } else { "movsx" };
            let src = if op & 1 == 0 { Size::B } else { Size::W };
            Some((
                mnem,
                vec![
                    reg_op(m.reg, vs, pfx),
                    rm_op(&m.rm, src, pfx, is_mem_rm(&m.rm)),
                ],
            ))
        }
        _ => None,
    };
    res.map(|(m, o)| (m, o, false))
}

/// Decode the SSE/SSE2 subset the listing renders, mirroring
/// [`crate::x86`]'s `decode_sse` opcode-by-opcode so the two decoders
/// agree on every `(prefix, opcode)` pair.
///
/// All members are `0F xx /r`; the mandatory `66`/`F3`/`F2` prefix
/// selects the packed/scalar, single/double variant and is consumed here
/// (never rendered as `rep`/`repne`). XMM operands name `xmm0`..`xmm15`;
/// a memory XMM operand takes no size keyword because the paired XMM
/// register — or the mnemonic itself — already fixes its width. The GPR
/// side of the converts and `movd`/`movq` is named for its width (REX.W
/// selects 64-bit), and a memory GPR operand does get a size keyword,
/// since nothing else pins it. A `(prefix, opcode)` pair outside the
/// subset — the MMX forms of `6E`/`7E`/`D6` without `66`, and the
/// converts without their `F3`/`F2` — is refused (`None`), never guessed.
fn decode_sse(cur: &mut Cursor, pfx: &Prefixes, op: u8) -> Option<(&'static str, Vec<Op>)> {
    let p66 = pfx.opsize;
    let f3 = pfx.rep == Some(0xF3);
    let f2 = pfx.rep == Some(0xF2);
    // GPR width for the converts and GPR<->XMM moves: REX.W selects 64-bit.
    let gw = if pfx.rex_w() { Size::Q } else { Size::D };
    let m = decode_modrm(cur, pfx)?;
    let xreg = Op::Reg(xmm_name(m.reg));
    // The r/m half as XMM-or-memory (mod=11 is an XMM register here); a
    // memory operand needs no size keyword.
    let xrm = match m.rm {
        Rm::Reg(n) => Op::Reg(xmm_name(n)),
        Rm::Mem { .. } => rm_op(&m.rm, Size::Q, pfx, false),
    };
    // The r/m half as a GPR of width `gw`, with a size keyword only when
    // it names memory.
    let grm = rm_op(&m.rm, gw, pfx, is_mem_rm(&m.rm));

    match op {
        // movups/movupd/movss/movsd: 10 = load (reg<-rm), 11 = store.
        0x10 | 0x11 => {
            let mnem = if f3 {
                "movss"
            } else if f2 {
                "movsd"
            } else if p66 {
                "movupd"
            } else {
                "movups"
            };
            if op == 0x10 {
                Some((mnem, vec![xreg, xrm]))
            } else {
                Some((mnem, vec![xrm, xreg]))
            }
        }
        // movaps/movapd: 28 = load, 29 = store.
        0x28 | 0x29 => {
            let mnem = if p66 { "movapd" } else { "movaps" };
            if op == 0x28 {
                Some((mnem, vec![xreg, xrm]))
            } else {
                Some((mnem, vec![xrm, xreg]))
            }
        }
        // ucomiss/ucomisd/comiss/comisd — the EFLAGS-writing members.
        0x2E | 0x2F => {
            let mnem = match (op, p66) {
                (0x2E, false) => "ucomiss",
                (0x2E, _) => "ucomisd",
                (0x2F, false) => "comiss",
                (_, _) => "comisd",
            };
            Some((mnem, vec![xreg, xrm]))
        }
        0x54 => Some((if p66 { "andpd" } else { "andps" }, vec![xreg, xrm])),
        0x57 => Some((if p66 { "xorpd" } else { "xorps" }, vec![xreg, xrm])),
        // add/mul/sub/div, packed/scalar single/double.
        0x58 | 0x59 | 0x5C | 0x5E => {
            let mnem = match (op, f3, f2, p66) {
                (0x58, true, _, _) => "addss",
                (0x58, _, true, _) => "addsd",
                (0x58, _, _, true) => "addpd",
                (0x58, ..) => "addps",
                (0x59, true, _, _) => "mulss",
                (0x59, _, true, _) => "mulsd",
                (0x59, _, _, true) => "mulpd",
                (0x59, ..) => "mulps",
                (0x5C, true, _, _) => "subss",
                (0x5C, _, true, _) => "subsd",
                (0x5C, _, _, true) => "subpd",
                (0x5C, ..) => "subps",
                (_, true, _, _) => "divss",
                (_, _, true, _) => "divsd",
                (_, _, _, true) => "divpd",
                (..) => "divps",
            };
            Some((mnem, vec![xreg, xrm]))
        }
        // cvtsi2ss/cvtsi2sd xmm, r/m(gpr) — requires F3/F2.
        0x2A => {
            if f3 {
                Some(("cvtsi2ss", vec![xreg, grm]))
            } else if f2 {
                Some(("cvtsi2sd", vec![xreg, grm]))
            } else {
                None
            }
        }
        // cvt(t)ss2si/sd2si r(gpr), xmm/mem — requires F3/F2.
        0x2C | 0x2D => {
            let mnem = match (op, f3, f2) {
                (0x2C, true, _) => "cvttss2si",
                (0x2C, _, true) => "cvttsd2si",
                (0x2D, true, _) => "cvtss2si",
                (0x2D, _, true) => "cvtsd2si",
                _ => return None,
            };
            Some((mnem, vec![reg_op(m.reg, gw, pfx), xrm]))
        }
        // 66 0F 6E: movd/movq xmm, r/m(gpr) (REX.W -> movq). No-66 is MMX.
        0x6E => {
            if !p66 {
                return None;
            }
            let mnem = if pfx.rex_w() { "movq" } else { "movd" };
            Some((mnem, vec![xreg, grm]))
        }
        // F3 0F 7E: movq xmm, xmm/m64. 66 0F 7E: movd/movq r/m(gpr), xmm.
        0x7E => {
            if f3 {
                Some(("movq", vec![xreg, xrm]))
            } else if p66 {
                let mnem = if pfx.rex_w() { "movq" } else { "movd" };
                Some((mnem, vec![grm, xreg]))
            } else {
                None
            }
        }
        // 66 0F D6: movq xmm/m64, xmm (store).
        0xD6 => {
            if !p66 {
                return None;
            }
            Some(("movq", vec![xrm, xreg]))
        }
        _ => None,
    }
}

/// Does this r/m half name memory (rather than a register)?
fn is_mem_rm(rm: &Rm) -> bool {
    matches!(rm, Rm::Mem { .. })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Decoder, X86Decoder};

    /// The VA every fixed-text expectation is rendered at.
    const VA: u64 = 0x401000;

    /// Hand-assembled encodings and their exact expected text.
    ///
    /// Each entry was written from the SDM opcode maps and checked field
    /// by field: the comment gives the encoding, so a disagreement is a
    /// bug in one of the two, not a matter of taste.
    const CORPUS: &[(&[u8], &str)] = &[
        // --- prologue / epilogue -------------------------------------
        // 55                     push rbp
        (&[0x55], "push rbp"),
        // 48 89 E5               REX.W 89 /r, modrm mod=11 reg=100 rm=101
        (&[0x48, 0x89, 0xE5], "mov rbp, rsp"),
        // 48 8B 45 F8            REX.W 8B /r, mod=01 reg=000 rm=101, disp8 -8
        (&[0x48, 0x8B, 0x45, 0xF8], "mov rax, [rbp-0x8]"),
        // 48 89 45 F8            REX.W 89 /r, mod=01 reg=000 rm=101, disp8 -8
        (&[0x48, 0x89, 0x45, 0xF8], "mov [rbp-0x8], rax"),
        // C9                     leave
        (&[0xC9], "leave"),
        // C3                     ret
        (&[0xC3], "ret"),
        // C2 08 00               ret imm16
        (&[0xC2, 0x08, 0x00], "ret 0x8"),
        // 41 54 / 41 5C          REX.B 50+r / 58+r
        (&[0x41, 0x54], "push r12"),
        (&[0x41, 0x5C], "pop r12"),
        // 66 55                  operand-size override shrinks push
        (&[0x66, 0x55], "push bp"),
        // 6A 01                  push imm8 (sign-extended)
        (&[0x6A, 0x01], "push 0x1"),
        // 6A FF                  push -1
        (&[0x6A, 0xFF], "push -0x1"),
        // 68 00 01 00 00         push imm32
        (&[0x68, 0x00, 0x01, 0x00, 0x00], "push 0x100"),
        // FF 75 F8               group 5 /6, mod=01 rm=101 disp8 -8
        (&[0xFF, 0x75, 0xF8], "push qword [rbp-0x8]"),
        // 8F 45 F8               group 1A /0
        (&[0x8F, 0x45, 0xF8], "pop qword [rbp-0x8]"),
        // --- traps and system ----------------------------------------
        (&[0xCC], "int3"),
        (&[0xCD, 0x80], "int 0x80"),
        (&[0x0F, 0x05], "syscall"),
        (&[0xF4], "hlt"),
        (&[0x0F, 0x0B], "ud2"),
        (&[0x90], "nop"),
        // F3 0F 1E FA            endbr64
        (&[0xF3, 0x0F, 0x1E, 0xFA], "endbr64"),
        // 0F 1F 44 00 00         nop with SIB: base=rax index=rax scale=1, disp8 0
        (&[0x0F, 0x1F, 0x44, 0x00, 0x00], "nop dword [rax+rax*1]"),
        // 66 0F 1F 84 00 ...     the nine-byte padding nop
        (
            &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
            "nop word [rax+rax*1]",
        ),
        // --- relative control flow (rendered at VA = 0x401000) -------
        // E8 rel32 = 0x00000FFB -> 0x401005 + 0xFFB
        (&[0xE8, 0xFB, 0x0F, 0x00, 0x00], "call 0x402000"),
        // E8 rel32 = 0 -> the next instruction
        (&[0xE8, 0x00, 0x00, 0x00, 0x00], "call 0x401005"),
        // E9 rel32 = -0x10 -> 0x401005 - 0x10
        (&[0xE9, 0xF0, 0xFF, 0xFF, 0xFF], "jmp 0x400ff5"),
        // EB FE                  the classic self-loop
        (&[0xEB, 0xFE], "jmp 0x401000"),
        // EB 20
        (&[0xEB, 0x20], "jmp 0x401022"),
        // 74 05                  je rel8
        (&[0x74, 0x05], "je 0x401007"),
        // 75 FB                  jne rel8, backwards
        (&[0x75, 0xFB], "jne 0x400ffd"),
        // 7F 00                  jg rel8 = 0
        (&[0x7F, 0x00], "jg 0x401002"),
        // 0F 84 rel32            je, six bytes long
        (&[0x0F, 0x84, 0x00, 0x01, 0x00, 0x00], "je 0x401106"),
        // 0F 8C rel32 negative   jl backwards
        (&[0x0F, 0x8C, 0x00, 0xFF, 0xFF, 0xFF], "jl 0x400f06"),
        // --- indirect control flow -----------------------------------
        // FF D0 / FF E0          group 5 /2 and /4, mod=11
        (&[0xFF, 0xD0], "call rax"),
        (&[0xFF, 0xE0], "jmp rax"),
        // 41 FF D3               REX.B -> r11
        (&[0x41, 0xFF, 0xD3], "call r11"),
        // FF 15 rel32            call qword [rip+2] -> 0x401008
        (
            &[0xFF, 0x15, 0x02, 0x00, 0x00, 0x00],
            "call qword [0x401008]",
        ),
        // FF 24 25 disp32        jmp qword [absolute]
        (
            &[0xFF, 0x24, 0x25, 0x00, 0x20, 0x40, 0x00],
            "jmp qword [0x402000]",
        ),
        // FF 20                  jmp qword [rax]
        (&[0xFF, 0x20], "jmp qword [rax]"),
        // --- data movement -------------------------------------------
        // B8 imm32
        (&[0xB8, 0x01, 0x00, 0x00, 0x00], "mov eax, 0x1"),
        // 66 B8 imm16
        (&[0x66, 0xB8, 0x34, 0x12], "mov ax, 0x1234"),
        // 48 B8 imm64
        (
            &[0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11],
            "mov rax, 0x1122334455667788",
        ),
        // B4 07                  mov ah, 7 (no REX -> high-byte alias)
        (&[0xB4, 0x07], "mov ah, 0x7"),
        // 41 B4 07               REX.B -> r12b, no high-byte alias
        (&[0x41, 0xB4, 0x07], "mov r12b, 0x7"),
        // 48 C7 C0 FF FF FF FF   mov rax, sign-extended -1
        (&[0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF], "mov rax, -0x1"),
        // C7 45 FC 0A 00 00 00   immediate to memory needs a size keyword
        (
            &[0xC7, 0x45, 0xFC, 0x0A, 0x00, 0x00, 0x00],
            "mov dword [rbp-0x4], 0xa",
        ),
        // C6 00 41               mov byte [rax], 0x41
        (&[0xC6, 0x00, 0x41], "mov byte [rax], 0x41"),
        // 88 07 / 8A 07          mov [rdi], al / mov al, [rdi]
        (&[0x88, 0x07], "mov [rdi], al"),
        (&[0x8A, 0x07], "mov al, [rdi]"),
        // 88 F0                  no REX: reg=110 is dh
        (&[0x88, 0xF0], "mov al, dh"),
        // 40 88 F0               bare REX: the same field is sil
        (&[0x40, 0x88, 0xF0], "mov al, sil"),
        // 48 8D 3D 00 00 00 00   lea rdi, [rip+0] -> the next instruction
        (
            &[0x48, 0x8D, 0x3D, 0x00, 0x00, 0x00, 0x00],
            "lea rdi, [0x401007]",
        ),
        // 48 8D 04 4B            lea rax, [rbx+rcx*2]
        (&[0x48, 0x8D, 0x04, 0x4B], "lea rax, [rbx+rcx*2]"),
        // 48 8B 04 C8            SIB scale=8
        (&[0x48, 0x8B, 0x04, 0xC8], "mov rax, [rax+rcx*8]"),
        // 4A 8B 04 E0            REX.WX: index becomes r12
        (&[0x4A, 0x8B, 0x04, 0xE0], "mov rax, [rax+r12*8]"),
        // 48 8B 80 00 01 00 00   disp32 base form
        (
            &[0x48, 0x8B, 0x80, 0x00, 0x01, 0x00, 0x00],
            "mov rax, [rax+0x100]",
        ),
        // 64 48 8B 04 25 28 ...  the stack-canary load
        (
            &[0x64, 0x48, 0x8B, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00],
            "mov rax, fs:[0x28]",
        ),
        // 93                     xchg eax, ebx
        (&[0x93], "xchg eax, ebx"),
        // 48 87 D8               xchg rax, rbx (Ev,Gv order: rm first)
        (&[0x48, 0x87, 0xD8], "xchg rax, rbx"),
        // 0F B6 C0 / 0F B7 C0
        (&[0x0F, 0xB6, 0xC0], "movzx eax, al"),
        (&[0x0F, 0xB7, 0xC0], "movzx eax, ax"),
        // 0F B6 03               memory source: size keyword required
        (&[0x0F, 0xB6, 0x03], "movzx eax, byte [rbx]"),
        // 48 0F BE 03            movsx rax, byte [rbx]
        (&[0x48, 0x0F, 0xBE, 0x03], "movsx rax, byte [rbx]"),
        // 48 63 D8               movsxd rbx, eax
        (&[0x48, 0x63, 0xD8], "movsxd rbx, eax"),
        // 48 0F 44 C1            cmove rax, rcx
        (&[0x48, 0x0F, 0x44, 0xC1], "cmove rax, rcx"),
        // 0F 94 C0               sete al
        (&[0x0F, 0x94, 0xC0], "sete al"),
        // 0F 95 03               setne byte [rbx]
        (&[0x0F, 0x95, 0x03], "setne byte [rbx]"),
        // --- ALU ------------------------------------------------------
        // 48 83 EC 20 / 48 83 C4 20
        (&[0x48, 0x83, 0xEC, 0x20], "sub rsp, 0x20"),
        (&[0x48, 0x83, 0xC4, 0x20], "add rsp, 0x20"),
        // 48 83 C4 F8            sign-extended negative imm8
        (&[0x48, 0x83, 0xC4, 0xF8], "add rsp, -0x8"),
        // 83 7D FC 00            cmp dword [rbp-4], 0
        (&[0x83, 0x7D, 0xFC, 0x00], "cmp dword [rbp-0x4], 0x0"),
        // 81 E1 FF 00 00 00      and ecx, 0xff
        (&[0x81, 0xE1, 0xFF, 0x00, 0x00, 0x00], "and ecx, 0xff"),
        // 80 3B 00               cmp byte [rbx], 0
        (&[0x80, 0x3B, 0x00], "cmp byte [rbx], 0x0"),
        // 01 D8 / 03 C3          add eax, ebx in both directions
        (&[0x01, 0xD8], "add eax, ebx"),
        (&[0x03, 0xC3], "add eax, ebx"),
        // 04 10 / 05 imm32       the accumulator short forms
        (&[0x04, 0x10], "add al, 0x10"),
        (&[0x05, 0x00, 0x01, 0x00, 0x00], "add eax, 0x100"),
        // 31 C0 / 48 31 C0       xor reg, reg is just xor
        (&[0x31, 0xC0], "xor eax, eax"),
        (&[0x48, 0x31, 0xC0], "xor rax, rax"),
        // 85 C0 / 84 C0          test
        (&[0x85, 0xC0], "test eax, eax"),
        (&[0x84, 0xC0], "test al, al"),
        (&[0xA8, 0x01], "test al, 0x1"),
        (&[0xA9, 0x00, 0x01, 0x00, 0x00], "test eax, 0x100"),
        // F7 C1 imm32            group 3 /0
        (&[0xF7, 0xC1, 0x01, 0x00, 0x00, 0x00], "test ecx, 0x1"),
        // F6 03 01               test byte [rbx], 1
        (&[0xF6, 0x03, 0x01], "test byte [rbx], 0x1"),
        // 48 FF C0 / 48 FF C8    inc/dec rax
        (&[0x48, 0xFF, 0xC0], "inc rax"),
        (&[0x48, 0xFF, 0xC8], "dec rax"),
        // FE C0 / FF 03          inc al / inc dword [rbx]
        (&[0xFE, 0xC0], "inc al"),
        (&[0xFF, 0x03], "inc dword [rbx]"),
        // 48 F7 D8 / 48 F7 D0    neg/not rax
        (&[0x48, 0xF7, 0xD8], "neg rax"),
        (&[0x48, 0xF7, 0xD0], "not rax"),
        // 48 F7 E1 / 48 F7 F9    mul/idiv rcx
        (&[0x48, 0xF7, 0xE1], "mul rcx"),
        (&[0x48, 0xF7, 0xF9], "idiv rcx"),
        // F7 33                  div dword [rbx]
        (&[0xF7, 0x33], "div dword [rbx]"),
        // 48 C1 E0 04            shl rax, 4
        (&[0x48, 0xC1, 0xE0, 0x04], "shl rax, 0x4"),
        // 48 D1 F8               sar rax, 1
        (&[0x48, 0xD1, 0xF8], "sar rax, 0x1"),
        // 48 D3 E8               shr rax, cl
        (&[0x48, 0xD3, 0xE8], "shr rax, cl"),
        // C1 65 FC 03            shl dword [rbp-4], 3
        (&[0xC1, 0x65, 0xFC, 0x03], "shl dword [rbp-0x4], 0x3"),
        // D2 23                  shl byte [rbx], cl
        (&[0xD2, 0x23], "shl byte [rbx], cl"),
        // 48 0F AF C1            imul rax, rcx
        (&[0x48, 0x0F, 0xAF, 0xC1], "imul rax, rcx"),
        // 6B C0 03               imul eax, eax, 3
        (&[0x6B, 0xC0, 0x03], "imul eax, eax, 0x3"),
        // 69 C0 E8 03 00 00      imul eax, eax, 0x3e8
        (
            &[0x69, 0xC0, 0xE8, 0x03, 0x00, 0x00],
            "imul eax, eax, 0x3e8",
        ),
        // 48 F7 2B               imul qword [rbx]
        (&[0x48, 0xF7, 0x2B], "imul qword [rbx]"),
        // --- SSE / SSE2 ----------------------------------------------
        // 0F 28 C1               movaps xmm0, xmm1 (no mandatory prefix)
        (&[0x0F, 0x28, 0xC1], "movaps xmm0, xmm1"),
        // 0F 10 C1               movups xmm0, xmm1
        (&[0x0F, 0x10, 0xC1], "movups xmm0, xmm1"),
        // 66 0F 28 C1            movapd xmm0, xmm1
        (&[0x66, 0x0F, 0x28, 0xC1], "movapd xmm0, xmm1"),
        // F3 0F 10 C1            movss xmm0, xmm1
        (&[0xF3, 0x0F, 0x10, 0xC1], "movss xmm0, xmm1"),
        // F2 0F 10 C0            movsd xmm0, xmm0
        (&[0xF2, 0x0F, 0x10, 0xC0], "movsd xmm0, xmm0"),
        // F2 0F 10 45 D8         movsd xmm0, [rbp-0x28] (load)
        (&[0xF2, 0x0F, 0x10, 0x45, 0xD8], "movsd xmm0, [rbp-0x28]"),
        // F2 0F 11 45 D8         movsd [rbp-0x28], xmm0 (store)
        (&[0xF2, 0x0F, 0x11, 0x45, 0xD8], "movsd [rbp-0x28], xmm0"),
        // F2 0F 58 D1            addsd xmm2, xmm1
        (&[0xF2, 0x0F, 0x58, 0xD1], "addsd xmm2, xmm1"),
        // F3 0F 58 00            addss xmm0, [rax]
        (&[0xF3, 0x0F, 0x58, 0x00], "addss xmm0, [rax]"),
        // F2 0F 59 D1            mulsd xmm2, xmm1
        (&[0xF2, 0x0F, 0x59, 0xD1], "mulsd xmm2, xmm1"),
        // 0F 58 C1               addps xmm0, xmm1 (packed, no prefix)
        (&[0x0F, 0x58, 0xC1], "addps xmm0, xmm1"),
        // 66 0F 58 C1            addpd xmm0, xmm1
        (&[0x66, 0x0F, 0x58, 0xC1], "addpd xmm0, xmm1"),
        // 0F 57 C0               xorps xmm0, xmm0
        (&[0x0F, 0x57, 0xC0], "xorps xmm0, xmm0"),
        // 66 0F 54 C1            andpd xmm0, xmm1
        (&[0x66, 0x0F, 0x54, 0xC1], "andpd xmm0, xmm1"),
        // 66 0F 2E C1            ucomisd xmm0, xmm1
        (&[0x66, 0x0F, 0x2E, 0xC1], "ucomisd xmm0, xmm1"),
        // 0F 2E C1               ucomiss xmm0, xmm1
        (&[0x0F, 0x2E, 0xC1], "ucomiss xmm0, xmm1"),
        // F2 0F 2A C3            cvtsi2sd xmm0, ebx
        (&[0xF2, 0x0F, 0x2A, 0xC3], "cvtsi2sd xmm0, ebx"),
        // F2 48 0F 2A C3         cvtsi2sd xmm0, rbx (REX.W)
        (&[0xF2, 0x48, 0x0F, 0x2A, 0xC3], "cvtsi2sd xmm0, rbx"),
        // F2 0F 2A 00            cvtsi2sd xmm0, dword [rax] (mem needs size)
        (&[0xF2, 0x0F, 0x2A, 0x00], "cvtsi2sd xmm0, dword [rax]"),
        // F3 0F 2A C3            cvtsi2ss xmm0, ebx
        (&[0xF3, 0x0F, 0x2A, 0xC3], "cvtsi2ss xmm0, ebx"),
        // F2 0F 2C C0            cvttsd2si eax, xmm0
        (&[0xF2, 0x0F, 0x2C, 0xC0], "cvttsd2si eax, xmm0"),
        // F2 48 0F 2D C0         cvtsd2si rax, xmm0 (REX.W)
        (&[0xF2, 0x48, 0x0F, 0x2D, 0xC0], "cvtsd2si rax, xmm0"),
        // 66 0F 6E C3            movd xmm0, ebx
        (&[0x66, 0x0F, 0x6E, 0xC3], "movd xmm0, ebx"),
        // 66 48 0F 6E C3         movq xmm0, rbx (REX.W)
        (&[0x66, 0x48, 0x0F, 0x6E, 0xC3], "movq xmm0, rbx"),
        // 66 0F 7E C3            movd ebx, xmm0 (store to GPR)
        (&[0x66, 0x0F, 0x7E, 0xC3], "movd ebx, xmm0"),
        // F3 0F 7E C1            movq xmm0, xmm1
        (&[0xF3, 0x0F, 0x7E, 0xC1], "movq xmm0, xmm1"),
        // 66 0F D6 01            movq [rcx], xmm0 (store)
        (&[0x66, 0x0F, 0xD6, 0x01], "movq [rcx], xmm0"),
        // --- prefixes -------------------------------------------------
        // F0 48 01 45 F8         lock add [rbp-8], rax
        (&[0xF0, 0x48, 0x01, 0x45, 0xF8], "lock add [rbp-0x8], rax"),
        // 98 / 48 98 / 66 98
        (&[0x98], "cwde"),
        (&[0x48, 0x98], "cdqe"),
        (&[0x66, 0x98], "cbw"),
        (&[0x99], "cdq"),
        (&[0x48, 0x99], "cqo"),
    ];

    /// Encodings this module refuses on purpose, with the reason.
    const REFUSED: &[(&[u8], &str)] = &[
        // lea with a register source (mod=11) is architecturally invalid.
        (&[0x48, 0x8D, 0xC0], "lea from a register"),
        // Group 5 /3 and /5 are far transfers, /7 is undefined.
        (&[0xFF, 0x18], "far call"),
        (&[0xFF, 0x38], "group 5 /7"),
        // Group 4 only defines /0 and /1.
        (&[0xFE, 0x10], "group 4 /2"),
        // Group 11 only defines /0.
        (&[0xC7, 0xC8, 0x00, 0x00, 0x00, 0x00], "group 11 /1"),
        // Group 1A only defines /0.
        (&[0x8F, 0xC8], "group 1A /1"),
        // Group 2 /6 is an undocumented alias.
        (&[0xC1, 0xF0, 0x01], "group 2 /6"),
        // The 67 address-size prefix is refused wholesale.
        (&[0x67, 0x8B, 0x00], "address-size override"),
        // A repeat prefix on anything but the CET markers and SSE.
        (&[0xF3, 0xC3], "rep ret"),
        // SSE forms the main decoder refuses, refused here too: the MMX
        // (no-66) encodings of 6E/7E/D6, and the converts without F3/F2.
        (&[0x0F, 0x6E, 0xC0], "movd/q without 66 (MMX)"),
        (&[0x0F, 0x7E, 0xC0], "7E without F3/66 (MMX)"),
        (&[0x0F, 0xD6, 0xC0], "movq store without 66 (MMX)"),
        (&[0x0F, 0x2A, 0xC0], "cvtsi2* without F3/F2"),
        (&[0x0F, 0x2C, 0xC0], "cvt*2si without F3/F2"),
        // A segment override with no memory operand to apply it to.
        (&[0x64, 0x48, 0x89, 0xC1], "fs with a register destination"),
        // lock with a register destination.
        (&[0xF0, 0x48, 0x01, 0xC1], "lock on a register"),
        // Opcode columns 6 and 7 are invalid in 64-bit mode.
        (&[0x06], "push es"),
        (&[0x0E], "push cs"),
        // Unmodelled two-byte map entries.
        (&[0x0F, 0xBA, 0xE0, 0x07], "bt group 8"),
        (&[0x0F, 0xC1, 0x03], "xadd"),
        // Truncated: no bytes at all.
        (&[], "empty"),
    ];

    /// Render at [`VA`], or `None`.
    fn fmt(bytes: &[u8]) -> Option<String> {
        X86Text.format(bytes, VA)
    }

    #[test]
    fn known_encodings_render_exactly() {
        for (bytes, expected) in CORPUS {
            let got = fmt(bytes);
            assert_eq!(
                got.as_deref(),
                Some(*expected),
                "encoding {bytes:02x?} rendered {got:?}"
            );
        }
    }

    #[test]
    fn corpus_consumes_every_byte() {
        for (bytes, expected) in CORPUS {
            let insn = decode(bytes).unwrap_or_else(|| panic!("{bytes:02x?} ({expected})"));
            assert_eq!(insn.len, bytes.len(), "{bytes:02x?} ({expected})");
        }
    }

    #[test]
    fn unmodelled_encodings_are_refused() {
        for (bytes, why) in REFUSED {
            assert_eq!(fmt(bytes), None, "{bytes:02x?} ({why}) should have no text");
        }
    }

    /// The mandatory `F2`/`F3` of an SSE encoding selects the variant; it
    /// must never leak into the listing as a `rep`/`repne` prefix, and the
    /// mnemonic must be the SSE one, not a repeat-modified string op.
    #[test]
    fn sse_forms_never_render_as_rep() {
        // Every SSE form in the corpus, plus a few F2/F3-carrying ones.
        let cases: &[(&[u8], &str)] = &[
            (&[0xF2, 0x0F, 0x10, 0xC0], "movsd"),
            (&[0xF3, 0x0F, 0x10, 0xC1], "movss"),
            (&[0xF2, 0x0F, 0x58, 0xD1], "addsd"),
            (&[0xF3, 0x0F, 0x58, 0x00], "addss"),
            (&[0xF2, 0x0F, 0x2A, 0xC3], "cvtsi2sd"),
            (&[0xF3, 0x0F, 0x2A, 0xC3], "cvtsi2ss"),
            (&[0xF2, 0x0F, 0x2C, 0xC0], "cvttsd2si"),
            (&[0xF3, 0x0F, 0x7E, 0xC1], "movq"),
        ];
        for (bytes, mnem) in cases {
            let text = fmt(bytes).unwrap_or_else(|| panic!("{bytes:02x?} should render"));
            assert!(
                !text.starts_with("rep") && !text.starts_with("repne"),
                "{bytes:02x?} rendered with a spurious repeat prefix: {text:?}"
            );
            assert!(
                text.starts_with(mnem),
                "{bytes:02x?} expected mnemonic {mnem}, got {text:?}"
            );
        }
    }

    /// Every rendered instruction must consume exactly as many bytes as
    /// [`crate::x86`] does, so the listing's text and the analysis's
    /// instruction boundaries can never drift apart.
    ///
    /// The one family this module renders that `x86` does not model is
    /// group 2 (shift/rotate). Those entries are allowed to have no
    /// cross-check, but the exemption is *checked*: an unmodelled
    /// encoding that is not a shift fails the test.
    #[test]
    fn rendered_length_agrees_with_the_flow_decoder() {
        const SHIFTS: [&str; 7] = ["rol", "ror", "rcl", "rcr", "shl", "shr", "sar"];
        for (bytes, expected) in CORPUS {
            let insn = decode(bytes).unwrap_or_else(|| panic!("{bytes:02x?} ({expected})"));
            match X86Decoder.decode_flow(bytes, VA) {
                Ok(d) => assert_eq!(
                    d.length as usize, insn.len,
                    "{bytes:02x?}: text consumed {} bytes, flow decoder {}",
                    insn.len, d.length
                ),
                Err(_) => assert!(
                    SHIFTS.iter().any(|m| expected.starts_with(m)),
                    "{bytes:02x?} ({expected}) is rendered but not modelled by x86::decode"
                ),
            }
        }
    }

    #[test]
    fn branch_targets_wrap_instead_of_overflowing() {
        // EB 00 at the very top of the address space: next = 0 (wrapped).
        assert_eq!(
            X86Text.format(&[0xEB, 0x00], u64::MAX - 1).as_deref(),
            Some("jmp 0x0")
        );
        // EB FE two bytes below the top: back to where it started.
        assert_eq!(
            X86Text.format(&[0xEB, 0xFE], u64::MAX - 1).as_deref(),
            Some("jmp 0xfffffffffffffffe")
        );
        // E9 -5 at VA 0: next = 5, target = 0.
        assert_eq!(
            X86Text
                .format(&[0xE9, 0xFB, 0xFF, 0xFF, 0xFF], 0)
                .as_deref(),
            Some("jmp 0x0")
        );
        // E9 -6 at VA 0 underflows to the top of the space.
        assert_eq!(
            X86Text
                .format(&[0xE9, 0xFA, 0xFF, 0xFF, 0xFF], 0)
                .as_deref(),
            Some("jmp 0xffffffffffffffff")
        );
        // A backwards call from a low address.
        assert_eq!(
            X86Text
                .format(&[0xE8, 0x00, 0x00, 0x00, 0x80], 0x1000)
                .as_deref(),
            Some("call 0xffffffff80001005")
        );
        // rel8 forward from the maximum address.
        assert_eq!(
            X86Text.format(&[0x74, 0x10], u64::MAX).as_deref(),
            Some("je 0x11")
        );
    }

    #[test]
    fn rip_relative_addresses_wrap_instead_of_overflowing() {
        // 48 8D 05 FF FF FF FF: lea rax, [rip-1], seven bytes long.
        assert_eq!(
            X86Text
                .format(&[0x48, 0x8D, 0x05, 0xFF, 0xFF, 0xFF, 0xFF], 0x401000)
                .as_deref(),
            Some("lea rax, [0x401006]")
        );
        assert_eq!(
            X86Text
                .format(
                    &[0x48, 0x8D, 0x05, 0xFF, 0xFF, 0xFF, 0xFF],
                    0xFFFF_FFFF_FFFF_FFF0
                )
                .as_deref(),
            Some("lea rax, [0xfffffffffffffff6]")
        );
        // At VA 0 the same encoding underflows to the top of the space.
        assert_eq!(
            X86Text
                .format(&[0x48, 0x8D, 0x05, 0xF9, 0xFF, 0xFF, 0xFF], 0)
                .as_deref(),
            Some("lea rax, [0x0]")
        );
    }

    /// Truncating any covered encoding must yield either no text or the
    /// text of a genuinely shorter instruction — and never a panic, and
    /// never a claim to have consumed bytes that are not there.
    #[test]
    fn truncation_sweep_never_panics() {
        for (bytes, _) in CORPUS {
            for len in 0..bytes.len() {
                let head = &bytes[..len];
                if let Some(insn) = decode(head) {
                    assert!(insn.len <= head.len(), "{head:02x?}");
                    assert!(!render(&insn, VA).is_empty(), "{head:02x?}");
                }
            }
        }
    }

    /// A tiny linear congruential generator (the "MMIX by Knuth"
    /// constants), used instead of `rand` so the sweep is dependency-free
    /// and byte-for-byte reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn next_byte(&mut self) -> u8 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // The high bits of an LCG are the well-distributed ones.
            (self.0 >> 56) as u8
        }
    }

    /// Formatting must be total: any byte string is either rendered or
    /// refused, never a panic. Where both this module and
    /// [`crate::x86`] accept the bytes, their lengths must agree.
    #[test]
    fn pseudo_random_sweep_never_panics() {
        let mut lcg = Lcg(0x1234_5678_9ABC_DEF0);
        let mut buf = [0u8; MAX_LEN];
        for round in 0..4000u32 {
            for b in buf.iter_mut() {
                *b = lcg.next_byte();
            }
            // Sweep every buffer length so truncation is exercised too.
            let take = (round as usize % (MAX_LEN + 1)).min(MAX_LEN);
            let bytes = &buf[..take];
            let va = (round as u64).wrapping_mul(0x1_0000_0001).wrapping_sub(3);
            if let Some(insn) = decode(bytes) {
                assert!(insn.len <= bytes.len(), "{bytes:02x?}");
                assert!(insn.len <= MAX_LEN, "{bytes:02x?}");
                let text = render(&insn, va);
                assert!(!text.is_empty(), "{bytes:02x?}");
                assert_eq!(X86Text.format(bytes, va).as_deref(), Some(text.as_str()));
                if let Ok(d) = X86Decoder.decode_flow(bytes, va) {
                    assert_eq!(d.length as usize, insn.len, "{bytes:02x?}");
                }
            }
        }
    }

    /// The same totality check over short, dense byte patterns, which hit
    /// the prefix loop and the group tables far more often than random
    /// bytes do.
    #[test]
    fn short_sequences_never_panic() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                let _ = X86Text.format(&[a], 0x1000);
                let _ = X86Text.format(&[a, b], 0x1000);
                let _ = X86Text.format(&[a, b, 0x00, 0x00, 0x00, 0x00, 0x00], 0x1000);
                let _ = X86Text.format(&[a, b, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF], u64::MAX);
            }
        }
    }

    #[test]
    fn trailing_bytes_are_not_consumed() {
        let insn = decode(&[0x90, 0xFF, 0xFF, 0xFF]).expect("nop");
        assert_eq!(insn.len, 1);
        let insn = decode(&[0xC3, 0x00, 0x00]).expect("ret");
        assert_eq!(insn.len, 1);
        // Text is unaffected by whatever follows the instruction.
        assert_eq!(
            fmt(&[0xEB, 0xFE, 0xCC, 0xCC]).as_deref(),
            Some("jmp 0x401000")
        );
    }

    #[test]
    fn over_long_encodings_are_refused() {
        // Sixteen `66` prefixes then a nop: one byte past the limit.
        let mut bytes = vec![0x66u8; 16];
        bytes.push(0x90);
        assert_eq!(fmt(&bytes), None);
        // Fourteen prefixes plus the opcode is exactly fifteen: fine.
        let mut ok = vec![0x66u8; 14];
        ok.push(0x90);
        assert_eq!(fmt(&ok).as_deref(), Some("xchg ax, ax"));
    }

    #[test]
    fn rex_is_voided_by_a_following_legacy_prefix() {
        // 48 66 B8 imm16: the REX.W is cancelled by the 66 that follows.
        assert_eq!(
            fmt(&[0x48, 0x66, 0xB8, 0x34, 0x12]).as_deref(),
            Some("mov ax, 0x1234")
        );
        // 66 48 B8 imm64: REX.W wins because it is adjacent to the opcode.
        assert_eq!(
            fmt(&[0x66, 0x48, 0xB8, 1, 0, 0, 0, 0, 0, 0, 0]).as_deref(),
            Some("mov rax, 0x1")
        );
    }

    #[test]
    fn byte_registers_follow_the_rex_rule() {
        // No REX: r/m 4..7 are ah/ch/dh/bh.
        assert_eq!(fmt(&[0x88, 0xE0]).as_deref(), Some("mov al, ah"));
        assert_eq!(fmt(&[0x00, 0xFB]).as_deref(), Some("add bl, bh"));
        // Any REX: the same fields are spl/bpl/sil/dil.
        assert_eq!(fmt(&[0x40, 0x00, 0xFB]).as_deref(), Some("add bl, dil"));
        assert_eq!(fmt(&[0x41, 0x00, 0xE0]).as_deref(), Some("add r8b, spl"));
    }

    #[test]
    fn memory_operands_format_cleanly() {
        // No base, no index: the displacement is the whole address.
        assert_eq!(
            fmt(&[0x48, 0x8B, 0x04, 0x25, 0x00, 0x20, 0x40, 0x00]).as_deref(),
            Some("mov rax, [0x402000]")
        );
        // Index with no base (SIB base=101, mod=00), scale 2.
        assert_eq!(
            fmt(&[0x48, 0x8B, 0x04, 0x4D, 0x00, 0x00, 0x00, 0x00]).as_deref(),
            Some("mov rax, [rcx*2]")
        );
        // A zero displacement is dropped when a register names the base.
        assert_eq!(
            fmt(&[0x48, 0x8B, 0x45, 0x00]).as_deref(),
            Some("mov rax, [rbp]")
        );
        // Full base + index*scale + disp8.
        assert_eq!(
            fmt(&[0x48, 0x8B, 0x44, 0x8B, 0x10]).as_deref(),
            Some("mov rax, [rbx+rcx*4+0x10]")
        );
        // ... and the negative-displacement spelling.
        assert_eq!(
            fmt(&[0x48, 0x8B, 0x44, 0x8B, 0xF0]).as_deref(),
            Some("mov rax, [rbx+rcx*4-0x10]")
        );
        // r13 as a base always carries an explicit displacement, because
        // mod=00 rm=101 is the RIP form.
        assert_eq!(
            fmt(&[0x49, 0x8B, 0x45, 0x00]).as_deref(),
            Some("mov rax, [r13]")
        );
        // r12 as a base forces a SIB byte with no index.
        assert_eq!(
            fmt(&[0x49, 0x8B, 0x04, 0x24]).as_deref(),
            Some("mov rax, [r12]")
        );
    }

    #[test]
    fn size_keywords_appear_only_where_needed() {
        // A register operand fixes the width: no keyword.
        assert_eq!(fmt(&[0x48, 0x89, 0x03]).as_deref(), Some("mov [rbx], rax"));
        assert_eq!(fmt(&[0x8B, 0x03]).as_deref(), Some("mov eax, [rbx]"));
        // Nothing else does: keyword required.
        assert_eq!(
            fmt(&[0x48, 0xC7, 0x03, 0x01, 0x00, 0x00, 0x00]).as_deref(),
            Some("mov qword [rbx], 0x1")
        );
        assert_eq!(fmt(&[0x66, 0xFF, 0x03]).as_deref(), Some("inc word [rbx]"));
        assert_eq!(fmt(&[0xFE, 0x0B]).as_deref(), Some("dec byte [rbx]"));
    }
}
