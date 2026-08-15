//! x86-64 instruction decoder (64-bit mode only).
//!
//! Clean-room implementation from the public Intel SDM Volume 2 opcode
//! maps and the AMD64 Architecture Programmer's Manual Volume 3. This
//! pass decodes a pragmatic, analysis-oriented subset of the ISA: the
//! moves, stack operations, arithmetic/logic groups, and — most
//! importantly — every common control-flow instruction, classified into
//! [`Flow`] so that control-flow recovery can be built on top.
//!
//! Deliberately out of scope for now (rejected with a typed error, never
//! given a guessed length): SSE/AVX and everything else behind mandatory
//! prefixes, the 0F 38 / 0F 3A maps, VEX/EVEX encodings, far
//! calls/jumps/returns, string instructions, and privileged
//! system instructions beyond `hlt`/`syscall`/`cpuid`/`rdtsc`.
//!
//! The decoder is total: every byte sequence either decodes or returns a
//! [`ParseError`]; it never panics and never reads past the input or the
//! architectural 15-byte instruction length limit.

use crate::error::{ParseError, Result};

/// Architectural maximum length of one instruction, in bytes.
///
/// An instruction whose prefixes + opcode + operands would exceed this is
/// a decode error (the CPU raises #GP for the same condition).
pub const MAX_INSTRUCTION_LEN: usize = 15;

/// Operand width of a register or operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    /// 8-bit (`al`, `r8b`, ...).
    W8,
    /// 16-bit (`ax`, `r8w`, ...).
    W16,
    /// 32-bit (`eax`, `r8d`, ...).
    W32,
    /// 64-bit (`rax`, `r8`, ...).
    W64,
}

/// One of the 16 general-purpose registers at a specific width.
///
/// `num` is the architectural register number 0-15 (rAX, rCX, rDX, rBX,
/// rSP, rBP, rSI, rDI, r8-r15). When `high_byte` is set the register is
/// the legacy high-byte alias of `num` (0 = `ah`, 1 = `ch`, 2 = `dh`,
/// 3 = `bh`), which is only encodable when no REX prefix is present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reg {
    pub num: u8,
    pub width: Width,
    pub high_byte: bool,
}

impl Reg {
    /// A plain GPR of the given number (0-15) and width.
    pub fn gpr(num: u8, width: Width) -> Reg {
        Reg {
            num: num & 0xF,
            width,
            high_byte: false,
        }
    }

    /// The byte register selected by a 3/4-bit field, honoring the legacy
    /// AH/CH/DH/BH aliases that apply when no REX prefix is present.
    fn byte_reg(num: u8, rex_present: bool) -> Reg {
        if !rex_present && (4..=7).contains(&num) {
            Reg {
                num: num - 4,
                width: Width::W8,
                high_byte: true,
            }
        } else {
            Reg::gpr(num, Width::W8)
        }
    }

    /// Conventional assembly name of this register (e.g. `"rax"`, `"r8d"`,
    /// `"ah"`).
    pub fn name(&self) -> &'static str {
        const Q: [&str; 16] = [
            "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11",
            "r12", "r13", "r14", "r15",
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
            "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b",
            "r12b", "r13b", "r14b", "r15b",
        ];
        const H: [&str; 4] = ["ah", "ch", "dh", "bh"];
        if self.high_byte {
            return H.get(self.num as usize).copied().unwrap_or("?");
        }
        let table = match self.width {
            Width::W64 => &Q,
            Width::W32 => &D,
            Width::W16 => &W,
            Width::W8 => &B,
        };
        table.get(self.num as usize).copied().unwrap_or("?")
    }
}

/// Segment register named by a legacy override prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    Es,
    Cs,
    Ss,
    Ds,
    Fs,
    Gs,
}

/// `F2`/`F3` repeat prefixes, recorded but not interpreted (the string
/// instructions they modify are outside this pass's coverage).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rep {
    /// `F3`.
    Rep,
    /// `F2`.
    Repne,
}

/// Condition code from the low nibble of a `Jcc`/`SETcc`/`CMOVcc` opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    O,
    No,
    B,
    Ae,
    E,
    Ne,
    Be,
    A,
    S,
    Ns,
    P,
    Np,
    L,
    Ge,
    Le,
    G,
}

impl Cond {
    fn from_nibble(n: u8) -> Cond {
        match n & 0xF {
            0 => Cond::O,
            1 => Cond::No,
            2 => Cond::B,
            3 => Cond::Ae,
            4 => Cond::E,
            5 => Cond::Ne,
            6 => Cond::Be,
            7 => Cond::A,
            8 => Cond::S,
            9 => Cond::Ns,
            10 => Cond::P,
            11 => Cond::Np,
            12 => Cond::L,
            13 => Cond::Ge,
            14 => Cond::Le,
            _ => Cond::G,
        }
    }
}

/// Decoded operation, at mnemonic granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Add,
    Or,
    Adc,
    Sbb,
    And,
    Sub,
    Xor,
    Cmp,
    Test,
    Mov,
    /// `movsx` (byte/word source, sign-extended).
    Movsx,
    /// `movzx` (byte/word source, zero-extended).
    Movzx,
    /// `movsxd` (opcode `63`, doubleword source sign-extended to 64 bits).
    Movsxd,
    Lea,
    Push,
    Pop,
    Xchg,
    Inc,
    Dec,
    Not,
    Neg,
    Mul,
    Imul,
    Div,
    Idiv,
    Nop,
    /// `cwde`/`cdqe` (opcode `98`; REX.W selects the 64-bit form).
    Cwde,
    /// `cdq`/`cqo` (opcode `99`; REX.W selects the 64-bit form).
    Cdq,
    Ret,
    Leave,
    Int3,
    /// `int imm8`.
    Int,
    Call,
    Jmp,
    /// Conditional jump (`70-7F` rel8, `0F 80-8F` rel32).
    Jcc(Cond),
    /// `0F 90-9F`: set byte on condition.
    Setcc(Cond),
    /// `0F 40-4F`: conditional move.
    Cmov(Cond),
    Syscall,
    Ud2,
    Cpuid,
    Rdtsc,
    Bt,
    Bts,
    Btr,
    Btc,
    Cmpxchg,
    Xadd,
    Hlt,
    /// `F3 0F 1E FA`: CET branch-target marker (no-op semantics).
    Endbr64,
    /// `F3 0F 1E FB`.
    Endbr32,
    /// An SSE/SSE2 instruction from the subset this decoder models: scalar
    /// and packed moves, the four floating arithmetic ops, the bitwise
    /// pair, the ordered/unordered compares, the int<->float converts, and
    /// the GPR<->XMM moves. `mnem` is the resolved assembly mnemonic
    /// (`movaps`, `addsd`, `cvtsi2sd`, ...). `writes_flags` is set for the
    /// compares (`comiss`/`comisd`/`ucomiss`/`ucomisd`), which are the only
    /// members that touch EFLAGS. The point of decoding these is to keep CFG
    /// recovery, the lifter, and devirt from truncating a function at its
    /// first floating-point instruction; the XMM datapath itself is not
    /// modeled as IR values.
    Sse {
        mnem: &'static str,
        writes_flags: bool,
    },
}

/// One decoded operand.
///
/// Immediates are stored sign-extended to `i64`; interpret them with the
/// instruction's operand width. For relative branches the operand is the
/// raw signed displacement — use [`Instruction::flow`] for the absolute
/// target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg(Reg),
    /// An XMM vector register, by number (0-15). The IR does not model the
    /// XMM datapath, so this operand carries only the register number for
    /// display; the SSE lifter treats such operands as opaque.
    Xmm(u8),
    Mem {
        base: Option<Reg>,
        index: Option<Reg>,
        /// Index scale factor (1, 2, 4 or 8); 1 when `index` is `None`.
        scale: u8,
        disp: i64,
        /// `[rip + disp]` addressing (mod=00, r/m=101 in 64-bit mode).
        rip_relative: bool,
    },
    Imm(i64),
}

/// Control-flow classification of one instruction: the shared
/// [`crate::model::Flow`], re-exported so `x86::Flow` stays a valid path.
///
/// Branch targets are absolute virtual addresses, computed from the
/// instruction VA the caller passed to [`decode`] (target = VA of the
/// *next* instruction + signed displacement, per the SDM).
pub use crate::model::Flow;

/// One decoded x86-64 instruction.
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    /// Total encoded length in bytes (prefixes included), at most 15.
    pub length: u8,
    pub opcode: Opcode,
    /// Operands in Intel order (destination first).
    pub operands: Vec<Operand>,
    /// Control-flow classification (see [`Flow`]).
    pub flow: Flow,
    /// `F0` lock prefix seen (not validated against lockable forms).
    pub lock: bool,
    /// `F2`/`F3` prefix seen.
    pub rep: Option<Rep>,
    /// Segment-override prefix seen (`2E`/`36`/`3E`/`26`/`64`/`65`).
    pub segment: Option<Segment>,
}

impl Instruction {
    /// Control-flow classification of this instruction.
    pub fn flow(&self) -> Flow {
        self.flow
    }
}

/// The decode error for an instruction that would exceed the 15-byte
/// architectural limit.
fn too_long() -> ParseError {
    ParseError::Unsupported(format!(
        "instruction exceeds the {MAX_INSTRUCTION_LEN}-byte length limit"
    ))
}

/// Bounds- and length-limit-checked cursor over the instruction bytes.
///
/// Reads past the end of the input are [`ParseError::UnexpectedEof`];
/// reads past the 15-byte architectural limit are a length error even
/// when more input is available.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(count).ok_or_else(too_long)?;
        if end > MAX_INSTRUCTION_LEN {
            return Err(too_long());
        }
        if end > self.data.len() {
            return Err(ParseError::UnexpectedEof {
                offset: self.pos,
                needed: count,
                available: self.data.len() - self.pos,
            });
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// imm8 / rel8, sign-extended.
    fn i8(&mut self) -> Result<i64> {
        Ok(self.u8()? as i8 as i64)
    }

    /// imm16, sign-extended.
    fn i16(&mut self) -> Result<i64> {
        let b = self.take(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]) as i64)
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    /// imm32 / rel32 / disp32, sign-extended.
    fn i32(&mut self) -> Result<i64> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
    }

    fn i64(&mut self) -> Result<i64> {
        let b = self.take(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
}

/// Prefix state accumulated before the opcode.
#[derive(Default)]
struct Prefixes {
    /// The REX byte (`40-4F`), if one immediately precedes the opcode.
    rex: Option<u8>,
    /// `66` operand-size override seen.
    opsize: bool,
    /// `67` address-size override seen.
    addrsize: bool,
    lock: bool,
    rep: Option<Rep>,
    segment: Option<Segment>,
}

impl Prefixes {
    fn rex_present(&self) -> bool {
        self.rex.is_some()
    }

    fn rex_w(&self) -> bool {
        self.rex.unwrap_or(0) & 0x8 != 0
    }

    /// REX.R as a bit already shifted into position 3.
    fn rex_r(&self) -> u8 {
        (self.rex.unwrap_or(0) & 0x4) << 1
    }

    fn rex_x(&self) -> u8 {
        (self.rex.unwrap_or(0) & 0x2) << 2
    }

    fn rex_b(&self) -> u8 {
        (self.rex.unwrap_or(0) & 0x1) << 3
    }

    /// Effective operand width for `v`-width operations: REX.W wins over
    /// the `66` prefix, which wins over the 32-bit default.
    fn vwidth(&self) -> Width {
        if self.rex_w() {
            Width::W64
        } else if self.opsize {
            Width::W16
        } else {
            Width::W32
        }
    }

    /// Effective width for push/pop/indirect-branch operands, which
    /// default to 64 bits in long mode (`66` can still shrink them; REX.W
    /// is redundant).
    fn stack_width(&self) -> Width {
        if self.opsize { Width::W16 } else { Width::W64 }
    }

    /// Width of registers used inside a memory address.
    fn addr_reg_width(&self) -> Width {
        if self.addrsize { Width::W32 } else { Width::W64 }
    }
}

/// The register-or-memory half of a decoded ModRM byte, before an operand
/// width has been applied.
enum Rm {
    /// mod=11: register number 0-15 (REX.B applied).
    Reg(u8),
    Mem {
        base: Option<u8>,
        index: Option<u8>,
        scale: u8,
        disp: i64,
        rip_relative: bool,
    },
}

/// A decoded ModRM byte (and any SIB + displacement that follows it).
struct ModRm {
    /// The reg field with REX.R applied (0-15). Group opcodes use only
    /// the low 3 bits (`reg & 7`), per the SDM's `/digit` notation.
    reg: u8,
    rm: Rm,
}

/// Decode a ModRM byte plus its SIB byte and displacement, if any
/// (SDM Vol 2, Table 2-2 and the 64-bit-mode notes in §2.2.1).
fn decode_modrm(cur: &mut Cursor, pfx: &Prefixes) -> Result<ModRm> {
    let byte = cur.u8()?;
    let mode = byte >> 6;
    let reg = ((byte >> 3) & 7) | pfx.rex_r();
    let rm_bits = byte & 7;

    if mode == 3 {
        return Ok(ModRm {
            reg,
            rm: Rm::Reg(rm_bits | pfx.rex_b()),
        });
    }

    let mut base = None;
    let mut index = None;
    let mut scale = 1u8;
    let mut rip_relative = false;
    // 0, 1 or 4 displacement bytes.
    let mut disp_size = match mode {
        0 => 0,
        1 => 1,
        _ => 4,
    };

    if rm_bits == 4 {
        // SIB byte follows.
        let sib = cur.u8()?;
        let index_bits = (sib >> 3) & 7;
        let base_bits = sib & 7;
        // index=100 with REX.X clear encodes "no index"; with REX.X set
        // it selects r12, which is a valid index.
        if index_bits != 4 || pfx.rex_x() != 0 {
            index = Some(index_bits | pfx.rex_x());
            scale = 1 << (sib >> 6);
        }
        // base=101 with mod=00 means "disp32, no base" regardless of
        // REX.B; otherwise the base register applies.
        if base_bits == 5 && mode == 0 {
            disp_size = 4;
        } else {
            base = Some(base_bits | pfx.rex_b());
        }
    } else if mode == 0 && rm_bits == 5 {
        // 64-bit mode: [rip + disp32].
        rip_relative = true;
        disp_size = 4;
    } else {
        base = Some(rm_bits | pfx.rex_b());
    }

    let disp = match disp_size {
        0 => 0,
        1 => cur.i8()?,
        _ => cur.i32()?,
    };

    Ok(ModRm {
        reg,
        rm: Rm::Mem {
            base,
            index,
            scale,
            disp,
            rip_relative,
        },
    })
}

/// Convert the r/m half of a ModRM into an [`Operand`] of the given width.
fn rm_operand(rm: &Rm, width: Width, pfx: &Prefixes) -> Operand {
    match *rm {
        Rm::Reg(num) => Operand::Reg(if width == Width::W8 {
            Reg::byte_reg(num, pfx.rex_present())
        } else {
            Reg::gpr(num, width)
        }),
        Rm::Mem {
            base,
            index,
            scale,
            disp,
            rip_relative,
        } => {
            let aw = pfx.addr_reg_width();
            Operand::Mem {
                base: base.map(|n| Reg::gpr(n, aw)),
                index: index.map(|n| Reg::gpr(n, aw)),
                scale,
                disp,
                rip_relative,
            }
        }
    }
}

/// The reg half of a ModRM as a register operand of the given width.
fn reg_operand(num: u8, width: Width, pfx: &Prefixes) -> Operand {
    Operand::Reg(if width == Width::W8 {
        Reg::byte_reg(num, pfx.rex_present())
    } else {
        Reg::gpr(num, width)
    })
}

/// Read an "immz" immediate: 16 bits under a `66` prefix, else 32 bits
/// (sign-extended to the operand width, including 64-bit under REX.W).
fn immz(cur: &mut Cursor, width: Width) -> Result<i64> {
    match width {
        Width::W16 => cur.i16(),
        _ => cur.i32(),
    }
}

/// Pre-VA control-flow classification produced by the opcode dispatch;
/// relative displacements are resolved into absolute [`Flow`] targets
/// once the total instruction length is known.
enum FlowKind {
    Seq,
    RelJump(i64),
    RelCond(i64),
    RelCall(i64),
    IndirectJump,
    IndirectCall,
    Ret,
    Interrupt,
    Halt,
}

/// Group-1 opcode selected by the ModRM reg field of `80`/`81`/`83`.
fn group1_opcode(reg: u8) -> Opcode {
    match reg & 7 {
        0 => Opcode::Add,
        1 => Opcode::Or,
        2 => Opcode::Adc,
        3 => Opcode::Sbb,
        4 => Opcode::And,
        5 => Opcode::Sub,
        6 => Opcode::Xor,
        _ => Opcode::Cmp,
    }
}

/// The arithmetic/logic opcode encoded in bits 3-5 of a `00-3D` opcode.
fn arith_opcode(bits: u8) -> Opcode {
    match bits & 7 {
        0 => Opcode::Add,
        1 => Opcode::Or,
        2 => Opcode::Adc,
        3 => Opcode::Sbb,
        4 => Opcode::And,
        5 => Opcode::Sub,
        6 => Opcode::Xor,
        _ => Opcode::Cmp,
    }
}

fn unknown_opcode(bytes: &[u8]) -> ParseError {
    let mut hex = String::new();
    for b in bytes {
        if !hex.is_empty() {
            hex.push(' ');
        }
        hex.push_str(&format!("{b:02x}"));
    }
    ParseError::Unsupported(format!("unrecognized or unmodeled opcode: {hex}"))
}

/// Decode one instruction from `bytes`, which starts at virtual address
/// `va` (used only to resolve relative branch targets in [`Flow`]).
///
/// On success the instruction's `length` tells how many bytes were
/// consumed; `bytes` may extend past the instruction. Truncated input
/// yields [`ParseError::UnexpectedEof`]; unmodeled opcodes and
/// over-long encodings yield [`ParseError::Unsupported`] — never a
/// guessed length, and never a panic.
pub fn decode(bytes: &[u8], va: u64) -> Result<Instruction> {
    let mut cur = Cursor::new(bytes);
    let mut pfx = Prefixes::default();

    // Prefix loop. A REX byte only takes effect when it immediately
    // precedes the opcode, so any legacy prefix that follows one voids it.
    let opcode_byte = loop {
        let b = cur.u8()?;
        match b {
            0xF0 => {
                pfx.lock = true;
                pfx.rex = None;
            }
            0xF2 => {
                pfx.rep = Some(Rep::Repne);
                pfx.rex = None;
            }
            0xF3 => {
                pfx.rep = Some(Rep::Rep);
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
                pfx.segment = Some(Segment::Es);
                pfx.rex = None;
            }
            0x2E => {
                pfx.segment = Some(Segment::Cs);
                pfx.rex = None;
            }
            0x36 => {
                pfx.segment = Some(Segment::Ss);
                pfx.rex = None;
            }
            0x3E => {
                pfx.segment = Some(Segment::Ds);
                pfx.rex = None;
            }
            0x64 => {
                pfx.segment = Some(Segment::Fs);
                pfx.rex = None;
            }
            0x65 => {
                pfx.segment = Some(Segment::Gs);
                pfx.rex = None;
            }
            0x40..=0x4F => pfx.rex = Some(b),
            other => break other,
        }
    };

    let (opcode, operands, kind) = dispatch(&mut cur, &pfx, opcode_byte)?;

    let length = cur.pos as u8;
    let next_va = va.wrapping_add(length as u64);
    let flow = match kind {
        FlowKind::Seq => Flow::Sequential,
        FlowKind::RelJump(d) => Flow::Jump(next_va.wrapping_add(d as u64)),
        FlowKind::RelCond(d) => Flow::CondJump(next_va.wrapping_add(d as u64)),
        FlowKind::RelCall(d) => Flow::Call(next_va.wrapping_add(d as u64)),
        FlowKind::IndirectJump => Flow::IndirectJump,
        FlowKind::IndirectCall => Flow::IndirectCall,
        FlowKind::Ret => Flow::Return,
        FlowKind::Interrupt => Flow::Interrupt,
        FlowKind::Halt => Flow::Halt,
    };

    Ok(Instruction {
        length,
        opcode,
        operands,
        flow,
        lock: pfx.lock,
        rep: pfx.rep,
        segment: pfx.segment,
    })
}

/// Dispatch on a one-byte-map opcode.
fn dispatch(cur: &mut Cursor, pfx: &Prefixes, op: u8) -> Result<(Opcode, Vec<Operand>, FlowKind)> {
    let vw = pfx.vwidth();
    let sw = pfx.stack_width();
    let seq = FlowKind::Seq;

    let out = match op {
        // Arithmetic/logic groups: add/or/adc/sbb/and/sub/xor/cmp, six
        // encodings each (Eb,Gb / Ev,Gv / Gb,Eb / Gv,Ev / AL,ib / rAX,iz).
        // Columns 6 and 7 (push es, daa, ...) are invalid in 64-bit mode.
        0x00..=0x3D if op & 7 <= 5 => {
            let mnem = arith_opcode(op >> 3);
            let ops = match op & 7 {
                0 => {
                    let m = decode_modrm(cur, pfx)?;
                    vec![
                        rm_operand(&m.rm, Width::W8, pfx),
                        reg_operand(m.reg, Width::W8, pfx),
                    ]
                }
                1 => {
                    let m = decode_modrm(cur, pfx)?;
                    vec![rm_operand(&m.rm, vw, pfx), reg_operand(m.reg, vw, pfx)]
                }
                2 => {
                    let m = decode_modrm(cur, pfx)?;
                    vec![
                        reg_operand(m.reg, Width::W8, pfx),
                        rm_operand(&m.rm, Width::W8, pfx),
                    ]
                }
                3 => {
                    let m = decode_modrm(cur, pfx)?;
                    vec![reg_operand(m.reg, vw, pfx), rm_operand(&m.rm, vw, pfx)]
                }
                4 => vec![
                    Operand::Reg(Reg::gpr(0, Width::W8)),
                    Operand::Imm(cur.i8()?),
                ],
                _ => vec![Operand::Reg(Reg::gpr(0, vw)), Operand::Imm(immz(cur, vw)?)],
            };
            (mnem, ops, seq)
        }
        // push/pop r64.
        0x50..=0x57 => {
            let r = (op & 7) | pfx.rex_b();
            (Opcode::Push, vec![Operand::Reg(Reg::gpr(r, sw))], seq)
        }
        0x58..=0x5F => {
            let r = (op & 7) | pfx.rex_b();
            (Opcode::Pop, vec![Operand::Reg(Reg::gpr(r, sw))], seq)
        }
        // movsxd Gv, Ed.
        0x63 => {
            let m = decode_modrm(cur, pfx)?;
            let dw = if pfx.rex_w() { Width::W64 } else { Width::W32 };
            (
                Opcode::Movsxd,
                vec![reg_operand(m.reg, dw, pfx), rm_operand(&m.rm, Width::W32, pfx)],
                seq,
            )
        }
        // push iz / ib.
        0x68 => (Opcode::Push, vec![Operand::Imm(immz(cur, vw)?)], seq),
        0x6A => (Opcode::Push, vec![Operand::Imm(cur.i8()?)], seq),
        // imul Gv, Ev, iz / ib.
        0x69 | 0x6B => {
            let m = decode_modrm(cur, pfx)?;
            let imm = if op == 0x69 {
                immz(cur, vw)?
            } else {
                cur.i8()?
            };
            (
                Opcode::Imul,
                vec![
                    reg_operand(m.reg, vw, pfx),
                    rm_operand(&m.rm, vw, pfx),
                    Operand::Imm(imm),
                ],
                seq,
            )
        }
        // jcc rel8.
        0x70..=0x7F => {
            let rel = cur.i8()?;
            (
                Opcode::Jcc(Cond::from_nibble(op)),
                vec![Operand::Imm(rel)],
                FlowKind::RelCond(rel),
            )
        }
        // Group 1: add/or/adc/sbb/and/sub/xor/cmp with immediate.
        // (82 is the invalid-in-64-bit alias of 80.)
        0x80 | 0x81 | 0x83 => {
            let m = decode_modrm(cur, pfx)?;
            let (w, imm) = match op {
                0x80 => (Width::W8, cur.i8()?),
                0x81 => (vw, immz(cur, vw)?),
                _ => (vw, cur.i8()?),
            };
            (
                group1_opcode(m.reg),
                vec![rm_operand(&m.rm, w, pfx), Operand::Imm(imm)],
                seq,
            )
        }
        // test / xchg / mov, byte and v-width /r forms.
        0x84 | 0x86 | 0x88 => {
            let m = decode_modrm(cur, pfx)?;
            let mnem = match op {
                0x84 => Opcode::Test,
                0x86 => Opcode::Xchg,
                _ => Opcode::Mov,
            };
            (
                mnem,
                vec![
                    rm_operand(&m.rm, Width::W8, pfx),
                    reg_operand(m.reg, Width::W8, pfx),
                ],
                seq,
            )
        }
        0x85 | 0x87 | 0x89 => {
            let m = decode_modrm(cur, pfx)?;
            let mnem = match op {
                0x85 => Opcode::Test,
                0x87 => Opcode::Xchg,
                _ => Opcode::Mov,
            };
            (
                mnem,
                vec![rm_operand(&m.rm, vw, pfx), reg_operand(m.reg, vw, pfx)],
                seq,
            )
        }
        0x8A => {
            let m = decode_modrm(cur, pfx)?;
            (
                Opcode::Mov,
                vec![
                    reg_operand(m.reg, Width::W8, pfx),
                    rm_operand(&m.rm, Width::W8, pfx),
                ],
                seq,
            )
        }
        0x8B => {
            let m = decode_modrm(cur, pfx)?;
            (
                Opcode::Mov,
                vec![reg_operand(m.reg, vw, pfx), rm_operand(&m.rm, vw, pfx)],
                seq,
            )
        }
        // lea Gv, M: the r/m half must be a memory form.
        0x8D => {
            let m = decode_modrm(cur, pfx)?;
            if matches!(m.rm, Rm::Reg(_)) {
                return Err(ParseError::Unsupported(
                    "lea with a register source (ModRM mod=11) is invalid".into(),
                ));
            }
            (
                Opcode::Lea,
                vec![reg_operand(m.reg, vw, pfx), rm_operand(&m.rm, vw, pfx)],
                seq,
            )
        }
        // Group 1A: pop r/m64 (/0 only).
        0x8F => {
            let m = decode_modrm(cur, pfx)?;
            if m.reg & 7 != 0 {
                return Err(unknown_opcode(&[op]));
            }
            (Opcode::Pop, vec![rm_operand(&m.rm, sw, pfx)], seq)
        }
        // nop, or xchg rAX, r8 when REX.B is set.
        0x90 if pfx.rex_b() == 0 => (Opcode::Nop, vec![], seq),
        // xchg rAX, r.
        0x90..=0x97 => {
            let r = (op & 7) | pfx.rex_b();
            (
                Opcode::Xchg,
                vec![
                    Operand::Reg(Reg::gpr(0, vw)),
                    Operand::Reg(Reg::gpr(r, vw)),
                ],
                seq,
            )
        }
        0x98 => (Opcode::Cwde, vec![], seq),
        0x99 => (Opcode::Cdq, vec![], seq),
        // test AL, ib / rAX, iz.
        0xA8 => (
            Opcode::Test,
            vec![
                Operand::Reg(Reg::gpr(0, Width::W8)),
                Operand::Imm(cur.i8()?),
            ],
            seq,
        ),
        0xA9 => (
            Opcode::Test,
            vec![Operand::Reg(Reg::gpr(0, vw)), Operand::Imm(immz(cur, vw)?)],
            seq,
        ),
        // mov r8, ib.
        0xB0..=0xB7 => {
            let r = (op & 7) | pfx.rex_b();
            (
                Opcode::Mov,
                vec![
                    Operand::Reg(Reg::byte_reg(r, pfx.rex_present())),
                    Operand::Imm(cur.i8()?),
                ],
                seq,
            )
        }
        // mov r, imm: the only imm64 form (with REX.W).
        0xB8..=0xBF => {
            let r = (op & 7) | pfx.rex_b();
            let imm = match vw {
                Width::W64 => cur.i64()?,
                Width::W16 => cur.i16()?,
                _ => cur.i32()?,
            };
            (
                Opcode::Mov,
                vec![Operand::Reg(Reg::gpr(r, vw)), Operand::Imm(imm)],
                seq,
            )
        }
        // ret imm16 / ret.
        0xC2 => {
            let imm = cur.u16()? as i64;
            (Opcode::Ret, vec![Operand::Imm(imm)], FlowKind::Ret)
        }
        0xC3 => (Opcode::Ret, vec![], FlowKind::Ret),
        // Group 11: mov r/m, imm (/0 only).
        0xC6 | 0xC7 => {
            let m = decode_modrm(cur, pfx)?;
            if m.reg & 7 != 0 {
                return Err(unknown_opcode(&[op]));
            }
            let (w, imm) = if op == 0xC6 {
                (Width::W8, cur.i8()?)
            } else {
                (vw, immz(cur, vw)?)
            };
            (
                Opcode::Mov,
                vec![rm_operand(&m.rm, w, pfx), Operand::Imm(imm)],
                seq,
            )
        }
        0xC9 => (Opcode::Leave, vec![], seq),
        0xCC => (Opcode::Int3, vec![], FlowKind::Interrupt),
        0xCD => (
            Opcode::Int,
            vec![Operand::Imm(cur.u8()? as i64)],
            FlowKind::Interrupt,
        ),
        // call/jmp rel32, jmp rel8.
        0xE8 => {
            let rel = cur.i32()?;
            (Opcode::Call, vec![Operand::Imm(rel)], FlowKind::RelCall(rel))
        }
        0xE9 => {
            let rel = cur.i32()?;
            (Opcode::Jmp, vec![Operand::Imm(rel)], FlowKind::RelJump(rel))
        }
        0xEB => {
            let rel = cur.i8()?;
            (Opcode::Jmp, vec![Operand::Imm(rel)], FlowKind::RelJump(rel))
        }
        0xF4 => (Opcode::Hlt, vec![], FlowKind::Halt),
        // Group 3: test/not/neg/mul/imul/div/idiv.
        0xF6 | 0xF7 => {
            let m = decode_modrm(cur, pfx)?;
            let w = if op == 0xF6 { Width::W8 } else { vw };
            let rm = rm_operand(&m.rm, w, pfx);
            match m.reg & 7 {
                0 | 1 => {
                    let imm = if op == 0xF6 {
                        cur.i8()?
                    } else {
                        immz(cur, vw)?
                    };
                    (Opcode::Test, vec![rm, Operand::Imm(imm)], seq)
                }
                2 => (Opcode::Not, vec![rm], seq),
                3 => (Opcode::Neg, vec![rm], seq),
                4 => (Opcode::Mul, vec![rm], seq),
                5 => (Opcode::Imul, vec![rm], seq),
                6 => (Opcode::Div, vec![rm], seq),
                _ => (Opcode::Idiv, vec![rm], seq),
            }
        }
        // Group 4: inc/dec r/m8.
        0xFE => {
            let m = decode_modrm(cur, pfx)?;
            let rm = rm_operand(&m.rm, Width::W8, pfx);
            match m.reg & 7 {
                0 => (Opcode::Inc, vec![rm], seq),
                1 => (Opcode::Dec, vec![rm], seq),
                _ => return Err(unknown_opcode(&[op])),
            }
        }
        // Group 5: inc/dec/call/jmp/push r/m.
        0xFF => {
            let m = decode_modrm(cur, pfx)?;
            match m.reg & 7 {
                0 => (Opcode::Inc, vec![rm_operand(&m.rm, vw, pfx)], seq),
                1 => (Opcode::Dec, vec![rm_operand(&m.rm, vw, pfx)], seq),
                2 => (
                    Opcode::Call,
                    vec![rm_operand(&m.rm, Width::W64, pfx)],
                    FlowKind::IndirectCall,
                ),
                4 => (
                    Opcode::Jmp,
                    vec![rm_operand(&m.rm, Width::W64, pfx)],
                    FlowKind::IndirectJump,
                ),
                6 => (Opcode::Push, vec![rm_operand(&m.rm, sw, pfx)], seq),
                // /3 and /5 are far call/jmp; /7 is undefined.
                _ => return Err(unknown_opcode(&[op])),
            }
        }
        // Two-byte map.
        0x0F => return dispatch_0f(cur, pfx),
        _ => return Err(unknown_opcode(&[op])),
    };
    Ok(out)
}

/// Dispatch on a two-byte-map (`0F xx`) opcode.
fn dispatch_0f(cur: &mut Cursor, pfx: &Prefixes) -> Result<(Opcode, Vec<Operand>, FlowKind)> {
    let vw = pfx.vwidth();
    let seq = FlowKind::Seq;
    let op = cur.u8()?;

    let out = match op {
        0x05 => (Opcode::Syscall, vec![], FlowKind::Interrupt),
        0x0B => (Opcode::Ud2, vec![], FlowKind::Interrupt),
        // F3 0F 1E FA/FB: endbr64/endbr32 (CET). Other 0F 1E forms are
        // reserved hint space we do not model.
        0x1E if pfx.rep == Some(Rep::Rep) => match cur.u8()? {
            0xFA => (Opcode::Endbr64, vec![], seq),
            0xFB => (Opcode::Endbr32, vec![], seq),
            _ => return Err(unknown_opcode(&[0x0F, op])),
        },
        // Multi-byte nop: 0F 1F /r.
        0x1F => {
            let m = decode_modrm(cur, pfx)?;
            (Opcode::Nop, vec![rm_operand(&m.rm, vw, pfx)], seq)
        }
        0x31 => (Opcode::Rdtsc, vec![], seq),
        // cmovcc Gv, Ev.
        0x40..=0x4F => {
            let m = decode_modrm(cur, pfx)?;
            (
                Opcode::Cmov(Cond::from_nibble(op)),
                vec![reg_operand(m.reg, vw, pfx), rm_operand(&m.rm, vw, pfx)],
                seq,
            )
        }
        // jcc rel32.
        0x80..=0x8F => {
            let rel = cur.i32()?;
            (
                Opcode::Jcc(Cond::from_nibble(op)),
                vec![Operand::Imm(rel)],
                FlowKind::RelCond(rel),
            )
        }
        // setcc r/m8 (the reg field is ignored).
        0x90..=0x9F => {
            let m = decode_modrm(cur, pfx)?;
            (
                Opcode::Setcc(Cond::from_nibble(op)),
                vec![rm_operand(&m.rm, Width::W8, pfx)],
                seq,
            )
        }
        0xA2 => (Opcode::Cpuid, vec![], seq),
        // bt/bts/btr/btc Ev, Gv.
        0xA3 | 0xAB | 0xB3 | 0xBB => {
            let m = decode_modrm(cur, pfx)?;
            let mnem = match op {
                0xA3 => Opcode::Bt,
                0xAB => Opcode::Bts,
                0xB3 => Opcode::Btr,
                _ => Opcode::Btc,
            };
            (
                mnem,
                vec![rm_operand(&m.rm, vw, pfx), reg_operand(m.reg, vw, pfx)],
                seq,
            )
        }
        // imul Gv, Ev.
        0xAF => {
            let m = decode_modrm(cur, pfx)?;
            (
                Opcode::Imul,
                vec![reg_operand(m.reg, vw, pfx), rm_operand(&m.rm, vw, pfx)],
                seq,
            )
        }
        // cmpxchg Eb,Gb / Ev,Gv.
        0xB0 | 0xB1 => {
            let m = decode_modrm(cur, pfx)?;
            let w = if op == 0xB0 { Width::W8 } else { vw };
            (
                Opcode::Cmpxchg,
                vec![rm_operand(&m.rm, w, pfx), reg_operand(m.reg, w, pfx)],
                seq,
            )
        }
        // movzx/movsx Gv, Eb/Ew.
        0xB6 | 0xB7 | 0xBE | 0xBF => {
            let m = decode_modrm(cur, pfx)?;
            let mnem = if op & 0x08 == 0 {
                Opcode::Movzx
            } else {
                Opcode::Movsx
            };
            let sw8 = if op & 1 == 0 { Width::W8 } else { Width::W16 };
            (
                mnem,
                vec![reg_operand(m.reg, vw, pfx), rm_operand(&m.rm, sw8, pfx)],
                seq,
            )
        }
        // Group 8: bt/bts/btr/btc Ev, ib.
        0xBA => {
            let m = decode_modrm(cur, pfx)?;
            let mnem = match m.reg & 7 {
                4 => Opcode::Bt,
                5 => Opcode::Bts,
                6 => Opcode::Btr,
                7 => Opcode::Btc,
                _ => return Err(unknown_opcode(&[0x0F, op])),
            };
            let rm = rm_operand(&m.rm, vw, pfx);
            (mnem, vec![rm, Operand::Imm(cur.i8()?)], seq)
        }
        // xadd Eb,Gb / Ev,Gv.
        0xC0 | 0xC1 => {
            let m = decode_modrm(cur, pfx)?;
            let w = if op == 0xC0 { Width::W8 } else { vw };
            (
                Opcode::Xadd,
                vec![rm_operand(&m.rm, w, pfx), reg_operand(m.reg, w, pfx)],
                seq,
            )
        }
        // The SSE/SSE2 subset (all `0F xx /r`, mandatory-prefix selected).
        0x10 | 0x11 | 0x28 | 0x29 | 0x2E | 0x2F | 0x54 | 0x57 | 0x58 | 0x59 | 0x5C | 0x5E
        | 0x2A | 0x2C | 0x2D | 0x6E | 0x7E | 0xD6 => return decode_sse(cur, pfx, op),
        _ => return Err(unknown_opcode(&[0x0F, op])),
    };
    Ok(out)
}

/// Decode the SSE/SSE2 subset [`Opcode::Sse`] covers. All are `0F xx /r`;
/// the mandatory `66`/`F3`/`F2` prefix selects the packed/scalar,
/// single/double variant. Modeling these — even coarsely — is what keeps
/// CFG recovery, the lifter, and devirt from stopping at the first
/// floating-point instruction, which on real x86-64 code truncates most
/// functions. Operand types are assigned so the destination is always
/// first (Intel order): a GPR destination (the `cvt*2si` converts and the
/// `movd`/`movq` stores) is a [`Operand::Reg`] so downstream def-analysis
/// sees the write; every other operand register is an [`Operand::Xmm`].
/// A `(prefix, opcode)` pair outside the subset — most notably the MMX
/// forms of `6E`/`7E`/`D6` and the converts without their `F3`/`F2`
/// prefix — is refused rather than guessed.
fn decode_sse(cur: &mut Cursor, pfx: &Prefixes, op: u8) -> Result<(Opcode, Vec<Operand>, FlowKind)> {
    let seq = FlowKind::Seq;
    let p66 = pfx.opsize;
    let f3 = pfx.rep == Some(Rep::Rep);
    let f2 = pfx.rep == Some(Rep::Repne);
    // GPR width for the converts and GPR<->XMM moves: REX.W selects 64-bit.
    let gw = if pfx.rex_w() { Width::W64 } else { Width::W32 };
    let m = decode_modrm(cur, pfx)?;
    let xreg = Operand::Xmm(m.reg);
    // The r/m half as XMM-or-memory (mod=11 is an XMM register here).
    let xrm = match m.rm {
        Rm::Reg(n) => Operand::Xmm(n),
        Rm::Mem { .. } => rm_operand(&m.rm, Width::W64, pfx),
    };
    let refuse = || Err(unknown_opcode(&[0x0F, op]));
    let sse = |mnem: &'static str, writes_flags: bool, dst: Operand, src: Operand| {
        Ok((Opcode::Sse { mnem, writes_flags }, vec![dst, src], seq))
    };

    match op {
        // movups/movupd/movss/movsd: 10 = load (reg<-rm), 11 = store (rm<-reg).
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
                sse(mnem, false, xreg, xrm)
            } else {
                sse(mnem, false, xrm, xreg)
            }
        }
        // movaps/movapd: 28 = load, 29 = store.
        0x28 | 0x29 => {
            let mnem = if p66 { "movapd" } else { "movaps" };
            if op == 0x28 {
                sse(mnem, false, xreg, xrm)
            } else {
                sse(mnem, false, xrm, xreg)
            }
        }
        // ucomiss/ucomisd/comiss/comisd — the only SSE members writing EFLAGS.
        0x2E | 0x2F => {
            let mnem = match (op, p66) {
                (0x2E, false) => "ucomiss",
                (0x2E, _) => "ucomisd",
                (0x2F, false) => "comiss",
                (_, _) => "comisd",
            };
            sse(mnem, true, xreg, xrm)
        }
        0x54 => sse(if p66 { "andpd" } else { "andps" }, false, xreg, xrm),
        0x57 => sse(if p66 { "xorpd" } else { "xorps" }, false, xreg, xrm),
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
            sse(mnem, false, xreg, xrm)
        }
        // cvtsi2ss/cvtsi2sd xmm, r/m(gpr) — requires F3/F2.
        0x2A => {
            if f3 {
                sse("cvtsi2ss", false, xreg, rm_operand(&m.rm, gw, pfx))
            } else if f2 {
                sse("cvtsi2sd", false, xreg, rm_operand(&m.rm, gw, pfx))
            } else {
                refuse()
            }
        }
        // cvt(t)ss2si/sd2si r(gpr), xmm/mem — requires F3/F2.
        0x2C | 0x2D => {
            let mnem = match (op, f3, f2) {
                (0x2C, true, _) => "cvttss2si",
                (0x2C, _, true) => "cvttsd2si",
                (0x2D, true, _) => "cvtss2si",
                (0x2D, _, true) => "cvtsd2si",
                _ => return refuse(),
            };
            sse(mnem, false, reg_operand(m.reg, gw, pfx), xrm)
        }
        // 66 0F 6E: movd/movq xmm, r/m(gpr) (REX.W -> movq). No-66 is MMX.
        0x6E => {
            if !p66 {
                return refuse();
            }
            let mnem = if pfx.rex_w() { "movq" } else { "movd" };
            sse(mnem, false, xreg, rm_operand(&m.rm, gw, pfx))
        }
        // F3 0F 7E: movq xmm, xmm/m64. 66 0F 7E: movd/movq r/m(gpr), xmm.
        0x7E => {
            if f3 {
                sse("movq", false, xreg, xrm)
            } else if p66 {
                let mnem = if pfx.rex_w() { "movq" } else { "movd" };
                sse(mnem, false, rm_operand(&m.rm, gw, pfx), xreg)
            } else {
                refuse()
            }
        }
        // 66 0F D6: movq xmm/m64, xmm (store).
        0xD6 => {
            if !p66 {
                return refuse();
            }
            sse("movq", false, xrm, xreg)
        }
        _ => refuse(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAX: u8 = 0;
    const RCX: u8 = 1;
    const RDX: u8 = 2;
    const RBX: u8 = 3;
    const RSP: u8 = 4;
    const RBP: u8 = 5;
    const RSI: u8 = 6;
    const RDI: u8 = 7;

    /// Decode at a fixed VA, asserting success.
    fn d(bytes: &[u8]) -> Instruction {
        decode(bytes, 0x1000).unwrap_or_else(|e| panic!("{bytes:02x?}: {e}"))
    }

    fn r(num: u8, width: Width) -> Operand {
        Operand::Reg(Reg::gpr(num, width))
    }

    fn r64(n: u8) -> Operand {
        r(n, Width::W64)
    }

    fn r32(n: u8) -> Operand {
        r(n, Width::W32)
    }

    fn r16(n: u8) -> Operand {
        r(n, Width::W16)
    }

    fn r8(n: u8) -> Operand {
        r(n, Width::W8)
    }

    fn mem(base: Option<u8>, index: Option<u8>, scale: u8, disp: i64) -> Operand {
        Operand::Mem {
            base: base.map(|n| Reg::gpr(n, Width::W64)),
            index: index.map(|n| Reg::gpr(n, Width::W64)),
            scale,
            disp,
            rip_relative: false,
        }
    }

    fn rip(disp: i64) -> Operand {
        Operand::Mem {
            base: None,
            index: None,
            scale: 1,
            disp,
            rip_relative: true,
        }
    }

    fn imm(v: i64) -> Operand {
        Operand::Imm(v)
    }

    /// Assert mnemonic, full-length consumption, and operands.
    fn check(bytes: &[u8], opcode: Opcode, operands: &[Operand]) {
        let ins = d(bytes);
        assert_eq!(ins.opcode, opcode, "opcode of {bytes:02x?}");
        assert_eq!(ins.length as usize, bytes.len(), "length of {bytes:02x?}");
        assert_eq!(ins.operands, operands, "operands of {bytes:02x?}");
    }

    // ---- moves, stack ops, and addressing forms ----

    #[test]
    fn push_pop_registers() {
        check(&[0x55], Opcode::Push, &[r64(RBP)]);
        check(&[0x5D], Opcode::Pop, &[r64(RBP)]);
        check(&[0x41, 0x55], Opcode::Push, &[r64(13)]);
        check(&[0x41, 0x5D], Opcode::Pop, &[r64(13)]);
        check(&[0x66, 0x50], Opcode::Push, &[r16(RAX)]);
        assert_eq!(d(&[0x55]).flow(), Flow::Sequential);
    }

    #[test]
    fn push_imm_mem_and_pop_rm() {
        check(&[0x68, 0x78, 0x56, 0x34, 0x12], Opcode::Push, &[imm(0x12345678)]);
        check(&[0x6A, 0xFF], Opcode::Push, &[imm(-1)]);
        // push qword [rsp+8]: FF /6, SIB with no index, rsp base.
        check(
            &[0xFF, 0x74, 0x24, 0x08],
            Opcode::Push,
            &[mem(Some(RSP), None, 1, 8)],
        );
        // pop r/m64 via 8F /0.
        check(&[0x8F, 0xC0], Opcode::Pop, &[r64(RAX)]);
    }

    #[test]
    fn mov_reg_reg_with_rex() {
        check(&[0x89, 0xE5], Opcode::Mov, &[r32(RBP), r32(RSP)]);
        check(&[0x48, 0x89, 0xE5], Opcode::Mov, &[r64(RBP), r64(RSP)]);
        // REX.W + REX.R: mov rdx, r13.
        check(&[0x4C, 0x89, 0xEA], Opcode::Mov, &[r64(RDX), r64(13)]);
        // mov Gv, Ev direction.
        check(&[0x48, 0x8B, 0xC1], Opcode::Mov, &[r64(RAX), r64(RCX)]);
        // Byte forms.
        check(&[0x88, 0xC8], Opcode::Mov, &[r8(RAX), r8(RCX)]);
        check(&[0x8A, 0xC8], Opcode::Mov, &[r8(RCX), r8(RAX)]);
    }

    #[test]
    fn mov_rip_relative() {
        check(
            &[0x48, 0x8B, 0x05, 0x78, 0x56, 0x34, 0x12],
            Opcode::Mov,
            &[r64(RAX), rip(0x12345678)],
        );
        // Negative displacement.
        check(
            &[0x8B, 0x0D, 0xFC, 0xFF, 0xFF, 0xFF],
            Opcode::Mov,
            &[r32(RCX), rip(-4)],
        );
    }

    #[test]
    fn mov_immediate_forms() {
        check(&[0xB0, 0x7F], Opcode::Mov, &[r8(RAX), imm(0x7F)]);
        check(
            &[0xB8, 0x44, 0x33, 0x22, 0x11],
            Opcode::Mov,
            &[r32(RAX), imm(0x11223344)],
        );
        check(&[0x66, 0xB8, 0x34, 0x12], Opcode::Mov, &[r16(RAX), imm(0x1234)]);
        // The only imm64 form: REX.W B8+r.
        check(
            &[0x48, 0xB8, 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01],
            Opcode::Mov,
            &[r64(RAX), imm(0x0123_4567_89AB_CDEF)],
        );
        // REX.B reaches the +r register: mov r9d, 1.
        check(
            &[0x41, 0xB9, 0x01, 0x00, 0x00, 0x00],
            Opcode::Mov,
            &[r32(9), imm(1)],
        );
        // C6/C7 /0: mov r/m, imm.
        check(
            &[0xC6, 0x00, 0xFF],
            Opcode::Mov,
            &[mem(Some(RAX), None, 1, 0), imm(-1)],
        );
        check(
            &[0xC7, 0x05, 0x00, 0x00, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00],
            Opcode::Mov,
            &[rip(0), imm(42)],
        );
    }

    #[test]
    fn byte_registers_honor_rex_and_high_byte_aliases() {
        // Without REX, encoding 4-7 selects ah/ch/dh/bh.
        let ins = d(&[0xB4, 0x01]); // mov ah, 1
        assert_eq!(
            ins.operands[0],
            Operand::Reg(Reg {
                num: 0,
                width: Width::W8,
                high_byte: true
            })
        );
        // With any REX, the same encoding selects spl.
        let ins = d(&[0x40, 0xB4, 0x01]); // mov spl, 1
        assert_eq!(ins.operands[0], r8(RSP));
        // Register names.
        assert_eq!(Reg::gpr(0, Width::W64).name(), "rax");
        assert_eq!(Reg::gpr(12, Width::W32).name(), "r12d");
        assert_eq!(Reg::gpr(12, Width::W8).name(), "r12b");
        assert_eq!(Reg::byte_reg(5, false).name(), "ch");
        assert_eq!(Reg::byte_reg(5, true).name(), "bpl");
    }

    #[test]
    fn lea_forms() {
        check(
            &[0x48, 0x8D, 0x3D, 0xD2, 0x04, 0x00, 0x00],
            Opcode::Lea,
            &[r64(RDI), rip(0x4D2)],
        );
        check(
            &[0x48, 0x8D, 0x44, 0x0A, 0x08],
            Opcode::Lea,
            &[r64(RAX), mem(Some(RDX), Some(RCX), 1, 8)],
        );
        // lea with a register r/m (mod=11) is invalid.
        assert!(decode(&[0x8D, 0xC0], 0).is_err());
    }

    #[test]
    fn sib_all_scales() {
        check(
            &[0x8B, 0x04, 0x18],
            Opcode::Mov,
            &[r32(RAX), mem(Some(RAX), Some(RBX), 1, 0)],
        );
        check(
            &[0x8B, 0x44, 0x58, 0x08],
            Opcode::Mov,
            &[r32(RAX), mem(Some(RAX), Some(RBX), 2, 8)],
        );
        check(
            &[0x8B, 0x84, 0x98, 0x00, 0x01, 0x00, 0x00],
            Opcode::Mov,
            &[r32(RAX), mem(Some(RAX), Some(RBX), 4, 0x100)],
        );
        check(
            &[0x8B, 0x04, 0xD8],
            Opcode::Mov,
            &[r32(RAX), mem(Some(RAX), Some(RBX), 8, 0)],
        );
    }

    #[test]
    fn sib_special_cases() {
        // Base field 101 with mod=00: disp32, no base.
        check(
            &[0x8B, 0x04, 0x8D, 0x44, 0x33, 0x22, 0x11],
            Opcode::Mov,
            &[r32(RAX), mem(None, Some(RCX), 4, 0x11223344)],
        );
        // ... and REX.B does not defeat the no-base escape.
        check(
            &[0x41, 0x8B, 0x04, 0x8D, 0x10, 0x00, 0x00, 0x00],
            Opcode::Mov,
            &[r32(RAX), mem(None, Some(RCX), 4, 0x10)],
        );
        // Index field 100 without REX.X: no index ([rsp]).
        check(
            &[0x8B, 0x04, 0x24],
            Opcode::Mov,
            &[r32(RAX), mem(Some(RSP), None, 1, 0)],
        );
        // Index field 100 *with* REX.X: r12 is a valid index.
        check(
            &[0x42, 0x8B, 0x04, 0xA0],
            Opcode::Mov,
            &[r32(RAX), mem(Some(RAX), Some(12), 4, 0)],
        );
    }

    #[test]
    fn disp8_and_disp32_bases() {
        check(
            &[0x8B, 0x45, 0xF8],
            Opcode::Mov,
            &[r32(RAX), mem(Some(RBP), None, 1, -8)],
        );
        check(
            &[0x8B, 0x80, 0x00, 0x02, 0x00, 0x00],
            Opcode::Mov,
            &[r32(RAX), mem(Some(RAX), None, 1, 0x200)],
        );
        // r13 as a base needs the same mod=01 disp8 escape rbp does.
        check(
            &[0x41, 0x8B, 0x45, 0x00],
            Opcode::Mov,
            &[r32(RAX), mem(Some(13), None, 1, 0)],
        );
    }

    // ---- arithmetic, logic, and the immediate groups ----

    #[test]
    fn arith_group_all_six_encodings() {
        check(&[0x00, 0xD8], Opcode::Add, &[r8(RAX), r8(RBX)]);
        check(&[0x01, 0xD8], Opcode::Add, &[r32(RAX), r32(RBX)]);
        check(&[0x02, 0xD8], Opcode::Add, &[r8(RBX), r8(RAX)]);
        check(
            &[0x03, 0x05, 0x10, 0x00, 0x00, 0x00],
            Opcode::Add,
            &[r32(RAX), rip(0x10)],
        );
        check(&[0x04, 0x05], Opcode::Add, &[r8(RAX), imm(5)]);
        check(
            &[0x05, 0x78, 0x56, 0x34, 0x12],
            Opcode::Add,
            &[r32(RAX), imm(0x12345678)],
        );
        // The same pattern selects the other seven group members.
        check(&[0x29, 0xD8], Opcode::Sub, &[r32(RAX), r32(RBX)]);
        check(&[0x31, 0xC0], Opcode::Xor, &[r32(RAX), r32(RAX)]);
        check(&[0x45, 0x31, 0xC0], Opcode::Xor, &[r32(8), r32(8)]);
        check(&[0x21, 0xF7], Opcode::And, &[r32(RDI), r32(RSI)]);
        check(&[0x09, 0xC8], Opcode::Or, &[r32(RAX), r32(RCX)]);
        check(&[0x11, 0xC8], Opcode::Adc, &[r32(RAX), r32(RCX)]);
        check(&[0x19, 0xC8], Opcode::Sbb, &[r32(RAX), r32(RCX)]);
        check(&[0x39, 0xC8], Opcode::Cmp, &[r32(RAX), r32(RCX)]);
        check(&[0x3C, 0xFF], Opcode::Cmp, &[r8(RAX), imm(-1)]);
        check(
            &[0x48, 0x3D, 0x00, 0x10, 0x00, 0x00],
            Opcode::Cmp,
            &[r64(RAX), imm(0x1000)],
        );
        // 66-prefixed iz immediate is 16 bits.
        check(&[0x66, 0x05, 0x34, 0x12], Opcode::Add, &[r16(RAX), imm(0x1234)]);
    }

    #[test]
    fn group1_immediates() {
        check(&[0x83, 0xC0, 0x01], Opcode::Add, &[r32(RAX), imm(1)]);
        // 83 sign-extends its imm8: and rsp, -16.
        check(&[0x48, 0x83, 0xE4, 0xF0], Opcode::And, &[r64(RSP), imm(-16)]);
        check(
            &[0x48, 0x81, 0xEC, 0x00, 0x01, 0x00, 0x00],
            Opcode::Sub,
            &[r64(RSP), imm(0x100)],
        );
        // 80 /7: cmp byte [rip+disp], imm8.
        check(
            &[0x80, 0x3D, 0x0A, 0x00, 0x00, 0x00, 0x00],
            Opcode::Cmp,
            &[rip(0x0A), imm(0)],
        );
    }

    #[test]
    fn test_forms() {
        check(&[0x85, 0xC0], Opcode::Test, &[r32(RAX), r32(RAX)]);
        check(&[0x48, 0x85, 0xFF], Opcode::Test, &[r64(RDI), r64(RDI)]);
        check(&[0x84, 0xC9], Opcode::Test, &[r8(RCX), r8(RCX)]);
        check(&[0xA8, 0x01], Opcode::Test, &[r8(RAX), imm(1)]);
        check(
            &[0xA9, 0x00, 0x00, 0x00, 0x80],
            Opcode::Test,
            &[r32(RAX), imm(-0x8000_0000)],
        );
        check(&[0xF6, 0xC1, 0x20], Opcode::Test, &[r8(RCX), imm(0x20)]);
        check(
            &[0x48, 0xF7, 0xC7, 0x44, 0x33, 0x22, 0x11],
            Opcode::Test,
            &[r64(RDI), imm(0x11223344)],
        );
    }

    #[test]
    fn group3_unary_ops() {
        check(&[0xF7, 0xD0], Opcode::Not, &[r32(RAX)]);
        check(&[0x48, 0xF7, 0xD8], Opcode::Neg, &[r64(RAX)]);
        check(&[0x48, 0xF7, 0xE1], Opcode::Mul, &[r64(RCX)]);
        check(&[0x48, 0xF7, 0xEB], Opcode::Imul, &[r64(RBX)]);
        check(&[0xF7, 0xF6], Opcode::Div, &[r32(RSI)]);
        check(&[0x48, 0xF7, 0xF9], Opcode::Idiv, &[r64(RCX)]);
        check(&[0xF6, 0xD8], Opcode::Neg, &[r8(RAX)]);
    }

    #[test]
    fn inc_dec_forms() {
        check(&[0xFE, 0xC0], Opcode::Inc, &[r8(RAX)]);
        check(&[0xFE, 0xC9], Opcode::Dec, &[r8(RCX)]);
        check(&[0xFF, 0xC0], Opcode::Inc, &[r32(RAX)]);
        check(&[0x48, 0xFF, 0xC9], Opcode::Dec, &[r64(RCX)]);
        check(
            &[0xFF, 0x45, 0x00],
            Opcode::Inc,
            &[mem(Some(RBP), None, 1, 0)],
        );
    }

    #[test]
    fn xchg_and_nop_family() {
        check(&[0x90], Opcode::Nop, &[]);
        check(&[0x66, 0x90], Opcode::Nop, &[]);
        // REX.B turns 90 into xchg rax, r8.
        check(&[0x49, 0x90], Opcode::Xchg, &[r64(RAX), r64(8)]);
        check(&[0x48, 0x91], Opcode::Xchg, &[r64(RAX), r64(RCX)]);
        check(&[0x86, 0xD8], Opcode::Xchg, &[r8(RAX), r8(RBX)]);
        check(&[0x87, 0xD8], Opcode::Xchg, &[r32(RAX), r32(RBX)]);
    }

    #[test]
    fn convert_and_leave() {
        check(&[0x98], Opcode::Cwde, &[]);
        check(&[0x48, 0x98], Opcode::Cwde, &[]); // cdqe
        check(&[0x99], Opcode::Cdq, &[]);
        check(&[0x48, 0x99], Opcode::Cdq, &[]); // cqo
        check(&[0xC9], Opcode::Leave, &[]);
    }

    #[test]
    fn extend_moves() {
        check(&[0x0F, 0xB6, 0xC0], Opcode::Movzx, &[r32(RAX), r8(RAX)]);
        check(&[0x0F, 0xB7, 0xC0], Opcode::Movzx, &[r32(RAX), r16(RAX)]);
        check(&[0x48, 0x0F, 0xBE, 0xC3], Opcode::Movsx, &[r64(RAX), r8(RBX)]);
        check(
            &[0x0F, 0xBF, 0x4D, 0xF6],
            Opcode::Movsx,
            &[r32(RCX), mem(Some(RBP), None, 1, -10)],
        );
        check(&[0x48, 0x63, 0xC8], Opcode::Movsxd, &[r64(RCX), r32(RAX)]);
        // Without REX.W, movsxd degenerates to a 32-bit move.
        check(&[0x63, 0xC8], Opcode::Movsxd, &[r32(RCX), r32(RAX)]);
    }

    #[test]
    fn imul_two_and_three_operand() {
        check(&[0x0F, 0xAF, 0xC3], Opcode::Imul, &[r32(RAX), r32(RBX)]);
        check(
            &[0x48, 0x6B, 0xC0, 0x0A],
            Opcode::Imul,
            &[r64(RAX), r64(RAX), imm(10)],
        );
        check(
            &[0x69, 0xC0, 0xE8, 0x03, 0x00, 0x00],
            Opcode::Imul,
            &[r32(RAX), r32(RAX), imm(1000)],
        );
    }

    #[test]
    fn cmov_and_setcc() {
        check(
            &[0x48, 0x0F, 0x44, 0xC1],
            Opcode::Cmov(Cond::E),
            &[r64(RAX), r64(RCX)],
        );
        check(&[0x0F, 0x4F, 0xCA], Opcode::Cmov(Cond::G), &[r32(RCX), r32(RDX)]);
        check(&[0x0F, 0x94, 0xC0], Opcode::Setcc(Cond::E), &[r8(RAX)]);
        check(&[0x0F, 0x95, 0xC1], Opcode::Setcc(Cond::Ne), &[r8(RCX)]);
        check(&[0x41, 0x0F, 0x93, 0xC4], Opcode::Setcc(Cond::Ae), &[r8(12)]);
    }

    #[test]
    fn bit_test_family() {
        check(&[0x0F, 0xA3, 0xD0], Opcode::Bt, &[r32(RAX), r32(RDX)]);
        check(&[0x48, 0x0F, 0xAB, 0xC8], Opcode::Bts, &[r64(RAX), r64(RCX)]);
        check(&[0x0F, 0xB3, 0xC8], Opcode::Btr, &[r32(RAX), r32(RCX)]);
        check(&[0x0F, 0xBB, 0xC8], Opcode::Btc, &[r32(RAX), r32(RCX)]);
        check(&[0x0F, 0xBA, 0xE0, 0x07], Opcode::Bt, &[r32(RAX), imm(7)]);
        check(&[0x0F, 0xBA, 0xF8, 0x07], Opcode::Btc, &[r32(RAX), imm(7)]);
        // Group 8 /0../3 are undefined.
        assert!(decode(&[0x0F, 0xBA, 0xC0, 0x07], 0).is_err());
    }

    #[test]
    fn cmpxchg_xadd_and_lock() {
        let ins = d(&[0xF0, 0x0F, 0xB1, 0x0F]); // lock cmpxchg [rdi], ecx
        assert_eq!(ins.opcode, Opcode::Cmpxchg);
        assert!(ins.lock);
        assert_eq!(ins.operands, &[mem(Some(RDI), None, 1, 0), r32(RCX)]);

        check(&[0x0F, 0xB0, 0xD9], Opcode::Cmpxchg, &[r8(RCX), r8(RBX)]);
        check(&[0x0F, 0xC1, 0xD8], Opcode::Xadd, &[r32(RAX), r32(RBX)]);
        let ins = d(&[0xF0, 0x48, 0x0F, 0xC1, 0x03]); // lock xadd [rbx], rax
        assert_eq!(ins.opcode, Opcode::Xadd);
        assert!(ins.lock);
        assert_eq!(ins.operands, &[mem(Some(RBX), None, 1, 0), r64(RAX)]);
    }

    #[test]
    fn multi_byte_nops_and_endbr() {
        check(&[0x0F, 0x1F, 0x00], Opcode::Nop, &[mem(Some(RAX), None, 1, 0)]);
        assert_eq!(d(&[0x0F, 0x1F, 0x40, 0x00]).length, 4);
        assert_eq!(
            d(&[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]).length,
            8
        );
        assert_eq!(
            d(&[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00]).opcode,
            Opcode::Nop
        );
        check(&[0xF3, 0x0F, 0x1E, 0xFA], Opcode::Endbr64, &[]);
        check(&[0xF3, 0x0F, 0x1E, 0xFB], Opcode::Endbr32, &[]);
        // 0F 1E without the F3 prefix is unmodeled hint space.
        assert!(decode(&[0x0F, 0x1E, 0xFA], 0).is_err());
        assert!(decode(&[0xF3, 0x0F, 0x1E, 0xC0], 0).is_err());
    }

    #[test]
    fn misc_system_ops() {
        check(&[0x0F, 0xA2], Opcode::Cpuid, &[]);
        check(&[0x0F, 0x31], Opcode::Rdtsc, &[]);
        assert_eq!(d(&[0x0F, 0xA2]).flow(), Flow::Sequential);
    }

    // ---- control flow: the primary deliverable ----

    #[test]
    fn call_rel32_flow() {
        let ins = decode(&[0xE8, 0x05, 0x00, 0x00, 0x00], 0x1000).unwrap();
        assert_eq!(ins.opcode, Opcode::Call);
        assert_eq!(ins.length, 5);
        assert_eq!(ins.flow(), Flow::Call(0x100A));
        // Negative displacement: call to the instruction's own address.
        let ins = decode(&[0xE8, 0xFB, 0xFF, 0xFF, 0xFF], 0x1000).unwrap();
        assert_eq!(ins.flow(), Flow::Call(0x1000));
    }

    #[test]
    fn jmp_rel_flow() {
        let ins = decode(&[0xE9, 0x00, 0x01, 0x00, 0x00], 0x2000).unwrap();
        assert_eq!(ins.flow(), Flow::Jump(0x2105));
        // jmp rel8 self-loop.
        let ins = decode(&[0xEB, 0xFE], 0x4000).unwrap();
        assert_eq!(ins.opcode, Opcode::Jmp);
        assert_eq!(ins.flow(), Flow::Jump(0x4000));
        // Target arithmetic wraps rather than panicking.
        let ins = decode(&[0xEB, 0x80], 0).unwrap();
        assert_eq!(ins.flow(), Flow::Jump((2i64 - 128) as u64));
    }

    #[test]
    fn jcc_rel8_flow_and_conditions() {
        let ins = decode(&[0x75, 0x10], 0x1000).unwrap();
        assert_eq!(ins.opcode, Opcode::Jcc(Cond::Ne));
        assert_eq!(ins.flow(), Flow::CondJump(0x1012));
        assert_eq!(d(&[0x70, 0x00]).opcode, Opcode::Jcc(Cond::O));
        assert_eq!(d(&[0x74, 0x00]).opcode, Opcode::Jcc(Cond::E));
        assert_eq!(d(&[0x7C, 0x00]).opcode, Opcode::Jcc(Cond::L));
        assert_eq!(d(&[0x7F, 0x00]).opcode, Opcode::Jcc(Cond::G));
    }

    #[test]
    fn jcc_rel32_flow() {
        let ins = decode(&[0x0F, 0x84, 0x00, 0x01, 0x00, 0x00], 0).unwrap();
        assert_eq!(ins.opcode, Opcode::Jcc(Cond::E));
        assert_eq!(ins.length, 6);
        assert_eq!(ins.flow(), Flow::CondJump(0x106));
        let ins = decode(&[0x0F, 0x8F, 0xFA, 0xFF, 0xFF, 0xFF], 0x1000).unwrap();
        assert_eq!(ins.opcode, Opcode::Jcc(Cond::G));
        assert_eq!(ins.flow(), Flow::CondJump(0x1000));
    }

    #[test]
    fn indirect_call_and_jmp() {
        // ff /2: call r/m64.
        let ins = d(&[0xFF, 0xD0]);
        assert_eq!(ins.opcode, Opcode::Call);
        assert_eq!(ins.flow(), Flow::IndirectCall);
        assert_eq!(ins.operands, &[r64(RAX)]);
        assert_eq!(d(&[0x41, 0xFF, 0xD5]).operands, &[r64(13)]);
        assert_eq!(d(&[0x41, 0xFF, 0xD5]).flow(), Flow::IndirectCall);
        // call [rip+disp32]: the PLT/GOT shape.
        let ins = d(&[0xFF, 0x15, 0x10, 0x00, 0x00, 0x00]);
        assert_eq!(ins.flow(), Flow::IndirectCall);
        assert_eq!(ins.operands, &[rip(0x10)]);
        // ff /4: jmp r/m64.
        let ins = d(&[0xFF, 0xE0]);
        assert_eq!(ins.opcode, Opcode::Jmp);
        assert_eq!(ins.flow(), Flow::IndirectJump);
        let ins = d(&[0xFF, 0x25, 0x00, 0x00, 0x00, 0x00]);
        assert_eq!(ins.flow(), Flow::IndirectJump);
        assert_eq!(ins.operands, &[rip(0)]);
        // jmp qword [rsp+8].
        let ins = d(&[0xFF, 0x64, 0x24, 0x08]);
        assert_eq!(ins.flow(), Flow::IndirectJump);
        assert_eq!(ins.operands, &[mem(Some(RSP), None, 1, 8)]);
    }

    #[test]
    fn returns_traps_and_halt() {
        let ins = d(&[0xC3]);
        assert_eq!(ins.opcode, Opcode::Ret);
        assert_eq!(ins.flow(), Flow::Return);
        let ins = d(&[0xC2, 0x08, 0x00]);
        assert_eq!(ins.flow(), Flow::Return);
        assert_eq!(ins.operands, &[imm(8)]);
        // rep ret (AMD branch-predictor idiom) still decodes.
        let ins = d(&[0xF3, 0xC3]);
        assert_eq!(ins.flow(), Flow::Return);
        assert_eq!(ins.rep, Some(Rep::Rep));

        assert_eq!(d(&[0x0F, 0x05]).flow(), Flow::Interrupt); // syscall
        assert_eq!(d(&[0xCC]).flow(), Flow::Interrupt); // int3
        let ins = d(&[0xCD, 0x80]);
        assert_eq!(ins.opcode, Opcode::Int);
        assert_eq!(ins.operands, &[imm(0x80)]);
        assert_eq!(ins.flow(), Flow::Interrupt);
        assert_eq!(d(&[0x0F, 0x0B]).flow(), Flow::Interrupt); // ud2
        assert_eq!(d(&[0xF4]).flow(), Flow::Halt); // hlt
    }

    // ---- prefixes ----

    #[test]
    fn segment_override_fs_canary_load() {
        // mov rax, fs:[0x28] — the glibc stack-protector canary load.
        let ins = d(&[0x64, 0x48, 0x8B, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00]);
        assert_eq!(ins.opcode, Opcode::Mov);
        assert_eq!(ins.segment, Some(Segment::Fs));
        assert_eq!(ins.operands, &[r64(RAX), mem(None, None, 1, 0x28)]);
    }

    #[test]
    fn rex_is_voided_by_a_following_legacy_prefix() {
        // 48 66 89 C8: the REX.W precedes a legacy prefix, so it is
        // ignored and the 66 makes this a 16-bit move.
        check(&[0x48, 0x66, 0x89, 0xC8], Opcode::Mov, &[r16(RAX), r16(RCX)]);
        // In the effective order (66 then REX) the REX wins.
        check(&[0x66, 0x48, 0x89, 0xC8], Opcode::Mov, &[r64(RAX), r64(RCX)]);
    }

    #[test]
    fn length_limit_is_fifteen_bytes() {
        // 14 operand-size prefixes + nop = exactly 15 bytes: legal.
        let mut buf = vec![0x66u8; 14];
        buf.push(0x90);
        let ins = decode(&buf, 0).unwrap();
        assert_eq!(ins.length, 15);
        assert_eq!(ins.opcode, Opcode::Nop);

        // One more prefix pushes the opcode to byte 16: error.
        let mut buf = vec![0x66u8; 15];
        buf.push(0x90);
        assert!(matches!(
            decode(&buf, 0).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        // A prefix wall with no opcode at all is also an error.
        assert!(decode(&[0x66; 32], 0).is_err());
    }

    // ---- robustness ----

    #[test]
    fn unknown_opcodes_are_typed_errors() {
        // Invalid-in-64-bit one-byte opcodes and unmodeled encodings.
        for bytes in [
            &[0x06][..],           // push es (invalid in 64-bit)
            &[0x27],               // daa (invalid)
            &[0x62, 0xC0],         // bound / EVEX space (unmodeled)
            &[0x82, 0xC0, 0x01],   // group-1 alias (invalid in 64-bit)
            &[0x9A],               // far call (unmodeled)
            &[0xEA],               // far jmp (unmodeled)
            &[0xD6],               // salc (invalid)
            &[0xF1],               // int1 (unmodeled)
            &[0x0F, 0xFF, 0xC0],   // ud0 space
            &[0x0F, 0xA4, 0xC0],   // shld (unmodeled)
            &[0xFE, 0xD0],         // group 4 /2 (undefined)
            &[0xFF, 0xF8],         // group 5 /7 (undefined)
            &[0xFF, 0xD8],         // group 5 /3: far call (unmodeled)
            &[0x8F, 0xC8],         // group 1A /1 (undefined)
            &[0xC6, 0xC8, 0x00],   // group 11 /1 (undefined)
        ] {
            assert!(
                matches!(decode(bytes, 0), Err(ParseError::Unsupported(_))),
                "{bytes:02x?} should be Unsupported"
            );
        }
    }

    /// Every flow-relevant and addressing-mode-relevant encoding, used by
    /// the truncation sweep below. Each entry must decode to exactly its
    /// own length.
    const CORPUS: &[&[u8]] = &[
        &[0x55],
        &[0x41, 0x55],
        &[0x48, 0x89, 0xE5],
        &[0x4C, 0x89, 0xEA],
        &[0x48, 0x8B, 0x05, 0x78, 0x56, 0x34, 0x12],
        &[0x48, 0x8D, 0x3D, 0xD2, 0x04, 0x00, 0x00],
        &[0x8B, 0x84, 0x98, 0x00, 0x01, 0x00, 0x00],
        &[0x8B, 0x04, 0x8D, 0x44, 0x33, 0x22, 0x11],
        &[0x8B, 0x45, 0xF8],
        &[0x48, 0xB8, 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01],
        &[0x66, 0xB8, 0x34, 0x12],
        &[0xC7, 0x05, 0x00, 0x00, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00],
        &[0x48, 0x81, 0xEC, 0x00, 0x01, 0x00, 0x00],
        &[0x48, 0x83, 0xE4, 0xF0],
        &[0x48, 0xF7, 0xD8],
        &[0xE8, 0x05, 0x00, 0x00, 0x00],
        &[0xE9, 0x00, 0x01, 0x00, 0x00],
        &[0xEB, 0xFE],
        &[0x75, 0x10],
        &[0x0F, 0x84, 0x00, 0x01, 0x00, 0x00],
        &[0x0F, 0x8F, 0xFA, 0xFF, 0xFF, 0xFF],
        &[0xFF, 0xD0],
        &[0x41, 0xFF, 0xD5],
        &[0xFF, 0x15, 0x10, 0x00, 0x00, 0x00],
        &[0xFF, 0xE0],
        &[0xFF, 0x25, 0x00, 0x00, 0x00, 0x00],
        &[0xFF, 0x64, 0x24, 0x08],
        &[0xC3],
        &[0xC2, 0x08, 0x00],
        &[0xF3, 0xC3],
        &[0x0F, 0x05],
        &[0xCC],
        &[0xCD, 0x80],
        &[0x0F, 0x0B],
        &[0xF4],
        &[0xF3, 0x0F, 0x1E, 0xFA],
        &[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
        &[0x64, 0x48, 0x8B, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00],
        &[0xF0, 0x48, 0x0F, 0xC1, 0x03],
        &[0x48, 0x0F, 0x44, 0xC1],
        &[0x0F, 0x94, 0xC0],
        &[0x0F, 0xBA, 0xE0, 0x07],
        &[0x48, 0x63, 0xC8],
        &[0x69, 0xC0, 0xE8, 0x03, 0x00, 0x00],
    ];

    #[test]
    fn truncation_sweep_never_panics() {
        for ins in CORPUS {
            let full = decode(ins, 0x1000)
                .unwrap_or_else(|e| panic!("corpus entry {ins:02x?} failed: {e}"));
            assert_eq!(full.length as usize, ins.len(), "{ins:02x?}");
            // Every strict prefix must be a typed error, never a panic
            // and never a bogus shorter decode.
            for len in 0..ins.len() {
                assert!(
                    decode(&ins[..len], 0x1000).is_err(),
                    "{ins:02x?} truncated to {len} should error"
                );
            }
        }
    }

    // ---- SSE/SSE2 subset ----

    fn xmm(n: u8) -> Operand {
        Operand::Xmm(n)
    }

    fn sse(mnem: &'static str, writes_flags: bool) -> Opcode {
        Opcode::Sse { mnem, writes_flags }
    }

    #[test]
    fn sse_moves_load_and_store_forms() {
        // movaps xmm1, xmm2 (reg<-rm).
        check(&[0x0F, 0x28, 0xCA], sse("movaps", false), &[xmm(1), xmm(2)]);
        // movapd xmm1, xmm2 (66 selects the double variant).
        check(&[0x66, 0x0F, 0x28, 0xCA], sse("movapd", false), &[xmm(1), xmm(2)]);
        // movsd [rbp - 0x28], xmm0 — the store form (11), rm is the dest.
        check(
            &[0xF2, 0x0F, 0x11, 0x45, 0xD8],
            sse("movsd", false),
            &[mem(Some(RBP), None, 1, -0x28), xmm(0)],
        );
        // movsd xmm0, [rbp - 0x28] — the load form (10).
        check(
            &[0xF2, 0x0F, 0x10, 0x45, 0xD8],
            sse("movsd", false),
            &[xmm(0), mem(Some(RBP), None, 1, -0x28)],
        );
        // movss selected by F3.
        check(&[0xF3, 0x0F, 0x10, 0xC1], sse("movss", false), &[xmm(0), xmm(1)]);
    }

    #[test]
    fn sse_arithmetic_prefix_selects_the_variant() {
        check(&[0xF2, 0x0F, 0x58, 0xD1], sse("addsd", false), &[xmm(2), xmm(1)]);
        check(&[0xF3, 0x0F, 0x59, 0xC1], sse("mulss", false), &[xmm(0), xmm(1)]);
        check(&[0x0F, 0x5C, 0xC1], sse("subps", false), &[xmm(0), xmm(1)]);
        check(&[0x66, 0x0F, 0x5E, 0xC1], sse("divpd", false), &[xmm(0), xmm(1)]);
        check(&[0x0F, 0x57, 0xC1], sse("xorps", false), &[xmm(0), xmm(1)]);
    }

    #[test]
    fn sse_compares_are_flagged_as_flag_writers() {
        // ucomisd / comiss are the EFLAGS-writing members.
        check(&[0x66, 0x0F, 0x2E, 0xC1], sse("ucomisd", true), &[xmm(0), xmm(1)]);
        check(&[0x0F, 0x2F, 0xC1], sse("comiss", true), &[xmm(0), xmm(1)]);
    }

    #[test]
    fn sse_converts_carry_a_gpr_operand_in_the_right_slot() {
        // cvtsi2sd xmm0, ebx — GPR source (32-bit here), XMM destination.
        check(
            &[0xF2, 0x0F, 0x2A, 0xC3],
            sse("cvtsi2sd", false),
            &[xmm(0), r32(RBX)],
        );
        // REX.W makes the GPR source 64-bit.
        check(
            &[0xF2, 0x48, 0x0F, 0x2A, 0xC3],
            sse("cvtsi2sd", false),
            &[xmm(0), r64(RBX)],
        );
        // cvttsd2si eax, xmm0 — GPR *destination* first, so def-analysis sees it.
        check(
            &[0xF2, 0x0F, 0x2C, 0xC0],
            sse("cvttsd2si", false),
            &[r32(RAX), xmm(0)],
        );
    }

    #[test]
    fn movd_movq_between_gpr_and_xmm() {
        // 66 0F 6E: movd xmm0, ecx (GPR->XMM).
        check(&[0x66, 0x0F, 0x6E, 0xC1], sse("movd", false), &[xmm(0), r32(RCX)]);
        // 66 REX.W 0F 6E: movq xmm0, rcx.
        check(
            &[0x66, 0x48, 0x0F, 0x6E, 0xC1],
            sse("movq", false),
            &[xmm(0), r64(RCX)],
        );
        // 66 0F 7E: movd ecx, xmm0 (XMM->GPR, GPR is the destination).
        check(&[0x66, 0x0F, 0x7E, 0xC1], sse("movd", false), &[r32(RCX), xmm(0)]);
        // F3 0F 7E: movq xmm0, xmm1 (XMM<-XMM).
        check(&[0xF3, 0x0F, 0x7E, 0xC1], sse("movq", false), &[xmm(0), xmm(1)]);
    }

    #[test]
    fn mmx_forms_without_the_sse_prefix_are_refused_not_guessed() {
        // 0F 6E / 7E / D6 without 66, and the converts without F3/F2, are
        // MMX (or invalid) forms this subset does not model: an error, never
        // a mislabeled SSE decode.
        for bytes in [
            &[0x0F, 0x6E, 0xC0][..],
            &[0x0F, 0x7E, 0xC0][..],
            &[0x0F, 0xD6, 0xC0][..],
            &[0x0F, 0x2A, 0xC0][..],
            &[0x0F, 0x2C, 0xC0][..],
        ] {
            assert!(decode(bytes, 0x1000).is_err(), "{bytes:02x?} must be refused");
        }
    }

    #[test]
    fn sse_instructions_are_sequential_flow() {
        // The whole point: an SSE op does not end a basic block.
        assert_eq!(d(&[0xF2, 0x0F, 0x58, 0xD1]).flow, Flow::Sequential);
        assert_eq!(d(&[0x0F, 0x28, 0xCA]).flow, Flow::Sequential);
    }

    #[test]
    fn trailing_bytes_are_not_consumed() {
        let ins = decode(&[0x90, 0xFF, 0xFF, 0xFF], 0).unwrap();
        assert_eq!(ins.length, 1);
        let ins = decode(&[0xC3, 0x00, 0x00], 0).unwrap();
        assert_eq!(ins.length, 1);
        assert_eq!(ins.flow(), Flow::Return);
    }

    /// Decode must be total: any result is fine, but it must not panic,
    /// and a successful decode must fit both the input and the 15-byte
    /// architectural limit.
    fn assert_total(bytes: &[u8]) {
        if let Ok(ins) = decode(bytes, 0) {
            assert!(ins.length as usize <= bytes.len(), "{bytes:02x?}");
            assert!(ins.length as usize <= MAX_INSTRUCTION_LEN, "{bytes:02x?}");
        }
    }

    #[test]
    fn pseudo_exhaustive_sweep() {
        let mut buf = [0u8; MAX_INSTRUCTION_LEN];
        for a in 0..=255u8 {
            buf[0] = a;
            // Every one-byte value alone and padded with zeros.
            assert_total(&[a]);
            for b in 0..=255u8 {
                buf[1] = b;
                // Every two-byte sequence, bare and zero-padded to the
                // 15-byte maximum.
                assert_total(&[a, b]);
                assert_total(&buf);
            }
            buf[1] = 0;
        }
    }
}
