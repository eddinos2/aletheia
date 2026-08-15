//! A small register-transfer IR — the substrate dataflow analysis and,
//! eventually, a decompiler are built on.
//!
//! A decoded machine instruction says *what* it is (`add rax, rcx`); it
//! does not say, in any form a analysis can compute over, *what it does*
//! to the machine's state. This module is that second thing: a compact,
//! side-effect-explicit intermediate representation into which each ISA's
//! instructions are *lifted*. One `add` becomes an assignment of a sum
//! plus the flag writes it implies; one `push` becomes a stack-pointer
//! decrement and a store. Everything an instruction does to registers,
//! memory, and flags is spelled out as data, and nothing is left implicit.
//!
//! # The shape of it
//!
//! The IR is an expression tree over three storage kinds, evaluated by a
//! short list of [`Stmt`]s per instruction:
//!
//! - [`Expr`] is a pure, side-effect-free value: a constant, the current
//!   contents of a register, a memory load, or an operator applied to
//!   sub-expressions. Evaluating an [`Expr`] never changes machine state.
//! - [`Stmt`] is the unit of effect: assign an [`Expr`] to a register,
//!   store an [`Expr`] to memory, or branch. A machine instruction lifts
//!   to a `Vec<Stmt>` executed in order.
//! - [`Reg`] names a storage cell: an *architectural* register (mapped
//!   from the ISA), a status *flag*, or a lifter *temporary*. Temporaries
//!   let a lift name an intermediate value once and reuse it — the source
//!   operand of an `xchg`, say — without inventing machine state.
//!
//! Widths are explicit on every value ([`Width`]), because the same
//! register participates at several widths (`al`/`ax`/`eax`/`rax`) and an
//! analysis must not conflate them. An operation's width is the width of
//! its operands; [`UnOp::Extend`] and [`UnOp::Truncate`] are the only
//! places a value changes width, and they say so.
//!
//! # What this module is and is not
//!
//! This is the IR *core*: the types, their construction, a total
//! well-formedness [`check`], and a deterministic [`render`] pretty-printer
//! in a stable textual form. It contains **no ISA knowledge** — lifting
//! `x86`/`aarch64` instructions into it lives in the per-ISA lifter
//! modules that build on this vocabulary, so the semantics of a `sub` are
//! written once per architecture and this file never grows an opcode
//! table. Keeping the two apart is what lets the lifters be fanned out and
//! verified against a fixed contract.
//!
//! # Contract
//!
//! Construction is infallible and allocation-bounded; [`check`] never
//! panics on any tree and reports the first structural fault it finds
//! (a width mismatch, a store of a non-value, a temporary read before it
//! is written within a block). [`render`] is deterministic: the same
//! statements always print the same bytes, so lifted output is itself
//! diffable and testable.

use std::fmt::Write as _;

// ---------------------------------------------------------------------------
// Widths
// ---------------------------------------------------------------------------

/// Bit width of an IR value. The IR models integer and pointer state only;
/// floating-point and vector state are represented, when a lift needs to
/// preserve them at all, as opaque [`Stmt::Intrinsic`] effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Width {
    /// 1 bit — the width of a flag and of a comparison result.
    W1,
    /// 8 bits.
    W8,
    /// 16 bits.
    W16,
    /// 32 bits.
    W32,
    /// 64 bits.
    W64,
}

impl Width {
    /// The width in bits.
    pub fn bits(self) -> u32 {
        match self {
            Width::W1 => 1,
            Width::W8 => 8,
            Width::W16 => 16,
            Width::W32 => 32,
            Width::W64 => 64,
        }
    }

    /// The width in bytes, rounded up (a `W1` occupies one addressable
    /// byte). Memory accesses never use `W1`; [`check`] enforces it.
    pub fn bytes(self) -> u32 {
        match self {
            Width::W1 => 1,
            Width::W8 => 1,
            Width::W16 => 2,
            Width::W32 => 4,
            Width::W64 => 8,
        }
    }

    /// The mask of the low `bits()` bits, or `u64::MAX` at `W64`.
    pub fn mask(self) -> u64 {
        match self {
            Width::W64 => u64::MAX,
            w => (1u64 << w.bits()) - 1,
        }
    }

    /// A short token for [`render`]: `b`, `w`, `d`, `q`, or `i1`.
    pub fn token(self) -> &'static str {
        match self {
            Width::W1 => "i1",
            Width::W8 => "b",
            Width::W16 => "w",
            Width::W32 => "d",
            Width::W64 => "q",
        }
    }
}

// ---------------------------------------------------------------------------
// Storage: registers, flags, temporaries
// ---------------------------------------------------------------------------

/// Which namespace a [`Reg`] lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Space {
    /// An architectural register, identified by an ISA-assigned number.
    /// The IR does not interpret the number; a lifter and its printer hook
    /// agree on what `Arch(0)` means (`rax`, `x0`, …).
    Arch,
    /// A processor status flag (carry, zero, …), by [`Flag`] discriminant.
    Flag,
    /// A lifter temporary, unique within one lifted instruction's
    /// statement list. Temporaries are write-before-read within a block.
    Temp,
}

/// A status flag. The set is the common ALU condition state; an ISA that
/// lacks one simply never writes it, and one with extras models them as
/// [`Stmt::Intrinsic`] effects rather than growing this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Flag {
    /// Carry / unsigned-overflow.
    Carry,
    /// Zero.
    Zero,
    /// Sign / negative.
    Sign,
    /// Signed overflow.
    Overflow,
    /// Parity of the low byte.
    Parity,
}

impl Flag {
    /// A short token for [`render`].
    pub fn token(self) -> &'static str {
        match self {
            Flag::Carry => "CF",
            Flag::Zero => "ZF",
            Flag::Sign => "SF",
            Flag::Overflow => "OF",
            Flag::Parity => "PF",
        }
    }
}

/// A storage cell: an architectural register, a flag, or a temporary,
/// at a definite [`Width`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Reg {
    /// Which namespace `num` is interpreted in.
    pub space: Space,
    /// Identity within `space`. For [`Space::Flag`] this is the [`Flag`]
    /// cast to a number; helpers keep the two consistent.
    pub num: u16,
    /// The width this reference reads or writes. A narrow reference to a
    /// wide architectural register (`eax` of `rax`) is modeled by width
    /// plus the ISA's numbering convention, not by a separate cell.
    pub width: Width,
}

impl Reg {
    /// An architectural register reference.
    pub fn arch(num: u16, width: Width) -> Reg {
        Reg {
            space: Space::Arch,
            num,
            width,
        }
    }

    /// A temporary reference.
    pub fn temp(num: u16, width: Width) -> Reg {
        Reg {
            space: Space::Temp,
            num,
            width,
        }
    }

    /// A flag cell (always [`Width::W1`]).
    pub fn flag(flag: Flag) -> Reg {
        Reg {
            space: Space::Flag,
            num: flag as u16,
            width: Width::W1,
        }
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// A pure, unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    /// Two's-complement negation, at the operand width.
    Neg,
    /// Bitwise complement, at the operand width.
    Not,
    /// Zero-extend the operand to the given (wider) width.
    ZeroExtend(Width),
    /// Sign-extend the operand to the given (wider) width.
    SignExtend(Width),
    /// Truncate the operand to the given (narrower) width.
    Truncate(Width),
}

/// A pure, binary operator. Comparisons yield a [`Width::W1`] value;
/// every other operator yields its operands' shared width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    /// Unsigned division; division by zero is a lift-defined trap, not a
    /// value, so a lifter that models `div` guards it explicitly.
    UDiv,
    /// Signed division.
    SDiv,
    /// Unsigned remainder.
    URem,
    /// Signed remainder.
    SRem,
    And,
    Or,
    Xor,
    /// Logical left shift.
    Shl,
    /// Logical (zero-filling) right shift.
    LShr,
    /// Arithmetic (sign-filling) right shift.
    AShr,
    /// Equality — yields `W1`.
    Eq,
    /// Inequality — yields `W1`.
    Ne,
    /// Unsigned less-than — yields `W1`.
    Ult,
    /// Unsigned less-or-equal — yields `W1`.
    Ule,
    /// Signed less-than — yields `W1`.
    Slt,
    /// Signed less-or-equal — yields `W1`.
    Sle,
}

impl BinOp {
    /// Whether the operator yields a [`Width::W1`] comparison result
    /// rather than a value at its operands' width.
    pub fn is_compare(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Ult | BinOp::Ule | BinOp::Slt | BinOp::Sle
        )
    }

    /// The infix token for [`render`].
    pub fn token(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::UDiv => "/u",
            BinOp::SDiv => "/s",
            BinOp::URem => "%u",
            BinOp::SRem => "%s",
            BinOp::And => "&",
            BinOp::Or => "|",
            BinOp::Xor => "^",
            BinOp::Shl => "<<",
            BinOp::LShr => ">>u",
            BinOp::AShr => ">>s",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Ult => "<u",
            BinOp::Ule => "<=u",
            BinOp::Slt => "<s",
            BinOp::Sle => "<=s",
        }
    }
}

/// A pure value expression. Evaluating an `Expr` yields a bit-vector of a
/// definite [`Width::width_of`] and never changes machine state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// A constant of the given width; `value` is stored masked to it.
    Const { value: u64, width: Width },
    /// The current contents of a storage cell.
    Reg(Reg),
    /// A memory load of `width` bytes from address `addr` (a `W64`
    /// expression). Loads are the only place reading depends on memory.
    Load { addr: Box<Expr>, width: Width },
    /// A unary operator applied to `operand`.
    Unary { op: UnOp, operand: Box<Expr> },
    /// A binary operator applied to `lhs` and `rhs`, which share a width
    /// (except shifts, whose `rhs` is a shift amount of any width).
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
}

impl Expr {
    /// A constant, masked to `width`.
    pub fn constant(value: u64, width: Width) -> Expr {
        Expr::Const {
            value: value & width.mask(),
            width,
        }
    }

    /// A register/flag/temporary read.
    pub fn reg(reg: Reg) -> Expr {
        Expr::Reg(reg)
    }

    /// A memory load of `width` from `addr`.
    pub fn load(addr: Expr, width: Width) -> Expr {
        Expr::Load {
            addr: Box::new(addr),
            width,
        }
    }

    /// A unary application.
    pub fn unary(op: UnOp, operand: Expr) -> Expr {
        Expr::Unary {
            op,
            operand: Box::new(operand),
        }
    }

    /// A binary application.
    pub fn binary(op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        Expr::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    /// The width this expression evaluates to, or `None` if the tree is
    /// malformed in a way that leaves the width undefined (a comparison's
    /// operands disagreeing does not affect *its* width, which is always
    /// `W1`; a non-comparison `Binary` with mismatched operands has no
    /// well-defined width and returns `None`). [`check`] is the total
    /// validator; this is the width oracle it and [`render`] consult.
    pub fn width_of(&self) -> Option<Width> {
        match self {
            Expr::Const { width, .. } => Some(*width),
            Expr::Reg(reg) => Some(reg.width),
            Expr::Load { width, .. } => Some(*width),
            Expr::Unary { op, operand } => match op {
                UnOp::Neg | UnOp::Not => operand.width_of(),
                UnOp::ZeroExtend(w) | UnOp::SignExtend(w) | UnOp::Truncate(w) => Some(*w),
            },
            Expr::Binary { op, lhs, rhs } => {
                if op.is_compare() {
                    Some(Width::W1)
                } else if matches!(op, BinOp::Shl | BinOp::LShr | BinOp::AShr) {
                    // A shift takes its width from the value being shifted;
                    // the amount may be any width.
                    lhs.width_of()
                } else {
                    match (lhs.width_of(), rhs.width_of()) {
                        (Some(a), Some(b)) if a == b => Some(a),
                        _ => None,
                    }
                }
            }
        }
    }

    /// The number of nodes in the tree, used to bound recursion in
    /// [`check`] and [`render`] and to keep a hostile lift from building
    /// an unbounded expression.
    fn node_count(&self) -> usize {
        match self {
            Expr::Const { .. } | Expr::Reg(_) => 1,
            Expr::Load { addr, .. } => 1 + addr.node_count(),
            Expr::Unary { operand, .. } => 1 + operand.node_count(),
            Expr::Binary { lhs, rhs, .. } => 1 + lhs.node_count() + rhs.node_count(),
        }
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

/// The control-flow character of a [`Stmt::Branch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchKind {
    /// A plain jump (conditional when the branch carries a `cond`).
    Jump,
    /// A call: the return address is pushed by the lift's own [`Stmt`]s,
    /// so this only records the transfer and its kind.
    Call,
    /// A return.
    Return,
}

/// One unit of machine effect. A lifted instruction is a `Vec<Stmt>`
/// executed top to bottom; a temporary written by an earlier statement is
/// visible to a later one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    /// `dst := value`. The widths of `dst` and `value` must match.
    Assign { dst: Reg, value: Expr },
    /// Store `value` to memory at `addr` (a `W64` expression). `value`'s
    /// width is the access size; it may not be `W1`.
    Store { addr: Expr, value: Expr },
    /// Transfer control. `cond`, when present, is a `W1` guard; `target`
    /// is a `W64` address expression (an absolute constant for a direct
    /// branch, a register/load for an indirect one).
    Branch {
        kind: BranchKind,
        cond: Option<Expr>,
        target: Expr,
    },
    /// An effect the IR does not model in detail — a floating-point op, a
    /// port access, a barrier. Named for the reader; its `reads`/`writes`
    /// record the state it touches so an analysis stays sound across it.
    Intrinsic {
        name: &'static str,
        writes: Vec<Reg>,
        reads: Vec<Expr>,
    },
}

// ---------------------------------------------------------------------------
// Well-formedness
// ---------------------------------------------------------------------------

/// Why a statement list is not well-formed. [`check`] returns the first
/// fault, with enough context to locate it in a lift.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// An assignment or store whose value width does not match its
    /// destination (assignment) or is `W1` (store).
    WidthMismatch {
        stmt: usize,
        expected: Width,
        found: Option<Width>,
    },
    /// A memory address expression that is not `W64`.
    BadAddress { stmt: usize, found: Option<Width> },
    /// A branch condition that is not `W1`.
    BadCondition { stmt: usize, found: Option<Width> },
    /// A binary operation whose non-shift operands disagree in width.
    OperandMismatch { stmt: usize },
    /// A temporary read in a statement before any statement wrote it.
    TempReadBeforeWrite { stmt: usize, temp: u16 },
    /// The statement list, or one expression in it, exceeds the size caps.
    TooLarge { stmt: usize },
}

/// Largest expression tree [`check`] and [`render`] will walk, per the
/// module contract that a lift cannot build unbounded IR.
pub const MAX_EXPR_NODES: usize = 4096;

/// Largest statement list one lifted unit may hold.
pub const MAX_STMTS: usize = 4096;

/// Validate a lifted statement list. Total and side-effect-free: returns
/// `Ok(())` for a well-formed list, or the first [`Fault`]. Never panics.
///
/// The temporary-liveness check treats the list as a straight-line block:
/// a [`Space::Temp`] register must be assigned by an earlier statement
/// before it is read. This catches the common lift bug of reusing a
/// temporary number across two instructions without redefining it.
pub fn check(stmts: &[Stmt]) -> Result<(), Fault> {
    if stmts.len() > MAX_STMTS {
        return Err(Fault::TooLarge { stmt: MAX_STMTS });
    }
    let mut written: Vec<u16> = Vec::new();
    for (i, stmt) in stmts.iter().enumerate() {
        match stmt {
            Stmt::Assign { dst, value } => {
                check_expr(value, i, &written)?;
                if value.width_of() != Some(dst.width) {
                    return Err(Fault::WidthMismatch {
                        stmt: i,
                        expected: dst.width,
                        found: value.width_of(),
                    });
                }
                if dst.space == Space::Temp {
                    written.push(dst.num);
                }
            }
            Stmt::Store { addr, value } => {
                check_expr(addr, i, &written)?;
                check_expr(value, i, &written)?;
                if addr.width_of() != Some(Width::W64) {
                    return Err(Fault::BadAddress {
                        stmt: i,
                        found: addr.width_of(),
                    });
                }
                match value.width_of() {
                    Some(Width::W1) | None => {
                        return Err(Fault::WidthMismatch {
                            stmt: i,
                            expected: Width::W8,
                            found: value.width_of(),
                        });
                    }
                    _ => {}
                }
            }
            Stmt::Branch { cond, target, .. } => {
                if let Some(c) = cond {
                    check_expr(c, i, &written)?;
                    if c.width_of() != Some(Width::W1) {
                        return Err(Fault::BadCondition {
                            stmt: i,
                            found: c.width_of(),
                        });
                    }
                }
                check_expr(target, i, &written)?;
                if target.width_of() != Some(Width::W64) {
                    return Err(Fault::BadAddress {
                        stmt: i,
                        found: target.width_of(),
                    });
                }
            }
            Stmt::Intrinsic { writes, reads, .. } => {
                for r in reads {
                    check_expr(r, i, &written)?;
                }
                for w in writes {
                    if w.space == Space::Temp {
                        written.push(w.num);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validate one expression: bounded size, consistent operand widths, and
/// no read of an unwritten temporary.
fn check_expr(expr: &Expr, stmt: usize, written: &[u16]) -> Result<(), Fault> {
    if expr.node_count() > MAX_EXPR_NODES {
        return Err(Fault::TooLarge { stmt });
    }
    check_expr_inner(expr, stmt, written)
}

fn check_expr_inner(expr: &Expr, stmt: usize, written: &[u16]) -> Result<(), Fault> {
    match expr {
        Expr::Const { .. } => Ok(()),
        Expr::Reg(reg) => {
            if reg.space == Space::Temp && !written.contains(&reg.num) {
                return Err(Fault::TempReadBeforeWrite {
                    stmt,
                    temp: reg.num,
                });
            }
            Ok(())
        }
        Expr::Load { addr, .. } => {
            check_expr_inner(addr, stmt, written)?;
            if addr.width_of() != Some(Width::W64) {
                return Err(Fault::BadAddress {
                    stmt,
                    found: addr.width_of(),
                });
            }
            Ok(())
        }
        Expr::Unary { op, operand } => {
            check_expr_inner(operand, stmt, written)?;
            // Extend must widen, truncate must narrow; a no-op or inverted
            // change is a lift bug worth catching.
            if let Some(from) = operand.width_of() {
                match op {
                    UnOp::ZeroExtend(to) | UnOp::SignExtend(to) if to.bits() <= from.bits() => {
                        return Err(Fault::WidthMismatch {
                            stmt,
                            expected: *to,
                            found: Some(from),
                        });
                    }
                    UnOp::Truncate(to) if to.bits() >= from.bits() => {
                        return Err(Fault::WidthMismatch {
                            stmt,
                            expected: *to,
                            found: Some(from),
                        });
                    }
                    _ => {}
                }
            }
            Ok(())
        }
        Expr::Binary { op, lhs, rhs } => {
            check_expr_inner(lhs, stmt, written)?;
            check_expr_inner(rhs, stmt, written)?;
            let shift = matches!(op, BinOp::Shl | BinOp::LShr | BinOp::AShr);
            if !shift {
                match (lhs.width_of(), rhs.width_of()) {
                    (Some(a), Some(b)) if a == b => {}
                    _ => return Err(Fault::OperandMismatch { stmt }),
                }
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// How [`render`] names an architectural register. A lifter supplies one
/// so the IR core needs no ISA register tables; `None` from the hook (an
/// unknown number) falls back to `r{num}`.
pub type RegNamer<'a> = &'a dyn Fn(u16, Width) -> Option<String>;

/// How [`render_with`] names *any* storage cell. `None` falls back to the
/// default form for the cell's space (`r{n}.{w}`, `t{n}.{w}`, the flag
/// token). A caller that renumbers registers — SSA renaming, say — uses
/// this hook to print its own names for every space, not just `Arch`.
pub type CellNamer<'a> = &'a dyn Fn(Reg) -> Option<String>;

/// Render a lifted statement list to a deterministic, human-readable
/// textual form, one statement per line. `namer` names architectural
/// registers; pass a closure returning `None` for the raw `r{n}` form.
///
/// The form is stable — the same statements always produce the same
/// string — so lifted output is testable and diffable. Never panics; an
/// oversized tree is truncated with a `…` marker rather than walked
/// past [`MAX_EXPR_NODES`].
pub fn render(stmts: &[Stmt], namer: RegNamer) -> String {
    render_with(stmts, &|reg: Reg| {
        if reg.space == Space::Arch {
            namer(reg.num, reg.width)
        } else {
            None
        }
    })
}

/// [`render`] with a whole-[`Reg`] namer hook, so a caller can name flags
/// and temporaries too (a plain [`render`] names only architectural
/// registers). Same determinism and no-panic contract as [`render`].
pub fn render_with(stmts: &[Stmt], namer: CellNamer) -> String {
    let mut out = String::new();
    for stmt in stmts.iter().take(MAX_STMTS) {
        render_stmt(stmt, namer, &mut out);
        out.push('\n');
    }
    out
}

fn render_stmt(stmt: &Stmt, namer: CellNamer, out: &mut String) {
    match stmt {
        Stmt::Assign { dst, value } => {
            render_reg(*dst, namer, out);
            out.push_str(" := ");
            render_expr(value, namer, out, 0);
        }
        Stmt::Store { addr, value } => {
            let w = value.width_of().unwrap_or(Width::W8);
            let _ = write!(out, "store.{} [", w.token());
            render_expr(addr, namer, out, 0);
            out.push_str("], ");
            render_expr(value, namer, out, 0);
        }
        Stmt::Branch { kind, cond, target } => {
            let verb = match kind {
                BranchKind::Jump => "goto",
                BranchKind::Call => "call",
                BranchKind::Return => "return",
            };
            out.push_str(verb);
            if let Some(c) = cond {
                out.push_str(" if ");
                render_expr(c, namer, out, 0);
                out.push(' ');
            } else {
                out.push(' ');
            }
            render_expr(target, namer, out, 0);
        }
        Stmt::Intrinsic {
            name,
            writes,
            reads,
        } => {
            for (i, w) in writes.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                render_reg(*w, namer, out);
            }
            if !writes.is_empty() {
                out.push_str(" := ");
            }
            let _ = write!(out, "{name}(");
            for (i, r) in reads.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                render_expr(r, namer, out, 0);
            }
            out.push(')');
        }
    }
}

fn render_reg(reg: Reg, namer: CellNamer, out: &mut String) {
    if let Some(name) = namer(reg) {
        out.push_str(&name);
        return;
    }
    match reg.space {
        Space::Arch => {
            let _ = write!(out, "r{}.{}", reg.num, reg.width.token());
        }
        Space::Temp => {
            let _ = write!(out, "t{}.{}", reg.num, reg.width.token());
        }
        Space::Flag => {
            let flag = flag_of(reg.num);
            out.push_str(flag.map(Flag::token).unwrap_or("F?"));
        }
    }
}

/// Recover a [`Flag`] from a stored discriminant, for rendering.
fn flag_of(num: u16) -> Option<Flag> {
    match num {
        0 => Some(Flag::Carry),
        1 => Some(Flag::Zero),
        2 => Some(Flag::Sign),
        3 => Some(Flag::Overflow),
        4 => Some(Flag::Parity),
        _ => None,
    }
}

/// Render an expression, parenthesizing by a coarse precedence so the
/// output is unambiguous without being noisy. `depth` bounds recursion.
fn render_expr(expr: &Expr, namer: CellNamer, out: &mut String, depth: usize) {
    if depth > MAX_EXPR_NODES {
        out.push('…');
        return;
    }
    match expr {
        Expr::Const { value, width } => {
            let _ = write!(out, "{:#x}.{}", value, width.token());
        }
        Expr::Reg(reg) => render_reg(*reg, namer, out),
        Expr::Load { addr, width } => {
            let _ = write!(out, "load.{} [", width.token());
            render_expr(addr, namer, out, depth + 1);
            out.push(']');
        }
        Expr::Unary { op, operand } => {
            let name = match op {
                UnOp::Neg => "-",
                UnOp::Not => "~",
                UnOp::ZeroExtend(w) => {
                    let _ = write!(out, "zext.{}(", w.token());
                    render_expr(operand, namer, out, depth + 1);
                    out.push(')');
                    return;
                }
                UnOp::SignExtend(w) => {
                    let _ = write!(out, "sext.{}(", w.token());
                    render_expr(operand, namer, out, depth + 1);
                    out.push(')');
                    return;
                }
                UnOp::Truncate(w) => {
                    let _ = write!(out, "trunc.{}(", w.token());
                    render_expr(operand, namer, out, depth + 1);
                    out.push(')');
                    return;
                }
            };
            out.push_str(name);
            out.push('(');
            render_expr(operand, namer, out, depth + 1);
            out.push(')');
        }
        Expr::Binary { op, lhs, rhs } => {
            out.push('(');
            render_expr(lhs, namer, out, depth + 1);
            let _ = write!(out, " {} ", op.token());
            render_expr(rhs, namer, out, depth + 1);
            out.push(')');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A register namer mapping 0/1/2 to rax/rcx/rdx for readable tests.
    fn namer(num: u16, _w: Width) -> Option<String> {
        match num {
            0 => Some("rax".to_string()),
            1 => Some("rcx".to_string()),
            2 => Some("rdx".to_string()),
            _ => None,
        }
    }

    fn render_one(stmt: Stmt) -> String {
        let s = render(std::slice::from_ref(&stmt), &namer);
        s.trim_end().to_string()
    }

    #[test]
    fn width_helpers_are_consistent() {
        assert_eq!(Width::W32.bits(), 32);
        assert_eq!(Width::W32.bytes(), 4);
        assert_eq!(Width::W32.mask(), 0xffff_ffff);
        assert_eq!(Width::W64.mask(), u64::MAX);
        assert_eq!(Width::W1.bytes(), 1);
    }

    #[test]
    fn a_constant_is_masked_to_its_width() {
        let e = Expr::constant(0x1_2345, Width::W8);
        assert_eq!(e, Expr::Const { value: 0x45, width: Width::W8 });
        assert_eq!(e.width_of(), Some(Width::W8));
    }

    #[test]
    fn a_simple_add_assignment_checks_and_renders() {
        // rax.q := (rax.q + rcx.q)
        let stmt = Stmt::Assign {
            dst: Reg::arch(0, Width::W64),
            value: Expr::binary(
                BinOp::Add,
                Expr::reg(Reg::arch(0, Width::W64)),
                Expr::reg(Reg::arch(1, Width::W64)),
            ),
        };
        assert_eq!(check(std::slice::from_ref(&stmt)), Ok(()));
        assert_eq!(render_one(stmt), "rax := (rax + rcx)");
    }

    #[test]
    fn a_comparison_yields_width_one() {
        let cmp = Expr::binary(
            BinOp::Ult,
            Expr::reg(Reg::arch(0, Width::W32)),
            Expr::reg(Reg::arch(1, Width::W32)),
        );
        assert_eq!(cmp.width_of(), Some(Width::W1));
        // A flag assignment from it is well-formed.
        let stmt = Stmt::Assign {
            dst: Reg::flag(Flag::Carry),
            value: cmp,
        };
        assert_eq!(check(std::slice::from_ref(&stmt)), Ok(()));
    }

    #[test]
    fn assignment_width_mismatch_is_a_fault() {
        // rax.q := rcx.d  — 32-bit value into a 64-bit cell.
        let stmt = Stmt::Assign {
            dst: Reg::arch(0, Width::W64),
            value: Expr::reg(Reg::arch(1, Width::W32)),
        };
        assert_eq!(
            check(std::slice::from_ref(&stmt)),
            Err(Fault::WidthMismatch {
                stmt: 0,
                expected: Width::W64,
                found: Some(Width::W32),
            })
        );
    }

    #[test]
    fn a_non_shift_binary_with_mismatched_operands_is_a_fault() {
        let stmt = Stmt::Assign {
            dst: Reg::arch(0, Width::W64),
            value: Expr::binary(
                BinOp::Add,
                Expr::reg(Reg::arch(0, Width::W64)),
                Expr::reg(Reg::arch(1, Width::W32)),
            ),
        };
        assert_eq!(check(std::slice::from_ref(&stmt)), Err(Fault::OperandMismatch { stmt: 0 }));
    }

    #[test]
    fn a_shift_may_mix_operand_widths() {
        // rax.q := (rax.q << rcx.b)
        let stmt = Stmt::Assign {
            dst: Reg::arch(0, Width::W64),
            value: Expr::binary(
                BinOp::Shl,
                Expr::reg(Reg::arch(0, Width::W64)),
                Expr::reg(Reg::arch(1, Width::W8)),
            ),
        };
        assert_eq!(check(std::slice::from_ref(&stmt)), Ok(()));
        // rcx is named, so its width suffix is carried by the name, not
        // printed; the shift amount's narrower width is still valid.
        assert_eq!(render_one(stmt), "rax := (rax << rcx)");
    }

    #[test]
    fn a_store_of_a_flag_width_value_is_rejected() {
        let stmt = Stmt::Store {
            addr: Expr::reg(Reg::arch(0, Width::W64)),
            value: Expr::reg(Reg::flag(Flag::Zero)),
        };
        assert!(matches!(
            check(std::slice::from_ref(&stmt)),
            Err(Fault::WidthMismatch { stmt: 0, .. })
        ));
    }

    #[test]
    fn a_store_address_must_be_64_bit() {
        let stmt = Stmt::Store {
            addr: Expr::reg(Reg::arch(0, Width::W32)),
            value: Expr::reg(Reg::arch(1, Width::W32)),
        };
        assert_eq!(
            check(std::slice::from_ref(&stmt)),
            Err(Fault::BadAddress { stmt: 0, found: Some(Width::W32) })
        );
    }

    #[test]
    fn a_branch_condition_must_be_width_one() {
        let stmt = Stmt::Branch {
            kind: BranchKind::Jump,
            cond: Some(Expr::reg(Reg::arch(0, Width::W32))),
            target: Expr::constant(0x1000, Width::W64),
        };
        assert_eq!(
            check(std::slice::from_ref(&stmt)),
            Err(Fault::BadCondition { stmt: 0, found: Some(Width::W32) })
        );
    }

    #[test]
    fn a_conditional_direct_branch_renders() {
        let stmt = Stmt::Branch {
            kind: BranchKind::Jump,
            cond: Some(Expr::reg(Reg::flag(Flag::Zero))),
            target: Expr::constant(0x401000, Width::W64),
        };
        assert_eq!(check(std::slice::from_ref(&stmt)), Ok(()));
        assert_eq!(render_one(stmt), "goto if ZF 0x401000.q");
    }

    #[test]
    fn extend_must_widen_and_truncate_must_narrow() {
        // zext to a narrower width is a fault.
        let bad = Stmt::Assign {
            dst: Reg::arch(0, Width::W8),
            value: Expr::unary(UnOp::ZeroExtend(Width::W8), Expr::reg(Reg::arch(1, Width::W32))),
        };
        assert!(matches!(
            check(std::slice::from_ref(&bad)),
            Err(Fault::WidthMismatch { stmt: 0, .. })
        ));
        // A proper widening checks out.
        let good = Stmt::Assign {
            dst: Reg::arch(0, Width::W64),
            value: Expr::unary(UnOp::SignExtend(Width::W64), Expr::reg(Reg::arch(1, Width::W32))),
        };
        assert_eq!(check(std::slice::from_ref(&good)), Ok(()));
    }

    #[test]
    fn a_temporary_must_be_written_before_it_is_read() {
        // t0 read with no prior write.
        let read_first = Stmt::Assign {
            dst: Reg::arch(0, Width::W64),
            value: Expr::reg(Reg::temp(0, Width::W64)),
        };
        assert_eq!(
            check(std::slice::from_ref(&read_first)),
            Err(Fault::TempReadBeforeWrite { stmt: 0, temp: 0 })
        );
        // Written, then read: the xchg idiom.
        let ok = vec![
            Stmt::Assign {
                dst: Reg::temp(0, Width::W64),
                value: Expr::reg(Reg::arch(0, Width::W64)),
            },
            Stmt::Assign {
                dst: Reg::arch(0, Width::W64),
                value: Expr::reg(Reg::arch(1, Width::W64)),
            },
            Stmt::Assign {
                dst: Reg::arch(1, Width::W64),
                value: Expr::reg(Reg::temp(0, Width::W64)),
            },
        ];
        assert_eq!(check(&ok), Ok(()));
    }

    #[test]
    fn a_push_lifts_to_a_decrement_and_a_store() {
        // rsp is arch 4 here. rsp := rsp - 8 ; store.q [rsp], rax
        let rsp = Reg::arch(4, Width::W64);
        let stmts = vec![
            Stmt::Assign {
                dst: rsp,
                value: Expr::binary(
                    BinOp::Sub,
                    Expr::reg(rsp),
                    Expr::constant(8, Width::W64),
                ),
            },
            Stmt::Store {
                addr: Expr::reg(rsp),
                value: Expr::reg(Reg::arch(0, Width::W64)),
            },
        ];
        assert_eq!(check(&stmts), Ok(()));
        // rsp (arch 4) is unnamed by this test's namer, so it prints in
        // the raw `r{n}.{w}` form; rax is named.
        let text = render(&stmts, &namer);
        assert_eq!(text, "r4.q := (r4.q - 0x8.q)\nstore.q [r4.q], rax\n");
    }

    #[test]
    fn an_intrinsic_records_its_reads_and_writes() {
        let stmt = Stmt::Intrinsic {
            name: "rdtsc",
            writes: vec![Reg::arch(0, Width::W64), Reg::arch(2, Width::W64)],
            reads: vec![],
        };
        assert_eq!(check(std::slice::from_ref(&stmt)), Ok(()));
        assert_eq!(render_one(stmt), "rax, rdx := rdtsc()");
    }

    #[test]
    fn an_oversized_statement_list_is_rejected_not_walked() {
        let one = Stmt::Assign {
            dst: Reg::arch(0, Width::W64),
            value: Expr::reg(Reg::arch(0, Width::W64)),
        };
        let many = vec![one; MAX_STMTS + 1];
        assert_eq!(check(&many), Err(Fault::TooLarge { stmt: MAX_STMTS }));
    }

    #[test]
    fn a_deeply_nested_expression_is_bounded() {
        // Build a left-leaning add chain past the node cap.
        let mut e = Expr::reg(Reg::arch(0, Width::W64));
        for _ in 0..(MAX_EXPR_NODES + 10) {
            e = Expr::binary(BinOp::Add, e, Expr::constant(1, Width::W64));
        }
        let stmt = Stmt::Assign {
            dst: Reg::arch(0, Width::W64),
            value: e,
        };
        assert_eq!(check(std::slice::from_ref(&stmt)), Err(Fault::TooLarge { stmt: 0 }));
    }

    #[test]
    fn render_with_names_every_space_and_falls_back_per_space() {
        let stmts = vec![
            Stmt::Assign {
                dst: Reg::flag(Flag::Zero),
                value: Expr::constant(1, Width::W1),
            },
            Stmt::Assign {
                dst: Reg::temp(0, Width::W64),
                value: Expr::reg(Reg::arch(0, Width::W64)),
            },
        ];
        let text = render_with(&stmts, &|reg: Reg| match reg.space {
            Space::Flag => Some("zf.1".to_string()),
            Space::Temp => Some(format!("tmp{}", reg.num)),
            Space::Arch => None, // falls back to the raw r{n}.{w} form
        });
        assert_eq!(text, "zf.1 := 0x1.i1\ntmp0 := r0.q\n");
        // The plain `render` is the Arch-only wrapper: same statements,
        // default flag and temporary forms.
        assert_eq!(render(&stmts, &namer), "ZF := 0x1.i1\nt0.q := rax\n");
    }

    #[test]
    fn rendering_is_deterministic() {
        let stmts = vec![
            Stmt::Assign {
                dst: Reg::arch(0, Width::W64),
                value: Expr::binary(
                    BinOp::Xor,
                    Expr::reg(Reg::arch(0, Width::W64)),
                    Expr::reg(Reg::arch(0, Width::W64)),
                ),
            },
            Stmt::Assign {
                dst: Reg::flag(Flag::Zero),
                value: Expr::constant(1, Width::W1),
            },
        ];
        assert_eq!(render(&stmts, &namer), render(&stmts, &namer));
    }
}
