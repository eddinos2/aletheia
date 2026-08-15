//! Pseudocode rendering: the structure tree spelled as C-like text.
//!
//! This is the join point of the decompiler pipeline — the first output a
//! reader is meant to *read*. [`crate::irstruct`] recovered the shape
//! (sequences, `if`/`else`, loops, `switch`, honest gotos) and
//! [`crate::irout`] assigned every SSA name a variable; this module walks
//! the tree and spells both as deterministic pseudocode. It computes
//! nothing new: conditions are fetched from the deciding blocks at print
//! time (the tree stores references, never expressions), storage is
//! spelled through [`OutOfSsa::var_of`], and the printer only *spells* —
//! simplification belongs to the SSA passes, never here.
//!
//! # The renderer is arch-agnostic
//!
//! Every storage cell prints as a variable `vN` through
//! [`OutOfSsa::var_of`], behind a [`VarNamer`] hook whose default is the
//! `vN` form — so stack/type recovery can plug richer names in later
//! without touching the printer. Because every name becomes `vN`, the
//! renderer needs no per-ISA register table: it is the same code for
//! x86-64 and aarch64.
//!
//! # Precedence-aware parenthesization
//!
//! [`crate::ir::render`] wraps every binary node in parentheses — faithful,
//! but unreadable. This module prints with a C-modeled precedence and
//! associativity table instead, parenthesizing a child **iff** it binds
//! looser than its context, or equally on the right (the non-associative
//! side) — the Ghidra `PrintC` rule. On top of that, *forced redundant
//! parentheses* are always emitted where bitwise, shift, and comparison
//! operators mix (except a same-operator `&`/`|`/`^` chain), because that
//! corner of C precedence is a documented defect readers mis-parse —
//! DREAM++'s readability finding (Yakdan, Dechand, Gerhards-Padilla,
//! Smith: "Helping Johnny to Analyze Malware", IEEE S&P 2016, on the
//! DREAM line of Yakdan et al., NDSS 2015). The width casts stay in the
//! functional `zext.q(x)`/`sext.q(x)`/`trunc.d(x)` form — unambiguous, no
//! parenthesis question — and signedness stays on the operator (`/u`,
//! `<s`, `>>s`) with widths explicit on constants (`0x7.d`), the house
//! style inherited from the IR renderer: no C casts are invented.
//!
//! The non-ambiguity claim is *proved*, not asserted: the tests hold a
//! tiny precedence reparser, and every rendered expression must reparse
//! to the tree it came from. A golden file alone cannot show an output
//! is unambiguous; the round trip can.
//!
//! # Edge copies: the obligation [`crate::irout`] hands the renderer
//!
//! [`OutOfSsa::edge_copies`] lists copies that belong **on** a CFG edge —
//! on a critical edge, hoisting them into the predecessor would clobber a
//! variable live on a sibling edge. The walk places each list at the
//! point its edge is realized in the tree: at the end of a block whose
//! flow is unconditional, before the `goto`/`break`/`continue` that takes
//! the edge, at the top of a leaf reached by exactly one pending edge
//! (before its label, so a `goto` into the label skips them), just before
//! a loop for its entry edge, or in a synthesized `else { }` for the
//! untaken side of an `if` with no else arm. **Honest limit:** a
//! `do`-`while` back edge and a conditional loop-exit edge that shares
//! its target with breaks have no textual site in C syntax without
//! synthesizing a condition (banned); their copies are reported in an
//! `/* unplaced edge copies ... */` marker rather than dropped or
//! misplaced — honesty over silent loss.
//!
//! # Honesty markers (comments, never silent)
//!
//! - `/* lift truncated */` — a block whose lift stopped early.
//! - `/* indirect jump: successors unknown */` — an unproven indirect
//!   jump, held opaque.
//! - `/* undecidable exits */` — a block whose successors nothing decides.
//! - `/* abi-assumed: ... */` — the rendered variables whose value is at
//!   some point asserted by the ABI ([`OutOfSsa::assumed`]).
//! - `/* reads bits its def never wrote: ... */` — the rendered variables
//!   read wider than their definition ([`OutOfSsa::partial`]).
//!
//! Each statement carries its block's VA as a right-margin `// 0x…`
//! comment, anchoring a future recompile-differential oracle (Liu & Wang,
//! "How far we have come", ISSTA 2020) and keeping every line traceable
//! to bytes. The lift does not carry per-instruction VAs through SSA, so
//! the anchor is block-granular; a provenance rider can sharpen it
//! without changing this format.
//!
//! # Contract
//!
//! [`render`] is pure, total and deterministic: equal inputs produce
//! byte-equal output, no input panics — a hand-broken tree, a foreign
//! block, a missing condition, or a malformed function render what is
//! valid and mark the rest. Expression size is bounded by
//! [`ir::MAX_EXPR_NODES`] and tree depth by [`irstruct::MAX_TREE_DEPTH`],
//! both truncating with an explicit marker, mirroring the IR renderer.
//! Statements are never reordered, invented, or silently dropped: a
//! leaf's terminating jump is the only statement the tree absorbs, and
//! the constructs that absorbed it print in its place — a `switch` case
//! that converges on the follow prints the `break;` that spells its
//! recorded exit edge. One reading caveat is inherent to reference-typed
//! conditions and recorded rather than papered over: a `while` header's
//! non-branch statements print at the top of the body, textually after
//! the test they feed, because the printer never duplicates or reorders
//! a statement to move them. Expression forwarding usually leaves the
//! header holding nothing but its collapsed relational branch, so on
//! real pipeline output the caveat is rarely visible.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::ir::{self, BinOp, BranchKind, Expr, Reg, Stmt, UnOp};
use crate::irout::{CopySlot, OutOfSsa};
use crate::irssa::{SsaBlock, SsaFunction};
use crate::irstruct::{self, Cond, LoopKind, Node, OpaqueReason};

/// How the renderer spells a variable id. `None` falls back to the
/// default `v{id}` form. Stack and type recovery plug in here later; the
/// hook keeps the printer free of any per-ISA or per-analysis naming.
pub type VarNamer<'a> = &'a dyn Fn(u32) -> Option<String>;

/// Spaces per indentation level.
const INDENT: usize = 4;

/// Column where the right-margin `// 0x…` VA comment starts.
const VA_COLUMN: usize = 48;

/// Render `root` over `f` with the default `v{id}` variable spelling.
/// See the module docs for the contract.
pub fn render(f: &SsaFunction, root: &Node, vars: &OutOfSsa) -> String {
    render_with(f, root, vars, &|_| None)
}

/// [`render`] with a [`VarNamer`] hook. Same contract.
pub fn render_with(f: &SsaFunction, root: &Node, vars: &OutOfSsa, namer: VarNamer) -> String {
    let mut labels = BTreeSet::new();
    goto_targets(root, &mut labels, 0);
    let mut r = Renderer {
        f,
        vars,
        namer,
        text: String::new(),
        used: BTreeSet::new(),
        emitted: BTreeSet::new(),
        labels,
        breaks: Vec::new(),
    };

    // The function-entry prologue: copies for the virtual entry edge (an
    // entry-block phi fed by a loop back to entry).
    let entry_copies = vars.entry_copies.clone();
    for copy in &entry_copies {
        let code = r.copy_code(copy.dst, copy.src);
        r.line(1, &code, Margin::Text("entry"));
    }
    let ctx = Ctx {
        after: None,
        brk: None,
        cont: None,
    };
    r.walk(root, BTreeSet::new(), ctx, 1, 0);

    // Copies the walk found no honest site for: reported, never dropped.
    let unplaced: Vec<(u64, u64)> = vars
        .edge_copies
        .keys()
        .filter(|e| !r.emitted.contains(e))
        .copied()
        .collect();
    for (p, s) in unplaced {
        let list = vars.edge_copies[&(p, s)]
            .iter()
            .map(|c| r.copy_code(c.dst, c.src))
            .collect::<Vec<_>>()
            .join(" ");
        r.line(
            1,
            &format!("/* unplaced edge copies loc_{p:x} -> loc_{s:x}: {list} */"),
            Margin::None,
        );
    }

    // Assemble: the marker preamble needs the used-variable set, which
    // only exists after the body has been rendered.
    let name = f
        .name
        .clone()
        .unwrap_or_else(|| format!("sub_{:x}", f.entry));
    let mut out = String::new();
    let _ = writeln!(out, "// {name} @ {:#018x} (pseudo)", f.entry);
    let _ = writeln!(out, "{name}() {{");
    if !f.live_in.is_empty() {
        let list = f
            .live_in
            .iter()
            .map(|&id| leaf_fallback(vars, namer, id))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "{:INDENT$}// live-in: {list}", "");
    }
    let assumed: Vec<String> = r
        .used
        .iter()
        .filter(|v| vars.assumed.contains(v))
        .map(|&v| namer(v).unwrap_or_else(|| format!("v{v}")))
        .collect();
    if !assumed.is_empty() {
        let _ = writeln!(out, "{:INDENT$}/* abi-assumed: {} */", "", assumed.join(", "));
    }
    let partial: Vec<String> = r
        .used
        .iter()
        .filter(|v| vars.partial.contains(v))
        .map(|&v| namer(v).unwrap_or_else(|| format!("v{v}")))
        .collect();
    if !partial.is_empty() {
        let _ = writeln!(
            out,
            "{:INDENT$}/* reads bits its def never wrote: {} */",
            "",
            partial.join(", ")
        );
    }
    out.push_str(&r.text);
    out.push_str("}\n");
    if !f.skipped.is_empty() {
        let list = f
            .skipped
            .iter()
            .map(|s| format!("loc_{s:x}"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "// unreachable, not decompiled: {list}");
    }
    out
}

/// The variable spelling of SSA name `id`, without recording a use — for
/// the live-in header line. Total: an id outside the map prints `v?{id}`.
fn leaf_fallback(vars: &OutOfSsa, namer: VarNamer, id: u16) -> String {
    match vars.var_of.get(id as usize) {
        Some(&v) => namer(v).unwrap_or_else(|| format!("v{v}")),
        None => format!("v?{id}"),
    }
}

// ---------------------------------------------------------------------------
// Precedence
// ---------------------------------------------------------------------------

/// The operator families whose mixtures get forced parentheses. C's
/// relative precedence of these three is the documented defect DREAM++
/// papers over with redundant parentheses; this renderer does the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    Arith,
    Shift,
    Bitwise,
    Compare,
}

fn family(op: BinOp) -> Family {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::UDiv | BinOp::SDiv | BinOp::URem
        | BinOp::SRem => Family::Arith,
        BinOp::Shl | BinOp::LShr | BinOp::AShr => Family::Shift,
        BinOp::And | BinOp::Or | BinOp::Xor => Family::Bitwise,
        BinOp::Eq | BinOp::Ne | BinOp::Ult | BinOp::Ule | BinOp::Slt | BinOp::Sle => {
            Family::Compare
        }
    }
}

/// Binding strength, C-modeled: multiplicative over additive over shift
/// over relational over equality over `&` over `^` over `|`. All the
/// binary operators here associate to the left.
fn prec(op: BinOp) -> u8 {
    match op {
        BinOp::Mul | BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem => 10,
        BinOp::Add | BinOp::Sub => 9,
        BinOp::Shl | BinOp::LShr | BinOp::AShr => 8,
        BinOp::Ult | BinOp::Ule | BinOp::Slt | BinOp::Sle => 7,
        BinOp::Eq | BinOp::Ne => 6,
        BinOp::And => 5,
        BinOp::Xor => 4,
        BinOp::Or => 3,
    }
}

/// Whether a binary child of `parent` needs parentheses: looser binding,
/// equal binding on the right (left association), or the forced-paren
/// rule for mixed bitwise/shift/comparison operators (a same-operator
/// `&`/`|`/`^` chain is the one exempt mixture — it is not a mixture).
fn needs_parens(parent: BinOp, child: BinOp, right: bool) -> bool {
    let noisy = |f: Family| !matches!(f, Family::Arith);
    if noisy(family(parent))
        && noisy(family(child))
        && !(parent == child && matches!(parent, BinOp::And | BinOp::Or | BinOp::Xor))
    {
        return true;
    }
    let (pp, cp) = (prec(parent), prec(child));
    cp < pp || (cp == pp && right)
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// How a leaf [`Reg`] is spelled — the renderer's one seam to the
/// variable map, kept as a closure so the tests can drive the expression
/// printer without an [`OutOfSsa`].
type LeafNamer<'a> = &'a mut dyn FnMut(Reg) -> String;

/// Render one expression at statement level.
fn expr_str(e: &Expr, leaf: LeafNamer) -> String {
    let mut out = String::new();
    let mut budget = ir::MAX_EXPR_NODES;
    write_expr(e, leaf, &mut out, None, &mut budget);
    out
}

/// Render one expression as an operand of a prefix form (`!`, `-`, `~`,
/// a deref): parenthesized unless it is self-delimiting.
fn operand_str(e: &Expr, leaf: LeafNamer) -> String {
    let mut out = String::new();
    let mut budget = ir::MAX_EXPR_NODES;
    write_operand(e, leaf, &mut out, &mut budget);
    out
}

/// The recursive expression printer. `parent` is the enclosing binary
/// operator and which side this child is on; `budget` bounds the walk at
/// [`ir::MAX_EXPR_NODES`], truncating with `…` — never a panic.
fn write_expr(
    e: &Expr,
    leaf: LeafNamer,
    out: &mut String,
    parent: Option<(BinOp, bool)>,
    budget: &mut usize,
) {
    if *budget == 0 {
        out.push('…');
        return;
    }
    *budget -= 1;
    match e {
        Expr::Const { value, width } => {
            let _ = write!(out, "{:#x}.{}", value, width.token());
        }
        Expr::Reg(reg) => {
            let name = leaf(*reg);
            out.push_str(&name);
        }
        Expr::Load { addr, width } => {
            let _ = write!(out, "*(u{}*)", width.bits());
            write_operand(addr, leaf, out, budget);
        }
        Expr::Unary { op, operand } => match op {
            UnOp::Neg => {
                out.push('-');
                write_operand(operand, leaf, out, budget);
            }
            UnOp::Not => {
                out.push('~');
                write_operand(operand, leaf, out, budget);
            }
            UnOp::ZeroExtend(w) | UnOp::SignExtend(w) | UnOp::Truncate(w) => {
                let name = match op {
                    UnOp::ZeroExtend(_) => "zext",
                    UnOp::SignExtend(_) => "sext",
                    _ => "trunc",
                };
                let _ = write!(out, "{name}.{}(", w.token());
                write_expr(operand, leaf, out, None, budget);
                out.push(')');
            }
        },
        Expr::Binary { op, lhs, rhs } => {
            let wrap = parent.is_some_and(|(p, right)| needs_parens(p, *op, right));
            if wrap {
                out.push('(');
            }
            write_expr(lhs, leaf, out, Some((*op, false)), budget);
            let _ = write!(out, " {} ", op.token());
            write_expr(rhs, leaf, out, Some((*op, true)), budget);
            if wrap {
                out.push(')');
            }
        }
    }
}

/// A prefix form's operand: parenthesized when it is a binary expression
/// or a `-`/`~` unary (so `--x` can never be printed), bare when it is
/// self-delimiting (a constant, a variable, a load, a cast).
fn write_operand(e: &Expr, leaf: LeafNamer, out: &mut String, budget: &mut usize) {
    let delimited = matches!(
        e,
        Expr::Binary { .. }
            | Expr::Unary {
                op: UnOp::Neg | UnOp::Not,
                ..
            }
    );
    if delimited {
        out.push('(');
        write_expr(e, leaf, out, None, budget);
        out.push(')');
    } else {
        write_expr(e, leaf, out, None, budget);
    }
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

/// What surrounds the node being walked, mirroring the structuring
/// verifier's context: the entry of whatever follows on fall-through,
/// and the enclosing loop's follow (`break` target) and header
/// (`continue` target).
#[derive(Debug, Clone, Copy)]
struct Ctx {
    after: Option<u64>,
    brk: Option<u64>,
    cont: Option<u64>,
}

/// What goes in a line's right margin.
enum Margin {
    None,
    /// A block VA: `// 0x…`.
    Va(u64),
    /// An edge: `// 0x… -> 0x…`.
    Edge(u64, u64),
    /// A fixed word.
    Text(&'static str),
}

struct Renderer<'a> {
    f: &'a SsaFunction,
    vars: &'a OutOfSsa,
    namer: VarNamer<'a>,
    /// The function body (inside the braces), assembled first; header
    /// and marker preamble need the used-variable set.
    text: String,
    /// Variables the text actually mentions, for the marker preamble.
    used: BTreeSet<u32>,
    /// Edges whose copies have been placed.
    emitted: BTreeSet<(u64, u64)>,
    /// In-tree `goto` targets: leaves at these VAs print a label.
    labels: BTreeSet<u64>,
    /// Per enclosing loop, the pending blocks its `break`s carried out —
    /// what the loop's follow sees as predecessors.
    breaks: Vec<BTreeSet<u64>>,
}

impl<'a> Renderer<'a> {
    // -- lines --------------------------------------------------------------

    fn line(&mut self, indent: usize, code: &str, margin: Margin) {
        let pad = INDENT * indent;
        let _ = write!(self.text, "{:pad$}{code}", "");
        let comment = match margin {
            Margin::None => None,
            Margin::Va(va) => Some(format!("// {va:#x}")),
            Margin::Edge(p, s) => Some(format!("// {p:#x} -> {s:#x}")),
            Margin::Text(t) => Some(format!("// {t}")),
        };
        if let Some(comment) = comment {
            let len = pad + code.chars().count();
            if len < VA_COLUMN {
                let _ = write!(self.text, "{:w$}", "", w = VA_COLUMN - len);
            } else {
                self.text.push(' ');
            }
            self.text.push_str(&comment);
        }
        self.text.push('\n');
    }

    // -- naming -------------------------------------------------------------

    /// Spell variable `v`, recording the mention for the marker preamble.
    fn vname(&mut self, v: u32) -> String {
        self.used.insert(v);
        (self.namer)(v).unwrap_or_else(|| format!("v{v}"))
    }

    fn copy_code(&mut self, dst: CopySlot, src: CopySlot) -> String {
        let d = self.slot(dst);
        let s = self.slot(src);
        format!("{d} = {s};")
    }

    fn slot(&mut self, s: CopySlot) -> String {
        match s {
            CopySlot::Var(v) => self.vname(v),
            CopySlot::Temp => "tmp".to_string(),
        }
    }

    /// Build a statement's code text; the leaf closure maps every SSA
    /// name through the variable map and records the mention.
    fn with_leaf<R>(&mut self, f: impl FnOnce(LeafNamer) -> R) -> R {
        let vars = self.vars;
        let namer = self.namer;
        let used = &mut self.used;
        let mut leaf = |r: Reg| -> String {
            match vars.var_of.get(r.num as usize) {
                Some(&v) => {
                    used.insert(v);
                    namer(v).unwrap_or_else(|| format!("v{v}"))
                }
                None => format!("v?{}", r.num),
            }
        };
        f(&mut leaf)
    }

    // -- copies -------------------------------------------------------------

    /// Place the copies of edge `(from, to)` here, once. A miss (no
    /// copies, or already placed) writes nothing.
    fn place(&mut self, from: u64, to: u64, indent: usize) {
        if !self.vars.edge_copies.contains_key(&(from, to)) || !self.emitted.insert((from, to)) {
            return;
        }
        let copies = self.vars.edge_copies[&(from, to)].clone();
        for c in copies {
            let code = self.copy_code(c.dst, c.src);
            self.line(indent, &code, Margin::Edge(from, to));
        }
    }

    /// Place copies from a pending set onto edge target `to` — only when
    /// the site belongs to exactly one edge; a merged site would execute
    /// one edge's copies on another edge's path.
    fn place_pending(&mut self, pending: &BTreeSet<u64>, to: u64, indent: usize) {
        if pending.len() == 1 {
            let p = *pending.iter().next().expect("len checked");
            self.place(p, to, indent);
        }
    }

    // -- statements ---------------------------------------------------------

    /// One statement's code, sans indentation and margin. Total on any
    /// statement; `goto_form` renders terminators explicitly (opaque and
    /// mid-block branches), which structured leaves never need.
    fn stmt_code(&mut self, stmt: &Stmt) -> String {
        self.with_leaf(|leaf| match stmt {
            Stmt::Assign { dst, value } => {
                format!("{} = {};", leaf(*dst), expr_str(value, leaf))
            }
            Stmt::Store { addr, value } => {
                let bits = value.width_of().map_or(8, |w| w.bits());
                format!(
                    "*(u{bits}*){} = {};",
                    operand_str(addr, leaf),
                    expr_str(value, leaf)
                )
            }
            Stmt::Branch { kind, cond, target } => match kind {
                BranchKind::Return => "return;".to_string(),
                BranchKind::Call => {
                    let t = match target {
                        Expr::Const { value, .. } => format!("{value:#x}"),
                        other => operand_str(other, leaf),
                    };
                    format!("call {t}();")
                }
                BranchKind::Jump => {
                    let dest = match target {
                        Expr::Const { value, .. } => format!("goto loc_{value:x};"),
                        other => format!("goto *{};", operand_str(other, leaf)),
                    };
                    match cond {
                        Some(c) => format!("if ({}) {dest}", expr_str(c, leaf)),
                        None => dest,
                    }
                }
            },
            Stmt::Intrinsic {
                name,
                writes,
                reads,
            } => {
                let args = reads
                    .iter()
                    .map(|r| expr_str(r, leaf))
                    .collect::<Vec<_>>()
                    .join(", ");
                if writes.is_empty() {
                    format!("{name}({args});")
                } else {
                    let w = writes.iter().map(|w| leaf(*w)).collect::<Vec<_>>().join(", ");
                    format!("{name}({args}); /* writes {w} */")
                }
            }
        })
    }

    /// A leaf's statements. A structured leaf's terminating jump is the
    /// one statement the tree absorbs (`skip_last_jump`); everything else
    /// prints, calls and returns included.
    fn stmts(&mut self, block: &SsaBlock, skip_last_jump: bool, indent: usize) {
        let n = block.stmts.len();
        for (i, stmt) in block.stmts.iter().enumerate() {
            if skip_last_jump
                && i + 1 == n
                && matches!(
                    stmt,
                    Stmt::Branch {
                        kind: BranchKind::Jump,
                        ..
                    }
                )
            {
                continue;
            }
            let code = self.stmt_code(stmt);
            self.line(indent, &code, Margin::Va(block.start));
        }
    }

    // -- conditions ---------------------------------------------------------

    /// The condition text a [`Cond`] denotes: the deciding block's guard,
    /// negated per its polarity. Total: a missing or non-conditional
    /// block yields an honest marker instead of an invented condition.
    fn cond_code(&mut self, cond: Cond) -> String {
        let Some(block) = self.f.blocks.get(&cond.block) else {
            return format!("/* no condition: loc_{:x} */", cond.block);
        };
        match block.stmts.last() {
            Some(Stmt::Branch {
                kind: BranchKind::Jump,
                cond: Some(g),
                ..
            }) => {
                let g = g.clone();
                self.with_leaf(|leaf| {
                    if cond.negated {
                        format!("!{}", operand_str(&g, leaf))
                    } else {
                        expr_str(&g, leaf)
                    }
                })
            }
            _ => format!("/* no condition: loc_{:x} */", cond.block),
        }
    }

    /// The `(taken, fall-through)` targets of a conditional block, for
    /// the untaken-edge copy site of an `if` with no else arm.
    fn sides(&self, block: u64) -> Option<(u64, u64)> {
        let b = self.f.blocks.get(&block)?;
        let Some(Stmt::Branch {
            kind: BranchKind::Jump,
            cond: Some(_),
            target: Expr::Const { value, .. },
        }) = b.stmts.last()
        else {
            return None;
        };
        let taken = *value;
        let mut seen = BTreeSet::new();
        let succs: Vec<u64> = b
            .successors
            .iter()
            .copied()
            .filter(|s| seen.insert(*s))
            .collect();
        if succs.len() != 2 {
            return None;
        }
        let other = if succs[0] == taken {
            succs[1]
        } else if succs[1] == taken {
            succs[0]
        } else {
            return None;
        };
        Some((taken, other))
    }

    // -- nodes --------------------------------------------------------------

    /// The block control reaches when it enters `node`, as far as the
    /// tree says — the structuring verifier's rule, reused for pending
    /// edges and `break` targets.
    fn entry_of(&self, node: &Node, ctx: Ctx, depth: usize) -> Option<u64> {
        if depth > irstruct::MAX_TREE_DEPTH {
            return None;
        }
        match node {
            Node::Block(b) | Node::Opaque { block: b, .. } | Node::Switch { block: b, .. } => {
                Some(*b)
            }
            Node::Seq(v) => v.iter().find_map(|c| self.entry_of(c, ctx, depth + 1)),
            Node::Loop { body, .. } => self.entry_of(body, ctx, depth + 1),
            Node::Goto(t) => Some(*t),
            Node::Break => ctx.brk,
            Node::Continue => ctx.cont,
            Node::If { .. } => None,
        }
    }

    /// Walk `node` with `pending` — the blocks whose realized out-edge
    /// arrives at this textual point — and return the pending set it
    /// leaves for what follows.
    fn walk(
        &mut self,
        node: &Node,
        pending: BTreeSet<u64>,
        ctx: Ctx,
        indent: usize,
        depth: usize,
    ) -> BTreeSet<u64> {
        if depth > irstruct::MAX_TREE_DEPTH {
            self.line(indent, "/* tree too deep, truncated */", Margin::None);
            return BTreeSet::new();
        }
        match node {
            Node::Block(b) => self.leaf(*b, &pending, None, indent),
            Node::Opaque { block, reason } => self.leaf(*block, &pending, Some(*reason), indent),
            Node::Seq(v) => {
                let mut p = pending;
                for (i, child) in v.iter().enumerate() {
                    let after = v[i + 1..]
                        .iter()
                        .find_map(|n| self.entry_of(n, ctx, depth + 1))
                        .or(ctx.after);
                    p = self.walk(child, p, Ctx { after, ..ctx }, indent, depth + 1);
                }
                p
            }
            Node::Goto(t) => {
                self.place_pending(&pending, *t, indent);
                let margin = single(&pending).map_or(Margin::None, Margin::Va);
                self.line(indent, &format!("goto loc_{t:x};"), margin);
                BTreeSet::new()
            }
            Node::Break => {
                if let Some(fw) = ctx.brk {
                    self.place_pending(&pending, fw, indent);
                }
                let margin = single(&pending).map_or(Margin::None, Margin::Va);
                self.line(indent, "break;", margin);
                if let Some(acc) = self.breaks.last_mut() {
                    acc.extend(pending);
                }
                BTreeSet::new()
            }
            Node::Continue => {
                if let Some(h) = ctx.cont {
                    self.place_pending(&pending, h, indent);
                }
                let margin = single(&pending).map_or(Margin::None, Margin::Va);
                self.line(indent, "continue;", margin);
                BTreeSet::new()
            }
            Node::If {
                cond,
                then_body,
                else_body,
            } => self.walk_if(*cond, then_body, else_body.as_deref(), pending, ctx, indent, depth),
            Node::Loop { kind, cond, body } => {
                self.walk_loop(*kind, *cond, body, pending, ctx, indent, depth)
            }
            Node::Switch { block, cases } => {
                self.walk_switch(*block, cases, pending, ctx, indent, depth)
            }
        }
    }

    /// A `Block`/`Opaque` leaf: incoming copies, label, statements,
    /// honesty markers, outgoing copies for an unconditional edge.
    fn leaf(
        &mut self,
        b: u64,
        pending: &BTreeSet<u64>,
        opaque: Option<OpaqueReason>,
        indent: usize,
    ) -> BTreeSet<u64> {
        // Copies for the one pending edge land before the label, so a
        // goto into the label does not execute them.
        self.place_pending(pending, b, indent);
        if self.labels.contains(&b) {
            self.line(indent, &format!("loc_{b:x}:"), Margin::None);
        }
        let Some(block) = self.f.blocks.get(&b) else {
            self.line(indent, &format!("/* missing block loc_{b:x} */"), Margin::None);
            return BTreeSet::new();
        };
        let block = block.clone();
        self.stmts(&block, opaque.is_none(), indent);
        if block.truncated || opaque == Some(OpaqueReason::Truncated) {
            self.line(indent, "/* lift truncated */", Margin::None);
        }
        match opaque {
            Some(OpaqueReason::IndirectJump) => {
                self.line(indent, "/* indirect jump: successors unknown */", Margin::None);
            }
            Some(OpaqueReason::Unstructurable) => {
                self.line(indent, "/* undecidable exits */", Margin::None);
            }
            _ => {}
        }
        let mut seen = BTreeSet::new();
        let succs: Vec<u64> = block
            .successors
            .iter()
            .copied()
            .filter(|s| seen.insert(*s))
            .collect();
        // A block leaving unconditionally is itself the site of its one
        // out-edge: its copies go after its statements.
        if succs.len() == 1 && self.f.blocks.contains_key(&succs[0]) {
            self.place(b, succs[0], indent);
        }
        if opaque == Some(OpaqueReason::Unstructurable) || succs.is_empty() {
            BTreeSet::new()
        } else {
            BTreeSet::from([b])
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_if(
        &mut self,
        cond: Cond,
        then_body: &Node,
        else_body: Option<&Node>,
        pending: BTreeSet<u64>,
        ctx: Ctx,
        indent: usize,
        depth: usize,
    ) -> BTreeSet<u64> {
        let code = self.cond_code(cond);
        self.line(indent, &format!("if ({code}) {{"), Margin::Va(cond.block));
        let mut out = self.walk(then_body, pending.clone(), ctx, indent + 1, depth + 1);
        match else_body {
            Some(e) => {
                self.line(indent, "} else {", Margin::None);
                out.extend(self.walk(e, pending, ctx, indent + 1, depth + 1));
                self.line(indent, "}", Margin::None);
            }
            None => {
                // The untaken edge falls to the follow. If it carries
                // copies, the only site C syntax offers is an else arm
                // synthesized to hold exactly them.
                let untaken = self.sides(cond.block).map(|(taken, fallthrough)| {
                    if cond.negated { taken } else { fallthrough }
                });
                match untaken {
                    Some(t)
                        if self.vars.edge_copies.contains_key(&(cond.block, t))
                            && !self.emitted.contains(&(cond.block, t)) =>
                    {
                        self.line(indent, "} else {", Margin::None);
                        self.place(cond.block, t, indent + 1);
                        self.line(indent, "}", Margin::None);
                    }
                    _ => self.line(indent, "}", Margin::None),
                }
                out.extend(pending);
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_loop(
        &mut self,
        kind: LoopKind,
        cond: Option<Cond>,
        body: &Node,
        pending: BTreeSet<u64>,
        ctx: Ctx,
        indent: usize,
        depth: usize,
    ) -> BTreeSet<u64> {
        let header = self.entry_of(body, ctx, depth + 1);
        // The loop-entry edge is realized just before the loop opens.
        if let Some(h) = header {
            self.place_pending(&pending, h, indent);
        }
        let do_form = matches!(kind, LoopKind::DoWhile | LoopKind::SelfLoop) && cond.is_some();
        match (do_form, cond) {
            (true, _) => {
                let margin = header.map_or(Margin::None, Margin::Va);
                self.line(indent, "do {", margin);
            }
            (false, Some(c)) => {
                let code = self.cond_code(c);
                self.line(indent, &format!("while ({code}) {{"), Margin::Va(c.block));
            }
            (false, None) => {
                let margin = header.map_or(Margin::None, Margin::Va);
                self.line(indent, "while (true) {", margin);
            }
        }
        let inner = Ctx {
            after: header,
            brk: ctx.after,
            cont: header,
        };
        self.breaks.push(BTreeSet::new());
        let tail = self.walk(body, BTreeSet::new(), inner, indent + 1, depth + 1);
        let broke = self.breaks.pop().unwrap_or_default();
        if do_form {
            let c = cond.expect("do_form requires a condition");
            let code = self.cond_code(c);
            self.line(indent, &format!("}} while ({code});"), Margin::Va(c.block));
        } else {
            self.line(indent, "}", Margin::None);
        }
        // Falling off the end of the body is the back edge; a pending
        // block with no edge to the header is a loop exit instead.
        let mut out = broke;
        for p in tail {
            let back = header.is_some_and(|h| {
                self.f
                    .blocks
                    .get(&p)
                    .is_some_and(|b| b.successors.contains(&h))
            });
            if !back {
                out.insert(p);
            }
        }
        if let Some(c) = cond {
            out.insert(c.block);
        }
        out
    }

    fn walk_switch(
        &mut self,
        block: u64,
        cases: &[(u64, Node)],
        pending: BTreeSet<u64>,
        ctx: Ctx,
        indent: usize,
        depth: usize,
    ) -> BTreeSet<u64> {
        self.place_pending(&pending, block, indent);
        if self.labels.contains(&block) {
            self.line(indent, &format!("loc_{block:x}:"), Margin::None);
        }
        let scrutinee = match self.f.blocks.get(&block).cloned() {
            Some(b) => {
                self.stmts(&b, true, indent);
                match b.stmts.last() {
                    Some(Stmt::Branch {
                        kind: BranchKind::Jump,
                        cond: None,
                        target,
                    }) => {
                        let target = target.clone();
                        self.with_leaf(|leaf| expr_str(&target, leaf))
                    }
                    _ => format!("/* no scrutinee: loc_{block:x} */"),
                }
            }
            None => {
                self.line(
                    indent,
                    &format!("/* missing block loc_{block:x} */"),
                    Margin::None,
                );
                format!("/* no scrutinee: loc_{block:x} */")
            }
        };
        self.line(indent, &format!("switch ({scrutinee}) {{"), Margin::Va(block));
        let mut out = BTreeSet::new();
        for (t, body) in cases {
            self.line(indent, &format!("case loc_{t:x}:"), Margin::None);
            let case_out = self.walk(body, BTreeSet::from([block]), ctx, indent + 1, depth + 1);
            // A case that falls out of the switch converges on the
            // follow; in C the spelling of that recorded edge is
            // `break;` — a realization, not invented flow.
            if !case_out.is_empty() {
                if let Some(fw) = ctx.after {
                    self.place_pending(&case_out, fw, indent + 1);
                }
                let margin = single(&case_out).map_or(Margin::None, Margin::Va);
                self.line(indent + 1, "break;", margin);
                out.extend(case_out);
            }
        }
        self.line(indent, "}", Margin::None);
        out
    }
}

/// The single element of a one-element set.
fn single(set: &BTreeSet<u64>) -> Option<u64> {
    if set.len() == 1 {
        set.iter().next().copied()
    } else {
        None
    }
}

/// Collect the tree's `goto` targets — the leaves that need a label.
fn goto_targets(node: &Node, acc: &mut BTreeSet<u64>, depth: usize) {
    if depth > irstruct::MAX_TREE_DEPTH {
        return;
    }
    match node {
        Node::Goto(t) => {
            acc.insert(*t);
        }
        Node::Seq(v) => {
            for c in v {
                goto_targets(c, acc, depth + 1);
            }
        }
        Node::If {
            then_body,
            else_body,
            ..
        } => {
            goto_targets(then_body, acc, depth + 1);
            if let Some(e) = else_body {
                goto_targets(e, acc, depth + 1);
            }
        }
        Node::Loop { body, .. } => goto_targets(body, acc, depth + 1),
        Node::Switch { cases, .. } => {
            for (_, body) in cases {
                goto_targets(body, acc, depth + 1);
            }
        }
        Node::Block(_) | Node::Opaque { .. } | Node::Break | Node::Continue => {}
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::callfx;
    use crate::ir::{Flag, Width};
    use crate::irout::{self, Copy};
    use crate::irssa;
    use crate::irssaopt;
    use crate::model::Arch;
    use crate::irlift;

    // -- construction helpers ----------------------------------------------

    fn c(value: u64, w: Width) -> Expr {
        Expr::constant(value, w)
    }
    fn ra(num: u16, w: Width) -> Reg {
        Reg::arch(num, w)
    }
    fn read(r: Reg) -> Expr {
        Expr::reg(r)
    }
    fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r)
    }
    fn assign(dst: Reg, value: Expr) -> Stmt {
        Stmt::Assign { dst, value }
    }
    /// `zf := rax == n` — a flag definition to branch on.
    fn set_flag(n: u64) -> Stmt {
        assign(
            Reg::flag(Flag::Zero),
            bin(BinOp::Eq, read(ra(0, Width::W64)), c(n, Width::W64)),
        )
    }
    /// `goto if zf -> target`.
    fn jcc(target: u64) -> Stmt {
        Stmt::Branch {
            kind: BranchKind::Jump,
            cond: Some(read(Reg::flag(Flag::Zero))),
            target: c(target, Width::W64),
        }
    }
    fn jmp(target: u64) -> Stmt {
        Stmt::Branch {
            kind: BranchKind::Jump,
            cond: None,
            target: c(target, Width::W64),
        }
    }
    fn ret() -> Stmt {
        Stmt::Branch {
            kind: BranchKind::Return,
            cond: None,
            target: read(ra(16, Width::W64)),
        }
    }
    fn indirect() -> Stmt {
        Stmt::Branch {
            kind: BranchKind::Jump,
            cond: None,
            target: read(ra(0, Width::W64)),
        }
    }

    fn block(start: u64, stmts: Vec<Stmt>, successors: Vec<u64>) -> irlift::LiftedBlock {
        irlift::LiftedBlock {
            start,
            end: start + 0x10,
            stmts,
            successors,
            truncated: false,
        }
    }

    fn func(entry: u64, blocks: Vec<irlift::LiftedBlock>) -> irlift::LiftedFunction {
        irlift::LiftedFunction {
            entry,
            name: None,
            arch: crate::model::Arch::X86_64,
            blocks: blocks.into_iter().map(|b| (b.start, b)).collect(),
        }
    }

    fn build(f: &irlift::LiftedFunction) -> SsaFunction {
        let ssa = irssa::construct(f).expect("well-formed input constructs");
        assert_eq!(irssa::check(&ssa), Ok(()), "input SSA must check");
        ssa
    }

    fn no_tables() -> BTreeMap<u64, Vec<u64>> {
        BTreeMap::new()
    }

    /// Structure + out-of-SSA over the raw construction.
    fn pipe(f: &irlift::LiftedFunction) -> (SsaFunction, Node, OutOfSsa) {
        pipe_tabled(f, &no_tables())
    }

    fn pipe_tabled(
        f: &irlift::LiftedFunction,
        tables: &BTreeMap<u64, Vec<u64>>,
    ) -> (SsaFunction, Node, OutOfSsa) {
        let ssa = build(f);
        let (root, _) = irstruct::structure(&ssa, tables);
        assert_eq!(irstruct::check(&ssa, tables, &root), Ok(()));
        let (vars, _) = irout::out_of_ssa(&ssa);
        assert_eq!(irout::check(&ssa, &vars), Ok(()));
        (ssa, root, vars)
    }

    /// The optimized pipeline (propagation + forwarding), for the shapes
    /// that only exist downstream of value movement.
    fn opt_pipe(f: &irlift::LiftedFunction) -> (SsaFunction, Node, OutOfSsa) {
        let ssa = build(f);
        let (opt, _) = irssaopt::optimize(&ssa);
        let (fwd, _) = irssaopt::forward(&opt);
        assert_eq!(irssa::check(&fwd), Ok(()));
        let (root, _) = irstruct::structure(&fwd, &no_tables());
        assert_eq!(irstruct::check(&fwd, &no_tables(), &root), Ok(()));
        let (vars, _) = irout::out_of_ssa(&fwd);
        assert_eq!(irout::check(&fwd, &vars), Ok(()));
        (fwd, root, vars)
    }

    /// Render and insist on the module's promises: total, and a second
    /// run is byte-identical.
    fn text(f: &SsaFunction, root: &Node, vars: &OutOfSsa) -> String {
        let t = render(f, root, vars);
        assert_eq!(t, render(f, root, vars), "rendering must be deterministic");
        t
    }

    /// The code column: margins stripped, indentation kept, header/close
    /// and full-line comments kept verbatim.
    fn code_lines(t: &str) -> Vec<String> {
        t.lines()
            .map(|l| {
                if l.trim_start().starts_with("//") {
                    return l.trim_end().to_string();
                }
                match l.find("  //") {
                    Some(pos) => l[..pos].trim_end().to_string(),
                    None => l.trim_end().to_string(),
                }
            })
            .collect()
    }

    /// `code_lines` with every `v{digits}` masked to `vN`, so structural
    /// goldens do not depend on variable numbering.
    fn shape(t: &str) -> Vec<String> {
        code_lines(t)
            .into_iter()
            .map(|l| {
                let mut out = String::new();
                let bytes = l.as_bytes();
                let mut i = 0;
                while i < bytes.len() {
                    if bytes[i] == b'v'
                        && i + 1 < bytes.len()
                        && bytes[i + 1].is_ascii_digit()
                        && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric())
                    {
                        out.push_str("vN");
                        i += 1;
                        while i < bytes.len() && bytes[i].is_ascii_digit() {
                            i += 1;
                        }
                    } else {
                        out.push(bytes[i] as char);
                        i += 1;
                    }
                }
                out
            })
            .collect()
    }

    // -- expression building for the precedence tests ------------------------

    fn v(n: u16) -> Expr {
        read(ra(n, Width::W64))
    }

    fn render_expr(e: &Expr) -> String {
        let mut leaf = |r: Reg| format!("v{}", r.num);
        expr_str(e, &mut leaf)
    }

    // -- 1: precedence goldens ----------------------------------------------

    #[test]
    fn left_association_prints_flat_and_right_nesting_parenthesizes() {
        assert_eq!(
            render_expr(&bin(BinOp::Add, bin(BinOp::Add, v(0), v(1)), v(2))),
            "v0 + v1 + v2"
        );
        assert_eq!(
            render_expr(&bin(BinOp::Add, v(0), bin(BinOp::Add, v(1), v(2)))),
            "v0 + (v1 + v2)"
        );
        assert_eq!(
            render_expr(&bin(BinOp::Sub, v(0), bin(BinOp::Sub, v(1), v(2)))),
            "v0 - (v1 - v2)"
        );
        assert_eq!(
            render_expr(&bin(BinOp::Sub, bin(BinOp::Sub, v(0), v(1)), v(2))),
            "v0 - v1 - v2"
        );
    }

    #[test]
    fn multiplicative_over_additive_needs_parens_only_where_binding_loosens() {
        assert_eq!(
            render_expr(&bin(BinOp::Mul, bin(BinOp::Add, v(0), v(1)), v(2))),
            "(v0 + v1) * v2"
        );
        assert_eq!(
            render_expr(&bin(BinOp::Add, bin(BinOp::Mul, v(0), v(1)), v(2))),
            "v0 * v1 + v2"
        );
        assert_eq!(
            render_expr(&bin(BinOp::UDiv, v(0), bin(BinOp::SDiv, v(1), v(2)))),
            "v0 /u (v1 /s v2)"
        );
    }

    #[test]
    fn relational_over_arithmetic_needs_no_parens() {
        assert_eq!(
            render_expr(&bin(BinOp::Eq, bin(BinOp::Add, v(0), v(1)), v(2))),
            "v0 + v1 == v2"
        );
        assert_eq!(
            render_expr(&bin(BinOp::Slt, v(0), bin(BinOp::Sub, v(1), v(2)))),
            "v0 <s v1 - v2"
        );
        assert_eq!(
            render_expr(&bin(BinOp::Ule, bin(BinOp::Mul, v(0), v(1)), v(2))),
            "v0 * v1 <=u v2"
        );
    }

    #[test]
    fn shift_vs_arithmetic_is_minimal_c_precedence() {
        // Additive binds tighter than shift, as in C: sufficient, minimal.
        assert_eq!(
            render_expr(&bin(BinOp::Shl, bin(BinOp::Add, v(0), v(1)), v(2))),
            "v0 + v1 << v2"
        );
        assert_eq!(
            render_expr(&bin(BinOp::Add, bin(BinOp::Shl, v(0), v(1)), v(2))),
            "(v0 << v1) + v2"
        );
    }

    // -- 2: the forced redundant parentheses (DREAM++) -----------------------

    #[test]
    fn bitwise_mixed_with_comparison_is_always_parenthesized() {
        // C precedence would let the first read without parens — the
        // documented defect. Forced parens spell it out.
        assert_eq!(
            render_expr(&bin(BinOp::And, v(0), bin(BinOp::Eq, v(1), v(2)))),
            "v0 & (v1 == v2)"
        );
        assert_eq!(
            render_expr(&bin(BinOp::Eq, bin(BinOp::And, v(0), v(1)), v(2))),
            "(v0 & v1) == v2"
        );
        assert_eq!(
            render_expr(&bin(BinOp::Ne, v(0), bin(BinOp::Or, v(1), v(2)))),
            "v0 != (v1 | v2)"
        );
    }

    #[test]
    fn shift_mixed_with_bitwise_is_always_parenthesized_but_chains_are_not() {
        assert_eq!(
            render_expr(&bin(BinOp::And, bin(BinOp::Shl, v(0), v(1)), v(2))),
            "(v0 << v1) & v2"
        );
        assert_eq!(
            render_expr(&bin(BinOp::Shl, bin(BinOp::And, v(0), v(1)), v(2))),
            "(v0 & v1) << v2"
        );
        assert_eq!(
            render_expr(&bin(BinOp::Or, bin(BinOp::And, v(0), v(1)), v(2))),
            "(v0 & v1) | v2"
        );
        // A same-operator chain is not a mixture.
        assert_eq!(
            render_expr(&bin(BinOp::And, bin(BinOp::And, v(0), v(1)), v(2))),
            "v0 & v1 & v2"
        );
        // The right side still parenthesizes: association must survive.
        assert_eq!(
            render_expr(&bin(BinOp::Xor, v(0), bin(BinOp::Xor, v(1), v(2)))),
            "v0 ^ (v1 ^ v2)"
        );
    }

    #[test]
    fn prefix_forms_casts_and_loads_render_unambiguously() {
        assert_eq!(
            render_expr(&Expr::unary(UnOp::Neg, bin(BinOp::Add, v(0), v(1)))),
            "-(v0 + v1)"
        );
        assert_eq!(
            render_expr(&Expr::unary(UnOp::Neg, Expr::unary(UnOp::Neg, v(0)))),
            "-(-v0)"
        );
        assert_eq!(render_expr(&Expr::unary(UnOp::Not, v(0))), "~v0");
        assert_eq!(
            render_expr(&Expr::unary(
                UnOp::ZeroExtend(Width::W64),
                Expr::unary(UnOp::Truncate(Width::W32), v(0)),
            )),
            "zext.q(trunc.d(v0))"
        );
        assert_eq!(
            render_expr(&Expr::load(bin(BinOp::Add, v(0), c(8, Width::W64)), Width::W32)),
            "*(u32*)(v0 + 0x8.q)"
        );
        assert_eq!(render_expr(&Expr::load(v(0), Width::W64)), "*(u64*)v0");
        assert_eq!(
            render_expr(&bin(BinOp::Sub, v(0), Expr::unary(UnOp::Neg, v(1)))),
            "v0 - -v1"
        );
    }

    // -- 3: the round-trip reparser — the parenthesization oracle ------------

    /// The reparser's tree: rendered text parsed back by precedence
    /// alone. If a rendering were ambiguous, some tree would reparse
    /// differently than it was meant.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Pt {
        Leaf(String),
        Const(u64, Width),
        Neg(Box<Pt>),
        Not(Box<Pt>),
        LNot(Box<Pt>),
        Load(u32, Box<Pt>),
        Cast(&'static str, Width, Box<Pt>),
        Bin(BinOp, Box<Pt>, Box<Pt>),
    }

    fn mirror(e: &Expr) -> Pt {
        match e {
            Expr::Const { value, width } => Pt::Const(*value, *width),
            Expr::Reg(r) => Pt::Leaf(format!("v{}", r.num)),
            Expr::Load { addr, width } => Pt::Load(width.bits(), Box::new(mirror(addr))),
            Expr::Unary { op, operand } => match op {
                UnOp::Neg => Pt::Neg(Box::new(mirror(operand))),
                UnOp::Not => Pt::Not(Box::new(mirror(operand))),
                UnOp::ZeroExtend(w) => Pt::Cast("zext", *w, Box::new(mirror(operand))),
                UnOp::SignExtend(w) => Pt::Cast("sext", *w, Box::new(mirror(operand))),
                UnOp::Truncate(w) => Pt::Cast("trunc", *w, Box::new(mirror(operand))),
            },
            Expr::Binary { op, lhs, rhs } => {
                Pt::Bin(*op, Box::new(mirror(lhs)), Box::new(mirror(rhs)))
            }
        }
    }

    struct Reparse<'a> {
        s: &'a str,
        i: usize,
    }

    const OPS: [(&str, BinOp); 19] = [
        (">>u", BinOp::LShr),
        (">>s", BinOp::AShr),
        ("<=u", BinOp::Ule),
        ("<=s", BinOp::Sle),
        ("<<", BinOp::Shl),
        ("==", BinOp::Eq),
        ("!=", BinOp::Ne),
        ("<u", BinOp::Ult),
        ("<s", BinOp::Slt),
        ("/u", BinOp::UDiv),
        ("/s", BinOp::SDiv),
        ("%u", BinOp::URem),
        ("%s", BinOp::SRem),
        ("+", BinOp::Add),
        ("-", BinOp::Sub),
        ("*", BinOp::Mul),
        ("&", BinOp::And),
        ("|", BinOp::Or),
        ("^", BinOp::Xor),
    ];

    impl<'a> Reparse<'a> {
        fn parse(src: &'a str) -> Option<Pt> {
            let mut p = Reparse { s: src, i: 0 };
            let e = p.expr(0)?;
            p.ws();
            if p.i == p.s.len() { Some(e) } else { None }
        }

        fn rest(&self) -> &'a str {
            &self.s[self.i..]
        }

        fn ws(&mut self) {
            while self.rest().starts_with(' ') {
                self.i += 1;
            }
        }

        fn eat(&mut self, t: &str) -> bool {
            self.ws();
            if self.rest().starts_with(t) {
                self.i += t.len();
                true
            } else {
                false
            }
        }

        fn expr(&mut self, min: u8) -> Option<Pt> {
            let mut lhs = self.unary()?;
            loop {
                self.ws();
                let Some(&(tok, op)) = OPS.iter().find(|(t, _)| self.rest().starts_with(t))
                else {
                    break;
                };
                if prec(op) < min {
                    break;
                }
                self.i += tok.len();
                let rhs = self.expr(prec(op) + 1)?;
                lhs = Pt::Bin(op, Box::new(lhs), Box::new(rhs));
            }
            Some(lhs)
        }

        fn width(&mut self) -> Option<Width> {
            for (tok, w) in [
                ("i1", Width::W1),
                ("b", Width::W8),
                ("w", Width::W16),
                ("d", Width::W32),
                ("q", Width::W64),
            ] {
                if self.rest().starts_with(tok) {
                    self.i += tok.len();
                    return Some(w);
                }
            }
            None
        }

        fn unary(&mut self) -> Option<Pt> {
            self.ws();
            if self.rest().starts_with("*(u") {
                self.i += 3;
                let start = self.i;
                while self.rest().starts_with(|c: char| c.is_ascii_digit()) {
                    self.i += 1;
                }
                let bits: u32 = self.s[start..self.i].parse().ok()?;
                if !self.eat("*)") {
                    return None;
                }
                return Some(Pt::Load(bits, Box::new(self.unary()?)));
            }
            if self.eat("!") {
                return Some(Pt::LNot(Box::new(self.unary()?)));
            }
            if self.eat("~") {
                return Some(Pt::Not(Box::new(self.unary()?)));
            }
            for name in ["zext", "sext", "trunc"] {
                if self.rest().starts_with(name) && self.s[self.i + name.len()..].starts_with('.')
                {
                    self.i += name.len() + 1;
                    let w = self.width()?;
                    if !self.eat("(") {
                        return None;
                    }
                    let e = self.expr(0)?;
                    if !self.eat(")") {
                        return None;
                    }
                    return Some(Pt::Cast(name, w, Box::new(e)));
                }
            }
            if self.rest().starts_with("0x") {
                self.i += 2;
                let start = self.i;
                while self.rest().starts_with(|c: char| c.is_ascii_hexdigit()) {
                    self.i += 1;
                }
                let value = u64::from_str_radix(&self.s[start..self.i], 16).ok()?;
                if !self.eat(".") {
                    return None;
                }
                return Some(Pt::Const(value, self.width()?));
            }
            if self.eat("-") {
                return Some(Pt::Neg(Box::new(self.unary()?)));
            }
            if self.eat("(") {
                let e = self.expr(0)?;
                if !self.eat(")") {
                    return None;
                }
                return Some(e);
            }
            let start = self.i;
            while self
                .rest()
                .starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == '?')
            {
                self.i += 1;
            }
            if self.i == start {
                return None;
            }
            Some(Pt::Leaf(self.s[start..self.i].to_string()))
        }
    }

    /// xorshift64*, seeded and deterministic.
    fn next(s: &mut u64) -> u64 {
        *s ^= *s >> 12;
        *s ^= *s << 25;
        *s ^= *s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    const ALL_OPS: [BinOp; 19] = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::UDiv,
        BinOp::SDiv,
        BinOp::URem,
        BinOp::SRem,
        BinOp::And,
        BinOp::Or,
        BinOp::Xor,
        BinOp::Shl,
        BinOp::LShr,
        BinOp::AShr,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::Ult,
        BinOp::Ule,
        BinOp::Slt,
        BinOp::Sle,
    ];

    fn gen_expr(rng: &mut u64, depth: usize) -> Expr {
        let r = next(rng);
        if depth == 0 || r % 10 < 2 {
            return if r.is_multiple_of(2) {
                v((next(rng) % 8) as u16)
            } else {
                let w = [Width::W8, Width::W32, Width::W64][(next(rng) % 3) as usize];
                Expr::constant(next(rng), w)
            };
        }
        match r % 10 {
            2 => Expr::unary(UnOp::Neg, gen_expr(rng, depth - 1)),
            3 => Expr::unary(UnOp::Not, gen_expr(rng, depth - 1)),
            4 => {
                let op = [
                    UnOp::ZeroExtend(Width::W64),
                    UnOp::SignExtend(Width::W64),
                    UnOp::Truncate(Width::W32),
                ][(next(rng) % 3) as usize];
                Expr::unary(op, gen_expr(rng, depth - 1))
            }
            5 => {
                let w = [Width::W32, Width::W64][(next(rng) % 2) as usize];
                Expr::load(gen_expr(rng, depth - 1), w)
            }
            _ => {
                let op = ALL_OPS[(next(rng) % 19) as usize];
                Expr::binary(op, gen_expr(rng, depth - 1), gen_expr(rng, depth - 1))
            }
        }
    }

    #[test]
    fn every_rendered_expression_reparses_to_the_same_tree() {
        let mut rng = 0x5EED_CAFE_F00D_1234u64;
        for i in 0..500 {
            let e = gen_expr(&mut rng, 4);
            let rendered = render_expr(&e);
            let parsed = Reparse::parse(&rendered);
            assert_eq!(
                parsed,
                Some(mirror(&e)),
                "iteration {i}: `{rendered}` does not round-trip ({e:?})"
            );
        }
    }

    #[test]
    fn a_negated_condition_reparses_as_the_negation_of_its_guard() {
        let mut rng = 0xDEAD_BEEF_0BAD_F00Du64;
        for _ in 0..100 {
            let e = gen_expr(&mut rng, 3);
            let mut leaf = |r: Reg| format!("v{}", r.num);
            let rendered = format!("!{}", operand_str(&e, &mut leaf));
            assert_eq!(
                Reparse::parse(&rendered),
                Some(Pt::LNot(Box::new(mirror(&e)))),
                "`{rendered}`"
            );
        }
    }

    // -- 4: the structure walk, construct by construct -----------------------

    #[test]
    fn a_straight_line_function_renders_as_a_flat_body() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![assign(ra(0, Width::W64), c(1, Width::W64))], vec![0x1010]),
                block(0x1010, vec![assign(ra(1, Width::W64), c(2, Width::W64))], vec![0x1020]),
                block(0x1020, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, vars) = pipe(&f);
        let t = text(&ssa, &root, &vars);
        assert_eq!(
            shape(&t),
            [
                "// sub_1000 @ 0x0000000000001000 (pseudo)",
                "sub_1000() {",
                "    // live-in: vN",
                "    vN = 0x1.q;",
                "    vN = 0x2.q;",
                "    return;",
                "}",
            ],
            "{t}"
        );
    }

    #[test]
    fn the_va_margin_sits_at_its_column_and_names_the_block() {
        let f = func(0x1000, vec![block(0x1000, vec![ret()], vec![])]);
        let (ssa, root, vars) = pipe(&f);
        let t = text(&ssa, &root, &vars);
        let line = t.lines().find(|l| l.contains("return;")).expect("a return line");
        // Four spaces of indent, the code, padding to the margin column,
        // then the block VA.
        assert_eq!(line, format!("    return;{:37}// 0x1000", ""));
    }

    #[test]
    fn a_diamond_renders_as_if_else_with_the_guard_fetched_at_print_time() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1020)], vec![0x1020, 0x1010]),
                block(0x1010, vec![assign(ra(1, Width::W64), c(7, Width::W64))], vec![0x1030]),
                block(0x1020, vec![assign(ra(1, Width::W64), c(9, Width::W64))], vec![0x1030]),
                block(0x1030, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, vars) = pipe(&f);
        let t = text(&ssa, &root, &vars);
        let s = shape(&t);
        assert_eq!(
            &s[2..],
            [
                "    // live-in: vN, vN",
                "    vN = vN == 0x1.q;",
                "    if (vN) {",
                "        vN = 0x9.q;",
                "    } else {",
                "        vN = 0x7.q;",
                "    }",
                "    return;",
                "}",
            ],
            "{t}"
        );
    }

    #[test]
    fn a_forwarded_relational_guard_appears_as_a_real_if_condition() {
        // The slice-5 exit shape: cmp/jcc plumbing forwarded into the
        // branch, so the pseudocode reads a relation, not a flag.
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![set_flag(0xffff_ffff), jcc(0x1020)],
                    vec![0x1020, 0x1010],
                ),
                block(0x1010, vec![assign(ra(1, Width::W64), c(7, Width::W64))], vec![0x1020]),
                block(0x1020, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, vars) = opt_pipe(&f);
        let t = text(&ssa, &root, &vars);
        assert!(
            t.contains("if (!(") && t.contains("== 0xffffffff.q))"),
            "the negated relational guard must render inline: {t}"
        );
    }

    #[test]
    fn a_while_loop_renders_with_its_continuation_condition() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![assign(ra(2, Width::W64), c(1, Width::W64))], vec![0x1010]),
                block(0x1010, vec![set_flag(0), jcc(0x1030)], vec![0x1030, 0x1020]),
                block(
                    0x1020,
                    vec![
                        assign(
                            ra(0, Width::W64),
                            bin(BinOp::Add, read(ra(0, Width::W64)), c(1, Width::W64)),
                        ),
                        jmp(0x1010),
                    ],
                    vec![0x1010],
                ),
                block(0x1030, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, vars) = pipe(&f);
        let t = text(&ssa, &root, &vars);
        let s = shape(&t);
        let w = s
            .iter()
            .position(|l| l.contains("while (!vN) {"))
            .unwrap_or_else(|| panic!("a while: {t}"));
        // The module's documented caveat: the header's non-branch
        // statements (here the un-forwarded flag computation) print at
        // the top of the body, then the increment, then the close.
        assert!(s[w + 1].contains("= vN == 0x0.q;"), "{t}");
        assert!(s[w + 2].contains("= vN + 0x1.q;"), "{t}");
        assert_eq!(s[w + 3], "    }", "{t}");
        assert!(!t.contains("do {"), "{t}");
    }

    #[test]
    fn a_do_while_loop_renders_with_the_test_at_the_bottom() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![assign(ra(2, Width::W64), c(1, Width::W64))], vec![0x1010]),
                block(
                    0x1010,
                    vec![
                        assign(
                            ra(0, Width::W64),
                            bin(BinOp::Add, read(ra(0, Width::W64)), c(1, Width::W64)),
                        ),
                        set_flag(0),
                        jcc(0x1010),
                    ],
                    vec![0x1010, 0x1020],
                ),
                block(0x1020, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, vars) = pipe(&f);
        let t = text(&ssa, &root, &vars);
        let s = shape(&t);
        let d = s
            .iter()
            .position(|l| l == "    do {")
            .unwrap_or_else(|| panic!("a do: {t}"));
        assert!(s[d + 1].contains("vN = vN + 0x1.q;"), "{t}");
        assert!(s[d + 3].contains("} while (vN);"), "{t}");
    }

    #[test]
    fn a_condition_free_loop_reads_while_true_and_exits_by_break() {
        // The exit test sits mid-body: neither header nor latch decides,
        // so the loop is `while (true)` and the exit is a break.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![assign(ra(1, Width::W64), c(1, Width::W64))], vec![0x1010]),
                block(0x1010, vec![set_flag(0), jcc(0x1030)], vec![0x1030, 0x1020]),
                block(0x1020, vec![jmp(0x1000)], vec![0x1000]),
                block(0x1030, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, vars) = pipe(&f);
        let t = text(&ssa, &root, &vars);
        let s = shape(&t);
        let w = s
            .iter()
            .position(|l| l == "    while (true) {")
            .unwrap_or_else(|| panic!("while true: {t}"));
        assert!(s[w..].iter().any(|l| l.trim() == "break;"), "{t}");
    }

    #[test]
    fn a_proven_jump_table_renders_as_a_switch_with_break_per_case() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![indirect()], vec![0x1010, 0x1020]),
                block(0x1010, vec![assign(ra(1, Width::W64), c(1, Width::W64))], vec![0x1030]),
                block(0x1020, vec![assign(ra(1, Width::W64), c(2, Width::W64))], vec![0x1030]),
                block(0x1030, vec![ret()], vec![]),
            ],
        );
        let tables = BTreeMap::from([(0x1008u64, vec![0x1010u64, 0x1020])]);
        let (ssa, root, vars) = pipe_tabled(&f, &tables);
        let t = text(&ssa, &root, &vars);
        let s = shape(&t);
        assert_eq!(
            &s[2..],
            [
                "    // live-in: vN, vN",
                "    switch (vN) {",
                "    case loc_1010:",
                "        vN = 0x1.q;",
                "        break;",
                "    case loc_1020:",
                "        vN = 0x2.q;",
                "        break;",
                "    }",
                "    return;",
                "}",
            ],
            "{t}"
        );
    }

    #[test]
    fn a_virtualized_edge_renders_as_a_goto_and_its_target_gets_a_label() {
        // Two-entry cycle: irreducible, so structuring virtualizes edges
        // into gotos; the pseudocode must label the targets. Both cycle
        // blocks carry conditions *and* stores, so neither the tail
        // re-split nor jump threading can buy the gotos back.
        let st = || Stmt::Store {
            addr: read(ra(5, Width::W64)),
            value: c(1, Width::W64),
        };
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1020)], vec![0x1020, 0x1010]),
                block(
                    0x1010,
                    vec![st(), set_flag(3), jcc(0x1020)],
                    vec![0x1020, 0x1030],
                ),
                block(
                    0x1020,
                    vec![st(), set_flag(2), jcc(0x1010)],
                    vec![0x1010, 0x1030],
                ),
                block(0x1030, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, vars) = pipe(&f);
        let t = text(&ssa, &root, &vars);
        assert!(t.contains("goto loc_"), "{t}");
        // Every rendered in-function goto target has a label.
        for l in t.lines() {
            let Some(pos) = l.find("goto loc_") else { continue };
            let target: String = l[pos + "goto loc_".len()..]
                .chars()
                .take_while(|c| c.is_ascii_hexdigit())
                .collect();
            assert!(
                t.contains(&format!("loc_{target}:")),
                "goto target loc_{target} must be labeled: {t}"
            );
        }
    }

    #[test]
    fn calls_and_intrinsics_render_with_targets_and_effect_writes() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    Stmt::Branch {
                        kind: BranchKind::Call,
                        cond: None,
                        target: c(0x2000, Width::W64),
                    },
                    ret(),
                ],
                vec![],
            )],
        );
        let abi = callfx::abi_for(Arch::X86_64).expect("x86-64 has an ABI");
        let lifted = callfx::apply(&f, &abi);
        let ssa = build(&lifted);
        let (root, _) = irstruct::structure(&ssa, &no_tables());
        let (vars, _) = irout::out_of_ssa(&ssa);
        let t = text(&ssa, &root, &vars);
        assert!(t.contains("call 0x2000();"), "{t}");
        assert!(t.contains("callfx(") && t.contains("/* writes "), "{t}");
        assert!(t.contains("/* abi-assumed: "), "{t}");
    }

    // -- 5: honesty markers --------------------------------------------------

    #[test]
    fn a_truncated_block_carries_the_lift_truncated_marker() {
        let mut b = block(0x1000, vec![assign(ra(0, Width::W64), c(1, Width::W64))], vec![0x1010]);
        b.truncated = true;
        let f = func(0x1000, vec![b, block(0x1010, vec![ret()], vec![])]);
        let (ssa, root, vars) = pipe(&f);
        let t = text(&ssa, &root, &vars);
        assert!(t.contains("/* lift truncated */"), "{t}");
        assert_eq!(t.matches("/* lift truncated */").count(), 1, "{t}");
    }

    #[test]
    fn an_unproven_indirect_jump_is_spelled_and_marked() {
        let f = func(0x1000, vec![block(0x1000, vec![indirect()], vec![])]);
        let (ssa, root, vars) = pipe(&f);
        let t = text(&ssa, &root, &vars);
        assert!(t.contains("goto *v"), "the jump itself must print: {t}");
        assert!(t.contains("/* indirect jump: successors unknown */"), "{t}");
    }

    #[test]
    fn undecidable_exits_are_marked_and_never_invented() {
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W64), c(1, Width::W64))],
                    vec![0x1010, 0x1020],
                ),
                block(0x1010, vec![ret()], vec![]),
                block(0x1020, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, vars) = pipe(&f);
        let t = text(&ssa, &root, &vars);
        assert!(t.contains("/* undecidable exits */"), "{t}");
    }

    #[test]
    fn a_partial_width_read_marks_its_variable() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W8), c(1, Width::W8)),
                    assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                    ret(),
                ],
                vec![],
            )],
        );
        let (ssa, root, vars) = pipe(&f);
        assert!(!ssa.partial.is_empty(), "the fixture must produce a partial read");
        let t = text(&ssa, &root, &vars);
        assert!(t.contains("/* reads bits its def never wrote: "), "{t}");
    }

    // -- 6: edge copies at the realized edge ---------------------------------

    #[test]
    fn a_coalesced_diamond_renders_zero_copies() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1020)], vec![0x1020, 0x1010]),
                block(0x1010, vec![assign(ra(1, Width::W64), c(7, Width::W64))], vec![0x1030]),
                block(0x1020, vec![assign(ra(1, Width::W64), c(9, Width::W64))], vec![0x1030]),
                block(
                    0x1030,
                    vec![assign(ra(2, Width::W64), read(ra(1, Width::W64))), ret()],
                    vec![],
                ),
            ],
        );
        let (ssa, root, vars) = pipe(&f);
        assert!(vars.edge_copies.is_empty(), "the phi web must coalesce");
        let t = text(&ssa, &root, &vars);
        assert!(!t.contains(" -> "), "no copy margins: {t}");
        assert!(!t.contains("unplaced"), "{t}");
    }

    /// The lost-copy shape on a natural while loop: the header computes
    /// `rcx := rax + 5` over the phi, the latch increments `rax` and
    /// *then* reads `rcx` — which expression forwarding splices into a
    /// read of the phi, so the phi is live just after the back-edge
    /// value's definition and the two interfere (Briggs et al. 1998,
    /// reached through the real passes exactly as `irout` documents).
    /// The surviving copy sits on the back edge, which leaves an
    /// unconditional latch — so it has an honest textual site, the
    /// bottom of the loop body.
    fn while_lost_copy() -> irlift::LiftedFunction {
        func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(2, Width::W64), c(1, Width::W64))],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![
                        assign(
                            ra(1, Width::W64),
                            bin(BinOp::Add, read(ra(0, Width::W64)), c(5, Width::W64)),
                        ),
                        set_flag(0),
                        jcc(0x1030),
                    ],
                    vec![0x1030, 0x1020],
                ),
                block(
                    0x1020,
                    vec![
                        assign(
                            ra(0, Width::W64),
                            bin(BinOp::Add, read(ra(0, Width::W64)), c(1, Width::W64)),
                        ),
                        assign(ra(3, Width::W64), read(ra(1, Width::W64))),
                        jmp(0x1010),
                    ],
                    vec![0x1010],
                ),
                block(0x1030, vec![ret()], vec![]),
            ],
        )
    }

    #[test]
    fn a_back_edge_copy_lands_at_the_bottom_of_the_loop_body() {
        let (ssa, root, vars) = opt_pipe(&while_lost_copy());
        assert!(
            vars.edge_copies.contains_key(&(0x1020, 0x1010)),
            "the lost-copy shape must leave a back-edge copy: {vars:?}"
        );
        let t = text(&ssa, &root, &vars);
        assert!(!t.contains("unplaced"), "the while back edge has a site: {t}");
        // The copy sits inside the loop, after the latch's last
        // statement, directly before the loop closes.
        let lines: Vec<&str> = t.lines().collect();
        let copy = lines
            .iter()
            .position(|l| l.contains("// 0x1020 -> 0x1010"))
            .unwrap_or_else(|| panic!("the placed copy: {t}"));
        assert!(lines[copy].trim_start().starts_with("v"), "{t}");
        assert_eq!(lines[copy + 1].trim(), "}", "{t}");
    }

    #[test]
    fn a_do_while_back_edge_copy_is_reported_unplaced_never_dropped() {
        // The same interference with the test at the latch: the back
        // edge leaves a two-way block, and `do { } while ()` has no
        // textual site for its copies. Honesty over silent loss.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![assign(ra(2, Width::W64), c(1, Width::W64))], vec![0x1010]),
                block(
                    0x1010,
                    vec![
                        assign(
                            ra(3, Width::W64),
                            bin(BinOp::Add, read(ra(0, Width::W64)), c(5, Width::W64)),
                        ),
                        assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                    ],
                    vec![0x1020],
                ),
                block(
                    0x1020,
                    vec![
                        assign(
                            ra(0, Width::W64),
                            bin(BinOp::Add, read(ra(0, Width::W64)), c(1, Width::W64)),
                        ),
                        set_flag(0),
                        jcc(0x1010),
                    ],
                    vec![0x1010, 0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(6, Width::W64), read(ra(1, Width::W64))), ret()],
                    vec![],
                ),
            ],
        );
        let (ssa, root, vars) = opt_pipe(&f);
        assert!(
            vars.edge_copies.keys().any(|&(p, _)| p == 0x1020),
            "the latch must carry a copy: {vars:?}"
        );
        let t = text(&ssa, &root, &vars);
        assert!(t.contains("/* unplaced edge copies loc_1020 -> "), "{t}");
    }

    #[test]
    fn entry_copies_render_as_a_prologue_before_the_body() {
        let f = func(0x1000, vec![block(0x1000, vec![ret()], vec![])]);
        let (ssa, root, mut vars) = pipe(&f);
        vars.entry_copies = vec![Copy {
            dst: CopySlot::Var(0),
            src: CopySlot::Temp,
        }];
        let t = text(&ssa, &root, &vars);
        let lines: Vec<&str> = t.lines().collect();
        let prologue = lines
            .iter()
            .position(|l| l.contains("v0 = tmp;"))
            .unwrap_or_else(|| panic!("prologue: {t}"));
        let body = lines.iter().position(|l| l.contains("return;")).expect("body");
        assert!(prologue < body, "{t}");
        assert!(lines[prologue].contains("// entry"), "{t}");
    }

    #[test]
    fn a_resplit_tail_with_a_phi_places_each_edge_copy_exactly_once() {
        // The SAILR tail re-split rider: two arms cross into a shared
        // `ret` tail whose φ demands residual copies. The structurer
        // virtualizes the pass-through arm's tail edge, the re-split
        // duplicates the tail into it, and so the duplicate leaf
        // realizes a *different* incoming edge than the original — the
        // pending-set walk must still place every copy exactly once per
        // realized edge, none dropped, none doubled.
        //
        // Entry 0x1040 keeps the merged tail's edge the lowest stuck
        // edge (into the pass-through arm 0x1000). `r3 := r0` at entry
        // aliases the live-in value; forwarding splices the alias's
        // reads back into `r0`'s entry version, so the tail reading the
        // alias refuses that version a place in the φ web (a copy on
        // the duplicated edge 0x1000 -> 0x1020), and the redefining
        // arm reading it after `r0 := 6` refuses the arm's version a
        // place in the second tail's web (a copy on the untaken edge
        // 0x1010 -> 0x1030) — the split-family interference irout
        // documents, reached through the real passes.
        let f = func(
            0x1040,
            vec![
                block(
                    0x1040,
                    vec![
                        assign(ra(3, Width::W64), read(ra(0, Width::W64))),
                        set_flag(1),
                        jcc(0x1000),
                    ],
                    vec![0x1000, 0x1010],
                ),
                block(0x1000, vec![set_flag(3), jcc(0x1020)], vec![0x1020, 0x1030]),
                block(
                    0x1010,
                    vec![
                        assign(ra(0, Width::W64), c(6, Width::W64)),
                        assign(ra(1, Width::W64), read(ra(3, Width::W64))),
                        set_flag(2),
                        jcc(0x1020),
                    ],
                    vec![0x1020, 0x1030],
                ),
                block(
                    0x1020,
                    vec![
                        assign(
                            ra(2, Width::W64),
                            bin(BinOp::Add, read(ra(0, Width::W64)), read(ra(3, Width::W64))),
                        ),
                        ret(),
                    ],
                    vec![],
                ),
                block(
                    0x1030,
                    vec![assign(ra(4, Width::W64), read(ra(0, Width::W64))), ret()],
                    vec![],
                ),
            ],
        );
        let (ssa, root, vars) = opt_pipe(&f);
        let (_, stats) = irstruct::structure(&ssa, &no_tables());
        assert_eq!(
            (stats.gotos, stats.duplications),
            (0, 1),
            "the re-split must buy the merged tail's goto back"
        );
        assert!(
            vars.edge_copies.contains_key(&(0x1000, 0x1020)),
            "the duplicated edge must carry a φ copy: {vars:?}"
        );
        assert!(
            vars.edge_copies.contains_key(&(0x1010, 0x1030)),
            "the redefining arm's untaken edge must carry one too: {vars:?}"
        );
        assert_eq!(vars.edge_copies.len(), 2, "{vars:?}");
        let t = text(&ssa, &root, &vars);
        assert!(!t.contains("goto"), "{t}");
        assert!(!t.contains("loc_1020:"), "a duplicated tail is never a label: {t}");
        assert!(!t.contains("unplaced"), "{t}");
        // Exactly once per realized edge: the duplicated edge's copy at
        // the duplicate leaf, the untaken edge's in a synthesized else.
        assert_eq!(t.matches("// 0x1000 -> 0x1020").count(), 1, "{t}");
        assert_eq!(t.matches("// 0x1010 -> 0x1030").count(), 1, "{t}");
        // Both occurrences of the tail render the block's own statements
        // byte-for-byte.
        assert_eq!(t.matches("return;").count(), 3, "{t}");
        let tail: Vec<String> = code_lines(&t)
            .into_iter()
            .zip(t.lines())
            .filter(|(_, raw)| raw.contains("// 0x1020"))
            .map(|(code, _)| code.trim().to_string())
            .collect();
        assert!(!tail.is_empty() && tail.len().is_multiple_of(2), "{t}");
        let (dup_site, original) = tail.split_at(tail.len() / 2);
        assert_eq!(dup_site, original, "the duplicate must render byte-equal: {t}");
    }

    #[test]
    fn a_tail_whose_outgoing_edge_carries_a_copy_keeps_one_textual_site() {
        // Inversion two's φ rider: the shared tail 0x1020 converges on
        // the join 0x1050, whose φ takes the *entry* version of r0 on
        // that edge — and the join also reads the alias r3 (spliced back
        // to that same entry version), so the version interferes with
        // the φ web and a residual copy lands on the tail's outgoing
        // edge (0x1020 -> 0x1050). A duplicate would realize that edge
        // at a second textual site the walk cannot serve, so the
        // re-split refuses: the goto and its label survive, and the
        // copy is placed exactly once, none dropped, none doubled.
        let f = func(
            0x1040,
            vec![
                block(
                    0x1040,
                    vec![
                        assign(ra(3, Width::W64), read(ra(0, Width::W64))),
                        set_flag(1),
                        jcc(0x1000),
                    ],
                    vec![0x1000, 0x1010],
                ),
                block(0x1000, vec![set_flag(3), jcc(0x1020)], vec![0x1020, 0x1030]),
                block(0x1010, vec![set_flag(2), jcc(0x1020)], vec![0x1020, 0x1030]),
                block(
                    0x1020,
                    vec![assign(ra(2, Width::W64), c(5, Width::W64)), jmp(0x1050)],
                    vec![0x1050],
                ),
                block(
                    0x1030,
                    vec![assign(ra(0, Width::W64), c(8, Width::W64)), jmp(0x1050)],
                    vec![0x1050],
                ),
                block(
                    0x1050,
                    vec![
                        assign(
                            ra(5, Width::W64),
                            bin(BinOp::Add, read(ra(0, Width::W64)), read(ra(3, Width::W64))),
                        ),
                        ret(),
                    ],
                    vec![],
                ),
            ],
        );
        let (ssa, root, vars) = opt_pipe(&f);
        assert!(
            vars.edge_copies.contains_key(&(0x1020, 0x1050)),
            "the tail's outgoing edge must carry a residual copy: {vars:?}"
        );
        // The sibling tail 0x1030's edge proves copy-free and may
        // split; 0x1020's may not — the copy has exactly one textual
        // placement, so its goto and label survive whatever happens
        // around them.
        let t = text(&ssa, &root, &vars);
        assert!(t.contains("goto loc_1020"), "{t}");
        assert_eq!(t.matches("loc_1020:").count(), 1, "one label, once: {t}");
        assert_eq!(
            t.matches("v5 = 0x5.q").count(),
            1,
            "the copy-carrying tail is never duplicated: {t}"
        );
        assert!(!t.contains("unplaced"), "{t}");
        assert_eq!(t.matches("// 0x1020 -> 0x1050").count(), 1, "{t}");
    }

    #[test]
    fn resplit_case_bodies_in_a_loop_spell_their_convergence_as_continue() {
        // The bash shape: a switch inside a loop whose case bodies cross
        // into two shared tails, both converging on the header. The
        // dispatch degrades (the cases do not converge on one follow),
        // and the re-split duplicates each case's chain into its case,
        // spelling the open edge to the header as `continue;` —
        // C-correct inside a case.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1070)], vec![0x1070, 0x1010]),
                block(
                    0x1010,
                    vec![indirect()],
                    vec![0x1020, 0x1030, 0x1040, 0x1048],
                ),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), c(1, Width::W64))],
                    vec![0x1050],
                ),
                block(
                    0x1030,
                    vec![assign(ra(2, Width::W64), c(2, Width::W64))],
                    vec![0x1050],
                ),
                block(
                    0x1040,
                    vec![assign(ra(3, Width::W64), c(3, Width::W64))],
                    vec![0x1060],
                ),
                block(
                    0x1048,
                    vec![assign(ra(6, Width::W64), c(6, Width::W64))],
                    vec![0x1060],
                ),
                block(
                    0x1050,
                    vec![assign(ra(4, Width::W64), c(4, Width::W64)), jmp(0x1000)],
                    vec![0x1000],
                ),
                block(
                    0x1060,
                    vec![assign(ra(5, Width::W64), c(5, Width::W64)), jmp(0x1000)],
                    vec![0x1000],
                ),
                block(0x1070, vec![ret()], vec![]),
            ],
        );
        let tables = BTreeMap::from([(0x1018u64, vec![0x1020u64, 0x1030, 0x1040, 0x1048])]);
        let (ssa, root, vars) = pipe_tabled(&f, &tables);
        let (_, stats) = irstruct::structure(&ssa, &tables);
        assert!(stats.duplications > 0, "the case tails must re-split");
        assert_eq!(stats.gotos, 0, "every case-body goto bought back");
        let t = text(&ssa, &root, &vars);
        assert!(!t.contains("goto"), "{t}");
        assert!(t.contains("switch"), "{t}");
        assert!(t.contains("case loc_1020:"), "{t}");
        // Each rewritten case ends by continuing the enclosing loop.
        assert!(t.matches("continue;").count() >= 4, "{t}");
    }

    #[test]
    fn a_threaded_condition_renders_at_both_sites_with_its_polarity() {
        // Inversion three's rider: the merged tail carries a condition,
        // so the thread duplicates the deciding block itself. The
        // renderer fetches the guard from the one block both `If`s
        // reference, so the condition text — negation polarity included
        // — is byte-identical at the original and the copy, and the
        // bought-back goto leaves no label behind.
        let f = func(
            0x1040,
            vec![
                block(0x1040, vec![set_flag(1), jcc(0x1000)], vec![0x1000, 0x1010]),
                block(0x1000, vec![set_flag(2), jcc(0x1020)], vec![0x1020, 0x1030]),
                block(0x1010, vec![set_flag(3), jcc(0x1020)], vec![0x1020, 0x1030]),
                block(0x1020, vec![set_flag(7), jcc(0x1030)], vec![0x1030, 0x1050]),
                block(0x1030, vec![ret()], vec![]),
                block(0x1050, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, vars) = pipe(&f);
        let (_, stats) = irstruct::structure(&ssa, &no_tables());
        assert_eq!(
            (stats.gotos, stats.threaded),
            (0, 1),
            "the thread must buy the deciding block's goto back"
        );
        let t = text(&ssa, &root, &vars);
        assert!(!t.contains("goto"), "{t}");
        assert!(!t.contains("loc_1020:"), "a threaded block is never a label: {t}");
        // Both occurrences spell the same negated guard, fetched from
        // the one deciding block at print time.
        let ifs: Vec<String> = code_lines(&t)
            .into_iter()
            .zip(t.lines())
            .filter(|(code, raw)| code.trim_start().starts_with("if (!") && raw.contains("// 0x1020"))
            .map(|(code, _)| code.trim().to_string())
            .collect();
        assert_eq!(ifs.len(), 2, "{t}");
        assert_eq!(ifs[0], ifs[1], "the copies' conditions must render byte-equal: {t}");
        // And the deciding block's own statements render at both sites:
        // the guard's defining assignment appears once per occurrence.
        assert_eq!(
            t.lines()
                .filter(|l| l.contains("// 0x1020") && l.contains("=="))
                .count(),
            2,
            "one guard definition at each of the two sites: {t}"
        );
    }

    #[test]
    fn a_threaded_latch_renders_continue_and_break_at_both_sites() {
        // The canonical threaded loop shape: the copy's `If` realizes
        // the back edge as `continue;` and the open side as `break;`,
        // with the un-negated guard — at both textual sites.
        let f = func(
            0x1400,
            vec![
                block(
                    0x1400,
                    vec![assign(ra(0, Width::W64), c(0, Width::W64)), jmp(0x1100)],
                    vec![0x1100],
                ),
                block(0x1100, vec![set_flag(1), jcc(0x2000)], vec![0x2000, 0x1040]),
                block(0x1040, vec![set_flag(2), jcc(0x1000)], vec![0x1000, 0x1010]),
                block(0x1000, vec![set_flag(3), jcc(0x1020)], vec![0x1020, 0x1030]),
                block(0x1010, vec![set_flag(4), jcc(0x1020)], vec![0x1020, 0x1030]),
                block(0x1020, vec![set_flag(7), jcc(0x1100)], vec![0x1100, 0x2000]),
                block(
                    0x1030,
                    vec![assign(ra(1, Width::W64), c(1, Width::W64)), jmp(0x1100)],
                    vec![0x1100],
                ),
                block(0x2000, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, vars) = pipe(&f);
        let (_, stats) = irstruct::structure(&ssa, &no_tables());
        assert_eq!((stats.gotos, stats.threaded), (0, 1));
        let t = text(&ssa, &root, &vars);
        assert!(!t.contains("goto"), "{t}");
        let sites: Vec<Vec<String>> = {
            let lines = code_lines(&t);
            lines
                .iter()
                .enumerate()
                .filter(|(i, l)| {
                    l.trim_start().starts_with("if (")
                        && t.lines().nth(*i).is_some_and(|raw| raw.contains("// 0x1020"))
                })
                .map(|(i, _)| {
                    lines[i..i + 4].iter().map(|l| l.trim().to_string()).collect()
                })
                .collect()
        };
        assert_eq!(sites.len(), 2, "{t}");
        assert_eq!(sites[0], sites[1], "{t}");
        assert!(sites[0][1] == "continue;", "{t}");
        assert!(sites[0][3] == "break;", "{t}");
    }

    // -- 7: total on malformed input -----------------------------------------

    #[test]
    fn a_tree_naming_a_foreign_block_is_marked_not_panicked() {
        let f = func(0x1000, vec![block(0x1000, vec![ret()], vec![])]);
        let (ssa, _, vars) = pipe(&f);
        let root = Node::Seq(vec![Node::Block(0x1000), Node::Block(0xdead)]);
        let t = text(&ssa, &root, &vars);
        assert!(t.contains("/* missing block loc_dead */"), "{t}");
    }

    #[test]
    fn an_if_over_a_non_conditional_block_gets_an_honest_marker() {
        let f = func(0x1000, vec![block(0x1000, vec![ret()], vec![])]);
        let (ssa, _, vars) = pipe(&f);
        let root = Node::If {
            cond: Cond {
                block: 0x1000,
                negated: false,
            },
            then_body: Box::new(Node::Block(0x1000)),
            else_body: None,
        };
        let t = text(&ssa, &root, &vars);
        assert!(t.contains("if (/* no condition: loc_1000 */) {"), "{t}");
    }

    #[test]
    fn a_malformed_function_still_renders_through_the_identity_posture() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![assign(ra(0, Width::W64), c(1, Width::W64)), ret()],
                vec![],
            )],
        );
        let mut ssa = build(&f);
        ssa.names.clear(); // now fails irssa::check
        assert!(irssa::check(&ssa).is_err());
        let (root, _) = irstruct::structure(&ssa, &no_tables());
        let (vars, _) = irout::out_of_ssa(&ssa);
        let t = text(&ssa, &root, &vars);
        assert!(t.contains("v?"), "unmapped names print the total fallback: {t}");
    }

    #[test]
    fn a_too_deep_tree_truncates_with_a_marker() {
        let f = func(0x1000, vec![block(0x1000, vec![ret()], vec![])]);
        let (ssa, _, vars) = pipe(&f);
        let mut root = Node::Block(0x1000);
        for _ in 0..(irstruct::MAX_TREE_DEPTH + 8) {
            root = Node::Seq(vec![root]);
        }
        let t = text(&ssa, &root, &vars);
        assert!(t.contains("/* tree too deep, truncated */"), "{t}");
    }

    fn gen_node(rng: &mut u64, vas: &[u64], depth: usize) -> Node {
        let pick = |r: &mut u64| vas[(next(r) % vas.len() as u64) as usize];
        let r = next(rng);
        if depth == 0 {
            return match r % 5 {
                0 => Node::Break,
                1 => Node::Continue,
                2 => Node::Goto(pick(rng)),
                3 => Node::Opaque {
                    block: pick(rng),
                    reason: OpaqueReason::Truncated,
                },
                _ => Node::Block(pick(rng)),
            };
        }
        match r % 6 {
            0 => Node::Seq(
                (0..(next(rng) % 3))
                    .map(|_| gen_node(rng, vas, depth - 1))
                    .collect(),
            ),
            1 => Node::If {
                cond: Cond {
                    block: pick(rng),
                    negated: next(rng).is_multiple_of(2),
                },
                then_body: Box::new(gen_node(rng, vas, depth - 1)),
                else_body: if next(rng).is_multiple_of(2) {
                    Some(Box::new(gen_node(rng, vas, depth - 1)))
                } else {
                    None
                },
            },
            2 => Node::Loop {
                kind: [LoopKind::SelfLoop, LoopKind::While, LoopKind::DoWhile]
                    [(next(rng) % 3) as usize],
                cond: if next(rng).is_multiple_of(2) {
                    Some(Cond {
                        block: pick(rng),
                        negated: false,
                    })
                } else {
                    None
                },
                body: Box::new(gen_node(rng, vas, depth - 1)),
            },
            3 => Node::Switch {
                block: pick(rng),
                cases: (0..(next(rng) % 3))
                    .map(|_| (pick(rng), gen_node(rng, vas, depth - 1)))
                    .collect(),
            },
            4 => Node::Goto(pick(rng)),
            _ => Node::Block(pick(rng)),
        }
    }

    #[test]
    fn hand_broken_trees_never_panic_and_render_deterministically() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1020)], vec![0x1020, 0x1010]),
                block(0x1010, vec![jmp(0x1020)], vec![0x1020]),
                block(0x1020, vec![ret()], vec![]),
            ],
        );
        let (ssa, _, vars) = pipe(&f);
        let vas = [0x1000u64, 0x1010, 0x1020, 0xdead, 0];
        let mut rng = 0x0DDB_A115_EED0u64;
        for _ in 0..200 {
            let root = gen_node(&mut rng, &vas, 4);
            // `text` asserts determinism; the loop asserts totality.
            let _ = text(&ssa, &root, &vars);
        }
    }

    #[test]
    fn rendering_never_mutates_its_inputs() {
        let (ssa, root, vars) = opt_pipe(&while_lost_copy());
        let (ssa2, root2, vars2) = (ssa.clone(), root.clone(), vars.clone());
        let a = render(&ssa, &root, &vars);
        let b = render(&ssa, &root, &vars);
        assert_eq!(a, b);
        assert_eq!(ssa, ssa2);
        assert_eq!(root, root2);
        assert_eq!(vars, vars2);
    }

    #[test]
    fn the_namer_hook_overrides_the_default_spelling() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![assign(ra(0, Width::W64), c(1, Width::W64)), ret()],
                vec![],
            )],
        );
        let (ssa, root, vars) = pipe(&f);
        let t = render_with(&ssa, &root, &vars, &|v| Some(format!("local_{v}")));
        assert!(t.contains("local_"), "{t}");
        assert!(!code_lines(&t).iter().any(|l| l.contains(" v0 ")), "{t}");
    }
}
