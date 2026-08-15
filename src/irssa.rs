//! Pruned SSA construction over a lifted IR control-flow graph.
//!
//! [`crate::irlift`] produces a function as basic blocks of straight-line
//! [`crate::ir`] statements; within a block, dataflow is explicit, but a
//! value that crosses a block boundary is still identified only by its
//! register name, so "which definition does this read see" needs a
//! whole-CFG analysis every time it is asked. This module answers it once:
//! [`construct`] renames every definition to a fresh *SSA name* — one
//! definition per name — and inserts φ-nodes where control-flow joins
//! merge distinct definitions, in *pruned* form (a φ is placed only where
//! the merged value is actually live, per the liveness [`crate::irflow`]
//! computes). The result is the def-use form every later decompiler pass
//! (expression forwarding, value numbering, structuring) wants to consume.
//!
//! # Representation
//!
//! SSA statements reuse [`ir::Stmt`]/[`ir::Expr`] unchanged: renaming
//! rewrites only [`ir::Reg::num`], which becomes an SSA name id indexing
//! [`SsaFunction::names`]. Width discipline is therefore untouched — every
//! renamed block still passes [`ir::check`] — and rendering reuses the IR
//! printer through [`ir::render_with`]. The cost is a cap of 65536 names
//! per function (`num` is `u16`); a function too large to name is an
//! explicit [`Unrepresentable::TooManyNames`], never a silent wrap.
//! φ-nodes are a separate per-block list ([`Phi`]); an argument keyed
//! `None` labels the virtual function-entry edge, which exists only when
//! the entry block has real predecessors (a loop back to entry).
//!
//! # Versioning semantics (aliasing doctrine)
//!
//! The IR models `eax`/`rax` as one *cell* at different widths. Matching
//! [`crate::irflow`]'s conservative doctrine, versions are per cell: a
//! definition at **any** width starts a new version of the whole cell.
//! Each name records the width its definition wrote — the low bits it is
//! guaranteed to carry. A use at that width or narrower is an exact
//! dependence (a sub-register read of low bits, per the ISA numbering
//! convention the lifters follow); a use *wider* than its reaching
//! definition reads bits that definition never wrote, and is recorded
//! honestly in [`SsaFunction::partial`] rather than silently claimed
//! exact. The x86 lifter writes general registers at `W64` and flags at
//! `W1`, so `partial` is empty on all lifted code; only hand-built IR can
//! populate it. Version 0 of a cell is the implicit at-entry value
//! (an argument, caller state), created on first unresolved use and
//! listed in [`SsaFunction::live_in`]; the entry state defines every bit
//! of a cell, so version 0 carries the full width. A φ carries the
//! minimum of its arguments' widths — the bits all inputs guarantee.
//!
//! # What is deliberately not modeled
//!
//! A [`ir::BranchKind::Call`]'s runtime clobbers are modeled only when
//! the input has been through [`crate::callfx::apply`] — as
//! `redump --ssa` does — which spells them out as ordinary intrinsic
//! writes and reads this module renames like any other definition and
//! use, no special case needed. On a raw lift the old caveat stands
//! verbatim: clobbers are **not** modeled, and def-use links reflect
//! the IR statements as written, exactly as [`irflow::liveness`] does.
//! An emergent bonus of the callfx form: argument registers the call
//! intrinsic reads with no prior definition become version-0
//! [`SsaFunction::live_in`] entries, so a function's rendered live-in
//! line starts to resemble its parameter list.
//!
//! # Contract
//!
//! [`construct`] is deterministic and total: equal inputs produce equal
//! output, no input panics, and every success passes [`check`]. Inputs it
//! cannot represent — a block failing [`ir::check`], or more definitions
//! than the name space holds — are explicit [`Unrepresentable`] errors.
//! Blocks unreachable from the entry take no part in SSA and are listed
//! in [`SsaFunction::skipped`]. [`check`] is total and side-effect-free:
//! it revalidates every invariant from scratch (dominance included) and
//! reports the first [`SsaFault`]. [`render`] is deterministic and
//! `\n`-terminated, in the style of [`irlift::render`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::ir::{self, Expr, Reg, Space, Stmt, Width};
use crate::model::Arch;
use crate::{aarch64_lift, irflow, irlift, x86_lift};

// ---------------------------------------------------------------------------
// The SSA form
// ---------------------------------------------------------------------------

/// A storage cell: a register/flag/temporary identity with the width
/// stripped. SSA versions are per cell — see the module docs.
pub type Cell = (Space, u16);

/// One SSA name: a version of a cell, with the width its definition is
/// guaranteed to carry. The name's id is its index in
/// [`SsaFunction::names`], and is what a renamed [`ir::Reg::num`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Name {
    /// The cell's namespace.
    pub space: Space,
    /// The cell's identity within `space` (the pre-SSA `num`).
    pub cell: u16,
    /// Version of the cell: 0 is the implicit at-entry value; definitions
    /// count up from 1 in renaming order.
    pub version: u32,
    /// The low bits this name is guaranteed to carry: the definition's
    /// width, the minimum argument width for a φ, or the full cell width
    /// for version 0.
    pub width: Width,
}

/// A φ-node: the merge of one cell's versions at a control-flow join.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi {
    /// The SSA name this φ defines.
    pub dst: u16,
    /// One argument per predecessor edge, `None` (the virtual
    /// function-entry edge, entry block only) first, then ascending
    /// predecessor VA. Each argument is the version of the φ's cell
    /// reaching the end of that predecessor.
    pub args: Vec<(Option<u64>, u16)>,
}

/// One basic block in SSA form: the φ-nodes at its head, then the block's
/// statements with every register occurrence renamed to an SSA name id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaBlock {
    /// VA of the first instruction (the lifted block's start).
    pub start: u64,
    /// VA one past the last byte (exclusive), carried from the lift.
    pub end: u64,
    /// φ-nodes, in ascending cell order (at most one per cell).
    pub phis: Vec<Phi>,
    /// The renamed statements. Shape and widths are exactly the lifted
    /// block's; only [`ir::Reg::num`] differs.
    pub stmts: Vec<Stmt>,
    /// Successor VAs, copied verbatim from the lifted block (out-of-
    /// function targets included; only in-function edges carry dataflow).
    pub successors: Vec<u64>,
    /// The lift stopped early in this block ([`irlift::LiftedBlock`]);
    /// carried through so the honest marker is never lost.
    pub truncated: bool,
}

/// One function in pruned SSA form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaFunction {
    /// Entry VA.
    pub entry: u64,
    /// Name carried over from the lifted function.
    pub name: Option<String>,
    /// Architecture carried over from the lifted function
    /// ([`irlift::LiftedFunction::arch`]): pure data that selects the
    /// [`Space::Arch`] register speller in [`render`]/[`display`], so an
    /// aarch64 function prints `x0`/`sp`, not `rax`. Every pass is
    /// arch-blind; only rendering reads it.
    pub arch: Arch,
    /// Blocks reachable from the entry, keyed by start VA.
    pub blocks: BTreeMap<u64, SsaBlock>,
    /// Start VAs of lifted blocks unreachable from the entry: excluded
    /// from SSA (dominance is undefined for them) and said so.
    pub skipped: Vec<u64>,
    /// The name table; a renamed [`ir::Reg::num`] indexes it.
    pub names: Vec<Name>,
    /// Ids of the version-0 names — cells read before any definition on
    /// some path, i.e. the values the function receives. Ascending.
    pub live_in: Vec<u16>,
    /// Positions `(block VA, statement index)` holding at least one use
    /// wider than its reaching definition — a read of bits that
    /// definition never wrote. Empty on lifter output (see module docs);
    /// exact by [`check`]. Ascending.
    pub partial: Vec<(u64, usize)>,
}

/// Why a lifted function has no SSA form. Both cases are explicit
/// refusals, never guesses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unrepresentable {
    /// A reachable block fails [`ir::check`]; renaming malformed IR would
    /// launder it into something that looks well-formed.
    MalformedBlock { block: u64, fault: ir::Fault },
    /// The function needs more SSA names than the `u16` id space holds.
    /// Call-effect modeling ([`crate::callfx::apply`]) raises the
    /// pressure: an x86-64 call adds 15 definitions (14 intrinsic
    /// writes plus the rsp restore) and an aarch64 call 24, so the cap
    /// trips near ~4,000 calls in one function — still an explicit
    /// error, never a silent wrap.
    TooManyNames,
}

impl std::fmt::Display for Unrepresentable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unrepresentable::MalformedBlock { block, .. } => {
                write!(f, "block {block:#x} fails the IR check")
            }
            Unrepresentable::TooManyNames => {
                write!(f, "more than 65536 SSA names")
            }
        }
    }
}

/// Why an [`SsaFunction`] is not well-formed. [`check`] returns the first
/// fault, with enough context to locate it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsaFault {
    /// A non-empty block map without the entry block.
    MissingEntry,
    /// A stored block unreachable from the entry.
    Unreachable { block: u64 },
    /// A block's statements fail the IR core check.
    Ir { block: u64, fault: ir::Fault },
    /// A register occurrence naming no SSA name, or one of another space.
    BadName { block: u64, num: u16 },
    /// An SSA name defined more than once (or a version-0 name defined
    /// at all — the entry defines it implicitly).
    Redefined { name: u16 },
    /// A non-entry SSA name with no definition.
    Undefined { name: u16 },
    /// Two SSA names claiming the same (cell, version).
    DuplicateVersion { name: u16 },
    /// `live_in` is not exactly the version-0 names, ascending.
    LiveIn,
    /// A definition occurrence whose width differs from its name's.
    DefWidth { block: u64, name: u16 },
    /// `partial` does not match the recomputed wider-than-definition set.
    Partial,
    /// A φ with wrong arity, edge order, cell, or ordering for its block.
    PhiShape { block: u64, index: usize },
    /// A φ claiming more bits than one of its arguments guarantees.
    PhiWidth { block: u64, index: usize },
    /// A use not dominated by its definition.
    NotDominated { block: u64, name: u16 },
}

// ---------------------------------------------------------------------------
// CFG analysis: reachability, dominators, dominance frontiers
// ---------------------------------------------------------------------------

/// The derived control-flow facts SSA construction and [`check`] share:
/// reachable blocks, in-function edges, immediate dominators, and
/// dominance frontiers. Recomputed from scratch by both, so [`check`]
/// trusts nothing [`construct`] wrote.
///
/// Exposed `pub(crate)` for [`crate::irstruct`], which needs this analysis
/// over the block graph — and, with the edges reversed, over the reverse
/// graph for post-dominators — and for [`crate::irout`], whose interference
/// test is dominance-based: one Cooper–Harvey–Kennedy implementation for
/// the crate, not a copy per consumer.
pub(crate) struct Cfg {
    #[allow(dead_code)] // read by construct; irout uses `SsaFunction::entry`
    pub(crate) entry: u64,
    /// Reachable block -> in-function successors, deduplicated, in the
    /// block's stored edge order.
    pub(crate) succs: BTreeMap<u64, Vec<u64>>,
    /// Reachable block -> predecessors, ascending, deduplicated.
    pub(crate) preds: BTreeMap<u64, Vec<u64>>,
    /// Immediate dominators; the entry (whose dominator is the virtual
    /// function-entry edge) is absent.
    pub(crate) idom: BTreeMap<u64, u64>,
    /// Dominance frontiers (with the virtual-root convention, so a block
    /// that dominates a predecessor of the entry has the entry in its
    /// frontier — the self-looping-entry case).
    frontiers: BTreeMap<u64, BTreeSet<u64>>,
}

impl Cfg {
    /// Analyze `raw` (block VA -> successor list, possibly naming VAs the
    /// map does not hold) from `entry`. Blocks unreachable from `entry` —
    /// and `entry` itself when absent — take no part. Deterministic and
    /// total; every loop is bounded by the block count.
    pub(crate) fn analyze(entry: u64, raw: &BTreeMap<u64, Vec<u64>>) -> Cfg {
        // Reachability, filtering edges to blocks the map holds.
        let mut reach: BTreeSet<u64> = BTreeSet::new();
        if raw.contains_key(&entry) {
            let mut stack = vec![entry];
            reach.insert(entry);
            while let Some(v) = stack.pop() {
                let Some(list) = raw.get(&v) else { continue };
                for &s in list {
                    if raw.contains_key(&s) && reach.insert(s) {
                        stack.push(s);
                    }
                }
            }
        }

        // In-function edges, deduplicated in stored order; predecessors.
        let mut succs: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for &v in &reach {
            let mut seen = BTreeSet::new();
            let list: Vec<u64> = raw
                .get(&v)
                .map(Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .copied()
                .filter(|s| reach.contains(s) && seen.insert(*s))
                .collect();
            succs.insert(v, list);
        }
        let mut preds: BTreeMap<u64, Vec<u64>> = reach.iter().map(|&v| (v, Vec::new())).collect();
        for (&v, list) in &succs {
            for &s in list {
                if let Some(p) = preds.get_mut(&s) {
                    p.push(v);
                }
            }
        }
        for p in preds.values_mut() {
            p.sort_unstable();
        }

        // Reverse postorder by iterative DFS in stored edge order.
        let mut post: Vec<u64> = Vec::with_capacity(reach.len());
        if reach.contains(&entry) {
            let mut visited: BTreeSet<u64> = BTreeSet::from([entry]);
            let mut stack: Vec<(u64, usize)> = vec![(entry, 0)];
            while let Some(top) = stack.last_mut() {
                let v = top.0;
                let next = succs.get(&v).and_then(|l| l.get(top.1)).copied();
                match next {
                    Some(s) => {
                        top.1 += 1;
                        if visited.insert(s) {
                            stack.push((s, 0));
                        }
                    }
                    None => {
                        post.push(v);
                        stack.pop();
                    }
                }
            }
        }
        let rpo: Vec<u64> = post.iter().rev().copied().collect();
        let index: BTreeMap<u64, usize> = rpo.iter().enumerate().map(|(i, &v)| (v, i)).collect();

        // Immediate dominators: the Cooper–Harvey–Kennedy iteration over
        // RPO indices (smaller index = closer to the entry).
        const UNDEF: usize = usize::MAX;
        let n = rpo.len();
        let preds_idx: Vec<Vec<usize>> = rpo
            .iter()
            .map(|v| {
                preds
                    .get(v)
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(|p| index.get(p).copied())
                    .collect()
            })
            .collect();
        let mut idom = vec![UNDEF; n];
        if n > 0 {
            idom[0] = 0;
        }
        let mut changed = true;
        while changed {
            changed = false;
            for i in 1..n {
                let mut new = UNDEF;
                for &p in &preds_idx[i] {
                    if idom[p] == UNDEF {
                        continue; // not yet processed this round
                    }
                    new = if new == UNDEF { p } else { intersect(new, p, &idom) };
                }
                if new != UNDEF && idom[i] != new {
                    idom[i] = new;
                    changed = true;
                }
            }
        }
        let idom_map: BTreeMap<u64, u64> = (1..n)
            .filter(|&i| idom[i] != UNDEF)
            .map(|i| (rpo[i], rpo[idom[i]]))
            .collect();

        // Dominance frontiers: for each edge p -> b, walk p's dominator
        // chain up to (excluding) idom(b), adding b to each frontier. The
        // entry's idom is the virtual root (absent), so the walk for a
        // predecessor of the entry runs through the entry inclusive —
        // which is exactly what makes a self-looping entry get its φ.
        let mut frontiers: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
        for &b in &rpo {
            let stop = idom_map.get(&b).copied();
            let Some(plist) = preds.get(&b) else { continue };
            for &p in plist {
                let mut runner = p;
                loop {
                    if Some(runner) == stop {
                        break;
                    }
                    frontiers.entry(runner).or_default().insert(b);
                    if runner == entry {
                        break;
                    }
                    match idom_map.get(&runner) {
                        Some(&up) => runner = up,
                        None => break, // defensively unreachable
                    }
                }
            }
        }

        Cfg {
            entry,
            succs,
            preds,
            idom: idom_map,
            frontiers,
        }
    }

    /// Whether `a` strictly dominates `b`: a walk up `b`'s dominator
    /// chain, bounded by the block count.
    pub(crate) fn strictly_dominates(&self, a: u64, b: u64) -> bool {
        if a == b {
            return false;
        }
        let mut cur = b;
        for _ in 0..=self.succs.len() {
            match self.idom.get(&cur) {
                Some(&d) if d == a => return true,
                Some(&d) => cur = d,
                None => return false, // reached the entry (virtual root above it)
            }
        }
        false
    }
}

/// The CHK two-finger intersection over RPO indices. Fuel-bounded so no
/// malformed intermediate state can loop; the fuel is never exhausted on
/// the well-formed chains the iteration actually produces.
fn intersect(mut a: usize, mut b: usize, idom: &[usize]) -> usize {
    let mut fuel = 2 * idom.len() + 2;
    while a != b {
        if fuel == 0 || a >= idom.len() || b >= idom.len() {
            return a.min(b); // defensively unreachable
        }
        fuel -= 1;
        if a > b {
            a = idom[a];
        } else {
            b = idom[b];
        }
    }
    a
}

// ---------------------------------------------------------------------------
// Statement walks
// ---------------------------------------------------------------------------

/// Visit every register occurrence in *use* position (an assign's value,
/// a store's operands, a branch's condition and target, an intrinsic's
/// reads — never an assign destination or intrinsic write). Shared with
/// [`crate::irssaopt`], which builds its def-use index from these walks
/// rather than duplicating them.
pub(crate) fn for_each_use(stmt: &Stmt, f: &mut dyn FnMut(Reg)) {
    match stmt {
        Stmt::Assign { value, .. } => expr_regs(value, 0, f),
        Stmt::Store { addr, value } => {
            expr_regs(addr, 0, f);
            expr_regs(value, 0, f);
        }
        Stmt::Branch { cond, target, .. } => {
            if let Some(c) = cond {
                expr_regs(c, 0, f);
            }
            expr_regs(target, 0, f);
        }
        Stmt::Intrinsic { reads, .. } => {
            for r in reads {
                expr_regs(r, 0, f);
            }
        }
    }
}

/// Visit every register occurrence in *definition* position, in
/// statement order.
pub(crate) fn for_each_def(stmt: &Stmt, f: &mut dyn FnMut(Reg)) {
    match stmt {
        Stmt::Assign { dst, .. } => f(*dst),
        Stmt::Intrinsic { writes, .. } => {
            for w in writes {
                f(*w);
            }
        }
        Stmt::Store { .. } | Stmt::Branch { .. } => {}
    }
}

/// Visit every register read in `e`, depth-bounded like the
/// [`crate::irflow`] walks.
pub(crate) fn expr_regs(e: &Expr, depth: usize, f: &mut dyn FnMut(Reg)) {
    if depth > ir::MAX_EXPR_NODES {
        return;
    }
    match e {
        Expr::Const { .. } => {}
        Expr::Reg(r) => f(*r),
        Expr::Load { addr, .. } => expr_regs(addr, depth + 1, f),
        Expr::Unary { operand, .. } => expr_regs(operand, depth + 1, f),
        Expr::Binary { lhs, rhs, .. } => {
            expr_regs(lhs, depth + 1, f);
            expr_regs(rhs, depth + 1, f);
        }
    }
}

/// The full width of a cell for its version-0 (at-entry) name: the entry
/// state defines every bit, and the IR core is ISA-blind, so this is the
/// widest the space allows.
fn full_width(space: Space) -> Width {
    match space {
        Space::Flag => Width::W1,
        Space::Arch | Space::Temp => Width::W64,
    }
}

// ---------------------------------------------------------------------------
// Liveness and φ placement
// ---------------------------------------------------------------------------

/// Cells live on entry to each reachable block: [`irflow::live_in`]
/// iterated to a backward fixpoint over the CFG, projected to cells.
/// `irflow`'s exact-reference kills make the register-level result a
/// sound over-approximation of the cell-level liveness the versioning
/// semantics need, so pruning by it never drops a required φ (at worst a
/// rare dead φ survives). Terminates: sets only grow, bounded by the
/// registers the function mentions.
fn cell_live_in(
    cfg: &Cfg,
    blocks: &BTreeMap<u64, &irlift::LiftedBlock>,
) -> BTreeMap<u64, BTreeSet<Cell>> {
    let mut live: BTreeMap<u64, BTreeSet<Reg>> =
        cfg.succs.keys().map(|&v| (v, BTreeSet::new())).collect();
    let mut work: BTreeSet<u64> = cfg.succs.keys().copied().collect();
    while let Some(b) = work.pop_last() {
        let mut out: BTreeSet<Reg> = BTreeSet::new();
        if let Some(list) = cfg.succs.get(&b) {
            for s in list {
                if let Some(set) = live.get(s) {
                    out.extend(set.iter().copied());
                }
            }
        }
        let Some(block) = blocks.get(&b) else { continue };
        let new_in = irflow::live_in(&block.stmts, &out);
        if live.get(&b) != Some(&new_in) {
            live.insert(b, new_in);
            if let Some(plist) = cfg.preds.get(&b) {
                work.extend(plist.iter().copied());
            }
        }
    }
    live.into_iter()
        .map(|(v, set)| (v, set.into_iter().map(|r| (r.space, r.num)).collect()))
        .collect()
}

/// Pruned φ placement: the classic iterated-dominance-frontier worklist
/// per cell, gated on the cell being live into the join. The entry block
/// counts as a definition site of every cell (the implicit version 0).
/// Returns, per block, the cells needing a φ, ascending.
fn place_phis(
    cfg: &Cfg,
    live: &BTreeMap<u64, BTreeSet<Cell>>,
    blocks: &BTreeMap<u64, &irlift::LiftedBlock>,
) -> BTreeMap<u64, Vec<Cell>> {
    let mut def_blocks: BTreeMap<Cell, BTreeSet<u64>> = BTreeMap::new();
    for (&va, block) in blocks {
        for stmt in &block.stmts {
            for_each_def(stmt, &mut |r| {
                def_blocks.entry((r.space, r.num)).or_default().insert(va);
            });
        }
    }
    let mut cells: BTreeSet<Cell> = def_blocks.keys().copied().collect();
    for set in live.values() {
        cells.extend(set.iter().copied());
    }

    let mut seeds: BTreeMap<u64, Vec<Cell>> =
        cfg.succs.keys().map(|&v| (v, Vec::new())).collect();
    for &cell in &cells {
        let mut sites = def_blocks.get(&cell).cloned().unwrap_or_default();
        sites.insert(cfg.entry);
        let mut placed: BTreeSet<u64> = BTreeSet::new();
        let mut work = sites.clone();
        while let Some(x) = work.pop_first() {
            let Some(frontier) = cfg.frontiers.get(&x) else { continue };
            for &y in frontier {
                if placed.contains(&y) {
                    continue;
                }
                if !live.get(&y).is_some_and(|s| s.contains(&cell)) {
                    continue; // pruned: the merged value is dead here
                }
                placed.insert(y);
                if !sites.contains(&y) {
                    work.insert(y); // a φ is itself a definition
                }
            }
        }
        for y in placed {
            if let Some(list) = seeds.get_mut(&y) {
                list.push(cell); // ascending: the outer loop is sorted
            }
        }
    }
    seeds
}

// ---------------------------------------------------------------------------
// Renaming
// ---------------------------------------------------------------------------

/// Name-allocation state shared across the dominator-tree walk.
#[derive(Default)]
struct Renamer {
    names: Vec<Name>,
    /// Per-cell stack of the names whose definitions dominate the point
    /// the walk is at. Version 0 is never on a stack; an empty stack
    /// means the at-entry value reaches.
    stacks: BTreeMap<Cell, Vec<u16>>,
    /// Next version per cell (definitions count from 1).
    counters: BTreeMap<Cell, u32>,
    /// Version-0 ids, created lazily on first unresolved use.
    entry_ids: BTreeMap<Cell, u16>,
}

impl Renamer {
    fn alloc(&mut self, cell: Cell, version: u32, width: Width) -> Result<u16, Unrepresentable> {
        let id = u16::try_from(self.names.len()).map_err(|_| Unrepresentable::TooManyNames)?;
        self.names.push(Name {
            space: cell.0,
            cell: cell.1,
            version,
            width,
        });
        Ok(id)
    }

    /// A fresh definition of `cell` at `width`, pushed onto its stack.
    fn def(&mut self, cell: Cell, width: Width) -> Result<u16, Unrepresentable> {
        let counter = self.counters.entry(cell).or_insert(1);
        let version = *counter;
        *counter += 1;
        let id = self.alloc(cell, version, width)?;
        self.stacks.entry(cell).or_default().push(id);
        Ok(id)
    }

    /// The version-0 (at-entry) name of `cell`, created on first demand.
    fn version0(&mut self, cell: Cell) -> Result<u16, Unrepresentable> {
        if let Some(&id) = self.entry_ids.get(&cell) {
            return Ok(id);
        }
        let id = self.alloc(cell, 0, full_width(cell.0))?;
        self.entry_ids.insert(cell, id);
        Ok(id)
    }

    /// The name a read of `cell` sees here: top of stack, else version 0.
    fn read(&mut self, cell: Cell) -> Result<u16, Unrepresentable> {
        match self.stacks.get(&cell).and_then(|s| s.last()) {
            Some(&id) => Ok(id),
            None => self.version0(cell),
        }
    }

    fn pop(&mut self, cell: Cell) {
        if let Some(s) = self.stacks.get_mut(&cell) {
            s.pop();
        }
    }

    /// Rewrite an expression's register reads to their reaching names.
    /// Recursion is bounded by the [`ir::check`]-enforced node cap of the
    /// pre-validated input.
    fn rename_expr(&mut self, e: &Expr) -> Result<Expr, Unrepresentable> {
        Ok(match e {
            Expr::Const { .. } => e.clone(),
            Expr::Reg(r) => Expr::Reg(Reg {
                space: r.space,
                num: self.read((r.space, r.num))?,
                width: r.width,
            }),
            Expr::Load { addr, width } => Expr::load(self.rename_expr(addr)?, *width),
            Expr::Unary { op, operand } => Expr::unary(*op, self.rename_expr(operand)?),
            Expr::Binary { op, lhs, rhs } => {
                Expr::binary(*op, self.rename_expr(lhs)?, self.rename_expr(rhs)?)
            }
        })
    }

    /// Rewrite one statement: uses read pre-statement versions, then each
    /// definition pushes a fresh name (recorded in `pushed` for the
    /// walk's exit undo).
    fn rename_stmt(&mut self, stmt: &Stmt, pushed: &mut Vec<Cell>) -> Result<Stmt, Unrepresentable> {
        Ok(match stmt {
            Stmt::Assign { dst, value } => {
                let value = self.rename_expr(value)?;
                let cell = (dst.space, dst.num);
                let id = self.def(cell, dst.width)?;
                pushed.push(cell);
                Stmt::Assign {
                    dst: Reg {
                        space: dst.space,
                        num: id,
                        width: dst.width,
                    },
                    value,
                }
            }
            Stmt::Store { addr, value } => Stmt::Store {
                addr: self.rename_expr(addr)?,
                value: self.rename_expr(value)?,
            },
            Stmt::Branch { kind, cond, target } => Stmt::Branch {
                kind: *kind,
                cond: cond.as_ref().map(|c| self.rename_expr(c)).transpose()?,
                target: self.rename_expr(target)?,
            },
            Stmt::Intrinsic {
                name,
                writes,
                reads,
            } => {
                let reads = reads
                    .iter()
                    .map(|r| self.rename_expr(r))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut renamed = Vec::with_capacity(writes.len());
                for w in writes {
                    let cell = (w.space, w.num);
                    let id = self.def(cell, w.width)?;
                    pushed.push(cell);
                    renamed.push(Reg {
                        space: w.space,
                        num: id,
                        width: w.width,
                    });
                }
                Stmt::Intrinsic {
                    name,
                    writes: renamed,
                    reads,
                }
            }
        })
    }
}

/// One dominator-tree walk frame: the block, the next child to descend
/// into, and the cells whose stacks this block pushed (popped on exit).
struct Frame {
    block: u64,
    child: usize,
    pushed: Vec<Cell>,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// Construct the pruned SSA form of a lifted function. Deterministic and
/// total; every `Ok` passes [`check`]. See the module docs for the
/// semantics and the two explicit [`Unrepresentable`] refusals.
pub fn construct(func: &irlift::LiftedFunction) -> Result<SsaFunction, Unrepresentable> {
    let raw: BTreeMap<u64, Vec<u64>> = func
        .blocks
        .iter()
        .map(|(&va, b)| (va, b.successors.clone()))
        .collect();
    let cfg = Cfg::analyze(func.entry, &raw);

    let reachable: BTreeMap<u64, &irlift::LiftedBlock> = func
        .blocks
        .iter()
        .filter(|(va, _)| cfg.succs.contains_key(va))
        .map(|(&va, b)| (va, b))
        .collect();
    let skipped: Vec<u64> = func
        .blocks
        .keys()
        .filter(|va| !cfg.succs.contains_key(va))
        .copied()
        .collect();

    // Renaming malformed IR would launder it; refuse instead.
    for (&va, block) in &reachable {
        ir::check(&block.stmts).map_err(|fault| Unrepresentable::MalformedBlock {
            block: va,
            fault,
        })?;
    }

    let live = cell_live_in(&cfg, &reachable);
    let seeds = place_phis(&cfg, &live, &reachable);

    // Dominator-tree children, ascending (BTreeMap iteration order).
    let mut children: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for (&b, &d) in &cfg.idom {
        children.entry(d).or_default().push(b);
    }

    let mut ren = Renamer::default();
    let mut renamed: BTreeMap<u64, Vec<Stmt>> = BTreeMap::new();
    let mut phi_dsts: BTreeMap<(u64, Cell), u16> = BTreeMap::new();
    let mut phi_args: BTreeMap<(u64, Cell), BTreeMap<Option<u64>, u16>> = BTreeMap::new();

    // Iterative pre-order walk of the dominator tree with per-frame undo,
    // so a long dominator chain cannot overflow the call stack.
    let enter = |b: u64,
                     ren: &mut Renamer,
                     renamed: &mut BTreeMap<u64, Vec<Stmt>>,
                     phi_dsts: &mut BTreeMap<(u64, Cell), u16>,
                     phi_args: &mut BTreeMap<(u64, Cell), BTreeMap<Option<u64>, u16>>|
     -> Result<Frame, Unrepresentable> {
        let mut pushed: Vec<Cell> = Vec::new();
        let block_seeds = seeds.get(&b).map(Vec::as_slice).unwrap_or(&[]);
        // The virtual function-entry edge carries the at-entry values.
        if b == cfg.entry {
            for &cell in block_seeds {
                let v0 = ren.version0(cell)?;
                phi_args.entry((b, cell)).or_default().insert(None, v0);
            }
        }
        // φ definitions precede the block's statements. The width starts
        // at the cell's full width and is lowered to the arguments'
        // minimum once every argument is known.
        for &cell in block_seeds {
            let id = ren.def(cell, full_width(cell.0))?;
            pushed.push(cell);
            phi_dsts.insert((b, cell), id);
        }
        let mut stmts = Vec::new();
        if let Some(block) = reachable.get(&b) {
            stmts.reserve(block.stmts.len());
            for stmt in &block.stmts {
                stmts.push(ren.rename_stmt(stmt, &mut pushed)?);
            }
        }
        renamed.insert(b, stmts);
        // This block's exit versions feed each successor's φ arguments.
        if let Some(list) = cfg.succs.get(&b) {
            for &s in list {
                for &cell in seeds.get(&s).map(Vec::as_slice).unwrap_or(&[]) {
                    let id = ren.read(cell)?;
                    phi_args.entry((s, cell)).or_default().insert(Some(b), id);
                }
            }
        }
        Ok(Frame {
            block: b,
            child: 0,
            pushed,
        })
    };

    if reachable.contains_key(&cfg.entry) {
        let mut stack = vec![enter(
            cfg.entry,
            &mut ren,
            &mut renamed,
            &mut phi_dsts,
            &mut phi_args,
        )?];
        while let Some(top) = stack.last_mut() {
            let next = children
                .get(&top.block)
                .and_then(|k| k.get(top.child))
                .copied();
            match next {
                Some(c) => {
                    top.child += 1;
                    let frame = enter(c, &mut ren, &mut renamed, &mut phi_dsts, &mut phi_args)?;
                    stack.push(frame);
                }
                None => {
                    let frame = stack.pop().unwrap_or(Frame {
                        block: 0,
                        child: 0,
                        pushed: Vec::new(),
                    });
                    for cell in frame.pushed {
                        ren.pop(cell);
                    }
                }
            }
        }
    }

    // Assemble the blocks: φs in cell order with one argument per
    // predecessor edge, in the canonical order.
    let mut blocks: BTreeMap<u64, SsaBlock> = BTreeMap::new();
    for (&va, block) in &reachable {
        let preds = cfg.preds.get(&va).map(Vec::as_slice).unwrap_or(&[]);
        let mut keys: Vec<Option<u64>> = Vec::with_capacity(preds.len() + 1);
        if va == cfg.entry && !preds.is_empty() {
            keys.push(None);
        }
        keys.extend(preds.iter().map(|&p| Some(p)));

        let mut phis = Vec::new();
        for &cell in seeds.get(&va).map(Vec::as_slice).unwrap_or(&[]) {
            let Some(&dst) = phi_dsts.get(&(va, cell)) else { continue };
            let got = phi_args.remove(&(va, cell)).unwrap_or_default();
            let mut args = Vec::with_capacity(keys.len());
            for &k in &keys {
                let id = match got.get(&k) {
                    Some(&id) => id,
                    // Defensively unreachable: every predecessor of a
                    // reachable block is itself reachable and walked.
                    None => ren.version0(cell)?,
                };
                args.push((k, id));
            }
            phis.push(Phi { dst, args });
        }
        blocks.insert(
            va,
            SsaBlock {
                start: block.start,
                end: block.end,
                phis,
                stmts: renamed.remove(&va).unwrap_or_default(),
                successors: block.successors.clone(),
                truncated: block.truncated,
            },
        );
    }

    // Lower each φ's width to the minimum its arguments guarantee. The
    // total width over all names strictly decreases every changing pass,
    // so the loop terminates; the bound is pure defense.
    for _ in 0..=ren.names.len() {
        let mut changed = false;
        for block in blocks.values() {
            for phi in &block.phis {
                let Some(mut w) = ren.names.get(phi.dst as usize).map(|n| n.width) else {
                    continue;
                };
                for &(_, arg) in &phi.args {
                    if let Some(a) = ren.names.get(arg as usize)
                        && a.width.bits() < w.bits()
                    {
                        w = a.width;
                    }
                }
                if let Some(n) = ren.names.get_mut(phi.dst as usize)
                    && n.width != w
                {
                    n.width = w;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Record every use wider than its reaching definition — the honest
    // marker for a read of bits that definition never wrote.
    let mut partial: BTreeSet<(u64, usize)> = BTreeSet::new();
    for (&va, block) in &blocks {
        for (i, stmt) in block.stmts.iter().enumerate() {
            for_each_use(stmt, &mut |r| {
                if let Some(n) = ren.names.get(r.num as usize)
                    && r.width.bits() > n.width.bits()
                {
                    partial.insert((va, i));
                }
            });
        }
    }

    let mut live_in: Vec<u16> = ren.entry_ids.values().copied().collect();
    live_in.sort_unstable();

    Ok(SsaFunction {
        entry: func.entry,
        name: func.name.clone(),
        arch: func.arch,
        blocks,
        skipped,
        names: ren.names,
        live_in,
        partial: partial.into_iter().collect(),
    })
}

// ---------------------------------------------------------------------------
// Well-formedness
// ---------------------------------------------------------------------------

/// Where an SSA name is defined, for the dominance test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DefSite {
    /// Defined by a φ at the head of `block` (before every statement).
    Phi { block: u64 },
    /// Defined by statement `index` of `block`.
    Stmt { block: u64, index: usize },
}

/// Validate an [`SsaFunction`] from scratch. Total and side-effect-free:
/// recomputes reachability and dominance from the stored blocks, then
/// checks every invariant the module docs promise — IR well-formedness
/// per block, one definition per name, dominance of every use, width
/// discipline with an exact [`SsaFunction::partial`] list, and canonical
/// φ shape. Returns the first [`SsaFault`]; never panics.
pub fn check(f: &SsaFunction) -> Result<(), SsaFault> {
    // Structure: the entry must be present, and every stored block
    // reachable — dominance is undefined for anything else.
    if !f.blocks.is_empty() && !f.blocks.contains_key(&f.entry) {
        return Err(SsaFault::MissingEntry);
    }
    for (&va, block) in &f.blocks {
        ir::check(&block.stmts).map_err(|fault| SsaFault::Ir { block: va, fault })?;
    }
    let raw: BTreeMap<u64, Vec<u64>> = f
        .blocks
        .iter()
        .map(|(&va, b)| (va, b.successors.clone()))
        .collect();
    let cfg = Cfg::analyze(f.entry, &raw);
    for &va in f.blocks.keys() {
        if !cfg.succs.contains_key(&va) {
            return Err(SsaFault::Unreachable { block: va });
        }
    }

    // Every occurrence must index the name table in its own space.
    let valid = |r: Reg| -> bool {
        f.names
            .get(r.num as usize)
            .is_some_and(|n| n.space == r.space)
    };
    for (&va, block) in &f.blocks {
        let mut bad: Option<u16> = None;
        let mut probe = |r: Reg| {
            if bad.is_none() && !valid(r) {
                bad = Some(r.num);
            }
        };
        for stmt in &block.stmts {
            for_each_use(stmt, &mut probe);
            for_each_def(stmt, &mut probe);
        }
        if let Some(num) = bad {
            return Err(SsaFault::BadName { block: va, num });
        }
        for phi in &block.phis {
            if f.names.get(phi.dst as usize).is_none() {
                return Err(SsaFault::BadName {
                    block: va,
                    num: phi.dst,
                });
            }
            for &(_, arg) in &phi.args {
                if f.names.get(arg as usize).is_none() {
                    return Err(SsaFault::BadName {
                        block: va,
                        num: arg,
                    });
                }
            }
        }
    }

    // (cell, version) pairs are unique.
    let mut versions: BTreeSet<(Space, u16, u32)> = BTreeSet::new();
    for (id, n) in f.names.iter().enumerate() {
        if !versions.insert((n.space, n.cell, n.version)) {
            return Err(SsaFault::DuplicateVersion { name: id as u16 });
        }
    }

    // Exactly one definition per non-entry name; none for version 0.
    let mut defs: Vec<Option<DefSite>> = vec![None; f.names.len()];
    let record = |id: u16, site: DefSite, defs: &mut Vec<Option<DefSite>>| -> Result<(), SsaFault> {
        let slot = &mut defs[id as usize];
        if slot.is_some() {
            return Err(SsaFault::Redefined { name: id });
        }
        *slot = Some(site);
        Ok(())
    };
    for (&va, block) in &f.blocks {
        for phi in &block.phis {
            record(phi.dst, DefSite::Phi { block: va }, &mut defs)?;
        }
        for (i, stmt) in block.stmts.iter().enumerate() {
            let mut fault = None;
            for_each_def(stmt, &mut |r| {
                if fault.is_some() {
                    return;
                }
                // A definition occurrence carries exactly its name's width.
                if f.names[r.num as usize].width != r.width {
                    fault = Some(SsaFault::DefWidth {
                        block: va,
                        name: r.num,
                    });
                    return;
                }
                if let Err(e) = record(r.num, DefSite::Stmt { block: va, index: i }, &mut defs) {
                    fault = Some(e);
                }
            });
            if let Some(e) = fault {
                return Err(e);
            }
        }
    }
    for (id, n) in f.names.iter().enumerate() {
        match (n.version, defs[id].is_some()) {
            (0, true) => return Err(SsaFault::Redefined { name: id as u16 }),
            (0, false) => {}
            (_, false) => return Err(SsaFault::Undefined { name: id as u16 }),
            (_, true) => {}
        }
    }

    // `live_in` is exactly the version-0 names, ascending.
    let expected: Vec<u16> = f
        .names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.version == 0)
        .map(|(id, _)| id as u16)
        .collect();
    if f.live_in != expected {
        return Err(SsaFault::LiveIn);
    }
    for &id in &f.live_in {
        if f.names[id as usize].width != full_width(f.names[id as usize].space) {
            return Err(SsaFault::LiveIn);
        }
    }

    // φ shape: strictly ascending cells; one argument per predecessor
    // edge in canonical order; matching cells; no width overclaim. The
    // `None` (function-entry) edge must carry the at-entry version.
    for (&va, block) in &f.blocks {
        let preds = cfg.preds.get(&va).map(Vec::as_slice).unwrap_or(&[]);
        let mut expected_keys: Vec<Option<u64>> = Vec::with_capacity(preds.len() + 1);
        if va == f.entry && !preds.is_empty() {
            expected_keys.push(None);
        }
        expected_keys.extend(preds.iter().map(|&p| Some(p)));
        let mut prev: Option<Cell> = None;
        for (index, phi) in block.phis.iter().enumerate() {
            let dn = &f.names[phi.dst as usize];
            let cell = (dn.space, dn.cell);
            if prev.is_some_and(|p| p >= cell) {
                return Err(SsaFault::PhiShape { block: va, index });
            }
            prev = Some(cell);
            let keys: Vec<Option<u64>> = phi.args.iter().map(|&(k, _)| k).collect();
            if keys != expected_keys || keys.is_empty() {
                return Err(SsaFault::PhiShape { block: va, index });
            }
            for &(k, arg) in &phi.args {
                let an = &f.names[arg as usize];
                if (an.space, an.cell) != cell {
                    return Err(SsaFault::PhiShape { block: va, index });
                }
                if k.is_none() && an.version != 0 {
                    return Err(SsaFault::PhiShape { block: va, index });
                }
                if dn.width.bits() > an.width.bits() {
                    return Err(SsaFault::PhiWidth { block: va, index });
                }
            }
        }
    }

    // The `partial` list is exactly the wider-than-definition use set.
    let mut partial: BTreeSet<(u64, usize)> = BTreeSet::new();
    for (&va, block) in &f.blocks {
        for (i, stmt) in block.stmts.iter().enumerate() {
            for_each_use(stmt, &mut |r| {
                if r.width.bits() > f.names[r.num as usize].width.bits() {
                    partial.insert((va, i));
                }
            });
        }
    }
    if f.partial != partial.into_iter().collect::<Vec<_>>() {
        return Err(SsaFault::Partial);
    }

    // Dominance: every use sees a definition that dominates it. Version 0
    // is defined at the (virtual) entry and dominates everything; a φ
    // definition precedes its block's statements; a statement's uses
    // precede its own definitions.
    let dominates_stmt_use = |site: DefSite, va: u64, i: usize| -> bool {
        match site {
            DefSite::Phi { block } => block == va || cfg.strictly_dominates(block, va),
            DefSite::Stmt { block, index } => {
                (block == va && index < i) || cfg.strictly_dominates(block, va)
            }
        }
    };
    let dominates_pred_exit = |site: DefSite, p: u64| -> bool {
        match site {
            DefSite::Phi { block } | DefSite::Stmt { block, .. } => {
                block == p || cfg.strictly_dominates(block, p)
            }
        }
    };
    for (&va, block) in &f.blocks {
        for (i, stmt) in block.stmts.iter().enumerate() {
            let mut fault = None;
            for_each_use(stmt, &mut |r| {
                if fault.is_some() {
                    return;
                }
                if let Some(site) = defs[r.num as usize]
                    && !dominates_stmt_use(site, va, i)
                {
                    fault = Some(SsaFault::NotDominated {
                        block: va,
                        name: r.num,
                    });
                }
            });
            if let Some(e) = fault {
                return Err(e);
            }
        }
        for phi in &block.phis {
            for &(k, arg) in &phi.args {
                let Some(p) = k else { continue }; // entry edge: version 0
                if let Some(site) = defs[arg as usize]
                    && !dominates_pred_exit(site, p)
                {
                    return Err(SsaFault::NotDominated {
                        block: va,
                        name: arg,
                    });
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// The display form of name `id` read at `width` (`None`: the name's own
/// width): the cell's conventional spelling — the function's architecture
/// picks [`x86_lift::reg_name`] or [`aarch64_lift::reg_name`], matching
/// [`irlift::render`] — with `#version` appended. Total: an id outside
/// the table prints as `<bad:{id}>`.
fn display(f: &SsaFunction, id: u16, width: Option<Width>) -> String {
    let Some(n) = f.names.get(id as usize) else {
        return format!("<bad:{id}>");
    };
    let w = width.unwrap_or(n.width);
    let base = match n.space {
        Space::Arch => match f.arch {
            Arch::Aarch64 => aarch64_lift::reg_name(n.cell, w),
            Arch::X86_64 | Arch::Other => x86_lift::reg_name(n.cell, w),
        }
        .unwrap_or_else(|| format!("r{}.{}", n.cell, w.token())),
        Space::Flag => match n.cell {
            0 => "CF".to_string(),
            1 => "ZF".to_string(),
            2 => "SF".to_string(),
            3 => "OF".to_string(),
            4 => "PF".to_string(),
            _ => "F?".to_string(),
        },
        Space::Temp => format!("t{}.{}", n.cell, w.token()),
    };
    format!("{base}#{}", n.version)
}

/// Render an SSA function to a deterministic, human-readable dump in the
/// style of [`irlift::render`]: a `; {name} @ {entry} (ssa)` header, a
/// `; live-in:` line when the function receives values, then per block a
/// label, its φ-nodes, its renamed IR, the honest `; (truncated)` marker,
/// and its edge line. Blocks excluded from SSA and partial-definition
/// reads are noted in a trailer rather than silently dropped.
/// `\n`-terminated; never panics, even on a hand-broken function.
pub fn render(func: &SsaFunction) -> String {
    let mut out = String::new();
    let name = func
        .name
        .clone()
        .unwrap_or_else(|| format!("sub_{:x}", func.entry));
    let _ = writeln!(out, "; {name} @ {:#018x} (ssa)", func.entry);
    if !func.live_in.is_empty() {
        let list = func
            .live_in
            .iter()
            .map(|&id| display(func, id, None))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "; live-in: {list}");
    }
    let namer = |reg: Reg| Some(display(func, reg.num, Some(reg.width)));
    for block in func.blocks.values() {
        let _ = writeln!(out, "loc_{:x}:", block.start);
        for phi in &block.phis {
            let args = phi
                .args
                .iter()
                .map(|&(k, id)| {
                    let v = display(func, id, None);
                    match k {
                        None => format!("entry: {v}"),
                        Some(p) => format!("loc_{p:x}: {v}"),
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "  {} := phi({args})", display(func, phi.dst, None));
        }
        for line in ir::render_with(&block.stmts, &namer).lines() {
            let _ = writeln!(out, "  {line}");
        }
        if block.truncated {
            let _ = writeln!(out, "  ; (truncated)");
        }
        if block.successors.is_empty() {
            let _ = writeln!(out, "  ; -> (none)");
        } else {
            let edges = block
                .successors
                .iter()
                .map(|s| format!("loc_{s:x}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "  ; -> {edges}");
        }
    }
    if !func.skipped.is_empty() {
        let list = func
            .skipped
            .iter()
            .map(|s| format!("loc_{s:x}"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "; unreachable, not in ssa: {list}");
    }
    if !func.partial.is_empty() {
        let list = func
            .partial
            .iter()
            .map(|&(va, i)| format!("loc_{va:x}/{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "; partial-def reads: {list}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, BranchKind, Flag};

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
    fn assign(dst: Reg, value: Expr) -> Stmt {
        Stmt::Assign { dst, value }
    }
    fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r)
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

    /// Construct and insist the result passes its own check — the
    /// module's core promise, asserted by every test that builds SSA.
    fn build(f: &irlift::LiftedFunction) -> SsaFunction {
        let ssa = construct(f).expect("well-formed input constructs");
        assert_eq!(check(&ssa), Ok(()), "constructed SSA must pass check");
        ssa
    }

    fn name(f: &SsaFunction, id: u16) -> Name {
        f.names[id as usize]
    }

    fn phi_count(f: &SsaFunction) -> usize {
        f.blocks.values().map(|b| b.phis.len()).sum()
    }

    /// Rebuild a register reference with a renamed `num`, for expected-
    /// value assertions.
    fn with_num(r: Reg, num: u16) -> Reg {
        Reg { num, ..r }
    }

    /// A diamond with the merged cell live at the join:
    /// entry defines rax, both arms redefine it, the merge reads it.
    fn diamond() -> irlift::LiftedFunction {
        func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W64), c(1, Width::W64))],
                    vec![0x1010, 0x1020],
                ),
                block(
                    0x1010,
                    vec![assign(ra(0, Width::W64), c(2, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(0, Width::W64), c(3, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        )
    }

    // -- dominators and frontiers ------------------------------------------

    #[test]
    fn dominators_and_frontiers_of_a_diamond() {
        let raw = BTreeMap::from([
            (1u64, vec![2u64, 3]),
            (2, vec![4]),
            (3, vec![4]),
            (4, vec![]),
        ]);
        let cfg = Cfg::analyze(1, &raw);
        assert_eq!(cfg.idom, BTreeMap::from([(2, 1), (3, 1), (4, 1)]));
        assert_eq!(cfg.frontiers.get(&2), Some(&BTreeSet::from([4])));
        assert_eq!(cfg.frontiers.get(&3), Some(&BTreeSet::from([4])));
        assert_eq!(cfg.frontiers.get(&1), None, "the entry's frontier is empty");
        assert!(cfg.strictly_dominates(1, 4));
        assert!(!cfg.strictly_dominates(2, 4));
        assert!(!cfg.strictly_dominates(4, 4));
    }

    #[test]
    fn dominators_and_frontiers_of_a_loop() {
        // 1 -> 2 -> 3 -> {2, 4}: the header 2 is its own frontier via the
        // back edge, which is what places a loop φ.
        let raw = BTreeMap::from([
            (1u64, vec![2u64]),
            (2, vec![3]),
            (3, vec![2, 4]),
            (4, vec![]),
        ]);
        let cfg = Cfg::analyze(1, &raw);
        assert_eq!(cfg.idom, BTreeMap::from([(2, 1), (3, 2), (4, 3)]));
        assert_eq!(cfg.frontiers.get(&2), Some(&BTreeSet::from([2])));
        assert_eq!(cfg.frontiers.get(&3), Some(&BTreeSet::from([2])));
    }

    #[test]
    fn unreachable_blocks_are_outside_the_analysis() {
        let raw = BTreeMap::from([(1u64, vec![2u64]), (2, vec![]), (9, vec![2])]);
        let cfg = Cfg::analyze(1, &raw);
        assert!(!cfg.succs.contains_key(&9));
        // The orphan's edge into reachable code contributes no pred.
        assert_eq!(cfg.preds.get(&2), Some(&vec![1]));
    }

    // -- straight-line code -------------------------------------------------

    #[test]
    fn straight_line_code_produces_no_phis_and_sequential_versions() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), c(1, Width::W64)),
                    assign(ra(0, Width::W64), c(2, Width::W64)),
                    assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                ],
                vec![],
            )],
        );
        let ssa = build(&f);
        assert_eq!(phi_count(&ssa), 0);
        assert!(ssa.live_in.is_empty());
        assert!(ssa.partial.is_empty());
        // rax#1, rax#2, rcx#1 — and the read sees the latest rax.
        assert_eq!(name(&ssa, 0).version, 1);
        assert_eq!(name(&ssa, 1).version, 2);
        assert_eq!((name(&ssa, 2).cell, name(&ssa, 2).version), (1, 1));
        let stmts = &ssa.blocks[&0x1000].stmts;
        assert_eq!(
            stmts[2],
            assign(
                with_num(ra(1, Width::W64), 2),
                read(with_num(ra(0, Width::W64), 1))
            )
        );
    }

    #[test]
    fn straight_line_render_is_exact() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), c(1, Width::W64)),
                    assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                ],
                vec![],
            )],
        );
        let ssa = build(&f);
        assert_eq!(
            render(&ssa),
            "; sub_1000 @ 0x0000000000001000 (ssa)\n\
             loc_1000:\n\
             \x20 rax#1 := 0x1.q\n\
             \x20 rcx#1 := rax#1\n\
             \x20 ; -> (none)\n"
        );
    }

    // -- the diamond ---------------------------------------------------------

    #[test]
    fn a_diamond_merges_exactly_one_phi() {
        let ssa = build(&diamond());
        assert_eq!(phi_count(&ssa), 1);
        let merge = &ssa.blocks[&0x1030];
        assert_eq!(merge.phis.len(), 1);
        let phi = &merge.phis[0];
        // The φ is a version of rax merging the two arm definitions.
        let dn = name(&ssa, phi.dst);
        assert_eq!((dn.space, dn.cell, dn.width), (Space::Arch, 0, Width::W64));
        assert_eq!(phi.args.len(), 2);
        assert_eq!(phi.args[0].0, Some(0x1010));
        assert_eq!(phi.args[1].0, Some(0x1020));
        assert_eq!(name(&ssa, phi.args[0].1).version, 2);
        assert_eq!(name(&ssa, phi.args[1].1).version, 3);
        // The merge's read sees the φ, not either arm.
        assert_eq!(
            merge.stmts[0],
            assign(
                with_num(ra(1, Width::W64), 4),
                read(with_num(ra(0, Width::W64), phi.dst))
            )
        );
        assert!(ssa.live_in.is_empty());
    }

    #[test]
    fn the_diamond_render_is_exact() {
        let ssa = build(&diamond());
        assert_eq!(
            render(&ssa),
            "; sub_1000 @ 0x0000000000001000 (ssa)\n\
             loc_1000:\n\
             \x20 rax#1 := 0x1.q\n\
             \x20 ; -> loc_1010, loc_1020\n\
             loc_1010:\n\
             \x20 rax#2 := 0x2.q\n\
             \x20 ; -> loc_1030\n\
             loc_1020:\n\
             \x20 rax#3 := 0x3.q\n\
             \x20 ; -> loc_1030\n\
             loc_1030:\n\
             \x20 rax#4 := phi(loc_1010: rax#2, loc_1020: rax#3)\n\
             \x20 rcx#1 := rax#4\n\
             \x20 ; -> (none)\n"
        );
    }

    #[test]
    fn a_value_dead_at_the_merge_gets_no_phi() {
        // Same diamond shape, but the merge ignores rax: pruning must
        // insert nothing.
        let mut f = diamond();
        f.blocks.get_mut(&0x1030).unwrap().stmts =
            vec![assign(ra(2, Width::W64), c(9, Width::W64))];
        let ssa = build(&f);
        assert_eq!(phi_count(&ssa), 0);
    }

    #[test]
    fn a_one_arm_definition_merges_with_the_entry_value() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010, 0x1020]),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), c(2, Width::W64))],
                    vec![0x1030],
                ),
                block(0x1020, vec![], vec![0x1030]),
                block(
                    0x1030,
                    vec![assign(ra(0, Width::W64), read(ra(1, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let ssa = build(&f);
        assert_eq!(phi_count(&ssa), 1);
        let phi = &ssa.blocks[&0x1030].phis[0];
        // Left arm brings its definition; the untouched arm brings the
        // at-entry value, which the function therefore receives.
        assert_eq!(name(&ssa, phi.args[0].1).version, 1);
        assert_eq!(name(&ssa, phi.args[1].1).version, 0);
        assert_eq!(ssa.live_in, vec![phi.args[1].1]);
        assert!(render(&ssa).contains("; live-in: rcx#0\n"), "{}", render(&ssa));
    }

    // -- loops ---------------------------------------------------------------

    #[test]
    fn a_loop_header_gets_a_phi() {
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W64), c(0, Width::W64))],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![assign(
                        ra(0, Width::W64),
                        bin(BinOp::Add, read(ra(0, Width::W64)), c(1, Width::W64)),
                    )],
                    vec![0x1010, 0x1020],
                ),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let ssa = build(&f);
        assert_eq!(phi_count(&ssa), 1);
        let header = &ssa.blocks[&0x1010];
        let phi = &header.phis[0];
        assert_eq!(phi.args.len(), 2);
        assert_eq!(phi.args[0].0, Some(0x1000));
        assert_eq!(phi.args[1].0, Some(0x1010));
        // The back-edge argument is the header's own new definition; the
        // increment reads the φ.
        let Stmt::Assign { dst, value } = &header.stmts[0] else {
            panic!("header holds the increment");
        };
        assert_eq!(phi.args[1].1, dst.num);
        let Expr::Binary { lhs, .. } = value else {
            panic!("increment is an add");
        };
        assert_eq!(**lhs, read(with_num(ra(0, Width::W64), phi.dst)));
        // The exit reads the increment's definition, not the φ.
        assert_eq!(
            ssa.blocks[&0x1020].stmts[0],
            assign(
                with_num(ra(1, Width::W64), 3),
                read(with_num(ra(0, Width::W64), dst.num))
            )
        );
        assert!(ssa.live_in.is_empty());
    }

    #[test]
    fn a_self_looping_entry_carries_the_virtual_entry_edge() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![assign(
                    ra(0, Width::W64),
                    bin(BinOp::Add, read(ra(0, Width::W64)), c(1, Width::W64)),
                )],
                vec![0x1000],
            )],
        );
        let ssa = build(&f);
        assert_eq!(phi_count(&ssa), 1);
        let phi = &ssa.blocks[&0x1000].phis[0];
        // First argument: the function-entry edge with the at-entry value.
        assert_eq!(phi.args[0].0, None);
        assert_eq!(name(&ssa, phi.args[0].1).version, 0);
        assert_eq!(phi.args[1].0, Some(0x1000));
        assert_eq!(ssa.live_in, vec![phi.args[0].1]);
        let text = render(&ssa);
        assert!(text.contains("phi(entry: rax#0, loc_1000: rax#2)"), "{text}");
    }

    // -- live-in, flags, partial definitions ---------------------------------

    #[test]
    fn a_use_with_no_definition_reads_the_entry_value() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![assign(ra(0, Width::W64), read(ra(1, Width::W64)))],
                vec![],
            )],
        );
        let ssa = build(&f);
        assert_eq!(ssa.live_in.len(), 1);
        let n = name(&ssa, ssa.live_in[0]);
        assert_eq!(
            (n.space, n.cell, n.version, n.width),
            (Space::Arch, 1, 0, Width::W64)
        );
    }

    #[test]
    fn flags_are_versioned_like_registers() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(
                        Reg::flag(Flag::Zero),
                        bin(BinOp::Eq, read(ra(0, Width::W64)), read(ra(1, Width::W64))),
                    ),
                    Stmt::Branch {
                        kind: BranchKind::Jump,
                        cond: Some(read(Reg::flag(Flag::Zero))),
                        target: c(0x2000, Width::W64),
                    },
                ],
                vec![],
            )],
        );
        let ssa = build(&f);
        let stmts = &ssa.blocks[&0x1000].stmts;
        let Stmt::Assign { dst, .. } = &stmts[0] else {
            panic!("flag write first");
        };
        let Stmt::Branch {
            cond: Some(Expr::Reg(r)),
            ..
        } = &stmts[1]
        else {
            panic!("guarded branch second");
        };
        assert_eq!(r.num, dst.num, "the guard reads the flag definition");
        assert!(render(&ssa).contains("ZF#1"), "{}", render(&ssa));
    }

    #[test]
    fn a_narrow_definition_read_wide_is_recorded_as_partial() {
        // al := 5 then a read of rax: the definition wrote 8 bits, the
        // use wants 64 — an honest partial-definition marker, never a
        // silent exact-dependence claim.
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W8), c(5, Width::W8)),
                    assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                ],
                vec![],
            )],
        );
        let ssa = build(&f);
        assert_eq!(ssa.partial, vec![(0x1000, 1)]);
        assert_eq!(name(&ssa, 0).width, Width::W8);
        assert!(
            render(&ssa).contains("; partial-def reads: loc_1000/1\n"),
            "{}",
            render(&ssa)
        );

        // A stale partial list is rejected.
        let mut broken = ssa.clone();
        broken.partial.clear();
        assert_eq!(check(&broken), Err(SsaFault::Partial));
    }

    // -- honesty: truncation, unreachable code, refusals ---------------------

    #[test]
    fn truncated_and_unreachable_blocks_are_carried_and_noted() {
        let mut f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W64), c(1, Width::W64))],
                    vec![],
                ),
                block(
                    0x2000,
                    vec![assign(ra(0, Width::W64), c(2, Width::W64))],
                    vec![],
                ),
            ],
        );
        f.blocks.get_mut(&0x1000).unwrap().truncated = true;
        let ssa = build(&f);
        assert!(ssa.blocks[&0x1000].truncated);
        assert_eq!(ssa.skipped, vec![0x2000]);
        assert!(!ssa.blocks.contains_key(&0x2000));
        let text = render(&ssa);
        assert!(text.contains("  ; (truncated)\n"), "{text}");
        assert!(text.contains("; unreachable, not in ssa: loc_2000\n"), "{text}");
    }

    #[test]
    fn a_malformed_reachable_block_is_refused() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![assign(ra(0, Width::W64), read(ra(1, Width::W32)))],
                vec![],
            )],
        );
        assert!(matches!(
            construct(&f),
            Err(Unrepresentable::MalformedBlock { block: 0x1000, .. })
        ));
    }

    #[test]
    fn a_malformed_unreachable_block_is_skipped_not_refused() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![]),
                block(
                    0x2000,
                    vec![assign(ra(0, Width::W64), read(ra(1, Width::W32)))],
                    vec![],
                ),
            ],
        );
        let ssa = build(&f);
        assert_eq!(ssa.skipped, vec![0x2000]);
    }

    #[test]
    fn a_function_too_large_to_name_is_refused() {
        // 17 chained blocks of MAX_STMTS definitions each need more names
        // than the u16 id space holds.
        let stmts: Vec<Stmt> = vec![assign(ra(0, Width::W64), c(0, Width::W64)); ir::MAX_STMTS];
        let blocks: Vec<irlift::LiftedBlock> = (0..17u64)
            .map(|i| {
                let va = 0x1000 + 0x100 * i;
                let succ = if i < 16 { vec![va + 0x100] } else { vec![] };
                block(va, stmts.clone(), succ)
            })
            .collect();
        assert_eq!(
            construct(&func(0x1000, blocks)),
            Err(Unrepresentable::TooManyNames)
        );
    }

    #[test]
    fn an_empty_function_constructs_to_an_empty_ssa() {
        let f = irlift::LiftedFunction {
            entry: 0x1000,
            name: Some("empty".into()),
            arch: Arch::X86_64,
            blocks: BTreeMap::new(),
        };
        let ssa = build(&f);
        assert!(ssa.blocks.is_empty() && ssa.names.is_empty());
        assert_eq!(render(&ssa), "; empty @ 0x0000000000001000 (ssa)\n");
    }

    // -- call effects (callfx integration) ------------------------------------

    /// A hand-built direct call, for blocks run through
    /// [`crate::callfx::apply`] before construction.
    fn call_stmt(target: u64) -> Stmt {
        Stmt::Branch {
            kind: BranchKind::Call,
            cond: None,
            target: c(target, Width::W64),
        }
    }

    /// Apply the x86-64 call effects, then construct and check.
    fn build_with_callfx(f: &irlift::LiftedFunction) -> SsaFunction {
        build(&crate::callfx::apply(f, &crate::callfx::x86_64()))
    }

    #[test]
    fn a_post_call_use_of_a_caller_saved_register_reads_the_call_definition() {
        // rax := 1 ; call ; rcx := rax — the post-call read must see the
        // call's clobber definition, not the stale pre-call value.
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), c(1, Width::W64)),
                    call_stmt(0x9000),
                    assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                ],
                vec![],
            )],
        );
        let ssa = build_with_callfx(&f);
        let stmts = &ssa.blocks[&0x1000].stmts;
        let Stmt::Assign { dst: pre, .. } = &stmts[0] else {
            panic!("the pre-call definition survives");
        };
        let Stmt::Intrinsic { writes, .. } = &stmts[2] else {
            panic!("callfx follows the call");
        };
        let clobber = writes
            .iter()
            .find(|w| {
                let n = name(&ssa, w.num);
                (n.space, n.cell) == (Space::Arch, 0)
            })
            .expect("rax is clobbered");
        let Stmt::Assign {
            value: Expr::Reg(use_reg),
            ..
        } = &stmts[4]
        else {
            panic!("the post-call read survives");
        };
        assert_ne!(use_reg.num, pre.num, "the cross-call link is severed");
        assert_eq!(use_reg.num, clobber.num, "the use reads the call's def");
        assert!(name(&ssa, use_reg.num).version > name(&ssa, pre.num).version);
    }

    #[test]
    fn a_callee_saved_register_keeps_its_name_across_a_call() {
        // rbx (cell 3) is preserved by the ABI: the post-call read links
        // straight to the pre-call definition, no severed edge.
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(3, Width::W64), c(1, Width::W64)),
                    call_stmt(0x9000),
                    assign(ra(1, Width::W64), read(ra(3, Width::W64))),
                ],
                vec![],
            )],
        );
        let ssa = build_with_callfx(&f);
        let stmts = &ssa.blocks[&0x1000].stmts;
        let Stmt::Assign { dst: pre, .. } = &stmts[0] else {
            panic!();
        };
        let Stmt::Assign {
            value: Expr::Reg(use_reg),
            ..
        } = &stmts[4]
        else {
            panic!();
        };
        assert_eq!(use_reg.num, pre.num, "rbx flows through the call unbroken");
    }

    #[test]
    fn argument_defs_feed_the_call_and_missing_arguments_become_live_in() {
        // rdi := 1 ; call — the intrinsic's rdi read links to the def,
        // while rsi (read by callfx, never defined) is received: the
        // live-in line starts to resemble a parameter list.
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(7, Width::W64), c(1, Width::W64)),
                    call_stmt(0x9000),
                ],
                vec![],
            )],
        );
        let ssa = build_with_callfx(&f);
        let stmts = &ssa.blocks[&0x1000].stmts;
        let Stmt::Intrinsic { reads, .. } = &stmts[2] else {
            panic!("callfx follows the call");
        };
        let rdi = reads
            .iter()
            .find_map(|r| match r {
                Expr::Reg(reg) if name(&ssa, reg.num).cell == 7 => Some(*reg),
                _ => None,
            })
            .expect("callfx reads rdi");
        assert_eq!(name(&ssa, rdi.num).version, 1, "the argument def is used");
        assert!(
            ssa.live_in.iter().any(|&id| {
                let n = name(&ssa, id);
                (n.space, n.cell, n.version) == (Space::Arch, 6, 0)
            }),
            "undefined rsi is received: {:?}",
            ssa.live_in
        );
    }

    #[test]
    fn the_rsp_chain_is_sequential_across_a_call_and_partial_stays_empty() {
        // The call lift's own push, the call, then callfx's pop: rsp is
        // never an intrinsic write, so its versions chain 0 → 1 → 2 and
        // frame tracking stays exact through the call.
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(
                        ra(4, Width::W64),
                        bin(BinOp::Sub, read(ra(4, Width::W64)), c(8, Width::W64)),
                    ),
                    Stmt::Store {
                        addr: read(ra(4, Width::W64)),
                        value: c(0x1005, Width::W64),
                    },
                    call_stmt(0x9000),
                ],
                vec![],
            )],
        );
        let ssa = build_with_callfx(&f);
        let stmts = &ssa.blocks[&0x1000].stmts;
        let Stmt::Intrinsic { writes, .. } = &stmts[3] else {
            panic!("callfx follows the call");
        };
        assert!(
            writes.iter().all(|w| {
                let n = name(&ssa, w.num);
                (n.space, n.cell) != (Space::Arch, 4)
            }),
            "rsp is not a clobber"
        );
        let Stmt::Assign { dst: push, .. } = &stmts[0] else {
            panic!();
        };
        let Stmt::Assign {
            dst: pop,
            value: Expr::Binary { lhs, .. },
        } = &stmts[4]
        else {
            panic!("the inserted pop is the last statement");
        };
        assert_eq!(name(&ssa, push.num).version, 1);
        assert_eq!(name(&ssa, pop.num).version, 2);
        assert_eq!(**lhs, read(with_num(ra(4, Width::W64), push.num)));
        assert!(ssa.partial.is_empty(), "every effect write is full-width");
    }

    #[test]
    fn a_call_in_one_arm_forces_a_phi_for_a_live_clobbered_cell() {
        // entry defines rax; only one arm calls; the merge reads rax —
        // the clobber must merge with the untouched arm's value in a φ.
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W64), c(1, Width::W64))],
                    vec![0x1010, 0x1020],
                ),
                block(0x1010, vec![call_stmt(0x9000)], vec![0x1030]),
                block(0x1020, vec![], vec![0x1030]),
                block(
                    0x1030,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let ssa = build_with_callfx(&f);
        // Of everything the call clobbers, only rax is live at the merge,
        // so pruning admits exactly one φ.
        assert_eq!(phi_count(&ssa), 1);
        let phi = &ssa.blocks[&0x1030].phis[0];
        let dn = name(&ssa, phi.dst);
        assert_eq!((dn.space, dn.cell), (Space::Arch, 0));
        let calling = name(&ssa, phi.args[0].1);
        let quiet = name(&ssa, phi.args[1].1);
        assert_eq!(phi.args[0].0, Some(0x1010));
        assert_eq!(phi.args[1].0, Some(0x1020));
        assert!(calling.version > 1, "the calling arm brings the clobber");
        assert_eq!(quiet.version, 1, "the quiet arm brings the entry def");
    }

    // -- the check rejects hand-broken SSA -----------------------------------

    #[test]
    fn check_rejects_a_duplicate_definition() {
        let mut ssa = build(&diamond());
        // Point the right arm's definition at the left arm's name.
        let Stmt::Assign { dst, .. } = &mut ssa.blocks.get_mut(&0x1020).unwrap().stmts[0] else {
            panic!();
        };
        dst.num = 1;
        assert_eq!(check(&ssa), Err(SsaFault::Redefined { name: 1 }));
    }

    #[test]
    fn check_rejects_a_use_its_definition_does_not_dominate() {
        let mut ssa = build(&diamond());
        // Make the merge read the left arm's definition directly: the
        // left arm does not dominate the merge.
        let Stmt::Assign { value, .. } = &mut ssa.blocks.get_mut(&0x1030).unwrap().stmts[0] else {
            panic!();
        };
        *value = read(with_num(ra(0, Width::W64), 1));
        assert_eq!(
            check(&ssa),
            Err(SsaFault::NotDominated {
                block: 0x1030,
                name: 1
            })
        );
    }

    #[test]
    fn check_rejects_a_phi_with_missing_or_foreign_arguments() {
        let mut ssa = build(&diamond());
        let dropped = ssa.blocks.get_mut(&0x1030).unwrap().phis[0].args.remove(0);
        assert_eq!(
            check(&ssa),
            Err(SsaFault::PhiShape {
                block: 0x1030,
                index: 0
            })
        );
        ssa.blocks.get_mut(&0x1030).unwrap().phis[0]
            .args
            .insert(0, dropped);
        assert_eq!(check(&ssa), Ok(()));
        // An argument naming another cell is as wrong as a missing one.
        ssa.blocks.get_mut(&0x1030).unwrap().phis[0].args[0].1 = 4; // rcx#1
        assert_eq!(
            check(&ssa),
            Err(SsaFault::PhiShape {
                block: 0x1030,
                index: 0
            })
        );
    }

    #[test]
    fn check_rejects_a_bad_name_a_bad_width_and_a_bad_live_in() {
        let good = build(&diamond());

        let mut ssa = good.clone();
        let Stmt::Assign { value, .. } = &mut ssa.blocks.get_mut(&0x1030).unwrap().stmts[0] else {
            panic!();
        };
        *value = read(with_num(ra(0, Width::W64), 999));
        assert_eq!(
            check(&ssa),
            Err(SsaFault::BadName {
                block: 0x1030,
                num: 999
            })
        );

        let mut ssa = good.clone();
        ssa.names[0].width = Width::W32; // the definition occurrence is W64
        assert_eq!(
            check(&ssa),
            Err(SsaFault::DefWidth {
                block: 0x1000,
                name: 0
            })
        );

        let mut ssa = good.clone();
        ssa.live_in = vec![0]; // name 0 is version 1, not an entry value
        assert_eq!(check(&ssa), Err(SsaFault::LiveIn));
    }

    #[test]
    fn check_rejects_an_unreachable_block_and_a_missing_entry() {
        let good = build(&diamond());

        let mut ssa = good.clone();
        ssa.blocks.insert(
            0x9000,
            SsaBlock {
                start: 0x9000,
                end: 0x9004,
                phis: Vec::new(),
                stmts: Vec::new(),
                successors: Vec::new(),
                truncated: false,
            },
        );
        assert_eq!(check(&ssa), Err(SsaFault::Unreachable { block: 0x9000 }));

        let mut ssa = good.clone();
        ssa.entry = 0x9999;
        assert_eq!(check(&ssa), Err(SsaFault::MissingEntry));
    }

    #[test]
    fn check_rejects_duplicate_versions_and_ir_faults() {
        let good = build(&diamond());

        let mut ssa = good.clone();
        ssa.names[1].version = 1; // collides with name 0's (rax, 1)
        assert_eq!(check(&ssa), Err(SsaFault::DuplicateVersion { name: 1 }));

        let mut ssa = good.clone();
        let Stmt::Assign { value, .. } = &mut ssa.blocks.get_mut(&0x1000).unwrap().stmts[0] else {
            panic!();
        };
        *value = c(1, Width::W32); // width mismatch against the W64 dst
        assert!(matches!(check(&ssa), Err(SsaFault::Ir { block: 0x1000, .. })));
    }

    // -- determinism ---------------------------------------------------------

    #[test]
    fn construction_and_rendering_are_deterministic() {
        let f = diamond();
        let a = build(&f);
        let b = build(&f);
        assert_eq!(a, b);
        assert_eq!(render(&a), render(&b));
    }

    // -- end to end through recovery and lifting -----------------------------

    #[test]
    fn end_to_end_an_x86_diamond_yields_one_phi_at_the_merge() {
        // xor eax,eax ; je +9 ; inc eax ; jmp +11 ; (pad) ; dec eax ;
        // mov ecx, eax ; ret — eax is live at the merge, so it (alone)
        // gets a φ; the flags and rsp are dead or unmerged and get none.
        let mut img = crate::elf::tests::synthetic_elf64();
        let code: &[u8] = &[
            0x31, 0xC0, // +0:  xor eax, eax
            0x74, 0x05, // +2:  je +9
            0xFF, 0xC0, // +4:  inc eax
            0xEB, 0x03, // +6:  jmp +11
            0x90, // +8:  (unreachable)
            0xFF, 0xC8, // +9:  dec eax
            0x89, 0xC1, // +11: mov ecx, eax
            0xC3, // +13: ret
        ];
        img[0x100..0x100 + code.len()].copy_from_slice(code);
        let image = crate::model::load(&img).unwrap();
        let program = crate::cfg::recover(image.as_ref()).unwrap();
        let entry = 0x40_1000u64;
        let lifted = irlift::lift_function(image.as_ref(), &program.functions[&entry])
            .expect("x86-64 lifts");

        let ssa = build(&lifted);
        assert_eq!(phi_count(&ssa), 1);
        let merge = &ssa.blocks[&(entry + 11)];
        assert_eq!(merge.phis.len(), 1);
        let dn = name(&ssa, merge.phis[0].dst);
        assert_eq!((dn.space, dn.cell), (Space::Arch, 0), "the φ merges rax");
        assert!(ssa.partial.is_empty(), "lifter output has no partial reads");
        let text = render(&ssa);
        assert!(text.contains(" := phi(loc_401004: "), "{text}");

        // The simplified lift constructs and checks too.
        build(&irlift::simplify(&lifted));
    }

    // -- bounded sweep -------------------------------------------------------

    /// xorshift64* with a fixed seed: deterministic, no wall clock.
    fn next(s: &mut u64) -> u64 {
        *s ^= *s >> 12;
        *s ^= *s << 25;
        *s ^= *s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[test]
    fn sweep_random_small_cfgs_always_construct_and_check() {
        let mut s = 0x0DDB_1A5E_5BAD_5EEDu64;
        for _ in 0..400 {
            let nblocks = 2 + next(&mut s) % 6;
            let vas: Vec<u64> = (0..nblocks).map(|i| 0x1000 + 0x10 * i).collect();
            let mut blocks = Vec::new();
            for &va in &vas {
                let nstmts = (next(&mut s) % 5) as usize;
                let mut stmts = Vec::new();
                for _ in 0..nstmts {
                    let r = (next(&mut s) % 4) as u16;
                    let r2 = (next(&mut s) % 4) as u16;
                    let k = next(&mut s) % 100;
                    stmts.push(match next(&mut s) % 6 {
                        0 => assign(ra(r, Width::W64), c(k, Width::W64)),
                        1 => assign(ra(r, Width::W64), read(ra(r2, Width::W64))),
                        2 => assign(
                            Reg::flag(Flag::Zero),
                            bin(BinOp::Eq, read(ra(r, Width::W64)), read(ra(r2, Width::W64))),
                        ),
                        // A narrow definition, to exercise the partial-
                        // read bookkeeping across arbitrary CFGs.
                        3 => assign(ra(r, Width::W32), c(k, Width::W32)),
                        4 => Stmt::Store {
                            addr: read(ra(r, Width::W64)),
                            value: read(ra(r2, Width::W64)),
                        },
                        _ => Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1000, Width::W64),
                        },
                    });
                }
                let nsucc = (next(&mut s) % 3) as usize;
                let succ = (0..nsucc)
                    .map(|_| vas[(next(&mut s) % nblocks) as usize])
                    .collect();
                blocks.push(block(va, stmts, succ));
            }
            let f = func(0x1000, blocks);
            // Every well-formed input constructs, checks, and renders
            // identically on a second run — no panic, ever.
            let ssa = build(&f);
            let again = build(&f);
            assert_eq!(ssa, again);
            assert_eq!(render(&ssa), render(&again));
        }
    }

    // -- architecture (aarch64) --------------------------------------------

    /// Decode and lift a straight-line run of A64 words at `va` into one
    /// temp-threaded statement list, via the real decoder and lifter.
    fn a64_stmts(words: &[u32], va: u64) -> Vec<Stmt> {
        let pairs: Vec<_> = words
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let va = va + 4 * i as u64;
                (
                    crate::aarch64::decode(&w.to_le_bytes(), va).expect("4 bytes decode"),
                    va,
                )
            })
            .collect();
        aarch64_lift::lift_block(&pairs)
    }

    /// An aarch64 diamond built from real lifted words: a `cbz` guard, an
    /// arm holding a modeled `add` plus an `Unknown` word's clobber
    /// intrinsic (LSE atomic), and a `ret` join.
    ///
    /// ```text
    /// 1000  cbz x0, 1010
    /// 1004  add x0, x0, #1 ; (unknown word)
    /// 1010  ret
    /// ```
    fn a64_func() -> irlift::LiftedFunction {
        let mut f = func(
            0x1000,
            vec![
                block(0x1000, a64_stmts(&[0xB400_0080], 0x1000), vec![0x1010, 0x1004]),
                block(
                    0x1004,
                    a64_stmts(&[0x9100_0400, 0xF8E9_0108], 0x1004),
                    vec![0x1010],
                ),
                block(0x1010, a64_stmts(&[0xD65F_03C0], 0x1010), vec![]),
            ],
        );
        f.arch = Arch::Aarch64;
        f
    }

    #[test]
    fn an_aarch64_function_constructs_and_carries_its_arch() {
        let ssa = build(&a64_func());
        assert_eq!(ssa.arch, Arch::Aarch64);
        // x0 is read before any definition: a version-0 live-in exists.
        assert!(
            ssa.live_in
                .iter()
                .any(|&id| (name(&ssa, id).space, name(&ssa, id).cell) == (Space::Arch, 0)),
            "x0 must be live-in: {:?}",
            ssa.live_in
        );
        // An x86 function still reports its own arch.
        assert_eq!(build(&diamond()).arch, Arch::X86_64);
    }

    #[test]
    fn an_aarch64_function_renders_aarch64_register_names() {
        let text = render(&build(&a64_func()));
        assert!(text.contains("x0#"), "{text}");
        assert!(text.contains("x30#"), "{text}");
        // The unknown word's clobber intrinsic writes the SP cell too.
        assert!(text.contains("sp#"), "{text}");
        assert!(!text.contains("rax"), "{text}");
    }

    #[test]
    fn the_downstream_passes_are_arch_blind_on_an_aarch64_function() {
        // The full pipeline over real lifted aarch64 statements, every
        // stage revalidated from scratch: construct → optimize → forward
        // → eliminate_dead → structure → out-of-SSA, zero check failures.
        let ssa = build(&a64_func());
        let (opt, _) = crate::irssaopt::optimize(&ssa);
        assert_eq!(check(&opt), Ok(()));
        let (fwd, _) = crate::irssaopt::forward(&opt);
        assert_eq!(check(&fwd), Ok(()));
        let live_out = crate::callfx::function_live_out(Arch::Aarch64).unwrap_or_default();
        let (swept, _) = crate::irssaopt::eliminate_dead(&fwd, &live_out);
        assert_eq!(check(&swept), Ok(()));
        assert_eq!(swept.arch, Arch::Aarch64, "the arch survives every pass");
        let tables = BTreeMap::new();
        let (root, _) = crate::irstruct::structure(&swept, &tables);
        assert_eq!(crate::irstruct::check(&swept, &tables, &root), Ok(()));
        let (out, _) = crate::irout::out_of_ssa(&swept);
        assert_eq!(crate::irout::check(&swept, &out), Ok(()));
    }
}
