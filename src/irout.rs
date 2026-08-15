//! Out-of-SSA translation: φ-webs coalesced into named variables.
//!
//! [`crate::irssa`] gave every definition its own name so the analyses in
//! [`crate::irssaopt`] could be sparse. Pseudocode cannot be printed in
//! that form: a reader wants `v3`, not `rax#7`, `rax#11` and a φ-node.
//! This module performs the translation *out* of SSA — every name is
//! assigned a **variable**, and the φ-nodes that cannot be honored by
//! name-sharing alone become explicit copies on control-flow edges.
//!
//! The result is **a map, not a rewrite**. [`out_of_ssa`] never mutates
//! its input and never returns a rewritten function: the [`SsaFunction`]
//! stays the analysis truth, and [`OutOfSsa`] is the rendering layer's
//! lookup table (`var_of[name]`) plus the edge copies the renderer must
//! emit. Two consumers can therefore hold different renditions of one
//! analyzed function, and no pass downstream has to re-establish the SSA
//! invariants a rewrite would have destroyed.
//!
//! # The algorithm: correct, then good
//!
//! Boissinot, Darte, Rastello, Dupont de Dinechin and Guillon
//! ("Revisiting Out-of-SSA Translation for Correctness, Code Quality and
//! Efficiency", CGO 2009) separate the two halves that older algorithms
//! entangle, and this module keeps that separation visible:
//!
//! 1. **Correct by construction.** Start from singleton congruence
//!    classes — one variable per name, φs *isolated*. Every φ argument
//!    then lands in a different variable from its φ, so every argument
//!    edge carries a copy. This is trivially correct: no two names share
//!    storage, so nothing can be clobbered.
//! 2. **Coalesce aggressively.** Merge a φ's class with an argument's
//!    class whenever no pair of members *interferes*. Two names
//!    interfere when their live ranges intersect **and** they carry
//!    different values (Boissinot's value-based interference; the
//!    congruence-class framing is Sreedhar, Ju, Gillies and Santhanam,
//!    SAS 1999). Live-range intersection uses the dominance property of
//!    strict SSA: two live ranges can intersect only if one definition
//!    dominates the other, so the test is "is the dominating name still
//!    live just after the dominated one's definition" — no interference
//!    graph is built. Value equality is copy-chasing through
//!    width-exact `Assign { value: Reg }` chains: the same roots
//!    [`crate::irssaopt`]'s propagation already proves.
//! 3. **Sequentialize.** The copies surviving on one edge are
//!    *parallel*; emitting them in an arbitrary order is the swap
//!    miscompile. They are ordered so that no copy overwrites a value a
//!    later copy still needs, and a cycle is broken with the single
//!    temporary slot ([`CopySlot::Temp`]) — Boissinot's Algorithm 1.
//!
//! Cytron et al.'s naive per-predecessor copy insertion is rejected
//! outright: Briggs, Cooper, Harvey and Simpson ("Practical Improvements
//! to the Construction and Destruction of Static Single Assignment
//! Form", SP&E 1998) documented its two miscompiles, the **lost copy**
//! (a φ's result still live where the copy is inserted) and the **swap**
//! (φs at one join forming a permutation). Both are regression fixtures
//! in this module's tests.
//!
//! # What the shape of this IR implies
//!
//! [`crate::irssa`] versions *cells*, and [`irssa::check`] enforces that
//! every argument of a φ names the φ's own cell. Coalescing only ever
//! unions a φ's class with an argument's class, so — by induction — **a
//! congruence class never mixes cells**, and this module enforces that
//! structurally rather than relying on the argument. Three consequences,
//! each worth stating because each is a claim a reader would otherwise
//! have to rediscover:
//!
//! - A variable belongs to exactly one machine cell, so the renderer can
//!   name it after that cell, and the version-0 (at-entry) names — the
//!   function's parameters-in-waiting — can never be merged with each
//!   other, only with later versions of their own cell.
//! - The **lost copy** is live, but only downstream of a pass that moves
//!   *values*. Renaming alone can never place a use of a φ's result past
//!   a redefinition of its cell — the reaching definition there is the
//!   redefinition — so two versions of one cell have disjoint live
//!   ranges on a raw construction, and nothing to coalesce wrongly.
//!   [`crate::irssaopt`]'s propagation and expression forwarding are what
//!   carry a φ's value into such a block, exactly as the published
//!   example needs copy propagation to arise; the fixture builds the
//!   shape through those real passes.
//! - The **swap** is not reachable *from φs alone*: one edge carries at
//!   most one copy per cell (a block holds at most one φ per cell), and
//!   a copy's two ends are variables of that same cell, so the copy
//!   graph on an edge has no cycle. The cycle-breaking machinery is
//!   implemented and tested directly against `sequentialize` anyway —
//!   the day cross-cell copy coalescing lands (a variable that lives in
//!   `rax` here and `rbx` there), permutations become reachable, and the
//!   correctness of that day should not depend on new code.
//!
//! # What the consumer owes: copies live on the *edge*
//!
//! [`OutOfSsa::edge_copies`] is keyed by `(predecessor, successor)`
//! because that is where its copies belong. On a critical edge — a
//! predecessor with several successors feeding a successor with several
//! predecessors — appending them to the predecessor instead would be
//! wrong: a variable of the φ's class may still be live along the
//! *other* successor, and the interference test deliberately allows
//! that (a name dead on this edge but live on the sibling edge does not
//! interfere with the φ). The renderer must therefore materialize the
//! edge, in the usual way: an edge block, or a copy placed under the
//! branch that takes it.
//!
//! # What this slice deliberately leaves on the table
//!
//! Only φ-related names are coalesced. A definition of a cell that no φ
//! mentions keeps its own variable even when it could obviously share
//! one — `rax#1 := 1` in a block both arms of a diamond overwrite is a
//! variable of its own. Merging those is *ordinary copy coalescing*, a
//! readability win with its own cost function (and the thing that would
//! make a variable span two cells, hence make swaps reachable); it is a
//! non-goal here, recorded rather than half-done.
//!
//! # Provenance: the honesty markers reach the renderer
//!
//! Two markers must survive onto variables so pseudocode can carry them.
//! An inventory of what the earlier slices actually built decided the
//! carriers:
//!
//! - **`assumed`** — DESIGN's `AbiAssumed`. Slice 2 shipped ABI effects
//!   not as a name-level tag but as ordinary IR: a
//!   [`crate::callfx::EFFECT_NAME`] intrinsic whose `writes` are the
//!   clobbers a *conforming callee* may perform. That intrinsic write is
//!   the carrier, and it is a better one than a tag would have been —
//!   it is the same fact [`irssa`] renamed and [`crate::irssaopt`] reasoned
//!   about, not a parallel channel that could drift. A variable is
//!   `assumed` when any of its names is defined by such a write: its
//!   value at that point is asserted by the ABI, not proven by the code.
//! - **`partial`** — [`SsaFunction::partial`] lists the `(block,
//!   statement)` positions holding a read wider than its reaching
//!   definition. A variable is `partial` when a name of its class is
//!   read that way at one of those positions.
//!
//! # `check` trusts nothing
//!
//! [`check`] re-derives dominance, liveness and value equality from the
//! [`SsaFunction`] and validates the [`OutOfSsa`] against them: no two
//! names sharing a variable interfere, every φ is resolved (argument
//! variable equal to the φ's, or the edge carries copies that make it
//! so), every edge's copy list is a valid sequentialization of the
//! parallel copy set the φs demand (proved by simulating it, including
//! that it disturbs no other variable), and `var_of` is dense and in
//! canonical order. What it deliberately cannot do is second-guess the
//! *interference test itself*, which it shares; that is what the
//! test-only SSA interpreter is for — it evaluates the SSA function and
//! the (variables + edge copies) rendition side by side on seeded inputs
//! and compares every observable value. The interpreter, not `check`, is
//! this slice's oracle.
//!
//! # Contract
//!
//! [`out_of_ssa`] is pure, total and deterministic: every container is a
//! `BTree`, every tie is broken by a total order (ascending φ block VA,
//! then cell, then argument order for merges; ascending name id for
//! variable numbering), so equal inputs give byte-equal output and no
//! input panics. Malformed input is not laundered: a function failing
//! [`irssa::check`] gets the identity map — one variable per name, no
//! copies, zeroed stats — the refuse-don't-guess posture the earlier
//! slices established. Nothing here renders; spelling variables is the
//! pseudocode slice's `CellNamer` hook.

use std::collections::{BTreeMap, BTreeSet};

use crate::callfx;
use crate::ir::{Expr, Stmt};
use crate::irssa::{self, Cell, SsaFunction};

/// Pair comparisons [`out_of_ssa`] will spend on interference tests
/// before it stops coalescing and keeps the copies it has. Quality, not
/// correctness: an unmerged class is a copy, never a wrong value. Real
/// functions use a fraction of it — over the /bin/ls sweep the heaviest
/// function spends under 16 thousand comparisons, a sixty-fourth of the
/// cap — so it only bounds the quadratic blow-up an adversarial input
/// could ask for.
const COALESCE_FUEL: usize = 1 << 20;

// ---------------------------------------------------------------------------
// The rendition
// ---------------------------------------------------------------------------

/// One end of a copy: a variable, or the single temporary slot that
/// breaks a cycle in a parallel copy set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CopySlot {
    /// The variable with this id.
    Var(u32),
    /// The edge's temporary. Live only inside one edge's copy list:
    /// written before it is read, dead by the end of the list.
    Temp,
}

/// One copy to execute on a control-flow edge: `dst := src`.
///
/// Both ends are [`CopySlot`]s rather than the destination being a plain
/// variable id, because breaking a cycle needs the `temp := v` copy —
/// the swap case has no other spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Copy {
    /// Where the value lands.
    pub dst: CopySlot,
    /// Where it comes from.
    pub src: CopySlot,
}

/// The variable assignment and residual copies for one function: what a
/// renderer needs to print SSA as named locals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutOfSsa {
    /// SSA name id -> variable id. Dense (`0..var_count` all used), every
    /// name mapped — a name with no uses still gets its class's variable.
    pub var_of: Vec<u32>,
    /// Number of variables: the human-facing count, and one past the
    /// largest id in `var_of`.
    pub var_count: u32,
    /// Residual copies per CFG edge, keyed `(predecessor VA, successor
    /// VA)`, already sequentialized: execute them in order on that edge.
    /// An edge with nothing to do is absent, never an empty list.
    pub edge_copies: BTreeMap<(u64, u64), Vec<Copy>>,
    /// Copies for the virtual function-entry edge — the `None`-keyed
    /// arguments of a φ in the entry block, which exist only when the
    /// entry has real predecessors (a loop back to entry). That edge has
    /// no `(pred, succ)` key to hang them on, so it gets its own list: a
    /// prologue, executed once before the entry block. (The plan's
    /// struct had no slot for these; leaving them unrepresentable would
    /// have made `check` unable to insist every φ is resolved.)
    pub entry_copies: Vec<Copy>,
    /// Variables whose value is, at some point, asserted by the ABI
    /// rather than proven by the code: a name of the class is defined by
    /// a [`crate::callfx`] intrinsic write. See the module docs.
    pub assumed: BTreeSet<u32>,
    /// Variables read, at some point, at a width their definition never
    /// wrote — the [`SsaFunction::partial`] marker carried onto
    /// variables.
    pub partial: BTreeSet<u32>,
}

/// What [`out_of_ssa`] did, for the caller's stats line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OutStats {
    /// φ-nodes translated: every φ in the function, since each is either
    /// coalesced away or resolved by edge copies.
    pub phis_resolved: usize,
    /// Copies emitted, over every edge and the entry prologue.
    pub copies: usize,
    /// Successful class merges — φ-argument pairs that turned out not to
    /// interfere and now share a variable.
    pub coalesced: usize,
}

/// Why an [`OutOfSsa`] is not a valid rendition of its function.
/// [`check`] returns the first fault it finds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutFault {
    /// `var_of` has the wrong length, holds an id outside `var_count`,
    /// leaves an id unused, or numbers the classes out of canonical
    /// (ascending first-member) order.
    Vars,
    /// Two names sharing a variable interfere.
    Interfering { a: u16, b: u16 },
    /// A variable's names do not all belong to one cell.
    MixedCells { var: u32 },
    /// Copies recorded for something that is not an in-function CFG
    /// edge.
    UnknownEdge { edge: (u64, u64) },
    /// An edge's copy list is not a valid sequentialization of the
    /// parallel copy set its φs demand: it leaves a φ unresolved,
    /// disturbs a variable it should not, or reads the temporary before
    /// writing it. `None` names the virtual function-entry edge.
    BadSequence { edge: Option<(u64, u64)> },
    /// The recorded provenance sets are not the recomputed ones.
    Provenance,
    /// Input that fails [`irssa::check`] was given anything other than
    /// the identity posture.
    NotIdentity,
    /// Verification would cost more than any output of [`out_of_ssa`]
    /// can: the classes are larger than the pass's own coalescing budget
    /// could have built. Unreachable for this module's output.
    TooLarge,
}

impl std::fmt::Display for OutFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutFault::Vars => write!(f, "the variable map is not dense and canonical"),
            OutFault::Interfering { a, b } => {
                write!(f, "names {a} and {b} interfere but share a variable")
            }
            OutFault::MixedCells { var } => write!(f, "variable {var} mixes cells"),
            OutFault::UnknownEdge { edge } => {
                write!(f, "copies on the non-edge {:#x} -> {:#x}", edge.0, edge.1)
            }
            OutFault::BadSequence { edge } => match edge {
                Some((p, s)) => write!(f, "invalid copy sequence on {p:#x} -> {s:#x}"),
                None => write!(f, "invalid copy sequence on the function-entry edge"),
            },
            OutFault::Provenance => write!(f, "the provenance sets are wrong"),
            OutFault::NotIdentity => write!(f, "malformed input was not left as the identity map"),
            OutFault::TooLarge => write!(f, "the classes are too large to verify"),
        }
    }
}

// ---------------------------------------------------------------------------
// Analysis: definition sites, liveness, value equality, interference
// ---------------------------------------------------------------------------

/// Where a name is defined — the point its live range starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Site {
    /// The virtual function entry: a version-0 (at-entry) name.
    Entry,
    /// A φ at the head of `block`. All of a block's φs define in
    /// parallel, at one point.
    Phi { block: u64 },
    /// Statement `index` of `block`. An intrinsic's several writes all
    /// define at this one point, in parallel.
    Stmt { block: u64, index: usize },
}

/// Everything the interference test needs, recomputed from the
/// [`SsaFunction`] alone. Built by both [`out_of_ssa`] and [`check`], so
/// neither trusts the other's analysis.
struct Facts<'a> {
    f: &'a SsaFunction,
    cfg: irssa::Cfg,
    /// Definition site per name id.
    site: Vec<Site>,
    /// Names live on exit from each block: the union over successors of
    /// their live-in plus the φ arguments this edge supplies.
    live_out: BTreeMap<u64, BTreeSet<u16>>,
    /// Names live at the function-entry point: live-in of the entry
    /// block plus the arguments of any entry φ on the virtual edge.
    entry_live: BTreeSet<u16>,
    /// Highest statement index at which a name is read, per block. A
    /// missing key means the name is not read in that block at all.
    max_use: BTreeMap<(u64, u16), usize>,
    /// Value root per name: the end of its width-exact copy chain. Two
    /// names with one root always hold the same runtime value.
    root: Vec<u16>,
}

impl<'a> Facts<'a> {
    /// Derive the facts. `None` when the liveness fixpoint exhausts its
    /// fuel — an input so large or so strange that the analysis refuses
    /// rather than applying a half-computed (and therefore unsound)
    /// liveness. Callers fall back to the identity posture.
    fn build(f: &'a SsaFunction) -> Option<Facts<'a>> {
        let raw: BTreeMap<u64, Vec<u64>> = f
            .blocks
            .iter()
            .map(|(&va, b)| (va, b.successors.clone()))
            .collect();
        let cfg = irssa::Cfg::analyze(f.entry, &raw);
        let n = f.names.len();

        // Definition sites, per-block use and definition sets, and the
        // last-read index of every name in every block.
        let mut site = vec![Site::Entry; n];
        let mut defs: BTreeMap<u64, BTreeSet<u16>> = BTreeMap::new();
        let mut uses: BTreeMap<u64, BTreeSet<u16>> = BTreeMap::new();
        let mut max_use: BTreeMap<(u64, u16), usize> = BTreeMap::new();
        for (&va, block) in &f.blocks {
            let d = defs.entry(va).or_default();
            for phi in &block.phis {
                if let Some(slot) = site.get_mut(phi.dst as usize) {
                    *slot = Site::Phi { block: va };
                }
                d.insert(phi.dst);
            }
            let u = uses.entry(va).or_default();
            for (i, stmt) in block.stmts.iter().enumerate() {
                irssa::for_each_use(stmt, &mut |r| {
                    u.insert(r.num);
                    let slot = max_use.entry((va, r.num)).or_insert(i);
                    *slot = (*slot).max(i);
                });
            }
            let d = defs.entry(va).or_default();
            for (i, stmt) in block.stmts.iter().enumerate() {
                irssa::for_each_def(stmt, &mut |r| {
                    if let Some(slot) = site.get_mut(r.num as usize) {
                        *slot = Site::Stmt {
                            block: va,
                            index: i,
                        };
                    }
                    d.insert(r.num);
                });
            }
        }

        // Backward liveness over SSA names. A φ defines its destination
        // at the head of its block, and *uses* each argument at the end
        // of the predecessor that supplies it — the two conventions that
        // make the dominance-based interference test work.
        let empty_defs = BTreeSet::new();
        let mut live_in: BTreeMap<u64, BTreeSet<u16>> =
            f.blocks.keys().map(|&va| (va, BTreeSet::new())).collect();
        let mut live_out: BTreeMap<u64, BTreeSet<u16>> = live_in.clone();
        let mut work: BTreeSet<u64> = f.blocks.keys().copied().collect();
        let mut fuel = f
            .blocks
            .len()
            .saturating_mul(n.saturating_add(4))
            .saturating_add(1024)
            .min(1 << 24);
        while let Some(b) = work.pop_last() {
            if fuel == 0 {
                return None; // defensively unreachable: liveness converges
            }
            fuel -= 1;
            let mut out: BTreeSet<u16> = BTreeSet::new();
            for &s in cfg.succs.get(&b).map(Vec::as_slice).unwrap_or(&[]) {
                if let Some(set) = live_in.get(&s) {
                    out.extend(set.iter().copied());
                }
                let Some(sblock) = f.blocks.get(&s) else {
                    continue;
                };
                for phi in &sblock.phis {
                    for &(k, arg) in &phi.args {
                        if k == Some(b) {
                            out.insert(arg);
                        }
                    }
                }
            }
            let mut new_in = out.clone();
            for d in defs.get(&b).unwrap_or(&empty_defs) {
                new_in.remove(d);
            }
            if let Some(u) = uses.get(&b) {
                // A name read in a block and defined in the same block is
                // defined earlier (dominance), so it is not live-in.
                new_in.extend(
                    u.iter()
                        .copied()
                        .filter(|id| !defs.get(&b).unwrap_or(&empty_defs).contains(id)),
                );
            }
            live_out.insert(b, out);
            if live_in.get(&b) != Some(&new_in) {
                live_in.insert(b, new_in);
                for &p in cfg.preds.get(&b).map(Vec::as_slice).unwrap_or(&[]) {
                    work.insert(p);
                }
            }
        }

        let mut entry_live = live_in.get(&f.entry).cloned().unwrap_or_default();
        if let Some(block) = f.blocks.get(&f.entry) {
            for phi in &block.phis {
                for &(k, arg) in &phi.args {
                    if k.is_none() {
                        entry_live.insert(arg);
                    }
                }
            }
        }

        // Value equality: chase width-exact copies to their root. A copy
        // narrower or wider than its source would not carry the same
        // value, so only the exact ones join a chain.
        let mut copy_src: Vec<Option<u16>> = vec![None; n];
        for block in f.blocks.values() {
            for stmt in &block.stmts {
                let Stmt::Assign { dst, value } = stmt else {
                    continue;
                };
                let Expr::Reg(src) = value else { continue };
                let (Some(dn), Some(sn)) = (
                    f.names.get(dst.num as usize),
                    f.names.get(src.num as usize),
                ) else {
                    continue;
                };
                if dst.width == src.width
                    && dn.width == dst.width
                    && sn.width == src.width
                    && let Some(slot) = copy_src.get_mut(dst.num as usize)
                {
                    *slot = Some(src.num);
                }
            }
        }
        let mut root: Vec<u16> = Vec::with_capacity(n);
        for id in 0..n {
            let mut cur = id as u16;
            for _ in 0..=n {
                match copy_src.get(cur as usize).copied().flatten() {
                    Some(next) if next != cur => cur = next,
                    _ => break,
                }
            }
            root.push(cur);
        }

        Some(Facts {
            f,
            cfg,
            site,
            live_out,
            entry_live,
            max_use,
            root,
        })
    }

    /// The cell a name versions.
    fn cell(&self, id: u16) -> Option<Cell> {
        self.f.names.get(id as usize).map(|n| (n.space, n.cell))
    }

    /// Whether `a`'s definition point strictly precedes `b`'s on every
    /// path — the strict-SSA dominance order over definition sites.
    fn precedes(&self, a: Site, b: Site) -> bool {
        match (a, b) {
            (Site::Entry, Site::Entry) => false,
            (Site::Entry, _) => true,
            (_, Site::Entry) => false,
            (Site::Phi { block: x }, Site::Phi { block: y }) => self.cfg.strictly_dominates(x, y),
            (Site::Phi { block: x }, Site::Stmt { block: y, .. }) => {
                x == y || self.cfg.strictly_dominates(x, y)
            }
            (Site::Stmt { block: x, .. }, Site::Phi { block: y }) => {
                x != y && self.cfg.strictly_dominates(x, y)
            }
            (
                Site::Stmt {
                    block: x,
                    index: ix,
                },
                Site::Stmt {
                    block: y,
                    index: iy,
                },
            ) => {
                if x == y {
                    ix < iy
                } else {
                    self.cfg.strictly_dominates(x, y)
                }
            }
        }
    }

    /// Whether `id` is live at the point immediately after `at` — the
    /// query the interference test asks about the *dominating* name at
    /// the dominated name's definition.
    fn live_after(&self, id: u16, at: Site) -> bool {
        match at {
            Site::Entry => self.entry_live.contains(&id),
            Site::Phi { block } => {
                self.live_out.get(&block).is_some_and(|s| s.contains(&id))
                    || self.max_use.contains_key(&(block, id))
            }
            Site::Stmt { block, index } => {
                self.live_out.get(&block).is_some_and(|s| s.contains(&id))
                    || self
                        .max_use
                        .get(&(block, id))
                        .is_some_and(|&last| last > index)
            }
        }
    }

    /// Boissinot's value interference: live ranges intersect **and** the
    /// values differ. In strict SSA two live ranges intersect only if
    /// one definition dominates the other (or both define at the same
    /// parallel point), which is what makes this a constant-time query
    /// per pair instead of an interference graph.
    fn interferes(&self, a: u16, b: u16) -> bool {
        if a == b {
            return false;
        }
        if self.root.get(a as usize) == self.root.get(b as usize) {
            return false; // same value: sharing storage is harmless
        }
        let (Some(&sa), Some(&sb)) = (self.site.get(a as usize), self.site.get(b as usize)) else {
            return true; // an id outside the table: refuse to merge it
        };
        if self.precedes(sa, sb) {
            self.live_after(a, sb)
        } else if self.precedes(sb, sa) {
            self.live_after(b, sa)
        } else if sa == sb {
            // Defined in parallel at one point: they intersect exactly
            // when both survive it.
            self.live_after(a, sa) && self.live_after(b, sb)
        } else {
            false // neither dominates the other: disjoint ranges
        }
    }
}

// ---------------------------------------------------------------------------
// Congruence classes
// ---------------------------------------------------------------------------

/// φ-congruence classes (Sreedhar et al. 1999) as a flat partition: the
/// representative of a class is its smallest member id, so the numbering
/// that follows is canonical without a second sort.
struct Classes {
    of: Vec<u16>,
    members: BTreeMap<u16, BTreeSet<u16>>,
}

impl Classes {
    fn new(n: usize) -> Classes {
        Classes {
            of: (0..n as u16).collect(),
            members: (0..n as u16).map(|i| (i, BTreeSet::from([i]))).collect(),
        }
    }

    fn rep(&self, id: u16) -> u16 {
        self.of.get(id as usize).copied().unwrap_or(id)
    }

    fn class(&self, rep: u16) -> &BTreeSet<u16> {
        static EMPTY: BTreeSet<u16> = BTreeSet::new();
        self.members.get(&rep).unwrap_or(&EMPTY)
    }

    /// Merge two distinct classes, keeping the smaller representative.
    fn merge(&mut self, a: u16, b: u16) {
        let (keep, gone) = if a <= b { (a, b) } else { (b, a) };
        let Some(moved) = self.members.remove(&gone) else {
            return;
        };
        for &m in &moved {
            if let Some(slot) = self.of.get_mut(m as usize) {
                *slot = keep;
            }
        }
        self.members.entry(keep).or_default().extend(moved);
    }
}

/// Whether two whole classes can share one variable: same cell, and no
/// interfering pair. `fuel` bounds the comparisons; running out answers
/// "no", which costs a copy and never correctness.
fn mergeable(cls: &Classes, facts: &Facts<'_>, a: u16, b: u16, fuel: &mut usize) -> bool {
    // The cell invariant, enforced rather than assumed (see module docs).
    if facts.cell(a) != facts.cell(b) {
        return false;
    }
    for &x in cls.class(a) {
        for &y in cls.class(b) {
            if *fuel == 0 {
                return false;
            }
            *fuel -= 1;
            if facts.interferes(x, y) {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Parallel copy sequentialization
// ---------------------------------------------------------------------------

/// Order a parallel copy set (`dst -> src`, destinations distinct) into
/// a sequence with the same effect: Boissinot et al.'s Algorithm 1. A
/// copy whose destination is still needed as a source is deferred; a
/// cycle — the swap case — is broken by parking one value in
/// [`CopySlot::Temp`]. Identity copies contribute nothing.
///
/// Deterministic: both worklists are seeded in ascending destination
/// order and used as stacks.
fn sequentialize(pairs: &BTreeMap<u32, u32>) -> Vec<Copy> {
    // An identity copy is not a copy; dropping it here is what keeps a
    // variable that merely appears in the set from looking like a cycle.
    let work: BTreeMap<u32, u32> = pairs
        .iter()
        .filter(|(d, s)| d != s)
        .map(|(&d, &s)| (d, s))
        .collect();

    let mut out: Vec<Copy> = Vec::new();
    // `loc[v]` is where the value originally in `v` currently lives; a
    // destination that is nobody's source has no entry, which is exactly
    // the "free to be overwritten" condition.
    let mut loc: BTreeMap<u32, CopySlot> = BTreeMap::new();
    for &src in work.values() {
        loc.insert(src, CopySlot::Var(src));
    }
    let mut ready: Vec<u32> = work
        .keys()
        .copied()
        .filter(|d| !loc.contains_key(d))
        .rev()
        .collect();
    let mut to_do: Vec<u32> = work.keys().copied().rev().collect();
    let mut done: BTreeSet<u32> = BTreeSet::new();
    // Each destination is written once and each cycle costs one extra
    // copy, so neither loop can run longer than this; the cap is
    // defense, never reached.
    let mut fuel = 4 * work.len() + 8;

    loop {
        while let Some(d) = ready.pop() {
            if fuel == 0 {
                return out; // defensively unreachable
            }
            fuel -= 1;
            if !done.insert(d) {
                continue; // already written
            }
            let Some(&src) = work.get(&d) else { continue };
            let from = loc.get(&src).copied().unwrap_or(CopySlot::Var(src));
            if from != CopySlot::Var(d) {
                out.push(Copy {
                    dst: CopySlot::Var(d),
                    src: from,
                });
            }
            loc.insert(src, CopySlot::Var(d));
            // The value that was in `src` now lives in `d` as well, so
            // if `src` is itself waiting to be overwritten, it is free.
            if from == CopySlot::Var(src) && work.contains_key(&src) && !done.contains(&src) {
                ready.push(src);
            }
        }
        let Some(b) = to_do.pop() else { break };
        if done.contains(&b) {
            continue;
        }
        // Nothing is free and `b` still owes its value: `b` sits on a
        // cycle. Park what it holds in the temporary — the swap case.
        out.push(Copy {
            dst: CopySlot::Temp,
            src: CopySlot::Var(b),
        });
        loc.insert(b, CopySlot::Temp);
        ready.push(b);
    }
    out
}

/// Execute a copy list against the parallel copy set it claims to
/// implement, returning whether it does. The state maps a variable to
/// the *original* content it now holds; a variable never written keeps
/// its own. Also proves the temporary is written before it is read and
/// that no variable outside the set is disturbed.
fn simulate(pairs: &BTreeMap<u32, u32>, copies: &[Copy]) -> bool {
    let mut state: BTreeMap<u32, u32> = BTreeMap::new();
    let mut temp: Option<u32> = None;
    let mut written: BTreeSet<u32> = BTreeSet::new();
    for copy in copies {
        let value = match copy.src {
            CopySlot::Var(v) => state.get(&v).copied().unwrap_or(v),
            CopySlot::Temp => match temp {
                Some(v) => v,
                None => return false, // read before write
            },
        };
        match copy.dst {
            CopySlot::Var(v) => {
                state.insert(v, value);
                written.insert(v);
            }
            CopySlot::Temp => temp = Some(value),
        }
    }
    // Every variable the sequence touched must hold what the parallel
    // semantics say, and every destination the set names must too.
    for v in written.iter().copied().chain(pairs.keys().copied()) {
        let want = pairs.get(&v).copied().unwrap_or(v);
        if state.get(&v).copied().unwrap_or(v) != want {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

/// One variable per name, no copies: the posture for input this module
/// refuses to interpret.
fn identity(f: &SsaFunction) -> OutOfSsa {
    let n = f.names.len() as u32;
    OutOfSsa {
        var_of: (0..n).collect(),
        var_count: n,
        edge_copies: BTreeMap::new(),
        entry_copies: Vec::new(),
        assumed: BTreeSet::new(),
        partial: BTreeSet::new(),
    }
}

/// Translate a function out of SSA: assign every name a variable and
/// place the copies the φs still need. See the module docs for the
/// algorithm, the provenance carriers, and the contract.
pub fn out_of_ssa(f: &SsaFunction) -> (OutOfSsa, OutStats) {
    if irssa::check(f).is_err() {
        return (identity(f), OutStats::default());
    }
    let Some(facts) = Facts::build(f) else {
        return (identity(f), OutStats::default());
    };

    // Step 1 and 2: singleton classes (φs isolated, correct by
    // construction), then coalesce in the deterministic order — φ block
    // VA ascending, φs in cell order, arguments in edge order.
    let mut cls = Classes::new(f.names.len());
    let mut fuel = COALESCE_FUEL;
    let mut stats = OutStats::default();
    for block in f.blocks.values() {
        for phi in &block.phis {
            stats.phis_resolved += 1;
            for &(_, arg) in &phi.args {
                let (ra, rb) = (cls.rep(phi.dst), cls.rep(arg));
                if ra == rb {
                    continue;
                }
                if mergeable(&cls, &facts, ra, rb, &mut fuel) {
                    cls.merge(ra, rb);
                    stats.coalesced += 1;
                }
            }
        }
    }

    // Dense variable ids in ascending first-member order.
    let mut var_of: Vec<u32> = Vec::with_capacity(f.names.len());
    let mut id_of: BTreeMap<u16, u32> = BTreeMap::new();
    for id in 0..f.names.len() as u16 {
        let rep = cls.rep(id);
        let next = id_of.len() as u32;
        var_of.push(*id_of.entry(rep).or_insert(next));
    }
    let var_count = id_of.len() as u32;
    let var = |id: u16| var_of.get(id as usize).copied().unwrap_or(0);

    // Step 3: the copies each φ argument still needs, gathered per edge
    // as a parallel set, then sequentialized.
    let mut per_edge: BTreeMap<(u64, u64), BTreeMap<u32, u32>> = BTreeMap::new();
    let mut entry_pairs: BTreeMap<u32, u32> = BTreeMap::new();
    for (&va, block) in &f.blocks {
        for phi in &block.phis {
            let dst = var(phi.dst);
            for &(k, arg) in &phi.args {
                let src = var(arg);
                if src == dst {
                    continue; // coalesced: the φ is already honored
                }
                match k {
                    Some(p) => {
                        per_edge.entry((p, va)).or_default().insert(dst, src);
                    }
                    None => {
                        entry_pairs.insert(dst, src);
                    }
                }
            }
        }
    }
    let mut edge_copies: BTreeMap<(u64, u64), Vec<Copy>> = BTreeMap::new();
    for (edge, pairs) in &per_edge {
        let list = sequentialize(pairs);
        if !list.is_empty() {
            stats.copies += list.len();
            edge_copies.insert(*edge, list);
        }
    }
    let entry_copies = sequentialize(&entry_pairs);
    stats.copies += entry_copies.len();

    let (assumed, partial) = provenance(f, &var_of);
    (
        OutOfSsa {
            var_of,
            var_count,
            edge_copies,
            entry_copies,
            assumed,
            partial,
        },
        stats,
    )
}

/// The two honesty markers carried from names onto variables: ABI
/// assumption from [`crate::callfx`] intrinsic writes, partiality from
/// the [`SsaFunction::partial`] positions. See the module docs for why
/// these are the carriers.
fn provenance(f: &SsaFunction, var_of: &[u32]) -> (BTreeSet<u32>, BTreeSet<u32>) {
    let var = |id: u16| var_of.get(id as usize).copied();
    let mut assumed = BTreeSet::new();
    for block in f.blocks.values() {
        for stmt in &block.stmts {
            let Stmt::Intrinsic { name, writes, .. } = stmt else {
                continue;
            };
            if *name != callfx::EFFECT_NAME {
                continue;
            }
            for w in writes {
                assumed.extend(var(w.num));
            }
        }
    }
    let mut partial = BTreeSet::new();
    for &(va, index) in &f.partial {
        let Some(stmt) = f.blocks.get(&va).and_then(|b| b.stmts.get(index)) else {
            continue;
        };
        irssa::for_each_use(stmt, &mut |r| {
            let wider = f
                .names
                .get(r.num as usize)
                .is_some_and(|n| r.width.bits() > n.width.bits());
            if wider {
                partial.extend(var(r.num));
            }
        });
    }
    (assumed, partial)
}

// ---------------------------------------------------------------------------
// Well-formedness
// ---------------------------------------------------------------------------

/// Validate a rendition against its function from scratch: interference
/// recomputed from dominance, liveness and value equality; every φ
/// resolved; every copy list a valid sequentialization; `var_of` dense
/// and canonical; provenance exact. Total and side-effect-free, never
/// panics, and returns the first [`OutFault`].
pub fn check(f: &SsaFunction, out: &OutOfSsa) -> Result<(), OutFault> {
    let facts = match irssa::check(f) {
        Ok(()) => Facts::build(f),
        Err(_) => None,
    };
    let Some(facts) = facts else {
        // Refused input: the identity posture, exactly.
        return if *out == identity(f) {
            Ok(())
        } else {
            Err(OutFault::NotIdentity)
        };
    };

    // `var_of` is total, in range, dense, and numbered in ascending
    // first-member order (the canonical numbering).
    if out.var_of.len() != f.names.len() || out.var_count as usize > out.var_of.len() {
        return Err(OutFault::Vars); // a dense map has no more variables than names
    }
    let mut first: Vec<Option<u16>> = vec![None; out.var_count as usize];
    for (id, &v) in out.var_of.iter().enumerate() {
        let Some(slot) = first.get_mut(v as usize) else {
            return Err(OutFault::Vars);
        };
        if slot.is_none() {
            *slot = Some(id as u16);
        }
    }
    let mut prev: Option<u16> = None;
    for slot in &first {
        match (*slot, prev) {
            (None, _) => return Err(OutFault::Vars), // an unused id: not dense
            (Some(f0), Some(p)) if f0 <= p => return Err(OutFault::Vars),
            (Some(f0), _) => prev = Some(f0),
        }
    }

    // No two names sharing a variable interfere, and no variable mixes
    // cells.
    let mut classes: BTreeMap<u32, Vec<u16>> = BTreeMap::new();
    for (id, &v) in out.var_of.iter().enumerate() {
        classes.entry(v).or_default().push(id as u16);
    }
    let mut fuel = 2 * COALESCE_FUEL + f.names.len() + 16;
    for (&v, members) in &classes {
        for (i, &a) in members.iter().enumerate() {
            if facts.cell(a) != facts.cell(members[0]) {
                return Err(OutFault::MixedCells { var: v });
            }
            for &b in &members[i + 1..] {
                if fuel == 0 {
                    return Err(OutFault::TooLarge);
                }
                fuel -= 1;
                if facts.interferes(a, b) {
                    return Err(OutFault::Interfering { a, b });
                }
            }
        }
    }

    // Every φ resolved, and every copy list a valid sequentialization of
    // the parallel set the φs demand.
    let var = |id: u16| out.var_of.get(id as usize).copied().unwrap_or(u32::MAX);
    let mut per_edge: BTreeMap<(u64, u64), BTreeMap<u32, u32>> = BTreeMap::new();
    let mut entry_pairs: BTreeMap<u32, u32> = BTreeMap::new();
    for (&va, block) in &f.blocks {
        for phi in &block.phis {
            let dst = var(phi.dst);
            for &(k, arg) in &phi.args {
                let src = var(arg);
                if src == dst {
                    continue;
                }
                match k {
                    Some(p) => {
                        per_edge.entry((p, va)).or_default().insert(dst, src);
                    }
                    None => {
                        entry_pairs.insert(dst, src);
                    }
                }
            }
        }
    }
    for edge in out.edge_copies.keys() {
        let real = facts
            .cfg
            .succs
            .get(&edge.0)
            .is_some_and(|l| l.contains(&edge.1));
        if !real {
            return Err(OutFault::UnknownEdge { edge: *edge });
        }
    }
    let empty: Vec<Copy> = Vec::new();
    let mut edges: BTreeSet<(u64, u64)> = per_edge.keys().copied().collect();
    edges.extend(out.edge_copies.keys().copied());
    for edge in edges {
        let pairs = per_edge.remove(&edge).unwrap_or_default();
        let copies = out.edge_copies.get(&edge).unwrap_or(&empty);
        if !simulate(&pairs, copies) {
            return Err(OutFault::BadSequence { edge: Some(edge) });
        }
    }
    if !simulate(&entry_pairs, &out.entry_copies) {
        return Err(OutFault::BadSequence { edge: None });
    }

    let (assumed, partial) = provenance(f, &out.var_of);
    if assumed != out.assumed || partial != out.partial {
        return Err(OutFault::Provenance);
    }
    Ok(())
}

// `pub(crate)` so the evaluation harness (`evalfx`) can drive the
// interpreter below as its semantic oracle. Still `#[cfg(test)]`: the
// module, and the interpreter with it, exist only under test and never
// for a dependent.
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ir::{BinOp, BranchKind, Flag, Reg, Space, UnOp, Width};
    use crate::model::Arch;
    use crate::{irflow, irlift, irssaopt};

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

    /// Construct SSA and insist it is well-formed — every fixture's
    /// starting point.
    fn build(f: &irlift::LiftedFunction) -> SsaFunction {
        let ssa = irssa::construct(f).expect("well-formed input constructs");
        assert_eq!(irssa::check(&ssa), Ok(()), "input SSA must check");
        ssa
    }

    /// The pipeline this module is meant to consume: construction, then
    /// propagation and expression forwarding. Both are what make the
    /// lost-copy shape reachable — they move a φ's *value* into a block
    /// past the point where the φ's cell was redefined, which plain
    /// renaming can never do (the reaching definition there is the
    /// redefinition, so the shape needs a pass that reasons about
    /// values, exactly as the published example needs copy propagation).
    fn pipeline(f: &irlift::LiftedFunction) -> SsaFunction {
        let ssa = build(f);
        let (opt, _) = irssaopt::optimize(&ssa);
        let (fwd, _) = irssaopt::forward(&opt);
        assert_eq!(irssa::check(&fwd), Ok(()), "pipeline output must check");
        fwd
    }

    /// Forwarding only. Expression forwarding is what makes live ranges
    /// of one cell overlap at all; propagation additionally *normalizes*
    /// copies — it rewrites a φ argument to its same-cell root — which
    /// reaches a low variable count by a different route and would hide
    /// the coalescer's own value test. This prefix of the pipeline keeps
    /// the copy chains standing so that test is what decides.
    fn forwarded(f: &irlift::LiftedFunction) -> SsaFunction {
        let (fwd, _) = irssaopt::forward(&build(f));
        assert_eq!(irssa::check(&fwd), Ok(()), "forwarded output must check");
        fwd
    }

    /// Translate, insisting on the module's promises: the rendition
    /// checks out, it is byte-reproducible, and the input is untouched.
    fn translate(f: &SsaFunction) -> (OutOfSsa, OutStats) {
        let before = f.clone();
        let (out, stats) = out_of_ssa(f);
        assert_eq!(f, &before, "the input must never be mutated");
        assert_eq!(check(f, &out), Ok(()), "the rendition must check");
        let (again, stats2) = out_of_ssa(f);
        assert_eq!(out, again, "translation must be deterministic");
        assert_eq!(stats, stats2);
        (out, stats)
    }

    /// The name of `cell` at `version`, for assertions.
    fn name_of(f: &SsaFunction, cell: u16, version: u32) -> u16 {
        f.names
            .iter()
            .position(|n| n.space == Space::Arch && n.cell == cell && n.version == version)
            .unwrap_or_else(|| panic!("no arch cell {cell} version {version}")) as u16
    }

    /// The φ destination for `cell` in `block`.
    fn phi_dst(f: &SsaFunction, block: u64, cell: u16) -> u16 {
        f.blocks[&block]
            .phis
            .iter()
            .find(|p| f.names[p.dst as usize].cell == cell)
            .unwrap_or_else(|| panic!("no phi for cell {cell} in {block:#x}"))
            .dst
    }

    fn total_copies(out: &OutOfSsa) -> usize {
        out.edge_copies.values().map(Vec::len).sum::<usize>() + out.entry_copies.len()
    }

    fn phi_count(f: &SsaFunction) -> usize {
        f.blocks.values().map(|b| b.phis.len()).sum()
    }

    // -- the interpreter: this slice's oracle ------------------------------

    /// xorshift64*, seeded and deterministic — no wall clock, the same
    /// generator the `irssaopt` sweeps use.
    fn next(s: &mut u64) -> u64 {
        *s ^= *s >> 12;
        *s ^= *s << 25;
        *s ^= *s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Side {
        /// The SSA function, evaluated over names and φ-nodes.
        Ssa,
        /// The rendition, evaluated over variables and edge copies.
        Out,
    }

    /// A tiny SSA interpreter that runs both readings of one function
    /// side by side: the SSA semantics (names, parallel φs) and the
    /// out-of-SSA rendition (variables, sequentialized edge copies).
    /// Every value either side computes is compared with the other's, so
    /// a lost copy or a mis-ordered swap shows up as a divergence at the
    /// first read that can see it. Memory is a `BTreeMap` per side —
    /// genuinely independent state, not one shared map — intrinsics
    /// havoc their writes from the seed, and the walk is step-capped so
    /// a loop terminates.
    ///
    /// Reads are compared masked to the bits the *definition* guarantees:
    /// a read wider than its definition (an [`SsaFunction::partial`]
    /// position) reads bits nobody wrote, and neither reading of the
    /// function owes the other an answer there.
    pub(crate) struct Interp<'a> {
        f: &'a SsaFunction,
        out: &'a OutOfSsa,
        names: BTreeMap<u16, u64>,
        vars: BTreeMap<u32, u64>,
        temp: Option<u64>,
        mem: [BTreeMap<u64, u64>; 2],
        havoc: BTreeMap<u16, u64>,
        seed: u64,
        faults: Vec<String>,
    }

    impl<'a> Interp<'a> {
        pub(crate) fn new(f: &'a SsaFunction, out: &'a OutOfSsa, seed: u64) -> Interp<'a> {
            let mut it = Interp {
                f,
                out,
                names: BTreeMap::new(),
                vars: BTreeMap::new(),
                temp: None,
                mem: [BTreeMap::new(), BTreeMap::new()],
                havoc: BTreeMap::new(),
                seed,
                faults: Vec::new(),
            };
            // The at-entry values: the same on both sides by definition.
            let mut s = seed | 1;
            for &id in &f.live_in {
                let w = f.names[id as usize].width;
                let v = next(&mut s) & w.mask();
                it.write(id, v);
            }
            it
        }

        fn var_of(&self, id: u16) -> u32 {
            self.out.var_of.get(id as usize).copied().unwrap_or(u32::MAX)
        }

        fn width_of(&self, id: u16) -> Width {
            self.f
                .names
                .get(id as usize)
                .map(|n| n.width)
                .unwrap_or(Width::W64)
        }

        /// Define `id` on both sides at once.
        fn write(&mut self, id: u16, value: u64) {
            let v = value & self.width_of(id).mask();
            self.names.insert(id, v);
            let var = self.var_of(id);
            self.vars.insert(var, v);
        }

        /// One side's view of a register read, masked to the bits its
        /// definition guarantees.
        fn look(&mut self, r: Reg, side: Side) -> u64 {
            let mask = r.width.mask() & self.width_of(r.num).mask();
            let raw = match side {
                Side::Ssa => self.names.get(&r.num).copied(),
                Side::Out => self.vars.get(&self.var_of(r.num)).copied(),
            };
            match raw {
                Some(v) => v & mask,
                None => {
                    self.faults
                        .push(format!("read of undefined name {}", r.num));
                    0
                }
            }
        }

        /// Substitute one side's values into `e`, resolving loads from
        /// that side's memory, so the fold below sees a closed term.
        fn subst(&mut self, e: &Expr, side: Side, depth: usize) -> Expr {
            if depth > 64 {
                return e.clone();
            }
            match e {
                Expr::Const { .. } => e.clone(),
                Expr::Reg(r) => {
                    let v = self.look(*r, side);
                    Expr::constant(v & r.width.mask(), r.width)
                }
                Expr::Load { addr, width } => {
                    let a = self.subst(addr, side, depth + 1);
                    match irflow::fold_expr(&a) {
                        Expr::Const { value, .. } => {
                            let m = &self.mem[usize::from(side == Side::Out)];
                            let v = m.get(&value).copied().unwrap_or(0);
                            Expr::constant(v & width.mask(), *width)
                        }
                        other => Expr::load(other, *width),
                    }
                }
                Expr::Unary { op, operand } => {
                    Expr::unary(*op, self.subst(operand, side, depth + 1))
                }
                Expr::Binary { op, lhs, rhs } => Expr::binary(
                    *op,
                    self.subst(lhs, side, depth + 1),
                    self.subst(rhs, side, depth + 1),
                ),
            }
        }

        /// Evaluate on one side, reusing [`irflow::fold_expr`] so the
        /// interpreter cannot drift from the IR's own semantics. `None`
        /// is an expression the folder refuses (division by zero), which
        /// both sides refuse identically.
        fn eval(&mut self, e: &Expr, side: Side) -> Option<u64> {
            let sub = self.subst(e, side, 0);
            match irflow::fold_expr(&sub) {
                Expr::Const { value, .. } => Some(value),
                _ => None,
            }
        }

        /// Evaluate on both sides and record a divergence.
        fn eval_both(&mut self, e: &Expr, what: &str) -> (u64, u64) {
            let a = self.eval(e, Side::Ssa);
            let b = self.eval(e, Side::Out);
            if a != b {
                self.faults
                    .push(format!("{what}: ssa {a:?} vs rendition {b:?}"));
            }
            (a.unwrap_or(0), b.unwrap_or(0))
        }

        /// Take the edge `prev -> block`: the SSA side evaluates the φs
        /// in parallel, the rendition side executes the edge's copies.
        fn transition(&mut self, prev: Option<u64>, block: u64) {
            let f = self.f;
            let Some(b) = f.blocks.get(&block) else {
                return;
            };
            // The SSA side reads every argument before writing any
            // destination: φs are parallel.
            let mut landing: Vec<(u16, u64)> = Vec::new();
            for phi in &b.phis {
                let Some(&(_, arg)) = phi.args.iter().find(|&&(k, _)| k == prev) else {
                    self.faults
                        .push(format!("no phi argument for the edge into {block:#x}"));
                    continue;
                };
                match self.names.get(&arg).copied() {
                    Some(v) => landing.push((phi.dst, v)),
                    None => self.faults.push(format!("phi argument {arg} is undefined")),
                }
            }
            // The rendition side.
            let copies = match prev {
                Some(p) => self
                    .out
                    .edge_copies
                    .get(&(p, block))
                    .cloned()
                    .unwrap_or_default(),
                None => self.out.entry_copies.clone(),
            };
            self.temp = None;
            for copy in copies {
                let v = match copy.src {
                    CopySlot::Var(v) => self.vars.get(&v).copied(),
                    CopySlot::Temp => self.temp,
                };
                let Some(v) = v else {
                    self.faults.push(format!(
                        "a copy reads an undefined slot on the edge into {block:#x}"
                    ));
                    continue;
                };
                match copy.dst {
                    CopySlot::Var(d) => {
                        self.vars.insert(d, v);
                    }
                    CopySlot::Temp => self.temp = Some(v),
                }
            }
            for (dst, v) in landing {
                let w = self.width_of(dst).mask();
                self.names.insert(dst, v & w);
            }
            // Every φ must now hold the same value on both sides — the
            // direct test that the rendition resolved it.
            let dsts: Vec<u16> = f.blocks[&block].phis.iter().map(|p| p.dst).collect();
            for dst in dsts {
                let w = self.width_of(dst).mask();
                let a = self.names.get(&dst).copied().map(|v| v & w);
                let b = self.vars.get(&self.var_of(dst)).copied().map(|v| v & w);
                if a != b {
                    self.faults
                        .push(format!("phi {dst} at {block:#x}: ssa {a:?} vs rendition {b:?}"));
                }
            }
        }

        /// Run from the entry for at most `steps` blocks, picking a
        /// successor pseudo-randomly (both readings follow one path, so
        /// the oracle is the values, not the control flow).
        pub(crate) fn run(&mut self, steps: usize) -> Vec<String> {
            let f = self.f;
            let mut rng = self.seed | 3;
            let mut prev: Option<u64> = None;
            let mut cur = f.entry;
            for _ in 0..steps {
                if !f.blocks.contains_key(&cur) {
                    break;
                }
                self.transition(prev, cur);
                let stmts = f.blocks[&cur].stmts.clone();
                for stmt in &stmts {
                    self.step(stmt);
                }
                let succs: Vec<u64> = f.blocks[&cur]
                    .successors
                    .iter()
                    .copied()
                    .filter(|s| f.blocks.contains_key(s))
                    .collect();
                if succs.is_empty() {
                    break;
                }
                prev = Some(cur);
                cur = succs[(next(&mut rng) % succs.len() as u64) as usize];
            }
            if self.mem[0] != self.mem[1] {
                self.faults.push("memory diverged".to_string());
            }
            std::mem::take(&mut self.faults)
        }

        fn step(&mut self, stmt: &Stmt) {
            match stmt {
                Stmt::Assign { dst, value } => {
                    let (a, b) = self.eval_both(value, "assign");
                    self.names.insert(dst.num, a & dst.width.mask());
                    let var = self.var_of(dst.num);
                    self.vars.insert(var, b & dst.width.mask());
                }
                Stmt::Store { addr, value } => {
                    let (aa, ab) = self.eval_both(addr, "store address");
                    let (va, vb) = self.eval_both(value, "store value");
                    self.mem[0].insert(aa, va);
                    self.mem[1].insert(ab, vb);
                }
                Stmt::Branch { cond, target, .. } => {
                    if let Some(cond) = cond {
                        self.eval_both(cond, "branch condition");
                    }
                    self.eval_both(target, "branch target");
                }
                Stmt::Intrinsic { writes, reads, .. } => {
                    for (i, r) in reads.iter().enumerate() {
                        self.eval_both(r, &format!("intrinsic read {i}"));
                    }
                    // Havoc: unknowable, but identical on both sides and
                    // fresh on each execution of the same write.
                    for w in writes {
                        let count = self.havoc.entry(w.num).or_insert(0);
                        *count += 1;
                        let mut s = self
                            .seed
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .wrapping_add(u64::from(w.num) << 8)
                            .wrapping_add(*count)
                            | 1;
                        let v = next(&mut s) & w.width.mask();
                        self.write(w.num, v);
                    }
                }
            }
        }
    }

    /// Interpret both readings on several seeds; no divergence allowed.
    fn interpret(f: &SsaFunction, out: &OutOfSsa) {
        for seed in [0x1234_5678_9ABC_DEF0u64, 0xDEAD_BEEF_CAFE_F00D, 7, 99] {
            let faults = Interp::new(f, out, seed).run(64);
            assert!(faults.is_empty(), "seed {seed:#x}: {faults:?}");
        }
    }

    /// Translate, check, and interpret — the full battery a fixture gets.
    fn verify(f: &SsaFunction) -> (OutOfSsa, OutStats) {
        let (out, stats) = translate(f);
        interpret(f, &out);
        (out, stats)
    }

    // -- 1: the lost copy (Briggs et al. 1998) ------------------------------

    /// The lost-copy shape. `rax` is merged at the loop header, the loop
    /// body redefines it, and the *header's* value is still read after
    /// the loop — which reaches the SSA only through propagation, which
    /// rewrites `rcx := rbx` into a read of the φ. Coalescing the φ with
    /// the back-edge argument here is exactly the Briggs miscompile: the
    /// exit would read the incremented value.
    fn loop_family(body: Expr) -> irlift::LiftedFunction {
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
                        assign(ra(0, Width::W64), body),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1010, Width::W64),
                        },
                    ],
                    vec![0x1010, 0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(6, Width::W64), read(ra(3, Width::W64)))],
                    vec![],
                ),
            ],
        )
    }

    /// The loop body recomputes the merged cell: the φ's value and the
    /// back edge's differ, and the φ's value is still needed after the
    /// loop, so the two cannot share a variable.
    fn lost_copy() -> irlift::LiftedFunction {
        loop_family(bin(
            BinOp::Add,
            read(ra(0, Width::W64)),
            c(1, Width::W64),
        ))
    }

    /// Fuse two names' variables and drop every copy — the naive
    /// destruction, so a fixture can prove it is really discriminating.
    fn fused(f: &SsaFunction, out: &OutOfSsa, a: u16, b: u16) -> OutOfSsa {
        let (va, vb) = (out.var_of[a as usize], out.var_of[b as usize]);
        let mut var_of: Vec<u32> = out
            .var_of
            .iter()
            .map(|&v| if v == vb { va } else { v })
            .collect();
        let mut id_of: BTreeMap<u32, u32> = BTreeMap::new();
        for v in var_of.iter_mut() {
            let next = id_of.len() as u32;
            *v = *id_of.entry(*v).or_insert(next);
        }
        let var_count = id_of.len() as u32;
        let (assumed, partial) = provenance(f, &var_of);
        OutOfSsa {
            var_of,
            var_count,
            edge_copies: BTreeMap::new(),
            entry_copies: Vec::new(),
            assumed,
            partial,
        }
    }

    #[test]
    fn a_lost_copy_splits_the_phi_from_its_back_edge_argument() {
        let f = pipeline(&lost_copy());
        let (out, stats) = verify(&f);

        let phi = phi_dst(&f, 0x1010, 0);
        let back = name_of(&f, 0, 2); // the loop body's redefinition
        assert_ne!(
            out.var_of[phi as usize], out.var_of[back as usize],
            "the φ and the value that flows round the back edge interfere"
        );
        // The at-entry value merges with the φ, so only the back edge
        // pays: one copy, on that edge, in that direction.
        assert_eq!(
            out.var_of[name_of(&f, 0, 0) as usize],
            out.var_of[phi as usize]
        );
        assert_eq!(
            out.edge_copies.get(&(0x1020, 0x1010)).map(Vec::as_slice),
            Some(
                [Copy {
                    dst: CopySlot::Var(out.var_of[phi as usize]),
                    src: CopySlot::Var(out.var_of[back as usize]),
                }]
                .as_slice()
            )
        );
        assert_eq!(total_copies(&out), 1);
        assert_eq!(stats.copies, 1);
    }

    #[test]
    fn coalescing_a_lost_copy_anyway_is_caught_by_check_and_the_interpreter() {
        let f = pipeline(&lost_copy());
        let (good, _) = out_of_ssa(&f);
        let phi = phi_dst(&f, 0x1010, 0);
        let back = name_of(&f, 0, 2);
        let naive = fused(&f, &good, phi, back);

        assert!(
            matches!(check(&f, &naive), Err(OutFault::Interfering { .. })),
            "{:?}",
            check(&f, &naive)
        );
        let diverged = [1u64, 2, 3, 5, 8, 13, 21, 34]
            .into_iter()
            .any(|seed| !Interp::new(&f, &naive, seed).run(64).is_empty());
        assert!(diverged, "the interpreter must observe the lost copy");
    }

    // -- 2: the swap (Briggs et al. 1998) -----------------------------------

    #[test]
    fn a_swap_sequentializes_through_the_temporary() {
        let pairs = BTreeMap::from([(1u32, 2u32), (2, 1)]);
        let seq = sequentialize(&pairs);
        assert_eq!(
            seq,
            vec![
                Copy {
                    dst: CopySlot::Temp,
                    src: CopySlot::Var(1)
                },
                Copy {
                    dst: CopySlot::Var(1),
                    src: CopySlot::Var(2)
                },
                Copy {
                    dst: CopySlot::Var(2),
                    src: CopySlot::Temp
                },
            ]
        );
        assert!(simulate(&pairs, &seq));

        // The naive emission — copies in an arbitrary order — is the
        // published miscompile, and the simulator says so.
        let naive = vec![
            Copy {
                dst: CopySlot::Var(1),
                src: CopySlot::Var(2),
            },
            Copy {
                dst: CopySlot::Var(2),
                src: CopySlot::Var(1),
            },
        ];
        assert!(!simulate(&pairs, &naive));
        assert!(!simulate(&pairs, &[]));
    }

    #[test]
    fn a_three_cycle_and_a_chain_sequentialize_with_one_temporary() {
        // A 3-cycle: one temporary is enough, whatever the length.
        let cycle = BTreeMap::from([(1u32, 2u32), (2, 3), (3, 1)]);
        let seq = sequentialize(&cycle);
        assert!(simulate(&cycle, &seq));
        assert_eq!(
            seq.iter()
                .filter(|c| c.dst == CopySlot::Temp || c.src == CopySlot::Temp)
                .count(),
            2,
            "one save and one restore"
        );

        // A chain needs no temporary at all, only the right order.
        let chain = BTreeMap::from([(1u32, 2u32), (2, 3)]);
        let seq = sequentialize(&chain);
        assert!(simulate(&chain, &seq));
        assert!(seq.iter().all(|c| c.dst != CopySlot::Temp));
        assert_eq!(seq.len(), 2);

        // Identity copies contribute nothing.
        assert!(sequentialize(&BTreeMap::from([(4u32, 4u32)])).is_empty());
        assert!(sequentialize(&BTreeMap::new()).is_empty());
    }

    #[test]
    fn random_parallel_copy_sets_sequentialize_correctly() {
        let mut s = 0x0DDB_5BAD_5EED_1A5Eu64;
        for _ in 0..2000 {
            let width = 1 + next(&mut s) % 6;
            let mut pairs: BTreeMap<u32, u32> = BTreeMap::new();
            for dst in 0..width as u32 {
                match next(&mut s) % 4 {
                    0 => {} // not a destination at all
                    _ => {
                        pairs.insert(dst, (next(&mut s) % width) as u32);
                    }
                }
            }
            let seq = sequentialize(&pairs);
            assert!(simulate(&pairs, &seq), "{pairs:?} -> {seq:?}");
            // At most one save per cycle plus one copy per destination.
            assert!(seq.len() <= 2 * pairs.len(), "{pairs:?} -> {seq:?}");
        }
    }

    /// A register swap in a loop: the shape that produces a φ
    /// permutation in a general SSA. Under `irssa`'s per-cell φs the
    /// copies stay one-per-cell (see the module docs), and the result
    /// must still be correct — which the interpreter proves.
    #[test]
    fn a_register_swap_in_a_loop_stays_correct_end_to_end() {
        let f = pipeline(&func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(ra(0, Width::W64), c(1, Width::W64)),
                        assign(ra(3, Width::W64), c(2, Width::W64)),
                    ],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![
                        assign(Reg::temp(0, Width::W64), read(ra(0, Width::W64))),
                        assign(ra(0, Width::W64), read(ra(3, Width::W64))),
                        assign(ra(3, Width::W64), read(Reg::temp(0, Width::W64))),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1010, Width::W64),
                        },
                    ],
                    vec![0x1010, 0x1020],
                ),
                block(
                    0x1020,
                    vec![Stmt::Store {
                        addr: read(ra(0, Width::W64)),
                        value: read(ra(3, Width::W64)),
                    }],
                    vec![],
                ),
            ],
        ));
        assert_eq!(phi_count(&f), 2, "one φ per swapped cell");
        let (out, _) = verify(&f);
        for list in out.edge_copies.values() {
            assert!(
                list.iter().all(|c| c.dst != CopySlot::Temp),
                "per-cell φs cannot form a permutation: {list:?}"
            );
        }
    }

    // -- 3: the exit-criterion shapes ---------------------------------------

    /// A diamond with the merged cell live at the join — `irssa`'s own
    /// fixture shape.
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

    #[test]
    fn a_diamond_phi_web_coalesces_to_one_variable_with_no_copies() {
        let f = build(&diamond());
        let (out, stats) = verify(&f);
        assert_eq!(phi_count(&f), 1);
        assert_eq!(total_copies(&out), 0, "the exit criterion for a diamond");
        assert_eq!(stats.copies, 0);
        assert_eq!(stats.phis_resolved, 1);

        // The whole φ web — both arms and the merge — is one variable.
        let merged = out.var_of[phi_dst(&f, 0x1030, 0) as usize];
        for version in 2..=3 {
            assert_eq!(out.var_of[name_of(&f, 0, version) as usize], merged);
        }
        // Five names, three variables: the merged web, the entry
        // definition both arms overwrite (not φ-related, so ordinary
        // copy coalescing — a documented non-goal — would be what
        // merges it), and rcx.
        assert_eq!(f.names.len(), 5);
        assert_eq!(out.var_count, 3);
    }

    #[test]
    fn straight_line_code_needs_no_copies() {
        let f = build(&func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), c(1, Width::W64)),
                    assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                    assign(
                        ra(0, Width::W64),
                        bin(BinOp::Add, read(ra(0, Width::W64)), c(2, Width::W64)),
                    ),
                ],
                vec![],
            )],
        ));
        let (out, stats) = verify(&f);
        assert_eq!(phi_count(&f), 0);
        assert_eq!(total_copies(&out), 0, "the exit criterion for a line");
        assert_eq!(stats, OutStats::default());
        // With no φ there is nothing to coalesce: one variable per name,
        // which is the documented non-goal (ordinary copy coalescing).
        assert_eq!(out.var_count as usize, f.names.len());
    }

    // -- 4 and 5: interference, and value equality beating it ---------------

    /// The entry defines `rax` and keeps a copy of it in `rbx`; the arm
    /// redefines `rax` and then reads the copy, so the entry version is
    /// still live where the arm's version is defined. Whether the two
    /// can share a variable is decided purely by whether they hold the
    /// same value.
    fn split_family(arm_def: Expr) -> irlift::LiftedFunction {
        func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(3, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1010, 0x1030],
                ),
                block(
                    0x1010,
                    vec![
                        assign(ra(0, Width::W64), arm_def),
                        assign(ra(1, Width::W64), read(ra(3, Width::W64))),
                    ],
                    vec![0x1020, 0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(0, Width::W64), c(7, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(2, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        )
    }

    #[test]
    fn an_interfering_argument_forces_a_split_and_an_edge_copy() {
        let f = pipeline(&split_family(c(6, Width::W64)));
        let (out, stats) = verify(&f);
        let phi = phi_dst(&f, 0x1030, 0);
        let first = name_of(&f, 0, 0);
        let arm = name_of(&f, 0, 1);

        assert_eq!(
            out.var_of[first as usize], out.var_of[phi as usize],
            "the at-entry version is dead at the join and merges"
        );
        assert_ne!(
            out.var_of[arm as usize], out.var_of[phi as usize],
            "the arm's version interferes with the still-live entry one"
        );
        assert_eq!(
            out.edge_copies.get(&(0x1010, 0x1030)).map(Vec::as_slice),
            Some(
                [Copy {
                    dst: CopySlot::Var(out.var_of[phi as usize]),
                    src: CopySlot::Var(out.var_of[arm as usize]),
                }]
                .as_slice()
            )
        );
        assert_eq!(stats.copies, 1);
    }

    #[test]
    fn a_copy_chain_lets_interfering_ranges_share_a_variable() {
        // The same loop, except the body re-assigns the *same value*
        // through a copy chain (`rax := rcx`, and `rcx` is a copy of the
        // φ). The live ranges still intersect — the φ's value is read
        // after the loop — but value equality says sharing is harmless,
        // so the whole web is one variable and the back edge is free.
        // Note the copy chain is followed here even though the
        // propagation pass gave up on it: its optimistic lattice will
        // not move a fact sideways, this walk is syntactic.
        let f = forwarded(&loop_family(read(ra(1, Width::W64))));
        let (out, stats) = verify(&f);
        let phi = phi_dst(&f, 0x1010, 0);
        let back = name_of(&f, 0, 2);
        assert_eq!(
            out.var_of[back as usize], out.var_of[phi as usize],
            "live ranges intersect, but the values are equal, so they coalesce"
        );
        assert_eq!(total_copies(&out), 0);
        assert_eq!(stats.coalesced, 2);
    }

    // -- 6: version-0 names --------------------------------------------------

    #[test]
    fn version_zero_parameters_keep_distinct_variables() {
        let f = build(&func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![Stmt::Branch {
                        kind: BranchKind::Jump,
                        cond: Some(read(Reg::flag(Flag::Zero))),
                        target: c(0x1020, Width::W64),
                    }],
                    vec![0x1010, 0x1020],
                ),
                block(
                    0x1010,
                    vec![assign(ra(0, Width::W64), c(1, Width::W64))],
                    vec![0x1020],
                ),
                block(
                    0x1020,
                    vec![
                        assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                        assign(ra(2, Width::W64), read(ra(3, Width::W64))),
                    ],
                    vec![],
                ),
            ],
        ));
        let (out, _) = verify(&f);
        let rax0 = name_of(&f, 0, 0);
        let rbx0 = name_of(&f, 3, 0);
        assert!(f.live_in.contains(&rax0) && f.live_in.contains(&rbx0));
        assert_ne!(
            out.var_of[rax0 as usize], out.var_of[rbx0 as usize],
            "two parameters never share storage: classes never mix cells"
        );
        // The at-entry value does merge with the φ that consumes it —
        // it is that variable's first value, i.e. the parameter.
        assert_eq!(
            out.var_of[rax0 as usize],
            out.var_of[phi_dst(&f, 0x1020, 0) as usize]
        );
    }

    // -- 7: entry φ, self loop, several φs at one join ----------------------

    #[test]
    fn a_self_looping_entry_resolves_its_function_entry_edge() {
        let f = build(&func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(
                            ra(0, Width::W64),
                            bin(BinOp::Add, read(ra(0, Width::W64)), c(1, Width::W64)),
                        ),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1000, Width::W64),
                        },
                    ],
                    vec![0x1000, 0x1010],
                ),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        ));
        let phi = phi_dst(&f, 0x1000, 0);
        assert!(
            f.blocks[&0x1000].phis[0].args.iter().any(|&(k, _)| k.is_none()),
            "the entry φ carries the virtual function-entry edge"
        );
        let (out, _) = verify(&f);
        // Nothing interferes here, so the whole web is one variable and
        // both the self edge and the entry edge come out empty.
        assert_eq!(out.var_of[name_of(&f, 0, 0) as usize], out.var_of[phi as usize]);
        assert_eq!(total_copies(&out), 0);

        // The entry edge is still validated: a copy list that disturbs a
        // variable there is rejected, `None`-keyed.
        let mut broken = out.clone();
        broken.entry_copies = vec![Copy {
            dst: CopySlot::Var(0),
            src: CopySlot::Var(out.var_count - 1),
        }];
        assert_eq!(check(&f, &broken), Err(OutFault::BadSequence { edge: None }));
    }

    #[test]
    fn several_phis_at_one_join_are_resolved_in_parallel() {
        // Two cells merged at one join, each with its entry version
        // still live there (through a propagated copy), so the shared
        // edge carries one copy per cell — a parallel copy set.
        let f = pipeline(&func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(ra(6, Width::W64), read(ra(0, Width::W64))),
                        assign(ra(7, Width::W64), read(ra(3, Width::W64))),
                    ],
                    vec![0x1010, 0x1020],
                ),
                block(
                    0x1010,
                    vec![
                        assign(ra(0, Width::W64), c(3, Width::W64)),
                        assign(ra(3, Width::W64), c(4, Width::W64)),
                    ],
                    vec![0x1020],
                ),
                block(
                    0x1020,
                    vec![
                        assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                        assign(ra(2, Width::W64), read(ra(3, Width::W64))),
                        assign(ra(8, Width::W64), read(ra(6, Width::W64))),
                        assign(ra(9, Width::W64), read(ra(7, Width::W64))),
                    ],
                    vec![],
                ),
            ],
        ));
        assert_eq!(phi_count(&f), 2);
        let (out, _) = verify(&f);
        let copies = out
            .edge_copies
            .get(&(0x1000, 0x1020))
            .expect("the entry-side edge pays for both φs");
        assert_eq!(copies.len(), 2);
        // Independent copies: no destination is another's source, so
        // the parallel set needs no ordering and no temporary.
        let dsts: BTreeSet<CopySlot> = copies.iter().map(|c| c.dst).collect();
        let srcs: BTreeSet<CopySlot> = copies.iter().map(|c| c.src).collect();
        assert_eq!(dsts.len(), 2);
        assert!(dsts.is_disjoint(&srcs));
    }

    // -- 8: the seeded sweep with the interpreter ---------------------------

    /// A deterministic stream of small random CFGs, mirroring the
    /// `irssaopt` sweeps' generator with call effects added to the mix so
    /// the `assumed` provenance and the interpreter's havoc are
    /// exercised.
    fn random_functions(count: usize, seed: u64) -> Vec<irlift::LiftedFunction> {
        let mut s = seed;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let nblocks = 2 + next(&mut s) % 6;
            let vas: Vec<u64> = (0..nblocks).map(|i| 0x1000 + 0x10 * i).collect();
            let mut blocks = Vec::new();
            for &va in &vas {
                let nstmts = (next(&mut s) % 6) as usize;
                let mut list = Vec::new();
                for _ in 0..nstmts {
                    let r = (next(&mut s) % 4) as u16;
                    let r2 = (next(&mut s) % 4) as u16;
                    let k = next(&mut s) % 100;
                    list.push(match next(&mut s) % 10 {
                        0 => assign(ra(r, Width::W64), c(k, Width::W64)),
                        1 => assign(ra(r, Width::W64), read(ra(r2, Width::W64))),
                        2 => assign(
                            Reg::flag(Flag::Zero),
                            bin(
                                BinOp::Eq,
                                read(ra(r, Width::W64)),
                                read(ra(r2, Width::W64)),
                            ),
                        ),
                        3 => assign(ra(r, Width::W32), c(k, Width::W32)),
                        4 => Stmt::Store {
                            addr: read(ra(r, Width::W64)),
                            value: read(ra(r2, Width::W64)),
                        },
                        5 => assign(
                            ra(r, Width::W64),
                            bin(BinOp::Add, read(ra(r2, Width::W64)), c(k, Width::W64)),
                        ),
                        6 => assign(
                            ra(r, Width::W64),
                            Expr::unary(
                                UnOp::ZeroExtend(Width::W64),
                                read(ra(r2, Width::W32)),
                            ),
                        ),
                        7 => assign(
                            ra(r, Width::W64),
                            Expr::load(read(ra(r2, Width::W64)), Width::W64),
                        ),
                        8 => Stmt::Intrinsic {
                            name: callfx::EFFECT_NAME,
                            writes: vec![ra(r, Width::W64)],
                            reads: vec![read(ra(r2, Width::W64))],
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
                blocks.push(block(va, list, succ));
            }
            out.push(func(0x1000, blocks));
        }
        out
    }

    #[test]
    fn sweep_random_small_cfgs_agree_with_the_interpreter() {
        let mut phis = 0usize;
        let mut names = 0usize;
        let mut vars = 0usize;
        let mut copies = 0usize;
        for f in random_functions(400, 0x5EED_1A5E_0DDB_5BAD) {
            let raw = build(&f);
            // Both the faithful construction and the optimized pipeline
            // this module is meant to consume.
            let (opt, _) = irssaopt::optimize(&raw);
            let (fwd, _) = irssaopt::forward(&opt);
            let (swept, _) = irssaopt::eliminate_dead(&fwd, &[]);
            for input in [&raw, &swept] {
                let (out, stats) = verify(input);
                phis += stats.phis_resolved;
                names += input.names.len();
                vars += out.var_count as usize;
                copies += stats.copies;
            }
        }
        assert!(phis > 0 && copies > 0, "the corpus must exercise both");
        assert!(vars < names, "coalescing must actually reduce the count");
    }

    // -- 9: posture, determinism, and the checker itself --------------------

    #[test]
    fn a_malformed_function_gets_the_identity_map() {
        let mut broken = build(&lost_copy());
        assert!(!broken.live_in.is_empty());
        broken.live_in.clear(); // no longer the version-0 set
        assert!(irssa::check(&broken).is_err());
        let (out, stats) = out_of_ssa(&broken);
        assert_eq!(out.var_of, (0..broken.names.len() as u32).collect::<Vec<_>>());
        assert_eq!(out.var_count as usize, broken.names.len());
        assert!(out.edge_copies.is_empty() && out.entry_copies.is_empty());
        assert!(out.assumed.is_empty() && out.partial.is_empty());
        assert_eq!(stats, OutStats::default());
        assert_eq!(check(&broken, &out), Ok(()), "the posture is the contract");

        // Anything else for refused input is a fault.
        let mut wrong = out.clone();
        wrong.var_of[0] = 1;
        assert_eq!(check(&broken, &wrong), Err(OutFault::NotIdentity));
    }

    #[test]
    fn an_empty_function_translates_to_nothing() {
        let empty = SsaFunction {
            entry: 0x1000,
            name: None,
            arch: Arch::X86_64,
            blocks: BTreeMap::new(),
            skipped: Vec::new(),
            names: Vec::new(),
            live_in: Vec::new(),
            partial: Vec::new(),
        };
        let (out, stats) = translate(&empty);
        assert_eq!(out.var_count, 0);
        assert!(out.var_of.is_empty());
        assert_eq!(stats, OutStats::default());
    }

    #[test]
    fn check_rejects_every_broken_rendition() {
        let f = pipeline(&lost_copy());
        let (good, _) = out_of_ssa(&f);
        assert_eq!(check(&f, &good), Ok(()));

        // A map of the wrong length.
        let mut short = good.clone();
        short.var_of.pop();
        assert_eq!(check(&f, &short), Err(OutFault::Vars));

        // An id outside the count.
        let mut wild = good.clone();
        wild.var_of[0] = good.var_count;
        assert_eq!(check(&f, &wild), Err(OutFault::Vars));

        // A hole in the numbering.
        let mut sparse = good.clone();
        sparse.var_count += 1;
        assert_eq!(check(&f, &sparse), Err(OutFault::Vars));

        // Ids not in ascending first-name order.
        let mut shuffled = good.clone();
        for v in shuffled.var_of.iter_mut() {
            *v = good.var_count - 1 - *v;
        }
        assert_eq!(check(&f, &shuffled), Err(OutFault::Vars));

        // A dropped copy leaves a φ unresolved.
        let mut dropped = good.clone();
        dropped.edge_copies.clear();
        assert_eq!(
            check(&f, &dropped),
            Err(OutFault::BadSequence {
                edge: Some((0x1020, 0x1010))
            })
        );

        // A copy on something that is not an edge.
        let mut bogus = good.clone();
        bogus.edge_copies.insert((0x1030, 0x1000), Vec::new());
        assert_eq!(
            check(&f, &bogus),
            Err(OutFault::UnknownEdge {
                edge: (0x1030, 0x1000)
            })
        );

        // Provenance that does not match the recomputation.
        let mut lying = good.clone();
        lying.assumed.insert(0);
        assert_eq!(check(&f, &lying), Err(OutFault::Provenance));
        let mut lying = good.clone();
        lying.partial.insert(0);
        assert_eq!(check(&f, &lying), Err(OutFault::Provenance));
    }

    #[test]
    fn the_stats_are_exact_on_a_known_fixture() {
        let f = pipeline(&lost_copy());
        let (_, stats) = translate(&f);
        assert_eq!(
            stats,
            OutStats {
                phis_resolved: 1,
                copies: 1,
                // the φ merges with the entry version; the back-edge
                // argument interferes and stays out.
                coalesced: 1,
            }
        );
    }

    #[test]
    fn the_provenance_markers_reach_the_variables() {
        // A call's clobber is ABI-assumed; a read wider than its
        // definition is partial. Both must land on the variable.
        let f = build(&func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W32), c(7, Width::W32)),
                    assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                    Stmt::Intrinsic {
                        name: callfx::EFFECT_NAME,
                        writes: vec![ra(2, Width::W64)],
                        reads: vec![read(ra(1, Width::W64))],
                    },
                ],
                vec![],
            )],
        ));
        let (out, _) = verify(&f);
        assert_eq!(f.partial, vec![(0x1000, 1)]);
        assert_eq!(
            out.partial,
            BTreeSet::from([out.var_of[name_of(&f, 0, 1) as usize]])
        );
        assert_eq!(
            out.assumed,
            BTreeSet::from([out.var_of[name_of(&f, 2, 1) as usize]])
        );
    }

    // -- 10: the real-binary sweep -------------------------------------------

    /// The x86-64 slice of a Mach-O universal binary, or the file itself
    /// when it is already thin. `None` when there is no such slice.
    fn x86_slice(data: &[u8]) -> Option<Vec<u8>> {
        match crate::macho::FatFile::parse(data) {
            Ok(fat) => fat
                .arch_by_cputype(crate::macho::CpuType::X86_64)?
                .slice(data)
                .ok()
                .map(<[u8]>::to_vec),
            Err(_) => Some(data.to_vec()), // already a thin image
        }
    }

    /// `/bin/ls` (x86-64 slice) through the whole pipeline — construct,
    /// optimize, forward, sweep — and out of SSA on every function. A
    /// no-op where the file is not a Mach-O x86-64 image (the synthetic
    /// coverage above then stands alone), like the other real-binary
    /// smoke tests in this crate.
    #[test]
    fn a_real_binary_translates_out_of_ssa() {
        let Ok(data) = std::fs::read("/bin/ls") else {
            return;
        };
        let Some(bytes) = x86_slice(&data) else {
            return;
        };
        let Ok(image) = crate::load(&bytes) else {
            return;
        };
        if image.arch() != crate::model::Arch::X86_64 {
            return;
        }
        let Ok(program) = crate::cfg::recover(image.as_ref()) else {
            return;
        };
        let abi = callfx::abi_for(image.arch());
        let live_out = callfx::function_live_out(image.arch()).unwrap_or_default();

        let (mut funcs, mut names, mut phis, mut vars, mut copies) = (0, 0, 0, 0, 0);
        let (mut coalesced, mut args, mut faults, mut zero_copy) = (0, 0, 0, 0);
        let (mut simple, mut simple_copies) = (0, 0);
        for func in program.functions.values() {
            let Some(lifted) = crate::irlift::lift_function(image.as_ref(), func) else {
                continue;
            };
            let lifted = match &abi {
                Some(abi) => callfx::apply(&lifted, abi),
                None => lifted,
            };
            let Ok(ssa) = irssa::construct(&lifted) else {
                continue;
            };
            let (opt, _) = irssaopt::optimize(&ssa);
            let (fwd, _) = irssaopt::forward(&opt);
            let (f, _) = irssaopt::eliminate_dead(&fwd, &live_out);

            let (out, stats) = out_of_ssa(&f);
            if check(&f, &out).is_err() {
                faults += 1;
            }
            let (again, stats2) = out_of_ssa(&f);
            assert_eq!(out, again, "byte-determinism on {:#x}", f.entry);
            assert_eq!(stats, stats2);

            funcs += 1;
            names += f.names.len();
            phis += stats.phis_resolved;
            vars += out.var_count as usize;
            copies += stats.copies;
            coalesced += stats.coalesced;
            args += f
                .blocks
                .values()
                .flat_map(|b| b.phis.iter())
                .map(|p| p.args.len())
                .sum::<usize>();
            if stats.copies == 0 {
                zero_copy += 1;
            }
            // Straight-line code and a single diamond: no back edge, at
            // most one join — the shape the exit criterion names.
            let looping = f.blocks.iter().any(|(&va, b)| {
                b.successors
                    .iter()
                    .any(|s| *s <= va && f.blocks.contains_key(s))
            });
            let mut joins = 0;
            for &va in f.blocks.keys() {
                let preds = f
                    .blocks
                    .values()
                    .filter(|b| b.successors.contains(&va))
                    .count();
                if preds > 1 {
                    joins += 1;
                }
            }
            if !looping && joins <= 1 {
                simple += 1;
                simple_copies += stats.copies;
            }
        }
        println!(
            "/bin/ls: {funcs} functions, {names} names, {phis} φs over {args} arguments \
             -> {vars} variables ({coalesced} coalesced), {copies} residual copies, \
             {zero_copy} functions copy-free, {simple} straight-line/one-diamond \
             functions with {simple_copies} copies, {faults} check failures"
        );
        assert!(funcs > 0, "the sweep must see real functions");
        assert_eq!(faults, 0, "every rendition must check");
        assert_eq!(
            simple_copies, 0,
            "straight-line and diamond code leaves no residual copy"
        );
        assert!(vars < names, "coalescing must reduce the variable count");
    }
}
