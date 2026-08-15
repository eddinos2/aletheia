//! Phoenix-style control-flow structuring: an SSA CFG to a structure tree.
//!
//! [`crate::irssa`] and [`crate::irssaopt`] leave a function as a graph of
//! basic blocks whose dataflow is clean but whose *control* flow is still
//! a mesh of edges. This module recovers the shape a reader thinks in —
//! sequences, `if`/`else`, loops, `switch` — by iterative structural
//! analysis: Schwartz, Lee, Woo and Brumley, USENIX Security 2013
//! (Phoenix), on the Cifuentes 1994 / Sharir 1980 schema lineage. The
//! result is a [`Node`] tree covering every reachable block exactly once,
//! plus [`StructStats`].
//!
//! # Conditions are references, never expressions
//!
//! An `If`/`Loop` condition is a [`Cond`] — the VA of the block whose
//! final conditional [`Stmt::Branch`] decides it, plus a polarity bit. The
//! tree never copies or rewrites an expression, so the same tree is valid
//! over the faithful SSA and over any optimized form of it; the renderer
//! fetches the expression at print time. **Polarity is defined as:** for a
//! block ending in `Branch { cond: Some(g), target: Const(t) }` with
//! fall-through `f`, `Cond { negated: false }` denotes `g` itself — true
//! means control goes to `t` — and `negated: true` denotes `!g`, true
//! means control goes to `f`. An [`Node::If`] runs `then_body` when its
//! (possibly negated) condition is true; a [`Node::Loop`] condition is the
//! *continuation* condition — true means stay in the loop.
//!
//! # Gotos are honest output, not failure
//!
//! Per SAILR's evidence (Basque, Wu, Kirat, Zhu, Brumley et al., USENIX
//! Security 2024), goto-free-by-construction structuring measures *worse*
//! against original source, because it buys the goto-freedom with
//! unbounded node duplication and synthesized conditions. So the collapse
//! never synthesizes a condition and never duplicates a block: when no
//! schema matches anywhere, it virtualizes exactly one edge into an
//! explicit [`Node::Goto`] and retries. The choice is total — the
//! removable edge with the lowest `(source region head VA, target VA)`,
//! which is the source block's own VA for the single-block regions that
//! have not collapsed yet. Each virtualization removes one edge from the
//! region graph, so termination is *structural*; the round budget below
//! is defense in depth, and firing it degrades to gotos rather than
//! refusing.
//!
//! # De-optimization: re-splitting merged tails (SAILR)
//!
//! The same SAILR evidence says most *spurious* gotos are the residue of
//! specific compiler transforms — jump threading, cross-jumping,
//! tail-merging — and the highest-value inversions re-split shared code
//! that several predecessors jump into. After an uncapped collapse,
//! [`structure`] therefore rewrites in-function [`Node::Goto`]s whose
//! target is a **copy-safe tail** into a duplicate of that tail. Two
//! inversions share the one classifier (`splittable_tail`), and a third
//! (`threadable_head`) inverts jump threading itself:
//!
//! - **Epilogue tails** (inversion one): a plain leaf whose every
//!   remaining edge leaves the function — a shared `return`, or a tail
//!   jump whose external goto the duplicate carries along.
//! - **Shared case tails** (inversion two): a plain leaf — or a chain of
//!   up to [`MAX_TAIL_CHAIN`] of them, each the next one's only spelled
//!   step — whose one remaining in-function edge converges on a single
//!   target, provably copy-free by the ground truth: the edge is absent
//!   from [`irout::out_of_ssa`]'s `edge_copies`, the copy set the
//!   renderer actually executes. (The first cut asked the SSA names —
//!   every φ argument its own definition — but coalescence folds
//!   different names for one value into one variable and emits nothing,
//!   and the φ-web narrowing replaced the approximation with irout's
//!   answer.) The duplicate owes that edge a
//!   realization, and a
//!   site is rewritten only where it can spell one honestly: a
//!   [`Node::Continue`] where the target is the enclosing loop's header,
//!   a [`Node::Break`] where it is the loop's own follow (never from
//!   inside a `switch` case, where C's `break` would leave the switch
//!   instead), or plain fall-through where the target is exactly the
//!   next textual consumer — the same conversions `tighten` applies to
//!   a loop's own gotos, decided per site so the split is refused
//!   rather than spending a duplicate to keep a goto.
//! - **Threaded conditions** (inversion three): the residue the first
//!   two inversions refuse by design — a goto whose target *carries a
//!   condition*, the signature of compiler jump threading routing
//!   several predecessors through one shared small conditional block.
//!   The inversion duplicates the deciding block itself into the
//!   goto-ing site: a small block — pure register/flag assignments
//!   only, at most [`MAX_THREAD_STMTS`] of them before the branch —
//!   ending in a conditional branch, materialized as the block's plain
//!   leaf followed by the real `If { cond: Cond { block: <the copy>,
//!   negated } }`. Conditions are (block, polarity) references, so the
//!   copy stays *referenceable* by the same identity scheme the first
//!   two inversions use — the leaf stores the VA — and [`check`]'s
//!   condition-honesty rules hold on the copy exactly as on the
//!   original (a duplicate that drops its branch is an
//!   [`StructFault::Undecided`] fault; one that funnels both
//!   polarities one way is a [`StructFault::Polarity`] fault). Both
//!   out-edges must be
//!   spelled honestly at the site: the arm as a `Continue`, a `Break`,
//!   the travelling goto of an external target, or — the composed
//!   round — an inline duplicate of the *fresh linear tail* the thread
//!   exposes, spelled by the case-tail classifier itself (the target's
//!   own copy-safe chain, currently the target of no goto or rewritten
//!   whole this round, so no duplicated block is ever also a goto
//!   target); the fall-through side as a `Continue`, plain
//!   fall-through, a `Break`, or an external goto. Either polarity may
//!   carry the arm; the cheapest spellable option wins, ties to the
//!   un-negated form.
//!
//! Threading runs where the cheaper inversions leave gotos — in the
//! same round loop, after them — so a thread that exposes a linear
//! tail composes with the chain inversion, and a round's chain
//! rewrites can unlock the next round's thread. The loop cannot
//! oscillate: a rewrite never creates an in-function goto, so the goto
//! count strictly falls on every productive round, and the shared
//! budget bounds total duplication.
//!
//! This is controlled duplication of provably byte-identical statement
//! lists, never invention:
//!
//! - **Byte-equal by construction, held by `check`.** A duplicated leaf
//!   is `Node::Block(va)` — tree shape, not block storage — so every
//!   occurrence renders the one statement list the [`SsaFunction`]
//!   holds. [`check`] pins the shape: an extra occurrence of a block is
//!   sanctioned only as the plain leaf of a copy-safe tail.
//! - **Counted and capped.** Each duplicate leaf is counted in
//!   [`StructStats::duplications`] against [`MAX_TAIL_SPLITS`] (SAILR's
//!   lesson that duplication must be bounded); a re-split that does not
//!   fit sets [`StructStats::dup_capped`] and its edges keep their
//!   gotos — degrade, never refuse.
//! - **Only splits that remove a goto.** The pass rewrites exactly the
//!   edges the collapse *did* virtualize, one goto bought back per
//!   rewritten site, so the goto count is monotonically non-increasing
//!   and a split that saves nothing is never made.
//! - **All-or-nothing per target.** Either every goto to a tail is
//!   re-split or none is, so a duplicated tail is never also a goto
//!   target and downstream renderers never face a twice-labeled block.
//!   A chain interior must already be label-free; rounds run to a
//!   fixpoint, so a target rewritten in one round can free a chain
//!   through it in the next.
//!
//! What still does not split, deliberately: an opaque tail, a tail
//! whose edges converge on more than one target, and any edge a
//! duplicate would realize that could carry φ copies — a duplicate
//! realizes its outgoing edges at every occurrence, and
//! [`crate::irout`]'s copies for an edge have exactly one textual
//! placement, so an edge that could carry them is never realized
//! twice; for a threaded condition that per-edge refusal applies to
//! *both* out-edges. A conditional block that is big or effectful —
//! anything but register assignments before its branch — never
//! threads: the SAILR spirit is a small threaded condition, not a
//! body. Nor does a site that cannot spell both edges without a new
//! in-function goto: a split that saves no goto is never made. The
//! pass does not run on a capped or
//! refused structuring — the degrade is the degrade — and with zero
//! duplications the output is bit-for-bit what the collapse alone
//! produced.
//!
//! # The schema catalog, tried in this order at each region head
//!
//! Cyclic before acyclic, and *graph-wide*: every loop header in the
//! region graph is offered schemas 3-4 before any head is offered 1, 2
//! or 5. Without that split an if-then somewhere in a body absorbs the
//! loop's own exit block, and the loop degrades to `while (true)` with
//! its test buried inside.
//!
//! 1. **Sequence** — a single-successor region whose successor has no
//!    other predecessor.
//! 2. **If-then-else / if-then** — a two-way head whose arms have no
//!    other predecessor and converge on one follow region (or terminate).
//! 3. **Self-loop** — a region whose remaining edge is to itself.
//! 4. **Natural while / do-while** — a back edge to a region-graph
//!    dominator, single-entry body, with the **follow node** by this
//!    deterministic rule: the immediate post-dominator of the header when
//!    one exists outside the body, else the most frequent exit-edge
//!    target, ties broken by lowest address. Every exit edge must reach
//!    that follow or the match is refused (a later round, or a goto,
//!    resolves it). In-body edges to the follow become [`Node::Break`],
//!    in-body edges to the header become [`Node::Continue`]. The kind is
//!    `While` when the header's conditional decides the exit, `DoWhile`
//!    when a latch's does, and `SelfLoop` when the body is one block; a
//!    loop with no conditional exit carries `cond: None` and reads
//!    `while (true)`, its exits being breaks.
//! 5. **Switch** — only at a block whose final statement is an indirect
//!    `Branch { kind: Jump, cond: None }` at a jump site the caller's
//!    `tables` proved (a table keyed inside `[block.start, block.end)`),
//!    whose recorded successors are non-empty and all proven targets.
//!
//! **Honest limit:** [`crate::cfg`] does not yet fold
//! [`crate::jumptable::successor_map`] into block successors, so on
//! today's real pipeline an indirect-jump block has no in-function case
//! edges and renders `Opaque { reason: IndirectJump }`. `Switch` is
//! exercised by synthetic tests until the CFG-folding rider lands; the
//! structurer takes the map as a parameter so nothing here changes then.
//!
//! # What is held, not structured
//!
//! A block whose lift was truncated, one ending in an indirect jump no
//! table proved, and one whose recorded successors cannot be decided
//! between (two or more edges with no conditional branch to choose them —
//! hand-built or malformed input; the lifters never emit it) become
//! [`Node::Opaque`] with the reason spelled out. An `Opaque` may sit
//! inside a `Seq`/`If`/`Loop` body, but nothing is ever absorbed *into*
//! it and it is never given a fall-through it does not have: an
//! undecidable block's edges are declared unrealized rather than invented,
//! and [`check`] holds the pass to exactly that. Out-of-function
//! successors (tail jumps) are realized as external `Goto`s and are never
//! counted as structuring edges. `SsaFunction::skipped` blocks are listed
//! by [`render`], never structured.
//!
//! # Contract
//!
//! [`structure`] is pure and total: it never mutates its input, never
//! duplicates a block beyond the sanctioned tail re-split above, never
//! synthesizes a condition, never reorders statements, and never panics
//! — on any input, including hand-broken ones. All containers are
//! `BTree*` and every tie-break is a total order, so equal inputs
//! produce byte-equal trees. Input that fails
//! [`irssa::check`] is refused rather than interpreted, the established
//! posture: the answer is the *degenerate* tree — every reachable block
//! as its own leaf with all of its edges as explicit `Goto`s, which
//! trivially satisfies [`check`] — with zeroed [`StructStats`].
//! [`check`] is the companion the tests trust over the pass: it
//! recomputes the partition, the realized-edge set, condition honesty and
//! opacity from scratch. [`render`] is deterministic and `\n`-terminated,
//! in the style of [`irssa::render`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::ir::{BranchKind, Expr, Stmt};
use crate::irout;
use crate::irssa::{self, SsaBlock, SsaFunction};

// ---------------------------------------------------------------------------
// The tree
// ---------------------------------------------------------------------------

/// A condition: the block whose final conditional branch decides it, plus
/// polarity. See the module docs for the exact meaning of `negated`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cond {
    /// VA of the deciding block.
    pub block: u64,
    /// `true` when the condition is the branch guard's negation.
    pub negated: bool,
}

/// Which loop shape a [`Node::Loop`] recovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopKind {
    /// One block branching to itself.
    SelfLoop,
    /// The exit test is at the header — the body's first block.
    While,
    /// The exit test is at a latch — a later block of the body.
    DoWhile,
}

/// Why a block is [`Node::Opaque`] instead of structured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpaqueReason {
    /// The lift stopped early in this block ([`SsaBlock::truncated`]).
    Truncated,
    /// The block ends in an indirect jump no jump table proved.
    IndirectJump,
    /// Two or more recorded successors with no conditional branch to
    /// decide between them: the edges are declared unrealized rather
    /// than invented.
    Unstructurable,
}

/// One node of a recovered structure tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    /// One basic block's straight-line statements (by VA). Its
    /// terminator is implied by the parent construct; a leaf never
    /// invents flow.
    Block(u64),
    /// Children executed in order.
    Seq(Vec<Node>),
    /// A two-way conditional. The deciding block is `cond.block`, which
    /// is the preceding sibling's last leaf, not a child of the `If`.
    If {
        cond: Cond,
        then_body: Box<Node>,
        else_body: Option<Box<Node>>,
    },
    /// A loop. `body` holds every block of the loop, the header first;
    /// `cond` — when present — is the *continuation* condition, tested
    /// by `cond.block`'s terminator. Falling off the end of `body` is
    /// the back edge to the header.
    Loop {
        kind: LoopKind,
        cond: Option<Cond>,
        body: Box<Node>,
    },
    /// A proven jump-table dispatch: the scrutinee block (which this node
    /// *is* the occurrence of) and one body per case target, in ascending
    /// target order.
    Switch { block: u64, cases: Vec<(u64, Node)> },
    /// Leave the enclosing loop for its follow.
    Break,
    /// Go back to the enclosing loop's header.
    Continue,
    /// An edge realized as an explicit jump — in-function or, for a tail
    /// jump, to a VA outside it.
    Goto(u64),
    /// A block held as-is: truncated, unproven-indirect, or with
    /// undecidable successors. Never absorbed, never given invented
    /// flow.
    Opaque { block: u64, reason: OpaqueReason },
}

/// What [`structure`] had to do to get there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StructStats {
    /// Iterations of the collapse loop, recursion included.
    pub rounds: usize,
    /// In-function edges virtualized into explicit [`Node::Goto`]s,
    /// minus the ones the tail re-split bought back. Out-of-function
    /// edges are not counted: they are not structuring edges.
    pub gotos: usize,
    /// The defensive round or nesting cap fired. The tree is still
    /// valid — the remainder degraded to gotos.
    pub capped: bool,
    /// Duplicate leaves the re-split passes spent, at most
    /// [`MAX_TAIL_SPLITS`]: every extra occurrence of a block in the
    /// tree, threaded deciding blocks and their inlined arm tails
    /// included.
    pub duplications: usize,
    /// Of the goto sites bought back, how many were threaded — rewritten
    /// into a duplicate of a condition-carrying block (inversion three).
    /// Each threaded site's leaves are also in `duplications`.
    pub threaded: usize,
    /// The duplication cap refused at least one re-split; the affected
    /// edges keep their gotos.
    pub dup_capped: bool,
}

/// Why a tree does not describe its function. [`check`] returns the first
/// fault, with enough context to locate it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructFault {
    /// A reachable block no node covers.
    Uncovered { block: u64 },
    /// A block covered more than once outside the sanctioned tail
    /// re-split: an extra occurrence that is not the plain leaf of a
    /// copy-safe tail, or more extras than [`MAX_TAIL_SPLITS`] allows.
    Duplicated { block: u64 },
    /// A node naming a block the function does not hold.
    Foreign { block: u64 },
    /// The tree realizes an edge the CFG does not have.
    InventedEdge { from: u64, to: u64 },
    /// A CFG edge no part of the tree realizes.
    DroppedEdge { from: u64, to: u64 },
    /// A condition naming a block that does not end in a conditional
    /// branch with a constant target.
    NotConditional { block: u64 },
    /// An `If` that is not preceded by exactly its deciding block.
    CondMisplaced { block: u64 },
    /// An arm entered by the wrong side of its condition, an `If` with
    /// two empty arms, or an else-less `If` whose untaken side is then
    /// realized to the wrong target or never realized — both polarities
    /// funneled one way would belie the branch.
    Polarity { block: u64 },
    /// A block that must be `Opaque` and is not, or one that is `Opaque`
    /// without cause, or with the wrong reason.
    Opacity { block: u64 },
    /// A `Switch` whose block is not a proven dispatch, or whose cases
    /// do not match its successors.
    BadSwitch { block: u64 },
    /// A loop whose kind, condition or body do not fit each other.
    BadLoop { block: u64 },
    /// A two-way block whose edge is realized with nothing deciding it —
    /// no `If` naming the block, no enclosing loop condition — so the
    /// occurrence would drop its branch. The rule holds duplicates of
    /// deciding blocks to the same condition honesty as originals.
    Undecided { block: u64 },
    /// A `Break`/`Continue` with no enclosing loop, or a `Goto`/`Break`
    /// reached with no live predecessor.
    LooseJump,
    /// The tree nests deeper than [`MAX_TREE_DEPTH`].
    TooDeep,
}

impl std::fmt::Display for StructFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StructFault::Uncovered { block } => write!(f, "block {block:#x} is not covered"),
            StructFault::Duplicated { block } => write!(f, "block {block:#x} is covered twice"),
            StructFault::Foreign { block } => write!(f, "block {block:#x} is not in the function"),
            StructFault::InventedEdge { from, to } => {
                write!(f, "invented edge {from:#x} -> {to:#x}")
            }
            StructFault::DroppedEdge { from, to } => {
                write!(f, "dropped edge {from:#x} -> {to:#x}")
            }
            StructFault::NotConditional { block } => {
                write!(f, "block {block:#x} has no conditional branch")
            }
            StructFault::CondMisplaced { block } => {
                write!(f, "the if at block {block:#x} is misplaced")
            }
            StructFault::Polarity { block } => write!(f, "wrong polarity at block {block:#x}"),
            StructFault::Opacity { block } => write!(f, "wrong opacity at block {block:#x}"),
            StructFault::Undecided { block } => {
                write!(f, "block {block:#x} realizes an edge with its branch undecided")
            }
            StructFault::BadSwitch { block } => write!(f, "unproven switch at block {block:#x}"),
            StructFault::BadLoop { block } => write!(f, "malformed loop at block {block:#x}"),
            StructFault::LooseJump => write!(f, "a jump with no target in scope"),
            StructFault::TooDeep => write!(f, "the tree nests too deeply"),
        }
    }
}

/// Deepest tree [`check`] and [`render`] walk. Structuring never nests
/// deeper than its own loop-nesting cap plus a small constant per level,
/// so this only ever bounds a hand-built tree.
pub const MAX_TREE_DEPTH: usize = 512;

/// Deepest loop nesting [`structure`] recurses into before degrading the
/// remainder to gotos. Real code nests a handful deep.
const MAX_STRUCT_DEPTH: usize = 64;

/// Most duplicate tail leaves the re-split pass spends per function —
/// SAILR's lesson that de-optimizing duplication must be bounded. A
/// chain duplicate spends one per leaf it spells. A re-split that does
/// not fit is skipped whole and its edges stay gotos. Raised from 16
/// when the shared-case-tail inversion landed, on the measured bash
/// x86-64 distribution: leaf-costed chains capped 18 functions at 16,
/// two at 32, and doubling again to 64 bought back only a fifth more
/// gotos for twice the duplication allowance.
pub const MAX_TAIL_SPLITS: usize = 32;

/// Longest chain of copy-safe leaves one duplicate may spell — the
/// case-body depth the shared-tail inversion inlines before it stops at
/// an open convergence edge. Real case bodies run this deep; anything
/// longer degrades to its gotos.
pub const MAX_TAIL_CHAIN: usize = 3;

/// Most statements a threadable conditional block may hold before its
/// branch — the SAILR spirit is a small threaded *condition*, not a
/// body. Derived from the bash x86-64 corpus through the optimized
/// pipeline: of the condition-carrying goto targets, 83% of the pure
/// ones carry at most two assignments before the branch and 97% at
/// most four; measured threads were 10 at a cap of 1, 12 at 2, 13 at
/// 4, and 8 added nothing the spellability and φ rules did not
/// already refuse — the knee is 4.
pub const MAX_THREAD_STMTS: usize = 4;

// ---------------------------------------------------------------------------
// The seam
// ---------------------------------------------------------------------------

/// A control-flow structuring strategy. One trait, one implementation
/// ([`Phoenix`]) — the seam exists so a SAILR-style strategy can be
/// compared later on the same regions (angr's pluggable-structurer
/// lesson), not as speculative abstraction.
pub trait Structurer {
    /// Structure `f`, using `tables` (jump site VA -> proven successors,
    /// as [`crate::jumptable::successor_map`] returns) to decide which
    /// indirect jumps may become a `Switch`.
    fn structure(&self, f: &SsaFunction, tables: &BTreeMap<u64, Vec<u64>>) -> (Node, StructStats);
}

/// The iterative region-collapse structurer described in the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Phoenix;

impl Structurer for Phoenix {
    fn structure(&self, f: &SsaFunction, tables: &BTreeMap<u64, Vec<u64>>) -> (Node, StructStats) {
        structure(f, tables)
    }
}

/// Structure `f` with the [`Phoenix`] strategy. See the module docs for
/// the contract; `tables` may be empty.
pub fn structure(f: &SsaFunction, tables: &BTreeMap<u64, Vec<u64>>) -> (Node, StructStats) {
    if irssa::check(f).is_err() {
        return (degenerate(f, tables), StructStats::default());
    }
    structure_budgeted(f, tables, default_budget(f))
}

/// The round budget [`structure`] grants: generous enough that only a
/// hand-built pathology can reach it.
fn default_budget(f: &SsaFunction) -> usize {
    let cfg = irssa::Cfg::analyze(f.entry, &raw_successors(f));
    let edges: usize = cfg.succs.values().map(Vec::len).sum();
    64 * (cfg.succs.len() + edges) + 256
}

/// [`structure`]'s body with the round budget spelled out, so a test can
/// force the cap and watch the pass degrade instead of refuse: the
/// collapse, then — on an uncapped result — the tail re-split.
fn structure_budgeted(
    f: &SsaFunction,
    tables: &BTreeMap<u64, Vec<u64>>,
    budget: usize,
) -> (Node, StructStats) {
    let (root, mut stats) = structure_raw(f, tables, budget);
    if stats.capped {
        return (root, stats); // the degrade is the degrade
    }
    let copies = copy_edges(f);
    let root = resplit_tails(f, tables, &copies, root, &mut stats, MAX_TAIL_SPLITS);
    (root, stats)
}

/// The collapse alone, without the tail re-split — the zero-duplication
/// baseline the re-split's scoped relaxation is tested against.
fn structure_raw(
    f: &SsaFunction,
    tables: &BTreeMap<u64, Vec<u64>>,
    budget: usize,
) -> (Node, StructStats) {
    let mut b = Builder {
        f,
        tables,
        budget,
        rounds: 0,
        gotos: 0,
        capped: false,
    };
    let regions = b.initial_regions();
    if regions.is_empty() {
        return (Node::Seq(Vec::new()), StructStats::default());
    }
    let root = b.collapse(regions, f.entry, 0);
    let stats = StructStats {
        rounds: b.rounds,
        gotos: b.gotos,
        capped: b.capped,
        duplications: 0,
        threaded: 0,
        dup_capped: false,
    };
    (root.node, stats)
}

/// The refusal answer, also the shape [`Builder::join`] degrades to:
/// every reachable block as its own leaf, every edge an explicit `Goto`.
fn degenerate(f: &SsaFunction, tables: &BTreeMap<u64, Vec<u64>>) -> Node {
    let mut b = Builder {
        f,
        tables,
        budget: 0,
        rounds: 0,
        gotos: 0,
        capped: false,
    };
    let regions = b.initial_regions();
    if regions.is_empty() {
        return Node::Seq(Vec::new());
    }
    b.join(regions, f.entry).node
}

// ---------------------------------------------------------------------------
// Block exits: the one classifier `structure`, `check` and `render` share
// ---------------------------------------------------------------------------

/// How control leaves a block (or, once regions collapse, a region).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Exits {
    /// Nothing left to realize.
    None,
    /// One successor, in or out of the function.
    One(u64),
    /// A conditional branch with a constant target: exactly two
    /// successors, `taken` being the branch's own target.
    Cond {
        block: u64,
        taken: u64,
        fallthrough: u64,
    },
    /// A proven jump-table dispatch: ascending in-function targets.
    Table { block: u64, targets: Vec<u64> },
    /// Two or more successors with nothing to decide between them. The
    /// edges are not realizable; the block is `Opaque`.
    Inexpressible,
}

impl Exits {
    /// The successor VAs this exit still has to realize.
    fn targets(&self) -> Vec<u64> {
        match self {
            Exits::None | Exits::Inexpressible => Vec::new(),
            Exits::One(t) => vec![*t],
            Exits::Cond {
                taken, fallthrough, ..
            } => vec![*taken, *fallthrough],
            Exits::Table { targets, .. } => targets.clone(),
        }
    }
}

/// A block's recorded successors, deduplicated in stored order.
fn successors(block: &SsaBlock) -> Vec<u64> {
    let mut seen = BTreeSet::new();
    block
        .successors
        .iter()
        .copied()
        .filter(|s| seen.insert(*s))
        .collect()
}

/// The block's final statement, if it has one.
fn terminator(block: &SsaBlock) -> Option<&Stmt> {
    block.stmts.last()
}

/// The `(taken, fall-through)` sides of a block that ends in a
/// conditional branch with a constant target and has exactly those two
/// successors.
fn cond_sides(block: &SsaBlock, all: &[u64]) -> Option<(u64, u64)> {
    let Some(Stmt::Branch {
        kind: BranchKind::Jump,
        cond: Some(_),
        target: Expr::Const { value, .. },
    }) = terminator(block)
    else {
        return None;
    };
    if all.len() != 2 {
        return None;
    }
    let taken = *value;
    let other = if all[0] == taken {
        all[1]
    } else if all[1] == taken {
        all[0]
    } else {
        return None;
    };
    Some((taken, other))
}

/// Whether the block ends in an indirect jump — the shape a jump table
/// has to prove before it can become a `Switch`.
fn is_indirect_jump(block: &SsaBlock) -> bool {
    matches!(
        terminator(block),
        Some(Stmt::Branch {
            kind: BranchKind::Jump,
            cond: None,
            target,
        }) if !matches!(target, Expr::Const { .. })
    )
}

/// The proven case targets of an indirect-jump block: the block's
/// recorded in-function successors, when a table keyed *inside*
/// `[start, end)` — the jump site is the block's last instruction, so at
/// most one table can be — proves every one of them. Ascending.
fn proven_table(
    f: &SsaFunction,
    tables: &BTreeMap<u64, Vec<u64>>,
    block: &SsaBlock,
    all: &[u64],
) -> Option<Vec<u64>> {
    if !is_indirect_jump(block) || all.is_empty() {
        return None;
    }
    let proven: &Vec<u64> = tables
        .range(block.start..block.end.max(block.start))
        .map(|(_, v)| v)
        .next()?;
    let mut targets: Vec<u64> = all.to_vec();
    targets.sort_unstable();
    if targets
        .iter()
        .all(|t| f.blocks.contains_key(t) && proven.contains(t))
    {
        Some(targets)
    } else {
        None
    }
}

/// Classify one block's exits. Shared by structuring and [`check`] so the
/// two can never drift.
fn exits(f: &SsaFunction, tables: &BTreeMap<u64, Vec<u64>>, va: u64) -> Exits {
    let Some(block) = f.blocks.get(&va) else {
        return Exits::None;
    };
    let all = successors(block);
    if all.is_empty() {
        return Exits::None;
    }
    if !block.truncated
        && let Some(targets) = proven_table(f, tables, block, &all)
    {
        return Exits::Table { block: va, targets };
    }
    if all.len() == 1 {
        return Exits::One(all[0]);
    }
    match cond_sides(block, &all) {
        Some((taken, fallthrough)) => Exits::Cond {
            block: va,
            taken,
            fallthrough,
        },
        None => Exits::Inexpressible,
    }
}

/// Why a block cannot be an ordinary [`Node::Block`], if it cannot.
fn opaque_reason(
    f: &SsaFunction,
    tables: &BTreeMap<u64, Vec<u64>>,
    va: u64,
) -> Option<OpaqueReason> {
    let block = f.blocks.get(&va)?;
    if block.truncated {
        return Some(OpaqueReason::Truncated);
    }
    let e = exits(f, tables, va);
    if is_indirect_jump(block) && !matches!(e, Exits::Table { .. }) {
        return Some(OpaqueReason::IndirectJump);
    }
    if matches!(e, Exits::Inexpressible) {
        return Some(OpaqueReason::Unstructurable);
    }
    None
}

/// Every block's recorded successors, for [`irssa::Cfg::analyze`].
fn raw_successors(f: &SsaFunction) -> BTreeMap<u64, Vec<u64>> {
    f.blocks
        .iter()
        .map(|(&va, b)| (va, b.successors.clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// Tree helpers
// ---------------------------------------------------------------------------

/// Append `tail` to `node`, flattening sequences so nesting stays flat.
fn seq_push(node: Node, tail: Node) -> Node {
    let mut v = match node {
        Node::Seq(v) => v,
        other => vec![other],
    };
    match tail {
        Node::Seq(t) => v.extend(t),
        other => v.push(other),
    }
    if v.len() == 1 {
        v.pop().unwrap_or(Node::Seq(Vec::new()))
    } else {
        Node::Seq(v)
    }
}

/// Split a trailing `Block(b)` leaf off a node — how a `Switch` takes
/// over its scrutinee's occurrence without duplicating it.
fn split_tail_block(node: Node) -> Result<(Node, u64), Node> {
    match node {
        Node::Block(b) => Ok((Node::Seq(Vec::new()), b)),
        Node::Seq(mut v) => match v.pop() {
            Some(Node::Block(b)) => Ok((Node::Seq(v), b)),
            Some(other) => {
                v.push(other);
                Err(Node::Seq(v))
            }
            None => Err(Node::Seq(v)),
        },
        other => Err(other),
    }
}

/// Turn this loop's own back-edge and exit gotos into `Continue` and
/// `Break`. A nested `Loop` is left alone: its `Continue` means *its*
/// header, so a goto that lands inside one has to stay a goto.
fn tighten(node: Node, header: u64, follow: Option<u64>, depth: usize) -> (Node, bool) {
    if depth > MAX_TREE_DEPTH {
        return (node, false);
    }
    let mut broke = false;
    let down = |n: Node, broke: &mut bool| {
        let (n, b) = tighten(n, header, follow, depth + 1);
        *broke |= b;
        n
    };
    let out = match node {
        Node::Goto(t) if t == header => Node::Continue,
        Node::Goto(t) if Some(t) == follow => {
            broke = true;
            Node::Break
        }
        Node::Seq(v) => Node::Seq(v.into_iter().map(|c| down(c, &mut broke)).collect()),
        Node::If {
            cond,
            then_body,
            else_body,
        } => Node::If {
            cond,
            then_body: Box::new(down(*then_body, &mut broke)),
            else_body: else_body.map(|e| Box::new(down(*e, &mut broke))),
        },
        Node::Switch { block, cases } => Node::Switch {
            block,
            cases: cases
                .into_iter()
                .map(|(t, b)| (t, down(b, &mut broke)))
                .collect(),
        },
        other => other,
    };
    (out, broke)
}

/// Drop a `Continue` that is the last thing a loop body does: falling off
/// the end of the body *is* the back edge, so the explicit jump is noise.
fn strip_trailing_continue(node: Node) -> Node {
    match node {
        Node::Seq(mut v) if v.len() > 1 && matches!(v.last(), Some(Node::Continue)) => {
            v.pop();
            if v.len() == 1 {
                v.pop().unwrap_or(Node::Seq(Vec::new()))
            } else {
                Node::Seq(v)
            }
        }
        other => other,
    }
}

/// Every block VA a node covers, in ascending order.
fn covered(node: &Node, out: &mut BTreeSet<u64>, depth: usize) {
    if depth > MAX_TREE_DEPTH {
        return;
    }
    match node {
        Node::Block(b) | Node::Opaque { block: b, .. } => {
            out.insert(*b);
        }
        Node::Seq(v) => {
            for c in v {
                covered(c, out, depth + 1);
            }
        }
        Node::If {
            then_body,
            else_body,
            ..
        } => {
            covered(then_body, out, depth + 1);
            if let Some(e) = else_body {
                covered(e, out, depth + 1);
            }
        }
        Node::Loop { body, .. } => covered(body, out, depth + 1),
        Node::Switch { block, cases } => {
            out.insert(*block);
            for (_, c) in cases {
                covered(c, out, depth + 1);
            }
        }
        Node::Break | Node::Continue | Node::Goto(_) => {}
    }
}

// ---------------------------------------------------------------------------
// The region graph
// ---------------------------------------------------------------------------

/// One collapsed piece of the function: the tree built for it, the blocks
/// it covers, and the edges it has not realized yet.
#[derive(Debug, Clone)]
struct Region {
    node: Node,
    blocks: BTreeSet<u64>,
    exit: Exits,
}

/// The iterative collapse.
struct Builder<'a> {
    f: &'a SsaFunction,
    tables: &'a BTreeMap<u64, Vec<u64>>,
    budget: usize,
    rounds: usize,
    gotos: usize,
    capped: bool,
}

impl Builder<'_> {
    /// One region per reachable block, with every out-of-function edge
    /// already realized as an external `Goto`.
    fn initial_regions(&mut self) -> BTreeMap<u64, Region> {
        let cfg = irssa::Cfg::analyze(self.f.entry, &raw_successors(self.f));
        let mut regions = BTreeMap::new();
        for &va in cfg.succs.keys() {
            let node = match opaque_reason(self.f, self.tables, va) {
                Some(reason) => Node::Opaque { block: va, reason },
                None => Node::Block(va),
            };
            let mut region = Region {
                node,
                blocks: BTreeSet::from([va]),
                exit: exits(self.f, self.tables, va),
            };
            if matches!(region.exit, Exits::Inexpressible) {
                region.exit = Exits::None;
            }
            // Tail jumps leave the function: realize them here, once, and
            // never count them as structuring edges.
            let external: Vec<u64> = region
                .exit
                .targets()
                .into_iter()
                .filter(|t| !self.f.blocks.contains_key(t))
                .collect();
            for t in external {
                redirect(&mut region, t, Node::Goto(t));
            }
            regions.insert(va, region);
        }
        regions
    }

    /// Collapse `regions` (a single-entry subgraph rooted at `entry`) to
    /// one region. Recurses once per loop nesting level.
    fn collapse(&mut self, mut regions: BTreeMap<u64, Region>, entry: u64, depth: usize) -> Region {
        loop {
            if self.budget == 0 || depth > MAX_STRUCT_DEPTH {
                self.capped = true;
                return self.join(regions, entry);
            }
            self.budget -= 1;
            self.rounds += 1;
            if regions.len() == 1 {
                let only = regions.values().next().expect("one region");
                if matches!(only.exit, Exits::None) {
                    return regions.into_values().next().expect("one region");
                }
            }
            if self.try_schemas(&mut regions, entry, depth) {
                continue;
            }
            if self.virtualize_lowest(&mut regions) {
                continue;
            }
            return self.join(regions, entry);
        }
    }

    /// Walk the region heads in post-order from `entry` (unreachable
    /// leftovers ascending after them) and apply the first schema that
    /// matches. `true` when something collapsed.
    ///
    /// Cyclic before acyclic — Phoenix's split — and *graph-wide*: every
    /// loop header is offered the loop schemas before any head is
    /// offered an acyclic one. Without that an if-then somewhere inside
    /// a body would absorb the loop's own exit block before the loop was
    /// ever recognized, and the loop would degrade to `while (true)`
    /// with the test buried in it.
    fn try_schemas(
        &mut self,
        regions: &mut BTreeMap<u64, Region>,
        entry: u64,
        depth: usize,
    ) -> bool {
        let graph = succ_map(regions);
        let cfg = irssa::Cfg::analyze(entry, &graph);
        let mut headers: BTreeSet<u64> = BTreeSet::new();
        for (&l, ss) in &graph {
            for &s in ss {
                if s == l || cfg.strictly_dominates(s, l) {
                    headers.insert(s);
                }
            }
        }
        let order = post_order(regions, entry);
        for &h in &order {
            if headers.contains(&h)
                && (self.try_self_loop(regions, h)
                    || self.try_loop(regions, &graph, &cfg, entry, h, depth))
            {
                return true;
            }
        }
        for &h in &order {
            if self.try_sequence(regions, entry, h)
                || self.try_if(regions, entry, h)
                || self.try_switch(regions, entry, h)
            {
                return true;
            }
        }
        false
    }

    /// Sequence: `h`'s single successor has no other predecessor.
    fn try_sequence(&mut self, regions: &mut BTreeMap<u64, Region>, entry: u64, h: u64) -> bool {
        let Some(Exits::One(n)) = regions.get(&h).map(|r| r.exit.clone()) else {
            return false;
        };
        if n == h || n == entry || !regions.contains_key(&n) {
            return false;
        }
        if preds(regions).get(&n).map(Vec::as_slice) != Some(&[h]) {
            return false;
        }
        let next = regions.remove(&n).expect("checked above");
        let r = regions.get_mut(&h).expect("checked above");
        r.node = seq_push(
            std::mem::replace(&mut r.node, Node::Seq(Vec::new())),
            next.node,
        );
        r.blocks.extend(next.blocks);
        r.exit = next.exit;
        true
    }

    /// If-then-else, then if-then with the taken arm, then if-then with
    /// the fall-through arm (which is the negated form).
    fn try_if(&mut self, regions: &mut BTreeMap<u64, Region>, entry: u64, h: u64) -> bool {
        let Some(Exits::Cond {
            block,
            taken,
            fallthrough,
        }) = regions.get(&h).map(|r| r.exit.clone())
        else {
            return false;
        };
        if taken == h || fallthrough == h {
            return false; // a self-loop, not a conditional region
        }
        let p = preds(regions);
        let owns = |t: u64| t != entry && p.get(&t).map(Vec::as_slice) == Some(&[h]);
        let plain = |t: u64| match regions.get(&t).map(|r| r.exit.clone()) {
            Some(Exits::None) => Some(None),
            Some(Exits::One(w)) => Some(Some(w)),
            _ => None,
        };

        // If-then-else: both arms are ours and converge.
        if owns(taken)
            && owns(fallthrough)
            && let (Some(a), Some(b)) = (plain(taken), plain(fallthrough))
            && let Some(follow) = converge(a, b)
        {
            let t = regions.remove(&taken).expect("checked above");
            let e = regions.remove(&fallthrough).expect("checked above");
            let r = regions.get_mut(&h).expect("checked above");
            let mut blocks = std::mem::take(&mut r.blocks);
            blocks.extend(t.blocks);
            blocks.extend(e.blocks);
            r.blocks = blocks;
            r.node = seq_push(
                std::mem::replace(&mut r.node, Node::Seq(Vec::new())),
                Node::If {
                    cond: Cond {
                        block,
                        negated: false,
                    },
                    then_body: Box::new(t.node),
                    else_body: Some(Box::new(e.node)),
                },
            );
            r.exit = match follow {
                Some(w) => Exits::One(w),
                None => Exits::None,
            };
            return true;
        }

        // If-then: one arm is ours and falls into the other side.
        for (arm, follow, negated) in [(taken, fallthrough, false), (fallthrough, taken, true)] {
            if !owns(arm) {
                continue;
            }
            let Some(inner) = plain(arm) else { continue };
            if !(inner.is_none() || inner == Some(follow)) {
                continue;
            }
            let a = regions.remove(&arm).expect("checked above");
            let r = regions.get_mut(&h).expect("checked above");
            r.blocks.extend(a.blocks);
            r.node = seq_push(
                std::mem::replace(&mut r.node, Node::Seq(Vec::new())),
                Node::If {
                    cond: Cond { block, negated },
                    then_body: Box::new(a.node),
                    else_body: None,
                },
            );
            r.exit = Exits::One(follow);
            return true;
        }
        false
    }

    /// Self-loop: the region's remaining edge comes back to its own head.
    fn try_self_loop(&mut self, regions: &mut BTreeMap<u64, Region>, h: u64) -> bool {
        let Some(exit) = regions.get(&h).map(|r| r.exit.clone()) else {
            return false;
        };
        let (cond, after) = match exit {
            Exits::One(t) if t == h => (None, Exits::None),
            Exits::Cond {
                block,
                taken,
                fallthrough,
            } if taken == h => (
                Some(Cond {
                    block,
                    negated: false,
                }),
                Exits::One(fallthrough),
            ),
            Exits::Cond {
                block,
                taken,
                fallthrough,
            } if fallthrough == h => (
                Some(Cond {
                    block,
                    negated: true,
                }),
                Exits::One(taken),
            ),
            _ => return false,
        };
        let r = regions.get_mut(&h).expect("checked above");
        let kind = loop_kind(r.blocks.len(), cond.map(|c| c.block), h);
        let body = strip_trailing_continue(std::mem::replace(&mut r.node, Node::Seq(Vec::new())));
        r.node = Node::Loop {
            kind,
            cond,
            body: Box::new(body),
        };
        r.exit = after;
        true
    }

    /// Natural while / do-while: a back edge to a dominating header, a
    /// single-entry body, and one follow every exit edge agrees on.
    fn try_loop(
        &mut self,
        regions: &mut BTreeMap<u64, Region>,
        graph: &BTreeMap<u64, Vec<u64>>,
        cfg: &irssa::Cfg,
        entry: u64,
        h: u64,
        depth: usize,
    ) -> bool {
        // Latches: regions with an edge back to a header that dominates
        // them. `h == l` is the self-loop schema's business.
        let latches: Vec<u64> = graph
            .iter()
            .filter(|(l, s)| s.contains(&h) && **l != h && cfg.strictly_dominates(h, **l))
            .map(|(l, _)| *l)
            .collect();
        if latches.is_empty() {
            return false;
        }
        let body = natural_loop(graph, h, &latches);
        if body.len() < 2 {
            return false;
        }
        // Single entry: nothing from outside may enter past the header.
        for (u, ss) in graph {
            if body.contains(u) {
                continue;
            }
            if ss.iter().any(|v| body.contains(v) && *v != h) {
                return false;
            }
        }
        // Exit edges and the follow node.
        let mut exit_edges: Vec<(u64, u64)> = Vec::new();
        for u in &body {
            for v in graph.get(u).map(Vec::as_slice).unwrap_or(&[]) {
                if !body.contains(v) {
                    exit_edges.push((*u, *v));
                }
            }
        }
        let follow = match follow_node(cfg, &body, &exit_edges, h) {
            Some(x) => {
                if exit_edges.iter().any(|(_, v)| *v != x) {
                    return false; // an exit the follow does not cover
                }
                Some(x)
            }
            None => {
                if !exit_edges.is_empty() {
                    return false;
                }
                None
            }
        };
        if follow == Some(entry) {
            // The function entry can never sit inside or after a region
            // it dominates; leave this to a goto.
            return false;
        }

        // The loop's own condition: the header's test if it decides the
        // exit, else the single latch whose test does.
        let mut primary: Option<(u64, Cond)> = None;
        if let Some(x) = follow {
            let candidate = |r: &Region| match &r.exit {
                Exits::Cond {
                    block,
                    taken,
                    fallthrough,
                } => {
                    let (t, f) = (*taken, *fallthrough);
                    if t == x && body.contains(&f) || f == x && body.contains(&t) {
                        Some(Cond {
                            block: *block,
                            negated: t == x,
                        })
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(c) = regions.get(&h).and_then(candidate) {
                primary = Some((h, c));
            } else {
                let mut found: Option<(u64, Cond)> = None;
                for l in &latches {
                    if let Some(c) = regions.get(l).and_then(candidate) {
                        if found.is_some() {
                            found = None;
                            break; // ambiguous: fall back to breaks
                        }
                        found = Some((*l, c));
                    }
                }
                primary = found;
            }
        }

        // Cut the body out: back edges become `Continue`, exit edges
        // become `Break`, and the loop's own test is detached (the
        // `Loop` node carries it).
        let mut sub: BTreeMap<u64, Region> = BTreeMap::new();
        for va in &body {
            let Some(mut r) = regions.get(va).cloned() else {
                return false;
            };
            let mut ok = true;
            if primary.map(|(p, _)| p) == Some(*va) {
                let x = follow.expect("a primary exit implies a follow");
                ok &= detach(&mut r, x);
            }
            for t in r.exit.targets() {
                if t == h || !body.contains(&t) {
                    // A goto is position-independent; `tighten` turns the
                    // ones that end up at this loop's own level into
                    // `Continue`/`Break` once the body is built.
                    ok &= redirect(&mut r, t, Node::Goto(t));
                }
            }
            if !ok {
                return false; // a table dispatch cannot carry a break
            }
            sub.insert(*va, r);
        }

        let built = self.collapse(sub, h, depth + 1);
        let kind = loop_kind(built.blocks.len(), primary.map(|(_, c)| c.block), h);
        let (tightened, breaks) = tighten(built.node, h, follow, 0);
        let node = Node::Loop {
            kind,
            cond: primary.map(|(_, c)| c),
            body: Box::new(strip_trailing_continue(tightened)),
        };
        // The loop leaves an edge for the follow region to consume only
        // if its own test or a `Break` carries one: an exit that stayed a
        // goto (because it sits inside a nested loop) is already realized.
        let pending_exit = primary.is_some() || breaks;
        for va in &body {
            regions.remove(va);
        }
        regions.insert(
            h,
            Region {
                node,
                blocks: built.blocks,
                exit: match follow.filter(|_| pending_exit) {
                    Some(x) => Exits::One(x),
                    None => Exits::None,
                },
            },
        );
        true
    }

    /// Switch: a proven dispatch whose cases are ours and converge.
    fn try_switch(&mut self, regions: &mut BTreeMap<u64, Region>, entry: u64, h: u64) -> bool {
        let Some(Exits::Table { block, targets }) = regions.get(&h).map(|r| r.exit.clone()) else {
            return false;
        };
        let p = preds(regions);
        let mut follow: Option<Option<u64>> = None;
        for t in &targets {
            if *t == entry || p.get(t).map(Vec::as_slice) != Some(&[h]) {
                return false;
            }
            let plain = match regions.get(t).map(|r| r.exit.clone()) {
                Some(Exits::None) => None,
                Some(Exits::One(w)) => Some(w),
                _ => return false,
            };
            follow = Some(match follow {
                None => plain,
                Some(prev) => match converge(prev, plain) {
                    Some(w) => w,
                    None => return false,
                },
            });
        }
        let mut cases = Vec::with_capacity(targets.len());
        for t in &targets {
            let r = regions.remove(t).expect("checked above");
            cases.push((*t, r.node, r.blocks));
        }
        let r = regions.get_mut(&h).expect("checked above");
        let node = std::mem::replace(&mut r.node, Node::Seq(Vec::new()));
        let (prefix, scrutinee) = match split_tail_block(node) {
            Ok(pair) => pair,
            // Defensively unreachable: a table exit ends in its block.
            Err(node) => {
                r.node = node;
                return false;
            }
        };
        debug_assert_eq!(scrutinee, block);
        let mut bodies = Vec::with_capacity(cases.len());
        for (t, node, blocks) in cases {
            r.blocks.extend(blocks);
            bodies.push((t, node));
        }
        r.node = seq_push(
            prefix,
            Node::Switch {
                block: scrutinee,
                cases: bodies,
            },
        );
        r.exit = match follow.flatten() {
            Some(w) => Exits::One(w),
            None => Exits::None,
        };
        true
    }

    /// Virtualize the lowest `(source region head VA, target VA)` edge.
    /// `false` when the graph has no edges left at all.
    fn virtualize_lowest(&mut self, regions: &mut BTreeMap<u64, Region>) -> bool {
        let mut best: Option<(u64, u64)> = None;
        for (&h, r) in regions.iter() {
            for t in r.exit.targets() {
                let key = (h, t);
                if best.is_none_or(|b| key < b) {
                    best = Some(key);
                }
            }
        }
        let Some((h, t)) = best else { return false };
        let mut r = regions.remove(&h).expect("the head we just read");
        self.virtualize(&mut r, t);
        regions.insert(h, r);
        true
    }

    /// Realize one of a region's edges as an explicit goto and count it.
    /// A dispatch has no room for a single goto, so its whole case set is
    /// realized at once.
    fn virtualize(&mut self, r: &mut Region, target: u64) {
        if let Exits::Table { targets, .. } = r.exit.clone() {
            let cases: Vec<(u64, Node)> = targets.iter().map(|&t| (t, Node::Goto(t))).collect();
            self.gotos += cases.len();
            let node = std::mem::replace(&mut r.node, Node::Seq(Vec::new()));
            r.node = match split_tail_block(node) {
                Ok((prefix, scrutinee)) => seq_push(
                    prefix,
                    Node::Switch {
                        block: scrutinee,
                        cases,
                    },
                ),
                // Defensively unreachable: a table exit ends in its own
                // block leaf. Keep the node; `check` reports the edges.
                Err(node) => node,
            };
            r.exit = Exits::None;
            return;
        }
        self.gotos += 1;
        if !redirect(r, target, Node::Goto(target)) {
            r.exit = Exits::None; // defensively unreachable
        }
    }

    /// The degrade: realize every remaining edge as a goto and lay the
    /// regions out entry first, then ascending. Each region ends with no
    /// live predecessor, so the concatenation invents no flow.
    fn join(&mut self, mut regions: BTreeMap<u64, Region>, entry: u64) -> Region {
        let heads: Vec<u64> = regions.keys().copied().collect();
        for r in regions.values_mut() {
            while let Some(&t) = r.exit.targets().first() {
                self.virtualize(r, t);
            }
        }
        let mut order: Vec<u64> = heads.iter().copied().filter(|h| *h != entry).collect();
        if regions.contains_key(&entry) {
            order.insert(0, entry);
        }
        let mut node = Node::Seq(Vec::new());
        let mut blocks = BTreeSet::new();
        for h in order {
            if let Some(r) = regions.remove(&h) {
                node = seq_push(node, r.node);
                blocks.extend(r.blocks);
            }
        }
        Region {
            node,
            blocks,
            exit: Exits::None,
        }
    }
}

/// Which loop kind a body of `blocks` blocks with its test at `cond` and
/// its header at `h` is.
fn loop_kind(blocks: usize, cond: Option<u64>, h: u64) -> LoopKind {
    if blocks <= 1 {
        LoopKind::SelfLoop
    } else if cond.is_none_or(|c| c == h) {
        LoopKind::While
    } else {
        LoopKind::DoWhile
    }
}

/// Two arms' follow regions, when they agree (an arm that terminates
/// agrees with anything).
fn converge(a: Option<u64>, b: Option<u64>) -> Option<Option<u64>> {
    match (a, b) {
        (None, x) | (x, None) => Some(x),
        (Some(x), Some(y)) if x == y => Some(Some(x)),
        _ => None,
    }
}

/// Realize one of a region's edges as `node`, keeping the rest. `false`
/// when the region's exit cannot carry it (a table dispatch).
fn redirect(r: &mut Region, target: u64, node: Node) -> bool {
    let take = |r: &mut Region| std::mem::replace(&mut r.node, Node::Seq(Vec::new()));
    match r.exit.clone() {
        Exits::One(t) if t == target => {
            r.node = seq_push(take(r), node);
            r.exit = Exits::None;
            true
        }
        Exits::Cond {
            block,
            taken,
            fallthrough,
        } if target == taken || target == fallthrough => {
            let negated = target == fallthrough;
            r.node = seq_push(
                take(r),
                Node::If {
                    cond: Cond { block, negated },
                    then_body: Box::new(node),
                    else_body: None,
                },
            );
            r.exit = Exits::One(if negated { taken } else { fallthrough });
            true
        }
        _ => false,
    }
}

/// Drop one of a two-way region's edges *without* realizing it — the
/// `Loop` node that owns the test realizes it instead.
fn detach(r: &mut Region, target: u64) -> bool {
    match r.exit.clone() {
        Exits::Cond {
            taken, fallthrough, ..
        } if target == taken || target == fallthrough => {
            r.exit = Exits::One(if target == taken { fallthrough } else { taken });
            true
        }
        _ => false,
    }
}

/// The region graph as a successor map, successors ascending.
fn succ_map(regions: &BTreeMap<u64, Region>) -> BTreeMap<u64, Vec<u64>> {
    regions
        .iter()
        .map(|(&h, r)| {
            let mut t: Vec<u64> = r
                .exit
                .targets()
                .into_iter()
                .filter(|x| regions.contains_key(x))
                .collect();
            t.sort_unstable();
            t.dedup();
            (h, t)
        })
        .collect()
}

/// Region head -> predecessor heads, ascending and deduplicated.
fn preds(regions: &BTreeMap<u64, Region>) -> BTreeMap<u64, Vec<u64>> {
    let mut p: BTreeMap<u64, Vec<u64>> = regions.keys().map(|&h| (h, Vec::new())).collect();
    for (&h, r) in regions {
        for t in r.exit.targets() {
            if let Some(list) = p.get_mut(&t)
                && !list.contains(&h)
            {
                list.push(h);
            }
        }
    }
    for list in p.values_mut() {
        list.sort_unstable();
    }
    p
}

/// Post-order over the region graph from `entry` in ascending successor
/// order, then any region the walk never reached, ascending.
fn post_order(regions: &BTreeMap<u64, Region>, entry: u64) -> Vec<u64> {
    let graph = succ_map(regions);
    let mut out = Vec::with_capacity(regions.len());
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    if graph.contains_key(&entry) {
        seen.insert(entry);
        let mut stack: Vec<(u64, usize)> = vec![(entry, 0)];
        while let Some(top) = stack.last_mut() {
            let v = top.0;
            match graph.get(&v).and_then(|l| l.get(top.1)).copied() {
                Some(s) => {
                    top.1 += 1;
                    if seen.insert(s) {
                        stack.push((s, 0));
                    }
                }
                None => {
                    out.push(v);
                    stack.pop();
                }
            }
        }
    }
    for &h in regions.keys() {
        if !seen.contains(&h) {
            out.push(h);
        }
    }
    out
}

/// The natural loop of `header`: the header plus every region that
/// reaches a latch without passing through the header.
fn natural_loop(graph: &BTreeMap<u64, Vec<u64>>, header: u64, latches: &[u64]) -> BTreeSet<u64> {
    let mut body: BTreeSet<u64> = BTreeSet::from([header]);
    let mut stack: Vec<u64> = Vec::new();
    for &l in latches {
        if body.insert(l) {
            stack.push(l);
        }
    }
    let mut rev: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for (u, ss) in graph {
        for v in ss {
            rev.entry(*v).or_default().push(*u);
        }
    }
    while let Some(v) = stack.pop() {
        for &p in rev.get(&v).map(Vec::as_slice).unwrap_or(&[]) {
            if body.insert(p) {
                stack.push(p);
            }
        }
    }
    body
}

/// The loop's follow node, by the documented rule: the header's immediate
/// post-dominator when one exists outside the body, else the most
/// frequent exit-edge target, ties broken by lowest address.
fn follow_node(
    cfg: &irssa::Cfg,
    body: &BTreeSet<u64>,
    exit_edges: &[(u64, u64)],
    header: u64,
) -> Option<u64> {
    if exit_edges.is_empty() {
        return None;
    }
    if let Some(x) = ipostdom(cfg, header)
        && !body.contains(&x)
    {
        return Some(x);
    }
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for (_, v) in exit_edges {
        *counts.entry(*v).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(va, n)| (n, std::cmp::Reverse(va)))
        .map(|(va, _)| va)
}

/// The immediate post-dominator of `va`: the dominator of the reverse
/// graph, rooted at a virtual exit above every sink. Reuses the one
/// Cooper–Harvey–Kennedy implementation in [`irssa`].
fn ipostdom(cfg: &irssa::Cfg, va: u64) -> Option<u64> {
    let mut virtual_exit = u64::MAX;
    while cfg.succs.contains_key(&virtual_exit) {
        virtual_exit = virtual_exit.checked_sub(1)?;
    }
    let mut rev: BTreeMap<u64, Vec<u64>> = cfg.succs.keys().map(|&v| (v, Vec::new())).collect();
    let mut sinks: Vec<u64> = Vec::new();
    for (&u, ss) in &cfg.succs {
        if ss.is_empty() {
            sinks.push(u);
        }
        for &v in ss {
            rev.entry(v).or_default().push(u);
        }
    }
    for list in rev.values_mut() {
        list.sort_unstable();
        list.dedup();
    }
    rev.insert(virtual_exit, sinks);
    let post = irssa::Cfg::analyze(virtual_exit, &rev);
    post.idom.get(&va).copied().filter(|&x| x != virtual_exit)
}

// ---------------------------------------------------------------------------
// The tail re-split (de-optimization)
// ---------------------------------------------------------------------------

/// How a copy-safe tail leaf closes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TailEnd {
    /// Every remaining edge leaves the function: nothing at all (a
    /// shared `return`), or the external target of a tail jump, which
    /// the duplicate carries along as its goto.
    Closed(Option<u64>),
    /// One in-function edge remains, to this φ-free convergence target.
    /// The duplicate owes its realization: a `Continue`, `Break`, or
    /// fall-through the site can honestly spell.
    Open(u64),
}

/// The CFG edges that carry out-of-SSA copies — [`irout::out_of_ssa`]'s
/// own `edge_copies` key set, the ground truth the renderer executes.
/// This is what the duplication rules must actually respect: the pseudo
/// walk places an edge's copies at exactly one textual site, so an edge
/// *with* copies must never be realized twice — but an edge whose φs all
/// coalesced away (different SSA names for one variable) carries
/// nothing, and refusing it was the name-identity approximation this
/// set replaces. `out_of_ssa` is deterministic on the function, so the
/// pass, [`check`], and the renderer all see the same set.
fn copy_edges(f: &SsaFunction) -> BTreeSet<(u64, u64)> {
    irout::out_of_ssa(f).0.edge_copies.into_keys().collect()
}

/// Whether the CFG edge `from -> to` carries no out-of-SSA copies —
/// nothing for the renderer to place, so a duplicate may realize the
/// edge at every occurrence. `copies` is [`copy_edges`]' answer for the
/// function.
fn edge_copy_free(copies: &BTreeSet<(u64, u64)>, from: u64, to: u64) -> bool {
    !copies.contains(&(from, to))
}

/// The per-leaf sanction the re-split pass and [`check`] share, so the
/// two can never drift: a block is a copy-safe tail leaf when the
/// collapse holds it as a plain leaf — no opaque reason — and its exits
/// are duplicable: none, one out-of-function edge, or one in-function
/// edge that is [`edge_copy_free`]. The copy condition keeps
/// [`crate::irout`]'s edge copies correct downstream: a duplicate
/// realizes its outgoing edge at every occurrence, the pseudo walk
/// places an edge's copies at one textual site — so an edge that could
/// carry copies is never realized twice.
fn splittable_tail(
    f: &SsaFunction,
    tables: &BTreeMap<u64, Vec<u64>>,
    copies: &BTreeSet<(u64, u64)>,
    va: u64,
) -> Option<TailEnd> {
    f.blocks.get(&va)?;
    if opaque_reason(f, tables, va).is_some() {
        return None;
    }
    match exits(f, tables, va) {
        Exits::None => Some(TailEnd::Closed(None)),
        Exits::One(x) => match f.blocks.contains_key(&x) {
            false => Some(TailEnd::Closed(Some(x))),
            true if edge_copy_free(copies, va, x) => Some(TailEnd::Open(x)),
            true => None,
        },
        _ => None,
    }
}

/// The per-head sanction inversion three and [`check`] share, so the
/// two can never drift: a block may be duplicated as a *deciding*
/// block when the collapse holds it as a plain leaf ending in a
/// conditional branch, its statements before the branch are pure
/// register assignments — no store, no call or intrinsic — and few
/// ([`MAX_THREAD_STMTS`]), and **both** live edges are
/// [`edge_copy_free`] wherever they stay in the function: a threaded
/// duplicate realizes both out-edges at every occurrence, and
/// [`crate::irout`]'s copies for an edge have exactly one textual
/// placement. Returns the `(taken, fall-through)` sides.
fn threadable_head(
    f: &SsaFunction,
    tables: &BTreeMap<u64, Vec<u64>>,
    copies: &BTreeSet<(u64, u64)>,
    va: u64,
) -> Option<(u64, u64)> {
    let block = f.blocks.get(&va)?;
    if opaque_reason(f, tables, va).is_some() {
        return None;
    }
    let Exits::Cond {
        taken, fallthrough, ..
    } = exits(f, tables, va)
    else {
        return None;
    };
    let n = block.stmts.len();
    if n == 0 || n - 1 > MAX_THREAD_STMTS {
        return None;
    }
    if !block.stmts[..n - 1]
        .iter()
        .all(|s| matches!(s, Stmt::Assign { .. }))
    {
        return None;
    }
    for t in [taken, fallthrough] {
        if f.blocks.contains_key(&t) && !edge_copy_free(copies, va, t) {
            return None;
        }
    }
    Some((taken, fallthrough))
}

/// One target's duplicate: the chain of blocks it spells, in flow
/// order, and how it closes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TailDup {
    chain: Vec<u64>,
    end: TailEnd,
}

/// The duplicate for gotos into `va`, or `None` when `va` is no
/// copy-safe tail. The chain extends greedily through successors that
/// are themselves copy-safe leaves **and** currently the target of no
/// goto (`counts`) — a chain interior must never need a label — up to
/// [`MAX_TAIL_CHAIN`] blocks, stopping before any cycle.
fn tail_chain(
    f: &SsaFunction,
    tables: &BTreeMap<u64, Vec<u64>>,
    copies: &BTreeSet<(u64, u64)>,
    va: u64,
    counts: &BTreeMap<u64, usize>,
) -> Option<TailDup> {
    let mut chain = Vec::new();
    let mut v = va;
    loop {
        let end = splittable_tail(f, tables, copies, v)?;
        chain.push(v);
        let TailEnd::Open(x) = end else {
            return Some(TailDup { chain, end });
        };
        if chain.len() == MAX_TAIL_CHAIN
            || chain.contains(&x)
            || counts.get(&x).is_some_and(|&n| n > 0)
            || splittable_tail(f, tables, copies, x).is_none()
        {
            return Some(TailDup {
                chain,
                end: TailEnd::Open(x),
            });
        }
        v = x;
    }
}

/// Occurrences of each in-function `Goto` target, ascending.
fn goto_counts(f: &SsaFunction, node: &Node, out: &mut BTreeMap<u64, usize>, depth: usize) {
    if depth > MAX_TREE_DEPTH {
        return;
    }
    match node {
        Node::Goto(t) => {
            if f.blocks.contains_key(t) {
                *out.entry(*t).or_default() += 1;
            }
        }
        Node::Seq(v) => {
            for c in v {
                goto_counts(f, c, out, depth + 1);
            }
        }
        Node::If {
            then_body,
            else_body,
            ..
        } => {
            goto_counts(f, then_body, out, depth + 1);
            if let Some(e) = else_body {
                goto_counts(f, e, out, depth + 1);
            }
        }
        Node::Loop { body, .. } => goto_counts(f, body, out, depth + 1),
        Node::Switch { cases, .. } => {
            for (_, c) in cases {
                goto_counts(f, c, out, depth + 1);
            }
        }
        Node::Block(_) | Node::Opaque { .. } | Node::Break | Node::Continue => {}
    }
}

/// How one goto site realizes its duplicate's remaining edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SiteEnd {
    /// The chain closes itself (a `Closed` tail): nothing to spell.
    Closed,
    /// The convergence target is the enclosing loop's header.
    Continue,
    /// The convergence target is where a `break` lands at this site.
    Break,
    /// The convergence target is exactly the next textual consumer:
    /// the pending-set machinery realizes the edge, and a `switch`
    /// renderer spells its `break;`.
    Fall,
}

/// What the re-split's site walks know about a textual position — the
/// verifier's `Ctx`, plus the facts eligibility needs. `plan_sites` and
/// `apply_sites` thread it identically, so what the plan approved is
/// exactly what the rewrite converts.
#[derive(Debug, Clone, Copy)]
struct SiteCtx {
    /// The enclosing loop's header: where `Continue` realizes pending.
    cont: Option<u64>,
    /// Where a `Break` at this position is realized — the enclosing
    /// loop's own next textual consumer.
    brk_fall: Option<u64>,
    /// The clean consumer control falls to from this position, when
    /// there is one an added pending block may safely reach (an `If`
    /// demands exactly its deciding block, so it is never one).
    fall: Option<u64>,
    /// A switch case sits between the innermost loop and here: a C
    /// `break` would leave the switch, not the loop, so `Break` is not
    /// spellable.
    in_case: bool,
}

/// Where `node` would realize a pending set entering it: `None` when it
/// consumes nothing (an empty sequence), `Some(None)` when it consumes
/// unsafely (an `If`, or a body with no decidable entry), and
/// `Some(Some(va))` for a clean realization at `va`. Mirrors the
/// verifier's pending flow.
fn consumes_at(node: &Node, ctx: SiteCtx, depth: usize) -> Option<Option<u64>> {
    if depth > MAX_TREE_DEPTH {
        return Some(None);
    }
    match node {
        Node::Seq(v) => v.iter().find_map(|c| consumes_at(c, ctx, depth + 1)),
        Node::Block(b) | Node::Opaque { block: b, .. } | Node::Switch { block: b, .. } => {
            Some(Some(*b))
        }
        Node::Goto(t) => Some(Some(*t)),
        Node::Loop { body, .. } => Some(leaf_entry(body, depth + 1)),
        Node::Continue => Some(ctx.cont),
        Node::Break => Some(ctx.brk_fall),
        Node::If { .. } => Some(None),
    }
}

/// The clean fall-through consumer for a position whose remaining
/// siblings are `rest`: the first thing that consumes pending, or the
/// enclosing construct's own fall.
fn fall_of(rest: &[Node], ctx: SiteCtx, depth: usize) -> Option<u64> {
    rest.iter()
        .find_map(|n| consumes_at(n, ctx, depth))
        .unwrap_or(ctx.fall)
}

/// The exit side of a loop's condition, by the verifier's rule — the
/// block the re-split must never inline into that loop's body: the
/// duplicate would put the recorded exit inside the body and belie the
/// loop's condition.
fn loop_leave(
    f: &SsaFunction,
    tables: &BTreeMap<u64, Vec<u64>>,
    cond: Option<Cond>,
) -> Option<u64> {
    cond.and_then(|c| match exits(f, tables, c.block) {
        Exits::Cond {
            taken, fallthrough, ..
        } => Some(if c.negated { taken } else { fallthrough }),
        _ => None,
    })
}

/// Whether a goto site with context `ctx` can spell `dup`'s duplicate
/// honestly, and how. `None` keeps the site's goto: a split that saves
/// no goto — or would inline an enclosing loop's recorded exit
/// (`leaves`) into its body — is never made.
fn site_end(dup: &TailDup, ctx: SiteCtx, leaves: &[u64]) -> Option<SiteEnd> {
    if dup.chain.iter().any(|b| leaves.contains(b)) {
        return None;
    }
    match dup.end {
        TailEnd::Closed(_) => Some(SiteEnd::Closed),
        TailEnd::Open(conv) => {
            if ctx.cont == Some(conv) {
                Some(SiteEnd::Continue)
            } else if ctx.fall == Some(conv) {
                Some(SiteEnd::Fall)
            } else if ctx.brk_fall == Some(conv) && !ctx.in_case && ctx.cont.is_some() {
                Some(SiteEnd::Break)
            } else {
                None
            }
        }
    }
}

/// The duplicate node for one site: the chain's plain leaves, then the
/// realization the site chose.
fn dup_node(dup: &TailDup, end: SiteEnd) -> Node {
    let mut v: Vec<Node> = dup.chain.iter().map(|&b| Node::Block(b)).collect();
    match (dup.end, end) {
        (TailEnd::Closed(Some(x)), _) => v.push(Node::Goto(x)),
        (TailEnd::Open(_), SiteEnd::Continue) => v.push(Node::Continue),
        (TailEnd::Open(_), SiteEnd::Break) => v.push(Node::Break),
        _ => {}
    }
    if v.len() == 1 {
        v.pop().unwrap_or(Node::Seq(Vec::new()))
    } else {
        Node::Seq(v)
    }
}

/// The prefix of `full` that spells `len` leaves: shorter than the
/// whole chain, it leaves an open edge to the next chain block.
fn chain_prefix(full: &TailDup, len: usize) -> TailDup {
    if len >= full.chain.len() {
        return full.clone();
    }
    TailDup {
        chain: full.chain[..len].to_vec(),
        end: TailEnd::Open(full.chain[len]),
    }
}

/// Inversion three's duplicate for one goto target: the deciding block,
/// its two sides, and — per side, when that side's target is a linear
/// tail the thread may expose — the target's own copy-safe chain for
/// the arm to inline (computed once per round from the goto counts, so
/// the plan and the rewrite can never disagree).
#[derive(Debug, Clone, PartialEq, Eq)]
struct ThreadDup {
    block: u64,
    taken: u64,
    fallthrough: u64,
    arm_taken: Option<TailDup>,
    arm_fallthrough: Option<TailDup>,
}

/// How a threaded duplicate's `If` arm realizes its edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArmEnd {
    /// The arm's target is the enclosing loop's header.
    Continue,
    /// The arm's target is where a `break` lands at this site.
    Break,
    /// The arm's target leaves the function: the external goto travels
    /// with the duplicate, uncounted like any tail jump.
    External,
    /// The arm inlines this many leaves of its target's copy-safe
    /// chain — the fresh linear tail the thread exposes, spelled by the
    /// case-tail machinery — closing as recorded.
    Chain(usize, SiteEnd),
}

/// How one goto site spells a threaded duplicate: the polarity of the
/// materialized `If`, the arm realization, and the open (fall-through)
/// side's realization. `SiteEnd::Closed` on the open side means an
/// external target whose goto travels with the duplicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ThreadEnd {
    negated: bool,
    arm: ArmEnd,
    open: SiteEnd,
}

/// Duplicate leaves one threaded site spends: the deciding block plus
/// whatever chain its arm inlines.
fn thread_cost(end: &ThreadEnd) -> usize {
    1 + match end.arm {
        ArmEnd::Chain(len, _) => len,
        _ => 0,
    }
}

/// How the `If` arm can realize the edge to `target` at a site with
/// context `ctx`, with its cost in duplicate leaves. The arm has no
/// fall-through of its own — falling out of the arm belongs to the
/// *other* edge — so an inlined chain may close only position-free
/// (`Closed`) or through the site's `Continue`/`Break`.
fn arm_spelling(
    f: &SsaFunction,
    target: u64,
    chain: &Option<TailDup>,
    ctx: SiteCtx,
    leaves: &[u64],
) -> Option<(usize, ArmEnd)> {
    if !f.blocks.contains_key(&target) {
        return Some((0, ArmEnd::External));
    }
    if ctx.cont == Some(target) {
        return Some((0, ArmEnd::Continue));
    }
    if ctx.brk_fall == Some(target) && !ctx.in_case && ctx.cont.is_some() {
        return Some((0, ArmEnd::Break));
    }
    let full = chain.as_ref()?;
    let arm_ctx = SiteCtx { fall: None, ..ctx };
    (1..=full.chain.len()).find_map(|len| {
        let prefix = chain_prefix(full, len);
        site_end(&prefix, arm_ctx, leaves).map(|end| (len, ArmEnd::Chain(len, end)))
    })
}

/// How the open (fall-through) side can realize the edge to `target`:
/// the same spellings a chain duplicate's open edge gets — `Continue`,
/// plain fall-through, `Break` — or, for an external target, the
/// travelling goto (`Closed`).
fn open_spelling(f: &SsaFunction, target: u64, ctx: SiteCtx) -> Option<SiteEnd> {
    if !f.blocks.contains_key(&target) {
        return Some(SiteEnd::Closed);
    }
    if ctx.cont == Some(target) {
        Some(SiteEnd::Continue)
    } else if ctx.fall == Some(target) {
        Some(SiteEnd::Fall)
    } else if ctx.brk_fall == Some(target) && !ctx.in_case && ctx.cont.is_some() {
        Some(SiteEnd::Break)
    } else {
        None
    }
}

/// Whether a goto site with context `ctx` can spell `dup`'s threaded
/// duplicate honestly, and how: both out-edges must realize without a
/// new in-function goto, so every rewritten site still buys back
/// exactly one goto. Either polarity may carry the arm; the cheapest
/// spellable option wins, ties to the un-negated form. `None` keeps
/// the site's goto — including when the deciding block is an enclosing
/// loop's recorded exit (`leaves`), which must never move inside that
/// loop's body.
fn thread_end(
    f: &SsaFunction,
    dup: &ThreadDup,
    ctx: SiteCtx,
    leaves: &[u64],
) -> Option<ThreadEnd> {
    if leaves.contains(&dup.block) {
        return None;
    }
    let mut best: Option<(usize, ThreadEnd)> = None;
    for (negated, arm_target, open_target, arm_chain) in [
        (false, dup.taken, dup.fallthrough, &dup.arm_taken),
        (true, dup.fallthrough, dup.taken, &dup.arm_fallthrough),
    ] {
        let Some((_, arm)) = arm_spelling(f, arm_target, arm_chain, ctx, leaves) else {
            continue;
        };
        let Some(open) = open_spelling(f, open_target, ctx) else {
            continue;
        };
        let end = ThreadEnd { negated, arm, open };
        let cost = thread_cost(&end);
        if best.as_ref().is_none_or(|(c, _)| cost < *c) {
            best = Some((cost, end));
        }
    }
    best.map(|(_, end)| end)
}

/// The threaded duplicate for one site: the deciding block's plain
/// leaf, the real `If` whose arm realizes one edge, and the open
/// side's realization after it.
fn thread_node(dup: &ThreadDup, end: ThreadEnd) -> Node {
    let (arm_target, open_target, arm_chain) = if end.negated {
        (dup.fallthrough, dup.taken, &dup.arm_fallthrough)
    } else {
        (dup.taken, dup.fallthrough, &dup.arm_taken)
    };
    let arm_body = match end.arm {
        ArmEnd::Continue => Node::Continue,
        ArmEnd::Break => Node::Break,
        ArmEnd::External => Node::Goto(arm_target),
        ArmEnd::Chain(len, chain_end) => match arm_chain {
            Some(full) => dup_node(&chain_prefix(full, len), chain_end),
            // Defensively unreachable: a `Chain` arm is only ever
            // derived from a stored chain.
            None => Node::Goto(arm_target),
        },
    };
    let mut v = vec![
        Node::Block(dup.block),
        Node::If {
            cond: Cond {
                block: dup.block,
                negated: end.negated,
            },
            then_body: Box::new(arm_body),
            else_body: None,
        },
    ];
    match end.open {
        SiteEnd::Continue => v.push(Node::Continue),
        SiteEnd::Break => v.push(Node::Break),
        SiteEnd::Closed => v.push(Node::Goto(open_target)),
        SiteEnd::Fall => {}
    }
    Node::Seq(v)
}

/// Pass one of a re-split round: record every in-function goto site's
/// context, per target — the split is all-or-nothing per target, so a
/// rewritten tail is never also a goto target, and the approval loop
/// needs every site's context to pick the cheapest duplicate all of
/// them can spell.
#[allow(clippy::too_many_arguments)]
fn plan_sites(
    f: &SsaFunction,
    tables: &BTreeMap<u64, Vec<u64>>,
    node: &Node,
    ctx: SiteCtx,
    leaves: &mut Vec<u64>,
    plan: &mut BTreeMap<u64, Vec<(SiteCtx, Vec<u64>)>>,
    depth: usize,
) {
    if depth > MAX_TREE_DEPTH {
        return;
    }
    match node {
        Node::Goto(t) => {
            if f.blocks.contains_key(t) {
                plan.entry(*t).or_default().push((ctx, leaves.clone()));
            }
        }
        Node::Seq(v) => {
            for (i, c) in v.iter().enumerate() {
                let fall = fall_of(&v[i + 1..], ctx, depth + 1);
                plan_sites(f, tables, c, SiteCtx { fall, ..ctx }, leaves, plan, depth + 1);
            }
        }
        Node::If {
            then_body,
            else_body,
            ..
        } => {
            plan_sites(f, tables, then_body, ctx, leaves, plan, depth + 1);
            if let Some(e) = else_body {
                plan_sites(f, tables, e, ctx, leaves, plan, depth + 1);
            }
        }
        Node::Loop { cond, body, .. } => {
            let header = leaf_entry(body, depth + 1);
            let inner = SiteCtx {
                cont: header,
                brk_fall: ctx.fall,
                fall: header,
                in_case: false,
            };
            let leave = loop_leave(f, tables, *cond);
            if let Some(x) = leave {
                leaves.push(x);
            }
            plan_sites(f, tables, body, inner, leaves, plan, depth + 1);
            if leave.is_some() {
                leaves.pop();
            }
        }
        Node::Switch { cases, .. } => {
            let inner = SiteCtx {
                in_case: true,
                ..ctx
            };
            for (_, c) in cases {
                plan_sites(f, tables, c, inner, leaves, plan, depth + 1);
            }
        }
        Node::Block(_) | Node::Opaque { .. } | Node::Break | Node::Continue => {}
    }
}

/// One round's approved rewrites, per goto target: the chain duplicates
/// of the first two inversions and the threaded duplicates of the
/// third. Disjoint by the classifiers — a chain head is a plain leaf, a
/// threaded head carries a condition.
#[derive(Debug, Clone, Default)]
struct Approved {
    chains: BTreeMap<u64, TailDup>,
    threads: BTreeMap<u64, ThreadDup>,
}

/// Pass two: substitute the approved duplicates for their gotos,
/// keeping a rewritten loop's kind consistent with its new covered
/// count — inlining a tail into a one-block body turns `SelfLoop` into
/// the `While` the verifier recomputes. Threads [`SiteCtx`] exactly as
/// `plan_sites` does, so a site the plan approved converts and no
/// other does.
#[allow(clippy::too_many_arguments)]
fn apply_sites(
    f: &SsaFunction,
    tables: &BTreeMap<u64, Vec<u64>>,
    node: Node,
    ctx: SiteCtx,
    leaves: &mut Vec<u64>,
    approved: &Approved,
    changed: &mut bool,
    depth: usize,
) -> Node {
    if depth > MAX_TREE_DEPTH {
        return node;
    }
    match node {
        Node::Goto(t) => {
            let chain = approved
                .chains
                .get(&t)
                .and_then(|d| Some((d, site_end(d, ctx, leaves)?)));
            match chain {
                Some((d, e)) => {
                    *changed = true;
                    dup_node(d, e)
                }
                None => match approved
                    .threads
                    .get(&t)
                    .and_then(|d| Some((d, thread_end(f, d, ctx, leaves)?)))
                {
                    Some((d, e)) => {
                        *changed = true;
                        thread_node(d, e)
                    }
                    None => Node::Goto(t),
                },
            }
        }
        Node::Seq(v) => {
            let falls: Vec<Option<u64>> = (0..v.len())
                .map(|i| fall_of(&v[i + 1..], ctx, depth + 1))
                .collect();
            Node::Seq(
                v.into_iter()
                    .zip(falls)
                    .map(|(c, fall)| {
                        apply_sites(
                            f,
                            tables,
                            c,
                            SiteCtx { fall, ..ctx },
                            leaves,
                            approved,
                            changed,
                            depth + 1,
                        )
                    })
                    .collect(),
            )
        }
        Node::If {
            cond,
            then_body,
            else_body,
        } => Node::If {
            cond,
            then_body: Box::new(apply_sites(
                f, tables, *then_body, ctx, leaves, approved, changed, depth + 1,
            )),
            else_body: else_body.map(|e| {
                Box::new(apply_sites(
                    f, tables, *e, ctx, leaves, approved, changed, depth + 1,
                ))
            }),
        },
        Node::Loop { kind, cond, body } => {
            let header = leaf_entry(&body, depth + 1);
            let inner_ctx = SiteCtx {
                cont: header,
                brk_fall: ctx.fall,
                fall: header,
                in_case: false,
            };
            let leave = loop_leave(f, tables, cond);
            if let Some(x) = leave {
                leaves.push(x);
            }
            let mut inner = false;
            let body = apply_sites(
                f, tables, *body, inner_ctx, leaves, approved, &mut inner, depth + 1,
            );
            if leave.is_some() {
                leaves.pop();
            }
            let kind = if inner {
                *changed = true;
                match leaf_entry(&body, 0) {
                    Some(h) => {
                        let mut blocks = BTreeSet::new();
                        covered(&body, &mut blocks, 0);
                        loop_kind(blocks.len(), cond.map(|c| c.block), h)
                    }
                    None => kind,
                }
            } else {
                kind
            };
            Node::Loop {
                kind,
                cond,
                body: Box::new(body),
            }
        }
        Node::Switch { block, cases } => {
            let inner = SiteCtx {
                in_case: true,
                ..ctx
            };
            Node::Switch {
                block,
                cases: cases
                    .into_iter()
                    .map(|(t, c)| {
                        (
                            t,
                            apply_sites(f, tables, c, inner, leaves, approved, changed, depth + 1),
                        )
                    })
                    .collect(),
            }
        }
        other => other,
    }
}

/// The SAILR de-optimization described in the module docs: rewrite every
/// in-function goto whose target is a copy-safe tail — or a bounded
/// chain of them — into a duplicate the site can spell honestly, then
/// every goto whose target is a threadable condition into the real `If`
/// its copy decides. All-or-nothing per target, greedily in ascending
/// target order under `cap` (duplicate leaves, shared by all three
/// inversions, chains before threads within a round), to a fixpoint: a
/// round's rewrites can make a chain interior label-free — or expose a
/// fresh tail through a threaded arm — and unlock the next round's.
/// Each rewritten site buys back exactly one goto and creates no
/// in-function goto, so the goto count strictly falls on every
/// productive round and the loop cannot oscillate; a target that does
/// not fit sets [`StructStats::dup_capped`] and keeps its gotos.
fn resplit_tails(
    f: &SsaFunction,
    tables: &BTreeMap<u64, Vec<u64>>,
    copies: &BTreeSet<(u64, u64)>,
    root: Node,
    stats: &mut StructStats,
    cap: usize,
) -> Node {
    let root_ctx = SiteCtx {
        cont: None,
        brk_fall: None,
        fall: None,
        in_case: false,
    };
    let mut root = root;
    let mut spent = 0usize;
    // Every productive round spends at least one duplicate leaf, so the
    // budget bounds the rounds; the range is defense in depth.
    for _ in 0..=cap {
        let mut counts = BTreeMap::new();
        goto_counts(f, &root, &mut counts, 0);
        let dups: BTreeMap<u64, TailDup> = counts
            .keys()
            .filter_map(|&t| tail_chain(f, tables, copies, t, &counts).map(|d| (t, d)))
            .collect();
        // Threading runs where the cheaper inversions leave gotos: the
        // condition-carrying targets no chain can take.
        let heads: Vec<(u64, u64, u64)> = counts
            .keys()
            .filter(|t| !dups.contains_key(t))
            .filter_map(|&t| threadable_head(f, tables, copies, t).map(|(a, b)| (t, a, b)))
            .collect();
        if dups.is_empty() && heads.is_empty() {
            break;
        }
        let mut plan: BTreeMap<u64, Vec<(SiteCtx, Vec<u64>)>> = BTreeMap::new();
        plan_sites(f, tables, &root, root_ctx, &mut Vec::new(), &mut plan, 0);
        let mut chains: BTreeMap<u64, TailDup> = BTreeMap::new();
        for (t, full) in &dups {
            let Some(sites) = plan.get(t).filter(|s| !s.is_empty()) else {
                continue;
            };
            // The cheapest duplicate every site can spell: the shortest
            // prefix of the chain, extending only as far as it must.
            let Some(dup) = (1..=full.chain.len()).map(|k| chain_prefix(full, k)).find(|d| {
                sites
                    .iter()
                    .all(|(ctx, leaves)| site_end(d, *ctx, leaves).is_some())
            }) else {
                continue;
            };
            let cost = sites.len() * dup.chain.len();
            if spent + cost > cap {
                stats.dup_capped = true;
                continue;
            }
            spent += cost;
            stats.duplications += cost;
            stats.gotos = stats.gotos.saturating_sub(sites.len());
            chains.insert(*t, dup);
        }
        let mut threads: BTreeMap<u64, ThreadDup> = BTreeMap::new();
        for (t, taken, fallthrough) in &heads {
            let Some(sites) = plan.get(t).filter(|s| !s.is_empty()) else {
                continue;
            };
            // An arm may inline a side's copy-safe chain only when no
            // goto into it will remain — the target of no goto, or one
            // this round's chain approvals rewrite whole — so no
            // duplicated block is ever also a goto target.
            let arm_chain = |x: u64| -> Option<TailDup> {
                if !f.blocks.contains_key(&x) {
                    return None;
                }
                if counts.get(&x).is_some_and(|&n| n > 0) && !chains.contains_key(&x) {
                    return None;
                }
                tail_chain(f, tables, copies, x, &counts)
            };
            let dup = ThreadDup {
                block: *t,
                taken: *taken,
                fallthrough: *fallthrough,
                arm_taken: arm_chain(*taken),
                arm_fallthrough: arm_chain(*fallthrough),
            };
            // All-or-nothing per target: every site must spell both
            // edges, or the target keeps all its gotos.
            let Some(cost) = sites.iter().try_fold(0usize, |acc, (ctx, leaves)| {
                thread_end(f, &dup, *ctx, leaves).map(|e| acc + thread_cost(&e))
            }) else {
                continue;
            };
            if spent + cost > cap {
                stats.dup_capped = true;
                continue;
            }
            spent += cost;
            stats.duplications += cost;
            stats.threaded += sites.len();
            stats.gotos = stats.gotos.saturating_sub(sites.len());
            threads.insert(*t, dup);
        }
        if chains.is_empty() && threads.is_empty() {
            break;
        }
        let approved = Approved { chains, threads };
        let mut changed = false;
        root = apply_sites(
            f, tables, root, root_ctx, &mut Vec::new(), &approved, &mut changed, 0,
        );
    }
    root
}

/// The block a node's entry reaches, by the verifier's rule minus the
/// loop context (`Break`/`Continue` resolve to nothing here) — enough
/// to recompute a rewritten loop's kind against the same header the
/// verifier will find.
fn leaf_entry(node: &Node, depth: usize) -> Option<u64> {
    if depth > MAX_TREE_DEPTH {
        return None;
    }
    match node {
        Node::Block(b) | Node::Opaque { block: b, .. } | Node::Switch { block: b, .. } => Some(*b),
        Node::Seq(v) => v.iter().find_map(|c| leaf_entry(c, depth + 1)),
        Node::Loop { body, .. } => leaf_entry(body, depth + 1),
        Node::Goto(t) => Some(*t),
        Node::Break | Node::Continue | Node::If { .. } => None,
    }
}

// ---------------------------------------------------------------------------
// check
// ---------------------------------------------------------------------------

/// What the walk knows about where control goes from here.
#[derive(Debug, Clone, Copy)]
struct Ctx {
    /// The block control reaches when the current node finishes.
    after: Option<u64>,
    /// The enclosing loop's follow, for `Break`.
    brk: Option<u64>,
    /// The enclosing loop's header, for `Continue`.
    cont: Option<u64>,
}

struct Verifier<'a> {
    f: &'a SsaFunction,
    tables: &'a BTreeMap<u64, Vec<u64>>,
    /// [`copy_edges`]' answer for `f` — recomputed here so the verifier
    /// sanctions duplicates by the same ground truth the pass used,
    /// never a drifted approximation.
    copies: BTreeSet<(u64, u64)>,
    expected: BTreeMap<u64, BTreeSet<u64>>,
    seen: BTreeMap<u64, usize>,
    realized: BTreeSet<(u64, u64)>,
    /// One accumulator per enclosing loop: a `Break` hands its pending
    /// blocks to the loop, which hands them to whatever follows it, so
    /// the edge is realized exactly where the loop's own exit test is.
    breaks: Vec<BTreeSet<u64>>,
    /// Two-way blocks currently pending with nothing yet deciding their
    /// branch. A plain realization of such a block's edge would drop the
    /// branch at that occurrence — the condition-honesty rule that holds
    /// deciding duplicates to the same standard as originals.
    undecided: BTreeSet<u64>,
    /// The condition blocks of the enclosing loops: the `Loop` node is
    /// their decider, so their occurrences inside the body are exempt
    /// from the undecided rule.
    loop_conds: Vec<u64>,
    /// What an else-less `If` still owes: its block's pending must next
    /// realize to exactly the untaken side. Anything else funnels both
    /// polarities one way — a [`StructFault::Polarity`].
    owed: BTreeMap<u64, u64>,
    fault: Option<StructFault>,
}

impl Verifier<'_> {
    fn fail(&mut self, fault: StructFault) {
        if self.fault.is_none() {
            self.fault = Some(fault);
        }
    }

    /// Record one occurrence of `b`. An extra occurrence is sanctioned
    /// only when `dup_ok` — the caller saw the plain leaf of a copy-safe
    /// tail, the one duplication the re-split pass may make; the total
    /// budget is held against [`MAX_TAIL_SPLITS`] in [`check`].
    fn cover(&mut self, b: u64, dup_ok: bool) {
        if !self.f.blocks.contains_key(&b) {
            self.fail(StructFault::Foreign { block: b });
            return;
        }
        let n = self.seen.entry(b).or_default();
        *n += 1;
        if *n > 1 && !dup_ok {
            self.fail(StructFault::Duplicated { block: b });
        }
    }

    fn realize(&mut self, from: &BTreeSet<u64>, to: u64) {
        for &p in from {
            if self.undecided.contains(&p) {
                self.fail(StructFault::Undecided { block: p });
            }
            if let Some(req) = self.owed.remove(&p)
                && req != to
            {
                self.fail(StructFault::Polarity { block: p });
            }
            if !self
                .expected
                .get(&p)
                .is_some_and(|set: &BTreeSet<u64>| set.contains(&to))
            {
                self.fail(StructFault::InventedEdge { from: p, to });
            }
            self.realized.insert((p, to));
        }
    }

    /// The block control reaches when it enters `node`, as far as the
    /// tree says. `None` for a node that passes control straight on.
    fn entry_target(&self, node: &Node, ctx: Ctx, depth: usize) -> Option<u64> {
        if depth > MAX_TREE_DEPTH {
            return None;
        }
        match node {
            Node::Block(b) | Node::Opaque { block: b, .. } | Node::Switch { block: b, .. } => {
                Some(*b)
            }
            Node::Seq(v) => v.iter().find_map(|c| self.entry_target(c, ctx, depth + 1)),
            Node::Loop { body, .. } => self.entry_target(body, ctx, depth + 1),
            Node::Goto(t) => Some(*t),
            Node::Break => ctx.brk,
            Node::Continue => ctx.cont,
            Node::If { .. } => None,
        }
    }

    /// Walk `node` with `pending` — the blocks whose terminator has not
    /// been realized yet — and return the pending set it leaves behind.
    fn walk(
        &mut self,
        node: &Node,
        pending: BTreeSet<u64>,
        ctx: Ctx,
        depth: usize,
    ) -> BTreeSet<u64> {
        if depth > MAX_TREE_DEPTH {
            self.fail(StructFault::TooDeep);
            return BTreeSet::new();
        }
        match node {
            Node::Block(b) => {
                if opaque_reason(self.f, self.tables, *b).is_some() {
                    self.fail(StructFault::Opacity { block: *b });
                }
                // The one sanction per inversion family, shared with the
                // pass: a copy-safe tail leaf or a threadable head.
                let dup_ok = splittable_tail(self.f, self.tables, &self.copies, *b).is_some()
                    || threadable_head(self.f, self.tables, &self.copies, *b).is_some();
                self.leaf(*b, &pending, dup_ok)
            }
            Node::Opaque { block, reason } => {
                if opaque_reason(self.f, self.tables, *block) != Some(*reason) {
                    self.fail(StructFault::Opacity { block: *block });
                }
                self.leaf(*block, &pending, false)
            }
            Node::Seq(v) => {
                let mut p = pending;
                for (i, child) in v.iter().enumerate() {
                    let after = v[i + 1..]
                        .iter()
                        .find_map(|n| self.entry_target(n, ctx, depth + 1))
                        .or(ctx.after);
                    p = self.walk(child, p, Ctx { after, ..ctx }, depth + 1);
                }
                p
            }
            Node::Goto(t) => {
                if pending.is_empty() {
                    self.fail(StructFault::LooseJump);
                }
                self.realize(&pending, *t);
                BTreeSet::new()
            }
            Node::Break => {
                match self.breaks.last_mut() {
                    Some(acc) if !pending.is_empty() => acc.extend(pending),
                    _ => self.fail(StructFault::LooseJump),
                }
                BTreeSet::new()
            }
            Node::Continue => match ctx.cont {
                Some(x) => {
                    if pending.is_empty() {
                        self.fail(StructFault::LooseJump);
                    }
                    self.realize(&pending, x);
                    BTreeSet::new()
                }
                None => {
                    self.fail(StructFault::LooseJump);
                    BTreeSet::new()
                }
            },
            Node::If {
                cond,
                then_body,
                else_body,
            } => self.walk_if(cond, then_body, else_body.as_deref(), pending, ctx, depth),
            Node::Loop { kind, cond, body } => {
                self.walk_loop(*kind, *cond, body, pending, ctx, depth)
            }
            Node::Switch { block, cases } => self.walk_switch(*block, cases, pending, ctx, depth),
        }
    }

    /// A `Block`/`Opaque` leaf: it consumes the pending edges, and leaves
    /// itself pending only if it has an edge left to realize. A two-way
    /// leaf goes pending *undecided* — something must spell its branch
    /// (an `If` naming it, or the enclosing `Loop` whose condition it
    /// is) before any of its edges realize.
    fn leaf(&mut self, b: u64, pending: &BTreeSet<u64>, dup_ok: bool) -> BTreeSet<u64> {
        self.cover(b, dup_ok);
        self.realize(pending, b);
        if self.expected.get(&b).is_some_and(|s| !s.is_empty()) {
            if matches!(exits(self.f, self.tables, b), Exits::Cond { .. })
                && !self.loop_conds.contains(&b)
            {
                self.undecided.insert(b);
            }
            BTreeSet::from([b])
        } else {
            BTreeSet::new()
        }
    }

    fn walk_if(
        &mut self,
        cond: &Cond,
        then_body: &Node,
        else_body: Option<&Node>,
        pending: BTreeSet<u64>,
        ctx: Ctx,
        depth: usize,
    ) -> BTreeSet<u64> {
        // The `If` is its block's decider: the pending block's edges may
        // realize from here on. Another occurrence's outstanding
        // obligation is suspended while this one's arms realize the same
        // block's edges — obligations are per occurrence, the map is per
        // block, and sibling copies of a deciding block may overlap.
        self.undecided.remove(&cond.block);
        let suspended = self.owed.remove(&cond.block);
        if pending.len() != 1 || !pending.contains(&cond.block) {
            self.fail(StructFault::CondMisplaced { block: cond.block });
        }
        let Some((taken, fallthrough)) = self.sides(cond.block) else {
            if let Some(req) = suspended {
                self.owed.insert(cond.block, req);
            }
            return BTreeSet::new();
        };
        let (then_t, else_t) = if cond.negated {
            (fallthrough, taken)
        } else {
            (taken, fallthrough)
        };
        let then_entry = self.entry_target(then_body, ctx, depth + 1);
        if then_entry.is_some_and(|e| e != then_t) {
            self.fail(StructFault::Polarity { block: cond.block });
        }
        let mut out = self.walk(then_body, pending.clone(), ctx, depth + 1);
        match else_body {
            Some(e) => {
                let else_entry = self.entry_target(e, ctx, depth + 1);
                if else_entry.is_some_and(|x| x != else_t) {
                    self.fail(StructFault::Polarity { block: cond.block });
                }
                if then_entry.is_none() && else_entry.is_none() {
                    self.fail(StructFault::Polarity { block: cond.block });
                }
                out.extend(self.walk(e, pending, ctx, depth + 1));
                if let Some(req) = suspended {
                    self.owed.insert(cond.block, req);
                }
            }
            None => {
                if then_entry.is_none() {
                    self.fail(StructFault::Polarity { block: cond.block });
                }
                // The untaken side is still owed: the block's pending
                // must next realize to exactly it (a suspended sibling
                // obligation collapses into this one — merged pendings
                // realize at one shared consumer).
                self.owed.insert(cond.block, else_t);
                out.extend(pending);
            }
        }
        out
    }

    fn walk_loop(
        &mut self,
        kind: LoopKind,
        cond: Option<Cond>,
        body: &Node,
        pending: BTreeSet<u64>,
        ctx: Ctx,
        depth: usize,
    ) -> BTreeSet<u64> {
        let Some(header) = self.entry_target(body, ctx, depth + 1) else {
            self.fail(StructFault::BadLoop { block: 0 });
            return BTreeSet::new();
        };
        self.realize(&pending, header);
        let mut blocks = BTreeSet::new();
        covered(body, &mut blocks, 0);
        if loop_kind(blocks.len(), cond.map(|c| c.block), header) != kind {
            self.fail(StructFault::BadLoop { block: header });
        }
        let inner = Ctx {
            after: Some(header),
            brk: ctx.after,
            cont: Some(header),
        };
        // The `Loop` node is its condition block's decider — its stay
        // edge realizes inside the body with no `If`, and the node
        // itself realizes the leave edge — so that block is exempt from
        // the undecided rule for the body's duration.
        if let Some(c) = cond {
            self.loop_conds.push(c.block);
        }
        self.breaks.push(BTreeSet::new());
        let tail = self.walk(body, BTreeSet::new(), inner, depth + 1);
        let broke = self.breaks.pop().unwrap_or_default();
        if cond.is_some() {
            self.loop_conds.pop();
        }
        // Falling off the end of the body is the back edge; a pending
        // block with no edge to the header is a loop exit instead. A
        // block whose *only* edge is the back edge always falls back,
        // even when that edge is already realized — a sanctioned
        // duplicate of it, spelled earlier as a `Continue`, realizes
        // the same edge first. The already-realized test only decides
        // two-way blocks, where the fall-off must be the *other* edge.
        let mut out: BTreeSet<u64> = broke;
        for p in tail {
            let edges = self.expected.get(&p);
            let expects_header = edges.is_some_and(|s: &BTreeSet<u64>| s.contains(&header));
            let only_header = edges.is_some_and(|s| s.len() == 1 && s.contains(&header));
            if expects_header && (only_header || !self.realized.contains(&(p, header))) {
                self.realize(&BTreeSet::from([p]), header);
            } else {
                out.insert(p);
            }
        }
        if let Some(c) = cond {
            match self.sides(c.block) {
                Some((taken, fallthrough)) => {
                    let (stay, leave) = if c.negated {
                        (fallthrough, taken)
                    } else {
                        (taken, fallthrough)
                    };
                    if !blocks.contains(&c.block)
                        || !blocks.contains(&stay)
                        || blocks.contains(&leave)
                    {
                        self.fail(StructFault::BadLoop { block: c.block });
                    }
                    out.insert(c.block);
                }
                None => return out,
            }
        }
        out
    }

    fn walk_switch(
        &mut self,
        block: u64,
        cases: &[(u64, Node)],
        pending: BTreeSet<u64>,
        ctx: Ctx,
        depth: usize,
    ) -> BTreeSet<u64> {
        self.realize(&pending, block);
        self.cover(block, false);
        let proven = match exits(self.f, self.tables, block) {
            Exits::Table { targets, .. } => targets,
            _ => {
                self.fail(StructFault::BadSwitch { block });
                Vec::new()
            }
        };
        let seen: Vec<u64> = cases.iter().map(|(t, _)| *t).collect();
        if seen != proven {
            self.fail(StructFault::BadSwitch { block });
        }
        let mut out = BTreeSet::new();
        for (t, body) in cases {
            if self
                .entry_target(body, ctx, depth + 1)
                .is_some_and(|e| e != *t)
            {
                self.fail(StructFault::BadSwitch { block });
            }
            out.extend(self.walk(body, BTreeSet::from([block]), ctx, depth + 1));
        }
        out
    }

    /// The `(taken, fall-through)` sides of a condition block, or a
    /// recorded fault.
    fn sides(&mut self, block: u64) -> Option<(u64, u64)> {
        match exits(self.f, self.tables, block) {
            Exits::Cond {
                taken, fallthrough, ..
            } => Some((taken, fallthrough)),
            _ => {
                self.fail(StructFault::NotConditional { block });
                None
            }
        }
    }
}

/// Validate a structure tree against its function. Total and
/// side-effect-free: it recomputes the partition, the realized-edge set,
/// condition honesty and opacity from scratch — trusting nothing
/// [`structure`] wrote — and returns the first [`StructFault`]. Never
/// panics, on any tree.
pub fn check(
    f: &SsaFunction,
    tables: &BTreeMap<u64, Vec<u64>>,
    root: &Node,
) -> Result<(), StructFault> {
    let expected: BTreeMap<u64, BTreeSet<u64>> = f
        .blocks
        .keys()
        .map(|&b| (b, exits(f, tables, b).targets().into_iter().collect()))
        .collect();
    let mut v = Verifier {
        f,
        tables,
        copies: copy_edges(f),
        expected,
        seen: BTreeMap::new(),
        realized: BTreeSet::new(),
        breaks: Vec::new(),
        undecided: BTreeSet::new(),
        loop_conds: Vec::new(),
        owed: BTreeMap::new(),
        fault: None,
    };
    let ctx = Ctx {
        after: None,
        brk: None,
        cont: None,
    };
    v.walk(root, BTreeSet::new(), ctx, 0);
    if let Some(fault) = v.fault {
        return Err(fault);
    }
    // A two-way block still undecided when the walk ends dropped its
    // branch at some occurrence — even if its edges were realized at
    // another one.
    if let Some(&b) = v.undecided.iter().next() {
        return Err(StructFault::Undecided { block: b });
    }
    // Partition of the reachable blocks: every block at least once, and
    // the extra occurrences (each already vetted by `cover` as the plain
    // leaf of a copy-safe tail) within the re-split budget.
    for &b in f.blocks.keys() {
        if !v.seen.contains_key(&b) {
            return Err(StructFault::Uncovered { block: b });
        }
    }
    let extras: usize = v.seen.values().map(|&n| n.saturating_sub(1)).sum();
    if extras > MAX_TAIL_SPLITS
        && let Some((&b, _)) = v.seen.iter().find(|&(_, &n)| n > 1)
    {
        return Err(StructFault::Duplicated { block: b });
    }
    // Every edge realized, none invented (the invented half is caught as
    // the walk goes).
    for (&from, targets) in &v.expected {
        for &to in targets {
            if !v.realized.contains(&(from, to)) {
                return Err(StructFault::DroppedEdge { from, to });
            }
        }
    }
    // An untaken side still owed when the walk ends was never realized
    // at its occurrence — even if the edge itself was realized at
    // another one.
    if let Some((&b, _)) = v.owed.iter().next() {
        return Err(StructFault::Polarity { block: b });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

fn render_cond(cond: &Option<Cond>) -> String {
    match cond {
        None => "true".to_string(),
        Some(c) if c.negated => format!("!cond loc_{:x}", c.block),
        Some(c) => format!("cond loc_{:x}", c.block),
    }
}

fn render_node(out: &mut String, node: &Node, indent: usize, depth: usize) {
    if depth > MAX_TREE_DEPTH {
        let _ = writeln!(out, "{:indent$}...", "", indent = indent);
        return;
    }
    let pad = indent;
    match node {
        Node::Block(b) => {
            let _ = writeln!(out, "{:pad$}block loc_{b:x}", "");
        }
        Node::Opaque { block, reason } => {
            let why = match reason {
                OpaqueReason::Truncated => "truncated",
                OpaqueReason::IndirectJump => "indirect jump",
                OpaqueReason::Unstructurable => "undecidable exits",
            };
            let _ = writeln!(out, "{:pad$}opaque loc_{block:x} ({why})", "");
        }
        Node::Seq(v) => {
            for c in v {
                render_node(out, c, indent, depth + 1);
            }
        }
        Node::If {
            cond,
            then_body,
            else_body,
        } => {
            let _ = writeln!(out, "{:pad$}if {}", "", render_cond(&Some(*cond)));
            render_node(out, then_body, indent + 2, depth + 1);
            if let Some(e) = else_body {
                let _ = writeln!(out, "{:pad$}else", "");
                render_node(out, e, indent + 2, depth + 1);
            }
        }
        Node::Loop { kind, cond, body } => {
            let word = match kind {
                LoopKind::SelfLoop => "self-loop",
                LoopKind::While => "while",
                LoopKind::DoWhile => "do-while",
            };
            let _ = writeln!(out, "{:pad$}{word} {}", "", render_cond(cond));
            render_node(out, body, indent + 2, depth + 1);
        }
        Node::Switch { block, cases } => {
            let _ = writeln!(out, "{:pad$}switch loc_{block:x}", "");
            for (t, body) in cases {
                let _ = writeln!(out, "{:pad$}  case loc_{t:x}:", "");
                render_node(out, body, indent + 4, depth + 1);
            }
        }
        Node::Break => {
            let _ = writeln!(out, "{:pad$}break", "");
        }
        Node::Continue => {
            let _ = writeln!(out, "{:pad$}continue", "");
        }
        Node::Goto(t) => {
            let _ = writeln!(out, "{:pad$}goto loc_{t:x}", "");
        }
    }
}

/// Render a structure tree to a deterministic, indented dump in the style
/// of [`irssa::render`]: a `; {name} @ {entry} (structure)` header, the
/// tree, and a trailer for the blocks SSA left out. `\n`-terminated;
/// never panics, even on a hand-broken tree.
pub fn render(f: &SsaFunction, root: &Node) -> String {
    let mut out = String::new();
    let name = f
        .name
        .clone()
        .unwrap_or_else(|| format!("sub_{:x}", f.entry));
    let _ = writeln!(out, "; {name} @ {:#018x} (structure)", f.entry);
    render_node(&mut out, root, 0, 0);
    if !f.skipped.is_empty() {
        let list = f
            .skipped
            .iter()
            .map(|s| format!("loc_{s:x}"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "; unreachable, not structured: {list}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, Flag, Reg, Width};
    use crate::irlift;
    use crate::model::Arch;

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
    /// `zf := rax == n` — a flag definition to branch on.
    fn set_flag(n: u64) -> Stmt {
        assign(
            Reg::flag(Flag::Zero),
            Expr::binary(BinOp::Eq, read(ra(0, Width::W64)), c(n, Width::W64)),
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
    /// `*(r5) := 1` — an effectful statement no inversion may thread.
    fn store() -> Stmt {
        Stmt::Store {
            addr: read(ra(5, Width::W64)),
            value: c(1, Width::W64),
        }
    }
    /// An indirect jump through a register: the shape a jump table has to
    /// prove.
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
            arch: Arch::X86_64,
            blocks: blocks.into_iter().map(|b| (b.start, b)).collect(),
        }
    }

    fn build(f: &irlift::LiftedFunction) -> SsaFunction {
        let ssa = irssa::construct(f).expect("well-formed input constructs");
        assert_eq!(irssa::check(&ssa), Ok(()), "input SSA must check");
        ssa
    }

    /// The SSA name a block's plain assignment gives `cell`.
    fn def_of(f: &SsaFunction, va: u64, cell: u16) -> u16 {
        f.blocks[&va]
            .stmts
            .iter()
            .find_map(|s| match s {
                Stmt::Assign { dst, .. } if f.names[dst.num as usize].cell == cell => {
                    Some(dst.num)
                }
                _ => None,
            })
            .expect("no assign to the cell in the block")
    }

    /// Rewrite the φ for `name`'s cell at `at` to take `name` on the
    /// edge from `pred` — the shape copy propagation leaves in the real
    /// pipeline (it rewrites a φ argument to its same-cell root). With
    /// the root defined at entry and the paths redefining the cell, the
    /// root interferes with the other arguments, so out-of-SSA must
    /// leave a real copy on that edge: raw construction alone is
    /// conventional (φ-webs coalesce whole, no copies anywhere), and a
    /// refusal fixture has to force the copy to exist.
    fn stale_phi_arg(f: &mut SsaFunction, at: u64, pred: u64, name: u16) {
        let cell = f.names[name as usize].cell;
        let idx = f.blocks[&at]
            .phis
            .iter()
            .position(|p| f.names[p.dst as usize].cell == cell)
            .expect("no phi for the cell at the join");
        let phi = &mut f.blocks.get_mut(&at).unwrap().phis[idx];
        phi.args
            .iter_mut()
            .find(|(k, _)| *k == Some(pred))
            .expect("no phi argument for the predecessor")
            .1 = name;
        assert_eq!(irssa::check(f), Ok(()), "the staled SSA must still check");
    }

    fn no_tables() -> BTreeMap<u64, Vec<u64>> {
        BTreeMap::new()
    }

    /// Structure and insist on the module's promises: the tree checks,
    /// and a second run is byte-identical.
    fn run(f: &SsaFunction, tables: &BTreeMap<u64, Vec<u64>>) -> (Node, StructStats) {
        let (root, stats) = structure(f, tables);
        assert_eq!(check(f, tables, &root), Ok(()), "output must pass check");
        let (again, s2) = structure(f, tables);
        assert_eq!(root, again, "structuring must be deterministic");
        assert_eq!(stats, s2);
        assert_eq!(render(f, &root), render(f, &again));
        (root, stats)
    }

    /// Structure a hand-built lifted function end to end.
    fn tree(f: &irlift::LiftedFunction) -> (SsaFunction, Node, StructStats) {
        let ssa = build(f);
        let (root, stats) = run(&ssa, &no_tables());
        (ssa, root, stats)
    }

    fn text(f: &SsaFunction, root: &Node) -> String {
        render(f, root)
    }

    // -- 1: the schemas in isolation ---------------------------------------

    #[test]
    fn a_straight_line_chain_is_one_sequence() {
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W64), c(1, Width::W64))],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), c(2, Width::W64))],
                    vec![0x1020],
                ),
                block(0x1020, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&f);
        assert_eq!(stats.gotos, 0);
        assert!(!stats.capped);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             block loc_1010\n\
             block loc_1020\n"
        );
    }

    /// entry branches to 0x1020 when the flag is set; the 0x1010 arm is
    /// the fall-through, so the recovered `if` is negated.
    fn if_then(taken_arm: bool) -> irlift::LiftedFunction {
        let (arm, target) = if taken_arm {
            (0x1010, 0x1010)
        } else {
            (0x1010, 0x1020)
        };
        func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![set_flag(1), jcc(target)],
                    vec![target, if taken_arm { 0x1020 } else { 0x1010 }],
                ),
                block(
                    arm,
                    vec![assign(ra(1, Width::W64), c(7, Width::W64))],
                    vec![0x1020],
                ),
                block(0x1020, vec![ret()], vec![]),
            ],
        )
    }

    #[test]
    fn an_if_then_on_the_taken_edge_is_un_negated() {
        let (ssa, root, stats) = tree(&if_then(true));
        assert_eq!(stats.gotos, 0);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             if cond loc_1000\n\
             \x20 block loc_1010\n\
             block loc_1020\n"
        );
    }

    #[test]
    fn an_if_then_on_the_fall_through_edge_is_negated() {
        let (ssa, root, stats) = tree(&if_then(false));
        assert_eq!(stats.gotos, 0);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             if !cond loc_1000\n\
             \x20 block loc_1010\n\
             block loc_1020\n"
        );
    }

    /// The diamond: both arms are single-predecessor and converge.
    fn diamond() -> irlift::LiftedFunction {
        func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1020)], vec![0x1020, 0x1010]),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), c(1, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), c(2, Width::W64))],
                    vec![0x1030],
                ),
                block(0x1030, vec![ret()], vec![]),
            ],
        )
    }

    #[test]
    fn a_diamond_is_an_if_then_else_with_the_taken_arm_first() {
        let (ssa, root, stats) = tree(&diamond());
        assert_eq!(stats.gotos, 0);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             if cond loc_1000\n\
             \x20 block loc_1020\n\
             else\n\
             \x20 block loc_1010\n\
             block loc_1030\n"
        );
    }

    #[test]
    fn a_block_branching_to_itself_is_a_self_loop() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1000)], vec![0x1000, 0x1010]),
                block(0x1010, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&f);
        assert_eq!(stats.gotos, 0);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             self-loop cond loc_1000\n\
             \x20 block loc_1000\n\
             block loc_1010\n"
        );
    }

    /// header tests and exits; the body falls back to the header.
    fn while_loop() -> irlift::LiftedFunction {
        func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W64), c(0, Width::W64))],
                    vec![0x1010],
                ),
                block(0x1010, vec![set_flag(1), jcc(0x1030)], vec![0x1030, 0x1020]),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), c(1, Width::W64)), jmp(0x1010)],
                    vec![0x1010],
                ),
                block(0x1030, vec![ret()], vec![]),
            ],
        )
    }

    #[test]
    fn a_natural_while_puts_the_test_at_the_header() {
        let (ssa, root, stats) = tree(&while_loop());
        assert_eq!(stats.gotos, 0);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             while !cond loc_1010\n\
             \x20 block loc_1010\n\
             \x20 block loc_1020\n\
             block loc_1030\n"
        );
        let Node::Seq(v) = &root else {
            panic!("a sequence")
        };
        let Node::Loop { kind, .. } = &v[1] else {
            panic!("a loop")
        };
        assert_eq!(*kind, LoopKind::While);
    }

    #[test]
    fn a_do_while_puts_the_test_at_the_latch() {
        // body first, latch tests and loops back.
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
                    vec![assign(ra(1, Width::W64), c(1, Width::W64))],
                    vec![0x1020],
                ),
                block(0x1020, vec![set_flag(1), jcc(0x1010)], vec![0x1010, 0x1030]),
                block(0x1030, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&f);
        assert_eq!(stats.gotos, 0);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             do-while cond loc_1020\n\
             \x20 block loc_1010\n\
             \x20 block loc_1020\n\
             block loc_1030\n"
        );
    }

    #[test]
    fn a_break_and_a_continue_land_inside_the_while() {
        // 1010: header test -> exit 1050 or body 1020
        // 1020: test -> break to 1050, else 1030
        // 1030: test -> continue to 1010, else 1040
        // 1040: -> 1010
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W64), c(0, Width::W64))],
                    vec![0x1010],
                ),
                block(0x1010, vec![set_flag(1), jcc(0x1050)], vec![0x1050, 0x1020]),
                block(0x1020, vec![set_flag(2), jcc(0x1050)], vec![0x1050, 0x1030]),
                block(0x1030, vec![set_flag(3), jcc(0x1010)], vec![0x1010, 0x1040]),
                block(0x1040, vec![jmp(0x1010)], vec![0x1010]),
                block(0x1050, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&f);
        assert_eq!(stats.gotos, 0);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             while !cond loc_1010\n\
             \x20 block loc_1010\n\
             \x20 block loc_1020\n\
             \x20 if cond loc_1020\n\
             \x20   break\n\
             \x20 block loc_1030\n\
             \x20 if cond loc_1030\n\
             \x20   continue\n\
             \x20 block loc_1040\n\
             block loc_1050\n"
        );
    }

    /// An indirect jump whose successors a table proves.
    fn switch_function() -> irlift::LiftedFunction {
        func(
            0x1000,
            vec![
                block(0x1000, vec![indirect()], vec![0x1010, 0x1020]),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), c(1, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), c(2, Width::W64))],
                    vec![0x1030],
                ),
                block(0x1030, vec![ret()], vec![]),
            ],
        )
    }

    #[test]
    fn a_proven_jump_table_becomes_a_switch() {
        let ssa = build(&switch_function());
        let tables = BTreeMap::from([(0x1008, vec![0x1010, 0x1020])]);
        let (root, stats) = run(&ssa, &tables);
        assert_eq!(stats.gotos, 0);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             switch loc_1000\n\
             \x20 case loc_1010:\n\
             \x20   block loc_1010\n\
             \x20 case loc_1020:\n\
             \x20   block loc_1020\n\
             block loc_1030\n"
        );
    }

    #[test]
    fn the_same_indirect_jump_without_a_table_is_opaque() {
        let ssa = build(&switch_function());
        let (root, _) = run(&ssa, &no_tables());
        let rendered = text(&ssa, &root);
        assert!(
            rendered.contains("opaque loc_1000 (indirect jump)"),
            "{rendered}"
        );
        // The unreachable-by-structure cases are still covered exactly
        // once, and the undecidable edges are declared, not invented.
        assert!(rendered.contains("block loc_1010"), "{rendered}");
        assert!(rendered.contains("block loc_1020"), "{rendered}");
    }

    // -- 2: nested combinations --------------------------------------------

    #[test]
    fn an_if_nests_inside_a_loop_body() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1040)], vec![0x1040, 0x1010]),
                block(0x1010, vec![set_flag(2), jcc(0x1030)], vec![0x1030, 0x1020]),
                block(0x1020, vec![jmp(0x1030)], vec![0x1030]),
                block(0x1030, vec![jmp(0x1000)], vec![0x1000]),
                block(0x1040, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&f);
        assert_eq!(stats.gotos, 0);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             while !cond loc_1000\n\
             \x20 block loc_1000\n\
             \x20 block loc_1010\n\
             \x20 if !cond loc_1010\n\
             \x20   block loc_1020\n\
             \x20 block loc_1030\n\
             block loc_1040\n"
        );
    }

    #[test]
    fn a_loop_nests_inside_an_if_else() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1020)], vec![0x1020, 0x1010]),
                // then arm: a self loop
                block(0x1010, vec![set_flag(2), jcc(0x1010)], vec![0x1010, 0x1030]),
                // else arm: straight line
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), c(9, Width::W64))],
                    vec![0x1030],
                ),
                block(0x1030, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&f);
        assert_eq!(stats.gotos, 0);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             if cond loc_1000\n\
             \x20 block loc_1020\n\
             else\n\
             \x20 self-loop cond loc_1010\n\
             \x20   block loc_1010\n\
             block loc_1030\n"
        );
    }

    #[test]
    fn two_diamonds_in_a_row_are_one_flat_sequence() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1020)], vec![0x1020, 0x1010]),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), c(1, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), c(2, Width::W64))],
                    vec![0x1030],
                ),
                block(0x1030, vec![set_flag(2), jcc(0x1050)], vec![0x1050, 0x1040]),
                block(
                    0x1040,
                    vec![assign(ra(2, Width::W64), c(3, Width::W64))],
                    vec![0x1060],
                ),
                block(
                    0x1050,
                    vec![assign(ra(2, Width::W64), c(4, Width::W64))],
                    vec![0x1060],
                ),
                block(0x1060, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&f);
        assert_eq!(stats.gotos, 0);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             if cond loc_1000\n\
             \x20 block loc_1020\n\
             else\n\
             \x20 block loc_1010\n\
             block loc_1030\n\
             if cond loc_1030\n\
             \x20 block loc_1050\n\
             else\n\
             \x20 block loc_1040\n\
             block loc_1060\n"
        );
    }

    // -- 3: irreducible ----------------------------------------------------

    /// A two-entry loop: 0x1010 and 0x1020 branch into each other and the
    /// entry can reach either, so no header dominates the cycle. The
    /// cycle blocks are *effectful* (a store each), so no inversion —
    /// jump threading included — may duplicate them and the virtualized
    /// edge honestly stays a goto.
    fn irreducible() -> irlift::LiftedFunction {
        func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1020)], vec![0x1020, 0x1010]),
                block(
                    0x1010,
                    vec![store(), set_flag(2), jcc(0x1020)],
                    vec![0x1020, 0x1030],
                ),
                block(
                    0x1020,
                    vec![store(), set_flag(3), jcc(0x1010)],
                    vec![0x1010, 0x1030],
                ),
                block(0x1030, vec![ret()], vec![]),
            ],
        )
    }

    #[test]
    fn an_irreducible_two_entry_loop_structures_with_gotos() {
        let (ssa, root, stats) = tree(&irreducible());
        assert!(stats.gotos >= 1, "an irreducible cycle needs a goto");
        assert!(!stats.capped);
        let rendered = text(&ssa, &root);
        assert!(rendered.contains("goto loc_"), "{rendered}");
        // Deterministic down to the byte across a fresh run.
        let (again, s2) = structure(&ssa, &no_tables());
        assert_eq!(rendered, render(&ssa, &again));
        assert_eq!(stats, s2);
    }

    // -- 4, 5, 6: honest markers -------------------------------------------

    #[test]
    fn a_truncated_block_is_opaque_in_place() {
        let mut f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1020)], vec![0x1020, 0x1010]),
                block(0x1010, vec![], vec![0x1020]),
                block(0x1020, vec![ret()], vec![]),
            ],
        );
        f.blocks.get_mut(&0x1010).expect("the arm").truncated = true;
        let (ssa, root, _) = tree(&f);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             if !cond loc_1000\n\
             \x20 opaque loc_1010 (truncated)\n\
             block loc_1020\n"
        );
    }

    #[test]
    fn an_out_of_function_edge_becomes_an_external_goto() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x9000)], vec![0x9000, 0x1010]),
                block(0x1010, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&f);
        assert_eq!(stats.gotos, 0, "a tail jump is not a structuring edge");
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             if cond loc_1000\n\
             \x20 goto loc_9000\n\
             block loc_1010\n"
        );
    }

    #[test]
    fn a_block_with_undecidable_successors_is_opaque_and_declares_its_edges() {
        // Two successors, no branch at all: nothing can decide between
        // them, so the edges are declared unrealized.
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
        let (ssa, root, _) = tree(&f);
        let rendered = text(&ssa, &root);
        assert!(
            rendered.contains("opaque loc_1000 (undecidable exits)"),
            "{rendered}"
        );
        assert!(rendered.contains("block loc_1010"), "{rendered}");
        assert!(rendered.contains("block loc_1020"), "{rendered}");
    }

    // -- 7: the cap --------------------------------------------------------

    #[test]
    fn a_forced_cap_degrades_to_gotos_and_still_checks() {
        let ssa = build(&diamond());
        // No rounds at all: every block stands alone, every edge a goto.
        let (bare, stats) = structure_budgeted(&ssa, &no_tables(), 0);
        assert!(stats.capped, "the budget was zero rounds");
        assert_eq!(check(&ssa, &no_tables(), &bare), Ok(()));
        assert_eq!(
            render(&ssa, &bare),
            render(&ssa, &degenerate(&ssa, &no_tables()))
        );
        // One round: the diamond collapses, the follow edge degrades.
        let (root, stats) = structure_budgeted(&ssa, &no_tables(), 1);
        assert!(stats.capped, "the budget was one round");
        assert_eq!(check(&ssa, &no_tables(), &root), Ok(()));
        assert_eq!(
            render(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             if cond loc_1000\n\
             \x20 block loc_1020\n\
             else\n\
             \x20 block loc_1010\n\
             goto loc_1030\n\
             block loc_1030\n"
        );
        // The real budget never fires on the same input.
        let (_, real) = run(&ssa, &no_tables());
        assert!(!real.capped);
    }

    // -- 8: seeded sweep ---------------------------------------------------

    /// xorshift64* with a fixed seed: deterministic, no wall clock.
    fn next(s: &mut u64) -> u64 {
        *s ^= *s >> 12;
        *s ^= *s << 25;
        *s ^= *s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A deterministic stream of small random CFGs — the same harness
    /// shape `irssaopt`'s sweeps use, with branch statements and edge
    /// counts that disagree often enough to exercise every honest marker.
    fn random_functions(count: usize, seed: u64) -> Vec<irlift::LiftedFunction> {
        let mut s = seed;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let nblocks = 2 + next(&mut s) % 14;
            let vas: Vec<u64> = (0..nblocks).map(|i| 0x1000 + 0x10 * i).collect();
            let mut blocks = Vec::new();
            for (i, &va) in vas.iter().enumerate() {
                let nsucc = (next(&mut s) % 3) as usize;
                let succ: Vec<u64> = (0..nsucc)
                    .map(|_| vas[(next(&mut s) % nblocks) as usize])
                    .collect();
                let mut list = vec![assign(
                    ra((next(&mut s) % 4) as u16, Width::W64),
                    c(next(&mut s) % 16, Width::W64),
                )];
                // Give most two-way blocks a real conditional terminator,
                // and leave the rest undecidable on purpose.
                match next(&mut s) % 4 {
                    0 => list.push(ret()),
                    1 => list.push(indirect()),
                    _ if !succ.is_empty() => {
                        let t = succ[(next(&mut s) as usize) % succ.len()];
                        list.push(if succ.len() > 1 { set_flag(1) } else { jmp(t) });
                        if succ.len() > 1 {
                            list.push(jcc(t));
                        }
                    }
                    _ => {}
                }
                let mut b = block(va, list, succ);
                b.truncated = i > 0 && next(&mut s).is_multiple_of(16);
                blocks.push(b);
            }
            out.push(func(0x1000, blocks));
        }
        out
    }

    #[test]
    fn sweep_random_small_cfgs_always_structure_and_check() {
        let mut with_gotos = 0usize;
        for f in random_functions(400, 0x5EED_1A5E_0DDB_5BAD) {
            let Ok(ssa) = irssa::construct(&f) else {
                continue;
            };
            if irssa::check(&ssa).is_err() {
                continue;
            }
            let (root, stats) = structure(&ssa, &no_tables());
            assert_eq!(
                check(&ssa, &no_tables(), &root),
                Ok(()),
                "check must hold on {:#x}",
                ssa.entry
            );
            assert!(!stats.capped, "the cap is unreachable on small input");
            if stats.gotos > 0 {
                with_gotos += 1;
            }
            // Byte-determinism.
            let (again, s2) = structure(&ssa, &no_tables());
            assert_eq!(root, again);
            assert_eq!(stats, s2);
            assert_eq!(render(&ssa, &root), render(&ssa, &again));
        }
        assert!(with_gotos > 0, "the corpus must exercise virtualization");
    }

    // -- 9: malformed input ------------------------------------------------

    #[test]
    fn malformed_input_gets_the_degenerate_tree_and_zeroed_stats() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1010)], vec![0x1010, 0x1020]),
                block(0x1010, vec![], vec![0x1020]),
                block(0x1020, vec![ret()], vec![]),
            ],
        );
        let mut ssa = build(&f);
        // Break the SSA: a use naming no name.
        ssa.blocks
            .get_mut(&0x1010)
            .expect("the arm")
            .stmts
            .push(assign(
                Reg {
                    num: 9999,
                    ..ra(0, Width::W64)
                },
                c(1, Width::W64),
            ));
        assert!(irssa::check(&ssa).is_err(), "the input is malformed");
        let (root, stats) = structure(&ssa, &no_tables());
        assert_eq!(stats, StructStats::default(), "stats are zeroed");
        assert_eq!(check(&ssa, &no_tables(), &root), Ok(()));
        assert_eq!(
            render(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             if cond loc_1000\n\
             \x20 goto loc_1010\n\
             goto loc_1020\n\
             block loc_1010\n\
             goto loc_1020\n\
             block loc_1020\n"
        );
    }

    // -- 10: the companion catches what the pass must not do ---------------

    #[test]
    fn check_rejects_a_duplicated_block() {
        // Duplication stays banned outside the sanctioned re-splits. An
        // arm like 0x1010, whose one in-function edge reaches the φ-free
        // join, is sanctioned as a copy-safe tail; the pure two-way head
        // 0x1000 is sanctioned as a threadable deciding block — but only
        // *deciding*: a trailing bare copy never spells its branch, so
        // the condition-honesty rule faults it. An effectful head is no
        // duplicate of any kind.
        let ssa = build(&diamond());
        let arm_if = || Node::If {
            cond: Cond {
                block: 0x1000,
                negated: false,
            },
            then_body: Box::new(Node::Block(0x1020)),
            else_body: Some(Box::new(Node::Block(0x1010))),
        };
        let dup_tail = Node::Seq(vec![
            Node::Block(0x1000),
            arm_if(),
            Node::Block(0x1030),
            Node::Block(0x1010),
            Node::Goto(0x1030),
        ]);
        assert_eq!(check(&ssa, &no_tables(), &dup_tail), Ok(()));
        let undecided_copy = Node::Seq(vec![
            Node::Block(0x1000),
            arm_if(),
            Node::Block(0x1030),
            Node::Block(0x1000),
        ]);
        assert!(matches!(
            check(&ssa, &no_tables(), &undecided_copy),
            Err(StructFault::Undecided { block: 0x1000 })
        ));
        // The same shape on an effectful head: not sanctioned at all.
        let mut f = diamond();
        f.blocks
            .get_mut(&0x1000)
            .expect("the head")
            .stmts
            .insert(0, store());
        let effectful = build(&f);
        let bad = Node::Seq(vec![
            Node::Block(0x1000),
            arm_if(),
            Node::Block(0x1030),
            Node::Block(0x1000),
            arm_if(),
        ]);
        assert!(matches!(
            check(&effectful, &no_tables(), &bad),
            Err(StructFault::Duplicated { block: 0x1000 })
        ));
    }

    #[test]
    fn check_rejects_an_invented_fall_through_out_of_an_opaque_block() {
        // 0x1000 has two undecidable successors; sequencing 0x1010 after
        // it claims an edge the tree may not realize.
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
        let ssa = build(&f);
        let bad = Node::Seq(vec![
            Node::Opaque {
                block: 0x1000,
                reason: OpaqueReason::Unstructurable,
            },
            Node::Block(0x1010),
            Node::Block(0x1020),
        ]);
        // The opaque leaves no pending edge, so the tree invents none —
        // but a `Block` in its place would claim the terminator.
        assert_eq!(check(&ssa, &no_tables(), &bad), Ok(()));
        let worse = Node::Seq(vec![
            Node::Block(0x1000),
            Node::Block(0x1010),
            Node::Block(0x1020),
        ]);
        assert!(matches!(
            check(&ssa, &no_tables(), &worse),
            Err(StructFault::Opacity { block: 0x1000 })
        ));
    }

    #[test]
    fn check_rejects_a_condition_that_names_an_unconditional_block() {
        let ssa = build(&diamond());
        let bad = Node::Seq(vec![
            Node::Block(0x1000),
            Node::If {
                cond: Cond {
                    block: 0x1030,
                    negated: false,
                },
                then_body: Box::new(Node::Block(0x1020)),
                else_body: Some(Box::new(Node::Block(0x1010))),
            },
            Node::Block(0x1030),
        ]);
        assert!(matches!(
            check(&ssa, &no_tables(), &bad),
            Err(StructFault::CondMisplaced { .. }) | Err(StructFault::NotConditional { .. })
        ));
    }

    #[test]
    fn check_rejects_a_flipped_polarity() {
        let ssa = build(&diamond());
        let bad = Node::Seq(vec![
            Node::Block(0x1000),
            Node::If {
                cond: Cond {
                    block: 0x1000,
                    negated: true,
                },
                then_body: Box::new(Node::Block(0x1020)),
                else_body: Some(Box::new(Node::Block(0x1010))),
            },
            Node::Block(0x1030),
        ]);
        assert!(matches!(
            check(&ssa, &no_tables(), &bad),
            Err(StructFault::Polarity { block: 0x1000 })
        ));
    }

    #[test]
    fn check_rejects_a_dropped_edge() {
        let ssa = build(&diamond());
        let bad = Node::Seq(vec![
            Node::Block(0x1000),
            Node::If {
                cond: Cond {
                    block: 0x1000,
                    negated: false,
                },
                then_body: Box::new(Node::Block(0x1020)),
                else_body: None,
            },
            Node::Block(0x1010),
            Node::Block(0x1030),
        ]);
        // 0x1020 falls to 0x1030 but the tree sequences 0x1010 after it.
        assert!(check(&ssa, &no_tables(), &bad).is_err());
    }

    #[test]
    fn the_trait_seam_and_the_free_function_agree() {
        let ssa = build(&diamond());
        let direct = structure(&ssa, &no_tables());
        let through = Phoenix.structure(&ssa, &no_tables());
        assert_eq!(direct, through);
    }

    #[test]
    fn an_empty_function_structures_to_an_empty_tree() {
        let f = func(0x1000, vec![]);
        let ssa = irssa::construct(&f).expect("an empty function constructs");
        let (root, stats) = structure(&ssa, &no_tables());
        assert_eq!(root, Node::Seq(Vec::new()));
        assert_eq!(stats, StructStats::default());
        assert_eq!(check(&ssa, &no_tables(), &root), Ok(()));
    }

    // -- 11: the SAILR tail re-split ---------------------------------------

    /// The collapse alone, with [`structure`]'s own budget — the
    /// zero-duplication baseline the re-split is measured against.
    fn raw(ssa: &SsaFunction, tables: &BTreeMap<u64, Vec<u64>>) -> (Node, StructStats) {
        structure_raw(ssa, tables, default_budget(ssa))
    }

    /// The tail-merged diamond cross-jumping leaves behind: two arms
    /// cross into a shared `ret` tail (and a second exit), so no acyclic
    /// schema fits until an edge is realized. The entry sits at the
    /// highest VA, making the merged tail's own edge the lowest — the
    /// exact edge the collapse would virtualize.
    fn merged_tail(tail: Vec<Stmt>, tail_succs: Vec<u64>) -> irlift::LiftedFunction {
        func(
            0x1040,
            vec![
                block(0x1040, vec![set_flag(1), jcc(0x1000)], vec![0x1000, 0x1010]),
                block(0x1000, vec![set_flag(2), jcc(0x1020)], vec![0x1020, 0x1030]),
                block(0x1010, vec![set_flag(3), jcc(0x1020)], vec![0x1020, 0x1030]),
                block(0x1020, tail, tail_succs),
                block(0x1030, vec![ret()], vec![]),
            ],
        )
    }

    #[test]
    fn a_tail_merged_diamond_resplits_instead_of_a_goto() {
        let (ssa, root, stats) = tree(&merged_tail(vec![ret()], vec![]));
        // The collapse alone spends exactly one goto on the merged tail.
        let (bare, bstats) = raw(&ssa, &no_tables());
        assert_eq!(bstats.gotos, 1);
        assert!(render(&ssa, &bare).contains("goto loc_1020"));
        // The re-split buys it back with one duplicate leaf.
        assert_eq!((stats.gotos, stats.duplications), (0, 1));
        assert!(!stats.dup_capped);
        let rendered = text(&ssa, &root);
        assert!(!rendered.contains("goto"), "{rendered}");
        // Both occurrences render the block's own statements — the leaf
        // stores the VA, so the duplicate cannot drift from its source.
        assert_eq!(rendered.matches("block loc_1020").count(), 2, "{rendered}");
    }

    #[test]
    fn a_merged_tail_jump_duplicates_with_its_external_goto() {
        let (ssa, root, stats) = tree(&merged_tail(vec![jmp(0x9000)], vec![0x9000]));
        assert_eq!((stats.gotos, stats.duplications), (0, 1));
        let rendered = text(&ssa, &root);
        assert!(!rendered.contains("goto loc_1020"), "{rendered}");
        assert_eq!(rendered.matches("block loc_1020").count(), 2, "{rendered}");
        assert_eq!(
            rendered.matches("goto loc_9000").count(),
            2,
            "the external goto travels with the duplicated tail: {rendered}"
        );
    }

    #[test]
    fn a_genuine_irreducible_goto_survives_with_zero_duplications() {
        // The irreducible cycle's goto targets carry in-function edges
        // *and* effectful statements: no copy-safe tail, no threadable
        // head, nothing to re-split, and the output is the collapse bit
        // for bit.
        let (ssa, root, stats) = tree(&irreducible());
        assert_eq!(stats.duplications, 0);
        assert!(!stats.dup_capped);
        let (bare, bstats) = raw(&ssa, &no_tables());
        assert_eq!(root, bare);
        assert_eq!(stats, bstats);
    }

    #[test]
    fn a_shared_tail_converging_in_function_resplits_as_a_fall_through() {
        // Inversion two: 0x1020 flows on to the φ-free join 0x1030
        // inside the function. The site's fall-through consumer *is*
        // that join, so the cheapest duplicate — the one leaf, no
        // trailing jump — realizes the open edge by falling into it.
        let (ssa, root, stats) = tree(&merged_tail(vec![jmp(0x1030)], vec![0x1030]));
        let (bare, bstats) = raw(&ssa, &no_tables());
        assert_eq!(bstats.gotos, 1);
        assert!(text(&ssa, &bare).contains("goto loc_1020"));
        assert_eq!((stats.gotos, stats.duplications), (0, 1));
        assert!(!stats.dup_capped);
        let rendered = text(&ssa, &root);
        assert!(!rendered.contains("goto"), "{rendered}");
        assert_eq!(rendered.matches("block loc_1020").count(), 2, "{rendered}");
        assert_eq!(
            rendered.matches("block loc_1030").count(),
            1,
            "the join itself is not duplicated: {rendered}"
        );
    }

    #[test]
    fn an_effectful_condition_carrying_tail_keeps_its_goto() {
        // The tail's two live edges have no linear duplicate, so the
        // first two inversions refuse; jump threading would take the
        // condition — but the store makes the block effectful, so it
        // refuses too, and the output is the collapse bit for bit.
        let mut f = merged_tail(vec![store(), set_flag(7), jcc(0x1030)], vec![0x1030, 0x1050]);
        f.blocks.insert(0x1050, block(0x1050, vec![ret()], vec![]));
        let (ssa, root, stats) = tree(&f);
        assert!(threadable_head(&ssa, &no_tables(), &copy_edges(&ssa), 0x1020).is_none());
        assert_eq!((stats.duplications, stats.threaded), (0, 0));
        assert!(!stats.dup_capped, "an ineligible tail is no candidate");
        let (bare, bstats) = raw(&ssa, &no_tables());
        assert_eq!(root, bare);
        assert_eq!(stats, bstats);
        assert!(text(&ssa, &root).contains("goto loc_1020"));
    }

    #[test]
    fn a_tail_whose_convergence_edge_carries_copies_keeps_its_goto() {
        // The join 0x1030 holds a φ over the arms' disagreeing r1, and
        // the tail's edge is staled to feed the entry's r1 — which is
        // live through both arms' redefinitions, so it interferes and
        // out-of-SSA must place a real copy on that edge. Copies have
        // exactly one textual placement. Refused.
        let mut f = merged_tail(
            vec![assign(ra(2, Width::W64), c(3, Width::W64)), jmp(0x1030)],
            vec![0x1030],
        );
        let arm = |va: u64, n: u64| {
            block(
                va,
                vec![
                    assign(ra(1, Width::W64), c(n, Width::W64)),
                    set_flag(n),
                    jcc(0x1020),
                ],
                vec![0x1020, 0x1030],
            )
        };
        f.blocks.insert(
            0x1040,
            block(
                0x1040,
                vec![assign(ra(1, Width::W64), c(9, Width::W64)), set_flag(1), jcc(0x1000)],
                vec![0x1000, 0x1010],
            ),
        );
        f.blocks.insert(0x1000, arm(0x1000, 1));
        f.blocks.insert(0x1010, arm(0x1010, 2));
        f.blocks.insert(
            0x1030,
            block(
                0x1030,
                vec![assign(ra(4, Width::W64), read(ra(1, Width::W64))), ret()],
                vec![],
            ),
        );
        let mut ssa = build(&f);
        let stale = def_of(&ssa, 0x1040, 1);
        stale_phi_arg(&mut ssa, 0x1030, 0x1020, stale);
        assert!(
            copy_edges(&ssa).contains(&(0x1020, 0x1030)),
            "the tail's edge must carry a real out-of-SSA copy for the fixture to bite"
        );
        let (root, stats) = run(&ssa, &no_tables());
        assert_eq!(stats.duplications, 0);
        assert!(!stats.dup_capped);
        let (bare, bstats) = raw(&ssa, &no_tables());
        assert_eq!(root, bare);
        assert_eq!(stats, bstats);
        assert!(text(&ssa, &root).contains("goto loc_1020"));
    }

    /// A dispatch whose cases share a tail: A and B cross into T, C goes
    /// its own way, so `try_switch`'s convergence test refuses and the
    /// whole case set degrades to gotos — the shape bash's switches
    /// leave behind.
    fn shared_case_tail() -> (SsaFunction, BTreeMap<u64, Vec<u64>>) {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![indirect()], vec![0x1010, 0x1020, 0x1030]),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), c(1, Width::W64))],
                    vec![0x1040],
                ),
                block(
                    0x1020,
                    vec![assign(ra(2, Width::W64), c(2, Width::W64))],
                    vec![0x1040],
                ),
                block(
                    0x1030,
                    vec![assign(ra(3, Width::W64), c(3, Width::W64))],
                    vec![0x1050],
                ),
                block(
                    0x1040,
                    vec![assign(ra(4, Width::W64), c(4, Width::W64))],
                    vec![0x1050],
                ),
                block(0x1050, vec![ret()], vec![]),
            ],
        );
        let tables = BTreeMap::from([(0x1008u64, vec![0x1010u64, 0x1020, 0x1030])]);
        (build(&f), tables)
    }

    #[test]
    fn shared_case_tails_structure_the_switch_across_resplit_rounds() {
        // Round one duplicates the epilogue; each round's rewrites make
        // the next round's chain interiors label-free, until every case
        // body is spelled inside its case and no goto remains.
        let (ssa, tables) = shared_case_tail();
        let (root, stats) = run(&ssa, &tables);
        let (bare, bstats) = raw(&ssa, &tables);
        assert!(bstats.gotos >= 5, "the degraded dispatch keeps its gotos");
        assert!(render(&ssa, &bare).contains("case loc_1010:\n    goto loc_1010"));
        assert_eq!(stats.gotos, 0, "every goto bought back");
        assert!(!stats.dup_capped);
        assert_eq!(stats.duplications, 11);
        let rendered = text(&ssa, &root);
        assert!(!rendered.contains("goto"), "{rendered}");
        // Each case spells its body, the shared tail inline where the
        // flow passes it, ending in the duplicated epilogue.
        assert!(
            rendered.contains(
                "  case loc_1010:\n    block loc_1010\n    block loc_1040\n    block loc_1050\n"
            ),
            "{rendered}"
        );
        assert!(
            rendered.contains(
                "  case loc_1020:\n    block loc_1020\n    block loc_1040\n    block loc_1050\n"
            ),
            "{rendered}"
        );
        assert!(
            rendered.contains("  case loc_1030:\n    block loc_1030\n    block loc_1050\n"),
            "{rendered}"
        );
    }

    #[test]
    fn a_guard_edge_into_the_default_body_resplits() {
        // The bounds-check guard jumps straight into the default case
        // body, giving it a second predecessor; the dispatch still
        // proves, and the guard's virtualized edge is bought back by
        // duplicating the default body into it.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1030)], vec![0x1030, 0x1010]),
                block(0x1010, vec![indirect()], vec![0x1020, 0x1030]),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), c(1, Width::W64))],
                    vec![0x1040],
                ),
                block(
                    0x1030,
                    vec![assign(ra(2, Width::W64), c(2, Width::W64))],
                    vec![0x1040],
                ),
                block(0x1040, vec![ret()], vec![]),
            ],
        );
        let ssa = build(&f);
        let tables = BTreeMap::from([(0x1018u64, vec![0x1020u64, 0x1030])]);
        let (root, stats) = run(&ssa, &tables);
        let (bare, bstats) = raw(&ssa, &tables);
        assert!(render(&ssa, &bare).contains("goto loc_1030"), "the guard edge degrades");
        assert!(stats.gotos < bstats.gotos);
        assert!(stats.duplications > 0);
        let rendered = text(&ssa, &root);
        assert!(!rendered.contains("goto loc_1030"), "{rendered}");
        assert!(rendered.contains("switch loc_1010"), "{rendered}");
    }

    #[test]
    fn a_tail_converging_on_the_loop_header_resplits_as_a_continue() {
        // Two arms cross into the shared latch tail T (0x1020) and the
        // pass-through latch R (0x1030), both converging on the header:
        // the collapse virtualizes the cross edges, and the re-split
        // spells the duplicates' open edge as `continue`.
        let f = func(
            0x1100,
            vec![
                block(0x1100, vec![set_flag(1), jcc(0x2000)], vec![0x2000, 0x1040]),
                block(0x1040, vec![set_flag(2), jcc(0x1000)], vec![0x1000, 0x1010]),
                block(0x1000, vec![set_flag(3), jcc(0x1020)], vec![0x1020, 0x1030]),
                block(0x1010, vec![set_flag(4), jcc(0x1020)], vec![0x1020, 0x1030]),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), c(1, Width::W64)), jmp(0x1100)],
                    vec![0x1100],
                ),
                block(
                    0x1030,
                    vec![assign(ra(2, Width::W64), c(2, Width::W64)), jmp(0x1100)],
                    vec![0x1100],
                ),
                block(0x2000, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&f);
        let (bare, bstats) = raw(&ssa, &no_tables());
        assert!(render(&ssa, &bare).contains("goto loc_1020"), "the cross edge degrades");
        // The header's φs are the degenerate loop-carried self-arguments
        // — provably copy-free, so they must not block the split.
        assert!(!ssa.blocks[&0x1100].phis.is_empty());
        assert!(stats.gotos < bstats.gotos);
        assert!(stats.duplications > 0);
        let rendered = text(&ssa, &root);
        assert!(!rendered.contains("goto loc_1020"), "{rendered}");
        assert!(rendered.contains("continue"), "{rendered}");
    }

    #[test]
    fn a_tail_converging_on_the_loop_follow_resplits_as_a_break() {
        // A conditional early exit through the tail T (0x1010), which
        // flows to the loop's own follow: the duplicate spells the open
        // edge as `break`. Entry sits high so the (A -> T) edge is the
        // lowest stuck edge, the order that lets the loop schema fire
        // after its virtualization.
        let f = func(
            0x1400,
            vec![
                block(
                    0x1400,
                    vec![assign(ra(0, Width::W64), c(0, Width::W64)), jmp(0x1100)],
                    vec![0x1100],
                ),
                block(0x1100, vec![set_flag(1), jcc(0x1300)], vec![0x1300, 0x1000]),
                block(0x1000, vec![set_flag(2), jcc(0x1010)], vec![0x1010, 0x1200]),
                block(
                    0x1200,
                    vec![assign(ra(1, Width::W64), c(1, Width::W64)), jmp(0x1100)],
                    vec![0x1100],
                ),
                block(
                    0x1010,
                    vec![assign(ra(2, Width::W64), c(2, Width::W64)), jmp(0x1300)],
                    vec![0x1300],
                ),
                block(0x1300, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&f);
        let (bare, bstats) = raw(&ssa, &no_tables());
        assert!(render(&ssa, &bare).contains("goto loc_1010"), "{}", render(&ssa, &bare));
        assert!(stats.gotos < bstats.gotos);
        assert!(stats.duplications > 0);
        let rendered = text(&ssa, &root);
        assert!(!rendered.contains("goto loc_1010"), "{rendered}");
        assert!(rendered.contains("break"), "{rendered}");
    }

    #[test]
    fn a_chain_that_does_not_fit_the_budget_degrades_to_its_gotos() {
        // Chains are leaf-costed against the one shared cap: with room
        // for three leaves, the epilogue and the first two-leaf chain
        // fit, the remaining targets set `dup_capped` and keep their
        // gotos — degrade, never refuse, and `check` still holds.
        let (ssa, tables) = shared_case_tail();
        let (bare, bstats) = raw(&ssa, &tables);
        let mut stats = bstats;
        let root = resplit_tails(&ssa, &tables, &copy_edges(&ssa), bare, &mut stats, 3);
        assert!(stats.dup_capped, "the corpus must overrun the small cap");
        assert!(stats.duplications <= 3);
        assert!(stats.gotos > 0, "the remainder keeps its gotos");
        assert_eq!(check(&ssa, &tables, &root), Ok(()));
        assert!(render(&ssa, &root).contains("goto loc_"));
    }

    #[test]
    fn a_chain_past_the_bound_keeps_its_goto() {
        // c1 -> c2 -> c3 -> c4 -> ret is four leaves deep: the chain
        // stops at MAX_TAIL_CHAIN with an open edge no site can spell,
        // so the target honestly keeps its gotos. One block shorter,
        // the chain closes and the same shape splits.
        let deep = func(
            0x1100,
            vec![
                block(0x1100, vec![set_flag(1), jcc(0x1000)], vec![0x1000, 0x1010]),
                block(0x1000, vec![set_flag(2), jcc(0x1020)], vec![0x1020, 0x1060]),
                block(0x1010, vec![set_flag(3), jcc(0x1020)], vec![0x1020, 0x1060]),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), c(1, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(2, Width::W64), c(2, Width::W64))],
                    vec![0x1040],
                ),
                block(
                    0x1040,
                    vec![assign(ra(3, Width::W64), c(3, Width::W64))],
                    vec![0x1050],
                ),
                block(0x1050, vec![ret()], vec![]),
                block(0x1060, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&deep);
        assert_eq!(stats.duplications, 0, "{}", text(&ssa, &root));
        assert!(!stats.dup_capped, "the bound is not the budget");
        assert!(text(&ssa, &root).contains("goto loc_1020"));

        let mut shallow = deep.clone();
        shallow.blocks.remove(&0x1040);
        shallow.blocks.get_mut(&0x1030).expect("c2").successors = vec![0x1050];
        shallow.blocks.get_mut(&0x1030).expect("c2").stmts =
            vec![assign(ra(2, Width::W64), c(2, Width::W64)), jmp(0x1050)];
        let (ssa, root, stats) = tree(&shallow);
        assert!(stats.duplications > 0, "{}", text(&ssa, &root));
        assert!(!text(&ssa, &root).contains("goto loc_1020"));
    }

    #[test]
    fn a_truncated_tail_keeps_its_goto() {
        let mut f = merged_tail(vec![], vec![]);
        f.blocks.get_mut(&0x1020).expect("the tail").truncated = true;
        let (ssa, root, stats) = tree(&f);
        assert_eq!(stats.duplications, 0);
        let rendered = text(&ssa, &root);
        assert!(rendered.contains("goto loc_1020"), "{rendered}");
        assert!(rendered.contains("opaque loc_1020 (truncated)"), "{rendered}");
    }

    #[test]
    fn a_target_that_does_not_fit_keeps_all_its_gotos() {
        // Two gotos into one `ret` tail, hand-realized: the re-split is
        // all-or-nothing per target, so a cap of one skips the target
        // whole — a partial split would leave a goto into a duplicated
        // (and so twice-labeled) tail.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1030)], vec![0x1030, 0x1010]),
                block(0x1010, vec![set_flag(2), jcc(0x1030)], vec![0x1030, 0x1020]),
                block(0x1020, vec![ret()], vec![]),
                block(0x1030, vec![ret()], vec![]),
            ],
        );
        let ssa = build(&f);
        let two_gotos = Node::Seq(vec![
            Node::Block(0x1000),
            Node::If {
                cond: Cond {
                    block: 0x1000,
                    negated: false,
                },
                then_body: Box::new(Node::Goto(0x1030)),
                else_body: None,
            },
            Node::Block(0x1010),
            Node::If {
                cond: Cond {
                    block: 0x1010,
                    negated: false,
                },
                then_body: Box::new(Node::Goto(0x1030)),
                else_body: None,
            },
            Node::Block(0x1020),
            Node::Block(0x1030),
        ]);
        assert_eq!(check(&ssa, &no_tables(), &two_gotos), Ok(()));
        let mut stats = StructStats {
            gotos: 2,
            ..Default::default()
        };
        let same = resplit_tails(&ssa, &no_tables(), &copy_edges(&ssa), two_gotos.clone(), &mut stats, 1);
        assert_eq!(same, two_gotos);
        assert!(stats.dup_capped);
        assert_eq!((stats.duplications, stats.gotos), (0, 2));
        // A cap of two holds the whole target: no goto — and so no goto
        // label — remains.
        let mut stats = StructStats {
            gotos: 2,
            ..Default::default()
        };
        let split = resplit_tails(&ssa, &no_tables(), &copy_edges(&ssa), two_gotos, &mut stats, 2);
        assert!(!stats.dup_capped);
        assert_eq!((stats.duplications, stats.gotos), (2, 0));
        assert_eq!(check(&ssa, &no_tables(), &split), Ok(()));
        assert!(!render(&ssa, &split).contains("goto"));
    }

    /// `count` crossing gadgets behind an if-chain dispatcher: every
    /// gadget leaves gotos into its two `ret` tails, so the corpus holds
    /// well over [`MAX_TAIL_SPLITS`] re-split candidates.
    fn gadget_corpus(count: u64) -> irlift::LiftedFunction {
        let mut blocks = Vec::new();
        for i in 0..count - 1 {
            let va = 0x1000 + 0x10 * i;
            let gadget = 0x2000 + 0x100 * i;
            let next = if i + 2 == count {
                0x2000 + 0x100 * (i + 1)
            } else {
                va + 0x10
            };
            blocks.push(block(va, vec![set_flag(i), jcc(gadget)], vec![gadget, next]));
        }
        for i in 0..count {
            let base = 0x2000 + 0x100 * i;
            blocks.push(block(base, vec![set_flag(1), jcc(base + 0x10)], vec![base + 0x10, base + 0x20]));
            blocks.push(block(base + 0x10, vec![set_flag(2), jcc(base + 0x30)], vec![base + 0x30, base + 0x40]));
            blocks.push(block(base + 0x20, vec![set_flag(3), jcc(base + 0x30)], vec![base + 0x30, base + 0x40]));
            blocks.push(block(base + 0x30, vec![ret()], vec![]));
            blocks.push(block(base + 0x40, vec![ret()], vec![]));
        }
        func(0x1000, blocks)
    }

    #[test]
    fn a_corpus_past_the_cap_degrades_to_gotos_and_still_checks() {
        let ssa = build(&gadget_corpus(24));
        let (root, stats) = run(&ssa, &no_tables());
        assert!(stats.dup_capped, "the corpus must overrun the cap");
        assert!(stats.duplications > 0 && stats.duplications <= MAX_TAIL_SPLITS);
        assert!(!stats.capped);
        // Every duplicate bought back exactly one goto; the remainder
        // keeps its gotos rather than being refused.
        let (_, bstats) = raw(&ssa, &no_tables());
        assert_eq!(stats.gotos, bstats.gotos - stats.duplications);
        assert!(text(&ssa, &root).contains("goto loc_"));
    }

    #[test]
    fn the_resplit_is_scoped_zero_duplications_is_the_collapse_bit_for_bit() {
        let mut resplit_some = 0usize;
        for f in random_functions(400, 0x5EED_1A5E_0DDB_5BAD) {
            let Ok(ssa) = irssa::construct(&f) else {
                continue;
            };
            if irssa::check(&ssa).is_err() {
                continue;
            }
            let (root, stats) = structure(&ssa, &no_tables());
            let (bare, bstats) = raw(&ssa, &no_tables());
            // Monotone: the re-split only ever buys gotos back, and a
            // chain spends at least one duplicate leaf per bought goto.
            assert!(stats.gotos <= bstats.gotos);
            assert!(bstats.gotos - stats.gotos <= stats.duplications);
            assert!(stats.threaded <= stats.duplications);
            if stats.duplications == 0 {
                assert_eq!(root, bare, "zero duplications must be the collapse bit for bit");
                assert_eq!(stats, bstats);
            } else {
                resplit_some += 1;
                assert_eq!(check(&ssa, &no_tables(), &root), Ok(()));
            }
        }
        assert!(resplit_some > 0, "the corpus must exercise the re-split");
    }

    #[test]
    fn duplications_count_exactly_the_extra_occurrences_in_the_tree() {
        fn occurrences(node: &Node, out: &mut BTreeMap<u64, usize>) {
            match node {
                Node::Block(b) | Node::Opaque { block: b, .. } => {
                    *out.entry(*b).or_default() += 1;
                }
                Node::Seq(v) => v.iter().for_each(|c| occurrences(c, out)),
                Node::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    occurrences(then_body, out);
                    if let Some(e) = else_body {
                        occurrences(e, out);
                    }
                }
                Node::Loop { body, .. } => occurrences(body, out),
                Node::Switch { block, cases } => {
                    *out.entry(*block).or_default() += 1;
                    cases.iter().for_each(|(_, c)| occurrences(c, out));
                }
                Node::Break | Node::Continue | Node::Goto(_) => {}
            }
        }
        let mut cases: Vec<(SsaFunction, BTreeMap<u64, Vec<u64>>)> = [
            merged_tail(vec![ret()], vec![]),
            merged_tail(vec![jmp(0x9000)], vec![0x9000]),
            merged_tail(vec![jmp(0x1030)], vec![0x1030]),
            gadget_corpus(24),
            irreducible(),
            cond_tail(),
            threaded_latch(),
        ]
        .into_iter()
        .map(|f| (build(&f), no_tables()))
        .collect();
        cases.push(shared_case_tail());
        for (ssa, tables) in cases {
            let (root, stats) = run(&ssa, &tables);
            let mut occ = BTreeMap::new();
            occurrences(&root, &mut occ);
            let extras: usize = occ.values().map(|&n| n - 1).sum();
            assert_eq!(extras, stats.duplications, "on {:#x}", ssa.entry);
        }
    }

    #[test]
    fn a_tail_inlined_into_a_self_loop_body_updates_the_loop_kind() {
        // A self-loop whose exit edge was hand-realized as a goto: the
        // re-split inlines the `ret` tail into the one-block body, and
        // the loop's kind must follow the covered count the verifier
        // recomputes.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1000)], vec![0x1000, 0x1010]),
                block(0x1010, vec![ret()], vec![]),
            ],
        );
        let ssa = build(&f);
        let gotoed = Node::Seq(vec![
            Node::Loop {
                kind: LoopKind::SelfLoop,
                cond: None,
                body: Box::new(Node::Seq(vec![
                    Node::Block(0x1000),
                    Node::If {
                        cond: Cond {
                            block: 0x1000,
                            negated: true,
                        },
                        then_body: Box::new(Node::Goto(0x1010)),
                        else_body: None,
                    },
                ])),
            },
            Node::Block(0x1010),
        ]);
        assert_eq!(check(&ssa, &no_tables(), &gotoed), Ok(()));
        let mut stats = StructStats {
            gotos: 1,
            ..Default::default()
        };
        let split = resplit_tails(&ssa, &no_tables(), &copy_edges(&ssa), gotoed, &mut stats, MAX_TAIL_SPLITS);
        assert_eq!((stats.duplications, stats.gotos), (1, 0));
        assert_eq!(check(&ssa, &no_tables(), &split), Ok(()));
        let Node::Seq(v) = &split else {
            panic!("a sequence")
        };
        let Node::Loop { kind, .. } = &v[0] else {
            panic!("a loop")
        };
        assert_eq!(*kind, LoopKind::While, "two covered blocks are no self-loop");
    }

    #[test]
    fn a_goto_to_an_enclosing_loops_exit_is_never_inlined() {
        // A nested loop holding a goto to the *outer* loop's exit block
        // (tighten only converts a loop's own level, so such a goto
        // survives collapse). Inlining the `ret` tail there would put
        // the outer loop's exit inside its body and belie the loop's
        // condition — the re-split must leave the whole target alone.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1030)], vec![0x1030, 0x1010]),
                block(0x1010, vec![set_flag(2), jcc(0x1030)], vec![0x1030, 0x1020]),
                block(0x1020, vec![set_flag(3), jcc(0x1010)], vec![0x1010, 0x1000]),
                block(0x1030, vec![ret()], vec![]),
            ],
        );
        let ssa = build(&f);
        let nested = Node::Seq(vec![
            Node::Loop {
                kind: LoopKind::While,
                cond: Some(Cond {
                    block: 0x1000,
                    negated: true,
                }),
                body: Box::new(Node::Seq(vec![
                    Node::Block(0x1000),
                    Node::Loop {
                        kind: LoopKind::DoWhile,
                        cond: Some(Cond {
                            block: 0x1020,
                            negated: false,
                        }),
                        body: Box::new(Node::Seq(vec![
                            Node::Block(0x1010),
                            Node::If {
                                cond: Cond {
                                    block: 0x1010,
                                    negated: false,
                                },
                                then_body: Box::new(Node::Goto(0x1030)),
                                else_body: None,
                            },
                            Node::Block(0x1020),
                        ])),
                    },
                ])),
            },
            Node::Block(0x1030),
        ]);
        assert_eq!(check(&ssa, &no_tables(), &nested), Ok(()));
        let mut stats = StructStats {
            gotos: 1,
            ..Default::default()
        };
        let same = resplit_tails(&ssa, &no_tables(), &copy_edges(&ssa), nested.clone(), &mut stats, MAX_TAIL_SPLITS);
        assert_eq!(same, nested, "the loop-exit target must be left alone");
        assert_eq!((stats.duplications, stats.gotos), (0, 1));
        assert!(!stats.dup_capped, "ineligibility is not the cap");
        assert_eq!(check(&ssa, &no_tables(), &same), Ok(()));
    }

    // -- 12: the SAILR jump threading (inversion three) ---------------------

    /// The merged tail carries a *condition* — jump threading's
    /// signature: no linear duplicate exists, both prior inversions
    /// refuse by design, and the goto used to survive.
    fn cond_tail() -> irlift::LiftedFunction {
        let mut f = merged_tail(vec![set_flag(7), jcc(0x1030)], vec![0x1030, 0x1050]);
        f.blocks.insert(0x1050, block(0x1050, vec![ret()], vec![]));
        f
    }

    #[test]
    fn a_condition_carrying_shared_tail_threads_into_a_real_if() {
        // Inversion three: the deciding block is duplicated into the
        // goto-ing site as its plain leaf plus the real `If` referencing
        // the copy — and the copy's arm inlines the fresh linear tail
        // the thread exposes, spelled by the case-tail classifier (the
        // composed inversions). Both occurrences render the one
        // statement list and the one honest condition reference.
        let (ssa, root, stats) = tree(&cond_tail());
        let (bare, bstats) = raw(&ssa, &no_tables());
        assert_eq!(bstats.gotos, 1);
        assert!(render(&ssa, &bare).contains("goto loc_1020"));
        assert_eq!(
            (stats.gotos, stats.duplications, stats.threaded),
            (0, 2, 1),
            "one threaded site: the deciding block plus its inlined arm tail"
        );
        assert!(!stats.dup_capped);
        assert_eq!(
            text(&ssa, &root),
            "; sub_1040 @ 0x0000000000001040 (structure)\n\
             block loc_1040\n\
             if cond loc_1040\n\
             \x20 block loc_1000\n\
             \x20 if cond loc_1000\n\
             \x20   block loc_1020\n\
             \x20   if !cond loc_1020\n\
             \x20     block loc_1050\n\
             else\n\
             \x20 block loc_1010\n\
             \x20 if cond loc_1010\n\
             \x20   block loc_1020\n\
             \x20   if !cond loc_1020\n\
             \x20     block loc_1050\n\
             block loc_1030\n"
        );
    }

    /// Two arms cross into a shared condition-carrying latch that
    /// decides continue-vs-break — the canonical threaded loop shape.
    fn threaded_latch() -> irlift::LiftedFunction {
        func(
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
        )
    }

    #[test]
    fn a_threaded_latch_spells_continue_and_break() {
        // The copy's arm realizes the back edge as `continue` and its
        // open side as `break` — both edges spelled, no goto spent, the
        // condition reference honest at both sites.
        let (ssa, root, stats) = tree(&threaded_latch());
        let (bare, bstats) = raw(&ssa, &no_tables());
        assert!(render(&ssa, &bare).contains("goto loc_1020"), "the cross edge degrades");
        assert!(bstats.gotos > 0);
        assert_eq!((stats.gotos, stats.duplications, stats.threaded), (0, 1, 1));
        let rendered = text(&ssa, &root);
        assert!(!rendered.contains("goto"), "{rendered}");
        let site = "      block loc_1020\n      if cond loc_1020\n        continue\n      break\n";
        assert_eq!(
            rendered.matches(site).count(),
            2,
            "both occurrences spell the same threaded condition: {rendered}"
        );
    }

    #[test]
    fn a_small_pure_irreducible_entry_threads_and_the_cycle_structures() {
        // The pure twin of `irreducible()`: with the stores gone, the
        // cycle blocks are small pure conditions, and one bounded
        // duplicate — classic node splitting — lets the two-entry cycle
        // structure as a `while`, every edge still verified.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1020)], vec![0x1020, 0x1010]),
                block(0x1010, vec![set_flag(2), jcc(0x1020)], vec![0x1020, 0x1030]),
                block(0x1020, vec![set_flag(3), jcc(0x1010)], vec![0x1010, 0x1030]),
                block(0x1030, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&f);
        assert_eq!((stats.gotos, stats.duplications, stats.threaded), (0, 2, 1));
        assert_eq!(
            text(&ssa, &root),
            "; sub_1000 @ 0x0000000000001000 (structure)\n\
             block loc_1000\n\
             if !cond loc_1000\n\
             \x20 block loc_1010\n\
             \x20 if !cond loc_1010\n\
             \x20   block loc_1030\n\
             while cond loc_1020\n\
             \x20 block loc_1020\n\
             \x20 block loc_1010\n\
             \x20 if cond loc_1010\n\
             \x20   continue\n\
             \x20 break\n\
             block loc_1030\n"
        );
    }

    #[test]
    fn a_threads_exposed_tail_composes_with_the_chain_inversion() {
        // The arm target 0x1030 is *also* a live goto target: the chain
        // inversion takes its direct goto and the thread's arm may
        // inline it only because that same round rewrites the target
        // whole — no duplicated block is ever also a goto target.
        let f = func(
            0x1200,
            vec![
                block(0x1200, vec![set_flag(1), jcc(0x1060)], vec![0x1060, 0x1100]),
                block(0x1060, vec![set_flag(2), jcc(0x1030)], vec![0x1030, 0x1100]),
                block(0x1100, vec![set_flag(3), jcc(0x1000)], vec![0x1000, 0x1010]),
                block(0x1000, vec![set_flag(4), jcc(0x1040)], vec![0x1040, 0x1020]),
                block(0x1010, vec![set_flag(5), jcc(0x1040)], vec![0x1040, 0x1020]),
                block(0x1020, vec![set_flag(7), jcc(0x1040)], vec![0x1040, 0x1030]),
                block(0x1030, vec![ret()], vec![]),
                block(0x1040, vec![ret()], vec![]),
            ],
        );
        let (ssa, root, stats) = tree(&f);
        assert_eq!((stats.gotos, stats.duplications, stats.threaded), (0, 4, 1));
        let rendered = text(&ssa, &root);
        assert!(!rendered.contains("goto"), "{rendered}");
        assert_eq!(rendered.matches("block loc_1030").count(), 3, "{rendered}");
        assert_eq!(rendered.matches("block loc_1020").count(), 2, "{rendered}");
        assert_eq!(rendered.matches("block loc_1040").count(), 2, "{rendered}");
    }

    #[test]
    fn an_oversized_condition_block_keeps_its_goto() {
        // The size budget, at its exact boundary: one assignment past
        // MAX_THREAD_STMTS refuses — bit for bit — and trimming it back
        // to the cap threads the same shape.
        let sized = |extra: usize| {
            let mut stmts: Vec<Stmt> = (0..extra)
                .map(|i| assign(ra(1, Width::W64), c(i as u64, Width::W64)))
                .collect();
            stmts.push(set_flag(7));
            stmts.push(jcc(0x1030));
            let mut f = merged_tail(stmts, vec![0x1030, 0x1050]);
            f.blocks.insert(0x1050, block(0x1050, vec![ret()], vec![]));
            f
        };
        let (ssa, root, stats) = tree(&sized(MAX_THREAD_STMTS));
        assert!(threadable_head(&ssa, &no_tables(), &copy_edges(&ssa), 0x1020).is_none());
        assert_eq!((stats.duplications, stats.threaded), (0, 0));
        assert!(!stats.dup_capped, "the bound is not the budget");
        let (bare, bstats) = raw(&ssa, &no_tables());
        assert_eq!(root, bare);
        assert_eq!(stats, bstats);
        assert!(text(&ssa, &root).contains("goto loc_1020"));

        let (ssa, root, stats) = tree(&sized(MAX_THREAD_STMTS - 1));
        assert!(threadable_head(&ssa, &no_tables(), &copy_edges(&ssa), 0x1020).is_some());
        assert!(stats.threaded > 0, "{}", text(&ssa, &root));
        assert!(!text(&ssa, &root).contains("goto loc_1020"));
    }

    /// The φ fixture, polarity-selectable: the arms define `r1`
    /// differently and the join reads it. Raw construction is
    /// conventional — the whole web coalesces and the edge set is
    /// empty — so the refusal variants stale the deciding block's edge
    /// argument to the entry's r1 (see [`stale_phi_arg`]), which
    /// interferes with the arms' definitions and forces a real copy
    /// with exactly one textual placement.
    fn phi_edge(taken_into_join: bool) -> irlift::LiftedFunction {
        let (taken, fallthrough) = if taken_into_join {
            (0x1030, 0x1050)
        } else {
            (0x1050, 0x1030)
        };
        let arm = |va: u64, n: u64| {
            block(
                va,
                vec![
                    assign(ra(1, Width::W64), c(n, Width::W64)),
                    set_flag(n),
                    jcc(0x1020),
                ],
                vec![0x1020, 0x1030],
            )
        };
        func(
            0x1040,
            vec![
                block(
                    0x1040,
                    vec![assign(ra(1, Width::W64), c(9, Width::W64)), set_flag(1), jcc(0x1000)],
                    vec![0x1000, 0x1010],
                ),
                arm(0x1000, 2),
                arm(0x1010, 3),
                block(
                    0x1020,
                    vec![set_flag(7), jcc(taken)],
                    vec![taken, fallthrough],
                ),
                block(
                    0x1030,
                    vec![assign(ra(4, Width::W64), read(ra(1, Width::W64))), ret()],
                    vec![],
                ),
                block(0x1050, vec![ret()], vec![]),
            ],
        )
    }

    #[test]
    fn a_real_copy_on_the_taken_edge_refuses_the_thread() {
        let mut ssa = build(&phi_edge(true));
        let stale = def_of(&ssa, 0x1040, 1);
        stale_phi_arg(&mut ssa, 0x1030, 0x1020, stale);
        let copies = copy_edges(&ssa);
        assert!(
            copies.contains(&(0x1020, 0x1030)),
            "the taken edge must carry a real out-of-SSA copy for the fixture to bite"
        );
        assert!(threadable_head(&ssa, &no_tables(), &copies, 0x1020).is_none());
        let (root, stats) = run(&ssa, &no_tables());
        assert_eq!((stats.duplications, stats.threaded), (0, 0));
        let (bare, bstats) = raw(&ssa, &no_tables());
        assert_eq!(root, bare);
        assert_eq!(stats, bstats);
        assert!(text(&ssa, &root).contains("goto loc_1020"));
    }

    #[test]
    fn a_real_copy_on_the_fallthrough_edge_refuses_the_thread() {
        let mut ssa = build(&phi_edge(false));
        let stale = def_of(&ssa, 0x1040, 1);
        stale_phi_arg(&mut ssa, 0x1030, 0x1020, stale);
        let copies = copy_edges(&ssa);
        assert!(
            copies.contains(&(0x1020, 0x1030)),
            "the fall-through edge must carry a real out-of-SSA copy for the fixture to bite"
        );
        assert!(threadable_head(&ssa, &no_tables(), &copies, 0x1020).is_none());
        let (root, stats) = run(&ssa, &no_tables());
        assert_eq!((stats.duplications, stats.threaded), (0, 0));
        let (bare, bstats) = raw(&ssa, &no_tables());
        assert_eq!(root, bare);
        assert_eq!(stats, bstats);
        assert!(text(&ssa, &root).contains("goto loc_1020"));
    }

    #[test]
    fn the_coalesced_phi_web_no_longer_refuses_the_thread() {
        // The same shape un-staled: different SSA names for one value,
        // refused by the old name-identity approximation — but the web
        // coalesces into one variable, irout's edge set is empty, and
        // the ground truth threads it.
        let ssa = build(&phi_edge(true));
        let copies = copy_edges(&ssa);
        assert!(
            copies.is_empty(),
            "raw construction is conventional: no copies anywhere"
        );
        assert!(threadable_head(&ssa, &no_tables(), &copies, 0x1020).is_some());
        let (root, stats) = run(&ssa, &no_tables());
        assert!(stats.threaded >= 1, "the narrowing must fire: {stats:?}");
        let rendered = text(&ssa, &root);
        assert!(!rendered.contains("goto"), "{rendered}");
    }

    #[test]
    fn a_thread_that_does_not_fit_the_budget_degrades_to_its_gotos() {
        // The site costs two leaves (the deciding block plus its inlined
        // arm tail): a cap of one refuses the target whole and keeps the
        // goto with `dup_capped` set; a cap of two threads it.
        let ssa = build(&cond_tail());
        let (bare, bstats) = raw(&ssa, &no_tables());
        let mut stats = bstats;
        let same = resplit_tails(&ssa, &no_tables(), &copy_edges(&ssa), bare.clone(), &mut stats, 1);
        assert!(stats.dup_capped, "the thread must overrun the small cap");
        assert_eq!((stats.duplications, stats.threaded), (0, 0));
        assert!(render(&ssa, &same).contains("goto loc_1020"));
        assert_eq!(check(&ssa, &no_tables(), &same), Ok(()));
        let mut stats = bstats;
        let split = resplit_tails(&ssa, &no_tables(), &copy_edges(&ssa), bare, &mut stats, 2);
        assert!(!stats.dup_capped);
        assert_eq!((stats.duplications, stats.threaded, stats.gotos), (2, 1, 0));
        assert_eq!(check(&ssa, &no_tables(), &split), Ok(()));
        assert!(!render(&ssa, &split).contains("goto"));
    }

    #[test]
    fn an_enclosing_loops_conditional_exit_never_threads() {
        // The outer loop's recorded exit block now carries a condition:
        // a threadable head by every local rule, but inlining it at the
        // inner goto would put the exit inside the body and belie the
        // outer loop's condition — the same `leaves` refusal the chain
        // inversions honor.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![set_flag(1), jcc(0x1030)], vec![0x1030, 0x1010]),
                block(0x1010, vec![set_flag(2), jcc(0x1030)], vec![0x1030, 0x1020]),
                block(0x1020, vec![set_flag(3), jcc(0x1010)], vec![0x1010, 0x1000]),
                block(0x1030, vec![set_flag(4), jcc(0x1040)], vec![0x1040, 0x1050]),
                block(0x1040, vec![ret()], vec![]),
                block(0x1050, vec![ret()], vec![]),
            ],
        );
        let ssa = build(&f);
        assert!(threadable_head(&ssa, &no_tables(), &copy_edges(&ssa), 0x1030).is_some());
        let nested = Node::Seq(vec![
            Node::Loop {
                kind: LoopKind::While,
                cond: Some(Cond {
                    block: 0x1000,
                    negated: true,
                }),
                body: Box::new(Node::Seq(vec![
                    Node::Block(0x1000),
                    Node::Loop {
                        kind: LoopKind::DoWhile,
                        cond: Some(Cond {
                            block: 0x1020,
                            negated: false,
                        }),
                        body: Box::new(Node::Seq(vec![
                            Node::Block(0x1010),
                            Node::If {
                                cond: Cond {
                                    block: 0x1010,
                                    negated: false,
                                },
                                then_body: Box::new(Node::Goto(0x1030)),
                                else_body: None,
                            },
                            Node::Block(0x1020),
                        ])),
                    },
                ])),
            },
            Node::Block(0x1030),
            Node::If {
                cond: Cond {
                    block: 0x1030,
                    negated: false,
                },
                then_body: Box::new(Node::Block(0x1040)),
                else_body: None,
            },
            Node::Block(0x1050),
        ]);
        assert_eq!(check(&ssa, &no_tables(), &nested), Ok(()));
        let mut stats = StructStats {
            gotos: 1,
            ..Default::default()
        };
        let same = resplit_tails(&ssa, &no_tables(), &copy_edges(&ssa), nested.clone(), &mut stats, MAX_TAIL_SPLITS);
        assert_eq!(same, nested, "the loop-exit head must be left alone");
        assert_eq!((stats.duplications, stats.threaded, stats.gotos), (0, 0, 1));
        assert!(!stats.dup_capped, "ineligibility is not the cap");
    }

    #[test]
    fn check_holds_a_deciding_duplicate_to_its_branch() {
        // A sanctioned deciding duplicate must spell its `If` and
        // realize its untaken side — the condition-honesty rules hold on
        // copies exactly as on originals, even though every CFG edge is
        // already realized at the original occurrence.
        let ssa = build(&diamond());
        let original = vec![
            Node::Block(0x1000),
            Node::If {
                cond: Cond {
                    block: 0x1000,
                    negated: false,
                },
                then_body: Box::new(Node::Block(0x1020)),
                else_body: Some(Box::new(Node::Block(0x1010))),
            },
            Node::Block(0x1030),
        ];
        let with = |rest: Vec<Node>| {
            let mut v = original.clone();
            v.extend(rest);
            Node::Seq(v)
        };
        // The honest copy: the `If` decides, the leftover realizes the
        // untaken side.
        let honest = with(vec![
            Node::Block(0x1000),
            Node::If {
                cond: Cond {
                    block: 0x1000,
                    negated: false,
                },
                then_body: Box::new(Node::Goto(0x1020)),
                else_body: None,
            },
            Node::Goto(0x1010),
        ]);
        assert_eq!(check(&ssa, &no_tables(), &honest), Ok(()));
        // A copy that gotos past its branch drops it.
        let dropped = with(vec![Node::Block(0x1000), Node::Goto(0x1020)]);
        assert!(matches!(
            check(&ssa, &no_tables(), &dropped),
            Err(StructFault::Undecided { block: 0x1000 })
        ));
        // A copy funneling both polarities to the taken side belies the
        // branch even though 0x1000 -> 0x1020 is a real edge.
        let funneled = with(vec![
            Node::Block(0x1000),
            Node::If {
                cond: Cond {
                    block: 0x1000,
                    negated: false,
                },
                then_body: Box::new(Node::Goto(0x1020)),
                else_body: None,
            },
            Node::Goto(0x1020),
        ]);
        assert!(matches!(
            check(&ssa, &no_tables(), &funneled),
            Err(StructFault::Polarity { block: 0x1000 })
        ));
        // A copy whose untaken side never realizes at all.
        let unrealized = with(vec![
            Node::Block(0x1000),
            Node::If {
                cond: Cond {
                    block: 0x1000,
                    negated: false,
                },
                then_body: Box::new(Node::Goto(0x1020)),
                else_body: None,
            },
        ]);
        assert!(matches!(
            check(&ssa, &no_tables(), &unrealized),
            Err(StructFault::Polarity { block: 0x1000 })
        ));
    }

    #[test]
    fn check_rejects_a_flipped_polarity_on_a_deciding_duplicate() {
        // The copy's `If` claims the negated polarity, so its arm should
        // be the fall-through side — but it gotos the taken side.
        let ssa = build(&diamond());
        let bad = Node::Seq(vec![
            Node::Block(0x1000),
            Node::If {
                cond: Cond {
                    block: 0x1000,
                    negated: false,
                },
                then_body: Box::new(Node::Block(0x1020)),
                else_body: Some(Box::new(Node::Block(0x1010))),
            },
            Node::Block(0x1030),
            Node::Block(0x1000),
            Node::If {
                cond: Cond {
                    block: 0x1000,
                    negated: true,
                },
                then_body: Box::new(Node::Goto(0x1020)),
                else_body: None,
            },
            Node::Goto(0x1010),
        ]);
        assert!(matches!(
            check(&ssa, &no_tables(), &bad),
            Err(StructFault::Polarity { block: 0x1000 })
        ));
    }

    #[test]
    fn check_accepts_bounded_tail_duplicates_and_rejects_a_flood() {
        let f = func(0x1000, vec![block(0x1000, vec![ret()], vec![])]);
        let ssa = build(&f);
        let dup = |n: usize| Node::Seq(vec![Node::Block(0x1000); n]);
        assert_eq!(check(&ssa, &no_tables(), &dup(1 + MAX_TAIL_SPLITS)), Ok(()));
        assert!(matches!(
            check(&ssa, &no_tables(), &dup(2 + MAX_TAIL_SPLITS)),
            Err(StructFault::Duplicated { block: 0x1000 })
        ));
    }

    #[test]
    fn check_rejects_a_duplicated_opaque_block() {
        let mut f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010]),
                block(0x1010, vec![ret()], vec![]),
            ],
        );
        f.blocks.get_mut(&0x1000).expect("the entry").truncated = true;
        let ssa = build(&f);
        let opaque = Node::Opaque {
            block: 0x1000,
            reason: OpaqueReason::Truncated,
        };
        let bad = Node::Seq(vec![opaque.clone(), opaque, Node::Block(0x1010)]);
        assert!(matches!(
            check(&ssa, &no_tables(), &bad),
            Err(StructFault::Duplicated { block: 0x1000 })
        ));
    }
}
