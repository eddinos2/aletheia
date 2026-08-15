# Plan: Phoenix-style control-flow structuring (`irstruct`, slice 6)

## Input / output

Input: an `irssa::SsaFunction` (any — faithful or optimized; the pass
reads CFG shape and branch *positions*, never expression content) plus
the proven jump tables as `&BTreeMap<u64, Vec<u64>>`
(`jumptable::successor_map`, keyed by jump-site VA; empty map is fine).
Output: a structure tree covering the function's reachable blocks, plus
`StructStats`. New module `src/irstruct.rs` (DESIGN.md slice 6 —
Schwartz et al., USENIX Security 2013 (Phoenix), on the
Cifuentes 1994 / Sharir 1980 schema lineage). New view
`redump --structure[=N]`.

## The key structural decisions

- **Conditions are references, not expressions.** An `If`/`Loop`
  condition is `Cond { block: u64, negated: bool }` — the VA of the
  block whose final conditional `Branch` decides it, plus polarity.
  The tree never copies or rewrites an expression, so it is valid over
  the faithful and the optimized SSA alike; the renderer (slice 8)
  fetches the expression at print time.
- **Gotos are honest output, not failure.** Per SAILR's evidence,
  no node duplication and no condition synthesis, ever: when no schema
  matches anywhere, virtualize exactly one edge into an explicit
  `Goto` and retry. Deterministic choice: the removable edge with the
  lowest `(source VA, target VA)`. Each virtualization removes one
  edge from the region graph, so termination is structural, not
  cap-dependent (the cap below is defense in depth).
- **One trait seam.** `trait Structurer { fn structure(...) }` with
  the Phoenix implementation as the only impl — angr's
  pluggable-structurer lesson, so a SAILR-style strategy can be
  compared later on the same regions. Keep the seam minimal; no
  premature abstraction beyond the one trait.

## The tree

```rust
pub enum Node {
    /// One basic block's straight-line statements (by VA). Its
    /// terminator is implied by the parent construct; a leaf never
    /// invents flow.
    Block(u64),
    Seq(Vec<Node>),
    If { cond: Cond, then_body: Box<Node>, else_body: Option<Box<Node>> },
    Loop { kind: LoopKind, cond: Option<Cond>, body: Box<Node> },
    /// Only where `tables` proved the dispatch: scrutinee block +
    /// (case target VA -> body) in ascending target order + the edges'
    /// provenance. Never synthesized from an unproven indirect jump.
    Switch { block: u64, cases: Vec<(u64, Node)> },
    Break,
    Continue,
    Goto(u64),
    /// Truncated, undecodable, or unproven-indirect-successor block:
    /// held, never absorbed, never given invented fall-through.
    Opaque { block: u64, reason: OpaqueReason },
}
pub struct Cond { pub block: u64, pub negated: bool }
pub enum LoopKind { SelfLoop, While, DoWhile }
pub struct StructStats {
    pub rounds: usize,
    pub gotos: usize,     // edges virtualized
    pub capped: bool,     // the defense cap fired; output is still valid
}
pub fn structure(f: &SsaFunction, tables: &BTreeMap<u64, Vec<u64>>)
    -> (Node, StructStats);
pub fn check(f: &SsaFunction, tables: &..., root: &Node) -> Result<(), StructFault>;
pub fn render(f: &SsaFunction, root: &Node) -> String; // indented tree, goldens
```

## Algorithm (Phoenix, iterative region collapse)

Work on a region graph initialized to one region per reachable block
(reachability from `f.entry` over in-function successor edges;
`f.skipped` stays listed, not structured). Iterate to a fixpoint: walk
regions in post-order and try the schema catalog at each head;
collapsing a match replaces the matched regions with one abstract
region carrying the built `Node`. When a full pass collapses nothing
and more than one region remains, virtualize one edge (rule above) and
continue. `MAX_ROUNDS`-style cap (`MAX_STRUCT_ROUNDS`, sized to blocks
+ edges, so it cannot fire on sane input) with `capped = true` and a
valid partial tree (remaining regions joined by `Goto`s) — degrade,
never refuse.

Schema catalog, tried in this order at each head:

- **Sequence**: single-successor chain of regions each with one
  predecessor.
- **If-then / if-then-else**: two-way conditional head; branches
  converge on a unique follow region (or a branch is empty). The
  branch protected by the taken edge is `negated: false` against the
  block's `Branch { cond, target }` — polarity is *defined* as: the
  branch statement's guard true means control goes to `target`; the
  `then_body` is the side the tree puts under the un-negated cond.
- **Self-loop**: a region whose only back edge is to itself →
  `LoopKind::SelfLoop`.
- **Natural while / do-while**: back edge to a dominating header
  (dominators over the region graph — reuse `irssa`'s CHK internals as
  `pub(crate)` if cleanly exposable, else a local CHK on the block
  graph; do NOT duplicate nontrivial code, prefer the expose). Follow
  node by the documented deterministic rule: immediate post-dominator
  of the header if one exists, else the most-frequent exit-edge
  target, ties broken by lowest address. In-body edges to the follow
  become `Break`, edges to the header become `Continue`; any other
  exit edge blocks the match (leave it to a later round or a goto).
  `While` when the header's conditional exits to the follow, `DoWhile`
  when the latch's does.
- **Switch**: a head whose final statement is an indirect
  `Branch { kind: Jump, cond: None }` at a jump-site VA present in
  `tables`, with the case regions single-predecessor and converging.
  Note the honest limit (record it in the module docs): `cfg::recover`
  does not yet fold `jumptable::successor_map` into block successors,
  so on today's real pipeline an indirect-jump block usually has no
  in-function case edges and renders `Opaque` — `Switch` is exercised
  by synthetic tests; the CFG-folding rider belongs to a later slice.

Out-of-function successor edges (tail jumps): realized as `Goto`
(external VA), never counted as structuring edges. A block with
`truncated: true`, or ending in an unproven indirect jump, becomes
`Opaque { reason }` — it may sit inside `Seq`/`If`/`Loop` bodies but
nothing is ever absorbed *into* it and it is never given a
fall-through it doesn't have.

## `check` (the companion, trusted over the pass)

- **Exact partition**: every reachable block appears in exactly one
  `Block`/`Opaque`/`Switch`-scrutinee position, exactly once
  (duplication is banned, so coverage is a partition).
- **Every CFG edge realized**: as fall-through inside a `Seq`, a
  structured construct's edge (if/loop/switch/break/continue), or an
  explicit `Goto`. No edge silently dropped, none invented.
- **Condition honesty**: each `Cond.block` names a block whose final
  statement is a conditional `Branch`; each `Switch.block` ends in the
  proven indirect jump.
- **Opaque honesty**: every truncated/unproven-indirect block is
  `Opaque` with the right reason; `skipped` blocks absent from the
  tree.
- Malformed input (fails `irssa::check`): refuse to interpret, the
  established posture — return the degenerate tree (every reachable
  block as `Block` + explicit `Goto`s in a `Seq`, which trivially
  passes `check`) with `rounds = 0`; document the choice in the
  module docs.

## Determinism & soundness summary

- All containers `BTree*`; post-order walk from sorted successors;
  the virtualization rule is a total order. Same input → byte-equal
  tree. Pure, total, no panics on any input.
- The pass never mutates the `SsaFunction`, never duplicates a block,
  never synthesizes a condition, never reorders statements.

## Module-by-module

- `src/irstruct.rs`: everything above + module docs in the house style
  (cite Phoenix/SAILR, state the schema order, the follow-node rule,
  the polarity definition, the Switch limit) + tests.
- `src/irssa.rs`: only if the CHK internals are exposed `pub(crate)`
  for reuse — no behavior change.
- `src/bin/redump.rs`: `--structure[=N]` — runs the `--ssa-opt`
  pipeline (optimize → forward → eliminate_dead) per function, then
  `structure` + `render`; prints stats line (`; structure: N gotos` when
  nonzero). Usage text updated.
- `ROADMAP.md`: Current-thread → slice 6 ✅ with verified evidence,
  Active → slice 7 (`irout`, Boissinot out-of-SSA).

## Test matrix (~20)

1. each schema in isolation, golden trees: sequence; if-then (both
   polarities); if-then-else; self-loop; while; do-while;
   break and continue inside a while; proven-table switch.
2. nested combinations: if inside loop, loop inside if-else, seq of
   diamonds — goldens.
3. irreducible graph (two-entry loop) → structures with minimal gotos,
   deterministic, `check` Ok.
4. truncated block → `Opaque` in place, parent structure unaffected.
5. unproven indirect jump → `Opaque`; same graph with the table in
   `tables` → `Switch`.
6. out-of-function edge → external `Goto`.
7. cap forced low (test-only knob or a pathological graph) → degrades
   to gotos, `capped`, `check` still Ok.
8. seeded random-CFG sweep (mirror `irssaopt`'s harness): construct →
   structure → `check` Ok, no panics, byte-determinism (twice →
   byte-equal render).
9. malformed input → documented degenerate output, zeroed stats.
10. redump e2e: calling + diamond fixtures show golden `--structure`
    output; `/bin/ls` x86-64 slice: zero `check` failures across all
    functions, byte-deterministic; report at least one real function
    rendering a `Loop` and one an `If`/`else`.

## Exit criteria (DESIGN, verbatim)

`redump --structure` renders golden trees for a fixture set including
at least one real recovered function with a loop, a diamond, and an
irreducible region (the /bin/ls sweep must surface all three; if no
irreducible region exists in /bin/ls, /bin/bash is the fallback — find
one, don't skip the criterion).

## Non-goals (this slice, per DESIGN)

- Condition synthesis, node duplication, DREAM/combing-style
  goto-free rewriting (SAILR pre-passes are slice 18, behind the
  trait seam).
- Folding `jumptable::successor_map` into `cfg::recover` (its own
  future rider; the structurer takes the map as a parameter).
- Out-of-SSA, variable naming, pseudocode text (slices 7–8).
- aarch64 end-to-end (still gated on the `irlift` dispatch slice; the
  pass is ISA-blind).
