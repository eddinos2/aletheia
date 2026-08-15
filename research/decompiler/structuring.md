# Control-flow structuring

Status: researched 2026-08-15. Sources are published papers and openly
documented open-source projects only (clean-room discipline per BRIEF.md).

## The problem

A recovered function arrives as a CFG — basic blocks and edges
(`cfg::Function`, lifted to IR blocks by `irlift`, renamed by `irssa`) —
but readable pseudocode needs `if`/`else`, loops, `switch`, `break`,
`continue`, and, where nothing else is honest, `goto`. Structuring is the
transformation from the graph domain to the syntax-tree domain. It must be
semantics-preserving (every path in the output executes the same blocks
under the same conditions as the input graph), total on hostile input
(irreducible graphs, blocks with no successors, truncated lifts), bounded
(no exponential duplication or non-terminating refinement), and honest
about what it could not structure — which for Aletheia specifically
includes blocks `irlift` truncated and edges `cfg` never had (indirect
jumps contribute no intra-function edges; `Terminator::Undecodable` and
`Terminator::Truncated` blocks end with no successors).

## The strongest published approaches

### 1. Interval / structural analysis — Cifuentes (1994)

Cifuentes, *Reverse Compilation Techniques*, PhD thesis, Queensland
University of Technology, 1994 (the `dcc` decompiler); building on
interval analysis (Allen/Cocke) and structural analysis (Sharir, 1980).
Loops are found via intervals (later: dominators + back edges), two-way
conditionals via a pattern schema over the dominator tree; anything that
matches no schema becomes a `goto`. Establishes the *schema-matching*
family: iterate over regions, match a fixed catalog (block sequence,
if-then, if-then-else, natural loop, …), collapse the match to a single
abstract node, repeat until one node remains.

- Strength: simple, fast (near-linear in practice), each collapse is
  locally verifiable, `goto` fallback makes it total.
- Weakness: a fixed schema catalog misses compiler-generated shapes, so
  unoptimized structurers of this family emit many gotos on optimized
  code; naive schemas can be *semantics-changing* if matched carelessly
  (the exact failure Phoenix fixed).

### 2. Phoenix — Schwartz, Lee, Woo, Brumley (USENIX Security 2013)

*Native x86 Decompilation Using Semantics-Preserving Structural Analysis
and Iterative Control-Flow Structuring.* Two amendments to the
Cifuentes/Sharir line:

1. **Semantics-preserving schemas only.** Each schema match is required
   to preserve behavior by construction (correct edge conditions, single
   entry), verified in their evaluation by recompiling and re-running
   test suites — the first structuring work with a correctness
   evaluation rather than only a goto count.
2. **Iterative refinement.** When no schema matches anywhere, do not
   give up on the whole region: pick one edge, *virtualize* it (replace
   it with an explicit `goto`-to-label pair, removing the edge from the
   graph), and retry. Edge choice is heuristic (prefer edges whose
   removal restores reducibility / enables a match); the mechanism is
   provably terminating because each round removes an edge.

- Strength: total and terminating on any graph, including irreducible
  ones; correctness is compositional (schemas) plus trivially sound
  fallback (a `goto` is exactly an edge); 30× fewer gotos than dcc-style
  baselines in their evaluation.
- Weakness: still emits gotos on optimized code; edge-choice heuristic
  affects output quality but never correctness.

### 3. Pattern-independent structuring — Yakdan, Eschweiler,
### Gerhards-Padilla, Smith (NDSS 2015), "No More Gotos" / DREAM

Abandons schemas. For an acyclic region, compute each node's *reaching
condition* (the boolean condition over branch predicates under which the
node executes), then synthesize nested `if`/`else`/`switch` directly from
simplified conditions; cyclic regions are restructured into single-entry,
single-successor loops first (irreducible entries handled by condition
duplication). Output is 100% goto-free by construction, with no code
duplication claimed. Follow-up work (Yakdan et al., *Helping Johnny to
Analyze Malware*, IEEE S&P 2016 — DREAM++) adds readability-motivated
transformations and user-study evidence.

- Strength: goto-free; handled real malware where schema matchers
  drowned in gotos; the reaching-condition idea is independently useful
  (it is essentially path-predicate computation over an acyclic region).
- Weakness: boolean condition simplification is the load-bearing wall —
  worst-case exponential (Quine–McCluskey-style minimization), and when
  it fails to simplify, output conditions are large and *less* readable
  than a goto; SAILR (below) measured DREAM-style output as structurally
  farthest from original source.

### 4. Combing — Gussoni, Di Federico, Fezzardi, Agosta (AsiaCCS 2020),
### rev.ng

*A Comb for Decompiled C Code.* Three stages: preprocess the CFG into a
hierarchy of nested DAGs (loops collapsed; irreducible regions made
reducible), **comb** each DAG by *duplicating* nodes (basic blocks or
already-collapsed regions) and inserting dummy nodes until every
conditional has properly nested arms, then match idiomatic C constructs
on the now-trivially-structured graph. Goto-free by construction;
semantics preserved because duplication and dummy insertion are both
behavior-preserving graph transforms.

- Strength: conceptually clean; every step is a small, checkable graph
  rewrite; goto-free without boolean blow-up.
- Weakness: code duplication — worst case exponential (bounded in
  practice but unbounded in principle), and duplicated code actively
  misleads an analyst (the same source line appears twice); SAILR
  measured combing's CFG edit distance from original source as the
  worst of the algorithms it compared.

### 5. The correction: goto minimization vs semantic fidelity — SAILR
### (Basque et al., USENIX Security 2024)

*Ahoy SAILR! There is No Need to DREAM of C: A Compiler-Aware
Structuring Algorithm for Binary Decompilation* (open-source, on angr;
also reimplements Phoenix, DREAM, and combing for comparison). Findings
that should steer any new decompiler:

- **Gotos are not the enemy.** Real source contains gotos (3,754 in
  Linux 6.1); most *spurious* gotos in decompiled output are artifacts
  of a small set of compiler optimizations (jump threading / cross
  jumping, common-subexpression tail merging, switch lowering).
  Inverting those specific transformations before structuring removes
  the spurious gotos while *keeping* the genuine ones.
- **Measure structure against source, not against zero.** They propose
  CFGED (control-flow-graph edit distance to the original source's
  graph). Goto-free algorithms (DREAM, combing) score *worse* on CFGED
  than goto-tolerant ones: eliminating gotos by condition duplication or
  code duplication moves output *away* from what the programmer wrote.
- With targeted de-optimization plus a Phoenix-style schema structurer,
  SAILR got ~3× fewer spurious gotos than IDA/Ghidra output at
  comparable or better CFGED.

The lesson: **a schema structurer with an honest goto fallback is the
right chassis; quality comes from targeted, provable de-optimizations
layered in front of it — not from banning gotos.** This aligns exactly
with Aletheia's proven-vs-heuristic doctrine: a `goto L` is a *proven*
rendering of an edge; a duplicated block or a synthesized boolean
condition is a rewrite that must earn its soundness.

### Aside: reducible-only algorithms

Relooper (Zakai, Emscripten, OOPSLA 2011) and its successors, and Ramsey,
*Beyond Relooper* (ICFP 2022, functional pearl), give clean
dominance-based algorithms that produce goto-free structured code — but
only for reducible graphs (irreducible input needs prior node splitting
or a dispatch variable). Useful as published, rigorous references for the
*reducible* core; not sufficient alone because binaries are not
guaranteed reducible.

## Trade-offs

| Approach | Gotos | Duplication | Condition blow-up | Irreducible CFGs | Termination/bounds | Fidelity (per SAILR) |
|---|---|---|---|---|---|---|
| Cifuentes schemas | many | none | none | goto fallback | linear-ish | baseline |
| Phoenix | few | none | none | edge virtualization → goto | provable (edge count decreases) | good |
| DREAM (No More Gotos) | zero | little | worst-case exponential | condition duplication | needs simplifier bound | poor (CFGED) |
| rev.ng combing | zero | worst-case exponential | none | node cloning | needs duplication cap | worst (CFGED) |
| SAILR | few, genuine | none | none | inherits Phoenix | provable + fixed de-opt passes | best (CFGED) |

## Recommendation for Aletheia

**Chassis: Phoenix-style iterative semantics-preserving structural
analysis, in a new module `src/irstruct.rs`,** consuming the
`irlift::LiftedFunction` CFG (block ids, address-ordered) with
conditions taken from each block's terminating `Stmt::Branch` (post-SSA,
post-`irflow` simplification, so branch predicates are already folded).
Produce a structure tree (`Struct::Seq/If/Loop/Switch/Goto/Block`), not
text — pseudocode emission is a separate topic/slice.

Why this fits the repo contracts:

- **Proven vs heuristic.** Every schema collapse preserves semantics by
  construction; the only "heuristic" is *which* edge to virtualize when
  stuck, and that choice can never change meaning — only how many gotos
  appear. This is the same shape as `irflow`'s "when in doubt, do
  nothing." DREAM's condition synthesis and combing's duplication both
  put a heavy rewrite on the soundness-critical path; Phoenix does not.
  SAILR's evidence says we lose nothing real by tolerating gotos.
- **Total, no panics, resource-capped.** Terminating by edge-count
  argument; add an explicit iteration cap (like `cfg::Config`'s caps)
  with a `stats` flag, so a hostile graph degrades to more gotos, never
  to divergence. Deterministic edge choice: lowest (source-address,
  target-address) pair among candidate edges — same doctrine as `cfg`'s
  B-tree address-order worklists, so equal inputs give equal trees.
- **Total `check` and deterministic `render`.** `irstruct::check`
  re-verifies from scratch that the structure tree covers every
  reachable block exactly once (duplication is banned, so this is an
  exact partition), that every CFG edge is realized as either
  fall-through, structured construct, or explicit `Goto`, and that each
  `If`/`Loop` condition matches the block's branch statement. `render`
  prints the tree deterministically for golden tests, in the style of
  `irssa::render`.
- **Truncated and edge-less blocks surfaced honestly.** A block with
  `LiftedBlock::truncated`, or a `Terminator::Undecodable` /
  `Terminator::Truncated` / indirect-jump terminator (no successor
  edges by `cfg` contract), is a first-class leaf:
  `Struct::Opaque { block, reason }`, rendered as an explicit marker
  (e.g. `/* lift truncated: undecodable at 0x… */` /
  `/* indirect jump: successors unknown */`). Structuring must treat
  "no successors" as "control leaves the structured region here" —
  never invent a fall-through, never silently absorb the block into a
  loop body as if it looped. `SsaFunction::skipped` (unreachable
  blocks) are listed but not structured, matching `irssa`.
- **Irreducible regions:** no node duplication (combing) and no
  dispatch-variable rewriting (relooper) in the first slice — both are
  rewrites that would need their own soundness story and blow-up caps.
  Edge virtualization handles irreducibility with gotos, provably.

**Second slice (separate, later): SAILR-style de-optimizations** as
CFG-level pre-passes in front of the structurer — starting with the
single highest-value one, reverting jump-threading/cross-jump merges
(re-splitting a shared tail that two predecessors jump into, which *is*
controlled duplication, but of provably-identical statement lists —
checkable, bounded, and flagged in stats as a rewrite). Keep each
de-optimization its own commit with its own `check` obligations.

Schema catalog for slice one (all single-entry, matched on the dominator
tree / postorder like Phoenix): sequence, if-then, if-then-else,
self-loop, natural while/do-while loop with `break`/`continue` edges to
the loop's unique follow node, and `switch` only where `jumptable.rs`
proved the table (a proven jump table is the one place indirect-jump
successors exist).

## Open questions

- Where do branch conditions live once structured — re-rendered from the
  SSA `Branch` expression, or negated/combined forms materialized in the
  tree? (Interacts with the `pseudocode-emission` topic; proposal: store
  a block id + polarity, never a rewritten expression, in slice one.)
- Loop follow-node selection when a loop has multiple exit targets:
  Phoenix picks by heuristic; need a deterministic, documented rule
  (e.g. immediate post-dominator of the loop header if one exists, else
  most-frequent exit target, ties by address).
- Should `irstruct` consume `irssa::SsaFunction` directly or the
  pre-SSA `irlift` CFG? Structure only needs blocks + edges + branch
  predicates; taking `SsaFunction` gets simplified predicates and the
  `skipped` list for free, but couples slice order to SSA. Proposal:
  take the CFG plus an optional SSA overlay.
- How aggressively to normalize before matching (e.g. collapsing empty
  blocks, folding `br cond → br cond` chains) — each normalization is a
  small rewrite needing its own soundness note.
- CFGED-style fidelity metric: worth reimplementing a small
  graph-edit-distance harness for our own regression tests once source-
  available test binaries exist? (SAILR's is open source but we are
  dependency-free; a bounded exact GED on small graphs is feasible.)

## Sources

- Cifuentes, C. *Reverse Compilation Techniques.* PhD thesis, Queensland
  University of Technology, 1994.
- Sharir, M. *Structural analysis: a new approach to flow analysis in
  optimizing compilers.* Computer Languages 5(3–4), 1980.
- Schwartz, E. J., Lee, J., Woo, M., Brumley, D. *Native x86
  Decompilation Using Semantics-Preserving Structural Analysis and
  Iterative Control-Flow Structuring.* USENIX Security 2013.
- Yakdan, K., Eschweiler, S., Gerhards-Padilla, E., Smith, M. *No More
  Gotos: Decompilation Using Pattern-Independent Control-Flow
  Structuring and Semantics-Preserving Transformations.* NDSS 2015.
- Yakdan, K., Dechand, S., Gerhards-Padilla, E., Smith, M. *Helping
  Johnny to Analyze Malware.* IEEE S&P 2016 (DREAM++).
- Gussoni, A., Di Federico, A., Fezzardi, P., Agosta, G. *A Comb for
  Decompiled C Code.* AsiaCCS 2020.
  (paper: https://rev.ng/downloads/asiaccs-2020-paper.pdf)
- Basque, Z. L., et al. *Ahoy SAILR! There is No Need to DREAM of C: A
  Compiler-Aware Structuring Algorithm for Binary Decompilation.*
  USENIX Security 2024.
  (https://www.usenix.org/conference/usenixsecurity24/presentation/basque)
- Ramsey, N. *Beyond Relooper: Recursive Translation of Unstructured
  Control Flow to Structured Control Flow.* ICFP 2022.
- Zakai, A. *Emscripten: an LLVM-to-JavaScript compiler.* OOPSLA 2011
  (relooper).
