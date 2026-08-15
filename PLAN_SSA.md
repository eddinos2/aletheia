# Plan: pruned SSA construction over the lifted IR CFG (`src/irssa.rs`)

## Input / output

Input: `irlift::LiftedFunction` (blocks keyed by start VA, each a checked
`Vec<ir::Stmt>` plus successors and an honest `truncated` flag). Output: a new
`SsaFunction` in pruned SSA form, plus a total `check` and a deterministic
`render`, and a `redump --ssa[=N]` subcommand.

## Representation (the key decision)

SSA statements REUSE `ir::Stmt`/`ir::Expr` unchanged: renaming rewrites only
`Reg::num`, which becomes an SSA *name id* indexing a side table
`names: Vec<Name>` mapping id -> (space, cell, version, width). This buys:

- `ir::check` runs unchanged on every renamed block (width discipline for free),
- rendering reuses the IR expression printer via a new `ir::render_with`
  (a whole-`Reg` namer hook; the existing `render` becomes a wrapper),
- no parallel expression type to maintain.

Cost: at most 65536 SSA names per function (`num` is `u16`). Exceeding the cap
is an explicit `Unrepresentable::TooManyNames` error, never a silent wrap.

Phis are a separate per-block list `Phi { dst: name id, args: [(pred, id)] }`,
where a pred of `None` labels the virtual function-entry edge (needed only when
the entry block has real predecessors, e.g. a loop back to entry).

## Versioning semantics (aliasing doctrine)

The IR models `eax`/`rax` as one cell at different widths. Matching `irflow`'s
conservative doctrine, SSA versions are per *cell* (space, num): a def at ANY
width starts a new version of the whole cell. Each name carries the width its
def wrote — the bits it is guaranteed to carry. A use at width <= the def width
is an exact dependence (sub-register read of low bits); a use *wider* than its
reaching def reads bits the def never wrote and is recorded honestly in a
`partial` list (rendered as a note, verified exactly by `check`). The x86
lifter always writes GPRs at W64 and flags at W1, so `partial` is empty on all
lifted code; only hand-built IR can populate it.

Version 0 of a cell is the implicit at-entry value (parameter / caller state),
created lazily on first unresolved use or phi argument, listed as `live_in`.
Its width is the full cell (W64; W1 for flags) since the entry state defines
every bit. A phi's width is the minimum of its argument widths (the bits all
inputs guarantee), computed by a decreasing fixpoint across phi-to-phi chains.

Deliberate non-goal (documented): calls' runtime clobbers are NOT modeled —
def-use links reflect the IR statements as written, exactly like
`irflow::liveness`. ABI-aware call effects are a future slice.

## Algorithms

1. Reachability from entry (successor edges filtered to blocks present in the
   function; out-of-function edge targets stay in `successors` for display).
   Unreachable blocks are excluded and listed in `skipped` (rendered note).
2. Dominators: Cooper–Harvey–Kennedy iterative idom over reverse postorder.
   The entry's idom is virtual (absent), which makes the Cytron/Cooper
   frontier walk correct even for a self-looping entry.
3. Dominance frontiers: per-edge runner walk up the idom chain.
4. Global liveness for pruning: per-block `irflow::live_in` (a new one-line
   public helper folding `irflow`'s existing per-statement transfer) iterated
   to a backward fixpoint over the CFG, then projected to cells. Exact-ref
   kills make this a sound over-approximation of cell liveness, so pruning
   never drops a needed phi (a rare extra phi is possible, and harmless).
5. Pruned phi insertion: classic Cytron worklist per cell over the frontiers,
   gated on cell-liveness at the join; the entry block counts as a def site of
   every cell (the implicit version 0).
6. Renaming: DFS over the dominator tree (children in address order), a stack
   of names per cell; phi defs at block entry, per-edge phi-argument fill from
   each predecessor's exit stacks; statement uses read pre-statement versions.

Everything iterates B-tree maps or sorted vectors: deterministic by
construction. All expression walks are depth-bounded; input blocks are
pre-validated with `ir::check` (a failing block is `Unrepresentable::
MalformedBlock` — redump never produces one, tests can).

## `check` invariants (total, never panics)

- every renamed block passes `ir::check` (widths, address/cond widths, caps);
- every `Reg::num` in stmts/phis indexes `names` and agrees on `space`;
- exactly one def per name: phi, assign, or intrinsic write — or version 0
  (entry), never both; (cell, version) pairs are unique;
- def occurrences carry exactly the name's width; use occurrences carry width
  <= the name's width, or their position is in `partial` — which must equal
  the recomputed set exactly (no missing, no stale entries);
- every use is dominated by its def: same block earlier (phis precede
  statements; intra-statement uses precede the statement's defs), or the def's
  block strictly dominates; a phi argument's def must dominate the
  predecessor's exit; version 0 dominates everything;
- phi shape: one argument per predecessor edge (None first, then ascending
  VA), argument cells match the phi's cell, phi width <= every argument width;
- structure: entry present (unless the function is empty), every stored block
  reachable, `live_in` = exactly the version-0 names, sorted.

## Reuse vs add

- Reuse: `ir::Stmt`/`Expr`/`check`, `irflow`'s liveness transfer (via the new
  `live_in` helper), `irlift::LiftedFunction` as the input contract, the
  existing render style and `x86_lift::reg_name` naming hook.
- Add: `ir::render_with` (backward-compatible generalization), `irflow::
  live_in`, the new `irssa` module (dominators, frontiers, pruned phi
  insertion, renaming, check, render), `redump --ssa[=N]`.
- `cfg.rs` has no dominator code today; it lives in `irssa` with the rest of
  the SSA machinery (a future structuring slice can lift it out).

## Test matrix

- straight-line block: no phis, sequential versions, golden render;
- diamond (def in both arms, use at merge): exactly one phi at the merge with
  correctly-labeled args; dead-at-merge cell: no phi (pruning);
- def in one arm only: phi merges the def with version 0;
- loop: phi at the header; self-looping entry: virtual-entry phi arg;
- live-in: a use with no def yields version 0, listed and rendered;
- partial-def read (def `al`, use `rax`): recorded, checked, render-noted;
- hand-broken SSA rejected: duplicate def, un-dominated use, wrong phi args,
  width violation, stale `partial`;
- truncated blocks keep their flag; unreachable blocks skipped and noted;
- malformed input block -> `MalformedBlock`; name overflow -> `TooManyNames`;
- determinism: build and render twice, byte-equal; end-to-end from a synthetic
  image through `cfg::recover` + `irlift::lift_function`;
- bounded seeded sweep (xorshift64*, matching the repo's style): random small
  CFGs of well-formed blocks -> construct -> `check` always Ok, no panic;
- redump: `--ssa` flag parse tests + an end-to-end dump containing phis.
