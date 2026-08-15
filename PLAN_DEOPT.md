# Plan: SAILR de-optimization pre-pass — re-split merged tails (`irstruct`)

## Goal

DESIGN.md slice 18. Most *spurious* gotos in structured output are the
residue of compiler transforms — jump threading, cross-jumping,
tail-merging (SAILR; Basque et al., USENIX Security 2024). The
highest-value inversion is re-splitting a shared tail that two or more
predecessors jump into: give one predecessor its own copy of the tail
and the goto disappears. Genuine source gotos survive untouched. This
is controlled duplication of provably-identical statement lists —
checkable, bounded, and flagged — never invention.

## Where it runs and what it rewrites

A pre-pass inside `irstruct`, in front of schema matching, on the
structurer's own view of the function — the `SsaFunction` is analysis
truth and is NEVER mutated (the standing rule since `irout`). The
natural representation: the structure tree may reference the same
block VA from more than one `Node::Block` leaf — duplication is tree
shape, not block storage. The builder chooses the exact mechanism
(pre-pass on the region graph vs duplication-aware virtualization
during collapse) and documents the choice; the requirements are only:

- A duplicated leaf renders the block's own statements — byte-equal IR
  to the original at the time of the split, asserted, not assumed.
- Each duplication is counted in `StructStats` with a total cap
  (SAILR's bound); at the cap the pass stops and remaining edges take
  the existing goto path — degrade, never refuse.
- Only tails that *remove a goto* are split (the pass runs where the
  structurer would otherwise virtualize an edge, or provably would);
  a split that saves nothing is not made.
- Edge copies stay honest: `irout` keys copies by (pred, succ) edge,
  and a duplicated leaf realizes a *different* incoming edge than the
  original — the `pseudo` pending-set walk must still place every
  copy exactly once per realized edge. This is the subtle interaction;
  test it directly (a φ-carrying merged tail with a residual copy).

## `check` and the partition invariant

`irstruct::check` today demands an exact partition — every reachable
block exactly once. Duplication knowingly relaxes that: `check` gains
a duplication-aware rule — every reachable block at least once, each
extra occurrence recorded in stats with its source, count within cap,
and the duplicated leaves' blocks byte-identical to their originals.
The relaxation is scoped: zero duplications must mean the old exact
partition, bit for bit, so every existing test keeps its meaning.

## Module-by-module

- `src/irstruct.rs`: the pre-pass, the `StructStats` fields, the
  `check` extension, module docs in the house style (cite SAILR; state
  the cap, the byte-equality obligation, and the goto-monotonicity
  claim). Do NOT touch `src/cfg.rs`, `src/jumptable.rs`,
  `src/irflow.rs`, `src/irssaopt.rs` (companion slices own them this
  wave). `src/pseudo.rs` logic unchanged; add the edge-copy rider
  test wherever the existing cross-module tests of that seam live.
- `src/bin/redump.rs`: only if `--structure`'s stats line should show
  the duplication count (it should — mirror the goto count's
  spelling); keep the diff minimal.
- `ROADMAP.md`: Current-thread entry with measured goto deltas.

## Test matrix (~14)

1. a tail-merged diamond (two predecessors jumping into a shared
   tail): goto disappears, tail duplicated once, stats say so,
   `check` passes.
2. a genuine unstructurable goto (irreducible fixture from the
   existing suite): survives, zero duplications.
3. the φ/edge-copy rider: a merged tail whose φ demands a residual
   copy — each realized edge gets its copy exactly once in the
   pseudocode, none dropped, none doubled.
4. cap behavior: corpus forcing the cap → degrade to gotos, stats
   flag, no refusal.
5. zero-duplication runs byte-identical to today's output (the
   scoped-relaxation guarantee) across the existing fixture set.
6. goto count monotonically non-increasing across every fixture
   (regression guard per DESIGN).
7. determinism; malformed-input posture unchanged.

## Exit criteria (demonstrate, don't assert)

Measured on the /bin/ls and /bin/bash x86-64 slices through
`--structure` and `--decompile`: gotos before → after (ls is 287
today), duplications spent, zero check failures, byte-deterministic,
and one real function whose spurious goto became straight-line code —
shown in the commit message, old vs new. If the corpus shows no
tail-merge shapes (possible — clang may not tail-merge here), say so
with the synthetic fixture as the evidence and the real-corpus numbers
as the honest null result.

## Non-goals

- The other SAILR inversions (jump-threading re-split, switch
  lowering) — each is its own future commit per DESIGN.
- Any structuring-schema change; any `SsaFunction` mutation.
- Duplicating non-identical or condition-carrying blocks.
