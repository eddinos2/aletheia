# Plan: the φ-web narrowing — coalescence-aware copy-freedom (`irstruct`)

## Goal

Jump threading's landed diagnosis: of bash's 6,657 condition-carrying
goto targets, 2,267 refuse on φ-copy edges, and the surviving 631
case-goto lines are φ-heavy — "the next lever there is
expression-level (φ-web narrowing or boolean merging), recorded, not
forced". Measured on the current tree, the lever is the narrowing:
`edge_copy_free` refuses any edge where a φ's argument is not the φ's
own name, but [`irout::out_of_ssa`] *coalesces* φ-webs — different SSA
names for one value end up one variable, and the edge's copy list is
empty. The renderer executes irout's `edge_copies`, not irstruct's
name-identity approximation; an edge absent from that map carries
nothing, and refusing it was overcaution, not soundness. Replacing the
approximation with the ground truth threads bash 13 → 166 sites and
buys back ~1,000 rendered gotos (16,185 → 15,161, measured on the
prototype), while ls stays byte-identical — the zero-change collapse.

## The narrowing

- `copy_edges(f)`: the key set of `irout::out_of_ssa(f).0.edge_copies`
  — computed once per `structure` (after the raw collapse, before the
  re-split rounds) and once per `check`. `out_of_ssa` is deterministic
  on the function, so pass, verifier, and renderer all see one truth;
  the classifier stays shared (`splittable_tail`, `threadable_head`,
  `tail_chain` take the set) and can never drift from `check`.
- `edge_copy_free(copies, from, to) := (from, to) ∉ copies`. Strictly
  narrower refusals than name-identity — same-name ends always share a
  variable, so every previously-free edge stays free — and exactly the
  downstream contract: the pseudo walk places an edge's copies at one
  textual site, so an edge *with* copies still refuses duplication.
- Nothing else bends: byte-identical duplication, the one shared
  `MAX_TAIL_SPLITS` budget, monotone goto buy-back, all-or-nothing
  per-target spellability, `check`'s condition honesty (`Undecided`,
  `Polarity`) — all unchanged. Zero-duplication runs stay bit-for-bit
  (ls proves it on a real binary).
- Cost: one extra `out_of_ssa` per `structure` and per `check` — the
  measured bash x86-64 `--decompile` wall time moves 24.2s → 25.4s.
  Honest, recorded, no caching layer invented for it.

## Module-by-module

- `src/irstruct.rs`: `copy_edges`, the narrowed `edge_copy_free`, the
  `copies` parameter through the three classifiers and
  `resplit_tails`, a `copies` field on the verifier, module docs
  (state the ground truth and what still refuses: edges whose copy
  list is non-empty, effectful heads, oversized heads, unspellable
  sites).
- Do NOT touch `src/irflow.rs`, `src/irssaopt.rs`, `src/fwd*.rs`,
  `src/aarch64*.rs`, `src/irout.rs` logic — companion slices own them
  this wave; `irout` is consumed read-only through its public API.
- `src/evalfx.rs`: FIXTURES rows updated in the same commit iff a
  metric legitimately moves (tail_merge's surviving pair is a
  plausible mover — its tails refused on the φ-copy edge).
- `ROADMAP.md`: the landed bullet with measured numbers.

## Test matrix (~8)

1. the narrowing fires: a diamond whose join φ takes two different
   names for one coalesced value — refused by name-identity, threaded
   or re-split under the ground truth, `check` passes, stats count it.
2. the narrowing still refuses: the same shape with genuinely
   interfering names (a real copy on the edge) — refused before and
   after, byte-identical trees.
3. both polarities of the threading φ-refusal fixture stay refused
   when the copy is real (the existing fixtures, re-pointed at the
   ground truth).
4. `edge_copy_free` unit: absent edge free, present edge not.
5. zero-duplication bit-for-bit; monotonicity; determinism (double
   structure runs equal).
6. the existing corpus of irstruct tests re-pointed at the new
   signatures, zero golden changes except where the narrowing
   legitimately fires (each one inspected and called out).

## Exit criteria (demonstrate, don't assert)

Measured on bash/ls x86-64 and brotli arm64, current-tree baselines
first, double runs byte-compared: rendered gotos, case-goto lines,
threaded sites, duplications, zero check failures. An exhibit
function old-vs-new. evalfx table either untouched or moved with the
mover named.

## Non-goals

- Boolean condition merging (a goto re-spelled because its target's
  condition is congruent to one already decided on the path) — that is
  `irflow`/`irssaopt` expression territory, recorded as the residue
  lever for a later slice, and this wave's irflow builder owns that
  file.
- Cross-cell copy coalescing in `irout` (its module docs record it as
  future work) — this slice consumes `out_of_ssa` as-is.
- Any relaxation of the byte-identity, budget, or honesty rules.
