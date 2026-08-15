# Plan: value-numbering-grade equality for the pair folds (`irflow`)

## Goal

The patterns-two slice diagnosed its residue to the root: surviving
comparison pairs whose halves reach the comparison through *different
SSA names for one value* — bash's `trunc.d(v204) + 1` against the
spliced sum — which structural equality correctly refuses. Measured
on the current tree (grep methods stated in the exit criteria), bash
keeps 100 two-`<s 0x0` pair lines and the classified samples show two
compounding causes:

1. **Name-splitting**: one half reads a full-width SSA name whose
   *definition* is the very tree the other half carries spliced —
   forwarding's duplication cap or width rules kept the name standing.
2. **Spelling-splitting**: even resolved, the halves spell one value
   two ways — the 64-bit definition spells `trunc.d(x + y)` where the
   32-bit lift spelled `trunc.d(x) + trunc.d(y)` — trunc-of-sum
   against sum-of-truncs.

The slice gives the pair matchers an equality *witness* for both:
two expressions are equal when they are congruent under (a) each SSA
name's unique definition and (b) the truncation homomorphism, both
directions of proof, no heuristic anywhere.

## Design

- **`VnDefs`** (new, `irflow`): an owned map from the exact defining
  [`Reg`] (space, name id, width — so a narrower read never resolves)
  to its *folded* right-hand side. Insertion is gated inside `irflow`
  — one doctrine, one place: load-free, division-free (a `veq` witness
  never reasons through memory or a trap), node-capped. `irssaopt`'s
  `forward` builds it once per round from the function's `Assign`s; a
  φ has no right-hand side and is naturally absent (φ-congruence is
  recorded as a deferred increment, not attempted). Non-φ SSA defs
  form a DAG, so resolution terminates; a fuel cap refuses hostile
  depth anyway — a refusal is a `false`, never a guess.
- **`veq(a, b, vn)`** (new): structural `==` fast path (byte-for-byte
  today's behavior), else compare canonical keys: resolve exact-width
  reads through `VnDefs`, push [`UnOp::Truncate`] through
  `Add/Sub/Mul/And/Or/Xor` and `Neg/Not` (truncation is a ring
  homomorphism — each op's theorem proved by the width-8/16 exhaustive
  oracle), cancel the width respellings through the existing
  `fold_width_identity`, fold constants. Shifts, divisions, and
  comparisons are *not* distributed over (the shift amount is taken
  modulo the width — a distribution would be unsound and has a
  negative test).
- **Threading**: `fold_expr`/`fold_stmt` keep their signatures
  (context `None`); new `fold_stmt_vn` is called from exactly
  `forward`'s `rewrite_stmt` — the real round and the fold-shrinks
  tentative share it, so they can never disagree. The context reaches
  only the pair-equality gates: `order_pair_operands`,
  `overflow_flag_operands`' internal occurrence equalities,
  `order_compose_ok`, `masked_order_pair`'s guard match, and
  `is_complement`. Every other structural `==` in the fold
  (`x - x → 0`, `x & x → x`, …) is deliberately untouched — recorded
  as deferred, not smuggled in.
- **Soundness of the rewrite**: matchers keep returning subtrees of
  the statement itself, so no resolved tree is ever *emitted* — `veq`
  only proves the dropped duplicate names the same value. The kept
  operands keep their existing `contains_load` gates, and congruence
  preserves load-freeness (resolution refuses load-bearing defs;
  structural sub-matches share their trees — stated as an induction in
  the module docs), so a load-bearing duplicate can never be dropped
  against a load-free kept copy.

## Module-by-module

- `src/irflow.rs`: `VnDefs`, `veq`/`vkey`, the trunc-distribution
  theorems, context threading through `fold_rec` →
  `fold_binary_identity`/`fold_unary_identity` → the pair matchers;
  `contains_div` moves here beside `contains_load` (one predicate,
  one doctrine — `irssaopt` re-uses it); module docs extended in the
  existing list style.
- `src/irssaopt.rs`: `fwd_round` builds the `VnDefs`, `Fwd` carries
  it, `rewrite_stmt` folds through `fold_stmt_vn`; its private
  `contains_div` deleted in favor of the shared one. No policy
  change: what forwards is decided exactly as before, only the
  re-fold sees more.
- Do NOT touch `src/irstruct.rs`, `src/aarch64*.rs`, `src/pseudo.rs`,
  `src/x86*.rs` — companion slices run in parallel.
- `src/evalfx.rs`: FIXTURES updated in this commit iff a metric
  legitimately moves, per its charter.
- `ROADMAP.md`: landed-slice bullet with measured retirements.

## Test matrix (~12)

1. Trunc-distribution oracle per op: all 65,536 W16 pairs truncated
   to W8, `trunc(a op b) == trunc(a) op trunc(b)`, plus `Neg`/`Not`
   over all 256; the shift near-miss refused (negative).
2. `veq` units: name-witness positive; narrow-read, load-backed def,
   div-backed def, unequal-value, and `sext` vs `zext` negatives.
3. e2e SSA fixture through `forward`: a pair whose halves only fold
   via the witness (name kept by the duplication cap on one side,
   sum-of-truncs spelling on the other) collapses to the relation;
   `check_preserved` green; determinism (double run byte-equal).
4. Existing suite green; goldens that change do so only where output
   honestly improves — each called out.

## Exit criteria (demonstrate, don't assert)

Measured on the current tree first, same grep stated with the number:
bash x86-64 two-`<s 0x0` pair lines (100 today) and `') <s 0x0'`
lines (908 today) before → after; brotli arm64 (54 / 59 today) and
ls (0) the same way; `--decompile=100000` double runs byte-identical,
zero check failures, `cargo test evalfx` green with any legitimate
table move in the same commit. One real bash function's condition
old-vs-new in the commit message. Report what remains, classified.

## Non-goals

- φ-congruence (deferred, recorded).
- Commutative/associative normalization inside `veq` (deferred).
- VN for the self-identities (`x - x`, `x ^ x`, `x & x`, `x | x`).
- The load-backed flag-operand splice (a sibling slice owns it).
- Any forwarding-policy change; any new IR operator; any change to
  what is *emitted* — the witness proves, the existing folds rewrite.
