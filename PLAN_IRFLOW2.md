# Plan: irflow patterns two — masked CCMP pairs + width-spelling equality (`irflow`)

## Goal

The two expression-level residues the last wave recorded, both owned
by `irflow`'s always-sound identity family:

1. **Masked CCMP pairs** (recorded by the a64 coverage slice): a
   decoded CCMP writes each flag as a select between the compare's
   flag expression and an imm4 bit, so a following conditional reads
   shapes like `(c & SF(a,b)) != (c & OF(a,b))` (and the select
   spellings the lift actually emits — take the real shapes from the
   brotli `--decompile` dump and from `aarch64_lift`'s CCMP lift, not
   from this sketch). When both sides carry the *same* guard and the
   inner pair is one of the recovered order patterns, the composition
   folds to the guarded relation. Measure first, then pattern: the
   slice starts by dumping and classifying the actual shapes on
   brotli (51 occurrences at the last count) and implements the
   patterns that cover the mass, refusing the rest honestly.
2. **Width-spelling equality**: both condrec and the forwarding
   slice recorded refusals where the paired operands are the same
   value spelled differently through W32 chains — `zext`/`sext`/
   `trunc` compositions that are provably equal but not structurally
   equal. Add the sound normalization identities (`trunc.d(zext.q(x))
   → x`-class, width-exact, always-sound only) so structural equality
   sees through them, and/or a width-aware equality helper used by
   the pair matcher; either way every identity is a two's-complement
   theorem, proved the house way.

## Soundness

The width-8 exhaustive oracle extends to every new pattern and every
normalization identity: all 65,536 operand pairs (and for the
guarded patterns, both guard values), folded result vs literal
computation, every polarity. Negative tests: different guards on the
two sides, near-miss widths, sign-vs-zero extension mismatches that
are NOT equal (e.g. `sext` vs `zext` of a possibly-negative value) —
refused. When in doubt, do nothing.

## Module-by-module

- `src/irflow.rs`: the patterns, the normalization identities, module
  docs extended in the existing list style, the oracle extensions.
- `src/irssaopt.rs`: docs-only touch if its coverage paragraph needs
  the update; no algorithm change.
- Do NOT touch src/irstruct.rs, src/aarch64*.rs, src/jumptable.rs,
  src/cfg.rs, src/pseudo.rs — companion slices run in parallel.
  Test goldens elsewhere update only where output honestly improves;
  call each out.
- `ROADMAP.md`: Current-thread entry with measured retirements.

## Test matrix (~14)

1. each masked-pair pattern golden (from real lifted CCMP sequences,
   e2e through construct → forward → fold), both polarities.
2. each normalization identity golden, plus its exhaustive oracle.
3. the negative set above; determinism; existing suite green.

## Exit criteria (demonstrate, don't assert)

Measured on the current tree first (baselines moved last wave —
re-derive, do not reuse): brotli arm64 paired/masked shape counts
before → after with the same grep method stated; bash/ls x86-64
paired counts (the W32 class should retire some of bash's remaining
90 inline pairs); byte-deterministic, zero check failures, and one
real chained-condition function (a `ccmp`-using brotli function)
old-vs-new in the commit message. Report what remains, classified.

## Non-goals

- Boolean recomposition across branches (`if (A && B)` from two
  Ifs) — structuring/expression-merging territory, not folding.
- Any forwarding-policy change (last wave's slice stands).
- New IR operators.
