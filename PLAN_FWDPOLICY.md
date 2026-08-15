# Plan: forwarding policy — splice multi-consumer flag defs (`irssaopt`)

## Goal

Retire the residual paired flag shapes the order-condition slice
(4a9793b) measured and diagnosed: when one `cmp` feeds two jccs, the
11-node OF tree has two uses and `forward`'s tier rule (compound pure
trees over `FWD_SMALL_NODES` forward only when single-use) refuses the
splice — so `irflow`'s pair patterns never see the pair in one
expression. ~20% of order conditions stay unrecovered, the milestone
ls comparator `sub_10000073f` among them. The DREAM++ concern behind
the cap (duplication hurts readability) does not apply here, because
the spliced tree immediately *folds to three nodes* (`a <s b`). Make
the policy see that.

## The policy change (and its two obstacles)

Splice-when-the-fold-shrinks: a compound, pure, load-free,
division-free definition may forward into *multiple* uses when the
use-site expression, after `irflow` folding, is no larger than it was
before the splice (strictly smaller for the def to also be swept —
builder picks the exact comparison and documents it; the requirement
is that no use site ever gets textually bigger and the decision is
deterministic, made per-def from the folded results, not a heuristic).
Tentative-substitute-then-fold-then-decide is the natural mechanism;
it must be bounded (existing node caps) and total.

Obstacle two, from the same diagnosis: the flag def's tree may read
temp-space names defined in its own block, which a use in *another*
block cannot legally read (`ir::check`'s block-local temp rule).
Handle it soundly — either splice the whole pure def-cone transitively
(temps' own defs folded in first) or refuse that def; never emit a
cross-block temp read. `irssa::check`/`ir::check` green is the
arbiter, as always.

Loads and division stay non-duplicable, unconditionally — the barrier
semantics are untouched; this slice only widens the *pure* tier.

## Module-by-module

- `src/irssaopt.rs`: the policy in `forward`, module docs updated
  where the tier rule is stated (the DREAM++ paragraph gains its
  fold-shrinks exception, cited honestly), `FwdStats` counts the
  multi-use splices. No other pass changes.
- Goldens elsewhere (redump/pseudo/irflow tests) update only where
  output honestly improves; list each in the report. Do NOT change
  logic in src/irflow.rs, src/irstruct.rs, src/jumptable.rs,
  src/aarch64*.rs — companion slices own them this wave.
- `ROADMAP.md`: Current-thread entry with the measured retirement.

## Test matrix (~12)

1. the two-consumer `cmp` fixture (one flag set, two jccs): both
   branch conditions read as relations after forward + fold.
2. the cross-block temp cone: spliced legally or refused, checks green
   either way; a hand-built case of each.
3. refusal: a multi-use def whose fold does NOT shrink stays put
   (byte-identical output to today).
4. loads/division never duplicated (negative fixtures).
5. determinism; stats count the splices; existing suite green with
   honestly-updated goldens only.

## Exit criteria (demonstrate, don't assert)

`--decompile` on ls/bash x86-64 and brotli arm64: the milestone
comparator prints as plain relational `if`s (old vs new in the commit
message); the paired-shape counts from 4a9793b's measurement method
drop (ls 20 / bash 362 / brotli 36 are the baselines); and total
pseudocode byte size per binary is reported before/after — the
readability cap exists for a reason, prove the output did not balloon.
Byte-deterministic, zero check failures.

## Non-goals

- Any new `irflow` pattern (they exist; this slice feeds them).
- Widening the load-bearing tier or barrier semantics.
- General expression-size optimization.
