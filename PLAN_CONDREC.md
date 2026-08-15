# Plan: order-condition recovery — the paired flag shapes (`irflow`)

## Goal

Retire the documented gap in `irssaopt`'s module docs ("What the
equality family does not cover"): the x86 signed-order conditions lift
to `SF ^ OF` shapes that survive forwarding as paired flag expressions
— visible in the milestone ls comparator as

    (v10 - v8 <s 0x0.q) == (((v10 ^ v8) & (v10 ^ v10 - v8)) <s 0x0.q)

— because collapsing them to `a <s b` needs the *pair* of flag
definitions recognized together, not a single-operator rewrite. This
slice adds that pairwise recognition to `irflow::fold_expr`'s
always-sound identity family, where the equality identities already
live, so it serves `--simplify`, `irssaopt::forward`, and therefore
`--decompile` in one place.

## The shapes (authority: `x86_lift::cond_expr` — read it first)

With `SF = (a - b) <s 0`, `OF = ((a ^ b) & (a ^ (a - b))) <s 0`,
`ZF = (a - b) == 0` (usually already folded to `a == b`), `CF = a <u b`:

- `SF != OF`               → `a <s b`          (jl)
- `SF == OF`               → `b <=s a`         (jge)
- `ZF | (SF != OF)`        → `a <=s b`         (jle)
- `!ZF & (SF == OF)`       → `b <s a`          (jg — however the lift
  actually spells the conjunction; take the real shapes from
  `cond_expr`, not this sketch)
- `CF | ZF`                → `a <=u b`         (jbe)
- `!CF & !ZF`              → `b <u a`          (ja)

plus whatever negation polarities `Branch` and `UnOp::Not` produce.
The existing `BinOp` set (`Slt Sle Ult Ule`) spells every result —
no new IR operator. Match rules, all mandatory:

- **Shared operands, structurally**: the `a` and `b` subtrees must be
  equal by the crate's expression equality, width-exact, on every
  occurrence in the pattern. Never rewrite on a near-miss.
- **Width-exact throughout**: the `<s 0`, the `- `, the `^`/`&` all at
  one width; a mixed-width pairing does not fold.
- Bounded by the existing `MAX_EXPR_NODES` discipline; matching is
  structural and total, no search.
- Constants included: after propagation `b` is often a literal — the
  patterns must match expression trees, not registers.

## Module-by-module

- `src/irflow.rs`: the pattern family in `fold_expr` (or a sibling
  helper it calls), module docs extended in the existing style —
  list the pairs exactly as the equality family is listed.
- `src/irssaopt.rs`: rewrite the "What the equality family does not
  cover" paragraph into "covered since" documentation pointing at
  `irflow`; no algorithm change (forwarding already puts the shapes
  in one expression — that was the design).
- Goldens: any existing test in `irflow`/`irssaopt`/`redump` whose
  expected output contains the old paired shapes updates honestly to
  the collapsed form. Do NOT change logic in `src/pseudo.rs`,
  `src/irstruct.rs`, `src/cfg.rs`, `src/jumptable.rs` (companion
  slices own them this wave); if one of their *test goldens* encodes
  the old shape, updating that golden text is permitted and must be
  called out in the report.
- `ROADMAP.md`: Current-thread entry; note the ls comparator now
  reads as real relations.

## Soundness

Each identity is a theorem of two's-complement arithmetic at fixed
width. Prove them the way this codebase proves such things: an
exhaustive oracle test at width 8 (all 65,536 (a, b) pairs, every
pattern, comparing the folded relational op against the literal flag
computation) — cheap, total, and stronger than any argument. When in
doubt, do nothing: the negative tests matter as much as the positive.

## Test matrix (~16)

1. each pattern folds at each width (goldens through lift → forward →
   fold for real jl/jge/jle/jg/jb/jbe/ja sequences).
2. the width-8 exhaustive oracle over every pattern and polarity.
3. negative: different `a`/`b` subtrees, mismatched widths, one flag
   from a different subtraction — all left exactly as-is.
4. negation polarities (`!` from a negated branch) fold to the
   inverted relation.
5. determinism; existing suite untouched except honestly-updated
   goldens.

## Exit criteria (demonstrate, don't assert)

`redump --decompile` on the /bin/ls x86-64 slice: the milestone
comparator `sub_10000073f` reads as plain relational `if`s (print it
in the commit message next to the old shape), byte-deterministic, and
a measured count: how many `<s 0` flag-pair shapes remain in the full
ls/bash pseudocode dump before vs after. aarch64 note: the arm64 lift
compares via the same NZCV plumbing — report (measure, do not fix)
whether its condition shapes need a companion pattern, as input for a
follow-up.

## Non-goals

- New IR operators; condition *simplification* beyond these proven
  identities (De Morgan pushing stays out of the printer AND out of
  scope here).
- Touching the structurer, the printer, or CFG recovery.
- aarch64-specific patterns (measured and reported only).
