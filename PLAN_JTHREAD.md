# Plan: SAILR inversion three — jump threading (`irstruct`)

## Goal

The case-tail slice measured what remains: the surviving bash case
gotos (and much of the general goto residue) target
*condition-carrying* blocks — no linear duplicate exists, so both
prior inversions refuse by design. That is jump threading's
signature (SAILR): the compiler routed several predecessors through a
shared conditional block, welding regions together. The inversion:
duplicate the small conditional block itself into a goto-ing
predecessor, so each copy has one predecessor and the schemas match.
This is inversion three, still controlled duplication — but now the
duplicate is an `If` head, not a leaf, so the contract needs one new
clause.

## The contract extension

- What may duplicate: a block whose statements are pure/cheap by a
  budget (the builder derives it from the corpus and documents it —
  the SAILR spirit is a *small* threaded condition, not a body),
  ending in a conditional branch whose both targets are already
  realized elsewhere or become gotos. The duplicate materializes as
  the real `If { cond: Cond { block: <the-copy>, .. } }` — and here is
  the clause: conditions are (block, polarity) references, so a
  duplicated deciding block must stay *referenceable*. Whatever
  identity scheme the first two inversions used for duplicate leaves
  extends to deciding duplicates; `irstruct::check`'s condition-honesty
  rules must hold on the copies exactly as on originals.
- Byte-identical statement list at split time, asserted — unchanged.
- One shared budget with the existing inversions (`MAX_TAIL_SPLITS`,
  currently 32); a threading split spends from the same pool. Only
  splits that remove at least one goto; goto monotonicity stays a
  regression guard; zero-duplication runs stay bit-for-bit.
- Edge copies: a duplicated conditional realizes *two* outgoing
  edges. The refusal rule from inversion two (an edge whose φ demands
  a residual copy refuses the split) applies per-edge here; build the
  φ fixture that forces each polarity.

## Ordering with the other inversions

Threading runs where the epilogue/case-tail passes leave gotos —
after them in the same round loop, so a thread that exposes a fresh
linear tail lets the next round's cheaper inversion take it (and
vice versa). Termination: the shared budget bounds total duplication;
document the loop order and why it cannot oscillate.

## Module-by-module

- `src/irstruct.rs`: the inversion, its eligibility classifier,
  `StructStats` field (`threaded` or the builder's spelling), `check`
  extension for deciding-block duplicates, module docs (inversion
  three; cite SAILR; state what still refuses — big blocks, effectful
  statements, φ-copy edges).
- `src/bin/redump.rs`: stats-line spelling only if surfaced; minimal.
- Do NOT touch src/irflow.rs, src/irssaopt.rs, src/aarch64*.rs,
  src/jumptable.rs, src/cfg.rs, and no src/pseudo.rs logic changes
  (tests there fine) — three companion slices run in parallel.
- `ROADMAP.md`: Current-thread entry with measured numbers.

## Test matrix (~14)

1. the canonical threaded shape: two predecessors goto a shared
   `if`-block; after inversion both structure, gotos gone, stats
   count it, `check` passes including condition honesty on the copy.
2. a threaded block whose copy exposes a linear tail the case-tail
   inversion then takes (the composed round).
3. refusals: effectful statements, over-budget statement list,
   φ-copy-demanding edge (both polarities), budget exhaustion
   degrades to gotos.
4. zero-duplication bit-for-bit across the existing corpus;
   monotonicity guard; determinism; malformed-input posture.
5. a pseudo rider: the duplicated condition renders correctly at both
   sites (negation polarity included).

## Exit criteria (demonstrate, don't assert)

Measured on bash/ls x86-64 and brotli arm64 (`--structure` +
`--decompile`, current-tree baselines re-measured first): goto and
case-goto deltas, threads spent, zero check failures,
byte-deterministic, one real threaded function old-vs-new in the
commit message. If the corpus yields little (possible — clang may
thread less than gcc), the honest null result with synthetic evidence.

## Non-goals

- Boolean condition *merging* (`if (A && B)` reconstruction) — that
  is expression-level work owned by irflow/irssaopt, not duplication.
- Switch-lowering inversion (a later slice).
- Any relaxation of the byte-identity or budget rules.
