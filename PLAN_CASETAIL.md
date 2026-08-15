# Plan: SAILR inversion two — shared case tails (`irstruct`)

## Goal

The folding slice measured it: bash renders 58 real `switch`es but
their 620 case *bodies* still print as gotos, because shared case
tails and guard-to-default edges fail `try_switch`'s
single-predecessor/convergence test, and the first re-split inversion
(3734d37) only duplicates epilogue tails — `splittable_tail` demands
every remaining edge leave the function. Generalize the inversion:
duplicate tails that converge on one in-function target, so case
bodies structure and the gotos disappear. Same SAILR contract:
byte-identical duplication, capped, counted, monotone.

## The generalization

- `splittable_tail` (or a sibling the builder factors) additionally
  accepts a plain non-opaque leaf whose in-function exits all reach a
  single convergence target `t`: the duplicate leaf is
  `Seq[Block(va), Goto(t)]`, and the existing `tighten` machinery
  turns that `Goto` into `Break`/`Continue` where `t` is the switch
  follow or a loop boundary — reuse it, do not re-implement.
- Chains: a case body may be two or three blocks deep before the
  shared tail; duplicate bounded chains only if the single-block form
  already proves out and the chain case stays within the same cap and
  byte-identity obligations (builder's call whether chains land this
  slice or are recorded as the next increment — an honest smaller
  slice beats an unproven bigger one).
- The budget: one cap, shared with the epilogue inversion
  (`MAX_TAIL_SPLITS` — raise it only with evidence, e.g. the measured
  per-function distribution on bash, and say so). Degrade to gotos at
  the cap, never refuse.
- `check`'s duplication rule already exists (extra occurrences,
  byte-identical, within cap); extend its eligibility classifier in
  lockstep with the pass — the zero-duplication bit-for-bit guarantee
  must keep holding, now over the wider classifier.

## Edge copies, again (the subtle test)

A duplicated leaf now realizes an *outgoing* in-function edge
(tail → convergence target). Two duplicates of the same tail each
realize that same (pred, succ) edge on different textual paths — the
pseudo pending-set walk must place the edge's copies once per path,
none dropped, none doubled, no label left dangling. Build the φ fixture
that forces a residual copy on that edge and assert the placements,
as 3734d37's rider did for the epilogue case. `src/pseudo.rs` logic
must not change; if the walk cannot place a copy, the honest
`/* unplaced */` marker is the fallback, counted in the report.

## Module-by-module

- `src/irstruct.rs`: the classifier generalization, cap policy,
  `check` extension, module docs updated (this is inversion two;
  cite SAILR, state what still does not split — condition-carrying
  tails, opaque tails, multi-target convergence).
- `src/bin/redump.rs`: nothing unless the stats line needs it.
- Do NOT touch src/irssaopt.rs, src/irflow.rs, src/jumptable.rs,
  src/cfg.rs, src/aarch64*.rs — companion slices own them.
- `ROADMAP.md`: Current-thread entry with the measured numbers.

## Test matrix (~12)

1. a switch whose two cases share a tail converging on the follow:
   both cases structure, tail duplicated, gotos gone, `Break` spelled,
   stats and `check` agree.
2. guard-to-default: the bounds-check edge into the default body
   structures instead of goto-ing.
3. the φ/edge-copy fixture above.
4. a condition-carrying shared tail refuses (negative).
5. cap/degrade behavior; zero-dup bit-for-bit across the existing
   corpus; goto monotonicity regression stays green; determinism.

## Exit criteria (demonstrate, don't assert)

Measured on bash x86-64 (the binary with the 58 switches): case-body
gotos 620 → as low as honesty allows, whole-dump goto count down from
16,411 (combined-tree baseline), duplications spent and capped
functions counted, one real switch printed old-vs-new in the commit
message with structured case bodies. ls (277 gotos) and brotli arm64
(794) reported too. Zero check failures, byte-deterministic.

## Non-goals

- Jump-threading re-split (inversion three, its own slice).
- Any `try_switch` schema relaxation — the schema is right; the tails
  are what must yield.
- Cross-function tails.
