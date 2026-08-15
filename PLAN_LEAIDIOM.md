# Plan: jump-table idiom — the split-block `lea` shape (`jumptable`)

## Goal

/bin/ls x86-64 proves zero tables: its dispatch splits the idiom
across blocks — the `lea rB, [rip+table]` (and sometimes the
`movsxd`) sits in a predecessor of the block holding the `jmp`, so
the current single-block matcher never sees the base register's
definition. Extend the matcher to resolve tracked register
definitions across block boundaries, keeping the standing proof
discipline: an exact idiom match with a validated table, or no table
at all.

## The mechanism

A bounded backward def-walk: when the in-block scan reaches the
block's top with the base (or index-transform) register still
unresolved, continue in the predecessor — only along
single-predecessor edges (dominance-safe by construction), only
through a small bounded number of blocks and instructions, stopping
dead at any intervening write to a tracked register, any call, or
anything the current matcher already treats as a barrier. The
existing idiom set is unchanged — same four shapes, same table
validation (`min_targets`, executable-region, entry-count caps);
only *where their instructions may sit* widens. The bounds-check
discovery already looks at a previous block; unify with that walk
rather than growing a second one, if the code allows it cleanly.

## Soundness

- Or-nothing per idiom: a partial cross-block match proves nothing.
- The def-walk must be deterministic and bounded (constants documented
  like the existing caps); no fixpoint, no search over multiple
  predecessor paths — one single-pred chain or refusal.
- `cfg::recover_with_tables` consumes the map unchanged; folding
  correctness (in-function targets, executable regions) already
  guards downstream. Do not touch `src/cfg.rs`.

## Module-by-module

- `src/jumptable.rs`: the cross-block walk, module docs updated (the
  idiom sketches gain the split-block variants; the caps documented),
  tests.
- Do NOT touch src/cfg.rs, src/irstruct.rs, src/irssaopt.rs,
  src/irflow.rs, src/aarch64*.rs — companion slices own them.
  src/bin/redump.rs only if a golden honestly moves; call it out.
- `ROADMAP.md`: Current-thread entry with the measured yield.

## Test matrix (~10)

1. each of the two `lea`-based idioms with the `lea` one block up:
   proves, targets exact (synthetic fixtures with real encodings).
2. `lea` two blocks up a single-pred chain: proves; past the bound:
   refuses.
3. a clobber of the base register between `lea` and `jmp`: refuses.
4. a multi-predecessor join between `lea` and `jmp`: refuses.
5. the existing single-block corpus: byte-identical results (the
   widening is strictly additive).
6. determinism; the resolve_folded fixpoint still terminates and
   its rounds/caps behave with the new matches.

## Exit criteria (demonstrate, don't assert)

/bin/ls x86-64: tables proven 0 → N (report N and the dispatch
functions), `--structure`/`--decompile` render its switches (show one
in the commit message), opaque(indirect) and goto deltas reported,
zero check failures, byte-deterministic. bash re-measured (its 37
tables must not regress; new split-block matches counted separately).
arm64 spot check on libbrotlidec: unchanged or improved, reported.

## Non-goals

- New idiom families (MSVC image-base, ARM `tbb/tbh`-style compact
  tables) — recorded if observed, not built.
- Multi-path or dataflow-general table discovery.
- Any folding-machinery change.
