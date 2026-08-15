# Plan: fold proven jump tables into CFG recovery (`cfg`)

## Goal

`cfg::recover` under-approximates indirect jumps: `Terminator::
IndirectJump` contributes no successors, so every jump-table dispatch
block reaches `irstruct` as `Opaque { IndirectJump }` and the `Switch`
schema is exercised by synthetic tests only (the honest limit recorded
in `irstruct`'s module docs). `jumptable::resolve` already *proves*
table targets. This slice folds them in: a dispatch block's proven
targets become its CFG successors, the successors clone through
`irlift`/`irssa` untouched (they already copy `block.successors`), and
real binaries exercise `Switch` end to end through `--structure` and
`--decompile`.

## The real problem is reachability, not plumbing

`jumptable::resolve` takes a `Program` — and the case bodies a table
targets may be code `recover` never discovered, precisely because the
indirect jump contributed no edges. A one-shot post-pass would attach
successor VAs with no blocks behind them. So the fold is a bounded
fixpoint inside recovery: recover → resolve tables → add the proven
successors and walk the newly reachable code into blocks → re-resolve
(new code can hold new tables) → until a round adds nothing, with a
small round cap (defense in depth; report `capped` in some visible
form if hit). Layering: `jumptable` depends on `cfg`, so the loop's
driver must live where it creates no cycle — a `cfg`-owned function
taking the resolved map as plain data per round, with the loop itself
in a thin free function (in `jumptable` or a caller-visible seam the
builder chooses and documents); the requirement is one obvious entry
point that callers use, and that plain `cfg::recover` stays available
and unchanged in behavior.

## Soundness

- Only *proven* tables fold — whatever `jumptable::resolve` already
  guarantees; this slice adds no new inference.
- A target outside the function under recovery does not fold (an
  out-of-function edge would corrupt every downstream pass); it is
  dropped with the drop visible in stats, not silently.
- The terminator stays `IndirectJump { import }` — the bytes' truth —
  only `successors` gains the proven targets, deduplicated, sorted,
  deterministic. Elsewhere successors remain an under-approximation,
  as the `cfg` module docs already state; update those docs to say
  "except proven jump tables".
- Everything downstream is invariant-checked already (`ir::check`,
  `irssa::check`, `irstruct::check`); the fold must keep them all
  green on real binaries.

## Module-by-module

- `src/cfg.rs`: the fold + fixpoint machinery per above; module-doc
  update (the under-approximation caveat gains its exception).
- `src/jumptable.rs`: whatever thin driver/seam the layering decision
  puts here; no change to the proof logic.
- `src/bin/redump.rs`: `--lift`/`--ssa`/`--ssa-opt`/`--structure`/
  `--decompile` recover through the folding entry point (they already
  compute `resolve` + `successor_map` for `structure` — reuse, do not
  compute twice). `--cfg`-style textual views may show the new edges;
  goldens updated honestly.
- `ROADMAP.md`: Current-thread entry with the measured numbers.
- Do NOT touch `src/irstruct.rs` or `src/irflow.rs`/`src/irssaopt.rs`
  (companion slices own them this wave). The stale "honest limit" note
  in `irstruct`'s docs is fixed at merge, not by this slice.

## Test matrix (~12)

1. a synthetic function with a proven table: after recovery the
   dispatch block's successors are exactly the table targets and every
   target has a block.
2. fixpoint: a table whose case body contains a *second* table —
   both fold, two rounds.
3. an unproven indirect jump stays successor-less.
4. an out-of-function target does not fold and the drop is visible.
5. determinism (twice → identical Program); round cap respected.
6. e2e: `--structure`/`--decompile` on the fixture render a real
   `switch` with case labels, no `Opaque`.
7. existing corpus invariants: full pipeline over /bin/ls and
   /bin/bash x86-64, zero check failures, byte-deterministic.

## Exit criteria (demonstrate, don't assert)

On real x86-64 binaries (/bin/ls, /bin/bash, and if neither has a
proven table, a Homebrew binary that does — find one): report N tables
folded, the before/after `Opaque{IndirectJump}` and goto counts from
`--structure`, and one concrete function rendering a real `switch` in
`--decompile` — printed in the commit message. aarch64: whatever
`jumptable` already proves on arm64 folds identically (same code
path); verify on the libbrotlidec dylib and report, but arm64 table
*proof* coverage is out of scope.

## Non-goals

- New table-proof heuristics (`jumptable`'s job, unchanged).
- Any `irstruct` schema change — `Switch` exists; this slice only
  feeds it real edges.
- Tail-call or indirect-call successor recovery.
