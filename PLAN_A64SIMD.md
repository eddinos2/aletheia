# Plan: aarch64 coverage three — SIMD/FP loads/stores + FP moves (`aarch64`)

## Goal

The measured ceiling is now SIMD/FP outright: q/d/s-register
`ldr/str/stp/ldp/ldur/stur`, `movi`, `fmov`, and the fcvt/scvtf/fcmp
family dominate every remaining-unknown tally. This slice decodes the
*load/store and move* mass — the addressing and register-file part,
which is mechanical — and lifts it precisely; FP arithmetic
(fadd/fmul/fdiv/fcmp and conversions) stays out except where noted,
because arithmetic semantics deserve their own slice.

In scope, by remaining-count order:
1. SIMD&FP register loads/stores: `LDR/STR` (immediate scaled +
   register-offset + unscaled `LDUR/STUR`) and `LDP/STP` for
   b/h/s/d/q sizes — the existing integer load/store machinery
   generalizes; the size field extends, the addressing modes are the
   ones already decoded.
2. `FMOV` (register↔register, register↔GPR, and immediate) and
   `MOVI`/`MVNI`/`FMOV` (vector immediate) — moves, not arithmetic.
3. If cheap after 1–2: `DUP`/`INS`/`UMOV`/`SMOV` element moves.
   Otherwise record as next-increment; do not stretch.

## The IR question (decide it honestly)

The IR has no vector types. The lift must not invent them. The
recorded doctrine from `SMULH`/`UMULH` applies: precise named
intrinsics over the exact cells read/written. The builder inventories
how `Space::Arch` models v0–v31 today (the first coverage slices left
SIMD regs to the `a64.unknown` clobber — there may be no vector cell
space yet) and either (a) adds the vector register cells (128-bit as
two 64-bit cells or one wide cell — whichever `ir`'s width model
permits without new operators) with plain `Load`/`Store`/`Assign`
lifts for the memory ops, or (b) lifts to per-op intrinsics with
exact reads/writes. Prefer (a) for loads/stores if the width model
allows — a `ldr q0, [x8]` that reads memory into a named cell beats
an intrinsic for every downstream pass — but the choice is the
builder's, made after inventory, documented in module docs, with the
constraint that `ir::check` passes and no downstream pass needs
changes.

## Module-by-module

- `src/aarch64.rs`: decode arms in the existing group style, reserved
  encodings → `Unknown` (never a near-miss), render in `otool`
  spelling, golden words assembler-verified (`clang -arch arm64`).
- `src/aarch64_lift.rs`: the lifts per the decision above; every
  block `ir::check`-green; the existing sweep fuzzes stay green.
- `src/jumptable.rs`: the forced `a64_defs` exhaustive-match ripple
  (additive arms only) — expected, as in both prior coverage slices.
- Do NOT touch src/irflow.rs, src/irssaopt.rs, src/irstruct.rs,
  src/cfg.rs — companion slices run in parallel.
- `ROADMAP.md`: Current-thread entry; ceiling list updated
  (FP arithmetic, exclusives/atomics, PAC remain).

## Test matrix (~18)

1. golden decode+render per form and size class (b/h/s/d/q; imm,
   reg-offset, unscaled, pair; pre/post-index where the integer forms
   support them), assembler-verified.
2. reserved probes per group → `Unknown`.
3. lift: `ir::check` on every form; a load then store round-trips
   the cell; the register-file modeling decision has its own direct
   tests; sweeps stay green.
4. determinism; decode-total fuzz.

## Exit criteria (demonstrate, don't assert)

Re-measure coverage on ffmpeg arm64 (97.30% / 1,283 unknown baseline)
and ls arm64e (95.10% / 187) — the SIMD load/store class should
collapse; report new percentages and the updated top-remaining tally.
Full pipeline on libbrotlidec and one FP-heavy Homebrew dylib: zero
check failures, byte-deterministic, and report the `a64.unknown`
clobber-intrinsic count delta (fewer clobbers = more precise SSA —
say how much).

## Non-goals

- FP/vector *arithmetic* semantics (fadd/fmul/fcmp/fcvt lifts beyond
  a decode-to-intrinsic if trivially cheap) — next increment.
- Exclusives/atomics, PAC — the recorded ceiling.
- Any IR operator or width-model change.
