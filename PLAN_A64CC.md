# Plan: aarch64 coverage two — CCMP/CCMN, LDUR/STUR, ADC/SBC (`aarch64`)

## Goal

The three highest-frequency non-SIMD residuals the coverage slice
(d20a587) recorded, in one focused slice:

1. **CCMP/CCMN** (immediate and register forms) — the conditional
   compare behind chained `&&`/`||` comparisons, deferred from the
   first slice because it "needs care in the flag model". Semantics:
   if `cond` holds, NZCV gets the flags of the compare
   (CCMP: `rn - op2`; CCMN: `rn + op2`); else NZCV gets the literal
   `nzcv` imm4.
2. **LDUR/STUR/LDURS\*** — unscaled 9-bit signed-offset loads/stores;
   decode and lift exactly like the existing scaled `Ldr`/`Str`
   family with the unscaled addressing mode (the lift is the easy
   half; the decode arm slots into the existing load/store group).
3. **ADC/ADCS/SBC/SBCS** — add/subtract with carry (the gap the first
   slice retargeted its unknown-probe tests to).

## The flag-model care (the actual work)

One NZCV model, still. CCMP lifts as: evaluate the existing condition
expression (the same `Cond` machinery `CSEL`/`B.cond` already use),
then each of the four flag cells gets a select between the compare's
flag expression (the very expressions the ADDS/SUBS lift writes —
reuse them, do not re-derive) and the corresponding imm4 bit constant.
ADC/SBC fold the carry-in into the existing add/sub value and flag
expressions per the ARM ARM (`rn + op2 + C`, SBC as `rn + ~op2 + C`)
— again through the one model, extended once, in one place, with the
extension visible in the module docs. Every lifted block passes
`ir::check`; the flag-write expressions for the S forms must be
textually the products of the shared model (test that, as d20a587
tested ADDS-vs-immediate).

Downstream note, measured not promised: a lifted CCMP feeds
`irssaopt::forward` + `irflow` as ordinary selects — whether chained
conditions actually collapse in pseudocode is reported as observed,
and any missing fold is recorded as `irflow` input, not patched here.

## Module-by-module

- `src/aarch64.rs`: decode arms (reserved encodings → `Unknown`,
  never a near-miss — e.g. CCMP's o2/o3 bits), render in the
  preferred spelling, golden words assembler-verified
  (`clang -arch arm64` + objdump, as before).
- `src/aarch64_lift.rs`: the lifts per above.
- Do NOT touch src/irssaopt.rs, src/irflow.rs, src/irstruct.rs,
  src/jumptable.rs, src/cfg.rs — companion slices own them.
- `ROADMAP.md`: Current-thread entry; update the honest-ceiling list
  (SIMD/FP, exclusives/atomics, PAC remain).

## Test matrix (~16)

1. golden decode+render per form (CCMP/CCMN imm+reg, LDUR/STUR all
   sizes + signed loads, ADC/SBC ± S), sf=0/1, assembler-verified.
2. reserved-encoding probes per group → `Unknown` (and find a new
   still-real gap for the generic unknown-probe tests if these were
   it).
3. lift: `ir::check` on every form; CCMP's four flag cells are
   selects over the shared model's expressions (textual reuse
   asserted); ADCS/SBCS flags via the same model; the existing sweep
   fuzzes stay green.
4. an e2e chained-condition block (`cmp; b.cond; ccmp; b.cond`)
   through construct → optimize → forward: sound SSA, checks green,
   rendered output eyeballed and recorded.
5. determinism; decode-total fuzz.

## Exit criteria (demonstrate, don't assert)

Re-run d20a587's coverage measurement: ffmpeg arm64 (96.89% decoded /
1,477 unknown baseline) and the ls arm64e slice (94.40% / 214) —
report the new percentages and the updated top-remaining list (should
now be SIMD/FP-dominated even more starkly). Full pipeline on
libbrotlidec: zero check failures, byte-deterministic, and note
whether its paired-shape count (36 at the condrec baseline) moves now
that CCMP decodes.

## Non-goals

- SIMD/FP, exclusives/atomics, PAC — the recorded ceiling.
- CSETM/rev/cls-style one-source data processing (next increment if
  frequency justifies).
- Any `irflow` pattern work (measured and reported only).
