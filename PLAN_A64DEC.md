# Plan: aarch64 decoder + lift coverage — integer data-processing (`aarch64`)

## Goal

Close the high-value integer data-processing gaps in the aarch64
decoder and its lift. Inventory (verified against `src/aarch64.rs`):
the decoder already has branches, Adr/Adrp, add/sub immediate,
MOVN/MOVZ/MOVK, the CSEL family, LDR/STR (imm + register-offset),
LDP/STP. What is missing is the meat of ordinary compiled code:

1. **Shifted-register arithmetic/logical** — ADD/SUB (shifted
   register) and AND/ORR/EOR/BIC (shifted register), plus the
   flag-setting forms ADDS/SUBS/ANDS/BICS. Aliases fall out free:
   CMP/CMN (SUBS/ADDS rd=zr), TST (ANDS rd=zr), MOV (ORR rn=zr),
   MVN (ORN rn=zr), NEG (SUB rn=zr).
2. **Extended-register ADD/SUB(S)** — the sp-legal form
   (`add x0, sp, w1, uxtw #2`); very common in prologue/epilogue and
   array addressing.
3. **Logical immediate** — AND/ORR/EOR/ANDS with the N:immr:imms
   bitmask-immediate encoding (implement `decode_bit_masks` exactly:
   reject the reserved all-ones element, replicate the rotated
   element pattern; property-test it).
4. **Bitfield** — SBFM/BFM/UBFM, with the aliases spelled at render
   time (LSL/LSR/ASR immediate, UBFX/SBFX, UXTB/UXTH,
   SXTB/SXTH/SXTW). Decode to the canonical form; alias spelling is
   display-only.
5. **Two-source** — LSLV/LSRV/ASRV/RORV (register shifts), UDIV/SDIV
   (div-by-zero yields 0 per the ARM ARM — the lift must model that,
   not trap).
6. **Three-source** — MADD/MSUB (aliases MUL/MNEG), and if cheap
   SMULL/UMULL/SMULH/UMULH; otherwise leave them `Unknown` and say so.

Everything else (FP/SIMD, atomics, system regs beyond what exists)
stays `Unknown` → conservative intrinsic in the lift. Coverage, not
completeness, is the goal; the ceiling must stay honest.

## Module-by-module

- `src/aarch64.rs`:
  - `Opcode` variants for the families above, in the existing style
    (sf/rd/rn/rm/shift-kind/amount fields, canonical not alias).
  - `decode_dp_reg`: the shifted-register logical + add/sub groups,
    extended-register add/sub, two-source, three-source — mirroring
    the existing CSEL arm's bit-slicing style. Reserved encodings
    (e.g. shift `ROR` on add/sub, sf=0 with imm6 ≥ 32) →
    `Opcode::Unknown`, never a wrong decode.
  - `decode_dp_imm`: logical-immediate and bitfield groups;
    `decode_bit_masks` as a standalone tested helper.
  - `flow()`: all new opcodes are `Flow::Normal`.
  - Render/`Display`: alias spelling where the ARM ARM prefers it
    (CMP/CMN/TST/MOV/MVN/NEG, LSL/LSR/ASR, UBFX/SBFX, MUL) so the
    disasm view reads like `otool`; canonical fallback otherwise.
- `src/aarch64_lift.rs`: lift each new opcode to existing `ir` ops —
  shifts/extends as the obvious `Shl/LShr/AShr` + `ZeroExtend/
  SignExtend/Truncate` compositions, bitfield as shift+mask, division
  with the ARM zero-divisor semantics (select on `rm == 0`), MADD/MSUB
  as mul+add/sub. Flag-setting forms write NZCV exactly the way the
  existing add/sub-immediate flag model does (reuse it — do not invent
  a second flag encoding). Every lifted block must pass `ir::check`.
- `ROADMAP.md`: Current-thread entry (decoder coverage: integer DP);
  note the honest remaining ceiling (FP/SIMD, atomics).

## Soundness

- A decode is either **exactly right or `Unknown`** — reserved and
  unallocated encodings must not decode to a near-miss. This is the
  decoder's standing invariant; keep it.
- `decode_bit_masks` is the one subtle algorithm — property-test
  against a straightforward reference implementation over all valid
  (N, immr, imms) triples (13-bit space, exhaustive is cheap).
- The lift stays best-effort by contract; new opcodes only shrink the
  intrinsic set, never change existing lifts. NZCV via the existing
  model only.

## Test matrix (~30)

1. golden decode+render for each new family, cross-checked word
   values assembled from the ARM ARM (or `as`-assembled bytes), both
   sf=0/sf=1, including sp-vs-zr register-31 cases where the form
   distinguishes them (extended add/sub: sp; shifted/logical: zr).
2. alias spelling: CMP/CMN/TST/MOV/MVN/NEG, LSL/LSR/ASR imm,
   UBFX/SBFX, UXTB/SXTW, MUL — rendered as the alias.
3. `decode_bit_masks` exhaustive vs reference; reserved triples
   rejected → `Unknown`.
4. reserved-encoding probes per family → `Unknown` (no near-miss).
5. lift: each family lifts, `ir::check` passes; UDIV/SDIV zero-divisor
   select present; flag-setting forms write the same NZCV cells as
   add/sub-immediate; CMP then B.cond round-trips through the
   existing condition model.
6. determinism + the existing total-decode fuzz keeps passing (decode
   is total over random words — new arms included).

## Exit criteria (demonstrate, don't assert)

Measure decoder coverage on a **real arm64 Mach-O** (`lipo -thin arm64`
of a system binary): count decoded-vs-`Unknown` words over all text
before and after this slice, and report the delta plus the top
remaining `Unknown` encodings by frequency. The point is a measured
ceiling, not a vibe. (If the irlift dispatch slice has landed by then,
also confirm the lifted intrinsic count drops; if not, the raw decode
count suffices.)

## Non-goals (this slice)

- FP/SIMD, atomics/exclusives, PAC, system-register traffic beyond
  what exists — stay `Unknown`, honestly.
- CCMP/CCMN and conditional-compare flag logic (worth a follow-up;
  needs care in the flag model).
- Any `irlift`/`irssa` change (that is the companion `PLAN_A64LIFT.md`
  slice); any pseudocode work (`PLAN_PSEUDO.md`).
