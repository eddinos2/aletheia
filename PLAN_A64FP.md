# Plan: aarch64 coverage four — FP arithmetic, exclusives, PAC (`aarch64` + `aarch64_lift` + `callfx`)

## Goal

The SIMD load/store slice left the ceiling at FP *arithmetic* outright.
Measured first on the current tree (decode-probe over each binary's
`__text`, every unknown word named by capstone):

- libmp3lame arm64: 3,799 unknown of 39,485 (90.38%) — scalar FP
  dominates: fmul 685+38, fadd 564+12, fsub 379, fcvt 364+25, fcmp
  333+59, fmadd 270+4, scvtf 194+73, fdiv 162+48, fcsel 123+9, fmsub
  90, fcvtzs 78+31, fabs/fneg/fsqrt/frint* ~100, fccmp 11; element
  moves (dup/ins/umov/smov spellings) ~200; the rest is the Advanced
  SIMD vector-ALU long tail (xtn/cmhi/ext/sshll/orr/add/…, many
  distinct ops, small counts each).
- ffmpeg arm64: 425 of 47,510 (99.11%) — same scalar FP mix, plus
  ldar 22 / stlr 10 / ldaddal 8 (ordered + LSE atomics), ror-immediate
  (EXTR alias) 9, rbit/clz/rev ~25 across the corpus.
- /bin/ls arm64e: 139 of 3,819 (96.36%) — udf padding 92, paciza 23,
  retab 20, braaz/blraaz 2.
- libbrotlidec arm64: 50 of 6,354 (99.21%) — nearly all vector-ALU
  tail; this slice moves it little, honestly.

In scope, by that count order:
1. **Scalar FP data processing**: 2-source (FMUL FDIV FADD FSUB FMAX
   FMIN FMAXNM FMINNM FNMUL), 3-source (FMADD FMSUB FNMADD FNMSUB),
   1-source (FABS FNEG FSQRT FCVT s↔d FRINT{N,P,M,Z,A,X,I}), FCMP/
   FCMPE (register and #0.0), FCCMP/FCCMPE, FCSEL, and the conversion
   rows: SCVTF/UCVTF (GPR→FP and scalar-integer), FCVTZS/FCVTZU and
   the rounding-directed FCVT{N,P,M,A}{S,U} (FP→GPR). Half precision
   stays refused everywhere (FEAT_FP16, the documented gap).
2. **Element moves** (the SIMD slice's recorded deferral): DUP
   (element→scalar, element→vector, GPR→vector), INS (GPR→element,
   element→element), UMOV/SMOV (element→GPR).
3. **Exclusives / ordered**: LDAR{,B,H}, STLR{,B,H}, LDXR/LDAXR{,B,H},
   STXR/STLXR{,B,H}. LSE atomics (ldaddal & co.) stay refused —
   recorded, 8 sites in the corpus.
4. **PAC + udf**: UDF #imm16 (a real terminator — /bin/ls arm64e pads
   with it, and today the recursive sweep runs straight through zero
   words into data); RETAA/RETAB; BRAA/BRAB/BLRAA/BLRAB and their Z
   forms; the dp-1source PAC row (PACIA/PACIB/AUTIA/AUTIB, their Z
   forms, PACIZA…, XPACI/XPACD); the four PAC hints (PACIASP/AUTIASP/
   PACIBSP/AUTIBSP) get named opcodes and honest lifts instead of the
   generic execute-as-NOP `Hint`.
5. **dp-1source integer row** (same decode block as PAC): CLZ CLS
   RBIT REV REV16 REV32; **EXTR** (the `ror #imm` alias); **LDPSW**.
6. **callfx vector ABI** (the SIMD slice's recorded next increment):
   AAPCS64 clobbers gain v0–v7 and v16–v31 (both halves) and the high
   halves of v8–v15 (only the *bottom 64 bits* of v8–v15 are
   callee-saved); uses gain v0–v7 (both halves — HFA/vector args);
   `function_live_out` gains v0–v7 (both halves, the return superset)
   and the low halves of v8–v15 (callee-saved must come back).

## Lift doctrine (per the SMULH precedent, decided up front)

The IR grows no FP operators. Exact-expressible ops lift exactly:
FABS/FNEG are sign-bit masks; FCSEL is the csel merge on the low
cells; DUP/INS/UMOV/SMOV are shifts/masks over the two 64-bit cells;
EXTR is two shifts and an or; exclusive loads are plain loads,
exclusive stores a plain store plus an intrinsic write of the status
register (success is unknowable statically — over-approximating the
store as taken matches source intent in retry loops, and irflow never
deletes a store). Everything with real FP semantics lifts to a
**precise named intrinsic over the exact cells**: `a64.fadd` writes
`vlo(rd)` and reads its two operand cells — never the 100-cell
`a64.unknown` clobber — and every scalar FP write is followed by the
architectural `vhi(rd) := 0`. FCMP's intrinsic writes exactly the
four NZCV flag cells, so a following `b.cond` keeps a *precise*
def-use chain even though the comparison itself stays opaque.
SCVTF→FP reads the GPR; FCVTZS→GPR writes the GPR cell. PAC hints
and the dp-1source PAC ops lift to intrinsics writing their one
target register (reading it and, for the SP-discriminated hints, sp);
RETAA/RETAB lift like RET; BRAA/BLRAA like BR/BLR (the auth trap is
not control flow the decompiler models — documented). UDF lifts like
BRK: a named exception intrinsic, `Flow` terminal so the sweep stops
at padding. CLZ/CLS/RBIT/REV* lift to per-op named intrinsics
(`a64.clz`, …) writing rd reading rn — bit-exact formulas exist but
are expression bloat, and the named form is the recorded doctrine.

## Module-by-module

- `src/aarch64.rs`: decode arms in the existing group style; reserved
  encodings → `Unknown` (never a near-miss; FP `type == 0b11`/`0b10`
  and every unallocated opcode/rmode combination probed by tests);
  render in `otool` spelling; golden words assembler-verified
  (`clang -arch arm64`/`arm64e`), every rendered spelling proven to
  re-assemble to the identical word.
- `src/aarch64_lift.rs`: lifts per the doctrine above; every block
  `ir::check`-green; sweep fuzzes stay green.
- `src/callfx.rs`: the vector-ABI extension; its unit tests name the
  exact cells both directions and re-assert the soundness directions.
- `src/jumptable.rs`: additive `a64_defs` arms (FCVTZ*/UMOV/SMOV/
  CLZ-row/EXTR/LDPSW/exclusive-load define an X register; nothing
  else does beyond writeback bases already handled).
- `src/aarch64_text.rs`: untouched — its never-guess subset already
  documents SIMD/FP as out of scope; the *listing* rendering of the
  new ops is a recorded next increment.
- Do NOT touch src/irflow.rs, src/irssaopt.rs, src/irstruct.rs,
  src/pseudo.rs — companion slices run in parallel.
- `ROADMAP.md`: landed-slice bullet before the Active block.

## Test matrix (~40)

1. Golden decode+render per form (each 2-source op both precisions;
   3-source; 1-source incl. all seven FRINT; fcmp/fcmpe × reg/zero;
   fccmp; fcsel; each conversion row both sf × both precisions; dup/
   ins/umov/smov per arrangement; exclusives per size; PAC row; udf;
   extr incl. the ror alias spelling; ldpsw; clz/cls/rbit/rev*).
2. Reserved probes per group → `Unknown` (FP type 10/11, rmode holes,
   opcode holes, imm5 = 0 element specs, LSE row stays refused).
3. Lift: `ir::check` on every new form; FABS/FNEG/FCSEL/DUP/INS/
   UMOV/SMOV/EXTR exactness tests (bit-level against hand-computed
   values); the scalar-write-zeroes-high rule on every FP
   destination; fcmp writes exactly the four flags; stxr writes
   status + the store; udf terminates; PAC lifts write their one
   register. Sweeps stay green.
4. callfx: the extended tables both directions.
5. Determinism; decode-total fuzz stays total.

## Exit criteria (demonstrate, don't assert)

Re-measure the same probe: lame 3,799 → (expect ≲300: the vector-ALU
tail), ffmpeg 425 → ≲40, ls 139 → ≲5, brotli 50 → ≲40 (honest: this
slice barely moves brotli). Full `--decompile` on lame + brotli +
ls.arm64e: zero check failures, byte-deterministic double runs,
`a64.unknown` clobber-site count deltas reported, `cargo test evalfx`
untouched (x86-64 fixtures see no aarch64 change).

## Non-goals

- Advanced SIMD vector ALU (three-same, two-reg-misc, shift-imm,
  across-lanes, ld1/st1 structure forms) — the remaining ceiling,
  recorded with counts.
- LSE atomics (the `ldaddal` row) — recorded, 8 corpus sites.
- FEAT_FP16 half precision anywhere.
- The listing text formatter (`aarch64_text`) for any new op.
- Any IR operator or width-model change.
