# Plan: aarch64 `irlift` dispatch + SSA-render arch-awareness (`irlift`)

## Goal

Light up the whole IR pipeline for AArch64. Today `irlift` dispatches
x86-64 only (`Arch::Aarch64 => return None`), so no aarch64
`LiftedFunction` — hence no SSA, no structuring, no anything — can
exist. The `aarch64_lift` module already exists and passes `ir::check`;
`callfx` already ships aarch64 tables (AAPCS64). This slice is the
wiring, plus making the renderers name aarch64 registers correctly. The
downstream passes (`irssa`, `callfx`, `irssaopt`, `irstruct`, `irout`,
`pseudo`) are **arch-blind** and need zero changes.

## Module-by-module

- `src/irlift.rs`:
  - `lift_function_capped`: add an `Arch::Aarch64` arm that lifts each
    block via `aarch64_lift::lift_block`, mirroring `lift_block_x86`'s
    structure — decode the block's instruction run
    (`aarch64::Instruction`, fixed 4-byte, `Flow`-classified), thread
    the instruction budget, set `LiftedBlock::truncated` on an
    undecodable word / unmapped VA / budget hit exactly as the x86 path
    does. `aarch64_lift::lift_block` takes `&[(Instruction, va)]`, so
    the decode-and-collect loop lives here (or a small
    `lift_block_aarch64` helper beside `lift_block_x86`). An
    `Opcode::Unknown` word lifts (in `aarch64_lift`) to a sound
    intrinsic — coverage, not correctness, is what is partial.
  - **Carry the arch.** Add `arch: ir::Arch` (or `crate::model::Arch`,
    whichever the crate uses) to `LiftedFunction`, set at lift time.
    Thread it into `SsaFunction` (via `irssa::construct`) so the
    renderers can pick the right register speller. This is the
    recommended carrier (the honest data-model fix); if the builder
    finds a cleaner one (e.g. a `RegNamer` threaded into `render`),
    use it and document the choice — the requirement is only that an
    aarch64 function renders aarch64 register names.
  - `irlift::render`: name `Space::Arch` cells via `aarch64_lift::reg_name`
    for an aarch64 function, `x86_lift::reg_name` for x86-64.
- `src/irssa.rs`: `SsaFunction` gains the `arch` field (carried from
  the lifted function); `display`/`render` dispatch the `Space::Arch`
  spelling on it (`x86_lift::reg_name` vs `aarch64_lift::reg_name`)
  instead of the current hardcoded `x86_lift::reg_name`. The flag/temp
  spelling is unchanged (shared cells). **Every existing test that
  builds a `SsaFunction`/`LiftedFunction` literal must gain the new
  field** — default to x86-64 so all current goldens are byte-identical
  (assert this: the x86 `--ssa`/`--lift` output must not move).
- `src/bin/redump.rs`: relax the `image.arch() != Arch::X86_64` gates in
  `print_lift`, `print_ssa_view`, `print_ssa_opt` (and `--structure`,
  `--decompile` if present at merge) to **also** admit `Arch::Aarch64`.
  `callfx::abi_for` / `function_live_out` already return aarch64
  tables, so the pipeline runs unchanged; keep the note honest —
  coverage is decoder-limited (see the companion decoder slice).
- `ROADMAP.md`: Current-thread entry; note this unblocks arm64/iOS and
  that the aarch64 pseudocode e2e now becomes a verification rider on
  `pseudo`.

## Soundness

- `aarch64_lift` is best-effort by contract: unmodeled opcodes lift to
  conservative intrinsics, every output passes `ir::check`. So every
  downstream invariant (`irssa::check`, `irssaopt` preservation,
  `irstruct::check`, `irout::check`) holds by construction — they never
  knew the ISA.
- The arch field is pure data; determinism and no-panic are preserved.
- No x86-64 output changes (the default-to-x86 rule on literals + a
  golden-stability assertion guarantee it).

## Test matrix (~14)

1. `lift_function` on an aarch64 `cfg::Function` returns `Some`, blocks
   lifted, `ir::check` passes on every block.
2. a truncated/undecodable aarch64 block sets `truncated`, keeps the
   valid prefix, no panic.
3. arch carried: an aarch64 `LiftedFunction`/`SsaFunction` reports
   `Arch::Aarch64`; an x86 one still `X86_64`.
4. render: an aarch64 function names `x0`/`sp`/`w0` correctly; an x86
   function is byte-identical to before (golden stability).
5. `irssa::construct` → `check` Ok on a hand-built aarch64 function
   (branches + a few modeled ops + an intrinsic from an `Unknown`).
6. the full pipeline (construct → optimize → forward → eliminate_dead →
   structure → out_of_ssa) runs on an aarch64 function with zero
   check failures — proving the passes are arch-blind.
7. redump gates: `--lift`/`--ssa`/`--ssa-opt`/`--structure` on an
   aarch64 image produce output, not the "x86-64 only" note.
8. determinism; no-panic fuzz on random 4-byte words through the lift.

## Exit criteria (demonstrate, don't assert)

A **real arm64 Mach-O** (this is an arm64 Mac — use the `arm64`/`arm64e`
slice of a system binary, e.g. `lipo -thin arm64 /bin/ls` or a
`/usr/lib` dylib) through `redump --lift`, `--ssa`, `--ssa-opt`, and
`--structure`: no panics, `ir::check`/`irssa::check`/`irstruct::check`
all pass across every function, registers render as `x0…x30`/`sp`,
byte-deterministic. Report one concrete function with its rendered SSA.
Record which **common opcodes still lift to intrinsics** (the
decoder-coverage gaps) as input for the companion decoder slice — an
honest ceiling, measured not guessed.

## Non-goals (this slice)

- Extending the aarch64 **decoder** (that is the companion
  `PLAN_A64DEC.md` slice — this one only wires what already decodes).
- aarch64 pseudocode e2e (rides on `pseudo` once both land — trivial
  verification, no new code).
- Any change to the arch-blind passes.
