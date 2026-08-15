# PLAN_IRSTACK — affine SP tracking + slot partition (DESIGN 9–10)

## Goal

Post-`irssaopt`, prove per SSA name `name = sp0 + c` where possible;
classify every load/store address as `StackOff(c)`, `NonStack`, or
`Unproven`; partition proven offsets into evidence-backed slots for
`local_N` naming via `pseudo::VarNamer`.

## Module

`src/irstack.rs` (new). Consumes `SsaFunction` after optimize/forward/DCE.
Does not mutate SSA. `redump --stack[=N]` dumps facts.

## Algorithm

Abstract domain over SSA names: `Affine(i64) | NotSp | Unknown`.
Join = equal-or-degrade. SP cell: x86-64 arch 4, aarch64 arch 31.
Entry live-in SP version → `Affine(0)`. Propagate through assigns;
`Add`/`Sub` with constant updates offset; other ops → `NotSp`/`Unknown`.
Address classification: `base+imm` / bare base when base is Affine.
Slots (slice B lite): group `StackOff` accesses by offset; width from
access size; render `local_<abs_off>`.

## Caps / honesty

Cap distinct affine constants and slots per function. Alloca / non-affine
SP stops claims below that point. Never invent slots without access evidence.

## Exit

Unit tests for push/sub frame both ISAs; `check` on facts; `--stack` CLI;
slots feedable to `VarNamer` (optional `--decompile` wiring later).
