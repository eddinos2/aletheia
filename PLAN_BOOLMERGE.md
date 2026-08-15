# Boolean merging — goto lever two (fidelity residue)

## Goal

When a `goto` targets a block whose branch condition is `veq`-congruent to
a condition already decided on the path, fold the goto into the existing
control (continue/break/else-if), reducing residual gotos after φ-web.

## Module

`src/irstruct.rs` (extend). Uses `irflow` value-numbering witness.

## Status

Staged only — interleave with `irstack` per multi-track board. Do not
start until `PLAN_IRSTACK` A/B has a green landing commit.
