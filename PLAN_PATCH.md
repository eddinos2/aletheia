# PLAN_PATCH — PatchSet preview + sibling apply (P1a)

## Goal

First-class patch object beyond IDA: JSON/text PatchSet with
preconditions (exact old bytes at VA), preview (dry-run report), apply
to sibling `*.patched` by default. aarch64 NOP helper from public
encodings. Resign recipe stub for Mach-O.

## Module

`src/patch.rs` + `redump --patch-preview` / `--patch-apply`.

## Exit

Round-trip test: build PatchSet, preview Ok, apply sibling, re-read
bytes match; wrong old-bytes precondition fails cleanly.
