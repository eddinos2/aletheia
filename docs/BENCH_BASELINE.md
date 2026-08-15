# Bench baseline (local)

Captured from a developer machine for adversarial comparison scaffolding.
**Not** a published claim against commercial suites — re-run on your hardware
and fill [GUI_BENCH_CHECKLIST.md](GUI_BENCH_CHECKLIST.md) § D.

## Machine

| Field | Value |
|---|---|
| Date | 2026-08-16 |
| OS | macOS (Darwin), arm64 host |
| Engine | `0.1.0` (from `health` / stamp) |
| Fixture hash (`fixtures/diamond`) | `0x8b3da7d3d2b208ad` |
| Profile | `debug` (default smoke) |

## Commands

```console
$ ./scripts/bench-smoke.sh           # debug binaries
$ ./scripts/bench-smoke.sh --release # optional release timings
```

Manual GUI timings: follow § B in [GUI_BENCH_CHECKLIST.md](GUI_BENCH_CHECKLIST.md).

## Headless results (debug, one run)

`BENCH_SMOKE_SUMMARY` excerpt — wall times in milliseconds where recorded:

| Step | Status | ms (if timed) | Note |
|---|---|---|---|
| mcp_health … mcp_patch_preview (13 checks) | PASS | — | see script output |
| mcp_batch_wall (all MCP reqs, one process) | PASS | ~40–45 | open→…→patch_preview |
| redump_listing | PASS | ~40–50 | `--listing=8` |
| redump_decompile | PASS | ~25–35 | `--decompile=4` |
| redump_json | PASS | ~20–30 | `--json` |
| redump_diff | PASS | ~20–30 | diamond vs shortcircuit |

Full machine-readable dump: re-run the script and copy the `BENCH_SMOKE_SUMMARY`
JSON block. GUI interactive steps are checklist-only (not automated here).
