# Adversarial / benchmark fixtures

Public fixtures for comparing Aletheia against other RE toolkits on
**workflow** (open → functions → decompile → rename → diff → patch), not
trademark parody.

## In-tree

| Path | Role |
|---|---|
| `fixtures/diamond` | CFG diamond / if-else structuring |
| `fixtures/loop_bc` | Loop back-edge |
| Other `fixtures/*` | Decoder / IR edge cases |

## Headless smoke

```console
./scripts/bench-smoke.sh
```

Prints `BENCH_SMOKE_SUMMARY` JSON. Exit 0 = all PASS.

## GUI timed checklist

See [GUI_BENCH_CHECKLIST.md](GUI_BENCH_CHECKLIST.md) and
[BENCH_BASELINE.md](BENCH_BASELINE.md).

## Suggested adversarial pack (external)

Add thin, redistributable binaries under `fixtures/adversarial/` (do not
commit proprietary samples):

1. **Go stripped** — pclntab / string recovery stress
2. **Rust panic paths** — demangle + type evidence
3. **Mach-O arm64e** — PAC + chained fixups (decrypt out of scope)
4. **ObjC / Swift** — `--objc` / `--swift` field listing
5. **Version pair** — two near-identical builds for `--diff` / `--patch-from-diff`

Score: time-to-first-decompile, rename round-trip via MCP, conflict honesty
in `--typefacts` (signed∩unsigned → `conflict`, never silent `int`).
