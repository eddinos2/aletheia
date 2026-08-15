# Adversarial fixture pack

Redistributable samples for scorecard / FLIRT / diff benches.
Generate or refresh with:

```console
./scripts/gen-adversarial-fixtures.sh
```

## Tracked layout

| Path | Role |
|---|---|
| `version_pair/{old,new}.bin` + `a.c`/`b.c` | patch-diff pair (tiny C) |
| `baseline_diamond` | copy of `fixtures/diamond` for local scorecard |
| `go_stripped/hello.go` | source; binary gitignored (rebuild via script) |
| `rust_panic/main.rs` | source; binary gitignored |
| `macho_arm64/` | placeholder for hand-added thin arm64 samples |

## Commands

```console
cargo run --bin redump -- fixtures/adversarial/version_pair/old.bin \
  --diff fixtures/adversarial/version_pair/new.bin
cargo run --bin redump -- fixtures/adversarial/baseline_diamond --flirt
cargo run --bin redump -- fixtures/adversarial/version_pair/old.bin \
  --patch-from-diff fixtures/adversarial/version_pair/new.bin
```

See [docs/ADVERSARIAL_FIXTURES.md](../../docs/ADVERSARIAL_FIXTURES.md) and
[docs/SCORECARD.md](../../docs/SCORECARD.md).
