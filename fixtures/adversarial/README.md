# Adversarial fixture pack

Place redistributable, **non-proprietary** samples here for scorecard runs.
Do not commit malware, App Store IPA ciphertext, or licensed firmware.

## Layout

```
fixtures/adversarial/
  README.md          (this file)
  go_stripped/       # optional: tiny Go hello, stripped
  rust_panic/        # optional: tiny Rust binary
  macho_arm64/       # optional: thin arm64 Mach-O (no FairPlay)
  version_pair/      # optional: old.bin + new.bin for diff/patch
```

## Minimum for CI / smoke

In-tree `fixtures/diamond` and `fixtures/loop_bc` already cover CFG +
decompile. Adversarial packs deepen library ID, Swift/ObjC, and version
diff — add when you can redistribute the bytes.

## Commands

```console
./scripts/bench-smoke.sh
cargo run --bin redump -- fixtures/diamond --typefacts=1
cargo run --bin redump -- path/to/swift_app --swift
```

See [ADVERSARIAL_FIXTURES.md](../../docs/ADVERSARIAL_FIXTURES.md) and
[SCORECARD.md](../../docs/SCORECARD.md).
