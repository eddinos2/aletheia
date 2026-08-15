# Aletheia

**Open-source binary analysis for researchers** — a clean-room Rust toolkit
for disassembly, recovery, decompilation, surgical patch/diff, and
**agent-driven reverse-engineering workflows**.

Named after Greek *aletheia*: truth brought out of concealment.

## Why this exists

Commercial reverse-engineering suites are powerful and expensive. Many
labs, students, and indie researchers hit a wall: locked databases,
awkward headless/automation, and tooling that was never designed for
AI agents. Aletheia is the opposite bet:

- **Open source** (MIT OR Apache-2.0) — no license wall for research.
- **Library-first + MCP** — plug into Cursor, Claude, or any agent to
  automate triage, rename, decompile, diff, and patch preview.
- **Deterministic & parallel** — same binary, same result, any thread count.
- **Git-native annotations** — your work is a text log you can PR, not a
  proprietary opaque database.
- **Honest analysis** — proven vs heuristic is visible; no silent guessing.

The long-term bar is simple: **match or beat closed suites where it
matters for modern research** (headless, agents, collaboration, Go/Rust,
patch/diff) while staying clean-room and open.

## Keywords

`reverse-engineering` · `disassembler` · `decompiler` · `binary-analysis` ·
`malware-analysis` · `vulnerability-research` · `patch-diffing` · `MCP` ·
`AI-agents` · `Rust` · `PE` · `ELF` · `Mach-O` · `AArch64` · `x86-64`

## What works today

| Area | State |
|---|---|
| PE / ELF64 / Mach-O loaders | Working (chained fixup walk) |
| x86-64 + AArch64 decode | Working (SIMD ALU deepening) |
| CFG, functions, xrefs, strings, jumptables | Working |
| Parallel analysis (deterministic) | Working |
| Open annotation DB (git-friendly) | Working |
| IR → SSA → MEM promote → structure → `--decompile` | Working (`local_*` + sig headers) |
| Go / Rust / C++ / ObjC / Swift recoveries | Present, uneven depth |
| Open FLIRT-style matcher (`--flirt`) | Early |
| PatchSet + patch-from-diff | Working |
| MCP server (`aletheia-mcp`) | Agent entry point |
| Native GUI (`aletheia-gui`) | Gate G1 workstation (xref nav, CFG, deltas) |
| Headless `--json` | Functions list |
| macOS `.app` / `.dmg` scripts | Unsigned local packaging |

```console
$ cargo run --bin redump -- ./target
$ cargo run --bin redump -- ./a.out --listing
$ cargo run --bin redump -- ./app --decompile=4
$ cargo run --bin redump -- old.bin --diff new.bin
$ cargo run --bin redump -- old.bin --patch-from-diff new.bin
$ cargo run --bin redump -- ./app --json
$ cargo run -p aletheia-mcp
$ cargo run -p aletheia-gui
```

Agent loop (MCP): `open` → `decompile` / `diff` / `listing` / `patch_preview` → `rename`.

GUI loop: ⌘O → select function → `y` decompile / `c` CFG / click xrefs /
`n` rename (delta) / `?` Why? — same protocol as agents. See
[crates/aletheia-gui/README.md](crates/aletheia-gui/README.md).

### Bench / headless smoke

```console
$ ./scripts/bench-smoke.sh           # MCP + redump vs fixtures/; prints BENCH_SMOKE_SUMMARY
$ ./scripts/bench-smoke.sh --release # optional
```

Timed open → functions → decompile → rename → xref → CFG → diff → patch
checklist (GUI interactive + headless): [docs/GUI_BENCH_CHECKLIST.md](docs/GUI_BENCH_CHECKLIST.md).
Fixture guidance: [docs/ADVERSARIAL_FIXTURES.md](docs/ADVERSARIAL_FIXTURES.md).
Local baseline numbers: [docs/BENCH_BASELINE.md](docs/BENCH_BASELINE.md).
Scorecard template: [docs/SCORECARD.md](docs/SCORECARD.md).
Scripting ABI: [docs/PLUGIN_ABI.md](docs/PLUGIN_ABI.md) (`aletheia::api`).

### macOS package (unsigned / ad-hoc)

```console
$ ./scripts/macos-app.sh --release
$ ./scripts/macos-dmg.sh --release
```

Notarization needs a real Apple Developer ID — see
[crates/aletheia-gui/README.md](crates/aletheia-gui/README.md). Scripts do
not fake signing.


## Design principles

1. **Library first** — CLI, MCP, and future UI are thin clients.
2. **Hostile input is normal** — malware and broken files; typed errors, no panics.
3. **Parallel by construction** — analysis scales across cores by design.
4. **Open, diffable annotations** — asserted facts only; computed facts regenerate.
5. **Zero mandatory deps** in the core crate.
6. **Clean-room** — public specs only. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Build

```console
$ cargo build --workspace
$ cargo test --workspace
$ cargo clippy --workspace --all-targets
```

Requires stable Rust (edition 2024).

## License

Dual-licensed under **MIT OR Apache-2.0**, at your option.

- [LICENSE-MIT](LICENSE-MIT)
- [LICENSE-APACHE](LICENSE-APACHE)

See [LICENSE](LICENSE) for the short form.

## Maintainer

[eddinos2](https://github.com/eddinos2)

## Roadmap

High-level plan and phase status: [ROADMAP.md](ROADMAP.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for clean-room rules and PR expectations.

## Research use

If Aletheia is useful for your lab or agent workflow, a star helps other reverse engineers find a clean-room alternative to closed databases. Issues and fixtures welcome.

