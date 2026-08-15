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
| Headless `--json` | Functions list |
| GUI / TUI | Spec only (Gate G1 open) |

```console
$ cargo run --bin redump -- ./target
$ cargo run --bin redump -- ./a.out --listing
$ cargo run --bin redump -- ./app --decompile=4
$ cargo run --bin redump -- old.bin --diff new.bin
$ cargo run --bin redump -- old.bin --patch-from-diff new.bin
$ cargo run --bin redump -- ./app --json
$ cargo run -p aletheia-mcp
```

Agent loop (MCP): `open` → `decompile` / `diff` / `listing` / `patch_preview` → `rename`.

### MCP headless smoke

With `aletheia-mcp` on `PATH` (or via `cargo run -p aletheia-mcp`):

1. Start the server over stdio from your agent host (Cursor / Claude / etc.).
2. Call `health`, then `open` on a local fixture under `fixtures/`.
3. Call `decompile` on a known function index and confirm non-empty pseudocode.
4. Optional: `why` on the same address to verify provenance text.

This is the Gate M1 manual smoke until the TUI lands — no GUI required.


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
