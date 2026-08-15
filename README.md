# Aletheia

**Clean-room binary analysis in Rust** — loaders, disassembly, recovery, and an
early decompiler path. Built for security research, malware analysis,
vulnerability work, interoperability, and teaching.

Named after Greek *aletheia*: truth as what is brought out of concealment.
That is the point of the tool.

Aletheia sits in the open tradition of [Ghidra](https://ghidra-sre.org/),
[radare2](https://rada.re/), and [Rizin](https://rizin.re/). Everything is
written from **public specs** (PE/COFF, ELF gABI, Intel SDM, Arm ARM) — not
from proprietary tool internals.

> Early project. The API will move. What is here aims to be solid: bounds-checked
> parsers, typed errors, deterministic analysis, and tests. See [ROADMAP.md](ROADMAP.md).

## Status (honest)

| Area | State |
|---|---|
| PE / ELF64 / Mach-O loaders | Working |
| x86-64 + AArch64 decode | Working (A64 still deepening) |
| CFG, functions, xrefs, strings, jumptables | Working |
| Parallel analysis (deterministic) | Working |
| Open annotation DB (git-friendly) | Working |
| IR → SSA → structure → `--decompile` | Working, readability still deepening |
| Go / Rust / C++ recoveries | Present, uneven depth |
| PatchSet preview / sibling apply | Early |
| MCP server (`aletheia-mcp`) | Early skeleton |
| GUI | Spec + mockup only |

```console
$ cargo run --bin redump -- program.exe
$ cargo run --bin redump -- ./a.out --listing
$ cargo run --bin redump -- ./app --decompile=4
$ cargo run --bin redump -- old.bin --diff new.bin
```

## Design bets

1. **Library first** — CLI and future UI are thin clients.
2. **Hostile input is normal** — malware and broken files; no panics on garbage.
3. **Parallel by construction** — same binary → same result at any thread count.
4. **Open, diffable annotations** — asserted facts in a documented format.
5. **Zero mandatory deps** in the core crate.
6. **Clean-room, always** — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Build

```console
$ cargo build
$ cargo test
$ cargo clippy --workspace --all-targets
```

Stable Rust, edition 2024.

## License

MIT OR Apache-2.0.

## Author

Maintained by [eddinos2](https://github.com/eddinos2).
