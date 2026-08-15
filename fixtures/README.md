# Ground-truth fixtures (DESIGN slice 19)

Small, known-shaped C programs compiled **once, offline** and committed
as bytes — the evaluation harness (`src/evalfx.rs`, run with
`cargo test evalfx`) measures the decompilation pipeline against them.
There is no build-time compile step and no build-time dependency: the
binaries in this directory are the fixtures, and this file is their
provenance record. Rebuilding them is never required; if one is ever
regenerated (new shape, new flags), update this file and the
expected-number table in `src/evalfx.rs` in the same commit.

Each `.c` file carries its hand-written source-level CFG in its header
comment; the same graph, machine-readable, is the GED ground truth in
the one fixture table in `src/evalfx.rs`. The real-binary sweeps
(/bin/ls, /bin/bash, libbrotlidec) are **not** fixtures — they stay
exit-criteria material, summarized in ROADMAP.md, never asserted in
tests.

## Provenance

Compiled on macOS (Darwin 25.3.0, arm64 host) with the system clang,
cross-targeting x86-64 Mach-O:

```
$ clang --version
Apple clang version 21.0.0 (clang-2100.1.1.101)
Target: arm64-apple-darwin25.3.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
```

Exact commands, run in this directory:

```
clang -arch x86_64 -O1 -o diamond      diamond.c
clang -arch x86_64 -O1 -o switch_dense switch_dense.c
clang -arch x86_64 -O1 -o loop_bc      loop_bc.c
clang -arch x86_64 -O1 -o tail_merge   tail_merge.c
clang -arch x86_64 -O1 -o shortcircuit shortcircuit.c
```

`-O1` is the deliberate choice: high enough that the compiler emits the
interesting artifacts (a real jump table for the dense switch, loop
rotation, cross-jumped tails), low enough that the source shape is
still recognizable. All five are plain Mach-O 64-bit x86-64
executables; symbols are kept so the harness can find each fixture
function by name.

## The five shapes

| fixture        | symbol          | shape it pins down                                      |
| -------------- | --------------- | ------------------------------------------------------- |
| `diamond`      | `_diamond`      | one two-way conditional + join (arms store to two different volatile globals — one sink gets legally if-converted to a cmov and no diamond survives) |
| `switch_dense` | `_switch_dense` | dense 6-case `switch` + default; every case a different op on an unknown second argument, so a value lookup table is impossible and a jump table is emitted (`lea`+`movslq`+`jmpq *` observed in the committed bytes) |
| `loop_bc`      | `_loop_bc`      | counted loop carrying both a `continue` and a `break` (rotated by the compiler: guard + do-while) |
| `tail_merge`   | `_tail_merge`   | identical tails at *different nesting depths*, which cross-jumping merges into shared blocks — the SAILR tail-merge shape (the merge is visible in the committed bytes) |
| `shortcircuit` | `_shortcircuit` | `&&`/`||` chain; the compiler flattens the middle of the chain to `setcc`+`or` and keeps the outer branches, so recovered-vs-source distance is a real, recorded number |

The volatile globals are load-bearing everywhere: they are what keeps
each arm's effect observable so no branch folds away at `-O1`.
