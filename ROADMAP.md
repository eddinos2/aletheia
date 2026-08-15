# Aletheia Roadmap

Living plan for architecture, phases, and what ships next. Detail moves;
the mission does not.

## Mission

Ship an **open** binary-analysis engine that researchers can actually own:

1. **Better economics** — no five-figure seat for core static analysis.
2. **Better automation** — first-class MCP/agent workflows for triage,
   decompile, rename, version diff, and surgical patch preview/apply.
3. **Better defaults for modern binaries** — Go/Rust/C++, parallel
   analysis, git-native annotations, deterministic headless output.

We compete on **openness, agent integration, and engineering contracts**
(determinism, provenance, clean-room) while deepening decompiler quality
and Apple/iOS support over time.

## Positioning

| Pain with incumbents | Aletheia answer |
|---|---|
| Price / closed cores | MIT OR Apache-2.0 |
| Proprietary DB / version lock | Git-mergeable annotation log |
| Headless ≠ GUI / weak scripting | One engine; CLI = MCP = future GUI |
| Agent/MCP hangs on single-thread SDKs | Native Rust, cancelable jobs |
| Weak Go/Rust story | First-class recoveries (deepening) |
| Patching as opaque byte edits | Auditable PatchSet + sibling apply |

**Depth-first:** PE + ELF + Mach-O × x86-64 + AArch64 before exotic ISAs.

## Architecture

```
            ┌────────────────────────────────────────────────┐
            │                   clients                      │
            │   redump · aletheia-mcp · TUI/GUI · CI agents  │
            └───────────────────────┬────────────────────────┘
                                    │  protocol / library API
   ┌────────────────────────────────┼─────────────────────────────────┐
   │                            analysis                              │
   │   CFG · funcs · xrefs · strings · types · patch · diff           │
   │            (parallel passes over independent functions)          │
   ├──────────────────────────────────────────────────────────────────┤
   │                        program model (traits)                    │
   │   Image · Decoder · Flow · annotate Db (open, diffable)          │
   ├────────────────────────────┬─────────────────────────────────────┤
   │   pe · elf · macho         │        x86-64 · aarch64             │
   └────────────────────────────┴─────────────────────────────────────┘
```

## Multi-track board (current)

| Track | Landed | Next |
|---|---|---|
| **Engine** | irstack, sig, MEM promote, irtype, **typebounds E3**, FLIRT sample, **api 1.0** | FLIRT corpus growth |
| **iOS / Apple** | chained walk, ObjC + selrefs, Swift types/fields/**conformances** | witness table slot depth |
| **Patch / Diff** | PatchSet, patch-from-diff, A64 assemble | codesign recipe depth |
| **Agents / UI** | MCP, `--json`, GUI G1, bench smoke, **api 1.0** | finer delta ids; notarized DMG |

Gates: see [GATES.md](GATES.md).

## Phase checklist (summary)

### Phase 0–3 — Foundations through annotation DB — done

Loaders, decoders, CFG/funcs/xrefs, parallel engine, open annotations.

### Phase 4 — Interface & scripting — in progress

- [x] Symbolized listing (`redump --listing`)
- [x] MCP server (`aletheia-mcp`: open/decompile/diff/patch/listing/stack/xrefs/rename)
- [x] Headless `--json` (functions)
- [x] Native GUI Gate G1 (`aletheia-gui` — xref nav, CFG graph, rename deltas)
- [x] Stable scripting / plugin ABI (`aletheia::api` + [docs/PLUGIN_ABI.md](docs/PLUGIN_ABI.md))

### Phase 5 — Differentiators + decompiler — in progress

- [x] Go / Rust / C++ recoveries (partial depth)
- [x] Binary diff (`--diff`)
- [x] Decompiler spine through `--decompile` (local_* + sig headers)
- [x] Stack slots + MEM promote (`irstack` / `mempromote`)
- [x] Callee/caller sig helpers + open FLIRT-style matcher (`flirt`)
- [x] Type evidence (`irtype`) + bounds lattice (`typebounds`, Gate E3)
- [x] PatchSet preview/apply + patch-from-diff
- [x] ObjC / Swift early recoveries (Swift field descriptors)

### Phase 6 — Poly-frontend — in progress

Engine owns truth; GUI/TUI/MCP compute nothing. See [DESIGN_GUI.md](DESIGN_GUI.md).
Native GUI Gate G1: navigate / rename / Why? / decompile / xref click-nav /
CFG graph / incremental rename deltas. Packaging:
[crates/aletheia-gui/README.md](crates/aletheia-gui/README.md).
Bench checklist: [docs/GUI_BENCH_CHECKLIST.md](docs/GUI_BENCH_CHECKLIST.md).

## Quality bar

- No panics on adversarial input (decompiler path still hardening).
- Resource caps on attacker-controlled counts.
- Deterministic dumps; eval harness (`cargo test evalfx`).
- Clean-room discipline — [CONTRIBUTING.md](CONTRIBUTING.md).

## Explicit non-goals

- A GUI that invents its own analysis facts.
- Breadth-chasing exotic ISAs before the big five pairs are deep.
- FairPlay decrypt / jailbreak tooling inside the engine.
- Reverse-engineering proprietary tool internals.

## Patching principles

Auditable PatchSet objects, preview before write, sibling `*.patched` by
default, aarch64 assemble from public encodings, resign as a printed
recipe — never silent overwrite, never claimed FairPlay bypass.

## Agent smoke

Gate M1:  →  →  without a GUI.
