# Aletheia GUI

Thin native workstation UI over the **same protocol** as `aletheia-mcp`
([protocol/PROTOCOL.md](../../protocol/PROTOCOL.md)). The GUI computes
nothing: open / functions / listing / decompile / rename / why / xrefs /
diff / patch_preview all go through `aletheia_mcp::handle_line`.

Design north star: [DESIGN_GUI.md](../../DESIGN_GUI.md). Visual tokens
match `design/aletheia-gui-mockup.html` (dark-first, trust channel,
keyboard-first).

## Run (dev)

```console
$ cargo run -p aletheia-gui
```

Then **⌘O** and pick `fixtures/diamond` (or any PE/ELF/Mach-O).

| Key | Action |
|---|---|
| ⌘O | Open binary |
| ⌘D | Open second binary → Diff |
| ⌘K | Command palette |
| `g` | Go to address / symbol |
| `n` | Rename (asserted → annotate log) |
| `y` / `u` | Decompile / Listing toggle |
| `?` | Pin provenance (Why?) |
| `p` | Patch preview (NOP recipe at entry) |

## Architecture

```
aletheia-gui  ──handle_line──►  aletheia-mcp (lib)  ──►  aletheia (engine)
     │                              │
     │                              └── same JSON as stdio MCP agents
     └── egui/eframe shell only
```

Core crate stays zero mandatory deps. GUI toolkit deps live only here.

## macOS `.app` / `.dmg`

```console
$ ./scripts/macos-app.sh --release   # → dist/Aletheia.app
$ ./scripts/macos-dmg.sh --release   # → dist/Aletheia-*-unsigned.dmg
```

Requires macOS + Xcode CLT. Builds are **ad-hoc signed / unsigned** for
local use. Distribution notarization needs an Apple Developer ID:

1. `codesign --deep --force --options runtime --sign "Developer ID Application: …" dist/Aletheia.app`
2. Package DMG, then `xcrun notarytool submit … --wait`
3. `xcrun stapler staple dist/Aletheia-*.dmg`

## Gate G1 status

Shipped here: navigate · rename · Why? · decompile toggle · diff buckets ·
patch preview — all over protocol. Remaining polish (CFG graph, richer
xref click-nav, incremental deltas) is tracked in [GATES.md](../../GATES.md).
