# Aletheia GUI

Thin native workstation UI over the **same protocol** as `aletheia-mcp`
([protocol/PROTOCOL.md](../../protocol/PROTOCOL.md)). The GUI computes
nothing: open / functions / listing / decompile / rename / why / xrefs /
cfg / locate / diff / patch_preview all go through
`aletheia_mcp::handle_line`.

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
| `n` | Rename (asserted → annotate log + wire `delta`) |
| `y` / `u` | Decompile / Listing toggle |
| `c` | CFG graph (engine edges, layered layout) |
| `[` / Backspace | Navigate back (xref / go-to stack) |
| `?` | Pin provenance (Why?) |
| `p` | Patch preview (NOP recipe at entry) |

Click an xref row in the right panel: outgoing jumps to `to`, incoming
jumps to `from` (via protocol `locate` when the VA is not an entry).

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

Requires macOS + Xcode CLT. `./scripts/macos-dmg.sh --release` builds the
app then packages `dist/Aletheia-<version>-unsigned.dmg` (create-dmg if
installed, else `hdiutil`).

### Signing honesty

Local scripts produce an **ad-hoc** signature when `codesign` is available
(`codesign --sign -`), otherwise a fully **unsigned** app/DMG. That is
enough to `open dist/Aletheia.app` on the build machine. It is **not**
Developer ID signing and **not** notarized — Gatekeeper will block
distribution to other Macs until you notarize with a paid Apple Developer
account. These scripts do not fake or stub notarization.

### Notarization prerequisites (you bring these)

Not shipped / not automated here. You need:

1. Apple Developer Program membership and a **Developer ID Application**
   certificate installed in the login keychain.
2. An App Store Connect API key (or Apple ID + app-specific password) for
   `notarytool`.
3. Hardened runtime entitlements appropriate for your binary (GUI + any
   spawned helpers).

Typical sequence after a release app build:

```console
$ codesign --deep --force --options runtime \
    --sign "Developer ID Application: Your Name (TEAMID)" \
    dist/Aletheia.app
$ ./scripts/macos-dmg.sh --release   # or package the already-signed .app
$ xcrun notarytool submit dist/Aletheia-*-unsigned.dmg \
    --apple-id "…" --team-id "…" --password "…" --wait
$ xcrun stapler staple dist/Aletheia-*.dmg
```

Rename the DMG away from `-unsigned` once stapled if you distribute it.
Until then, treat artifacts as local-only.

## Bench / smoke

Headless protocol + `redump`: `./scripts/bench-smoke.sh`  
Timed GUI checklist: [docs/GUI_BENCH_CHECKLIST.md](../../docs/GUI_BENCH_CHECKLIST.md)

## Gate G1 status

All G1 bullets green: navigate · rename · Why? · decompile · bidirectional
xref click-nav · CFG graph · incremental rename deltas — tracked in
[GATES.md](../../GATES.md).
