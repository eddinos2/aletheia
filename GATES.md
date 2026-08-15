# Aletheia phase gates (multi-track)

Gates must green before that track expands. Checked by humans + CI where
noted.

## Gate E1 — Engine deepen

- [x] `--decompile` prints `local_*` names from `irstack` slots on at least
      one fixture function
- [x] Callee-side `sig` params render as `f(a, b)` shape (or documented
      interim: stack dump proves slots)
- [x] `cargo test evalfx` green; `irstack` unit tests green

**Done:** `irstack` + `mempromote` namers and `sig::render_header` wired
into `redump --decompile` and `aletheia-mcp` decompile.

## Gate I1 — iOS foundation

- [x] Chained fixups header + imports parse on a real or synthetic image
- [x] Encrypted segment detected and reported (no silent decompile of
      ciphertext without a warning)
- [x] PAC branches remain real Flow (already true post-wave3)

## Gate P1a — Patch

- [x] PatchSet schema in `patch`
- [x] `--patch-preview` / `--patch-apply` sibling write
- [x] Precondition failure is typed, not panic

## Gate M1 — MCP

- [x] `aletheia-mcp` stdio skeleton with `health` / `open` / `decompile` /
      `why` tools (progress/cancel stubs)
- [x] Agent can decompile without GUI (manual smoke via stdio JSON)

## Gate G1 — GUI / protocol surface

- [x] Navigate + rename + Why? + decompile toggle over protocol
      (`aletheia-gui` → `aletheia_mcp::handle_line`)
- [x] Richer xref click-navigation + CFG graph layout (frontend-only)
- [x] Incremental analysis deltas on the wire

**Shipped:** three-region egui workstation, trust channel, diff buckets,
patch preview, clickable bidirectional xrefs (`locate` + nav stack),
layered CFG view over engine `cfg` edges, rename `delta` /
`invalidate` (see [protocol/PROTOCOL.md](protocol/PROTOCOL.md)),
macOS `.app`/`.dmg` scripts (unsigned). Checklist:
[docs/GUI_BENCH_CHECKLIST.md](docs/GUI_BENCH_CHECKLIST.md).
