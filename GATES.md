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
into `redump --decompile` and `aletheia-mcp` decompile. Caller-side
`sig::try_confirm_returns` upgrades AbiAssumed returns when the call
graph is cheap; typed prototypes via `irtype` + `types::ParamTypeMap::render_proto`.

## Gate E2 — Engine fidelity (post-G1)

- [x] `confirm_returns` on decompile / MCP / `--sigs` when CG edges ≤ cap
- [x] Typed pseudo headers from irtype evidence (safe presentation)
- [x] ObjC `__objc_selrefs` listing
- [x] Swift `__swift5_proto` conformance detail (protocol/type/flags)
- [x] A64 `PRFM` (unsigned offset) + scalar `CRC32*`
- [x] Patch `assemble_patch` beyond NOP (RET / B / BR / MOVZ)
- [x] Open FLIRT sample under `testdata/flirt/sample.corpus`

## Gate E3 — Type bounds lattice (DESIGN 16–17)

- [x] Finite lattice with per-name `[lower .. upper]` (`src/typebounds.rs`)
- [x] Directional φ / copy propagation; signed∩unsigned → explicit Conflict
- [x] `check`: lower ≤ upper (or dual Conflict); `--typefacts` dumps bounds
- [x] Presentation: Proven / Guess / `/* conflicting evidence */` tokens

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

## Gate F1 — FLIRT / adversarial pack

- [x] Expanded open corpus at `testdata/flirt/sample.corpus`
- [x] `--flirt` defaults to that corpus when present
- [x] `scripts/gen-adversarial-fixtures.sh` + `fixtures/adversarial/version_pair`
- [x] Scorecard template remains [docs/SCORECARD.md](docs/SCORECARD.md)

## Gate S1 — Scripting / plugin ABI

- [x] Versioned `aletheia::api` (`API_VERSION` 1.0.0) with `AnalysisSession`
- [x] Documented contract in [docs/PLUGIN_ABI.md](docs/PLUGIN_ABI.md)
- [x] MCP `health` exposes `api_version` (wire handshake)
- [x] In-process tests: open → functions → decompile → rename

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
