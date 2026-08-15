# Aletheia phase gates (multi-track)

Gates must green before that track expands. Checked by humans + CI where
noted.

## Gate E1 — Engine deepen

- [ ] `--decompile` prints `local_*` names from `irstack` slots on at least
      one fixture function
- [ ] Callee-side `sig` params render as `f(a, b)` shape (or documented
      interim: stack dump proves slots)
- [ ] `cargo test evalfx` green; `irstack` unit tests green

**Interim (this landing):** `irstack` affine+slots + `--stack` CLI —
Gate E1 partial until `sig` lands.

## Gate I1 — iOS foundation

- [ ] Chained fixups header + imports parse on a real or synthetic image
- [ ] Encrypted segment detected and reported (no silent decompile of
      ciphertext without a warning)
- [ ] PAC branches remain real Flow (already true post-wave3)

## Gate P1a — Patch

- [x] PatchSet schema in `patch`
- [x] `--patch-preview` / `--patch-apply` sibling write
- [x] Precondition failure is typed, not panic

## Gate M1 — MCP

- [x] `aletheia-mcp` stdio skeleton with `health` / `open` / `decompile` /
      `why` tools (progress/cancel stubs)
- [ ] Agent can decompile without GUI (manual smoke)

## Gate G1 — TUI

- [ ] Navigate + rename + Why? + decompile toggle over protocol
