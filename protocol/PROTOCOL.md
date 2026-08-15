# Aletheia protocol v1 (Track 4 spine)

Engine-owned messages. Frontends (CLI, MCP, TUI, GUI) **compute nothing** —
they only render. Schema evolves with TUI use; this file is the starting
contract for `aletheia-mcp` and Gate M1.

## Session

| Method | Request | Response |
|---|---|---|
| `health` | `{}` | `{ ok, engine_version, busy_jobs }` |
| `open` | `{ path }` or `{ bytes_b64 }` | `{ session_id, arch, hash, encrypted? }` |
| `analyze` | `{ session_id, threads? }` | `{ job_id }` → progress events |
| `cancel` | `{ job_id }` | `{ cancelled }` |
| `functions` | `{ session_id, limit? }` | `[{ va, name?, source }]` |
| `listing` | `{ session_id, entry, max_insns? }` | text |
| `decompile` | `{ session_id, entry }` | text + `stamp` |
| `stack` | `{ session_id, entry }` | `irstack` dump |
| `xrefs` | `{ session_id, va }` | to/from |
| `rename` | `{ session_id, anchor, name }` | annotate log tip |
| `diff` | `{ session_a, session_b }` | buckets |
| `patch_preview` | `{ session_id, patchset }` | report |
| `patch_apply` | `{ session_id, patchset, sibling? }` | `{ path }` |
| `why` | `{ session_id, fact_id }` | provenance chain |

## Rules

- Every response carries `stamp = { binary_hash, engine_version }`.
- Long tools return `job_id`; `health` never blocks on jobs.
- Heuristic vs proven vs asserted is explicit on every fact (trust channel).
- Patch apply defaults to sibling write; never silent in-place.
