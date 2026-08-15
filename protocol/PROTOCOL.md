# Aletheia protocol v1 (Track 4 spine)

Engine-owned messages. Frontends (CLI, MCP, GUI) **compute nothing** —
they only render. Implemented by `aletheia_mcp::handle_line` (stdio binary
`aletheia-mcp` and in-process `aletheia-gui`).

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
| `xrefs` | `{ session_id, va }` | `from` / `to` rows `{ from, to, kind, label? }` |
| `cfg` | `{ session_id, entry }` | `{ blocks, edges, block_count }` — successor edges only |
| `locate` | `{ session_id, va }` | `{ function?, block?, exact_entry }` |
| `rename` | `{ session_id, anchor, name }` | `{ tip, delta }` — see [Deltas](#incremental-deltas) |
| `diff` | `{ session_a, session_b }` | buckets |
| `patch_preview` | `{ session_id, patchset }` | report |
| `patch_apply` | `{ session_id, patchset, sibling? }` | `{ path }` |
| `why` | `{ session_id, va \| fact_id }` | provenance chain (`funcs::Source`) |

## Rules

- Every response carries `stamp = { binary_hash, engine_version }`.
- Long tools return `job_id`; `health` never blocks on jobs.
- Heuristic vs proven vs asserted is explicit on every fact (trust channel).
- Patch apply defaults to sibling write; never silent in-place.
- GUI and agents must stay isomorphic: one dispatch path, two transports.
- Frontends may **lay out** graphs; they must never invent CFG edges or
  xref endpoints. `cfg.edges` / `xrefs.from|to` are engine successors only.

## Incremental deltas

After an assertion (`rename` today; future annotate/comment), the engine
returns a small `delta` so clients patch navigator state **without** a
blind full `functions` reload:

```json
{
  "tip": { "seq": 1, "id": "0x…", "field": "name", "va": "0x…", "name": "…" },
  "delta": {
    "kind": "annotate",
    "functions": [{ "va": "0x…", "name": "…", "source": "asserted" }],
    "invalidate": [
      { "view": "listing", "va": "0x…" },
      { "view": "why", "va": "0x…" },
      { "view": "decompile", "va": "0x…" }
    ]
  },
  "stamp": { "hash": "0x…", "engine_version": "…" }
}
```

Client policy:

1. Apply `delta.functions` into the in-memory navigator (name + `source`).
2. Soft-refetch only the `invalidate` views that match the current caret.
3. Optional: call `functions` again — asserted names overlay CFG names
   via `annotate::Db` — but that is a clean refetch, not required after
   every rename.

`invalidate` is advisory. Unknown `view` strings are ignored. Granularity
today is per-function VA; finer per-fact ids can land later without
breaking this shape.

## Navigation helpers

- **`xrefs`**: click `from` rows → jump to `to`; click `to` rows → jump
  to `from` (bidirectional). Use `locate` when the address is not a
  function entry.
- **`cfg`**: `{ start, end, terminator, successors }` per block plus
  explicit `{ from, to }` edges copied from successors. Layout is a
  frontend concern (layered / Sugiyama-lite).
- **`locate`**: maps any VA to owning `{ function, block }` from recovered
  CFG ranges — never invents a container.
