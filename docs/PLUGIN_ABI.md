# Aletheia scripting / plugin ABI

Stable contract for embedders, agents, and future language bindings.
**Engine owns truth** — hosts render; they do not invent analysis facts.

## Versions

| Symbol | Meaning |
|---|---|
| `aletheia::api::API_VERSION` | Scripting surface (`"1.0.0"`). **Major** bumps are breaking. |
| `CARGO_PKG_VERSION` / `ENGINE_VERSION` | Crate / binary version. |
| Protocol `aletheia/1` | NDJSON wire shape in [`protocol/PROTOCOL.md`](../protocol/PROTOCOL.md). |

Handshake (in-process):

```rust
use aletheia::api;
assert!(api::handshake().contains(api::API_VERSION));
```

Out-of-process: `health` on `aletheia-mcp` includes `engine_version`; hosts
may also call `open` → `decompile` as the wire twin of the Rust API.

## In-process API (`aletheia::api`)

```rust
use aletheia::api::AnalysisSession;

let mut s = AnalysisSession::open_path("target.bin")?;
let fns = s.functions(64)?;
let text = s.decompile(fns[0].va)?;
s.rename(fns[0].va, "main_logic")?;
```

Methods (v1.0): `open_path`, `open_bytes`, `arch`, `hash`, `encrypted`,
`functions`, `listing`, `decompile`, `rename`, `why`, `xrefs`.

Errors are [`ApiError`] — never panics on hostile input.

## Wire isomorphism

| API | MCP / GUI method |
|---|---|
| `open_*` | `open` |
| `functions` | `functions` |
| `decompile` | `decompile` |
| `listing` | `listing` |
| `rename` | `rename` (+ delta) |
| `why` | `why` |
| `xrefs` | `xrefs` |

## Plugin loading (future)

v1 does **not** `dlopen` guest `.so`/`.dylib` inside the core crate (keeps
zero mandatory deps and a clean trust boundary). Hosts that want dynamic
plugins should:

1. Speak protocol NDJSON to `aletheia-mcp`, or
2. Link `aletheia` and call `api::AnalysisSession`, or
3. (Later) ship an optional `aletheia-plugin-host` crate with a versioned
   `extern "C"` table gated on `API_VERSION` major.

## Compatibility promise

- Additive methods and fields may land in minor versions.
- Removing / renaming / changing semantics of existing methods requires a
  major `API_VERSION` bump and a ROADMAP note.
