# GUI + engine smoke / bench checklist

Use this after GUI or protocol changes, and when benchmarking Aletheia
against other RE workflows (agent-native open → decompile → rename → diff).

Fixture: `fixtures/diamond` (Mach-O x86_64). Alternate: `fixtures/shortcircuit`.

## A. Protocol / MCP (headless)

```console
$ cargo build -p aletheia-mcp
$ printf '%s\n' \
  '{"id":1,"method":"health","params":{}}' \
  '{"id":2,"method":"open","params":{"path":"fixtures/diamond"}}' \
  '{"id":3,"method":"functions","params":{"session":"s1","limit":8}}' \
  '{"id":4,"method":"decompile","params":{"session":"s1","entry":0}}' \
  '{"id":5,"method":"why","params":{"session":"s1","va":"0x100000e70"}}' \
  | ./target/debug/aletheia-mcp
```

Adjust `va` from the `functions` response. Expect: `ok:true`, non-empty
`pseudocode`, and a `chain` with CLAIM/SOURCE/VERDICT on `why`.

## B. GUI smoke (manual, ~2 min)

1. `cargo run -p aletheia-gui`
2. ⌘O → `fixtures/diamond`
3. Function list populates with trust marks (● proven / ○ heuristic)
4. Select a named function → Listing shows symbolized disasm
5. Press `y` → Decompile pane shows pseudocode (`local_*` / sig header)
6. Press `?` → Provenance pin shows CLAIM / SOURCE / VERDICT
7. Press `n`, rename to `bench_renamed`, Enter → navigator updates (asserted)
8. ⌘D → open `fixtures/shortcircuit` → Diff buckets + report
9. Press `p` → Patch preview text for NOP-at-entry

Pass if no panic, stamp visible in top bar, and rename survives a
re-select of the same function in-session.

## C. Packaging smoke (macOS)

```console
$ ./scripts/macos-app.sh --release && open dist/Aletheia.app
$ ./scripts/macos-dmg.sh --release && ls -lh dist/*.dmg
```

## D. Timing notes (for later adversarial comparison)

Record wall time for: cold open+functions, first decompile, rename round-trip,
diff of two fixtures. Engine version + binary hash are on the stamp; include
them in any published bench table.
