# GUI + engine bench checklist

Adversarial workflow comparison vs commercial RE suites: **same analyst steps**,
timed, pass/fail. Not a trademark parody — measure open → functions → decompile →
rename → xref nav → CFG → diff → patch preview.

**Fixtures** (Mach-O x86_64, see [fixtures/README.md](../fixtures/README.md)):

| Path | Role |
|---|---|
| `fixtures/diamond` | Primary: `_diamond` @ `0x1000003d0` |
| `fixtures/shortcircuit` | Diff partner / alternate open |
| `fixtures/loop_bc`, `fixtures/switch_dense`, `fixtures/tail_merge` | Optional stress |

**Engine stamp:** every MCP/GUI response carries `stamp = { hash, engine_version }`.
Record both in any published table.

**Headless automation:** run `./scripts/bench-smoke.sh` first (MCP + `redump`).
GUI interactive steps below stay manual.

**Baseline numbers:** [docs/BENCH_BASELINE.md](BENCH_BASELINE.md).

---

## Timing rules

- Wall clock from first keystroke/click that starts the step to visible stable UI
  (or MCP JSON line complete for headless).
- Cold = process just started; warm = same process, binary already open.
- Mark **PASS** / **FAIL** / **SKIP**. FAIL on panic, blank pane, invented CFG edges,
  or rename that vanishes on re-select.

Target budget for a full manual GUI pass on `diamond` + one diff: **≤ 5 min**.

---

## A. Headless smoke (scripted)

```console
$ ./scripts/bench-smoke.sh
```

| # | Step | Fixture / VA | Expect | P/F | Time |
|---|---|---|---|---|---|
| A1 | `health` | — | `ok:true`, `engine_version` | | |
| A2 | `open` | `fixtures/diamond` | `session_id`, arch `X86_64`, hash | | |
| A3 | `functions` | session, limit 16 | includes `_diamond` | | |
| A4 | `decompile` | entry `0x1000003d0` | non-empty `pseudocode` (`local_*` / sig) | | |
| A5 | `why` | va `0x1000003d0` | chain labs CLAIM / SOURCE / VERDICT | | |
| A6 | `cfg` | entry `0x1000003d0` | `blocks` + `edges` + `block_count` ≥ 1 | | |
| A7 | `xrefs` | va `0x1000003d0` | `from` and/or `to` with `kind` | | |
| A8 | `locate` | va `0x1000003d4` | `function` = `0x1000003d0`, `exact_entry:false` | | |
| A9 | `rename` | va → `bench_renamed` | `delta.kind=annotate`, invalidate views | | |
| A10 | `open` + `diff` | `diamond` vs `shortcircuit` | report + counts/hunks | | |
| A11 | `patch_preview` | va `0x1000003d0` | non-empty `report`, edits present | | |
| A12 | `redump` listing / decompile / `--json` / `--diff` | same fixtures | exit 0, non-empty stdout | | |

Script prints a machine-readable `BENCH_SMOKE_SUMMARY` JSON block. Exit 0 = all PASS.

---

## B. GUI interactive (manual, ~3–5 min)

Start: `cargo run -p aletheia-gui` (or `open dist/Aletheia.app`).

| # | Step | Action | Fixture / note | Expect | P/F | Time |
|---|---|---|---|---|---|---|
| B1 | Open | ⌘O | `fixtures/diamond` | Stamp in top bar; no panic | | |
| B2 | Functions | — | navigator list | Names + trust ● proven / ○ heuristic | | |
| B3 | Select | click `_diamond` | `0x1000003d0` | Listing symbolized | | |
| B4 | Decompile | `y` | same | Pseudocode pane non-empty | | |
| B5 | Rename | `n` → `bench_renamed` Enter | delta path | Navigator updates; status invalidate; no full reload flash | | |
| B6 | Re-select | click away then `_diamond` / renamed | — | Name still `bench_renamed` | | |
| B7 | Xref nav | XREFS panel: click incoming `to` row | caller of `_diamond` | Jumps to caller; `[` returns | | |
| B8 | CFG | `c` | `_diamond` | Layered blocks; edges ⊆ engine successors only | | |
| B9 | Why? | `?` | caret on function | Provenance pin CLAIM / SOURCE / VERDICT | | |
| B10 | Diff | ⌘D | `fixtures/shortcircuit` | Diff buckets + report text | | |
| B11 | Patch preview | `p` | NOP-at-entry recipe | Preview text; no apply unless intended | | |

**Pass criterion:** no panic; stamp visible; CFG edges match engine; rename survives re-select.

---

## C. Packaging smoke (macOS)

```console
$ ./scripts/macos-app.sh --release && open dist/Aletheia.app
$ ./scripts/macos-dmg.sh --release && ls -lh dist/*.dmg
```

| # | Step | Expect | P/F |
|---|---|---|---|
| C1 | `.app` builds | `dist/Aletheia.app` launches | |
| C2 | `.dmg` builds | `dist/Aletheia-*-unsigned.dmg` exists | |

Builds are **ad-hoc / unsigned** for local use. Notarization requires a real Apple
Developer ID — see [crates/aletheia-gui/README.md](../crates/aletheia-gui/README.md).

---

## D. Comparison table (fill when benching a suite)

Record wall times (seconds) on the **same machine** for Aletheia vs suite under test.
Use identical fixtures. Do not invent competitor numbers.

| Workflow step | Aletheia GUI | Aletheia headless | Suite X (manual) | Suite X (scripted) |
|---|---|---|---|---|
| Cold open + functions | | A2+A3 | | |
| First decompile | B4 | A4 | | |
| Rename round-trip | B5–B6 | A9 | | |
| Xref jump + back | B7 | A7+A8 | | |
| CFG view | B8 | A6 | | |
| Diff two fixtures | B10 | A10 | | |
| Patch preview | B11 | A11 | | |
| Engine / product version | stamp | stamp | | |
| Binary hash (`diamond`) | stamp.hash | stamp.hash | n/a | n/a |
