# Public scorecard (workflow)

Compare Aletheia to other RE toolkits on **measurable workflows**, not
marketing. Fill locally; optional CI can run the headless column.

## Environment

| Field | Value |
|---|---|
| Date | |
| Host OS / CPU | |
| Aletheia commit | |
| Competitor + version | |
| Fixture set | see [ADVERSARIAL_FIXTURES.md](ADVERSARIAL_FIXTURES.md) |

## Headless (scripted)

```console
./scripts/bench-smoke.sh
```

| Step | Aletheia | Competitor | Notes |
|---|---|---|---|
| Open thin binary | | | |
| List functions | | | |
| Decompile entry | | | |
| Diff two builds | | | |
| Patch preview | | | |
| Typefacts conflict honesty | | | signed∩unsigned → `conflict` |

Paste `BENCH_SMOKE_SUMMARY` JSON here:

```json
```

## GUI (manual timed)

Use [GUI_BENCH_CHECKLIST.md](GUI_BENCH_CHECKLIST.md).

| Step | Aletheia (s) | Competitor (s) | Pass? |
|---|---|---|---|
| Open → first decompile | | | |
| Rename round-trip | | | |
| Xref click-nav | | | |
| CFG view | | | |
| Diff buckets | | | |

## Scoring rubric (suggested)

- **3** — works, honest (proven vs heuristic), fast enough for triage
- **2** — works with caveats / incomplete depth
- **1** — fails or silently wrong
- **0** — unsupported

Dimensions: openness, agent/MCP, annotations (git), patch auditability,
decompiler fidelity, iOS depth, GUI polish.

## Non-goals

Do not claim FairPlay bypass, proprietary DB compatibility, or “drop-in
IDA UI clone.” Score workflows researchers actually run.
