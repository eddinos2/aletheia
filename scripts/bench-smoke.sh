#!/usr/bin/env bash
# Headless bench smoke: MCP protocol + redump against fixtures/.
# Prints a machine-readable BENCH_SMOKE_SUMMARY JSON block; exit 0 iff all PASS.
#
# Usage: ./scripts/bench-smoke.sh [--release]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROFILE=debug
CARGO_FLAGS=()
TARGET_DIR="$ROOT/target/debug"
if [[ "${1:-}" == "--release" ]]; then
  PROFILE=release
  CARGO_FLAGS=(--release)
  TARGET_DIR="$ROOT/target/release"
fi

DIAMOND="$ROOT/fixtures/diamond"
SHORT="$ROOT/fixtures/shortcircuit"
DIAMOND_VA="0x1000003d0"
DIAMOND_INTERIOR="0x1000003d4"

for f in "$DIAMOND" "$SHORT"; do
  if [[ ! -f "$f" ]]; then
    echo "error: missing fixture $f" >&2
    exit 1
  fi
done

echo "== build ($PROFILE) =="
cargo build -p aletheia-mcp -p aletheia --bin redump "${CARGO_FLAGS[@]+"${CARGO_FLAGS[@]}"}"
MCP="$TARGET_DIR/aletheia-mcp"
REDUMP="$TARGET_DIR/redump"
if [[ ! -x "$MCP" || ! -x "$REDUMP" ]]; then
  echo "error: expected binaries missing under $TARGET_DIR" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
REQ="$TMP/req.jsonl"
MCP_OUT="$TMP/mcp.out"
RESULTS="$TMP/results.jsonl"
: >"$RESULTS"

ms_now() {
  python3 -c 'import time; print(int(time.time()*1000))'
}

# Append one result line: name status ms_optional note
record() {
  local name="$1" status="$2" ms="${3:-}" note="${4:-}"
  python3 -c '
import json,sys
name,status,ms,note=sys.argv[1:5]
o={"name":name,"status":status}
if ms!="":
  o["ms"]=int(ms)
if note!="":
  o["note"]=note
print(json.dumps(o,separators=(",",":")))
' "$name" "$status" "$ms" "$note" >>"$RESULTS"
}

run_mcp_batch() {
  local t0 t1
  t0="$(ms_now)"
  "$MCP" <"$REQ" >"$MCP_OUT" 2>"$TMP/mcp.err" || {
    record "mcp_batch" "FAIL" "" "aletheia-mcp exited non-zero"
    return 1
  }
  t1="$(ms_now)"
  echo "$((t1 - t0))"
}

# --- MCP pipeline (single process, sequential sessions s1 then s2) ---
cat >"$REQ" <<EOF
{"id":1,"method":"health","params":{}}
{"id":2,"method":"open","params":{"path":"$DIAMOND"}}
{"id":3,"method":"functions","params":{"session":"s1","limit":16}}
{"id":4,"method":"decompile","params":{"session":"s1","entry":"$DIAMOND_VA"}}
{"id":5,"method":"why","params":{"session":"s1","va":"$DIAMOND_VA"}}
{"id":6,"method":"cfg","params":{"session":"s1","entry":"$DIAMOND_VA"}}
{"id":7,"method":"xrefs","params":{"session":"s1","va":"$DIAMOND_VA"}}
{"id":8,"method":"locate","params":{"session":"s1","va":"$DIAMOND_INTERIOR"}}
{"id":9,"method":"rename","params":{"session":"s1","va":"$DIAMOND_VA","name":"bench_renamed"}}
{"id":10,"method":"functions","params":{"session":"s1","limit":16}}
{"id":11,"method":"open","params":{"path":"$SHORT"}}
{"id":12,"method":"diff","params":{"session_a":"s1","session_b":"s2"}}
{"id":13,"method":"patch_preview","params":{"session":"s1","va":"$DIAMOND_VA"}}
EOF

echo "== MCP protocol =="
MCP_MS="$(run_mcp_batch)" || true

python3 - "$MCP_OUT" "$RESULTS" "$DIAMOND_VA" <<'PY'
import json, sys
path, results_path, diamond_va = sys.argv[1:4]
lines = [ln.strip() for ln in open(path) if ln.strip()]
by_id = {}
for ln in lines:
    try:
        o = json.loads(ln)
    except json.JSONDecodeError as e:
        with open(results_path, "a") as f:
            f.write(json.dumps({"name":"mcp_parse","status":"FAIL","note":str(e)}) + "\n")
        sys.exit(0)
    by_id[o.get("id")] = o

def ok(o):
    return bool(o) and o.get("ok") is True and "result" in o

def result(oid):
    return (by_id.get(oid) or {}).get("result") or {}

checks = []

def add(name, passed, note=""):
    checks.append((name, "PASS" if passed else "FAIL", note))

o1 = by_id.get(1)
add("mcp_health", ok(o1) and "engine_version" in result(1),
    result(1).get("engine_version", ""))

o2 = by_id.get(2)
r2 = result(2)
add("mcp_open_diamond", ok(o2) and r2.get("session_id") == "s1" and r2.get("arch") == "X86_64",
    r2.get("hash", ""))

r3 = result(3)
names = [f.get("name") for f in (r3.get("functions") or [])]
add("mcp_functions", ok(by_id.get(3)) and "_diamond" in names, f"names={names}")

r4 = result(4)
pc = r4.get("pseudocode") or ""
add("mcp_decompile", ok(by_id.get(4)) and len(pc) > 0, f"pseudo_len={len(pc)}")

r5 = result(5)
labs = [c.get("lab") for c in (r5.get("chain") or [])]
add("mcp_why", ok(by_id.get(5)) and all(x in labs for x in ("CLAIM", "SOURCE", "VERDICT")),
    f"labs={labs}")

r6 = result(6)
add("mcp_cfg", ok(by_id.get(6)) and isinstance(r6.get("blocks"), list)
    and isinstance(r6.get("edges"), list) and int(r6.get("block_count") or 0) >= 1,
    f"block_count={r6.get('block_count')}")

r7 = result(7)
add("mcp_xrefs", ok(by_id.get(7)) and (
    isinstance(r7.get("from"), list) or isinstance(r7.get("to"), list)),
    f"total={r7.get('total')}")

r8 = result(8)
add("mcp_locate", ok(by_id.get(8)) and r8.get("function") == diamond_va
    and r8.get("exact_entry") is False)

r9 = result(9)
delta = r9.get("delta") or {}
add("mcp_rename", ok(by_id.get(9)) and delta.get("kind") == "annotate"
    and "bench_renamed" in json.dumps(r9),
    f"delta={delta.get('kind')}")

r10 = result(10)
names10 = [f.get("name") for f in (r10.get("functions") or [])]
add("mcp_rename_visible", ok(by_id.get(10)) and "bench_renamed" in names10,
    f"names={names10}")

r11 = result(11)
add("mcp_open_shortcircuit", ok(by_id.get(11)) and r11.get("session_id") == "s2")

r12 = result(12)
add("mcp_diff", ok(by_id.get(12)) and isinstance(r12.get("report"), str)
    and len(r12.get("report") or "") > 0,
    f"keys={sorted(r12.keys())}")

r13 = result(13)
rep = r13.get("report") or ""
add("mcp_patch_preview", ok(by_id.get(13)) and len(rep) > 0,
    f"report_len={len(rep)}")

with open(results_path, "a") as f:
    for name, status, note in checks:
        o = {"name": name, "status": status}
        if note:
            o["note"] = note
        f.write(json.dumps(o, separators=(",", ":")) + "\n")
PY

# Attach batch wall time to a synthetic row
record "mcp_batch_wall" "PASS" "$MCP_MS" "all MCP requests in one process"

# --- redump headless ---
echo "== redump =="
run_redump() {
  local name="$1"; shift
  local t0 t1 out ec=0
  t0="$(ms_now)"
  set +e
  out="$("$REDUMP" "$@" 2>"$TMP/redump.err")"
  ec=$?
  set -e
  t1="$(ms_now)"
  local ms=$((t1 - t0))
  if [[ $ec -ne 0 ]]; then
    record "$name" "FAIL" "$ms" "exit=$ec"
    return
  fi
  if [[ -z "${out// }" ]]; then
    record "$name" "FAIL" "$ms" "empty stdout"
    return
  fi
  record "$name" "PASS" "$ms" "stdout_len=${#out}"
}

run_redump "redump_listing" "$DIAMOND" --listing=8
run_redump "redump_decompile" "$DIAMOND" --decompile=4
run_redump "redump_json" "$DIAMOND" --json
run_redump "redump_diff" "$DIAMOND" --diff "$SHORT"

# --- summary ---
python3 - "$RESULTS" "$PROFILE" <<'PY'
import json, sys, platform, os
results_path, profile = sys.argv[1:3]
rows = [json.loads(ln) for ln in open(results_path) if ln.strip()]
failed = [r for r in rows if r.get("status") != "PASS"]
passed = [r for r in rows if r.get("status") == "PASS"]
summary = {
    "suite": "aletheia-bench-smoke",
    "profile": profile,
    "host": {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    },
    "cwd": os.getcwd(),
    "pass": len(passed),
    "fail": len(failed),
    "total": len(rows),
    "ok": len(failed) == 0,
    "steps": rows,
}
print()
print("BENCH_SMOKE_SUMMARY")
print(json.dumps(summary, indent=2, sort_keys=False))
sys.exit(0 if summary["ok"] else 1)
PY
