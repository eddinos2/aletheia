#!/usr/bin/env bash
# Build tiny redistributable adversarial fixtures when a C compiler is present.
# Never downloads proprietary samples. Safe to re-run (idempotent).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/fixtures/adversarial"
mkdir -p "$OUT/version_pair" "$OUT/go_stripped" "$OUT/rust_panic" "$OUT/macho_arm64"

have() { command -v "$1" >/dev/null 2>&1; }

CC="${CC:-}"
if [[ -z "$CC" ]]; then
  if have clang; then CC=clang
  elif have gcc; then CC=gcc
  else
    echo "gen-adversarial-fixtures: no clang/gcc; wrote layout only under $OUT"
    exit 0
  fi
fi

# version_pair: two near-identical builds (NOP padding differs) for diff/patch.
cat >"$OUT/version_pair/a.c" <<'EOF'
int add(int x, int y) { return x + y; }
int main(void) { return add(2, 3); }
EOF
cat >"$OUT/version_pair/b.c" <<'EOF'
int add(int x, int y) { return x + y; }
int main(void) {
  /* intentional trivial change for patch-diff benches */
  return add(2, 4);
}
EOF

"$CC" -O0 -g0 -fno-ident -o "$OUT/version_pair/old.bin" "$OUT/version_pair/a.c" 2>/dev/null \
  || "$CC" -O0 -o "$OUT/version_pair/old.bin" "$OUT/version_pair/a.c"
"$CC" -O0 -g0 -fno-ident -o "$OUT/version_pair/new.bin" "$OUT/version_pair/b.c" 2>/dev/null \
  || "$CC" -O0 -o "$OUT/version_pair/new.bin" "$OUT/version_pair/b.c"

# Copy in-tree diamond as a known-good thin ELF/Mach-O for scorecard baseline.
if [[ -f "$ROOT/fixtures/diamond" ]]; then
  cp -f "$ROOT/fixtures/diamond" "$OUT/baseline_diamond"
fi

# Optional Go/Rust — only if toolchains exist (not required).
if have go; then
  mkdir -p "$OUT/go_stripped/src"
  cat >"$OUT/go_stripped/hello.go" <<'EOF'
package main
import "fmt"
func main() { fmt.Println("aletheia") }
EOF
  (cd "$OUT/go_stripped" && go build -ldflags="-s -w" -o hello hello.go) || true
fi

if have rustc; then
  cat >"$OUT/rust_panic/main.rs" <<'EOF'
fn main() { let x = 1 + 1; std::process::exit(x); }
EOF
  rustc -O -C debuginfo=0 -o "$OUT/rust_panic/hello" "$OUT/rust_panic/main.rs" 2>/dev/null || true
fi

echo "gen-adversarial-fixtures: OK → $OUT"
ls -la "$OUT/version_pair" 2>/dev/null || true
