# Plan: pseudocode renderer + `redump --decompile` (`pseudo`, slice 8)

## Input / output

Input, per function: the `irssa::SsaFunction` (post `--ssa-opt`
pipeline: optimize → forward → eliminate_dead), its `irstruct::Node`
tree (`irstruct::structure`), and its `irout::OutOfSsa` variable map
(`irout::out_of_ssa`). Output: deterministic pseudocode text. New
module `src/pseudo.rs` + `redump --decompile[=N]`. **This is the
milestone commit** — the first end-to-end pseudocode (DESIGN.md slice
8; Ghidra `PrintC` precedence lineage; DREAM++ readability line,
Yakdan et al. 2015/2016).

## The renderer is arch-agnostic

Every storage cell is spelled as a **variable** `vN` through
`OutOfSsa::var_of` (SSA name id → variable id), via a `CellNamer`-style
hook whose default is `vN` — so `irstack`/`irtype` names plug in later
without touching the printer. Because names become `vN`, the renderer
needs no per-ISA register table: it is the same for x86-64 and
aarch64. (Live-in parameters may optionally carry their register
spelling as a comment, not as the name — a nicety, keep it behind the
same hook.)

## Precedence-aware parenthesization (the core of the slice)

`ir::render_with` today wraps **every** binary node in parens
(`(a op b)`), which is unreadable. Replace that, in `pseudo` only
(leave `ir::render` faithful for `--lift`/`--ssa`):

- A precedence + associativity table over `ir::BinOp`
  (`Add Sub Mul UDiv SDiv URem SRem And Or Xor Shl LShr AShr
  Eq Ne Ult Ule Slt Sle`) and `ir::UnOp` (`Neg Not` + the width casts
  `ZeroExtend/SignExtend/Truncate`). Model C precedence for the
  arithmetic/relational/bitwise families.
- Parenthesize a child **iff** it binds looser than its context, or
  equal on the non-associative side (Ghidra `PrintC` rule).
- **Forced redundant parentheses** wherever bitwise/shift operators mix
  with comparisons or with each other (the documented C-precedence
  defect — DREAM++). This is a readability override, always applied
  even when C precedence would not require it.
- Width casts (`zext.q`/`sext`/`trunc`) render as the existing
  functional form `cast.w(x)` — unambiguous, no paren question.
- Signedness stays on the operator (`/u`, `%u`, `<s`, `<=u`) per
  `ir::BinOp::token` — no invented C casts; widths explicit
  (`0x7.d`). House style inherited from `ir::render_with`.

**Round-trip invariant (tested, not shipped):** every operator prints
with enough parentheses that a tiny test-only precedence reparser
recovers the same tree. This is the slice's real correctness oracle for
parenthesization — a golden file alone cannot prove non-ambiguity.

## Structure-tree walk

The `Node` tree drives statement order; each construct fetches its
condition expression from the deciding block at print time (conditions
are stored as `Cond { block, negated }` references, not expressions):

- `Block(va)` → the block's statements, minus its terminator `Branch`,
  each rendered: `Assign` → `vN = expr;`, `Store` → `*(t*)addr = val;`,
  `Branch::Call` → `call target(...);` (args not recovered yet —
  render the target, note nothing about params), `Intrinsic` →
  `intrinsic_name(reads...); /* writes... */`.
- `Seq` → its children in order.
- `If { cond, then, else }` → `if (COND) { … } else { … }` where COND
  is the deciding block's branch guard expression, negated per
  `cond.negated`, rendered through the precedence machinery.
- `Loop { While, cond } / { DoWhile, cond } / { SelfLoop }` →
  `while (COND) { … }` / `do { … } while (COND);` / the self-loop form;
  `cond: None` → `while (true)`.
- `Switch { block, cases }` → `switch (SCRUT) { case K: … }` (only the
  proven-table switches `irstruct` emits).
- `Break`/`Continue`/`Goto(va)` → `break;` / `continue;` /
  `goto loc_va;`.
- `Opaque { block, reason }` → the block's statements followed by the
  honesty comment for the reason.

## Edge copies (the `irout` obligation)

`OutOfSsa::edge_copies[(pred,succ)]` and `entry_copies` are rendered as
`vD = vS;` (or the temp form) **at the point the edge is realized in
the tree** — before a `Goto`, or before the successor on a
fall-through. Slice 7's written obligation: copies belong on the edge;
on a critical edge, hoisting them into the predecessor would clobber a
sibling-live variable. Render them at the realized-edge site; where the
tree gives an edge no such site (should not happen for a checked tree),
emit `/* unplaced edge copies */` rather than dropping them — honesty
over silent loss. Document whatever limitation the walk actually has.

## Honesty markers (comments, never silent)

- `/* lift truncated */` — a block with `truncated`.
- `/* reads bits its def never wrote */` — a variable in
  `OutOfSsa::partial`.
- `/* indirect jump: successors unknown */` — an `Opaque::IndirectJump`.
- `/* undecidable exits */` — an `Opaque::Unstructurable`.
- `/* abi-assumed */` — a variable in `OutOfSsa::assumed`.

Each statement carries its originating VA as a right-margin comment
(`// 0x…`) — anchors a future recompile-differential oracle (Liu &
Wang, ISSTA 2020) and keeps every line traceable to bytes.

## Invariants / `check`-style guarantees

- **Deterministic bytes** — golden files; same inputs → byte-equal
  output (assert twice-equal).
- **Depth-bounded** by `ir::MAX_EXPR_NODES`, truncating with an explicit
  `…` (mirror `ir::render_with`); **total** on a malformed tree or a
  malformed function (render what is valid, mark the rest — never
  panic).
- **Round-trip** parenthesization property (the reparser test above).

## Module-by-module

- `src/pseudo.rs`: the renderer, the precedence table, the reparser
  test harness, module docs in the house style (cite PrintC + DREAM++,
  state the forced-paren rule, the vN naming hook, the marker set),
  tests.
- `src/bin/redump.rs`: `--decompile[=N]` — per function run
  `--ssa-opt` pipeline → `irstruct::structure` (with the jump-table map
  as today's `--structure` does) → `irout::out_of_ssa` → `pseudo`.
  x86-64 only for now (same gate as the other IR views); usage text.
- `src/lib.rs`: register `pseudo`.
- `ROADMAP.md`: Current-thread slice 8 ✅ (the milestone), Active
  pointer to the next wave.

## Test matrix (~24)

1. precedence goldens for representative `BinOp`/`UnOp` pairings
   (arith vs arith, arith vs relational, bitwise vs relational,
   shift vs arith), each proving minimal-but-sufficient parens.
2. forced-paren cases: bitwise-mixed-with-comparison and
   shift-mixed-with-bitwise always parenthesized even where C would
   not require it.
3. the round-trip reparser over a generated set of random small
   expression trees → same tree, always.
4. each `Node` kind rendered (block, seq, if/else, while, do-while,
   self-loop, switch, break, continue, goto, opaque×3 reasons).
5. each honesty marker fires from the right source (truncated,
   partial, indirect, unstructurable, abi-assumed).
6. edge copies rendered at the realized edge; a diamond with a
   coalesced φ-web shows zero copies; a case needing a copy shows it
   in the right place.
7. malformed tree / malformed function → total, marked, no panic;
   fuzz over hand-broken trees.
8. determinism (twice → byte-equal).
9. redump e2e: calling + diamond fixtures → golden `--decompile`;
   `/bin/ls` x86-64 slice: no panics, byte-deterministic, and **at
   least one real recovered nontrivial function emits stable, reviewed
   pseudocode** — report it in full in the commit message (the
   milestone evidence). The forwarded `if (rax == -1)`-style condition
   from slice 5 should appear as a real relational `if`.

## Exit criteria (DESIGN, scoped)

`redump --decompile` on a real x86-64 function emits stable, reviewed
pseudocode. **This is the milestone commit.** DESIGN's "both x86-64 and
aarch64" aarch64 half is deliberately deferred to *after* the aarch64
`irlift` dispatch slice lands (the pass is arch-agnostic, so it is a
verification rider, not new code) — recorded as such, not skipped.

## Non-goals (this slice)

- Condition simplification (De Morgan, negation pushing) — those are
  proven W1 rewrites that belong in `irssaopt`, never in the printer;
  the printer only spells.
- Argument/parameter recovery (calls render targets, not args),
  type-aware C casts, struct/array syntax — `irstack`/`irtype` later.
- Line-wrapping beyond a fixed-column rule (Oppen/Wadler only if real
  output demands it).
- aarch64 e2e (rides the dispatch slice).
