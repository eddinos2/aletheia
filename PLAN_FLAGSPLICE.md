# Plan: the load-backed flag-operand splice (`irssaopt` + `irflow`)

## Goal

Retire the load-backed half of the paired-flag residue that the
forwarding-policy slice measured and diagnosed: one `cmp` with a
*memory operand* feeding two jccs. The lift spells the memory operand
as a block-local temp backed by a load; the SF/OF trees read that
temp, so the pair can never legally sit in the second jcc's block
(`ir::check`'s temp rule), the load may not forward (multi-use), and
`irflow`'s pair patterns — whose shared-operand equality requires
load-freedom — could not fire even if it did. The milestone comparator
`sub_10000073f` is the type specimen: `jg` folds in the `cmp`'s own
block, `jge` one block later still reads `if (v11 == v12)` over two
standing flag definitions. Measured on the current tree, the shape is
the *unswept OF-tree assignment* (`vN = ((a ^ b) & (a ^ a - b)) <s 0x0`
as a statement): ls 19, bash 196, brotli 12.

Explicitly out of scope: pairs whose halves reach the comparison
through different SSA names for one value (value-numbering territory —
a sibling slice owns it), and every `fold_expr` identity outside the
pair families' shared-operand rule.

## The two theorems this slice stands on

**One expression, one memory state.** Expressions are effect-free
(`Store`/`Intrinsic`/`Branch` are statements), so every `Load` node
inside a single statement's expression reads the same memory state.
Two structurally equal load-bearing subtrees in one expression are
therefore *value*-equal: equal pure address operands (SSA names read
at one point), equal width, equal state. A pair rewrite that keeps one
copy and drops the duplicate is value-sound. What it changes is the
dynamic *count* of loads — observable only for volatile/MMIO memory —
and that rides the same conforming-code assumption `callfx` already
records; documented, not hidden.

**The effect-clear region.** A load-bearing tree re-evaluated at a use
site reads the value it had at its definition iff no statement on any
def→use path between the two touches memory. The def dominates the use
(SSA), so every dynamic path to the use passes the def and then stays
inside the region `R = {blocks reachable from the def's block that
reach the use's block}`. If every statement in the window (def block
after the def, intermediate blocks whole, use block before the use) is
effect-free — no `Store`, no `Intrinsic`, every terminator inside `R` a
known-target `Jump` (a `Call`/`Return`/unproven-indirect terminator
refuses) — the re-read is provably the same value. Cyclic regions are
refused outright this slice (the acyclicity check is cheap and the
paired-jcc shape never needs a cycle); a region over
`MAX_LOAD_REGION_BLOCKS` (4) is refused and counted, never
approximated.

## The mechanism (`irssaopt::plan_forwards`)

1. **The cone gains loads.** A definition whose tree reads block-local
   temps currently splices only where the cascade already folded the
   temps' *pure* defs in. New: build the cone tree by inlining a
   temp's def when that def is a load (transitively, pure defs as
   before); a temp def that is neither pure nor a load refuses as
   today. Division anywhere in the cone refuses — the trap doctrine is
   untouched.
2. **Load-bearing defs earn sites jointly, all-or-nothing, under a
   function-level shrink.** The per-site textual test cannot see this
   shape (the site `Eq(v11, v12)` is 3 nodes; the folded relation with
   the load inline is bigger — the win is the two dying definition
   statements). So: a load-bearing cone def may clear its sites only
   when (a) *every* use of the def is a cleared site (the def is
   guaranteed to sweep — no rendered load ever remains at the def
   *and* inline), capped at `MAX_LOAD_SPLICE_SITES` (2, the
   one-cmp-two-jccs shape; over-cap counted); (b) every site passes
   its window — the same-block between-scan measured from the
   *earliest inlined load's* index, or the effect-clear region rule
   cross-block; (c) the *joint* tentative fold — co-substituting
   sibling names read by the same site whose defs are also
   cone-eligible candidates, so the pair is judged as the pair —
   leaves the function strictly smaller by whole-statement accounting:
   Σ folded site sizes < Σ standing site sizes + Σ statement sizes of
   the defs (and transitively dead cone temps) the clearing kills.
   Joint clearing commits both defs' sites together or neither;
   deterministic (BTree order, pairs in name order).
3. `FwdStats` counts the joint splices (`load_pair_spliced`) and the
   region/cap refusals; `irssa::check`/`ir::check` stay the arbiter —
   never a cross-block temp read, spliced copies are temp-free by
   construction.

## The equality relaxation (`irflow`)

The pair families' shared-operand guards — `signed_order_pair`,
`order_compose_ok`, `masked_order_pair`, `not_of_flag_select`, and the
unsigned twins if guarded the same way — currently refuse any
load-bearing operand. Under the one-expression theorem the structural
equality they already demand *is* value equality, so the guards relax
to allow load-bearing shared operands. Nothing else relaxes: the
annihilation family (`x & x → x`, `x * 0 → 0`, …) keeps its load-free
guards — this slice touches only the rules whose refusal the pair
doctrine itself documents.

## Module-by-module

- `src/irssaopt.rs`: the cone-with-loads, the joint earn, the region
  scan, module docs (the load-bearing tier's paragraph gains its
  all-or-nothing joint exception, the conforming-code note recorded),
  `FwdStats` counters.
- `src/irflow.rs`: the four (± unsigned twins) guard relaxations, each
  doc comment updated to cite the one-expression theorem; the module
  "Soundness" section gains the paragraph.
- Goldens update only where output honestly improves; each listed.
- No changes to `src/irstruct.rs`, `src/jumptable.rs`,
  `src/aarch64*.rs`, `src/pseudo.rs` logic — sibling slices own them.
- `ROADMAP.md`: the landed bullet with measured numbers.

## Test matrix (~14)

1. E2e: the two-jcc memory-operand `cmp` (hand-lifted, both x86 shape
   and the A64 subs shape if cheap): construct → optimize → forward →
   eliminate_dead renders both branches relational, flag defs and the
   temp swept.
2. Window refusals, each fixture-forced: an intervening `Store`; an
   intervening `Intrinsic` (call); a `Call` terminator inside the
   region; a cyclic region; a region past the block cap.
3. All-or-nothing: a third use that cannot clear (a φ argument) refuses
   the whole def — output byte-identical to today.
4. Function-shrink refusal: a pair whose fold does not shrink the
   function stays put.
5. Joint determinism: the same function twice, identical plans.
6. `irflow` unit: each relaxed pattern fires with structurally equal
   load-bearing operands and still refuses unequal ones; the
   annihilation family still refuses loads.
7. Division in the cone refuses.
8. Existing suite green; goldens updated only where honestly improved.

## Exit criteria (demonstrate, don't assert)

`--decompile` on ls/bash x86-64 and libbrotlidec arm64, double runs
byte-compared, zero check failures: the milestone comparator renders
both jccs as relations with the loads inline (old vs new in the
bullet); the unswept OF-tree assignment counts fall from ls 19 /
bash 196 / brotli 12; total pseudocode bytes per binary reported
before/after (the dying definitions must outweigh the inline load
spellings). `cargo test evalfx` green — gotos/CFGED can't move
(expression-level slice); if a fixture metric legitimately moves, the
table updates in the same commit. Semantic spot checks are the safety
net: the SSA interpreter must not diverge at any stage.

## Non-goals

- Value-numbering equality across SSA names (sibling slice).
- Relaxing the annihilation identities' load guards.
- Hoisting or sinking loads anywhere the region rule can't prove;
  volatile-correct load-count preservation (recorded assumption).
- Division/trap movement of any kind.
