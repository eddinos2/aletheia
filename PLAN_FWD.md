# Plan: expression forwarding on SSA (`irssaopt`, slice 5)

## Input / output

Input: an `irssa::SsaFunction` (normally already through `optimize`).
Output: a **new** `SsaFunction` in which single-use and small pure
definitions are substituted into their uses as whole expressions —
`ZF#3 := (t2 == 0); goto if ~(ZF#3)` becomes `goto if ~((t2 == 0))`,
then relational folding turns `((a - b) == 0)` into `(a == b)` — plus
an `FwdStats`. Extends `src/irssaopt.rs` (DESIGN.md slice 5,
van Emmerik 2007); `redump --ssa-opt` becomes optimize → `forward` →
`eliminate_dead`.

## The key structural decision: forward leaves the def standing

`forward` substitutes RHS trees into use sites but never deletes or
reorders statements — the forwarded def simply loses its uses and the
*existing* DCE sweeps it in the same `--ssa-opt` pipeline. Payoff:
slice 3's `check_preserved` applies to `forward` verbatim (equal
statement counts, only use expressions change), the pass stays simple,
and each pass keeps one job.

## Forwarding rules (DESIGN's three tiers, plus the guards)

A def `n := RHS` (a `Stmt::Assign` whose dst is a name) is a candidate.
Its RHS is substituted at a use site `Reg(n, w)` only when
`w == names[n].width` (an exact-width read; a narrower read keeps the
name — splicing a tree under a truncating read would need a wrapper
node and buys nothing) and the site is not an intrinsic read (slice 3's
barrier, kept: `callfx` reads model observed *registers*) and not a φ
argument (name ids, not expressions). Then by RHS class:

- **(a) Trivial** (a constant or a bare name): always — this is what
  `optimize` already does; `forward` need not re-implement it, the
  pipeline order handles it. Document, don't duplicate.
- **(b) Compound, pure, load-free, division-free:** substitute into
  *all* uses when the RHS is small (`FWD_SMALL_NODES = 8` expression
  nodes — the readability constant; DREAM++'s duplication finding
  caps how much tree may be copied to several sites), otherwise only
  when the def has exactly one use.
- **(c) Load-bearing RHS:** only into uses in the def's **own block**,
  **after** the def, with **no intervening `Store`, `Intrinsic`, or
  `Branch`** between def and use — `irflow::propagate`'s barrier set
  lifted onto SSA names (an intervening call is a `Branch` + intrinsic,
  so both gates catch it). Single-use only (a load is an effect
  observation; duplicating its tree textually to N sites reads as N
  loads even when sound — keep it named unless it moves once).
- **(d) Division-bearing RHS** (any div/mod node): same-block,
  no intervening `Branch` — a potential trap must not move past a
  guard (DESIGN, verbatim). Combines with (c) when both apply.

Post-substitution the statement is re-folded (`irflow::fold_stmt`), and
a substitution that would push the statement past `ir::MAX_EXPR_NODES`
is skipped and counted (`stats.size_skipped`) — never truncated.

Rounds: substitution can cascade (`t1 := a+b; t2 := t1*2; use t2`), so
`forward` iterates build-index → one substitution sweep until a round
changes nothing, `MAX_ROUNDS = 8` cap (same constant discipline as
`optimize`); on cap the *last completed round's* output is returned
(every intermediate state is sound — unlike `optimize`'s optimistic
lattice, forwarding has no unsound transient) with `capped = true`.

## Relational folding (the exit criterion's second half)

Forwarding alone yields `goto if ~((rdx - 0x7) == 0)`. New fold
identities in `irflow::fold_expr` (shared doctrine, so `--simplify`
benefits too), each sound under two's-complement wrapping:

- `(a - b) == 0` → `a == b`; `(a - b) != 0` → `a != b` (subtraction is
  injective in b for fixed a under wrapping — sound at every width);
- `(a ^ b) == 0` → `a == b` (same argument);
- `~(a == b)` → `a != b`, `~(a != b)` → `a == b` (W1 boolean negation
  of a comparison — only where the operand is literally a comparison
  node);
- double negation `~~x` → `x` at W1 if not already present.

**Inventory step for the builder:** read `ir::BinOp`'s actual
comparison/negation operators and `x86_lift`'s flag expressions first;
implement exactly the identities expressible with existing operators.
If signed-order jcc patterns (`SF ^ OF`, the `jle` shape) are not
representable as a single existing comparison op, do NOT invent IR
operators in this slice — record the missing patterns in the module
docs as pseudo-slice input. The committed exit criterion is the
equality family (`je`/`jne` plumbing → `==`/`!=` conditions), which the
lifter emits as `(x == 0)` shapes today.

## Soundness summary

- Substitution uses the SSA guarantee: the def dominates every use, and
  a pure load-free division-free RHS evaluates identically at any
  dominated site (its operands are SSA names — immutable values).
- Loads and divisions move only within the barrier windows above.
- No statement added/removed/reordered; no CFG mutation; intrinsic
  reads and φ arguments untouched; `partial` recomputed (forwarding an
  exact-width tree cannot create a wider-than-def read).
- Every output passes `irssa::check` and slice 3's `check_preserved`.
- Malformed input returned unchanged (zeroed stats), the established
  posture.

```rust
pub struct FwdStats {
    pub rounds: usize,
    pub forwards: usize,     // use sites rewritten
    pub size_skipped: usize, // substitutions refused by MAX_EXPR_NODES
    pub capped: bool,
}
pub fn forward(func: &SsaFunction) -> (SsaFunction, FwdStats);
```

## Module-by-module

- `src/irssaopt.rs`: `forward`, `FwdStats`, the barrier scan, module-doc
  section (tiers, guards, the missing-signed-patterns note), tests.
- `src/irflow.rs`: the relational identities in `fold_expr` +unit tests
  (each identity, both polarities, plus a width-mismatch non-fold and a
  no-erasure check: identities never drop a load-bearing side —
  operands of a comparison are pure by `ir::check`? verify, don't
  assume: if a comparison operand can hold a Load, the identity keeps
  both operands, which it does by construction — state it in the test).
- `src/bin/redump.rs`: `--ssa-opt` pipeline gains `forward` between
  optimize and DCE; usage text mentions forwarded expressions.
- `ROADMAP.md`: Current-thread → slice 6 (`irstruct`, Phoenix
  structuring) — wave 2 complete.

## Test matrix (~22)

1. canonical x86 golden: lifted `cmp`+`je` block through
   optimize→forward→DCE renders `goto if (<reg> == <const>)` (or its
   `~` polarity for `jne`) — flag plumbing gone end-to-end; both
   polarities.
2. multi-use compound def above `FWD_SMALL_NODES` stays named; at or
   under the constant forwards to all uses; exactly-one-use large def
   forwards; boundary test at the constant.
3. load forwarding: same-block after-def no-barrier forwards;
   blocked by intervening store / intrinsic / branch / cross-block use;
   multi-use load def never forwards.
4. division: forwards within its block before a branch; blocked past a
   branch; blocked cross-block.
5. narrower-width use keeps the name; exact-width sibling forwards.
6. intrinsic reads and φ arguments byte-identical through `forward`.
7. chain cascade closes in one `forward` call; round 2 is a no-op
   (idempotence); determinism (twice → byte-equal).
8. `MAX_EXPR_NODES`: near-cap statement refuses the substitution,
   `size_skipped` counted, output checks.
9. relational identities unit tests in `irflow` (each identity, wrong
   width no-fold, no operand erased).
10. `check_preserved` asserted on every `forward` output (equal
    counts); `irssa::check` everywhere; malformed input unchanged;
    empty function; no panics.
11. seeded sweep: construct → optimize → forward → eliminate_dead →
    check + check_preserved(forward stage) + check_swept(dce stage) Ok,
    no caps, no panics.
12. redump e2e: calling fixture + diamond fixture goldens show a
    relational branch condition under `--ssa-opt` and unchanged `--ssa`;
    byte-determinism; `/bin/ls` x86-64 slice: zero failures, at least
    one function renders a `== / !=` branch condition where `--ssa`
    shows flag reads (report the concrete example).

## Non-goals (this slice, per DESIGN)

- Signed/unsigned order patterns needing new IR operators (documented
  as pseudo-slice input; the equality family is the committed scope).
- aarch64 end-to-end (`irlift` dispatch still x86-64-only; the pass is
  ISA-blind — the `SUBS`+`B.cond` case rides with the dispatch slice).
- GVN, commutative canonicalization, store-to-load forwarding
  (memory slices), any CFG or statement-order change.
- Condition *synthesis* across blocks — that is structuring's job.
