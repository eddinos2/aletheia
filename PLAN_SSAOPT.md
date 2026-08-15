# Plan: sparse constant/copy propagation on SSA + φ-simplification (`src/irssaopt.rs`)

## Input / output

Input: an `irssa::SsaFunction` (normally lift → `callfx::apply` →
`irssa::construct`, as `redump --ssa` builds it). Output: a **new**
`SsaFunction` — same type, never mutated in place — in which every use of
a name whose value is proven constant reads the constant, every use of a
plain copy reads the copied-from name, and φ-nodes that merge one value
(`φ(x,…,x)`, `φ(x, self)`, including through copy chains) are removed;
plus a `Stats` record. New module `src/irssaopt.rs`, wired into a new
`redump --ssa-opt[=N]`.

## Slice boundary (fixed by `research/decompiler/DESIGN.md`)

This is DESIGN.md **slice 3** ("`irssaopt` A") exactly. DESIGN splits the
SSA-cleaning wave into three commits, and this plan covers only the
first:

- **This slice:** sparse worklist constant/copy propagation over def-use
  chains (Cytron et al., TOPLAS 1991 — on SSA the single-definition
  property *is* the analysis) with folding via `irflow::fold_expr`, and
  φ-collapse of `φ(x,…,x)` / `φ(x, self)`.
- **Slice 4 (not here):** conservative DCE. This pass deliberately
  leaves now-dead definitions in place (`rax#1 := 0x5.q` stays after its
  uses are rewritten) — statement count and order are *invariants* here,
  exactly like `irflow::propagate`'s contract.
- **Slice 5 (not here):** expression forwarding (compound RHS
  substitution, the cmp+jcc flag collapse, load forwarding).
- **Deferred by DESIGN explicitly:** full SCCP — its edge
  deletion mutates the CFG that structuring will consume. No
  executability tracking, no edge or block removal, ever, in this
  module.

## Representation (the key decision)

The pass is **`SsaFunction` → `SsaFunction` over a def-use index built
here**, because `irssa` stores no use lists — its names are indexed only
by occurrence. One deterministic pass over the blocks builds:

```rust
/// Where a name occurrence in use position lives.
#[derive(PartialOrd, Ord, ...)]
enum UseSite {
    /// Inside statement `index` of `block` (any number of occurrences).
    Stmt { block: u64, index: usize },
    /// Argument `arg` of φ number `phi` in `block` (the value observed
    /// at the predecessor's exit).
    PhiArg { block: u64, phi: usize, arg: usize },
}
/// name id -> every use site, ascending.       (BTreeMap<u16, BTreeSet<UseSite>>)
/// name id -> its definition:                  (Vec<Def>, indexed by id)
enum Def {
    Entry,                                    // version 0
    Assign { block: u64, index: usize },      // single dst
    IntrinsicWrite { block: u64, index: usize },
    Phi { block: u64, phi: usize },
}
```

The index is scaffolding *inside* `irssaopt`, not a new public shape on
`SsaFunction`: `check`'s from-scratch doctrine means downstream passes
that need uses rebuild them (slice 4 will, cheaply), and the SSA type
stays exactly what `irssa::check`/`render` already validate. The walkers
that enumerate occurrences already exist as `irssa`'s private
`for_each_use` / `for_each_def` / `expr_regs`; they become `pub(crate)`
rather than being duplicated (same crate, no dependency cycle:
`irssaopt` consumes `irssa` and `irflow`, nothing consumes `irssaopt`).

## The lattice and transfer

Per name, a three-level value, computed optimistically to a fixpoint by
a sparse worklist over the def-use index:

```
Top  (no evidence yet)
  >  Known: Const(value, width)   — the def's width; value masked to it
        or Copy(root)             — same runtime value as name `root`
  >  Bottom (varies / unknowable)
```

Transfer per definition kind:

- **`Entry` (version 0)** and **`IntrinsicWrite`**: `Bottom`, always.
  This is where call clobbers kill facts *for free*: a `callfx`
  intrinsic write is a fresh name with `Bottom`, so nothing propagates
  through a call for a caller-saved cell, while a callee-saved cell
  keeps its (unchanged) name and its fact — no special case anywhere.
- **`Assign`**: clone the RHS; substitute each register read at width
  `wu` by its current value when eligible (`Const(c,_)` with
  `wu <= names[n].width` → `Const(c & wu.mask(), wu)`; `Copy(root)`
  with `wu <= names[root].width` → `Reg(root, wu)`; `Top`/`Bottom`/
  wider-than-def reads stay); fold with `irflow::fold_expr`. Result
  `Const` → `Const(v, dst.width)`; result a bare `Reg(r)` whose
  occurrence width is not wider than `names[r].width` → `Copy(root(r))`;
  anything else → `Top` if some substituted read was still `Top`
  (revisit later), else `Bottom`.
- **`Phi`**: meet over the arguments, each resolved first — an argument
  equal to the φ's own dst is skipped (`φ(x, self)`); an argument whose
  value is `Top` is skipped (optimism); `Copy` chains follow to their
  root (`resolve(x)`: while `value(x) == Copy(y)` take `y`; a `Const`
  resolves to that constant). Then: all resolved to the same name `r` →
  `Copy(r)`; all resolved to constants equal in value *and width* →
  that `Const`; nothing resolved (all skipped) → `Top`; otherwise
  `Bottom`. (Masked-to-min-width constant equality is a possible later
  precision win; strict equality is enough for lifted code, where GPR
  defs are all W64.)

`resolve` terminates because a copy's source definition strictly
dominates the copy (SSA dominance), so copy chains are acyclic; φ-self
edges are skipped before resolving.

Worklist: seed every name id (a `BTreeSet<u16>`, `pop_first` — fully
deterministic); when a name's value lowers, re-enqueue the names whose
definitions use it (use sites → enclosing assign's dst or φ's dst; uses
inside stores/branches/intrinsic reads feed no definition). Each name
moves at most Top → Known → Bottom, with one representation refinement
inside Known (`Copy(a)` → `Const` when `a` itself lowers), so total
lowering steps are ≤ 3·names and the fixpoint is reached in
O(names + use-edges) worklist pops. A defensive cap
(`8 * (names + total_uses) + 8` pops — provably unreachable) aborts the
whole pass: **on cap the input is returned unchanged** with
`stats.capped = true`; capped never means partially-applied optimistic
facts, which would be unsound.

Because there is no executability tracking, the meet runs over *all*
φ arguments — the conservative, always-sound reading of the sparse
engine; the "conditional" half of Wegman–Zadeck is exactly what DESIGN
defers. Note: the constant-through-φ meet is the one place this plan
spells out more than DESIGN's two named rules ("RHS folds to constant
or bare name"; "φ(x,…,x)/φ(x,self) collapse") — it is the same cited
sparse engine, it adds no new invariant surface (a `Const`-valued φ is
*kept*, only its uses are rewritten), and it is required for the
"constant through phi" behavior the test matrix pins.

## Rewrite rules (after the fixpoint; one deterministic pass)

For every statement of every block, in address then index order:

- A register read `Reg { num: n, width: wu }` in use position is
  replaced by:
  - `Const(c & wu.mask(), wu)` when `value(n)` resolves to `Const(c, w)`
    and `wu <= names[n].width` (a sub-register read of a constant is its
    low bits; a read *wider* than the def — a `partial` position — is
    never rewritten: those bits are honestly unknown);
  - `Reg { space: names[root].space, num: root, width: wu }` when it
    resolves to `Copy(root)` and `wu <= names[root].width` (guaranteed
    ≥ the copy's width by the record rule, so no new `partial` entries
    can ever be created — the marker set only shrinks, and a shrink is
    honest: the substituted def guarantees strictly more bits);
  - **Temp-space guard:** a `Copy` whose root lives in `Space::Temp` is
    substituted only at sites in the root's *defining block*
    (for a φ-argument site, the predecessor block). `ir::check` treats
    each block as straight-line and rejects a temporary read with no
    earlier write in the same list (`TempReadBeforeWrite`); a cross-
    block temp read would make the output fail `irssa::check`'s per-
    block IR validation. Constants have no space and substitute
    anywhere.
- **Intrinsic reads are never rewritten** (they still pin their names
  via the use index; their occurrences just stay verbatim). This is a
  deliberate divergence from `irflow::propagate`, which does substitute
  into intrinsic reads: a `callfx` read is the model of "the callee
  observes this argument *register*", and keeping the register identity
  is what lets slice-4 DCE stay trivially sound (argument setups remain
  pinned by an intrinsic use) and lets the signature slices (12–13)
  read argument registers off the call. Nothing in this slice's exit
  criteria needs intrinsic-read rewriting.
- Branch condition and target expressions *are* rewritten (the point of
  the slice: a cross-block constant reaches its branch condition); the
  `Branch` statement itself — kind, presence of cond, position — is
  untouched. Store address/value are rewritten; the `Store` stays.
- After substitution, every rewritten statement is re-folded with
  `irflow::fold_stmt` (promoted to `pub`; folds each expression,
  preserves structure and effects). Substitution replaces a `Reg` node
  by a `Const` or `Reg` node and folding only shrinks, so expression
  size never grows — `ir::MAX_EXPR_NODES` holds by construction.
- φ arguments: an argument id `x` is replaced by `root(x)` when
  `value(x)` resolves to `Copy(root)` (subject to the temp-space guard;
  the width invariant `dst.width <= arg.width` survives because
  `names[root].width >= names[x].width` always). Constants cannot be
  substituted into φ arguments — they are name ids, not expressions; a
  `Const`-valued φ instead has its *dst's* uses rewritten.

Statement count, order, and every definition occurrence (assign dsts,
intrinsic writes) are byte-identical to the input. Only use expressions
and φ argument ids change, φ lists may shrink, and the name table may
shrink (below).

## φ-simplification and name-table compaction

A φ is **removed** iff its fixpoint value is `Copy(r)` — which is
precisely the `φ(x,…,x)` / `φ(x, self)` set, generalized through copy
chains and cascaded through φ-of-φ (the meet already resolved both) —
*and* every use site of its dst is substitutable (width rule
`use_width <= names[r].width`, temp-space guard). Single-predecessor
and single-non-self-argument φs are the degenerate all-same case and
need no separate rule. A φ that fails the substitutability guard (only
constructible by hand-built IR; lifted code defs are full-width) is
kept, with its eligible uses still rewritten. Soundness of the
dominance claim: `r`'s definition dominates every predecessor's exit
(it dominates each argument's def, which dominates its edge), hence
every path to the φ's block passes `r`'s def, hence `r` dominates the
φ position and, transitively, every use the φ dominated.

Removing a φ orphans its dst name, and `irssa::check` faults any
non-version-0 name with no definition (`SsaFault::Undefined`). So the
pass ends with a **name-table compaction**: collect the removed φ dsts,
build the order-preserving old-id → new-id map over the survivors, and
remap every `Reg::num` (uses and defs), every φ dst and argument, and
`live_in`. Version-0 names are never removed (they are `Bottom`, never
a φ dst), so `live_in` stays the exact version-0 set; `(cell, version)`
uniqueness survives (removal cannot collide); version numbering keeps
gaps (`rax#3` may follow `rax#1`), which `check` permits — it requires
uniqueness, not density. `partial` is recomputed from scratch at the
end (same recomputation `construct` does): positions can only leave the
set (a folded identity may delete a wide read's subtree; a copy
substitution may replace a wide read's def with a wider one), never
join it.

## Soundness rules carried over (stated per consumer)

- **Loads are never folded away or deleted.** Substitution sources are
  `Const` or bare `Reg` only — load-free by construction, so no fact
  depends on memory and stores need no invalidation (the `irflow`
  argument, now function-wide). Statements are never deleted, and
  `fold_expr`'s identities already refuse to erase a load-bearing side.
  A def whose RHS contains a load is `Bottom` — its *dst* propagates
  nothing; its own reads are still rewritten (the address may fold).
- **Division by zero is never folded** — `irflow::fold_expr` returns
  divide-by-zero shapes unfolded, verbatim doctrine. Substituting a
  proven-zero divisor is sound (the runtime value was zero; the trap
  stays written as a trap).
- **Call clobbers kill facts by SSA structure.** A `callfx` intrinsic
  write is an opaque def (`Bottom`); post-call uses read that name, so
  a pre-call constant in a caller-saved register never crosses the
  call. Callee-saved registers keep their name and their fact — the
  test matrix pins both directions.
- **Sub-register/width rules.** A use is rewritten only at
  `use_width <= def_width` (exact low-bits dependence per the module's
  aliasing doctrine); `partial` positions are left alone; copy facts
  are only recorded when the source guarantees at least the copy's
  width; constants are masked to the reading width. No rewrite can add
  a `partial` entry.
- **No CFG mutation.** Block set, `start`/`end`, `successors`,
  `truncated`, `entry`, `name`, `skipped` are copied verbatim.
  Constant branch conditions are left as constant *expressions* on
  intact branches — edge decisions belong to structuring (DESIGN's
  SCCP note).
- **Truncated blocks** get no special handling: their SSA already
  reflects the honest prefix, and the pass neither extends nor trusts
  anything past it.

## The pass

```rust
/// Deterministic counters; `capped` means "input returned unchanged".
pub struct Stats {
    pub rounds: usize,       // outer passes run (>= 1)
    pub rewrites: usize,     // use occurrences + phi args replaced
    pub phis_removed: usize,
    pub names_removed: usize,
    pub capped: bool,
}
pub fn optimize(func: &SsaFunction) -> (SsaFunction, Stats);
```

`optimize` = an outer loop of (build index → sparse fixpoint → rewrite
→ φ-removal → compaction), iterated until a round changes nothing, at
most `MAX_ROUNDS = 8` rounds — mirroring `irflow::MAX_ROUNDS`, which is
private, so the constant is restated here with a doc-comment saying so.
One round reaches closure for these rules (the analysis already
evaluates through substitutions); the loop is defense plus the DESIGN
requirement, and the idempotence test pins round 2 as a no-op. Pure,
total, no panics, no I/O; every map/loop is a BTree or index order, so
equal inputs give byte-equal outputs. Malformed inputs (hand-built SSA
that fails `irssa::check`) are not laundered: `optimize` first runs
`irssa::check` and returns the input unchanged (with `capped = false`,
`rounds = 0`) on any fault — the same refuse-don't-guess posture as
`construct`, without inventing a new error type for a pass that can
always decline.

A companion differential check, used by every test (and the seeded
sweep), verifies the preservation contract without trusting the pass:

```rust
pub fn check_preserved(input: &SsaFunction, output: &SsaFunction) -> Result<(), Preserved>;
```

comparing modulo the compaction remap by canonicalizing every id to its
`(space, cell, version, width)` tuple: block keys/`end`/`successors`/
`truncated`/`entry`/`skipped` equal; per block equal statement count and
per-index equal discriminant; branch kinds and cond-presence equal;
intrinsic names, write tuples, and read expressions (tuple-canonical)
equal; output φs a subset of input φs by dst tuple; output names a
subset of input names, with the version-0 subset identical. Output must
additionally pass `irssa::check` — asserted by every test.

## Module-by-module changes

- **`src/irssaopt.rs` (new):** module docs (scope, the DESIGN citation
  line — Cytron et al. 1991 sparse propagation; SCCP deferral note —
  lattice, soundness rules, the intrinsic-read divergence from
  `irflow`, determinism/totality contract); `UseSite`/`Def` index;
  lattice + sparse fixpoint; rewrite; φ-removal + compaction;
  `Stats`, `optimize`, `check_preserved`; tests.
- **`src/irssa.rs`:** `for_each_use`, `for_each_def`, `expr_regs`
  become `pub(crate)` (no behavior change); one module-doc sentence:
  cross-block propagation lives in `irssaopt`, this module stays the
  faithful construction.
- **`src/irflow.rs`:** `fold_stmt` becomes `pub` with a doc-comment (no
  behavior change); one module-doc sentence pointing SSA-level,
  cross-block propagation at `irssaopt` (this module remains the
  per-block, pre-SSA library).
- **`src/lib.rs`:** `pub mod irssaopt;` (after `irssa`).
- **`src/bin/redump.rs`:** `--ssa-opt[=N]` flag + usage text; a
  `print_ssa_opt` that reuses `print_ssa`'s pipeline (lift →
  `callfx::abi_for`/`apply` → `irssa::construct`) then `optimize` and
  `irssa::render` — factor the shared pipeline into a small helper
  rather than duplicating it. When `stats.capped`, print an honest
  `; note: optimization capped, output unoptimized` line above the
  function. Same x86-64 gate and `Unrepresentable` note as `--ssa`.
- **`ROADMAP.md`:** move the Current-thread pointer to slice 4
  (conservative DCE) when committing.

## Reuse vs add

- Reuse: `SsaFunction` and all `irssa` machinery (`check`, `render`,
  the occurrence walkers via `pub(crate)`), `irflow::fold_expr` +
  `fold_stmt` (via `pub`) — the fold logic is *shared*, not duplicated,
  which is exactly why divide-by-zero and load doctrine carry over
  verbatim; `callfx` untouched (its intrinsic already behaves as the
  opaque def the lattice needs); `redump`'s existing lift+callfx+ssa
  pipeline.
- Add: `src/irssaopt.rs` (index, lattice, rewrite, φ-removal,
  compaction, `Stats`, `optimize`, `check_preserved`), the
  `--ssa-opt[=N]` flag, the two `pub`/`pub(crate)` promotions, module-
  doc sentences in `irssa`/`irflow`, `pub mod irssaopt;`.
- Not reused: `irflow::propagate`/`substitute` — they are keyed by
  exact `Reg` within one straight-line block and substitute into
  intrinsic reads; the SSA pass needs name-id keys, width adaptation,
  and the intrinsic-read barrier. The fold layer is the shared part;
  the propagation drivers are necessarily different algorithms.

## CLI: `--ssa-opt`, not folded into `--ssa`

`--lift` is the faithful view and `--simplify` the cleaned one; the SSA
tier mirrors that: `--ssa` stays the faithful construction (plus the
call-effects modeling that is its point), `--ssa-opt` is the cleaned
view. Keeping the raw view matters for the proof trail (the honest
before/after is itself the exit criterion) and keeps `--ssa` goldens
stable. **Recorded tension:** DESIGN's slice-5 exit criterion says
"`redump --ssa` … shows relational branch conditions", implying the
optimized view eventually *becomes* `--ssa`; that renaming decision is
deferred to slice 5, when the output is worth making the default —
nothing in this slice forecloses it (`--ssa-opt` is one match arm).

## Exit criteria (DESIGN slice 3: golden fixture + re-check + no regressions)

- On a checked-in synthetic x86-64 image whose entry sets a register to
  a constant, branches through a diamond that does not redefine it, and
  uses it in the merge block's comparison: `redump --ssa-opt` shows the
  constant (`0x5.q`) in the merge-block expression and a constant-
  expression branch condition where `redump --ssa` shows `rax#1` — the
  cross-block constant reaching its use, with the branch and CFG intact.
- Every optimized function re-passes `irssa::check` and
  `check_preserved`; on a real binary (the `/bin/ls` x86-64 slice used
  to validate `callfx`): zero check failures across all functions,
  zero caps, byte-identical output across runs.
- Full existing suite (885+ tests) still green; clippy clean.

## Test matrix (~30)

Engine (unit, building SSA via `irssa::construct` from hand-built
`irlift` functions unless noted; every test asserts `irssa::check` +
`check_preserved` on the output):

1. straight-line: constant def, later same-block use rewritten; stmt
   count/order unchanged; the dead def remains (slice-4 boundary pinned).
2. cross-block: constant in entry, use two blocks later rewritten; a
   conditional branch's cond becomes a constant expression; `Branch`
   statement, successors, and block set untouched.
3. copy chain across a diamond: `b := a` in entry, merge-block uses of
   `b` read `a`.
4. multi-hop copy chain (`b:=a; c:=b; d:=c`): uses of `d` read `a`
   (root resolution).
5. φ-collapse golden: both arms copy the same source (`rcx := rax`),
   merge φ(rcx#1, rcx#2) removed, merge use reads `rax#1`; exact
   `render` golden.
6. loop-invariant φ(init, self): body does not redefine the cell → φ
   removed, body uses read the init name.
7. real induction φ (i = i + 1 around a back edge): value `Bottom`,
   nothing rewritten, φ kept — output equals input.
8. constant through φ: two arms assign the same constant via distinct
   defs → φ *kept* (args differ), dst uses rewritten to the constant.
9. φ over differing constants (or same value, different widths) →
   `Bottom`, untouched.
10. φ-of-φ cascade: inner φ collapses, outer φ (now all-same through
    the copy) collapses too, in one `optimize` call.
11. callfx clobber kills the fact: `rax := 5`; call; post-call use
    reads the intrinsic's fresh version, not `5` (build via
    `callfx::apply` + `construct`).
12. callee-saved survives the call: `rbx := 5` before, use after → `5`.
13. intrinsic reads verbatim: `rdi := 5`; call — the `callfx` read
    still says `rdi#1` while an ordinary use of the same name elsewhere
    is rewritten to `5`.
14. div-by-zero: proven-zero divisor substituted, division not folded;
    also a literal `x /u 0` stays unfolded end-to-end.
15. loads: a def with a load RHS yields no fact and is never deleted;
    a store's address is rewritten, the store kept; `load & 0` is not
    folded to `0` (fold guard, cross-checked at this level).
16. width: hand-built W32 def, W64 use (a `partial` position) is not
    rewritten and stays listed; a W16 use of the same def is rewritten
    to the truncated constant; `partial` recomputed exactly.
17. copy record guard: hand-built `b.W64 := a` where `a`'s name is W32
    → no copy fact, uses untouched.
18. temp-space guard: copy with a `Space::Temp` root is substituted in
    the root's block, not across blocks; output blocks still pass
    `ir::check` (the `TempReadBeforeWrite` trap this guard exists for).
19. φ kept when a dst use is wider than the root (hand-built
    `SsaFunction`): eligible uses rewritten, φ retained, check passes.
20. compaction: removed-φ dsts leave the name table; ids remapped
    everywhere; version gaps tolerated; `live_in` remapped and still
    exactly the version-0 set.
21. hand-broken input (fails `irssa::check`) → returned unchanged,
    `rounds = 0`; empty function → unchanged; no panics.
22. `check_preserved` negatives: a dropped store, a mutated branch
    kind, an added statement, a removed version-0 name — each rejected.
23. determinism: `optimize` twice on the same input → byte-equal
    renders; idempotence: `optimize(optimize(f).0)` changes nothing and
    round 2 of the outer loop is a no-op.
24. stats: `rewrites`/`phis_removed`/`names_removed` exact on a known
    fixture; `capped == false` everywhere in the suite.
25. seeded sweep (xorshift64*, repo style): random small CFGs →
    `construct` → `optimize` → `irssa::check` Ok, `check_preserved` Ok,
    never capped, no panic.

`redump`:

26. `--ssa-opt` parse: bare, `=N`, malformed `=x` rejected; does not
    imply `--ssa` or `--lift`.
27. e2e golden: the synthetic diamond image — `--ssa-opt` shows the
    constant at the merge use, `--ssa` on the same image still shows
    `rax#1` (raw view intact); run twice, byte-equal.
28. e2e φ-collapse: synthetic image whose SSA has a collapsible φ —
    gone under `--ssa-opt`, present under `--ssa`.
29. e2e with calls: `--ssa-opt` on the calling-function fixture keeps
    `callfx(` reads verbatim and does not propagate a caller-saved
    constant past the call.
30. non-x86-64 image → the same one-line note as `--ssa`; fat Mach-O
    note preserved.

## Non-goals (this slice, per DESIGN)

- Dead-code elimination — slice 4 removes the dead defs and dead φs
  this pass exposes (a `Const`-valued φ with no remaining uses is
  slice-4 sweep material, deliberately left standing here).
- Expression forwarding (compound RHS substitution, flag→relational
  branch collapse, any load forwarding) — slice 5.
- SCCP executability: no edge deletion, no branch-decision annotation,
  no unreachable-arm pruning of φ meets.
- Memory facts of any kind (store-to-load, MEM versioning — slice 11);
  commutative canonicalization / GVN (recorded DESIGN non-goal);
  masked-width constant meets at φs (noted precision follow-up).
- Rewriting intrinsic reads (deliberate barrier, argued above).
- aarch64 end-to-end (`irlift` still dispatches x86-64 only; the pass
  itself is ISA-blind and needs nothing when that slice lands).
- Any change to `irssa`'s construction/check/render algorithms or
  `irflow`'s passes beyond the two visibility promotions.
