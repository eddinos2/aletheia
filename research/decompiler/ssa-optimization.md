# SSA-based simplification for decompilation

Topic file for the decompiler design phase (see `BRIEF.md`). Published
sources only; every recommendation names the Aletheia module it lands in.

## The problem

`irssa::construct` now gives Aletheia pruned SSA over the lifted IR CFG:
one definition per name, φ-nodes at joins, def-use resolvable in O(1).
But the SSA statements are still the raw lift's statements — flag
recomputations nobody reads across blocks, values threaded through
copies, constants that a branch three blocks later tests — and
`irflow`'s three passes (fold, propagate, eliminate) stop at every block
boundary because pre-SSA they cannot know which definition a cross-block
read sees. The task is the classic middle-end trio recast for a
decompiler: cross-block constant/copy/expression propagation on SSA,
dead-code elimination on SSA, and eventually translation *out* of SSA
into named variables for pseudocode emission — with the decompiler-
specific twists that memory is not versioned, loads may fault or alias,
call clobbers are not yet modeled, and the output must stay honest and
readable rather than merely fast.

## Strongest published approaches

### 1. Cytron et al. 1991 — the SSA substrate and its naive destruction

R. Cytron, J. Ferrante, B. Rosen, M. Wegman, F. K. Zadeck,
"Efficiently Computing Static Single Assignment Form and the Control
Dependence Graph", *ACM TOPLAS* 13(4), 1991. The baseline for
everything here: dominance-frontier φ-placement (Aletheia's `irssa`
already implements the pruned variant, per Choi–Cytron–Ferrante 1991),
def-use chains as the sparse propagation medium, and the observation
that on SSA, copy propagation degenerates to substitution (a use of
`x2` where `x2 = x1` just becomes a use of `x1` — no kill analysis
needed, the single-definition property *is* the analysis). Their
out-of-SSA translation — replace each φ by copies in the predecessors —
is the naive baseline, later shown broken in edge cases (see #4).

### 2. Wegman & Zadeck 1991 — sparse conditional constant propagation

M. N. Wegman, F. K. Zadeck, "Constant Propagation with Conditional
Branches", *ACM TOPLAS* 13(2), 1991 (SCCP). The optimal-in-class
constant propagator: a worklist over SSA edges *and* CFG edges
simultaneously, lattice ⊤/const/⊥ per name, evaluating φ-nodes only
over arguments from executable edges. Finds every constant that simple
propagation finds *plus* those visible only once provably-untaken
branches are ignored, in time linear in the SSA graph. The full
algorithm wants to delete unreachable edges afterwards, which mutates
the CFG; the propagation core is separable from the CFG surgery and is
valuable on its own (a branch on a proven constant can simply be
*annotated* as decided rather than removed).

### 3. van Emmerik 2007 — SSA applied to decompilation specifically

M. J. van Emmerik, "Static Single Assignment for Decompilation", PhD
thesis, University of Queensland, 2007 (implemented in the open-source
Boomerang decompiler). The key source for what is *different* about a
decompiler's use of SSA:

- **Expression propagation is the workhorse.** Substituting a
  definition's whole RHS into its uses (not just constants/copies)
  collapses the lift's flag computations into the conditions that read
  them (`ZF = (a - b == 0)` propagated into `branch if ZF` yields
  `branch if a == b`) and rebuilds source-level expressions. On SSA
  this is safe *for pure, memory-free expressions* wherever the use is;
  it is limited for expressions containing loads, because memory is not
  single-assignment — a store, call, or intrinsic between def and use
  may change what the load reads.
- **Propagation is a readability trade-off, not just an optimization.**
  Substituting a multi-use definition duplicates its expression at every
  use; Boomerang's practice is to always forward trivial RHSs
  (constants, bare registers) but forward compound expressions only
  into a *single* use, keeping the assignment (a future local variable)
  otherwise.
- **DCE removes the lift's noise.** After propagation, unread flag
  definitions and dead temporaries fall out as SSA names with no uses;
  eliminating them is what shrinks a raw lift toward source shape.
- **Out-of-SSA for decompilation targets variables, not registers.**
  There is no register allocator downstream; φ-webs that don't
  interfere coalesce into one named local, and interference forces
  either a fresh variable or an explicit copy — the copy *is* pseudocode
  (`tmp = x;`), so minimizing copies is a readability goal.

### 4. Out-of-SSA done correctly — Briggs et al. 1998; Sreedhar et al. 1999; Boissinot et al. 2009

P. Briggs, K. Cooper, T. Harvey, L. T. Simpson, "Practical Improvements
to the Construction and Destruction of Static Single Assignment Form",
*Software: Practice & Experience* 28(8), 1998 — documents the **lost-copy**
and **swap** problems: Cytron's naive per-predecessor copy insertion
miscompiles when a φ's result is live past the block or when φs at one
join form a permutation, because the inserted copies execute
sequentially while φs are semantically parallel.
V. C. Sreedhar, R. Ju, D. Gillies, V. Santhanam, "Translating Out of
Static Single Assignment Form", *SAS* 1999 — frames destruction as
coalescing φ-congruence classes with interference checks, inserting
copies only to break interference (their "Method III" is the
copy-minimizing one).
B. Boissinot, A. Darte, F. Rastello, B. Dupont de Dinechin, C. Guillon,
"Revisiting Out-of-SSA Translation for Correctness, Code Quality and
Efficiency", *CGO* 2009 — the modern reference: isolate φs with parallel
copies first (making correctness trivial), then coalesce aggressively
using dominance-based interference (value-interference), sequentialize
the surviving parallel copies handling the swap case explicitly. Clean
separation of *correct* from *good*, which suits Aletheia's
proven-vs-heuristic doctrine.

Also relevant: F. Rastello & F. Bouchez Tichadou (eds.), *SSA-based
Compiler Design*, Springer 2022 — a compendium chapter-referencing all
of the above, useful as a secondary check. Open-source corroboration:
Boomerang implements van Emmerik directly; RetDec (Avast, open source)
delegates exactly this pass set to LLVM's SSA passes (SCCP, GVN, DCE)
and documents the readability cost of compiler-grade normalization;
Ghidra's published p-code/high-function sources show the same shape —
SSA-ish varnode versioning, forward substitution into expression trees,
dead varnode elimination — as the path from lift to pseudocode.

## Trade-offs

| Choice | For | Against | Verdict for Aletheia |
|---|---|---|---|
| Simple sparse const/copy prop (worklist over def-use, reuse `irflow::fold_expr`) | Small, total, no CFG mutation; output trivially re-`check`able | Misses constants guarded by decided branches | **First slice.** Matches "when in doubt, do nothing" |
| Full SCCP (Wegman–Zadeck) | Optimal constants; identifies dead edges | Wants CFG edge deletion; interacts with `cfg`/structuring; more state | Later slice; run the lattice, *annotate* decided branches, defer edge surgery to structuring |
| Expression forwarding (van Emmerik) | Rebuilds source expressions; collapses flag defs into conditions; biggest readability win | Duplication at multi-use sites; load-bearing RHSs need memory barriers Aletheia can't yet prove cross-block | Forward pure load-free RHSs cross-block; load-containing RHSs only within a block with no intervening `Store`/`Intrinsic`/call (exactly `irflow`'s doctrine, now SSA-sparse) |
| Global value numbering (Alpern–Wegman–Zadeck 1988) | Detects equal recomputations across blocks | Decompiler gain is modest before type/variable recovery; redundancy elimination can *hurt* readability | Defer; revisit after pseudocode-emission research |
| Conservative DCE on SSA (mark from effects, sweep pure defs) | Linear, removes lift noise, never touches `Store`/`Branch`/`Intrinsic` | Keeps computations feeding dead branches | **First slice**, alongside propagation |
| Aggressive DCE (control-dependence based, Cytron et al.) | Also removes dead branches | Needs RDF; deleting branches belongs with structuring | Defer |
| Out-of-SSA: naive Cytron copies | Trivial | Lost-copy and swap bugs (Briggs 1998) — real miscompiles | Never |
| Out-of-SSA: Sreedhar III / Boissinot 2009 | Correct; minimizes copies → readable locals | More machinery (interference via dominance + liveness) | Boissinot's structure (isolate → coalesce → sequentialize) when the emission slice needs it |

## Concrete recommendation for Aletheia

Land a new module, `src/irssaopt.rs`, consuming and producing
`irssa::SsaFunction` (the existing `irflow` stays as-is: it runs pre-SSA
inside `irlift` and its `fold_expr`/liveness are reused as libraries).
Three one-commit slices, in order:

1. **Sparse constant/copy propagation + φ-simplification.**
   Build def-use chains over the `SsaFunction` (a `BTreeMap<name-id,
   Vec<use-site>>`, deterministic by construction). Worklist in name-id
   order: a def whose RHS folds (via `irflow::fold_expr`) to a constant
   or a bare SSA name substitutes into all uses; φ(x, x, …) and
   φ(x, self) collapse to x. No CFG mutation; a branch whose condition
   becomes constant is left in place (annotation is a later consumer's
   job). Caps: bounded rounds like `irflow::MAX_ROUNDS`, expression
   growth bounded by `ir::MAX_EXPR_NODES`. Output must pass
   `irssa::check`; the pass is total and never panics.
2. **DCE on SSA.** Mark: every name used by a `Store`, `Branch`,
   `Intrinsic`, φ that is itself live, `live_in`-reachable use, any
   `partial` use, and function live-out; propagate liveness backward
   through def-use. Sweep: delete pure assignments to unmarked names
   *only when the RHS is load-free* — the standing `irflow` doctrine
   (a `Load` may fault; deleting it is unproven) carried over verbatim.
   Never remove a `Store`/`Branch`/`Intrinsic` or a φ a live name needs.
3. **Expression forwarding (van Emmerik).** Substitute a def's RHS into
   its uses when: RHS is trivial (constant/name) — always; RHS is
   compound, pure, and load-free — into any number of uses if small,
   else single-use only (readability rule, tunable constant); RHS
   contains a `Load` — only within the def's own block with no
   intervening `Store`/`Intrinsic`/call-branch, i.e. the exact barrier
   set `irflow::propagate` honors today, now applied on SSA names.
   Division stays unforwardable across a branch (a trap must not move
   past a condition that guarded it — same honesty as "division by zero
   is never folded").

Each slice: deterministic (BTree everywhere, worklists in id order),
total `check` after, deterministic `render` diff-tested against small
lifted fixtures (the flag-collapse case — `cmp`+`jcc` folding to a
relational branch — is the canonical exit criterion for slice 3).
Out-of-SSA (Boissinot 2009 structure, Sreedhar-style φ-web coalescing
into named locals) is **not** one of these slices; it belongs with the
pseudocode-emission phase and should be designed against that topic
file, since its cost function is readability (copies are visible code).

## Open questions

- **Call effects first?** ROADMAP queues ABI call-clobber modeling as
  the next slice. Propagation across a call without it is only sound
  for temporaries/flags the IR proves are rewritten anyway; slice 1–3
  are still correct (def-use links already stop at nothing, matching
  `irflow::liveness`), but their *yield* across calls is honest only
  once clobbers exist. Order: call effects → irssaopt maximizes yield;
  the passes themselves don't depend on it.
- **Memory SSA.** Cross-block forwarding of load-bearing expressions
  needs a versioned memory state (ties into the `variable-recovery`
  topic's memory-SSA discussion). Deliberately out of scope here.
- **Where does SCCP's edge deletion live?** Killing provably-untaken
  edges changes the CFG that structuring consumes; whether that is an
  `irssaopt` rewrite or a structuring-time annotation should be fixed
  by the `structuring` topic file.
- **How much propagation is too much?** van Emmerik and the Dream++
  readability line pull opposite ways from compiler practice; the
  single-use/size thresholds in slice 3 need empirical tuning against
  real lifted corpora once emission exists.
