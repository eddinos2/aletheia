# Plan: out-of-SSA into named locals (`irout`, slice 7)

## Input / output

Input: an `irssa::SsaFunction` (normally post-`optimize`/`forward`/
`eliminate_dead`, but any checked one). Output: **a map, not a rewrite**
— the SSA function is never mutated:

```rust
pub struct OutOfSsa {
    /// SSA name id -> variable id. Dense, deterministic, every name
    /// mapped (a name with no uses still gets its class's variable).
    pub var_of: Vec<u32>,
    /// Human-facing count (max id + 1).
    pub var_count: u32,
    /// Residual copies per CFG edge, `(pred VA, succ VA)`-keyed,
    /// already sequentialized: execute in order on that edge.
    /// A swap uses the explicit temp variable recorded in the copy.
    pub edge_copies: BTreeMap<(u64, u64), Vec<Copy>>,
    /// Variables carrying honesty markers forward: the variable ids
    /// whose value at some point is ABI-assumed (built from callfx
    /// effects) or partial (from `SsaFunction::partial`). See the
    /// provenance section.
    pub assumed: BTreeSet<u32>,
    pub partial: BTreeSet<u32>,
}
pub struct Copy { pub dst: u32, pub src: CopySrc } // src: Var(u32) | temp
pub struct OutStats { pub phis_resolved: usize, pub copies: usize,
                      pub coalesced: usize }
pub fn out_of_ssa(f: &SsaFunction) -> (OutOfSsa, OutStats);
pub fn check(f: &SsaFunction, out: &OutOfSsa) -> Result<(), OutFault>;
```

New module `src/irout.rs` (DESIGN.md slice 7 — Boissinot, Darte,
Rastello, Dupont de Dinechin, Guillon, CGO 2009; congruence framing
from Sreedhar et al., SAS 1999; the lost-copy and swap miscompiles from
Briggs et al., SP&E 1998 are the regression fixtures). Read
`research/decompiler/ssa-optimization.md` §4 and
`variable-recovery.md` before implementing. No redump flag in this
slice — the consumer is slice 8's renderer; a small stats line can ride
into `--decompile` then.

## Algorithm (correct-then-good, the doctrine)

1. **Correct by construction:** conceptually isolate every φ with
   parallel copies on its incoming edges (Boissinot's step 1). In map
   terms: start from singleton classes; a φ and its arguments *want*
   one class, and every edge whose argument ends up in a different
   class from the φ's gets a copy.
2. **Coalesce aggressively:** merge a φ's class with an argument's
   class when no *value* interference results — two names interfere
   only if their live ranges intersect AND they carry different
   values (Boissinot's dominance-based test: liveness computed over
   the SSA graph — def-to-uses, φ args live-out of their predecessor
   — with the intersection checked using the dominance property;
   value equality via copy-chasing through `Assign { value: Reg }`
   chains, the same roots slice 3's propagation already proves).
   Deterministic merge order: ascending φ block VA, then cell, then
   argument order.
3. **Sequentialize** each edge's surviving parallel copies: emit
   ready copies first (destination not read by a pending copy);
   break cycles with the one temp variable (the swap case). The two
   published miscompile shapes MUST have dedicated fixtures.

Version-0 names (`live_in`) keep cell identity: names of the same
cell may share a variable only through coalescing like everyone else,
but the *entry* values must not be merged into conflicting classes —
they are the function's parameters-in-waiting.

## Provenance (DESIGN: "`AbiAssumed` and `partial` survive")

Inventory step for the builder: find how the callfx-inserted effects
mark ABI assumption (the `callfx` intrinsic and
`callfx::function_live_out`) and what `SsaFunction::partial`
positions mean (see `irssa` module docs). A variable is `assumed` if
any of its names is defined by a callfx intrinsic write (a clobber
the ABI, not the code, asserts); `partial` if any of its names'
occurrences appear at a `partial` position. If the inventory shows a
cleaner carrier for either, use it and document the choice — the
requirement is that the markers reach slice 8's renderer, not the
exact encoding.

## `check` (recomputes, trusts nothing)

- No two names sharing a variable interfere (interference recomputed
  from scratch inside `check` — liveness + dominance + value
  equality, independently of the pass's own analysis).
- Every φ resolved: for each φ and each argument edge, either the
  argument's variable equals the φ's, or that edge carries a copy
  making it so.
- Each edge's copy list is a valid sequentialization (simulate it:
  no destination read after it is overwritten, temp used only in a
  cycle, list minimal is NOT required — validity is).
- `var_of` dense (ids `0..var_count` all used), deterministic.
- Malformed input (fails `irssa::check`): identity map (one variable
  per name), no copies, zeroed stats — refuse to interpret, the
  established posture.

## Test matrix (~18)

1. **lost-copy fixture** (φ result live past the block, Briggs Fig.
   shape) → correct copies, verified by the interpreter (below).
2. **swap fixture** (φ permutation across a back edge) → temp-using
   sequence, interpreter-verified.
3. diamond φ-web with non-interfering args → one variable, zero
   copies (the exit-criterion shape).
4. interference forces a split → distinct variables + edge copy.
5. copy-chain value equality: interfering ranges, same value → still
   coalesced, zero copies.
6. version-0 names: parameters keep distinct variables when both
   live.
7. self-loop φ; entry-block φ with the `None` edge; multiple φs one
   block sharing edges (parallel-copy semantics, not sequential).
8. **seeded sweep with a tiny SSA interpreter** (test-only): random
   small CFGs through `irssa::construct` (reuse the existing seeded
   harness pattern from `irssaopt`), evaluate the SSA function and
   the (variables + edge copies) rendition side by side on seeded
   pseudo-random inputs — deterministic seeds; memory as a
   `BTreeMap`; intrinsics havoc their writes from the seed; bounded
   steps (loops capped) — every observable register value at every
   block boundary must agree. This is the slice's real oracle.
9. determinism (twice → identical `OutOfSsa`), `check` Ok on every
   output, malformed input → identity posture, no panics.
10. real-binary sweep: `/bin/ls` x86-64 slice through the full
    pipeline then `out_of_ssa` on every function — zero `check`
    failures, φ-count → variable-count reduction reported (record
    the aggregate numbers in the commit message).

## Exit criteria (DESIGN, verbatim)

On the fixture corpus, φ-count → variable-count reduction reported;
zero residual copies on straight-line and simple diamond code.

## Non-goals (this slice)

- Variable *naming* (`vN` spelling is slice 8's `CellNamer` hook) and
  any rendering.
- Stack-slot variables, argument recovery (`irstack`, slices 9–10).
- Rewriting the IR out of SSA — the output is a map the renderer
  consumes; the SSA function stays the analysis truth.
- Copy minimality beyond Boissinot's coalescing (no ILP, no second
  pass).

## ROADMAP rider

Current-thread: slice 7 ✅ entry with the verified evidence; leave
Active pointing wherever the concurrent slices' merge order puts it
(the maintainer resolves the final text at merge — write the entry,
don't fight the pointer).
