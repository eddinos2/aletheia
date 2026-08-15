# Plan: conservative dead-code elimination on SSA (`irssaopt`, slice 4)

## Input / output

Input: an `irssa::SsaFunction` (normally already through
`irssaopt::optimize`, which exposes the dead defs) plus the
architecture's function live-out register set. Output: a **new**
`SsaFunction` with unread pure definitions and unneeded φs removed, plus
a `DceStats`. Extends `src/irssaopt.rs` (DESIGN.md slice 4); wired into
`redump --ssa-opt` after `optimize`.

## The key decision: live-out roots are *cells*, not reaching versions

DESIGN lists "function live-out" among the mark roots. In *pruned* SSA
this is subtle: a version that reaches a `Return` without being read
there is not materialized (no φ exists for a cell nobody reads), so
"the version of rax at this return" is not recoverable post-construction
without redoing the dominator walk. This slice therefore marks
conservatively: **every definition of a live-out *cell* is a root**,
whatever its version. Sound (never deletes a return value or a
callee-saved restore), linear, and it still kills exactly the noise the
exit criterion names — flag writes and temporaries are never live-out
cells. Documented precision limit: a genuinely dead def of a live-out
cell (rax overwritten before any use) survives this slice; narrowing
needs reaching-version analysis or the signature slices (DESIGN 12–13),
deferred.

## The live-out tables (`src/callfx.rs`, additive)

ABI knowledge lives in `callfx`. New free function, same style as the
existing tables, cross-checked against the lifter namers in tests:

- `pub fn function_live_out(arch: model::Arch) -> Option<Vec<ir::Reg>>`
- x86-64: rax(0), rdx(2) (return pair), rbx(3), rsp(4), rbp(5),
  r12–r15 — all W64. No flags (dead at return under every ABI), no
  caller-saved scratch.
- aarch64: x0–x8 (return superset, over-approximation is sound — extra
  roots only keep more), x19–x28, x29, sp(31) — all W64. x30 is the
  `ret` target and gets marked through the branch read. No flags.
- `Other` → `None` (redump's arch gate already prevents reaching it).

Direction of soundness: live-out is an *over*-approximation — an extra
root keeps a dead def (lost precision); a missing root deletes a live
one (unsound). Generous supersets only.

## The algorithm: mark-and-sweep over the def-use index

Reuses slice 3's machinery: the `UseSite`/`Def` index, the name-table
compaction, the `partial` recomputation.

**Mark** (worklist over name ids, `BTreeSet`, deterministic):
- Seed: every name read anywhere inside a `Store`, `Branch`, or
  `Intrinsic` statement (these statements are never deleted, so their
  reads are unconditionally live) — intrinsic reads included, which is
  what keeps argument setups pinned across `callfx` sites; and every
  name whose `(space, cell)` is in the live-out set and whose def is an
  assign dst, a φ dst, or an intrinsic write. Version-0 names are never
  swept (they are `live_in`, the function's honest input signature) and
  need no seeding.
- Propagate: a marked name whose def is an `Assign` marks every name
  read by that assign's RHS; a marked φ dst marks every argument name.
  Fixpoint in O(names + uses).

**Sweep** (one pass, address then index order):
- Delete an `Assign` iff its dst name is unmarked **and** the RHS is
  load-free (`Load` may fault; deleting one is unproven — `irflow`
  doctrine verbatim). An unmarked load-bearing assign is kept and
  counted in `stats.kept_loads`.
- Delete a φ iff its dst is unmarked.
- Never delete a `Store`, `Branch`, or `Intrinsic`; never reorder;
  never touch the CFG (blocks, successors, `truncated`, `entry`,
  `live_in` semantics). Control-dependence DCE is structuring's job
  (DESIGN, verbatim).
- Then: name-table compaction over the swept dst names (slice 3's
  compaction, reused — factor it into a shared helper if it is not
  already callable), `partial` recomputed, `live_in` unchanged as the
  exact version-0 set.

Note the interaction that makes "φ needed only by a dead name is swept
with it" fall out for free: the dead assign's reads were never marked,
so the φ feeding only that assign stays unmarked and sweeps in the same
pass — no iteration needed. A single mark+sweep is a fixpoint (marking
is transitive); an idempotence test pins that.

```rust
pub struct DceStats {
    pub stmts_removed: usize,
    pub phis_removed: usize,
    pub names_removed: usize,
    pub kept_loads: usize, // dead but load-bearing, honestly kept
}
pub fn eliminate_dead(func: &SsaFunction, live_out: &[ir::Reg]) -> (SsaFunction, DceStats);
```

Same posture as `optimize`: input failing `irssa::check` is returned
unchanged (zeroed stats); pure, total, deterministic, no panics.

## Differential contract

`check_preserved` asserts equal statement counts, so DCE gets its own
`check_swept(input, output, live_out)`:
- output statements are a subsequence of input's (canonical-tuple
  comparison, modulo compaction remap);
- every removed statement is an `Assign` with load-free RHS or a φ;
- no removed dst tuple occurs in any output expression or φ argument,
  and no removed dst's cell is in `live_out` (for assign/φ defs);
- no `Store`/`Branch`/`Intrinsic` removed; block set/successors/
  `truncated`/`entry` equal; version-0 names identical.
Every test asserts `irssa::check` + `check_swept` on the output.

## redump

`--ssa-opt` becomes optimize → `eliminate_dead` (live-out from
`callfx::function_live_out`). Per function, when `stmts_removed > 0`,
print one comment line above the function:
`; dce: removed N of M statements` — the measured-reduction exit
criterion, deterministic. `--ssa` stays the faithful view. Usage text
updated.

## Module-by-module

- `src/irssaopt.rs`: mark/sweep, `DceStats`, `eliminate_dead`,
  `check_swept`, module-doc section (roots, the cells-not-versions
  decision and its documented precision limit, load doctrine), tests.
- `src/callfx.rs`: `function_live_out` + table tests (namer
  cross-check, flags absent, callee-saved present, x86-64 return pair).
- `src/bin/redump.rs`: pipeline + dce note + usage text.
- `ROADMAP.md`: Current-thread pointer → slice 5 (expression
  forwarding).

## Test matrix (~20)

1. cmp+jcc canonical: of `cmp`'s flag writes only the one the `jcc`
   reads survives (exact golden) — the DESIGN exit fixture.
2. flag def consumed two blocks later survives.
3. dead temp chain (`t0 := …; t1 := t0`, unread) fully swept; the
   shared root stays if another use is live.
4. φ feeding only a dead assign swept with it, same pass.
5. live-out: dead-looking `rax := 5` with no later use survives
   (return-value protection); same for rbx (callee-saved), rsp;
   a dead `rcx := 5` (caller-saved non-return, no call after) is swept.
6. load-bearing dead assign kept, `kept_loads` counted.
7. argument setup before a call survives (pinned by the `callfx` read).
8. stores/branches/intrinsics never removed (hand-built adversarial
   block); CFG fields byte-identical.
9. compaction: swept names leave the table, ids remapped, version-0 set
   identical, `partial` recomputed.
10. malformed input returned unchanged; empty function; no panics.
11. `check_swept` negatives: removed store, removed live-out def,
    output with a dangling read — each rejected.
12. determinism (twice → byte-equal) and idempotence (second
    `eliminate_dead` removes nothing).
13. optimize→dce composition: the slice-3 diamond fixture now loses its
    dead constant def after propagation.
14. seeded sweep (repo xorshift64* style): construct → optimize →
    eliminate_dead → `irssa::check` + `check_swept` Ok, no panics.
15. callfx table tests: `function_live_out` per arch (namer
    cross-check, flags absent, Other → None).
16. redump e2e: `--ssa-opt` on the calling fixture shows the dce note
    and drops dead flag writes; `--ssa` unchanged; byte-determinism;
    `/bin/ls` x86-64 slice: zero check failures, measurable reduction.

## Non-goals (this slice, per DESIGN)

- Reaching-version live-out precision (documented above; signatures
  slices own it).
- Control-dependence / dead-branch DCE — structuring owns CFG shape.
- Expression forwarding — slice 5.
- Removing `Store`s to provably dead stack slots — needs the memory
  slices (11+).
- Any `irssa` construction change (no live-out seeding of the pruning
  liveness — the cells-not-versions root rule exists precisely to avoid
  it).
