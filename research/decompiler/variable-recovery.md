# Variable recovery: stack slots, a-locs, and aliasing honesty

Topic 4 of the decompiler research brief (`BRIEF.md`). Published sources
only; every recommendation names the Aletheia module it lands in.

## The problem

Machine code has no variables — it has registers and a flat memory, and
the compiler has erased the mapping from source variables onto them. A
local variable survives only as a cluster of accesses at some offset
from the stack pointer (or frame pointer); a struct survives only as a
family of related offsets from a common base. Variable recovery is the
task of re-inventing named storage from those access patterns: deciding
that `[sp0 - 8]` is one 8-byte slot, that writes to it and reads from it
are defs and uses of the *same* thing, and that no indirect store
elsewhere could have changed it in between. The last part is the hard
part and the honesty problem: promoting a memory slot to a register-like
SSA value is only sound if nothing aliases it, and a decompiler that
promotes optimistically silently miscompiles the very code (address-taken
locals, `alloca`, pointer arithmetic) a reverse engineer most needs to
see correctly. For Aletheia the contract is fixed in advance: a slot is
*proven* or it is not a variable — the analysis must never invent
storage it cannot justify, mirroring the "under-approximation, never a
guess" doctrine of `cfg.rs` and the load/store conservatism already in
`irflow.rs`.

## Strongest published approaches

### 1. VSA + a-locs + ASI: the Balakrishnan–Reps / DIVINE lineage

- Balakrishnan & Reps, *Analyzing Memory Accesses in x86 Executables*,
  CC 2004 — Value-Set Analysis (VSA): an abstract interpretation that
  tracks, for every register and abstract location, a *value set* — a
  strided interval per memory *region* (global, per-procedure stack
  "AR" region, per-allocation-site heap region). Addresses become
  (region, strided-interval-of-offsets) pairs, so a store's possible
  targets are a computable over-approximation.
- Balakrishnan & Reps, *DIVINE: DIscovering Variables IN Executables*,
  VMCAI 2007 — closes the chicken-and-egg loop: VSA needs a-locs
  (abstract locations) to track, but a-locs come from access patterns
  VSA discovers. DIVINE iterates VSA with Aggregate Structure
  Identification (ASI, Ramalingam et al., POPL 1999): each round, the
  offsets and strides VSA observed partition each region into finer
  a-locs, and VSA reruns over the refined set. Reported recovery: 88%
  of local variables and 89% of heap-object fields, versus 83% / 0% for
  the prior purely syntactic ("one a-loc per observed offset")
  technique.
- Balakrishnan & Reps, *WYSINWYX: What You See Is Not What You
  eXecute*, TOPLAS 2010 — the journal consolidation; documents the
  soundness caveats (unresolved indirect jumps/calls force weakening)
  and the cost: whole-program, interprocedural, context-sensitive
  fixpoint over a non-trivial abstract domain.

The enduring vocabulary even if the full machinery is not adopted:
**regions** (never confuse a stack offset with a global address),
**a-locs justified by observed accesses** (never invent a slot no
instruction touches), and **refinement by iteration** rather than
one-shot guessing.

### 2. SSA-resolved stack accesses: van Emmerik / Boomerang

Van Emmerik's PhD thesis (*Static Single Assignment for Decompilation*,
University of Queensland, 2007; the basis of the open-source Boomerang
decompiler) takes the opposite design point: no separate pointer
analysis. Put the code in SSA, propagate expressions, and stack accesses
normalize *by themselves* into `m[sp0 + k]` for the constant `k`s the
compiler used — because compilers overwhelmingly address locals as
constant offsets from a value that is an affine function of the entry
stack pointer. Slots are then named per distinct `k` (with extents from
the access widths), and "preserved" registers/slots are proven by
showing the SSA value at exit equals the value at entry. Aliasing is
handled by a blunt but honest rule: an indirect store whose address is
not a resolvable `sp0 + k` invalidates propagation across it. This is
cheap, intra-procedural, and degrades gracefully — exactly the shape of
analysis Aletheia's per-function pipeline already has.

### 3. Scalable best-effort splitting: SecondWrite

ElWazeer, Anand, Kotha, Smithson & Barua, *Scalable Variable and Data
Type Detection in a Binary Rewriter*, PLDI 2013 (SecondWrite): a
deliberately cheapened VSA — symbolic stack-height tracking plus a
lightweight points-to lattice — that splits each frame into variables by
observed access offsets and runs ~352× faster than full VSA at
comparable precision on their benchmarks, scaling to
millions-of-instruction binaries. Its published lesson for Aletheia is a
calibration point: the expensive parts of VSA (context sensitivity,
strided intervals over globals/heap) buy little for *stack-local*
variable recovery specifically; affine sp-tracking plus an
escape/alias check captures most of the value. Because SecondWrite
rewrites binaries (miscompilation is fatal), its fallback discipline is
also instructive: when splitting cannot be proven safe, the frame is
kept as one opaque blob — never a wrong split.

### 4. Memory SSA: Chow et al. HSSA, and its decompiler echoes

Chow, Chan, Liu, Lo & Streich, *Effective Representation of Aliases and
Indirect Memory Operations in SSA Form*, CC 1996 (HSSA): extend SSA to
memory with *virtual variables*, χ (may-def) and μ (may-use) operators,
so indirect stores get SSA versions too and alias-safe forwarding
becomes ordinary def-use reasoning; a single virtual variable can stand
for "all memory I can't distinguish", preventing version explosion.
Decompiler-side, the same idea appears in openly documented systems:
Ghidra's open-source decompiler runs *heritage* (SSA construction over
varnodes) with dedicated stack-pointer tracking so stack locations
become SSA-versioned storage, inserting INDIRECT ops at calls/stores to
model may-defs (all in the public `decompile/cpp` sources); angr's
`variable_recovery_fast` (open source, documented) tracks sp offsets
per block, creates `SimStackVariable`s keyed by offset, and then
*unifies* SSA-grained variables that denote the same slot. Both are
existence proofs that "one memory version stream + per-slot promotion
where proven" is the practical production design.

## Trade-offs

| Approach | Cost | Precision on stack locals | Aliasing honesty | Fit to Aletheia contracts |
|---|---|---|---|---|
| Full VSA + DIVINE/ASI | Whole-program fixpoint, complex domain, months of work | High (also heap/global fields) | Sound *if* CFG is complete — a bad assumption post-`cfg.rs` under-approximation | Poor as a first slice; right vocabulary, wrong weight |
| Van Emmerik SSA propagation | Intra-procedural, near-linear on top of existing SSA | High for compiler-generated frames | Honest by construction: unresolved store ⇒ barrier | Excellent — `irssa.rs` already exists; totality and caps easy |
| SecondWrite-style splitting | Linear-ish, scalable | High for stack; medium elsewhere | Fallback-to-blob discipline, published | Good calibration; same core as (2) |
| Memory SSA (HSSA-style) | One extra pseudo-cell in existing SSA machinery | Enabler, not a recovery itself | χ at every unproven store/call = exact honesty | Excellent — small delta to `irssa.rs` |

Common failure modes the literature warns about, restated as Aletheia
invariants: (a) a frame pointer is an *optimization*, not a given —
track `sp` and `bp`(-like) registers as affine `sp0 + c` facts, not by
register name; (b) `alloca`/dynamic adjustment breaks affine tracking —
the analysis must detect "sp no longer affine" and stop claiming slots
below that point, not guess; (c) an address-taken slot (its `sp0 + k`
value flows anywhere other than a load/store address position, into a
call, or into a store's *value*) may alias anything — it can be *named*
but never *promoted*; (d) calls may write callee-visible memory — until
the call-effects slice lands, a call is a full memory χ.

## Concrete recommendation for Aletheia

Three one-commit slices, in order, all per-function, deterministic
(BTreeMap iteration), capped, and panic-free:

1. **`irstack.rs` (new): affine stack-pointer tracking.** Over the SSA
   from `irssa::construct`, a forward analysis computing for each SSA
   name, where provable, the fact `name = sp0 + c` (`sp0` = version 0
   of the stack register; the analysis is a tiny abstract domain:
   `Affine(c) | NotSp | Unknown`, join = equal-or-degrade). Total
   `check`; renders each block's sp-height facts deterministically.
   Output additionally classifies every `Load`/`Store` address as
   `StackOff(c)`, `NonStack` (provably not sp-derived, e.g. a global
   constant address), or `Unproven`. This is van Emmerik's propagation
   result and angr's `variable_recovery_fast` offset tracking, as pure
   dataflow.

2. **Stack-slot identification (extends `irstack.rs`).** Partition the
   set of proven `StackOff` accesses into byte-range slots (ASI-lite:
   split at every observed access boundary; a slot's width is what the
   accesses justify, overlapping accesses merge into one slot with
   sub-ranges noted — never widen beyond evidence). Every slot carries
   its justification (the access list). An `Unproven` store, or a call,
   is recorded as a *clobber barrier*. No output is produced for frames
   where sp goes non-affine except above the last affine point — the
   SecondWrite blob-fallback, honestly labeled.

3. **Memory versioning in `irssa.rs` (HSSA-lite).** Add one pseudo-cell
   `MEM` to SSA construction: every `Store`, call, and `Intrinsic`
   defines a new version (a χ); every `Load` uses the current version
   (a μ). Then promote a slot to an ordinary SSA value **only** when
   (i) all its accesses are proven `StackOff`, (ii) its address never
   escapes (the affine value is used solely in load/store address
   position), and (iii) no clobber barrier's may-target-set can include
   it — with calls conservatively including everything until the
   call-effects slice refines them. Everything not promoted stays a
   `Load`/`Store` against versioned `MEM`: visible, ordered, honest.

Full VSA/DIVINE is *not* recommended as a slice: its soundness premise
(complete CFG, whole-program) contradicts `cfg.rs`'s deliberate
under-approximation, and SecondWrite's published numbers show the
cheap analysis reaches comparable stack-local precision. Revisit VSA
vocabulary (regions, strided intervals) only if/when global and heap
variable recovery are scheduled.

## Open questions

- **Interaction with the call-effects slice** (next per ROADMAP): once
  callee save/clobber summaries exist, barrier (2)/(3) can exempt slots
  a callee provably cannot reach — what is the summary format that
  stays "proven vs heuristic" labeled?
- **Frame-pointer chains**: `bp = sp0 + c` then accesses via `bp - k`
  fall out of the affine domain automatically, but leaf-frame `sp`
  re-adjustments around variadic calls (x86-64 `sub rsp, 8` alignment
  dances) need the domain to survive multiple adjustments — cap on the
  number of distinct affine constants per function?
- **Aliased-but-named slots**: an address-taken local should still be
  *displayed* as a variable in pseudocode even though it is never
  promoted. Where does the proven/heuristic boundary get rendered —
  a marker in the eventual pseudocode emitter (`pseudocode-emission`
  topic), or a field on the slot?
- **AArch64 pre/post-index addressing** writes back the base register;
  the affine domain handles it, but the lifter must emit the writeback
  as a separate assignment (verify `aarch64_lift.rs` already does).
- **Overlapping slot evidence** (a `W64` store, `W32` reads of both
  halves): report one slot with two sub-ranges or two slots with a
  parent? DIVINE's ASI answer is a tree; is a flat range list enough
  for the first slice?

## Sources

- Balakrishnan & Reps, DIVINE, VMCAI 2007 —
  <https://research.cs.wisc.edu/wpis/papers/vmcai07.invited.pdf>
- Balakrishnan & Reps, WYSINWYX, TOPLAS 2010 —
  <https://www.semanticscholar.org/paper/f19306e70c7374a6d9e9133bc419a2ef9678e7c2>
- Van Emmerik, *Static Single Assignment for Decompilation*, PhD
  thesis, University of Queensland, 2007 (Boomerang, open source).
- ElWazeer et al., PLDI 2013 —
  <https://terpconnect.umd.edu/~barua/elwazeer-PLDI-2013.pdf>
- Chow et al., CC 1996 (HSSA) —
  <https://link.springer.com/chapter/10.1007/3-540-61053-7_66>
- Ramalingam, Field & Tip, *Aggregate Structure Identification and Its
  Application to Program Analysis*, POPL 1999.
- Ghidra decompiler sources (heritage, stack-pointer tracking; Apache-2) —
  <https://github.com/NationalSecurityAgency/ghidra/blob/master/Ghidra/Features/Decompiler/src/decompile/cpp/varnode.hh>
- angr `variable_recovery` / `variable_recovery_fast` (BSD, documented) —
  <https://docs.angr.io/en/v9.2.81/_modules/angr/analyses/variable_recovery/variable_recovery_fast.html>
