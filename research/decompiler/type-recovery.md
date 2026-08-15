# Type recovery on low-level code

Topic 3 of the decompiler research phase (see `BRIEF.md`). Published
sources only; every recommendation names the Aletheia module it lands in.

## The problem

A stripped binary contains no types: compilation erases `struct`,
`unsigned`, and `char *` down to widths and operations, and type-unsafe
source idioms plus type-unsafe optimizations mean the erased information
is not merely hidden but partially *destroyed*. Type recovery is the
inference problem of reconstituting source-level types from how values
are *used* — a value fed to a load is pointer-like, an operand of a
signed comparison is signed-integer-like — and it is intrinsically
under-determined: many source types are consistent with one binary, so
any tool that prints a single C type is choosing among consistent
answers. That makes this topic the sharpest test of Aletheia's
proven-vs-heuristic convention: the *evidence* (how a value is used) can
be sound; the *verdict* (the one type displayed) almost never is.

## Strongest published approaches

### 1. TIE — Lee, Avgerinos & Brumley, NDSS 2011

"TIE: Principled Reverse Engineering of Types in Binary Programs"
([paper](https://users.ece.cmu.edu/~dbrumley/pdf/Lee,%20Avgerinos,%20Brumley_2011_TIE%20Principled%20Reverse%20Engineering%20of%20Types%20in%20Binary%20Programs.pdf),
[NDSS page](https://www.ndss-symposium.org/ndss2011/tie-principled-reverse-engineering-of-types-in-binary-programs/)).
Static, constraint-based inference over BAP's lifted IR. Its two lasting
contributions are methodological:

- **A finite type lattice with upper *and* lower bounds per variable.**
  Constraint solving computes, for each variable, the most specific type
  the evidence forces (lower bound) and the most general type the
  evidence permits (upper bound). The result is a *range*, not a single
  type; picking a printable type from the range is an explicitly
  separate, final step.
- **Conservativeness and precision as distinct metrics.** A result is
  *conservative* if the ground-truth type lies within the inferred
  range; *precise* if the range is tight. TIE evaluates both and reports
  ~90% conservativeness — i.e. it is honest that even the range is
  sometimes wrong.

Constraint generation itself is sound relative to the IR semantics
(e.g. a value dereferenced at width `w` is a pointer to a `w`-sized
cell; an operand of signed division is signed). What is *not* sound:
TIE sits on top of value-set analysis (Balakrishnan & Reps' VSA
lineage) for variable recovery, which over/under-approximates on real
binaries, and the final range→type policy is a heuristic by
construction.

### 2. Retypd — Noonan, Loginov & Cok, PLDI 2016

"Polymorphic Type Inference for Machine Code"
([arXiv](https://arxiv.org/abs/1603.05495),
[ACM](https://dl.acm.org/doi/10.1145/2908080.2908119)); a clean-room
reimplementation is open-sourced by GrammaTech with an unusually good
algorithm note
([retypd/reference/type-recovery.rst](https://github.com/GrammaTech/retypd/blob/master/reference/type-recovery.rst)).
The state of the art in expressiveness:

- **Subtyping, not unification.** Type-unsafe machine code makes
  bidirectional unification collapse distinct types globally: one bad
  cast pollutes everything it unifies with. Directional subtype
  constraints contain the damage.
- **Covariant loads, contravariant stores.** Its key technical insight:
  a pointer's `.load` capability is covariant and its `.store`
  capability contravariant. Modeling this recovers pointer `const`
  annotations with 98% recall and keeps memory typing from degenerating.
- **Per-procedure polymorphic type schemes** (so `memcpy`-like callees
  don't force all callers' types together) and **recursive types**
  (linked structures), with constraint-set simplification via pushdown
  automaton saturation.

Soundness split: the constraint *calculus* is formally developed and its
simplification proven to preserve the constraint set's meaning. The
*translation* of instructions into constraints embeds heuristics
(which `add`s are pointer arithmetic, calling-convention assumptions),
and the final step from inferred "sketches" to displayable C types is,
again, a policy. Costs: the saturation machinery is heavyweight;
GrammaTech's reimplementation documents real-world scaling and
engineering pain. BinSub (Smith, SAS 2024,
[arXiv](https://arxiv.org/abs/2409.01841)) reformulates retypd via
algebraic subtyping and reports ~63× speedup at comparable precision —
evidence that retypd's *ideas* survive far simpler machinery.

### 3. Unification-based inference (SecondWrite; Ghidra's propagation)

SecondWrite (Elwazeer et al., PLDI 2013, "Scalable variable and data
type detection in a binary rewriter") and Ghidra's open-source
decompiler both use fast unification-flavored propagation of type facts
through dataflow, seeded by known function signatures. It scales
linearly and is simple, but the retypd paper's critique is on point:
unification is symmetric, so it cannot represent "T is *at least* a
pointer to something readable" without committing, and one type-unsafe
idiom (a union, a cast, `memcpy`) merges types that should stay apart.
Ghidra mitigates with decompiler-loop heuristics and user overrides —
i.e. it accepts heuristic verdicts and makes them editable.

### 4. Dynamic and learning-based lines (noted, not applicable)

REWARDS (Lin et al., NDSS 2010) and Howard (Slowinska et al., NDSS
2011) recover types from *execution traces* — sound only for covered
paths, and Aletheia does not execute inputs. ML approaches (DIRTY,
StateFormer, Idioms) predict types from learned priors — pure heuristic
with no evidence trail, and they require models Aletheia's
no-external-deps rule excludes. Both lines are out of scope; they are
listed so the gap is a decision, not an omission.

## Trade-offs

| Axis | TIE (bounds lattice) | Retypd (subtyping) | Unification |
| --- | --- | --- | --- |
| Evidence honesty | Best: keeps a range, separates verdict | Good: sketches retain structure | Poor: commits early, errors spread |
| Expressiveness | Scalar + pointer-to; weak on structs | Recursive structs, polymorphism, const | Whatever the seed signatures carry |
| Machinery cost | Small finite lattice, meet/join | Saturation/PDA; heavy (BinSub: simplifiable) | Trivial (union-find) |
| Termination proof | Finite lattice height — easy | Proven, but intricate | Easy |
| Dependencies | Needs variable recovery (VSA) for memory | Needs calling conventions + variable recovery | Needs seeds |
| Sound core / heuristic rim | Constraints sound; range→type heuristic | Calculus sound; constraint-gen + sketch→C heuristic | Merging itself is the heuristic |

Two structural conclusions that are *sound across all of the
literature*: (a) evidence collection and type presentation must be
separate phases, because only the first can be defended; (b) constraint
flow must be directional (subtyping-shaped), because machine code is
type-unsafe and symmetric merging is unrecoverable.

## Recommendation for Aletheia

New module `irtype.rs`, consuming `irssa::SsaFunction` — SSA is the
right substrate because def-use is explicit, one definition per name
means one type per name, and `SsaFunction::partial` already flags the
width-honesty edge cases. Staged as three one-commit slices:

1. **Evidence facts (proven).** One pass over SSA statements emits, per
   SSA name, a set of *usage facts* that are direct restatements of the
   IR: `LoadedFrom(w)` / `StoredTo(w)` (the name was the address of a
   load/store of width `w`), `SignedUse` (operand of `Sar`/signed
   cmp/div/`Extend(signed)`), `UnsignedUse`, `BoolUse` (W1 contexts),
   `ArithWith(const)`. Each fact cites the statement it came from —
   the evidence trail *is* the proof. Total, no solving, no policy.
   Caps: facts bounded by statement count, which `irlift` already caps
   (`MAX_FUNCTION_INSNS`); deterministic via `BTreeMap` keyed by SSA
   name, matching `cfg`'s determinism doctrine.
2. **Bounds propagation (proven, TIE-shaped).** A small finite lattice —
   roughly `⊥ ≤ {int_w signed/unsigned/unknown, ptr(to-width), bool} ≤
   num_w ≤ ⊤` per width — with upper and lower bounds per SSA name,
   propagated *directionally* along def-use and φ edges (retypd's
   lesson applied to TIE's machinery: constraints flow as subtyping,
   never unification). Finite lattice height gives a proven-terminating
   fixpoint; a `check` function verifies `lower ≤ upper` and
   width-consistency for every name, in the spirit of `ir::check`.
   Conflicting evidence yields an explicit `Conflict` bound — reported,
   never papered over.
3. **Presentation policy (heuristic, labeled).** A separate function
   maps a bounds range to one display type for the pseudocode printer,
   and the output type carries a provenance tag (`Proven { facts }` vs
   `Guess { range }`) so `annotate`/render can mark guessed types —
   the same honesty channel `irlift` uses for `truncated`.

Explicitly deferred, with reasons: Retypd-style struct/recursive types
and polymorphic schemes wait until variable recovery
(`variable-recovery.md` topic) and call effects exist, since memory
typing without alias honesty would violate the "when in doubt, do
nothing" doctrine `irflow` established; if/when memory typing lands,
adopt retypd's covariant-load/contravariant-store rule and prefer the
BinSub-style simple formulation over PDA saturation.

## Open questions

- Where do seed signatures come from? `gotype`/`rustmeta`/demangled
  symbols already recover rich type names for some binaries — plumbing
  those in as *ground-truth lower bounds* at call sites is high-value
  and sound-relative-to-metadata, but needs the calling-convention
  topic resolved first.
- Interprocedural propagation: per-function bounds first, then
  call-graph propagation? TIE is whole-program; Aletheia's caps argue
  for per-function with explicit call-boundary facts.
- Should the bounds lattice model `ptr(T)` one level deep only (TIE
  does, effectively)? One level keeps the lattice finite trivially;
  deeper pointer structure reintroduces retypd's machinery.
- How does `Conflict` render? A wrong-but-confident type is worse than
  `/* conflicting evidence */ u64` — needs a decision in the
  pseudocode-emission topic.
