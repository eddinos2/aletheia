# Decompiler research & design brief

Goal: turn "a decompiler is a multi-year project" (ROADMAP, Phase 5) into
an ordered sequence of one-commit slices on top of the existing stack
(`ir` → `x86_lift`/`aarch64_lift` → `irlift` → `irflow` → `irssa`).
This phase produces **design documents, not code**: one file per topic
below, then a synthesized `DESIGN.md` that fixes the slice order.

## Ground rules

- **Published sources only.** Academic papers, theses, and *openly
  documented* open-source projects (Ghidra, RetDec, angr). Never another
  tool's proprietary internals — clean-room discipline (CONTRIBUTING.md)
  applies to research too.
- **Grounded in this repo.** Every recommendation must name the Aletheia
  module it lands in and respect the existing contracts: width-explicit
  IR, total `check` functions, deterministic render, no external deps,
  proven-vs-heuristic honesty, resource caps, no panics on any input.
- **Checkpoint to disk first.** Each topic file is written *before* the
  researching agent reports back, so an interrupted run loses nothing.
  A re-run improves existing files instead of starting over.

## Topics (one file each, `<slug>.md` in this directory)

1. `ssa-optimization` — SSA-based simplification for decompilation:
   cross-block expression/copy/constant propagation, DCE on SSA,
   out-of-SSA translation. Baselines: Cytron et al.; van Emmerik's
   thesis (SSA applied to decompilation).
2. `structuring` — control-flow structuring: interval/structural
   analysis (Cifuentes), Phoenix (Schwartz et al. 2013), pattern-
   independent structuring / "No More Gotos" (Yakdan et al. 2015),
   rev.ng's comb approach. Goto minimization vs correctness.
3. `type-recovery` — type inference on low-level code: TIE (Lee et al.
   2011), Retypd (Noonan et al. 2016), unification vs subtyping-based
   approaches; what is sound vs heuristic.
4. `variable-recovery` — stack-slot and variable identification: stack
   layout analysis, VSA/DIVINE lineage (Balakrishnan & Reps), memory
   SSA; aliasing honesty.
5. `calling-conventions` — parameter/return recovery: ABI-knowledge-
   driven vs dataflow-inferred signatures, varargs, mixed cases.
6. `pseudocode-emission` — from SSA + structure to readable output:
   expression-tree building, operator precedence/parenthesization,
   readability findings from the Dream++ line of work.
7. `incumbent-architectures` — architecture survey of *open* systems
   (Ghidra's public design docs/source, RetDec, angr): pipeline shapes,
   IR choices, what they got right/wrong per published evaluations.
   Lessons only — no proprietary internals.

## Per-topic file structure

- The problem, in one paragraph.
- The 2–4 strongest published approaches, cited (author, year, venue).
- Trade-offs table or prose.
- A concrete recommendation for Aletheia, mapped onto existing modules.
- Open questions.

## Synthesis (`DESIGN.md`)

Reads whatever topic files exist (noting gaps honestly), checks git log
and `PLAN_SSA.md` for what has already landed, and fixes the slice
sequence: for each slice — goal, module, chosen algorithm (cited),
invariants/`check`s, test matrix, exit criteria.

## How to run

From a Claude Code session in this repo:
`Workflow({ name: "decompiler-research" })`
(script: `.claude/workflows/decompiler-research.js`). Safe to re-run —
existing topic files are extended, not clobbered, and a partial run's
files survive on disk regardless of how the run ends.
