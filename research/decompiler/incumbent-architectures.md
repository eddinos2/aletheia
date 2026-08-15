# Architecture survey of open decompiler systems

Topic file for the decompiler design phase (see `BRIEF.md`). Published
sources only: Ghidra's public design documents and source, RetDec's
open-sourced code and its authors' conference material, angr's papers
and documentation, and peer-reviewed evaluations of their output.
Clean-room discipline applies — no proprietary internals of any closed
tool are used or referenced; where an evaluation paper happens to also
measure a closed tool, only the *published measurement* is cited, never
its mechanism.

## The problem

Aletheia has the front half of a decompiler (`ir` →
`x86_lift`/`aarch64_lift` → `irlift` → `irflow` → `irssa`) and must
choose the shape of the back half without repeating a decade of other
projects' recorded mistakes. Three open systems have shipped complete
pipelines and been evaluated in the literature: Ghidra (NSA, Apache-2.0,
open since 2019), RetDec (AVG/Avast, open since 2017), and angr (UCSB
line of research, BSD). They embody three distinct answers to the same
design questions — who writes the lifter, what the mid-level IR is,
how simplification is organized, and what sits between SSA and emitted
pseudocode — and the published evaluations of their output let those
answers be compared on evidence rather than taste.

## The three systems

### 1. Ghidra — spec-driven lift into a small owned RTL, rule-based simplification

Public design docs ship in-repo (`Ghidra/Features/Decompiler/src/main/doc`,
the SLEIGH manual, and the decompiler's Doxygen). Architecture, per
those documents:

- **SLEIGH** is a processor-specification language (a descendant of
  SLED — Ramsey & Fernández, "Specifying Representations of Machine
  Instructions", *TOPLAS* 19(3), 1997) that describes *both* instruction
  encodings and their semantics; a compiler turns `.slaspec` into tables
  the runtime interprets. One spec per ISA yields disassembly and
  lifting for every downstream consumer.
- **P-code** is the owned register-transfer IR every ISA lifts into:
  operations over *varnodes* (address-space, offset, **explicit byte
  size**), with an escape hatch (`CALLOTHER`) for operations the IR
  does not model — semantics are never silently invented.
- The **decompiler proper is a standalone C++ engine** (the Java
  framework talks to it over a pipe) that knows nothing about any ISA:
  raw p-code → CFG → SSA form (the "heritage" phase — Cytron-style
  dominance-frontier placement over varnodes) → a fixpoint engine of
  many small, local, individually-justifiable **rewrite rules**
  (algebraic simplification, dead code, copy/expression propagation)
  interleaved with **type propagation** → control-flow structuring →
  C emission. Calling-convention knowledge lives in per-compiler
  XML "cspec" files, as data rather than code.

Lessons: (a) a strict ISA/IR firewall — the analysis engine never grows
an opcode table — is exactly Aletheia's `ir.rs` contract, validated at
industrial scale; (b) width/size-explicit values on every operand,
as in Aletheia's `Width`, are load-bearing for correctness; (c) a large
population of tiny provable rules beats a few monolithic passes for
extensibility and for arguing soundness one rule at a time; (d) an
explicit intrinsic escape hatch preserves honesty about unmodeled
semantics. The cost Ghidra pays: SLEIGH is a whole language with its
own compiler — worth it at ~50 processor modules, a multi-year detour
at two.

### 2. RetDec — lift into a reused compiler IR (LLVM)

Křoustek, Matula & Zemek, "RetDec: An Open-Source Machine-Code
Decompiler", Botconf 2017 (and the Pass-the-Salt 2018 update; source on
GitHub). Architecture per the authors: Capstone-decoded instructions →
**LLVM IR** (`capstone2llvmir`) → **62 passes** over LLVM IR (27 written
by RetDec, 35 borrowed stock from LLVM) → a backend IR ("BIR") → C (or
a Python-like rendering).

The bet was that reusing a mature compiler IR buys free optimization
infrastructure. The recorded outcome is the cautionary tale:

- LLVM IR is designed for *codegen*, not for representing
  partially-recovered machine state; machine-level facts (flags, exact
  stack discipline, aliasing of sub-registers) must be encoded
  awkwardly and are easily destroyed by stock passes that are sound for
  a compiler but meaning-losing for a decompiler.
- Borrowed passes optimize for emitted-code speed, not readability, so
  RetDec still needed its own 27 passes *plus* a second IR (BIR) before
  emission — the "free" middle end did not remove the need to build one.
- The dependency footprint (LLVM, Capstone, more) made the system heavy
  to build and maintain; upstream development largely stalled after
  2022, and comparative evaluations since (e.g. the SAILR evaluation,
  USENIX Security 2024; DecompileBench, arXiv 2025) place its output
  quality behind Ghidra's and angr's on structure and correctness
  measures.

Lesson for Aletheia: the no-external-deps rule is not just hygiene here —
the one incumbent that outsourced its IR and middle end got the least
architectural benefit from it. A decompiler wants an IR whose invariants
it owns.

### 3. angr — borrowed analysis IR below, invented decompilation IR above

Shoshitaishvili et al., "SOK: (State of) The Art of War: Offensive
Techniques in Binary Analysis", *IEEE S&P* 2016, describes the platform;
the decompiler is documented in angr's own docs and in Basque et al.
(below). Architecture: binaries lift to **VEX** — Valgrind's IR
(Nethercote & Seward, *PLDI* 2007), reused via PyVEX — which serves
symbolic execution well; but VEX was designed for instrumentation and
**cannot represent C-style nested expressions**, so the decompiler
begins by re-lifting the VEX CFG into **AIL** (angr Intermediate
Language): statement-level, typed expressions, virtual variables in
place of VEX temporaries. The **Clinic** pipeline then runs staged
passes over AIL (SSA-ification and expression folding, calling-
convention and variable recovery, simplification), after which
**RegionIdentifier** decomposes the graph into a region hierarchy and a
**RecursiveStructurer** applies a pluggable structuring algorithm
(Phoenix, DREAM, or SAILR) before `CStructuredCodeGenerator` emits C.

Basque et al., "Ahoy SAILR! There is No Need to DREAM of C: A
Compiler-Aware Structuring Algorithm for Binary Decompilation",
*USENIX Security* 2024, is both the decompiler's flagship result and
the field's best published evaluation methodology: it reimplements
Phoenix, DREAM, and rev.ng's combing inside one pipeline, introduces a
ground-truth metric (CFG edit distance against the source-compiled
graph, plus goto counts), and shows that most "spurious" gotos are the
residue of specific *compiler* transformations that a structurer can
recognize and invert — and that eliminating every goto (DREAM, combing)
*hurts* similarity to the original source.

Lessons: (a) a borrowed lift IR forces a second lift the moment you
want readable output — angr effectively paid for two front ends;
(b) the decompilation-side IR wants statement granularity *and*
expression trees (Aletheia's `ir::Expr` already has the trees; the gap
is a layer where multi-statement computations re-fuse into expressions);
(c) structuring should be a pluggable stage over a region hierarchy,
not welded to emission; (d) evaluation against compiled-from-source
ground truth is now the published standard.

## Published evaluations of output quality

- **Liu & Wang, "How Far We Have Come: Testing Decompilation
  Correctness of C Decompilers", *ISSTA* 2020** — EMI-style differential
  testing; 1,423 error-triggering inputs and 13 confirmed bugs in the
  open decompilers tested. Two findings matter architecturally:
  recompilable-and-correct output is achievable on real functions more
  often than folklore held, and the recurring bug classes are exactly
  IR-contract violations (width/sign confusion, mis-modeled flag
  semantics, bad pointer arithmetic) — the class Aletheia's total
  `ir::check` and width discipline are built to exclude by construction.
- **Basque et al., *USENIX Security* 2024 (SAILR)** — structure quality
  must be measured against source ground truth; goto-freeness is the
  wrong objective function; compiler-awareness beats pattern coverage.
- **Cao et al., "Evaluating the Effectiveness of Decompilers",
  *ISSTA* 2024** — corroborates that open decompilers (Ghidra, angr,
  RetDec) differ far more in variable/type recovery and structuring
  quality than in raw instruction-semantics fidelity: the front half is
  a solved-shape problem; the back half is where quality is decided.

## Trade-offs

| Question | Ghidra | RetDec | angr | Evidence-backed answer for a new system |
|---|---|---|---|---|
| Who writes the lifter | SLEIGH spec, generated | Capstone + translator | Valgrind's VEX, reused | Handwritten per-ISA is fine at 2 ISAs; the firewall (analysis code never sees opcodes) is the part that matters |
| Mid-level IR | Owned RTL (p-code), size-explicit | Reused compiler IR (LLVM) | Reused (VEX) + owned (AIL) on top | Own a small width-explicit RTL; reuse cost exceeds benefit (RetDec), or forces a second IR anyway (angr) |
| Simplification | Many small rules to fixpoint | Monolithic borrowed passes | Staged pass pipeline (Clinic) | Small individually-provable rewrites, bounded fixpoint — matches `irflow` already |
| SSA | Yes ("heritage") | LLVM's | Yes (AIL SSA) | Unanimous; `irssa` is the right substrate |
| Layer between SSA and C | Structuring over p-code + emission rules | BIR | Region hierarchy + structured nodes | A distinct structured/expression layer is needed; no incumbent emits straight from flat RTL |
| Structuring | Own algorithm, gotos allowed | Pattern-based | Pluggable: Phoenix/DREAM/SAILR | Region-based, pluggable, gotos permitted as honest output (SAILR result) |
| Unmodeled semantics | `CALLOTHER` intrinsic | Weakly handled | Simprocedures/intrinsics | Explicit intrinsics — Aletheia's `ir::Stmt::Intrinsic` already conforms |
| Determinism / totality / caps | Per-function limits; not a stated contract | Not a stated contract | Timeouts; Python exceptions | None makes Aletheia's guarantees; keeping them is a genuine differentiator, and caps have precedent |

## Concrete recommendation for Aletheia

1. **Keep the converged pipeline shape; Aletheia's front half already
   matches it.** All three systems arrive at: per-ISA lift → owned
   arch-neutral RTL → SSA → simplification → variable/type recovery →
   region-based structuring → emission. `ir`/`x86_lift`/`aarch64_lift`/
   `irlift`/`irflow`/`irssa` are the first five boxes; no re-platforming
   is indicated by any published result.
2. **Do not adopt an external or generated IR at any layer.** RetDec is
   the recorded experiment; the no-deps rule and the LLVM outcome agree.
   Likewise defer any SLEIGH-style spec language: at two ISAs the
   firewall in `ir.rs` (no opcode knowledge outside lifters) captures
   the whole benefit at none of the cost.
3. **Grow simplification as Ghidra grows it: one small rule, one proof
   obligation, one commit** — `irflow`'s "when in doubt, do nothing"
   fixpoint is already this pattern; future SSA-level passes should land
   the same way rather than as monolithic phases.
4. **Plan one new layer, not zero: a structured representation between
   `irssa` and pseudocode** (region hierarchy + re-fused expression
   statements — the AIL/structured-node lesson). It should be a new
   module (working name `irstruct`) consuming `irssa`, with its own
   total `check` (single-entry regions, every CFG edge accounted for)
   and deterministic render, and with structuring pluggable behind one
   trait so Phoenix-style and SAILR-style strategies can be compared on
   the same regions.
5. **Adopt the published evaluation methodology into the test matrix
   now, not at the end**: fixture functions compiled offline from
   checked-in C alongside checked-in bytes (no build-time deps), with
   goto counts and structure-similarity against the known source CFG as
   regression metrics, plus Liu & Wang-style semantic spot checks. Emit
   a `goto` rather than a wrong region — SAILR's data says this is also
   what *readers* are better served by.
6. **Keep the contracts the incumbents lack.** No incumbent offers
   deterministic byte-identical output, totality, and no-panic
   guarantees as a stated contract; published bug studies (Liu & Wang)
   show their failure modes are precisely the ones Aletheia's `check`
   discipline excludes. This is the differentiator to protect in every
   new slice, including resource caps (Ghidra's per-function limits are
   the precedent) on structuring and emission.

## Open questions

- Where exactly out-of-SSA and expression re-fusion live: inside
  `irssa` as a consumer API, or entirely in the future `irstruct` —
  interacts with the `ssa-optimization` topic's out-of-SSA findings.
- Call-effect modeling order: Ghidra's data-driven compiler specs
  suggest ABI clobber/argument tables as data (`calling-conventions`
  topic); the roadmap's call-effects slice should decide table format
  before SSA-level propagation crosses calls.
- How much of SAILR's compiler-awareness is portable without its
  corpus: which specific goto-inducing transformations are worth
  recognizing in a first structuring slice, and which are follow-ups.
- Ground-truth corpus mechanics under the no-deps rule: checked-in
  compiled fixtures need a documented offline provenance path in
  CONTRIBUTING terms.

## Sources

- Ghidra source and design docs: <https://github.com/NationalSecurityAgency/ghidra>
  (decompiler docs under `Ghidra/Features/Decompiler/src/main/doc`;
  SLEIGH manual in `GhidraDocs/languages`).
- N. Ramsey, M. Fernández, "Specifying Representations of Machine
  Instructions", *TOPLAS* 19(3), 1997.
- J. Křoustek, P. Matula, P. Zemek, "RetDec: An Open-Source
  Machine-Code Decompiler", Botconf 2017; slides
  <https://www.botconf.eu/wp-content/uploads/formidable/2/2017-KroustekMatulaZemek-retdec-slides-botconf-2017.pdf>;
  Pass-the-Salt 2018 update <https://2018.pass-the-salt.org/files/talks/04-retdec.pdf>.
- Y. Shoshitaishvili et al., "SOK: (State of) The Art of War: Offensive
  Techniques in Binary Analysis", *IEEE S&P* 2016.
- N. Nethercote, J. Seward, "Valgrind: A Framework for Heavyweight
  Dynamic Binary Instrumentation", *PLDI* 2007 (VEX IR).
- angr decompiler documentation: <https://docs.angr.io/en/latest/analyses/decompiler.html>.
- Z. L. Basque et al., "Ahoy SAILR! There is No Need to DREAM of C: A
  Compiler-Aware Structuring Algorithm for Binary Decompilation",
  *USENIX Security* 2024,
  <https://www.usenix.org/conference/usenixsecurity24/presentation/basque>.
- Z. Liu, S. Wang, "How Far We Have Come: Testing Decompilation
  Correctness of C Decompilers", *ISSTA* 2020,
  <https://dl.acm.org/doi/10.1145/3395363.3397370>.
- Y. Cao et al., "Evaluating the Effectiveness of Decompilers",
  *ISSTA* 2024, <https://dl.acm.org/doi/10.1145/3650212.3652144>.
