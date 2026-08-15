# Aletheia decompiler: engineering design

Synthesized 2026-08-15 from the seven topic files in this directory
(`ssa-optimization`, `structuring`, `type-recovery`, `variable-recovery`,
`calling-conventions`, `pseudocode-emission`, `incumbent-architectures` —
all seven exist; no research gaps to report) plus the repo state: git log
and ROADMAP confirm `irssa` (pruned SSA, commit `6d2e3fc`, plan in
`PLAN_SSA.md`) has landed on top of `ir` → `x86_lift`/`aarch64_lift` →
`irlift` → `irflow` → `cfg`, and the ROADMAP's Current thread queues an
ABI call-effects slice *before* SSA-based optimization. Published sources
only; clean-room discipline per `BRIEF.md` and `CONTRIBUTING.md`.

## Pipeline shape (the survey's verdict, adopted)

Every complete open system (Ghidra, RetDec, angr — see
`incumbent-architectures.md`) converged on: per-ISA lift → owned
width-explicit RTL → SSA → simplification → variable/signature/type
recovery → region-based structuring → emission. Aletheia's front half
already matches; nothing indicates re-platforming. The recorded failures
we design around: RetDec's borrowed IR (violates no-deps and lost
machine-level facts), goto-free structuring (DREAM/combing measured
*worse* against source ground truth by SAILR), and eager type
unification (one type-unsafe idiom pollutes everything — retypd's
critique). The differentiators to protect in every slice, because no
incumbent states them as contracts: determinism (byte-identical output
for equal inputs), totality (no panics on any input), total from-scratch
`check` functions, resource caps with honest stats flags, and explicit
proven-vs-heuristic provenance on every claim.

Target pipeline (new modules in **bold**):

```
cfg → irlift → irssa(+call effects, +MEM) → irssaopt → irstack → sig
                                     ↓
                        irstruct → irout → pseudo → redump --decompile
                                     ↑
                                  irtype (annotations)
```

## Ordering rationale

1. **Trust before speed** (slices 1–2): SSA def-use links across calls
   are silently wrong until call clobbers are modeled — the gap `irssa`
   documents. Everything downstream consumes def-use, so this lands
   first (ROADMAP agrees).
2. **Clean SSA before everything that reads it** (3–5): propagation and
   DCE shrink the raw lift toward source shape; structuring, variables,
   and types all get simpler inputs.
3. **End-to-end early** (6–8): a first honest `--decompile` (structure
   tree + named-by-id locals + suffixed operators) needs only
   structuring, out-of-SSA, and a renderer. Visible value per slice, and
   every later slice improves output that already exists.
4. **Deepen** (9–17): variables, then signatures (stack-parameter
   detection needs the stack analysis), then types (needs variables and
   call facts to type memory honestly).
5. **Fidelity** (18–19): SAILR-style de-optimizations and a
   ground-truth evaluation harness, once there is output to measure.

Every slice below is one commit, per the working-mode rule, with a
`PLAN_*.md` written before implementation. Common contract, stated once
and inherited by all slices: deterministic (BTree/sorted iteration only),
total, panic-free, capped (caps set stats flags, never diverge), output
re-passes upstream `check`s, and the slice's own `check` revalidates its
invariants from scratch.

---

## Slice 1 — `abi`: calling-convention tables as data

- **Goal:** the platform ABI as declarative, clean-room data: what a
  conforming call clobbers, where arguments and returns live.
- **Module:** `src/abi.rs` (new; pure data + lookup, no I/O).
- **Algorithm/citation:** table shape after Ghidra's published cspec
  model (ordered int/float arg registers, return storage, killed-by-call
  set, unaffected set, stack-arg base, shadow/red-zone sizes, sret
  register, varargs metadata); contents written from the primary specs:
  x86-64 SysV psABI (Matz et al.), Microsoft x64 documentation, AAPCS64
  (Arm IHI 0055) including the Apple variant divergence on anonymous
  varargs. Selection keyed (arch, container): ELF/Mach-O/PE →
  SysV/Apple-AAPCS64/Win64 — deterministic, Aletheia already knows both.
- **Invariants / `check`:** every listed register names a cell the ISA
  lifters actually emit (`x86_lift`/`aarch64_lift` numbering); arg
  sequences are duplicate-free; killed-by-call and unaffected sets are
  disjoint; return storage widths are valid `ir::Width`s. `check` is
  compile-time-ish but still total and callable in tests.
- **Test matrix:** table content asserted against the spec documents
  (register-by-register goldens for all three conventions); selection
  tests per container/arch pair; Go-detected binaries (via `gopcln`)
  map to a `NonStandardAbi` marker rather than a wrong table.
- **Exit criteria:** `abi::convention(arch, platform)` returns the right
  table for the three supported (arch, OS) combinations; zero unsafe,
  zero deps, doc-comment cites the specs.

## Slice 2 — `irssa` call effects: ABI-assumed defs/uses at calls

- **Goal:** make SSA def-use trustworthy across calls: no link threads a
  caller-saved register through a `BranchKind::Call`.
- **Module:** `src/irssa.rs` (extend; consumes `abi`).
- **Algorithm/citation:** during renaming, each call site *defines* new
  versions of the ABI killed-by-call set and *uses* the ABI argument
  set, exactly the template-as-sound-default architecture Ghidra's
  published cspec model and van Emmerik (PhD, Queensland, 2007) both
  converge on (`calling-conventions.md` §2, §1). Every version so
  created carries a new provenance tag `AbiAssumed` (a sibling of the
  existing `partial` honesty channel) — never claimed dataflow-proven.
  Unknown/indirect/import callees keep the full template (the only
  sound default).
- **Invariants / `check`:** every `AbiAssumed` name's def site is a
  call; no def-use edge crosses a call through a killed-by-call cell
  without an intervening `AbiAssumed` version; all prior `irssa::check`
  invariants (one def per name, dominance, phi shape) still hold; the
  module doc-comment's "call clobbers unmodeled" paragraph is replaced
  by the new honest statement.
- **Test matrix:** value in caller-saved reg live across a call now
  reaches version-(n+1), not n; callee-saved reg links unchanged; arg
  registers stay live up to the call; provenance renders as a note;
  determinism (build+render twice, byte-equal); seeded sweep: random
  CFGs with calls → construct → `check` Ok, no panic; existing 860+
  tests still green.
- **Exit criteria:** `redump --ssa` on a real binary shows versioned
  call clobbers with `AbiAssumed` notes; no def-use edge through RAX/X0
  across any call.

## Slice 3 — `irssaopt` A: sparse constant/copy propagation + φ-simplify

- **Goal:** cross-block constant and copy propagation — the pass
  `irflow` cannot do because it stops at block boundaries.
- **Module:** `src/irssaopt.rs` (new; `SsaFunction` → `SsaFunction`;
  `irflow` stays as the pre-SSA library whose `fold_expr` is reused).
- **Algorithm/citation:** sparse worklist propagation over def-use
  chains — on SSA, copy propagation is substitution because the
  single-definition property *is* the analysis (Cytron, Ferrante,
  Rosen, Wegman, Zadeck, TOPLAS 1991). A def whose RHS folds (via
  `irflow::fold_expr`) to a constant or bare name substitutes into all
  uses; φ(x,…,x) and φ(x, self) collapse to x. Full SCCP
  (Wegman–Zadeck, TOPLAS 1991) is deliberately deferred: its edge
  deletion mutates the CFG structuring will consume; when it lands, it
  *annotates* decided branches, never deletes edges.
- **Invariants / `check`:** no CFG mutation (block set, edges, and
  effectful statements unchanged); output passes `irssa::check`; rounds
  bounded like `irflow::MAX_ROUNDS`; expression growth bounded by
  `ir::MAX_EXPR_NODES`. Division is never folded to a trap
  (`irflow` doctrine verbatim).
- **Test matrix:** constant defined in one block, tested three blocks
  later → branch condition becomes constant expression (branch itself
  untouched); copy chains across a diamond collapse; φ-collapse
  goldens; cap-hit sets the stats flag and output still checks; seeded
  sweep; byte-equal determinism.
- **Exit criteria:** golden lifted fixture where a cross-block constant
  reaches its use; every output re-checks; no test regressions.

## Slice 4 — `irssaopt` B: conservative DCE on SSA

- **Goal:** remove the lift's noise — unread flag definitions and dead
  temporaries — function-wide.
- **Module:** `src/irssaopt.rs` (extend).
- **Algorithm/citation:** mark-and-sweep liveness over def-use (Cytron
  et al. 1991; van Emmerik 2007 on DCE as the pass that shrinks a raw
  lift toward source shape). Mark: names used by `Store`/`Branch`/
  `Intrinsic`, live φs, `partial` uses, `live_in`-reachable uses, and
  function live-out; propagate backward. Sweep: delete pure assignments
  to unmarked names **only when the RHS is load-free** — a `Load` may
  fault, deleting it is unproven (`irflow` doctrine carried over).
  Aggressive control-dependence DCE (removing dead branches) is
  deferred to structuring, which owns CFG shape.
- **Invariants / `check`:** no `Store`/`Branch`/`Intrinsic`/needed-φ
  removed; every deleted name provably unmarked; output passes
  `irssa::check`; linear in statements.
- **Test matrix:** flag defs consumed by no branch vanish; flag def
  consumed two blocks later survives; load-bearing dead assignment
  survives with a stats note; φ needed only by a dead name is swept
  with it; seeded sweep; determinism.
- **Exit criteria:** statement-count reduction measured and printed by
  the stats on a real lifted corpus; canonical fixture: `cmp`'s four
  flag writes reduce to the one flag the `jcc` reads.

## Slice 5 — `irssaopt` C: expression forwarding

- **Goal:** rebuild source-level expressions: collapse
  `ZF = (a-b == 0); branch if ZF` into `branch if a == b`.
- **Module:** `src/irssaopt.rs` (extend).
- **Algorithm/citation:** van Emmerik 2007 (the decompiler-specific
  workhorse; `ssa-optimization.md` §3): substitute a def's RHS into its
  uses when (a) RHS trivial (constant/name) — always; (b) RHS compound,
  pure, load-free — into any number of uses if under a size constant,
  else single-use only (readability rule, per DREAM++'s finding that
  duplication hurts — Yakdan et al., IEEE S&P 2016); (c) RHS contains a
  `Load` — only within the def's own block with no intervening
  `Store`/`Intrinsic`/call, exactly `irflow::propagate`'s barrier set
  lifted onto SSA names. Division never forwards past a branch (a trap
  must not move past its guard).
- **Invariants / `check`:** forwarded trees stay under
  `ir::MAX_EXPR_NODES`; load-bearing forwards are same-block and
  barrier-free (checkable by position); output passes `irssa::check`;
  every remaining multi-statement def is multi-use, effectful, or
  capped.
- **Test matrix:** the flag-collapse canonical case (x86 `cmp`+`jcc`
  and aarch64 `SUBS`+`B.cond` → relational branch — both ISAs); multi-
  use compound def stays named; load forwarding blocked by an
  intervening store; size-cap boundary; seeded sweep; determinism.
- **Exit criteria:** `redump --ssa` on a real function shows relational
  branch conditions where the raw lift showed flag plumbing, on both
  ISAs.

## Slice 6 — `irstruct`: Phoenix-style control-flow structuring

- **Goal:** CFG → structure tree (`Seq/If/Loop/Switch/Goto/Block/
  Opaque`), gotos permitted as honest output.
- **Module:** `src/irstruct.rs` (new; consumes the CFG plus the SSA
  overlay for simplified branch predicates; conditions stored as block
  id + polarity, never a rewritten expression).
- **Algorithm/citation:** iterative semantics-preserving structural
  analysis — Schwartz, Lee, Woo, Brumley, USENIX Security 2013
  (Phoenix), on the Cifuentes 1994 / Sharir 1980 schema lineage. Schema
  catalog: sequence, if-then, if-then-else, self-loop, natural
  while/do-while with break/continue to the unique follow node, and
  `switch` only where `jumptable` proved the table. When nothing
  matches, virtualize one edge into an explicit goto and retry —
  terminating because each round removes an edge; deterministic edge
  choice by lowest (source-VA, target-VA). DREAM (Yakdan et al., NDSS
  2015) and combing (Gussoni et al., AsiaCCS 2020) are rejected as the
  chassis on SAILR's evidence (Basque et al., USENIX Security 2024):
  goto-free-by-construction measures *farther* from original source;
  a `goto` is a proven rendering of an edge, duplication and condition
  synthesis are rewrites that must earn soundness. Structuring sits
  behind one trait so a SAILR-style strategy can be compared later on
  the same regions (angr's pluggable-structurer lesson).
- **Invariants / `check`:** the tree covers every reachable block
  exactly once (duplication banned ⇒ exact partition); every CFG edge
  realized as fall-through, structured construct, or explicit `Goto`;
  each `If`/`Loop` condition matches its block's branch statement;
  truncated/undecodable/indirect-successor blocks appear as
  `Opaque { reason }` — never absorbed, never given invented
  fall-through; `skipped` blocks listed, not structured; iteration cap
  with stats flag; loop follow node by documented deterministic rule
  (immediate post-dominator of header if any, else most-frequent exit,
  ties by address).
- **Test matrix:** each schema in isolation (golden trees); nested
  combinations; irreducible graph → minimal gotos, deterministic;
  proven jump table → `Switch`; truncated block → `Opaque` rendered
  marker; cap-hit degrades to gotos; seeded sweep over random CFGs →
  `check` Ok, no panic; determinism.
- **Exit criteria:** `redump --structure` renders golden trees for a
  fixture set including at least one real recovered function with a
  loop, a diamond, and an irreducible region.

## Slice 7 — `irout`: out-of-SSA into named locals

- **Goal:** φ-webs coalesce into the named variables pseudocode will
  print; surviving copies are visible code, so minimizing them is a
  readability goal, not an allocator's.
- **Module:** `src/irout.rs` (new; `SsaFunction` → variable assignment
  map + residual copy list).
- **Algorithm/citation:** Boissinot, Darte, Rastello, Dupont de
  Dinechin, Guillon, CGO 2009 — isolate φs with parallel copies first
  (correctness trivial), coalesce aggressively with dominance-based
  value-interference, then sequentialize surviving parallel copies with
  explicit swap handling; coalescing framed as Sreedhar et al., SAS
  1999 φ-congruence classes. Cytron's naive per-predecessor copies are
  rejected outright — the lost-copy and swap miscompiles (Briggs,
  Cooper, Harvey, Simpson, SP&E 1998). The clean correct-then-good
  separation is exactly the proven-vs-heuristic doctrine.
- **Invariants / `check`:** no two interfering names share a variable
  (recomputed interference from dominance + liveness); every φ
  resolved; residual copies form a valid sequentialization (swap cases
  via temporary); variable ids dense and deterministic; `AbiAssumed`
  and `partial` provenance survives onto variables.
- **Test matrix:** lost-copy fixture (φ result live past block) and
  swap fixture (φ permutation) — both must produce correct copies, the
  two published miscompile shapes as regression tests; diamond φ-web
  coalesces to one variable, zero copies; interference forces split;
  seeded sweep with a tiny SSA interpreter comparing pre/post values on
  random inputs; determinism.
- **Exit criteria:** on the fixture corpus, φ-count → variable-count
  reduction reported; zero residual copies on straight-line and simple
  diamond code.

## Slice 8 — `pseudo`: the renderer + `redump --decompile`

- **Goal:** first end-to-end pseudocode: structure tree + variables +
  expressions to deterministic text.
- **Module:** `src/pseudo.rs` (new) + `redump --decompile` wiring in
  `src/bin`.
- **Algorithm/citation:** precedence/associativity-table rendering as
  in Ghidra's open-source `PrintC` — parenthesize iff the child binds
  looser than context (or equal on the non-associative side) — **plus**
  forced redundant parentheses wherever bitwise/shift operators mix
  with comparisons or each other (the documented C-precedence defect;
  DREAM++ readability line, Yakdan et al. 2015/2016). Condition
  simplification (De Morgan, negation pushing) lands as proven W1
  rewrites in `irssaopt`, never in the printer: the printer only
  spells. House style inherited from `ir::render_with`: signedness on
  operators (`/u`, `<s`) until type recovery proves better — no
  invented C casts; widths explicit; naming via a `CellNamer`-style
  hook (default `vN`, so `irstack`/`irtype` names plug in later
  without touching the printer); statements carry originating VA as a
  right-margin comment (anchors the future Liu & Wang-style
  recompile-differential oracle, ISSTA 2020). Honesty markers rendered
  as comments: `/* lift truncated */`, `/* reads bits its def never
  wrote */`, `/* indirect jump: successors unknown */`,
  `/* abi-assumed */`. Line wrapping: fixed column rule now; Oppen
  (TOPLAS 1980)/Wadler layout only if real output demands it.
- **Invariants / `check`:** deterministic bytes (golden files);
  depth-bounded by `ir::MAX_EXPR_NODES` with explicit `…` truncation;
  total on malformed trees; round-trip property: every operator prints
  with enough parentheses that a reparse by precedence is unambiguous
  (tested by a tiny expression reparser in tests, not shipped).
- **Test matrix:** precedence goldens for every `BinOp`/`UnOp` pair;
  forced-paren cases; each `Struct` node kind; honesty markers; a real
  recovered function end-to-end golden per ISA; byte-equality on
  repeated runs; no-panic fuzz over hand-broken trees.
- **Exit criteria:** `redump --decompile` on a checked-in fixture
  binary emits stable, reviewed pseudocode for a nontrivial function on
  both x86-64 and aarch64. This is the milestone commit.

## Slice 9 — `irstack` A: affine stack-pointer tracking

- **Goal:** prove, per SSA name, `name = sp0 + c` where possible;
  classify every load/store address as `StackOff(c)`, `NonStack`, or
  `Unproven`.
- **Module:** `src/irstack.rs` (new; over `SsaFunction`, post-irssaopt
  so copies/constants are already propagated).
- **Algorithm/citation:** van Emmerik 2007 (SSA propagation normalizes
  stack accesses to constant offsets by itself; frame pointers are
  handled as values, not register names) as pure dataflow over a tiny
  abstract domain `Affine(c) | NotSp | Unknown`, join =
  equal-or-degrade; the same shape as angr's openly documented
  `variable_recovery_fast` offset tracking. Full VSA/DIVINE
  (Balakrishnan & Reps, CC 2004 / VMCAI 2007) is explicitly *not*
  adopted: its whole-program soundness premise contradicts `cfg`'s
  deliberate under-approximation, and SecondWrite's published numbers
  (ElWazeer et al., PLDI 2013) show cheap affine tracking reaches
  comparable stack-local precision.
- **Invariants / `check`:** facts are per-SSA-name (flow-insensitivity
  is free on SSA); `alloca`/dynamic adjustment detected as "sp no
  longer affine" — no claims below that point, honestly flagged; cap
  on distinct affine constants per function; AArch64 pre/post-index
  writeback verified to arrive as a separate assignment from
  `aarch64_lift` (test, and fix the lifter if not).
- **Test matrix:** prologue/epilogue frames (push/sub styles, both
  ISAs); frame-pointer chains (`bp = sp0 - c` then `bp - k` accesses);
  alignment re-adjustment around calls; alloca fixture → honest stop;
  classification goldens; seeded sweep; determinism.
- **Exit criteria:** on real fixtures, every compiler-generated local
  access classifies `StackOff`; `redump` gains a per-function stack
  fact dump.

## Slice 10 — `irstack` B: stack-slot identification

- **Goal:** partition proven `StackOff` accesses into evidence-backed
  slots — the raw material for named locals.
- **Module:** `src/irstack.rs` (extend).
- **Algorithm/citation:** ASI-lite — split at every observed access
  boundary, slot width is exactly what accesses justify, overlaps merge
  with sub-ranges noted (Ramalingam, Field, Tip, POPL 1999, as scoped
  by DIVINE, Balakrishnan & Reps VMCAI 2007: a-locs justified by
  observed accesses, never invented). Fallback discipline from
  SecondWrite: where splitting cannot be proven (sp non-affine,
  `Unproven` stores present), the frame region stays one opaque blob,
  labeled. Every slot carries its justification (the access list) —
  the evidence trail is the proof.
- **Invariants / `check`:** slots are disjoint byte ranges (sub-ranges
  nested); every slot byte is touched by a cited access; `Unproven`
  stores and calls recorded as clobber barriers; no slot claimed below
  a non-affine point.
- **Test matrix:** adjacent distinct locals stay distinct; W64 store +
  two W32 reads → one slot, two sub-ranges; overlapping evidence
  goldens; blob fallback fixture; determinism; seeded sweep.
- **Exit criteria:** slot table renders for real fixtures; `pseudo`'s
  namer hook can print `local_8` style names from slots (address-taken
  slots named but marked, never promoted — that is slice 11's job).

## Slice 11 — `irssa` MEM versioning + proven slot promotion

- **Goal:** version memory so alias honesty becomes ordinary def-use
  reasoning, then promote only provably-unaliased stack slots to SSA
  values.
- **Module:** `src/irssa.rs` (extend; consumes `irstack` + `abi`).
- **Algorithm/citation:** HSSA-lite — one pseudo-cell `MEM`; every
  `Store`/call/`Intrinsic` defines a new version (χ), every `Load` uses
  the current version (μ) (Chow, Chan, Liu, Lo, Streich, CC 1996); the
  same design Ghidra's public heritage/INDIRECT machinery and angr's
  documented variable unification embody. Promotion criteria, all
  three required: (i) all accesses proven `StackOff`; (ii) address
  never escapes (the affine value used solely in load/store address
  position); (iii) no clobber barrier may target it — calls
  conservatively clobber all memory except what the ABI's unaffected
  contract plus future call summaries exempt. Everything unpromoted
  stays a Load/Store against versioned MEM: visible, ordered, honest.
- **Invariants / `check`:** MEM versions form a valid SSA chain (one
  def per version, dominance); a promoted slot has its three criteria
  re-derivable by `check`; promotion never changes the set of effectful
  statements' relative order; promoted values re-enter `irssaopt`
  cleanly (idempotent re-run).
- **Test matrix:** spill/reload pair promotes and then folds away in
  irssaopt; address-taken local never promotes but renders named with
  marker; store through unknown pointer blocks promotion of everything
  it may reach; call barrier honored; end-to-end pseudocode diff shows
  locals replacing load/store noise; seeded sweep; determinism.
- **Exit criteria:** fixture where a compiler spill disappears from
  pseudocode while an address-taken local visibly survives as memory.

## Slice 12 — `sig` A: callee-side signature inference

- **Goal:** per-function `Signature { int_args, float_args,
  stack_bytes, returns, provenance }` from the callee's own dataflow.
- **Module:** `src/sig.rs` (new; side table keyed by VA — keeps `cfg`
  ISA-agnostic — consumed by `annotate`/`listing`/`pseudo`).
- **Algorithm/citation:** parameters = locations live-in at entry (van
  Emmerik 2007; Cifuentes 1994): intersect `SsaFunction::live_in`
  (already exactly the read-before-write set) with the ABI argument
  sequence; prefix rule for arity (evidence for arg k implies arity ≥
  k — Ghidra's published prototype-pruning model; TypeArmor, van der
  Veen et al., IEEE S&P 2016 validates intra-procedural scans at
  scale). Stack parameters via `irstack`: loads at positive in-bounds
  entry-sp offsets (above return address / in Win64 shadow space).
  Filter callee-save save/restore pairs (spill/reload of an
  `unaffected` register otherwise unused is not a parameter — van
  Emmerik's classic false positive).
- **Invariants / `check`:** every claimed argument cites its witness
  (the live-in name or stack load); provenance ∈ {`SymbolDerived`,
  `DataflowProven`, `AbiAssumed`, `Heuristic`} ranked like
  `funcs::Source`; arity respects the prefix rule; no argument outside
  the ABI candidate set is ever claimed at this slice.
- **Test matrix:** 0–7 int args, float args, mixed (Win64 positional
  aliasing: arg 2 is RDX *or* XMM1); stack args; callee-save
  save/restore false-positive fixture; unused-parameter invisibility
  documented as expected (caller side fixes it, slice 13); goldens per
  ISA/ABI; determinism.
- **Exit criteria:** `redump --sigs` prints witnessed signatures for
  fixture corpus; demangled-symbol functions (slice 14 hooks) reserved.

## Slice 13 — `sig` B: caller-side consensus and returns

- **Goal:** returns (invisible from the callee alone) and arity
  refinement from call sites.
- **Module:** `src/sig.rs` (extend; uses `cfg::call_graph`).
- **Algorithm/citation:** a return exists iff some caller reads
  RAX/X0/XMM0/V0 after the call before redefining it (van Emmerik
  2007); per-callsite arity by backward write-before-call scan
  (TypeArmor's rule); bottom-up over the call-graph SCC condensation
  with a bounded iteration cap — recursion or cap leaves the
  ABI-assumed default, flagged. Caller disagreement resolves to the
  maximum witnessed, tagged `Heuristic`. `CallTarget::Import` keeps
  full ABI assumptions (until an import-signature table exists);
  `CallTarget::Unknown` always keeps them. Thunk-only tail-call
  awareness initially (`cfg` already detects import thunks); general
  tail calls are a recorded open item.
- **Invariants / `check`:** fixpoint bounded and monotone (facts only
  tighten from the ABI default); every refinement cites a callsite;
  disagreements never silently averaged.
- **Test matrix:** return read in one of three callers → return
  claimed with citations; dead RAX after all calls → no return;
  recursive pair → flagged default; SCC cap; import/indirect defaults;
  determinism.
- **Exit criteria:** pseudocode renders `v0 = f(a, b)` vs `f(a, b)`
  correctly on fixtures, each claim carrying provenance.

## Slice 14 — `sig` C: varargs patterns + symbol-derived signatures

- **Goal:** stop varargs functions from inferring 6/8/14 false
  parameters; import the highest-trust signature source.
- **Module:** `src/sig.rs` (extend; consumes `cxxdemangle`/`demangle`/
  `rustmeta`/`gotype` names).
- **Algorithm/citation:** recognize the three documented `va_start`
  prologues from the ABI specs themselves — SysV register-save-area
  spill guarded by `test al, al`; Win64 shadow-space spill of RCX–R9;
  AAPCS64 X0–X7/Q0–Q7 dump — and mark `varargs(fixed_arity=k)`,
  heuristic-tagged, suppressing the false wide signature. SysV
  callsites' AL write gives the vector-arg count for free. Mangled
  symbols carry full parameter types: rank `SymbolDerived` above all
  inference, and use it as a free oracle *against* slices 12–13 in
  tests.
- **Invariants / `check`:** pattern matches cite the matched
  instruction range; symbol-derived signatures must be consistent with
  ABI candidate storage or the conflict is reported, not hidden.
- **Test matrix:** the three prologue fixtures (from spec-described
  layouts, clean-room); a printf-like callsite with AL; demangled C++
  fixture where symbol signature confirms/contradicts dataflow —
  contradiction renders as a conflict note; determinism.
- **Exit criteria:** on a real binary with libc varargs imports and
  mangled C++ symbols, no false 6-arg signatures and symbol-typed
  signatures win with provenance shown.

## Slices 15–17 — `irtype`: evidence, bounds, presentation

Three commits, one module `src/irtype.rs`, consuming `SsaFunction`
(one def per name ⇒ one type per name).

- **15 — Evidence facts (proven).** One pass emits per-name usage
  facts that restate the IR: `LoadedFrom(w)`/`StoredTo(w)` (name used
  as address), `SignedUse`/`UnsignedUse` (signed vs unsigned
  ops/compares/extends), `BoolUse` (W1 contexts), `ArithWith(const)`.
  Each fact cites its statement — the evidence trail is the proof. No
  solving, no policy. Facts bounded by statement count (already capped
  by `irlift`). *Check:* every fact's citation replays (re-derivable
  from the cited statement). *Tests:* per-fact goldens; pointer-ish
  load/store patterns; determinism. *Exit:* `redump --typefacts`.
  Citation: the fact language is TIE's constraint-generation layer
  (Lee, Avgerinos, Brumley, NDSS 2011), which is sound relative to IR
  semantics.
- **16 — Bounds propagation (proven, TIE-shaped, subtyping-directed).**
  Finite lattice (⊥ ≤ {int_w signed/unsigned/unknown, ptr(to-width,
  one level deep), bool} ≤ num_w ≤ ⊤) with **upper and lower bounds
  per name** (TIE's ranges — verdicts stay separate), propagated
  *directionally* along def-use and φ edges — retypd's lesson (Noonan,
  Loginov, Cok, PLDI 2016): machine code is type-unsafe, symmetric
  unification is unrecoverable, so constraints flow as subtyping.
  Conflicting evidence yields an explicit `Conflict` bound, reported
  never papered over. Finite lattice height ⇒ proven termination.
  *Check:* `lower ≤ upper` and width-consistency for every name.
  *Tests:* signed/unsigned mixes; pointer vs int arithmetic; conflict
  fixture; fixpoint-bound; determinism. *Exit:* bounds render per name.
  Struct/recursive types and polymorphic schemes deliberately deferred
  until call summaries and memory typing mature; when they land, adopt
  retypd's covariant-load/contravariant-store rule via the simpler
  BinSub formulation (Smith, SAS 2024) rather than PDA saturation.
- **17 — Presentation policy (heuristic, labeled).** A separate
  function maps a range to one display type, tagged `Proven { facts }`
  vs `Guess { range }`; `pseudo` renders guesses with a marker and
  `Conflict` as `/* conflicting evidence */ u64` — a wrong-but-
  confident type is worse than an honest one. Operator signedness
  suffixes drop from pseudocode only where the bound proves
  signedness. *Check:* every displayed type is within its name's
  range. *Tests:* golden pseudocode with and without proven types;
  conflict rendering; determinism. *Exit:* fixture pseudocode shows
  `int`-like declarations where evidence proves them and suffixed
  operators where it does not.

Seed signatures from slice 14 plumb in as ground-truth lower bounds at
call boundaries when both slices exist (sound relative to metadata).

## Slice 18 — `irstruct` de-optimization pre-pass (SAILR)

- **Goal:** remove *spurious* gotos — the residue of specific compiler
  transforms — while keeping genuine ones.
- **Module:** `src/irstruct.rs` (pre-passes in front of the
  structurer, each its own commit if more than one lands).
- **Algorithm/citation:** SAILR (Basque et al., USENIX Security 2024):
  most spurious gotos come from jump threading / cross-jumping /
  tail-merging; invert the highest-value one first — re-splitting a
  shared tail two predecessors jump into. That *is* controlled
  duplication, but of provably-identical statement lists: checkable
  (byte-equal IR), bounded (duplication cap with stats flag), and
  flagged as a rewrite in stats.
- **Invariants / `check`:** duplicated blocks are IR-identical to
  their source at the time of the split; duplication counted and
  capped; structure `check` still passes; goto count monotonically
  non-increasing on the fixture corpus (regression-guarded).
- **Test matrix:** a compiler-tail-merged fixture (checked-in bytes
  from documented offline provenance) whose goto disappears; a genuine
  source goto that survives; cap behavior; determinism.
- **Exit criteria:** measured goto-count drop on the fixture corpus
  with zero `check` regressions.

## Slice 19 — evaluation harness: ground-truth fixtures + metrics

- **Goal:** adopt the published evaluation methodology as regression
  infrastructure, not a paper exercise.
- **Module:** `tests/` fixtures + a small metrics helper (no new src
  module unless a bounded exact graph-edit-distance helper earns one).
- **Algorithm/citation:** checked-in fixture binaries compiled offline
  from checked-in C (documented provenance path per CONTRIBUTING — no
  build-time deps), with (a) goto counts, (b) structure similarity
  against the known source CFG — a bounded exact CFGED on small graphs,
  dependency-free (metric per SAILR), and (c) Liu & Wang-style semantic
  spot checks (ISSTA 2020) via the tiny SSA interpreter from slice 7
  comparing input/output behavior on concrete values.
- **Invariants / `check`:** metrics are deterministic; fixtures'
  provenance documented; regressions fail CI-style (`cargo test`).
- **Test matrix:** the harness *is* the test matrix; seed it with the
  fixtures accumulated by slices 6–18.
- **Exit criteria:** one command reports goto count, CFGED-lite, and
  semantic spot-check status across the corpus; numbers recorded in
  the ROADMAP so future slices show movement.

---

## What is deliberately not in the sequence (recorded, not omitted)

- **Full VSA/DIVINE** — soundness premise (whole-program, complete
  CFG) contradicts `cfg`'s under-approximation; cheap affine tracking
  reaches comparable stack-local precision (SecondWrite, PLDI 2013).
- **DREAM/combing structuring** — goto-free-by-construction measured
  worse against source (SAILR); rejected as chassis, reaching-condition
  ideas may inform later condition simplification.
- **SCCP edge deletion** — propagation lattice welcome later; CFG
  surgery belongs to structuring's owner, as annotation first.
- **ML/dynamic type and signature recovery** (EKLAVYA, REWARDS, Howard,
  DIRTY) — nondeterministic, data/dependency-heavy, no evidence trail;
  excluded by the no-deps, determinism, and honesty contracts.
- **External or generated IRs, SLEIGH-style spec languages** — RetDec
  is the recorded experiment against borrowed IRs; at two ISAs the
  `ir.rs` firewall captures the benefit at none of the cost.
- **Global value numbering** — modest decompiler gain, can hurt
  readability; revisit after emission exists.

## Cross-cutting open questions carried forward from the topics

- `Signature` side-table format and how `annotate`/`listing` consume it
  early (calling-conventions).
- Import-signature table (clean-room libc/Win32 prototypes from public
  docs): value is high, maintenance cost real — decide at slice 13.
- Commutative-operand canonical ordering: lives in `irflow`'s fold or
  a future value-numbering pass, never the printer (emission).
- Cross-block load forwarding needs memory versions (slice 11) before
  `irssaopt` C can extend; re-visit after both land.
- `--strict-c` compilable-output mode: only after type recovery, to
  unlock recompile-differential testing (Liu & Wang) as a fourth
  metric.
