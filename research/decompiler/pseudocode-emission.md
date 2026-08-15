# Pseudocode emission and readability

Topic 6 of the decompiler research phase (see `BRIEF.md`). Published
sources only; every recommendation names the Aletheia module it lands in.

## The problem

After SSA construction (`irssa`) and control-flow structuring (topic 2),
a function exists as φ-resolved three-address statements hanging off a
structured region tree. Emission must turn that into text a human reads:
(a) *expression-tree construction* — deciding which SSA definitions get
folded forward into their uses to form compound expressions, and which
become named locals; (b) *rendering* — printing those trees with an
operator-precedence discipline that is unambiguous, minimally noisy, and
byte-for-byte deterministic; and (c) *honesty* — the output must never
claim more than the analysis proved (signedness, widths, truncations,
unmodeled call effects). Readability is not taste: the DREAM++ line of
work measured it experimentally, and its results constrain the design.

## Strongest published approaches

### 1. Expression propagation in SSA — van Emmerik (PhD thesis, Univ. of Queensland, 2007)

*Static Single Assignment for Decompilation* is the baseline for (a).
Because SSA gives one definition per name, forward substitution of a
definition's right-hand side into its uses is a local, order-independent
rewrite — van Emmerik shows classical expression propagation is both
simplified and strengthened by SSA. The conditions under which folding a
def `x = e` into a use is safe and desirable:

- **Single use** (or the expression is trivially cheap — a constant or a
  register copy): duplicating a non-trivial `e` into several uses makes
  output *longer*, so multi-use defs become named locals instead.
- **No intervening interference**: if `e` contains a memory `Load`, no
  `Store`, call, or `Intrinsic` may occur between def and use (the load's
  value could change; a load may also fault, so it must not be
  duplicated or reordered past effects). If `e` reads a cell, no
  redefinition of that cell may intervene — in pruned SSA over `irssa`
  this is free for SSA-named cells, and only memory needs the check.
- **Size cap**: folding is bounded so a pathological chain does not
  produce an unreadable (or unrenderable) mega-expression.

Out-of-SSA translation (φ-webs to variables) is also treated in the
thesis and feeds topic 1 (`ssa-optimization`); emission consumes its
result as variable names.

### 2. DREAM / DREAM++ — Yakdan et al. (NDSS 2015; IEEE S&P 2016)

*No More Gotos* (NDSS 2015) is primarily structuring, but its emission
side established **semantics-preserving readability transformations**:
condition simplification (negation pushing, De Morgan's laws, collapsing
`!(a == b)` to `a != b`), congruence-based naming of repeated
subexpressions, and loop-form selection. *Helping Johnny to Analyze
Malware* (IEEE S&P 2016) then ran a controlled user study (students and
professional malware analysts): with the readability-optimized DREAM++
output, participants solved **3× more analysis tasks than with Hex-Rays
and 2× more than with plain DREAM**, and a majority preferred it. The
transferable findings, independent of any tool's internals:

- Fewer, well-named variables beat literal three-address form; but
  *over*-folding into giant expressions also hurts — DREAM++ names
  common subexpressions instead of duplicating them.
- Simplified, positively-phrased conditions measurably help.
- Every transformation must be semantics-preserving; readability edits
  that can change meaning were explicitly out of scope. This aligns
  exactly with Aletheia's proven-vs-heuristic doctrine.

The dewolf decompiler (Enders et al., NDSS BAR Workshop 2023) continued
this line, driving emission choices from user surveys; SAILR (Basque et
al., USENIX Security 2024) pushed back on aggressive transformation,
measuring output *against the original source* and showing that
transformations which stray far from compiled structure hurt — a caution
against readability rewrites the evidence does not support.

### 3. Precedence-driven rendering — Ghidra's open-source C printer; classic pretty-printing

Ghidra (open source, NSA) renders its high-level p-code AST through a
printer (`PrintC` in the public tree) driven by an **operator precedence
and associativity table**: a child expression is parenthesized iff its
operator binds looser than its context requires (equal precedence on the
non-associative side also parenthesizes). This is the standard minimal
scheme and the right skeleton. Two published cautions temper "minimal":

- C's precedence for `&`, `|`, `^`, shifts, and comparisons is a known
  defect (compilers ship `-Wparentheses` because humans misread it);
  readability work (DREAM++'s condition handling; dewolf's survey
  feedback) supports emitting *redundant* parentheses whenever bitwise,
  shift, and relational operators mix, regardless of what precedence
  would allow.
- Line-breaking of long expressions is a solved problem: Oppen's
  *Prettyprinting* (TOPLAS 1980) and Wadler's *A prettier printer*
  (2003) give linear-time, deterministic layout algorithms with bounded
  lookahead — implementable dependency-free if wrapping is wanted at
  all. (A simpler fixed rule — break after top-level operators past a
  column — is also deterministic and may be enough.)

### 4. Compilable output as a test oracle — RetDec; Liu & Wang (ISSTA 2020)

RetDec (Avast, open source) emits genuine C, which enables
recompile-and-compare testing; Liu & Wang, *How far we have come:
testing decompilation correctness of C decompilers* (ISSTA 2020), used
recompilation + differential execution to find hundreds of correctness
bugs in shipping decompilers. Emitting a C-like surface that a compiler
*could* accept (even if Aletheia never promises compilability) keeps this
oracle available and disciplines the renderer against ambiguity.

## Trade-offs

| Choice | Readability | Soundness risk | Fit to Aletheia contracts |
|---|---|---|---|
| No folding (render three-address SSA) | Poor (DREAM++ study) | None | Trivially deterministic; a fine *first* slice |
| Fold single-use pure defs (van Emmerik) | Good | None if interference rules enforced | Local, checkable rule; bounded by expr caps |
| Unlimited folding | Degrades (mega-expressions) | Load duplication/reordering | Violates resource-cap spirit |
| Minimal parens (pure precedence table) | Good until bitwise/relational mix | Ambiguity to *humans*, not semantics | Deterministic |
| Full parens (current `ir::render`) | Noisy at scale | None | What `render_expr` does today — right for IR, wrong for pseudocode |
| Hybrid: minimal + forced parens on mixed bitwise/shift/relational | Best per published usability findings | None | Recommended |
| C type-carried signedness now | Reads naturally | **Unsound before type recovery** — invents casts | Rejected: operators keep `/u`, `<s` suffixes until topic 3 lands |

## What the existing `ir` render conventions imply

`ir::render_with` (see `src/ir.rs`) fixes house style the pseudocode
renderer must either inherit or consciously diverge from, and the
contracts it must inherit unconditionally:

- **Determinism as a contract**: "the same statements always print the
  same bytes." The pseudocode renderer gets the same sentence in its
  doc-comment, the same golden-file tests, and no iteration over any
  unordered container.
- **Total and capped**: `render_expr` depth-bounds by `MAX_EXPR_NODES`
  and prints `…` rather than recursing unboundedly; the pseudocode
  renderer keeps the bound and the explicit truncation marker. No
  panics on any tree, including malformed ones.
- **Naming via hook**: `RegNamer`/`CellNamer` keep ISA knowledge out of
  the core. The pseudocode renderer takes the same shape of hook for
  variable names, so out-of-SSA naming (topic 1) and future
  type-informed naming plug in without touching the printer.
- **Signedness on operators, not types**: `/u`, `/s`, `>>u`, `<s`,
  `<=u` tokens. Until type recovery (topic 3) proves signedness, the
  pseudocode keeps these suffixed operators — honest, if unusual —
  rather than emitting C casts it cannot justify.
- **Widths are explicit**: constants render `0x2a.d`; `zext.q(...)`,
  `trunc.b(...)` are the only width changes. Pseudocode moves width to
  variable *declarations* once variables exist, keeps cast-style
  `(u64)`-analogous spellings for extend/truncate, and never lets a
  width disappear.
- **Honesty markers already exist upstream**: `irlift`'s `truncated`
  flag and `irssa`'s `partial` reads must surface in output as explicit
  comments (`/* lift truncated */`, `/* reads bits its def never
  wrote */`), matching the "an honest signal, never hidden" doctrine.

## Concrete recommendation for Aletheia

Two slices, each one commit, consuming `irssa::SsaFunction` (and later
the structuring module's region tree):

1. **`src/irexpr.rs` — expression forwarding.** Over a checked
   `SsaFunction`, fold a def into its use iff: exactly one use; the use
   is reachable without leaving the def's block (first slice:
   same-block only — dominance-based cross-block folding can follow);
   if the def's expression contains a `Load`, no `Store`, `Branch` with
   call kind, or `Intrinsic` sits between def and use; the folded tree
   stays under `ir::MAX_EXPR_NODES`. Multi-use defs and over-cap trees
   stay as assignments (they become named locals). Deterministic
   worklist in statement order; output re-passes `ir::check`; a total
   `check` of its own asserts every remaining def is multi-use, effectful,
   or capped. This is van Emmerik's rule set restated over existing types
   — no new IR.
2. **`src/pseudo.rs` — the renderer.** A precedence/associativity table
   over `ir::BinOp`/`ir::UnOp` mirroring C's ordering; parenthesize on
   looser-or-equal-nonassoc child, **plus forced parentheses whenever
   `And`/`Or`/`Xor`/shift operators mix with comparisons or with each
   other across levels**. Operator tokens keep the IR's signedness
   suffixes until type recovery replaces them provably. Variable names
   through a `CellNamer`-style hook (default `vN` from SSA name ids —
   already deterministic). Statements carry their originating VA as an
   optional right-margin comment (anchors for diffing and for the
   Liu & Wang-style recompile oracle later). Truncation/partial flags
   render as explicit comments. Contract block verbatim from `ir`:
   deterministic bytes, no panics, depth caps, total.

Condition simplification (De Morgan, negation pushing — the cheap half
of DREAM++'s wins) lands in `irflow`/`irexpr` as *proven* algebraic
rewrites on `W1` expressions, not in the printer: the printer never
rewrites, only spells.

## Open questions

- Cross-block forwarding: same-block folding misses the common
  `flag := cmp; branch flag` split only when a block boundary
  intervenes — measure on real lifts before adding dominance-based
  folding complexity.
- Should commutative-operand canonical order (for deterministic and
  *stable* output under equivalent inputs) be fixed in `irflow`'s fold,
  in value numbering (topic 1), or nowhere? It must not live in the
  printer.
- Emit strictly C-compilable output (unlocks recompile-differential
  testing per Liu & Wang) vs. honest pseudo with suffixed operators —
  likely: pseudo now, a `--strict-c` mode only after type recovery.
- Line wrapping: adopt an Oppen-style layouter, or a fixed column rule?
  Defer until real output shows the need.
- How much of DREAM++'s naming (congruence-based names for repeated
  subexpressions) is worth doing before variable recovery (topic 4)
  provides real names?

## Sources

- M. J. van Emmerik, *Static Single Assignment for Decompilation*, PhD
  thesis, University of Queensland, 2007.
- K. Yakdan, S. Eschweiler, E. Gerhards-Padilla, M. Smith, *No More
  Gotos: Decompilation Using Pattern-Independent Control-Flow
  Structuring and Semantics-Preserving Transformations*, NDSS 2015.
- K. Yakdan, S. Dechand, E. Gerhards-Padilla, M. Smith, *Helping Johnny
  to Analyze Malware: A Usability-Optimized Decompiler and Malware
  Analysis User Study*, IEEE S&P 2016.
- S. Enders et al., *dewolf: Improving Decompilation by Leveraging User
  Surveys*, NDSS BAR Workshop 2023.
- Z. L. Basque et al., *Ahoy SAILR! There is No Need to DREAM of C: A
  Compiler-Aware Structuring Algorithm for Binary Decompilation*,
  USENIX Security 2024.
- Ghidra decompiler C printer (open source; `PrintC` in the public
  ghidra repository) — precedence-table rendering.
- D. C. Oppen, *Prettyprinting*, ACM TOPLAS 1980; P. Wadler, *A
  Prettier Printer*, 2003.
- Z. Liu, S. Wang, *How Far We Have Come: Testing Decompilation
  Correctness of C Decompilers*, ISSTA 2020.
- RetDec (Avast), open-source decompiler emitting compilable C —
  public repository and documentation.
