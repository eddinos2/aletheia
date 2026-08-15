# Calling-convention and signature recovery

## The problem

A lifted function is a bag of register and memory effects with no notion of
"parameter" or "return value": the caller materializes arguments into
ABI-designated registers and stack slots, the callee reads some subset of
them, and the return value is whatever the caller happens to use out of the
return register afterward. Signature recovery must decide, per function,
(a) which at-entry locations are parameters, (b) whether and where a value
is returned, and (c) which registers a call *clobbers* — the last being a
prerequisite for every interprocedural dataflow pass, because an unmodeled
clobber makes SSA def-use links across calls silently wrong (exactly the
gap `irssa` documents today: "call clobbers unmodeled"). The design axis is
**ABI-knowledge-driven** (assume the platform convention and fill in a
template) versus **dataflow-inferred** (observe which locations are
read-before-written in the callee and written-before-call in the callers);
every serious open system ends up hybridizing the two, and the honest
question is which facts each side is allowed to assert.

## Strongest published approaches

### 1. SSA-liveness inference — van Emmerik (PhD thesis, Univ. of Queensland, 2007), building on Cifuentes (dcc thesis, 1994)

Van Emmerik's "Static Single Assignment for Decompilation" gives the
canonical dataflow formulation: **parameters are the locations live-in at
function entry** (used before defined, after filtering callee-save
save/restore pairs), and **returns are the locations defined by the callee
that are live after the call in at least one caller**. Both are computed as
an interprocedural fixpoint over the call graph, because a callee's
parameter set depends on its own callees' clobber sets. Cifuentes' earlier
dcc did the same with classic reaching-definitions/liveness before SSA.
Strength: needs no ABI table, works on nonstandard/custom conventions
(hand-written asm, whole-program-optimized internal functions). Weakness:
callee-save save/restore pairs, register spills, and dead argument writes
produce false parameters/returns without careful filtering; recursion and
indirect calls force conservative assumptions; a parameter that the callee
never touches (passed through to a further call, or genuinely unused) is
invisible.

### 2. ABI template + dataflow pruning — Ghidra's prototype model (open source; documented in the decompiler docs and `*.cspec` compiler specs)

Ghidra encodes each convention as declarative data: an ordered list of
parameter storage locations (integer and float register sequences, then
stack), return storage by size, the *killed-by-call* (caller-saved) set,
the *unaffected* (callee-saved) set, stack-pointer bias, and per-compiler
quirks — one XML `cspec` per (arch, OS, compiler). The decompiler then runs
both directions of dataflow and **prunes the template**: candidate inputs
are trimmed to those actually consumed, with a *prefix rule* — since the
ABI assigns argument registers in order, evidence for the k-th argument
register implies arity ≥ k even if an earlier one is unused — and unknown
callees default to the full template's clobber assumption. This is the
architecture to copy: the ABI supplies the *candidate lattice and the
sound default*; dataflow supplies the *witnesses* that shrink it.

### 3. Arity by liveness at scale — TypeArmor (van der Veen et al., IEEE S&P 2016) and SecondWrite (ElWazeer et al., PLDI 2013)

TypeArmor recovers argument *counts* (and coarse widths, extended by τCFI,
Muntean et al., RAID 2018) for CFI: at each **callee**, a forward
read-before-write scan over the ABI argument registers gives an
upper/lower bound on arity; at each **callsite**, a backward
write-before-call scan gives the count the caller prepared. Published
evaluation shows high accuracy on optimized x86-64 binaries using purely
intra-procedural scans plus the prefix rule — evidence that Aletheia's
first slice does not need a whole-program fixpoint to be useful.
SecondWrite (a rewriting decompiler) reports the same structure inside an
LLVM-IR lifter, adding stack-argument detection via a stack-height
analysis, and documents the failure modes (spill slots mistaken for stack
args; float/int register file interplay).

### 4. Convention *identification* and learned recovery — angr's CallingConventionAnalysis (open source; angr, Shoshitaishvili et al., IEEE S&P 2016) and EKLAVYA (Chua et al., USENIX Security 2017)

angr collects per-function "facts" (registers/stack offsets read before
write, values live at returns) and then *matches them against a library of
known convention templates* (SimCC subclasses per platform), picking the
convention that explains the facts — useful when one binary mixes
conventions (Win64 + custom fastcalls, Go's pre-1.17 stack ABI). EKLAVYA
instead trains an RNN on instruction sequences to predict arity/types;
published accuracy is good but it is nondeterministic, data-dependent, and
dependency-heavy — cited here as the road Aletheia deliberately does not
take (violates no-deps, determinism, and proven-vs-heuristic honesty).

### The ABI ground truth itself (primary published sources)

- **x86-64 SysV psABI** (Matz, Hubička, Jaeger, Mitchell, eds.): integer
  args RDI,RSI,RDX,RCX,R8,R9; vector args XMM0–7; returns RAX(+RDX),
  XMM0(+XMM1); caller-saved = args + RAX,R10,R11 + all XMM; callee-saved
  RBX,RBP,R12–R15; stack args at [entry-RSP+8]; 128-byte red zone.
  **Varargs:** AL carries the count of vector registers used at the call;
  a `va_start` callee spills all six GPR args (and, guarded by `test
  al,al`, XMM0–7) into a 176-byte register save area — a crisp,
  documented, recognizable pattern.
- **Microsoft x64** (openly documented by Microsoft): RCX,RDX,R8,R9 with
  int/float *positional aliasing* (arg 2 is RDX **or** XMM1, never both);
  32-byte shadow space always allocated by the caller; returns RAX/XMM0;
  callee-saved includes RSI,RDI and XMM6–15. **Varargs:** caller passes
  floats duplicated in the GPR; callee spills RCX–R9 to the shadow space.
- **AAPCS64** (Arm IHI 0055): args X0–X7, V0–V7; returns X0(+X1)/V0;
  X8 = indirect-result (sret) pointer; X9–X15,X16/X17 (IP0/IP1),X18
  caller-saved/reserved; X19–X28 callee-saved; NGRN/NSRN allocation.
  **Varargs:** anonymous args use the same registers as named ones
  (unlike Apple's documented AAPCS64 variant, which passes all anonymous
  args on the stack — a real divergence Aletheia's Mach-O path must encode);
  a `va_start` callee dumps X0–X7/Q0–Q7 into a save area and builds the
  documented five-field `va_list`.

## Trade-offs

| Axis | ABI-driven | Dataflow-inferred |
|---|---|---|
| Soundness of *clobber* modeling | Sound by spec for conforming code; the only safe default for unknown/indirect/import callees | Sound only after a converged whole-call-graph fixpoint; unsound if any callee is unanalyzed |
| Custom/internal conventions | Wrong (silently) | Correct — the only source that can see them |
| Unused parameters | Recovered (via caller side + prefix rule) | Invisible from the callee alone |
| Cost | O(1) table lookup | Callee scan is cheap; caller-consensus + fixpoint needs the call graph and caps |
| Honesty labeling | "assumed per ABI" | "witnessed by dataflow" |
| Varargs | Detectable only as a documented code pattern | Register-save-area spill looks like 14 false parameters unless the pattern is recognized first |

The synthesis every mature open system converges on: **ABI template as the
prior and the sound default; dataflow as evidence that refines it; explicit
provenance on every fact.** That maps perfectly onto Aletheia's
proven-vs-heuristic doctrine.

## Concrete recommendation for Aletheia

**Slice 0 — `abi.rs` (new module, pure data).** Declarative convention
tables keyed by (arch, platform), mirroring Ghidra's cspec *shape* but
written clean-room from the psABI/Microsoft/Arm documents above: ordered
int/float arg registers, return registers, killed-by-call set, unaffected
set, stack-arg base offset, shadow-space/red-zone sizes, varargs metadata
(AL protocol, save-area size/layout), and the sret register (RDI hidden
arg / X8). Aletheia already knows the platform from the container
(`elf`/`pe`/`macho` → SysV vs Win64 vs Apple-AAPCS64), so selection is
deterministic. Total, no I/O, trivially testable.

**Slice 1 — ABI-assumed call effects (the ROADMAP's queued call-effects
slice).** In `irssa` (with `irflow`'s liveness), treat each
`BranchKind::Call` as: *def* of the ABI killed-by-call set (starting new
versions, so no def-use link crosses a call through a caller-saved
register) and *use* of the full ABI argument set (keeping argument setup
live). Every version created this way is tagged `AbiAssumed`, a new
provenance enum alongside the existing `partial` honesty channel — never
claimed as dataflow-proven. This alone makes SSA def-use trustworthy and
unblocks cross-block propagation.

**Slice 2 — callee-side signature inference.** A new `sig.rs` pass per
function: intersect `SsaFunction::live_in` (version-0 cells — Aletheia
already computes exactly the read-before-write set van Emmerik's method
needs) with the ABI argument sequence; apply the prefix rule for arity;
detect stack parameters as loads at positive, in-bounds offsets from the
entry stack pointer (above the return address / inside Win64 shadow
space). Filter callee-save save/restore pairs (a push/pop or spill/reload
of an `unaffected` register whose value is otherwise unused is not a
parameter — van Emmerik's classic false-positive). Output: `Signature {
int_args, float_args, stack_bytes, returns, provenance }` per function,
rendered deterministically.

**Slice 3 — caller-side consensus and returns.** Using `cfg`'s
`call_graph` (already a deterministic BTree), refine: a return value
exists iff some caller reads RAX/X0 (or XMM0/V0) after the call before
redefining it; arity at each callsite is the backward write-before-call
scan (TypeArmor's rule); disagreements resolve to the maximum witnessed,
tagged `Heuristic` when callers disagree. Run bottom-up over the
call-graph SCC condensation with a bounded iteration cap (resource-caps
doctrine); recursion or a hit cap leaves the ABI-assumed default in place,
flagged. `CallTarget::Import` callees keep full ABI assumptions unless a
future import-signature table refines them; `CallTarget::Unknown`
(indirect) always keeps them.

**Slice 4 — varargs and metadata sources.** Recognize the three documented
`va_start` prologue patterns (SysV register-save-area + `test al,al`
guard; Win64 shadow-space spill of RCX–R9; AAPCS64 X0–X7/Q0–Q7 dump) and
mark the function `varargs(fixed_arity=k)` — heuristic-tagged, and it
*suppresses* the otherwise-inferred 6/8/14-argument false signature. At
callsites to varargs functions, SysV's AL write gives the vector-arg
count for free. Independently, `cxxdemangle`/`rustmeta` symbol names
encode full parameter types for mangled functions — when present, that is
the highest-trust source (rank it like `funcs::Source` ranks starts:
symbol-derived > dataflow-proven > ABI-assumed > heuristic), and it doubles
as a free test oracle against the inference passes.

Feeds from existing metadata, summarized: `funcs::Source` precedence
pattern → provenance ranking; `cfg::call_graph` + `CallTarget` → caller
consensus and import/indirect defaults; `irssa::live_in` → the callee
read-before-write set; `irflow` liveness → after-call return-register
liveness; container type → ABI selection.

## Open questions

- Where does `Signature` live — a field on `cfg::Function`, or a separate
  side table keyed by VA (keeping `cfg` ISA-agnostic argues for the side
  table)? The side table also lets `annotate`/`listing` consume it early.
- Stack-argument detection needs at least a per-block stack-pointer delta
  analysis; is that a prerequisite mini-slice inside `irflow`, and does it
  share machinery with the variable-recovery topic's stack-layout pass?
- Import signatures: embedding even a small clean-room table of libc/Win32
  prototypes (from public man pages / MS docs) would sharpen caller-side
  facts enormously — worth the binary-size and maintenance cost, or defer?
- Sub-register argument widths (an `int` arg only ever touched as `edi`):
  the SSA width machinery can witness the *used* width, but the psABI says
  high bits of argument registers are unspecified — report used-width as a
  hint for the type-recovery topic rather than a claim?
- Tail calls (`jmp` to another function): the callee's arguments are the
  caller's live argument registers at the jump — `cfg` already detects
  import tail-call thunks; does slice 3 need general tail-call awareness
  from day one, or is thunk-only acceptable initially?
- Go binaries pre-1.17 use a stack-only internal ABI (openly documented in
  the Go ABI spec); `gopcln` already detects Go — should ABI selection key
  on that and simply mark such functions `NonStandardAbi` rather than
  mis-infer?
