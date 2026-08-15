# Aletheia Roadmap

This document is the project's working plan: the architecture we're building
toward, the phases to get there, and the reasoning behind the order. It is a
living document — phases get refined as earlier ones land.

## Positioning

General-purpose binary analysis tools (IDA Pro, Ghidra, Binary Ninja) derive
their value from **breadth** (many formats × many architectures) multiplied by
**depth** (analysis quality on each). A young project cannot win on breadth,
so Aletheia's strategy is:

- **Depth-first on the big three**: PE + ELF + Mach-O, x86-64 + AArch64.
  That covers Windows, Linux, Android, macOS, and iOS binaries — the
  overwhelming majority of real-world analysis work.
- **Structural advantages the incumbents can't easily retrofit.** The most
  common practitioner complaints about the market leader (beyond price) are
  single-threaded analysis, a proprietary version-locked database, API churn,
  awkward headless use, and weak Go/Rust binary support. Every one of those
  is an architecture decision, and greenfield projects get to make them
  correctly from the start. See [Design principles](README.md#design-principles).

## Architecture

```
            ┌────────────────────────────────────────────────┐
            │                   clients                      │
            │   redump CLI · future TUI · scripts · CI       │
            └───────────────────────┬────────────────────────┘
                                    │  stable library API
   ┌────────────────────────────────┼─────────────────────────────────┐
   │                            analysis                              │
   │   CFG recovery · function discovery · xrefs · strings · types    │
   │            (parallel passes over independent functions)          │
   ├──────────────────────────────────────────────────────────────────┤
   │                        program model (traits)                    │
   │   Image: sections, symbols, entry points, addr↔offset mapping    │
   │   Decoder: bytes → Instruction (+ Flow classification)           │
   │   Database: names, comments, types — open, diffable format       │
   ├────────────────────────────┬─────────────────────────────────────┤
   │          loaders           │              decoders               │
   │   pe  ·  elf  ·  macho     │        x86-64  ·  aarch64           │
   ├────────────────────────────┴─────────────────────────────────────┤
   │                          foundation                              │
   │        reader (bounds-checked) · error (typed, no panics)        │
   └──────────────────────────────────────────────────────────────────┘
```

The load-bearing decision: **analysis is written once against the trait
layer**, never against a specific format or ISA. Adding MIPS or RISC-V later
means writing a decoder, not touching the analysis engine.

## Phases

### Phase 0 — Foundations ✅ / 🚧 (current)

- [x] Typed error model; bounds-checked little-endian reader
- [x] PE/COFF loader: DOS/COFF/optional headers (PE32 & PE32+), data
      directories, section table, RVA→offset mapping
- [x] PE import table: hint/name & ordinal imports, IAT slot RVAs,
      null-ILT fallback, adversarial-input caps
- [x] `redump` CLI (dumpbin-style; validated against real binaries)
- [x] ELF64 loader: headers, program/section headers, symbols,
      vaddr→offset mapping
- [x] x86-64 decoder: prefixes/REX/ModRM/SIB/RIP-relative, common one-byte +
      0F opcode maps, `Flow` classification for control-flow recovery,
      15-byte-limit fuzz hygiene

**Exit criteria:** parse-and-dump works on real PE and ELF binaries; the
decoder round-trips a hand-assembled test corpus and never panics on any
byte sequence.

### Phase 1 — Breadth of the big three

- [x] **Mach-O loader** — headers, load commands, segments/sections, symtab,
      fat/universal binaries. Unlocks the entire local macOS system as a
      test corpus.
- [x] **PE export table** + delay-load imports; ELF dynamic section &
      relocations (the subset needed to resolve calls through the PLT/GOT).
- [x] **AArch64 decoder** — fixed-width 4-byte instructions from the public
      Arm ARM encodings; branch/call/return classification first, full
      operand decoding second.
- [x] **The trait layer** — `Image`, `Decoder`, `Flow` formalized once three
      loaders and two decoders exist to generalize over (rule: no trait
      before the second concrete implementation).

**Exit criteria:** one analysis-facing API loads any of the five
format/arch combinations; `redump` grows `--macho`/`--elf` parity.

### Phase 2 — The analysis core

This is where a loader collection becomes an *analysis tool*.

- [x] **Control-flow recovery**: recursive descent from entry points,
      exports, and symbol tables; basic-block construction; call-graph
      edges through direct calls and IAT/PLT-resolved imports.
- [x] **Function discovery** beyond entry points: prologue heuristics,
      exception/unwind info (PE `.pdata`, ELF `.eh_frame`, Mach-O
      `LC_FUNCTION_STARTS`) — high-precision sources first, heuristics last.
- [x] **Parallel pass engine**: functions analyzed as independent work units
      across all cores. Determinism requirement: same binary in, same
      database out, regardless of thread count.
- [x] **Switch-table recovery** (jump tables) — a classic brittleness point
      in every tool; needs per-compiler pattern knowledge.
- [x] **Cross-references & strings**: code→code, code→data xrefs; string
      extraction with encoding detection.

**Exit criteria:** for a corpus of known binaries, function boundaries and
CFGs validated against compiler-emitted symbol/unwind ground truth;
analysis throughput benchmarked and scaling with cores.

### Phase 3 — The database ✅

The annotation layer — where an analyst's hours of work live, and where the
incumbent's proprietary format is weakest. Implemented in [`anchor`] (the
relocation-independent identity) and [`annotate`] (the store and format).

- [x] Open, documented, versioned format for analysis results and user
      annotations (names, comments, types, function overrides) —
      `aletheia-annotations v1`, see [`annotate::Db::serialize`].
- [x] **Git-mergeable by design**: deterministic ordering, stable
      addressing (content-relative anchors so annotations survive rebasing
      the binary — [`anchor::Anchor`]'s rebase-invariant `shape`),
      line-oriented so diffs are reviewable.
- [x] Separation of *computed* facts (reproducible from the binary) from
      *asserted* facts (human annotations) — only asserted facts are
      stored; computed facts are regenerated and never in the file.
- [x] Undo/redo as a first-class operation on the assertion log
      ([`annotate::Db::undo`] / [`annotate::Db::redo`]).

**Exit criteria:** two people can annotate the same binary on branches and
merge with git alone. ✅ — proven by
`annotate::tests::two_branches_merge_by_line_union` and
`concurrent_same_field_edits_resolve_deterministically`.

### Phase 4 — Interface & scripting 🚧

- [x] Disassembly listing renderer (symbolized operands, xref annotations)
      — the `objdump -d`-but-good experience, in the CLI first.
      [`listing::render`] + `redump --listing`/`--db`; x86-64 and AArch64
      instruction text in [`x86_text`] / [`aarch64_text`] behind the
      [`asmtext::AsmFormatter`] seam. Validated end-to-end on real
      binaries (branch/call operands symbolized to callee names).
- [ ] Interactive TUI explorer (navigate xrefs, rename, comment).
- [ ] Scripting API with semver discipline — one coherent surface, no
      layered legacy APIs. Likely embedded scripting or a stable
      Rust-plugin ABI; decided by real use cases, not upfront.
- [ ] Headless batch mode as a documented, supported workflow (it already
      exists de facto because the library is the product).

### Phase 5 — Differentiators

Where Aletheia stops chasing and starts leading:

- [x] **First-class Go binary support**: `gopclntab`-driven function
      recovery and naming ✅ ([`gopcln`], all four layouts, wired into
      `cfg` seeding+naming — a stripped Go binary recovers its full named
      function set); runtime type metadata ✅ ([`gotype`],
      `redump --gotypes`: moduledata located from the pclntab anchor with
      a typelinks probe disambiguating go1.18/go1.20+ layouts, named
      types with kinds, and itab interface-satisfaction pairs — a
      stripped go1.25 binary yields its full type graph, `gh` yields
      12k types / 1.9k itabs in under 2s); string extraction ✅
      ([`gostrings`], `redump --gostrings`: literals recovered by their
      (pointer, length) references — code-materialized `lea`/`ADRP+ADD`
      + length idioms and data-resident header pairs — with a boundary
      gate that makes the packed blob self-delimiting. On a real stripped
      Go binary, adjacent literals the C-string scanner would merge come
      back distinct with exact boundaries: `"tlsrsakex"`, `" says woof"`,
      `"1220703125"` packed back-to-back, recovered separately).
- [~] **First-class Rust binary support**: symbol demangling ✅
      ([`demangle`], legacy + v0, wired into the listing); panic-metadata
      mining ✅ ([`rustmeta`], `redump --rustmeta`:
      `core::panic::Location` records mined from data sections and
      attributed to functions via xrefs — source-file/line hints that
      survive stripping, validated on a real binary down to the exact
      line/column of an `unwrap`); trait-object call resolution still to
      come.
- [x] **C++ structure recovery**: Itanium `_Z` symbol demangling ✅
      ([`cxxdemangle`], names/types/substitutions/operators/special names,
      verified against `c++filt` on real libstdc++ symbols, wired into the
      listing after Rust gets first refusal); vtables, RTTI, and the class
      hierarchy ✅ ([`vtable`], `redump --vtables`: typeinfo parsing with
      `__si`/`__vmi` base edges, multiple-inheritance sub-vtable
      splitting, virtual-base prefixes, symbol-driven plus a
      cross-validated scan tier for stripped binaries; validated on a
      real clang++ MI + virtual-inheritance hierarchy. Known limitation:
      unapplied relocations — PIE/chained-fixup images recover less, never
      wrongly). Devirtualization and member-layout inference are future
      work atop the IR.
- [x] **Binary diffing** — matching functions across two versions of a
      binary ([`diff`], `redump <old> --diff <new>`). Nearly free on top
      of the anchor layer: the rebase-invariant shape hash and raw-bytes
      hash already define function identity, so a diff is anchor
      resolution plus bucketing (unchanged / moved / modified / uncertain
      / added / removed), with names carried across versions and no
      identity heuristics of its own. Validated on two real clang++
      builds: a uniform relink reports as moves, a changed call
      displacement as modified, and a reshaped `main` honestly lands in
      uncertain rather than being guessed.
- [x] **Virtual-call resolution** — turning an indirect `call [reg+off]`
      into candidate targets ([`devirt`], `redump --devirt`). Two tiers on
      the recovered vtables: a sound slot-offset superset always (every
      class's function at that slot), refined to a single exact target
      when the calling function establishes the class in the open (a
      `lea`/`mov` of a known vtable address point). x86-64; unit-validated
      with hand-encoded dispatch sequences (a real-binary e2e wants an
      x86-64 image with recoverable RTTI — the local arm64 toolchain emits
      RTTI, the x86-64 cross-compile did not).
- [~] **Lifting to an IR** — the prerequisite for dataflow analysis and,
      eventually, a decompiler. (A decompiler is a multi-year project on its
      own; the IR is designed so one can grow on top, but decompilation is
      deliberately *not* promised earlier than this.) The IR *core* ✅
      ([`ir`]): a width-explicit register-transfer IR — `Expr` (pure
      values over registers/flags/temporaries, memory loads, typed
      operators), `Stmt` (assign / store / branch / intrinsic), a total
      well-formedness `check` (width agreement, W64 addresses, W1
      conditions, temp write-before-read, size caps) and a deterministic
      `render`. No ISA knowledge lives here. Next: per-ISA lifters onto
      this contract. x86-64 lifter ✅ ([`x86_lift`], `redump --lift`):
      moves, LEA, the ALU ops with correct flag semantics, stack
      (push/pop/leave), calls/jumps/jcc with flag-expression conditions,
      setcc/cmov, precise sub-register write model (32-bit writes
      zero-extend, 8/16-bit merge), unmodeled opcodes lifting to sound
      intrinsics — every output passes `ir::check`, verified by a 2-byte
      exhaustive decode/lift/check sweep. Validated on a real x86-64
      binary: `push rbp` → `rsp -= 8; store [rsp], rbp`, `lea` resolves
      the rip-relative address, `xor eax,eax` shows the zero-extend and
      flag writes. Whole-function lift ✅ ([`irlift`]): each recovered
      function's blocks walked, decoded, and lifted into an IR CFG with
      temporaries threaded per block, honest `truncated` marks where the
      walk stops, resource caps against hostile ranges; `redump --lift`
      renders through it. Dataflow passes ✅ ([`irflow`]): constant
      folding, copy/constant propagation, and liveness-driven dead-code
      elimination — sound by construction (loads never deleted, stores/
      branches/intrinsics never dropped, division-by-zero never folded,
      conservative sub-register aliasing), composed to a bounded fixpoint
      by `simplify`; `irlift::simplify` applies them per block and
      `redump --simplify` shows the cleaned lift (dead flag writes gone
      on real binaries). Still to come: aarch64 lifting (needs decoder
      operand detail first).
- [ ] Signature/FLIRT-style library-function identification, built as an
      open community corpus.

### Phase 6 — The surface: one engine, many frontends

The end-goal interface. Not "a GUI" but a **poly-frontend architecture**:
a headless analysis engine that owns all truth, and interchangeable thin
clients — native app, browser UI, TUI — that render it. The model is the
Language Server Protocol: the engine is to reverse engineering what an
LSP server is to editing, and frontends compete on ergonomics, never on
correctness.

Principles (these are the "better than IDA" thesis, stated precisely):

- **Frontends compute nothing.** Every fact on screen — a boundary, a
  name, an operand, an xref — comes from the engine over a canonical,
  versioned protocol. A renderer that cannot introduce analysis errors
  is a renderer that cannot lie.
- **Provenance on demand.** Any displayed fact can answer *"why do you
  believe this?"* — down to the bytes and the rule that produced it
  (`funcs::Source`, `anchor::Resolution` already carry this). Surgical
  trust means auditable, not just usually-right.
- **Proven vs. heuristic, visibly distinct.** Prologue-scan guesses and
  weakly-reattached annotations render differently from metadata-backed
  facts. No silent guessing anywhere in the stack.
- **Deterministic and incremental.** Same binary, same engine version →
  the same analysis, byte-for-byte, at any thread count (the parallel
  engine already guarantees this). Re-analysis after an edit is
  incremental; the UI never blocks on a full re-run.
- **The database stays open.** Whatever the frontend, the artifact an
  analyst produces is the Phase 3 git-mergeable format — no UI-private
  state that locks work into one client.
- **Core stays zero-dep.** The engine and protocol live in the
  dependency-free crate; GUI toolkits (native or web) live in separate
  frontend crates that depend on it, never the reverse.

Sequencing: protocol first (defined by what the Phase 4 TUI actually
needs — real use cases, not upfront design), then the first graphical
client on top of it, then parity across clients as a conformance suite
over the protocol.

**Exit criteria:** two different frontends drive the same engine over the
same protocol with zero analysis logic of their own, and an analyst can
switch between them mid-session with nothing lost.

## Quality bar (applies to every phase)

- **No panics on any input, ever.** Truncation sweeps and adversarial-input
  tests accompany every parser; fuzzing (cargo-fuzz) joins CI once the
  surface stabilizes.
- **Resource caps on attacker-controlled counts** (section counts, import
  tables, string lengths) — a crafted 4 KB file must not allocate 4 GB.
- **Synthetic test images in-memory** (no binary fixtures in git) plus
  smoke tests against real system binaries where available.
- **Ground-truth validation**: parsers checked against `dumpbin`/`readelf`/
  `otool` output; decoders against assembler output.
- **Benchmarks are features**: analysis speed is a headline goal and gets
  measured, not assumed.
- **Clean-room discipline** throughout — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Working mode & session continuity

Development runs as roadmap-driven sessions (human + agent). Sessions can
be cut off at any moment — API interruptions, model switches, crashes —
and the protocol below exists so a cutoff costs **zero work**. The repo is
the only source of truth; nothing important may live solely in a
session's conversation memory.

- **One slice, one commit.** Every work item is a focused slice that
  lands as a single reviewed commit (see the git log for the format:
  `module: what it does`, lowercase prefix).
- **Plan before build, on disk.** A slice's design is written to a
  `PLAN_*.md` in the repo *before* implementation starts. If the session
  dies mid-build, the next session re-reads the plan and re-dispatches.
- **Build in worktrees, commit early on the branch.** Implementation
  happens on a worktree branch and is committed there before review —
  never held only in an agent's context. Merge to master only after
  review.
- **One focused agent per stage, never a broad fan-out.** Empirically: a
  single plan agent followed by a single builder agent ships slices
  (x86 lifter, aarch64 lifter). An 8-agent parallel research fan-out
  burned ~500k tokens on 2026-08-15 and produced nothing recoverable.
  Retry a failed slice by re-dispatching the *same focused* agent — never
  by widening the fan-out.
- **Resume protocol** (run at the start of every session):
  1. Read this section's *Current thread* block below.
  2. `git worktree list` + `git branch -a` — find in-flight branches.
  3. Read any `PLAN_*.md` at the repo root.
  4. Continue or re-dispatch the active slice. A transient interruption
     is a retry, not a restart: the plan and any branch commits survive.

### Current thread

- aarch64 lifter ✅ — landed as `d5f2847` (`aarch64_lift`).
- decompiler slice 1 ✅ — pruned SSA over the lifted IR CFG, landed as
  `6d2e3fc` (`irssa`, plan in `PLAN_SSA.md`): CHK dominators +
  frontiers, `live_in`-pruned Cytron phi placement, dominator-tree
  renaming that rewrites only `Reg::num` so every block still passes
  `ir::check`, plus an `irssa::check` (one def per name, dominance of
  every use, canonical phis) and `redump --ssa`. 860 tests, clippy
  clean. Honest limits recorded in the module: per-cell versions,
  wider-than-def reads listed in `partial`, call clobbers unmodeled.
- decompiler slice 2 ✅ — ABI-aware call effects (`callfx`, plan in
  `PLAN_CALLFX.md`): a per-arch `CallAbi` (x86-64 as the SysV/Win64
  union, aarch64 as AAPCS64) and `callfx::apply`, which inserts one
  `callfx` intrinsic — clobber writes, argument-superset reads — plus
  the x86-64 rsp restore after every `Branch::Call`. The effects are
  ordinary IR statements, so `ir`, `irflow`, and `irssa` needed zero
  algorithm changes; wired into `redump --ssa` only, `--lift` /
  `--simplify` stay faithful. 885 tests, clippy clean; verified on the
  /bin/ls x86-64 slice (369 call sites, zero SSA refusals,
  byte-deterministic). The aarch64 table shipped ready-made
  (cross-checked against `aarch64_lift::reg_name`), but the
  register-naming/dispatch rider stays queued: `irlift` still
  dispatches only x86-64, so no aarch64 `LiftedFunction` — and hence no
  aarch64 SSA render — can exist yet; the namer hookup belongs to the
  irlift-aarch64-dispatch slice.
- decompiler slice 3 ✅ — sparse constant/copy propagation and
  phi-simplification on SSA (`irssaopt`, plan in `PLAN_SSAOPT.md`): a
  def-use index built from `irssa`'s occurrence walks, the three-level
  lattice solved optimistically by a deterministic worklist, rewriting
  that substitutes proven constants and copy roots into every use, and
  removal of the phis that merge one value (through copy chains and
  cascaded through phi-of-phi), with a name-table compaction. Folding
  is `irflow::fold_expr`/`fold_stmt`, shared not duplicated, so the
  divide-by-zero and load doctrine carries over verbatim; intrinsic
  reads are deliberately never rewritten (the callfx register identity
  is what keeps slice 4 sound); no CFG mutation, no statement deleted
  or reordered. `redump --ssa-opt[=N]` prints the cleaned view while
  `--ssa` stays the faithful one. 920 tests, clippy clean; verified on
  the /bin/ls and /bin/bash x86-64 slices (1,904 functions, zero check
  or preservation failures, zero caps, byte-deterministic, idempotent).
- decompiler slice 4 ✅ — conservative dead-code elimination on SSA
  (`irssaopt::eliminate_dead`, plan in `PLAN_DCE.md`): mark-and-sweep
  liveness over slice 3's def-use index, rooted at the reads of every
  statement the pass never deletes (stores, branches, `callfx`
  intrinsics — which is what pins argument setups — and load-bearing
  assigns) plus the function live-out. Live-out roots are *cells*, not
  reaching versions: pruned SSA does not materialize the version at a
  return, so every definition of a live-out cell is a root, with the
  precision limit (a genuinely dead def of a live-out cell survives)
  documented in the module and owned by the signature slices. The tables
  are `callfx::function_live_out` per arch. A `Load` is never deleted
  (`irflow::contains_load`, shared not duplicated; the kept ones are
  counted in `DceStats::kept_loads`), the CFG is untouched, and the
  companion `check_swept` proves each output is the input minus
  justified deletions. `redump --ssa-opt` now runs optimize → sweep and
  prints `; dce: removed N of M statements`. 943 tests, clippy clean;
  verified on the /bin/ls x86-64 slice: 125 functions, 3,880 of 12,679
  statements removed (30.6%), zero `irssa::check` or `check_swept`
  failures, byte-deterministic and idempotent — a `cmp`'s four flag
  writes reduce to the one the `jcc` reads.
- decompiler slice 5 ✅ — expression forwarding on SSA
  (`irssaopt::forward`, plan in `PLAN_FWD.md`): each definition's whole
  right-hand side is substituted into its uses, rebuilding source-level
  expressions out of the lift's one-operation-per-statement form. The
  def is left standing — forwarding never adds, deletes, or reorders a
  statement and never touches the CFG, so slice 3's `check_preserved`
  applies verbatim and the existing DCE sweeps what the pass emptied.
  Tiers per DESIGN: trivial right-hand sides stay `optimize`'s job;
  compound pure load-free division-free trees go to every use under
  `FWD_SMALL_NODES = 8` nodes and single-use above it (DREAM++'s
  duplication finding); a load-bearing tree is single-use, same-block,
  barrier-free (`irflow::propagate`'s `Store`/`Intrinsic`/`Branch` set
  lifted onto SSA names); a division never crosses a branch. Exact-width
  reads only, never into an intrinsic read or a φ argument, and a tree
  reading a temporary only lands where that temporary is written.
  Substitutions cascade over rounds to a fixpoint (`MAX_ROUNDS = 8`;
  on the cap the last completed round is returned — every forwarding
  state is sound), and one past `ir::MAX_EXPR_NODES` is refused whole
  and counted, never truncated. The other half is in `irflow::fold_expr`
  (shared, so `--simplify` benefits): `(a - b) == 0 → a == b`,
  `(a ^ b) != 0 → a != b`, `~(a == b) → a != b`, `~~x → x` at W1 — the
  equality family, which is exactly the `je`/`jne` plumbing. The signed
  and unsigned *order* conditions (`SF ^ OF`, `CF | ZF`) need the flag
  *pair* recognized together, not a single-operator rewrite; that is
  recorded in the module docs as input for a later condition-recovery
  slice rather than invented here. `redump --ssa-opt` now runs
  optimize → forward → sweep. 974 tests, clippy clean; verified on the
  /bin/ls x86-64 slice: 125 functions, 14,600 forwards, zero
  `irssa::check` / `check_preserved` / `check_swept` failures, zero
  caps, zero size refusals, byte-deterministic and idempotent. The
  exit criterion holds end to end — `cmp eax, -1` + `je` in
  `sub_100000978` rendered `t2.d#15 := (t0.d#15 - 0xffffffff.d);
  ZF#34 := (t2.d#15 == 0x0.d); goto if ZF#34` before and
  `goto if (trunc.d(rax#21) == 0xffffffff.d)` after, with the `jne`
  polarity in the successor block collapsing to `!=` across the block
  boundary. Flag-read branches fell from 340 to 44 (the survivors are
  `callfx` intrinsic writes, which have no right-hand side), 380
  branches now render a relational condition, and the sweep behind it
  now removes 5,689 of 12,679 statements (44.9%, up from 30.6%).
- decompiler slice 6 ✅ — Phoenix-style control-flow structuring
  (`irstruct`, plan in `PLAN_STRUCT.md`): the SSA CFG becomes a `Node`
  tree — `Block`/`Seq`/`If`/`Loop`/`Switch`/`Break`/`Continue`/`Goto`/
  `Opaque` — by iterative region collapse (Schwartz et al., USENIX
  Security 2013). Conditions are *references*, never expressions:
  `Cond { block, negated }` names the block whose final conditional
  `Branch` decides, with polarity defined as "un-negated means control
  goes to the branch's own target", so the same tree is valid over the
  faithful and the optimized SSA and the renderer fetches the expression
  at print time. Schema catalog in the planned order (sequence,
  if-then/if-then-else, self-loop, natural while/do-while, proven-table
  switch), with Phoenix's cyclic/acyclic split applied *graph-wide*:
  every loop header is offered the loop schemas before any head is
  offered an acyclic one, or an if-then absorbs the loop's own exit
  block and the loop degrades to `while (true)`. The follow node is the
  header's immediate post-dominator when one exists outside the body
  (post-dominators are `irssa`'s Cooper–Harvey–Kennedy pass over the
  reverse graph with a virtual exit — the `Cfg` internals were exposed
  `pub(crate)` rather than copied), else the most frequent exit target,
  ties by lowest address; any exit edge the follow does not cover
  refuses the match. Loop bodies are collapsed by recursion with their
  back and exit edges cut to gotos, which `tighten` turns into
  `Continue`/`Break` at that loop's own level only — a goto that lands
  inside a *nested* loop stays a goto, because `continue` there would
  mean the inner header. Per SAILR (Basque et al., USENIX Security
  2024) nothing is ever duplicated and no condition is ever
  synthesized: when no schema matches, exactly one edge — lowest
  `(source region head VA, target VA)` — is virtualized into an
  explicit `Goto` and the round retries, so termination is structural
  and the round budget is only defense in depth (on the cap the
  remainder degrades to gotos, never a refusal). Truncated blocks,
  indirect jumps no table proved, and blocks with two or more
  successors and no conditional to decide between them are
  `Opaque { reason }` — held, never absorbed, and an undecidable
  block's edges are declared unrealized rather than invented.
  Out-of-function tail jumps are external `Goto`s and are not counted
  as structuring edges; malformed input (fails `irssa::check`) gets the
  degenerate every-block-plus-gotos tree with zeroed stats. The
  companion `irstruct::check` is what the tests trust: exact partition
  of the reachable blocks, every CFG edge realized exactly as
  fall-through / construct edge / goto with none invented, condition
  and switch honesty, opacity honesty. `redump --structure[=N]` runs
  the `--ssa-opt` pipeline then structures and renders, with
  `; structure: N gotos`. 1,004 tests (+30), clippy clean, no new
  rustdoc warnings; verified on the /bin/ls and /bin/bash x86-64 slices:
  1,904 functions, **zero `check` failures**, zero caps, byte-identical
  across two runs (ls dump `c5d6cac6…`). /bin/ls: 125 functions, 285
  gotos concentrated in 12 of them — the ones carrying an irreducible
  or many-exit cycle (12 of the 125 are irreducible by the
  dominance test) — so 113 functions structure goto-free. Real evidence, all
  three shapes from the exit criterion: a loop —
  `sub_10000255f` renders `do-while cond loc_100002be1` over a
  six-block body with two nested `if`s (`if !cond loc_100002b9d`
  around `if !cond loc_100002ba1`); a diamond —
  `sub_100002439` renders `if cond loc_100002439 / block loc_100002475
  / else / block loc_10000245d`; and an irreducible region —
  `sub_100002398` (14 blocks, two-entry cycle) structures with 10
  deterministic gotos plus a recovered `while true` over
  `loc_10000240a`. `Switch` is exercised only by synthetic tests today:
  `cfg::recover` does not yet fold `jumptable::successor_map` into
  block successors, so real dispatch blocks render
  `opaque … (indirect jump)` (106 opaque leaves in ls) — the CFG
  folding rider belongs to a later slice, and the structurer already
  takes the map as a parameter so nothing here changes then.
- decompiler slice 7 ✅ — out-of-SSA into named variables (`irout`, plan
  in `PLAN_OUT.md`): Boissinot et al. (CGO 2009) as a **map, not a
  rewrite** — `out_of_ssa` returns `var_of` (name → variable), the
  residual per-edge copies, and the provenance sets, and never touches
  the `SsaFunction`, which stays the analysis truth. Correct-then-good
  kept visible: singleton congruence classes isolate every φ (nothing
  can be clobbered), then a φ's class merges with an argument's whenever
  no pair *value-interferes* — dominance-based live-range intersection
  over SSA-name liveness (φ arguments live out of their predecessor) AND
  different values, value equality by copy-chasing width-exact
  `Assign { value: Reg }` chains — then each edge's surviving parallel
  copies are sequentialized with Boissinot's Algorithm 1, breaking a
  cycle with the one temporary (the Briggs swap). Merge order is total
  (φ block VA, then cell, then argument order) and variables are
  numbered by ascending first member, so the output is byte-reproducible.
  Two facts about this IR are recorded in the module rather than left
  for a reader to rediscover: a congruence class never mixes cells (a φ
  merges one cell), so a variable names one machine cell and two
  parameters can never share storage — and so a φ-only coalescer cannot
  produce a permutation on an edge, which is why the swap machinery is
  proved against `sequentialize` and its simulator directly; and the
  **lost copy** is only reachable downstream of a value-moving pass —
  plain renaming can never read a φ's result past a redefinition of its
  cell — so the fixture builds it through the real `optimize` +
  `forward`, exactly as the published example needs copy propagation.
  Provenance carriers, after the inventory the plan asked for: `assumed`
  is a name defined by a `callfx` intrinsic write (slice 2 shipped
  `AbiAssumed` as ordinary IR, not a tag — the better carrier, since it
  is the same fact every other pass reasoned about) and `partial` is a
  name read wider than its definition at an `SsaFunction::partial`
  position. `check` recomputes interference, φ resolution, sequence
  validity (by simulating each edge's list, including that it disturbs
  no other variable), density and provenance from scratch; malformed
  input gets the identity map. The real oracle is a test-only SSA
  interpreter that evaluates the SSA function and the
  (variables + edge copies) rendition side by side on seeded inputs with
  independent memories and havocked intrinsics — it catches unsound
  coalescing that `check`, which shares the interference test, provably
  cannot (verified by mutation: forcing `interferes` to `false` leaves
  `check` happy and the interpreter screaming). 994 tests, clippy clean;
  verified on the /bin/ls x86-64 slice: 125 functions, 11,673 names and
  1,776 φs over 4,270 arguments → 8,212 variables (3,461 names
  coalesced away, 29.6%) with only 53 residual copies — 112 of 125
  functions are copy-free, and all 104 straight-line/one-diamond
  functions leave zero residual copies (the DESIGN exit criterion),
  zero `check` failures, byte-deterministic. No renderer and no
  `redump` flag this slice: the consumer is slice 8. Left on the table
  and documented: ordinary copy coalescing (a definition no φ mentions
  keeps its own variable), which is also what would make a variable span
  two cells and the swap case reachable.
- aarch64 `irlift` dispatch ✅ (plan in `PLAN_A64LIFT.md`):
  `lift_function` now dispatches `aarch64_lift::lift_block` (fixed
  4-byte walk mirroring the x86 path's truncation honesty), the arch is
  carried as pure data on `LiftedFunction`/`SsaFunction`, and the
  `irlift`/`irssa` renderers spell `Space::Arch` cells per-arch
  (`x0`…`x30`/`sp` vs `rax`…), with `redump`'s IR-view gates admitting
  `Aarch64`. Downstream passes needed zero algorithm changes — proven,
  not assumed: a real arm64 dylib (libbrotlidec, 62 functions, 12k
  words) runs construct → optimize → forward → DCE → structure →
  out-of-SSA with zero check failures, zero refusals, byte-deterministic;
  x86-64 output is byte-identical to master on the real /bin/ls slice
  across all four IR views. This unblocks arm64/iOS end-to-end; the
  aarch64 pseudocode e2e is now a verification rider on `pseudo`.
  Honest ceiling, measured: 12.9% of the dylib's `__text` words still
  lift to the `a64.unknown` clobber intrinsic — top mnemonics mov/add/
  cmp/lsl/lsr/sub/and/orr/bic (register-operand ALU, shifts, bitfields),
  exactly `PLAN_A64DEC.md`'s territory — and on arm64e (/bin/ls) the
  PAC-heavy unknowns inflate SSA name pressure until 126 of 131
  functions refuse with the explicit `TooManyNames` (never a panic or a
  check failure; the decoder slice is what retires this).
- aarch64 decoder coverage ✅ — integer data-processing (`aarch64` +
  `aarch64_lift`, plan in `PLAN_A64DEC.md`): shifted-register add/sub
  and the eight-member logical group, extended-register add/sub (the
  sp-legal form), logical (bitmask) immediates with `decode_bit_masks`
  proved against a bit-by-bit reference over the exhaustive 13-bit
  space (11,328 valid triples, every reserved one refused), the
  bitfield trio `SBFM`/`BFM`/`UBFM` decoded canonically with the full
  preferred-alias spelling at render (`lsl`/`lsr`/`asr`, `ubfx`/`sbfx`/
  `*bfiz`, `uxtb/h`, `sxtb/h/w`, `bfi`/`bfxil`/`bfc`), the two-source
  shifts and divides, and the three-source multiplies including the
  widening `L` forms; `SMULH`/`UMULH` decode but lift to a precise
  named intrinsic (no 128-bit multiply in the IR). Every golden word
  cross-checked against the system assembler. The lift reuses the one
  add/sub NZCV model (proved textually identical between the immediate
  and register forms), `ANDS` clears C/V, and `UDIV`/`SDIV` carry the
  architectural zero-divisor rule structurally — divisor forced to 1
  and result masked to 0 on `rm == 0`, so no evaluator ever divides by
  zero. Two forced ripples, both behavior-preserving: `jumptable` reads
  the dispatch `ADD` from the decoded forms (its raw-word re-parser
  deleted), and `gostrings` reads Go's `ORR Xd, XZR, #len` from
  `LogImm` (its re-parser kept as the tests' oracle). 1,047 tests
  (+23), clippy clean. Measured on real Mach-Os: /bin/ls arm64e slice
  83.35% → 94.40% decoded (636 → 214 unknown of 3,819 words), ffmpeg
  arm64 82.81% → 96.89% (8,165 → 1,477 of 47,510). The remaining
  ceiling, honestly: SIMD/FP and their loads/stores dominate what is
  left, then `CCMP`/`CCMN`, `LDUR`/`STUR`, exclusives/atomics, and PAC
  (`retab`, the one-source auth group) on arm64e binaries — all
  declared non-goals of this slice.
- decompiler slice 8 ✅ — **the milestone: first end-to-end pseudocode**
  (`pseudo` + `redump --decompile[=N]`, plan in `PLAN_PSEUDO.md`). The
  wave-3 join point: the `irstruct` tree and the `irout` variable map
  become deterministic C-like text. Arch-agnostic by construction —
  every cell prints as `vN` through `OutOfSsa::var_of` behind a
  `VarNamer` hook (default `v{id}`), so the renderer holds no ISA
  register table and aarch64 rides the dispatch slice for free.
  Precedence-aware parenthesization replaces `render_with`'s
  every-node parens: a C-modeled precedence/associativity table,
  parens iff a child binds looser or equally on the right (Ghidra
  PrintC rule), plus DREAM++'s *forced redundant parens* wherever
  bitwise/shift/comparison operators mix (Yakdan et al., NDSS 2015 /
  IEEE S&P 2016), same-op `&`/`|`/`^` chains exempt. The
  non-ambiguity claim is proved, not asserted: a test-only precedence
  reparser round-trips 500 random trees back to the exact tree —
  the slice's real oracle, since a golden file cannot show an output
  is unambiguous. Conditions are fetched from the deciding block at
  print time and negated per `Cond::negated` (`!` + operand);
  signedness stays on the operator (`/u`, `<s`), widths explicit
  (`0x7.d`), casts functional (`zext.q(x)`), loads/stores
  `*(uN*)addr`. Edge copies render at the realized edge, tracked by a
  verifier-style pending-set walk: end of an unconditionally-leaving
  block, before the `goto`/`break`/`continue` that takes the edge,
  before a leaf's label (so a goto into the label skips them), before
  a loop for its entry edge, and in a synthesized `else { }` for an
  if-then's untaken side. The honest limit is written down and
  regression-tested: a `do`-`while` back edge (conditional source)
  has no C-syntax site without synthesizing a condition (banned), so
  its copies emit an `/* unplaced edge copies ... */` marker — never
  dropped, never hoisted into a predecessor where they could clobber
  a sibling-live variable (slice 7's written obligation). Honesty
  markers all reach the text: `/* lift truncated */`,
  `/* indirect jump: successors unknown */`, `/* undecidable exits */`,
  and per-function `/* abi-assumed: ... */` / `/* reads bits its def
  never wrote: ... */` preambles over exactly the variables the text
  mentions. Every statement carries its block VA as a right-margin
  `// 0x…` anchor (block-granular — the lift carries no per-stmt VAs
  yet), the hook for a recompile-differential oracle (Liu & Wang,
  ISSTA 2020). A `switch` case converging on the follow prints the
  `break;` that spells its recorded exit edge; a `while` header's
  leftover non-branch statements print at the body top (reference-typed
  conditions; usually invisible after forwarding — recorded, not
  papered over). Total on hand-broken trees (missing blocks,
  non-conditional `If`s, 500-deep nesting, 200-tree fuzz) and on
  malformed functions via the identity posture, `v?N` fallback
  included. 1,062 tests (+38), clippy clean, no new rustdoc warnings.
  Verified on the /bin/ls and /bin/bash x86-64 slices: 1,904
  functions, zero panics, byte-identical across two runs (ls dump
  `ae701c52…`). ls: 125 functions, 409 `if`s, 55 loops, 287 gotos —
  every rendered `goto loc_x` has its label — and 52 of slice 7's 53
  residual copies placed at their realized edges, the 1 conditional
  back-edge copy reported unplaced. The slice-5 forwarded `-1` guard
  appears as a real relational `if`: `sub_100000978` (the getopt
  loop) reads `if (trunc.d(v206) == 0xffffffff.d) { break; }`, and
  `sub_10000073f` — a two-field signed comparator — was read and
  judged end to end (nested `if`s over forwarded relations, the
  documented un-recovered `SF^OF` order-condition shape visible and
  honest). Left on the table, recorded: per-stmt VA provenance,
  condition simplification (an `irssaopt` job, never the printer's),
  and argument/type recovery (`irstack`/`irtype`); the aarch64 e2e
  rider was cashed in at merge (see Active below).
- CFG jump-table folding ✅ — proven dispatch edges reach the CFG
  (`cfg::recover_with_tables` + `jumptable::resolve_folded`, plan in
  `PLAN_CFGFOLD.md`): recovery accepts the proven `jump_site ->
  targets` map as plain data and folds it at the indirect-jump arm —
  successors gain the proven targets (deduplicated, ascending, kept to
  executable regions with the drops counted in
  `Stats::table_targets_dropped`, never silent) while the terminator
  stays `IndirectJump`, and the case bodies are walked into blocks by
  the ordinary leader machinery (splits, callee seeding, the
  address-taken fixpoint all apply). Because case bodies are code only
  the folded edges reach and can hold further tables, the entry point
  is `jumptable::resolve_folded`: recover → resolve rounds to the
  joint map fixpoint, each round a from-scratch recovery so the result
  is a pure function of the image, capped by
  `Config::max_fold_rounds` (default 8, `Folded::capped` visible when
  hit). Plain `recover` is behavior-identical (empty map), the
  parallel engine is untouched, and `redump`'s `--lift`/`--ssa`/
  `--ssa-opt`/`--structure`/`--decompile` all recover through the
  folding entry point. 1,110 tests (+12), clippy clean; verified on
  real binaries with zero `irssa`/`irstruct`/`irout` check failures
  and byte-identical double runs: /bin/bash x86-64 proves 37 tables in
  2 rounds → 58 dispatch blocks folded, `opaque (indirect jump)` 486 →
  433, 58 real `switch` renders (e.g. `sub_100014e14` decompiles
  `switch (v16 + 0x100014fbc.q)` over 11 labeled cases); arm64
  libbrotlidec proves 5 tables → 7 folds, opaque 18 → 13, 7 switches
  through the identical code path. Honest notes: /bin/ls x86-64 has
  no idiom-matching table (0 folds, output byte-identical to master);
  every real switch's case *bodies* still render as gotos — shared
  case tails and guard-to-default edges fail `try_switch`'s
  single-predecessor/convergence test, SAILR de-opt territory — and
  goto counts rise with the newly reached code (bash 15,391 → 17,326,
  brotli 237 → 842): real, previously invisible control flow, not a
  regression.
- condition recovery ✅ — the signed/unsigned order jcc flag pairs
  (`irflow`, plan in `PLAN_CONDREC.md`): the paired `SF^OF` / `CF|ZF`
  shapes collapse to single relational operators inside `fold_expr`'s
  always-sound identity family — `SF != OF → a <s b`, `SF == OF →
  b <=s a` (the pair recognized together; shared operands structurally
  equal and width-exact on every occurrence, or no rewrite), the
  `jle`/`jg` compositions `(a == b) | (a <s b) → a <=s b` and
  `(a != b) & (b <=s a) → b <s a` with their unsigned `jbe`/`ja`
  twins, `~(a <s b) → b <=s a` for all four order operators, and the
  `W1` boolean-constant finishes (`x != 0 → x`, …) that complete a
  pattern whose other flag already folded to a constant (`cmp a, 0`
  zeroes `OF` outright). No new IR operator, no `irssaopt` algorithm
  change (forwarded conditions simply fold further), and the
  load-duplication doctrine carries over: shared operands must be
  load-free. Proved, not argued: a width-8 exhaustive oracle folds
  every condition composition of both lifts (je/jne/jl/jge/jle/jg/
  jb/jae/jbe/ja and A64's hs/lo/hi/ls), in both polarities, and
  checks the folded relational op against the literal flag computation
  on all 65,536 operand pairs through a test-only evaluator; near-miss,
  mixed-width, foreign-subtraction, add-overflow, and load-bearing
  negatives stay byte-identical. 1,111 tests (+13), clippy clean.
  Measured on the real `--decompile` dumps (all byte-deterministic):
  flag-pair conditions ls 43 → 20, bash 1,841 → 362, and on arm64
  libbrotlidec 54 → 36 — which also answers the aarch64 question: no
  companion pattern family is needed (same N/V model, NOT-borrow C
  covered by the same compositions). Honest limit, diagnosed and
  recorded rather than papered over: the milestone comparator
  `sub_10000073f` still shows its pairs, because its one `cmp` feeds
  *two* jccs — the 11-node `OF` tree then has two uses, so `forward`'s
  DREAM++ size tier (`FWD_SMALL_NODES`) refuses to splice it, and the
  second jcc's condition could not legally read the first block's
  temps anyway (`ir::check`'s block-local temp rule). Retiring those
  residuals is a forwarding-policy question (count only *eligible*
  uses, or splice when the folded result shrinks), owned by a future
  `irssaopt` slice, plus — on A64 — operand spellings diverging
  through `zext`/`sext` chains at W32, which the structural-equality
  contract correctly refuses.
- decompiler slice 18 (first inversion) ✅ — SAILR tail re-split
  de-optimization (`irstruct`, plan in `PLAN_DEOPT.md`). After an
  uncapped collapse, every in-function `Goto` whose target is a
  copy-safe tail — a plain leaf whose every remaining edge leaves the
  function (shared `ret` or tail-jump epilogue) — is rewritten into a
  duplicate of that leaf: tree shape, not block storage, so both
  occurrences render the one statement list and byte-equality holds by
  construction with `check` pinning the shape. Counted in
  `StructStats::duplications` against `MAX_TAIL_SPLITS` (16);
  all-or-nothing per target (a duplicated tail is never also a goto
  target, so `pseudo` never faces a twice-labeled block); a target
  that does not fit sets `dup_capped` and keeps its gotos; a target
  that is some enclosing loop's recorded exit is left alone (inlining
  it would belie the loop's condition — found by the bash corpus, now
  a fixture). Each replacement buys back exactly one goto, so
  monotonicity is structural, and zero duplications is the collapse
  bit for bit (regression-tested against `structure_raw` over the
  sweep corpus). `check` relaxes exactly as planned: every reachable
  block at least once, extras only as plain leaves of copy-safe tails,
  within the cap. The φ/edge-copy rider is proved in `pseudo`'s tests:
  a duplicated tail realizes a different incoming edge than the
  original, and the pending-set walk places each edge's copies exactly
  once (the organic split-family interference, reached through the
  real passes). `redump --structure` adds `; structure: N
  duplications`. 1,112 tests (+14), clippy clean. Measured on the
  /bin/ls and /bin/bash x86-64 slices through `--structure` and
  `--decompile`: ls 287 → 277 rendered gotos (10 duplications, 9
  functions), bash 15,391 → 14,567 (824 duplications, 370 functions,
  2 now goto-free, 9 hit the cap and degraded honestly), zero check
  failures across all 1,904 functions, zero caps, byte-identical
  across two runs. Exhibit: bash `sub_100022686`'s
  `goto loc_10002269c` — a cross-jump into a label buried two `if`s
  deep — is now the shared return epilogue inline at both sites, its
  `v19 = v48` edge copy still placed exactly once.
- aarch64 coverage two ✅ — CCMP/CCMN, LDUR/STUR, ADC/SBC (`aarch64` +
  `aarch64_lift`, plan in `PLAN_A64CC.md`): the three highest-frequency
  non-SIMD residuals of the first coverage slice. The conditional
  compares (immediate and register, reserved o2/o3/S bits refused),
  the unscaled `LDUR`/`STUR`/`LDURS{B,H,W}` family (decoded beside the
  scaled forms, lifted through the same shared load/store bodies with
  the offset mode — statement lists provably identical where the
  encodings overlap), and `ADC`/`ADCS`/`SBC`/`SBCS` with the `NGC{S}`
  aliases; every golden word assembler-verified. Still one NZCV model:
  the flag expressions now live in `nzcv_model`, extended exactly once
  for the carry-in (C gains the wraparound term `c & (res == lhs)`;
  SBC routes through the add half as `rn + NOT(rm) + C`), and CCMP's
  four flag cells are branchless selects whose true arms are the
  model's products verbatim (asserted structurally in tests, as the
  first slice proved ADDS-imm vs ADDS-reg). A `cmp;b.eq;ccmp;b.lt`
  chained-condition block runs construct → optimize → forward: checks
  green, deterministic, the first condition folds to a relational and
  the ccmp-fed branch keeps its condition-masked `(c & SF) != (c & OF)`
  pair — collapsing it is recorded as `irflow`/forwarding input, not
  patched here. Measured: ffmpeg arm64 96.89% → 97.30% decoded (1,477
  → 1,283 unknown of 47,510), /bin/ls arm64e slice 94.40% → 95.10%
  (214 → 187 of 3,819); the remainder is now SIMD/FP almost outright
  (`movi`/`fmov`/`scvtf`/`fcmp`/... and the q/d-register loads and
  stores), then exclusives/acquire-release (`ldxr`/`ldar`/`stlr`), PAC
  on arm64e (`paciza`/`retab`/`braaz`), `LDPSW`, and the `EXTR`-based
  `ror`-immediate. libbrotlidec `--decompile`: 64 functions, zero
  check failures, byte-deterministic, gotos 807 → 807 and switches
  7 → 7 stable, `a64.unknown` clobbers 8,468 → 8,447 — and the paired
  flag-shape count *rises* (44 → 51 sites) because newly decoded CCMPs
  surface select-masked pairs that previously hid inside the clobber
  intrinsic: real conditions made visible, owned by the
  forwarding-policy/`irflow` work. One forced ripple: `jumptable`'s
  deliberately exhaustive `a64_defs` classifies the new opcodes (loads
  define Rt, ADC/SBC define Rd, CCMP defines nothing). 1,149 tests
  (+12), clippy clean.
- forwarding policy ✅ — splice-when-the-fold-shrinks (`irssaopt`, plan
  in `PLAN_FWDPOLICY.md`): `forward`'s DREAM++ duplication cap gains
  its one measured exception — a pure, load-free, division-free tree
  past `FWD_SMALL_NODES` may forward into *multiple* uses when a
  tentative substitute-then-fold (the very `irflow::fold_stmt` the
  real splice gets) leaves every spliceable use-site statement
  strictly smaller than it stands. Decided per definition from the
  folded results: a site the tentative cannot splice is dropped, one
  that would grow refuses the whole def, and
  `FwdStats::multi_spliced` counts what earned. Loads and division
  stay non-duplicable unconditionally; the cross-block temp cone
  resolves through the existing cascade (temps' pure defs fold in
  first) or is refused — never a cross-block temp read, with
  `irssa::check` the arbiter. 1,148 tests (+11), clippy clean, zero
  existing goldens changed. Measured on the same tree through
  `--decompile`, byte-deterministic, zero check failures: inline
  paired flag conditions fall ls 19 → 0, bash 239 → 90, brotli
  52 → 45 (the `') <s 0x0'` line count: ls 21 → 20, bash
  1,120 → 917, brotli 61 → 50 — bases above 4a9793b's because
  jump-table folding since surfaced new code), and pseudocode bytes
  *shrink*: ls 591,486 → 591,092, bash 23,103,247 → 23,076,717,
  brotli 3,565,947 → 3,564,604. The milestone comparator
  `sub_10000073f` now opens `if (!(v8 <s v10))` /
  `if (!(v14 <s v15))` where the compositions stood; its inner
  `if (v11 == v12)` stays, diagnosed to the root: the flag trees read
  the `cmp`'s memory operand — a load-backed block-local temp — so
  the pair can never legally sit in the second jcc's block
  (`ir::check`'s temp rule) and the load must not duplicate; the
  remaining bash/brotli pairs are the W32 `zext`/`sext` operand
  spellings the structural-equality contract correctly refuses
  (recorded since 4a9793b). A `<=` threshold was measured
  byte-identical on all three binaries, so the strict `<` stands
  documented.
- jump-table split-block idiom ✅ — the backward def-walk crosses
  block boundaries (`jumptable`, plan in `PLAN_LEAIDIOM.md`): register
  definitions now resolve over the straight-line chain of blocks
  ending at the dispatch block — single-predecessor edges, and at a
  join only the predecessor that *dominates* it with every skipped
  edge a loop edge (standard iterative RPO dominators, function-local;
  the loop-preserves-the-base leap is documented in the module) —
  capped by `Config::max_walk_blocks` (default 8, `0` restores
  single-block matching), decoded into one instruction stream the
  unchanged idiom matchers run over, so every shape/clobber/validation
  rule applies across blocks exactly as within one. `call` now
  clobbers the caller-saved set instead of everything (rbx/rbp/r12-r15
  survive — callee-saved in both SysV and Win64 — and x19-x29 on A64):
  the split shape exists because compilers park the table base in a
  callee-saved register across the dispatch loop's calls, /bin/ls's
  `getopt` loop being the type specimen found by disassembling the
  real binary (`lea r13, [rip+table]` three blocks and one `call`
  above the `jmp rax`). 1,147 tests (+10), clippy clean; strictly
  additive on the corpus — every previously proven table byte-identical
  (bash's 37, brotli's 5; brotli's `--decompile` output md5-unchanged).
  Measured yield: /bin/ls x86-64 proves 0 → 1 table (the option
  dispatch in `sub_100000978`: jump 0x100000b1b, table 0x1000014b0,
  bound from `cmp ecx, 0x5b` → 92/92 targets validated, 2 rounds,
  1 fold) and `--decompile` renders its 92-case
  `switch (v219 + 0x1000014b0.q)`; gotos 383 → 504 and opaque markers
  106 → 108, both from the newly reached case bodies (main's marker
  became the switch; three newly reached stub functions each carry an
  unproven indirect jump). /bin/bash x86-64 proves 37 → 44 tables
  (+7 split-block sites: `sub_10000c048`, `sub_1000114f4`,
  `sub_100012b61`, `sub_1000135ac`, `sub_1000185a8`, and two in
  `sub_100068d9c`), 66 dispatch blocks folded in 2 rounds, switches
  58 → 66, opaque 433 → 425. Zero check failures, byte-deterministic
  double runs on all three binaries. Honest notes: the join rule
  needed dominance, not the plan's single-pred-only sketch — bash's
  `sub_1000135ac` re-enters its loop *at the call block* so the
  chain's exit is dominated, not single-pred — and the depth cap is 8
  because folding case bodies splits preheaders again (a depth-4 walk
  oscillated on that same function, visible as `Folded::capped`).
- decompiler slice 18 (second inversion) ✅ — SAILR shared-case-tail
  re-split (`irstruct`, plan in `PLAN_CASETAIL.md`). The classifier
  generalizes: a copy-safe tail is now also a plain leaf — or a chain
  of up to `MAX_TAIL_CHAIN` (3) of them, interiors label-free — whose
  one in-function edge converges on a single target, provided that
  edge is provably copy-free (every φ at the target takes its own
  definition on it; a loop header's carried live-ins qualify, a real
  merge does not — the layering-honest proxy for `irout`'s residual
  copies, which have exactly one textual placement and so must never
  sit on a twice-realized edge). The duplicate owes the open edge a
  realization decided per site from a context walk mirroring `check`'s
  pending flow: `Continue` where the target is the enclosing loop's
  header, `Break` where it is the loop's own next consumer (never
  from inside a switch case, where C's `break` means the switch),
  plain fall-through where it is exactly the site's next textual
  consumer — and the cheapest sufficient prefix of the chain is
  chosen, all-or-nothing per target, to a round fixpoint (a round's
  rewrites make chain interiors label-free and unlock the next).
  `check` widened in lockstep (extras sanctioned per-leaf by the one
  shared classifier) and its loop back-edge rule refined: a pending
  block whose *only* edge is the back edge always falls back, even
  re-realized — a duplicate's earlier `continue` realizes the same
  edge first (found as a real corpus `check` failure, kept as a
  fixture). `MAX_TAIL_SPLITS` raised 16 → 32 on measured evidence:
  leaf-costed chains capped 18 bash functions at 16, two at 32, and
  64 bought a fifth more gotos for twice the duplication. Zero
  duplications is still the collapse bit for bit; `pseudo` logic
  untouched, with the new φ rider proving a residual copy on a tail's
  *outgoing* edge refuses the split and keeps one textual placement,
  none dropped, none doubled. 1,147 tests (+10), clippy clean.
  Measured (`--structure`/`--decompile`): bash x86-64 16,411 → 16,068
  rendered gotos, case-body goto lines 602 → 574, 1,403 duplications
  across 412 functions, 2 capped; ls x86-64 277 → 276; libbrotlidec
  arm64 794 → 779 (`--decompile`); zero `check` failures, unplaced
  copies unchanged (231/2/26), byte-identical across runs. What still
  keeps its gotos, honestly: condition-carrying case bodies (two live
  edges have no linear duplicate — the bulk of bash's remaining 574),
  copy-carrying convergence edges, and multi-target convergence.
- aarch64 coverage three ✅ — SIMD&FP loads/stores + moves (`aarch64` +
  `aarch64_lift`, plan in `PLAN_A64SIMD.md`): the measured SIMD
  load/store ceiling. Decoded and lifted: `LDR`/`STR` of b/h/s/d/q in
  every addressing mode the integer forms have (unsigned offset,
  pre/post-index, unscaled `LDUR`/`STUR`, register offset, literal),
  `LDP`/`STP` for s/d/q, `FMOV` (register, ↔general including the
  `Vn.D[1]` lane, scalar and vector immediate), and `MOVI`/`MVNI`;
  reserved encodings refused (vector `ORR`/`BIC` immediates,
  half-precision `FMOV`, `LDNP`, opc/size holes), every golden word
  assembler-verified and every rendered spelling proven to re-assemble
  to the identical word. The plan's IR question decided as option (a),
  after inventory (no vector cells existed — SIMD state lived only in
  the `a64.unknown` clobber): the width model caps at `W64`, so each
  128-bit `Vn` is two 64-bit `Space::Arch` cells — 32–63 the low
  halves (`d0`…`d31`), 64–95 the high (`v0hi`…`v31hi`) — with plain
  load/store/assign lifts; scalar writes zero the high cell (the
  architectural rule), the q forms access `addr`/`addr + 8`,
  `FMOV Vd.D[1], Xn` is the one isolated half write, and
  `a64.unknown` now clobbers all 100 cells so unmodeled FP arithmetic
  stays sound. No downstream pass changed; the one forced ripple is
  jumptable's exhaustive `a64_defs` (SIMD ops write no X register
  beyond a writeback base; only FMOV→GPR defines Rd). Measured:
  ffmpeg arm64 97.30% → 99.11% decoded (1,283 → 425 unknown of
  47,510), /bin/ls arm64e 95.10% → 96.36% (187 → 139 of 3,819),
  libmp3lame arm64 80.95% → 90.11% (7,569 → 3,931 of 39,741),
  libbrotlidec 98.11% → 99.21% (120 → 50); the remaining tally is FP
  arithmetic outright (scvtf/fcmp/fdiv/fmul/fcvtzs/fcvt/ucvtf/fadd),
  then exclusives/acquire-release (`ldar`/`stlr`), `dup`, and PAC +
  `udf` padding on arm64e. `--decompile` full pipeline, zero check
  failures, byte-deterministic double runs: libbrotlidec
  `a64.unknown` sites 8,447 → 82 (output 21,200 → 12,966 lines — the
  `movi v0.2d, #0` + `str q/d` zeroing idioms now render as plain
  constant stores), libmp3lame 42,179 → 9,167. Honest notes, each a
  recorded next increment: (1) `callfx`'s AAPCS64 tables do not yet
  cover the vector file (caller-saved v-register clobbers at calls,
  v0–v7 in the live-out set), so cross-call vector dataflow rides the
  same conforming-code assumption the GPR tables already encode;
  (2) the 100-write clobber pushes *data symbols misclassified as
  functions* (brotli's `_k*` tables 0 → 5, lame's Huffman `_tabN`
  86 → 106) past the 65,536-SSA-name cap into the honest `no ssa`
  refusal — no real code function regressed; (3) the
  `DUP`/`INS`/`UMOV`/`SMOV` element moves deferred per the plan's
  do-not-stretch rule. 1,203 tests (+23), clippy clean.
- decompiler slice 19 ✅ — evaluation harness (`evalfx`, plan in
  `PLAN_EVAL.md`): the published evaluation methodology as regression
  infrastructure — one command, `cargo test evalfx` (a naming
  convention, not new CLI), runs five checked-in ground-truth fixtures
  through the exact `--decompile` pipeline and fails when a metric
  moves. Fixtures per the provenance rule: small known-shaped C
  (diamond, dense 6-case switch that emits a real jump table, loop
  with break/continue, SAILR cross-nesting tail merge, `&&`/`||`
  chain), compiled offline once with the system clang
  (`-arch x86_64 -O1`, version and commands verbatim in
  `fixtures/README.md`), binaries committed as bytes, each with its
  hand-written source CFG as GED ground truth — no build-time deps.
  Three metrics per DESIGN: goto count (`StructStats::gotos`, the
  SAILR metric); CFGED-lite, a dependency-free *exact* unit-cost graph
  edit distance (complete branch-and-bound over partial injective
  mappings, proved against hand-computed distances) that refuses
  graphs over 8 nodes explicitly — a documented non-metric, never an
  approximation, and a refusal must be a real over-cap refusal so
  shrinkage cannot hide behind one; and Liu & Wang-style semantic spot
  checks — at every pipeline stage (construct, optimize, forward,
  eliminate_dead) slice 7's SSA interpreter runs the SSA reading
  against its out-of-SSA rendition on seeded inputs, zero divergence
  allowed. Placement follows the interpreter's privacy: `evalfx` is a
  `#[cfg(test)]` module in `src/` (DESIGN's "`tests/` fixtures" is
  satisfied by `fixtures/`, deviation documented), and the one
  visibility ripple is `irout`'s test module going `pub(crate)` —
  still `#[cfg(test)]`, never visible to a dependent. Every expected
  number lives in ONE table (`FIXTURES` in `src/evalfx.rs`); a slice
  that legitimately moves a metric updates that table in the same
  commit. Initial numbers (blocks/edges, gotos, CFGED): diamond 4/4,
  0, 0 — recovered CFG isomorphic to source; switch_dense 9/13, 0,
  refused (9 > 8), one recovered `Switch`; loop_bc 8/10, 0, 6 (loop
  rotation + guard + split exits, honestly priced); tail_merge 7/8,
  **2**, 6 (the SAILR shape: cross-jumped volatile tails refuse the
  copy-safe re-split, both gotos survive); shortcircuit 6/8, **2**, 2
  (the flattened middle of the chain; the re-split visibly duplicates
  the join). Which metric catches which queued fidelity follow-up:
  jump threading — goto count (tail_merge's and shortcircuit's four
  gotos are its exact target, and the count may only fall); masked
  CCMP pair — semantic spot checks (a wrong flag fold diverges the
  interpreter), goto count secondary via condition-driven structuring;
  W32 zext/sext spelling — semantic checks plus shortcircuit's
  numbers (its `setcc`/`or` chain is that exact shape); load-backed
  flag-operand splice — semantic checks at the forward stage; a64
  SIMD/FP loads/stores — honestly *uncovered* here (x86-64 fixtures
  only, per plan): the libbrotlidec sweep remains its only guard
  until the queued arm64 fixture rider lands. Determinism asserted:
  double pipeline runs byte-equal (trees, stats, stages, report).
  1,185 tests (+5), clippy clean.
- decompiler slice 18 (third inversion) ✅ — SAILR jump threading
  (`irstruct`, plan in `PLAN_JTHREAD.md`). The residue the first two
  inversions refuse by design — gotos whose target *carries a
  condition*, compiler jump threading's signature — is inverted by
  duplicating the deciding block itself into the goto-ing site: its
  plain leaf plus the real `If { cond: Cond { block: <the copy>,
  negated } }`, so the copy stays referenceable by the same
  tree-shape identity scheme (the leaf stores the VA) and byte-equal
  by construction. One new classifier (`threadable_head`, shared by
  pass and `check`): a plain leaf ending in a conditional branch,
  pure register assignments only (no store/call/intrinsic), at most
  `MAX_THREAD_STMTS` (4 — bash-derived knee: threads 10/12/13/13 at
  caps 1/2/4/8) of them, and *both* live edges provably copy-free
  (inversion two's per-edge φ refusal, both polarities
  fixture-forced). Both out-edges must spell without a new
  in-function goto — arm as `Continue`/`Break`/travelling external
  goto/an inline duplicate of the fresh linear tail the thread
  exposes (the case-tail classifier itself, allowed only when no
  goto into it will remain, so no duplicated block is ever also a
  goto target); open side as `Continue`/fall-through/`Break`/external
  — cheapest spellable polarity wins, ties un-negated, so every
  threaded site still buys back exactly one goto and the round loop
  (chains, then threads, to fixpoint on the shared
  `MAX_TAIL_SPLITS` budget) cannot oscillate. `check` holds copies to
  condition honesty harder than planned: a deciding duplicate is
  sanctioned only via the shared classifier, a two-way pending block
  must be *decided* (its `If`, or the enclosing loop's condition)
  before any edge realizes (`StructFault::Undecided`), and an
  else-less `If` owes its untaken side its next realization
  (funneling both polarities one way is a `Polarity` fault) — two
  rules that also tighten the verifier for originals. 1,193 tests
  (+13, two of them `pseudo` riders proving the copy's condition
  renders byte-equal at both sites, negation polarity included, with
  no label left behind; zero `pseudo` logic changes). Clippy clean.
  Measured (`--structure`/`--decompile`, current-tree baselines
  re-verified first): bash x86-64 16,198 → 16,185 rendered gotos (13
  threaded sites across 13 functions, duplications 1,406 → 1,434,
  zero check failures across 1,782 functions); ls x86-64 and
  libbrotlidec arm64 thread zero and are **byte-identical** to
  baseline (395 / 768-struct 770-dec gotos, md5-equal dumps) — the
  zero-duplication bit-for-bit promise on whole real binaries; all
  dumps byte-deterministic across double runs. The honest small
  yield, diagnosed: of bash's 6,657 condition-carrying goto targets,
  2,716 are effectful, 2,267 refuse on φ-copy edges, 1,615 on
  spellability (both edges, every site, all-or-nothing per target),
  129 oversized — and case-body goto lines stay 631: the surviving
  case gotos are φ-heavy or unspellable, not size-refused, so the
  next lever there is expression-level (φ-web narrowing or boolean
  merging), recorded, not forced. Exhibit: bash `sub_100003601`'s
  `goto loc_100003681` — a cross-jump into a shared bounds test —
  now reads `block loc_100003681; if cond loc_100003681 { block
  loc_10000368e }` falling to `loc_100003686`, both polarities
  spelled, the arm inlining the same epilogue the original's chain
  dup carries. Landing seam (built in parallel with slice 19): the
  evalfx prediction that tail_merge's and shortcircuit's four gotos
  were this slice's exact target resolved as *refusals* — those
  targets are effectful (volatile-store tails) or φ-copy-carrying, so
  the `FIXTURES` table is untouched and `cargo test evalfx` passes
  unchanged on the merged tree; the count "may only fall" held as
  monotone-zero, not a fall.
- decompiler goto lever ✅ — the φ-web narrowing (`irstruct`, plan in
  `PLAN_GOTOEXPR.md`): jump threading's diagnosed residue, measured
  first and decided by evidence. `edge_copy_free` asked the SSA names
  (every φ argument its own definition) — an approximation of what the
  renderer executes. The ground truth is [`irout::out_of_ssa`]'s
  `edge_copies`: coalescence folds different SSA names for one value
  into one variable and emits *nothing*, so an edge absent from that
  map carries no copy and refusing it was overcaution. The narrowing:
  `copy_edges(f)` (the map's key set, computed once per `structure`
  and once per `check` — deterministic on the function, so pass,
  verifier, and renderer see one truth) replaces the name test inside
  the shared classifiers; nothing else bends — byte-identical
  duplication, the one `MAX_TAIL_SPLITS` budget, monotone buy-back,
  all-or-nothing spellability, `Undecided`/`Polarity` honesty all
  unchanged. One discovery the fixtures now document: raw SSA
  construction is *conventional* — φ-webs coalesce whole and no edge
  ever carries a copy — so every refusal fixture must force a real
  copy (`stale_phi_arg` rewrites a join argument to the entry's name,
  the shape copy propagation leaves, making it interfere with the
  arms' definitions), and the un-staled twin of the threading fixture
  is the positive: refused by the old spelling, threaded under the
  truth. Measured (release, double runs byte-compared, zero check
  failures): bash x86-64 rendered gotos 16,185 → 15,161 (structure
  stats 16,182 → 15,158), threaded sites 13 → 166, duplications
  1,434 → 3,701 within the unchanged cap, case-body goto lines
  622 → 595 (the case-label-adjacent count; the landed bullets' 631
  counted differently); brotli arm64 739 → 704 with its 65 case
  gotos untouched; ls **byte-identical** (395) — the zero-yield
  collapse bit for bit. Costs, honest: `--decompile` bash wall time
  24.2s → 25.4s (two extra `out_of_ssa` runs per function), and the
  bash dump grows 23,418,270 → 24,643,500 bytes — inlined duplicates
  are the mechanism, not a regression. evalfx: tail_merge 2 → 1 and
  shortcircuit 2 → 1, exactly the movers the harness queued for this
  lever ("the count may only fall" — it fell), `FIXTURES` updated in
  this commit, CFGED and semantic spot checks unchanged. Exhibit:
  bash 0x100000d93's `goto loc_100000d9c` now reads
  `mov(*0x10008c968, 0x1); continue;` — the volatile-store tail
  duplicated into its site and spelled as the loop edge it is. 1,235
  tests (+1 net: one positive added, three premises re-forced),
  clippy clean. Honest residue: every surviving φ-refusal now *is* a
  real copy by construction — the approximation is gone — so the
  remaining levers are the recorded ones: effectful deciding blocks,
  genuinely copy-carrying edges (irout's own future cross-cell
  coalescing would shrink these), unspellable sites, and boolean
  merging (a goto whose target's condition is congruent to one
  already decided), which is `irflow`/`irssaopt` expression territory
  queued for its own slice.
- irflow value-numbering equality ✅ — the pair-fold witness (`irflow` +
  `irssaopt`, plan in `PLAN_VNEQ.md`): the residue patterns-two
  diagnosed to the root — comparison-pair halves reaching the fold
  through *different SSA names* for one value — retired by proof, not
  splice. Measured first: bash's surviving pairs classify as
  name-split (one half a bare truncated read of the very sum the other
  half carries spliced — forwarding's duplication cap, working as
  designed) compounded by spelling-split (the 64-bit definition's
  `trunc.d(x + y)` against the 32-bit lift's
  `trunc.d(x) + trunc.d(y)`). The witness, `veq`: a structural
  fast path (bit-for-bit the old behavior, the only path without
  context), else canonical keys — exact-width reads resolved through
  `VnDefs` (every pure assignment keyed by its defining register;
  the purity gate — load-free, division-free, node-capped — enforced
  at admission in `irflow`, one doctrine one place; φs naturally
  absent, so resolution walks a DAG), truncations pushed through
  `Add/Sub/Mul/And/Or/Xor/Neg/Not` (truncation is a ring
  homomorphism — every op proved on all 65,536 W16 pairs; shifts
  refused, the modulo-width near-miss pinned), width respellings
  cancelled through the existing identities; fuel exhaustion or a
  load anywhere refuses. Threaded to exactly the pair-equality
  gates (`order_pair_operands`, the OF shape's internal occurrences,
  `order_compose_ok`, the masked pair's guard, `is_complement`) via
  `fold_stmt_vn`, called from exactly `forward`'s re-fold — real
  round and fold-shrinks tentative share it, so they cannot drift.
  The witness only proves: matchers keep returning the sign half's
  own subtrees, nothing resolved is ever emitted, and veq-true
  implies both sides load-free, so the kept operand's load gate
  covers the dropped duplicate. 1,240 tests (+6: the homomorphism
  oracle, the witness units and refusals, the e2e pair through
  `forward` with `check_preserved`), clippy clean. Measured
  (`--decompile=100000`, double runs byte-identical, zero check
  failures, goto counts untouched — this slice is expression-level):
  bash x86-64 two-`<s 0x0` lines 100 → 78 — 30 of the 78 are
  single-comparison cmov/select spellings the grep double-counts,
  so true pairs 68 → 48 — `') <s 0x0'` 908 → 887, dump
  23,418,371 → 23,415,345 bytes; the exhibit: 0x100003ab7's
  five-line flag pair now reads
  `if (trunc.d(v41) <=s trunc.d(v188) + (trunc.d(v85) -
  trunc.d(v82)) + 0x1.d)`. brotli arm64 54 → 52 (true pairs
  42 → 40), 59 → 57; ls **byte-identical** to baseline — the
  zero-fire collapse on a whole binary. `cargo test evalfx` green
  with the FIXTURES table untouched, honestly: the five fixtures
  never carry the multi-width name-split shape. Residue, classified:
  φ-defined names (φ-congruence deferred, recorded), load-backed
  operands (the sibling flag-splice slice's), and the cmov/select
  spellings (a different idiom class, never pairs). Also deferred,
  recorded: commutative/associative normalization in the witness,
  and the self-identities (`x - x`, `x ^ x`) stay structural.
- load-backed flag splice ✅ — the memory-operand `cmp` pair
  (`irssaopt` + `irflow`, plan in `PLAN_FLAGSPLICE.md`): the residue the
  forwarding-policy slice diagnosed — one `cmp` with a memory operand
  feeding two jccs, the flag trees reading a load-backed block-local
  temp, so the pair could never sit in the second jcc's block — retired
  on two theorems. *One expression, one memory state*: statements are
  the only effects, so structurally equal load-bearing subtrees inside
  one expression are value-equal, and the pair families' shared-operand
  guards (`signed_order_pair`, `order_compose_ok`, `masked_order_pair`,
  `not_of_flag_select`) relax to load-tolerant equality — the
  annihilation identities keep their load-free guards (deleting a load
  is a different contract from keeping one of two equal copies). *The
  effect-clear region*: a load-bearing cone (block-local temps inlined,
  `cone_expr`) re-reads its value at a cross-block use iff every
  statement on every def→use path is effect-free — recorded successors
  are the CFG's semantics exactly as for DCE and `optimize`; the region
  is refused, never approximated, when cyclic, past 4 blocks,
  `truncated`, or carrying any `Store`/`Intrinsic`/`Call`. On those,
  `forward` gains the load-cone joint splice (`plan_load_pairs`):
  load-bearing definitions earn their sites *jointly, all-or-nothing*
  (every use a cleared branch condition, at most 2 — so the definition
  and its exclusively-owned temps are guaranteed to sweep and the load
  is never rendered both standing and inline) under a function-level
  strict shrink by whole-statement accounting, because the per-site
  test cannot see this shape: the site grows by an inline load spelling
  while two flag definitions die. Two measured gates that earned their
  keep: branch-conditions-only (an ungated run relocated loads into
  standing assignments and *stranded* previously pure trees outside the
  pure fold-shrinks tier — bash's standing OF-tree assignments rose
  196 → 244; gated, they return to baseline exactly), and the joint
  tentative refusing any site another plan entry already claims.
  1,241 tests (+7: the e2e memory-operand comparator from the milestone
  function's real bytes through construct → optimize → forward →
  eliminate_dead, window refusals fixture-forced — store before the
  use, store between, call on the path, cyclic region, over-cap
  region — the φ-use all-or-nothing refusal, the assignment-site
  refusal, division-in-the-cone), clippy clean, `cargo test evalfx`
  green with the FIXTURES table untouched (expression-level slice;
  the interpreter's spot checks are the safety net). Measured on the
  corpus, double runs byte-identical, zero check failures: /bin/ls
  x86-64 retires 18 of its 19 standing OF-tree assignments across 10
  comparator functions — the milestone `sub_10000073f` now reads
  `if (!(v8 <s v10))` / `if (v8 <=s *(u64*)(v6 + 0x30.q))` with all
  four flag definitions swept, 18 inline-load relations where the
  baseline had none — bytes 620,001 → 617,635, gotos 503 → 503 and the
  switch untouched. /bin/bash and libbrotlidec arm64 are
  **byte-identical** to baseline: the honest zero — bash's 196
  standing pairs sit behind calls in the def→use window (`callfx`
  between the `cmp` and the second jcc, e.g. 0x1000045ec) or reach the
  halves through different SSA names, brotli's 12 are A64 `subs`
  shapes (a load/store architecture has no memory-operand compare) —
  both refusal classes recorded, the second being value-numbering
  territory. ls's one survivor (0x10000202a) reads its operands
  through `trunc.d` respellings: the same different-SSA-names class.
- aarch64 coverage four ✅ — FP arithmetic, exclusives, PAC + udf
  (`aarch64` + `aarch64_lift` + `callfx`, plan in `PLAN_A64FP.md`).
  Measured first (decode probe over each binary's `__text`, every
  unknown word named by capstone): scalar FP arithmetic owned the
  tally. Decoded and lifted: the full scalar FP data-processing space
  — two-source (FMUL/FDIV/FADD/FSUB/FMAX/FMIN/FMAXNM/FMINNM/FNMUL),
  three-source (FMADD/FMSUB/FNMADD/FNMSUB), one-source (FABS/FNEG/
  FSQRT, FCVT s↔d, all seven FRINT), FCMP/FCMPE, FCCMP/FCCMPE, FCSEL,
  SCVTF/UCVTF (from GPR and scalar-integer), FCVT{N,P,M,Z,A}{S,U} (to
  GPR) — plus the SIMD slice's deferred element moves (DUP/INS/UMOV/
  SMOV, every arrangement), exclusives/ordered (LDAR/STLR, LDXR/LDAXR,
  STXR/STLXR, every size), pointer authentication (RETAA/RETAB, the
  BRAA/BLRAA family, the dp-1source I-key row, the four PAC hints —
  named and honestly lifted, no longer execute-as-NOP `Hint`s), `UDF
  #imm16` as a real `Flow::Halt`, and the ride-alongs the tally
  justified: RBIT/REV*/CLZ/CLS, EXTR (`ror #imm` alias), LDPSW. Half
  precision refused throughout; golden words assembler-verified
  (`clang -arch arm64`/`arm64e`, otool read-back), every spelling
  re-assembles to its word. The lift doctrine, per the SMULH
  precedent: exact where bits allow (FABS/FNEG sign masks, FCSEL's
  csel merge, the element moves as shifts/masks over the two half
  cells, EXTR, exclusive loads as plain loads) and **precise named
  intrinsics over exact cells** for real FP semantics — `a64.fadd`
  writes vlo(rd) and reads its two operands, `a64.fcmp` writes exactly
  the four NZCV flags — never the 100-cell clobber, so FP dataflow
  keeps real def-use chains (lame renders 567 precise `a64.fcmp`
  sites); every scalar FP write is followed by the architectural
  `vhi := 0`; STXR emits its store (over-approximated as taken —
  source-level retry loops read naturally) plus an opaque status def.
  `callfx` closes its recorded vector-ABI gap both directions: clobbers
  gain v0–v7/v16–v31 whole and the *high halves* of v8–v15 (only their
  bottom 64 bits are callee-saved), uses gain v0–v7 whole, live-out
  gains v0–v7 plus the callee-saved d8–d15. Measured (same probe,
  before → after): libmp3lame 90.38% → **99.14%** decoded (3,799 → 340
  of 39,485), ffmpeg 99.11% → **99.96%** (425 → 19 of 47,510), /bin/ls
  arm64e 96.36% → **99.97%** (139 → 1 of 3,819), libbrotlidec
  99.21% → 99.42% (50 → 37 of 6,354 — honestly barely moved: its
  residue is vector ALU). Full `--decompile`, byte-deterministic
  double runs, zero check failures: the `no ssa` cap refusals go to
  **zero** on all three (lame 106 → 0, ls 126 → 0, brotli 5 → 0 — the
  SIMD slice's data-symbols-past-the-cap note resolved: data regions
  now halt at their first zero word since `UDF #0` terminates, instead
  of sweeping 65k SSA names), ls decompiles 2,451 → 9,741 lines of
  real arm64e recovery, lame `a64.unknown` sites 9,167 → 6,900 and its
  output grows 62.3 MB → 94.6 MB — precise FP defs let forwarding
  build real expressions where the clobber used to sever every chain.
  One rise, diagnosed: brotli's rendered `a64.unknown` sites 82 → 90
  because more real code decompiles past former refusals; its decode
  gap fell. Honest residue, all recorded with counts: the Advanced
  SIMD vector ALU (lame 340: fmul.2d 58, xtn 30, fadd.4s 29, add/cmhi
  50, the long tail), LSE atomics (ffmpeg's 8 `ldaddal`), fixed-point
  `fcvtzs #scale` (5), LD1/ST1 structure forms, D-key PAC ops; the
  listing text formatter (`aarch64_text`) still renders none of the
  SIMD/FP space — its never-guess doctrine, a separate increment; PAC
  auth traps are not modeled as control flow (RETAA returns, BRAA
  jumps, like their plain forms). 1,257 tests (+23), clippy clean,
  `cargo test evalfx` untouched (x86-64 fixtures see no aarch64
  change).
- **Active: multi-track board (post–wave 3 replan, 2026-08-16).**
  Wave three is closed (φ-web, VN pair-fold, load-backed flag splice,
  a64 FP/PAC/exclusives/udf + callfx vector ABI). Four gated parallel
  tracks share one spine (protocol, annotate DB, evalfx/CI). Gates
  before expanding each track — see `GATES.md`.

  | Track | Now | Next slices (plan-first) | Gate |
  |---|---|---|---|
  | **1 Engine** | deepen MVP | `irstack` → `sig` → MEM promote → `irtype`; interleaved fidelity: boolean-merge, SIMD ALU, VN-φ, irout coalesce, LSE, cmov idiom | **E1** named locals+params |
  | **2 iOS** | PAC Flow landed (BRAA/RETAA as branch/return) | chained fixups → encrypt/codesign detect → ObjC → Swift → IPA/DSC | **I1** fixups+imports on arm64e |
  | **3 Patch/Diff** | P1: patching non-goal **reversed** | PatchSet schema → preview → sibling apply → a64 assemble → diff hunks → resign recipes | **P1a** preview/apply CLI |
  | **4 Surfaces** | DESIGN_GUI north-star | protocol v1 → `aletheia-mcp` (cancel/progress) → TUI → Gen-Z GUI | **M1** MCP decompile+diff |

  Incumbent pain → product requirements (must win): one analysis story
  across CLI/MCP/GUI (no headless≠GUI split); parallel+cancelable jobs
  (no IDA main-thread MCP hang); git-native asserted facts; provenance
  Why?; PatchSet as auditable object (beyond IDA byte-edit); modern-lang
  + iOS as named phases.
- irflow patterns two ✅ — masked CCMP pairs + width-spelling equality
  (`irflow`, plan in `PLAN_IRFLOW2.md`): the two expression-level
  residues the last wave recorded, both landed in `fold_expr`'s
  always-sound identity family. Measured first on the current tree
  (13 real `ccmp` sites in arm64 libbrotlidec, classified one by one):
  the rendered CCMP residue is three *negated bit-set selects*
  `~((c & x) | ~c)` (b.ne/b.lo over a select whose imm4 bit is set,
  the `~c` spelled literally or as the folded flipped comparison), one
  masked signed pair `(c & SF) != (c & OF)`, and one csel-fed pair
  over materialized `N`/`V` selects — the rest fold already (single
  Z/C reads) or sit behind unproven indirect jumps. The patterns:
  `~((c & x) | ~c) → c & ~x` for any `W1` x, and the masked signed
  pair in all four imm4-bit combinations —
  `(c & SF) != (c & OF) → c & (a <s b)`, `== → ~c | (b <=s a)`, one
  side carrying `| ~c` swaps the results — with the guard shared
  structurally, deduplicated, and load-free. The width-spelling class:
  `trunc(zext(x))`/`trunc(sext(x))` cancel to `x` width-exact, narrow
  to `trunc(x)` below, re-target the extension between — the theorems
  that let structural equality see through a W32 write's respelling —
  while `zext(trunc(x))`, `sext(trunc(x))`, and sign-vs-zero near
  misses stay refused, malformed chains never laundered. Proof is the
  width-8 exhaustive oracle extended to every new identity: all
  65,536 operand pairs × both guard values × both polarities × every
  bit combination, plus a lifted `cmp;b.eq;ccmp;b.lt` chain e2e
  through construct → optimize → forward → sweep now rendering
  `goto if ((w0 != w1) & (w2 <s w3))`. 1,193 tests (+13), clippy
  clean, zero existing goldens changed. Measured on the real
  `--decompile` dumps, byte-identical double runs, zero check
  failures, gotos/switches stable: all three rendered negated selects
  collapse (brotli `~(((` 3 → 0 — _ReadHuffmanCode now reads
  `if (!((v107 != 0x1.q) & (v109 != 0x0.q)))`), inline pairs fall
  brotli 51 → 49, bash 103 → 100 (lines with two `<s 0x0`), the
  `') <s 0x0'` count brotli 57 → 54, bash 918 → 908, and the
  respellings collapse wholesale — `trunc.d(zext.q(` brotli
  964 → 9, bash 11,373 → 386, ls 213 → 1 — shrinking pseudocode
  bytes brotli 3,602,175 → 3,588,835, bash 23,573,737 → 23,408,560,
  ls 623,091 → 620,102. Honest residue, diagnosed to the root: the
  surviving pairs' operands are load-backed (the dedup may not drop a
  load copy — _ProcessCommands' 0x78b4 masked pair) or reach the two
  halves through *different SSA names* for one value (bash's
  `trunc.d(v204) + 1` vs the spliced sum), which structural equality
  correctly refuses — value-numbering territory, queued above. One
  ripple called out: bash's unplaced-edge-copy markers 232 → 234,
  because smaller normalized trees newly pass forwarding tiers and
  shift two loop-edge copies to the honest marker; brotli and ls
  unchanged (29, 2).
  the masked CCMP pair pattern for `irflow` (`(c & SF) != (c & OF)`
  shapes now visible on arm64, recorded by the coverage slice); the
  W32 zext/sext operand-spelling refusals (condrec + fwd both hit
  them); a64 FP/vector *arithmetic* (the coverage ceiling now that the
  SIMD loads/stores landed: scvtf/fcmp/fdiv/fmul/fcvt*), exclusives,
  PAC, the `DUP`/`INS`/`UMOV`/`SMOV` element moves, and the `callfx`
  vector-ABI extension (caller-saved v-clobbers at calls, v0–v7
  live-out — recorded by the SIMD slice); and the load-backed
  flag-operand splice refusal (the milestone comparator's one
  remaining composed `if`). Plan first (`PLAN_*.md` at the repo
  root), then dispatch one builder per slice.
- **Research/design phase ✅** — the checkpointed `decompiler-research`
  workflow ran (all seven topics, no gaps) and landed as `63b9008`:
  seven literature files plus `research/decompiler/DESIGN.md`, which is
  now the authority on decompiler slice order — 19 one-commit slices in
  five waves (trust → clean SSA → end-to-end early → deepen →
  fidelity), each with module, cited algorithm, invariants, test
  matrix, and exit criteria. The call-effects slice above completed
  wave 1.
- **Queued next slices** (per `DESIGN.md`; read it before planning any
  of these): wave 2 is done (propagation, DCE, forwarding); wave 3 is
  under way — structuring (`irstruct`) landed, next are Boissinot
  out-of-SSA (`irout`) and the pseudocode renderer (`pseudo`, wiring
  `redump --decompile`). Two riders the structuring slice named and
  deliberately left queued: folding `jumptable::successor_map` into
  `cfg::recover` (which is what will let real dispatch blocks render
  `Switch` instead of `Opaque`), and SAILR's de-optimization pre-passes
  (DESIGN slice 18), which are what will retire the shared-tail gotos
  the /bin/bash sweep shows. The aarch64 `irlift` dispatch with the
  SSA-render register-naming rider slots in alongside, and carries the
  `SUBS`+`B.cond` half of slice 5's exit criterion (the forwarding pass
  is ISA-blind, so it rides along for free). Condition recovery for the
  signed/unsigned *order* jcc shapes — the flag pairs slice 5 documented
  as out of scope — wants its own slice before the renderer. Each slice
  is its own plan + build + commit; builders run on Opus with the plan
  fully specified on disk first.

## Explicit non-goals

- A GUI that **computes its own analysis** — forever. Engine-first,
  poly-frontend (Phase 6 / `DESIGN_GUI.md`); frontends render protocol
  truth only.
- Breadth-chasing: no MIPS/PowerPC/AVR/8051 until the big five
  format/arch pairs are deep.
- FairPlay decrypt / jailbreak tooling inside the engine — bring
  decrypted bytes; we own everything after.
- Anything that requires reverse-engineering another tool's proprietary
  formats or internals.

## Patching principles (P1 — non-goal reversed 2026-08-16)

Binary *patching* is in scope. It is a first-class, auditable concern —
not IDA-style silent IDB byte theatre. Contracts:

- **PatchSet** objects: target hash, edits with old/new bytes, intent,
  provenance, preconditions (anchor/shape), optional postconditions.
- **Preview before write**; default apply writes a sibling `*.patched`
  (never silent in-place overwrite).
- **aarch64 assemble** first-class from public ARM encodings.
- **Apple resign** is a printed recipe (`codesign`/`ldid` + entitlements
  extract), not shipped secrets or FairPlay bypass.
- Verify by re-load + anchor diff. See `src/patch.rs` and Track 3.
