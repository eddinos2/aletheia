# Plan: ABI-aware call effects on the lifted IR (`src/callfx.rs`)

## Input / output

Input: an `irlift::LiftedFunction` plus a per-architecture `CallAbi`
summary. Output: the same function with every `ir::Stmt::Branch { kind:
Call }` followed by explicit IR statements spelling out what the call does
to registers — clobber definitions, argument uses, the stack-pointer
restore — so that `irssa::construct`'s def-use links become trustworthy
across call sites. New module `src/callfx.rs`, wired into `redump --ssa`.

## Representation (the key decision)

Call effects are **ordinary IR statements inserted by a separate pass**,
not a new statement kind and not part of the instruction lifters:

- Immediately after each `Branch { kind: Call }`, insert one
  `Stmt::Intrinsic { name: "callfx", writes: <clobbered cells>, reads:
  <argument-register superset + sp> }`, then (x86-64 only) the return-pop
  `Stmt::Assign { rsp, rsp + 8 }`.
- `ir`, `irflow`, and `irssa` need **zero algorithm changes**: intrinsic
  writes are already definition sites in `irssa` renaming
  (`rename_stmt`'s `Intrinsic` arm allocates fresh names per write) and
  already kill/gen in `irflow::transfer` and invalidate in
  `irflow::propagate`. The clobbers become new SSA versions, phis appear
  downstream of calls exactly where Cytron placement demands, argument
  registers stay live into the call, and `partial` stays empty because
  every effect write is at the full cell width (`W64` GPRs, `W1` flags).
- The lift itself stays *faithful*: an ABI is a modeling assumption about
  the callee, not instruction semantics, so `x86_lift`/`aarch64_lift` and
  `redump --lift`/`--simplify` output are untouched. Effects are opt-in
  via `callfx::apply`; `redump --ssa` opts in.

Why "after the branch": the CFG (`cfg::Terminator::Call`) ends a block at
every call with the fallthrough as its only successor, so statements
after the `Branch` read naturally as "on return". Backward liveness then
does the right thing (post-call reads of a clobbered cell are cut at the
intrinsic; the branch's own target expression — an indirect `call rax` —
still reads the *pre*-clobber value because it precedes the intrinsic in
statement order). `ir::check` imposes no branch-position rule. This
statement-ordering convention is documented in the module docs.

Semantics of the block: the call transfers, the callee's whole execution
is summarized by the one intrinsic, control reaches the fallthrough.

## The `CallAbi` model and its two tables

```rust
pub struct CallAbi {
    /// Cells a conforming callee may overwrite: caller-saved GPRs as
    /// full-width W64 `ir::Reg`s (return-value registers included — a
    /// return def IS a clobber def) plus the W1 flag cells the ISA's
    /// lifter models. These become the intrinsic's `writes`.
    pub clobbers: Vec<ir::Reg>,
    /// Registers a conforming callee may read: the argument registers as
    /// a superset (argument counts are unknown), plus the stack pointer
    /// (stack arguments / return address). The intrinsic's `reads`,
    /// as `Expr::Reg` at W64.
    pub uses: Vec<ir::Reg>,
    /// The stack-pointer cell, at W64.
    pub sp: ir::Reg,
    /// Bytes the return pops (the callee's `ret` consuming the pushed
    /// return address): 8 on x86-64, 0 on aarch64. Nonzero emits the
    /// `sp := sp + sp_pop` assign after the intrinsic.
    pub sp_pop: u64,
}
```

The tables live in `callfx` itself (constructor fns, fixed order,
deterministic), with unit tests pinning every number against
`x86_lift::reg_name` / `aarch64_lift::reg_name` so the numbering
correspondence with the lifters is *verified*, not assumed. (Putting them
in the lifters would make `callfx` and the lifters mutually dependent for
no gain; the tests carry the coupling instead.)

`callfx::x86_64()` — deliberately the **union of SysV AMD64 and Microsoft
x64**, so one table is sound for ELF/Mach-O *and* PE images (Win64's
clobbers {rax rcx rdx r8-r11} and register args {rcx rdx r8 r9} are both
strict subsets of SysV's):

- clobbers (writes, ascending num, then flags in discriminant order):
  rax(0), rcx(1), rdx(2), rsi(6), rdi(7), r8, r9, r10, r11 at W64;
  CF, ZF, SF, OF, PF at W1. Callee-saved rbx(3), rsp(4), rbp(5), r12-r15
  are absent — the ABI preserves them, so their SSA version flows through
  the call unbroken.
- uses (reads, argument order then extras): rdi, rsi, rdx, rcx, r8, r9
  (the six SysV integer args), then rax (SysV varargs vector count — `al`
  is a real input compilers set with `xor eax,eax`), r10 (static chain),
  and rsp (return address, stack args). All W64.
- sp = rsp(4, W64), sp_pop = 8. Composed with the lift's own
  `rsp -= 8; store [rsp], retaddr`, the net rsp change across a call is
  zero and rsp/rbp frame tracking stays *exact* through calls — modeling
  rsp as a clobber instead would destroy stack-slot reasoning forever.

`callfx::aarch64()` — AAPCS64 (Apple arm64 agrees on the GPR sets; x18 is
platform-reserved and conservatively clobbered):

- clobbers: x0–x18 and x30 at W64 (x30 because the callee's own calls
  trash LR; the `bl` lift's own `x30 := retaddr` def precedes the branch
  and this def supersedes it after the call); CF, ZF, SF, OF at W1 —
  exactly the four NZCV flags `aarch64_lift` models, no Parity.
  Callee-saved x19–x28, x29(FP), and SP(31) are absent.
- uses: x0–x7 (args), x8 (indirect-result pointer), sp(31). All W64.
- sp = sp(31, W64), sp_pop = 0 — `bl`/`ret` do not touch SP, so no
  adjust statement is emitted.

`callfx::abi_for(arch: model::Arch) -> Option<CallAbi>`: `X86_64` →
`x86_64()`, `Aarch64` → `aarch64()`, `Other` → `None`.

## Soundness directions (stated per consumer)

Both lists are **over-approximations, in the direction each consumer
needs**, and both directions tolerate over-inclusion:

- **Defs (clobbers) over-approximated.** For propagation: any fact about
  a possibly-clobbered cell must die at the call — extra clobbers only
  kill more facts (lost precision, never wrong values). For SSA: an extra
  clobber def merely severs a def-use link that a preserved register
  would have kept (the post-call use reads the intrinsic's "unknown"
  def) — conservative, never a wrong link. Under-approximating defs would
  let a stale constant survive a call: unsound. So every register the ABI
  does not *guarantee* preserved is in `clobbers`.
- **Uses (args) over-approximated.** For DCE: a definition feeding a
  possible argument must never be deleted — extra uses only keep more
  defs alive. Under-approximating uses would delete a live argument
  setup: unsound. So the full argument-register set (plus rax/r10/x8
  extras) is read at every call regardless of the callee's real arity.
- **Memory** needs no modeling here: `irflow` never deletes a load-bearing
  assign, a store, or an intrinsic, and neither pass reasons about memory
  contents, so stack-passed arguments and callee memory effects are
  already conservatively safe.
- **Trust boundary (documented):** this models a *conforming* callee. A
  callee violating its ABI (hand-written asm clobbering rbx) is outside
  the model — the same assumption every ABI-aware analysis makes, stated
  in the module docs rather than implied.

Direct vs indirect vs unknown callees get the **same** summary in this
slice: a known import's prototype could narrow uses (or mark noreturn),
but that is per-callee knowledge, deferred (non-goal below).

## `apply` (the pass)

`pub fn apply(func: &irlift::LiftedFunction, abi: &CallAbi) ->
irlift::LiftedFunction` — pure, deterministic, total, no panics:

- For each block, scan `stmts`; after every `Branch { kind: Call }`
  insert the intrinsic, then the sp-pop assign when `sp_pop != 0`.
  Every `Branch::Call` occurrence is handled (lifted code has at most one,
  as the block terminator; hand-built input may have several).
- If insertion would push the block past `ir::MAX_STMTS`, insert nothing
  for that block and set its `truncated` flag — the existing honest
  "this block's model is incomplete" marker (such a block is at the
  irlift trim cap and already effectively truncated; reachable only in
  synthetic tests).
- Everything else — entry, name, block keys, bounds, successors,
  existing `truncated` flags — is copied verbatim. Blocks with no
  `Branch::Call` (jumps, returns, tail-call jmps, truncated lifts that
  never reached their call, `generic_intrinsic("call", …)` fallbacks) are
  unchanged; each of those is already conservative on its own terms.
- Output invariant: every block that passed `ir::check` before still
  passes it (the intrinsic reads plain W64 regs, the assign is
  width-agreeing; add a debug assertion nowhere — tests verify it).

## How the consumers change

- **`irssa`** (`src/irssa.rs`): *no code change.* Rewrite the module-doc
  section "What is deliberately not modeled" to: call clobbers are
  modeled when the input has been through `callfx::apply` (as
  `redump --ssa` does); on a raw lift the old caveat stands verbatim.
  Emergent bonus worth a doc sentence and a test: argument registers read
  by `callfx` with no prior def become version-0 `live_in` entries, so a
  function's rendered live-in line starts to resemble its parameter list.
  Name-cap pressure: an x86-64 call adds 15 defs (14 intrinsic writes +
  the rsp assign), aarch64 24, so `TooManyNames` now trips near ~4,000
  calls per function — still an explicit error, never a wrap; note it in
  the `Unrepresentable::TooManyNames` doc.
- **`irflow`** (`src/irflow.rs`): *no code change.* `transfer` already
  kills intrinsic writes and gens its reads (so per-block DCE with a
  callfx'd block can now delete a dead pre-call def of a pure-clobber
  register like r11 while argument defs stay pinned), `propagate` already
  invalidates per intrinsic write, `default_live_out` already collects
  intrinsic arch writes. `propagate`'s clear-everything-at-`Branch` rule
  stays: within a block it is dead code after a call terminator anyway,
  and narrowing it to clobbers-only pays off only in the *cross-block*
  propagation slice, which will consume the callfx defs through SSA
  instead. Add a short module-doc paragraph saying calls become explicit
  intrinsics upstream (`callfx`) and the existing conservative rules are
  why no special-casing is needed here.
- **`redump`** (`src/bin/redump.rs`): `print_ssa`'s pipeline becomes
  `lift_function` → `callfx::abi_for(image.arch())` → `apply` →
  `irssa::construct` (the arch gate already restricts to x86-64, so
  `abi_for` is always `Some` there; keep the `abi_for` dispatch anyway so
  the aarch64 path lights up with the future irlift-dispatch slice). No
  new flag: trustworthy-across-calls is the *point* of `--ssa`, and
  `--lift`/`--simplify` remain the faithful views. Update the `--ssa`
  usage text to mention modeled call effects.
- **`lib.rs`**: `pub mod callfx;` (alphabetical, between `asmtext` and
  `cfg`).

Rendered form (via the existing intrinsic renderer, nothing new):

```
  call 0x401020.q
  rax#4, rcx#5, …, PF#3 := callfx(rdi#2, rsi#1, …, rsp#3)
  rsp#4 := (rsp#3 + 0x8.q)
```

## The aarch64 register-naming rider: out (mostly)

The roadmap rider — `irssa::display` names registers via
`x86_lift::reg_name` — is **not** resolved in this slice, deliberately:
`irlift::lift_function` still dispatches only x86-64 (`Arch::Aarch64 =>
None`), so no aarch64 `LiftedFunction`, and hence no aarch64 SSA render,
can exist today; fixing the namer would be untestable end-to-end and
needs an arch tag threaded through `LiftedFunction`, which belongs to the
irlift-aarch64-dispatch slice. What this slice *does* ship for aarch64 is
the `callfx::aarch64()` table with its lifter-numbering cross-check
tests, so when that dispatch slice lands, call effects are ready-made.
Update the roadmap's Current thread note accordingly when committing.

## Reuse vs add

- Reuse: `ir::Stmt::Intrinsic` as the effect carrier (no IR change at
  all), `irssa`'s existing intrinsic-def renaming, `irflow`'s existing
  intrinsic transfer/invalidate, `irlift::LiftedFunction` as the
  in/out contract, both lifters' `reg_name` hooks as test oracles.
- Add: `src/callfx.rs` (`CallAbi`, `x86_64()`, `aarch64()`, `abi_for`,
  `apply`), the `redump --ssa` pipeline hookup, doc updates in `irssa`
  and `irflow`, `pub mod callfx;` in `src/lib.rs`.

## Test matrix (~25–30 tests)

`callfx` unit tests:
- `x86_64()` table exact — every clobber/use number and width asserted,
  and cross-checked through `x86_lift::reg_name` (`0 → "rax"`, …);
  rbx/rbp/r12–r15 absent from clobbers; sp = rsp, sp_pop = 8;
- `aarch64()` table exact via `aarch64_lift::reg_name` (`31 → "sp"`);
  exactly four flags (no Parity); x19–x29 and sp absent; sp_pop = 0;
- `abi_for` dispatch: X86_64 / Aarch64 / Other;
- `apply` on a lifted direct-call block: intrinsic then rsp-assign appear
  immediately after the `Branch::Call`, block passes `ir::check`, golden
  render of the tail;
- `apply` on an indirect call (`call rax`): the branch target still reads
  the pre-clobber value (target expr precedes the intrinsic);
- aarch64 table through `apply` on a hand-built block: no sp assign;
- non-call blocks (jump / return / tail-call jmp) byte-identical;
  truncated block without a call unchanged; multiple hand-built calls in
  one block each get effects;
- overflow: a block at `ir::MAX_STMTS` ending in a call → no insertion,
  `truncated` set, still passes `ir::check`;
- structure preservation (entry/name/keys/bounds/successors) and
  determinism (apply twice, equal).

`irssa` integration (new tests in `irssa`, building on `callfx::apply`):
- def of a caller-saved reg before a call, use after: the use links to
  the intrinsic's fresh version, not the pre-call def (versions differ);
- def of callee-saved rbx before, use after: same SSA name across the
  call (no severed link);
- an argument-register def flows into the intrinsic's read; an undefined
  argument register read by callfx appears in `live_in`;
- rsp chain across a call: push-def → intrinsic (no rsp def) → pop-def,
  sequential versions, `partial` empty;
- a diamond where only one arm calls: phi at the merge for a clobbered
  cell that is live there; `irssa::check` passes on every constructed
  function above.

`irflow` (behavioral, no code change to verify but contracts to pin):
- `eliminate_dead` on a callfx'd block deletes a dead pre-call r11 def
  and keeps a pre-call rdi (argument) def;
- `liveness`/`live_in`: clobbered cells not live across the intrinsic,
  argument cells live before the call.

`redump` end-to-end:
- a synthetic x86-64 image whose entry calls a second function: `--ssa`
  output contains `callfx(` with versioned clobbers and the rsp restore;
  run twice, byte-equal;
- `--lift` and `--simplify` output on the same image contains **no**
  `callfx` (the faithful views are untouched).

Invariants throughout: every output block passes `ir::check`, every
constructed SSA passes `irssa::check`, no panics on any input (including
hand-built pathological blocks), full determinism.

## Non-goals (this slice)

- Per-callee knowledge: import prototypes, noreturn callees (the
  fallthrough edge stays), arity-narrowed argument sets, callee-pops
  conventions. One conservative summary per architecture, applied to
  every call site, direct or indirect.
- Floating-point/vector argument and return modeling — the IR carries no
  FP state (module contract in `ir`); xmm/v-register effects remain
  outside the model.
- Cross-block constant/copy propagation and dead-phi elimination — the
  next slice; this one exists so that slice is sound.
- The aarch64 `irlift` dispatch and the SSA-render register-naming
  hookup (see the rider section — deferred with reasons).
- Any change to `ir`'s types, `irssa`'s algorithms, or `irflow`'s passes.
