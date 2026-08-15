//! Sparse constant/copy propagation, φ-simplification, expression
//! forwarding, and conservative dead-code elimination on SSA.
//!
//! [`crate::irflow`] simplifies one straight-line block; it stops at every
//! block boundary, because before SSA "which definition does this read
//! see" has no cheap answer across a join. [`crate::irssa`] answered that
//! question once and for all — one definition per name — so the classic
//! sparse propagation of Cytron, Ferrante, Rosen, Wegman and Zadeck
//! (TOPLAS 1991) applies directly: on SSA the single-definition property
//! *is* the analysis, and copy propagation is substitution. This module is
//! that pass. It takes an [`SsaFunction`] and returns a **new** one in
//! which every use of a name proven constant reads the constant, every use
//! of a plain copy reads the copied-from name, and a φ that merges one
//! value (`φ(x,…,x)`, `φ(x, self)`, including through copy chains and
//! cascaded through φ-of-φ) is gone.
//!
//! # What [`optimize`] does not do
//!
//! - **No dead-code elimination.** A definition whose uses were all
//!   rewritten stays exactly where it was (`rax#1 := 0x5.q` survives its
//!   last reader); statement count and order are invariants for
//!   [`optimize`], as they are for [`irflow::propagate`]. Sweeping them
//!   is [`eliminate_dead`], the last pass in this module, run after
//!   [`optimize`] has exposed the dead definitions.
//! - **No expression forwarding.** Only a whole `Const` or a bare name is
//!   ever substituted, never a compound right-hand side. Compound trees are
//!   [`forward`]'s business, the second pass in this module, run between
//!   [`optimize`] and [`eliminate_dead`].
//! - **No SCCP.** Full conditional propagation (Wegman–Zadeck, TOPLAS
//!   1991) deletes unexecutable edges, and the CFG belongs to the
//!   structuring pass that will consume it. There is no executability
//!   tracking here: the φ meet runs over *all* arguments, which is the
//!   conservative, always-sound reading. A branch whose condition folds to
//!   a constant keeps its edges and its `Branch` statement; only the
//!   condition *expression* changes.
//! - **No memory facts.** A load is never folded away, never deleted, and
//!   never a propagation source.
//!
//! # The lattice
//!
//! Per SSA name, computed optimistically to a fixpoint by a sparse
//! worklist over a def-use index built here (the SSA form stores no use
//! lists; [`irssa::check`]'s from-scratch doctrine means every consumer
//! rebuilds what it needs):
//!
//! ```text
//! Top  (no evidence yet)
//!   >  Known: Const(value, width)  — the definition's width, value masked
//!         or Copy(root)            — the same runtime value as `root`
//!   >  Bottom (varies / unknowable)
//! ```
//!
//! A version-0 name (the at-entry value) and an intrinsic write are
//! `Bottom`, always. That is where call clobbers kill facts for free: a
//! [`crate::callfx`] intrinsic write is a fresh name with no value, so no
//! fact crosses a call for a caller-saved cell, while a callee-saved cell
//! keeps its name *and* its fact — no special case anywhere in this file.
//!
//! # Termination
//!
//! Each name moves at most `Top → Known → Bottom`, with one representation
//! refinement inside `Known` (a `Copy` whose root turns out constant), so
//! there are at most three lowering steps per name and the fixpoint is
//! reached in `O(names + use-edges)` worklist pops; a defensive cap on the
//! pop count aborts the pass and returns the input **unchanged** rather
//! than leaving optimistic mid-fixpoint facts applied, which would be
//! unsound. `resolve` — the walk that follows a copy chain to its root —
//! is bounded by the name count and answers `Cycle` rather than looping:
//! the lowering rule makes a copy cycle unreachable (a recorded `Copy(r)`
//! never re-points, and a φ only ever copies a `Bottom`-valued name), but
//! a *transient* mid-fixpoint cycle between two mutually referencing φs is
//! exactly the kind of thing an optimistic solver must not hang on, so the
//! bound is unconditional and a `Cycle` answer makes the meet `Bottom`.
//!
//! # Soundness rules (each stated per consumer)
//!
//! - **Widths.** A use is rewritten only when its width is not wider than
//!   its definition guarantees ([`SsaFunction::partial`] positions read
//!   bits the definition never wrote and are left alone), a copy fact is
//!   recorded only when the source guarantees at least the copy's width,
//!   and a substituted constant is masked to the reading width. No rewrite
//!   can *add* a `partial` entry; the marker set only shrinks.
//! - **Temporaries.** [`crate::ir::check`] treats a block as straight-line and
//!   rejects a [`crate::ir::Space::Temp`] read with no earlier write in the same
//!   list, so a copy whose root is a temporary is substituted only inside
//!   the root's defining block (for a φ argument, the predecessor block).
//! - **Intrinsic reads are never rewritten** — a deliberate divergence
//!   from [`irflow::propagate`], which does substitute into them. A
//!   [`crate::callfx`] read models "the callee observes this argument
//!   *register*"; keeping the register identity is what lets a later DCE
//!   stay trivially sound (argument setups stay pinned) and lets signature
//!   recovery read argument registers off the call.
//! - **Division by zero is never folded** and **loads are never folded
//!   away**: folding is [`irflow::fold_expr`], shared rather than
//!   duplicated, so that doctrine carries over verbatim. Substitution
//!   sources are `Const` or bare `Reg` only — load-free by construction —
//!   so no fact depends on memory and stores need no invalidation.
//! - **No CFG mutation.** Block set, bounds, successors, `truncated`,
//!   `entry`, `name`, and `skipped` are copied verbatim.
//!
//! # Expression forwarding
//!
//! [`forward`] is the second pass: it substitutes a definition's *whole
//! right-hand side* into its use sites (van Emmerik 2007, the
//! decompiler-specific workhorse), which is what rebuilds source-level
//! expressions out of a lift's one-operation-per-statement form —
//! `ZF#3 := ((a - b) == 0); goto if ZF#3` becomes
//! `goto if ((a - b) == 0)`, and [`irflow::fold_expr`]'s relational
//! identities finish the job as `goto if (a == b)`. Flag plumbing gone.
//!
//! **The def stays standing.** `forward` never deletes, adds, or reorders
//! a statement and never touches the CFG: the forwarded definition simply
//! loses its uses, and [`eliminate_dead`] — already in the same pipeline —
//! sweeps it. So [`check_preserved`] applies to `forward` verbatim, and
//! each pass keeps one job.
//!
//! **When a use site is eligible.** The read must be *exact-width*
//! (`width == names[n].width`; splicing a tree under a narrower,
//! truncating read would need a wrapper node and buys nothing), it must
//! not be an intrinsic read (the [`crate::callfx`] register identity again
//! — same barrier [`optimize`] keeps), and it must not be a φ argument
//! (those name versions, not expressions). Every [`Space::Temp`] name the
//! spliced tree reads must be defined in the *use's* block, since
//! [`crate::ir::check`] reads a block as straight-line and rejects a
//! temporary read with no earlier write in the same list.
//!
//! **Then by right-hand-side class**, DESIGN's tiers:
//!
//! - **Trivial** (a constant or a bare name): [`optimize`] already does
//!   this, so `forward` skips it rather than implementing it twice.
//! - **Compound, pure, load-free, division-free:** into *all* uses when the
//!   tree is small (`FWD_SMALL_NODES`), else only when the definition has
//!   exactly one use — DREAM++'s finding that duplication hurts
//!   readability (Yakdan et al., IEEE S&P 2016) is what caps the copying —
//!   unless the copy provably shrinks. A bigger pure tree still goes to
//!   several uses when a *tentative* substitute-then-fold
//!   ([`irflow::fold_stmt`], the very fold the real splice gets) leaves
//!   every cleared use-site statement strictly smaller than it stands.
//!   Decided per definition, deterministically, from the folded results: a
//!   site the tentative cannot splice (narrower reads only, or no node
//!   budget) is simply not cleared, while one that would not shrink
//!   refuses the whole definition. The paired flag shapes are the
//!   motivating case — an 11-node overflow tree spliced beside its sign
//!   twin folds to the 3-node relation, no duplication at all — so
//!   DREAM++'s readability concern is honored by construction, not
//!   waived.
//! - **Load-bearing:** single-use, and only into a use in the definition's
//!   own block, after it, with no intervening [`Stmt::Store`],
//!   [`Stmt::Intrinsic`], or [`Stmt::Branch`] — [`irflow::propagate`]'s
//!   barrier set lifted onto SSA names (a call is a `Branch` *and* an
//!   intrinsic, so either gate catches it). A load is an effect
//!   observation: duplicating its tree to N sites reads as N loads even
//!   where it is sound, so it moves only if it moves once — with one
//!   all-or-nothing exception, the *load-cone joint splice*
//!   ([`plan_load_pairs`]): a definition whose cone (block-local temps
//!   inlined) is load-bearing may copy into **every** one of its uses at
//!   once, so the definition and its exclusively-owned temps are
//!   guaranteed to sweep and the load is never rendered both standing and
//!   inline. Every use must be a *branch condition* (relocating a load
//!   into a standing assignment merely moves it, and strands a
//!   previously pure tree outside the pure fold-shrinks tier — measured
//!   on bash before the gate), each needs a provably effect-clear
//!   window — the same between-scan same-block, or the bounded acyclic
//!   effect-clear region cross-block ([`effect_clear_region`]) — and the
//!   group's joint tentative fold must leave the *function* strictly
//!   smaller by whole-statement accounting. The memory-operand `cmp` feeding two
//!   jccs is the motivating case: each jcc renders one relation with the
//!   load inline, exactly the two reads the source spelled. What the
//!   window cannot prove — a volatile/MMIO location where the re-read
//!   *count* itself is the semantics — rides the same conforming-code
//!   assumption [`crate::callfx`] records.
//! - **Division-bearing** (any div/rem node): same block, no intervening
//!   `Branch` — a potential trap must not move past a guard. Combines with
//!   the load rule when both apply.
//!
//! Substitution is sound on the SSA guarantee alone: the definition
//! dominates every use, and every name its right-hand side reads is
//! defined at a point dominating the definition, so no path can redefine
//! one between the two (a path from such a definition to the use that
//! avoided the forwarded definition would contradict its dominance). The
//! values are therefore identical at both points; only loads and divisions,
//! which depend on state the names do not carry, need the positional
//! windows above.
//!
//! A substituted statement is re-folded ([`irflow::fold_stmt`]), and a
//! substitution that would push an expression past
//! [`crate::ir::MAX_EXPR_NODES`] is refused and counted
//! ([`FwdStats::size_skipped`]) — never truncated. Substitution cascades
//! (`t1 := a + b; t2 := t1 * 2; use t2`), so `forward` rebuilds its index
//! and sweeps again until a round changes nothing, `MAX_ROUNDS` as the
//! bound. Unlike [`optimize`]'s optimistic lattice, every intermediate
//! state here is sound, so hitting the cap returns the *last completed
//! round's* output with `capped = true` rather than the input.
//!
//! **The order conditions — covered since condition recovery.** The
//! identities above were originally only the equality family
//! (`(a - b) == 0 → a == b`, `(a ^ b) != 0 → a != b`, `~(a == b) →
//! a != b`) — exactly the `je`/`jne` plumbing — while the x86
//! signed-order conditions lift to `SF ^ OF` shapes
//! (`x86_lift::cond_expr`'s `(SF == OF)`, `(SF != OF)`, and the
//! `jle`/`jg` conjunctions over them) whose collapse to `(a <s b)` needs
//! the *pair* of flag definitions recognized together, not a
//! single-operator rewrite. Forwarding already puts both flags into one
//! expression — that was the design — so the pairwise patterns live in
//! [`irflow::fold_expr`]'s order family (see [`irflow`]'s
//! "Order-condition recovery"): `SF != OF → a <s b`, `SF == OF →
//! b <=s a`, the `jle`/`jg` compositions over their collapsed halves,
//! and the unsigned pair the same way — `CF | ZF → a <=u b` (`jbe`),
//! `~CF & ~ZF → b <u a` (`ja`) — with A64's NOT-borrow `C` covered by
//! the identical compositions. `forward`'s spliced conditions simply fold
//! further, and the emptied flag definitions sweep as before. The
//! "forwarding already puts both flags into one expression" premise held
//! only for single-consumer flags, though: one `cmp` feeding *two* jccs
//! left the 11-node overflow tree with two uses, which the duplication
//! cap refused — the residual ~20% of order conditions measured on the
//! real corpora. The fold-shrinks exception above is what retires that
//! residue: the tentative fold sees the pair collapse, so the splice is
//! exactly the case the cap was never meant to block. Two later residue
//! classes also live in [`irflow`], not here: operand spellings
//! diverging through W32 `zext`/`sext`/`trunc` chains (its width-spelling
//! normalization) and the condition-masked pairs a CCMP leaves (its
//! masked order family) — both fold under the same tentative
//! substitute-then-fold, so this pass again needed no algorithm change.
//! What survived that, measured: pairs whose operands are load-backed —
//! retired by the load-cone joint splice above together with [`irflow`]'s
//! one-expression equality (two structurally equal load subtrees in one
//! statement read one memory state) — and pairs whose halves reach the
//! comparison through *different* SSA names for the same value, which
//! structural equality honestly cannot see. The second class is retired
//! by the value-numbering witness: [`fwd_round`] hands its re-fold an
//! [`irflow::VnDefs`] — every pure (load-free, division-free) assignment,
//! keyed by its defining register — and [`irflow`]'s pair matchers may
//! then prove two spellings name one value by resolving a full-width read
//! through its unique definition and normalizing the truncation spelling
//! (see [`irflow`]'s "The equality witness"). Only the *proof* changed:
//! what forwards, and what any fold emits, is decided exactly as before.
//! The two witnesses never mix: [`irflow::VnDefs`] resolution is
//! load-free, so a load-bearing equality is always structural and rides
//! the one-expression theorem alone.
//!
//! # Dead-code elimination
//!
//! [`eliminate_dead`] is the third pass: mark-and-sweep liveness over
//! the same def-use index (Cytron et al. 1991; van Emmerik 2007 on DCE
//! as the pass that shrinks a raw lift toward source shape). SSA makes
//! it a graph reachability question — a definition is live exactly when
//! it is reachable backward from something observable.
//!
//! **Roots.** A statement this pass never deletes has unconditionally
//! live reads, so every read of a [`Stmt::Store`], a [`Stmt::Branch`],
//! a [`Stmt::Intrinsic`], or a *load-bearing* [`Stmt::Assign`] is a
//! root. Intrinsic reads being roots is what pins argument setups across
//! a [`crate::callfx`] site — and is why [`optimize`] refuses to rewrite
//! them. The second root set is the function's live-out.
//!
//! **Live-out roots are cells, not versions.** [`irssa`] builds *pruned*
//! SSA: a version that reaches a `Return` without being read there is
//! never materialized (no φ exists for a cell nobody reads), so "the
//! version of rax at this return" is not recoverable after construction
//! without redoing the dominator walk. This pass therefore marks
//! conservatively: **every definition of a live-out cell is a root**,
//! whatever its version. Sound (a return value or a callee-saved restore
//! can never be deleted), linear, and it still kills exactly the lift
//! noise the pass exists for — flag writes and temporaries are never
//! live-out cells. The documented precision limit: a genuinely dead
//! definition of a live-out cell (an `rax` overwritten before any use)
//! survives this pass. Narrowing it needs reaching-version analysis or
//! recovered signatures; neither is invented here. The live-out set
//! itself is [`crate::callfx::function_live_out`], an over-approximation
//! by construction — an extra root keeps a dead definition, a missing
//! one would delete a live one.
//!
//! **Sweep.** One pass in address then index order. An `Assign` goes iff
//! its destination is unmarked *and* its right-hand side is load-free: a
//! [`Expr::Load`] may fault, so deleting one is unproven — the
//! [`irflow`] doctrine verbatim, sharing [`irflow`]'s own predicate. An
//! unmarked load-bearing assign is kept and counted in
//! [`DceStats::kept_loads`]. A φ goes iff its destination is unmarked.
//! Nothing is reordered and the CFG — blocks, successors, `truncated`,
//! `entry` — is copied verbatim: control-dependence DCE (deleting a
//! branch) belongs to the structuring pass that owns CFG shape. Then the
//! name table is compacted and `partial` recomputed, reusing
//! [`optimize`]'s own machinery.
//!
//! A single mark-and-sweep is already a fixpoint, because marking is
//! transitive: a φ read only by a dead assign is never marked (the dead
//! assign's reads never were), so it sweeps in the same pass. The
//! idempotence test pins that.
//!
//! # Contract
//!
//! [`optimize`] is pure, total, and deterministic: every map and worklist
//! is a `BTree` or index order, so equal inputs give byte-equal outputs,
//! and no input panics. Malformed input is not laundered — a function that
//! fails [`irssa::check`] is returned unchanged with `rounds = 0`, the
//! refuse-don't-guess posture [`irssa::construct`] already takes.
//! [`forward`] and [`eliminate_dead`] take the same posture and the same
//! guarantees. [`check_preserved`] and [`check_swept`] are the companion
//! differential checks every test runs: they verify each pass's contract —
//! preservation for [`optimize`] and [`forward`], a justified subsequence
//! for [`eliminate_dead`] — without trusting the pass.

use std::collections::{BTreeMap, BTreeSet};

use crate::ir::{self, BranchKind, Expr, Reg, Space, Stmt, Width};
use crate::irflow;
use crate::irssa::{self, Cell, Name, Phi, SsaBlock, SsaFunction};

/// Outer passes [`optimize`] runs before returning what it has. One round
/// reaches closure for these rules (the analysis already evaluates through
/// substitutions), so the loop is defense; the value mirrors `irflow`'s
/// own `MAX_ROUNDS`, which is private to that module and so restated here.
/// [`forward`], whose substitutions genuinely cascade one level per round,
/// uses the same bound — the same constant discipline, one constant.
const MAX_ROUNDS: usize = 8;

/// Largest right-hand side, in expression nodes, that [`forward`] copies
/// into *more than one* use site unconditionally. A bigger pure tree may
/// still earn multiple sites through the fold-shrinks exception (every
/// tentatively folded site strictly smaller — see the module docs); a
/// bigger load- or division-bearing tree moves only when it moves once.
/// The readability constant, from DREAM++'s finding that duplication
/// hurts (Yakdan et al., IEEE S&P 2016), not a soundness one.
const FWD_SMALL_NODES: usize = 8;

/// Depth at which the rewriting walk stops recursing and returns the
/// subexpression unchanged, mirroring `irflow`'s rewrite bound: a tree
/// deeper than this only arises from input [`crate::ir::check`] already rejects,
/// and the cap keeps a rewrite that builds new owned expressions from
/// exhausting the stack.
const REWRITE_DEPTH: usize = 512;

// ---------------------------------------------------------------------------
// The def-use index
// ---------------------------------------------------------------------------

/// Where a name occurrence in *use* position lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum UseSite {
    /// Inside statement `index` of `block` (any number of occurrences).
    Stmt { block: u64, index: usize },
    /// Argument `arg` of φ number `phi` in `block` — the value observed at
    /// the predecessor's exit.
    PhiArg { block: u64, phi: usize, arg: usize },
}

/// Where an SSA name is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Def {
    /// Version 0: the at-entry value. Opaque.
    Entry,
    /// The single destination of statement `index` of `block`.
    Assign { block: u64, index: usize },
    /// One write of the intrinsic at statement `index` of `block`. Opaque.
    IntrinsicWrite { block: u64, index: usize },
    /// φ number `phi` at the head of `block`.
    Phi { block: u64, phi: usize },
}

/// The def-use scaffolding, rebuilt from scratch every round.
struct Index {
    /// Name id -> its use sites, ascending.
    uses: BTreeMap<u16, BTreeSet<UseSite>>,
    /// Name id -> its definition, indexed by id.
    defs: Vec<Def>,
    /// Total use occurrences, for the worklist cap.
    total_uses: usize,
}

/// Build the index in one deterministic pass over the blocks. Names with
/// no definition occurrence keep [`Def::Entry`]: on a function that passes
/// [`irssa::check`] those are exactly the version-0 names.
fn build_index(f: &SsaFunction) -> Index {
    let mut uses: BTreeMap<u16, BTreeSet<UseSite>> = BTreeMap::new();
    let mut defs = vec![Def::Entry; f.names.len()];
    let mut total_uses = 0usize;
    let set_def = |id: u16, def: Def, defs: &mut Vec<Def>| {
        if let Some(slot) = defs.get_mut(id as usize) {
            *slot = def;
        }
    };
    for (&va, block) in &f.blocks {
        for (p, phi) in block.phis.iter().enumerate() {
            set_def(phi.dst, Def::Phi { block: va, phi: p }, &mut defs);
            for (a, &(_, arg)) in phi.args.iter().enumerate() {
                uses.entry(arg).or_default().insert(UseSite::PhiArg {
                    block: va,
                    phi: p,
                    arg: a,
                });
                total_uses += 1;
            }
        }
        for (i, stmt) in block.stmts.iter().enumerate() {
            irssa::for_each_use(stmt, &mut |r| {
                uses.entry(r.num).or_default().insert(UseSite::Stmt {
                    block: va,
                    index: i,
                });
                total_uses += 1;
            });
            let opaque = matches!(stmt, Stmt::Intrinsic { .. });
            irssa::for_each_def(stmt, &mut |r| {
                let def = if opaque {
                    Def::IntrinsicWrite {
                        block: va,
                        index: i,
                    }
                } else {
                    Def::Assign {
                        block: va,
                        index: i,
                    }
                };
                set_def(r.num, def, &mut defs);
            });
        }
    }
    Index {
        uses,
        defs,
        total_uses,
    }
}

// ---------------------------------------------------------------------------
// The lattice
// ---------------------------------------------------------------------------

/// A name's position in the lattice (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Value {
    /// No evidence yet — optimism, revisited when an input lowers.
    Top,
    /// Proven constant, at the definition's width, value masked to it.
    Const { value: u64, width: Width },
    /// Proven to hold the same runtime value as name `root`.
    Copy(u16),
    /// Varies, or unknowable.
    Bottom,
}

/// What following a name's copy chain lands on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Resolved {
    /// Still `Top`: no evidence, so the φ meet skips this argument.
    Top,
    /// The chain ends at this name, whose own value is opaque.
    Name(u16),
    /// The chain ends at a constant.
    Const { value: u64, width: Width },
    /// The walk exceeded the name count — a copy cycle. Unusable, so the
    /// meet that saw it lowers to `Bottom` (see the module's termination
    /// note).
    Cycle,
}

/// Follow `start`'s copy chain to its root. Bounded by the name count: a
/// chain visits a new name at every step, so more steps than names is a
/// cycle, and no input — including a transient mid-fixpoint one — loops.
fn resolve(values: &[Value], start: u16) -> Resolved {
    let mut cur = start;
    for _ in 0..=values.len() {
        match values.get(cur as usize) {
            // Defensively unreachable: ids index the name table.
            None => return Resolved::Name(cur),
            Some(Value::Top) => return Resolved::Top,
            Some(Value::Bottom) => return Resolved::Name(cur),
            Some(&Value::Const { value, width }) => return Resolved::Const { value, width },
            Some(&Value::Copy(next)) => cur = next,
        }
    }
    Resolved::Cycle
}

/// Meet over a φ's arguments: an argument equal to the φ's own destination
/// is skipped (`φ(x, self)`), a `Top` argument is skipped (optimism), and
/// the rest are resolved through their copy chains. All landing on the
/// same name gives a copy of it; all landing on constants equal in value
/// *and* width gives that constant; nothing left gives `Top`; anything
/// else — including a copy cycle — gives `Bottom`.
fn meet_phi(values: &[Value], dst: u16, args: &[(Option<u64>, u16)]) -> Value {
    let mut acc: Option<Resolved> = None;
    for &(_, arg) in args {
        if arg == dst {
            continue;
        }
        let r = resolve(values, arg);
        match r {
            Resolved::Top => continue,
            Resolved::Cycle => return Value::Bottom,
            // A chain back to the φ itself proves nothing.
            Resolved::Name(n) if n == dst => return Value::Bottom,
            _ => {}
        }
        match acc {
            None => acc = Some(r),
            Some(prev) if prev == r => {}
            Some(_) => return Value::Bottom,
        }
    }
    match acc {
        None => Value::Top,
        Some(Resolved::Name(r)) => Value::Copy(r),
        Some(Resolved::Const { value, width }) => Value::Const { value, width },
        // Skipped above; kept total rather than unreachable!().
        Some(_) => Value::Bottom,
    }
}

/// The lowering step, or `None` when `new` is not below `cur`. Every
/// accepted move descends: `Top` accepts anything, a `Known` value accepts
/// `Bottom` and the one refinement (a copy whose root proved constant),
/// and any other `Known → Known` change descends to `Bottom` rather than
/// oscillating — sound, and it keeps the per-name step count at most three.
fn lower(cur: Value, new: Value) -> Option<Value> {
    if cur == new {
        return None;
    }
    match (cur, new) {
        (Value::Bottom, _) => None,
        (_, Value::Top) => None,
        (Value::Top, v) => Some(v),
        (_, Value::Bottom) => Some(Value::Bottom),
        (Value::Copy(_), v @ Value::Const { .. }) => Some(v),
        _ => Some(Value::Bottom),
    }
}

// ---------------------------------------------------------------------------
// The transfer functions
// ---------------------------------------------------------------------------

/// Substitute proven values into `e` for *evaluation* only: the result is
/// inspected for shape, never emitted, so the emission-side guards
/// (temporaries, intrinsic reads) do not apply here. `saw_top` records
/// that some read had no evidence yet, which keeps the definition
/// optimistic instead of dropping it to `Bottom`.
fn eval_expr(
    f: &SsaFunction,
    values: &[Value],
    e: &Expr,
    depth: usize,
    saw_top: &mut bool,
) -> Expr {
    if depth > REWRITE_DEPTH {
        return e.clone();
    }
    match e {
        Expr::Const { .. } => e.clone(),
        Expr::Reg(r) => {
            let Some(n) = f.names.get(r.num as usize) else {
                return e.clone(); // defensively unreachable
            };
            if r.width.bits() > n.width.bits() {
                return e.clone(); // a `partial` read: the high bits are unknown
            }
            match resolve(values, r.num) {
                Resolved::Top => {
                    *saw_top = true;
                    e.clone()
                }
                Resolved::Const { value, .. } => Expr::constant(value, r.width),
                Resolved::Name(root) if root != r.num => match f.names.get(root as usize) {
                    Some(rn) if r.width.bits() <= rn.width.bits() => Expr::Reg(Reg {
                        space: rn.space,
                        num: root,
                        width: r.width,
                    }),
                    _ => e.clone(),
                },
                _ => e.clone(),
            }
        }
        Expr::Load { addr, width } => {
            Expr::load(eval_expr(f, values, addr, depth + 1, saw_top), *width)
        }
        Expr::Unary { op, operand } => {
            Expr::unary(*op, eval_expr(f, values, operand, depth + 1, saw_top))
        }
        Expr::Binary { op, lhs, rhs } => Expr::binary(
            *op,
            eval_expr(f, values, lhs, depth + 1, saw_top),
            eval_expr(f, values, rhs, depth + 1, saw_top),
        ),
    }
}

/// The value of an assignment's destination: substitute, fold, classify.
/// A folded constant at the destination's width is a constant fact; a
/// folded bare register no wider than that name guarantees is a copy fact
/// (unless that register itself has no value yet, in which case the
/// definition stays optimistic); anything else varies.
fn eval_assign(f: &SsaFunction, values: &[Value], dst: Reg, value: &Expr) -> Value {
    let mut saw_top = false;
    let folded = irflow::fold_expr(&eval_expr(f, values, value, 0, &mut saw_top));
    let unknown = if saw_top { Value::Top } else { Value::Bottom };
    match &folded {
        Expr::Const { value, width } if *width == dst.width => Value::Const {
            value: *value & width.mask(),
            width: *width,
        },
        Expr::Reg(r) if r.num != dst.num => {
            // The copy-record guard: a read wider than its definition
            // carries bits that definition never wrote, so it is no copy.
            let exact = f
                .names
                .get(r.num as usize)
                .is_some_and(|n| r.width.bits() <= n.width.bits());
            match (exact, resolve(values, r.num)) {
                (false, _) => unknown,
                (true, Resolved::Top) => Value::Top,
                (true, _) => Value::Copy(r.num),
            }
        }
        _ => unknown,
    }
}

/// Run the sparse worklist to its fixpoint, or `None` when the defensive
/// pop cap trips — in which case the caller applies *nothing*, since
/// optimistic mid-fixpoint facts are not sound to use.
fn solve(f: &SsaFunction, index: &Index) -> Option<Vec<Value>> {
    let mut values = vec![Value::Top; f.names.len()];
    let mut work: BTreeSet<u16> = BTreeSet::new();
    for id in 0..f.names.len() {
        let id = id as u16;
        if matches!(
            index.defs[id as usize],
            Def::Entry | Def::IntrinsicWrite { .. }
        ) {
            values[id as usize] = Value::Bottom;
        }
        work.insert(id);
    }
    // Provably unreachable: each name lowers at most three times, and each
    // lowering enqueues at most its use sites' definitions.
    let cap = 8usize
        .saturating_mul(f.names.len().saturating_add(index.total_uses))
        .saturating_add(8);
    let mut pops = 0usize;

    while let Some(id) = work.pop_first() {
        pops += 1;
        if pops > cap {
            return None;
        }
        let new = match index.defs[id as usize] {
            Def::Entry | Def::IntrinsicWrite { .. } => Value::Bottom,
            Def::Assign { block, index: i } => {
                match f.blocks.get(&block).and_then(|b| b.stmts.get(i)) {
                    Some(Stmt::Assign { dst, value }) => eval_assign(f, &values, *dst, value),
                    _ => Value::Bottom, // defensively unreachable
                }
            }
            Def::Phi { block, phi } => match f.blocks.get(&block).and_then(|b| b.phis.get(phi)) {
                Some(p) => meet_phi(&values, p.dst, &p.args),
                None => Value::Bottom, // defensively unreachable
            },
        };
        let Some(v) = lower(values[id as usize], new) else {
            continue;
        };
        values[id as usize] = v;
        // Re-evaluate every definition that reads this name. A use inside
        // a store, a branch, or an intrinsic read feeds no definition.
        let Some(sites) = index.uses.get(&id) else {
            continue;
        };
        for site in sites {
            match *site {
                UseSite::Stmt { block, index: i } => {
                    if let Some(Stmt::Assign { dst, .. }) =
                        f.blocks.get(&block).and_then(|b| b.stmts.get(i))
                    {
                        work.insert(dst.num);
                    }
                }
                UseSite::PhiArg { block, phi, .. } => {
                    if let Some(p) = f.blocks.get(&block).and_then(|b| b.phis.get(phi)) {
                        work.insert(p.dst);
                    }
                }
            }
        }
    }
    Some(values)
}

// ---------------------------------------------------------------------------
// Rewriting
// ---------------------------------------------------------------------------

/// One round's read-only view: the function, its index, and the fixpoint.
struct Pass<'a> {
    f: &'a SsaFunction,
    index: &'a Index,
    values: &'a [Value],
}

/// The block a name is defined in, or `None` for a version-0 name (whose
/// definition is the virtual function entry). Shared by the substituting
/// passes, which both need it for the temporary rule.
fn def_block(index: &Index, id: u16) -> Option<u64> {
    match index.defs.get(id as usize) {
        Some(&Def::Assign { block, .. })
        | Some(&Def::IntrinsicWrite { block, .. })
        | Some(&Def::Phi { block, .. }) => Some(block),
        _ => None,
    }
}

impl Pass<'_> {
    /// The block a name is defined in, or `None` for a version-0 name.
    fn def_block(&self, id: u16) -> Option<u64> {
        def_block(self.index, id)
    }

    /// Whether a substitution naming `root` may be emitted inside `block`.
    /// A temporary is write-before-read *within a block* per
    /// [`crate::ir::check`], so a cross-block temporary read would make the
    /// output fail [`irssa::check`]'s per-block IR validation.
    fn temp_ok(&self, root: u16, block: u64) -> bool {
        match self.f.names.get(root as usize).map(|n| n.space) {
            Some(Space::Temp) => self.def_block(root) == Some(block),
            Some(_) => true,
            None => false, // defensively unreachable
        }
    }

    /// The replacement expression for a use of `num` at `width` inside
    /// `block`, or `None` to keep the occurrence verbatim.
    fn replacement(&self, num: u16, width: Width, block: u64) -> Option<Expr> {
        let n = self.f.names.get(num as usize)?;
        if width.bits() > n.width.bits() {
            return None; // a `partial` read: never rewritten
        }
        match resolve(self.values, num) {
            Resolved::Const { value, .. } => Some(Expr::constant(value, width)),
            Resolved::Name(root) if root != num => {
                let rn = self.f.names.get(root as usize)?;
                if width.bits() > rn.width.bits() || !self.temp_ok(root, block) {
                    return None;
                }
                Some(Expr::Reg(Reg {
                    space: rn.space,
                    num: root,
                    width,
                }))
            }
            _ => None,
        }
    }

    /// Rewrite every eligible register read in `e`, counting replacements.
    fn rewrite_expr(&self, e: &Expr, block: u64, depth: usize, count: &mut usize) -> Expr {
        if depth > REWRITE_DEPTH {
            return e.clone();
        }
        match e {
            Expr::Const { .. } => e.clone(),
            Expr::Reg(r) => match self.replacement(r.num, r.width, block) {
                Some(x) => {
                    *count += 1;
                    x
                }
                None => e.clone(),
            },
            Expr::Load { addr, width } => {
                Expr::load(self.rewrite_expr(addr, block, depth + 1, count), *width)
            }
            Expr::Unary { op, operand } => {
                Expr::unary(*op, self.rewrite_expr(operand, block, depth + 1, count))
            }
            Expr::Binary { op, lhs, rhs } => Expr::binary(
                *op,
                self.rewrite_expr(lhs, block, depth + 1, count),
                self.rewrite_expr(rhs, block, depth + 1, count),
            ),
        }
    }

    /// Rewrite one statement's *use* expressions and re-fold it. The
    /// statement's kind, its definitions, and an [`crate::ir::Stmt::Intrinsic`] in
    /// its entirety are untouched; a statement nothing substituted into is
    /// returned verbatim, so folding never changes an input on its own.
    fn rewrite_stmt(&self, stmt: &Stmt, block: u64, count: &mut usize) -> Stmt {
        let mut here = 0usize;
        let rewritten = match stmt {
            // Intrinsic reads model "the callee observes this register";
            // the register identity is the point, so they stay verbatim.
            Stmt::Intrinsic { .. } => return stmt.clone(),
            Stmt::Assign { dst, value } => Stmt::Assign {
                dst: *dst,
                value: self.rewrite_expr(value, block, 0, &mut here),
            },
            Stmt::Store { addr, value } => Stmt::Store {
                addr: self.rewrite_expr(addr, block, 0, &mut here),
                value: self.rewrite_expr(value, block, 0, &mut here),
            },
            Stmt::Branch { kind, cond, target } => Stmt::Branch {
                kind: *kind,
                cond: cond
                    .as_ref()
                    .map(|c| self.rewrite_expr(c, block, 0, &mut here)),
                target: self.rewrite_expr(target, block, 0, &mut here),
            },
        };
        if here == 0 {
            return stmt.clone();
        }
        *count += here;
        // Substitution replaces a `Reg` node by a `Const` or `Reg` node and
        // folding only shrinks, so `ir::MAX_EXPR_NODES` holds by
        // construction.
        irflow::fold_stmt(&rewritten)
    }

    /// The replacement id for φ argument `arg` on predecessor edge `pred`
    /// of a φ over `cell`, or `None` to keep it verbatim. A φ merges one
    /// cell, so a copy root in another cell is not substitutable here —
    /// the φ's *destination*'s uses are what such a fact rewrites.
    fn phi_arg_replacement(&self, cell: Cell, pred: Option<u64>, arg: u16) -> Option<u16> {
        let pred = pred?; // the function-entry edge carries version 0
        let root = match resolve(self.values, arg) {
            Resolved::Name(r) if r != arg => r,
            _ => return None,
        };
        let an = self.f.names.get(arg as usize)?;
        let rn = self.f.names.get(root as usize)?;
        if (rn.space, rn.cell) != cell || rn.width.bits() < an.width.bits() {
            return None;
        }
        if !self.temp_ok(root, pred) {
            return None;
        }
        Some(root)
    }

    /// Whether every use of φ destination `dst` can be rewritten, so that
    /// removing the φ orphans nothing. Uses inside φs that are themselves
    /// going away (`pending`) do not need to be substitutable — that is
    /// what collapses a φ-of-φ cascade in one round.
    fn phi_removable(&self, dst: u16, pending: &BTreeSet<u16>) -> bool {
        let Some(sites) = self.index.uses.get(&dst) else {
            return true;
        };
        for site in sites {
            match *site {
                UseSite::Stmt { block, index: i } => {
                    let Some(stmt) = self.f.blocks.get(&block).and_then(|b| b.stmts.get(i)) else {
                        return false; // defensively unreachable
                    };
                    if matches!(stmt, Stmt::Intrinsic { .. }) {
                        return false; // intrinsic reads are never rewritten
                    }
                    let mut ok = true;
                    irssa::for_each_use(stmt, &mut |r| {
                        if r.num == dst && self.replacement(dst, r.width, block).is_none() {
                            ok = false;
                        }
                    });
                    if !ok {
                        return false;
                    }
                }
                UseSite::PhiArg { block, phi, arg } => {
                    let Some(p) = self.f.blocks.get(&block).and_then(|b| b.phis.get(phi)) else {
                        return false; // defensively unreachable
                    };
                    if pending.contains(&p.dst) {
                        continue;
                    }
                    let Some(cell) = self.f.names.get(p.dst as usize).map(|n| (n.space, n.cell))
                    else {
                        return false; // defensively unreachable
                    };
                    let pred = p.args.get(arg).map(|&(k, _)| k).unwrap_or(None);
                    if self.phi_arg_replacement(cell, pred, dst).is_none() {
                        return false;
                    }
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Name-table compaction
// ---------------------------------------------------------------------------

/// Rewrite every `Reg::num` in `e` through the old-id -> new-id map. An id
/// the map dropped is left alone (defensively unreachable: every use of a
/// removed φ's destination was substituted first).
fn remap_expr(e: &Expr, map: &[Option<u16>], depth: usize) -> Expr {
    if depth > REWRITE_DEPTH {
        return e.clone();
    }
    match e {
        Expr::Const { .. } => e.clone(),
        Expr::Reg(r) => match map.get(r.num as usize).copied().flatten() {
            Some(num) => Expr::Reg(Reg { num, ..*r }),
            None => e.clone(),
        },
        Expr::Load { addr, width } => Expr::load(remap_expr(addr, map, depth + 1), *width),
        Expr::Unary { op, operand } => Expr::unary(*op, remap_expr(operand, map, depth + 1)),
        Expr::Binary { op, lhs, rhs } => Expr::binary(
            *op,
            remap_expr(lhs, map, depth + 1),
            remap_expr(rhs, map, depth + 1),
        ),
    }
}

/// [`remap_expr`] for one register occurrence (a definition position).
fn remap_reg(r: Reg, map: &[Option<u16>]) -> Reg {
    match map.get(r.num as usize).copied().flatten() {
        Some(num) => Reg { num, ..r },
        None => r,
    }
}

/// [`remap_expr`] across a whole statement, uses and definitions alike.
fn remap_stmt(stmt: &Stmt, map: &[Option<u16>]) -> Stmt {
    match stmt {
        Stmt::Assign { dst, value } => Stmt::Assign {
            dst: remap_reg(*dst, map),
            value: remap_expr(value, map, 0),
        },
        Stmt::Store { addr, value } => Stmt::Store {
            addr: remap_expr(addr, map, 0),
            value: remap_expr(value, map, 0),
        },
        Stmt::Branch { kind, cond, target } => Stmt::Branch {
            kind: *kind,
            cond: cond.as_ref().map(|c| remap_expr(c, map, 0)),
            target: remap_expr(target, map, 0),
        },
        Stmt::Intrinsic {
            name,
            writes,
            reads,
        } => Stmt::Intrinsic {
            name,
            writes: writes.iter().map(|w| remap_reg(*w, map)).collect(),
            reads: reads.iter().map(|r| remap_expr(r, map, 0)).collect(),
        },
    }
}

/// Drop `removed` from the name table and renumber everything that names
/// an SSA id. Order-preserving, so version gaps stay (`check` requires
/// unique `(cell, version)` pairs, not dense ones) and `live_in` — which
/// version-0 names, never φ destinations, always survive — stays the exact
/// ascending version-0 set.
fn compact(f: SsaFunction, removed: &BTreeSet<u16>) -> SsaFunction {
    if removed.is_empty() {
        return f;
    }
    let mut map: Vec<Option<u16>> = vec![None; f.names.len()];
    let mut names: Vec<Name> = Vec::with_capacity(f.names.len() - removed.len());
    for (id, n) in f.names.iter().enumerate() {
        if removed.contains(&(id as u16)) {
            continue;
        }
        // The survivor count never exceeds the input's, whose ids fit a
        // `u16`, so the new id does too.
        map[id] = Some(names.len() as u16);
        names.push(*n);
    }
    let blocks: BTreeMap<u64, SsaBlock> = f
        .blocks
        .iter()
        .map(|(&va, b)| {
            (
                va,
                SsaBlock {
                    start: b.start,
                    end: b.end,
                    phis: b
                        .phis
                        .iter()
                        .map(|p| Phi {
                            dst: map.get(p.dst as usize).copied().flatten().unwrap_or(p.dst),
                            args: p
                                .args
                                .iter()
                                .map(|&(k, a)| {
                                    (k, map.get(a as usize).copied().flatten().unwrap_or(a))
                                })
                                .collect(),
                        })
                        .collect(),
                    stmts: b.stmts.iter().map(|s| remap_stmt(s, &map)).collect(),
                    successors: b.successors.clone(),
                    truncated: b.truncated,
                },
            )
        })
        .collect();
    let live_in: Vec<u16> = f
        .live_in
        .iter()
        .filter_map(|&id| map.get(id as usize).copied().flatten())
        .collect();
    SsaFunction {
        entry: f.entry,
        name: f.name,
        arch: f.arch,
        blocks,
        skipped: f.skipped,
        names,
        live_in,
        partial: f.partial,
    }
}

/// Recompute [`SsaFunction::partial`] from scratch, exactly as
/// [`irssa::construct`] does: positions holding a read wider than its
/// definition guarantees. A rewrite can only remove such a position, never
/// add one, but the list is rebuilt rather than adjusted.
fn recompute_partial(f: &mut SsaFunction) {
    let mut partial: BTreeSet<(u64, usize)> = BTreeSet::new();
    for (&va, block) in &f.blocks {
        for (i, stmt) in block.stmts.iter().enumerate() {
            irssa::for_each_use(stmt, &mut |r| {
                if let Some(n) = f.names.get(r.num as usize)
                    && r.width.bits() > n.width.bits()
                {
                    partial.insert((va, i));
                }
            });
        }
    }
    f.partial = partial.into_iter().collect();
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

/// What one [`optimize`] call did. Deterministic counters; `capped` means
/// the defensive worklist bound tripped and **the input was returned
/// unchanged**, never partially optimized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    /// Outer rounds run, the last of which is the one that changed
    /// nothing and so proved the fixpoint. Zero only for an input
    /// [`irssa::check`] rejected.
    pub rounds: usize,
    /// Use occurrences and φ arguments replaced.
    pub rewrites: usize,
    /// φ-nodes removed.
    pub phis_removed: usize,
    /// Names dropped by the compaction (one per removed φ).
    pub names_removed: usize,
    /// The pop cap tripped; the output is the input, unoptimized.
    pub capped: bool,
}

/// One round's counters.
#[derive(Debug, Clone, Copy, Default)]
struct Round {
    rewrites: usize,
    phis_removed: usize,
    names_removed: usize,
}

/// Build the index, solve, rewrite, remove φs, compact — one round.
/// `None` means the worklist cap tripped.
fn round(f: &SsaFunction) -> Option<(SsaFunction, Round)> {
    let index = build_index(f);
    let values = solve(f, &index)?;
    let pass = Pass {
        f,
        index: &index,
        values: &values,
    };
    let mut stats = Round::default();

    // The φs whose fixpoint value is a copy: `φ(x,…,x)` and `φ(x, self)`,
    // generalized through copy chains. Start optimistic and drop the ones
    // whose uses cannot all be rewritten; a drop can only cost another
    // candidate its "the φ reading me is going away" exemption, so the
    // loop shrinks the set monotonically and is bounded by its size.
    let mut pending: BTreeSet<u16> = BTreeSet::new();
    for block in f.blocks.values() {
        for phi in &block.phis {
            if matches!(values.get(phi.dst as usize), Some(Value::Copy(_))) {
                pending.insert(phi.dst);
            }
        }
    }
    for _ in 0..=pending.len() {
        let mut dropped = false;
        for dst in pending.clone() {
            if !pass.phi_removable(dst, &pending) {
                pending.remove(&dst);
                dropped = true;
            }
        }
        if !dropped {
            break;
        }
    }

    let mut blocks: BTreeMap<u64, SsaBlock> = BTreeMap::new();
    for (&va, block) in &f.blocks {
        let stmts: Vec<Stmt> = block
            .stmts
            .iter()
            .map(|s| pass.rewrite_stmt(s, va, &mut stats.rewrites))
            .collect();
        let mut phis: Vec<Phi> = Vec::with_capacity(block.phis.len());
        for phi in &block.phis {
            if pending.contains(&phi.dst) {
                stats.phis_removed += 1;
                continue;
            }
            let Some(cell) = f.names.get(phi.dst as usize).map(|n| (n.space, n.cell)) else {
                phis.push(phi.clone()); // defensively unreachable
                continue;
            };
            let args = phi
                .args
                .iter()
                .map(|&(k, a)| match pass.phi_arg_replacement(cell, k, a) {
                    Some(root) => {
                        stats.rewrites += 1;
                        (k, root)
                    }
                    None => (k, a),
                })
                .collect();
            phis.push(Phi { dst: phi.dst, args });
        }
        blocks.insert(
            va,
            SsaBlock {
                start: block.start,
                end: block.end,
                phis,
                stmts,
                successors: block.successors.clone(),
                truncated: block.truncated,
            },
        );
    }

    stats.names_removed = pending.len();
    let out = SsaFunction {
        entry: f.entry,
        name: f.name.clone(),
        arch: f.arch,
        blocks,
        skipped: f.skipped.clone(),
        names: f.names.clone(),
        live_in: f.live_in.clone(),
        partial: f.partial.clone(),
    };
    let mut out = compact(out, &pending);
    recompute_partial(&mut out);
    Some((out, stats))
}

/// Propagate constants and copies through `func` and remove the φs that
/// merge one value, returning a new function and what was done.
///
/// Pure, total, deterministic, and never panicking — see the module docs
/// for the lattice, the soundness rules, and the two ways the pass
/// declines: an input that fails [`irssa::check`] is returned unchanged
/// with `rounds = 0` (malformed input is not laundered), and a tripped
/// worklist cap returns the input unchanged with `capped = true` (partial
/// optimistic facts are not sound to apply).
pub fn optimize(func: &SsaFunction) -> (SsaFunction, Stats) {
    let mut stats = Stats::default();
    if irssa::check(func).is_err() {
        return (func.clone(), stats);
    }
    let mut cur = func.clone();
    for _ in 0..MAX_ROUNDS {
        stats.rounds += 1;
        let Some((next, r)) = round(&cur) else {
            return (
                func.clone(),
                Stats {
                    rounds: stats.rounds,
                    capped: true,
                    ..Stats::default()
                },
            );
        };
        stats.rewrites += r.rewrites;
        stats.phis_removed += r.phis_removed;
        stats.names_removed += r.names_removed;
        if next == cur {
            break;
        }
        cur = next;
    }
    (cur, stats)
}

// ---------------------------------------------------------------------------
// Expression forwarding
// ---------------------------------------------------------------------------

/// What one [`forward`] call did. Deterministic counters; all zero for an
/// input [`irssa::check`] rejected, which is returned unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FwdStats {
    /// Rounds run, the last of which is the one that changed nothing and so
    /// proved the fixpoint — unless `capped`, when the bound ran out first.
    pub rounds: usize,
    /// Use occurrences rewritten: one per spliced tree.
    pub forwards: usize,
    /// Of `forwards`, the splices made under the fold-shrinks exception:
    /// a pure tree past `FWD_SMALL_NODES` copied into several uses
    /// because every tentatively folded use site came out strictly
    /// smaller — the paired flag shapes collapsing to one relation is the
    /// motivating case.
    pub multi_spliced: usize,
    /// Of `forwards`, the splices made under the load-cone joint
    /// exception: a load-bearing definition (its block-local load temps
    /// inlined into the cone) copied into *every* one of its uses at
    /// once — all-or-nothing, so the definition is guaranteed to sweep
    /// and the load is never rendered both at its definition and
    /// inline — under provably effect-clear windows and a function-level
    /// strict shrink. The memory-operand `cmp` feeding two jccs is the
    /// motivating case.
    pub load_pair_spliced: usize,
    /// Substitutions refused because the statement would have exceeded
    /// [`crate::ir::MAX_EXPR_NODES`]. The refusals *standing at the
    /// fixpoint* (the last round's count), not a running total: a refusal
    /// is re-evaluated every round, so summing would multiply it by the
    /// round count.
    pub size_skipped: usize,
    /// The round bound ran out. The output is the last completed round's —
    /// every intermediate forwarding state is sound, so unlike
    /// [`optimize`]'s optimistic lattice there is nothing to roll back.
    pub capped: bool,
}

/// One forwarding round's counters.
#[derive(Debug, Clone, Copy, Default)]
struct FwdRound {
    forwards: usize,
    multi_spliced: usize,
    load_pair_spliced: usize,
    size_skipped: usize,
}

/// The node count of `e`, or a value above [`crate::ir::MAX_EXPR_NODES`]
/// when the walk hits its depth bound — a tree that deep is refused, never
/// measured short.
fn expr_nodes(e: &Expr, depth: usize) -> usize {
    if depth > REWRITE_DEPTH {
        return ir::MAX_EXPR_NODES + 1;
    }
    match e {
        Expr::Const { .. } | Expr::Reg(_) => 1,
        Expr::Load { addr, .. } => 1 + expr_nodes(addr, depth + 1),
        Expr::Unary { operand, .. } => 1 + expr_nodes(operand, depth + 1),
        Expr::Binary { lhs, rhs, .. } => {
            1 + expr_nodes(lhs, depth + 1) + expr_nodes(rhs, depth + 1)
        }
    }
}

// `contains_div` — whether a tree holds a division or remainder, whose
// position relative to the guards is load-bearing — moved next to
// [`irflow::contains_load`] when the value-numbering witness
// ([`irflow::VnDefs`]) began sharing it: one predicate, one doctrine.

/// The [`Space::Temp`] names `e` reads. A spliced tree may only land in a
/// block that defines all of them (see the module docs).
fn temps_read(e: &Expr) -> BTreeSet<u16> {
    let mut set = BTreeSet::new();
    irssa::expr_regs(e, 0, &mut |r| {
        if r.space == Space::Temp {
            set.insert(r.num);
        }
    });
    set
}

/// How many times `id` occurs in use position anywhere in the function, φ
/// arguments included. The index keys use *sites*, and one statement may
/// read a name twice, so the occurrence count is taken here.
fn use_occurrences(f: &SsaFunction, index: &Index, id: u16) -> usize {
    let Some(sites) = index.uses.get(&id) else {
        return 0;
    };
    let mut n = 0usize;
    for site in sites {
        match *site {
            UseSite::PhiArg { .. } => n += 1,
            UseSite::Stmt { block, index: i } => {
                if let Some(stmt) = f.blocks.get(&block).and_then(|b| b.stmts.get(i)) {
                    irssa::for_each_use(stmt, &mut |r| {
                        if r.num == id {
                            n += 1;
                        }
                    });
                }
            }
        }
    }
    n
}

/// The barrier a load-bearing tree may not cross: [`irflow::propagate`]'s
/// set, lifted onto SSA statements.
fn effect_barrier(stmt: &Stmt) -> bool {
    matches!(
        stmt,
        Stmt::Store { .. } | Stmt::Intrinsic { .. } | Stmt::Branch { .. }
    )
}

/// The barrier a trapping tree may not cross: a guard.
fn branch_barrier(stmt: &Stmt) -> bool {
    matches!(stmt, Stmt::Branch { .. })
}

/// The substitutions one round will make: the tree per forwardable name,
/// and the exact use positions it may land in.
#[derive(Default)]
struct FwdPlan {
    /// Name id -> the right-hand side to splice.
    rhs: BTreeMap<u16, Expr>,
    /// `(block, statement index, name)` triples cleared for substitution.
    sites: BTreeSet<(u64, usize, u16)>,
    /// Names cleared under the fold-shrinks exception — a multi-use pure
    /// definition past [`FWD_SMALL_NODES`] whose every cleared site
    /// tentatively folded strictly smaller — counted separately in
    /// [`FwdStats::multi_spliced`].
    multi: BTreeSet<u16>,
    /// Names cleared under the load-cone joint exception — a load-bearing
    /// cone whose every use cleared together, all-or-nothing, under the
    /// function-level shrink — counted in [`FwdStats::load_pair_spliced`].
    load_multi: BTreeSet<u16>,
}

/// Decide, from the current function, every substitution this round may
/// make. Deterministic: the walk is in block then index order, and every
/// container is a `BTree`.
fn plan_forwards(f: &SsaFunction, index: &Index, vn: &irflow::VnDefs) -> FwdPlan {
    let mut plan = FwdPlan::default();
    for (&dva, dblock) in &f.blocks {
        for (di, stmt) in dblock.stmts.iter().enumerate() {
            let Stmt::Assign { dst, value } = stmt else {
                continue;
            };
            // A trivial right-hand side is `optimize`'s to substitute; the
            // pipeline order handles it, so this pass does not.
            if matches!(value, Expr::Const { .. } | Expr::Reg(_)) {
                continue;
            }
            let nodes = expr_nodes(value, 0);
            if nodes > ir::MAX_EXPR_NODES {
                continue; // defensively unreachable on checked input
            }
            let loads = irflow::contains_load(value, 0);
            let divs = irflow::contains_div(value, 0);
            let temps = temps_read(value);
            let single = use_occurrences(f, index, dst.num) == 1;
            // The duplication cap: a load-bearing tree, or one too big to
            // read twice, moves only when it moves exactly once — except
            // that a *pure* tree past the size cap may still earn several
            // sites below, by proving every folded site shrinks. A load or
            // a division never duplicates, whatever the fold would do.
            let earn = !single && !loads && !divs && nodes > FWD_SMALL_NODES;
            if (loads || nodes > FWD_SMALL_NODES) && !single && !earn {
                continue;
            }
            let Some(sites) = index.uses.get(&dst.num) else {
                continue;
            };
            let mut cleared: Vec<(u64, usize)> = Vec::new();
            for site in sites {
                // A φ argument names a version, not an expression.
                let UseSite::Stmt { block, index: ui } = *site else {
                    continue;
                };
                let Some(ublock) = f.blocks.get(&block) else {
                    continue; // defensively unreachable
                };
                match ublock.stmts.get(ui) {
                    // An intrinsic read models "the callee observes this
                    // register"; the register identity is the point.
                    Some(Stmt::Intrinsic { .. }) | None => continue,
                    Some(_) => {}
                }
                // A temporary is write-before-read within a block, so a
                // tree reading one may only land where it is written.
                if temps
                    .iter()
                    .any(|&t| def_block(index, t) != Some(block))
                {
                    continue;
                }
                if loads || divs {
                    // Both positional tiers need the use to sit after the
                    // definition in the definition's own block.
                    if block != dva || ui <= di {
                        continue;
                    }
                    let between = &dblock.stmts[di + 1..ui];
                    if loads && between.iter().any(effect_barrier) {
                        continue;
                    }
                    if divs && between.iter().any(branch_barrier) {
                        continue;
                    }
                }
                cleared.push((block, ui));
            }
            if earn {
                // Splice-when-the-fold-shrinks: tentatively substitute at
                // each cleared site and fold. A site the tentative cannot
                // splice proves nothing and is dropped; every site it can
                // splice must come out strictly smaller, or the whole
                // definition is refused — the per-def decision, taken
                // deterministically from the folded results.
                let mut earned: Vec<(u64, usize)> = Vec::new();
                let mut grows = false;
                for &(block, ui) in &cleared {
                    match tentative_fold(f, vn, dst.num, value, block, ui) {
                        None => {}
                        Some(true) => earned.push((block, ui)),
                        Some(false) => {
                            grows = true;
                            break;
                        }
                    }
                }
                if grows {
                    continue;
                }
                cleared = earned;
                if !cleared.is_empty() {
                    plan.multi.insert(dst.num);
                }
            }
            if !cleared.is_empty() {
                for (block, ui) in cleared {
                    plan.sites.insert((block, ui, dst.num));
                }
                plan.rhs.insert(dst.num, value.clone());
            }
        }
    }
    // The load-cone joint rescue picks up what the tiers above could not
    // fully clear; it only ever replaces a *partial* claim of its own
    // members, never another definition's.
    plan_load_pairs(f, index, &mut plan, vn);
    plan
}

/// One forwarding round's read-only view: the function, its plan, and the
/// value-numbering context the re-fold's equality witness reads.
struct Fwd<'a> {
    f: &'a SsaFunction,
    plan: &'a FwdPlan,
    vn: &'a irflow::VnDefs,
}

/// The round's [`irflow::VnDefs`]: every pure assignment, keyed by its
/// defining register. The purity gate (load-free, division-free,
/// node-capped) lives in [`irflow::VnDefs::add`]; a φ has no right-hand
/// side and is naturally absent.
fn build_vn(f: &SsaFunction) -> irflow::VnDefs {
    let mut vn = irflow::VnDefs::new();
    for block in f.blocks.values() {
        for stmt in &block.stmts {
            if let Stmt::Assign { dst, value } = stmt {
                vn.add(*dst, value);
            }
        }
    }
    vn
}

impl Fwd<'_> {
    /// The tree to splice for a read of `r` in statement `index` of
    /// `block`, or `None` to keep the occurrence verbatim.
    fn splice(&self, r: Reg, block: u64, index: usize) -> Option<&Expr> {
        // Exact-width reads only: a narrower read keeps the name.
        let n = self.f.names.get(r.num as usize)?;
        if r.width != n.width || !self.plan.sites.contains(&(block, index, r.num)) {
            return None;
        }
        self.plan.rhs.get(&r.num)
    }

    /// Splice into every cleared read in `e`. `budget` is how many nodes
    /// this expression may still grow by; a substitution that would exceed
    /// it is refused and counted, never truncated. The spliced tree is not
    /// walked into — a cascade is the next round's business.
    fn rewrite_expr(
        &self,
        e: &Expr,
        block: u64,
        index: usize,
        budget: &mut usize,
        r: &mut FwdRound,
        depth: usize,
    ) -> Expr {
        if depth > REWRITE_DEPTH {
            return e.clone();
        }
        match e {
            Expr::Const { .. } => e.clone(),
            Expr::Reg(reg) => match self.splice(*reg, block, index) {
                Some(rhs) => {
                    // The splice replaces one node with the whole tree.
                    let added = expr_nodes(rhs, 0).saturating_sub(1);
                    if added > *budget {
                        r.size_skipped += 1;
                        return e.clone();
                    }
                    *budget -= added;
                    r.forwards += 1;
                    if self.plan.multi.contains(&reg.num) {
                        r.multi_spliced += 1;
                    }
                    if self.plan.load_multi.contains(&reg.num) {
                        r.load_pair_spliced += 1;
                    }
                    rhs.clone()
                }
                None => e.clone(),
            },
            Expr::Load { addr, width } => Expr::load(
                self.rewrite_expr(addr, block, index, budget, r, depth + 1),
                *width,
            ),
            Expr::Unary { op, operand } => Expr::unary(
                *op,
                self.rewrite_expr(operand, block, index, budget, r, depth + 1),
            ),
            Expr::Binary { op, lhs, rhs: rh } => Expr::binary(
                *op,
                self.rewrite_expr(lhs, block, index, budget, r, depth + 1),
                self.rewrite_expr(rh, block, index, budget, r, depth + 1),
            ),
        }
    }

    /// [`Fwd::rewrite_expr`] on one top-level expression, under its own
    /// node budget — [`crate::ir::check`] bounds each expression, not the
    /// statement, so the budget is per expression too.
    fn rewrite_top(&self, e: &Expr, block: u64, index: usize, r: &mut FwdRound) -> Expr {
        let mut budget = ir::MAX_EXPR_NODES.saturating_sub(expr_nodes(e, 0));
        self.rewrite_expr(e, block, index, &mut budget, r, 0)
    }

    /// Rewrite one statement's *use* expressions and re-fold it. Kind,
    /// destinations, and intrinsics in their entirety are untouched; a
    /// statement nothing was spliced into is returned verbatim, so folding
    /// never changes an input on its own.
    fn rewrite_stmt(&self, stmt: &Stmt, block: u64, index: usize, r: &mut FwdRound) -> Stmt {
        let before = r.forwards;
        let out = match stmt {
            Stmt::Intrinsic { .. } => return stmt.clone(),
            Stmt::Assign { dst, value } => Stmt::Assign {
                dst: *dst,
                value: self.rewrite_top(value, block, index, r),
            },
            Stmt::Store { addr, value } => Stmt::Store {
                addr: self.rewrite_top(addr, block, index, r),
                value: self.rewrite_top(value, block, index, r),
            },
            Stmt::Branch { kind, cond, target } => Stmt::Branch {
                kind: *kind,
                cond: cond
                    .as_ref()
                    .map(|c| self.rewrite_top(c, block, index, r)),
                target: self.rewrite_top(target, block, index, r),
            },
        };
        if r.forwards == before {
            return stmt.clone();
        }
        irflow::fold_stmt_vn(&out, self.vn)
    }
}

/// The statement's total expression size: the sum of [`expr_nodes`] over
/// its top-level expressions — the measure the fold-shrinks exception
/// compares across a tentative splice. An intrinsic holds no spliceable
/// expression and never reaches the comparison; its size is zero.
fn stmt_nodes(stmt: &Stmt) -> usize {
    match stmt {
        Stmt::Assign { value, .. } => expr_nodes(value, 0),
        Stmt::Store { addr, value } => expr_nodes(addr, 0) + expr_nodes(value, 0),
        Stmt::Branch { cond, target, .. } => {
            cond.as_ref().map_or(0, |c| expr_nodes(c, 0)) + expr_nodes(target, 0)
        }
        Stmt::Intrinsic { .. } => 0,
    }
}

/// The fold-shrinks tentative for one candidate site: splice `value` for
/// every exact-width read of `id` in the statement at (`block`, `ui`) and
/// fold, through the very [`Fwd::rewrite_stmt`] the real round runs, then
/// answer whether the statement came out strictly smaller than it stands.
/// `None` means no splice happens there at all — the statement reads the
/// name only at narrower widths, or the node budget refuses — so the site
/// proves nothing either way and is simply not cleared.
fn tentative_fold(
    f: &SsaFunction,
    vn: &irflow::VnDefs,
    id: u16,
    value: &Expr,
    block: u64,
    ui: usize,
) -> Option<bool> {
    let stmt = f.blocks.get(&block)?.stmts.get(ui)?;
    let mut plan = FwdPlan::default();
    plan.rhs.insert(id, value.clone());
    plan.sites.insert((block, ui, id));
    let fwd = Fwd { f, plan: &plan, vn };
    let mut r = FwdRound::default();
    let folded = fwd.rewrite_stmt(stmt, block, ui, &mut r);
    if r.forwards == 0 {
        return None;
    }
    Some(stmt_nodes(&folded) < stmt_nodes(stmt))
}

/// The load-cone joint exception's caps: a load may render at most this
/// many inline copies (the one-`cmp`-two-jccs shape is the target), and
/// the cross-block effect-clear region may span at most this many blocks.
/// An over-cap shape is refused and keeps today's output — never
/// approximated.
const MAX_LOAD_SPLICE_SITES: usize = 2;
const MAX_LOAD_REGION_BLOCKS: usize = 4;

/// One load-cone rescue candidate: a definition the main plan could not
/// fully clear, whose tree — its block-local temps inlined transitively —
/// is a temp-free, load-bearing cone.
struct LoadCand {
    id: u16,
    dva: u64,
    di: usize,
    cone: Expr,
    /// The temps the cone inlined; their definitions die with this one
    /// when nothing else reads them.
    inlined: BTreeSet<u16>,
    /// The distinct use sites, all statement positions.
    sites: BTreeSet<(u64, usize)>,
}

/// Build the definition's *cone*: the tree with every block-local temp
/// read replaced by its defining right-hand side, transitively. A spliced
/// copy must be temp-free to land in another block, and a temp backed by a
/// load is exactly what the pure cascade cannot fold in. `first_load` is
/// lowered to the earliest statement index whose load the cone re-executes;
/// `inlined` records the temps. `None` refuses: a temp defined by anything
/// but a plain same-block assignment, a width-changing read, or a walk past
/// the depth bound.
fn cone_expr(
    f: &SsaFunction,
    index: &Index,
    dva: u64,
    e: &Expr,
    first_load: &mut usize,
    inlined: &mut BTreeSet<u16>,
    depth: usize,
) -> Option<Expr> {
    if depth > REWRITE_DEPTH {
        return None;
    }
    match e {
        Expr::Const { .. } => Some(e.clone()),
        Expr::Reg(r) => {
            if f.names.get(r.num as usize).map(|n| n.space) != Some(Space::Temp) {
                return Some(e.clone());
            }
            let Some(&Def::Assign { block, index: ti }) = index.defs.get(r.num as usize)
            else {
                return None;
            };
            if block != dva {
                return None; // defensively unreachable: temps are block-local
            }
            let Some(Stmt::Assign { dst, value }) =
                f.blocks.get(&dva).and_then(|b| b.stmts.get(ti))
            else {
                return None;
            };
            if r.width != dst.width {
                return None; // a narrower read keeps the name
            }
            let sub = cone_expr(f, index, dva, value, first_load, inlined, depth + 1)?;
            if irflow::contains_load(&sub, 0) {
                *first_load = (*first_load).min(ti);
            }
            inlined.insert(r.num);
            Some(sub)
        }
        Expr::Load { addr, width } => Some(Expr::load(
            cone_expr(f, index, dva, addr, first_load, inlined, depth + 1)?,
            *width,
        )),
        Expr::Unary { op, operand } => Some(Expr::unary(
            *op,
            cone_expr(f, index, dva, operand, first_load, inlined, depth + 1)?,
        )),
        Expr::Binary { op, lhs, rhs } => Some(Expr::binary(
            *op,
            cone_expr(f, index, dva, lhs, first_load, inlined, depth + 1)?,
            cone_expr(f, index, dva, rhs, first_load, inlined, depth + 1)?,
        )),
    }
}

/// The cross-block window: every statement on every def→use path between
/// the earliest re-executed load and the use must be effect-free, so the
/// re-read is provably the value the definition saw. The region — every
/// block on any recorded path from `a` to `b` — covers every dynamic
/// path because the definition dominates the use: an execution reaching
/// the use passed the definition and then stayed on blocks that reach `b`.
/// Recorded successors are the CFG's semantics here exactly as they are
/// for [`eliminate_dead`]'s liveness and [`optimize`]'s lattice — an
/// unproven indirect jump has no recorded edges and is an honest exit.
/// Refused, never approximated: a region past [`MAX_LOAD_REGION_BLOCKS`],
/// a cyclic region (a re-arrival at the use would re-read after the
/// region's own statements — this also catches every recorded return path
/// into `b`), a `truncated` region block (its unlifted statements are
/// unscannable), a `Store`/`Intrinsic` in the window, or a
/// `Call`/`Return` terminator inside the region.
fn effect_clear_region(f: &SsaFunction, a: u64, first_load: usize, b: u64, ui: usize) -> bool {
    // Forward reachability from `a` over recorded edges.
    let mut fwd: BTreeSet<u64> = BTreeSet::new();
    let mut queue = vec![a];
    while let Some(v) = queue.pop() {
        if !fwd.insert(v) {
            continue;
        }
        let Some(block) = f.blocks.get(&v) else {
            continue; // an out-of-function target: no edges to follow
        };
        for &s in &block.successors {
            if f.blocks.contains_key(&s) {
                queue.push(s);
            }
        }
    }
    if !fwd.contains(&b) {
        return false;
    }
    // Backward reachability to `b`.
    let mut preds: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for (&va, block) in &f.blocks {
        for &s in &block.successors {
            preds.entry(s).or_default().push(va);
        }
    }
    let mut back: BTreeSet<u64> = BTreeSet::new();
    let mut queue = vec![b];
    while let Some(v) = queue.pop() {
        if !back.insert(v) {
            continue;
        }
        if let Some(ps) = preds.get(&v) {
            for &p in ps {
                queue.push(p);
            }
        }
    }
    let region: BTreeSet<u64> = fwd.intersection(&back).copied().collect();
    if !region.contains(&a) || !region.contains(&b) || region.len() > MAX_LOAD_REGION_BLOCKS {
        return false;
    }
    // Acyclicity within the region, by Kahn's algorithm on the
    // region-restricted edges.
    let mut indeg: BTreeMap<u64, usize> = region.iter().map(|&v| (v, 0)).collect();
    for &v in &region {
        if let Some(block) = f.blocks.get(&v) {
            for &s in &block.successors {
                if let Some(d) = indeg.get_mut(&s) {
                    *d += 1;
                }
            }
        }
    }
    let mut ready: Vec<u64> = indeg
        .iter()
        .filter(|&(_, &d)| d == 0)
        .map(|(&v, _)| v)
        .collect();
    let mut seen = 0usize;
    while let Some(v) = ready.pop() {
        seen += 1;
        if let Some(block) = f.blocks.get(&v) {
            for &s in &block.successors {
                if let Some(d) = indeg.get_mut(&s) {
                    *d -= 1;
                    if *d == 0 {
                        ready.push(s);
                    }
                }
            }
        }
    }
    if seen != region.len() {
        return false;
    }
    // The statement scan: `a` after the load, intermediates whole, `b`
    // before the use. Plain jumps are pure control; everything else that
    // is not an assignment touches state.
    let window_ok = |stmts: &[Stmt]| {
        stmts.iter().all(|s| {
            matches!(s, Stmt::Assign { .. })
                || matches!(
                    s,
                    Stmt::Branch {
                        kind: BranchKind::Jump,
                        ..
                    }
                )
        })
    };
    for &v in &region {
        let Some(block) = f.blocks.get(&v) else {
            return false;
        };
        if block.truncated {
            return false; // unlifted statements cannot be scanned
        }
        let ok = if v == a {
            block.stmts.get(first_load + 1..).is_some_and(window_ok)
        } else if v == b {
            block.stmts.get(..ui).is_some_and(window_ok)
        } else {
            window_ok(&block.stmts)
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Whether the cone's loads may re-execute at (`ub`, `ui`): the same-block
/// between-scan from the earliest load, or the cross-block effect-clear
/// region.
fn load_window_ok(f: &SsaFunction, dva: u64, first_load: usize, ub: u64, ui: usize) -> bool {
    if ub == dva {
        if ui <= first_load {
            return false;
        }
        let Some(block) = f.blocks.get(&dva) else {
            return false;
        };
        let Some(between) = block.stmts.get(first_load + 1..ui) else {
            return false;
        };
        return !between.iter().any(effect_barrier);
    }
    effect_clear_region(f, dva, first_load, ub, ui)
}

/// The load-cone joint rescue: definitions the main plan could not fully
/// clear because a load stands in the way — the `cmp`-with-a-memory-operand
/// feeding two jccs is the type specimen — earn their sites *jointly,
/// all-or-nothing, under a function-level strict shrink*. Every use of
/// every member must clear (the definition is guaranteed to sweep, so a
/// load is never rendered both at its definition and inline), every window
/// must be provably effect-clear, the joint tentative fold — the group
/// substituted together, so the pair is judged as the pair — must leave no
/// member name standing at any site, and the whole-statement accounting
/// (folded sites versus standing sites plus the dying definitions and
/// their exclusively-owned temps) must strictly shrink. The per-site
/// textual test cannot see this shape: the site grows by an inline load
/// spelling while the two flag definitions die.
fn plan_load_pairs(f: &SsaFunction, index: &Index, plan: &mut FwdPlan, vn: &irflow::VnDefs) {
    let mut cands: Vec<LoadCand> = Vec::new();
    for (&dva, dblock) in &f.blocks {
        for (di, stmt) in dblock.stmts.iter().enumerate() {
            let Stmt::Assign { dst, value } = stmt else {
                continue;
            };
            if matches!(value, Expr::Const { .. } | Expr::Reg(_)) {
                continue;
            }
            let Some(usesites) = index.uses.get(&dst.num) else {
                continue;
            };
            // Every use must be a statement read at the definition's
            // exact width — a φ argument, an intrinsic read, or a
            // narrower read cannot clear, and a partial clearing would
            // duplicate the load without sweeping the definition.
            let mut sites: BTreeSet<(u64, usize)> = BTreeSet::new();
            let mut ok = true;
            for site in usesites {
                let UseSite::Stmt { block, index: ui } = *site else {
                    ok = false;
                    break;
                };
                match f.blocks.get(&block).and_then(|blk| blk.stmts.get(ui)) {
                    Some(Stmt::Intrinsic { .. }) | None => {
                        ok = false;
                        break;
                    }
                    Some(u) => {
                        let mut widths_ok = true;
                        irssa::for_each_use(u, &mut |r| {
                            if r.num == dst.num && r.width != dst.width {
                                widths_ok = false;
                            }
                        });
                        if !widths_ok {
                            ok = false;
                            break;
                        }
                        sites.insert((block, ui));
                    }
                }
            }
            if !ok || sites.is_empty() || sites.len() > MAX_LOAD_SPLICE_SITES {
                continue;
            }
            // The main plan finished this definition: keep its claim.
            if sites
                .iter()
                .all(|&(b, ui)| plan.sites.contains(&(b, ui, dst.num)))
            {
                continue;
            }
            let mut first_load = if irflow::contains_load(value, 0) {
                di
            } else {
                usize::MAX
            };
            let mut inlined = BTreeSet::new();
            let Some(cone) = cone_expr(f, index, dva, value, &mut first_load, &mut inlined, 0)
            else {
                continue;
            };
            if !irflow::contains_load(&cone, 0)
                || irflow::contains_div(&cone, 0)
                || expr_nodes(&cone, 0) > ir::MAX_EXPR_NODES
                || first_load == usize::MAX
            {
                continue;
            }
            if !sites
                .iter()
                .all(|&(b, ui)| load_window_ok(f, dva, first_load, b, ui))
            {
                continue;
            }
            cands.push(LoadCand {
                id: dst.num,
                dva,
                di,
                cone,
                inlined,
                sites,
            });
        }
    }
    if cands.is_empty() {
        return;
    }
    // Group the candidates that share a site — the pair earns together or
    // not at all — and try each group in ascending name order.
    let mut by_site: BTreeMap<(u64, usize), Vec<usize>> = BTreeMap::new();
    for (i, c) in cands.iter().enumerate() {
        for &s in &c.sites {
            by_site.entry(s).or_default().push(i);
        }
    }
    let mut assigned = vec![false; cands.len()];
    for i in 0..cands.len() {
        if assigned[i] {
            continue;
        }
        assigned[i] = true;
        let mut group = vec![i];
        let mut queue = vec![i];
        while let Some(k) = queue.pop() {
            for &s in &cands[k].sites {
                for &j in by_site.get(&s).into_iter().flatten() {
                    if !assigned[j] {
                        assigned[j] = true;
                        group.push(j);
                        queue.push(j);
                    }
                }
            }
        }
        group.sort_unstable();
        try_load_group(f, index, plan, &cands, &group, vn);
    }
}

/// One group's joint tentative and, on success, its committed claim. See
/// [`plan_load_pairs`] for the contract.
fn try_load_group(
    f: &SsaFunction,
    index: &Index,
    plan: &mut FwdPlan,
    cands: &[LoadCand],
    group: &[usize],
    vn: &irflow::VnDefs,
) {
    let names: BTreeSet<u16> = group.iter().map(|&k| cands[k].id).collect();
    let gsites: BTreeSet<(u64, usize)> = group
        .iter()
        .flat_map(|&k| cands[k].sites.iter().copied())
        .collect();
    // A site the main plan already claimed for an outside name would make
    // the joint tentative diverge from the real round; leave it be.
    for &(b, ui) in &gsites {
        if plan
            .sites
            .range((b, ui, 0)..=(b, ui, u16::MAX))
            .any(|&(_, _, n)| !names.contains(&n))
        {
            return;
        }
    }
    let mut mini = FwdPlan::default();
    for &k in group {
        mini.rhs.insert(cands[k].id, cands[k].cone.clone());
        for &(b, ui) in &cands[k].sites {
            mini.sites.insert((b, ui, cands[k].id));
        }
    }
    let fwd = Fwd { f, plan: &mini, vn };
    let mut after = 0usize;
    let mut before = 0usize;
    for &(b, ui) in &gsites {
        let Some(stmt) = f.blocks.get(&b).and_then(|blk| blk.stmts.get(ui)) else {
            return;
        };
        // Branch conditions only: relocating a load into a standing
        // assignment merely moves it — and strands a previously pure
        // tree outside the pure fold-shrinks tier (measured on bash:
        // standing OF-tree assignments rose 196 → 244 without this
        // gate). The slice's target is the flag pair at the jcc.
        if !matches!(stmt, Stmt::Branch { .. }) {
            return;
        }
        let mut r = FwdRound::default();
        let folded = fwd.rewrite_stmt(stmt, b, ui, &mut r);
        if r.forwards == 0 || r.size_skipped != 0 {
            return;
        }
        // All-or-nothing: no member name may survive at any site.
        let mut survives = false;
        irssa::for_each_use(&folded, &mut |r| {
            if names.contains(&r.num) {
                survives = true;
            }
        });
        if survives {
            return;
        }
        after += stmt_nodes(&folded);
        before += stmt_nodes(stmt);
    }
    // The dying definitions: the members, plus every inlined temp whose
    // uses all lie inside member definition statements.
    let member_defs: BTreeSet<(u64, usize)> =
        group.iter().map(|&k| (cands[k].dva, cands[k].di)).collect();
    for &k in group {
        if let Some(stmt) = f
            .blocks
            .get(&cands[k].dva)
            .and_then(|blk| blk.stmts.get(cands[k].di))
        {
            before += stmt_nodes(stmt);
        }
    }
    let temps: BTreeSet<u16> = group
        .iter()
        .flat_map(|&k| cands[k].inlined.iter().copied())
        .collect();
    for &t in &temps {
        let Some(tuses) = index.uses.get(&t) else {
            continue;
        };
        let all_in = tuses.iter().all(|site| {
            matches!(*site, UseSite::Stmt { block, index: i } if member_defs.contains(&(block, i)))
        });
        if !all_in {
            continue;
        }
        if let Some(&Def::Assign { block, index: ti }) = index.defs.get(t as usize)
            && let Some(stmt) = f.blocks.get(&block).and_then(|blk| blk.stmts.get(ti))
        {
            before += stmt_nodes(stmt);
        }
    }
    if after >= before {
        return; // the function does not strictly shrink
    }
    // Commit: the cone replaces any partial main-plan claim, every site.
    for &k in group {
        let id = cands[k].id;
        let stale: Vec<(u64, usize, u16)> = plan
            .sites
            .iter()
            .filter(|&&(_, _, n)| n == id)
            .copied()
            .collect();
        for s in stale {
            plan.sites.remove(&s);
        }
        plan.multi.remove(&id);
        plan.rhs.insert(id, cands[k].cone.clone());
        for &(b, ui) in &cands[k].sites {
            plan.sites.insert((b, ui, id));
        }
        plan.load_multi.insert(id);
    }
}

/// Build the index, plan, and splice — one round. No name goes away (the
/// definitions stay standing), so there is nothing to compact; `partial` is
/// recomputed because a spliced tree carries its own reads to a new
/// position.
fn fwd_round(f: &SsaFunction) -> (SsaFunction, FwdRound) {
    let index = build_index(f);
    let vn = build_vn(f);
    let plan = plan_forwards(f, &index, &vn);
    let fwd = Fwd {
        f,
        plan: &plan,
        vn: &vn,
    };
    let mut r = FwdRound::default();

    let mut blocks: BTreeMap<u64, SsaBlock> = BTreeMap::new();
    for (&va, block) in &f.blocks {
        let stmts: Vec<Stmt> = block
            .stmts
            .iter()
            .enumerate()
            .map(|(i, s)| fwd.rewrite_stmt(s, va, i, &mut r))
            .collect();
        blocks.insert(
            va,
            SsaBlock {
                start: block.start,
                end: block.end,
                phis: block.phis.clone(),
                stmts,
                successors: block.successors.clone(),
                truncated: block.truncated,
            },
        );
    }

    let mut out = SsaFunction {
        entry: f.entry,
        name: f.name.clone(),
        arch: f.arch,
        blocks,
        skipped: f.skipped.clone(),
        names: f.names.clone(),
        live_in: f.live_in.clone(),
        partial: f.partial.clone(),
    };
    recompute_partial(&mut out);
    (out, r)
}

/// Substitute definitions' right-hand sides into their uses, rebuilding
/// source-level expressions out of the lift's one-operation-per-statement
/// form. Returns a new function and what was done.
///
/// See the module docs for the tiers, the guards, and why substitution is
/// sound on the SSA dominance guarantee alone. Never deletes, adds, or
/// reorders a statement, never touches the CFG, never rewrites an intrinsic
/// read or a φ argument: the forwarded definitions are left standing for
/// [`eliminate_dead`] to sweep, so [`check_preserved`] holds on every
/// output. Pure, total, deterministic, and never panicking; malformed input
/// — one that fails [`irssa::check`] — comes back unchanged with zeroed
/// stats, the posture [`optimize`] and [`eliminate_dead`] already take.
pub fn forward(func: &SsaFunction) -> (SsaFunction, FwdStats) {
    let mut stats = FwdStats::default();
    if irssa::check(func).is_err() {
        return (func.clone(), stats);
    }
    let mut cur = func.clone();
    for round_no in 0..MAX_ROUNDS {
        let (next, r) = fwd_round(&cur);
        stats.rounds += 1;
        stats.forwards += r.forwards;
        stats.multi_spliced += r.multi_spliced;
        stats.load_pair_spliced += r.load_pair_spliced;
        // Standing refusals, not a running total: see `FwdStats`.
        stats.size_skipped = r.size_skipped;
        if next == cur {
            break;
        }
        cur = next;
        // The last round still changed something: the bound, not a
        // fixpoint, ended the loop. The output stays — it is a completed
        // round's, and every forwarding state is sound.
        if round_no + 1 == MAX_ROUNDS {
            stats.capped = true;
        }
    }
    (cur, stats)
}

// ---------------------------------------------------------------------------
// Dead-code elimination
// ---------------------------------------------------------------------------

/// What one [`eliminate_dead`] call did. Deterministic counters; all
/// zero for an input [`irssa::check`] rejected, which is returned
/// unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DceStats {
    /// Statements deleted — always unmarked, load-free
    /// [`Stmt::Assign`]s, never an effect.
    pub stmts_removed: usize,
    /// φ-nodes deleted.
    pub phis_removed: usize,
    /// Names dropped by the compaction: one per deleted statement or φ.
    pub names_removed: usize,
    /// Unmarked assignments kept anyway because their right-hand side
    /// holds a [`Expr::Load`] that may fault. Dead but honestly kept —
    /// the number is the pass's own admission of what it cannot prove.
    pub kept_loads: usize,
}

/// Whether `stmt` survives the sweep whatever the marking says, so that
/// its reads are unconditionally live: the three effects DCE never
/// deletes, plus a load-bearing assignment (see the load doctrine in the
/// module docs). Seeding *these* reads is what keeps the sweep
/// well-formed — a kept statement never reads a name that went away.
fn pinned(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Assign { value, .. } => irflow::contains_load(value, 0),
        Stmt::Store { .. } | Stmt::Branch { .. } | Stmt::Intrinsic { .. } => true,
    }
}

/// Mark `id` live, enqueueing it the first time only, so the worklist
/// pops each name at most once and the walk is linear.
fn seed(id: u16, marked: &mut BTreeSet<u16>, work: &mut Vec<u16>) {
    if marked.insert(id) {
        work.push(id);
    }
}

/// The live name set: the least set containing every read of a [`pinned`]
/// statement and every non-entry definition of a cell in `live_out`,
/// closed under "a marked name's definition marks what that definition
/// reads". The result is order-independent (a least fixpoint over a
/// monotone rule), so the `Vec` worklist costs no determinism.
fn mark(f: &SsaFunction, index: &Index, live_out: &BTreeSet<Cell>) -> BTreeSet<u16> {
    let mut marked: BTreeSet<u16> = BTreeSet::new();
    let mut work: Vec<u16> = Vec::new();

    for block in f.blocks.values() {
        for stmt in &block.stmts {
            if pinned(stmt) {
                irssa::for_each_use(stmt, &mut |r| seed(r.num, &mut marked, &mut work));
            }
        }
    }
    // The live-out roots, by *cell*: see the module docs for why a
    // version cannot be named here. A version-0 name needs no seeding —
    // it has no definition to delete.
    for (id, n) in f.names.iter().enumerate() {
        let defined = !matches!(index.defs.get(id), Some(Def::Entry) | None);
        if defined && live_out.contains(&(n.space, n.cell)) {
            seed(id as u16, &mut marked, &mut work);
        }
    }

    while let Some(id) = work.pop() {
        match index.defs.get(id as usize) {
            Some(&Def::Assign { block, index: i }) => {
                if let Some(stmt) = f.blocks.get(&block).and_then(|b| b.stmts.get(i)) {
                    irssa::for_each_use(stmt, &mut |r| seed(r.num, &mut marked, &mut work));
                }
            }
            Some(&Def::Phi { block, phi }) => {
                if let Some(p) = f.blocks.get(&block).and_then(|b| b.phis.get(phi)) {
                    for &(_, arg) in &p.args {
                        seed(arg, &mut marked, &mut work);
                    }
                }
            }
            // An at-entry value and an intrinsic write are opaque: there
            // is no right-hand side to walk, and an intrinsic's own reads
            // were already seeded — it is pinned.
            _ => {}
        }
    }
    marked
}

/// Whether the definition of `id` may be deleted at all, independent of
/// liveness: a version-0 name has no definition to delete, and a name the
/// table does not hold is not this pass's to touch (defensively
/// unreachable on input [`irssa::check`] accepted).
fn sweepable(f: &SsaFunction, id: u16) -> bool {
    f.names.get(id as usize).is_some_and(|n| n.version != 0)
}

/// Remove the definitions nothing observes: unmarked load-free
/// assignments and unmarked φs, marking from the effects, the
/// load-bearing assignments, and `live_out` (an over-approximated set of
/// [`ir::Reg`]s whose *cells* are observable at return —
/// [`crate::callfx::function_live_out`] builds one per architecture).
///
/// Returns a new function and what was swept. Never deletes a
/// [`Stmt::Store`], [`Stmt::Branch`], or [`Stmt::Intrinsic`]; never
/// deletes an assignment whose value holds a [`Expr::Load`]; never
/// touches the CFG; never reorders. Pure, total, deterministic, and never
/// panicking — and, like [`optimize`], it does not launder malformed
/// input: a function that fails [`irssa::check`] comes back unchanged
/// with zeroed stats.
///
/// One call is a fixpoint (marking is transitive), so calling it twice
/// removes nothing the second time. [`check_swept`] is the companion
/// differential check.
pub fn eliminate_dead(func: &SsaFunction, live_out: &[Reg]) -> (SsaFunction, DceStats) {
    let mut stats = DceStats::default();
    if irssa::check(func).is_err() {
        return (func.clone(), stats);
    }
    let cells: BTreeSet<Cell> = live_out.iter().map(|r| (r.space, r.num)).collect();
    let index = build_index(func);
    let marked = mark(func, &index, &cells);

    let mut removed: BTreeSet<u16> = BTreeSet::new();
    let mut blocks: BTreeMap<u64, SsaBlock> = BTreeMap::new();
    for (&va, block) in &func.blocks {
        let mut stmts: Vec<Stmt> = Vec::with_capacity(block.stmts.len());
        for stmt in &block.stmts {
            if let Stmt::Assign { dst, value } = stmt
                && !marked.contains(&dst.num)
                && sweepable(func, dst.num)
            {
                if irflow::contains_load(value, 0) {
                    stats.kept_loads += 1;
                } else {
                    stats.stmts_removed += 1;
                    removed.insert(dst.num);
                    continue;
                }
            }
            stmts.push(stmt.clone());
        }
        let mut phis: Vec<Phi> = Vec::with_capacity(block.phis.len());
        for phi in &block.phis {
            if !marked.contains(&phi.dst) && sweepable(func, phi.dst) {
                stats.phis_removed += 1;
                removed.insert(phi.dst);
                continue;
            }
            phis.push(phi.clone());
        }
        blocks.insert(
            va,
            SsaBlock {
                start: block.start,
                end: block.end,
                phis,
                stmts,
                successors: block.successors.clone(),
                truncated: block.truncated,
            },
        );
    }

    stats.names_removed = removed.len();
    let out = SsaFunction {
        entry: func.entry,
        name: func.name.clone(),
        arch: func.arch,
        blocks,
        skipped: func.skipped.clone(),
        names: func.names.clone(),
        live_in: func.live_in.clone(),
        partial: func.partial.clone(),
    };
    let mut out = compact(out, &removed);
    recompute_partial(&mut out);
    (out, stats)
}

// ---------------------------------------------------------------------------
// The preservation check
// ---------------------------------------------------------------------------

/// How an output broke the preservation contract. [`check_preserved`]
/// returns the first one it finds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Preserved {
    /// `entry`, `name`, or `skipped` changed.
    Function,
    /// The block set changed.
    Blocks,
    /// A block's bounds, successors, or `truncated` flag changed.
    BlockShape { block: u64 },
    /// A block's statement count changed.
    StmtCount { block: u64 },
    /// A statement changed kind.
    StmtKind { block: u64, index: usize },
    /// A branch's kind or the presence of its condition changed.
    Branch { block: u64, index: usize },
    /// An intrinsic's name, writes, or reads changed.
    Intrinsic { block: u64, index: usize },
    /// A block gained a φ, or one it kept changed destination.
    Phis { block: u64 },
    /// The output names are not a subset of the input's.
    Names,
    /// The version-0 (at-entry) names are not identical.
    LiveIn,
}

/// A name's identity, independent of its id: what the compaction must
/// preserve.
type Canon = (Space, u16, u32, Width);

fn canon(names: &[Name], id: u16) -> Option<Canon> {
    names
        .get(id as usize)
        .map(|n| (n.space, n.cell, n.version, n.width))
}

/// Whether two expressions are equal modulo the id remapping: identical
/// trees whose register occurrences name the same `(space, cell, version,
/// width)` at the same read width.
fn expr_eq(a: &Expr, an: &[Name], b: &Expr, bn: &[Name], depth: usize) -> bool {
    if depth > REWRITE_DEPTH {
        return true; // defensively unreachable
    }
    match (a, b) {
        (
            Expr::Const {
                value: x,
                width: xw,
            },
            Expr::Const {
                value: y,
                width: yw,
            },
        ) => x == y && xw == yw,
        (Expr::Reg(x), Expr::Reg(y)) => {
            x.width == y.width && canon(an, x.num) == canon(bn, y.num) && canon(an, x.num).is_some()
        }
        (Expr::Load { addr: x, width: xw }, Expr::Load { addr: y, width: yw }) => {
            xw == yw && expr_eq(x, an, y, bn, depth + 1)
        }
        (Expr::Unary { op: xo, operand: x }, Expr::Unary { op: yo, operand: y }) => {
            xo == yo && expr_eq(x, an, y, bn, depth + 1)
        }
        (
            Expr::Binary {
                op: xo,
                lhs: xl,
                rhs: xr,
            },
            Expr::Binary {
                op: yo,
                lhs: yl,
                rhs: yr,
            },
        ) => xo == yo && expr_eq(xl, an, yl, bn, depth + 1) && expr_eq(xr, an, yr, bn, depth + 1),
        _ => false,
    }
}

/// A name-table identity violation, in the vocabulary both differential
/// checks share.
enum NameFault {
    /// The output names are not a subset of the input's.
    Names,
    /// The version-0 (at-entry) names, or `live_in`, are not identical.
    LiveIn,
}

/// The name-table rules **both** passes must obey, factored so the two
/// checks cannot drift: every output name is an input name (nothing
/// invented), the version-0 set is identical (no at-entry value dropped),
/// and `live_in` names the same cells.
fn check_names(input: &SsaFunction, output: &SsaFunction) -> Result<(), NameFault> {
    let in_names: BTreeSet<Canon> = input
        .names
        .iter()
        .map(|n| (n.space, n.cell, n.version, n.width))
        .collect();
    for n in &output.names {
        if !in_names.contains(&(n.space, n.cell, n.version, n.width)) {
            return Err(NameFault::Names);
        }
    }
    let entry_names = |f: &SsaFunction| -> Vec<Canon> {
        f.names
            .iter()
            .filter(|n| n.version == 0)
            .map(|n| (n.space, n.cell, n.version, n.width))
            .collect()
    };
    if entry_names(input) != entry_names(output) {
        return Err(NameFault::LiveIn);
    }
    let live = |f: &SsaFunction| -> Option<Vec<Canon>> {
        f.live_in.iter().map(|&id| canon(&f.names, id)).collect()
    };
    if live(input) != live(output) || live(output).is_none() {
        return Err(NameFault::LiveIn);
    }
    Ok(())
}

/// Whether two blocks agree on everything neither pass may change: the
/// bounds, the successor list, and the `truncated` flag.
fn shape_eq(a: &SsaBlock, b: &SsaBlock) -> bool {
    a.start == b.start
        && a.end == b.end
        && a.successors == b.successors
        && a.truncated == b.truncated
}

/// The statement's kind, for the discriminant comparison.
fn kind_of(stmt: &Stmt) -> u8 {
    match stmt {
        Stmt::Assign { .. } => 0,
        Stmt::Store { .. } => 1,
        Stmt::Branch { .. } => 2,
        Stmt::Intrinsic { .. } => 3,
    }
}

/// Verify the preservation contract of [`optimize`] without trusting it:
/// the CFG, the statement sequence's shape, every effect, and the name
/// table's identity survive, modulo the compaction's renumbering.
///
/// Total and side-effect-free; returns the first [`Preserved`] violation.
/// Every test in this module runs it, together with [`irssa::check`], on
/// every output.
pub fn check_preserved(input: &SsaFunction, output: &SsaFunction) -> Result<(), Preserved> {
    if input.entry != output.entry || input.name != output.name || input.skipped != output.skipped {
        return Err(Preserved::Function);
    }
    if input.blocks.len() != output.blocks.len() || input.blocks.keys().ne(output.blocks.keys()) {
        return Err(Preserved::Blocks);
    }

    // The names the output keeps must be input names, with the version-0
    // set identical — nothing invented, no at-entry value dropped.
    match check_names(input, output) {
        Ok(()) => {}
        Err(NameFault::Names) => return Err(Preserved::Names),
        Err(NameFault::LiveIn) => return Err(Preserved::LiveIn),
    }

    for (&va, a) in &input.blocks {
        let Some(b) = output.blocks.get(&va) else {
            return Err(Preserved::Blocks);
        };
        if !shape_eq(a, b) {
            return Err(Preserved::BlockShape { block: va });
        }
        if a.stmts.len() != b.stmts.len() {
            return Err(Preserved::StmtCount { block: va });
        }
        for (i, (x, y)) in a.stmts.iter().zip(b.stmts.iter()).enumerate() {
            if kind_of(x) != kind_of(y) {
                return Err(Preserved::StmtKind {
                    block: va,
                    index: i,
                });
            }
            match (x, y) {
                (
                    Stmt::Branch {
                        kind: xk, cond: xc, ..
                    },
                    Stmt::Branch {
                        kind: yk, cond: yc, ..
                    },
                ) => {
                    if xk != yk || xc.is_some() != yc.is_some() {
                        return Err(Preserved::Branch {
                            block: va,
                            index: i,
                        });
                    }
                }
                (
                    Stmt::Intrinsic {
                        name: xn,
                        writes: xw,
                        reads: xr,
                    },
                    Stmt::Intrinsic {
                        name: yn,
                        writes: yw,
                        reads: yr,
                    },
                ) => {
                    let writes_eq = xw.len() == yw.len()
                        && xw.iter().zip(yw).all(|(p, q)| {
                            p.width == q.width
                                && canon(&input.names, p.num) == canon(&output.names, q.num)
                        });
                    let reads_eq = xr.len() == yr.len()
                        && xr
                            .iter()
                            .zip(yr)
                            .all(|(p, q)| expr_eq(p, &input.names, q, &output.names, 0));
                    if xn != yn || !writes_eq || !reads_eq {
                        return Err(Preserved::Intrinsic {
                            block: va,
                            index: i,
                        });
                    }
                }
                _ => {}
            }
            // A definition occurrence is never rewritten, only renumbered.
            let mut defs_a: Vec<Option<Canon>> = Vec::new();
            let mut defs_b: Vec<Option<Canon>> = Vec::new();
            irssa::for_each_def(x, &mut |r| defs_a.push(canon(&input.names, r.num)));
            irssa::for_each_def(y, &mut |r| defs_b.push(canon(&output.names, r.num)));
            if defs_a != defs_b || defs_a.iter().any(Option::is_none) {
                return Err(Preserved::StmtKind {
                    block: va,
                    index: i,
                });
            }
        }

        // φs may only leave: every output φ is an input φ of this block.
        let in_phis: BTreeSet<Option<Canon>> =
            a.phis.iter().map(|p| canon(&input.names, p.dst)).collect();
        if b.phis.len() > a.phis.len() {
            return Err(Preserved::Phis { block: va });
        }
        for p in &b.phis {
            let c = canon(&output.names, p.dst);
            if c.is_none() || !in_phis.contains(&c) {
                return Err(Preserved::Phis { block: va });
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The sweep check
// ---------------------------------------------------------------------------

/// How an output broke [`eliminate_dead`]'s contract. [`check_swept`]
/// returns the first one it finds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Swept {
    /// `entry`, `name`, or `skipped` changed.
    Function,
    /// The block set changed.
    Blocks,
    /// A block's bounds, successors, or `truncated` flag changed.
    BlockShape { block: u64 },
    /// The output's statements are not a subsequence of the input's:
    /// something was added, rewritten, or reordered.
    NotSubsequence { block: u64 },
    /// A [`Stmt::Store`], [`Stmt::Branch`], or [`Stmt::Intrinsic`] was
    /// removed. Never allowed, at any liveness.
    RemovedEffect { block: u64, index: usize },
    /// A removed assignment's value holds a [`Expr::Load`], which may
    /// fault: deleting it is unproven.
    RemovedLoad { block: u64, index: usize },
    /// A removed assignment defines a live-out cell (or, defensively, a
    /// version-0 name, which has no definition to delete).
    RemovedLiveOut { block: u64, index: usize },
    /// A block gained a φ, or its φs were reordered or rewritten.
    Phis { block: u64 },
    /// A removed φ defines a live-out cell.
    RemovedPhiLiveOut { block: u64, phi: usize },
    /// A removed name is still named somewhere in the output — or an
    /// output occurrence names nothing at all.
    Dangling,
    /// The output names are not a subset of the input's.
    Names,
    /// The version-0 (at-entry) names are not identical.
    LiveIn,
}

/// Whether two statements are the same statement modulo the compaction's
/// renumbering: same kind, same effects, and every register occurrence —
/// definitions included — naming the same `(space, cell, version, width)`
/// at the same access width. [`eliminate_dead`] rewrites nothing, so an
/// output statement must match its input statement exactly under this
/// relation.
fn stmt_eq(a: &Stmt, an: &[Name], b: &Stmt, bn: &[Name]) -> bool {
    let reg_eq = |x: &Reg, y: &Reg| {
        x.width == y.width && canon(an, x.num) == canon(bn, y.num) && canon(an, x.num).is_some()
    };
    match (a, b) {
        (Stmt::Assign { dst: xd, value: xv }, Stmt::Assign { dst: yd, value: yv }) => {
            reg_eq(xd, yd) && expr_eq(xv, an, yv, bn, 0)
        }
        (
            Stmt::Store {
                addr: xa,
                value: xv,
            },
            Stmt::Store {
                addr: ya,
                value: yv,
            },
        ) => expr_eq(xa, an, ya, bn, 0) && expr_eq(xv, an, yv, bn, 0),
        (
            Stmt::Branch {
                kind: xk,
                cond: xc,
                target: xt,
            },
            Stmt::Branch {
                kind: yk,
                cond: yc,
                target: yt,
            },
        ) => {
            xk == yk
                && match (xc, yc) {
                    (None, None) => true,
                    (Some(p), Some(q)) => expr_eq(p, an, q, bn, 0),
                    _ => false,
                }
                && expr_eq(xt, an, yt, bn, 0)
        }
        (
            Stmt::Intrinsic {
                name: xn,
                writes: xw,
                reads: xr,
            },
            Stmt::Intrinsic {
                name: yn,
                writes: yw,
                reads: yr,
            },
        ) => {
            xn == yn
                && xw.len() == yw.len()
                && xw.iter().zip(yw).all(|(p, q)| reg_eq(p, q))
                && xr.len() == yr.len()
                && xr.iter().zip(yr).all(|(p, q)| expr_eq(p, an, q, bn, 0))
        }
        _ => false,
    }
}

/// Verify [`eliminate_dead`]'s contract without trusting it: the output
/// is the input minus a set of *justified* deletions.
///
/// - the CFG — block set, bounds, successors, `truncated`, `entry` — and
///   the name-table identity rules of [`check_preserved`] hold;
/// - every output statement sequence is a subsequence of its input's,
///   compared modulo the compaction's renumbering, so nothing was added,
///   rewritten, or reordered;
/// - every removed statement is an [`Stmt::Assign`] with a load-free
///   value whose destination is neither a live-out cell nor version 0;
///   every removed φ likewise defines no live-out cell;
/// - no removed name is still named anywhere in the output.
///
/// `live_out` is the same set the pass was given. Total and
/// side-effect-free; returns the first [`Swept`] violation.
pub fn check_swept(
    input: &SsaFunction,
    output: &SsaFunction,
    live_out: &[Reg],
) -> Result<(), Swept> {
    if input.entry != output.entry || input.name != output.name || input.skipped != output.skipped {
        return Err(Swept::Function);
    }
    if input.blocks.len() != output.blocks.len() || input.blocks.keys().ne(output.blocks.keys()) {
        return Err(Swept::Blocks);
    }
    match check_names(input, output) {
        Ok(()) => {}
        Err(NameFault::Names) => return Err(Swept::Names),
        Err(NameFault::LiveIn) => return Err(Swept::LiveIn),
    }
    let cells: BTreeSet<Cell> = live_out.iter().map(|r| (r.space, r.num)).collect();
    // The names whose definition went away, by identity.
    let mut gone: BTreeSet<Canon> = BTreeSet::new();

    for (&va, a) in &input.blocks {
        let Some(b) = output.blocks.get(&va) else {
            return Err(Swept::Blocks);
        };
        if !shape_eq(a, b) {
            return Err(Swept::BlockShape { block: va });
        }

        // The greedy subsequence match: an input statement either is the
        // next output statement or was removed, and every output
        // statement must be consumed.
        let mut next = 0usize;
        for (i, x) in a.stmts.iter().enumerate() {
            if let Some(y) = b.stmts.get(next)
                && stmt_eq(x, &input.names, y, &output.names)
            {
                next += 1;
                continue;
            }
            let Stmt::Assign { dst, value } = x else {
                return Err(Swept::RemovedEffect {
                    block: va,
                    index: i,
                });
            };
            if irflow::contains_load(value, 0) {
                return Err(Swept::RemovedLoad {
                    block: va,
                    index: i,
                });
            }
            let Some(n) = input.names.get(dst.num as usize) else {
                return Err(Swept::Names);
            };
            if n.version == 0 || cells.contains(&(n.space, n.cell)) {
                return Err(Swept::RemovedLiveOut {
                    block: va,
                    index: i,
                });
            }
            gone.insert((n.space, n.cell, n.version, n.width));
        }
        if next != b.stmts.len() {
            return Err(Swept::NotSubsequence { block: va });
        }

        // The same subsequence rule over the block's φs, which the pass
        // deletes but never rewrites.
        let mut next = 0usize;
        for (p, phi) in a.phis.iter().enumerate() {
            let Some(c) = canon(&input.names, phi.dst) else {
                return Err(Swept::Names);
            };
            if let Some(q) = b.phis.get(next)
                && canon(&output.names, q.dst) == Some(c)
            {
                let args_eq = phi.args.len() == q.args.len()
                    && phi.args.iter().zip(&q.args).all(|(&(xk, xa), &(yk, ya))| {
                        xk == yk && canon(&input.names, xa) == canon(&output.names, ya)
                    });
                if !args_eq {
                    return Err(Swept::Phis { block: va });
                }
                next += 1;
                continue;
            }
            let Some(n) = input.names.get(phi.dst as usize) else {
                return Err(Swept::Names);
            };
            if cells.contains(&(n.space, n.cell)) {
                return Err(Swept::RemovedPhiLiveOut { block: va, phi: p });
            }
            gone.insert(c);
        }
        if next != b.phis.len() {
            return Err(Swept::Phis { block: va });
        }
    }

    // Nothing removed may still be named: uses, definitions, φ
    // destinations, and φ arguments alike.
    let mut dangling = false;
    let probe = |id: u16, names: &[Name], dangling: &mut bool| match canon(names, id) {
        Some(c) if gone.contains(&c) => *dangling = true,
        None => *dangling = true,
        _ => {}
    };
    for block in output.blocks.values() {
        for phi in &block.phis {
            probe(phi.dst, &output.names, &mut dangling);
            for &(_, arg) in &phi.args {
                probe(arg, &output.names, &mut dangling);
            }
        }
        for stmt in &block.stmts {
            let mut hit = false;
            irssa::for_each_use(stmt, &mut |r| probe(r.num, &output.names, &mut hit));
            irssa::for_each_def(stmt, &mut |r| probe(r.num, &output.names, &mut hit));
            dangling |= hit;
        }
    }
    if dangling {
        return Err(Swept::Dangling);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{self, BinOp, BranchKind, Flag, UnOp};
    use crate::model::Arch;
    use crate::{callfx, irlift};

    // -- construction helpers ----------------------------------------------

    fn c(value: u64, w: Width) -> Expr {
        Expr::constant(value, w)
    }
    fn ra(num: u16, w: Width) -> Reg {
        Reg::arch(num, w)
    }
    fn read(r: Reg) -> Expr {
        Expr::reg(r)
    }
    fn assign(dst: Reg, value: Expr) -> Stmt {
        Stmt::Assign { dst, value }
    }
    fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::binary(op, l, r)
    }

    fn block(start: u64, stmts: Vec<Stmt>, successors: Vec<u64>) -> irlift::LiftedBlock {
        irlift::LiftedBlock {
            start,
            end: start + 4,
            stmts,
            successors,
            truncated: false,
        }
    }

    fn func(entry: u64, blocks: Vec<irlift::LiftedBlock>) -> irlift::LiftedFunction {
        irlift::LiftedFunction {
            entry,
            name: None,
            arch: Arch::X86_64,
            blocks: blocks.into_iter().map(|b| (b.start, b)).collect(),
        }
    }

    fn build(f: &irlift::LiftedFunction) -> SsaFunction {
        let ssa = irssa::construct(f).expect("well-formed input constructs");
        assert_eq!(irssa::check(&ssa), Ok(()), "input SSA must check");
        ssa
    }

    /// Optimize and insist on the module's two promises: the output is
    /// well-formed SSA, and it preserves everything it must.
    fn run(input: &SsaFunction) -> (SsaFunction, Stats) {
        let (out, stats) = optimize(input);
        assert_eq!(irssa::check(&out), Ok(()), "output must pass irssa::check");
        assert_eq!(check_preserved(input, &out), Ok(()), "output must preserve");
        (out, stats)
    }

    /// Optimize a hand-built lifted function end to end.
    fn opt(f: &irlift::LiftedFunction) -> (SsaFunction, Stats) {
        run(&build(f))
    }

    fn phi_count(f: &SsaFunction) -> usize {
        f.blocks.values().map(|b| b.phis.len()).sum()
    }

    fn stmts(f: &SsaFunction, va: u64) -> &[Stmt] {
        &f.blocks[&va].stmts
    }

    /// The rendered text, for the goldens.
    fn text(f: &SsaFunction) -> String {
        irssa::render(f)
    }

    // -- 1: straight line ---------------------------------------------------

    #[test]
    fn a_constant_reaches_a_later_use_in_the_same_block() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), c(5, Width::W64)),
                    assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                ],
                vec![],
            )],
        );
        let (out, stats) = opt(&f);
        assert_eq!(stats.rewrites, 1);
        assert!(!stats.capped);
        // The use reads the constant...
        assert_eq!(
            stmts(&out, 0x1000)[1],
            assign(
                Reg {
                    num: 1,
                    ..ra(1, Width::W64)
                },
                c(5, Width::W64)
            )
        );
        // ...and the now-dead definition stays: DCE is the next slice.
        assert_eq!(stmts(&out, 0x1000).len(), 2);
        assert_eq!(
            stmts(&out, 0x1000)[0],
            assign(
                Reg {
                    num: 0,
                    ..ra(0, Width::W64)
                },
                c(5, Width::W64)
            )
        );
    }

    // -- 2: cross-block -----------------------------------------------------

    /// entry sets rax to 5 and branches; neither arm redefines it; the
    /// merge compares it and branches on the flag.
    fn cross_block_constant() -> irlift::LiftedFunction {
        func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(ra(0, Width::W64), c(5, Width::W64)),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1020, Width::W64),
                        },
                    ],
                    vec![0x1010, 0x1020],
                ),
                block(
                    0x1010,
                    vec![assign(ra(2, Width::W64), c(1, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(2, Width::W64), c(2, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![
                        assign(
                            Reg::flag(Flag::Zero),
                            bin(BinOp::Eq, read(ra(0, Width::W64)), c(5, Width::W64)),
                        ),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1000, Width::W64),
                        },
                    ],
                    vec![0x1000],
                ),
            ],
        )
    }

    #[test]
    fn a_constant_crosses_blocks_and_folds_a_branch_condition() {
        let input = build(&cross_block_constant());
        let (out, stats) = run(&input);
        assert!(stats.rewrites >= 1);
        // The comparison folded to a constant condition (the flag's
        // destination is an SSA name id, so only the value is pinned)...
        let Stmt::Assign { dst, value } = &stmts(&out, 0x1030)[0] else {
            panic!("expected the flag assignment")
        };
        assert_eq!(dst.space, Space::Flag);
        assert_eq!(*value, c(1, Width::W1));
        // ...and the branch itself — kind, condition slot, target — and
        // the CFG are untouched.
        match (&stmts(&input, 0x1030)[1], &stmts(&out, 0x1030)[1]) {
            (
                Stmt::Branch {
                    kind: a,
                    target: at,
                    ..
                },
                Stmt::Branch {
                    kind: b,
                    cond,
                    target: bt,
                },
            ) => {
                assert_eq!(a, b);
                assert!(cond.is_some());
                assert_eq!(at, bt);
            }
            other => panic!("branch shape changed: {other:?}"),
        }
        assert_eq!(
            input.blocks.keys().collect::<Vec<_>>(),
            out.blocks.keys().collect::<Vec<_>>()
        );
        for (&va, b) in &out.blocks {
            assert_eq!(b.successors, input.blocks[&va].successors);
        }
    }

    // -- 3, 4: copy chains --------------------------------------------------

    #[test]
    fn a_copy_crosses_a_diamond() {
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1010, 0x1020],
                ),
                block(0x1010, vec![], vec![0x1030]),
                block(0x1020, vec![], vec![0x1030]),
                block(
                    0x1030,
                    vec![assign(ra(2, Width::W64), read(ra(1, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let (out, _) = opt(&f);
        // rcx#1 := rax#0 makes the merge read rax#0 directly.
        let Stmt::Assign {
            value: Expr::Reg(r),
            ..
        } = &stmts(&out, 0x1030)[0]
        else {
            panic!("expected an assignment from a register");
        };
        assert_eq!(out.names[r.num as usize].cell, 0, "reads rax, not rcx");
        assert_eq!(out.names[r.num as usize].version, 0);
    }

    #[test]
    fn a_multi_hop_copy_chain_resolves_to_its_root() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(1, Width::W64), read(ra(0, Width::W64))), // b := a
                    assign(ra(2, Width::W64), read(ra(1, Width::W64))), // c := b
                    assign(ra(3, Width::W64), read(ra(2, Width::W64))), // d := c
                    assign(ra(6, Width::W64), read(ra(3, Width::W64))), // use of d
                ],
                vec![],
            )],
        );
        let (out, _) = opt(&f);
        for i in 1..4 {
            let Stmt::Assign {
                value: Expr::Reg(r),
                ..
            } = &stmts(&out, 0x1000)[i]
            else {
                panic!("expected a register copy");
            };
            assert_eq!(out.names[r.num as usize].cell, 0, "hop {i} reads the root");
            assert_eq!(out.names[r.num as usize].version, 0);
        }
    }

    // -- 5, 6, 7: φ behavior ------------------------------------------------

    #[test]
    fn a_phi_of_one_value_collapses_with_an_exact_render() {
        // Both arms copy rax#0 into rcx; the merge's φ merges one value.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010, 0x1020]),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(2, Width::W64), read(ra(1, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        assert_eq!(phi_count(&input), 1, "the input has the φ");
        let (out, stats) = run(&input);
        assert_eq!(phi_count(&out), 0);
        assert_eq!(stats.phis_removed, 1);
        assert_eq!(stats.names_removed, 1);
        assert_eq!(
            text(&out),
            "; sub_1000 @ 0x0000000000001000 (ssa)\n\
             ; live-in: rax#0\n\
             loc_1000:\n\
             \x20 ; -> loc_1010, loc_1020\n\
             loc_1010:\n\
             \x20 rcx#1 := rax#0\n\
             \x20 ; -> loc_1030\n\
             loc_1020:\n\
             \x20 rcx#2 := rax#0\n\
             \x20 ; -> loc_1030\n\
             loc_1030:\n\
             \x20 rdx#1 := rax#0\n\
             \x20 ; -> (none)\n"
        );
    }

    #[test]
    fn a_loop_invariant_phi_collapses_to_its_initial_value() {
        // entry defines rax; the body reads it and loops; nothing
        // redefines the cell, so φ(rax#1, self) is rax#1.
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W64), read(ra(3, Width::W64)))],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![
                        assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1010, Width::W64),
                        },
                    ],
                    vec![0x1010, 0x1020],
                ),
                block(0x1020, vec![], vec![]),
            ],
        );
        let input = build(&f);
        let (out, _) = run(&input);
        assert!(phi_count(&out) < phi_count(&input) || phi_count(&input) == 0);
        let Stmt::Assign {
            value: Expr::Reg(r),
            ..
        } = &stmts(&out, 0x1010)[0]
        else {
            panic!("expected a register read");
        };
        // The body reads rbx#0, the value the invariant copy carries.
        assert_eq!(out.names[r.num as usize].cell, 3);
        assert_eq!(out.names[r.num as usize].version, 0);
    }

    #[test]
    fn a_real_induction_phi_varies_and_nothing_moves() {
        // rax := rax + 1 around the back edge: the φ is a genuine merge.
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W64), c(0, Width::W64))],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![
                        assign(
                            ra(0, Width::W64),
                            bin(BinOp::Add, read(ra(0, Width::W64)), c(1, Width::W64)),
                        ),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1010, Width::W64),
                        },
                    ],
                    vec![0x1010, 0x1020],
                ),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        let (out, stats) = run(&input);
        assert_eq!(out, input, "an induction variable yields no facts");
        assert_eq!(stats.rewrites, 0);
        assert_eq!(stats.phis_removed, 0);
    }

    // -- 8, 9: constants through φ -----------------------------------------

    #[test]
    fn a_phi_over_equal_constants_is_kept_and_its_uses_rewritten() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010, 0x1020]),
                block(
                    0x1010,
                    vec![assign(ra(0, Width::W64), c(7, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(0, Width::W64), c(7, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        let (out, _) = run(&input);
        assert_eq!(phi_count(&out), 1, "a constant-valued φ is kept");
        assert_eq!(
            stmts(&out, 0x1030)[0],
            assign(
                Reg {
                    num: match &stmts(&out, 0x1030)[0] {
                        Stmt::Assign { dst, .. } => dst.num,
                        _ => unreachable!(),
                    },
                    ..ra(1, Width::W64)
                },
                c(7, Width::W64)
            )
        );
    }

    #[test]
    fn a_phi_over_differing_constants_varies() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010, 0x1020]),
                block(
                    0x1010,
                    vec![assign(ra(0, Width::W64), c(7, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(0, Width::W64), c(8, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        let (out, stats) = run(&input);
        assert_eq!(out, input);
        assert_eq!(stats.rewrites, 0);
    }

    #[test]
    fn a_phi_over_equal_values_at_different_widths_varies() {
        // rax#1 is W32, rax#2 is W64: the φ's arguments agree in value but
        // not in the bits they guarantee, so strict equality declines.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010, 0x1020]),
                block(
                    0x1010,
                    vec![assign(ra(0, Width::W32), c(7, Width::W32))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(0, Width::W64), c(7, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(1, Width::W32), read(ra(0, Width::W32)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        let (out, stats) = run(&input);
        assert_eq!(out, input);
        assert_eq!(stats.rewrites, 0);
    }

    // -- 10: φ-of-φ ---------------------------------------------------------

    #[test]
    fn a_phi_of_a_phi_cascades_in_one_call() {
        // An inner diamond merges two copies of rax#0 into a φ; an outer
        // join merges that φ with another copy of the same value.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010, 0x1020]),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1030],
                ),
                // Inner merge, then a second join.
                block(0x1030, vec![], vec![0x1050]),
                block(
                    0x1040,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1050],
                ),
                block(
                    0x1050,
                    vec![assign(ra(2, Width::W64), read(ra(1, Width::W64)))],
                    vec![],
                ),
            ],
        );
        // 0x1040 must be reachable: branch to it from the entry.
        let mut f = f;
        if let Some(b) = f.blocks.get_mut(&0x1000) {
            b.successors = vec![0x1010, 0x1020, 0x1040];
        }
        let input = build(&f);
        assert_eq!(phi_count(&input), 2, "an inner and an outer φ");
        let (out, stats) = run(&input);
        assert_eq!(phi_count(&out), 0, "both collapse in one call");
        assert_eq!(stats.phis_removed, 2);
    }

    // -- 11, 12, 13: calls --------------------------------------------------

    /// A block that sets rax (caller-saved), rbx (callee-saved) and rdi
    /// (an argument register), calls, then reads all three.
    fn calling_function() -> irlift::LiftedFunction {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), c(5, Width::W64)),
                    assign(ra(3, Width::W64), c(7, Width::W64)),
                    assign(ra(7, Width::W64), c(9, Width::W64)),
                    assign(ra(5, Width::W64), read(ra(7, Width::W64))),
                    Stmt::Branch {
                        kind: BranchKind::Call,
                        cond: None,
                        target: c(0x2000, Width::W64),
                    },
                    assign(ra(2, Width::W64), read(ra(0, Width::W64))),
                    assign(ra(6, Width::W64), read(ra(3, Width::W64))),
                ],
                vec![],
            )],
        );
        callfx::apply(&f, &callfx::x86_64())
    }

    #[test]
    fn a_call_clobber_kills_a_caller_saved_fact() {
        let f = calling_function();
        let (out, _) = opt(&f);
        // The post-call read of rax is the intrinsic's fresh version, not
        // the constant 5: the last two statements read registers.
        let last = stmts(&out, 0x1000);
        let Stmt::Assign {
            value: Expr::Reg(r),
            ..
        } = &last[last.len() - 2]
        else {
            panic!("the caller-saved read must stay a register read");
        };
        assert_eq!(out.names[r.num as usize].cell, 0);
        assert!(out.names[r.num as usize].version > 1, "a post-call version");
    }

    #[test]
    fn a_callee_saved_fact_survives_the_call() {
        let f = calling_function();
        let (out, _) = opt(&f);
        let last = stmts(&out, 0x1000);
        assert_eq!(
            last[last.len() - 1],
            assign(
                match &last[last.len() - 1] {
                    Stmt::Assign { dst, .. } => *dst,
                    _ => unreachable!(),
                },
                c(7, Width::W64)
            ),
            "rbx is callee-saved, so its constant crosses the call"
        );
    }

    #[test]
    fn intrinsic_reads_stay_verbatim_while_ordinary_uses_are_rewritten() {
        let f = calling_function();
        let input = build(&f);
        let (out, _) = run(&input);
        let i = stmts(&out, 0x1000)
            .iter()
            .position(|s| matches!(s, Stmt::Intrinsic { .. }))
            .expect("callfx was inserted");
        assert_eq!(stmts(&out, 0x1000)[i], stmts(&input, 0x1000)[i]);
        let Stmt::Intrinsic { reads, .. } = &stmts(&out, 0x1000)[i] else {
            unreachable!()
        };
        assert!(
            reads.iter().all(|r| matches!(r, Expr::Reg(_))),
            "every callfx read is still a register: {reads:?}"
        );
        // The ordinary use of the same rdi name *is* rewritten.
        assert_eq!(
            stmts(&out, 0x1000)[3],
            assign(
                match &stmts(&out, 0x1000)[3] {
                    Stmt::Assign { dst, .. } => *dst,
                    _ => unreachable!(),
                },
                c(9, Width::W64)
            )
        );
    }

    // -- 14, 15: traps and memory ------------------------------------------

    #[test]
    fn a_proven_zero_divisor_is_substituted_but_never_folded() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(1, Width::W64), c(0, Width::W64)),
                    assign(
                        ra(2, Width::W64),
                        bin(
                            BinOp::UDiv,
                            read(ra(0, Width::W64)),
                            read(ra(1, Width::W64)),
                        ),
                    ),
                    assign(
                        ra(3, Width::W64),
                        bin(BinOp::UDiv, read(ra(0, Width::W64)), c(0, Width::W64)),
                    ),
                ],
                vec![],
            )],
        );
        let (out, _) = opt(&f);
        for i in [1, 2] {
            let Stmt::Assign { value, .. } = &stmts(&out, 0x1000)[i] else {
                unreachable!()
            };
            assert!(
                matches!(
                    value,
                    Expr::Binary {
                        op: BinOp::UDiv,
                        ..
                    }
                ),
                "the division stays written as a division: {value:?}"
            );
        }
        let Stmt::Assign {
            value: Expr::Binary { rhs, .. },
            ..
        } = &stmts(&out, 0x1000)[1]
        else {
            unreachable!()
        };
        assert_eq!(**rhs, c(0, Width::W64), "the divisor was substituted");
    }

    #[test]
    fn loads_yield_no_facts_survive_and_keep_their_folding_guard() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(1, Width::W64), c(0x40, Width::W64)),
                    // A load-bearing definition: no fact, never deleted.
                    assign(
                        ra(0, Width::W64),
                        Expr::load(read(ra(1, Width::W64)), Width::W64),
                    ),
                    assign(ra(2, Width::W64), read(ra(0, Width::W64))),
                    // `load & 0` must not fold to 0.
                    assign(
                        ra(3, Width::W64),
                        bin(
                            BinOp::And,
                            Expr::load(read(ra(1, Width::W64)), Width::W64),
                            read(ra(6, Width::W64)),
                        ),
                    ),
                    Stmt::Store {
                        addr: read(ra(1, Width::W64)),
                        value: read(ra(0, Width::W64)),
                    },
                ],
                vec![],
            )],
        );
        // rsi (6) is zero, so the `& 0` identity would fire were it legal.
        let mut f = f;
        if let Some(b) = f.blocks.get_mut(&0x1000) {
            b.stmts
                .insert(0, assign(ra(6, Width::W64), c(0, Width::W64)));
        }
        let (out, _) = opt(&f);
        let s = stmts(&out, 0x1000);
        assert_eq!(s.len(), 6, "no statement was added or deleted");
        assert!(
            matches!(
                &s[2],
                Stmt::Assign {
                    value: Expr::Load { .. },
                    ..
                }
            ),
            "the load survives: {:?}",
            s[2]
        );
        let Stmt::Assign { value, .. } = &s[4] else {
            unreachable!()
        };
        assert!(
            matches!(value, Expr::Binary { op: BinOp::And, .. }),
            "`load & 0` is not folded away: {value:?}"
        );
        // The store's address was rewritten to the constant, and it stays.
        let Stmt::Store { addr, .. } = &s[5] else {
            panic!("the store must survive")
        };
        assert_eq!(*addr, c(0x40, Width::W64));
    }

    // -- 16, 17, 18, 19: widths, copies, temporaries ------------------------

    #[test]
    fn a_read_wider_than_its_definition_is_left_alone() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W32), c(0x1234_5678, Width::W32)),
                    assign(ra(1, Width::W64), read(ra(0, Width::W64))), // partial
                    assign(ra(2, Width::W16), read(ra(0, Width::W16))), // exact
                ],
                vec![],
            )],
        );
        let input = build(&f);
        assert_eq!(input.partial, vec![(0x1000, 1)]);
        let (out, _) = run(&input);
        assert!(
            matches!(
                &stmts(&out, 0x1000)[1],
                Stmt::Assign {
                    value: Expr::Reg(_),
                    ..
                }
            ),
            "the partial read stays a register read"
        );
        let Stmt::Assign { value, .. } = &stmts(&out, 0x1000)[2] else {
            unreachable!()
        };
        assert_eq!(*value, c(0x5678, Width::W16), "truncated to the read width");
        assert_eq!(out.partial, vec![(0x1000, 1)], "recomputed exactly");
    }

    #[test]
    fn a_copy_wider_than_its_source_records_no_fact() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    // rax#1 is W32 wide and copies rcx#0.
                    assign(ra(0, Width::W32), read(ra(1, Width::W32))),
                    // A W64 read of it claims bits rax#1 never wrote.
                    assign(ra(2, Width::W64), read(ra(0, Width::W64))),
                    assign(ra(3, Width::W64), read(ra(2, Width::W64))),
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = run(&input);
        assert_eq!(out, input, "no fact, so nothing moves");
        assert_eq!(stats.rewrites, 0);
    }

    #[test]
    fn a_temporary_root_substitutes_only_inside_its_own_block() {
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(
                            Reg::temp(0, Width::W64),
                            bin(BinOp::Add, read(ra(1, Width::W64)), c(1, Width::W64)),
                        ),
                        assign(ra(0, Width::W64), read(Reg::temp(0, Width::W64))),
                        assign(ra(2, Width::W64), read(ra(0, Width::W64))),
                    ],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![assign(ra(3, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let (out, _) = opt(&f);
        // In the defining block the temporary is substituted...
        let Stmt::Assign {
            value: Expr::Reg(r),
            ..
        } = &stmts(&out, 0x1000)[2]
        else {
            panic!("expected a register read")
        };
        assert_eq!(out.names[r.num as usize].space, Space::Temp);
        // ...but never across the block boundary, where `ir::check` would
        // report a temporary read before its write.
        let Stmt::Assign {
            value: Expr::Reg(r),
            ..
        } = &stmts(&out, 0x1010)[0]
        else {
            panic!("expected a register read")
        };
        assert_eq!(out.names[r.num as usize].space, Space::Arch);
        for b in out.blocks.values() {
            assert_eq!(ir::check(&b.stmts), Ok(()));
        }
    }

    #[test]
    fn a_phi_whose_use_is_wider_than_its_root_is_kept() {
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W32), read(ra(1, Width::W32)))],
                    vec![0x1010, 0x1020],
                ),
                block(
                    0x1010,
                    vec![assign(ra(2, Width::W32), read(ra(0, Width::W32)))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(2, Width::W32), read(ra(0, Width::W32)))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![
                        // A W64 read of the W32-wide φ: not substitutable.
                        assign(ra(3, Width::W64), read(ra(2, Width::W64))),
                        // A W32 read of it: substitutable.
                        assign(ra(6, Width::W32), read(ra(2, Width::W32))),
                    ],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        assert_eq!(phi_count(&input), 1);
        let (out, stats) = run(&input);
        assert_eq!(phi_count(&out), 1, "the φ is kept");
        assert_eq!(stats.phis_removed, 0);
        // The eligible use was still rewritten — through the copy chain
        // to its root, rcx#0.
        let Stmt::Assign {
            value: Expr::Reg(r),
            ..
        } = &stmts(&out, 0x1030)[1]
        else {
            panic!("expected a register read")
        };
        assert_eq!(out.names[r.num as usize].cell, 1);
        assert_eq!(out.names[r.num as usize].version, 0);
    }

    // -- 20: compaction -----------------------------------------------------

    #[test]
    fn removing_a_phi_compacts_the_name_table() {
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(3, Width::W64), c(1, Width::W64))],
                    vec![0x1010, 0x1020],
                ),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(2, Width::W64), read(ra(1, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        let (out, stats) = run(&input);
        assert_eq!(stats.names_removed, 1);
        assert_eq!(out.names.len(), input.names.len() - 1);
        // `live_in` survives the renumbering and is still exactly the
        // version-0 set, ascending.
        let zeros: Vec<u16> = out
            .names
            .iter()
            .enumerate()
            .filter(|(_, n)| n.version == 0)
            .map(|(i, _)| i as u16)
            .collect();
        assert_eq!(out.live_in, zeros);
        // Version numbering keeps its gaps: rcx#1 and rcx#2 survive while
        // the φ's rcx#3 is gone.
        let versions: Vec<u32> = out
            .names
            .iter()
            .filter(|n| n.cell == 1 && n.space == Space::Arch)
            .map(|n| n.version)
            .collect();
        assert_eq!(versions, vec![1, 2]);
    }

    // -- 21: refusals -------------------------------------------------------

    #[test]
    fn a_malformed_function_is_returned_unchanged() {
        let mut broken = build(&cross_block_constant());
        broken.live_in.clear(); // no longer the version-0 set
        assert!(irssa::check(&broken).is_err());
        let (out, stats) = optimize(&broken);
        assert_eq!(out, broken);
        assert_eq!(stats.rounds, 0);
        assert!(!stats.capped);
    }

    #[test]
    fn an_empty_function_is_returned_unchanged() {
        let empty = SsaFunction {
            entry: 0x1000,
            name: None,
            arch: Arch::X86_64,
            blocks: BTreeMap::new(),
            skipped: Vec::new(),
            names: Vec::new(),
            live_in: Vec::new(),
            partial: Vec::new(),
        };
        let (out, stats) = run(&empty);
        assert_eq!(out, empty);
        assert_eq!(stats.rewrites, 0);
        assert!(!stats.capped);
    }

    // -- 22: the preservation check itself ---------------------------------

    #[test]
    fn check_preserved_rejects_every_broken_output() {
        let input = build(&cross_block_constant());
        let (good, _) = run(&input);
        assert_eq!(check_preserved(&input, &good), Ok(()));

        // A dropped statement.
        let mut dropped = good.clone();
        dropped.blocks.get_mut(&0x1030).unwrap().stmts.pop();
        assert_eq!(
            check_preserved(&input, &dropped),
            Err(Preserved::StmtCount { block: 0x1030 })
        );

        // A mutated branch kind.
        let mut mutated = good.clone();
        if let Some(Stmt::Branch { kind, .. }) =
            mutated.blocks.get_mut(&0x1030).unwrap().stmts.get_mut(1)
        {
            *kind = BranchKind::Return;
        }
        assert_eq!(
            check_preserved(&input, &mutated),
            Err(Preserved::Branch {
                block: 0x1030,
                index: 1
            })
        );

        // An added statement.
        let mut added = good.clone();
        added
            .blocks
            .get_mut(&0x1010)
            .unwrap()
            .stmts
            .push(Stmt::Store {
                addr: c(0x40, Width::W64),
                value: c(1, Width::W64),
            });
        assert_eq!(
            check_preserved(&input, &added),
            Err(Preserved::StmtCount { block: 0x1010 })
        );

        // A removed version-0 name.
        let mut lost = good.clone();
        lost.names.retain(|n| n.version != 0);
        assert_eq!(check_preserved(&input, &lost), Err(Preserved::LiveIn));

        // A block that vanished.
        let mut gone = good.clone();
        gone.blocks.remove(&0x1020);
        assert_eq!(check_preserved(&input, &gone), Err(Preserved::Blocks));
    }

    // -- 23, 24: determinism, idempotence, stats ---------------------------

    #[test]
    fn optimization_is_deterministic_and_idempotent() {
        let input = build(&cross_block_constant());
        let (a, sa) = run(&input);
        let (b, sb) = run(&input);
        assert_eq!(a, b);
        assert_eq!(text(&a), text(&b));
        assert_eq!(sa, sb);

        let (again, stats) = run(&a);
        assert_eq!(again, a, "a second pass changes nothing");
        assert_eq!(stats.rewrites, 0);
        assert_eq!(stats.phis_removed, 0);
        assert_eq!(stats.rounds, 1, "round 1 already proves the fixpoint");
    }

    #[test]
    fn the_stats_are_exact_on_a_known_fixture() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010, 0x1020]),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![
                        assign(ra(2, Width::W64), read(ra(1, Width::W64))),
                        assign(ra(3, Width::W64), read(ra(1, Width::W64))),
                    ],
                    vec![],
                ),
            ],
        );
        let (_, stats) = opt(&f);
        assert_eq!(stats.rewrites, 2, "two uses of the collapsed φ");
        assert_eq!(stats.phis_removed, 1);
        assert_eq!(stats.names_removed, 1);
        assert!(!stats.capped);
        assert_eq!(stats.rounds, 2, "one changing round, then the no-op");
    }

    // -- the copy-cycle guard ----------------------------------------------

    #[test]
    fn resolve_answers_cycle_instead_of_looping() {
        // The transient a -> b -> a shape an optimistic solver can hold
        // mid-fixpoint before external inputs propagate.
        let values = vec![Value::Copy(1), Value::Copy(0)];
        assert_eq!(resolve(&values, 0), Resolved::Cycle);
        assert_eq!(resolve(&values, 1), Resolved::Cycle);
        // A self-copy, and a longer ring.
        assert_eq!(resolve(&[Value::Copy(0)], 0), Resolved::Cycle);
        let ring = vec![Value::Copy(1), Value::Copy(2), Value::Copy(0)];
        assert_eq!(resolve(&ring, 0), Resolved::Cycle);
        // A chain that ends is still followed to its root.
        let chain = vec![Value::Copy(1), Value::Copy(2), Value::Bottom];
        assert_eq!(resolve(&chain, 0), Resolved::Name(2));
    }

    #[test]
    fn a_phi_argument_in_a_copy_cycle_meets_to_bottom() {
        // φ_a = phi(x still Top, u) with u := b, φ_b = phi(y, v) with
        // v := a — the mutually-referencing φs of the review note. Names:
        // 0 = φ_a, 1 = φ_b, 2 = x (Top), 3 = u (Copy of φ_b).
        let values = vec![Value::Copy(1), Value::Copy(0), Value::Top, Value::Copy(1)];
        assert_eq!(
            meet_phi(&values, 0, &[(Some(0x1000), 2), (Some(0x1010), 3)]),
            Value::Bottom,
            "an unresolvable argument makes the meet Bottom, not a hang"
        );
        // The other direction of the same cycle.
        assert_eq!(meet_phi(&values, 1, &[(Some(0x1000), 0)]), Value::Bottom);
    }

    #[test]
    fn mutually_referencing_loop_phis_terminate() {
        // Two loops whose φs feed each other through copies: the shape the
        // transient cycle would arise in. It must optimize, not hang.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010]),
                block(
                    0x1010,
                    vec![
                        assign(ra(0, Width::W64), read(ra(1, Width::W64))),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1020, Width::W64),
                        },
                    ],
                    vec![0x1020, 0x1030],
                ),
                block(
                    0x1020,
                    vec![
                        assign(ra(1, Width::W64), read(ra(0, Width::W64))),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1010, Width::W64),
                        },
                    ],
                    vec![0x1010, 0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(2, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let (out, stats) = opt(&f);
        assert!(!stats.capped);
        assert_eq!(irssa::check(&out), Ok(()));
    }

    // -- 25: seeded sweep ---------------------------------------------------

    /// xorshift64* with a fixed seed: deterministic, no wall clock.
    fn next(s: &mut u64) -> u64 {
        *s ^= *s >> 12;
        *s ^= *s << 25;
        *s ^= *s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A deterministic stream of small random CFGs, shared by the two
    /// seeded sweeps. `loads` adds a load-bearing assignment to the
    /// statement mix — the shape [`eliminate_dead`] must keep — and is
    /// off for the propagation sweep so that corpus stays exactly the one
    /// slice 3 shipped with.
    fn random_functions(count: usize, seed: u64, loads: bool) -> Vec<irlift::LiftedFunction> {
        let mut s = seed;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let nblocks = 2 + next(&mut s) % 6;
            let vas: Vec<u64> = (0..nblocks).map(|i| 0x1000 + 0x10 * i).collect();
            let mut blocks = Vec::new();
            for &va in &vas {
                let nstmts = (next(&mut s) % 6) as usize;
                let mut list = Vec::new();
                for _ in 0..nstmts {
                    let r = (next(&mut s) % 4) as u16;
                    let r2 = (next(&mut s) % 4) as u16;
                    let k = next(&mut s) % 100;
                    let kinds = if loads { 9 } else { 8 };
                    list.push(match next(&mut s) % kinds {
                        8 => assign(
                            ra(r, Width::W64),
                            Expr::load(read(ra(r2, Width::W64)), Width::W64),
                        ),
                        0 => assign(ra(r, Width::W64), c(k, Width::W64)),
                        1 => assign(ra(r, Width::W64), read(ra(r2, Width::W64))),
                        2 => assign(
                            Reg::flag(Flag::Zero),
                            bin(BinOp::Eq, read(ra(r, Width::W64)), read(ra(r2, Width::W64))),
                        ),
                        3 => assign(ra(r, Width::W32), c(k, Width::W32)),
                        4 => Stmt::Store {
                            addr: read(ra(r, Width::W64)),
                            value: read(ra(r2, Width::W64)),
                        },
                        5 => assign(
                            ra(r, Width::W64),
                            bin(BinOp::Add, read(ra(r2, Width::W64)), c(k, Width::W64)),
                        ),
                        6 => assign(
                            ra(r, Width::W64),
                            Expr::unary(UnOp::ZeroExtend(Width::W64), read(ra(r2, Width::W32))),
                        ),
                        _ => Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1000, Width::W64),
                        },
                    });
                }
                let nsucc = (next(&mut s) % 3) as usize;
                let succ = (0..nsucc)
                    .map(|_| vas[(next(&mut s) % nblocks) as usize])
                    .collect();
                blocks.push(block(va, list, succ));
            }
            out.push(func(0x1000, blocks));
        }
        out
    }

    #[test]
    fn sweep_random_small_cfgs_always_optimize_and_check() {
        for f in random_functions(400, 0x5EED_1A5E_0DDB_5BAD, false) {
            let input = build(&f);
            let (out, stats) = run(&input);
            assert!(!stats.capped, "the cap is unreachable on real input");
            // Determinism: the same input optimizes to the same bytes.
            let (again, _) = optimize(&input);
            assert_eq!(out, again);
            assert_eq!(text(&out), text(&again));
            // Idempotence: a second pass finds nothing left.
            let (twice, s2) = run(&out);
            assert_eq!(twice, out);
            assert_eq!(s2.rewrites, 0);
            assert_eq!(s2.phis_removed, 0);
        }
    }

    // =======================================================================
    // Dead-code elimination
    // =======================================================================

    /// The x86-64 live-out set, the one the `redump` pipeline uses.
    fn live_out() -> Vec<Reg> {
        callfx::function_live_out(crate::model::Arch::X86_64).expect("x86-64 is modeled")
    }

    /// Sweep with an explicit live-out set, insisting on the pass's two
    /// promises: the output is well-formed SSA, and every deletion is
    /// justified.
    fn sweep_with(input: &SsaFunction, lo: &[Reg]) -> (SsaFunction, DceStats) {
        let (out, stats) = eliminate_dead(input, lo);
        assert_eq!(irssa::check(&out), Ok(()), "output must pass irssa::check");
        assert_eq!(
            check_swept(input, &out, lo),
            Ok(()),
            "every deletion must be justified"
        );
        (out, stats)
    }

    /// [`sweep_with`] under the architecture's own live-out set.
    fn sweep(input: &SsaFunction) -> (SsaFunction, DceStats) {
        sweep_with(input, &live_out())
    }

    // -- 1: the canonical cmp+jcc fixture -----------------------------------

    /// What an x86 `cmp rax, rbx` followed by `je` lifts to: four flag
    /// definitions, exactly one of which the branch reads.
    fn cmp_then_jcc() -> irlift::LiftedFunction {
        let (a, b) = (read(ra(0, Width::W64)), read(ra(3, Width::W64)));
        func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(Reg::flag(Flag::Zero), bin(BinOp::Eq, a.clone(), b.clone())),
                        assign(Reg::flag(Flag::Sign), bin(BinOp::Slt, a.clone(), b.clone())),
                        assign(
                            Reg::flag(Flag::Carry),
                            bin(BinOp::Ult, a.clone(), b.clone()),
                        ),
                        assign(Reg::flag(Flag::Overflow), bin(BinOp::Sle, a, b)),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1020, Width::W64),
                        },
                    ],
                    vec![0x1010, 0x1020],
                ),
                block(0x1010, vec![], vec![]),
                block(0x1020, vec![], vec![]),
            ],
        )
    }

    #[test]
    fn of_a_cmps_flag_writes_only_the_one_the_branch_reads_survives() {
        let input = build(&cmp_then_jcc());
        let (out, stats) = sweep(&input);
        assert_eq!(stats.stmts_removed, 3, "SF, CF and OF are dead");
        assert_eq!(stats.names_removed, 3);
        assert_eq!(stats.kept_loads, 0);
        assert_eq!(stats.phis_removed, 0);
        assert_eq!(
            text(&out),
            "; sub_1000 @ 0x0000000000001000 (ssa)\n\
             ; live-in: rax#0, rbx#0\n\
             loc_1000:\n\
             \x20 ZF#1 := (rax#0 == rbx#0)\n\
             \x20 goto if ZF#1 0x1020.q\n\
             \x20 ; -> loc_1010, loc_1020\n\
             loc_1010:\n\
             \x20 ; -> (none)\n\
             loc_1020:\n\
             \x20 ; -> (none)\n"
        );
    }

    // -- 2: a flag consumed later -------------------------------------------

    #[test]
    fn a_flag_definition_consumed_two_blocks_later_survives() {
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(
                        Reg::flag(Flag::Zero),
                        bin(BinOp::Eq, read(ra(0, Width::W64)), c(5, Width::W64)),
                    )],
                    vec![0x1010],
                ),
                block(0x1010, vec![], vec![0x1020]),
                block(
                    0x1020,
                    vec![Stmt::Branch {
                        kind: BranchKind::Jump,
                        cond: Some(read(Reg::flag(Flag::Zero))),
                        target: c(0x1000, Width::W64),
                    }],
                    vec![0x1000],
                ),
            ],
        );
        let input = build(&f);
        let (out, stats) = sweep(&input);
        assert_eq!(stats.stmts_removed, 0, "the branch two blocks on reads it");
        assert_eq!(out, input);
    }

    // -- 3: dead chains and shared roots ------------------------------------

    #[test]
    fn a_dead_temp_chain_is_swept_whole() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(
                        Reg::temp(0, Width::W64),
                        bin(BinOp::Add, read(ra(0, Width::W64)), c(1, Width::W64)),
                    ),
                    assign(ra(1, Width::W64), read(Reg::temp(0, Width::W64))),
                    assign(ra(6, Width::W64), read(ra(1, Width::W64))),
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = sweep(&input);
        assert_eq!(stats.stmts_removed, 3, "rcx and rsi are not live-out");
        assert!(stmts(&out, 0x1000).is_empty());
        assert_eq!(out.names.len(), input.names.len() - 3);
    }

    #[test]
    fn a_shared_root_stays_when_one_of_its_uses_is_live() {
        // The same chain, plus a store that reads its root: the root and
        // the store stay, the dead branch of the chain goes.
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(
                        Reg::temp(0, Width::W64),
                        bin(BinOp::Add, read(ra(0, Width::W64)), c(1, Width::W64)),
                    ),
                    assign(ra(1, Width::W64), read(Reg::temp(0, Width::W64))),
                    Stmt::Store {
                        addr: c(0x40, Width::W64),
                        value: read(Reg::temp(0, Width::W64)),
                    },
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = sweep(&input);
        assert_eq!(stats.stmts_removed, 1, "only the dead rcx copy");
        let s = stmts(&out, 0x1000);
        assert_eq!(s.len(), 2);
        let Stmt::Assign { dst, .. } = &s[0] else {
            panic!("the shared temporary root stays")
        };
        assert_eq!(out.names[dst.num as usize].space, Space::Temp);
        assert!(matches!(&s[1], Stmt::Store { .. }));
    }

    // -- 4: a φ swept with its only reader ----------------------------------

    #[test]
    fn a_phi_feeding_only_a_dead_assign_is_swept_in_the_same_pass() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010, 0x1020]),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), c(1, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), c(2, Width::W64))],
                    vec![0x1030],
                ),
                // rsi is neither live-out nor read, so the φ over rcx
                // feeds nothing at all.
                block(
                    0x1030,
                    vec![assign(ra(6, Width::W64), read(ra(1, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        assert_eq!(phi_count(&input), 1, "the input has the φ");
        let (out, stats) = sweep(&input);
        assert_eq!(phi_count(&out), 0, "no iteration needed");
        assert_eq!(stats.phis_removed, 1);
        assert_eq!(stats.stmts_removed, 3, "the use and both φ arguments");
        assert_eq!(stats.names_removed, 4);
        for b in out.blocks.values() {
            assert!(b.stmts.is_empty() && b.phis.is_empty());
        }
    }

    // -- 5: live-out protection ---------------------------------------------

    #[test]
    fn a_dead_looking_definition_of_a_live_out_cell_survives() {
        // rax (return), rbx (callee-saved) and rsp (the stack) are
        // pinned; rcx (caller-saved, no call after) is not.
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), c(5, Width::W64)),
                    assign(ra(3, Width::W64), c(6, Width::W64)),
                    assign(ra(4, Width::W64), c(7, Width::W64)),
                    assign(ra(1, Width::W64), c(8, Width::W64)),
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = sweep(&input);
        assert_eq!(stats.stmts_removed, 1, "only rcx");
        let cells: Vec<u16> = stmts(&out, 0x1000)
            .iter()
            .map(|s| match s {
                Stmt::Assign { dst, .. } => out.names[dst.num as usize].cell,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(cells, vec![0, 3, 4], "rax, rbx and rsp are live-out");

        // With an empty live-out set every one of them goes: the roots,
        // not the sweep, are what protect them.
        let (bare, stats) = sweep_with(&input, &[]);
        assert_eq!(stats.stmts_removed, 4);
        assert!(stmts(&bare, 0x1000).is_empty());
    }

    // -- 6: the load doctrine -----------------------------------------------

    #[test]
    fn a_dead_load_bearing_assign_is_kept_and_counted() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(Reg::temp(0, Width::W64), c(0x40, Width::W64)),
                    // Nothing reads rcx#1, but the load may fault.
                    assign(
                        ra(1, Width::W64),
                        Expr::load(read(Reg::temp(0, Width::W64)), Width::W64),
                    ),
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = sweep(&input);
        assert_eq!(stats.stmts_removed, 0);
        assert_eq!(stats.kept_loads, 1, "dead, load-bearing, honestly kept");
        // Its address computation is kept too: a kept statement never
        // reads a name that went away.
        assert_eq!(out, input);
    }

    // -- 7: argument setups across a call -----------------------------------

    #[test]
    fn an_argument_setup_before_a_call_survives_the_sweep() {
        let input = build(&calling_function());
        let (out, stats) = sweep(&input);
        // The post-call copy into rsi is the one dead statement: every
        // other definition is live-out (rax, rbx, rdx, rbp, rsp) or read
        // by the call effect (rdi).
        assert_eq!(stats.stmts_removed, 1);
        let i = stmts(&out, 0x1000)
            .iter()
            .position(|s| matches!(s, Stmt::Intrinsic { .. }))
            .expect("the call effect survives");
        assert_eq!(stmts(&out, 0x1000)[i], stmts(&input, 0x1000)[i]);
        let Stmt::Intrinsic { reads, .. } = &stmts(&out, 0x1000)[i] else {
            unreachable!()
        };
        let rdi = reads
            .iter()
            .find_map(|e| match e {
                Expr::Reg(r) if out.names[r.num as usize].cell == 7 => Some(r.num),
                _ => None,
            })
            .expect("rdi is read at the call");
        assert!(
            stmts(&out, 0x1000)
                .iter()
                .any(|s| matches!(s, Stmt::Assign { dst, .. } if dst.num == rdi)),
            "the argument setup is pinned by the callfx read"
        );
    }

    // -- 8: the effects are never removed -----------------------------------

    #[test]
    fn stores_branches_and_intrinsics_are_never_removed() {
        // Adversarial: every definition in the block is dead, and the
        // intrinsic's writes and the store's operands are unread.
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        Stmt::Intrinsic {
                            name: "adversary",
                            writes: vec![ra(1, Width::W64), Reg::flag(Flag::Zero)],
                            reads: vec![read(ra(6, Width::W64))],
                        },
                        assign(ra(1, Width::W64), c(5, Width::W64)),
                        Stmt::Store {
                            addr: c(0x40, Width::W64),
                            value: c(1, Width::W64),
                        },
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: None,
                            target: c(0x1010, Width::W64),
                        },
                    ],
                    vec![0x1010],
                ),
                block(0x1010, vec![], vec![]),
            ],
        );
        let input = build(&f);
        let (out, stats) = sweep(&input);
        assert_eq!(stats.stmts_removed, 1, "only the dead rcx assign");
        let kinds: Vec<u8> = stmts(&out, 0x1000).iter().map(kind_of).collect();
        assert_eq!(kinds, vec![3, 1, 2], "intrinsic, store, branch");
        // The CFG fields are byte-identical.
        assert_eq!(out.entry, input.entry);
        assert_eq!(out.skipped, input.skipped);
        assert!(out.blocks.keys().eq(input.blocks.keys()));
        for (&va, b) in &out.blocks {
            assert!(shape_eq(&input.blocks[&va], b), "block {va:#x} changed");
        }
    }

    // -- 9: compaction and `partial` ----------------------------------------

    #[test]
    fn sweeping_compacts_the_name_table_and_recomputes_partial() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(3, Width::W64), c(1, Width::W64)), // rbx: live-out
                    assign(ra(1, Width::W32), c(2, Width::W32)), // rcx#1: dead
                    // A wider-than-definition read of it: a `partial`
                    // position that goes away with the statement.
                    assign(ra(6, Width::W64), read(ra(1, Width::W64))),
                ],
                vec![],
            )],
        );
        let input = build(&f);
        assert_eq!(input.partial, vec![(0x1000, 2)]);
        let (out, stats) = sweep(&input);
        assert_eq!(stats.stmts_removed, 2);
        assert_eq!(stats.names_removed, 2);
        assert_eq!(out.names.len(), input.names.len() - 2);
        assert_eq!(out.partial, Vec::new(), "recomputed, not adjusted");
        // The version-0 set is identical, and `live_in` still names
        // exactly it, ascending, after the renumbering.
        let zeros: Vec<u16> = out
            .names
            .iter()
            .enumerate()
            .filter(|(_, n)| n.version == 0)
            .map(|(i, _)| i as u16)
            .collect();
        assert_eq!(out.live_in, zeros);
        let entry_names = |f: &SsaFunction| -> Vec<Name> {
            f.names.iter().filter(|n| n.version == 0).copied().collect()
        };
        assert_eq!(entry_names(&out), entry_names(&input));
        // Every surviving occurrence names a name that still exists.
        for b in out.blocks.values() {
            for s in &b.stmts {
                irssa::for_each_use(s, &mut |r| assert!(out.names.get(r.num as usize).is_some()));
                irssa::for_each_def(s, &mut |r| assert!(out.names.get(r.num as usize).is_some()));
            }
        }
    }

    // -- 10: refusals -------------------------------------------------------

    #[test]
    fn a_malformed_function_is_returned_unswept() {
        let mut broken = build(&cmp_then_jcc());
        broken.live_in.clear(); // no longer the version-0 set
        assert!(irssa::check(&broken).is_err());
        let (out, stats) = eliminate_dead(&broken, &live_out());
        assert_eq!(out, broken, "malformed input is not laundered");
        assert_eq!(stats, DceStats::default());
    }

    #[test]
    fn an_empty_function_sweeps_to_itself() {
        let empty = SsaFunction {
            entry: 0x1000,
            name: None,
            arch: Arch::X86_64,
            blocks: BTreeMap::new(),
            skipped: Vec::new(),
            names: Vec::new(),
            live_in: Vec::new(),
            partial: Vec::new(),
        };
        let (out, stats) = sweep(&empty);
        assert_eq!(out, empty);
        assert_eq!(stats, DceStats::default());
    }

    // -- 11: the sweep check itself -----------------------------------------

    #[test]
    fn check_swept_rejects_every_broken_output() {
        let input = build(&func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), c(5, Width::W64)), // rax: live-out
                    assign(ra(1, Width::W64), c(6, Width::W64)), // rcx#1: dead
                    assign(ra(6, Width::W64), read(ra(1, Width::W64))), // rsi: dead
                    Stmt::Store {
                        addr: c(0x40, Width::W64),
                        value: c(1, Width::W64),
                    },
                ],
                vec![],
            )],
        ));
        let lo = live_out();
        let (good, stats) = sweep(&input);
        assert_eq!(stats.stmts_removed, 2);
        assert_eq!(check_swept(&input, &good, &lo), Ok(()));

        // A removed store.
        let mut no_store = good.clone();
        no_store.blocks.get_mut(&0x1000).unwrap().stmts.pop();
        assert_eq!(
            check_swept(&input, &no_store, &lo),
            Err(Swept::RemovedEffect {
                block: 0x1000,
                index: 3
            })
        );

        // A removed definition of a live-out cell.
        let mut no_rax = input.clone();
        no_rax.blocks.get_mut(&0x1000).unwrap().stmts.remove(0);
        assert_eq!(
            check_swept(&input, &no_rax, &lo),
            Err(Swept::RemovedLiveOut {
                block: 0x1000,
                index: 0
            })
        );

        // A dangling read: rcx#1's definition removed, its reader kept.
        let mut dangling = input.clone();
        dangling.blocks.get_mut(&0x1000).unwrap().stmts.remove(1);
        assert_eq!(check_swept(&input, &dangling, &lo), Err(Swept::Dangling));

        // An added statement is no subsequence.
        let mut added = good.clone();
        added
            .blocks
            .get_mut(&0x1000)
            .unwrap()
            .stmts
            .push(Stmt::Store {
                addr: c(0x48, Width::W64),
                value: c(2, Width::W64),
            });
        assert_eq!(
            check_swept(&input, &added, &lo),
            Err(Swept::NotSubsequence { block: 0x1000 })
        );

        // A changed CFG.
        let mut reshaped = good.clone();
        reshaped.blocks.get_mut(&0x1000).unwrap().truncated = true;
        assert_eq!(
            check_swept(&input, &reshaped, &lo),
            Err(Swept::BlockShape { block: 0x1000 })
        );

        // A removed load-bearing assignment.
        let with_load = build(&func(
            0x1000,
            vec![block(
                0x1000,
                vec![assign(
                    ra(1, Width::W64),
                    Expr::load(c(0x40, Width::W64), Width::W64),
                )],
                vec![],
            )],
        ));
        let mut no_load = with_load.clone();
        no_load.blocks.get_mut(&0x1000).unwrap().stmts.clear();
        assert_eq!(
            check_swept(&with_load, &no_load, &lo),
            Err(Swept::RemovedLoad {
                block: 0x1000,
                index: 0
            })
        );
    }

    #[test]
    fn a_phi_over_a_live_out_cell_is_a_root_and_its_removal_is_rejected() {
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010, 0x1020]),
                block(
                    0x1010,
                    vec![assign(ra(0, Width::W64), c(1, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(0, Width::W64), c(2, Width::W64))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(6, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        assert_eq!(phi_count(&input), 1, "a φ over rax, a live-out cell");
        let (out, stats) = sweep(&input);
        assert_eq!(stats.phis_removed, 0);
        assert_eq!(phi_count(&out), 1);
        // And the check refuses an output that removed it anyway.
        let mut gone = input.clone();
        gone.blocks.get_mut(&0x1030).unwrap().phis.clear();
        assert_eq!(
            check_swept(&input, &gone, &live_out()),
            Err(Swept::RemovedPhiLiveOut {
                block: 0x1030,
                phi: 0
            })
        );
    }

    // -- 12: determinism and idempotence ------------------------------------

    #[test]
    fn the_sweep_is_deterministic_and_idempotent() {
        let input = build(&cmp_then_jcc());
        let (a, sa) = sweep(&input);
        let (b, sb) = sweep(&input);
        assert_eq!(a, b);
        assert_eq!(text(&a), text(&b));
        assert_eq!(sa, sb);
        let (again, stats) = sweep(&a);
        assert_eq!(again, a, "one mark-and-sweep is already a fixpoint");
        assert_eq!(stats, DceStats::default());
    }

    // -- 13: optimize then eliminate_dead -----------------------------------

    #[test]
    fn propagation_then_sweeping_removes_the_definitions_it_orphaned() {
        // The slice-3 φ-collapse fixture: propagation rewrites the merge
        // to read rax#0 and leaves both rcx copies standing; the sweep
        // takes them, and keeps rdx, which is live-out.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010, 0x1020]),
                block(
                    0x1010,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(ra(1, Width::W64), read(ra(0, Width::W64)))],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(2, Width::W64), read(ra(1, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        let (opt, _) = run(&input);
        let (out, stats) = sweep(&opt);
        assert_eq!(stats.stmts_removed, 2, "both orphaned copies");
        assert_eq!(
            text(&out),
            "; sub_1000 @ 0x0000000000001000 (ssa)\n\
             ; live-in: rax#0\n\
             loc_1000:\n\
             \x20 ; -> loc_1010, loc_1020\n\
             loc_1010:\n\
             \x20 ; -> loc_1030\n\
             loc_1020:\n\
             \x20 ; -> loc_1030\n\
             loc_1030:\n\
             \x20 rdx#1 := rax#0\n\
             \x20 ; -> (none)\n"
        );
    }

    // -- 14: seeded sweep ---------------------------------------------------

    #[test]
    fn sweep_random_small_cfgs_always_eliminate_and_check() {
        let lo = live_out();
        for f in random_functions(400, 0x0DDB_5BAD_5EED_1A5E, true) {
            let input = build(&f);
            let (opt, ostats) = run(&input);
            assert!(!ostats.capped);
            let (out, stats) = sweep_with(&opt, &lo);
            // Determinism: the same input sweeps to the same bytes.
            let (again, stats2) = eliminate_dead(&opt, &lo);
            assert_eq!(out, again);
            assert_eq!(text(&out), text(&again));
            assert_eq!(stats, stats2);
            // Idempotence: a second sweep removes nothing. It may still
            // *count* a dead load-bearing assign, which stays dead and
            // stays kept — that counter is a standing admission, not a
            // record of work done.
            let (twice, s2) = sweep_with(&out, &lo);
            assert_eq!(twice, out);
            assert_eq!(s2.stmts_removed, 0);
            assert_eq!(s2.phis_removed, 0);
            assert_eq!(s2.names_removed, 0);
            assert_eq!(s2.kept_loads, stats.kept_loads);
            // The counters agree with the shapes they claim.
            assert_eq!(
                stats.names_removed,
                stats.stmts_removed + stats.phis_removed
            );
        }
    }

    // =======================================================================
    // Expression forwarding
    // =======================================================================

    /// Forward, insisting on the pass's promises: the output is well-formed
    /// SSA, it preserves everything it must, and — the invariant the whole
    /// design rests on — it neither adds nor removes a statement.
    fn fwd(input: &SsaFunction) -> (SsaFunction, FwdStats) {
        let (out, stats) = forward(input);
        assert_eq!(irssa::check(&out), Ok(()), "output must pass irssa::check");
        assert_eq!(check_preserved(input, &out), Ok(()), "output must preserve");
        for (&va, b) in &out.blocks {
            assert_eq!(
                b.stmts.len(),
                input.blocks[&va].stmts.len(),
                "block {va:#x} changed statement count"
            );
        }
        (out, stats)
    }

    /// The `--ssa-opt` pipeline: propagate, forward, sweep — each stage
    /// checked by its own companion.
    fn pipeline(f: &irlift::LiftedFunction) -> SsaFunction {
        let input = build(f);
        let (opt, _) = run(&input);
        let (forwarded, stats) = fwd(&opt);
        assert!(!stats.capped, "the round bound is not reached on real input");
        let (swept, _) = sweep(&forwarded);
        swept
    }

    /// The rendered statement lines of one block, trimmed: the golden view
    /// of what a reader would see.
    fn lines(f: &SsaFunction, va: u64) -> Vec<String> {
        let all = text(f);
        let label = format!("loc_{va:x}:");
        let mut out = Vec::new();
        let mut inside = false;
        for line in all.lines() {
            if line.starts_with("loc_") {
                inside = line == label;
                continue;
            }
            let t = line.trim();
            if inside && !t.starts_with(';') && !t.is_empty() {
                out.push(t.to_string());
            }
        }
        out
    }

    /// A pure balanced tree of `2 * leaves - 1` nodes over rcx and small
    /// non-zero constants — balanced so its *depth* stays far under the
    /// rewrite bound while its node count approaches the cap, non-zero so
    /// no `x + 0` identity shrinks it.
    fn wide_tree(leaves: usize) -> Expr {
        let mut level: Vec<Expr> = (0..leaves.max(1))
            .map(|i| {
                if i % 2 == 0 {
                    read(ra(1, Width::W64))
                } else {
                    c((i % 5 + 1) as u64, Width::W64)
                }
            })
            .collect();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut it = level.into_iter();
            while let Some(a) = it.next() {
                match it.next() {
                    Some(b) => next.push(bin(BinOp::Add, a, b)),
                    None => next.push(a),
                }
            }
            level = next;
        }
        level.pop().unwrap_or_else(|| c(0, Width::W64))
    }

    /// A pure tree of exactly `nodes` nodes: a chain of `+ 1`s for the odd
    /// counts, wrapped in a complement for the even ones.
    fn tree(nodes: usize) -> Expr {
        let wrap = nodes.is_multiple_of(2);
        let target = if wrap { nodes - 1 } else { nodes };
        let mut e = read(ra(1, Width::W64));
        let mut have = 1usize;
        while have + 2 <= target {
            e = bin(BinOp::Add, e, c(1, Width::W64));
            have += 2;
        }
        if wrap {
            e = Expr::unary(UnOp::Not, e);
        }
        e
    }

    // -- 1: the canonical x86 collapse --------------------------------------

    /// A real `cmp rdx, 7` plus a conditional jump, lifted: exactly what the
    /// decoder and the x86 lifter produce, not a hand-written approximation.
    fn lifted_cmp_and_jcc(jcc: u8) -> irlift::LiftedFunction {
        let cmp = crate::x86::decode(&[0x48, 0x83, 0xFA, 0x07], 0x1000).expect("cmp decodes");
        let jmp = crate::x86::decode(&[jcc, 0x04], 0x1004).expect("jcc decodes");
        let stmts = crate::x86_lift::lift_block(&[(cmp, 0x1000), (jmp, 0x1004)]);
        assert_eq!(ir::check(&stmts), Ok(()), "the lift is well-formed");
        func(
            0x1000,
            vec![
                irlift::LiftedBlock {
                    start: 0x1000,
                    end: 0x1006,
                    stmts,
                    successors: vec![0x1006, 0x100a],
                    truncated: false,
                },
                block(0x1006, vec![], vec![]),
                block(0x100a, vec![], vec![]),
            ],
        )
    }

    #[test]
    fn a_lifted_cmp_and_je_collapse_into_an_equality_branch() {
        // The DESIGN slice-5 exit criterion: `cmp rdx, 7` + `je` through
        // optimize → forward → sweep *is* the comparison, with the whole
        // flag computation gone.
        let out = pipeline(&lifted_cmp_and_jcc(0x74));
        assert_eq!(
            lines(&out, 0x1000),
            vec!["goto if (rdx#0 == 0x7.q) 0x100a.q"]
        );
        let rendered = text(&out);
        assert!(
            !rendered.contains("ZF") && !rendered.contains("SF"),
            "no flag plumbing survives:\n{rendered}"
        );
    }

    #[test]
    fn a_lifted_cmp_and_jne_collapse_into_the_opposite_polarity() {
        let out = pipeline(&lifted_cmp_and_jcc(0x75));
        assert_eq!(
            lines(&out, 0x1000),
            vec!["goto if (rdx#0 != 0x7.q) 0x100a.q"]
        );
    }

    #[test]
    fn every_lifted_order_jcc_collapses_into_a_relational_branch() {
        // The condition-recovery exit shape: each signed and unsigned
        // order jcc over a real lifted `cmp rdx, 7`, through
        // optimize → forward → sweep, is a single relational branch with
        // the whole paired flag computation gone.
        for (jcc, want) in [
            (0x7C, "goto if (rdx#0 <s 0x7.q) 0x100a.q"),  // jl
            (0x7D, "goto if (0x7.q <=s rdx#0) 0x100a.q"), // jge
            (0x7E, "goto if (rdx#0 <=s 0x7.q) 0x100a.q"), // jle
            (0x7F, "goto if (0x7.q <s rdx#0) 0x100a.q"),  // jg
            (0x72, "goto if (rdx#0 <u 0x7.q) 0x100a.q"),  // jb
            (0x73, "goto if (0x7.q <=u rdx#0) 0x100a.q"), // jae
            (0x76, "goto if (rdx#0 <=u 0x7.q) 0x100a.q"), // jbe
            (0x77, "goto if (0x7.q <u rdx#0) 0x100a.q"),  // ja
        ] {
            let out = pipeline(&lifted_cmp_and_jcc(jcc));
            assert_eq!(lines(&out, 0x1000), vec![want], "jcc 0x{jcc:02x}");
            let rendered = text(&out);
            for flag in ["ZF", "SF", "CF", "OF"] {
                assert!(
                    !rendered.contains(flag),
                    "jcc 0x{jcc:02x}: no flag plumbing survives:\n{rendered}"
                );
            }
        }
    }

    #[test]
    fn a_pair_split_across_ssa_names_folds_through_forward() {
        // The value-numbering witness end to end (see irflow's "The
        // equality witness"): the 64-bit sum is past FWD_SMALL_NODES with
        // a second, growing use — the fold-shrinks tentative refuses the
        // whole definition, so it stays named. The sign flag spells the
        // 32-bit sum outright; the overflow flag reads the name
        // truncated. Both flags are single-use and splice into the
        // branch, whose re-fold proves the two spellings one value and
        // collapses the pair — keeping the sign half's own spelling.
        let w = Width::W32;
        let t = |e: &Expr| Expr::unary(UnOp::Truncate(w), e.clone());
        let (x, y, z) = (
            read(ra(0, Width::W64)),
            read(ra(1, Width::W64)),
            read(ra(2, Width::W64)),
        );
        // 10 nodes: past the duplication cap.
        let sum64 = bin(
            BinOp::Add,
            Expr::unary(UnOp::ZeroExtend(Width::W64), bin(BinOp::Sub, t(&x), t(&y))),
            bin(BinOp::Add, z.clone(), c(0x32, Width::W64)),
        );
        assert!(expr_nodes(&sum64, 0) > FWD_SMALL_NODES);
        let a_spelled = bin(
            BinOp::Add,
            bin(BinOp::Sub, t(&x), t(&y)),
            bin(BinOp::Add, t(&z), c(0x32, w)),
        );
        let a_named = t(&read(ra(4, Width::W64)));
        let b = read(ra(3, w));
        let sf = bin(
            BinOp::Slt,
            bin(BinOp::Sub, a_spelled.clone(), b.clone()),
            c(0, w),
        );
        let of = bin(
            BinOp::Slt,
            bin(
                BinOp::And,
                bin(BinOp::Xor, a_named.clone(), b.clone()),
                bin(
                    BinOp::Xor,
                    a_named.clone(),
                    bin(BinOp::Sub, a_named.clone(), b.clone()),
                ),
            ),
            c(0, w),
        );
        let input = build(&func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(ra(4, Width::W64), sum64),
                        assign(ra(5, Width::W64), read(ra(4, Width::W64))),
                        assign(Reg::flag(Flag::Sign), sf),
                        assign(Reg::flag(Flag::Overflow), of),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(bin(
                                BinOp::Ne,
                                read(Reg::flag(Flag::Sign)),
                                read(Reg::flag(Flag::Overflow)),
                            )),
                            target: c(0x100a, Width::W64),
                        },
                    ],
                    vec![0x1006, 0x100a],
                ),
                block(0x1006, vec![], vec![]),
                block(0x100a, vec![], vec![]),
            ],
        ));
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 2, "both flags splice; the sum stays named");
        let Some(Stmt::Branch { cond: Some(cond), .. }) = stmts(&out, 0x1000).last() else {
            panic!("the branch survives");
        };
        let Expr::Binary { op: BinOp::Slt, lhs, .. } = cond else {
            panic!("the pair collapses to the relation: {cond:?}");
        };
        assert!(
            matches!(&**lhs, Expr::Binary { op: BinOp::Add, .. }),
            "the kept operand is the sign half's spelled sum: {lhs:?}"
        );
        // The named sum's definition stands untouched, and the round is
        // deterministic.
        assert_eq!(stmts(&out, 0x1000)[0], stmts(&input, 0x1000)[0]);
        assert_eq!(fwd(&input).0, out);
    }

    // -- 2: the duplication cap ---------------------------------------------

    /// `rax := <value>`, then `uses` separate copies of rax.
    fn def_with_uses(value: Expr, uses: usize) -> irlift::LiftedFunction {
        let mut list = vec![assign(ra(0, Width::W64), value)];
        for i in 0..uses {
            list.push(assign(ra(2 + i as u16, Width::W64), read(ra(0, Width::W64))));
        }
        func(0x1000, vec![block(0x1000, list, vec![])])
    }

    #[test]
    fn a_small_compound_definition_forwards_to_every_use() {
        let value = bin(BinOp::Add, read(ra(1, Width::W64)), c(5, Width::W64));
        let input = build(&def_with_uses(value, 3));
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 3, "all three uses take the tree");
        assert_eq!(stats.size_skipped, 0);
        assert!(!stats.capped);
        for i in 1..=3 {
            let Stmt::Assign { value: v, .. } = &stmts(&out, 0x1000)[i] else {
                panic!("expected an assignment")
            };
            assert!(
                matches!(v, Expr::Binary { op: BinOp::Add, .. }),
                "use {i} reads the tree, not the name: {v:?}"
            );
        }
        // The definition itself stays standing: sweeping it is the next
        // pass's job, not this one's.
        assert_eq!(stmts(&out, 0x1000).len(), 4);
        assert_eq!(stmts(&out, 0x1000)[0], stmts(&input, 0x1000)[0]);
    }

    #[test]
    fn a_big_compound_definition_with_several_uses_stays_named() {
        let value = tree(FWD_SMALL_NODES + 1);
        assert_eq!(expr_nodes(&value, 0), FWD_SMALL_NODES + 1);
        let input = build(&def_with_uses(value, 2));
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0, "too big to copy to two sites");
        assert_eq!(out, input);
    }

    #[test]
    fn a_big_compound_definition_with_one_use_forwards() {
        let input = build(&def_with_uses(tree(FWD_SMALL_NODES + 1), 1));
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 1, "one use moves the tree, never copies it");
        let Stmt::Assign { value: v, .. } = &stmts(&out, 0x1000)[1] else {
            panic!("expected an assignment")
        };
        assert_eq!(expr_nodes(v, 0), FWD_SMALL_NODES + 1);
    }

    #[test]
    fn the_duplication_cap_sits_exactly_at_the_constant() {
        // At the constant both uses take the tree; one node past it,
        // neither does.
        assert_eq!(expr_nodes(&tree(FWD_SMALL_NODES), 0), FWD_SMALL_NODES);
        let at = build(&def_with_uses(tree(FWD_SMALL_NODES), 2));
        let (_, stats) = fwd(&at);
        assert_eq!(stats.forwards, 2);

        let past = build(&def_with_uses(tree(FWD_SMALL_NODES + 1), 2));
        let (_, stats) = fwd(&past);
        assert_eq!(stats.forwards, 0);
    }

    // -- 2b: the fold-shrinks exception -------------------------------------

    /// A real `cmp rdx, 7` whose flags feed *two* conditional jumps in two
    /// blocks — the multi-consumer shape the fold-shrinks exception exists
    /// for, lifted by the real decoder and lifter, not approximated.
    fn lifted_cmp_and_two_jccs(jcc1: u8, jcc2: u8) -> irlift::LiftedFunction {
        let cmp = crate::x86::decode(&[0x48, 0x83, 0xFA, 0x07], 0x1000).expect("cmp decodes");
        let j1 = crate::x86::decode(&[jcc1, 0x0a], 0x1004).expect("first jcc decodes");
        let head = crate::x86_lift::lift_block(&[(cmp, 0x1000), (j1, 0x1004)]);
        let j2 = crate::x86::decode(&[jcc2, 0x10], 0x1006).expect("second jcc decodes");
        let second = crate::x86_lift::lift_block(&[(j2, 0x1006)]);
        assert_eq!(ir::check(&head), Ok(()), "the first lift is well-formed");
        assert_eq!(ir::check(&second), Ok(()), "the second lift is well-formed");
        func(
            0x1000,
            vec![
                irlift::LiftedBlock {
                    start: 0x1000,
                    end: 0x1006,
                    stmts: head,
                    successors: vec![0x1006, 0x1010],
                    truncated: false,
                },
                irlift::LiftedBlock {
                    start: 0x1006,
                    end: 0x1008,
                    stmts: second,
                    successors: vec![0x1008, 0x1018],
                    truncated: false,
                },
                block(0x1008, vec![], vec![]),
                block(0x1010, vec![], vec![]),
                block(0x1018, vec![], vec![]),
            ],
        )
    }

    /// The x86 subtraction-overflow shape `((a ^ b) & (a ^ (a - b))) <s 0`
    /// over `a` and `b`, with the inner subtraction spelled by `sub` — the
    /// 11-node tree (or 9 with a temp for the subtraction) the exception
    /// exists to move.
    fn overflow_tree(a: Expr, b: Expr, sub: Expr) -> Expr {
        bin(
            BinOp::Slt,
            bin(
                BinOp::And,
                bin(BinOp::Xor, a.clone(), b),
                bin(BinOp::Xor, a, sub),
            ),
            c(0, Width::W64),
        )
    }

    #[test]
    fn one_cmp_feeding_two_jccs_reads_as_two_relational_branches() {
        // The residual the order-condition slice diagnosed: one `cmp`, two
        // jccs, so the 11-node OF tree has two uses and the plain cap
        // refused it. Under the fold-shrinks exception both branch
        // conditions collapse to relations and every flag write sweeps.
        for (jcc1, jcc2, want1, want2) in [
            (
                0x7Cu8, // jl
                0x7Du8, // jge
                "goto if (rdx#0 <s 0x7.q) 0x1010.q",
                "goto if (0x7.q <=s rdx#0) 0x1018.q",
            ),
            (
                0x7E, // jle
                0x7F, // jg
                "goto if (rdx#0 <=s 0x7.q) 0x1010.q",
                "goto if (0x7.q <s rdx#0) 0x1018.q",
            ),
        ] {
            let out = pipeline(&lifted_cmp_and_two_jccs(jcc1, jcc2));
            assert_eq!(lines(&out, 0x1000), vec![want1.to_string()], "jcc {jcc1:#04x}");
            assert_eq!(lines(&out, 0x1006), vec![want2.to_string()], "jcc {jcc2:#04x}");
            let rendered = text(&out);
            for flag in ["ZF", "SF", "CF", "OF"] {
                assert!(
                    !rendered.contains(flag),
                    "jcc pair {jcc1:#04x}/{jcc2:#04x}: no flag plumbing survives:\n{rendered}"
                );
            }
        }
    }

    #[test]
    fn the_exception_counts_its_splices() {
        let input = build(&lifted_cmp_and_two_jccs(0x7C, 0x7D));
        let (opt, _) = run(&input);
        let (_, stats) = fwd(&opt);
        assert_eq!(
            stats.multi_spliced, 2,
            "the OF tree earned exactly its two branch sites"
        );
        assert!(
            stats.forwards > stats.multi_spliced,
            "the ordinary tiers still account for the rest"
        );
    }

    #[test]
    fn a_big_pure_tree_whose_fold_does_not_shrink_stays_named() {
        // Two compound uses of a past-the-cap tree that no identity
        // collapses: splicing would make both sites bigger, so the
        // exception refuses and the output is byte-identical.
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), tree(FWD_SMALL_NODES + 1)),
                    assign(
                        ra(2, Width::W64),
                        bin(BinOp::Xor, read(ra(0, Width::W64)), read(ra(3, Width::W64))),
                    ),
                    assign(
                        ra(4, Width::W64),
                        bin(BinOp::Xor, read(ra(0, Width::W64)), read(ra(5, Width::W64))),
                    ),
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0, "no site shrinks, so nothing splices");
        assert_eq!(stats.multi_spliced, 0);
        assert_eq!(out, input);
    }

    #[test]
    fn one_growing_site_refuses_the_whole_definition() {
        // The OF tree with one branch site that would collapse beside its
        // inline sign twin, and one arithmetic use that would only grow:
        // the per-def decision keeps the tree named at *both* sites.
        let a = || read(ra(1, Width::W64));
        let of = Reg::flag(Flag::Overflow);
        let sf_shape = bin(
            BinOp::Slt,
            bin(BinOp::Sub, a(), c(7, Width::W64)),
            c(0, Width::W64),
        );
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(
                            of,
                            overflow_tree(
                                a(),
                                c(7, Width::W64),
                                bin(BinOp::Sub, a(), c(7, Width::W64)),
                            ),
                        ),
                        assign(
                            ra(3, Width::W64),
                            bin(
                                BinOp::Or,
                                Expr::unary(UnOp::ZeroExtend(Width::W64), read(of)),
                                c(1, Width::W64),
                            ),
                        ),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(bin(BinOp::Ne, sf_shape, read(of))),
                            target: c(0x1010, Width::W64),
                        },
                    ],
                    vec![0x1004, 0x1010],
                ),
                block(0x1004, vec![], vec![]),
                block(0x1010, vec![], vec![]),
            ],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0, "the growing site refuses the branch site too");
        assert_eq!(out, input);
    }

    #[test]
    fn an_equal_size_fold_is_not_a_shrink() {
        // A past-the-cap tree of constants folds to a single node — the
        // same size a bare-copy site already has. The comparison is
        // strict, so the definition stays named and the output is
        // byte-identical; a constant right-hand side is `optimize`'s to
        // substitute, never this exception's.
        let mut consts = c(1, Width::W64);
        for _ in 0..4 {
            consts = bin(BinOp::Add, consts, c(1, Width::W64));
        }
        assert_eq!(expr_nodes(&consts, 0), FWD_SMALL_NODES + 1);
        let input = build(&def_with_uses(consts, 2));
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0, "equal size is not a shrink");
        assert_eq!(out, input);
    }

    #[test]
    fn a_narrow_only_site_is_dropped_while_a_shrinking_site_earns() {
        // One site reads the name only at a narrower width — no splice can
        // happen there, so it proves nothing and is dropped — while the
        // other site folds strictly smaller (`x - x → 0`) and earns.
        let mut consts = c(1, Width::W64);
        for _ in 0..4 {
            consts = bin(BinOp::Add, consts, c(1, Width::W64));
        }
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), consts),
                    assign(
                        ra(2, Width::W64),
                        bin(BinOp::Sub, read(ra(0, Width::W64)), read(ra(0, Width::W64))),
                    ),
                    assign(
                        ra(3, Width::W64),
                        Expr::unary(UnOp::ZeroExtend(Width::W64), read(ra(0, Width::W32))),
                    ),
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.multi_spliced, 2, "both occurrences at the one shrinking site");
        let Stmt::Assign { value: v, .. } = &stmts(&out, 0x1000)[1] else {
            panic!("expected an assignment")
        };
        assert_eq!(*v, c(0, Width::W64), "the shrinking site folded to zero");
        assert_eq!(
            stmts(&out, 0x1000)[2],
            stmts(&input, 0x1000)[2],
            "the narrow site keeps the name"
        );
    }

    #[test]
    fn a_cross_block_temp_cone_splices_through_the_cascade() {
        // Obstacle two from the plan: the OF tree reads a temp defined in
        // its own block, which a use in another block cannot legally read.
        // The temp's own pure def folds into the tree in an earlier round,
        // and the then temp-free tree crosses — the whole pure def-cone
        // spliced transitively, with `irssa::check` as the arbiter.
        let a = || read(ra(1, Width::W64));
        let t = Reg::temp(0, Width::W64);
        let sf = Reg::flag(Flag::Sign);
        let of = Reg::flag(Flag::Overflow);
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(t, bin(BinOp::Sub, a(), c(7, Width::W64))),
                        assign(sf, bin(BinOp::Slt, read(t), c(0, Width::W64))),
                        assign(of, overflow_tree(a(), c(7, Width::W64), read(t))),
                    ],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![Stmt::Branch {
                        kind: BranchKind::Jump,
                        cond: Some(bin(BinOp::Ne, read(sf), read(of))),
                        target: c(0x1030, Width::W64),
                    }],
                    vec![0x1020, 0x1030],
                ),
                block(
                    0x1020,
                    vec![Stmt::Branch {
                        kind: BranchKind::Jump,
                        cond: Some(bin(BinOp::Eq, read(sf), read(of))),
                        target: c(0x1040, Width::W64),
                    }],
                    vec![0x1030, 0x1040],
                ),
                block(0x1030, vec![], vec![]),
                block(0x1040, vec![], vec![]),
            ],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.multi_spliced, 2, "the temp-free tree earned both branch sites");
        assert_eq!(
            lines(&out, 0x1010),
            vec!["goto if (rcx#0 <s 0x7.q) 0x1030.q"]
        );
        assert_eq!(
            lines(&out, 0x1020),
            vec!["goto if (0x7.q <=s rcx#0) 0x1040.q"]
        );
    }

    #[test]
    fn an_unsplicable_temp_cone_is_refused_with_checks_green() {
        // The refusal half: the temp is a load parked behind a store, so
        // no round can fold it into the tree, and the cross-block sites
        // stay refused — no cross-block temp read is ever emitted, which
        // `fwd`'s `irssa::check` assertion arbitrates.
        let a = || read(ra(1, Width::W64));
        let t = Reg::temp(0, Width::W64);
        let of = Reg::flag(Flag::Overflow);
        let sf_shape = || {
            bin(
                BinOp::Slt,
                bin(BinOp::Sub, a(), c(7, Width::W64)),
                c(0, Width::W64),
            )
        };
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(t, Expr::load(read(ra(2, Width::W64)), Width::W64)),
                        Stmt::Store {
                            addr: read(ra(4, Width::W64)),
                            value: read(ra(5, Width::W64)),
                        },
                        assign(of, overflow_tree(a(), c(7, Width::W64), read(t))),
                    ],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![Stmt::Branch {
                        kind: BranchKind::Jump,
                        cond: Some(bin(BinOp::Ne, sf_shape(), read(of))),
                        target: c(0x1030, Width::W64),
                    }],
                    vec![0x1020, 0x1030],
                ),
                block(
                    0x1020,
                    vec![Stmt::Branch {
                        kind: BranchKind::Jump,
                        cond: Some(bin(BinOp::Eq, sf_shape(), read(of))),
                        target: c(0x1040, Width::W64),
                    }],
                    vec![0x1030, 0x1040],
                ),
                block(0x1030, vec![], vec![]),
                block(0x1040, vec![], vec![]),
            ],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0, "the load-behind-a-store cone cannot move");
        assert_eq!(out, input);
    }

    #[test]
    fn a_load_bearing_tree_never_takes_the_exception() {
        // A past-the-cap load-bearing tree with two uses: the exception is
        // for pure trees only, so the gate refuses before any fold is
        // consulted — N sites would read as N loads.
        let mut value = load_at(1);
        while expr_nodes(&value, 0) <= FWD_SMALL_NODES {
            value = bin(BinOp::Add, value, c(1, Width::W64));
        }
        let input = build(&def_with_uses(value, 2));
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0, "a load never duplicates");
        assert_eq!(out, input);
    }

    #[test]
    fn a_division_bearing_tree_never_takes_the_exception() {
        // Same gate for a division: a potential trap never duplicates,
        // whatever the fold would make of the sites.
        let mut value = division();
        while expr_nodes(&value, 0) <= FWD_SMALL_NODES {
            value = bin(BinOp::Add, value, c(1, Width::W64));
        }
        let input = build(&def_with_uses(value, 2));
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0, "a division never duplicates");
        assert_eq!(out, input);
    }

    #[test]
    fn the_exception_is_deterministic_and_idempotent() {
        let input = build(&lifted_cmp_and_two_jccs(0x7C, 0x7D));
        let (opt, _) = run(&input);
        let (a, sa) = fwd(&opt);
        let (b, sb) = forward(&opt);
        assert_eq!(a, b, "the same input forwards to the same function");
        assert_eq!(text(&a), text(&b));
        assert_eq!(sa, sb);
        let (twice, s2) = fwd(&a);
        assert_eq!(twice, a);
        assert_eq!(s2.forwards, 0);
        assert_eq!(s2.multi_spliced, 0);
    }

    // -- 3: loads -----------------------------------------------------------

    fn load_at(reg: u16) -> Expr {
        Expr::load(read(ra(reg, Width::W64)), Width::W64)
    }

    #[test]
    fn a_load_forwards_within_its_block_when_nothing_intervenes() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), load_at(1)),
                    assign(ra(2, Width::W64), read(ra(0, Width::W64))),
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 1);
        let Stmt::Assign { value: v, .. } = &stmts(&out, 0x1000)[1] else {
            panic!("expected an assignment")
        };
        assert!(irflow::contains_load(v, 0), "the use reads the load: {v:?}");
    }

    #[test]
    fn a_load_never_crosses_a_store_a_call_or_an_intrinsic() {
        let barriers = [
            Stmt::Store {
                addr: read(ra(3, Width::W64)),
                value: read(ra(4, Width::W64)),
            },
            // A call is a `Branch` *and*, after `callfx`, an intrinsic:
            // either gate catches it.
            Stmt::Branch {
                kind: BranchKind::Call,
                cond: None,
                target: c(0x2000, Width::W64),
            },
            Stmt::Intrinsic {
                name: "barrier",
                writes: vec![ra(7, Width::W64)],
                reads: vec![],
            },
        ];
        for barrier in barriers {
            let f = func(
                0x1000,
                vec![block(
                    0x1000,
                    vec![
                        assign(ra(0, Width::W64), load_at(1)),
                        barrier.clone(),
                        assign(ra(2, Width::W64), read(ra(0, Width::W64))),
                    ],
                    vec![],
                )],
            );
            let input = build(&f);
            let (out, stats) = fwd(&input);
            assert_eq!(stats.forwards, 0, "crossed {barrier:?}");
            assert_eq!(out, input);
        }
    }

    /// A `W1` comparison over the fixture load — the load-bearing branch
    /// condition the joint splice targets.
    fn load_cond_def() -> Stmt {
        assign(
            ra(0, Width::W1),
            bin(BinOp::Ult, load_at(1), read(ra(3, Width::W64))),
        )
    }
    fn cond_branch(cond: Reg, target: u64) -> Stmt {
        Stmt::Branch {
            kind: BranchKind::Jump,
            cond: Some(read(cond)),
            target: c(target, Width::W64),
        }
    }

    #[test]
    fn a_load_forwards_across_a_block_boundary_when_the_region_is_clear() {
        // The load-cone joint splice: a load-bearing condition whose
        // def→use region is provably effect-clear moves into the branch —
        // the definition sweeps, so the load still renders exactly once.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![load_cond_def()], vec![0x1010]),
                block(
                    0x1010,
                    vec![cond_branch(ra(0, Width::W1), 0x1020)],
                    vec![0x1018, 0x1020],
                ),
                block(0x1018, vec![], vec![]),
                block(0x1020, vec![], vec![]),
            ],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 1, "the clear region admits the move");
        assert_eq!(stats.load_pair_spliced, 1);
        assert_ne!(out, input);
        assert!(irssa::check(&out).is_ok());
    }

    #[test]
    fn a_load_refuses_a_site_that_is_not_a_branch_condition() {
        // The same clear region, but the use is a standing assignment:
        // relocating the load would only move it — and strand a
        // previously pure tree outside the pure fold-shrinks tier — so
        // the joint splice is branch-conditions-only.
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W64), load_at(1))],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![assign(ra(2, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0, "an assignment site refuses");
        assert_eq!(out, input);
    }

    #[test]
    fn a_load_refuses_a_block_boundary_when_the_region_is_dirty() {
        // Same shape, but the region carries an effect between the load
        // and its use — a store in the target block, then a store in an
        // intermediate block, then a call terminator on the path.
        let store_before_use = func(
            0x1000,
            vec![
                block(0x1000, vec![load_cond_def()], vec![0x1010]),
                block(
                    0x1010,
                    vec![
                        Stmt::Store {
                            addr: c(0x2000, Width::W64),
                            value: c(0, Width::W64),
                        },
                        cond_branch(ra(0, Width::W1), 0x1020),
                    ],
                    vec![0x1018, 0x1020],
                ),
                block(0x1018, vec![], vec![]),
                block(0x1020, vec![], vec![]),
            ],
        );
        let store_between = func(
            0x1000,
            vec![
                block(0x1000, vec![load_cond_def()], vec![0x1010]),
                block(
                    0x1010,
                    vec![Stmt::Store {
                        addr: c(0x2000, Width::W64),
                        value: c(0, Width::W64),
                    }],
                    vec![0x1020],
                ),
                block(
                    0x1020,
                    vec![cond_branch(ra(0, Width::W1), 0x1030)],
                    vec![0x1028, 0x1030],
                ),
                block(0x1028, vec![], vec![]),
                block(0x1030, vec![], vec![]),
            ],
        );
        let call_on_path = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        load_cond_def(),
                        Stmt::Branch {
                            kind: BranchKind::Call,
                            cond: None,
                            target: c(0x3000, Width::W64),
                        },
                    ],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![cond_branch(ra(0, Width::W1), 0x1020)],
                    vec![0x1018, 0x1020],
                ),
                block(0x1018, vec![], vec![]),
                block(0x1020, vec![], vec![]),
            ],
        );
        for f in [store_before_use, store_between, call_on_path] {
            let input = build(&f);
            let (out, stats) = fwd(&input);
            assert_eq!(stats.forwards, 0, "the dirty region refuses");
            assert_eq!(out, input);
        }
    }

    #[test]
    fn a_load_refuses_a_cyclic_region() {
        // The use block loops back through the definition's block: a
        // re-arrival would re-read after the region's own statements, so
        // the acyclicity gate refuses outright.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![load_cond_def()], vec![0x1010]),
                block(
                    0x1010,
                    vec![cond_branch(ra(0, Width::W1), 0x1010)],
                    vec![0x1010, 0x1020],
                ),
                block(0x1020, vec![], vec![]),
            ],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0, "a cyclic region refuses");
        assert_eq!(out, input);
    }

    #[test]
    fn a_load_refuses_a_region_past_the_block_cap() {
        // Four effect-free blocks between the load and its use: the
        // region (six blocks) is over MAX_LOAD_REGION_BLOCKS and refuses
        // outright — a documented non-move, never an approximation.
        let mut blocks = vec![block(0x1000, vec![load_cond_def()], vec![0x1010])];
        for i in 0..4u64 {
            blocks.push(block(0x1010 + 0x10 * i, vec![], vec![0x1020 + 0x10 * i]));
        }
        blocks.push(block(
            0x1050,
            vec![cond_branch(ra(0, Width::W1), 0x1060)],
            vec![0x1058, 0x1060],
        ));
        blocks.push(block(0x1058, vec![], vec![]));
        blocks.push(block(0x1060, vec![], vec![]));
        let f = func(0x1000, blocks);
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0, "an over-cap region refuses");
        assert_eq!(out, input);
    }

    #[test]
    fn a_load_with_a_phi_use_refuses_whole() {
        // The load's value reaches a φ: that use can never clear, so the
        // all-or-nothing rule refuses the definition entirely — a partial
        // clearing would render the load both standing and inline.
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(ra(0, Width::W64), load_at(1)),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1010, Width::W64),
                        },
                    ],
                    vec![0x1010, 0x1020],
                ),
                block(
                    0x1010,
                    vec![assign(
                        ra(0, Width::W64),
                        bin(BinOp::Add, read(ra(0, Width::W64)), c(1, Width::W64)),
                    )],
                    vec![0x1020],
                ),
                block(
                    0x1020,
                    vec![assign(ra(2, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.load_pair_spliced, 0, "a φ use refuses the def");
        assert_eq!(out, input);
    }

    #[test]
    fn a_division_in_the_cone_refuses() {
        // The cone inlines a load-backed temp, but the definition also
        // divides: a potential trap never moves, cone or no cone.
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(Reg::temp(0, Width::W64), load_at(1)),
                        assign(
                            ra(2, Width::W64),
                            bin(
                                BinOp::UDiv,
                                read(Reg::temp(0, Width::W64)),
                                read(ra(3, Width::W64)),
                            ),
                        ),
                    ],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![assign(ra(4, Width::W64), read(ra(2, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.load_pair_spliced, 0, "a division in the cone refuses");
        // The single-use temp load may forward within its block (the
        // existing tier); the division itself must not cross the boundary.
        let Some(Stmt::Assign { value, .. }) =
            out.blocks.get(&0x1010).and_then(|b| b.stmts.first())
        else {
            panic!("the use block keeps its statement");
        };
        assert!(
            matches!(value, Expr::Reg(_)),
            "the division stays in its own block: {value:?}"
        );
    }

    #[test]
    fn the_memory_operand_cmp_with_two_jccs_composes_end_to_end() {
        // The milestone comparator's exact shape, from the real bytes:
        // `cmp [rdx+0x30], r8` feeding `jg` then `jge` one block later.
        // Through construct → optimize → forward → eliminate_dead both
        // branches must read as relations, the second with the load
        // inline — the load-cone joint splice composed with irflow's
        // one-expression pair equality.
        use crate::model::Arch;
        use crate::{irlift, x86, x86_lift};

        let block = |start: u64, bytes: &[u8], successors: Vec<u64>| {
            let mut insns = Vec::new();
            let mut off = 0usize;
            while off < bytes.len() {
                let insn = x86::decode(&bytes[off..], start + off as u64)
                    .expect("fixture bytes decode");
                let len = insn.length as usize;
                insns.push((insn, start + off as u64));
                off += len;
            }
            irlift::LiftedBlock {
                start,
                end: start + bytes.len() as u64,
                stmts: x86_lift::lift_block(&insns),
                successors,
                truncated: false,
            }
        };
        let lifted = irlift::LiftedFunction {
            entry: 0x1000,
            name: None,
            arch: Arch::X86_64,
            blocks: [
                // cmp [rdx+0x30], r8 ; jg 0x1020
                block(
                    0x1000,
                    &[0x4C, 0x39, 0x42, 0x30, 0x7F, 0x1A],
                    vec![0x1006, 0x1020],
                ),
                // jge 0x100f
                block(0x1006, &[0x7D, 0x07], vec![0x1008, 0x100F]),
                // mov eax, -1 ; jmp 0x1020
                block(
                    0x1008,
                    &[0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xEB, 0x11],
                    vec![0x1020],
                ),
                // mov eax, 1
                block(0x100F, &[0xB8, 0x01, 0x00, 0x00, 0x00], vec![0x1020]),
                // ret
                block(0x1020, &[0xC3], vec![]),
            ]
            .into_iter()
            .map(|b| (b.start, b))
            .collect(),
        };
        let ssa = irssa::construct(&lifted).expect("the comparator shape constructs");
        let (opt, _) = optimize(&ssa);
        let (fwded, stats) = forward(&opt);
        assert!(
            stats.load_pair_spliced >= 1,
            "the joint splice fires: {stats:?}"
        );
        let (out, _) = eliminate_dead(&fwded, &[]);
        assert!(irssa::check(&out).is_ok());
        // The second jcc's condition reads the load inline and no flag
        // shape survives anywhere: every branch condition is a relation
        // over values, not flag names.
        let cond_of = |va: u64| {
            let Some(Stmt::Branch { cond: Some(c), .. }) =
                out.blocks.get(&va).and_then(|b| b.stmts.last())
            else {
                panic!("block {va:#x} ends in a conditional branch");
            };
            c.clone()
        };
        let jge = cond_of(0x1006);
        assert!(
            irflow::contains_load(&jge, 0),
            "the jge reads the load inline: {jge:?}"
        );
        for va in [0x1000u64, 0x1006] {
            let mut flags = 0usize;
            irssa::expr_regs(&cond_of(va), 0, &mut |r| {
                if out
                    .names
                    .get(r.num as usize)
                    .is_some_and(|n| n.space == Space::Flag)
                {
                    flags += 1;
                }
            });
            assert_eq!(flags, 0, "no flag name survives in {va:#x}'s condition");
        }
    }

    #[test]
    fn a_multi_use_load_never_forwards() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), load_at(1)),
                    assign(ra(2, Width::W64), read(ra(0, Width::W64))),
                    assign(ra(3, Width::W64), read(ra(0, Width::W64))),
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0, "two sites would read as two loads");
        assert_eq!(out, input);
    }

    // -- 4: divisions -------------------------------------------------------

    fn division() -> Expr {
        bin(BinOp::UDiv, read(ra(1, Width::W64)), read(ra(3, Width::W64)))
    }

    #[test]
    fn a_division_forwards_within_its_block_before_a_branch() {
        let f = func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(ra(0, Width::W64), division()),
                        assign(ra(2, Width::W64), read(ra(0, Width::W64))),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: None,
                            target: c(0x1010, Width::W64),
                        },
                    ],
                    vec![0x1010],
                ),
                block(0x1010, vec![], vec![]),
            ],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 1);
        let Stmt::Assign { value: v, .. } = &stmts(&out, 0x1000)[1] else {
            panic!("expected an assignment")
        };
        assert!(irflow::contains_div(v, 0), "the use reads the division: {v:?}");
    }

    #[test]
    fn a_division_never_crosses_a_branch_or_a_block_boundary() {
        // Past a guard in its own block: the trap must not move past the
        // branch that may be exactly what avoids it.
        let guarded = build(&func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![
                        assign(ra(0, Width::W64), division()),
                        Stmt::Branch {
                            kind: BranchKind::Jump,
                            cond: Some(read(Reg::flag(Flag::Zero))),
                            target: c(0x1010, Width::W64),
                        },
                        assign(ra(2, Width::W64), read(ra(0, Width::W64))),
                    ],
                    vec![0x1010],
                ),
                block(0x1010, vec![], vec![]),
            ],
        ));
        let (out, stats) = fwd(&guarded);
        assert_eq!(stats.forwards, 0);
        assert_eq!(out, guarded);

        // And never into another block, guard or no guard.
        let crossing = build(&func(
            0x1000,
            vec![
                block(
                    0x1000,
                    vec![assign(ra(0, Width::W64), division())],
                    vec![0x1010],
                ),
                block(
                    0x1010,
                    vec![assign(ra(2, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        ));
        let (out, stats) = fwd(&crossing);
        assert_eq!(stats.forwards, 0);
        assert_eq!(out, crossing);
    }

    // -- 5: widths ----------------------------------------------------------

    #[test]
    fn a_narrower_read_keeps_the_name_while_its_exact_sibling_forwards() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(
                        ra(0, Width::W64),
                        bin(BinOp::Add, read(ra(1, Width::W64)), c(5, Width::W64)),
                    ),
                    // A 32-bit read of a 64-bit definition: splicing a tree
                    // under it would need a truncating wrapper node.
                    assign(
                        ra(2, Width::W64),
                        Expr::unary(UnOp::ZeroExtend(Width::W64), read(ra(0, Width::W32))),
                    ),
                    assign(ra(3, Width::W64), read(ra(0, Width::W64))),
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 1, "only the exact-width read takes the tree");
        assert_eq!(
            stmts(&out, 0x1000)[1],
            stmts(&input, 0x1000)[1],
            "the narrow read is untouched"
        );
        let Stmt::Assign { value: v, .. } = &stmts(&out, 0x1000)[2] else {
            panic!("expected an assignment")
        };
        assert!(matches!(v, Expr::Binary { .. }), "{v:?}");
    }

    // -- 6: the two positions never rewritten -------------------------------

    #[test]
    fn an_intrinsic_read_is_never_forwarded_into() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(
                        ra(0, Width::W64),
                        bin(BinOp::Add, read(ra(1, Width::W64)), c(5, Width::W64)),
                    ),
                    Stmt::Intrinsic {
                        name: "callfx",
                        writes: vec![ra(7, Width::W64)],
                        reads: vec![read(ra(0, Width::W64))],
                    },
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(
            stats.forwards, 0,
            "the callfx register identity is the point"
        );
        assert_eq!(out, input);
    }

    #[test]
    fn phi_arguments_are_byte_identical_through_forwarding() {
        // A diamond whose arms each compute a compound value; the merge
        // reads the φ, which merges *versions* and takes no expression.
        let f = func(
            0x1000,
            vec![
                block(0x1000, vec![], vec![0x1010, 0x1020]),
                block(
                    0x1010,
                    vec![assign(
                        ra(0, Width::W64),
                        bin(BinOp::Add, read(ra(1, Width::W64)), c(1, Width::W64)),
                    )],
                    vec![0x1030],
                ),
                block(
                    0x1020,
                    vec![assign(
                        ra(0, Width::W64),
                        bin(BinOp::Add, read(ra(1, Width::W64)), c(2, Width::W64)),
                    )],
                    vec![0x1030],
                ),
                block(
                    0x1030,
                    vec![assign(ra(2, Width::W64), read(ra(0, Width::W64)))],
                    vec![],
                ),
            ],
        );
        let input = build(&f);
        assert_eq!(phi_count(&input), 1, "the fixture has the φ");
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0, "a φ argument names a version");
        assert_eq!(out.blocks[&0x1030].phis, input.blocks[&0x1030].phis);
        assert_eq!(out, input);
    }

    // -- 7: cascade, determinism, idempotence -------------------------------

    #[test]
    fn a_chain_cascades_into_one_expression_in_one_call() {
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(
                        ra(0, Width::W64),
                        bin(BinOp::Add, read(ra(1, Width::W64)), c(1, Width::W64)),
                    ),
                    assign(
                        ra(2, Width::W64),
                        bin(BinOp::Mul, read(ra(0, Width::W64)), c(2, Width::W64)),
                    ),
                    assign(ra(3, Width::W64), read(ra(2, Width::W64))),
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert!(stats.rounds >= 2, "the cascade needs more than one round");
        assert!(!stats.capped);
        assert_eq!(lines(&out, 0x1000)[2], "rbx#1 := ((rcx#0 + 0x1.q) * 0x2.q)");
        // A second call finds nothing left: the fixpoint is real.
        let (twice, again) = fwd(&out);
        assert_eq!(twice, out);
        assert_eq!(again.forwards, 0);
    }

    #[test]
    fn forwarding_is_deterministic_and_idempotent() {
        let input = build(&lifted_cmp_and_jcc(0x74));
        let (a, sa) = fwd(&input);
        let (b, sb) = forward(&input);
        assert_eq!(a, b, "the same input forwards to the same function");
        assert_eq!(text(&a), text(&b));
        assert_eq!(sa, sb);
        let (twice, s2) = fwd(&a);
        assert_eq!(twice, a);
        assert_eq!(s2.forwards, 0);
        assert_eq!(s2.size_skipped, sa.size_skipped);
    }

    // -- 8: the node cap ----------------------------------------------------

    #[test]
    fn a_substitution_past_the_node_cap_is_refused_and_counted() {
        // A definition just under the cap and a use site with no room for
        // it: the substitution is refused whole, never truncated.
        let big = wide_tree(2001);
        assert_eq!(expr_nodes(&big, 0), 4001);
        assert!(expr_nodes(&big, 0) <= ir::MAX_EXPR_NODES);
        let f = func(
            0x1000,
            vec![block(
                0x1000,
                vec![
                    assign(ra(0, Width::W64), big),
                    assign(
                        ra(2, Width::W64),
                        bin(BinOp::Add, read(ra(0, Width::W64)), wide_tree(100)),
                    ),
                ],
                vec![],
            )],
        );
        let input = build(&f);
        let (out, stats) = fwd(&input);
        assert_eq!(stats.forwards, 0);
        assert_eq!(stats.size_skipped, 1, "refused once, and said so");
        assert_eq!(out, input, "nothing was truncated to make it fit");
    }

    // -- 10: the contract ---------------------------------------------------

    #[test]
    fn a_malformed_function_forwards_to_itself() {
        let mut broken = build(&cross_block_constant());
        broken.live_in.clear(); // no longer the version-0 set
        assert!(irssa::check(&broken).is_err());
        let (out, stats) = forward(&broken);
        assert_eq!(out, broken, "malformed input is not laundered");
        assert_eq!(stats, FwdStats::default());
    }

    #[test]
    fn an_empty_function_forwards_to_itself() {
        let empty = SsaFunction {
            entry: 0x1000,
            name: None,
            arch: Arch::X86_64,
            blocks: BTreeMap::new(),
            skipped: Vec::new(),
            names: Vec::new(),
            live_in: Vec::new(),
            partial: Vec::new(),
        };
        let (out, stats) = fwd(&empty);
        assert_eq!(out, empty);
        assert_eq!(stats.forwards, 0);
        assert_eq!(stats.rounds, 1);
        assert!(!stats.capped);
    }

    #[test]
    fn forwarding_preserves_every_statement_of_a_calling_function() {
        // Stores, a call, a callfx intrinsic and pure definitions in one
        // function: counts and kinds all survive.
        let input = build(&calling_function());
        let (out, stats) = fwd(&input);
        assert_eq!(check_preserved(&input, &out), Ok(()));
        assert!(!stats.capped);
        for (&va, b) in &input.blocks {
            assert_eq!(b.stmts.len(), out.blocks[&va].stmts.len());
        }
    }

    // -- 11: seeded sweep ---------------------------------------------------

    #[test]
    fn sweep_random_small_cfgs_always_forward_and_check() {
        let lo = live_out();
        for f in random_functions(400, 0x5EED_0DDB_1A5E_5BAD, true) {
            let input = build(&f);
            let (opt, ostats) = run(&input);
            assert!(!ostats.capped);
            let (forwarded, stats) = fwd(&opt);
            assert!(!stats.capped, "the round bound is not reached on real input");
            // Determinism: the same input forwards to the same bytes.
            let (again, stats2) = forward(&opt);
            assert_eq!(forwarded, again);
            assert_eq!(text(&forwarded), text(&again));
            assert_eq!(stats, stats2);
            // Idempotence: a second pass splices nothing.
            let (twice, s2) = fwd(&forwarded);
            assert_eq!(twice, forwarded);
            assert_eq!(s2.forwards, 0);
            // And the sweep behind it stays justified.
            let (swept, _) = sweep_with(&forwarded, &lo);
            assert_eq!(irssa::check(&swept), Ok(()));
        }
    }
}
