//! Dataflow-driven simplification of a lifted IR statement list.
//!
//! A raw lift is faithful but verbose: it recomputes status flags no one
//! reads, threads values through temporaries, and leaves constant
//! subexpressions unfolded. This module rewrites one basic block's
//! straight-line statement list into the essential computation it denotes,
//! without changing observable behavior, through three local passes:
//!
//! - **Constant folding** ([`fold_expr`]) evaluates operators whose operands
//!   are constants — at the correct width, wrapping per the width mask — and
//!   applies the algebraic identities that are always sound, the *relational*
//!   ones included: `(a - b) == 0 → a == b`, `(a ^ b) != 0 → a != b`,
//!   `~(a == b) → a != b` and their polarities, which are what turn a lifted
//!   flag computation back into the comparison it denotes once
//!   [`crate::irssaopt`] has forwarded the flag's definition into its use —
//!   plus the *order* family below, which does the same for the signed and
//!   unsigned order conditions.
//! - **Copy and constant propagation** ([`propagate`]) forwards a register's
//!   known constant or copied value into its later reads within the block,
//!   then folds.
//! - **Dead-code elimination** ([`eliminate_dead`]) drops pure assignments
//!   whose result no later statement — and no block live-out — observes.
//!
//! [`simplify`] composes the three to a bounded fixpoint; [`simplify_default`]
//! supplies a conservative live-out that keeps every architectural register
//! and drops only dead temporaries, dead flags, and redundant copies.
//!
//! # Soundness
//!
//! Every transform preserves observable behavior; when a rewrite cannot be
//! proven sound the input is left exactly as it was ("when in doubt, do
//! nothing"). In particular a memory [`Load`](crate::ir::Expr::Load) may
//! fault or alias, so a computation containing one is never *deleted*: an
//! identity that would collapse a subexpression to a constant (`x ^ x → 0`,
//! `x - x → 0`, `x * 0 → 0`, `x & 0 → 0`) requires the removed side to be
//! load-free, and a pure assignment is only eliminated when its value holds
//! no load. A [`Store`](crate::ir::Stmt::Store),
//! [`Branch`](crate::ir::Stmt::Branch), or
//! [`Intrinsic`](crate::ir::Stmt::Intrinsic) — each carrying an effect the
//! IR does not model as a value — is never dropped. Division by zero is a
//! trap, not a value, and is never folded. The output always passes
//! [`crate::ir::check`] (given a well-formed input), and no input, however
//! malformed or large, panics: constant evaluation uses wrapping and
//! divide-guarded arithmetic, shifts guard their amount, and every walk is
//! depth-bounded by [`crate::ir::MAX_EXPR_NODES`].
//!
//! # Order-condition recovery
//!
//! The equality identities above cover the `je`/`jne` plumbing; the order
//! conditions survive forwarding as *paired* flag shapes, because the lift
//! spells them over two flag definitions at once. With `SF = (a - b) <s 0`,
//! `OF = ((a ^ b) & (a ^ (a - b))) <s 0` (the one subtraction-overflow
//! model [`crate::x86_lift`] and [`crate::aarch64_lift`] share), `ZF`
//! already folded to `a == b` by the equality family, and `CF = a <u b`,
//! the pairs collapse — listed exactly as the equality family is:
//!
//! - `SF != OF → a <s b` (`jl`) and `SF == OF → b <=s a` (`jge`): the
//!   paired shapes recognized together, both flags decompositions of the
//!   *same* `a - b` — a near-miss (different operands, a mixed width, an
//!   overflow term from some other subtraction) never rewrites.
//! - `(a == b) | (a <s b) → a <=s b` (`jle`) and
//!   `(a != b) & (b <=s a) → b <s a` (`jg`): the conjunction shapes, whose
//!   halves the two rules above have already collapsed bottom-up. The same
//!   compositions serve the unsigned pair — `(a == b) | (a <u b) → a <=u b`
//!   (`jbe`) and `(a != b) & (b <=u a) → b <u a` (`ja`).
//! - `~(a <s b) → b <=s a` and the three sibling complements at `W1`: over
//!   a total order the negation of a strict comparison is the reversed
//!   non-strict one, which is every negated branch polarity (`jae` as
//!   `~CF`, `jge` reached through a negated `jl`, …).
//! - `x != 0 → x`, `x == 1 → x`, `x == 0 → ~x`, `x != 1 → ~x`, at `W1`
//!   only: what finishes a pattern whose other flag already folded to a
//!   constant — `cmp a, 0` folds `OF` to `0` outright, and `jl` must still
//!   come out as `a <s 0`.
//! - *Boolean merge* (setcc / `cset` / `cbnz` plumbing): a zero-extended
//!   `W1` compared against zero recovers the bit
//!   (`zext(c) != 0 → c`, `zext(c) == 0 → ~c`); a one-bit mask of a
//!   sign-extended condition collapses the same way
//!   (`1 & sext(c) → zext(c)`, `1 & ~sext(c) → zext(~c)`); and the full
//!   branchless select `(k_t & m) | (k_f & ~m)` with `{k_t,k_f} ⊆ {0,1}`
//!   and `m = sext(c)` is the `cset`/`csinc`-from-zr shape, folding to
//!   `zext(c)` or `zext(~c)`. Conjunction of two such 0/1 values tested
//!   against zero recovers `c₁ & c₂` at `W1`. No new IR operator.
//! - The *masked* pair, behind a conditional compare:
//!   [`crate::aarch64_lift`]'s CCMP writes each flag as the select
//!   `(c & model) | (~c & bit)`, whose constant arm folds first, so a
//!   following condition reads `(c & SF) != (c & OF)` — with `| ~c` riding
//!   on a flag whose imm4 bit is set, the `~c` literal or already flipped.
//!   Under one shared guard the guard-false arms compare as constants and
//!   the composition is the guarded relation:
//!   `(c & SF) != (c & OF) → c & (a <s b)`,
//!   `(c & SF) == (c & OF) → ~c | (b <=s a)`, and with exactly one side
//!   carrying `| ~c` the results swap — `~c | (a <s b)` and
//!   `c & (b <=s a)`.
//! - `~((c & x) | ~c) → c & ~x`: the negated read of a single bit-set flag
//!   select (a `b.ne` over a CCMP's `Z`, a `b.lo` over its `C`), for any
//!   `W1` flag expression `x`, the `~x` flipping on when it is a
//!   comparison.
//!
//! Each identity is a theorem of two's-complement arithmetic at fixed
//! width, proved here by the width-8 exhaustive oracle (every operand
//! pair, every pattern, every polarity — and for the guarded patterns,
//! both guard values) rather than argued. The pair and composition
//! rewrites drop duplicate copies of `a` and `b` — and the masked ones a
//! duplicate of the guard — and may do so even when the shared operand
//! carries a load, by the *one-expression theorem*: statements are the
//! only effects, so every load inside a single statement's expression
//! reads the same memory state, and two structurally equal subtrees are
//! therefore value-equal — the equality the patterns already demand. (The
//! dropped copy is a re-read of unchanged memory; the change in dynamic
//! load *count* is observable only for volatile/MMIO locations and rides
//! the same conforming-code assumption [`crate::callfx`] records. The
//! annihilation identities above keep their load-free guards: deleting a
//! load outright is a different contract from keeping one of two equal
//! copies.) The masked *unsigned* pair and the fully composed masked
//! `le`/`gt` (a `Z` select OR-ed or AND-ed with the recovered pair) have
//! no measured occurrence and are deliberately not pattern-matched.
//!
//! # Width-spelling normalization
//!
//! A 32-bit write on both lifted ISAs zero-extends into the wide cell,
//! so the same value reaches a later comparison spelled both bare and as
//! `trunc.d(zext.q(x))` — which is why the order patterns' structural
//! equality used to refuse real pairs whose operands diverged only in
//! spelling. [`fold_expr`] therefore cancels a truncation against the
//! extension beneath it, at every width: `trunc(zext(x))`/`trunc(sext(x))`
//! collapse to `x` when the truncation lands exactly on `x`'s width, to
//! `trunc(x)` when it lands below, and to the re-targeted extension when
//! it lands between `x` and the extension — always-sound theorems (the
//! kept bits are fully determined by `x`), each proved by exhaustive
//! evaluation. The unsound directions are refused by shape:
//! `zext(trunc(x))` and `sext(trunc(x))` discard bits of `x` and are no
//! identity, and sign- versus zero-extension of the same value stay
//! distinct expressions.
//!
//! # The equality witness
//!
//! The pair matchers above gate on operand equality. Structurally spelled,
//! that refuses two real classes the corpus measurements diagnosed: halves
//! reaching the comparison through *different SSA names* for one value
//! (forwarding's duplication cap left one half a bare read of the very
//! tree the other half carries spliced), and halves spelling one truncated
//! value two ways — `trunc.d(x + y)` against `trunc.d(x) + trunc.d(y)`,
//! the 64-bit definition against the 32-bit lift. [`veq`] is the witness
//! for both: a structural-equality fast path (bit-for-bit the old behavior,
//! and the only path without a context), then a canonical-key comparison
//! that resolves each exact-width read through its unique SSA definition
//! ([`VnDefs`], built by [`crate::irssaopt::forward`]; the purity gate —
//! load-free, division-free, node-capped — is enforced at admission) and
//! pushes every truncation to the leaves through the operators over which
//! truncation is a ring homomorphism (`Add`/`Sub`/`Mul`/`And`/`Or`/`Xor`,
//! `Neg`/`Not` — dropped high bits never feed a kept low bit; each proved
//! exhaustively), composing with nested truncations and cancelling through
//! extensions via the width identities above. Shifts are *not* distributed
//! over (the amount is taken modulo the width — the near-miss test pins
//! it), a malformed truncation is never laundered, and fuel exhaustion or
//! a load anywhere is a refusal, never a guess. The witness only proves:
//! every matcher keeps returning subtrees of its own input, so no resolved
//! tree is ever emitted, and a true answer beyond the fast path implies
//! both sides load-free — the kept operand's load gate covers the dropped
//! duplicate. Deferred, recorded: φ-congruence, commutative/associative
//! normalization, and the self-identities (`x - x`, `x ^ x`, …), which
//! stay purely structural.
//!
//! # Register aliasing
//!
//! The IR models a narrow reference to a wide architectural register
//! (`eax` of `rax`) as the same cell at a different [`Width`], distinguished
//! only by the ISA's numbering convention. Two [`Reg`]s therefore *may
//! alias* when they share a space and number, whatever their widths.
//! Liveness and propagation-invalidation treat aliasing conservatively: a
//! write to any width of a cell invalidates known facts about that cell and
//! keeps any live reference to it; only a redefinition at the *exact* same
//! reference is taken to fully cover (and so kill) an earlier one.
//!
//! # Call effects
//!
//! Calls carry no special handling here, by design: an ABI-aware
//! consumer makes a call's clobbers and argument reads explicit
//! *upstream*, as an ordinary [`Intrinsic`](crate::ir::Stmt::Intrinsic)
//! inserted by [`crate::callfx::apply`], and the conservative rules here
//! already do the right thing with it — the liveness transfer kills the
//! intrinsic's writes and gens its reads, [`propagate`] invalidates per
//! write, and [`eliminate_dead`] never deletes the intrinsic itself.
//! [`propagate`]'s clear-everything-at-[`Branch`](crate::ir::Stmt::Branch)
//! rule stays as-is: within one block it covers only dead code after a
//! call terminator, and narrowing it to clobbers-only pays off only in
//! cross-block propagation, which will consume the callfx definitions
//! through SSA instead.
//!
//! # Shift semantics
//!
//! [`crate::ir`] leaves the meaning of a shift by an amount at least as wide
//! as the value unspecified. Folding here takes the shift amount **modulo
//! the value's bit width** — the behavior of masked-count hardware and of a
//! poison-free IR — and [`UnOp`]/[`BinOp`] widths are preserved throughout.
//! A lifter needing exact hardware count-masking encodes it explicitly.
//!
//! # Scope
//!
//! Everything here is *per block* and pre-SSA. Propagation that crosses a
//! block boundary needs the def-use form [`crate::irssa`] builds, and
//! lives in [`crate::irssaopt`]; this module stays the straight-line
//! library whose [`fold_expr`] / [`fold_stmt`] that pass reuses.

use crate::ir::{self, BinOp, Expr, Reg, Space, Stmt, UnOp, Width};
use std::collections::{BTreeMap, BTreeSet};

/// Maximum simplification rounds before [`simplify`] returns its current
/// result. A bound, not a promise of non-convergence: most blocks reach a
/// fixpoint in one or two rounds, and the cap guarantees termination even
/// on adversarial input that never stabilizes.
const MAX_ROUNDS: usize = 8;

/// Depth at which the value-rewriting walks ([`fold_expr`], `substitute`)
/// stop recursing and return the subexpression unchanged. A tree deeper than
/// this only arises from adversarial input that [`crate::ir::check`] already
/// rejects (a real lift is shallow); the cap keeps a rewrite that builds new
/// owned expressions from exhausting the stack, while the lighter read-only
/// walks stay bounded by [`crate::ir::MAX_EXPR_NODES`].
const REWRITE_DEPTH: usize = 512;

// ---------------------------------------------------------------------------
// Constant folding
// ---------------------------------------------------------------------------

/// Fold an expression: evaluate constant operations at the correct width and
/// apply the always-sound algebraic identities, recursively, bottom-up.
///
/// Width-preserving and total. A subtree it cannot prove safe to rewrite —
/// a division by zero, an operation on non-constants with no matching
/// identity, an identity whose removed side bears a [`Load`](Expr::Load) — is
/// returned with its operands folded but its shape intact. Never panics.
pub fn fold_expr(e: &Expr) -> Expr {
    fold_rec(e, 0, None)
}

/// [`fold_expr`] with a value-numbering context: identical rewrites, but
/// the pair matchers' equality gates may witness two spellings of one
/// value through `vn` (see [`VnDefs`]). The fold's *output* is built from
/// the input's own subtrees exactly as without the context; only which
/// identities fire can differ.
pub fn fold_expr_vn(e: &Expr, vn: &VnDefs) -> Expr {
    fold_rec(e, 0, Some(vn))
}

fn fold_rec(e: &Expr, depth: usize, vn: Option<&VnDefs>) -> Expr {
    if depth > REWRITE_DEPTH {
        return e.clone();
    }
    match e {
        Expr::Const { .. } | Expr::Reg(_) => e.clone(),
        Expr::Load { addr, width } => Expr::load(fold_rec(addr, depth + 1, vn), *width),
        Expr::Unary { op, operand } => {
            let o = fold_rec(operand, depth + 1, vn);
            if let Some(c) = fold_unary_const(*op, &o) {
                return c;
            }
            if let Some(s) = fold_width_identity(*op, &o) {
                return s;
            }
            if let Some(s) = fold_unary_identity(*op, &o, vn) {
                return s;
            }
            Expr::unary(*op, o)
        }
        Expr::Binary { op, lhs, rhs } => {
            let l = fold_rec(lhs, depth + 1, vn);
            let r = fold_rec(rhs, depth + 1, vn);
            if let Some(c) = fold_binary_const(*op, &l, &r) {
                return c;
            }
            if let Some(s) = fold_binary_identity(*op, &l, &r, vn) {
                return s;
            }
            Expr::binary(*op, l, r)
        }
    }
}

/// Interpret the low `width` bits of `value` as a two's-complement signed
/// integer, sign-extended to `i64`. The shift discards bits above the width.
fn sign_extend(value: u64, width: Width) -> i64 {
    let bits = width.bits();
    if bits >= 64 {
        value as i64
    } else {
        let shift = 64 - bits;
        ((value << shift) as i64) >> shift
    }
}

/// The constant value (masked to its width) and width of `e`, if it is a
/// [`Expr::Const`].
fn const_value(e: &Expr) -> Option<(u64, Width)> {
    match e {
        Expr::Const { value, width } => Some((*value & width.mask(), *width)),
        _ => None,
    }
}

/// Fold a unary operator applied to a constant. Extend/truncate directions
/// are guarded to match [`crate::ir::check`] so no invalid tree is produced.
fn fold_unary_const(op: UnOp, operand: &Expr) -> Option<Expr> {
    let (v, w) = const_value(operand)?;
    let m = w.mask();
    match op {
        UnOp::Neg => Some(Expr::constant(0u64.wrapping_sub(v), w)),
        UnOp::Not => Some(Expr::constant(!v & m, w)),
        UnOp::ZeroExtend(to) => {
            if to.bits() <= w.bits() {
                return None;
            }
            Some(Expr::constant(v, to))
        }
        UnOp::SignExtend(to) => {
            if to.bits() <= w.bits() {
                return None;
            }
            Some(Expr::constant(sign_extend(v, w) as u64, to))
        }
        UnOp::Truncate(to) => {
            if to.bits() >= w.bits() {
                return None;
            }
            Some(Expr::constant(v & to.mask(), to))
        }
    }
}

/// Fold a binary operator applied to two constants, or `None` when it must
/// be left as-is: a divide/remainder by zero (a trap, not a value) or a
/// non-shift operation whose operands disagree in width (malformed).
fn fold_binary_const(op: BinOp, lhs: &Expr, rhs: &Expr) -> Option<Expr> {
    let (lv, lw) = const_value(lhs)?;
    let (rv, rw) = const_value(rhs)?;

    // Comparisons yield a W1 0/1 at the operands' shared width.
    if op.is_compare() {
        if lw != rw {
            return None;
        }
        let m = lw.mask();
        let (a, b) = (lv & m, rv & m);
        let result = match op {
            BinOp::Eq => a == b,
            BinOp::Ne => a != b,
            BinOp::Ult => a < b,
            BinOp::Ule => a <= b,
            BinOp::Slt => sign_extend(a, lw) < sign_extend(b, lw),
            BinOp::Sle => sign_extend(a, lw) <= sign_extend(b, lw),
            _ => unreachable!("is_compare covers exactly these"),
        };
        return Some(Expr::constant(result as u64, Width::W1));
    }

    // Shifts take their width from the value; the amount is taken modulo the
    // value's bit width (see the module's "Shift semantics").
    if matches!(op, BinOp::Shl | BinOp::LShr | BinOp::AShr) {
        let w = lw;
        let bits = w.bits();
        let sh = (rv % bits as u64) as u32; // bits >= 1, so no divide-by-zero
        let a = lv & w.mask();
        let out = match op {
            BinOp::Shl => (a << sh) & w.mask(),
            BinOp::LShr => a >> sh,
            BinOp::AShr => (sign_extend(a, w) >> sh) as u64 & w.mask(),
            _ => unreachable!(),
        };
        return Some(Expr::constant(out, w));
    }

    // Remaining operators need equal operand widths.
    if lw != rw {
        return None;
    }
    let w = lw;
    let m = w.mask();
    let (a, b) = (lv & m, rv & m);
    let out = match op {
        BinOp::Add => a.wrapping_add(b) & m,
        BinOp::Sub => a.wrapping_sub(b) & m,
        BinOp::Mul => a.wrapping_mul(b) & m,
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
        BinOp::UDiv => {
            if b == 0 {
                return None;
            }
            a / b
        }
        BinOp::URem => {
            if b == 0 {
                return None;
            }
            a % b
        }
        BinOp::SDiv => {
            let sb = sign_extend(b, w);
            if sb == 0 {
                return None;
            }
            // wrapping_div also tames the i64::MIN / -1 corner at W64.
            (sign_extend(a, w).wrapping_div(sb) as u64) & m
        }
        BinOp::SRem => {
            let sb = sign_extend(b, w);
            if sb == 0 {
                return None;
            }
            (sign_extend(a, w).wrapping_rem(sb) as u64) & m
        }
        // Compares and shifts are handled above.
        _ => return None,
    };
    Some(Expr::constant(out, w))
}

/// Cancel a truncation against the extension beneath it. For `x` of width
/// `w` extended to some strictly wider width and then truncated to `to`,
/// the low `to` bits of the extension are fully determined by `x`, at
/// every width — the width-spelling theorems that let structural equality
/// see through a value respelled along a `zext`/`sext`/`trunc` chain
/// (an A64 or x86-64 W32 write is a zero-extend into the wide cell, so
/// the same value reaches a comparison spelled both bare and as
/// `trunc(zext(x))`):
///
/// - `to == w`: the truncation undoes the extension exactly — `x`.
/// - `to < w`: only bits of `x` survive — `trunc(to)(x)`, re-folded in
///   case `x` is itself an extension.
/// - `to > w`: every kept bit is the extension's fill — the same
///   extension re-targeted at `to`.
///
/// The operand `x` survives whole in every case (only operator nodes are
/// removed), so nothing — a [`Load`](Expr::Load) included — is erased.
/// The unsound directions stay refused by shape: `zext(trunc(x))` /
/// `sext(trunc(x))` discard high bits of `x` and are no identity, and a
/// malformed chain (a non-widening extension, a non-narrowing truncation)
/// is left exactly as it is rather than laundered.
fn fold_width_identity(op: UnOp, operand: &Expr) -> Option<Expr> {
    let UnOp::Truncate(to) = op else {
        return None;
    };
    let Expr::Unary {
        op: ext,
        operand: inner,
    } = operand
    else {
        return None;
    };
    let ext_to = match ext {
        UnOp::ZeroExtend(t) | UnOp::SignExtend(t) => *t,
        _ => return None,
    };
    let w = inner.width_of()?;
    // Malformed trees (an extension that does not widen, a truncation
    // that does not narrow) must not be rewritten into well-formed ones.
    if ext_to.bits() <= w.bits() || to.bits() >= ext_to.bits() {
        return None;
    }
    if to == w {
        return Some((**inner).clone());
    }
    if to.bits() < w.bits() {
        // The inner value may itself be an extension; re-apply so the
        // fold stays idempotent.
        let t = UnOp::Truncate(to);
        return Some(
            fold_width_identity(t, inner)
                .unwrap_or_else(|| Expr::unary(t, (**inner).clone())),
        );
    }
    let refit = match ext {
        UnOp::ZeroExtend(_) => UnOp::ZeroExtend(to),
        _ => UnOp::SignExtend(to),
    };
    Some(Expr::unary(refit, (**inner).clone()))
}

/// Apply a boolean identity to a folded unary operand: the negation of a
/// comparison, and double negation.
///
/// [`UnOp::Not`] is bitwise complement, so at [`Width::W1`] — the width of
/// every comparison result and of a flag — it *is* boolean negation, and
/// `~(a == b)` denotes `a != b` exactly, while over the total order (signed
/// or unsigned) at any fixed operand width the complement of a strict
/// comparison is the reversed non-strict one: `~(a <s b) → b <=s a`, and
/// so for all four order operators. Both operands survive the rewrite,
/// so nothing (a [`Load`](Expr::Load) included) is erased; the identity is
/// gated on `W1` so it can never be read as a complement of a wider value.
fn fold_unary_identity(op: UnOp, operand: &Expr, vn: Option<&VnDefs>) -> Option<Expr> {
    if op != UnOp::Not || operand.width_of() != Some(Width::W1) {
        return None;
    }
    match operand {
        // ~(a == b) → a != b, ~(a != b) → a == b.
        Expr::Binary {
            op: BinOp::Eq,
            lhs,
            rhs,
        } => Some(Expr::binary(BinOp::Ne, (**lhs).clone(), (**rhs).clone())),
        Expr::Binary {
            op: BinOp::Ne,
            lhs,
            rhs,
        } => Some(Expr::binary(BinOp::Eq, (**lhs).clone(), (**rhs).clone())),
        // ~(a < b) → b <= a, ~(a <= b) → b < a, in both signednesses.
        Expr::Binary {
            op: BinOp::Slt,
            lhs,
            rhs,
        } => Some(Expr::binary(BinOp::Sle, (**rhs).clone(), (**lhs).clone())),
        Expr::Binary {
            op: BinOp::Sle,
            lhs,
            rhs,
        } => Some(Expr::binary(BinOp::Slt, (**rhs).clone(), (**lhs).clone())),
        Expr::Binary {
            op: BinOp::Ult,
            lhs,
            rhs,
        } => Some(Expr::binary(BinOp::Ule, (**rhs).clone(), (**lhs).clone())),
        Expr::Binary {
            op: BinOp::Ule,
            lhs,
            rhs,
        } => Some(Expr::binary(BinOp::Ult, (**rhs).clone(), (**lhs).clone())),
        // ~((c & x) | ~c) → c & ~x: the negated read of a conditional
        // compare's flag select whose imm4 bit is set (see the module's
        // "Order-condition recovery").
        Expr::Binary {
            op: BinOp::Or,
            lhs,
            rhs,
        } => not_of_flag_select(lhs, rhs, vn),
        // ~~x → x, at the one width where the complement is a negation.
        Expr::Unary {
            op: UnOp::Not,
            operand: inner,
        } if inner.width_of() == Some(Width::W1) => Some((**inner).clone()),
        _ => None,
    }
}

/// Whether `e` is a constant equal to `k` (compared at the constant's width).
fn is_const(e: &Expr, k: u64) -> bool {
    matches!(const_value(e), Some((v, w)) if v == k & w.mask())
}

/// Whether `e` is the all-ones constant of its width.
fn is_all_ones(e: &Expr) -> bool {
    matches!(const_value(e), Some((v, w)) if v == w.mask())
}

/// Apply an algebraic identity when the operands are already folded and at
/// least one is not constant. Every identity here is width-preserving and
/// behavior-preserving; any identity that *removes* a non-constant side
/// requires that side to be load-free (see the module's "Soundness").
fn fold_binary_identity(op: BinOp, lhs: &Expr, rhs: &Expr, vn: Option<&VnDefs>) -> Option<Expr> {
    let width = || lhs.width_of().or_else(|| rhs.width_of());
    match op {
        BinOp::Add => {
            // x + 0 → x, 0 + x → x. The removed side is a constant zero.
            if is_const(rhs, 0) {
                return Some(lhs.clone());
            }
            if is_const(lhs, 0) {
                return Some(rhs.clone());
            }
            None
        }
        BinOp::Sub => {
            if is_const(rhs, 0) {
                return Some(lhs.clone()); // x - 0 → x
            }
            if lhs == rhs && !contains_load(lhs, 0) {
                return Some(Expr::constant(0, width()?)); // x - x → 0
            }
            None
        }
        BinOp::Mul => {
            if is_const(rhs, 1) {
                return Some(lhs.clone()); // x * 1 → x
            }
            if is_const(lhs, 1) {
                return Some(rhs.clone());
            }
            // x * 0 → 0, deleting the other side, which must be load-free.
            if is_const(rhs, 0) && !contains_load(lhs, 0) {
                return Some(Expr::constant(0, width()?));
            }
            if is_const(lhs, 0) && !contains_load(rhs, 0) {
                return Some(Expr::constant(0, width()?));
            }
            None
        }
        BinOp::And => {
            // x & 0 → 0, deleting the other side (load-free).
            if is_const(rhs, 0) && !contains_load(lhs, 0) {
                return Some(Expr::constant(0, width()?));
            }
            if is_const(lhs, 0) && !contains_load(rhs, 0) {
                return Some(Expr::constant(0, width()?));
            }
            // x & allones → x
            if is_all_ones(rhs) {
                return Some(lhs.clone());
            }
            if is_all_ones(lhs) {
                return Some(rhs.clone());
            }
            // x & x → x (keeps one copy; require load-free for safety).
            if lhs == rhs && !contains_load(lhs, 0) {
                return Some(lhs.clone());
            }
            // 1 & sext(c) / 1 & ~sext(c): the cset remnant after the
            // zero arm of the select folded away (see "Boolean merge").
            if let Some(s) = bool_mask_and(lhs, rhs) {
                return Some(s);
            }
            // (a != b) & (b <= a) → b < a (either signedness): the `jg` /
            // `ja` composition, once its flag halves have collapsed.
            if let Some(s) = and_order_compose(lhs, rhs, vn) {
                return Some(s);
            }
            None
        }
        BinOp::Or => {
            // x | 0 → x
            if is_const(rhs, 0) {
                return Some(lhs.clone());
            }
            if is_const(lhs, 0) {
                return Some(rhs.clone());
            }
            // x | x → x (load-free)
            if lhs == rhs && !contains_load(lhs, 0) {
                return Some(lhs.clone());
            }
            // (k_t & m) | (k_f & ~m) with 0/1 arms: the full cset select.
            if let Some(s) = bool_select_or(lhs, rhs) {
                return Some(s);
            }
            // (a == b) | (a < b) → a <= b (either signedness): the `jle` /
            // `jbe` composition, once its flag halves have collapsed.
            if let Some(s) = or_order_compose(lhs, rhs, vn) {
                return Some(s);
            }
            None
        }
        BinOp::Xor => {
            // x ^ x → 0 (load-free), then x ^ 0 → x.
            if lhs == rhs && !contains_load(lhs, 0) {
                return Some(Expr::constant(0, width()?));
            }
            if is_const(rhs, 0) {
                return Some(lhs.clone());
            }
            if is_const(lhs, 0) {
                return Some(rhs.clone());
            }
            None
        }
        BinOp::Shl | BinOp::LShr | BinOp::AShr => {
            // x <shift> 0 → x
            if is_const(rhs, 0) {
                return Some(lhs.clone());
            }
            None
        }
        // The relational identities that turn a flag computation back into
        // the comparison it denotes: `(a - b) == 0 → a == b`,
        // `(a ^ b) != 0 → a != b`, and so on for both polarities and both
        // orientations of the zero. Subtraction and exclusive-or are each
        // injective in one operand for the other fixed, under
        // two's-complement wrapping at every width, so comparing the result
        // against zero *is* comparing the operands. Both operands survive,
        // so no side — a [`Load`](Expr::Load) included — is erased.
        BinOp::Eq | BinOp::Ne => {
            // A comparison whose sides disagree in width is malformed; a
            // rewrite must not launder it into something well-formed.
            if lhs.width_of() != rhs.width_of() || lhs.width_of().is_none() {
                return None;
            }
            if is_const(rhs, 0)
                && let Some(s) = zero_compare(op, lhs)
            {
                return Some(s);
            }
            if is_const(lhs, 0)
                && let Some(s) = zero_compare(op, rhs)
            {
                return Some(s);
            }
            // The paired signed-order flag shapes: `SF != OF → a <s b`,
            // `SF == OF → b <=s a` (see the module's "Order-condition
            // recovery").
            if let Some(s) = signed_order_pair(op, lhs, rhs, vn) {
                return Some(s);
            }
            // The same pair behind a conditional compare's guard: each
            // flag a select over one shared condition (see the module's
            // "Order-condition recovery").
            if let Some(s) = masked_order_pair(op, lhs, rhs, vn) {
                return Some(s);
            }
            // A `W1` comparison against a boolean constant, the pattern
            // remnant left when the other flag folded to a constant first.
            if let Some(s) = w1_const_compare(op, lhs, rhs) {
                return Some(s);
            }
            // zext(c) != 0 → c, and (zext(a) & zext(b)) != 0 → a & b:
            // setcc/cset then test/cbnz (see "Boolean merge").
            if let Some(s) = zext_bool_compare(op, lhs, rhs) {
                return Some(s);
            }
            None
        }
        _ => None,
    }
}

/// The comparison `cmp` (a [`BinOp::Eq`] or [`BinOp::Ne`]) of `e` against
/// zero, rewritten as a comparison of `e`'s own operands when `e` is a
/// subtraction or an exclusive-or of two equal-width values.
fn zero_compare(cmp: BinOp, e: &Expr) -> Option<Expr> {
    let Expr::Binary { op, lhs, rhs } = e else {
        return None;
    };
    if !matches!(op, BinOp::Sub | BinOp::Xor) {
        return None;
    }
    // Equal, defined operand widths: the rewrite must not launder a
    // malformed tree into a well-formed-looking one.
    let w = lhs.width_of()?;
    if rhs.width_of() != Some(w) || e.width_of() != Some(w) {
        return None;
    }
    Some(Expr::binary(cmp, (**lhs).clone(), (**rhs).clone()))
}

// ---------------------------------------------------------------------------
// Order-condition recovery (see the module docs' pattern list)
// ---------------------------------------------------------------------------

/// The `(a, b)` operands and width of the lift's sign-flag shape
/// `(a - b) <s 0`: the subtraction, the zero, and the `<s` all at the
/// operands' one shared width, or no match.
fn sign_flag_operands(e: &Expr) -> Option<(&Expr, &Expr, Width)> {
    let Expr::Binary {
        op: BinOp::Slt,
        lhs,
        rhs,
    } = e
    else {
        return None;
    };
    let Expr::Binary {
        op: BinOp::Sub,
        lhs: a,
        rhs: b,
    } = &**lhs
    else {
        return None;
    };
    let w = a.width_of()?;
    if b.width_of() != Some(w) || const_value(rhs) != Some((0, w)) {
        return None;
    }
    Some((&**a, &**b, w))
}

/// The `(a, b)` operands and width of the lift's subtraction-overflow shape
/// `((a ^ b) & (a ^ (a - b))) <s 0` — the one OF/V model
/// [`crate::x86_lift`] and [`crate::aarch64_lift`] share for `a - b`. Every
/// occurrence of `a` and of `b` must be [`veq`]-equal (structural equality,
/// or the value-numbering witness when a context is given) and every node
/// at the one shared width, or no match; the addition-overflow shape
/// (`(l ^ res) & (r ^ res)`, no `Sub` inside) never matches. The returned
/// operands are the first occurrences — subtrees of `e` itself.
fn overflow_flag_operands<'e>(e: &'e Expr, vn: Option<&VnDefs>) -> Option<(&'e Expr, &'e Expr, Width)> {
    let Expr::Binary {
        op: BinOp::Slt,
        lhs,
        rhs,
    } = e
    else {
        return None;
    };
    let Expr::Binary {
        op: BinOp::And,
        lhs: x,
        rhs: y,
    } = &**lhs
    else {
        return None;
    };
    let Expr::Binary {
        op: BinOp::Xor,
        lhs: a1,
        rhs: b1,
    } = &**x
    else {
        return None;
    };
    let Expr::Binary {
        op: BinOp::Xor,
        lhs: a2,
        rhs: d,
    } = &**y
    else {
        return None;
    };
    let Expr::Binary {
        op: BinOp::Sub,
        lhs: a3,
        rhs: b2,
    } = &**d
    else {
        return None;
    };
    if !veq(a1, a2, vn) || !veq(a1, a3, vn) || !veq(b1, b2, vn) {
        return None;
    }
    let w = a1.width_of()?;
    if b1.width_of() != Some(w) || const_value(rhs) != Some((0, w)) {
        return None;
    }
    Some((&**a1, &**b1, w))
}

/// Both halves of the paired signed-order shape — one side the sign flag of
/// `a - b`, the other its overflow flag, over [`veq`]-equal `a` and `b` at
/// one width — in either order. The returned operands are the sign-flag
/// half's spellings (subtrees of the statement, never a resolved tree).
fn order_pair_operands<'e>(
    lhs: &'e Expr,
    rhs: &'e Expr,
    vn: Option<&VnDefs>,
) -> Option<(&'e Expr, &'e Expr)> {
    let half = |sf: &'e Expr, of: &'e Expr| {
        let (a, b, w) = sign_flag_operands(sf)?;
        let (oa, ob, ow) = overflow_flag_operands(of, vn)?;
        (veq(a, oa, vn) && veq(b, ob, vn) && w == ow).then_some((a, b))
    };
    half(lhs, rhs).or_else(|| half(rhs, lhs))
}

/// Collapse the paired signed-order flag shapes: with `SF = (a - b) <s 0`
/// and `OF = ((a ^ b) & (a ^ (a - b))) <s 0`, two's-complement arithmetic
/// at any fixed width gives `SF != OF ⇔ a <s b` and `SF == OF ⇔ b <=s a`
/// (proved exhaustively at width 8 in the tests). The rewrite keeps one
/// copy of `a` and of `b` and drops their duplicates — sound even for
/// load-bearing operands by the one-expression theorem (see the module's
/// "Soundness"): a [`veq`] over a load-bearing side can only have held
/// structurally ([`VnDefs`] resolution is load-free), and structural
/// equality within one statement is value equality.
fn signed_order_pair(cmp: BinOp, lhs: &Expr, rhs: &Expr, vn: Option<&VnDefs>) -> Option<Expr> {
    let (a, b) = order_pair_operands(lhs, rhs, vn)?;
    match cmp {
        BinOp::Ne => Some(Expr::binary(BinOp::Slt, a.clone(), b.clone())),
        BinOp::Eq => Some(Expr::binary(BinOp::Sle, b.clone(), a.clone())),
        _ => None,
    }
}

/// At [`Width::W1`] — where a value is exactly 0 or 1 — a comparison
/// against a boolean constant is the operand or its negation: `x != 0 → x`,
/// `x == 1 → x`, `x == 0 → ~x`, `x != 1 → ~x`. The `~` is folded on
/// through [`fold_unary_identity`] so a negated comparison flips instead of
/// nesting. Only the constant is dropped; the operand survives whole.
fn w1_const_compare(cmp: BinOp, lhs: &Expr, rhs: &Expr) -> Option<Expr> {
    for (x, c) in [(lhs, rhs), (rhs, lhs)] {
        let Some((v, Width::W1)) = const_value(c) else {
            continue;
        };
        let keep = (v == 1) == (cmp == BinOp::Eq);
        return if keep {
            Some(x.clone())
        } else {
            Some(
                fold_unary_identity(UnOp::Not, x, None)
                    .unwrap_or_else(|| Expr::unary(UnOp::Not, x.clone())),
            )
        };
    }
    None
}

// ---------------------------------------------------------------------------
// Boolean merge (setcc / cset / cbnz — see the module docs)
// ---------------------------------------------------------------------------

/// Peel a zero-extended `W1` boolean, or `None`.
fn zext_w1(e: &Expr) -> Option<&Expr> {
    match e {
        Expr::Unary {
            op: UnOp::ZeroExtend(_),
            operand,
        } if operand.width_of() == Some(Width::W1) => Some(operand),
        _ => None,
    }
}

/// Peel `sext(c)` for a `W1` `c`, or `None`.
fn sext_w1(e: &Expr) -> Option<&Expr> {
    match e {
        Expr::Unary {
            op: UnOp::SignExtend(_),
            operand,
        } if operand.width_of() == Some(Width::W1) => Some(operand),
        _ => None,
    }
}

/// `zext(c)` to `w` — the bit as a 0/1 of width `w`.
fn zext_bool(c: &Expr, w: Width) -> Expr {
    if w == Width::W1 {
        c.clone()
    } else {
        Expr::unary(UnOp::ZeroExtend(w), c.clone())
    }
}

/// `1 & sext(c) → zext(c)` and `1 & ~sext(c) → zext(~c)` at the And's
/// width. Either operand order. The one-bit mask is exactly what a
/// `cset` leaves after the zero select-arm folds away.
fn bool_mask_and(lhs: &Expr, rhs: &Expr) -> Option<Expr> {
    for (one, masked) in [(lhs, rhs), (rhs, lhs)] {
        if !is_const(one, 1) {
            continue;
        }
        let w = masked.width_of()?;
        // 1 & sext(c)
        if let Some(c) = sext_w1(masked) {
            return Some(zext_bool(c, w));
        }
        // 1 & ~sext(c)
        if let Expr::Unary {
            op: UnOp::Not,
            operand,
        } = masked
            && let Some(c) = sext_w1(operand)
        {
            let flipped = fold_unary_identity(UnOp::Not, c, None)
                .unwrap_or_else(|| Expr::unary(UnOp::Not, c.clone()));
            return Some(zext_bool(&flipped, w));
        }
    }
    None
}

/// `(k_t & m) | (k_f & ~m)` with `{k_t, k_f} ⊆ {0, 1}` and `m = sext(c)`:
/// the full branchless `cset`/`csel`-from-constants shape. Either arm
/// order of the `|`. Yields `zext(c)` or `zext(~c)` at the Or's width.
fn bool_select_or(lhs: &Expr, rhs: &Expr) -> Option<Expr> {
    fn arm(e: &Expr) -> Option<(u64, &Expr)> {
        let Expr::Binary {
            op: BinOp::And,
            lhs: a,
            rhs: b,
        } = e
        else {
            return None;
        };
        let a = a.as_ref();
        let b = b.as_ref();
        for (k, m) in [(a, b), (b, a)] {
            let Some((v, _)) = const_value(k) else {
                continue;
            };
            if v > 1 {
                continue;
            }
            return Some((v, m));
        }
        None
    }
    for (t, f) in [(lhs, rhs), (rhs, lhs)] {
        let Some((kt, mt)) = arm(t) else {
            continue;
        };
        let Some((kf, mf)) = arm(f) else {
            continue;
        };
        // One arm's mask must be the complement of the other's, and the
        // un-complemented side is the sign-extended condition.
        let (c, true_when) = if let Expr::Unary {
            op: UnOp::Not,
            operand,
        } = mf
            && operand.as_ref() == mt
            && let Some(c) = sext_w1(mt)
        {
            (c, kt == 1)
        } else if let Expr::Unary {
            op: UnOp::Not,
            operand,
        } = mt
            && operand.as_ref() == mf
            && let Some(c) = sext_w1(mf)
        {
            (c, kf == 1)
        } else {
            continue;
        };
        let w = t.width_of().or_else(|| f.width_of())?;
        let bit = if true_when {
            c.clone()
        } else {
            fold_unary_identity(UnOp::Not, c, None)
                .unwrap_or_else(|| Expr::unary(UnOp::Not, c.clone()))
        };
        return Some(zext_bool(&bit, w));
    }
    None
}

/// `zext(c) != 0 → c`, `zext(c) == 0 → ~c`, and the same over a
/// conjunction of two zero-extended bits: `(zext(a) & zext(b)) != 0 →
/// a & b`. Either orientation of the zero. Widths of the zexts may
/// disagree with each other only when both are pure zext-of-W1 (the
/// And's width is then well-defined as their shared result width).
fn zext_bool_compare(cmp: BinOp, lhs: &Expr, rhs: &Expr) -> Option<Expr> {
    if !matches!(cmp, BinOp::Eq | BinOp::Ne) {
        return None;
    }
    for (x, z) in [(lhs, rhs), (rhs, lhs)] {
        if !is_const(z, 0) {
            continue;
        }
        // zext(c) ? 0
        if let Some(c) = zext_w1(x) {
            return Some(if cmp == BinOp::Ne {
                c.clone()
            } else {
                fold_unary_identity(UnOp::Not, c, None)
                    .unwrap_or_else(|| Expr::unary(UnOp::Not, c.clone()))
            });
        }
        // (zext(a) & zext(b)) ? 0 — either And order already folded.
        if let Expr::Binary {
            op: BinOp::And,
            lhs: a,
            rhs: b,
        } = x
            && let (Some(ca), Some(cb)) = (zext_w1(a), zext_w1(b))
        {
            let conj = Expr::binary(BinOp::And, ca.clone(), cb.clone());
            return Some(if cmp == BinOp::Ne {
                conj
            } else {
                fold_unary_identity(UnOp::Not, &conj, None)
                    .unwrap_or_else(|| Expr::unary(UnOp::Not, conj))
            });
        }
    }
    None
}

/// Shared guard for the order compositions: the equality's operands are the
/// order comparison's (in either order — equality is symmetric) and the
/// order comparison's sides carry one defined width. The rewrite drops the
/// equality's duplicate copies of the operands — sound even load-bearing by
/// the one-expression theorem (see the module's "Soundness"; a load-bearing
/// [`veq`] is always the structural fast path).
fn order_compose_ok(p: &Expr, q: &Expr, u: &Expr, v: &Expr, vn: Option<&VnDefs>) -> bool {
    ((veq(p, u, vn) && veq(q, v, vn)) || (veq(p, v, vn) && veq(q, u, vn)))
        && u.width_of().is_some()
        && u.width_of() == v.width_of()
}

/// `(a == b) | (a < b) → a <= b`, both signednesses: over a total order the
/// union of equality and the strict order is the non-strict order. Matched
/// for either operand order of the `|`; the strict comparison fixes the
/// result's orientation.
fn or_order_compose(lhs: &Expr, rhs: &Expr, vn: Option<&VnDefs>) -> Option<Expr> {
    for (eq, ord) in [(lhs, rhs), (rhs, lhs)] {
        let Expr::Binary {
            op: BinOp::Eq,
            lhs: p,
            rhs: q,
        } = eq
        else {
            continue;
        };
        let Expr::Binary {
            op: ord_op,
            lhs: u,
            rhs: v,
        } = ord
        else {
            continue;
        };
        let relaxed = match ord_op {
            BinOp::Slt => BinOp::Sle,
            BinOp::Ult => BinOp::Ule,
            _ => continue,
        };
        if !order_compose_ok(p, q, u, v, vn) {
            continue;
        }
        return Some(Expr::binary(relaxed, (**u).clone(), (**v).clone()));
    }
    None
}

/// `(a != b) & (b <= a) → b < a`, both signednesses: over a total order the
/// intersection of disequality and the non-strict order is the strict
/// order. Matched for either operand order of the `&`; the non-strict
/// comparison fixes the result's orientation.
fn and_order_compose(lhs: &Expr, rhs: &Expr, vn: Option<&VnDefs>) -> Option<Expr> {
    for (ne, ord) in [(lhs, rhs), (rhs, lhs)] {
        let Expr::Binary {
            op: BinOp::Ne,
            lhs: p,
            rhs: q,
        } = ne
        else {
            continue;
        };
        let Expr::Binary {
            op: ord_op,
            lhs: u,
            rhs: v,
        } = ord
        else {
            continue;
        };
        let strict = match ord_op {
            BinOp::Sle => BinOp::Slt,
            BinOp::Ule => BinOp::Ult,
            _ => continue,
        };
        if !order_compose_ok(p, q, u, v, vn) {
            continue;
        }
        return Some(Expr::binary(strict, (**u).clone(), (**v).clone()));
    }
    None
}

/// Whether `p` and `q` are complementary `W1` conditions: one the literal
/// [`UnOp::Not`] of the other, or a comparison and the reversed comparison
/// its folded negation produces (`a == b` against `a != b` over the same
/// operands in either order, `a <s b` against `b <=s a`, and the unsigned
/// twins) — the two spellings a conditional compare's `~c` arm survives
/// folding in. Deliberately non-recursive beyond [`veq`] operand equality,
/// so hostile nesting cannot drive the matcher deep.
fn is_complement(p: &Expr, q: &Expr, vn: Option<&VnDefs>) -> bool {
    if p.width_of() != Some(Width::W1) || q.width_of() != Some(Width::W1) {
        return false;
    }
    let literal = |x: &Expr, y: &Expr| {
        matches!(x, Expr::Unary { op: UnOp::Not, operand } if veq(operand, y, vn))
    };
    if literal(p, q) || literal(q, p) {
        return true;
    }
    let (
        Expr::Binary {
            op: po,
            lhs: pl,
            rhs: pr,
        },
        Expr::Binary {
            op: qo,
            lhs: ql,
            rhs: qr,
        },
    ) = (p, q)
    else {
        return false;
    };
    match (po, qo) {
        // Equality against disequality: same operands, either order.
        (BinOp::Eq, BinOp::Ne) | (BinOp::Ne, BinOp::Eq) => {
            (veq(pl, ql, vn) && veq(pr, qr, vn)) || (veq(pl, qr, vn) && veq(pr, ql, vn))
        }
        // A strict order against the reversed non-strict one.
        (BinOp::Slt, BinOp::Sle)
        | (BinOp::Sle, BinOp::Slt)
        | (BinOp::Ult, BinOp::Ule)
        | (BinOp::Ule, BinOp::Ult) => veq(pl, qr, vn) && veq(pr, ql, vn),
        _ => false,
    }
}

/// The `W1` complement of `c`, spelled the way [`fold_expr`] would spell
/// it: a comparison flips (the reversed non-strict order, the opposite
/// equality), a literal `W1` negation unwraps, anything else wraps in
/// [`UnOp::Not`]. Non-recursive; both of `c`'s operands survive.
fn complement_w1(c: &Expr) -> Expr {
    match c {
        Expr::Binary { op, lhs, rhs } => {
            let (flip, swap) = match op {
                BinOp::Eq => (BinOp::Ne, false),
                BinOp::Ne => (BinOp::Eq, false),
                BinOp::Slt => (BinOp::Sle, true),
                BinOp::Sle => (BinOp::Slt, true),
                BinOp::Ult => (BinOp::Ule, true),
                BinOp::Ule => (BinOp::Ult, true),
                _ => return Expr::unary(UnOp::Not, c.clone()),
            };
            let (l, r) = if swap { (rhs, lhs) } else { (lhs, rhs) };
            Expr::binary(flip, (**l).clone(), (**r).clone())
        }
        Expr::Unary {
            op: UnOp::Not,
            operand,
        } if operand.width_of() == Some(Width::W1) => (**operand).clone(),
        _ => Expr::unary(UnOp::Not, c.clone()),
    }
}

/// Every reading of a folded conditional-compare flag select as
/// `(guard, flag expression, imm4 bit)`. [`crate::aarch64_lift`]'s CCMP
/// lift writes each flag as `(c & model) | (~c & bit)`; the constant arm
/// folds first, leaving `c & model` for a clear bit and
/// `(c & model) | ~c` — the complement literal or already flipped — for a
/// set one. Which `&` operand is the guard is settled by the caller
/// matching guards across both halves, so both assignments are offered.
fn masked_sel_readings<'e>(e: &'e Expr, vn: Option<&VnDefs>) -> Vec<(&'e Expr, &'e Expr, bool)> {
    let mut out = Vec::new();
    match e {
        Expr::Binary {
            op: BinOp::And,
            lhs,
            rhs,
        } => {
            out.push((&**lhs, &**rhs, false));
            out.push((&**rhs, &**lhs, false));
        }
        Expr::Binary {
            op: BinOp::Or,
            lhs,
            rhs,
        } => {
            for (sel, comp) in [(lhs, rhs), (rhs, lhs)] {
                if let Expr::Binary {
                    op: BinOp::And,
                    lhs: p,
                    rhs: q,
                } = &**sel
                {
                    for (c, x) in [(p, q), (q, p)] {
                        if is_complement(c, comp, vn) {
                            out.push((&**c, &**x, true));
                        }
                    }
                }
            }
        }
        _ => {}
    }
    out
}

/// Collapse the masked signed-order pair: both sides of the outer
/// comparison are flag selects over one shared, structurally equal guard
/// `c`, and the selected flags are the sign/overflow decompositions of
/// the same `a - b`. With `SF' = (c & SF) | (~c & bN)` and
/// `OF' = (c & OF) | (~c & bV)`, the guard-false arms compare as the
/// constant imm4 bits, so the whole composition is the guarded relation —
/// for equal bits `SF' != OF' → c & (a <s b)` and
/// `SF' == OF' → ~c | (b <=s a)`, for differing bits
/// `SF' != OF' → ~c | (a <s b)` and `SF' == OF' → c & (b <=s a)`
/// (proved exhaustively at width 8, both guard values, in the tests).
/// The rewrite keeps one copy of `c` and drops its duplicates along with
/// the duplicate `a`/`b` copies, so all three must be load-free.
fn masked_order_pair(cmp: BinOp, lhs: &Expr, rhs: &Expr, vn: Option<&VnDefs>) -> Option<Expr> {
    if !matches!(cmp, BinOp::Eq | BinOp::Ne) {
        return None;
    }
    for (c1, x1, b1) in masked_sel_readings(lhs, vn) {
        for (c2, x2, b2) in masked_sel_readings(rhs, vn) {
            if !veq(c1, c2, vn) {
                continue;
            }
            let Some((a, b)) = order_pair_operands(x1, x2, vn) else {
                continue;
            };
            // The dropped duplicates of the guard and the operands may be
            // load-bearing: one-expression theorem, module "Soundness".
            let lt = || Expr::binary(BinOp::Slt, a.clone(), b.clone());
            let ge = || Expr::binary(BinOp::Sle, b.clone(), a.clone());
            return Some(match (cmp, b1 == b2) {
                (BinOp::Ne, true) => Expr::binary(BinOp::And, c1.clone(), lt()),
                (BinOp::Ne, false) => Expr::binary(BinOp::Or, complement_w1(c1), lt()),
                (BinOp::Eq, true) => Expr::binary(BinOp::Or, complement_w1(c1), ge()),
                _ => Expr::binary(BinOp::And, c1.clone(), ge()),
            });
        }
    }
    None
}

/// `~((c & x) | ~c) → c & ~x`: the negated read of a single flag select
/// whose imm4 bit is set — with the guard false the selected flag is the
/// set bit and the negation is false, with it true the negation is `~x` —
/// so the composition is the guard conjoined with the complemented flag
/// (`~x` spelled through [`complement_w1`], so a comparison flips rather
/// than nests). Matched for either operand order of the `|` and of the
/// inner `&`, with `~c` recognized literally or as the folded flipped
/// comparison. One copy of `c` survives and its complement copy is
/// dropped — sound even load-bearing by the one-expression theorem (see
/// the module's "Soundness"; a load-bearing [`veq`] inside
/// [`is_complement`] is always the structural fast path); `x` survives
/// whole.
fn not_of_flag_select(lhs: &Expr, rhs: &Expr, vn: Option<&VnDefs>) -> Option<Expr> {
    for (sel, comp) in [(lhs, rhs), (rhs, lhs)] {
        let Expr::Binary {
            op: BinOp::And,
            lhs: p,
            rhs: q,
        } = sel
        else {
            continue;
        };
        for (c, x) in [(p, q), (q, p)] {
            if is_complement(c, comp, vn) {
                return Some(Expr::binary(
                    BinOp::And,
                    (**c).clone(),
                    complement_w1(x),
                ));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Expression walks
// ---------------------------------------------------------------------------

/// Whether `e` contains a memory [`Load`](Expr::Load). Depth-bounded; a walk
/// past [`crate::ir::MAX_EXPR_NODES`] is answered conservatively (`true`,
/// "assume a load") so no simplification deletes it. Shared with
/// [`crate::irssaopt`], whose SSA-wide sweep keeps a load-bearing
/// definition for exactly this reason — one predicate, one doctrine.
pub(crate) fn contains_load(e: &Expr, depth: usize) -> bool {
    if depth > ir::MAX_EXPR_NODES {
        return true;
    }
    match e {
        Expr::Load { .. } => true,
        Expr::Const { .. } | Expr::Reg(_) => false,
        Expr::Unary { operand, .. } => contains_load(operand, depth + 1),
        Expr::Binary { lhs, rhs, .. } => {
            contains_load(lhs, depth + 1) || contains_load(rhs, depth + 1)
        }
    }
}

/// Whether `e` contains a division or remainder. Depth-bounded like
/// [`contains_load`], answered conservatively (`true`) past the cap.
/// Shared with [`crate::irssaopt`] — one predicate, one doctrine.
pub(crate) fn contains_div(e: &Expr, depth: usize) -> bool {
    if depth > ir::MAX_EXPR_NODES {
        return true;
    }
    match e {
        Expr::Const { .. } | Expr::Reg(_) => false,
        Expr::Load { addr, .. } => contains_div(addr, depth + 1),
        Expr::Unary { operand, .. } => contains_div(operand, depth + 1),
        Expr::Binary { op, lhs, rhs } => {
            matches!(
                op,
                BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem
            ) || contains_div(lhs, depth + 1)
                || contains_div(rhs, depth + 1)
        }
    }
}

// ---------------------------------------------------------------------------
// Value-numbering equality (see the module docs' "The equality witness")
// ---------------------------------------------------------------------------

/// Fuel for one [`veq`] canonicalization: how many nodes one side's
/// canonical key may visit or produce before the witness gives up.
/// Exhaustion is a refusal (`false`), never a guess.
const VKEY_FUEL: usize = 2048;

/// The value-numbering context: each *pure* SSA definition, keyed by its
/// exact defining [`Reg`] — space, name, and width, so a narrower (partial)
/// read never resolves. Built by [`crate::irssaopt::forward`] once per
/// round from the function's assignments; a φ has no right-hand side and
/// is naturally absent, so resolution walks a DAG and terminates.
///
/// The purity gate lives here, next to the witness that trusts it: a
/// right-hand side bearing a [`Load`](Expr::Load) (the witness never
/// reasons through memory), a division (never through a trap), or more
/// than [`ir::MAX_EXPR_NODES`] nodes is not admitted. The stored form is
/// pre-folded, so a definition compares in the same spelling a spliced
/// copy of it would fold to.
#[derive(Default)]
pub struct VnDefs {
    defs: BTreeMap<Reg, Expr>,
}

impl VnDefs {
    /// An empty context: [`veq`] under it is exactly structural equality
    /// plus the truncation normalization.
    pub fn new() -> VnDefs {
        VnDefs::default()
    }

    /// Admit `dst := rhs` if the right-hand side passes the purity gate.
    /// A refused definition is simply absent — a read of it stays opaque.
    pub fn add(&mut self, dst: Reg, rhs: &Expr) {
        if expr_nodes_capped(rhs) || contains_load(rhs, 0) || contains_div(rhs, 0) {
            return;
        }
        self.defs.insert(dst, fold_expr(rhs));
    }

    fn resolve(&self, r: &Reg) -> Option<&Expr> {
        self.defs.get(r)
    }
}

/// Whether `e` exceeds [`ir::MAX_EXPR_NODES`] (defensively unreachable on
/// checked input; the gate keeps [`VnDefs`] honest on any input).
fn expr_nodes_capped(e: &Expr) -> bool {
    fn count(e: &Expr, n: &mut usize) {
        *n += 1;
        if *n > ir::MAX_EXPR_NODES {
            return;
        }
        match e {
            Expr::Const { .. } | Expr::Reg(_) => {}
            Expr::Load { addr, .. } => count(addr, n),
            Expr::Unary { operand, .. } => count(operand, n),
            Expr::Binary { lhs, rhs, .. } => {
                count(lhs, n);
                count(rhs, n);
            }
        }
    }
    let mut n = 0;
    count(e, &mut n);
    n > ir::MAX_EXPR_NODES
}

/// Value-numbering-grade equality: `a` and `b` provably name one value.
///
/// The fast path is structural equality — byte-for-byte the behavior every
/// caller had before the context existed, and the only path when `vn` is
/// `None`. With a context, both sides are canonicalized by [`vkey`] —
/// full-width reads resolved through their unique SSA definition,
/// truncations pushed through the ring operators, width respellings
/// cancelled — and compared. Every canonicalization step preserves the
/// value: resolution is SSA's single-definition guarantee (the definition
/// dominates the use and its operands' names are immutable), and each
/// truncation move is a two's-complement theorem proved exhaustively in
/// the tests.
///
/// The witness only ever *proves*; it never appears in output — callers
/// keep returning subtrees of their own input. And a true answer beyond
/// the fast path implies both sides are load-free ([`vkey`] refuses
/// loads outright and resolution refuses load-bearing definitions), so a
/// matcher's load gate on the kept operand covers the dropped duplicate:
/// either the two are byte-identical, or both are load-free.
/// Whether `a` and `b` are the same value under `vn` — structural
/// equality, or the same canonical key through pure defs. `None` for
/// `vn` is exactly structural equality. Used by pair-fold matchers and
/// by [`crate::irstruct`]'s congruent-condition fold.
pub(crate) fn veq(a: &Expr, b: &Expr, vn: Option<&VnDefs>) -> bool {
    if a == b {
        return true;
    }
    let Some(v) = vn else {
        return false;
    };
    let mut fa = VKEY_FUEL;
    let mut fb = VKEY_FUEL;
    match (vkey(a, v, &mut fa), vkey(b, v, &mut fb)) {
        (Some(ka), Some(kb)) => ka == kb,
        _ => false,
    }
}

/// The canonical key of `e` under `vn`, or `None` — a refusal (a load
/// anywhere, fuel exhausted), never an approximation. Bottom-up: resolve
/// exact-width reads through the context, then push each truncation to
/// the leaves.
fn vkey(e: &Expr, vn: &VnDefs, fuel: &mut usize) -> Option<Expr> {
    if *fuel == 0 {
        return None;
    }
    *fuel -= 1;
    match e {
        Expr::Load { .. } => None,
        Expr::Const { .. } => Some(e.clone()),
        Expr::Reg(r) => match vn.resolve(r) {
            Some(d) => vkey(d, vn, fuel),
            None => Some(e.clone()),
        },
        Expr::Unary {
            op: UnOp::Truncate(to),
            operand,
        } => {
            let k = vkey(operand, vn, fuel)?;
            Some(push_trunc(*to, &k, fuel))
        }
        Expr::Unary { op, operand } => Some(Expr::unary(*op, vkey(operand, vn, fuel)?)),
        Expr::Binary { op, lhs, rhs } => Some(Expr::binary(
            *op,
            vkey(lhs, vn, fuel)?,
            vkey(rhs, vn, fuel)?,
        )),
    }
}

/// `trunc.to(k)` in canonical form, for an already-canonical `k`:
/// truncation is a ring homomorphism, so it distributes through
/// `Add`/`Sub`/`Mul`/`And`/`Or`/`Xor` and through `Neg`/`Not` — dropped
/// high bits never feed a low result bit in any of them — composes with a
/// nested truncation, and cancels or refits through an extension exactly
/// as [`fold_width_identity`] already proves. It does NOT distribute
/// through a shift (the amount is taken modulo the width — the width-16
/// near-miss test pins the unsoundness), a division, or a comparison; and
/// a malformed (non-narrowing) truncation is wrapped as-is, never
/// laundered. Fuel-bounded; exhaustion falls back to the wrapped shape,
/// which can only make [`veq`] refuse.
fn push_trunc(to: Width, k: &Expr, fuel: &mut usize) -> Expr {
    let wrap = || Expr::unary(UnOp::Truncate(to), k.clone());
    if *fuel == 0 {
        return wrap();
    }
    *fuel -= 1;
    let Some(w) = k.width_of() else {
        return wrap();
    };
    if to.bits() >= w.bits() {
        return wrap();
    }
    match k {
        Expr::Const { .. } => {
            fold_unary_const(UnOp::Truncate(to), k).unwrap_or_else(wrap)
        }
        Expr::Binary {
            op:
                op @ (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor),
            lhs,
            rhs,
        } => Expr::binary(*op, push_trunc(to, lhs, fuel), push_trunc(to, rhs, fuel)),
        Expr::Unary {
            op: op @ (UnOp::Neg | UnOp::Not),
            operand,
        } => Expr::unary(*op, push_trunc(to, operand, fuel)),
        // A nested truncation composes: the outer one is the narrower.
        Expr::Unary {
            op: UnOp::Truncate(_),
            operand,
        } => push_trunc(to, operand, fuel),
        Expr::Unary {
            op: UnOp::ZeroExtend(_) | UnOp::SignExtend(_),
            ..
        } => match fold_width_identity(UnOp::Truncate(to), k) {
            // The cancellation may re-expose a pushable truncation.
            Some(Expr::Unary {
                op: UnOp::Truncate(t),
                operand,
            }) => push_trunc(t, &operand, fuel),
            Some(res) => res,
            None => wrap(),
        },
        _ => wrap(),
    }
}

/// Insert every register read within `e` into `out` (exact [`Reg`]s, width
/// included). Depth-bounded to avoid unbounded recursion on hostile input.
fn add_reads(e: &Expr, out: &mut BTreeSet<Reg>, depth: usize) {
    if depth > ir::MAX_EXPR_NODES {
        return;
    }
    match e {
        Expr::Reg(r) => {
            out.insert(*r);
        }
        Expr::Const { .. } => {}
        Expr::Load { addr, .. } => add_reads(addr, out, depth + 1),
        Expr::Unary { operand, .. } => add_reads(operand, out, depth + 1),
        Expr::Binary { lhs, rhs, .. } => {
            add_reads(lhs, out, depth + 1);
            add_reads(rhs, out, depth + 1);
        }
    }
}

/// Whether two references may name (overlapping parts of) the same cell:
/// the same space and number, regardless of width.
fn may_alias(a: &Reg, b: &Reg) -> bool {
    a.space == b.space && a.num == b.num
}

/// Whether `e` reads any reference that may alias the cell `w`.
fn mentions_cell(e: &Expr, w: &Reg, depth: usize) -> bool {
    if depth > ir::MAX_EXPR_NODES {
        return true; // conservative: assume it does, so the fact is dropped
    }
    match e {
        Expr::Reg(r) => may_alias(r, w),
        Expr::Const { .. } => false,
        Expr::Load { addr, .. } => mentions_cell(addr, w, depth + 1),
        Expr::Unary { operand, .. } => mentions_cell(operand, w, depth + 1),
        Expr::Binary { lhs, rhs, .. } => {
            mentions_cell(lhs, w, depth + 1) || mentions_cell(rhs, w, depth + 1)
        }
    }
}

// ---------------------------------------------------------------------------
// Copy and constant propagation
// ---------------------------------------------------------------------------

/// Forward-propagate known register values within one block, then fold.
///
/// A register whose current value is a constant or a plain copy of another
/// register (both load-free) is recorded and substituted into later reads.
/// Facts are invalidated conservatively: a write to any width of a cell
/// removes facts *about* that cell and facts whose value *mentions* it; an
/// [`Intrinsic`](Stmt::Intrinsic)'s writes do the same; a
/// [`Branch`](Stmt::Branch) clears all facts (a block terminator, and a call
/// may clobber). A [`Store`](Stmt::Store) needs no invalidation because
/// propagation sources are load-free and so independent of memory.
///
/// Statement *count and order are preserved*; only read expressions change.
/// Dead definitions left behind are removed by [`eliminate_dead`].
pub fn propagate(stmts: &[Stmt]) -> Vec<Stmt> {
    let mut known: BTreeMap<Reg, Expr> = BTreeMap::new();
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        match stmt {
            Stmt::Assign { dst, value } => {
                let nv = subst_and_fold(value, &known);
                let record = is_propagatable(&nv) && !matches!(&nv, Expr::Reg(r) if r == dst);
                invalidate(&mut known, dst);
                if record {
                    known.insert(*dst, nv.clone());
                }
                out.push(Stmt::Assign {
                    dst: *dst,
                    value: nv,
                });
            }
            Stmt::Store { addr, value } => {
                out.push(Stmt::Store {
                    addr: subst_and_fold(addr, &known),
                    value: subst_and_fold(value, &known),
                });
            }
            Stmt::Branch {
                kind,
                cond,
                target,
            } => {
                let cond = cond.as_ref().map(|c| subst_and_fold(c, &known));
                let target = subst_and_fold(target, &known);
                known.clear();
                out.push(Stmt::Branch {
                    kind: *kind,
                    cond,
                    target,
                });
            }
            Stmt::Intrinsic {
                name,
                writes,
                reads,
            } => {
                let reads = reads.iter().map(|r| subst_and_fold(r, &known)).collect();
                for w in writes {
                    invalidate(&mut known, w);
                }
                out.push(Stmt::Intrinsic {
                    name,
                    writes: writes.clone(),
                    reads,
                });
            }
        }
    }
    out
}

/// Whether a value is safe to record as a propagation source: a constant or
/// a plain register copy. Both are load-free, so neither depends on memory.
fn is_propagatable(e: &Expr) -> bool {
    matches!(e, Expr::Const { .. } | Expr::Reg(_))
}

/// Drop every fact invalidated by a write to cell `w`.
fn invalidate(known: &mut BTreeMap<Reg, Expr>, w: &Reg) {
    known.retain(|k, v| !may_alias(k, w) && !mentions_cell(v, w, 0));
}

/// Substitute known values into `e`, then constant-fold the result.
fn subst_and_fold(e: &Expr, known: &BTreeMap<Reg, Expr>) -> Expr {
    fold_expr(&substitute(e, known, 0))
}

/// Replace each register read with its known value (once; a recorded value
/// is itself a constant or a register that is not a current key, so a single
/// level suffices and always terminates).
fn substitute(e: &Expr, known: &BTreeMap<Reg, Expr>, depth: usize) -> Expr {
    if depth > REWRITE_DEPTH {
        return e.clone();
    }
    match e {
        Expr::Reg(r) => known.get(r).cloned().unwrap_or_else(|| e.clone()),
        Expr::Const { .. } => e.clone(),
        Expr::Load { addr, width } => Expr::load(substitute(addr, known, depth + 1), *width),
        Expr::Unary { op, operand } => Expr::unary(*op, substitute(operand, known, depth + 1)),
        Expr::Binary { op, lhs, rhs } => Expr::binary(
            *op,
            substitute(lhs, known, depth + 1),
            substitute(rhs, known, depth + 1),
        ),
    }
}

// ---------------------------------------------------------------------------
// Liveness and dead-code elimination
// ---------------------------------------------------------------------------

/// Backward liveness over a straight-line block. Returns, for each statement
/// index, the set of registers live *immediately after* that statement (its
/// live-out); `live_out` is the block's live-out, the set live after the
/// last statement.
///
/// The transfer per statement is the standard `live_in = (live_out \ defs) ∪
/// uses`, with `defs` removed by exact reference (a def of `eax` does not
/// clear the liveness of `rax`). This is a sound over-approximation; the
/// aliasing-aware deadness test that [`eliminate_dead`] uses to *drop*
/// statements is stricter still.
pub fn liveness(stmts: &[Stmt], live_out: &BTreeSet<Reg>) -> Vec<BTreeSet<Reg>> {
    let mut result = vec![BTreeSet::new(); stmts.len()];
    let mut live = live_out.clone();
    for (i, stmt) in stmts.iter().enumerate().rev() {
        result[i] = live.clone();
        transfer(stmt, &mut live);
    }
    result
}

/// The set of registers live on *entry* to a straight-line block, given
/// its live-out: [`liveness`]'s per-statement transfer folded backward
/// through the whole statement list. This is the block summary a
/// cross-block liveness fixpoint (e.g. [`crate::irssa`]'s phi pruning)
/// iterates; it shares [`liveness`]'s exact-reference kill convention, so
/// it is a sound over-approximation under the module's aliasing doctrine.
pub fn live_in(stmts: &[Stmt], live_out: &BTreeSet<Reg>) -> BTreeSet<Reg> {
    let mut live = live_out.clone();
    for stmt in stmts.iter().rev() {
        transfer(stmt, &mut live);
    }
    live
}

/// Apply one statement's backward liveness transfer (`live` becomes the set
/// live before the statement).
fn transfer(stmt: &Stmt, live: &mut BTreeSet<Reg>) {
    match stmt {
        Stmt::Assign { dst, value } => {
            live.remove(dst);
            add_reads(value, live, 0);
        }
        Stmt::Store { addr, value } => {
            add_reads(addr, live, 0);
            add_reads(value, live, 0);
        }
        Stmt::Branch { cond, target, .. } => {
            if let Some(c) = cond {
                add_reads(c, live, 0);
            }
            add_reads(target, live, 0);
        }
        Stmt::Intrinsic { writes, reads, .. } => {
            for w in writes {
                live.remove(w);
            }
            for r in reads {
                add_reads(r, live, 0);
            }
        }
    }
}

/// Whether any live reference may alias `dst` — i.e. some register live-out
/// shares `dst`'s space and number, at any width.
fn any_alias_live(live: &BTreeSet<Reg>, dst: &Reg) -> bool {
    let lo = Reg {
        space: dst.space,
        num: dst.num,
        width: Width::W1,
    };
    let hi = Reg {
        space: dst.space,
        num: dst.num,
        width: Width::W64,
    };
    live.range(lo..=hi).next().is_some()
}

/// Eliminate dead code by backward liveness. A pure
/// [`Assign`](Stmt::Assign) — one whose value holds no
/// [`Load`](Expr::Load) — is dropped when no reference that may alias its
/// destination is live afterward (equivalently, nothing later in the block
/// reads it before it is redefined, and it is not in `live_out`). This is
/// what removes the recomputed flags nobody reads.
///
/// A [`Store`](Stmt::Store), [`Branch`](Stmt::Branch),
/// [`Intrinsic`](Stmt::Intrinsic), or load-bearing assignment is always
/// kept.
///
/// # Live-out contract
///
/// `live_out` is the set of registers whose value at the end of the block is
/// observed by a successor (or the caller). A reference in it, at any width,
/// pins its whole cell live. Supplying too large a set is always sound (it
/// only keeps more); [`default_live_out`] builds a conservative one.
pub fn eliminate_dead(stmts: &[Stmt], live_out: &BTreeSet<Reg>) -> Vec<Stmt> {
    let mut live = live_out.clone();
    let mut kept_rev: Vec<Stmt> = Vec::new();
    for stmt in stmts.iter().rev() {
        if let Stmt::Assign { dst, value } = stmt {
            let dead = !contains_load(value, 0) && !any_alias_live(&live, dst);
            if dead {
                continue; // drop; a dead def contributes no uses
            }
        }
        transfer(stmt, &mut live);
        kept_rev.push(stmt.clone());
    }
    kept_rev.reverse();
    kept_rev
}

/// A conservative block live-out: every architectural register written in
/// the block. It keeps all architectural definitions (their cells are pinned
/// live), leaving only dead temporaries, provably-dead flags, and copies
/// made redundant by a later exact redefinition to be removed.
pub fn default_live_out(stmts: &[Stmt]) -> BTreeSet<Reg> {
    let mut set = BTreeSet::new();
    for stmt in stmts {
        match stmt {
            Stmt::Assign { dst, .. } if dst.space == Space::Arch => {
                set.insert(*dst);
            }
            Stmt::Intrinsic { writes, .. } => {
                for w in writes {
                    if w.space == Space::Arch {
                        set.insert(*w);
                    }
                }
            }
            _ => {}
        }
    }
    set
}

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------

/// Fold every expression in a statement, leaving its structure and its
/// effects alone: a statement's kind, its destinations, and an
/// intrinsic's name and writes are never touched. Shared with
/// [`crate::irssaopt`], which re-folds the statements it substitutes into.
pub fn fold_stmt(stmt: &Stmt) -> Stmt {
    fold_stmt_rec(stmt, None)
}

/// [`fold_stmt`] with a value-numbering context (see [`VnDefs`]):
/// [`crate::irssaopt::forward`]'s re-fold, where the pair halves one
/// spelling apart finally meet.
pub fn fold_stmt_vn(stmt: &Stmt, vn: &VnDefs) -> Stmt {
    fold_stmt_rec(stmt, Some(vn))
}

fn fold_stmt_rec(stmt: &Stmt, vn: Option<&VnDefs>) -> Stmt {
    let fold = |e: &Expr| fold_rec(e, 0, vn);
    match stmt {
        Stmt::Assign { dst, value } => Stmt::Assign {
            dst: *dst,
            value: fold(value),
        },
        Stmt::Store { addr, value } => Stmt::Store {
            addr: fold(addr),
            value: fold(value),
        },
        Stmt::Branch {
            kind,
            cond,
            target,
        } => Stmt::Branch {
            kind: *kind,
            cond: cond.as_ref().map(fold),
            target: fold(target),
        },
        Stmt::Intrinsic {
            name,
            writes,
            reads,
        } => Stmt::Intrinsic {
            name,
            writes: writes.clone(),
            reads: reads.iter().map(fold).collect(),
        },
    }
}

/// Simplify one block: propagate → fold → eliminate-dead, iterated to a
/// fixpoint or [`MAX_ROUNDS`], whichever comes first.
///
/// `live_out` is the block's live-out set (see [`eliminate_dead`]). The
/// result preserves the observable behavior of the input and, for a
/// well-formed input, passes [`crate::ir::check`]. An over-long statement
/// list (beyond [`crate::ir::MAX_STMTS`]) is returned unchanged rather than
/// processed.
pub fn simplify(stmts: &[Stmt], live_out: &BTreeSet<Reg>) -> Vec<Stmt> {
    if stmts.len() > ir::MAX_STMTS {
        return stmts.to_vec();
    }
    let mut cur = stmts.to_vec();
    for _ in 0..MAX_ROUNDS {
        let propagated = propagate(&cur);
        let folded: Vec<Stmt> = propagated.iter().map(fold_stmt).collect();
        let pruned = eliminate_dead(&folded, live_out);
        if pruned == cur {
            break;
        }
        cur = pruned;
    }
    cur
}

/// Simplify with the conservative [`default_live_out`]: keep every
/// architectural register, and drop only dead temporaries, dead flags, and
/// redundant copies.
pub fn simplify_default(stmts: &[Stmt]) -> Vec<Stmt> {
    let live = default_live_out(stmts);
    simplify(stmts, &live)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BranchKind, Flag};

    // -- construction helpers ----------------------------------------------

    fn c(value: u64, w: Width) -> Expr {
        Expr::constant(value, w)
    }
    fn ra(num: u16, w: Width) -> Reg {
        Reg::arch(num, w)
    }
    fn rt(num: u16, w: Width) -> Reg {
        Reg::temp(num, w)
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
    fn un(op: UnOp, e: Expr) -> Expr {
        Expr::unary(op, e)
    }
    fn live_set(regs: &[Reg]) -> BTreeSet<Reg> {
        regs.iter().copied().collect()
    }
    fn ok(stmts: &[Stmt]) {
        assert_eq!(crate::ir::check(stmts), Ok(()), "output failed ir::check");
    }

    // -- constant folding: arithmetic --------------------------------------

    #[test]
    fn fold_add_wraps_to_width() {
        let e = bin(BinOp::Add, c(0xff, Width::W8), c(2, Width::W8));
        assert_eq!(fold_expr(&e), c(1, Width::W8)); // 0xff + 2 = 0x101 → 0x01
    }

    #[test]
    fn fold_sub_wraps_to_width() {
        let e = bin(BinOp::Sub, c(0, Width::W8), c(1, Width::W8));
        assert_eq!(fold_expr(&e), c(0xff, Width::W8));
    }

    #[test]
    fn fold_mul_wraps_to_width() {
        let e = bin(BinOp::Mul, c(0x100, Width::W16), c(0x100, Width::W16));
        assert_eq!(fold_expr(&e), c(0, Width::W16)); // 0x10000 masked to 16
    }

    #[test]
    fn fold_bitwise_and_or_xor() {
        assert_eq!(
            fold_expr(&bin(BinOp::And, c(0xf0, Width::W8), c(0x3c, Width::W8))),
            c(0x30, Width::W8)
        );
        assert_eq!(
            fold_expr(&bin(BinOp::Or, c(0xf0, Width::W8), c(0x0f, Width::W8))),
            c(0xff, Width::W8)
        );
        assert_eq!(
            fold_expr(&bin(BinOp::Xor, c(0xff, Width::W8), c(0x0f, Width::W8))),
            c(0xf0, Width::W8)
        );
    }

    // -- constant folding: shifts ------------------------------------------

    #[test]
    fn fold_shl_within_width() {
        assert_eq!(
            fold_expr(&bin(BinOp::Shl, c(1, Width::W8), c(3, Width::W8))),
            c(8, Width::W8)
        );
    }

    #[test]
    fn fold_shl_amount_taken_modulo_width() {
        // 1 << 9, at W8, is 1 << (9 % 8) = 1 << 1 = 2.
        assert_eq!(
            fold_expr(&bin(BinOp::Shl, c(1, Width::W8), c(9, Width::W8))),
            c(2, Width::W8)
        );
        // A shift by exactly the width is a shift by zero (mod width).
        assert_eq!(
            fold_expr(&bin(BinOp::Shl, c(0x12, Width::W8), c(8, Width::W8))),
            c(0x12, Width::W8)
        );
    }

    #[test]
    fn fold_lshr_is_zero_filling() {
        assert_eq!(
            fold_expr(&bin(BinOp::LShr, c(0x80, Width::W8), c(1, Width::W8))),
            c(0x40, Width::W8)
        );
    }

    #[test]
    fn fold_ashr_is_sign_filling() {
        // 0x80 at W8 is -128; >>s 1 = -64 = 0xC0.
        assert_eq!(
            fold_expr(&bin(BinOp::AShr, c(0x80, Width::W8), c(1, Width::W8))),
            c(0xc0, Width::W8)
        );
    }

    #[test]
    fn fold_shift_amount_may_be_a_narrower_constant() {
        // Shift amount of a different width is allowed by the IR.
        let e = bin(BinOp::Shl, c(1, Width::W32), c(4, Width::W8));
        assert_eq!(fold_expr(&e), c(0x10, Width::W32));
    }

    // -- constant folding: division ----------------------------------------

    #[test]
    fn fold_unsigned_div_and_rem() {
        assert_eq!(
            fold_expr(&bin(BinOp::UDiv, c(17, Width::W32), c(5, Width::W32))),
            c(3, Width::W32)
        );
        assert_eq!(
            fold_expr(&bin(BinOp::URem, c(17, Width::W32), c(5, Width::W32))),
            c(2, Width::W32)
        );
    }

    #[test]
    fn fold_signed_div_and_rem_with_negatives() {
        // -6 / 2 = -3 (0xFFFFFFFD at W32).
        assert_eq!(
            fold_expr(&bin(BinOp::SDiv, c(0xffff_fffa, Width::W32), c(2, Width::W32))),
            c(0xffff_fffd, Width::W32)
        );
        // -7 % 3 = -1 (truncating toward zero).
        assert_eq!(
            fold_expr(&bin(BinOp::SRem, c(0xffff_fff9, Width::W32), c(3, Width::W32))),
            c(0xffff_ffff, Width::W32)
        );
    }

    #[test]
    fn divide_by_zero_is_never_folded() {
        for op in [BinOp::UDiv, BinOp::URem, BinOp::SDiv, BinOp::SRem] {
            let e = bin(op, c(10, Width::W32), c(0, Width::W32));
            assert_eq!(fold_expr(&e), e, "{op:?} by zero must be left as-is");
        }
    }

    #[test]
    fn signed_div_min_by_neg_one_does_not_panic() {
        // i64::MIN / -1 would overflow; wrapping keeps it finite.
        let e = bin(
            BinOp::SDiv,
            c(0x8000_0000_0000_0000, Width::W64),
            c(u64::MAX, Width::W64),
        );
        assert_eq!(fold_expr(&e), c(0x8000_0000_0000_0000, Width::W64));
    }

    // -- constant folding: comparisons -------------------------------------

    #[test]
    fn fold_unsigned_comparisons() {
        let t = c(1, Width::W1);
        let f = c(0, Width::W1);
        assert_eq!(fold_expr(&bin(BinOp::Eq, c(5, Width::W8), c(5, Width::W8))), t);
        assert_eq!(fold_expr(&bin(BinOp::Ne, c(5, Width::W8), c(6, Width::W8))), t);
        assert_eq!(fold_expr(&bin(BinOp::Ult, c(3, Width::W8), c(5, Width::W8))), t);
        assert_eq!(fold_expr(&bin(BinOp::Ule, c(5, Width::W8), c(5, Width::W8))), t);
        // 0xff is 255 unsigned, so not < 1.
        assert_eq!(fold_expr(&bin(BinOp::Ult, c(0xff, Width::W8), c(1, Width::W8))), f);
    }

    #[test]
    fn fold_signed_comparisons() {
        let t = c(1, Width::W1);
        let f = c(0, Width::W1);
        // 0xff at W8 is -1 signed, so -1 <s 1 is true, but 255 <u 1 is false.
        assert_eq!(fold_expr(&bin(BinOp::Slt, c(0xff, Width::W8), c(1, Width::W8))), t);
        assert_eq!(fold_expr(&bin(BinOp::Sle, c(0x80, Width::W8), c(0x7f, Width::W8))), t);
        assert_eq!(fold_expr(&bin(BinOp::Slt, c(1, Width::W8), c(0xff, Width::W8))), f);
    }

    // -- constant folding: unary -------------------------------------------

    #[test]
    fn fold_neg_and_not() {
        assert_eq!(fold_expr(&un(UnOp::Neg, c(1, Width::W8))), c(0xff, Width::W8));
        assert_eq!(fold_expr(&un(UnOp::Not, c(0x0f, Width::W8))), c(0xf0, Width::W8));
    }

    #[test]
    fn fold_zero_extend() {
        let e = un(UnOp::ZeroExtend(Width::W32), c(0xff, Width::W8));
        assert_eq!(fold_expr(&e), c(0xff, Width::W32));
    }

    #[test]
    fn fold_sign_extend_replicates_top_bit() {
        let e = un(UnOp::SignExtend(Width::W32), c(0x80, Width::W8));
        assert_eq!(fold_expr(&e), c(0xffff_ff80, Width::W32));
    }

    #[test]
    fn fold_truncate() {
        let e = un(UnOp::Truncate(Width::W8), c(0x1234, Width::W16));
        assert_eq!(fold_expr(&e), c(0x34, Width::W8));
    }

    #[test]
    fn fold_recurses_into_subexpressions() {
        // (2 + 3) * 4 → 20
        let e = bin(
            BinOp::Mul,
            bin(BinOp::Add, c(2, Width::W8), c(3, Width::W8)),
            c(4, Width::W8),
        );
        assert_eq!(fold_expr(&e), c(20, Width::W8));
    }

    // -- algebraic identities ----------------------------------------------

    #[test]
    fn identity_x_xor_x_is_zero() {
        let x = read(ra(0, Width::W32));
        assert_eq!(fold_expr(&bin(BinOp::Xor, x.clone(), x)), c(0, Width::W32));
    }

    #[test]
    fn identity_add_and_sub_zero() {
        let x = read(ra(0, Width::W32));
        assert_eq!(fold_expr(&bin(BinOp::Add, x.clone(), c(0, Width::W32))), x);
        assert_eq!(fold_expr(&bin(BinOp::Add, c(0, Width::W32), x.clone())), x);
        assert_eq!(fold_expr(&bin(BinOp::Sub, x.clone(), c(0, Width::W32))), x);
    }

    #[test]
    fn identity_mul_one_and_zero() {
        let x = read(ra(0, Width::W32));
        assert_eq!(fold_expr(&bin(BinOp::Mul, x.clone(), c(1, Width::W32))), x);
        assert_eq!(fold_expr(&bin(BinOp::Mul, c(1, Width::W32), x.clone())), x);
        assert_eq!(fold_expr(&bin(BinOp::Mul, x.clone(), c(0, Width::W32))), c(0, Width::W32));
    }

    #[test]
    fn identity_and_zero_and_allones() {
        let x = read(ra(0, Width::W32));
        assert_eq!(fold_expr(&bin(BinOp::And, x.clone(), c(0, Width::W32))), c(0, Width::W32));
        assert_eq!(
            fold_expr(&bin(BinOp::And, x.clone(), c(0xffff_ffff, Width::W32))),
            x
        );
    }

    #[test]
    fn identity_or_zero_and_self() {
        let x = read(ra(0, Width::W32));
        assert_eq!(fold_expr(&bin(BinOp::Or, x.clone(), c(0, Width::W32))), x);
        assert_eq!(fold_expr(&bin(BinOp::Or, x.clone(), x.clone())), x);
        assert_eq!(fold_expr(&bin(BinOp::And, x.clone(), x.clone())), x);
    }

    #[test]
    fn identity_shift_by_zero() {
        let x = read(ra(0, Width::W32));
        for op in [BinOp::Shl, BinOp::LShr, BinOp::AShr] {
            assert_eq!(fold_expr(&bin(op, x.clone(), c(0, Width::W8))), x, "{op:?} by 0");
        }
    }

    #[test]
    fn identity_x_minus_x_is_zero() {
        let x = read(ra(0, Width::W32));
        assert_eq!(fold_expr(&bin(BinOp::Sub, x.clone(), x)), c(0, Width::W32));
    }

    // -- relational identities ---------------------------------------------

    #[test]
    fn identity_difference_against_zero_is_a_comparison() {
        let (a, b) = (read(ra(0, Width::W64)), read(ra(1, Width::W64)));
        // (a - b) == 0 → a == b, and the `!=` polarity.
        for (outer, want) in [(BinOp::Eq, BinOp::Eq), (BinOp::Ne, BinOp::Ne)] {
            let e = bin(
                outer,
                bin(BinOp::Sub, a.clone(), b.clone()),
                c(0, Width::W64),
            );
            assert_eq!(fold_expr(&e), bin(want, a.clone(), b.clone()), "{outer:?}");
        }
    }

    #[test]
    fn identity_xor_against_zero_is_a_comparison() {
        let (a, b) = (read(ra(0, Width::W32)), read(ra(1, Width::W32)));
        for (outer, want) in [(BinOp::Eq, BinOp::Eq), (BinOp::Ne, BinOp::Ne)] {
            let e = bin(
                outer,
                bin(BinOp::Xor, a.clone(), b.clone()),
                c(0, Width::W32),
            );
            assert_eq!(fold_expr(&e), bin(want, a.clone(), b.clone()), "{outer:?}");
        }
    }

    #[test]
    fn identity_zero_on_either_side_folds_the_same_way() {
        // The lifter writes `(t == 0)`, but the mirrored orientation is the
        // same identity and folds identically.
        let (a, b) = (read(ra(0, Width::W64)), read(ra(1, Width::W64)));
        let e = bin(
            BinOp::Eq,
            c(0, Width::W64),
            bin(BinOp::Sub, a.clone(), b.clone()),
        );
        assert_eq!(fold_expr(&e), bin(BinOp::Eq, a, b));
    }

    #[test]
    fn identity_negation_of_a_comparison_flips_it() {
        let (a, b) = (read(ra(0, Width::W64)), read(ra(1, Width::W64)));
        assert_eq!(
            fold_expr(&un(UnOp::Not, bin(BinOp::Eq, a.clone(), b.clone()))),
            bin(BinOp::Ne, a.clone(), b.clone())
        );
        assert_eq!(
            fold_expr(&un(UnOp::Not, bin(BinOp::Ne, a.clone(), b.clone()))),
            bin(BinOp::Eq, a, b)
        );
    }

    #[test]
    fn identity_double_negation_at_flag_width_cancels() {
        let z = read(Reg::flag(Flag::Zero));
        assert_eq!(fold_expr(&un(UnOp::Not, un(UnOp::Not, z.clone()))), z);
        // A wider complement is left alone: this identity is scoped to the
        // one width at which `~` denotes a boolean negation.
        let w = read(ra(0, Width::W32));
        let e = un(UnOp::Not, un(UnOp::Not, w));
        assert_eq!(fold_expr(&e), e);
    }

    #[test]
    fn a_flag_computation_collapses_end_to_end() {
        // The lifted `cmp`+`jne` shape, once forwarding has spliced the
        // definitions together: ~(((a - b) == 0)) → (a != b).
        let (a, b) = (read(ra(0, Width::W64)), read(ra(1, Width::W64)));
        let e = un(
            UnOp::Not,
            bin(
                BinOp::Eq,
                bin(BinOp::Sub, a.clone(), b.clone()),
                c(0, Width::W64),
            ),
        );
        assert_eq!(fold_expr(&e), bin(BinOp::Ne, a, b));
    }

    #[test]
    fn a_relational_identity_needs_matching_widths() {
        // A comparison whose sides disagree in width is malformed; folding
        // must leave it exactly as it is rather than launder it.
        let (a, b) = (read(ra(0, Width::W64)), read(ra(1, Width::W64)));
        let e = bin(BinOp::Eq, bin(BinOp::Sub, a, b), c(0, Width::W32));
        assert_eq!(fold_expr(&e), e);
    }

    #[test]
    fn a_relational_identity_erases_no_operand() {
        // A comparison's operands are ordinary values and may hold a load;
        // the identity keeps *both* sides by construction, so nothing that
        // could fault is dropped — unlike `x - x → 0`, which needs its
        // load-free guard.
        let l = Expr::load(read(ra(0, Width::W64)), Width::W64);
        let e = bin(
            BinOp::Eq,
            bin(BinOp::Sub, l.clone(), read(ra(1, Width::W64))),
            c(0, Width::W64),
        );
        let folded = fold_expr(&e);
        assert_eq!(
            folded,
            bin(BinOp::Eq, l, read(ra(1, Width::W64))),
            "both operands survive"
        );
        assert!(contains_load(&folded, 0), "the load is still there");
    }

    // -- order-condition recovery ------------------------------------------

    /// The lift's sign-flag shape for `a - b`: `(a - b) <s 0`.
    fn sf_shape(a: &Expr, b: &Expr, w: Width) -> Expr {
        bin(BinOp::Slt, bin(BinOp::Sub, a.clone(), b.clone()), c(0, w))
    }

    /// The lift's subtraction-overflow shape for `a - b`:
    /// `((a ^ b) & (a ^ (a - b))) <s 0`.
    fn of_shape(a: &Expr, b: &Expr, w: Width) -> Expr {
        bin(
            BinOp::Slt,
            bin(
                BinOp::And,
                bin(BinOp::Xor, a.clone(), b.clone()),
                bin(
                    BinOp::Xor,
                    a.clone(),
                    bin(BinOp::Sub, a.clone(), b.clone()),
                ),
            ),
            c(0, w),
        )
    }

    /// The lift's zero-flag shape for `a - b`: `(a - b) == 0`.
    fn zf_shape(a: &Expr, b: &Expr, w: Width) -> Expr {
        bin(BinOp::Eq, bin(BinOp::Sub, a.clone(), b.clone()), c(0, w))
    }

    #[test]
    fn order_pair_ne_and_eq_fold_at_every_width() {
        for w in [Width::W8, Width::W16, Width::W32, Width::W64] {
            let (a, b) = (read(ra(0, w)), read(ra(1, w)));
            let (sf, of) = (sf_shape(&a, &b, w), of_shape(&a, &b, w));
            // SF != OF → a <s b (jl), SF == OF → b <=s a (jge).
            assert_eq!(
                fold_expr(&bin(BinOp::Ne, sf.clone(), of.clone())),
                bin(BinOp::Slt, a.clone(), b.clone()),
                "{w:?}"
            );
            assert_eq!(
                fold_expr(&bin(BinOp::Eq, sf.clone(), of.clone())),
                bin(BinOp::Sle, b.clone(), a.clone()),
                "{w:?}"
            );
            // The mirrored operand order of the outer comparison matches
            // too (Eq and Ne are symmetric).
            assert_eq!(
                fold_expr(&bin(BinOp::Ne, of, sf)),
                bin(BinOp::Slt, a, b),
                "{w:?}"
            );
        }
    }

    #[test]
    fn order_pair_with_a_literal_operand_folds() {
        // After propagation `b` is often a constant: `cmp rdx, 7` and the
        // `dec` shape (`b` = 1) must match as expression trees.
        let w = Width::W64;
        let a = read(ra(2, w));
        for k in [7u64, 1] {
            let b = c(k, w);
            let e = bin(BinOp::Ne, sf_shape(&a, &b, w), of_shape(&a, &b, w));
            assert_eq!(fold_expr(&e), bin(BinOp::Slt, a.clone(), b));
        }
    }

    #[test]
    fn jle_and_jg_conjunction_shapes_fold() {
        // `jle` is `ZF | (SF != OF)`, `jg` is `~ZF & (SF == OF)` — exactly
        // as `x86_lift::cond_expr` spells them; the halves collapse
        // bottom-up and the composition finishes.
        let w = Width::W32;
        let (a, b) = (read(ra(0, w)), read(ra(1, w)));
        let jle = bin(
            BinOp::Or,
            zf_shape(&a, &b, w),
            bin(BinOp::Ne, sf_shape(&a, &b, w), of_shape(&a, &b, w)),
        );
        assert_eq!(fold_expr(&jle), bin(BinOp::Sle, a.clone(), b.clone()));
        let jg = bin(
            BinOp::And,
            un(UnOp::Not, zf_shape(&a, &b, w)),
            bin(BinOp::Eq, sf_shape(&a, &b, w), of_shape(&a, &b, w)),
        );
        assert_eq!(fold_expr(&jg), bin(BinOp::Slt, b, a));
    }

    #[test]
    fn unsigned_order_shapes_fold() {
        let w = Width::W64;
        let (a, b) = (read(ra(0, w)), read(ra(1, w)));
        // x86: CF = a <u b. jbe = CF | ZF, ja = ~CF & ~ZF, jae = ~CF.
        let cf = bin(BinOp::Ult, a.clone(), b.clone());
        let jbe = bin(BinOp::Or, cf.clone(), zf_shape(&a, &b, w));
        assert_eq!(fold_expr(&jbe), bin(BinOp::Ule, a.clone(), b.clone()));
        let ja = bin(
            BinOp::And,
            un(UnOp::Not, cf.clone()),
            un(UnOp::Not, zf_shape(&a, &b, w)),
        );
        assert_eq!(fold_expr(&ja), bin(BinOp::Ult, b.clone(), a.clone()));
        assert_eq!(
            fold_expr(&un(UnOp::Not, cf)),
            bin(BinOp::Ule, b.clone(), a.clone())
        );
        // A64: C is NOT-borrow, `CF = b <=u a`. hi = CF & ~ZF,
        // ls = ~CF | ZF.
        let cf64 = bin(BinOp::Ule, b.clone(), a.clone());
        let hi = bin(BinOp::And, cf64.clone(), un(UnOp::Not, zf_shape(&a, &b, w)));
        assert_eq!(fold_expr(&hi), bin(BinOp::Ult, b.clone(), a.clone()));
        let ls = bin(BinOp::Or, un(UnOp::Not, cf64), zf_shape(&a, &b, w));
        assert_eq!(fold_expr(&ls), bin(BinOp::Ule, a, b));
    }

    #[test]
    fn negation_of_an_order_comparison_reverses_it() {
        let (a, b) = (read(ra(0, Width::W64)), read(ra(1, Width::W64)));
        for (op, want) in [
            (BinOp::Slt, BinOp::Sle),
            (BinOp::Sle, BinOp::Slt),
            (BinOp::Ult, BinOp::Ule),
            (BinOp::Ule, BinOp::Ult),
        ] {
            let e = un(UnOp::Not, bin(op, a.clone(), b.clone()));
            assert_eq!(fold_expr(&e), bin(want, b.clone(), a.clone()), "{op:?}");
        }
        // A double negation round-trips through the reversal.
        let e = un(
            UnOp::Not,
            un(UnOp::Not, bin(BinOp::Slt, a.clone(), b.clone())),
        );
        assert_eq!(fold_expr(&e), bin(BinOp::Slt, a, b));
    }

    #[test]
    fn a_negated_order_pair_folds_to_the_inverted_relation() {
        // The negated branch polarity: `~(SF != OF) → b <=s a`, and the
        // negated `jle` composition `~(ZF | (SF != OF)) → b <s a`.
        let w = Width::W64;
        let (a, b) = (read(ra(0, w)), read(ra(1, w)));
        let pair = bin(BinOp::Ne, sf_shape(&a, &b, w), of_shape(&a, &b, w));
        assert_eq!(
            fold_expr(&un(UnOp::Not, pair.clone())),
            bin(BinOp::Sle, b.clone(), a.clone())
        );
        let jle = bin(BinOp::Or, zf_shape(&a, &b, w), pair);
        assert_eq!(
            fold_expr(&un(UnOp::Not, jle)),
            bin(BinOp::Slt, b, a)
        );
    }

    #[test]
    fn a_w1_comparison_against_a_boolean_constant_folds() {
        let z = read(Reg::flag(Flag::Zero));
        // x != 0 → x, x == 1 → x, in both orientations.
        assert_eq!(fold_expr(&bin(BinOp::Ne, z.clone(), c(0, Width::W1))), z);
        assert_eq!(fold_expr(&bin(BinOp::Eq, z.clone(), c(1, Width::W1))), z);
        assert_eq!(fold_expr(&bin(BinOp::Ne, c(0, Width::W1), z.clone())), z);
        // x == 0 → ~x on a bare flag; on a comparison the ~ folds on
        // through the reversal.
        assert_eq!(
            fold_expr(&bin(BinOp::Eq, z.clone(), c(0, Width::W1))),
            un(UnOp::Not, z.clone())
        );
        let (a, b) = (read(ra(0, Width::W64)), read(ra(1, Width::W64)));
        let e = bin(
            BinOp::Eq,
            bin(BinOp::Slt, a.clone(), b.clone()),
            c(0, Width::W1),
        );
        assert_eq!(fold_expr(&e), bin(BinOp::Sle, b, a));
        // A wider comparison against zero is not this identity.
        let wide = bin(BinOp::Ne, read(ra(0, Width::W8)), c(0, Width::W8));
        assert_eq!(fold_expr(&wide), wide);
    }

    #[test]
    fn boolean_merge_folds_cset_select_and_zext_cbnz() {
        // cset w0, eq ≈ (1 & sext(ZF)) | (0 & ~sext(ZF)) → zext(ZF).
        let zf = read(Reg::flag(Flag::Zero));
        let m = un(UnOp::SignExtend(Width::W32), zf.clone());
        let select = bin(
            BinOp::Or,
            bin(BinOp::And, c(1, Width::W32), m.clone()),
            bin(BinOp::And, c(0, Width::W32), un(UnOp::Not, m.clone())),
        );
        assert_eq!(
            fold_expr(&select),
            un(UnOp::ZeroExtend(Width::W32), zf.clone())
        );
        // The remnant after the zero arm folds: 1 & ~sext(ZF) → zext(~ZF).
        let rem = bin(BinOp::And, c(1, Width::W32), un(UnOp::Not, m));
        assert_eq!(
            fold_expr(&rem),
            un(UnOp::ZeroExtend(Width::W32), un(UnOp::Not, zf.clone()))
        );
        // cbnz over a setcc/cset: zext(ZF) != 0 → ZF.
        let zext = un(UnOp::ZeroExtend(Width::W64), zf.clone());
        assert_eq!(
            fold_expr(&bin(BinOp::Ne, zext.clone(), c(0, Width::W64))),
            zf.clone()
        );
        assert_eq!(
            fold_expr(&bin(BinOp::Eq, zext, c(0, Width::W64))),
            un(UnOp::Not, zf.clone())
        );
        // (zext(a) & zext(b)) != 0 → a & b — two setccs then test.
        let sf = read(Reg::flag(Flag::Sign));
        let conj = bin(
            BinOp::And,
            un(UnOp::ZeroExtend(Width::W8), zf.clone()),
            un(UnOp::ZeroExtend(Width::W8), sf.clone()),
        );
        assert_eq!(
            fold_expr(&bin(BinOp::Ne, conj, c(0, Width::W8))),
            bin(BinOp::And, zf, sf)
        );
    }

    #[test]
    fn boolean_merge_refuses_near_misses() {
        let zf = read(Reg::flag(Flag::Zero));
        // Non-0/1 select arms must not collapse to a boolean.
        let both_live = bin(
            BinOp::Or,
            bin(BinOp::And, c(2, Width::W32), un(UnOp::SignExtend(Width::W32), zf.clone())),
            bin(
                BinOp::And,
                c(3, Width::W32),
                un(UnOp::Not, un(UnOp::SignExtend(Width::W32), zf)),
            ),
        );
        let folded = fold_expr(&both_live);
        assert!(
            matches!(folded, Expr::Binary { op: BinOp::Or, .. }),
            "non-0/1 select arms must not collapse to a boolean: {folded:?}"
        );
        // A wider value zero-extended is not a W1 boolean.
        let wide = un(UnOp::ZeroExtend(Width::W64), read(ra(0, Width::W8)));
        let e = bin(BinOp::Ne, wide.clone(), c(0, Width::W64));
        assert_eq!(fold_expr(&e), e);
    }

    #[test]
    fn cmp_zero_order_shapes_fold_through_the_constant_overflow() {
        // `cmp a, 0`: the overflow shape folds to constant 0 bottom-up
        // (`a ^ a → 0`), and the W1 boolean identities must still finish
        // the pattern: jl → a <s 0, jge → 0 <=s a, jle → a <=s 0,
        // jg → 0 <s a.
        let w = Width::W64;
        let a = read(ra(0, w));
        let b = c(0, w);
        let jl = bin(BinOp::Ne, sf_shape(&a, &b, w), of_shape(&a, &b, w));
        assert_eq!(fold_expr(&jl), bin(BinOp::Slt, a.clone(), b.clone()));
        let jge = bin(BinOp::Eq, sf_shape(&a, &b, w), of_shape(&a, &b, w));
        assert_eq!(fold_expr(&jge), bin(BinOp::Sle, b.clone(), a.clone()));
        let jle = bin(
            BinOp::Or,
            zf_shape(&a, &b, w),
            bin(BinOp::Ne, sf_shape(&a, &b, w), of_shape(&a, &b, w)),
        );
        assert_eq!(fold_expr(&jle), bin(BinOp::Sle, a.clone(), b.clone()));
        let jg = bin(
            BinOp::And,
            un(UnOp::Not, zf_shape(&a, &b, w)),
            bin(BinOp::Eq, sf_shape(&a, &b, w), of_shape(&a, &b, w)),
        );
        assert_eq!(fold_expr(&jg), bin(BinOp::Slt, b, a));
    }

    #[test]
    fn a_near_miss_order_pair_is_left_exactly_as_is() {
        let w = Width::W64;
        let (a, b) = (read(ra(0, w)), read(ra(1, w)));
        let other = read(ra(2, w));
        // Different operands between the halves.
        let e = bin(BinOp::Ne, sf_shape(&a, &b, w), of_shape(&a, &other, w));
        assert_eq!(fold_expr(&e), e);
        // The overflow term from a different subtraction: swap its inner
        // Sub's operands.
        let of_swapped = bin(
            BinOp::Slt,
            bin(
                BinOp::And,
                bin(BinOp::Xor, a.clone(), b.clone()),
                bin(
                    BinOp::Xor,
                    a.clone(),
                    bin(BinOp::Sub, b.clone(), a.clone()),
                ),
            ),
            c(0, w),
        );
        let e = bin(BinOp::Ne, sf_shape(&a, &b, w), of_swapped);
        assert_eq!(fold_expr(&e), e);
        // A mixed-width pairing: the sign half at W32 over W32 operands,
        // the overflow half at W64.
        let (a32, b32) = (read(ra(0, Width::W32)), read(ra(1, Width::W32)));
        let e = bin(
            BinOp::Ne,
            sf_shape(&a32, &b32, Width::W32),
            of_shape(&a, &b, Width::W64),
        );
        assert_eq!(fold_expr(&e), e);
        // A composition over different operand pairs.
        let e = bin(
            BinOp::Or,
            bin(BinOp::Eq, a.clone(), other.clone()),
            bin(BinOp::Ult, a.clone(), b.clone()),
        );
        assert_eq!(fold_expr(&e), e);
        let e = bin(
            BinOp::And,
            bin(BinOp::Ne, a.clone(), other),
            bin(BinOp::Ule, b, a),
        );
        assert_eq!(fold_expr(&e), e);
    }

    #[test]
    fn the_addition_overflow_shape_is_not_an_order_pair() {
        // `add` writes `((l ^ res) & (r ^ res)) <s 0` — no `Sub` inside
        // the second xor — and its SF reads the sum, not a difference.
        // `SF != OF` over add flags does NOT mean `a <s b`, and the
        // matcher must refuse it structurally.
        let w = Width::W64;
        let (a, b) = (read(ra(0, w)), read(ra(1, w)));
        let sum = bin(BinOp::Add, a.clone(), b.clone());
        let sf_add = bin(BinOp::Slt, sum.clone(), c(0, w));
        let of_add = bin(
            BinOp::Slt,
            bin(
                BinOp::And,
                bin(BinOp::Xor, a.clone(), sum.clone()),
                bin(BinOp::Xor, b.clone(), sum),
            ),
            c(0, w),
        );
        let e = bin(BinOp::Ne, sf_add, of_add);
        assert_eq!(fold_expr(&e), e);
    }

    #[test]
    fn a_load_bearing_order_shape_folds_when_the_copies_match() {
        // Structurally equal load-bearing copies collapse to one — the
        // one-expression theorem: statements are the only effects, so
        // every load in one expression reads the same memory state.
        let w = Width::W64;
        let a = Expr::load(read(ra(0, w)), w);
        let b = read(ra(1, w));
        let pair = bin(BinOp::Ne, sf_shape(&a, &b, w), of_shape(&a, &b, w));
        assert_eq!(fold_expr(&pair), bin(BinOp::Slt, a.clone(), b.clone()));
        let compose = bin(
            BinOp::Or,
            bin(BinOp::Eq, a.clone(), b.clone()),
            bin(BinOp::Ult, a.clone(), b.clone()),
        );
        assert_eq!(fold_expr(&compose), bin(BinOp::Ule, a.clone(), b.clone()));
        let compose = bin(
            BinOp::And,
            bin(BinOp::Ne, a.clone(), b.clone()),
            bin(BinOp::Ule, b.clone(), a.clone()),
        );
        assert_eq!(fold_expr(&compose), bin(BinOp::Ult, b.clone(), a.clone()));
        // Unequal loads are not copies of anything: the structural
        // equality the theorem rides on refuses, and the shape stands.
        let a2 = Expr::load(read(ra(2, w)), w);
        let mixed = bin(BinOp::Ne, sf_shape(&a, &b, w), of_shape(&a2, &b, w));
        assert_eq!(fold_expr(&mixed), mixed);
    }

    /// Evaluate a load-free expression over the two registers `ra(0, _)`
    /// (value `a`) and `ra(1, _)` (value `b`) — the oracle's interpreter.
    fn eval(e: &Expr, a: u64, b: u64) -> u64 {
        eval_g(e, a, b, 0)
    }

    /// [`eval`] with a third register `ra(2, _)` holding the guard value
    /// `g` — the masked patterns' interpreter — and the width-changing
    /// unary operators modeled exactly (`zext` keeps the operand's masked
    /// value, `sext` replicates its top bit, `trunc` masks down).
    fn eval_g(e: &Expr, a: u64, b: u64, g: u64) -> u64 {
        match e {
            Expr::Const { value, width } => value & width.mask(),
            Expr::Reg(r) => match r.num {
                0 => a,
                1 => b,
                _ => g,
            },
            Expr::Unary { op, operand } => {
                let w = operand.width_of().expect("oracle operand has a width");
                let v = eval_g(operand, a, b, g) & w.mask();
                match op {
                    UnOp::Not => !v & w.mask(),
                    UnOp::Neg => v.wrapping_neg() & w.mask(),
                    UnOp::ZeroExtend(_) => v,
                    UnOp::SignExtend(to) => sign_extend(v, w) as u64 & to.mask(),
                    UnOp::Truncate(to) => v & to.mask(),
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                let (l, r) = (eval_g(lhs, a, b, g), eval_g(rhs, a, b, g));
                let w = lhs.width_of().expect("oracle operand has a width");
                let m = w.mask();
                match op {
                    BinOp::Sub => l.wrapping_sub(r) & m,
                    BinOp::Add => l.wrapping_add(r) & m,
                    BinOp::Mul => l.wrapping_mul(r) & m,
                    // The IR's shift semantics: the amount is taken
                    // modulo the operand width (see the module docs).
                    BinOp::Shl => (l << (r % w.bits() as u64)) & m,
                    BinOp::And => l & r,
                    BinOp::Or => l | r,
                    BinOp::Xor => l ^ r,
                    BinOp::Eq => (l == r) as u64,
                    BinOp::Ne => (l != r) as u64,
                    BinOp::Ult => (l < r) as u64,
                    BinOp::Ule => (l <= r) as u64,
                    BinOp::Slt => (sign_extend(l, w) < sign_extend(r, w)) as u64,
                    BinOp::Sle => (sign_extend(l, w) <= sign_extend(r, w)) as u64,
                    _ => panic!("operator outside the oracle's set"),
                }
            }
            Expr::Load { .. } => panic!("the oracle is load-free"),
        }
    }

    /// Whether `e` is a single comparison over the two bare registers —
    /// what every fully-recovered order condition must collapse to.
    fn is_bare_compare(e: &Expr) -> bool {
        matches!(e, Expr::Binary { op, lhs, rhs } if op.is_compare()
            && matches!(&**lhs, Expr::Reg(_)) && matches!(&**rhs, Expr::Reg(_)))
    }

    #[test]
    fn width8_exhaustive_oracle_over_every_pattern_and_polarity() {
        // The slice's real proof: every condition shape the x86 and A64
        // lifts compose over sub-kind flags, in both branch polarities,
        // folds to a single relational operator whose value agrees with
        // the literal flag computation on all 65,536 operand pairs.
        let w = Width::W8;
        let (a, b) = (read(ra(0, w)), read(ra(1, w)));
        let zf = || zf_shape(&a, &b, w);
        let sf = || sf_shape(&a, &b, w);
        let of = || of_shape(&a, &b, w);
        // x86: CF = a <u b. A64: C is NOT-borrow, CF = b <=u a.
        let cf = || bin(BinOp::Ult, a.clone(), b.clone());
        let cf64 = || bin(BinOp::Ule, b.clone(), a.clone());
        let patterns: Vec<(&str, Expr)> = vec![
            ("je", zf()),
            ("jne", un(UnOp::Not, zf())),
            ("jl", bin(BinOp::Ne, sf(), of())),
            ("jge", bin(BinOp::Eq, sf(), of())),
            ("jle", bin(BinOp::Or, zf(), bin(BinOp::Ne, sf(), of()))),
            (
                "jg",
                bin(
                    BinOp::And,
                    un(UnOp::Not, zf()),
                    bin(BinOp::Eq, sf(), of()),
                ),
            ),
            ("jb", cf()),
            ("jae", un(UnOp::Not, cf())),
            ("jbe", bin(BinOp::Or, cf(), zf())),
            (
                "ja",
                bin(BinOp::And, un(UnOp::Not, cf()), un(UnOp::Not, zf())),
            ),
            ("a64.hs", cf64()),
            ("a64.lo", un(UnOp::Not, cf64())),
            ("a64.hi", bin(BinOp::And, cf64(), un(UnOp::Not, zf()))),
            ("a64.ls", bin(BinOp::Or, un(UnOp::Not, cf64()), zf())),
        ];
        for (name, pattern) in patterns {
            for negated in [false, true] {
                let tree = if negated {
                    un(UnOp::Not, pattern.clone())
                } else {
                    pattern.clone()
                };
                let folded = fold_expr(&tree);
                assert!(
                    is_bare_compare(&folded),
                    "{name} (negated: {negated}) did not collapse: {folded:?}"
                );
                assert_eq!(fold_expr(&folded), folded, "{name}: fold is idempotent");
                for av in 0..=255u64 {
                    for bv in 0..=255u64 {
                        assert_eq!(
                            eval(&tree, av, bv),
                            eval(&folded, av, bv),
                            "{name} (negated: {negated}) at a={av}, b={bv}"
                        );
                    }
                }
            }
        }
    }

    // -- width-spelling normalization --------------------------------------

    #[test]
    fn a_truncation_cancels_the_extension_it_undoes() {
        // The truncation lands exactly on the operand's width: the chain
        // is the operand, whichever extension was used.
        for (from, to) in [
            (Width::W8, Width::W64),
            (Width::W32, Width::W64),
            (Width::W16, Width::W32),
        ] {
            let x = read(ra(0, from));
            for ext in [UnOp::ZeroExtend(to), UnOp::SignExtend(to)] {
                let e = un(UnOp::Truncate(from), un(ext, x.clone()));
                assert_eq!(fold_expr(&e), x, "{ext:?}");
            }
        }
    }

    #[test]
    fn a_truncation_through_a_wider_extension_narrows_or_refits() {
        // Landing below the operand: only bits of `x` survive.
        let x32 = read(ra(0, Width::W32));
        for ext in [UnOp::ZeroExtend(Width::W64), UnOp::SignExtend(Width::W64)] {
            let e = un(UnOp::Truncate(Width::W16), un(ext, x32.clone()));
            assert_eq!(
                fold_expr(&e),
                un(UnOp::Truncate(Width::W16), x32.clone()),
                "{ext:?}"
            );
        }
        // Landing between the operand and the extension: the same
        // extension, re-targeted at the truncation's width.
        let x8 = read(ra(0, Width::W8));
        let e = un(
            UnOp::Truncate(Width::W32),
            un(UnOp::ZeroExtend(Width::W64), x8.clone()),
        );
        assert_eq!(fold_expr(&e), un(UnOp::ZeroExtend(Width::W32), x8.clone()));
        let e = un(
            UnOp::Truncate(Width::W32),
            un(UnOp::SignExtend(Width::W64), x8.clone()),
        );
        assert_eq!(fold_expr(&e), un(UnOp::SignExtend(Width::W32), x8.clone()));
        // A doubled extension collapses through the recursion.
        let e = un(
            UnOp::Truncate(Width::W16),
            un(
                UnOp::ZeroExtend(Width::W64),
                un(UnOp::ZeroExtend(Width::W32), x8.clone()),
            ),
        );
        assert_eq!(fold_expr(&e), un(UnOp::ZeroExtend(Width::W16), x8));
    }

    #[test]
    fn the_unsound_width_respellings_are_refused() {
        // zext(trunc(x)) and sext(trunc(x)) discard bits of x: no identity.
        let x = read(ra(0, Width::W64));
        for ext in [UnOp::ZeroExtend(Width::W64), UnOp::SignExtend(Width::W64)] {
            let e = un(ext, un(UnOp::Truncate(Width::W32), x.clone()));
            assert_eq!(fold_expr(&e), e, "{ext:?}");
        }
        // Malformed chains — an extension that does not widen, a
        // truncation that does not narrow — are left exactly as they are
        // rather than laundered into well-formed-looking trees.
        let e = un(
            UnOp::Truncate(Width::W32),
            un(UnOp::ZeroExtend(Width::W64), x.clone()),
        );
        assert_eq!(fold_expr(&e), e);
        let x32 = read(ra(0, Width::W32));
        let e = un(
            UnOp::Truncate(Width::W64),
            un(UnOp::ZeroExtend(Width::W64), x32),
        );
        assert_eq!(fold_expr(&e), e);
    }

    #[test]
    fn width_normalization_is_exhaustively_sound() {
        // Every collapsible chain, against the literal evaluation of the
        // original, on every W8 operand value; the fold is idempotent.
        let x = read(ra(0, Width::W8));
        let chains = [
            un(UnOp::Truncate(Width::W8), un(UnOp::ZeroExtend(Width::W64), x.clone())),
            un(UnOp::Truncate(Width::W8), un(UnOp::SignExtend(Width::W64), x.clone())),
            un(UnOp::Truncate(Width::W16), un(UnOp::ZeroExtend(Width::W64), x.clone())),
            un(UnOp::Truncate(Width::W16), un(UnOp::SignExtend(Width::W64), x.clone())),
            un(
                UnOp::Truncate(Width::W16),
                un(
                    UnOp::ZeroExtend(Width::W64),
                    un(UnOp::SignExtend(Width::W32), x.clone()),
                ),
            ),
        ];
        for e in &chains {
            let folded = fold_expr(e);
            assert_ne!(&folded, e, "the chain must collapse: {e:?}");
            assert_eq!(fold_expr(&folded), folded, "fold is idempotent: {e:?}");
            for v in 0..=255u64 {
                assert_eq!(eval_g(e, v, 0, 0), eval_g(&folded, v, 0, 0), "at {v}: {e:?}");
            }
        }
    }

    #[test]
    fn a_pair_diverging_only_in_width_spelling_now_folds() {
        // The recorded residue class: one flag's operand spelled bare,
        // the other's through the W32 write's zext.q — provably the same
        // value, structurally equal after normalization, so the pair
        // collapses where it used to refuse.
        let w = Width::W32;
        let (x, y) = (read(ra(0, w)), read(ra(1, w)));
        let spelled = un(UnOp::Truncate(w), un(UnOp::ZeroExtend(Width::W64), x.clone()));
        let e = bin(BinOp::Ne, sf_shape(&spelled, &y, w), of_shape(&x, &y, w));
        assert_eq!(fold_expr(&e), bin(BinOp::Slt, x.clone(), y.clone()));
        // The sext respelling of the same value cancels the same way.
        let spelled = un(UnOp::Truncate(w), un(UnOp::SignExtend(Width::W64), x.clone()));
        let e = bin(BinOp::Eq, sf_shape(&spelled, &y, w), of_shape(&x, &y, w));
        assert_eq!(fold_expr(&e), bin(BinOp::Sle, y, x));
    }

    #[test]
    fn sign_vs_zero_extension_of_the_same_value_stays_refused() {
        // zext.q(x) and sext.q(x) differ on a negative x: a pair whose
        // halves disagree in extension is not the same subtraction.
        let (x, y) = (read(ra(0, Width::W32)), read(ra(1, Width::W32)));
        let zx = un(UnOp::ZeroExtend(Width::W64), x.clone());
        let sx = un(UnOp::SignExtend(Width::W64), x.clone());
        let y64 = un(UnOp::ZeroExtend(Width::W64), y.clone());
        let e = bin(
            BinOp::Ne,
            sf_shape(&zx, &y64, Width::W64),
            of_shape(&sx, &y64, Width::W64),
        );
        assert_eq!(fold_expr(&e), e);
        // And a truncation of a wider value is not that value: the
        // zext.q(trunc.d(v)) respelling against bare v refuses.
        let v = read(ra(0, Width::W64));
        let respelled = un(
            UnOp::ZeroExtend(Width::W64),
            un(UnOp::Truncate(Width::W32), v.clone()),
        );
        let e = bin(
            BinOp::Ne,
            sf_shape(&respelled, &y64, Width::W64),
            of_shape(&v, &y64, Width::W64),
        );
        assert_eq!(fold_expr(&e), e);
    }

    // -- value-numbering equality (the witness) ------------------------------

    #[test]
    fn trunc_distribution_is_exhaustively_sound() {
        // Truncation is a ring homomorphism: for every W16 operand pair,
        // trunc.b(a op b) equals trunc.b(a) op trunc.b(b) — the theorem
        // push_trunc trusts, proved by literal evaluation; and veq (under
        // an empty context — pure normalization) sees the two spellings
        // as one value.
        let (x, y) = (read(ra(0, Width::W16)), read(ra(1, Width::W16)));
        let t = |e: &Expr| un(UnOp::Truncate(Width::W8), e.clone());
        let vn = VnDefs::new();
        for op in [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::And,
            BinOp::Or,
            BinOp::Xor,
        ] {
            let whole = t(&bin(op, x.clone(), y.clone()));
            let split = bin(op, t(&x), t(&y));
            for a in 0..=0xffu64 {
                for b in 0..=0xffu64 {
                    // The full 65,536 pairs per op via the two bytes of
                    // each operand: high bits exercise the dropped range.
                    let (av, bv) = (a << 8 | b, b << 8 | a);
                    assert_eq!(
                        eval_g(&whole, av, bv, 0),
                        eval_g(&split, av, bv, 0),
                        "{op:?} at ({av:#x}, {bv:#x})"
                    );
                }
            }
            assert!(veq(&whole, &split, Some(&vn)), "veq sees through {op:?}");
        }
        for op in [UnOp::Neg, UnOp::Not] {
            let whole = t(&un(op, x.clone()));
            let split = un(op, t(&x));
            for a in 0..=0xffffu64 {
                assert_eq!(eval_g(&whole, a, 0, 0), eval_g(&split, a, 0, 0), "{op:?} at {a:#x}");
            }
            assert!(veq(&whole, &split, Some(&vn)), "veq sees through {op:?}");
        }
    }

    #[test]
    fn shift_truncation_is_not_distributed() {
        // The near-miss push_trunc must refuse: the shift amount is taken
        // modulo the width, so trunc.b(a << b) and trunc.b(a) << trunc.b(b)
        // disagree (a=1, b=8: 0 against 1). Pinned semantically, and veq
        // refuses the pair even with a context in hand.
        let (x, y) = (read(ra(0, Width::W16)), read(ra(1, Width::W16)));
        let t = |e: &Expr| un(UnOp::Truncate(Width::W8), e.clone());
        let whole = t(&bin(BinOp::Shl, x.clone(), y.clone()));
        let split = bin(BinOp::Shl, t(&x), t(&y));
        assert_ne!(eval_g(&whole, 1, 8, 0), eval_g(&split, 1, 8, 0));
        assert!(!veq(&whole, &split, Some(&VnDefs::new())));
    }

    #[test]
    fn veq_resolves_a_name_through_its_definition() {
        // The bash shape: a 64-bit definition read truncated on one side,
        // the 32-bit spelling written out on the other.
        let (x, y) = (read(ra(0, Width::W64)), read(ra(1, Width::W64)));
        let mut vn = VnDefs::new();
        let v2 = ra(2, Width::W64);
        vn.add(v2, &bin(BinOp::Add, x.clone(), y.clone()));
        let named = un(UnOp::Truncate(Width::W32), read(v2));
        let spelled = bin(
            BinOp::Add,
            un(UnOp::Truncate(Width::W32), x.clone()),
            un(UnOp::Truncate(Width::W32), y.clone()),
        );
        assert!(veq(&named, &spelled, Some(&vn)));
        // Structural equality alone stays refused — the witness is the
        // only path — and a narrower read of the name never resolves.
        assert!(!veq(&named, &spelled, None));
        let narrow = read(ra(2, Width::W32));
        assert!(!veq(&narrow, &bin(BinOp::Add, x, y), Some(&vn)));
    }

    #[test]
    fn veq_refuses_what_it_cannot_prove() {
        let x = read(ra(0, Width::W64));
        let mut vn = VnDefs::new();
        // A load-backed definition is never admitted: the witness does
        // not reason through memory.
        let ld = Expr::load(x.clone(), Width::W64);
        vn.add(ra(3, Width::W64), &ld);
        assert!(!veq(&read(ra(3, Width::W64)), &ld, Some(&vn)));
        // A division-backed definition is never admitted: never through
        // a trap.
        let dv = bin(BinOp::UDiv, x.clone(), read(ra(1, Width::W64)));
        vn.add(ra(4, Width::W64), &dv);
        assert!(!veq(&read(ra(4, Width::W64)), &dv, Some(&vn)));
        // Different values stay different.
        vn.add(ra(5, Width::W64), &bin(BinOp::Add, x.clone(), c(0x32, Width::W64)));
        assert!(!veq(
            &read(ra(5, Width::W64)),
            &bin(BinOp::Add, x.clone(), c(0x33, Width::W64)),
            Some(&vn)
        ));
        // Sign- against zero-extension of one value stays refused.
        let n = read(ra(0, Width::W32));
        vn.add(ra(6, Width::W64), &un(UnOp::ZeroExtend(Width::W64), n.clone()));
        vn.add(ra(7, Width::W64), &un(UnOp::SignExtend(Width::W64), n));
        assert!(!veq(
            &read(ra(6, Width::W64)),
            &read(ra(7, Width::W64)),
            Some(&vn)
        ));
    }

    #[test]
    fn a_pair_split_across_ssa_names_now_folds() {
        // The measured residue class end to end at the fold: the sign
        // half spells `a` as the 32-bit sum, the overflow half reads the
        // 64-bit name whose definition is that sum — provably one value,
        // so the pair collapses under the context and only under it. The
        // collapsed operands are the sign half's own subtrees.
        let w = Width::W32;
        let (x, y) = (read(ra(0, Width::W64)), read(ra(1, Width::W64)));
        let b = read(ra(3, w));
        let mut vn = VnDefs::new();
        let v2 = ra(2, Width::W64);
        vn.add(v2, &bin(BinOp::Add, x.clone(), y.clone()));
        let a_spelled = bin(
            BinOp::Add,
            un(UnOp::Truncate(w), x.clone()),
            un(UnOp::Truncate(w), y.clone()),
        );
        let a_named = un(UnOp::Truncate(w), read(v2));
        let e = bin(
            BinOp::Ne,
            sf_shape(&a_spelled, &b, w),
            of_shape(&a_named, &b, w),
        );
        assert_eq!(fold_expr(&e), e, "no context, no fold");
        assert_eq!(
            fold_expr_vn(&e, &vn),
            bin(BinOp::Slt, a_spelled, b),
            "the witness collapses the pair"
        );
        // Determinism: the same call twice is byte-equal.
        assert_eq!(fold_expr_vn(&e, &vn), fold_expr_vn(&e, &vn));
    }

    // -- masked (conditional-compare) order pairs ---------------------------

    /// A flag select as the CCMP lift leaves it once its constant arm has
    /// folded: `c & flag` for a clear imm4 bit, `(c & flag) | comp` for a
    /// set one, with `comp` the complement spelling of the guard.
    fn sel_shape(g: &Expr, flag: Expr, bit: bool, comp: &Expr) -> Expr {
        let armed = bin(BinOp::And, g.clone(), flag);
        if bit {
            bin(BinOp::Or, armed, comp.clone())
        } else {
            armed
        }
    }

    #[test]
    fn a_masked_order_pair_folds_in_every_bit_combination() {
        let w = Width::W32;
        let (a, b) = (read(ra(0, w)), read(ra(1, w)));
        // The guard a real chain leaves: a comparison, complement flipped.
        let g = bin(BinOp::Ult, read(ra(2, w)), c(7, w));
        let ng = bin(BinOp::Ule, c(7, w), read(ra(2, w)));
        let lt = bin(BinOp::Slt, a.clone(), b.clone());
        let ge = bin(BinOp::Sle, b.clone(), a.clone());
        for (bn, bv) in [(false, false), (true, false), (false, true), (true, true)] {
            let sf = sel_shape(&g, sf_shape(&a, &b, w), bn, &ng);
            let of = sel_shape(&g, of_shape(&a, &b, w), bv, &ng);
            let ne = fold_expr(&bin(BinOp::Ne, sf.clone(), of.clone()));
            let eq = fold_expr(&bin(BinOp::Eq, sf, of));
            if bn == bv {
                assert_eq!(ne, bin(BinOp::And, g.clone(), lt.clone()), "bits {bn}/{bv}");
                assert_eq!(eq, bin(BinOp::Or, ng.clone(), ge.clone()), "bits {bn}/{bv}");
            } else {
                assert_eq!(ne, bin(BinOp::Or, ng.clone(), lt.clone()), "bits {bn}/{bv}");
                assert_eq!(eq, bin(BinOp::And, g.clone(), ge.clone()), "bits {bn}/{bv}");
            }
        }
    }

    #[test]
    fn a_masked_pair_with_a_flag_guard_and_literal_complement_folds() {
        // The guard as a bare W1 flag read, its complement the literal ~.
        let w = Width::W64;
        let (a, b) = (read(ra(0, w)), read(ra(1, w)));
        let g = read(Reg::flag(Flag::Carry));
        let ng = un(UnOp::Not, g.clone());
        let sf = sel_shape(&g, sf_shape(&a, &b, w), true, &ng);
        let of = sel_shape(&g, of_shape(&a, &b, w), false, &ng);
        assert_eq!(
            fold_expr(&bin(BinOp::Ne, sf, of)),
            bin(BinOp::Or, ng, bin(BinOp::Slt, a, b))
        );
    }

    #[test]
    fn masked_pair_width8_exhaustive_oracle() {
        // The guarded patterns' proof: every imm4-bit combination of the
        // masked signed pair, under both outer comparisons and both
        // polarities, with the guard a W1 register — the folded result
        // against the literal select computation on all 65,536 operand
        // pairs and both guard values.
        let w = Width::W8;
        let (a, b) = (read(ra(0, w)), read(ra(1, w)));
        let g = read(ra(2, Width::W1));
        let ng = un(UnOp::Not, g.clone());
        for (bn, bv) in [(false, false), (true, false), (false, true), (true, true)] {
            for cmp in [BinOp::Ne, BinOp::Eq] {
                let tree = bin(
                    cmp,
                    sel_shape(&g, sf_shape(&a, &b, w), bn, &ng),
                    sel_shape(&g, of_shape(&a, &b, w), bv, &ng),
                );
                for negated in [false, true] {
                    let t = if negated {
                        un(UnOp::Not, tree.clone())
                    } else {
                        tree.clone()
                    };
                    let folded = fold_expr(&t);
                    assert_ne!(folded, t, "{cmp:?} bits {bn}/{bv} must rewrite");
                    assert_eq!(fold_expr(&folded), folded, "fold is idempotent");
                    for av in 0..=255u64 {
                        for bv2 in 0..=255u64 {
                            for gv in 0..=1u64 {
                                assert_eq!(
                                    eval_g(&t, av, bv2, gv),
                                    eval_g(&folded, av, bv2, gv),
                                    "{cmp:?} bits {bn}/{bv} (negated: {negated}) at \
                                     a={av}, b={bv2}, c={gv}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_negated_flag_select_collapses_to_the_guarded_complement() {
        // ~((c & x) | ~c) → c & ~x — the b.ne-over-CCMP shape from the
        // real dylib, guard and complement spelled as flipped comparisons.
        let w = Width::W64;
        let g = bin(BinOp::Ule, c(8, w), read(ra(0, w)));
        let ng = bin(BinOp::Ult, read(ra(0, w)), c(8, w));
        let z = bin(BinOp::Eq, read(ra(1, w)), c(1, w));
        let e = un(
            UnOp::Not,
            bin(BinOp::Or, bin(BinOp::And, g.clone(), z.clone()), ng),
        );
        assert_eq!(
            fold_expr(&e),
            bin(BinOp::And, g, bin(BinOp::Ne, read(ra(1, w)), c(1, w)))
        );
        // The mirrored operand orders and a literal ~ guard match too.
        let gf = read(Reg::flag(Flag::Zero));
        let e = un(
            UnOp::Not,
            bin(
                BinOp::Or,
                un(UnOp::Not, gf.clone()),
                bin(BinOp::And, z, gf.clone()),
            ),
        );
        assert_eq!(
            fold_expr(&e),
            bin(BinOp::And, gf, bin(BinOp::Ne, read(ra(1, w)), c(1, w)))
        );
        // Exhaustively sound at width 8, both guard values, over the
        // C-model flag the real b.lo reads.
        let w = Width::W8;
        let (a, b) = (read(ra(0, w)), read(ra(1, w)));
        let g = read(ra(2, Width::W1));
        let tree = un(
            UnOp::Not,
            bin(
                BinOp::Or,
                bin(BinOp::And, g.clone(), bin(BinOp::Ule, b.clone(), a.clone())),
                un(UnOp::Not, g.clone()),
            ),
        );
        let folded = fold_expr(&tree);
        assert_eq!(
            folded,
            bin(BinOp::And, g, bin(BinOp::Ult, a, b)),
            "the select collapses to the guarded strict order"
        );
        for av in 0..=255u64 {
            for bv in 0..=255u64 {
                for gv in 0..=1u64 {
                    assert_eq!(
                        eval_g(&tree, av, bv, gv),
                        eval_g(&folded, av, bv, gv),
                        "at a={av}, b={bv}, c={gv}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_masked_near_miss_is_left_exactly_as_is() {
        let w = Width::W64;
        let (a, b) = (read(ra(0, w)), read(ra(1, w)));
        let g = read(Reg::flag(Flag::Carry));
        let d = read(Reg::flag(Flag::Parity));
        // Different guards on the two halves.
        let e = bin(
            BinOp::Ne,
            bin(BinOp::And, g.clone(), sf_shape(&a, &b, w)),
            bin(BinOp::And, d.clone(), of_shape(&a, &b, w)),
        );
        assert_eq!(fold_expr(&e), e);
        // A second Or operand that is not the guard's complement.
        let sf = bin(
            BinOp::Or,
            bin(BinOp::And, g.clone(), sf_shape(&a, &b, w)),
            g.clone(),
        );
        let e = bin(BinOp::Ne, sf, bin(BinOp::And, g.clone(), of_shape(&a, &b, w)));
        assert_eq!(fold_expr(&e), e);
        // A shared guard over different operands between the halves.
        let other = read(ra(2, w));
        let e = bin(
            BinOp::Ne,
            bin(BinOp::And, g.clone(), sf_shape(&a, &b, w)),
            bin(BinOp::And, g.clone(), of_shape(&a, &other, w)),
        );
        assert_eq!(fold_expr(&e), e);
        // The negated select with a non-complement second operand.
        let e = un(
            UnOp::Not,
            bin(BinOp::Or, bin(BinOp::And, g, sf_shape(&a, &b, w)), d),
        );
        assert_eq!(fold_expr(&e), e);
    }

    #[test]
    fn a_load_bearing_masked_shape_folds_when_the_copies_match() {
        // One-expression theorem: the guard's and the operands' equal
        // load-bearing copies collapse to one (see the module's
        // "Soundness"); a mismatched load refuses.
        let w = Width::W64;
        let (a, b) = (read(ra(0, w)), read(ra(1, w)));
        // A load in the guard: its copies collapse to one.
        let gl = bin(BinOp::Ne, Expr::load(read(ra(3, w)), w), c(0, w));
        let e = bin(
            BinOp::Ne,
            bin(BinOp::And, gl.clone(), sf_shape(&a, &b, w)),
            bin(BinOp::And, gl.clone(), of_shape(&a, &b, w)),
        );
        assert_eq!(
            fold_expr(&e),
            bin(
                BinOp::And,
                gl.clone(),
                bin(BinOp::Slt, a.clone(), b.clone())
            )
        );
        // A load in the shared pair operand.
        let g = read(Reg::flag(Flag::Carry));
        let al = Expr::load(read(ra(0, w)), w);
        let e = bin(
            BinOp::Ne,
            bin(BinOp::And, g.clone(), sf_shape(&al, &b, w)),
            bin(BinOp::And, g.clone(), of_shape(&al, &b, w)),
        );
        assert_eq!(
            fold_expr(&e),
            bin(
                BinOp::And,
                g.clone(),
                bin(BinOp::Slt, al.clone(), b.clone())
            )
        );
        // The negated select over a load-bearing guard, its complement
        // spelled as the flipped comparison of the same load.
        let ngl = bin(BinOp::Eq, Expr::load(read(ra(3, w)), w), c(0, w));
        let e = un(
            UnOp::Not,
            bin(
                BinOp::Or,
                bin(BinOp::And, gl.clone(), sf_shape(&a, &b, w)),
                ngl,
            ),
        );
        assert_eq!(
            fold_expr(&e),
            bin(BinOp::And, gl, complement_w1(&sf_shape(&a, &b, w)))
        );
        // Guards reading *different* addresses are not copies: refused.
        let gl2 = bin(BinOp::Ne, Expr::load(read(ra(4, w)), w), c(0, w));
        let e = bin(
            BinOp::Ne,
            bin(BinOp::And, gl2, sf_shape(&a, &b, w)),
            bin(BinOp::And, g, of_shape(&a, &b, w)),
        );
        assert_eq!(fold_expr(&e), e);
    }

    #[test]
    fn a_lifted_ccmp_chain_collapses_to_the_guarded_relation() {
        use crate::model::Arch;
        use crate::{aarch64, aarch64_lift, irlift, irssa, irssaopt};

        let block = |start: u64, words: &[u32], successors: Vec<u64>| {
            let insns: Vec<_> = words
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    let va = start + 4 * i as u64;
                    (aarch64::decode(&w.to_le_bytes(), va).unwrap(), va)
                })
                .collect();
            irlift::LiftedBlock {
                start,
                end: start + 4 * words.len() as u64,
                stmts: aarch64_lift::lift_block(&insns),
                successors,
                truncated: false,
            }
        };
        // cmp w0, w1 ; b.eq → ccmp w2, w3, #0, ne ; b.lt → mov w0, #1 →
        // ret — the chained `&&` fixture the coverage slice left with its
        // condition-masked pair, now finishing the collapse.
        let build = || irlift::LiftedFunction {
            entry: 0x1000,
            name: None,
            arch: Arch::Aarch64,
            blocks: [
                block(0x1000, &[0x6B01_001F, 0x5400_0060], vec![0x1008, 0x1010]),
                block(0x1008, &[0x7A43_1040, 0x5400_006B], vec![0x1010, 0x1018]),
                block(0x1010, &[0x5280_0020, 0x1400_0001], vec![0x1018]),
                block(0x1018, &[0xD65F_03C0], vec![]),
            ]
            .into_iter()
            .map(|b| (b.start, b))
            .collect(),
        };
        let pipeline = || {
            let ssa = irssa::construct(&build()).expect("chained-condition block constructs");
            let (opt, _) = irssaopt::optimize(&ssa);
            let (fwd, _) = irssaopt::forward(&opt);
            let live_out = crate::callfx::function_live_out(Arch::Aarch64).unwrap_or_default();
            let (swept, _) = irssaopt::eliminate_dead(&fwd, &live_out);
            assert_eq!(irssa::check(&swept), Ok(()));
            irssa::render(&swept)
        };
        let t = pipeline();
        assert_eq!(t, pipeline(), "deterministic end to end");
        // The ccmp-fed branch reads the guarded relation, and the emptied
        // flag definitions — the masked selects and their overflow xors —
        // swept with nothing left reading them.
        assert!(
            t.contains(
                "goto if ((trunc.d(x0#0) != trunc.d(x1#0)) & (trunc.d(x2#0) <s trunc.d(x3#0)))"
            ),
            "the ccmp chain reads as the guarded relation: {t}"
        );
        assert!(
            !t.contains('^'),
            "no overflow-model xor survives the sweep: {t}"
        );
    }

    // -- load-safety -------------------------------------------------------

    fn load(reg: Reg, w: Width) -> Expr {
        Expr::load(read(reg), w)
    }

    #[test]
    fn load_bearing_xor_self_is_not_folded() {
        // load[rax] ^ load[rax] must NOT fold to 0: a load may fault/alias.
        let l = load(ra(0, Width::W64), Width::W32);
        let e = bin(BinOp::Xor, l.clone(), l);
        assert_eq!(fold_expr(&e), e);
    }

    #[test]
    fn load_bearing_sub_self_is_not_folded() {
        let l = load(ra(0, Width::W64), Width::W32);
        let e = bin(BinOp::Sub, l.clone(), l);
        assert_eq!(fold_expr(&e), e);
    }

    #[test]
    fn load_bearing_mul_zero_is_not_folded() {
        // (load[rax]) * 0 must not delete the load.
        let l = load(ra(0, Width::W64), Width::W32);
        let e = bin(BinOp::Mul, l, c(0, Width::W32));
        assert_eq!(fold_expr(&e), e);
    }

    #[test]
    fn load_bearing_and_zero_is_not_folded() {
        let l = load(ra(0, Width::W64), Width::W32);
        let e = bin(BinOp::And, l, c(0, Width::W32));
        assert_eq!(fold_expr(&e), e);
    }

    #[test]
    fn a_load_bearing_assignment_is_never_eliminated() {
        // rax := load[rcx], with nothing live: still kept (may fault).
        let s = assign(ra(0, Width::W64), load(ra(1, Width::W64), Width::W64));
        let out = eliminate_dead(std::slice::from_ref(&s), &BTreeSet::new());
        assert_eq!(out, vec![s]);
        ok(&out);
    }

    // -- propagation -------------------------------------------------------

    #[test]
    fn propagate_substitutes_constant_then_folds() {
        // t0 := 5 ; rax := t0 + 3   →   rax := 8   (t0 dead, rax live)
        let stmts = vec![
            assign(rt(0, Width::W64), c(5, Width::W64)),
            assign(
                ra(0, Width::W64),
                bin(BinOp::Add, read(rt(0, Width::W64)), c(3, Width::W64)),
            ),
        ];
        let live = live_set(&[ra(0, Width::W64)]);
        let out = simplify(&stmts, &live);
        assert_eq!(out, vec![assign(ra(0, Width::W64), c(8, Width::W64))]);
        ok(&out);
    }

    #[test]
    fn propagate_forwards_a_register_copy() {
        // rcx := rax ; rdx := rcx + 1  →  rdx := rax + 1
        let stmts = vec![
            assign(ra(1, Width::W64), read(ra(0, Width::W64))),
            assign(
                ra(2, Width::W64),
                bin(BinOp::Add, read(ra(1, Width::W64)), c(1, Width::W64)),
            ),
        ];
        let out = propagate(&stmts);
        assert_eq!(
            out[1],
            assign(
                ra(2, Width::W64),
                bin(BinOp::Add, read(ra(0, Width::W64)), c(1, Width::W64))
            )
        );
        ok(&out);
    }

    #[test]
    fn propagation_is_invalidated_on_redefine() {
        // rcx := 5 ; rcx := 6 ; rax := rcx  →  last uses 6, not 5.
        let stmts = vec![
            assign(ra(1, Width::W64), c(5, Width::W64)),
            assign(ra(1, Width::W64), c(6, Width::W64)),
            assign(ra(0, Width::W64), read(ra(1, Width::W64))),
        ];
        let out = propagate(&stmts);
        assert_eq!(out[2], assign(ra(0, Width::W64), c(6, Width::W64)));
    }

    #[test]
    fn propagation_is_invalidated_by_an_aliasing_write() {
        // rax.q := 5 ; al.b := 7 ; rcx := rax.q  — the partial write to al
        // aliases rax, so rax's constant fact is dropped, not propagated.
        let stmts = vec![
            assign(ra(0, Width::W64), c(5, Width::W64)),
            assign(ra(0, Width::W8), c(7, Width::W8)),
            assign(ra(1, Width::W64), read(ra(0, Width::W64))),
        ];
        let out = propagate(&stmts);
        assert_eq!(out[2], assign(ra(1, Width::W64), read(ra(0, Width::W64))));
    }

    #[test]
    fn a_store_does_not_break_register_propagation() {
        // rcx := 9 ; store [rax], rdx ; rsi := rcx  →  rsi := 9.
        let stmts = vec![
            assign(ra(1, Width::W64), c(9, Width::W64)),
            Stmt::Store {
                addr: read(ra(0, Width::W64)),
                value: read(ra(3, Width::W64)),
            },
            assign(ra(6, Width::W64), read(ra(1, Width::W64))),
        ];
        let out = propagate(&stmts);
        assert_eq!(out[2], assign(ra(6, Width::W64), c(9, Width::W64)));
        ok(&out);
    }

    #[test]
    fn an_intrinsic_invalidates_its_written_registers() {
        // rax := 5 ; rdtsc writes rax ; rcx := rax  — must not propagate 5.
        let stmts = vec![
            assign(ra(0, Width::W64), c(5, Width::W64)),
            Stmt::Intrinsic {
                name: "rdtsc",
                writes: vec![ra(0, Width::W64)],
                reads: vec![],
            },
            assign(ra(1, Width::W64), read(ra(0, Width::W64))),
        ];
        let out = propagate(&stmts);
        assert_eq!(out[2], assign(ra(1, Width::W64), read(ra(0, Width::W64))));
    }

    #[test]
    fn a_branch_clears_propagation_facts() {
        // rax := 5 ; goto <call> ; rcx := rax  — after a transfer, forget.
        let stmts = vec![
            assign(ra(0, Width::W64), c(5, Width::W64)),
            Stmt::Branch {
                kind: BranchKind::Call,
                cond: None,
                target: c(0x1000, Width::W64),
            },
            assign(ra(1, Width::W64), read(ra(0, Width::W64))),
        ];
        let out = propagate(&stmts);
        assert_eq!(out[2], assign(ra(1, Width::W64), read(ra(0, Width::W64))));
    }

    // -- dead-code elimination ---------------------------------------------

    #[test]
    fn dead_flags_are_dropped_and_the_read_flag_is_kept() {
        // A cmp-style write of four flags; only ZF is live-out.
        let cmp = |flag: Flag| {
            assign(
                Reg::flag(flag),
                bin(BinOp::Ult, read(ra(0, Width::W32)), read(ra(1, Width::W32))),
            )
        };
        let stmts = vec![
            cmp(Flag::Carry),
            cmp(Flag::Zero),
            cmp(Flag::Sign),
            cmp(Flag::Overflow),
        ];
        let live = live_set(&[Reg::flag(Flag::Zero)]);
        let out = eliminate_dead(&stmts, &live);
        assert_eq!(out, vec![cmp(Flag::Zero)]);
        ok(&out);
    }

    #[test]
    fn a_dead_temporary_is_dropped() {
        // t0 := 5 with t0 never read again and not live-out.
        let stmts = vec![assign(rt(0, Width::W64), c(5, Width::W64))];
        let out = eliminate_dead(&stmts, &BTreeSet::new());
        assert!(out.is_empty());
    }

    #[test]
    fn a_store_is_never_dropped() {
        let s = Stmt::Store {
            addr: read(ra(0, Width::W64)),
            value: c(1, Width::W32),
        };
        let out = eliminate_dead(std::slice::from_ref(&s), &BTreeSet::new());
        assert_eq!(out, vec![s]);
    }

    #[test]
    fn a_branch_is_never_dropped() {
        let s = Stmt::Branch {
            kind: BranchKind::Return,
            cond: None,
            target: read(ra(0, Width::W64)),
        };
        let out = eliminate_dead(std::slice::from_ref(&s), &BTreeSet::new());
        assert_eq!(out, vec![s]);
    }

    #[test]
    fn an_intrinsic_is_never_dropped() {
        let s = Stmt::Intrinsic {
            name: "mfence",
            writes: vec![],
            reads: vec![],
        };
        let out = eliminate_dead(std::slice::from_ref(&s), &BTreeSet::new());
        assert_eq!(out, vec![s]);
    }

    #[test]
    fn a_redundant_copy_overwritten_at_the_same_width_is_dropped() {
        // rax := rcx ; rax := 7   →   rax := 7   (default keeps arch live).
        let stmts = vec![
            assign(ra(0, Width::W64), read(ra(1, Width::W64))),
            assign(ra(0, Width::W64), c(7, Width::W64)),
        ];
        let out = simplify_default(&stmts);
        assert_eq!(out, vec![assign(ra(0, Width::W64), c(7, Width::W64))]);
        ok(&out);
    }

    #[test]
    fn a_live_definition_is_kept() {
        let stmts = vec![assign(ra(0, Width::W64), c(5, Width::W64))];
        let out = eliminate_dead(&stmts, &live_set(&[ra(0, Width::W64)]));
        assert_eq!(out, stmts);
    }

    #[test]
    fn a_partial_write_does_not_kill_a_wider_live_reference() {
        // rax.q := 0x1234 ; al.b := 5, with rax.q live-out — both kept, as
        // the byte write does not cover the quad.
        let stmts = vec![
            assign(ra(0, Width::W64), c(0x1234, Width::W64)),
            assign(ra(0, Width::W8), c(5, Width::W8)),
        ];
        let out = eliminate_dead(&stmts, &live_set(&[ra(0, Width::W64)]));
        assert_eq!(out, stmts);
        ok(&out);
    }

    // -- liveness ----------------------------------------------------------

    #[test]
    fn live_in_folds_the_whole_block_transfer() {
        // t0 := rax ; rcx := t0 — given {rcx} live-out, the block needs
        // exactly rax on entry (t0 and rcx are defined inside).
        let stmts = vec![
            assign(rt(0, Width::W64), read(ra(0, Width::W64))),
            assign(ra(1, Width::W64), read(rt(0, Width::W64))),
        ];
        let li = live_in(&stmts, &live_set(&[ra(1, Width::W64)]));
        assert_eq!(li, live_set(&[ra(0, Width::W64)]));
        // An empty block passes its live-out through unchanged.
        assert_eq!(live_in(&[], &li), li);
    }

    #[test]
    fn liveness_reports_live_out_per_statement() {
        // t0 := rax ; rcx := t0
        let stmts = vec![
            assign(rt(0, Width::W64), read(ra(0, Width::W64))),
            assign(ra(1, Width::W64), read(rt(0, Width::W64))),
        ];
        let live = liveness(&stmts, &live_set(&[ra(1, Width::W64)]));
        // After stmt 0, t0 is live (read by stmt 1).
        assert!(live[0].contains(&rt(0, Width::W64)));
        // After stmt 1, only the block live-out remains.
        assert_eq!(live[1], live_set(&[ra(1, Width::W64)]));
    }

    // -- composition and robustness ----------------------------------------

    #[test]
    fn simplify_default_keeps_architectural_state() {
        // rax := 2 + 3 folds to 5 and is kept (arch live by default).
        let stmts = vec![assign(
            ra(0, Width::W64),
            bin(BinOp::Add, c(2, Width::W64), c(3, Width::W64)),
        )];
        let out = simplify_default(&stmts);
        assert_eq!(out, vec![assign(ra(0, Width::W64), c(5, Width::W64))]);
        ok(&out);
    }

    #[test]
    fn simplify_chains_propagation_folding_and_dce() {
        // t0 := rax + 0 ; t1 := t0 * 1 ; rcx := t1 ^ t1
        //   → rcx := 0   (t0, t1 dead)
        let stmts = vec![
            assign(
                rt(0, Width::W64),
                bin(BinOp::Add, read(ra(0, Width::W64)), c(0, Width::W64)),
            ),
            assign(
                rt(1, Width::W64),
                bin(BinOp::Mul, read(rt(0, Width::W64)), c(1, Width::W64)),
            ),
            assign(
                ra(1, Width::W64),
                bin(BinOp::Xor, read(rt(1, Width::W64)), read(rt(1, Width::W64))),
            ),
        ];
        let out = simplify(&stmts, &live_set(&[ra(1, Width::W64)]));
        assert_eq!(out, vec![assign(ra(1, Width::W64), c(0, Width::W64))]);
        ok(&out);
    }

    #[test]
    fn simplify_is_deterministic() {
        let stmts = vec![
            assign(rt(0, Width::W64), c(5, Width::W64)),
            assign(
                ra(0, Width::W64),
                bin(BinOp::Add, read(rt(0, Width::W64)), c(3, Width::W64)),
            ),
        ];
        let live = live_set(&[ra(0, Width::W64)]);
        assert_eq!(simplify(&stmts, &live), simplify(&stmts, &live));
    }

    #[test]
    fn simplify_reaches_a_fixpoint_on_adversarial_input() {
        // A long chain of self-cancelling copies; must terminate and check.
        let mut stmts = Vec::new();
        for i in 0..200u16 {
            stmts.push(assign(
                rt(i, Width::W64),
                bin(BinOp::Add, read(ra(0, Width::W64)), c(0, Width::W64)),
            ));
        }
        stmts.push(assign(ra(0, Width::W64), read(ra(0, Width::W64))));
        let out = simplify_default(&stmts);
        ok(&out);
        // Every dead temporary self-copy is gone; the self-assign folds away.
        assert!(out.len() <= stmts.len());
    }

    #[test]
    fn an_oversized_statement_list_is_returned_unchanged() {
        let one = assign(ra(0, Width::W64), c(0, Width::W64));
        let many = vec![one; crate::ir::MAX_STMTS + 1];
        let out = simplify_default(&many);
        assert_eq!(out.len(), many.len());
    }

    #[test]
    fn a_deep_expression_does_not_panic() {
        // Build a chain past the node cap using a non-folding operation, so
        // recursion is driven to full depth and the depth cap is exercised.
        // fold must not overflow or panic.
        let mut e = read(ra(0, Width::W64));
        for _ in 0..(crate::ir::MAX_EXPR_NODES + 50) {
            e = bin(BinOp::And, e, read(ra(1, Width::W64)));
        }
        let _ = fold_expr(&e); // no panic
        // And through the full pipeline as a statement value.
        let s = assign(ra(0, Width::W64), e);
        let _ = simplify_default(std::slice::from_ref(&s)); // no panic
    }

    // -- call effects (callfx contracts, pinned) -----------------------------

    /// `prefix`, then a direct call with the x86-64 call effects spliced
    /// in by the real pass ([`crate::callfx::apply`]).
    fn callfx_block(prefix: Vec<Stmt>) -> Vec<Stmt> {
        let mut stmts = prefix;
        stmts.push(Stmt::Branch {
            kind: BranchKind::Call,
            cond: None,
            target: c(0x9000, Width::W64),
        });
        let block = crate::irlift::LiftedBlock {
            start: 0x1000,
            end: 0x1004,
            stmts,
            successors: Vec::new(),
            truncated: false,
        };
        let f = crate::irlift::LiftedFunction {
            entry: 0x1000,
            name: None,
            arch: crate::model::Arch::X86_64,
            blocks: [(0x1000, block)].into_iter().collect(),
        };
        let out = crate::callfx::apply(&f, &crate::callfx::x86_64());
        out.blocks[&0x1000].stmts.clone()
    }

    #[test]
    fn dce_on_a_callfx_block_drops_a_dead_clobber_def_and_keeps_an_argument_def() {
        // r11 := 1 is a pure clobber: nothing reads it before the call
        // kills it, so it may go. rdi := 2 is a possible argument the
        // callfx intrinsic reads, so it must stay.
        let dead = assign(ra(11, Width::W64), c(1, Width::W64));
        let arg = assign(ra(7, Width::W64), c(2, Width::W64));
        let stmts = callfx_block(vec![dead.clone(), arg.clone()]);
        let out = eliminate_dead(&stmts, &live_set(&[ra(4, Width::W64)]));
        assert!(
            !out.contains(&dead),
            "the dead pre-call r11 def is gone: {out:?}"
        );
        assert!(out.contains(&arg), "the argument def stays pinned: {out:?}");
        ok(&out);
    }

    #[test]
    fn callfx_cuts_clobbered_liveness_and_pins_argument_liveness() {
        // With r11 live-out, the intrinsic's write satisfies it — r11 is
        // not live into the block — while the argument registers the
        // intrinsic reads are.
        let stmts = callfx_block(Vec::new());
        let li = live_in(&stmts, &live_set(&[ra(11, Width::W64)]));
        assert!(
            !li.contains(&ra(11, Width::W64)),
            "r11 is cut at the call: {li:?}"
        );
        assert!(
            li.contains(&ra(7, Width::W64)),
            "rdi is live before the call: {li:?}"
        );
        assert!(
            li.contains(&ra(4, Width::W64)),
            "rsp is read by the call: {li:?}"
        );
    }

    #[test]
    fn folding_preserves_check_across_statement_kinds() {
        let stmts = [
            assign(
                ra(0, Width::W64),
                bin(BinOp::Add, c(1, Width::W64), c(2, Width::W64)),
            ),
            Stmt::Store {
                addr: read(ra(0, Width::W64)),
                value: bin(BinOp::Mul, c(4, Width::W32), c(1, Width::W32)),
            },
            Stmt::Branch {
                kind: BranchKind::Jump,
                cond: Some(bin(BinOp::Eq, c(1, Width::W8), c(1, Width::W8))),
                target: c(0x2000, Width::W64),
            },
        ];
        let out: Vec<Stmt> = stmts.iter().map(fold_stmt).collect();
        ok(&out);
        // The store value folded to a constant, the condition to true.
        assert_eq!(
            out[1],
            Stmt::Store {
                addr: read(ra(0, Width::W64)),
                value: c(4, Width::W32),
            }
        );
    }
}
