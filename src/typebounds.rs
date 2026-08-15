//! Type bounds lattice (DESIGN irtype slices 16–17).
//!
//! Finite TIE-shaped lattice with **per-name lower and upper bounds**,
//! propagated directionally along def-use and φ edges (retypd lesson:
//! machine code is type-unsafe, so constraints are subtyping — not
//! symmetric unification). Conflicting evidence yields an explicit
//! [`Point::Conflict`], never papered over.
//!
//! Lattice (height finite ⇒ termination):
//! ```text
//!   ⊥  ≤  { bool, int_w^{s|u|?}, ptr(to-w?) }  ≤  num_w  ≤  ⊤
//! ```
//! `Conflict` is absorbing under [`join`] / [`meet`] when signedness or
//! ptr/int disagree without a common numeric upper.
//!
//! Presentation ([`present_bound`]): Proven / Guess / Conflict markers
//! for honest pseudocode headers and dumps.
//!
//! # Contract
//!
//! - Never panics; caps truncate name coverage.
//! - Deterministic (`BTreeMap` by SSA name id).
//! - [`check`]: `lower ≤ upper` (or Conflict) and width-consistency.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::ir::{Expr, Stmt, Width};
use crate::irssa::SsaFunction;
use crate::irtype::{FactKind, TypeFacts};

/// Cap on names that receive an explicit bound record.
pub const MAX_BOUND_NAMES: usize = 65_536;

/// Cap on φ / def-use propagation rounds (lattice height is tiny; this
/// is a hostile-input brake, not a correctness limit).
pub const MAX_FIXPOINT_ROUNDS: usize = 64;

/// Signedness on an integer lattice point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Signedness {
    Signed,
    Unsigned,
    Unknown,
}

/// One point in the finite type lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Point {
    Bottom,
    Bool,
    Int {
        width: Width,
        signed: Signedness,
    },
    /// Pointer, optionally to a known access width (one level).
    Ptr {
        pointee: Option<Width>,
    },
    /// Numeric top for a width (signedness erased).
    Num {
        width: Width,
    },
    Top,
    /// Explicit conflict — never silently collapsed to `int`.
    Conflict,
}

impl Point {
    pub fn token(self) -> String {
        match self {
            Point::Bottom => "⊥".into(),
            Point::Bool => "bool".into(),
            Point::Int {
                width,
                signed: Signedness::Signed,
            } => format!("i{}", width.bits()),
            Point::Int {
                width,
                signed: Signedness::Unsigned,
            } => format!("u{}", width.bits()),
            Point::Int {
                width,
                signed: Signedness::Unknown,
            } => format!("int{}", width.bits()),
            Point::Ptr {
                pointee: Some(w),
            } => format!("ptr.{}", w.bits() / 8),
            Point::Ptr { pointee: None } => "ptr".into(),
            Point::Num { width } => format!("num{}", width.bits()),
            Point::Top => "⊤".into(),
            Point::Conflict => "conflict".into(),
        }
    }

    pub fn width(self) -> Option<Width> {
        match self {
            Point::Bool => Some(Width::W1),
            Point::Int { width, .. } | Point::Num { width } => Some(width),
            Point::Ptr {
                pointee: Some(w),
            } => Some(w),
            _ => None,
        }
    }
}

/// Inclusive range `[lower, upper]` in the lattice (or Conflict).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bound {
    pub lower: Point,
    pub upper: Point,
}

impl Bound {
    pub fn top() -> Self {
        Self {
            lower: Point::Bottom,
            upper: Point::Top,
        }
    }

    pub fn conflict() -> Self {
        Self {
            lower: Point::Conflict,
            upper: Point::Conflict,
        }
    }

    pub fn is_conflict(self) -> bool {
        matches!(self.lower, Point::Conflict) || matches!(self.upper, Point::Conflict)
    }

    pub fn render(self) -> String {
        if self.is_conflict() {
            "conflict".into()
        } else {
            format!("[{} .. {}]", self.lower.token(), self.upper.token())
        }
    }
}

/// Per-function bound table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundMap {
    pub entry: u64,
    pub by_name: BTreeMap<u16, Bound>,
    pub capped: bool,
    pub rounds: usize,
}

impl BoundMap {
    pub fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "; typebounds entry={:#x} names={} rounds={}",
            self.entry,
            self.by_name.len(),
            self.rounds
        );
        if self.capped {
            out.push_str("; note: typebounds name cap hit\n");
        }
        for (name, b) in &self.by_name {
            let _ = writeln!(out, "  n{name} {}", b.render());
        }
        out
    }
}

/// Trust for presentation after bounds (slice 17).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundTrust {
    Proven,
    Guess,
    Conflict,
}

impl BoundTrust {
    pub fn token(self) -> &'static str {
        match self {
            BoundTrust::Proven => "proven",
            BoundTrust::Guess => "guess",
            BoundTrust::Conflict => "conflict",
        }
    }
}

/// Display token derived from a bound range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundDisplay {
    pub trust: BoundTrust,
    pub token: String,
    pub name: Option<u16>,
}

impl BoundDisplay {
    pub fn render(&self) -> String {
        match self.name {
            Some(n) => format!("{}:{} ({})", self.token, n, self.trust.token()),
            None => format!("{} ({})", self.token, self.trust.token()),
        }
    }

    /// Pseudocode-facing token: conflict gets an honest comment prefix.
    pub fn pseudo_token(&self) -> String {
        match self.trust {
            BoundTrust::Conflict => format!("/* conflicting evidence */ {}", self.token),
            BoundTrust::Guess => format!("/*?*/ {}", self.token),
            BoundTrust::Proven => self.token.clone(),
        }
    }
}

fn combine_signedness(sa: Signedness, sb: Signedness) -> Result<Signedness, ()> {
    match (sa, sb) {
        (Signedness::Signed, Signedness::Signed) => Ok(Signedness::Signed),
        (Signedness::Unsigned, Signedness::Unsigned) => Ok(Signedness::Unsigned),
        (Signedness::Unknown, Signedness::Unknown) => Ok(Signedness::Unknown),
        (Signedness::Unknown, s) | (s, Signedness::Unknown) => Ok(s),
        (Signedness::Signed, Signedness::Unsigned)
        | (Signedness::Unsigned, Signedness::Signed) => Err(()),
    }
}

/// Least upper bound. Absorbing [`Point::Conflict`] on irreconcilable atoms.
pub fn join(a: Point, b: Point) -> Point {
    if a == b {
        return a;
    }
    if matches!(a, Point::Conflict) || matches!(b, Point::Conflict) {
        return Point::Conflict;
    }
    if matches!(a, Point::Bottom) {
        return b;
    }
    if matches!(b, Point::Bottom) {
        return a;
    }
    if matches!(a, Point::Top) || matches!(b, Point::Top) {
        return Point::Top;
    }

    match (a, b) {
        (Point::Bool, Point::Bool) => Point::Bool,
        (Point::Bool, Point::Int { width, .. }) | (Point::Int { width, .. }, Point::Bool)
            if width == Width::W1 =>
        {
            Point::Num { width: Width::W1 }
        }
        (Point::Bool, Point::Num { width }) | (Point::Num { width }, Point::Bool)
            if width == Width::W1 =>
        {
            Point::Num { width: Width::W1 }
        }
        (Point::Bool, _) | (_, Point::Bool) => Point::Top,

        (
            Point::Int {
                width: wa,
                signed: sa,
            },
            Point::Int {
                width: wb,
                signed: sb,
            },
        ) => {
            if wa != wb {
                return Point::Top;
            }
            match combine_signedness(sa, sb) {
                Ok(signed) => Point::Int {
                    width: wa,
                    signed,
                },
                Err(()) => Point::Conflict,
            }
        }
        (Point::Int { width, .. }, Point::Num { width: wn })
        | (Point::Num { width: wn }, Point::Int { width, .. }) => {
            if width == wn {
                Point::Num { width }
            } else {
                Point::Top
            }
        }
        (Point::Num { width: wa }, Point::Num { width: wb }) => {
            if wa == wb {
                Point::Num { width: wa }
            } else {
                Point::Top
            }
        }

        (
            Point::Ptr {
                pointee: pa,
            },
            Point::Ptr {
                pointee: pb,
            },
        ) => Point::Ptr {
            pointee: match (pa, pb) {
                (Some(x), Some(y)) if x == y => Some(x),
                (Some(x), None) | (None, Some(x)) => Some(x),
                (Some(_), Some(_)) => None,
                (None, None) => None,
            },
        },
        (Point::Ptr { .. }, Point::Int { .. })
        | (Point::Int { .. }, Point::Ptr { .. })
        | (Point::Ptr { .. }, Point::Num { .. })
        | (Point::Num { .. }, Point::Ptr { .. }) => Point::Conflict,

        _ => Point::Top,
    }
}

/// Greatest lower bound.
pub fn meet(a: Point, b: Point) -> Point {
    if a == b {
        return a;
    }
    if matches!(a, Point::Conflict) || matches!(b, Point::Conflict) {
        return Point::Conflict;
    }
    if matches!(a, Point::Top) {
        return b;
    }
    if matches!(b, Point::Top) {
        return a;
    }
    if matches!(a, Point::Bottom) || matches!(b, Point::Bottom) {
        return Point::Bottom;
    }

    match (a, b) {
        (Point::Bool, Point::Bool) => Point::Bool,
        (Point::Bool, Point::Int { width, .. }) | (Point::Int { width, .. }, Point::Bool)
            if width == Width::W1 =>
        {
            Point::Bool
        }
        (Point::Bool, Point::Num { width }) | (Point::Num { width }, Point::Bool)
            if width == Width::W1 =>
        {
            Point::Bool
        }

        (
            Point::Int {
                width: wa,
                signed: sa,
            },
            Point::Int {
                width: wb,
                signed: sb,
            },
        ) => {
            if wa != wb {
                return Point::Bottom;
            }
            match combine_signedness(sa, sb) {
                Ok(signed) => Point::Int {
                    width: wa,
                    signed,
                },
                Err(()) => Point::Conflict,
            }
        }
        (Point::Int { width, signed }, Point::Num { width: wn })
        | (Point::Num { width: wn }, Point::Int { width, signed }) => {
            if width == wn {
                Point::Int { width, signed }
            } else {
                Point::Bottom
            }
        }
        (Point::Num { width: wa }, Point::Num { width: wb }) => {
            if wa == wb {
                Point::Num { width: wa }
            } else {
                Point::Bottom
            }
        }

        (
            Point::Ptr {
                pointee: pa,
            },
            Point::Ptr {
                pointee: pb,
            },
        ) => match (pa, pb) {
            (Some(x), Some(y)) if x == y => Point::Ptr { pointee: Some(x) },
            (Some(x), Some(y)) if x != y => Point::Bottom,
            (Some(x), None) | (None, Some(x)) => Point::Ptr { pointee: Some(x) },
            (None, None) => Point::Ptr { pointee: None },
            _ => Point::Bottom,
        },

        (Point::Ptr { .. }, _) | (_, Point::Ptr { .. }) => Point::Bottom,
        _ => Point::Bottom,
    }
}

/// Lattice order: `a ≤ b`.
pub fn leq(a: Point, b: Point) -> bool {
    if matches!(a, Point::Conflict) || matches!(b, Point::Conflict) {
        return matches!(a, Point::Conflict) && matches!(b, Point::Conflict);
    }
    // a ≤ b iff meet(a,b) == a  (and join(a,b) == b)
    meet(a, b) == a && join(a, b) == b
}

fn fact_point(kind: FactKind, name_width: Width) -> Point {
    match kind {
        FactKind::BoolUse => Point::Bool,
        FactKind::SignedUse => Point::Int {
            width: name_width,
            signed: Signedness::Signed,
        },
        FactKind::UnsignedUse => Point::Int {
            width: name_width,
            signed: Signedness::Unsigned,
        },
        FactKind::LoadedFrom(w) | FactKind::StoredTo(w) => Point::Ptr { pointee: Some(w) },
        FactKind::PtrAddr => Point::Ptr { pointee: None },
        FactKind::ArithWith(_) => Point::Num {
            width: name_width,
        },
    }
}

/// Seed lower bounds from evidence facts; upper starts at a width-aware Top/Num.
pub fn seed(f: &SsaFunction, facts: &TypeFacts) -> BoundMap {
    let mut by_name: BTreeMap<u16, Bound> = BTreeMap::new();
    let mut capped = false;

    for (nid, list) in &facts.by_name {
        if by_name.len() >= MAX_BOUND_NAMES {
            capped = true;
            break;
        }
        let width = f
            .names
            .get(*nid as usize)
            .map(|n| n.width)
            .unwrap_or(Width::W64);
        let mut lower = Point::Bottom;
        for fact in list {
            lower = join(lower, fact_point(fact.kind, width));
            if matches!(lower, Point::Conflict) {
                break;
            }
        }
        let upper = if matches!(lower, Point::Conflict) {
            Point::Conflict
        } else if matches!(lower, Point::Ptr { .. }) {
            Point::Ptr {
                pointee: match lower {
                    Point::Ptr { pointee } => pointee,
                    _ => None,
                },
            }
        } else if matches!(lower, Point::Bool) {
            Point::Num { width: Width::W1 }
        } else {
            Point::Num { width }
        };
        let bound = if matches!(lower, Point::Conflict) {
            Bound::conflict()
        } else {
            Bound { lower, upper }
        };
        by_name.insert(*nid, bound);
    }

    // Names that exist but have no facts: Bottom..Top (implicit).
    BoundMap {
        entry: f.entry,
        by_name,
        capped,
        rounds: 0,
    }
}

fn get_bound(map: &BoundMap, nid: u16) -> Bound {
    map.by_name.get(&nid).copied().unwrap_or_else(Bound::top)
}

/// Raise `dst`'s lower bound by joining with `src`'s lower (def → use /
/// φ arg → φ dst subtyping).
fn flow_lower(map: &mut BoundMap, dst: u16, src: u16, capped: &mut bool) {
    let src_b = get_bound(map, src);
    let mut dst_b = get_bound(map, dst);
    let new_lower = join(dst_b.lower, src_b.lower);
    if new_lower == dst_b.lower {
        return;
    }
    dst_b.lower = new_lower;
    if matches!(dst_b.lower, Point::Conflict) {
        dst_b = Bound::conflict();
    } else if !leq(dst_b.lower, dst_b.upper) && !matches!(dst_b.upper, Point::Top) {
        // Lower rose above upper → conflict.
        dst_b = Bound::conflict();
    }
    if map.by_name.len() >= MAX_BOUND_NAMES && !map.by_name.contains_key(&dst) {
        *capped = true;
        return;
    }
    map.by_name.insert(dst, dst_b);
}

/// Propagate along φ edges and assignment copies until fixpoint or cap.
pub fn propagate(f: &SsaFunction, mut map: BoundMap) -> BoundMap {
    let mut rounds = 0usize;
    loop {
        if rounds >= MAX_FIXPOINT_ROUNDS {
            break;
        }
        let before = map.by_name.clone();
        let mut capped = map.capped;

        for block in f.blocks.values() {
            for phi in &block.phis {
                for &(_pred, src) in &phi.args {
                    flow_lower(&mut map, phi.dst, src, &mut capped);
                }
            }
            for stmt in &block.stmts {
                if let Stmt::Assign {
                    dst,
                    value: Expr::Reg(src),
                } = stmt
                {
                    flow_lower(&mut map, dst.num, src.num, &mut capped);
                }
            }
        }

        map.capped |= capped;
        rounds += 1;
        if map.by_name == before {
            break;
        }
    }
    map.rounds = rounds;
    map
}

/// Seed + propagate.
pub fn analyze(f: &SsaFunction, facts: &TypeFacts) -> BoundMap {
    propagate(f, seed(f, facts))
}

/// Map a bound to a display type (slice 17).
pub fn present_bound(f: &SsaFunction, map: &BoundMap, name_id: u16) -> BoundDisplay {
    let width = f
        .names
        .get(name_id as usize)
        .map(|n| n.width)
        .unwrap_or(Width::W64);
    let Some(b) = map.by_name.get(&name_id) else {
        return BoundDisplay {
            trust: BoundTrust::Guess,
            token: format!("int{}", width.bits()),
            name: Some(name_id),
        };
    };
    if b.is_conflict() {
        return BoundDisplay {
            trust: BoundTrust::Conflict,
            token: format!("int{}", width.bits()),
            name: Some(name_id),
        };
    }
    // Prefer lower when it is a concrete atom; else upper if tighter than Top.
    let (trust, point) = match b.lower {
        Point::Bottom => {
            if matches!(b.upper, Point::Top) {
                (BoundTrust::Guess, Point::Num { width })
            } else {
                (BoundTrust::Guess, b.upper)
            }
        }
        Point::Conflict => (BoundTrust::Conflict, Point::Num { width }),
        p => {
            // Proven when lower is a concrete atom (not Bottom/Top/Num-only).
            let proven = matches!(
                p,
                Point::Bool
                    | Point::Int { .. }
                    | Point::Ptr { .. }
            );
            (
                if proven {
                    BoundTrust::Proven
                } else {
                    BoundTrust::Guess
                },
                p,
            )
        }
    };
    let token = match point {
        Point::Bottom | Point::Top => format!("int{}", width.bits()),
        Point::Conflict => format!("int{}", width.bits()),
        other => other.token(),
    };
    BoundDisplay {
        trust,
        token,
        name: Some(name_id),
    }
}

/// Invariant check: ranges well-formed; Conflict self-consistent.
pub fn check(f: &SsaFunction, map: &BoundMap) -> Result<(), String> {
    if map.entry != f.entry {
        return Err("entry mismatch".into());
    }
    if map.by_name.len() > MAX_BOUND_NAMES {
        return Err("bound names exceed MAX_BOUND_NAMES".into());
    }
    for (nid, b) in &map.by_name {
        if f.names.get(*nid as usize).is_none() {
            return Err(format!("unknown name id {nid}"));
        }
        if b.is_conflict() {
            if !matches!(b.lower, Point::Conflict) || !matches!(b.upper, Point::Conflict) {
                return Err(format!("n{nid}: conflict not dual"));
            }
            continue;
        }
        if !leq(b.lower, b.upper) {
            return Err(format!(
                "n{nid}: lower {} ≰ upper {}",
                b.lower.token(),
                b.upper.token()
            ));
        }
        // Width consistency when both sides expose a width.
        if let (Some(lw), Some(uw)) = (b.lower.width(), b.upper.width())
            && lw != uw
            && !matches!(b.upper, Point::Top)
            && !matches!(b.lower, Point::Bottom | Point::Ptr { .. } | Point::Bool)
            && !matches!(b.upper, Point::Ptr { .. } | Point::Bool)
            && matches!(b.lower, Point::Int { .. } | Point::Num { .. })
            && matches!(b.upper, Point::Int { .. } | Point::Num { .. })
        {
            return Err(format!("n{nid}: width mismatch {lw:?} vs {uw:?}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Space, Width};
    use crate::irssa::{Name, Phi, SsaBlock};
    use crate::irtype::{Fact, FactKind, TypeFacts};
    use crate::model::Arch;
    use std::collections::BTreeMap;

    fn name(width: Width) -> Name {
        Name {
            space: Space::Arch,
            cell: 0,
            version: 0,
            width,
        }
    }

    fn empty_ssa(entry: u64, n_names: u16) -> SsaFunction {
        let mut names = Vec::new();
        for i in 0..n_names {
            let mut n = name(Width::W64);
            n.cell = i;
            n.version = if i == 0 { 0 } else { 1 };
            names.push(n);
        }
        SsaFunction {
            entry,
            name: None,
            arch: Arch::X86_64,
            blocks: BTreeMap::new(),
            skipped: vec![],
            names,
            live_in: vec![],
            partial: vec![],
        }
    }

    #[test]
    fn join_signed_unsigned_is_conflict() {
        let a = Point::Int {
            width: Width::W32,
            signed: Signedness::Signed,
        };
        let b = Point::Int {
            width: Width::W32,
            signed: Signedness::Unsigned,
        };
        assert_eq!(join(a, b), Point::Conflict);
        assert!(!leq(a, b));
    }

    #[test]
    fn meet_int_under_num() {
        let i = Point::Int {
            width: Width::W64,
            signed: Signedness::Signed,
        };
        let n = Point::Num { width: Width::W64 };
        assert_eq!(meet(i, n), i);
        assert!(leq(i, n));
    }

    #[test]
    fn ptr_vs_int_join_conflicts() {
        let p = Point::Ptr {
            pointee: Some(Width::W64),
        };
        let i = Point::Int {
            width: Width::W64,
            signed: Signedness::Unknown,
        };
        assert_eq!(join(p, i), Point::Conflict);
    }

    #[test]
    fn seed_signed_unsigned_facts_conflict() {
        let mut f = empty_ssa(0x1000, 2);
        f.names[1].width = Width::W32;
        let facts = TypeFacts {
            entry: 0x1000,
            by_name: BTreeMap::from([(
                1u16,
                vec![
                    Fact {
                        name: 1,
                        kind: FactKind::SignedUse,
                        block: 0x1000,
                        stmt: 0,
                    },
                    Fact {
                        name: 1,
                        kind: FactKind::UnsignedUse,
                        block: 0x1000,
                        stmt: 1,
                    },
                ],
            )]),
            capped: false,
            fact_count: 2,
        };
        let map = seed(&f, &facts);
        assert!(map.by_name[&1].is_conflict());
        check(&f, &map).unwrap();
    }

    #[test]
    fn present_conflict_is_honest() {
        let mut f = empty_ssa(0x1000, 2);
        f.names[1].width = Width::W32;
        let mut map = BoundMap {
            entry: 0x1000,
            by_name: BTreeMap::new(),
            capped: false,
            rounds: 0,
        };
        map.by_name.insert(1, Bound::conflict());
        let d = present_bound(&f, &map, 1);
        assert_eq!(d.trust, BoundTrust::Conflict);
        assert!(d.pseudo_token().contains("conflicting evidence"));
    }

    #[test]
    fn phi_propagates_lower_bound() {
        let mut names = vec![name(Width::W64); 4];
        for (i, n) in names.iter_mut().enumerate() {
            n.cell = i as u16;
            n.version = i as u32;
        }
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x1000,
            SsaBlock {
                start: 0x1000,
                end: 0x1004,
                phis: vec![Phi {
                    dst: 3,
                    args: vec![(Some(0x1000), 1), (Some(0x1008), 2)],
                }],
                stmts: vec![],
                successors: vec![],
                truncated: false,
            },
        );
        let f = SsaFunction {
            entry: 0x1000,
            name: None,
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![],
            partial: vec![],
        };
        let facts = TypeFacts {
            entry: 0x1000,
            by_name: BTreeMap::from([
                (
                    1u16,
                    vec![Fact {
                        name: 1,
                        kind: FactKind::SignedUse,
                        block: 0x1000,
                        stmt: 0,
                    }],
                ),
                (
                    2u16,
                    vec![Fact {
                        name: 2,
                        kind: FactKind::SignedUse,
                        block: 0x1000,
                        stmt: 0,
                    }],
                ),
            ]),
            capped: false,
            fact_count: 2,
        };
        let map = analyze(&f, &facts);
        check(&f, &map).unwrap();
        let b3 = map.by_name.get(&3).expect("phi dst bound");
        assert!(
            matches!(
                b3.lower,
                Point::Int {
                    signed: Signedness::Signed,
                    ..
                }
            ),
            "expected signed lower on φ, got {}",
            b3.render()
        );
    }

    #[test]
    fn determinism() {
        let f = empty_ssa(0x2000, 3);
        let facts = TypeFacts {
            entry: 0x2000,
            by_name: BTreeMap::from([(
                2u16,
                vec![Fact {
                    name: 2,
                    kind: FactKind::PtrAddr,
                    block: 0x2000,
                    stmt: 0,
                }],
            )]),
            capped: false,
            fact_count: 1,
        };
        let a = analyze(&f, &facts).render();
        let b = analyze(&f, &facts).render();
        assert_eq!(a, b);
    }

    #[test]
    fn join_meet_idempotent() {
        let points = [
            Point::Bottom,
            Point::Bool,
            Point::Int {
                width: Width::W8,
                signed: Signedness::Unknown,
            },
            Point::Ptr { pointee: None },
            Point::Num { width: Width::W8 },
            Point::Top,
        ];
        for &p in &points {
            assert_eq!(join(p, p), p);
            assert_eq!(meet(p, p), p);
            assert!(leq(p, p));
        }
    }
}
