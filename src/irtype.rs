//! Type evidence facts from SSA (DESIGN irtype slices 15–17).
//!
//! One pass over SSA statements emits per-name usage facts that restate
//! the IR: load/store address widths, signed vs unsigned operator uses,
//! W1 (bool) contexts, and pointer hints when a LEA-shaped add/sub feeds
//! an address use. Each fact cites its statement — the evidence trail is
//! the proof.
//!
//! Bounds (slice 16) and honest presentation (slice 17) live in
//! [`crate::typebounds`]: finite lattice, directional φ/def-use
//! propagation, explicit [`typebounds::Point::Conflict`].
//!
//! Total: never panics. Caps truncate rather than grow without bound.
//! Deterministic via [`BTreeMap`] keyed by SSA name id.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::ir::{BinOp, Expr, Reg, Space, Stmt, UnOp, Width};
use crate::irssa::SsaFunction;
use crate::sig::Signature;
use crate::types::{ParamTypeMap, TypeTable, attach_sig_params};

/// Cap on total facts emitted for one function.
pub const MAX_FACTS: usize = 65_536;

/// Cap on facts retained per SSA name (oldest by encounter order kept).
pub const MAX_FACTS_PER_NAME: usize = 64;

/// One usage fact restating an IR site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FactKind {
    /// `name` was the address (or address base) of a load of `width`.
    LoadedFrom(Width),
    /// `name` was the address (or address base) of a store of `width`.
    StoredTo(Width),
    /// Operand of a signed compare / div / rem / arithmetic right shift /
    /// sign-extend.
    SignedUse,
    /// Operand of an unsigned compare / div / rem / logical right shift /
    /// zero-extend.
    UnsignedUse,
    /// Appeared in a `W1` context (branch condition or compare result).
    BoolUse,
    /// Participated in add/sub/mul with constant `k` (masked to operand width).
    ArithWith(u64),
    /// Destination of a LEA-shaped `add`/`sub` (reg ± const) that is later
    /// used as a memory address — pointer evidence beyond raw load/store.
    PtrAddr,
}

impl FactKind {
    fn token(self) -> String {
        match self {
            FactKind::LoadedFrom(w) => format!("load.{}", w.bits() / 8),
            FactKind::StoredTo(w) => format!("store.{}", w.bits() / 8),
            FactKind::SignedUse => "signed".into(),
            FactKind::UnsignedUse => "unsigned".into(),
            FactKind::BoolUse => "bool".into(),
            FactKind::ArithWith(k) => format!("arith+{k:#x}"),
            FactKind::PtrAddr => "ptr".into(),
        }
    }
}

/// One cited fact: SSA name + kind + statement location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fact {
    pub name: u16,
    pub kind: FactKind,
    /// Block start VA.
    pub block: u64,
    /// Index into that block's `stmts`.
    pub stmt: usize,
}

/// Per-function evidence table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeFacts {
    pub entry: u64,
    /// Facts grouped by SSA name id (sorted keys, deterministic).
    pub by_name: BTreeMap<u16, Vec<Fact>>,
    /// True when [`MAX_FACTS`] or per-name caps truncated output.
    pub capped: bool,
    /// Total facts retained (sum of `by_name` lengths).
    pub fact_count: usize,
}

impl TypeFacts {
    /// Deterministic multi-line dump for CLI / tests.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "; typefacts entry={:#x} names={} facts={}",
            self.entry,
            self.by_name.len(),
            self.fact_count
        );
        if self.capped {
            out.push_str("; note: typefacts cap hit\n");
        }
        for (name, facts) in &self.by_name {
            for fact in facts {
                let _ = writeln!(
                    out,
                    "  n{name} {} @ {:#x}:{}",
                    fact.kind.token(),
                    fact.block,
                    fact.stmt
                );
            }
        }
        out
    }
}

/// Collect evidence facts from `f`. Total: never panics.
pub fn collect(f: &SsaFunction) -> TypeFacts {
    let mut by_name: BTreeMap<u16, Vec<Fact>> = BTreeMap::new();
    let mut fact_count = 0usize;
    let mut capped = false;

    // Names defined by LEA-shaped add/sub (reg ± const). Used when that
    // name later appears as a memory address.
    let mut lea_shaped: BTreeSet<u16> = BTreeSet::new();

    let order = block_order(f);
    for &bva in &order {
        let Some(block) = f.blocks.get(&bva) else {
            continue;
        };
        for (si, stmt) in block.stmts.iter().enumerate() {
            if fact_count >= MAX_FACTS {
                capped = true;
                break;
            }
            match stmt {
                Stmt::Assign { dst, value } => {
                    if is_lea_shaped(value) {
                        lea_shaped.insert(dst.num);
                    }
                    walk_value_facts(
                        value,
                        bva,
                        si,
                        &mut by_name,
                        &mut fact_count,
                        &mut capped,
                    );
                }
                Stmt::Store { addr, value } => {
                    emit_addr_facts(
                        addr,
                        FactKind::StoredTo(value.width_of().unwrap_or(Width::W64)),
                        bva,
                        si,
                        &lea_shaped,
                        &mut by_name,
                        &mut fact_count,
                        &mut capped,
                    );
                    walk_value_facts(
                        value,
                        bva,
                        si,
                        &mut by_name,
                        &mut fact_count,
                        &mut capped,
                    );
                }
                Stmt::Branch { cond, target, .. } => {
                    if let Some(c) = cond {
                        emit_bool_regs(c, bva, si, &mut by_name, &mut fact_count, &mut capped);
                        walk_value_facts(
                            c,
                            bva,
                            si,
                            &mut by_name,
                            &mut fact_count,
                            &mut capped,
                        );
                    }
                    walk_value_facts(
                        target,
                        bva,
                        si,
                        &mut by_name,
                        &mut fact_count,
                        &mut capped,
                    );
                }
                Stmt::Intrinsic { reads, .. } => {
                    for r in reads {
                        walk_value_facts(
                            r,
                            bva,
                            si,
                            &mut by_name,
                            &mut fact_count,
                            &mut capped,
                        );
                    }
                }
            }
            // Loads nested in any expression of this statement.
            walk_loads(
                stmt,
                bva,
                si,
                &lea_shaped,
                &mut by_name,
                &mut fact_count,
                &mut capped,
            );
        }
        if fact_count >= MAX_FACTS {
            capped = true;
            break;
        }
    }

    TypeFacts {
        entry: f.entry,
        by_name,
        capped,
        fact_count,
    }
}

/// Attach signature placeholder types, then refine from evidence on
/// params that carry a live-in [`crate::sig::Param::name_id`].
pub fn attach_sig_with_evidence(
    f: &SsaFunction,
    sig: &Signature,
    facts: &TypeFacts,
    table: &mut TypeTable,
) -> ParamTypeMap {
    let mut map = attach_sig_params(sig, table);
    refine_sig_params(f, sig, facts, table, &mut map);
    map
}

/// Upgrade param / return [`TypeId`]s in `map` when evidence on the
/// corresponding SSA name supports signedness or pointer-ness.
pub fn refine_sig_params(
    f: &SsaFunction,
    sig: &Signature,
    facts: &TypeFacts,
    table: &mut TypeTable,
    map: &mut ParamTypeMap,
) {
    for (i, p) in sig.params.iter().enumerate() {
        let Some((_, tid)) = map.params.iter_mut().find(|(ix, _)| *ix == i as u16) else {
            continue;
        };
        let Some(nid) = p.name_id else {
            continue;
        };
        if let Some(new_id) = present_as_type(f, facts, nid, p.width, table) {
            *tid = new_id;
        }
    }
    if let (Some(ret), Some(r)) = (map.ret.as_mut(), sig.returns.first()) {
        // Return cells lack a stable live-in name_id; leave width-only
        // placeholder unless we later thread caller evidence.
        let _ = (ret, r, f);
    }
}

/// Trust label for a display type: evidence-backed vs placeholder vs conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trust {
    /// At least one fact / bound justifies the displayed token.
    Proven,
    /// No facts — width-only / unknown placeholder.
    Unproven,
    /// Signedness or ptr/int evidence disagree — honest conflict.
    Conflict,
}

impl Trust {
    pub fn token(self) -> &'static str {
        match self {
            Trust::Proven => "proven",
            Trust::Unproven => "unproven",
            Trust::Conflict => "conflict",
        }
    }
}

/// A printable type token with honesty marker for later `pseudo` use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayType {
    pub trust: Trust,
    /// Short token (`i32`, `u64`, `ptr`, `int64`, `bool`, `unknown`).
    pub token: String,
    /// SSA name this describes (when known).
    pub name: Option<u16>,
}

impl DisplayType {
    pub fn render(&self) -> String {
        match self.name {
            Some(n) => format!("{}:{} ({})", self.token, n, self.trust.token()),
            None => format!("{} ({})", self.token, self.trust.token()),
        }
    }

    /// Pseudocode-facing form (conflict comment when needed).
    pub fn pseudo_token(&self) -> String {
        match self.trust {
            Trust::Conflict => format!("/* conflicting evidence */ {}", self.token),
            Trust::Unproven => format!("/*?*/ {}", self.token),
            Trust::Proven => self.token.clone(),
        }
    }
}

/// Presentation helper: prefer [`crate::typebounds`] when evidence exists;
/// otherwise width-only unproven. Conflict is never papered over as a
/// confident `int`.
pub fn present(f: &SsaFunction, facts: &TypeFacts, name_id: u16) -> DisplayType {
    let width = f
        .names
        .get(name_id as usize)
        .map(|n| n.width)
        .unwrap_or(Width::W64);
    if facts.by_name.get(&name_id).is_none_or(|l| l.is_empty()) {
        return DisplayType {
            trust: Trust::Unproven,
            token: format!("int{}", width.bits()),
            name: Some(name_id),
        };
    }
    let bounds = crate::typebounds::analyze(f, facts);
    let d = crate::typebounds::present_bound(f, &bounds, name_id);
    let trust = match d.trust {
        crate::typebounds::BoundTrust::Proven => Trust::Proven,
        crate::typebounds::BoundTrust::Guess => Trust::Unproven,
        crate::typebounds::BoundTrust::Conflict => Trust::Conflict,
    };
    DisplayType {
        trust,
        token: d.token,
        name: Some(name_id),
    }
}

/// From-scratch sanity check: facts cite real names/stmts and replay.
pub fn check(f: &SsaFunction, facts: &TypeFacts) -> Result<(), String> {
    if facts.entry != f.entry {
        return Err("entry mismatch".into());
    }
    if facts.fact_count > MAX_FACTS {
        return Err("facts exceed MAX_FACTS".into());
    }
    let mut counted = 0usize;
    for (nid, list) in &facts.by_name {
        if f.names.get(*nid as usize).is_none() {
            return Err(format!("unknown name id {nid}"));
        }
        if list.len() > MAX_FACTS_PER_NAME {
            return Err(format!("name {nid} exceeds MAX_FACTS_PER_NAME"));
        }
        for fact in list {
            if fact.name != *nid {
                return Err(format!("fact name {} != map key {nid}", fact.name));
            }
            let Some(block) = f.blocks.get(&fact.block) else {
                return Err(format!("fact cites missing block {:#x}", fact.block));
            };
            let Some(stmt) = block.stmts.get(fact.stmt) else {
                return Err(format!(
                    "fact cites missing stmt {:#x}:{}",
                    fact.block, fact.stmt
                ));
            };
            if !fact_replays(stmt, fact) {
                return Err(format!(
                    "fact {} @ {:#x}:{} does not replay",
                    fact.kind.token(),
                    fact.block,
                    fact.stmt
                ));
            }
            counted += 1;
        }
    }
    if counted != facts.fact_count {
        return Err(format!(
            "fact_count {} != summed {}",
            facts.fact_count, counted
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn block_order(f: &SsaFunction) -> Vec<u64> {
    let mut v: Vec<u64> = f.blocks.keys().copied().collect();
    v.sort_unstable();
    v
}

fn push_fact(
    by_name: &mut BTreeMap<u16, Vec<Fact>>,
    fact_count: &mut usize,
    capped: &mut bool,
    fact: Fact,
) {
    if *fact_count >= MAX_FACTS {
        *capped = true;
        return;
    }
    let list = by_name.entry(fact.name).or_default();
    if list.len() >= MAX_FACTS_PER_NAME {
        *capped = true;
        return;
    }
    // Dedup identical kind at same site.
    if list
        .iter()
        .any(|e| e.kind == fact.kind && e.block == fact.block && e.stmt == fact.stmt)
    {
        return;
    }
    list.push(fact);
    *fact_count += 1;
}

fn is_lea_shaped(e: &Expr) -> bool {
    match e {
        Expr::Binary {
            op: BinOp::Add | BinOp::Sub,
            lhs,
            rhs,
        } => {
            matches!(lhs.as_ref(), Expr::Reg(_)) && matches!(rhs.as_ref(), Expr::Const { .. })
                || matches!(rhs.as_ref(), Expr::Reg(_)) && matches!(lhs.as_ref(), Expr::Const { .. })
                || matches!(lhs.as_ref(), Expr::Reg(_)) && matches!(rhs.as_ref(), Expr::Reg(_))
        }
        Expr::Unary {
            op: UnOp::Truncate(_) | UnOp::ZeroExtend(_) | UnOp::SignExtend(_),
            operand,
        } => is_lea_shaped(operand),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_addr_facts(
    addr: &Expr,
    kind: FactKind,
    block: u64,
    stmt: usize,
    lea_shaped: &BTreeSet<u16>,
    by_name: &mut BTreeMap<u16, Vec<Fact>>,
    fact_count: &mut usize,
    capped: &mut bool,
) {
    let mut regs = Vec::new();
    collect_regs(addr, &mut regs);
    for r in regs {
        if r.space == Space::Flag {
            continue;
        }
        push_fact(
            by_name,
            fact_count,
            capped,
            Fact {
                name: r.num,
                kind,
                block,
                stmt,
            },
        );
        if lea_shaped.contains(&r.num) {
            push_fact(
                by_name,
                fact_count,
                capped,
                Fact {
                    name: r.num,
                    kind: FactKind::PtrAddr,
                    block,
                    stmt,
                },
            );
        }
    }
}

fn emit_bool_regs(
    e: &Expr,
    block: u64,
    stmt: usize,
    by_name: &mut BTreeMap<u16, Vec<Fact>>,
    fact_count: &mut usize,
    capped: &mut bool,
) {
    // Only W1-typed registers in a condition are bool uses; compare
    // operands stay under SignedUse/UnsignedUse.
    let mut regs = Vec::new();
    collect_regs(e, &mut regs);
    for r in regs {
        if r.width != Width::W1 {
            continue;
        }
        push_fact(
            by_name,
            fact_count,
            capped,
            Fact {
                name: r.num,
                kind: FactKind::BoolUse,
                block,
                stmt,
            },
        );
    }
}

fn walk_loads(
    stmt: &Stmt,
    block: u64,
    stmt_ix: usize,
    lea_shaped: &BTreeSet<u16>,
    by_name: &mut BTreeMap<u16, Vec<Fact>>,
    fact_count: &mut usize,
    capped: &mut bool,
) {
    match stmt {
        Stmt::Assign { value, .. } => {
            walk_loads_expr(value, block, stmt_ix, lea_shaped, by_name, fact_count, capped)
        }
        Stmt::Store { addr, value } => {
            walk_loads_expr(addr, block, stmt_ix, lea_shaped, by_name, fact_count, capped);
            walk_loads_expr(
                value,
                block,
                stmt_ix,
                lea_shaped,
                by_name,
                fact_count,
                capped,
            );
        }
        Stmt::Branch { cond, target, .. } => {
            if let Some(c) = cond {
                walk_loads_expr(c, block, stmt_ix, lea_shaped, by_name, fact_count, capped);
            }
            walk_loads_expr(
                target,
                block,
                stmt_ix,
                lea_shaped,
                by_name,
                fact_count,
                capped,
            );
        }
        Stmt::Intrinsic { reads, .. } => {
            for r in reads {
                walk_loads_expr(r, block, stmt_ix, lea_shaped, by_name, fact_count, capped);
            }
        }
    }
}

fn walk_loads_expr(
    e: &Expr,
    block: u64,
    stmt: usize,
    lea_shaped: &BTreeSet<u16>,
    by_name: &mut BTreeMap<u16, Vec<Fact>>,
    fact_count: &mut usize,
    capped: &mut bool,
) {
    match e {
        Expr::Load { addr, width } => {
            emit_addr_facts(
                addr,
                FactKind::LoadedFrom(*width),
                block,
                stmt,
                lea_shaped,
                by_name,
                fact_count,
                capped,
            );
            walk_loads_expr(
                addr,
                block,
                stmt,
                lea_shaped,
                by_name,
                fact_count,
                capped,
            );
        }
        Expr::Unary { operand, .. } => {
            walk_loads_expr(
                operand,
                block,
                stmt,
                lea_shaped,
                by_name,
                fact_count,
                capped,
            )
        }
        Expr::Binary { lhs, rhs, .. } => {
            walk_loads_expr(lhs, block, stmt, lea_shaped, by_name, fact_count, capped);
            walk_loads_expr(rhs, block, stmt, lea_shaped, by_name, fact_count, capped);
        }
        Expr::Const { .. } | Expr::Reg(_) => {}
    }
}

fn walk_value_facts(
    e: &Expr,
    block: u64,
    stmt: usize,
    by_name: &mut BTreeMap<u16, Vec<Fact>>,
    fact_count: &mut usize,
    capped: &mut bool,
) {
    match e {
        Expr::Const { .. } | Expr::Reg(_) => {}
        Expr::Load { addr, .. } => {
            walk_value_facts(addr, block, stmt, by_name, fact_count, capped);
        }
        Expr::Unary { op, operand } => {
            match op {
                UnOp::SignExtend(_) => emit_signedness(
                    operand,
                    FactKind::SignedUse,
                    block,
                    stmt,
                    by_name,
                    fact_count,
                    capped,
                ),
                UnOp::ZeroExtend(_) => emit_signedness(
                    operand,
                    FactKind::UnsignedUse,
                    block,
                    stmt,
                    by_name,
                    fact_count,
                    capped,
                ),
                _ => {}
            }
            walk_value_facts(operand, block, stmt, by_name, fact_count, capped);
        }
        Expr::Binary { op, lhs, rhs } => {
            match op {
                BinOp::SDiv | BinOp::SRem | BinOp::AShr | BinOp::Slt | BinOp::Sle => {
                    emit_signedness(
                        lhs,
                        FactKind::SignedUse,
                        block,
                        stmt,
                        by_name,
                        fact_count,
                        capped,
                    );
                    if !matches!(op, BinOp::AShr) {
                        emit_signedness(
                            rhs,
                            FactKind::SignedUse,
                            block,
                            stmt,
                            by_name,
                            fact_count,
                            capped,
                        );
                    }
                }
                BinOp::UDiv | BinOp::URem | BinOp::LShr | BinOp::Ult | BinOp::Ule => {
                    emit_signedness(
                        lhs,
                        FactKind::UnsignedUse,
                        block,
                        stmt,
                        by_name,
                        fact_count,
                        capped,
                    );
                    if !matches!(op, BinOp::LShr) {
                        emit_signedness(
                            rhs,
                            FactKind::UnsignedUse,
                            block,
                            stmt,
                            by_name,
                            fact_count,
                            capped,
                        );
                    }
                }
                BinOp::Add | BinOp::Sub | BinOp::Mul => {
                    if let Expr::Const { value, width } = lhs.as_ref() {
                        emit_arith_regs(
                            rhs,
                            *value & width.mask(),
                            block,
                            stmt,
                            by_name,
                            fact_count,
                            capped,
                        );
                    }
                    if let Expr::Const { value, width } = rhs.as_ref() {
                        emit_arith_regs(
                            lhs,
                            *value & width.mask(),
                            block,
                            stmt,
                            by_name,
                            fact_count,
                            capped,
                        );
                    }
                }
                BinOp::Eq | BinOp::Ne => {
                    // Equality is signedness-neutral; still bool result.
                }
                _ => {}
            }
            if op.is_compare() {
                // Result width is W1 — mark destination via BoolUse on
                // operands only when they are W1; otherwise skip.
            }
            walk_value_facts(lhs, block, stmt, by_name, fact_count, capped);
            walk_value_facts(rhs, block, stmt, by_name, fact_count, capped);
        }
    }
}

fn emit_signedness(
    e: &Expr,
    kind: FactKind,
    block: u64,
    stmt: usize,
    by_name: &mut BTreeMap<u16, Vec<Fact>>,
    fact_count: &mut usize,
    capped: &mut bool,
) {
    let mut regs = Vec::new();
    collect_regs(e, &mut regs);
    for r in regs {
        if r.space == Space::Flag {
            continue;
        }
        push_fact(
            by_name,
            fact_count,
            capped,
            Fact {
                name: r.num,
                kind,
                block,
                stmt,
            },
        );
    }
}

fn emit_arith_regs(
    e: &Expr,
    k: u64,
    block: u64,
    stmt: usize,
    by_name: &mut BTreeMap<u16, Vec<Fact>>,
    fact_count: &mut usize,
    capped: &mut bool,
) {
    let mut regs = Vec::new();
    collect_regs(e, &mut regs);
    for r in regs {
        if r.space == Space::Flag {
            continue;
        }
        push_fact(
            by_name,
            fact_count,
            capped,
            Fact {
                name: r.num,
                kind: FactKind::ArithWith(k),
                block,
                stmt,
            },
        );
    }
}

fn collect_regs(e: &Expr, out: &mut Vec<Reg>) {
    match e {
        Expr::Reg(r) => out.push(*r),
        Expr::Const { .. } => {}
        Expr::Load { addr, .. } => collect_regs(addr, out),
        Expr::Unary { operand, .. } => collect_regs(operand, out),
        Expr::Binary { lhs, rhs, .. } => {
            collect_regs(lhs, out);
            collect_regs(rhs, out);
        }
    }
}

#[allow(dead_code)] // retained for goldens / future presentation without bounds
fn summarize_facts(list: &[Fact], width: Width) -> String {
    let mut has_ptr = false;
    let mut load_w: Option<Width> = None;
    let mut signed = false;
    let mut unsigned = false;
    let mut boolish = false;
    for f in list {
        match f.kind {
            FactKind::LoadedFrom(w) => {
                has_ptr = true;
                load_w = Some(w);
            }
            FactKind::StoredTo(w) => {
                has_ptr = true;
                load_w = load_w.or(Some(w));
            }
            FactKind::PtrAddr => has_ptr = true,
            FactKind::SignedUse => signed = true,
            FactKind::UnsignedUse => unsigned = true,
            FactKind::BoolUse => boolish = true,
            FactKind::ArithWith(_) => {}
        }
    }
    if has_ptr {
        return match load_w {
            Some(w) => format!("ptr.{}", w.bits() / 8),
            None => "ptr".into(),
        };
    }
    if boolish && width == Width::W1 {
        return "bool".into();
    }
    match (signed, unsigned) {
        (true, false) => format!("i{}", width.bits()),
        (false, true) => format!("u{}", width.bits()),
        _ => format!("int{}", width.bits()),
    }
}

fn present_as_type(
    f: &SsaFunction,
    facts: &TypeFacts,
    name_id: u16,
    width: Width,
    table: &mut TypeTable,
) -> Option<crate::types::TypeId> {
    let list = facts.by_name.get(&name_id)?;
    if list.is_empty() {
        return None;
    }
    let disp = present(f, facts, name_id);
    if disp.trust != Trust::Proven {
        return None;
    }
    let mut has_ptr = false;
    let mut pointee_w: Option<Width> = None;
    let mut signed: Option<bool> = None;
    let mut saw_signed = false;
    let mut saw_unsigned = false;
    for fact in list {
        match fact.kind {
            FactKind::LoadedFrom(w) | FactKind::StoredTo(w) => {
                has_ptr = true;
                pointee_w = Some(w);
            }
            FactKind::PtrAddr => has_ptr = true,
            FactKind::SignedUse => saw_signed = true,
            FactKind::UnsignedUse => saw_unsigned = true,
            FactKind::BoolUse | FactKind::ArithWith(_) => {}
        }
    }
    if saw_signed && !saw_unsigned {
        signed = Some(true);
    } else if saw_unsigned && !saw_signed {
        signed = Some(false);
    }
    if has_ptr {
        let pointee = match pointee_w {
            Some(w) => table.intern_int(w, None),
            None => table.intern_unknown(),
        };
        return Some(table.intern_ptr(pointee));
    }
    Some(table.intern_int(width, signed))
}

/// Re-derive whether `fact` is justified by `stmt` (citation replay).
fn fact_replays(stmt: &Stmt, fact: &Fact) -> bool {
    match fact.kind {
        FactKind::LoadedFrom(w) => stmt_has_load_addr(stmt, fact.name, w),
        FactKind::StoredTo(w) => match stmt {
            Stmt::Store { addr, value } => {
                value.width_of() == Some(w) && expr_mentions_reg(addr, fact.name)
            }
            _ => false,
        },
        FactKind::PtrAddr => match stmt {
            Stmt::Store { addr, .. } => expr_mentions_reg(addr, fact.name),
            Stmt::Assign { value, .. } => expr_load_mentions(value, fact.name),
            Stmt::Branch { cond, target, .. } => {
                cond.as_ref()
                    .is_some_and(|c| expr_load_mentions(c, fact.name))
                    || expr_load_mentions(target, fact.name)
            }
            Stmt::Intrinsic { reads, .. } => reads.iter().any(|r| expr_load_mentions(r, fact.name)),
        },
        FactKind::SignedUse => stmt_has_signed_use(stmt, fact.name),
        FactKind::UnsignedUse => stmt_has_unsigned_use(stmt, fact.name),
        FactKind::BoolUse => match stmt {
            Stmt::Branch {
                cond: Some(c), ..
            } => expr_mentions_reg(c, fact.name),
            _ => false,
        },
        FactKind::ArithWith(k) => stmt_has_arith_with(stmt, fact.name, k),
    }
}

fn stmt_has_load_addr(stmt: &Stmt, name: u16, w: Width) -> bool {
    match stmt {
        Stmt::Assign { value, .. } => expr_has_load(value, name, w),
        Stmt::Store { addr, value } => {
            expr_has_load(addr, name, w) || expr_has_load(value, name, w)
        }
        Stmt::Branch { cond, target, .. } => {
            cond.as_ref()
                .is_some_and(|c| expr_has_load(c, name, w))
                || expr_has_load(target, name, w)
        }
        Stmt::Intrinsic { reads, .. } => reads.iter().any(|r| expr_has_load(r, name, w)),
    }
}

fn expr_has_load(e: &Expr, name: u16, w: Width) -> bool {
    match e {
        Expr::Load { addr, width } => *width == w && expr_mentions_reg(addr, name),
        Expr::Unary { operand, .. } => expr_has_load(operand, name, w),
        Expr::Binary { lhs, rhs, .. } => {
            expr_has_load(lhs, name, w) || expr_has_load(rhs, name, w)
        }
        _ => false,
    }
}

fn expr_load_mentions(e: &Expr, name: u16) -> bool {
    match e {
        Expr::Load { addr, .. } => expr_mentions_reg(addr, name) || expr_load_mentions(addr, name),
        Expr::Unary { operand, .. } => expr_load_mentions(operand, name),
        Expr::Binary { lhs, rhs, .. } => {
            expr_load_mentions(lhs, name) || expr_load_mentions(rhs, name)
        }
        _ => false,
    }
}

fn expr_mentions_reg(e: &Expr, name: u16) -> bool {
    match e {
        Expr::Reg(r) => r.num == name,
        Expr::Const { .. } => false,
        Expr::Load { addr, .. } => expr_mentions_reg(addr, name),
        Expr::Unary { operand, .. } => expr_mentions_reg(operand, name),
        Expr::Binary { lhs, rhs, .. } => {
            expr_mentions_reg(lhs, name) || expr_mentions_reg(rhs, name)
        }
    }
}

fn stmt_has_signed_use(stmt: &Stmt, name: u16) -> bool {
    stmt_exprs(stmt).any(|e| expr_has_signed_use(e, name))
}

fn stmt_has_unsigned_use(stmt: &Stmt, name: u16) -> bool {
    stmt_exprs(stmt).any(|e| expr_has_unsigned_use(e, name))
}

fn stmt_has_arith_with(stmt: &Stmt, name: u16, k: u64) -> bool {
    stmt_exprs(stmt).any(|e| expr_has_arith_with(e, name, k))
}

fn stmt_exprs(stmt: &Stmt) -> impl Iterator<Item = &Expr> {
    let mut v: Vec<&Expr> = Vec::new();
    match stmt {
        Stmt::Assign { value, .. } => v.push(value),
        Stmt::Store { addr, value } => {
            v.push(addr);
            v.push(value);
        }
        Stmt::Branch { cond, target, .. } => {
            if let Some(c) = cond {
                v.push(c);
            }
            v.push(target);
        }
        Stmt::Intrinsic { reads, .. } => {
            for r in reads {
                v.push(r);
            }
        }
    }
    v.into_iter()
}

fn expr_has_signed_use(e: &Expr, name: u16) -> bool {
    match e {
        Expr::Unary {
            op: UnOp::SignExtend(_),
            operand,
        } => expr_mentions_reg(operand, name) || expr_has_signed_use(operand, name),
        Expr::Binary { op, lhs, rhs } => {
            let here = matches!(
                op,
                BinOp::SDiv | BinOp::SRem | BinOp::AShr | BinOp::Slt | BinOp::Sle
            ) && (expr_mentions_reg(lhs, name)
                || (!matches!(op, BinOp::AShr) && expr_mentions_reg(rhs, name)));
            here || expr_has_signed_use(lhs, name) || expr_has_signed_use(rhs, name)
        }
        Expr::Unary { operand, .. } => expr_has_signed_use(operand, name),
        Expr::Load { addr, .. } => expr_has_signed_use(addr, name),
        _ => false,
    }
}

fn expr_has_unsigned_use(e: &Expr, name: u16) -> bool {
    match e {
        Expr::Unary {
            op: UnOp::ZeroExtend(_),
            operand,
        } => expr_mentions_reg(operand, name) || expr_has_unsigned_use(operand, name),
        Expr::Binary { op, lhs, rhs } => {
            let here = matches!(
                op,
                BinOp::UDiv | BinOp::URem | BinOp::LShr | BinOp::Ult | BinOp::Ule
            ) && (expr_mentions_reg(lhs, name)
                || (!matches!(op, BinOp::LShr) && expr_mentions_reg(rhs, name)));
            here || expr_has_unsigned_use(lhs, name) || expr_has_unsigned_use(rhs, name)
        }
        Expr::Unary { operand, .. } => expr_has_unsigned_use(operand, name),
        Expr::Load { addr, .. } => expr_has_unsigned_use(addr, name),
        _ => false,
    }
}

fn expr_has_arith_with(e: &Expr, name: u16, k: u64) -> bool {
    match e {
        Expr::Binary {
            op: BinOp::Add | BinOp::Sub | BinOp::Mul,
            lhs,
            rhs,
        } => {
            let hit = match (lhs.as_ref(), rhs.as_ref()) {
                (Expr::Const { value, width }, other)
                | (other, Expr::Const { value, width }) => {
                    (*value & width.mask()) == k && expr_mentions_reg(other, name)
                }
                _ => false,
            };
            hit || expr_has_arith_with(lhs, name, k) || expr_has_arith_with(rhs, name, k)
        }
        Expr::Unary { operand, .. } => expr_has_arith_with(operand, name, k),
        Expr::Load { addr, .. } => expr_has_arith_with(addr, name, k),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, Expr, Reg, Space, Stmt, UnOp, Width};
    use crate::irssa::{Name, SsaBlock, SsaFunction};
    use crate::model::Arch;
    use crate::sig;
    use crate::types;
    use std::collections::BTreeMap;

    fn name(cell: u16, version: u32, w: Width) -> Name {
        Name {
            space: Space::Arch,
            cell,
            version,
            width: w,
        }
    }

    fn reg(id: u16, w: Width) -> Reg {
        Reg {
            space: Space::Arch,
            num: id,
            width: w,
        }
    }

    fn func_with(stmts: Vec<Stmt>, names: Vec<Name>, live_in: Vec<u16>) -> SsaFunction {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x1000,
            SsaBlock {
                start: 0x1000,
                end: 0x1010,
                phis: vec![],
                stmts,
                successors: vec![],
                truncated: false,
            },
        );
        SsaFunction {
            entry: 0x1000,
            name: Some("t".into()),
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in,
            partial: vec![],
        }
    }

    #[test]
    fn load_store_emit_width_facts() {
        // n0 = live-in pointer; load through it; store through n1.
        let stmts = vec![
            Stmt::Assign {
                dst: reg(2, Width::W32),
                value: Expr::Load {
                    addr: Box::new(Expr::Reg(reg(0, Width::W64))),
                    width: Width::W32,
                },
            },
            Stmt::Store {
                addr: Expr::Reg(reg(1, Width::W64)),
                value: Expr::Reg(reg(2, Width::W32)),
            },
        ];
        let names = vec![
            name(7, 0, Width::W64), // rdi-ish
            name(6, 0, Width::W64), // rsi-ish
            name(0, 1, Width::W32),
        ];
        let f = func_with(stmts, names, vec![0, 1]);
        let facts = collect(&f);
        assert!(check(&f, &facts).is_ok(), "{:?}", check(&f, &facts));
        let n0 = facts.by_name.get(&0).expect("n0 facts");
        assert!(
            n0.iter()
                .any(|x| x.kind == FactKind::LoadedFrom(Width::W32)),
            "{n0:?}"
        );
        let n1 = facts.by_name.get(&1).expect("n1 facts");
        assert!(
            n1.iter()
                .any(|x| x.kind == FactKind::StoredTo(Width::W32)),
            "{n1:?}"
        );
        let dump = facts.render();
        assert!(dump.contains("load.4"), "{dump}");
        assert!(dump.contains("store.4"), "{dump}");
    }

    #[test]
    fn signed_compare_emits_signed_use() {
        let stmts = vec![Stmt::Assign {
            dst: reg(2, Width::W1),
            value: Expr::Binary {
                op: BinOp::Slt,
                lhs: Box::new(Expr::Reg(reg(0, Width::W64))),
                rhs: Box::new(Expr::Reg(reg(1, Width::W64))),
            },
        }];
        let names = vec![
            name(7, 0, Width::W64),
            name(6, 0, Width::W64),
            name(0, 1, Width::W1),
        ];
        let f = func_with(stmts, names, vec![0, 1]);
        let facts = collect(&f);
        assert!(check(&f, &facts).is_ok());
        assert!(
            facts
                .by_name
                .get(&0)
                .unwrap()
                .iter()
                .any(|x| x.kind == FactKind::SignedUse)
        );
        assert!(
            facts
                .by_name
                .get(&1)
                .unwrap()
                .iter()
                .any(|x| x.kind == FactKind::SignedUse)
        );
        let d = present(&f, &facts, 0);
        assert_eq!(d.trust, Trust::Proven);
        assert_eq!(d.token, "i64");
    }

    #[test]
    fn lea_add_used_as_address_emits_ptr() {
        // n2 = n0 + 8; load *[n2]
        let stmts = vec![
            Stmt::Assign {
                dst: reg(2, Width::W64),
                value: Expr::Binary {
                    op: BinOp::Add,
                    lhs: Box::new(Expr::Reg(reg(0, Width::W64))),
                    rhs: Box::new(Expr::constant(8, Width::W64)),
                },
            },
            Stmt::Assign {
                dst: reg(3, Width::W8),
                value: Expr::Load {
                    addr: Box::new(Expr::Reg(reg(2, Width::W64))),
                    width: Width::W8,
                },
            },
        ];
        let names = vec![
            name(7, 0, Width::W64),
            name(6, 0, Width::W64),
            name(0, 1, Width::W64),
            name(1, 1, Width::W8),
        ];
        let f = func_with(stmts, names, vec![0]);
        let facts = collect(&f);
        assert!(check(&f, &facts).is_ok());
        let n2 = facts.by_name.get(&2).expect("n2");
        assert!(
            n2.iter()
                .any(|x| x.kind == FactKind::LoadedFrom(Width::W8)),
            "{n2:?}"
        );
        assert!(
            n2.iter().any(|x| x.kind == FactKind::PtrAddr),
            "expected PtrAddr on lea result: {n2:?}"
        );
        assert!(
            facts
                .by_name
                .get(&0)
                .unwrap()
                .iter()
                .any(|x| matches!(x.kind, FactKind::ArithWith(8)))
        );
        let d = present(&f, &facts, 2);
        assert_eq!(d.trust, Trust::Proven);
        assert!(d.token.starts_with("ptr"), "{}", d.token);
    }

    #[test]
    fn present_unproven_without_facts() {
        let f = func_with(
            vec![],
            vec![name(7, 0, Width::W64)],
            vec![0],
        );
        let facts = collect(&f);
        let d = present(&f, &facts, 0);
        assert_eq!(d.trust, Trust::Unproven);
        assert_eq!(d.token, "int64");
        assert!(d.render().contains("unproven"));
    }

    #[test]
    fn attach_sig_refines_pointer_param() {
        // live-in n0 used as load address → param becomes ptr.
        let stmts = vec![Stmt::Assign {
            dst: reg(1, Width::W32),
            value: Expr::Load {
                addr: Box::new(Expr::Reg(reg(0, Width::W64))),
                width: Width::W32,
            },
        }];
        let names = vec![name(7, 0, Width::W64), name(0, 1, Width::W32)];
        let f = func_with(stmts, names, vec![0]);
        let facts = collect(&f);
        let sig = sig::recover(&f);
        let mut table = TypeTable::new();
        let map = attach_sig_with_evidence(&f, &sig, &facts, &mut table);
        assert!(types::check(&table, Some(&map)).is_ok());
        assert!(!map.params.is_empty());
        let tid = map.params[0].1;
        assert!(
            matches!(table.get(tid), Some(types::TypeKind::Ptr { .. })),
            "got {:?}",
            table.get(tid)
        );
    }

    #[test]
    fn determinism_collect_twice() {
        let stmts = vec![Stmt::Assign {
            dst: reg(1, Width::W1),
            value: Expr::Binary {
                op: BinOp::Ult,
                lhs: Box::new(Expr::Reg(reg(0, Width::W32))),
                rhs: Box::new(Expr::constant(1, Width::W32)),
            },
        }];
        let names = vec![name(7, 0, Width::W32), name(0, 1, Width::W1)];
        let f = func_with(stmts, names, vec![0]);
        let a = collect(&f);
        let b = collect(&f);
        assert_eq!(a, b);
        assert_eq!(a.render(), b.render());
    }

    #[test]
    fn zero_extend_is_unsigned() {
        let stmts = vec![Stmt::Assign {
            dst: reg(1, Width::W64),
            value: Expr::unary(UnOp::ZeroExtend(Width::W64), Expr::Reg(reg(0, Width::W32))),
        }];
        let names = vec![name(7, 0, Width::W32), name(0, 1, Width::W64)];
        let f = func_with(stmts, names, vec![0]);
        let facts = collect(&f);
        assert!(check(&f, &facts).is_ok());
        assert!(
            facts
                .by_name
                .get(&0)
                .unwrap()
                .iter()
                .any(|x| x.kind == FactKind::UnsignedUse)
        );
        assert_eq!(present(&f, &facts, 0).token, "u32");
    }
}
