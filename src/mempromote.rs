//! Stack-slot promotion into SSA (DESIGN slice 11 lite).
//!
//! Given [`crate::irstack::StackFacts`] and the SSA they came from, classify
//! each evidence-backed slot as:
//! - [`Decision::Promote`] — clearly a local affine SP access with no
//!   address escape and no memory clobber barrier; safe to rewrite as an
//!   SSA value (named local).
//! - [`Decision::Candidate`] — still labeled `local_*` for rendering, but
//!   must stay as load/store (address-taken, call barrier, or unproven
//!   store).
//!
//! [`apply`] rewrites [`Decision::Promote`] slots with HSSA-lite: per-block
//! last-store forwarding and φ at control-flow merges when multiple defs
//! reach. Address-taken / barrier slots stay memory. Total: never panics.
//! Caps truncate / refuse the rewrite (return the input unchanged).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::ir::{Expr, Reg, Space, Stmt, UnOp, Width};
use crate::irssa::{Name, Phi, SsaFunction};
use crate::irstack::{AddrClass, SpFact, StackFacts};

/// Cap on promotion decisions emitted per function.
pub const MAX_PROMOTE: usize = 512;

/// Cap on additional SSA names [`apply`] may allocate.
pub const MAX_APPLY_NAMES: usize = 4096;

/// Arch cell base for promoted stack slots (well above GPRs / SP).
pub const PROMOTE_CELL_BASE: u16 = 0x8000;

/// Whether a stack slot may become an SSA-style local for pseudo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// No escape, no clobber barrier — promote to a named local.
    Promote,
    /// Keep as memory, but still expose `local_*` for naming.
    Candidate,
}

impl Decision {
    fn token(self) -> &'static str {
        match self {
            Decision::Promote => "promote",
            Decision::Candidate => "candidate",
        }
    }
}

/// One slot's promotion verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotDecision {
    pub base: i64,
    pub size: u64,
    /// Stable `local_<abs_base>` label for [`crate::pseudo`] hooks.
    pub name: String,
    pub decision: Decision,
    /// Short honesty tag for dumps (`ok`, `addr-taken`, `call-barrier`, …).
    pub reason: &'static str,
    /// Access sites copied from the justifying [`crate::irstack::StackSlot`].
    pub evidence: Vec<(u64, usize, i64, Width)>,
}

/// Per-function promotion table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromoteFacts {
    pub entry: u64,
    pub slots: Vec<SlotDecision>,
    pub has_call_barrier: bool,
    pub has_unproven_store: bool,
    pub promote_capped: bool,
}

impl PromoteFacts {
    /// Deterministic multi-line dump for CLI / tests.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let promoted = self
            .slots
            .iter()
            .filter(|s| s.decision == Decision::Promote)
            .count();
        let _ = writeln!(
            out,
            "; promote entry={:#x} slots={} promoted={} call_barrier={} unproven_store={}",
            self.entry,
            self.slots.len(),
            promoted,
            self.has_call_barrier,
            self.has_unproven_store
        );
        if self.promote_capped {
            out.push_str("; note: promote cap hit\n");
        }
        for s in &self.slots {
            let _ = writeln!(
                out,
                "  {} base={:+} size={} {} ({}) evidence={}",
                s.name,
                s.base,
                s.size,
                s.decision.token(),
                s.reason,
                s.evidence.len()
            );
        }
        out
    }

    /// Lookup by stack offset (same range rule as [`crate::irstack::slot_label`]).
    pub fn name_at(&self, offset: i64) -> Option<&str> {
        self.slots
            .iter()
            .find(|s| offset >= s.base && offset < s.base + s.size as i64)
            .map(|s| s.name.as_str())
    }

    /// Promoted locals only — ready for rewrite / namer.
    pub fn promoted_names(&self) -> BTreeMap<i64, String> {
        self.slots
            .iter()
            .filter(|s| s.decision == Decision::Promote)
            .map(|s| (s.base, s.name.clone()))
            .collect()
    }
}

/// Out-of-SSA variable id → `local_*` for affine stack-address SSA names
/// and (after [`apply`]) promoted slot value names.
/// Prefers [`PromoteFacts`] labels; falls back to [`irstack::slot_namer`].
/// Ready for [`crate::pseudo::render_with`]'s [`crate::pseudo::VarNamer`].
pub fn var_namer(
    ssa: &SsaFunction,
    stack: &StackFacts,
    promote: &PromoteFacts,
    var_of: &[u32],
) -> BTreeMap<u32, String> {
    let from_slots = crate::irstack::slot_namer(stack);
    let cell_locals = promoted_arch_cells(promote);
    let mut map = BTreeMap::new();
    for (ssa_id, &var_id) in var_of.iter().enumerate() {
        let ssa_id = ssa_id as u16;
        let name = match stack.name_facts.get(&ssa_id) {
            Some(SpFact::Affine(c)) => promote
                .name_at(*c)
                .map(|s| s.to_string())
                .or_else(|| from_slots.get(&ssa_id).cloned()),
            _ => from_slots.get(&ssa_id).cloned(),
        };
        let name = name.or_else(|| {
            let n = ssa.names.get(ssa_id as usize)?;
            if n.space != Space::Arch {
                return None;
            }
            cell_locals.get(&n.cell).cloned()
        });
        if let Some(name) = name {
            map.entry(var_id).or_insert(name);
        }
    }
    map
}

/// Arch cell → `local_*` for promoted slots, matching [`apply`]'s allocation.
fn promoted_arch_cells(facts: &PromoteFacts) -> BTreeMap<u16, String> {
    let mut out = BTreeMap::new();
    for (i, s) in facts
        .slots
        .iter()
        .filter(|s| s.decision == Decision::Promote)
        .enumerate()
    {
        let Some(cell) = PROMOTE_CELL_BASE.checked_add(i as u16) else {
            break;
        };
        out.insert(cell, s.name.clone());
    }
    out
}

/// Classify slots from `stack` against `f`. Total.
pub fn promote(f: &SsaFunction, stack: &StackFacts) -> PromoteFacts {
    let has_call_barrier = function_has_call(f);
    let has_unproven_store = has_unproven_store(f, stack);
    let escaped_offsets = address_taken_offsets(f, stack);

    let mut slots = Vec::new();
    let mut promote_capped = false;
    for slot in &stack.slots {
        if slots.len() >= MAX_PROMOTE {
            promote_capped = true;
            break;
        }
        let name = format!("local_{}", slot.base.unsigned_abs());
        let addr_taken = escaped_offsets.iter().any(|&off| {
            off >= slot.base && off < slot.base + slot.size as i64
        });

        let (decision, reason) = if has_call_barrier {
            (Decision::Candidate, "call-barrier")
        } else if has_unproven_store {
            (Decision::Candidate, "unproven-store")
        } else if stack.sp_escaped {
            (Decision::Candidate, "sp-escaped")
        } else if addr_taken {
            (Decision::Candidate, "addr-taken")
        } else {
            (Decision::Promote, "ok")
        };

        slots.push(SlotDecision {
            base: slot.base,
            size: slot.size,
            name,
            decision,
            reason,
            evidence: slot.evidence.clone(),
        });
    }

    PromoteFacts {
        entry: f.entry,
        slots,
        has_call_barrier,
        has_unproven_store,
        promote_capped,
    }
}

/// Rewrite [`Decision::Promote`] stack slots into SSA values (HSSA-lite).
///
/// - Same-block: last store forwards to later loads.
/// - Merges: insert φ when predecessor reaching defs disagree.
/// - [`Decision::Candidate`] / address-taken sites stay as load/store.
///
/// On cap refusal, empty promote set, or entry mismatch, returns `ssa`
/// unchanged (cloned). Never panics.
pub fn apply(ssa: &SsaFunction, facts: &PromoteFacts) -> SsaFunction {
    if facts.entry != ssa.entry {
        return ssa.clone();
    }
    let promoted: Vec<&SlotDecision> = facts
        .slots
        .iter()
        .filter(|s| s.decision == Decision::Promote)
        .collect();
    if promoted.is_empty() {
        return ssa.clone();
    }

    match apply_inner(ssa, &promoted) {
        Some(out) => out,
        None => ssa.clone(),
    }
}

/// Unproven classification on a Store site is a conservative clobber.
fn has_unproven_store(f: &SsaFunction, stack: &StackFacts) -> bool {
    for &(bva, ix, class, _) in &stack.accesses {
        if !matches!(class, AddrClass::Unproven) {
            continue;
        }
        let Some(block) = f.blocks.get(&bva) else {
            continue;
        };
        if matches!(block.stmts.get(ix), Some(Stmt::Store { .. })) {
            return true;
        }
    }
    false
}

fn function_has_call(f: &SsaFunction) -> bool {
    use crate::ir::BranchKind;
    for block in f.blocks.values() {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Branch {
                    kind: BranchKind::Call,
                    ..
                } => return true,
                Stmt::Intrinsic { name, .. } if *name == "callfx" => return true,
                _ => {}
            }
        }
    }
    false
}

/// Offsets whose Affine SSA names appear outside load/store address position.
fn address_taken_offsets(f: &SsaFunction, stack: &StackFacts) -> BTreeSet<i64> {
    // name id → affine offset, excluding the live SP-cell lineage used only
    // for frame setup: we still track every Affine fact, but SP-cell names
    // used in SP updates are filtered at use sites.
    let sp_cell = stack.sp_cell;
    let mut affine: BTreeMap<u16, i64> = BTreeMap::new();
    for (&id, fact) in &stack.name_facts {
        if let SpFact::Affine(c) = *fact {
            affine.insert(id, c);
        }
    }

    let mut escaped = BTreeSet::new();
    for block in f.blocks.values() {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Assign { dst, value } => {
                    let dst_is_sp = f
                        .names
                        .get(dst.num as usize)
                        .map(|n| n.space == crate::ir::Space::Arch && n.cell == sp_cell)
                        .unwrap_or(false);
                    // Frame setup: assigning into the SP cell is not escape.
                    if dst_is_sp {
                        continue;
                    }
                    // Any other def that consumes an Affine name materializes
                    // the address as a first-class value (LEA / mov of &local).
                    collect_affine_uses(value, &affine, &mut escaped);
                }
                Stmt::Store { addr, value } => {
                    // Address position: not escape.
                    let _ = addr;
                    collect_affine_uses(value, &affine, &mut escaped);
                }
                Stmt::Branch { cond, target, .. } => {
                    if let Some(c) = cond {
                        collect_affine_uses(c, &affine, &mut escaped);
                    }
                    collect_affine_uses(target, &affine, &mut escaped);
                }
                Stmt::Intrinsic { reads, .. } => {
                    for r in reads {
                        collect_affine_uses(r, &affine, &mut escaped);
                    }
                }
            }
        }
        // Load/Store address trees are never walked here — address position
        // is not escape. φ joining of address temps is deferred.
    }
    let _ = sp_cell;
    escaped
}

fn collect_affine_uses(e: &Expr, affine: &BTreeMap<u16, i64>, escaped: &mut BTreeSet<i64>) {
    match e {
        Expr::Reg(r) => {
            if let Some(&off) = affine.get(&r.num) {
                escaped.insert(off);
            }
        }
        Expr::Const { .. } => {}
        // Address position of a load is not escape (same rule as Store addr).
        Expr::Load { .. } => {}
        Expr::Unary { operand, .. } => collect_affine_uses(operand, affine, escaped),
        Expr::Binary { lhs, rhs, .. } => {
            collect_affine_uses(lhs, affine, escaped);
            collect_affine_uses(rhs, affine, escaped);
        }
    }
}

// ---------------------------------------------------------------------------
// apply — HSSA-lite rewrite
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reach {
    None,
    Def(u16),
    Conflict,
}

/// Join at a multi-predecessor merge: disagreeing defs (including
/// `None` vs `Def`) become [`Reach::Conflict`] so a φ is placed.
fn join_reach(a: Reach, b: Reach) -> Reach {
    match (a, b) {
        (Reach::None, Reach::None) => Reach::None,
        (Reach::Def(x), Reach::Def(y)) if x == y => Reach::Def(x),
        (Reach::Conflict, _) | (_, Reach::Conflict) => Reach::Conflict,
        (Reach::None, Reach::Def(_))
        | (Reach::Def(_), Reach::None)
        | (Reach::Def(_), Reach::Def(_)) => Reach::Conflict,
    }
}

struct SlotCell {
    /// Synthetic Arch cell for this promoted slot.
    cell: u16,
    /// Next version to allocate (definitions start at 1).
    next_ver: u32,
    /// Lazily allocated version-0 (uninitialized / entry) name.
    v0: Option<u16>,
}

struct ApplyState {
    out: SsaFunction,
    /// offset → slot machinery
    slots: BTreeMap<i64, SlotCell>,
    /// (block, stmt_ix) → promoted offset for that access
    sites: BTreeMap<(u64, usize), i64>,
    /// Store sites → preallocated def name
    store_defs: BTreeMap<(u64, usize), u16>,
    /// Names allocated this apply (for cap accounting)
    allocated: usize,
}

impl ApplyState {
    fn alloc_name(&mut self, cell: u16, version: u32, width: Width) -> Option<u16> {
        if self.allocated >= MAX_APPLY_NAMES {
            return None;
        }
        let id = u16::try_from(self.out.names.len()).ok()?;
        self.out.names.push(Name {
            space: Space::Arch,
            cell,
            version,
            width,
        });
        self.allocated += 1;
        Some(id)
    }

    fn alloc_store_def(&mut self, offset: i64) -> Option<u16> {
        let sc = self.slots.get_mut(&offset)?;
        let ver = sc.next_ver;
        sc.next_ver = sc.next_ver.saturating_add(1);
        if sc.next_ver == 0 {
            // version counter wrapped — refuse
            return None;
        }
        let cell = sc.cell;
        self.alloc_name(cell, ver, Width::W64)
    }

    fn ensure_v0(&mut self, offset: i64) -> Option<u16> {
        if let Some(id) = self.slots.get(&offset).and_then(|s| s.v0) {
            return Some(id);
        }
        let cell = self.slots.get(&offset)?.cell;
        let id = self.alloc_name(cell, 0, Width::W64)?;
        if let Some(sc) = self.slots.get_mut(&offset) {
            sc.v0 = Some(id);
        }
        Some(id)
    }
}

fn apply_inner(ssa: &SsaFunction, promoted: &[&SlotDecision]) -> Option<SsaFunction> {
    let mut sites: BTreeMap<(u64, usize), i64> = BTreeMap::new();
    for slot in promoted {
        for &(bva, ix, off, _) in &slot.evidence {
            // Evidence offset should match the slot base for our partition.
            let _ = off;
            sites.insert((bva, ix), slot.base);
        }
    }
    if sites.is_empty() {
        return Some(ssa.clone());
    }

    let mut next_idx = 0u16;
    let mut slot_cells: BTreeMap<i64, SlotCell> = BTreeMap::new();
    for slot in promoted {
        let cell = PROMOTE_CELL_BASE.checked_add(next_idx)?;
        next_idx = next_idx.saturating_add(1);
        if next_idx == 0 {
            return None;
        }
        slot_cells.insert(
            slot.base,
            SlotCell {
                cell,
                next_ver: 1,
                v0: None,
            },
        );
    }

    let mut st = ApplyState {
        out: ssa.clone(),
        slots: slot_cells,
        sites,
        store_defs: BTreeMap::new(),
        allocated: 0,
    };

    // Preallocate one SSA def name per promoted Store.
    let store_sites: Vec<(u64, usize, i64)> = st
        .sites
        .iter()
        .filter_map(|(&(b, i), &off)| {
            let block = st.out.blocks.get(&b)?;
            match block.stmts.get(i)? {
                Stmt::Store { .. } => Some((b, i, off)),
                _ => None,
            }
        })
        .collect();
    for (b, i, off) in store_sites {
        let id = st.alloc_store_def(off)?;
        st.store_defs.insert((b, i), id);
    }

    let preds = predecessors(&st.out);
    let block_vas: Vec<u64> = st.out.blocks.keys().copied().collect();

    // Last store def per block per slot (generation).
    let mut block_gen: BTreeMap<u64, BTreeMap<i64, u16>> = BTreeMap::new();
    for &b in &block_vas {
        let mut slot_gen = BTreeMap::new();
        let Some(block) = st.out.blocks.get(&b) else {
            continue;
        };
        for (i, stmt) in block.stmts.iter().enumerate() {
            if matches!(stmt, Stmt::Store { .. })
                && let Some(&off) = st.sites.get(&(b, i))
                && let Some(&id) = st.store_defs.get(&(b, i))
            {
                slot_gen.insert(off, id);
            }
        }
        block_gen.insert(b, slot_gen);
    }

    let slot_offs: Vec<i64> = st.slots.keys().copied().collect();

    // Reaching defs: in/out per block per slot. Insert φ on Conflict.
    let mut block_in: BTreeMap<u64, BTreeMap<i64, Reach>> = BTreeMap::new();
    let mut block_out: BTreeMap<u64, BTreeMap<i64, Reach>> = BTreeMap::new();
    let mut phi_dst: BTreeMap<(u64, i64), u16> = BTreeMap::new();

    for &b in &block_vas {
        let mut inn = BTreeMap::new();
        let mut outm = BTreeMap::new();
        for &off in &slot_offs {
            inn.insert(off, Reach::None);
            let reach = block_gen
                .get(&b)
                .and_then(|g| g.get(&off).copied())
                .map(Reach::Def)
                .unwrap_or(Reach::None);
            outm.insert(off, reach);
        }
        block_in.insert(b, inn);
        block_out.insert(b, outm);
    }

    // Iterate: join preds → allocate φ on Conflict → refresh outs.
    let max_iters = block_vas
        .len()
        .saturating_mul(slot_offs.len().saturating_add(1))
        .saturating_add(2);
    for _ in 0..max_iters {
        let mut changed = false;
        for &b in &block_vas {
            let plist = preds.get(&b).map(Vec::as_slice).unwrap_or(&[]);
            let entry_loop = b == st.out.entry && !plist.is_empty();
            let multi = plist.len() > 1 || entry_loop;
            for &off in &slot_offs {
                let joined = if !multi {
                    // Single pred (or no preds): forward.
                    match plist.first() {
                        Some(&p) => block_out
                            .get(&p)
                            .and_then(|m| m.get(&off).copied())
                            .unwrap_or(Reach::None),
                        None => Reach::None,
                    }
                } else {
                    let mut j = Reach::None;
                    let mut first = !entry_loop;
                    for &p in plist {
                        let pred_out = block_out
                            .get(&p)
                            .and_then(|m| m.get(&off).copied())
                            .unwrap_or(Reach::None);
                        if first {
                            j = pred_out;
                            first = false;
                        } else {
                            j = join_reach(j, pred_out);
                        }
                    }
                    j
                };

                let inn_val = if matches!(joined, Reach::Conflict) {
                    if let Some(&id) = phi_dst.get(&(b, off)) {
                        Reach::Def(id)
                    } else {
                        let cell = st.slots.get(&off)?.cell;
                        let ver = {
                            let sc = st.slots.get_mut(&off)?;
                            let v = sc.next_ver;
                            sc.next_ver = sc.next_ver.saturating_add(1);
                            if sc.next_ver == 0 {
                                return None;
                            }
                            v
                        };
                        let id = st.alloc_name(cell, ver, Width::W64)?;
                        phi_dst.insert((b, off), id);
                        changed = true;
                        Reach::Def(id)
                    }
                } else {
                    joined
                };

                let old_in = block_in.get(&b).and_then(|m| m.get(&off).copied());
                if old_in != Some(inn_val) {
                    block_in.get_mut(&b)?.insert(off, inn_val);
                    changed = true;
                }

                let from_gen = block_gen
                    .get(&b)
                    .and_then(|g| g.get(&off).copied())
                    .map(Reach::Def);
                let out_val = from_gen.unwrap_or(inn_val);
                let old_out = block_out.get(&b).and_then(|m| m.get(&off).copied());
                if old_out != Some(out_val) {
                    block_out.get_mut(&b)?.insert(off, out_val);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Materialize φ nodes with args from predecessor outs.
    for (&(b, off), &dst) in &phi_dst {
        let plist = preds.get(&b).map(Vec::as_slice).unwrap_or(&[]);
        let mut args: Vec<(Option<u64>, u16)> = Vec::new();
        if b == st.out.entry && !plist.is_empty() {
            let v0 = st.ensure_v0(off)?;
            args.push((None, v0));
        }
        for &p in plist {
            let id = match block_out
                .get(&p)
                .and_then(|m| m.get(&off).copied())
                .unwrap_or(Reach::None)
            {
                Reach::Def(id) => id,
                Reach::None | Reach::Conflict => st.ensure_v0(off)?,
            };
            args.push((Some(p), id));
        }
        if args.is_empty() {
            continue;
        }
        let block = st.out.blocks.get_mut(&b)?;
        block.phis.push(Phi { dst, args });
    }

    // Sort phis by (space, cell) ascending — required by irssa::check.
    for block in st.out.blocks.values_mut() {
        block.phis.sort_by_key(|phi| {
            st.out
                .names
                .get(phi.dst as usize)
                .map(|n| (n.space, n.cell))
                .unwrap_or((Space::Temp, u16::MAX))
        });
    }

    // Rewrite statements: Store → Assign; Load-of-slot → Reg.
    // Pre-resolve any v0 names a load might need so we do not borrow
    // `out.blocks` mutably while allocating into `out.names`.
    let mut load_v0_needed: BTreeSet<i64> = BTreeSet::new();
    for &b in &block_vas {
        let inn = block_in.get(&b).cloned().unwrap_or_default();
        let mut cur: BTreeMap<i64, u16> = BTreeMap::new();
        for (&off, &r) in &inn {
            if let Reach::Def(id) = r {
                cur.insert(off, id);
            }
        }
        let Some(block) = st.out.blocks.get(&b) else {
            continue;
        };
        for (i, stmt) in block.stmts.iter().enumerate() {
            match (stmt, st.sites.get(&(b, i)).copied()) {
                (Stmt::Store { .. }, Some(off)) => {
                    if let Some(&id) = st.store_defs.get(&(b, i)) {
                        cur.insert(off, id);
                    }
                }
                (Stmt::Assign { value: Expr::Load { .. }, .. }, Some(off))
                    if !cur.contains_key(&off) =>
                {
                    load_v0_needed.insert(off);
                }
                _ => {}
            }
        }
    }
    for off in load_v0_needed {
        st.ensure_v0(off)?;
    }
    // φ args may also have demanded v0 already via ensure_v0 above.

    for &b in &block_vas {
        let inn: BTreeMap<i64, Reach> = block_in.get(&b).cloned().unwrap_or_default();
        let mut cur: BTreeMap<i64, u16> = BTreeMap::new();
        for (&off, &r) in &inn {
            if let Reach::Def(id) = r {
                cur.insert(off, id);
            }
        }

        let stmts = st.out.blocks.get(&b)?.stmts.clone();
        let mut new_stmts = Vec::with_capacity(stmts.len());
        for (i, stmt) in stmts.into_iter().enumerate() {
            let site = st.sites.get(&(b, i)).copied();
            match (stmt, site) {
                (Stmt::Store { value, addr }, Some(off)) => {
                    let Some(&dst_id) = st.store_defs.get(&(b, i)) else {
                        new_stmts.push(Stmt::Store { addr, value });
                        continue;
                    };
                    let value = widen_to_w64(value);
                    cur.insert(off, dst_id);
                    new_stmts.push(Stmt::Assign {
                        dst: Reg {
                            space: Space::Arch,
                            num: dst_id,
                            width: Width::W64,
                        },
                        value,
                    });
                }
                (
                    Stmt::Assign {
                        dst,
                        value: Expr::Load { width, .. },
                    },
                    Some(off),
                ) => {
                    let id = match cur.get(&off).copied() {
                        Some(id) => id,
                        None => {
                            let id = st.slots.get(&off).and_then(|s| s.v0)?;
                            cur.insert(off, id);
                            id
                        }
                    };
                    new_stmts.push(Stmt::Assign {
                        dst,
                        value: Expr::reg(Reg {
                            space: Space::Arch,
                            num: id,
                            width,
                        }),
                    });
                }
                (other, _) => new_stmts.push(other),
            }
        }
        st.out.blocks.get_mut(&b)?.stmts = new_stmts;
    }

    // live_in = all version-0 names, ascending.
    st.out.live_in = st
        .out
        .names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.version == 0)
        .map(|(id, _)| id as u16)
        .collect();

    // Recompute partial (wider-than-def uses).
    let mut partial = BTreeSet::new();
    for (&va, block) in &st.out.blocks {
        for (i, stmt) in block.stmts.iter().enumerate() {
            walk_stmt_regs(stmt, &mut |r| {
                if let Some(n) = st.out.names.get(r.num as usize)
                    && r.width.bits() > n.width.bits()
                {
                    partial.insert((va, i));
                }
            });
        }
    }
    st.out.partial = partial.into_iter().collect();

    Some(st.out)
}

fn widen_to_w64(value: Expr) -> Expr {
    match value.width_of() {
        Some(Width::W64) | None => value,
        Some(w) if w.bits() < 64 => Expr::unary(UnOp::ZeroExtend(Width::W64), value),
        Some(_) => value,
    }
}

fn predecessors(f: &SsaFunction) -> BTreeMap<u64, Vec<u64>> {
    let mut preds: BTreeMap<u64, Vec<u64>> =
        f.blocks.keys().map(|&v| (v, Vec::new())).collect();
    for (&b, block) in &f.blocks {
        for &s in &block.successors {
            if let Some(p) = preds.get_mut(&s) {
                p.push(b);
            }
        }
    }
    for p in preds.values_mut() {
        p.sort_unstable();
        p.dedup();
    }
    preds
}

fn walk_stmt_regs(stmt: &Stmt, f: &mut dyn FnMut(Reg)) {
    let walk_expr = &mut |e: &Expr, f: &mut dyn FnMut(Reg)| {
        fn rec(e: &Expr, f: &mut dyn FnMut(Reg)) {
            match e {
                Expr::Const { .. } => {}
                Expr::Reg(r) => f(*r),
                Expr::Load { addr, .. } => rec(addr, f),
                Expr::Unary { operand, .. } => rec(operand, f),
                Expr::Binary { lhs, rhs, .. } => {
                    rec(lhs, f);
                    rec(rhs, f);
                }
            }
        }
        rec(e, f);
    };
    match stmt {
        Stmt::Assign { dst, value } => {
            walk_expr(value, f);
            f(*dst);
        }
        Stmt::Store { addr, value } => {
            walk_expr(addr, f);
            walk_expr(value, f);
        }
        Stmt::Branch { cond, target, .. } => {
            if let Some(c) = cond {
                walk_expr(c, f);
            }
            walk_expr(target, f);
        }
        Stmt::Intrinsic { writes, reads, .. } => {
            for r in reads {
                walk_expr(r, f);
            }
            for w in writes {
                f(*w);
            }
        }
    }
}

/// From-scratch sanity check. Total.
pub fn check(f: &SsaFunction, stack: &StackFacts, facts: &PromoteFacts) -> Result<(), String> {
    if facts.entry != f.entry {
        return Err("entry mismatch".into());
    }
    if facts.entry != stack.entry {
        return Err("stack entry mismatch".into());
    }
    if facts.slots.len() > MAX_PROMOTE {
        return Err("slots exceed MAX_PROMOTE".into());
    }
    if !facts.promote_capped && facts.slots.len() != stack.slots.len() {
        return Err(format!(
            "slot count {} != stack slots {}",
            facts.slots.len(),
            stack.slots.len()
        ));
    }
    for (i, s) in facts.slots.iter().enumerate() {
        if s.name.is_empty() {
            return Err(format!("empty name at slot {i}"));
        }
        if s.size == 0 {
            return Err(format!("zero-size slot {}", s.name));
        }
        if s.evidence.is_empty() {
            return Err(format!("no evidence for {}", s.name));
        }
        if s.decision == Decision::Promote {
            if facts.has_call_barrier {
                return Err(format!("{} promoted under call barrier", s.name));
            }
            if facts.has_unproven_store {
                return Err(format!("{} promoted under unproven store", s.name));
            }
            if s.reason != "ok" {
                return Err(format!("{} promote reason is not ok", s.name));
            }
        }
        // Base must match a stack slot when not capped mid-way.
        if i < stack.slots.len() && s.base != stack.slots[i].base {
            return Err(format!(
                "base mismatch at {i}: {} vs {}",
                s.base, stack.slots[i].base
            ));
        }
    }
    Ok(())
}

/// Check that [`apply`] removed promoted Store / slot-Load sites.
/// Candidate / non-promoted memory ops may remain. Total.
pub fn check_applied(
    before: &SsaFunction,
    after: &SsaFunction,
    facts: &PromoteFacts,
) -> Result<(), String> {
    if after.entry != before.entry {
        return Err("apply entry changed".into());
    }
    let promoted_sites: BTreeSet<(u64, usize)> = facts
        .slots
        .iter()
        .filter(|s| s.decision == Decision::Promote)
        .flat_map(|s| s.evidence.iter().map(|&(b, i, _, _)| (b, i)))
        .collect();
    if promoted_sites.is_empty() {
        return Ok(());
    }
    // After a successful rewrite, those indices should not still be
    // Store / Load-of-memory for the same shape. We only assert: no
    // Store remains at a promoted store site index (stmt may have been
    // replaced by Assign at the same index).
    for &(b, i) in &promoted_sites {
        let Some(block) = after.blocks.get(&b) else {
            return Err(format!("missing block {b:#x} after apply"));
        };
        let Some(stmt) = block.stmts.get(i) else {
            return Err(format!("missing stmt {b:#x}:{i} after apply"));
        };
        if matches!(stmt, Stmt::Store { .. }) {
            return Err(format!(
                "promoted store still present at {b:#x}:{i}"
            ));
        }
        if let Stmt::Assign {
            value: Expr::Load { .. },
            ..
        } = stmt
        {
            return Err(format!(
                "promoted load still present at {b:#x}:{i}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, BranchKind, Expr, Reg, Space, Stmt, Width};
    use crate::irssa::{Name, SsaBlock, SsaFunction};
    use crate::irstack;
    use crate::model::Arch;
    use std::collections::BTreeMap;

    fn name(cell: u16, version: u32, w: Width) -> Name {
        Name {
            space: Space::Arch,
            cell,
            version,
            width: w,
        }
    }

    fn arch_reg(num: u16, w: Width) -> Reg {
        Reg {
            space: Space::Arch,
            num,
            width: w,
        }
    }

    /// sp#0 live-in; sp#1 := sp#0 - 0x20; store [sp#1], rax#0
    fn mini_frame() -> SsaFunction {
        let names = vec![
            name(4, 0, Width::W64),
            name(4, 1, Width::W64),
            name(0, 0, Width::W64),
        ];
        let stmts = vec![
            Stmt::Assign {
                dst: arch_reg(1, Width::W64),
                value: Expr::binary(
                    BinOp::Sub,
                    Expr::reg(arch_reg(0, Width::W64)),
                    Expr::constant(0x20, Width::W64),
                ),
            },
            Stmt::Store {
                addr: Expr::reg(arch_reg(1, Width::W64)),
                value: Expr::reg(arch_reg(2, Width::W64)),
            },
        ];
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
            name: Some("frame".into()),
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![0, 2],
            partial: vec![],
        }
    }

    /// Same frame, but `rax := sp#1` (address taken into a GP reg).
    fn addr_taken_frame() -> SsaFunction {
        let mut f = mini_frame();
        // names: 3 = rax#1 after lea
        f.names.push(name(0, 1, Width::W64));
        let block = f.blocks.get_mut(&0x1000).unwrap();
        block.stmts.push(Stmt::Assign {
            dst: arch_reg(3, Width::W64),
            value: Expr::reg(arch_reg(1, Width::W64)),
        });
        f
    }

    /// Frame plus a call.
    fn frame_with_call() -> SsaFunction {
        let mut f = mini_frame();
        let block = f.blocks.get_mut(&0x1000).unwrap();
        block.stmts.push(Stmt::Branch {
            kind: BranchKind::Call,
            cond: None,
            target: Expr::constant(0x2000, Width::W64),
        });
        f
    }

    /// store [sp], rax; rbx := load [sp] — same block forwarding.
    fn spill_reload() -> SsaFunction {
        let names = vec![
            name(4, 0, Width::W64), // 0 rsp#0
            name(4, 1, Width::W64), // 1 rsp#1
            name(0, 0, Width::W64), // 2 rax#0
            name(1, 1, Width::W64), // 3 rbx#1
        ];
        let stmts = vec![
            Stmt::Assign {
                dst: arch_reg(1, Width::W64),
                value: Expr::binary(
                    BinOp::Sub,
                    Expr::reg(arch_reg(0, Width::W64)),
                    Expr::constant(0x20, Width::W64),
                ),
            },
            Stmt::Store {
                addr: Expr::reg(arch_reg(1, Width::W64)),
                value: Expr::reg(arch_reg(2, Width::W64)),
            },
            Stmt::Assign {
                dst: arch_reg(3, Width::W64),
                value: Expr::load(Expr::reg(arch_reg(1, Width::W64)), Width::W64),
            },
        ];
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x1000,
            SsaBlock {
                start: 0x1000,
                end: 0x1020,
                phis: vec![],
                stmts,
                successors: vec![],
                truncated: false,
            },
        );
        SsaFunction {
            entry: 0x1000,
            name: Some("spill".into()),
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![0, 2],
            partial: vec![],
        }
    }

    /// Diamond: both arms store, merge loads — needs φ.
    fn diamond_stores() -> SsaFunction {
        // names:
        // 0: rsp#0, 1: rsp#1, 2: rax#0 (cond / unused), 3: rbx#1 (load dst)
        let names = vec![
            name(4, 0, Width::W64),
            name(4, 1, Width::W64),
            name(0, 0, Width::W64),
            name(1, 1, Width::W64),
        ];
        let mut blocks = BTreeMap::new();
        // entry: sp := sp - 0x20; br cond → then / else
        blocks.insert(
            0x1000,
            SsaBlock {
                start: 0x1000,
                end: 0x1008,
                phis: vec![],
                stmts: vec![
                    Stmt::Assign {
                        dst: arch_reg(1, Width::W64),
                        value: Expr::binary(
                            BinOp::Sub,
                            Expr::reg(arch_reg(0, Width::W64)),
                            Expr::constant(0x20, Width::W64),
                        ),
                    },
                    Stmt::Branch {
                        kind: BranchKind::Jump,
                        cond: Some(Expr::constant(1, Width::W1)),
                        target: Expr::constant(0x1010, Width::W64),
                    },
                ],
                successors: vec![0x1010, 0x1020],
                truncated: false,
            },
        );
        blocks.insert(
            0x1010,
            SsaBlock {
                start: 0x1010,
                end: 0x1018,
                phis: vec![],
                stmts: vec![
                    Stmt::Store {
                        addr: Expr::reg(arch_reg(1, Width::W64)),
                        value: Expr::constant(1, Width::W64),
                    },
                    Stmt::Branch {
                        kind: BranchKind::Jump,
                        cond: None,
                        target: Expr::constant(0x1030, Width::W64),
                    },
                ],
                successors: vec![0x1030],
                truncated: false,
            },
        );
        blocks.insert(
            0x1020,
            SsaBlock {
                start: 0x1020,
                end: 0x1028,
                phis: vec![],
                stmts: vec![
                    Stmt::Store {
                        addr: Expr::reg(arch_reg(1, Width::W64)),
                        value: Expr::constant(2, Width::W64),
                    },
                    Stmt::Branch {
                        kind: BranchKind::Jump,
                        cond: None,
                        target: Expr::constant(0x1030, Width::W64),
                    },
                ],
                successors: vec![0x1030],
                truncated: false,
            },
        );
        blocks.insert(
            0x1030,
            SsaBlock {
                start: 0x1030,
                end: 0x1038,
                phis: vec![],
                stmts: vec![Stmt::Assign {
                    dst: arch_reg(3, Width::W64),
                    value: Expr::load(Expr::reg(arch_reg(1, Width::W64)), Width::W64),
                }],
                successors: vec![],
                truncated: false,
            },
        );
        SsaFunction {
            entry: 0x1000,
            name: Some("diamond".into()),
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![0, 2],
            partial: vec![],
        }
    }

    #[test]
    fn leaf_spill_promotes() {
        let f = mini_frame();
        let stack = irstack::analyze(&f);
        let facts = promote(&f, &stack);
        assert!(check(&f, &stack, &facts).is_ok(), "{:?}", check(&f, &stack, &facts));
        assert!(!facts.has_call_barrier);
        assert!(!facts.slots.is_empty());
        assert!(
            facts.slots.iter().any(|s| s.decision == Decision::Promote),
            "{}",
            facts.render()
        );
        assert!(facts.render().contains("promote"), "{}", facts.render());
        assert_eq!(facts.name_at(-0x20), Some("local_32"));
    }

    #[test]
    fn var_namer_feeds_pseudo_local_and_sig_header() {
        // Gate E1 wiring proof without a full binary lift: stack slots →
        // VarNamer → `local_*` in the body, and sig header as prototype.
        let f = mini_frame();
        let stack = irstack::analyze(&f);
        let facts = promote(&f, &stack);
        let (vars, _) = crate::irout::out_of_ssa(&f);
        let names = var_namer(&f, &stack, &facts, &vars.var_of);
        assert!(
            names.values().any(|n| n.starts_with("local_")),
            "{names:?}"
        );
        let tables = BTreeMap::new();
        let (root, _) = crate::irstruct::structure(&f, &tables);
        let sig = crate::sig::recover(&f);
        let header = sig.render_header();
        assert!(header.contains('('), "{header}");
        let namer = |v: u32| names.get(&v).cloned();
        let text = crate::pseudo::render_with_proto(&f, &root, &vars, &namer, Some(&header));
        assert!(text.contains("local_"), "{text}");
        assert!(text.contains(&format!("{header} {{")), "{text}");
    }

    #[test]
    fn address_taken_is_candidate() {
        let f = addr_taken_frame();
        let stack = irstack::analyze(&f);
        let facts = promote(&f, &stack);
        assert!(check(&f, &stack, &facts).is_ok());
        let slot = facts
            .slots
            .iter()
            .find(|s| s.base == -0x20)
            .expect("slot");
        assert_eq!(slot.decision, Decision::Candidate);
        assert_eq!(slot.reason, "addr-taken");
        assert_eq!(slot.name, "local_32");
    }

    #[test]
    fn call_barrier_blocks_promote() {
        let f = frame_with_call();
        let stack = irstack::analyze(&f);
        let facts = promote(&f, &stack);
        assert!(check(&f, &stack, &facts).is_ok());
        assert!(facts.has_call_barrier);
        assert!(facts
            .slots
            .iter()
            .all(|s| s.decision == Decision::Candidate));
        assert!(facts.render().contains("call-barrier"), "{}", facts.render());
    }

    #[test]
    fn apply_forwards_same_block_spill_reload() {
        let f = spill_reload();
        let stack = irstack::analyze(&f);
        let facts = promote(&f, &stack);
        assert!(check(&f, &stack, &facts).is_ok());
        assert!(
            facts.slots.iter().any(|s| s.decision == Decision::Promote),
            "{}",
            facts.render()
        );
        let out = apply(&f, &facts);
        assert!(
            crate::irssa::check(&out).is_ok(),
            "{:?}",
            crate::irssa::check(&out)
        );
        assert!(check_applied(&f, &out, &facts).is_ok());
        let block = &out.blocks[&0x1000];
        // stmt1 was Store → Assign to promoted local
        assert!(matches!(block.stmts[1], Stmt::Assign { .. }));
        // stmt2 was Load → Assign from that local (same name id as store def)
        let Stmt::Assign {
            value: Expr::Reg(r),
            ..
        } = &block.stmts[2]
        else {
            panic!("expected load→reg, got {:?}", block.stmts[2]);
        };
        let Stmt::Assign { dst, .. } = &block.stmts[1] else {
            panic!("expected store→assign");
        };
        assert_eq!(r.num, dst.num);
        // No memory ops left on the slot.
        assert!(!block.stmts.iter().any(|s| matches!(s, Stmt::Store { .. })));
    }

    #[test]
    fn apply_inserts_phi_at_diamond_merge() {
        let f = diamond_stores();
        let stack = irstack::analyze(&f);
        let facts = promote(&f, &stack);
        assert!(check(&f, &stack, &facts).is_ok());
        assert!(
            facts.slots.iter().any(|s| s.decision == Decision::Promote),
            "{}",
            facts.render()
        );
        let out = apply(&f, &facts);
        assert!(
            crate::irssa::check(&out).is_ok(),
            "{:?}",
            crate::irssa::check(&out)
        );
        assert!(check_applied(&f, &out, &facts).is_ok());
        let merge = &out.blocks[&0x1030];
        assert!(
            !merge.phis.is_empty(),
            "expected φ at merge, got {}",
            crate::irssa::render(&out)
        );
        let Stmt::Assign {
            value: Expr::Reg(r),
            ..
        } = &merge.stmts[0]
        else {
            panic!("expected promoted load, got {:?}", merge.stmts[0]);
        };
        assert!(
            merge.phis.iter().any(|p| p.dst == r.num),
            "load should use φ dst"
        );
    }

    #[test]
    fn apply_skips_address_taken() {
        let f = addr_taken_frame();
        let stack = irstack::analyze(&f);
        let facts = promote(&f, &stack);
        let out = apply(&f, &facts);
        // Candidate only — identity rewrite.
        assert_eq!(out.blocks[&0x1000].stmts.len(), f.blocks[&0x1000].stmts.len());
        assert!(matches!(
            out.blocks[&0x1000].stmts[1],
            Stmt::Store { .. }
        ));
    }

    #[test]
    fn apply_noop_when_nothing_promoted() {
        let f = frame_with_call();
        let stack = irstack::analyze(&f);
        let facts = promote(&f, &stack);
        let out = apply(&f, &facts);
        assert_eq!(out, f);
    }
}
