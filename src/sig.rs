//! Callee-side signature recovery (DESIGN slice 12), with hooks for
//! stack-arg bytes (irstack), caller-side return confirmation (slice 13
//! helper), and symbol-derived arity (slice 14 demangle hints).
//!
//! After callfx + SSA opt, version-0 cells in [`SsaFunction::live_in`] are
//! exactly the read-before-write set. Intersect that set with the ABI
//! argument-register sequence; the highest witnessed index (per int /
//! float class) is the arity under the prefix rule. The ABI primary
//! return cell is recorded as [`Provenance::AbiAssumed`] until
//! [`confirm_returns`] upgrades it from caller evidence.
//!
//! Total: never panics. Caps truncate rather than grow without bound.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::cfg::CallTarget;
use crate::ir::{BranchKind, Expr, Space, Stmt, Width};
use crate::irssa::SsaFunction;
use crate::irstack::{AddrClass, StackFacts};
use crate::model::Arch;

/// Cap on claimed parameters per function (int + float combined).
pub const MAX_PARAMS: usize = 64;
/// Cap on claimed return cells per function.
pub const MAX_RETURNS: usize = 8;
/// Cap on caller SSAs examined by [`confirm_returns`] /
/// [`callers_of`].
pub const MAX_RETURN_CALLERS: usize = 64;

/// How a signature fact was obtained. Ranked like [`crate::funcs::Source`]:
/// symbol-derived (slice 14) > dataflow-proven > ABI-assumed > heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Provenance {
    /// Demangled / metadata prototypes (slice 14).
    SymbolDerived,
    /// Witnessed by a live-in SSA name of an ABI argument cell, or by a
    /// caller reading the ABI return cell after the call.
    DataflowProven,
    /// Taken from the ABI table without a callee-side witness.
    AbiAssumed,
    /// Softened or filled by a heuristic rule.
    Heuristic,
}

impl Provenance {
    fn token(self) -> &'static str {
        match self {
            Provenance::SymbolDerived => "symbol",
            Provenance::DataflowProven => "dataflow",
            Provenance::AbiAssumed => "abi",
            Provenance::Heuristic => "heuristic",
        }
    }
}

/// Integer vs floating / vector ABI argument class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    Int,
    Float,
}

impl ParamKind {
    fn token(self) -> &'static str {
        match self {
            ParamKind::Int => "int",
            ParamKind::Float => "float",
        }
    }
}

/// One claimed parameter with its witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    /// Zero-based index in the ABI sequence for [`Self::kind`].
    pub index: u16,
    pub kind: ParamKind,
    /// Architectural cell number (`Space::Arch`).
    pub cell: u16,
    /// Live-in SSA name id that witnesses this cell, when provenance is
    /// [`Provenance::DataflowProven`].
    pub name_id: Option<u16>,
    pub width: Width,
    pub provenance: Provenance,
}

/// One claimed return location (ABI cell at this slice).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReturnCell {
    pub cell: u16,
    pub width: Width,
    pub provenance: Provenance,
}

/// Per-function callee-side signature facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    pub entry: u64,
    pub name: Option<String>,
    pub arch: Arch,
    /// Parameters in display order: all int args by index, then float.
    pub params: Vec<Param>,
    pub returns: Vec<ReturnCell>,
    /// Stack-argument bytes witnessed by positive-offset entry-SP loads
    /// (above the return address on x86-64; ≥0 on AArch64).
    pub stack_bytes: u64,
    pub params_capped: bool,
    pub returns_capped: bool,
}

impl Signature {
    /// `f(a, b)`-style header for listing / pseudocode hooks.
    pub fn render_header(&self) -> String {
        render_header(self)
    }

    /// Deterministic multi-line dump for CLI / tests.
    pub fn render(&self) -> String {
        render(self)
    }
}

/// ABI argument cell sequences and the primary return cell for `arch`.
struct AbiSig {
    int_args: &'static [u16],
    float_args: &'static [u16],
    /// Primary integer return cell (rax / x0). Empty when unknown.
    returns: &'static [u16],
}

fn abi_sig(arch: Arch) -> Option<AbiSig> {
    match arch {
        // SysV AMD64 integer args; Win64 is a subset so the union stays
        // sound for PE. XMM float args are not yet cells in the x86
        // lifter — float sequence is empty until that lands.
        Arch::X86_64 => Some(AbiSig {
            int_args: &[7, 6, 2, 1, 8, 9], // rdi rsi rdx rcx r8 r9
            float_args: &[],
            returns: &[0], // rax
        }),
        Arch::Aarch64 => Some(AbiSig {
            int_args: &[0, 1, 2, 3, 4, 5, 6, 7], // x0–x7
            // AAPCS64 FP/vector args in the low halves of v0–v7.
            float_args: &[32, 33, 34, 35, 36, 37, 38, 39],
            returns: &[0], // x0
        }),
        Arch::Other => None,
    }
}

/// Lowest entry-SP offset that can hold a stack argument (bytes).
fn stack_arg_floor(arch: Arch) -> i64 {
    match arch {
        // Return address occupies [rsp+0, rsp+8).
        Arch::X86_64 => 8,
        // LR is in x30; stack args begin at the entry SP.
        Arch::Aarch64 => 0,
        Arch::Other => 0,
    }
}

/// Recover callee-side signature facts from optimized SSA. Runs
/// [`irstack::analyze`] for stack-arg bytes. Total.
pub fn recover(f: &SsaFunction) -> Signature {
    let stack = crate::irstack::analyze(f);
    recover_with_stack(f, &stack)
}

/// Like [`recover`], but reuses an existing [`StackFacts`] dump (e.g.
/// when `redump --stack` already computed it).
pub fn recover_with_stack(f: &SsaFunction, stack: &StackFacts) -> Signature {
    let mut sig = Signature {
        entry: f.entry,
        name: f.name.clone(),
        arch: f.arch,
        params: Vec::new(),
        returns: Vec::new(),
        stack_bytes: 0,
        params_capped: false,
        returns_capped: false,
    };

    let Some(abi) = abi_sig(f.arch) else {
        sig.stack_bytes = stack_arg_bytes(f, stack);
        return sig;
    };

    // live_in name id → (cell, width) for Arch cells only.
    let mut live_cells: BTreeMap<u16, (u16, Width)> = BTreeMap::new();
    for &id in &f.live_in {
        let Some(n) = f.names.get(id as usize) else {
            continue;
        };
        if n.space != Space::Arch {
            continue;
        }
        // First witness wins (live_in is ascending; deterministic).
        live_cells.entry(n.cell).or_insert((id, n.width));
    }

    claim_class(
        &mut sig.params,
        &mut sig.params_capped,
        ParamKind::Int,
        abi.int_args,
        &live_cells,
    );
    claim_class(
        &mut sig.params,
        &mut sig.params_capped,
        ParamKind::Float,
        abi.float_args,
        &live_cells,
    );

    sig.stack_bytes = stack_arg_bytes(f, stack);

    for &cell in abi.returns {
        if sig.returns.len() >= MAX_RETURNS {
            sig.returns_capped = true;
            break;
        }
        sig.returns.push(ReturnCell {
            cell,
            width: Width::W64,
            provenance: Provenance::AbiAssumed,
        });
    }

    if let Some(arity) = symbol_arity_hint(f.name.as_deref()) {
        apply_symbol_arity(&mut sig, arity, &abi);
    }

    sig
}

/// Bytes of stack-argument area witnessed by loads at entry-SP offsets
/// at or above [`stack_arg_floor`]. Stores (spills) do not count.
fn stack_arg_bytes(f: &SsaFunction, facts: &StackFacts) -> u64 {
    let floor = stack_arg_floor(f.arch);
    let mut max_end: u64 = 0;
    for &(bva, ix, class, width) in &facts.accesses {
        let AddrClass::StackOff(off) = class else {
            continue;
        };
        if off < floor {
            continue;
        }
        let Some(block) = f.blocks.get(&bva) else {
            continue;
        };
        let Some(stmt) = block.stmts.get(ix) else {
            continue;
        };
        // Only loads read incoming stack args; stores are spills/homes.
        if !matches!(
            stmt,
            Stmt::Assign {
                value: Expr::Load { .. },
                ..
            }
        ) {
            continue;
        }
        let size = (width.bits() / 8) as u64;
        let end = (off as u64).saturating_add(size);
        max_end = max_end.max(end);
    }
    max_end.saturating_sub(floor as u64)
}

/// Prefix-rule claim for one ABI argument class.
fn claim_class(
    params: &mut Vec<Param>,
    capped: &mut bool,
    kind: ParamKind,
    sequence: &[u16],
    live_cells: &BTreeMap<u16, (u16, Width)>,
) {
    // Highest witnessed index in this class; None → no params of this kind.
    let mut highest: Option<usize> = None;
    for (i, &cell) in sequence.iter().enumerate() {
        if live_cells.contains_key(&cell) {
            highest = Some(i);
        }
    }
    let Some(hi) = highest else {
        return;
    };
    for (i, &cell) in sequence.iter().enumerate().take(hi + 1) {
        if params.len() >= MAX_PARAMS {
            *capped = true;
            return;
        }
        match live_cells.get(&cell) {
            Some(&(name_id, width)) => params.push(Param {
                index: i as u16,
                kind,
                cell,
                name_id: Some(name_id),
                width,
                provenance: Provenance::DataflowProven,
            }),
            None => params.push(Param {
                index: i as u16,
                kind,
                cell,
                name_id: None,
                width: Width::W64,
                // Gap under the prefix rule: arity requires this slot,
                // but the callee never read it.
                provenance: Provenance::AbiAssumed,
            }),
        }
    }
}

/// Parameter count from a demangled / prototype-shaped name, when the
/// trailing `(…)` list is parseable. Tries the string as-is, then
/// [`crate::cxxdemangle::try_demangle`].
pub fn symbol_arity_hint(name: Option<&str>) -> Option<usize> {
    let name = name?;
    if let Some(n) = prototype_arity(name) {
        return Some(n);
    }
    crate::cxxdemangle::try_demangle(name).and_then(|d| prototype_arity(&d))
}

/// Count top-level parameters in a C++-style `name(type, …)` prototype.
/// Returns `None` when no trailing parameter list is present.
fn prototype_arity(s: &str) -> Option<usize> {
    let s = strip_trailing_qualifiers(s.trim());
    let bytes = s.as_bytes();
    if bytes.last() != Some(&b')') {
        return None;
    }
    let mut depth = 0i32;
    let mut open = None;
    for (i, &b) in bytes.iter().enumerate().rev() {
        match b {
            b')' => depth += 1,
            b'(' => {
                depth -= 1;
                if depth == 0 {
                    open = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let start = open? + 1;
    let end = bytes.len() - 1;
    let inner = s[start..end].trim();
    if inner.is_empty() || inner == "void" {
        return Some(0);
    }
    let mut count = 1usize;
    depth = 0;
    for &b in inner.as_bytes() {
        match b {
            b'(' | b'<' | b'[' => depth += 1,
            b')' | b'>' | b']' => depth -= 1,
            b',' if depth == 0 => {
                count += 1;
                if count > MAX_PARAMS {
                    return None;
                }
            }
            _ => {}
        }
    }
    Some(count)
}

fn strip_trailing_qualifiers(s: &str) -> &str {
    let mut s = s.trim_end();
    loop {
        let next = s
            .strip_suffix("const")
            .or_else(|| s.strip_suffix("volatile"))
            .or_else(|| s.strip_suffix("noexcept"))
            .or_else(|| s.strip_suffix("&&"))
            .or_else(|| s.strip_suffix('&'))
            .map(str::trim_end);
        match next {
            Some(n) if n.len() < s.len() => s = n,
            _ => break,
        }
    }
    s.trim_end()
}

/// Prefer symbol arity when it does not contradict dataflow-proven
/// register params and fits the ABI (+ stack) capacity.
fn apply_symbol_arity(sig: &mut Signature, arity: usize, abi: &AbiSig) {
    if arity > MAX_PARAMS {
        return;
    }
    let stack_slots = (sig.stack_bytes / 8) as usize;
    let capacity = abi.int_args.len().saturating_add(stack_slots);
    if arity > capacity {
        return;
    }

    // Highest dataflow-proven int index + 1; symbol must not shrink below.
    let min_keep = sig
        .params
        .iter()
        .filter(|p| p.kind == ParamKind::Int && p.provenance == Provenance::DataflowProven)
        .map(|p| p.index as usize + 1)
        .max()
        .unwrap_or(0);
    if arity < min_keep {
        return;
    }

    let int_params: Vec<Param> = sig
        .params
        .iter()
        .filter(|p| p.kind == ParamKind::Int)
        .cloned()
        .collect();
    let float_params: Vec<Param> = sig
        .params
        .iter()
        .filter(|p| p.kind == ParamKind::Float)
        .cloned()
        .collect();

    // Register-backed symbol arity only fills the int ABI sequence; stack
    // bytes already account for overflow beyond registers.
    let reg_arity = arity.min(abi.int_args.len());
    let mut new_ints = Vec::new();
    for i in 0..reg_arity {
        if new_ints.len() >= MAX_PARAMS {
            sig.params_capped = true;
            break;
        }
        let cell = abi.int_args[i];
        if let Some(existing) = int_params.iter().find(|p| p.index as usize == i) {
            let mut p = existing.clone();
            // Symbol confirms this slot; keep the live-in witness if any.
            p.provenance = Provenance::SymbolDerived;
            new_ints.push(p);
        } else {
            new_ints.push(Param {
                index: i as u16,
                kind: ParamKind::Int,
                cell,
                name_id: None,
                width: Width::W64,
                provenance: Provenance::SymbolDerived,
            });
        }
    }

    sig.params = new_ints;
    for p in float_params {
        if sig.params.len() >= MAX_PARAMS {
            sig.params_capped = true;
            break;
        }
        sig.params.push(p);
    }
}

/// Invert a caller→callees [`crate::cfg::Program::call_graph`] into the
/// list of callers of `callee` (direct [`CallTarget::Function`] only).
/// Ascending, capped at [`MAX_RETURN_CALLERS`].
pub fn callers_of(call_graph: &BTreeMap<u64, BTreeSet<CallTarget>>, callee: u64) -> Vec<u64> {
    let mut out = Vec::new();
    for (&caller, targets) in call_graph {
        if targets.contains(&CallTarget::Function(callee)) {
            out.push(caller);
            if out.len() >= MAX_RETURN_CALLERS {
                break;
            }
        }
    }
    out
}

/// True when `caller` contains a direct call to `callee_entry` after
/// which an ABI return cell is read before being redefined. Understands
/// optional [`crate::callfx`] intrinsics. Total.
pub fn caller_reads_return(caller: &SsaFunction, callee_entry: u64) -> bool {
    let Some(abi) = abi_sig(caller.arch) else {
        return false;
    };
    let ret_cells: BTreeSet<u16> = abi.returns.iter().copied().collect();
    if ret_cells.is_empty() {
        return false;
    }

    for block in caller.blocks.values() {
        for (i, stmt) in block.stmts.iter().enumerate() {
            if !is_direct_call_to(stmt, callee_entry) {
                continue;
            }
            if scans_return_use(caller, &block.stmts[i + 1..], &ret_cells) {
                return true;
            }
            // One-block fallthrough: CFG ends a call block with the
            // successor holding "on return" code when callfx split it.
            for &succ in &block.successors {
                let Some(sb) = caller.blocks.get(&succ) else {
                    continue;
                };
                if scans_return_use(caller, &sb.stmts, &ret_cells) {
                    return true;
                }
            }
        }
    }
    false
}

fn is_direct_call_to(stmt: &Stmt, callee: u64) -> bool {
    matches!(
        stmt,
        Stmt::Branch {
            kind: BranchKind::Call,
            target: Expr::Const { value, .. },
            ..
        } if *value == callee
    )
}

/// After a call (and optional callfx), does any stmt use a return cell
/// before redefining it?
fn scans_return_use(caller: &SsaFunction, stmts: &[Stmt], ret_cells: &BTreeSet<u16>) -> bool {
    let mut watch: BTreeSet<u16> = BTreeSet::new();
    let mut saw_callfx = false;

    for stmt in stmts {
        match stmt {
            Stmt::Intrinsic {
                name,
                writes,
                ..
            } if *name == crate::callfx::EFFECT_NAME => {
                saw_callfx = true;
                for w in writes {
                    if let Some(n) = caller.names.get(w.num as usize)
                        && n.space == Space::Arch
                        && ret_cells.contains(&n.cell)
                    {
                        watch.insert(w.num);
                    }
                }
            }
            Stmt::Assign { dst, value } => {
                // Only names produced by callfx (or a post-call ret-cell
                // def when callfx is absent) count — never a stale
                // pre-call live-in of rax/x0.
                if expr_uses_names(value, &watch) {
                    return true;
                }
                if let Some(n) = caller.names.get(dst.num as usize)
                    && n.space == Space::Arch
                    && ret_cells.contains(&n.cell)
                {
                    watch.retain(|id| {
                        caller
                            .names
                            .get(*id as usize)
                            .map(|wn| wn.cell != n.cell)
                            .unwrap_or(true)
                    });
                    if !saw_callfx {
                        watch.insert(dst.num);
                    }
                }
            }
            Stmt::Store { addr, value } => {
                if expr_uses_names(addr, &watch) || expr_uses_names(value, &watch) {
                    return true;
                }
            }
            Stmt::Branch { cond, target, .. } => {
                if let Some(c) = cond
                    && expr_uses_names(c, &watch)
                {
                    return true;
                }
                if expr_uses_names(target, &watch) {
                    return true;
                }
            }
            Stmt::Intrinsic { reads, writes, .. } => {
                for r in reads {
                    if expr_uses_names(r, &watch) {
                        return true;
                    }
                }
                for w in writes {
                    if let Some(n) = caller.names.get(w.num as usize)
                        && n.space == Space::Arch
                        && ret_cells.contains(&n.cell)
                    {
                        watch.retain(|id| {
                            caller
                                .names
                                .get(*id as usize)
                                .map(|wn| wn.cell != n.cell)
                                .unwrap_or(true)
                        });
                    }
                }
            }
        }
    }
    false
}

fn expr_uses_names(e: &Expr, names: &BTreeSet<u16>) -> bool {
    if names.is_empty() {
        return false;
    }
    match e {
        Expr::Reg(r) => names.contains(&r.num),
        Expr::Const { .. } => false,
        Expr::Load { addr, .. } => expr_uses_names(addr, names),
        Expr::Unary { operand, .. } => expr_uses_names(operand, names),
        Expr::Binary { lhs, rhs, .. } => {
            expr_uses_names(lhs, names) || expr_uses_names(rhs, names)
        }
    }
}

/// Upgrade [`Provenance::AbiAssumed`] return cells to
/// [`Provenance::DataflowProven`] when any examined caller reads the
/// return register after calling `sig.entry`. Caps at
/// [`MAX_RETURN_CALLERS`]. Leaves [`Provenance::SymbolDerived`] alone.
///
/// Call-graph wiring stays out of [`recover`]: `redump` (or another
/// driver) gathers caller SSAs via [`callers_of`] + lift and passes them
/// here.
pub fn confirm_returns(sig: &mut Signature, callers: &[&SsaFunction]) {
    let mut confirmed = false;
    for caller in callers.iter().take(MAX_RETURN_CALLERS) {
        if caller_reads_return(caller, sig.entry) {
            confirmed = true;
            break;
        }
    }
    if !confirmed {
        return;
    }
    for r in &mut sig.returns {
        if r.provenance == Provenance::AbiAssumed {
            r.provenance = Provenance::DataflowProven;
        }
    }
}

/// From-scratch sanity check. Total.
pub fn check(f: &SsaFunction, sig: &Signature) -> Result<(), String> {
    if sig.entry != f.entry {
        return Err("entry mismatch".into());
    }
    if sig.arch != f.arch {
        return Err("arch mismatch".into());
    }
    if sig.params.len() > MAX_PARAMS {
        return Err("params exceed MAX_PARAMS".into());
    }
    if sig.returns.len() > MAX_RETURNS {
        return Err("returns exceed MAX_RETURNS".into());
    }

    let Some(abi) = abi_sig(f.arch) else {
        if !sig.params.is_empty() || !sig.returns.is_empty() {
            return Err("facts claimed without an ABI table".into());
        }
        return Ok(());
    };

    let live: BTreeMap<u16, u16> = f
        .live_in
        .iter()
        .filter_map(|&id| {
            let n = f.names.get(id as usize)?;
            (n.space == Space::Arch).then_some((n.cell, id))
        })
        .collect();

    // Params must stay inside the ABI candidate set; DataflowProven must
    // cite a live-in name of that cell.
    for p in &sig.params {
        let seq = match p.kind {
            ParamKind::Int => abi.int_args,
            ParamKind::Float => abi.float_args,
        };
        let Some(&cell_at) = seq.get(p.index as usize) else {
            return Err(format!(
                "param {} {:?} index out of ABI sequence",
                p.index, p.kind
            ));
        };
        if cell_at != p.cell {
            return Err(format!(
                "param {} {:?} cell {} != ABI cell {}",
                p.index, p.kind, p.cell, cell_at
            ));
        }
        match p.provenance {
            Provenance::DataflowProven => {
                let Some(id) = p.name_id else {
                    return Err("DataflowProven param without name_id".into());
                };
                match live.get(&p.cell) {
                    Some(&live_id) if live_id == id => {}
                    Some(_) => {
                        return Err(format!(
                            "name_id {id} is not the live-in for cell {}",
                            p.cell
                        ))
                    }
                    None => {
                        return Err(format!("DataflowProven cell {} not in live_in", p.cell))
                    }
                }
                if f.names.get(id as usize).is_none() {
                    return Err(format!("unknown name id {id}"));
                }
            }
            Provenance::AbiAssumed | Provenance::Heuristic | Provenance::SymbolDerived => {
                if p.name_id.is_some() {
                    // Allowed but unused at this slice; ignore.
                }
            }
        }
    }

    // Prefix density per kind: indices 0..n-1 contiguous, no holes.
    for kind in [ParamKind::Int, ParamKind::Float] {
        let idxs: Vec<u16> = sig
            .params
            .iter()
            .filter(|p| p.kind == kind)
            .map(|p| p.index)
            .collect();
        for (expect, &got) in idxs.iter().enumerate() {
            if got as usize != expect {
                return Err(format!(
                    "{:?} params not a dense prefix: expected index {expect}, got {got}",
                    kind
                ));
            }
        }
    }

    for r in &sig.returns {
        if !abi.returns.contains(&r.cell) {
            return Err(format!("return cell {} outside ABI returns", r.cell));
        }
    }

    Ok(())
}

/// `name(a, b)` header. Letter names wrap to `a0`, `a1`, … after `z`.
/// When [`Signature::stack_bytes`] is non-zero, appends `; stack=N`.
pub fn render_header(sig: &Signature) -> String {
    let name = match sig.name.as_deref() {
        Some(n) if !n.is_empty() => n.to_string(),
        _ => format!("sub_{:x}", sig.entry),
    };
    let args = (0..sig.params.len())
        .map(arg_letter)
        .collect::<Vec<_>>()
        .join(", ");
    if sig.stack_bytes == 0 {
        format!("{name}({args})")
    } else if args.is_empty() {
        format!("{name}(stack={})", sig.stack_bytes)
    } else {
        format!("{name}({args}; stack={})", sig.stack_bytes)
    }
}

fn arg_letter(i: usize) -> String {
    if i < 26 {
        ((b'a' + i as u8) as char).to_string()
    } else {
        format!("a{}", i - 26)
    }
}

/// Multi-line dump mirroring [`crate::irstack::StackFacts::render`].
pub fn render(sig: &Signature) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "; sig entry={:#x} params={} returns={} stack_bytes={} header={}",
        sig.entry,
        sig.params.len(),
        sig.returns.len(),
        sig.stack_bytes,
        render_header(sig)
    );
    if sig.params_capped {
        out.push_str("; note: param cap hit\n");
    }
    if sig.returns_capped {
        out.push_str("; note: return cap hit\n");
    }
    for p in &sig.params {
        let witness = match p.name_id {
            Some(id) => format!("name#{id}"),
            None => "-".into(),
        };
        let _ = writeln!(
            out,
            "  param[{}] {} cell={} {} {} .{} {}",
            p.index,
            p.kind.token(),
            p.cell,
            cell_spell(sig.arch, p.cell, p.width),
            witness,
            p.width.bits() / 8,
            p.provenance.token()
        );
    }
    for (i, r) in sig.returns.iter().enumerate() {
        let _ = writeln!(
            out,
            "  return[{i}] cell={} {} .{} {}",
            r.cell,
            cell_spell(sig.arch, r.cell, r.width),
            r.width.bits() / 8,
            r.provenance.token()
        );
    }
    out
}

fn cell_spell(arch: Arch, cell: u16, width: Width) -> String {
    match arch {
        Arch::Aarch64 => crate::aarch64_lift::reg_name(cell, width),
        Arch::X86_64 | Arch::Other => crate::x86_lift::reg_name(cell, width),
    }
    .unwrap_or_else(|| format!("r{cell}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, Expr, Reg, Space, Stmt, Width};
    use crate::irssa::{Name, SsaBlock, SsaFunction};
    use crate::model::Arch;
    use std::collections::{BTreeMap, BTreeSet};

    fn name(cell: u16, version: u32, w: Width) -> Name {
        Name {
            space: Space::Arch,
            cell,
            version,
            width: w,
        }
    }

    /// Hand-built: live-in rdi#0, rsi#0; body reads them into temps.
    fn two_int_args() -> SsaFunction {
        // names: 0=rdi#0, 1=rsi#0, 2=rax#1 (defined)
        let names = vec![
            name(7, 0, Width::W64),
            name(6, 0, Width::W64),
            name(0, 1, Width::W64),
        ];
        let stmts = vec![Stmt::Assign {
            dst: Reg {
                space: Space::Arch,
                num: 2,
                width: Width::W64,
            },
            value: Expr::reg(Reg {
                space: Space::Arch,
                num: 0,
                width: Width::W64,
            }),
        }];
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
            name: Some("add".into()),
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![0, 1],
            partial: vec![],
        }
    }

    /// Live-in rdi and rdx but not rsi → prefix fills rsi as AbiAssumed.
    fn gap_middle_arg() -> SsaFunction {
        let names = vec![
            name(7, 0, Width::W64), // 0: rdi
            name(2, 0, Width::W64), // 1: rdx
        ];
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x2000,
            SsaBlock {
                start: 0x2000,
                end: 0x2008,
                phis: vec![],
                stmts: vec![],
                successors: vec![],
                truncated: false,
            },
        );
        SsaFunction {
            entry: 0x2000,
            name: None,
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![0, 1],
            partial: vec![],
        }
    }

    #[test]
    fn two_params_render_as_f_a_b() {
        let f = two_int_args();
        let sig = recover(&f);
        assert!(check(&f, &sig).is_ok(), "{:?}", check(&f, &sig));
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0].cell, 7);
        assert_eq!(sig.params[1].cell, 6);
        assert_eq!(sig.params[0].provenance, Provenance::DataflowProven);
        assert_eq!(render_header(&sig), "add(a, b)");
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(sig.returns[0].cell, 0);
        assert_eq!(sig.returns[0].provenance, Provenance::AbiAssumed);
        let dump = render(&sig);
        assert!(dump.contains("header=add(a, b)"), "{dump}");
        assert!(dump.contains("dataflow"), "{dump}");
    }

    #[test]
    fn prefix_rule_fills_gap() {
        let f = gap_middle_arg();
        let sig = recover(&f);
        assert!(check(&f, &sig).is_ok());
        assert_eq!(sig.params.len(), 3); // rdi, rsi(gap), rdx
        assert_eq!(sig.params[0].provenance, Provenance::DataflowProven);
        assert_eq!(sig.params[1].cell, 6); // rsi
        assert_eq!(sig.params[1].provenance, Provenance::AbiAssumed);
        assert_eq!(sig.params[2].provenance, Provenance::DataflowProven);
        assert_eq!(render_header(&sig), "sub_2000(a, b, c)");
    }

    #[test]
    fn callee_save_live_in_is_not_a_param() {
        // rbx (cell 3) live-in alone — not an ABI arg.
        let names = vec![name(3, 0, Width::W64)];
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x3000,
            SsaBlock {
                start: 0x3000,
                end: 0x3004,
                phis: vec![],
                stmts: vec![],
                successors: vec![],
                truncated: false,
            },
        );
        let f = SsaFunction {
            entry: 0x3000,
            name: Some("leaf".into()),
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![0],
            partial: vec![],
        };
        let sig = recover(&f);
        assert!(check(&f, &sig).is_ok());
        assert!(sig.params.is_empty());
        assert_eq!(render_header(&sig), "leaf()");
    }

    #[test]
    fn aarch64_int_and_float() {
        // x0 + d0 (cell 32) live-in.
        let names = vec![name(0, 0, Width::W64), name(32, 0, Width::W64)];
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x4000,
            SsaBlock {
                start: 0x4000,
                end: 0x4008,
                phis: vec![],
                stmts: vec![],
                successors: vec![],
                truncated: false,
            },
        );
        let f = SsaFunction {
            entry: 0x4000,
            name: Some("mixed".into()),
            arch: Arch::Aarch64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![0, 1],
            partial: vec![],
        };
        let sig = recover(&f);
        assert!(check(&f, &sig).is_ok());
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0].kind, ParamKind::Int);
        assert_eq!(sig.params[0].cell, 0);
        assert_eq!(sig.params[1].kind, ParamKind::Float);
        assert_eq!(sig.params[1].cell, 32);
        assert_eq!(render_header(&sig), "mixed(a, b)");
        assert_eq!(sig.returns[0].cell, 0);
    }

    #[test]
    fn zero_args_still_has_abi_return() {
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x5000,
            SsaBlock {
                start: 0x5000,
                end: 0x5004,
                phis: vec![],
                stmts: vec![],
                successors: vec![],
                truncated: false,
            },
        );
        let f = SsaFunction {
            entry: 0x5000,
            name: None,
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names: vec![],
            live_in: vec![],
            partial: vec![],
        };
        let sig = recover(&f);
        assert!(check(&f, &sig).is_ok());
        assert!(sig.params.is_empty());
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(render_header(&sig), "sub_5000()");
    }

    #[test]
    fn recover_is_deterministic() {
        let f = two_int_args();
        assert_eq!(recover(&f), recover(&f));
    }

    /// Callee loads [rsp+8] and [rsp+0x10] — two stack args (16 bytes
    /// above the return address).
    fn stack_arg_loads() -> SsaFunction {
        // name0 = rsp#0 Affine(0); name1 = rsp+8 addr; name2 = load dst
        let names = vec![
            name(4, 0, Width::W64), // rsp
            name(10, 1, Width::W64),
            name(11, 1, Width::W64),
            name(12, 1, Width::W64),
            name(13, 1, Width::W64),
        ];
        let rsp = Reg {
            space: Space::Arch,
            num: 0,
            width: Width::W64,
        };
        let stmts = vec![
            // t1 = rsp + 8
            Stmt::Assign {
                dst: Reg {
                    space: Space::Arch,
                    num: 1,
                    width: Width::W64,
                },
                value: Expr::Binary {
                    op: BinOp::Add,
                    lhs: Box::new(Expr::reg(rsp)),
                    rhs: Box::new(Expr::constant(8, Width::W64)),
                },
            },
            // t2 = load [t1]
            Stmt::Assign {
                dst: Reg {
                    space: Space::Arch,
                    num: 2,
                    width: Width::W64,
                },
                value: Expr::load(
                    Expr::reg(Reg {
                        space: Space::Arch,
                        num: 1,
                        width: Width::W64,
                    }),
                    Width::W64,
                ),
            },
            // t3 = rsp + 0x10
            Stmt::Assign {
                dst: Reg {
                    space: Space::Arch,
                    num: 3,
                    width: Width::W64,
                },
                value: Expr::Binary {
                    op: BinOp::Add,
                    lhs: Box::new(Expr::reg(rsp)),
                    rhs: Box::new(Expr::constant(0x10, Width::W64)),
                },
            },
            // t4 = load [t3]
            Stmt::Assign {
                dst: Reg {
                    space: Space::Arch,
                    num: 4,
                    width: Width::W64,
                },
                value: Expr::load(
                    Expr::reg(Reg {
                        space: Space::Arch,
                        num: 3,
                        width: Width::W64,
                    }),
                    Width::W64,
                ),
            },
        ];
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x6000,
            SsaBlock {
                start: 0x6000,
                end: 0x6020,
                phis: vec![],
                stmts,
                successors: vec![],
                truncated: false,
            },
        );
        SsaFunction {
            entry: 0x6000,
            name: Some("stacky".into()),
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![0],
            partial: vec![],
        }
    }

    #[test]
    fn positive_offset_loads_set_stack_bytes_in_header() {
        let f = stack_arg_loads();
        let sig = recover(&f);
        assert!(check(&f, &sig).is_ok(), "{:?}", check(&f, &sig));
        assert_eq!(sig.stack_bytes, 16); // [8,16) ∪ [16,24) → end 24 − floor 8
        assert_eq!(render_header(&sig), "stacky(stack=16)");
        let dump = render(&sig);
        assert!(dump.contains("stack_bytes=16"), "{dump}");
        assert!(dump.contains("header=stacky(stack=16)"), "{dump}");
    }

    #[test]
    fn negative_frame_store_is_not_stack_args() {
        // Classic prologue spill at rsp-0x20 must not set stack_bytes.
        let names = vec![
            name(4, 0, Width::W64),
            name(4, 1, Width::W64),
            name(0, 0, Width::W64),
        ];
        let stmts = vec![
            Stmt::Assign {
                dst: Reg {
                    space: Space::Arch,
                    num: 1,
                    width: Width::W64,
                },
                value: Expr::Binary {
                    op: BinOp::Sub,
                    lhs: Box::new(Expr::reg(Reg {
                        space: Space::Arch,
                        num: 0,
                        width: Width::W64,
                    })),
                    rhs: Box::new(Expr::constant(0x20, Width::W64)),
                },
            },
            Stmt::Store {
                addr: Expr::reg(Reg {
                    space: Space::Arch,
                    num: 1,
                    width: Width::W64,
                }),
                value: Expr::reg(Reg {
                    space: Space::Arch,
                    num: 2,
                    width: Width::W64,
                }),
            },
        ];
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x6100,
            SsaBlock {
                start: 0x6100,
                end: 0x6110,
                phis: vec![],
                stmts,
                successors: vec![],
                truncated: false,
            },
        );
        let f = SsaFunction {
            entry: 0x6100,
            name: Some("frame".into()),
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![0, 2],
            partial: vec![],
        };
        let sig = recover(&f);
        assert_eq!(sig.stack_bytes, 0);
        assert_eq!(render_header(&sig), "frame()");
    }

    #[test]
    fn symbol_derived_extends_arity_from_demangled_prototype() {
        let mut f = two_int_args();
        // Dataflow sees 2; symbol claims 4 — fill rdi..rcx as SymbolDerived.
        f.name = Some("widget::add(int, int, int, int)".into());
        let sig = recover(&f);
        assert!(check(&f, &sig).is_ok(), "{:?}", check(&f, &sig));
        assert_eq!(sig.params.len(), 4);
        assert_eq!(sig.params[0].provenance, Provenance::SymbolDerived);
        assert_eq!(sig.params[0].name_id, Some(0)); // kept live-in witness
        assert_eq!(sig.params[2].provenance, Provenance::SymbolDerived);
        assert_eq!(sig.params[2].cell, 2); // rdx
        assert_eq!(sig.params[3].cell, 1); // rcx
        assert_eq!(render_header(&sig), "widget::add(int, int, int, int)(a, b, c, d)");
    }

    #[test]
    fn symbol_arity_from_mangled_cxx_name() {
        let mut f = two_int_args();
        // _Z3fooii → foo(int, int); matches dataflow arity → SymbolDerived.
        f.name = Some("_Z3fooii".into());
        let sig = recover(&f);
        assert!(check(&f, &sig).is_ok());
        assert_eq!(sig.params.len(), 2);
        assert!(sig
            .params
            .iter()
            .all(|p| p.provenance == Provenance::SymbolDerived));
    }

    #[test]
    fn symbol_does_not_shrink_below_dataflow() {
        let mut f = two_int_args();
        f.name = Some("narrow(int)".into()); // arity 1 < dataflow 2
        let sig = recover(&f);
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0].provenance, Provenance::DataflowProven);
    }

    #[test]
    fn prototype_arity_parses_nested_types() {
        assert_eq!(
            prototype_arity("f(void (*)(int, int), char const*)"),
            Some(2)
        );
        assert_eq!(prototype_arity("bar()"), Some(0));
        assert_eq!(prototype_arity("bar(void)"), Some(0));
        assert_eq!(prototype_arity("no_proto"), None);
        assert_eq!(symbol_arity_hint(Some("_Z3fooiPKc")), Some(2));
    }

    #[test]
    fn callers_of_inverts_call_graph_with_cap() {
        let mut g: BTreeMap<u64, BTreeSet<CallTarget>> = BTreeMap::new();
        g.insert(0x100, BTreeSet::from([CallTarget::Function(0x500)]));
        g.insert(0x200, BTreeSet::from([CallTarget::Function(0x500)]));
        g.insert(0x300, BTreeSet::from([CallTarget::Import("x".into())]));
        assert_eq!(callers_of(&g, 0x500), vec![0x100, 0x200]);
        assert!(callers_of(&g, 0x999).is_empty());
    }

    /// Caller: call callee, then use rax (no callfx — ret cell use).
    fn caller_that_reads_rax(callee: u64) -> SsaFunction {
        // names: 0 = rax#1, 1 = temp holding the use
        let names = vec![
            name(0, 1, Width::W64),
            Name {
                space: Space::Temp,
                cell: 0,
                version: 1,
                width: Width::W64,
            },
        ];
        let stmts = vec![
            Stmt::Branch {
                kind: BranchKind::Call,
                cond: None,
                target: Expr::constant(callee, Width::W64),
            },
            // Simulate return landing in rax, then a use.
            Stmt::Assign {
                dst: Reg {
                    space: Space::Arch,
                    num: 0,
                    width: Width::W64,
                },
                value: Expr::constant(0, Width::W64),
            },
            Stmt::Assign {
                dst: Reg {
                    space: Space::Temp,
                    num: 1,
                    width: Width::W64,
                },
                value: Expr::reg(Reg {
                    space: Space::Arch,
                    num: 0,
                    width: Width::W64,
                }),
            },
        ];
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x7000,
            SsaBlock {
                start: 0x7000,
                end: 0x7010,
                phis: vec![],
                stmts,
                successors: vec![],
                truncated: false,
            },
        );
        SsaFunction {
            entry: 0x7000,
            name: Some("caller".into()),
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![],
            partial: vec![],
        }
    }

    #[test]
    fn confirm_returns_upgrades_abi_assumed_when_caller_reads() {
        let callee = two_int_args();
        let mut sig = recover(&callee);
        assert_eq!(sig.returns[0].provenance, Provenance::AbiAssumed);
        let caller = caller_that_reads_rax(callee.entry);
        assert!(caller_reads_return(&caller, callee.entry));
        confirm_returns(&mut sig, &[&caller]);
        assert_eq!(sig.returns[0].provenance, Provenance::DataflowProven);
    }

    #[test]
    fn confirm_returns_noop_without_read() {
        let callee = two_int_args();
        let mut sig = recover(&callee);
        // Caller with a call but no ret use.
        let names = vec![name(0, 0, Width::W64)];
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x7100,
            SsaBlock {
                start: 0x7100,
                end: 0x7108,
                phis: vec![],
                stmts: vec![Stmt::Branch {
                    kind: BranchKind::Call,
                    cond: None,
                    target: Expr::constant(callee.entry, Width::W64),
                }],
                successors: vec![],
                truncated: false,
            },
        );
        let caller = SsaFunction {
            entry: 0x7100,
            name: None,
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![],
            partial: vec![],
        };
        assert!(!caller_reads_return(&caller, callee.entry));
        confirm_returns(&mut sig, &[&caller]);
        assert_eq!(sig.returns[0].provenance, Provenance::AbiAssumed);
    }

    #[test]
    fn stack_bytes_plus_reg_params_in_header() {
        let mut f = two_int_args();
        // Graft a stack load at rsp+8 onto the two-arg body.
        f.names.push(name(4, 0, Width::W64)); // rsp live-in
        f.names.push(name(20, 1, Width::W64)); // addr
        f.names.push(name(21, 1, Width::W64)); // load
        f.live_in.push(3);
        let block = f.blocks.get_mut(&0x1000).unwrap();
        block.stmts.push(Stmt::Assign {
            dst: Reg {
                space: Space::Arch,
                num: 4,
                width: Width::W64,
            },
            value: Expr::Binary {
                op: BinOp::Add,
                lhs: Box::new(Expr::reg(Reg {
                    space: Space::Arch,
                    num: 3,
                    width: Width::W64,
                })),
                rhs: Box::new(Expr::constant(8, Width::W64)),
            },
        });
        block.stmts.push(Stmt::Assign {
            dst: Reg {
                space: Space::Arch,
                num: 5,
                width: Width::W64,
            },
            value: Expr::load(
                Expr::reg(Reg {
                    space: Space::Arch,
                    num: 4,
                    width: Width::W64,
                }),
                Width::W64,
            ),
        });
        let sig = recover(&f);
        assert_eq!(sig.stack_bytes, 8);
        assert_eq!(render_header(&sig), "add(a, b; stack=8)");
    }
}
