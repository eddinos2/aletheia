//! Callee-side signature recovery (DESIGN slice 12).
//!
//! After callfx + SSA opt, version-0 cells in [`SsaFunction::live_in`] are
//! exactly the read-before-write set. Intersect that set with the ABI
//! argument-register sequence; the highest witnessed index (per int /
//! float class) is the arity under the prefix rule. The ABI primary
//! return cell is recorded as [`Provenance::AbiAssumed`] — caller-side
//! confirmation is slice 13.
//!
//! Total: never panics. Caps truncate rather than grow without bound.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::ir::{Space, Width};
use crate::irssa::SsaFunction;
use crate::model::Arch;

/// Cap on claimed parameters per function (int + float combined).
pub const MAX_PARAMS: usize = 64;
/// Cap on claimed return cells per function.
pub const MAX_RETURNS: usize = 8;

/// How a signature fact was obtained. Ranked like [`crate::funcs::Source`]:
/// symbol-derived (slice 14) > dataflow-proven > ABI-assumed > heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Provenance {
    /// Reserved for demangled / metadata prototypes (slice 14).
    SymbolDerived,
    /// Witnessed by a live-in SSA name of an ABI argument cell.
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
    /// Stack-argument bytes (always 0 until stack-arg recovery).
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

/// Recover callee-side signature facts from optimized SSA. Total.
pub fn recover(f: &SsaFunction) -> Signature {
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

    sig
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
                    Some(_) => return Err(format!("name_id {id} is not the live-in for cell {}", p.cell)),
                    None => return Err(format!("DataflowProven cell {} not in live_in", p.cell)),
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
pub fn render_header(sig: &Signature) -> String {
    let name = match sig.name.as_deref() {
        Some(n) if !n.is_empty() => n.to_string(),
        _ => format!("sub_{:x}", sig.entry),
    };
    let args = (0..sig.params.len())
        .map(arg_letter)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{name}({args})")
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
    use crate::ir::{Expr, Reg, Space, Stmt, Width};
    use crate::irssa::{Name, SsaBlock, SsaFunction};
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
        let names = vec![
            name(0, 0, Width::W64),
            name(32, 0, Width::W64),
        ];
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
}
