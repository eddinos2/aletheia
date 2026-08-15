//! Whole-function lifting: a recovered function's machine code to an IR
//! control-flow graph.
//!
//! [`crate::x86_lift`] lifts one instruction; this module lifts a whole
//! [`crate::cfg::Function`] — it walks each basic block's byte range with
//! the architecture's decoder, lifts every instruction, and threads
//! temporaries so the block reads as one well-formed [`crate::ir`]
//! statement list. The result is an IR CFG: the same block structure the
//! CFG recovery found, with each block's machine instructions replaced by
//! their IR. Architecture-dispatched (x86-64 via [`crate::x86_lift`],
//! AArch64 via [`crate::aarch64_lift`]; an architecture with no lifter
//! yields `None`, so a caller can print a note instead of silence), and
//! the chosen architecture is carried on the [`LiftedFunction`] so every
//! renderer downstream spells registers in the right ISA's names.
//! Best-effort by contract: an undecodable instruction or an unmapped
//! address ends a block's lift with its [`LiftedBlock::truncated`] flag
//! set — an honest signal, never hidden — resource caps bound hostile
//! block ranges, and no input ever panics.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::model::{Arch, Image};
use crate::{aarch64, aarch64_lift, cfg, ir, irflow, x86, x86_lift};

// ---------------------------------------------------------------------------
// Resource caps
// ---------------------------------------------------------------------------

/// Largest byte span one block's walk will decode. A recovered block is
/// far smaller in practice; a hand-built or hostile range past the cap
/// (or with `end < start`) lifts to a truncated empty block.
const MAX_BLOCK_BYTES: u64 = 1 << 20;

/// Total instructions lifted across one function, bounding a pathological
/// CFG. Hitting the budget truncates the block being walked and every
/// later block of the function.
const MAX_FUNCTION_INSNS: usize = 1 << 20;

// ---------------------------------------------------------------------------
// The lifted CFG
// ---------------------------------------------------------------------------

/// One basic block with its machine instructions replaced by their IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftedBlock {
    /// VA of the first instruction (the block's [`cfg::BasicBlock::start`]).
    pub start: u64,
    /// VA one past the last byte the CFG recovered (exclusive).
    pub end: u64,
    /// The block's instructions lifted in order, with temporaries threaded
    /// from one counter so the whole list passes [`ir::check`].
    pub stmts: Vec<ir::Stmt>,
    /// Intra-function edges, copied from the recovered block.
    pub successors: Vec<u64>,
    /// The byte walk ended before `end`: an undecodable instruction, an
    /// unmapped VA, an absurd block range, or a resource cap. The lifted
    /// prefix in `stmts` is still valid; nothing past the stop is guessed.
    pub truncated: bool,
}

/// One function lifted to an IR control-flow graph: the block structure
/// [`cfg::recover`] found, each block carrying its IR statement list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftedFunction {
    /// Entry VA.
    pub entry: u64,
    /// Name carried over from the recovered function.
    pub name: Option<String>,
    /// Architecture the function was lifted from. Pure data with one job:
    /// it selects the [`ir::Space::Arch`] register speller
    /// ([`x86_lift::reg_name`] vs [`aarch64_lift::reg_name`]) in every
    /// renderer downstream, so an aarch64 function prints `x0`/`sp`, not
    /// `rax`.
    pub arch: Arch,
    /// Lifted blocks keyed by start VA.
    pub blocks: BTreeMap<u64, LiftedBlock>,
}

/// Lift one recovered function into an IR CFG.
///
/// `None` when the image's architecture has no lifter — x86-64 and
/// AArch64 are implemented, so only [`Arch::Other`] returns `None` and a
/// caller can print a note rather than nothing. Deterministic and total:
/// the same inputs always produce the same result, and no input panics.
pub fn lift_function(image: &dyn Image, func: &cfg::Function) -> Option<LiftedFunction> {
    lift_function_capped(image, func, MAX_FUNCTION_INSNS)
}

/// [`lift_function`] with a caller-visible instruction budget, so the cap
/// path is testable without a million-instruction fixture.
fn lift_function_capped(
    image: &dyn Image,
    func: &cfg::Function,
    max_insns: usize,
) -> Option<LiftedFunction> {
    let lift_block: fn(&dyn Image, &cfg::BasicBlock, &mut usize) -> LiftedBlock =
        match image.arch() {
            Arch::X86_64 => lift_block_x86,
            Arch::Aarch64 => lift_block_aarch64,
            Arch::Other => return None,
        };
    let mut budget = max_insns;
    let blocks = func
        .blocks
        .iter()
        .map(|(&start, block)| (start, lift_block(image, block, &mut budget)))
        .collect();
    Some(LiftedFunction {
        entry: func.entry,
        name: func.name.clone(),
        arch: image.arch(),
        blocks,
    })
}

/// Lift every function of a recovered program, keyed by entry VA.
/// Functions the architecture cannot lift are omitted. Deterministic:
/// B-tree iteration in, B-tree map out.
pub fn lift_program(image: &dyn Image, program: &cfg::Program) -> BTreeMap<u64, LiftedFunction> {
    program
        .functions
        .iter()
        .filter_map(|(&entry, func)| lift_function(image, func).map(|lf| (entry, lf)))
        .collect()
}

// ---------------------------------------------------------------------------
// Per-block lifting (x86-64, aarch64)
// ---------------------------------------------------------------------------

/// Walk one block's byte range, decode and collect its instructions, and
/// lift them as a single temp-threaded statement list.
fn lift_block_x86(image: &dyn Image, block: &cfg::BasicBlock, budget: &mut usize) -> LiftedBlock {
    let mut truncated = false;
    let mut pairs: Vec<(x86::Instruction, u64)> = Vec::new();

    // An absurd range (backwards, or wider than any sane block) is not
    // walked at all: truncated empty.
    let sane = block.end >= block.start && block.end - block.start <= MAX_BLOCK_BYTES;
    if !sane {
        truncated = true;
    } else {
        let mut va = block.start;
        while va < block.end {
            if *budget == 0 {
                truncated = true;
                break;
            }
            let Some(off) = image.va_to_offset(va) else {
                truncated = true;
                break;
            };
            let Some(bytes) = image.bytes().get(off..) else {
                truncated = true;
                break;
            };
            let Ok(insn) = x86::decode(bytes, va) else {
                truncated = true;
                break;
            };
            let len = u64::from(insn.length);
            if len == 0 {
                truncated = true; // a zero-length decode cannot advance
                break;
            }
            *budget -= 1;
            pairs.push((insn, va));
            let Some(next) = va.checked_add(len) else {
                truncated = true;
                break;
            };
            va = next;
        }
    }

    // One threaded lift for the whole block: `lift_block` renumbers
    // temporaries from a single counter, which is what lets the list pass
    // `ir::check`'s single-namespace temp rule.
    let mut stmts = x86_lift::lift_block(&pairs);

    // `ir::check` caps a block at `MAX_STMTS`; a block long enough to
    // exceed it is cut back to the largest instruction prefix that fits,
    // and marked truncated. (Each instruction's statement count is the
    // same at any temp base, so single-instruction lifts size the prefix.)
    if stmts.len() > ir::MAX_STMTS {
        truncated = true;
        let mut total = 0usize;
        let mut keep = 0usize;
        for (insn, va) in &pairs {
            let n = x86_lift::lift(insn, *va).len();
            if total + n > ir::MAX_STMTS {
                break;
            }
            total += n;
            keep += 1;
        }
        pairs.truncate(keep);
        stmts = x86_lift::lift_block(&pairs);
    }

    LiftedBlock {
        start: block.start,
        end: block.end,
        stmts,
        successors: block.successors.clone(),
        truncated,
    }
}

/// [`lift_block_x86`] for A64: the same walk over fixed 4-byte words.
/// [`aarch64::decode`] is total on a full word (an unmodeled encoding is
/// [`aarch64::Opcode::Unknown`], which lifts to a sound clobber-everything
/// intrinsic — coverage, not correctness, is what is partial), so the only
/// decode failure is a word cut short by the mapping's edge, and the
/// truncation causes are exactly the x86 path's: an unmapped VA, a partial
/// word, an absurd range, or a resource cap.
fn lift_block_aarch64(
    image: &dyn Image,
    block: &cfg::BasicBlock,
    budget: &mut usize,
) -> LiftedBlock {
    let mut truncated = false;
    let mut pairs: Vec<(aarch64::Instruction, u64)> = Vec::new();

    // An absurd range (backwards, or wider than any sane block) is not
    // walked at all: truncated empty.
    let sane = block.end >= block.start && block.end - block.start <= MAX_BLOCK_BYTES;
    if !sane {
        truncated = true;
    } else {
        let mut va = block.start;
        while va < block.end {
            if *budget == 0 {
                truncated = true;
                break;
            }
            let Some(off) = image.va_to_offset(va) else {
                truncated = true;
                break;
            };
            let Some(bytes) = image.bytes().get(off..) else {
                truncated = true;
                break;
            };
            let Ok(insn) = aarch64::decode(bytes, va) else {
                truncated = true; // fewer than 4 mapped bytes remain
                break;
            };
            *budget -= 1;
            pairs.push((insn, va));
            let Some(next) = va.checked_add(aarch64::Instruction::SIZE as u64) else {
                truncated = true;
                break;
            };
            va = next;
        }
    }

    // One threaded lift for the whole block, exactly as the x86 path:
    // temporaries drawn from a single counter so the list passes
    // `ir::check`.
    let mut stmts = aarch64_lift::lift_block(&pairs);

    // Cut back to the largest instruction prefix that fits `ir::check`'s
    // statement cap, marked truncated. (Statement counts are temp-base
    // independent here too.)
    if stmts.len() > ir::MAX_STMTS {
        truncated = true;
        let mut total = 0usize;
        let mut keep = 0usize;
        for (insn, va) in &pairs {
            let n = aarch64_lift::lift(insn, *va).len();
            if total + n > ir::MAX_STMTS {
                break;
            }
            total += n;
            keep += 1;
        }
        pairs.truncate(keep);
        stmts = aarch64_lift::lift_block(&pairs);
    }

    LiftedBlock {
        start: block.start,
        end: block.end,
        stmts,
        successors: block.successors.clone(),
        truncated,
    }
}

// ---------------------------------------------------------------------------
// Simplification
// ---------------------------------------------------------------------------

/// Simplify every block of a lifted function through the
/// [`crate::irflow`] passes: constant folding, copy propagation, and
/// dead-code elimination, each under [`irflow::simplify_default`]'s
/// conservative live-out (every architectural register stays live across
/// the block edge; only dead temporaries, dead flags, and redundant
/// copies go). Block structure, bounds, successors, and the `truncated`
/// flag are untouched — this rewrites each block's statement list and
/// nothing else. Deterministic, behavior-preserving, and total.
pub fn simplify(func: &LiftedFunction) -> LiftedFunction {
    LiftedFunction {
        entry: func.entry,
        name: func.name.clone(),
        arch: func.arch,
        blocks: func
            .blocks
            .iter()
            .map(|(&start, block)| {
                (
                    start,
                    LiftedBlock {
                        start: block.start,
                        end: block.end,
                        stmts: irflow::simplify_default(&block.stmts),
                        successors: block.successors.clone(),
                        truncated: block.truncated,
                    },
                )
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// The [`ir::Space::Arch`] register speller for `arch`: the crate-wide
/// dispatch every IR renderer shares, so a lifted aarch64 function prints
/// `x0`…`x30`/`sp` and an x86-64 one `rax`…`r15`. [`Arch::Other`] keeps
/// the x86-64 spelling — no lifted function can carry it (the dispatch
/// refuses), so it is reachable only from hand-built IR, where the old
/// behavior stays byte-identical.
pub fn reg_namer(arch: Arch) -> ir::RegNamer<'static> {
    match arch {
        Arch::Aarch64 => &aarch64_lift::reg_name,
        Arch::X86_64 | Arch::Other => &x86_lift::reg_name,
    }
}

/// Render a lifted function to a deterministic, human-readable dump: a
/// `; {name} @ {entry:#018x}` header (falling back to `sub_{entry:x}`
/// for an unnamed function), then per block a `loc_{start:x}:` label, the
/// block's IR indented via [`ir::render`] with the function's
/// architecture selecting the namer ([`reg_namer`]), a `; (truncated)`
/// marker when the block's lift stopped early, and a trailing
/// `; -> loc_..., loc_...` edge line (`; -> (none)` for a block with no
/// successors). `\n`-terminated.
pub fn render(func: &LiftedFunction) -> String {
    let mut out = String::new();
    let name = func
        .name
        .clone()
        .unwrap_or_else(|| format!("sub_{:x}", func.entry));
    let namer = reg_namer(func.arch);
    let _ = writeln!(out, "; {name} @ {:#018x}", func.entry);
    for block in func.blocks.values() {
        let _ = writeln!(out, "loc_{:x}:", block.start);
        for line in ir::render(&block.stmts, namer).lines() {
            let _ = writeln!(out, "  {line}");
        }
        if block.truncated {
            let _ = writeln!(out, "  ; (truncated)");
        }
        if block.successors.is_empty() {
            let _ = writeln!(out, "  ; -> (none)");
        } else {
            let edges = block
                .successors
                .iter()
                .map(|s| format!("loc_{s:x}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "  ; -> {edges}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImportSlot, Perms, Region, Symbol};

    /// Base VA the in-memory test image maps its code at.
    const BASE: u64 = 0x1000;

    /// A minimal in-memory image: one executable region holding `code`
    /// at [`BASE`], entry at [`BASE`], identity VA-to-offset mapping.
    struct TestImage {
        arch: Arch,
        code: Vec<u8>,
    }

    fn image(code: &[u8]) -> TestImage {
        TestImage {
            arch: Arch::X86_64,
            code: code.to_vec(),
        }
    }

    impl Image for TestImage {
        fn arch(&self) -> Arch {
            self.arch
        }
        fn entry_points(&self) -> Vec<u64> {
            vec![BASE]
        }
        fn regions(&self) -> Vec<Region> {
            vec![Region {
                name: ".text".into(),
                va: BASE,
                size: self.code.len() as u64,
                perms: Perms {
                    r: true,
                    w: false,
                    x: true,
                },
            }]
        }
        fn symbols(&self) -> Vec<Symbol> {
            Vec::new()
        }
        fn import_slots(&self) -> Vec<ImportSlot> {
            Vec::new()
        }
        fn va_to_offset(&self, va: u64) -> Option<usize> {
            let off = va.checked_sub(BASE)?;
            (off < self.code.len() as u64).then_some(off as usize)
        }
        fn bytes(&self) -> &[u8] {
            &self.code
        }
    }

    /// A conditional diamond (the [`crate::cfg`] fixture shape):
    ///
    /// ```text
    /// 1000  xor eax, eax
    /// 1002  je   1009
    /// 1004  inc eax
    /// 1006  jmp 100b
    /// 1008  (unreachable)
    /// 1009  dec eax
    /// 100b  ret
    /// ```
    const DIAMOND: &[u8] = &[
        0x31, 0xC0, // +0: xor eax, eax
        0x74, 0x05, // +2: je +9
        0xFF, 0xC0, // +4: inc eax
        0xEB, 0x03, // +6: jmp +11
        0x90, // +8: (unreachable)
        0xFF, 0xC8, // +9: dec eax
        0xC3, // +11: ret
    ];

    fn recover(img: &TestImage) -> cfg::Program {
        cfg::recover(img).expect("x86-64 recovers")
    }

    /// Recover `code` and lift its entry function.
    fn lift_entry(code: &[u8]) -> LiftedFunction {
        let img = image(code);
        let program = recover(&img);
        lift_function(&img, &program.functions[&BASE]).expect("x86-64 lifts")
    }

    /// A hand-built single-block function over `[start, end)`, for the
    /// truncation paths [`cfg::recover`] would never produce itself.
    fn raw_function(start: u64, end: u64) -> cfg::Function {
        cfg::Function {
            entry: start,
            name: None,
            blocks: BTreeMap::from([(
                start,
                cfg::BasicBlock {
                    start,
                    end,
                    terminator: cfg::Terminator::Undecodable,
                    successors: Vec::new(),
                },
            )]),
        }
    }

    // ---- structure ----

    #[test]
    fn lifted_function_mirrors_the_recovered_block_structure() {
        let img = image(DIAMOND);
        let program = recover(&img);
        let func = &program.functions[&BASE];
        let lifted = lift_function(&img, func).expect("x86-64 lifts");

        assert_eq!(lifted.entry, func.entry);
        assert_eq!(lifted.name, func.name);
        assert_eq!(
            lifted.blocks.keys().copied().collect::<Vec<_>>(),
            func.blocks.keys().copied().collect::<Vec<_>>()
        );
        assert_eq!(
            lifted.blocks.keys().copied().collect::<Vec<_>>(),
            [BASE, BASE + 4, BASE + 9, BASE + 11]
        );
    }

    #[test]
    fn block_bounds_and_successors_are_preserved() {
        let img = image(DIAMOND);
        let program = recover(&img);
        let func = &program.functions[&BASE];
        let lifted = lift_function(&img, func).expect("x86-64 lifts");
        for (start, block) in &func.blocks {
            let lb = &lifted.blocks[start];
            assert_eq!(lb.start, block.start);
            assert_eq!(lb.end, block.end);
            assert_eq!(lb.successors, block.successors);
        }
    }

    #[test]
    fn every_lifted_block_passes_ir_check() {
        let lifted = lift_entry(DIAMOND);
        for (start, block) in &lifted.blocks {
            assert_eq!(ir::check(&block.stmts), Ok(()), "block {start:#x}");
        }
    }

    #[test]
    fn clean_code_truncates_no_block() {
        let lifted = lift_entry(DIAMOND);
        assert!(lifted.blocks.values().all(|b| !b.truncated));
    }

    #[test]
    fn a_conditional_block_ends_in_a_guarded_branch_with_two_successors() {
        let lifted = lift_entry(DIAMOND);
        let cond = &lifted.blocks[&BASE];
        assert_eq!(cond.successors, [BASE + 9, BASE + 4]);
        let Some(ir::Stmt::Branch { kind, cond: c, .. }) = cond.stmts.last() else {
            panic!("conditional block must end in a Branch");
        };
        assert_eq!(*kind, ir::BranchKind::Jump);
        assert!(c.is_some(), "the jcc branch carries a W1 guard");
    }

    #[test]
    fn the_branch_target_constant_is_the_taken_successor() {
        let lifted = lift_entry(DIAMOND);
        let Some(ir::Stmt::Branch { target, .. }) = lifted.blocks[&BASE].stmts.last() else {
            panic!("conditional block must end in a Branch");
        };
        assert_eq!(
            *target,
            ir::Expr::constant(BASE + 9, ir::Width::W64),
            "the je target folds to the taken successor's VA"
        );
    }

    #[test]
    fn a_return_block_ends_in_a_return_branch() {
        let lifted = lift_entry(DIAMOND);
        let ret = &lifted.blocks[&(BASE + 11)];
        assert!(ret.successors.is_empty());
        assert!(matches!(
            ret.stmts.last(),
            Some(ir::Stmt::Branch {
                kind: ir::BranchKind::Return,
                cond: None,
                ..
            })
        ));
    }

    #[test]
    fn temps_are_threaded_uniquely_across_a_block() {
        // pop rbp ; pop rbx ; ret — three temp-using instructions in one
        // block must draw distinct temporaries from one counter.
        let lifted = lift_entry(&[0x5D, 0x5B, 0xC3]);
        let block = &lifted.blocks[&BASE];
        assert_eq!(ir::check(&block.stmts), Ok(()));
        let temps: Vec<u16> = block
            .stmts
            .iter()
            .filter_map(|s| match s {
                ir::Stmt::Assign { dst, .. } if dst.space == ir::Space::Temp => Some(dst.num),
                _ => None,
            })
            .collect();
        assert_eq!(temps, [0, 1, 2]);
    }

    // ---- truncation ----

    #[test]
    fn an_undecodable_byte_truncates_the_block_but_keeps_the_prefix() {
        // push rbp, then an unmodeled x87 opcode inside the range.
        let img = image(&[0x55, 0xD8, 0x00, 0x06]);
        let func = raw_function(BASE, BASE + 4);
        let lifted = lift_function(&img, &func).expect("x86-64 lifts");
        let block = &lifted.blocks[&BASE];
        assert!(block.truncated);
        // The push's two statements survive; nothing past the stop.
        assert_eq!(block.stmts.len(), 2);
        assert_eq!(ir::check(&block.stmts), Ok(()));
    }

    #[test]
    fn an_unmapped_va_truncates_the_block() {
        // Four mapped nops, but the block claims eight bytes.
        let img = image(&[0x90, 0x90, 0x90, 0x90]);
        let func = raw_function(BASE, BASE + 8);
        let lifted = lift_function(&img, &func).expect("x86-64 lifts");
        assert!(lifted.blocks[&BASE].truncated);
    }

    #[test]
    fn a_backwards_block_range_is_truncated_empty() {
        let img = image(&[0x90; 4]);
        let func = raw_function(BASE + 4, BASE);
        let lifted = lift_function(&img, &func).expect("x86-64 lifts");
        let block = &lifted.blocks[&(BASE + 4)];
        assert!(block.truncated);
        assert!(block.stmts.is_empty());
    }

    #[test]
    fn an_oversized_block_range_is_truncated_empty() {
        let img = image(&[0x90; 4]);
        let func = raw_function(BASE, BASE + (1 << 20) + 1);
        let lifted = lift_function(&img, &func).expect("x86-64 lifts");
        let block = &lifted.blocks[&BASE];
        assert!(block.truncated);
        assert!(block.stmts.is_empty());
    }

    #[test]
    fn an_oversized_statement_list_is_cut_at_the_ir_cap() {
        // 2050 pushes lift to 4100 statements — past `ir::MAX_STMTS`. The
        // block is cut to the largest prefix that fits (2048 pushes) and
        // still passes `ir::check`.
        let code = vec![0x55u8; 2050];
        let img = image(&code);
        let func = raw_function(BASE, BASE + 2050);
        let lifted = lift_function(&img, &func).expect("x86-64 lifts");
        let block = &lifted.blocks[&BASE];
        assert!(block.truncated);
        assert_eq!(block.stmts.len(), ir::MAX_STMTS);
        assert_eq!(ir::check(&block.stmts), Ok(()));
    }

    #[test]
    fn the_instruction_budget_bounds_a_pathological_function() {
        let img = image(DIAMOND);
        let program = recover(&img);
        let func = &program.functions[&BASE];
        // Budget of three: xor and je fit, the second block's inc spends
        // the last unit, and everything after is truncated.
        let lifted = lift_function_capped(&img, func, 3).expect("x86-64 lifts");
        assert!(!lifted.blocks[&BASE].truncated);
        assert!(lifted.blocks[&(BASE + 4)].truncated);
        assert!(lifted.blocks[&(BASE + 9)].truncated);
        assert!(lifted.blocks[&(BASE + 9)].stmts.is_empty());
        for block in lifted.blocks.values() {
            assert_eq!(ir::check(&block.stmts), Ok(()));
        }
    }

    // ---- architecture dispatch ----

    #[test]
    fn an_unknown_arch_lifts_to_none() {
        let func = raw_function(BASE, BASE + 4);
        let img = TestImage {
            arch: Arch::Other,
            code: vec![0x90; 4],
        };
        assert_eq!(lift_function(&img, &func), None);
    }

    #[test]
    fn lift_program_omits_functions_no_lifter_covers() {
        let program = cfg::Program {
            functions: BTreeMap::from([(BASE, raw_function(BASE, BASE + 4))]),
            call_graph: BTreeMap::new(),
            stats: cfg::Stats::default(),
        };
        let img = TestImage {
            arch: Arch::Other,
            code: vec![0x90; 4],
        };
        assert!(lift_program(&img, &program).is_empty());
    }

    #[test]
    fn the_arch_is_carried_and_survives_simplify() {
        let lifted = lift_entry(&[0xC3]);
        assert_eq!(lifted.arch, Arch::X86_64);
        assert_eq!(simplify(&lifted).arch, Arch::X86_64);
        let a64 = lift_a64_entry(&[RET]);
        assert_eq!(a64.arch, Arch::Aarch64);
        assert_eq!(simplify(&a64).arch, Arch::Aarch64);
    }

    // ---- aarch64 lifting ----

    /// An in-memory aarch64 image holding `words` little-endian at [`BASE`].
    fn a64_image(words: &[u32]) -> TestImage {
        TestImage {
            arch: Arch::Aarch64,
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    /// Recover `words` as aarch64 code and lift the entry function.
    fn lift_a64_entry(words: &[u32]) -> LiftedFunction {
        let img = a64_image(words);
        let program = cfg::recover(&img).expect("aarch64 recovers");
        lift_function(&img, &program.functions[&BASE]).expect("aarch64 lifts")
    }

    /// `ret` (`ret x30`).
    const RET: u32 = 0xD65F_03C0;

    /// A conditional diamond, the aarch64 twin of [`DIAMOND`]:
    ///
    /// ```text
    /// 1000  cbz x0, 100c
    /// 1004  add x0, x0, #1
    /// 1008  b   100c
    /// 100c  ret
    /// ```
    const A64_DIAMOND: &[u32] = &[
        0xB400_0060, // +0: cbz x0, +12
        0x9100_0400, // +4: add x0, x0, #1
        0x1400_0001, // +8: b +4
        RET,         // +12: ret
    ];

    #[test]
    fn an_aarch64_function_lifts_with_the_recovered_block_structure() {
        let img = a64_image(A64_DIAMOND);
        let program = cfg::recover(&img).expect("aarch64 recovers");
        let func = &program.functions[&BASE];
        let lifted = lift_function(&img, func).expect("aarch64 lifts");
        assert_eq!(lifted.arch, Arch::Aarch64);
        assert_eq!(
            lifted.blocks.keys().copied().collect::<Vec<_>>(),
            func.blocks.keys().copied().collect::<Vec<_>>()
        );
        for (start, block) in &lifted.blocks {
            assert_eq!(ir::check(&block.stmts), Ok(()), "block {start:#x}");
            assert!(!block.truncated, "block {start:#x}");
        }
        // The cbz block ends in a guarded branch to the taken successor.
        let Some(ir::Stmt::Branch { kind, cond, target }) = lifted.blocks[&BASE].stmts.last()
        else {
            panic!("the cbz block must end in a Branch");
        };
        assert_eq!(*kind, ir::BranchKind::Jump);
        assert!(cond.is_some(), "cbz carries a guard");
        assert_eq!(*target, ir::Expr::constant(BASE + 12, ir::Width::W64));
    }

    #[test]
    fn an_unmapped_aarch64_va_truncates_the_block() {
        // One mapped nop, but the block claims eight bytes.
        let img = a64_image(&[0xD503_201F]);
        let func = raw_function(BASE, BASE + 8);
        let lifted = lift_function(&img, &func).expect("aarch64 lifts");
        let block = &lifted.blocks[&BASE];
        assert!(block.truncated);
        assert!(block.stmts.is_empty()); // the nop lifts to nothing
    }

    #[test]
    fn a_partial_aarch64_word_truncates_the_block_but_keeps_the_prefix() {
        // `movz x0, #42`, then two stray bytes: the second word cannot
        // decode, the first survives.
        let mut img = a64_image(&[0xD280_0540]);
        img.code.extend([0xC0, 0x03]);
        let func = raw_function(BASE, BASE + 8);
        let lifted = lift_function(&img, &func).expect("aarch64 lifts");
        let block = &lifted.blocks[&BASE];
        assert!(block.truncated);
        assert_eq!(block.stmts.len(), 1);
        assert_eq!(ir::check(&block.stmts), Ok(()));
    }

    #[test]
    fn an_unknown_aarch64_word_lifts_to_an_intrinsic_not_a_truncation() {
        // An LSE atomic (ldaddal) decodes as Unknown and lifts to the
        // sound clobber-everything intrinsic — coverage, not correctness,
        // is what is partial, so the walk continues to the ret.
        let img = a64_image(&[0xF8E9_0108, RET]);
        let func = raw_function(BASE, BASE + 8);
        let lifted = lift_function(&img, &func).expect("aarch64 lifts");
        let block = &lifted.blocks[&BASE];
        assert!(!block.truncated);
        assert_eq!(ir::check(&block.stmts), Ok(()));
        assert!(
            block
                .stmts
                .iter()
                .any(|s| matches!(s, ir::Stmt::Intrinsic { name, .. } if *name == "a64.unknown")),
            "{:?}",
            block.stmts
        );
    }

    #[test]
    fn a_simd_alu_word_lifts_into_the_ssa_path() {
        // orr v1.16b + ret: precise Or over the vector cells, no
        // a64.unknown clobber, block continues.
        let img = a64_image(&[0x4EA1_1C21, RET]);
        let func = raw_function(BASE, BASE + 8);
        let lifted = lift_function(&img, &func).expect("aarch64 lifts");
        let block = &lifted.blocks[&BASE];
        assert!(!block.truncated);
        assert_eq!(ir::check(&block.stmts), Ok(()));
        assert!(
            block.stmts.iter().any(|s| matches!(
                s,
                ir::Stmt::Assign {
                    value: ir::Expr::Binary { op: ir::BinOp::Or, .. },
                    ..
                }
            )),
            "{:?}",
            block.stmts
        );
        assert!(
            !block
                .stmts
                .iter()
                .any(|s| matches!(s, ir::Stmt::Intrinsic { name, .. } if *name == "a64.unknown")),
            "{:?}",
            block.stmts
        );
    }

    #[test]
    fn render_of_an_aarch64_return_function_is_exact() {
        let lifted = lift_a64_entry(&[RET]);
        assert_eq!(
            render(&lifted),
            "; sub_1000 @ 0x0000000000001000\n\
             loc_1000:\n\
             \x20 return x30\n\
             \x20 ; -> (none)\n"
        );
    }

    #[test]
    fn render_names_aarch64_registers() {
        // sub sp, sp, #16 ; movz x0, #42 ; ret.
        let lifted = lift_a64_entry(&[0xD100_43FF, 0xD280_0540, RET]);
        let text = render(&lifted);
        assert!(text.contains("sp := (sp - 0x10.q)"), "{text}");
        assert!(text.contains("x0 := 0x2a.q"), "{text}");
        assert!(text.contains("return x30"), "{text}");
        assert!(!text.contains("rax"), "{text}");
    }

    #[test]
    fn aarch64_lifting_is_deterministic() {
        let img = a64_image(A64_DIAMOND);
        let program = cfg::recover(&img).expect("aarch64 recovers");
        assert_eq!(lift_program(&img, &program), lift_program(&img, &program));
        let lifted = lift_function(&img, &program.functions[&BASE]);
        assert_eq!(lifted, lift_function(&img, &program.functions[&BASE]));
        assert_eq!(
            render(&lifted.expect("aarch64 lifts")),
            render(&lift_function(&img, &program.functions[&BASE]).expect("aarch64 lifts"))
        );
    }

    #[test]
    fn random_aarch64_words_lift_soundly_without_panicking() {
        // A deterministic LCG stream of arbitrary words through the whole
        // block walk: no panic, and every lifted block passes `ir::check`.
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut words = Vec::new();
        for _ in 0..256 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            words.push((state >> 24) as u32);
        }
        let img = a64_image(&words);
        let func = raw_function(BASE, BASE + 4 * words.len() as u64);
        let lifted = lift_function(&img, &func).expect("aarch64 lifts");
        for (start, block) in &lifted.blocks {
            assert_eq!(ir::check(&block.stmts), Ok(()), "block {start:#x}");
        }
        assert_eq!(lift_function(&img, &func), lift_function(&img, &func));
    }

    // ---- whole-program lifting ----

    #[test]
    fn lift_program_lifts_every_recovered_function() {
        // Entry calls a second function at +9.
        let code: &[u8] = &[
            0x31, 0xC0, // +0: xor eax, eax
            0xE8, 0x02, 0x00, 0x00, 0x00, // +2: call +9
            0xC3, // +7: ret
            0x90, // +8: pad
            0x90, // +9: callee: nop
            0xC3, // +10: ret
        ];
        let img = image(code);
        let program = recover(&img);
        assert_eq!(
            program.functions.keys().copied().collect::<Vec<_>>(),
            [BASE, BASE + 9]
        );
        let lifted = lift_program(&img, &program);
        assert_eq!(lifted.keys().copied().collect::<Vec<_>>(), [BASE, BASE + 9]);
        for (entry, func) in &program.functions {
            assert_eq!(
                lifted[entry],
                lift_function(&img, func).expect("x86-64 lifts")
            );
        }
    }

    #[test]
    fn an_empty_function_lifts_to_an_empty_lifted_function() {
        let img = image(&[0x90; 4]);
        let func = cfg::Function {
            entry: BASE,
            name: Some("empty".into()),
            blocks: BTreeMap::new(),
        };
        let lifted = lift_function(&img, &func).expect("x86-64 lifts");
        assert_eq!(lifted.entry, BASE);
        assert_eq!(lifted.name.as_deref(), Some("empty"));
        assert!(lifted.blocks.is_empty());
        assert_eq!(render(&lifted), "; empty @ 0x0000000000001000\n");
    }

    #[test]
    fn lifting_is_deterministic() {
        let img = image(DIAMOND);
        let program = recover(&img);
        assert_eq!(lift_program(&img, &program), lift_program(&img, &program));
        assert_eq!(
            lift_function(&img, &program.functions[&BASE]),
            lift_function(&img, &program.functions[&BASE])
        );
    }

    // ---- simplification ----

    #[test]
    fn simplify_preserves_the_block_structure() {
        let lifted = lift_entry(DIAMOND);
        let simplified = simplify(&lifted);
        assert_eq!(simplified.entry, lifted.entry);
        assert_eq!(simplified.name, lifted.name);
        assert_eq!(
            simplified.blocks.keys().copied().collect::<Vec<_>>(),
            lifted.blocks.keys().copied().collect::<Vec<_>>()
        );
        for (start, block) in &lifted.blocks {
            let sb = &simplified.blocks[start];
            assert_eq!(sb.start, block.start);
            assert_eq!(sb.end, block.end);
            assert_eq!(sb.successors, block.successors);
            assert_eq!(sb.truncated, block.truncated);
        }
    }

    #[test]
    fn every_simplified_block_passes_ir_check() {
        let lifted = simplify(&lift_entry(DIAMOND));
        for (start, block) in &lifted.blocks {
            assert_eq!(ir::check(&block.stmts), Ok(()), "block {start:#x}");
        }
    }

    #[test]
    fn simplify_drops_dead_flag_updates() {
        // xor eax, eax ; ret — the raw lift assigns every status flag;
        // nothing reads them before the return, so none survives. (The
        // xor itself is NOT folded to `rax := 0`: the lifter reads each
        // operand through its own temporary, and `irflow::propagate`
        // forwards only constants and plain copies, so the `x ^ x`
        // identity never sees syntactically equal sides. A documented
        // stay-sound limitation, not a promise.)
        let lifted = lift_entry(&[0x31, 0xC0, 0xC3]);
        let raw = &lifted.blocks[&BASE].stmts;
        assert!(
            raw.iter().any(|s| matches!(
                s,
                ir::Stmt::Assign { dst, .. } if dst.space == ir::Space::Flag
            )),
            "the raw lift does write flags: {raw:?}"
        );
        let simplified = simplify(&lifted);
        let stmts = &simplified.blocks[&BASE].stmts;
        assert!(stmts.len() < raw.len(), "{stmts:?}");
        assert!(
            stmts.iter().all(|s| !matches!(
                s,
                ir::Stmt::Assign { dst, .. } if dst.space == ir::Space::Flag
            )),
            "no flag write survives when nothing reads it: {stmts:?}"
        );
        assert_eq!(ir::check(stmts), Ok(()));
    }

    #[test]
    fn simplify_keeps_a_conditional_branch_and_its_guard() {
        let simplified = simplify(&lift_entry(DIAMOND));
        let cond = &simplified.blocks[&BASE];
        let Some(ir::Stmt::Branch { kind, cond: c, .. }) = cond.stmts.last() else {
            panic!("the conditional block still ends in a Branch");
        };
        assert_eq!(*kind, ir::BranchKind::Jump);
        assert!(c.is_some(), "the guard is observable and must survive");
    }

    #[test]
    fn simplify_is_deterministic_and_idempotent_on_the_fixture() {
        let lifted = lift_entry(DIAMOND);
        let once = simplify(&lifted);
        assert_eq!(once, simplify(&lifted));
        assert_eq!(once, simplify(&once));
    }

    // ---- rendering ----

    #[test]
    fn render_of_a_one_return_function_is_exact() {
        // A single `ret`: the golden dump, byte for byte.
        let lifted = lift_entry(&[0xC3]);
        assert_eq!(
            render(&lifted),
            "; sub_1000 @ 0x0000000000001000\n\
             loc_1000:\n\
             \x20 t0.q := load.q [rsp]\n\
             \x20 rsp := (rsp + 0x8.q)\n\
             \x20 return t0.q\n\
             \x20 ; -> (none)\n"
        );
    }

    #[test]
    fn render_shows_header_labels_ir_and_edges() {
        let lifted = lift_entry(DIAMOND);
        let text = render(&lifted);
        assert!(text.starts_with("; sub_1000 @ 0x0000000000001000\n"), "{text}");
        for label in ["loc_1000:", "loc_1004:", "loc_1009:", "loc_100b:"] {
            assert!(text.contains(label), "{text}");
        }
        // The conditional block's edge line lists taken then fallthrough.
        assert!(text.contains("  ; -> loc_1009, loc_1004\n"), "{text}");
        // The return block has no successors.
        assert!(text.contains("  ; -> (none)\n"), "{text}");
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn render_names_a_named_function() {
        let mut lifted = lift_entry(&[0xC3]);
        lifted.name = Some("main".into());
        assert!(render(&lifted).starts_with("; main @ 0x0000000000001000\n"));
    }

    #[test]
    fn render_marks_a_truncated_block() {
        let img = image(&[0x55, 0xD8, 0x00, 0x06]);
        let func = raw_function(BASE, BASE + 4);
        let lifted = lift_function(&img, &func).expect("x86-64 lifts");
        assert!(render(&lifted).contains("  ; (truncated)\n"));
    }

    #[test]
    fn render_is_deterministic() {
        let lifted = lift_entry(DIAMOND);
        assert_eq!(render(&lifted), render(&lifted));
    }
}
