//! Control-flow recovery: recursive descent from entry points, exports,
//! and symbol tables into functions, basic blocks, and a call graph.
//!
//! Written once against the trait layer: outside its tests this module
//! names only [`crate::model`] and [`crate::error`] — never a container
//! format or an instruction set.
//!
//! # Method
//!
//! [`recover`] seeds a function worklist from every high-precision
//! function-start source that lies in an executable region —
//! [`Image::entry_points`], each [`SymbolKind::Function`] symbol, the Go
//! pclntab, and the compiler's unwind/metadata table
//! ([`Image::function_starts_hint`]: Mach-O `LC_FUNCTION_STARTS`, PE
//! `.pdata`, ELF `.eh_frame`) — then descends recursively. Within a
//! function, instructions are decoded linearly inside executable regions;
//! a basic block ends at the first non-[`Flow::Sequential`] instruction or
//! on reaching an already-known block leader. Conditional branches
//! contribute both the taken and fall-through edges; calls contribute the
//! return fall-through edge *and* seed the callee as a new function; a
//! later-discovered branch target that lands mid-block splits the
//! existing block at that instruction boundary. Every worklist is a
//! B-tree iterated in address order, so recovery is deterministic: the
//! same image always yields the same [`Program`].
//!
//! Once those sources drain, a second pass harvests *address-taken* code
//! pointers ([`crate::xref::XrefKind::AddressOf`]) from the functions
//! recovered so far: a function whose address is only stored, never called
//! — a Rust `main` handed to `lang_start`, a callback, a vtable slot —
//! is invisible to the seeds above but leaves an `AddressOf` reference. A
//! harvested target is seeded only when it lies in an executable region,
//! does not fall in the interior of an already-recovered block, and opens
//! with a canonical prologue ([`crate::funcs::looks_like_function_start`]),
//! so a pointer into data or into the middle of a function never invents
//! one. Newly recovered functions may take further addresses, so the pass
//! iterates to a bounded fixpoint.
//!
//! # Under-approximation, never a guess
//!
//! - An instruction that cannot be decoded — unknown encoding, bytes
//!   outside every executable region or with no file backing, or a
//!   virtual-address wraparound — ends its block with
//!   [`Terminator::Undecodable`] and no successors.
//! - Indirect jumps contribute no intra-function edges — except at a
//!   jump site whose *proven* table targets the caller supplied to
//!   [`recover_with_tables`]: those fold in as ordinary successors (and
//!   their code is walked into blocks) while the terminator stays
//!   [`Terminator::IndirectJump`], the bytes' truth. Indirect calls
//!   and jumps through a statically addressed memory cell
//!   ([`crate::model::Decoded::mem_target`]) are matched against
//!   [`Image::import_slots`]: a hit becomes a
//!   [`CallTarget::Import`] call-graph edge (a matching jump is a tail
//!   call, and when it is the function's first instruction the function
//!   is an import *thunk*, so direct calls to it also resolve to the
//!   import). An unmatched indirect call is [`CallTarget::Unknown`].
//!   AArch64 import calls (ADRP+LDR sequences) are not resolved and
//!   appear as `Unknown` — see `Decoded::mem_target`.
//! - Branch and call targets outside every executable region are
//!   recorded as edges ([`BasicBlock::successors`] /
//!   [`CallTarget::Function`]) but never decoded.
//! - A branch target inside an existing block but *not* on a decoded
//!   instruction boundary (overlapping x86 code) becomes a separate,
//!   overlapping block instead of a split.
//!
//! # Resource caps
//!
//! Hostile images reach the decoder, so [`Config`] hard-caps the total
//! decoded instructions, total blocks (splits included), and the size of
//! each worklist. Hitting the instruction or block cap stops recovery —
//! the block being built ends with [`Terminator::Truncated`] and all
//! remaining pending work is dropped; hitting the worklist cap drops the
//! newly discovered target. Each cap sets its flag in [`Program::stats`],
//! so a truncated result is always identifiable. The result is still
//! deterministic and recovery always terminates.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{ParseError, Result};
use crate::model::{Decoder, Flow, Image, SymbolKind, decoder_for};

/// Enough bytes for any single instruction on the supported ISAs
/// (x86-64 caps at 15; A64 is always 4).
const MAX_INSN_BYTES: u64 = 16;

/// Upper bound on address-taken (`AddressOf`) seeding rounds. Each round
/// harvests code pointers from the functions recovered so far and seeds
/// any that begin a new function; the loop stops early once a round adds
/// nothing. The cap bounds the work on hostile images regardless.
const MAX_ADDRESSOF_ROUNDS: usize = 8;

/// Resource caps for [`recover_with`]. See the module docs for the
/// semantics of hitting each cap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Maximum instructions decoded across the whole image.
    pub max_instructions: usize,
    /// Maximum basic blocks across the whole image (splits included).
    pub max_blocks: usize,
    /// Maximum size of any one worklist (pending function seeds, or
    /// pending block leaders within a function).
    pub max_worklist: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_instructions: 1_000_000,
            max_blocks: 262_144,
            max_worklist: 262_144,
        }
    }
}

/// Why a basic block ended. [`BasicBlock::successors`] restates the
/// intra-function edges in the fixed order documented per variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    /// Unconditional direct jump. Successors: `[target]`.
    Jump(u64),
    /// Conditional branch. Successors: `[taken, fallthrough]`
    /// (deduplicated when equal).
    CondJump { taken: u64, fallthrough: u64 },
    /// Jump through a register or memory; `import` names the imported
    /// symbol when the memory operand is an import slot (a tail call).
    /// Successors: none — unless a proven jump table at this site was
    /// supplied to [`recover_with_tables`], which folds its targets in
    /// (ascending, deduplicated) while the terminator stays the bytes'
    /// truth.
    IndirectJump { import: Option<String> },
    /// Direct call; the callee becomes a function, not a successor.
    /// Successors: `[fallthrough]`.
    Call { target: u64, fallthrough: u64 },
    /// Indirect call; `import` as for [`Terminator::IndirectJump`].
    /// Successors: `[fallthrough]`.
    IndirectCall {
        import: Option<String>,
        fallthrough: u64,
    },
    /// Return to the caller. Successors: none.
    Return,
    /// Software interrupt / syscall / trap; execution may resume.
    /// Successors: `[fallthrough]`.
    Interrupt { fallthrough: u64 },
    /// Halt. Successors: none.
    Halt,
    /// The next instruction could not be decoded (unknown encoding,
    /// unreadable or out-of-region bytes, VA wraparound). The block is
    /// under-approximated: no successors, nothing guessed. `end` is the
    /// address of the undecodable byte.
    Undecodable,
    /// [`Config::max_instructions`] or [`Config::max_blocks`] hit while
    /// building this block ([`Stats`] has the flag); recovery stopped
    /// here. Successors: none.
    Truncated,
    /// Linear decode reached an already-known block leader.
    /// Successors: `[leader]`.
    FallThrough(u64),
}

/// A maximal single-entry straight-line run of instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    /// VA of the first instruction.
    pub start: u64,
    /// VA one past the last decoded byte (exclusive).
    pub end: u64,
    /// Why the block ended.
    pub terminator: Terminator,
    /// Intra-function control-flow edges, in the fixed order documented
    /// on [`Terminator`]. Targets outside every executable region are
    /// recorded here but never decoded.
    pub successors: Vec<u64>,
}

/// One recovered function: an entry point and the blocks reachable from
/// it by intra-function edges. Functions that share code each carry
/// their own copy of the shared blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    /// Entry VA.
    pub entry: u64,
    /// Name from [`Image::symbols`], when one matches the entry VA
    /// ([`SymbolKind::Function`] symbols take precedence).
    pub name: Option<String>,
    /// Blocks keyed by start VA.
    pub blocks: BTreeMap<u64, BasicBlock>,
}

/// One outgoing call-graph edge.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CallTarget {
    /// Direct call to a recovered (or out-of-region) function entry.
    Function(u64),
    /// Call resolved to an import, either through an import slot's
    /// memory operand or through a direct call to a jump thunk.
    Import(String),
    /// Indirect call whose target could not be resolved.
    Unknown,
}

/// Recovery statistics and truncation flags. Any set flag means the
/// [`Program`] is a documented under-approximation of an over-budget
/// image, not a complete recovery.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stats {
    /// Instructions decoded.
    pub instructions: usize,
    /// Basic blocks built (splits included; blocks duplicated into
    /// multiple functions are counted per copy).
    pub blocks: usize,
    /// [`Config::max_instructions`] was hit; recovery stopped there.
    pub instruction_cap_hit: bool,
    /// [`Config::max_blocks`] was hit; recovery stopped there.
    pub block_cap_hit: bool,
    /// [`Config::max_worklist`] was hit; at least one discovered target
    /// was dropped.
    pub worklist_cap_hit: bool,
    /// Number of parallel fixed-point rounds the [`crate::parallel`]
    /// engine ran. Left `0` by [`recover`] (single-threaded, no rounds).
    pub rounds: usize,
    /// Worker-pool size the [`crate::parallel`] engine used. Left `0` by
    /// [`recover`]. This field is the *only* part of a parallel
    /// [`Program`] that varies with the requested thread count — the
    /// recovered functions, call graph, and every other stat are
    /// byte-for-byte thread-count-independent.
    pub threads_used: usize,
    /// The [`crate::parallel`] function cap was hit; discovered functions
    /// were dropped. Left `false` by [`recover`], which has no such cap.
    pub function_cap_hit: bool,
    /// Indirect-jump blocks whose successors were folded from a proven
    /// jump table supplied to [`recover_with_tables`] (a dispatch block
    /// copied into multiple functions counts per copy, exactly like
    /// [`Stats::blocks`]). Left `0` by [`recover`], which folds nothing.
    pub tables_folded: usize,
    /// Supplied jump-table targets dropped instead of folded because they
    /// lie outside every executable region: no block could ever back such
    /// an edge, and an edge without a block would corrupt every
    /// downstream pass. The drop is counted here, never silent.
    pub table_targets_dropped: usize,
}

/// The recovered control-flow view of an image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    /// Recovered functions keyed by entry VA.
    pub functions: BTreeMap<u64, Function>,
    /// Outgoing call edges keyed by caller entry VA. Functions that
    /// make no calls have no entry.
    pub call_graph: BTreeMap<u64, BTreeSet<CallTarget>>,
    pub stats: Stats,
}

/// Recover functions, basic blocks, and the call graph from `image`
/// with the default [`Config`] caps.
///
/// Fails with [`ParseError::Unsupported`] when the image's architecture
/// has no decoder ([`crate::model::Arch::Other`]). Never panics on any
/// input: undecodable code is under-approximated (see module docs) and
/// hostile images are bounded by the caps.
pub fn recover(image: &dyn Image) -> Result<Program> {
    recover_with(image, &Config::default())
}

/// [`recover`] with caller-supplied resource caps.
pub fn recover_with(image: &dyn Image, config: &Config) -> Result<Program> {
    recover_with_tables(image, config, &BTreeMap::new())
}

/// [`recover_with`], additionally folding proven jump-table successors
/// into the recovered CFG.
///
/// `tables` maps an indirect jump's site VA to the targets a table proof
/// established for it — the shape [`crate::jumptable::successor_map`]
/// returns, taken here as plain data so the proof pass stays layered
/// above this module. At a jump site the map names (and [`crate::cfg`]
/// did not already resolve as an import tail call), the proven targets
/// become the block's successors — deduplicated, ascending — and are
/// walked into blocks like any other discovered edge, so a fold can
/// reach case bodies no prior seed found. The terminator stays
/// [`Terminator::IndirectJump`]: only the edges gain what was proven. A
/// supplied target outside every executable region is dropped, with the
/// drop counted in [`Stats::table_targets_dropped`], never silent.
///
/// With an empty map this is exactly [`recover_with`]. Because the case
/// bodies a table targets may themselves hold further tables that only
/// resolve once their code exists, callers wanting the fixpoint should
/// use [`crate::jumptable::resolve_folded`], the one entry point that
/// iterates recover → resolve rounds; this function is its single round.
pub fn recover_with_tables(
    image: &dyn Image,
    config: &Config,
    tables: &BTreeMap<u64, Vec<u64>>,
) -> Result<Program> {
    let mut rec = Recovery::new(image, config, tables)?;

    rec.drain();

    // Second-order seeding from address-taken code pointers. A function
    // whose address is only *stored* (never called) — a Rust `main` handed
    // to `lang_start`, a callback, a vtable slot — surfaces only as an
    // [`crate::xref::XrefKind::AddressOf`] reference. Harvest those targets
    // that begin a plausible function and were missed by every prior
    // source, seed them, and drain again; a newly recovered function may
    // itself take further addresses, so iterate to a (bounded) fixpoint.
    for _ in 0..MAX_ADDRESSOF_ROUNDS {
        if rec.capped() || !rec.seed_from_addressof() {
            break;
        }
        rec.drain();
    }

    // Resolve direct calls to import thunks: `call f` where `f`'s first
    // instruction is `jmp [import slot]` is a call to that import.
    let thunks = rec.thunks;
    let call_graph = rec
        .call_graph
        .into_iter()
        .map(|(caller, targets)| {
            let targets = targets
                .into_iter()
                .map(|t| match t {
                    CallTarget::Function(va) => thunks
                        .get(&va)
                        .map_or(CallTarget::Function(va), |n| CallTarget::Import(n.clone())),
                    other => other,
                })
                .collect();
            (caller, targets)
        })
        .collect();

    Ok(Program {
        functions: rec.functions,
        call_graph,
        stats: rec.stats,
    })
}

/// Whole-image recovery state.
struct Recovery<'a> {
    image: &'a dyn Image,
    decoder: &'a dyn Decoder,
    bytes: &'a [u8],
    /// Executable `[start, end)` ranges, sorted.
    exec: Vec<(u64, u64)>,
    /// Import slot VA -> imported name.
    slots: BTreeMap<u64, String>,
    /// Function entry VA -> symbol name.
    names: BTreeMap<u64, String>,
    /// Proven jump-table successors, jump site VA -> targets, supplied
    /// by [`recover_with_tables`] (empty for plain [`recover`]).
    tables: &'a BTreeMap<u64, Vec<u64>>,
    config: &'a Config,
    stats: Stats,
    /// Entries of functions recognized as import thunks, with the
    /// imported name.
    thunks: BTreeMap<u64, String>,
    /// Function entries awaiting recovery (address order).
    pending: BTreeSet<u64>,
    functions: BTreeMap<u64, Function>,
    call_graph: BTreeMap<u64, BTreeSet<CallTarget>>,
}

/// Per-function descent state.
struct FuncCfg {
    entry: u64,
    blocks: BTreeMap<u64, BasicBlock>,
    /// Every known block start (built or pending).
    leaders: BTreeSet<u64>,
    /// Leaders awaiting a block build (address order).
    pending: BTreeSet<u64>,
    /// Start VA of every instruction decoded in this function, for
    /// boundary-correct block splitting.
    insn_starts: BTreeSet<u64>,
}

impl<'a> Recovery<'a> {
    /// Build recovery state for `image` and seed its worklist from every
    /// high-precision function-start source: declared entry points,
    /// `Function`-kind symbols, Go pclntab entries, and unwind/metadata
    /// tables (Mach-O `LC_FUNCTION_STARTS`, PE `.pdata`, ELF `.eh_frame`).
    /// Each seed is filtered through [`Recovery::seed_function`]'s
    /// executable-region and dedup checks. Address-taken seeding happens
    /// later, once these functions have been recovered.
    ///
    /// Fails with [`ParseError::Unsupported`] when the architecture has no
    /// decoder — the same contract as [`recover`].
    fn new(
        image: &'a dyn Image,
        config: &'a Config,
        tables: &'a BTreeMap<u64, Vec<u64>>,
    ) -> Result<Recovery<'a>> {
        let arch = image.arch();
        let Some(decoder) = decoder_for(arch) else {
            return Err(ParseError::Unsupported(format!(
                "control-flow recovery: no decoder for architecture {arch:?}"
            )));
        };

        // Executable address ranges, in address order.
        let mut exec: Vec<(u64, u64)> = image
            .regions()
            .iter()
            .filter(|r| r.perms.x && r.size > 0)
            .map(|r| (r.va, r.va.saturating_add(r.size)))
            .collect();
        exec.sort_unstable();
        exec.dedup();

        // slot VA -> import name (first name wins on a duplicate slot).
        let mut slots = BTreeMap::new();
        for s in image.import_slots() {
            slots.entry(s.slot_va).or_insert(s.name);
        }

        // entry VA -> name; function symbols take precedence, and within a
        // kind the (VA, name)-sorted order makes the pick deterministic.
        let symbols = image.symbols();
        let mut names: BTreeMap<u64, String> = BTreeMap::new();
        for sym in symbols.iter().filter(|s| s.kind == SymbolKind::Function) {
            names.entry(sym.va).or_insert_with(|| sym.name.clone());
        }
        for sym in &symbols {
            names.entry(sym.va).or_insert_with(|| sym.name.clone());
        }

        // A Go binary names its functions in the pclntab, not the symbol
        // table, so recover those once and use them for both seeding and
        // naming. Loader symbols still take precedence (`or_insert`); the
        // pclntab fills in the far larger stripped-binary case. Empty and
        // cheap on any non-Go image.
        let go_funcs = crate::gopcln::recover(image);
        for f in &go_funcs {
            names.entry(f.va).or_insert_with(|| f.name.clone());
        }

        let mut rec = Recovery {
            image,
            decoder,
            bytes: image.bytes(),
            exec,
            slots,
            names,
            tables,
            config,
            stats: Stats::default(),
            thunks: BTreeMap::new(),
            pending: BTreeSet::new(),
            functions: BTreeMap::new(),
            call_graph: BTreeMap::new(),
        };

        for ep in image.entry_points() {
            rec.seed_function(ep);
        }
        for sym in symbols.iter().filter(|s| s.kind == SymbolKind::Function) {
            rec.seed_function(sym.va);
        }
        for f in &go_funcs {
            rec.seed_function(f.va);
        }
        // Compiler-emitted unwind/metadata tables list exact function
        // starts and survive stripping — the symbol table does not.
        // Seeding them recovers functions that are never *called* directly
        // (e.g. a Rust `main` whose address is only handed to `lang_start`).
        for va in image.function_starts_hint() {
            rec.seed_function(va);
        }

        Ok(rec)
    }

    /// The executable region containing `va`, if any (first match in
    /// address order).
    fn exec_region(&self, va: u64) -> Option<(u64, u64)> {
        self.exec.iter().copied().find(|&(s, e)| va >= s && va < e)
    }

    /// Whether a recovery-stopping cap has been hit.
    fn capped(&self) -> bool {
        self.stats.instruction_cap_hit || self.stats.block_cap_hit
    }

    /// Queue `va` as a function entry if it is executable and new.
    fn seed_function(&mut self, va: u64) {
        if self.exec_region(va).is_none()
            || self.functions.contains_key(&va)
            || self.pending.contains(&va)
        {
            return;
        }
        if self.pending.len() >= self.config.max_worklist {
            self.stats.worklist_cap_hit = true;
            return;
        }
        self.pending.insert(va);
    }

    /// Recover every pending function entry, in address order, until the
    /// worklist drains or a recovery-stopping cap is hit.
    fn drain(&mut self) {
        while let Some(entry) = self.pending.pop_first() {
            if self.functions.contains_key(&entry) {
                continue;
            }
            if self.capped() {
                break; // truncated: flags already set, pending work dropped
            }
            let func = self.recover_function(entry);
            self.functions.insert(entry, func);
        }
    }

    /// Harvest address-taken code pointers from the functions recovered so
    /// far and seed any that begin a new, plausible function. Returns
    /// whether at least one fresh entry was queued.
    ///
    /// Gating keeps this from inventing functions: a target must lie in an
    /// executable region (never in data — that is the whole point of
    /// [`Walk::is_data`] in the xref pass, and non-executable targets are
    /// rejected here too), must not fall in the interior of an
    /// already-recovered block (so a pointer into the *middle* of a
    /// function cannot split it into a bogus new one), and must open with a
    /// canonical prologue ([`crate::funcs::looks_like_function_start`]).
    fn seed_from_addressof(&mut self) -> bool {
        let mut added = false;
        for target in addressof_seeds(self.image, &self.functions) {
            if self.pending.contains(&target) || self.functions.contains_key(&target) {
                continue;
            }
            let before = self.pending.len();
            self.seed_function(target);
            added |= self.pending.len() != before;
        }
        added
    }

    /// Recursive descent over one function.
    fn recover_function(&mut self, entry: u64) -> Function {
        let mut f = FuncCfg {
            entry,
            blocks: BTreeMap::new(),
            leaders: BTreeSet::from([entry]),
            pending: BTreeSet::from([entry]),
            insn_starts: BTreeSet::new(),
        };
        while let Some(start) = f.pending.pop_first() {
            if f.blocks.contains_key(&start) {
                continue;
            }
            if self.capped() {
                break; // truncated: pending leaders dropped
            }
            if self.stats.blocks >= self.config.max_blocks {
                self.stats.block_cap_hit = true;
                break;
            }
            self.build_block(&mut f, start);
        }
        Function {
            entry,
            name: self.names.get(&entry).cloned(),
            blocks: f.blocks,
        }
    }

    /// Decode one basic block starting at `start`, insert it, and queue
    /// its successors.
    fn build_block(&mut self, f: &mut FuncCfg, start: u64) {
        let entry = f.entry;
        let mut cur = start;
        // Each arm yields (terminator, exclusive end, successors).
        let (terminator, end, successors) = loop {
            if cur != start && f.leaders.contains(&cur) {
                break (Terminator::FallThrough(cur), cur, vec![cur]);
            }
            if self.stats.instructions >= self.config.max_instructions {
                self.stats.instruction_cap_hit = true;
                break (Terminator::Truncated, cur, Vec::new());
            }
            let Some((_, region_end)) = self.exec_region(cur) else {
                break (Terminator::Undecodable, cur, Vec::new());
            };
            let Some(off) = self.image.va_to_offset(cur) else {
                break (Terminator::Undecodable, cur, Vec::new());
            };
            // Window an instruction may decode from: bounded by the
            // region, the file, and the longest supported encoding, so a
            // decode can never run past the executable range.
            let window = (region_end - cur)
                .min(MAX_INSN_BYTES)
                .min(self.bytes.len().saturating_sub(off) as u64)
                as usize;
            let Ok(d) = self.decoder.decode_flow(&self.bytes[off..off + window], cur) else {
                break (Terminator::Undecodable, cur, Vec::new());
            };
            self.stats.instructions += 1;
            f.insn_starts.insert(cur);
            let Some(next) = cur.checked_add(u64::from(d.length)) else {
                break (Terminator::Undecodable, cur, Vec::new());
            };
            match d.flow {
                Flow::Sequential => cur = next,
                Flow::Jump(t) => break (Terminator::Jump(t), next, vec![t]),
                Flow::CondJump(t) => {
                    let succ = if t == next { vec![t] } else { vec![t, next] };
                    break (
                        Terminator::CondJump {
                            taken: t,
                            fallthrough: next,
                        },
                        next,
                        succ,
                    );
                }
                Flow::IndirectJump => {
                    let import = d.mem_target.and_then(|s| self.slots.get(&s).cloned());
                    if let Some(name) = &import {
                        // A jump through an import slot is a tail call;
                        // as the function's first instruction it makes
                        // the whole function an import thunk.
                        if cur == entry {
                            self.thunks.insert(entry, name.clone());
                        }
                        self.add_call_edge(entry, CallTarget::Import(name.clone()));
                    }
                    // A proven jump table at this site folds its targets
                    // in as successors (an import tail call leaves the
                    // function; a table entry for it is not believed).
                    let succ = if import.is_none() {
                        self.table_successors(cur)
                    } else {
                        Vec::new()
                    };
                    break (Terminator::IndirectJump { import }, next, succ);
                }
                Flow::Call(t) => {
                    self.add_call_edge(entry, CallTarget::Function(t));
                    self.seed_function(t);
                    break (
                        Terminator::Call {
                            target: t,
                            fallthrough: next,
                        },
                        next,
                        vec![next],
                    );
                }
                Flow::IndirectCall => {
                    let import = d.mem_target.and_then(|s| self.slots.get(&s).cloned());
                    let edge = import
                        .clone()
                        .map_or(CallTarget::Unknown, CallTarget::Import);
                    self.add_call_edge(entry, edge);
                    break (
                        Terminator::IndirectCall {
                            import,
                            fallthrough: next,
                        },
                        next,
                        vec![next],
                    );
                }
                Flow::Return => break (Terminator::Return, next, Vec::new()),
                Flow::Interrupt => {
                    break (Terminator::Interrupt { fallthrough: next }, next, vec![next]);
                }
                Flow::Halt => break (Terminator::Halt, next, Vec::new()),
            }
        };

        f.blocks.insert(
            start,
            BasicBlock {
                start,
                end,
                terminator,
                successors: successors.clone(),
            },
        );
        self.stats.blocks += 1;

        for s in successors {
            if self.exec_region(s).is_some() {
                self.add_leader(f, s);
            }
            // Out-of-region targets stay recorded as edges, undecoded.
        }
    }

    /// Register `target` as a block leader: split the block it lands
    /// inside (when it hits an instruction boundary) or queue it for a
    /// fresh build.
    fn add_leader(&mut self, f: &mut FuncCfg, target: u64) {
        if f.leaders.contains(&target) {
            return;
        }
        // A later-discovered target inside an already-built block, on a
        // decoded instruction boundary, splits that block. (Inside a
        // block but *not* on a boundary — overlapping code — falls
        // through below and becomes a separate, overlapping block.)
        let split_at = f
            .blocks
            .range(..=target)
            .next_back()
            .filter(|&(&bs, b)| target > bs && target < b.end && f.insn_starts.contains(&target))
            .map(|(&bs, _)| bs);
        if let Some(bs) = split_at {
            if self.stats.blocks >= self.config.max_blocks {
                self.stats.block_cap_hit = true;
                return;
            }
            let Some(head) = f.blocks.get_mut(&bs) else {
                return; // unreachable: `bs` came from this map
            };
            let tail = BasicBlock {
                start: target,
                end: head.end,
                terminator: head.terminator.clone(),
                successors: head.successors.clone(),
            };
            head.end = target;
            head.terminator = Terminator::FallThrough(target);
            head.successors = vec![target];
            f.blocks.insert(target, tail);
            f.leaders.insert(target);
            self.stats.blocks += 1;
            return;
        }
        if f.pending.len() >= self.config.max_worklist {
            self.stats.worklist_cap_hit = true;
            return;
        }
        f.leaders.insert(target);
        f.pending.insert(target);
    }

    fn add_call_edge(&mut self, caller: u64, target: CallTarget) {
        self.call_graph.entry(caller).or_default().insert(target);
    }

    /// The proven jump-table successors folded in at jump site `va`:
    /// the supplied targets kept to executable regions, deduplicated,
    /// ascending. A target outside every executable region is dropped
    /// and counted ([`Stats::table_targets_dropped`]) — no block could
    /// ever back that edge. Empty (and free) when the caller supplied
    /// no table for `va`.
    fn table_successors(&mut self, va: u64) -> Vec<u64> {
        let Some(targets) = self.tables.get(&va) else {
            return Vec::new();
        };
        let mut fold: Vec<u64> = targets
            .iter()
            .copied()
            .filter(|&t| self.exec_region(t).is_some())
            .collect();
        self.stats.table_targets_dropped += targets.len() - fold.len();
        fold.sort_unstable();
        fold.dedup();
        if !fold.is_empty() {
            self.stats.tables_folded += 1;
        }
        fold
    }
}

/// One function recovered in isolation by [`recover_function`]: the
/// function itself plus everything the caller needs to stitch it into a
/// whole-program view without any shared state during the descent.
///
/// This is the pure work unit the [`crate::parallel`] engine runs across
/// threads: it is a total function of `(image, entry, config)`, so two
/// calls with equal inputs return equal `FunctionCfg`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCfg {
    /// The recovered function (entry, name, blocks).
    pub function: Function,
    /// Outgoing call edges of this function, *before* import-thunk
    /// resolution (direct calls appear as [`CallTarget::Function`] even
    /// when the callee turns out to be a thunk — the caller resolves
    /// those against the thunk map it assembles from every function's
    /// [`FunctionCfg::thunk`]).
    pub call_edges: BTreeSet<CallTarget>,
    /// Direct-call targets that land in an executable region: the newly
    /// discovered function entries this function contributes to the
    /// worklist. A subset of the [`CallTarget::Function`] entries in
    /// [`FunctionCfg::call_edges`].
    pub callees: BTreeSet<u64>,
    /// `Some(name)` when this function is an import thunk — its first
    /// instruction is a jump through an import slot — so a direct call to
    /// its entry resolves to `name`.
    pub thunk: Option<String>,
    /// Per-function decode counts and cap flags. Caps in [`Config`] are
    /// applied *per function* here (each function gets the full budget),
    /// unlike [`recover`], whose caps are whole-image. The
    /// [`crate::parallel`] driver sums `instructions`/`blocks` and ORs the
    /// flags across functions.
    pub stats: Stats,
}

/// Recover exactly one function: descend from `entry` into its blocks and
/// call edges, *without* recursing into or queuing its callees.
///
/// A pure, self-contained unit of work with no cross-function shared
/// state — the building block the [`crate::parallel`] engine distributes
/// across threads. Fails with [`ParseError::Unsupported`] when the image's
/// architecture has no decoder, exactly as [`recover`] does. Never panics:
/// undecodable code is under-approximated and the [`Config`] caps
/// (applied per function; see [`FunctionCfg::stats`]) bound hostile input.
pub fn recover_function(image: &dyn Image, entry: u64, config: &Config) -> Result<FunctionCfg> {
    let arch = image.arch();
    let Some(decoder) = decoder_for(arch) else {
        return Err(ParseError::Unsupported(format!(
            "control-flow recovery: no decoder for architecture {arch:?}"
        )));
    };

    let mut exec = exec_ranges(image);
    exec.sort_unstable();
    exec.dedup();

    let mut slots = BTreeMap::new();
    for s in image.import_slots() {
        slots.entry(s.slot_va).or_insert(s.name);
    }

    let mut w = FuncWorker {
        image,
        decoder,
        bytes: image.bytes(),
        exec: &exec,
        slots: &slots,
        config,
        entry,
        stats: Stats::default(),
        call_edges: BTreeSet::new(),
        callees: BTreeSet::new(),
        thunk: None,
        blocks: BTreeMap::new(),
        leaders: BTreeSet::from([entry]),
        pending: BTreeSet::from([entry]),
        insn_starts: BTreeSet::new(),
    };

    while let Some(start) = w.pending.pop_first() {
        if w.blocks.contains_key(&start) {
            continue;
        }
        if w.capped() {
            break; // truncated: pending leaders dropped
        }
        if w.stats.blocks >= w.config.max_blocks {
            w.stats.block_cap_hit = true;
            break;
        }
        w.build_block(start);
    }

    Ok(FunctionCfg {
        function: Function {
            entry,
            name: name_for(image, entry),
            blocks: w.blocks,
        },
        call_edges: w.call_edges,
        callees: w.callees,
        thunk: w.thunk,
        stats: w.stats,
    })
}

/// The complete set of function entries [`recover`] seeds from, in
/// address order: the high-precision sources ([`Image::entry_points`],
/// [`SymbolKind::Function`] symbols, the Go pclntab, and unwind/metadata
/// [`Image::function_starts_hint`]) kept to executable regions, plus the
/// address-taken (`AddressOf`) fixpoint `recover` runs on top of them.
/// Exposed so the [`crate::parallel`] engine seeds its worklist
/// identically to the single-threaded driver; callees are omitted (the
/// engine rediscovers them by descent, and recovery is seed-monotone).
pub fn function_seeds(image: &dyn Image) -> Vec<u64> {
    let exec = exec_ranges(image);
    let in_exec = |va: u64| exec.iter().any(|&(s, e)| va >= s && va < e);
    let mut seeds: BTreeSet<u64> = BTreeSet::new();
    for ep in image.entry_points() {
        if in_exec(ep) {
            seeds.insert(ep);
        }
    }
    for s in image.symbols() {
        if s.kind == SymbolKind::Function && in_exec(s.va) {
            seeds.insert(s.va);
        }
    }
    // Unwind/metadata tables (Mach-O `LC_FUNCTION_STARTS`, PE `.pdata`,
    // ELF `.eh_frame`) name exact function starts and survive stripping.
    // Mirror `recover`'s seeding so the parallel engine starts identically.
    for va in image.function_starts_hint() {
        if in_exec(va) {
            seeds.insert(va);
        }
    }
    // A Go binary's pclntab enumerates every function it contains, which
    // is the only complete seed set for a stripped Go image; fold those
    // entries in too. Empty and cheap on any non-Go image.
    for f in crate::gopcln::recover(image) {
        if in_exec(f.va) {
            seeds.insert(f.va);
        }
    }

    // Address-taken seeds (the `AddressOf` fixpoint `recover` runs). The
    // high-precision seeds above are recovered first; each round harvests
    // code pointers from the functions found so far and adds any that begin
    // a new function, exactly as [`Recovery::seed_from_addressof`] does, so
    // the parallel engine's seed set matches the single-threaded driver's.
    // Callees are intentionally *not* folded in here — the parallel engine
    // rediscovers them by descent, and recovery is seed-monotone.
    let config = Config::default();
    let mut extra: BTreeSet<u64> = BTreeSet::new();
    for _ in 0..MAX_ADDRESSOF_ROUNDS {
        let functions = recover_base_plus(image, &config, &extra);
        let mut grew = false;
        for target in addressof_seeds(image, &functions) {
            grew |= extra.insert(target);
        }
        if !grew {
            break;
        }
    }
    seeds.extend(extra);

    seeds.into_iter().collect()
}

/// Recover from the high-precision seeds plus the `extra` address-taken
/// seeds and return just the functions — the per-round program the parallel
/// seed fixpoint walks to harvest more [`crate::xref::XrefKind::AddressOf`]
/// targets. [`Recovery::new`] already seeds the high-precision sources, so
/// only `extra` is added here. Mirrors [`recover`]'s drain (callees
/// discovered, no address-taken fixpoint).
fn recover_base_plus(
    image: &dyn Image,
    config: &Config,
    extra: &BTreeSet<u64>,
) -> BTreeMap<u64, Function> {
    let no_tables = BTreeMap::new();
    let Ok(mut rec) = Recovery::new(image, config, &no_tables) else {
        return BTreeMap::new();
    };
    for &va in extra {
        rec.seed_function(va);
    }
    rec.drain();
    rec.functions
}

/// Harvest address-taken code pointers ([`crate::xref::XrefKind::AddressOf`])
/// from `functions` that begin a new, plausible function.
///
/// The gate keeps this from inventing functions: a target must lie in an
/// executable region (never in data), must not already be a recovered
/// function, must not fall in the *interior* of a recovered block (so a
/// pointer into the middle of a function cannot split it into a bogus one),
/// and must open with a canonical prologue
/// ([`crate::funcs::looks_like_function_start`]). Returns the surviving
/// targets in address order. Never panics.
fn addressof_seeds(image: &dyn Image, functions: &BTreeMap<u64, Function>) -> BTreeSet<u64> {
    // A snapshot program for the xref walk (it reads only block extents).
    let program = Program {
        functions: functions.clone(),
        call_graph: BTreeMap::new(),
        stats: Stats::default(),
    };
    let Ok(xrefs) = crate::xref::compute(image, &program) else {
        return BTreeSet::new();
    };

    let exec = exec_ranges(image);
    let in_exec = |va: u64| exec.iter().any(|&(s, e)| va >= s && va < e);

    // Interior coverage: (block start -> max end) over every recovered
    // block, so a target strictly inside a block can be rejected.
    let mut covered: BTreeMap<u64, u64> = BTreeMap::new();
    for f in functions.values() {
        for b in f.blocks.values() {
            covered
                .entry(b.start)
                .and_modify(|e| *e = (*e).max(b.end))
                .or_insert(b.end);
        }
    }
    let interior = |va: u64| {
        covered
            .range(..=va)
            .next_back()
            .is_some_and(|(&s, &e)| va > s && va < e)
    };

    let mut out = BTreeSet::new();
    for x in xrefs.iter() {
        if x.kind != crate::xref::XrefKind::AddressOf {
            continue;
        }
        let target = x.to;
        if functions.contains_key(&target) || !in_exec(target) || interior(target) {
            continue;
        }
        if crate::funcs::looks_like_function_start(image, target) {
            out.insert(target);
        }
    }
    out
}

/// Executable `[start, end)` ranges of `image` (unsorted).
fn exec_ranges(image: &dyn Image) -> Vec<(u64, u64)> {
    image
        .regions()
        .iter()
        .filter(|r| r.perms.x && r.size > 0)
        .map(|r| (r.va, r.va.saturating_add(r.size)))
        .collect()
}

/// The name [`recover`] assigns a function entry: the first
/// [`SymbolKind::Function`] symbol at `entry` (symbols are (VA, name)
/// sorted, so "first" is deterministic), else the first symbol of any
/// kind at `entry`, else `None`.
fn name_for(image: &dyn Image, entry: u64) -> Option<String> {
    let syms = image.symbols();
    syms.iter()
        .find(|s| s.va == entry && s.kind == SymbolKind::Function)
        .or_else(|| syms.iter().find(|s| s.va == entry))
        .map(|s| s.name.clone())
}

/// Per-function descent state and outputs for [`recover_function`]. The
/// block-building logic mirrors [`Recovery`] exactly, but every effect
/// that is whole-image in [`Recovery`] (the stats counters, the thunk
/// map, the call graph, the callee worklist) is captured *locally* here,
/// so a descent touches no shared state and is a pure function of its
/// inputs.
struct FuncWorker<'a> {
    image: &'a dyn Image,
    decoder: &'a dyn Decoder,
    bytes: &'a [u8],
    exec: &'a [(u64, u64)],
    slots: &'a BTreeMap<u64, String>,
    config: &'a Config,
    entry: u64,
    stats: Stats,
    call_edges: BTreeSet<CallTarget>,
    callees: BTreeSet<u64>,
    thunk: Option<String>,
    blocks: BTreeMap<u64, BasicBlock>,
    leaders: BTreeSet<u64>,
    pending: BTreeSet<u64>,
    insn_starts: BTreeSet<u64>,
}

impl FuncWorker<'_> {
    /// The executable region containing `va`, if any.
    fn exec_region(&self, va: u64) -> Option<(u64, u64)> {
        self.exec.iter().copied().find(|&(s, e)| va >= s && va < e)
    }

    /// Whether a recovery-stopping cap has been hit.
    fn capped(&self) -> bool {
        self.stats.instruction_cap_hit || self.stats.block_cap_hit
    }

    /// Decode one basic block starting at `start`, insert it, and queue
    /// its successors. Mirrors [`Recovery::build_block`].
    fn build_block(&mut self, start: u64) {
        let entry = self.entry;
        let mut cur = start;
        let (terminator, end, successors) = loop {
            if cur != start && self.leaders.contains(&cur) {
                break (Terminator::FallThrough(cur), cur, vec![cur]);
            }
            if self.stats.instructions >= self.config.max_instructions {
                self.stats.instruction_cap_hit = true;
                break (Terminator::Truncated, cur, Vec::new());
            }
            let Some((_, region_end)) = self.exec_region(cur) else {
                break (Terminator::Undecodable, cur, Vec::new());
            };
            let Some(off) = self.image.va_to_offset(cur) else {
                break (Terminator::Undecodable, cur, Vec::new());
            };
            let window = (region_end - cur)
                .min(MAX_INSN_BYTES)
                .min(self.bytes.len().saturating_sub(off) as u64)
                as usize;
            let Ok(d) = self.decoder.decode_flow(&self.bytes[off..off + window], cur) else {
                break (Terminator::Undecodable, cur, Vec::new());
            };
            self.stats.instructions += 1;
            self.insn_starts.insert(cur);
            let Some(next) = cur.checked_add(u64::from(d.length)) else {
                break (Terminator::Undecodable, cur, Vec::new());
            };
            match d.flow {
                Flow::Sequential => cur = next,
                Flow::Jump(t) => break (Terminator::Jump(t), next, vec![t]),
                Flow::CondJump(t) => {
                    let succ = if t == next { vec![t] } else { vec![t, next] };
                    break (
                        Terminator::CondJump {
                            taken: t,
                            fallthrough: next,
                        },
                        next,
                        succ,
                    );
                }
                Flow::IndirectJump => {
                    let import = d.mem_target.and_then(|s| self.slots.get(&s).cloned());
                    if let Some(name) = &import {
                        if cur == entry {
                            self.thunk = Some(name.clone());
                        }
                        self.call_edges.insert(CallTarget::Import(name.clone()));
                    }
                    break (Terminator::IndirectJump { import }, next, Vec::new());
                }
                Flow::Call(t) => {
                    self.call_edges.insert(CallTarget::Function(t));
                    if self.exec_region(t).is_some() {
                        self.callees.insert(t);
                    }
                    break (
                        Terminator::Call {
                            target: t,
                            fallthrough: next,
                        },
                        next,
                        vec![next],
                    );
                }
                Flow::IndirectCall => {
                    let import = d.mem_target.and_then(|s| self.slots.get(&s).cloned());
                    let edge = import
                        .clone()
                        .map_or(CallTarget::Unknown, CallTarget::Import);
                    self.call_edges.insert(edge);
                    break (
                        Terminator::IndirectCall {
                            import,
                            fallthrough: next,
                        },
                        next,
                        vec![next],
                    );
                }
                Flow::Return => break (Terminator::Return, next, Vec::new()),
                Flow::Interrupt => {
                    break (Terminator::Interrupt { fallthrough: next }, next, vec![next]);
                }
                Flow::Halt => break (Terminator::Halt, next, Vec::new()),
            }
        };

        self.blocks.insert(
            start,
            BasicBlock {
                start,
                end,
                terminator,
                successors: successors.clone(),
            },
        );
        self.stats.blocks += 1;

        for s in successors {
            if self.exec_region(s).is_some() {
                self.add_leader(s);
            }
        }
    }

    /// Register `target` as a block leader, splitting on an instruction
    /// boundary or queuing a fresh build. Mirrors [`Recovery::add_leader`].
    fn add_leader(&mut self, target: u64) {
        if self.leaders.contains(&target) {
            return;
        }
        let split_at = self
            .blocks
            .range(..=target)
            .next_back()
            .filter(|&(&bs, b)| {
                target > bs && target < b.end && self.insn_starts.contains(&target)
            })
            .map(|(&bs, _)| bs);
        if let Some(bs) = split_at {
            if self.stats.blocks >= self.config.max_blocks {
                self.stats.block_cap_hit = true;
                return;
            }
            let Some(head) = self.blocks.get_mut(&bs) else {
                return;
            };
            let tail = BasicBlock {
                start: target,
                end: head.end,
                terminator: head.terminator.clone(),
                successors: head.successors.clone(),
            };
            head.end = target;
            head.terminator = Terminator::FallThrough(target);
            head.successors = vec![target];
            self.blocks.insert(target, tail);
            self.leaders.insert(target);
            self.stats.blocks += 1;
            return;
        }
        if self.pending.len() >= self.config.max_worklist {
            self.stats.worklist_cap_hit = true;
            return;
        }
        self.leaders.insert(target);
        self.pending.insert(target);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::tests::synthetic_elf64;
    use crate::model::load;
    use crate::pe::tests::{synthetic_pe64, with_imports};

    /// Image base of the synthetic PE fixtures.
    const PE_BASE: u64 = 0x1_4000_0000;
    /// Entry VA of the synthetic PE fixtures (RVA 0x1000, file 0x200).
    const PE_ENTRY: u64 = PE_BASE + 0x1000;
    /// VA of the synthetic ELF .text (== its entry point, file 0x100).
    const ELF_TEXT: u64 = 0x40_1000;

    fn recover_bytes(img: &[u8]) -> Program {
        let image = load(img).unwrap();
        recover(image.as_ref()).unwrap()
    }

    fn block(start: u64, end: u64, terminator: Terminator, successors: &[u64]) -> BasicBlock {
        BasicBlock {
            start,
            end,
            terminator,
            successors: successors.to_vec(),
        }
    }

    #[test]
    fn x86_conditional_diamond_has_exact_blocks_and_edges() {
        let mut img = synthetic_pe64();
        let code: &[u8] = &[
            0x31, 0xC0, // +0:  xor eax, eax
            0x74, 0x05, // +2:  je +9
            0xFF, 0xC0, // +4:  inc eax
            0xEB, 0x03, // +6:  jmp +11
            0x90, // +8:  (unreachable)
            0xFF, 0xC8, // +9:  dec eax
            0xC3, // +11: ret
        ];
        img[0x200..0x200 + code.len()].copy_from_slice(code);
        let p = recover_bytes(&img);

        assert_eq!(p.functions.keys().copied().collect::<Vec<_>>(), [PE_ENTRY]);
        let e = PE_ENTRY;
        let f = &p.functions[&e];
        let expected = BTreeMap::from([
            (
                e,
                block(
                    e,
                    e + 4,
                    Terminator::CondJump {
                        taken: e + 9,
                        fallthrough: e + 4,
                    },
                    &[e + 9, e + 4],
                ),
            ),
            (e + 4, block(e + 4, e + 8, Terminator::Jump(e + 11), &[e + 11])),
            (
                e + 9,
                block(e + 9, e + 11, Terminator::FallThrough(e + 11), &[e + 11]),
            ),
            (e + 11, block(e + 11, e + 12, Terminator::Return, &[])),
        ]);
        assert_eq!(f.blocks, expected);
        // The unreachable byte at +8 belongs to no block.
        assert_eq!(p.stats.instructions, 6);
        assert_eq!(p.stats.blocks, 4);
        assert!(!p.stats.instruction_cap_hit && !p.stats.block_cap_hit);
        assert!(p.call_graph.is_empty());
    }

    #[test]
    fn later_jump_target_splits_existing_block() {
        let mut img = synthetic_pe64();
        let code: &[u8] = &[
            0x31, 0xC0, // +0:  xor eax, eax
            0xFF, 0xC0, // +2:  inc eax
            0xEB, 0x04, // +4:  jmp +10
            0x90, 0x90, 0x90, 0x90, // +6..+10: (unreachable)
            0xEB, 0xF6, // +10: jmp back into the middle of [+0, +6): +2
        ];
        img[0x200..0x200 + code.len()].copy_from_slice(code);
        let p = recover_bytes(&img);

        // The entry block [+0, +6) is built first as one straight run;
        // the jump at +10 then discovers +2 and must split it on the
        // instruction boundary.
        let e = PE_ENTRY;
        let expected = BTreeMap::from([
            (e, block(e, e + 2, Terminator::FallThrough(e + 2), &[e + 2])),
            (e + 2, block(e + 2, e + 6, Terminator::Jump(e + 10), &[e + 10])),
            (
                e + 10,
                block(e + 10, e + 12, Terminator::Jump(e + 2), &[e + 2]),
            ),
        ]);
        assert_eq!(p.functions[&e].blocks, expected);
        assert_eq!(p.stats.blocks, 3);
    }

    /// Entry VA used by [`import_call_fixture`] (clear of the import
    /// data that `with_imports` lays at RVA 0x1000..0x109E).
    const IMP_ENTRY: u64 = PE_BASE + 0x10A0;
    const IMP_THUNK: u64 = PE_BASE + 0x10B0;

    /// `with_imports` plus real code: the entry calls the ExitProcess
    /// IAT slot directly, then calls a `jmp [IAT]` thunk for ordinal 7.
    fn import_call_fixture() -> Vec<u8> {
        let mut img = with_imports();
        // Repoint the entry RVA from 0x1000 (import data) to 0x10A0.
        let opt = 0x80 + 4 + 20;
        img[opt + 16..opt + 20].copy_from_slice(&0x10A0u32.to_le_bytes());
        let code: &[u8] = &[
            // 0x10A0: call [rip - 0x46]  -> IAT slot RVA 0x1060
            0xFF, 0x15, 0xBA, 0xFF, 0xFF, 0xFF,
            // 0x10A6: call 0x10B0 (the thunk)
            0xE8, 0x05, 0x00, 0x00, 0x00,
            // 0x10AB: ret
            0xC3, //
            0x90, 0x90, 0x90, 0x90, // padding to 0x10B0
            // 0x10B0: jmp [rip - 0x4E]  -> IAT slot RVA 0x1068
            0xFF, 0x25, 0xB2, 0xFF, 0xFF, 0xFF,
        ];
        img[0x2A0..0x2A0 + code.len()].copy_from_slice(code); // RVA 0x10A0
        img
    }

    #[test]
    fn iat_indirected_calls_resolve_to_imports() {
        let p = recover_bytes(&import_call_fixture());

        assert_eq!(
            p.functions.keys().copied().collect::<Vec<_>>(),
            [IMP_ENTRY, IMP_THUNK]
        );

        // Entry: the IAT call and the thunk call each end a block.
        let f = &p.functions[&IMP_ENTRY];
        let e = IMP_ENTRY;
        let expected = BTreeMap::from([
            (
                e,
                block(
                    e,
                    e + 6,
                    Terminator::IndirectCall {
                        import: Some("KERNEL32.dll!ExitProcess".into()),
                        fallthrough: e + 6,
                    },
                    &[e + 6],
                ),
            ),
            (
                e + 6,
                block(
                    e + 6,
                    e + 11,
                    Terminator::Call {
                        target: IMP_THUNK,
                        fallthrough: e + 11,
                    },
                    &[e + 11],
                ),
            ),
            (e + 11, block(e + 11, e + 12, Terminator::Return, &[])),
        ]);
        assert_eq!(f.blocks, expected);

        // The thunk is a single jmp-through-IAT block.
        assert_eq!(
            p.functions[&IMP_THUNK].blocks,
            BTreeMap::from([(
                IMP_THUNK,
                block(
                    IMP_THUNK,
                    IMP_THUNK + 6,
                    Terminator::IndirectJump {
                        import: Some("KERNEL32.dll!#7".into())
                    },
                    &[],
                ),
            )])
        );

        // Call graph: the IAT call resolves directly, and the direct
        // call to the thunk resolves through it; the thunk's tail call
        // is its own import edge.
        assert_eq!(
            p.call_graph,
            BTreeMap::from([
                (
                    IMP_ENTRY,
                    BTreeSet::from([
                        CallTarget::Import("KERNEL32.dll!ExitProcess".into()),
                        CallTarget::Import("KERNEL32.dll!#7".into()),
                    ]),
                ),
                (
                    IMP_THUNK,
                    BTreeSet::from([CallTarget::Import("KERNEL32.dll!#7".into())]),
                ),
            ])
        );
    }

    /// The x86-64 ELF fixture rebuilt as AArch64: same layout, A64 code.
    fn aarch64_elf_fixture() -> Vec<u8> {
        let mut img = synthetic_elf64();
        img[18..20].copy_from_slice(&183u16.to_le_bytes()); // EM_AARCH64
        // .text at file 0x100 == VA 0x40_1000; symbols: main at +0,
        // helper at +0x20 (both SymbolKind::Function).
        let words: &[(usize, u32)] = &[
            (0x00, 0x5400_0040), // b.eq +8
            (0x04, 0x9400_000F), // bl +0x3c -> 0x40_1040 (no symbol)
            (0x08, 0xD65F_03C0), // ret
            (0x20, 0xD65F_03C0), // helper: ret
            (0x40, 0xD503_201F), // callee: nop
            (0x44, 0xD65F_03C0), //         ret
        ];
        for &(off, w) in words {
            img[0x100 + off..0x100 + off + 4].copy_from_slice(&w.to_le_bytes());
        }
        img
    }

    #[test]
    fn aarch64_bl_discovers_callee_and_bcond_diamonds() {
        let p = recover_bytes(&aarch64_elf_fixture());
        let t = ELF_TEXT;

        // main and helper are seeded from symbols/entry; the unnamed
        // callee at +0x40 is discovered through the BL.
        assert_eq!(
            p.functions.keys().copied().collect::<Vec<_>>(),
            [t, t + 0x20, t + 0x40]
        );

        let main = &p.functions[&t];
        assert_eq!(main.name.as_deref(), Some("main"));
        let expected = BTreeMap::from([
            (
                t,
                block(
                    t,
                    t + 4,
                    Terminator::CondJump {
                        taken: t + 8,
                        fallthrough: t + 4,
                    },
                    &[t + 8, t + 4],
                ),
            ),
            (
                t + 4,
                block(
                    t + 4,
                    t + 8,
                    Terminator::Call {
                        target: t + 0x40,
                        fallthrough: t + 8,
                    },
                    &[t + 8],
                ),
            ),
            (t + 8, block(t + 8, t + 12, Terminator::Return, &[])),
        ]);
        assert_eq!(main.blocks, expected);

        let helper = &p.functions[&(t + 0x20)];
        assert_eq!(helper.name.as_deref(), Some("helper"));
        assert_eq!(
            helper.blocks,
            BTreeMap::from([(t + 0x20, block(t + 0x20, t + 0x24, Terminator::Return, &[]))])
        );

        let callee = &p.functions[&(t + 0x40)];
        assert_eq!(callee.name, None);
        assert_eq!(
            callee.blocks,
            BTreeMap::from([(t + 0x40, block(t + 0x40, t + 0x48, Terminator::Return, &[]))])
        );

        assert_eq!(
            p.call_graph,
            BTreeMap::from([(t, BTreeSet::from([CallTarget::Function(t + 0x40)]))])
        );
    }

    /// PE fixture whose whole .text is single-byte NOPs (the import
    /// directory the base fixture points into .text is dropped, since
    /// the NOPs overwrite it).
    fn nop_sled_fixture() -> Vec<u8> {
        let mut img = synthetic_pe64();
        let dirs = 0x80 + 4 + 20 + 112;
        img[dirs + 8..dirs + 16].fill(0);
        img[0x200..0x300].fill(0x90); // RVA 0x1000..0x1100: 256 nops
        img
    }

    #[test]
    fn instruction_cap_truncates_cleanly() {
        let img = nop_sled_fixture();
        let image = load(&img).unwrap();
        let config = Config {
            max_instructions: 10,
            ..Config::default()
        };
        let p = recover_with(image.as_ref(), &config).unwrap();

        assert!(p.stats.instruction_cap_hit);
        assert_eq!(p.stats.instructions, 10);
        assert_eq!(
            p.functions[&PE_ENTRY].blocks,
            BTreeMap::from([(
                PE_ENTRY,
                block(PE_ENTRY, PE_ENTRY + 10, Terminator::Truncated, &[]),
            )])
        );

        // Without the cap the same sled decodes to the region's end and
        // stops honestly (no successors, nothing decoded past .text).
        let p = recover(image.as_ref()).unwrap();
        assert!(!p.stats.instruction_cap_hit);
        assert_eq!(p.stats.instructions, 256);
        assert_eq!(
            p.functions[&PE_ENTRY].blocks[&PE_ENTRY],
            block(PE_ENTRY, PE_ENTRY + 0x100, Terminator::Undecodable, &[]),
        );
    }

    #[test]
    fn recovery_is_deterministic() {
        for img in [
            import_call_fixture(),
            aarch64_elf_fixture(),
            nop_sled_fixture(),
        ] {
            assert_eq!(recover_bytes(&img), recover_bytes(&img));
        }
    }

    #[test]
    fn garbage_entry_bytes_never_panic() {
        // One decodable nop, then an unmodeled x87 opcode: the entry
        // function exists and its block is Undecodable-terminated.
        let mut img = synthetic_pe64();
        img[0x200..0x204].copy_from_slice(&[0x90, 0xD8, 0x00, 0x06]);
        let p = recover_bytes(&img);
        assert_eq!(
            p.functions[&PE_ENTRY].blocks,
            BTreeMap::from([(
                PE_ENTRY,
                block(PE_ENTRY, PE_ENTRY + 1, Terminator::Undecodable, &[]),
            )])
        );
        assert!(p.call_graph.is_empty());

        // Fuzz-ish sweep: single-byte corruptions of the entry window
        // must never panic (results only need to exist).
        for b in 0..=0xFFu8 {
            let mut img = synthetic_pe64();
            img[0x200] = b;
            let image = load(&img).unwrap();
            let _ = recover(image.as_ref()).unwrap();
        }
    }

    #[test]
    fn unsupported_arch_is_a_typed_error() {
        let mut img = synthetic_elf64();
        img[18..20].copy_from_slice(&0xF00u16.to_le_bytes()); // unknown e_machine
        let image = load(&img).unwrap();
        assert!(matches!(
            recover(image.as_ref()),
            Err(ParseError::Unsupported(_))
        ));
    }

    #[test]
    fn image_with_no_executable_seeds_recovers_nothing() {
        let img = crate::elf::tests::synthetic_dynamic_elf64();
        let p = recover_bytes(&img);
        assert!(p.functions.is_empty());
        assert!(p.call_graph.is_empty());
        assert_eq!(p.stats, Stats::default());
    }

    // --- Second-order seed sources: unwind hints and address-taken code
    // pointers -------------------------------------------------------------

    use crate::model::{Arch, Perms, Region, Symbol};

    /// A fully controllable [`Image`] with explicit regions, so a test can
    /// pin executable vs non-executable memory and the metadata sources
    /// (`function_starts_hint`, symbols, entry points) independently. Each
    /// region maps linearly from its `va` to `file_off` in `bytes`.
    struct Mock {
        arch: Arch,
        bytes: Vec<u8>,
        /// `(va, size, executable, file_off)` per region.
        regions: Vec<(u64, u64, bool, usize)>,
        entries: Vec<u64>,
        syms: Vec<Symbol>,
        hints: Vec<u64>,
    }

    impl Image for Mock {
        fn arch(&self) -> Arch {
            self.arch
        }
        fn entry_points(&self) -> Vec<u64> {
            self.entries.clone()
        }
        fn regions(&self) -> Vec<Region> {
            self.regions
                .iter()
                .enumerate()
                .map(|(i, &(va, size, x, _))| Region {
                    name: format!("r{i}"),
                    va,
                    size,
                    perms: Perms { r: true, w: !x, x },
                })
                .collect()
        }
        fn symbols(&self) -> Vec<Symbol> {
            self.syms.clone()
        }
        fn import_slots(&self) -> Vec<crate::model::ImportSlot> {
            Vec::new()
        }
        fn va_to_offset(&self, va: u64) -> Option<usize> {
            for &(rva, size, _, off) in &self.regions {
                if va >= rva && va - rva < size {
                    let o = off + (va - rva) as usize;
                    return (o < self.bytes.len()).then_some(o);
                }
            }
            None
        }
        fn bytes(&self) -> &[u8] {
            &self.bytes
        }
        fn function_starts_hint(&self) -> Vec<u64> {
            self.hints.clone()
        }
    }

    /// Encode a run of A64 words into a byte buffer at word offsets.
    fn a64(words: &[(usize, u32)], len: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; len];
        for &(off, w) in words {
            bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }
        bytes
    }

    /// An uncalled function whose address is only *taken* — the shape of a
    /// Rust `main` handed to `lang_start` — is recovered from its
    /// `AddressOf` reference alone. The entry materializes 0x1040 with
    /// `ADRP`+`ADD` and returns without ever calling it; 0x1040 opens with
    /// a canonical prologue, so the address-taken pass seeds it.
    #[test]
    fn address_taken_uncalled_function_is_recovered() {
        let text = a64(
            &[
                (0x00, 0x9000_0000), // adrp x0, page(0x1000) -> 0x1000
                (0x04, 0x9101_0000), // add  x0, x0, #0x40    -> 0x1040
                (0x08, 0xD65F_03C0), // ret
                (0x40, 0xA9BF_7BFD), // stp x29,x30,[sp,#-0x10]!  (prologue)
                (0x44, 0xD65F_03C0), // ret
            ],
            0x100,
        );
        let m = Mock {
            arch: Arch::Aarch64,
            bytes: text,
            regions: vec![(0x1000, 0x100, true, 0)],
            entries: vec![0x1000],
            syms: vec![],
            hints: vec![],
        };
        let p = recover(&m).unwrap();
        assert_eq!(
            p.functions.keys().copied().collect::<Vec<_>>(),
            [0x1000, 0x1040],
            "the address-taken function at 0x1040 must be recovered"
        );
        // It is genuinely uncalled: nothing in the call graph points at it.
        assert!(
            p.call_graph
                .values()
                .flatten()
                .all(|t| *t != CallTarget::Function(0x1040))
        );
    }

    /// The gate never invents a function for a pointer into non-executable
    /// memory: the same `ADRP`+`ADD` now computes 0x2000, which lands in a
    /// read/write data region carrying prologue-shaped bytes. The exec test
    /// rejects it, so no function is created there.
    #[test]
    fn address_of_into_non_executable_memory_creates_no_function() {
        // .text at 0x1000 (exec, file 0), .data at 0x2000 (rw, file 0x100).
        let mut bytes = a64(
            &[
                (0x00, 0xB000_0000), // adrp x0, +1 page -> 0x2000
                (0x04, 0x9100_0000), // add  x0, x0, #0   -> 0x2000
                (0x08, 0xD65F_03C0), // ret
            ],
            0x100,
        );
        // Prologue-shaped bytes at the data target, to prove it is the
        // executable test — not the prologue test — that rejects it.
        bytes.extend_from_slice(&0xA9BF_7BFDu32.to_le_bytes());
        bytes.resize(0x200, 0);
        let m = Mock {
            arch: Arch::Aarch64,
            bytes,
            regions: vec![(0x1000, 0x100, true, 0), (0x2000, 0x100, false, 0x100)],
            entries: vec![0x1000],
            syms: vec![],
            hints: vec![],
        };
        let p = recover(&m).unwrap();
        assert_eq!(
            p.functions.keys().copied().collect::<Vec<_>>(),
            [0x1000],
            "a data pointer into non-executable memory must not become a function"
        );
    }

    /// An address-taken pointer into the *interior* of an existing function
    /// never splits it into a bogus new one — even when the target is a
    /// real prologue reached mid-block. The entry runs straight through
    /// 0x1040 (a `stp` frame push that decodes as ordinary sequential code
    /// here) to its `ret`, so 0x1040 is interior; the interior gate, not
    /// the prologue gate, is what rejects it.
    #[test]
    fn address_of_into_a_function_interior_creates_no_function() {
        let mut words = vec![
            (0x00usize, 0x9000_0000u32), // adrp x0, page(0x1000) -> 0x1000
            (0x04, 0x9101_0000),         // add  x0, x0, #0x40    -> 0x1040
        ];
        // Sequential nops from 0x08 through 0x3C so the block runs into
        // 0x40 without a leader there.
        let mut off = 0x08;
        while off < 0x40 {
            words.push((off, 0xD503_201F)); // nop
            off += 4;
        }
        words.push((0x40, 0xA9BF_7BFD)); // stp x29,x30,[sp,#-0x10]! (prologue)
        words.push((0x44, 0xD65F_03C0)); // ret
        let text = a64(&words, 0x100);
        let m = Mock {
            arch: Arch::Aarch64,
            bytes: text,
            regions: vec![(0x1000, 0x100, true, 0)],
            entries: vec![0x1000],
            syms: vec![],
            hints: vec![],
        };
        let p = recover(&m).unwrap();
        assert_eq!(
            p.functions.keys().copied().collect::<Vec<_>>(),
            [0x1000],
            "a pointer into a function's interior must not create a function"
        );
    }

    /// Unwind/metadata seeding: a function that is never called and never
    /// address-taken is still recovered when the image's
    /// `function_starts_hint` (Mach-O `LC_FUNCTION_STARTS`, PE `.pdata`,
    /// ELF `.eh_frame`) names it. Only executable hints are honored.
    #[test]
    fn unwind_hint_seeds_an_uncalled_function() {
        let text = a64(
            &[
                (0x00, 0xD65F_03C0), // entry: ret
                (0x40, 0xA9BF_7BFD), // hinted, uncalled function: prologue
                (0x44, 0xD65F_03C0), // ret
            ],
            0x100,
        );
        let m = Mock {
            arch: Arch::Aarch64,
            bytes: text,
            regions: vec![(0x1000, 0x100, true, 0)],
            entries: vec![0x1000],
            syms: vec![],
            // 0x1040 is executable and honored; 0x9999 is not mapped and
            // dropped by `seed_function`'s region check.
            hints: vec![0x1040, 0x9999],
        };
        let p = recover(&m).unwrap();
        assert_eq!(
            p.functions.keys().copied().collect::<Vec<_>>(),
            [0x1000, 0x1040],
        );
    }

    // --- Proven jump-table folding ([`recover_with_tables`]) ---------------

    /// PE fixture with an unresolved indirect jump at +2 and two case
    /// bodies at +0x10 / +0x20 that only a folded edge can reach.
    fn dispatch_fixture() -> Vec<u8> {
        let mut img = synthetic_pe64();
        let code: &[u8] = &[
            0x31, 0xC0, // +0: xor eax, eax
            0xFF, 0xE0, // +2: jmp rax
        ];
        img[0x200..0x200 + code.len()].copy_from_slice(code);
        img[0x210] = 0xC3; // +0x10: ret (case 0)
        img[0x220] = 0xC3; // +0x20: ret (case 1)
        img
    }

    fn recover_tables(img: &[u8], map: &BTreeMap<u64, Vec<u64>>) -> Program {
        let image = load(img).unwrap();
        recover_with_tables(image.as_ref(), &Config::default(), map).unwrap()
    }

    /// The core fold: the dispatch block's successors become exactly the
    /// proven targets and every target is walked into a block — code
    /// plain [`recover`] proves it cannot reach.
    #[test]
    fn proven_table_successors_fold_and_their_code_is_walked() {
        let img = dispatch_fixture();
        let e = PE_ENTRY;

        // Without a table the jump is an honest dead end.
        let plain = recover_bytes(&img);
        assert_eq!(
            plain.functions[&e].blocks,
            BTreeMap::from([(
                e,
                block(e, e + 4, Terminator::IndirectJump { import: None }, &[]),
            )])
        );
        assert_eq!(plain.stats.tables_folded, 0);

        // With one (supplied unsorted, with a duplicate: the fold owns
        // the successor order, ascending and deduplicated).
        let map = BTreeMap::from([(e + 2, vec![e + 0x20, e + 0x10, e + 0x20])]);
        let p = recover_tables(&img, &map);
        let expected = BTreeMap::from([
            (
                e,
                block(
                    e,
                    e + 4,
                    Terminator::IndirectJump { import: None },
                    &[e + 0x10, e + 0x20],
                ),
            ),
            (e + 0x10, block(e + 0x10, e + 0x11, Terminator::Return, &[])),
            (e + 0x20, block(e + 0x20, e + 0x21, Terminator::Return, &[])),
        ]);
        assert_eq!(p.functions[&e].blocks, expected);
        assert_eq!(p.stats.tables_folded, 1);
        assert_eq!(p.stats.table_targets_dropped, 0);
    }

    /// A supplied target outside every executable region is dropped —
    /// no block could back the edge — and the drop is counted, never
    /// silent. The in-region target still folds.
    #[test]
    fn out_of_region_table_targets_are_dropped_visibly() {
        let img = dispatch_fixture();
        let e = PE_ENTRY;
        let map = BTreeMap::from([(e + 2, vec![e + 0x10, 0x9999_9999])]);
        let p = recover_tables(&img, &map);
        assert_eq!(p.functions[&e].blocks[&e].successors, [e + 0x10]);
        assert_eq!(p.stats.tables_folded, 1);
        assert_eq!(p.stats.table_targets_dropped, 1);
    }

    /// A folded target landing mid-block on an instruction boundary
    /// splits the block, exactly as a late direct-branch target would.
    #[test]
    fn a_folded_target_landing_mid_block_splits_it() {
        let mut img = synthetic_pe64();
        let code: &[u8] = &[
            0x31, 0xC0, // +0: xor eax, eax
            0xFF, 0xC0, // +2: inc eax
            0xFF, 0xE0, // +4: jmp rax
        ];
        img[0x200..0x200 + code.len()].copy_from_slice(code);
        let e = PE_ENTRY;
        let map = BTreeMap::from([(e + 4, vec![e + 2])]);
        let p = recover_tables(&img, &map);
        let expected = BTreeMap::from([
            (e, block(e, e + 2, Terminator::FallThrough(e + 2), &[e + 2])),
            (
                e + 2,
                block(e + 2, e + 6, Terminator::IndirectJump { import: None }, &[e + 2]),
            ),
        ]);
        assert_eq!(p.functions[&e].blocks, expected);
    }

    /// An import tail call leaves the function; a supplied table entry
    /// for its site is not believed and nothing folds.
    #[test]
    fn import_tail_call_sites_never_fold() {
        let img = import_call_fixture();
        let map = BTreeMap::from([(IMP_THUNK, vec![IMP_ENTRY])]);
        let p = recover_tables(&img, &map);
        assert!(p.functions[&IMP_THUNK].blocks[&IMP_THUNK].successors.is_empty());
        assert_eq!(p.stats.tables_folded, 0);
        // The fixture without the map is bit-for-bit the same program.
        assert_eq!(p, recover_bytes(&img));
    }

    /// Folding is deterministic, and an empty map is plain [`recover`].
    #[test]
    fn folding_is_deterministic_and_an_empty_map_is_plain_recovery() {
        let img = dispatch_fixture();
        let e = PE_ENTRY;
        let map = BTreeMap::from([(e + 2, vec![e + 0x10, e + 0x20])]);
        assert_eq!(recover_tables(&img, &map), recover_tables(&img, &map));
        assert_eq!(recover_tables(&img, &BTreeMap::new()), recover_bytes(&img));
    }
}
