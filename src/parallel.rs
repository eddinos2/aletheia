//! Parallel analysis pass engine: deterministic function-parallel
//! control-flow recovery.
//!
//! [`analyze`] recovers the same [`cfg::Program`] as [`cfg::recover`], but
//! spreads the per-function work across a hand-rolled worker pool (no
//! third-party runtime — only [`std::thread`] and [`std::sync`]). Each
//! function is an independent unit: [`cfg::recover_function`] is a pure
//! function of `(image, entry, config)` with no cross-function shared
//! mutable state, so the analysis of one function can never influence
//! another's result.
//!
//! # Determinism — the headline property
//!
//! The recovered program is **byte-for-byte identical regardless of the
//! thread count** (1 vs N) and regardless of the order workers happen to
//! finish. Three properties combine to guarantee it:
//!
//! 1. **Pure work units.** [`cfg::recover_function`] shares no mutable
//!    state between functions, so a function's blocks, call edges, thunk
//!    status, and decode counts depend only on `(image, entry, config)` —
//!    never on which thread ran it or what ran alongside it.
//! 2. **Barriered rounds.** Recovery proceeds in rounds. A round analyzes
//!    the whole current worklist in parallel, joins every worker (a
//!    barrier), *then* the driver merges results and derives the next
//!    round's worklist. No result from round *n+1* exists before round
//!    *n* is fully merged, so the set of functions in each round is fixed
//!    independently of scheduling.
//! 3. **Ordered single-threaded merge.** Results are merged in ascending
//!    entry-address order into [`BTreeMap`]s. Because the merge is
//!    address-ordered and the reductions are commutative (set union,
//!    integer sums, boolean OR), the assembled [`cfg::Program`] does not
//!    depend on intra-round completion order.
//!
//! The one value that legitimately varies with the requested thread count
//! is [`cfg::Stats::threads_used`], which records the pool size; every
//! other field of the returned program is thread-count-independent.
//!
//! # Sharing the image across threads
//!
//! [`crate::model::Image`] is not `Sync`, so a `&dyn Image` cannot be
//! shared across threads directly. Instead each worker is handed the
//! immutable image bytes (`&[u8]`, which *is* `Send + Sync`) and calls
//! [`crate::model::load`] to reconstruct its own read-only [`Image`] view.
//! Re-loading is a pure function of the bytes, so every worker sees an
//! identical image — determinism is preserved and no `unsafe` or `Sync`
//! shim is needed. The cost is one header re-parse per worker per round
//! (not per function); decoding dominates for any non-trivial binary.
//!
//! # Relationship to [`cfg::recover`]
//!
//! [`analyze`] discovers the same closure of functions (identical seeds
//! via [`cfg::function_seeds`], identical transitive call-target
//! discovery) and builds each function with the same block logic, so on
//! any image that stays within its caps the recovered `functions` and
//! `call_graph` are **equal** to [`cfg::recover`]'s, as are
//! `stats.instructions` and `stats.blocks`. Two intentional, documented
//! differences:
//!
//! - [`Config::cfg`]'s instruction/block/worklist caps apply **per
//!   function** here (each function gets the full budget), whereas
//!   [`cfg::recover`] applies them whole-image. On within-budget images
//!   this is unobservable.
//! - [`analyze`] additionally honors [`Config::max_functions`], a cap on
//!   the total number of recovered functions (setting
//!   [`cfg::Stats::function_cap_hit`] when exceeded), and reports
//!   [`cfg::Stats::rounds`] and [`cfg::Stats::threads_used`].

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::cfg::{self, CallTarget, Function, FunctionCfg, Stats};
use crate::error::Result;
use crate::model::{Image, decoder_for, load};

/// Configuration for [`analyze`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Worker-pool size. `0` selects
    /// [`std::thread::available_parallelism`] (falling back to `1`). Any
    /// value yields the same recovered program; only
    /// [`cfg::Stats::threads_used`] reflects it.
    pub threads: usize,
    /// Per-function control-flow recovery caps (see [`cfg::Config`]).
    /// Applied per function; see the module docs.
    pub cfg: cfg::Config,
    /// Maximum number of functions to recover. Discovery stops cleanly
    /// once this many functions are merged, in ascending entry-address
    /// order, and [`cfg::Stats::function_cap_hit`] is set.
    pub max_functions: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            threads: 0,
            cfg: cfg::Config::default(),
            max_functions: 1_048_576,
        }
    }
}

/// Recover a [`cfg::Program`] from `image` in parallel with the default
/// [`Config`] (auto-selected thread count, default caps).
///
/// Deterministic: the returned program is identical to
/// `analyze_with(image, &Config::default())` and, save for
/// [`cfg::Stats::threads_used`], identical across any thread count. See
/// the module docs. Fails with [`crate::error::ParseError::Unsupported`]
/// when the image's architecture has no decoder, exactly as
/// [`cfg::recover`] does.
pub fn analyze(image: &dyn Image) -> Result<cfg::Program> {
    analyze_with(image, &Config::default())
}

/// [`analyze`] with a caller-supplied [`Config`].
pub fn analyze_with(image: &dyn Image, config: &Config) -> Result<cfg::Program> {
    // Reject an undecodable architecture up front, with the same typed
    // error as `cfg::recover`, before spawning any thread. Every worker
    // re-loads these same bytes, so this decision holds for all of them.
    let arch = image.arch();
    if decoder_for(arch).is_none() {
        return Err(crate::error::ParseError::Unsupported(format!(
            "control-flow recovery: no decoder for architecture {arch:?}"
        )));
    }

    let bytes = image.bytes();
    let threads = resolve_threads(config.threads);

    let mut functions: BTreeMap<u64, Function> = BTreeMap::new();
    // Pre-thunk-resolution call edges, keyed by caller entry.
    let mut call_edges: BTreeMap<u64, BTreeSet<CallTarget>> = BTreeMap::new();
    // Entry -> imported name for every recovered import thunk.
    let mut thunks: BTreeMap<u64, String> = BTreeMap::new();
    let mut stats = Stats::default();

    let mut work: Vec<u64> = cfg::function_seeds(image);
    let mut rounds = 0usize;

    'outer: loop {
        if work.is_empty() {
            break;
        }
        let remaining = config.max_functions.saturating_sub(functions.len());
        if remaining == 0 {
            stats.function_cap_hit = true;
            break;
        }
        // Cap this round to the remaining function budget, keeping the
        // lowest entry addresses (the worklist is address-sorted). This is
        // the last round when it fires.
        let capped_round = work.len() > remaining;
        if capped_round {
            stats.function_cap_hit = true;
            work.truncate(remaining);
        }

        rounds += 1;
        let results = run_round(bytes, &work, threads, &config.cfg);

        // Merge in ascending entry order (results are sorted by entry):
        // address-ordered, single-threaded, order-independent reductions.
        let mut next: BTreeSet<u64> = BTreeSet::new();
        for (entry, outcome) in results {
            match outcome {
                Ok(fc) => {
                    stats.instructions += fc.stats.instructions;
                    stats.blocks += fc.stats.blocks;
                    stats.instruction_cap_hit |= fc.stats.instruction_cap_hit;
                    stats.block_cap_hit |= fc.stats.block_cap_hit;
                    stats.worklist_cap_hit |= fc.stats.worklist_cap_hit;
                    if !fc.call_edges.is_empty() {
                        call_edges.insert(entry, fc.call_edges);
                    }
                    if let Some(name) = fc.thunk {
                        thunks.insert(entry, name);
                    }
                    for callee in fc.callees {
                        if !functions.contains_key(&callee) {
                            next.insert(callee);
                        }
                    }
                    functions.insert(entry, fc.function);
                }
                // A worker panicked on this entry (never expected for a
                // valid image): record an empty function so the failure is
                // localized to this entry and the rest of the run stands.
                Err(()) => {
                    functions.entry(entry).or_insert(Function {
                        entry,
                        name: None,
                        blocks: BTreeMap::new(),
                    });
                }
            }
        }

        if capped_round {
            break 'outer;
        }
        work = next.into_iter().collect();
    }

    // Resolve direct calls to import thunks, exactly as `cfg::recover`
    // does after its worklist drains.
    let call_graph = call_edges
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

    stats.rounds = rounds;
    stats.threads_used = threads;

    Ok(cfg::Program {
        functions,
        call_graph,
        stats,
    })
}

/// Analyze one round's `work` entries across a fixed pool of `threads`
/// scoped workers, returning `(entry, outcome)` pairs sorted by entry.
///
/// The workers pull entries from a shared atomic cursor (the hand-rolled
/// work queue) and push results under a mutex; the driver joins them all
/// (the round barrier) before returning. The returned order is
/// deterministic (sorted by entry) regardless of which worker handled
/// which entry.
fn run_round(
    bytes: &[u8],
    work: &[u64],
    threads: usize,
    cfg_config: &cfg::Config,
) -> Vec<(u64, std::result::Result<FunctionCfg, ()>)> {
    let cursor = AtomicUsize::new(0);
    let results: Mutex<Vec<(u64, std::result::Result<FunctionCfg, ()>)>> =
        Mutex::new(Vec::with_capacity(work.len()));
    // No point spawning more workers than there is work.
    let pool = threads.max(1).min(work.len().max(1));

    std::thread::scope(|scope| {
        for _ in 0..pool {
            scope.spawn(|| {
                // Each worker reconstructs its own read-only image view
                // from the shared immutable bytes: sidesteps `Image: Sync`
                // and stays a pure function of the bytes.
                let Ok(img) = load(bytes) else {
                    return; // unreachable: the driver already loaded these
                };
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(&entry) = work.get(i) else {
                        break;
                    };
                    // A per-function panic must never poison the whole run.
                    let outcome = catch_unwind(AssertUnwindSafe(|| {
                        cfg::recover_function(img.as_ref(), entry, cfg_config)
                    }));
                    let recorded = match outcome {
                        Ok(Ok(fc)) => Ok(fc),
                        // A typed recovery error or a panic both localize
                        // to this entry.
                        Ok(Err(_)) | Err(_) => Err(()),
                    };
                    push(&results, (entry, recorded));
                }
            });
        }
    });

    let mut out = results.into_inner().unwrap_or_else(|e| e.into_inner());
    out.sort_by_key(|(entry, _)| *entry);
    out
}

/// Push one result under the mutex, recovering a poisoned lock (a lock is
/// only ever held for the push itself, which cannot panic, so poisoning is
/// not expected — but never unwrap-panic on it either).
fn push<T>(results: &Mutex<Vec<T>>, item: T) {
    match results.lock() {
        Ok(mut v) => v.push(item),
        Err(poisoned) => poisoned.into_inner().push(item),
    }
}

/// Resolve a requested thread count to a concrete pool size: `0` means
/// [`std::thread::available_parallelism`], falling back to `1`.
fn resolve_threads(requested: usize) -> usize {
    if requested != 0 {
        return requested;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{BasicBlock, Terminator};
    use crate::elf::tests::synthetic_elf64;
    use crate::model::load;
    use crate::pe::tests::{synthetic_pe64, with_imports};

    const PE_BASE: u64 = 0x1_4000_0000;
    const PE_ENTRY: u64 = PE_BASE + 0x1000;

    fn analyze_bytes_with(img: &[u8], config: &Config) -> cfg::Program {
        let image = load(img).unwrap();
        analyze_with(image.as_ref(), config).unwrap()
    }

    fn recover_bytes(img: &[u8]) -> cfg::Program {
        let image = load(img).unwrap();
        cfg::recover(image.as_ref()).unwrap()
    }

    /// Drop the two runtime-only stats (`rounds` is deterministic and kept;
    /// `threads_used` is the sole thread-count-dependent field) so two runs
    /// at different thread counts compare byte-for-byte.
    fn mask_threads(mut p: cfg::Program) -> cfg::Program {
        p.stats.threads_used = 0;
        p
    }

    /// Zero the parallel-only stats so a parallel program compares against
    /// a `cfg::recover` program (which never sets them).
    fn mask_parallel_stats(mut p: cfg::Program) -> cfg::Program {
        p.stats.rounds = 0;
        p.stats.threads_used = 0;
        p.stats.function_cap_hit = false;
        p
    }

    /// The x86-64 AArch64-free PE multi-function fixtures below reuse the
    /// base PE fixture's `.text` (RVA 0x1000, file 0x200, 0x100 bytes) and
    /// drop the import data directory so the eager import parse sees none.
    fn blank_pe() -> Vec<u8> {
        let mut img = synthetic_pe64();
        let dirs = 0x80 + 4 + 20 + 112;
        img[dirs + 8..dirs + 16].fill(0);
        img
    }

    /// A cyclic call chain of `n` x86 functions at an 8-byte stride in
    /// `.text`: function `i` does `call fn[(i+1) % n]; ret`, so all `n`
    /// are discovered transitively from the single entry (function 0), the
    /// cycle exercising callee de-duplication. Requires `8 * n <= 0x100`.
    fn call_chain_pe(n: usize) -> Vec<u8> {
        assert!(8 * n <= 0x100, "chain overflows .text");
        let mut img = blank_pe();
        let base_rva: i64 = 0x1000;
        for i in 0..n {
            let call_rva = base_rva + 8 * i as i64;
            let next_rva = call_rva + 5; // call rel32 is 5 bytes
            let target_rva = base_rva + 8 * ((i + 1) % n) as i64;
            let rel = (target_rva - next_rva) as i32;
            let file = 0x200 + 8 * i;
            img[file] = 0xE8; // call rel32
            img[file + 1..file + 5].copy_from_slice(&rel.to_le_bytes());
            img[file + 5] = 0xC3; // ret
        }
        img
    }

    /// An entry function that directly calls `k` leaf helpers, each a lone
    /// `ret`. Every helper is discovered straight from the entry, so
    /// corrupting one does not hide the others. Entry at RVA 0x1000,
    /// helpers at RVA 0x1040 + 4*j. Requires `5 * k + 1 <= 0x40`.
    fn fan_out_pe(k: usize) -> Vec<u8> {
        assert!(5 * k < 0x40, "entry body overflows into helpers");
        let mut img = blank_pe();
        let base_rva: i64 = 0x1000;
        let helper0_rva: i64 = 0x1040;
        for j in 0..k {
            let call_rva = base_rva + 5 * j as i64;
            let next_rva = call_rva + 5;
            let target_rva = helper0_rva + 4 * j as i64;
            let rel = (target_rva - next_rva) as i32;
            let file = 0x200 + 5 * j;
            img[file] = 0xE8;
            img[file + 1..file + 5].copy_from_slice(&rel.to_le_bytes());
        }
        img[0x200 + 5 * k] = 0xC3; // entry's trailing ret
        for j in 0..k {
            img[0x240 + 4 * j] = 0xC3; // helper j: ret
        }
        img
    }

    /// The x86 IAT + thunk fixture from `cfg`'s tests, rebuilt here so the
    /// parallel engine is exercised against import resolution and thunks.
    fn import_call_fixture() -> Vec<u8> {
        let mut img = with_imports();
        let opt = 0x80 + 4 + 20;
        img[opt + 16..opt + 20].copy_from_slice(&0x10A0u32.to_le_bytes());
        let code: &[u8] = &[
            0xFF, 0x15, 0xBA, 0xFF, 0xFF, 0xFF, // call [IAT ExitProcess]
            0xE8, 0x05, 0x00, 0x00, 0x00, // call thunk
            0xC3, // ret
            0x90, 0x90, 0x90, 0x90, //
            0xFF, 0x25, 0xB2, 0xFF, 0xFF, 0xFF, // jmp [IAT #7]
        ];
        img[0x2A0..0x2A0 + code.len()].copy_from_slice(code);
        img
    }

    /// The AArch64 ELF fixture from `cfg`'s tests (main/helper/callee).
    fn aarch64_elf_fixture() -> Vec<u8> {
        let mut img = synthetic_elf64();
        img[18..20].copy_from_slice(&183u16.to_le_bytes());
        let words: &[(usize, u32)] = &[
            (0x00, 0x5400_0040),
            (0x04, 0x9400_000F),
            (0x08, 0xD65F_03C0),
            (0x20, 0xD65F_03C0),
            (0x40, 0xD503_201F),
            (0x44, 0xD65F_03C0),
        ];
        for &(off, w) in words {
            img[0x100 + off..0x100 + off + 4].copy_from_slice(&w.to_le_bytes());
        }
        img
    }

    /// Every multi-function fixture, for the sweeping tests.
    fn multi_function_fixtures() -> Vec<Vec<u8>> {
        vec![
            call_chain_pe(24),
            fan_out_pe(8),
            import_call_fixture(),
            aarch64_elf_fixture(),
        ]
    }

    #[test]
    fn determinism_across_thread_counts_byte_for_byte() {
        for img in multi_function_fixtures() {
            let baseline = mask_threads(analyze_bytes_with(&img, &config_threads(1)));
            for t in [2, 3, 4, 8, 16, 32] {
                let other = mask_threads(analyze_bytes_with(&img, &config_threads(t)));
                assert_eq!(baseline, other, "thread count {t} changed the output");
            }
            // `threads_used` is the only field that tracks the pool size.
            let one = analyze_bytes_with(&img, &config_threads(1));
            let eight = analyze_bytes_with(&img, &config_threads(8));
            assert_eq!(one.stats.threads_used, 1);
            assert_eq!(eight.stats.threads_used, 8);
            assert_eq!(one.stats.rounds, eight.stats.rounds);
        }
    }

    #[test]
    fn determinism_is_stable_across_repeated_runs() {
        // Re-run several times to shake out any completion-order flakiness.
        let img = call_chain_pe(24);
        let first = mask_threads(analyze_bytes_with(&img, &config_threads(8)));
        for _ in 0..20 {
            let again = mask_threads(analyze_bytes_with(&img, &config_threads(8)));
            assert_eq!(first, again);
        }
    }

    #[test]
    fn equivalent_to_single_threaded_recover() {
        for img in multi_function_fixtures() {
            let par = analyze_bytes_with(&img, &config_threads(8));
            let seq = recover_bytes(&img);
            // Functions, call graph, and decode counts match exactly; only
            // the parallel-only stats (rounds/threads_used) differ.
            assert_eq!(par.functions, seq.functions);
            assert_eq!(par.call_graph, seq.call_graph);
            assert_eq!(par.stats.instructions, seq.stats.instructions);
            assert_eq!(par.stats.blocks, seq.stats.blocks);
            assert_eq!(mask_parallel_stats(par), seq);
        }
    }

    #[test]
    fn discovers_the_full_call_chain() {
        let img = call_chain_pe(24);
        let p = analyze_bytes_with(&img, &config_threads(4));
        assert_eq!(p.functions.len(), 24);
        // The cyclic chain is discovered one function per round until it
        // closes back on function 0.
        assert!(p.stats.rounds >= 2);
        // Every function makes exactly one direct call.
        assert_eq!(p.call_graph.len(), 24);
    }

    #[test]
    fn many_functions_high_thread_count_no_panic_or_deadlock() {
        // Widest fixture at a thread count far above the core count.
        let img = call_chain_pe(32);
        let p = analyze_bytes_with(&img, &config_threads(64));
        assert_eq!(p.functions.len(), 32);
        assert_eq!(p, recover_and_mask(&img, p.stats.threads_used, p.stats.rounds));
    }

    /// Rebuild the equivalent `cfg::recover` program with the parallel-only
    /// stats filled in, for a full-program equality assertion.
    fn recover_and_mask(img: &[u8], threads_used: usize, rounds: usize) -> cfg::Program {
        let mut seq = recover_bytes(img);
        seq.stats.rounds = rounds;
        seq.stats.threads_used = threads_used;
        seq
    }

    #[test]
    fn undecodable_entry_is_recorded_others_still_analyzed() {
        // Corrupt helper #3's entry to a leading nop then an unmodeled x87
        // opcode: its function is recorded (one Undecodable-terminated
        // block) while the other seven helpers recover normally.
        let mut img = fan_out_pe(8);
        let bad = 0x240 + 4 * 3; // helper 3 file offset
        img[bad] = 0x90; // nop
        img[bad + 1] = 0xD8; // x87 escape ...
        img[bad + 2] = 0x00; // ... unmodeled -> Undecodable
        let p = analyze_bytes_with(&img, &config_threads(8));

        // Entry + 8 helpers all present.
        assert_eq!(p.functions.len(), 9);
        let bad_entry = PE_BASE + 0x1040 + 4 * 3;
        assert_eq!(
            p.functions[&bad_entry].blocks,
            BTreeMap::from([(
                bad_entry,
                BasicBlock {
                    start: bad_entry,
                    end: bad_entry + 1,
                    terminator: Terminator::Undecodable,
                    successors: vec![],
                },
            )])
        );
        // A healthy helper still recovered as a lone ret block.
        let good_entry = PE_BASE + 0x1040;
        assert_eq!(
            p.functions[&good_entry].blocks[&good_entry].terminator,
            Terminator::Return
        );
        // Still equals single-threaded recovery.
        assert_eq!(p.functions, recover_bytes(&img).functions);
    }

    #[test]
    fn max_functions_cap_stops_cleanly() {
        let img = call_chain_pe(24);
        let config = Config {
            threads: 4,
            cfg: cfg::Config::default(),
            max_functions: 5,
        };
        let p = analyze_bytes_with(&img, &config);
        assert_eq!(p.functions.len(), 5);
        assert!(p.stats.function_cap_hit);
        // The five lowest-address functions are the ones kept.
        let kept: Vec<u64> = p.functions.keys().copied().collect();
        assert_eq!(
            kept,
            (0..5).map(|i| PE_ENTRY + 8 * i).collect::<Vec<_>>()
        );
        // The cap is deterministic across thread counts too.
        let other = analyze_bytes_with(
            &img,
            &Config {
                threads: 16,
                ..config.clone()
            },
        );
        assert_eq!(mask_threads(p), mask_threads(other));
    }

    #[test]
    fn max_functions_cap_exactly_at_seed_count_is_not_hit() {
        // 8 helpers + 1 entry = 9 functions; a cap of exactly 9 must not
        // trip the flag.
        let img = fan_out_pe(8);
        let p = analyze_bytes_with(
            &img,
            &Config {
                threads: 4,
                cfg: cfg::Config::default(),
                max_functions: 9,
            },
        );
        assert_eq!(p.functions.len(), 9);
        assert!(!p.stats.function_cap_hit);
    }

    #[test]
    fn auto_thread_count_fallback_works() {
        // threads:0 selects available_parallelism (>= 1) and recovers the
        // same program as an explicit single thread.
        let img = call_chain_pe(24);
        let auto = analyze_bytes_with(&img, &config_threads(0));
        assert!(auto.stats.threads_used >= 1);
        assert_eq!(
            mask_threads(auto),
            mask_threads(analyze_bytes_with(&img, &config_threads(1)))
        );
        // Default config also uses the auto path.
        let image = load(&img).unwrap();
        let def = analyze(image.as_ref()).unwrap();
        assert!(def.stats.threads_used >= 1);
    }

    #[test]
    fn single_function_diamond_matches_recover() {
        // A lone conditional diamond (no calls): the degenerate one-round,
        // one-function case still equals recover.
        let mut img = blank_pe();
        let code: &[u8] = &[
            0x31, 0xC0, 0x74, 0x05, 0xFF, 0xC0, 0xEB, 0x03, 0x90, 0xFF, 0xC8, 0xC3,
        ];
        img[0x200..0x200 + code.len()].copy_from_slice(code);
        let p = analyze_bytes_with(&img, &config_threads(8));
        assert_eq!(p.functions, recover_bytes(&img).functions);
        assert!(p.call_graph.is_empty());
        assert_eq!(p.stats.rounds, 1);
    }

    #[test]
    fn empty_image_recovers_nothing() {
        let img = crate::elf::tests::synthetic_dynamic_elf64();
        let p = analyze_bytes_with(&img, &config_threads(8));
        assert!(p.functions.is_empty());
        assert!(p.call_graph.is_empty());
        assert_eq!(p.stats.rounds, 0);
        // Everything but threads_used matches recover's empty result.
        assert_eq!(mask_parallel_stats(p).stats, recover_bytes(&img).stats);
    }

    #[test]
    fn unsupported_arch_is_a_typed_error() {
        let mut img = synthetic_elf64();
        img[18..20].copy_from_slice(&0xF00u16.to_le_bytes());
        let image = load(&img).unwrap();
        assert!(matches!(
            analyze(image.as_ref()),
            Err(crate::error::ParseError::Unsupported(_))
        ));
    }

    fn config_threads(threads: usize) -> Config {
        Config {
            threads,
            ..Config::default()
        }
    }
}
