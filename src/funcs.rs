//! Function discovery: where a program's functions begin.
//!
//! [`discover`] combines four sources of function-start addresses, in
//! descending order of trust, and reports each start once tagged with
//! the highest-precedence source that found it:
//!
//! 1. [`Source::EntryPoint`] — the image's declared entry points
//!    ([`Image::entry_points`]).
//! 2. [`Source::Symbol`] — `Function`-kind symbols in executable
//!    regions ([`Image::symbols`]).
//! 3. [`Source::Unwind`] — compiler-emitted unwind/metadata tables
//!    ([`Image::function_starts_hint`]): PE `.pdata`, ELF `.eh_frame`,
//!    Mach-O `LC_FUNCTION_STARTS`. Authoritative and stripping-proof.
//! 4. [`Source::Prologue`] — a conservative scan for standard function
//!    prologues in executable regions, the only heuristic source, used
//!    last and only for addresses the first three missed.
//!
//! The high-precision sources come straight from the trait layer; the
//! prologue scan reads code through [`Image`] and [`decoder_for`] alone.
//! Outside its tests this module names no container format or ISA.
//!
//! [`seeds`] returns just the addresses, ready to feed
//! [`crate::cfg::recover`]. Everything here is deterministic (results in
//! ascending address order) and total (no input panics).

use std::collections::BTreeMap;

use crate::model::{Arch, Image, SymbolKind, decoder_for};

/// Where a discovered function start came from. Ordered most- to
/// least-authoritative; the `Ord` derive matches that ranking, so a
/// smaller `Source` wins when one address has several.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Source {
    /// A declared image entry point.
    EntryPoint,
    /// A `Function`-kind symbol.
    Symbol,
    /// A compiler-emitted unwind/metadata table entry.
    Unwind,
    /// An entry named in a Go `pclntab` — authoritative for Go binaries,
    /// which carry their own complete function table even when stripped.
    GoPclntab,
    /// A heuristic prologue-pattern match.
    Prologue,
}

/// One discovered function start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Function {
    /// Virtual address of the first instruction.
    pub va: u64,
    /// The highest-precedence source that found this address.
    pub source: Source,
}

/// Discovery settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Run the heuristic prologue scan. When false, only the three
    /// high-precision sources contribute.
    pub scan_prologues: bool,
    /// Cap on the number of discovered functions. The scan stops adding
    /// once this many distinct starts are known.
    pub max_functions: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            scan_prologues: true,
            max_functions: 1 << 20,
        }
    }
}

/// Discover function starts in `image` with default settings.
pub fn discover(image: &dyn Image) -> Vec<Function> {
    discover_with(image, &Config::default())
}

/// Discover function starts in `image`, honoring `config`.
pub fn discover_with(image: &dyn Image, config: &Config) -> Vec<Function> {
    let regions = image.regions();
    // Executable, file-backed spans, as (va, byte slice), for the
    // in-region tests and the prologue scan. Within one region the
    // mapping is linear, so `va + k` is byte `k` of the slice.
    let exec: Vec<(u64, &[u8])> = regions
        .iter()
        .filter(|r| r.perms.x && r.size != 0)
        .filter_map(|r| {
            let off = image.va_to_offset(r.va)?;
            let bytes = image.bytes();
            let len = (r.size as usize).min(bytes.len().saturating_sub(off));
            (len != 0).then(|| (r.va, &bytes[off..off + len]))
        })
        .collect();
    let in_exec = |va: u64| exec.iter().any(|(base, s)| va >= *base && va - *base < s.len() as u64);

    // Insert keeping the highest-precedence (smallest) source per VA.
    let mut found: BTreeMap<u64, Source> = BTreeMap::new();
    let add = |va: u64, source: Source, found: &mut BTreeMap<u64, Source>| {
        found
            .entry(va)
            .and_modify(|s| {
                if source < *s {
                    *s = source;
                }
            })
            .or_insert(source);
    };

    for va in image.entry_points() {
        add(va, Source::EntryPoint, &mut found);
    }
    for sym in image.symbols() {
        if sym.kind == SymbolKind::Function && in_exec(sym.va) {
            add(sym.va, Source::Symbol, &mut found);
        }
    }
    for va in image.function_starts_hint() {
        if in_exec(va) {
            add(va, Source::Unwind, &mut found);
        }
    }
    // Go's pclntab is the definitive function table of a Go binary, so it
    // is folded in at high precedence — it names thousands of functions a
    // stripped image exposes no other way. Empty (and cheap) elsewhere.
    for f in crate::gopcln::recover(image) {
        if in_exec(f.va) {
            add(f.va, Source::GoPclntab, &mut found);
        }
    }

    if config.scan_prologues && found.len() < config.max_functions {
        scan_prologues(image, &exec, &mut found, config.max_functions);
    }

    found
        .into_iter()
        .take(config.max_functions)
        .map(|(va, source)| Function { va, source })
        .collect()
}

/// Just the discovered addresses, for seeding [`crate::cfg::recover`].
pub fn seeds(image: &dyn Image) -> Vec<u64> {
    discover(image).into_iter().map(|f| f.va).collect()
}

/// Whether `va` plausibly *begins* a function in `image`.
///
/// The same conservative test the prologue scan applies, exposed for seed
/// sources that only *suspect* a start — a data pointer of kind
/// [`crate::xref::XrefKind::AddressOf`] whose target lands in code, say —
/// and must confirm it before promoting it to a function. A candidate
/// must lie in an executable, file-backed region, be ISA-aligned, decode
/// cleanly, and open with a canonical prologue. Returns `false` for
/// [`Arch::Other`] and never panics.
pub fn looks_like_function_start(image: &dyn Image, va: u64) -> bool {
    let arch = image.arch();
    let Some(decoder) = decoder_for(arch) else {
        return false;
    };
    // Must map to a file-backed byte inside an executable region.
    if !image
        .regions()
        .iter()
        .any(|r| r.perms.x && r.size != 0 && va >= r.va && va - r.va < r.size)
    {
        return false;
    }
    let Some(off) = image.va_to_offset(va) else {
        return false;
    };
    let bytes = image.bytes();
    let Some(slice) = bytes.get(off..) else {
        return false;
    };
    match arch {
        Arch::X86_64 => is_x86_prologue(slice) && decoder.decode_flow(slice, va).is_ok(),
        Arch::Aarch64 => {
            // Fixed 4-byte instructions: a real start is word-aligned.
            if !va.is_multiple_of(4) || slice.len() < 4 {
                return false;
            }
            let word = u32::from_le_bytes(slice[..4].try_into().unwrap());
            is_aarch64_prologue(word) && decoder.decode_flow(slice, va).is_ok()
        }
        Arch::Other => false,
    }
}

/// Scan executable regions for standard prologues, adding starts not
/// already found. Conservative: a candidate must both match a known
/// pattern and decode cleanly.
fn scan_prologues(
    image: &dyn Image,
    exec: &[(u64, &[u8])],
    found: &mut BTreeMap<u64, Source>,
    max_functions: usize,
) {
    let arch = image.arch();
    let Some(decoder) = decoder_for(arch) else {
        return;
    };
    for (base, slice) in exec {
        match arch {
            Arch::X86_64 => {
                let mut i = 0;
                while i < slice.len() {
                    if found.len() >= max_functions {
                        return;
                    }
                    if is_x86_prologue(&slice[i..]) {
                        let va = base.wrapping_add(i as u64);
                        if !found.contains_key(&va)
                            && decoder.decode_flow(&slice[i..], va).is_ok()
                        {
                            found.insert(va, Source::Prologue);
                        }
                    }
                    i += 1;
                }
            }
            Arch::Aarch64 => {
                // Fixed 4-byte instructions: only word-aligned starts.
                let mut i = 0;
                while i + 4 <= slice.len() {
                    if found.len() >= max_functions {
                        return;
                    }
                    let word = u32::from_le_bytes(slice[i..i + 4].try_into().unwrap());
                    // `paciasp` may precede the frame store; either alone
                    // is a strong start signal.
                    if is_aarch64_prologue(word) {
                        let va = base.wrapping_add(i as u64);
                        if !found.contains_key(&va)
                            && decoder.decode_flow(&slice[i..], va).is_ok()
                        {
                            found.insert(va, Source::Prologue);
                        }
                    }
                    i += 4;
                }
            }
            Arch::Other => return,
        }
    }
}

/// `endbr64` (`F3 0F 1E FA`) or `push rbp; mov rbp, rsp`
/// (`55 48 89 E5`) — the two canonical x86-64 function openings.
fn is_x86_prologue(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xF3, 0x0F, 0x1E, 0xFA]) || bytes.starts_with(&[0x55, 0x48, 0x89, 0xE5])
}

/// A canonical AArch64 function opening. Real compiler output (Rust and
/// clang at any optimization level) opens a non-leaf function with one of
/// a small, well-defined family, all of which touch the stack pointer or
/// the pointer-auth state on entry:
///
/// - `paciasp` / `pacibsp` — pointer-auth sign of the link register.
/// - `stp <Xt>, <Xt2>, [sp, #-imm]!` — a callee-saved (or frame-record)
///   pair pushed with pre-index writeback onto `sp`. The pair is *any*
///   register pair, not only `x29`/`x30`: an optimized function saves
///   whatever callee-saved registers it uses first.
/// - `sub sp, sp, #imm` — a plain frame allocation.
///
/// Leaf functions that touch neither the stack nor the auth state (opening
/// with a bare `mov`/`b`) are deliberately *not* matched: those openings
/// are indistinguishable from mid-function code and would invite false
/// starts.
fn is_aarch64_prologue(word: u32) -> bool {
    if word == 0xD503_233F || word == 0xD503_237F {
        return true; // paciasp / pacibsp
    }
    let rt = word & 0x1F;
    let rt2 = (word >> 10) & 0x1F;
    let rn = (word >> 5) & 0x1F;
    let rd = word & 0x1F;
    // STP <Xt>, <Xt2>, [sp, #-imm]! — 64-bit, pre-index store onto SP.
    let opc = word >> 30; // 0b10 = 64-bit
    let klass = (word >> 27) & 0x7; // 0b101 = load/store pair
    let v = (word >> 26) & 0x1; // 0 = general registers
    let mode = (word >> 23) & 0x7; // 0b011 = pre-index
    let load = (word >> 22) & 0x1; // 0 = store
    let is_frame_push = opc == 0b10
        && klass == 0b101
        && v == 0
        && mode == 0b011
        && load == 0
        && rn == 31
        && rt != rt2;
    // SUB sp, sp, #imm — 64-bit subtract-immediate with SP as source/dest.
    let is_sp_alloc = (word >> 23) == 0b1_1010_0010 && rd == 31 && rn == 31;
    is_frame_push || is_sp_alloc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImportSlot, Perms, Region, Symbol, SymbolKind};

    /// A fully controllable single-region [`Image`] whose region maps at
    /// file offset 0, so byte `k` sits at `region_va + k`.
    struct Mock {
        arch: Arch,
        bytes: Vec<u8>,
        region_va: u64,
        exec: bool,
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
            vec![Region {
                name: "text".into(),
                va: self.region_va,
                size: self.bytes.len() as u64,
                perms: Perms {
                    r: true,
                    w: false,
                    x: self.exec,
                },
            }]
        }
        fn symbols(&self) -> Vec<Symbol> {
            self.syms.clone()
        }
        fn import_slots(&self) -> Vec<ImportSlot> {
            Vec::new()
        }
        fn va_to_offset(&self, va: u64) -> Option<usize> {
            let off = va.checked_sub(self.region_va)? as usize;
            (off < self.bytes.len()).then_some(off)
        }
        fn bytes(&self) -> &[u8] {
            &self.bytes
        }
        fn function_starts_hint(&self) -> Vec<u64> {
            self.hints.clone()
        }
    }

    fn func_sym(va: u64) -> Symbol {
        Symbol {
            name: format!("f_{va:x}"),
            va,
            kind: SymbolKind::Function,
        }
    }

    #[test]
    fn matches_canonical_prologues() {
        assert!(is_x86_prologue(&[0xF3, 0x0F, 0x1E, 0xFA, 0x00])); // endbr64
        assert!(is_x86_prologue(&[0x55, 0x48, 0x89, 0xE5])); // push rbp; mov rbp,rsp
        assert!(!is_x86_prologue(&[0x90, 0x90, 0x90, 0x90])); // nops
        assert!(is_aarch64_prologue(0xD503_233F)); // paciasp
        assert!(is_aarch64_prologue(0xD503_237F)); // pacibsp (Apple/Rust)
        assert!(is_aarch64_prologue(0xA9BF_7BFD)); // stp x29,x30,[sp,#-0x10]!
        // Optimized frames push arbitrary callee-saved pairs, not only
        // x29/x30 — all are prologues (observed in real Rust arm64 output).
        assert!(is_aarch64_prologue(0xA9BC_5FF8)); // stp x24,x23,[sp,#-0x40]!
        assert!(is_aarch64_prologue(0xA9BD_57F6)); // stp x22,x21,[sp,#-0x30]!
        assert!(is_aarch64_prologue(0xD102_83FF)); // sub sp, sp, #0xa0
        assert!(is_aarch64_prologue(0xD100_83FF)); // sub sp, sp, #0x20
        assert!(!is_aarch64_prologue(0xD65F_03C0)); // ret
        assert!(!is_aarch64_prologue(0xAA01_03E2)); // mov x2, x1 (leaf: not matched)
        assert!(!is_aarch64_prologue(0x1400_387C)); // b (not matched)
        // A store pair that is *not* pre-index writeback onto SP is not a
        // frame push: `stp x29,x30,[sp,#0x10]` (plain offset) must not match.
        assert!(!is_aarch64_prologue(0xA901_7BFD));
    }

    #[test]
    fn looks_like_function_start_gates_on_region_and_prologue() {
        // A64 image: prologue at 0x2000, a `ret` (non-prologue) at 0x2004.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0xA9BF_7BFDu32.to_le_bytes()); // stp x29,x30,..!
        bytes.extend_from_slice(&0xD65F_03C0u32.to_le_bytes()); // ret
        let exec = Mock {
            arch: Arch::Aarch64,
            bytes: bytes.clone(),
            region_va: 0x2000,
            exec: true,
            entries: vec![],
            syms: vec![],
            hints: vec![],
        };
        assert!(looks_like_function_start(&exec, 0x2000)); // prologue, aligned
        assert!(!looks_like_function_start(&exec, 0x2004)); // ret, not a start
        assert!(!looks_like_function_start(&exec, 0x2002)); // misaligned
        assert!(!looks_like_function_start(&exec, 0x9999)); // outside region

        // The same prologue bytes in a non-executable region never qualify.
        let data = Mock {
            arch: Arch::Aarch64,
            bytes,
            region_va: 0x2000,
            exec: false,
            entries: vec![],
            syms: vec![],
            hints: vec![],
        };
        assert!(!looks_like_function_start(&data, 0x2000));
    }

    #[test]
    fn entry_point_outranks_a_prologue_at_the_same_va() {
        // 0x1000: endbr64 (also the entry). 0x1004: push rbp; mov rbp,rsp.
        let m = Mock {
            arch: Arch::X86_64,
            bytes: vec![0xF3, 0x0F, 0x1E, 0xFA, 0x55, 0x48, 0x89, 0xE5, 0xC3],
            region_va: 0x1000,
            exec: true,
            entries: vec![0x1000],
            syms: vec![],
            hints: vec![],
        };
        let got = discover(&m);
        assert_eq!(
            got,
            vec![
                Function { va: 0x1000, source: Source::EntryPoint },
                Function { va: 0x1004, source: Source::Prologue },
            ]
        );
        // seeds() is just the addresses.
        assert_eq!(seeds(&m), vec![0x1000, 0x1004]);
        // Determinism.
        assert_eq!(discover(&m), got);
    }

    #[test]
    fn symbol_outranks_unwind_and_prologue() {
        // A Function symbol, an unwind hint, and a real prologue all at
        // 0x1004; the symbol (highest precedence of the three) wins.
        let m = Mock {
            arch: Arch::X86_64,
            bytes: vec![0x90, 0x90, 0x90, 0x90, 0x55, 0x48, 0x89, 0xE5, 0xC3],
            region_va: 0x1000,
            exec: true,
            entries: vec![],
            syms: vec![func_sym(0x1004)],
            hints: vec![0x1004],
        };
        assert_eq!(
            discover(&m),
            vec![Function { va: 0x1004, source: Source::Symbol }]
        );
    }

    #[test]
    fn unwind_hint_outside_executable_region_is_dropped() {
        let m = Mock {
            arch: Arch::X86_64,
            bytes: vec![0xC3, 0xC3],
            region_va: 0x1000,
            exec: false, // not executable
            entries: vec![],
            syms: vec![],
            hints: vec![0x1000],
        };
        assert!(discover(&m).is_empty());
    }

    #[test]
    fn prologue_scan_can_be_disabled() {
        let m = Mock {
            arch: Arch::X86_64,
            bytes: vec![0x55, 0x48, 0x89, 0xE5, 0xC3],
            region_va: 0x1000,
            exec: true,
            entries: vec![],
            syms: vec![],
            hints: vec![],
        };
        let cfg = Config {
            scan_prologues: false,
            ..Config::default()
        };
        assert!(discover_with(&m, &cfg).is_empty());
        // With scanning on, the prologue at the region start is found.
        assert_eq!(discover(&m).len(), 1);
    }

    #[test]
    fn aarch64_prologues_are_word_aligned_scans() {
        // paciasp at 0x2000, stp frame record at 0x2004.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0xD503_233Fu32.to_le_bytes());
        bytes.extend_from_slice(&0xA9BF_7BFDu32.to_le_bytes());
        bytes.extend_from_slice(&0xD65F_03C0u32.to_le_bytes()); // ret
        let m = Mock {
            arch: Arch::Aarch64,
            bytes,
            region_va: 0x2000,
            exec: true,
            entries: vec![],
            syms: vec![],
            hints: vec![],
        };
        assert_eq!(
            discover(&m),
            vec![
                Function { va: 0x2000, source: Source::Prologue },
                Function { va: 0x2004, source: Source::Prologue },
            ]
        );
    }

    #[test]
    fn arbitrary_bytes_never_panic_and_stay_sorted() {
        let bytes: Vec<u8> = (0..=255u8).cycle().take(1024).collect();
        let m = Mock {
            arch: Arch::X86_64,
            bytes,
            region_va: 0x1000,
            exec: true,
            entries: vec![0x1000],
            syms: vec![],
            hints: vec![],
        };
        let got = discover(&m);
        assert!(got.windows(2).all(|w| w[0].va < w[1].va));
        assert_eq!(discover(&m), got); // deterministic
    }

    #[test]
    fn integrates_with_a_real_loader() {
        // Discovery over an actual parsed ELF image goes through the
        // trait unchanged: deterministic, sorted, seeds mirror discover.
        let data = crate::elf::tests::synthetic_elf64();
        let img = crate::model::ElfImage::parse(&data).unwrap();
        let a = discover(&img);
        let b = discover(&img);
        assert_eq!(a, b);
        assert!(a.windows(2).all(|w| w[0].va < w[1].va));
        assert_eq!(seeds(&img), a.iter().map(|f| f.va).collect::<Vec<_>>());
    }
}
