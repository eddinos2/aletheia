//! Rust panic metadata — mining `core::panic::Location` for source facts.
//!
//! Every `panic!`, `unwrap`, `expect`, arithmetic-overflow check, and
//! slice-index check in a Rust binary references a static
//! `core::panic::Location { file: &str, line: u32, col: u32 }`, and those
//! records survive stripping. [`mine`] finds them in the data sections
//! and [`attribute`] connects them to the functions that reference them,
//! recovering source-file (and line) hints for otherwise nameless code —
//! the naming hint a listing can someday show beside `sub_140001000`.
//! Best-effort by contract: conservative validation over recall, empty
//! result when nothing qualifies, and no input ever panics.
//!
//! Mining is symbol-independent: because the `Location` records live in
//! read-only data, [`mine`] finds the same sites on a stripped binary as on
//! an unstripped one. *Attribution* is not — [`attribute`] connects a site
//! to a function through the recovered call/xref graph, so it degrades with
//! whatever function discovery finds. A function reached only through a code
//! pointer (never directly called) may hold its facts unattributed on a
//! stripped image even though the site itself was mined.
//!
//! # The record
//!
//! On a 64-bit target a `Location` is a 24-byte, 8-byte-aligned record in
//! read-only data:
//!
//! ```text
//! +0   file_ptr : u64   VA of the path bytes (a `&str`, never NUL-terminated)
//! +8   file_len : u64   length of the path, in bytes
//! +16  line     : u32   1-based source line
//! +20  col      : u32   1-based source column
//! ```
//!
//! The path is a compile-time string: `src/main.rs` for the crate being
//! built, `/rustc/<hash>/library/core/src/option.rs` for a record inlined
//! from the standard library.
//!
//! # What is accepted, and why so little
//!
//! A false hint misleads an analyst; a missed one costs nothing. Every
//! 8-aligned position in a readable, non-executable region is read as a
//! candidate and accepted only when *all* of these hold:
//!
//! - `file_ptr .. file_ptr + file_len` lies inside one readable region
//!   and is backed by file bytes;
//! - `1 <= file_len <= 512`;
//! - those bytes are valid UTF-8 with no NUL and no ASCII control
//!   character (`< 0x20` or `0x7F`);
//! - the path ends in the byte-exact suffix `.rs`;
//! - `1 <= line <= 1_000_000` and `1 <= col <= 10_000`.
//!
//! Deliberate limits of that rule, all costing recall rather than
//! precision: only the 64-bit little-endian layout is decoded (both
//! architectures this crate decodes are 64-bit), and only non-executable
//! regions are scanned — a Mach-O image that keeps its `__const` inside
//! the executable `__TEXT` segment therefore yields nothing rather than
//! risking a text-section false positive.
//!
//! # Attribution
//!
//! [`attribute`] builds the crate's [`xref`] index over a recovered
//! [`cfg::Program`] and keeps every *data* reference whose target is a
//! mined site, attributing it to the function whose basic block contains
//! the referencing instruction. References landing at `va + 8` and
//! `va + 16` also count: a compiler that materializes the parts of the
//! record separately (the `&str` pointer, its length, or the
//! `line`/`col` word) points into the middle of the record rather than at
//! its head. Exact hits win over those aliases when both are possible.
//!
//! # Determinism and resource caps
//!
//! Sites come back sorted by VA and deduplicated; facts live in B-trees
//! keyed by address and by `(file, line)`, so the same `(image, program)`
//! always yields the same [`Attribution`] and the same [`render`] text.
//! The scan is linear in the bytes of the regions it walks and stops at
//! `MAX_SITES` records; a function keeps at most `MAX_FACTS_PER_FUNCTION`
//! facts, and [`render`]'s file histogram lists at most
//! `MAX_HISTOGRAM_FILES` files. Every read is bounds-checked and every
//! address computation is checked or saturating.

use std::collections::{BTreeMap, BTreeSet};

use crate::model::Image;
use crate::{cfg, xref};

/// Encoded size of a 64-bit `core::panic::Location` record.
const RECORD_SIZE: u64 = 24;

/// Alignment a `Location` record is emitted at, and the scan step.
const RECORD_ALIGN: u64 = 8;

/// Upper bound on mined sites. Real Rust binaries hold thousands to tens
/// of thousands; this leaves generous headroom while bounding the work a
/// hostile image full of record-shaped bytes can demand.
const MAX_SITES: usize = 1 << 20;

/// Upper bound on an accepted path's byte length. Real paths — including
/// `/rustc/<hash>/library/...` — are far shorter.
const MAX_PATH_LEN: u64 = 512;

/// Upper bound on an accepted `line`.
const MAX_LINE: u32 = 1_000_000;

/// Upper bound on an accepted `col`.
const MAX_COL: u32 = 10_000;

/// Upper bound on the facts retained for one function.
const MAX_FACTS_PER_FUNCTION: usize = 4096;

/// Upper bound on the files [`render`] lists in its histogram.
const MAX_HISTOGRAM_FILES: usize = 32;

/// One recovered `core::panic::Location`: a validated source position
/// planted in read-only data by a panicking construct.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PanicSite {
    /// Virtual address of the 24-byte record itself.
    pub va: u64,
    /// Source path, exactly as the compiler embedded it (validated UTF-8
    /// ending in `.rs`).
    pub file: String,
    /// 1-based source line.
    pub line: u32,
    /// 1-based source column.
    pub col: u32,
}

/// One source fact attributed to a function: the file and line of a
/// `Location` its code references. Ordered by `(file, line)`, which is
/// the order [`Attribution`] presents facts in.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fact {
    /// Source path from the referenced [`PanicSite`].
    pub file: String,
    /// 1-based source line from the referenced [`PanicSite`].
    pub line: u32,
}

/// Which functions reference which source positions: function entry VA →
/// the set of `(file, line)` facts its code points at.
///
/// Functions that reference no site are absent, so the map is exactly the
/// set of functions this pass has something to say about.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Attribution {
    /// Facts keyed by function entry VA, each set sorted by
    /// `(file, line)` and capped at `MAX_FACTS_PER_FUNCTION`.
    pub by_function: BTreeMap<u64, BTreeSet<Fact>>,
}

impl Attribution {
    /// The facts attributed to `entry`, in `(file, line)` order. Empty
    /// when the function references no site.
    pub fn facts(&self, entry: u64) -> impl Iterator<Item = &Fact> {
        self.by_function.get(&entry).into_iter().flatten()
    }

    /// The distinct files attributed to `entry`, in lexicographic order.
    pub fn files(&self, entry: u64) -> impl Iterator<Item = &str> {
        // Facts are ordered by (file, line), so distinct files are the
        // runs of equal `file` — adjacent deduplication suffices.
        self.facts(entry)
            .scan(None::<&str>, |prev, fact| {
                let file = fact.file.as_str();
                let first = *prev != Some(file);
                *prev = Some(file);
                Some(if first { Some(file) } else { None })
            })
            .flatten()
    }

    /// The file most of `entry`'s facts come from — the naming hint for
    /// an otherwise anonymous function. Ties break lexicographically
    /// (the smallest path wins). `None` when nothing is attributed.
    pub fn dominant_file(&self, entry: u64) -> Option<&str> {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for fact in self.facts(entry) {
            *counts.entry(fact.file.as_str()).or_default() += 1;
        }
        counts
            .into_iter()
            // Highest count wins; among equal counts the *reversed* name
            // comparison makes the lexicographically smallest the max.
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(file, _)| file)
    }

    /// Number of functions with at least one attributed fact.
    pub fn len(&self) -> usize {
        self.by_function.len()
    }

    /// Whether nothing was attributed to any function.
    pub fn is_empty(&self) -> bool {
        self.by_function.is_empty()
    }
}

/// Mine every `core::panic::Location` record an image's readable,
/// non-executable regions hold, with the default site cap.
///
/// Returns the sites sorted by VA and deduplicated by VA, or an empty
/// vector for a non-Rust image (and for any image whose records fail the
/// validation the module docs describe). Never errors, never panics.
pub fn mine(image: &dyn Image) -> Vec<PanicSite> {
    mine_with(image, MAX_SITES)
}

/// [`mine`] with a caller-supplied cap on how many sites are retained.
///
/// Reaching `max_sites` stops the scan; the result is then a prefix of
/// the full one in address order, never a different set.
pub fn mine_with(image: &dyn Image, max_sites: usize) -> Vec<PanicSite> {
    let miner = Miner {
        image,
        bytes: image.bytes(),
        readable: ranges(image, |r| r.perms.r),
    };
    let mut out: Vec<PanicSite> = Vec::new();
    'scan: for (start, end) in ranges(image, |r| r.perms.r && !r.perms.x) {
        let mut va = align_up(start);
        while va.saturating_add(RECORD_SIZE) <= end {
            if out.len() >= max_sites {
                break 'scan;
            }
            if let Some(site) = miner.site_at(va) {
                out.push(site);
            }
            let Some(next) = va.checked_add(RECORD_ALIGN) else {
                break 'scan;
            };
            va = next;
        }
    }
    finalize(out)
}

/// Sort by VA (then by content, for determinism) and keep one site per
/// VA — regions that overlap can offer the same record twice.
fn finalize(mut sites: Vec<PanicSite>) -> Vec<PanicSite> {
    sites.sort();
    sites.dedup_by_key(|s| s.va);
    sites
}

/// The `[start, end)` VA ranges of the regions matching `keep`, sorted
/// and deduplicated. Zero-sized and overflowing regions are dropped.
fn ranges(image: &dyn Image, keep: impl Fn(&crate::model::Region) -> bool) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = image
        .regions()
        .iter()
        .filter(|r| r.size > 0 && keep(r))
        .filter_map(|r| Some((r.va, r.va.checked_add(r.size)?)))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The first [`RECORD_ALIGN`]-aligned address at or after `va`
/// (saturating, so an address near `u64::MAX` cannot wrap).
fn align_up(va: u64) -> u64 {
    match va % RECORD_ALIGN {
        0 => va,
        rem => va.saturating_add(RECORD_ALIGN - rem),
    }
}

/// Whole-image scan state: the image, its bytes, and the readable ranges
/// a `file_ptr` is allowed to point into.
struct Miner<'a> {
    image: &'a dyn Image,
    bytes: &'a [u8],
    readable: Vec<(u64, u64)>,
}

impl Miner<'_> {
    /// Whether `[va, va + len)` lies wholly inside one readable region.
    fn is_readable(&self, va: u64, len: u64) -> bool {
        let Some(end) = va.checked_add(len) else {
            return false;
        };
        self.readable.iter().any(|&(s, e)| va >= s && end <= e)
    }

    /// The `len` file-backed bytes mapped at `va`, or `None` when the VA
    /// is unmapped, has no file backing, or the range runs off the end.
    fn read(&self, va: u64, len: usize) -> Option<&[u8]> {
        let off = self.image.va_to_offset(va)?;
        self.bytes.get(off..off.checked_add(len)?)
    }

    /// Decode and fully validate the candidate record at `va`.
    fn site_at(&self, va: u64) -> Option<PanicSite> {
        let record = self.read(va, RECORD_SIZE as usize)?;
        let file_ptr = u64_at(record, 0)?;
        let file_len = u64_at(record, 8)?;
        let line = u32_at(record, 16)?;
        let col = u32_at(record, 20)?;

        // Cheap scalar gates first: they reject nearly every candidate
        // without touching the string.
        if !(1..=MAX_LINE).contains(&line) || !(1..=MAX_COL).contains(&col) {
            return None;
        }
        if !(1..=MAX_PATH_LEN).contains(&file_len) {
            return None;
        }
        if !self.is_readable(file_ptr, file_len) {
            return None;
        }
        let path = self.read(file_ptr, usize::try_from(file_len).ok()?)?;
        // A compile-time Rust path: `.rs`-suffixed, printable, UTF-8.
        if !path.ends_with(b".rs") {
            return None;
        }
        if path.iter().any(|&b| b < 0x20 || b == 0x7F) {
            return None; // NUL and every other ASCII control
        }
        let file = std::str::from_utf8(path).ok()?;
        Some(PanicSite {
            va,
            file: file.to_string(),
            line,
            col,
        })
    }
}

/// Little-endian `u64` at `off` in `b`, or `None` if it does not fit.
fn u64_at(b: &[u8], off: usize) -> Option<u64> {
    let s = b.get(off..off.checked_add(8)?)?;
    Some(u64::from_le_bytes(s.try_into().ok()?))
}

/// Little-endian `u32` at `off` in `b`, or `None` if it does not fit.
fn u32_at(b: &[u8], off: usize) -> Option<u32> {
    let s = b.get(off..off.checked_add(4)?)?;
    Some(u32::from_le_bytes(s.try_into().ok()?))
}

/// Attribute the image's mined sites to the functions of `program`.
///
/// `program` must be the [`cfg::Program`] recovered from `image`. Sites
/// are mined with [`mine`]; use [`attribute_sites`] to reuse a set
/// already mined. An image whose architecture has no decoder — the one
/// case [`xref::compute`] rejects — yields an empty [`Attribution`]
/// rather than an error: this pass is a hint provider, not a loader.
/// Never panics.
pub fn attribute(image: &dyn Image, program: &cfg::Program) -> Attribution {
    attribute_sites(image, program, &mine(image))
}

/// [`attribute`] over an already-mined set of sites.
///
/// `sites` need not be sorted; sites that nothing references simply
/// contribute nothing.
pub fn attribute_sites(
    image: &dyn Image,
    program: &cfg::Program,
    sites: &[PanicSite],
) -> Attribution {
    let mut out = Attribution::default();
    if sites.is_empty() {
        return out;
    }
    let Ok(xrefs) = xref::compute(image, program) else {
        return out; // no decoder for this architecture
    };

    // Referenced VA → site. Exact record addresses are inserted first so
    // they win over the `+8` / `+16` aliases of a neighbouring record.
    let mut targets: BTreeMap<u64, &PanicSite> = BTreeMap::new();
    for site in sites {
        targets.insert(site.va, site);
    }
    for site in sites {
        for delta in [8u64, 16] {
            if let Some(va) = site.va.checked_add(delta) {
                targets.entry(va).or_insert(site);
            }
        }
    }

    for (&entry, func) in &program.functions {
        for block in func.blocks.values() {
            if block.end <= block.start {
                continue; // empty or malformed extent: nothing to query
            }
            for refs in xrefs.by_from.range(block.start..block.end) {
                for xref in refs.1 {
                    // Only data references: a call or branch landing on a
                    // record address is control flow, not a Location use.
                    if !xref.kind.is_data() {
                        continue;
                    }
                    let Some(site) = targets.get(&xref.to) else {
                        continue;
                    };
                    let facts = out.by_function.entry(entry).or_default();
                    if facts.len() < MAX_FACTS_PER_FUNCTION {
                        facts.insert(Fact {
                            file: site.file.clone(),
                            line: site.line,
                        });
                    }
                }
            }
        }
    }
    out
}

/// Render a deterministic report of the mined sites and their
/// attribution: the site count, a capped histogram of the files the
/// sites name, then one line per attributed function —
/// `{entry VA}  {dominant file}  ({n} site(s))`, where `n` counts the
/// distinct `(file, line)` facts attributed to it.
///
/// Sections with nothing to show are omitted. The result always ends in
/// a newline.
pub fn render(sites: &[PanicSite], attribution: &Attribution) -> String {
    if sites.is_empty() && attribution.is_empty() {
        return "no Rust panic metadata recovered\n".to_string();
    }
    let mut out = format!("Rust panic sites: {}\n", sites.len());

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for site in sites {
        *counts.entry(site.file.as_str()).or_default() += 1;
    }
    if !counts.is_empty() {
        // Most sites first; ties by path, so the order is total.
        let mut ranked: Vec<(&str, usize)> = counts.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        out.push_str(&format!("\n; -- files ({}) --\n", ranked.len()));
        let shown = ranked.len().min(MAX_HISTOGRAM_FILES);
        for &(file, n) in &ranked[..shown] {
            out.push_str(&format!("  {n:>6}  {file}\n"));
        }
        if let Some(rest) = ranked.len().checked_sub(shown).filter(|&r| r > 0) {
            // The histogram is capped and count-ranked, so the hidden tail is
            // the *lowest*-count files: say so, or a single-site file (a
            // crate's own source, typically) reads as missing rather than
            // merely ranked below the fold.
            let min_shown = ranked[shown - 1].1;
            out.push_str(&format!(
                "  ... {rest} more file(s), each with \u{2264} {min_shown} site(s)\n"
            ));
        }
    }

    if !attribution.is_empty() {
        out.push_str(&format!("\n; -- functions ({}) --\n", attribution.len()));
        for (&entry, facts) in &attribution.by_function {
            let file = attribution.dominant_file(entry).unwrap_or("(unknown)");
            out.push_str(&format!(
                "  {entry:#018x}  {file}  ({} site(s))\n",
                facts.len()
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Arch, ImportSlot, Perms, Region, Symbol, SymbolKind, load};

    /// VA (and file offset — the fixture image is identity-mapped) of the
    /// synthetic code region.
    const CODE_VA: u64 = 0x1000;
    /// VA of the synthetic read-only data region, far enough above the
    /// code that even the largest fixture body fits below it.
    const DATA_VA: u64 = 0x4_0000;

    // ---- Fixture image ---------------------------------------------------

    /// Identity-mapped image (VA == file offset) with a `.text` region at
    /// [`CODE_VA`] and a `.rdata` region at [`DATA_VA`] — the gopcln mock
    /// idiom, sized to whatever the test writes into it.
    struct FakeImage {
        data: Vec<u8>,
        regions: Vec<Region>,
        entries: Vec<u64>,
        symbols: Vec<Symbol>,
    }

    impl Image for FakeImage {
        fn arch(&self) -> Arch {
            Arch::X86_64
        }
        fn entry_points(&self) -> Vec<u64> {
            self.entries.clone()
        }
        fn regions(&self) -> Vec<Region> {
            self.regions.clone()
        }
        fn symbols(&self) -> Vec<Symbol> {
            self.symbols.clone()
        }
        fn import_slots(&self) -> Vec<ImportSlot> {
            Vec::new()
        }
        fn va_to_offset(&self, va: u64) -> Option<usize> {
            let off = usize::try_from(va).ok()?;
            (off <= self.data.len()).then_some(off)
        }
        fn bytes(&self) -> &[u8] {
            &self.data
        }
    }

    /// Builder for the read-only data region: strings and 24-byte
    /// `Location` records, laid out at [`DATA_VA`].
    #[derive(Default)]
    struct Data {
        bytes: Vec<u8>,
    }

    impl Data {
        fn va(&self) -> u64 {
            DATA_VA + self.bytes.len() as u64
        }

        fn align(&mut self) {
            while !self.bytes.len().is_multiple_of(RECORD_ALIGN as usize) {
                self.bytes.push(0);
            }
        }

        /// Append raw path bytes, returning `(va, len)` for a record.
        fn string(&mut self, path: &[u8]) -> (u64, u64) {
            let va = self.va();
            self.bytes.extend_from_slice(path);
            (va, path.len() as u64)
        }

        /// Append an 8-aligned record with the given fields, returning its
        /// VA.
        fn record(&mut self, file_ptr: u64, file_len: u64, line: u32, col: u32) -> u64 {
            self.align();
            let va = self.va();
            self.bytes.extend_from_slice(&file_ptr.to_le_bytes());
            self.bytes.extend_from_slice(&file_len.to_le_bytes());
            self.bytes.extend_from_slice(&line.to_le_bytes());
            self.bytes.extend_from_slice(&col.to_le_bytes());
            va
        }

        /// The common case: a path string plus a record pointing at it.
        fn site(&mut self, path: &str, line: u32, col: u32) -> u64 {
            let (ptr, len) = self.string(path.as_bytes());
            self.record(ptr, len, line, col)
        }
    }

    /// Builder for the code region, laid out at [`CODE_VA`].
    #[derive(Default)]
    struct Code {
        bytes: Vec<u8>,
    }

    impl Code {
        fn va(&self) -> u64 {
            CODE_VA + self.bytes.len() as u64
        }

        /// `lea rax, [rip + disp32]` (7 bytes) resolving to `target`.
        fn lea(&mut self, target: u64) {
            let next = self.va().wrapping_add(7) as i64;
            let disp = (target as i64 - next) as i32;
            self.bytes.extend_from_slice(&[0x48, 0x8D, 0x05]);
            self.bytes.extend_from_slice(&disp.to_le_bytes());
        }

        fn ret(&mut self) {
            self.bytes.push(0xC3);
        }

        /// Pad with `nop` until the next instruction lands on `va`.
        fn pad_to(&mut self, va: u64) {
            while self.va() < va {
                self.bytes.push(0x90);
            }
        }
    }

    /// Assemble a fixture image from a code body and a data body, with
    /// `symbols` naming extra function entries for control-flow seeding.
    fn image(code: &Code, data: &Data, symbols: &[(&str, u64)]) -> FakeImage {
        let mut buf = vec![0u8; DATA_VA as usize];
        assert!(
            code.bytes.len() <= (DATA_VA - CODE_VA) as usize,
            "code body overruns the data region"
        );
        buf[CODE_VA as usize..CODE_VA as usize + code.bytes.len()].copy_from_slice(&code.bytes);
        buf.extend_from_slice(&data.bytes);
        FakeImage {
            data: buf,
            regions: vec![
                Region {
                    name: ".text".into(),
                    va: CODE_VA,
                    size: code.bytes.len().max(1) as u64,
                    perms: Perms {
                        r: true,
                        w: false,
                        x: true,
                    },
                },
                Region {
                    name: ".rdata".into(),
                    va: DATA_VA,
                    size: data.bytes.len() as u64,
                    perms: Perms {
                        r: true,
                        w: false,
                        x: false,
                    },
                },
            ],
            entries: vec![CODE_VA],
            symbols: symbols
                .iter()
                .map(|&(name, va)| Symbol {
                    name: name.into(),
                    va,
                    kind: SymbolKind::Function,
                })
                .collect(),
        }
    }

    /// A data-only fixture: no code, just the records under test.
    fn data_image(data: &Data) -> FakeImage {
        image(&Code::default(), data, &[])
    }

    /// Mine, recover control flow, and attribute — the whole pipeline a
    /// caller runs.
    fn pipeline(img: &FakeImage) -> (Vec<PanicSite>, Attribution) {
        let sites = mine(img);
        let program = cfg::recover(img).expect("x86-64 recovers");
        let attribution = attribute(img, &program);
        (sites, attribution)
    }

    // ---- Mining: acceptance ---------------------------------------------

    #[test]
    fn a_valid_record_is_mined() {
        let mut data = Data::default();
        let va = data.site("src/main.rs", 42, 9);
        let sites = mine(&data_image(&data));

        assert_eq!(
            sites,
            vec![PanicSite {
                va,
                file: "src/main.rs".into(),
                line: 42,
                col: 9,
            }]
        );
    }

    #[test]
    fn a_rustc_library_path_is_mined() {
        let mut data = Data::default();
        let path = "/rustc/9b00956e56009bab2aa15d7bff10916599e3d6d6/library/core/src/option.rs";
        let va = data.site(path, 1, 1);
        let sites = mine(&data_image(&data));

        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].va, va);
        assert_eq!(sites[0].file, path);
    }

    #[test]
    fn records_in_a_writable_region_are_still_mined() {
        // `.data.rel.ro` is writable until the loader is done with it;
        // only the *executable* bit disqualifies a region.
        let mut data = Data::default();
        data.site("src/lib.rs", 7, 5);
        let mut img = data_image(&data);
        img.regions[1].perms.w = true;
        assert_eq!(mine(&img).len(), 1);
    }

    // ---- Mining: refusals ------------------------------------------------

    #[test]
    fn a_path_without_the_rs_suffix_is_refused() {
        for path in ["src/main.c", "src/main.rs.bak", "src/main.RS", "rs"] {
            let mut data = Data::default();
            data.site(path, 1, 1);
            assert!(mine(&data_image(&data)).is_empty(), "{path}");
        }
    }

    #[test]
    fn invalid_utf8_in_the_path_is_refused() {
        let mut data = Data::default();
        // 0xFF is not a legal UTF-8 byte anywhere.
        let (ptr, len) = data.string(b"src/\xffain.rs");
        data.record(ptr, len, 1, 1);
        assert!(mine(&data_image(&data)).is_empty());
    }

    #[test]
    fn an_embedded_nul_or_control_byte_is_refused() {
        for bad in [0u8, 0x09, 0x0A, 0x1F, 0x7F] {
            let mut data = Data::default();
            let (ptr, len) = data.string(&[b's', b'r', b'c', bad, b'a', b'.', b'r', b's']);
            data.record(ptr, len, 1, 1);
            assert!(mine(&data_image(&data)).is_empty(), "{bad:#04x}");
        }
    }

    #[test]
    fn out_of_range_line_and_column_are_refused() {
        let cases: &[(u32, u32)] = &[
            (0, 1),               // zero line
            (1, 0),               // zero column
            (MAX_LINE + 1, 1),    // implausible line
            (1, MAX_COL + 1),     // implausible column
            (u32::MAX, u32::MAX), // hostile both
        ];
        for &(line, col) in cases {
            let mut data = Data::default();
            data.site("src/main.rs", line, col);
            assert!(mine(&data_image(&data)).is_empty(), "{line}:{col}");
        }
        // The boundary values themselves are accepted.
        let mut data = Data::default();
        data.site("src/main.rs", MAX_LINE, MAX_COL);
        assert_eq!(mine(&data_image(&data)).len(), 1);
    }

    #[test]
    fn an_implausible_or_zero_file_length_is_refused() {
        let mut data = Data::default();
        let (ptr, _) = data.string(b"src/main.rs");
        data.record(ptr, 0, 1, 1); // empty path
        data.record(ptr, MAX_PATH_LEN + 1, 1, 1); // past the cap
        data.record(ptr, u64::MAX, 1, 1); // hostile length
        assert!(mine(&data_image(&data)).is_empty());
    }

    #[test]
    fn a_file_pointer_outside_every_readable_region_is_refused() {
        let mut data = Data::default();
        data.record(0xDEAD_0000, 11, 1, 1); // nothing is mapped there
        data.record(u64::MAX - 4, 11, 1, 1); // and this one would overflow
        assert!(mine(&data_image(&data)).is_empty());
    }

    #[test]
    fn a_path_running_past_the_end_of_its_region_is_refused() {
        let mut data = Data::default();
        let (ptr, len) = data.string(b"src/main.rs");
        // The length claims more bytes than the region holds after `ptr`.
        data.record(ptr, len + 4096, 1, 1);
        assert!(mine(&data_image(&data)).is_empty());
    }

    #[test]
    fn a_record_truncated_at_the_section_edge_is_refused() {
        let mut data = Data::default();
        let va = data.site("src/main.rs", 3, 4);
        // Cut the region eight bytes short of the record's end: the last
        // candidate no longer fits, and nothing is read past the edge.
        let mut img = data_image(&data);
        img.regions[1].size = va + 16 - DATA_VA;
        assert!(mine(&img).is_empty());

        // The same record with the *buffer* truncated instead: the read
        // fails rather than the extent check, and still no panic.
        let mut img = data_image(&data);
        img.data.truncate((va + 16) as usize);
        assert!(mine(&img).is_empty());
    }

    #[test]
    fn records_in_an_executable_region_are_not_scanned() {
        let mut data = Data::default();
        data.site("src/main.rs", 1, 1);
        let mut img = data_image(&data);
        img.regions[1].perms.x = true;
        assert!(mine(&img).is_empty());
    }

    #[test]
    fn an_unaligned_record_is_not_scanned() {
        let mut data = Data::default();
        // A one-byte shim before the record leaves it 8-misaligned.
        data.align();
        data.bytes.push(0);
        let (ptr, len) = data.string(b"src/main.rs");
        let va = DATA_VA + data.bytes.len() as u64;
        assert!(!va.is_multiple_of(RECORD_ALIGN));
        data.bytes.extend_from_slice(&ptr.to_le_bytes());
        data.bytes.extend_from_slice(&len.to_le_bytes());
        data.bytes.extend_from_slice(&1u32.to_le_bytes());
        data.bytes.extend_from_slice(&1u32.to_le_bytes());
        assert!(mine(&data_image(&data)).is_empty());
    }

    // ---- Mining: ordering, dedup, caps ----------------------------------

    #[test]
    fn sites_come_back_sorted_and_deduplicated() {
        let mut data = Data::default();
        let (ptr, len) = data.string(b"src/main.rs");
        let a = data.record(ptr, len, 10, 1);
        let b = data.record(ptr, len, 20, 1);
        let c = data.record(ptr, len, 30, 1);
        assert!(a < b && b < c);

        // Two overlapping regions both cover the records: each is offered
        // twice and must be reported once.
        let mut img = data_image(&data);
        img.regions.push(img.regions[1].clone());
        img.regions.push(Region {
            name: ".rodata".into(),
            va: DATA_VA,
            size: img.regions[1].size,
            perms: img.regions[1].perms,
        });

        let sites = mine(&img);
        assert_eq!(
            sites.iter().map(|s| s.va).collect::<Vec<_>>(),
            vec![a, b, c]
        );
        assert_eq!(
            sites.iter().map(|s| s.line).collect::<Vec<_>>(),
            vec![10, 20, 30]
        );
    }

    #[test]
    fn the_site_cap_truncates_the_scan() {
        let mut data = Data::default();
        let (ptr, len) = data.string(b"src/main.rs");
        for line in 1..=5u32 {
            data.record(ptr, len, line, 1);
        }
        let img = data_image(&data);
        assert_eq!(mine(&img).len(), 5);

        for cap in 0..=5usize {
            let capped = mine_with(&img, cap);
            assert_eq!(capped.len(), cap, "cap {cap}");
            // A truncated result is a prefix of the full one.
            assert_eq!(capped, mine(&img)[..cap]);
        }
        // The default cap is the documented one.
        assert_eq!(MAX_SITES, 1 << 20);
    }

    // ---- Attribution -----------------------------------------------------

    #[test]
    fn a_referencing_function_is_attributed_the_site() {
        let mut data = Data::default();
        let site = data.site("src/main.rs", 12, 5);

        let mut code = Code::default();
        code.lea(site);
        code.ret();
        let img = image(&code, &data, &[]);

        let (sites, attribution) = pipeline(&img);
        assert_eq!(sites.len(), 1);
        assert_eq!(attribution.len(), 1);
        assert_eq!(
            attribution.facts(CODE_VA).cloned().collect::<Vec<_>>(),
            vec![Fact {
                file: "src/main.rs".into(),
                line: 12
            }]
        );
        assert_eq!(
            attribution.files(CODE_VA).collect::<Vec<_>>(),
            vec!["src/main.rs"]
        );
        assert_eq!(attribution.dominant_file(CODE_VA), Some("src/main.rs"));
    }

    #[test]
    fn references_to_the_middle_of_a_record_also_attribute() {
        // A compiler that materializes the parts separately points at
        // `va + 8` (the length) or `va + 16` (the line/col word).
        let mut data = Data::default();
        let site = data.site("src/mid.rs", 77, 3);

        for delta in [8u64, 16] {
            let mut code = Code::default();
            code.lea(site + delta);
            code.ret();
            let img = image(&code, &data, &[]);
            let (_, attribution) = pipeline(&img);
            assert_eq!(
                attribution.facts(CODE_VA).cloned().collect::<Vec<_>>(),
                vec![Fact {
                    file: "src/mid.rs".into(),
                    line: 77
                }],
                "+{delta}"
            );
        }

        // A reference just past the record (+24) belongs to whatever
        // follows, not to this site.
        let mut code = Code::default();
        code.lea(site + 24);
        code.ret();
        let img = image(&code, &data, &[]);
        let (_, attribution) = pipeline(&img);
        assert!(attribution.is_empty());
    }

    #[test]
    fn an_exact_hit_outranks_a_neighbours_alias() {
        // Two sites eight bytes apart, so the second's exact address is
        // also the first's `+8` alias. Real records never collide like
        // this (they are 24 bytes wide), so the set is supplied directly.
        let mut data = Data::default();
        let first = data.site("src/first.rs", 1, 1);
        let mut code = Code::default();
        code.lea(first + 8);
        code.ret();
        let img = image(&code, &data, &[]);
        let program = cfg::recover(&img).expect("x86-64 recovers");

        let sites = vec![
            PanicSite {
                va: first,
                file: "src/first.rs".into(),
                line: 1,
                col: 1,
            },
            PanicSite {
                va: first + 8,
                file: "src/second.rs".into(),
                line: 2,
                col: 2,
            },
        ];
        // The exact match wins, whichever order the sites arrive in.
        for order in [sites.clone(), sites.iter().rev().cloned().collect()] {
            let attribution = attribute_sites(&img, &program, &order);
            assert_eq!(
                attribution.facts(CODE_VA).cloned().collect::<Vec<_>>(),
                vec![Fact {
                    file: "src/second.rs".into(),
                    line: 2
                }]
            );
        }

        // And with only the first site known, the alias still resolves.
        let attribution = attribute_sites(&img, &program, &sites[..1]);
        assert_eq!(
            attribution.facts(CODE_VA).cloned().collect::<Vec<_>>(),
            vec![Fact {
                file: "src/first.rs".into(),
                line: 1
            }]
        );
    }

    #[test]
    fn a_function_that_references_nothing_is_absent() {
        let mut data = Data::default();
        let site = data.site("src/main.rs", 5, 1);

        // `main` references the site; `helper` at 0x1100 is a bare `ret`.
        let helper = CODE_VA + 0x100;
        let mut code = Code::default();
        code.lea(site);
        code.ret();
        code.pad_to(helper);
        code.ret();
        let img = image(&code, &data, &[("helper", helper)]);

        let program = cfg::recover(&img).expect("x86-64 recovers");
        assert!(program.functions.contains_key(&helper));

        let attribution = attribute(&img, &program);
        assert_eq!(attribution.len(), 1);
        assert!(attribution.by_function.contains_key(&CODE_VA));
        assert!(!attribution.by_function.contains_key(&helper));
        assert!(attribution.facts(helper).next().is_none());
        assert_eq!(attribution.dominant_file(helper), None);
    }

    #[test]
    fn the_dominant_file_is_the_most_referenced_then_lexicographic() {
        // `b.rs` twice, `a.rs` once: the count decides.
        let mut data = Data::default();
        let a = data.site("src/a.rs", 1, 1);
        let b1 = data.site("src/b.rs", 1, 1);
        let b2 = data.site("src/b.rs", 2, 1);

        let mut code = Code::default();
        for target in [a, b1, b2] {
            code.lea(target);
        }
        code.ret();
        let img = image(&code, &data, &[]);
        let (_, attribution) = pipeline(&img);
        assert_eq!(attribution.dominant_file(CODE_VA), Some("src/b.rs"));
        assert_eq!(
            attribution.files(CODE_VA).collect::<Vec<_>>(),
            vec!["src/a.rs", "src/b.rs"]
        );

        // One fact each: the lexicographically smaller path wins the tie.
        let mut code = Code::default();
        for target in [a, b1] {
            code.lea(target);
        }
        code.ret();
        let img = image(&code, &data, &[]);
        let (_, attribution) = pipeline(&img);
        assert_eq!(attribution.dominant_file(CODE_VA), Some("src/a.rs"));
    }

    #[test]
    fn the_same_fact_referenced_twice_is_recorded_once() {
        let mut data = Data::default();
        let site = data.site("src/main.rs", 9, 9);
        let mut code = Code::default();
        code.lea(site);
        code.lea(site + 8);
        code.lea(site);
        code.ret();
        let img = image(&code, &data, &[]);
        let (_, attribution) = pipeline(&img);
        assert_eq!(attribution.facts(CODE_VA).count(), 1);
    }

    #[test]
    fn the_per_function_fact_cap_is_enforced() {
        // One function referencing more distinct (file, line) facts than
        // the cap allows keeps exactly the cap.
        let count = MAX_FACTS_PER_FUNCTION + 1;
        let mut data = Data::default();
        let (ptr, len) = data.string(b"src/big.rs");
        let mut records = Vec::with_capacity(count);
        for i in 0..count {
            records.push(data.record(ptr, len, i as u32 + 1, 1));
        }

        let mut code = Code::default();
        for &target in &records {
            code.lea(target);
        }
        code.ret();
        let img = image(&code, &data, &[]);

        let sites = mine(&img);
        assert_eq!(sites.len(), count);
        let program = cfg::recover(&img).expect("x86-64 recovers");
        let attribution = attribute_sites(&img, &program, &sites);
        assert_eq!(attribution.facts(CODE_VA).count(), MAX_FACTS_PER_FUNCTION);
    }

    #[test]
    fn attribution_without_sites_or_a_decoder_is_empty() {
        // No sites: nothing to attribute, and no xref work attempted.
        let data = Data::default();
        let mut code = Code::default();
        code.ret();
        let img = image(&code, &data, &[]);
        let program = cfg::recover(&img).expect("x86-64 recovers");
        assert!(attribute(&img, &program).is_empty());
        assert!(attribute_sites(&img, &program, &[]).is_empty());

        // An architecture with no decoder degrades to empty, not an error.
        struct NoDecoder(FakeImage);
        impl Image for NoDecoder {
            fn arch(&self) -> Arch {
                Arch::Other
            }
            fn entry_points(&self) -> Vec<u64> {
                self.0.entry_points()
            }
            fn regions(&self) -> Vec<Region> {
                self.0.regions()
            }
            fn symbols(&self) -> Vec<Symbol> {
                self.0.symbols()
            }
            fn import_slots(&self) -> Vec<ImportSlot> {
                Vec::new()
            }
            fn va_to_offset(&self, va: u64) -> Option<usize> {
                self.0.va_to_offset(va)
            }
            fn bytes(&self) -> &[u8] {
                self.0.bytes()
            }
        }
        let mut data = Data::default();
        data.site("src/main.rs", 1, 1);
        let img = NoDecoder(data_image(&data));
        // Mining is architecture-blind: the site is still found...
        assert_eq!(mine(&img).len(), 1);
        // ...but there is no code to attribute it to.
        let program = cfg::Program {
            functions: BTreeMap::new(),
            call_graph: BTreeMap::new(),
            stats: cfg::Stats::default(),
        };
        assert!(attribute(&img, &program).is_empty());
    }

    // ---- Empty and non-Rust images ---------------------------------------

    #[test]
    fn a_non_rust_image_yields_nothing() {
        let elf = crate::elf::tests::synthetic_elf64();
        let image = load(&elf).expect("synthetic ELF loads");
        let program = cfg::recover(image.as_ref()).expect("x86-64 recovers");
        let sites = mine(image.as_ref());
        let attribution = attribute(image.as_ref(), &program);

        assert!(sites.is_empty());
        assert!(attribution.is_empty());
        assert_eq!(attribution.len(), 0);
        assert_eq!(
            render(&sites, &attribution),
            "no Rust panic metadata recovered\n"
        );
    }

    #[test]
    fn an_image_with_no_regions_yields_nothing() {
        let img = FakeImage {
            data: Vec::new(),
            regions: Vec::new(),
            entries: Vec::new(),
            symbols: Vec::new(),
        };
        assert!(mine(&img).is_empty());
    }

    // ---- Rendering -------------------------------------------------------

    #[test]
    fn the_report_is_golden_and_newline_terminated() {
        let mut data = Data::default();
        let a = data.site("src/main.rs", 10, 5);
        let b = data.site("src/main.rs", 20, 5);
        let c = data.site("src/lib.rs", 30, 7);

        let helper = CODE_VA + 0x100;
        let mut code = Code::default();
        code.lea(a);
        code.lea(b);
        code.ret();
        code.pad_to(helper);
        code.lea(c);
        code.ret();
        let img = image(&code, &data, &[("helper", helper)]);

        let (sites, attribution) = pipeline(&img);
        let expected = "\
Rust panic sites: 3

; -- files (2) --
       2  src/main.rs
       1  src/lib.rs

; -- functions (2) --
  0x0000000000001000  src/main.rs  (2 site(s))
  0x0000000000001100  src/lib.rs  (1 site(s))
";
        let text = render(&sites, &attribution);
        assert_eq!(text, expected);
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn the_file_histogram_is_capped() {
        let extra = 8;
        let files = MAX_HISTOGRAM_FILES + extra;
        let mut data = Data::default();
        for i in 0..files {
            data.site(&format!("src/f{i:03}.rs"), 1, 1);
        }
        let sites = mine(&data_image(&data));
        assert_eq!(sites.len(), files);

        let text = render(&sites, &Attribution::default());
        assert_eq!(
            text.lines().filter(|l| l.contains(".rs")).count(),
            MAX_HISTOGRAM_FILES
        );
        assert!(
            text.contains(&format!("... {extra} more file(s)")),
            "{text}"
        );
        assert!(text.contains(&format!("; -- files ({files}) --")), "{text}");
        // Nothing attributed: the functions section is omitted entirely.
        assert!(!text.contains("functions"), "{text}");
        assert!(text.ends_with('\n'));
    }

    // ---- Determinism and hostile input -----------------------------------

    #[test]
    fn recovery_is_deterministic() {
        let mut data = Data::default();
        let a = data.site("src/z.rs", 3, 1);
        let b = data.site("src/a.rs", 4, 1);
        let mut code = Code::default();
        code.lea(a);
        code.lea(b + 16);
        code.ret();
        let img = image(&code, &data, &[]);

        let first = pipeline(&img);
        let second = pipeline(&img);
        assert_eq!(first.0, second.0);
        assert_eq!(first.1, second.1);
        assert_eq!(render(&first.0, &first.1), render(&second.0, &second.1));
    }

    #[test]
    fn hostile_bytes_never_panic() {
        // A data region of nothing but 0xFF: every field is maximal.
        let data = Data {
            bytes: vec![0xFFu8; 4096],
        };
        assert!(mine(&data_image(&data)).is_empty());

        // Pseudo-random data, fixed seed: only bounded, panic-free output
        // is asserted.
        let mut state: u64 = 0x5eed_1234_abcd_ef01;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut total = 0usize;
        for _ in 0..64 {
            let data = Data {
                bytes: (0..1024).map(|_| next() as u8).collect(),
            };
            let mut img = data_image(&data);
            // Also stress the region bookkeeping itself.
            img.regions[1].size = u64::from(next()) % 2048;
            total = total.saturating_add(mine(&img).len());
        }
        assert!(total < 10_000, "unbounded mining: {total}");

        // Random *code* over a real record set: recovery, attribution and
        // rendering must all survive.
        let mut data = Data::default();
        data.site("src/main.rs", 1, 1);
        for _ in 0..32 {
            let code = Code {
                bytes: (0..256).map(|_| next() as u8).collect(),
            };
            let img = image(&code, &data, &[]);
            let sites = mine(&img);
            if let Ok(program) = cfg::recover(&img) {
                let attribution = attribute_sites(&img, &program, &sites);
                let _ = render(&sites, &attribution);
            }
        }
    }

    #[test]
    fn addresses_near_the_top_of_the_space_do_not_wrap() {
        // A region ending at u64::MAX: the scan must terminate, not wrap.
        let mut data = Data::default();
        data.site("src/main.rs", 1, 1);
        let mut img = data_image(&data);
        img.regions.push(Region {
            name: "high".into(),
            va: u64::MAX - 31,
            size: 32,
            perms: Perms {
                r: true,
                w: false,
                x: false,
            },
        });
        // The high region is unmapped in the buffer, so it contributes
        // nothing; the real record is still found.
        assert_eq!(mine(&img).len(), 1);
        assert_eq!(align_up(u64::MAX), u64::MAX);
        assert_eq!(align_up(1), 8);
        assert_eq!(align_up(8), 8);
    }
}
