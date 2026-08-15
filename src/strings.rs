//! String extraction with encoding detection.
//!
//! Written once against the trait layer: outside its tests this module
//! names only [`crate::model`] — never a container format or an
//! instruction set.
//!
//! # Method
//!
//! [`extract`] sweeps the raw file buffer ([`Image::bytes`]) — not the
//! mapped image — so strings in headers, debug directories, and other
//! unmapped file areas are found too; each hit is then *attributed* to a
//! region when its file offset is backed by one.
//!
//! Two scanners run over the buffer, their hits merged into one stream
//! in ascending file-offset order:
//!
//! - **ASCII** ([`Encoding::Ascii`]): a maximal run of printable bytes,
//!   `0x20..=0x7E`. Tab, newline, and every other control byte terminate
//!   a run rather than extending it (v1 keeps the alphabet strict, which
//!   is what makes the two scanners non-overlapping — see below).
//! - **UTF-16LE** ([`Encoding::Utf16Le`]): a maximal run of
//!   `(printable ASCII byte, 0x00)` pairs — the shape Windows binaries
//!   store their wide literals in. Runs are found at any file offset,
//!   odd ones included, since a buffer offset need not share the parity
//!   of the VA the loader maps it at.
//!
//! Both scanners require at least [`Config::min_len`] *characters*
//! (not bytes), so a UTF-16LE hit spans `2 * len` bytes.
//!
//! ## No double counting
//!
//! `0x00` is not printable, so UTF-16LE text yields ASCII runs of length
//! one and can never be re-reported as an ASCII string at any
//! `min_len >= 2`. As a belt-and-braces rule the merge also drops any
//! candidate whose byte span is *contained* in the span already emitted,
//! which covers the degenerate `min_len == 1` configuration.
//!
//! # Offset to VA
//!
//! [`Image`] maps VA to offset, not the reverse, so the inverse is built
//! per region: probe [`Image::va_to_offset`] at the region's first byte
//! to get its file start, then binary-search the longest prefix of the
//! region that keeps mapping linearly
//! (`offset = file_start + (va - region.va)`). That prefix is the
//! region's *file-backed extent*, and inside it
//! `va = region.va + (offset - file_start)`.
//!
//! This is why a region's file offset may differ freely from its VA (PE
//! `PointerToRawData` vs `VirtualAddress`, ELF `p_offset` vs `p_vaddr`,
//! Mach-O `fileoff` vs `vmaddr`): only the loader's own mapping is ever
//! consulted. Zero-fill tails (`.bss`, a `memsz > filesz` segment) end
//! the extent, since their VAs map to no file offset; a region with no
//! file backing at all contributes no extent. The extent is additionally
//! clipped to the region's reported [`Region::size`] and to the end of
//! the buffer.
//!
//! Regions may overlap (ELF sections nested in segments, a Mach-O
//! segment covering the headers). An offset covered by several extents
//! is attributed to the one earliest in [`Image::regions`] order, and an
//! offset covered by none yields `va: None`, `region: None` — unless
//! [`Config::require_mapped`] drops it instead.
//!
//! # Resource caps
//!
//! A crafted file must not exhaust memory, so [`Config`] caps both the
//! number of results ([`Config::max_strings`]) and the length of each
//! ([`Config::max_len`]): peak allocation is bounded by
//! `max_strings * max_len` bytes of text regardless of input size. A run
//! longer than `max_len` is **split**, not truncated: it is emitted as
//! consecutive chunks of `max_len` characters each (the last one
//! shorter), every chunk carrying its own offset, VA, and region, so no
//! bytes are lost. Chunks are emitted even when shorter than `min_len` —
//! the minimum applies to the run, the maximum to each emitted piece.
//! Reaching `max_strings` stops the scan cleanly and returns what was
//! found so far.
//!
//! # Determinism
//!
//! Results are in ascending [`FoundString::file_offset`] order; the same
//! image and config always yield exactly the same vector. (Two hits can
//! share an offset only when `min_len == 1`, where a UTF-16LE hit is
//! ordered before the one-character ASCII hit at the same byte.)
//!
//! # Deferred
//!
//! v1 detects ASCII and UTF-16LE only. UTF-8 sequences with non-ASCII
//! code points, UTF-16BE, UTF-32, and legacy code pages are not
//! detected; their ASCII-range substrings still surface as ASCII runs.
//! Language/entropy heuristics for filtering junk are also out of scope.

use crate::model::{Image, Region};

/// How a [`FoundString`]'s bytes encode its text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// One byte per character, all in `0x20..=0x7E`.
    Ascii,
    /// Two bytes per character, little-endian, every character in
    /// `0x20..=0x7E` (so every second byte is `0x00`).
    Utf16Le,
}

/// One extracted string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundString {
    /// Virtual address of the first byte, when the offset lies in a
    /// region's file-backed extent; `None` for unmapped file areas
    /// (headers, symbol tables, overlays).
    pub va: Option<u64>,
    /// Offset of the first byte into [`Image::bytes`].
    pub file_offset: usize,
    /// The encoding this run was detected as.
    pub encoding: Encoding,
    /// The decoded text: exactly the characters of this run, no
    /// terminator. At most [`Config::max_len`] characters.
    pub text: String,
    /// Name of the region owning `file_offset`, paired with `va`.
    pub region: Option<String>,
}

/// Scan settings and resource caps for [`extract`]. See the module docs
/// for what hitting each cap does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Minimum characters for a run to be reported. Values below 1 are
    /// treated as 1 (a zero minimum would report every byte).
    pub min_len: usize,
    /// Maximum total strings returned; the scan stops on reaching it.
    pub max_strings: usize,
    /// Maximum characters per returned string. Longer runs are split
    /// into consecutive chunks of this size. Values below 1 are treated
    /// as 1.
    pub max_len: usize,
    /// Scan for UTF-16LE runs in addition to ASCII.
    pub scan_utf16: bool,
    /// Report only strings whose offset falls in a region's file-backed
    /// extent (i.e. drop every hit that would have `va: None`).
    pub require_mapped: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            min_len: 4,
            max_strings: 100_000,
            max_len: 4096,
            scan_utf16: true,
            require_mapped: false,
        }
    }
}

/// Extract printable strings from `image`'s file buffer, in ascending
/// file-offset order. Never panics and never allocates unboundedly: see
/// the module docs for the caps and for how offsets become VAs.
pub fn extract(image: &dyn Image, config: &Config) -> Vec<FoundString> {
    let data = image.bytes();
    let min_len = config.min_len.max(1);
    let max_len = config.max_len.max(1);
    let mut out = Vec::new();
    if config.max_strings == 0 || data.is_empty() {
        return out;
    }
    let map = RegionMap::build(image);

    // Two independent scanners advanced by their own cursor; the lower
    // hit is emitted each round, so the merged stream is sorted and the
    // cap applies to it as a whole (only one run per scanner is ever
    // buffered).
    let mut ascii_hit = next_ascii_run(data, 0, min_len);
    let mut utf16_hit = if config.scan_utf16 {
        next_utf16_run(data, 0, min_len)
    } else {
        None
    };
    // End of the most recently emitted byte span; candidates ending
    // inside it are contained in it and dropped.
    let mut covered_end = 0usize;

    while out.len() < config.max_strings {
        // Ties (possible only at min_len == 1) go to UTF-16LE, whose
        // span then contains the one-character ASCII hit.
        let (start, chars, encoding) = match (ascii_hit, utf16_hit) {
            (None, None) => break,
            (Some((s, n)), None) => (s, n, Encoding::Ascii),
            (None, Some((s, n))) => (s, n, Encoding::Utf16Le),
            (Some((a, an)), Some((u, un))) => {
                if u <= a {
                    (u, un, Encoding::Utf16Le)
                } else {
                    (a, an, Encoding::Ascii)
                }
            }
        };
        let stride = match encoding {
            Encoding::Ascii => 1usize,
            Encoding::Utf16Le => 2usize,
        };
        let end = start.saturating_add(chars.saturating_mul(stride));

        // Advance the scanner that produced this hit before any early
        // `continue`, so the loop always makes progress.
        match encoding {
            Encoding::Ascii => ascii_hit = next_ascii_run(data, end, min_len),
            Encoding::Utf16Le => utf16_hit = next_utf16_run(data, end, min_len),
        }
        if end <= covered_end {
            continue; // fully inside the previous hit
        }
        covered_end = covered_end.max(end);

        let mut done = 0usize;
        while done < chars {
            if out.len() >= config.max_strings {
                break;
            }
            let take = (chars - done).min(max_len);
            let offset = start.saturating_add(done.saturating_mul(stride));
            let text = decode(data, offset, take, stride);
            let mapped = map.find(offset);
            if !(config.require_mapped && mapped.is_none()) {
                out.push(FoundString {
                    va: mapped.map(|(va, _)| va),
                    file_offset: offset,
                    encoding,
                    text,
                    region: mapped.map(|(_, name)| name.to_string()),
                });
            }
            done += take;
        }
    }
    out
}

/// Printable ASCII, the alphabet of both scanners.
fn is_printable(b: u8) -> bool {
    (0x20..=0x7E).contains(&b)
}

/// Decode `chars` characters starting at `offset`, one every `stride`
/// bytes. Every byte read is printable ASCII by construction, so the
/// `as char` cast is exact.
fn decode(data: &[u8], offset: usize, chars: usize, stride: usize) -> String {
    let mut s = String::with_capacity(chars);
    for i in 0..chars {
        let Some(idx) = offset.checked_add(i.saturating_mul(stride)) else {
            break;
        };
        match data.get(idx) {
            Some(&b) => s.push(b as char),
            None => break,
        }
    }
    s
}

/// The first maximal printable run of at least `min_len` bytes starting
/// at or after `from`, as `(offset, length)`.
fn next_ascii_run(data: &[u8], from: usize, min_len: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i < data.len() {
        if !is_printable(data[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < data.len() && is_printable(data[i]) {
            i += 1;
        }
        if i - start >= min_len {
            return Some((start, i - start));
        }
        // Shorter than the minimum: the whole run is skipped, and no
        // longer run can start inside it.
    }
    None
}

/// The first maximal run of at least `min_len` `(printable, 0x00)` pairs
/// starting at or after `from`, as `(offset, character count)`.
fn next_utf16_run(data: &[u8], from: usize, min_len: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 1 < data.len() {
        if !(is_printable(data[i]) && data[i + 1] == 0) {
            i += 1;
            continue;
        }
        let start = i;
        let mut pairs = 0usize;
        while i + 1 < data.len() && is_printable(data[i]) && data[i + 1] == 0 {
            pairs += 1;
            i += 2;
        }
        if pairs >= min_len {
            return Some((start, pairs));
        }
        // A shorter run cannot hide a longer one: a run starting at an
        // even offset inside it is a suffix of it, and one starting at
        // an odd offset would begin on a 0x00 byte.
    }
    None
}

/// One region's file-backed extent: `[start, end)` in the buffer, whose
/// first byte is at virtual address `va`.
#[derive(Debug, Clone, Copy)]
struct Extent {
    start: usize,
    end: usize,
    va: u64,
    /// Index into [`RegionMap::regions`]; also the tie-break for
    /// overlapping extents (earliest region wins).
    region: usize,
}

/// The offset-to-(VA, region) inverse of [`Image::va_to_offset`], built
/// by probing the image. See the module docs.
struct RegionMap {
    regions: Vec<Region>,
    /// Sorted by `(start, end, region)`.
    extents: Vec<Extent>,
    /// `max_end[i]` is the largest `end` among `extents[..=i]`, which
    /// bounds how far back a stabbing query has to walk.
    max_end: Vec<usize>,
}

impl RegionMap {
    fn build(image: &dyn Image) -> RegionMap {
        let regions = image.regions();
        let file_len = image.bytes().len();
        let mut extents = Vec::new();
        for (i, r) in regions.iter().enumerate() {
            if r.size == 0 {
                continue;
            }
            let Some(start) = image.va_to_offset(r.va) else {
                continue; // no file backing at the region's first byte
            };
            if start >= file_len {
                continue;
            }
            // Longest prefix that keeps mapping linearly, clipped to the
            // region size and the buffer. Probing only the last byte of
            // a candidate prefix assumes the file-backed part is a
            // prefix — true of every format modeled here, where a
            // region is one contiguous mapping optionally followed by a
            // zero-fill tail.
            let cap = r.size.min((file_len - start) as u64);
            let mut lo = 1u64; // the first byte is known to map
            let mut hi = cap;
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                if maps_linearly(image, r.va, start, mid) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            extents.push(Extent {
                start,
                end: start.saturating_add(lo as usize),
                va: r.va,
                region: i,
            });
        }
        extents.sort_by_key(|e| (e.start, e.end, e.region));
        let mut max_end = Vec::with_capacity(extents.len());
        let mut running = 0usize;
        for e in &extents {
            running = running.max(e.end);
            max_end.push(running);
        }
        RegionMap {
            regions,
            extents,
            max_end,
        }
    }

    /// The VA and region name of `offset`, or `None` when no region's
    /// file-backed extent covers it.
    fn find(&self, offset: usize) -> Option<(u64, &str)> {
        let idx = self.extents.partition_point(|e| e.start <= offset);
        let mut best: Option<&Extent> = None;
        for j in (0..idx).rev() {
            if self.max_end[j] <= offset {
                break; // nothing at or before j can still cover offset
            }
            let e = &self.extents[j];
            if offset < e.end {
                match best {
                    Some(b) if b.region <= e.region => {}
                    _ => best = Some(e),
                }
            }
        }
        let e = best?;
        let va = e.va.wrapping_add((offset - e.start) as u64);
        Some((va, self.regions[e.region].name.as_str()))
    }
}

/// Does `va + (k - 1)` map to `start + (k - 1)`, i.e. is the region's
/// first `k` bytes one linear file-backed run?
fn maps_linearly(image: &dyn Image, va: u64, start: usize, k: u64) -> bool {
    let step = k.saturating_sub(1);
    let (Some(last_va), Ok(step_usize)) = (va.checked_add(step), usize::try_from(step)) else {
        return false;
    };
    let Some(last_off) = start.checked_add(step_usize) else {
        return false;
    };
    image.va_to_offset(last_va) == Some(last_off)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::tests::synthetic_elf64;
    use crate::macho::tests::synthetic_macho64;
    use crate::model::{Arch, ImportSlot, Perms, Symbol, load};
    use crate::model::{PeImage, Region};
    use crate::pe::tests::synthetic_pe64;

    const PE_BASE: u64 = 0x1_4000_0000;
    /// `.text`: RVA 0x1000 (VA `PE_BASE + 0x1000`) at file offset 0x200,
    /// virtual size 0x100 — so the file-backed extent is [0x200, 0x300),
    /// and every planted string below sits inside it.
    const PE_TEXT_VA: u64 = PE_BASE + 0x1000;
    /// File offset of the section table's ".text" name in the fixture
    /// (optional header at 0x98, section table 0xF0 bytes later).
    const PE_SECNAME_OFF: usize = 0x188;

    /// A PE fixture with `ascii` planted at file offset `a_off` and
    /// `wide` planted as UTF-16LE at `w_off`, both inside `.text`. The
    /// base fixture's import data directory is cleared first: it points
    /// into `.text`, and planting text there would otherwise make the
    /// eager import parse fail.
    fn pe_with_strings(a_off: usize, ascii: &str, w_off: usize, wide: &str) -> Vec<u8> {
        let mut img = synthetic_pe64();
        let dirs = 0x80 + 4 + 20 + 112;
        img[dirs + 8..dirs + 16].fill(0);
        img[a_off..a_off + ascii.len()].copy_from_slice(ascii.as_bytes());
        for (i, b) in wide.bytes().enumerate() {
            img[w_off + i * 2] = b;
        }
        img
    }

    /// A hand-built [`Image`] for edge cases no loader will produce:
    /// empty buffers, overlapping regions, regions that map nowhere,
    /// regions with a zero-fill tail.
    struct FakeImage {
        data: Vec<u8>,
        /// `(name, va, memory size, file start, file-backed size)`.
        regions: Vec<(String, u64, u64, usize, u64)>,
    }

    impl FakeImage {
        /// Regions given as `(name, va, size, file start)`, fully
        /// file-backed.
        fn new(data: Vec<u8>, regions: &[(&str, u64, u64, usize)]) -> FakeImage {
            FakeImage {
                data,
                regions: regions
                    .iter()
                    .map(|(n, va, size, off)| (n.to_string(), *va, *size, *off, *size))
                    .collect(),
            }
        }

        /// One region of `size` bytes at `va` whose first `filesz` bytes
        /// live at `file_start` and whose tail is zero-fill (`.bss`-like).
        fn with_zero_fill(
            data: Vec<u8>,
            name: &str,
            va: u64,
            size: u64,
            file_start: usize,
            filesz: u64,
        ) -> FakeImage {
            FakeImage {
                data,
                regions: vec![(name.to_string(), va, size, file_start, filesz)],
            }
        }
    }

    impl Image for FakeImage {
        fn arch(&self) -> Arch {
            Arch::Other
        }
        fn entry_points(&self) -> Vec<u64> {
            Vec::new()
        }
        fn regions(&self) -> Vec<Region> {
            self.regions
                .iter()
                .map(|(name, va, size, _, _)| Region {
                    name: name.clone(),
                    va: *va,
                    size: *size,
                    perms: Perms {
                        r: true,
                        w: false,
                        x: false,
                    },
                })
                .collect()
        }
        fn symbols(&self) -> Vec<Symbol> {
            Vec::new()
        }
        fn import_slots(&self) -> Vec<ImportSlot> {
            Vec::new()
        }
        fn va_to_offset(&self, va: u64) -> Option<usize> {
            // First region whose *file-backed* part contains va wins,
            // as in the real loaders.
            let (_, rva, _, start, _) = self
                .regions
                .iter()
                .find(|(_, rva, _, _, filesz)| va >= *rva && va - *rva < *filesz)?;
            let off = start.checked_add(usize::try_from(va - rva).ok()?)?;
            (off < self.data.len()).then_some(off)
        }
        fn bytes(&self) -> &[u8] {
            &self.data
        }
    }

    /// Deterministic pseudo-random bytes (xorshift64*), so the
    /// robustness tests are reproducible without a dependency.
    fn pseudo_random(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 33) as u8
            })
            .collect()
    }

    #[test]
    fn finds_ascii_and_utf16_with_offset_va_encoding_and_region() {
        let img = pe_with_strings(0x210, "HelloWorld", 0x240, "Windows");
        let pe = PeImage::parse(&img).unwrap();
        let found = extract(&pe, &Config::default());

        assert_eq!(
            found,
            [
                // The section table's own name: real text, but the PE
                // headers are in no region, so it maps to no VA.
                FoundString {
                    va: None,
                    file_offset: PE_SECNAME_OFF,
                    encoding: Encoding::Ascii,
                    text: ".text".into(),
                    region: None,
                },
                FoundString {
                    va: Some(PE_TEXT_VA + 0x10),
                    file_offset: 0x210,
                    encoding: Encoding::Ascii,
                    text: "HelloWorld".into(),
                    region: Some(".text".into()),
                },
                FoundString {
                    va: Some(PE_TEXT_VA + 0x40),
                    file_offset: 0x240,
                    encoding: Encoding::Utf16Le,
                    text: "Windows".into(),
                    region: Some(".text".into()),
                },
            ]
        );
        // The VA of every mapped hit inverts back to its own offset.
        for s in &found {
            if let Some(va) = s.va {
                assert_eq!(pe.va_to_offset(va), Some(s.file_offset));
            }
        }
    }

    #[test]
    fn utf16_is_not_also_reported_as_ascii() {
        let img = pe_with_strings(0x210, "abcd", 0x240, "WideString");
        let pe = PeImage::parse(&img).unwrap();
        let found = extract(&pe, &Config::default());

        let wide: Vec<&FoundString> = found.iter().filter(|s| s.file_offset >= 0x240).collect();
        assert_eq!(wide.len(), 1, "{found:#?}");
        assert_eq!(wide[0].encoding, Encoding::Utf16Le);
        assert_eq!(wide[0].text, "WideString");

        // With the UTF-16 scanner off, the wide string leaves only
        // one-character ASCII runs behind — nothing at the default
        // minimum length.
        let cfg = Config {
            scan_utf16: false,
            ..Config::default()
        };
        let found = extract(&pe, &cfg);
        assert!(found.iter().all(|s| s.encoding == Encoding::Ascii));
        assert!(found.iter().all(|s| s.file_offset < 0x240), "{found:#?}");
    }

    #[test]
    fn min_len_filters_short_runs() {
        // "abc" is three characters; "ab" is two wide characters.
        let img = pe_with_strings(0x210, "abc", 0x240, "ab");
        let pe = PeImage::parse(&img).unwrap();

        let found = extract(&pe, &Config::default());
        assert!(
            found.iter().all(|s| s.text != "abc" && s.text != "ab"),
            "{found:#?}"
        );

        let cfg = Config {
            min_len: 2,
            ..Config::default()
        };
        let found = extract(&pe, &cfg);
        assert!(found.iter().any(|s| s.text == "abc"));
        assert!(
            found
                .iter()
                .any(|s| s.text == "ab" && s.encoding == Encoding::Utf16Le)
        );

        // min_len 0 is clamped to 1, never to "every byte is a string".
        let cfg = Config {
            min_len: 0,
            ..Config::default()
        };
        let found = extract(&pe, &cfg);
        assert!(found.iter().all(|s| !s.text.is_empty()));
    }

    #[test]
    fn max_len_splits_runs_into_chunks() {
        let img = pe_with_strings(0x210, "ABCDEFGHIJ", 0x240, "abcdef");
        let pe = PeImage::parse(&img).unwrap();
        let cfg = Config {
            max_len: 4,
            ..Config::default()
        };
        let found = extract(&pe, &cfg);

        // The ASCII run splits into 4 + 4 + 2 characters, each chunk
        // carrying its own offset and VA; the trailing chunk is kept
        // even though it is shorter than min_len.
        let ascii: Vec<&FoundString> = found
            .iter()
            .filter(|s| (0x210..0x240).contains(&s.file_offset))
            .collect();
        assert_eq!(
            ascii
                .iter()
                .map(|s| (s.file_offset, s.text.as_str(), s.va))
                .collect::<Vec<_>>(),
            [
                (0x210, "ABCD", Some(PE_TEXT_VA + 0x10)),
                (0x214, "EFGH", Some(PE_TEXT_VA + 0x14)),
                (0x218, "IJ", Some(PE_TEXT_VA + 0x18)),
            ]
        );
        // UTF-16 chunks advance two bytes per character.
        let wide: Vec<(usize, &str)> = found
            .iter()
            .filter(|s| s.encoding == Encoding::Utf16Le)
            .map(|s| (s.file_offset, s.text.as_str()))
            .collect();
        assert_eq!(wide, [(0x240, "abcd"), (0x248, "ef")]);

        // No text ever exceeds max_len, even clamped to 1.
        for max_len in [1usize, 2, 3, 7] {
            let cfg = Config {
                max_len,
                ..Config::default()
            };
            assert!(
                extract(&pe, &cfg).iter().all(|s| s.text.chars().count() <= max_len),
                "max_len {max_len}"
            );
        }
    }

    #[test]
    fn max_strings_caps_output_cleanly() {
        let img = pe_with_strings(0x210, "ABCDEFGHIJKLMNOP", 0x240, "widestring");
        let pe = PeImage::parse(&img).unwrap();
        let full = extract(&pe, &Config::default());
        assert!(full.len() >= 3, "{full:#?}");

        for cap in 0..=full.len() {
            let cfg = Config {
                max_strings: cap,
                ..Config::default()
            };
            let capped = extract(&pe, &cfg);
            assert_eq!(capped.len(), cap);
            // A cap truncates the full result; it never reorders or
            // rewrites it.
            assert_eq!(capped, full[..cap]);
        }

        // The cap counts chunks, not runs: one long run alone fills it.
        let cfg = Config {
            max_len: 2,
            max_strings: 3,
            ..Config::default()
        };
        assert_eq!(extract(&pe, &cfg).len(), 3);
    }

    #[test]
    fn require_mapped_drops_unmapped_hits() {
        let img = pe_with_strings(0x210, "HelloWorld", 0x240, "Windows");
        let pe = PeImage::parse(&img).unwrap();
        let cfg = Config {
            require_mapped: true,
            ..Config::default()
        };
        let found = extract(&pe, &cfg);

        assert!(!found.is_empty());
        assert!(found.iter().all(|s| s.va.is_some() && s.region.is_some()));
        // The header hit is gone; the two planted ones remain.
        assert_eq!(
            found.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(),
            ["HelloWorld", "Windows"]
        );
    }

    #[test]
    fn attribution_uses_the_file_backed_extent_only() {
        // The region claims 0x1000 bytes of memory at VA 0x1000 but only
        // its first 0x18 bytes are file-backed (at offset 0x10): the
        // zero-fill tail must not swallow later file offsets.
        let mut data = vec![0u8; 0x40];
        data[0x10..0x18].copy_from_slice(b"mapped!!");
        data[0x28..0x30].copy_from_slice(b"unmapped");
        let img = FakeImage::with_zero_fill(data, "data", 0x1000, 0x1000, 0x10, 0x18);
        let found = extract(&img, &Config::default());

        assert_eq!(
            found
                .iter()
                .map(|s| (s.file_offset, s.text.as_str(), s.va, s.region.as_deref()))
                .collect::<Vec<_>>(),
            [
                (0x10, "mapped!!", Some(0x1000), Some("data")),
                (0x28, "unmapped", None, None),
            ]
        );
    }

    #[test]
    fn overlapping_regions_attribute_to_the_first() {
        let mut data = vec![0u8; 0x40];
        data[0x20..0x28].copy_from_slice(b"overlap!");
        // Two regions cover offset 0x20; `regions()` order decides.
        let img = FakeImage::new(
            data.clone(),
            &[("outer", 0x1000, 0x40, 0), ("inner", 0x2000, 0x10, 0x20)],
        );
        let found = extract(&img, &Config::default());
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].region.as_deref(), Some("outer"));
        assert_eq!(found[0].va, Some(0x1020));

        let img = FakeImage::new(
            data,
            &[("inner", 0x2000, 0x10, 0x20), ("outer", 0x1000, 0x40, 0)],
        );
        let found = extract(&img, &Config::default());
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].region.as_deref(), Some("inner"));
        assert_eq!(found[0].va, Some(0x2000));
    }

    #[test]
    fn hostile_and_degenerate_images_do_not_panic() {
        let cases: Vec<FakeImage> = vec![
            // Empty buffer, with and without regions.
            FakeImage::new(Vec::new(), &[]),
            FakeImage::new(Vec::new(), &[("r", 0, u64::MAX, 0)]),
            // Text with no regions at all.
            FakeImage::new(b"a string with no regions".to_vec(), &[]),
            // Zero-size region, region mapping past the end, region at
            // the top of the address space (offset arithmetic must not
            // overflow), and a region whose VA maps nowhere.
            FakeImage::new(
                b"stringstringstring".to_vec(),
                &[
                    ("zero", 0x1000, 0, 0),
                    ("past", 0x2000, 0x10, 0x1000),
                    ("top", u64::MAX - 4, 8, 0),
                    ("gap", 0x3000, 0x10, usize::MAX),
                ],
            ),
            // Wildly overlapping regions over pseudo-random bytes.
            FakeImage::new(
                pseudo_random(4096, 0xDEAD_BEEF),
                &(0..64)
                    .map(|i| ("r", i * 3, 0x400, i as usize * 7))
                    .collect::<Vec<_>>(),
            ),
        ];
        let configs = [
            Config::default(),
            Config {
                min_len: 1,
                max_strings: 32,
                max_len: 1,
                scan_utf16: true,
                require_mapped: false,
            },
            Config {
                min_len: usize::MAX,
                max_strings: usize::MAX,
                max_len: usize::MAX,
                scan_utf16: false,
                require_mapped: true,
            },
            Config {
                min_len: 0,
                max_strings: 0,
                max_len: 0,
                scan_utf16: true,
                require_mapped: true,
            },
        ];
        for img in &cases {
            for cfg in &configs {
                let found = extract(img, cfg);
                assert!(found.len() <= cfg.max_strings);
                assert!(found.iter().all(|s| s.text.chars().count() <= cfg.max_len.max(1)));
                assert!(found.iter().all(|s| s.file_offset < img.bytes().len().max(1)));
            }
        }
    }

    #[test]
    fn pseudo_random_input_stays_bounded_and_sorted() {
        for seed in 1..8u64 {
            let data = pseudo_random(1 << 16, seed.wrapping_mul(0x9E37_79B9));
            let img = FakeImage::new(data, &[("all", 0x40_0000, 1 << 16, 0)]);
            let cfg = Config {
                min_len: 4,
                max_strings: 64,
                ..Config::default()
            };
            let found = extract(&img, &cfg);
            assert!(found.len() <= 64);
            assert!(found.windows(2).all(|w| w[0].file_offset <= w[1].file_offset));
            // Every reported byte really is printable ASCII.
            assert!(
                found
                    .iter()
                    .all(|s| s.text.bytes().all(|b| (0x20..=0x7E).contains(&b)))
            );
        }
    }

    #[test]
    fn truncated_images_are_an_error_not_a_panic() {
        // Loading a truncated image usually fails; whenever it does
        // load, extraction over the stump must still be well behaved.
        for full in [synthetic_pe64(), synthetic_elf64(), synthetic_macho64()] {
            for len in [0, 1, 0x10, 0x40, 0x80, 0x100, 0x180, full.len() - 1] {
                if let Ok(img) = load(&full[..len]) {
                    let found = extract(img.as_ref(), &Config::default());
                    assert!(found.iter().all(|s| s.file_offset < len));
                }
            }
        }
    }

    #[test]
    fn extraction_is_deterministic_and_sorted_for_every_format() {
        let mut pe = pe_with_strings(0x210, "PeString", 0x240, "PeWide");
        // Also plant a wide string in the (unmapped) header area.
        for (i, b) in "Header".bytes().enumerate() {
            pe[0x120 + i * 2] = b;
        }
        for buf in [pe, synthetic_elf64(), synthetic_macho64()] {
            let img = load(&buf).unwrap();
            let a = extract(img.as_ref(), &Config::default());
            let b = extract(img.as_ref(), &Config::default());
            assert_eq!(a, b);
            assert!(!a.is_empty());
            assert!(a.windows(2).all(|w| w[0].file_offset <= w[1].file_offset));
            // Whenever a VA was assigned, it inverts back to the offset.
            for s in &a {
                if let Some(va) = s.va {
                    assert_eq!(img.va_to_offset(va), Some(s.file_offset));
                    assert!(s.region.is_some());
                }
            }
        }
    }

    #[test]
    fn format_blind_extraction_finds_real_payloads() {
        // ELF: the PT_INTERP path, which lives outside every allocated
        // section and so is found but unmapped.
        let elf_buf = synthetic_elf64();
        let elf = load(&elf_buf).unwrap();
        let found = extract(elf.as_ref(), &Config::default());
        let interp = found
            .iter()
            .find(|s| s.text == "/lib64/ld-linux-x86-64.so.2")
            .unwrap_or_else(|| panic!("interp not extracted: {found:#?}"));
        assert_eq!(interp.encoding, Encoding::Ascii);
        assert_eq!(interp.file_offset, 0xB0);
        assert_eq!(interp.va, None);
        // ...and the section-header string table, likewise unmapped.
        assert!(found.iter().any(|s| s.text.contains(".shstrtab")));

        // Mach-O: segment and symbol names, with __TEXT covering the
        // headers, so those hits do carry a VA.
        let mach_buf = synthetic_macho64();
        let mach = load(&mach_buf).unwrap();
        let found = extract(mach.as_ref(), &Config::default());
        let text = found
            .iter()
            .find(|s| s.text.starts_with("__TEXT"))
            .unwrap_or_else(|| panic!("__TEXT not extracted: {found:#?}"));
        assert_eq!(text.region.as_deref(), Some("__TEXT"));
        assert_eq!(text.va, Some(0x1_0000_0000 + text.file_offset as u64));
    }

    /// Smoke-test against real system binaries when present (a no-op
    /// where they are absent). Guards the invariants on megabyte-scale
    /// real input, which no synthetic fixture exercises.
    #[test]
    fn smoke_extracts_real_binaries_if_present() {
        for path in ["/bin/ls", "/bin/sh", "/usr/lib/dyld"] {
            let Ok(file) = std::fs::read(path) else {
                continue;
            };
            // macOS system binaries are fat: pick a slice, as `load`'s
            // docs prescribe.
            let data = match crate::macho::FatFile::parse(&file) {
                Ok(fat) => match fat.arches().first().and_then(|a| a.slice(&file).ok()) {
                    Some(slice) => slice.to_vec(),
                    None => continue,
                },
                Err(_) => file,
            };
            let Ok(img) = load(&data) else { continue };
            let cfg = Config::default();
            let found = extract(img.as_ref(), &cfg);
            assert!(!found.is_empty(), "{path}");
            assert!(found.len() <= cfg.max_strings, "{path}");
            assert!(
                found.windows(2).all(|w| w[0].file_offset <= w[1].file_offset),
                "{path}"
            );
            for s in &found {
                assert!(s.file_offset < data.len(), "{path}");
                assert!(s.text.chars().count() <= cfg.max_len, "{path}");
                assert_eq!(s.va.is_some(), s.region.is_some(), "{path}");
                if let Some(va) = s.va {
                    assert_eq!(img.va_to_offset(va), Some(s.file_offset), "{path}");
                }
            }
            // Real images always carry mapped text (section names,
            // format strings, linker paths).
            assert!(found.iter().any(|s| s.va.is_some()), "{path}");
        }
    }

    #[test]
    fn ascii_run_boundaries_are_exact() {
        // Runs stop at control bytes (tab and newline included) and at
        // the buffer's end; a run ending at EOF is still reported.
        let data = b"\tabcd\nefgh\0ijklmnop".to_vec();
        let img = FakeImage::new(data, &[]);
        let found = extract(&img, &Config::default());
        assert_eq!(
            found
                .iter()
                .map(|s| (s.file_offset, s.text.as_str()))
                .collect::<Vec<_>>(),
            [(1, "abcd"), (6, "efgh"), (11, "ijklmnop")]
        );
    }

    #[test]
    fn utf16_runs_are_found_at_odd_offsets_too() {
        // A wide string starting at an odd file offset (the leading
        // 0x00 belongs to a preceding field).
        let mut data = vec![0u8; 32];
        for (i, b) in "Odd!".bytes().enumerate() {
            data[1 + i * 2] = b;
        }
        let img = FakeImage::new(data, &[]);
        let found = extract(&img, &Config::default());
        assert_eq!(found.len(), 1, "{found:#?}");
        assert_eq!(found[0].file_offset, 1);
        assert_eq!(found[0].encoding, Encoding::Utf16Le);
        assert_eq!(found[0].text, "Odd!");
    }
}
