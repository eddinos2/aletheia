//! Go `gopclntab` function-table recovery.
//!
//! A Go binary carries its own symbol table even when stripped of ELF /
//! Mach-O / PE symbols: the runtime's *program-counter line table*
//! (`pclntab`) maps every function's entry PC to its metadata, and that
//! metadata names the function. Recovering `(entry VA, name)` pairs from it
//! is the single richest source of symbols in an otherwise stripped Go
//! image — the same table `go tool objdump` and Ghidra's Go analyzers read.
//!
//! This module implements a clean-room parser for the four header layouts
//! Go has shipped, dispatched by the leading magic word (whose byte order
//! also fixes the table's endianness — Go is overwhelmingly little-endian,
//! but a big-endian magic is honored too):
//!
//! - **Go 1.2** — magic `0xfffffffb`. Header is `magic, 0, 0, pcquantum,
//!   ptrsize, nfunc` (the last pointer-sized). The `functab` follows
//!   immediately: `nfunc` pairs of `(entry, funcoff)`, each field
//!   pointer-sized, then a sentinel max-PC. `funcoff` is an offset **from
//!   the table base** to a `_func` struct that begins `entry` (pointer)
//!   then `nameoff` (`int32`); `nameoff` is likewise an offset from the
//!   table base to a NUL-terminated name. Entries are absolute VAs.
//!
//! - **Go 1.16** — magic `0xfffffffa`. The header grew a directory of
//!   pointer-sized offsets (all relative to the table base): `nfunc,
//!   nfiles, funcnametab, cutab, filetab, pctab, pclntab`. `functab` lives
//!   at the `pclntab` offset, still `(entry, funcoff)` pointer-sized pairs;
//!   `funcoff` is now relative to that `pclntab` offset, and a `_func`'s
//!   `nameoff` is relative to the `funcnametab` offset. Entries stay
//!   absolute VAs.
//!
//! - **Go 1.18** — magic `0xfffffff0`. The header inserted `textStart`
//!   (a VA) after `nfiles`, and function entries became **pc-relative**:
//!   `functab` entries are two `uint32`s `(entryoff, funcoff)` regardless
//!   of pointer size, and a `_func` begins `entryOff` (`uint32`) then
//!   `nameOff` (`int32`). A function's VA is `textStart + entryOff`.
//!
//! - **Go 1.20** — magic `0xfffffff1`. Identical structure to 1.18; only
//!   the magic changed, so it shares the 1.18 parser.
//!
//! # Best-effort contract
//!
//! [`recover`] never errors and never panics on any input — truncated
//! tables, lying counts, offsets past the end, and cyclic-looking data all
//! yield *fewer* functions (possibly none), never a crash. Every read is
//! bounds-checked; a bad offset skips one function rather than aborting the
//! parse. Attacker-controlled counts are capped so a small hostile blob can
//! neither loop forever nor allocate unboundedly:
//!
//! - [`MAX_FUNCS`] caps how many `functab` entries are walked.
//! - [`MAX_NAME_LEN`] caps a recovered name's length (it is UTF-8 decoded
//!   lossily, and an embedded NUL ends it earlier).
//! - [`MAX_SCAN_BYTES`] and [`MAX_SCAN_CANDIDATES`] bound the work of
//!   locating the table by scanning when it is not in a named region.

use crate::model::Image;

/// Go 1.2 header magic.
const MAGIC_12: u32 = 0xffff_fffb;
/// Go 1.16 header magic.
const MAGIC_116: u32 = 0xffff_fffa;
/// Go 1.18 header magic.
const MAGIC_118: u32 = 0xffff_fff0;
/// Go 1.20 header magic (structurally identical to 1.18).
const MAGIC_120: u32 = 0xffff_fff1;

/// Upper bound on `functab` entries walked, whatever `nfunc` claims. Real
/// Go binaries top out in the tens of thousands of functions; this leaves
/// generous headroom while capping a hostile `nfunc` (e.g. `u32::MAX`).
const MAX_FUNCS: usize = 1_000_000;

/// Upper bound on a recovered name's byte length. A name is truncated here
/// if no NUL is seen first; real Go names are far shorter.
const MAX_NAME_LEN: usize = 4096;

/// Per-region cap on bytes scanned when locating the table by signature.
const MAX_SCAN_BYTES: usize = 64 * 1024 * 1024;

/// Cap on how many magic-shaped positions the locate-scan will fully
/// validate, bounding total work across all regions.
const MAX_SCAN_CANDIDATES: usize = 256;

/// A recovered Go function: its entry virtual address and its name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoFunc {
    /// Entry point, as a virtual address (pc-relative layouts add the
    /// module's `textStart`).
    pub va: u64,
    /// Function name, decoded from `funcnametab` (lossy UTF-8).
    pub name: String,
}

/// Which header layout a magic selects. Go 1.20 shares [`Version::V118`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Version {
    V12,
    V116,
    V118,
}

/// Byte order of a table, fixed by the endianness of its magic word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

/// A bounds-checked, byte-order-aware view over a table's bytes.
///
/// Every accessor validates the requested range with checked arithmetic
/// and returns `None` on any overflow or out-of-bounds read, so a lying
/// offset can never panic or read past the buffer.
struct View<'a> {
    data: &'a [u8],
    endian: Endian,
}

impl<'a> View<'a> {
    fn slice(&self, off: usize, len: usize) -> Option<&'a [u8]> {
        let end = off.checked_add(len)?;
        self.data.get(off..end)
    }

    fn u32(&self, off: usize) -> Option<u32> {
        let b = self.slice(off, 4)?;
        let a = [b[0], b[1], b[2], b[3]];
        Some(match self.endian {
            Endian::Little => u32::from_le_bytes(a),
            Endian::Big => u32::from_be_bytes(a),
        })
    }

    fn u64(&self, off: usize) -> Option<u64> {
        let b = self.slice(off, 8)?;
        let a = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
        Some(match self.endian {
            Endian::Little => u64::from_le_bytes(a),
            Endian::Big => u64::from_be_bytes(a),
        })
    }

    /// Read a pointer-sized unsigned value (`ptrsize` is 4 or 8; any other
    /// width yields `None`).
    fn uptr(&self, off: usize, ptrsize: u8) -> Option<u64> {
        match ptrsize {
            4 => self.u32(off).map(u64::from),
            8 => self.u64(off),
            _ => None,
        }
    }

    /// Read a NUL-terminated name starting at `off`, decoded lossily and
    /// truncated at [`MAX_NAME_LEN`] if no NUL appears first. Returns `None`
    /// only when `off` is itself out of bounds.
    fn name(&self, off: usize) -> Option<String> {
        let rest = self.data.get(off..)?;
        let window = &rest[..rest.len().min(MAX_NAME_LEN)];
        let end = window.iter().position(|&b| b == 0).unwrap_or(window.len());
        Some(String::from_utf8_lossy(&window[..end]).into_owned())
    }
}

/// Classify the first four bytes as a known magic, returning the layout and
/// the endianness that matched. The magics are self-distinguishing under
/// byte reversal, so there is no little/big ambiguity.
fn detect(head: &[u8]) -> Option<(Version, Endian)> {
    let b = head.get(0..4)?;
    let a = [b[0], b[1], b[2], b[3]];
    let candidates = [
        (u32::from_le_bytes(a), Endian::Little),
        (u32::from_be_bytes(a), Endian::Big),
    ];
    for (word, endian) in candidates {
        let version = match word {
            MAGIC_12 => Some(Version::V12),
            MAGIC_116 => Some(Version::V116),
            MAGIC_118 | MAGIC_120 => Some(Version::V118),
            _ => None,
        };
        if let Some(v) = version {
            return Some((v, endian));
        }
    }
    None
}

/// True if `b` opens with a known magic followed by the fixed
/// `00 00 <pcquantum> <ptrsize>` shape with plausible `pcquantum`
/// (1, 2, or 4) and `ptrsize` (4 or 8) — the cheap gate the locate-scan
/// applies before attempting a full parse.
fn is_header_shape(b: &[u8]) -> bool {
    b.len() >= 8
        && detect(b).is_some()
        && b[4] == 0
        && b[5] == 0
        && matches!(b[6], 1 | 2 | 4)
        && matches!(b[7], 4 | 8)
}

/// Recover Go functions from an in-memory `pclntab` image.
///
/// `text_base` is used as `textStart` for the pc-relative (1.18/1.20)
/// layouts only when the table's own embedded `textStart` is zero; the
/// 1.2 and 1.16 layouts store absolute entry VAs and ignore it. Returns
/// the functions sorted by VA and deduplicated by VA, or an empty vector
/// for an unrecognized or unparseable table. Never panics.
pub fn recover_from(table: &[u8], text_base: u64) -> Vec<GoFunc> {
    if table.len() < 8 {
        return Vec::new();
    }
    let Some((version, endian)) = detect(table) else {
        return Vec::new();
    };
    let pcquantum = table[6];
    let ptrsize = table[7];
    if !matches!(ptrsize, 4 | 8) || !matches!(pcquantum, 1 | 2 | 4) {
        return Vec::new();
    }
    let view = View {
        data: table,
        endian,
    };
    let funcs = match version {
        Version::V12 => parse_v12(&view, ptrsize),
        Version::V116 => parse_v116(&view, ptrsize),
        Version::V118 => parse_v118(&view, ptrsize, text_base),
    };
    finalize(funcs)
}

/// Parse the Go 1.2 layout (see the module docs).
fn parse_v12(view: &View, ptrsize: u8) -> Vec<GoFunc> {
    let ps = ptrsize as usize;
    let Some(nfunc) = view.uptr(8, ptrsize) else {
        return Vec::new();
    };
    let nfunc = (nfunc as usize).min(MAX_FUNCS);
    let functab_base = 8 + ps;
    let entry_size = 2 * ps;
    let mut out = Vec::new();
    for i in 0..nfunc {
        let Some(ent_off) = functab_base.checked_add(i * entry_size) else {
            break;
        };
        // Second field of the pair: offset from the table base to `_func`.
        let Some(funcoff) = view.uptr(ent_off + ps, ptrsize) else {
            break;
        };
        let fo = funcoff as usize;
        // `_func`: entry (pointer-sized VA) then nameoff (int32), the name
        // offset relative to the table base.
        let (Some(entry), Some(nameoff)) = (view.uptr(fo, ptrsize), view.u32(fo.wrapping_add(ps)))
        else {
            continue;
        };
        push_func(&mut out, view, entry, nameoff as usize);
    }
    out
}

/// Parse the Go 1.16 layout (see the module docs).
fn parse_v116(view: &View, ptrsize: u8) -> Vec<GoFunc> {
    let ps = ptrsize as usize;
    // Header directory: nfunc, nfiles, funcnametab, cutab, filetab, pctab,
    // pclntab — all pointer-sized, all relative to the table base.
    let (Some(nfunc), Some(funcname_off), Some(pcln_off)) = (
        view.uptr(8, ptrsize),
        view.uptr(8 + 2 * ps, ptrsize),
        view.uptr(8 + 6 * ps, ptrsize),
    ) else {
        return Vec::new();
    };
    let nfunc = (nfunc as usize).min(MAX_FUNCS);
    let functab_base = pcln_off as usize;
    let funcname_base = funcname_off as usize;
    let entry_size = 2 * ps;
    let mut out = Vec::new();
    for i in 0..nfunc {
        let Some(ent_off) = functab_base.checked_add(i * entry_size) else {
            break;
        };
        let Some(funcoff) = view.uptr(ent_off.wrapping_add(ps), ptrsize) else {
            break;
        };
        // `funcoff` is relative to the functab (pclntab) base.
        let Some(fo) = functab_base.checked_add(funcoff as usize) else {
            continue;
        };
        let (Some(entry), Some(nameoff)) = (view.uptr(fo, ptrsize), view.u32(fo.wrapping_add(ps)))
        else {
            continue;
        };
        // `nameoff` is relative to the funcnametab base.
        let Some(name_at) = funcname_base.checked_add(nameoff as usize) else {
            continue;
        };
        if let Some(name) = view.name(name_at)
            && !name.is_empty()
        {
            out.push(GoFunc { va: entry, name });
        }
    }
    out
}

/// Parse the Go 1.18 / 1.20 layout (see the module docs).
fn parse_v118(view: &View, ptrsize: u8, text_base: u64) -> Vec<GoFunc> {
    let ps = ptrsize as usize;
    // Header directory: nfunc, nfiles, textStart, funcnametab, cutab,
    // filetab, pctab, pclntab — all pointer-sized.
    let (Some(nfunc), Some(text_start), Some(funcname_off), Some(pcln_off)) = (
        view.uptr(8, ptrsize),
        view.uptr(8 + 2 * ps, ptrsize),
        view.uptr(8 + 3 * ps, ptrsize),
        view.uptr(8 + 7 * ps, ptrsize),
    ) else {
        return Vec::new();
    };
    let text = if text_start != 0 {
        text_start
    } else {
        text_base
    };
    let nfunc = (nfunc as usize).min(MAX_FUNCS);
    let functab_base = pcln_off as usize;
    let funcname_base = funcname_off as usize;
    // functab entries are two uint32s regardless of pointer size.
    let entry_size = 8usize;
    let mut out = Vec::new();
    for i in 0..nfunc {
        let Some(ent_off) = functab_base.checked_add(i * entry_size) else {
            break;
        };
        let Some(funcoff) = view.u32(ent_off.wrapping_add(4)) else {
            break;
        };
        let Some(fo) = functab_base.checked_add(funcoff as usize) else {
            continue;
        };
        // `_func`: entryOff (uint32, pc-relative to textStart) then
        // nameOff (int32, relative to the funcnametab base).
        let (Some(entryoff), Some(nameoff)) = (view.u32(fo), view.u32(fo.wrapping_add(4))) else {
            continue;
        };
        let Some(name_at) = funcname_base.checked_add(nameoff as usize) else {
            continue;
        };
        if let Some(name) = view.name(name_at)
            && !name.is_empty()
        {
            out.push(GoFunc {
                va: text.wrapping_add(u64::from(entryoff)),
                name,
            });
        }
    }
    out
}

/// Resolve a table-base-relative `nameoff` (the 1.2 layout) and record the
/// function if the name is non-empty.
fn push_func(out: &mut Vec<GoFunc>, view: &View, va: u64, nameoff: usize) {
    if let Some(name) = view.name(nameoff)
        && !name.is_empty()
    {
        out.push(GoFunc { va, name });
    }
}

/// Sort by VA (then name, for determinism) and drop duplicate VAs.
fn finalize(mut funcs: Vec<GoFunc>) -> Vec<GoFunc> {
    funcs.sort_by(|a, b| a.va.cmp(&b.va).then_with(|| a.name.cmp(&b.name)));
    funcs.dedup_by_key(|f| f.va);
    funcs
}

/// Recover Go functions from a loaded [`Image`], best-effort.
///
/// Returns the functions sorted by VA and deduplicated by VA, or an empty
/// vector for a non-Go or unparseable image. Never errors, never panics.
/// The table is located by [`locate`]: a `.gopclntab` / `__gopclntab`
/// region first, otherwise a bounded signature scan.
pub fn recover(image: &dyn Image) -> Vec<GoFunc> {
    match locate(image) {
        Some(table) => recover_from(table, 0),
        None => Vec::new(),
    }
}

/// Whether `image` appears to contain a locatable Go `pclntab`.
pub fn is_go(image: &dyn Image) -> bool {
    locate(image).is_some()
}

/// Locate the `pclntab` bytes within an image: a section/segment named
/// `.gopclntab` (ELF/PE) or `__gopclntab` (Mach-O) if present, else a
/// bounded scan of readable regions for the header signature.
fn locate(image: &dyn Image) -> Option<&[u8]> {
    let bytes = image.bytes();
    for r in image.regions() {
        if (r.name == ".gopclntab" || r.name == "__gopclntab")
            && let Some(off) = image.va_to_offset(r.va)
        {
            let end = off.saturating_add(r.size as usize).min(bytes.len());
            if let Some(slice) = bytes.get(off..end)
                && is_header_shape(slice)
            {
                return Some(slice);
            }
        }
    }
    scan(image)
}

/// Scan an image's readable regions (or the whole buffer when it exposes
/// none) for a header-shaped position whose full parse yields functions.
/// Bounded by [`MAX_SCAN_BYTES`] per region and [`MAX_SCAN_CANDIDATES`]
/// validation attempts overall.
fn scan(image: &dyn Image) -> Option<&[u8]> {
    let bytes = image.bytes();
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for r in image.regions() {
        if r.perms.r
            && let Some(off) = image.va_to_offset(r.va)
        {
            let end = off.saturating_add(r.size as usize).min(bytes.len());
            if off < end {
                ranges.push((off, end));
            }
        }
    }
    if ranges.is_empty() {
        ranges.push((0, bytes.len()));
    }
    let mut candidates = 0usize;
    for (start, end) in ranges {
        let region = &bytes[start..end];
        let scan_len = region.len().min(MAX_SCAN_BYTES);
        let mut i = 0;
        while i + 8 <= scan_len {
            if is_header_shape(&region[i..]) {
                candidates += 1;
                if candidates > MAX_SCAN_CANDIDATES {
                    return None;
                }
                let table = &region[i..];
                if !recover_from(table, 0).is_empty() {
                    return Some(table);
                }
            }
            i += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::tests::synthetic_elf64;
    use crate::model::{Arch, ImportSlot, Perms, Region, Symbol};

    // ---- Hand-built minimal tables (ptrsize=8, pcquantum=1) -------------

    fn push_u32(v: &mut Vec<u8>, x: u32, be: bool) {
        if be {
            v.extend_from_slice(&x.to_be_bytes());
        } else {
            v.extend_from_slice(&x.to_le_bytes());
        }
    }

    fn push_u64(v: &mut Vec<u8>, x: u64, be: bool) {
        if be {
            v.extend_from_slice(&x.to_be_bytes());
        } else {
            v.extend_from_slice(&x.to_le_bytes());
        }
    }

    fn header_prefix(v: &mut Vec<u8>, magic: u32, be: bool) {
        if be {
            v.extend_from_slice(&magic.to_be_bytes());
        } else {
            v.extend_from_slice(&magic.to_le_bytes());
        }
        v.push(0); // pad1
        v.push(0); // pad2
        v.push(1); // pcquantum
        v.push(8); // ptrsize
    }

    /// Go 1.2 table with `main`@0x1000 and `helper`@0x2000.
    fn build_v12(be: bool) -> Vec<u8> {
        let mut v = Vec::new();
        header_prefix(&mut v, MAGIC_12, be); // 0..8
        push_u64(&mut v, 2, be); // nfunc                         8..16
        // functab: (entry, funcoff) pairs + sentinel            16..64
        push_u64(&mut v, 0x1000, be);
        push_u64(&mut v, 64, be); // _func0 at table+64
        push_u64(&mut v, 0x2000, be);
        push_u64(&mut v, 76, be); // _func1 at table+76
        push_u64(&mut v, 0x3000, be);
        push_u64(&mut v, 0, be); // sentinel
        // _func0: entry, nameoff                                64..76
        push_u64(&mut v, 0x1000, be);
        push_u32(&mut v, 88, be); // "main" at table+88
        // _func1: entry, nameoff                                76..88
        push_u64(&mut v, 0x2000, be);
        push_u32(&mut v, 93, be); // "helper" at table+93
        v.extend_from_slice(b"main\0"); // 88..93
        v.extend_from_slice(b"helper\0"); // 93..100
        v
    }

    /// Go 1.16 table with `main`@0x401000 and `helper`@0x402000.
    fn build_v116(be: bool) -> Vec<u8> {
        let mut v = Vec::new();
        header_prefix(&mut v, MAGIC_116, be); // 0..8
        push_u64(&mut v, 2, be); // nfunc            8..16
        push_u64(&mut v, 0, be); // nfiles           16..24
        push_u64(&mut v, 64, be); // funcnametab     24..32
        push_u64(&mut v, 0, be); // cutab            32..40
        push_u64(&mut v, 0, be); // filetab          40..48
        push_u64(&mut v, 0, be); // pctab            48..56
        push_u64(&mut v, 76, be); // pclntab         56..64
        v.extend_from_slice(b"main\0"); // 64..69
        v.extend_from_slice(b"helper\0"); // 69..76
        // functab at 76: (entry, funcoff) pairs + sentinel   76..124
        push_u64(&mut v, 0x401000, be);
        push_u64(&mut v, 48, be); // _func0 at pclntab+48 = 124
        push_u64(&mut v, 0x402000, be);
        push_u64(&mut v, 60, be); // _func1 at pclntab+60 = 136
        push_u64(&mut v, 0x403000, be);
        push_u64(&mut v, 0, be); // sentinel
        // _func0                                              124..136
        push_u64(&mut v, 0x401000, be);
        push_u32(&mut v, 0, be); // nameoff -> funcnametab+0 = "main"
        // _func1                                              136..148
        push_u64(&mut v, 0x402000, be);
        push_u32(&mut v, 5, be); // nameoff -> funcnametab+5 = "helper"
        v
    }

    /// Go 1.18/1.20 table (magic selectable) with textStart 0x400000,
    /// `main`@0x401000 and `helper`@0x402000.
    fn build_v118(magic: u32, be: bool) -> Vec<u8> {
        let mut v = Vec::new();
        header_prefix(&mut v, magic, be); // 0..8
        push_u64(&mut v, 2, be); // nfunc              8..16
        push_u64(&mut v, 0, be); // nfiles             16..24
        push_u64(&mut v, 0x400000, be); // textStart   24..32
        push_u64(&mut v, 72, be); // funcnametab       32..40
        push_u64(&mut v, 0, be); // cutab              40..48
        push_u64(&mut v, 0, be); // filetab            48..56
        push_u64(&mut v, 0, be); // pctab              56..64
        push_u64(&mut v, 84, be); // pclntab           64..72
        v.extend_from_slice(b"main\0"); // 72..77
        v.extend_from_slice(b"helper\0"); // 77..84
        // functab at 84: (entryoff, funcoff) uint32 pairs + sentinel 84..108
        push_u32(&mut v, 0x1000, be);
        push_u32(&mut v, 24, be); // _func0 at pclntab+24 = 108
        push_u32(&mut v, 0x2000, be);
        push_u32(&mut v, 32, be); // _func1 at pclntab+32 = 116
        push_u32(&mut v, 0x3000, be);
        push_u32(&mut v, 0, be); // sentinel
        // _func0: entryOff, nameOff                                108..116
        push_u32(&mut v, 0x1000, be);
        push_u32(&mut v, 0, be); // "main"
        // _func1: entryOff, nameOff                                116..124
        push_u32(&mut v, 0x2000, be);
        push_u32(&mut v, 5, be); // "helper"
        v
    }

    // ---- Layout parsing --------------------------------------------------

    #[test]
    fn recovers_go12_layout() {
        let funcs = recover_from(&build_v12(false), 0);
        assert_eq!(
            funcs,
            vec![
                GoFunc {
                    va: 0x1000,
                    name: "main".into()
                },
                GoFunc {
                    va: 0x2000,
                    name: "helper".into()
                },
            ]
        );
    }

    #[test]
    fn recovers_go116_layout() {
        let funcs = recover_from(&build_v116(false), 0);
        assert_eq!(
            funcs,
            vec![
                GoFunc {
                    va: 0x401000,
                    name: "main".into()
                },
                GoFunc {
                    va: 0x402000,
                    name: "helper".into()
                },
            ]
        );
    }

    #[test]
    fn recovers_go118_and_go120_layout() {
        let expected = vec![
            GoFunc {
                va: 0x401000,
                name: "main".into(),
            },
            GoFunc {
                va: 0x402000,
                name: "helper".into(),
            },
        ];
        // Both magics route to the same parser and recover the same pairs.
        assert_eq!(recover_from(&build_v118(MAGIC_118, false), 0), expected);
        assert_eq!(recover_from(&build_v118(MAGIC_120, false), 0), expected);
    }

    #[test]
    fn text_base_fallback_when_textstart_zero() {
        // Zero the embedded textStart (offset 24..32) and supply text_base.
        let mut t = build_v118(MAGIC_118, false);
        t[24..32].fill(0);
        let funcs = recover_from(&t, 0x400000);
        assert_eq!(funcs[0].va, 0x401000);
        assert_eq!(funcs[1].va, 0x402000);
    }

    // ---- Version dispatch ------------------------------------------------

    #[test]
    fn unknown_magic_yields_empty() {
        let mut t = build_v118(MAGIC_118, false);
        t[0..4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        assert!(recover_from(&t, 0).is_empty());
    }

    #[test]
    fn implausible_ptrsize_or_pcquantum_yields_empty() {
        let mut t = build_v12(false);
        t[7] = 3; // ptrsize neither 4 nor 8
        assert!(recover_from(&t, 0).is_empty());
        let mut t = build_v12(false);
        t[6] = 7; // pcquantum not 1/2/4
        assert!(recover_from(&t, 0).is_empty());
    }

    // ---- Endianness ------------------------------------------------------

    #[test]
    fn big_endian_tables_parse() {
        assert_eq!(
            recover_from(&build_v12(true), 0),
            vec![
                GoFunc {
                    va: 0x1000,
                    name: "main".into()
                },
                GoFunc {
                    va: 0x2000,
                    name: "helper".into()
                },
            ]
        );
        assert_eq!(recover_from(&build_v118(MAGIC_118, true), 0).len(), 2);
        assert_eq!(recover_from(&build_v116(true), 0).len(), 2);
    }

    // ---- Truncation ------------------------------------------------------

    #[test]
    fn truncation_sweep_never_panics() {
        for full in [
            build_v12(false),
            build_v116(false),
            build_v118(MAGIC_118, false),
            build_v118(MAGIC_120, true),
        ] {
            for len in 0..=full.len() {
                let out = recover_from(&full[..len], 0);
                // A prefix can only ever recover a subset of the whole.
                assert!(out.len() <= 2, "len {len}: {} funcs", out.len());
            }
        }
    }

    // ---- Hostile inputs --------------------------------------------------

    #[test]
    fn hostile_counts_are_bounded_and_safe() {
        // nfunc = u64::MAX (well past MAX_FUNCS) must not over-allocate or
        // loop: the walk breaks as soon as a functab entry is OOB.
        let mut t = build_v118(MAGIC_118, false);
        t[8..16].fill(0xff);
        let out = recover_from(&t, 0);
        assert!(out.len() < 1000);

        // funcoff past the end: the _func read fails, the function is
        // skipped, no panic.
        let mut t = build_v118(MAGIC_118, false);
        t[88..92].copy_from_slice(&0xffff_ffffu32.to_le_bytes()); // funcoff0
        let _ = recover_from(&t, 0);

        // nameoff past the end: the name read fails, function skipped.
        let mut t = build_v118(MAGIC_118, false);
        t[112..116].copy_from_slice(&0xffff_ffffu32.to_le_bytes()); // _func0 nameOff
        let out = recover_from(&t, 0);
        assert!(out.len() <= 2);

        // Pointer-sized nfunc = u64::MAX on the 1.2 layout, too.
        let mut t = build_v12(false);
        t[8..16].fill(0xff);
        assert!(recover_from(&t, 0).len() < 1000);
    }

    // ---- Deterministic fuzz ---------------------------------------------

    #[test]
    fn fixed_seed_fuzz_is_bounded_and_panic_free() {
        // A tiny LCG (fixed seed) drives a few thousand blobs, each opening
        // with a real magic then random bytes. We only assert the parse
        // terminates and the total is bounded — the point is no panic and
        // no runaway allocation.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let magics = [MAGIC_12, MAGIC_116, MAGIC_118, MAGIC_120];
        let mut total = 0usize;
        for _ in 0..4000 {
            let mut blob = Vec::new();
            let magic = magics[(next() as usize) % magics.len()];
            blob.extend_from_slice(&magic.to_le_bytes());
            blob.push(0);
            blob.push(0);
            blob.push(1);
            blob.push(8);
            let extra = (next() as usize) % 512;
            for _ in 0..extra {
                blob.push(next() as u8);
            }
            let out = recover_from(&blob, next() as u64);
            total = total.saturating_add(out.len());
        }
        assert!(total < 10_000_000, "unbounded fuzz output: {total}");
    }

    #[test]
    fn fully_random_blobs_never_panic() {
        let mut state: u64 = 0xdead_beef_0bad_f00d;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for _ in 0..4000 {
            let len = (next() as usize) % 256;
            let blob: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            let _ = recover_from(&blob, next() as u64);
        }
    }

    // ---- Image-level recovery via a minimal mock ------------------------

    /// Identity-mapped image (VA == file offset) exposing a chosen set of
    /// regions over a byte buffer — enough to exercise `locate`/`recover`.
    struct FakeImage {
        data: Vec<u8>,
        regions: Vec<Region>,
    }

    impl Image for FakeImage {
        fn arch(&self) -> Arch {
            Arch::X86_64
        }
        fn entry_points(&self) -> Vec<u64> {
            Vec::new()
        }
        fn regions(&self) -> Vec<Region> {
            self.regions.clone()
        }
        fn symbols(&self) -> Vec<Symbol> {
            Vec::new()
        }
        fn import_slots(&self) -> Vec<ImportSlot> {
            Vec::new()
        }
        fn va_to_offset(&self, va: u64) -> Option<usize> {
            let off = va as usize;
            (off <= self.data.len()).then_some(off)
        }
        fn bytes(&self) -> &[u8] {
            &self.data
        }
    }

    fn image_with_table(name: &str, perms_r: bool) -> FakeImage {
        let table = build_v118(MAGIC_118, false);
        let base = 0x100usize;
        let mut data = vec![0u8; base];
        data.extend_from_slice(&table);
        FakeImage {
            data,
            regions: vec![Region {
                name: name.into(),
                va: base as u64,
                size: table.len() as u64,
                perms: Perms {
                    r: perms_r,
                    w: false,
                    x: false,
                },
            }],
        }
    }

    #[test]
    fn recover_finds_named_gopclntab() {
        let img = image_with_table(".gopclntab", true);
        let funcs = recover(&img);
        assert_eq!(funcs.len(), 2);
        assert_eq!(funcs[0].va, 0x401000);
        assert!(is_go(&img));
    }

    #[test]
    fn recover_finds_table_via_scan() {
        // Unnamed region: the named-region path misses, the scan succeeds.
        let img = image_with_table("", true);
        assert_eq!(recover(&img).len(), 2);
    }

    #[test]
    fn recover_on_non_go_image_is_empty() {
        let elf = synthetic_elf64();
        let img = crate::model::ElfImage::parse(&elf).unwrap();
        assert!(recover(&img).is_empty());
        assert!(!is_go(&img));
    }
}
