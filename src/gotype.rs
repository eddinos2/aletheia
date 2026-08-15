//! Go runtime type metadata — named types and interface tables.
//!
//! The counterpart to [`crate::gopcln`]'s function recovery: a Go binary
//! also embeds its runtime type graph (`runtime._type` records reached
//! through `moduledata`), and this module recovers the named types, their
//! kinds, and the interface-satisfaction pairs the `itab` records encode.
//! Best-effort by contract: a binary without recognizable Go metadata
//! yields an empty result, and no input ever panics.
//!
//! # The chain from bytes to types
//!
//! The runtime's own source is the specification — `runtime/symtab.go`
//! (`moduledata`), `internal/abi/type.go` (type records and their name
//! encoding), and `runtime/iface.go` (`itab`). Recovery follows the same
//! chain the runtime itself walks, from the one anchor a stripped binary
//! cannot hide:
//!
//! 1. **The `pclntab`.** Located exactly as [`crate::gopcln`] locates it —
//!    a `.gopclntab` / `__gopclntab` region first, else a bounded scan for
//!    the header signature, confirmed by actually parsing functions out of
//!    it. Its *virtual address* is the anchor for everything below.
//! 2. **`moduledata`.** Its first field is `pcHeader *pcHeader`, which
//!    points at the `pclntab`. Scanning the readable regions for a
//!    pointer-sized word equal to the `pclntab` VA therefore nominates
//!    every plausible `moduledata`; [`Layout`] then decides.
//! 3. **`typelinks`.** A `[]int32` of offsets from `moduledata.types`,
//!    each naming an `internal/abi.Type` record: `Size, PtrBytes, Hash,
//!    TFlag, Align, FieldAlign, Kind, Equal, GCData, Str, PtrToThis`. The
//!    `Str` field is a `nameOff` — also relative to `types` — addressing a
//!    go1.17+ encoded name: a flag byte, a uvarint length, then that many
//!    UTF-8 bytes.
//! 4. **`itablinks`.** A `[]*itab`, each `itab { inter *interfacetype,
//!    _type *Type, hash u32, _ u32, fun [...]uintptr }`. An
//!    `interfacetype` begins with an embedded `Type`, so both halves of
//!    the pair parse with the same record reader; the `fun` array is
//!    walked while its entries land in executable memory.
//!
//! # Which `moduledata` layouts are accepted
//!
//! The go1.18+ field order is fixed up to `enoptrbss`, after which the set
//! differs slightly by release — go1.20 inserted the `covctrs, ecovctrs`
//! pair. Rather than fingerprint the toolchain version, [`validate`] tries
//! a small window of plausible layouts ([`LAYOUTS`]) and accepts the first
//! that *structurally* checks out: `minpc <= maxpc`, `text <= etext`,
//! `types <= etypes`, `minpc` and `text` executable, `types` readable, the
//! `typelinks` and `itablinks` slice headers sane (`len <= cap`, both
//! bounded, the base readable), and — the discriminator that keeps one
//! layout from validating another's binary — at least one of the first few
//! `typelinks` entries resolving to a well-formed, printably-named type
//! record. If no layout checks out, the candidate is refused; if no
//! candidate validates, [`recover`] returns the empty result.
//!
//! # Best-effort contract
//!
//! [`recover`] never errors and never panics on any input. Every read is
//! bounds-checked against both the owning region and the file buffer, all
//! offset arithmetic is checked, and attacker-controlled counts are capped
//! so a small hostile blob can neither loop forever nor allocate
//! unboundedly: [`MAX_TYPELINKS`], [`MAX_ITABS`], [`MAX_NAME_LEN`],
//! [`MAX_ITAB_METHODS`], [`MAX_UVARINT_BYTES`], and the scan bounds
//! [`MAX_SCAN_BYTES`] / [`MAX_MODULEDATA_CANDIDATES`] /
//! [`MAX_PCLNTAB_CANDIDATES`]. Only 64-bit little-endian images are
//! recovered; anything else yields the empty result.

use crate::gopcln;
use crate::model::{Image, Region};

// ---------------------------------------------------------------------------
// Caps
// ---------------------------------------------------------------------------

/// Upper bound on `typelinks` entries, whatever the slice header claims.
/// A header past this is refused outright rather than clamped: a real Go
/// binary's type list is orders of magnitude smaller, so an oversized one
/// is evidence the candidate is not a `moduledata` at all.
const MAX_TYPELINKS: usize = 1 << 20;

/// Upper bound on `itablinks` entries, applied the same way.
const MAX_ITABS: usize = 1 << 18;

/// Upper bound on a decoded type name's byte length.
const MAX_NAME_LEN: usize = 4096;

/// Upper bound on `itab.fun` entries collected for one `itab`.
const MAX_ITAB_METHODS: usize = 1024;

/// Upper bound on bytes consumed by one uvarint (Go's own reader is
/// unbounded; ours refuses anything longer, and any value above `2^32`).
const MAX_UVARINT_BYTES: usize = 10;

/// Cap on how many `pcHeader`-shaped words the `moduledata` scan will
/// fully validate, bounding total work on an image seeded with the
/// `pclntab` VA in every slot.
const MAX_MODULEDATA_CANDIDATES: usize = 1024;

/// Cap on how many distinct `pclntab` VAs are tried as anchors.
const MAX_PCLNTAB_CANDIDATES: usize = 4;

/// Per-region cap on bytes scanned, for both the `pclntab` signature scan
/// and the `moduledata` pointer scan.
const MAX_SCAN_BYTES: usize = 64 * 1024 * 1024;

/// How many leading `typelinks` entries [`validate`] probes before
/// deciding a candidate layout does not describe real type metadata.
const MAX_TYPELINK_PROBES: usize = 4;

/// How many overlapping regions [`Mem::region_at`] inspects at one VA.
const MAX_REGION_PROBE: usize = 16;

// ---------------------------------------------------------------------------
// Fixed layout constants (64-bit, little-endian)
// ---------------------------------------------------------------------------

/// Pointer size of the images this module recovers.
const PTR: u64 = 8;

/// Size of an `internal/abi.Type` record on a 64-bit target.
const TYPE_SIZE: usize = 48;

/// Byte offsets within an `internal/abi.Type` record.
const TYPE_TFLAG: usize = 20;
const TYPE_KIND: usize = 23;
const TYPE_STR: usize = 40;

/// `abi.TFlagExtraStar`: the stored name carries an extraneous `*`.
const TFLAG_EXTRA_STAR: u8 = 1 << 1;

/// `abi.KindMask`: the low five bits of `Type.Kind` select the kind.
const KIND_MASK: u8 = (1 << 5) - 1;

/// Offset of `itab._type` (`itab.inter` is at 0).
const ITAB_TYPE: u64 = 8;

/// Offset of `itab.fun[0]`, past `inter`, `_type`, `hash` and its padding.
const ITAB_FUN: u64 = 24;

/// The go1.18+ `interfacetype` embeds a `Type` whose kind is this.
const KIND_INTERFACE: u8 = 20;

/// `Type.Kind` values, in `internal/abi` order. Index 0 (`Invalid`) is a
/// placeholder: [`kind_name`] rejects it, and any index past the end.
const KIND_NAMES: [&str; 27] = [
    "invalid",
    "bool",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uintptr",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "array",
    "chan",
    "func",
    "interface",
    "map",
    "ptr",
    "slice",
    "string",
    "struct",
    "unsafe.Pointer",
];

/// The `pclntab` header magics, in the byte order a 64-bit little-endian
/// Go image writes them (go1.2, go1.16, go1.18, go1.20).
const PCLNTAB_MAGICS: [u32; 4] = [0xffff_fffb, 0xffff_fffa, 0xffff_fff0, 0xffff_fff1];

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One recovered Go named type: where its `_type` record lives, what the
/// runtime calls it, and which kind it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoType {
    /// Virtual address of the `internal/abi.Type` record — this type's
    /// identity, and the key [`recover`] sorts and deduplicates on.
    pub va: u64,
    /// The runtime's name for the type, with the `TFlagExtraStar` prefix
    /// already stripped (so `*main.Widget` stored with that flag is
    /// reported as `main.Widget`).
    pub name: String,
    /// Kind name from the `reflect.Kind` vocabulary — `struct`, `ptr`,
    /// `slice`, `unsafe.Pointer`, and so on.
    pub kind: &'static str,
}

/// One recovered interface table: proof that a concrete type satisfies an
/// interface, plus the method addresses the runtime dispatches through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Itab {
    /// Virtual address of the `itab` record.
    pub va: u64,
    /// Name of the interface type (`itab.inter`).
    pub interface_name: String,
    /// Name of the concrete type (`itab._type`).
    pub type_name: String,
    /// Method entry VAs from `itab.fun`, in table order — collected while
    /// they land in executable memory, capped at [`MAX_ITAB_METHODS`].
    pub methods: Vec<u64>,
}

/// Everything [`recover`] found: named types and interface tables, each
/// sorted by VA and deduplicated on it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GoTypes {
    /// Named types reached through `moduledata.typelinks`.
    pub types: Vec<GoType>,
    /// Interface tables reached through `moduledata.itablinks`.
    pub itabs: Vec<Itab>,
}

impl GoTypes {
    /// Whether nothing at all was recovered.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty() && self.itabs.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Recover Go runtime type metadata from a loaded [`Image`], best-effort.
///
/// Returns the named types and `itab` pairs, each sorted by VA and
/// deduplicated on it — deterministic for a given image — or an empty
/// [`GoTypes`] for a non-Go image, a 32-bit or big-endian image, or one
/// whose `moduledata` does not structurally check out. Never errors,
/// never panics.
pub fn recover(image: &dyn Image) -> GoTypes {
    let mem = Mem::new(image);
    for pclntab_va in locate_pclntab(&mem) {
        if let Some(md) = find_moduledata(&mem, pclntab_va) {
            return collect(&mem, &md);
        }
    }
    GoTypes::default()
}

/// Render recovered metadata as a readable report.
///
/// Deterministic for a given input and always newline-terminated, so a CLI
/// can print it verbatim. An empty result renders as a single explanatory
/// line rather than an empty string.
pub fn render(t: &GoTypes) -> String {
    if t.is_empty() {
        return "no Go type metadata recovered\n".to_string();
    }
    let mut out = format!(
        "Go type metadata: {} types, {} itabs\n",
        t.types.len(),
        t.itabs.len()
    );
    for ty in &t.types {
        out.push_str(&format!(
            "  {}  {:<14}  {}\n",
            hex_va(ty.va),
            ty.kind,
            ty.name
        ));
    }
    if !t.itabs.is_empty() {
        out.push_str("\nitabs\n");
        for it in &t.itabs {
            out.push_str(&format!(
                "  {}  {} <- {}  [{} methods]\n",
                hex_va(it.va),
                it.interface_name,
                it.type_name,
                it.methods.len()
            ));
        }
    }
    out
}

/// Canonical VA spelling for the report.
fn hex_va(va: u64) -> String {
    format!("0x{va:016x}")
}

// ---------------------------------------------------------------------------
// Bounds-checked memory view
// ---------------------------------------------------------------------------

/// A read-only, bounds-checked view of an image's mapped memory.
///
/// Every accessor resolves a VA through the image's regions *and* its
/// VA-to-offset mapping, so a read can neither straddle a region boundary
/// nor leave the file buffer. All arithmetic is checked; no path here can
/// panic on hostile data.
struct Mem<'a> {
    image: &'a dyn Image,
    /// The image's regions, sorted by (VA, size), zero-size ones dropped.
    regions: Vec<Region>,
}

impl<'a> Mem<'a> {
    fn new(image: &'a dyn Image) -> Mem<'a> {
        let mut regions = image.regions();
        regions.retain(|r| r.size > 0);
        regions.sort_by_key(|r| (r.va, r.size));
        Mem { image, regions }
    }

    /// The region containing `va`, if any.
    fn region_at(&self, va: u64) -> Option<&Region> {
        let idx = self.regions.partition_point(|r| r.va <= va);
        self.regions[..idx]
            .iter()
            .rev()
            .take(MAX_REGION_PROBE)
            .find(|r| va.wrapping_sub(r.va) < r.size)
    }

    /// `len` file-backed bytes at `va`, all inside one region.
    fn bytes_at(&self, va: u64, len: usize) -> Option<&'a [u8]> {
        let region = self.region_at(va)?;
        let end = va.checked_add(len as u64)?;
        if end > region.va.checked_add(region.size)? {
            return None;
        }
        let off = self.image.va_to_offset(va)?;
        let stop = off.checked_add(len)?;
        self.image.bytes().get(off..stop)
    }

    fn u8_at(&self, va: u64) -> Option<u8> {
        self.bytes_at(va, 1).map(|b| b[0])
    }

    fn i32_at(&self, va: u64) -> Option<i32> {
        let b = self.bytes_at(va, 4)?;
        Some(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64_at(&self, va: u64) -> Option<u64> {
        let b = self.bytes_at(va, 8)?;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Whether `va` lies in an executable region — the test `text`,
    /// `minpc`, and an `itab.fun` entry must pass.
    fn is_exec(&self, va: u64) -> bool {
        self.region_at(va).is_some_and(|r| r.perms.x)
    }

    /// Whether `va` lies in a readable region — where `moduledata`, the
    /// type records, and the two link arrays live.
    fn is_readable(&self, va: u64) -> bool {
        self.region_at(va).is_some_and(|r| r.perms.r)
    }

    /// The file-backed bytes of one region, clipped to the buffer.
    fn region_bytes(&self, r: &Region) -> Option<&'a [u8]> {
        let off = self.image.va_to_offset(r.va)?;
        let end = off
            .saturating_add(r.size as usize)
            .min(self.image.bytes().len());
        self.image.bytes().get(off..end)
    }

    /// `(base VA, file-backed bytes)` for every readable region, in the
    /// canonical region order — the ranges both scans sweep.
    fn readable_ranges(&self) -> Vec<(u64, &'a [u8])> {
        self.regions
            .iter()
            .filter(|r| r.perms.r)
            .filter_map(|r| Some((r.va, self.region_bytes(r)?)))
            .filter(|(_, b)| !b.is_empty())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Step 1 — locating the pclntab
// ---------------------------------------------------------------------------

/// True if `b` opens with a 64-bit little-endian `pclntab` header: a known
/// magic, the fixed `00 00` pad, a plausible `pcquantum` (1, 2, or 4), and
/// `ptrsize` 8. The same cheap gate [`crate::gopcln`] applies, narrowed to
/// the images this module recovers.
fn is_header_shape(b: &[u8]) -> bool {
    b.len() >= 8
        && PCLNTAB_MAGICS.contains(&u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        && b[4] == 0
        && b[5] == 0
        && matches!(b[6], 1 | 2 | 4)
        && b[7] == 8
}

/// Candidate `pclntab` virtual addresses, best first: a region named
/// `.gopclntab` / `__gopclntab` whose head has the header shape, then a
/// bounded signature scan of the readable regions, each hit confirmed by
/// [`gopcln::recover_from`] actually yielding functions. At most
/// [`MAX_PCLNTAB_CANDIDATES`] VAs, deduplicated, order preserved.
fn locate_pclntab(mem: &Mem) -> Vec<u64> {
    /// Append `va` unless it is already a candidate.
    fn push(out: &mut Vec<u64>, va: u64) {
        if !out.contains(&va) {
            out.push(va);
        }
    }

    let mut out: Vec<u64> = Vec::new();
    for r in &mem.regions {
        if (r.name == ".gopclntab" || r.name == "__gopclntab")
            && let Some(b) = mem.region_bytes(r)
            && is_header_shape(b)
        {
            push(&mut out, r.va);
        }
    }
    for (base, data) in mem.readable_ranges() {
        if out.len() >= MAX_PCLNTAB_CANDIDATES {
            break;
        }
        let limit = data.len().min(MAX_SCAN_BYTES);
        let mut i = 0usize;
        while i + 8 <= limit {
            if is_header_shape(&data[i..]) && !gopcln::recover_from(&data[i..], 0).is_empty() {
                push(&mut out, base.wrapping_add(i as u64));
                if out.len() >= MAX_PCLNTAB_CANDIDATES {
                    break;
                }
            }
            i += 1;
        }
    }
    out.truncate(MAX_PCLNTAB_CANDIDATES);
    out
}

// ---------------------------------------------------------------------------
// Step 2 — locating and validating moduledata
// ---------------------------------------------------------------------------

/// Word indices, in pointer-sized units from the `moduledata` base, of the
/// fields recovery reads. Paired fields (`minpc`/`maxpc`, `text`/`etext`,
/// `types`/`etypes`) are addressed by the first of the pair; slices by
/// their `ptr` word, with `len` and `cap` following.
#[derive(Debug, Clone, Copy)]
struct Layout {
    minpc: u64,
    text: u64,
    types: u64,
    typelinks: u64,
    itablinks: u64,
}

/// The go1.18+ layouts this module accepts, tried in order.
///
/// Everything through `enoptrbss` is common: `pcHeader` (1 word), the six
/// slices `funcnametab, cutab, filetab, pctab, pclntable, ftab` (3 words
/// each), `findfunctab`, `minpc`, `maxpc`, then the ten section-bound
/// words `text, etext, noptrdata, enoptrdata, data, edata, bss, ebss,
/// noptrbss, enoptrbss` — putting the next field at word 32. From there
/// the layouts differ only in the `covctrs, ecovctrs` pair go1.20 added
/// ahead of `end, gcdata, gcbss, types, etypes, rodata, gofunc`,
/// `textsectmap`, `typelinks`, and `itablinks`.
const LAYOUTS: [Layout; 2] = [
    // go1.20+ — `covctrs, ecovctrs` present.
    Layout {
        minpc: 20,
        text: 22,
        types: 37,
        typelinks: 44,
        itablinks: 47,
    },
    // go1.18 / go1.19 — no coverage-counter pair.
    Layout {
        minpc: 20,
        text: 22,
        types: 35,
        typelinks: 42,
        itablinks: 45,
    },
];

/// The fields of a validated `moduledata` that recovery consumes.
#[derive(Debug, Clone, Copy)]
struct Moduledata {
    /// `moduledata.types`: the base every type and name offset is
    /// measured from.
    types: u64,
    typelinks_ptr: u64,
    typelinks_len: usize,
    itablinks_ptr: u64,
    itablinks_len: usize,
}

/// Scan the readable regions for a pointer-sized word equal to
/// `pclntab_va` — a candidate `moduledata.pcHeader` — and return the first
/// candidate that validates under some [`LAYOUTS`] entry.
///
/// The sweep is linear in region bytes (one aligned word per step) and
/// stops after [`MAX_MODULEDATA_CANDIDATES`] validation attempts.
fn find_moduledata(mem: &Mem, pclntab_va: u64) -> Option<Moduledata> {
    let mut probes = 0usize;
    for (base, data) in mem.readable_ranges() {
        // Align the sweep to a pointer boundary in VA space.
        let skip = (PTR - base % PTR) % PTR;
        let limit = data.len().min(MAX_SCAN_BYTES);
        let mut off = skip as usize;
        while off + 8 <= limit {
            let b = &data[off..off + 8];
            let word = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
            if word == pclntab_va {
                probes += 1;
                if probes > MAX_MODULEDATA_CANDIDATES {
                    return None;
                }
                let md_va = base.wrapping_add(off as u64);
                for layout in LAYOUTS {
                    if let Some(md) = validate(mem, md_va, layout) {
                        return Some(md);
                    }
                }
            }
            off += 8;
        }
    }
    None
}

/// Structurally validate the `moduledata` candidate at `md_va` under one
/// field layout, returning the fields recovery needs on success.
///
/// The candidate's word 0 already equals the `pclntab` VA by construction;
/// everything else the module docs list is checked here, ending with the
/// `typelinks` probe that distinguishes one layout from another.
fn validate(mem: &Mem, md_va: u64, layout: Layout) -> Option<Moduledata> {
    let word = |i: u64| -> Option<u64> { mem.u64_at(md_va.checked_add(i.checked_mul(PTR)?)?) };

    let (minpc, maxpc) = (word(layout.minpc)?, word(layout.minpc + 1)?);
    let (text, etext) = (word(layout.text)?, word(layout.text + 1)?);
    let (types, etypes) = (word(layout.types)?, word(layout.types + 1)?);
    let (tl_ptr, tl_len, tl_cap) = (
        word(layout.typelinks)?,
        word(layout.typelinks + 1)?,
        word(layout.typelinks + 2)?,
    );
    let (it_ptr, it_len, it_cap) = (
        word(layout.itablinks)?,
        word(layout.itablinks + 1)?,
        word(layout.itablinks + 2)?,
    );

    // Ordering of the module's address ranges.
    if minpc > maxpc || text > etext || types > etypes {
        return None;
    }
    // The code bounds must name code, and the type arena must be mapped.
    if !mem.is_exec(minpc) || !mem.is_exec(text) || !mem.is_readable(types) {
        return None;
    }
    // Slice headers: `len <= cap`, both within this module's caps, and a
    // mapped base whenever the slice is non-empty.
    if tl_len > tl_cap || tl_cap > MAX_TYPELINKS as u64 {
        return None;
    }
    if it_len > it_cap || it_cap > MAX_ITABS as u64 {
        return None;
    }
    if tl_len > 0 && !mem.is_readable(tl_ptr) {
        return None;
    }
    if it_len > 0 && !mem.is_readable(it_ptr) {
        return None;
    }
    // The discriminator: real `typelinks` resolve to real type records.
    if tl_len > 0 && !probe_typelinks(mem, types, tl_ptr, tl_len as usize) {
        return None;
    }
    Some(Moduledata {
        types,
        typelinks_ptr: tl_ptr,
        typelinks_len: tl_len as usize,
        itablinks_ptr: it_ptr,
        itablinks_len: it_len as usize,
    })
}

/// Whether at least one of the first [`MAX_TYPELINK_PROBES`] `typelinks`
/// entries resolves to a well-formed type record. Cheap, and decisive: an
/// array of anything else (pointers, counters, unrelated offsets) read as
/// `types`-relative `int32`s essentially never lands on a record with a
/// legal kind and a printable, length-prefixed name.
fn probe_typelinks(mem: &Mem, types: u64, tl_ptr: u64, tl_len: usize) -> bool {
    (0..tl_len.min(MAX_TYPELINK_PROBES))
        .filter_map(|i| typelink_target(mem, types, tl_ptr, i))
        .any(|va| read_type(mem, types, va).is_some())
}

/// The VA of the type record named by `typelinks[i]`, or `None` when the
/// entry is unreadable or its offset is negative or out of range.
fn typelink_target(mem: &Mem, types: u64, tl_ptr: u64, i: usize) -> Option<u64> {
    let slot = tl_ptr.checked_add((i as u64).checked_mul(4)?)?;
    let off = mem.i32_at(slot)?;
    if off < 0 {
        return None;
    }
    types.checked_add(off as u64)
}

// ---------------------------------------------------------------------------
// Steps 3 and 4 — type records, names, and itabs
// ---------------------------------------------------------------------------

/// Walk a validated `moduledata`'s `typelinks` and `itablinks`, then put
/// the results in canonical order.
fn collect(mem: &Mem, md: &Moduledata) -> GoTypes {
    let mut types = Vec::new();
    for i in 0..md.typelinks_len.min(MAX_TYPELINKS) {
        let Some(va) = typelink_target(mem, md.types, md.typelinks_ptr, i) else {
            continue;
        };
        if let Some(t) = read_type(mem, md.types, va) {
            types.push(t);
        }
    }

    let mut itabs = Vec::new();
    for i in 0..md.itablinks_len.min(MAX_ITABS) {
        let Some(slot) = (i as u64)
            .checked_mul(PTR)
            .and_then(|d| md.itablinks_ptr.checked_add(d))
        else {
            break;
        };
        let Some(va) = mem.u64_at(slot) else {
            break;
        };
        if let Some(it) = read_itab(mem, md.types, va) {
            itabs.push(it);
        }
    }

    types.sort_by(|a, b| a.va.cmp(&b.va).then_with(|| a.name.cmp(&b.name)));
    types.dedup_by_key(|t| t.va);
    itabs.sort_by(|a, b| {
        a.va.cmp(&b.va)
            .then_with(|| a.interface_name.cmp(&b.interface_name))
            .then_with(|| a.type_name.cmp(&b.type_name))
    });
    itabs.dedup_by_key(|i| i.va);
    GoTypes { types, itabs }
}

/// The kind name for a `Type.Kind` byte, or `None` for `Invalid` (0) and
/// for the unassigned values above `unsafe.Pointer`. Only the low five
/// bits select the kind; the rest are `KindDirectIface` / `KindGCProg`.
fn kind_name(kind: u8) -> Option<&'static str> {
    let k = (kind & KIND_MASK) as usize;
    if k == 0 {
        return None;
    }
    KIND_NAMES.get(k).copied()
}

/// Parse the `internal/abi.Type` record at `va`, with `types` as the base
/// its `Str` name offset is measured from.
///
/// `None` when the record is unreadable, its kind is not a real kind, or
/// its name does not decode — which is exactly what makes this function
/// usable as [`probe_typelinks`]'s structural test.
fn read_type(mem: &Mem, types: u64, va: u64) -> Option<GoType> {
    let rec = mem.bytes_at(va, TYPE_SIZE)?;
    let tflag = rec[TYPE_TFLAG];
    let kind = kind_name(rec[TYPE_KIND])?;
    let str_off = i32::from_le_bytes([
        rec[TYPE_STR],
        rec[TYPE_STR + 1],
        rec[TYPE_STR + 2],
        rec[TYPE_STR + 3],
    ]);
    if str_off < 0 {
        return None;
    }
    let mut name = read_name(mem, types.checked_add(str_off as u64)?)?;
    // `TFlagExtraStar`: the string is shared with the pointer-to type, so
    // the leading `*` belongs to that one, not to this record.
    if tflag & TFLAG_EXTRA_STAR != 0 && name.starts_with('*') {
        name.remove(0);
    }
    if name.is_empty() {
        return None;
    }
    Some(GoType { va, name, kind })
}

/// Decode a go1.17+ encoded name at `va`: a flag byte (bit 0 exported,
/// bit 1 tag data follows, bit 2 embedded), a uvarint byte length, then
/// that many bytes of UTF-8.
///
/// `None` unless every part reads in bounds and the result is a plausible
/// name: flags confined to the low nibble, a non-zero length no greater
/// than [`MAX_NAME_LEN`], and no control bytes in the body. Go type names
/// are printable by construction (`[]int`, `map[string]main.T`, …), so the
/// last test costs nothing real and rejects a great deal of garbage.
fn read_name(mem: &Mem, va: u64) -> Option<String> {
    let flag = mem.u8_at(va)?;
    if flag & 0xf0 != 0 {
        return None;
    }
    let (len, used) = read_uvarint(mem, va.checked_add(1)?)?;
    let len = usize::try_from(len).ok()?;
    if len == 0 || len > MAX_NAME_LEN {
        return None;
    }
    let body_at = va.checked_add(1)?.checked_add(used as u64)?;
    let body = mem.bytes_at(body_at, len)?;
    if body.iter().any(|&b| b < 0x20 || b == 0x7f) {
        return None;
    }
    Some(String::from_utf8_lossy(body).into_owned())
}

/// Read Go's varint encoding (seven bits per byte, least significant
/// group first, high bit continuing) at `va`, returning the value and the
/// bytes consumed.
///
/// `None` when a byte is unreadable, when the encoding runs past
/// [`MAX_UVARINT_BYTES`], or when the value exceeds `2^32` — no name is
/// that long, and the bound keeps the shift and the caller's arithmetic
/// far from overflow.
fn read_uvarint(mem: &Mem, va: u64) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    for i in 0..MAX_UVARINT_BYTES {
        let byte = mem.u8_at(va.checked_add(i as u64)?)?;
        value |= u64::from(byte & 0x7f) << (7 * i);
        if value > u64::from(u32::MAX) {
            return None;
        }
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

/// Parse the `itab` at `va`: both type names through [`read_type`], then
/// the `fun` array while its entries point into executable memory.
///
/// `None` unless `inter` resolves to a record whose kind really is
/// `interface` and `_type` resolves to any well-formed record — the two
/// checks `runtime/iface.go`'s own invariants guarantee.
fn read_itab(mem: &Mem, types: u64, va: u64) -> Option<Itab> {
    let inter_va = mem.u64_at(va)?;
    let type_va = mem.u64_at(va.checked_add(ITAB_TYPE)?)?;
    let inter = read_type(mem, types, inter_va)?;
    if inter.kind != KIND_NAMES[KIND_INTERFACE as usize] {
        return None;
    }
    let concrete = read_type(mem, types, type_va)?;

    let mut methods = Vec::new();
    for i in 0..MAX_ITAB_METHODS {
        let Some(slot) = (i as u64)
            .checked_mul(PTR)
            .and_then(|d| va.checked_add(ITAB_FUN)?.checked_add(d))
        else {
            break;
        };
        let Some(entry) = mem.u64_at(slot) else {
            break;
        };
        // The array is not length-prefixed; it ends where it stops
        // naming code.
        if !mem.is_exec(entry) {
            break;
        }
        methods.push(entry);
    }
    Some(Itab {
        va,
        interface_name: inter.name,
        type_name: concrete.name,
        methods,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::tests::synthetic_elf64;
    use crate::model::{Arch, ImportSlot, Perms, Symbol};

    // ---- Synthetic image fixture ----------------------------------------

    const IMAGE_LEN: usize = 0x8000;
    const TEXT_VA: u64 = 0x1000;
    const ETEXT_VA: u64 = 0x1f00;
    const RODATA_VA: u64 = 0x2000;
    const DATA_VA: u64 = 0x4000;
    const PCLNTAB_VA: u64 = 0x2000;
    const PCLNTAB_LEN: u64 = 104;
    const TYPES_VA: u64 = 0x2400;
    const ETYPES_VA: u64 = 0x2800;
    const TYPE0_VA: u64 = 0x2400;
    const TYPE1_VA: u64 = 0x2440;
    const IFACE_VA: u64 = 0x2480;
    const NAME0_OFF: i32 = 0x100;
    const NAME1_OFF: i32 = 0x120;
    const NAME2_OFF: i32 = 0x140;
    const TYPELINKS_VA: u64 = 0x2900;
    const MD_VA: u64 = 0x4000;
    const ITABLINKS_VA: u64 = 0x4200;
    const ITAB_VA: u64 = 0x4300;
    const METHOD0_VA: u64 = 0x1100;
    const METHOD1_VA: u64 = 0x1180;

    /// Identity-mapped image (VA == file offset) over a byte buffer, the
    /// same idiom [`crate::gopcln`]'s tests use.
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
            let off = usize::try_from(va).ok()?;
            (off <= self.data.len()).then_some(off)
        }
        fn bytes(&self) -> &[u8] {
            &self.data
        }
    }

    fn region(name: &str, va: u64, size: u64, r: bool, w: bool, x: bool) -> Region {
        Region {
            name: name.into(),
            va,
            size,
            perms: Perms { r, w, x },
        }
    }

    fn w64(buf: &mut [u8], at: u64, v: u64) {
        let at = at as usize;
        buf[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }

    fn w32(buf: &mut [u8], at: u64, v: u32) {
        let at = at as usize;
        buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn wbytes(buf: &mut [u8], at: u64, b: &[u8]) {
        let at = at as usize;
        buf[at..at + b.len()].copy_from_slice(b);
    }

    /// Encode a go1.17+ name: flag byte, uvarint length, bytes.
    fn encode_name(flag: u8, s: &str) -> Vec<u8> {
        let mut out = vec![flag];
        let mut n = s.len();
        loop {
            let mut b = (n & 0x7f) as u8;
            n >>= 7;
            if n != 0 {
                b |= 0x80;
            }
            out.push(b);
            if n == 0 {
                break;
            }
        }
        out.extend_from_slice(s.as_bytes());
        out
    }

    /// A minimal but genuinely parseable go1.18 `pclntab` (one function,
    /// `main` at `textStart + 0x100`), so the locate step's confirmation
    /// through [`gopcln::recover_from`] succeeds.
    fn write_pclntab(buf: &mut [u8]) {
        let t = PCLNTAB_VA;
        w32(buf, t, 0xffff_fff0); // magic (go1.18)
        buf[(t + 4) as usize] = 0; // pad
        buf[(t + 5) as usize] = 0; // pad
        buf[(t + 6) as usize] = 1; // pcquantum
        buf[(t + 7) as usize] = 8; // ptrsize
        w64(buf, t + 8, 1); // nfunc
        w64(buf, t + 16, 0); // nfiles
        w64(buf, t + 24, TEXT_VA); // textStart
        w64(buf, t + 32, 72); // funcnametab
        w64(buf, t + 40, 0); // cutab
        w64(buf, t + 48, 0); // filetab
        w64(buf, t + 56, 0); // pctab
        w64(buf, t + 64, 80); // pclntab (functab)
        wbytes(buf, t + 72, b"main\0");
        // functab: one (entryoff, funcoff) pair, then the sentinel.
        w32(buf, t + 80, 0x100);
        w32(buf, t + 84, 16); // _func at functab+16 = table+96
        w32(buf, t + 88, 0xf00);
        w32(buf, t + 92, 0);
        // _func: entryOff, nameOff (relative to funcnametab).
        w32(buf, t + 96, 0x100);
        w32(buf, t + 100, 0);
    }

    /// Write an `internal/abi.Type` record.
    fn write_type(buf: &mut [u8], at: u64, tflag: u8, kind: u8, str_off: i32) {
        w64(buf, at, 16); // Size
        w64(buf, at + 8, 0); // PtrBytes
        w32(buf, at + 16, 0x1111_2222); // Hash
        buf[(at + 20) as usize] = tflag;
        buf[(at + 21) as usize] = 8; // Align
        buf[(at + 22) as usize] = 8; // FieldAlign
        buf[(at + 23) as usize] = kind;
        w64(buf, at + 24, 0); // Equal
        w64(buf, at + 32, 0); // GCData
        w32(buf, at + 40, str_off as u32); // Str
        w32(buf, at + 44, 0); // PtrToThis
    }

    /// Word index of `moduledata` fields, for the layout with (`true`) or
    /// without (`false`) go1.20's `covctrs, ecovctrs` pair.
    struct MdIndex {
        types: u64,
        typelinks: u64,
        itablinks: u64,
    }

    fn md_index(covctrs: bool) -> MdIndex {
        // Word 32 is the first field after `enoptrbss`.
        let end = if covctrs { 34 } else { 32 };
        let types = end + 3; // end, gcdata, gcbss
        let textsectmap = types + 2 + 2; // types, etypes, rodata, gofunc
        MdIndex {
            types,
            typelinks: textsectmap + 3,
            itablinks: textsectmap + 6,
        }
    }

    fn write_moduledata(buf: &mut [u8], covctrs: bool) {
        let idx = md_index(covctrs);
        let w = |i: u64| MD_VA + i * PTR;
        w64(buf, w(0), PCLNTAB_VA); // pcHeader
        w64(buf, w(20), TEXT_VA); // minpc
        w64(buf, w(21), ETEXT_VA); // maxpc
        w64(buf, w(22), TEXT_VA); // text
        w64(buf, w(23), ETEXT_VA); // etext
        w64(buf, w(idx.types), TYPES_VA);
        w64(buf, w(idx.types + 1), ETYPES_VA);
        w64(buf, w(idx.typelinks), TYPELINKS_VA);
        w64(buf, w(idx.typelinks + 1), 2);
        w64(buf, w(idx.typelinks + 2), 2);
        w64(buf, w(idx.itablinks), ITABLINKS_VA);
        w64(buf, w(idx.itablinks + 1), 1);
        w64(buf, w(idx.itablinks + 2), 1);
    }

    /// The full fixture: a `pclntab`, a `moduledata`, two named types
    /// (the second carrying `TFlagExtraStar`), an interface type, and one
    /// `itab` with two text-resident methods.
    fn go_image(covctrs: bool) -> FakeImage {
        let mut data = vec![0u8; IMAGE_LEN];
        write_pclntab(&mut data);

        // Type records and their names, all inside [types, etypes).
        write_type(&mut data, TYPE0_VA, 0x04, 25, NAME0_OFF); // struct
        write_type(&mut data, TYPE1_VA, 0x06, 25, NAME1_OFF); // struct, extra *
        write_type(&mut data, IFACE_VA, 0x04, 20, NAME2_OFF); // interface
        wbytes(
            &mut data,
            TYPES_VA + NAME0_OFF as u64,
            &encode_name(0x01, "main.Widget"),
        );
        wbytes(
            &mut data,
            TYPES_VA + NAME1_OFF as u64,
            &encode_name(0x01, "*main.Config"),
        );
        wbytes(
            &mut data,
            TYPES_VA + NAME2_OFF as u64,
            &encode_name(0x01, "main.Stringer"),
        );

        // typelinks: int32 offsets from `types` to the two named types.
        w32(&mut data, TYPELINKS_VA, 0x00);
        w32(&mut data, TYPELINKS_VA + 4, 0x40);

        write_moduledata(&mut data, covctrs);

        // itablinks -> one itab: main.Stringer <- main.Widget.
        w64(&mut data, ITABLINKS_VA, ITAB_VA);
        w64(&mut data, ITAB_VA, IFACE_VA);
        w64(&mut data, ITAB_VA + ITAB_TYPE, TYPE0_VA);
        w32(&mut data, ITAB_VA + 16, 0x2222_3333); // hash
        w64(&mut data, ITAB_VA + ITAB_FUN, METHOD0_VA);
        w64(&mut data, ITAB_VA + ITAB_FUN + 8, METHOD1_VA);
        // The next slot stays zero, which is not code: the walk stops.

        // Plant a `ret` at each method so .text is not all zeros.
        data[METHOD0_VA as usize] = 0xc3;
        data[METHOD1_VA as usize] = 0xc3;

        FakeImage {
            data,
            regions: vec![
                region(".text", TEXT_VA, 0x1000, true, false, true),
                region(".rodata", RODATA_VA, 0x2000, true, false, false),
                region(".noptrdata", DATA_VA, 0x1000, true, true, false),
            ],
        }
    }

    fn expected() -> GoTypes {
        GoTypes {
            types: vec![
                GoType {
                    va: TYPE0_VA,
                    name: "main.Widget".into(),
                    kind: "struct",
                },
                GoType {
                    va: TYPE1_VA,
                    name: "main.Config".into(),
                    kind: "struct",
                },
            ],
            itabs: vec![Itab {
                va: ITAB_VA,
                interface_name: "main.Stringer".into(),
                type_name: "main.Widget".into(),
                methods: vec![METHOD0_VA, METHOD1_VA],
            }],
        }
    }

    /// A one-region readable image over `bytes`, mapped at VA 0 — enough
    /// to exercise the name and uvarint readers directly.
    fn flat_image(bytes: Vec<u8>) -> FakeImage {
        let size = bytes.len() as u64;
        FakeImage {
            data: bytes,
            regions: vec![region("flat", 0, size, true, false, false)],
        }
    }

    // ---- Happy path ------------------------------------------------------

    #[test]
    fn recovers_types_and_itabs_from_a_synthetic_module() {
        let img = go_image(true);
        assert_eq!(recover(&img), expected());
    }

    #[test]
    fn recovers_the_layout_without_the_covctrs_pair() {
        let img = go_image(false);
        assert_eq!(recover(&img), expected());
    }

    #[test]
    fn recovers_when_the_pclntab_sits_in_a_named_region() {
        let mut img = go_image(true);
        img.regions.push(region(
            ".gopclntab",
            PCLNTAB_VA,
            PCLNTAB_LEN,
            true,
            false,
            false,
        ));
        img.regions.sort_by_key(|r| (r.va, r.size));
        assert_eq!(recover(&img), expected());
    }

    #[test]
    fn recovery_is_deterministic() {
        let img = go_image(true);
        assert_eq!(recover(&img), recover(&img));
        assert_eq!(render(&recover(&img)), render(&recover(&img)));
    }

    // ---- Kinds -----------------------------------------------------------

    #[test]
    fn kind_table_matches_the_reflect_vocabulary() {
        for (k, want) in [
            (1u8, "bool"),
            (2, "int"),
            (6, "int64"),
            (11, "uint64"),
            (12, "uintptr"),
            (13, "float32"),
            (16, "complex128"),
            (17, "array"),
            (18, "chan"),
            (19, "func"),
            (20, "interface"),
            (21, "map"),
            (22, "ptr"),
            (23, "slice"),
            (24, "string"),
            (25, "struct"),
            (26, "unsafe.Pointer"),
        ] {
            assert_eq!(kind_name(k), Some(want), "kind {k}");
            // The high three bits are KindDirectIface / KindGCProg flags
            // and must not disturb the mapping.
            assert_eq!(kind_name(k | 0xe0), Some(want), "flagged kind {k}");
        }
        assert_eq!(kind_name(0), None); // Invalid
        for k in 27..=31u8 {
            assert_eq!(kind_name(k), None, "unassigned kind {k}");
        }
    }

    #[test]
    fn a_type_records_kind_reaches_the_result() {
        let mut img = go_image(true);
        write_type(&mut img.data, TYPE1_VA, 0x04, 23, NAME0_OFF); // slice
        let out = recover(&img);
        assert_eq!(out.types[1].kind, "slice");
        assert_eq!(out.types[1].name, "main.Widget");
    }

    // ---- Name decoding ---------------------------------------------------

    #[test]
    fn name_uvarint_edge_cases_decode_or_refuse() {
        // One-byte length.
        let img = flat_image(encode_name(0x01, "abc"));
        assert_eq!(read_name(&Mem::new(&img), 0).as_deref(), Some("abc"));

        // Two-byte length (200 > 0x7f forces a continuation byte).
        let long = "x".repeat(200);
        let bytes = encode_name(0x00, &long);
        assert_eq!(bytes[1], 0xc8, "low group");
        assert_eq!(bytes[2], 0x01, "high group");
        let img = flat_image(bytes);
        assert_eq!(read_name(&Mem::new(&img), 0).as_deref(), Some(&long[..]));

        // Exactly at the cap, and one past it.
        for (len, want) in [(MAX_NAME_LEN, true), (MAX_NAME_LEN + 1, false)] {
            let img = flat_image(encode_name(0x00, &"y".repeat(len)));
            assert_eq!(read_name(&Mem::new(&img), 0).is_some(), want, "len {len}");
        }

        // Zero length.
        let img = flat_image(encode_name(0x00, ""));
        assert_eq!(read_name(&Mem::new(&img), 0), None);

        // Continuation bytes forever: refused at MAX_UVARINT_BYTES.
        let img = flat_image(vec![0x00; 64]);
        let mut all_set = vec![0x00u8];
        all_set.extend(std::iter::repeat_n(0x80u8, 63));
        assert_eq!(read_name(&Mem::new(&flat_image(all_set)), 0), None);
        assert_eq!(read_name(&Mem::new(&img), 0), None); // length 0

        // Truncated body: the length promises more than the image holds.
        let mut short = encode_name(0x01, "abcdef");
        short.truncate(4);
        assert_eq!(read_name(&Mem::new(&flat_image(short)), 0), None);

        // Reserved flag bits.
        let img = flat_image(encode_name(0xf0, "abc"));
        assert_eq!(read_name(&Mem::new(&img), 0), None);

        // Control bytes in the body.
        let mut ctrl = encode_name(0x01, "ab");
        let last = ctrl.len() - 1;
        ctrl[last] = 0x07;
        assert_eq!(read_name(&Mem::new(&flat_image(ctrl)), 0), None);

        // Nothing there at all.
        assert_eq!(read_name(&Mem::new(&flat_image(Vec::new())), 0), None);
    }

    #[test]
    fn uvarint_refuses_values_above_four_gigabytes() {
        // 0xffffffff encodes in five bytes and is accepted; one more bit
        // is not.
        let img = flat_image(vec![0xff, 0xff, 0xff, 0xff, 0x0f]);
        assert_eq!(read_uvarint(&Mem::new(&img), 0), Some((0xffff_ffff, 5)));
        let img = flat_image(vec![0xff, 0xff, 0xff, 0xff, 0x1f]);
        assert_eq!(read_uvarint(&Mem::new(&img), 0), None);
    }

    #[test]
    fn extra_star_flag_strips_exactly_one_leading_star() {
        let img = go_image(true);
        // TYPE1 stores "*main.Config" with TFlagExtraStar.
        assert_eq!(recover(&img).types[1].name, "main.Config");

        // Same stored string without the flag keeps the star.
        let mut img = go_image(true);
        write_type(&mut img.data, TYPE1_VA, 0x04, 25, NAME1_OFF);
        assert_eq!(recover(&img).types[1].name, "*main.Config");

        // The flag set but no star present: the name is unchanged.
        let mut img = go_image(true);
        write_type(&mut img.data, TYPE1_VA, 0x06, 25, NAME0_OFF);
        assert_eq!(recover(&img).types[1].name, "main.Widget");
    }

    // ---- Refusals --------------------------------------------------------

    #[test]
    fn a_truncated_moduledata_is_refused() {
        // Cut the buffer (and the owning region) just past `pcHeader`, so
        // the slice headers are not file-backed.
        let mut img = go_image(true);
        img.data.truncate((MD_VA + 0x40) as usize);
        img.regions[2].size = 0x40;
        assert_eq!(recover(&img), GoTypes::default());
    }

    #[test]
    fn typelinks_len_greater_than_cap_is_refused() {
        let idx = md_index(true);
        let mut img = go_image(true);
        w64(&mut img.data, MD_VA + (idx.typelinks + 1) * PTR, 3); // len
        w64(&mut img.data, MD_VA + (idx.typelinks + 2) * PTR, 2); // cap
        assert_eq!(recover(&img), GoTypes::default());
    }

    #[test]
    fn itablinks_len_greater_than_cap_is_refused() {
        let idx = md_index(true);
        let mut img = go_image(true);
        w64(&mut img.data, MD_VA + (idx.itablinks + 1) * PTR, 9); // len
        w64(&mut img.data, MD_VA + (idx.itablinks + 2) * PTR, 1); // cap
        assert_eq!(recover(&img), GoTypes::default());
    }

    #[test]
    fn oversized_slice_counts_are_refused_not_walked() {
        let idx = md_index(true);
        for len in [MAX_TYPELINKS as u64 + 1, 1 << 40, u64::MAX] {
            let mut img = go_image(true);
            w64(&mut img.data, MD_VA + (idx.typelinks + 1) * PTR, len);
            w64(&mut img.data, MD_VA + (idx.typelinks + 2) * PTR, len);
            assert_eq!(recover(&img), GoTypes::default(), "typelinks len {len}");
        }
        for len in [MAX_ITABS as u64 + 1, 1 << 40, u64::MAX] {
            let mut img = go_image(true);
            w64(&mut img.data, MD_VA + (idx.itablinks + 1) * PTR, len);
            w64(&mut img.data, MD_VA + (idx.itablinks + 2) * PTR, len);
            assert_eq!(recover(&img), GoTypes::default(), "itablinks len {len}");
        }
    }

    #[test]
    fn a_name_offset_out_of_range_skips_only_that_type() {
        for bad in [0x7fff_0000u32, 0xffff_ffff /* negative int32 */] {
            let mut img = go_image(true);
            w32(&mut img.data, TYPE1_VA + 40, bad);
            let out = recover(&img);
            assert_eq!(out.types.len(), 1, "str {bad:#x}");
            assert_eq!(out.types[0].name, "main.Widget");
            assert_eq!(out.itabs.len(), 1);
        }
    }

    #[test]
    fn a_typelink_offset_out_of_range_skips_only_that_entry() {
        for bad in [0x7fff_0000u32, 0xffff_ffff] {
            let mut img = go_image(true);
            w32(&mut img.data, TYPELINKS_VA + 4, bad);
            let out = recover(&img);
            assert_eq!(out.types.len(), 1, "typelink {bad:#x}");
            assert_eq!(out.types[0].va, TYPE0_VA);
        }
    }

    #[test]
    fn an_itab_whose_interface_is_not_an_interface_is_dropped() {
        let mut img = go_image(true);
        write_type(&mut img.data, IFACE_VA, 0x04, 25, NAME2_OFF); // struct
        let out = recover(&img);
        assert_eq!(out.types.len(), 2);
        assert!(out.itabs.is_empty());
    }

    #[test]
    fn itab_methods_stop_at_the_first_non_code_entry() {
        // A data-resident third entry ends the array.
        let mut img = go_image(true);
        w64(&mut img.data, ITAB_VA + ITAB_FUN + 16, ITAB_VA);
        assert_eq!(recover(&img).itabs[0].methods, vec![METHOD0_VA, METHOD1_VA]);

        // A non-code first entry yields no methods at all.
        let mut img = go_image(true);
        w64(&mut img.data, ITAB_VA + ITAB_FUN, RODATA_VA);
        assert!(recover(&img).itabs[0].methods.is_empty());
    }

    #[test]
    fn a_wrong_pcheader_pointer_finds_no_moduledata() {
        let mut img = go_image(true);
        w64(&mut img.data, MD_VA, PCLNTAB_VA + 8);
        assert_eq!(recover(&img), GoTypes::default());
    }

    #[test]
    fn unmapped_or_disordered_module_bounds_are_refused() {
        let idx = md_index(true);
        // minpc > maxpc.
        let mut img = go_image(true);
        w64(&mut img.data, MD_VA + 21 * PTR, TEXT_VA - 1);
        assert!(recover(&img).is_empty());
        // text not executable.
        let mut img = go_image(true);
        w64(&mut img.data, MD_VA + 22 * PTR, RODATA_VA);
        assert!(recover(&img).is_empty());
        // types unmapped.
        let mut img = go_image(true);
        w64(&mut img.data, MD_VA + idx.types * PTR, 0xdead_0000);
        assert!(recover(&img).is_empty());
        // types > etypes.
        let mut img = go_image(true);
        w64(&mut img.data, MD_VA + (idx.types + 1) * PTR, TYPES_VA - 1);
        assert!(recover(&img).is_empty());
        // typelinks base unmapped.
        let mut img = go_image(true);
        w64(&mut img.data, MD_VA + idx.typelinks * PTR, 0xdead_0000);
        assert!(recover(&img).is_empty());
    }

    #[test]
    fn a_non_go_image_yields_nothing() {
        let elf = synthetic_elf64();
        let img = crate::model::ElfImage::parse(&elf).unwrap();
        assert_eq!(recover(&img), GoTypes::default());
        assert!(recover(&img).is_empty());
    }

    #[test]
    fn an_image_with_no_regions_yields_nothing() {
        let img = FakeImage {
            data: vec![0u8; 0x100],
            regions: Vec::new(),
        };
        assert!(recover(&img).is_empty());
    }

    // ---- Adversarial -----------------------------------------------------

    #[test]
    fn garbage_at_every_moduledata_word_never_panics() {
        let idx = md_index(true);
        let words = idx.itablinks + 3;
        for w in 0..words {
            for fill in [0x0000_0000_0000_0000u64, u64::MAX, PCLNTAB_VA, MD_VA] {
                let mut img = go_image(true);
                w64(&mut img.data, MD_VA + w * PTR, fill);
                let out = recover(&img);
                assert!(out.types.len() <= 2, "word {w} fill {fill:#x}");
                assert!(out.itabs.len() <= 1, "word {w} fill {fill:#x}");
            }
        }
    }

    #[test]
    fn garbage_at_every_type_record_byte_never_panics() {
        for at in [TYPE0_VA, TYPE1_VA, IFACE_VA] {
            for i in 0..TYPE_SIZE as u64 {
                let mut img = go_image(true);
                img.data[(at + i) as usize] = 0xff;
                let out = recover(&img);
                assert!(out.types.len() <= 2, "type {at:#x} byte {i}");
            }
        }
    }

    #[test]
    fn garbage_at_every_itab_byte_never_panics() {
        for i in 0..(ITAB_FUN + 24) {
            let mut img = go_image(true);
            img.data[(ITAB_VA + i) as usize] = 0xff;
            let out = recover(&img);
            assert_eq!(out.types.len(), 2, "itab byte {i}");
            assert!(out.itabs.len() <= 1, "itab byte {i}");
        }
    }

    #[test]
    fn self_referential_offsets_terminate() {
        // typelinks pointing at the moduledata, types rebased onto it, and
        // an itab whose halves point at itself: every read still resolves,
        // and nothing recurses.
        let idx = md_index(true);
        let mut img = go_image(true);
        w64(&mut img.data, MD_VA + idx.types * PTR, MD_VA);
        w64(&mut img.data, MD_VA + (idx.types + 1) * PTR, MD_VA);
        w64(&mut img.data, MD_VA + idx.typelinks * PTR, MD_VA);
        w64(&mut img.data, ITABLINKS_VA, ITABLINKS_VA);
        w64(&mut img.data, ITAB_VA, ITAB_VA);
        w64(&mut img.data, ITAB_VA + ITAB_TYPE, ITAB_VA);
        let out = recover(&img);
        assert!(out.types.len() <= 2);
        assert!(out.itabs.len() <= 1);

        // A typelink whose offset lands back on the moduledata's own
        // pcHeader word is likewise inert.
        let mut img = go_image(true);
        w32(&mut img.data, TYPELINKS_VA, 0);
        w32(&mut img.data, TYPELINKS_VA + 4, 0);
        assert_eq!(recover(&img).types.len(), 1); // deduped by VA
    }

    #[test]
    fn fixed_seed_mutation_fuzz_is_bounded_and_panic_free() {
        // A tiny LCG (fixed seed) scribbles over the metadata-bearing
        // ranges of the fixture. The point is that nothing panics and the
        // output stays bounded, not that anything is recovered.
        let mut state: u64 = 0x5eed_1234_abcd_0f0fu64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut total = 0usize;
        for _ in 0..400 {
            let mut img = go_image(next() % 2 == 0);
            for _ in 0..24 {
                let at = match next() % 4 {
                    0 => MD_VA + u64::from(next() % 0x180),
                    1 => TYPES_VA + u64::from(next() % 0x200),
                    2 => ITAB_VA + u64::from(next() % 0x40),
                    _ => TYPELINKS_VA + u64::from(next() % 0x10),
                };
                img.data[at as usize] = next() as u8;
            }
            let out = recover(&img);
            total = total
                .saturating_add(out.types.len())
                .saturating_add(out.itabs.len());
            let _ = render(&out);
        }
        assert!(total < 1_000_000, "unbounded fuzz output: {total}");
    }

    #[test]
    fn fully_random_images_never_panic() {
        let mut state: u64 = 0xfeed_face_dead_beefu64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for _ in 0..200 {
            let len = 0x400 + (next() as usize % 0x400);
            let data: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            let size = len as u64;
            let img = FakeImage {
                data,
                regions: vec![
                    region(".text", 0, size / 2, true, false, true),
                    region(".data", size / 2, size / 2, true, true, false),
                ],
            };
            let _ = render(&recover(&img));
        }
    }

    // ---- Rendering -------------------------------------------------------

    #[test]
    fn render_is_readable_and_newline_terminated() {
        let out = render(&expected());
        assert_eq!(
            out,
            concat!(
                "Go type metadata: 2 types, 1 itabs\n",
                "  0x0000000000002400  struct          main.Widget\n",
                "  0x0000000000002440  struct          main.Config\n",
                "\nitabs\n",
                "  0x0000000000004300  main.Stringer <- main.Widget  [2 methods]\n",
            )
        );
        assert!(out.ends_with('\n'));
    }

    #[test]
    fn render_of_an_empty_result_explains_itself() {
        assert_eq!(
            render(&GoTypes::default()),
            "no Go type metadata recovered\n"
        );
    }
}
