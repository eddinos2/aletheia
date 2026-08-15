//! C++ structure recovery — vtables, RTTI, and the class hierarchy.
//!
//! Recovers Itanium C++ ABI runtime structures from a loaded image:
//! vtables (offset-to-top, typeinfo slot, virtual function pointers),
//! `std::type_info` objects and their `_ZTS` name strings, and the
//! inheritance edges encoded by `__si_class_type_info` /
//! `__vmi_class_type_info`. Symbol-driven when `_ZTV`/`_ZTI` symbols
//! survive, heuristic data-section scan when stripped. Best-effort by
//! contract: absence of RTTI yields an empty result, and no input ever
//! panics.
//!
//! # Layouts (Itanium C++ ABI §2.9, §2.5.2–2.5.3)
//!
//! Every layout below is 64-bit and little-endian; pointers are 8 bytes.
//!
//! **Typeinfo objects.** Every `std::type_info` derivative begins with a
//! vptr (the address point of the *abi class's own* vtable — one of
//! `__class_type_info`, `__si_class_type_info`, `__vmi_class_type_info`)
//! followed by a pointer to the NUL-terminated mangled type name (the
//! `_ZTS` payload — a bare `<type>` encoding such as `6Widget` or
//! `N3foo3BarE`). The derived layouts add:
//!
//! - `__class_type_info` — nothing; the class has no bases.
//! - `__si_class_type_info` — `[base: pointer to typeinfo]`, one public,
//!   non-virtual base at offset 0.
//! - `__vmi_class_type_info` — `[flags: u32][base_count: u32]` then
//!   `base_count` × `[base: pointer][offset_flags: 8 bytes]`, where the
//!   low 8 bits of `offset_flags` are flags (`0x1` virtual, `0x2` public)
//!   and the arithmetic shift `offset_flags >> 8` is the signed offset of
//!   that base within the derived object.
//!
//! **Vtables.** A `_ZTV` object is `[offset-to-top: i64][typeinfo:
//! pointer]` followed by virtual function pointers. Under multiple
//! inheritance a single `_ZTV` symbol covers several consecutive
//! sub-vtables: a fresh `[offset-to-top][typeinfo]` header appears inside
//! the object, and because every sub-vtable of one class repeats *that
//! class's* typeinfo pointer, a repeat of the leading typeinfo VA at a
//! negative offset-to-top is what splits the groups here. A slot counts as
//! a function pointer only if it targets an executable region, or is zero
//! (pure-virtual and offset entries) — zeros are kept so slot indices stay
//! meaningful.
//!
//! # Two tiers, with provenance
//!
//! 1. [`Provenance::Symbol`] — `_ZTV*`/`_ZTI*`/`_ZTS*` symbols (and the
//!    Mach-O `__ZTV*` spelling, whose extra leading underscore is stripped
//!    exactly as the demanglers strip it) locate the objects exactly. A
//!    `_ZTV` object's extent runs to the next `_ZTV` symbol or the end of
//!    its region.
//! 2. [`Provenance::Scan`] — for stripped images: a conservative sweep of
//!    readable non-executable regions for typeinfo-shaped pairs, each of
//!    which must then be *cross-validated* by another candidate naming it
//!    as a base or by a vtable-shaped object naming it in its typeinfo
//!    slot. A candidate that fails cross-validation is dropped, never
//!    reported as a guess. The scan runs only when the symbol tier found
//!    nothing: where `_ZTI` symbols exist they are complete by
//!    construction, so a scan could only add noise.
//!
//! # Best-effort contract
//!
//! [`recover`] never errors and never panics on any input. Every read goes
//! through the bounds-checked [`Mem`] view over [`Image`]; every offset is
//! checked arithmetic; a body that does not parse costs its class its
//! *bases*, or the class itself, never the run. Attacker-controlled counts
//! are capped: [`MAX_BASES`] per class, [`MAX_SLOTS`] per sub-vtable,
//! [`MAX_SUBTABLES`] per vtable object, [`MAX_CLASSES`] overall,
//! [`MAX_NAME_LEN`] per name, and [`MAX_SCAN_BYTES`] per scanned region.
//! Base edges are recorded as typeinfo VAs and resolved by lookup, never
//! by traversal, so a self-referential or cyclic hierarchy terminates by
//! construction.
//!
//! **Known limitation.** Recovery reads the image as it appears in the
//! file. In a PIE/shared image whose vtable and typeinfo pointers are
//! supplied by dynamic relocations, those slots read as zero; that yields
//! *fewer* classes and emptier vtables, never wrong ones.
//!
//! Written clean-room from the public Itanium C++ ABI, not ported from any
//! runtime implementation.

use std::collections::{BTreeMap, BTreeSet};

use crate::cxxdemangle;
use crate::model::{Image, Region};

// ---------------------------------------------------------------------------
// Resource caps
// ---------------------------------------------------------------------------

/// Pointer width of the images this module recovers from, and hence the
/// stride of every vtable slot and typeinfo field.
const PTR: u64 = 8;

/// Maximum base classes read from one `__vmi_class_type_info`, whatever
/// its `base_count` claims.
pub const MAX_BASES: usize = 256;

/// Maximum slots recorded for one sub-vtable.
pub const MAX_SLOTS: usize = 4096;

/// Maximum sub-vtables split out of one `_ZTV` object.
pub const MAX_SUBTABLES: usize = 64;

/// Maximum classes reported from one image.
pub const MAX_CLASSES: usize = 65536;

/// Maximum length of a mangled type-name string, in bytes.
pub const MAX_NAME_LEN: usize = 4096;

/// Per-region cap on bytes swept by the heuristic scan.
pub const MAX_SCAN_BYTES: u64 = 64 * 1024 * 1024;

/// Cap on the extent of one `_ZTV` object, whatever the next symbol or the
/// region end says.
const MAX_OBJECT_BYTES: u64 = 1 << 20;

/// A run of zero slots longer than this is section padding, not a run of
/// pure-virtual entries: the walk stops and the run is dropped.
const MAX_ZERO_RUN: usize = 8;

/// Largest plausible magnitude for an offset-to-top or a base offset.
const MAX_OFFSET: u64 = 1 << 24;

/// How far into a `_ZTV` object the header search looks. Classes with
/// virtual bases prefix the object with virtual-base offsets, so the
/// `[offset-to-top][typeinfo]` header is not always at offset 0.
const MAX_VTABLE_PREFIX: u64 = 64 * PTR;

/// How many preceding entries a region lookup probes, so that overlapping
/// regions (which some formats permit) still resolve.
const MAX_REGION_PROBE: usize = 8;

/// Bits defined for `__vmi_class_type_info::__flags`
/// (`__non_diamond_repeat_mask`, `__diamond_shaped_mask`, and the
/// implementation's unknown-mask bits). Anything above them makes a
/// candidate implausible.
const VMI_FLAGS_MASK: u32 = 0x1f;

/// `__base_class_type_info::__virtual_mask`.
const BASE_VIRTUAL_MASK: u64 = 0x1;

/// `__base_class_type_info::__public_mask`.
const BASE_PUBLIC_MASK: u64 = 0x2;

/// `__base_class_type_info::__offset_shift`.
const OFFSET_SHIFT: u32 = 8;

/// Prefix a demangled `_ZTI` symbol carries, stripped to leave the bare
/// class name.
const TYPEINFO_FOR: &str = "typeinfo for ";

// ---------------------------------------------------------------------------
// Public model
// ---------------------------------------------------------------------------

/// How a class was located, so callers can rank how much to trust it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Provenance {
    /// Located by an `_ZTI`/`_ZTV` symbol: exact, nothing guessed.
    Symbol,
    /// Located by the conservative data-section scan of a stripped image
    /// and cross-validated against another RTTI structure.
    Scan,
}

impl Provenance {
    /// Short label for reports.
    pub fn label(self) -> &'static str {
        match self {
            Provenance::Symbol => "symbol",
            Provenance::Scan => "scan",
        }
    }
}

/// Which `std::type_info` derivative a class's typeinfo object is, which
/// is exactly what fixes the shape of its base list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TypeInfoKind {
    /// `__class_type_info`: no base classes.
    Class,
    /// `__si_class_type_info`: one public, non-virtual base at offset 0.
    SingleInheritance,
    /// `__vmi_class_type_info`: multiple and/or virtual inheritance.
    MultipleInheritance,
}

impl TypeInfoKind {
    /// Short label for reports.
    pub fn label(self) -> &'static str {
        match self {
            TypeInfoKind::Class => "no bases",
            TypeInfoKind::SingleInheritance => "single inheritance",
            TypeInfoKind::MultipleInheritance => "multiple inheritance",
        }
    }
}

/// One inheritance edge, as the derived class's typeinfo object records it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseRef {
    /// Virtual address of the base class's typeinfo object.
    pub typeinfo_va: u64,
    /// Demangled base-class name, when the base's own typeinfo was
    /// recovered or its name string could be read; `None` when the base
    /// lives outside this image.
    pub name: Option<String>,
    /// Signed offset of the base subobject within the derived object. Zero
    /// for single inheritance; for a virtual base this is the offset of
    /// the *virtual-base offset slot*, per the ABI.
    pub offset: i64,
    /// The base is inherited virtually (`__virtual_mask`).
    pub is_virtual: bool,
    /// The base is inherited publicly (`__public_mask`).
    pub is_public: bool,
}

/// One sub-vtable: a `[offset-to-top][typeinfo]` header and the virtual
/// function pointers that follow its address point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vtable {
    /// Virtual address of this sub-vtable's header (not its address point;
    /// the address point stored in objects is `va + 16`).
    pub va: u64,
    /// Signed displacement from a subobject using this sub-vtable back to
    /// the top of the complete object. Zero for the primary sub-vtable.
    pub offset_to_top: i64,
    /// Virtual address of the owning class's typeinfo object, or 0 when
    /// the slot is null (an image built without RTTI, or a pointer left to
    /// a dynamic relocation).
    pub typeinfo_va: u64,
    /// Virtual function pointers in slot order. Zeros are kept in place —
    /// they are pure-virtual or offset entries, and dropping them would
    /// shift every later slot index.
    pub slots: Vec<u64>,
}

/// One recovered C++ class: its typeinfo object, its vtables, and its
/// direct bases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CxxClass {
    /// Demangled class name (`foo::Bar`), or the raw mangled `<type>`
    /// encoding when the grammar did not cover it.
    pub name: String,
    /// Virtual address of the class's typeinfo object — this type's
    /// identity, and the key `recover` sorts and deduplicates on.
    pub typeinfo_va: u64,
    /// Which `std::type_info` derivative the typeinfo object is.
    pub kind: TypeInfoKind,
    /// Vtables found for this class, sorted by VA. Under multiple
    /// inheritance one `_ZTV` object contributes several.
    pub vtables: Vec<Vtable>,
    /// Direct base classes, in the order the typeinfo object lists them.
    pub bases: Vec<BaseRef>,
    /// Which tier recovered the class.
    pub provenance: Provenance,
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Recover the C++ class hierarchy from a loaded [`Image`], best-effort.
///
/// Returns the classes sorted by typeinfo VA and deduplicated on it —
/// deterministic for a given image — or an empty vector when the image
/// carries no recoverable RTTI. Never errors, never panics.
///
/// The symbol tier runs first; the heuristic scan runs only if it found
/// nothing (see the module docs).
pub fn recover(image: &dyn Image) -> Vec<CxxClass> {
    let mem = Mem::new(image);
    let mut classes = from_symbols(&mem);
    if classes.is_empty() {
        classes = from_scan(&mem);
    }
    finish(&mem, classes)
}

/// Render a recovered hierarchy as a readable report.
///
/// Deterministic for a given input and always newline-terminated, so a CLI
/// can print it verbatim. An empty slice renders as a single explanatory
/// line rather than an empty string.
pub fn render(classes: &[CxxClass]) -> String {
    if classes.is_empty() {
        return "no C++ RTTI recovered\n".to_string();
    }
    let symbol = classes
        .iter()
        .filter(|c| c.provenance == Provenance::Symbol)
        .count();
    let scan = classes.len().saturating_sub(symbol);
    let mut out = format!(
        "C++ classes: {} ({symbol} symbol-driven, {scan} scan-recovered)\n",
        classes.len()
    );
    for c in classes {
        out.push_str(&format!(
            "\n{}  [{}, {}]\n",
            c.name,
            c.provenance.label(),
            c.kind.label()
        ));
        out.push_str(&format!("  typeinfo {}\n", hex_va(c.typeinfo_va)));
        for b in &c.bases {
            out.push_str(&format!(
                "  base     {}  typeinfo {}  offset {}{}{}\n",
                b.name.as_deref().unwrap_or("(unknown)"),
                hex_va(b.typeinfo_va),
                b.offset,
                if b.is_virtual { "  virtual" } else { "" },
                if b.is_public {
                    "  public"
                } else {
                    "  non-public"
                },
            ));
        }
        for v in &c.vtables {
            out.push_str(&format!(
                "  vtable   {}  offset-to-top {}  slots {}\n",
                hex_va(v.va),
                v.offset_to_top,
                v.slots.len()
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
/// nor leave the file buffer. All arithmetic is checked; there is no path
/// here that can panic on hostile data.
struct Mem<'a> {
    image: &'a dyn Image,
    /// The image's regions, sorted by VA, zero-size ones dropped.
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

    fn u32_at(&self, va: u64) -> Option<u32> {
        let b = self.bytes_at(va, 4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64_at(&self, va: u64) -> Option<u64> {
        let b = self.bytes_at(va, 8)?;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a pointer-sized value at `base + delta`, checked both ways.
    fn ptr_at(&self, base: u64, delta: u64) -> Option<u64> {
        self.u64_at(base.checked_add(delta)?)
    }

    /// A NUL-terminated mangled-name string at `va`: printable ASCII only,
    /// at least two bytes, at most [`MAX_NAME_LEN`], never crossing out of
    /// its region. `None` if any of that fails.
    fn mangled_str_at(&self, va: u64) -> Option<&'a str> {
        let region = self.region_at(va)?;
        let avail = region.va.checked_add(region.size)?.checked_sub(va)?;
        let want = avail.min(MAX_NAME_LEN as u64) as usize;
        let window = self.bytes_at(va, want).or_else(|| {
            // The region may outrun the file buffer (zero-fill tails);
            // fall back to the longest readable prefix.
            (1..want)
                .rev()
                .find_map(|n| self.bytes_at(va, n))
                .filter(|_| want > 1)
        })?;
        let end = window.iter().position(|&b| b == 0)?;
        let body = window.get(..end)?;
        if body.len() < 2 || !body.iter().all(|&b| (0x21..=0x7e).contains(&b)) {
            return None;
        }
        std::str::from_utf8(body).ok()
    }

    /// Whether `va` lies in an executable region — the test a vtable slot
    /// must pass to count as a function pointer.
    fn is_exec(&self, va: u64) -> bool {
        self.region_at(va).is_some_and(|r| r.perms.x)
    }

    /// Whether `va` lies in a readable, non-executable region — where
    /// typeinfo objects, vtables, and name strings live.
    fn is_data(&self, va: u64) -> bool {
        self.region_at(va).is_some_and(|r| r.perms.r && !r.perms.x)
    }

    /// The end VA of the region holding `va`, or `va` itself if unmapped.
    fn region_end(&self, va: u64) -> u64 {
        self.region_at(va)
            .map_or(va, |r| r.va.saturating_add(r.size))
    }

    /// 8-aligned, bounded (start, end) VA ranges the heuristic scan sweeps:
    /// the readable non-executable regions.
    fn scan_ranges(&self) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        for r in &self.regions {
            if !(r.perms.r && !r.perms.x) {
                continue;
            }
            let start = r.va.saturating_add(PTR - 1) & !(PTR - 1);
            let end = r.va.saturating_add(r.size.min(MAX_SCAN_BYTES));
            if start < end {
                out.push((start, end));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

/// Strip the extra leading underscore a Mach-O assembler prepends, the
/// same way the demanglers accept `__Z` alongside `_Z`.
fn strip_mach_underscore(name: &str) -> &str {
    if name.starts_with("__Z") {
        &name[1..]
    } else {
        name
    }
}

/// True if `payload` opens the way a `<type>` encoding must: a length digit
/// (`6Widget`), `N` (a nested name), or `S` (a substitution or a `std`
/// abbreviation, as in `St9exception`).
fn is_type_encoding(payload: &str) -> bool {
    matches!(payload.as_bytes().first(), Some(b'0'..=b'9' | b'N' | b'S'))
}

/// Render a `_ZTS` payload — a bare `<type>` encoding — as a source-level
/// class name.
///
/// The trick is to hand the payload back to the full demangler by
/// synthesizing the `_ZTI` symbol it would have come from, then stripping
/// the `"typeinfo for "` the demangler prints: the whole type grammar
/// (nested names, templates, substitutions) comes along for free. `None`
/// when the payload is not a type this crate's grammar covers.
fn class_name(payload: &str) -> Option<String> {
    // Some runtimes prefix a non-unique type name with `*`; it is not part
    // of the encoding.
    let payload = payload.strip_prefix('*').unwrap_or(payload);
    if payload.len() > MAX_NAME_LEN || !is_type_encoding(payload) {
        return None;
    }
    let rendered = cxxdemangle::try_demangle(&format!("_ZTI{payload}"))?;
    rendered.strip_prefix(TYPEINFO_FOR).map(str::to_string)
}

/// [`class_name`], falling back to the raw payload. Used where the payload
/// came from a symbol and is therefore trustworthy even when the grammar
/// does not cover it.
fn class_name_or_raw(payload: &str) -> String {
    class_name(payload).unwrap_or_else(|| payload.to_string())
}

/// The class name of the typeinfo object at `va`, read through its name
/// pointer. `None` when the object is unreadable, the string is not a
/// plausible `<type>` encoding, or the grammar refuses it.
fn typeinfo_name_at(mem: &Mem, va: u64) -> Option<String> {
    let name_ptr = mem.ptr_at(va, PTR)?;
    if name_ptr == 0 {
        return None;
    }
    class_name(mem.mangled_str_at(name_ptr)?)
}

/// Whether `va` plausibly holds a typeinfo object: a vptr that is null or
/// an aligned data pointer, and a readable, demanglable name string.
fn looks_like_typeinfo(mem: &Mem, va: u64) -> bool {
    let Some(vptr) = mem.u64_at(va) else {
        return false;
    };
    if vptr % PTR != 0 || (vptr != 0 && !mem.is_data(vptr)) {
        return false;
    }
    typeinfo_name_at(mem, va).is_some()
}

// ---------------------------------------------------------------------------
// Typeinfo bodies
// ---------------------------------------------------------------------------

/// The `_ZTV` symbols of the three abi classes whose address points a
/// typeinfo object's vptr names, mapped to the layout each implies.
fn abi_class_vtable_kind(name: &str) -> Option<TypeInfoKind> {
    match name {
        "_ZTVN10__cxxabiv117__class_type_infoE" => Some(TypeInfoKind::Class),
        "_ZTVN10__cxxabiv120__si_class_type_infoE" => Some(TypeInfoKind::SingleInheritance),
        "_ZTVN10__cxxabiv121__vmi_class_type_infoE" => Some(TypeInfoKind::MultipleInheritance),
        _ => None,
    }
}

/// A parsed typeinfo body.
struct Parsed {
    kind: TypeInfoKind,
    bases: Vec<BaseRef>,
}

/// Parse the typeinfo object at `va`.
///
/// Two ways to learn which derivative it is, in order of authority:
///
/// 1. Its vptr names one of the abi classes' vtables (`abi`, populated
///    from the image's own symbols). Statically linked images say so
///    outright, and then the body is read as declared.
/// 2. Otherwise the layout is *inferred from the body's shape* against
///    `known` — the set of addresses believed to hold typeinfo objects.
///    A `__si` body's third field points at one of them; a `__vmi` body's
///    third and fourth fields are a plausible flags/count pair followed by
///    exactly that many plausible base entries. Neither shape matching
///    means "no bases", which is also the truth for `__class_type_info`.
fn parse_typeinfo(
    mem: &Mem,
    va: u64,
    known: &BTreeSet<u64>,
    abi: &BTreeMap<u64, TypeInfoKind>,
) -> Parsed {
    if let Some(vptr) = mem.u64_at(va)
        && let Some(&kind) = abi.get(&vptr)
    {
        let bases = match kind {
            TypeInfoKind::Class => Vec::new(),
            TypeInfoKind::SingleInheritance => si_base(mem, va, None).into_iter().collect(),
            TypeInfoKind::MultipleInheritance => vmi_bases(mem, va, None).unwrap_or_default(),
        };
        return Parsed { kind, bases };
    }
    if let Some(base) = si_base(mem, va, Some(known)) {
        return Parsed {
            kind: TypeInfoKind::SingleInheritance,
            bases: vec![base],
        };
    }
    if let Some(bases) = vmi_bases(mem, va, Some(known)) {
        return Parsed {
            kind: TypeInfoKind::MultipleInheritance,
            bases,
        };
    }
    Parsed {
        kind: TypeInfoKind::Class,
        bases: Vec::new(),
    }
}

/// Whether `va` is believed to name a typeinfo object: either the caller
/// enumerated it, or its body reads like one.
fn is_typeinfo_ref(mem: &Mem, va: u64, known: &BTreeSet<u64>) -> bool {
    known.contains(&va) || looks_like_typeinfo(mem, va)
}

/// The single base of an `__si_class_type_info` body.
///
/// `known` present means the layout is being *inferred*, so the base
/// pointer must land on a believed typeinfo; absent means the vptr already
/// declared the layout and the pointer is taken as read.
fn si_base(mem: &Mem, va: u64, known: Option<&BTreeSet<u64>>) -> Option<BaseRef> {
    let base = mem.ptr_at(va, 2 * PTR)?;
    if base == 0 {
        return None;
    }
    if let Some(known) = known
        && !is_typeinfo_ref(mem, base, known)
    {
        return None;
    }
    Some(BaseRef {
        typeinfo_va: base,
        name: None,
        offset: 0,
        is_virtual: false,
        is_public: true,
    })
}

/// The base list of a `__vmi_class_type_info` body (see [`si_base`] for
/// what `known` selects). Inference is all-or-nothing: every one of the
/// claimed bases must check out, or the shape is refused.
fn vmi_bases(mem: &Mem, va: u64, known: Option<&BTreeSet<u64>>) -> Option<Vec<BaseRef>> {
    let flags = mem.u32_at(va.checked_add(2 * PTR)?)?;
    let count = mem.u32_at(va.checked_add(2 * PTR + 4)?)?;
    let inferring = known.is_some();
    if count == 0 || (inferring && flags & !VMI_FLAGS_MASK != 0) {
        return None;
    }
    if inferring && count as usize > MAX_BASES {
        return None;
    }
    let wanted = (count as usize).min(MAX_BASES);
    let mut out = Vec::with_capacity(wanted.min(16));
    for i in 0..wanted {
        let entry = match va.checked_add(3 * PTR + (i as u64).checked_mul(2 * PTR)?) {
            Some(e) => e,
            None => break,
        };
        let (Some(base), Some(offset_flags)) = (mem.u64_at(entry), mem.ptr_at(entry, PTR)) else {
            break;
        };
        if base == 0 {
            break;
        }
        if let Some(known) = known
            && !is_typeinfo_ref(mem, base, known)
        {
            return None;
        }
        out.push(BaseRef {
            typeinfo_va: base,
            name: None,
            // Arithmetic shift: the offset is signed (ABI §2.9.5).
            offset: (offset_flags as i64) >> OFFSET_SHIFT,
            is_virtual: offset_flags & BASE_VIRTUAL_MASK != 0,
            is_public: offset_flags & BASE_PUBLIC_MASK != 0,
        });
    }
    if out.is_empty() || (inferring && out.len() != wanted) {
        return None;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Vtable objects
// ---------------------------------------------------------------------------

/// One sub-vtable plus, when the object continues, where the next
/// sub-vtable's header begins.
struct Group {
    vtable: Vtable,
    next_header: Option<u64>,
}

/// Whether the qword `value` at `at` opens another sub-vtable of the class
/// whose typeinfo is `ti`: a negative, plausible offset-to-top followed by
/// a repeat of that same typeinfo pointer. Secondary sub-vtables always
/// carry a negative offset-to-top, which is what keeps this apart from a
/// zero (pure-virtual) slot.
fn is_secondary_header(mem: &Mem, at: u64, value: u64, ti: u64) -> bool {
    let offset_to_top = value as i64;
    offset_to_top < 0
        && offset_to_top.unsigned_abs() <= MAX_OFFSET
        && mem.ptr_at(at, PTR) == Some(ti)
}

/// Parse the sub-vtable whose header starts at `at`, stopping at `end`.
fn parse_group(mem: &Mem, at: u64, end: u64) -> Option<Group> {
    let offset_to_top = mem.u64_at(at)? as i64;
    let typeinfo_va = mem.ptr_at(at, PTR)?;
    let mut p = at.checked_add(2 * PTR)?;
    let mut slots: Vec<u64> = Vec::new();
    let mut next_header = None;
    let mut zero_run = 0usize;
    let mut padding = false;
    while slots.len() < MAX_SLOTS {
        let Some(next) = p.checked_add(PTR) else {
            break;
        };
        if next > end {
            break;
        }
        let Some(value) = mem.u64_at(p) else {
            break;
        };
        if typeinfo_va != 0 && !slots.is_empty() && is_secondary_header(mem, p, value, typeinfo_va)
        {
            next_header = Some(p);
            break;
        }
        // A slot is a function pointer only if it targets code; zero is a
        // pure-virtual or offset entry and is kept for its index.
        if value != 0 && !mem.is_exec(value) {
            break;
        }
        zero_run = if value == 0 { zero_run + 1 } else { 0 };
        if zero_run > MAX_ZERO_RUN {
            padding = true;
            break;
        }
        slots.push(value);
        p = next;
    }
    if padding {
        slots.truncate(slots.len().saturating_sub(zero_run.saturating_sub(1)));
    }
    Some(Group {
        vtable: Vtable {
            va: at,
            offset_to_top,
            typeinfo_va,
            slots,
        },
        next_header,
    })
}

/// Split the vtable object whose first header is at `header` into its
/// sub-vtables, stopping at `end`.
fn parse_vtable_at(mem: &Mem, header: u64, end: u64) -> Vec<Vtable> {
    let mut out = Vec::new();
    let mut at = header;
    while out.len() < MAX_SUBTABLES {
        let Some(group) = parse_group(mem, at, end) else {
            break;
        };
        let next = group.next_header;
        out.push(group.vtable);
        match next {
            Some(n) if n > at => at = n,
            _ => break,
        }
    }
    out
}

/// Locate the `[offset-to-top][typeinfo]` header inside the `_ZTV` object
/// starting at `start`.
///
/// A class with virtual bases prefixes its vtable with virtual-base
/// offsets, so the header is not always at offset 0. When the class's own
/// typeinfo VA is known from a symbol, the header is the first slot pair
/// whose second half repeats it; otherwise the object is taken to start
/// with its header, which is the layout of every class without virtual
/// bases.
fn header_start(mem: &Mem, start: u64, end: u64, expected_ti: Option<u64>) -> u64 {
    let Some(ti) = expected_ti else {
        return start;
    };
    let limit = start.saturating_add(MAX_VTABLE_PREFIX).min(end);
    let mut p = start;
    while p < limit {
        if mem.ptr_at(p, PTR) == Some(ti) {
            return p;
        }
        match p.checked_add(PTR) {
            Some(next) => p = next,
            None => break,
        }
    }
    start
}

/// Parse a whole `_ZTV` object: find its header, then split it into
/// sub-vtables.
fn parse_vtable_object(mem: &Mem, start: u64, end: u64, expected_ti: Option<u64>) -> Vec<Vtable> {
    parse_vtable_at(mem, header_start(mem, start, end, expected_ti), end)
}

// ---------------------------------------------------------------------------
// Tier 1: symbol-driven recovery
// ---------------------------------------------------------------------------

/// The RTTI symbols of an image, bucketed by what they name.
#[derive(Default)]
struct RttiSymbols {
    /// Typeinfo object VA to its `_ZTI` payload (the mangled `<type>`).
    typeinfo: BTreeMap<u64, String>,
    /// Name-string VA to its `_ZTS` payload.
    typename: BTreeMap<u64, String>,
    /// Vtable object VA to its `_ZTV` payload.
    vtable: BTreeMap<u64, String>,
    /// `_ZTI` payload to the typeinfo VA carrying it, so a `_ZTV` can find
    /// the typeinfo of the class it belongs to.
    by_payload: BTreeMap<String, u64>,
    /// VAs a typeinfo vptr may hold, mapped to the layout they imply: both
    /// the abi vtable's own VA and its address point (VA + 16), since the
    /// vptr stored in an object points at the latter.
    abi: BTreeMap<u64, TypeInfoKind>,
}

/// Bucket the image's `_ZTV`/`_ZTI`/`_ZTS` symbols, Mach-O spellings
/// included.
fn rtti_symbols(mem: &Mem) -> RttiSymbols {
    let mut out = RttiSymbols::default();
    for sym in mem.image.symbols() {
        let name = strip_mach_underscore(&sym.name);
        if let Some(kind) = abi_class_vtable_kind(name) {
            out.abi.insert(sym.va, kind);
            if let Some(point) = sym.va.checked_add(2 * PTR) {
                out.abi.insert(point, kind);
            }
            continue;
        }
        let Some((tag, payload)) = ["_ZTI", "_ZTS", "_ZTV"]
            .iter()
            .find_map(|tag| name.strip_prefix(tag).map(|payload| (*tag, payload)))
        else {
            continue;
        };
        if payload.is_empty() || payload.len() > MAX_NAME_LEN {
            continue;
        }
        match tag {
            "_ZTI" => {
                out.typeinfo.insert(sym.va, payload.to_string());
                out.by_payload.entry(payload.to_string()).or_insert(sym.va);
            }
            "_ZTS" => {
                out.typename.insert(sym.va, payload.to_string());
            }
            _ => {
                out.vtable
                    .entry(sym.va)
                    .or_insert_with(|| payload.to_string());
            }
        }
    }
    out
}

/// The `_ZTS` payload naming the typeinfo object at `va`, for the case
/// where the image kept the name symbol but the string it points at is
/// unreadable (a zero-fill tail) or is not a `<type>` this crate's grammar
/// covers.
fn typename_symbol_at(mem: &Mem, syms: &RttiSymbols, va: u64) -> Option<String> {
    let name_ptr = mem.ptr_at(va, PTR)?;
    syms.typename.get(&name_ptr).map(|p| class_name_or_raw(p))
}

/// Parse every `_ZTV` object, keyed by the typeinfo VA of the class that
/// owns it. An object's extent runs to the next `_ZTV` symbol or the end of
/// its region, whichever comes first (and never past [`MAX_OBJECT_BYTES`]).
fn vtables_by_typeinfo(mem: &Mem, syms: &RttiSymbols) -> BTreeMap<u64, Vec<Vtable>> {
    let starts: Vec<u64> = syms.vtable.keys().copied().collect();
    let mut out: BTreeMap<u64, Vec<Vtable>> = BTreeMap::new();
    for (i, &va) in starts.iter().enumerate() {
        let payload = match syms.vtable.get(&va) {
            Some(p) => p.as_str(),
            None => continue,
        };
        let expected_ti = syms.by_payload.get(payload).copied();
        let mut end = mem.region_end(va).min(va.saturating_add(MAX_OBJECT_BYTES));
        if let Some(&next) = starts.get(i + 1)
            && next > va
            && next < end
        {
            end = next;
        }
        for vt in parse_vtable_object(mem, va, end, expected_ti) {
            // The symbol's own class wins over a typeinfo slot that a
            // dynamic relocation may have left null.
            let key = match (expected_ti, vt.typeinfo_va) {
                (Some(ti), _) => ti,
                (None, 0) => continue,
                (None, ti) => ti,
            };
            out.entry(key).or_default().push(vt);
        }
    }
    out
}

/// Recover classes from the image's RTTI symbols.
fn from_symbols(mem: &Mem) -> Vec<CxxClass> {
    let syms = rtti_symbols(mem);
    if syms.typeinfo.is_empty() && syms.vtable.is_empty() {
        return Vec::new();
    }
    let mut vtables = vtables_by_typeinfo(mem, &syms);

    // Typeinfo objects a `_ZTV` header points at, but no `_ZTI` symbol
    // named, still describe a class — provided the body reads like one.
    let mut known: BTreeSet<u64> = syms.typeinfo.keys().copied().collect();
    for &va in vtables.keys() {
        if !known.contains(&va)
            && (looks_like_typeinfo(mem, va) || typename_symbol_at(mem, &syms, va).is_some())
        {
            known.insert(va);
        }
    }

    let mut out = Vec::new();
    for &va in &known {
        if out.len() >= MAX_CLASSES {
            break;
        }
        let name = match syms.typeinfo.get(&va) {
            Some(payload) => class_name_or_raw(payload),
            None => {
                match typeinfo_name_at(mem, va).or_else(|| typename_symbol_at(mem, &syms, va)) {
                    Some(name) => name,
                    // No symbol and no readable name: refuse rather than
                    // invent an identity for it.
                    None => continue,
                }
            }
        };
        let parsed = parse_typeinfo(mem, va, &known, &syms.abi);
        out.push(CxxClass {
            name,
            typeinfo_va: va,
            kind: parsed.kind,
            vtables: vtables.remove(&va).unwrap_or_default(),
            bases: parsed.bases,
            provenance: Provenance::Symbol,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Tier 2: heuristic scan of a stripped image
// ---------------------------------------------------------------------------

/// Sweep the readable non-executable regions for typeinfo-shaped pairs: a
/// vptr into a data region (the abi class's vtable) followed by a pointer
/// to a plausible, demanglable mangled type name.
fn scan_candidates(mem: &Mem) -> BTreeMap<u64, String> {
    let mut out = BTreeMap::new();
    for (start, end) in mem.scan_ranges() {
        let mut va = start;
        while va < end {
            if out.len() >= MAX_CLASSES {
                return out;
            }
            let Some(next) = va.checked_add(PTR) else {
                break;
            };
            if let Some(vptr) = mem.u64_at(va)
                && vptr != 0
                && vptr % PTR == 0
                && mem.is_data(vptr)
                && let Some(name) = typeinfo_name_at(mem, va)
            {
                out.insert(va, name);
            }
            va = next;
        }
    }
    out
}

/// Sweep the same regions for vtable-shaped objects — a plausible
/// offset-to-top, a typeinfo slot naming one of `candidates`, and at least
/// one executable slot — and collect them per typeinfo VA.
///
/// This is both the second half of a candidate's cross-validation and the
/// scan tier's source of vtables.
fn scan_vtables(mem: &Mem, candidates: &BTreeMap<u64, String>) -> BTreeMap<u64, Vec<Vtable>> {
    let mut out: BTreeMap<u64, Vec<Vtable>> = BTreeMap::new();
    for (start, region_end) in mem.scan_ranges() {
        let mut va = start;
        while va < region_end {
            let Some(next) = va.checked_add(PTR) else {
                break;
            };
            let plausible = mem.u64_at(va).is_some_and(|top| {
                let top = top as i64;
                top.unsigned_abs() <= MAX_OFFSET
            }) && mem
                .ptr_at(va, PTR)
                .is_some_and(|ti| candidates.contains_key(&ti));
            if !plausible {
                va = next;
                continue;
            }
            let end = region_end.min(va.saturating_add(MAX_OBJECT_BYTES));
            let group = parse_vtable_at(mem, va, end);
            let functions = group
                .iter()
                .flat_map(|v| v.slots.iter())
                .any(|&slot| slot != 0);
            if !functions {
                va = next;
                continue;
            }
            // Skip past everything this object consumed, so its interior
            // sub-vtables are not rediscovered as separate objects.
            let mut resume = next;
            for vt in group {
                let consumed = vt
                    .va
                    .saturating_add(2 * PTR)
                    .saturating_add((vt.slots.len() as u64).saturating_mul(PTR));
                resume = resume.max(consumed);
                out.entry(vt.typeinfo_va).or_default().push(vt);
            }
            va = resume;
        }
    }
    out
}

/// Recover classes from a stripped image by scanning, keeping only the
/// candidates another RTTI structure corroborates.
fn from_scan(mem: &Mem) -> Vec<CxxClass> {
    let candidates = scan_candidates(mem);
    if candidates.is_empty() {
        return Vec::new();
    }
    let known: BTreeSet<u64> = candidates.keys().copied().collect();
    let abi = BTreeMap::new(); // stripped: no abi vtable symbols to trust
    let parsed: BTreeMap<u64, Parsed> = known
        .iter()
        .map(|&va| (va, parse_typeinfo(mem, va, &known, &abi)))
        .collect();

    // Cross-validation: a candidate survives only if another candidate
    // names it as a base, or a vtable-shaped object names it as its type.
    let referenced: BTreeSet<u64> = parsed
        .values()
        .flat_map(|p| p.bases.iter().map(|b| b.typeinfo_va))
        .collect();
    let mut vtables = scan_vtables(mem, &candidates);

    let mut out = Vec::new();
    for (va, name) in candidates {
        if out.len() >= MAX_CLASSES {
            break;
        }
        let vts = vtables.remove(&va).unwrap_or_default();
        if vts.is_empty() && !referenced.contains(&va) {
            continue;
        }
        let Some(parsed) = parsed.get(&va) else {
            continue;
        };
        out.push(CxxClass {
            name,
            typeinfo_va: va,
            kind: parsed.kind,
            vtables: vts,
            bases: parsed.bases.clone(),
            provenance: Provenance::Scan,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Finishing
// ---------------------------------------------------------------------------

/// Put the result in canonical order and fill in base-class names.
///
/// Names are resolved by *lookup*, never by walking the hierarchy, so a
/// cyclic or self-referential base graph costs nothing and terminates.
fn finish(mem: &Mem, mut classes: Vec<CxxClass>) -> Vec<CxxClass> {
    classes.sort_by(|a, b| (a.typeinfo_va, a.name.as_str()).cmp(&(b.typeinfo_va, b.name.as_str())));
    classes.dedup_by_key(|c| c.typeinfo_va);
    classes.truncate(MAX_CLASSES);

    let names: BTreeMap<u64, String> = classes
        .iter()
        .map(|c| (c.typeinfo_va, c.name.clone()))
        .collect();
    for class in &mut classes {
        class.vtables.sort_by_key(|v| (v.va, v.offset_to_top));
        class.vtables.dedup_by_key(|v| v.va);
        class.vtables.truncate(MAX_SUBTABLES);
        for base in &mut class.bases {
            base.name = names
                .get(&base.typeinfo_va)
                .cloned()
                .or_else(|| typeinfo_name_at(mem, base.typeinfo_va));
        }
    }
    classes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Arch, ImportSlot, Perms, Symbol, SymbolKind};

    // ---- Synthetic images ------------------------------------------------

    /// Size of every synthetic image's buffer.
    const IMG_SIZE: usize = 0x1_0000;
    /// Readable, non-executable region (typeinfo, vtables, name strings).
    const DATA_VA: u64 = 0x1000;
    const DATA_END: u64 = 0x8000;
    /// Executable region (vtable slot targets).
    const TEXT_VA: u64 = 0x8000;
    const TEXT_END: u64 = 0x9000;

    /// Identity-mapped image (VA == file offset) with one data and one
    /// text region — the smallest thing `recover` can be run against.
    struct FakeImage {
        data: Vec<u8>,
        regions: Vec<Region>,
        symbols: Vec<Symbol>,
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
            self.symbols.clone()
        }
        fn import_slots(&self) -> Vec<ImportSlot> {
            Vec::new()
        }
        fn va_to_offset(&self, va: u64) -> Option<usize> {
            let off = usize::try_from(va).ok()?;
            (off < self.data.len()).then_some(off)
        }
        fn bytes(&self) -> &[u8] {
            &self.data
        }
    }

    impl FakeImage {
        fn new() -> FakeImage {
            FakeImage {
                data: vec![0u8; IMG_SIZE],
                regions: vec![
                    Region {
                        name: ".data.rel.ro".into(),
                        va: DATA_VA,
                        size: DATA_END - DATA_VA,
                        perms: Perms {
                            r: true,
                            w: false,
                            x: false,
                        },
                    },
                    Region {
                        name: ".text".into(),
                        va: TEXT_VA,
                        size: TEXT_END - TEXT_VA,
                        perms: Perms {
                            r: true,
                            w: false,
                            x: true,
                        },
                    },
                ],
                symbols: Vec::new(),
            }
        }

        fn u64(&mut self, va: u64, value: u64) -> &mut Self {
            let off = va as usize;
            self.data[off..off + 8].copy_from_slice(&value.to_le_bytes());
            self
        }

        fn u32(&mut self, va: u64, value: u32) -> &mut Self {
            let off = va as usize;
            self.data[off..off + 4].copy_from_slice(&value.to_le_bytes());
            self
        }

        fn str(&mut self, va: u64, s: &str) -> &mut Self {
            let off = va as usize;
            self.data[off..off + s.len()].copy_from_slice(s.as_bytes());
            self.data[off + s.len()] = 0;
            self
        }

        fn sym(&mut self, name: &str, va: u64) -> &mut Self {
            self.symbols.push(Symbol {
                name: name.into(),
                va,
                kind: SymbolKind::Data,
            });
            self
        }

        /// A typeinfo object: vptr, name pointer, and the name string.
        fn typeinfo(&mut self, va: u64, vptr: u64, name_va: u64, payload: &str) -> &mut Self {
            self.u64(va, vptr)
                .u64(va + 8, name_va)
                .str(name_va, payload)
        }

        /// A value that is neither zero nor a code pointer, which ends a
        /// vtable's slot walk the way the next object's bytes would.
        fn terminator(&mut self, va: u64) -> &mut Self {
            self.u64(va, 0x0bad_0bad_0bad_0bad)
        }
    }

    /// VAs of the abi class vtables the fixtures point typeinfo vptrs at.
    const ABI_CLASS: u64 = 0x1200;
    const ABI_SI: u64 = 0x1240;
    const ABI_VMI: u64 = 0x1280;

    /// Declare the three abi class vtables as symbols, so typeinfo vptrs
    /// classify by authority rather than by shape.
    fn with_abi_symbols(img: &mut FakeImage) {
        img.sym("_ZTVN10__cxxabiv117__class_type_infoE", ABI_CLASS)
            .sym("_ZTVN10__cxxabiv120__si_class_type_infoE", ABI_SI)
            .sym("_ZTVN10__cxxabiv121__vmi_class_type_infoE", ABI_VMI);
    }

    /// The address point a typeinfo vptr holds for `abi_vtable_va`.
    fn point(abi_vtable_va: u64) -> u64 {
        abi_vtable_va + 16
    }

    // ---- Symbol-driven recovery -----------------------------------------

    #[test]
    fn symbol_driven_class_with_vtable() {
        let mut img = FakeImage::new();
        with_abi_symbols(&mut img);
        img.typeinfo(0x2000, point(ABI_CLASS), 0x2100, "6Widget")
            .u64(0x3000, 0) // offset-to-top
            .u64(0x3008, 0x2000) // typeinfo
            .u64(0x3010, TEXT_VA) // slot 0
            .u64(0x3018, TEXT_VA + 0x10) // slot 1
            .u64(0x3020, 0) // slot 2: pure virtual
            .u64(0x3028, TEXT_VA + 0x20) // slot 3
            .terminator(0x3030)
            .sym("_ZTI6Widget", 0x2000)
            .sym("_ZTV6Widget", 0x3000);

        let classes = recover(&img);
        assert_eq!(classes.len(), 1);
        let c = &classes[0];
        assert_eq!(c.name, "Widget");
        assert_eq!(c.typeinfo_va, 0x2000);
        assert_eq!(c.kind, TypeInfoKind::Class);
        assert_eq!(c.provenance, Provenance::Symbol);
        assert!(c.bases.is_empty());
        assert_eq!(c.vtables.len(), 1);
        assert_eq!(c.vtables[0].va, 0x3000);
        assert_eq!(c.vtables[0].offset_to_top, 0);
        assert_eq!(c.vtables[0].typeinfo_va, 0x2000);
        // Zero slots are kept in place so slot indices stay meaningful.
        assert_eq!(
            c.vtables[0].slots,
            vec![TEXT_VA, TEXT_VA + 0x10, 0, TEXT_VA + 0x20]
        );
    }

    #[test]
    fn mach_o_underscore_prefixed_symbols_are_accepted() {
        let mut img = FakeImage::new();
        img.typeinfo(0x2000, 0, 0x2100, "6Widget")
            .u64(0x3000, 0)
            .u64(0x3008, 0x2000)
            .u64(0x3010, TEXT_VA)
            .terminator(0x3018)
            .sym("__ZTI6Widget", 0x2000)
            .sym("__ZTV6Widget", 0x3000);

        let classes = recover(&img);
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].name, "Widget");
        assert_eq!(classes[0].vtables[0].slots, vec![TEXT_VA]);
    }

    #[test]
    fn si_chain_resolves_base_names_through_the_demangler() {
        let mut img = FakeImage::new();
        with_abi_symbols(&mut img);
        // Base is a nested name, so the `_ZTI` trick has to run the whole
        // type grammar to render it.
        img.typeinfo(0x2000, point(ABI_CLASS), 0x2200, "N3foo4BaseE")
            .typeinfo(0x2020, point(ABI_SI), 0x2240, "N3foo7DerivedE")
            .u64(0x2030, 0x2000) // __si base pointer
            .sym("_ZTIN3foo4BaseE", 0x2000)
            .sym("_ZTIN3foo7DerivedE", 0x2020);

        let classes = recover(&img);
        assert_eq!(classes.len(), 2);
        assert_eq!(classes[0].name, "foo::Base");
        assert_eq!(classes[0].kind, TypeInfoKind::Class);
        assert_eq!(classes[1].name, "foo::Derived");
        assert_eq!(classes[1].kind, TypeInfoKind::SingleInheritance);
        assert_eq!(
            classes[1].bases,
            vec![BaseRef {
                typeinfo_va: 0x2000,
                name: Some("foo::Base".into()),
                offset: 0,
                is_virtual: false,
                is_public: true,
            }]
        );
    }

    #[test]
    fn si_layout_is_inferred_without_abi_symbols() {
        // Same shape, but no abi vtable symbols: the layout must be
        // inferred from the body against the known typeinfo set.
        let mut img = FakeImage::new();
        img.typeinfo(0x2000, ABI_CLASS, 0x2200, "4Base")
            .typeinfo(0x2020, ABI_SI, 0x2240, "7Derived")
            .u64(0x2030, 0x2000)
            .sym("_ZTI4Base", 0x2000)
            .sym("_ZTI7Derived", 0x2020);

        let classes = recover(&img);
        assert_eq!(classes[1].kind, TypeInfoKind::SingleInheritance);
        assert_eq!(classes[1].bases[0].name.as_deref(), Some("Base"));
    }

    #[test]
    fn vmi_class_decodes_two_bases_with_flags_and_offsets() {
        let mut img = FakeImage::new();
        with_abi_symbols(&mut img);
        img.typeinfo(0x2000, point(ABI_CLASS), 0x2300, "4Left")
            .typeinfo(0x2020, point(ABI_CLASS), 0x2320, "5Right")
            .typeinfo(0x2040, point(ABI_VMI), 0x2340, "7Derived")
            .u32(0x2050, 0x2) // flags: diamond-shaped
            .u32(0x2054, 2) // base_count
            .u64(0x2058, 0x2000) // base 0
            .u64(0x2060, (0i64 << OFFSET_SHIFT) as u64 | BASE_PUBLIC_MASK)
            .u64(0x2068, 0x2020) // base 1
            .u64(
                0x2070,
                ((16i64 << OFFSET_SHIFT) as u64) | BASE_PUBLIC_MASK | BASE_VIRTUAL_MASK,
            )
            .sym("_ZTI4Left", 0x2000)
            .sym("_ZTI5Right", 0x2020)
            .sym("_ZTI7Derived", 0x2040);

        let classes = recover(&img);
        assert_eq!(classes.len(), 3);
        let derived = &classes[2];
        assert_eq!(derived.name, "Derived");
        assert_eq!(derived.kind, TypeInfoKind::MultipleInheritance);
        assert_eq!(derived.bases.len(), 2);
        assert_eq!(derived.bases[0].name.as_deref(), Some("Left"));
        assert_eq!(derived.bases[0].offset, 0);
        assert!(!derived.bases[0].is_virtual);
        assert!(derived.bases[0].is_public);
        assert_eq!(derived.bases[1].name.as_deref(), Some("Right"));
        assert_eq!(derived.bases[1].offset, 16);
        assert!(derived.bases[1].is_virtual);
        assert!(derived.bases[1].is_public);
    }

    #[test]
    fn vmi_offset_flags_shift_is_arithmetic() {
        // A negative base offset must sign-extend through the >> 8.
        let mut img = FakeImage::new();
        with_abi_symbols(&mut img);
        img.typeinfo(0x2000, point(ABI_CLASS), 0x2300, "4Base")
            .typeinfo(0x2020, point(ABI_VMI), 0x2340, "7Derived")
            .u32(0x2030, 0)
            .u32(0x2034, 1)
            .u64(0x2038, 0x2000)
            .u64(0x2040, ((-24i64) << OFFSET_SHIFT) as u64) // non-public
            .sym("_ZTI4Base", 0x2000)
            .sym("_ZTI7Derived", 0x2020);

        let classes = recover(&img);
        let base = &classes[1].bases[0];
        assert_eq!(base.offset, -24);
        assert!(!base.is_public);
        assert!(!base.is_virtual);
    }

    #[test]
    fn multiple_inheritance_vtable_splits_on_the_interior_header() {
        let mut img = FakeImage::new();
        with_abi_symbols(&mut img);
        img.typeinfo(0x2000, point(ABI_CLASS), 0x2100, "7Derived")
            // Primary sub-vtable.
            .u64(0x3000, 0)
            .u64(0x3008, 0x2000)
            .u64(0x3010, TEXT_VA)
            .u64(0x3018, TEXT_VA + 8)
            // Secondary sub-vtable: negative offset-to-top, same typeinfo.
            .u64(0x3020, (-16i64) as u64)
            .u64(0x3028, 0x2000)
            .u64(0x3030, TEXT_VA + 0x10)
            .terminator(0x3038)
            .sym("_ZTI7Derived", 0x2000)
            .sym("_ZTV7Derived", 0x3000);

        let classes = recover(&img);
        assert_eq!(classes.len(), 1);
        let vts = &classes[0].vtables;
        assert_eq!(vts.len(), 2);
        assert_eq!(vts[0].va, 0x3000);
        assert_eq!(vts[0].offset_to_top, 0);
        assert_eq!(vts[0].slots, vec![TEXT_VA, TEXT_VA + 8]);
        assert_eq!(vts[1].va, 0x3020);
        assert_eq!(vts[1].offset_to_top, -16);
        assert_eq!(vts[1].slots, vec![TEXT_VA + 0x10]);
    }

    #[test]
    fn virtual_base_offsets_before_the_header_are_skipped() {
        // A class with virtual bases prefixes its vtable object with
        // vbase/vcall offsets; the `_ZTV` symbol still points at the very
        // start, so the header has to be found by its typeinfo repeat.
        let mut img = FakeImage::new();
        img.typeinfo(0x2000, 0, 0x2100, "7Derived")
            .u64(0x3000, 24) // vbase offset
            .u64(0x3008, 8) // vcall offset
            .u64(0x3010, 0) // offset-to-top
            .u64(0x3018, 0x2000) // typeinfo
            .u64(0x3020, TEXT_VA)
            .terminator(0x3028)
            .sym("_ZTI7Derived", 0x2000)
            .sym("_ZTV7Derived", 0x3000);

        let classes = recover(&img);
        assert_eq!(classes[0].vtables.len(), 1);
        assert_eq!(classes[0].vtables[0].va, 0x3010);
        assert_eq!(classes[0].vtables[0].slots, vec![TEXT_VA]);
    }

    #[test]
    fn vtable_extent_stops_at_the_next_ztv_symbol() {
        let mut img = FakeImage::new();
        img.typeinfo(0x2000, 0, 0x2100, "1A")
            .typeinfo(0x2020, 0, 0x2120, "1B")
            .u64(0x3000, 0)
            .u64(0x3008, 0x2000)
            .u64(0x3010, TEXT_VA)
            .u64(0x3018, TEXT_VA) // belongs to A, but B's symbol starts here
            .u64(0x3020, 0)
            .u64(0x3028, 0x2020)
            .u64(0x3030, TEXT_VA + 8)
            .terminator(0x3038)
            .sym("_ZTI1A", 0x2000)
            .sym("_ZTI1B", 0x2020)
            .sym("_ZTV1A", 0x3000)
            .sym("_ZTV1B", 0x3020);

        let classes = recover(&img);
        assert_eq!(classes.len(), 2);
        assert_eq!(classes[0].vtables[0].slots.len(), 2);
        assert_eq!(classes[1].vtables[0].slots, vec![TEXT_VA + 8]);
    }

    #[test]
    fn trailing_section_padding_is_not_reported_as_slots() {
        // No terminator and no following symbol: the object runs to the
        // end of the region over nothing but zeros, which are padding, not
        // a hundred pure-virtual slots.
        let mut img = FakeImage::new();
        img.typeinfo(0x2000, 0, 0x2100, "6Widget")
            .u64(0x3000, 0)
            .u64(0x3008, 0x2000)
            .u64(0x3010, TEXT_VA)
            .sym("_ZTI6Widget", 0x2000)
            .sym("_ZTV6Widget", 0x3000);

        let classes = recover(&img);
        assert_eq!(classes[0].vtables[0].slots, vec![TEXT_VA]);
    }

    #[test]
    fn slots_outside_an_executable_region_end_the_walk() {
        let mut img = FakeImage::new();
        img.typeinfo(0x2000, 0, 0x2100, "6Widget")
            .u64(0x3000, 0)
            .u64(0x3008, 0x2000)
            .u64(0x3010, TEXT_VA)
            .u64(0x3018, 0x2100) // data, not code: not a virtual function
            .u64(0x3020, TEXT_VA + 8)
            .sym("_ZTI6Widget", 0x2000)
            .sym("_ZTV6Widget", 0x3000);

        assert_eq!(recover(&img)[0].vtables[0].slots, vec![TEXT_VA]);
    }

    #[test]
    fn zts_symbol_names_a_typeinfo_whose_string_is_unreadable() {
        // No `_ZTI` symbol: the class is discovered through its vtable's
        // typeinfo slot, and its name string is binary garbage — but the
        // `_ZTS` symbol sitting on that string still names the class.
        let mut img = FakeImage::new();
        img.u64(0x2000, 0) // vptr (left to a dynamic relocation)
            .u64(0x2008, 0x2100) // name pointer
            .u64(0x2100, 0x0000_0000_0000_0101) // not a mangled string
            .u64(0x3000, 0)
            .u64(0x3008, 0x2000)
            .u64(0x3010, TEXT_VA)
            .terminator(0x3018)
            .sym("_ZTS6Widget", 0x2100)
            .sym("_ZTV6Widget", 0x3000);

        let classes = recover(&img);
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].name, "Widget");
        assert_eq!(classes[0].typeinfo_va, 0x2000);
        assert_eq!(classes[0].vtables[0].slots, vec![TEXT_VA]);
    }

    // ---- Heuristic scan --------------------------------------------------

    /// A stripped image holding Base/Derived plus a decoy typeinfo that
    /// nothing references and no vtable claims.
    fn stripped_image() -> FakeImage {
        let mut img = FakeImage::new();
        img.typeinfo(0x2000, ABI_CLASS, 0x2200, "4Base")
            .typeinfo(0x2020, ABI_SI, 0x2240, "7Derived")
            .u64(0x2030, 0x2000) // __si base pointer -> Base
            .typeinfo(0x2060, ABI_CLASS, 0x2280, "5Decoy")
            // Derived's vtable: cross-validates Derived.
            .u64(0x3000, 0)
            .u64(0x3008, 0x2020)
            .u64(0x3010, TEXT_VA)
            .u64(0x3018, TEXT_VA + 8)
            .terminator(0x3020);
        img
    }

    #[test]
    fn scan_recovers_a_cross_validated_pair_from_a_stripped_image() {
        let classes = recover(&stripped_image());
        assert_eq!(classes.len(), 2, "{:?}", classes);
        assert_eq!(classes[0].name, "Base");
        assert_eq!(classes[0].typeinfo_va, 0x2000);
        assert_eq!(classes[0].provenance, Provenance::Scan);
        assert_eq!(classes[1].name, "Derived");
        assert_eq!(classes[1].kind, TypeInfoKind::SingleInheritance);
        assert_eq!(classes[1].bases[0].name.as_deref(), Some("Base"));
        assert_eq!(classes[1].vtables.len(), 1);
        assert_eq!(classes[1].vtables[0].slots, vec![TEXT_VA, TEXT_VA + 8]);
    }

    #[test]
    fn scan_drops_an_uncrossvalidated_decoy() {
        let classes = recover(&stripped_image());
        assert!(
            !classes.iter().any(|c| c.name == "Decoy"),
            "a candidate nothing corroborates must be refused, not guessed"
        );
    }

    #[test]
    fn scan_does_not_run_when_symbols_are_present() {
        // Same bytes, but one `_ZTI` symbol: the symbol tier answers, and
        // the scan (which would also have found Base) never runs, so the
        // provenance of everything reported is Symbol.
        let mut img = stripped_image();
        img.sym("_ZTI7Derived", 0x2020);
        let classes = recover(&img);
        assert!(classes.iter().all(|c| c.provenance == Provenance::Symbol));
        assert!(classes.iter().any(|c| c.name == "Derived"));
    }

    #[test]
    fn scan_rejects_a_typeinfo_whose_name_is_not_a_type_encoding() {
        let mut img = FakeImage::new();
        // `hello` is not a `<type>`: no length prefix, no `N`/`S`.
        img.typeinfo(0x2000, ABI_CLASS, 0x2200, "hello")
            .typeinfo(0x2020, ABI_SI, 0x2240, "7Derived")
            .u64(0x2030, 0x2000)
            .u64(0x3000, 0)
            .u64(0x3008, 0x2020)
            .u64(0x3010, TEXT_VA)
            .terminator(0x3018);

        let classes = recover(&img);
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].name, "Derived");
        // The base survives as an address with no resolvable name.
        assert_eq!(classes[0].bases.len(), 0);
    }

    // ---- Hostile and degenerate inputs ----------------------------------

    #[test]
    fn image_without_rtti_recovers_nothing() {
        let img = FakeImage::new();
        assert!(recover(&img).is_empty());
        assert_eq!(render(&[]), "no C++ RTTI recovered\n");
    }

    #[test]
    fn truncated_typeinfo_body_is_refused_not_guessed() {
        let mut img = FakeImage::new();
        // The symbol names a typeinfo in the last 8 bytes of the region:
        // the vptr reads, the name pointer does not.
        img.sym("_ZTI6Widget", DATA_END - 8);
        let classes = recover(&img);
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].name, "Widget"); // the symbol still names it
        assert_eq!(classes[0].kind, TypeInfoKind::Class);
        assert!(classes[0].bases.is_empty());
        assert!(classes[0].vtables.is_empty());

        // And a symbol pointing outside every region at all.
        let mut img = FakeImage::new();
        img.sym("_ZTI6Widget", u64::MAX - 3).sym("_ZTV6Widget", 0);
        assert_eq!(recover(&img).len(), 1);
    }

    #[test]
    fn hostile_base_count_is_capped() {
        let mut img = FakeImage::new();
        with_abi_symbols(&mut img);
        img.typeinfo(0x2000, point(ABI_VMI), 0x2100, "7Derived")
            .u32(0x2010, 0)
            .u32(0x2014, u32::MAX);
        // Every base entry points at the one real typeinfo, so nothing
        // short-circuits the walk except the cap and the region end.
        for i in 0..512u64 {
            img.u64(0x2018 + i * 16, 0x2000).u64(0x2020 + i * 16, 2);
        }
        img.sym("_ZTI7Derived", 0x2000);

        let classes = recover(&img);
        assert_eq!(classes.len(), 1);
        assert!(
            classes[0].bases.len() <= MAX_BASES,
            "{} bases",
            classes[0].bases.len()
        );
        assert!(!render(&classes).is_empty());
    }

    #[test]
    fn garbage_typeinfo_bodies_never_panic() {
        let mut img = FakeImage::new();
        with_abi_symbols(&mut img);
        for (i, byte) in img.data.iter_mut().enumerate() {
            // Deterministic pseudo-garbage across the whole buffer.
            *byte = (i as u8).wrapping_mul(31).wrapping_add(7);
        }
        for va in [0x1000u64, 0x2000, 0x3000, DATA_END - 1, TEXT_VA] {
            img.sym("_ZTI7Garbage", va);
            img.sym("_ZTV7Garbage", va);
        }
        let classes = recover(&img);
        assert!(classes.len() <= MAX_CLASSES);
        let _ = render(&classes);
    }

    #[test]
    fn self_referential_and_cyclic_bases_terminate() {
        let mut img = FakeImage::new();
        with_abi_symbols(&mut img);
        // A is its own base; B and C name each other.
        img.typeinfo(0x2000, point(ABI_SI), 0x2200, "1A")
            .u64(0x2010, 0x2000)
            .typeinfo(0x2020, point(ABI_SI), 0x2220, "1B")
            .u64(0x2030, 0x2040)
            .typeinfo(0x2040, point(ABI_SI), 0x2240, "1C")
            .u64(0x2050, 0x2020)
            .sym("_ZTI1A", 0x2000)
            .sym("_ZTI1B", 0x2020)
            .sym("_ZTI1C", 0x2040);

        let classes = recover(&img);
        assert_eq!(classes.len(), 3);
        assert_eq!(classes[0].bases[0].typeinfo_va, 0x2000);
        assert_eq!(classes[0].bases[0].name.as_deref(), Some("A"));
        assert_eq!(classes[1].bases[0].name.as_deref(), Some("C"));
        assert_eq!(classes[2].bases[0].name.as_deref(), Some("B"));
        let _ = render(&classes);
    }

    #[test]
    fn self_referential_vtable_typeinfo_terminates() {
        // A vtable whose typeinfo slot points at itself, and a header whose
        // "secondary" repeats forever if the walk ever failed to advance.
        let mut img = FakeImage::new();
        img.u64(0x3000, 0)
            .u64(0x3008, 0x3000)
            .u64(0x3010, TEXT_VA)
            .u64(0x3018, (-8i64) as u64)
            .u64(0x3020, 0x3000)
            .u64(0x3028, TEXT_VA)
            .sym("_ZTV1A", 0x3000);
        let classes = recover(&img);
        assert!(classes.len() <= 1);
        let _ = render(&classes);
    }

    #[test]
    fn fixed_seed_fuzz_never_panics() {
        let mut state: u64 = 0x5eed_1234_abcd_0001;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for _ in 0..64 {
            let mut img = FakeImage::new();
            for byte in img.data.iter_mut() {
                *byte = next() as u8;
            }
            let payloads = ["6Widget", "N3foo3BarE", "St9exception", "1A", "*4Base"];
            for _ in 0..8 {
                let va = (next() as u64) % (IMG_SIZE as u64);
                let p = payloads[(next() as usize) % payloads.len()];
                match next() % 3 {
                    0 => img.sym(&format!("_ZTI{p}"), va),
                    1 => img.sym(&format!("_ZTV{p}"), va),
                    _ => img.sym(&format!("_ZTS{p}"), va),
                };
            }
            let classes = recover(&img);
            assert!(classes.len() <= MAX_CLASSES);
            for c in &classes {
                assert!(c.bases.len() <= MAX_BASES);
                assert!(c.vtables.len() <= MAX_SUBTABLES);
                for v in &c.vtables {
                    assert!(v.slots.len() <= MAX_SLOTS);
                }
            }
            let _ = render(&classes);
        }
    }

    // ---- Determinism and reporting --------------------------------------

    #[test]
    fn recovery_is_deterministic_and_sorted_by_typeinfo_va() {
        let mut img = FakeImage::new();
        with_abi_symbols(&mut img);
        img.typeinfo(0x2000, point(ABI_CLASS), 0x2300, "1C")
            .typeinfo(0x2020, point(ABI_CLASS), 0x2320, "1A")
            .typeinfo(0x2040, point(ABI_CLASS), 0x2340, "1B")
            // Symbols deliberately out of address order.
            .sym("_ZTI1B", 0x2040)
            .sym("_ZTI1C", 0x2000)
            .sym("_ZTI1A", 0x2020);

        let first = recover(&img);
        assert_eq!(
            first.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            ["C", "A", "B"]
        );
        assert_eq!(first, recover(&img));
        assert_eq!(render(&first), render(&recover(&img)));
    }

    #[test]
    fn render_reports_classes_bases_and_slot_counts() {
        let mut img = FakeImage::new();
        with_abi_symbols(&mut img);
        img.typeinfo(0x2000, point(ABI_CLASS), 0x2200, "4Base")
            .typeinfo(0x2020, point(ABI_SI), 0x2240, "7Derived")
            .u64(0x2030, 0x2000)
            .u64(0x3000, 0)
            .u64(0x3008, 0x2020)
            .u64(0x3010, TEXT_VA)
            .u64(0x3018, TEXT_VA + 8)
            .terminator(0x3020)
            .sym("_ZTI4Base", 0x2000)
            .sym("_ZTI7Derived", 0x2020)
            .sym("_ZTV7Derived", 0x3000);

        let text = render(&recover(&img));
        assert!(text.ends_with('\n'));
        assert!(text.contains("C++ classes: 2 (2 symbol-driven, 0 scan-recovered)"));
        assert!(text.contains("Base  [symbol, no bases]"));
        assert!(text.contains("Derived  [symbol, single inheritance]"));
        assert!(text.contains("typeinfo 0x0000000000002000"));
        assert!(text.contains("base     Base  typeinfo 0x0000000000002000  offset 0  public"));
        assert!(text.contains("vtable   0x0000000000003000  offset-to-top 0  slots 2"));
    }

    #[test]
    fn render_labels_scan_provenance() {
        let text = render(&recover(&stripped_image()));
        assert!(text.contains("(0 symbol-driven, 2 scan-recovered)"));
        assert!(text.contains("[scan, "));
    }

    // ---- Name rendering --------------------------------------------------

    #[test]
    fn class_names_come_from_the_type_grammar() {
        assert_eq!(class_name("6Widget").as_deref(), Some("Widget"));
        assert_eq!(class_name("N3foo3BarE").as_deref(), Some("foo::Bar"));
        assert_eq!(
            class_name("St9exception").as_deref(),
            Some("std::exception")
        );
        // A non-unique name's `*` prefix is not part of the encoding.
        assert_eq!(class_name("*6Widget").as_deref(), Some("Widget"));
        // Not a `<type>` encoding, or not covered by the grammar.
        assert_eq!(class_name("hello"), None);
        assert_eq!(class_name(""), None);
        assert_eq!(class_name("9Truncated_"), None);
        // The raw payload survives when the grammar refuses it.
        assert_eq!(class_name_or_raw("hello"), "hello");
    }
}
