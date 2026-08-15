//! Swift reflection metadata inventory from Mach-O `__swift5_*` sections.
//!
//! Clean-room layout from Apple's published Swift ABI documents
//! (`docs/ABI/TypeMetadata.rst`, `docs/ABI/Mangling.rst`) and the Mach-O
//! section names the Swift toolchain emits (`__swift5_types`,
//! `__swift5_typeref`, `__swift5_reflstr`, …). This early track:
//!
//! 1. Inventories every `__swift5_*` section (segment, name, VA, size).
//! 2. Walks `__swift5_types` as an array of 32-bit relative direct
//!    pointers to type context descriptors, recovering the nominal name
//!    string and kind (class / struct / enum) when the descriptor is
//!    readable.
//! 3. Optionally samples printable C-strings from `__swift5_reflstr`
//!    and mangled-looking UTF-8 runs from `__swift5_typeref`.
//!
//! Full Swift demangling (`$s…` → readable signatures) is deliberately
//! out of scope here — names are reported as recovered on disk.
//!
//! # Contract
//!
//! - Never panics on any input: every read is bounds-checked.
//! - Missing Swift sections yields an empty [`SwiftImage`] (`Ok`), not
//!   an error — most Mach-O images simply have no Swift metadata.
//! - Caps bound hostile counts; truncation is reported on the image.
//! - Typed [`SwiftError`] covers hard failures (section bytes that claim
//!   to exist but overrun the file). Per-descriptor corruption is
//!   skipped rather than aborting the whole recover.

use std::fmt;
use std::fmt::Write as _;

use crate::macho::{MachFile, Section64};

/// Default cap on type descriptors listed by [`recover`].
pub const DEFAULT_MAX_TYPES: usize = 4096;

/// Absolute upper bound on type descriptors walked from `__swift5_types`.
pub const MAX_TYPES: usize = 65_536;

/// Cap on reflection / typeref strings retained.
pub const MAX_STRINGS: usize = 4096;

/// Cap on a recovered C-string / mangled typeref fragment.
pub const MAX_CSTR_LEN: usize = 1024;

/// Cap on `__swift5_*` sections inventoried (hostile section tables).
pub const MAX_SECTIONS: usize = 256;

/// Cap on parent-descriptor walks when resolving a module prefix.
const MAX_PARENT_DEPTH: usize = 8;

/// Minimum size of a type context descriptor before kind-specific fields:
/// `Flags` + `Parent` + `Name` + `AccessFunction` + `Fields`.
const TYPE_DESC_MIN: u64 = 20;

/// `ContextDescriptorFlags` kind mask (low 5 bits).
const KIND_MASK: u32 = 0x1F;

/// Public `ContextDescriptorKind` values used here.
const KIND_MODULE: u32 = 0;
const KIND_CLASS: u32 = 16;
const KIND_STRUCT: u32 = 17;
const KIND_ENUM: u32 = 18;

/// Why Swift recovery refused to produce a usable image view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwiftError {
    /// The section header claims bytes that overrun the file buffer.
    SectionTruncated {
        segname: String,
        sectname: String,
        offset: u32,
        size: u64,
        file_len: usize,
    },
    /// A count or size field exceeded a hard safety cap.
    CapExceeded {
        what: &'static str,
        value: usize,
        cap: usize,
    },
}

impl fmt::Display for SwiftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SwiftError::SectionTruncated {
                segname,
                sectname,
                offset,
                size,
                file_len,
            } => write!(
                f,
                "Mach-O section {segname},{sectname} at file {offset:#x} size {size:#x} \
                 overruns file length {file_len:#x}"
            ),
            SwiftError::CapExceeded { what, value, cap } => {
                write!(f, "Swift {what} count {value} exceeds cap {cap}")
            }
        }
    }
}

impl std::error::Error for SwiftError {}

pub type SwiftResult<T> = std::result::Result<T, SwiftError>;

/// Kind of a recovered nominal type context descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwiftTypeKind {
    Class,
    Struct,
    Enum,
    /// Kind bits outside class/struct/enum, still named when readable.
    Other(u32),
}

impl SwiftTypeKind {
    fn from_flags(flags: u32) -> Self {
        match flags & KIND_MASK {
            KIND_CLASS => SwiftTypeKind::Class,
            KIND_STRUCT => SwiftTypeKind::Struct,
            KIND_ENUM => SwiftTypeKind::Enum,
            k => SwiftTypeKind::Other(k),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            SwiftTypeKind::Class => "class",
            SwiftTypeKind::Struct => "struct",
            SwiftTypeKind::Enum => "enum",
            SwiftTypeKind::Other(_) => "type",
        }
    }
}

impl fmt::Display for SwiftTypeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One inventoried `__swift5_*` Mach-O section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwiftSection {
    pub segname: String,
    pub sectname: String,
    pub addr: u64,
    pub size: u64,
    pub offset: u32,
}

/// One recovered nominal type from `__swift5_types`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwiftType {
    /// Virtual address of the type context descriptor.
    pub va: u64,
    /// Nominal name string from the descriptor's `Name` relative pointer.
    pub name: String,
    /// Module name from a parent module descriptor, when resolved.
    pub module: Option<String>,
    pub kind: SwiftTypeKind,
    /// Raw `ContextDescriptorFlags` word.
    pub flags: u32,
}

/// Recovered Swift metadata for one thin Mach-O image.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SwiftImage {
    /// All `__swift5_*` sections found (capped).
    pub sections: Vec<SwiftSection>,
    /// True when more than [`MAX_SECTIONS`] Swift sections existed.
    pub sections_capped: bool,
    /// Nominal types from `__swift5_types`.
    pub types: Vec<SwiftType>,
    /// True when `__swift5_types` held more entries than the type cap.
    pub types_capped: bool,
    /// File VA of `__swift5_types` when present.
    pub types_va: Option<u64>,
    /// Reflection field-name strings sampled from `__swift5_reflstr`.
    pub refl_strings: Vec<String>,
    /// True when reflection-string recovery hit [`MAX_STRINGS`].
    pub refl_strings_capped: bool,
    /// Mangled-looking fragments sampled from `__swift5_typeref`.
    pub typerefs: Vec<String>,
    /// True when typeref recovery hit [`MAX_STRINGS`].
    pub typerefs_capped: bool,
}

impl SwiftImage {
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty() && self.types.is_empty()
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// True when any `__swift5_*` section was present.
    pub fn has_swift_sections(&self) -> bool {
        !self.sections.is_empty()
    }
}

/// Recover Swift metadata with the default type cap.
pub fn recover(mach: &MachFile, data: &[u8]) -> SwiftResult<SwiftImage> {
    recover_capped(mach, data, DEFAULT_MAX_TYPES)
}

/// Recover Swift metadata, keeping at most `max_types` type descriptors.
pub fn recover_capped(mach: &MachFile, data: &[u8], max_types: usize) -> SwiftResult<SwiftImage> {
    let max_types = max_types.min(MAX_TYPES);
    let (sections, sections_capped) = inventory_sections(mach, data)?;
    if sections.is_empty() {
        return Ok(SwiftImage::default());
    }

    let mut image = SwiftImage {
        sections,
        sections_capped,
        ..SwiftImage::default()
    };

    if let Some(sect) = find_section(mach, "__swift5_types") {
        image.types_va = Some(sect.addr);
        let (types, capped) = parse_types_section(mach, data, sect, max_types)?;
        image.types = types;
        image.types_capped = capped;
    }

    if let Some(sect) = find_section(mach, "__swift5_reflstr") {
        let (strs, capped) = sample_cstrs(data, sect, MAX_STRINGS)?;
        image.refl_strings = strs;
        image.refl_strings_capped = capped;
    }

    if let Some(sect) = find_section(mach, "__swift5_typeref") {
        let (strs, capped) = sample_typerefs(data, sect, MAX_STRINGS)?;
        image.typerefs = strs;
        image.typerefs_capped = capped;
    }

    Ok(image)
}

/// Render a human-readable listing for `redump --swift`.
pub fn render(image: &SwiftImage, max_types: usize) -> String {
    let mut out = String::new();
    if !image.has_swift_sections() {
        out.push_str("  (no Swift metadata; no __swift5_* sections)\n");
        return out;
    }

    let _ = writeln!(
        out,
        "  {} __swift5_* section(s){}",
        image.sections.len(),
        if image.sections_capped {
            " (section list capped)"
        } else {
            ""
        }
    );
    for s in &image.sections {
        let _ = writeln!(
            out,
            "    {},{}  va {:#x}  size {:#x}  file {:#x}",
            s.segname, s.sectname, s.addr, s.size, s.offset
        );
    }

    if image.types_va.is_none() {
        let _ = writeln!(out, "  (no __swift5_types section; type list skipped)");
    } else if image.types.is_empty() {
        let _ = writeln!(out, "  (no type descriptors recovered from __swift5_types)");
    } else {
        let shown = image.types.len().min(max_types);
        let _ = writeln!(
            out,
            "  {} type(s) recovered{}",
            image.types.len(),
            if image.types_capped {
                " (type list capped)"
            } else {
                ""
            }
        );
        for ty in image.types.iter().take(shown) {
            let qual = match &ty.module {
                Some(m) if !m.is_empty() => format!("{m}.{}", ty.name),
                _ => ty.name.clone(),
            };
            let _ = writeln!(
                out,
                "    {} {}  // va {:#x}  flags {:#x}",
                ty.kind, qual, ty.va, ty.flags
            );
        }
        if shown < image.types.len() {
            let _ = writeln!(
                out,
                "  … {} more type(s) omitted (cap {max_types})",
                image.types.len() - shown
            );
        }
    }

    if !image.refl_strings.is_empty() {
        let _ = writeln!(
            out,
            "  {} reflection string(s){}",
            image.refl_strings.len(),
            if image.refl_strings_capped {
                " (capped)"
            } else {
                ""
            }
        );
        for s in image.refl_strings.iter().take(32) {
            let _ = writeln!(out, "    refl \"{s}\"");
        }
        if image.refl_strings.len() > 32 {
            let _ = writeln!(
                out,
                "    … {} more reflection string(s) omitted",
                image.refl_strings.len() - 32
            );
        }
    }

    if !image.typerefs.is_empty() {
        let _ = writeln!(
            out,
            "  {} typeref fragment(s){}",
            image.typerefs.len(),
            if image.typerefs_capped {
                " (capped)"
            } else {
                ""
            }
        );
        for s in image.typerefs.iter().take(32) {
            let _ = writeln!(out, "    typeref \"{s}\"");
        }
        if image.typerefs.len() > 32 {
            let _ = writeln!(
                out,
                "    … {} more typeref(s) omitted",
                image.typerefs.len() - 32
            );
        }
    }

    out
}

fn is_swift5_section(sectname: &str) -> bool {
    sectname.starts_with("__swift5_")
}

fn inventory_sections(mach: &MachFile, data: &[u8]) -> SwiftResult<(Vec<SwiftSection>, bool)> {
    let mut out = Vec::new();
    let mut capped = false;
    for seg in &mach.segments {
        for sect in &seg.sections {
            if !is_swift5_section(&sect.sectname) {
                continue;
            }
            // Validate claimed bytes exist (even if we only inventory).
            let _ = section_bytes(data, sect)?;
            if out.len() >= MAX_SECTIONS {
                capped = true;
                continue;
            }
            out.push(SwiftSection {
                segname: sect.segname.clone(),
                sectname: sect.sectname.clone(),
                addr: sect.addr,
                size: sect.size,
                offset: sect.offset,
            });
        }
    }
    Ok((out, capped))
}

fn find_section<'a>(mach: &'a MachFile, sectname: &str) -> Option<&'a Section64> {
    mach.segments
        .iter()
        .flat_map(|s| s.sections.iter())
        .find(|s| s.sectname == sectname)
}

fn section_bytes<'a>(data: &'a [u8], sect: &Section64) -> SwiftResult<&'a [u8]> {
    let start = sect.offset as usize;
    let size = sect.size as usize;
    let end = start.checked_add(size).ok_or_else(|| SwiftError::SectionTruncated {
        segname: sect.segname.clone(),
        sectname: sect.sectname.clone(),
        offset: sect.offset,
        size: sect.size,
        file_len: data.len(),
    })?;
    if end > data.len() {
        return Err(SwiftError::SectionTruncated {
            segname: sect.segname.clone(),
            sectname: sect.sectname.clone(),
            offset: sect.offset,
            size: sect.size,
            file_len: data.len(),
        });
    }
    Ok(&data[start..end])
}

fn bytes_at<'a>(mach: &MachFile, data: &'a [u8], va: u64, len: usize) -> Option<&'a [u8]> {
    let off = mach.vaddr_to_offset(va).ok()?;
    let end = off.checked_add(len)?;
    data.get(off..end)
}

fn read_u32(mach: &MachFile, data: &[u8], va: u64) -> Option<u32> {
    let b = bytes_at(mach, data, va, 4)?;
    Some(u32::from_le_bytes(b.try_into().ok()?))
}

fn read_i32(mach: &MachFile, data: &[u8], va: u64) -> Option<i32> {
    read_u32(mach, data, va).map(|u| u as i32)
}

fn relative_va(field_va: u64, rel: i32) -> Option<u64> {
    if rel == 0 {
        return None;
    }
    if rel >= 0 {
        field_va.checked_add(rel as u64)
    } else {
        field_va.checked_sub((-rel) as u64)
    }
}

fn read_cstr(mach: &MachFile, data: &[u8], va: u64) -> Option<String> {
    if va == 0 {
        return None;
    }
    let off = mach.vaddr_to_offset(va).ok()?;
    let rest = data.get(off..)?;
    let lim = rest.len().min(MAX_CSTR_LEN);
    let window = &rest[..lim];
    let len = window.iter().position(|&b| b == 0).unwrap_or(lim);
    if len == 0 {
        return None;
    }
    let bytes = &window[..len];
    if !bytes
        .iter()
        .all(|&b| (0x20..=0x7E).contains(&b) || b == b'\t')
    {
        return None;
    }
    Some(String::from_utf8_lossy(bytes).into_owned())
}

fn parse_types_section(
    mach: &MachFile,
    data: &[u8],
    sect: &Section64,
    max_types: usize,
) -> SwiftResult<(Vec<SwiftType>, bool)> {
    let bytes = section_bytes(data, sect)?;
    let n = bytes.len() / 4;
    if n > MAX_TYPES {
        return Err(SwiftError::CapExceeded {
            what: "types relative pointers",
            value: n,
            cap: MAX_TYPES,
        });
    }

    let mut types = Vec::new();
    let mut capped = false;
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        if types.len() >= max_types {
            capped = true;
            break;
        }
        let mut word = [0u8; 4];
        word.copy_from_slice(chunk);
        let rel = i32::from_le_bytes(word);
        let entry_va = sect.addr.checked_add((i as u64).saturating_mul(4));
        let Some(entry_va) = entry_va else {
            break;
        };
        let Some(desc_va) = relative_va(entry_va, rel) else {
            continue;
        };
        if types.iter().any(|t: &SwiftType| t.va == desc_va) {
            continue;
        }
        if let Some(ty) = parse_type_descriptor(mach, data, desc_va) {
            types.push(ty);
        }
    }
    if !capped && n > max_types {
        capped = true;
    }
    Ok((types, capped))
}

fn parse_type_descriptor(mach: &MachFile, data: &[u8], desc_va: u64) -> Option<SwiftType> {
    let _ = bytes_at(mach, data, desc_va, TYPE_DESC_MIN as usize)?;
    let flags = read_u32(mach, data, desc_va)?;
    let kind = SwiftTypeKind::from_flags(flags);
    // Name is at +8 (after Flags + Parent).
    let name_rel = read_i32(mach, data, desc_va + 8)?;
    let name_va = relative_va(desc_va + 8, name_rel)?;
    let name = read_cstr(mach, data, name_va)?;
    if !looks_like_swift_ident(&name) {
        return None;
    }
    let module = resolve_module_name(mach, data, desc_va);
    Some(SwiftType {
        va: desc_va,
        name,
        module,
        kind,
        flags,
    })
}

fn looks_like_swift_ident(s: &str) -> bool {
    let b = s.as_bytes();
    if b.is_empty() || b.len() > 256 {
        return false;
    }
    // Nominal names are Swift identifiers; reject mangled `$s…` here —
    // those belong in typeref sampling.
    if b[0] == b'$' {
        return false;
    }
    b.iter().all(|&c| {
        c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || c == b'$' || c == b' '
    }) && b.iter().any(|&c| c.is_ascii_alphanumeric() || c == b'_')
}

fn resolve_module_name(mach: &MachFile, data: &[u8], desc_va: u64) -> Option<String> {
    let mut cur = desc_va;
    for _ in 0..MAX_PARENT_DEPTH {
        let parent_rel = read_i32(mach, data, cur + 4)?;
        let parent_va = relative_va(cur + 4, parent_rel)?;
        let flags = read_u32(mach, data, parent_va)?;
        if flags & KIND_MASK == KIND_MODULE {
            let name_rel = read_i32(mach, data, parent_va + 8)?;
            let name_va = relative_va(parent_va + 8, name_rel)?;
            return read_cstr(mach, data, name_va).filter(|s| looks_like_swift_ident(s));
        }
        cur = parent_va;
    }
    None
}

fn sample_cstrs(
    data: &[u8],
    sect: &Section64,
    max: usize,
) -> SwiftResult<(Vec<String>, bool)> {
    let bytes = section_bytes(data, sect)?;
    let mut out = Vec::new();
    let mut capped = false;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i] != 0 {
            i += 1;
        }
        let len = (i - start).min(MAX_CSTR_LEN);
        let slice = &bytes[start..start + len];
        if slice
            .iter()
            .all(|&b| (0x20..=0x7E).contains(&b))
            && !slice.is_empty()
        {
            if out.len() >= max {
                capped = true;
                break;
            }
            let s = String::from_utf8_lossy(slice).into_owned();
            if !out.iter().any(|e| e == &s) {
                out.push(s);
            }
        }
        if i < bytes.len() {
            i += 1; // skip NUL
        }
    }
    Ok((out, capped))
}

fn sample_typerefs(
    data: &[u8],
    sect: &Section64,
    max: usize,
) -> SwiftResult<(Vec<String>, bool)> {
    let bytes = section_bytes(data, sect)?;
    let mut out = Vec::new();
    let mut capped = false;
    let mut i = 0;
    while i < bytes.len() {
        // Swift typerefs are often NUL-terminated mangled strings starting
        // with `$s` / `$S` (new / old mangling). Also accept printable runs.
        if bytes[i] == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i] != 0 {
            i += 1;
        }
        let len = (i - start).min(MAX_CSTR_LEN);
        let slice = &bytes[start..start + len];
        if slice.len() >= 2
            && slice
                .iter()
                .all(|&b| (0x20..=0x7E).contains(&b))
        {
            let s = String::from_utf8_lossy(slice).into_owned();
            // Prefer mangled forms; still keep short printable fragments.
            let keep = s.starts_with("$s")
                || s.starts_with("$S")
                || s.starts_with("_T")
                || (s.len() >= 3 && s.chars().all(|c| c.is_ascii_alphanumeric() || "_$.".contains(c)));
            if keep {
                if out.len() >= max {
                    capped = true;
                    break;
                }
                if !out.iter().any(|e| e == &s) {
                    out.push(s);
                }
            }
        }
        if i < bytes.len() {
            i += 1;
        }
    }
    Ok((out, capped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macho::{
        LC_SEGMENT_64, MH_DYLDLINK, MH_MAGIC_64, MH_NOUNDEFS, MH_PIE, MH_TWOLEVEL, VM_PROT_EXECUTE,
        VM_PROT_READ,
    };

    const IMG_SIZE: usize = 0x800;
    const TEXT_VM: u64 = 0x1_0000_0000;

    // Content offsets inside the file / VAs (all in the first 0x400 mapped
    // as __TEXT file-backed from offset 0).
    const NAME_OFF: usize = 0x280; // "Demo\0"
    const MODNAME_OFF: usize = 0x288; // "App\0"
    const REFL_OFF: usize = 0x290; // "count\0"
    const TYPREF_OFF: usize = 0x2A0; // "$s3App4DemoV\0"
    const MODULE_OFF: usize = 0x2C0; // module descriptor
    const STRUCT_OFF: usize = 0x2D0; // struct descriptor
    const TYPES_OFF: usize = 0x300; // __swift5_types (one rel32)
    const REFLSECT_OFF: usize = 0x310;
    const TYPREFSECT_OFF: usize = 0x320;

    fn put(img: &mut [u8], off: usize, bytes: &[u8]) {
        img[off..off + bytes.len()].copy_from_slice(bytes);
    }
    fn put32(img: &mut [u8], off: usize, v: u32) {
        put(img, off, &v.to_le_bytes());
    }
    fn put64(img: &mut [u8], off: usize, v: u64) {
        put(img, off, &v.to_le_bytes());
    }
    fn put_i32(img: &mut [u8], off: usize, v: i32) {
        put32(img, off, v as u32);
    }

    fn write_section(
        img: &mut [u8],
        at: usize,
        sectname: &[u8],
        segname: &[u8],
        addr: u64,
        size: u64,
        offset: u32,
    ) {
        put(img, at, sectname);
        put(img, at + 16, segname);
        put64(img, at + 32, addr);
        put64(img, at + 40, size);
        put32(img, at + 48, offset);
        put32(img, at + 52, 2); // align 2^2
    }

    fn rel32(from_va: u64, to_va: u64) -> i32 {
        let delta = to_va as i64 - from_va as i64;
        i32::try_from(delta).expect("synthetic rel in range")
    }

    /// Thin arm64 Mach-O with one Swift struct `App.Demo` and sample
    /// `__swift5_reflstr` / `__swift5_typeref` strings.
    fn synthetic_swift_macho() -> Vec<u8> {
        let mut img = vec![0u8; IMG_SIZE];

        // One LC_SEGMENT_64 __TEXT with 5 sections.
        // cmdsize = 72 + 5*80 = 472.
        let ncmds = 1u32;
        let sizeofcmds = 472u32;

        put32(&mut img, 0, MH_MAGIC_64);
        put32(&mut img, 4, 0x0100_000C); // ARM64
        put32(&mut img, 8, 0);
        put32(&mut img, 12, 2); // MH_EXECUTE
        put32(&mut img, 16, ncmds);
        put32(&mut img, 20, sizeofcmds);
        put32(&mut img, 24, MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE);

        let t = 0x20;
        put32(&mut img, t, LC_SEGMENT_64);
        put32(&mut img, t + 4, 472);
        put(&mut img, t + 8, b"__TEXT");
        put64(&mut img, t + 24, TEXT_VM);
        put64(&mut img, t + 32, 0x400);
        put64(&mut img, t + 40, 0);
        put64(&mut img, t + 48, 0x400);
        put32(&mut img, t + 56, VM_PROT_READ | VM_PROT_EXECUTE);
        put32(&mut img, t + 60, VM_PROT_READ | VM_PROT_EXECUTE);
        put32(&mut img, t + 64, 5); // nsects

        write_section(
            &mut img,
            t + 72,
            b"__text",
            b"__TEXT",
            TEXT_VM + 0x200,
            0x20,
            0x200,
        );
        write_section(
            &mut img,
            t + 72 + 80,
            b"__const",
            b"__TEXT",
            TEXT_VM + MODULE_OFF as u64,
            0x40,
            MODULE_OFF as u32,
        );
        write_section(
            &mut img,
            t + 72 + 160,
            b"__swift5_types",
            b"__TEXT",
            TEXT_VM + TYPES_OFF as u64,
            4,
            TYPES_OFF as u32,
        );
        write_section(
            &mut img,
            t + 72 + 240,
            b"__swift5_reflstr",
            b"__TEXT",
            TEXT_VM + REFLSECT_OFF as u64,
            8,
            REFLSECT_OFF as u32,
        );
        write_section(
            &mut img,
            t + 72 + 320,
            b"__swift5_typeref",
            b"__TEXT",
            TEXT_VM + TYPREFSECT_OFF as u64,
            0x10,
            TYPREFSECT_OFF as u32,
        );

        // Strings.
        put(&mut img, NAME_OFF, b"Demo\0");
        put(&mut img, MODNAME_OFF, b"App\0");
        put(&mut img, REFL_OFF, b"count\0");
        put(&mut img, TYPREF_OFF, b"$s3App4DemoV\0");

        // Module descriptor at MODULE_OFF.
        let mod_va = TEXT_VM + MODULE_OFF as u64;
        put32(&mut img, MODULE_OFF, KIND_MODULE); // Flags
        put_i32(&mut img, MODULE_OFF + 4, 0); // Parent null
        put_i32(
            &mut img,
            MODULE_OFF + 8,
            rel32(mod_va + 8, TEXT_VM + MODNAME_OFF as u64),
        );

        // Struct descriptor at STRUCT_OFF.
        let struct_va = TEXT_VM + STRUCT_OFF as u64;
        put32(&mut img, STRUCT_OFF, KIND_STRUCT); // Flags
        put_i32(
            &mut img,
            STRUCT_OFF + 4,
            rel32(struct_va + 4, mod_va),
        ); // Parent → module
        put_i32(
            &mut img,
            STRUCT_OFF + 8,
            rel32(struct_va + 8, TEXT_VM + NAME_OFF as u64),
        ); // Name
        put_i32(&mut img, STRUCT_OFF + 12, 0); // AccessFunction
        put_i32(&mut img, STRUCT_OFF + 16, 0); // Fields
        put32(&mut img, STRUCT_OFF + 20, 0); // NumFields
        put32(&mut img, STRUCT_OFF + 24, 0); // FieldOffsetVectorOffset

        // __swift5_types: one relative pointer to the struct descriptor.
        let types_va = TEXT_VM + TYPES_OFF as u64;
        put_i32(&mut img, TYPES_OFF, rel32(types_va, struct_va));

        // Reflection / typeref section payloads (copy of the string bytes).
        put(&mut img, REFLSECT_OFF, b"count\0\0\0");
        put(&mut img, TYPREFSECT_OFF, b"$s3App4DemoV\0\0\0");

        img
    }

    #[test]
    fn empty_macho_is_ok_empty() {
        // Minimal __TEXT with only __text — no Swift sections.
        let mut img = vec![0u8; 0x200];
        put32(&mut img, 0, MH_MAGIC_64);
        put32(&mut img, 4, 0x0100_000C);
        put32(&mut img, 12, 2);
        put32(&mut img, 16, 1);
        put32(&mut img, 20, 152); // 72 + 80
        put32(&mut img, 24, MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE);
        let t = 0x20;
        put32(&mut img, t, LC_SEGMENT_64);
        put32(&mut img, t + 4, 152);
        put(&mut img, t + 8, b"__TEXT");
        put64(&mut img, t + 24, TEXT_VM);
        put64(&mut img, t + 32, 0x200);
        put64(&mut img, t + 40, 0);
        put64(&mut img, t + 48, 0x200);
        put32(&mut img, t + 56, VM_PROT_READ | VM_PROT_EXECUTE);
        put32(&mut img, t + 60, VM_PROT_READ | VM_PROT_EXECUTE);
        put32(&mut img, t + 64, 1);
        write_section(
            &mut img,
            t + 72,
            b"__text",
            b"__TEXT",
            TEXT_VM + 0x100,
            0x10,
            0x100,
        );

        let mach = MachFile::parse(&img).unwrap();
        let sw = recover(&mach, &img).unwrap();
        assert!(sw.is_empty());
        assert!(!sw.has_swift_sections());
        let text = render(&sw, 16);
        assert!(text.contains("no __swift5_*"), "{text}");
    }

    #[test]
    fn synthetic_recovers_struct_and_sections() {
        let img = synthetic_swift_macho();
        let mach = MachFile::parse(&img).unwrap();
        assert!(mach.section_by_name("__TEXT", "__swift5_types").is_some());

        let sw = recover(&mach, &img).expect("swift");
        assert!(sw.has_swift_sections());
        assert_eq!(sw.sections.len(), 3);
        assert!(
            sw.sections
                .iter()
                .any(|s| s.sectname == "__swift5_types")
        );
        assert_eq!(sw.len(), 1);
        assert_eq!(sw.types[0].name, "Demo");
        assert_eq!(sw.types[0].module.as_deref(), Some("App"));
        assert_eq!(sw.types[0].kind, SwiftTypeKind::Struct);
        assert!(sw.refl_strings.iter().any(|s| s == "count"));
        assert!(sw.typerefs.iter().any(|s| s == "$s3App4DemoV"));
    }

    #[test]
    fn render_lists_types_and_inventory() {
        let img = synthetic_swift_macho();
        let mach = MachFile::parse(&img).unwrap();
        let sw = recover(&mach, &img).unwrap();
        let text = render(&sw, 16);
        assert!(text.contains("__swift5_types"), "{text}");
        assert!(text.contains("struct App.Demo"), "{text}");
        assert!(text.contains("refl \"count\""), "{text}");
        assert!(text.contains("typeref \"$s3App4DemoV\""), "{text}");
    }

    #[test]
    fn type_cap_truncates_without_panic() {
        let img = synthetic_swift_macho();
        let mach = MachFile::parse(&img).unwrap();
        let sw = recover_capped(&mach, &img, 0).unwrap();
        assert!(sw.types.is_empty());
        assert!(sw.types_capped);
        assert!(sw.has_swift_sections());
    }

    #[test]
    fn section_overrun_is_typed_error() {
        let mut img = synthetic_swift_macho();
        // Lie in __swift5_types size so claimed bytes overrun the file.
        let t = 0x20;
        put64(&mut img, t + 72 + 160 + 40, 0x10_000);
        let mach = MachFile::parse(&img).unwrap();
        let err = recover(&mach, &img).unwrap_err();
        assert!(
            matches!(err, SwiftError::SectionTruncated { .. }),
            "{err}"
        );
    }
}
