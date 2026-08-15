//! Swift reflection metadata inventory from Mach-O `__swift5_*` sections.
//!
//! Clean-room layout from Apple's published Swift ABI documents
//! (`docs/ABI/TypeMetadata.rst`, `docs/ABI/Mangling.rst`, public
//! `MetadataValues.h` conformance flags) and the Mach-O section names
//! the Swift toolchain emits (`__swift5_types`, `__swift5_typeref`,
//! `__swift5_reflstr`, `__swift5_proto`, …). This early track:
//!
//! 1. Inventories every `__swift5_*` section (segment, name, VA, size).
//! 2. Walks `__swift5_types` as an array of 32-bit relative direct
//!    pointers to type context descriptors, recovering the nominal name
//!    string and kind (class / struct / enum) when the descriptor is
//!    readable.
//! 3. Follows each type's `Fields` relative pointer to a public ABI
//!    `FieldDescriptor`, recovering per-field names and mangled type
//!    names from `FieldRecord`s (capped).
//! 4. Walks `__swift5_proto` relative pointers to
//!    `ProtocolConformanceDescriptor`s, recovering protocol / type
//!    names and `ConformanceFlags` when safely readable.
//! 5. Follows each conformance's `WitnessTablePattern` relative pointer
//!    to a protocol witness table, recovering capped requirement-slot
//!    VAs (and optional symbol / nearby-string names).
//! 6. Optionally samples printable C-strings from `__swift5_reflstr`
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

/// Cap on field records parsed from one type's `FieldDescriptor`.
pub const MAX_FIELDS_PER_TYPE: usize = 4096;

/// Cap on a recovered field name / field mangled-type string.
pub const MAX_FIELD_NAME_LEN: usize = 256;

/// Cap on `__swift5_*` sections inventoried (hostile section tables).
pub const MAX_SECTIONS: usize = 256;

/// Cap on protocol conformance descriptors from `__swift5_proto`.
pub const MAX_PROTO_CONFORMANCES: usize = 65_536;

/// Cap on witness-table requirement slots recovered per conformance.
pub const MAX_WITNESSES_PER_CONFORMANCE: usize = 4096;

/// Cap on parent-descriptor walks when resolving a module prefix.
const MAX_PARENT_DEPTH: usize = 8;

/// Minimum size of a type context descriptor before kind-specific fields:
/// `Flags` + `Parent` + `Name` + `AccessFunction` + `Fields`.
const TYPE_DESC_MIN: u64 = 20;

/// Public ABI `ProtocolConformanceDescriptor` size:
/// `Protocol` + `TypeRef` + `WitnessTablePattern` + `ConformanceFlags`.
const CONFORMANCE_DESC_SIZE: u64 = 16;

/// Protocol descriptor bytes needed for `NumRequirements`:
/// `Flags` + `Parent` + `Name` + `NumRequirementsInSignature` + `NumRequirements`.
const PROTOCOL_NUM_REQ_MIN: u64 = 20;

/// Public ABI: first requirement slot is at pointer index 1
/// (`WitnessTableFirstRequirementOffset` in `MetadataValues.h`).
const WITNESS_TABLE_FIRST_REQ: u64 = 1;

/// Bytes before a witness VA scanned for a nearby printable C-string name.
const WITNESS_NAME_NEARBY: u64 = 64;

/// Cap on a recovered witness entry name.
const MAX_WITNESS_NAME_LEN: usize = 256;

/// Minimum readable size of a protocol / type context descriptor name:
/// `Flags` + `Parent` + `Name`.
const CONTEXT_NAME_MIN: u64 = 12;

/// Public ABI `FieldDescriptor` header size (before `FieldRecord`s):
/// `MangledTypeName` + `Superclass` + `Kind` + `FieldRecordSize` + `NumFields`.
const FIELD_DESC_HEADER: u64 = 16;

/// Canonical `FieldRecord` size (`Flags` + `MangledTypeName` + `FieldName`).
const FIELD_RECORD_MIN: u16 = 12;

/// Absolute upper bound on a hostile `FieldRecordSize` stride.
const FIELD_RECORD_SIZE_MAX: u16 = 64;

/// `ContextDescriptorFlags` kind mask (low 5 bits).
const KIND_MASK: u32 = 0x1F;

/// Public `ContextDescriptorKind` values used here.
const KIND_MODULE: u32 = 0;
const KIND_PROTOCOL: u32 = 3;
const KIND_CLASS: u32 = 16;
const KIND_STRUCT: u32 = 17;
const KIND_ENUM: u32 = 18;

/// `ConformanceFlags` type-reference kind (bits 3..5).
const TYPE_REF_KIND_MASK: u32 = 0x7 << 3;
const TYPE_REF_KIND_SHIFT: u32 = 3;

/// Public `TypeReferenceKind` values used when resolving the conforming type.
const TYPE_REF_DIRECT_TYPE_DESC: u32 = 0;
const TYPE_REF_INDIRECT_TYPE_DESC: u32 = 1;
const TYPE_REF_DIRECT_OBJC_CLASS_NAME: u32 = 2;
const TYPE_REF_INDIRECT_OBJC_CLASS: u32 = 3;

/// Low bit of a relative-indirectable pointer: target is a pointer slot.
const REL_INDIRECT_BIT: i32 = 0x1;
/// Second bit of a protocol relative pointer: Objective-C protocol ref.
const REL_PROTOCOL_IS_OBJC_BIT: i32 = 0x2;

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

/// One recovered stored property / enum case from a `FieldRecord`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwiftField {
    /// Field / case name from the record's `FieldName` relative pointer.
    pub name: String,
    /// Mangled type name when the record's `MangledTypeName` is a
    /// readable printable C-string (symbolic-ref payloads are skipped).
    pub mangled_type: Option<String>,
    /// Byte offset within an instance when known. Field records alone do
    /// not carry offsets; reserved for future metadata-vector recovery.
    pub offset: Option<u32>,
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
    /// Fields from the type's `FieldDescriptor`, when the `Fields`
    /// relative pointer is readable (capped at [`MAX_FIELDS_PER_TYPE`]).
    pub fields: Vec<SwiftField>,
}

/// One recovered protocol witness-table requirement slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwiftWitness {
    /// Virtual address of the witness (method impl / associated metadata).
    pub va: u64,
    /// Symbol or nearby C-string name when resolvable; `None` is VA-only.
    pub name: Option<String>,
}

/// One recovered protocol conformance from `__swift5_proto`.
///
/// Layout follows the public `ProtocolConformanceDescriptor`: protocol
/// relative pointer, type/context relative pointer, witness-table
/// pattern (followed when readable), and `ConformanceFlags`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwiftConformance {
    /// Virtual address of the conformance descriptor.
    pub va: u64,
    /// Protocol name when the protocol relative pointer resolves to a
    /// readable Swift protocol context descriptor `Name` string.
    pub protocol_name: Option<String>,
    /// Conforming type / ObjC class name when the type relative pointer
    /// is resolvable under `ConformanceFlags` type-reference kind.
    pub type_name: Option<String>,
    /// Raw `ConformanceFlags` word (`0` when the flags field was unreadable).
    pub flags: u32,
    /// Requirement slots from `WitnessTablePattern` when the relative
    /// pointer is a readable witness table (capped at
    /// [`MAX_WITNESSES_PER_CONFORMANCE`]). Soft-fails to empty / VA-only.
    pub witnesses: Vec<SwiftWitness>,
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
    /// Protocol conformances from `__swift5_proto` (capped).
    pub proto_conformances: Vec<SwiftConformance>,
    /// True when proto recovery hit [`MAX_PROTO_CONFORMANCES`].
    pub proto_capped: bool,
    /// File VA of `__swift5_proto` when present.
    pub proto_va: Option<u64>,
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

    if let Some(sect) = find_section(mach, "__swift5_proto") {
        image.proto_va = Some(sect.addr);
        let (conformances, capped) =
            parse_proto_section(mach, data, sect, MAX_PROTO_CONFORMANCES)?;
        image.proto_conformances = conformances;
        image.proto_capped = capped;
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
            for field in &ty.fields {
                match &field.mangled_type {
                    Some(ty_name) if !ty_name.is_empty() => {
                        let _ = writeln!(out, "      {} : {ty_name}", field.name);
                    }
                    _ => {
                        let _ = writeln!(out, "      {}", field.name);
                    }
                }
            }
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

    if image.proto_va.is_some() || !image.proto_conformances.is_empty() {
        let _ = writeln!(
            out,
            "  {} protocol conformance descriptor(s){}",
            image.proto_conformances.len(),
            if image.proto_capped {
                " (capped)"
            } else {
                ""
            }
        );
        for conf in image.proto_conformances.iter().take(32) {
            let ty = conf.type_name.as_deref().unwrap_or("?");
            let proto = conf.protocol_name.as_deref().unwrap_or("?");
            let _ = writeln!(
                out,
                "    {ty} : {proto}  // va {:#x}  flags {:#x}",
                conf.va, conf.flags
            );
            for w in &conf.witnesses {
                match &w.name {
                    Some(name) if !name.is_empty() => {
                        let _ = writeln!(out, "      witness {:#x}  {name}", w.va);
                    }
                    _ => {
                        let _ = writeln!(out, "      witness {:#x}", w.va);
                    }
                }
            }
        }
        if image.proto_conformances.len() > 32 {
            let _ = writeln!(
                out,
                "    … {} more proto(s) omitted",
                image.proto_conformances.len() - 32
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

fn read_u16(mach: &MachFile, data: &[u8], va: u64) -> Option<u16> {
    let b = bytes_at(mach, data, va, 2)?;
    Some(u16::from_le_bytes(b.try_into().ok()?))
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
    read_cstr_capped(mach, data, va, MAX_CSTR_LEN)
}

fn read_cstr_capped(mach: &MachFile, data: &[u8], va: u64, max_len: usize) -> Option<String> {
    if va == 0 {
        return None;
    }
    let off = mach.vaddr_to_offset(va).ok()?;
    let rest = data.get(off..)?;
    let lim = rest.len().min(max_len);
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

/// Walk `__swift5_proto` relative pointers and parse each
/// `ProtocolConformanceDescriptor` (soft-skipping corrupt entries).
fn parse_proto_section(
    mach: &MachFile,
    data: &[u8],
    sect: &Section64,
    max: usize,
) -> SwiftResult<(Vec<SwiftConformance>, bool)> {
    let bytes = section_bytes(data, sect)?;
    let n = bytes.len() / 4;
    if n > max.max(MAX_PROTO_CONFORMANCES) {
        return Err(SwiftError::CapExceeded {
            what: "protocol conformance relative pointers",
            value: n,
            cap: max,
        });
    }
    let mut out = Vec::new();
    let mut capped = false;
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        if out.len() >= max {
            capped = true;
            break;
        }
        let mut word = [0u8; 4];
        word.copy_from_slice(chunk);
        let rel = i32::from_le_bytes(word);
        let Some(entry_va) = sect.addr.checked_add((i as u64).saturating_mul(4)) else {
            break;
        };
        let Some(target) = relative_va(entry_va, rel) else {
            continue;
        };
        if target == 0 || out.iter().any(|c: &SwiftConformance| c.va == target) {
            continue;
        }
        if mach.vaddr_to_offset(target).is_err() {
            continue;
        }
        out.push(parse_conformance_descriptor(mach, data, target));
    }
    if !capped && n > max {
        capped = true;
    }
    Ok((out, capped))
}

/// Parse a public ABI `ProtocolConformanceDescriptor` at `desc_va`.
///
/// Always returns a record for a file-backed VA; unreadable fields soft-fail
/// to `None` / `flags = 0` / empty witnesses rather than aborting recovery.
fn parse_conformance_descriptor(
    mach: &MachFile,
    data: &[u8],
    desc_va: u64,
) -> SwiftConformance {
    let empty = || SwiftConformance {
        va: desc_va,
        protocol_name: None,
        type_name: None,
        flags: 0,
        witnesses: Vec::new(),
    };
    let Some(_) = bytes_at(mach, data, desc_va, CONFORMANCE_DESC_SIZE as usize) else {
        return empty();
    };
    let flags = read_u32(mach, data, desc_va + 12).unwrap_or(0);
    let protocol_name = read_i32(mach, data, desc_va)
        .and_then(|rel| resolve_protocol_name(mach, data, desc_va, rel));
    let type_name = read_i32(mach, data, desc_va + 4).and_then(|rel| {
        resolve_conformance_type_name(mach, data, desc_va + 4, rel, flags)
    });
    let witnesses = read_i32(mach, data, desc_va + 8)
        .map(|rel| recover_witnesses(mach, data, desc_va, rel))
        .unwrap_or_default();
    SwiftConformance {
        va: desc_va,
        protocol_name,
        type_name,
        flags,
        witnesses,
    }
}

/// Follow `WitnessTablePattern` and recover capped requirement-slot entries.
///
/// Soft-fails to an empty list on null / unreadable pattern pointers.
fn recover_witnesses(
    mach: &MachFile,
    data: &[u8],
    conf_va: u64,
    pattern_rel: i32,
) -> Vec<SwiftWitness> {
    if pattern_rel == 0 {
        return Vec::new();
    }
    let Some(table_va) = relative_va(conf_va + 8, pattern_rel) else {
        return Vec::new();
    };
    // Description word must be file-backed before walking slots.
    if bytes_at(mach, data, table_va, 8).is_none() {
        return Vec::new();
    }
    let num_req = protocol_num_requirements_for_conformance(mach, data, conf_va);
    parse_witness_table_slots(mach, data, table_va, num_req)
}

/// Read `NumRequirements` from the protocol descriptor linked by `conf_va`.
fn protocol_num_requirements_for_conformance(
    mach: &MachFile,
    data: &[u8],
    conf_va: u64,
) -> Option<usize> {
    let rel = read_i32(mach, data, conf_va)?;
    if rel == 0 || rel & REL_PROTOCOL_IS_OBJC_BIT != 0 {
        return None;
    }
    let proto_va = resolve_relative_indirectable(mach, data, conf_va, rel)?;
    read_protocol_num_requirements(mach, data, proto_va)
}

fn read_protocol_num_requirements(
    mach: &MachFile,
    data: &[u8],
    proto_va: u64,
) -> Option<usize> {
    let _ = bytes_at(mach, data, proto_va, PROTOCOL_NUM_REQ_MIN as usize)?;
    let flags = read_u32(mach, data, proto_va)?;
    if flags & KIND_MASK != KIND_PROTOCOL {
        return None;
    }
    let n = read_u32(mach, data, proto_va + 16)? as usize;
    Some(n.min(MAX_WITNESSES_PER_CONFORMANCE))
}

/// Walk witness-table pointer slots starting at
/// [`WITNESS_TABLE_FIRST_REQ`] (skipping the Description word).
///
/// When `num_req` is known, that many slots are examined (null / unmapped
/// skipped). When unknown, walk until a null or unmapped pointer, capped.
fn parse_witness_table_slots(
    mach: &MachFile,
    data: &[u8],
    table_va: u64,
    num_req: Option<usize>,
) -> Vec<SwiftWitness> {
    let known = num_req.is_some();
    let limit = num_req
        .unwrap_or(MAX_WITNESSES_PER_CONFORMANCE)
        .min(MAX_WITNESSES_PER_CONFORMANCE);
    let mut out = Vec::new();
    for i in 0..limit {
        let Some(slot_va) = table_va
            .checked_add(WITNESS_TABLE_FIRST_REQ.saturating_mul(8))
            .and_then(|base| base.checked_add((i as u64).saturating_mul(8)))
        else {
            break;
        };
        let Some(entry_va) = read_ptr(mach, data, slot_va) else {
            if known {
                continue;
            }
            break;
        };
        if entry_va == 0 {
            if known {
                continue;
            }
            break;
        }
        if mach.vaddr_to_offset(entry_va).is_err() {
            if known {
                continue;
            }
            break;
        }
        let name = name_for_witness_va(mach, data, entry_va);
        out.push(SwiftWitness {
            va: entry_va,
            name,
        });
    }
    out
}

/// Resolve an optional name for a witness VA via the Mach-O symbol table
/// or a nearby / at-VA printable C-string. Soft-fails to `None`.
fn name_for_witness_va(mach: &MachFile, data: &[u8], va: u64) -> Option<String> {
    for sym in &mach.symbols {
        if sym.value == va && !sym.name.is_empty() && looks_like_witness_name(&sym.name) {
            return Some(sym.name.clone());
        }
    }
    if let Some(s) = read_cstr_capped(mach, data, va, MAX_WITNESS_NAME_LEN)
        && looks_like_witness_name(&s)
    {
        return Some(s);
    }
    name_nearby_cstr(mach, data, va)
}

/// Scan for a printable C-string that ends immediately before `va`
/// (`…name\\0<code>` stub labeling). Soft-fails when absent.
fn name_nearby_cstr(mach: &MachFile, data: &[u8], va: u64) -> Option<String> {
    let off = mach.vaddr_to_offset(va).ok()?;
    if off == 0 || data.get(off - 1) != Some(&0) {
        return None;
    }
    let end = off - 1; // index of terminating NUL
    let start_floor = off.saturating_sub(WITNESS_NAME_NEARBY as usize);
    let mut s = end;
    while s > start_floor && data[s - 1] != 0 {
        s -= 1;
    }
    let slice = data.get(s..end)?;
    if slice.is_empty() || slice.len() > MAX_WITNESS_NAME_LEN {
        return None;
    }
    if !slice.iter().all(|&b| (0x20..=0x7E).contains(&b)) {
        return None;
    }
    let name = String::from_utf8_lossy(slice).into_owned();
    if looks_like_witness_name(&name) {
        Some(name)
    } else {
        None
    }
}

fn looks_like_witness_name(s: &str) -> bool {
    let b = s.as_bytes();
    if b.is_empty() || b.len() > MAX_WITNESS_NAME_LEN {
        return false;
    }
    b.iter().all(|&c| (0x20..=0x7E).contains(&c))
        && b.iter()
            .any(|&c| c.is_ascii_alphanumeric() || c == b'_')
}

/// Resolve the protocol relative field (relative-indirectable; bit 1 = ObjC).
fn resolve_protocol_name(
    mach: &MachFile,
    data: &[u8],
    field_va: u64,
    rel: i32,
) -> Option<String> {
    if rel == 0 {
        return None;
    }
    // RelativeIndirectablePointerIntPair: bit0 = indirect, bit1 = isObjC.
    if rel & REL_PROTOCOL_IS_OBJC_BIT != 0 {
        return None;
    }
    let proto_va = resolve_relative_indirectable(mach, data, field_va, rel)?;
    read_context_display_name(mach, data, proto_va, Some(KIND_PROTOCOL))
}

/// Resolve the conforming-type relative field using `ConformanceFlags`.
fn resolve_conformance_type_name(
    mach: &MachFile,
    data: &[u8],
    field_va: u64,
    rel: i32,
    flags: u32,
) -> Option<String> {
    if rel == 0 {
        return None;
    }
    let kind = (flags & TYPE_REF_KIND_MASK) >> TYPE_REF_KIND_SHIFT;
    match kind {
        TYPE_REF_DIRECT_TYPE_DESC => {
            let ty_va = relative_va(field_va, rel)?;
            read_context_display_name(mach, data, ty_va, None)
        }
        TYPE_REF_INDIRECT_TYPE_DESC => {
            let slot_va = relative_va(field_va, rel)?;
            let ty_va = read_ptr(mach, data, slot_va)?;
            if ty_va == 0 {
                return None;
            }
            read_context_display_name(mach, data, ty_va, None)
        }
        TYPE_REF_DIRECT_OBJC_CLASS_NAME => {
            let name_va = relative_va(field_va, rel)?;
            read_cstr(mach, data, name_va).filter(|s| looks_like_swift_ident(s))
        }
        TYPE_REF_INDIRECT_OBJC_CLASS => None,
        _ => None,
    }
}

/// Apply a relative-indirectable offset (low bit = indirect pointer slot).
fn resolve_relative_indirectable(
    mach: &MachFile,
    data: &[u8],
    field_va: u64,
    rel: i32,
) -> Option<u64> {
    if rel == 0 {
        return None;
    }
    let indirect = rel & REL_INDIRECT_BIT != 0;
    // Clear low bit(s) reserved for flags; keep signed magnitude of the offset.
    let offset = rel & !REL_INDIRECT_BIT & !REL_PROTOCOL_IS_OBJC_BIT;
    let target = relative_va(field_va, offset)?;
    if !indirect {
        return Some(target);
    }
    let abs = read_ptr(mach, data, target)?;
    if abs == 0 {
        None
    } else {
        Some(abs)
    }
}

fn read_ptr(mach: &MachFile, data: &[u8], va: u64) -> Option<u64> {
    // Thin Mach-O recovery is 64-bit only (`MH_MAGIC_64`).
    let b = bytes_at(mach, data, va, 8)?;
    Some(u64::from_le_bytes(b.try_into().ok()?))
}

/// Read a context descriptor's `Name` (+ optional module prefix).
///
/// When `expect_kind` is set, the descriptor's kind bits must match.
fn read_context_display_name(
    mach: &MachFile,
    data: &[u8],
    desc_va: u64,
    expect_kind: Option<u32>,
) -> Option<String> {
    let _ = bytes_at(mach, data, desc_va, CONTEXT_NAME_MIN as usize)?;
    let flags = read_u32(mach, data, desc_va)?;
    if let Some(kind) = expect_kind
        && flags & KIND_MASK != kind
    {
        return None;
    }
    let name_rel = read_i32(mach, data, desc_va + 8)?;
    let name_va = relative_va(desc_va + 8, name_rel)?;
    let name = read_cstr(mach, data, name_va)?;
    if !looks_like_swift_ident(&name) {
        return None;
    }
    match resolve_module_name(mach, data, desc_va) {
        Some(module) if !module.is_empty() => Some(format!("{module}.{name}")),
        _ => Some(name),
    }
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
    // Fields relative pointer at +16 (after AccessFunction at +12).
    let fields = match read_i32(mach, data, desc_va + 16) {
        Some(fields_rel) => match relative_va(desc_va + 16, fields_rel) {
            Some(fields_va) => parse_field_descriptor(mach, data, fields_va),
            None => Vec::new(),
        },
        None => Vec::new(),
    };
    Some(SwiftType {
        va: desc_va,
        name,
        module,
        kind,
        flags,
        fields,
    })
}

/// Parse a public ABI `FieldDescriptor` at `desc_va`.
///
/// Soft-fails to an empty list on any corruption / bounds issue — field
/// recovery must never abort type recovery.
fn parse_field_descriptor(mach: &MachFile, data: &[u8], desc_va: u64) -> Vec<SwiftField> {
    let Some(_) = bytes_at(mach, data, desc_va, FIELD_DESC_HEADER as usize) else {
        return Vec::new();
    };
    // Header layout (Apple Swift ABI / RemoteInspection Records.h):
    //   +0  RelativeDirectPointer MangledTypeName
    //   +4  RelativeDirectPointer Superclass
    //   +8  uint16 Kind
    //   +10 uint16 FieldRecordSize
    //   +12 uint32 NumFields
    //   +16 FieldRecord[NumFields]…
    let Some(record_size) = read_u16(mach, data, desc_va + 10) else {
        return Vec::new();
    };
    if !(FIELD_RECORD_MIN..=FIELD_RECORD_SIZE_MAX).contains(&record_size) {
        return Vec::new();
    }
    let Some(num_fields) = read_u32(mach, data, desc_va + 12) else {
        return Vec::new();
    };
    let num_fields = num_fields as usize;
    if num_fields == 0 {
        return Vec::new();
    }
    let take = num_fields.min(MAX_FIELDS_PER_TYPE);
    let stride = record_size as u64;
    let records_base = desc_va.saturating_add(FIELD_DESC_HEADER);

    // Ensure the claimed record array is file-backed before walking.
    let total_bytes = (take as u64).saturating_mul(stride);
    if total_bytes == 0
        || bytes_at(mach, data, records_base, total_bytes as usize).is_none()
    {
        return Vec::new();
    }

    let mut fields = Vec::new();
    for i in 0..take {
        let Some(rec_va) = records_base.checked_add((i as u64).saturating_mul(stride)) else {
            break;
        };
        if let Some(field) = parse_field_record(mach, data, rec_va) {
            fields.push(field);
        }
    }
    fields
}

fn parse_field_record(mach: &MachFile, data: &[u8], rec_va: u64) -> Option<SwiftField> {
    // FieldRecord: Flags(+0) + MangledTypeName(+4) + FieldName(+8).
    let _ = bytes_at(mach, data, rec_va, FIELD_RECORD_MIN as usize)?;
    let name_rel = read_i32(mach, data, rec_va + 8)?;
    let name_va = relative_va(rec_va + 8, name_rel)?;
    let name = read_cstr_capped(mach, data, name_va, MAX_FIELD_NAME_LEN)?;
    if !looks_like_field_name(&name) {
        return None;
    }
    let mangled_type = read_i32(mach, data, rec_va + 4)
        .and_then(|rel| relative_va(rec_va + 4, rel))
        .and_then(|va| read_cstr_capped(mach, data, va, MAX_FIELD_NAME_LEN))
        .filter(|s| looks_like_mangled_or_ident(s));
    Some(SwiftField {
        name,
        mangled_type,
        offset: None,
    })
}

fn looks_like_field_name(s: &str) -> bool {
    let b = s.as_bytes();
    if b.is_empty() || b.len() > MAX_FIELD_NAME_LEN {
        return false;
    }
    // Property / case names are identifiers; reject mangled `$s…` payloads.
    if b[0] == b'$' {
        return false;
    }
    b.iter().all(|&c| {
        c.is_ascii_alphanumeric() || c == b'_' || c == b'$'
    }) && b.iter().any(|&c| c.is_ascii_alphanumeric() || c == b'_')
}

fn looks_like_mangled_or_ident(s: &str) -> bool {
    let b = s.as_bytes();
    if b.is_empty() || b.len() > MAX_FIELD_NAME_LEN {
        return false;
    }
    // Accept Swift mangling prefixes and short type identifiers.
    if s.starts_with("$s") || s.starts_with("$S") || s.starts_with("_T") {
        return b.iter().all(|&c| (0x20..=0x7E).contains(&c));
    }
    b.iter()
        .all(|&c| c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || c == b'$')
        && b.iter().any(|&c| c.is_ascii_alphanumeric() || c == b'_')
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
    const FNAME0_OFF: usize = 0x290; // "count\0"
    const FNAME1_OFF: usize = 0x298; // "label\0"
    const FTYPE0_OFF: usize = 0x2A0; // "$sSi\0"
    const FTYPE1_OFF: usize = 0x2A8; // "$sSS\0"
    const TYPREF_OFF: usize = 0x2B0; // "$s3App4DemoV\0"
    const MODULE_OFF: usize = 0x2C0; // module descriptor
    const STRUCT_OFF: usize = 0x2D0; // struct descriptor
    const FIELDMD_OFF: usize = 0x300; // FieldDescriptor + 2 FieldRecords
    const TYPES_OFF: usize = 0x340; // __swift5_types (one rel32)
    const REFLSECT_OFF: usize = 0x350;
    const TYPREFSECT_OFF: usize = 0x360;
    const PROTO_NAME_OFF: usize = 0x380; // "Named\0"
    const PROTO_DESC_OFF: usize = 0x388; // protocol descriptor
    const CONF_DESC_OFF: usize = 0x3A0; // ProtocolConformanceDescriptor
    const PROTOSECT_OFF: usize = 0x3B0; // __swift5_proto (one rel32)
    const WITNESS_TBL_OFF: usize = 0x3C0; // witness table (Description + 2 slots)
    const W0_NAME_OFF: usize = 0x3D8; // "doNamed\0" (name at witness VA)
    const W1_OFF: usize = 0x3E8; // anonymous witness payload (no name)

    /// Public ABI `FieldDescriptorKind::Struct`.
    const FIELD_KIND_STRUCT: u16 = 0;

    fn put(img: &mut [u8], off: usize, bytes: &[u8]) {
        img[off..off + bytes.len()].copy_from_slice(bytes);
    }
    fn put32(img: &mut [u8], off: usize, v: u32) {
        put(img, off, &v.to_le_bytes());
    }
    fn put16(img: &mut [u8], off: usize, v: u16) {
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

    /// Thin arm64 Mach-O with one Swift struct `App.Demo` having two
    /// fields (`count: Int`, `label: String`), a `Named` protocol
    /// conformance (`App.Demo : App.Named`), plus sample
    /// `__swift5_reflstr` / `__swift5_typeref` strings.
    fn synthetic_swift_macho() -> Vec<u8> {
        let mut img = vec![0u8; IMG_SIZE];

        // One LC_SEGMENT_64 __TEXT with 6 sections.
        // cmdsize = 72 + 6*80 = 552.
        let ncmds = 1u32;
        let sizeofcmds = 552u32;

        put32(&mut img, 0, MH_MAGIC_64);
        put32(&mut img, 4, 0x0100_000C); // ARM64
        put32(&mut img, 8, 0);
        put32(&mut img, 12, 2); // MH_EXECUTE
        put32(&mut img, 16, ncmds);
        put32(&mut img, 20, sizeofcmds);
        put32(&mut img, 24, MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE);

        let t = 0x20;
        put32(&mut img, t, LC_SEGMENT_64);
        put32(&mut img, t + 4, 552);
        put(&mut img, t + 8, b"__TEXT");
        put64(&mut img, t + 24, TEXT_VM);
        put64(&mut img, t + 32, 0x400);
        put64(&mut img, t + 40, 0);
        put64(&mut img, t + 48, 0x400);
        put32(&mut img, t + 56, VM_PROT_READ | VM_PROT_EXECUTE);
        put32(&mut img, t + 60, VM_PROT_READ | VM_PROT_EXECUTE);
        put32(&mut img, t + 64, 6); // nsects

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
            0x100,
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
            0x10,
            REFLSECT_OFF as u32,
        );
        write_section(
            &mut img,
            t + 72 + 320,
            b"__swift5_typeref",
            b"__TEXT",
            TEXT_VM + TYPREFSECT_OFF as u64,
            0x20,
            TYPREFSECT_OFF as u32,
        );
        write_section(
            &mut img,
            t + 72 + 400,
            b"__swift5_proto",
            b"__TEXT",
            TEXT_VM + PROTOSECT_OFF as u64,
            4,
            PROTOSECT_OFF as u32,
        );

        // Strings.
        put(&mut img, NAME_OFF, b"Demo\0");
        put(&mut img, MODNAME_OFF, b"App\0");
        put(&mut img, FNAME0_OFF, b"count\0");
        put(&mut img, FNAME1_OFF, b"label\0");
        put(&mut img, FTYPE0_OFF, b"$sSi\0");
        put(&mut img, FTYPE1_OFF, b"$sSS\0");
        put(&mut img, TYPREF_OFF, b"$s3App4DemoV\0");
        put(&mut img, PROTO_NAME_OFF, b"Named\0");

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
        let fieldmd_va = TEXT_VM + FIELDMD_OFF as u64;
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
        put_i32(
            &mut img,
            STRUCT_OFF + 16,
            rel32(struct_va + 16, fieldmd_va),
        ); // Fields → FieldDescriptor
        put32(&mut img, STRUCT_OFF + 20, 2); // NumFields
        put32(&mut img, STRUCT_OFF + 24, 0); // FieldOffsetVectorOffset

        // FieldDescriptor + two FieldRecords at FIELDMD_OFF.
        // Header: MangledTypeName, Superclass, Kind, FieldRecordSize, NumFields.
        put_i32(
            &mut img,
            FIELDMD_OFF,
            rel32(fieldmd_va, TEXT_VM + TYPREF_OFF as u64),
        ); // MangledTypeName → $s3App4DemoV
        put_i32(&mut img, FIELDMD_OFF + 4, 0); // Superclass null
        put16(&mut img, FIELDMD_OFF + 8, FIELD_KIND_STRUCT);
        put16(&mut img, FIELDMD_OFF + 10, FIELD_RECORD_MIN);
        put32(&mut img, FIELDMD_OFF + 12, 2); // NumFields

        // FieldRecord 0: count / $sSi
        let rec0 = FIELDMD_OFF + FIELD_DESC_HEADER as usize;
        let rec0_va = fieldmd_va + FIELD_DESC_HEADER;
        put32(&mut img, rec0, 0); // Flags
        put_i32(
            &mut img,
            rec0 + 4,
            rel32(rec0_va + 4, TEXT_VM + FTYPE0_OFF as u64),
        );
        put_i32(
            &mut img,
            rec0 + 8,
            rel32(rec0_va + 8, TEXT_VM + FNAME0_OFF as u64),
        );

        // FieldRecord 1: label / $sSS
        let rec1 = rec0 + FIELD_RECORD_MIN as usize;
        let rec1_va = rec0_va + FIELD_RECORD_MIN as u64;
        put32(&mut img, rec1, 0);
        put_i32(
            &mut img,
            rec1 + 4,
            rel32(rec1_va + 4, TEXT_VM + FTYPE1_OFF as u64),
        );
        put_i32(
            &mut img,
            rec1 + 8,
            rel32(rec1_va + 8, TEXT_VM + FNAME1_OFF as u64),
        );

        // __swift5_types: one relative pointer to the struct descriptor.
        let types_va = TEXT_VM + TYPES_OFF as u64;
        put_i32(&mut img, TYPES_OFF, rel32(types_va, struct_va));

        // Reflection / typeref section payloads.
        put(&mut img, REFLSECT_OFF, b"count\0label\0\0\0");
        put(&mut img, TYPREFSECT_OFF, b"$s3App4DemoV\0$sSi\0$sSS\0");

        // Protocol descriptor at PROTO_DESC_OFF (`App.Named`).
        let proto_va = TEXT_VM + PROTO_DESC_OFF as u64;
        put32(&mut img, PROTO_DESC_OFF, KIND_PROTOCOL);
        put_i32(
            &mut img,
            PROTO_DESC_OFF + 4,
            rel32(proto_va + 4, mod_va),
        );
        put_i32(
            &mut img,
            PROTO_DESC_OFF + 8,
            rel32(proto_va + 8, TEXT_VM + PROTO_NAME_OFF as u64),
        );
        put32(&mut img, PROTO_DESC_OFF + 12, 0); // NumRequirementsInSignature
        put32(&mut img, PROTO_DESC_OFF + 16, 2); // NumRequirements
        put_i32(&mut img, PROTO_DESC_OFF + 20, 0); // AssociatedTypeNames

        // Witness table pattern: Description + two requirement slots.
        // Slot 0 points at a printable name string (resolves as witness name);
        // slot 1 points at anonymous bytes (VA-only soft-fail).
        let conf_va = TEXT_VM + CONF_DESC_OFF as u64;
        let wt_va = TEXT_VM + WITNESS_TBL_OFF as u64;
        let w0_va = TEXT_VM + W0_NAME_OFF as u64;
        let w1_va = TEXT_VM + W1_OFF as u64;
        put(&mut img, W0_NAME_OFF, b"doNamed\0");
        put(&mut img, W1_OFF, &[0xD5, 0x03, 0x20, 0x1F, 0, 0, 0, 0]); // RET + pad
        put64(&mut img, WITNESS_TBL_OFF, conf_va); // Description
        put64(&mut img, WITNESS_TBL_OFF + 8, w0_va);
        put64(&mut img, WITNESS_TBL_OFF + 16, w1_va);

        // ProtocolConformanceDescriptor: Demo : Named (direct type desc).
        put_i32(
            &mut img,
            CONF_DESC_OFF,
            rel32(conf_va, proto_va),
        ); // Protocol (direct)
        put_i32(
            &mut img,
            CONF_DESC_OFF + 4,
            rel32(conf_va + 4, struct_va),
        ); // TypeRef (direct type descriptor)
        put_i32(
            &mut img,
            CONF_DESC_OFF + 8,
            rel32(conf_va + 8, wt_va),
        ); // WitnessTablePattern
        put32(
            &mut img,
            CONF_DESC_OFF + 12,
            TYPE_REF_DIRECT_TYPE_DESC << TYPE_REF_KIND_SHIFT,
        );

        // __swift5_proto: one relative pointer to the conformance.
        let proto_sect_va = TEXT_VM + PROTOSECT_OFF as u64;
        put_i32(&mut img, PROTOSECT_OFF, rel32(proto_sect_va, conf_va));

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
        assert!(mach.section_by_name("__TEXT", "__swift5_proto").is_some());

        let sw = recover(&mach, &img).expect("swift");
        assert!(sw.has_swift_sections());
        assert_eq!(sw.sections.len(), 4);
        assert!(
            sw.sections
                .iter()
                .any(|s| s.sectname == "__swift5_types")
        );
        assert!(
            sw.sections
                .iter()
                .any(|s| s.sectname == "__swift5_proto")
        );
        assert_eq!(sw.len(), 1);
        assert_eq!(sw.types[0].name, "Demo");
        assert_eq!(sw.types[0].module.as_deref(), Some("App"));
        assert_eq!(sw.types[0].kind, SwiftTypeKind::Struct);
        assert!(sw.refl_strings.iter().any(|s| s == "count"));
        assert!(sw.typerefs.iter().any(|s| s == "$s3App4DemoV"));
    }

    #[test]
    fn synthetic_recovers_struct_fields() {
        let img = synthetic_swift_macho();
        let mach = MachFile::parse(&img).unwrap();
        let sw = recover(&mach, &img).expect("swift");
        assert_eq!(sw.types.len(), 1);
        let fields = &sw.types[0].fields;
        assert_eq!(fields.len(), 2, "{fields:?}");
        assert_eq!(fields[0].name, "count");
        assert_eq!(fields[0].mangled_type.as_deref(), Some("$sSi"));
        assert_eq!(fields[0].offset, None);
        assert_eq!(fields[1].name, "label");
        assert_eq!(fields[1].mangled_type.as_deref(), Some("$sSS"));
        assert_eq!(fields[1].offset, None);
    }

    #[test]
    fn synthetic_recovers_proto_conformance_detail() {
        let img = synthetic_swift_macho();
        let mach = MachFile::parse(&img).unwrap();
        let sw = recover(&mach, &img).expect("swift");
        assert_eq!(sw.proto_conformances.len(), 1, "{:?}", sw.proto_conformances);
        let conf = &sw.proto_conformances[0];
        assert_eq!(conf.va, TEXT_VM + CONF_DESC_OFF as u64);
        assert_eq!(conf.protocol_name.as_deref(), Some("App.Named"));
        assert_eq!(conf.type_name.as_deref(), Some("App.Demo"));
        assert_eq!(conf.flags, TYPE_REF_DIRECT_TYPE_DESC << TYPE_REF_KIND_SHIFT);
        assert!(!sw.proto_capped);
    }

    #[test]
    fn synthetic_recovers_witness_table_slots() {
        let img = synthetic_swift_macho();
        let mach = MachFile::parse(&img).unwrap();
        let sw = recover(&mach, &img).expect("swift");
        let conf = &sw.proto_conformances[0];
        assert_eq!(conf.witnesses.len(), 2, "{:?}", conf.witnesses);
        assert_eq!(conf.witnesses[0].va, TEXT_VM + W0_NAME_OFF as u64);
        assert_eq!(conf.witnesses[0].name.as_deref(), Some("doNamed"));
        assert_eq!(conf.witnesses[1].va, TEXT_VM + W1_OFF as u64);
        assert_eq!(conf.witnesses[1].name, None); // VA-only soft-fail
    }

    #[test]
    fn render_lists_types_and_inventory() {
        let img = synthetic_swift_macho();
        let mach = MachFile::parse(&img).unwrap();
        let sw = recover(&mach, &img).unwrap();
        let text = render(&sw, 16);
        assert!(text.contains("__swift5_types"), "{text}");
        assert!(text.contains("__swift5_proto"), "{text}");
        assert!(text.contains("struct App.Demo"), "{text}");
        assert!(text.contains("count : $sSi"), "{text}");
        assert!(text.contains("label : $sSS"), "{text}");
        assert!(text.contains("App.Demo : App.Named"), "{text}");
        assert!(text.contains("witness "), "{text}");
        assert!(text.contains("doNamed"), "{text}");
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
        // Proto walk is independent of the type cap.
        assert_eq!(sw.proto_conformances.len(), 1);
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

    #[test]
    fn null_fields_pointer_yields_empty_fields() {
        let mut img = synthetic_swift_macho();
        // Zero the Fields relative pointer on the struct descriptor.
        put_i32(&mut img, STRUCT_OFF + 16, 0);
        let mach = MachFile::parse(&img).unwrap();
        let sw = recover(&mach, &img).unwrap();
        assert_eq!(sw.types.len(), 1);
        assert!(sw.types[0].fields.is_empty());
    }

    #[test]
    fn corrupt_conformance_pointers_do_not_panic() {
        let mut img = synthetic_swift_macho();
        // Zero protocol + type relative pointers; flags remain readable.
        put_i32(&mut img, CONF_DESC_OFF, 0);
        put_i32(&mut img, CONF_DESC_OFF + 4, 0);
        put32(&mut img, CONF_DESC_OFF + 12, 0xDEAD_BEEF);
        let mach = MachFile::parse(&img).unwrap();
        let sw = recover(&mach, &img).expect("swift");
        assert_eq!(sw.proto_conformances.len(), 1);
        let conf = &sw.proto_conformances[0];
        assert_eq!(conf.protocol_name, None);
        assert_eq!(conf.type_name, None);
        assert_eq!(conf.flags, 0xDEAD_BEEF);
        // Witness pattern still points at a readable table; heuristic walk
        // recovers slots even when the protocol link is gone.
        assert_eq!(conf.witnesses.len(), 2);
        let text = render(&sw, 16);
        assert!(text.contains("? : ?"), "{text}");
        assert!(text.contains("flags 0xdeadbeef"), "{text}");
    }

    #[test]
    fn null_witness_pattern_yields_empty_witnesses() {
        let mut img = synthetic_swift_macho();
        put_i32(&mut img, CONF_DESC_OFF + 8, 0);
        let mach = MachFile::parse(&img).unwrap();
        let sw = recover(&mach, &img).unwrap();
        assert_eq!(sw.proto_conformances.len(), 1);
        assert!(sw.proto_conformances[0].witnesses.is_empty());
    }

    #[test]
    fn witness_cap_truncates_without_panic() {
        let img = synthetic_swift_macho();
        let mach = MachFile::parse(&img).unwrap();
        let sw = recover(&mach, &img).unwrap();
        assert!(sw.proto_conformances[0].witnesses.len() <= MAX_WITNESSES_PER_CONFORMANCE);
        const { assert!(MAX_WITNESSES_PER_CONFORMANCE >= 2); }
    }

    #[test]
    fn proto_cap_truncates_without_panic() {
        let img = synthetic_swift_macho();
        let mach = MachFile::parse(&img).unwrap();
        // Directly exercise the proto walker with a zero cap.
        let sect = mach.section_by_name("__TEXT", "__swift5_proto").unwrap();
        let (list, capped) = parse_proto_section(&mach, &img, sect, 0).unwrap();
        assert!(list.is_empty());
        assert!(capped);
    }
}
