//! Objective-C metadata recovery from Mach-O `__objc_*` sections.
//!
//! Clean-room layout from Apple's published ObjC runtime headers
//! (`objc-runtime-new.h` in the APSL objc4 sources) and the Mach-O
//! section names the linker emits (`__objc_classlist`, `__objc_const`,
//! `__objc_methname`, …). This pass walks the on-disk class list, follows
//! each class's `class_ro_t` (masking the low flag bits of
//! `class_data_bits_t`), and reads base method lists — both the classic
//! absolute `method_t` (24-byte entries) and the relative / small form
//! (12-byte entries, WWDC 2020).
//!
//! # Contract
//!
//! - Never panics on any input: every read is bounds-checked.
//! - Missing `__objc_classlist` yields an empty [`ObjcImage`] (`Ok`), not
//!   an error — many Mach-O images simply have no ObjC.
//! - Caps bound hostile counts; truncation is reported on the image.
//! - Typed [`ObjcError`] covers hard failures (e.g. section bytes that
//!   claim to exist but overrun the file). Per-class / per-method
//!   corruption is skipped rather than aborting the whole recover.
//!
//! Swift metadata lives in [`crate::swift`], not here.

use std::fmt;
use std::fmt::Write as _;

use crate::macho::{MachFile, Section64};

/// Default cap on classes returned by [`recover`].
pub const DEFAULT_MAX_CLASSES: usize = 4096;

/// Absolute upper bound on classes walked from `__objc_classlist`.
pub const MAX_CLASSES: usize = 65_536;

/// Cap on methods retained per class (instance + class combined).
pub const MAX_METHODS_PER_CLASS: usize = 4096;

/// Cap on a recovered C-string name / selector / type encoding.
pub const MAX_CSTR_LEN: usize = 1024;

/// Cap on selector references recovered from `__objc_selrefs`.
pub const MAX_SELREFS: usize = 65_536;

/// Low bits of `class_data_bits_t` that encode flags, not the pointer
/// (`FAST_DATA_MASK` clears these — public objc4 layout).
const CLASS_DATA_FLAG_MASK: u64 = 0x7;

/// `method_list_t.entsizeAndFlags`: relative (small) method entries.
const METHOD_LIST_RELATIVE: u32 = 0x8000_0000;
/// Relative entries whose name offset points at the selector string
/// directly (not through an `__objc_selrefs` slot).
const METHOD_LIST_DIRECT_SEL: u32 = 0x4000_0000;
/// Bits of `entsizeAndFlags` that carry the per-entry stride.
const METHOD_LIST_ENTSIZE_MASK: u32 = 0x0000_FFFC;

/// Size of an on-disk 64-bit `objc_class` before trailing padding:
/// `isa` + `superclass` + `cache_t` (16) + `bits`.
const CLASS_T_SIZE: u64 = 0x28;
/// Offset of `class_data_bits_t bits` inside `objc_class`.
const CLASS_T_BITS_OFF: u64 = 0x20;
/// Offset of `isa` (metaclass pointer) inside `objc_class`.
const CLASS_T_ISA_OFF: u64 = 0x00;

/// LP64 `class_ro_t`: `name` and `baseMethods` field offsets.
const CLASS_RO_NAME_OFF: u64 = 0x18;
const CLASS_RO_METHODS_OFF: u64 = 0x20;
const CLASS_RO_MIN_SIZE: u64 = 0x28;

/// `class_rw_t.ro` when the data pointer is a realized rw (offset 8).
const CLASS_RW_RO_OFF: u64 = 0x08;

/// Why ObjC recovery refused to produce a usable image view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjcError {
    /// The section header claims bytes that overrun the file buffer.
    SectionTruncated {
        segname: String,
        sectname: String,
        offset: u32,
        size: u64,
        file_len: usize,
    },
    /// A required virtual address is not file-backed.
    Unmapped(u64),
    /// A count or size field exceeded a hard safety cap.
    CapExceeded {
        what: &'static str,
        value: usize,
        cap: usize,
    },
}

impl fmt::Display for ObjcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjcError::SectionTruncated {
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
            ObjcError::Unmapped(va) => {
                write!(f, "ObjC pointer {va:#x} is not file-backed")
            }
            ObjcError::CapExceeded { what, value, cap } => {
                write!(f, "ObjC {what} count {value} exceeds cap {cap}")
            }
        }
    }
}

impl std::error::Error for ObjcError {}

pub type ObjcResult<T> = std::result::Result<T, ObjcError>;

/// One recovered ObjC method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjcMethod {
    /// Selector name (`SEL`), e.g. `doThing` or `initWithFrame:`.
    pub name: String,
    /// Type encoding string when present (`@encode` form), else empty.
    pub types: String,
    /// Implementation VA (`IMP`), when the list entry resolved.
    pub imp: Option<u64>,
    /// Whether this came from the metaclass (a `+` method).
    pub is_class: bool,
}

/// One `__objc_selrefs` slot: the pointer's own VA and the selector string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjcSelRef {
    /// Virtual address of the selref pointer itself (xref target site).
    pub va: u64,
    /// Selector C-string when the pointee resolved.
    pub name: String,
}

/// One recovered ObjC class from `__objc_classlist`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjcClass {
    /// Virtual address of the `objc_class` structure.
    pub va: u64,
    /// Class name from `class_ro_t.name`.
    pub name: String,
    /// Virtual address of the metaclass (`isa`), when mapped.
    pub metaclass_va: Option<u64>,
    /// Base methods (instance and class), capped per class.
    pub methods: Vec<ObjcMethod>,
    /// True when method recovery hit [`MAX_METHODS_PER_CLASS`].
    pub methods_capped: bool,
}

/// Recovered ObjC metadata for one thin Mach-O image.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ObjcImage {
    pub classes: Vec<ObjcClass>,
    /// True when `__objc_classlist` held more pointers than the class cap.
    pub classes_capped: bool,
    /// File VA of the class list section that was walked, when any.
    pub classlist_va: Option<u64>,
    /// Selector references from `__objc_selrefs` (capped).
    pub selrefs: Vec<ObjcSelRef>,
    /// True when selref recovery hit [`MAX_SELREFS`].
    pub selrefs_capped: bool,
    /// File VA of `__objc_selrefs` when present.
    pub selrefs_va: Option<u64>,
}

impl ObjcImage {
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty() && self.selrefs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.classes.len()
    }
}

/// Recover ObjC classes/methods with the default class cap.
pub fn recover(mach: &MachFile, data: &[u8]) -> ObjcResult<ObjcImage> {
    recover_capped(mach, data, DEFAULT_MAX_CLASSES)
}

/// Recover ObjC classes/methods, keeping at most `max_classes` classes.
pub fn recover_capped(mach: &MachFile, data: &[u8], max_classes: usize) -> ObjcResult<ObjcImage> {
    let max_classes = max_classes.min(MAX_CLASSES);
    let (selrefs, selrefs_capped, selrefs_va) = recover_selrefs(mach, data)?;

    let Some(sect) = find_section(mach, "__objc_classlist") else {
        return Ok(ObjcImage {
            selrefs,
            selrefs_capped,
            selrefs_va,
            ..ObjcImage::default()
        });
    };
    let classlist = section_bytes(data, sect)?;
    let classlist_va = Some(sect.addr);

    let n_ptrs = classlist.len() / 8;
    if n_ptrs > MAX_CLASSES {
        return Err(ObjcError::CapExceeded {
            what: "classlist pointers",
            value: n_ptrs,
            cap: MAX_CLASSES,
        });
    }

    let mut classes = Vec::new();
    let mut classes_capped = false;
    for chunk in classlist.chunks_exact(8) {
        if classes.len() >= max_classes {
            classes_capped = true;
            break;
        }
        let mut word = [0u8; 8];
        word.copy_from_slice(chunk);
        let class_va = u64::from_le_bytes(word);
        if class_va == 0 {
            continue;
        }
        // Skip duplicates from a hostile/repeated list.
        if classes.iter().any(|c: &ObjcClass| c.va == class_va) {
            continue;
        }
        if let Some(cls) = parse_class(mach, data, class_va) {
            classes.push(cls);
        }
    }
    if !classes_capped && n_ptrs > max_classes {
        classes_capped = true;
    }

    Ok(ObjcImage {
        classes,
        classes_capped,
        classlist_va,
        selrefs,
        selrefs_capped,
        selrefs_va,
    })
}

/// Walk `__objc_selrefs`: each 8-byte slot is a pointer to a selector
/// C-string in `__objc_methname` (or equivalent). Total; caps truncate.
fn recover_selrefs(
    mach: &MachFile,
    data: &[u8],
) -> ObjcResult<(Vec<ObjcSelRef>, bool, Option<u64>)> {
    let Some(sect) = find_section(mach, "__objc_selrefs") else {
        return Ok((Vec::new(), false, None));
    };
    let bytes = section_bytes(data, sect)?;
    let n_ptrs = bytes.len() / 8;
    if n_ptrs > MAX_SELREFS {
        return Err(ObjcError::CapExceeded {
            what: "selrefs pointers",
            value: n_ptrs,
            cap: MAX_SELREFS,
        });
    }
    let mut out = Vec::new();
    let mut capped = false;
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        if out.len() >= MAX_SELREFS {
            capped = true;
            break;
        }
        let mut word = [0u8; 8];
        word.copy_from_slice(chunk);
        let sel_va = u64::from_le_bytes(word);
        if sel_va == 0 {
            continue;
        }
        let Some(name) = read_cstr(mach, data, sel_va) else {
            continue;
        };
        let slot_va = sect.addr.saturating_add((i as u64).saturating_mul(8));
        out.push(ObjcSelRef { va: slot_va, name });
    }
    if !capped && n_ptrs > out.len() && out.len() >= MAX_SELREFS {
        capped = true;
    }
    Ok((out, capped, Some(sect.addr)))
}

/// Render a human-readable listing for `redump --objc`.
pub fn render(image: &ObjcImage, max_classes: usize) -> String {
    let mut out = String::new();
    if image.classes.is_empty() && image.selrefs.is_empty() {
        out.push_str("  (no ObjC classes recovered");
        if image.classlist_va.is_none() {
            out.push_str("; no __objc_classlist section");
        }
        out.push_str(")\n");
        return out;
    }

    if !image.classes.is_empty() {
        let shown = image.classes.len().min(max_classes);
        let _ = writeln!(
            out,
            "  {} class(es) recovered{}",
            image.classes.len(),
            if image.classes_capped {
                " (class list capped)"
            } else {
                ""
            }
        );
        for cls in image.classes.iter().take(shown) {
            let _ = writeln!(out, "  @interface {}  // va {:#x}", cls.name, cls.va);
            for m in &cls.methods {
                let mark = if m.is_class { '+' } else { '-' };
                let types = if m.types.is_empty() {
                    String::new()
                } else {
                    format!("  // {}", m.types)
                };
                let imp = match m.imp {
                    Some(va) => format!("  imp {:#x}", va),
                    None => String::new(),
                };
                let _ = writeln!(out, "    {mark}[{} {}]{imp}{types}", cls.name, m.name);
            }
            if cls.methods_capped {
                let _ = writeln!(out, "    // methods capped at {MAX_METHODS_PER_CLASS}");
            }
            let _ = writeln!(out, "  @end");
        }
        if shown < image.classes.len() {
            let _ = writeln!(
                out,
                "  … {} more class(es) omitted (cap {max_classes})",
                image.classes.len() - shown
            );
        }
    }

    if !image.selrefs.is_empty() || image.selrefs_va.is_some() {
        let _ = writeln!(
            out,
            "  {} selref(s){}",
            image.selrefs.len(),
            if image.selrefs_capped {
                " (selrefs capped)"
            } else {
                ""
            }
        );
        let sel_cap = max_classes.saturating_mul(4).clamp(16, 256);
        for s in image.selrefs.iter().take(sel_cap) {
            let _ = writeln!(out, "    @{:#x}  @selector({})", s.va, s.name);
        }
        if image.selrefs.len() > sel_cap {
            let _ = writeln!(
                out,
                "    … {} more selref(s) omitted",
                image.selrefs.len() - sel_cap
            );
        }
    }
    out
}

fn find_section<'a>(mach: &'a MachFile, sectname: &str) -> Option<&'a Section64> {
    mach.segments
        .iter()
        .flat_map(|s| s.sections.iter())
        .find(|s| s.sectname == sectname)
}

fn section_bytes<'a>(data: &'a [u8], sect: &Section64) -> ObjcResult<&'a [u8]> {
    let start = sect.offset as usize;
    let size = sect.size as usize;
    let end = start.checked_add(size).ok_or_else(|| ObjcError::SectionTruncated {
        segname: sect.segname.clone(),
        sectname: sect.sectname.clone(),
        offset: sect.offset,
        size: sect.size,
        file_len: data.len(),
    })?;
    if end > data.len() {
        return Err(ObjcError::SectionTruncated {
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

fn read_u64(mach: &MachFile, data: &[u8], va: u64) -> Option<u64> {
    let b = bytes_at(mach, data, va, 8)?;
    Some(u64::from_le_bytes(b.try_into().ok()?))
}

fn read_i32(mach: &MachFile, data: &[u8], va: u64) -> Option<i32> {
    read_u32(mach, data, va).map(|u| u as i32)
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
    // Require printable-ish ObjC identifiers / encodings: reject NULs and
    // mostly-control blobs that a hostile pointer might land on.
    let bytes = &window[..len];
    if !bytes
        .iter()
        .all(|&b| (0x20..=0x7E).contains(&b) || b == b'\t')
    {
        return None;
    }
    Some(String::from_utf8_lossy(bytes).into_owned())
}

fn parse_class(mach: &MachFile, data: &[u8], class_va: u64) -> Option<ObjcClass> {
    let _ = bytes_at(mach, data, class_va, CLASS_T_SIZE as usize)?;
    let bits = read_u64(mach, data, class_va + CLASS_T_BITS_OFF)?;
    let data_ptr = bits & !CLASS_DATA_FLAG_MASK;
    if data_ptr == 0 {
        return None;
    }
    let ro_va = resolve_class_ro(mach, data, data_ptr)?;
    let name = read_cstr(mach, data, read_u64(mach, data, ro_va + CLASS_RO_NAME_OFF)?)?;
    let methods_va = read_u64(mach, data, ro_va + CLASS_RO_METHODS_OFF).unwrap_or(0);

    let mut methods = Vec::new();
    let mut methods_capped = false;
    if methods_va != 0 {
        let (ms, capped) = parse_method_list(mach, data, methods_va, false);
        methods.extend(ms);
        methods_capped |= capped;
    }

    let isa = read_u64(mach, data, class_va + CLASS_T_ISA_OFF).unwrap_or(0);
    let metaclass_va = if isa != 0 && isa != class_va && mach.vaddr_to_offset(isa).is_ok() {
        Some(isa)
    } else {
        None
    };

    if let Some(meta_va) = metaclass_va
        && let Some(meta_bits) = read_u64(mach, data, meta_va + CLASS_T_BITS_OFF)
    {
        let meta_data = meta_bits & !CLASS_DATA_FLAG_MASK;
        if let Some(meta_ro) = resolve_class_ro(mach, data, meta_data) {
            let meta_methods = read_u64(mach, data, meta_ro + CLASS_RO_METHODS_OFF).unwrap_or(0);
            if meta_methods != 0 && methods.len() < MAX_METHODS_PER_CLASS {
                let (ms, capped) = parse_method_list(mach, data, meta_methods, true);
                for m in ms {
                    if methods.len() >= MAX_METHODS_PER_CLASS {
                        methods_capped = true;
                        break;
                    }
                    methods.push(m);
                }
                methods_capped |= capped;
            }
        }
    }

    Some(ObjcClass {
        va: class_va,
        name,
        metaclass_va,
        methods,
        methods_capped,
    })
}

/// Resolve a data pointer to a `class_ro_t` VA.
///
/// On disk the pointer is usually the ro itself. When the name slot does
/// not look like a C string, try the `class_rw_t` layout (`ro` at +8).
fn resolve_class_ro(mach: &MachFile, data: &[u8], data_ptr: u64) -> Option<u64> {
    if data_ptr == 0 {
        return None;
    }
    let _ = bytes_at(mach, data, data_ptr, CLASS_RO_MIN_SIZE as usize)?;
    let name_ptr = read_u64(mach, data, data_ptr + CLASS_RO_NAME_OFF)?;
    if read_cstr(mach, data, name_ptr).is_some() {
        return Some(data_ptr);
    }
    // Try as class_rw_t → ro.
    let ro = read_u64(mach, data, data_ptr + CLASS_RW_RO_OFF)?;
    if ro == 0 {
        return None;
    }
    let _ = bytes_at(mach, data, ro, CLASS_RO_MIN_SIZE as usize)?;
    let name_ptr = read_u64(mach, data, ro + CLASS_RO_NAME_OFF)?;
    if read_cstr(mach, data, name_ptr).is_some() {
        Some(ro)
    } else {
        None
    }
}

fn parse_method_list(
    mach: &MachFile,
    data: &[u8],
    list_va: u64,
    is_class: bool,
) -> (Vec<ObjcMethod>, bool) {
    let Some(entsize_and_flags) = read_u32(mach, data, list_va) else {
        return (Vec::new(), false);
    };
    let Some(count) = read_u32(mach, data, list_va + 4) else {
        return (Vec::new(), false);
    };
    let entsize = entsize_and_flags & METHOD_LIST_ENTSIZE_MASK;
    if entsize == 0 || count == 0 {
        return (Vec::new(), false);
    }
    let relative = entsize_and_flags & METHOD_LIST_RELATIVE != 0;
    let direct_sel = entsize_and_flags & METHOD_LIST_DIRECT_SEL != 0;

    let count = (count as usize).min(MAX_METHODS_PER_CLASS.saturating_add(1));
    let mut methods = Vec::new();
    let mut capped = false;
    let entries_va = list_va + 8;

    for i in 0..count {
        if methods.len() >= MAX_METHODS_PER_CLASS {
            capped = true;
            break;
        }
        let entry_va = match (entsize as u64).checked_mul(i as u64).and_then(|o| entries_va.checked_add(o))
        {
            Some(va) => va,
            None => break,
        };
        let meth = if relative {
            if entsize < 12 {
                break;
            }
            parse_relative_method(mach, data, entry_va, direct_sel, is_class)
        } else {
            if entsize < 24 {
                break;
            }
            parse_absolute_method(mach, data, entry_va, is_class)
        };
        if let Some(m) = meth {
            methods.push(m);
        }
    }
    if count > MAX_METHODS_PER_CLASS {
        capped = true;
    }
    (methods, capped)
}

fn parse_absolute_method(
    mach: &MachFile,
    data: &[u8],
    entry_va: u64,
    is_class: bool,
) -> Option<ObjcMethod> {
    let name_ptr = read_u64(mach, data, entry_va)?;
    let types_ptr = read_u64(mach, data, entry_va + 8).unwrap_or(0);
    let imp = read_u64(mach, data, entry_va + 16).filter(|&v| v != 0);
    let name = read_cstr(mach, data, name_ptr)?;
    let types = read_cstr(mach, data, types_ptr).unwrap_or_default();
    Some(ObjcMethod {
        name,
        types,
        imp,
        is_class,
    })
}

fn parse_relative_method(
    mach: &MachFile,
    data: &[u8],
    entry_va: u64,
    direct_sel: bool,
    is_class: bool,
) -> Option<ObjcMethod> {
    let name_rel = read_i32(mach, data, entry_va)?;
    let types_rel = read_i32(mach, data, entry_va + 4)?;
    let imp_rel = read_i32(mach, data, entry_va + 8)?;

    let name_target = relative_va(entry_va, name_rel)?;
    let name = if direct_sel {
        read_cstr(mach, data, name_target)?
    } else {
        // Name field points at a selector reference (pointer to the string).
        let sel_ptr = read_u64(mach, data, name_target)?;
        read_cstr(mach, data, sel_ptr)?
    };
    let types = relative_va(entry_va + 4, types_rel)
        .and_then(|va| read_cstr(mach, data, va))
        .unwrap_or_default();
    let imp = relative_va(entry_va + 8, imp_rel).filter(|&v| v != 0);

    Some(ObjcMethod {
        name,
        types,
        imp,
        is_class,
    })
}

fn relative_va(field_va: u64, rel: i32) -> Option<u64> {
    if rel >= 0 {
        field_va.checked_add(rel as u64)
    } else {
        field_va.checked_sub((-rel) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macho::{
        LC_SEGMENT_64, MH_DYLDLINK, MH_MAGIC_64, MH_NOUNDEFS, MH_PIE, MH_TWOLEVEL, VM_PROT_EXECUTE,
        VM_PROT_READ, VM_PROT_WRITE,
    };

    const IMG_SIZE: usize = 0x800;
    const TEXT_VM: u64 = 0x1_0000_0000;
    const DATA_VM: u64 = TEXT_VM + 0x400;
    const DATA_FILE: usize = 0x400;

    // Content lives after the load-command blob (ends at 0x330) so section
    // headers are not stomped by string / method-list bytes.
    const CLASSNAME_OFF: usize = 0x340;
    const METHNAME_OFF: usize = 0x350;
    const METHTYPE_OFF: usize = 0x370;
    const METHLIST_OFF: usize = 0x390;
    const CLASS_RO_OFF: usize = 0x480; // in __DATA file range
    const META_RO_OFF: usize = 0x4B0;

    const CLASSLIST_OFF: usize = DATA_FILE; // 0x400
    const CLASS_OFF: usize = DATA_FILE + 0x10; // 0x410
    const META_OFF: usize = DATA_FILE + 0x40; // 0x440
    const SELREF_OFF: usize = DATA_FILE + 0x70; // 0x470

    fn put(img: &mut [u8], off: usize, bytes: &[u8]) {
        img[off..off + bytes.len()].copy_from_slice(bytes);
    }
    fn put32(img: &mut [u8], off: usize, v: u32) {
        put(img, off, &v.to_le_bytes());
    }
    fn put64(img: &mut [u8], off: usize, v: u64) {
        put(img, off, &v.to_le_bytes());
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
        put32(img, at + 52, 3); // align 2^3
    }

    /// Thin arm64 Mach-O with one ObjC class `Demo` and two absolute methods.
    fn synthetic_objc_macho(relative_methods: bool) -> Vec<u8> {
        let mut img = vec![0u8; IMG_SIZE];

        // Two LC_SEGMENT_64: __TEXT (5 sects) + __DATA (3 sects).
        // cmdsize TEXT = 72 + 5*80 = 472; DATA = 72 + 3*80 = 312; total 784.
        let ncmds = 2u32;
        let sizeofcmds = 784u32;

        put32(&mut img, 0, MH_MAGIC_64);
        put32(&mut img, 4, 0x0100_000C); // ARM64
        put32(&mut img, 8, 0);
        put32(&mut img, 12, 2); // MH_EXECUTE
        put32(&mut img, 16, ncmds);
        put32(&mut img, 20, sizeofcmds);
        put32(&mut img, 24, MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE);

        // __TEXT segment.
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
            TEXT_VM + 0x280,
            0x20,
            0x280,
        );
        write_section(
            &mut img,
            t + 72 + 80,
            b"__objc_classname",
            b"__TEXT",
            TEXT_VM + CLASSNAME_OFF as u64,
            0x10,
            CLASSNAME_OFF as u32,
        );
        write_section(
            &mut img,
            t + 72 + 160,
            b"__objc_methname",
            b"__TEXT",
            TEXT_VM + METHNAME_OFF as u64,
            0x20,
            METHNAME_OFF as u32,
        );
        write_section(
            &mut img,
            t + 72 + 240,
            b"__objc_methtype",
            b"__TEXT",
            TEXT_VM + METHTYPE_OFF as u64,
            0x20,
            METHTYPE_OFF as u32,
        );
        write_section(
            &mut img,
            t + 72 + 320,
            b"__objc_const",
            b"__TEXT",
            TEXT_VM + METHLIST_OFF as u64,
            0x70,
            METHLIST_OFF as u32,
        );

        // __DATA segment.
        let d = t + 472;
        put32(&mut img, d, LC_SEGMENT_64);
        put32(&mut img, d + 4, 312);
        put(&mut img, d + 8, b"__DATA");
        put64(&mut img, d + 24, DATA_VM);
        put64(&mut img, d + 32, 0x400);
        put64(&mut img, d + 40, DATA_FILE as u64);
        put64(&mut img, d + 48, (IMG_SIZE - DATA_FILE) as u64);
        put32(&mut img, d + 56, VM_PROT_READ | VM_PROT_WRITE);
        put32(&mut img, d + 60, VM_PROT_READ | VM_PROT_WRITE);
        put32(&mut img, d + 64, 3);

        write_section(
            &mut img,
            d + 72,
            b"__objc_classlist",
            b"__DATA",
            DATA_VM,
            8,
            CLASSLIST_OFF as u32,
        );
        write_section(
            &mut img,
            d + 72 + 80,
            b"__objc_data",
            b"__DATA",
            DATA_VM + 0x10,
            0x60,
            CLASS_OFF as u32,
        );
        write_section(
            &mut img,
            d + 72 + 160,
            b"__objc_selrefs",
            b"__DATA",
            DATA_VM + 0x70,
            8,
            SELREF_OFF as u32,
        );

        // Strings.
        put(&mut img, CLASSNAME_OFF, b"Demo\0");
        put(&mut img, METHNAME_OFF, b"doThing\0count\0");
        put(&mut img, METHTYPE_OFF, b"v16@0:8\0q16@0:8\0");

        let name_demo = TEXT_VM + CLASSNAME_OFF as u64;
        let sel_dothing = TEXT_VM + METHNAME_OFF as u64;
        let sel_count = TEXT_VM + METHNAME_OFF as u64 + 8; // "count"
        let ty_v = TEXT_VM + METHTYPE_OFF as u64;
        let ty_q = TEXT_VM + METHTYPE_OFF as u64 + 8;
        let imp0 = TEXT_VM + 0x280;
        let imp1 = TEXT_VM + 0x288;

        let methlist_va = TEXT_VM + METHLIST_OFF as u64;
        if relative_methods {
            // entsize 12 | RELATIVE | DIRECT_SEL, count 2.
            put32(
                &mut img,
                METHLIST_OFF,
                12 | METHOD_LIST_RELATIVE | METHOD_LIST_DIRECT_SEL,
            );
            put32(&mut img, METHLIST_OFF + 4, 2);
            // Entry 0 at METHLIST_OFF+8.
            let e0 = METHLIST_OFF + 8;
            let e0_va = methlist_va + 8;
            let name_rel = (sel_dothing as i64 - e0_va as i64) as i32;
            let types_rel = (ty_v as i64 - (e0_va + 4) as i64) as i32;
            let imp_rel = (imp0 as i64 - (e0_va + 8) as i64) as i32;
            put32(&mut img, e0, name_rel as u32);
            put32(&mut img, e0 + 4, types_rel as u32);
            put32(&mut img, e0 + 8, imp_rel as u32);
            // Entry 1.
            let e1 = e0 + 12;
            let e1_va = e0_va + 12;
            let name_rel = (sel_count as i64 - e1_va as i64) as i32;
            let types_rel = (ty_q as i64 - (e1_va + 4) as i64) as i32;
            let imp_rel = (imp1 as i64 - (e1_va + 8) as i64) as i32;
            put32(&mut img, e1, name_rel as u32);
            put32(&mut img, e1 + 4, types_rel as u32);
            put32(&mut img, e1 + 8, imp_rel as u32);
        } else {
            put32(&mut img, METHLIST_OFF, 24); // absolute entsize
            put32(&mut img, METHLIST_OFF + 4, 2);
            let e0 = METHLIST_OFF + 8;
            put64(&mut img, e0, sel_dothing);
            put64(&mut img, e0 + 8, ty_v);
            put64(&mut img, e0 + 16, imp0);
            let e1 = e0 + 24;
            put64(&mut img, e1, sel_count);
            put64(&mut img, e1 + 8, ty_q);
            put64(&mut img, e1 + 16, imp1);
        }

        // class_ro for Demo (instance methods).
        put32(&mut img, CLASS_RO_OFF, 0); // flags
        put32(&mut img, CLASS_RO_OFF + 4, 0); // instanceStart
        put32(&mut img, CLASS_RO_OFF + 8, 8); // instanceSize
        put32(&mut img, CLASS_RO_OFF + 12, 0); // reserved
        put64(&mut img, CLASS_RO_OFF + 16, 0); // ivarLayout
        put64(&mut img, CLASS_RO_OFF + 24, name_demo);
        put64(&mut img, CLASS_RO_OFF + 32, methlist_va);
        // rest zeroed

        // meta class_ro (no methods) — name still "Demo".
        put32(&mut img, META_RO_OFF, 1); // RO_META-ish flag bit, ignored
        put32(&mut img, META_RO_OFF + 8, 40);
        put64(&mut img, META_RO_OFF + 24, name_demo);

        let class_va = DATA_VM + 0x10;
        let meta_va = DATA_VM + 0x40;
        let class_ro_va = DATA_VM + (CLASS_RO_OFF - DATA_FILE) as u64;
        let meta_ro_va = DATA_VM + (META_RO_OFF - DATA_FILE) as u64;

        // classlist
        put64(&mut img, CLASSLIST_OFF, class_va);

        // class_t
        put64(&mut img, CLASS_OFF, meta_va); // isa
        put64(&mut img, CLASS_OFF + 8, 0); // superclass
        // cache left 0
        put64(&mut img, CLASS_OFF + 0x20, class_ro_va); // bits → ro

        // metaclass_t
        put64(&mut img, META_OFF, meta_va); // isa → self
        put64(&mut img, META_OFF + 8, 0);
        put64(&mut img, META_OFF + 0x20, meta_ro_va);

        // __objc_selrefs: one slot pointing at "doThing".
        put64(&mut img, SELREF_OFF, sel_dothing);

        img
    }

    #[test]
    fn recovers_absolute_methods_from_synthetic_classlist() {
        let img = synthetic_objc_macho(false);
        let mach = MachFile::parse(&img).expect("mach-o");
        assert!(mach.section_by_name("__DATA", "__objc_classlist").is_some());

        let objc = recover(&mach, &img).expect("objc");
        assert_eq!(objc.len(), 1);
        assert_eq!(objc.classes[0].name, "Demo");
        assert_eq!(objc.classes[0].methods.len(), 2);
        assert_eq!(objc.classes[0].methods[0].name, "doThing");
        assert_eq!(objc.classes[0].methods[0].types, "v16@0:8");
        assert!(!objc.classes[0].methods[0].is_class);
        assert_eq!(objc.classes[0].methods[1].name, "count");
        assert_eq!(
            objc.classes[0].methods[0].imp,
            Some(TEXT_VM + 0x280)
        );
    }

    #[test]
    fn recovers_relative_direct_sel_methods() {
        let img = synthetic_objc_macho(true);
        let mach = MachFile::parse(&img).unwrap();
        let objc = recover(&mach, &img).unwrap();
        assert_eq!(objc.classes[0].name, "Demo");
        assert_eq!(objc.classes[0].methods.len(), 2);
        assert_eq!(objc.classes[0].methods[0].name, "doThing");
        assert_eq!(objc.classes[0].methods[1].name, "count");
    }

    #[test]
    fn missing_classlist_is_empty_ok() {
        // Reuse the library's non-ObjC synthetic by building a bare header
        // image: parse the absolute fixture then clear the section name so
        // find_section misses — easier: recover against macho without the
        // section via a tiny TEXT-only image.
        let mut img = synthetic_objc_macho(false);
        // Rename the classlist section so it is not found.
        let d = 0x20 + 472;
        put(&mut img, d + 72, b"__not_classlist\0");
        // Also hide selrefs so the image is truly empty of ObjC facts.
        put(&mut img, d + 72 + 160, b"__not_selrefs\0\0");
        let mach = MachFile::parse(&img).unwrap();
        let objc = recover(&mach, &img).unwrap();
        assert!(objc.is_empty());
        assert!(objc.classlist_va.is_none());
        let text = render(&objc, 16);
        assert!(text.contains("no __objc_classlist"), "{text}");
    }

    #[test]
    fn render_lists_classes_and_methods() {
        let img = synthetic_objc_macho(false);
        let mach = MachFile::parse(&img).unwrap();
        let objc = recover(&mach, &img).unwrap();
        let text = render(&objc, 16);
        assert!(text.contains("@interface Demo"), "{text}");
        assert!(text.contains("-[Demo doThing]"), "{text}");
        assert!(text.contains("-[Demo count]"), "{text}");
        assert!(text.contains("@end"), "{text}");
        assert!(text.contains("@selector(doThing)"), "{text}");
    }

    #[test]
    fn recovers_selrefs() {
        let img = synthetic_objc_macho(false);
        let mach = MachFile::parse(&img).unwrap();
        let objc = recover(&mach, &img).unwrap();
        assert_eq!(objc.selrefs.len(), 1);
        assert_eq!(objc.selrefs[0].name, "doThing");
        assert_eq!(objc.selrefs[0].va, DATA_VM + 0x70);
    }

    #[test]
    fn class_cap_truncates_without_panic() {
        let img = synthetic_objc_macho(false);
        let mach = MachFile::parse(&img).unwrap();
        let objc = recover_capped(&mach, &img, 0).unwrap();
        assert!(objc.classes.is_empty());
        assert!(objc.classes_capped);
        // Selrefs still recover with a zero class cap.
        assert!(!objc.selrefs.is_empty());
    }

    #[test]
    fn section_overrun_is_typed_error() {
        let mut img = synthetic_objc_macho(false);
        // Lie in the section header before parse: claim a classlist that
        // runs past EOF. Segment extents are not re-checked against the
        // section size, so Mach-O parse still succeeds.
        let d = 0x20 + 472;
        put64(&mut img, d + 72 + 40, 0x10_000); // __objc_classlist.size
        let mach = MachFile::parse(&img).unwrap();
        let err = recover(&mach, &img).unwrap_err();
        assert!(
            matches!(err, ObjcError::SectionTruncated { .. }),
            "{err}"
        );
    }
}
