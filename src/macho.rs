//! Mach-O image parser.
//!
//! Clean-room implementation from Apple's published Mach-O layouts: the
//! `mach-o/loader.h`, `mach-o/fat.h`, and `mach-o/nlist.h` headers
//! released under the APSL, and the "OS X ABI Mach-O File Format
//! Reference". This pass handles 64-bit little-endian thin images
//! (`MH_MAGIC_64`, the on-disk byte order for x86-64 and arm64): the
//! `mach_header_64`, the load-command walk (`LC_SEGMENT_64` with nested
//! `section_64` entries, `LC_SYMTAB` with names resolved from the string
//! table, `LC_MAIN`, `LC_LOAD_DYLIB`), fat/universal containers
//! (`FAT_MAGIC` / `FAT_MAGIC_64`) with bounds-validated slice selection,
//! and virtual-address to file-offset mapping through segments.
//!
//! # The fat-header byte-order trap
//!
//! A thin Mach-O header is stored in the byte order of its target CPU
//! (little-endian for every current Apple target), but the fat header and
//! its `fat_arch` table are *always big-endian*, regardless of host or
//! target. Reading a perfectly valid fat file with a little-endian reader
//! therefore yields [`FAT_CIGAM`] (`0xBEBAFECA`) — that value is not a
//! distinct file kind, it is the reader holding the header upside down.
//! [`FatFile::parse`] reads the header big-endian as the format requires;
//! a file whose fat header is genuinely little-endian on disk (big-endian
//! read yields `FAT_CIGAM`) violates the format and is rejected as
//! unsupported rather than guessed at. Symmetrically, [`MachFile::parse`]
//! recognizes a little-endian read of a fat container and redirects the
//! caller to [`FatFile`] instead of reporting a generic bad signature.
//!
//! Deliberately out of scope for now (rejected or ignored, never
//! mis-parsed): 32-bit images (`MH_MAGIC`), big-endian thin images
//! (`MH_CIGAM_64`, historical PowerPC), classic dyld bind/rebase
//! opcodes (non-chained), and relocation records. **Chained fixups**
//! (`LC_DYLD_CHAINED_FIXUPS`) parse the public `mach-o/fixup-chains.h`
//! layouts: header, import names, `dyld_chained_starts_in_image` /
//! per-segment page starts, and a chain walk for common 64-bit pointer
//! formats (see [`ChainedFixups`]). **Encryption info** and **code
//! signature** load commands are detected — see [`EncryptionInfo`].

use crate::error::{ParseError, Result};
use crate::reader::Reader;

/// 64-bit Mach-O magic, as read little-endian from an x86-64/arm64 image.
pub const MH_MAGIC_64: u32 = 0xFEED_FACF;
/// Byte-swapped [`MH_MAGIC_64`]: the file is a big-endian 64-bit Mach-O.
pub const MH_CIGAM_64: u32 = 0xCFFA_EDFE;
/// 32-bit Mach-O magic (unsupported in this pass).
pub const MH_MAGIC: u32 = 0xFEED_FACE;
/// Byte-swapped [`MH_MAGIC`]: a big-endian 32-bit Mach-O (unsupported).
pub const MH_CIGAM: u32 = 0xCEFA_EDFE;

/// Fat/universal container magic. The fat header is big-endian on disk,
/// so this is the value of the first four bytes *read big-endian*.
pub const FAT_MAGIC: u32 = 0xCAFE_BABE;
/// Fat container with 64-bit `fat_arch_64` entries, read big-endian.
pub const FAT_MAGIC_64: u32 = 0xCAFE_BABF;
/// Byte-swapped [`FAT_MAGIC`]. Seeing this from a big-endian read means
/// the header was written little-endian (invalid); seeing it from a
/// little-endian read means the file is a normal fat container that the
/// reader is holding upside down. See the module docs.
pub const FAT_CIGAM: u32 = 0xBEBA_FECA;
/// Byte-swapped [`FAT_MAGIC_64`].
pub const FAT_CIGAM_64: u32 = 0xBFBA_FECA;

/// Size of a `mach_header_64`.
const MACH_HEADER_64_SIZE: usize = 32;
/// Size of the common `load_command` prefix (`cmd` + `cmdsize`).
const LOAD_COMMAND_MIN_SIZE: u64 = 8;
/// Size of a `segment_command_64` before its `section_64` entries.
const SEGMENT_COMMAND_64_SIZE: u64 = 72;
/// Size of one `section_64` entry.
const SECTION_64_SIZE: u64 = 80;
/// Size of a `symtab_command`.
const SYMTAB_COMMAND_SIZE: usize = 24;
/// Size of one `nlist_64` symbol table entry.
const NLIST_64_SIZE: u64 = 16;
/// Size of an `entry_point_command` (`LC_MAIN`).
const ENTRY_POINT_COMMAND_SIZE: usize = 24;
/// Minimum size of a `dylib_command` (before its name string).
const DYLIB_COMMAND_SIZE: usize = 24;
/// Size of a `fat_header`.
const FAT_HEADER_SIZE: u64 = 8;
/// Size of one 32-bit `fat_arch` entry.
const FAT_ARCH_SIZE: u64 = 20;
/// Size of one `fat_arch_64` entry.
const FAT_ARCH_64_SIZE: u64 = 32;

/// Cap on the number of architectures a fat header may declare. Real
/// universal binaries carry a handful of slices; a count above this is
/// treated as hostile rather than allocated for.
pub const MAX_FAT_ARCHES: u32 = 64;

/// `LC_SYMTAB`: symbol table location.
pub const LC_SYMTAB: u32 = 0x02;
/// `LC_LOAD_DYLIB`: a dynamically linked shared library dependency.
pub const LC_LOAD_DYLIB: u32 = 0x0C;
/// `LC_SEGMENT_64`: a 64-bit segment mapping.
pub const LC_SEGMENT_64: u32 = 0x19;
/// `LC_MAIN` (`0x28 | LC_REQ_DYLD`): entry point as a file offset.
pub const LC_MAIN: u32 = 0x8000_0028;
/// `LC_FUNCTION_STARTS`: a `linkedit_data_command` pointing at a
/// ULEB128 delta-encoded table of function-start offsets.
pub const LC_FUNCTION_STARTS: u32 = 0x26;
/// `LC_CODE_SIGNATURE` (`linkedit_data_command`).
pub const LC_CODE_SIGNATURE: u32 = 0x1D;
/// `LC_ENCRYPTION_INFO_64`.
pub const LC_ENCRYPTION_INFO_64: u32 = 0x2C;
/// `LC_DYLD_CHAINED_FIXUPS` (`0x34 | LC_REQ_DYLD`).
pub const LC_DYLD_CHAINED_FIXUPS: u32 = 0x8000_0034;

/// Cap on chained-fixup import table entries.
pub const MAX_CHAINED_IMPORTS: u32 = 65_536;
/// Cap on `dyld_chained_starts_in_image.seg_count`.
pub const MAX_CHAINED_SEGS: u32 = 256;
/// Cap on `dyld_chained_starts_in_segment.page_count` per segment.
pub const MAX_CHAINED_PAGES: u32 = 65_536;
/// Cap on total pointer-chain steps across the image (hostile loops).
pub const MAX_CHAIN_STEPS: usize = 1_048_576;
/// Cap on bind slots emitted from the chain walk.
pub const MAX_CHAINED_BINDS: usize = 65_536;

/// `DYLD_CHAINED_PTR_ARM64E` — stride 8.
pub const DYLD_CHAINED_PTR_ARM64E: u16 = 1;
/// `DYLD_CHAINED_PTR_64` — stride 4, target is vmaddr.
pub const DYLD_CHAINED_PTR_64: u16 = 2;
/// `DYLD_CHAINED_PTR_64_OFFSET` — stride 4, target is vm offset.
pub const DYLD_CHAINED_PTR_64_OFFSET: u16 = 6;
/// `DYLD_CHAINED_PTR_ARM64E_KERNEL` — stride 4.
pub const DYLD_CHAINED_PTR_ARM64E_KERNEL: u16 = 7;
/// `DYLD_CHAINED_PTR_ARM64E_USERLAND` — stride 8.
pub const DYLD_CHAINED_PTR_ARM64E_USERLAND: u16 = 9;
/// `DYLD_CHAINED_PTR_ARM64E_FIRMWARE` — stride 4.
pub const DYLD_CHAINED_PTR_ARM64E_FIRMWARE: u16 = 10;
/// `DYLD_CHAINED_PTR_ARM64E_USERLAND24` — stride 8, 24-bit bind ordinal.
pub const DYLD_CHAINED_PTR_ARM64E_USERLAND24: u16 = 12;

/// `DYLD_CHAINED_PTR_START_NONE` in `page_start[]`.
const DYLD_CHAINED_PTR_START_NONE: u16 = 0xFFFF;
/// `DYLD_CHAINED_PTR_START_MULTI` flag in `page_start[]` (32-bit multi-start).
const DYLD_CHAINED_PTR_START_MULTI: u16 = 0x8000;

/// `DYLD_CHAINED_IMPORT` / `_ADDEND` / `_ADDEND64`.
const DYLD_CHAINED_IMPORT: u32 = 1;
const DYLD_CHAINED_IMPORT_ADDEND: u32 = 2;
const DYLD_CHAINED_IMPORT_ADDEND64: u32 = 3;

/// Header flag: the object has no undefined references (`MH_NOUNDEFS`).
pub const MH_NOUNDEFS: u32 = 0x1;
/// Header flag: input for the dynamic linker (`MH_DYLDLINK`).
pub const MH_DYLDLINK: u32 = 0x4;
/// Header flag: uses two-level name space bindings (`MH_TWOLEVEL`).
pub const MH_TWOLEVEL: u32 = 0x80;
/// Header flag: position-independent executable (`MH_PIE`).
pub const MH_PIE: u32 = 0x20_0000;

/// Segment permission flag: readable (`VM_PROT_READ`).
pub const VM_PROT_READ: u32 = 0x1;
/// Segment permission flag: writable (`VM_PROT_WRITE`).
pub const VM_PROT_WRITE: u32 = 0x2;
/// Segment permission flag: executable (`VM_PROT_EXECUTE`).
pub const VM_PROT_EXECUTE: u32 = 0x4;

/// `n_type` mask for debug (stab) entries.
pub const N_STAB: u8 = 0xE0;
/// `n_type` mask for the symbol type bits.
pub const N_TYPE: u8 = 0x0E;
/// `n_type` flag: external (visible to other images).
pub const N_EXT: u8 = 0x01;

/// Target architecture (`cputype`). The 64-bit values carry the
/// `CPU_ARCH_ABI64` (`0x0100_0000`) bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuType {
    /// `CPU_TYPE_X86_64` (`0x0100_0007`).
    X86_64,
    /// `CPU_TYPE_ARM64` (`0x0100_000C`); arm64e differs only in subtype.
    Arm64,
    /// A cputype we recognize the encoding of but do not model.
    Other(i32),
}

impl From<i32> for CpuType {
    fn from(raw: i32) -> Self {
        match raw {
            0x0100_0007 => CpuType::X86_64,
            0x0100_000C => CpuType::Arm64,
            other => CpuType::Other(other),
        }
    }
}

/// File type (`filetype` in the header).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// Relocatable object file (`MH_OBJECT`, 1).
    Object,
    /// Demand-paged executable (`MH_EXECUTE`, 2).
    Execute,
    /// Core dump (`MH_CORE`, 4).
    Core,
    /// Dynamically bound shared library (`MH_DYLIB`, 6).
    Dylib,
    /// Dynamic link editor, i.e. dyld itself (`MH_DYLINKER`, 7).
    Dylinker,
    /// Dynamically bound bundle (`MH_BUNDLE`, 8).
    Bundle,
    /// Debug-symbol companion file (`MH_DSYM`, 10).
    Dsym,
    /// A file type we recognize the encoding of but do not model.
    Other(u32),
}

impl From<u32> for FileType {
    fn from(raw: u32) -> Self {
        match raw {
            1 => FileType::Object,
            2 => FileType::Execute,
            4 => FileType::Core,
            6 => FileType::Dylib,
            7 => FileType::Dylinker,
            8 => FileType::Bundle,
            10 => FileType::Dsym,
            other => FileType::Other(other),
        }
    }
}

/// The Mach-O file header (`mach_header_64`), minus the magic we validate
/// and discard and the trailing reserved word.
#[derive(Debug, Clone)]
pub struct MachHeader {
    pub cputype: CpuType,
    /// CPU subtype (`cpusubtype`), e.g. `CPU_SUBTYPE_ARM64E` (2); the top
    /// byte carries capability bits.
    pub cpusubtype: i32,
    pub filetype: FileType,
    /// Number of load commands following the header.
    pub ncmds: u32,
    /// Total size in bytes of the load-command region.
    pub sizeofcmds: u32,
    /// `MH_*` flag bits.
    pub flags: u32,
}

/// A `section_64` entry nested inside an `LC_SEGMENT_64` command.
#[derive(Debug, Clone)]
pub struct Section64 {
    /// Section name (e.g. `__text`), NUL-trimmed from its 16-byte field.
    pub sectname: String,
    /// Name of the segment this section belongs to (e.g. `__TEXT`).
    pub segname: String,
    /// Virtual address of the section's first byte in memory.
    pub addr: u64,
    /// Size of the section in bytes.
    pub size: u64,
    /// File offset of the section contents.
    pub offset: u32,
    /// Alignment as a power of two.
    pub align: u32,
    /// Section type and attribute flags (`S_*` / `S_ATTR_*`).
    pub flags: u32,
}

/// An `LC_SEGMENT_64` load command with its nested sections.
#[derive(Debug, Clone)]
pub struct Segment64 {
    /// Segment name (e.g. `__TEXT`), NUL-trimmed from its 16-byte field.
    pub segname: String,
    /// Virtual address of the segment's first byte in memory.
    pub vmaddr: u64,
    /// Number of bytes the segment occupies in memory.
    pub vmsize: u64,
    /// File offset of the segment's first byte.
    pub fileoff: u64,
    /// Number of bytes of the segment present in the file.
    pub filesize: u64,
    /// Maximum permitted protection (`VM_PROT_*` bits).
    pub maxprot: u32,
    /// Initial protection (`VM_PROT_*` bits).
    pub initprot: u32,
    /// `SG_*` segment flags.
    pub flags: u32,
    pub sections: Vec<Section64>,
}

impl Segment64 {
    pub fn is_read(&self) -> bool {
        self.initprot & VM_PROT_READ != 0
    }

    pub fn is_write(&self) -> bool {
        self.initprot & VM_PROT_WRITE != 0
    }

    pub fn is_execute(&self) -> bool {
        self.initprot & VM_PROT_EXECUTE != 0
    }

    /// Whether `vaddr` falls inside this segment's *file-backed* range.
    ///
    /// Bytes in `[filesize, vmsize)` are zero-fill with no file backing
    /// (and `__PAGEZERO` has `filesize` 0 entirely), so only the first
    /// `filesize` bytes are considered mappable.
    pub fn contains_vaddr(&self, vaddr: u64) -> bool {
        vaddr >= self.vmaddr && vaddr - self.vmaddr < self.filesize
    }
}

/// Symbol type, from the `N_TYPE` bits of `n_type` (or [`SymbolKind::Debug`]
/// when any `N_STAB` bit is set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// `N_UNDF` (0x0): undefined, expected from another image.
    Undefined,
    /// `N_ABS` (0x2): absolute, not relative to any section.
    Absolute,
    /// `N_SECT` (0xE): defined in the section numbered by `sect`.
    Defined,
    /// `N_PBUD` (0xC): prebound undefined.
    Prebound,
    /// `N_INDR` (0xA): indirect, defined as an alias of another symbol.
    Indirect,
    /// Any `N_STAB` debugging entry.
    Debug,
    /// A type value we recognize the encoding of but do not model.
    Other(u8),
}

impl SymbolKind {
    fn from_n_type(n_type: u8) -> SymbolKind {
        if n_type & N_STAB != 0 {
            return SymbolKind::Debug;
        }
        match n_type & N_TYPE {
            0x0 => SymbolKind::Undefined,
            0x2 => SymbolKind::Absolute,
            0xE => SymbolKind::Defined,
            0xC => SymbolKind::Prebound,
            0xA => SymbolKind::Indirect,
            other => SymbolKind::Other(other),
        }
    }
}

/// A symbol table entry (`nlist_64`) with its name resolved through the
/// string table named by the `LC_SYMTAB` command.
#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    /// Symbol value (`n_value`); for defined symbols, a virtual address.
    pub value: u64,
    pub kind: SymbolKind,
    /// Whether the `N_EXT` bit is set (visible to other images).
    pub external: bool,
    /// 1-based section ordinal for [`SymbolKind::Defined`] symbols,
    /// counting sections across all segments in load-command order;
    /// 0 (`NO_SECT`) otherwise.
    pub sect: u8,
    /// Type- and linker-specific description bits (`n_desc`).
    pub desc: u16,
}

/// An imported library from an `LC_LOAD_DYLIB` command.
#[derive(Debug, Clone)]
pub struct Dylib {
    /// Install name, e.g. `/usr/lib/libSystem.B.dylib`.
    pub name: String,
    /// Build timestamp (conventionally 2 for prebuilt libraries).
    pub timestamp: u32,
    /// Version encoded as `xxxx.yy.zz` in 16.8.8 bits.
    pub current_version: u32,
    /// Compatibility version, same encoding.
    pub compatibility_version: u32,
}

/// Parsed `LC_ENCRYPTION_INFO_64` — ciphertext must not be silently decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptionInfo {
    pub cryptoff: u32,
    pub cryptsize: u32,
    pub cryptid: u32,
}

/// One bind slot discovered by walking chained-fixup pointer chains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainedBindSlot {
    /// Virtual address of the pointer slot in the preferred address space.
    pub slot_va: u64,
    /// Import table ordinal resolved against [`ChainedFixups::import_names`].
    pub ordinal: u32,
    /// Imported symbol name (empty if the ordinal was out of range).
    pub name: String,
}

/// Chained-fixups header, import names, and bind slots from a chain walk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainedFixups {
    pub fixups_version: u32,
    pub imports_count: u32,
    pub imports_format: u32,
    pub symbols_format: u32,
    /// Import symbol names in table order (empty string if unresolved).
    pub import_names: Vec<String>,
    /// Bind entries found by walking `dyld_chained_starts_*` pointer chains.
    pub bind_slots: Vec<ChainedBindSlot>,
    pub dataoff: u32,
    pub datasize: u32,
}

/// A parsed 64-bit little-endian thin Mach-O image.
#[derive(Debug, Clone)]
pub struct MachFile {
    pub header: MachHeader,
    /// `LC_SEGMENT_64` segments in load-command order.
    pub segments: Vec<Segment64>,
    /// Symbols from the `LC_SYMTAB` command.
    pub symbols: Vec<Symbol>,
    /// Libraries imported via `LC_LOAD_DYLIB`, in load-command order.
    pub dylibs: Vec<Dylib>,
    /// Absolute function-start VAs decoded from `LC_FUNCTION_STARTS`,
    /// sorted and deduplicated; empty when the command is absent.
    pub function_starts: Vec<u64>,
    /// Present when `LC_ENCRYPTION_INFO_64` was seen (`cryptid != 0`
    /// means the range is encrypted on disk).
    pub encryption: Option<EncryptionInfo>,
    /// Code-signature blob location, if `LC_CODE_SIGNATURE` present.
    pub code_signature: Option<(u32, u32)>,
    /// Chained fixups summary, if `LC_DYLD_CHAINED_FIXUPS` present.
    pub chained_fixups: Option<ChainedFixups>,
    entry_offset: Option<u64>,
}

impl MachFile {
    /// Parse a thin Mach-O image from raw file bytes.
    ///
    /// Only `MH_MAGIC_64` little-endian images are accepted. 32-bit and
    /// big-endian images are rejected with [`ParseError::Unsupported`]
    /// rather than mis-parsed, and a fat container (recognized by its
    /// magic appearing byte-swapped to a little-endian read; see the
    /// module docs) is redirected to [`FatFile::parse`].
    pub fn parse(data: &[u8]) -> Result<MachFile> {
        let header = parse_mach_header(data)?;

        check_range(
            MACH_HEADER_64_SIZE as u64,
            header.sizeofcmds as u64,
            data.len(),
            "load command region",
        )?;
        // Every load command is at least 8 bytes, so ncmds is bounded by
        // sizeofcmds; a crafted ncmds can never drive a long parse loop.
        if header.ncmds as u64 * LOAD_COMMAND_MIN_SIZE > header.sizeofcmds as u64 {
            return Err(ParseError::Unsupported(format!(
                "ncmds is {} but sizeofcmds ({:#x}) has room for at most {} commands",
                header.ncmds,
                header.sizeofcmds,
                header.sizeofcmds as u64 / LOAD_COMMAND_MIN_SIZE
            )));
        }
        let cmds_end = MACH_HEADER_64_SIZE as u64 + header.sizeofcmds as u64;

        let mut segments = Vec::new();
        let mut symbols = Vec::new();
        let mut dylibs = Vec::new();
        let mut entry_offset = None;
        // (dataoff, datasize) of LC_FUNCTION_STARTS, decoded after the
        // walk once the segments (and thus the __TEXT base) are known.
        let mut function_starts_data = None;
        let mut encryption = None;
        let mut code_signature = None;
        let mut chained_linkedit = None;

        let mut off = MACH_HEADER_64_SIZE as u64;
        let mut r = Reader::new(data);
        for i in 0..header.ncmds {
            if cmds_end - off < LOAD_COMMAND_MIN_SIZE {
                return Err(ParseError::Unsupported(format!(
                    "load command {i} begins past the end of the command region"
                )));
            }
            r.seek(off as usize)?;
            let cmd = r.u32()?;
            let cmdsize = r.u32()? as u64;
            if cmdsize < LOAD_COMMAND_MIN_SIZE {
                return Err(ParseError::Unsupported(format!(
                    "load command {i} has cmdsize {cmdsize}, smaller than the 8-byte load_command header"
                )));
            }
            let next = off.checked_add(cmdsize).ok_or_else(|| {
                ParseError::Unsupported(format!(
                    "load command {i} cmdsize {cmdsize:#x} overflows a 64-bit offset"
                ))
            })?;
            if next > cmds_end {
                return Err(ParseError::Unsupported(format!(
                    "load command {i} (cmdsize {cmdsize:#x}) extends past sizeofcmds"
                )));
            }
            // In-bounds by the checks above: off < next <= cmds_end <= len.
            let body = &data[off as usize..next as usize];
            match cmd {
                LC_SEGMENT_64 => segments.push(parse_segment_64(body)?),
                LC_SYMTAB => symbols.extend(parse_symtab(data, body)?),
                LC_MAIN => {
                    if entry_offset.is_none() {
                        entry_offset = Some(parse_entry_point(body)?);
                    }
                }
                LC_LOAD_DYLIB => dylibs.push(parse_dylib(body)?),
                LC_FUNCTION_STARTS if function_starts_data.is_none() => {
                    function_starts_data = Some(parse_linkedit_data(body)?);
                }
                LC_ENCRYPTION_INFO_64 if encryption.is_none() => {
                    encryption = Some(parse_encryption_info_64(body)?);
                }
                LC_CODE_SIGNATURE if code_signature.is_none() => {
                    code_signature = Some(parse_linkedit_data(body)?);
                }
                LC_DYLD_CHAINED_FIXUPS if chained_linkedit.is_none() => {
                    chained_linkedit = Some(parse_linkedit_data(body)?);
                }
                // Unknown or unmodeled commands are skipped, never an error:
                // new LC_* values appear with every OS release.
                _ => {}
            }
            off = next;
        }

        // The delta table is relative to the image's first mapped
        // segment (the one covering the header, conventionally __TEXT):
        // its `vmaddr` is the base the first delta is added to.
        let base_vmaddr = segments
            .iter()
            .find(|s| s.fileoff == 0 && s.filesize != 0)
            .or_else(|| segments.first())
            .map(|s| s.vmaddr)
            .unwrap_or(0);
        let function_starts = match function_starts_data {
            Some((dataoff, datasize)) => {
                decode_function_starts(data, dataoff, datasize, base_vmaddr)
            }
            None => Vec::new(),
        };

        let chained_fixups = match chained_linkedit {
            Some((dataoff, datasize)) => {
                Some(parse_chained_fixups(data, dataoff, datasize, &segments)?)
            }
            None => None,
        };

        Ok(MachFile {
            header,
            segments,
            symbols,
            dylibs,
            function_starts,
            encryption,
            code_signature,
            chained_fixups,
            entry_offset,
        })
    }

    /// True when the image declares an encrypted range with nonzero cryptid.
    pub fn is_encrypted(&self) -> bool {
        self.encryption
            .as_ref()
            .is_some_and(|e| e.cryptid != 0)
    }

    /// Map a virtual address to a file offset via the segments.
    ///
    /// This is the Mach-O analogue of [`crate::pe::PeFile::rva_to_offset`]
    /// and [`crate::elf::ElfFile::vaddr_to_offset`], and it shares the
    /// [`ParseError::UnmappedVaddr`] variant with the latter. Only
    /// file-backed bytes are mappable: `__PAGEZERO` (`filesize` 0) and a
    /// segment's zero-fill tail have no file offset.
    pub fn vaddr_to_offset(&self, vaddr: u64) -> Result<usize> {
        let seg = self
            .segments
            .iter()
            .find(|s| s.contains_vaddr(vaddr))
            .ok_or(ParseError::UnmappedVaddr(vaddr))?;
        seg.fileoff
            .checked_add(vaddr - seg.vmaddr)
            .map(|off| off as usize)
            .ok_or(ParseError::UnmappedVaddr(vaddr))
    }

    /// The segment containing `vaddr` in its file-backed range, if any.
    pub fn segment_for_vaddr(&self, vaddr: u64) -> Option<&Segment64> {
        self.segments.iter().find(|s| s.contains_vaddr(vaddr))
    }

    /// The first segment with the given name (e.g. `__TEXT`), if any.
    pub fn segment_by_name(&self, name: &str) -> Option<&Segment64> {
        self.segments.iter().find(|s| s.segname == name)
    }

    /// The first section with the given segment and section name
    /// (e.g. `__TEXT`, `__text`), if any.
    pub fn section_by_name(&self, segname: &str, sectname: &str) -> Option<&Section64> {
        self.segments
            .iter()
            .flat_map(|s| s.sections.iter())
            .find(|s| s.segname == segname && s.sectname == sectname)
    }

    /// The first symbol with the given name, if any.
    pub fn symbol_by_name(&self, name: &str) -> Option<&Symbol> {
        self.symbols.iter().find(|s| s.name == name)
    }

    /// Entry point from `LC_MAIN` (`entryoff`), if present.
    ///
    /// `loader.h` documents `entryoff` as the `__TEXT`-relative file
    /// offset of `main()`; since `__TEXT` conventionally begins at file
    /// offset 0, it is in practice the file offset of the entry point.
    /// `None` for images without `LC_MAIN` (dylibs, or executables old
    /// enough to use `LC_UNIXTHREAD` instead).
    pub fn entry_offset(&self) -> Option<u64> {
        self.entry_offset
    }
}

/// One architecture slice of a fat/universal container.
#[derive(Debug, Clone)]
pub struct FatArch {
    pub cputype: CpuType,
    pub cpusubtype: i32,
    /// File offset of the slice's first byte (widened from u32 for
    /// `FAT_MAGIC` containers).
    pub offset: u64,
    /// Size of the slice in bytes.
    pub size: u64,
    /// Alignment of the slice as a power of two.
    pub align: u32,
}

impl FatArch {
    /// The bytes of this slice within the fat container.
    ///
    /// Bounds are re-checked against `data` so a [`FatArch`] can never
    /// index out of a buffer, even one it was not parsed from.
    pub fn slice<'a>(&self, data: &'a [u8]) -> Result<&'a [u8]> {
        check_range(self.offset, self.size, data.len(), "fat arch slice")?;
        Ok(&data[self.offset as usize..(self.offset + self.size) as usize])
    }

    /// Parse this slice as a thin Mach-O image.
    pub fn parse(&self, data: &[u8]) -> Result<MachFile> {
        MachFile::parse(self.slice(data)?)
    }
}

/// A parsed fat/universal container header: the architecture table only,
/// with every slice's bounds validated against the file.
#[derive(Debug, Clone)]
pub struct FatFile {
    arches: Vec<FatArch>,
}

impl FatFile {
    /// Parse a fat/universal container header from raw file bytes.
    ///
    /// The fat header is big-endian on disk regardless of host or target
    /// byte order, and is read accordingly (see the module docs for the
    /// `FAT_CIGAM` trap this avoids). Both `FAT_MAGIC` (32-bit `fat_arch`
    /// entries) and `FAT_MAGIC_64` (`fat_arch_64`) are accepted; every
    /// slice's `[offset, offset + size)` must lie within the file.
    pub fn parse(data: &[u8]) -> Result<FatFile> {
        let mut r = Reader::new(data);
        let magic = be_u32(&mut r)?;
        let is64 = match magic {
            FAT_MAGIC => false,
            FAT_MAGIC_64 => true,
            // A big-endian read seeing the byte-swapped magic means the
            // header was written little-endian, which the format forbids.
            // (A little-endian *read* of a valid fat file sees the same
            // value — the classic confusion this arm's message untangles.)
            FAT_CIGAM | FAT_CIGAM_64 => {
                return Err(ParseError::Unsupported(
                    "fat header is little-endian on disk (FAT_CIGAM); the fat header is defined \
                     to be big-endian regardless of architecture"
                        .into(),
                ));
            }
            other => {
                return Err(ParseError::BadSignature {
                    what: "fat (universal)",
                    offset: 0,
                    found: other,
                    expected: FAT_MAGIC,
                });
            }
        };

        let nfat_arch = be_u32(&mut r)?;
        if nfat_arch > MAX_FAT_ARCHES {
            return Err(ParseError::Unsupported(format!(
                "fat header declares {nfat_arch} architectures, above the cap of {MAX_FAT_ARCHES}"
            )));
        }
        let entsize = if is64 { FAT_ARCH_64_SIZE } else { FAT_ARCH_SIZE };
        check_table(FAT_HEADER_SIZE, nfat_arch as u64, entsize, data.len(), "fat arch")?;

        let mut arches = Vec::with_capacity(nfat_arch as usize);
        for _ in 0..nfat_arch {
            let cputype = CpuType::from(be_u32(&mut r)? as i32);
            let cpusubtype = be_u32(&mut r)? as i32;
            let (offset, size) = if is64 {
                (be_u64(&mut r)?, be_u64(&mut r)?)
            } else {
                (be_u32(&mut r)? as u64, be_u32(&mut r)? as u64)
            };
            let align = be_u32(&mut r)?;
            if is64 {
                r.skip(4)?; // fat_arch_64 reserved word
            }
            check_range(offset, size, data.len(), "fat arch slice")?;
            arches.push(FatArch {
                cputype,
                cpusubtype,
                offset,
                size,
                align,
            });
        }
        Ok(FatFile { arches })
    }

    /// The architecture slices declared by the container.
    pub fn arches(&self) -> &[FatArch] {
        &self.arches
    }

    /// The first slice with the given cputype, if any.
    pub fn arch_by_cputype(&self, cputype: CpuType) -> Option<&FatArch> {
        self.arches.iter().find(|a| a.cputype == cputype)
    }
}

/// Read a big-endian u32; the fat header is big-endian on disk (module docs).
fn be_u32(r: &mut Reader) -> Result<u32> {
    Ok(u32::from_be_bytes(r.take(4)?.try_into().unwrap()))
}

/// Read a big-endian u64 (used by `fat_arch_64` entries).
fn be_u64(r: &mut Reader) -> Result<u64> {
    Ok(u64::from_be_bytes(r.take(8)?.try_into().unwrap()))
}

/// Verify that `[offset, offset + size)` lies within the file, with
/// overflow-safe arithmetic.
fn check_range(offset: u64, size: u64, file_len: usize, what: &str) -> Result<()> {
    let end = offset.checked_add(size).ok_or_else(|| {
        ParseError::Unsupported(format!(
            "{what} range {offset:#x}+{size:#x} overflows a 64-bit offset"
        ))
    })?;
    if end > file_len as u64 {
        return Err(ParseError::UnexpectedEof {
            offset: offset.min(file_len as u64) as usize,
            needed: size as usize,
            available: (file_len as u64).saturating_sub(offset) as usize,
        });
    }
    Ok(())
}

/// Verify that a table of `count` entries of `entsize` bytes starting at
/// `offset` lies within the file, so a crafted count (nsyms, nfat_arch)
/// can never drive an oversized allocation or out-of-bounds read.
fn check_table(offset: u64, count: u64, entsize: u64, file_len: usize, what: &str) -> Result<()> {
    let total = count.checked_mul(entsize).ok_or_else(|| {
        ParseError::Unsupported(format!(
            "{what} table size ({count} entries of {entsize} bytes) overflows"
        ))
    })?;
    check_range(offset, total, file_len, what)
}

/// Read the NUL-terminated string at `name_off` inside a string table
/// whose bounds the caller has already validated against the file.
fn strtab_string(
    data: &[u8],
    strtab_off: usize,
    strtab_size: usize,
    name_off: usize,
    what: &str,
) -> Result<String> {
    if name_off == 0 {
        // n_strx 0 conventionally names the empty string.
        return Ok(String::new());
    }
    if name_off >= strtab_size {
        return Err(ParseError::Unsupported(format!(
            "{what} name offset {name_off:#x} is past the end of its string table ({strtab_size:#x} bytes)"
        )));
    }
    let table = &data[strtab_off..strtab_off + strtab_size];
    let bytes = &table[name_off..];
    let len = bytes.iter().position(|&b| b == 0).ok_or_else(|| {
        ParseError::Unsupported(format!(
            "{what} name at string-table offset {name_off:#x} is not NUL-terminated within the table"
        ))
    })?;
    // Non-UTF-8 names are replaced rather than rejected: strings inside
    // binaries are untrusted input and must never abort a parse.
    Ok(String::from_utf8_lossy(&bytes[..len]).into_owned())
}

/// Trim a fixed 16-byte name field (segname/sectname) at its first NUL.
fn fixed_name(bytes: &[u8]) -> String {
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..len]).into_owned()
}

fn parse_mach_header(data: &[u8]) -> Result<MachHeader> {
    let mut r = Reader::new(data);
    let magic = r.u32()?;
    match magic {
        MH_MAGIC_64 => {}
        MH_MAGIC | MH_CIGAM => {
            return Err(ParseError::Unsupported(
                "32-bit Mach-O (MH_MAGIC); only 64-bit images are supported in this pass".into(),
            ));
        }
        MH_CIGAM_64 => {
            return Err(ParseError::Unsupported(
                "big-endian Mach-O (magic reads byte-swapped as MH_CIGAM_64); only little-endian \
                 targets (x86-64, arm64) are supported in this pass"
                    .into(),
            ));
        }
        // A little-endian read of a fat container's big-endian magic
        // yields the byte-swapped FAT values — see the module docs.
        FAT_CIGAM | FAT_CIGAM_64 => {
            return Err(ParseError::Unsupported(
                "fat/universal container (FAT magic read little-endian appears byte-swapped); \
                 parse with FatFile and select a slice"
                    .into(),
            ));
        }
        other => {
            return Err(ParseError::BadSignature {
                what: "Mach-O",
                offset: 0,
                found: other,
                expected: MH_MAGIC_64,
            });
        }
    }

    let cputype = CpuType::from(r.u32()? as i32);
    let cpusubtype = r.u32()? as i32;
    let filetype = FileType::from(r.u32()?);
    let ncmds = r.u32()?;
    let sizeofcmds = r.u32()?;
    let flags = r.u32()?;
    let _reserved = r.u32()?;

    Ok(MachHeader {
        cputype,
        cpusubtype,
        filetype,
        ncmds,
        sizeofcmds,
        flags,
    })
}

/// Parse an `LC_SEGMENT_64` command body (including the 8-byte prefix).
fn parse_segment_64(body: &[u8]) -> Result<Segment64> {
    if (body.len() as u64) < SEGMENT_COMMAND_64_SIZE {
        return Err(ParseError::Unsupported(format!(
            "LC_SEGMENT_64 cmdsize {} is smaller than a segment_command_64 ({SEGMENT_COMMAND_64_SIZE} bytes)",
            body.len()
        )));
    }
    let mut r = Reader::new(body);
    r.seek(LOAD_COMMAND_MIN_SIZE as usize)?;
    let segname = fixed_name(r.take(16)?);
    let vmaddr = r.u64()?;
    let vmsize = r.u64()?;
    let fileoff = r.u64()?;
    let filesize = r.u64()?;
    let maxprot = r.u32()?;
    let initprot = r.u32()?;
    let nsects = r.u32()?;
    let flags = r.u32()?;

    // Cannot overflow: nsects is a u32 and SECTION_64_SIZE is 80, so the
    // product is far below u64::MAX. The bound against cmdsize is what
    // stops a crafted nsects from allocating or reading out of range.
    let needed = SEGMENT_COMMAND_64_SIZE + nsects as u64 * SECTION_64_SIZE;
    if needed > body.len() as u64 {
        return Err(ParseError::Unsupported(format!(
            "LC_SEGMENT_64 '{segname}' declares {nsects} sections needing {needed:#x} bytes, but cmdsize is {:#x}",
            body.len()
        )));
    }

    let mut sections = Vec::with_capacity(nsects as usize);
    for _ in 0..nsects {
        let sectname = fixed_name(r.take(16)?);
        let sect_segname = fixed_name(r.take(16)?);
        let addr = r.u64()?;
        let size = r.u64()?;
        let offset = r.u32()?;
        let align = r.u32()?;
        let _reloff = r.u32()?;
        let _nreloc = r.u32()?;
        let sect_flags = r.u32()?;
        r.skip(12)?; // reserved1..reserved3
        sections.push(Section64 {
            sectname,
            segname: sect_segname,
            addr,
            size,
            offset,
            align,
            flags: sect_flags,
        });
    }

    Ok(Segment64 {
        segname,
        vmaddr,
        vmsize,
        fileoff,
        filesize,
        maxprot,
        initprot,
        flags,
        sections,
    })
}

/// Parse an `LC_SYMTAB` command body; `data` is the whole file, since the
/// symbol and string tables live outside the load-command region.
fn parse_symtab(data: &[u8], body: &[u8]) -> Result<Vec<Symbol>> {
    if body.len() < SYMTAB_COMMAND_SIZE {
        return Err(ParseError::Unsupported(format!(
            "LC_SYMTAB cmdsize {} is smaller than a symtab_command ({SYMTAB_COMMAND_SIZE} bytes)",
            body.len()
        )));
    }
    let mut r = Reader::new(body);
    r.seek(LOAD_COMMAND_MIN_SIZE as usize)?;
    let symoff = r.u32()?;
    let nsyms = r.u32()?;
    let stroff = r.u32()?;
    let strsize = r.u32()?;

    // Bounds-check both tables before allocating anything: nsyms is
    // thereby capped at file_len / 16, so a crafted count in a small file
    // can never drive a giant allocation.
    check_table(symoff as u64, nsyms as u64, NLIST_64_SIZE, data.len(), "symbol table")?;
    check_range(stroff as u64, strsize as u64, data.len(), "symbol string table")?;

    let mut symbols = Vec::with_capacity(nsyms as usize);
    let mut r = Reader::new(data);
    r.seek(symoff as usize)?;
    for _ in 0..nsyms {
        let n_strx = r.u32()?;
        let n_type = r.u8()?;
        let n_sect = r.u8()?;
        let n_desc = r.u16()?;
        let n_value = r.u64()?;
        let name = strtab_string(
            data,
            stroff as usize,
            strsize as usize,
            n_strx as usize,
            "symbol",
        )?;
        symbols.push(Symbol {
            name,
            value: n_value,
            kind: SymbolKind::from_n_type(n_type),
            external: n_type & N_EXT != 0,
            sect: n_sect,
            desc: n_desc,
        });
    }
    Ok(symbols)
}

/// Parse an `LC_MAIN` command body, returning `entryoff`.
fn parse_entry_point(body: &[u8]) -> Result<u64> {
    if body.len() < ENTRY_POINT_COMMAND_SIZE {
        return Err(ParseError::Unsupported(format!(
            "LC_MAIN cmdsize {} is smaller than an entry_point_command ({ENTRY_POINT_COMMAND_SIZE} bytes)",
            body.len()
        )));
    }
    let mut r = Reader::new(body);
    r.seek(LOAD_COMMAND_MIN_SIZE as usize)?;
    let entryoff = r.u64()?;
    let _stacksize = r.u64()?;
    Ok(entryoff)
}

/// Parse an `LC_LOAD_DYLIB` command body. The library name is an `lc_str`:
/// an offset from the start of the command to a NUL-terminated string that
/// must lie within the command's own `cmdsize`.
fn parse_dylib(body: &[u8]) -> Result<Dylib> {
    if body.len() < DYLIB_COMMAND_SIZE {
        return Err(ParseError::Unsupported(format!(
            "LC_LOAD_DYLIB cmdsize {} is smaller than a dylib_command ({DYLIB_COMMAND_SIZE} bytes)",
            body.len()
        )));
    }
    let mut r = Reader::new(body);
    r.seek(LOAD_COMMAND_MIN_SIZE as usize)?;
    let name_off = r.u32()? as usize;
    let timestamp = r.u32()?;
    let current_version = r.u32()?;
    let compatibility_version = r.u32()?;

    if name_off >= body.len() {
        return Err(ParseError::Unsupported(format!(
            "LC_LOAD_DYLIB name offset {name_off:#x} is outside its cmdsize {:#x}",
            body.len()
        )));
    }
    let bytes = &body[name_off..];
    // The name is NUL-terminated inside the command (which is NUL-padded
    // to an 8-byte multiple); tolerate a missing terminator by taking the
    // rest of the command.
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let name = String::from_utf8_lossy(&bytes[..len]).into_owned();

    Ok(Dylib {
        name,
        timestamp,
        current_version,
        compatibility_version,
    })
}

/// Size of a `linkedit_data_command` (`cmd`, `cmdsize`, `dataoff`,
/// `datasize`).
const LINKEDIT_DATA_COMMAND_SIZE: usize = 16;

/// Cap on function starts decoded from `LC_FUNCTION_STARTS`, so a
/// crafted `datasize` cannot over-allocate. Larger than any real
/// program's function count.
const MAX_FUNCTION_STARTS: usize = 1 << 21;

/// Parse a `linkedit_data_command` body, returning `(dataoff, datasize)`
/// — the file location of the payload the command points at.
fn parse_linkedit_data(body: &[u8]) -> Result<(u32, u32)> {
    if body.len() < LINKEDIT_DATA_COMMAND_SIZE {
        return Err(ParseError::Unsupported(format!(
            "linkedit_data_command cmdsize {} is smaller than {LINKEDIT_DATA_COMMAND_SIZE} bytes",
            body.len()
        )));
    }
    let mut r = Reader::new(body);
    r.seek(LOAD_COMMAND_MIN_SIZE as usize)?;
    let dataoff = r.u32()?;
    let datasize = r.u32()?;
    Ok((dataoff, datasize))
}

/// `encryption_info_command_64` body (including the 8-byte LC prefix).
const ENCRYPTION_INFO_64_SIZE: usize = 24;

fn parse_encryption_info_64(body: &[u8]) -> Result<EncryptionInfo> {
    if body.len() < ENCRYPTION_INFO_64_SIZE {
        return Err(ParseError::Unsupported(format!(
            "LC_ENCRYPTION_INFO_64 cmdsize {} too small",
            body.len()
        )));
    }
    let mut r = Reader::new(body);
    r.seek(LOAD_COMMAND_MIN_SIZE as usize)?;
    let cryptoff = r.u32()?;
    let cryptsize = r.u32()?;
    let cryptid = r.u32()?;
    Ok(EncryptionInfo {
        cryptoff,
        cryptsize,
        cryptid,
    })
}

/// `dyld_chained_fixups_header` size.
const CHAINED_FIXUPS_HEADER_SIZE: usize = 28;
/// Fixed header of `dyld_chained_starts_in_segment` before `page_start[]`.
const CHAINED_STARTS_SEG_HEADER: usize = 22;

fn import_entry_size(imports_format: u32) -> Option<u64> {
    match imports_format {
        DYLD_CHAINED_IMPORT => Some(4),
        DYLD_CHAINED_IMPORT_ADDEND => Some(8),
        DYLD_CHAINED_IMPORT_ADDEND64 => Some(16),
        _ => None,
    }
}

/// Stride in bytes for a `DYLD_CHAINED_PTR_*` format we know how to walk.
fn pointer_stride(pointer_format: u16) -> Option<u64> {
    match pointer_format {
        DYLD_CHAINED_PTR_ARM64E
        | DYLD_CHAINED_PTR_ARM64E_USERLAND
        | DYLD_CHAINED_PTR_ARM64E_USERLAND24 => Some(8),
        DYLD_CHAINED_PTR_ARM64E_KERNEL | DYLD_CHAINED_PTR_ARM64E_FIRMWARE => Some(4),
        DYLD_CHAINED_PTR_64 | DYLD_CHAINED_PTR_64_OFFSET => Some(4),
        _ => None,
    }
}

/// Decode one chain pointer. Returns `(is_bind, ordinal, next)` where
/// `next` is the chain step count (multiply by stride for byte distance).
fn decode_chain_pointer(pointer_format: u16, raw: u64) -> Option<(bool, u32, u32)> {
    match pointer_format {
        DYLD_CHAINED_PTR_64 | DYLD_CHAINED_PTR_64_OFFSET => {
            let bind = ((raw >> 63) & 1) != 0;
            let next = ((raw >> 51) & 0xFFF) as u32;
            let ordinal = (raw & 0xFF_FFFF) as u32;
            Some((bind, ordinal, next))
        }
        DYLD_CHAINED_PTR_ARM64E
        | DYLD_CHAINED_PTR_ARM64E_KERNEL
        | DYLD_CHAINED_PTR_ARM64E_USERLAND
        | DYLD_CHAINED_PTR_ARM64E_FIRMWARE => {
            // auth is bit 63; bind is bit 62; next is bits 51..61 (11 bits).
            let bind = ((raw >> 62) & 1) != 0;
            let next = ((raw >> 51) & 0x7FF) as u32;
            let ordinal = (raw & 0xFFFF) as u32;
            Some((bind, ordinal, next))
        }
        DYLD_CHAINED_PTR_ARM64E_USERLAND24 => {
            let bind = ((raw >> 62) & 1) != 0;
            let next = ((raw >> 51) & 0x7FF) as u32;
            let ordinal = (raw & 0xFF_FFFF) as u32;
            Some((bind, ordinal, next))
        }
        _ => None,
    }
}

fn preferred_load_address(segments: &[Segment64]) -> u64 {
    segments
        .iter()
        .find(|s| s.fileoff == 0 && s.filesize != 0)
        .or_else(|| segments.first())
        .map(|s| s.vmaddr)
        .unwrap_or(0)
}

fn read_u64_at_va(data: &[u8], segments: &[Segment64], va: u64) -> Option<u64> {
    let seg = segments.iter().find(|s| s.contains_vaddr(va))?;
    let off = seg.fileoff.checked_add(va - seg.vmaddr)? as usize;
    if off.checked_add(8)? > data.len() {
        return None;
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[off..off + 8]);
    Some(u64::from_le_bytes(buf))
}

fn parse_chained_fixups(
    data: &[u8],
    dataoff: u32,
    datasize: u32,
    segments: &[Segment64],
) -> Result<ChainedFixups> {
    check_range(
        dataoff as u64,
        datasize as u64,
        data.len(),
        "chained fixups blob",
    )?;
    if (datasize as usize) < CHAINED_FIXUPS_HEADER_SIZE {
        return Err(ParseError::Unsupported(
            "chained fixups blob smaller than header".into(),
        ));
    }
    let base = dataoff as usize;
    let blob = &data[base..base + datasize as usize];
    let mut r = Reader::new(blob);
    let fixups_version = r.u32()?;
    let starts_offset = r.u32()?;
    let imports_offset = r.u32()?;
    let symbols_offset = r.u32()?;
    let imports_count = r.u32()?;
    let imports_format = r.u32()?;
    let symbols_format = r.u32()?;

    if imports_count > MAX_CHAINED_IMPORTS {
        return Err(ParseError::Unsupported(format!(
            "chained imports_count {imports_count} exceeds cap {MAX_CHAINED_IMPORTS}"
        )));
    }

    let mut import_names = Vec::new();
    if symbols_format == 0
        && let Some(entsize) = import_entry_size(imports_format)
    {
        let imp_off = imports_offset as usize;
        let sym_off = symbols_offset as usize;
        check_table(
            imp_off as u64,
            imports_count as u64,
            entsize,
            blob.len(),
            "chained imports",
        )?;
        if sym_off > blob.len() {
            return Err(ParseError::Unsupported(
                "chained symbols_offset past blob".into(),
            ));
        }
        let mut ir = Reader::new(blob);
        ir.seek(imp_off)?;
        for _ in 0..imports_count {
            let name_offset = match imports_format {
                DYLD_CHAINED_IMPORT | DYLD_CHAINED_IMPORT_ADDEND => {
                    let word = ir.u32()?;
                    if imports_format == DYLD_CHAINED_IMPORT_ADDEND {
                        let _addend = ir.u32()?;
                    }
                    (word >> 9) as usize
                }
                DYLD_CHAINED_IMPORT_ADDEND64 => {
                    let word = ir.u64()?;
                    let _addend = ir.u64()?;
                    (word >> 32) as usize
                }
                _ => unreachable!(),
            };
            let name = read_c_string(blob, sym_off.saturating_add(name_offset))?;
            import_names.push(name);
        }
    }

    let bind_slots = if starts_offset != 0 {
        walk_chained_binds(data, blob, starts_offset as usize, segments, &import_names)?
    } else {
        Vec::new()
    };

    Ok(ChainedFixups {
        fixups_version,
        imports_count,
        imports_format,
        symbols_format,
        import_names,
        bind_slots,
        dataoff,
        datasize,
    })
}

/// Walk `dyld_chained_starts_in_image` / per-segment page chains and
/// collect bind slots. Unknown pointer formats and MULTI page starts are
/// skipped (honest partial); hostile counts refuse.
fn walk_chained_binds(
    data: &[u8],
    blob: &[u8],
    starts_offset: usize,
    segments: &[Segment64],
    import_names: &[String],
) -> Result<Vec<ChainedBindSlot>> {
    if starts_offset >= blob.len() {
        return Err(ParseError::Unsupported(
            "chained starts_offset past blob".into(),
        ));
    }
    let starts = &blob[starts_offset..];
    if starts.len() < 4 {
        return Err(ParseError::Unsupported(
            "chained starts truncated before seg_count".into(),
        ));
    }
    let mut sr = Reader::new(starts);
    let seg_count = sr.u32()?;
    if seg_count > MAX_CHAINED_SEGS {
        return Err(ParseError::Unsupported(format!(
            "chained seg_count {seg_count} exceeds cap {MAX_CHAINED_SEGS}"
        )));
    }
    check_table(
        4,
        seg_count as u64,
        4,
        starts.len(),
        "chained seg_info_offset",
    )?;

    let image_base = preferred_load_address(segments);
    let mut bind_slots = Vec::new();
    let mut steps = 0usize;

    for i in 0..seg_count {
        let info_off = {
            let mut rr = Reader::new(starts);
            rr.seek(4 + (i as usize) * 4)?;
            rr.u32()?
        };
        if info_off == 0 {
            continue;
        }
        let info_base = info_off as usize;
        if info_base
            .checked_add(CHAINED_STARTS_SEG_HEADER)
            .is_none_or(|e| e > starts.len())
        {
            return Err(ParseError::Unsupported(
                "chained starts_in_segment header past blob".into(),
            ));
        }
        let mut ir = Reader::new(starts);
        ir.seek(info_base)?;
        let _size = ir.u32()?;
        let page_size = ir.u16()? as u64;
        let pointer_format = ir.u16()?;
        let segment_offset = ir.u64()?;
        let _max_valid_pointer = ir.u32()?;
        let page_count = ir.u16()? as u32;

        if page_size != 0x1000 && page_size != 0x4000 {
            return Err(ParseError::Unsupported(format!(
                "chained page_size {page_size:#x} is not 0x1000 or 0x4000"
            )));
        }
        if page_count > MAX_CHAINED_PAGES {
            return Err(ParseError::Unsupported(format!(
                "chained page_count {page_count} exceeds cap {MAX_CHAINED_PAGES}"
            )));
        }
        let page_starts_off = info_base + CHAINED_STARTS_SEG_HEADER;
        check_table(
            page_starts_off as u64,
            page_count as u64,
            2,
            starts.len(),
            "chained page_start",
        )?;

        let Some(stride) = pointer_stride(pointer_format) else {
            // Format we do not decode: skip this segment rather than guess.
            continue;
        };

        let seg_va = image_base.wrapping_add(segment_offset);
        for page_idx in 0..page_count {
            let mut pr = Reader::new(starts);
            pr.seek(page_starts_off + (page_idx as usize) * 2)?;
            let page_start = pr.u16()?;
            if page_start == DYLD_CHAINED_PTR_START_NONE {
                continue;
            }
            // Multi-start pages (32-bit) are out of scope for this 64-bit walk.
            if page_start & DYLD_CHAINED_PTR_START_MULTI != 0 {
                continue;
            }

            let mut loc = seg_va
                .wrapping_add((page_idx as u64).wrapping_mul(page_size))
                .wrapping_add(page_start as u64);

            loop {
                if steps >= MAX_CHAIN_STEPS {
                    return Err(ParseError::Unsupported(format!(
                        "chained fixup walk exceeds step cap {MAX_CHAIN_STEPS}"
                    )));
                }
                steps += 1;

                let Some(raw) = read_u64_at_va(data, segments, loc) else {
                    break;
                };
                let Some((is_bind, ordinal, next)) = decode_chain_pointer(pointer_format, raw)
                else {
                    break;
                };
                if is_bind {
                    if bind_slots.len() >= MAX_CHAINED_BINDS {
                        return Err(ParseError::Unsupported(format!(
                            "chained bind slots exceed cap {MAX_CHAINED_BINDS}"
                        )));
                    }
                    let name = import_names
                        .get(ordinal as usize)
                        .cloned()
                        .unwrap_or_default();
                    bind_slots.push(ChainedBindSlot {
                        slot_va: loc,
                        ordinal,
                        name,
                    });
                }
                if next == 0 {
                    break;
                }
                loc = loc.wrapping_add((next as u64).wrapping_mul(stride));
            }
        }
    }

    Ok(bind_slots)
}

fn read_c_string(blob: &[u8], off: usize) -> Result<String> {
    if off >= blob.len() {
        return Ok(String::new());
    }
    let end = blob[off..]
        .iter()
        .position(|&b| b == 0)
        .map(|i| off + i)
        .unwrap_or(blob.len());
    let raw = &blob[off..end];
    Ok(String::from_utf8_lossy(raw).into_owned())
}

/// Decode the `LC_FUNCTION_STARTS` payload: a run of ULEB128 deltas, the
/// first added to `base_vmaddr` (the image's first mapped segment) and
/// each subsequent one to the running address. A zero delta or the end
/// of the payload terminates. Returns absolute VAs, sorted and
/// deduplicated. Never panics: an out-of-range or truncated payload
/// simply yields fewer entries.
fn decode_function_starts(data: &[u8], dataoff: u32, datasize: u32, base_vmaddr: u64) -> Vec<u64> {
    let start = dataoff as usize;
    let end = start.saturating_add(datasize as usize).min(data.len());
    if start >= end {
        return Vec::new();
    }
    let payload = &data[start..end];
    let mut out = Vec::new();
    let mut addr = base_vmaddr;
    let mut i = 0;
    let mut first = true;
    while i < payload.len() && out.len() < MAX_FUNCTION_STARTS {
        let Some((delta, next)) = read_uleb128(payload, i) else {
            break;
        };
        i = next;
        // A zero delta (other than a leading one) marks the end of the
        // table; a leading zero means the first function is at the base.
        if delta == 0 && !first {
            break;
        }
        addr = addr.wrapping_add(delta);
        out.push(addr);
        first = false;
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Read an unsigned LEB128 value from `buf` at `pos`, returning the value
/// and the index just past it, or `None` if it runs off the end or
/// exceeds 64 bits.
fn read_uleb128(buf: &[u8], pos: usize) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    let mut i = pos;
    loop {
        let byte = *buf.get(i)?;
        i += 1;
        if shift < 64 {
            result |= ((byte & 0x7F) as u64) << shift;
        } else if byte & 0x7F != 0 {
            return None; // overlong: significant bits past bit 63
        }
        shift += 7;
        if byte & 0x80 == 0 {
            return Some((result, i));
        }
        if shift >= 128 {
            return None; // pathologically long
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const IMG_SIZE: usize = 0x300;
    const TEXT_VMADDR: u64 = 0x1_0000_0000;
    const ENTRYOFF: u64 = 0x140;
    const SEG_CMD_OFF: usize = 0x20;
    const SYMTAB_CMD_OFF: usize = 0xB8;
    const MAIN_CMD_OFF: usize = 0xD0;
    const DYLIB_CMD_OFF: usize = 0xE8;
    const UNKNOWN_CMD_OFF: usize = 0x120;
    const SYMS_OFF: usize = 0x200;
    const STRS_OFF: usize = 0x240;
    /// The string table runs exactly to end-of-file so that every
    /// truncated prefix of the image fails to parse (truncation sweep).
    const STRS_SIZE: usize = IMG_SIZE - STRS_OFF;
    const DYLIB_NAME: &str = "/usr/lib/libSystem.B.dylib";

    fn put(img: &mut [u8], off: usize, bytes: &[u8]) {
        img[off..off + bytes.len()].copy_from_slice(bytes);
    }

    fn put32(img: &mut [u8], off: usize, v: u32) {
        put(img, off, &v.to_le_bytes());
    }

    fn put64(img: &mut [u8], off: usize, v: u64) {
        put(img, off, &v.to_le_bytes());
    }

    fn put32be(img: &mut [u8], off: usize, v: u32) {
        put(img, off, &v.to_be_bytes());
    }

    fn put64be(img: &mut [u8], off: usize, v: u64) {
        put(img, off, &v.to_be_bytes());
    }

    fn write_nlist(img: &mut [u8], idx: usize, n_strx: u32, n_type: u8, n_sect: u8, value: u64) {
        let base = SYMS_OFF + idx * 16;
        put32(img, base, n_strx);
        img[base + 4] = n_type;
        img[base + 5] = n_sect;
        // n_desc left 0.
        put64(img, base + 8, value);
    }

    /// Build a minimal but well-formed thin arm64 Mach-O executable:
    /// an `__TEXT` segment with one `__text` section, an `LC_SYMTAB` with
    /// three symbols (defined external, defined local, undefined external),
    /// an `LC_MAIN`, an `LC_LOAD_DYLIB`, and one unknown load command that
    /// the parser must skip. Shared with the trait-layer tests in
    /// [`crate::model`].
    pub(crate) fn synthetic_macho64() -> Vec<u8> {
        let mut img = vec![0u8; IMG_SIZE];

        // mach_header_64.
        put32(&mut img, 0, MH_MAGIC_64);
        put32(&mut img, 4, 0x0100_000C); // cputype: CPU_TYPE_ARM64
        put32(&mut img, 8, 0); // cpusubtype
        put32(&mut img, 12, 2); // filetype: MH_EXECUTE
        put32(&mut img, 16, 5); // ncmds
        put32(&mut img, 20, 0x110); // sizeofcmds (272)
        put32(&mut img, 24, MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE);

        // LC_SEGMENT_64 "__TEXT" with one section, cmdsize 72 + 80 = 152.
        let s = SEG_CMD_OFF;
        put32(&mut img, s, LC_SEGMENT_64);
        put32(&mut img, s + 4, 152);
        put(&mut img, s + 8, b"__TEXT");
        put64(&mut img, s + 24, TEXT_VMADDR); // vmaddr
        put64(&mut img, s + 32, 0x4000); // vmsize (zero-fill tail past filesize)
        put64(&mut img, s + 40, 0); // fileoff
        put64(&mut img, s + 48, IMG_SIZE as u64); // filesize
        put32(&mut img, s + 56, VM_PROT_READ | VM_PROT_EXECUTE); // maxprot
        put32(&mut img, s + 60, VM_PROT_READ | VM_PROT_EXECUTE); // initprot
        put32(&mut img, s + 64, 1); // nsects
        // section_64 "__text".
        let t = s + 72;
        put(&mut img, t, b"__text");
        put(&mut img, t + 16, b"__TEXT");
        put64(&mut img, t + 32, TEXT_VMADDR + ENTRYOFF); // addr
        put64(&mut img, t + 40, 0x40); // size
        put32(&mut img, t + 48, ENTRYOFF as u32); // offset
        put32(&mut img, t + 52, 2); // align (2^2)
        put32(&mut img, t + 64, 0x8000_0400); // S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS

        // LC_SYMTAB.
        let y = SYMTAB_CMD_OFF;
        put32(&mut img, y, LC_SYMTAB);
        put32(&mut img, y + 4, 24);
        put32(&mut img, y + 8, SYMS_OFF as u32); // symoff
        put32(&mut img, y + 12, 3); // nsyms
        put32(&mut img, y + 16, STRS_OFF as u32); // stroff
        put32(&mut img, y + 20, STRS_SIZE as u32); // strsize

        // LC_MAIN.
        let m = MAIN_CMD_OFF;
        put32(&mut img, m, LC_MAIN);
        put32(&mut img, m + 4, 24);
        put64(&mut img, m + 8, ENTRYOFF); // entryoff
        // stacksize left 0.

        // LC_LOAD_DYLIB: 24-byte dylib_command + name, padded to cmdsize 56.
        let d = DYLIB_CMD_OFF;
        put32(&mut img, d, LC_LOAD_DYLIB);
        put32(&mut img, d + 4, 56);
        put32(&mut img, d + 8, 24); // lc_str name offset
        put32(&mut img, d + 12, 2); // timestamp
        put32(&mut img, d + 16, 0x0001_0000); // current_version 1.0.0
        put32(&mut img, d + 20, 0x0001_0000); // compatibility_version
        put(&mut img, d + 24, DYLIB_NAME.as_bytes());

        // An unknown load command the parser must skip, cmdsize 16.
        let u = UNKNOWN_CMD_OFF;
        put32(&mut img, u, 0x3F); // not a modeled LC_* value
        put32(&mut img, u + 4, 16);
        put64(&mut img, u + 8, 0xDEAD_BEEF_DEAD_BEEF); // opaque payload

        // Symbols: _main (defined external), _helper (defined local),
        // _printf (undefined external).
        write_nlist(&mut img, 0, 1, 0x0F, 1, TEXT_VMADDR + ENTRYOFF); // N_SECT | N_EXT
        write_nlist(&mut img, 1, 7, 0x0E, 1, TEXT_VMADDR + ENTRYOFF + 0x20); // N_SECT
        write_nlist(&mut img, 2, 15, 0x01, 0, 0); // N_UNDF | N_EXT

        // String table: offsets 1, 7, 15.
        put(&mut img, STRS_OFF, b"\0_main\0_helper\0_printf\0");

        img
    }

    /// Build a fat container holding two copies of the thin image, the
    /// first re-labeled x86-64 so the slices are distinguishable.
    fn synthetic_fat(fat64: bool) -> Vec<u8> {
        let thin = synthetic_macho64();
        let mut slice_a = thin.clone();
        put32(&mut slice_a, 4, 0x0100_0007); // cputype: CPU_TYPE_X86_64
        let slice_b = thin; // arm64

        let off_a = 0x80;
        let off_b = off_a + slice_a.len();
        let mut img = vec![0u8; off_b + slice_b.len()];

        // fat_header — big-endian on disk, always.
        put32be(&mut img, 0, if fat64 { FAT_MAGIC_64 } else { FAT_MAGIC });
        put32be(&mut img, 4, 2); // nfat_arch

        let entsize = if fat64 { 32 } else { 20 };
        for (i, (cputype, off, size)) in [
            (0x0100_0007u32, off_a, slice_a.len()),
            (0x0100_000C, off_b, slice_b.len()),
        ]
        .iter()
        .enumerate()
        {
            let base = 8 + i * entsize;
            put32be(&mut img, base, *cputype);
            put32be(&mut img, base + 4, 0); // cpusubtype
            if fat64 {
                put64be(&mut img, base + 8, *off as u64);
                put64be(&mut img, base + 16, *size as u64);
                put32be(&mut img, base + 24, 14); // align (2^14)
            } else {
                put32be(&mut img, base + 8, *off as u32);
                put32be(&mut img, base + 12, *size as u32);
                put32be(&mut img, base + 16, 14);
            }
        }

        put(&mut img, off_a, &slice_a);
        put(&mut img, off_b, &slice_b);
        img
    }

    #[test]
    fn parses_synthetic_thin_header() {
        let macho = MachFile::parse(&synthetic_macho64()).unwrap();
        assert_eq!(macho.header.cputype, CpuType::Arm64);
        assert_eq!(macho.header.cpusubtype, 0);
        assert_eq!(macho.header.filetype, FileType::Execute);
        assert_eq!(macho.header.ncmds, 5);
        assert_eq!(macho.header.sizeofcmds, 0x110);
        assert_eq!(
            macho.header.flags,
            MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL | MH_PIE
        );
        assert!(macho.header.flags & MH_PIE != 0);
        assert_eq!(macho.segments.len(), 1);
        assert_eq!(macho.symbols.len(), 3);
        assert_eq!(macho.dylibs.len(), 1);
    }

    #[test]
    fn parses_segment_with_sections() {
        let macho = MachFile::parse(&synthetic_macho64()).unwrap();
        let text = macho.segment_by_name("__TEXT").unwrap();
        assert_eq!(text.vmaddr, TEXT_VMADDR);
        assert_eq!(text.vmsize, 0x4000);
        assert_eq!(text.fileoff, 0);
        assert_eq!(text.filesize, IMG_SIZE as u64);
        assert!(text.is_read());
        assert!(text.is_execute());
        assert!(!text.is_write());
        assert_eq!(text.sections.len(), 1);

        let sect = macho.section_by_name("__TEXT", "__text").unwrap();
        assert_eq!(sect.sectname, "__text");
        assert_eq!(sect.segname, "__TEXT");
        assert_eq!(sect.addr, TEXT_VMADDR + ENTRYOFF);
        assert_eq!(sect.size, 0x40);
        assert_eq!(sect.offset, ENTRYOFF as u32);
        assert_eq!(sect.align, 2);
        assert_eq!(sect.flags, 0x8000_0400);

        assert!(macho.segment_by_name("__DATA").is_none());
        assert!(macho.section_by_name("__TEXT", "__stubs").is_none());
    }

    #[test]
    fn parses_symbols_with_names_and_kinds() {
        let macho = MachFile::parse(&synthetic_macho64()).unwrap();

        let main = macho.symbol_by_name("_main").unwrap();
        assert_eq!(main.kind, SymbolKind::Defined);
        assert!(main.external);
        assert_eq!(main.sect, 1);
        assert_eq!(main.value, TEXT_VMADDR + ENTRYOFF);

        let helper = macho.symbol_by_name("_helper").unwrap();
        assert_eq!(helper.kind, SymbolKind::Defined);
        assert!(!helper.external);

        let printf = macho.symbol_by_name("_printf").unwrap();
        assert_eq!(printf.kind, SymbolKind::Undefined);
        assert!(printf.external);
        assert_eq!(printf.sect, 0); // NO_SECT

        assert!(macho.symbol_by_name("_absent").is_none());
    }

    #[test]
    fn parses_dylibs_and_entry_point() {
        let macho = MachFile::parse(&synthetic_macho64()).unwrap();
        assert_eq!(macho.entry_offset(), Some(ENTRYOFF));

        let dylib = &macho.dylibs[0];
        assert_eq!(dylib.name, DYLIB_NAME);
        assert_eq!(dylib.timestamp, 2);
        assert_eq!(dylib.current_version, 0x0001_0000);
        assert_eq!(dylib.compatibility_version, 0x0001_0000);
    }

    #[test]
    fn maps_vaddrs_through_segments() {
        let macho = MachFile::parse(&synthetic_macho64()).unwrap();
        assert_eq!(macho.vaddr_to_offset(TEXT_VMADDR).unwrap(), 0);
        assert_eq!(
            macho.vaddr_to_offset(TEXT_VMADDR + ENTRYOFF).unwrap(),
            ENTRYOFF as usize
        );
        // Last file-backed byte maps; the zero-fill tail does not.
        let last = TEXT_VMADDR + IMG_SIZE as u64 - 1;
        assert_eq!(macho.vaddr_to_offset(last).unwrap(), IMG_SIZE - 1);
        assert_eq!(
            macho.vaddr_to_offset(last + 1),
            Err(ParseError::UnmappedVaddr(last + 1))
        );
        assert_eq!(
            macho.vaddr_to_offset(0),
            Err(ParseError::UnmappedVaddr(0))
        );
        assert!(macho.segment_for_vaddr(TEXT_VMADDR).is_some());
        assert!(macho.segment_for_vaddr(0).is_none());
    }

    #[test]
    fn unknown_load_commands_are_skipped() {
        // The base image already carries one unknown command; swap its id
        // for another unmodeled value (including a LC_REQ_DYLD-flagged one)
        // and the parse must still succeed untouched.
        for cmd in [0x2Au32, 0x8000_1234] {
            let mut img = synthetic_macho64();
            put32(&mut img, UNKNOWN_CMD_OFF, cmd);
            let macho = MachFile::parse(&img).unwrap();
            assert_eq!(macho.segments.len(), 1);
            assert_eq!(macho.symbols.len(), 3);
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut img = synthetic_macho64();
        put32(&mut img, 0, 0x1234_5678);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::BadSignature { what: "Mach-O", .. }
        ));
    }

    #[test]
    fn rejects_32bit_big_endian_and_fat_as_thin() {
        // 32-bit magic.
        let mut img = synthetic_macho64();
        put32(&mut img, 0, MH_MAGIC);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        // Big-endian 64-bit image: on-disk bytes FE ED FA CF read
        // little-endian are MH_CIGAM_64.
        let mut img = synthetic_macho64();
        put32(&mut img, 0, MH_CIGAM_64);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        // A fat container handed to the thin parser: its big-endian magic
        // reads little-endian as FAT_CIGAM, and the error must say to use
        // FatFile rather than report a generic bad signature.
        let fat = synthetic_fat(false);
        match MachFile::parse(&fat).unwrap_err() {
            ParseError::Unsupported(msg) => assert!(msg.contains("FatFile"), "{msg}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn rejects_huge_ncmds_without_allocating() {
        // ncmds far beyond what sizeofcmds can hold must be rejected up
        // front, not iterated or allocated for.
        let mut img = synthetic_macho64();
        put32(&mut img, 16, 0xFFFF_FFFF);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));
    }

    #[test]
    fn rejects_bad_cmdsize() {
        // cmdsize 0: would loop forever if not rejected.
        let mut img = synthetic_macho64();
        put32(&mut img, SEG_CMD_OFF + 4, 0);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        // cmdsize smaller than the load_command header.
        let mut img = synthetic_macho64();
        put32(&mut img, SEG_CMD_OFF + 4, 4);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        // cmdsize extending past the declared command region.
        let mut img = synthetic_macho64();
        put32(&mut img, SEG_CMD_OFF + 4, 0x120);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        // Near-u32::MAX cmdsize: must not overflow the offset arithmetic.
        let mut img = synthetic_macho64();
        put32(&mut img, SEG_CMD_OFF + 4, 0xFFFF_FFF8);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));
    }

    #[test]
    fn rejects_nsects_overflowing_command() {
        // nsects needing more bytes than cmdsize provides.
        let mut img = synthetic_macho64();
        put32(&mut img, SEG_CMD_OFF + 64, 2);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        // A huge nsects in a 4 KB-scale file must never allocate for the
        // declared count.
        let mut img = synthetic_macho64();
        put32(&mut img, SEG_CMD_OFF + 64, 0xFFFF_FFFF);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));
    }

    #[test]
    fn rejects_symbol_table_out_of_bounds() {
        // nsyms whose table would dwarf the file: rejected before any
        // allocation (the count is capped by file_len / 16).
        let mut img = synthetic_macho64();
        put32(&mut img, SYMTAB_CMD_OFF + 12, 0x1000_0000);
        assert!(MachFile::parse(&img).is_err());

        // String table off the end of the file.
        let mut img = synthetic_macho64();
        put32(&mut img, SYMTAB_CMD_OFF + 16, 0x1000);
        assert!(MachFile::parse(&img).is_err());

        // Symbol name offset past the end of the string table.
        let mut img = synthetic_macho64();
        put32(&mut img, SYMS_OFF, 0x1000);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));
    }

    #[test]
    fn rejects_dylib_name_offset_out_of_range() {
        // lc_str offset at or past cmdsize.
        let mut img = synthetic_macho64();
        put32(&mut img, DYLIB_CMD_OFF + 8, 56);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        let mut img = synthetic_macho64();
        put32(&mut img, DYLIB_CMD_OFF + 8, 0xFFFF);
        assert!(matches!(
            MachFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));
    }

    #[test]
    fn truncated_thin_image_is_an_error_not_a_panic() {
        let img = synthetic_macho64();
        // The string table runs exactly to end-of-file, so every truncated
        // prefix must fail — and must never panic.
        for len in 0..img.len() {
            assert!(MachFile::parse(&img[..len]).is_err(), "len {len:#x}");
        }
        assert!(MachFile::parse(&img).is_ok());
    }

    #[test]
    fn parses_fat32_and_selects_slices() {
        let img = synthetic_fat(false);
        let fat = FatFile::parse(&img).unwrap();
        assert_eq!(fat.arches().len(), 2);

        let a = &fat.arches()[0];
        assert_eq!(a.cputype, CpuType::X86_64);
        assert_eq!(a.offset, 0x80);
        assert_eq!(a.size, IMG_SIZE as u64);
        assert_eq!(a.align, 14);
        assert_eq!(fat.arches()[1].cputype, CpuType::Arm64);

        // Slice selection and per-slice parsing.
        let arm = fat.arch_by_cputype(CpuType::Arm64).unwrap();
        let slice = arm.slice(&img).unwrap();
        assert_eq!(slice.len(), IMG_SIZE);
        let macho = arm.parse(&img).unwrap();
        assert_eq!(macho.header.cputype, CpuType::Arm64);
        assert!(macho.symbol_by_name("_main").is_some());

        let x86 = fat.arch_by_cputype(CpuType::X86_64).unwrap();
        assert_eq!(x86.parse(&img).unwrap().header.cputype, CpuType::X86_64);

        assert!(fat.arch_by_cputype(CpuType::Other(18)).is_none());

        // Slice bounds are re-checked against whatever buffer is passed.
        assert!(arm.slice(&img[..0x100]).is_err());
    }

    #[test]
    fn parses_fat64() {
        let img = synthetic_fat(true);
        let fat = FatFile::parse(&img).unwrap();
        assert_eq!(fat.arches().len(), 2);
        assert_eq!(fat.arches()[0].cputype, CpuType::X86_64);
        assert_eq!(fat.arches()[1].offset, 0x80 + IMG_SIZE as u64);
        let macho = fat.arches()[1].parse(&img).unwrap();
        assert_eq!(macho.header.cputype, CpuType::Arm64);
    }

    #[test]
    fn fat_rejects_byte_swapped_and_bad_magic() {
        // A fat header written little-endian on disk reads big-endian as
        // FAT_CIGAM: format violation, not a byte order to adapt to.
        let mut img = synthetic_fat(false);
        put32(&mut img, 0, FAT_MAGIC); // little-endian write = wrong order
        assert!(matches!(
            FatFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        // A thin image handed to the fat parser is a bad signature.
        assert!(matches!(
            FatFile::parse(&synthetic_macho64()).unwrap_err(),
            ParseError::BadSignature {
                what: "fat (universal)",
                ..
            }
        ));
    }

    #[test]
    fn fat_rejects_out_of_bounds_and_huge_counts() {
        // Arch count above the cap.
        let mut img = synthetic_fat(false);
        put32be(&mut img, 4, 0xFFFF_FFFF);
        assert!(matches!(
            FatFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        // Count under the cap but with a table larger than the file.
        let mut small = vec![0u8; 16];
        put32be(&mut small, 0, FAT_MAGIC);
        put32be(&mut small, 4, 60);
        assert!(matches!(
            FatFile::parse(&small).unwrap_err(),
            ParseError::UnexpectedEof { .. }
        ));

        // Second slice extended one byte past end-of-file.
        let mut img = synthetic_fat(false);
        put32be(&mut img, 8 + 20 + 12, IMG_SIZE as u32 + 1); // arch[1].size
        assert!(matches!(
            FatFile::parse(&img).unwrap_err(),
            ParseError::UnexpectedEof { .. }
        ));

        // fat_arch_64 offset+size overflowing a 64-bit offset.
        let mut img = synthetic_fat(true);
        put64be(&mut img, 8 + 8, u64::MAX); // arch[0].offset
        assert!(matches!(
            FatFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));
    }

    #[test]
    fn truncated_fat_image_is_an_error_not_a_panic() {
        for fat64 in [false, true] {
            let img = synthetic_fat(fat64);
            // The last slice runs to end-of-file, so every truncated
            // prefix must fail — and must never panic.
            for len in 0..img.len() {
                assert!(FatFile::parse(&img[..len]).is_err(), "fat64={fat64} len {len:#x}");
            }
            assert!(FatFile::parse(&img).is_ok());
        }
    }

    /// Smoke-test against a real Mach-O binary when one is present on this
    /// machine (macOS system binaries; on other platforms /bin/ls is ELF
    /// or absent, and the test is a no-op with synthetic coverage standing
    /// alone).
    #[test]
    fn smoke_parses_real_macho_if_present() {
        let path = "/bin/ls";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let Ok(data) = std::fs::read(path) else {
            return;
        };
        if data.len() < 4 {
            return;
        }

        let be = u32::from_be_bytes(data[0..4].try_into().unwrap());
        let le = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let thin_copy;
        let thin: &[u8] = if be == FAT_MAGIC || be == FAT_MAGIC_64 {
            let fat = FatFile::parse(&data).unwrap_or_else(|e| panic!("{path}: {e}"));
            assert!(!fat.arches().is_empty(), "{path}: empty fat container");
            // Pick the first slice that is a little-endian 64-bit image.
            let Some(slice) = fat.arches().iter().find_map(|arch| {
                let s = arch.slice(&data).unwrap_or_else(|e| panic!("{path}: {e}"));
                (s.len() >= 4 && u32::from_le_bytes(s[0..4].try_into().unwrap()) == MH_MAGIC_64)
                    .then_some(s)
            }) else {
                return; // no 64-bit little-endian slice: out of scope
            };
            thin_copy = slice.to_vec();
            &thin_copy
        } else if le == MH_MAGIC_64 {
            &data
        } else {
            return; // e.g. an ELF /bin/ls on Linux: out of scope here
        };

        let macho = MachFile::parse(thin).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert!(
            macho.segments.iter().any(|s| s.segname == "__TEXT"),
            "{path}: no __TEXT segment"
        );
        assert!(!macho.dylibs.is_empty(), "{path}: no imported dylibs");
        // Symbol parsing must succeed; system binaries keep at least their
        // undefined imports in the symbol table.
        assert!(!macho.symbols.is_empty(), "{path}: empty symbol table");
    }

    #[test]
    fn uleb128_reads_single_and_multi_byte() {
        assert_eq!(read_uleb128(&[0x00], 0), Some((0, 1)));
        assert_eq!(read_uleb128(&[0x7F], 0), Some((127, 1)));
        // 0x80 0x01 = 128; two-byte little-endian base-128.
        assert_eq!(read_uleb128(&[0x80, 0x01], 0), Some((128, 2)));
        assert_eq!(read_uleb128(&[0xE5, 0x8E, 0x26], 0), Some((624485, 3)));
        // Runs off the end (continuation bit set, no next byte).
        assert_eq!(read_uleb128(&[0x80], 0), None);
    }

    #[test]
    fn function_starts_accumulate_deltas_from_base() {
        // base 0x1000, deltas 0x10, 0x20, 0x8 → 0x1010, 0x1030, 0x1038.
        let payload = [0x10, 0x20, 0x08];
        let starts = decode_function_starts(&payload, 0, payload.len() as u32, 0x1000);
        assert_eq!(starts, vec![0x1010, 0x1030, 0x1038]);
    }

    #[test]
    fn function_starts_stop_at_zero_delta() {
        // A non-leading zero terminates; bytes after it are ignored.
        let payload = [0x10, 0x00, 0x20];
        let starts = decode_function_starts(&payload, 0, payload.len() as u32, 0x1000);
        assert_eq!(starts, vec![0x1010]);
    }

    #[test]
    fn function_starts_out_of_range_dataoff_is_empty() {
        let payload = [0x10, 0x20];
        // dataoff past the buffer end must yield nothing, never panic.
        assert!(decode_function_starts(&payload, 99, 2, 0x1000).is_empty());
        // datasize past the end is clamped, not read past the buffer.
        assert_eq!(
            decode_function_starts(&payload, 0, 999, 0x1000),
            vec![0x1010, 0x1030]
        );
    }

    #[test]
    fn chained_fixups_parse_import_names() {
        // Header + one DYLD_CHAINED_IMPORT + symbol "puts\0"
        let mut blob = vec![0u8; 64];
        // version, starts_off, imports_off=28, symbols_off=32, count=1, format=1, symfmt=0
        blob[0..4].copy_from_slice(&0u32.to_le_bytes());
        blob[4..8].copy_from_slice(&0u32.to_le_bytes());
        blob[8..12].copy_from_slice(&28u32.to_le_bytes());
        blob[12..16].copy_from_slice(&32u32.to_le_bytes());
        blob[16..20].copy_from_slice(&1u32.to_le_bytes());
        blob[20..24].copy_from_slice(&1u32.to_le_bytes());
        blob[24..28].copy_from_slice(&0u32.to_le_bytes());
        // import word: name_offset=0 in symbols → bits 9.. = 0
        blob[28..32].copy_from_slice(&0u32.to_le_bytes());
        blob[32..37].copy_from_slice(b"puts\0");
        let cf = parse_chained_fixups(&blob, 0, blob.len() as u32, &[]).unwrap();
        assert_eq!(cf.imports_count, 1);
        assert_eq!(cf.import_names, vec!["puts".to_string()]);
        assert!(cf.bind_slots.is_empty());
    }

    #[test]
    fn chained_imports_cap_refuses() {
        let mut blob = vec![0u8; 28];
        blob[16..20].copy_from_slice(&(MAX_CHAINED_IMPORTS + 1).to_le_bytes());
        assert!(parse_chained_fixups(&blob, 0, 28, &[]).is_err());
    }

    #[test]
    fn chained_seg_count_cap_refuses() {
        let mut blob = vec![0u8; 64];
        blob[4..8].copy_from_slice(&28u32.to_le_bytes()); // starts at 28
        blob[28..32].copy_from_slice(&(MAX_CHAINED_SEGS + 1).to_le_bytes());
        assert!(parse_chained_fixups(&blob, 0, blob.len() as u32, &[]).is_err());
    }

    /// Minimal segment covering `[vmaddr, vmaddr+filesize)` at fileoff 0.
    fn test_seg(vmaddr: u64, filesize: u64) -> Segment64 {
        Segment64 {
            segname: "__DATA".into(),
            vmaddr,
            vmsize: filesize,
            fileoff: 0,
            filesize,
            maxprot: VM_PROT_READ | VM_PROT_WRITE,
            initprot: VM_PROT_READ | VM_PROT_WRITE,
            flags: 0,
            sections: Vec::new(),
        }
    }

    #[test]
    fn chained_fixups_walk_ptr64_offset_bind() {
        // File layout:
        //   [0x100]: DYLD_CHAINED_PTR_64_OFFSET bind ordinal 0, next 0
        //   [0x800]: fixups blob (header + starts + import + "puts")
        // Segment at preferred base 0x1_0000_0000, fileoff 0 → VA 0x1_0000_0100.
        const BASE: u64 = 0x1_0000_0000;
        let mut data = vec![0u8; 0x1000];
        let raw: u64 = 1u64 << 63; // bind=1, ordinal=0, next=0
        data[0x100..0x108].copy_from_slice(&raw.to_le_bytes());

        let blob_off = 0x800usize;
        // Header: starts=28, imports=28+4+4+24=60, symbols=64, count=1, fmt=1
        // starts at blob+28:
        //   seg_count=1, seg_info_offset[0]=8
        //   at +8: size=24, page_size=0x1000, fmt=6, seg_off=0, max=0, pages=1, start=0x100
        let imports_off = 60u32;
        let symbols_off = 64u32;
        let mut blob = vec![0u8; 96];
        blob[0..4].copy_from_slice(&0u32.to_le_bytes());
        blob[4..8].copy_from_slice(&28u32.to_le_bytes());
        blob[8..12].copy_from_slice(&imports_off.to_le_bytes());
        blob[12..16].copy_from_slice(&symbols_off.to_le_bytes());
        blob[16..20].copy_from_slice(&1u32.to_le_bytes());
        blob[20..24].copy_from_slice(&DYLD_CHAINED_IMPORT.to_le_bytes());
        blob[24..28].copy_from_slice(&0u32.to_le_bytes());

        // starts_in_image at 28
        blob[28..32].copy_from_slice(&1u32.to_le_bytes()); // seg_count
        blob[32..36].copy_from_slice(&8u32.to_le_bytes()); // seg_info_offset[0]
        // starts_in_segment at 28+8=36
        blob[36..40].copy_from_slice(&24u32.to_le_bytes()); // size
        blob[40..42].copy_from_slice(&0x1000u16.to_le_bytes()); // page_size
        blob[42..44].copy_from_slice(&DYLD_CHAINED_PTR_64_OFFSET.to_le_bytes());
        blob[44..52].copy_from_slice(&0u64.to_le_bytes()); // segment_offset
        blob[52..56].copy_from_slice(&0u32.to_le_bytes()); // max_valid_pointer
        blob[56..58].copy_from_slice(&1u16.to_le_bytes()); // page_count
        blob[58..60].copy_from_slice(&0x100u16.to_le_bytes()); // page_start[0]

        blob[60..64].copy_from_slice(&0u32.to_le_bytes()); // import
        blob[64..69].copy_from_slice(b"puts\0");

        data[blob_off..blob_off + blob.len()].copy_from_slice(&blob);

        let segs = [test_seg(BASE, 0x1000)];
        let cf = parse_chained_fixups(&data, blob_off as u32, blob.len() as u32, &segs).unwrap();
        assert_eq!(cf.import_names, vec!["puts".to_string()]);
        assert_eq!(
            cf.bind_slots,
            vec![ChainedBindSlot {
                slot_va: BASE + 0x100,
                ordinal: 0,
                name: "puts".into(),
            }]
        );
    }

    #[test]
    fn chained_fixups_walk_arm64e_userland24_bind() {
        const BASE: u64 = 0x1_0000_0000;
        let mut data = vec![0u8; 0x1000];
        // arm64e bind24: bind bit 62, ordinal in low 24, next=0
        let raw: u64 = 1u64 << 62;
        data[0x200..0x208].copy_from_slice(&raw.to_le_bytes());

        let blob_off = 0x800usize;
        let mut blob = vec![0u8; 96];
        blob[4..8].copy_from_slice(&28u32.to_le_bytes());
        blob[8..12].copy_from_slice(&60u32.to_le_bytes());
        blob[12..16].copy_from_slice(&64u32.to_le_bytes());
        blob[16..20].copy_from_slice(&1u32.to_le_bytes());
        blob[20..24].copy_from_slice(&DYLD_CHAINED_IMPORT.to_le_bytes());

        blob[28..32].copy_from_slice(&1u32.to_le_bytes());
        blob[32..36].copy_from_slice(&8u32.to_le_bytes());
        blob[36..40].copy_from_slice(&24u32.to_le_bytes());
        blob[40..42].copy_from_slice(&0x1000u16.to_le_bytes());
        blob[42..44].copy_from_slice(&DYLD_CHAINED_PTR_ARM64E_USERLAND24.to_le_bytes());
        blob[44..52].copy_from_slice(&0u64.to_le_bytes());
        blob[56..58].copy_from_slice(&1u16.to_le_bytes());
        blob[58..60].copy_from_slice(&0x200u16.to_le_bytes());

        blob[60..64].copy_from_slice(&0u32.to_le_bytes());
        blob[64..70].copy_from_slice(b"write\0");
        data[blob_off..blob_off + blob.len()].copy_from_slice(&blob);

        let segs = [test_seg(BASE, 0x1000)];
        let cf = parse_chained_fixups(&data, blob_off as u32, blob.len() as u32, &segs).unwrap();
        assert_eq!(
            cf.bind_slots,
            vec![ChainedBindSlot {
                slot_va: BASE + 0x200,
                ordinal: 0,
                name: "write".into(),
            }]
        );
    }

    #[test]
    fn chained_fixups_rebase_is_not_a_bind_slot() {
        const BASE: u64 = 0x1_0000_0000;
        let mut data = vec![0u8; 0x1000];
        // bind bit clear → rebase; must not appear as an import slot.
        let raw: u64 = 0x1234;
        data[0x100..0x108].copy_from_slice(&raw.to_le_bytes());

        let blob_off = 0x800usize;
        let mut blob = vec![0u8; 96];
        blob[4..8].copy_from_slice(&28u32.to_le_bytes());
        blob[8..12].copy_from_slice(&60u32.to_le_bytes());
        blob[12..16].copy_from_slice(&64u32.to_le_bytes());
        blob[16..20].copy_from_slice(&1u32.to_le_bytes());
        blob[20..24].copy_from_slice(&DYLD_CHAINED_IMPORT.to_le_bytes());
        blob[28..32].copy_from_slice(&1u32.to_le_bytes());
        blob[32..36].copy_from_slice(&8u32.to_le_bytes());
        blob[36..40].copy_from_slice(&24u32.to_le_bytes());
        blob[40..42].copy_from_slice(&0x1000u16.to_le_bytes());
        blob[42..44].copy_from_slice(&DYLD_CHAINED_PTR_64_OFFSET.to_le_bytes());
        blob[56..58].copy_from_slice(&1u16.to_le_bytes());
        blob[58..60].copy_from_slice(&0x100u16.to_le_bytes());
        blob[60..64].copy_from_slice(&0u32.to_le_bytes());
        blob[64..69].copy_from_slice(b"puts\0");
        data[blob_off..blob_off + blob.len()].copy_from_slice(&blob);

        let cf =
            parse_chained_fixups(&data, blob_off as u32, blob.len() as u32, &[test_seg(BASE, 0x1000)])
                .unwrap();
        assert!(cf.bind_slots.is_empty());
    }

    #[test]
    fn chained_bad_page_size_refuses() {
        let mut blob = vec![0u8; 128];
        blob[4..8].copy_from_slice(&28u32.to_le_bytes());
        blob[28..32].copy_from_slice(&1u32.to_le_bytes());
        blob[32..36].copy_from_slice(&8u32.to_le_bytes());
        blob[36..40].copy_from_slice(&24u32.to_le_bytes());
        blob[40..42].copy_from_slice(&0x2000u16.to_le_bytes()); // not 0x1000/0x4000
        blob[42..44].copy_from_slice(&DYLD_CHAINED_PTR_64_OFFSET.to_le_bytes());
        blob[56..58].copy_from_slice(&1u16.to_le_bytes());
        blob[58..60].copy_from_slice(&0u16.to_le_bytes());
        assert!(parse_chained_fixups(&blob, 0, blob.len() as u32, &[]).is_err());
    }

    #[test]
    fn chained_fixups_walk_follows_next_stride() {
        const BASE: u64 = 0x1_0000_0000;
        let mut data = vec![0u8; 0x1000];
        // First bind ordinal 0, next=2 → +8 bytes (stride 4).
        let raw0: u64 = (1u64 << 63) | (2u64 << 51);
        // Second bind ordinal 1, next=0.
        let raw1: u64 = (1u64 << 63) | 1;
        data[0x100..0x108].copy_from_slice(&raw0.to_le_bytes());
        data[0x108..0x110].copy_from_slice(&raw1.to_le_bytes());

        let blob_off = 0x800usize;
        let mut blob = vec![0u8; 112];
        blob[4..8].copy_from_slice(&28u32.to_le_bytes());
        blob[8..12].copy_from_slice(&60u32.to_le_bytes());
        blob[12..16].copy_from_slice(&68u32.to_le_bytes());
        blob[16..20].copy_from_slice(&2u32.to_le_bytes());
        blob[20..24].copy_from_slice(&DYLD_CHAINED_IMPORT.to_le_bytes());
        blob[28..32].copy_from_slice(&1u32.to_le_bytes());
        blob[32..36].copy_from_slice(&8u32.to_le_bytes());
        blob[36..40].copy_from_slice(&24u32.to_le_bytes());
        blob[40..42].copy_from_slice(&0x1000u16.to_le_bytes());
        blob[42..44].copy_from_slice(&DYLD_CHAINED_PTR_64_OFFSET.to_le_bytes());
        blob[56..58].copy_from_slice(&1u16.to_le_bytes());
        blob[58..60].copy_from_slice(&0x100u16.to_le_bytes());
        blob[60..64].copy_from_slice(&0u32.to_le_bytes()); // name_offset 0 → puts
        // second import: name_offset = 5 ("puts\0" then "gets\0")
        blob[64..68].copy_from_slice(&(5u32 << 9).to_le_bytes());
        blob[68..78].copy_from_slice(b"puts\0gets\0");
        data[blob_off..blob_off + blob.len()].copy_from_slice(&blob);

        let cf =
            parse_chained_fixups(&data, blob_off as u32, blob.len() as u32, &[test_seg(BASE, 0x1000)])
                .unwrap();
        assert_eq!(
            cf.bind_slots,
            vec![
                ChainedBindSlot {
                    slot_va: BASE + 0x100,
                    ordinal: 0,
                    name: "puts".into(),
                },
                ChainedBindSlot {
                    slot_va: BASE + 0x108,
                    ordinal: 1,
                    name: "gets".into(),
                },
            ]
        );
    }

    #[test]
    fn encryption_info_parses() {
        let mut body = vec![0u8; 24];
        body[0..4].copy_from_slice(&LC_ENCRYPTION_INFO_64.to_le_bytes());
        body[4..8].copy_from_slice(&24u32.to_le_bytes());
        body[8..12].copy_from_slice(&0x1000u32.to_le_bytes());
        body[12..16].copy_from_slice(&0x2000u32.to_le_bytes());
        body[16..20].copy_from_slice(&1u32.to_le_bytes());
        let e = parse_encryption_info_64(&body).unwrap();
        assert_eq!(e.cryptid, 1);
        assert_eq!(e.cryptoff, 0x1000);
    }
}
