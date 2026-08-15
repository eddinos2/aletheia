//! ELF image parser.
//!
//! Clean-room implementation from the public System V gABI ELF
//! specification (SCO gABI, `man 5 elf`), the x86-64 psABI, and the
//! AArch64 ELF ABI. This pass handles 64-bit little-endian objects
//! (ELFCLASS64 + ELFDATA2LSB): the ELF header, program headers, section
//! headers with names resolved through the section-header string table,
//! symbols from `.symtab`/`.dynsym` with names from their linked string
//! tables, virtual-address to file-offset mapping through PT_LOAD
//! segments, the dynamic section (`PT_DYNAMIC`, or `SHT_DYNAMIC` as a
//! fallback) with `DT_NEEDED`/`DT_SONAME` resolved to strings, and the
//! `DT_RELA`/`DT_JMPREL` relocation tables — enough to join PLT GOT
//! slots against the dynamic symbol table and recover import names.
//!
//! Deliberately out of scope for now (rejected or ignored, never
//! mis-parsed): ELFCLASS32, big-endian encodings, the extended-numbering
//! escape hatches (`PN_XNUM` / `SHN_XINDEX`), a REL-format *main*
//! dynamic relocation table (`DT_REL`/`DT_RELSZ`; the x86-64 and AArch64
//! ABIs use RELA there — though a REL-format PLT table declared via
//! `DT_PLTREL` = `DT_REL` *is* handled, with implicit zero addends),
//! packed `DT_RELR` relocations, symbol versioning
//! (`DT_VERNEED`/`DT_VERSYM`), and the `DT_HASH`/`DT_GNU_HASH` lookup
//! tables.

use crate::error::{ParseError, Result};
use crate::reader::Reader;

/// `"\x7fELF"` interpreted as a little-endian u32.
pub const ELF_MAGIC: u32 = 0x464C_457F;
/// `e_ident[EI_CLASS]` value for 64-bit objects.
pub const ELFCLASS64: u8 = 2;
/// `e_ident[EI_DATA]` value for two's-complement little-endian.
pub const ELFDATA2LSB: u8 = 1;

/// Size of one ELF64 program header entry.
const PHDR64_SIZE: u64 = 56;
/// Size of one ELF64 section header entry.
const SHDR64_SIZE: u64 = 64;
/// Size of one ELF64 symbol table entry.
const SYM64_SIZE: u64 = 24;
/// Size of one ELF64 dynamic-section entry (`Elf64_Dyn`).
const DYN64_SIZE: u64 = 16;
/// Size of one ELF64 relocation record with addend (`Elf64_Rela`).
const RELA64_SIZE: u64 = 24;
/// Size of one ELF64 relocation record without addend (`Elf64_Rel`).
const REL64_SIZE: u64 = 16;
/// Cap on dynamic-section entries scanned while looking for `DT_NULL`,
/// so a crafted segment that never terminates cannot make the parser
/// build an arbitrarily large entry list. Real dynamic sections hold a
/// few dozen entries.
const MAX_DYN_ENTRIES: usize = 4096;

/// Reserved section index meaning "no section" / extended numbering markers.
const SHN_XINDEX: u16 = 0xFFFF;
/// `e_phnum` sentinel meaning the real count lives in section header 0.
const PN_XNUM: u16 = 0xFFFF;

/// Object file type (`e_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElfType {
    /// Relocatable file (`ET_REL`).
    Rel,
    /// Executable file (`ET_EXEC`).
    Exec,
    /// Shared object or position-independent executable (`ET_DYN`).
    Dyn,
    /// Core dump (`ET_CORE`).
    Core,
    /// A type value we recognize the encoding of but do not model.
    Other(u16),
}

impl From<u16> for ElfType {
    fn from(raw: u16) -> Self {
        match raw {
            1 => ElfType::Rel,
            2 => ElfType::Exec,
            3 => ElfType::Dyn,
            4 => ElfType::Core,
            other => ElfType::Other(other),
        }
    }
}

/// Target architecture (`e_machine`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Machine {
    /// `EM_X86_64` (62).
    X86_64,
    /// `EM_AARCH64` (183).
    Aarch64,
    /// `EM_RISCV` (243).
    RiscV,
    /// A machine value we recognize the encoding of but do not model.
    Other(u16),
}

impl From<u16> for Machine {
    fn from(raw: u16) -> Self {
        match raw {
            62 => Machine::X86_64,
            183 => Machine::Aarch64,
            243 => Machine::RiscV,
            other => Machine::Other(other),
        }
    }
}

/// The ELF file header (`Elf64_Ehdr`), minus the identity bytes we
/// validate and discard (class, data encoding, ident version).
#[derive(Debug, Clone)]
pub struct ElfHeader {
    /// OS/ABI identification (`e_ident[EI_OSABI]`); 0 is System V / none.
    pub osabi: u8,
    /// ABI version (`e_ident[EI_ABIVERSION]`).
    pub abi_version: u8,
    pub elf_type: ElfType,
    pub machine: Machine,
    /// Virtual address of the entry point, 0 if none.
    pub entry: u64,
    /// File offset of the program header table, 0 if none.
    pub phoff: u64,
    /// File offset of the section header table, 0 if none.
    pub shoff: u64,
    /// Processor-specific flags (`e_flags`).
    pub flags: u32,
    /// Size of this header (`e_ehsize`).
    pub ehsize: u16,
    /// Size of one program header entry.
    pub phentsize: u16,
    /// Number of program header entries.
    pub phnum: u16,
    /// Size of one section header entry.
    pub shentsize: u16,
    /// Number of section header entries.
    pub shnum: u16,
    /// Section index of the section-header string table.
    pub shstrndx: u16,
}

/// Program header segment type (`p_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
    /// `PT_LOAD` (1): loadable segment.
    Load,
    /// `PT_DYNAMIC` (2): dynamic linking information.
    Dynamic,
    /// `PT_INTERP` (3): path of the program interpreter.
    Interp,
    /// `PT_NOTE` (4): auxiliary notes.
    Note,
    /// `PT_PHDR` (6): the program header table itself.
    Phdr,
    /// `PT_TLS` (7): thread-local storage template.
    Tls,
    /// `PT_GNU_STACK` (0x6474E551): stack permission marker.
    GnuStack,
    /// `PT_GNU_RELRO` (0x6474E552): read-only-after-relocation region.
    GnuRelro,
    /// A segment type we recognize the encoding of but do not model.
    Other(u32),
}

impl From<u32> for SegmentType {
    fn from(raw: u32) -> Self {
        match raw {
            1 => SegmentType::Load,
            2 => SegmentType::Dynamic,
            3 => SegmentType::Interp,
            4 => SegmentType::Note,
            6 => SegmentType::Phdr,
            7 => SegmentType::Tls,
            0x6474_E551 => SegmentType::GnuStack,
            0x6474_E552 => SegmentType::GnuRelro,
            other => SegmentType::Other(other),
        }
    }
}

/// Segment permission flag: executable (`PF_X`).
pub const PF_X: u32 = 1;
/// Segment permission flag: writable (`PF_W`).
pub const PF_W: u32 = 2;
/// Segment permission flag: readable (`PF_R`).
pub const PF_R: u32 = 4;

/// A program header table entry (`Elf64_Phdr`).
#[derive(Debug, Clone)]
pub struct ProgramHeader {
    pub segment_type: SegmentType,
    /// Permission flags (`PF_R` / `PF_W` / `PF_X` bits).
    pub flags: u32,
    /// File offset of the segment's first byte.
    pub offset: u64,
    /// Virtual address of the segment's first byte in memory.
    pub vaddr: u64,
    /// Physical address (irrelevant on most platforms).
    pub paddr: u64,
    /// Number of bytes of the segment present in the file.
    pub filesz: u64,
    /// Number of bytes the segment occupies in memory.
    pub memsz: u64,
    /// Required alignment.
    pub align: u64,
}

impl ProgramHeader {
    pub fn is_read(&self) -> bool {
        self.flags & PF_R != 0
    }

    pub fn is_write(&self) -> bool {
        self.flags & PF_W != 0
    }

    pub fn is_execute(&self) -> bool {
        self.flags & PF_X != 0
    }

    /// Whether `vaddr` falls inside this segment's *file-backed* range.
    ///
    /// Bytes in `[filesz, memsz)` are zero-fill (e.g. `.bss`) with no file
    /// backing, so only the first `filesz` bytes are considered mappable.
    pub fn contains_vaddr(&self, vaddr: u64) -> bool {
        vaddr >= self.vaddr && vaddr - self.vaddr < self.filesz
    }
}

/// Section header type (`sh_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionType {
    /// `SHT_NULL` (0): inactive header.
    Null,
    /// `SHT_PROGBITS` (1): program-defined contents.
    Progbits,
    /// `SHT_SYMTAB` (2): full symbol table.
    Symtab,
    /// `SHT_STRTAB` (3): string table.
    Strtab,
    /// `SHT_RELA` (4): relocations with addends.
    Rela,
    /// `SHT_HASH` (5): symbol hash table.
    Hash,
    /// `SHT_DYNAMIC` (6): dynamic linking information.
    Dynamic,
    /// `SHT_NOTE` (7): notes.
    Note,
    /// `SHT_NOBITS` (8): occupies no file space (e.g. `.bss`).
    Nobits,
    /// `SHT_REL` (9): relocations without addends.
    Rel,
    /// `SHT_DYNSYM` (11): minimal dynamic-linking symbol table.
    Dynsym,
    /// `SHT_INIT_ARRAY` (14): pointers to initialization functions.
    InitArray,
    /// `SHT_FINI_ARRAY` (15): pointers to termination functions.
    FiniArray,
    /// A section type we recognize the encoding of but do not model.
    Other(u32),
}

impl From<u32> for SectionType {
    fn from(raw: u32) -> Self {
        match raw {
            0 => SectionType::Null,
            1 => SectionType::Progbits,
            2 => SectionType::Symtab,
            3 => SectionType::Strtab,
            4 => SectionType::Rela,
            5 => SectionType::Hash,
            6 => SectionType::Dynamic,
            7 => SectionType::Note,
            8 => SectionType::Nobits,
            9 => SectionType::Rel,
            11 => SectionType::Dynsym,
            14 => SectionType::InitArray,
            15 => SectionType::FiniArray,
            other => SectionType::Other(other),
        }
    }
}

/// Section flag: writable at run time (`SHF_WRITE`).
pub const SHF_WRITE: u64 = 0x1;
/// Section flag: occupies memory at run time (`SHF_ALLOC`).
pub const SHF_ALLOC: u64 = 0x2;
/// Section flag: contains executable instructions (`SHF_EXECINSTR`).
pub const SHF_EXECINSTR: u64 = 0x4;

/// A section header table entry (`Elf64_Shdr`) with its name resolved
/// through the section-header string table.
#[derive(Debug, Clone)]
pub struct SectionHeader {
    /// Section name (e.g. `.text`), resolved from `.shstrtab`; empty when
    /// the file carries no section-header string table.
    pub name: String,
    pub section_type: SectionType,
    /// `SHF_*` flag bits.
    pub flags: u64,
    /// Virtual address of the section in memory, 0 if not allocated.
    pub addr: u64,
    /// File offset of the section contents (meaningless for `SHT_NOBITS`).
    pub offset: u64,
    /// Size of the section in bytes.
    pub size: u64,
    /// Section-header-table index link; meaning depends on the type
    /// (for symbol tables: the index of the associated string table).
    pub link: u32,
    /// Extra type-dependent information.
    pub info: u32,
    /// Required alignment.
    pub addralign: u64,
    /// Entry size for sections holding fixed-size records, else 0.
    pub entsize: u64,
}

/// Symbol type, from the low nibble of `st_info`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// `STT_NOTYPE` (0).
    Notype,
    /// `STT_OBJECT` (1): data object.
    Object,
    /// `STT_FUNC` (2): function or other executable code.
    Func,
    /// `STT_SECTION` (3): associated with a section.
    Section,
    /// `STT_FILE` (4): source file name.
    File,
    /// A symbol type we recognize the encoding of but do not model.
    Other(u8),
}

impl From<u8> for SymbolKind {
    fn from(raw: u8) -> Self {
        match raw {
            0 => SymbolKind::Notype,
            1 => SymbolKind::Object,
            2 => SymbolKind::Func,
            3 => SymbolKind::Section,
            4 => SymbolKind::File,
            other => SymbolKind::Other(other),
        }
    }
}

/// Symbol binding, from the high nibble of `st_info`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolBind {
    /// `STB_LOCAL` (0).
    Local,
    /// `STB_GLOBAL` (1).
    Global,
    /// `STB_WEAK` (2).
    Weak,
    /// A binding we recognize the encoding of but do not model.
    Other(u8),
}

impl From<u8> for SymbolBind {
    fn from(raw: u8) -> Self {
        match raw {
            0 => SymbolBind::Local,
            1 => SymbolBind::Global,
            2 => SymbolBind::Weak,
            other => SymbolBind::Other(other),
        }
    }
}

/// A symbol table entry (`Elf64_Sym`) with its name resolved through the
/// table's linked string table.
#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    /// Symbol value; for most defined symbols, a virtual address.
    pub value: u64,
    /// Associated size in bytes, 0 if unknown.
    pub size: u64,
    pub kind: SymbolKind,
    pub bind: SymbolBind,
    /// Raw section index this symbol is defined relative to.
    /// 0 (`SHN_UNDEF`) means undefined; 0xFFF1 (`SHN_ABS`) absolute;
    /// 0xFFF2 (`SHN_COMMON`) common block.
    pub shndx: u16,
}

// Dynamic table tags (`d_tag`) for the entries this pass models. Values
// are from the gABI; everything else stays available raw through
// [`DynamicSection::entries`].

/// Terminates the dynamic table.
pub const DT_NULL: u64 = 0;
/// String-table offset of a needed library's name.
pub const DT_NEEDED: u64 = 1;
/// Total size in bytes of the PLT relocation table (`DT_JMPREL`).
pub const DT_PLTRELSZ: u64 = 2;
/// Virtual address of the PLT/GOT (processor-specific layout).
pub const DT_PLTGOT: u64 = 3;
/// Virtual address of the dynamic string table (`.dynstr`).
pub const DT_STRTAB: u64 = 5;
/// Virtual address of the dynamic symbol table (`.dynsym`).
pub const DT_SYMTAB: u64 = 6;
/// Virtual address of the RELA relocation table (`.rela.dyn`).
pub const DT_RELA: u64 = 7;
/// Total size in bytes of the `DT_RELA` table.
pub const DT_RELASZ: u64 = 8;
/// Size in bytes of one `DT_RELA` entry.
pub const DT_RELAENT: u64 = 9;
/// Size in bytes of the dynamic string table.
pub const DT_STRSZ: u64 = 10;
/// Size in bytes of one dynamic symbol table entry.
pub const DT_SYMENT: u64 = 11;
/// String-table offset of this shared object's own name.
pub const DT_SONAME: u64 = 14;
/// REL-format relocation table marker; as a `DT_PLTREL` *value* it
/// declares the PLT table holds `Elf64_Rel` records.
pub const DT_REL: u64 = 17;
/// Format of the PLT relocation table: `DT_RELA` or `DT_REL`.
pub const DT_PLTREL: u64 = 20;
/// Virtual address of the PLT relocation table (`.rela.plt`).
pub const DT_JMPREL: u64 = 23;

/// A raw dynamic-section entry (`Elf64_Dyn`).
///
/// `value` is the union of the spec's `d_val` and `d_ptr`: for pointer
/// tags (`DT_STRTAB`, `DT_SYMTAB`, `DT_RELA`, `DT_JMPREL`, `DT_PLTGOT`)
/// it is a *virtual address*, not a file offset, and must be mapped
/// through the `PT_LOAD` segments before dereferencing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynEntry {
    pub tag: u64,
    pub value: u64,
}

/// The parsed dynamic section (`PT_DYNAMIC` / `.dynamic`).
///
/// The commonly needed tags are broken out as typed fields (`None` when
/// the tag is absent); every entry additionally remains available raw in
/// [`DynamicSection::entries`]. All `*tab`/`rela`/`jmprel`/`pltgot`
/// fields hold virtual addresses straight from the file — map them via
/// [`ElfFile::vaddr_to_offset`] before use.
#[derive(Debug, Clone)]
pub struct DynamicSection {
    /// All entries in file order, the `DT_NULL` terminator excluded.
    pub entries: Vec<DynEntry>,
    /// Needed libraries (`DT_NEEDED`), resolved to names.
    pub needed: Vec<String>,
    /// This object's own name (`DT_SONAME`), resolved.
    pub soname: Option<String>,
    /// Virtual address of the dynamic string table (`DT_STRTAB`).
    pub strtab: Option<u64>,
    /// Size in bytes of the dynamic string table (`DT_STRSZ`).
    pub strsz: Option<u64>,
    /// Virtual address of the dynamic symbol table (`DT_SYMTAB`).
    pub symtab: Option<u64>,
    /// Size in bytes of one dynamic symbol entry (`DT_SYMENT`).
    pub syment: Option<u64>,
    /// Virtual address of the RELA relocation table (`DT_RELA`).
    pub rela: Option<u64>,
    /// Size in bytes of the RELA table (`DT_RELASZ`).
    pub relasz: Option<u64>,
    /// Size in bytes of one RELA entry (`DT_RELAENT`).
    pub relaent: Option<u64>,
    /// Virtual address of the PLT relocation table (`DT_JMPREL`).
    pub jmprel: Option<u64>,
    /// Size in bytes of the PLT relocation table (`DT_PLTRELSZ`).
    pub pltrelsz: Option<u64>,
    /// Format of the PLT relocation table (`DT_PLTREL`): the value
    /// [`DT_RELA`] or [`DT_REL`].
    pub pltrel: Option<u64>,
    /// Virtual address of the PLT/GOT (`DT_PLTGOT`).
    pub pltgot: Option<u64>,
}

// Relocation type values from the processor-specific ABI supplements
// (low 32 bits of `r_info`).

/// x86-64 psABI: absolute 64-bit, S + A.
pub const R_X86_64_64: u32 = 1;
/// x86-64 psABI: set a GOT entry to the symbol's address.
pub const R_X86_64_GLOB_DAT: u32 = 6;
/// x86-64 psABI: set a PLT GOT slot to the resolved function address.
pub const R_X86_64_JUMP_SLOT: u32 = 7;
/// x86-64 psABI: load base + addend.
pub const R_X86_64_RELATIVE: u32 = 8;
/// AArch64 ELF ABI: absolute 64-bit, S + A.
pub const R_AARCH64_ABS64: u32 = 257;
/// AArch64 ELF ABI: set a GOT entry to the symbol's address.
pub const R_AARCH64_GLOB_DAT: u32 = 1025;
/// AArch64 ELF ABI: set a PLT GOT slot to the resolved function address.
pub const R_AARCH64_JUMP_SLOT: u32 = 1026;
/// AArch64 ELF ABI: load base + addend.
pub const R_AARCH64_RELATIVE: u32 = 1027;

/// Relocation type: the call-resolution subset, machine-neutral.
///
/// The same logical operation has a different numeric encoding on each
/// machine (x86-64 psABI vs AArch64 ELF ABI), so decoding needs the
/// header's `e_machine`; the variants here abstract over it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocType {
    /// PLT lazy-binding slot: the dynamic linker stores the resolved
    /// function address into the GOT entry at `r_offset`
    /// (`R_X86_64_JUMP_SLOT` / `R_AARCH64_JUMP_SLOT`).
    JumpSlot,
    /// GOT data slot: the symbol's address is stored at `r_offset`
    /// (`R_X86_64_GLOB_DAT` / `R_AARCH64_GLOB_DAT`).
    GlobDat,
    /// Base-relative fixup: load base + addend stored at `r_offset`
    /// (`R_X86_64_RELATIVE` / `R_AARCH64_RELATIVE`).
    Relative,
    /// Absolute 64-bit: symbol + addend stored at `r_offset`
    /// (`R_X86_64_64` / `R_AARCH64_ABS64`).
    Abs64,
    /// Any other type, kept as its raw machine-specific value.
    Other(u32),
}

impl RelocType {
    /// Decode the low 32 bits of `r_info` for the given machine. Types
    /// on machines without a modeled psABI subset stay [`RelocType::Other`].
    pub fn from_raw(machine: Machine, raw: u32) -> RelocType {
        match (machine, raw) {
            (Machine::X86_64, R_X86_64_64) => RelocType::Abs64,
            (Machine::X86_64, R_X86_64_GLOB_DAT) => RelocType::GlobDat,
            (Machine::X86_64, R_X86_64_JUMP_SLOT) => RelocType::JumpSlot,
            (Machine::X86_64, R_X86_64_RELATIVE) => RelocType::Relative,
            (Machine::Aarch64, R_AARCH64_ABS64) => RelocType::Abs64,
            (Machine::Aarch64, R_AARCH64_GLOB_DAT) => RelocType::GlobDat,
            (Machine::Aarch64, R_AARCH64_JUMP_SLOT) => RelocType::JumpSlot,
            (Machine::Aarch64, R_AARCH64_RELATIVE) => RelocType::Relative,
            (_, other) => RelocType::Other(other),
        }
    }
}

/// A relocation record (`Elf64_Rela`, or `Elf64_Rel` with an implicit
/// zero addend), with `r_info` split into its symbol index and type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rela {
    /// `r_offset`: virtual address the relocation patches. For
    /// [`RelocType::JumpSlot`] this is the GOT slot the PLT jumps
    /// through.
    pub offset: u64,
    /// Dynamic-symbol-table index from the high 32 bits of `r_info`;
    /// 0 is the reserved null symbol (no symbol association).
    pub sym: u32,
    /// Relocation type from the low 32 bits of `r_info`, decoded per
    /// the header's machine.
    pub reloc_type: RelocType,
    /// `r_addend`; always 0 for REL-format records.
    pub addend: i64,
}

/// One resolved PLT import: a GOT slot joined to the symbol name the
/// dynamic linker will bind it to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PltImport {
    /// Virtual address of the GOT slot (`r_offset` of the PLT
    /// relocation). A `call [got_slot]` or a PLT stub loading this slot
    /// calls the named import.
    pub got_slot: u64,
    /// Imported symbol name from the dynamic symbol/string tables.
    pub name: String,
    /// The relocation type (normally [`RelocType::JumpSlot`]).
    pub reloc_type: RelocType,
}

/// A parsed 64-bit little-endian ELF image.
#[derive(Debug, Clone)]
pub struct ElfFile {
    pub header: ElfHeader,
    pub program_headers: Vec<ProgramHeader>,
    pub sections: Vec<SectionHeader>,
    /// Symbols from `SHT_SYMTAB` sections (the full symbol table).
    pub symtab: Vec<Symbol>,
    /// Symbols from `SHT_DYNSYM` sections (dynamic-linking symbols).
    pub dynsym: Vec<Symbol>,
    /// The dynamic section, if the file has one.
    pub dynamic: Option<DynamicSection>,
    /// Relocations from the `DT_RELA` table (conventionally `.rela.dyn`).
    pub rela_dyn: Vec<Rela>,
    /// Relocations from the `DT_JMPREL` table (conventionally
    /// `.rela.plt`); RELA- or REL-format per `DT_PLTREL`.
    pub rela_plt: Vec<Rela>,
    interp: Option<String>,
    plt_imports: Vec<PltImport>,
}

impl ElfFile {
    /// Parse an ELF image from raw file bytes.
    ///
    /// Only ELFCLASS64 + ELFDATA2LSB objects are accepted; 32-bit and
    /// big-endian files are rejected with [`ParseError::Unsupported`]
    /// rather than mis-parsed.
    pub fn parse(data: &[u8]) -> Result<ElfFile> {
        let header = parse_elf_header(data)?;
        let program_headers = parse_program_headers(data, &header)?;
        let sections = parse_section_headers(data, &header)?;

        let mut symtab = Vec::new();
        let mut dynsym = Vec::new();
        for section in &sections {
            match section.section_type {
                SectionType::Symtab => {
                    symtab.extend(parse_symbol_table(data, section, &sections)?)
                }
                SectionType::Dynsym => {
                    dynsym.extend(parse_symbol_table(data, section, &sections)?)
                }
                _ => {}
            }
        }

        let interp = parse_interpreter(data, &program_headers)?;

        let dynamic = parse_dynamic(data, &program_headers, &sections)?;
        let (rela_dyn, rela_plt, plt_imports) = match &dynamic {
            Some(dyn_info) => {
                let rela_dyn = parse_rela_dyn(data, &program_headers, header.machine, dyn_info)?;
                let rela_plt = parse_plt_relocs(data, &program_headers, header.machine, dyn_info)?;
                let plt_imports = resolve_plt_imports(data, &program_headers, dyn_info, &rela_plt)?;
                (rela_dyn, rela_plt, plt_imports)
            }
            None => (Vec::new(), Vec::new(), Vec::new()),
        };

        Ok(ElfFile {
            header,
            program_headers,
            sections,
            symtab,
            dynsym,
            dynamic,
            rela_dyn,
            rela_plt,
            interp,
            plt_imports,
        })
    }

    /// Map a virtual address to a file offset via the `PT_LOAD` segments.
    ///
    /// This is the ELF analogue of [`crate::pe::PeFile::rva_to_offset`].
    /// Only file-backed bytes are mappable: addresses inside a segment's
    /// zero-fill tail (`.bss`) have no file offset and are reported as
    /// [`ParseError::UnmappedVaddr`].
    pub fn vaddr_to_offset(&self, vaddr: u64) -> Result<usize> {
        let ph = self
            .program_headers
            .iter()
            .filter(|ph| ph.segment_type == SegmentType::Load)
            .find(|ph| ph.contains_vaddr(vaddr))
            .ok_or(ParseError::UnmappedVaddr(vaddr))?;
        let offset = ph.offset.checked_add(vaddr - ph.vaddr).ok_or_else(|| {
            ParseError::Unsupported(format!(
                "PT_LOAD file offset for vaddr {vaddr:#x} overflows a 64-bit offset"
            ))
        })?;
        Ok(offset as usize)
    }

    /// The `PT_LOAD` segment containing `vaddr`, if any.
    pub fn segment_for_vaddr(&self, vaddr: u64) -> Option<&ProgramHeader> {
        self.program_headers
            .iter()
            .filter(|ph| ph.segment_type == SegmentType::Load)
            .find(|ph| ph.contains_vaddr(vaddr))
    }

    /// The first section with the given name, if any.
    pub fn section_by_name(&self, name: &str) -> Option<&SectionHeader> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// The first symbol with the given name, searching `.symtab` first
    /// and then `.dynsym`.
    pub fn symbol_by_name(&self, name: &str) -> Option<&Symbol> {
        self.symtab
            .iter()
            .chain(self.dynsym.iter())
            .find(|s| s.name == name)
    }

    /// GOT-slot-to-import-name pairs from the PLT relocation table
    /// (`DT_JMPREL`), joined against the dynamic symbol table
    /// (`DT_SYMTAB`) and string table (`DT_STRTAB`). This is the join
    /// that lets control-flow recovery resolve `call [got]` / PLT calls
    /// to import names.
    ///
    /// Empty when the file has no dynamic section or no PLT relocation
    /// table. Relocations with symbol index 0 (the reserved null symbol)
    /// are omitted. Out-of-range symbol indices and missing
    /// `DT_SYMTAB`/`DT_STRTAB`/`DT_STRSZ` entries are typed parse-time
    /// errors, so a constructed [`ElfFile`] always carries a fully
    /// resolved list.
    pub fn plt_imports(&self) -> &[PltImport] {
        &self.plt_imports
    }

    /// Path of the program interpreter (dynamic linker) from `PT_INTERP`,
    /// e.g. `/lib64/ld-linux-x86-64.so.2`.
    pub fn interpreter(&self) -> Option<String> {
        self.interp.clone()
    }

    /// Heuristic: is this a position-independent executable?
    ///
    /// `ET_DYN` covers both shared libraries and PIEs; the presence of a
    /// `PT_INTERP` segment is the conventional tell that an `ET_DYN` file
    /// is an executable rather than a library. Caveat: statically linked
    /// PIEs have no interpreter and will be reported as non-PIE by this
    /// heuristic, and a library could in principle carry `PT_INTERP`.
    pub fn is_pie(&self) -> bool {
        self.header.elf_type == ElfType::Dyn && self.interp.is_some()
    }
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
/// `offset` lies within the file, so a crafted count can never drive an
/// oversized allocation or out-of-bounds read.
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
        // Offset 0 conventionally names the empty string.
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

fn parse_elf_header(data: &[u8]) -> Result<ElfHeader> {
    let mut r = Reader::new(data);

    let magic = r.u32()?;
    if magic != ELF_MAGIC {
        return Err(ParseError::BadSignature {
            what: "ELF",
            offset: 0,
            found: magic,
            expected: ELF_MAGIC,
        });
    }

    let class = r.u8()?;
    let encoding = r.u8()?;
    let ident_version = r.u8()?;
    let osabi = r.u8()?;
    let abi_version = r.u8()?;
    r.skip(7)?; // e_ident padding

    if class != ELFCLASS64 {
        return Err(ParseError::Unsupported(format!(
            "ELF class {class} (only ELFCLASS64 / 64-bit is supported in this pass; ELFCLASS32 is 1)"
        )));
    }
    if encoding != ELFDATA2LSB {
        return Err(ParseError::Unsupported(format!(
            "ELF data encoding {encoding} (only ELFDATA2LSB / little-endian is supported in this pass; big-endian is 2)"
        )));
    }
    if ident_version != 1 {
        return Err(ParseError::Unsupported(format!(
            "ELF identity version {ident_version}, expected 1 (EV_CURRENT)"
        )));
    }

    let elf_type = ElfType::from(r.u16()?);
    let machine = Machine::from(r.u16()?);
    let version = r.u32()?;
    if version != 1 {
        return Err(ParseError::Unsupported(format!(
            "ELF version {version}, expected 1 (EV_CURRENT)"
        )));
    }

    let entry = r.u64()?;
    let phoff = r.u64()?;
    let shoff = r.u64()?;
    let flags = r.u32()?;
    let ehsize = r.u16()?;
    let phentsize = r.u16()?;
    let phnum = r.u16()?;
    let shentsize = r.u16()?;
    let shnum = r.u16()?;
    let shstrndx = r.u16()?;

    if phnum == PN_XNUM {
        return Err(ParseError::Unsupported(
            "extended program header numbering (e_phnum == PN_XNUM) is not supported".into(),
        ));
    }
    if shnum == 0 && shoff != 0 {
        return Err(ParseError::Unsupported(
            "extended section numbering (e_shnum == 0 with e_shoff != 0) is not supported".into(),
        ));
    }
    if shstrndx == SHN_XINDEX {
        return Err(ParseError::Unsupported(
            "extended section-name-table index (e_shstrndx == SHN_XINDEX) is not supported".into(),
        ));
    }

    Ok(ElfHeader {
        osabi,
        abi_version,
        elf_type,
        machine,
        entry,
        phoff,
        shoff,
        flags,
        ehsize,
        phentsize,
        phnum,
        shentsize,
        shnum,
        shstrndx,
    })
}

fn parse_program_headers(data: &[u8], header: &ElfHeader) -> Result<Vec<ProgramHeader>> {
    if header.phnum == 0 || header.phoff == 0 {
        return Ok(Vec::new());
    }
    let entsize = header.phentsize as u64;
    if entsize < PHDR64_SIZE {
        return Err(ParseError::Unsupported(format!(
            "e_phentsize is {entsize}, smaller than an Elf64_Phdr ({PHDR64_SIZE} bytes)"
        )));
    }
    check_table(
        header.phoff,
        header.phnum as u64,
        entsize,
        data.len(),
        "program header",
    )?;

    let mut headers = Vec::with_capacity(header.phnum as usize);
    let mut r = Reader::new(data);
    for i in 0..header.phnum as u64 {
        // In-bounds by the table check above.
        r.seek((header.phoff + i * entsize) as usize)?;
        let segment_type = SegmentType::from(r.u32()?);
        let flags = r.u32()?;
        let offset = r.u64()?;
        let vaddr = r.u64()?;
        let paddr = r.u64()?;
        let filesz = r.u64()?;
        let memsz = r.u64()?;
        let align = r.u64()?;
        headers.push(ProgramHeader {
            segment_type,
            flags,
            offset,
            vaddr,
            paddr,
            filesz,
            memsz,
            align,
        });
    }
    Ok(headers)
}

/// A section header before its name has been resolved.
struct RawSection {
    name_off: u32,
    section_type: SectionType,
    flags: u64,
    addr: u64,
    offset: u64,
    size: u64,
    link: u32,
    info: u32,
    addralign: u64,
    entsize: u64,
}

fn parse_section_headers(data: &[u8], header: &ElfHeader) -> Result<Vec<SectionHeader>> {
    if header.shnum == 0 || header.shoff == 0 {
        return Ok(Vec::new());
    }
    let entsize = header.shentsize as u64;
    if entsize < SHDR64_SIZE {
        return Err(ParseError::Unsupported(format!(
            "e_shentsize is {entsize}, smaller than an Elf64_Shdr ({SHDR64_SIZE} bytes)"
        )));
    }
    check_table(
        header.shoff,
        header.shnum as u64,
        entsize,
        data.len(),
        "section header",
    )?;

    let mut raw = Vec::with_capacity(header.shnum as usize);
    let mut r = Reader::new(data);
    for i in 0..header.shnum as u64 {
        r.seek((header.shoff + i * entsize) as usize)?;
        raw.push(RawSection {
            name_off: r.u32()?,
            section_type: SectionType::from(r.u32()?),
            flags: r.u64()?,
            addr: r.u64()?,
            offset: r.u64()?,
            size: r.u64()?,
            link: r.u32()?,
            info: r.u32()?,
            addralign: r.u64()?,
            entsize: r.u64()?,
        });
    }

    // Resolve names through the section-header string table. shstrndx of
    // SHN_UNDEF (0) means the file carries no section names.
    let shstrtab = if header.shstrndx == 0 {
        None
    } else {
        let idx = header.shstrndx as usize;
        let strtab = raw.get(idx).ok_or_else(|| {
            ParseError::Unsupported(format!(
                "e_shstrndx {} is out of range for {} sections",
                header.shstrndx, header.shnum
            ))
        })?;
        check_range(
            strtab.offset,
            strtab.size,
            data.len(),
            "section-header string table",
        )?;
        Some((strtab.offset as usize, strtab.size as usize))
    };

    let mut sections = Vec::with_capacity(raw.len());
    for s in &raw {
        let name = match shstrtab {
            Some((off, size)) => strtab_string(data, off, size, s.name_off as usize, "section")?,
            None => String::new(),
        };
        sections.push(SectionHeader {
            name,
            section_type: s.section_type,
            flags: s.flags,
            addr: s.addr,
            offset: s.offset,
            size: s.size,
            link: s.link,
            info: s.info,
            addralign: s.addralign,
            entsize: s.entsize,
        });
    }
    Ok(sections)
}

fn parse_symbol_table(
    data: &[u8],
    table: &SectionHeader,
    sections: &[SectionHeader],
) -> Result<Vec<Symbol>> {
    // sh_entsize should be sizeof(Elf64_Sym); tolerate 0 as "default".
    let entsize = if table.entsize == 0 {
        SYM64_SIZE
    } else {
        table.entsize
    };
    if entsize < SYM64_SIZE {
        return Err(ParseError::Unsupported(format!(
            "symbol table '{}' has entry size {entsize}, smaller than an Elf64_Sym ({SYM64_SIZE} bytes)",
            table.name
        )));
    }
    check_range(table.offset, table.size, data.len(), "symbol table")?;

    let strtab = sections.get(table.link as usize).ok_or_else(|| {
        ParseError::Unsupported(format!(
            "symbol table '{}' links to string table index {}, out of range for {} sections",
            table.name,
            table.link,
            sections.len()
        ))
    })?;
    check_range(strtab.offset, strtab.size, data.len(), "symbol string table")?;
    let (str_off, str_size) = (strtab.offset as usize, strtab.size as usize);

    let count = table.size / entsize;
    let mut symbols = Vec::with_capacity(count as usize);
    let mut r = Reader::new(data);
    for i in 0..count {
        r.seek((table.offset + i * entsize) as usize)?;
        let name_off = r.u32()?;
        let info = r.u8()?;
        let _other = r.u8()?; // st_other: visibility, not modeled here
        let shndx = r.u16()?;
        let value = r.u64()?;
        let size = r.u64()?;
        let name = strtab_string(data, str_off, str_size, name_off as usize, "symbol")?;
        symbols.push(Symbol {
            name,
            value,
            size,
            kind: SymbolKind::from(info & 0x0F),
            bind: SymbolBind::from(info >> 4),
            shndx,
        });
    }
    Ok(symbols)
}

fn parse_interpreter(data: &[u8], program_headers: &[ProgramHeader]) -> Result<Option<String>> {
    let Some(ph) = program_headers
        .iter()
        .find(|ph| ph.segment_type == SegmentType::Interp)
    else {
        return Ok(None);
    };
    check_range(ph.offset, ph.filesz, data.len(), "PT_INTERP segment")?;
    let bytes = &data[ph.offset as usize..(ph.offset + ph.filesz) as usize];
    // The path is NUL-terminated inside the segment; tolerate a missing
    // terminator by taking the whole segment.
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    Ok(Some(String::from_utf8_lossy(&bytes[..len]).into_owned()))
}

/// Map a `size`-byte virtual-address range to a file offset through the
/// `PT_LOAD` segments, requiring the whole range to be file-backed.
///
/// This is the vaddr-trap chokepoint: dynamic-section pointers
/// (`DT_STRTAB`, `DT_SYMTAB`, `DT_RELA`, `DT_JMPREL`, ...) are virtual
/// addresses, not file offsets, so every consumer of a `d_ptr` value
/// must come through here rather than indexing the file with it.
fn map_vaddr_range(
    program_headers: &[ProgramHeader],
    file_len: usize,
    vaddr: u64,
    size: u64,
    what: &str,
) -> Result<usize> {
    let ph = program_headers
        .iter()
        .filter(|ph| ph.segment_type == SegmentType::Load)
        .find(|ph| ph.contains_vaddr(vaddr))
        .ok_or(ParseError::UnmappedVaddr(vaddr))?;
    let into_seg = vaddr - ph.vaddr;
    let offset = ph.offset.checked_add(into_seg).ok_or_else(|| {
        ParseError::Unsupported(format!(
            "{what} file offset for vaddr {vaddr:#x} overflows a 64-bit offset"
        ))
    })?;
    // contains_vaddr guarantees into_seg < filesz, so this cannot wrap.
    let available = ph.filesz - into_seg;
    if size > available {
        return Err(ParseError::UnexpectedEof {
            offset: offset.min(file_len as u64) as usize,
            needed: size as usize,
            available: available as usize,
        });
    }
    check_range(offset, size, file_len, what)?;
    Ok(offset as usize)
}

/// Locate and parse the dynamic section: `PT_DYNAMIC` when present
/// (section headers may be stripped), else an `SHT_DYNAMIC` section,
/// else `None`.
///
/// Entries are read until `DT_NULL`, the end of the segment, or
/// [`MAX_DYN_ENTRIES`], whichever comes first — a missing terminator is
/// tolerated, never looped on.
fn parse_dynamic(
    data: &[u8],
    program_headers: &[ProgramHeader],
    sections: &[SectionHeader],
) -> Result<Option<DynamicSection>> {
    let (offset, size) = if let Some(ph) = program_headers
        .iter()
        .find(|ph| ph.segment_type == SegmentType::Dynamic)
    {
        (ph.offset, ph.filesz)
    } else if let Some(sh) = sections
        .iter()
        .find(|s| s.section_type == SectionType::Dynamic)
    {
        (sh.offset, sh.size)
    } else {
        return Ok(None);
    };
    check_range(offset, size, data.len(), "dynamic section")?;

    let count = ((size / DYN64_SIZE) as usize).min(MAX_DYN_ENTRIES);
    let mut entries = Vec::with_capacity(count.min(64));
    let mut r = Reader::new(data);
    for i in 0..count {
        // In-bounds by the range check above.
        r.seek(offset as usize + i * DYN64_SIZE as usize)?;
        let tag = r.u64()?;
        let value = r.u64()?;
        if tag == DT_NULL {
            break;
        }
        entries.push(DynEntry { tag, value });
    }

    let mut dynamic = DynamicSection {
        entries,
        needed: Vec::new(),
        soname: None,
        strtab: None,
        strsz: None,
        symtab: None,
        syment: None,
        rela: None,
        relasz: None,
        relaent: None,
        jmprel: None,
        pltrelsz: None,
        pltrel: None,
        pltgot: None,
    };
    let mut needed_offs = Vec::new();
    let mut soname_off = None;
    for e in &dynamic.entries {
        match e.tag {
            DT_NEEDED => needed_offs.push(e.value),
            DT_SONAME => soname_off = Some(e.value),
            DT_STRTAB => dynamic.strtab = Some(e.value),
            DT_STRSZ => dynamic.strsz = Some(e.value),
            DT_SYMTAB => dynamic.symtab = Some(e.value),
            DT_SYMENT => dynamic.syment = Some(e.value),
            DT_RELA => dynamic.rela = Some(e.value),
            DT_RELASZ => dynamic.relasz = Some(e.value),
            DT_RELAENT => dynamic.relaent = Some(e.value),
            DT_JMPREL => dynamic.jmprel = Some(e.value),
            DT_PLTRELSZ => dynamic.pltrelsz = Some(e.value),
            DT_PLTREL => dynamic.pltrel = Some(e.value),
            DT_PLTGOT => dynamic.pltgot = Some(e.value),
            _ => {}
        }
    }

    // DT_NEEDED / DT_SONAME hold offsets into the dynamic string table;
    // the table itself may be declared before or after them, so resolve
    // only once the whole table has been scanned.
    if !needed_offs.is_empty() || soname_off.is_some() {
        let (str_off, str_size) = dynamic_strtab(data, program_headers, &dynamic)?;
        for off in needed_offs {
            dynamic
                .needed
                .push(strtab_string(data, str_off, str_size, off as usize, "DT_NEEDED library")?);
        }
        if let Some(off) = soname_off {
            dynamic.soname = Some(strtab_string(data, str_off, str_size, off as usize, "DT_SONAME")?);
        }
    }

    Ok(Some(dynamic))
}

/// Resolve `DT_STRTAB`/`DT_STRSZ` to a validated (file offset, size)
/// pair, mapping the virtual address through the `PT_LOAD` segments.
fn dynamic_strtab(
    data: &[u8],
    program_headers: &[ProgramHeader],
    dynamic: &DynamicSection,
) -> Result<(usize, usize)> {
    let vaddr = dynamic.strtab.ok_or_else(|| {
        ParseError::Unsupported(
            "dynamic section references strings but carries no DT_STRTAB".into(),
        )
    })?;
    let size = dynamic.strsz.ok_or_else(|| {
        ParseError::Unsupported("dynamic section has DT_STRTAB but no DT_STRSZ to bound it".into())
    })?;
    let off = map_vaddr_range(program_headers, data.len(), vaddr, size, "dynamic string table")?;
    Ok((off, size as usize))
}

/// Parse the main `DT_RELA` relocation table, if declared.
fn parse_rela_dyn(
    data: &[u8],
    program_headers: &[ProgramHeader],
    machine: Machine,
    dynamic: &DynamicSection,
) -> Result<Vec<Rela>> {
    let Some(vaddr) = dynamic.rela else {
        return Ok(Vec::new());
    };
    let size = dynamic.relasz.ok_or_else(|| {
        ParseError::Unsupported("dynamic section has DT_RELA but no DT_RELASZ to bound it".into())
    })?;
    let entsize = dynamic.relaent.unwrap_or(0);
    parse_reloc_table(
        data,
        program_headers,
        machine,
        vaddr,
        size,
        entsize,
        true,
        "RELA relocation table",
    )
}

/// Parse the PLT relocation table (`DT_JMPREL`), if declared. Its format
/// is RELA or REL according to `DT_PLTREL`; both are accepted (REL
/// records get an implicit zero addend).
fn parse_plt_relocs(
    data: &[u8],
    program_headers: &[ProgramHeader],
    machine: Machine,
    dynamic: &DynamicSection,
) -> Result<Vec<Rela>> {
    let Some(vaddr) = dynamic.jmprel else {
        return Ok(Vec::new());
    };
    let size = dynamic.pltrelsz.ok_or_else(|| {
        ParseError::Unsupported(
            "dynamic section has DT_JMPREL but no DT_PLTRELSZ to bound it".into(),
        )
    })?;
    let with_addend = match dynamic.pltrel {
        Some(DT_RELA) => true,
        Some(DT_REL) => false,
        Some(other) => {
            return Err(ParseError::Unsupported(format!(
                "DT_PLTREL is {other}, expected DT_RELA (7) or DT_REL (17)"
            )));
        }
        None => {
            return Err(ParseError::Unsupported(
                "dynamic section has DT_JMPREL but no DT_PLTREL declaring its format".into(),
            ));
        }
    };
    parse_reloc_table(
        data,
        program_headers,
        machine,
        vaddr,
        size,
        0,
        with_addend,
        "PLT relocation table",
    )
}

/// Parse a relocation table at a virtual address: `Elf64_Rela` records
/// when `with_addend`, `Elf64_Rel` otherwise.
///
/// The declared size is validated against the file-backed extent of the
/// containing `PT_LOAD` segment before anything is read or allocated, so
/// a crafted `DT_RELASZ`/`DT_PLTRELSZ` can never drive an oversized
/// allocation: the entry count is bounded by the file size.
#[allow(clippy::too_many_arguments)]
fn parse_reloc_table(
    data: &[u8],
    program_headers: &[ProgramHeader],
    machine: Machine,
    vaddr: u64,
    size: u64,
    entsize: u64,
    with_addend: bool,
    what: &str,
) -> Result<Vec<Rela>> {
    let min = if with_addend { RELA64_SIZE } else { REL64_SIZE };
    // Tolerate a missing/zero entry size as "default", as for symbols.
    let entsize = if entsize == 0 { min } else { entsize };
    if entsize < min {
        return Err(ParseError::Unsupported(format!(
            "{what} entry size {entsize} is smaller than a relocation record ({min} bytes)"
        )));
    }
    if size == 0 {
        return Ok(Vec::new());
    }
    let offset = map_vaddr_range(program_headers, data.len(), vaddr, size, what)?;

    let count = size / entsize;
    let mut relocs = Vec::with_capacity(count as usize);
    let mut r = Reader::new(data);
    for i in 0..count {
        r.seek(offset + (i * entsize) as usize)?;
        let r_offset = r.u64()?;
        let r_info = r.u64()?;
        let addend = if with_addend { r.u64()? as i64 } else { 0 };
        relocs.push(Rela {
            offset: r_offset,
            sym: (r_info >> 32) as u32,
            reloc_type: RelocType::from_raw(machine, r_info as u32),
            addend,
        });
    }
    Ok(relocs)
}

/// Join the PLT relocations against the dynamic symbol table
/// (`DT_SYMTAB`) and string table (`DT_STRTAB`), yielding one
/// [`PltImport`] per symbol-bearing relocation.
///
/// Relocations with symbol index 0 are skipped. The dynamic section
/// carries no symbol *count*, so each index is instead bounds-checked
/// against the file-backed extent of the `PT_LOAD` segment holding the
/// symbol table: indices whose record would fall outside it are typed
/// errors.
fn resolve_plt_imports(
    data: &[u8],
    program_headers: &[ProgramHeader],
    dynamic: &DynamicSection,
    rela_plt: &[Rela],
) -> Result<Vec<PltImport>> {
    if rela_plt.is_empty() {
        return Ok(Vec::new());
    }
    let symtab = dynamic.symtab.ok_or_else(|| {
        ParseError::Unsupported(
            "PLT relocations reference symbols but the dynamic section has no DT_SYMTAB".into(),
        )
    })?;
    let syment = dynamic.syment.unwrap_or(SYM64_SIZE);
    if syment < SYM64_SIZE {
        return Err(ParseError::Unsupported(format!(
            "DT_SYMENT is {syment}, smaller than an Elf64_Sym ({SYM64_SIZE} bytes)"
        )));
    }
    let (str_off, str_size) = dynamic_strtab(data, program_headers, dynamic)?;

    let mut imports = Vec::with_capacity(rela_plt.len());
    let mut r = Reader::new(data);
    for reloc in rela_plt {
        if reloc.sym == 0 {
            // Index 0 is the reserved null symbol: nothing to import.
            continue;
        }
        let sym_vaddr = (reloc.sym as u64)
            .checked_mul(syment)
            .and_then(|delta| symtab.checked_add(delta))
            .ok_or_else(|| {
                ParseError::Unsupported(format!(
                    "dynamic symbol index {} overflows the symbol table address",
                    reloc.sym
                ))
            })?;
        let sym_off = map_vaddr_range(
            program_headers,
            data.len(),
            sym_vaddr,
            SYM64_SIZE,
            "dynamic symbol",
        )?;
        r.seek(sym_off)?;
        let name_off = r.u32()?;
        let name = strtab_string(data, str_off, str_size, name_off as usize, "dynamic symbol")?;
        imports.push(PltImport {
            got_slot: reloc.offset,
            name,
            reloc_type: reloc.reloc_type,
        });
    }
    Ok(imports)
}

/// DWARF exception-handling pointer encodings, low nibble = format.
const DW_EH_PE_OMIT: u8 = 0xFF;
const DW_EH_PE_ABSPTR: u8 = 0x00;
const DW_EH_PE_UDATA2: u8 = 0x02;
const DW_EH_PE_UDATA4: u8 = 0x03;
const DW_EH_PE_UDATA8: u8 = 0x04;
const DW_EH_PE_SDATA2: u8 = 0x0A;
const DW_EH_PE_SDATA4: u8 = 0x0B;
const DW_EH_PE_SDATA8: u8 = 0x0C;
/// Application (high nibble) = value is relative to its own location.
const DW_EH_PE_PCREL: u8 = 0x10;

/// Cap on `.eh_frame` records walked, bounding work on a crafted section.
const MAX_EH_FRAME_RECORDS: usize = 1 << 21;

impl ElfFile {
    /// Function-start VAs recovered from the `.eh_frame` call-frame
    /// records, sorted and deduplicated.
    ///
    /// Each FDE (frame description entry) names a function through its
    /// `initial_location` field, decoded per the pointer encoding its
    /// CIE advertises via the `augmentation` `R` byte. This is a
    /// compiler-emitted source of function starts that survives symbol
    /// stripping, so it is a high-precision discovery input on Linux
    /// binaries.
    ///
    /// Handled encodings: `DW_EH_PE_absptr` (native 8-byte absolute),
    /// `udata4`/`sdata4`/`udata8`/`sdata8` (absolute), and any of these
    /// with the `pcrel` application (the common `pcrel|sdata4`, 0x1B,
    /// case in position-independent executables). FDEs whose encoding
    /// uses `uleb128`/`sleb128` or a non-pcrel relative base
    /// (`datarel`, `textrel`, ...) are skipped rather than guessed.
    /// Never panics: a malformed record ends the walk.
    pub fn eh_frame_function_starts(&self, data: &[u8]) -> Vec<u64> {
        let Some(sec) = self.sections.iter().find(|s| s.name == ".eh_frame") else {
            return Vec::new();
        };
        if sec.size == 0 {
            return Vec::new();
        }
        let sec_off = sec.offset as usize;
        let sec_end = match sec_off.checked_add(sec.size as usize) {
            Some(e) if e <= data.len() => e,
            _ => return Vec::new(),
        };
        let sec_bytes = &data[sec_off..sec_end];

        // FDE encoding advertised by each CIE, keyed by the CIE's offset
        // within the section (the value an FDE's back-pointer resolves
        // to). Absent entries default to DW_EH_PE_absptr.
        let mut cie_enc: std::collections::BTreeMap<usize, u8> = std::collections::BTreeMap::new();
        let mut out = Vec::new();
        let mut pos = 0usize;

        for _ in 0..MAX_EH_FRAME_RECORDS {
            if pos + 4 > sec_bytes.len() {
                break;
            }
            let len32 = u32::from_le_bytes(sec_bytes[pos..pos + 4].try_into().unwrap());
            // A zero length terminates the table; the 64-bit extended
            // form (0xffffffff) is rare in .eh_frame and not handled.
            if len32 == 0 || len32 == 0xFFFF_FFFF {
                break;
            }
            let content = pos + 4;
            let Some(next) = content.checked_add(len32 as usize) else {
                break;
            };
            if next > sec_bytes.len() {
                break;
            }
            // The CIE_pointer / CIE_id field sits at the start of content.
            if content + 4 > sec_bytes.len() {
                break;
            }
            let id = u32::from_le_bytes(sec_bytes[content..content + 4].try_into().unwrap());
            if id == 0 {
                // CIE: record its advertised FDE encoding.
                let enc = parse_cie_fde_encoding(&sec_bytes[content + 4..next]);
                cie_enc.insert(pos, enc.unwrap_or(DW_EH_PE_ABSPTR));
            } else {
                // FDE: the CIE it points to sits `id` bytes before this
                // record's CIE-pointer field.
                if let Some(cie_off) = content.checked_sub(id as usize) {
                    let enc = cie_enc
                        .get(&cie_off)
                        .copied()
                        .unwrap_or(DW_EH_PE_ABSPTR);
                    // initial_location is the field just after the
                    // CIE-pointer, at section offset `content + 4`.
                    let field_off = content + 4;
                    if let Some(va) = decode_eh_pointer(sec_bytes, field_off, enc, sec.addr)
                        && va != 0
                    {
                        out.push(va);
                    }
                }
            }
            pos = next;
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Byte width of an encoded pointer's value, or `None` for the variable
/// (LEB128) or omitted forms this reader does not decode.
fn eh_pointer_size(enc: u8) -> Option<usize> {
    match enc & 0x0F {
        DW_EH_PE_ABSPTR => Some(8),
        DW_EH_PE_UDATA2 | DW_EH_PE_SDATA2 => Some(2),
        DW_EH_PE_UDATA4 | DW_EH_PE_SDATA4 => Some(4),
        DW_EH_PE_UDATA8 | DW_EH_PE_SDATA8 => Some(8),
        _ => None,
    }
}

/// Decode an encoded pointer at `field_off` in `sec_bytes`, where
/// `sec_addr` is the section's virtual address (so the field's own VA is
/// `sec_addr + field_off`). Returns the absolute VA, or `None` if the
/// encoding is unsupported or the read runs off the section.
fn decode_eh_pointer(sec_bytes: &[u8], field_off: usize, enc: u8, sec_addr: u64) -> Option<u64> {
    if enc == DW_EH_PE_OMIT {
        return None;
    }
    let size = eh_pointer_size(enc)?;
    let end = field_off.checked_add(size)?;
    if end > sec_bytes.len() {
        return None;
    }
    let raw = &sec_bytes[field_off..end];
    // Sign-extend the signed formats; zero-extend the unsigned ones.
    let value: u64 = match enc & 0x0F {
        DW_EH_PE_SDATA2 => i16::from_le_bytes(raw.try_into().ok()?) as i64 as u64,
        DW_EH_PE_SDATA4 => i32::from_le_bytes(raw.try_into().ok()?) as i64 as u64,
        DW_EH_PE_SDATA8 => i64::from_le_bytes(raw.try_into().ok()?) as u64,
        DW_EH_PE_UDATA2 => u16::from_le_bytes(raw.try_into().ok()?) as u64,
        DW_EH_PE_UDATA4 => u32::from_le_bytes(raw.try_into().ok()?) as u64,
        DW_EH_PE_UDATA8 | DW_EH_PE_ABSPTR => u64::from_le_bytes(raw.try_into().ok()?),
        _ => return None,
    };
    // Only absolute and pcrel applications are supported; other bases
    // (datarel/textrel/funcrel) need context this reader doesn't track.
    match enc & 0x70 {
        0x00 => Some(value),
        DW_EH_PE_PCREL => {
            let field_va = sec_addr.wrapping_add(field_off as u64);
            Some(field_va.wrapping_add(value))
        }
        _ => None,
    }
}

/// Extract the FDE pointer encoding (`R`) a CIE advertises, given the CIE
/// body starting just after its 4-byte CIE-id. Returns `None` (caller
/// defaults to absptr) when the CIE has no augmentation data or no `R`.
fn parse_cie_fde_encoding(body: &[u8]) -> Option<u8> {
    let mut r = Reader::new(body);
    let version = r.u8().ok()?;
    let aug = r.cstr().ok()?;
    // code_alignment_factor (ULEB), data_alignment_factor (SLEB).
    read_uleb(&mut r)?;
    read_sleb(&mut r)?;
    // return_address_register: ULEB in v3+, a single byte in v1.
    if version >= 3 {
        read_uleb(&mut r)?;
    } else {
        r.u8().ok()?;
    }
    if !aug.starts_with('z') {
        return None; // no augmentation data block, hence no R byte
    }
    let aug_len = read_uleb(&mut r)? as usize;
    let aug_data = r.take(aug_len).ok()?;
    // Walk the augmentation chars after 'z', consuming aug_data in step.
    let mut ar = Reader::new(aug_data);
    for c in aug.chars().skip(1) {
        match c {
            'R' => return ar.u8().ok(),
            'L' => {
                ar.u8().ok()?; // LSDA encoding byte
            }
            'P' => {
                let penc = ar.u8().ok()?;
                let psize = eh_pointer_size(penc)?;
                ar.skip(psize).ok()?;
            }
            'S' | 'B' | 'G' => {}
            _ => break,
        }
    }
    None
}

/// Read an unsigned LEB128 from a [`Reader`], or `None` on EOF/overflow.
fn read_uleb(r: &mut Reader) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    loop {
        let byte = r.u8().ok()?;
        if shift < 64 {
            result |= ((byte & 0x7F) as u64) << shift;
        } else if byte & 0x7F != 0 {
            return None;
        }
        shift += 7;
        if byte & 0x80 == 0 {
            return Some(result);
        }
        if shift >= 128 {
            return None;
        }
    }
}

/// Read a signed LEB128 from a [`Reader`], or `None` on EOF/overflow. The
/// value itself is unused (the CIE alignment factors are skipped), but it
/// must be consumed to keep the cursor aligned.
fn read_sleb(r: &mut Reader) -> Option<i64> {
    let mut result: i64 = 0;
    let mut shift = 0u32;
    loop {
        let byte = r.u8().ok()?;
        if shift < 64 {
            result |= ((byte & 0x7F) as i64) << shift;
        }
        shift += 7;
        if byte & 0x80 == 0 {
            if shift < 64 && byte & 0x40 != 0 {
                result |= -1i64 << shift;
            }
            return Some(result);
        }
        if shift >= 128 {
            return None;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const IMG_SIZE: usize = 0x440;
    const PHOFF: usize = 0x40;
    const INTERP_OFF: usize = 0xB0;
    const TEXT_OFF: usize = 0x100;
    const TEXT_VADDR: u64 = 0x40_1000;
    const STRTAB_OFF: usize = 0x200;
    const SYMTAB_OFF: usize = 0x220;
    const SHSTRTAB_OFF: usize = 0x280;
    const SHOFF: usize = 0x300;
    const INTERP: &[u8] = b"/lib64/ld-linux-x86-64.so.2\0";

    fn put(img: &mut [u8], off: usize, bytes: &[u8]) {
        img[off..off + bytes.len()].copy_from_slice(bytes);
    }

    fn put16(img: &mut [u8], off: usize, v: u16) {
        put(img, off, &v.to_le_bytes());
    }

    fn put32(img: &mut [u8], off: usize, v: u32) {
        put(img, off, &v.to_le_bytes());
    }

    fn put64(img: &mut [u8], off: usize, v: u64) {
        put(img, off, &v.to_le_bytes());
    }

    #[allow(clippy::too_many_arguments)]
    fn write_phdr(
        img: &mut [u8],
        idx: usize,
        ptype: u32,
        flags: u32,
        offset: u64,
        vaddr: u64,
        filesz: u64,
        memsz: u64,
        align: u64,
    ) {
        let base = PHOFF + idx * 56;
        put32(img, base, ptype);
        put32(img, base + 4, flags);
        put64(img, base + 8, offset);
        put64(img, base + 16, vaddr);
        put64(img, base + 24, vaddr); // paddr
        put64(img, base + 32, filesz);
        put64(img, base + 40, memsz);
        put64(img, base + 48, align);
    }

    #[allow(clippy::too_many_arguments)]
    fn write_shdr_at(
        img: &mut [u8],
        table_off: usize,
        idx: usize,
        name_off: u32,
        shtype: u32,
        flags: u64,
        addr: u64,
        offset: u64,
        size: u64,
        link: u32,
        info: u32,
        entsize: u64,
    ) {
        let base = table_off + idx * 64;
        put32(img, base, name_off);
        put32(img, base + 4, shtype);
        put64(img, base + 8, flags);
        put64(img, base + 16, addr);
        put64(img, base + 24, offset);
        put64(img, base + 32, size);
        put32(img, base + 40, link);
        put32(img, base + 44, info);
        put64(img, base + 48, 16); // addralign
        put64(img, base + 56, entsize);
    }

    #[allow(clippy::too_many_arguments)]
    fn write_shdr(
        img: &mut [u8],
        idx: usize,
        name_off: u32,
        shtype: u32,
        flags: u64,
        addr: u64,
        offset: u64,
        size: u64,
        link: u32,
        info: u32,
        entsize: u64,
    ) {
        write_shdr_at(
            img, SHOFF, idx, name_off, shtype, flags, addr, offset, size, link, info, entsize,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn write_sym_at(
        img: &mut [u8],
        table_off: usize,
        idx: usize,
        name_off: u32,
        info: u8,
        shndx: u16,
        value: u64,
        size: u64,
    ) {
        let base = table_off + idx * 24;
        put32(img, base, name_off);
        img[base + 4] = info;
        img[base + 5] = 0; // st_other
        put16(img, base + 6, shndx);
        put64(img, base + 8, value);
        put64(img, base + 16, size);
    }

    fn write_sym(
        img: &mut [u8],
        idx: usize,
        name_off: u32,
        info: u8,
        shndx: u16,
        value: u64,
        size: u64,
    ) {
        write_sym_at(img, SYMTAB_OFF, idx, name_off, info, shndx, value, size);
    }

    fn write_dyn(img: &mut [u8], table_off: usize, idx: usize, tag: u64, value: u64) {
        let base = table_off + idx * 16;
        put64(img, base, tag);
        put64(img, base + 8, value);
    }

    fn write_rela(
        img: &mut [u8],
        table_off: usize,
        idx: usize,
        r_offset: u64,
        sym: u32,
        rtype: u32,
        addend: i64,
    ) {
        let base = table_off + idx * 24;
        put64(img, base, r_offset);
        put64(img, base + 8, ((sym as u64) << 32) | rtype as u64);
        put64(img, base + 16, addend as u64);
    }

    fn write_rel(img: &mut [u8], table_off: usize, idx: usize, r_offset: u64, sym: u32, rtype: u32) {
        let base = table_off + idx * 16;
        put64(img, base, r_offset);
        put64(img, base + 8, ((sym as u64) << 32) | rtype as u64);
    }

    /// Build a minimal but well-formed ELF64 (little-endian, x86-64,
    /// ET_DYN with PT_INTERP, i.e. a PIE): two program headers (PT_LOAD,
    /// PT_INTERP) and five sections (null, .text, .symtab, .strtab,
    /// .shstrtab) with two named symbols. Shared with the trait-layer
    /// tests in [`crate::model`].
    pub(crate) fn synthetic_elf64() -> Vec<u8> {
        let mut img = vec![0u8; IMG_SIZE];

        // ELF header.
        put(&mut img, 0, b"\x7fELF");
        img[4] = ELFCLASS64;
        img[5] = ELFDATA2LSB;
        img[6] = 1; // EI_VERSION
        img[7] = 0; // EI_OSABI: System V
        put16(&mut img, 16, 3); // e_type: ET_DYN
        put16(&mut img, 18, 62); // e_machine: EM_X86_64
        put32(&mut img, 20, 1); // e_version
        put64(&mut img, 24, TEXT_VADDR); // e_entry
        put64(&mut img, 32, PHOFF as u64); // e_phoff
        put64(&mut img, 40, SHOFF as u64); // e_shoff
        put16(&mut img, 52, 64); // e_ehsize
        put16(&mut img, 54, 56); // e_phentsize
        put16(&mut img, 56, 2); // e_phnum
        put16(&mut img, 58, 64); // e_shentsize
        put16(&mut img, 60, 5); // e_shnum
        put16(&mut img, 62, 4); // e_shstrndx: .shstrtab

        // Program headers: an R+X PT_LOAD and the PT_INTERP.
        write_phdr(
            &mut img,
            0,
            1, // PT_LOAD
            PF_R | PF_X,
            TEXT_OFF as u64,
            TEXT_VADDR,
            0x80,
            0x80,
            0x1000,
        );
        write_phdr(
            &mut img,
            1,
            3, // PT_INTERP
            PF_R,
            INTERP_OFF as u64,
            0x40_00B0,
            INTERP.len() as u64,
            INTERP.len() as u64,
            1,
        );
        put(&mut img, INTERP_OFF, INTERP);

        // String tables.
        put(&mut img, STRTAB_OFF, b"\0main\0helper\0"); // 13 bytes
        put(&mut img, SHSTRTAB_OFF, b"\0.text\0.symtab\0.strtab\0.shstrtab\0"); // 33 bytes

        // Symbols: null, then main (global func), then helper (local func).
        write_sym(&mut img, 1, 1, 0x12, 1, TEXT_VADDR, 0x20);
        write_sym(&mut img, 2, 6, 0x02, 1, TEXT_VADDR + 0x20, 0x10);

        // Sections: null, .text, .symtab, .strtab, .shstrtab.
        write_shdr(&mut img, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        write_shdr(
            &mut img,
            1,
            1, // ".text"
            1, // SHT_PROGBITS
            SHF_ALLOC | SHF_EXECINSTR,
            TEXT_VADDR,
            TEXT_OFF as u64,
            0x80,
            0,
            0,
            0,
        );
        write_shdr(
            &mut img,
            2,
            7, // ".symtab"
            2, // SHT_SYMTAB
            0,
            0,
            SYMTAB_OFF as u64,
            3 * 24,
            3, // link -> .strtab
            2, // info: first non-local symbol index
            24,
        );
        write_shdr(&mut img, 3, 15, 3, 0, 0, STRTAB_OFF as u64, 13, 0, 0, 0);
        write_shdr(&mut img, 4, 23, 3, 0, 0, SHSTRTAB_OFF as u64, 33, 0, 0, 0);

        img
    }

    // Layout of the synthetic *dynamic* executable image. One PT_LOAD
    // maps the whole file at DYN_BASE, so vaddr == DYN_BASE + offset.
    const DYN_BASE: u64 = 0x40_0000;
    const DYN_IMG_SIZE: usize = 0x370;
    const DYN_STR_OFF: usize = 0xC0;
    const DYN_SYM_OFF: usize = 0x100;
    const DYN_JMPREL_OFF: usize = 0x160;
    const DYN_RELA_OFF: usize = 0x1B0;
    const DYN_DYN_OFF: usize = 0x200;
    const DYN_SHOFF: usize = 0x2F0;
    // Name offsets: 1 "libc.so.6", 11 "libm.so.6", 21 "read",
    // 26 "write", 32 "close", 38 "mylib.so".
    const DYNSTR: &[u8] = b"\0libc.so.6\0libm.so.6\0read\0write\0close\0mylib.so\0";
    /// GOT slots patched by the three JUMP_SLOT relocations.
    const GOT_SLOTS: [u64; 3] = [DYN_BASE + 0x3018, DYN_BASE + 0x3020, DYN_BASE + 0x3028];

    // Indices of the tags in the synthetic dynamic table, for patching.
    const DYN_IDX_NEEDED0: usize = 0;
    const DYN_IDX_STRTAB: usize = 3;
    const DYN_IDX_RELASZ: usize = 8;
    const DYN_IDX_PLTRELSZ: usize = 11;
    const DYN_IDX_PLTREL: usize = 12;

    /// File offset of dynamic entry `idx`'s d_tag (d_val/d_ptr is at +8).
    fn dyn_tag_off(idx: usize) -> usize {
        DYN_DYN_OFF + idx * 16
    }

    /// Build a synthetic dynamically linked ELF64 (x86-64, ET_DYN):
    /// a PT_LOAD covering the whole file plus PT_DYNAMIC; .dynstr; a
    /// four-entry .dynsym (null, read, write, close); a three-entry RELA
    /// JMPREL table of JUMP_SLOT relocations; a two-entry DT_RELA table
    /// (RELATIVE + GLOB_DAT); and an unnamed section table (null +
    /// SHT_DYNAMIC) so the section-based fallback path is exercisable.
    /// Shared with the trait-layer tests in [`crate::model`].
    pub(crate) fn synthetic_dynamic_elf64() -> Vec<u8> {
        let mut img = vec![0u8; DYN_IMG_SIZE];

        // ELF header.
        put(&mut img, 0, b"\x7fELF");
        img[4] = ELFCLASS64;
        img[5] = ELFDATA2LSB;
        img[6] = 1; // EI_VERSION
        put16(&mut img, 16, 3); // e_type: ET_DYN
        put16(&mut img, 18, 62); // e_machine: EM_X86_64
        put32(&mut img, 20, 1); // e_version
        put64(&mut img, 32, PHOFF as u64); // e_phoff
        put64(&mut img, 40, DYN_SHOFF as u64); // e_shoff
        put16(&mut img, 52, 64); // e_ehsize
        put16(&mut img, 54, 56); // e_phentsize
        put16(&mut img, 56, 2); // e_phnum
        put16(&mut img, 58, 64); // e_shentsize
        put16(&mut img, 60, 2); // e_shnum
        put16(&mut img, 62, 0); // e_shstrndx: no section names

        // Program headers: PT_LOAD over the whole file, then PT_DYNAMIC.
        write_phdr(
            &mut img,
            0,
            1, // PT_LOAD
            PF_R | PF_X,
            0,
            DYN_BASE,
            DYN_IMG_SIZE as u64,
            DYN_IMG_SIZE as u64,
            0x1000,
        );
        write_phdr(
            &mut img,
            1,
            2, // PT_DYNAMIC
            PF_R,
            DYN_DYN_OFF as u64,
            DYN_BASE + DYN_DYN_OFF as u64,
            0xF0,
            0xF0,
            8,
        );

        put(&mut img, DYN_STR_OFF, DYNSTR);

        // .dynsym: null, then read/write/close as undefined global funcs.
        write_sym_at(&mut img, DYN_SYM_OFF, 1, 21, 0x12, 0, 0, 0);
        write_sym_at(&mut img, DYN_SYM_OFF, 2, 26, 0x12, 0, 0, 0);
        write_sym_at(&mut img, DYN_SYM_OFF, 3, 32, 0x12, 0, 0, 0);

        // JMPREL: three R_X86_64_JUMP_SLOT relocs against read/write/close.
        for (i, &slot) in GOT_SLOTS.iter().enumerate() {
            write_rela(
                &mut img,
                DYN_JMPREL_OFF,
                i,
                slot,
                (i + 1) as u32,
                R_X86_64_JUMP_SLOT,
                0,
            );
        }

        // RELA: a base-relative fixup and a GOT data slot for "read".
        write_rela(&mut img, DYN_RELA_OFF, 0, DYN_BASE + 0x2E00, 0, R_X86_64_RELATIVE, 0x1234);
        write_rela(&mut img, DYN_RELA_OFF, 1, DYN_BASE + 0x3008, 1, R_X86_64_GLOB_DAT, 0);

        // Dynamic table (indices must match the DYN_IDX_* constants).
        let entries: &[(u64, u64)] = &[
            (DT_NEEDED, 1),                                // "libc.so.6"
            (DT_NEEDED, 11),                               // "libm.so.6"
            (DT_SONAME, 38),                               // "mylib.so"
            (DT_STRTAB, DYN_BASE + DYN_STR_OFF as u64),
            (DT_STRSZ, DYNSTR.len() as u64),
            (DT_SYMTAB, DYN_BASE + DYN_SYM_OFF as u64),
            (DT_SYMENT, 24),
            (DT_RELA, DYN_BASE + DYN_RELA_OFF as u64),
            (DT_RELASZ, 2 * 24),
            (DT_RELAENT, 24),
            (DT_JMPREL, DYN_BASE + DYN_JMPREL_OFF as u64),
            (DT_PLTRELSZ, 3 * 24),
            (DT_PLTREL, DT_RELA),
            (DT_PLTGOT, DYN_BASE + 0x3000),
            (DT_NULL, 0),
        ];
        for (i, &(tag, value)) in entries.iter().enumerate() {
            write_dyn(&mut img, DYN_DYN_OFF, i, tag, value);
        }

        // Sections: null + an unnamed SHT_DYNAMIC, for the fallback path.
        write_shdr_at(&mut img, DYN_SHOFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        write_shdr_at(
            &mut img,
            DYN_SHOFF,
            1,
            0,
            6, // SHT_DYNAMIC
            SHF_ALLOC,
            DYN_BASE + DYN_DYN_OFF as u64,
            DYN_DYN_OFF as u64,
            0xF0,
            0,
            0,
            16,
        );

        img
    }

    /// Build a header + PT_LOAD + PT_DYNAMIC image whose dynamic segment
    /// holds `count` non-null entries and never a DT_NULL terminator.
    fn synthetic_unterminated_dynamic(count: usize) -> Vec<u8> {
        let dyn_off = 0xC0usize;
        let size = dyn_off + count * 16;
        let mut img = vec![0u8; size];

        put(&mut img, 0, b"\x7fELF");
        img[4] = ELFCLASS64;
        img[5] = ELFDATA2LSB;
        img[6] = 1;
        put16(&mut img, 16, 3); // ET_DYN
        put16(&mut img, 18, 62); // EM_X86_64
        put32(&mut img, 20, 1);
        put64(&mut img, 32, PHOFF as u64);
        put16(&mut img, 52, 64);
        put16(&mut img, 54, 56);
        put16(&mut img, 56, 2); // e_phnum; e_shnum stays 0 (no sections)

        write_phdr(&mut img, 0, 1, PF_R, 0, DYN_BASE, size as u64, size as u64, 0x1000);
        write_phdr(
            &mut img,
            1,
            2, // PT_DYNAMIC
            PF_R,
            dyn_off as u64,
            DYN_BASE + dyn_off as u64,
            (count * 16) as u64,
            (count * 16) as u64,
            8,
        );
        for i in 0..count {
            write_dyn(&mut img, dyn_off, i, 21, 0); // DT_DEBUG: modeled raw only
        }
        img
    }

    #[test]
    fn parses_synthetic_elf64_header() {
        let elf = ElfFile::parse(&synthetic_elf64()).unwrap();
        assert_eq!(elf.header.elf_type, ElfType::Dyn);
        assert_eq!(elf.header.machine, Machine::X86_64);
        assert_eq!(elf.header.entry, TEXT_VADDR);
        assert_eq!(elf.header.phnum, 2);
        assert_eq!(elf.header.shnum, 5);
        assert_eq!(elf.header.shstrndx, 4);
        assert_eq!(elf.program_headers.len(), 2);
        assert_eq!(elf.sections.len(), 5);
    }

    #[test]
    fn parses_program_headers_and_flags() {
        let elf = ElfFile::parse(&synthetic_elf64()).unwrap();
        let load = &elf.program_headers[0];
        assert_eq!(load.segment_type, SegmentType::Load);
        assert!(load.is_read());
        assert!(load.is_execute());
        assert!(!load.is_write());
        assert_eq!(load.offset, TEXT_OFF as u64);
        assert_eq!(load.vaddr, TEXT_VADDR);
        assert_eq!(load.filesz, 0x80);
        assert_eq!(load.align, 0x1000);
        assert_eq!(elf.program_headers[1].segment_type, SegmentType::Interp);
    }

    #[test]
    fn resolves_section_names() {
        let elf = ElfFile::parse(&synthetic_elf64()).unwrap();
        let names: Vec<&str> = elf.sections.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["", ".text", ".symtab", ".strtab", ".shstrtab"]);

        let text = elf.section_by_name(".text").unwrap();
        assert_eq!(text.section_type, SectionType::Progbits);
        assert_eq!(text.flags, SHF_ALLOC | SHF_EXECINSTR);
        assert_eq!(text.addr, TEXT_VADDR);
        assert_eq!(text.offset, TEXT_OFF as u64);
        assert_eq!(text.size, 0x80);

        let symtab = elf.section_by_name(".symtab").unwrap();
        assert_eq!(symtab.section_type, SectionType::Symtab);
        assert_eq!(symtab.link, 3);
        assert_eq!(symtab.entsize, 24);

        assert!(elf.section_by_name(".missing").is_none());
    }

    #[test]
    fn parses_symbols_with_names() {
        let elf = ElfFile::parse(&synthetic_elf64()).unwrap();
        assert_eq!(elf.symtab.len(), 3); // null + main + helper
        assert!(elf.dynsym.is_empty());

        let main = elf.symbol_by_name("main").unwrap();
        assert_eq!(main.kind, SymbolKind::Func);
        assert_eq!(main.bind, SymbolBind::Global);
        assert_eq!(main.value, TEXT_VADDR);
        assert_eq!(main.size, 0x20);
        assert_eq!(main.shndx, 1);

        let helper = elf.symbol_by_name("helper").unwrap();
        assert_eq!(helper.bind, SymbolBind::Local);
        assert_eq!(helper.value, TEXT_VADDR + 0x20);

        assert!(elf.symbol_by_name("absent").is_none());
    }

    #[test]
    fn maps_vaddrs_through_load_segments() {
        let elf = ElfFile::parse(&synthetic_elf64()).unwrap();
        assert_eq!(elf.vaddr_to_offset(TEXT_VADDR).unwrap(), TEXT_OFF);
        assert_eq!(elf.vaddr_to_offset(TEXT_VADDR + 0x10).unwrap(), TEXT_OFF + 0x10);
        // Last file-backed byte maps; one past it does not.
        assert_eq!(elf.vaddr_to_offset(TEXT_VADDR + 0x7F).unwrap(), TEXT_OFF + 0x7F);
        assert_eq!(
            elf.vaddr_to_offset(TEXT_VADDR + 0x80),
            Err(ParseError::UnmappedVaddr(TEXT_VADDR + 0x80))
        );
        assert_eq!(
            elf.vaddr_to_offset(0x90_0000),
            Err(ParseError::UnmappedVaddr(0x90_0000))
        );
        assert!(elf.segment_for_vaddr(TEXT_VADDR).is_some());
        assert!(elf.segment_for_vaddr(0).is_none());
    }

    #[test]
    fn interpreter_and_pie_heuristic() {
        let elf = ElfFile::parse(&synthetic_elf64()).unwrap();
        assert_eq!(
            elf.interpreter().as_deref(),
            Some("/lib64/ld-linux-x86-64.so.2")
        );
        assert!(elf.is_pie());

        // Same image as ET_EXEC: not a PIE.
        let mut img = synthetic_elf64();
        put16(&mut img, 16, 2); // ET_EXEC
        let exec = ElfFile::parse(&img).unwrap();
        assert!(!exec.is_pie());
        assert_eq!(exec.header.elf_type, ElfType::Exec);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut img = synthetic_elf64();
        img[0] = b'X';
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::BadSignature { what: "ELF", .. }
        ));
    }

    #[test]
    fn rejects_class32_and_big_endian() {
        let mut img = synthetic_elf64();
        img[4] = 1; // ELFCLASS32
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        let mut img = synthetic_elf64();
        img[5] = 2; // ELFDATA2MSB
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));
    }

    #[test]
    fn rejects_oversized_table_counts_without_allocating() {
        // A crafted e_shnum whose table would dwarf the file must be
        // rejected up front, not drive a giant allocation or EOF loop.
        let mut img = synthetic_elf64();
        put16(&mut img, 60, 0xFFFE); // e_shnum
        assert!(ElfFile::parse(&img).is_err());

        let mut img = synthetic_elf64();
        put16(&mut img, 56, 0xFFFE); // e_phnum
        assert!(ElfFile::parse(&img).is_err());
    }

    #[test]
    fn rejects_name_offset_past_strtab_end() {
        let mut img = synthetic_elf64();
        // .text section's sh_name -> far past the end of .shstrtab.
        put32(&mut img, SHOFF + 64, 0x1000);
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        // Symbol name offset past the end of .strtab.
        let mut img = synthetic_elf64();
        put32(&mut img, SYMTAB_OFF + 24, 0x1000);
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));
    }

    #[test]
    fn rejects_out_of_range_shstrndx_and_symtab_link() {
        let mut img = synthetic_elf64();
        put16(&mut img, 62, 100); // e_shstrndx out of range
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        let mut img = synthetic_elf64();
        put32(&mut img, SHOFF + 2 * 64 + 40, 99); // .symtab sh_link out of range
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));
    }

    #[test]
    fn truncated_image_is_an_error_not_a_panic() {
        let img = synthetic_elf64();
        // The section header table is the last thing in the file, so every
        // truncated prefix must fail — and must never panic.
        for len in 0..img.len() {
            assert!(ElfFile::parse(&img[..len]).is_err(), "len {len:#x}");
        }
        assert!(ElfFile::parse(&img).is_ok());
    }

    #[test]
    fn parses_dynamic_section_entries() {
        let elf = ElfFile::parse(&synthetic_dynamic_elf64()).unwrap();
        let d = elf.dynamic.as_ref().unwrap();
        assert_eq!(d.needed, ["libc.so.6", "libm.so.6"]);
        assert_eq!(d.soname.as_deref(), Some("mylib.so"));
        assert_eq!(d.strtab, Some(DYN_BASE + DYN_STR_OFF as u64));
        assert_eq!(d.strsz, Some(DYNSTR.len() as u64));
        assert_eq!(d.symtab, Some(DYN_BASE + DYN_SYM_OFF as u64));
        assert_eq!(d.syment, Some(24));
        assert_eq!(d.rela, Some(DYN_BASE + DYN_RELA_OFF as u64));
        assert_eq!(d.relasz, Some(48));
        assert_eq!(d.relaent, Some(24));
        assert_eq!(d.jmprel, Some(DYN_BASE + DYN_JMPREL_OFF as u64));
        assert_eq!(d.pltrelsz, Some(72));
        assert_eq!(d.pltrel, Some(DT_RELA));
        assert_eq!(d.pltgot, Some(DYN_BASE + 0x3000));
        // The raw list carries everything except the DT_NULL terminator.
        assert_eq!(d.entries.len(), 14);
        assert_eq!(d.entries[0], DynEntry { tag: DT_NEEDED, value: 1 });
    }

    #[test]
    fn parses_rela_and_plt_relocation_tables() {
        let elf = ElfFile::parse(&synthetic_dynamic_elf64()).unwrap();

        assert_eq!(elf.rela_dyn.len(), 2);
        assert_eq!(elf.rela_dyn[0].reloc_type, RelocType::Relative);
        assert_eq!(elf.rela_dyn[0].offset, DYN_BASE + 0x2E00);
        assert_eq!(elf.rela_dyn[0].sym, 0);
        assert_eq!(elf.rela_dyn[0].addend, 0x1234);
        assert_eq!(elf.rela_dyn[1].reloc_type, RelocType::GlobDat);
        assert_eq!(elf.rela_dyn[1].sym, 1);

        assert_eq!(elf.rela_plt.len(), 3);
        for (i, r) in elf.rela_plt.iter().enumerate() {
            assert_eq!(r.reloc_type, RelocType::JumpSlot);
            assert_eq!(r.offset, GOT_SLOTS[i]);
            assert_eq!(r.sym, (i + 1) as u32);
            assert_eq!(r.addend, 0);
        }
    }

    #[test]
    fn plt_import_join_maps_got_slots_to_names() {
        let elf = ElfFile::parse(&synthetic_dynamic_elf64()).unwrap();
        let imports: Vec<(u64, &str)> = elf
            .plt_imports()
            .iter()
            .map(|i| (i.got_slot, i.name.as_str()))
            .collect();
        assert_eq!(
            imports,
            [
                (GOT_SLOTS[0], "read"),
                (GOT_SLOTS[1], "write"),
                (GOT_SLOTS[2], "close"),
            ]
        );
        assert!(
            elf.plt_imports()
                .iter()
                .all(|i| i.reloc_type == RelocType::JumpSlot)
        );
    }

    #[test]
    fn dynamic_section_found_via_sht_dynamic_fallback() {
        // Demote PT_DYNAMIC to PT_NOTE: the parser must fall back to the
        // SHT_DYNAMIC section header and land on the same table.
        let mut img = synthetic_dynamic_elf64();
        put32(&mut img, PHOFF + 56, 4); // second phdr's p_type -> PT_NOTE
        let elf = ElfFile::parse(&img).unwrap();
        let d = elf.dynamic.as_ref().unwrap();
        assert_eq!(d.needed, ["libc.so.6", "libm.so.6"]);
        assert_eq!(elf.plt_imports().len(), 3);
    }

    #[test]
    fn missing_dt_null_stops_at_segment_end() {
        // Fill the whole dynamic segment with non-null entries: the
        // parser must stop at the segment boundary, not run off it.
        let mut img = synthetic_dynamic_elf64();
        for i in 0..15 {
            write_dyn(&mut img, DYN_DYN_OFF, i, 21, 0); // DT_DEBUG
        }
        let elf = ElfFile::parse(&img).unwrap();
        let d = elf.dynamic.as_ref().unwrap();
        assert_eq!(d.entries.len(), 15);
        assert!(d.needed.is_empty());
        assert!(elf.rela_dyn.is_empty());
        assert!(elf.plt_imports().is_empty());
    }

    #[test]
    fn dyn_entry_cap_bounds_unterminated_tables() {
        let img = synthetic_unterminated_dynamic(MAX_DYN_ENTRIES + 8);
        let elf = ElfFile::parse(&img).unwrap();
        assert_eq!(elf.dynamic.as_ref().unwrap().entries.len(), MAX_DYN_ENTRIES);
    }

    #[test]
    fn dt_strtab_outside_any_load_is_an_error() {
        let mut img = synthetic_dynamic_elf64();
        put64(&mut img, dyn_tag_off(DYN_IDX_STRTAB) + 8, 0x99_0000);
        assert_eq!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::UnmappedVaddr(0x99_0000)
        );

        // A DT_NEEDED offset past the end of the string table.
        let mut img = synthetic_dynamic_elf64();
        put64(&mut img, dyn_tag_off(DYN_IDX_NEEDED0) + 8, 999);
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));
    }

    #[test]
    fn huge_reloc_table_sizes_are_rejected_up_front() {
        // DT_RELASZ far past the segment: rejected before any read or
        // allocation, since the table must be file-backed.
        let mut img = synthetic_dynamic_elf64();
        put64(&mut img, dyn_tag_off(DYN_IDX_RELASZ) + 8, 0xFFFF_FFFF);
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::UnexpectedEof { .. }
        ));

        // Same for DT_PLTRELSZ.
        let mut img = synthetic_dynamic_elf64();
        put64(&mut img, dyn_tag_off(DYN_IDX_PLTRELSZ) + 8, u64::MAX);
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::UnexpectedEof { .. }
        ));

        // DT_RELA without DT_RELASZ has no valid bound at all.
        let mut img = synthetic_dynamic_elf64();
        put64(&mut img, dyn_tag_off(DYN_IDX_RELASZ), 21); // tag -> DT_DEBUG
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));
    }

    #[test]
    fn out_of_range_r_sym_is_an_error() {
        // Symbol index 999 lands far outside the PT_LOAD's file-backed
        // range, so the join must fail with a typed error.
        let mut img = synthetic_dynamic_elf64();
        put64(
            &mut img,
            DYN_JMPREL_OFF + 8,
            (999u64 << 32) | R_X86_64_JUMP_SLOT as u64,
        );
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::UnmappedVaddr(_)
        ));
    }

    #[test]
    fn null_symbol_index_is_skipped_not_joined() {
        let mut img = synthetic_dynamic_elf64();
        put64(&mut img, DYN_JMPREL_OFF + 8, R_X86_64_JUMP_SLOT as u64); // sym -> 0
        let elf = ElfFile::parse(&img).unwrap();
        // The relocation itself is still recorded; only the join skips it.
        assert_eq!(elf.rela_plt.len(), 3);
        let names: Vec<&str> = elf.plt_imports().iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names, ["write", "close"]);
    }

    #[test]
    fn rel_format_plt_table_is_supported() {
        // Rewrite the JMPREL table as Elf64_Rel and declare it via
        // DT_PLTREL = DT_REL: same join, implicit zero addends.
        let mut img = synthetic_dynamic_elf64();
        for (i, &slot) in GOT_SLOTS.iter().enumerate() {
            write_rel(&mut img, DYN_JMPREL_OFF, i, slot, (i + 1) as u32, R_X86_64_JUMP_SLOT);
        }
        put64(&mut img, dyn_tag_off(DYN_IDX_PLTRELSZ) + 8, 3 * 16);
        put64(&mut img, dyn_tag_off(DYN_IDX_PLTREL) + 8, DT_REL);
        let elf = ElfFile::parse(&img).unwrap();
        assert_eq!(elf.rela_plt.len(), 3);
        assert!(elf.rela_plt.iter().all(|r| r.addend == 0));
        let names: Vec<&str> = elf.plt_imports().iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names, ["read", "write", "close"]);
    }

    #[test]
    fn bad_or_missing_plt_metadata_is_rejected() {
        // DT_PLTREL with a nonsense value.
        let mut img = synthetic_dynamic_elf64();
        put64(&mut img, dyn_tag_off(DYN_IDX_PLTREL) + 8, 5);
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        // DT_JMPREL without DT_PLTREL declaring the record format.
        let mut img = synthetic_dynamic_elf64();
        put64(&mut img, dyn_tag_off(DYN_IDX_PLTREL), 21); // tag -> DT_DEBUG
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));

        // DT_JMPREL without DT_PLTRELSZ bounding the table.
        let mut img = synthetic_dynamic_elf64();
        put64(&mut img, dyn_tag_off(DYN_IDX_PLTRELSZ), 21); // tag -> DT_DEBUG
        assert!(matches!(
            ElfFile::parse(&img).unwrap_err(),
            ParseError::Unsupported(_)
        ));
    }

    #[test]
    fn aarch64_reloc_types_decode_per_machine() {
        let mut img = synthetic_dynamic_elf64();
        put16(&mut img, 18, 183); // e_machine: EM_AARCH64
        for i in 0..3usize {
            put64(
                &mut img,
                DYN_JMPREL_OFF + i * 24 + 8,
                (((i + 1) as u64) << 32) | R_AARCH64_JUMP_SLOT as u64,
            );
        }
        put64(&mut img, DYN_RELA_OFF + 8, R_AARCH64_RELATIVE as u64);
        put64(
            &mut img,
            DYN_RELA_OFF + 24 + 8,
            (1u64 << 32) | R_AARCH64_GLOB_DAT as u64,
        );
        let elf = ElfFile::parse(&img).unwrap();
        assert_eq!(elf.header.machine, Machine::Aarch64);
        assert!(elf.rela_plt.iter().all(|r| r.reloc_type == RelocType::JumpSlot));
        assert_eq!(elf.rela_dyn[0].reloc_type, RelocType::Relative);
        assert_eq!(elf.rela_dyn[1].reloc_type, RelocType::GlobDat);
        let names: Vec<&str> = elf.plt_imports().iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names, ["read", "write", "close"]);

        // The same raw values on the wrong machine stay Other(_).
        assert_eq!(
            RelocType::from_raw(Machine::X86_64, R_AARCH64_JUMP_SLOT),
            RelocType::Other(R_AARCH64_JUMP_SLOT)
        );
        assert_eq!(
            RelocType::from_raw(Machine::RiscV, R_X86_64_JUMP_SLOT),
            RelocType::Other(R_X86_64_JUMP_SLOT)
        );
    }

    #[test]
    fn truncated_dynamic_image_is_an_error_not_a_panic() {
        let img = synthetic_dynamic_elf64();
        // The section header table runs to the exact end of the file, so
        // every truncated prefix must fail — and must never panic.
        for len in 0..img.len() {
            assert!(ElfFile::parse(&img[..len]).is_err(), "len {len:#x}");
        }
        assert!(ElfFile::parse(&img).is_ok());
    }

    /// Smoke-test against a real ELF binary when one is present on this
    /// machine (they won't be on macOS, where system binaries are Mach-O;
    /// the test is then a no-op and synthetic coverage stands alone).
    #[test]
    fn smoke_parses_real_elf_if_present() {
        let candidates = [
            "/lib64/ld-linux-x86-64.so.2",
            "/usr/lib/x86_64-linux-gnu/libc.so.6",
            "/lib/ld-musl-x86_64.so.1",
            "/usr/bin/env",
            "/bin/ls",
        ];
        for path in candidates {
            let Ok(data) = std::fs::read(path) else {
                continue;
            };
            if data.len() < 6 || &data[0..4] != b"\x7fELF" {
                continue; // not ELF (e.g. Mach-O on macOS)
            }
            if data[4] != ELFCLASS64 || data[5] != ELFDATA2LSB {
                continue; // outside this pass's scope
            }
            let elf = ElfFile::parse(&data).unwrap_or_else(|e| panic!("{path}: {e}"));
            assert!(!elf.program_headers.is_empty(), "{path}: no program headers");
            assert!(
                elf.program_headers
                    .iter()
                    .any(|ph| ph.segment_type == SegmentType::Load),
                "{path}: no PT_LOAD"
            );
        }
    }

    #[test]
    fn eh_leb128_signed_and_unsigned() {
        let mut r = Reader::new(&[0xE5, 0x8E, 0x26]);
        assert_eq!(read_uleb(&mut r), Some(624485));
        // -2 as sleb128 is a single byte 0x7E.
        let mut r = Reader::new(&[0x7E]);
        assert_eq!(read_sleb(&mut r), Some(-2));
        // 0x80 0x7F sleb128 = -128.
        let mut r = Reader::new(&[0x80, 0x7F]);
        assert_eq!(read_sleb(&mut r), Some(-128));
    }

    #[test]
    fn eh_pointer_pcrel_sdata4_is_relative_to_its_own_va() {
        // Field at section offset 8; section VA 0x4000 → field VA 0x4008.
        // sdata4 value +0x100, pcrel → target 0x4108.
        let mut sec = vec![0u8; 12];
        sec[8..12].copy_from_slice(&0x100i32.to_le_bytes());
        let enc = DW_EH_PE_PCREL | DW_EH_PE_SDATA4; // 0x1B
        assert_eq!(decode_eh_pointer(&sec, 8, enc, 0x4000), Some(0x4108));
        // A negative offset resolves backward.
        sec[8..12].copy_from_slice(&(-0x10i32).to_le_bytes());
        assert_eq!(decode_eh_pointer(&sec, 8, enc, 0x4000), Some(0x3FF8));
    }

    #[test]
    fn eh_pointer_absolute_and_unsupported_forms() {
        let mut sec = vec![0u8; 16];
        sec[0..8].copy_from_slice(&0x1234_5678_9ABCu64.to_le_bytes());
        // absptr → 8-byte absolute value, no relocation against the VA.
        assert_eq!(
            decode_eh_pointer(&sec, 0, DW_EH_PE_ABSPTR, 0x4000),
            Some(0x1234_5678_9ABC)
        );
        // udata4 absolute: the low 4 little-endian bytes.
        assert_eq!(decode_eh_pointer(&sec, 0, DW_EH_PE_UDATA4, 0), Some(0x5678_9ABC));
        // omit → nothing; a datarel application we don't support → None.
        assert_eq!(decode_eh_pointer(&sec, 0, DW_EH_PE_OMIT, 0), None);
        assert_eq!(decode_eh_pointer(&sec, 0, 0x30 | DW_EH_PE_SDATA4, 0), None);
        // Reading past the section end is None, never a panic.
        assert_eq!(decode_eh_pointer(&sec, 14, DW_EH_PE_UDATA4, 0), None);
    }

    #[test]
    fn cie_augmentation_yields_fde_encoding() {
        // CIE body after the 4-byte CIE-id: version 1, aug "zR", code
        // align 1 (uleb), data align -8 (sleb 0x78), return reg 16 (u8),
        // aug data len 1 (uleb), then the R encoding byte 0x1B.
        let body = [1, b'z', b'R', 0, 1, 0x78, 16, 1, 0x1B];
        assert_eq!(parse_cie_fde_encoding(&body), Some(0x1B));

        // "zPLR": personality (P) with a udata4 pointer, then L, then R.
        // aug data = [P-enc=0x03, 4 pointer bytes, L-enc=0x00, R-enc=0x1B].
        let body = [
            1, b'z', b'P', b'L', b'R', 0, 1, 0x78, 16, 7, 0x03, 0xAA, 0xBB, 0xCC, 0xDD, 0x00, 0x1B,
        ];
        assert_eq!(parse_cie_fde_encoding(&body), Some(0x1B));

        // No 'z' augmentation → no encoding byte (caller defaults absptr).
        let body = [1, 0, 1, 0x78, 16];
        assert_eq!(parse_cie_fde_encoding(&body), None);
    }
}
