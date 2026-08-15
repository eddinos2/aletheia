//! Format- and ISA-neutral trait layer.
//!
//! Analysis is written once against the types here, never against a
//! specific container format or instruction set:
//!
//! - [`Image`] is the format-neutral view of a loaded binary — entry
//!   points, memory regions, symbols, import slots, and VA-to-file-offset
//!   mapping — implemented by [`PeImage`], [`ElfImage`], and
//!   [`MachImage`], and obtained format-blind via [`load`].
//! - [`Decoder`] is the ISA-neutral instruction-flow decoder — length and
//!   [`Flow`] classification only, which is exactly what control-flow
//!   recovery consumes — implemented by [`X86Decoder`] and
//!   [`Aarch64Decoder`] and selected from an image's [`Arch`] via
//!   [`decoder_for`].
//! - [`Flow`] is the single control-flow vocabulary shared by every
//!   decoder ([`crate::x86`] and [`crate::aarch64`] re-export it).
//!
//! Virtual-address arithmetic in this module wraps rather than panics:
//! a hostile image base or RVA must never abort analysis.

use crate::error::{ParseError, Result};
use crate::pe::exports::{ExportDirectory, ExportTarget};
use crate::pe::{ImportSymbol, ImportedDll, PeFile};
use crate::{aarch64, elf, macho, x86};

/// Control-flow classification of one decoded instruction.
///
/// The single source of truth shared by every ISA decoder. Branch targets
/// are absolute virtual addresses, computed by the decoder from the VA
/// the caller supplied for the instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Execution falls through to the next instruction.
    Sequential,
    /// Unconditional branch to a statically known target.
    Jump(u64),
    /// Conditional branch: taken edge to the target, fall-through edge to
    /// the next instruction.
    CondJump(u64),
    /// Unconditional branch whose target is unknown statically
    /// (through a register or memory).
    IndirectJump,
    /// Call to a statically known target (falls through on return).
    Call(u64),
    /// Call whose target is unknown statically.
    IndirectCall,
    /// Return to the caller.
    Return,
    /// Trap into the kernel or debugger (software interrupt, syscall,
    /// breakpoint, undefined-instruction trap). Execution may resume
    /// after the instruction.
    Interrupt,
    /// Halt execution.
    Halt,
}

/// Instruction-set architecture of an image's code, selecting a decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// x86-64, decoded by [`X86Decoder`].
    X86_64,
    /// AArch64 (A64), decoded by [`Aarch64Decoder`].
    Aarch64,
    /// An architecture with no decoder in this crate.
    Other,
}

/// The flow-level result of decoding one instruction: everything
/// control-flow recovery needs, nothing more.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decoded {
    /// Total encoded length in bytes.
    pub length: u8,
    /// Control-flow classification.
    pub flow: Flow,
    /// For [`Flow::IndirectCall`] / [`Flow::IndirectJump`] through a
    /// statically addressed memory cell — x86 `call/jmp [rip+disp]` or
    /// absolute `[disp32]` — the absolute VA of that cell. This is the
    /// join point control-flow recovery matches against
    /// [`Image::import_slots`] to resolve IAT/GOT-indirected calls.
    ///
    /// `None` for register-indirect forms, for every other [`Flow`], and
    /// always on AArch64: its import calls go through `ADRP`+`LDR`
    /// register sequences, which flow-level decoding does not track.
    pub mem_target: Option<u64>,
}

/// ISA-neutral instruction-flow decoder (object safe).
///
/// Implementations delegate to the rich per-ISA decoders; use those
/// directly when operands matter. Errors are the underlying decoder's
/// errors, surfaced as-is: truncated input is
/// [`ParseError::UnexpectedEof`], and an unrecognized or unmodeled
/// encoding (x86 only — A64 decoding is total) is
/// [`ParseError::Unsupported`] *with no length*. Recursive descent must
/// treat such an error as a basic-block terminator: there is no reliable
/// way to skip an undecodable variable-length instruction.
pub trait Decoder {
    /// Decode one instruction from the start of `bytes`, located at
    /// virtual address `va`, to its length and [`Flow`] effect.
    fn decode_flow(&self, bytes: &[u8], va: u64) -> Result<Decoded>;
}

/// [`Decoder`] for x86-64, delegating to [`x86::decode`].
#[derive(Debug, Clone, Copy, Default)]
pub struct X86Decoder;

impl Decoder for X86Decoder {
    fn decode_flow(&self, bytes: &[u8], va: u64) -> Result<Decoded> {
        let ins = x86::decode(bytes, va)?;
        // The slot VA of an indirect branch through a statically
        // addressed memory operand: `[rip+disp]` resolves against the
        // next instruction's VA (SDM Vol 2 §2.2.1.6), and a no-base,
        // no-index `[disp32]` is absolute.
        let mem_target = match ins.flow {
            Flow::IndirectCall | Flow::IndirectJump => match ins.operands.first() {
                Some(&x86::Operand::Mem {
                    base: None,
                    index: None,
                    disp,
                    rip_relative,
                    ..
                }) => Some(if rip_relative {
                    va.wrapping_add(ins.length as u64).wrapping_add(disp as u64)
                } else {
                    disp as u64
                }),
                _ => None,
            },
            _ => None,
        };
        Ok(Decoded {
            length: ins.length,
            flow: ins.flow,
            mem_target,
        })
    }
}

/// [`Decoder`] for AArch64, delegating to [`aarch64::decode`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Aarch64Decoder;

impl Decoder for Aarch64Decoder {
    fn decode_flow(&self, bytes: &[u8], va: u64) -> Result<Decoded> {
        let ins = aarch64::decode(bytes, va)?;
        Ok(Decoded {
            length: aarch64::Instruction::SIZE as u8,
            flow: ins.flow,
            // A64 indirect branches are register-only (BR/BLR); import
            // calls indirect through ADRP+LDR sequences that a single-
            // instruction flow decode cannot resolve. See `Decoded`.
            mem_target: None,
        })
    }
}

/// The decoder for an architecture, or `None` for [`Arch::Other`].
pub fn decoder_for(arch: Arch) -> Option<&'static dyn Decoder> {
    match arch {
        Arch::X86_64 => Some(&X86Decoder),
        Arch::Aarch64 => Some(&Aarch64Decoder),
        Arch::Other => None,
    }
}

/// Memory permissions of a [`Region`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Perms {
    /// Readable.
    pub r: bool,
    /// Writable.
    pub w: bool,
    /// Executable.
    pub x: bool,
}

/// One mapped range of the image: the format-neutral unification of PE
/// sections, ELF sections/segments, and Mach-O segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    /// Region name (e.g. `.text`, `__TEXT`); may be empty or synthesized
    /// for formats/files that carry no names.
    pub name: String,
    /// Virtual address of the region's first byte.
    pub va: u64,
    /// Size of the region in memory, in bytes (may exceed its file-backed
    /// extent; the tail is zero-fill).
    pub size: u64,
    pub perms: Perms,
}

/// Coarse classification of a [`Symbol`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// Executable code.
    Function,
    /// A data object.
    Data,
    /// Anything else, or a kind the format does not distinguish.
    Other,
}

/// A named virtual address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    pub va: u64,
    pub kind: SymbolKind,
}

/// One import-resolution slot: a pointer-sized location the loader
/// patches with the address of a named import. A `call`/`jmp` (or a PLT
/// stub load) indirecting through `slot_va` calls `name` — this is the
/// join control-flow recovery uses to resolve indirect calls to imports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportSlot {
    /// Virtual address of the slot (PE IAT entry, ELF GOT entry).
    pub slot_va: u64,
    /// Imported symbol name; for PE, qualified as `"dll!func"` (or
    /// `"dll!#ordinal"` for by-ordinal imports).
    pub name: String,
}

/// Format-neutral view of a loaded binary image (object safe).
///
/// Everything Phase-2 control-flow recovery needs: where code may start
/// ([`Image::entry_points`]), what memory looks like ([`Image::regions`]),
/// what things are called ([`Image::symbols`], [`Image::import_slots`]),
/// and how to read bytes at a VA ([`Image::va_to_offset`] +
/// [`Image::bytes`]).
pub trait Image {
    /// The architecture of the image's code, selecting a [`Decoder`]
    /// via [`decoder_for`].
    fn arch(&self) -> Arch;

    /// Known code entry points, as virtual addresses, sorted and deduped.
    ///
    /// Included per format:
    /// - PE: the optional header's entry point (when nonzero) plus every
    ///   non-forwarded export's VA (PE exports carry no type information;
    ///   in practice they are overwhelmingly functions).
    /// - ELF: `e_entry` (when nonzero) plus every defined `STT_FUNC`
    ///   symbol in `.dynsym` (the dynamically exported functions).
    /// - Mach-O: the `LC_MAIN` entry resolved to a VA through the
    ///   segments. (Exported-symbol entry points are not included: the
    ///   Mach-O symbol table does not distinguish functions from data.)
    fn entry_points(&self) -> Vec<u64>;

    /// The image's mapped regions, unified across formats:
    /// - PE: the section table.
    /// - ELF: the `SHF_ALLOC` sections when the file has any, else the
    ///   `PT_LOAD` segments (section headers may be stripped) with
    ///   synthesized names (`load0`, `load1`, ...). Allocated ELF
    ///   sections are reported readable; write/execute come from
    ///   `SHF_WRITE`/`SHF_EXECINSTR`.
    /// - Mach-O: the `LC_SEGMENT_64` segments (skipping zero-size ones);
    ///   permissions are the segment's initial protection.
    fn regions(&self) -> Vec<Region>;

    /// Named addresses, sorted by (VA, name) and deduped. Undefined and
    /// unnamed symbols are omitted. Included per format:
    /// - PE: named, non-forwarded exports (reported as
    ///   [`SymbolKind::Function`]; see [`Image::entry_points`]).
    /// - ELF: defined symbols from `.symtab` and `.dynsym`
    ///   (`STT_FUNC` → `Function`, `STT_OBJECT` → `Data`; file and
    ///   section markers omitted).
    /// - Mach-O: defined (`N_SECT`) symbols; kind is inferred from the
    ///   permissions of the segment owning the symbol's section
    ///   (executable → `Function`, else `Data`).
    fn symbols(&self) -> Vec<Symbol>;

    /// Import-resolution slots (see [`ImportSlot`]), per format:
    /// - PE: IAT entries from the import directory, names qualified as
    ///   `"dll!func"` / `"dll!#ordinal"` (delay-load imports are not
    ///   included in this pass).
    /// - ELF: GOT slots from the PLT relocation table joined to dynamic
    ///   symbol names ([`elf::ElfFile::plt_imports`]).
    /// - Mach-O: chained-fixup import *names* when
    ///   `LC_DYLD_CHAINED_FIXUPS` is present (slot VAs still require a
    ///   full chain walk — names-only until that slice); empty otherwise.
    fn import_slots(&self) -> Vec<ImportSlot>;

    /// Map a virtual address to an offset into [`Image::bytes`].
    ///
    /// `None` when the VA is outside every file-backed range (headers
    /// aside, zero-fill tails such as `.bss` have no file offset) or the
    /// mapped offset lies past the end of the buffer.
    fn va_to_offset(&self, va: u64) -> Option<usize>;

    /// The raw file buffer this image was parsed from.
    fn bytes(&self) -> &[u8];

    /// Function-start VAs the image's own compiler-emitted metadata
    /// declares, sorted and deduplicated — the highest-precision source
    /// for [`crate::funcs`] discovery. Defaults to empty; formats that
    /// carry such a table override it:
    /// - PE: the exception directory (`.pdata`).
    /// - ELF: the `.eh_frame` FDE initial locations.
    /// - Mach-O: the `LC_FUNCTION_STARTS` table.
    fn function_starts_hint(&self) -> Vec<u64> {
        Vec::new()
    }
}

/// Parse `data` as whichever format its magic bytes declare, returning
/// the format-neutral [`Image`] view — the one analysis-facing API that
/// loads any supported format.
///
/// Sniffed magics: `MZ` (PE), `\x7fELF` (ELF), and the thin Mach-O /
/// fat-container magics. Files recognized as Mach-O variants outside the
/// supported subset (32-bit, big-endian, fat containers — select a slice
/// with [`macho::FatFile`] and load that) fail with the underlying
/// parser's typed error; anything unrecognized is
/// [`ParseError::Unsupported`].
pub fn load(data: &[u8]) -> Result<Box<dyn Image + '_>> {
    if data.len() >= 2 && data[..2] == *b"MZ" {
        return Ok(Box::new(PeImage::parse(data)?));
    }
    if data.len() >= 4 {
        match u32::from_le_bytes([data[0], data[1], data[2], data[3]]) {
            elf::ELF_MAGIC => return Ok(Box::new(ElfImage::parse(data)?)),
            macho::MH_MAGIC_64
            | macho::MH_MAGIC
            | macho::MH_CIGAM
            | macho::MH_CIGAM_64
            | macho::FAT_CIGAM
            | macho::FAT_CIGAM_64 => return Ok(Box::new(MachImage::parse(data)?)),
            _ => {}
        }
    }
    Err(ParseError::Unsupported(
        "unrecognized image format: no PE (MZ), ELF (\\x7fELF), or Mach-O magic".into(),
    ))
}

// PE section characteristics (PE spec §Section Flags).
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;

/// [`Image`] over a parsed PE file and the buffer it came from.
///
/// The import and export directories are parsed eagerly by
/// [`PeImage::parse`], so a malformed directory is a parse-time error
/// (matching [`elf::ElfFile::parse`], which resolves its PLT imports up
/// front) and the trait methods are infallible.
pub struct PeImage<'a> {
    pe: PeFile,
    data: &'a [u8],
    imports: Vec<ImportedDll>,
    exports: Option<ExportDirectory>,
}

impl<'a> PeImage<'a> {
    /// Parse a PE image (headers plus import/export directories).
    pub fn parse(data: &'a [u8]) -> Result<PeImage<'a>> {
        let pe = PeFile::parse(data)?;
        let imports = pe.imports(data)?;
        let exports = pe.exports(data)?;
        Ok(PeImage {
            pe,
            data,
            imports,
            exports,
        })
    }

    /// The underlying rich [`PeFile`], for format-specific analysis.
    pub fn file(&self) -> &PeFile {
        &self.pe
    }

    fn base(&self) -> u64 {
        self.pe.optional.image_base
    }
}

impl Image for PeImage<'_> {
    fn arch(&self) -> Arch {
        match self.pe.coff.machine {
            crate::pe::Machine::X86_64 => Arch::X86_64,
            crate::pe::Machine::Arm64 => Arch::Aarch64,
            _ => Arch::Other,
        }
    }

    fn entry_points(&self) -> Vec<u64> {
        let mut eps = Vec::new();
        if self.pe.optional.entry_point_rva != 0 {
            eps.push(self.base().wrapping_add(self.pe.optional.entry_point_rva as u64));
        }
        if let Some(dir) = &self.exports {
            for e in &dir.exports {
                if let ExportTarget::Rva(rva) = e.target {
                    eps.push(self.base().wrapping_add(rva as u64));
                }
            }
        }
        eps.sort_unstable();
        eps.dedup();
        eps
    }

    fn regions(&self) -> Vec<Region> {
        self.pe
            .sections
            .iter()
            .map(|s| Region {
                name: s.name.clone(),
                va: self.base().wrapping_add(s.virtual_address as u64),
                // VirtualSize is the in-memory truth; object-style files
                // may leave it 0 and use the raw size instead.
                size: if s.virtual_size != 0 {
                    s.virtual_size as u64
                } else {
                    s.size_of_raw_data as u64
                },
                perms: Perms {
                    r: s.characteristics & IMAGE_SCN_MEM_READ != 0,
                    w: s.characteristics & IMAGE_SCN_MEM_WRITE != 0,
                    x: s.characteristics & IMAGE_SCN_MEM_EXECUTE != 0,
                },
            })
            .collect()
    }

    fn symbols(&self) -> Vec<Symbol> {
        let mut out: Vec<Symbol> = self
            .exports
            .iter()
            .flat_map(|dir| dir.exports.iter())
            .filter_map(|e| {
                let name = e.name.clone()?;
                let ExportTarget::Rva(rva) = e.target else {
                    return None; // forwarders resolve in another image
                };
                Some(Symbol {
                    name,
                    va: self.base().wrapping_add(rva as u64),
                    kind: SymbolKind::Function,
                })
            })
            .collect();
        sort_symbols(&mut out);
        out
    }

    fn import_slots(&self) -> Vec<ImportSlot> {
        let mut slots = Vec::new();
        for dll in &self.imports {
            for f in &dll.functions {
                let name = match &f.symbol {
                    ImportSymbol::Name { name, .. } => format!("{}!{name}", dll.name),
                    ImportSymbol::Ordinal(n) => format!("{}!#{n}", dll.name),
                };
                slots.push(ImportSlot {
                    slot_va: self.base().wrapping_add(f.iat_rva as u64),
                    name,
                });
            }
        }
        slots
    }

    fn va_to_offset(&self, va: u64) -> Option<usize> {
        let rva = u32::try_from(va.checked_sub(self.base())?).ok()?;
        let off = self.pe.rva_to_offset(rva).ok()?;
        (off < self.data.len()).then_some(off)
    }

    fn bytes(&self) -> &[u8] {
        self.data
    }

    fn function_starts_hint(&self) -> Vec<u64> {
        self.pe
            .pdata_function_starts(self.data)
            .into_iter()
            .map(|rva| self.base().wrapping_add(rva as u64))
            .collect()
    }
}

/// [`Image`] over a parsed ELF file and the buffer it came from.
pub struct ElfImage<'a> {
    elf: elf::ElfFile,
    data: &'a [u8],
}

impl<'a> ElfImage<'a> {
    /// Parse an ELF image.
    pub fn parse(data: &'a [u8]) -> Result<ElfImage<'a>> {
        Ok(ElfImage {
            elf: elf::ElfFile::parse(data)?,
            data,
        })
    }

    /// The underlying rich [`elf::ElfFile`], for format-specific analysis.
    pub fn file(&self) -> &elf::ElfFile {
        &self.elf
    }
}

impl Image for ElfImage<'_> {
    fn arch(&self) -> Arch {
        match self.elf.header.machine {
            elf::Machine::X86_64 => Arch::X86_64,
            elf::Machine::Aarch64 => Arch::Aarch64,
            _ => Arch::Other,
        }
    }

    fn entry_points(&self) -> Vec<u64> {
        let mut eps = Vec::new();
        if self.elf.header.entry != 0 {
            eps.push(self.elf.header.entry);
        }
        for s in &self.elf.dynsym {
            if s.kind == elf::SymbolKind::Func && s.shndx != 0 && s.value != 0 {
                eps.push(s.value);
            }
        }
        eps.sort_unstable();
        eps.dedup();
        eps
    }

    fn regions(&self) -> Vec<Region> {
        let sections: Vec<Region> = self
            .elf
            .sections
            .iter()
            .filter(|s| s.flags & elf::SHF_ALLOC != 0)
            .map(|s| Region {
                name: s.name.clone(),
                va: s.addr,
                size: s.size,
                perms: Perms {
                    r: true, // SHF_ALLOC: mapped, hence readable
                    w: s.flags & elf::SHF_WRITE != 0,
                    x: s.flags & elf::SHF_EXECINSTR != 0,
                },
            })
            .collect();
        if !sections.is_empty() {
            return sections;
        }
        // Section headers stripped: fall back to the loader's view.
        self.elf
            .program_headers
            .iter()
            .filter(|ph| ph.segment_type == elf::SegmentType::Load)
            .enumerate()
            .map(|(i, ph)| Region {
                name: format!("load{i}"),
                va: ph.vaddr,
                size: ph.memsz,
                perms: Perms {
                    r: ph.is_read(),
                    w: ph.is_write(),
                    x: ph.is_execute(),
                },
            })
            .collect()
    }

    fn symbols(&self) -> Vec<Symbol> {
        let mut out: Vec<Symbol> = self
            .elf
            .symtab
            .iter()
            .chain(self.elf.dynsym.iter())
            .filter(|s| {
                !s.name.is_empty()
                    && s.shndx != 0 // defined, not an import
                    && !matches!(s.kind, elf::SymbolKind::File | elf::SymbolKind::Section)
            })
            .map(|s| Symbol {
                name: s.name.clone(),
                va: s.value,
                kind: match s.kind {
                    elf::SymbolKind::Func => SymbolKind::Function,
                    elf::SymbolKind::Object => SymbolKind::Data,
                    _ => SymbolKind::Other,
                },
            })
            .collect();
        sort_symbols(&mut out);
        out
    }

    fn import_slots(&self) -> Vec<ImportSlot> {
        self.elf
            .plt_imports()
            .iter()
            .map(|i| ImportSlot {
                slot_va: i.got_slot,
                name: i.name.clone(),
            })
            .collect()
    }

    fn va_to_offset(&self, va: u64) -> Option<usize> {
        let off = self.elf.vaddr_to_offset(va).ok()?;
        (off < self.data.len()).then_some(off)
    }

    fn bytes(&self) -> &[u8] {
        self.data
    }

    fn function_starts_hint(&self) -> Vec<u64> {
        self.elf.eh_frame_function_starts(self.data)
    }
}

/// [`Image`] over a parsed thin Mach-O file and the buffer it came from.
///
/// Fat/universal containers are not an [`Image`]: select a slice with
/// [`macho::FatFile`] and load that slice's bytes instead.
pub struct MachImage<'a> {
    mach: macho::MachFile,
    data: &'a [u8],
}

impl<'a> MachImage<'a> {
    /// Parse a thin Mach-O image.
    pub fn parse(data: &'a [u8]) -> Result<MachImage<'a>> {
        Ok(MachImage {
            mach: macho::MachFile::parse(data)?,
            data,
        })
    }

    /// The underlying rich [`macho::MachFile`], for format-specific
    /// analysis.
    pub fn file(&self) -> &macho::MachFile {
        &self.mach
    }

    /// Map a file offset back to a VA through the file-backed segments
    /// (the inverse of [`macho::MachFile::vaddr_to_offset`]).
    fn offset_to_vaddr(&self, off: u64) -> Option<u64> {
        self.mach
            .segments
            .iter()
            .find(|s| s.filesize > 0 && off >= s.fileoff && off - s.fileoff < s.filesize)
            .map(|s| s.vmaddr.wrapping_add(off - s.fileoff))
    }
}

impl Image for MachImage<'_> {
    fn arch(&self) -> Arch {
        match self.mach.header.cputype {
            macho::CpuType::X86_64 => Arch::X86_64,
            macho::CpuType::Arm64 => Arch::Aarch64,
            macho::CpuType::Other(_) => Arch::Other,
        }
    }

    fn entry_points(&self) -> Vec<u64> {
        // LC_MAIN's entryoff is a file offset; resolve it to a VA.
        self.mach
            .entry_offset()
            .and_then(|off| self.offset_to_vaddr(off))
            .into_iter()
            .collect()
    }

    fn regions(&self) -> Vec<Region> {
        self.mach
            .segments
            .iter()
            .filter(|s| s.vmsize > 0)
            .map(|s| Region {
                name: s.segname.clone(),
                va: s.vmaddr,
                size: s.vmsize,
                perms: Perms {
                    r: s.is_read(),
                    w: s.is_write(),
                    x: s.is_execute(),
                },
            })
            .collect()
    }

    fn symbols(&self) -> Vec<Symbol> {
        // n_sect is a 1-based ordinal over all sections in load-command
        // order; a symbol is code iff its section's segment is executable.
        let sections: Vec<&macho::Section64> = self
            .mach
            .segments
            .iter()
            .flat_map(|s| s.sections.iter())
            .collect();
        let mut out: Vec<Symbol> = self
            .mach
            .symbols
            .iter()
            .filter(|s| !s.name.is_empty() && s.kind == macho::SymbolKind::Defined)
            .map(|s| {
                let kind = (s.sect as usize)
                    .checked_sub(1)
                    .and_then(|i| sections.get(i))
                    .map_or(SymbolKind::Other, |sec| {
                        let exec = self
                            .mach
                            .segment_by_name(&sec.segname)
                            .is_some_and(|seg| seg.is_execute());
                        if exec { SymbolKind::Function } else { SymbolKind::Data }
                    });
                Symbol {
                    name: s.name.clone(),
                    va: s.value,
                    kind,
                }
            })
            .collect();
        sort_symbols(&mut out);
        out
    }

    fn import_slots(&self) -> Vec<ImportSlot> {
        // Full GOT/stub VA resolution needs a chain walk (next slice).
        // When chained fixups are present we still surface named imports
        // at VA 0 as placeholders so tooling can see the name table —
        // consumers that require a real slot VA must wait for the walk.
        let Some(cf) = &self.mach.chained_fixups else {
            return Vec::new();
        };
        cf.import_names
            .iter()
            .filter(|n| !n.is_empty())
            .map(|name| ImportSlot {
                slot_va: 0,
                name: name.clone(),
            })
            .collect()
    }

    fn va_to_offset(&self, va: u64) -> Option<usize> {
        let off = self.mach.vaddr_to_offset(va).ok()?;
        (off < self.data.len()).then_some(off)
    }

    fn bytes(&self) -> &[u8] {
        self.data
    }

    fn function_starts_hint(&self) -> Vec<u64> {
        self.mach.function_starts.clone()
    }
}

/// Canonical [`Image::symbols`] order: by (VA, name), exact dupes
/// (e.g. a symbol present in both `.symtab` and `.dynsym`) removed.
fn sort_symbols(symbols: &mut Vec<Symbol>) {
    symbols.sort_by(|a, b| (a.va, a.name.as_str()).cmp(&(b.va, b.name.as_str())));
    symbols.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::tests::{synthetic_dynamic_elf64, synthetic_elf64};
    use crate::macho::tests::synthetic_macho64;
    use crate::pe::exports::tests::with_exports;
    use crate::pe::tests::{synthetic_pe64, with_imports};

    /// Image bases / layout constants of the synthetic fixtures, restated
    /// here so a drifting fixture fails these assertions loudly.
    const PE_BASE: u64 = 0x1_4000_0000;
    const ELF_TEXT: u64 = 0x40_1000;
    const ELF_DYN_BASE: u64 = 0x40_0000;
    const MACH_TEXT: u64 = 0x1_0000_0000;
    const MACH_ENTRYOFF: u64 = 0x140;

    /// The format-blind projection: everything here is written against
    /// `dyn Image`, never a concrete format.
    struct View {
        arch: Arch,
        entries: Vec<u64>,
        regions: Vec<Region>,
        symbols: Vec<Symbol>,
        slots: Vec<ImportSlot>,
    }

    fn describe<I: Image + ?Sized>(img: &I) -> View {
        View {
            arch: img.arch(),
            entries: img.entry_points(),
            regions: img.regions(),
            symbols: img.symbols(),
            slots: img.import_slots(),
        }
    }

    #[test]
    fn pe_image_view_is_faithful() {
        let img = with_imports();
        let pe = PeImage::parse(&img).unwrap();
        let v = describe(&pe);

        assert_eq!(v.arch, Arch::X86_64);
        assert_eq!(v.entries, [PE_BASE + 0x1000]);
        assert_eq!(
            v.regions,
            [Region {
                name: ".text".into(),
                va: PE_BASE + 0x1000,
                size: 0x100,
                perms: Perms {
                    r: true,
                    w: false,
                    x: true
                },
            }]
        );
        assert!(v.symbols.is_empty()); // no export directory in this fixture
        assert_eq!(
            v.slots,
            [
                ImportSlot {
                    slot_va: PE_BASE + 0x1060,
                    name: "KERNEL32.dll!ExitProcess".into()
                },
                ImportSlot {
                    slot_va: PE_BASE + 0x1068,
                    name: "KERNEL32.dll!#7".into()
                },
            ]
        );

        // VA→offset: entry maps to .text's file bytes; below-base and
        // unmapped VAs do not.
        assert_eq!(pe.va_to_offset(PE_BASE + 0x1000), Some(0x200));
        assert_eq!(pe.va_to_offset(PE_BASE - 1), None);
        assert_eq!(pe.va_to_offset(PE_BASE + 0x9000), None);
        assert_eq!(pe.bytes().len(), img.len());
    }

    #[test]
    fn pe_exports_become_symbols_and_entry_points() {
        let mut img = with_exports();
        // The base fixture's import data directory (index 1) points into
        // .text, which this fixture overwrote with export data; drop it
        // so the eager import parse sees "no import directory".
        let dirs = 0x80 + 4 + 20 + 112;
        img[dirs + 8..dirs + 16].fill(0);
        let pe = PeImage::parse(&img).unwrap();
        let v = describe(&pe);

        // Named RVA export only: the ordinal-only export has no name and
        // the forwarder has no VA in this image.
        assert_eq!(
            v.symbols,
            [Symbol {
                name: "Alpha".into(),
                va: PE_BASE + 0x1111,
                kind: SymbolKind::Function,
            }]
        );
        // Entry point plus both RVA exports (named or not), sorted.
        assert_eq!(
            v.entries,
            [PE_BASE + 0x1000, PE_BASE + 0x1111, PE_BASE + 0x2222]
        );
    }

    #[test]
    fn elf_image_view_is_faithful() {
        let img = synthetic_elf64();
        let elf = ElfImage::parse(&img).unwrap();
        let v = describe(&elf);

        assert_eq!(v.arch, Arch::X86_64);
        assert_eq!(v.entries, [ELF_TEXT]);
        assert_eq!(
            v.regions,
            [Region {
                name: ".text".into(),
                va: ELF_TEXT,
                size: 0x80,
                perms: Perms {
                    r: true,
                    w: false,
                    x: true
                },
            }]
        );
        assert_eq!(
            v.symbols,
            [
                Symbol {
                    name: "main".into(),
                    va: ELF_TEXT,
                    kind: SymbolKind::Function
                },
                Symbol {
                    name: "helper".into(),
                    va: ELF_TEXT + 0x20,
                    kind: SymbolKind::Function
                },
            ]
        );
        assert!(v.slots.is_empty());

        assert_eq!(elf.va_to_offset(ELF_TEXT), Some(0x100));
        assert_eq!(elf.va_to_offset(0x90_0000), None);
    }

    #[test]
    fn elf_plt_imports_become_import_slots() {
        let img = synthetic_dynamic_elf64();
        let elf = ElfImage::parse(&img).unwrap();
        let v = describe(&elf);

        assert_eq!(
            v.slots,
            [
                ImportSlot {
                    slot_va: ELF_DYN_BASE + 0x3018,
                    name: "read".into()
                },
                ImportSlot {
                    slot_va: ELF_DYN_BASE + 0x3020,
                    name: "write".into()
                },
                ImportSlot {
                    slot_va: ELF_DYN_BASE + 0x3028,
                    name: "close".into()
                },
            ]
        );
        // No e_entry, and every dynsym function is undefined (imported):
        // nothing qualifies as an entry point or a defined symbol.
        assert!(v.entries.is_empty());
        assert!(v.symbols.is_empty());
    }

    #[test]
    fn macho_image_view_is_faithful() {
        let img = synthetic_macho64();
        let mach = MachImage::parse(&img).unwrap();
        let v = describe(&mach);

        assert_eq!(v.arch, Arch::Aarch64);
        // LC_MAIN's file offset resolved through __TEXT (fileoff 0).
        assert_eq!(v.entries, [MACH_TEXT + MACH_ENTRYOFF]);
        assert_eq!(
            v.regions,
            [Region {
                name: "__TEXT".into(),
                va: MACH_TEXT,
                size: 0x4000,
                perms: Perms {
                    r: true,
                    w: false,
                    x: true
                },
            }]
        );
        // Defined symbols only; both live in the executable __TEXT
        // segment, so both classify as functions. _printf is undefined.
        assert_eq!(
            v.symbols,
            [
                Symbol {
                    name: "_main".into(),
                    va: MACH_TEXT + MACH_ENTRYOFF,
                    kind: SymbolKind::Function
                },
                Symbol {
                    name: "_helper".into(),
                    va: MACH_TEXT + MACH_ENTRYOFF + 0x20,
                    kind: SymbolKind::Function
                },
            ]
        );
        assert!(v.slots.is_empty()); // bind opcodes deferred

        assert_eq!(
            mach.va_to_offset(MACH_TEXT + MACH_ENTRYOFF),
            Some(MACH_ENTRYOFF as usize)
        );
        assert_eq!(mach.va_to_offset(0), None);
    }

    #[test]
    fn load_sniffs_every_format_and_rejects_garbage() {
        assert_eq!(load(&with_imports()).unwrap().arch(), Arch::X86_64);
        assert_eq!(load(&synthetic_elf64()).unwrap().arch(), Arch::X86_64);
        assert_eq!(load(&synthetic_macho64()).unwrap().arch(), Arch::Aarch64);

        for garbage in [&b""[..], b"\x7f", b"garbage bytes", &[0u8; 64]] {
            let Err(err) = load(garbage) else {
                panic!("{garbage:02x?}: garbage must not load");
            };
            assert!(matches!(err, ParseError::Unsupported(_)), "{err}");
        }
        // Recognized magic, malformed body: the parser's error, not a
        // sniffing error.
        let mut bad_pe = with_imports();
        bad_pe.truncate(0x40);
        assert!(load(&bad_pe).is_err());
    }

    #[test]
    fn decoders_agree_through_the_trait_object() {
        struct Case {
            dec: &'static dyn Decoder,
            call: &'static [u8],
            call_len: u8,
            ret: &'static [u8],
            ret_len: u8,
        }
        let x86: &'static dyn Decoder = &X86Decoder;
        let a64: &'static dyn Decoder = &Aarch64Decoder;
        let cases = [
            // call rel32 (+0) then ret.
            Case {
                dec: x86,
                call: &[0xE8, 0, 0, 0, 0],
                call_len: 5,
                ret: &[0xC3],
                ret_len: 1,
            },
            // bl #+4 (0x94000001) then ret (0xD65F03C0), little-endian.
            Case {
                dec: a64,
                call: &[0x01, 0x00, 0x00, 0x94],
                call_len: 4,
                ret: &[0xC0, 0x03, 0x5F, 0xD6],
                ret_len: 4,
            },
        ];
        for c in cases {
            let d = c.dec.decode_flow(c.call, 0x1000).unwrap();
            assert_eq!(d.length, c.call_len);
            assert_eq!(d.flow, Flow::Call(0x1000 + c.call_len as u64));

            let d = c.dec.decode_flow(c.ret, 0x2000).unwrap();
            assert_eq!(d.length, c.ret_len);
            assert_eq!(d.flow, Flow::Return);
        }

        // x86 unknown opcode: an error with no length — recursive descent
        // treats it as a block terminator.
        assert!(matches!(
            x86.decode_flow(&[0xD8, 0x00], 0).unwrap_err(),
            ParseError::Unsupported(_)
        ));
        // Truncated input is EOF on both ISAs.
        assert!(matches!(
            x86.decode_flow(&[], 0).unwrap_err(),
            ParseError::UnexpectedEof { .. }
        ));
        assert!(matches!(
            a64.decode_flow(&[0xC0, 0x03], 0).unwrap_err(),
            ParseError::UnexpectedEof { .. }
        ));
    }

    #[test]
    fn arch_selects_a_decoder() {
        assert!(decoder_for(Arch::X86_64).is_some());
        assert!(decoder_for(Arch::Aarch64).is_some());
        assert!(decoder_for(Arch::Other).is_none());
    }

    #[test]
    fn analysis_is_format_and_isa_blind_end_to_end() {
        // Plant a `ret` at each fixture's entry point, then run ONE loop —
        // load, find entry, map to bytes, pick decoder, classify — that
        // never names a format or an ISA.
        let mut pe = synthetic_pe64();
        pe[0x200] = 0xC3; // entry RVA 0x1000 -> file 0x200 (x86 ret)

        let mut elf = synthetic_elf64();
        elf[0x100] = 0xC3; // e_entry 0x40_1000 -> file 0x100

        let mut mach = synthetic_macho64();
        mach[MACH_ENTRYOFF as usize..MACH_ENTRYOFF as usize + 4]
            .copy_from_slice(&0xD65F_03C0u32.to_le_bytes()); // A64 ret

        for buf in [&pe, &elf, &mach] {
            let img = load(buf).unwrap();
            let v = describe(img.as_ref());
            let entry = v.entries[0];
            assert!(
                v.regions.iter().any(|r| r.perms.x
                    && entry >= r.va
                    && entry - r.va < r.size),
                "entry {entry:#x} must land in an executable region"
            );
            let off = img.va_to_offset(entry).unwrap();
            let dec = decoder_for(v.arch).unwrap();
            let d = dec.decode_flow(&img.bytes()[off..], entry).unwrap();
            assert_eq!(d.flow, Flow::Return);
        }
    }
}
