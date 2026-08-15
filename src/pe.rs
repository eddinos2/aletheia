//! PE/COFF image parser.
//!
//! Clean-room implementation from the public Microsoft PE format
//! specification ("PE Format", Microsoft Learn). Parses the DOS stub
//! header, COFF file header, optional header (PE32 and PE32+), the data
//! directories, and the section table, and maps RVAs to file offsets.

use crate::error::{ParseError, Result};
use crate::reader::Reader;

pub mod delay;
pub mod exports;
mod imports;
pub use imports::{ImportSymbol, ImportedDll, ImportedFunction};

pub const DOS_MAGIC: u16 = 0x5A4D; // "MZ"
pub const PE_SIGNATURE: u32 = 0x0000_4550; // "PE\0\0"
pub const OPTIONAL_MAGIC_PE32: u16 = 0x10B;
pub const OPTIONAL_MAGIC_PE32_PLUS: u16 = 0x20B;

const E_LFANEW_OFFSET: usize = 0x3C;
const SECTION_HEADER_SIZE: usize = 40;

/// Target machine, from the COFF file header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Machine {
    X86,
    X86_64,
    Arm,
    Arm64,
    /// A machine value we recognize the encoding of but do not model.
    Other(u16),
}

impl From<u16> for Machine {
    fn from(raw: u16) -> Self {
        match raw {
            0x014C => Machine::X86,
            0x8664 => Machine::X86_64,
            0x01C0 | 0x01C4 => Machine::Arm,
            0xAA64 => Machine::Arm64,
            other => Machine::Other(other),
        }
    }
}

/// COFF file header (immediately after the PE signature).
#[derive(Debug, Clone)]
pub struct CoffHeader {
    pub machine: Machine,
    pub number_of_sections: u16,
    pub time_date_stamp: u32,
    pub size_of_optional_header: u16,
    pub characteristics: u16,
}

/// Whether the optional header is PE32 (32-bit) or PE32+ (64-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeFormat {
    Pe32,
    Pe32Plus,
}

/// One entry in the optional header's data directory table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataDirectory {
    pub rva: u32,
    pub size: u32,
}

/// Well-known data directory indices (PE spec §Optional Header Data Directories).
pub mod directory_index {
    pub const EXPORT: usize = 0;
    pub const IMPORT: usize = 1;
    pub const RESOURCE: usize = 2;
    pub const EXCEPTION: usize = 3;
    pub const BASE_RELOCATION: usize = 5;
    pub const DEBUG: usize = 6;
    pub const TLS: usize = 9;
    pub const IAT: usize = 12;
    pub const DELAY_IMPORT: usize = 13;
}

/// The optional header, normalized across PE32 and PE32+.
#[derive(Debug, Clone)]
pub struct OptionalHeader {
    pub format: PeFormat,
    pub entry_point_rva: u32,
    pub image_base: u64,
    pub section_alignment: u32,
    pub file_alignment: u32,
    pub size_of_image: u32,
    pub size_of_headers: u32,
    pub subsystem: u16,
    pub dll_characteristics: u16,
    pub data_directories: Vec<DataDirectory>,
}

/// A section table entry.
#[derive(Debug, Clone)]
pub struct Section {
    /// Section name with trailing NULs stripped (e.g. ".text").
    pub name: String,
    pub virtual_size: u32,
    pub virtual_address: u32,
    pub size_of_raw_data: u32,
    pub pointer_to_raw_data: u32,
    pub characteristics: u32,
}

impl Section {
    /// Whether `rva` falls inside this section's virtual range.
    ///
    /// The in-memory extent of a section is `virtual_size` rounded up to the
    /// section alignment, but bytes beyond `max(virtual_size, size_of_raw_data)`
    /// are zero-fill with no file backing, so only the raw-backed range is
    /// considered mappable.
    pub fn contains_rva(&self, rva: u32) -> bool {
        let extent = self.virtual_size.max(self.size_of_raw_data);
        rva >= self.virtual_address && rva - self.virtual_address < extent
    }
}

/// A parsed PE image.
#[derive(Debug, Clone)]
pub struct PeFile {
    pub coff: CoffHeader,
    pub optional: OptionalHeader,
    pub sections: Vec<Section>,
}

impl PeFile {
    /// Parse a PE image from raw file bytes.
    pub fn parse(data: &[u8]) -> Result<PeFile> {
        let mut r = Reader::new(data);

        // DOS header: just the magic and the offset to the PE header.
        let dos_magic = r.u16()?;
        if dos_magic != DOS_MAGIC {
            return Err(ParseError::BadSignature {
                what: "DOS (MZ)",
                offset: 0,
                found: dos_magic as u32,
                expected: DOS_MAGIC as u32,
            });
        }
        r.seek(E_LFANEW_OFFSET)?;
        let e_lfanew = r.u32()? as usize;

        r.seek(e_lfanew)?;
        let signature = r.u32()?;
        if signature != PE_SIGNATURE {
            return Err(ParseError::BadSignature {
                what: "PE",
                offset: e_lfanew,
                found: signature,
                expected: PE_SIGNATURE,
            });
        }

        let coff = parse_coff_header(&mut r)?;
        let optional_start = r.pos();
        let optional = parse_optional_header(&mut r)?;

        // The section table begins right after the optional header, whose
        // declared size is authoritative (it may include padding we didn't read).
        r.seek(optional_start + coff.size_of_optional_header as usize)?;
        let mut sections = Vec::with_capacity(coff.number_of_sections as usize);
        for _ in 0..coff.number_of_sections {
            sections.push(parse_section_header(&mut r)?);
        }

        Ok(PeFile {
            coff,
            optional,
            sections,
        })
    }

    /// Map an RVA to a file offset via the section table.
    pub fn rva_to_offset(&self, rva: u32) -> Result<usize> {
        // RVAs inside the headers (before the first section) map 1:1.
        if rva < self.optional.size_of_headers {
            return Ok(rva as usize);
        }
        self.sections
            .iter()
            .find(|s| s.contains_rva(rva))
            // Checked: `rva - virtual_address` cannot underflow (contains_rva
            // guarantees rva >= virtual_address), but a section whose raw
            // pointer plus the in-section delta overruns the 32-bit
            // file-offset space addresses no real byte, so it maps nowhere
            // rather than wrapping to a bogus offset.
            .and_then(|s| (rva - s.virtual_address).checked_add(s.pointer_to_raw_data))
            .map(|off| off as usize)
            .ok_or(ParseError::UnmappedRva(rva))
    }

    /// The section containing `rva`, if any.
    pub fn section_for_rva(&self, rva: u32) -> Option<&Section> {
        self.sections.iter().find(|s| s.contains_rva(rva))
    }
}

fn parse_coff_header(r: &mut Reader) -> Result<CoffHeader> {
    let machine = Machine::from(r.u16()?);
    let number_of_sections = r.u16()?;
    let time_date_stamp = r.u32()?;
    let _pointer_to_symbol_table = r.u32()?;
    let _number_of_symbols = r.u32()?;
    let size_of_optional_header = r.u16()?;
    let characteristics = r.u16()?;
    Ok(CoffHeader {
        machine,
        number_of_sections,
        time_date_stamp,
        size_of_optional_header,
        characteristics,
    })
}

fn parse_optional_header(r: &mut Reader) -> Result<OptionalHeader> {
    let magic_offset = r.pos();
    let magic = r.u16()?;
    let format = match magic {
        OPTIONAL_MAGIC_PE32 => PeFormat::Pe32,
        OPTIONAL_MAGIC_PE32_PLUS => PeFormat::Pe32Plus,
        other => {
            return Err(ParseError::BadSignature {
                what: "optional header",
                offset: magic_offset,
                found: other as u32,
                expected: OPTIONAL_MAGIC_PE32_PLUS as u32,
            });
        }
    };

    let _linker_version = r.u16()?;
    let _size_of_code = r.u32()?;
    let _size_of_initialized_data = r.u32()?;
    let _size_of_uninitialized_data = r.u32()?;
    let entry_point_rva = r.u32()?;
    let _base_of_code = r.u32()?;

    // PE32 has BaseOfData (u32) then ImageBase (u32); PE32+ drops BaseOfData
    // and widens ImageBase to u64.
    let image_base = match format {
        PeFormat::Pe32 => {
            let _base_of_data = r.u32()?;
            r.u32()? as u64
        }
        PeFormat::Pe32Plus => r.u64()?,
    };

    let section_alignment = r.u32()?;
    let file_alignment = r.u32()?;
    let _os_version = r.u32()?;
    let _image_version = r.u32()?;
    let _subsystem_version = r.u32()?;
    let _win32_version = r.u32()?;
    let size_of_image = r.u32()?;
    let size_of_headers = r.u32()?;
    let _checksum = r.u32()?;
    let subsystem = r.u16()?;
    let dll_characteristics = r.u16()?;

    // Stack/heap reserve+commit are pointer-sized.
    match format {
        PeFormat::Pe32 => r.skip(4 * 4)?,
        PeFormat::Pe32Plus => r.skip(8 * 4)?,
    }
    let _loader_flags = r.u32()?;

    let number_of_rva_and_sizes = r.u32()?;
    // The spec fixes the table at 16 entries; tolerate fewer, reject absurd
    // counts that would make us read megabytes of "directories".
    if number_of_rva_and_sizes > 16 {
        return Err(ParseError::Unsupported(format!(
            "NumberOfRvaAndSizes is {number_of_rva_and_sizes}, expected at most 16"
        )));
    }
    let mut data_directories = Vec::with_capacity(number_of_rva_and_sizes as usize);
    for _ in 0..number_of_rva_and_sizes {
        data_directories.push(DataDirectory {
            rva: r.u32()?,
            size: r.u32()?,
        });
    }

    Ok(OptionalHeader {
        format,
        entry_point_rva,
        image_base,
        section_alignment,
        file_alignment,
        size_of_image,
        size_of_headers,
        subsystem,
        dll_characteristics,
        data_directories,
    })
}

fn parse_section_header(r: &mut Reader) -> Result<Section> {
    let start = r.pos();
    let name_bytes = r.take(8)?;
    let name_len = name_bytes.iter().position(|&b| b == 0).unwrap_or(8);
    let name = String::from_utf8_lossy(&name_bytes[..name_len]).into_owned();

    let virtual_size = r.u32()?;
    let virtual_address = r.u32()?;
    let size_of_raw_data = r.u32()?;
    let pointer_to_raw_data = r.u32()?;
    // Relocations, line numbers (deprecated), and their counts.
    r.seek(start + SECTION_HEADER_SIZE - 4)?;
    let characteristics = r.u32()?;

    Ok(Section {
        name,
        virtual_size,
        virtual_address,
        size_of_raw_data,
        pointer_to_raw_data,
        characteristics,
    })
}

/// One `RUNTIME_FUNCTION` record in the exception directory is three
/// u32 fields on both x64 and ARM64: the function's begin RVA, its end
/// RVA (or, for ARM64 packed unwind, a packed word), and the unwind
/// info. Only the begin RVA is needed to locate a function start.
const RUNTIME_FUNCTION_SIZE: usize = 12;

/// Cap on exception-directory records parsed, so a crafted directory
/// size can neither over-allocate nor drive a long loop. Real images
/// have at most a few hundred thousand functions.
const MAX_PDATA_FUNCTIONS: usize = 1 << 20;

impl PeFile {
    /// Function-start RVAs declared in the exception directory
    /// (`.pdata`), sorted and deduplicated.
    ///
    /// The exception directory (optional-header data directory index 3)
    /// is an array of `RUNTIME_FUNCTION` records present on x64 and
    /// ARM64 images; its first field is a function's begin RVA. This is
    /// an authoritative, compiler-emitted list of function starts, so it
    /// is the highest-precision discovery source on Windows binaries.
    /// Returns empty when the directory is absent or empty. Never
    /// panics: a malformed or truncated table simply yields fewer
    /// entries.
    pub fn pdata_function_starts(&self, data: &[u8]) -> Vec<u32> {
        let Some(dir) = self.optional.data_directories.get(directory_index::EXCEPTION) else {
            return Vec::new();
        };
        if dir.rva == 0 || dir.size == 0 {
            return Vec::new();
        }
        let count = (dir.size as usize / RUNTIME_FUNCTION_SIZE).min(MAX_PDATA_FUNCTIONS);
        let Ok(base_off) = self.rva_to_offset(dir.rva) else {
            return Vec::new();
        };
        let mut r = Reader::new(data);
        if r.seek(base_off).is_err() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for _ in 0..count {
            let Ok(begin) = r.u32() else { break };
            // Skip the end/packed and unwind-info fields.
            if r.skip(RUNTIME_FUNCTION_SIZE - 4).is_err() {
                break;
            }
            if begin != 0 {
                out.push(begin);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Re-exported for the trait-layer tests in [`crate::model`]
    /// (`imports` is a private module, so its fixture is surfaced here).
    pub(crate) use super::imports::tests::with_imports;

    /// Build a minimal but well-formed PE32+ image with one .text section.
    /// Shared with submodule tests (e.g. imports) as a base fixture.
    pub(crate) fn synthetic_pe64() -> Vec<u8> {
        let e_lfanew: u32 = 0x80;
        let mut img = vec![0u8; 0x400];

        // DOS header.
        img[0..2].copy_from_slice(&DOS_MAGIC.to_le_bytes());
        img[0x3C..0x40].copy_from_slice(&e_lfanew.to_le_bytes());

        let pe = e_lfanew as usize;
        img[pe..pe + 4].copy_from_slice(&PE_SIGNATURE.to_le_bytes());

        // COFF header.
        let coff = pe + 4;
        img[coff..coff + 2].copy_from_slice(&0x8664u16.to_le_bytes()); // machine: x86-64
        img[coff + 2..coff + 4].copy_from_slice(&1u16.to_le_bytes()); // 1 section
        // size_of_optional_header: 0xF0 (standard for PE32+ with 16 dirs)
        img[coff + 16..coff + 18].copy_from_slice(&0xF0u16.to_le_bytes());
        img[coff + 18..coff + 20].copy_from_slice(&0x0022u16.to_le_bytes()); // exe | large-address-aware

        // Optional header (PE32+).
        let opt = coff + 20;
        img[opt..opt + 2].copy_from_slice(&OPTIONAL_MAGIC_PE32_PLUS.to_le_bytes());
        img[opt + 16..opt + 20].copy_from_slice(&0x1000u32.to_le_bytes()); // entry point RVA
        img[opt + 24..opt + 32].copy_from_slice(&0x1_4000_0000u64.to_le_bytes()); // image base
        img[opt + 32..opt + 36].copy_from_slice(&0x1000u32.to_le_bytes()); // section alignment
        img[opt + 36..opt + 40].copy_from_slice(&0x200u32.to_le_bytes()); // file alignment
        img[opt + 56..opt + 60].copy_from_slice(&0x2000u32.to_le_bytes()); // size of image
        img[opt + 60..opt + 64].copy_from_slice(&0x200u32.to_le_bytes()); // size of headers
        img[opt + 68..opt + 70].copy_from_slice(&3u16.to_le_bytes()); // subsystem: console
        img[opt + 108..opt + 112].copy_from_slice(&16u32.to_le_bytes()); // NumberOfRvaAndSizes
        // Import directory (index 1) points into .text for the test.
        let dirs = opt + 112;
        img[dirs + 8..dirs + 12].copy_from_slice(&0x1010u32.to_le_bytes());
        img[dirs + 12..dirs + 16].copy_from_slice(&0x40u32.to_le_bytes());

        // Section table: one ".text" section, RVA 0x1000 -> file 0x200.
        let sec = opt + 0xF0;
        img[sec..sec + 5].copy_from_slice(b".text");
        img[sec + 8..sec + 12].copy_from_slice(&0x100u32.to_le_bytes()); // virtual size
        img[sec + 12..sec + 16].copy_from_slice(&0x1000u32.to_le_bytes()); // virtual address
        img[sec + 16..sec + 20].copy_from_slice(&0x200u32.to_le_bytes()); // size of raw data
        img[sec + 20..sec + 24].copy_from_slice(&0x200u32.to_le_bytes()); // pointer to raw data
        img[sec + 36..sec + 40].copy_from_slice(&0x6000_0020u32.to_le_bytes()); // code|exec|read

        img
    }

    /// Place `n` RUNTIME_FUNCTION records at RVA 0x1060 (file 0x260) and
    /// point the exception directory (index 3) at them.
    fn with_pdata(begins: &[u32]) -> Vec<u8> {
        let mut img = synthetic_pe64();
        let dirs = 0x80 + 4 + 20 + 112;
        let exc = dirs + 3 * 8;
        img[exc..exc + 4].copy_from_slice(&0x1060u32.to_le_bytes());
        img[exc + 4..exc + 8].copy_from_slice(&(begins.len() as u32 * 12).to_le_bytes());
        for (i, &begin) in begins.iter().enumerate() {
            let rec = 0x260 + i * 12;
            img[rec..rec + 4].copy_from_slice(&begin.to_le_bytes());
            img[rec + 4..rec + 8].copy_from_slice(&(begin + 0x10).to_le_bytes()); // end
        }
        img
    }

    #[test]
    fn pdata_function_starts_from_exception_directory() {
        let img = with_pdata(&[0x1040, 0x1000, 0x1000]); // unsorted + duplicate
        let pe = PeFile::parse(&img).unwrap();
        // Sorted, deduplicated RVAs.
        assert_eq!(pe.pdata_function_starts(&img), vec![0x1000, 0x1040]);
    }

    #[test]
    fn pdata_absent_or_hostile_size_never_panics() {
        // No exception directory → empty.
        let img = synthetic_pe64();
        assert!(PeFile::parse(&img).unwrap().pdata_function_starts(&img).is_empty());

        // A wildly oversized directory must be bounded by the buffer, not
        // over-read or panic: only the real (nonzero) records come back.
        let mut img = with_pdata(&[0x1000, 0x1040]);
        let exc = (0x80 + 4 + 20 + 112) + 3 * 8;
        img[exc + 4..exc + 8].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        let pe = PeFile::parse(&img).unwrap();
        assert_eq!(pe.pdata_function_starts(&img), vec![0x1000, 0x1040]);
    }

    #[test]
    fn parses_synthetic_pe64() {
        let img = synthetic_pe64();
        let pe = PeFile::parse(&img).unwrap();

        assert_eq!(pe.coff.machine, Machine::X86_64);
        assert_eq!(pe.coff.number_of_sections, 1);
        assert_eq!(pe.optional.format, PeFormat::Pe32Plus);
        assert_eq!(pe.optional.entry_point_rva, 0x1000);
        assert_eq!(pe.optional.image_base, 0x1_4000_0000);
        assert_eq!(pe.optional.data_directories.len(), 16);
        assert_eq!(
            pe.optional.data_directories[directory_index::IMPORT],
            DataDirectory {
                rva: 0x1010,
                size: 0x40
            }
        );

        assert_eq!(pe.sections.len(), 1);
        let text = &pe.sections[0];
        assert_eq!(text.name, ".text");
        assert_eq!(text.virtual_address, 0x1000);
    }

    #[test]
    fn maps_rvas_through_sections() {
        let pe = PeFile::parse(&synthetic_pe64()).unwrap();
        // Entry point: RVA 0x1000 is the start of .text at file offset 0x200.
        assert_eq!(pe.rva_to_offset(0x1000).unwrap(), 0x200);
        assert_eq!(pe.rva_to_offset(0x1010).unwrap(), 0x210);
        // Header RVAs map 1:1.
        assert_eq!(pe.rva_to_offset(0x40).unwrap(), 0x40);
        // Past every section: unmappable.
        assert_eq!(pe.rva_to_offset(0x9000), Err(ParseError::UnmappedRva(0x9000)));
    }

    #[test]
    fn rva_offset_that_would_overflow_the_file_pointer_maps_nowhere() {
        // A section whose raw-data pointer plus the in-section delta overruns
        // the 32-bit file-offset space addresses no real byte. The mapping
        // must fail cleanly with UnmappedRva rather than wrapping (release) or
        // panicking (debug) on the overflowing add.
        let mut pe = PeFile::parse(&synthetic_pe64()).unwrap();
        pe.sections[0].pointer_to_raw_data = u32::MAX;
        assert_eq!(pe.sections[0].virtual_address, 0x1000);
        // Delta 1 over the section start: 1 + u32::MAX overflows u32.
        assert_eq!(pe.rva_to_offset(0x1001), Err(ParseError::UnmappedRva(0x1001)));
    }

    #[test]
    fn rejects_bad_dos_magic() {
        let mut img = synthetic_pe64();
        img[0] = b'X';
        let err = PeFile::parse(&img).unwrap_err();
        assert!(matches!(err, ParseError::BadSignature { what: "DOS (MZ)", .. }));
    }

    #[test]
    fn rejects_bad_pe_signature() {
        let mut img = synthetic_pe64();
        img[0x80] = 0;
        let err = PeFile::parse(&img).unwrap_err();
        assert!(matches!(err, ParseError::BadSignature { what: "PE", .. }));
    }

    #[test]
    fn truncated_image_is_an_error_not_a_panic() {
        let img = synthetic_pe64();
        for len in [0, 1, 0x40, 0x84, 0x98, 0x100] {
            assert!(PeFile::parse(&img[..len]).is_err(), "len {len:#x}");
        }
    }
}
