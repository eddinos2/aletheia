//! PE delay-load import table parsing.
//!
//! Walks the delay-load directory (`IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT`):
//! an array of `IMAGE_DELAYLOAD_DESCRIPTOR` entries terminated by an
//! all-zero entry, one per lazily-bound DLL. Each descriptor names its DLL
//! and points at a delay import address table (patched on first call) and a
//! delay import name table, which has the same entry format as the regular
//! import lookup table and is parsed the same way.

use crate::error::{ParseError, Result};
use crate::reader::Reader;

use super::imports::MAX_DLLS;
use super::{ImportedFunction, PeFile, directory_index};

/// Each delay-load descriptor is eight u32 fields.
const DESCRIPTOR_SIZE: usize = 32;

/// Longest accepted DLL name, in bytes.
const MAX_NAME_LEN: usize = 4096;

/// All delay-loaded imports from a single DLL.
///
/// The delay-load analogue of [`ImportedDll`](super::ImportedDll): the
/// loader does not bind these at load time; the first call through a delay
/// IAT slot resolves the DLL and patches the slot instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelayImportedDll {
    pub name: String,
    /// Raw `Attributes` field. Bit 0 (`RvaBased`) is set on every modern
    /// image; descriptors with it clear are rejected during parsing.
    pub attributes: u32,
    /// RVA of the delay import address table.
    pub iat_rva: u32,
    /// RVA of the delay import name table (the delay-load lookup table).
    pub int_rva: u32,
    pub functions: Vec<ImportedFunction>,
}

impl PeFile {
    /// Parse the delay-load import table. `data` must be the same buffer
    /// this file was parsed from. Returns an empty list if the image has no
    /// delay-load directory.
    ///
    /// A descriptor whose `Attributes.RvaBased` bit is clear uses the legacy
    /// form whose fields are absolute virtual addresses rather than RVAs;
    /// those are rejected with [`ParseError::Unsupported`] instead of being
    /// misread as RVAs.
    pub fn delay_imports(&self, data: &[u8]) -> Result<Vec<DelayImportedDll>> {
        let Some(dir) = self
            .optional
            .data_directories
            .get(directory_index::DELAY_IMPORT)
        else {
            return Ok(Vec::new());
        };
        if dir.rva == 0 {
            return Ok(Vec::new());
        }

        let mut dlls = Vec::new();
        let mut desc_rva = dir.rva;
        loop {
            if dlls.len() >= MAX_DLLS {
                return Err(ParseError::Unsupported(format!(
                    "delay-load table has more than {MAX_DLLS} descriptors"
                )));
            }

            let mut r = Reader::new(data);
            r.seek(self.rva_to_offset(desc_rva)?)?;
            let attributes = r.u32()?;
            let name_rva = r.u32()?; // DllNameRVA
            let _module_handle_rva = r.u32()?;
            let iat_rva = r.u32()?; // ImportAddressTableRVA
            let int_rva = r.u32()?; // ImportNameTableRVA
            let _bound_iat_rva = r.u32()?;
            let _unload_info_rva = r.u32()?;
            let _time_date_stamp = r.u32()?;

            // The descriptor array is terminated by an all-zero entry.
            if attributes == 0 && name_rva == 0 && iat_rva == 0 && int_rva == 0 {
                break;
            }

            if attributes & 1 == 0 {
                return Err(ParseError::Unsupported(format!(
                    "delay-load descriptor at RVA {desc_rva:#010x} is not RvaBased (legacy VA-based form)"
                )));
            }

            let name = self.cstr_at_rva_bounded(data, name_rva, MAX_NAME_LEN)?;
            // The delay INT has the same entry format as the regular ILT
            // (by-ordinal flag in the top bit, else a hint/name RVA), with
            // the same PE32 vs PE32+ entry width.
            let functions = if int_rva != 0 {
                self.parse_thunks(data, int_rva, iat_rva)?
            } else {
                Vec::new()
            };

            dlls.push(DelayImportedDll {
                name,
                attributes,
                iat_rva,
                int_rva,
                functions,
            });
            // Checked: a descriptor array near the top of the 4 GiB RVA
            // space must fail cleanly, not overflow.
            desc_rva = desc_rva
                .checked_add(DESCRIPTOR_SIZE as u32)
                .ok_or_else(|| {
                    ParseError::Unsupported(
                        "delay-load descriptor table overruns the RVA space".into(),
                    )
                })?;
        }
        Ok(dlls)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::synthetic_pe64;
    use super::super::{
        DOS_MAGIC, ImportSymbol, OPTIONAL_MAGIC_PE32, PE_SIGNATURE, PeFile, PeFormat,
    };
    use super::*;

    /// File offset backing RVA 0x1000 (start of .text) in both fixtures.
    const BASE: usize = 0x200;

    /// Build a minimal PE32 (32-bit) sibling of `synthetic_pe64`: same
    /// e_lfanew, one .text section at RVA 0x1000 -> file 0x200.
    fn synthetic_pe32() -> Vec<u8> {
        let e_lfanew: u32 = 0x80;
        let mut img = vec![0u8; 0x400];

        img[0..2].copy_from_slice(&DOS_MAGIC.to_le_bytes());
        img[0x3C..0x40].copy_from_slice(&e_lfanew.to_le_bytes());

        let pe = e_lfanew as usize;
        img[pe..pe + 4].copy_from_slice(&PE_SIGNATURE.to_le_bytes());

        // COFF header.
        let coff = pe + 4;
        img[coff..coff + 2].copy_from_slice(&0x014Cu16.to_le_bytes()); // machine: x86
        img[coff + 2..coff + 4].copy_from_slice(&1u16.to_le_bytes()); // 1 section
        // size_of_optional_header: 0xE0 (standard for PE32 with 16 dirs)
        img[coff + 16..coff + 18].copy_from_slice(&0xE0u16.to_le_bytes());
        img[coff + 18..coff + 20].copy_from_slice(&0x0102u16.to_le_bytes()); // exe | 32-bit

        // Optional header (PE32).
        let opt = coff + 20;
        img[opt..opt + 2].copy_from_slice(&OPTIONAL_MAGIC_PE32.to_le_bytes());
        img[opt + 16..opt + 20].copy_from_slice(&0x1000u32.to_le_bytes()); // entry point RVA
        img[opt + 28..opt + 32].copy_from_slice(&0x0040_0000u32.to_le_bytes()); // image base
        img[opt + 32..opt + 36].copy_from_slice(&0x1000u32.to_le_bytes()); // section alignment
        img[opt + 36..opt + 40].copy_from_slice(&0x200u32.to_le_bytes()); // file alignment
        img[opt + 56..opt + 60].copy_from_slice(&0x2000u32.to_le_bytes()); // size of image
        img[opt + 60..opt + 64].copy_from_slice(&0x200u32.to_le_bytes()); // size of headers
        img[opt + 68..opt + 70].copy_from_slice(&3u16.to_le_bytes()); // subsystem: console
        img[opt + 92..opt + 96].copy_from_slice(&16u32.to_le_bytes()); // NumberOfRvaAndSizes

        // Section table: one ".text" section, RVA 0x1000 -> file 0x200.
        let sec = opt + 0xE0;
        img[sec..sec + 5].copy_from_slice(b".text");
        img[sec + 8..sec + 12].copy_from_slice(&0x100u32.to_le_bytes()); // virtual size
        img[sec + 12..sec + 16].copy_from_slice(&0x1000u32.to_le_bytes()); // virtual address
        img[sec + 16..sec + 20].copy_from_slice(&0x200u32.to_le_bytes()); // size of raw data
        img[sec + 20..sec + 24].copy_from_slice(&0x200u32.to_le_bytes()); // pointer to raw data
        img[sec + 36..sec + 40].copy_from_slice(&0x6000_0020u32.to_le_bytes()); // code|exec|read

        img
    }

    /// Lay a delay-load directory into .text (RVA 0x1000 -> file 0x200):
    /// one USER32.dll descriptor with a by-name and a by-ordinal entry,
    /// then the all-zero terminator. `dirs` is the file offset of the data
    /// directory table; `wide` selects PE32+ (u64) vs PE32 (u32) thunks.
    fn lay_delay(img: &mut [u8], dirs: usize, wide: bool) {
        let put32 = |img: &mut [u8], off: usize, v: u32| {
            img[BASE + off..BASE + off + 4].copy_from_slice(&v.to_le_bytes());
        };
        let put64 = |img: &mut [u8], off: usize, v: u64| {
            img[BASE + off..BASE + off + 8].copy_from_slice(&v.to_le_bytes());
        };

        // IMAGE_DELAYLOAD_DESCRIPTOR 0 at RVA 0x1000.
        put32(img, 0x00, 1); // Attributes: RvaBased
        put32(img, 0x04, 0x1080); // DllNameRVA
        put32(img, 0x08, 0x10F0); // ModuleHandleRVA
        put32(img, 0x0C, 0x1060); // ImportAddressTableRVA
        put32(img, 0x10, 0x1040); // ImportNameTableRVA
        // Descriptor 1 at RVA 0x1020 is already all zeros: terminator.

        // Delay INT at RVA 0x1040; the delay IAT at 0x1060 mirrors it
        // before the first call binds it.
        if wide {
            put64(img, 0x40, 0x1090);
            put64(img, 0x48, (1u64 << 63) | 7);
            put64(img, 0x60, 0x1090);
            put64(img, 0x68, (1u64 << 63) | 7);
        } else {
            put32(img, 0x40, 0x1090);
            put32(img, 0x44, 0x8000_0007);
            put32(img, 0x60, 0x1090);
            put32(img, 0x64, 0x8000_0007);
        }

        // DLL name and hint/name entry.
        img[BASE + 0x80..BASE + 0x8B].copy_from_slice(b"USER32.dll\0");
        img[BASE + 0x90..BASE + 0x92].copy_from_slice(&0x0010u16.to_le_bytes());
        img[BASE + 0x92..BASE + 0x9E].copy_from_slice(b"MessageBoxA\0");

        // Point the delay-load data directory (index 13) at the descriptors.
        let entry = dirs + directory_index::DELAY_IMPORT * 8;
        img[entry..entry + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        img[entry + 4..entry + 8].copy_from_slice(&0x40u32.to_le_bytes());
    }

    fn with_delay_pe64() -> Vec<u8> {
        let mut img = synthetic_pe64();
        lay_delay(&mut img, 0x80 + 4 + 20 + 112, true);
        img
    }

    fn with_delay_pe32() -> Vec<u8> {
        let mut img = synthetic_pe32();
        lay_delay(&mut img, 0x80 + 4 + 20 + 96, false);
        img
    }

    #[test]
    fn parses_pe32plus_delay_imports() {
        let img = with_delay_pe64();
        let pe = PeFile::parse(&img).unwrap();
        let dlls = pe.delay_imports(&img).unwrap();

        assert_eq!(
            dlls,
            vec![DelayImportedDll {
                name: "USER32.dll".into(),
                attributes: 1,
                iat_rva: 0x1060,
                int_rva: 0x1040,
                functions: vec![
                    ImportedFunction {
                        symbol: ImportSymbol::Name {
                            hint: 0x0010,
                            name: "MessageBoxA".into(),
                        },
                        iat_rva: 0x1060,
                    },
                    ImportedFunction {
                        symbol: ImportSymbol::Ordinal(7),
                        iat_rva: 0x1068,
                    },
                ],
            }]
        );
    }

    #[test]
    fn parses_pe32_delay_imports_with_narrow_thunks() {
        let img = with_delay_pe32();
        let pe = PeFile::parse(&img).unwrap();
        assert_eq!(pe.optional.format, PeFormat::Pe32);

        let dlls = pe.delay_imports(&img).unwrap();
        assert_eq!(dlls.len(), 1);
        assert_eq!(dlls[0].name, "USER32.dll");
        assert_eq!(
            dlls[0].functions,
            vec![
                ImportedFunction {
                    symbol: ImportSymbol::Name {
                        hint: 0x0010,
                        name: "MessageBoxA".into(),
                    },
                    iat_rva: 0x1060,
                },
                // PE32 IAT slots are 4 bytes wide, not 8.
                ImportedFunction {
                    symbol: ImportSymbol::Ordinal(7),
                    iat_rva: 0x1064,
                },
            ]
        );
    }

    #[test]
    fn no_delay_directory_means_no_delay_imports() {
        let img = synthetic_pe64();
        let pe = PeFile::parse(&img).unwrap();
        assert_eq!(pe.delay_imports(&img).unwrap(), Vec::new());
    }

    #[test]
    fn legacy_va_based_descriptor_is_rejected() {
        let mut img = with_delay_pe64();
        img[BASE..BASE + 4].fill(0); // clear Attributes.RvaBased
        let pe = PeFile::parse(&img).unwrap();
        let err = pe.delay_imports(&img).unwrap_err();
        assert!(matches!(err, ParseError::Unsupported(_)));
    }

    #[test]
    fn unterminated_descriptor_table_is_an_error_not_a_hang() {
        let mut img = with_delay_pe64();
        // Overwrite the all-zero terminator with garbage: the walk must
        // fail on the bogus descriptor instead of running away.
        img[BASE + 0x20..BASE + 0x40].fill(0xFF);
        let pe = PeFile::parse(&img).unwrap();
        assert_eq!(
            pe.delay_imports(&img),
            Err(ParseError::UnmappedRva(0xFFFF_FFFF))
        );
    }

    #[test]
    fn truncated_image_never_panics() {
        let img = with_delay_pe64();
        for len in 0..img.len() {
            if let Ok(pe) = PeFile::parse(&img[..len]) {
                let _ = pe.delay_imports(&img[..len]);
            }
        }
    }
}
