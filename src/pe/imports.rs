//! PE import table parsing.
//!
//! Walks the import directory (`IMAGE_DIRECTORY_ENTRY_IMPORT`): an array of
//! import descriptors, one per DLL, each pointing at an import lookup table
//! (ILT) whose entries name a symbol either by ordinal or by hint/name, and
//! an import address table (IAT) the loader patches at runtime.

use crate::error::{ParseError, Result};
use crate::reader::Reader;

use super::{PeFile, PeFormat, directory_index};

/// Each import descriptor is five u32 fields.
const DESCRIPTOR_SIZE: usize = 20;

/// Hard cap on descriptors/thunks so a crafted image can't make us loop over
/// gigabytes of "imports". Real binaries import a few hundred DLLs at most.
/// Shared with the delay-load parser, which walks a parallel structure.
pub(crate) const MAX_DLLS: usize = 4096;
const MAX_FUNCTIONS_PER_DLL: usize = 65536;

/// How one imported symbol is identified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportSymbol {
    /// Import by hint/name: the hint is an index into the exporting DLL's
    /// name table, used as a lookup accelerator.
    Name { hint: u16, name: String },
    /// Import by ordinal only.
    Ordinal(u16),
}

/// One imported function and the IAT slot the loader binds it through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedFunction {
    pub symbol: ImportSymbol,
    /// RVA of this function's IAT entry (what call sites indirect through).
    pub iat_rva: u32,
}

/// All imports from a single DLL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedDll {
    pub name: String,
    pub functions: Vec<ImportedFunction>,
}

impl PeFile {
    /// Parse the import table. `data` must be the same buffer this file was
    /// parsed from. Returns an empty list if the image has no import directory.
    pub fn imports(&self, data: &[u8]) -> Result<Vec<ImportedDll>> {
        let Some(dir) = self.optional.data_directories.get(directory_index::IMPORT) else {
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
                    "import table has more than {MAX_DLLS} descriptors"
                )));
            }

            let mut r = Reader::new(data);
            r.seek(self.rva_to_offset(desc_rva)?)?;
            let ilt_rva = r.u32()?; // OriginalFirstThunk
            let _time_date_stamp = r.u32()?;
            let _forwarder_chain = r.u32()?;
            let name_rva = r.u32()?;
            let iat_rva = r.u32()?; // FirstThunk

            // The descriptor array is terminated by an all-zero entry.
            if ilt_rva == 0 && name_rva == 0 && iat_rva == 0 {
                break;
            }

            let name = self.cstr_at_rva(data, name_rva)?;
            // Some linkers emit a null OriginalFirstThunk; the IAT then holds
            // the pre-bind lookup values and serves as the ILT.
            let lookup_rva = if ilt_rva != 0 { ilt_rva } else { iat_rva };
            let functions = self.parse_thunks(data, lookup_rva, iat_rva)?;

            dlls.push(ImportedDll { name, functions });
            // Checked: a descriptor array near the top of the 4 GiB RVA
            // space must fail cleanly, not overflow.
            desc_rva = desc_rva.checked_add(DESCRIPTOR_SIZE as u32).ok_or_else(|| {
                ParseError::Unsupported("import descriptor table overruns the RVA space".into())
            })?;
        }
        Ok(dlls)
    }

    /// Walk an import lookup table starting at `lookup_rva`, pairing each
    /// entry with its IAT slot starting at `iat_rva`. Shared with the
    /// delay-load parser: the delay import name table has the same entry
    /// format as the regular ILT.
    pub(crate) fn parse_thunks(
        &self,
        data: &[u8],
        lookup_rva: u32,
        iat_rva: u32,
    ) -> Result<Vec<ImportedFunction>> {
        let entry_size = match self.optional.format {
            PeFormat::Pe32 => 4,
            PeFormat::Pe32Plus => 8,
        };
        let mut r = Reader::new(data);
        r.seek(self.rva_to_offset(lookup_rva)?)?;

        let mut functions = Vec::new();
        loop {
            if functions.len() >= MAX_FUNCTIONS_PER_DLL {
                return Err(ParseError::Unsupported(format!(
                    "import lookup table has more than {MAX_FUNCTIONS_PER_DLL} entries"
                )));
            }

            // Widen PE32 thunks to u64 with the ordinal flag moved to bit 63
            // so both formats decode identically below.
            let thunk = match self.optional.format {
                PeFormat::Pe32 => {
                    let v = r.u32()? as u64;
                    (v & 0x7FFF_FFFF) | ((v & 0x8000_0000) << 32)
                }
                PeFormat::Pe32Plus => r.u64()?,
            };
            if thunk == 0 {
                break;
            }

            let index = functions.len() as u32;
            let symbol = if thunk & (1 << 63) != 0 {
                ImportSymbol::Ordinal(thunk as u16)
            } else {
                // Hint/name entry: u16 hint followed by a NUL-terminated name.
                let mut nr = Reader::new(data);
                nr.seek(self.rva_to_offset(thunk as u32)?)?;
                let hint = nr.u16()?;
                let name = nr.cstr()?;
                ImportSymbol::Name { hint, name }
            };
            functions.push(ImportedFunction {
                symbol,
                // Saturating: a crafted image can set `iat_rva` near the top of
                // the 32-bit RVA space, so slot `index` can run past u32::MAX.
                // Such a slot addresses no real byte; saturate to u32::MAX
                // rather than panicking (debug) or wrapping to a small bogus
                // RVA that could alias a genuine one (release).
                iat_rva: iat_rva.saturating_add(index.saturating_mul(entry_size)),
            });
        }
        Ok(functions)
    }

    /// Read a NUL-terminated string located at `rva`.
    fn cstr_at_rva(&self, data: &[u8], rva: u32) -> Result<String> {
        let mut r = Reader::new(data);
        r.seek(self.rva_to_offset(rva)?)?;
        r.cstr()
    }

    /// Read a NUL-terminated string at `rva`, refusing strings longer than
    /// `max_len` bytes so an image with a missing terminator can't hand us
    /// (or make us scan) the rest of the file as a "name". Shared with the
    /// export and delay-load parsers.
    pub(crate) fn cstr_at_rva_bounded(
        &self,
        data: &[u8],
        rva: u32,
        max_len: usize,
    ) -> Result<String> {
        let offset = self.rva_to_offset(rva)?;
        if offset >= data.len() {
            return Err(ParseError::UnexpectedEof {
                offset,
                needed: 1,
                available: 0,
            });
        }
        let end = data.len().min(offset.saturating_add(max_len).saturating_add(1));
        let window = &data[offset..end];
        match window.iter().position(|&b| b == 0) {
            Some(len) => Ok(String::from_utf8_lossy(&window[..len]).into_owned()),
            None if window.len() > max_len => Err(ParseError::Unsupported(format!(
                "string at RVA {rva:#010x} exceeds {max_len} bytes"
            ))),
            None => Err(ParseError::UnexpectedEof {
                offset,
                needed: window.len() + 1,
                available: window.len(),
            }),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::tests::synthetic_pe64;
    use super::*;

    /// Lay an import directory into the synthetic image's .text section
    /// (RVA 0x1000 maps to file offset 0x200). Shared with the
    /// trait-layer tests in [`crate::model`].
    pub(crate) fn with_imports() -> Vec<u8> {
        let mut img = synthetic_pe64();
        let base = 0x200; // file offset of RVA 0x1000

        let put32 = |img: &mut Vec<u8>, off: usize, v: u32| {
            img[base + off..base + off + 4].copy_from_slice(&v.to_le_bytes());
        };
        let put64 = |img: &mut Vec<u8>, off: usize, v: u64| {
            img[base + off..base + off + 8].copy_from_slice(&v.to_le_bytes());
        };

        // Descriptor 0 at RVA 0x1000: KERNEL32.dll.
        put32(&mut img, 0x00, 0x1040); // OriginalFirstThunk (ILT)
        put32(&mut img, 0x0C, 0x1080); // Name
        put32(&mut img, 0x10, 0x1060); // FirstThunk (IAT)
        // Descriptor 1 at RVA 0x1014 is already all zeros: terminator.

        // ILT at RVA 0x1040: by-name entry, by-ordinal entry, terminator.
        put64(&mut img, 0x40, 0x1090); // -> hint/name at RVA 0x1090
        put64(&mut img, 0x48, (1 << 63) | 7); // ordinal #7
        // IAT at RVA 0x1060 mirrors the ILT before binding.
        put64(&mut img, 0x60, 0x1090);
        put64(&mut img, 0x68, (1 << 63) | 7);

        // DLL name at RVA 0x1080.
        img[base + 0x80..base + 0x8D].copy_from_slice(b"KERNEL32.dll\0");
        // Hint/name at RVA 0x1090: hint 0x0152, "ExitProcess".
        img[base + 0x90..base + 0x92].copy_from_slice(&0x0152u16.to_le_bytes());
        img[base + 0x92..base + 0x9E].copy_from_slice(b"ExitProcess\0");

        // Point the import data directory at the descriptor array.
        let dirs = 0x80 + 4 + 20 + 112; // e_lfanew + sig + COFF + offset of dirs
        img[dirs + 8..dirs + 12].copy_from_slice(&0x1000u32.to_le_bytes());
        img[dirs + 12..dirs + 16].copy_from_slice(&40u32.to_le_bytes());

        img
    }

    #[test]
    fn parses_imports_by_name_and_ordinal() {
        let img = with_imports();
        let pe = PeFile::parse(&img).unwrap();
        let dlls = pe.imports(&img).unwrap();

        assert_eq!(dlls.len(), 1);
        assert_eq!(dlls[0].name, "KERNEL32.dll");
        assert_eq!(
            dlls[0].functions,
            vec![
                ImportedFunction {
                    symbol: ImportSymbol::Name {
                        hint: 0x0152,
                        name: "ExitProcess".into()
                    },
                    iat_rva: 0x1060,
                },
                ImportedFunction {
                    symbol: ImportSymbol::Ordinal(7),
                    iat_rva: 0x1068,
                },
            ]
        );
    }

    #[test]
    fn no_import_directory_means_no_imports() {
        let img = synthetic_pe64();
        let mut img = img;
        // Zero the import directory entry set by the base fixture.
        let dirs = 0x80 + 4 + 20 + 112;
        img[dirs + 8..dirs + 16].fill(0);
        let pe = PeFile::parse(&img).unwrap();
        assert_eq!(pe.imports(&img).unwrap(), Vec::new());
    }

    #[test]
    fn null_ilt_falls_back_to_iat() {
        let mut img = with_imports();
        // Zero OriginalFirstThunk; parsing should walk the IAT instead.
        img[0x200..0x204].fill(0);
        let pe = PeFile::parse(&img).unwrap();
        let dlls = pe.imports(&img).unwrap();
        assert_eq!(dlls[0].functions.len(), 2);
        assert_eq!(
            dlls[0].functions[0].symbol,
            ImportSymbol::Name {
                hint: 0x0152,
                name: "ExitProcess".into()
            }
        );
    }

    #[test]
    fn dangling_name_rva_is_an_error() {
        let mut img = with_imports();
        // Point the DLL name RVA outside every section.
        img[0x20C..0x210].copy_from_slice(&0x9000u32.to_le_bytes());
        let pe = PeFile::parse(&img).unwrap();
        assert_eq!(pe.imports(&img), Err(ParseError::UnmappedRva(0x9000)));
    }

    #[test]
    fn a_top_of_space_iat_rva_saturates_instead_of_overflowing() {
        // A crafted image can place the IAT near u32::MAX; the per-slot RVA
        // of a later entry then runs past the 32-bit space. It must saturate,
        // not panic (debug) or wrap to a small bogus RVA (release).
        let img = with_imports();
        let pe = PeFile::parse(&img).unwrap();
        // Walk the valid ILT at RVA 0x1040 (two entries) but claim the IAT
        // starts four bytes below the top of the RVA space.
        let funcs = pe.parse_thunks(&img, 0x1040, u32::MAX - 4).unwrap();
        assert_eq!(funcs.len(), 2);
        assert_eq!(funcs[0].iat_rva, u32::MAX - 4);
        // Slot 1 = (u32::MAX - 4) + 8 overflows u32 → saturates to u32::MAX.
        assert_eq!(funcs[1].iat_rva, u32::MAX);
    }
}
