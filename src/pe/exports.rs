//! PE export table parsing.
//!
//! Walks the export directory (`IMAGE_DIRECTORY_ENTRY_EXPORT`): one
//! `IMAGE_EXPORT_DIRECTORY` giving the DLL's own name and ordinal base,
//! plus three tables — the Export Address Table (EAT, one slot per exported
//! ordinal), the Name Pointer Table, and the Ordinal Table (parallel arrays
//! that attach names to EAT slots).

use crate::error::{ParseError, Result};
use crate::reader::Reader;

use super::{PeFile, directory_index};

/// Hard caps on the attacker-controlled table counts so a crafted image
/// can't make us allocate or loop over gigabytes of "exports". Even the
/// largest real DLLs export a few tens of thousands of symbols.
const MAX_FUNCTIONS: usize = 65536;
const MAX_NAMES: usize = 65536;

/// Longest accepted DLL, export, or forwarder name, in bytes.
const MAX_NAME_LEN: usize = 4096;

/// Where an exported symbol resolves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportTarget {
    /// RVA of the exported code or data inside this image.
    Rva(u32),
    /// A forwarder: the loader resolves this export from another DLL. The
    /// string has the form `"TARGETDLL.FunctionName"` (or `"TARGETDLL.#42"`
    /// for a by-ordinal forward). An EAT entry is a forwarder exactly when
    /// its RVA lands inside the export data directory's own range.
    Forwarder(String),
}

/// One exported symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Export {
    /// Biased ordinal (EAT index plus the directory's ordinal base).
    pub ordinal: u32,
    /// Name from the name pointer table; `None` for ordinal-only exports.
    pub name: Option<String>,
    pub target: ExportTarget,
}

/// The parsed export directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportDirectory {
    /// The image's own name as recorded in the directory (e.g. "FOO.dll").
    pub dll_name: String,
    /// Ordinal number of EAT slot 0 (the `Base` field).
    pub ordinal_base: u32,
    /// Exports in EAT order. Zero (unused) EAT slots are omitted.
    pub exports: Vec<Export>,
}

impl PeFile {
    /// Parse the export directory. `data` must be the same buffer this file
    /// was parsed from. Returns `Ok(None)` if the image has no export
    /// directory.
    ///
    /// Malformed-input policy:
    /// - a name pointer whose string RVA cannot be mapped is an error
    ///   (matching how [`PeFile::imports`] treats dangling name RVAs);
    /// - an ordinal-table entry that indexes past the EAT is skipped — the
    ///   name is dropped but every EAT slot survives — because the name
    ///   tables are only an index over the EAT and one bad pair shouldn't
    ///   discard real exports;
    /// - when several names reference the same EAT slot, the first one wins
    ///   (the name pointer table is sorted, so that's the lexicographically
    ///   lowest name).
    pub fn exports(&self, data: &[u8]) -> Result<Option<ExportDirectory>> {
        let Some(&dir) = self.optional.data_directories.get(directory_index::EXPORT) else {
            return Ok(None);
        };
        if dir.rva == 0 {
            return Ok(None);
        }

        // IMAGE_EXPORT_DIRECTORY: ten 32-bit fields.
        let mut r = Reader::new(data);
        r.seek(self.rva_to_offset(dir.rva)?)?;
        let _characteristics = r.u32()?;
        let _time_date_stamp = r.u32()?;
        let _version = r.u32()?; // major/minor u16 pair
        let name_rva = r.u32()?;
        let ordinal_base = r.u32()?;
        let number_of_functions = r.u32()?;
        let number_of_names = r.u32()?;
        let eat_rva = r.u32()?; // AddressOfFunctions
        let names_rva = r.u32()?; // AddressOfNames
        let ordinals_rva = r.u32()?; // AddressOfNameOrdinals

        // Cap both counts BEFORE any allocation or loop sized by them.
        if number_of_functions as usize > MAX_FUNCTIONS {
            return Err(ParseError::Unsupported(format!(
                "export directory declares {number_of_functions} functions, more than {MAX_FUNCTIONS}"
            )));
        }
        if number_of_names as usize > MAX_NAMES {
            return Err(ParseError::Unsupported(format!(
                "export directory declares {number_of_names} names, more than {MAX_NAMES}"
            )));
        }

        let dll_name = if name_rva != 0 {
            self.cstr_at_rva_bounded(data, name_rva, MAX_NAME_LEN)?
        } else {
            String::new()
        };

        // Export Address Table: one u32 per exported ordinal.
        let mut eat = Vec::with_capacity(number_of_functions as usize);
        if number_of_functions > 0 {
            let mut er = Reader::new(data);
            er.seek(self.rva_to_offset(eat_rva)?)?;
            for _ in 0..number_of_functions {
                eat.push(er.u32()?);
            }
        }

        // Attach names: the name pointer table and the ordinal table are
        // parallel arrays, and each ordinal is an unbiased EAT index.
        let mut names: Vec<Option<String>> = vec![None; eat.len()];
        if number_of_names > 0 {
            let mut np = Reader::new(data);
            np.seek(self.rva_to_offset(names_rva)?)?;
            let mut op = Reader::new(data);
            op.seek(self.rva_to_offset(ordinals_rva)?)?;
            for _ in 0..number_of_names {
                let name_ptr = np.u32()?;
                let index = op.u16()? as usize;
                let name = self.cstr_at_rva_bounded(data, name_ptr, MAX_NAME_LEN)?;
                // Out-of-range index: name skipped. Duplicate: first wins.
                if let Some(slot) = names.get_mut(index)
                    && slot.is_none()
                {
                    *slot = Some(name);
                }
            }
        }

        let mut exports = Vec::new();
        for (i, (&rva, name)) in eat.iter().zip(names.iter_mut()).enumerate() {
            if rva == 0 {
                continue; // unused EAT slot
            }
            // A forwarder is an EAT entry that points back inside the export
            // directory's data-directory range instead of at code or data.
            let target = if rva >= dir.rva && rva - dir.rva < dir.size {
                ExportTarget::Forwarder(self.cstr_at_rva_bounded(data, rva, MAX_NAME_LEN)?)
            } else {
                ExportTarget::Rva(rva)
            };
            exports.push(Export {
                // Wrapping: a hostile Base near u32::MAX must not panic.
                ordinal: ordinal_base.wrapping_add(i as u32),
                name: name.take(),
                target,
            });
        }

        Ok(Some(ExportDirectory {
            dll_name,
            ordinal_base,
            exports,
        }))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::PeFile;
    use super::super::tests::synthetic_pe64;
    use super::*;

    /// File offset of the data directory table in the synthetic image
    /// (e_lfanew + signature + COFF header + offset of the directories).
    const DIRS: usize = 0x80 + 4 + 20 + 112;
    /// File offset backing RVA 0x1000 (start of .text).
    const BASE: usize = 0x200;

    fn put16(img: &mut [u8], off: usize, v: u16) {
        img[BASE + off..BASE + off + 2].copy_from_slice(&v.to_le_bytes());
    }
    fn put32(img: &mut [u8], off: usize, v: u32) {
        img[BASE + off..BASE + off + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// Lay an export directory into .text (RVA 0x1000 -> file 0x200) with a
    /// named export, an unused EAT slot, an ordinal-only export, and a
    /// forwarder, all in one image. Shared with the trait-layer tests in
    /// [`crate::model`].
    pub(crate) fn with_exports() -> Vec<u8> {
        let mut img = synthetic_pe64();

        // IMAGE_EXPORT_DIRECTORY at RVA 0x1000.
        put32(&mut img, 0x0C, 0x10A0); // Name -> "TOOLKIT.dll"
        put32(&mut img, 0x10, 5); // Base (ordinal base)
        put32(&mut img, 0x14, 4); // NumberOfFunctions
        put32(&mut img, 0x18, 2); // NumberOfNames
        put32(&mut img, 0x1C, 0x1030); // AddressOfFunctions
        put32(&mut img, 0x20, 0x1050); // AddressOfNames
        put32(&mut img, 0x24, 0x1060); // AddressOfNameOrdinals

        // EAT: code RVA / unused slot / code RVA / forwarder.
        put32(&mut img, 0x30, 0x1111);
        put32(&mut img, 0x34, 0);
        put32(&mut img, 0x38, 0x2222);
        put32(&mut img, 0x3C, 0x10C0); // inside the directory range

        // Name pointer table and parallel ordinal (EAT index) table.
        put32(&mut img, 0x50, 0x1090); // "Alpha" -> EAT[0]
        put32(&mut img, 0x54, 0x1098); // "Beta"  -> EAT[3]
        put16(&mut img, 0x60, 0);
        put16(&mut img, 0x62, 3);

        img[BASE + 0x90..BASE + 0x96].copy_from_slice(b"Alpha\0");
        img[BASE + 0x98..BASE + 0x9D].copy_from_slice(b"Beta\0");
        img[BASE + 0xA0..BASE + 0xAC].copy_from_slice(b"TOOLKIT.dll\0");
        img[BASE + 0xC0..BASE + 0xCB].copy_from_slice(b"OTHER.Func\0");

        // Export data directory (index 0): RVA 0x1000, size 0x100, so EAT
        // values inside 0x1000..0x1100 read back as forwarder strings.
        img[DIRS..DIRS + 4].copy_from_slice(&0x1000u32.to_le_bytes());
        img[DIRS + 4..DIRS + 8].copy_from_slice(&0x100u32.to_le_bytes());
        img
    }

    #[test]
    fn parses_named_ordinal_only_and_forwarded_exports() {
        let img = with_exports();
        let pe = PeFile::parse(&img).unwrap();
        let dir = pe.exports(&img).unwrap().unwrap();

        assert_eq!(dir.dll_name, "TOOLKIT.dll");
        assert_eq!(dir.ordinal_base, 5);
        assert_eq!(
            dir.exports,
            vec![
                Export {
                    ordinal: 5,
                    name: Some("Alpha".into()),
                    target: ExportTarget::Rva(0x1111),
                },
                // EAT slot 1 is zero: skipped, so ordinal 6 never appears.
                Export {
                    ordinal: 7,
                    name: None,
                    target: ExportTarget::Rva(0x2222),
                },
                Export {
                    ordinal: 8,
                    name: Some("Beta".into()),
                    target: ExportTarget::Forwarder("OTHER.Func".into()),
                },
            ]
        );
    }

    #[test]
    fn no_export_directory_means_none() {
        let img = synthetic_pe64();
        let pe = PeFile::parse(&img).unwrap();
        assert_eq!(pe.exports(&img).unwrap(), None);
    }

    #[test]
    fn empty_export_directory_parses_to_no_exports() {
        let mut img = with_exports();
        put32(&mut img, 0x14, 0); // NumberOfFunctions
        put32(&mut img, 0x18, 0); // NumberOfNames
        let pe = PeFile::parse(&img).unwrap();
        let dir = pe.exports(&img).unwrap().unwrap();
        assert_eq!(dir.dll_name, "TOOLKIT.dll");
        assert_eq!(dir.exports, Vec::new());
    }

    #[test]
    fn adversarial_counts_are_rejected_before_allocation() {
        // NumberOfFunctions, then NumberOfNames.
        for field in [0x14, 0x18] {
            let mut img = with_exports();
            put32(&mut img, field, 0xFFFF_FFFF);
            let pe = PeFile::parse(&img).unwrap();
            let err = pe.exports(&img).unwrap_err();
            assert!(matches!(err, ParseError::Unsupported(_)), "field {field:#x}");
        }
    }

    #[test]
    fn out_of_bounds_name_pointer_is_an_error() {
        let mut img = with_exports();
        put32(&mut img, 0x50, 0x9000); // "Alpha" string RVA -> unmapped
        let pe = PeFile::parse(&img).unwrap();
        assert_eq!(pe.exports(&img), Err(ParseError::UnmappedRva(0x9000)));
    }

    #[test]
    fn unterminated_name_at_end_of_image_is_an_error() {
        let mut img = with_exports();
        // Point "Alpha"'s name pointer at bytes that run to the end of the
        // file with no NUL: RVA 0x11F0 backs file offset 0x3F0.
        put32(&mut img, 0x50, 0x11F0);
        img[0x3F0..0x400].fill(b'A');
        let pe = PeFile::parse(&img).unwrap();
        assert!(pe.exports(&img).is_err());
    }

    #[test]
    fn out_of_range_name_ordinal_is_skipped() {
        let mut img = with_exports();
        put16(&mut img, 0x62, 99); // "Beta" now indexes past the 4-slot EAT
        let pe = PeFile::parse(&img).unwrap();
        let dir = pe.exports(&img).unwrap().unwrap();
        // The forwarder lost its name; every EAT slot survived.
        assert_eq!(dir.exports.len(), 3);
        assert_eq!(dir.exports[0].name.as_deref(), Some("Alpha"));
        assert_eq!(dir.exports[2].name, None);
    }

    #[test]
    fn duplicate_names_for_one_slot_keep_the_first() {
        let mut img = with_exports();
        put16(&mut img, 0x62, 0); // "Beta" now also names EAT[0]
        let pe = PeFile::parse(&img).unwrap();
        let dir = pe.exports(&img).unwrap().unwrap();
        assert_eq!(dir.exports[0].name.as_deref(), Some("Alpha"));
        assert_eq!(dir.exports[2].name, None);
    }

    #[test]
    fn truncated_image_never_panics() {
        let img = with_exports();
        for len in 0..img.len() {
            if let Ok(pe) = PeFile::parse(&img[..len]) {
                let _ = pe.exports(&img[..len]);
            }
        }
    }
}
