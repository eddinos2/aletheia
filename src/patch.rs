//! Surgical binary patching — PatchSet objects beyond IDA-style byte theatre.
//!
//! A [`PatchSet`] records exact old→new byte edits at virtual addresses,
//! with optional preconditions and an Apple resign recipe stub. Preview
//! never writes; apply defaults to a sibling `*.patched` path.
//!
//! See `PLAN_PATCH.md` and ROADMAP Patching principles (P1).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{ParseError, Result};
use crate::model::Image;

/// Engine version stamp embedded in every PatchSet (bump when schema changes).
pub const PATCH_SCHEMA: &str = "aletheia-patchset-v1";

/// One surgical edit at a virtual address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchEdit {
    pub va: u64,
    pub old_bytes: Vec<u8>,
    pub new_bytes: Vec<u8>,
    /// Free-form intent (e.g. "nop branch", "force true").
    pub intent: String,
}

/// Optional Apple-side post-apply hints (engine never runs codesign itself).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AppleResignHint {
    pub preserve_entitlements: bool,
    /// Printed recipe lines for the operator / CI.
    pub recipe_lines: Vec<String>,
}

/// Auditable patch object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchSet {
    pub schema: String,
    /// FNV-1a 64 of the target file bytes at recipe creation (not crypto).
    pub target_hash: u64,
    pub edits: Vec<PatchEdit>,
    pub apple: Option<AppleResignHint>,
}

/// Why preview/apply refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchFault {
    Empty,
    LengthMismatch { va: u64 },
    UnmappedVa { va: u64 },
    OldBytesMismatch { va: u64 },
    HashMismatch { expected: u64, actual: u64 },
    Io(String),
}

impl std::fmt::Display for PatchFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatchFault::Empty => write!(f, "patch set has no edits"),
            PatchFault::LengthMismatch { va } => {
                write!(f, "edit at {va:#x}: old/new length differ")
            }
            PatchFault::UnmappedVa { va } => write!(f, "VA {va:#x} is not file-backed"),
            PatchFault::OldBytesMismatch { va } => {
                write!(f, "precondition failed at {va:#x}: old bytes do not match")
            }
            PatchFault::HashMismatch { expected, actual } => {
                write!(f, "target hash {expected:#x} != file {actual:#x}")
            }
            PatchFault::Io(s) => write!(f, "I/O: {s}"),
        }
    }
}

/// FNV-1a 64 over raw bytes — deterministic stamp, not a security hash.
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// AArch64 `NOP` — `HINT #0` encoding `D503201F` (Arm ARM).
pub fn aarch64_nop() -> [u8; 4] {
    [0x1F, 0x20, 0x03, 0xD5]
}

/// Unconditional `B` to PC-relative target (imm26, ±128MB). Little-endian bytes.
pub fn aarch64_b(from_va: u64, to_va: u64) -> Option<[u8; 4]> {
    let off = to_va as i64 - from_va as i64;
    if off % 4 != 0 {
        return None;
    }
    let imm26 = off / 4;
    if !(-(1 << 25)..(1 << 25)).contains(&imm26) {
        return None;
    }
    let word = 0x1400_0000u32 | ((imm26 as u32) & 0x03FF_FFFF);
    Some(word.to_le_bytes())
}

impl PatchSet {
    pub fn new(target_bytes: &[u8], edits: Vec<PatchEdit>) -> PatchSet {
        PatchSet {
            schema: PATCH_SCHEMA.into(),
            target_hash: fnv1a64(target_bytes),
            edits,
            apple: None,
        }
    }

    /// Attach a standard resign recipe for a Mach-O path.
    pub fn with_macho_resign_recipe(mut self, binary_path: &str) -> Self {
        self.apple = Some(AppleResignHint {
            preserve_entitlements: true,
            recipe_lines: vec![
                format!("# extract entitlements (host tooling)"),
                format!("codesign -d --entitlements :- {binary_path} > /tmp/ent.xml"),
                format!("# re-sign ad-hoc (jailbreak) or with your identity:"),
                format!("codesign -s - --entitlements /tmp/ent.xml --force {binary_path}"),
            ],
        });
        self
    }

    /// Deterministic text report (preview).
    pub fn render_preview(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "; {} hash={:#x} edits={}\n",
            self.schema,
            self.target_hash,
            self.edits.len()
        ));
        for e in &self.edits {
            out.push_str(&format!(
                "  {va:#x}  {old} -> {new}  ; {intent}\n",
                va = e.va,
                old = hex_bytes(&e.old_bytes),
                new = hex_bytes(&e.new_bytes),
                intent = e.intent
            ));
        }
        if let Some(a) = &self.apple {
            out.push_str("; apple resign recipe:\n");
            for line in &a.recipe_lines {
                out.push_str(line);
                out.push('\n');
            }
        }
        out
    }

    /// Validate edits against `image` without writing.
    pub fn preview(&self, image: &dyn Image) -> std::result::Result<String, PatchFault> {
        if self.edits.is_empty() {
            return Err(PatchFault::Empty);
        }
        let actual = fnv1a64(image.bytes());
        if actual != self.target_hash {
            return Err(PatchFault::HashMismatch {
                expected: self.target_hash,
                actual,
            });
        }
        for e in &self.edits {
            if e.old_bytes.len() != e.new_bytes.len() {
                return Err(PatchFault::LengthMismatch { va: e.va });
            }
            let off = image
                .va_to_offset(e.va)
                .ok_or(PatchFault::UnmappedVa { va: e.va })?;
            let end = off + e.old_bytes.len();
            let bytes = image.bytes();
            if end > bytes.len() {
                return Err(PatchFault::UnmappedVa { va: e.va });
            }
            if bytes[off..end] != e.old_bytes[..] {
                return Err(PatchFault::OldBytesMismatch { va: e.va });
            }
        }
        Ok(self.render_preview())
    }

    /// Apply to a sibling path (default: `input` + `.patched`).
    pub fn apply_sibling(
        &self,
        image: &dyn Image,
        input_path: &Path,
    ) -> std::result::Result<PathBuf, PatchFault> {
        self.preview(image)?;
        let mut buf = image.bytes().to_vec();
        // Map VA→offset again on the buffer via image (offsets stable).
        for e in &self.edits {
            let off = image
                .va_to_offset(e.va)
                .ok_or(PatchFault::UnmappedVa { va: e.va })?;
            buf[off..off + e.new_bytes.len()].copy_from_slice(&e.new_bytes);
        }
        let out = sibling_patched(input_path);
        std::fs::write(&out, &buf).map_err(|e| PatchFault::Io(e.to_string()))?;
        Ok(out)
    }
}

fn sibling_patched(input: &Path) -> PathBuf {
    let mut s = input.as_os_str().to_os_string();
    s.push(".patched");
    PathBuf::from(s)
}

fn hex_bytes(b: &[u8]) -> String {
    b.iter()
        .map(|x| format!("{x:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build a single-site NOP patch at `va` if the mapped bytes match `old`.
pub fn nop_patch(image: &dyn Image, va: u64, old: &[u8], intent: &str) -> Result<PatchSet> {
    let off = image
        .va_to_offset(va)
        .ok_or(ParseError::UnmappedVaddr(va))?;
    let bytes = image.bytes();
    if off + old.len() > bytes.len() || bytes[off..off + old.len()] != old[..] {
        return Err(ParseError::Unsupported(format!(
            "bytes at {va:#x} do not match precondition"
        )));
    }
    let new = if old.len() == 4 {
        aarch64_nop().to_vec()
    } else {
        vec![0x90; old.len()] // x86 NOP fill
    };
    if new.len() != old.len() {
        return Err(ParseError::Unsupported(
            "NOP fill length mismatch".into(),
        ));
    }
    Ok(PatchSet::new(
        bytes,
        vec![PatchEdit {
            va,
            old_bytes: old.to_vec(),
            new_bytes: new,
            intent: intent.into(),
        }],
    ))
}

/// Instruction-level hunk sketch for a Modified diff pair: byte windows
/// at each entry (placeholder for richer IR alignment).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffHunk {
    pub old_va: u64,
    pub new_va: u64,
    pub old_prefix: Vec<u8>,
    pub new_prefix: Vec<u8>,
}

/// Collect prefix hunks for modified pairs (up to `cap` bytes each).
pub fn modified_hunks(
    old: &dyn Image,
    new: &dyn Image,
    pairs: &[(u64, u64)],
    cap: usize,
) -> Vec<DiffHunk> {
    let mut out = Vec::new();
    for &(ova, nva) in pairs {
        let Some(oo) = old.va_to_offset(ova) else {
            continue;
        };
        let Some(no) = new.va_to_offset(nva) else {
            continue;
        };
        let ob = old.bytes();
        let nb = new.bytes();
        let ol = (oo + cap).min(ob.len());
        let nl = (no + cap).min(nb.len());
        out.push(DiffHunk {
            old_va: ova,
            new_va: nva,
            old_prefix: ob[oo..ol].to_vec(),
            new_prefix: nb[no..nl].to_vec(),
        });
    }
    out
}

/// Stable ordering helper for tests.
pub fn edits_by_va(set: &PatchSet) -> BTreeMap<u64, &PatchEdit> {
    set.edits.iter().map(|e| (e.va, e)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Arch, Image, ImportSlot, Perms, Region, Symbol};

    struct FakeImg {
        bytes: Vec<u8>,
        base: u64,
    }

    impl Image for FakeImg {
        fn arch(&self) -> Arch {
            Arch::Aarch64
        }
        fn entry_points(&self) -> Vec<u64> {
            vec![self.base]
        }
        fn regions(&self) -> Vec<Region> {
            vec![Region {
                name: "__TEXT".into(),
                va: self.base,
                size: self.bytes.len() as u64,
                perms: Perms {
                    r: true,
                    w: false,
                    x: true,
                },
            }]
        }
        fn symbols(&self) -> Vec<Symbol> {
            vec![]
        }
        fn import_slots(&self) -> Vec<ImportSlot> {
            vec![]
        }
        fn va_to_offset(&self, va: u64) -> Option<usize> {
            if va >= self.base && ((va - self.base) as usize) < self.bytes.len() {
                Some((va - self.base) as usize)
            } else {
                None
            }
        }
        fn bytes(&self) -> &[u8] {
            &self.bytes
        }
    }

    #[test]
    fn a64_nop_encoding() {
        assert_eq!(aarch64_nop(), [0x1F, 0x20, 0x03, 0xD5]);
    }

    #[test]
    fn preview_and_apply_round_trip() {
        let mut bytes = vec![0u8; 16];
        bytes[0..4].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        let img = FakeImg {
            bytes: bytes.clone(),
            base: 0x1000,
        };
        let set = PatchSet::new(
            &bytes,
            vec![PatchEdit {
                va: 0x1000,
                old_bytes: vec![0xAA, 0xBB, 0xCC, 0xDD],
                new_bytes: aarch64_nop().to_vec(),
                intent: "nop".into(),
            }],
        );
        assert!(set.preview(&img).is_ok());
        let dir = std::env::temp_dir().join(format!("aletheia-patch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("blob.bin");
        std::fs::write(&input, &bytes).unwrap();
        // Re-bind image to written file contents
        let img2 = FakeImg {
            bytes: std::fs::read(&input).unwrap(),
            base: 0x1000,
        };
        let out = set.apply_sibling(&img2, &input).unwrap();
        assert!(
            out.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".patched"))
                || out.to_string_lossy().ends_with(".patched")
        );
        let patched = std::fs::read(&out).unwrap();
        assert_eq!(&patched[0..4], &aarch64_nop());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn precondition_rejects_wrong_old_bytes() {
        let bytes = vec![1, 2, 3, 4];
        let img = FakeImg {
            bytes: bytes.clone(),
            base: 0,
        };
        let set = PatchSet::new(
            &bytes,
            vec![PatchEdit {
                va: 0,
                old_bytes: vec![9, 9, 9, 9],
                new_bytes: aarch64_nop().to_vec(),
                intent: "x".into(),
            }],
        );
        assert!(matches!(
            set.preview(&img),
            Err(PatchFault::OldBytesMismatch { .. })
        ));
    }
}
