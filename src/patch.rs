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

/// `RET` (return via X30/LR) — encoding `D65F03C0`.
pub fn aarch64_ret() -> [u8; 4] {
    [0xC0, 0x03, 0x5F, 0xD6]
}

/// `BR <Xn>` — branch to register (`D61F0000 | (rn << 5)`).
pub fn aarch64_br(rn: u8) -> Option<[u8; 4]> {
    if rn > 30 {
        return None;
    }
    let word = 0xD61F_0000u32 | (u32::from(rn) << 5);
    Some(word.to_le_bytes())
}

/// `MOVZ <Xd|Wd>, #imm16, LSL #shift` — public move-wide encoding.
/// `shift` is the bit shift (0/16/32/48); `sf` selects 64-bit.
pub fn aarch64_movz(sf: bool, rd: u8, imm16: u16, shift: u8) -> Option<[u8; 4]> {
    if rd > 31 {
        return None;
    }
    let hw = match shift {
        0 => 0u32,
        16 => 1,
        32 if sf => 2,
        48 if sf => 3,
        _ => return None,
    };
    let word = 0x5280_0000u32
        | (u32::from(sf) << 31)
        | (hw << 21)
        | (u32::from(imm16) << 5)
        | u32::from(rd);
    Some(word.to_le_bytes())
}

/// Named public encodings the patch assembler can emit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsmOp {
    /// Architecture-default NOP fill (A64 4-byte / x86 0x90).
    Nop,
    /// AArch64 `RET`.
    Ret,
    /// AArch64 `B` to absolute VA.
    Branch { to: u64 },
    /// AArch64 `BR Xn`.
    Br { rn: u8 },
    /// AArch64 `MOVZ` (64-bit when `sf`).
    Movz {
        sf: bool,
        rd: u8,
        imm16: u16,
        shift: u8,
    },
}

/// Assemble `op` at `va` into little-endian bytes matching `old.len()` when
/// possible. Returns `None` when the encoding is unsupported or length
/// cannot match (same-length PatchSet rule).
pub fn assemble_bytes(arch: crate::model::Arch, va: u64, old_len: usize, op: &AsmOp) -> Option<Vec<u8>> {
    use crate::model::Arch;
    match (arch, op) {
        (Arch::Aarch64, AsmOp::Nop) if old_len == 4 => Some(aarch64_nop().to_vec()),
        (Arch::Aarch64, AsmOp::Ret) if old_len == 4 => Some(aarch64_ret().to_vec()),
        (Arch::Aarch64, AsmOp::Branch { to }) if old_len == 4 => {
            aarch64_b(va, *to).map(|b| b.to_vec())
        }
        (Arch::Aarch64, AsmOp::Br { rn }) if old_len == 4 => aarch64_br(*rn).map(|b| b.to_vec()),
        (Arch::Aarch64, AsmOp::Movz { sf, rd, imm16, shift }) if old_len == 4 => {
            aarch64_movz(*sf, *rd, *imm16, *shift).map(|b| b.to_vec())
        }
        (Arch::X86_64, AsmOp::Nop) if old_len > 0 => Some(vec![0x90; old_len]),
        (Arch::X86_64, AsmOp::Ret) if old_len == 1 => Some(vec![0xC3]),
        _ => None,
    }
}

/// Build a same-length patch at `va` from a public [`AsmOp`].
pub fn assemble_patch(
    image: &dyn Image,
    va: u64,
    old: &[u8],
    op: AsmOp,
    intent: &str,
) -> Result<PatchSet> {
    let off = image
        .va_to_offset(va)
        .ok_or(ParseError::UnmappedVaddr(va))?;
    let bytes = image.bytes();
    if off + old.len() > bytes.len() || bytes[off..off + old.len()] != old[..] {
        return Err(ParseError::Unsupported(format!(
            "bytes at {va:#x} do not match precondition"
        )));
    }
    let new = assemble_bytes(image.arch(), va, old.len(), &op).ok_or_else(|| {
        ParseError::Unsupported(format!(
            "cannot assemble {:?} at {va:#x} for {}-byte site",
            op,
            old.len()
        ))
    })?;
    if new.len() != old.len() {
        return Err(ParseError::Unsupported(
            "assembled length mismatch".into(),
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
    assemble_patch(image, va, old, AsmOp::Nop, intent)
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

/// Turn every [`crate::diff::MatchKind::Modified`] pair in `diff` into a
/// [`DiffHunk`] (prefix window of up to `cap` bytes at each entry).
pub fn hunks_from_modified(
    old: &dyn Image,
    new: &dyn Image,
    diff: &crate::diff::Diff,
    cap: usize,
) -> Vec<DiffHunk> {
    let pairs = diff.modified_pairs();
    modified_hunks(old, new, &pairs, cap)
}

/// Default prefix window (bytes) when turning Modified pairs into hunks /
/// a PatchSet for CLI / MCP.
pub const DEFAULT_HUNK_CAP: usize = 32;

/// Build a same-length [`PatchSet`] from [`DiffHunk`]s against `target_bytes`
/// (the *old* image). Each hunk contributes one edit at `old_va` over the
/// overlapping prefix (`min(old,new)`); identical windows are skipped.
pub fn patchset_from_hunks(target_bytes: &[u8], hunks: &[DiffHunk]) -> PatchSet {
    let mut edits = Vec::new();
    for h in hunks {
        let n = h.old_prefix.len().min(h.new_prefix.len());
        if n == 0 {
            continue;
        }
        let old = &h.old_prefix[..n];
        let new = &h.new_prefix[..n];
        if old == new {
            continue;
        }
        edits.push(PatchEdit {
            va: h.old_va,
            old_bytes: old.to_vec(),
            new_bytes: new.to_vec(),
            intent: format!("diff-hunk old={:#x} new={:#x}", h.old_va, h.new_va),
        });
    }
    PatchSet::new(target_bytes, edits)
}

/// Diff → [`hunks_from_modified`] → [`patchset_from_hunks`] for the old image.
pub fn patchset_from_modified(
    old: &dyn Image,
    new: &dyn Image,
    diff: &crate::diff::Diff,
    cap: usize,
) -> PatchSet {
    let hunks = hunks_from_modified(old, new, diff, cap);
    patchset_from_hunks(old.bytes(), &hunks)
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
    fn a64_public_encodings_round_trip() {
        assert_eq!(aarch64_ret(), [0xC0, 0x03, 0x5F, 0xD6]);
        assert_eq!(aarch64_br(0).unwrap(), [0x00, 0x00, 0x1F, 0xD6]);
        // MOVZ X0, #42
        assert_eq!(
            aarch64_movz(true, 0, 42, 0).unwrap(),
            [0x40, 0x05, 0x80, 0xD2]
        );
        let b = aarch64_b(0x1000, 0x1010).unwrap();
        assert_eq!(u32::from_le_bytes(b), 0x1400_0004);
    }

    #[test]
    fn assemble_ret_patch() {
        let mut bytes = vec![0u8; 8];
        bytes[0..4].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        let img = FakeImg {
            bytes: bytes.clone(),
            base: 0x1000,
        };
        let set = assemble_patch(
            &img,
            0x1000,
            &bytes[0..4],
            AsmOp::Ret,
            "force ret",
        )
        .unwrap();
        assert_eq!(set.edits[0].new_bytes, aarch64_ret());
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

    #[test]
    fn hunks_from_modified_follows_diff_pairs() {
        use crate::cfg::recover;
        use crate::diff::{self, MatchKind};
        use crate::model::load;
        use crate::pe::tests::synthetic_pe64;

        const MOV_1: &[u8] = &[0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3];
        const MOV_2: &[u8] = &[0xB8, 0x02, 0x00, 0x00, 0x00, 0xC3];

        fn pe_with(code: &[u8]) -> Vec<u8> {
            let mut img = synthetic_pe64();
            let opt = 0x80 + 4 + 20;
            let dirs = opt + 112;
            img[dirs + 8..dirs + 16].fill(0);
            img[opt + 16..opt + 20].copy_from_slice(&0x1000u32.to_le_bytes());
            let off = 0x200;
            img[off..off + code.len()].copy_from_slice(code);
            img
        }

        let old_bytes = pe_with(MOV_1);
        let new_bytes = pe_with(MOV_2);
        let old_image = load(&old_bytes).expect("old loads");
        let new_image = load(&new_bytes).expect("new loads");
        let old_program = recover(old_image.as_ref()).expect("old recovers");
        let new_program = recover(new_image.as_ref()).expect("new recovers");
        let d = diff::diff(
            old_image.as_ref(),
            &old_program,
            new_image.as_ref(),
            &new_program,
        );
        assert_eq!(d.modified_pairs().len(), 1);
        assert!(
            d.of_kind("modified")
                .iter()
                .all(|p| p.kind == MatchKind::Modified)
        );
        let hunks = hunks_from_modified(old_image.as_ref(), new_image.as_ref(), &d, 6);
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].old_prefix, MOV_1);
        assert_eq!(hunks[0].new_prefix, MOV_2);

        let set = patchset_from_modified(old_image.as_ref(), new_image.as_ref(), &d, 6);
        assert_eq!(set.edits.len(), 1);
        assert_eq!(set.edits[0].old_bytes, MOV_1);
        assert_eq!(set.edits[0].new_bytes, MOV_2);
        assert!(set.preview(old_image.as_ref()).is_ok());
    }

    #[test]
    fn patchset_from_hunks_skips_identical_and_empty() {
        let bytes = vec![0u8; 8];
        let set = patchset_from_hunks(
            &bytes,
            &[
                DiffHunk {
                    old_va: 0,
                    new_va: 0,
                    old_prefix: vec![1, 2],
                    new_prefix: vec![1, 2],
                },
                DiffHunk {
                    old_va: 4,
                    new_va: 4,
                    old_prefix: vec![],
                    new_prefix: vec![9],
                },
                DiffHunk {
                    old_va: 2,
                    new_va: 2,
                    old_prefix: vec![0, 0, 0],
                    new_prefix: vec![0xaa, 0xbb],
                },
            ],
        );
        assert_eq!(set.edits.len(), 1);
        assert_eq!(set.edits[0].va, 2);
        assert_eq!(set.edits[0].old_bytes, vec![0, 0]);
        assert_eq!(set.edits[0].new_bytes, vec![0xaa, 0xbb]);
    }
}
