//! Content-relative anchoring — stable identity for a function that
//! survives relocation of the binary it lives in.
//!
//! Phase 3's annotation database keys every human assertion (a name, a
//! comment, a type) to a *location* in the program. If that key were the
//! function's absolute virtual address, then rebasing the image, or
//! recompiling it so the linker shifts everything by a few bytes, would
//! orphan every annotation. Analysts' hours of work must not evaporate on
//! a rebuild.
//!
//! An [`Anchor`] is therefore layered, most- to least-durable:
//!
//! - `shape` — an FNV-1a fingerprint of the function's *instruction
//!   stream shape*: the sequence of `(length, control-flow class)` pairs.
//!   Branch **target addresses are deliberately excluded** from the fold,
//!   which is exactly what makes the fingerprint invariant to where the
//!   binary is loaded. Two builds of the same function that differ only in
//!   link-time addresses share a `shape`.
//! - `bytes` — an FNV-1a fingerprint of the raw function body. This *does*
//!   capture embedded addresses, so it matches only the identical binary;
//!   it is the fast, unambiguous path for re-analyzing the same file.
//! - `abs` — the entry virtual address. A last-resort hint, used only to
//!   disambiguate when `shape` alone is not unique.
//!
//! [`AnchorIndex`] resolves an anchor against a freshly recovered
//! [`cfg::Program`] using that priority: exact `bytes`, then `shape`, then
//! `abs`. Resolution reports *how* it matched (see [`Resolution`]) so the
//! database can distinguish a confident hit from a best-effort one.
//!
//! The fold is deterministic: blocks are visited in entry-VA order
//! ([`BTreeMap`] iterates sorted), so an anchor depends only on the
//! function's bytes, never on the order analysis discovered them.

use std::collections::BTreeMap;

use crate::cfg;
use crate::model::{Flow, Image, decoder_for};

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A running FNV-1a 64-bit hash.
///
/// Non-cryptographic and dependency-free — anchoring needs a *stable*,
/// well-distributed identity, not collision resistance against an
/// adversary. FNV-1a is byte-order-fixed and reproducible across
/// platforms, which is what a git-mergeable on-disk key requires. Shared
/// crate-wide so the annotation log derives its record ids the same way.
#[derive(Clone, Copy)]
pub(crate) struct Fnv(u64);

impl Fnv {
    pub(crate) fn new() -> Self {
        Fnv(FNV_OFFSET)
    }

    pub(crate) fn byte(&mut self, b: u8) {
        self.0 ^= b as u64;
        self.0 = self.0.wrapping_mul(FNV_PRIME);
    }

    pub(crate) fn slice(&mut self, s: &[u8]) {
        for &b in s {
            self.byte(b);
        }
    }

    /// Fold a value little-endian, so the fold is byte-order-independent.
    pub(crate) fn u64(&mut self, v: u64) {
        self.slice(&v.to_le_bytes());
    }

    pub(crate) fn finish(self) -> u64 {
        self.0
    }
}

/// Stable, layered identity for one function.
///
/// Cheap to store (three `u64`s and a count) and cheap to compare. The
/// fields are ordered by durability; see the module docs for how each is
/// built and why `shape` survives relocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Anchor {
    /// Rebase-invariant fingerprint of the instruction-stream shape.
    pub shape: u64,
    /// Exact fingerprint of the raw function body (same-binary fast path).
    pub bytes: u64,
    /// Decoded instruction count — a cheap corroborating discriminator
    /// that cuts `shape` collisions between unrelated functions.
    pub insns: u32,
    /// Entry virtual address — a fallback disambiguation hint only.
    pub abs: u64,
}

impl Anchor {
    /// The relocation-independent identity of the anchored function:
    /// `(shape, bytes, insns)`, with the `abs` hint excluded.
    ///
    /// Two anchors for the same function across a rebase share an
    /// identity even though their `abs` differs, so the annotation store
    /// treats them as one target. Used as the store's grouping key.
    pub fn identity(&self) -> (u64, u64, u32) {
        (self.shape, self.bytes, self.insns)
    }
}

/// Fixed small-integer tag for a control-flow class.
///
/// Only the *class* is folded into `shape`; the target VA carried by
/// [`Flow::Jump`], [`Flow::CondJump`], and [`Flow::Call`] is dropped on
/// purpose — that address is precisely the relocation-sensitive part.
fn flow_tag(flow: Flow) -> u8 {
    match flow {
        Flow::Sequential => 0,
        Flow::Jump(_) => 1,
        Flow::CondJump(_) => 2,
        Flow::IndirectJump => 3,
        Flow::Call(_) => 4,
        Flow::IndirectCall => 5,
        Flow::Return => 6,
        Flow::Interrupt => 7,
        Flow::Halt => 8,
    }
}

/// Compute the [`Anchor`] for one recovered function.
///
/// Walks the function's blocks in entry-VA order, folding each block's
/// raw bytes into `bytes` and each instruction's `(length, flow class)`
/// into `shape`. A block whose start does not map to a file offset, or an
/// instruction that fails to decode, ends that block's contribution
/// early rather than aborting — an anchor is always producible.
pub fn of_function(image: &dyn Image, func: &cfg::Function) -> Anchor {
    let decoder = decoder_for(image.arch());
    let image_bytes = image.bytes();

    let mut shape = Fnv::new();
    let mut raw = Fnv::new();
    let mut insns: u32 = 0;

    for blk in func.blocks.values() {
        let span = blk.end.saturating_sub(blk.start) as usize;
        let Some(off) = image.va_to_offset(blk.start) else {
            continue;
        };
        let end = off.saturating_add(span).min(image_bytes.len());
        if off >= end {
            continue;
        }
        let body = &image_bytes[off..end];
        raw.slice(body);

        // Decode the block's instruction stream for the shape fold. The
        // VA-to-offset mapping is linear within a region, so byte `k` of
        // `body` is at VA `blk.start + k`.
        let Some(decoder) = decoder else {
            continue;
        };
        let mut cursor = 0usize;
        while cursor < body.len() {
            let va = blk.start + cursor as u64;
            match decoder.decode_flow(&body[cursor..], va) {
                Ok(d) if d.length > 0 => {
                    shape.byte(d.length);
                    shape.byte(flow_tag(d.flow));
                    insns += 1;
                    cursor += d.length as usize;
                }
                _ => break,
            }
        }
    }

    // Fold the instruction count last so functions of different lengths
    // are unlikely to share a shape even when a prefix matches.
    shape.u64(insns as u64);

    Anchor {
        shape: shape.finish(),
        bytes: raw.finish(),
        insns,
        abs: func.entry,
    }
}

/// How an [`Anchor`] matched the current program.
///
/// Ordered by confidence. The database uses the variant to decide whether
/// an annotation reattached cleanly or needs analyst review after a
/// binary change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// Exact raw-byte match: the identical function in the identical
    /// binary. The carried VA is the resolved entry.
    Exact(u64),
    /// Rebase-invariant shape match, unique after instruction-count
    /// filtering. Survives relocation; the carried VA is the entry.
    Shape(u64),
    /// Neither `bytes` nor `shape` matched, but the recorded entry VA
    /// still names a function. Weakest evidence.
    Absolute(u64),
    /// Several functions share the shape and the count, and the entry VA
    /// did not pick one out. The database surfaces the candidates rather
    /// than guessing.
    Ambiguous(Vec<u64>),
    /// Nothing in the current program matches.
    Unresolved,
}

/// An index over a recovered program's anchors, for resolving stored
/// anchors back to current entry VAs.
///
/// Built once per analysis and reused across every annotation.
pub struct AnchorIndex {
    /// Entry VA to its anchor.
    anchors: BTreeMap<u64, Anchor>,
    /// Raw-byte fingerprint to the entry VAs that produce it.
    by_bytes: BTreeMap<u64, Vec<u64>>,
    /// Shape fingerprint to the entry VAs that produce it.
    by_shape: BTreeMap<u64, Vec<u64>>,
}

impl AnchorIndex {
    /// Build an index over every function in `program`.
    pub fn build(image: &dyn Image, program: &cfg::Program) -> Self {
        let mut anchors = BTreeMap::new();
        let mut by_bytes: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        let mut by_shape: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for (&entry, func) in &program.functions {
            let a = of_function(image, func);
            by_bytes.entry(a.bytes).or_default().push(entry);
            by_shape.entry(a.shape).or_default().push(entry);
            anchors.insert(entry, a);
        }
        AnchorIndex {
            anchors,
            by_bytes,
            by_shape,
        }
    }

    /// The anchor for `entry`, if it names a function in this program.
    pub fn anchor_of(&self, entry: u64) -> Option<&Anchor> {
        self.anchors.get(&entry)
    }

    /// Resolve a stored `anchor` to a current entry VA.
    ///
    /// Tries the layers in durability order: exact bytes, then shape, then
    /// the absolute-address hint. Candidate lists are filtered by
    /// instruction count first, and the stored `abs` disambiguates a tie.
    pub fn resolve(&self, anchor: &Anchor) -> Resolution {
        // 1. Same-binary exact match.
        if let Some(hit) = self
            .by_bytes
            .get(&anchor.bytes)
            .and_then(|vas| self.pick(vas, anchor))
        {
            return Resolution::Exact(hit);
        }

        // 2. Rebase-invariant shape match.
        if let Some(vas) = self.by_shape.get(&anchor.shape) {
            let matches: Vec<u64> = vas
                .iter()
                .copied()
                .filter(|va| self.anchors[va].insns == anchor.insns)
                .collect();
            match matches.as_slice() {
                [] => {}
                [va] => return Resolution::Shape(*va),
                many => {
                    if many.contains(&anchor.abs) {
                        return Resolution::Shape(anchor.abs);
                    }
                    return Resolution::Ambiguous(many.to_vec());
                }
            }
        }

        // 3. Absolute-address fallback.
        if self.anchors.contains_key(&anchor.abs) {
            return Resolution::Absolute(anchor.abs);
        }

        Resolution::Unresolved
    }

    /// From candidate entry VAs sharing a fingerprint, pick the one that
    /// also matches the instruction count, preferring the recorded `abs`
    /// on a tie. Returns `None` if none qualify.
    fn pick(&self, vas: &[u64], anchor: &Anchor) -> Option<u64> {
        let matches: Vec<u64> = vas
            .iter()
            .copied()
            .filter(|va| self.anchors[va].insns == anchor.insns)
            .collect();
        match matches.as_slice() {
            [] => None,
            [va] => Some(*va),
            many => Some(if many.contains(&anchor.abs) {
                anchor.abs
            } else {
                many[0]
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::recover;
    use crate::model::load;
    use crate::pe::tests::synthetic_pe64;

    /// Image base of the synthetic PE fixture.
    const PE_BASE: u64 = 0x1_4000_0000;
    /// Entry VA of the synthetic PE fixture (RVA 0x1000, file 0x200).
    const PE_ENTRY: u64 = PE_BASE + 0x1000;

    /// Build a synthetic PE with `code` at the entry and recover it.
    fn program_with(code: &[u8]) -> (Vec<u8>, cfg::Program) {
        let mut img = synthetic_pe64();
        img[0x200..0x200 + code.len()].copy_from_slice(code);
        let program = {
            let image = load(&img).unwrap();
            recover(image.as_ref()).unwrap()
        };
        (img, program)
    }

    /// The anchor of the entry function of a synthetic PE holding `code`.
    fn anchor_of_entry(code: &[u8]) -> Anchor {
        let (img, program) = program_with(code);
        let image = load(&img).unwrap();
        of_function(image.as_ref(), &program.functions[&PE_ENTRY])
    }

    #[test]
    fn anchor_is_deterministic() {
        let code: &[u8] = &[0x31, 0xC0, 0xFF, 0xC0, 0xC3]; // xor;inc;ret
        assert_eq!(anchor_of_entry(code), anchor_of_entry(code));
    }

    #[test]
    fn different_shaped_bodies_have_different_shapes() {
        let a = anchor_of_entry(&[0x31, 0xC0, 0xC3]); // xor eax,eax; ret (2B seq + ret)
        let b = anchor_of_entry(&[0x90, 0xC3]); // nop; ret (1B seq + ret)
        assert_ne!(a.shape, b.shape);
        assert_ne!(a.bytes, b.bytes);
    }

    #[test]
    fn same_shape_different_opcodes_collide_on_shape_not_bytes() {
        // `shape` is a structural fingerprint: two functions with the same
        // instruction lengths and flow classes legitimately share it. That
        // is by design — `bytes` (and `abs`) are what tell them apart.
        let a = anchor_of_entry(&[0x31, 0xC0, 0xC3]); // xor eax,eax; ret
        let b = anchor_of_entry(&[0xFF, 0xC0, 0xC3]); // inc eax; ret
        assert_eq!(a.shape, b.shape, "identical structure shares a shape");
        assert_ne!(a.bytes, b.bytes, "raw bytes still distinguish them");
    }

    #[test]
    fn shape_survives_a_relocated_call_target() {
        // Same instructions, but the rel32 of a direct `call` differs —
        // exactly what a rebase or relink changes. `shape` must ignore it
        // while `bytes` must not.
        // e8 <rel32> call; c3 ret.
        let near = anchor_of_entry(&[0xE8, 0x10, 0x00, 0x00, 0x00, 0xC3]);
        let far = anchor_of_entry(&[0xE8, 0x40, 0x00, 0x00, 0x00, 0xC3]);
        assert_eq!(near.shape, far.shape, "shape must ignore call target");
        assert_eq!(near.insns, far.insns);
        assert_ne!(near.bytes, far.bytes, "bytes must capture the target");
    }

    #[test]
    fn resolve_exact_same_binary() {
        let code: &[u8] = &[0x31, 0xC0, 0xFF, 0xC0, 0xC3];
        let (img, program) = program_with(code);
        let image = load(&img).unwrap();
        let index = AnchorIndex::build(image.as_ref(), &program);
        let anchor = *index.anchor_of(PE_ENTRY).unwrap();
        assert_eq!(index.resolve(&anchor), Resolution::Exact(PE_ENTRY));
    }

    #[test]
    fn resolve_by_shape_after_target_shift() {
        // Anchor captured against one build; resolve against another that
        // differs only in a call target. The bytes hash misses, the shape
        // hits.
        let stored = anchor_of_entry(&[0xE8, 0x10, 0x00, 0x00, 0x00, 0xC3]);
        let (img, program) = program_with(&[0xE8, 0x40, 0x00, 0x00, 0x00, 0xC3]);
        let image = load(&img).unwrap();
        let index = AnchorIndex::build(image.as_ref(), &program);
        assert_eq!(index.resolve(&stored), Resolution::Shape(PE_ENTRY));
    }

    #[test]
    fn resolve_unresolved_when_absent() {
        let stored = anchor_of_entry(&[0x31, 0xC0, 0xFF, 0xC0, 0xC3]);
        // A program whose entry function is a completely different body.
        let (img, program) = program_with(&[0x90, 0x90, 0x90, 0xCC]); // nops; int3
        let image = load(&img).unwrap();
        let index = AnchorIndex::build(image.as_ref(), &program);
        // The stored function's shape/bytes are absent; its abs VA does
        // still name a function here, so this falls back to Absolute
        // rather than Unresolved. Assert the weaker-but-correct outcome.
        assert_eq!(index.resolve(&stored), Resolution::Absolute(PE_ENTRY));
    }
}
