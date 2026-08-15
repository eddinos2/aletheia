//! Go string recovery — reconstructing string literals from a Go binary.
//!
//! Go string data is not NUL-terminated: a `string` is a
//! (pointer, length) header, and the bytes it points at sit packed
//! end-to-end in a read-only blob with no delimiters, so the generic
//! C-string scanner in [`crate::strings`] cannot recover them — it
//! run-togethers adjacent literals (`"hello"` and `"world"` stored as
//! `helloworld` become one garbage string) and mis-splits wherever an
//! incidental NUL lands. This module recovers them the way the runtime
//! addresses them — from the (pointer, length) pairs the code and the
//! type metadata materialize — yielding the exact literal set with no
//! run-together or truncated fragments. Best-effort by contract: a
//! binary without recognizable Go string structure yields an empty
//! result, and no input ever panics.
//!
//! # Method
//!
//! [`recover`] gates on [`gopcln::is_go`] — a non-Go image yields
//! nothing — then runs two independent tiers and merges them.
//!
//! ## Tier 1 — instruction-materialized headers ([`Source::Code`])
//!
//! Go code loads a literal by materializing its address and its length
//! as immediates a few instructions apart. Each executable region is
//! swept linearly with the rich per-ISA decoder ([`x86::decode`] /
//! [`aarch64::decode`], not the flow-level [`crate::model::Decoder`] —
//! this needs operands):
//!
//! - **x86-64**: `lea r64, [rip + disp32]` computes the address
//!   `next_insn_va + disp` (SDM Vol 2 §2.2.1.6); a following
//!   `mov reg/mem, imm` supplies the length.
//! - **AArch64**: `ADRP Xd, page` followed by `ADD Xd, Xn, #lo12`
//!   computes the address (a bare `ADR` does so on its own); a
//!   following move-immediate supplies the length. A64 splits that move
//!   across two encodings and Go's assembler uses both, so both are
//!   read: `MOVZ Wd/Xd, #imm` for a 16-bit field, and the logical
//!   `ORR Xd, XZR, #bitmask` (decoded as
//!   [`aarch64::Opcode::LogImm`]) for a rotated run of set bits —
//!   which is how a length of 3 or 4 arrives.
//!
//! The two halves are paired through a sliding window spanning at most
//! [`LOOKAHEAD`] instructions, holding up to [`MAX_PENDING`] of each. The
//! pairing is **symmetric**: Go emits the address first or the length
//! first depending on how the register allocator ordered them, so an
//! address pairs with a waiting length just as a length pairs with a
//! waiting address. Whichever half completes the pair takes the *newest*
//! waiting partner that validates (see below) and consumes it, since Go
//! emits the two immediates back to back. This tier yields exact
//! boundaries, which is the whole point: `"hello"` and `"world"` come
//! back as two strings.
//!
//! ## Tier 2 — data-resident headers ([`Source::Data`])
//!
//! Many string headers live in data as adjacent word pairs — inside
//! reflect type data, error tables, method name arrays. Each readable
//! region is swept at 8-byte alignment for a `[ptr: 8][len: 8]`
//! little-endian pair whose pointer targets a read-only region and whose
//! length validates.
//!
//! ## The boundary gate
//!
//! Neither tier's raw output can be trusted on its own. A data pointer
//! into the *middle* of a longer literal validates perfectly well, and so
//! does an unrelated constant that a tier-1 window happened to pair with
//! a real address — on a stock Go binary that one accident produced a
//! 13 KiB "literal" spanning most of the blob, which then shadowed every
//! genuine string inside it.
//!
//! Both tiers therefore pass a **boundary gate**: the byte one past a
//! candidate's run must be a place a literal can plausibly end — the
//! start of some *other* candidate (either tier, gated or not), a byte
//! that is not text (a NUL or a control byte), or the edge of the
//! pointer's own region. A run that stops in the middle of the packed
//! blob with nothing else referencing that point is dropped. This is what
//! makes the packed blob self-delimiting: real neighbors vouch for each
//! other's boundaries, and a spurious length has nothing to land on.
//!
//! ## Validation
//!
//! A candidate `(p, n)` is accepted only when `1 <= n <=` [`MAX_STR_LEN`],
//! `p` lies in a region that is readable and **not writable** (the
//! read-only blob may still be executable — Mach-O keeps `__rodata`
//! inside the r-x `__TEXT` segment), `[p, p + n)` is entirely inside that
//! one region *and* file-backed, every byte is a permitted text byte
//! (tab, newline and carriage return pass; every other C0 control, NUL,
//! and `0x7F` fail), and the run as a whole is valid UTF-8.
//!
//! ## Merge, dedup, and overlap resolution
//!
//! The two tiers are concatenated and gated, then:
//!
//! 1. **Dedup** by `(va, len)`. A literal found by both tiers is reported
//!    once, labeled [`Source::Code`] — the more precise provenance wins.
//! 2. **Overlap resolution** is a greedy maximal non-overlapping
//!    selection over a total ranking: tier 1 before tier 2, then longer
//!    before shorter, then lower VA first. Walking that ranking, a
//!    candidate is accepted unless its `[va, va + len)` range intersects
//!    an already-accepted string. So a code-referenced start always beats
//!    an overlapping data-only one whatever their lengths; among equals
//!    the longer wins, and anything fully contained in it is dropped.
//!    Real Go literals are packed, never overlapping, so this only ever
//!    discards false sub-pointers.
//!
//! The result is sorted by `(va, len)` and is byte-identical across runs
//! on the same image.
//!
//! # Resource caps
//!
//! Every input is attacker-controlled, so the work is bounded up front:
//! [`MAX_STR_LEN`] caps one literal, [`MAX_STRINGS`] caps the candidates
//! the two tiers may raise between them, [`MAX_REGION_SCAN`] caps the
//! bytes swept per region in either tier, and [`LOOKAHEAD`] caps the
//! tier-1 instruction window with [`MAX_PENDING`] half-headers tracked in
//! each side of it. Both sweeps are linear in the region size; an
//! undecodable x86 byte advances the scan by one byte and clears the
//! window rather than aborting it.
//!
//! Peak *memory* needs one more rule, because a crafted image can aim a
//! million pointers at the same 64 KiB run: candidates are carried as
//! `(va, len, source)` triples and their text is decoded only after the
//! gate and the overlap resolution have run. Survivors are disjoint by
//! construction, so the returned text totals no more than the image's own
//! read-only bytes.
//!
//! # Deferred
//!
//! 64-bit little-endian images only: tier 2 assumes 8-byte words and
//! little-endian order, which is every Go target this crate loads in
//! practice. Tier 1 models the `LEA`/`ADRP`+`ADD` idioms only — a length
//! held in a register, computed, or spilled and reloaded is not tracked,
//! and absolute (non-rip-relative) `lea` forms, which Go's
//! position-independent output does not emit, are ignored.

use crate::model::{Arch, Image, Region};
use crate::{aarch64, gopcln, x86};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Upper bound on one recovered literal's byte length. Real Go literals
/// are far shorter; this caps a hostile length word.
const MAX_STR_LEN: usize = 1 << 16;

/// Upper bound on how many strings are returned in total.
const MAX_STRINGS: usize = 1 << 20;

/// Per-region cap on bytes swept, in either tier.
const MAX_REGION_SCAN: usize = 64 * 1024 * 1024;

/// How many instructions a half-materialized header stays live in the
/// tier-1 pairing window, waiting for its other half.
const LOOKAHEAD: usize = 8;

/// How many un-paired addresses — and, separately, how many un-paired
/// lengths — the tier-1 window holds at once.
const MAX_PENDING: usize = 8;

/// How far back [`Mem::region_at`] walks through regions that start at or
/// before a VA, bounding the cost of a lookup on a pathologically nested
/// region list.
const MAX_REGION_PROBE: usize = 64;

/// Word size of the (pointer, length) headers tier 2 scans for.
const WORD: usize = 8;

/// Characters of a literal shown per [`render`] line.
const PREVIEW: usize = 64;

/// Which tier recovered a [`GoString`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Tier 1: an address and a length materialized as immediates by the
    /// code (`lea`+`mov`, `ADRP`+`ADD`+`MOVZ`).
    Code,
    /// Tier 2: an adjacent `(pointer, length)` word pair in data.
    Data,
}

impl Source {
    /// Short label for reports.
    pub fn label(self) -> &'static str {
        match self {
            Source::Code => "code",
            Source::Data => "data",
        }
    }

    /// Ranking key for overlap resolution: tier 1 outranks tier 2.
    fn rank(self) -> u8 {
        match self {
            Source::Code => 0,
            Source::Data => 1,
        }
    }
}

/// One recovered Go string literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoString {
    /// Virtual address of the first byte of the literal's data.
    pub va: u64,
    /// Length in bytes, exactly as the recovered header states it (never
    /// zero, never above [`MAX_STR_LEN`]).
    pub len: usize,
    /// The decoded text: exactly `len` bytes of validated UTF-8, no
    /// terminator.
    pub text: String,
    /// Which tier found it (see [`Source`]).
    pub source: Source,
}

/// Recover Go string literals from a loaded image, best-effort.
///
/// Returns the literals sorted by `(va, len)`, or an empty vector for a
/// non-Go image or one with no recoverable string structure. Never
/// errors, never panics, and never allocates unboundedly: see the module
/// docs for the two tiers, the merge rule, and the caps.
pub fn recover(image: &dyn Image) -> Vec<GoString> {
    if !gopcln::is_go(image) {
        return Vec::new();
    }
    let mem = Mem::new(image);
    let mut all = Vec::new();
    match image.arch() {
        Arch::X86_64 => scan_code_x86(&mem, &mut all),
        Arch::Aarch64 => scan_code_a64(&mem, &mut all),
        Arch::Other => {}
    }
    scan_data(&mem, &mut all);
    materialize(&mem, resolve(gate(&mem, all)))
}

/// Render recovered strings as a readable report.
///
/// Deterministic for a given input and always newline-terminated, so a
/// CLI can print it verbatim. An empty result renders as a single
/// explanatory line rather than an empty string.
pub fn render(strings: &[GoString]) -> String {
    if strings.is_empty() {
        return "no Go strings recovered\n".to_string();
    }
    let code = strings
        .iter()
        .filter(|s| s.source == Source::Code)
        .count();
    let data = strings.len().saturating_sub(code);
    let mut out = format!(
        "Go strings: {} ({code} code-referenced, {data} data-referenced)\n",
        strings.len()
    );
    for s in strings {
        out.push_str(&format!(
            "  0x{:016x}  {:<4}  \"{}\"\n",
            s.va,
            s.source.label(),
            escape(&s.text, PREVIEW)
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Bounds-checked memory view
// ---------------------------------------------------------------------------

/// A read-only, bounds-checked view of an image's mapped memory.
///
/// Every accessor resolves a VA through the image's regions *and* its
/// VA-to-offset mapping, so a read can neither straddle a region boundary
/// nor leave the file buffer. All arithmetic is checked; no path here can
/// panic on hostile data.
struct Mem<'a> {
    image: &'a dyn Image,
    /// The image's regions, sorted by (VA, size), zero-size ones dropped.
    regions: Vec<Region>,
}

impl<'a> Mem<'a> {
    fn new(image: &'a dyn Image) -> Mem<'a> {
        let mut regions = image.regions();
        regions.retain(|r| r.size > 0);
        regions.sort_by_key(|r| (r.va, r.size));
        Mem { image, regions }
    }

    /// The region containing `va`, if any.
    fn region_at(&self, va: u64) -> Option<&Region> {
        let idx = self.regions.partition_point(|r| r.va <= va);
        self.regions[..idx]
            .iter()
            .rev()
            .take(MAX_REGION_PROBE)
            .find(|r| va.wrapping_sub(r.va) < r.size)
    }

    /// `len` file-backed bytes at `va`, all inside one region.
    fn bytes_at(&self, va: u64, len: usize) -> Option<&'a [u8]> {
        let region = self.region_at(va)?;
        let end = va.checked_add(len as u64)?;
        if end > region.va.checked_add(region.size)? {
            return None;
        }
        let off = self.image.va_to_offset(va)?;
        let stop = off.checked_add(len)?;
        self.image.bytes().get(off..stop)
    }

    /// Whether `va` lies in a region a Go string blob may live in:
    /// readable and not writable. Executable is *not* disqualifying —
    /// Mach-O keeps `__rodata` inside the r-x `__TEXT` segment.
    fn is_read_only(&self, va: u64) -> bool {
        self.region_at(va)
            .is_some_and(|r| r.perms.r && !r.perms.w)
    }

    /// The file-backed bytes of one region, clipped to the buffer.
    fn region_bytes(&self, r: &Region) -> Option<&'a [u8]> {
        let off = self.image.va_to_offset(r.va)?;
        let end = off
            .saturating_add(r.size as usize)
            .min(self.image.bytes().len());
        self.image.bytes().get(off..end)
    }

    /// `(base VA, file-backed bytes)` for every region passing `keep`, in
    /// the canonical region order.
    fn ranges(&self, keep: impl Fn(&Region) -> bool) -> Vec<(u64, &'a [u8])> {
        self.regions
            .iter()
            .filter(|r| keep(r))
            .filter_map(|r| Some((r.va, self.region_bytes(r)?)))
            .filter(|(_, b)| !b.is_empty())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Candidate validation
// ---------------------------------------------------------------------------

/// Whether `b` may appear in a recovered literal. Tab, newline and
/// carriage return are ordinary in Go source strings; every other C0
/// control, NUL, and `0x7F` mark bytes that are not text.
fn is_text_byte(b: u8) -> bool {
    match b {
        b'\t' | b'\n' | b'\r' => true,
        0x00..=0x1F | 0x7F => false,
        _ => true,
    }
}

/// A validated header, before its text is decoded.
///
/// Candidates carry no `String`: a hostile image can present a million
/// pointers at the same 64 KiB run, and materializing each one's text
/// during the scan would allocate gigabytes. Text is decoded once, for
/// the survivors of [`gate`] and [`resolve`] only — and those are
/// disjoint, so their total size is bounded by the image itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cand {
    va: u64,
    len: usize,
    source: Source,
}

/// Validate a `(pointer, length)` header, or `None` when any of the
/// module's acceptance rules fails.
fn candidate(mem: &Mem, va: u64, len: u64, source: Source) -> Option<Cand> {
    let len = usize::try_from(len).ok()?;
    if len == 0 || len > MAX_STR_LEN || !mem.is_read_only(va) {
        return None;
    }
    let bytes = mem.bytes_at(va, len)?;
    if !bytes.iter().all(|&b| is_text_byte(b)) {
        return None;
    }
    std::str::from_utf8(bytes).ok()?;
    Some(Cand { va, len, source })
}

/// Decode the text of each surviving candidate, dropping any whose bytes
/// no longer read back (unreachable — [`candidate`] already validated
/// them — but this keeps the step total).
fn materialize(mem: &Mem, cands: Vec<Cand>) -> Vec<GoString> {
    cands
        .into_iter()
        .filter_map(|c| {
            let bytes = mem.bytes_at(c.va, c.len)?;
            let text = std::str::from_utf8(bytes).ok()?;
            Some(GoString {
                va: c.va,
                len: c.len,
                text: text.to_string(),
                source: c.source,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tier 1 — instruction-materialized headers
// ---------------------------------------------------------------------------

/// The sliding pairing window: the half-materialized string headers seen
/// in the last [`LOOKAHEAD`] instructions.
///
/// Go emits the two immediates in either order — the address first when
/// the header is built in registers, the length first when the register
/// allocator got there the other way round — so both halves wait for
/// their partner and either one completes the pair.
#[derive(Debug, Default)]
struct Window {
    /// Materialized addresses awaiting a length, with the instructions
    /// each has left to live.
    addrs: VecDeque<(u64, usize)>,
    /// Length immediates awaiting an address, likewise.
    lens: VecDeque<(u64, usize)>,
}

impl Window {
    /// Age both halves by one instruction, dropping the expired.
    fn age(&mut self) {
        for q in [&mut self.addrs, &mut self.lens] {
            for e in q.iter_mut() {
                e.1 = e.1.saturating_sub(1);
            }
            q.retain(|e| e.1 > 0);
        }
    }

    /// Forget everything (a decode desynchronized: the window is a lie).
    fn clear(&mut self) {
        self.addrs.clear();
        self.lens.clear();
    }

    /// Record `value` in `q`, evicting the oldest past [`MAX_PENDING`].
    fn remember(q: &mut VecDeque<(u64, usize)>, value: u64) {
        if q.len() >= MAX_PENDING {
            q.pop_front();
        }
        q.push_back((value, LOOKAHEAD));
    }

    /// Offer a materialized address: pair it with the newest waiting
    /// length that validates, else park it awaiting one.
    fn offer_addr(&mut self, mem: &Mem, va: u64, out: &mut Vec<Cand>) {
        for idx in (0..self.lens.len()).rev() {
            let Some(&(len, _)) = self.lens.get(idx) else {
                continue;
            };
            if let Some(s) = candidate(mem, va, len, Source::Code) {
                out.push(s);
                self.lens.remove(idx);
                return;
            }
        }
        Window::remember(&mut self.addrs, va);
    }

    /// Offer a length immediate: pair it with the newest waiting address
    /// that validates, else park it awaiting one.
    fn offer_len(&mut self, mem: &Mem, len: u64, out: &mut Vec<Cand>) {
        if len == 0 || len > MAX_STR_LEN as u64 {
            return;
        }
        for idx in (0..self.addrs.len()).rev() {
            let Some(&(va, _)) = self.addrs.get(idx) else {
                continue;
            };
            if let Some(s) = candidate(mem, va, len, Source::Code) {
                out.push(s);
                self.addrs.remove(idx);
                return;
            }
        }
        Window::remember(&mut self.lens, len);
    }
}

/// Sweep the executable regions of an x86-64 image for `lea`+`mov imm`
/// string headers.
fn scan_code_x86(mem: &Mem, out: &mut Vec<Cand>) {
    for (base, bytes) in mem.ranges(|r| r.perms.x) {
        let scan_len = bytes.len().min(MAX_REGION_SCAN);
        let mut window = Window::default();
        let mut i = 0usize;
        while i < scan_len {
            if out.len() >= MAX_STRINGS {
                return;
            }
            let va = base.wrapping_add(i as u64);
            let limit = (scan_len - i).min(x86::MAX_INSTRUCTION_LEN);
            let Ok(ins) = x86::decode(&bytes[i..i + limit], va) else {
                // An undecodable byte cannot be skipped reliably on a
                // variable-length ISA: resynchronize one byte on and
                // abandon the window rather than trust it.
                window.clear();
                i += 1;
                continue;
            };
            if ins.length == 0 {
                window.clear();
                i += 1;
                continue;
            }
            window.age();
            match ins.opcode {
                x86::Opcode::Lea => {
                    if let Some(target) = x86_lea_target(&ins, va)
                        && mem.is_read_only(target)
                    {
                        window.offer_addr(mem, target, out);
                    }
                }
                x86::Opcode::Mov => {
                    if let Some(x86::Operand::Imm(imm)) = ins.operands.last()
                        && let Ok(len) = u64::try_from(*imm)
                    {
                        window.offer_len(mem, len, out);
                    }
                }
                // A (pointer, length) pair is always materialized within one
                // basic block, so a half-formed header must not outlive a
                // control transfer: clear it rather than let a stray pointer
                // pair with the next block's length.
                x86::Opcode::Ret | x86::Opcode::Call | x86::Opcode::Jmp | x86::Opcode::Jcc(_) => {
                    window.clear();
                }
                _ => {}
            }
            i += ins.length as usize;
        }
    }
}

/// The absolute VA a `lea r64, [rip + disp32]` computes, or `None` for
/// every other addressing form.
fn x86_lea_target(ins: &x86::Instruction, va: u64) -> Option<u64> {
    let x86::Operand::Mem {
        base: None,
        index: None,
        disp,
        rip_relative: true,
        ..
    } = *ins.operands.get(1)?
    else {
        return None;
    };
    // `[rip + disp]` is relative to the *next* instruction (SDM §2.2.1.6).
    Some(
        va.wrapping_add(u64::from(ins.length))
            .wrapping_add(disp as u64),
    )
}

/// Sweep the executable regions of an AArch64 image for
/// `ADR`/`ADRP`+`ADD` plus `MOVZ` string headers.
fn scan_code_a64(mem: &Mem, out: &mut Vec<Cand>) {
    use aarch64::Opcode as O;
    let size = aarch64::Instruction::SIZE;
    for (base, bytes) in mem.ranges(|r| r.perms.x) {
        let scan_len = bytes.len().min(MAX_REGION_SCAN);
        let mut window = Window::default();
        // Per-register page address most recently set by an `ADRP`; the
        // `ADD #lo12` that completes the pair reads it.
        let mut page = [None::<u64>; 32];
        let mut i = 0usize;
        while i + size <= scan_len {
            if out.len() >= MAX_STRINGS {
                return;
            }
            let va = base.wrapping_add(i as u64);
            let Ok(ins) = aarch64::decode(&bytes[i..i + size], va) else {
                // A64 decoding is total; a short tail is the only error,
                // and the loop bound already excludes it.
                break;
            };
            window.age();
            match ins.opcode {
                O::Adrp { rd, target } => page[rd as usize & 31] = Some(target),
                O::Adr { rd, target } => {
                    page[rd as usize & 31] = None;
                    if mem.is_read_only(target) {
                        window.offer_addr(mem, target, out);
                    }
                }
                O::AddImm {
                    sf: true,
                    set_flags: false,
                    rd,
                    rn,
                    imm,
                } => {
                    if let Some(p) = page[rn as usize & 31] {
                        let target = p.wrapping_add(u64::from(imm));
                        if mem.is_read_only(target) {
                            window.offer_addr(mem, target, out);
                        }
                    }
                    // The destination now holds a full address, not a page.
                    page[rd as usize & 31] = None;
                }
                O::Movz { rd, imm, shift, .. } => {
                    page[rd as usize & 31] = None;
                    let len = u64::from(imm)
                        .checked_shl(u32::from(shift))
                        .unwrap_or(u64::MAX);
                    window.offer_len(mem, len, out);
                }
                O::Movn { rd, .. }
                | O::Movk { rd, .. }
                | O::SubImm { rd, .. }
                | O::AddImm { rd, .. } => page[rd as usize & 31] = None,
                O::Ldr { rt, .. } | O::LdrLit { rt, .. } => page[rt as usize & 31] = None,
                O::Ldp { rt, rt2, .. } => {
                    page[rt as usize & 31] = None;
                    page[rt2 as usize & 31] = None;
                }
                // Go's assembler renders `MOVD $n, Rd` as the logical
                // move `ORR Rd, ZR, #n` whenever `n` is a bitmask
                // immediate, which every small run of set bits is — a
                // length of 3 or 4 arrives this way, not as `MOVZ`.
                // The decoder now delivers these directly as `LogImm`
                // (it once left them `Unknown` for [`a64_move_bitmask`]
                // to re-parse); only the `Rn == ZR` form is a move.
                O::LogImm {
                    op: aarch64::LogOp::Orr,
                    set_flags: false,
                    rd,
                    rn: 31,
                    imm,
                    ..
                } => {
                    page[rd as usize & 31] = None;
                    window.offer_len(mem, imm, out);
                }
                // A (pointer, length) pair is materialized within one basic
                // block; clear a half-formed header at a control transfer so
                // a stray pointer can't pair with the next block's length.
                O::Ret { .. } | O::B { .. } | O::Br { .. } | O::Bl { .. } | O::Blr { .. } => {
                    window.clear();
                }
                _ => {}
            }
            i += size;
        }
    }
}

/// The `(destination, value)` a `MOV <Rd>, #imm` written as
/// `ORR <Rd>, <ZR>, #<bitmask>` materializes, or `None` for every other
/// encoding.
///
/// A64 has no move-immediate for a general constant: `MOVZ` covers a
/// 16-bit field at a 16-bit-aligned shift, and everything else that is a
/// rotated run of set bits goes through the logical-immediate `ORR` with
/// `XZR` as the source (ARM ARM C4.1.93, `DecodeBitMasks`). Go's
/// assembler picks whichever fits, so a string length of 3, 4, 6, 7, 8 …
/// arrives as `ORR` while 5, 9, 10 … arrive as `MOVZ`. Only the
/// `Rn == ZR` form is a move; `ORR Rd, Rn, #imm` with a real source is
/// not, and yields `None`.
///
/// The scan itself now reads these from the decoder's `LogImm` form;
/// this raw-word reader survives as the tests' independent oracle for
/// building bitmask-immediate fixtures.
#[cfg(test)]
fn a64_move_bitmask(word: u32) -> Option<(u8, u64)> {
    // sf | opc=01 | 100100 | N | immr | imms | Rn | Rd — bits 30:23 pin
    // the ORR (immediate) encoding.
    if word & 0x7F80_0000 != 0x3200_0000 || (word >> 5) & 31 != 31 {
        return None;
    }
    let sf = word >> 31 & 1;
    let n = word >> 22 & 1;
    let immr = word >> 16 & 0x3F;
    let imms = word >> 10 & 0x3F;
    // `N:NOT(imms)` names the element width: its highest set bit is
    // log2(esize), and a 32-bit operand may not set `N`.
    if sf == 0 && n == 1 {
        return None;
    }
    let width = (n << 6) | (!imms & 0x3F);
    if width == 0 {
        return None;
    }
    let len = 31 - width.leading_zeros();
    let esize = 1u32 << len;
    let s = imms & (esize - 1);
    // An all-ones element is the reserved encoding.
    if s == esize - 1 {
        return None;
    }
    let r = immr & (esize - 1);
    let mask = if esize >= 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    let elem = if s + 1 >= 64 {
        u64::MAX
    } else {
        (1u64 << (s + 1)) - 1
    };
    let rotated = if r == 0 {
        elem
    } else {
        ((elem >> r) | (elem << (esize - r))) & mask
    };
    let mut value = 0u64;
    let mut shift = 0u32;
    while shift < 64 {
        value |= rotated << shift;
        shift += esize;
    }
    if sf == 0 {
        value &= 0xFFFF_FFFF;
    }
    Some(((word & 31) as u8, value))
}

// ---------------------------------------------------------------------------
// Tier 2 — data-resident headers
// ---------------------------------------------------------------------------

/// Sweep the readable regions for `[ptr: 8][len: 8]` headers, appending
/// to `out` until the two tiers together reach [`MAX_STRINGS`].
fn scan_data(mem: &Mem, out: &mut Vec<Cand>) {
    'regions: for (base, bytes) in mem.ranges(|r| r.perms.r) {
        let scan_len = bytes.len().min(MAX_REGION_SCAN);
        // Walk the region's own 8-byte-aligned virtual addresses, which
        // need not be aligned to the region's file offset.
        let skew = (WORD - (base as usize % WORD)) % WORD;
        let mut i = skew;
        while i + 2 * WORD <= scan_len {
            if out.len() >= MAX_STRINGS {
                break 'regions;
            }
            // The length word is the cheap discriminator: a plausible
            // length leaves the top six bytes zero, so this rejects
            // almost every position before any region lookup.
            let len = read_u64(bytes, i + WORD);
            if len != 0 && len <= MAX_STR_LEN as u64 {
                let ptr = read_u64(bytes, i);
                if let Some(s) = candidate(mem, ptr, len, Source::Data) {
                    out.push(s);
                }
            }
            i += WORD;
        }
    }
}

/// The boundary gate: keep only candidates that end where a literal
/// plausibly can — at another candidate's start, at a byte that is not
/// text (a NUL or a control byte), or at the edge of the pointer's own
/// region. See the module docs for why both tiers need it.
fn gate(mem: &Mem, candidates: Vec<Cand>) -> Vec<Cand> {
    let starts: BTreeSet<u64> = candidates.iter().map(|s| s.va).collect();
    candidates
        .into_iter()
        .filter(|s| {
            let end = s.va.saturating_add(s.len as u64);
            starts.contains(&end) || byte_after(mem, s.va, s.len).is_none_or(|b| !is_text_byte(b))
        })
        .collect()
}

/// Little-endian `u64` at `off`, or 0 when the window is short (the
/// caller's bound makes that unreachable; this keeps the read total).
fn read_u64(bytes: &[u8], off: usize) -> u64 {
    match bytes.get(off..off.saturating_add(WORD)) {
        Some(b) => u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        None => 0,
    }
}

/// The byte one past `[va, va + len)`, when it is still inside the same
/// region and file-backed; `None` once the run reaches the region's edge.
fn byte_after(mem: &Mem, va: u64, len: usize) -> Option<u8> {
    let region = mem.region_at(va)?;
    let end = va.checked_add(len as u64)?;
    if end >= region.va.checked_add(region.size)? {
        return None;
    }
    mem.bytes_at(end, 1).map(|b| b[0])
}

// ---------------------------------------------------------------------------
// Merge, dedup, overlap resolution
// ---------------------------------------------------------------------------

/// Dedup by `(va, len)` and resolve overlaps, returning the survivors
/// sorted by `(va, len)`. See the module docs for the exact rule.
fn resolve(mut all: Vec<Cand>) -> Vec<Cand> {
    all.sort_by(|a, b| {
        a.va.cmp(&b.va)
            .then_with(|| a.len.cmp(&b.len))
            .then_with(|| a.source.rank().cmp(&b.source.rank()))
    });
    // `dedup_by` drops the *later* of each equal pair, so the tier-1
    // label survives a literal both tiers found.
    all.dedup_by(|a, b| a.va == b.va && a.len == b.len);
    all.truncate(MAX_STRINGS);

    // Greedy maximal non-overlapping selection over the ranking
    // (tier, longer, lower VA).
    let mut order: Vec<usize> = (0..all.len()).collect();
    order.sort_by_key(|&i| {
        (
            all[i].source.rank(),
            std::cmp::Reverse(all[i].len),
            all[i].va,
        )
    });
    // Accepted ranges as `va -> end`; disjoint by construction, so a
    // clash can only be with the nearest neighbor on either side.
    let mut accepted: BTreeMap<u64, u64> = BTreeMap::new();
    let mut keep = vec![false; all.len()];
    for &i in &order {
        let va = all[i].va;
        let end = va.saturating_add(all[i].len as u64);
        let clashes = accepted
            .range(..=va)
            .next_back()
            .is_some_and(|(_, &e)| e > va)
            || accepted.range(va..).next().is_some_and(|(&s, _)| s < end);
        if !clashes {
            accepted.insert(va, end);
            keep[i] = true;
        }
    }

    let mut out: Vec<Cand> = all
        .into_iter()
        .zip(keep)
        .filter_map(|(s, k)| k.then_some(s))
        .collect();
    out.sort_by(|a, b| a.va.cmp(&b.va).then_with(|| a.len.cmp(&b.len)));
    out
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Escape `text` for inline display and truncate it to `max` characters,
/// marking a truncation with a trailing `...` — the same alphabet
/// [`crate::listing`] uses, so a report survives being diffed.
fn escape(text: &str, max: usize) -> String {
    let mut out = String::new();
    for (i, ch) in text.chars().enumerate() {
        if i >= max {
            out.push_str("...");
            break;
        }
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7F => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ImportSlot, Perms, Symbol};

    // -- fixtures ----------------------------------------------------------

    const TEXT_VA: u64 = 0x1000;
    const RODATA_VA: u64 = 0x2000;
    const DATA_VA: u64 = 0x3000;
    const PCLN_VA: u64 = 0x4000;

    /// An image built from explicit regions over one flat buffer, with
    /// VA-to-offset resolved through the regions themselves — enough to
    /// exercise both tiers without a real container format.
    struct FakeImage {
        arch: Arch,
        data: Vec<u8>,
        /// `(name, va, size, file offset, permissions)`.
        regions: Vec<(String, u64, u64, usize, Perms)>,
    }

    impl Image for FakeImage {
        fn arch(&self) -> Arch {
            self.arch
        }
        fn entry_points(&self) -> Vec<u64> {
            Vec::new()
        }
        fn regions(&self) -> Vec<Region> {
            self.regions
                .iter()
                .map(|(name, va, size, _, perms)| Region {
                    name: name.clone(),
                    va: *va,
                    size: *size,
                    perms: *perms,
                })
                .collect()
        }
        fn symbols(&self) -> Vec<Symbol> {
            Vec::new()
        }
        fn import_slots(&self) -> Vec<ImportSlot> {
            Vec::new()
        }
        fn va_to_offset(&self, va: u64) -> Option<usize> {
            let (_, rva, _, off, _) = self
                .regions
                .iter()
                .find(|(_, rva, size, _, _)| va >= *rva && va - *rva < *size)?;
            let at = off.checked_add(usize::try_from(va - rva).ok()?)?;
            (at < self.data.len()).then_some(at)
        }
        fn bytes(&self) -> &[u8] {
            &self.data
        }
    }

    fn ro() -> Perms {
        Perms {
            r: true,
            w: false,
            x: false,
        }
    }
    fn rx() -> Perms {
        Perms {
            r: true,
            w: false,
            x: true,
        }
    }
    fn rw() -> Perms {
        Perms {
            r: true,
            w: true,
            x: false,
        }
    }

    /// The eight bytes `gopcln::locate` needs to call an image Go: the
    /// Go 1.18 magic, the `00 00` pad, `pcquantum` 1 and `ptrsize` 8.
    fn go_header() -> Vec<u8> {
        let mut v = 0xffff_fff0u32.to_le_bytes().to_vec();
        v.extend_from_slice(&[0, 0, 1, 8]);
        v
    }

    /// A Go-looking image with a `.text`, a `.rodata`, a `.data` and a
    /// `.gopclntab` region, each placed at its own file offset.
    fn go_image(arch: Arch, text: &[u8], rodata: &[u8], data: &[u8]) -> FakeImage {
        let pcln = go_header();
        let mut buf = Vec::new();
        let text_off = buf.len();
        buf.extend_from_slice(text);
        let rodata_off = buf.len();
        buf.extend_from_slice(rodata);
        let data_off = buf.len();
        buf.extend_from_slice(data);
        let pcln_off = buf.len();
        buf.extend_from_slice(&pcln);
        let mut regions = vec![(
            ".gopclntab".to_string(),
            PCLN_VA,
            pcln.len() as u64,
            pcln_off,
            ro(),
        )];
        if !text.is_empty() {
            regions.push((".text".into(), TEXT_VA, text.len() as u64, text_off, rx()));
        }
        if !rodata.is_empty() {
            regions.push((
                ".rodata".into(),
                RODATA_VA,
                rodata.len() as u64,
                rodata_off,
                ro(),
            ));
        }
        if !data.is_empty() {
            regions.push((".data".into(), DATA_VA, data.len() as u64, data_off, rw()));
        }
        FakeImage {
            arch,
            data: buf,
            regions,
        }
    }

    // -- hand-assembled instruction helpers --------------------------------

    /// `lea <reg64>, [rip + disp32]`, seven bytes:
    /// `REX.W (0x48)`, opcode `8D`, ModRM `mod=00 reg=<reg> rm=101`
    /// (the 64-bit rip-relative form), then the little-endian disp32.
    /// The displacement is taken from the *next* instruction's VA, so a
    /// `lea` at `va` targeting `target` uses `target - (va + 7)`.
    fn lea_rip(reg: u8, va: u64, target: u64) -> Vec<u8> {
        let disp = (target.wrapping_sub(va.wrapping_add(7))) as u32;
        let mut v = vec![0x48, 0x8D, (reg & 7) << 3 | 0x05];
        v.extend_from_slice(&disp.to_le_bytes());
        v
    }

    /// `mov <reg32>, imm32`, five bytes: opcode `B8+rd` then the
    /// little-endian imm32.
    fn mov_imm32(reg: u8, imm: u32) -> Vec<u8> {
        let mut v = vec![0xB8 | (reg & 7)];
        v.extend_from_slice(&imm.to_le_bytes());
        v
    }

    /// `adrp <Xd>, <page>`: `op=1`, `immlo` in bits 30:29, `10000` in
    /// bits 28:24, `immhi` in bits 23:5, `Rd` in bits 4:0. The immediate
    /// is the signed count of 4 KiB pages from the instruction's own page.
    fn adrp(rd: u8, va: u64, target: u64) -> Vec<u8> {
        let pages = ((target & !0xFFF) as i64 - (va & !0xFFF) as i64) >> 12;
        let imm = (pages as u32) & 0x1F_FFFF;
        let w = 0x9000_0000u32
            | (imm & 3) << 29
            | (imm >> 2) << 5
            | u32::from(rd & 31);
        w.to_le_bytes().to_vec()
    }

    /// `add <Xd>, <Xn>, #imm12` (64-bit, no shift, no flags).
    fn add_imm(rd: u8, rn: u8, imm: u32) -> Vec<u8> {
        let w = 0x9100_0000u32 | (imm & 0xFFF) << 10 | u32::from(rn & 31) << 5 | u32::from(rd & 31);
        w.to_le_bytes().to_vec()
    }

    /// `movz <Wd>, #imm16` (32-bit form, no shift).
    fn movz(rd: u8, imm: u16) -> Vec<u8> {
        let w = 0x5280_0000u32 | u32::from(imm) << 5 | u32::from(rd & 31);
        w.to_le_bytes().to_vec()
    }

    /// `orr <Xd>, xzr, #<bitmask>` — the `MOV <Xd>, #imm` A64 uses for a
    /// constant `MOVZ` cannot express. `immr`/`imms` are the raw
    /// `DecodeBitMasks` fields: with `N` set the element is 64 bits wide
    /// and the value is `ones(imms + 1)` rotated right by `immr`.
    fn orr_zr(rd: u8, immr: u32, imms: u32) -> Vec<u8> {
        let w = 0xB240_0000u32
            | (immr & 0x3F) << 16
            | (imms & 0x3F) << 10
            | 31 << 5
            | u32::from(rd & 31);
        w.to_le_bytes().to_vec()
    }

    /// `orr <Xd>, xzr, #<n>` for the small `n` Go emits as a string
    /// length, found by asking [`a64_move_bitmask`] what each encoding
    /// means. Panics for an `n` no bitmask immediate can hold.
    fn orr_len(rd: u8, n: u64) -> Vec<u8> {
        for immr in 0..64u32 {
            for imms in 0..64u32 {
                let word = orr_zr(rd, immr, imms);
                let raw = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                if a64_move_bitmask(raw) == Some((rd & 31, n)) {
                    return word;
                }
            }
        }
        panic!("{n} is not a bitmask immediate");
    }

    /// A `(pointer, length)` word pair, little-endian.
    fn pair(ptr: u64, len: u64) -> Vec<u8> {
        let mut v = ptr.to_le_bytes().to_vec();
        v.extend_from_slice(&len.to_le_bytes());
        v
    }

    /// Deterministic pseudo-random bytes (xorshift64), so the robustness
    /// tests are reproducible without a dependency.
    fn pseudo_random(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 33) as u8
            })
            .collect()
    }

    fn texts(strings: &[GoString]) -> Vec<&str> {
        strings.iter().map(|s| s.text.as_str()).collect()
    }

    // -- tier 1: the core win ----------------------------------------------

    #[test]
    fn adjacent_literals_with_no_separator_are_two_distinct_strings() {
        // "hello" and "world" packed as `helloworld`, each addressed by a
        // `lea` and measured by the `mov imm` that follows it.
        let mut text = Vec::new();
        text.extend(lea_rip(0, TEXT_VA, RODATA_VA)); // lea rax, "hello"
        text.extend(mov_imm32(1, 5)); // mov ecx, 5
        text.extend(lea_rip(2, TEXT_VA + 12, RODATA_VA + 5)); // lea rdx, "world"
        text.extend(mov_imm32(6, 5)); // mov esi, 5
        text.push(0xC3); // ret
        let img = go_image(Arch::X86_64, &text, b"helloworld", &[]);

        let found = recover(&img);
        assert_eq!(texts(&found), ["hello", "world"], "{found:#?}");
        assert!(found.iter().all(|s| s.source == Source::Code));
        assert_eq!(found[0].va, RODATA_VA);
        assert_eq!(found[1].va, RODATA_VA + 5);
        // The generic C-string scanner's failure mode, explicitly refused.
        assert!(found.iter().all(|s| s.text != "helloworld"));
    }

    #[test]
    fn aarch64_adrp_add_movz_recovers_a_literal() {
        let mut text = Vec::new();
        text.extend(adrp(0, TEXT_VA, RODATA_VA));
        text.extend(add_imm(0, 0, 0));
        text.extend(movz(1, 5));
        text.extend(adrp(2, TEXT_VA + 12, RODATA_VA + 5));
        text.extend(add_imm(2, 2, 5));
        text.extend(movz(3, 5));
        text.extend(0xD65F_03C0u32.to_le_bytes()); // ret
        let img = go_image(Arch::Aarch64, &text, b"helloworld", &[]);

        let found = recover(&img);
        assert_eq!(texts(&found), ["hello", "world"], "{found:#?}");
        assert!(found.iter().all(|s| s.source == Source::Code));
    }

    #[test]
    fn aarch64_lengths_moved_as_logical_immediates_still_split_the_blob() {
        // The shape a real go1.25 arm64 build emits for
        // `func (Dog) Noise() string { return "woof" }`: the length is a
        // `MOV Xd, #4`, which the assembler encodes as
        // `ORR Xd, XZR, #4` — not `MOVZ`, which cannot hold 4 rotated out
        // of a run of ones. Recognizing only `MOVZ` loses both literals
        // *and* leaves the address parked in the window, where a later
        // unrelated `MOVZ` fabricates one run-together string over both.
        let mut text = Vec::new();
        text.extend(adrp(0, TEXT_VA, RODATA_VA));
        text.extend(add_imm(0, 0, 0));
        text.extend(orr_len(1, 4)); // mov x1, #4
        text.extend(0xD65F_03C0u32.to_le_bytes()); // ret
        text.extend(adrp(0, TEXT_VA + 16, RODATA_VA + 4));
        text.extend(add_imm(0, 0, 4));
        text.extend(orr_len(1, 4)); // mov x1, #4
        text.extend(0xD65F_03C0u32.to_le_bytes()); // ret
        let img = go_image(Arch::Aarch64, &text, b"woofmeow\0\0\0\0\0\0\0\0", &[]);

        let found = recover(&img);
        assert_eq!(texts(&found), ["woof", "meow"], "{found:#?}");
        assert!(found.iter().all(|s| s.source == Source::Code));
        assert_eq!(found[0].va, RODATA_VA);
        assert_eq!(found[1].va, RODATA_VA + 4);
    }

    #[test]
    fn a_logical_immediate_move_is_decoded_only_in_its_move_form() {
        // `ORR X1, XZR, #4` — one bit, rotated right by 62 out of a
        // 64-bit element — is `MOV X1, #4`.
        assert_eq!(a64_move_bitmask(0xB27E_03E1), Some((1, 4)));
        // `ORR X1, XZR, #3` — two bits, unrotated.
        assert_eq!(a64_move_bitmask(0xB240_07E1), Some((1, 3)));
        // The 32-bit form yields a 32-bit value, and may not set `N`.
        assert_eq!(a64_move_bitmask(0x3200_03E2), Some((2, 1)));
        assert_eq!(a64_move_bitmask(0x3240_07E1), None);
        // A real `ORR X1, X2, #3` is a computation, not a move.
        assert_eq!(a64_move_bitmask(0xB240_0441), None);
        // An all-ones element is the reserved `imms` encoding.
        assert_eq!(a64_move_bitmask(0xB27F_FFE1), None);
        // And everything outside the encoding is refused, including the
        // `MOVZ` the other arm already handles.
        for word in [0u32, 0x5280_0081, 0x9100_0000, 0xD65F_03C0, u32::MAX] {
            assert_eq!(a64_move_bitmask(word), None, "{word:#010x}");
        }
    }

    #[test]
    fn a_length_further_than_the_lookahead_window_is_not_matched() {
        let mut text = lea_rip(0, TEXT_VA, RODATA_VA);
        // Nine `nop`s: more than LOOKAHEAD instructions before the length.
        text.extend(std::iter::repeat_n(0x90u8, 9));
        text.extend(mov_imm32(1, 5));
        let img = go_image(Arch::X86_64, &text, b"helloworld", &[]);
        assert!(recover(&img).is_empty());
    }

    #[test]
    fn a_length_materialized_before_its_address_still_pairs() {
        // Go emits the two immediates in whichever order the register
        // allocator produced; the window pairs them either way round.
        let mut text = mov_imm32(1, 5); // mov ecx, 5
        text.extend(lea_rip(0, TEXT_VA + 5, RODATA_VA)); // lea rax, "hello"
        let img = go_image(Arch::X86_64, &text, b"hello\0\0\0\0\0\0\0\0\0\0\0", &[]);

        let found = recover(&img);
        assert_eq!(texts(&found), ["hello"], "{found:#?}");
        assert_eq!(found[0].source, Source::Code);
    }

    #[test]
    fn a_code_pair_ending_mid_blob_is_gated_out() {
        // A `mov` immediate that is not really this literal's length: the
        // run stops inside the packed blob at a byte nothing else starts
        // at, which is how a stray constant fabricates a huge "string".
        let mut text = lea_rip(0, TEXT_VA, RODATA_VA);
        text.extend(mov_imm32(1, 8));
        let img = go_image(Arch::X86_64, &text, b"helloworldextra!", &[]);
        assert!(recover(&img).is_empty(), "{:#?}", recover(&img));

        // The same pairing against a NUL-delimited blob is honest, so the
        // gate passes it.
        let img = go_image(Arch::X86_64, &text, b"hellowor\0extra!!", &[]);
        assert_eq!(texts(&recover(&img)), ["hellowor"]);
    }

    #[test]
    fn a_lea_into_a_writable_region_is_not_a_string_header() {
        let mut text = lea_rip(0, TEXT_VA, DATA_VA);
        text.extend(mov_imm32(1, 5));
        let img = go_image(Arch::X86_64, &text, b"unused..", b"helloworld......");
        // `.data` is writable, so it cannot hold the read-only blob.
        assert!(recover(&img).iter().all(|s| s.va != DATA_VA));
    }

    // -- tier 2 -------------------------------------------------------------

    #[test]
    fn a_pointer_length_pair_in_data_recovers_a_literal() {
        // The literal is NUL-followed, so the boundary gate passes on its
        // own without a second reference.
        let img = go_image(
            Arch::X86_64,
            &[],
            b"config.yaml\0\0\0\0\0",
            &pair(RODATA_VA, 11),
        );
        let found = recover(&img);
        assert_eq!(texts(&found), ["config.yaml"], "{found:#?}");
        assert_eq!(found[0].source, Source::Data);
        assert_eq!(found[0].len, 11);
        assert_eq!(found[0].va, RODATA_VA);
    }

    #[test]
    fn a_mid_string_false_pointer_is_refused() {
        // (RODATA+2, 4) is "llow": a real substring, but it ends inside
        // the blob at a byte nothing else starts at.
        let mut data = pair(RODATA_VA + 2, 4);
        data.extend(pair(RODATA_VA, 10)); // the honest header
        let img = go_image(Arch::X86_64, &[], b"helloworld\0\0\0\0\0\0", &data);

        let found = recover(&img);
        assert_eq!(texts(&found), ["helloworld"], "{found:#?}");
        assert!(found.iter().all(|s| s.text != "llow"));
    }

    #[test]
    fn a_multibyte_utf8_literal_survives_intact() {
        let literal = "héllo wörld ☃";
        let mut blob = literal.as_bytes().to_vec();
        blob.push(0);
        while !blob.len().is_multiple_of(WORD) {
            blob.push(0);
        }
        let img = go_image(
            Arch::X86_64,
            &[],
            &blob,
            &pair(RODATA_VA, literal.len() as u64),
        );
        let found = recover(&img);
        assert_eq!(texts(&found), [literal], "{found:#?}");
        assert_eq!(found[0].len, literal.len());
        assert!(found[0].len > literal.chars().count());
    }

    #[test]
    fn an_invalid_utf8_run_is_refused() {
        // 0xFF is never valid UTF-8, in any position.
        let img = go_image(
            Arch::X86_64,
            &[],
            b"ab\xffcd\0\0\0\0\0\0\0\0\0\0\0",
            &pair(RODATA_VA, 5),
        );
        assert!(recover(&img).is_empty());
    }

    #[test]
    fn a_run_containing_a_control_byte_is_refused() {
        // 0x07 (BEL) is not a permitted text byte; tab/newline are.
        let ok = go_image(
            Arch::X86_64,
            &[],
            b"a\tb\nc\0\0\0\0\0\0\0\0\0\0\0",
            &pair(RODATA_VA, 5),
        );
        assert_eq!(texts(&recover(&ok)), ["a\tb\nc"]);

        let bad = go_image(
            Arch::X86_64,
            &[],
            b"a\x07b\nc\0\0\0\0\0\0\0\0\0\0\0",
            &pair(RODATA_VA, 5),
        );
        assert!(recover(&bad).is_empty());
    }

    // -- rejection rules ----------------------------------------------------

    #[test]
    fn lengths_out_of_range_are_refused() {
        let blob = b"helloworld\0\0\0\0\0\0";
        for len in [0u64, MAX_STR_LEN as u64 + 1, u64::MAX] {
            let img = go_image(Arch::X86_64, &[], blob, &pair(RODATA_VA, len));
            assert!(recover(&img).is_empty(), "len {len}");
        }
        // The boundary itself is honored, not the range around it.
        let img = go_image(Arch::X86_64, &[], blob, &pair(RODATA_VA, 10));
        assert_eq!(texts(&recover(&img)), ["helloworld"]);
    }

    #[test]
    fn pointers_into_unmapped_or_unreadable_memory_are_refused() {
        let img = go_image(
            Arch::X86_64,
            &[],
            b"helloworld\0\0\0\0\0\0",
            &pair(0x9_0000, 5), // maps to no region at all
        );
        assert!(recover(&img).is_empty());

        // A mapped but non-readable region cannot hold a literal either.
        let mut img = go_image(
            Arch::X86_64,
            &[],
            b"helloworld\0\0\0\0\0\0",
            &pair(RODATA_VA, 10),
        );
        assert!(!recover(&img).is_empty());
        for r in &mut img.regions {
            if r.0 == ".rodata" {
                r.4 = Perms {
                    r: false,
                    w: false,
                    x: false,
                };
            }
        }
        assert!(recover(&img).is_empty());
    }

    #[test]
    fn a_run_truncated_at_the_region_edge_is_refused() {
        // The blob is 16 bytes; a 16-byte run starting eight in would
        // leave the region.
        let img = go_image(
            Arch::X86_64,
            &[],
            b"abcdefghijklmnop",
            &pair(RODATA_VA + 8, 16),
        );
        assert!(recover(&img).is_empty());
        // The same run clipped to the edge is fine (and the gate passes
        // because it ends exactly at the region's end).
        let img = go_image(
            Arch::X86_64,
            &[],
            b"abcdefghijklmnop",
            &pair(RODATA_VA + 8, 8),
        );
        assert_eq!(texts(&recover(&img)), ["ijklmnop"]);
    }

    // -- merge, dedup, overlap ---------------------------------------------

    #[test]
    fn an_overlap_keeps_the_code_referenced_string() {
        // Tier 1 says [RODATA, 10) = "helloworld"; tier 2 says
        // [RODATA+5, 5) = "world", which is contained in it.
        let mut text = lea_rip(0, TEXT_VA, RODATA_VA);
        text.extend(mov_imm32(1, 10));
        let img = go_image(
            Arch::X86_64,
            &text,
            b"helloworld\0\0\0\0\0\0",
            &pair(RODATA_VA + 5, 5),
        );
        let found = recover(&img);
        assert_eq!(texts(&found), ["helloworld"], "{found:#?}");
        assert_eq!(found[0].source, Source::Code);
    }

    #[test]
    fn an_overlap_between_equals_keeps_the_longer_string() {
        // Two tier-2 pairs, both passing the gate (the shorter one ends
        // where the longer one does, at the blob's NUL).
        let mut data = pair(RODATA_VA + 5, 5);
        data.extend(pair(RODATA_VA, 10));
        let img = go_image(Arch::X86_64, &[], b"helloworld\0\0\0\0\0\0", &data);
        let found = recover(&img);
        assert_eq!(texts(&found), ["helloworld"], "{found:#?}");
        assert_eq!(found[0].source, Source::Data);
    }

    #[test]
    fn a_literal_found_by_both_tiers_is_reported_once_as_code() {
        let mut text = lea_rip(0, TEXT_VA, RODATA_VA);
        text.extend(mov_imm32(1, 5));
        let img = go_image(
            Arch::X86_64,
            &text,
            b"hello\0\0\0\0\0\0\0\0\0\0\0",
            &pair(RODATA_VA, 5),
        );
        let found = recover(&img);
        assert_eq!(found.len(), 1, "{found:#?}");
        assert_eq!(found[0].text, "hello");
        assert_eq!(found[0].source, Source::Code);
    }

    #[test]
    fn many_pointers_at_one_run_collapse_to_a_single_string() {
        // The shape that would blow memory up if candidates carried their
        // text: thousands of headers naming the same literal.
        let mut data = Vec::new();
        for _ in 0..4096 {
            data.extend(pair(RODATA_VA, 5));
        }
        let img = go_image(Arch::X86_64, &[], b"hello\0\0\0\0\0\0\0\0\0\0\0", &data);
        let found = recover(&img);
        assert_eq!(texts(&found), ["hello"], "{} strings", found.len());
    }

    #[test]
    fn results_are_sorted_by_va_and_never_overlap() {
        let mut text = Vec::new();
        text.extend(lea_rip(0, TEXT_VA, RODATA_VA + 5));
        text.extend(mov_imm32(1, 5));
        text.extend(lea_rip(2, TEXT_VA + 12, RODATA_VA));
        text.extend(mov_imm32(6, 5));
        let img = go_image(Arch::X86_64, &text, b"helloworld", &[]);
        let found = recover(&img);
        assert_eq!(texts(&found), ["hello", "world"]);
        assert!(found.windows(2).all(|w| w[0].va < w[1].va));
        assert!(
            found
                .windows(2)
                .all(|w| w[0].va + w[0].len as u64 <= w[1].va)
        );
    }

    // -- gating and determinism --------------------------------------------

    #[test]
    fn a_non_go_image_yields_nothing() {
        // The same layout with no `.gopclntab`: `is_go` fails, so neither
        // tier runs even though the headers are all there.
        let mut text = lea_rip(0, TEXT_VA, RODATA_VA);
        text.extend(mov_imm32(1, 5));
        let mut img = go_image(
            Arch::X86_64,
            &text,
            b"hello\0\0\0\0\0\0\0\0\0\0\0",
            &pair(RODATA_VA, 5),
        );
        assert!(!recover(&img).is_empty());

        img.regions.retain(|r| r.0 != ".gopclntab");
        // Blank the magic too, so the signature scan cannot find it.
        for b in img.data.iter_mut().rev().take(8) {
            *b = 0;
        }
        assert!(recover(&img).is_empty());

        // A real, non-Go ELF is likewise empty.
        let elf = crate::elf::tests::synthetic_elf64();
        let parsed = crate::model::ElfImage::parse(&elf).unwrap();
        assert!(recover(&parsed).is_empty());
        assert_eq!(render(&recover(&parsed)), "no Go strings recovered\n");
    }

    #[test]
    fn recovery_and_rendering_are_deterministic() {
        let mut text = Vec::new();
        text.extend(lea_rip(0, TEXT_VA, RODATA_VA));
        text.extend(mov_imm32(1, 5));
        text.extend(lea_rip(2, TEXT_VA + 12, RODATA_VA + 5));
        text.extend(mov_imm32(6, 5));
        let mut data = pair(RODATA_VA + 10, 6);
        data.extend(pair(RODATA_VA, 5));
        let img = go_image(Arch::X86_64, &text, b"helloworldagain!", &data);

        let a = recover(&img);
        let b = recover(&img);
        assert_eq!(a, b);
        assert_eq!(render(&a), render(&b));
        assert!(!a.is_empty());
    }

    #[test]
    fn render_is_readable_and_newline_terminated() {
        let strings = vec![
            GoString {
                va: 0x2000,
                len: 5,
                text: "he\"l\n".into(),
                source: Source::Code,
            },
            GoString {
                va: 0x2005,
                len: 3,
                text: "wo\u{1}".into(),
                source: Source::Data,
            },
        ];
        let out = render(&strings);
        assert!(out.ends_with('\n'));
        assert!(out.starts_with("Go strings: 2 (1 code-referenced, 1 data-referenced)\n"));
        assert!(out.contains("0x0000000000002000  code  \"he\\\"l\\n\""));
        assert!(out.contains("0x0000000000002005  data  \"wo\\x01\""));
        assert_eq!(render(&[]), "no Go strings recovered\n");
    }

    #[test]
    fn render_caps_the_preview_of_a_long_literal() {
        let long = "x".repeat(PREVIEW + 40);
        let out = render(&[GoString {
            va: 0x2000,
            len: long.len(),
            text: long,
            source: Source::Data,
        }]);
        assert!(out.contains(&format!("\"{}...\"", "x".repeat(PREVIEW))));
        assert!(out.lines().count() == 2);
    }

    // -- adversarial inputs -------------------------------------------------

    #[test]
    fn adversarial_bytes_in_code_and_data_stay_bounded_and_panic_free() {
        for seed in 1..8u64 {
            let text = pseudo_random(4096, seed.wrapping_mul(0x9E37_79B9));
            let rodata = pseudo_random(4096, seed.wrapping_mul(0xC2B2_AE35));
            let data = pseudo_random(4096, seed.wrapping_mul(0x27D4_EB2F));
            for arch in [Arch::X86_64, Arch::Aarch64, Arch::Other] {
                let img = go_image(arch, &text, &rodata, &data);
                let found = recover(&img);
                assert!(found.len() <= MAX_STRINGS);
                for s in &found {
                    assert!(s.len >= 1 && s.len <= MAX_STR_LEN);
                    assert_eq!(s.text.len(), s.len);
                    assert!(s.text.bytes().all(is_text_byte));
                }
                assert!(
                    found
                        .windows(2)
                        .all(|w| w[0].va + w[0].len as u64 <= w[1].va)
                );
                let _ = render(&found);
            }
        }
    }

    #[test]
    fn degenerate_images_do_not_panic() {
        let cases: Vec<FakeImage> = vec![
            // No regions at all.
            FakeImage {
                arch: Arch::X86_64,
                data: go_header(),
                regions: Vec::new(),
            },
            // A region at the top of the address space, one whose size
            // overflows past it, one mapping nowhere, and a zero-size one.
            FakeImage {
                arch: Arch::Aarch64,
                data: {
                    let mut v = go_header();
                    v.extend_from_slice(&[0x41; 64]);
                    v
                },
                regions: vec![
                    (".gopclntab".into(), 0x1000, 8, 0, ro()),
                    ("top".into(), u64::MAX - 4, 64, 8, rx()),
                    ("huge".into(), 0xFFFF_0000, u64::MAX, 8, ro()),
                    ("gap".into(), 0x5000, 16, usize::MAX, rw()),
                    ("zero".into(), 0x6000, 0, 8, ro()),
                ],
            },
            // Every region claiming the whole address space at once.
            FakeImage {
                arch: Arch::X86_64,
                data: {
                    let mut v = go_header();
                    v.extend(pseudo_random(512, 0xDEAD_BEEF));
                    v
                },
                regions: (0..32)
                    .map(|i| {
                        (
                            format!("r{i}"),
                            i as u64 * 3,
                            u64::MAX / 2,
                            i as usize * 7,
                            if i % 2 == 0 { rx() } else { rw() },
                        )
                    })
                    .chain(std::iter::once((
                        ".gopclntab".to_string(),
                        0,
                        8,
                        0,
                        ro(),
                    )))
                    .collect(),
            },
        ];
        for img in &cases {
            let found = recover(img);
            assert!(found.len() <= MAX_STRINGS);
            let _ = render(&found);
        }
    }

    #[test]
    fn truncating_the_buffer_never_panics() {
        let mut text = Vec::new();
        text.extend(lea_rip(0, TEXT_VA, RODATA_VA));
        text.extend(mov_imm32(1, 5));
        let full = go_image(
            Arch::X86_64,
            &text,
            b"helloworld\0\0\0\0\0\0",
            &pair(RODATA_VA, 10),
        );
        for len in 0..=full.data.len() {
            let img = FakeImage {
                arch: full.arch,
                data: full.data[..len].to_vec(),
                regions: full.regions.clone(),
            };
            let found = recover(&img);
            assert!(found.iter().all(|s| s.len <= MAX_STR_LEN));
        }
    }

    // -- unit-level rules ---------------------------------------------------

    #[test]
    fn text_bytes_admit_the_whitespace_go_sources_use() {
        assert!([b'\t', b'\n', b'\r', b' ', b'~', 0x80, 0xFF]
            .iter()
            .all(|&b| is_text_byte(b)));
        assert!([0x00u8, 0x01, 0x07, 0x0B, 0x1F, 0x7F]
            .iter()
            .all(|&b| !is_text_byte(b)));
    }

    #[test]
    fn source_labels_are_stable() {
        assert_eq!(Source::Code.label(), "code");
        assert_eq!(Source::Data.label(), "data");
        assert!(Source::Code.rank() < Source::Data.rank());
    }

    #[test]
    fn resolve_deduplicates_and_drops_contained_strings() {
        let s = |va: u64, len: usize, source| Cand { va, len, source };
        // Same (va, len) from both tiers collapses to the code label.
        let out = resolve(vec![s(0x10, 4, Source::Data), s(0x10, 4, Source::Code)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].source, Source::Code);

        // Contained-in-a-longer is dropped; a disjoint neighbor survives.
        let out = resolve(vec![
            s(0x10, 8, Source::Data),
            s(0x12, 2, Source::Data),
            s(0x18, 4, Source::Data),
        ]);
        assert_eq!(
            out.iter().map(|g| (g.va, g.len)).collect::<Vec<_>>(),
            [(0x10, 8), (0x18, 4)]
        );

        // A code-referenced start wins over a longer data-only overlap.
        let out = resolve(vec![s(0x10, 16, Source::Data), s(0x14, 2, Source::Code)]);
        assert_eq!(
            out.iter().map(|g| (g.va, g.len, g.source)).collect::<Vec<_>>(),
            [(0x14, 2, Source::Code)]
        );
    }
}
