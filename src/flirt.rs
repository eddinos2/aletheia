//! Open FLIRT-style function signature matching (Phase 5 stub).
//!
//! Matches the first *N* bytes of each function against a **documented text
//! corpus** — CRC-32 fingerprints or byte patterns with optional wildcards —
//! and yields [`Candidate`]s tagged [`sig::Provenance::SymbolDerived`] for
//! naming / signature recovery.
//!
//! # Corpus format (not proprietary IDA `.sig`)
//!
//! Lines are UTF-8 text. Blank lines and `#` comments are ignored.
//!
//! ```text
//! # IEEE CRC-32 (poly 0xEDB88320) of the first PREFIX_LEN bytes, hex, TAB, name
//! a1b2c3d4    memcpy
//!
//! # Byte pattern: whitespace-separated hex bytes; ?? is a wildcard
//! 55 48 89 e5 ?? ??    my_frame_setup
//! ```
//!
//! CRC lines use a fixed prefix length ([`DEFAULT_PREFIX_LEN`], overridable
//! on the [`Corpus`]). Pattern lines match their own length (capped at
//! [`MAX_PATTERN_LEN`]). This is an *open* interchange format inspired by
//! the *idea* of library identification; it does **not** parse Hex-Rays /
//! IDA binary `.sig` / `.pat` files.
//!
//! # Contract
//!
//! - Total: never panics; every read is bounds-checked.
//! - Caps truncate corpus entries and match output rather than grow unbounded.
//! - Deterministic: matches sorted by `(va, name)`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::funcs;
use crate::model::Image;
use crate::sig::{Provenance, Signature};

/// Bytes hashed for CRC-32 corpus lines when the corpus does not override.
pub const DEFAULT_PREFIX_LEN: usize = 32;

/// Cap on entries retained from one corpus file.
pub const MAX_CORPUS_ENTRIES: usize = 65_536;

/// Cap on match candidates returned for one image.
pub const MAX_MATCHES: usize = 65_536;

/// Cap on candidates kept per function VA (CRC collisions / multi-pattern).
pub const MAX_PER_FUNCTION: usize = 8;

/// Cap on a recovered symbol name length.
pub const MAX_NAME_LEN: usize = 512;

/// Cap on the length of one byte-pattern entry.
pub const MAX_PATTERN_LEN: usize = 256;

/// Absolute upper bound on prefix length for CRC hashing.
pub const MAX_PREFIX_LEN: usize = 256;

/// How a candidate was produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchKind {
    /// CRC-32 of the first `prefix_len` bytes equaled a corpus fingerprint.
    Crc32 { crc: u32, prefix_len: usize },
    /// Leading bytes matched a corpus pattern (wildcards allowed).
    Pattern { len: usize },
}

/// One naming candidate for [`crate::sig`] / listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// Function entry VA that matched.
    pub va: u64,
    /// Corpus symbol name.
    pub name: String,
    /// Always [`Provenance::SymbolDerived`] for FLIRT hits.
    pub provenance: Provenance,
    pub kind: MatchKind,
}

/// Loaded open-format signature corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Corpus {
    /// CRC-32 → names (insertion order preserved per CRC via Vec).
    crc: BTreeMap<u32, Vec<String>>,
    /// Byte patterns: `None` = wildcard.
    patterns: Vec<(Vec<Option<u8>>, String)>,
    /// Prefix length used when hashing for CRC lines.
    pub prefix_len: usize,
    /// True when [`MAX_CORPUS_ENTRIES`] truncated the load.
    pub capped: bool,
    /// Number of entries retained (CRC names + patterns).
    pub entry_count: usize,
}

impl Corpus {
    /// Empty corpus (useful for `--flirt` with no path).
    pub fn empty() -> Self {
        Self {
            crc: BTreeMap::new(),
            patterns: Vec::new(),
            prefix_len: DEFAULT_PREFIX_LEN,
            capped: false,
            entry_count: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }
}

/// Result of matching a corpus against an image's functions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchReport {
    pub candidates: Vec<Candidate>,
    /// True when [`MAX_MATCHES`] truncated output.
    pub capped: bool,
    /// True when the corpus had no entries (honest empty dump).
    pub corpus_empty: bool,
}

impl MatchReport {
    pub fn render(&self) -> String {
        render(self)
    }
}

/// Parse an open text corpus. Total: malformed lines are skipped with no
/// panic; only structural overflow sets [`Corpus::capped`].
pub fn parse_corpus(text: &str) -> Corpus {
    parse_corpus_with(text, DEFAULT_PREFIX_LEN)
}

/// Like [`parse_corpus`], with an explicit CRC prefix length (clamped to
/// `1..=MAX_PREFIX_LEN`).
pub fn parse_corpus_with(text: &str, prefix_len: usize) -> Corpus {
    let prefix_len = prefix_len.clamp(1, MAX_PREFIX_LEN);
    let mut corpus = Corpus {
        crc: BTreeMap::new(),
        patterns: Vec::new(),
        prefix_len,
        capped: false,
        entry_count: 0,
    };

    for raw in text.lines() {
        if corpus.entry_count >= MAX_CORPUS_ENTRIES {
            corpus.capped = true;
            break;
        }
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((left, name)) = split_tab_or_ws(line) else {
            continue;
        };
        let name = sanitize_name(name);
        if name.is_empty() {
            continue;
        }

        if let Some(crc) = parse_crc32_token(left) {
            corpus.crc.entry(crc).or_default().push(name);
            corpus.entry_count += 1;
            continue;
        }
        if let Some(pat) = parse_byte_pattern(left) {
            corpus.patterns.push((pat, name));
            corpus.entry_count += 1;
        }
    }
    corpus
}

/// Load a corpus file from disk.
pub fn load_corpus(path: &Path) -> Result<Corpus, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(parse_corpus(&text))
}

/// Discover functions on `image` and match them against `corpus`.
pub fn match_image(image: &dyn Image, corpus: &Corpus) -> MatchReport {
    let fns = funcs::discover(image);
    match_entries(image, fns.iter().map(|f| f.va), corpus)
}

/// Match explicit function entry VAs against `corpus`.
pub fn match_entries<I>(image: &dyn Image, entries: I, corpus: &Corpus) -> MatchReport
where
    I: IntoIterator<Item = u64>,
{
    let mut report = MatchReport {
        candidates: Vec::new(),
        capped: false,
        corpus_empty: corpus.is_empty(),
    };
    if corpus.is_empty() {
        return report;
    }

    let mut per_va: BTreeMap<u64, usize> = BTreeMap::new();
    let mut entries: Vec<u64> = entries.into_iter().collect();
    entries.sort_unstable();
    entries.dedup();

    for va in entries {
        if report.candidates.len() >= MAX_MATCHES {
            report.capped = true;
            break;
        }
        let Some(bytes) = bytes_at(image, va, corpus.prefix_len.max(MAX_PATTERN_LEN)) else {
            continue;
        };

        // CRC path: need a full prefix.
        if bytes.len() >= corpus.prefix_len {
            let crc = crc32_ieee(&bytes[..corpus.prefix_len]);
            if let Some(names) = corpus.crc.get(&crc) {
                for name in names {
                    if !push_candidate(
                        &mut report,
                        &mut per_va,
                        va,
                        name.clone(),
                        MatchKind::Crc32 {
                            crc,
                            prefix_len: corpus.prefix_len,
                        },
                    ) {
                        return report;
                    }
                }
            }
        }

        for (pat, name) in &corpus.patterns {
            if pat.len() > bytes.len() {
                continue;
            }
            if pattern_matches(pat, &bytes[..pat.len()])
                && !push_candidate(
                    &mut report,
                    &mut per_va,
                    va,
                    name.clone(),
                    MatchKind::Pattern { len: pat.len() },
                )
            {
                return report;
            }
        }
    }

    report
        .candidates
        .sort_by(|a, b| (a.va, &a.name).cmp(&(b.va, &b.name)));
    report
}

/// Apply the first [`Candidate`] for `sig.entry` as [`Signature::name`] when
/// the signature has no name yet. Leaves param provenance to
/// [`sig::recover`] (which reads the name for demangle arity).
pub fn apply_to_signature(sig: &mut Signature, candidates: &[Candidate]) {
    if sig.name.is_some() {
        return;
    }
    for c in candidates {
        if c.va == sig.entry && c.provenance == Provenance::SymbolDerived {
            sig.name = Some(c.name.clone());
            return;
        }
    }
}

/// First SymbolDerived name per VA (deterministic: lowest name wins after sort).
pub fn name_map(candidates: &[Candidate]) -> BTreeMap<u64, String> {
    let mut map = BTreeMap::new();
    for c in candidates {
        if c.provenance != Provenance::SymbolDerived {
            continue;
        }
        map.entry(c.va).or_insert_with(|| c.name.clone());
    }
    map
}

/// Structural check for a report (tests / redump honesty).
pub fn check(report: &MatchReport) -> Result<(), String> {
    if report.corpus_empty && !report.candidates.is_empty() {
        return Err("empty corpus produced candidates".into());
    }
    if report.candidates.len() > MAX_MATCHES {
        return Err(format!(
            "match count {} exceeds MAX_MATCHES {}",
            report.candidates.len(),
            MAX_MATCHES
        ));
    }
    let mut prev: Option<(&u64, &str)> = None;
    let mut per_va: BTreeMap<u64, usize> = BTreeMap::new();
    for c in &report.candidates {
        if c.provenance != Provenance::SymbolDerived {
            return Err(format!(
                "candidate at {:#x} provenance is {:?}, expected SymbolDerived",
                c.va, c.provenance
            ));
        }
        if c.name.is_empty() || c.name.len() > MAX_NAME_LEN {
            return Err(format!("bad name length for {:#x}", c.va));
        }
        if let Some((pva, pname)) = prev
            && (pva, pname) > (&c.va, c.name.as_str())
        {
            return Err("candidates not sorted by (va, name)".into());
        }
        prev = Some((&c.va, c.name.as_str()));
        let n = per_va.entry(c.va).or_insert(0);
        *n += 1;
        if *n > MAX_PER_FUNCTION {
            return Err(format!(
                "more than MAX_PER_FUNCTION ({MAX_PER_FUNCTION}) at {:#x}",
                c.va
            ));
        }
        match c.kind {
            MatchKind::Crc32 { prefix_len, .. } => {
                if prefix_len == 0 || prefix_len > MAX_PREFIX_LEN {
                    return Err("invalid CRC prefix_len".into());
                }
            }
            MatchKind::Pattern { len } => {
                if len == 0 || len > MAX_PATTERN_LEN {
                    return Err("invalid pattern len".into());
                }
            }
        }
    }
    Ok(())
}

/// Check a loaded corpus.
pub fn check_corpus(corpus: &Corpus) -> Result<(), String> {
    if corpus.prefix_len == 0 || corpus.prefix_len > MAX_PREFIX_LEN {
        return Err("invalid corpus prefix_len".into());
    }
    if corpus.entry_count > MAX_CORPUS_ENTRIES {
        return Err("corpus entry_count exceeds cap".into());
    }
    let mut counted = 0usize;
    for names in corpus.crc.values() {
        for n in names {
            if n.is_empty() || n.len() > MAX_NAME_LEN {
                return Err("bad CRC name".into());
            }
            counted += 1;
        }
    }
    for (pat, name) in &corpus.patterns {
        if pat.is_empty() || pat.len() > MAX_PATTERN_LEN {
            return Err("bad pattern length".into());
        }
        if name.is_empty() || name.len() > MAX_NAME_LEN {
            return Err("bad pattern name".into());
        }
        counted += 1;
    }
    if counted != corpus.entry_count {
        return Err(format!(
            "entry_count {} != summed entries {counted}",
            corpus.entry_count
        ));
    }
    Ok(())
}

/// Deterministic multi-line dump for CLI / tests.
pub fn render(report: &MatchReport) -> String {
    let mut out = String::new();
    if report.corpus_empty {
        let _ = writeln!(
            out,
            "; flirt: empty corpus (open text: CRC32\\tNAME or hex-pattern\\tNAME; not IDA .sig)"
        );
        return out;
    }
    let _ = writeln!(
        out,
        "; flirt matches={} capped={}",
        report.candidates.len(),
        report.capped
    );
    for c in &report.candidates {
        let how = match c.kind {
            MatchKind::Crc32 { crc, prefix_len } => {
                format!("crc32={crc:08x}/n={prefix_len}")
            }
            MatchKind::Pattern { len } => format!("pattern/n={len}"),
        };
        let _ = writeln!(
            out,
            "{:#x}\t{}\t{}\t[{}]",
            c.va,
            c.name,
            how,
            provenance_token(c.provenance)
        );
    }
    out
}

fn provenance_token(p: Provenance) -> &'static str {
    // Keep dump token aligned with sig::Provenance::token (private there).
    match p {
        Provenance::SymbolDerived => "symbol",
        Provenance::DataflowProven => "dataflow",
        Provenance::AbiAssumed => "abi",
        Provenance::Heuristic => "heuristic",
    }
}

fn push_candidate(
    report: &mut MatchReport,
    per_va: &mut BTreeMap<u64, usize>,
    va: u64,
    name: String,
    kind: MatchKind,
) -> bool {
    if report.candidates.len() >= MAX_MATCHES {
        report.capped = true;
        return false;
    }
    let n = per_va.entry(va).or_insert(0);
    if *n >= MAX_PER_FUNCTION {
        return true; // skip this VA's extras, keep scanning others
    }
    // Dedup identical (va, name).
    if report
        .candidates
        .iter()
        .any(|c| c.va == va && c.name == name)
    {
        return true;
    }
    *n += 1;
    report.candidates.push(Candidate {
        va,
        name,
        provenance: Provenance::SymbolDerived,
        kind,
    });
    true
}

fn bytes_at(image: &dyn Image, va: u64, max: usize) -> Option<Vec<u8>> {
    let off = image.va_to_offset(va)?;
    let buf = image.bytes();
    if off >= buf.len() {
        return None;
    }
    let avail = (buf.len() - off).min(max);
    if avail == 0 {
        return None;
    }
    // Prefer executable region when present; still allow any mapped bytes.
    Some(buf[off..off + avail].to_vec())
}

fn pattern_matches(pat: &[Option<u8>], bytes: &[u8]) -> bool {
    if pat.len() != bytes.len() {
        return false;
    }
    pat.iter()
        .zip(bytes.iter())
        .all(|(p, b)| p.is_none_or(|want| want == *b))
}

/// IEEE CRC-32 (ISO 3309 / ITU-T V.42 / zlib), public polynomial.
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xffff_ffff;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let lsb = crc & 1;
            crc >>= 1;
            if lsb != 0 {
                crc ^= 0xedb8_8320;
            }
        }
    }
    !crc
}

fn sanitize_name(s: &str) -> String {
    let t = s.trim();
    if t.len() > MAX_NAME_LEN {
        t[..MAX_NAME_LEN].to_string()
    } else {
        t.to_string()
    }
}

/// Split on the first TAB, else on the last whitespace run before the name
/// when the left side looks like a CRC or pattern (conservative: require TAB
/// for CRC/pattern ↔ name separation).
fn split_tab_or_ws(line: &str) -> Option<(&str, &str)> {
    if let Some((a, b)) = line.split_once('\t') {
        let a = a.trim();
        let b = b.trim();
        if a.is_empty() || b.is_empty() {
            return None;
        }
        return Some((a, b));
    }
    // Fallback: last space-separated token is the name when left is hex-ish.
    let mut parts = line.rsplitn(2, char::is_whitespace);
    let name = parts.next()?.trim();
    let left = parts.next()?.trim();
    if name.is_empty() || left.is_empty() {
        return None;
    }
    Some((left, name))
}

fn parse_crc32_token(s: &str) -> Option<u32> {
    let s = s.trim();
    // Bare 8-hex-digit CRC (optionally 0x-prefixed). No spaces → not a pattern.
    if s.contains(char::is_whitespace) {
        return None;
    }
    let hex = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    if hex.is_empty() || hex.len() > 8 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    // Ambiguous with a single-byte pattern like `55` — require ≥2 hex digits
    // and treat short tokens as patterns instead when they fail CRC shape.
    // Convention: CRC lines are exactly 8 hex digits (zero-padded) or 0x + up to 8.
    if hex.len() != 8 && !s.starts_with("0x") && !s.starts_with("0X") {
        return None;
    }
    u32::from_str_radix(hex, 16).ok()
}

fn parse_byte_pattern(s: &str) -> Option<Vec<Option<u8>>> {
    let mut out = Vec::new();
    for tok in s.split_whitespace() {
        if out.len() >= MAX_PATTERN_LEN {
            return None;
        }
        if tok == "??" || tok == "?" {
            out.push(None);
            continue;
        }
        let hex = tok.strip_prefix("0x").or_else(|| tok.strip_prefix("0X")).unwrap_or(tok);
        if hex.len() != 2 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        out.push(Some(u8::from_str_radix(hex, 16).ok()?));
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Arch, ImportSlot, Perms, Region, Symbol};
    use crate::sig::Signature;

    struct FakeImage {
        arch: Arch,
        bytes: Vec<u8>,
        regions: Vec<(u64, u64, usize)>, // va, size, file_off
        entries: Vec<u64>,
        hints: Vec<u64>,
    }

    impl Image for FakeImage {
        fn arch(&self) -> Arch {
            self.arch
        }
        fn entry_points(&self) -> Vec<u64> {
            self.entries.clone()
        }
        fn regions(&self) -> Vec<Region> {
            self.regions
                .iter()
                .enumerate()
                .map(|(i, &(va, size, _))| Region {
                    name: format!("r{i}"),
                    va,
                    size,
                    perms: Perms {
                        r: true,
                        w: false,
                        x: true,
                    },
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
            for &(rva, size, off) in &self.regions {
                if va >= rva && va - rva < size {
                    let o = off + (va - rva) as usize;
                    return (o < self.bytes.len()).then_some(o);
                }
            }
            None
        }
        fn bytes(&self) -> &[u8] {
            &self.bytes
        }
        fn function_starts_hint(&self) -> Vec<u64> {
            self.hints.clone()
        }
    }

    fn synth(prefix: &[u8]) -> FakeImage {
        let mut bytes = prefix.to_vec();
        // Pad so CRC prefix is available.
        while bytes.len() < DEFAULT_PREFIX_LEN {
            bytes.push(0x90);
        }
        bytes.extend_from_slice(&[0xc3]); // ret
        FakeImage {
            arch: Arch::X86_64,
            bytes,
            regions: vec![(0x1000, 0x100, 0)],
            entries: vec![0x1000],
            hints: vec![0x1000],
        }
    }

    #[test]
    fn crc32_ieee_known_vector() {
        // ISO 3309 / common "123456789" vector.
        assert_eq!(crc32_ieee(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn parse_crc_and_pattern_lines() {
        let text = "\
# comment
cbf43926\thello_crc
55 48 89 e5 ??\tprolog
";
        let c = parse_corpus(text);
        check_corpus(&c).unwrap();
        assert_eq!(c.entry_count, 2);
        assert!(c.crc.contains_key(&0xcbf4_3926));
        assert_eq!(c.patterns.len(), 1);
        assert_eq!(c.patterns[0].1, "prolog");
    }

    #[test]
    fn match_crc_yields_symbol_derived() {
        let payload = b"123456789";
        let mut bytes = payload.to_vec();
        while bytes.len() < DEFAULT_PREFIX_LEN {
            bytes.push(0);
        }
        let crc = crc32_ieee(&bytes[..DEFAULT_PREFIX_LEN]);
        let corpus = parse_corpus(&format!("{crc:08x}\tlib_foo"));
        let img = FakeImage {
            arch: Arch::X86_64,
            bytes: bytes.clone(),
            regions: vec![(0x1000, bytes.len() as u64, 0)],
            entries: vec![0x1000],
            hints: vec![0x1000],
        };
        let report = match_image(&img, &corpus);
        check(&report).unwrap();
        assert_eq!(report.candidates.len(), 1);
        let c = &report.candidates[0];
        assert_eq!(c.va, 0x1000);
        assert_eq!(c.name, "lib_foo");
        assert_eq!(c.provenance, Provenance::SymbolDerived);
        assert!(matches!(c.kind, MatchKind::Crc32 { .. }));
    }

    #[test]
    fn match_pattern_with_wildcard() {
        let img = synth(&[0x55, 0x48, 0x89, 0xe5, 0x31, 0xc0]);
        let corpus = parse_corpus("55 48 89 e5 ?? ??\tframe");
        let report = match_image(&img, &corpus);
        check(&report).unwrap();
        assert_eq!(report.candidates.len(), 1);
        assert_eq!(report.candidates[0].name, "frame");
        assert_eq!(
            report.candidates[0].provenance,
            Provenance::SymbolDerived
        );
    }

    #[test]
    fn empty_corpus_note_and_check() {
        let report = match_image(&synth(&[0x90]), &Corpus::empty());
        assert!(report.corpus_empty);
        assert!(report.candidates.is_empty());
        check(&report).unwrap();
        let text = report.render();
        assert!(text.contains("empty corpus"), "{text}");
        assert!(text.contains("not IDA"), "{text}");
    }

    #[test]
    fn apply_to_signature_sets_name() {
        let candidates = vec![Candidate {
            va: 0x1000,
            name: "memcpy".into(),
            provenance: Provenance::SymbolDerived,
            kind: MatchKind::Pattern { len: 4 },
        }];
        let mut sig = Signature {
            entry: 0x1000,
            name: None,
            arch: Arch::X86_64,
            params: Vec::new(),
            returns: Vec::new(),
            stack_bytes: 0,
            params_capped: false,
            returns_capped: false,
        };
        apply_to_signature(&mut sig, &candidates);
        assert_eq!(sig.name.as_deref(), Some("memcpy"));
        // Does not overwrite.
        sig.name = Some("keep".into());
        apply_to_signature(&mut sig, &candidates);
        assert_eq!(sig.name.as_deref(), Some("keep"));
    }

    #[test]
    fn corpus_cap_sets_flag() {
        let mut text = String::new();
        for i in 0..(MAX_CORPUS_ENTRIES + 3) {
            let _ = writeln!(text, "{:08x}\tname_{i}", i as u32);
        }
        let c = parse_corpus(&text);
        assert!(c.capped);
        assert_eq!(c.entry_count, MAX_CORPUS_ENTRIES);
        check_corpus(&c).unwrap();
    }

    #[test]
    fn sample_corpus_file_loads() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/flirt/sample.corpus");
        let corpus = load_corpus(&path).expect("sample corpus");
        assert!(!corpus.is_empty(), "sample corpus should have entries");
        assert!(corpus.entry_count >= 4);
        check_corpus(&corpus).unwrap();
    }
}
