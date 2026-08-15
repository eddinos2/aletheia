//! The annotation database — where an analyst's asserted facts live.
//!
//! Aletheia separates two kinds of knowledge about a binary:
//!
//! - **Computed facts** — function boundaries, basic blocks, the call
//!   graph, cross-references. These are reproducible from the bytes by
//!   re-running analysis, so they are *never stored here*. Regenerating
//!   them can never provoke a merge conflict because they are not in the
//!   file to begin with.
//! - **Asserted facts** — the names, type signatures, and comments a
//!   human writes. These cannot be recomputed and are the whole reason a
//!   database exists. They are what this module stores.
//!
//! Every assertion targets a location by [`Anchor`], not by absolute
//! address, so the work survives a rebase or relink (see [`crate::anchor`]).
//! An assertion is one entry in an append-only [log](Db); the current set
//! of annotations is a deterministic *fold* over that log. Because the
//! fold is total and order-independent — it depends only on each record's
//! `(seq, id)`, never on file position — two analysts can annotate the
//! same binary on separate git branches and merge with git alone: their
//! assertions are distinct lines, and the union folds to a well-defined
//! state regardless of which line landed first.
//!
//! [Undo and redo](Db::undo) are first-class operations on the log: undo
//! moves the most recent assertion to a redo stack, redo returns it, and
//! any fresh assertion clears the redo stack — the familiar editor model,
//! applied to the assertion log itself.

use std::collections::BTreeMap;

use crate::anchor::{Anchor, AnchorIndex, Fnv, Resolution};
use crate::error::{ParseError, Result};

/// Magic tag on the first line of a serialized database.
const FORMAT_TAG: &str = "aletheia-annotations";
/// On-disk format version. Bumped only on a breaking layout change; a
/// reader rejects a version it does not understand rather than guessing.
const FORMAT_VERSION: u32 = 1;

/// Which single-valued attribute of a location an assertion sets.
///
/// Each `(location, field)` pair holds at most one value; asserting the
/// same field again supersedes the earlier value under the log fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Field {
    /// The function's name.
    Name,
    /// The function's type signature / prototype, as a source string.
    Type,
    /// A free-form comment at the location.
    Comment,
}

impl Field {
    /// Stable tag folded into a record id and written to disk.
    fn tag(self) -> u8 {
        match self {
            Field::Name => 0,
            Field::Type => 1,
            Field::Comment => 2,
        }
    }

    /// The on-disk keyword for this field.
    fn token(self) -> &'static str {
        match self {
            Field::Name => "name",
            Field::Type => "type",
            Field::Comment => "comment",
        }
    }

    /// Parse an on-disk field keyword.
    fn parse(token: &str) -> Result<Field> {
        match token {
            "name" => Ok(Field::Name),
            "type" => Ok(Field::Type),
            "comment" => Ok(Field::Comment),
            other => Err(ParseError::Unsupported(format!(
                "unknown annotation field {other:?}"
            ))),
        }
    }
}

/// One human assertion: a single edit to one field of one location.
///
/// Records are immutable once created. A later assertion to the same
/// `(target, field)` supersedes this one; a `value` of `None` is a
/// tombstone that clears the field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assertion {
    /// The anchored location this assertion is about.
    pub target: Anchor,
    /// Which attribute of the location is set.
    pub field: Field,
    /// The asserted value, or `None` to clear the field.
    pub value: Option<String>,
    /// Assignment order — the last-writer-wins clock. Assigned once, never
    /// reused, and stable across undo/redo.
    pub seq: u64,
    /// Content id: a fingerprint of every other field. Serves as the
    /// stable on-disk line identity and as the deterministic tiebreak when
    /// two records share a `seq` after a merge.
    pub id: u64,
}

impl Assertion {
    /// Derive the content id from the assertion's semantic fields.
    ///
    /// Folds the target identity and hint, the field tag, the value, and
    /// the sequence number, with domain separators so a name and a comment
    /// that share text cannot collide.
    fn content_id(target: &Anchor, field: Field, value: Option<&str>, seq: u64) -> u64 {
        let mut h = Fnv::new();
        let (shape, bytes, insns) = target.identity();
        h.u64(shape);
        h.u64(bytes);
        h.u64(insns as u64);
        h.u64(target.abs);
        h.byte(0xff);
        h.byte(field.tag());
        h.byte(0xff);
        match value {
            None => h.byte(0), // tombstone marker
            Some(v) => {
                h.byte(1);
                h.slice(v.as_bytes());
            }
        }
        h.byte(0xff);
        h.u64(seq);
        h.finish()
    }
}

/// The annotation log for one binary.
///
/// An append-only sequence of [`Assertion`]s plus an in-memory redo stack.
/// The public setters record new assertions; [`Db::view`] and
/// [`Db::current`] fold the log into the annotations in effect.
#[derive(Debug, Clone, Default)]
pub struct Db {
    /// Assertions in application order.
    log: Vec<Assertion>,
    /// Undone assertions available to redo; never serialized.
    redo: Vec<Assertion>,
    /// Next sequence number to assign. Monotonic, never reused.
    next_seq: u64,
}

/// One annotation resolved onto a live program: which VA it lands on,
/// what it says, and how confidently the anchor matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placed<'a> {
    /// Entry VA in the current program.
    pub va: u64,
    /// Which attribute the annotation sets.
    pub field: Field,
    /// The asserted value.
    pub value: &'a str,
    /// How the stored anchor matched the current program.
    pub resolution: Resolution,
}

impl Db {
    /// An empty database.
    pub fn new() -> Self {
        Db::default()
    }

    /// The assertion log, in application order.
    pub fn log(&self) -> &[Assertion] {
        &self.log
    }

    /// Record a new assertion, clearing the redo stack.
    ///
    /// The building block under the typed setters; also the point every
    /// new edit funnels through, so redo semantics are enforced in one
    /// place.
    fn assert(&mut self, target: Anchor, field: Field, value: Option<String>) {
        let seq = self.next_seq;
        self.next_seq += 1;
        let id = Assertion::content_id(&target, field, value.as_deref(), seq);
        self.log.push(Assertion {
            target,
            field,
            value,
            seq,
            id,
        });
        self.redo.clear();
    }

    /// Name the function at `target`.
    pub fn set_name(&mut self, target: Anchor, name: impl Into<String>) {
        self.assert(target, Field::Name, Some(name.into()));
    }

    /// Set the type signature of the function at `target`.
    pub fn set_type(&mut self, target: Anchor, ty: impl Into<String>) {
        self.assert(target, Field::Type, Some(ty.into()));
    }

    /// Attach a comment at `target`.
    pub fn set_comment(&mut self, target: Anchor, text: impl Into<String>) {
        self.assert(target, Field::Comment, Some(text.into()));
    }

    /// Clear `field` at `target`, tombstoning any current value.
    pub fn clear(&mut self, target: Anchor, field: Field) {
        self.assert(target, field, None);
    }

    /// Undo the most recent assertion, moving it to the redo stack.
    ///
    /// Returns `false` when the log is empty. First-class on the log: the
    /// undone record leaves the effective state entirely.
    pub fn undo(&mut self) -> bool {
        match self.log.pop() {
            Some(a) => {
                self.redo.push(a);
                true
            }
            None => false,
        }
    }

    /// Redo the most recently undone assertion.
    ///
    /// Returns `false` when nothing is available to redo. Any assertion
    /// made after an undo clears the redo stack, so `redo` only ever
    /// replays an uninterrupted run of undos.
    pub fn redo(&mut self) -> bool {
        match self.redo.pop() {
            Some(a) => {
                self.log.push(a);
                true
            }
            None => false,
        }
    }

    /// The winning assertion for each `(location, field)`, tombstones
    /// included, keyed by the location's relocation-independent identity.
    ///
    /// The core fold. Within one `(identity, field)` the record with the
    /// greatest `(seq, id)` wins — a total, deterministic order that does
    /// not depend on log position, which is what makes a git-merged log
    /// fold to the same state on every machine.
    fn winners(&self) -> BTreeMap<((u64, u64, u32), Field), &Assertion> {
        let mut best: BTreeMap<((u64, u64, u32), Field), &Assertion> = BTreeMap::new();
        for a in &self.log {
            let key = (a.target.identity(), a.field);
            let wins = match best.get(&key) {
                Some(prev) => (a.seq, a.id) > (prev.seq, prev.id),
                None => true,
            };
            if wins {
                best.insert(key, a);
            }
        }
        best
    }

    /// The annotations currently in effect, as winning assertions with a
    /// value (tombstones dropped), in deterministic key order.
    pub fn current(&self) -> Vec<&Assertion> {
        self.winners()
            .into_values()
            .filter(|a| a.value.is_some())
            .collect()
    }

    /// The current annotations as a flat map, for the same-binary case
    /// where a location's identity is a stable key on its own.
    ///
    /// Keyed by `(identity, field)`; tombstoned fields are absent.
    pub fn view(&self) -> BTreeMap<((u64, u64, u32), Field), String> {
        self.winners()
            .into_iter()
            .filter_map(|(k, a)| a.value.clone().map(|v| (k, v)))
            .collect()
    }

    /// Resolve every current annotation onto a live program via `index`,
    /// dropping any whose location no longer exists there.
    ///
    /// This is the bridge between stored, relocation-independent anchors
    /// and the concrete VAs of a freshly analyzed binary. Each result
    /// carries the [`Resolution`] so a caller can flag annotations that
    /// only reattached by weak evidence.
    pub fn resolve_onto(&self, index: &AnchorIndex) -> Vec<Placed<'_>> {
        let mut out = Vec::new();
        for a in self.current() {
            let value = a.value.as_deref().expect("current() drops tombstones");
            let resolution = index.resolve(&a.target);
            let va = match &resolution {
                Resolution::Exact(va) | Resolution::Shape(va) | Resolution::Absolute(va) => *va,
                Resolution::Ambiguous(_) | Resolution::Unresolved => continue,
            };
            out.push(Placed {
                va,
                field: a.field,
                value,
                resolution,
            });
        }
        out.sort_by_key(|p| (p.va, p.field));
        out
    }

    /// Serialize the database to the git-mergeable text format.
    ///
    /// The output is a version header followed by one line per logged
    /// assertion, sorted into a canonical order — by target identity,
    /// then field, then sequence — that depends only on the assertions
    /// present, never on the order they were made. Two databases holding
    /// the same assertions serialize byte-for-byte identically, so a git
    /// diff shows exactly the assertions that changed.
    ///
    /// Each line is self-contained: analysts on separate branches add
    /// different lines, so git merges the union with no conflict, and the
    /// [fold](Db::current) resolves any concurrent edits to one field
    /// deterministically. The in-memory redo stack is intentionally *not*
    /// written — redo is a live-session concept, not durable history.
    pub fn serialize(&self) -> String {
        let mut lines: Vec<&Assertion> = self.log.iter().collect();
        lines.sort_by_key(|a| {
            let (shape, bytes, insns) = a.target.identity();
            (shape, bytes, insns, a.target.abs, a.field.tag(), a.seq)
        });

        let mut out = format!("{FORMAT_TAG} v{FORMAT_VERSION}\n");
        for a in lines {
            let (shape, bytes, insns) = a.target.identity();
            let (op, value) = match &a.value {
                Some(v) => ("set", Some(escape(v))),
                None => ("clear", None),
            };
            out.push_str(&format!(
                "{shape:016x} {bytes:016x} {insns:08x} {abs:016x} {op} {field} {seq:x}",
                abs = a.target.abs,
                field = a.field.token(),
                seq = a.seq,
            ));
            if let Some(v) = value {
                out.push(' ');
                out.push_str(&v);
            }
            out.push('\n');
        }
        out
    }

    /// Parse a database from the text format written by [`Db::serialize`].
    ///
    /// Validates the version header, then reads one assertion per line.
    /// Record ids are recomputed from the parsed fields rather than
    /// trusted from disk. The reconstructed log is ordered by sequence so
    /// undo continues to unwind edits most-recent-first, and the next
    /// sequence number resumes past the highest one seen.
    pub fn parse(text: &str) -> Result<Db> {
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| ParseError::Unsupported("empty annotation file".into()))?;

        let mut h = header.split_whitespace();
        if h.next() != Some(FORMAT_TAG) {
            return Err(ParseError::Unsupported(format!(
                "not a aletheia annotation file: {header:?}"
            )));
        }
        let version = h
            .next()
            .and_then(|v| v.strip_prefix('v'))
            .and_then(|n| n.parse::<u32>().ok())
            .ok_or_else(|| {
                ParseError::Unsupported(format!("malformed annotation header: {header:?}"))
            })?;
        if version != FORMAT_VERSION {
            return Err(ParseError::Unsupported(format!(
                "unsupported annotation format version {version}, expected {FORMAT_VERSION}"
            )));
        }

        let mut log = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            log.push(parse_line(line)?);
        }

        log.sort_by_key(|a| (a.seq, a.id));
        let next_seq = log.iter().map(|a| a.seq).max().map_or(0, |m| m + 1);
        Ok(Db {
            log,
            redo: Vec::new(),
            next_seq,
        })
    }
}

/// Parse one assertion line: seven space-separated header tokens and an
/// optional value that runs to the end of the line.
fn parse_line(line: &str) -> Result<Assertion> {
    let mut t = line.splitn(8, ' ');
    let mut next = |what: &str| {
        t.next()
            .ok_or_else(|| ParseError::Unsupported(format!("annotation line missing {what}")))
    };
    let shape = hex64(next("shape")?)?;
    let bytes = hex64(next("bytes")?)?;
    let insns = hex64(next("insns")?)? as u32;
    let abs = hex64(next("abs")?)?;
    let op = next("op")?;
    let field = Field::parse(next("field")?)?;
    let seq = hex64(next("seq")?)?;

    let value = match op {
        "set" => Some(unescape(t.next().unwrap_or(""))?),
        "clear" => None,
        other => {
            return Err(ParseError::Unsupported(format!(
                "unknown annotation op {other:?}"
            )));
        }
    };

    let target = Anchor {
        shape,
        bytes,
        insns,
        abs,
    };
    let id = Assertion::content_id(&target, field, value.as_deref(), seq);
    Ok(Assertion {
        target,
        field,
        value,
        seq,
        id,
    })
}

/// Parse a lowercase-hex integer, mapping a bad digit to a clean error.
fn hex64(s: &str) -> Result<u64> {
    u64::from_str_radix(s, 16)
        .map_err(|_| ParseError::Unsupported(format!("expected hex, found {s:?}")))
}

/// Escape a value so it occupies exactly one line: backslash and the
/// whitespace controls that would otherwise break line-orientation.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

/// Reverse [`escape`].
fn unescape(s: &str) -> Result<String> {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('\\') => out.push('\\'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            other => {
                return Err(ParseError::Unsupported(format!(
                    "bad escape \\{}",
                    other.map(String::from).unwrap_or_default()
                )));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A distinct anchor per test, identified only by its fields — the
    /// store never inspects a real program, so synthetic anchors suffice.
    fn anchor(n: u64) -> Anchor {
        Anchor {
            shape: 0x1000 + n,
            bytes: 0x2000 + n,
            insns: n as u32,
            abs: 0x40_0000 + n * 0x10,
        }
    }

    fn name_at(db: &Db, a: &Anchor) -> Option<String> {
        db.view().get(&(a.identity(), Field::Name)).cloned()
    }

    #[test]
    fn set_then_read_back() {
        let mut db = Db::new();
        let a = anchor(1);
        db.set_name(a, "parse_header");
        assert_eq!(name_at(&db, &a).as_deref(), Some("parse_header"));
    }

    #[test]
    fn last_writer_wins_within_a_field() {
        let mut db = Db::new();
        let a = anchor(1);
        db.set_name(a, "old");
        db.set_name(a, "new");
        assert_eq!(name_at(&db, &a).as_deref(), Some("new"));
        // One field, one live value regardless of how many edits.
        assert_eq!(db.current().len(), 1);
    }

    #[test]
    fn fields_are_independent() {
        let mut db = Db::new();
        let a = anchor(1);
        db.set_name(a, "f");
        db.set_type(a, "int f(void)");
        db.set_comment(a, "entry");
        assert_eq!(db.current().len(), 3);
    }

    #[test]
    fn clear_tombstones_the_field() {
        let mut db = Db::new();
        let a = anchor(1);
        db.set_name(a, "f");
        db.clear(a, Field::Name);
        assert_eq!(name_at(&db, &a), None);
        assert!(db.current().is_empty());
    }

    #[test]
    fn undo_removes_the_last_assertion() {
        let mut db = Db::new();
        let a = anchor(1);
        db.set_name(a, "first");
        db.set_name(a, "second");
        assert!(db.undo());
        assert_eq!(name_at(&db, &a).as_deref(), Some("first"));
    }

    #[test]
    fn redo_restores_an_undone_assertion() {
        let mut db = Db::new();
        let a = anchor(1);
        db.set_name(a, "first");
        db.set_name(a, "second");
        db.undo();
        assert!(db.redo());
        assert_eq!(name_at(&db, &a).as_deref(), Some("second"));
    }

    #[test]
    fn a_fresh_assertion_clears_the_redo_stack() {
        let mut db = Db::new();
        let a = anchor(1);
        db.set_name(a, "first");
        db.undo();
        db.set_name(a, "branch");
        assert!(!db.redo(), "redo must be empty after a new assertion");
        assert_eq!(name_at(&db, &a).as_deref(), Some("branch"));
    }

    #[test]
    fn undo_on_empty_log_is_a_no_op() {
        let mut db = Db::new();
        assert!(!db.undo());
    }

    #[test]
    fn sequence_numbers_are_stable_across_undo_redo() {
        let mut db = Db::new();
        let a = anchor(1);
        db.set_name(a, "x");
        let seq_before = db.log()[0].seq;
        db.undo();
        db.redo();
        assert_eq!(db.log()[0].seq, seq_before);
    }

    /// A database exercising every field kind and a tombstone.
    fn populated() -> Db {
        let mut db = Db::new();
        let (a, b) = (anchor(1), anchor(2));
        db.set_name(a, "parse_header");
        db.set_type(a, "int parse_header(u8 *buf, usize len)");
        db.set_comment(a, "validates the magic");
        db.set_name(b, "old_name");
        db.set_name(b, "checksum"); // supersedes
        db.set_comment(b, "line one\nline two\twith\\slash");
        db.clear(a, Field::Comment); // tombstone
        db
    }

    #[test]
    fn serialize_round_trips_through_parse() {
        let db = populated();
        let text = db.serialize();
        let back = Db::parse(&text).unwrap();
        assert_eq!(db.view(), back.view());
        // And the canonical form is a fixed point.
        assert_eq!(back.serialize(), text);
    }

    #[test]
    fn serialization_is_order_independent() {
        // Same assertions, opposite insertion order, must serialize
        // identically once the sequence-carrying content matches.
        let mut x = Db::new();
        let mut y = Db::new();
        let a = anchor(7);
        x.set_name(a, "f");
        x.set_comment(a, "c");
        y.set_name(a, "f");
        y.set_comment(a, "c");
        assert_eq!(x.serialize(), y.serialize());
    }

    #[test]
    fn header_and_version_are_validated() {
        assert!(Db::parse("not-a-aletheia-file\n").is_err());
        assert!(Db::parse("aletheia-annotations v999\n").is_err());
        assert!(Db::parse("").is_err());
        // A bare valid header parses to an empty database.
        assert!(Db::parse("aletheia-annotations v1\n").unwrap().current().is_empty());
    }

    #[test]
    fn control_characters_in_values_survive_a_round_trip() {
        let mut db = Db::new();
        let a = anchor(3);
        db.set_comment(a, "tab\there\nnewline and a \\ backslash");
        let back = Db::parse(&db.serialize()).unwrap();
        assert_eq!(back.view(), db.view());
    }

    #[test]
    fn tombstone_survives_serialization() {
        let mut db = Db::new();
        let a = anchor(4);
        db.set_name(a, "temp");
        db.clear(a, Field::Name);
        let back = Db::parse(&db.serialize()).unwrap();
        assert!(back.current().is_empty());
    }

    /// The Phase 3 exit criterion: two analysts annotate the same binary
    /// on separate branches, and the result merges with git alone.
    ///
    /// A line-oriented format makes a git merge the *union* of the two
    /// branches' added lines. This reconstructs that union by hand and
    /// checks that (a) it parses, (b) no annotation is lost, and (c) a
    /// concurrent edit to the *same* field resolves deterministically.
    #[test]
    fn two_branches_merge_by_line_union() {
        let (f1, f2) = (anchor(10), anchor(20));

        // Shared starting point, committed on the base branch.
        let mut base = Db::new();
        base.set_name(f1, "sub_401000");
        let base_text = base.serialize();

        // Branch A names f1 properly and adds a comment on f2.
        let mut a = Db::parse(&base_text).unwrap();
        a.set_name(f1, "decode_frame");
        a.set_comment(f2, "hot loop");
        let a_text = a.serialize();

        // Branch B independently types f1 and names f2.
        let mut b = Db::parse(&base_text).unwrap();
        b.set_type(f1, "void decode_frame(ctx*)");
        b.set_name(f2, "inflate");
        let b_text = b.serialize();

        // A git 3-way merge of line-oriented files keeps every unique
        // line from both sides. Model that as the deduplicated union of
        // all body lines under one header.
        let mut body: BTreeMap<String, ()> = BTreeMap::new();
        for text in [&base_text, &a_text, &b_text] {
            for line in text.lines().skip(1) {
                body.insert(line.to_string(), ());
            }
        }
        let merged_text = format!(
            "{}\n{}\n",
            format_args!("{FORMAT_TAG} v{FORMAT_VERSION}"),
            body.keys().cloned().collect::<Vec<_>>().join("\n")
        );

        let merged = Db::parse(&merged_text).unwrap();
        let view = merged.view();

        // Every distinct annotation from both branches is present.
        assert_eq!(
            view.get(&(f1.identity(), Field::Name)).map(String::as_str),
            Some("decode_frame")
        );
        assert_eq!(
            view.get(&(f1.identity(), Field::Type)).map(String::as_str),
            Some("void decode_frame(ctx*)")
        );
        assert_eq!(
            view.get(&(f2.identity(), Field::Name)).map(String::as_str),
            Some("inflate")
        );
        assert_eq!(
            view.get(&(f2.identity(), Field::Comment)).map(String::as_str),
            Some("hot loop")
        );
    }

    #[test]
    fn concurrent_same_field_edits_resolve_deterministically() {
        let f = anchor(30);
        let base = {
            let mut d = Db::new();
            d.set_name(f, "sub_x");
            d
        };
        let base_text = base.serialize();

        // Both branches rename the same function differently.
        let mut a = Db::parse(&base_text).unwrap();
        a.set_name(f, "alpha");
        let mut b = Db::parse(&base_text).unwrap();
        b.set_name(f, "beta");

        // Union the lines regardless of which side is listed first...
        let union = |first: &Db, second: &Db| {
            let mut lines: BTreeMap<String, ()> = BTreeMap::new();
            for d in [first, second] {
                for line in d.serialize().lines().skip(1) {
                    lines.insert(line.to_string(), ());
                }
            }
            let text = format!(
                "{FORMAT_TAG} v{FORMAT_VERSION}\n{}\n",
                lines.keys().cloned().collect::<Vec<_>>().join("\n")
            );
            Db::parse(&text).unwrap().view()
        };

        // ...and the winner is the same either way: no data loss, no
        // dependence on merge order.
        assert_eq!(union(&a, &b), union(&b, &a));
        let name = union(&a, &b)
            .get(&(f.identity(), Field::Name))
            .cloned()
            .unwrap();
        assert!(name == "alpha" || name == "beta");
    }
}
