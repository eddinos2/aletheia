//! Lightweight type vocabulary shared with [`crate::irtype`].
//!
//! Holds [`TypeId`]s and coarse [`TypeKind`]s so callee-side signatures
//! ([`crate::sig`]) can attach placeholder parameter types; [`crate::irtype`]
//! then refines those ids from SSA evidence. Minimal and total — no
//! constraint solver here.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::ir::Width;
use crate::sig::Signature;

/// Cap on distinct types allocated in one [`TypeTable`].
pub const MAX_TYPES: usize = 4096;

/// Opaque handle into a [`TypeTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeId(pub u32);

/// Coarse recovered / placeholder kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeKind {
    /// No evidence yet.
    Unknown,
    /// Integer of a known IR width; `signed` is optional until bounds land.
    Int {
        width: Width,
        signed: Option<bool>,
    },
    /// Pointer to another table entry (one level; pointee may be Unknown).
    Ptr {
        pointee: TypeId,
    },
}

impl TypeKind {
    fn token(&self) -> String {
        match self {
            TypeKind::Unknown => "unknown".into(),
            TypeKind::Int {
                width,
                signed: Some(true),
            } => format!("i{}", width.bits()),
            TypeKind::Int {
                width,
                signed: Some(false),
            } => format!("u{}", width.bits()),
            TypeKind::Int { width, signed: None } => format!("int{}", width.bits()),
            TypeKind::Ptr { pointee } => format!("ptr(t{})", pointee.0),
        }
    }
}

/// Growing table of [`TypeKind`]s. Deterministic ids (insertion order).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TypeTable {
    kinds: BTreeMap<TypeId, TypeKind>,
    next: u32,
    capped: bool,
}

impl TypeTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    pub fn capped(&self) -> bool {
        self.capped
    }

    pub fn get(&self, id: TypeId) -> Option<&TypeKind> {
        self.kinds.get(&id)
    }

    /// Insert `kind`, returning its id. On cap, returns the existing
    /// Unknown singleton if any, else allocates nothing and yields
    /// `TypeId(0)` with `capped` set — callers should prefer
    /// [`Self::intern_unknown`].
    pub fn intern(&mut self, kind: TypeKind) -> TypeId {
        if self.kinds.len() >= MAX_TYPES {
            self.capped = true;
            return self.intern_unknown();
        }
        let id = TypeId(self.next);
        self.next = self.next.saturating_add(1);
        self.kinds.insert(id, kind);
        id
    }

    pub fn intern_unknown(&mut self) -> TypeId {
        for (id, k) in &self.kinds {
            if matches!(k, TypeKind::Unknown) {
                return *id;
            }
        }
        if self.kinds.len() >= MAX_TYPES {
            self.capped = true;
            return TypeId(0);
        }
        self.intern(TypeKind::Unknown)
    }

    pub fn intern_int(&mut self, width: Width, signed: Option<bool>) -> TypeId {
        self.intern(TypeKind::Int { width, signed })
    }

    pub fn intern_ptr(&mut self, pointee: TypeId) -> TypeId {
        self.intern(TypeKind::Ptr { pointee })
    }
}

/// Parameter types attached from a [`Signature`] (placeholder Int widths).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamTypeMap {
    pub entry: u64,
    /// `(param index in sig.params, TypeId)`.
    pub params: Vec<(u16, TypeId)>,
    /// Optional primary return type when the signature claims a return.
    pub ret: Option<TypeId>,
}

impl ParamTypeMap {
    pub fn render(&self, table: &TypeTable) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "; types entry={:#x} params={} ret={}",
            self.entry,
            self.params.len(),
            if self.ret.is_some() { 1 } else { 0 }
        );
        for &(ix, id) in &self.params {
            let tok = table
                .get(id)
                .map(|k| display_token(k, table))
                .unwrap_or_else(|| "?".into());
            let _ = writeln!(out, "  param[{ix}] t{} {tok}", id.0);
        }
        if let Some(id) = self.ret {
            let tok = table
                .get(id)
                .map(|k| display_token(k, table))
                .unwrap_or_else(|| "?".into());
            let _ = writeln!(out, "  return t{} {tok}", id.0);
        }
        out
    }

    /// C-ish prototype for `pseudo` headers using table tokens
    /// (`u64 foo(i32 a, ptr b)`). Falls back to width-only ints.
    pub fn render_proto(&self, sig: &Signature, table: &TypeTable) -> String {
        let name = match sig.name.as_deref() {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => format!("sub_{:x}", sig.entry),
        };
        let ret = self
            .ret
            .and_then(|id| table.get(id))
            .map(|k| display_token(k, table))
            .unwrap_or_else(|| "int".into());
        let mut args = Vec::new();
        for (i, p) in sig.params.iter().enumerate() {
            let letter = if i < 26 {
                ((b'a' + i as u8) as char).to_string()
            } else {
                format!("a{}", i - 26)
            };
            let tok = self
                .params
                .iter()
                .find(|(ix, _)| *ix == i as u16)
                .and_then(|(_, id)| table.get(*id))
                .map(|k| display_token(k, table))
                .unwrap_or_else(|| format!("int{}", p.width.bits()));
            args.push(format!("{tok} {letter}"));
        }
        let arg_s = args.join(", ");
        if sig.stack_bytes == 0 {
            format!("{ret} {name}({arg_s})")
        } else if arg_s.is_empty() {
            format!("{ret} {name}(stack={})", sig.stack_bytes)
        } else {
            format!("{ret} {name}({arg_s}; stack={})", sig.stack_bytes)
        }
    }
}

/// Presentation token: collapse `ptr(tN)` with unknown pointee to `ptr`.
pub fn display_token(kind: &TypeKind, table: &TypeTable) -> String {
    match kind {
        TypeKind::Ptr { pointee } => match table.get(*pointee) {
            Some(TypeKind::Unknown) | None => "ptr".into(),
            Some(inner) => format!("ptr<{}>", display_token(inner, table)),
        },
        other => other.token(),
    }
}

/// Attach placeholder integer types from callee-side signature widths.
///
/// Each param becomes `Int { width, signed: None }` and the first return
/// cell (if any) the same. Call [`crate::irtype::attach_sig_with_evidence`]
/// to refine ids from SSA usage facts without replacing this entry point.
pub fn attach_sig_params(sig: &Signature, table: &mut TypeTable) -> ParamTypeMap {
    let mut params = Vec::new();
    for (i, p) in sig.params.iter().enumerate() {
        if params.len() >= crate::sig::MAX_PARAMS {
            break;
        }
        let id = table.intern_int(p.width, None);
        params.push((i as u16, id));
    }
    let ret = sig.returns.first().map(|r| table.intern_int(r.width, None));
    ParamTypeMap {
        entry: sig.entry,
        params,
        ret,
    }
}

/// Sanity check for a table + optional param map. Total.
pub fn check(table: &TypeTable, params: Option<&ParamTypeMap>) -> Result<(), String> {
    if table.kinds.len() > MAX_TYPES {
        return Err("types exceed MAX_TYPES".into());
    }
    for (id, kind) in &table.kinds {
        if id.0 >= table.next && table.next > 0 {
            // Allow id==0 after cap fallback.
            if !(table.capped && id.0 == 0) {
                return Err(format!("type id {} >= next {}", id.0, table.next));
            }
        }
        if let TypeKind::Ptr { pointee } = kind
            && table.get(*pointee).is_none()
            && !(table.capped && pointee.0 == 0)
        {
            return Err(format!("ptr t{} pointee missing", id.0));
        }
    }
    if let Some(p) = params {
        for &(_, id) in &p.params {
            if table.get(id).is_none() && !(table.capped && id.0 == 0) {
                return Err(format!("param type t{} missing", id.0));
            }
        }
        if let Some(id) = p.ret
            && table.get(id).is_none()
            && !(table.capped && id.0 == 0)
        {
            return Err(format!("return type t{} missing", id.0));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Width;
    use crate::irssa::{Name, SsaBlock, SsaFunction};
    use crate::model::Arch;
    use crate::sig;
    use crate::ir::{Expr, Reg, Space, Stmt};
    use std::collections::BTreeMap;

    fn name(cell: u16, version: u32, w: Width) -> Name {
        Name {
            space: Space::Arch,
            cell,
            version,
            width: w,
        }
    }

    fn two_int_args() -> SsaFunction {
        let names = vec![
            name(7, 0, Width::W64),
            name(6, 0, Width::W64),
            name(0, 1, Width::W64),
        ];
        let stmts = vec![Stmt::Assign {
            dst: Reg {
                space: Space::Arch,
                num: 2,
                width: Width::W64,
            },
            value: Expr::reg(Reg {
                space: Space::Arch,
                num: 0,
                width: Width::W64,
            }),
        }];
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0x1000,
            SsaBlock {
                start: 0x1000,
                end: 0x1010,
                phis: vec![],
                stmts,
                successors: vec![],
                truncated: false,
            },
        );
        SsaFunction {
            entry: 0x1000,
            name: Some("add".into()),
            arch: Arch::X86_64,
            blocks,
            skipped: vec![],
            names,
            live_in: vec![0, 1],
            partial: vec![],
        }
    }

    #[test]
    fn intern_int_and_ptr() {
        let mut t = TypeTable::new();
        let i = t.intern_int(Width::W32, Some(true));
        let u = t.intern_unknown();
        let p = t.intern_ptr(u);
        assert_eq!(t.get(i), Some(&TypeKind::Int {
            width: Width::W32,
            signed: Some(true),
        }));
        assert!(matches!(t.get(p), Some(TypeKind::Ptr { .. })));
        assert!(check(&t, None).is_ok());
    }

    #[test]
    fn attach_from_sig_params() {
        let f = two_int_args();
        let s = sig::recover(&f);
        let mut table = TypeTable::new();
        let map = attach_sig_params(&s, &mut table);
        assert!(check(&table, Some(&map)).is_ok());
        assert_eq!(map.params.len(), 2);
        assert!(map.ret.is_some());
        let dump = map.render(&table);
        assert!(dump.contains("param[0]"), "{dump}");
        assert!(dump.contains("int64"), "{dump}");
        assert_eq!(map.entry, s.entry);
        let proto = map.render_proto(&s, &table);
        assert!(proto.starts_with("int64 add("), "{proto}");
        assert!(proto.contains("int64 a"), "{proto}");
        assert!(proto.contains("int64 b"), "{proto}");
    }

    #[test]
    fn display_token_collapses_unknown_pointee() {
        let mut t = TypeTable::new();
        let u = t.intern_unknown();
        let p = t.intern_ptr(u);
        assert_eq!(display_token(t.get(p).unwrap(), &t), "ptr");
        let i = t.intern_int(Width::W32, Some(false));
        let p2 = t.intern_ptr(i);
        assert_eq!(display_token(t.get(p2).unwrap(), &t), "ptr<u32>");
    }
}
