//! Stable scripting / embedder API (Phase 4 — plugin ABI foundation).
//!
//! This is the **library-facing contract** for agents, GUIs, and future
//! language bindings. Versioned independently of crate semver so hosts can
//! refuse an incompatible engine:
//!
//! ```text
//! API_VERSION = "1.0.0"   // major.minor.patch — major bumps break hosts
//! ```
//!
//! Design rules (see `docs/PLUGIN_ABI.md`):
//!
//! 1. **Engine owns truth** — this module never invents CFG, xrefs, or types.
//! 2. **Total** — no panics on hostile bytes; errors are [`ApiError`].
//! 3. **Deterministic** — same bytes + same API_VERSION ⇒ same dumps.
//! 4. **Zero deps** — pure Rust; FFI / pyo3 wrappers live outside the core.
//!
//! The MCP/GUI protocol (`protocol/PROTOCOL.md`) is the *wire* shape of the
//! same contract. Prefer [`AnalysisSession`] in-process; prefer NDJSON over
//! stdio for out-of-process agents.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use crate::annotate;
use crate::anchor;
use crate::callfx;
use crate::cfg;
use crate::funcs;
use crate::irlift;
use crate::irout;
use crate::irssa;
use crate::irssaopt;
use crate::irstruct;
use crate::irstack;
use crate::irtype;
use crate::jumptable;
use crate::listing;
use crate::macho;
use crate::mempromote;
use crate::model::{Arch, Image};
use crate::patch::fnv1a64;
use crate::pseudo;
use crate::sig;
use crate::types;
use crate::xref;
use crate::load;

/// Scripting ABI version. Hosts should gate on the major component.
pub const API_VERSION: &str = "1.0.0";

/// Engine package version (crate).
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Cap on functions returned by [`AnalysisSession::functions`].
pub const MAX_FUNCTIONS: usize = 65_536;

/// Why a scripting call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiError {
    Io(String),
    Load(String),
    Unsupported(String),
    CapExceeded {
        what: &'static str,
        value: usize,
        cap: usize,
    },
    NotFound(String),
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::Io(e) => write!(f, "io: {e}"),
            ApiError::Load(e) => write!(f, "load: {e}"),
            ApiError::Unsupported(e) => write!(f, "unsupported: {e}"),
            ApiError::CapExceeded { what, value, cap } => {
                write!(f, "{what} {value} exceeds cap {cap}")
            }
            ApiError::NotFound(e) => write!(f, "not found: {e}"),
        }
    }
}

impl std::error::Error for ApiError {}

/// One recovered function summary for scripting hosts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionInfo {
    pub va: u64,
    pub name: Option<String>,
    /// Provenance token: `cfg`, `asserted`, `symbol`, …
    pub source: String,
}

/// Open binary analysis session — the primary scripting object.
pub struct AnalysisSession {
    label: String,
    data: Vec<u8>,
    hash: u64,
    encrypted: bool,
    db: annotate::Db,
}

impl AnalysisSession {
    /// Open from a filesystem path.
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self, ApiError> {
        let path = path.as_ref();
        let data = fs::read(path).map_err(|e| ApiError::Io(format!("{}: {e}", path.display())))?;
        Self::open_bytes(data, path.display().to_string())
    }

    /// Open from an owned byte buffer with a diagnostic label.
    pub fn open_bytes(data: Vec<u8>, label: impl Into<String>) -> Result<Self, ApiError> {
        let label = label.into();
        {
            let _image = load(&data).map_err(|e| ApiError::Load(format!("{label}: {e}")))?;
        }
        let encrypted = macho::MachFile::parse(&data)
            .map(|m| m.is_encrypted())
            .unwrap_or(false);
        let hash = fnv1a64(&data);
        Ok(Self {
            label,
            data,
            hash,
            encrypted,
            db: annotate::Db::new(),
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn encrypted(&self) -> bool {
        self.encrypted
    }

    pub fn api_version() -> &'static str {
        API_VERSION
    }

    pub fn engine_version() -> &'static str {
        ENGINE_VERSION
    }

    fn image(&self) -> Result<Box<dyn Image + '_>, ApiError> {
        load(&self.data).map_err(|e| ApiError::Load(format!("{}: {e}", self.label)))
    }

    pub fn arch(&self) -> Result<Arch, ApiError> {
        Ok(self.image()?.arch())
    }

    /// Recovered functions, capped and sorted by VA.
    pub fn functions(&self, limit: usize) -> Result<Vec<FunctionInfo>, ApiError> {
        let limit = limit.min(MAX_FUNCTIONS);
        let image = self.image()?;
        let program = cfg::recover(image.as_ref())
            .map_err(|e| ApiError::Unsupported(format!("{}: {e}", self.label)))?;
        let sources: BTreeMap<u64, funcs::Source> = funcs::discover(image.as_ref())
            .into_iter()
            .map(|f| (f.va, f.source))
            .collect();
        let index = anchor::AnchorIndex::build(image.as_ref(), &program);
        let mut asserted: BTreeMap<u64, String> = BTreeMap::new();
        for placed in self.db.resolve_onto(&index) {
            if placed.field == annotate::Field::Name {
                asserted.insert(placed.va, placed.value.to_string());
            }
        }
        let mut out = Vec::new();
        for (va, func) in program.functions.iter().take(limit) {
            let (name, source) = if let Some(n) = asserted.get(va) {
                (Some(n.clone()), "asserted".into())
            } else {
                let source = sources
                    .get(va)
                    .map(|s| format!("{s:?}").to_ascii_lowercase())
                    .unwrap_or_else(|| "cfg".into());
                (func.name.clone(), source)
            };
            out.push(FunctionInfo {
                va: *va,
                name,
                source,
            });
        }
        Ok(out)
    }

    /// Symbolized listing for one function entry.
    pub fn listing(&self, entry: u64, max_lines: usize) -> Result<String, ApiError> {
        let image = self.image()?;
        let program = cfg::recover(image.as_ref())
            .map_err(|e| ApiError::Unsupported(format!("{e}")))?;
        let opts = listing::Options {
            max_lines: max_lines.clamp(1, 262_144),
            max_functions: 1,
            ..listing::Options::default()
        };
        Ok(listing::render_function(
            image.as_ref(),
            &program,
            entry,
            Some(&self.db),
            &opts,
        ))
    }

    /// Full decompile pipeline for one entry (same honesty as `--decompile`).
    pub fn decompile(&self, entry: u64) -> Result<String, ApiError> {
        let image = self.image()?;
        if !matches!(image.arch(), Arch::X86_64 | Arch::Aarch64) {
            return Err(ApiError::Unsupported(
                "decompile needs x86-64 or aarch64".into(),
            ));
        }
        let folded = jumptable::resolve_folded(image.as_ref())
            .map_err(|e| ApiError::Unsupported(format!("{e}")))?;
        let program = &folded.program;
        let tables = jumptable::successor_map(&folded.tables);
        let func = program
            .functions
            .get(&entry)
            .ok_or_else(|| ApiError::NotFound(format!("function {entry:#x}")))?;
        let Some(lifted) = irlift::lift_function(image.as_ref(), func) else {
            return Err(ApiError::Unsupported(format!(
                "lift failed for {entry:#x}"
            )));
        };
        let lifted = match callfx::abi_for(image.arch()) {
            Some(abi) => callfx::apply(&lifted, &abi),
            None => lifted,
        };
        let ssa = irssa::construct(&lifted)
            .map_err(|e| ApiError::Unsupported(format!("ssa: {e}")))?;
        let (opt, _) = irssaopt::optimize(&ssa);
        let (fwd, _) = irssaopt::forward(&opt);
        let live_out = callfx::function_live_out(image.arch()).unwrap_or_default();
        let (swept, _) = irssaopt::eliminate_dead(&fwd, &live_out);
        let stack = irstack::analyze(&swept);
        let promote = mempromote::promote(&swept, &stack);
        let swept = mempromote::apply(&swept, &promote);
        let mut signature = sig::recover(&swept);
        sig::try_confirm_returns(&mut signature, program, image.as_ref());
        let type_facts = irtype::collect(&swept);
        let mut type_table = types::TypeTable::new();
        let type_map =
            irtype::attach_sig_with_evidence(&swept, &signature, &type_facts, &mut type_table);
        let (root, _stats) = irstruct::structure(&swept, &tables);
        let (vars, _) = irout::out_of_ssa(&swept);
        let names = mempromote::var_namer(&swept, &stack, &promote, &vars.var_of);
        let namer = |v: u32| names.get(&v).cloned();
        let header = type_map.render_proto(&signature, &type_table);
        Ok(pseudo::render_with_proto(
            &swept,
            &root,
            &vars,
            &namer,
            Some(&header),
        ))
    }

    /// Assert a name at a function entry (git-friendly annotation).
    pub fn rename(&mut self, va: u64, name: &str) -> Result<(), ApiError> {
        if name.is_empty() {
            return Err(ApiError::Unsupported("empty name".into()));
        }
        let image = load(&self.data).map_err(|e| ApiError::Load(format!("{e}")))?;
        let program = cfg::recover(image.as_ref())
            .map_err(|e| ApiError::Unsupported(format!("{e}")))?;
        let func = program
            .functions
            .get(&va)
            .ok_or_else(|| ApiError::NotFound(format!("function {va:#x}")))?;
        let target = anchor::of_function(image.as_ref(), func);
        self.db.set_name(target, name);
        Ok(())
    }

    /// Provenance / discovery source for a VA (Why?).
    pub fn why(&self, va: u64) -> Result<String, ApiError> {
        let image = self.image()?;
        let program = cfg::recover(image.as_ref())
            .map_err(|e| ApiError::Unsupported(format!("{e}")))?;
        let sources: BTreeMap<u64, funcs::Source> = funcs::discover(image.as_ref())
            .into_iter()
            .map(|f| (f.va, f.source))
            .collect();
        if let Some(func) = program.functions.get(&va) {
            let src = sources
                .get(&va)
                .map(|s| format!("{s:?}"))
                .unwrap_or_else(|| "Cfg".into());
            return Ok(format!(
                "function {va:#x} source={src} name={:?}",
                func.name
            ));
        }
        Ok(format!("va {va:#x}: no recovered function entry"))
    }

    /// Xrefs involving `va` (from / to counts + samples).
    pub fn xrefs(&self, va: u64) -> Result<String, ApiError> {
        let image = self.image()?;
        let program = cfg::recover(image.as_ref())
            .map_err(|e| ApiError::Unsupported(format!("{e}")))?;
        let table = xref::compute(image.as_ref(), &program)
            .map_err(|e| ApiError::Unsupported(format!("{e}")))?;
        let from = table.refs_from(va);
        let to = table.refs_to(va);
        let mut out = format!(
            "; xrefs va={va:#x} from={} to={}\n",
            from.len(),
            to.len()
        );
        for x in from.iter().chain(to.iter()).take(64) {
            out.push_str(&format!("  {x:?}\n"));
        }
        Ok(out)
    }
}

/// Machine-readable handshake for plugin hosts.
pub fn handshake() -> String {
    format!(
        "{{\"api_version\":\"{API_VERSION}\",\"engine_version\":\"{ENGINE_VERSION}\",\"protocol\":\"aletheia/1\"}}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_mentions_versions() {
        let h = handshake();
        assert!(h.contains(API_VERSION));
        assert!(h.contains("aletheia/1"));
    }

    #[test]
    fn open_diamond_lists_and_decompiles() {
        let path = "fixtures/diamond";
        let Ok(mut s) = AnalysisSession::open_path(path) else {
            return;
        };
        assert!(!s.encrypted());
        let fns = s.functions(8).expect("functions");
        assert!(!fns.is_empty());
        let entry = fns[0].va;
        let text = s.decompile(entry).expect("decompile");
        assert!(
            text.contains('(') || text.contains("sub_"),
            "unexpected decompile: {text}"
        );
        s.rename(entry, "audit_renamed").expect("rename");
        let fns2 = s.functions(8).expect("functions after rename");
        assert!(
            fns2
                .iter()
                .any(|f| f.va == entry && f.name.as_deref() == Some("audit_renamed")),
            "{fns2:?}"
        );
    }

    #[test]
    fn refuse_empty_rename() {
        let Ok(mut s) = AnalysisSession::open_path("fixtures/diamond") else {
            return;
        };
        let fns = s.functions(1).unwrap();
        if let Some(f) = fns.first() {
            assert!(s.rename(f.va, "").is_err());
        }
    }
}
