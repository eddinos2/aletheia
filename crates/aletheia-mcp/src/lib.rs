//! Aletheia protocol dispatch (shared by `aletheia-mcp` stdio and the GUI).
//!
//! Speaks newline-delimited JSON requests (see `protocol/PROTOCOL.md`):
//!   {"id":1,"method":"health","params":{}}
//!   {"id":2,"method":"open","params":{"path":"/bin/ls"}}
//!   {"id":3,"method":"functions","params":{"session":"s1","limit":32}}
//!   {"id":4,"method":"listing","params":{"session":"s1","entry":"0x1000"}}
//!   {"id":5,"method":"decompile","params":{"session":"s1","entry":"0x1000"}}
//!   {"id":6,"method":"stack","params":{"session":"s1","entry":"0x1000"}}
//!   {"id":7,"method":"xrefs","params":{"session":"s1","va":"0x1000"}}
//!   {"id":8,"method":"rename","params":{"session":"s1","va":"0x1000","name":"foo"}}
//!   {"id":9,"method":"diff","params":{"session_a":"s1","session_b":"s2"}}
//!   {"id":10,"method":"patch_preview","params":{"session":"s1","va":"0x1000"}}
//!   {"id":11,"method":"patch_apply","params":{"session":"s1","va":"0x1000"}}
//!   {"id":12,"method":"why","params":{"session":"s1","va":"0x1000"}}
//!   {"id":13,"method":"cfg","params":{"session":"s1","entry":"0x1000"}}
//!   {"id":14,"method":"locate","params":{"session":"s1","va":"0x1000"}}
//!
//! Frontends compute nothing: they call [`handle_line`] (or the stdio binary)
//! and render the engine's facts + provenance. Rename responses carry a
//! small `delta` so GUIs can patch navigator state without a full refetch.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aletheia::annotate;
use aletheia::patch::{self, fnv1a64};
use aletheia::{
    anchor, callfx, cfg, diff, funcs, irlift, irout, irssa, irssaopt, irstruct, irstack, irtype,
    jumptable, listing, mempromote, pseudo, sig, xref,
};

const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_HUNK_CAP: usize = 32;
const DEFAULT_FUNC_LIMIT: usize = 4096;

pub struct Session {
    path: String,
    data: Vec<u8>,
    hash: u64,
    db: annotate::Db,
}

pub struct State {
    pub sessions: BTreeMap<String, Session>,
    next_id: AtomicU64,
    /// In-flight long tools.
    busy: AtomicU64,
}

impl State {
    pub fn new() -> Self {
        Self {
            sessions: BTreeMap::new(),
            next_id: AtomicU64::new(1),
            busy: AtomicU64::new(0),
        }
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

/// Dispatch one NDJSON protocol request. Shared by the stdio MCP binary and
/// in-process GUI clients so both stay isomorphic with agents.
pub fn handle_line(state: &Arc<Mutex<State>>, line: &str) -> String {
    // Tiny hand-rolled extractors — no serde dependency in this skeleton.
    let id = extract_number(line, "\"id\"").unwrap_or(0);
    let method = extract_string(line, "\"method\"").unwrap_or_default();
    match method.as_str() {
        "health" => health(state, id),
        "open" => open(state, id, line),
        "functions" => with_busy(state, id, || functions(state, id, line)),
        "listing" => with_busy(state, id, || listing_method(state, id, line)),
        "decompile" => with_busy(state, id, || decompile(state, id, line)),
        "stack" => with_busy(state, id, || stack_method(state, id, line)),
        "xrefs" => with_busy(state, id, || xrefs_method(state, id, line)),
        "rename" => rename_method(state, id, line),
        "diff" => with_busy(state, id, || diff_sessions(state, id, line)),
        "patch_preview" => with_busy(state, id, || patch_preview(state, id, line)),
        "patch_apply" => with_busy(state, id, || patch_apply(state, id, line)),
        "why" => why(state, id, line),
        "cfg" => with_busy(state, id, || cfg_method(state, id, line)),
        "locate" => with_busy(state, id, || locate_method(state, id, line)),
        "cancel" => cancel(state, id),
        _ => json_err(id, &format!("unknown method `{method}`")),
    }
}

fn health(state: &Arc<Mutex<State>>, id: u64) -> String {
    let busy = state.lock().unwrap().busy.load(Ordering::Relaxed);
    json_ok(
        id,
        &format!(
            r#"{{"ok":true,"engine_version":"{ENGINE_VERSION}","api_version":"{}","busy_jobs":{busy},"stamp":{}}}"#,
            aletheia::api::API_VERSION,
            stamp_engine()
        ),
    )
}

fn open(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let path = match extract_string(line, "\"path\"") {
        Some(p) => p,
        None => return json_err(id, "open requires params.path"),
    };
    match std::fs::read(&path) {
        Ok(data) => {
            let hash = fnv1a64(&data);
            let arch = aletheia::load(&data)
                .map(|i| format!("{:?}", i.arch()))
                .unwrap_or_else(|_| "unknown".into());
            let encrypted = aletheia::macho::MachFile::parse(&data)
                .map(|m| m.is_encrypted())
                .unwrap_or(false);
            let sid = {
                let mut st = state.lock().unwrap();
                let n = st.next_id.fetch_add(1, Ordering::Relaxed);
                let sid = format!("s{n}");
                st.sessions.insert(
                    sid.clone(),
                    Session {
                        path: path.clone(),
                        data,
                        hash,
                        db: annotate::Db::new(),
                    },
                );
                sid
            };
            json_ok(
                id,
                &format!(
                    r#"{{"session_id":"{sid}","path":{path_json},"arch":"{arch}","hash":"0x{hash:x}","encrypted":{encrypted},"stamp":{stamp}}}"#,
                    path_json = json_string(&path),
                    stamp = stamp_hash(hash),
                ),
            )
        }
        Err(e) => json_err(id, &format!("open failed: {e}")),
    }
}

fn functions(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let session = match session_id(line) {
        Some(s) => s,
        None => return json_err(id, "functions requires session"),
    };
    let limit = extract_number(line, "\"limit\"").unwrap_or(DEFAULT_FUNC_LIMIT as u64) as usize;
    let st = state.lock().unwrap();
    let sess = match st.sessions.get(&session) {
        Some(s) => s,
        None => return json_err(id, "unknown session"),
    };
    let image = match aletheia::load(&sess.data) {
        Ok(i) => i,
        Err(e) => return json_err(id, &e.to_string()),
    };
    let program = match cfg::recover(image.as_ref()) {
        Ok(p) => p,
        Err(e) => return json_err(id, &e.to_string()),
    };
    let sources: BTreeMap<u64, funcs::Source> = funcs::discover(image.as_ref())
        .into_iter()
        .map(|f| (f.va, f.source))
        .collect();
    // Overlay asserted names from annotate::Db (same precedence as listing).
    let index = aletheia::anchor::AnchorIndex::build(image.as_ref(), &program);
    let mut asserted: BTreeMap<u64, String> = BTreeMap::new();
    for placed in sess.db.resolve_onto(&index) {
        if placed.field == annotate::Field::Name {
            asserted.insert(placed.va, placed.value.to_string());
        }
    }
    let mut items = Vec::new();
    for (i, (&va, func)) in program.functions.iter().enumerate() {
        if i >= limit {
            break;
        }
        let (name, source) = if let Some(n) = asserted.get(&va) {
            (format!(r#""{}""#, escape_json(n)), "asserted")
        } else {
            let name = match &func.name {
                Some(n) => format!(r#""{}""#, escape_json(n)),
                None => "null".into(),
            };
            let source = sources.get(&va).map(source_label).unwrap_or("cfg");
            (name, source)
        };
        items.push(format!(
            r#"{{"va":"0x{va:x}","name":{name},"source":"{source}"}}"#
        ));
    }
    let omitted = program.functions.len().saturating_sub(items.len());
    json_ok(
        id,
        &format!(
            r#"{{"session_id":"{session}","functions":[{}],"total":{},"omitted":{omitted},"stamp":{}}}"#,
            items.join(","),
            program.functions.len(),
            stamp_hash(sess.hash),
        ),
    )
}

fn listing_method(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let session = match session_id(line) {
        Some(s) => s,
        None => return json_err(id, "listing requires session"),
    };
    let entry = extract_string(line, "\"entry\"")
        .and_then(|s| parse_hex(&s))
        .or_else(|| extract_number(line, "\"entry\""));
    let Some(entry) = entry else {
        return json_err(id, "listing requires params.entry");
    };
    let max_lines = extract_number(line, "\"max_insns\"")
        .or_else(|| extract_number(line, "\"max_lines\""))
        .unwrap_or(65_536) as usize;

    let result = (|| {
        let st = state.lock().unwrap();
        let sess = st
            .sessions
            .get(&session)
            .ok_or_else(|| "unknown session".to_string())?;
        let image = aletheia::load(&sess.data).map_err(|e| e.to_string())?;
        let program = cfg::recover(image.as_ref()).map_err(|e| e.to_string())?;
        let opts = listing::Options {
            max_functions: 1,
            max_lines,
            ..listing::Options::default()
        };
        let text = listing::render_function(image.as_ref(), &program, entry, Some(&sess.db), &opts);
        Ok::<_, String>((text, sess.hash))
    })();

    match result {
        Ok((text, hash)) => json_ok(
            id,
            &format!(
                r#"{{"session_id":"{session}","entry":"0x{entry:x}","listing":"{escaped}","stamp":{stamp}}}"#,
                escaped = escape_json(&text),
                stamp = stamp_hash(hash),
            ),
        ),
        Err(e) => json_err(id, &e),
    }
}

fn decompile(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let session = match session_id(line) {
        Some(s) => s,
        None => return json_err(id, "decompile requires session"),
    };
    let entry = extract_string(line, "\"entry\"")
        .and_then(|s| parse_hex(&s))
        .or_else(|| extract_number(line, "\"entry\""));
    let result = (|| {
        let st = state.lock().unwrap();
        let sess = st
            .sessions
            .get(&session)
            .ok_or_else(|| "unknown session".to_string())?;
        let image = aletheia::load(&sess.data).map_err(|e| e.to_string())?;
        let folded = jumptable::resolve_folded(image.as_ref()).map_err(|e| e.to_string())?;
        let program = &folded.program;
        let tables = jumptable::successor_map(&folded.tables);
        let func = match entry {
            Some(va) => program.functions.get(&va),
            None => image
                .entry_points()
                .into_iter()
                .find_map(|va| program.functions.get(&va))
                .or_else(|| program.functions.values().next()),
        }
        .ok_or_else(|| "no function".to_string())?;
        let lifted = irlift::lift_function(image.as_ref(), func)
            .ok_or_else(|| "lift failed".to_string())?;
        let lifted = match callfx::abi_for(image.arch()) {
            Some(abi) => callfx::apply(&lifted, &abi),
            None => lifted,
        };
        let ssa = irssa::construct(&lifted).map_err(|e| e.to_string())?;
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
        let mut type_table = aletheia::types::TypeTable::new();
        let type_map =
            irtype::attach_sig_with_evidence(&swept, &signature, &type_facts, &mut type_table);
        let (root, _) = irstruct::structure(&swept, &tables);
        let (vars, _) = irout::out_of_ssa(&swept);
        let names = mempromote::var_namer(&swept, &stack, &promote, &vars.var_of);
        let namer = |v: u32| names.get(&v).cloned();
        let header = type_map.render_proto(&signature, &type_table);
        let text = pseudo::render_with_proto(&swept, &root, &vars, &namer, Some(&header));
        let hash = sess.hash;
        let entry_va = func.entry;
        Ok::<_, String>((text, hash, entry_va))
    })();
    match result {
        Ok((text, hash, entry_va)) => json_ok(
            id,
            &format!(
                r#"{{"session_id":"{session}","entry":"0x{entry_va:x}","pseudocode":"{escaped}","stamp":{stamp}}}"#,
                escaped = escape_json(&text),
                stamp = stamp_hash(hash),
            ),
        ),
        Err(e) => json_err(id, &e),
    }
}

fn stack_method(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let session = match session_id(line) {
        Some(s) => s,
        None => return json_err(id, "stack requires session"),
    };
    let entry = extract_string(line, "\"entry\"")
        .and_then(|s| parse_hex(&s))
        .or_else(|| extract_number(line, "\"entry\""));
    let Some(entry) = entry else {
        return json_err(id, "stack requires params.entry");
    };

    let result = (|| {
        let st = state.lock().unwrap();
        let sess = st
            .sessions
            .get(&session)
            .ok_or_else(|| "unknown session".to_string())?;
        let image = aletheia::load(&sess.data).map_err(|e| e.to_string())?;
        let folded = jumptable::resolve_folded(image.as_ref()).map_err(|e| e.to_string())?;
        let func = folded
            .program
            .functions
            .get(&entry)
            .ok_or_else(|| format!("no function at {entry:#x}"))?;
        let lifted = irlift::lift_function(image.as_ref(), func)
            .ok_or_else(|| "lift failed".to_string())?;
        let lifted = match callfx::abi_for(image.arch()) {
            Some(abi) => callfx::apply(&lifted, &abi),
            None => lifted,
        };
        let ssa = irssa::construct(&lifted).map_err(|e| e.to_string())?;
        let (opt, _) = irssaopt::optimize(&ssa);
        let (fwd, _) = irssaopt::forward(&opt);
        let live_out = callfx::function_live_out(image.arch()).unwrap_or_default();
        let (swept, _) = irssaopt::eliminate_dead(&fwd, &live_out);
        let facts = irstack::analyze(&swept);
        let text = facts.render();
        Ok::<_, String>((text, sess.hash))
    })();

    match result {
        Ok((text, hash)) => json_ok(
            id,
            &format!(
                r#"{{"session_id":"{session}","entry":"0x{entry:x}","stack":"{escaped}","stamp":{stamp}}}"#,
                escaped = escape_json(&text),
                stamp = stamp_hash(hash),
            ),
        ),
        Err(e) => json_err(id, &e),
    }
}

fn xrefs_method(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let session = match session_id(line) {
        Some(s) => s,
        None => return json_err(id, "xrefs requires session"),
    };
    let va = extract_string(line, "\"va\"")
        .and_then(|s| parse_hex(&s))
        .or_else(|| extract_number(line, "\"va\""));
    let Some(va) = va else {
        return json_err(id, "xrefs requires params.va");
    };

    let result = (|| {
        let st = state.lock().unwrap();
        let sess = st
            .sessions
            .get(&session)
            .ok_or_else(|| "unknown session".to_string())?;
        let image = aletheia::load(&sess.data).map_err(|e| e.to_string())?;
        let program = cfg::recover(image.as_ref()).map_err(|e| e.to_string())?;
        let xrefs = xref::compute(image.as_ref(), &program).map_err(|e| e.to_string())?;
        let from_json: Vec<String> = xrefs
            .refs_from(va)
            .iter()
            .map(|x| {
                let label = xrefs
                    .to_label(x)
                    .map(|s| format!(r#""{}""#, escape_json(s)))
                    .unwrap_or_else(|| "null".into());
                format!(
                    r#"{{"from":"0x{:x}","to":"0x{:x}","kind":"{}","label":{label}}}"#,
                    x.from,
                    x.to,
                    x.kind.as_str(),
                )
            })
            .collect();
        let to_json: Vec<String> = xrefs
            .refs_to(va)
            .iter()
            .map(|x| {
                let label = xrefs
                    .to_label(x)
                    .map(|s| format!(r#""{}""#, escape_json(s)))
                    .unwrap_or_else(|| "null".into());
                format!(
                    r#"{{"from":"0x{:x}","to":"0x{:x}","kind":"{}","label":{label}}}"#,
                    x.from,
                    x.to,
                    x.kind.as_str(),
                )
            })
            .collect();
        Ok::<_, String>((from_json, to_json, sess.hash, xrefs.len()))
    })();

    match result {
        Ok((from_json, to_json, hash, total)) => json_ok(
            id,
            &format!(
                r#"{{"session_id":"{session}","va":"0x{va:x}","from":[{from}],"to":[{to}],"total":{total},"stamp":{stamp}}}"#,
                from = from_json.join(","),
                to = to_json.join(","),
                stamp = stamp_hash(hash),
            ),
        ),
        Err(e) => json_err(id, &e),
    }
}

fn rename_method(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let session = match session_id(line) {
        Some(s) => s,
        None => return json_err(id, "rename requires session"),
    };
    let va = extract_string(line, "\"va\"")
        .and_then(|s| parse_hex(&s))
        .or_else(|| extract_string(line, "\"anchor\"").and_then(|s| parse_hex(&s)))
        .or_else(|| extract_number(line, "\"va\""));
    let Some(va) = va else {
        return json_err(id, "rename requires params.va (or anchor as hex VA)");
    };
    let Some(name) = extract_string(line, "\"name\"") else {
        return json_err(id, "rename requires params.name");
    };

    let result = (|| {
        let mut st = state.lock().unwrap();
        let sess = st
            .sessions
            .get_mut(&session)
            .ok_or_else(|| "unknown session".to_string())?;
        let image = aletheia::load(&sess.data).map_err(|e| e.to_string())?;
        let program = cfg::recover(image.as_ref()).map_err(|e| e.to_string())?;
        let func = program
            .functions
            .get(&va)
            .ok_or_else(|| format!("no function at {va:#x}"))?;
        let target = anchor::of_function(image.as_ref(), func);
        sess.db.set_name(target, &name);
        let tip = sess
            .db
            .log()
            .last()
            .map(|a| {
                format!(
                    r#"{{"seq":{},"id":"0x{:x}","field":"name","va":"0x{va:x}","name":"{}"}}"#,
                    a.seq,
                    a.id,
                    escape_json(&name),
                )
            })
            .unwrap_or_else(|| "null".into());
        let hash = sess.hash;
        // Incremental delta: patch one navigator row; soft-invalidate
        // views that embed the name (listing / why). Decompile text does
        // not currently fold annotate names — still listed so clients can
        // choose a clean refetch policy without guessing.
        let delta = format!(
            r#"{{"kind":"annotate","functions":[{{"va":"0x{va:x}","name":"{n}","source":"asserted"}}],"invalidate":[{{"view":"listing","va":"0x{va:x}"}},{{"view":"why","va":"0x{va:x}"}},{{"view":"decompile","va":"0x{va:x}"}}]}}"#,
            n = escape_json(&name),
        );
        Ok::<_, String>((tip, delta, hash))
    })();

    match result {
        Ok((tip, delta, hash)) => json_ok(
            id,
            &format!(
                r#"{{"session_id":"{session}","tip":{tip},"delta":{delta},"stamp":{stamp}}}"#,
                stamp = stamp_hash(hash),
            ),
        ),
        Err(e) => json_err(id, &e),
    }
}

/// Engine-owned CFG for one function: blocks + successor edges only.
/// Frontends may lay out the graph; they must not invent edges.
fn cfg_method(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let session = match session_id(line) {
        Some(s) => s,
        None => return json_err(id, "cfg requires session"),
    };
    let entry = extract_string(line, "\"entry\"")
        .and_then(|s| parse_hex(&s))
        .or_else(|| extract_string(line, "\"va\"").and_then(|s| parse_hex(&s)))
        .or_else(|| extract_number(line, "\"entry\""));
    let Some(entry) = entry else {
        return json_err(id, "cfg requires params.entry");
    };

    let result = (|| {
        let st = state.lock().unwrap();
        let sess = st
            .sessions
            .get(&session)
            .ok_or_else(|| "unknown session".to_string())?;
        let image = aletheia::load(&sess.data).map_err(|e| e.to_string())?;
        let program = cfg::recover(image.as_ref()).map_err(|e| e.to_string())?;
        let func = program
            .functions
            .get(&entry)
            .ok_or_else(|| format!("no function at {entry:#x}"))?;
        let mut blocks = Vec::new();
        let mut edges = Vec::new();
        for (&start, block) in &func.blocks {
            let term = terminator_label(&block.terminator);
            let succs: Vec<String> = block
                .successors
                .iter()
                .map(|s| format!(r#""0x{s:x}""#))
                .collect();
            blocks.push(format!(
                r#"{{"start":"0x{start:x}","end":"0x{:x}","terminator":"{term}","successors":[{}]}}"#,
                block.end,
                succs.join(","),
            ));
            for &to in &block.successors {
                edges.push(format!(
                    r#"{{"from":"0x{start:x}","to":"0x{to:x}"}}"#
                ));
            }
        }
        let name = match &func.name {
            Some(n) => format!(r#""{}""#, escape_json(n)),
            None => "null".into(),
        };
        Ok::<_, String>((
            name,
            blocks,
            edges,
            func.blocks.len(),
            sess.hash,
        ))
    })();

    match result {
        Ok((name, blocks, edges, nblocks, hash)) => json_ok(
            id,
            &format!(
                r#"{{"session_id":"{session}","entry":"0x{entry:x}","name":{name},"blocks":[{blocks}],"edges":[{edges}],"block_count":{nblocks},"stamp":{stamp}}}"#,
                blocks = blocks.join(","),
                edges = edges.join(","),
                stamp = stamp_hash(hash),
            ),
        ),
        Err(e) => json_err(id, &e),
    }
}

/// Map an arbitrary VA to the recovered function / block that owns it.
/// Used by xref click-navigation when the target is not a function entry.
fn locate_method(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let session = match session_id(line) {
        Some(s) => s,
        None => return json_err(id, "locate requires session"),
    };
    let va = extract_string(line, "\"va\"")
        .and_then(|s| parse_hex(&s))
        .or_else(|| extract_number(line, "\"va\""));
    let Some(va) = va else {
        return json_err(id, "locate requires params.va");
    };

    let result = (|| {
        let st = state.lock().unwrap();
        let sess = st
            .sessions
            .get(&session)
            .ok_or_else(|| "unknown session".to_string())?;
        let image = aletheia::load(&sess.data).map_err(|e| e.to_string())?;
        let program = cfg::recover(image.as_ref()).map_err(|e| e.to_string())?;

        // Exact entry wins.
        if program.functions.contains_key(&va) {
            let block = program.functions[&va]
                .blocks
                .get(&va)
                .map(|b| b.start);
            return Ok::<_, String>((
                Some(va),
                block,
                true,
                sess.hash,
            ));
        }

        // Otherwise: first function (entry ascending) whose block range
        // covers `va`. Engine fact only — no invented containing function.
        let mut found_entry = None;
        let mut found_block = None;
        for (&entry, func) in &program.functions {
            for (&start, block) in &func.blocks {
                if va >= block.start && va < block.end {
                    found_entry = Some(entry);
                    found_block = Some(start);
                    break;
                }
            }
            if found_entry.is_some() {
                break;
            }
        }
        Ok::<_, String>((found_entry, found_block, false, sess.hash))
    })();

    match result {
        Ok((func, block, exact_entry, hash)) => {
            let func_json = match func {
                Some(f) => format!(r#""0x{f:x}""#),
                None => "null".into(),
            };
            let block_json = match block {
                Some(b) => format!(r#""0x{b:x}""#),
                None => "null".into(),
            };
            json_ok(
                id,
                &format!(
                    r#"{{"session_id":"{session}","va":"0x{va:x}","function":{func_json},"block":{block_json},"exact_entry":{exact_entry},"stamp":{stamp}}}"#,
                    stamp = stamp_hash(hash),
                ),
            )
        }
        Err(e) => json_err(id, &e),
    }
}

fn diff_sessions(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let a = extract_string(line, "\"session_a\"")
        .or_else(|| extract_string(line, "\"old\""))
        .or_else(|| extract_string(line, "\"session\""));
    let b = extract_string(line, "\"session_b\"")
        .or_else(|| extract_string(line, "\"new\""))
        .or_else(|| extract_string(line, "\"other\""));
    let (session_a, session_b) = match (a, b) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            return json_err(
                id,
                "diff requires session_a and session_b (aliases: old/new, session/other)",
            );
        }
    };
    let hunk_cap = extract_number(line, "\"hunk_cap\"").unwrap_or(DEFAULT_HUNK_CAP as u64) as usize;

    let result = (|| {
        let st = state.lock().unwrap();
        let old_sess = st
            .sessions
            .get(&session_a)
            .ok_or_else(|| format!("unknown session `{session_a}`"))?;
        let new_sess = st
            .sessions
            .get(&session_b)
            .ok_or_else(|| format!("unknown session `{session_b}`"))?;
        let old_image = aletheia::load(&old_sess.data).map_err(|e| e.to_string())?;
        let new_image = aletheia::load(&new_sess.data).map_err(|e| e.to_string())?;
        let old_program = cfg::recover(old_image.as_ref()).map_err(|e| e.to_string())?;
        let new_program = cfg::recover(new_image.as_ref()).map_err(|e| e.to_string())?;
        let d = diff::diff(
            old_image.as_ref(),
            &old_program,
            new_image.as_ref(),
            &new_program,
        );
        let report = diff::render(&d);
        let counts = d.counts();
        let hunks = patch::hunks_from_modified(
            old_image.as_ref(),
            new_image.as_ref(),
            &d,
            hunk_cap,
        );
        let hunk_json: Vec<String> = hunks
            .iter()
            .map(|h| {
                format!(
                    r#"{{"old_va":"0x{:x}","new_va":"0x{:x}","old_prefix":"{}","new_prefix":"{}"}}"#,
                    h.old_va,
                    h.new_va,
                    hex_bytes(&h.old_prefix),
                    hex_bytes(&h.new_prefix),
                )
            })
            .collect();
        Ok::<_, String>((
            report,
            counts,
            hunk_json,
            old_sess.hash,
            new_sess.hash,
        ))
    })();

    match result {
        Ok((report, counts, hunk_json, old_hash, new_hash)) => json_ok(
            id,
            &format!(
                r#"{{"session_a":"{session_a}","session_b":"{session_b}","report":"{report}","counts":{{"unchanged":{u},"moved":{m},"modified":{modi},"uncertain":{unc},"added":{a},"removed":{r}}},"hunks":[{hunks}],"stamp":{{"old_hash":"0x{old_hash:x}","new_hash":"0x{new_hash:x}","engine_version":"{ENGINE_VERSION}"}}}}"#,
                report = escape_json(&report),
                u = counts.unchanged,
                m = counts.moved,
                modi = counts.modified,
                unc = counts.uncertain,
                a = counts.added,
                r = counts.removed,
                hunks = hunk_json.join(","),
            ),
        ),
        Err(e) => json_err(id, &e),
    }
}

fn patch_preview(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let session = match session_id(line) {
        Some(s) => s,
        None => return json_err(id, "patch_preview requires session"),
    };
    let va = extract_string(line, "\"va\"")
        .and_then(|s| parse_hex(&s))
        .or_else(|| extract_number(line, "\"va\""));
    let Some(va) = va else {
        return json_err(id, "patch_preview requires params.va (NOP preview)");
    };
    let intent = extract_string(line, "\"intent\"").unwrap_or_else(|| "mcp patch_preview".into());

    let result = (|| {
        let st = state.lock().unwrap();
        let sess = st
            .sessions
            .get(&session)
            .ok_or_else(|| "unknown session".to_string())?;
        let image = aletheia::load(&sess.data).map_err(|e| e.to_string())?;
        let off = image
            .va_to_offset(va)
            .ok_or_else(|| format!("VA {va:#x} unmapped"))?;
        let bytes = image.bytes();
        let len = if matches!(image.arch(), aletheia::Arch::Aarch64) {
            4
        } else {
            1
        };
        if off + len > bytes.len() {
            return Err(format!("VA {va:#x} past end of file"));
        }
        let old = bytes[off..off + len].to_vec();
        let mut set = patch::nop_patch(image.as_ref(), va, &old, &intent)
            .map_err(|e| e.to_string())?;
        if sess.path.contains(".app/") || sess.path.ends_with(".dylib") || looks_macho(&sess.data)
        {
            set = set.with_macho_resign_recipe(&sess.path);
        }
        let report = set.preview(image.as_ref()).map_err(|e| e.to_string())?;
        Ok::<_, String>((report, sess.hash, set.target_hash, set.edits.len()))
    })();

    match result {
        Ok((report, hash, target_hash, edits)) => json_ok(
            id,
            &format!(
                r#"{{"session_id":"{session}","va":"0x{va:x}","edits":{edits},"target_hash":"0x{target_hash:x}","report":"{escaped}","stamp":{stamp}}}"#,
                escaped = escape_json(&report),
                stamp = stamp_hash(hash),
            ),
        ),
        Err(e) => json_err(id, &e),
    }
}

fn patch_apply(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let session = match session_id(line) {
        Some(s) => s,
        None => return json_err(id, "patch_apply requires session"),
    };
    let va = extract_string(line, "\"va\"")
        .and_then(|s| parse_hex(&s))
        .or_else(|| extract_number(line, "\"va\""));
    let Some(va) = va else {
        return json_err(id, "patch_apply requires params.va (NOP apply)");
    };
    let intent = extract_string(line, "\"intent\"").unwrap_or_else(|| "mcp patch_apply".into());

    let result = (|| {
        let st = state.lock().unwrap();
        let sess = st
            .sessions
            .get(&session)
            .ok_or_else(|| "unknown session".to_string())?;
        let image = aletheia::load(&sess.data).map_err(|e| e.to_string())?;
        let off = image
            .va_to_offset(va)
            .ok_or_else(|| format!("VA {va:#x} unmapped"))?;
        let bytes = image.bytes();
        let len = if matches!(image.arch(), aletheia::Arch::Aarch64) {
            4
        } else {
            1
        };
        if off + len > bytes.len() {
            return Err(format!("VA {va:#x} past end of file"));
        }
        let old = bytes[off..off + len].to_vec();
        let mut set = patch::nop_patch(image.as_ref(), va, &old, &intent)
            .map_err(|e| e.to_string())?;
        if sess.path.contains(".app/") || sess.path.ends_with(".dylib") || looks_macho(&sess.data)
        {
            set = set.with_macho_resign_recipe(&sess.path);
        }
        let out_path = set
            .apply_sibling(image.as_ref(), std::path::Path::new(&sess.path))
            .map_err(|e| e.to_string())?;
        Ok::<_, String>((out_path.display().to_string(), sess.hash, set.target_hash))
    })();

    match result {
        Ok((out_path, hash, target_hash)) => json_ok(
            id,
            &format!(
                r#"{{"session_id":"{session}","va":"0x{va:x}","path":{path_json},"target_hash":"0x{target_hash:x}","stamp":{stamp}}}"#,
                path_json = json_string(&out_path),
                stamp = stamp_hash(hash),
            ),
        ),
        Err(e) => json_err(id, &e),
    }
}

fn why(state: &Arc<Mutex<State>>, id: u64, line: &str) -> String {
    let session = session_id(line);
    let fact = extract_string(line, "\"fact_id\"").unwrap_or_default();
    let va = extract_string(line, "\"va\"")
        .and_then(|s| parse_hex(&s))
        .or_else(|| extract_string(line, "\"entry\"").and_then(|s| parse_hex(&s)))
        .or_else(|| extract_number(line, "\"va\""))
        .or_else(|| parse_hex(&fact));

    let sess_json = match &session {
        Some(s) => format!(r#""{s}""#),
        None => "null".into(),
    };

    // Prefer a real funcs::Source chain when session + VA are known.
    if let (Some(session), Some(va)) = (session.clone(), va) {
        let st = state.lock().unwrap();
        if let Some(sess) = st.sessions.get(&session)
            && let Ok(image) = aletheia::load(&sess.data)
        {
                let sources: BTreeMap<u64, funcs::Source> = funcs::discover(image.as_ref())
                    .into_iter()
                    .map(|f| (f.va, f.source))
                    .collect();
                let source = sources.get(&va).copied();
                let name = cfg::recover(image.as_ref())
                    .ok()
                    .and_then(|p| p.functions.get(&va).and_then(|f| f.name.clone()));
                let (trust, verdict, rule, negative) = match source {
                    Some(funcs::Source::EntryPoint) => (
                        "proven",
                        "proven — image entry point",
                        "Image::entry_points",
                        "n/a — entry is authoritative",
                    ),
                    Some(funcs::Source::Symbol) => (
                        "proven",
                        "proven — metadata-backed Function symbol",
                        "Image::symbols (Function-kind in executable region)",
                        "prologue scan not required",
                    ),
                    Some(funcs::Source::Unwind) => (
                        "proven",
                        "proven — unwind / function-starts metadata",
                        "Image::function_starts_hint (.pdata / .eh_frame / LC_FUNCTION_STARTS)",
                        "prologue scan not required",
                    ),
                    Some(funcs::Source::GoPclntab) => (
                        "proven",
                        "proven — Go pclntab entry",
                        "gopcln / pclntab",
                        "prologue scan not required",
                    ),
                    Some(funcs::Source::Prologue) => (
                        "heuristic",
                        "heuristic — verify before relying on it",
                        "matched a conservative prologue pattern",
                        "no Symbol, Unwind, EntryPoint, or GoPclntab cover this address",
                    ),
                    None => (
                        "heuristic",
                        "unknown — address not in funcs::discover set",
                        "no discovery source recorded",
                        "function may exist only via CFG recovery",
                    ),
                };
                let source_label = source.map(|s| source_label(&s)).unwrap_or("unknown");
                let name_json = match &name {
                    Some(n) => format!(r#""{}""#, escape_json(n)),
                    None => "null".into(),
                };
                let claim = match &name {
                    Some(n) => format!("name = \"{n}\" @ 0x{va:x}"),
                    None => format!("function start @ 0x{va:x}"),
                };
                return json_ok(
                    id,
                    &format!(
                        r#"{{"session_id":{sess_json},"fact_id":"{fact}","va":"0x{va:x}","name":{name_json},"trust":"{trust}","source":"{source_label}","chain":[{{"lab":"CLAIM","val":"{claim}"}},{{"lab":"SOURCE","val":"funcs::Source::{src} — {rule}"}},{{"lab":"NEGATIVE","val":"{negative}"}},{{"lab":"VERDICT","val":"{verdict}"}}],"stamp":{stamp}}}"#,
                        fact = escape_json(&fact),
                        claim = escape_json(&claim),
                        src = match source {
                            Some(funcs::Source::EntryPoint) => "EntryPoint",
                            Some(funcs::Source::Symbol) => "Symbol",
                            Some(funcs::Source::Unwind) => "Unwind",
                            Some(funcs::Source::GoPclntab) => "GoPclntab",
                            Some(funcs::Source::Prologue) => "Prologue",
                            None => "Unknown",
                        },
                        rule = escape_json(rule),
                        negative = escape_json(negative),
                        verdict = escape_json(verdict),
                        stamp = stamp_hash(sess.hash),
                    ),
                );
        }
    }

    json_ok(
        id,
        &format!(
            r#"{{"session_id":{sess_json},"fact_id":"{fact}","note":"pass session + va (or fact_id as hex VA) for funcs::Source provenance","stamp":{}}}"#,
            stamp_engine(),
            fact = escape_json(&fact),
        ),
    )
}

fn cancel(state: &Arc<Mutex<State>>, id: u64) -> String {
    // Cooperative cancel stub: long tools today run synchronously on the
    // request thread. Report current busy count so agents can poll health.
    let busy = state.lock().unwrap().busy.load(Ordering::Relaxed);
    json_ok(
        id,
        &format!(
            r#"{{"cancelled":true,"busy_jobs":{busy},"note":"cooperative cancel stub; in-flight sync tools finish their current request","stamp":{}}}"#,
            stamp_engine()
        ),
    )
}

fn with_busy(state: &Arc<Mutex<State>>, id: u64, f: impl FnOnce() -> String) -> String {
    state.lock().unwrap().busy.fetch_add(1, Ordering::Relaxed);
    let reply = f();
    state.lock().unwrap().busy.fetch_sub(1, Ordering::Relaxed);
    let _ = id;
    reply
}

fn session_id(line: &str) -> Option<String> {
    extract_string(line, "\"session\"")
        .or_else(|| extract_string(line, "\"session_id\""))
}

fn stamp_hash(hash: u64) -> String {
    format!(r#"{{"hash":"0x{hash:x}","engine_version":"{ENGINE_VERSION}"}}"#)
}

fn stamp_engine() -> String {
    format!(r#"{{"engine_version":"{ENGINE_VERSION}"}}"#)
}

fn source_label(source: &funcs::Source) -> &'static str {
    match source {
        funcs::Source::EntryPoint => "entry",
        funcs::Source::Symbol => "symbol",
        funcs::Source::Unwind => "unwind",
        funcs::Source::GoPclntab => "gopclntab",
        funcs::Source::Prologue => "prologue",
    }
}

fn terminator_label(t: &cfg::Terminator) -> &'static str {
    match t {
        cfg::Terminator::Jump(_) => "jump",
        cfg::Terminator::CondJump { .. } => "cond",
        cfg::Terminator::IndirectJump { .. } => "ijmp",
        cfg::Terminator::Call { .. } => "call",
        cfg::Terminator::IndirectCall { .. } => "icall",
        cfg::Terminator::Return => "ret",
        cfg::Terminator::Interrupt { .. } => "int",
        cfg::Terminator::Halt => "halt",
        cfg::Terminator::Undecodable => "undec",
        cfg::Terminator::Truncated => "trunc",
        cfg::Terminator::FallThrough(_) => "fall",
    }
}

fn looks_macho(data: &[u8]) -> bool {
    data.len() >= 4
        && matches!(
            u32::from_le_bytes([data[0], data[1], data[2], data[3]]),
            0xFEED_FACF | 0xFEED_FACE | 0xBEBA_FECA | 0xCFFA_EDFE | 0xCEFA_EDFE
        )
}

fn hex_bytes(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn json_ok(id: u64, result: &str) -> String {
    format!(r#"{{"id":{id},"ok":true,"result":{result}}}"#)
}

fn json_err(id: u64, msg: &str) -> String {
    format!(
        r#"{{"id":{id},"ok":false,"error":"{}"}}"#,
        escape_json(msg)
    )
}

fn json_string(s: &str) -> String {
    format!(r#""{}""#, escape_json(s))
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn extract_string(line: &str, key: &str) -> Option<String> {
    let i = line.find(key)?;
    let rest = &line[i + key.len()..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let mut chars = rest[1..].chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => {
                if let Some(n) = chars.next() {
                    out.push(n);
                }
            }
            c => out.push(c),
        }
    }
    None
}

fn extract_number(line: &str, key: &str) -> Option<u64> {
    let i = line.find(key)?;
    let rest = &line[i + key.len()..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let num: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num.parse().ok()
}

fn parse_hex(s: &str) -> Option<u64> {
    let s = s
        .trim()
        .strip_prefix("0x")
        .or_else(|| s.trim().strip_prefix("0X"))
        .unwrap_or(s.trim());
    u64::from_str_radix(s, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures")
            .join(name)
    }

    fn open_diamond(state: &Arc<Mutex<State>>) -> String {
        let path = fixture("diamond");
        let line = format!(
            r#"{{"id":1,"method":"open","params":{{"path":"{}"}}}}"#,
            path.display()
        );
        let reply = handle_line(state, &line);
        assert!(reply.contains(r#""ok":true"#), "{reply}");
        extract_string(&reply, "\"session_id\"").expect("session_id")
    }

    #[test]
    fn rename_returns_incremental_delta() {
        let state = Arc::new(Mutex::new(State::new()));
        let sid = open_diamond(&state);
        let line = format!(
            r#"{{"id":2,"method":"rename","params":{{"session":"{sid}","va":"0x1000003d0","name":"g1_renamed"}}}}"#
        );
        let reply = handle_line(&state, &line);
        assert!(reply.contains(r#""ok":true"#), "{reply}");
        assert!(reply.contains(r#""delta""#), "{reply}");
        assert!(reply.contains(r#""kind":"annotate""#), "{reply}");
        assert!(reply.contains(r#""source":"asserted""#), "{reply}");
        assert!(reply.contains(r#""invalidate""#), "{reply}");
        assert!(reply.contains("g1_renamed"), "{reply}");

        // functions refetch must surface asserted name without inventing facts
        let funcs = handle_line(
            &state,
            &format!(
                r#"{{"id":3,"method":"functions","params":{{"session":"{sid}","limit":64}}}}"#
            ),
        );
        assert!(funcs.contains("g1_renamed"), "{funcs}");
        assert!(funcs.contains(r#""source":"asserted""#), "{funcs}");
    }

    #[test]
    fn cfg_returns_engine_blocks_and_edges_only() {
        let state = Arc::new(Mutex::new(State::new()));
        let sid = open_diamond(&state);
        let reply = handle_line(
            &state,
            &format!(
                r#"{{"id":4,"method":"cfg","params":{{"session":"{sid}","entry":"0x1000003d0"}}}}"#
            ),
        );
        assert!(reply.contains(r#""ok":true"#), "{reply}");
        assert!(reply.contains(r#""blocks":["#), "{reply}");
        assert!(reply.contains(r#""edges":["#), "{reply}");
        assert!(reply.contains(r#""terminator""#), "{reply}");
        // diamond has a branch → at least one cond/edge
        assert!(reply.contains(r#""block_count""#), "{reply}");
    }

    #[test]
    fn locate_maps_entry_and_interior_va() {
        let state = Arc::new(Mutex::new(State::new()));
        let sid = open_diamond(&state);
        let entry = handle_line(
            &state,
            &format!(
                r#"{{"id":5,"method":"locate","params":{{"session":"{sid}","va":"0x1000003d0"}}}}"#
            ),
        );
        assert!(entry.contains(r#""exact_entry":true"#), "{entry}");
        assert!(entry.contains(r#""function":"0x1000003d0""#), "{entry}");

        // Interior byte of diamond (entry+4) should still resolve to the function.
        let interior = handle_line(
            &state,
            &format!(
                r#"{{"id":6,"method":"locate","params":{{"session":"{sid}","va":"0x1000003d4"}}}}"#
            ),
        );
        assert!(interior.contains(r#""function":"0x1000003d0""#), "{interior}");
        assert!(interior.contains(r#""exact_entry":false"#), "{interior}");
    }

    #[test]
    fn xrefs_are_bidirectional_click_targets() {
        let state = Arc::new(Mutex::new(State::new()));
        let sid = open_diamond(&state);
        // main calls diamond — refs_to(diamond) should include a from site.
        let reply = handle_line(
            &state,
            &format!(
                r#"{{"id":7,"method":"xrefs","params":{{"session":"{sid}","va":"0x1000003d0"}}}}"#
            ),
        );
        assert!(reply.contains(r#""ok":true"#), "{reply}");
        assert!(reply.contains(r#""from":["#) || reply.contains(r#""to":["#), "{reply}");
        assert!(reply.contains(r#""kind""#), "{reply}");
    }
}
