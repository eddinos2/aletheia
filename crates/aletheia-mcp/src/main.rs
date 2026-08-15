//! Minimal stdio JSON-RPC MCP-shaped server for Aletheia (Gate M1).
//!
//! Speaks newline-delimited JSON requests:
//!   {"id":1,"method":"health","params":{}}
//!   {"id":2,"method":"open","params":{"path":"/bin/ls"}}
//!   {"id":3,"method":"decompile","params":{"session":"...","entry":"0x1000"}}
//!
//! Designed so `health` never blocks on analysis (jobs tracked separately).
//! Full MCP SDK wiring can wrap this without changing the engine.

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aletheia::patch::fnv1a64;
use aletheia::{callfx, irlift, irout, irssa, irssaopt, irstruct, jumptable, pseudo};

const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

struct Session {
    #[allow(dead_code)]
    path: String,
    data: Vec<u8>,
    hash: u64,
}

struct State {
    sessions: BTreeMap<String, Session>,
    next_id: AtomicU64,
    busy: AtomicU64,
}

fn main() {
    let state = Arc::new(Mutex::new(State {
        sessions: BTreeMap::new(),
        next_id: AtomicU64::new(1),
        busy: AtomicU64::new(0),
    }));
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let reply = handle_line(&state, line);
        let _ = writeln!(stdout, "{reply}");
        let _ = stdout.flush();
    }
}

fn handle_line(state: &Arc<Mutex<State>>, line: &str) -> String {
    // Tiny hand-rolled extractors — no serde dependency in this skeleton.
    let id = extract_number(line, "\"id\"").unwrap_or(0);
    let method = extract_string(line, "\"method\"").unwrap_or_default();
    match method.as_str() {
        "health" => {
            let busy = state.lock().unwrap().busy.load(Ordering::Relaxed);
            json_ok(
                id,
                &format!(
                    r#"{{"ok":true,"engine_version":"{ENGINE_VERSION}","busy_jobs":{busy}}}"#
                ),
            )
        }
        "open" => {
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
                            },
                        );
                        sid
                    };
                    json_ok(
                        id,
                        &format!(
                            r#"{{"session_id":"{sid}","path":{path:?},"arch":"{arch}","hash":"0x{hash:x}"}}"#
                        ),
                    )
                }
                Err(e) => json_err(id, &format!("open failed: {e}")),
            }
        }
        "decompile" => {
            let session = match extract_string(line, "\"session\"").or_else(|| extract_string(line, "\"session_id\"")) {
                Some(s) => s,
                None => return json_err(id, "decompile requires session"),
            };
            let entry = extract_string(line, "\"entry\"")
                .and_then(|s| parse_hex(&s))
                .or_else(|| extract_number(line, "\"entry\""));
            state.lock().unwrap().busy.fetch_add(1, Ordering::Relaxed);
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
                    None => program.functions.values().next(),
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
                let (root, _) = irstruct::structure(&swept, &tables);
                let (vars, _) = irout::out_of_ssa(&swept);
                let text = pseudo::render(&swept, &root, &vars);
                let hash = sess.hash;
                Ok::<_, String>((text, hash))
            })();
            state.lock().unwrap().busy.fetch_sub(1, Ordering::Relaxed);
            match result {
                Ok((text, hash)) => {
                    let escaped = escape_json(&text);
                    json_ok(
                        id,
                        &format!(
                            r#"{{"pseudocode":"{escaped}","stamp":{{"hash":"0x{hash:x}","engine_version":"{ENGINE_VERSION}"}}}}"#
                        ),
                    )
                }
                Err(e) => json_err(id, &e),
            }
        }
        "why" => json_ok(
            id,
            r#"{"note":"provenance pins wire funcs::Source / anchor::Resolution; full fact_id graph is TUI Gate G1"}"#,
        ),
        "cancel" => json_ok(id, r#"{"cancelled":true,"note":"cooperative cancel stub"}"#),
        _ => json_err(id, &format!("unknown method `{method}`")),
    }
}

fn json_ok(id: u64, result: &str) -> String {
    format!(r#"{{"id":{id},"ok":true,"result":{result}}}"#)
}

fn json_err(id: u64, msg: &str) -> String {
    format!(r#"{{"id":{id},"ok":false,"error":"{}"}}"#, escape_json(msg))
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
    let s = s.trim().strip_prefix("0x").or_else(|| s.trim().strip_prefix("0X")).unwrap_or(s.trim());
    u64::from_str_radix(s, 16).ok()
}
