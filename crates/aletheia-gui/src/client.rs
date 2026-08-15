//! Thin protocol client: in-process [`aletheia_mcp::handle_line`] only.
//! GUI never derives analysis facts — it parses engine JSON.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use aletheia_mcp::State;

static REQ_ID: AtomicU64 = AtomicU64::new(1);

pub struct Client {
    state: Arc<Mutex<State>>,
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(State::new())),
        }
    }

    fn call(&self, method: &str, params: &str) -> Result<String, String> {
        let id = REQ_ID.fetch_add(1, Ordering::Relaxed);
        let line = format!(r#"{{"id":{id},"method":"{method}","params":{params}}}"#);
        let reply = aletheia_mcp::handle_line(&self.state, &line);
        if reply.contains(r#""ok":false"#) {
            let err = extract_string(&reply, "\"error\"").unwrap_or_else(|| reply.clone());
            return Err(err);
        }
        // Prefer the `result` object when present.
        if let Some(start) = reply.find(r#""result":"#) {
            let rest = &reply[start + 9..];
            return Ok(rest.trim().to_string());
        }
        Ok(reply)
    }

    pub fn health(&self) -> Result<Health, String> {
        let r = self.call("health", "{}")?;
        Ok(Health {
            engine_version: extract_string(&r, "\"engine_version\"").unwrap_or_default(),
            busy_jobs: extract_number(&r, "\"busy_jobs\"").unwrap_or(0),
        })
    }

    pub fn open(&self, path: &str) -> Result<OpenInfo, String> {
        let params = format!(r#"{{"path":"{}"}}"#, escape_json(path));
        let r = self.call("open", &params)?;
        Ok(OpenInfo {
            session_id: extract_string(&r, "\"session_id\"").unwrap_or_default(),
            path: extract_string(&r, "\"path\"").unwrap_or_else(|| path.to_string()),
            arch: extract_string(&r, "\"arch\"").unwrap_or_default(),
            hash: extract_string(&r, "\"hash\"").unwrap_or_default(),
            encrypted: r.contains(r#""encrypted":true"#),
        })
    }

    pub fn functions(&self, session: &str, limit: usize) -> Result<Vec<FuncRow>, String> {
        let params = format!(r#"{{"session":"{session}","limit":{limit}}}"#);
        let r = self.call("functions", &params)?;
        Ok(parse_functions(&r))
    }

    pub fn listing(&self, session: &str, entry: u64) -> Result<String, String> {
        let params = format!(r#"{{"session":"{session}","entry":"0x{entry:x}"}}"#);
        let r = self.call("listing", &params)?;
        extract_string(&r, "\"listing\"").ok_or_else(|| "no listing".into())
    }

    pub fn decompile(&self, session: &str, entry: u64) -> Result<String, String> {
        let params = format!(r#"{{"session":"{session}","entry":"0x{entry:x}"}}"#);
        let r = self.call("decompile", &params)?;
        extract_string(&r, "\"pseudocode\"").ok_or_else(|| "no pseudocode".into())
    }

    /// Rename returns an engine `delta` so the GUI can patch in place.
    pub fn rename(&self, session: &str, va: u64, name: &str) -> Result<Delta, String> {
        let params = format!(
            r#"{{"session":"{session}","va":"0x{va:x}","name":"{}"}}"#,
            escape_json(name)
        );
        let r = self.call("rename", &params)?;
        Ok(parse_delta(&r))
    }

    pub fn why(&self, session: &str, va: u64) -> Result<WhyInfo, String> {
        let params = format!(r#"{{"session":"{session}","va":"0x{va:x}","fact_id":"0x{va:x}"}}"#);
        let r = self.call("why", &params)?;
        Ok(WhyInfo {
            trust: extract_string(&r, "\"trust\"").unwrap_or_else(|| "unknown".into()),
            source: extract_string(&r, "\"source\"").unwrap_or_default(),
            text: format_why_chain(&r),
            raw: r,
        })
    }

    pub fn xrefs(&self, session: &str, va: u64) -> Result<XrefsInfo, String> {
        let params = format!(r#"{{"session":"{session}","va":"0x{va:x}"}}"#);
        let r = self.call("xrefs", &params)?;
        Ok(XrefsInfo {
            from: parse_xref_rows(&extract_array_blob(&r, "\"from\"")),
            to: parse_xref_rows(&extract_array_blob(&r, "\"to\"")),
            total: extract_number(&r, "\"total\"").unwrap_or(0),
        })
    }

    pub fn cfg(&self, session: &str, entry: u64) -> Result<CfgInfo, String> {
        let params = format!(r#"{{"session":"{session}","entry":"0x{entry:x}"}}"#);
        let r = self.call("cfg", &params)?;
        Ok(CfgInfo {
            entry,
            name: extract_string(&r, "\"name\""),
            blocks: parse_cfg_blocks(&r),
            edges: parse_cfg_edges(&r),
        })
    }

    pub fn locate(&self, session: &str, va: u64) -> Result<LocateInfo, String> {
        let params = format!(r#"{{"session":"{session}","va":"0x{va:x}"}}"#);
        let r = self.call("locate", &params)?;
        Ok(LocateInfo {
            va,
            function: extract_string(&r, "\"function\"").and_then(|s| parse_hex(&s)),
            block: extract_string(&r, "\"block\"").and_then(|s| parse_hex(&s)),
            exact_entry: r.contains(r#""exact_entry":true"#),
        })
    }

    pub fn diff(&self, session_a: &str, session_b: &str) -> Result<DiffInfo, String> {
        let params = format!(r#"{{"session_a":"{session_a}","session_b":"{session_b}"}}"#);
        let r = self.call("diff", &params)?;
        Ok(DiffInfo {
            report: extract_string(&r, "\"report\"").unwrap_or_default(),
            unchanged: extract_nested_number(&r, "\"unchanged\"").unwrap_or(0),
            moved: extract_nested_number(&r, "\"moved\"").unwrap_or(0),
            modified: extract_nested_number(&r, "\"modified\"").unwrap_or(0),
            uncertain: extract_nested_number(&r, "\"uncertain\"").unwrap_or(0),
            added: extract_nested_number(&r, "\"added\"").unwrap_or(0),
            removed: extract_nested_number(&r, "\"removed\"").unwrap_or(0),
            hunks_blob: extract_array_blob(&r, "\"hunks\""),
        })
    }

    pub fn patch_preview(&self, session: &str, va: u64) -> Result<String, String> {
        let params = format!(r#"{{"session":"{session}","va":"0x{va:x}"}}"#);
        let r = self.call("patch_preview", &params)?;
        extract_string(&r, "\"report\"").ok_or_else(|| "no patch report".into())
    }
}

#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct Health {
    pub engine_version: String,
    pub busy_jobs: u64,
}

#[derive(Clone, Default)]
pub struct OpenInfo {
    pub session_id: String,
    pub path: String,
    pub arch: String,
    pub hash: String,
    pub encrypted: bool,
}

#[derive(Clone)]
pub struct FuncRow {
    pub va: u64,
    pub name: Option<String>,
    pub source: String,
}

#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct WhyInfo {
    pub trust: String,
    pub source: String,
    pub text: String,
    pub raw: String,
}

#[derive(Clone, Debug)]
pub struct XrefRow {
    pub from: u64,
    pub to: u64,
    pub kind: String,
    pub label: Option<String>,
}

#[derive(Clone, Default)]
pub struct XrefsInfo {
    pub from: Vec<XrefRow>,
    pub to: Vec<XrefRow>,
    pub total: u64,
}

#[derive(Clone, Debug)]
pub struct CfgBlock {
    pub start: u64,
    pub end: u64,
    pub terminator: String,
    pub successors: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct CfgEdge {
    pub from: u64,
    pub to: u64,
}

#[derive(Clone, Default)]
pub struct CfgInfo {
    pub entry: u64,
    pub name: Option<String>,
    pub blocks: Vec<CfgBlock>,
    pub edges: Vec<CfgEdge>,
}

#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct LocateInfo {
    pub va: u64,
    pub function: Option<u64>,
    pub block: Option<u64>,
    pub exact_entry: bool,
}

#[derive(Clone, Default)]
pub struct Invalidate {
    pub view: String,
    pub va: u64,
}

#[derive(Clone, Default)]
pub struct Delta {
    pub kind: String,
    pub functions: Vec<FuncRow>,
    pub invalidate: Vec<Invalidate>,
}

#[derive(Clone, Default)]
pub struct DiffInfo {
    pub report: String,
    pub unchanged: u64,
    pub moved: u64,
    pub modified: u64,
    pub uncertain: u64,
    pub added: u64,
    pub removed: u64,
    pub hunks_blob: String,
}

fn parse_functions(r: &str) -> Vec<FuncRow> {
    let Some(arr_start) = r.find(r#""functions":["#) else {
        return Vec::new();
    };
    let rest = &r[arr_start + 13..];
    parse_object_array(rest)
        .into_iter()
        .filter_map(|obj| {
            let va = extract_string(&obj, "\"va\"").and_then(|s| parse_hex(&s))?;
            let name = extract_string(&obj, "\"name\"");
            let source = extract_string(&obj, "\"source\"").unwrap_or_else(|| "cfg".into());
            Some(FuncRow { va, name, source })
        })
        .collect()
}

fn parse_delta(r: &str) -> Delta {
    let Some(idx) = r.find(r#""delta":"#) else {
        return Delta::default();
    };
    let rest = &r[idx + 8..];
    let blob = extract_object_blob(rest).unwrap_or_default();
    let kind = extract_string(&blob, "\"kind\"").unwrap_or_else(|| "annotate".into());
    let functions = if let Some(i) = blob.find(r#""functions":["#) {
        parse_object_array(&blob[i + 13..])
            .into_iter()
            .filter_map(|obj| {
                let va = extract_string(&obj, "\"va\"").and_then(|s| parse_hex(&s))?;
                let name = extract_string(&obj, "\"name\"");
                let source =
                    extract_string(&obj, "\"source\"").unwrap_or_else(|| "asserted".into());
                Some(FuncRow { va, name, source })
            })
            .collect()
    } else {
        Vec::new()
    };
    let invalidate = if let Some(i) = blob.find(r#""invalidate":["#) {
        parse_object_array(&blob[i + 14..])
            .into_iter()
            .filter_map(|obj| {
                let view = extract_string(&obj, "\"view\"")?;
                let va = extract_string(&obj, "\"va\"").and_then(|s| parse_hex(&s))?;
                Some(Invalidate { view, va })
            })
            .collect()
    } else {
        Vec::new()
    };
    Delta {
        kind,
        functions,
        invalidate,
    }
}

fn parse_xref_rows(arr: &str) -> Vec<XrefRow> {
    if !arr.starts_with('[') {
        return Vec::new();
    }
    parse_object_array(&arr[1..])
        .into_iter()
        .filter_map(|obj| {
            let from = extract_string(&obj, "\"from\"").and_then(|s| parse_hex(&s))?;
            let to = extract_string(&obj, "\"to\"").and_then(|s| parse_hex(&s))?;
            let kind = extract_string(&obj, "\"kind\"").unwrap_or_else(|| "?".into());
            let label = extract_string(&obj, "\"label\"");
            Some(XrefRow {
                from,
                to,
                kind,
                label,
            })
        })
        .collect()
}

fn parse_cfg_blocks(r: &str) -> Vec<CfgBlock> {
    let Some(i) = r.find(r#""blocks":["#) else {
        return Vec::new();
    };
    parse_object_array(&r[i + 10..])
        .into_iter()
        .filter_map(|obj| {
            let start = extract_string(&obj, "\"start\"").and_then(|s| parse_hex(&s))?;
            let end = extract_string(&obj, "\"end\"").and_then(|s| parse_hex(&s))?;
            let terminator =
                extract_string(&obj, "\"terminator\"").unwrap_or_else(|| "?".into());
            let successors = extract_hex_array(&obj, "\"successors\"");
            Some(CfgBlock {
                start,
                end,
                terminator,
                successors,
            })
        })
        .collect()
}

fn parse_cfg_edges(r: &str) -> Vec<CfgEdge> {
    let Some(i) = r.find(r#""edges":["#) else {
        return Vec::new();
    };
    parse_object_array(&r[i + 9..])
        .into_iter()
        .filter_map(|obj| {
            let from = extract_string(&obj, "\"from\"").and_then(|s| parse_hex(&s))?;
            let to = extract_string(&obj, "\"to\"").and_then(|s| parse_hex(&s))?;
            Some(CfgEdge { from, to })
        })
        .collect()
}

fn parse_object_array(rest: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let bytes = rest.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    out.push(rest[start..=i].to_string());
                }
            }
            b']' if depth == 0 => break,
            _ => {}
        }
        i += 1;
    }
    out
}

fn extract_object_blob(s: &str) -> Option<String> {
    let s = s.trim_start();
    if !s.starts_with('{') {
        return None;
    }
    let mut depth = 0i32;
    for (j, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[..=j].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn extract_hex_array(s: &str, key: &str) -> Vec<u64> {
    let blob = extract_array_blob(s, key);
    if blob.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut search = blob.as_str();
    while let Some(i) = search.find("0x") {
        let rest = &search[i..];
        let end = rest
            .find(|c: char| !(c.is_ascii_hexdigit() || c == 'x' || c == 'X'))
            .unwrap_or(rest.len());
        if let Some(v) = parse_hex(&rest[..end]) {
            out.push(v);
        }
        search = &rest[end..];
    }
    out
}

fn format_why_chain(r: &str) -> String {
    if let Some(idx) = r.find(r#""chain":["#) {
        let mut lines = Vec::new();
        let rest = &r[idx..];
        // Pull lab/val pairs.
        let mut search = rest;
        while let Some(lab) = extract_string(search, "\"lab\"") {
            let after_lab = search.find("\"lab\"").map(|i| &search[i..]).unwrap_or(search);
            let val = extract_string(after_lab, "\"val\"").unwrap_or_default();
            lines.push(format!("{lab:<10} {val}"));
            if let Some(next) = after_lab.find("\"val\"") {
                search = &after_lab[next + 5..];
            } else {
                break;
            }
            if lines.len() > 12 {
                break;
            }
        }
        if !lines.is_empty() {
            return lines.join("\n");
        }
    }
    extract_string(r, "\"note\"").unwrap_or_else(|| r.to_string())
}

fn extract_array_blob(s: &str, key: &str) -> String {
    let Some(i) = s.find(key) else {
        return String::new();
    };
    let rest = &s[i + key.len()..];
    let Some(colon) = rest.find(':') else {
        return String::new();
    };
    let rest = rest[colon + 1..].trim_start();
    if !rest.starts_with('[') {
        return String::new();
    }
    let mut depth = 0i32;
    for (j, c) in rest.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return rest[..=j].to_string();
                }
            }
            _ => {}
        }
    }
    String::new()
}

fn extract_nested_number(s: &str, key: &str) -> Option<u64> {
    extract_number(s, key)
}

pub fn extract_string(line: &str, key: &str) -> Option<String> {
    let i = line.find(key)?;
    let rest = &line[i + key.len()..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    if rest == "null" || rest.starts_with("null") {
        return None;
    }
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
                    match n {
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let hex: String = chars.by_ref().take(4).collect();
                            if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                                if let Some(ch) = char::from_u32(cp) {
                                    out.push(ch);
                                }
                            }
                        }
                        other => out.push(other),
                    }
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
    let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

pub fn parse_hex(s: &str) -> Option<u64> {
    let s = s
        .trim()
        .strip_prefix("0x")
        .or_else(|| s.trim().strip_prefix("0X"))
        .unwrap_or(s.trim());
    u64::from_str_radix(s, 16).ok()
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
