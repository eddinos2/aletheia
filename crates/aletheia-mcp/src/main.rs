//! Stdio transport for the Aletheia protocol (Gate M1).
//!
//! Speaks newline-delimited JSON requests — see `aletheia_mcp::handle_line`
//! and `protocol/PROTOCOL.md`. The GUI and agents share this dispatch path.

use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};

use aletheia_mcp::State;

fn main() {
    let state = Arc::new(Mutex::new(State::new()));
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let reply = aletheia_mcp::handle_line(&state, line);
        let _ = writeln!(stdout, "{reply}");
        let _ = stdout.flush();
    }
}
