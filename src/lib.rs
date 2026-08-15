//! Aletheia — a clean-room binary-analysis toolkit.
//!
//! Implemented from public format specifications (PE/COFF, ELF, Mach-O) for
//! security research, malware analysis, interoperability, and education.
//!
//! Current components:
//! - [`error`]: shared parse-error types
//! - [`reader`]: bounds-checked little-endian buffer reader
//! - [`pe`]: PE/COFF loader (headers, sections, imports/exports/delay-load)
//! - [`elf`]: ELF64 loader (headers, symbols, dynamic section, relocations)
//! - [`macho`]: Mach-O loader (thin and fat/universal images)
//! - [`objc`]: Objective-C class/method recovery from `__objc_*` sections
//! - [`swift`]: Swift `__swift5_*` inventory, types/fields, protocol conformances + witnesses
//! - [`flirt`]: open FLIRT-style CRC/pattern signature matching (text corpus)
//! - [`x86`]: x86-64 instruction decoder with flow classification
//! - [`aarch64`]: AArch64 instruction decoder with flow classification
//! - [`model`]: format- and ISA-neutral trait layer ([`Image`], [`Decoder`],
//!   the shared [`Flow`], and [`load`]) that analysis is written against
//! - [`cfg`]: control-flow recovery over the trait layer — recursive
//!   descent into functions, basic blocks, and a call graph
//! - [`funcs`]: function discovery via prologues and unwind info
//! - [`jumptable`]: switch/jump-table recovery for indirect branches
//! - [`parallel`]: deterministic function-parallel analysis engine
//! - [`strings`]: string extraction with encoding detection
//! - [`xref`]: cross-reference recovery (code->code, code->data)

pub mod aarch64;
pub mod aarch64_lift;
pub mod aarch64_text;
pub mod anchor;
pub mod annotate;
pub mod api;
pub mod asmtext;
pub mod callfx;
pub mod cfg;
pub mod cxxdemangle;
pub mod demangle;
pub mod devirt;
pub mod diff;
pub mod gopcln;
pub mod gostrings;
pub mod gotype;
pub mod ir;
pub mod irflow;
pub mod irlift;
pub mod irout;
pub mod irssa;
pub mod irssaopt;
pub mod irstruct;
pub mod irstack;
pub mod mempromote;
pub mod sig;
pub mod types;
pub mod irtype;
pub mod typebounds;
pub mod elf;
pub mod error;
// The slice-19 evaluation harness: test-only, so it can drive `irout`'s
// test-only SSA interpreter as its semantic oracle. Run: `cargo test evalfx`.
#[cfg(test)]
mod evalfx;
pub mod flirt;
pub mod funcs;
pub mod jumptable;
pub mod listing;
pub mod macho;
pub mod model;
pub mod objc;
pub mod parallel;
pub mod swift;
pub mod patch;
pub mod pe;
pub mod pseudo;
pub mod reader;
pub mod rustmeta;
pub mod strings;
pub mod vtable;
pub mod x86;
pub mod x86_lift;
pub mod x86_text;
pub mod xref;

pub use error::{ParseError, Result};
pub use model::{Arch, Decoder, Flow, Image, load};
pub use pe::PeFile;
