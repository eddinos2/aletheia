//! Error types for the loader.

use std::fmt;

/// Errors produced while parsing a binary image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Reached end of buffer while more bytes were required.
    UnexpectedEof {
        offset: usize,
        needed: usize,
        available: usize,
    },
    /// A magic number / signature did not match the expected value.
    BadSignature {
        what: &'static str,
        offset: usize,
        found: u32,
        expected: u32,
    },
    /// A field held a value the parser cannot handle.
    Unsupported(String),
    /// An RVA could not be mapped to a file offset via any section.
    UnmappedRva(u32),
    /// A virtual address could not be mapped to a file offset via any
    /// PT_LOAD segment (ELF analogue of [`ParseError::UnmappedRva`]).
    UnmappedVaddr(u64),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedEof {
                offset,
                needed,
                available,
            } => write!(
                f,
                "unexpected end of input at offset {offset:#x}: needed {needed} bytes, {available} available"
            ),
            ParseError::BadSignature {
                what,
                offset,
                found,
                expected,
            } => write!(
                f,
                "bad {what} signature at offset {offset:#x}: found {found:#010x}, expected {expected:#010x}"
            ),
            ParseError::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            ParseError::UnmappedRva(rva) => write!(f, "RVA {rva:#010x} is not contained in any section"),
            ParseError::UnmappedVaddr(va) => write!(
                f,
                "virtual address {va:#x} is not contained in any PT_LOAD segment"
            ),
        }
    }
}

impl std::error::Error for ParseError {}

pub type Result<T> = std::result::Result<T, ParseError>;
