//! Bounds-checked little-endian reader over a byte buffer.

use crate::error::{ParseError, Result};

/// A cursor over an immutable byte slice with checked reads.
///
/// Every read validates the remaining length and returns
/// [`ParseError::UnexpectedEof`] instead of panicking, so malformed or
/// truncated inputs are always a recoverable error.
#[derive(Debug, Clone, Copy)]
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    /// Current offset from the start of the buffer.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Number of bytes left to read.
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// Move the cursor to an absolute offset.
    pub fn seek(&mut self, offset: usize) -> Result<()> {
        if offset > self.data.len() {
            return Err(ParseError::UnexpectedEof {
                offset,
                needed: 0,
                available: 0,
            });
        }
        self.pos = offset;
        Ok(())
    }

    /// Advance the cursor by `count` bytes without reading them.
    pub fn skip(&mut self, count: usize) -> Result<()> {
        self.take(count).map(|_| ())
    }

    /// Read `count` raw bytes.
    pub fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if self.remaining() < count {
            return Err(ParseError::UnexpectedEof {
                offset: self.pos,
                needed: count,
                available: self.remaining(),
            });
        }
        let slice = &self.data[self.pos..self.pos + count];
        self.pos += count;
        Ok(slice)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// Read a NUL-terminated string, leaving the cursor after the NUL.
    ///
    /// Non-UTF-8 bytes are replaced rather than rejected: strings inside
    /// binaries are untrusted input and must never abort a parse.
    pub fn cstr(&mut self) -> Result<String> {
        let rest = &self.data[self.pos..];
        let len = rest
            .iter()
            .position(|&b| b == 0)
            .ok_or(ParseError::UnexpectedEof {
                offset: self.pos,
                needed: rest.len() + 1,
                available: rest.len(),
            })?;
        let s = String::from_utf8_lossy(&rest[..len]).into_owned();
        self.pos += len + 1;
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_scalars_in_order() {
        let mut r = Reader::new(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]);
        assert_eq!(r.u8().unwrap(), 0x01);
        assert_eq!(r.u16().unwrap(), 0x0302);
        assert_eq!(r.u32().unwrap(), 0x07060504);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn eof_reports_offset_and_shortfall() {
        let mut r = Reader::new(&[0xAA, 0xBB]);
        r.u8().unwrap();
        let err = r.u32().unwrap_err();
        assert_eq!(
            err,
            ParseError::UnexpectedEof {
                offset: 1,
                needed: 4,
                available: 1
            }
        );
    }

    #[test]
    fn seek_past_end_fails() {
        let mut r = Reader::new(&[0; 4]);
        assert!(r.seek(4).is_ok());
        assert!(r.seek(5).is_err());
    }
}
