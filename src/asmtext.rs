//! Instruction-text rendering — the seam between flow decoding and the
//! human-readable listing.
//!
//! The [`crate::model::Decoder`] trait deliberately stops at length and
//! control-flow classification, because that is all analysis needs. A
//! listing needs more: the mnemonic and operands an analyst reads. That
//! is a separate concern with a separate trait, so the text renderers can
//! grow instruction coverage without touching the decoders that analysis
//! correctness depends on.
//!
//! [`AsmFormatter::format`] is best-effort by contract: `None` means "no
//! text for this encoding yet", and the caller falls back to raw bytes.
//! A formatter must never guess — wrong text in a listing is worse than
//! `db` bytes, because an analyst will trust it.
//!
//! Implementations: [`crate::x86_text::X86Text`] and
//! [`crate::aarch64_text::A64Text`], selected via [`formatter_for`] the
//! same way [`crate::model::decoder_for`] selects a decoder.

use crate::model::Arch;

/// Renders one instruction to listing text.
pub trait AsmFormatter {
    /// Render the instruction at the start of `bytes`, located at virtual
    /// address `va`, as `"mnemonic operands"` in Intel-style syntax.
    ///
    /// Returns `None` when the encoding is not covered; the caller falls
    /// back to raw bytes. Must never panic on any input.
    fn format(&self, bytes: &[u8], va: u64) -> Option<String>;
}

/// The text formatter for `arch`, if this crate has one.
pub fn formatter_for(arch: Arch) -> Option<&'static dyn AsmFormatter> {
    match arch {
        Arch::X86_64 => Some(&crate::x86_text::X86Text),
        Arch::Aarch64 => Some(&crate::aarch64_text::A64Text),
        Arch::Other => None,
    }
}
