//! `redump` — dump the structure of a PE, ELF, or Mach-O image, in the
//! spirit of `dumpbin` / `objdump -x` / `readelf`-style inspection.
//!
//! Usage: `redump <file> [--headers] [--sections] [--imports] [--symbols]
//! [--exports] [--disasm[=N]] [--listing[=N]] [--db <path>] [--diff <new>]`
//!
//! The input format is sniffed from its magic bytes: `MZ` selects the PE
//! path, `\x7fELF` the ELF path, the Mach-O magics (thin 64-bit
//! little-endian `MH_MAGIC_64`, or the big-endian fat `FAT_MAGIC` /
//! `FAT_MAGIC_64` container) the Mach-O paths; anything else is rejected.
//! With no flags, everything except the disassembly is dumped. All values
//! come from the `aletheia` library; this binary only formats them. Errors
//! are reported on stderr and exit with status 1 — malformed input never
//! panics.

use std::io::Write;
use std::process::ExitCode;

use aletheia::elf::{self, ElfFile};
use aletheia::macho::{self, CpuType, FatFile, MachFile};
use aletheia::pe::delay::DelayImportedDll;
use aletheia::pe::exports::{ExportDirectory, ExportTarget};
use aletheia::pe::{ImportSymbol, ImportedDll, Machine, PeFile, PeFormat, Section, directory_index};
use aletheia::model::Arch;
use aletheia::{
    aarch64, annotate, callfx, cfg, devirt, diff, gostrings, gotype, irlift, irout, irssa,
    irssaopt, irstruct, irstack, jumptable, listing, mempromote, patch, pseudo, rustmeta, sig,
    vtable, x86,
};

const USAGE: &str = "usage: redump <file> [--headers] [--sections] [--imports] [--symbols] [--exports] [--disasm[=N]] [--listing[=N]] [--db <path>] [--diff <new>]

  --headers     file / optional / program headers
  --sections    section tables (for Mach-O: segments with their sections)
  --imports     imported symbols. PE: import and delay-import tables; ELF:
                NEEDED libraries, PLT imports, and undefined dynamic
                symbols; Mach-O: dylib dependencies and undefined external
                symbols
  --symbols     symbol tables
  --exports     exported symbols. PE: the export table; ELF: defined
                dynamic symbols (the nearest ELF analogue of an export
                table); Mach-O: defined external symbols
  --disasm[=N]  linear-sweep disassembly of N instructions (default 32)
                starting at the program entry point (x86-64 and AArch64)
  --listing[=N] symbolized listing of the recovered program: functions,
                basic blocks, branch targets, and cross-references, for at
                most N functions (default 4096)
  --db <path>   annotation database applied to --listing, supplying the
                names, types, and comments an analyst asserted
  --diff <new>  match <file>'s functions against <new>, a second build of
                the same program: unchanged / moved / modified / uncertain
                / added / removed, with names carried across
  --vtables     recover C++ structures per the Itanium ABI: vtables,
                typeinfo objects, and the class hierarchy they encode
  --rustmeta    mine Rust core::panic::Location records and attribute
                them to functions: source-file and line hints that
                survive stripping
  --gotypes     recover Go runtime type metadata: named types with their
                kinds, and the interface-satisfaction pairs the itab
                records encode
  --devirt      resolve C++ virtual / indirect calls to candidate targets
                using the recovered vtables (exact when the class is
                established in the caller, else the sound slot superset)
  --gostrings   recover Go string literals by their (pointer, length)
                references, with exact boundaries the C-string scanner
                cannot get on packed, unterminated Go string data
  --lift[=N]    lift the recovered program to the register-transfer IR and
                print it (x86-64 and aarch64), for at most N functions
                (default 4)
  --simplify    run the IR dataflow passes (constant folding, copy
                propagation, dead-code elimination) on the lifted IR
                before printing; implies --lift
  --ssa[=N]     construct pruned SSA over the lifted IR — versioned
                definitions, phi-nodes at control-flow joins, and
                ABI-modeled call effects (clobbers, argument reads, the
                stack-pointer restore) — and print it (x86-64 and aarch64),
                for at most N functions (default 4)
  --ssa-opt[=N] the same SSA, then sparse constant/copy propagation,
                phi-simplification, expression forwarding, and
                conservative dead-code elimination across blocks: uses of
                a proven constant or copy read it directly, phi-nodes
                merging one value are gone, each definition's expression
                is forwarded into its uses (so a compare's flag plumbing
                collapses into the relational condition the branch
                tests), and definitions nothing observes are swept (never
                a store, a branch, a call effect, a load, or a value live
                at return), with the CFG intact and the per-function
                reduction printed (x86-64 and aarch64, default 4 functions)
  --structure[=N]
                run the --ssa-opt pipeline, then recover control-flow
                structure from the SSA CFG: sequences, if/else, loops
                with break and continue, switch where a jump table
                proved the dispatch, and explicit gotos for the edges no
                schema covers (x86-64 and aarch64, default 4 functions)
  --decompile[=N]
                the full pipeline: --ssa-opt, control-flow structuring,
                out-of-SSA variable naming, then C-like pseudocode with
                precedence-aware expressions, per-line address anchors,
                and honesty markers for everything the analysis could
                not prove (x86-64 and aarch64, default 4 functions)

  --stack[=N]   affine SP tracking + stack slots (DESIGN irstack), up to
                N functions (default 4)
  --promote[=N] stack-slot promotion candidates (MEM promote helper), up to
                N functions (default 4)
  --sigs[=N]    callee-side signature recovery (DESIGN sig), up to
                N functions (default 4)
  --patch-nop=<va>
                preview a same-length NOP PatchSet at VA (hex), hash-bound
  --patch-apply-nop=<va>
                apply that PatchSet to sibling <file>.patched

With no selective flag, everything except --disasm, --listing, --diff,
--vtables, --rustmeta, --gotypes, --devirt, --gostrings, --lift,
--simplify, --ssa, --ssa-opt, --structure, --decompile, --stack,
--promote, and --sigs is dumped.";

/// Instructions swept by `--disasm` when no count is given.
const DEFAULT_DISASM_COUNT: usize = 32;

/// Functions listed by `--listing` when no count is given: the library's
/// own default cap, so the flag inherits the renderer's contract rather
/// than inventing a second one.
const DEFAULT_LISTING_FUNCTIONS: usize = 4096;

/// Functions lifted to IR by `--lift` when no count is given. IR is far
/// more voluminous per function than a listing, so the default is small.
const DEFAULT_LIFT_FUNCTIONS: usize = 4;

/// Which parts of the image to print.
#[derive(Debug, Clone, Copy)]
struct Options {
    headers: bool,
    sections: bool,
    imports: bool,
    symbols: bool,
    exports: bool,
    /// `Some(n)`: disassemble up to `n` instructions from the entry point.
    disasm: Option<usize>,
    /// `Some(n)`: render the recovered program listing, up to `n`
    /// functions.
    listing: Option<usize>,
    /// Recover and print C++ vtables, typeinfo, and the class hierarchy.
    vtables: bool,
    /// Mine and print Rust panic-metadata source hints.
    rustmeta: bool,
    /// Recover and print Go type metadata and itab pairs.
    gotypes: bool,
    /// Resolve C++ virtual/indirect calls against recovered vtables.
    devirt: bool,
    /// Recover Go string literals by their (pointer, length) references.
    gostrings: bool,
    /// `Some(n)`: lift the recovered program to IR, up to `n` functions.
    lift: Option<usize>,
    /// Simplify the lifted IR through the dataflow passes before
    /// printing. Implies `lift` when that was not given itself.
    simplify: bool,
    /// `Some(n)`: construct and print pruned SSA over the lifted IR, up
    /// to `n` functions.
    ssa: Option<usize>,
    /// `Some(n)`: the same, then run the SSA optimizer over it.
    ssa_opt: Option<usize>,
    /// `Some(n)`: the optimized SSA, then recover its control-flow
    /// structure and print the tree.
    structure: Option<usize>,
    /// `Some(n)`: the full pipeline through out-of-SSA and the
    /// pseudocode renderer.
    decompile: Option<usize>,
    /// `Some(n)`: dump affine SP / stack-slot facts for up to `n` functions.
    stack: Option<usize>,
    /// `Some(n)`: dump stack-slot promotion decisions for up to `n` functions.
    promote: Option<usize>,
    /// `Some(n)`: dump callee-side signatures for up to `n` functions.
    sigs: Option<usize>,
    /// Preview a 4-byte NOP PatchSet at this VA (aarch64 NOP / x86 0x90).
    patch_nop: Option<u64>,
    /// When set with `patch_nop`, apply to sibling `*.patched`.
    patch_apply: bool,
    /// True when no selective flag was given, i.e. "dump everything".
    all: bool,
}

/// Input format, sniffed from the leading magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Pe,
    Elf,
    /// A thin Mach-O image (or a near-miss the library rejects precisely:
    /// 32-bit or big-endian variants of the Mach-O magic).
    MachO,
    /// A fat/universal Mach-O container (big-endian `CA FE BA BE/BF`).
    MachOFat,
}

/// Sniff the container format from the first bytes of the file.
///
/// The Mach-O arm matches every byte order and width of the Mach magic —
/// not just the supported `MH_MAGIC_64` little-endian form — so that the
/// library's precise "unsupported" diagnostics surface instead of a
/// generic "unrecognized format".
fn parse_hex_u64(raw: &str) -> Result<u64, String> {
    let s = raw.trim();
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    u64::from_str_radix(s, 16).map_err(|_| format!("invalid hex VA `{raw}`"))
}

fn sniff_format(data: &[u8]) -> Option<Format> {
    match data {
        [0x7F, b'E', b'L', b'F', ..] => Some(Format::Elf),
        [b'M', b'Z', ..] => Some(Format::Pe),
        // MH_MAGIC_64 / MH_MAGIC written little-endian on disk...
        [0xCF | 0xCE, 0xFA, 0xED, 0xFE, ..] => Some(Format::MachO),
        // ...or the same magics written big-endian (historical targets).
        [0xFE, 0xED, 0xFA, 0xCE | 0xCF, ..] => Some(Format::MachO),
        // FAT_MAGIC / FAT_MAGIC_64: the fat header is always big-endian...
        [0xCA, 0xFE, 0xBA, 0xBE | 0xBF, ..] => Some(Format::MachOFat),
        // ...and one written little-endian (format violation) still routes
        // to the fat parser for its precise FAT_CIGAM diagnostic.
        [0xBE | 0xBF, 0xBA, 0xFE, 0xCA, ..] => Some(Format::MachOFat),
        _ => None,
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("redump: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let (path, opts, db_path, diff_path) = parse_args()?;
    let data = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
    // The database is read and parsed before any output: a bad `--db`
    // is a usage error, and reporting it after half a dump has been
    // printed would bury it.
    let db = match &db_path {
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
            Some(annotate::Db::parse(&text).map_err(|e| format!("{p}: {e}"))?)
        }
        None => None,
    };
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    dump(&path, &data, opts, db.as_ref(), &mut out)?;
    if let Some(new_path) = diff_path {
        let new_data = std::fs::read(&new_path).map_err(|e| format!("{new_path}: {e}"))?;
        print_diff(&path, &data, &new_path, &new_data, &mut out)?;
    }
    Ok(())
}

/// Recover both programs and render the function-level diff. The old
/// side's errors are reported under the old path and likewise for the
/// new — with two inputs in play, "which file was bad" is the first
/// thing the message must answer.
fn print_diff(
    old_path: &str,
    old_data: &[u8],
    new_path: &str,
    new_data: &[u8],
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{old_path}: {e}");
    writeln!(out, "\nDIFF (against {new_path})").map_err(io)?;

    let old_image = aletheia::load(old_data).map_err(|e| format!("{old_path}: {e}"))?;
    let old_program = cfg::recover(old_image.as_ref()).map_err(|e| format!("{old_path}: {e}"))?;
    let new_image = aletheia::load(new_data).map_err(|e| format!("{new_path}: {e}"))?;
    let new_program = cfg::recover(new_image.as_ref()).map_err(|e| format!("{new_path}: {e}"))?;

    let d = diff::diff(
        old_image.as_ref(),
        &old_program,
        new_image.as_ref(),
        &new_program,
    );
    out.write_all(diff::render(&d).as_bytes()).map_err(io)
}

/// Detect the format of `data` and dump it to `out`. This is the whole
/// program minus argv and file I/O, so tests can drive it directly.
///
/// `db` supplies the analyst annotations the `--listing` section applies.
/// That listing comes last and goes through the library's format-blind
/// [`aletheia::load`] path rather than the per-format parsers above, so
/// one code path covers PE, ELF, and thin Mach-O. A fat container has no
/// single image to recover, so it gets a note instead of an error: the
/// rest of the dump is still valid output.
fn dump(
    path: &str,
    data: &[u8],
    opts: Options,
    db: Option<&annotate::Db>,
    out: &mut impl Write,
) -> Result<(), String> {
    let format = sniff_format(data);
    match format {
        Some(Format::Pe) => dump_pe(path, data, opts, out)?,
        Some(Format::Elf) => dump_elf(path, data, opts, out)?,
        Some(Format::MachO) => dump_macho(path, data, opts, out)?,
        Some(Format::MachOFat) => dump_fat(path, data, opts, out)?,
        None => {
            return Err(format!(
                "{path}: unrecognized format: expected a PE (`MZ`), ELF (`\\x7fELF`), or Mach-O \
                 (thin `MH_MAGIC_64` or fat `CAFEBABE`) magic at offset 0"
            ));
        }
    }
    if let Some(max) = opts.listing {
        print_listing(path, data, format, max, db, out)?;
    }
    if opts.vtables {
        print_vtables(path, data, format, out)?;
    }
    if opts.rustmeta {
        print_rustmeta(path, data, format, out)?;
    }
    if opts.gotypes {
        print_gotypes(path, data, format, out)?;
    }
    if opts.devirt {
        print_devirt(path, data, format, out)?;
    }
    if opts.gostrings {
        print_gostrings(path, data, format, out)?;
    }
    if let Some(max) = opts.lift {
        print_lift(path, data, format, max, opts.simplify, out)?;
    }
    if let Some(max) = opts.ssa {
        print_ssa(path, data, format, max, out)?;
    }
    if let Some(max) = opts.ssa_opt {
        print_ssa_opt(path, data, format, max, out)?;
    }
    if let Some(max) = opts.structure {
        print_structure(path, data, format, max, out)?;
    }
    if let Some(max) = opts.decompile {
        print_decompile(path, data, format, max, out)?;
    }
    if let Some(max) = opts.stack {
        print_stack(path, data, format, max, out)?;
    }
    if let Some(max) = opts.promote {
        print_promote(path, data, format, max, out)?;
    }
    if let Some(max) = opts.sigs {
        print_sigs(path, data, format, max, out)?;
    }
    if let Some(va) = opts.patch_nop {
        print_patch_nop(path, data, format, va, opts.patch_apply, out)?;
    }
    Ok(())
}


fn print_promote(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    max_functions: usize,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nPROMOTE").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice first)"
        )
        .map_err(io);
    }
    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    if !matches!(image.arch(), Arch::X86_64 | Arch::Aarch64) {
        return writeln!(out, "  (slot promotion needs x86-64 or aarch64)").map_err(io);
    }
    if let Ok(mach) = aletheia::macho::MachFile::parse(data)
        && mach.is_encrypted()
    {
        writeln!(
            out,
            "// warning: encrypted segment (cryptid!=0) — analysis may be garbage"
        )
        .map_err(io)?;
    }
    let folded =
        jumptable::resolve_folded(image.as_ref()).map_err(|e| format!("{path}: {e}"))?;
    let program = folded.program;
    for func in program.functions.values().take(max_functions) {
        let Some(lifted) = irlift::lift_function(image.as_ref(), func) else {
            continue;
        };
        let lifted = match callfx::abi_for(image.arch()) {
            Some(abi) => callfx::apply(&lifted, &abi),
            None => lifted,
        };
        let ssa = match irssa::construct(&lifted) {
            Ok(s) => s,
            Err(e) => {
                writeln!(out, "// sub_{:x}: no ssa ({e})", lifted.entry).map_err(io)?;
                continue;
            }
        };
        let (opt, _) = irssaopt::optimize(&ssa);
        let (fwd, _) = irssaopt::forward(&opt);
        let live_out = callfx::function_live_out(image.arch()).unwrap_or_default();
        let (swept, _) = irssaopt::eliminate_dead(&fwd, &live_out);
        let stack = irstack::analyze(&swept);
        if let Err(e) = irstack::check(&swept, &stack) {
            writeln!(out, "// stack check failed: {e}").map_err(io)?;
        }
        let facts = mempromote::promote(&swept, &stack);
        if let Err(e) = mempromote::check(&swept, &stack, &facts) {
            writeln!(out, "// promote check failed: {e}").map_err(io)?;
        }
        out.write_all(facts.render().as_bytes()).map_err(io)?;
    }
    Ok(())
}

fn print_sigs(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    max_functions: usize,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nSIGNATURES").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice first)"
        )
        .map_err(io);
    }
    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    if !matches!(image.arch(), Arch::X86_64 | Arch::Aarch64) {
        return writeln!(out, "  (signature recovery needs x86-64 or aarch64)").map_err(io);
    }
    if let Ok(mach) = aletheia::macho::MachFile::parse(data)
        && mach.is_encrypted()
    {
        writeln!(
            out,
            "// warning: LC_ENCRYPTION_INFO_64 cryptid!=0 — on-disk text may be ciphertext"
        )
        .map_err(io)?;
    }
    let folded =
        jumptable::resolve_folded(image.as_ref()).map_err(|e| format!("{path}: {e}"))?;
    let program = folded.program;
    for func in program.functions.values().take(max_functions) {
        let Some(lifted) = irlift::lift_function(image.as_ref(), func) else {
            continue;
        };
        let lifted = match callfx::abi_for(image.arch()) {
            Some(abi) => callfx::apply(&lifted, &abi),
            None => lifted,
        };
        let ssa = match irssa::construct(&lifted) {
            Ok(s) => s,
            Err(e) => {
                writeln!(out, "// sub_{:x}: no ssa ({e})", lifted.entry).map_err(io)?;
                continue;
            }
        };
        let (opt, _) = irssaopt::optimize(&ssa);
        let (fwd, _) = irssaopt::forward(&opt);
        let live_out = callfx::function_live_out(image.arch()).unwrap_or_default();
        let (swept, _) = irssaopt::eliminate_dead(&fwd, &live_out);
        let facts = sig::recover(&swept);
        if let Err(e) = sig::check(&swept, &facts) {
            writeln!(out, "// check failed: {e}").map_err(io)?;
        }
        out.write_all(facts.render().as_bytes()).map_err(io)?;
    }
    Ok(())
}

fn print_stack(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    max_functions: usize,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nSTACK FACTS").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice first)"
        )
        .map_err(io);
    }
    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    if !matches!(image.arch(), Arch::X86_64 | Arch::Aarch64) {
        return writeln!(out, "  (stack analysis needs x86-64 or aarch64)").map_err(io);
    }
    if let Ok(mach) = aletheia::macho::MachFile::parse(data) {
        if mach.is_encrypted() {
            writeln!(
                out,
                "// warning: LC_ENCRYPTION_INFO_64 cryptid!=0 — on-disk text may be ciphertext"
            )
            .map_err(io)?;
        }
        if let Some(cf) = &mach.chained_fixups {
            writeln!(
                out,
                "// chained fixups: {} import name(s), {} bind slot(s)",
                cf.import_names.iter().filter(|n| !n.is_empty()).count(),
                cf.bind_slots.len()
            )
            .map_err(io)?;
        }
    }
    let folded =
        jumptable::resolve_folded(image.as_ref()).map_err(|e| format!("{path}: {e}"))?;
    let program = folded.program;
    for func in program.functions.values().take(max_functions) {
        let Some(lifted) = irlift::lift_function(image.as_ref(), func) else {
            continue;
        };
        let lifted = match callfx::abi_for(image.arch()) {
            Some(abi) => callfx::apply(&lifted, &abi),
            None => lifted,
        };
        let ssa = match irssa::construct(&lifted) {
            Ok(s) => s,
            Err(e) => {
                writeln!(out, "// sub_{:x}: no ssa ({e})", lifted.entry).map_err(io)?;
                continue;
            }
        };
        let (opt, _) = irssaopt::optimize(&ssa);
        let (fwd, _) = irssaopt::forward(&opt);
        let live_out = callfx::function_live_out(image.arch()).unwrap_or_default();
        let (swept, _) = irssaopt::eliminate_dead(&fwd, &live_out);
        let facts = irstack::analyze(&swept);
        if let Err(e) = irstack::check(&swept, &facts) {
            writeln!(out, "// check failed: {e}").map_err(io)?;
        }
        out.write_all(facts.render().as_bytes()).map_err(io)?;
    }
    Ok(())
}

fn print_patch_nop(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    va: u64,
    apply: bool,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nPATCH").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(out, "  (extract a thin slice first)").map_err(io);
    }
    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    let off = image
        .va_to_offset(va)
        .ok_or_else(|| format!("{path}: VA {va:#x} unmapped"))?;
    let bytes = image.bytes();
    let len = if matches!(image.arch(), Arch::Aarch64) {
        4
    } else {
        1
    };
    if off + len > bytes.len() {
        return Err(format!("{path}: VA {va:#x} past end of file"));
    }
    let old = bytes[off..off + len].to_vec();
    let set = patch::nop_patch(image.as_ref(), va, &old, "cli --patch-nop")
        .map_err(|e| format!("{path}: {e}"))?;
    let set = if matches!(format, Some(Format::MachO)) {
        set.with_macho_resign_recipe(path)
    } else {
        set
    };
    match set.preview(image.as_ref()) {
        Ok(report) => out.write_all(report.as_bytes()).map_err(io)?,
        Err(e) => return Err(format!("{path}: patch preview: {e}")),
    }
    if apply {
        let out_path = set
            .apply_sibling(image.as_ref(), std::path::Path::new(path))
            .map_err(|e| format!("{path}: patch apply: {e}"))?;
        writeln!(out, "; wrote {}", out_path.display()).map_err(io)?;
    }
    Ok(())
}

/// The whole pipeline: the `--ssa-opt` passes, [`irstruct::structure`],
/// [`irout::out_of_ssa`], then [`pseudo::render`] — the decompiler's
/// first end-to-end output. Same gates as the other IR views: x86-64
/// only, a refused function gets an honest one-line note.
fn print_decompile(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    max_functions: usize,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nPSEUDOCODE").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice and lift that)"
        )
        .map_err(io);
    }

    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    if !matches!(image.arch(), Arch::X86_64 | Arch::Aarch64) {
        return writeln!(out, "  (IR lifting is implemented for x86-64 and aarch64 only)")
            .map_err(io);
    }
    // Recovery and table resolution run to their joint fixpoint, so the
    // proven dispatch edges are already folded into block successors and
    // an indirect jump a table proved can become a real `switch`.
    if let Ok(mach) = aletheia::macho::MachFile::parse(data)
        && mach.is_encrypted()
    {
        writeln!(
            out,
            "// warning: encrypted segment (cryptid!=0) — decompile may be garbage"
        )
        .map_err(io)?;
    }
    let folded =
        jumptable::resolve_folded(image.as_ref()).map_err(|e| format!("{path}: {e}"))?;
    if folded.capped {
        writeln!(out, "// note: table folding capped, some dispatches stay opaque")
            .map_err(io)?;
    }
    let tables = jumptable::successor_map(&folded.tables);
    let program = folded.program;

    for func in program.functions.values().take(max_functions) {
        let Some(lifted) = irlift::lift_function(image.as_ref(), func) else {
            continue;
        };
        let lifted = match callfx::abi_for(image.arch()) {
            Some(abi) => callfx::apply(&lifted, &abi),
            None => lifted,
        };
        let name = lifted
            .name
            .clone()
            .unwrap_or_else(|| format!("sub_{:x}", lifted.entry));
        let ssa = match irssa::construct(&lifted) {
            Ok(ssa) => ssa,
            Err(e) => {
                writeln!(out, "// {name} @ {:#018x}: no ssa ({e})", lifted.entry).map_err(io)?;
                continue;
            }
        };
        let (opt, _) = irssaopt::optimize(&ssa);
        let (fwd, _) = irssaopt::forward(&opt);
        let live_out = callfx::function_live_out(image.arch()).unwrap_or_default();
        let (swept, _) = irssaopt::eliminate_dead(&fwd, &live_out);
        let (root, stats) = irstruct::structure(&swept, &tables);
        if stats.capped {
            writeln!(out, "// note: structuring capped, remainder is gotos").map_err(io)?;
        }
        let (vars, _) = irout::out_of_ssa(&swept);
        out.write_all(pseudo::render(&swept, &root, &vars).as_bytes())
            .map_err(io)?;
    }
    Ok(())
}

/// Lift the recovered program's functions, apply the architecture's
/// ABI call effects ([`callfx::apply`]) so def-use links are
/// trustworthy across calls, construct pruned SSA over each, and print
/// the faithful construction. x86-64 and aarch64, like `--lift` (the
/// `--lift`/`--simplify` views stay faithful and get no call effects).
fn print_ssa(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    max_functions: usize,
    out: &mut impl Write,
) -> Result<(), String> {
    print_ssa_view(path, data, format, max_functions, false, out)
}

/// The same pipeline, then [`irssaopt::optimize`]: the cleaned SSA view.
/// `--ssa` stays the faithful construction — the honest before/after is
/// itself the proof trail — so this is a second section, not a mode of
/// the first.
fn print_ssa_opt(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    max_functions: usize,
    out: &mut impl Write,
) -> Result<(), String> {
    print_ssa_view(path, data, format, max_functions, true, out)
}

/// The `--ssa-opt` pipeline, then [`irstruct::structure`]: the recovered
/// control-flow tree. The structurer is fed the *optimized* SSA because
/// its conditions are references to the deciding blocks, and those blocks
/// read best after forwarding has collapsed the flag plumbing into the
/// relation the branch really tests.
fn print_structure(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    max_functions: usize,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nIR STRUCTURE").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice and lift that)"
        )
        .map_err(io);
    }

    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    if !matches!(image.arch(), Arch::X86_64 | Arch::Aarch64) {
        return writeln!(out, "  (IR lifting is implemented for x86-64 and aarch64 only)")
            .map_err(io);
    }
    // Recovery and table resolution run to their joint fixpoint: the
    // proven `jump_site -> targets` map is folded into block successors
    // (walking the case bodies into blocks), which is the only evidence
    // that lets an indirect jump become a `Switch` instead of `Opaque`.
    let folded =
        jumptable::resolve_folded(image.as_ref()).map_err(|e| format!("{path}: {e}"))?;
    if folded.capped {
        writeln!(out, "; note: table folding capped, some dispatches stay opaque")
            .map_err(io)?;
    }
    let tables = jumptable::successor_map(&folded.tables);
    let program = folded.program;

    for func in program.functions.values().take(max_functions) {
        let Some(lifted) = irlift::lift_function(image.as_ref(), func) else {
            continue;
        };
        let lifted = match callfx::abi_for(image.arch()) {
            Some(abi) => callfx::apply(&lifted, &abi),
            None => lifted,
        };
        let name = lifted
            .name
            .clone()
            .unwrap_or_else(|| format!("sub_{:x}", lifted.entry));
        let ssa = match irssa::construct(&lifted) {
            Ok(ssa) => ssa,
            Err(e) => {
                writeln!(out, "; {name} @ {:#018x}: no ssa ({e})", lifted.entry).map_err(io)?;
                continue;
            }
        };
        let (opt, _) = irssaopt::optimize(&ssa);
        let (fwd, _) = irssaopt::forward(&opt);
        let live_out = callfx::function_live_out(image.arch()).unwrap_or_default();
        let (swept, _) = irssaopt::eliminate_dead(&fwd, &live_out);
        let (root, stats) = irstruct::structure(&swept, &tables);
        if stats.capped {
            writeln!(out, "; note: structuring capped, remainder is gotos").map_err(io)?;
        }
        if stats.gotos > 0 {
            writeln!(out, "; structure: {} gotos", stats.gotos).map_err(io)?;
        }
        if stats.duplications > 0 {
            writeln!(out, "; structure: {} duplications", stats.duplications).map_err(io)?;
        }
        if stats.threaded > 0 {
            writeln!(out, "; structure: {} threaded", stats.threaded).map_err(io)?;
        }
        out.write_all(irstruct::render(&swept, &root).as_bytes())
            .map_err(io)?;
    }
    Ok(())
}

/// The shared body of `--ssa` and `--ssa-opt`: one pipeline, one arch
/// gate, one refusal note. A function the SSA construction refuses
/// ([`irssa::Unrepresentable`]) gets an honest one-line note rather than
/// silence, and an optimization that hit its defensive cap says so above
/// the function it prints unoptimized.
fn print_ssa_view(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    max_functions: usize,
    optimize: bool,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    let header = if optimize {
        "\nIR SSA (optimized)"
    } else {
        "\nIR SSA"
    };
    writeln!(out, "{header}").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice and lift that)"
        )
        .map_err(io);
    }

    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    if !matches!(image.arch(), Arch::X86_64 | Arch::Aarch64) {
        return writeln!(out, "  (IR lifting is implemented for x86-64 and aarch64 only)")
            .map_err(io);
    }
    // Recover through the table-folding fixpoint, so the SSA views show
    // the same dispatch edges (and reached case bodies) the structuring
    // and pseudocode views are built on.
    let program = jumptable::resolve_folded(image.as_ref())
        .map_err(|e| format!("{path}: {e}"))?
        .program;

    for func in program.functions.values().take(max_functions) {
        let Some(lifted) = irlift::lift_function(image.as_ref(), func) else {
            continue;
        };
        // The arch gate above admits exactly the arches `callfx` ships
        // tables for, so an ABI always exists here; the dispatch keeps
        // the honest fallback anyway.
        let lifted = match callfx::abi_for(image.arch()) {
            Some(abi) => callfx::apply(&lifted, &abi),
            None => lifted,
        };
        match irssa::construct(&lifted) {
            Ok(ssa) => {
                let shown = if optimize {
                    let (opt, stats) = irssaopt::optimize(&ssa);
                    if stats.capped {
                        writeln!(out, "; note: optimization capped, output unoptimized")
                            .map_err(io)?;
                    }
                    // Forwarding splices each definition's expression into
                    // its uses — the flag plumbing collapses into the
                    // relational condition the branch really tests — and
                    // leaves the emptied definitions for the sweep.
                    let (fwd, fstats) = irssaopt::forward(&opt);
                    if fstats.capped {
                        writeln!(out, "; note: forwarding capped, output partly forwarded")
                            .map_err(io)?;
                    }
                    // Propagation exposes the dead definitions; the sweep
                    // removes them. The live-out set is the ABI's, so a
                    // return value and a callee-saved restore are pinned.
                    let live_out = callfx::function_live_out(image.arch()).unwrap_or_default();
                    let total: usize = fwd.blocks.values().map(|b| b.stmts.len()).sum();
                    let (swept, dce) = irssaopt::eliminate_dead(&fwd, &live_out);
                    if dce.stmts_removed > 0 {
                        writeln!(
                            out,
                            "; dce: removed {} of {total} statements",
                            dce.stmts_removed
                        )
                        .map_err(io)?;
                    }
                    swept
                } else {
                    ssa
                };
                out.write_all(irssa::render(&shown).as_bytes())
                    .map_err(io)?;
            }
            Err(e) => {
                let name = lifted
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("sub_{:x}", lifted.entry));
                writeln!(out, "; {name} @ {:#018x}: no ssa ({e})", lifted.entry).map_err(io)?;
            }
        }
    }
    Ok(())
}

/// Lift the recovered program's functions to IR and print it. x86-64 and
/// aarch64 (aarch64 coverage is decoder-limited: an unmodeled word lifts
/// to a sound clobber intrinsic); on any other architecture this prints a
/// one-line note rather than nothing, keeping the rest of the dump valid.
fn print_lift(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    max_functions: usize,
    simplify: bool,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nIR LIFT").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice and lift that)"
        )
        .map_err(io);
    }

    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    if !matches!(image.arch(), Arch::X86_64 | Arch::Aarch64) {
        return writeln!(out, "  (IR lifting is implemented for x86-64 and aarch64 only)")
            .map_err(io);
    }
    // Recover through the table-folding fixpoint, matching the other IR
    // views: a proven dispatch's case bodies are lifted, not invisible.
    let program = jumptable::resolve_folded(image.as_ref())
        .map_err(|e| format!("{path}: {e}"))?
        .program;

    // The whole-function lift and its deterministic dump live in `irlift`;
    // this view just selects the first `max_functions` and prints them.
    for func in program.functions.values().take(max_functions) {
        if let Some(lifted) = irlift::lift_function(image.as_ref(), func) {
            let shown = if simplify {
                irlift::simplify(&lifted)
            } else {
                lifted
            };
            out.write_all(irlift::render(&shown).as_bytes()).map_err(io)?;
        }
    }
    Ok(())
}

/// Recover Go string literals and render them.
fn print_gostrings(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nGO STRINGS").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice and dump that)"
        )
        .map_err(io);
    }

    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    let strings = gostrings::recover(image.as_ref());
    out.write_all(gostrings::render(&strings).as_bytes())
        .map_err(io)
}

/// Resolve virtual/indirect calls against recovered vtables and render.
fn print_devirt(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nVIRTUAL CALLS").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice and dump that)"
        )
        .map_err(io);
    }

    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    let program = cfg::recover(image.as_ref()).map_err(|e| format!("{path}: {e}"))?;
    let sites = devirt::resolve(image.as_ref(), &program);
    out.write_all(devirt::render(&sites).as_bytes()).map_err(io)
}

/// Recover and render the Go runtime type metadata.
fn print_gotypes(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nGO TYPE METADATA").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice and dump that)"
        )
        .map_err(io);
    }

    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    let types = gotype::recover(image.as_ref());
    out.write_all(gotype::render(&types).as_bytes()).map_err(io)
}

/// Mine panic metadata and render the source-hint report.
fn print_rustmeta(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nRUST PANIC METADATA").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice and dump that)"
        )
        .map_err(io);
    }

    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    let program = cfg::recover(image.as_ref()).map_err(|e| format!("{path}: {e}"))?;
    let sites = rustmeta::mine(image.as_ref());
    let attribution = rustmeta::attribute_sites(image.as_ref(), &program, &sites);
    out.write_all(rustmeta::render(&sites, &attribution).as_bytes())
        .map_err(io)
}

/// Recover and render the C++ structures the image's RTTI encodes.
fn print_vtables(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nC++ STRUCTURES").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice and dump that)"
        )
        .map_err(io);
    }

    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    let classes = vtable::recover(image.as_ref());
    writeln!(out, "  {} class(es) recovered", classes.len()).map_err(io)?;
    out.write_all(vtable::render(&classes).as_bytes()).map_err(io)
}

/// Recover the program and render its listing.
fn print_listing(
    path: &str,
    data: &[u8],
    format: Option<Format>,
    max_functions: usize,
    db: Option<&annotate::Db>,
    out: &mut impl Write,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{path}: {e}");
    writeln!(out, "\nLISTING").map_err(io)?;
    if format == Some(Format::MachOFat) {
        return writeln!(
            out,
            "  (a fat container holds no single image; extract a slice and list that)"
        )
        .map_err(io);
    }

    let image = aletheia::load(data).map_err(|e| format!("{path}: {e}"))?;
    let program = cfg::recover(image.as_ref()).map_err(|e| format!("{path}: {e}"))?;
    let opts = listing::Options {
        max_functions,
        ..listing::Options::default()
    };
    writeln!(
        out,
        "  {} function(s) recovered, {} instruction(s) decoded",
        program.functions.len(),
        program.stats.instructions
    )
    .map_err(io)?;
    let text = listing::render(image.as_ref(), &program, db, &opts);
    out.write_all(text.as_bytes()).map_err(io)
}

/// Parse `<file>` plus the selective flags from `std::env::args`.
fn parse_args() -> Result<(String, Options, Option<String>, Option<String>), String> {
    parse_args_from(std::env::args().skip(1))
}

/// The argv parse, over any argument sequence so it is testable without
/// a process. With no selective flag every part (except the disassembly
/// and the listing) is selected.
///
/// Returns the input path, the selection, the `--db` path, and the
/// `--diff` path when each was given. The database is a *modifier*, not
/// a selection, so it does not by itself suppress the default "dump
/// everything"; `--diff` *is* a selection (its output is the point of
/// the invocation), so it does.
fn parse_args_from<I: Iterator<Item = String>>(
    args: I,
) -> Result<(String, Options, Option<String>, Option<String>), String> {
    let mut path = None;
    let mut db = None;
    let mut diff = None;
    let mut opts = Options {
        headers: false,
        sections: false,
        imports: false,
        symbols: false,
        exports: false,
        disasm: None,
        listing: None,
        vtables: false,
        rustmeta: false,
        gotypes: false,
        devirt: false,
        gostrings: false,
        lift: None,
        simplify: false,
        ssa: None,
        ssa_opt: None,
        structure: None,
            decompile: None,
            stack: None,
            promote: None,
            sigs: None,
            patch_nop: None,
            patch_apply: false,
            all: false,
        };
    let mut any_flag = false;
    let mut args = args.peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--headers" => (opts.headers, any_flag) = (true, true),
            "--sections" => (opts.sections, any_flag) = (true, true),
            "--imports" => (opts.imports, any_flag) = (true, true),
            "--symbols" => (opts.symbols, any_flag) = (true, true),
            "--exports" => (opts.exports, any_flag) = (true, true),
            "--disasm" => (opts.disasm, any_flag) = (Some(DEFAULT_DISASM_COUNT), true),
            "--listing" => {
                (opts.listing, any_flag) = (Some(DEFAULT_LISTING_FUNCTIONS), true);
            }
            "--vtables" => (opts.vtables, any_flag) = (true, true),
            "--rustmeta" => (opts.rustmeta, any_flag) = (true, true),
            "--gotypes" => (opts.gotypes, any_flag) = (true, true),
            "--devirt" => (opts.devirt, any_flag) = (true, true),
            "--gostrings" => (opts.gostrings, any_flag) = (true, true),
            "--lift" => (opts.lift, any_flag) = (Some(DEFAULT_LIFT_FUNCTIONS), true),
            "--simplify" => (opts.simplify, any_flag) = (true, true),
            "--ssa" => (opts.ssa, any_flag) = (Some(DEFAULT_LIFT_FUNCTIONS), true),
            "--ssa-opt" => (opts.ssa_opt, any_flag) = (Some(DEFAULT_LIFT_FUNCTIONS), true),
            "--structure" => (opts.structure, any_flag) = (Some(DEFAULT_LIFT_FUNCTIONS), true),
            "--decompile" => (opts.decompile, any_flag) = (Some(DEFAULT_LIFT_FUNCTIONS), true),
            "--stack" => (opts.stack, any_flag) = (Some(DEFAULT_LIFT_FUNCTIONS), true),
            "--promote" => (opts.promote, any_flag) = (Some(DEFAULT_LIFT_FUNCTIONS), true),
            "--sigs" => (opts.sigs, any_flag) = (Some(DEFAULT_LIFT_FUNCTIONS), true),
            flag if flag.starts_with("--patch-nop=") => {
                let raw = &flag["--patch-nop=".len()..];
                let va = parse_hex_u64(raw)
                    .map_err(|e| format!("--patch-nop: {e}"))?;
                opts.patch_nop = Some(va);
                any_flag = true;
            }
            flag if flag.starts_with("--patch-apply-nop=") => {
                let raw = &flag["--patch-apply-nop=".len()..];
                let va = parse_hex_u64(raw)
                    .map_err(|e| format!("--patch-apply-nop: {e}"))?;
                opts.patch_nop = Some(va);
                opts.patch_apply = true;
                any_flag = true;
            }
            "--db" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("`--db` requires a path\n{USAGE}"))?;
                db = Some(value);
            }
            "--diff" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("`--diff` requires a path\n{USAGE}"))?;
                (diff, any_flag) = (Some(value), true);
            }
            "-h" | "--help" => return Err(USAGE.to_string()),
            flag if flag.starts_with("--disasm=") => {
                let count = flag["--disasm=".len()..]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid instruction count in `{flag}`\n{USAGE}"))?;
                (opts.disasm, any_flag) = (Some(count), true);
            }
            flag if flag.starts_with("--listing=") => {
                let count = flag["--listing=".len()..]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid function count in `{flag}`\n{USAGE}"))?;
                (opts.listing, any_flag) = (Some(count), true);
            }
            flag if flag.starts_with("--lift=") => {
                let count = flag["--lift=".len()..]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid function count in `{flag}`\n{USAGE}"))?;
                (opts.lift, any_flag) = (Some(count), true);
            }
            flag if flag.starts_with("--ssa=") => {
                let count = flag["--ssa=".len()..]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid function count in `{flag}`\n{USAGE}"))?;
                (opts.ssa, any_flag) = (Some(count), true);
            }
            flag if flag.starts_with("--ssa-opt=") => {
                let count = flag["--ssa-opt=".len()..]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid function count in `{flag}`\n{USAGE}"))?;
                (opts.ssa_opt, any_flag) = (Some(count), true);
            }
            flag if flag.starts_with("--structure=") => {
                let count = flag["--structure=".len()..]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid function count in `{flag}`\n{USAGE}"))?;
                (opts.structure, any_flag) = (Some(count), true);
            }
            flag if flag.starts_with("--decompile=") => {
                let count = flag["--decompile=".len()..]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid function count in `{flag}`\n{USAGE}"))?;
                (opts.decompile, any_flag) = (Some(count), true);
            }
            flag if flag.starts_with("--stack=") => {
                let count = flag["--stack=".len()..]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid function count in `{flag}`\n{USAGE}"))?;
                (opts.stack, any_flag) = (Some(count), true);
            }
            flag if flag.starts_with("--promote=") => {
                let count = flag["--promote=".len()..]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid function count in `{flag}`\n{USAGE}"))?;
                (opts.promote, any_flag) = (Some(count), true);
            }
            flag if flag.starts_with("--sigs=") => {
                let count = flag["--sigs=".len()..]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid function count in `{flag}`\n{USAGE}"))?;
                (opts.sigs, any_flag) = (Some(count), true);
            }
            flag if flag.starts_with("--db=") => db = Some(flag["--db=".len()..].to_string()),
            flag if flag.starts_with("--diff=") => {
                (diff, any_flag) = (Some(flag["--diff=".len()..].to_string()), true);
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown option `{flag}`\n{USAGE}"));
            }
            file => {
                if path.replace(file.to_string()).is_some() {
                    return Err(format!("more than one input file given\n{USAGE}"));
                }
            }
        }
    }

    let path = path.ok_or_else(|| USAGE.to_string())?;
    // `--simplify` alone means "show me the simplified lift".
    if opts.simplify && opts.lift.is_none() {
        opts.lift = Some(DEFAULT_LIFT_FUNCTIONS);
    }
    if !any_flag {
        opts = Options {
            headers: true,
            sections: true,
            imports: true,
            symbols: true,
            exports: true,
            disasm: None,
            listing: None,
            vtables: false,
            rustmeta: false,
            gotypes: false,
            devirt: false,
            gostrings: false,
            lift: None,
            simplify: false,
            ssa: None,
            ssa_opt: None,
            structure: None,
            decompile: None,
            stack: None,
            promote: None,
            sigs: None,
            patch_nop: None,
            patch_apply: false,
            all: true,
        };
    }
    Ok((path, opts, db, diff))
}

// ---------------------------------------------------------------------------
// PE printing
// ---------------------------------------------------------------------------

/// Width of the label column in header dumps.
const LABEL: usize = 22;

/// Parsed PE payloads gathered ahead of rendering; each is present only
/// when its flag requested it.
struct PeParts {
    imports: Option<Vec<ImportedDll>>,
    delay: Option<Vec<DelayImportedDll>>,
    /// `Some(None)` when exports were requested but the image has no
    /// export directory.
    exports: Option<Option<ExportDirectory>>,
}

fn dump_pe(path: &str, data: &[u8], opts: Options, out: &mut impl Write) -> Result<(), String> {
    let pe = PeFile::parse(data).map_err(|e| format!("{path}: {e}"))?;
    let parts = PeParts {
        imports: if opts.imports {
            Some(pe.imports(data).map_err(|e| format!("{path}: imports: {e}"))?)
        } else {
            None
        },
        delay: if opts.imports {
            Some(
                pe.delay_imports(data)
                    .map_err(|e| format!("{path}: delay imports: {e}"))?,
            )
        } else {
            None
        },
        exports: if opts.exports {
            Some(pe.exports(data).map_err(|e| format!("{path}: exports: {e}"))?)
        } else {
            None
        },
    };
    render_pe(path, &pe, data, &parts, opts, out).map_err(|e| format!("{path}: {e}"))
}

fn render_pe(
    path: &str,
    pe: &PeFile,
    data: &[u8],
    parts: &PeParts,
    opts: Options,
    out: &mut impl Write,
) -> std::io::Result<()> {
    writeln!(out, "{path}: {} image", format_name(pe.optional.format))?;

    if opts.headers {
        print_file_header(pe, out)?;
        print_optional_header(pe, out)?;
        print_data_directories(pe, out)?;
    }
    if opts.sections {
        print_sections(pe, out)?;
    }
    if let Some(dlls) = &parts.imports {
        print_imports(dlls, out)?;
    }
    if let Some(dlls) = &parts.delay {
        print_delay_imports(dlls, out)?;
    }
    if let Some(dir) = &parts.exports {
        print_exports(dir.as_ref(), out)?;
    }
    // PE/COFF symbol tables are not modeled by the library; say so rather
    // than silently ignoring an explicit request. The default "dump all"
    // path skips the note to keep existing PE output unchanged.
    if opts.symbols && !opts.all {
        writeln!(out, "\nSYMBOLS")?;
        writeln!(out, "  (COFF symbol tables are not modeled for PE images)")?;
    }
    if let Some(count) = opts.disasm {
        print_pe_disasm(pe, data, count, out)?;
    }
    Ok(())
}

fn print_file_header(pe: &PeFile, out: &mut impl Write) -> std::io::Result<()> {
    let c = &pe.coff;
    writeln!(out, "\nFILE HEADER")?;
    writeln!(out, "  {:LABEL$} {}", "machine", machine_name(c.machine))?;
    writeln!(out, "  {:LABEL$} {}", "number of sections", c.number_of_sections)?;
    writeln!(
        out,
        "  {:LABEL$} {:#010x} ({})",
        "time date stamp",
        c.time_date_stamp,
        format_utc(c.time_date_stamp)
    )?;
    print_flags(
        "characteristics",
        c.characteristics as u32,
        COFF_CHARACTERISTICS,
        out,
    )
}

fn print_optional_header(pe: &PeFile, out: &mut impl Write) -> std::io::Result<()> {
    let o = &pe.optional;
    writeln!(out, "\nOPTIONAL HEADER")?;
    writeln!(out, "  {:LABEL$} {}", "format", format_name(o.format))?;
    writeln!(out, "  {:LABEL$} {:#x}", "entry point RVA", o.entry_point_rva)?;
    writeln!(out, "  {:LABEL$} {:#x}", "image base", o.image_base)?;
    writeln!(out, "  {:LABEL$} {:#x}", "section alignment", o.section_alignment)?;
    writeln!(out, "  {:LABEL$} {:#x}", "file alignment", o.file_alignment)?;
    writeln!(out, "  {:LABEL$} {:#x}", "size of image", o.size_of_image)?;
    writeln!(out, "  {:LABEL$} {:#x}", "size of headers", o.size_of_headers)?;
    writeln!(
        out,
        "  {:LABEL$} {} ({})",
        "subsystem",
        o.subsystem,
        subsystem_name(o.subsystem)
    )?;
    print_flags(
        "DLL characteristics",
        o.dll_characteristics as u32,
        DLL_CHARACTERISTICS,
        out,
    )
}

fn print_data_directories(pe: &PeFile, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nDATA DIRECTORIES")?;
    for (i, dir) in pe.optional.data_directories.iter().enumerate() {
        writeln!(
            out,
            "  [{i:2}] {:16} RVA {:#010x}  size {:#x}",
            directory_name(i),
            dir.rva,
            dir.size
        )?;
    }
    Ok(())
}

fn print_sections(pe: &PeFile, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nSECTIONS")?;
    for (i, s) in pe.sections.iter().enumerate() {
        print_section(i + 1, s, out)?;
    }
    Ok(())
}

fn print_section(index: usize, s: &Section, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(
        out,
        "  #{index} {}",
        if s.name.is_empty() { "<unnamed>" } else { &s.name }
    )?;
    writeln!(
        out,
        "     virtual size    {:<12} RVA         {:#x}",
        format!("{:#x}", s.virtual_size),
        s.virtual_address
    )?;
    writeln!(
        out,
        "     raw size        {:<12} file offset {:#x}",
        format!("{:#x}", s.size_of_raw_data),
        s.pointer_to_raw_data
    )?;
    writeln!(
        out,
        "     characteristics {:#010x}   {}",
        s.characteristics,
        flag_list(s.characteristics, SECTION_CHARACTERISTICS)
    )
}

fn print_imports(dlls: &[ImportedDll], out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nIMPORTS")?;
    if dlls.is_empty() {
        writeln!(out, "  (none)")?;
        return Ok(());
    }
    for dll in dlls {
        print_import_functions(&dll.name, &dll.functions, out)?;
    }
    Ok(())
}

/// The delay-load twin of [`print_imports`]: same shape, its own heading.
fn print_delay_imports(dlls: &[DelayImportedDll], out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nDELAY IMPORTS")?;
    if dlls.is_empty() {
        writeln!(out, "  (none)")?;
        return Ok(());
    }
    for dll in dlls {
        print_import_functions(&dll.name, &dll.functions, out)?;
    }
    Ok(())
}

/// One DLL's functions, shared between the import and delay-import dumps.
fn print_import_functions(
    dll_name: &str,
    functions: &[aletheia::pe::ImportedFunction],
    out: &mut impl Write,
) -> std::io::Result<()> {
    let n = functions.len();
    let plural = if n == 1 { "" } else { "s" };
    writeln!(out, "  {dll_name} ({n} function{plural})")?;
    for f in functions {
        match &f.symbol {
            ImportSymbol::Name { hint, name } => {
                writeln!(out, "    IAT {:#010x}  hint {hint:5}  {name}", f.iat_rva)?;
            }
            ImportSymbol::Ordinal(ord) => {
                writeln!(out, "    IAT {:#010x}  ordinal #{ord}", f.iat_rva)?;
            }
        }
    }
    Ok(())
}

fn print_exports(dir: Option<&ExportDirectory>, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nEXPORTS")?;
    let Some(dir) = dir else {
        writeln!(out, "  (no export directory)")?;
        return Ok(());
    };
    writeln!(
        out,
        "  {} (ordinal base {})",
        if dir.dll_name.is_empty() { "<unnamed>" } else { &dir.dll_name },
        dir.ordinal_base
    )?;
    if dir.exports.is_empty() {
        writeln!(out, "  (no exported symbols)")?;
        return Ok(());
    }
    writeln!(out, "   {:>7}  {:<28} target", "ordinal", "name")?;
    for e in &dir.exports {
        let name = e.name.as_deref().unwrap_or("-");
        match &e.target {
            ExportTarget::Rva(rva) => {
                writeln!(out, "   {:>7}  {name:<28} RVA {rva:#x}", e.ordinal)?;
            }
            ExportTarget::Forwarder(fwd) => {
                writeln!(out, "   {:>7}  {name:<28} -> {fwd}", e.ordinal)?;
            }
        }
    }
    Ok(())
}

fn print_pe_disasm(
    pe: &PeFile,
    data: &[u8],
    count: usize,
    out: &mut impl Write,
) -> std::io::Result<()> {
    print_disasm_heading(count, out)?;
    let arch = match pe.coff.machine {
        Machine::X86_64 => DisasmArch::X86_64,
        Machine::Arm64 => DisasmArch::Aarch64,
        other => {
            return writeln!(out, "  (no decoder for machine {})", machine_name(other));
        }
    };
    let rva = pe.optional.entry_point_rva;
    if rva == 0 {
        return writeln!(out, "  (no entry point: AddressOfEntryPoint is 0)");
    }
    let va = pe.optional.image_base.wrapping_add(rva as u64);
    match pe.rva_to_offset(rva) {
        Ok(offset) => print_disasm_listing(data, offset, va, arch, count, out),
        Err(e) => writeln!(out, "  (entry point RVA {rva:#x} is unmappable: {e})"),
    }
}

/// Print a `value (decoded flag names)` pair, one flag per line under the label.
fn print_flags(
    label: &str,
    value: u32,
    table: &[(u32, &str)],
    out: &mut impl Write,
) -> std::io::Result<()> {
    writeln!(out, "  {label:LABEL$} {value:#06x}")?;
    for &(bit, name) in table {
        if value & bit != 0 {
            writeln!(out, "  {:LABEL$}   {name}", "")?;
        }
    }
    Ok(())
}

/// Render set bits of `value` as `A | B | C` (or `-` when none are known).
fn flag_list(value: u32, table: &[(u32, &str)]) -> String {
    let names: Vec<&str> = table
        .iter()
        .filter(|&&(bit, _)| value & bit != 0)
        .map(|&(_, name)| name)
        .collect();
    if names.is_empty() {
        "-".to_string()
    } else {
        names.join(" | ")
    }
}

// ---------------------------------------------------------------------------
// ELF printing
// ---------------------------------------------------------------------------

fn dump_elf(path: &str, data: &[u8], opts: Options, out: &mut impl Write) -> Result<(), String> {
    let elf = ElfFile::parse(data).map_err(|e| format!("{path}: {e}"))?;
    render_elf(path, &elf, data, opts, out).map_err(|e| format!("{path}: {e}"))
}

fn render_elf(
    path: &str,
    elf: &ElfFile,
    data: &[u8],
    opts: Options,
    out: &mut impl Write,
) -> std::io::Result<()> {
    writeln!(out, "{path}: ELF64 image")?;

    if opts.headers {
        print_elf_header(elf, out)?;
        print_elf_program_headers(elf, out)?;
    }
    if opts.sections {
        print_elf_sections(elf, out)?;
    }
    if opts.symbols {
        print_elf_symbols(".symtab", &elf.symtab, out)?;
        print_elf_symbols(".dynsym", &elf.dynsym, out)?;
    }
    if opts.imports {
        print_elf_dynamic(elf, out)?;
        print_elf_plt_imports(elf, out)?;
        print_elf_imports(elf, out)?;
    }
    // The default dump already lists every dynamic symbol under SYMBOLS;
    // the exports view is only broken out on explicit request.
    if opts.exports && !opts.all {
        print_elf_exports(elf, out)?;
    }
    if let Some(count) = opts.disasm {
        print_elf_disasm(elf, data, count, out)?;
    }
    Ok(())
}

fn print_elf_header(elf: &ElfFile, out: &mut impl Write) -> std::io::Result<()> {
    let h = &elf.header;
    writeln!(out, "\nFILE HEADER")?;
    writeln!(out, "  {:LABEL$} ELF64, little-endian", "class")?;
    writeln!(out, "  {:LABEL$} {}", "type", elf_type_name(h.elf_type, elf.is_pie()))?;
    writeln!(out, "  {:LABEL$} {}", "machine", elf_machine_name(h.machine))?;
    writeln!(
        out,
        "  {:LABEL$} {} (ABI version {})",
        "OS/ABI",
        osabi_name(h.osabi),
        h.abi_version
    )?;
    writeln!(out, "  {:LABEL$} {:#x}", "entry point", h.entry)?;
    writeln!(
        out,
        "  {:LABEL$} {} entries at offset {:#x}",
        "program headers", h.phnum, h.phoff
    )?;
    writeln!(
        out,
        "  {:LABEL$} {} entries at offset {:#x}",
        "section headers", h.shnum, h.shoff
    )?;
    writeln!(out, "  {:LABEL$} {:#x}", "processor flags", h.flags)?;
    if let Some(interp) = elf.interpreter() {
        writeln!(out, "  {:LABEL$} {interp}", "interpreter")?;
    }
    Ok(())
}

fn print_elf_program_headers(elf: &ElfFile, out: &mut impl Write) -> std::io::Result<()> {
    let phs = &elf.program_headers;
    writeln!(out, "\nPROGRAM HEADERS ({} entries)", phs.len())?;
    if phs.is_empty() {
        writeln!(out, "  (none)")?;
        return Ok(());
    }
    writeln!(
        out,
        "   {:>2} {:<11} {:<5} {:<10} {:<12} {:<10} {:<10} align",
        "#", "type", "flags", "offset", "vaddr", "filesz", "memsz"
    )?;
    for (i, ph) in phs.iter().enumerate() {
        writeln!(
            out,
            "   {i:>2} {:<11} {:<5} {:<10} {:<12} {:<10} {:<10} {:#x}",
            segment_type_name(ph.segment_type),
            segment_perms(ph),
            format!("{:#x}", ph.offset),
            format!("{:#x}", ph.vaddr),
            format!("{:#x}", ph.filesz),
            format!("{:#x}", ph.memsz),
            ph.align
        )?;
    }
    Ok(())
}

fn print_elf_sections(elf: &ElfFile, out: &mut impl Write) -> std::io::Result<()> {
    let sections = &elf.sections;
    writeln!(out, "\nSECTION HEADERS ({} entries)", sections.len())?;
    if sections.is_empty() {
        writeln!(out, "  (none)")?;
        return Ok(());
    }
    writeln!(
        out,
        "   {:>2} {:<18} {:<11} {:<5} {:<12} {:<10} size",
        "#", "name", "type", "flags", "addr", "offset"
    )?;
    for (i, s) in sections.iter().enumerate() {
        writeln!(
            out,
            "   {i:>2} {:<18} {:<11} {:<5} {:<12} {:<10} {:#x}",
            if s.name.is_empty() { "-" } else { &s.name },
            section_type_name(s.section_type),
            section_flags(s.flags),
            format!("{:#x}", s.addr),
            format!("{:#x}", s.offset),
            s.size
        )?;
    }
    Ok(())
}

fn print_elf_symbols(
    table_name: &str,
    symbols: &[elf::Symbol],
    out: &mut impl Write,
) -> std::io::Result<()> {
    writeln!(out, "\nSYMBOLS ({table_name}: {} entries)", symbols.len())?;
    if symbols.is_empty() {
        writeln!(out, "  (none)")?;
        return Ok(());
    }
    writeln!(
        out,
        "   {:>3} {:<12} {:<8} {:<6} {:<7} {:>4}  name",
        "#", "value", "size", "bind", "type", "sec"
    )?;
    for (i, sym) in symbols.iter().enumerate() {
        let line = format!(
            "   {i:>3} {:<12} {:<8} {:<6} {:<7} {:>4}  {}",
            format!("{:#x}", sym.value),
            format!("{:#x}", sym.size),
            symbol_bind_name(sym.bind),
            symbol_kind_name(sym.kind),
            symbol_section_name(sym.shndx),
            sym.name
        );
        writeln!(out, "{}", line.trim_end())?;
    }
    Ok(())
}

/// The `DT_NEEDED` / `DT_SONAME` view of the dynamic section: what this
/// object is called and what it must be linked against.
fn print_elf_dynamic(elf: &ElfFile, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nDYNAMIC")?;
    let Some(dynamic) = &elf.dynamic else {
        writeln!(out, "  (no dynamic section)")?;
        return Ok(());
    };
    let mut any = false;
    if let Some(soname) = &dynamic.soname {
        writeln!(out, "  {:<8} {soname}", "SONAME")?;
        any = true;
    }
    for lib in &dynamic.needed {
        writeln!(out, "  {:<8} {lib}", "NEEDED")?;
        any = true;
    }
    if !any {
        writeln!(out, "  (no NEEDED or SONAME entries)")?;
    }
    Ok(())
}

/// GOT-slot-to-name pairs from the PLT relocation table: the join that
/// resolves `call [got]` / PLT calls to import names.
fn print_elf_plt_imports(elf: &ElfFile, out: &mut impl Write) -> std::io::Result<()> {
    let plt = elf.plt_imports();
    writeln!(out, "\nPLT IMPORTS ({} entries)", plt.len())?;
    if plt.is_empty() {
        writeln!(out, "  (none)")?;
        return Ok(());
    }
    writeln!(out, "   {:<14} symbol", "GOT slot")?;
    for import in plt {
        writeln!(out, "   {:<14} {}", format!("{:#x}", import.got_slot), import.name)?;
    }
    Ok(())
}

/// Format-aware `--imports` for ELF: the undefined entries of `.dynsym`
/// are the symbols the dynamic linker must resolve from elsewhere.
fn print_elf_imports(elf: &ElfFile, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nDYNAMIC IMPORTS (undefined .dynsym entries)")?;
    let mut any = false;
    for sym in &elf.dynsym {
        if sym.shndx == 0 && !sym.name.is_empty() {
            any = true;
            writeln!(
                out,
                "  {:<6} {:<7} {}",
                symbol_bind_name(sym.bind),
                symbol_kind_name(sym.kind),
                sym.name
            )?;
        }
    }
    if !any {
        writeln!(out, "  (none)")?;
    }
    Ok(())
}

/// Format-aware `--exports` for ELF: the defined entries of `.dynsym` are
/// what the dynamic linker can resolve *from* this object — the nearest
/// ELF analogue of a PE export table.
fn print_elf_exports(elf: &ElfFile, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nDYNAMIC EXPORTS (defined .dynsym entries)")?;
    let mut any = false;
    for sym in &elf.dynsym {
        if sym.shndx != 0 && !sym.name.is_empty() {
            any = true;
            writeln!(
                out,
                "  {:<12} {:<6} {:<7} {}",
                format!("{:#x}", sym.value),
                symbol_bind_name(sym.bind),
                symbol_kind_name(sym.kind),
                sym.name
            )?;
        }
    }
    if !any {
        writeln!(out, "  (none)")?;
    }
    Ok(())
}

fn print_elf_disasm(
    elf: &ElfFile,
    data: &[u8],
    count: usize,
    out: &mut impl Write,
) -> std::io::Result<()> {
    print_disasm_heading(count, out)?;
    let arch = match elf.header.machine {
        elf::Machine::X86_64 => DisasmArch::X86_64,
        elf::Machine::Aarch64 => DisasmArch::Aarch64,
        other => {
            return writeln!(out, "  (no decoder for machine {})", elf_machine_name(other));
        }
    };
    let entry = elf.header.entry;
    if entry == 0 {
        return writeln!(out, "  (no entry point: e_entry is 0)");
    }
    match elf.vaddr_to_offset(entry) {
        Ok(offset) => print_disasm_listing(data, offset, entry, arch, count, out),
        Err(e) => writeln!(out, "  (entry point {entry:#x} is unmappable: {e})"),
    }
}

// ---------------------------------------------------------------------------
// Mach-O printing
// ---------------------------------------------------------------------------

fn dump_macho(path: &str, data: &[u8], opts: Options, out: &mut impl Write) -> Result<(), String> {
    let mach = MachFile::parse(data).map_err(|e| format!("{path}: {e}"))?;
    writeln!(out, "{path}: Mach-O 64-bit image").map_err(|e| format!("{path}: {e}"))?;
    render_macho(&mach, data, opts, out).map_err(|e| format!("{path}: {e}"))
}

fn dump_fat(path: &str, data: &[u8], opts: Options, out: &mut impl Write) -> Result<(), String> {
    let fat = FatFile::parse(data).map_err(|e| format!("{path}: {e}"))?;
    render_fat(path, &fat, data, opts, out).map_err(|e| format!("{path}: {e}"))
}

fn render_fat(
    path: &str,
    fat: &FatFile,
    data: &[u8],
    opts: Options,
    out: &mut impl Write,
) -> std::io::Result<()> {
    let arches = fat.arches();
    let plural = if arches.len() == 1 { "" } else { "s" };
    writeln!(
        out,
        "{path}: Mach-O universal (fat) container, {} slice{plural}",
        arches.len()
    )?;
    writeln!(out, "   {:>2} {:<12} {:<12} {:<12} align", "#", "cputype", "offset", "size")?;
    for (i, arch) in arches.iter().enumerate() {
        writeln!(
            out,
            "   {i:>2} {:<12} {:<12} {:<12} 2^{}",
            cputype_name(arch.cputype),
            format!("{:#x}", arch.offset),
            format!("{:#x}", arch.size),
            arch.align
        )?;
    }

    // Dump every slice, clearly delimited. A slice outside the library's
    // scope (e.g. a 32-bit arch) is noted and skipped, not a fatal error:
    // the other slices are still worth dumping.
    for (i, arch) in arches.iter().enumerate() {
        writeln!(out, "\n==== slice {i}: {} ====", cputype_name(arch.cputype))?;
        // Bounds were validated by FatFile::parse against this same buffer.
        match (arch.slice(data), arch.parse(data)) {
            (Ok(slice), Ok(mach)) => render_macho(&mach, slice, opts, out)?,
            (_, Err(e)) | (Err(e), _) => writeln!(out, "  (slice not dumped: {e})")?,
        }
    }
    Ok(())
}

/// Dump one thin Mach-O image. `data` must be the bytes the image was
/// parsed from (for a fat slice: the slice, not the whole container).
fn render_macho(
    mach: &MachFile,
    data: &[u8],
    opts: Options,
    out: &mut impl Write,
) -> std::io::Result<()> {
    if opts.headers {
        print_macho_header(mach, out)?;
    }
    if opts.sections {
        print_macho_segments(mach, out)?;
    }
    if opts.symbols {
        print_macho_symbols(mach, out)?;
    }
    if opts.imports {
        print_macho_imports(mach, out)?;
    }
    // As for ELF: the default dump already lists every symbol, so the
    // exports view is only broken out on explicit request.
    if opts.exports && !opts.all {
        print_macho_exports(mach, out)?;
    }
    if let Some(count) = opts.disasm {
        print_macho_disasm(mach, data, count, out)?;
    }
    Ok(())
}

fn print_macho_header(mach: &MachFile, out: &mut impl Write) -> std::io::Result<()> {
    let h = &mach.header;
    writeln!(out, "\nFILE HEADER")?;
    writeln!(out, "  {:LABEL$} {}", "cpu type", cputype_name(h.cputype))?;
    writeln!(out, "  {:LABEL$} {:#x}", "cpu subtype", h.cpusubtype)?;
    writeln!(out, "  {:LABEL$} {}", "file type", macho_filetype_name(h.filetype))?;
    writeln!(
        out,
        "  {:LABEL$} {} ({:#x} bytes)",
        "load commands", h.ncmds, h.sizeofcmds
    )?;
    print_flags("flags", h.flags, MACH_FLAGS, out)?;
    match mach.entry_offset() {
        Some(off) => writeln!(out, "  {:LABEL$} file offset {off:#x} (LC_MAIN)", "entry point")?,
        None => writeln!(out, "  {:LABEL$} (none: no LC_MAIN command)", "entry point")?,
    }
    Ok(())
}

fn print_macho_segments(mach: &MachFile, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nSEGMENTS ({} entries)", mach.segments.len())?;
    if mach.segments.is_empty() {
        writeln!(out, "  (none)")?;
        return Ok(());
    }
    for (i, seg) in mach.segments.iter().enumerate() {
        writeln!(
            out,
            "  #{} {} ({})",
            i + 1,
            if seg.segname.is_empty() { "<unnamed>" } else { &seg.segname },
            macho_seg_perms(seg)
        )?;
        writeln!(
            out,
            "     vmaddr  {:<14} vmsize   {:#x}",
            format!("{:#x}", seg.vmaddr),
            seg.vmsize
        )?;
        writeln!(
            out,
            "     fileoff {:<14} filesize {:#x}",
            format!("{:#x}", seg.fileoff),
            seg.filesize
        )?;
        for sect in &seg.sections {
            writeln!(
                out,
                "       {:<18} addr {:<14} size {:<10} offset {:<10} align 2^{}",
                sect.sectname,
                format!("{:#x}", sect.addr),
                format!("{:#x}", sect.size),
                format!("{:#x}", sect.offset),
                sect.align
            )?;
        }
    }
    Ok(())
}

fn print_macho_symbols(mach: &MachFile, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nSYMBOLS ({} entries)", mach.symbols.len())?;
    if mach.symbols.is_empty() {
        writeln!(out, "  (none)")?;
        return Ok(());
    }
    writeln!(out, "   {:>4} {:<14} {:<6} {:<4} {:>4}  name", "#", "value", "kind", "ext", "sect")?;
    for (i, sym) in mach.symbols.iter().enumerate() {
        let line = format!(
            "   {i:>4} {:<14} {:<6} {:<4} {:>4}  {}",
            format!("{:#x}", sym.value),
            macho_symbol_kind_name(sym.kind),
            if sym.external { "yes" } else { "no" },
            sym.sect,
            sym.name
        );
        writeln!(out, "{}", line.trim_end())?;
    }
    Ok(())
}

/// Format-aware `--imports` for Mach-O: the dylibs the image links against
/// plus the undefined external symbols dyld must resolve from them.
fn print_macho_imports(mach: &MachFile, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nDYLIBS ({} entries)", mach.dylibs.len())?;
    if mach.dylibs.is_empty() {
        writeln!(out, "  (none)")?;
    }
    for dylib in &mach.dylibs {
        writeln!(
            out,
            "  {} (version {}, compat {})",
            dylib.name,
            dylib_version(dylib.current_version),
            dylib_version(dylib.compatibility_version)
        )?;
    }

    writeln!(out, "\nUNDEFINED SYMBOLS (external imports)")?;
    let mut any = false;
    for sym in &mach.symbols {
        if sym.kind == macho::SymbolKind::Undefined && sym.external && !sym.name.is_empty() {
            any = true;
            writeln!(out, "  {}", sym.name)?;
        }
    }
    if !any {
        writeln!(out, "  (none)")?;
    }
    Ok(())
}

/// Format-aware `--exports` for Mach-O: defined external symbols are what
/// other images can bind against — the Mach-O analogue of an export table.
fn print_macho_exports(mach: &MachFile, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nEXPORTS (defined external symbols)")?;
    let mut any = false;
    for sym in &mach.symbols {
        if sym.kind == macho::SymbolKind::Defined && sym.external && !sym.name.is_empty() {
            any = true;
            writeln!(out, "  {:<14} {}", format!("{:#x}", sym.value), sym.name)?;
        }
    }
    if !any {
        writeln!(out, "  (none)")?;
    }
    Ok(())
}

fn print_macho_disasm(
    mach: &MachFile,
    data: &[u8],
    count: usize,
    out: &mut impl Write,
) -> std::io::Result<()> {
    print_disasm_heading(count, out)?;
    let arch = match mach.header.cputype {
        CpuType::X86_64 => DisasmArch::X86_64,
        CpuType::Arm64 => DisasmArch::Aarch64,
        other => {
            return writeln!(out, "  (no decoder for cpu type {})", cputype_name(other));
        }
    };
    let Some(entryoff) = mach.entry_offset() else {
        return writeln!(out, "  (no entry point: no LC_MAIN command)");
    };
    // LC_MAIN's entryoff is a file offset; recover the VA it maps to from
    // the segment whose file-backed range contains it.
    let va = mach
        .segments
        .iter()
        .find(|s| s.filesize > 0 && entryoff >= s.fileoff && entryoff - s.fileoff < s.filesize)
        .map(|s| s.vmaddr.wrapping_add(entryoff - s.fileoff));
    match va {
        Some(va) => print_disasm_listing(data, entryoff as usize, va, arch, count, out),
        None => writeln!(
            out,
            "  (entry file offset {entryoff:#x} is not inside any segment's file-backed range)"
        ),
    }
}

// ---------------------------------------------------------------------------
// Disassembly (--disasm)
// ---------------------------------------------------------------------------

/// Which instruction decoder to use, derived from the image's machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DisasmArch {
    X86_64,
    Aarch64,
}

fn disasm_arch_name(arch: DisasmArch) -> &'static str {
    match arch {
        DisasmArch::X86_64 => "x86-64",
        DisasmArch::Aarch64 => "AArch64",
    }
}

fn print_disasm_heading(count: usize, out: &mut impl Write) -> std::io::Result<()> {
    writeln!(out, "\nDISASSEMBLY (up to {count} instructions from the entry point)")
}

/// Linear sweep: decode and print instructions from `offset`/`va` until
/// `count` instructions, end of file, or a decode error (reported, then
/// the sweep stops cleanly).
fn print_disasm_listing(
    data: &[u8],
    mut offset: usize,
    mut va: u64,
    arch: DisasmArch,
    count: usize,
    out: &mut impl Write,
) -> std::io::Result<()> {
    writeln!(
        out,
        "  entry {va:#x} (file offset {offset:#x}), {}",
        disasm_arch_name(arch)
    )?;
    for _ in 0..count {
        if offset >= data.len() {
            writeln!(out, "  (end of file)")?;
            break;
        }
        let bytes = &data[offset..];
        let decoded = match arch {
            DisasmArch::X86_64 => {
                x86::decode(bytes, va).map(|insn| (insn.length as usize, format_x86(&insn)))
            }
            DisasmArch::Aarch64 => aarch64::decode(bytes, va)
                .map(|insn| (aarch64::Instruction::SIZE, insn.to_string())),
        };
        let (length, text) = match decoded {
            Ok(pair) => pair,
            Err(e) => {
                writeln!(out, "  {:>12}: (sweep stopped: {e})", format!("{va:#x}"))?;
                break;
            }
        };
        let hex: Vec<String> = bytes[..length].iter().map(|b| format!("{b:02x}")).collect();
        writeln!(out, "  {:>12}: {:<21} {text}", format!("{va:#x}"), hex.join(" "))?;
        offset += length;
        va = va.wrapping_add(length as u64);
    }
    Ok(())
}

// -- x86 text rendering (the AArch64 decoder brings its own Display) --------

/// Render one decoded x86-64 instruction compactly in Intel-ish syntax.
fn format_x86(insn: &x86::Instruction) -> String {
    let mut text = String::new();
    if insn.lock {
        text.push_str("lock ");
    }
    // `endbr32/64` and the SSE opcodes are encoded behind an F3/F2 prefix
    // that is a mandatory part of the opcode, not a real `rep`/`repne`.
    let mandatory_prefix = matches!(
        insn.opcode,
        x86::Opcode::Endbr32 | x86::Opcode::Endbr64 | x86::Opcode::Sse { .. }
    );
    match insn.rep {
        Some(x86::Rep::Rep) if !mandatory_prefix => text.push_str("rep "),
        Some(x86::Rep::Repne) if !mandatory_prefix => text.push_str("repne "),
        _ => {}
    }
    text.push_str(&x86_mnemonic(insn.opcode));

    // Relative branches: show the absolute target the flow analysis
    // computed instead of the raw displacement operand.
    let operands: Vec<String> = match insn.flow {
        x86::Flow::Jump(t) | x86::Flow::CondJump(t) | x86::Flow::Call(t) => {
            vec![format!("{t:#x}")]
        }
        _ => insn
            .operands
            .iter()
            .map(|op| format_x86_operand(op, insn.segment))
            .collect(),
    };
    if !operands.is_empty() {
        text.push(' ');
        text.push_str(&operands.join(", "));
    }
    text
}

fn x86_mnemonic(op: x86::Opcode) -> String {
    use x86::Opcode::*;
    let simple = match op {
        Jcc(c) => return format!("j{}", x86_cond_suffix(c)),
        Setcc(c) => return format!("set{}", x86_cond_suffix(c)),
        Cmov(c) => return format!("cmov{}", x86_cond_suffix(c)),
        Add => "add",
        Or => "or",
        Adc => "adc",
        Sbb => "sbb",
        And => "and",
        Sub => "sub",
        Xor => "xor",
        Cmp => "cmp",
        Test => "test",
        Mov => "mov",
        Movsx => "movsx",
        Movzx => "movzx",
        Movsxd => "movsxd",
        Lea => "lea",
        Push => "push",
        Pop => "pop",
        Xchg => "xchg",
        Inc => "inc",
        Dec => "dec",
        Not => "not",
        Neg => "neg",
        Mul => "mul",
        Imul => "imul",
        Div => "div",
        Idiv => "idiv",
        Nop => "nop",
        Cwde => "cwde",
        Cdq => "cdq",
        Ret => "ret",
        Leave => "leave",
        Int3 => "int3",
        Int => "int",
        Call => "call",
        Jmp => "jmp",
        Syscall => "syscall",
        Ud2 => "ud2",
        Cpuid => "cpuid",
        Rdtsc => "rdtsc",
        Bt => "bt",
        Bts => "bts",
        Btr => "btr",
        Btc => "btc",
        Cmpxchg => "cmpxchg",
        Xadd => "xadd",
        Hlt => "hlt",
        Endbr64 => "endbr64",
        Endbr32 => "endbr32",
        Sse { mnem, .. } => mnem,
    };
    simple.to_string()
}

fn x86_cond_suffix(c: x86::Cond) -> &'static str {
    use x86::Cond::*;
    match c {
        O => "o",
        No => "no",
        B => "b",
        Ae => "ae",
        E => "e",
        Ne => "ne",
        Be => "be",
        A => "a",
        S => "s",
        Ns => "ns",
        P => "p",
        Np => "np",
        L => "l",
        Ge => "ge",
        Le => "le",
        G => "g",
    }
}

/// Signed immediate as `0x..` / `-0x..`.
fn x86_imm(v: i64) -> String {
    if v < 0 {
        format!("-{:#x}", v.unsigned_abs())
    } else {
        format!("{v:#x}")
    }
}

fn x86_segment_prefix(seg: Option<x86::Segment>) -> &'static str {
    match seg {
        None => "",
        Some(x86::Segment::Es) => "es:",
        Some(x86::Segment::Cs) => "cs:",
        Some(x86::Segment::Ss) => "ss:",
        Some(x86::Segment::Ds) => "ds:",
        Some(x86::Segment::Fs) => "fs:",
        Some(x86::Segment::Gs) => "gs:",
    }
}

fn format_x86_operand(op: &x86::Operand, seg: Option<x86::Segment>) -> String {
    match *op {
        x86::Operand::Reg(r) => r.name().to_string(),
        x86::Operand::Xmm(n) => format!("xmm{n}"),
        x86::Operand::Imm(v) => x86_imm(v),
        x86::Operand::Mem {
            base,
            index,
            scale,
            disp,
            rip_relative,
        } => {
            let mut expr = String::new();
            if rip_relative {
                expr.push_str("rip");
            }
            if let Some(b) = base {
                if !expr.is_empty() {
                    expr.push_str(" + ");
                }
                expr.push_str(b.name());
            }
            if let Some(ix) = index {
                if !expr.is_empty() {
                    expr.push_str(" + ");
                }
                expr.push_str(ix.name());
                if scale > 1 {
                    expr.push_str(&format!("*{scale}"));
                }
            }
            if expr.is_empty() {
                expr = x86_imm(disp);
            } else if disp > 0 {
                expr.push_str(&format!(" + {disp:#x}"));
            } else if disp < 0 {
                expr.push_str(&format!(" - {:#x}", disp.unsigned_abs()));
            }
            format!("{}[{expr}]", x86_segment_prefix(seg))
        }
    }
}

// ---------------------------------------------------------------------------
// ELF name tables (public gABI spec)
// ---------------------------------------------------------------------------

fn elf_type_name(t: elf::ElfType, is_pie: bool) -> String {
    match t {
        elf::ElfType::Rel => "REL (relocatable)".to_string(),
        elf::ElfType::Exec => "EXEC (executable)".to_string(),
        elf::ElfType::Dyn if is_pie => "DYN (position-independent executable)".to_string(),
        elf::ElfType::Dyn => "DYN (shared object)".to_string(),
        elf::ElfType::Core => "CORE (core dump)".to_string(),
        elf::ElfType::Other(raw) => format!("unknown ({raw:#06x})"),
    }
}

fn elf_machine_name(m: elf::Machine) -> String {
    match m {
        elf::Machine::X86_64 => "x86-64".to_string(),
        elf::Machine::Aarch64 => "AArch64".to_string(),
        elf::Machine::RiscV => "RISC-V".to_string(),
        elf::Machine::Other(raw) => format!("unknown ({raw:#06x})"),
    }
}

/// `e_ident[EI_OSABI]` names for the values seen in practice.
fn osabi_name(osabi: u8) -> String {
    match osabi {
        0 => "System V".to_string(),
        3 => "GNU/Linux".to_string(),
        6 => "Solaris".to_string(),
        9 => "FreeBSD".to_string(),
        12 => "OpenBSD".to_string(),
        other => format!("other ({other})"),
    }
}

fn segment_type_name(t: elf::SegmentType) -> String {
    match t {
        elf::SegmentType::Load => "LOAD".to_string(),
        elf::SegmentType::Dynamic => "DYNAMIC".to_string(),
        elf::SegmentType::Interp => "INTERP".to_string(),
        elf::SegmentType::Note => "NOTE".to_string(),
        elf::SegmentType::Phdr => "PHDR".to_string(),
        elf::SegmentType::Tls => "TLS".to_string(),
        elf::SegmentType::GnuStack => "GNU_STACK".to_string(),
        elf::SegmentType::GnuRelro => "GNU_RELRO".to_string(),
        elf::SegmentType::Other(raw) => format!("{raw:#010x}"),
    }
}

/// Segment permissions as an `rwx` triple.
fn segment_perms(ph: &elf::ProgramHeader) -> String {
    format!(
        "{}{}{}",
        if ph.is_read() { 'r' } else { '-' },
        if ph.is_write() { 'w' } else { '-' },
        if ph.is_execute() { 'x' } else { '-' }
    )
}

fn section_type_name(t: elf::SectionType) -> String {
    match t {
        elf::SectionType::Null => "NULL".to_string(),
        elf::SectionType::Progbits => "PROGBITS".to_string(),
        elf::SectionType::Symtab => "SYMTAB".to_string(),
        elf::SectionType::Strtab => "STRTAB".to_string(),
        elf::SectionType::Rela => "RELA".to_string(),
        elf::SectionType::Hash => "HASH".to_string(),
        elf::SectionType::Dynamic => "DYNAMIC".to_string(),
        elf::SectionType::Note => "NOTE".to_string(),
        elf::SectionType::Nobits => "NOBITS".to_string(),
        elf::SectionType::Rel => "REL".to_string(),
        elf::SectionType::Dynsym => "DYNSYM".to_string(),
        elf::SectionType::InitArray => "INIT_ARRAY".to_string(),
        elf::SectionType::FiniArray => "FINI_ARRAY".to_string(),
        elf::SectionType::Other(raw) => format!("{raw:#x}"),
    }
}

/// Section `SHF_*` flags as letters: `W` write, `A` alloc, `X` exec,
/// with `+` marking any further bits and `-` standing for none.
fn section_flags(flags: u64) -> String {
    let known = elf::SHF_WRITE | elf::SHF_ALLOC | elf::SHF_EXECINSTR;
    let mut s = String::new();
    if flags & elf::SHF_WRITE != 0 {
        s.push('W');
    }
    if flags & elf::SHF_ALLOC != 0 {
        s.push('A');
    }
    if flags & elf::SHF_EXECINSTR != 0 {
        s.push('X');
    }
    if flags & !known != 0 {
        s.push('+');
    }
    if s.is_empty() {
        s.push('-');
    }
    s
}

fn symbol_bind_name(b: elf::SymbolBind) -> String {
    match b {
        elf::SymbolBind::Local => "LOCAL".to_string(),
        elf::SymbolBind::Global => "GLOBAL".to_string(),
        elf::SymbolBind::Weak => "WEAK".to_string(),
        elf::SymbolBind::Other(raw) => format!("B{raw}"),
    }
}

fn symbol_kind_name(k: elf::SymbolKind) -> String {
    match k {
        elf::SymbolKind::Notype => "NOTYPE".to_string(),
        elf::SymbolKind::Object => "OBJECT".to_string(),
        elf::SymbolKind::Func => "FUNC".to_string(),
        elf::SymbolKind::Section => "SECTION".to_string(),
        elf::SymbolKind::File => "FILE".to_string(),
        elf::SymbolKind::Other(raw) => format!("T{raw}"),
    }
}

/// Render a symbol's `st_shndx`: the reserved indices by name, real
/// section indices as numbers.
fn symbol_section_name(shndx: u16) -> String {
    match shndx {
        0 => "UND".to_string(),      // SHN_UNDEF
        0xFFF1 => "ABS".to_string(), // SHN_ABS
        0xFFF2 => "COM".to_string(), // SHN_COMMON
        n => n.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Mach-O name tables (Apple's published loader.h / nlist.h values)
// ---------------------------------------------------------------------------

fn cputype_name(c: CpuType) -> String {
    match c {
        CpuType::X86_64 => "x86-64".to_string(),
        CpuType::Arm64 => "arm64".to_string(),
        CpuType::Other(raw) => format!("unknown ({raw:#010x})"),
    }
}

fn macho_filetype_name(t: macho::FileType) -> String {
    match t {
        macho::FileType::Object => "MH_OBJECT (relocatable object)".to_string(),
        macho::FileType::Execute => "MH_EXECUTE (executable)".to_string(),
        macho::FileType::Core => "MH_CORE (core dump)".to_string(),
        macho::FileType::Dylib => "MH_DYLIB (shared library)".to_string(),
        macho::FileType::Dylinker => "MH_DYLINKER (dynamic linker)".to_string(),
        macho::FileType::Bundle => "MH_BUNDLE (bundle)".to_string(),
        macho::FileType::Dsym => "MH_DSYM (debug symbols)".to_string(),
        macho::FileType::Other(raw) => format!("unknown ({raw:#x})"),
    }
}

fn macho_symbol_kind_name(k: macho::SymbolKind) -> String {
    match k {
        macho::SymbolKind::Undefined => "UNDEF".to_string(),
        macho::SymbolKind::Absolute => "ABS".to_string(),
        macho::SymbolKind::Defined => "SECT".to_string(),
        macho::SymbolKind::Prebound => "PBUD".to_string(),
        macho::SymbolKind::Indirect => "INDR".to_string(),
        macho::SymbolKind::Debug => "STAB".to_string(),
        macho::SymbolKind::Other(raw) => format!("T{raw:#x}"),
    }
}

/// Initial-protection bits as an `rwx` triple.
fn macho_seg_perms(seg: &macho::Segment64) -> String {
    format!(
        "{}{}{}",
        if seg.is_read() { 'r' } else { '-' },
        if seg.is_write() { 'w' } else { '-' },
        if seg.is_execute() { 'x' } else { '-' }
    )
}

/// A dylib version packed as 16.8.8 bits, rendered `x.y.z`.
fn dylib_version(v: u32) -> String {
    format!("{}.{}.{}", v >> 16, (v >> 8) & 0xFF, v & 0xFF)
}

/// `MH_*` header flags this dump names (the common subset).
const MACH_FLAGS: &[(u32, &str)] = &[
    (macho::MH_NOUNDEFS, "NOUNDEFS"),
    (macho::MH_DYLDLINK, "DYLDLINK"),
    (macho::MH_TWOLEVEL, "TWOLEVEL"),
    (macho::MH_PIE, "PIE"),
];

// ---------------------------------------------------------------------------
// PE name tables (public PE spec)
// ---------------------------------------------------------------------------

fn machine_name(m: Machine) -> String {
    match m {
        Machine::X86 => "x86".to_string(),
        Machine::X86_64 => "x86-64".to_string(),
        Machine::Arm => "ARM".to_string(),
        Machine::Arm64 => "ARM64".to_string(),
        Machine::Other(raw) => format!("unknown ({raw:#06x})"),
    }
}

fn format_name(f: PeFormat) -> &'static str {
    match f {
        PeFormat::Pe32 => "PE32",
        PeFormat::Pe32Plus => "PE32+",
    }
}

/// IMAGE_SUBSYSTEM_* names.
fn subsystem_name(subsystem: u16) -> &'static str {
    match subsystem {
        0 => "unknown",
        1 => "native",
        2 => "Windows GUI",
        3 => "Windows console",
        5 => "OS/2 console",
        7 => "POSIX console",
        8 => "native Win9x driver",
        9 => "Windows CE GUI",
        10 => "EFI application",
        11 => "EFI boot service driver",
        12 => "EFI runtime driver",
        13 => "EFI ROM image",
        14 => "Xbox",
        16 => "Windows boot application",
        _ => "unrecognized",
    }
}

/// Well-known data directory names, indexed as in [`directory_index`].
fn directory_name(index: usize) -> &'static str {
    match index {
        directory_index::EXPORT => "Export",
        directory_index::IMPORT => "Import",
        directory_index::RESOURCE => "Resource",
        directory_index::EXCEPTION => "Exception",
        4 => "Certificate",
        directory_index::BASE_RELOCATION => "Base Relocation",
        directory_index::DEBUG => "Debug",
        7 => "Architecture",
        8 => "Global Ptr",
        directory_index::TLS => "TLS",
        10 => "Load Config",
        11 => "Bound Import",
        directory_index::IAT => "IAT",
        directory_index::DELAY_IMPORT => "Delay Import",
        14 => "CLR Runtime",
        _ => "Reserved",
    }
}

/// IMAGE_FILE_* characteristics (COFF file header).
const COFF_CHARACTERISTICS: &[(u32, &str)] = &[
    (0x0001, "RELOCS_STRIPPED"),
    (0x0002, "EXECUTABLE_IMAGE"),
    (0x0004, "LINE_NUMS_STRIPPED"),
    (0x0008, "LOCAL_SYMS_STRIPPED"),
    (0x0010, "AGGRESSIVE_WS_TRIM"),
    (0x0020, "LARGE_ADDRESS_AWARE"),
    (0x0080, "BYTES_REVERSED_LO"),
    (0x0100, "32BIT_MACHINE"),
    (0x0200, "DEBUG_STRIPPED"),
    (0x0400, "REMOVABLE_RUN_FROM_SWAP"),
    (0x0800, "NET_RUN_FROM_SWAP"),
    (0x1000, "SYSTEM"),
    (0x2000, "DLL"),
    (0x4000, "UP_SYSTEM_ONLY"),
    (0x8000, "BYTES_REVERSED_HI"),
];

/// IMAGE_DLLCHARACTERISTICS_* flags (optional header).
const DLL_CHARACTERISTICS: &[(u32, &str)] = &[
    (0x0020, "HIGH_ENTROPY_VA"),
    (0x0040, "DYNAMIC_BASE"),
    (0x0080, "FORCE_INTEGRITY"),
    (0x0100, "NX_COMPAT"),
    (0x0200, "NO_ISOLATION"),
    (0x0400, "NO_SEH"),
    (0x0800, "NO_BIND"),
    (0x1000, "APPCONTAINER"),
    (0x2000, "WDM_DRIVER"),
    (0x4000, "GUARD_CF"),
    (0x8000, "TERMINAL_SERVER_AWARE"),
];

/// IMAGE_SCN_* section characteristics.
const SECTION_CHARACTERISTICS: &[(u32, &str)] = &[
    (0x0000_0020, "CODE"),
    (0x0000_0040, "INITIALIZED_DATA"),
    (0x0000_0080, "UNINITIALIZED_DATA"),
    (0x0100_0000, "LNK_NRELOC_OVFL"),
    (0x0200_0000, "MEM_DISCARDABLE"),
    (0x0400_0000, "MEM_NOT_CACHED"),
    (0x0800_0000, "MEM_NOT_PAGED"),
    (0x1000_0000, "MEM_SHARED"),
    (0x2000_0000, "MEM_EXECUTE"),
    (0x4000_0000, "MEM_READ"),
    (0x8000_0000, "MEM_WRITE"),
];

// ---------------------------------------------------------------------------
// Timestamp formatting
// ---------------------------------------------------------------------------

/// Format a COFF `TimeDateStamp` (seconds since the Unix epoch, UTC) as
/// `YYYY-MM-DD HH:MM:SS UTC`, without external date crates.
///
/// Days-to-civil conversion follows the public-domain algorithm from Howard
/// Hinnant's "chrono-Compatible Low-Level Date Algorithms".
fn format_utc(timestamp: u32) -> String {
    let secs = timestamp as u64;
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (hour, min, sec) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    // Shift epoch from 1970-01-01 to 0000-03-01 so leap days land at
    // year-end of the era arithmetic.
    let days = days + 719_468;
    let era = days / 146_097;
    let doe = days % 146_097; // day of 400-year era
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year, Mar 1 = 0
    let mp = (5 * doy + 2) / 153; // month index, Mar = 0
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    format!("{year:04}-{month:02}-{day:02} {hour:02}:{min:02}:{sec:02} UTC")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn opts_all() -> Options {
        Options {
            headers: true,
            sections: true,
            imports: true,
            symbols: true,
            exports: true,
            disasm: None,
            listing: None,
            vtables: false,
            rustmeta: false,
            gotypes: false,
            devirt: false,
            gostrings: false,
            lift: None,
            simplify: false,
            ssa: None,
            ssa_opt: None,
            structure: None,
            decompile: None,
            stack: None,
            promote: None,
            sigs: None,
            patch_nop: None,
            patch_apply: false,
            all: true,
        }
    }

    fn opts_none() -> Options {
        Options {
            headers: false,
            sections: false,
            imports: false,
            symbols: false,
            exports: false,
            disasm: None,
            listing: None,
            vtables: false,
            rustmeta: false,
            gotypes: false,
            devirt: false,
            gostrings: false,
            lift: None,
            simplify: false,
            ssa: None,
            ssa_opt: None,
            structure: None,
            decompile: None,
            stack: None,
            promote: None,
            sigs: None,
            patch_nop: None,
            patch_apply: false,
            all: false,
        }
    }

    fn opts_only(headers: bool, sections: bool, imports: bool, symbols: bool) -> Options {
        Options {
            headers,
            sections,
            imports,
            symbols,
            ..opts_none()
        }
    }

    fn opts_exports() -> Options {
        Options {
            exports: true,
            ..opts_none()
        }
    }

    fn opts_disasm(count: usize) -> Options {
        Options {
            disasm: Some(count),
            ..opts_none()
        }
    }

    fn opts_listing(max_functions: usize) -> Options {
        Options {
            listing: Some(max_functions),
            ..opts_none()
        }
    }

    /// Run the full dump pipeline into a String.
    fn dump_to_string(data: &[u8], opts: Options) -> Result<String, String> {
        dump_to_string_with(data, opts, None)
    }

    /// [`dump_to_string`] with an annotation database.
    fn dump_to_string_with(
        data: &[u8],
        opts: Options,
        db: Option<&annotate::Db>,
    ) -> Result<String, String> {
        let mut buf = Vec::new();
        dump("test.bin", data, opts, db, &mut buf)?;
        Ok(String::from_utf8(buf).expect("dump output is UTF-8"))
    }

    // -- synthetic fixtures --------------------------------------------

    fn put(img: &mut [u8], off: usize, bytes: &[u8]) {
        img[off..off + bytes.len()].copy_from_slice(bytes);
    }

    fn put16(img: &mut [u8], off: usize, v: u16) {
        put(img, off, &v.to_le_bytes());
    }

    fn put32(img: &mut [u8], off: usize, v: u32) {
        put(img, off, &v.to_le_bytes());
    }

    fn put64(img: &mut [u8], off: usize, v: u64) {
        put(img, off, &v.to_le_bytes());
    }

    fn put32be(img: &mut [u8], off: usize, v: u32) {
        put(img, off, &v.to_be_bytes());
    }

    /// A minimal well-formed PE32+ image with one `.text` section and an
    /// empty (zero-terminated) import directory.
    fn synthetic_pe64() -> Vec<u8> {
        let e_lfanew: u32 = 0x80;
        let mut img = vec![0u8; 0x400];

        // DOS header.
        put16(&mut img, 0, 0x5A4D); // "MZ"
        put32(&mut img, 0x3C, e_lfanew);

        let pe = e_lfanew as usize;
        put32(&mut img, pe, 0x0000_4550); // "PE\0\0"

        // COFF header.
        let coff = pe + 4;
        put16(&mut img, coff, 0x8664); // machine: x86-64
        put16(&mut img, coff + 2, 1); // 1 section
        put16(&mut img, coff + 16, 0xF0); // size of optional header (PE32+, 16 dirs)
        put16(&mut img, coff + 18, 0x0022); // exe | large-address-aware

        // Optional header (PE32+).
        let opt = coff + 20;
        put16(&mut img, opt, 0x20B); // PE32+ magic
        put32(&mut img, opt + 16, 0x1000); // entry point RVA
        put64(&mut img, opt + 24, 0x1_4000_0000); // image base
        put32(&mut img, opt + 32, 0x1000); // section alignment
        put32(&mut img, opt + 36, 0x200); // file alignment
        put32(&mut img, opt + 56, 0x2000); // size of image
        put32(&mut img, opt + 60, 0x200); // size of headers
        put16(&mut img, opt + 68, 3); // subsystem: console
        put32(&mut img, opt + 108, 16); // NumberOfRvaAndSizes
        // Import directory (index 1) points at zeroed .text bytes, so the
        // first descriptor is the all-zero terminator: zero imports.
        let dirs = opt + 112;
        put32(&mut img, dirs + 8, 0x1010);
        put32(&mut img, dirs + 12, 0x40);

        // Section table: one ".text" section, RVA 0x1000 -> file 0x200.
        let sec = opt + 0xF0;
        put(&mut img, sec, b".text");
        put32(&mut img, sec + 8, 0x100); // virtual size
        put32(&mut img, sec + 12, 0x1000); // virtual address
        put32(&mut img, sec + 16, 0x200); // size of raw data
        put32(&mut img, sec + 20, 0x200); // pointer to raw data
        put32(&mut img, sec + 36, 0x6000_0020); // code | exec | read

        img
    }

    /// File offset of the PE32+ data directory table in `synthetic_pe64`.
    const PE_DIRS: usize = 0x80 + 4 + 20 + 112;
    /// File offset backing RVA 0x1000 (start of .text) in `synthetic_pe64`.
    const PE_TEXT: usize = 0x200;

    /// `synthetic_pe64` with an export directory laid into .text: a named
    /// export, an unused EAT slot, an ordinal-only export, and a forwarder.
    fn synthetic_pe64_with_exports() -> Vec<u8> {
        let mut img = synthetic_pe64();
        let p32 = |img: &mut [u8], off: usize, v: u32| put32(img, PE_TEXT + off, v);
        let p16 = |img: &mut [u8], off: usize, v: u16| put16(img, PE_TEXT + off, v);

        // Zero the import directory: its RVA would otherwise land inside
        // the export data and read back as a garbage descriptor.
        put32(&mut img, PE_DIRS + 8, 0);
        put32(&mut img, PE_DIRS + 12, 0);

        // IMAGE_EXPORT_DIRECTORY at RVA 0x1000.
        p32(&mut img, 0x0C, 0x10A0); // Name -> "TOOLKIT.dll"
        p32(&mut img, 0x10, 5); // Base (ordinal base)
        p32(&mut img, 0x14, 4); // NumberOfFunctions
        p32(&mut img, 0x18, 2); // NumberOfNames
        p32(&mut img, 0x1C, 0x1030); // AddressOfFunctions
        p32(&mut img, 0x20, 0x1050); // AddressOfNames
        p32(&mut img, 0x24, 0x1060); // AddressOfNameOrdinals

        // EAT: code RVA / unused slot / code RVA / forwarder.
        p32(&mut img, 0x30, 0x1111);
        p32(&mut img, 0x34, 0);
        p32(&mut img, 0x38, 0x2222);
        p32(&mut img, 0x3C, 0x10C0); // inside the directory range

        // Name pointer table and parallel ordinal (EAT index) table.
        p32(&mut img, 0x50, 0x1090); // "Alpha" -> EAT[0]
        p32(&mut img, 0x54, 0x1098); // "Beta"  -> EAT[3]
        p16(&mut img, 0x60, 0);
        p16(&mut img, 0x62, 3);

        put(&mut img, PE_TEXT + 0x90, b"Alpha\0");
        put(&mut img, PE_TEXT + 0x98, b"Beta\0");
        put(&mut img, PE_TEXT + 0xA0, b"TOOLKIT.dll\0");
        put(&mut img, PE_TEXT + 0xC0, b"OTHER.Func\0");

        // Export data directory (index 0): RVA 0x1000, size 0x100.
        put32(&mut img, PE_DIRS, 0x1000);
        put32(&mut img, PE_DIRS + 4, 0x100);
        img
    }

    /// `synthetic_pe64` with one delay-load descriptor (USER32.dll, one
    /// by-name and one by-ordinal entry) laid into .text.
    fn synthetic_pe64_with_delay() -> Vec<u8> {
        let mut img = synthetic_pe64();
        let p32 = |img: &mut [u8], off: usize, v: u32| put32(img, PE_TEXT + off, v);
        let p64 = |img: &mut [u8], off: usize, v: u64| put64(img, PE_TEXT + off, v);

        // Zero the regular import directory (it points at bytes now used
        // by the delay structures).
        put32(&mut img, PE_DIRS + 8, 0);
        put32(&mut img, PE_DIRS + 12, 0);

        // IMAGE_DELAYLOAD_DESCRIPTOR 0 at RVA 0x1000; descriptor 1 is
        // already all zeros: the terminator.
        p32(&mut img, 0x00, 1); // Attributes: RvaBased
        p32(&mut img, 0x04, 0x1080); // DllNameRVA
        p32(&mut img, 0x08, 0x10F0); // ModuleHandleRVA
        p32(&mut img, 0x0C, 0x1060); // ImportAddressTableRVA
        p32(&mut img, 0x10, 0x1040); // ImportNameTableRVA

        // Delay INT at RVA 0x1040 (PE32+, 8-byte thunks); IAT mirrors it.
        p64(&mut img, 0x40, 0x1090);
        p64(&mut img, 0x48, (1u64 << 63) | 7);
        p64(&mut img, 0x60, 0x1090);
        p64(&mut img, 0x68, (1u64 << 63) | 7);

        // DLL name and hint/name entry.
        put(&mut img, PE_TEXT + 0x80, b"USER32.dll\0");
        put16(&mut img, PE_TEXT + 0x90, 0x0010);
        put(&mut img, PE_TEXT + 0x92, b"MessageBoxA\0");

        // Delay-load data directory (index 13).
        let entry = PE_DIRS + directory_index::DELAY_IMPORT * 8;
        put32(&mut img, entry, 0x1000);
        put32(&mut img, entry + 4, 0x40);
        img
    }

    const ELF_PHOFF: usize = 0x40;
    const ELF_INTERP_OFF: usize = 0xB0;
    const ELF_TEXT_OFF: usize = 0x100;
    const ELF_TEXT_VADDR: u64 = 0x40_1000;
    const ELF_STRTAB_OFF: usize = 0x200;
    const ELF_SYMTAB_OFF: usize = 0x220;
    const ELF_SHSTRTAB_OFF: usize = 0x280;
    const ELF_SHOFF: usize = 0x300;
    const ELF_INTERP: &[u8] = b"/lib64/ld-linux-x86-64.so.2\0";

    #[allow(clippy::too_many_arguments)]
    fn write_phdr_at(
        img: &mut [u8],
        phoff: usize,
        idx: usize,
        ptype: u32,
        flags: u32,
        offset: u64,
        vaddr: u64,
        filesz: u64,
        memsz: u64,
        align: u64,
    ) {
        let base = phoff + idx * 56;
        put32(img, base, ptype);
        put32(img, base + 4, flags);
        put64(img, base + 8, offset);
        put64(img, base + 16, vaddr);
        put64(img, base + 24, vaddr); // paddr
        put64(img, base + 32, filesz);
        put64(img, base + 40, memsz);
        put64(img, base + 48, align);
    }

    #[allow(clippy::too_many_arguments)]
    fn write_phdr(
        img: &mut [u8],
        idx: usize,
        ptype: u32,
        flags: u32,
        offset: u64,
        vaddr: u64,
        filesz: u64,
        memsz: u64,
        align: u64,
    ) {
        write_phdr_at(img, ELF_PHOFF, idx, ptype, flags, offset, vaddr, filesz, memsz, align);
    }

    #[allow(clippy::too_many_arguments)]
    fn write_shdr(
        img: &mut [u8],
        idx: usize,
        name_off: u32,
        shtype: u32,
        flags: u64,
        addr: u64,
        offset: u64,
        size: u64,
        link: u32,
        entsize: u64,
    ) {
        let base = ELF_SHOFF + idx * 64;
        put32(img, base, name_off);
        put32(img, base + 4, shtype);
        put64(img, base + 8, flags);
        put64(img, base + 16, addr);
        put64(img, base + 24, offset);
        put64(img, base + 32, size);
        put32(img, base + 40, link);
        put64(img, base + 48, 16); // addralign
        put64(img, base + 56, entsize);
    }

    fn write_sym_at(
        img: &mut [u8],
        table_off: usize,
        idx: usize,
        name_off: u32,
        info: u8,
        shndx: u16,
        value: u64,
    ) {
        let base = table_off + idx * 24;
        put32(img, base, name_off);
        img[base + 4] = info;
        put16(img, base + 6, shndx);
        put64(img, base + 8, value);
        put64(img, base + 16, 0x20); // size
    }

    fn write_sym(img: &mut [u8], idx: usize, name_off: u32, info: u8, shndx: u16, value: u64) {
        write_sym_at(img, ELF_SYMTAB_OFF, idx, name_off, info, shndx, value);
    }

    /// A minimal well-formed ELF64 PIE (little-endian, x86-64, ET_DYN with
    /// PT_INTERP): PT_LOAD + PT_INTERP program headers, and five sections
    /// (null, .text, .dynsym, .strtab, .shstrtab). The .dynsym table holds
    /// a defined `main` and an undefined `printf` (a dynamic import).
    fn synthetic_elf64() -> Vec<u8> {
        let mut img = vec![0u8; 0x440];

        // ELF header.
        put(&mut img, 0, b"\x7fELF");
        img[4] = 2; // ELFCLASS64
        img[5] = 1; // ELFDATA2LSB
        img[6] = 1; // EI_VERSION
        put16(&mut img, 16, 3); // e_type: ET_DYN
        put16(&mut img, 18, 62); // e_machine: EM_X86_64
        put32(&mut img, 20, 1); // e_version
        put64(&mut img, 24, ELF_TEXT_VADDR); // e_entry
        put64(&mut img, 32, ELF_PHOFF as u64); // e_phoff
        put64(&mut img, 40, ELF_SHOFF as u64); // e_shoff
        put16(&mut img, 52, 64); // e_ehsize
        put16(&mut img, 54, 56); // e_phentsize
        put16(&mut img, 56, 2); // e_phnum
        put16(&mut img, 58, 64); // e_shentsize
        put16(&mut img, 60, 5); // e_shnum
        put16(&mut img, 62, 4); // e_shstrndx: .shstrtab

        // Program headers: an R+X PT_LOAD and the PT_INTERP.
        write_phdr(
            &mut img,
            0,
            1, // PT_LOAD
            elf::PF_R | elf::PF_X,
            ELF_TEXT_OFF as u64,
            ELF_TEXT_VADDR,
            0x80,
            0x80,
            0x1000,
        );
        write_phdr(
            &mut img,
            1,
            3, // PT_INTERP
            elf::PF_R,
            ELF_INTERP_OFF as u64,
            0x40_00B0,
            ELF_INTERP.len() as u64,
            ELF_INTERP.len() as u64,
            1,
        );
        put(&mut img, ELF_INTERP_OFF, ELF_INTERP);

        // String tables.
        put(&mut img, ELF_STRTAB_OFF, b"\0main\0printf\0"); // 13 bytes
        put(
            &mut img,
            ELF_SHSTRTAB_OFF,
            b"\0.text\0.dynsym\0.strtab\0.shstrtab\0", // 33 bytes
        );

        // Symbols: null, then main (global func, in .text), then printf
        // (global func, undefined: a dynamic import).
        write_sym(&mut img, 1, 1, 0x12, 1, ELF_TEXT_VADDR);
        write_sym(&mut img, 2, 6, 0x12, 0, 0);

        // Sections: null, .text, .dynsym, .strtab, .shstrtab.
        write_shdr(&mut img, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        write_shdr(
            &mut img,
            1,
            1, // ".text"
            1, // SHT_PROGBITS
            elf::SHF_ALLOC | elf::SHF_EXECINSTR,
            ELF_TEXT_VADDR,
            ELF_TEXT_OFF as u64,
            0x80,
            0,
            0,
        );
        write_shdr(
            &mut img,
            2,
            7,  // ".dynsym"
            11, // SHT_DYNSYM
            elf::SHF_ALLOC,
            0,
            ELF_SYMTAB_OFF as u64,
            3 * 24,
            3, // link -> .strtab
            24,
        );
        write_shdr(&mut img, 3, 15, 3, 0, 0, ELF_STRTAB_OFF as u64, 13, 0, 0);
        write_shdr(&mut img, 4, 23, 3, 0, 0, ELF_SHSTRTAB_OFF as u64, 33, 0, 0);

        img
    }

    // Layout of the synthetic *dynamic* ELF image (adapted from the
    // library's own dynamic-section fixture). One PT_LOAD maps the whole
    // file at DYN_BASE, so vaddr == DYN_BASE + offset.
    const DYN_BASE: u64 = 0x40_0000;
    const DYN_IMG_SIZE: usize = 0x300;
    const DYN_STR_OFF: usize = 0xC0;
    const DYN_SYM_OFF: usize = 0x100;
    const DYN_JMPREL_OFF: usize = 0x160;
    const DYN_DYN_OFF: usize = 0x200;
    // Name offsets: 1 "libc.so.6", 11 "libm.so.6", 21 "read",
    // 26 "write", 32 "close", 38 "mylib.so".
    const DYNSTR: &[u8] = b"\0libc.so.6\0libm.so.6\0read\0write\0close\0mylib.so\0";

    /// A synthetic dynamically linked ELF64 (x86-64, ET_DYN): PT_LOAD over
    /// the whole file plus PT_DYNAMIC; .dynstr; three undefined dynamic
    /// symbols (read/write/close); a three-entry RELA JMPREL table of
    /// JUMP_SLOT relocations; DT_NEEDED x2 and a DT_SONAME.
    fn synthetic_dynamic_elf64() -> Vec<u8> {
        let mut img = vec![0u8; DYN_IMG_SIZE];

        // ELF header (no section headers; everything via program headers).
        put(&mut img, 0, b"\x7fELF");
        img[4] = 2; // ELFCLASS64
        img[5] = 1; // ELFDATA2LSB
        img[6] = 1; // EI_VERSION
        put16(&mut img, 16, 3); // e_type: ET_DYN
        put16(&mut img, 18, 62); // e_machine: EM_X86_64
        put32(&mut img, 20, 1); // e_version
        put64(&mut img, 32, ELF_PHOFF as u64); // e_phoff
        put16(&mut img, 52, 64); // e_ehsize
        put16(&mut img, 54, 56); // e_phentsize
        put16(&mut img, 56, 2); // e_phnum
        put16(&mut img, 58, 64); // e_shentsize

        // Program headers: PT_LOAD over the whole file, then PT_DYNAMIC.
        write_phdr(
            &mut img,
            0,
            1, // PT_LOAD
            elf::PF_R | elf::PF_X,
            0,
            DYN_BASE,
            DYN_IMG_SIZE as u64,
            DYN_IMG_SIZE as u64,
            0x1000,
        );
        write_phdr(
            &mut img,
            1,
            2, // PT_DYNAMIC
            elf::PF_R,
            DYN_DYN_OFF as u64,
            DYN_BASE + DYN_DYN_OFF as u64,
            0xF0,
            0xF0,
            8,
        );

        put(&mut img, DYN_STR_OFF, DYNSTR);

        // .dynsym: null, then read/write/close as undefined global funcs.
        write_sym_at(&mut img, DYN_SYM_OFF, 1, 21, 0x12, 0, 0);
        write_sym_at(&mut img, DYN_SYM_OFF, 2, 26, 0x12, 0, 0);
        write_sym_at(&mut img, DYN_SYM_OFF, 3, 32, 0x12, 0, 0);

        // JMPREL: three R_X86_64_JUMP_SLOT relocs against read/write/close.
        for i in 0..3u64 {
            let base = DYN_JMPREL_OFF + i as usize * 24;
            put64(&mut img, base, DYN_BASE + 0x3018 + 8 * i); // r_offset (GOT slot)
            put64(&mut img, base + 8, ((i + 1) << 32) | elf::R_X86_64_JUMP_SLOT as u64);
        }

        // Dynamic table.
        let entries: &[(u64, u64)] = &[
            (elf::DT_NEEDED, 1),  // "libc.so.6"
            (elf::DT_NEEDED, 11), // "libm.so.6"
            (elf::DT_SONAME, 38), // "mylib.so"
            (elf::DT_STRTAB, DYN_BASE + DYN_STR_OFF as u64),
            (elf::DT_STRSZ, DYNSTR.len() as u64),
            (elf::DT_SYMTAB, DYN_BASE + DYN_SYM_OFF as u64),
            (elf::DT_SYMENT, 24),
            (elf::DT_JMPREL, DYN_BASE + DYN_JMPREL_OFF as u64),
            (elf::DT_PLTRELSZ, 3 * 24),
            (elf::DT_PLTREL, elf::DT_RELA),
            (elf::DT_NULL, 0),
        ];
        for (i, &(tag, value)) in entries.iter().enumerate() {
            let base = DYN_DYN_OFF + i * 16;
            put64(&mut img, base, tag);
            put64(&mut img, base + 8, value);
        }

        img
    }

    // -- Mach-O fixtures (adapted from the library's macho test builder) --

    const MACH_IMG_SIZE: usize = 0x300;
    const MACH_TEXT_VMADDR: u64 = 0x1_0000_0000;
    const MACH_ENTRYOFF: u64 = 0x140;
    const MACH_MAIN_CMD_OFF: usize = 0xD0;
    const MACH_SYMS_OFF: usize = 0x200;
    const MACH_STRS_OFF: usize = 0x240;
    /// The string table runs exactly to end-of-file so that every
    /// truncated prefix of the image fails to parse.
    const MACH_STRS_SIZE: usize = MACH_IMG_SIZE - MACH_STRS_OFF;
    const MACH_DYLIB_NAME: &str = "/usr/lib/libSystem.B.dylib";

    fn write_nlist(img: &mut [u8], idx: usize, n_strx: u32, n_type: u8, n_sect: u8, value: u64) {
        let base = MACH_SYMS_OFF + idx * 16;
        put32(img, base, n_strx);
        img[base + 4] = n_type;
        img[base + 5] = n_sect;
        // n_desc left 0.
        put64(img, base + 8, value);
    }

    /// Build a minimal but well-formed thin arm64 Mach-O executable: an
    /// `__TEXT` segment with one `__text` section, an `LC_SYMTAB` with
    /// three symbols (defined external, defined local, undefined external),
    /// an `LC_MAIN`, and an `LC_LOAD_DYLIB`. Three known A64 instructions
    /// (`nop; movz x0, #42; ret`) sit at the entry point for `--disasm`.
    fn synthetic_macho64() -> Vec<u8> {
        let mut img = vec![0u8; MACH_IMG_SIZE];

        // mach_header_64.
        put32(&mut img, 0, macho::MH_MAGIC_64);
        put32(&mut img, 4, 0x0100_000C); // cputype: CPU_TYPE_ARM64
        put32(&mut img, 8, 0); // cpusubtype
        put32(&mut img, 12, 2); // filetype: MH_EXECUTE
        put32(&mut img, 16, 4); // ncmds
        put32(&mut img, 20, 0x100); // sizeofcmds (256)
        put32(
            &mut img,
            24,
            macho::MH_NOUNDEFS | macho::MH_DYLDLINK | macho::MH_TWOLEVEL | macho::MH_PIE,
        );

        // LC_SEGMENT_64 "__TEXT" with one section, cmdsize 72 + 80 = 152.
        let s = 0x20;
        put32(&mut img, s, macho::LC_SEGMENT_64);
        put32(&mut img, s + 4, 152);
        put(&mut img, s + 8, b"__TEXT");
        put64(&mut img, s + 24, MACH_TEXT_VMADDR); // vmaddr
        put64(&mut img, s + 32, 0x4000); // vmsize
        put64(&mut img, s + 40, 0); // fileoff
        put64(&mut img, s + 48, MACH_IMG_SIZE as u64); // filesize
        put32(&mut img, s + 56, macho::VM_PROT_READ | macho::VM_PROT_EXECUTE); // maxprot
        put32(&mut img, s + 60, macho::VM_PROT_READ | macho::VM_PROT_EXECUTE); // initprot
        put32(&mut img, s + 64, 1); // nsects
        // section_64 "__text".
        let t = s + 72;
        put(&mut img, t, b"__text");
        put(&mut img, t + 16, b"__TEXT");
        put64(&mut img, t + 32, MACH_TEXT_VMADDR + MACH_ENTRYOFF); // addr
        put64(&mut img, t + 40, 0x40); // size
        put32(&mut img, t + 48, MACH_ENTRYOFF as u32); // offset
        put32(&mut img, t + 52, 2); // align (2^2)
        put32(&mut img, t + 64, 0x8000_0400); // pure + some instructions

        // LC_SYMTAB.
        let y = 0xB8;
        put32(&mut img, y, macho::LC_SYMTAB);
        put32(&mut img, y + 4, 24);
        put32(&mut img, y + 8, MACH_SYMS_OFF as u32); // symoff
        put32(&mut img, y + 12, 3); // nsyms
        put32(&mut img, y + 16, MACH_STRS_OFF as u32); // stroff
        put32(&mut img, y + 20, MACH_STRS_SIZE as u32); // strsize

        // LC_MAIN.
        let m = MACH_MAIN_CMD_OFF;
        put32(&mut img, m, macho::LC_MAIN);
        put32(&mut img, m + 4, 24);
        put64(&mut img, m + 8, MACH_ENTRYOFF); // entryoff

        // LC_LOAD_DYLIB: 24-byte dylib_command + name, padded to cmdsize 56.
        let d = 0xE8;
        put32(&mut img, d, macho::LC_LOAD_DYLIB);
        put32(&mut img, d + 4, 56);
        put32(&mut img, d + 8, 24); // lc_str name offset
        put32(&mut img, d + 12, 2); // timestamp
        put32(&mut img, d + 16, 0x0001_0203); // current_version 1.2.3
        put32(&mut img, d + 20, 0x0001_0000); // compatibility_version 1.0.0
        put(&mut img, d + 24, MACH_DYLIB_NAME.as_bytes());

        // Entry-point code: nop; movz x0, #42; ret.
        put32(&mut img, MACH_ENTRYOFF as usize, 0xD503_201F);
        put32(&mut img, MACH_ENTRYOFF as usize + 4, 0xD280_0540);
        put32(&mut img, MACH_ENTRYOFF as usize + 8, 0xD65F_03C0);

        // Symbols: _main (defined external), _helper (defined local),
        // _printf (undefined external).
        write_nlist(&mut img, 0, 1, 0x0F, 1, MACH_TEXT_VMADDR + MACH_ENTRYOFF); // N_SECT | N_EXT
        write_nlist(&mut img, 1, 7, 0x0E, 1, MACH_TEXT_VMADDR + MACH_ENTRYOFF + 0x20); // N_SECT
        write_nlist(&mut img, 2, 15, 0x01, 0, 0); // N_UNDF | N_EXT

        // String table: offsets 1, 7, 15.
        put(&mut img, MACH_STRS_OFF, b"\0_main\0_helper\0_printf\0");

        img
    }

    /// Build a fat container holding two copies of the thin image, the
    /// first re-labeled x86-64 so the slices are distinguishable.
    fn synthetic_fat() -> Vec<u8> {
        let thin = synthetic_macho64();
        let mut slice_a = thin.clone();
        put32(&mut slice_a, 4, 0x0100_0007); // cputype: CPU_TYPE_X86_64
        let slice_b = thin; // arm64

        let off_a = 0x80;
        let off_b = off_a + slice_a.len();
        let mut img = vec![0u8; off_b + slice_b.len()];

        // fat_header — big-endian on disk, always.
        put32be(&mut img, 0, macho::FAT_MAGIC);
        put32be(&mut img, 4, 2); // nfat_arch

        for (i, (cputype, off, size)) in [
            (0x0100_0007u32, off_a, slice_a.len()),
            (0x0100_000C, off_b, slice_b.len()),
        ]
        .iter()
        .enumerate()
        {
            let base = 8 + i * 20;
            put32be(&mut img, base, *cputype);
            put32be(&mut img, base + 4, 0); // cpusubtype
            put32be(&mut img, base + 8, *off as u32);
            put32be(&mut img, base + 12, *size as u32);
            put32be(&mut img, base + 16, 14); // align (2^14)
        }

        put(&mut img, off_a, &slice_a);
        put(&mut img, off_b, &slice_b);
        img
    }

    // -- format sniffing -----------------------------------------------

    #[test]
    fn sniffs_elf_magic() {
        assert_eq!(sniff_format(b"\x7fELF\x02\x01"), Some(Format::Elf));
        assert_eq!(sniff_format(&synthetic_elf64()), Some(Format::Elf));
    }

    #[test]
    fn sniffs_pe_magic() {
        assert_eq!(sniff_format(b"MZ\x90\x00"), Some(Format::Pe));
        assert_eq!(sniff_format(&synthetic_pe64()), Some(Format::Pe));
    }

    #[test]
    fn sniffs_macho_magics() {
        // Thin 64-bit LE (MH_MAGIC_64 on disk), and the near-miss variants
        // routed to the thin parser for a precise diagnostic.
        assert_eq!(sniff_format(&synthetic_macho64()), Some(Format::MachO));
        assert_eq!(sniff_format(&[0xCF, 0xFA, 0xED, 0xFE]), Some(Format::MachO));
        assert_eq!(sniff_format(&[0xCE, 0xFA, 0xED, 0xFE]), Some(Format::MachO)); // 32-bit
        assert_eq!(sniff_format(&[0xFE, 0xED, 0xFA, 0xCF]), Some(Format::MachO)); // big-endian
        // Fat containers (big-endian magic).
        assert_eq!(sniff_format(&synthetic_fat()), Some(Format::MachOFat));
        assert_eq!(sniff_format(&[0xCA, 0xFE, 0xBA, 0xBE]), Some(Format::MachOFat));
        assert_eq!(sniff_format(&[0xCA, 0xFE, 0xBA, 0xBF]), Some(Format::MachOFat));
    }

    #[test]
    fn rejects_unknown_and_truncated_magic() {
        assert_eq!(sniff_format(b""), None);
        assert_eq!(sniff_format(b"M"), None);
        assert_eq!(sniff_format(b"\x7fEL"), None); // ELF magic cut short
        assert_eq!(sniff_format(&[0xCA, 0xFE, 0xBA]), None); // fat magic cut short
        assert_eq!(sniff_format(b"#!/bin/sh\n"), None);
    }

    #[test]
    fn unrecognized_format_is_a_clean_error() {
        let err = dump_to_string(b"random garbage bytes", opts_all()).unwrap_err();
        assert!(err.contains("unrecognized format"), "{err}");
        assert!(err.contains("test.bin"), "{err}");
    }

    #[test]
    fn unsupported_macho_variants_are_clean_errors() {
        // A 32-bit magic routes to the thin parser and comes back as a
        // precise "unsupported", not a generic "unrecognized format".
        let err = dump_to_string(&[0xCE, 0xFA, 0xED, 0xFE], opts_all()).unwrap_err();
        assert!(err.contains("unsupported"), "{err}");
        // A fat header written little-endian violates the format.
        let mut img = synthetic_fat();
        img[..4].copy_from_slice(&macho::FAT_MAGIC.to_le_bytes());
        let err = dump_to_string(&img, opts_all()).unwrap_err();
        assert!(err.contains("unsupported"), "{err}");
    }

    // -- ELF rendering ---------------------------------------------------

    #[test]
    fn elf_dump_renders_all_parts() {
        let out = dump_to_string(&synthetic_elf64(), opts_all()).unwrap();

        // Banner and file header.
        assert!(out.contains("test.bin: ELF64 image"), "{out}");
        assert!(out.contains("FILE HEADER"), "{out}");
        assert!(out.contains("DYN (position-independent executable)"), "{out}");
        assert!(out.contains("x86-64"), "{out}");
        assert!(out.contains("0x401000"), "{out}");
        assert!(out.contains("/lib64/ld-linux-x86-64.so.2"), "{out}");

        // Program headers.
        assert!(out.contains("PROGRAM HEADERS (2 entries)"), "{out}");
        assert!(out.contains("LOAD"), "{out}");
        assert!(out.contains("r-x"), "{out}");
        assert!(out.contains("INTERP"), "{out}");

        // Section headers.
        assert!(out.contains("SECTION HEADERS (5 entries)"), "{out}");
        assert!(out.contains(".text"), "{out}");
        assert!(out.contains("PROGBITS"), "{out}");
        assert!(out.contains("AX"), "{out}");
        assert!(out.contains(".shstrtab"), "{out}");

        // Symbols: .dynsym is populated, .symtab is empty.
        assert!(out.contains("SYMBOLS (.dynsym: 3 entries)"), "{out}");
        assert!(out.contains("SYMBOLS (.symtab: 0 entries)"), "{out}");
        assert!(out.contains("main"), "{out}");
        assert!(out.contains("GLOBAL"), "{out}");
        assert!(out.contains("FUNC"), "{out}");
        assert!(out.contains("UND"), "{out}");

        // Dynamic info: this fixture has no dynamic section, and the dump
        // says so instead of omitting the sections.
        assert!(out.contains("DYNAMIC"), "{out}");
        assert!(out.contains("(no dynamic section)"), "{out}");
        assert!(out.contains("PLT IMPORTS (0 entries)"), "{out}");

        // Format-aware imports: the undefined dynsym entry.
        assert!(out.contains("DYNAMIC IMPORTS"), "{out}");
        assert!(out.contains("printf"), "{out}");
    }

    #[test]
    fn elf_dynamic_and_plt_tables_render() {
        let out = dump_to_string(&synthetic_dynamic_elf64(), opts_all()).unwrap();

        assert!(out.contains("DYNAMIC"), "{out}");
        assert!(out.contains("SONAME   mylib.so"), "{out}");
        assert!(out.contains("NEEDED   libc.so.6"), "{out}");
        assert!(out.contains("NEEDED   libm.so.6"), "{out}");

        assert!(out.contains("PLT IMPORTS (3 entries)"), "{out}");
        assert!(out.contains("0x403018"), "{out}");
        assert!(out.contains("read"), "{out}");
        assert!(out.contains("write"), "{out}");
        assert!(out.contains("close"), "{out}");
    }

    #[test]
    fn elf_exports_flag_lists_defined_dynamic_symbols() {
        let out = dump_to_string(&synthetic_elf64(), opts_exports()).unwrap();
        assert!(out.contains("DYNAMIC EXPORTS"), "{out}");
        assert!(out.contains("main"), "{out}");
        assert!(!out.contains("printf"), "{out}"); // undefined: an import
        assert!(!out.contains("FILE HEADER"), "{out}");
        // The default "dump all" output keeps symbols in one place instead
        // of duplicating them as an exports view.
        let all = dump_to_string(&synthetic_elf64(), opts_all()).unwrap();
        assert!(!all.contains("DYNAMIC EXPORTS"), "{all}");
    }

    #[test]
    fn elf_selective_flags_pick_their_parts() {
        let img = synthetic_elf64();

        let sections = dump_to_string(&img, opts_only(false, true, false, false)).unwrap();
        assert!(sections.contains("SECTION HEADERS"), "{sections}");
        assert!(!sections.contains("FILE HEADER"), "{sections}");
        assert!(!sections.contains("SYMBOLS"), "{sections}");
        assert!(!sections.contains("DYNAMIC IMPORTS"), "{sections}");

        let symbols = dump_to_string(&img, opts_only(false, false, false, true)).unwrap();
        assert!(symbols.contains("SYMBOLS (.dynsym: 3 entries)"), "{symbols}");
        assert!(!symbols.contains("SECTION HEADERS"), "{symbols}");
        assert!(!symbols.contains("PROGRAM HEADERS"), "{symbols}");

        let headers = dump_to_string(&img, opts_only(true, false, false, false)).unwrap();
        assert!(headers.contains("FILE HEADER"), "{headers}");
        assert!(headers.contains("PROGRAM HEADERS"), "{headers}");
        assert!(!headers.contains("SECTION HEADERS"), "{headers}");
    }

    #[test]
    fn truncated_elf_is_an_error_not_a_panic() {
        let img = synthetic_elf64();
        // Every truncated prefix must fail cleanly: shorter than the magic
        // it's "unrecognized format", after that a parse error.
        for len in 0..img.len() {
            assert!(
                dump_to_string(&img[..len], opts_all()).is_err(),
                "len {len:#x}"
            );
        }
        assert!(dump_to_string(&img, opts_all()).is_ok());
    }

    // -- PE rendering ------------------------------------------------------

    #[test]
    fn pe_dump_still_renders_all_parts() {
        let out = dump_to_string(&synthetic_pe64(), opts_all()).unwrap();
        assert!(out.contains("test.bin: PE32+ image"), "{out}");
        assert!(out.contains("FILE HEADER"), "{out}");
        assert!(out.contains("x86-64"), "{out}");
        assert!(out.contains("OPTIONAL HEADER"), "{out}");
        assert!(out.contains("DATA DIRECTORIES"), "{out}");
        assert!(out.contains("SECTIONS"), "{out}");
        assert!(out.contains(".text"), "{out}");
        assert!(out.contains("IMPORTS"), "{out}");
        // Newly surfaced parts of the default dump.
        assert!(out.contains("DELAY IMPORTS"), "{out}");
        assert!(out.contains("EXPORTS"), "{out}");
        assert!(out.contains("(no export directory)"), "{out}");
        // The default "dump all" PE output is unchanged: no SYMBOLS note.
        assert!(!out.contains("SYMBOLS"), "{out}");
    }

    #[test]
    fn pe_exports_render_names_ordinals_and_forwarders() {
        let out = dump_to_string(&synthetic_pe64_with_exports(), opts_all()).unwrap();
        assert!(out.contains("EXPORTS"), "{out}");
        assert!(out.contains("TOOLKIT.dll (ordinal base 5)"), "{out}");
        // Named export with its RVA.
        assert!(out.contains("Alpha"), "{out}");
        assert!(out.contains("RVA 0x1111"), "{out}");
        // Ordinal-only export: dash for the name; unused slot 6 skipped.
        assert!(out.contains("5  Alpha"), "{out}");
        assert!(out.contains("7  -"), "{out}");
        assert!(!out.contains("6  "), "{out}");
        // Forwarder rendered as an arrow, not an RVA.
        assert!(out.contains("Beta"), "{out}");
        assert!(out.contains("-> OTHER.Func"), "{out}");
    }

    #[test]
    fn pe_exports_selective_flag_prints_only_exports() {
        let out = dump_to_string(&synthetic_pe64_with_exports(), opts_exports()).unwrap();
        assert!(out.contains("EXPORTS"), "{out}");
        assert!(out.contains("Alpha"), "{out}");
        assert!(!out.contains("FILE HEADER"), "{out}");
        assert!(!out.contains("SECTIONS"), "{out}");
        assert!(!out.contains("IMPORTS"), "{out}");
    }

    #[test]
    fn pe_delay_imports_render_like_imports() {
        let out = dump_to_string(&synthetic_pe64_with_delay(), opts_all()).unwrap();
        assert!(out.contains("DELAY IMPORTS"), "{out}");
        assert!(out.contains("USER32.dll (2 functions)"), "{out}");
        assert!(out.contains("MessageBoxA"), "{out}");
        assert!(out.contains("ordinal #7"), "{out}");
    }

    #[test]
    fn pe_explicit_symbols_flag_prints_a_note() {
        let out =
            dump_to_string(&synthetic_pe64(), opts_only(false, false, false, true)).unwrap();
        assert!(out.contains("SYMBOLS"), "{out}");
        assert!(out.contains("not modeled"), "{out}");
        assert!(!out.contains("FILE HEADER"), "{out}");
    }

    #[test]
    fn corrupt_pe_after_valid_magic_is_an_error() {
        let mut img = synthetic_pe64();
        img[0x80] = 0; // clobber the PE signature
        assert!(dump_to_string(&img, opts_all()).is_err());
    }

    // -- Mach-O rendering --------------------------------------------------

    #[test]
    fn macho_dump_renders_all_parts() {
        let out = dump_to_string(&synthetic_macho64(), opts_all()).unwrap();

        assert!(out.contains("test.bin: Mach-O 64-bit image"), "{out}");
        assert!(out.contains("FILE HEADER"), "{out}");
        assert!(out.contains("arm64"), "{out}");
        assert!(out.contains("MH_EXECUTE (executable)"), "{out}");
        assert!(out.contains("PIE"), "{out}");
        assert!(out.contains("file offset 0x140 (LC_MAIN)"), "{out}");

        assert!(out.contains("SEGMENTS (1 entries)"), "{out}");
        assert!(out.contains("__TEXT"), "{out}");
        assert!(out.contains("r-x"), "{out}");
        assert!(out.contains("__text"), "{out}");
        assert!(out.contains("0x100000140"), "{out}");

        assert!(out.contains("SYMBOLS (3 entries)"), "{out}");
        assert!(out.contains("_main"), "{out}");
        assert!(out.contains("_helper"), "{out}");
        assert!(out.contains("SECT"), "{out}");
        assert!(out.contains("UNDEF"), "{out}");

        // Imports: the dylib and the undefined external symbol.
        assert!(out.contains("DYLIBS (1 entries)"), "{out}");
        assert!(out.contains("/usr/lib/libSystem.B.dylib (version 1.2.3, compat 1.0.0)"), "{out}");
        assert!(out.contains("UNDEFINED SYMBOLS"), "{out}");
        assert!(out.contains("_printf"), "{out}");
    }

    #[test]
    fn macho_selective_flags_pick_their_parts() {
        let img = synthetic_macho64();

        let imports = dump_to_string(&img, opts_only(false, false, true, false)).unwrap();
        assert!(imports.contains("DYLIBS"), "{imports}");
        assert!(imports.contains("libSystem"), "{imports}");
        assert!(imports.contains("_printf"), "{imports}");
        assert!(!imports.contains("FILE HEADER"), "{imports}");
        assert!(!imports.contains("SEGMENTS"), "{imports}");
        // The symbol *table* stays out (its locals never show); only the
        // undefined-externals list appears.
        assert!(!imports.contains("_helper"), "{imports}");
        assert!(!imports.contains("SYMBOLS (3 entries)"), "{imports}");

        let exports = dump_to_string(&img, opts_exports()).unwrap();
        assert!(exports.contains("EXPORTS (defined external symbols)"), "{exports}");
        assert!(exports.contains("_main"), "{exports}");
        assert!(!exports.contains("_helper"), "{exports}"); // local
        assert!(!exports.contains("_printf"), "{exports}"); // undefined
        assert!(!exports.contains("SEGMENTS"), "{exports}");
    }

    #[test]
    fn fat_dump_lists_slices_then_dumps_each() {
        let out = dump_to_string(&synthetic_fat(), opts_all()).unwrap();

        assert!(out.contains("Mach-O universal (fat) container, 2 slices"), "{out}");
        assert!(out.contains("==== slice 0: x86-64 ===="), "{out}");
        assert!(out.contains("==== slice 1: arm64 ===="), "{out}");
        // Both slices fully dumped: one header, symbol table, and import
        // view apiece.
        assert_eq!(out.matches("FILE HEADER").count(), 2, "{out}");
        assert_eq!(out.matches("UNDEFINED SYMBOLS").count(), 2, "{out}");
        assert_eq!(out.matches("libSystem").count(), 2, "{out}");
    }

    #[test]
    fn truncated_macho_is_an_error_not_a_panic() {
        let img = synthetic_macho64();
        // The string table runs exactly to end-of-file, so every truncated
        // prefix must fail — and must never panic.
        for len in 0..img.len() {
            assert!(
                dump_to_string(&img[..len], opts_all()).is_err(),
                "len {len:#x}"
            );
        }
        assert!(dump_to_string(&img, opts_all()).is_ok());
    }

    // -- disassembly -------------------------------------------------------

    #[test]
    fn disasm_macho_arm64_decodes_known_entry_instructions() {
        let out = dump_to_string(&synthetic_macho64(), opts_disasm(3)).unwrap();
        assert!(out.contains("DISASSEMBLY (up to 3 instructions"), "{out}");
        assert!(out.contains("entry 0x100000140 (file offset 0x140), AArch64"), "{out}");
        assert!(out.contains("nop"), "{out}");
        assert!(out.contains("movz x0, #0x2a"), "{out}");
        assert!(out.contains("ret"), "{out}");
        // Raw bytes shown little-endian as stored.
        assert!(out.contains("1f 20 03 d5"), "{out}");
        // Nothing else was selected.
        assert!(!out.contains("FILE HEADER"), "{out}");
    }

    #[test]
    fn disasm_pe_x86_decodes_known_entry_instructions() {
        let mut img = synthetic_pe64();
        // Entry RVA 0x1000 -> file 0x200: push rbp; mov rbp, rsp; ret.
        put(&mut img, 0x200, &[0x55, 0x48, 0x89, 0xE5, 0xC3]);
        let out = dump_to_string(&img, opts_disasm(3)).unwrap();
        assert!(out.contains("entry 0x140001000 (file offset 0x200), x86-64"), "{out}");
        assert!(out.contains("push rbp"), "{out}");
        assert!(out.contains("mov rbp, rsp"), "{out}");
        assert!(out.contains("48 89 e5"), "{out}");
        assert!(out.contains("ret"), "{out}");
    }

    #[test]
    fn disasm_elf_x86_decodes_known_entry_instructions() {
        let mut img = synthetic_elf64();
        // Entry 0x401000 -> file 0x100: xor rax, rax; ret.
        put(&mut img, ELF_TEXT_OFF, &[0x48, 0x31, 0xC0, 0xC3]);
        let out = dump_to_string(&img, opts_disasm(2)).unwrap();
        assert!(out.contains("entry 0x401000 (file offset 0x100), x86-64"), "{out}");
        assert!(out.contains("xor rax, rax"), "{out}");
        assert!(out.contains("ret"), "{out}");
    }

    #[test]
    fn disasm_stops_cleanly_on_undecodable_bytes() {
        let mut img = synthetic_elf64();
        // 0F 0B is ud2 (decodable); 0F FF is not modeled -> clean stop.
        put(&mut img, ELF_TEXT_OFF, &[0x90, 0x0F, 0xFF]);
        let out = dump_to_string(&img, opts_disasm(8)).unwrap();
        assert!(out.contains("nop"), "{out}");
        assert!(out.contains("sweep stopped"), "{out}");
    }

    #[test]
    fn disasm_reports_missing_entry_point() {
        // ELF with e_entry = 0.
        let mut img = synthetic_elf64();
        put64(&mut img, 24, 0);
        let out = dump_to_string(&img, opts_disasm(4)).unwrap();
        assert!(out.contains("no entry point"), "{out}");

        // Mach-O without LC_MAIN: relabel the command as an unknown LC.
        let mut img = synthetic_macho64();
        put32(&mut img, MACH_MAIN_CMD_OFF, 0x3F);
        let out = dump_to_string(&img, opts_disasm(4)).unwrap();
        assert!(out.contains("no entry point"), "{out}");
    }

    #[test]
    fn disasm_reports_unmappable_entry_point() {
        let mut img = synthetic_elf64();
        put64(&mut img, 24, 0xDEAD_0000); // entry in no PT_LOAD segment
        let out = dump_to_string(&img, opts_disasm(4)).unwrap();
        assert!(out.contains("unmappable"), "{out}");
    }

    #[test]
    fn fat_dump_with_disasm_sweeps_each_slice() {
        let out = dump_to_string(&synthetic_fat(), opts_disasm(3)).unwrap();
        // The arm64 slice decodes its A64 words; the x86-64-labeled slice
        // holds the same bytes, which the x86 decoder sweeps as far as it
        // can — both slices must at least reach their listing.
        assert_eq!(out.matches("DISASSEMBLY").count(), 2, "{out}");
        assert!(out.contains("movz x0, #0x2a"), "{out}");
    }

    // -- listing (--listing / --db) ---------------------------------------

    #[test]
    fn listing_renders_the_recovered_program_for_every_format() {
        // PE: entry RVA 0x1000 -> file 0x200, `call +0`; ret.
        let mut pe = synthetic_pe64();
        put(&mut pe, 0x200, &[0xE8, 0x00, 0x00, 0x00, 0x00, 0xC3]);
        let out = dump_to_string(&pe, opts_listing(16)).unwrap();
        assert!(out.contains("\nLISTING\n"), "{out}");
        assert!(out.contains("sub_140001000:"), "{out}");
        assert!(out.contains("0x0000000140001000"), "{out}");
        // Nothing else was selected.
        assert!(!out.contains("FILE HEADER"), "{out}");

        // ELF: entry 0x401000 -> file 0x100.
        let mut elf = synthetic_elf64();
        put(&mut elf, ELF_TEXT_OFF, &[0x31, 0xC0, 0xC3]);
        let out = dump_to_string(&elf, opts_listing(16)).unwrap();
        assert!(out.contains("@ 0x0000000000401000"), "{out}");

        // Mach-O: the fixture's entry is A64 `nop; movz x0, #42; ret`.
        let out = dump_to_string(&synthetic_macho64(), opts_listing(16)).unwrap();
        assert!(out.contains("@ 0x0000000100000140"), "{out}");
    }

    #[test]
    fn listing_of_a_fat_container_is_a_note_not_an_error() {
        let out = dump_to_string(&synthetic_fat(), opts_listing(16)).unwrap();
        assert!(out.contains("a fat container holds no single image"), "{out}");
    }

    #[test]
    fn listing_max_function_count_is_honored() {
        // Two functions: the entry calls the one just past its `ret`.
        let mut pe = synthetic_pe64();
        put(&mut pe, 0x200, &[0xE8, 0x01, 0x00, 0x00, 0x00, 0xC3, 0xC3]);
        // The base fixture aims its import directory into .text; clear it
        // so those bytes stay code.
        put32(&mut pe, 0x80 + 4 + 20 + 112 + 8, 0);
        put32(&mut pe, 0x80 + 4 + 20 + 112 + 12, 0);

        let all = dump_to_string(&pe, opts_listing(16)).unwrap();
        assert!(all.contains("sub_140001006:"), "{all}");

        let one = dump_to_string(&pe, opts_listing(1)).unwrap();
        assert!(!one.contains("sub_140001006:"), "{one}");
        assert!(one.contains("more function(s) not shown"), "{one}");
    }

    #[test]
    fn listing_applies_an_annotation_database() {
        let mut pe = synthetic_pe64();
        put(&mut pe, 0x200, &[0x31, 0xC0, 0xC3]);

        // Anchor the entry function of exactly this image, then name it.
        let image = aletheia::load(&pe).unwrap();
        let program = cfg::recover(image.as_ref()).unwrap();
        let entry = *program.functions.keys().next().unwrap();
        let target = aletheia::anchor::of_function(image.as_ref(), &program.functions[&entry]);
        let mut db = annotate::Db::new();
        db.set_name(target, "decode_frame");

        // Round-trip through the on-disk format the `--db` flag reads.
        let db = annotate::Db::parse(&db.serialize()).expect("serialized db parses");

        let out = dump_to_string_with(&pe, opts_listing(16), Some(&db)).unwrap();
        assert!(out.contains("decode_frame:"), "{out}");
        assert!(!out.contains("sub_140001000:"), "{out}");
    }

    #[test]
    fn listing_of_an_undecodable_architecture_is_an_error() {
        // EM_RISCV: the loader accepts it, no decoder recovers it.
        let mut img = synthetic_elf64();
        put16(&mut img, 18, 243);
        let err = dump_to_string(&img, opts_listing(16)).unwrap_err();
        assert!(err.contains("no decoder for architecture"), "{err}");
    }

    // -- argument parsing --------------------------------------------------

    fn parse(args: &[&str]) -> Result<(String, Options, Option<String>), String> {
        // Most tests predate `--diff`; keep their 3-tuple shape and let
        // the diff-specific tests call `parse_args_from` directly.
        parse_args_from(args.iter().map(|s| s.to_string()))
            .map(|(path, opts, db, _)| (path, opts, db))
    }

    fn parse_diff(args: &[&str]) -> Result<Option<String>, String> {
        parse_args_from(args.iter().map(|s| s.to_string())).map(|(_, _, _, diff)| diff)
    }

    #[test]
    fn diff_flag_parses_both_spellings_and_is_a_selection() {
        let (_, opts, _, diff) =
            parse_args_from(["a.exe", "--diff", "b.exe"].iter().map(|s| s.to_string())).unwrap();
        assert_eq!(diff.as_deref(), Some("b.exe"));
        // `--diff` selects its own output, so the default dump is off.
        assert!(!opts.all && !opts.headers, "{opts:?}");

        assert_eq!(
            parse_diff(&["a.exe", "--diff=b.exe"]).unwrap().as_deref(),
            Some("b.exe")
        );

        let err = parse_diff(&["a.exe", "--diff"]).unwrap_err();
        assert!(err.contains("`--diff` requires a path"), "{err}");
    }

    #[test]
    fn vtables_flag_is_a_selection_and_dumps_an_empty_result() {
        let (_, opts, _) = parse(&["a.exe", "--vtables"]).unwrap();
        assert!(opts.vtables);
        assert!(!opts.all && !opts.headers, "{opts:?}");

        // No RTTI in the synthetic image: the section still prints, with
        // an honest zero.
        let img = synthetic_elf64();
        let text = dump_to_string(
            &img,
            Options {
                vtables: true,
                ..opts_none()
            },
        )
        .unwrap();
        assert!(text.contains("C++ STRUCTURES"), "{text}");
        assert!(text.contains("0 class(es) recovered"), "{text}");
    }

    #[test]
    fn rustmeta_flag_is_a_selection_and_dumps_an_empty_result() {
        let (_, opts, _) = parse(&["a.exe", "--rustmeta"]).unwrap();
        assert!(opts.rustmeta);
        assert!(!opts.all && !opts.headers, "{opts:?}");

        let img = synthetic_elf64();
        let text = dump_to_string(
            &img,
            Options {
                rustmeta: true,
                ..opts_none()
            },
        )
        .unwrap();
        assert!(text.contains("RUST PANIC METADATA"), "{text}");
        assert!(text.contains("no Rust panic metadata recovered"), "{text}");
    }

    #[test]
    fn gotypes_flag_is_a_selection_and_dumps_an_empty_result() {
        let (_, opts, _) = parse(&["a.exe", "--gotypes"]).unwrap();
        assert!(opts.gotypes);
        assert!(!opts.all && !opts.headers, "{opts:?}");

        let img = synthetic_elf64();
        let text = dump_to_string(
            &img,
            Options {
                gotypes: true,
                ..opts_none()
            },
        )
        .unwrap();
        assert!(text.contains("GO TYPE METADATA"), "{text}");
        assert!(text.contains("no Go type metadata recovered"), "{text}");
    }

    #[test]
    fn devirt_flag_is_a_selection_and_dumps_an_empty_result() {
        let (_, opts, _) = parse(&["a.exe", "--devirt"]).unwrap();
        assert!(opts.devirt);
        assert!(!opts.all && !opts.headers, "{opts:?}");

        let img = synthetic_elf64();
        let text = dump_to_string(
            &img,
            Options {
                devirt: true,
                ..opts_none()
            },
        )
        .unwrap();
        assert!(text.contains("VIRTUAL CALLS"), "{text}");
        assert!(text.contains("no virtual call sites resolved"), "{text}");
    }

    #[test]
    fn lift_flag_parses_with_and_without_a_count() {
        let (_, opts, _) = parse(&["a.exe", "--lift"]).unwrap();
        assert_eq!(opts.lift, Some(DEFAULT_LIFT_FUNCTIONS));
        assert!(!opts.all && !opts.headers, "{opts:?}");

        let (_, opts, _) = parse(&["a.exe", "--lift=1"]).unwrap();
        assert_eq!(opts.lift, Some(1));

        let err = parse(&["a.exe", "--lift=x"]).unwrap_err();
        assert!(err.contains("invalid function count"), "{err}");
    }

    #[test]
    fn simplify_flag_implies_lift_and_composes_with_a_count() {
        let (_, opts, _) = parse(&["a.exe", "--simplify"]).unwrap();
        assert!(opts.simplify);
        assert_eq!(opts.lift, Some(DEFAULT_LIFT_FUNCTIONS));
        assert!(!opts.all && !opts.headers, "{opts:?}");

        // An explicit count wins regardless of flag order.
        let (_, opts, _) = parse(&["a.exe", "--simplify", "--lift=2"]).unwrap();
        assert!(opts.simplify);
        assert_eq!(opts.lift, Some(2));
        let (_, opts, _) = parse(&["a.exe", "--lift=2", "--simplify"]).unwrap();
        assert_eq!(opts.lift, Some(2));

        // Without the modifier the lift stays raw.
        let (_, opts, _) = parse(&["a.exe", "--lift"]).unwrap();
        assert!(!opts.simplify);
    }

    #[test]
    fn simplified_lift_of_an_x86_program_drops_dead_flag_writes() {
        // `xor eax, eax ; ret`: the raw lift writes every status flag;
        // nothing reads them, so the simplified dump has none.
        let mut img = synthetic_elf64();
        put(&mut img, ELF_TEXT_OFF, &[0x31, 0xC0, 0xC3]);
        let raw = dump_to_string(
            &img,
            Options {
                lift: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        assert!(raw.contains("ZF :="), "{raw}");
        let simplified = dump_to_string(
            &img,
            Options {
                lift: Some(4),
                simplify: true,
                ..opts_none()
            },
        )
        .unwrap();
        assert!(simplified.contains("IR LIFT"), "{simplified}");
        assert!(!simplified.contains("ZF :="), "{simplified}");
        assert!(simplified.contains("return"), "{simplified}");
    }

    #[test]
    fn lift_of_an_x86_program_prints_ir() {
        // A tiny x86-64 function: `mov eax, ecx ; ret` at the ELF entry.
        let mut img = synthetic_elf64();
        // 89 C8 = mov eax, ecx ; C3 = ret, placed at the .text file offset
        // (the entry VA maps there).
        put(&mut img, ELF_TEXT_OFF, &[0x89, 0xC8, 0xC3]);
        let text = dump_to_string(
            &img,
            Options {
                lift: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        assert!(text.contains("IR LIFT"), "{text}");
        // The 32-bit move zero-extends into the 64-bit cell.
        assert!(text.contains("zext"), "{text}");
        assert!(text.contains("return"), "{text}");
    }

    #[test]
    fn ssa_flag_parses_with_and_without_a_count() {
        let (_, opts, _) = parse(&["a.exe", "--ssa"]).unwrap();
        assert_eq!(opts.ssa, Some(DEFAULT_LIFT_FUNCTIONS));
        assert!(!opts.all && !opts.headers, "{opts:?}");
        assert_eq!(opts.lift, None, "--ssa does not imply --lift");

        let (_, opts, _) = parse(&["a.exe", "--ssa=2"]).unwrap();
        assert_eq!(opts.ssa, Some(2));

        let err = parse(&["a.exe", "--ssa=x"]).unwrap_err();
        assert!(err.contains("invalid function count"), "{err}");
    }

    #[test]
    fn ssa_of_an_x86_diamond_prints_a_phi_at_the_merge() {
        // xor eax,eax ; je +9 ; inc eax ; jmp +11 ; (pad) ; dec eax ;
        // mov ecx, eax ; ret — eax is defined in both arms and read at
        // the merge, so the SSA dump carries exactly one phi.
        let mut img = synthetic_elf64();
        put(
            &mut img,
            ELF_TEXT_OFF,
            &[
                0x31, 0xC0, 0x74, 0x05, 0xFF, 0xC0, 0xEB, 0x03, 0x90, 0xFF, 0xC8, 0x89, 0xC1,
                0xC3,
            ],
        );
        let text = dump_to_string(
            &img,
            Options {
                ssa: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        assert!(text.contains("IR SSA"), "{text}");
        assert!(text.contains("(ssa)"), "{text}");
        assert!(text.contains(" := phi("), "{text}");
        assert!(text.contains("rax#"), "{text}");
    }

    /// A synthetic ELF whose entry calls a second function:
    /// `xor eax,eax ; call +9 ; ret`, callee `nop ; ret`.
    fn calling_elf64() -> Vec<u8> {
        let mut img = synthetic_elf64();
        put(
            &mut img,
            ELF_TEXT_OFF,
            &[
                0x31, 0xC0, // xor eax, eax
                0xE8, 0x02, 0x00, 0x00, 0x00, // call +9
                0xC3, // ret
                0x90, // pad
                0x90, // callee: nop
                0xC3, // ret
            ],
        );
        img
    }

    #[test]
    fn ssa_of_a_calling_function_shows_call_effects_and_is_deterministic() {
        let img = calling_elf64();
        let opts = Options {
            ssa: Some(4),
            ..opts_none()
        };
        let text = dump_to_string(&img, opts).unwrap();
        assert!(text.contains("IR SSA"), "{text}");
        // The call is followed by the effect intrinsic with versioned
        // clobbers, then the explicit rsp restore.
        assert!(text.contains(" := callfx("), "{text}");
        assert!(text.contains("rax#"), "{text}");
        assert!(text.contains("rsp#"), "{text}");
        assert!(text.contains("+ 0x8.q)"), "{text}");
        // Byte-deterministic: a second run prints the same bytes.
        assert_eq!(text, dump_to_string(&img, opts).unwrap());
    }

    // -- --structure ---------------------------------------------------

    #[test]
    fn structure_flag_parses_with_and_without_a_count() {
        let (_, opts, _) = parse(&["a.exe", "--structure"]).unwrap();
        assert_eq!(opts.structure, Some(DEFAULT_LIFT_FUNCTIONS));
        assert_eq!(opts.ssa_opt, None, "--structure does not imply --ssa-opt");
        assert_eq!(opts.ssa, None, "--structure does not imply --ssa");

        let (_, opts, _) = parse(&["a.exe", "--structure=2"]).unwrap();
        assert_eq!(opts.structure, Some(2));

        let err = parse(&["a.exe", "--structure=x"]).unwrap_err();
        assert!(err.contains("invalid function count"), "{err}");
    }

    #[test]
    fn structure_of_a_diamond_recovers_an_if_else() {
        let img = diamond_constant_elf64();
        let opts = Options {
            structure: Some(4),
            ..opts_none()
        };
        let text = dump_to_string(&img, opts).unwrap();
        assert!(text.contains("IR STRUCTURE"), "{text}");
        assert!(text.contains("(structure)"), "{text}");
        // The `je` diamond: a head block, both arms, and the merge — no
        // goto anywhere, since the schemas cover it exactly.
        let body: Vec<&str> = text
            .lines()
            .skip_while(|l| !l.contains("(structure)"))
            .skip(1)
            .take(6)
            .collect();
        assert_eq!(
            body,
            [
                "block loc_401000",
                // `je` taken: the far arm is the un-negated side.
                "if cond loc_401000",
                "  block loc_40100d",
                "else",
                "  block loc_401009",
                "block loc_40100f",
            ],
            "{text}"
        );
        assert!(!text.contains("goto"), "{text}");
        assert!(!text.contains("; structure:"), "{text}");
        // Byte-deterministic: a second run prints the same bytes.
        assert_eq!(text, dump_to_string(&img, opts).unwrap());
    }

    #[test]
    fn structure_of_a_calling_function_is_a_flat_sequence() {
        let img = calling_elf64();
        let opts = Options {
            structure: Some(4),
            ..opts_none()
        };
        let text = dump_to_string(&img, opts).unwrap();
        assert!(text.contains("IR STRUCTURE"), "{text}");
        // A call is an ordinary statement to the structurer: one block,
        // one leaf, no construct at all.
        assert!(text.contains("block loc_401000"), "{text}");
        assert!(!text.contains("if "), "{text}");
        assert!(!text.contains("while "), "{text}");
        assert_eq!(text, dump_to_string(&img, opts).unwrap());
    }

    /// A synthetic ELF whose entry dispatches through a proven jump
    /// table: `cmp edi, 3 ; ja default ; lea rcx, [rip + table] ;
    /// movsxd rax, [rcx + rdi*4] ; add rax, rcx ; jmp rax`, with four
    /// `ret` cases behind self-relative offsets at +0x40. The case
    /// bodies are reachable only through the folded table edges.
    fn jump_table_elf64() -> Vec<u8> {
        let mut img = synthetic_elf64();
        put(&mut img, ELF_TEXT_OFF, &[0x83, 0xFF, 0x03]); // cmp edi, 3
        put(&mut img, ELF_TEXT_OFF + 0x03, &[0x77, 0x2B]); // ja +0x30
        // lea rcx, [rip + 0x34] -> table at +0x40
        put(
            &mut img,
            ELF_TEXT_OFF + 0x05,
            &[0x48, 0x8D, 0x0D, 0x34, 0x00, 0x00, 0x00],
        );
        put(&mut img, ELF_TEXT_OFF + 0x0C, &[0x48, 0x63, 0x04, 0xB9]); // movsxd rax,[rcx+rdi*4]
        put(&mut img, ELF_TEXT_OFF + 0x10, &[0x48, 0x01, 0xC8]); // add rax, rcx
        put(&mut img, ELF_TEXT_OFF + 0x13, &[0xFF, 0xE0]); // jmp rax
        put(&mut img, ELF_TEXT_OFF + 0x30, &[0xC3]); // default: ret
        for (i, case) in [0x50usize, 0x54, 0x58, 0x5C].into_iter().enumerate() {
            put32(&mut img, ELF_TEXT_OFF + 0x40 + 4 * i, (case - 0x40) as u32);
            put(&mut img, ELF_TEXT_OFF + case, &[0xC3]); // case i: ret
        }
        img
    }

    #[test]
    fn structure_of_a_proven_jump_table_is_a_switch_not_an_opaque() {
        let img = jump_table_elf64();
        let opts = Options {
            structure: Some(4),
            ..opts_none()
        };
        let text = dump_to_string(&img, opts).unwrap();
        // The folded dispatch renders a real switch over the four proven
        // cases — no opaque leaf, no goto.
        assert!(text.contains("switch loc_401005"), "{text}");
        for case in ["loc_401050", "loc_401054", "loc_401058", "loc_40105c"] {
            assert!(text.contains(&format!("case {case}:")), "{text}");
        }
        assert!(!text.contains("opaque"), "{text}");
        assert!(!text.contains("goto"), "{text}");
        // Byte-deterministic: a second run prints the same bytes.
        assert_eq!(text, dump_to_string(&img, opts).unwrap());
    }

    // -- --decompile ---------------------------------------------------

    #[test]
    fn decompile_flag_parses_with_and_without_a_count() {
        let (_, opts, _) = parse(&["a.exe", "--decompile"]).unwrap();
        assert_eq!(opts.decompile, Some(DEFAULT_LIFT_FUNCTIONS));
        assert_eq!(opts.structure, None, "--decompile does not imply --structure");
        assert_eq!(opts.ssa_opt, None, "--decompile does not imply --ssa-opt");

        let (_, opts, _) = parse(&["a.exe", "--decompile=2"]).unwrap();
        assert_eq!(opts.decompile, Some(2));

        let err = parse(&["a.exe", "--decompile=x"]).unwrap_err();
        assert!(err.contains("invalid function count"), "{err}");
    }

    #[test]
    fn sigs_flag_parses_with_and_without_a_count() {
        let (_, opts, _) = parse(&["a.exe", "--sigs"]).unwrap();
        assert_eq!(opts.sigs, Some(DEFAULT_LIFT_FUNCTIONS));
        assert_eq!(opts.stack, None, "--sigs does not imply --stack");

        let (_, opts, _) = parse(&["a.exe", "--sigs=2"]).unwrap();
        assert_eq!(opts.sigs, Some(2));

        let err = parse(&["a.exe", "--sigs=x"]).unwrap_err();
        assert!(err.contains("invalid function count"), "{err}");
    }

    #[test]
    fn promote_flag_parses_with_and_without_a_count() {
        let (_, opts, _) = parse(&["a.exe", "--promote"]).unwrap();
        assert_eq!(opts.promote, Some(DEFAULT_LIFT_FUNCTIONS));
        assert_eq!(opts.stack, None, "--promote does not imply --stack");

        let (_, opts, _) = parse(&["a.exe", "--promote=2"]).unwrap();
        assert_eq!(opts.promote, Some(2));

        let err = parse(&["a.exe", "--promote=x"]).unwrap_err();
        assert!(err.contains("invalid function count"), "{err}");
    }

    #[test]
    fn decompile_of_a_diamond_emits_an_if_else_with_a_relational_guard() {
        let img = diamond_constant_elf64();
        let opts = Options {
            decompile: Some(4),
            ..opts_none()
        };
        let text = dump_to_string(&img, opts).unwrap();
        assert!(text.contains("PSEUDOCODE"), "{text}");
        assert!(text.contains("(pseudo)"), "{text}");
        // The `test ecx, ecx` / `je` head renders as a real conditional
        // over the forwarded relation, with braced arms and no goto.
        assert!(text.contains("if ("), "{text}");
        assert!(text.contains("} else {"), "{text}");
        assert!(text.contains("== 0x0.d)"), "{text}");
        assert!(text.contains("return;"), "{text}");
        assert!(!text.contains("goto"), "{text}");
        // Byte-deterministic: a second run prints the same bytes.
        assert_eq!(text, dump_to_string(&img, opts).unwrap());
    }

    #[test]
    fn decompile_of_a_calling_function_spells_the_call_and_its_assumptions() {
        let img = calling_elf64();
        let opts = Options {
            decompile: Some(4),
            ..opts_none()
        };
        let text = dump_to_string(&img, opts).unwrap();
        // The call renders with its target; the ABI-assumed clobbers are
        // declared, never silent.
        assert!(text.contains("call 0x"), "{text}");
        assert!(text.contains("callfx("), "{text}");
        assert!(text.contains("/* abi-assumed: "), "{text}");
        assert_eq!(text, dump_to_string(&img, opts).unwrap());
    }

    #[test]
    fn decompile_of_a_proven_jump_table_spells_switch_and_cases() {
        let img = jump_table_elf64();
        let opts = Options {
            decompile: Some(4),
            ..opts_none()
        };
        let text = dump_to_string(&img, opts).unwrap();
        // The proven dispatch is C: a switch with labeled cases, not the
        // indirect-jump honesty marker.
        assert!(text.contains("switch ("), "{text}");
        assert!(text.contains("case loc_401050:"), "{text}");
        assert!(text.contains("case loc_40105c:"), "{text}");
        assert!(!text.contains("/* indirect jump"), "{text}");
        assert!(!text.contains("goto"), "{text}");
        assert_eq!(text, dump_to_string(&img, opts).unwrap());
    }

    #[test]
    fn decompile_admits_an_aarch64_image() {
        // Same gate as the other IR views: the synthetic arm64 Mach-O
        // (`nop; movz x0, #42; ret`) decompiles instead of drawing the
        // unsupported-arch note — the renderer is arch-agnostic, so the
        // constant reaches the pseudocode.
        let opts = Options {
            decompile: Some(4),
            ..opts_none()
        };
        let arm = dump_to_string(&synthetic_macho64(), opts).unwrap();
        assert!(arm.contains("PSEUDOCODE"), "{arm}");
        assert!(!arm.contains("IR lifting is implemented"), "{arm}");
        assert!(arm.contains("0x2a.q"), "{arm}");
    }

    /// A synthetic ELF whose entry sets eax to a constant, branches
    /// through a diamond that never redefines it, and compares it in the
    /// merge block: the cross-block constant the propagation slice exists
    /// to reach. The merge's `cmp` is followed by a `je`, so exactly one
    /// of its four flag writes is read — the canonical fixture the
    /// dead-code slice reduces.
    fn diamond_constant_elf64() -> Vec<u8> {
        let mut img = synthetic_elf64();
        put(
            &mut img,
            ELF_TEXT_OFF,
            &[
                0xB8, 0x05, 0x00, 0x00, 0x00, // mov eax, 5
                0x85, 0xC9, // test ecx, ecx
                0x74, 0x04, // je +4 (the far arm)
                0x89, 0xCA, // mov edx, ecx   (near arm)
                0xEB, 0x02, // jmp +2 (the merge)
                0x89, 0xCB, // mov ebx, ecx   (far arm)
                0x39, 0xC1, // cmp ecx, eax   (merge: reads eax)
                0x74, 0x01, // je +1 (reads ZF, and only ZF)
                0x90, // nop
                0xC3, // ret
            ],
        );
        img
    }

    /// A synthetic ELF whose SSA carries a collapsible phi: both arms
    /// copy the same 64-bit register into rdx, so the merge's
    /// phi(rdx#1, rdx#2) merges one value.
    fn collapsible_phi_elf64() -> Vec<u8> {
        let mut img = synthetic_elf64();
        put(
            &mut img,
            ELF_TEXT_OFF,
            &[
                0x48, 0x89, 0xC8, // mov rax, rcx
                0x48, 0x85, 0xC0, // test rax, rax
                0x74, 0x05, // je +5 (the far arm)
                0x48, 0x89, 0xC2, // mov rdx, rax  (near arm)
                0xEB, 0x03, // jmp +3 (the merge)
                0x48, 0x89, 0xC2, // mov rdx, rax  (far arm)
                0x48, 0x89, 0xD1, // mov rcx, rdx  (merge: reads the phi)
                0xC3, // ret
            ],
        );
        img
    }

    #[test]
    fn ssa_opt_flag_parses_with_and_without_a_count() {
        let (_, opts, _) = parse(&["a.exe", "--ssa-opt"]).unwrap();
        assert_eq!(opts.ssa_opt, Some(DEFAULT_LIFT_FUNCTIONS));
        assert!(!opts.all && !opts.headers, "{opts:?}");
        assert_eq!(opts.ssa, None, "--ssa-opt does not imply --ssa");
        assert_eq!(opts.lift, None, "--ssa-opt does not imply --lift");

        let (_, opts, _) = parse(&["a.exe", "--ssa-opt=2"]).unwrap();
        assert_eq!(opts.ssa_opt, Some(2));

        let err = parse(&["a.exe", "--ssa-opt=x"]).unwrap_err();
        assert!(err.contains("invalid function count"), "{err}");
    }

    #[test]
    fn ssa_opt_carries_a_constant_across_blocks_to_its_use() {
        let img = diamond_constant_elf64();
        let raw = dump_to_string(
            &img,
            Options {
                ssa: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        let opt = dump_to_string(
            &img,
            Options {
                ssa_opt: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        assert!(opt.contains("IR SSA (optimized)"), "{opt}");

        // The merge block's comparison reads the versioned name in the
        // faithful view and the constant in the optimized one.
        let merge = |text: &str| -> String {
            text.split("loc_40100f:")
                .nth(1)
                .unwrap_or_default()
                .to_string()
        };
        assert!(
            merge(&raw).contains("rax#1"),
            "the faithful view keeps the name:\n{raw}"
        );
        assert!(
            merge(&opt).contains("0x5.d") && !merge(&opt).contains("rax#1"),
            "the constant reaches the merge use:\n{opt}"
        );
        // The branch and the block set survive: the same labels and edge
        // lines in both views.
        let labels = |text: &str| -> Vec<String> {
            text.lines()
                .filter(|l| l.starts_with("loc_") || l.trim_start().starts_with("; ->"))
                .map(|l| l.to_string())
                .collect()
        };
        assert_eq!(labels(&raw), labels(&opt));
        // Byte-deterministic across runs.
        assert_eq!(
            opt,
            dump_to_string(
                &img,
                Options {
                    ssa_opt: Some(4),
                    ..opts_none()
                }
            )
            .unwrap()
        );
    }

    #[test]
    fn ssa_opt_sweeps_the_dead_flag_writes_and_reports_the_reduction() {
        // The fixture's merge `cmp ecx, eax` writes four flags and the
        // `je` after it reads exactly one: the DESIGN exit case.
        let img = diamond_constant_elf64();
        let raw = dump_to_string(
            &img,
            Options {
                ssa: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        let opt = dump_to_string(
            &img,
            Options {
                ssa_opt: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        // The faithful view keeps all four of the merge `cmp`'s flag
        // writes...
        let raw_merge = raw.split("loc_40100f:").nth(1).unwrap_or_default();
        for flag in ["ZF#", "SF#", "CF#", "OF#"] {
            assert!(
                raw_merge.contains(flag),
                "the faithful view keeps {flag}:\n{raw}"
            );
        }
        // ...the optimized one keeps none of them: the three the branch
        // never reads are swept, and the one it does read is forwarded
        // into the branch and swept behind it.
        let merge = opt.split("loc_40100f:").nth(1).unwrap_or_default();
        for flag in ["ZF#", "SF#", "CF#", "OF#"] {
            assert!(
                !merge.contains(flag),
                "the merge's flag {flag} survived:\n{opt}"
            );
        }
        // The measured reduction is printed, once, above the function.
        let note = opt
            .lines()
            .find(|l| l.starts_with("; dce: removed "))
            .expect("the dce note is printed");
        let numbers: Vec<usize> = note
            .split_whitespace()
            .filter_map(|w| w.parse().ok())
            .collect();
        assert_eq!(numbers.len(), 2, "{note}");
        assert!(numbers[0] > 0 && numbers[0] < numbers[1], "{note}");
        assert!(note.ends_with(" statements"), "{note}");
        // `--ssa` stays the faithful view: no note, nothing swept.
        assert!(!raw.contains("; dce:"), "{raw}");
        // Byte-deterministic across runs.
        assert_eq!(
            opt,
            dump_to_string(
                &img,
                Options {
                    ssa_opt: Some(4),
                    ..opts_none()
                }
            )
            .unwrap()
        );
    }

    #[test]
    fn ssa_opt_forwards_the_flag_plumbing_into_a_relational_branch() {
        // The DESIGN slice-5 exit criterion, end to end: the merge's
        // `cmp ecx, eax` + `je` renders as the comparison the branch
        // really tests, with no flag name left anywhere in the function.
        let img = diamond_constant_elf64();
        let raw = dump_to_string(
            &img,
            Options {
                ssa: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        let opt = dump_to_string(
            &img,
            Options {
                ssa_opt: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        // The faithful view branches on a flag read...
        assert!(
            raw.lines().any(|l| l.contains("goto if ZF#")),
            "the faithful view shows the flag plumbing:\n{raw}"
        );
        // ...the optimized one branches on the relation itself.
        assert!(
            opt.lines()
                .any(|l| l.contains("goto if (trunc.d(rcx#0) == 0x5.d)")),
            "the branch reads the comparison:\n{opt}"
        );
        assert!(
            !opt.contains("ZF#") && !opt.contains("SF#"),
            "no flag survives the collapse:\n{opt}"
        );
        // The CFG is untouched: the same labels and edges in both views.
        let labels = |text: &str| -> Vec<String> {
            text.lines()
                .filter(|l| l.starts_with("loc_") || l.trim_start().starts_with("; ->"))
                .map(|l| l.to_string())
                .collect()
        };
        assert_eq!(labels(&raw), labels(&opt));
        // Byte-deterministic across runs.
        assert_eq!(
            opt,
            dump_to_string(
                &img,
                Options {
                    ssa_opt: Some(4),
                    ..opts_none()
                }
            )
            .unwrap()
        );
    }

    #[test]
    fn ssa_opt_keeps_the_call_effects_and_their_argument_setups_after_the_sweep() {
        let img = calling_elf64();
        let opt = dump_to_string(
            &img,
            Options {
                ssa_opt: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        // The effect intrinsic is never swept, and neither is the rsp
        // restore that follows it (rsp is live-out).
        assert!(opt.contains(" := callfx("), "{opt}");
        assert!(opt.contains("+ 0x8.q)"), "{opt}");
    }

    #[test]
    fn ssa_opt_collapses_a_phi_the_faithful_view_shows() {
        let img = collapsible_phi_elf64();
        let raw = dump_to_string(
            &img,
            Options {
                ssa: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        let opt = dump_to_string(
            &img,
            Options {
                ssa_opt: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        assert!(raw.contains(" := phi("), "{raw}");
        assert!(!opt.contains(" := phi("), "{opt}");
    }

    #[test]
    fn ssa_opt_keeps_call_effects_verbatim_and_stops_at_the_call() {
        let img = calling_elf64();
        let opt = dump_to_string(
            &img,
            Options {
                ssa_opt: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        // The call-effect intrinsic and its register reads are untouched.
        assert!(opt.contains(" := callfx("), "{opt}");
        let call_line = opt
            .lines()
            .find(|l| l.contains(":= callfx("))
            .unwrap_or_default()
            .to_string();
        assert!(call_line.contains("rdi#"), "{call_line}");
        assert!(
            !call_line.contains("0x"),
            "callfx reads stay register reads: {call_line}"
        );
        // The pre-call `xor eax, eax` constant does not cross the call:
        // the post-call rsp restore reads the clobbered version.
        assert!(opt.contains("rsp#"), "{opt}");
    }

    #[test]
    fn ssa_opt_of_a_fat_image_is_still_the_container_note() {
        let opts = Options {
            ssa_opt: Some(4),
            ..opts_none()
        };
        let fat = dump_to_string(&synthetic_fat(), opts).unwrap();
        assert!(
            fat.contains("a fat container holds no single image"),
            "{fat}"
        );
    }

    #[test]
    fn the_ir_views_admit_an_aarch64_image() {
        // The synthetic arm64 Mach-O's entry is `nop; movz x0, #42; ret`:
        // every IR view produces output in aarch64 register names, not
        // the unsupported-arch note.
        let img = synthetic_macho64();

        let lift = dump_to_string(
            &img,
            Options {
                lift: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        assert!(!lift.contains("IR lifting is implemented"), "{lift}");
        assert!(lift.contains("x0 := 0x2a.q"), "{lift}");
        assert!(lift.contains("return x30"), "{lift}");

        let ssa = dump_to_string(
            &img,
            Options {
                ssa: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        assert!(ssa.contains("(ssa)"), "{ssa}");
        assert!(ssa.contains("x0#"), "{ssa}");
        assert!(ssa.contains("x30#"), "{ssa}");
        assert!(!ssa.contains("rax"), "{ssa}");

        let opt = dump_to_string(
            &img,
            Options {
                ssa_opt: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        assert!(opt.contains("IR SSA (optimized)"), "{opt}");
        assert!(opt.contains("x0#"), "{opt}");

        let structure = dump_to_string(
            &img,
            Options {
                structure: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        assert!(structure.contains("(structure)"), "{structure}");

        // Byte-determinism across runs, per view.
        let makes: [fn(usize) -> Options; 4] = [
            |n| Options {
                lift: Some(n),
                ..opts_none()
            },
            |n| Options {
                ssa: Some(n),
                ..opts_none()
            },
            |n| Options {
                ssa_opt: Some(n),
                ..opts_none()
            },
            |n| Options {
                structure: Some(n),
                ..opts_none()
            },
        ];
        for make in makes {
            assert_eq!(
                dump_to_string(&img, make(4)).unwrap(),
                dump_to_string(&img, make(4)).unwrap()
            );
        }
    }

    #[test]
    fn lift_and_simplify_stay_faithful_with_no_call_effects() {
        let img = calling_elf64();
        let lifted = dump_to_string(
            &img,
            Options {
                lift: Some(4),
                ..opts_none()
            },
        )
        .unwrap();
        assert!(lifted.contains("call "), "{lifted}");
        assert!(!lifted.contains("callfx"), "{lifted}");

        let simplified = dump_to_string(
            &img,
            Options {
                lift: Some(4),
                simplify: true,
                ..opts_none()
            },
        )
        .unwrap();
        assert!(simplified.contains("call "), "{simplified}");
        assert!(!simplified.contains("callfx"), "{simplified}");
    }

    #[test]
    fn gostrings_flag_is_a_selection_and_dumps_an_empty_result() {
        let (_, opts, _) = parse(&["a.exe", "--gostrings"]).unwrap();
        assert!(opts.gostrings);
        assert!(!opts.all && !opts.headers, "{opts:?}");

        // The synthetic ELF is not a Go image, so recovery is gated off.
        let img = synthetic_elf64();
        let text = dump_to_string(
            &img,
            Options {
                gostrings: true,
                ..opts_none()
            },
        )
        .unwrap();
        assert!(text.contains("GO STRINGS"), "{text}");
        assert!(text.contains("no Go strings recovered"), "{text}");
    }

    #[test]
    fn diff_of_an_image_against_itself_reports_no_differences() {
        let img = synthetic_elf64();
        let mut out = Vec::new();
        print_diff("old.elf", &img, "new.elf", &img, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("DIFF (against new.elf)"), "{text}");
        assert!(text.contains("no differences"), "{text}");
    }

    #[test]
    fn listing_flag_parses_with_and_without_a_count() {
        let (path, opts, db) = parse(&["a.exe", "--listing"]).unwrap();
        assert_eq!(path, "a.exe");
        assert_eq!(opts.listing, Some(DEFAULT_LISTING_FUNCTIONS));
        assert_eq!(db, None);
        // A selective flag suppresses the default "dump everything".
        assert!(!opts.all && !opts.headers);

        let (_, opts, _) = parse(&["a.exe", "--listing=7"]).unwrap();
        assert_eq!(opts.listing, Some(7));

        let err = parse(&["a.exe", "--listing=x"]).unwrap_err();
        assert!(err.contains("invalid function count"), "{err}");
    }

    #[test]
    fn db_flag_parses_both_spellings_and_is_not_a_selection() {
        let (_, opts, db) = parse(&["a.exe", "--db", "notes.ann", "--listing"]).unwrap();
        assert_eq!(db.as_deref(), Some("notes.ann"));
        assert_eq!(opts.listing, Some(DEFAULT_LISTING_FUNCTIONS));

        let (_, opts, db) = parse(&["a.exe", "--db=notes.ann"]).unwrap();
        assert_eq!(db.as_deref(), Some("notes.ann"));
        // `--db` alone selects nothing, so the default dump still applies.
        assert!(opts.all && opts.headers, "{opts:?}");

        let err = parse(&["a.exe", "--db"]).unwrap_err();
        assert!(err.contains("`--db` requires a path"), "{err}");
    }

    #[test]
    fn a_bad_annotation_file_is_a_clean_error() {
        let err = annotate::Db::parse("not a aletheia file\n").unwrap_err();
        assert!(format!("{err}").contains("not a aletheia annotation file"), "{err}");
    }

    // -- real binaries (best effort) -------------------------------------

    /// Smoke-test against real ELF binaries when present (they won't be on
    /// macOS, where system binaries are Mach-O; the synthetic coverage
    /// above then stands alone).
    #[test]
    fn smoke_dumps_real_elf_if_present() {
        let candidates = [
            "/lib64/ld-linux-x86-64.so.2",
            "/usr/lib/x86_64-linux-gnu/libc.so.6",
            "/lib/ld-musl-x86_64.so.1",
            "/usr/bin/env",
            "/bin/ls",
        ];
        for path in candidates {
            let Ok(data) = std::fs::read(path) else {
                continue;
            };
            if sniff_format(&data) != Some(Format::Elf) {
                continue; // not ELF (e.g. Mach-O on macOS)
            }
            if data.len() < 6 || data[4] != elf::ELFCLASS64 || data[5] != elf::ELFDATA2LSB {
                continue; // outside the parser's scope
            }
            let out = dump_to_string(&data, opts_all())
                .unwrap_or_else(|e| panic!("{path}: {e}"));
            assert!(out.contains("ELF64 image"), "{path}");
            assert!(out.contains("PROGRAM HEADERS"), "{path}");
        }
    }

    /// Smoke-test against real Mach-O binaries when present (macOS hosts;
    /// elsewhere the paths are ELF or absent and the test is a no-op).
    #[test]
    fn smoke_dumps_real_macho_if_present() {
        for path in ["/bin/ls", "/usr/lib/dyld"] {
            let Ok(data) = std::fs::read(path) else {
                continue;
            };
            match sniff_format(&data) {
                Some(Format::MachO) | Some(Format::MachOFat) => {}
                _ => continue,
            }
            let out = dump_to_string(&data, opts_all())
                .unwrap_or_else(|e| panic!("{path}: {e}"));
            assert!(out.contains("__TEXT"), "{path}: {out}");
            assert!(out.contains("SYMBOLS"), "{path}");
            // Every macOS binary links libSystem (dyld being the exception
            // that proves the rule; it still has an LC_LOAD_DYLIB or none).
            if out.contains("Mach-O universal") {
                assert!(out.contains("==== slice 0:"), "{path}");
            }
        }
    }
}
