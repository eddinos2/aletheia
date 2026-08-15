# PLAN_CHAINED — Mach-O chained fixups + Apple LC detect

## Goal

Parse `LC_DYLD_CHAINED_FIXUPS` (public dyld layouts): header, imports
table, symbol names. Detect `LC_ENCRYPTION_INFO_64` and
`LC_CODE_SIGNATURE` and surface them so analysis can refuse loudly on
encrypted text. First import-slot wiring from bind imports where the
starts table is walkable.

## Module

Extend `src/macho.rs`; optional thin `src/macho/chained.rs` if size
warrants. Update `Image::import_slots` for Mach-O when fixups present.

## Caps

`MAX_CHAINED_IMPORTS`, bounded table preflight like ELF `check_table`.

## Exit

Synthetic fixture with header+imports parses; encrypted LC sets a flag;
unknown hostile counts refuse without allocating.
