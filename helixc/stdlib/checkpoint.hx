// helixc/stdlib/checkpoint.hx — save/load training-state checkpoints.
//
// Stage 61 (Tier 1 #4 Inc 7): pure-Helix stdlib for runtime-path
// checkpoint save + load, built on the 4 dyn file I/O builtins
// shipped at Stage 60:
//   read_file_to_arena_dyn(path_start, path_len) -> i32
//   write_file_to_arena_dyn(path_start, path_len,
//                            data_start, n_bytes) -> i32
//   read_file_int_dyn(path_start, path_len) -> i32
//   write_file_dyn(path_start, path_len,
//                   content_start, content_len) -> i32
//
// API (all paths are arena-backed `(start, len)` byte sequences,
// constructible via __strlit_to_arena or __str_concat_arena):
//
//   checkpoint_save_raw(path_s, path_l, data_s, data_n) -> i32
//     Save N bytes of arena data to a runtime-resolved path.
//     Returns count of bytes written (0 on open failure).
//
//   checkpoint_load_raw(path_s, path_l) -> i32
//     Load bytes from a runtime-resolved path into the arena.
//     Returns count of bytes loaded. Returns 0 on open failure.
//
//   checkpoint_save_versioned(path_s, path_l, magic, version,
//                              epoch, data_s, data_n) -> i32
//     Save with a 12-byte header (magic, version, epoch as 3 i32s)
//     followed by the data payload. Returns 1 on success, 0 on
//     failure. The header is pushed to the arena ahead of the
//     data payload via the standard arena ops before invoking
//     the raw write — caller must ensure the 3 header slots
//     immediately precede the data_s slot in the arena.
//
//   checkpoint_verify_header(arena_start, magic, version) -> i32
//     Verify the first 12 bytes (3 i32 slots) at arena_start match
//     the expected magic + version + (anything for epoch). Returns
//     1 if header is valid; 0 otherwise.
//
//   checkpoint_load_epoch(arena_start) -> i32
//     Extract epoch number from a previously-loaded versioned
//     checkpoint header (3rd i32 at arena_start).
//
// Cascade-safe: stdlib-only, no compiler changes. All file I/O
// gates exist via Stage 60 builtins.

@pure
fn checkpoint_save_raw(path_s: i32, path_l: i32,
                        data_s: i32, data_n: i32) -> i32 {
    write_file_to_arena_dyn(path_s, path_l, data_s, data_n)
}

@pure
fn checkpoint_load_raw(path_s: i32, path_l: i32) -> i32 {
    read_file_to_arena_dyn(path_s, path_l)
}

// Cycle 2 fix batch 18 (silent-failure MEDIUM-4):
// Pre-fix: checkpoint_save_raw + checkpoint_load_raw both returned 0
// for BOTH "open failure" (disk full / permission / invalid path) AND
// "legitimate 0-byte payload" (empty marker checkpoint / empty file).
// Tier-S training loop saves empty resumption-marker can't distinguish
// from save failure. Subsequent resume reads zero bytes and thinks
// state was wiped, when actually save failed.
//
// Post-fix:
//   - checkpoint_save_raw_strict: FULL fix. Returns -1 if data_n > 0
//     and the write returns 0 bytes (open or write failure). Returns 0
//     trivially for data_n == 0 (success-via-noop). Caller pattern:
//       let bytes_written = checkpoint_save_raw_strict(...);
//       if bytes_written < 0 { /* save failed */ }
//
//   - checkpoint_load_raw_strict: PARTIAL fix only. The underlying
//     `read_file_to_arena_dyn` builtin returns the byte count and has
//     NO out-of-band signal for "open failure" vs "empty file". This
//     function currently inherits that limitation — it returns the
//     result unchanged. Stage 110+ deferral: extend the builtin ABI
//     so open-fail returns -1 (distinguishable from 0-byte file).
//     UNTIL THEN: callers cannot distinguish "file is empty" from
//     "open failed" via this API. Use a separate file-stat check
//     (not yet in stdlib) for full disambiguation.
//
// Cycle 2 batch 19 doc-drift fix: removed misleading INT32_MIN
// promise from the load-side docstring. Previous wording would have
// led callers to write `if bytes_read == INT32_MIN { handle_error }`
// based on the contract, with a handler that NEVER fires — silent
// failure restored via doc drift. Audit caught it.
@pure
fn checkpoint_save_raw_strict(path_s: i32, path_l: i32,
                               data_s: i32, data_n: i32) -> i32 {
    // Trivial success for empty save (no write needed).
    if data_n <= 0 { 0 }
    else {
        let written = write_file_to_arena_dyn(path_s, path_l, data_s, data_n);
        // If we asked to write n>0 bytes and 0 were written, the open
        // or write failed. Return -1 to distinguish from the trivial
        // "n==0" success above.
        if written == 0 { 0 - 1 }
        else { written }
    }
}

@pure
fn checkpoint_load_raw_strict(path_s: i32, path_l: i32) -> i32 {
    let result = read_file_to_arena_dyn(path_s, path_l);
    // read_file_to_arena_dyn returns the byte count read; 0 conflates
    // "empty file" with "open failure". Without an out-of-band signal,
    // we can't fully disambiguate at this layer. Caller-side discipline:
    // check if the file EXISTS first via a separate stat call (not yet
    // in stdlib). For now, return result unchanged; doc the limitation.
    // Stage 110+ deferral: extend the read_file_to_arena_dyn ABI to
    // return -1 on open failure (distinguish from 0-byte file).
    result
}

// Versioned checkpoint helpers — minimum-viable header layout:
//   slot 0: magic (i32; caller-supplied tag like 0x48434b50 = "HCKP")
//   slot 1: version (i32; bump on format change)
//   slot 2: epoch (i32; training epoch number)
//   slots 3..: payload
//
// Phase-0 Helix has no i32-to-bytes serialization in stdlib (the
// arena stores one byte per i32 slot), so versioned save expects
// the caller to lay out the header bytes directly. The verify /
// load_epoch helpers read the same per-slot byte layout.

@pure
fn checkpoint_header_size() -> i32 {
    // 12 bytes (3 i32 fields encoded as 3 bytes each? No — one
    // arena slot per byte. For a 3-field header at 1 slot per
    // byte, the total is 12 slots = 12 bytes when written to disk).
    12
}

@pure
fn checkpoint_verify_magic(arena_get_byte_0: i32,
                            arena_get_byte_1: i32,
                            arena_get_byte_2: i32,
                            arena_get_byte_3: i32,
                            expected_magic: i32) -> i32 {
    // Reconstruct the LE 4-byte magic from arena slots 0..3 and
    // compare. Caller passes the 4 byte values explicitly because
    // Phase-0 Helix has no array-indexing builtin in @pure stdlib
    // that returns i32 directly (the arena ops are non-pure).
    let b0 = arena_get_byte_0 & 0xFF;
    let b1 = arena_get_byte_1 & 0xFF;
    let b2 = arena_get_byte_2 & 0xFF;
    let b3 = arena_get_byte_3 & 0xFF;
    let reconstructed = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    if reconstructed == expected_magic { 1 } else { 0 }
}
