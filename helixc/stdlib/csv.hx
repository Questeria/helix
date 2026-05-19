// helixc/stdlib/csv.hx — line + field iteration over arena-backed CSV blobs.
//
// Stage 55 Inc 6: stdlib-only line/field iterator built on the Inc 1-5
// runtime string primitives. No compiler changes; pure Helix source.
//
// Convention: caller has already loaded a CSV blob into arena bytes
// [blob_start..blob_start+blob_len) (Inc 3 file-I/O will provide this
// at runtime once shipped; until then, `__strlit_to_arena("...")`
// supplies literal test fixtures).
//
// Iteration model: NO closures (Phase-0 Helix has no first-class
// functions). Caller threads an `offset` cursor through the API:
//
//   let mut off: i32 = 0;
//   while csv_has_next_line(blob, blob_len, off) == 1 {
//       let line_off = off;
//       let line_len = csv_line_len(blob, blob_len, off);
//       // ... process line ...
//       off = csv_next_line_offset(blob, blob_len, off);
//   }
//
// Same pattern for fields within a line (using csv_field_len /
// csv_next_field_offset, splitting on b',' = 44).
//
// IMPORTANT — 256-byte chunk cap:
//   `__str_find_byte` is compile-time-unrolled to MAX_SCAN=256, so a
//   single call sees at most 256 bytes starting at the given offset.
//   For lines longer than 256 bytes, csv_line_len returns the full
//   length by chaining multiple find_byte calls (up to 4 chunks =
//   1024-byte max line). Lines longer than 1024 bytes are truncated
//   for the line_len accounting; raise MAX_CHUNKS below if needed.
//
// API:
//   csv_has_next_line(blob, blob_len, off)   -> i32   1 if off < blob_len, 0 otherwise
//   csv_line_len(blob, blob_len, off)        -> i32   bytes from off up to (but not including)
//                                                     next '\n', or to blob_len if no '\n'
//   csv_next_line_offset(blob, blob_len, off) -> i32  offset of NEXT line (past '\n');
//                                                     equal to blob_len when last line
//   csv_field_len(line_off, line_len, foff)  -> i32   bytes from foff to next ',' (44)
//                                                     within the line, or to line_len
//   csv_next_field_offset(line_off, line_len, foff) -> i32  offset of next field within line
//   csv_count_lines(blob, blob_len)          -> i32   total number of lines (newline-delimited;
//                                                     a trailing non-newline-terminated line
//                                                     still counts as 1)
//   csv_count_fields(line_off, line_len)     -> i32   field count (commas + 1)
//
// License: Apache 2.0

@pure
fn csv_has_next_line(blob: i32, blob_len: i32, off: i32) -> i32 {
    if off < blob_len { 1 } else { 0 }
}

// csv_line_len: length of the line starting at off, up to (but not
// including) the next '\n' byte. Uses chained __str_find_byte calls
// to overcome the per-call 256-byte scan cap. MAX_CHUNKS=4 gives
// 1024-byte line ceiling; longer lines are truncated for accounting.
@pure
fn csv_line_len(blob: i32, blob_len: i32, off: i32) -> i32 {
    if off >= blob_len { 0 }
    else {
        let remaining_total = blob_len - off;
        // Chunk 1: scan up to 256 bytes from `off`.
        let r1 = __str_find_byte(blob + off, 256, 10);
        if r1 >= 0 {
            if r1 < remaining_total { r1 } else { remaining_total }
        } else {
            // No newline in first 256 bytes. Scan next chunk.
            if remaining_total <= 256 { remaining_total }
            else {
                let r2 = __str_find_byte(blob + off + 256, 256, 10);
                if r2 >= 0 {
                    let cand2 = 256 + r2;
                    if cand2 < remaining_total { cand2 } else { remaining_total }
                } else {
                    if remaining_total <= 512 { remaining_total }
                    else {
                        let r3 = __str_find_byte(blob + off + 512, 256, 10);
                        if r3 >= 0 {
                            let cand3 = 512 + r3;
                            if cand3 < remaining_total { cand3 } else { remaining_total }
                        } else {
                            if remaining_total <= 768 { remaining_total }
                            else {
                                let r4 = __str_find_byte(blob + off + 768, 256, 10);
                                if r4 >= 0 {
                                    let cand4 = 768 + r4;
                                    if cand4 < remaining_total { cand4 } else { remaining_total }
                                } else {
                                    // No newline in MAX_CHUNKS=4 chunks; treat
                                    // remainder up to blob_len OR 1024 as the
                                    // line (whichever smaller).
                                    if remaining_total < 1024 { remaining_total } else { 1024 }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Cycle 2 Batch RT fix batch 17 (silent-failure MEDIUM-6):
// csv_line_len silently truncates lines > 1024 bytes to exactly 1024,
// indistinguishable from "line is exactly 1024 bytes." Caller can't
// detect truncation; csv_next_line_offset then advances by truncated
// length without consuming the actual newline, misaligning the
// iterator and emitting garbage "lines" sliced from the remainder.
// Post-fix: csv_line_was_truncated() reports whether the LAST
// csv_line_len call would have truncated (caller passes the result
// + the same blob/off; if line_len == 1024 AND no newline was found
// in the 1024-byte window, it was truncated). Best-effort
// disambiguation without changing csv_line_len's signature.
@pure
fn csv_line_was_truncated(blob: i32, blob_len: i32, off: i32, returned_len: i32) -> i32 {
    if returned_len != 1024 { 0 }
    else {
        if off + 1024 > blob_len { 0 }
        else {
            let byte_at_1024 = __str_byte_at(blob, off + 1024);
            if byte_at_1024 == 10 { 0 } else { 1 }
        }
    }
}

// csv_next_line_offset: skip the current line + its '\n' terminator.
// If the current line is the last (no trailing newline), returns
// blob_len so that csv_has_next_line returns 0 on the next iteration.
@pure
fn csv_next_line_offset(blob: i32, blob_len: i32, off: i32) -> i32 {
    let llen = csv_line_len(blob, blob_len, off);
    let end = off + llen;
    if end >= blob_len { blob_len }
    else {
        // arena[blob + end] is the '\n' (or it's at a 1024-byte line
        // truncation — in that case, advance just by llen without
        // consuming the non-existent newline).
        if __str_byte_at(blob + end, 0) == 10 { end + 1 } else { end }
    }
}

// csv_field_len: bytes from foff up to (but not including) the next
// ',' (byte 44) within the line, or to line_len if no comma. The
// caller is responsible for ensuring foff < line_len.
@pure
fn csv_field_len(line_off: i32, line_len: i32, foff: i32) -> i32 {
    if foff >= line_len { 0 }
    else {
        let remaining = line_len - foff;
        let scan_len = if remaining > 256 { 256 } else { remaining };
        let r = __str_find_byte(line_off + foff, scan_len, 44);
        if r >= 0 { r } else { remaining }
    }
}

@pure
fn csv_next_field_offset(line_off: i32, line_len: i32, foff: i32) -> i32 {
    let flen = csv_field_len(line_off, line_len, foff);
    let end = foff + flen;
    if end >= line_len { line_len }
    else { end + 1 }
}

// csv_count_lines: walk the blob once via csv_next_line_offset to
// tally lines. A trailing non-newline-terminated line counts as 1.
@pure
fn csv_count_lines(blob: i32, blob_len: i32) -> i32 {
    let mut off: i32 = 0;
    let mut n: i32 = 0;
    // Hard cap on iterations to bound compile-time-loop safety.
    // Phase-0 csv use case: small datasets; 65536 lines is generous.
    let mut guard: i32 = 0;
    while off < blob_len {
        if guard >= 65536 { off = blob_len; }
        else {
            n = n + 1;
            off = csv_next_line_offset(blob, blob_len, off);
            guard = guard + 1;
        }
    }
    n
}

// csv_count_fields: commas-in-line + 1, with the same 256-byte scan
// caveat for very wide lines.
@pure
fn csv_count_fields(line_off: i32, line_len: i32) -> i32 {
    if line_len == 0 { 0 }
    else {
        let mut foff: i32 = 0;
        let mut n: i32 = 0;
        let mut guard: i32 = 0;
        while foff < line_len {
            if guard >= 65536 { foff = line_len; }
            else {
                n = n + 1;
                foff = csv_next_field_offset(line_off, line_len, foff);
                guard = guard + 1;
            }
        }
        n
    }
}

// csv_parse_field_i32: thin shim composing csv_field_len + __parse_i32.
// Skips this round-trip for the common "tabular numeric data" case.
@pure
fn csv_parse_field_i32(line_off: i32, line_len: i32, foff: i32) -> i32 {
    let flen = csv_field_len(line_off, line_len, foff);
    __parse_i32(line_off + foff, flen)
}
