// helixc/stdlib/string.hx — arena-backed byte string.
//
// Phase 1.9 follow-on: a "carry-pair" String, mirroring the Vec convention.
// Each i32 arena slot holds one ASCII byte (lower 8 bits). Caller threads
// (start, len) through every call. Trades space for indexing simplicity —
// AGI work is dominated by structured tensors / arenas, not string-heavy
// code, so 1-byte-per-slot is acceptable for a Phase 1 stdlib.
//
// Convention: build one String at a time; interleaved pushes from two
// strings would interleave bytes because the arena is global.
//
// API:
//   string_new()                            -> i32   start = current arena length
//   string_push(start, len, b)              -> i32   pushes byte; returns new len
//   string_get(start, i)                    -> i32   byte at index i (lower 8 bits)
//   string_eq(a, an, b, bn)                 -> i32   1 if equal, 0 otherwise
//   string_index_of(start, len, byte)       -> i32   first index of byte, -1 if missing
//   string_starts_with(s, sn, p, pn)        -> i32   1 if s starts with p
//   string_from_int(n)                      -> i32   appends ASCII digits of n to arena;
//                                                    returns count of bytes written.
//                                                    Caller saves __arena_len() BEFORE
//                                                    calling to recover the start index.
//   string_to_int(start, len)               -> i32   parses decimal int from slice;
//                                                    accepts optional leading '-';
//                                                    non-digit bytes are skipped silently.
//   string_ends_with(s, sn, suf, sufn)      -> i32   1 if s ends with suf, 0 otherwise.
//   string_count_byte(start, len, byte)     -> i32   number of occurrences of byte in slice.
//   string_last_index_of(start, len, byte)  -> i32   last index of byte, -1 if missing.
//   string_concat(a, an, b, bn)             -> i32   allocate new copy of a ++ b.
//   string_substring(start, len, off, n)    -> i32   allocate new copy of [off, off+n).
//   string_compare(a, an, b, bn)            -> i32   lex order: -1 / 0 / 1.
//
// License: Apache 2.0

@pure
fn string_new() -> i32 {
    __arena_len()
}

fn string_push(start: i32, len: i32, b: i32) -> i32 {
    __arena_push(b);
    len + 1
}

// DEPRECATED for new code (batch 15 deprecation sweep):
// No bounds check. Use string_get_checked(start, len, i) in
// safety-critical code. Retained for backward compat.
@pure
fn string_get(start: i32, i: i32) -> i32 {
    __arena_get(start + i)
}

// Cycle 1 Batch RT fix batch 11 (silent-failure HIGH-1):
// Bounds-checked variant of string_get. Pre-fix `string_get` had ZERO
// bounds checking — OOB indices returned whatever arena bytes were
// there (possibly another string's data, a tensor footer magic,
// freed-but-overwritten state). Tier-S users (security/AGI safety)
// treated the return as authoritative when iterating beyond len.
// Post-fix: this new variant takes the length explicitly + returns
// -1 sentinel on OOB. Callers that want safety pass the len; existing
// callers of `string_get` are unchanged (preserves backward compat).
@pure
fn string_get_checked(start: i32, len: i32, i: i32) -> i32 {
    if i < 0 { 0 - 1 }
    else {
        if i >= len { 0 - 1 }
        else { __arena_get(start + i) }
    }
}

// ============================================================
// Cycle 1 Batch RT fix batch 13 (type-design HIGH-1):
// Sibling-container magic+footer+ok invariant for String.
// Same defect class + fix as vec.hx — original string_new() /
// string_push() / string_get() unchanged for backward compat;
// safety-critical callers migrate to *_checked variants.
// ============================================================

// Batch 14 (MEDIUM-F): respaced to avoid bit-flip collision with
// vec_magic (7007012). Adjacent magic values are a one-bit-error
// silent-corruption hazard.
@pure fn string_magic() -> i32 { 7007013 }
@pure fn string_footer(cap: i32) -> i32 { 0 - string_magic() - cap }

fn string_new_checked(cap: i32) -> i32 {
    if cap <= 0 { 0 - 1 }
    else { if cap > 2147483647 - 3 { 0 - 1 }
    else {
        __arena_push(string_magic());
        __arena_push(cap);
        let start = __arena_len();
        let mut i: i32 = 0;
        while i < cap {
            __arena_push(0);
            i = i + 1;
        }
        __arena_push(string_footer(cap));
        start
    } }
}

@pure
fn string_ok(start: i32, cap: i32, len: i32) -> i32 {
    if cap <= 0 { 0 }
    else { if len < 0 { 0 }
    else { if len > cap { 0 }
    else { if start < 2 { 0 }
    else { if __arena_get(start - 2) != string_magic() { 0 }
    else { if __arena_get(start - 1) != cap { 0 }
    else { if start + cap >= __arena_len() { 0 }
    else { if __arena_get(start + cap) != string_footer(cap) { 0 }
    else { if arena_span_in_tensor_payload(start - 2, cap + 3) != 0 { 0 }
    else { 1 } } } } } } } } }
}

@pure
fn string_eq(a: i32, an: i32, b: i32, bn: i32) -> i32 {
    if an != bn { 0 }
    else {
        let mut i: i32 = 0;
        let mut eq: i32 = 1;
        while i < an {
            if __arena_get(a + i) != __arena_get(b + i) { eq = 0; }
            i = i + 1;
        }
        eq
    }
}

@pure
fn string_index_of(start: i32, len: i32, byte: i32) -> i32 {
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < len {
        if __arena_get(start + i) == byte {
            if found < 0 { found = i; }
        }
        i = i + 1;
    }
    found
}

@pure
fn string_starts_with(s: i32, sn: i32, p: i32, pn: i32) -> i32 {
    if sn < pn { 0 }
    else {
        let mut i: i32 = 0;
        let mut ok: i32 = 1;
        while i < pn {
            if __arena_get(s + i) != __arena_get(p + i) { ok = 0; }
            i = i + 1;
        }
        ok
    }
}

fn string_from_int(n: i32) -> i32 {
    // Restart 50 A1: INT32_MIN special case. `x = 0 - x` would wrap
    // back to INT32_MIN for n == -2147483648, the digit-count loop
    // would terminate immediately (count = 0), and the function would
    // return rc=1 with only the byte '-' written. Hard-code the 11
    // bytes for "-2147483648" so the contract "writes the printable
    // decimal of n" holds for the entire i32 domain.
    if n == 0 - 2147483647 - 1 {
        __arena_push(45);   // '-'
        __arena_push(50);   // '2'
        __arena_push(49);   // '1'
        __arena_push(52);   // '4'
        __arena_push(55);   // '7'
        __arena_push(52);   // '4'
        __arena_push(56);   // '8'
        __arena_push(51);   // '3'
        __arena_push(54);   // '6'
        __arena_push(52);   // '4'
        __arena_push(56);   // '8'
        11
    } else { if n == 0 {
        __arena_push(48);
        1
    } else {
        let mut x: i32 = n;
        let mut neg: i32 = 0;
        if x < 0 {
            neg = 1;
            x = 0 - x;
        }
        if neg == 1 {
            __arena_push(45);
        }
        let mut tmp: i32 = x;
        let mut count: i32 = 0;
        while tmp > 0 {
            count = count + 1;
            tmp = tmp / 10;
        }
        let digits_start: i32 = __arena_len();
        let mut i: i32 = 0;
        while i < count {
            __arena_push(0);
            i = i + 1;
        }
        let mut j: i32 = 0;
        while j < count {
            let digit: i32 = x % 10;
            __arena_set(digits_start + (count - 1 - j), 48 + digit);
            x = x / 10;
            j = j + 1;
        }
        if neg == 1 { count + 1 } else { count }
    } }
}

@pure
fn string_ends_with(s: i32, sn: i32, suf: i32, sufn: i32) -> i32 {
    if sn < sufn { 0 }
    else {
        let off: i32 = sn - sufn;
        let mut i: i32 = 0;
        let mut ok: i32 = 1;
        while i < sufn {
            if __arena_get(s + off + i) != __arena_get(suf + i) { ok = 0; }
            i = i + 1;
        }
        ok
    }
}

@pure
fn string_count_byte(start: i32, len: i32, byte: i32) -> i32 {
    let mut i: i32 = 0;
    let mut n: i32 = 0;
    while i < len {
        if __arena_get(start + i) == byte { n = n + 1; }
        i = i + 1;
    }
    n
}

@pure
fn string_last_index_of(start: i32, len: i32, byte: i32) -> i32 {
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < len {
        if __arena_get(start + i) == byte { found = i; }
        i = i + 1;
    }
    found
}

@pure
fn string_to_int(start: i32, len: i32) -> i32 {
    if len == 0 { 0 }
    else {
        let mut i: i32 = 0;
        let mut neg: i32 = 0;
        let first: i32 = __arena_get(start);
        if first == 45 {
            neg = 1;
            i = 1;
        }
        // Restart 51 A3: i64 accumulator + saturation. Previously a 10-digit
        // input like "2147483648" silently wrapped acc * 10 + 8 back to INT32_MIN.
        let mut acc: i64 = 0_i64;
        while i < len {
            let b: i32 = __arena_get(start + i);
            if b >= 48 {
                if b <= 57 {
                    acc = acc * 10_i64 + ((b - 48) as i64);
                    if acc > 2147483647_i64 { acc = 2147483647_i64; }
                }
            }
            i = i + 1;
        }
        if neg == 1 {
            let neg_acc = 0_i64 - acc;
            let min_i32 = (0_i64 - 2147483647_i64) - 1_i64;
            if neg_acc < min_i32 { (0 - 2147483647) - 1 } else { neg_acc as i32 }
        } else {
            acc as i32
        }
    }
}

// Cycle 3 R1 fix batch 20 (RT HIGH-9): string_to_int_strict variants.
// Pre-fix: string_to_int silently skips non-digit bytes. "abc123" parses
// to 123. "12.5" parses to 125. "" parses to 0. "-" parses to 0. Caller
// has zero feedback that the input was malformed.
//
// Post-fix:
//   - string_is_int(start, len) -> i32: returns 1 if ALL non-leading-sign
//     bytes are digits AND len > 0 (or len > 1 if leading sign); 0
//     otherwise. Caller pattern:
//       if string_is_int(s, n) == 1 { let v = string_to_int(s, n); ... }
//   - string_to_int_strict(start, len) -> i32: returns INT32_MIN sentinel
//     when input is malformed (non-digit byte found, or empty, or just "-").
//     Otherwise behaves like string_to_int (saturated).
//
// Caller-side discipline still required for negative numbers since the
// legitimate value INT32_MIN is itself the sentinel. For full
// disambiguation use string_is_int + string_to_int.
@pure
fn string_is_int(start: i32, len: i32) -> i32 {
    if len <= 0 { 0 }
    else {
        let first: i32 = __arena_get(start);
        let mut i: i32 = 0;
        if first == 45 { i = 1; }  // leading '-'
        if i >= len { 0 }  // just "-" or empty
        else {
            let mut ok: i32 = 1;
            while i < len {
                let b: i32 = __arena_get(start + i);
                if b < 48 { ok = 0; }
                else { if b > 57 { ok = 0; } }
                i = i + 1;
            }
            ok
        }
    }
}

@pure
fn string_to_int_strict(start: i32, len: i32) -> i32 {
    if string_is_int(start, len) == 0 { (0 - 2147483647) - 1 }
    else { string_to_int(start, len) }
}

// string_concat(a, an, b, bn): allocate a new arena-backed string
// that's a[0..an] followed by b[0..bn]. Returns the new start
// index; new len is an + bn. Mirrors vec_concat. NOT @pure (arena
// mutation). Useful for building messages / paths / labels without
// in-place mutation of either input.
fn string_concat(a: i32, an: i32, b: i32, bn: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < an {
        __arena_push(__arena_get(a + i));
        i = i + 1;
    }
    let mut j: i32 = 0;
    while j < bn {
        __arena_push(__arena_get(b + j));
        j = j + 1;
    }
    s
}

// string_substring(start, len, off, n): allocate a new arena-backed
// string copy of bytes [off, off+n). Saturates: off < 0 treated as
// 0, off >= len returns empty, off + n > len truncated to len - off.
// Returns the new start index; caller can reconstruct the saturated
// length as min(n, len - max(off, 0)) when needed. NOT @pure.
fn string_substring(start: i32, len: i32, off: i32, n: i32) -> i32 {
    let s: i32 = __arena_len();
    let off2 = if off < 0 { 0 } else { off };
    let avail = len - off2;
    let take = if avail < 0 { 0 }
               else { if n < 0 { 0 }
                      else { if n > avail { avail } else { n } } };
    let mut i: i32 = 0;
    while i < take {
        __arena_push(__arena_get(start + off2 + i));
        i = i + 1;
    }
    s
}

// string_compare(a, an, b, bn): lex order over byte values with
// length tiebreaker. Returns -1 if a < b, 0 if a == b, 1 if a > b.
// Mirrors libc memcmp/strcmp semantics. @pure.
@pure
fn string_compare(a: i32, an: i32, b: i32, bn: i32) -> i32 {
    let mut min: i32 = an;
    if bn < an { min = bn; }
    let mut i: i32 = 0;
    let mut diff: i32 = 0;
    let mut decided: i32 = 0;
    while i < min {
        if decided == 0 {
            let av = __arena_get(a + i);
            let bv = __arena_get(b + i);
            if av < bv { diff = 0 - 1; decided = 1; }
            else { if av > bv { diff = 1; decided = 1; } }
        }
        i = i + 1;
    }
    if decided == 1 { diff }
    else {
        if an < bn { 0 - 1 }
        else { if an > bn { 1 } else { 0 } }
    }
}

// string_contains(s, sn, pat, pn): does string `s` contain substring
// `pat`? Returns 1 if yes, 0 if no. O(sn * pn) naive search — fine
// for small strings; replace with KMP for production. Empty pattern
// matches anywhere (returns 1).
@pure
fn string_contains(s: i32, sn: i32, pat: i32, pn: i32) -> i32 {
    if pn == 0 { 1 }
    else { if pn > sn { 0 }
    else {
        let mut i: i32 = 0;
        let mut found: i32 = 0;
        let last_start = sn - pn;
        while i <= last_start {
            if found == 0 {
                let mut j: i32 = 0;
                let mut match_len: i32 = 0;
                while j < pn {
                    if __arena_get(s + i + j) == __arena_get(pat + j) {
                        match_len = match_len + 1;
                    };
                    j = j + 1;
                }
                if match_len == pn { found = 1; };
            };
            i = i + 1;
        }
        found
    } }
}

// string_replace_byte(start, len, from, to): allocate a new string
// where every occurrence of byte `from` is replaced with byte `to`.
// Original string untouched. Returns the new string's start; length
// stays the same so caller already knows.
fn string_replace_byte(start: i32, len: i32, from: i32, to: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < len {
        let b = __arena_get(start + i);
        if b == from { __arena_push(to); } else { __arena_push(b); }
        i = i + 1;
    }
    s
}

// string_to_upper(start, len): allocate a new string with ASCII
// lowercase letters (a-z = 97..122) converted to uppercase (A-Z =
// 65..90). Non-letter bytes pass through unchanged. UTF-8 multi-byte
// sequences are NOT handled — Phase-0 byte-level only.
fn string_to_upper(start: i32, len: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < len {
        let b = __arena_get(start + i);
        if b >= 97 {
            if b <= 122 { __arena_push(b - 32); }
            else { __arena_push(b); }
        } else {
            __arena_push(b);
        }
        i = i + 1;
    }
    s
}

// string_to_lower(start, len): allocate a new string with ASCII
// uppercase letters converted to lowercase. Inverse of string_to_upper.
fn string_to_lower(start: i32, len: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < len {
        let b = __arena_get(start + i);
        if b >= 65 {
            if b <= 90 { __arena_push(b + 32); }
            else { __arena_push(b); }
        } else {
            __arena_push(b);
        }
        i = i + 1;
    }
    s
}

// string_starts_with_byte(start, len, b): @pure. Check if the string's
// first byte equals b. Returns 1 if yes, 0 otherwise. Returns 0 for
// empty strings.
@pure
fn string_starts_with_byte(start: i32, len: i32, b: i32) -> i32 {
    if len == 0 { 0 }
    else { if __arena_get(start) == b { 1 } else { 0 } }
}

// string_ends_with_byte(start, len, b): @pure. Symmetric.
@pure
fn string_ends_with_byte(start: i32, len: i32, b: i32) -> i32 {
    if len == 0 { 0 }
    else { if __arena_get(start + len - 1) == b { 1 } else { 0 } }
}

// string_trim_left_byte(start, len, b): @pure. Returns the count of
// leading bytes equal to b (the "skip" count). Caller can compute
// the trimmed slice as (start + skip, len - skip). For trimming
// whitespace pass b=32 (ASCII space).
@pure
fn string_trim_left_byte(start: i32, len: i32, b: i32) -> i32 {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut result: i32 = len;
    while i < len {
        if found == 0 {
            if __arena_get(start + i) != b {
                result = i;
                found = 1;
            };
        };
        i = i + 1;
    }
    result
}

// string_trim_right_byte(start, len, b): @pure. Returns the trimmed
// length (len minus trailing-b count). Walks from the right, returns
// the first index from the right where the byte != b, plus 1.
@pure
fn string_trim_right_byte(start: i32, len: i32, b: i32) -> i32 {
    let mut i: i32 = len;
    let mut found: i32 = 0;
    let mut result: i32 = 0;
    while i > 0 {
        if found == 0 {
            if __arena_get(start + i - 1) != b {
                result = i;
                found = 1;
            };
        };
        i = i - 1;
    }
    result
}

// string_repeat(start, len, n): allocate a new string with the input
// repeated n times. n=0 returns an empty string at the current arena
// position. Output length = len * n.
fn string_repeat(start: i32, len: i32, n: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut k: i32 = 0;
    while k < n {
        let mut i: i32 = 0;
        while i < len {
            __arena_push(__arena_get(start + i));
            i = i + 1;
        }
        k = k + 1;
    }
    s
}

// string_split_first(start, len, delim): @pure. Returns the index
// of the first byte equal to delim, or -1 if absent. Useful for
// "split into key/value at first '=' " patterns.
@pure
fn string_split_first(start: i32, len: i32, delim: i32) -> i32 {
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < len {
        if found < 0 {
            if __arena_get(start + i) == delim { found = i; };
        };
        i = i + 1;
    }
    found
}

// string_count_byte_n(start, len, byte): @pure. Alias for the existing
// string_count_byte (kept for naming-symmetry with the count_eq family
// in iterators.hx). Just calls through.
@pure
fn string_count_byte_n(start: i32, len: i32, byte: i32) -> i32 {
    string_count_byte(start, len, byte)
}

// string_is_ascii(start, len): @pure. Returns 1 if every byte is in
// the 7-bit ASCII range (0..127), 0 otherwise. Useful gate for
// downstream code that assumes single-byte chars.
@pure
fn string_is_ascii(start: i32, len: i32) -> i32 {
    let mut i: i32 = 0;
    let mut ok: i32 = 1;
    while i < len {
        let b = __arena_get(start + i);
        if b > 127 { ok = 0; };
        if b < 0 { ok = 0; };
        i = i + 1;
    }
    ok
}

// string_is_digit_only(start, len): @pure. Returns 1 if every byte is
// '0'..'9' (48..57), 0 otherwise. Empty string returns 0 (no digits =
// not a number). Useful gate before string_to_int.
@pure
fn string_is_digit_only(start: i32, len: i32) -> i32 {
    if len == 0 { 0 }
    else {
        let mut i: i32 = 0;
        let mut ok: i32 = 1;
        while i < len {
            let b = __arena_get(start + i);
            if b < 48 { ok = 0; };
            if b > 57 { ok = 0; };
            i = i + 1;
        }
        ok
    }
}

// string_pad_left(start, len, pad_byte, target_len): allocate a new
// string left-padded with pad_byte to reach target_len. If len >=
// target_len, returns a copy without padding. Common use: align text
// in tabular output ("  42" with pad_byte=32).
fn string_pad_left(start: i32, len: i32, pad_byte: i32, target_len: i32) -> i32 {
    let s: i32 = __arena_len();
    if len >= target_len {
        // No padding; just copy.
        let mut i: i32 = 0;
        while i < len {
            __arena_push(__arena_get(start + i));
            i = i + 1;
        }
    } else {
        let pad_count = target_len - len;
        let mut i: i32 = 0;
        while i < pad_count {
            __arena_push(pad_byte);
            i = i + 1;
        }
        let mut j: i32 = 0;
        while j < len {
            __arena_push(__arena_get(start + j));
            j = j + 1;
        }
    }
    s
}

// string_pad_right(start, len, pad_byte, target_len): symmetric to
// pad_left — pads at the END.
fn string_pad_right(start: i32, len: i32, pad_byte: i32, target_len: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < len {
        __arena_push(__arena_get(start + i));
        i = i + 1;
    }
    if len < target_len {
        let pad_count = target_len - len;
        let mut j: i32 = 0;
        while j < pad_count {
            __arena_push(pad_byte);
            j = j + 1;
        }
    }
    s
}

// string_replace_first_byte(start, len, from, to): allocate a new
// string with the first occurrence of byte `from` replaced by `to`.
// If `from` is absent, returns an exact copy. Useful for "change
// first '=' to ' ' to split key/value differently" patterns.
fn string_replace_first_byte(start: i32, len: i32, from: i32, to: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    let mut replaced: i32 = 0;
    while i < len {
        let b = __arena_get(start + i);
        if replaced == 0 {
            if b == from { __arena_push(to); replaced = 1; }
            else { __arena_push(b); };
        } else { __arena_push(b); };
        i = i + 1;
    }
    s
}

// string_skip_n(start, len, n): @pure. Returns the offset to skip n
// bytes (or len if n > len). Caller computes (start + skip, len - skip)
// for the remaining slice. Equivalent to bounded saturating clamp.
@pure
fn string_skip_n(start: i32, len: i32, n: i32) -> i32 {
    if n < 0 { 0 }
    else { if n > len { len } else { n } }
}

// string_pad_center(start, len, pad_byte, target_len): allocate a new
// string with pad_byte added to BOTH sides, balanced as evenly as
// possible (extra byte goes RIGHT when total padding is odd). Returns
// a copy when len >= target_len.
fn string_pad_center(start: i32, len: i32, pad_byte: i32, target_len: i32) -> i32 {
    let s: i32 = __arena_len();
    if len >= target_len {
        let mut i: i32 = 0;
        while i < len {
            __arena_push(__arena_get(start + i));
            i = i + 1;
        }
    } else {
        let total_pad = target_len - len;
        let left_pad = total_pad / 2;
        let right_pad = total_pad - left_pad;
        let mut i: i32 = 0;
        while i < left_pad { __arena_push(pad_byte); i = i + 1; }
        let mut j: i32 = 0;
        while j < len { __arena_push(__arena_get(start + j)); j = j + 1; }
        let mut k: i32 = 0;
        while k < right_pad { __arena_push(pad_byte); k = k + 1; }
    }
    s
}

// string_translate_byte(start, len, from, to): allocate a new string
// where every occurrence of byte `from` becomes `to`. Same as
// string_replace_byte; alternate name for clarity in linguistic-style
// "translate one byte to another" call sites.
fn string_translate_byte(start: i32, len: i32, from: i32, to: i32) -> i32 {
    string_replace_byte(start, len, from, to)
}

// string_count_prefix(start, len, prefix_start, prefix_len): @pure.
// Count of leading bytes that match prefix as a sequence. If the
// string starts with the full prefix, returns prefix_len; otherwise
// returns the length of the longest matching initial segment.
@pure
fn string_count_prefix(start: i32, len: i32, prefix_start: i32, prefix_len: i32) -> i32 {
    let mut max_match = prefix_len;
    if len < max_match { max_match = len; }
    let mut i: i32 = 0;
    let mut matched: i32 = 0;
    while i < max_match {
        if matched == i {
            if __arena_get(start + i) == __arena_get(prefix_start + i) {
                matched = matched + 1;
            };
        };
        i = i + 1;
    }
    matched
}

// string_index_of_n(start, len, byte, n): @pure. Returns the index of
// the n-th occurrence of byte in the string, or -1 if absent. n is
// 0-indexed (n=0 is first match, n=1 is second, etc.). Useful for
// "split at the third '/'" patterns.
@pure
fn string_index_of_n(start: i32, len: i32, byte: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    let mut counted: i32 = 0;
    while i < len {
        if found < 0 {
            if __arena_get(start + i) == byte {
                if counted == n { found = i; }
                else { counted = counted + 1; };
            };
        };
        i = i + 1;
    }
    found
}

// string_count_lines(start, len): @pure. Count of newlines (byte 10).
// Useful as a precursor to splitting "lines" — caller can size a vec.
@pure
fn string_count_lines(start: i32, len: i32) -> i32 {
    string_count_byte(start, len, 10)
}

// string_eq_ignore_case_ascii(a, an, b, bn): @pure. Equality ignoring
// ASCII case. Bytes outside A-Z / a-z compare unchanged.
@pure
fn string_eq_ignore_case_ascii(a: i32, an: i32, b: i32, bn: i32) -> i32 {
    if an != bn { 0 }
    else {
        let mut i: i32 = 0;
        let mut eq: i32 = 1;
        while i < an {
            let av = __arena_get(a + i);
            let bv = __arena_get(b + i);
            // Normalize to lowercase for comparison.
            let mut an_v = av;
            if av >= 65 { if av <= 90 { an_v = av + 32; }; };
            let mut bn_v = bv;
            if bv >= 65 { if bv <= 90 { bn_v = bv + 32; }; };
            if an_v != bn_v { eq = 0; };
            i = i + 1;
        }
        eq
    }
}

// string_first_index_at_or_after(start, len, off, byte): @pure.
// Returns first index of byte at index >= off, or -1 if absent.
// Useful for repeated split passes.
@pure
fn string_first_index_at_or_after(start: i32, len: i32, off: i32, byte: i32) -> i32 {
    let mut i: i32 = if off < 0 { 0 } else { off };
    let mut found: i32 = 0 - 1;
    while i < len {
        if found < 0 {
            if __arena_get(start + i) == byte { found = i; };
        };
        i = i + 1;
    }
    found
}

// string_strip_byte(start, len, byte): allocate new string with all
// occurrences of byte removed. Useful for "strip whitespace" or
// "strip commas" patterns.
fn string_strip_byte(start: i32, len: i32, byte: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < len {
        let b = __arena_get(start + i);
        if b != byte { __arena_push(b); };
        i = i + 1;
    }
    s
}

// string_min_byte(start, len): @pure. Smallest byte value. 0 if empty.
@pure
fn string_min_byte(start: i32, len: i32) -> i32 {
    if len == 0 { 0 }
    else {
        let mut i: i32 = 1;
        let mut best: i32 = __arena_get(start);
        while i < len {
            let v = __arena_get(start + i);
            if v < best { best = v; }
            i = i + 1;
        }
        best
    }
}

// string_max_byte(start, len): @pure. Largest byte value. 0 if empty.
@pure
fn string_max_byte(start: i32, len: i32) -> i32 {
    if len == 0 { 0 }
    else {
        let mut i: i32 = 1;
        let mut best: i32 = __arena_get(start);
        while i < len {
            let v = __arena_get(start + i);
            if v > best { best = v; }
            i = i + 1;
        }
        best
    }
}

// string_first_byte(start, len): @pure. v[0] or 0 if empty.
@pure
fn string_first_byte(start: i32, len: i32) -> i32 {
    if len == 0 { 0 } else { __arena_get(start) }
}

// string_last_byte(start, len): @pure. v[len-1] or 0 if empty.
@pure
fn string_last_byte(start: i32, len: i32) -> i32 {
    if len == 0 { 0 } else { __arena_get(start + len - 1) }
}

// string_len_after_trim_left(start, len, byte): @pure. Length remaining
// after trimming leading occurrences of byte. Pair with trim_left_byte
// for offset+remaining-length pattern.
@pure
fn string_len_after_trim_left(start: i32, len: i32, byte: i32) -> i32 {
    len - string_trim_left_byte(start, len, byte)
}

// string_is_empty(start, len): @pure. 1 if len == 0.
@pure
fn string_is_empty(start: i32, len: i32) -> i32 {
    if len == 0 { 1 } else { 0 }
}

// string_byte_count(start, len): @pure. Same as len; alias for clarity
// in API call sites that operate on multiple strings.
@pure
fn string_byte_count(start: i32, len: i32) -> i32 {
    len
}

// string_chars_eq_in_range(start, len, lo, hi): @pure. 1 if all bytes
// are in inclusive range [lo, hi], 0 otherwise. Empty string returns 1
// (vacuously true).
@pure
fn string_chars_eq_in_range(start: i32, len: i32, lo: i32, hi: i32) -> i32 {
    let mut i: i32 = 0;
    let mut ok: i32 = 1;
    while i < len {
        let b = __arena_get(start + i);
        if b < lo { ok = 0; }
        if b > hi { ok = 0; }
        i = i + 1;
    }
    ok
}

// string_count_ge_byte(start, len, byte): @pure. Count of bytes >= byte.
@pure
fn string_count_ge_byte(start: i32, len: i32, byte: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < len {
        if __arena_get(start + i) >= byte { total = total + 1; }
        i = i + 1;
    }
    total
}

// string_count_le_byte(start, len, byte): @pure. Count of bytes <= byte.
@pure
fn string_count_le_byte(start: i32, len: i32, byte: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < len {
        if __arena_get(start + i) <= byte { total = total + 1; }
        i = i + 1;
    }
    total
}

// string_byte_at_or(start, len, idx, default): @pure. Bounded-access
// version of string_get; returns default if idx is out of range.
@pure
fn string_byte_at_or(start: i32, len: i32, idx: i32, default: i32) -> i32 {
    if idx < 0 { default }
    else { if idx >= len { default } else { __arena_get(start + idx) } }
}

// string_eq_byte_at(start, len, idx, byte): @pure. 1 if v[idx] == byte
// AND idx in range. 0 otherwise (out of range OR mismatched).
@pure
fn string_eq_byte_at(start: i32, len: i32, idx: i32, byte: i32) -> i32 {
    if idx < 0 { 0 }
    else { if idx >= len { 0 }
    else { if __arena_get(start + idx) == byte { 1 } else { 0 } } }
}

// string_count_lt_byte(start, len, byte): @pure. Count of bytes < byte.
@pure
fn string_count_lt_byte(start: i32, len: i32, byte: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < len {
        if __arena_get(start + i) < byte { total = total + 1; }
        i = i + 1;
    }
    total
}

// string_count_gt_byte(start, len, byte): @pure. Count of bytes > byte.
@pure
fn string_count_gt_byte(start: i32, len: i32, byte: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < len {
        if __arena_get(start + i) > byte { total = total + 1; }
        i = i + 1;
    }
    total
}

// string_count_alpha(start, len): @pure. Count of ASCII letters
// (a-z + A-Z). Useful for text classification.
@pure
fn string_count_alpha(start: i32, len: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < len {
        let b = __arena_get(start + i);
        if b >= 65 {
            if b <= 90 { total = total + 1; }
            else { if b >= 97 { if b <= 122 { total = total + 1; }; }; };
        };
        i = i + 1;
    }
    total
}

// string_count_digit(start, len): @pure. Count of '0'..'9' (48..57).
@pure
fn string_count_digit(start: i32, len: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < len {
        let b = __arena_get(start + i);
        if b >= 48 { if b <= 57 { total = total + 1; }; };
        i = i + 1;
    }
    total
}

// Stage 55 Inc 2b — string_to_f64(start, len): @pure decimal-string
// to f64 parser. Composes existing __parse_i32 + bit-cast intrinsics.
//
// Accepts: optional leading '-', integer part (digits), optional '.',
// fractional part (digits). Examples:
//   "3.14" -> 3.14
//   "42"   -> 42.0
//   "-1.5" -> -1.5
//   "0.001" -> 0.001
//
// Non-digit bytes outside the optional sign or single '.' are treated
// as end-of-number (whatever was parsed up to that point is returned).
// Phase-0 limitations: no exponent notation (1e10), no NaN/Inf strings.
// Caller wraps with their own validation if those are needed.
@pure
fn string_to_f64(start: i32, len: i32) -> f64 {
    if len == 0 { 0.0_f64 }
    else {
        let first = __arena_get(start);
        let neg = if first == 45 { 1 } else { 0 };
        let body_start = if neg == 1 { start + 1 } else { start };
        let body_len = if neg == 1 { len - 1 } else { len };
        // Find decimal point within body.
        let dot_off = __str_find_byte(body_start, body_len, 46);
        let int_len = if dot_off < 0 { body_len } else { dot_off };
        let int_part = __parse_i32(body_start, int_len);
        let mut value: f64 = (int_part as f64);
        if dot_off >= 0 {
            let frac_start = body_start + dot_off + 1;
            let frac_len = body_len - dot_off - 1;
            if frac_len > 0 {
                let frac_int = __parse_i32(frac_start, frac_len);
                // Compute 10^frac_len as f64. Cap at 18 digits (f64
                // precision limit; beyond that, fractional bits drop
                // off anyway).
                let mut divisor: f64 = 1.0_f64;
                let mut i: i32 = 0;
                let cap = if frac_len > 18 { 18 } else { frac_len };
                while i < cap {
                    divisor = divisor * 10.0_f64;
                    i = i + 1;
                }
                value = value + (frac_int as f64) / divisor;
            }
        }
        if neg == 1 { 0.0_f64 - value } else { value }
    }
}
