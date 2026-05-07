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

@pure
fn string_get(start: i32, i: i32) -> i32 {
    __arena_get(start + i)
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
    if n == 0 {
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
    }
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
        let mut acc: i32 = 0;
        while i < len {
            let b: i32 = __arena_get(start + i);
            if b >= 48 {
                if b <= 57 {
                    acc = acc * 10 + (b - 48);
                }
            }
            i = i + 1;
        }
        if neg == 1 { 0 - acc } else { acc }
    }
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
