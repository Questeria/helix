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
