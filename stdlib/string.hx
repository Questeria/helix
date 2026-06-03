// =====================================================================
// Helix v1.2 rich String (charter docs/HELIX_COMPLETION.md §1.6 item H-2).
// Pure-Helix, arena-backed, NO kovc.hx change -> fixpoint-safe (library-
// level; touches NO struct-field STORE -- every mutable byte/header cell
// lives in the arena reached via __arena_push/__arena_get/__arena_set/
// __arena_len, and the String newtype is a read-only field struct, exactly
// the monomorphization surface H-1's Vec<T>/HashMap newtypes already ride).
// IDENTICAL fixpoint-safety argument to H-1's collections.hx (byte-identical
// to the H-3 mint bdff0049...).
//
// API (matches the charter §1.6 H-2 done-criterion):
//   str_new() -> i32                    -- handle of a fresh empty String;
//                                          wrap at the call site as
//                                          String { h: str_new() }
//   str_push_byte(s: String, b) -> i32  -- append one byte (grows on full)
//   str_len(s: String) -> i32           -- byte length
//   str_byte_at(s: String, i) -> i32    -- the i-th byte (0-based)
//   str_concat(a: String, b: String) -> i32   -- HANDLE of a fresh String =
//                                          a's bytes ++ b's; wrap at the call
//                                          site as String { h: str_concat(..) }
//   str_eq(a: String, b: String) -> i32 -- 1 iff same length AND same bytes
//
// WHY str_new / str_concat RETURN A HANDLE (i32), not a String value:
// returning a struct BY VALUE from a function is a known latent codegen gap
// in this from-raw compiler (a returned newtype/payload struct mis-lowers ->
// SIGSEGV; the same class as the tracker's arm_enum_payload3 enum-return-by-
// value SIGILL, a v-next fix). So -- EXACTLY as H-1's vec_with_cap returns
// the handle and the corpus wraps Vec::<i32>{ h: .. } -- the String
// constructors return the arena handle and the caller wraps it in the
// String newtype. Passing a String BY VALUE as a parameter (push_byte/len/
// byte_at/eq/concat args) is fully supported; only struct RETURN is avoided.
//
// WHY BYTE-BY-BYTE (not "from a string literal"): kovc lowers a string
// LITERAL used as a runtime value to `mov eax, 0` (kovc.hx:9575-9582 --
// "strings are only meaningful as the first arg of a file/strlen builtin"),
// so a literal exposes NO runtime byte pointer; the compile-time fold
// __strlen("...") (kovc.hx:5648) yields only a literal's byte LENGTH. So a
// rich String is built by appending byte values, and str_eq/str_concat
// operate purely over the arena-stored bytes at RUNTIME (nothing folds).
//
// MODEL (identical to stdlib/collections.hx Vec<T>): the arena is a single
// global append-only store; a String is a HANDLE = the arena index of its
// 3-cell header; growth allocates a fresh 2x byte region with __arena_push
// and relocates the live bytes via __arena_set. Each byte occupies one i32
// cell (low 8 bits meaningful; str_push_raw masks with & 255).
// =====================================================================

// Header block (3 contiguous arena cells at handle h):
//   arena[h+0] = len    (# bytes in use)
//   arena[h+1] = cap    (capacity of the current byte region)
//   arena[h+2] = data   (arena index of byte[0]; bytes at arena[data..data+cap))
struct String { h: i32 }

// Allocate an empty byte region with initial capacity cap0 (>=1); push the 3
// header cells + cap0 zero byte cells; fill cap+data. Returns the handle.
fn str_with_cap(cap0: i32) -> i32 {
    let h = __arena_push(0);    // h+0 : len  = 0
    __arena_push(0);            // h+1 : cap  (filled below)
    __arena_push(0);            // h+2 : data (filled below)
    let data = __arena_len();   // first byte cell = current cursor
    let mut i = 0;
    while i < cap0 {
        __arena_push(0);
        i = i + 1;
    };
    __arena_set(h + 1, cap0);
    __arena_set(h + 2, data);
    h
}

// Grow a full String: allocate a 2x byte region, copy the live bytes over,
// repoint cap+data. Only called by str_push_raw when len==cap.
fn str_grow(h: i32) -> i32 {
    let len = __arena_get(h);
    let cap = __arena_get(h + 1);
    let old = __arena_get(h + 2);
    let ncap = cap * 2;
    let ndata = __arena_len();
    let mut i = 0;
    while i < ncap {
        __arena_push(0);
        i = i + 1;
    };
    let mut j = 0;
    while j < len {
        __arena_set(ndata + j, __arena_get(old + j));
        j = j + 1;
    };
    __arena_set(h + 1, ncap);
    __arena_set(h + 2, ndata);
    0
}

// Append byte b (low 8 bits) to the String at handle h, growing if full.
fn str_push_raw(h: i32, b: i32) -> i32 {
    let len = __arena_get(h);
    let cap = __arena_get(h + 1);
    if len >= cap {
        str_grow(h);
    };
    let data = __arena_get(h + 2);
    __arena_set(data + len, b & 255);
    __arena_set(h, len + 1);
    0
}

// ---- the charter §1.6 H-2 public API ------------------------------------

// str_new() -> handle of a fresh empty String (initial cap 4, grows on demand).
// Wrap at the call site: `let s = String { h: str_new() };`.
fn str_new() -> i32 { str_with_cap(4) }

// str_push_byte(s, b): append one byte to s. Returns 0.
fn str_push_byte(s: String, b: i32) -> i32 { str_push_raw(s.h, b) }

// str_len(s): the byte length.
fn str_len(s: String) -> i32 { __arena_get(s.h) }

// str_byte_at(s, i): the i-th byte (0-based). Caller ensures 0<=i<len.
fn str_byte_at(s: String, i: i32) -> i32 {
    __arena_get(__arena_get(s.h + 2) + i)
}

// str_concat(a, b) -> handle of a FRESH String = a's bytes followed by b's.
// Neither input is mutated. Wrap: `let c = String { h: str_concat(a, b) };`.
fn str_concat(a: String, b: String) -> i32 {
    let rh = str_new();
    let la = str_len(a);
    let mut i = 0;
    while i < la {
        str_push_raw(rh, str_byte_at(a, i));
        i = i + 1;
    };
    let lb = str_len(b);
    let mut j = 0;
    while j < lb {
        str_push_raw(rh, str_byte_at(b, j));
        j = j + 1;
    };
    rh
}

// str_eq(a, b): 1 iff a and b have the SAME length AND the SAME bytes, else 0.
// Short-circuits on a length mismatch, then compares byte-for-byte; the first
// differing byte yields 0 (no false "equal" on a prefix match).
fn str_eq(a: String, b: String) -> i32 {
    let la = str_len(a);
    let lb = str_len(b);
    if la != lb {
        0
    } else {
        let mut i = 0;
        let mut same = 1;
        while i < la {
            if str_byte_at(a, i) != str_byte_at(b, i) {
                same = 0;
            };
            i = i + 1;
        };
        same
    }
}

// Convenience method form (mirrors H-1's impl<T> Vec<T> -- the SAME read-only
// newtype field-struct surface; self is passed BY VALUE, which is supported).
// The receiver must be a struct-LITERAL binding (e.g. `let s = String { h:
// str_new() };`), exactly as H1_vec's `let v = Vec::<i32>{..}; v.push(..)`.
impl String {
    fn len(self) -> i32 { __arena_get(self.h) }
    fn push_byte(self, b: i32) -> i32 { str_push_raw(self.h, b) }
    fn byte_at(self, i: i32) -> i32 { __arena_get(__arena_get(self.h + 2) + i) }
    fn eq(self, other: String) -> i32 { str_eq(self, other) }
}
