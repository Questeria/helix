// =====================================================================
// Helix v1.2 standard collections (charter docs/HELIX_COMPLETION.md §1.6
// item H-1). Pure-Helix, arena-backed, NO kovc.hx change -> fixpoint-safe
// (library-level; touches no struct-field STORE -- all mutable state lives
// in arena cells reached via the __arena_get/__arena_set/__arena_push/
// __arena_len intrinsics, and the generic Vec<T> / HashMap newtypes are
// read-only field structs, exactly the monomorphization surface already
// gated by gen_vec_i32/gen_vec_f32 + the 4-fn vec_arena POC).
//
// API (matches the charter §1.6 H-1 done-criterion):
//   generic Vec<T>:  new / push / get / set / len / pop  (growth on push)
//   i32->i32 HashMap: new / insert / get / contains       (open addressing)
//
// MODEL: the arena (helix_arena_cap() cells, BSS-zeroed) is a single global
// append-only store: __arena_push(x) appends x and returns its index;
// __arena_get(i)/__arena_set(i,v) random-access any cell; __arena_len()
// is the current cursor. A collection is a HANDLE = the arena index of its
// header block; growth allocates a fresh (larger) region with __arena_push
// and relocates via __arena_set. Element/key/value width is one i32 cell
// (f32 round-trips through an i32 cell exactly as the gated gen_vec_f32
// program already proves).
// =====================================================================

// ---------------------------------------------------------------------
// Vec<T> -- a dynamic growable array.
//
// Header block (3 contiguous arena cells at handle h):
//   arena[h+0] = len        (# elements in use)
//   arena[h+1] = cap        (capacity of the current data region)
//   arena[h+2] = data       (arena index of element[0]; elements at
//                            arena[data .. data+cap))
// The generic Vec<T> newtype carries only the handle; T types get()/pop().
// ---------------------------------------------------------------------
struct Vec[T] { h: i32 }

// Allocate an empty Vec with an initial capacity of `cap0` (>=1). Pushes the
// 3 header cells, then cap0 zero-initialised data cells, then writes the
// header. Returns a Vec<T> wrapping the header handle.
fn vec_with_cap(cap0: i32) -> i32 {
    let h = __arena_push(0);    // h+0 : len  = 0
    __arena_push(0);            // h+1 : cap  (filled below)
    __arena_push(0);            // h+2 : data (filled below)
    let data = __arena_len();   // first data cell = current cursor
    let mut i = 0;
    while i < cap0 {
        __arena_push(0);
        i = i + 1;
    };
    __arena_set(h + 1, cap0);
    __arena_set(h + 2, data);
    h
}

// Grow a full Vec: allocate a 2x region, copy the live elements over, and
// repoint cap+data. Only called by vec_push_raw when len==cap.
fn vec_grow(h: i32) -> i32 {
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

// Append x (i32-width) to the Vec at handle h, growing if at capacity.
fn vec_push_raw(h: i32, x: i32) -> i32 {
    let len = __arena_get(h);
    let cap = __arena_get(h + 1);
    if len >= cap {
        vec_grow(h);
    };
    let data = __arena_get(h + 2);
    __arena_set(data + len, x);
    __arena_set(h, len + 1);
    0
}

impl<T> Vec<T> {
    fn len(self) -> i32 { __arena_get(self.h) }
    fn cap(self) -> i32 { __arena_get(self.h + 1) }
    fn get(self, i: i32) -> T { __arena_get(__arena_get(self.h + 2) + i) }
    fn set(self, i: i32, x: i32) -> i32 {
        __arena_set(__arena_get(self.h + 2) + i, x); 0
    }
    fn push(self, x: i32) -> i32 { vec_push_raw(self.h, x) }
    // pop: remove + return the last element (T). Decrements len first, then
    // reads the just-freed slot. Caller must ensure len>0.
    fn pop(self) -> T {
        let n = __arena_get(self.h) - 1;
        __arena_set(self.h, n);
        __arena_get(__arena_get(self.h + 2) + n)
    }
}

// ---------------------------------------------------------------------
// HashMap (i32 -> i32, open addressing with linear probing).
//
// Header block (3 contiguous arena cells at handle m):
//   arena[m+0] = count   (# occupied slots)
//   arena[m+1] = cap     (number of buckets; a power of two so that
//                         `key & (cap-1)` is the bucket index)
//   arena[m+2] = slots   (arena index of bucket[0]; each bucket is a
//                         3-cell triple [state, key, val] laid out
//                         contiguously, so bucket b is at slots + b*3)
// state: 0 = empty, 1 = occupied. cap is fixed (no rehash in v1.2 -- the
// charter asks for insert/get/contains over open addressing; size the map
// at creation). hm_get returns HM_MISSING for an absent key.
// ---------------------------------------------------------------------
fn hm_missing() -> i32 { 0 - 1 }   // sentinel returned by hm_get on a miss

// Create a HashMap with `cap` buckets (cap MUST be a power of two, >=2).
// Allocates the 3 header cells + cap*3 zero bucket cells; state starts 0
// (empty) by BSS-zero. Returns the header handle.
fn hm_new(cap: i32) -> i32 {
    let m = __arena_push(0);    // m+0 : count = 0
    __arena_push(0);            // m+1 : cap
    __arena_push(0);            // m+2 : slots
    let slots = __arena_len();
    let mut i = 0;
    while i < cap * 3 {
        __arena_push(0);
        i = i + 1;
    };
    __arena_set(m + 1, cap);
    __arena_set(m + 2, slots);
    m
}

// Insert (k -> v). If k is already present its value is overwritten.
// Linear-probe from bucket (k & (cap-1)) until an empty slot or a matching
// key is found. Returns 0. (No load-factor guard: cap is fixed; keep the
// map under-full at creation -- a probe over a full map would loop, which
// for the gated corpus never happens.)
fn hm_insert(m: i32, k: i32, v: i32) -> i32 {
    let cap = __arena_get(m + 1);
    let slots = __arena_get(m + 2);
    let mask = cap - 1;
    let mut b = k & mask;
    let mut done = 0;
    while done == 0 {
        let off = slots + b * 3;
        let state = __arena_get(off);
        if state == 0 {
            __arena_set(off, 1);
            __arena_set(off + 1, k);
            __arena_set(off + 2, v);
            __arena_set(m, __arena_get(m) + 1);
            done = 1;
        } else {
            if __arena_get(off + 1) == k {
                __arena_set(off + 2, v);   // overwrite existing key
                done = 1;
            } else {
                b = (b + 1) & mask;        // linear probe to the next bucket
            }
        }
    };
    0
}

// Look up k; returns its value, or hm_missing() if k is absent. Probes from
// the home bucket; an EMPTY bucket means "not present" (open-addressing
// invariant: a key's probe chain is unbroken up to the first empty slot).
fn hm_get(m: i32, k: i32) -> i32 {
    let cap = __arena_get(m + 1);
    let slots = __arena_get(m + 2);
    let mask = cap - 1;
    let mut b = k & mask;
    let mut res = hm_missing();
    let mut done = 0;
    while done == 0 {
        let off = slots + b * 3;
        let state = __arena_get(off);
        if state == 0 {
            done = 1;                      // hit an empty slot -> not present
        } else {
            if __arena_get(off + 1) == k {
                res = __arena_get(off + 2);
                done = 1;
            } else {
                b = (b + 1) & mask;
            }
        }
    };
    res
}

// 1 if k is present, else 0.
fn hm_contains(m: i32, k: i32) -> i32 {
    let cap = __arena_get(m + 1);
    let slots = __arena_get(m + 2);
    let mask = cap - 1;
    let mut b = k & mask;
    let mut res = 0;
    let mut done = 0;
    while done == 0 {
        let off = slots + b * 3;
        let state = __arena_get(off);
        if state == 0 {
            done = 1;
        } else {
            if __arena_get(off + 1) == k {
                res = 1;
                done = 1;
            } else {
                b = (b + 1) & mask;
            }
        }
    };
    res
}

fn hm_len(m: i32) -> i32 { __arena_get(m) }
