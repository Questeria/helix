// helixc/stdlib/iterators.hx — iterator-style operations over arena Vec<i32>.
//
// Phase 1 stdlib closer: matches the carry-pair convention from vec.hx.
// No closures — higher-order ops are specialised by op-tag or scalar arg.
// Output ops (range/map/zip/filter) append to the global arena and return
// the new (start) index. Caller computes count from the input or the
// returned value.
//
// API:
//   range_to_vec(lo, hi)                 -> i32   appends lo..hi (exclusive); returns start
//   vec_min(start, count)                -> i32   smallest element (0 if empty)
//   vec_count_eq(start, count, target)   -> i32   how many elements equal target
//   vec_count_lt(start, count, t)        -> i32   how many elements are < t
//   vec_count_gt(start, count, t)        -> i32   how many elements are > t
//   vec_count_le(start, count, t)        -> i32   how many elements are <= t
//   vec_count_ge(start, count, t)        -> i32   how many elements are >= t
//   vec_count_ne(start, count, t)        -> i32   how many elements != t
//   vec_fold_op(start, count, init, op)  -> i32   reduce with op 0:add 1:mul 2:max 3:min
//   vec_map_add_scalar(start, count, k)  -> i32   appends [x+k for x in v]; returns new start
//   vec_map_mul_scalar(start, count, k)  -> i32   appends [x*k for x in v]; returns new start
//   vec_zip_add(a, b, count)             -> i32   appends [a[i]+b[i]]; returns new start
//   vec_zip_mul(a, b, count)             -> i32   appends [a[i]*b[i]]; returns new start
//   vec_filter_lt(start, count, t)       -> i32   appends elems < t; returns kept count.
//                                                 Caller saves __arena_len() BEFORE calling
//                                                 to recover the new start index.
//   vec_filter_gt(start, count, t)       -> i32   appends elems > t; returns kept count.
//   vec_filter_eq(start, count, t)       -> i32   appends elems == t; returns kept count.
//   vec_zip_sub(a, b, count)             -> i32   appends [a[i]-b[i]]; returns new start.
//   vec_argmin(start, count)             -> i32   index of smallest element (-1 if empty).
//   vec_argmax(start, count)             -> i32   index of largest element (-1 if empty).
//   vec_dot(a, b, count)                 -> i32   dot product sum(a[i]*b[i]).
//   vec_zip_min(a, b, count)             -> i32   appends [min(a[i], b[i])]; returns new start.
//   vec_zip_max(a, b, count)             -> i32   appends [max(a[i], b[i])]; returns new start.
//
// License: Apache 2.0

@pure
fn range_to_vec(lo: i32, hi: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = lo;
    while i < hi {
        __arena_push(i);
        i = i + 1;
    }
    s
}

@pure
fn vec_min(start: i32, count: i32) -> i32 {
    if count == 0 { 0 }
    else {
        let mut i: i32 = 1;
        let mut best: i32 = __arena_get(start);
        while i < count {
            let v = __arena_get(start + i);
            if v < best { best = v; }
            i = i + 1;
        }
        best
    }
}

@pure
fn vec_count_eq(start: i32, count: i32, target: i32) -> i32 {
    let mut i: i32 = 0;
    let mut n: i32 = 0;
    while i < count {
        if __arena_get(start + i) == target { n = n + 1; }
        i = i + 1;
    }
    n
}

@pure
fn vec_count_lt(start: i32, count: i32, t: i32) -> i32 {
    let mut i: i32 = 0;
    let mut n: i32 = 0;
    while i < count {
        if __arena_get(start + i) < t { n = n + 1; }
        i = i + 1;
    }
    n
}

@pure
fn vec_count_gt(start: i32, count: i32, t: i32) -> i32 {
    let mut i: i32 = 0;
    let mut n: i32 = 0;
    while i < count {
        if __arena_get(start + i) > t { n = n + 1; }
        i = i + 1;
    }
    n
}

@pure
fn vec_count_le(start: i32, count: i32, t: i32) -> i32 {
    let mut i: i32 = 0;
    let mut n: i32 = 0;
    while i < count {
        if __arena_get(start + i) <= t { n = n + 1; }
        i = i + 1;
    }
    n
}

@pure
fn vec_count_ge(start: i32, count: i32, t: i32) -> i32 {
    let mut i: i32 = 0;
    let mut n: i32 = 0;
    while i < count {
        if __arena_get(start + i) >= t { n = n + 1; }
        i = i + 1;
    }
    n
}

@pure
fn vec_count_ne(start: i32, count: i32, t: i32) -> i32 {
    let mut i: i32 = 0;
    let mut n: i32 = 0;
    while i < count {
        if __arena_get(start + i) != t { n = n + 1; }
        i = i + 1;
    }
    n
}

@pure
fn vec_fold_op(start: i32, count: i32, init: i32, op: i32) -> i32 {
    let mut i: i32 = 0;
    let mut acc: i32 = init;
    while i < count {
        let v = __arena_get(start + i);
        if op == 0 { acc = acc + v; }
        if op == 1 { acc = acc * v; }
        if op == 2 { if v > acc { acc = v; } }
        if op == 3 { if v < acc { acc = v; } }
        i = i + 1;
    }
    acc
}

fn vec_map_add_scalar(start: i32, count: i32, k: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        __arena_push(__arena_get(start + i) + k);
        i = i + 1;
    }
    s
}

fn vec_map_mul_scalar(start: i32, count: i32, k: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        __arena_push(__arena_get(start + i) * k);
        i = i + 1;
    }
    s
}

fn vec_zip_add(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        __arena_push(__arena_get(a + i) + __arena_get(b + i));
        i = i + 1;
    }
    s
}

fn vec_zip_mul(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        __arena_push(__arena_get(a + i) * __arena_get(b + i));
        i = i + 1;
    }
    s
}

fn vec_filter_lt(start: i32, count: i32, t: i32) -> i32 {
    let mut i: i32 = 0;
    let mut kept: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        if v < t {
            __arena_push(v);
            kept = kept + 1;
        }
        i = i + 1;
    }
    kept
}

fn vec_filter_gt(start: i32, count: i32, t: i32) -> i32 {
    let mut i: i32 = 0;
    let mut kept: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        if v > t {
            __arena_push(v);
            kept = kept + 1;
        }
        i = i + 1;
    }
    kept
}

fn vec_filter_eq(start: i32, count: i32, t: i32) -> i32 {
    let mut i: i32 = 0;
    let mut kept: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        if v == t {
            __arena_push(v);
            kept = kept + 1;
        }
        i = i + 1;
    }
    kept
}

fn vec_zip_sub(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        __arena_push(__arena_get(a + i) - __arena_get(b + i));
        i = i + 1;
    }
    s
}

@pure
fn vec_argmin(start: i32, count: i32) -> i32 {
    if count == 0 { 0 - 1 }
    else {
        let mut i: i32 = 1;
        let mut best_i: i32 = 0;
        let mut best: i32 = __arena_get(start);
        while i < count {
            let v = __arena_get(start + i);
            if v < best { best = v; best_i = i; }
            i = i + 1;
        }
        best_i
    }
}

@pure
fn vec_argmax(start: i32, count: i32) -> i32 {
    if count == 0 { 0 - 1 }
    else {
        let mut i: i32 = 1;
        let mut best_i: i32 = 0;
        let mut best: i32 = __arena_get(start);
        while i < count {
            let v = __arena_get(start + i);
            if v > best { best = v; best_i = i; }
            i = i + 1;
        }
        best_i
    }
}

@pure
fn vec_dot(a: i32, b: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let mut acc: i32 = 0;
    while i < count {
        acc = acc + __arena_get(a + i) * __arena_get(b + i);
        i = i + 1;
    }
    acc
}

fn vec_zip_min(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        let x = __arena_get(a + i);
        let y = __arena_get(b + i);
        if x < y { __arena_push(x); } else { __arena_push(y); }
        i = i + 1;
    }
    s
}

fn vec_zip_max(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        let x = __arena_get(a + i);
        let y = __arena_get(b + i);
        if x > y { __arena_push(x); } else { __arena_push(y); }
        i = i + 1;
    }
    s
}
