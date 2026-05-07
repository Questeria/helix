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
//   vec_abs_sum(start, count)            -> i32   sum of |v[i]| (L1 norm).
//   vec_sum_squares(start, count)        -> i32   sum of v[i]*v[i] (squared L2 norm).
//   vec_clamp_inplace(s, c, lo, hi)      -> i32   clip elems to [lo, hi] in place; returns start.
//   vec_offset_inplace(start, count, k)  -> i32   adds k to each elem in place; returns start.
//                                                 In-place mirror of vec_map_add_scalar.
//   vec_fill_inplace(start, count, x)    -> i32   sets every elem to x in place; returns start.
//                                                 Useful for zero/constant init.
//   vec_swap_inplace(start, i, j)        -> i32   swap elems at indices i and j; returns start.
//                                                 Sort-step primitive.
//   vec_l1_distance(a, b, count)         -> i32   sum of |a[i] - b[i]| (L1 distance / Manhattan).
//   vec_l2_squared_distance(a, b, count) -> i32   sum of (a[i] - b[i])^2 (squared Euclidean).
//   vec_max_abs(start, count)            -> i32   max of |v[i]| (Linf norm proxy; 0 if empty).
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

fn vec_map_neg(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        __arena_push(0 - __arena_get(start + i));
        i = i + 1;
    }
    s
}

fn vec_map_abs(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        if v < 0 { __arena_push(0 - v); } else { __arena_push(v); }
        i = i + 1;
    }
    s
}

fn vec_map_relu(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        if v < 0 { __arena_push(0); } else { __arena_push(v); }
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

fn vec_filter_le(start: i32, count: i32, t: i32) -> i32 {
    let mut i: i32 = 0;
    let mut kept: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        if v <= t {
            __arena_push(v);
            kept = kept + 1;
        }
        i = i + 1;
    }
    kept
}

fn vec_filter_ge(start: i32, count: i32, t: i32) -> i32 {
    let mut i: i32 = 0;
    let mut kept: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        if v >= t {
            __arena_push(v);
            kept = kept + 1;
        }
        i = i + 1;
    }
    kept
}

fn vec_filter_ne(start: i32, count: i32, t: i32) -> i32 {
    let mut i: i32 = 0;
    let mut kept: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        if v != t {
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

fn vec_abs_sum(start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let mut acc: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        if v < 0 { acc = acc - v; } else { acc = acc + v; }
        i = i + 1;
    }
    acc
}

fn vec_sum_squares(start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let mut acc: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        acc = acc + v * v;
        i = i + 1;
    }
    acc
}

fn vec_clamp_inplace(start: i32, count: i32, lo: i32, hi: i32) -> i32 {
    let mut i: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        if v < lo { __arena_set(start + i, lo); }
        else { if v > hi { __arena_set(start + i, hi); } }
        i = i + 1;
    }
    start
}

fn vec_relu_inplace(start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        if v < 0 { __arena_set(start + i, 0); }
        i = i + 1;
    }
    start
}

fn vec_negate_inplace(start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        __arena_set(start + i, 0 - v);
        i = i + 1;
    }
    start
}

fn vec_scale_inplace(start: i32, count: i32, k: i32) -> i32 {
    let mut i: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        __arena_set(start + i, v * k);
        i = i + 1;
    }
    start
}

fn vec_offset_inplace(start: i32, count: i32, k: i32) -> i32 {
    let mut i: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        __arena_set(start + i, v + k);
        i = i + 1;
    }
    start
}

fn vec_fill_inplace(start: i32, count: i32, x: i32) -> i32 {
    let mut i: i32 = 0;
    while i < count {
        __arena_set(start + i, x);
        i = i + 1;
    }
    start
}

fn vec_swap_inplace(start: i32, i: i32, j: i32) -> i32 {
    let a = __arena_get(start + i);
    let b = __arena_get(start + j);
    __arena_set(start + i, b);
    __arena_set(start + j, a);
    start
}

@pure
fn vec_l1_distance(a: i32, b: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let mut acc: i32 = 0;
    while i < count {
        let d = __arena_get(a + i) - __arena_get(b + i);
        if d < 0 { acc = acc - d; } else { acc = acc + d; }
        i = i + 1;
    }
    acc
}

@pure
fn vec_l2_squared_distance(a: i32, b: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let mut acc: i32 = 0;
    while i < count {
        let d = __arena_get(a + i) - __arena_get(b + i);
        acc = acc + d * d;
        i = i + 1;
    }
    acc
}

@pure
fn vec_max_abs(start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let mut best: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        let av = if v < 0 { 0 - v } else { v };
        if av > best { best = av; }
        i = i + 1;
    }
    best
}

// vec_map_square(start, count): allocate a new arena slice with each
// element squared. Allocating mirror of the implicit pattern in
// vec_sum_squares; useful when the caller needs the squared values
// retained (e.g. variance computation, L2 element-wise norms).
fn vec_map_square(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        __arena_push(v * v);
        i = i + 1;
    }
    s
}

// vec_cumsum(start, count): allocate a new arena slice where
// out[i] = sum(in[0..=i]). Standard cumulative-sum / prefix-sum.
// out[0] = in[0], out[1] = in[0]+in[1], ..., out[count-1] = total.
fn vec_cumsum(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    let mut acc: i32 = 0;
    while i < count {
        acc = acc + __arena_get(start + i);
        __arena_push(acc);
        i = i + 1;
    }
    s
}

// vec_diff(start, count): allocate a new arena slice with the
// first-order differences out[i] = in[i+1] - in[i]. Output length is
// count - 1 (returns the start of the new slice). For count <= 1 the
// output is empty (length 0). Useful for discrete derivatives.
fn vec_diff(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    if count > 1 {
        let mut i: i32 = 0;
        let n_minus_1 = count - 1;
        while i < n_minus_1 {
            __arena_push(__arena_get(start + i + 1) - __arena_get(start + i));
            i = i + 1;
        }
    };
    s
}

// vec_map_clamp(start, count, lo, hi): allocating mirror of
// vec_clamp_inplace. Returns a new slice where each element is
// clamped to [lo, hi].
fn vec_map_clamp(start: i32, count: i32, lo: i32, hi: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        let clamped = if v < lo { lo } else { if v > hi { hi } else { v } };
        __arena_push(clamped);
        i = i + 1;
    }
    s
}
