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

// vec_reverse_alloc(start, count): allocating mirror of
// vec_reverse_inplace. Returns a new slice containing the input
// elements in reverse order. Original input untouched.
fn vec_reverse_alloc(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        __arena_push(__arena_get(start + count - 1 - i));
        i = i + 1;
    }
    s
}

// vec_repeat(value, count): allocate a new slice of length `count`
// filled with `value`. Allocating mirror of vec_fill_inplace; useful
// for initializing accumulators or padding arrays. count <= 0
// returns an empty slice.
fn vec_repeat(value: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    if count > 0 {
        let mut i: i32 = 0;
        while i < count {
            __arena_push(value);
            i = i + 1;
        }
    };
    s
}

// vec_zip_mod(a, b, count): element-wise modulo a[i] % b[i].
// Returns a new slice. b[i] == 0 emits ud2 (handled by Helix's
// integer-mod trap on division by zero — std behavior, not stdlib's
// concern). Useful for hashing into buckets and modular arithmetic.
fn vec_zip_mod(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        __arena_push(__arena_get(a + i) % __arena_get(b + i));
        i = i + 1;
    }
    s
}

// vec_take(start, count, n): return a new slice with the first `n`
// elements of the input. If n >= count, copies all elements; if n
// <= 0, returns empty. Useful for bounded prefix reads, top-N
// inspection. Allocating; original input untouched.
fn vec_take(start: i32, count: i32, n: i32) -> i32 {
    let s: i32 = __arena_len();
    let take = if n < 0 { 0 } else { if n > count { count } else { n } };
    let mut i: i32 = 0;
    while i < take {
        __arena_push(__arena_get(start + i));
        i = i + 1;
    }
    s
}

// vec_drop(start, count, n): return a new slice with the first `n`
// elements DROPPED. Saturates: n >= count returns empty, n <= 0
// returns full copy. Companion to vec_take.
fn vec_drop(start: i32, count: i32, n: i32) -> i32 {
    let s: i32 = __arena_len();
    let drop = if n < 0 { 0 } else { if n > count { count } else { n } };
    let remaining = count - drop;
    let mut i: i32 = 0;
    while i < remaining {
        __arena_push(__arena_get(start + drop + i));
        i = i + 1;
    }
    s
}

// vec_concat(a, na, b, nb): allocate a new slice that's `a[0..na]`
// followed by `b[0..nb]`. Useful for stitching subresults; avoids
// the temptation to mutate either input.
fn vec_concat(a: i32, na: i32, b: i32, nb: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < na {
        __arena_push(__arena_get(a + i));
        i = i + 1;
    }
    let mut j: i32 = 0;
    while j < nb {
        __arena_push(__arena_get(b + j));
        j = j + 1;
    }
    s
}

// vec_zip_div(a, b, count): element-wise division a[i] / b[i].
// Returns a new slice. b[i] == 0 emits the standard integer-div
// division-by-zero trap. Companion to vec_zip_mul / vec_zip_mod.
fn vec_zip_div(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        __arena_push(__arena_get(a + i) / __arena_get(b + i));
        i = i + 1;
    }
    s
}

// vec_zip_eq(a, b, count): element-wise equality returning 0/1 bools.
// Useful for masking and counting matches between two vecs of equal
// length. Sum of the result = number of equal elements.
fn vec_zip_eq(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        let av = __arena_get(a + i);
        let bv = __arena_get(b + i);
        if av == bv { __arena_push(1); } else { __arena_push(0); };
        i = i + 1;
    }
    s
}

// vec_mean(start, count): arithmetic mean via integer division
// (sum / count). Useful for ML running averages, stats, sanity
// checks. count <= 0 returns 0 (avoid div-by-zero trap).
@pure
fn vec_mean(start: i32, count: i32) -> i32 {
    if count <= 0 { 0 }
    else {
        let mut i: i32 = 0;
        let mut acc: i32 = 0;
        while i < count {
            acc = acc + __arena_get(start + i);
            i = i + 1;
        }
        acc / count
    }
}

// vec_argsort(start, count): selection sort returning a NEW slice of
// indices i_0, i_1, ..., i_{count-1} such that
// input[i_0] <= input[i_1] <= ... <= input[i_{count-1}].
// Original input untouched. O(count^2) time — fine for small N
// (~1000); useful for top-K selection, ranking. Selection sort
// chosen for code simplicity over heap or merge sort.
fn vec_argsort(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    // Initialize indices [0, 1, 2, ..., count-1]
    let mut i: i32 = 0;
    while i < count {
        __arena_push(i);
        i = i + 1;
    }
    // Selection sort by input[indices[k]] ascending
    let mut k: i32 = 0;
    while k < count {
        let mut min_pos: i32 = k;
        let mut j: i32 = k + 1;
        while j < count {
            let idx_j = __arena_get(s + j);
            let idx_min = __arena_get(s + min_pos);
            if __arena_get(start + idx_j) < __arena_get(start + idx_min) {
                min_pos = j;
            };
            j = j + 1;
        }
        // Swap s[k] <-> s[min_pos] if needed
        if min_pos != k {
            let tmp = __arena_get(s + k);
            __arena_set(s + k, __arena_get(s + min_pos));
            __arena_set(s + min_pos, tmp);
        };
        k = k + 1;
    }
    s
}

// vec_zip_lt/gt/le/ge/ne(a, b, count): element-wise comparisons
// returning 0/1 bool vecs. Companion to vec_zip_eq. Sum of result
// = number of positions matching the predicate. Useful for masking,
// metric counting (e.g. how many elements of `actual` are below
// `target`), set-like operations.
fn vec_zip_lt(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        if __arena_get(a + i) < __arena_get(b + i) { __arena_push(1); }
        else { __arena_push(0); };
        i = i + 1;
    }
    s
}

fn vec_zip_gt(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        if __arena_get(a + i) > __arena_get(b + i) { __arena_push(1); }
        else { __arena_push(0); };
        i = i + 1;
    }
    s
}

fn vec_zip_le(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        if __arena_get(a + i) <= __arena_get(b + i) { __arena_push(1); }
        else { __arena_push(0); };
        i = i + 1;
    }
    s
}

fn vec_zip_ge(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        if __arena_get(a + i) >= __arena_get(b + i) { __arena_push(1); }
        else { __arena_push(0); };
        i = i + 1;
    }
    s
}

fn vec_zip_ne(a: i32, b: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < count {
        if __arena_get(a + i) != __arena_get(b + i) { __arena_push(1); }
        else { __arena_push(0); };
        i = i + 1;
    }
    s
}

// vec_take_while(start, count, pivot): take elements while x[i] < pivot.
// Stops at the FIRST element >= pivot. Returns the count of taken elements
// (0..count). The taken slice is exactly start[0..returned_count]; no
// new allocation. Common use: parse a sorted vec up to a threshold.
@pure
fn vec_take_while(start: i32, count: i32, pivot: i32) -> i32 {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut result: i32 = count;
    while i < count {
        if found == 0 {
            if __arena_get(start + i) >= pivot {
                result = i;
                found = 1;
            };
        };
        i = i + 1;
    }
    result
}

// vec_drop_while(start, count, pivot): symmetric to vec_take_while.
// Returns the index of the FIRST element >= pivot (the "drop count").
// Caller can pass start+returned_count + (count-returned_count) as the
// remaining slice. @pure.
@pure
fn vec_drop_while(start: i32, count: i32, pivot: i32) -> i32 {
    vec_take_while(start, count, pivot)
}

// vec_dedup_consecutive(start, count): allocate a new vec containing
// `start[0..count]` with adjacent duplicates collapsed. e.g.
// [1,1,2,3,3,3,4] → [1,2,3,4]. Returns the new vec's start index;
// caller can compute new length as the difference between the next
// __arena_len() and the returned start, or use a separate fn.
fn vec_dedup_consecutive(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    if count == 0 { s }
    else {
        __arena_push(__arena_get(start));
        let mut i: i32 = 1;
        while i < count {
            let prev = __arena_get(start + i - 1);
            let cur = __arena_get(start + i);
            if cur != prev { __arena_push(cur); };
            i = i + 1;
        }
        s
    }
}

// vec_count_distinct_consecutive(start, count): count of elements in
// the deduped form of vec_dedup_consecutive — useful when caller wants
// the size BEFORE calling dedup. Walks the input and counts boundaries.
@pure
fn vec_count_distinct_consecutive(start: i32, count: i32) -> i32 {
    if count == 0 { 0 }
    else {
        let mut i: i32 = 1;
        let mut total: i32 = 1;
        while i < count {
            let prev = __arena_get(start + i - 1);
            let cur = __arena_get(start + i);
            if cur != prev { total = total + 1; };
            i = i + 1;
        }
        total
    }
}

// vec_running_max(start, count): allocate a new vec where r[i] = max
// over v[0..=i]. r[0] = v[0]; r[i] = max(r[i-1], v[i]). Useful for
// online max queries.
fn vec_running_max(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    if count == 0 { s }
    else {
        let mut best: i32 = __arena_get(start);
        __arena_push(best);
        let mut i: i32 = 1;
        while i < count {
            let v = __arena_get(start + i);
            if v > best { best = v; };
            __arena_push(best);
            i = i + 1;
        }
        s
    }
}

// vec_running_min(start, count): companion to vec_running_max.
fn vec_running_min(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    if count == 0 { s }
    else {
        let mut best: i32 = __arena_get(start);
        __arena_push(best);
        let mut i: i32 = 1;
        while i < count {
            let v = __arena_get(start + i);
            if v < best { best = v; };
            __arena_push(best);
            i = i + 1;
        }
        s
    }
}

// vec_rotate_left_alloc(start, count, k): allocate a new vec with
// elements rotated left by k positions. r[i] = v[(i + k) % count].
// k can be larger than count or negative; we normalize to [0, count).
fn vec_rotate_left_alloc(start: i32, count: i32, k: i32) -> i32 {
    let s: i32 = __arena_len();
    if count == 0 { s }
    else {
        // Normalize k to [0, count).
        let mut kn: i32 = k % count;
        if kn < 0 { kn = kn + count; };
        let mut i: i32 = 0;
        while i < count {
            let src_idx = (i + kn) % count;
            __arena_push(__arena_get(start + src_idx));
            i = i + 1;
        }
        s
    }
}

// vec_window_sum(start, count, win): allocate a new vec of length
// (count - win + 1) where r[i] = sum(v[i..i+win]). Uses rolling sum
// for O(count) total. If win > count, returns empty.
fn vec_window_sum(start: i32, count: i32, win: i32) -> i32 {
    let s: i32 = __arena_len();
    if win > count { s }
    else { if win <= 0 { s }
    else {
        // Initial window sum.
        let mut acc: i32 = 0;
        let mut i: i32 = 0;
        while i < win {
            acc = acc + __arena_get(start + i);
            i = i + 1;
        }
        __arena_push(acc);
        // Slide: subtract leaving element, add entering.
        let mut j: i32 = win;
        while j < count {
            acc = acc - __arena_get(start + j - win);
            acc = acc + __arena_get(start + j);
            __arena_push(acc);
            j = j + 1;
        }
        s
    }}
}

// vec_all_eq(start, count, target): @pure. Returns 1 if every element
// equals target, 0 otherwise. Empty vec returns 1 (vacuously true).
@pure
fn vec_all_eq(start: i32, count: i32, target: i32) -> i32 {
    let mut i: i32 = 0;
    let mut ok: i32 = 1;
    while i < count {
        if __arena_get(start + i) != target { ok = 0; }
        i = i + 1;
    }
    ok
}

// vec_any_eq(start, count, target): @pure. Returns 1 if any element
// equals target, 0 otherwise. Empty vec returns 0.
@pure
fn vec_any_eq(start: i32, count: i32, target: i32) -> i32 {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    while i < count {
        if __arena_get(start + i) == target { found = 1; }
        i = i + 1;
    }
    found
}

// vec_is_sorted_asc(start, count): @pure. Returns 1 if elements are
// in non-decreasing order, 0 otherwise. count <= 1 returns 1.
@pure
fn vec_is_sorted_asc(start: i32, count: i32) -> i32 {
    let mut i: i32 = 1;
    let mut ok: i32 = 1;
    while i < count {
        if __arena_get(start + i) < __arena_get(start + i - 1) { ok = 0; }
        i = i + 1;
    }
    ok
}

// vec_is_sorted_desc(start, count): @pure. Returns 1 if elements are
// in non-increasing order, 0 otherwise. count <= 1 returns 1.
@pure
fn vec_is_sorted_desc(start: i32, count: i32) -> i32 {
    let mut i: i32 = 1;
    let mut ok: i32 = 1;
    while i < count {
        if __arena_get(start + i) > __arena_get(start + i - 1) { ok = 0; }
        i = i + 1;
    }
    ok
}

// vec_index_of_pure(start, count, target): @pure. Returns the first
// index where v[i] == target, or -1 if not present. Renamed from
// "find_eq" for the iterator-pattern naming convention.
@pure
fn vec_index_of_pure(start: i32, count: i32, target: i32) -> i32 {
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < count {
        if found < 0 {
            if __arena_get(start + i) == target { found = i; };
        };
        i = i + 1;
    }
    found
}

// vec_window_max(start, count, win): allocate a new vec of length
// (count - win + 1) where r[i] = max(v[i..i+win]). O(count*win) naive
// but fine for small win. Returns empty if win > count or win <= 0.
fn vec_window_max(start: i32, count: i32, win: i32) -> i32 {
    let s: i32 = __arena_len();
    if win > count { s }
    else { if win <= 0 { s }
    else {
        let last_start = count - win;
        let mut i: i32 = 0;
        while i <= last_start {
            let mut best = __arena_get(start + i);
            let mut k: i32 = 1;
            while k < win {
                let v = __arena_get(start + i + k);
                if v > best { best = v; };
                k = k + 1;
            }
            __arena_push(best);
            i = i + 1;
        }
        s
    }}
}

// vec_window_min(start, count, win): companion to vec_window_max.
fn vec_window_min(start: i32, count: i32, win: i32) -> i32 {
    let s: i32 = __arena_len();
    if win > count { s }
    else { if win <= 0 { s }
    else {
        let last_start = count - win;
        let mut i: i32 = 0;
        while i <= last_start {
            let mut best = __arena_get(start + i);
            let mut k: i32 = 1;
            while k < win {
                let v = __arena_get(start + i + k);
                if v < best { best = v; };
                k = k + 1;
            }
            __arena_push(best);
            i = i + 1;
        }
        s
    }}
}

// vec_count_in_range(start, count, lo, hi): @pure. Count of indices
// where lo <= v[i] <= hi (inclusive). Empty input returns 0.
@pure
fn vec_count_in_range(start: i32, count: i32, lo: i32, hi: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        if v >= lo {
            if v <= hi { total = total + 1; };
        };
        i = i + 1;
    }
    total
}

// vec_pairwise_diff(start, count): allocate a new vec of length
// (count - 1) where r[i] = v[i+1] - v[i]. Discrete derivative.
// Empty if count <= 1.
fn vec_pairwise_diff(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    if count <= 1 { s }
    else {
        let mut i: i32 = 1;
        while i < count {
            __arena_push(__arena_get(start + i) - __arena_get(start + i - 1));
            i = i + 1;
        }
        s
    }
}

// vec_argmax_in_range(start, lo, hi): @pure. Index of largest element
// in v[lo..hi). Returns -1 if hi <= lo. Symmetric to tf1d_argmax_in_range
// but for integer vecs.
@pure
fn vec_argmax_in_range(start: i32, lo: i32, hi: i32) -> i32 {
    if hi <= lo { 0 - 1 }
    else {
        let mut i: i32 = lo + 1;
        let mut best_idx: i32 = lo;
        let mut best: i32 = __arena_get(start + lo);
        while i < hi {
            let v = __arena_get(start + i);
            if v > best { best = v; best_idx = i; }
            i = i + 1;
        }
        best_idx
    }
}

// vec_argmin_in_range(start, lo, hi): @pure. Companion. -1 if empty.
@pure
fn vec_argmin_in_range(start: i32, lo: i32, hi: i32) -> i32 {
    if hi <= lo { 0 - 1 }
    else {
        let mut i: i32 = lo + 1;
        let mut best_idx: i32 = lo;
        let mut best: i32 = __arena_get(start + lo);
        while i < hi {
            let v = __arena_get(start + i);
            if v < best { best = v; best_idx = i; }
            i = i + 1;
        }
        best_idx
    }
}

// vec_sum_in_range(start, lo, hi): @pure. Partial sum over v[lo..hi).
@pure
fn vec_sum_in_range(start: i32, lo: i32, hi: i32) -> i32 {
    let mut i: i32 = lo;
    let mut total: i32 = 0;
    while i < hi {
        total = total + __arena_get(start + i);
        i = i + 1;
    }
    total
}

// vec_reverse_inplace(start, count): swap pairs (i, count-1-i) in place.
// Returns start. Useful before sort-friendly operations or for visual
// reversal.
fn vec_reverse_inplace(start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let half = count / 2;
    while i < half {
        let j = count - 1 - i;
        let a = __arena_get(start + i);
        let b = __arena_get(start + j);
        __arena_set(start + i, b);
        __arena_set(start + j, a);
        i = i + 1;
    }
    start
}

// vec_unique_alloc(start, count): allocate a new vec containing each
// distinct value from input. Order of first occurrence is preserved.
// O(count^2) — fine for small inputs; use a hashmap for big sets.
fn vec_unique_alloc(start: i32, count: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut written: i32 = 0;
    let mut i: i32 = 0;
    while i < count {
        let v = __arena_get(start + i);
        // Linear scan over already-written portion.
        let mut j: i32 = 0;
        let mut seen: i32 = 0;
        while j < written {
            if __arena_get(s + j) == v { seen = 1; };
            j = j + 1;
        }
        if seen == 0 {
            __arena_push(v);
            written = written + 1;
        };
        i = i + 1;
    }
    s
}

// vec_intersect(a, an, b, bn): allocate a new vec containing elements
// that appear in BOTH a and b. Preserves a's order, dedupes implicitly
// (won't add a duplicate if it's already been added). O(an * bn).
fn vec_intersect(a: i32, an: i32, b: i32, bn: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut written: i32 = 0;
    let mut i: i32 = 0;
    while i < an {
        let v = __arena_get(a + i);
        // In b?
        let mut j: i32 = 0;
        let mut in_b: i32 = 0;
        while j < bn {
            if __arena_get(b + j) == v { in_b = 1; };
            j = j + 1;
        }
        if in_b == 1 {
            // Already written?
            let mut k: i32 = 0;
            let mut seen: i32 = 0;
            while k < written {
                if __arena_get(s + k) == v { seen = 1; };
                k = k + 1;
            }
            if seen == 0 {
                __arena_push(v);
                written = written + 1;
            };
        };
        i = i + 1;
    }
    s
}

// vec_difference(a, an, b, bn): allocate a new vec with elements in a
// but NOT in b. Preserves a's order. Dedupes. O(an * bn).
fn vec_difference(a: i32, an: i32, b: i32, bn: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut written: i32 = 0;
    let mut i: i32 = 0;
    while i < an {
        let v = __arena_get(a + i);
        let mut j: i32 = 0;
        let mut in_b: i32 = 0;
        while j < bn {
            if __arena_get(b + j) == v { in_b = 1; };
            j = j + 1;
        }
        if in_b == 0 {
            let mut k: i32 = 0;
            let mut seen: i32 = 0;
            while k < written {
                if __arena_get(s + k) == v { seen = 1; };
                k = k + 1;
            }
            if seen == 0 {
                __arena_push(v);
                written = written + 1;
            };
        };
        i = i + 1;
    }
    s
}

// vec_concat3(a, an, b, bn, c, cn): concat 3 vecs into one new alloc.
// Common pattern when building messages or splice-then-reassemble.
fn vec_concat3(a: i32, an: i32, b: i32, bn: i32, c: i32, cn: i32) -> i32 {
    let s: i32 = __arena_len();
    let mut i: i32 = 0;
    while i < an { __arena_push(__arena_get(a + i)); i = i + 1; }
    let mut j: i32 = 0;
    while j < bn { __arena_push(__arena_get(b + j)); j = j + 1; }
    let mut k: i32 = 0;
    while k < cn { __arena_push(__arena_get(c + k)); k = k + 1; }
    s
}
