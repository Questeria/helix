// helixc/stdlib/vec.hx — arena-backed Vec<i32>.
//
// Phase 1.9: a "carry-pair" Vec<i32>. The vec is represented by two
// integers — start (arena index of slot 0) and count (current length).
// The caller threads start+count through pushes; the values are stored
// in the global arena. Inspired by hbs_lib_vec.hx but cleaned up + given
// a stdlib home.
//
// Convention: build one Vec at a time; interleaved pushes mix slots
// because the arena is global. For AGI work this is fine since most
// list-building is sequential.
//
// API:
//   vec_new()                  -> i32           start = current arena length
//   vec_push(start, count, x)  -> i32           returns new count
//   vec_get(start, i)          -> i32           read element at index i
//   vec_set(start, i, x)       -> i32           write; returns x
//   vec_sum(start, count)      -> i32           sum all elements
//   vec_max(start, count)      -> i32           largest element (0 if empty)
//   vec_product(start, count)  -> i32           product of all elements (1 if empty — multiplicative identity)
//   vec_first(start, count)    -> i32           v[0] (0 if empty)
//   vec_last(start, count)     -> i32           v[count-1] (0 if empty)
//   vec_index_of(start, count, target) -> i32   first matching index, -1 if none
//   vec_contains(start, count, target) -> i32   1 if target present, else 0
//   vec_eq(a_start, b_start, count) -> i32      1 if all elements equal, else 0
//   vec_reverse_inplace(start, count) -> i32    reverses in place; returns start
//
// License: Apache 2.0

@pure
fn vec_new() -> i32 {
    __arena_len()
}

fn vec_push(start: i32, count: i32, x: i32) -> i32 {
    __arena_push(x);
    count + 1
}

@pure
fn vec_get(start: i32, i: i32) -> i32 {
    __arena_get(start + i)
}

fn vec_set(start: i32, i: i32, x: i32) -> i32 {
    __arena_set(start + i, x);
    x
}

// Restart 53 A3: i64 accumulator + INT32 saturation. Sibling of
// ti1d_sum (restart 51 A2) and iterators.hx vec_sum_pure (restart 53).
// The vec.hx companion was missed in the original sweep.
@pure
fn vec_sum(start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i64 = 0_i64;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    while i < count {
        total = total + (__arena_get(start + i) as i64);
        if total > hi { total = hi; }
        else { if total < lo { total = lo; } };
        i = i + 1;
    }
    total as i32
}

@pure
fn vec_max(start: i32, count: i32) -> i32 {
    if count == 0 { 0 }
    else {
        let mut i: i32 = 1;
        let mut best: i32 = __arena_get(start);
        while i < count {
            let v = __arena_get(start + i);
            if v > best { best = v; }
            i = i + 1;
        }
        best
    }
}

// Restart 53 A3: i64 accumulator + INT32 saturation. Sibling of
// ti1d_prod (restart 50 A3). A single multiplicand of even modest
// magnitude (~46341) quickly overflows the i32 product.
@pure
fn vec_product(start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let mut p: i64 = 1_i64;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    while i < count {
        p = p * (__arena_get(start + i) as i64);
        if p > hi { p = hi; }
        else { if p < lo { p = lo; } };
        i = i + 1;
    }
    p as i32
}

@pure
fn vec_first(start: i32, count: i32) -> i32 {
    if count == 0 { 0 }
    else { __arena_get(start) }
}

@pure
fn vec_last(start: i32, count: i32) -> i32 {
    if count == 0 { 0 }
    else { __arena_get(start + count - 1) }
}

@pure
fn vec_index_of(start: i32, count: i32, target: i32) -> i32 {
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < count {
        if __arena_get(start + i) == target {
            if found < 0 { found = i; }
        }
        i = i + 1;
    }
    found
}

@pure
fn vec_contains(start: i32, count: i32, target: i32) -> i32 {
    let mut i: i32 = 0;
    let mut hit: i32 = 0;
    while i < count {
        if __arena_get(start + i) == target { hit = 1; }
        i = i + 1;
    }
    hit
}

@pure
fn vec_eq(a_start: i32, b_start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let mut equal: i32 = 1;
    while i < count {
        if __arena_get(a_start + i) != __arena_get(b_start + i) { equal = 0; }
        i = i + 1;
    }
    equal
}

fn vec_reverse_inplace(start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let half: i32 = count / 2;
    while i < half {
        let lo: i32 = __arena_get(start + i);
        let hi: i32 = __arena_get(start + count - 1 - i);
        __arena_set(start + i, hi);
        __arena_set(start + count - 1 - i, lo);
        i = i + 1;
    }
    start
}
