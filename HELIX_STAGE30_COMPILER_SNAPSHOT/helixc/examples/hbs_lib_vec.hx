// hbs_lib_vec.hx
//
// HBS library: Vec<i32> built on the arena allocator. Implements the
// (start, count) carry-pair convention recommended by the research
// agent — each Vec is a pair of integers carried by the caller, NOT
// an in-arena structure with mutable internal pointers (Helix has no
// mutable struct fields yet).
//
// API:
//   vec_new() -> i32                 — returns the start slot index;
//                                      a fresh arena push reservation
//   vec_push(start, count, x) -> i32 — push x into the next slot;
//                                      caller must pass count+1 next time
//   vec_get(start, i) -> i32         — read slot i
//   vec_set(start, i, x) -> i32      — write slot i
//   vec_len_from_count(c) -> i32     — passthrough (the "count" is the length)
//
// LIMITATION: this is a SHARED arena, so vec_new() doesn't actually
// reserve a contiguous region. If two vecs interleave their pushes,
// they'll mix. For HBS bootstrap the convention is "construct one
// vec at a time, fully, before starting another" — which works fine.
// Concurrent vecs would need a separate arena per vec, which would
// require multiple arenas (a future tick).

@total
fn vec_new() -> i32 {
    // The "start" of this vec is the current arena length.
    __arena_len()
}

@total
fn vec_push(start: i32, count: i32, x: i32) -> i32 {
    // Push x into the arena. Returns the new count.
    __arena_push(x);
    count + 1
}

@total
fn vec_get(start: i32, i: i32) -> i32 {
    __arena_get(start + i)
}

@total
fn vec_set(start: i32, i: i32, x: i32) -> i32 {
    __arena_set(start + i, x);
    x
}

// Sum the elements of a vec.
@partial
fn vec_sum(start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < count {
        total = total + vec_get(start, i);
        i = i + 1;
    }
    total
}

// Find the maximum.
@partial
fn vec_max(start: i32, count: i32) -> i32 {
    if count <= 0 {
        0
    } else {
        let mut i: i32 = 1;
        let mut m: i32 = vec_get(start, 0);
        while i < count {
            let v = vec_get(start, i);
            if v > m { m = v; }
            i = i + 1;
        }
        m
    }
}

// Linear search; returns index of first match or -1.
@partial
fn vec_find(start: i32, count: i32, target: i32) -> i32 {
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < count {
        if vec_get(start, i) == target {
            found = i;
            i = count;        // exit the loop
        } else {
            i = i + 1;
        }
    }
    found
}

fn main() -> i32 {
    // Build a vec of [10, 20, 7, 5, 30].
    let s = vec_new();
    let c0 = 0;
    let c1 = vec_push(s, c0, 10);
    let c2 = vec_push(s, c1, 20);
    let c3 = vec_push(s, c2, 7);
    let c4 = vec_push(s, c3, 5);
    let c5 = vec_push(s, c4, 30);

    let total = vec_sum(s, c5);    // 72
    let m = vec_max(s, c5);         // 30
    let idx_7 = vec_find(s, c5, 7); // 2

    // Verify: 72 - 30 + 0 = 42  if find_idx_7 == 2 (the 3rd elem)
    if idx_7 == 2 {
        total - m
    } else {
        0
    }
}
