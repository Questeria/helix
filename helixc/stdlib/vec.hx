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
//   vec_index_of(start, count, target) -> i32   first matching index, -1 if none
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

@pure
fn vec_sum(start: i32, count: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < count {
        total = total + __arena_get(start + i);
        i = i + 1;
    }
    total
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
