// helixc/stdlib/hashmap.hx — arena-backed integer-keyed hash map.
//
// Phase 1.9 follow-on: a fixed-capacity, linear-probing hash map
// keyed by i32, valued by i32. Each bucket consumes 3 arena slots:
//   bucket+0  occupied flag (0 = empty, 1 = used)
//   bucket+1  key
//   bucket+2  value
//
// API uses the carry-pair convention from vec.hx — the caller threads
// (start, cap) through every call. Initial cap must be > 0; the map
// is full when cap inserts have happened (probing bails out at cap).
//
// API:
//   hashmap_new(cap)                      -> i32   start = arena index of bucket 0
//   hashmap_put(start, cap, k, v)         -> i32   1 if new, 0 if updated, -1 if full
//   hashmap_get(start, cap, k, default)   -> i32   value or default if missing
//   hashmap_has(start, cap, k)            -> i32   1 if present, 0 otherwise
//   hashmap_size(start, cap)              -> i32   count of occupied buckets
//
// License: Apache 2.0

fn hashmap_new(cap: i32) -> i32 {
    let start = __arena_len();
    let mut i: i32 = 0;
    while i < cap {
        __arena_push(0);
        __arena_push(0);
        __arena_push(0);
        i = i + 1;
    }
    start
}

@pure
fn hashmap_hash(k: i32, cap: i32) -> i32 {
    let mut h = k * 31 + 7;
    if h < 0 { h = 0 - h; }
    h % cap
}

fn hashmap_put(start: i32, cap: i32, k: i32, v: i32) -> i32 {
    let mut idx = hashmap_hash(k, cap);
    let mut probes: i32 = 0;
    let mut result: i32 = 0 - 1;
    while probes < cap {
        let base = start + idx * 3;
        let occ = __arena_get(base);
        if occ == 0 {
            __arena_set(base, 1);
            __arena_set(base + 1, k);
            __arena_set(base + 2, v);
            result = 1;
            probes = cap;
        } else {
            if __arena_get(base + 1) == k {
                __arena_set(base + 2, v);
                result = 0;
                probes = cap;
            } else {
                idx = idx + 1;
                if idx >= cap { idx = 0; }
                probes = probes + 1;
            }
        }
    }
    result
}

@pure
fn hashmap_get(start: i32, cap: i32, k: i32, default: i32) -> i32 {
    let mut idx = hashmap_hash(k, cap);
    let mut probes: i32 = 0;
    let mut result = default;
    while probes < cap {
        let base = start + idx * 3;
        let occ = __arena_get(base);
        if occ == 0 {
            probes = cap;
        } else {
            if __arena_get(base + 1) == k {
                result = __arena_get(base + 2);
                probes = cap;
            } else {
                idx = idx + 1;
                if idx >= cap { idx = 0; }
                probes = probes + 1;
            }
        }
    }
    result
}

@pure
fn hashmap_has(start: i32, cap: i32, k: i32) -> i32 {
    let mut idx = hashmap_hash(k, cap);
    let mut probes: i32 = 0;
    let mut result: i32 = 0;
    while probes < cap {
        let base = start + idx * 3;
        let occ = __arena_get(base);
        if occ == 0 {
            probes = cap;
        } else {
            if __arena_get(base + 1) == k {
                result = 1;
                probes = cap;
            } else {
                idx = idx + 1;
                if idx >= cap { idx = 0; }
                probes = probes + 1;
            }
        }
    }
    result
}

@pure
fn hashmap_size(start: i32, cap: i32) -> i32 {
    let mut i: i32 = 0;
    let mut count: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            count = count + 1;
        }
        i = i + 1;
    }
    count
}

// hashmap_clear(start, cap) -> i32
//   Empty every bucket in place by zeroing the occupancy flag. Key/value
//   slots are not cleared (they're dead until the bucket is reused). Returns
//   start so the caller can chain. NOT @pure (arena mutation).
fn hashmap_clear(start: i32, cap: i32) -> i32 {
    let mut i: i32 = 0;
    while i < cap {
        __arena_set(start + i * 3, 0);
        i = i + 1;
    }
    start
}

// hashmap_keys(start, cap) -> i32
//   Allocate a fresh arena slice and push the key of every occupied bucket
//   into it (in bucket order, NOT insertion order). Returns the start index
//   of the new slice; pair with hashmap_size(start, cap) for the count.
//   NOT @pure (arena push).
fn hashmap_keys(start: i32, cap: i32) -> i32 {
    let dst = __arena_len();
    let mut i: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            __arena_push(__arena_get(base + 1));
        }
        i = i + 1;
    }
    dst
}

// hashmap_values(start, cap) -> i32
//   Companion to hashmap_keys: pushes the value of every occupied bucket
//   into a fresh arena slice. Bucket order matches hashmap_keys exactly,
//   so the pair is index-aligned. NOT @pure (arena push).
fn hashmap_values(start: i32, cap: i32) -> i32 {
    let dst = __arena_len();
    let mut i: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            __arena_push(__arena_get(base + 2));
        }
        i = i + 1;
    }
    dst
}

// hashmap_increment(start, cap, k, delta): atomic get-then-put pattern
// for accumulator maps (e.g. word counts). If key present, adds delta
// to its value; if absent, inserts with value=delta. Returns the new
// value at the key, or -1 if the map is full and the key wasn't there.
fn hashmap_increment(start: i32, cap: i32, k: i32, delta: i32) -> i32 {
    let mut idx = hashmap_hash(k, cap);
    let mut probes: i32 = 0;
    let mut result: i32 = 0 - 1;
    while probes < cap {
        let base = start + idx * 3;
        let occ = __arena_get(base);
        if occ == 0 {
            __arena_set(base, 1);
            __arena_set(base + 1, k);
            __arena_set(base + 2, delta);
            result = delta;
            probes = cap;
        } else {
            if __arena_get(base + 1) == k {
                let new_val = __arena_get(base + 2) + delta;
                __arena_set(base + 2, new_val);
                result = new_val;
                probes = cap;
            } else {
                idx = idx + 1;
                if idx >= cap { idx = 0; };
                probes = probes + 1;
            };
        };
    }
    result
}

// hashmap_swap(start, cap, k, new_v): set value to new_v, return old
// value. If key was absent, inserts and returns -1. If map full and
// key not present, returns -1 (caller can't distinguish from a true
// -1 old value — known limitation).
fn hashmap_swap(start: i32, cap: i32, k: i32, new_v: i32) -> i32 {
    let mut idx = hashmap_hash(k, cap);
    let mut probes: i32 = 0;
    let mut result: i32 = 0 - 1;
    while probes < cap {
        let base = start + idx * 3;
        let occ = __arena_get(base);
        if occ == 0 {
            __arena_set(base, 1);
            __arena_set(base + 1, k);
            __arena_set(base + 2, new_v);
            probes = cap;
        } else {
            if __arena_get(base + 1) == k {
                result = __arena_get(base + 2);
                __arena_set(base + 2, new_v);
                probes = cap;
            } else {
                idx = idx + 1;
                if idx >= cap { idx = 0; };
                probes = probes + 1;
            };
        };
    }
    result
}

// hashmap_get_or(start, cap, k, default): @pure. Cleaner-name alias
// for hashmap_get's existing default-fallback API. Convenience for
// call sites where "or" reads more naturally.
@pure
fn hashmap_get_or(start: i32, cap: i32, k: i32, default: i32) -> i32 {
    hashmap_get(start, cap, k, default)
}

// hashmap_max_value(start, cap): @pure. Walks all occupied buckets,
// returns the maximum value. Returns 0 for an empty map (no occupied
// buckets); caller should check hashmap_size beforehand if 0 is a
// valid value to disambiguate.
@pure
fn hashmap_max_value(start: i32, cap: i32) -> i32 {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut best: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            let v = __arena_get(base + 2);
            if found == 0 { best = v; found = 1; }
            else { if v > best { best = v; }; };
        };
        i = i + 1;
    }
    best
}
