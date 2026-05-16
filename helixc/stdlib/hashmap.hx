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

@pure fn hashmap_magic() -> i32 { 7007001 }

@pure fn hashmap_footer(cap: i32) -> i32 {
    0 - hashmap_magic() - cap
}

@pure fn hashmap_data_len(cap: i32) -> i32 {
    if cap <= 0 { 0 }
    else { if cap > 2147483647 / 3 { 0 }
    else { cap * 3 } }
}

fn hashmap_new(cap: i32) -> i32 {
    let data_len = hashmap_data_len(cap);
    if data_len == 0 { 0 - 1 }
    else {
        __arena_push(hashmap_magic());
        __arena_push(cap);
        let start = __arena_len();
        let mut i: i32 = 0;
        while i < cap {
            __arena_push(0);
            __arena_push(0);
            __arena_push(0);
            i = i + 1;
        }
        __arena_push(hashmap_footer(cap));
        start
    }
}

@pure
fn hashmap_ok(start: i32, cap: i32) -> i32 {
    let data_len = hashmap_data_len(cap);
    if data_len == 0 { 0 }
    else { if start < 2 { 0 }
    else { if __arena_get(start - 2) != hashmap_magic() { 0 }
    else { if __arena_get(start - 1) != cap { 0 }
    else { if data_len > 2147483647 - start { 0 }
    else { if start + data_len >= __arena_len() { 0 }
    else { if __arena_get(start + data_len) != hashmap_footer(cap) { 0 }
    else { 1 } } } } } } }
}

@pure
fn hashmap_hash(k: i32, cap: i32) -> i32 {
    if cap <= 0 { 0 }
    else {
        let mut h = k * 31 + 7;
        let r = h % cap;
        if r < 0 { r + cap } else { r }
    }
}

fn hashmap_put(start: i32, cap: i32, k: i32, v: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 - 1 }
    else {
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
}

@pure
fn hashmap_get(start: i32, cap: i32, k: i32, default: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { default }
    else {
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
}

@pure
fn hashmap_has(start: i32, cap: i32, k: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
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
}

@pure
fn hashmap_size(start: i32, cap: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
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
}

// hashmap_clear(start, cap) -> i32
//   Empty every bucket in place by zeroing the occupancy flag. Key/value
//   slots are not cleared (they're dead until the bucket is reused). Returns
//   start so the caller can chain. NOT @pure (arena mutation).
fn hashmap_clear(start: i32, cap: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 - 1 }
    else {
    let mut i: i32 = 0;
    while i < cap {
        __arena_set(start + i * 3, 0);
        i = i + 1;
    }
    start
    }
}

// hashmap_keys(start, cap) -> i32
//   Allocate a fresh arena slice and push the key of every occupied bucket
//   into it (in bucket order, NOT insertion order). Returns the start index
//   of the new slice; pair with hashmap_size(start, cap) for the count.
//   NOT @pure (arena push).
fn hashmap_keys(start: i32, cap: i32) -> i32 {
    let dst = __arena_len();
    if hashmap_ok(start, cap) == 0 { dst }
    else {
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
}

// hashmap_values(start, cap) -> i32
//   Companion to hashmap_keys: pushes the value of every occupied bucket
//   into a fresh arena slice. Bucket order matches hashmap_keys exactly,
//   so the pair is index-aligned. NOT @pure (arena push).
fn hashmap_values(start: i32, cap: i32) -> i32 {
    let dst = __arena_len();
    if hashmap_ok(start, cap) == 0 { dst }
    else {
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
}

// hashmap_increment(start, cap, k, delta): atomic get-then-put pattern
// for accumulator maps (e.g. word counts). If key present, adds delta
// to its value; if absent, inserts with value=delta. Returns the new
// value at the key, or -1 if the map is full and the key wasn't there.
fn hashmap_increment(start: i32, cap: i32, k: i32, delta: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 - 1 }
    else {
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
}

// hashmap_swap(start, cap, k, new_v): set value to new_v, return old
// value. If key was absent, inserts and returns -1. If map full and
// key not present, returns -1 (caller can't distinguish from a true
// -1 old value — known limitation).
fn hashmap_swap(start: i32, cap: i32, k: i32, new_v: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 - 1 }
    else {
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
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
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
}

// hashmap_count_value_eq(start, cap, target): @pure. Count of occupied
// buckets whose value equals target. Useful for "how many entries map
// to N?" queries (e.g. histogram analysis).
@pure
fn hashmap_count_value_eq(start: i32, cap: i32, target: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            if __arena_get(base + 2) == target { total = total + 1; };
        };
        i = i + 1;
    }
    total
    }
}

// hashmap_sum_values(start, cap): @pure. Sum of values across all
// occupied buckets. Useful for "total count" pattern (e.g. word count
// totals from a histogram).
@pure
fn hashmap_sum_values(start: i32, cap: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i64 = 0_i64;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            total = total + (__arena_get(base + 2) as i64);
        };
        i = i + 1;
    }
    if total > 2147483647_i64 { 2147483647 }
    else { if total < ((0_i64 - 2147483647_i64) - 1_i64) { (0 - 2147483647) - 1 }
    else { total as i32 } }
    }
}

// hashmap_min_value(start, cap): @pure. Walks all occupied buckets,
// returns the minimum value. Returns 0 for an empty map. Caller should
// check hashmap_size beforehand if 0 is a valid value.
@pure
fn hashmap_min_value(start: i32, cap: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut best: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            let v = __arena_get(base + 2);
            if found == 0 { best = v; found = 1; }
            else { if v < best { best = v; }; };
        };
        i = i + 1;
    }
    best
    }
}

// hashmap_load_factor_x100(start, cap): @pure. Returns size * 100 / cap
// — load factor as a percentage. Useful for resize-decision heuristics
// (e.g. "rebuild if load > 75").
@pure
fn hashmap_load_factor_x100(start: i32, cap: i32) -> i32 {
    if cap == 0 { 0 }
    else { hashmap_size(start, cap) * 100 / cap }
}

// hashmap_count_above_threshold(start, cap, threshold): @pure. Count
// of occupied buckets whose value > threshold. "How many words appeared
// more than N times?" pattern.
@pure
fn hashmap_count_above_threshold(start: i32, cap: i32, threshold: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            if __arena_get(base + 2) > threshold { total = total + 1; };
        };
        i = i + 1;
    }
    total
    }
}

// hashmap_max_key(start, cap): @pure. Maximum key across occupied
// buckets. Returns 0 for empty (caller checks size to disambiguate).
@pure
fn hashmap_max_key(start: i32, cap: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut best: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            let k = __arena_get(base + 1);
            if found == 0 { best = k; found = 1; }
            else { if k > best { best = k; }; };
        };
        i = i + 1;
    }
    best
    }
}

// hashmap_min_key(start, cap): @pure. Companion.
@pure
fn hashmap_min_key(start: i32, cap: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut best: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            let k = __arena_get(base + 1);
            if found == 0 { best = k; found = 1; }
            else { if k < best { best = k; }; };
        };
        i = i + 1;
    }
    best
    }
}

// hashmap_sum_keys(start, cap): @pure. Sum of all keys across occupied
// buckets. Useful sanity check / fingerprint.
@pure
fn hashmap_sum_keys(start: i32, cap: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            total = total + __arena_get(base + 1);
        };
        i = i + 1;
    }
    total
    }
}

// hashmap_count_below_threshold(start, cap, threshold): @pure. Count
// of buckets with value < threshold. Companion to count_above.
@pure
fn hashmap_count_below_threshold(start: i32, cap: i32, threshold: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            if __arena_get(base + 2) < threshold { total = total + 1; };
        };
        i = i + 1;
    }
    total
    }
}

// hashmap_has_value(start, cap, target): @pure. 1 if any bucket has
// value == target, 0 otherwise. Useful for "does this map contain
// the value X?" without caring about which key.
@pure
fn hashmap_has_value(start: i32, cap: i32, target: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            if __arena_get(base + 2) == target { found = 1; };
        };
        i = i + 1;
    }
    found
    }
}

// hashmap_argmax_key(start, cap): @pure. Key whose value is largest.
// Returns 0 for empty map. Mirror of "find the most-frequent word".
@pure
fn hashmap_argmax_key(start: i32, cap: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut best_v: i32 = 0;
    let mut best_k: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            let v = __arena_get(base + 2);
            let k = __arena_get(base + 1);
            if found == 0 { best_v = v; best_k = k; found = 1; }
            else { if v > best_v { best_v = v; best_k = k; }; };
        };
        i = i + 1;
    }
    best_k
    }
}

// hashmap_argmin_key(start, cap): @pure. Companion.
@pure
fn hashmap_argmin_key(start: i32, cap: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut best_v: i32 = 0;
    let mut best_k: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            let v = __arena_get(base + 2);
            let k = __arena_get(base + 1);
            if found == 0 { best_v = v; best_k = k; found = 1; }
            else { if v < best_v { best_v = v; best_k = k; }; };
        };
        i = i + 1;
    }
    best_k
    }
}

// hashmap_count_key_eq(start, cap, key): @pure. Returns 1 if key
// present in map, 0 otherwise. Functionally equivalent to hashmap_has
// but named symmetrically with hashmap_count_value_eq.
@pure
fn hashmap_count_key_eq(start: i32, cap: i32, key: i32) -> i32 {
    hashmap_has(start, cap, key)
}

// hashmap_max_key_with_value(start, cap, target): @pure. Largest key
// among buckets whose value == target. Returns 0 for none.
@pure
fn hashmap_max_key_with_value(start: i32, cap: i32, target: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut best: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            if __arena_get(base + 2) == target {
                let k = __arena_get(base + 1);
                if found == 0 { best = k; found = 1; }
                else { if k > best { best = k; }; };
            };
        };
        i = i + 1;
    }
    best
    }
}

// hashmap_avg_value_x100(start, cap): @pure. Sum / size as percent
// (multiplied by 100 to avoid integer truncation). Useful for rough
// averaging in count-based maps. Returns 0 for empty maps.
@pure
fn hashmap_avg_value_x100(start: i32, cap: i32) -> i32 {
    let n = hashmap_size(start, cap);
    if n == 0 { 0 }
    else {
        let mut i: i32 = 0;
        let mut total: i64 = 0_i64;
        while i < cap {
            let base = start + i * 3;
            if __arena_get(base) == 1 {
                total = total + (__arena_get(base + 2) as i64);
            };
            i = i + 1;
        }
        let scaled: i64 = total * 100_i64 / (n as i64);
        if scaled > 2147483647_i64 { 2147483647 }
        else { if scaled < ((0_i64 - 2147483647_i64) - 1_i64) { (0 - 2147483647) - 1 }
        else { scaled as i32 } }
    }
}

// hashmap_is_empty(start, cap): @pure. 1 if map has no entries.
@pure
fn hashmap_is_empty(start: i32, cap: i32) -> i32 {
    if hashmap_size(start, cap) == 0 { 1 } else { 0 }
}

// hashmap_min_key_with_value(start, cap, target): @pure. Smallest key
// among buckets whose value == target. Returns 0 for none.
@pure
fn hashmap_min_key_with_value(start: i32, cap: i32, target: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    let mut best: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            if __arena_get(base + 2) == target {
                let k = __arena_get(base + 1);
                if found == 0 { best = k; found = 1; }
                else { if k < best { best = k; }; };
            };
        };
        i = i + 1;
    }
    best
    }
}

// hashmap_count_value_in_range(start, cap, lo, hi): @pure. Count of
// occupied buckets whose value is in [lo, hi] inclusive.
@pure
fn hashmap_count_value_in_range(start: i32, cap: i32, lo: i32, hi: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            let v = __arena_get(base + 2);
            if v >= lo {
                if v <= hi { total = total + 1; };
            };
        };
        i = i + 1;
    }
    total
    }
}

// hashmap_capacity(start, cap): @pure. Returns cap (alias for clarity).
@pure
fn hashmap_capacity(start: i32, cap: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 } else { cap }
}

// hashmap_remaining_slots(start, cap): @pure. cap - size. How many
// more inserts can fit before linear-probe failure.
@pure
fn hashmap_remaining_slots(start: i32, cap: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else { cap - hashmap_size(start, cap) }
}

// hashmap_count_key_in_range(start, cap, lo, hi): @pure. Count of
// occupied buckets whose key is in [lo, hi] inclusive.
@pure
fn hashmap_count_key_in_range(start: i32, cap: i32, lo: i32, hi: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            let k = __arena_get(base + 1);
            if k >= lo {
                if k <= hi { total = total + 1; };
            };
        };
        i = i + 1;
    }
    total
    }
}

// hashmap_count_key_above(start, cap, threshold): @pure. Count of keys
// greater than threshold.
@pure
fn hashmap_count_key_above(start: i32, cap: i32, threshold: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            if __arena_get(base + 1) > threshold { total = total + 1; };
        };
        i = i + 1;
    }
    total
    }
}

// hashmap_count_key_below(start, cap, threshold): @pure. Companion.
@pure
fn hashmap_count_key_below(start: i32, cap: i32, threshold: i32) -> i32 {
    if hashmap_ok(start, cap) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < cap {
        let base = start + i * 3;
        if __arena_get(base) == 1 {
            if __arena_get(base + 1) < threshold { total = total + 1; };
        };
        i = i + 1;
    }
    total
    }
}

// hashmap_max_value_with_key(start, cap, key_target): @pure. Returns
// the value at key_target, or 0 if absent. Alias-y but symmetric with
// the *_with_value family.
@pure
fn hashmap_max_value_with_key(start: i32, cap: i32, key_target: i32) -> i32 {
    hashmap_get(start, cap, key_target, 0)
}
