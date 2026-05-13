// helixc/stdlib/agi_memory.hx — AGI working memory primitives.
//
// Phase 4 step 1: a small key-value working memory in Helix. Bounded
// capacity (default 16 slots). Items stored as (key, value, recency)
// triples. Eviction is least-recently-used (LRU).
//
// This is one of the foundational pieces an AGI uses for short-term
// state management — analogous to the prefrontal cortex's working
// memory in cognitive architectures (Soar, ACT-R, EPIC). Long-term
// episodic memory comes in step 2; semantic memory in step 3.
//
// API:
//   wm_new()                   -> i32       allocate empty WM, return start
//   wm_store(start, key, val)  -> i32       store (key, val); evict LRU if full
//   wm_load(start, key)        -> i32       return val if present, -1 if absent
//   wm_size(start)             -> i32       number of slots in use
//   wm_clear(start)            -> i32       drop all items (size -> 0)
//
// Layout (per WM):
//   slot 0: size (current count, 0..16)
//   slot 1: tick (monotonic counter for recency)
//   slot 2..2+16*3: 16 entries, each 3 slots (key, val, last_used_tick)
//
// License: Apache 2.0

// Foundational AGI-safe scalar refinements. These keep common uncertainty
// and physical-measure constraints available by default instead of forcing
// every Helix program to redefine them locally.
type Confidence = f64 where 0.0 <= self <= 1.0;
type Probability = f64 where 0.0 <= self <= 1.0;
type DistanceMeters = f64 where self >= 0.0;

@pure fn wm_capacity() -> i32 { 16 }

fn wm_new() -> i32 {
    let start = __arena_len();
    __arena_push(0);   // size
    __arena_push(0);   // tick counter
    let mut i: i32 = 0;
    let cap = wm_capacity();
    while i < cap {
        __arena_push(0);   // key
        __arena_push(0);   // val
        __arena_push(0);   // last_used_tick
        i = i + 1;
    }
    start
}

@pure fn wm_size(start: i32) -> i32 {
    __arena_get(start)
}

fn wm_clear(start: i32) -> i32 {
    __arena_set(start, 0);
    __arena_set(start + 1, 0);
    0
}

// Internal helper: linear scan for a key. Returns slot offset (2..2+cap*3
// stepping by 3) or -1 if not found.
@pure
fn wm_find(start: i32, key: i32) -> i32 {
    let mut i: i32 = 0;
    let count = __arena_get(start);
    let mut found: i32 = 0 - 1;
    while i < count {
        let off = start + 2 + i * 3;
        if __arena_get(off) == key {
            if found < 0 { found = off; }
        }
        i = i + 1;
    }
    found
}

// Internal helper: find LRU slot offset.
@pure
fn wm_lru_slot(start: i32) -> i32 {
    let cap = wm_capacity();
    let mut i: i32 = 0;
    let mut best_off: i32 = start + 2;
    let mut best_tick: i32 = __arena_get(start + 2 + 2);
    while i < cap {
        let off = start + 2 + i * 3;
        let tick = __arena_get(off + 2);
        if tick < best_tick {
            best_tick = tick;
            best_off = off;
        }
        i = i + 1;
    }
    best_off
}

// Store key/value. If key already present, update in place (refreshing
// recency). If full, evict the LRU entry. Returns the new size.
fn wm_store(start: i32, key: i32, val: i32) -> i32 {
    let new_tick = __arena_get(start + 1) + 1;
    __arena_set(start + 1, new_tick);
    let existing = wm_find(start, key);
    if existing >= 0 {
        __arena_set(existing + 1, val);
        __arena_set(existing + 2, new_tick);
        __arena_get(start)
    } else {
        let count = __arena_get(start);
        let cap = wm_capacity();
        if count < cap {
            let off = start + 2 + count * 3;
            __arena_set(off, key);
            __arena_set(off + 1, val);
            __arena_set(off + 2, new_tick);
            __arena_set(start, count + 1);
            count + 1
        } else {
            let off = wm_lru_slot(start);
            __arena_set(off, key);
            __arena_set(off + 1, val);
            __arena_set(off + 2, new_tick);
            cap
        }
    }
}

// Load value for key; refresh recency. Returns -1 if absent.
fn wm_load(start: i32, key: i32) -> i32 {
    let off = wm_find(start, key);
    if off < 0 { 0 - 1 }
    else {
        let new_tick = __arena_get(start + 1) + 1;
        __arena_set(start + 1, new_tick);
        __arena_set(off + 2, new_tick);
        __arena_get(off + 1)
    }
}

// =========================================================================
// Phase 4 step 2: episodic memory — timestamped event log.
// =========================================================================
//
// Append-only ring buffer of (timestamp, event_kind, payload) triples.
// The AGI's "what happened when" memory. Capacity 64; oldest entries
// overwrite when full. Useful for credit assignment, planning by
// retrospection, and "I tried X earlier" deduplication.
//
// Layout:
//   slot 0: head (next write index, 0..cap)
//   slot 1: count (entries written, capped at cap once filled)
//   slot 2: tick (monotonic event timestamp)
//   slot 3..3+cap*3: cap entries, each 3 slots (tick, kind, payload)

@pure fn ep_capacity() -> i32 { 64 }

fn ep_new() -> i32 {
    let start = __arena_len();
    __arena_push(0);   // head
    __arena_push(0);   // count
    __arena_push(0);   // tick
    let mut i: i32 = 0;
    let cap = ep_capacity();
    while i < cap {
        __arena_push(0);   // tick
        __arena_push(0);   // kind
        __arena_push(0);   // payload
        i = i + 1;
    }
    start
}

@pure fn ep_count(start: i32) -> i32 {
    __arena_get(start + 1)
}

@pure fn ep_tick(start: i32) -> i32 {
    __arena_get(start + 2)
}

// Append an event (kind, payload). Returns new tick.
fn ep_record(start: i32, kind: i32, payload: i32) -> i32 {
    let cap = ep_capacity();
    let head = __arena_get(start);
    let new_tick = __arena_get(start + 2) + 1;
    __arena_set(start + 2, new_tick);
    let off = start + 3 + head * 3;
    __arena_set(off, new_tick);
    __arena_set(off + 1, kind);
    __arena_set(off + 2, payload);
    let new_head = (head + 1) % cap;
    __arena_set(start, new_head);
    let cnt = __arena_get(start + 1);
    if cnt < cap {
        __arena_set(start + 1, cnt + 1);
    }
    new_tick
}

// Read i'th event payload (chronological; 0 = oldest still in buffer).
// Returns -1 if i >= count.
@pure
fn ep_payload_at(start: i32, i: i32) -> i32 {
    let cap = ep_capacity();
    let cnt = __arena_get(start + 1);
    if i >= cnt { 0 - 1 }
    else {
        let head = __arena_get(start);
        let pos = if cnt < cap {
            i
        } else {
            (head + i) % cap
        };
        let off = start + 3 + pos * 3;
        __arena_get(off + 2)
    }
}

// Search backwards for the most recent event of `kind`. Return its
// payload, or -1 if no event of that kind in the buffer.
@pure
fn ep_recent_kind(start: i32, kind: i32) -> i32 {
    let cap = ep_capacity();
    let cnt = __arena_get(start + 1);
    let head = __arena_get(start);
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < cnt {
        let pos = if cnt < cap {
            cnt - 1 - i
        } else {
            (head + cap - 1 - i) % cap
        };
        let off = start + 3 + pos * 3;
        if __arena_get(off + 1) == kind {
            if found < 0 { found = __arena_get(off + 2); }
        }
        i = i + 1;
    }
    found
}

// =========================================================================
// Accessors mirroring option_*/result_* style — pure (no LRU-tick mutation).
// =========================================================================

// 1 if key is present in WM, 0 otherwise. @pure (no recency refresh).
@pure
fn wm_has(start: i32, key: i32) -> i32 {
    let off = wm_find(start, key);
    if off < 0 { 0 } else { 1 }
}

// Read value for key without refreshing recency. Returns -1 if absent.
// Sibling of wm_load (which mutates the LRU tick); use this when only
// a read is needed and the read should not disturb eviction order.
@pure
fn wm_peek(start: i32, key: i32) -> i32 {
    let off = wm_find(start, key);
    if off < 0 { 0 - 1 } else { __arena_get(off + 1) }
}

// Sibling of ep_payload_at: read i'th event kind in chronological order
// (0 = oldest still in buffer). Returns -1 if i >= count.
@pure
fn ep_kind_at(start: i32, i: i32) -> i32 {
    let cap = ep_capacity();
    let cnt = __arena_get(start + 1);
    if i >= cnt { 0 - 1 }
    else {
        let head = __arena_get(start);
        let pos = if cnt < cap {
            i
        } else {
            (head + i) % cap
        };
        let off = start + 3 + pos * 3;
        __arena_get(off + 1)
    }
}

// Count events of a given kind in the buffer.
@pure
fn ep_count_kind(start: i32, kind: i32) -> i32 {
    let cap = ep_capacity();
    let cnt = __arena_get(start + 1);
    let head = __arena_get(start);
    let mut i: i32 = 0;
    let mut count: i32 = 0;
    while i < cnt {
        let pos = if cnt < cap {
            i
        } else {
            (head + i) % cap
        };
        let off = start + 3 + pos * 3;
        if __arena_get(off + 1) == kind { count = count + 1; }
        i = i + 1;
    }
    count
}
