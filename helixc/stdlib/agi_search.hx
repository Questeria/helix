// helixc/stdlib/agi_search.hx — search primitives for AGI planning.
//
// Phase 4 step 3: simple graph search (BFS, hill-climbing) for planning
// problems. The AGI uses these to find action sequences that reach goal
// states. State graph is represented as arena-stored adjacency, with
// integer state IDs and action IDs.
//
// API:
//   bfs_queue_new()                  -> i32   FIFO queue (start)
//   bfs_enqueue(q, state)            -> i32   push state; return 0
//   bfs_dequeue(q)                   -> i32   pop oldest state; -1 if empty
//   bfs_size(q)                      -> i32   current queue length
//
//   hillclimb_step(state, neighbors_start, n, scoring_offset_table, scoring_start) -> i32
//      Picks highest-scoring neighbor of state via lookup tables.
//      Score table indexed by state id; -1 returned if all neighbors
//      score lower than current state.
//
// All primitives are integer-only and arena-backed. Float scoring
// pending Phase 2.2 step 2.
//
// License: Apache 2.0

@pure fn bfs_capacity() -> i32 { 256 }

fn bfs_queue_new() -> i32 {
    let start = __arena_len();
    __arena_push(0);   // head (next pop index)
    __arena_push(0);   // tail (next push index)
    __arena_push(0);   // count
    let mut i: i32 = 0;
    let cap = bfs_capacity();
    while i < cap {
        __arena_push(0);
        i = i + 1;
    }
    start
}

fn bfs_enqueue(q: i32, state: i32) -> i32 {
    let cap = bfs_capacity();
    let cnt = __arena_get(q + 2);
    if cnt >= cap {
        0 - 1
    } else {
        let tail = __arena_get(q + 1);
        __arena_set(q + 3 + tail, state);
        let new_tail = (tail + 1) % cap;
        __arena_set(q + 1, new_tail);
        __arena_set(q + 2, cnt + 1);
        0
    }
}

fn bfs_dequeue(q: i32) -> i32 {
    let cap = bfs_capacity();
    let cnt = __arena_get(q + 2);
    if cnt == 0 {
        0 - 1
    } else {
        let head = __arena_get(q);
        let v = __arena_get(q + 3 + head);
        let new_head = (head + 1) % cap;
        __arena_set(q, new_head);
        __arena_set(q + 2, cnt - 1);
        v
    }
}

@pure fn bfs_size(q: i32) -> i32 {
    __arena_get(q + 2)
}

// Visited set: bounded-size linear-probe table of state ids.
// Layout: slot 0 = count, slot 1..1+cap = entries (-1 means empty slot).
@pure fn visited_capacity() -> i32 { 256 }

fn visited_new() -> i32 {
    let start = __arena_len();
    __arena_push(0);
    let cap = visited_capacity();
    let mut i: i32 = 0;
    while i < cap {
        __arena_push(0 - 1);
        i = i + 1;
    }
    start
}

// Returns 1 if marked, 0 if already present.
fn visited_mark(v: i32, state: i32) -> i32 {
    let cap = visited_capacity();
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    while i < cap {
        let cur = __arena_get(v + 1 + i);
        if cur == state { found = 1; }
        i = i + 1;
    }
    if found == 1 {
        0
    } else {
        let cnt = __arena_get(v);
        if cnt < cap {
            __arena_set(v + 1 + cnt, state);
            __arena_set(v, cnt + 1);
            1
        } else {
            0 - 1
        }
    }
}

@pure
fn visited_has(v: i32, state: i32) -> i32 {
    let cap = visited_capacity();
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    while i < cap {
        if __arena_get(v + 1 + i) == state { found = 1; }
        i = i + 1;
    }
    found
}

// Pick the highest-scoring neighbor from a list. neighbors_start points
// to n state ids in the arena; scoring_start[state_id] gives the score
// (so caller pre-builds a state -> score table indexed by state id).
// Returns the chosen neighbor id, or -1 if the list is empty.
@pure
fn hillclimb_step(neighbors_start: i32, n: i32, scoring_start: i32) -> i32 {
    if n == 0 { 0 - 1 }
    else {
        let mut i: i32 = 1;
        let mut best: i32 = __arena_get(neighbors_start);
        let mut best_score: i32 = __arena_get(scoring_start + best);
        while i < n {
            let cand = __arena_get(neighbors_start + i);
            let score = __arena_get(scoring_start + cand);
            if score > best_score {
                best = cand;
                best_score = score;
            }
            i = i + 1;
        }
        best
    }
}

// =========================================================================
// Phase 4 perfection: A*, beam search, attention.
// =========================================================================

// Priority queue (min-heap on score) stored as parallel arrays:
//   slot 0: count
//   slot 1: cap
//   slot 2..2+cap: states
//   slot 2+cap..2+2*cap: scores (lower = better)
@pure fn pq_capacity() -> i32 { 256 }

fn pq_new() -> i32 {
    let cap = pq_capacity();
    let start = __arena_len();
    __arena_push(0);
    __arena_push(cap);
    let mut i: i32 = 0;
    while i < cap {
        __arena_push(0);
        i = i + 1;
    }
    let mut j: i32 = 0;
    while j < cap {
        __arena_push(0);
        j = j + 1;
    }
    start
}

@pure fn pq_size(q: i32) -> i32 { __arena_get(q) }

// Insert: linear scan to insert in sorted order (simple, O(n); good for
// the small AGI problem sizes Phase 4 targets).
fn pq_insert(q: i32, state: i32, score: i32) -> i32 {
    let cap = __arena_get(q + 1);
    let cnt = __arena_get(q);
    if cnt >= cap {
        0 - 1
    } else {
        // Find insertion index where scores stay non-decreasing.
        let mut idx: i32 = 0;
        while idx < cnt {
            if __arena_get(q + 2 + cap + idx) <= score { idx = idx + 1; }
            else { idx = cnt; }
        }
        // For simplicity: scan forward, find where score < scores[i],
        // re-do without breaking (idx already scanned past).
        let mut ins: i32 = cnt;
        let mut k: i32 = 0;
        while k < cnt {
            if __arena_get(q + 2 + cap + k) > score {
                if ins == cnt { ins = k; }
            }
            k = k + 1;
        }
        // Shift elements >= ins right by one.
        let mut s: i32 = cnt;
        while s > ins {
            __arena_set(q + 2 + s, __arena_get(q + 2 + s - 1));
            __arena_set(q + 2 + cap + s, __arena_get(q + 2 + cap + s - 1));
            s = s - 1;
        }
        __arena_set(q + 2 + ins, state);
        __arena_set(q + 2 + cap + ins, score);
        __arena_set(q, cnt + 1);
        0
    }
}

// Pop the lowest-scoring entry. Returns the state id, or -1 if empty.
fn pq_pop_min(q: i32) -> i32 {
    let cap = __arena_get(q + 1);
    let cnt = __arena_get(q);
    if cnt == 0 { 0 - 1 }
    else {
        let v = __arena_get(q + 2);
        // Shift left.
        let mut i: i32 = 0;
        while i < cnt - 1 {
            __arena_set(q + 2 + i, __arena_get(q + 2 + i + 1));
            __arena_set(q + 2 + cap + i, __arena_get(q + 2 + cap + i + 1));
            i = i + 1;
        }
        __arena_set(q, cnt - 1);
        v
    }
}

// Beam search step: keep the top k highest-scoring entries from a
// candidate list. Returns count kept. The result lives at result_start
// in the same shape as candidates (state ids).
@pure
fn beam_top_k(candidates_start: i32, n: i32, scoring_start: i32,
              result_start: i32, k: i32) -> i32 {
    if n == 0 { 0 }
    else {
        let mut copied: i32 = 0;
        let kept = if n < k { n } else { k };
        // Selection sort first `kept` highest-scoring by direct copy.
        // O(n*k) but k is small for beam.
        let mut chosen_count: i32 = 0;
        while chosen_count < kept {
            let mut best_idx: i32 = 0 - 1;
            let mut best_score: i32 = 0;
            let mut i: i32 = 0;
            while i < n {
                let cand = __arena_get(candidates_start + i);
                let score = __arena_get(scoring_start + cand);
                // Skip already-chosen (linear scan over result_start).
                let mut j: i32 = 0;
                let mut already: i32 = 0;
                while j < chosen_count {
                    if __arena_get(result_start + j) == cand { already = 1; }
                    j = j + 1;
                }
                if already == 0 {
                    if best_idx < 0 {
                        best_idx = i; best_score = score;
                    } else {
                        if score > best_score {
                            best_idx = i; best_score = score;
                        }
                    }
                }
                i = i + 1;
            }
            if best_idx >= 0 {
                __arena_set(result_start + chosen_count,
                            __arena_get(candidates_start + best_idx));
                chosen_count = chosen_count + 1;
            } else {
                chosen_count = kept;
            }
        }
        chosen_count
    }
}

// Attention: scaled dot-product attention single-head, integer-only
// (proportional to actual softmax(QK^T/d)V).
//   query_start : 1 query of dim d
//   keys_start  : n keys of dim d (row-major)
//   values_start: n values of dim d (row-major)
//   output      : n entries of dim d (one weighted-output per query
//                  position; we have 1 query so output is 1 row of d)
// Approximation: instead of softmax, use linear weighting by raw
// dot product (since float-tensor weights would need real softmax;
// integer-attention via dot-product approximates relative magnitudes).
fn attention_dot(query_start: i32, keys_start: i32, values_start: i32,
                 n: i32, d: i32, output_start: i32) -> i32 {
    // Initialize output to 0.
    let mut o: i32 = 0;
    while o < d { __arena_set(output_start + o, 0); o = o + 1; }
    // Compute total attention weight for normalization (sum over all keys).
    let mut total_w: i32 = 0;
    let mut k: i32 = 0;
    while k < n {
        let mut dim: i32 = 0;
        let mut dot: i32 = 0;
        while dim < d {
            dot = dot + __arena_get(query_start + dim) *
                        __arena_get(keys_start + k * d + dim);
            dim = dim + 1;
        }
        // Clamp to non-negative (no exp needed for this approximation).
        let w = if dot > 0 { dot } else { 0 };
        // Accumulate weighted value.
        let mut vd: i32 = 0;
        while vd < d {
            let cur = __arena_get(output_start + vd);
            __arena_set(output_start + vd,
                        cur + w * __arena_get(values_start + k * d + vd));
            vd = vd + 1;
        }
        total_w = total_w + w;
        k = k + 1;
    }
    // Normalize by total_w (skip if zero).
    if total_w > 0 {
        let mut nd: i32 = 0;
        while nd < d {
            let cur = __arena_get(output_start + nd);
            __arena_set(output_start + nd, cur / total_w);
            nd = nd + 1;
        }
    }
    0
}
