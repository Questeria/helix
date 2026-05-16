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
@pure fn bfs_magic() -> i32 { 6106101 }
@pure fn bfs_slot_count() -> i32 { 3 + bfs_capacity() }
@pure fn bfs_footer() -> i32 { 0 - bfs_magic() - bfs_slot_count() - 1 }

fn bfs_queue_new() -> i32 {
    __arena_push(bfs_magic());
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
    __arena_push(bfs_footer());
    start
}

@pure fn bfs_ok(q: i32) -> i32 {
    let cap = bfs_capacity();
    if q <= 0 { 0 }
    else { if __arena_get(q - 1) != bfs_magic() { 0 }
    else { if q > 2147483647 - bfs_slot_count() { 0 }
    else { if q + bfs_slot_count() >= __arena_len() { 0 }
    else { if __arena_get(q + bfs_slot_count()) != bfs_footer() { 0 }
    else { if arena_span_in_tensor_payload(q - 1, bfs_slot_count() + 2) != 0 { 0 }
    else {
        let head = __arena_get(q);
        let tail = __arena_get(q + 1);
        let cnt = __arena_get(q + 2);
        if head < 0 { 0 }
        else { if head >= cap { 0 }
        else { if tail < 0 { 0 }
        else { if tail >= cap { 0 }
        else { if cnt < 0 { 0 }
        else { if cnt > cap { 0 } else { 1 } } } } } }
    }}}}}}
}

fn bfs_enqueue(q: i32, state: i32) -> i32 {
    let cap = bfs_capacity();
    if bfs_ok(q) == 0 { 0 - 1 }
    else {
    let cnt = __arena_get(q + 2);
    if cnt >= cap { 0 - 1
    } else {
        let tail = __arena_get(q + 1);
        __arena_set(q + 3 + tail, state);
        let new_tail = (tail + 1) % cap;
        __arena_set(q + 1, new_tail);
        __arena_set(q + 2, cnt + 1);
        0
    }
    }
}

fn bfs_dequeue(q: i32) -> i32 {
    let cap = bfs_capacity();
    if bfs_ok(q) == 0 { 0 - 1 }
    else {
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
}

@pure fn bfs_size(q: i32) -> i32 {
    if bfs_ok(q) == 0 { 0 } else { __arena_get(q + 2) }
}

@pure fn bfs_is_empty(q: i32) -> i32 {
    if bfs_ok(q) == 0 { 1 }
    else { if __arena_get(q + 2) == 0 { 1 } else { 0 } }
}

// Visited set: bounded-size linear-probe table of state ids.
// Layout: slot 0 = count, slot 1..1+cap = entries (-1 means empty slot).
@pure fn visited_capacity() -> i32 { 256 }
@pure fn visited_magic() -> i32 { 6206201 }
@pure fn visited_slot_count() -> i32 { 1 + visited_capacity() }
@pure fn visited_footer() -> i32 { 0 - visited_magic() - visited_slot_count() - 1 }

fn visited_new() -> i32 {
    __arena_push(visited_magic());
    let start = __arena_len();
    __arena_push(0);
    let cap = visited_capacity();
    let mut i: i32 = 0;
    while i < cap {
        __arena_push(0 - 1);
        i = i + 1;
    }
    __arena_push(visited_footer());
    start
}

@pure fn visited_ok(v: i32) -> i32 {
    let cap = visited_capacity();
    if v <= 0 { 0 }
    else { if __arena_get(v - 1) != visited_magic() { 0 }
    else { if v > 2147483647 - visited_slot_count() { 0 }
    else { if v + visited_slot_count() >= __arena_len() { 0 }
    else { if __arena_get(v + visited_slot_count()) != visited_footer() { 0 }
    else { if arena_span_in_tensor_payload(v - 1, visited_slot_count() + 2) != 0 { 0 }
    else {
        let cnt = __arena_get(v);
        if cnt < 0 { 0 }
        else { if cnt > cap { 0 } else { 1 } }
    }}}}}}
}

// Returns 1 if marked, 0 if already present.
fn visited_mark(v: i32, state: i32) -> i32 {
    let cap = visited_capacity();
    if visited_ok(v) == 0 { 0 - 1 }
    else {
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
}

@pure
fn visited_has(v: i32, state: i32) -> i32 {
    let cap = visited_capacity();
    if visited_ok(v) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    while i < cap {
        if __arena_get(v + 1 + i) == state { found = 1; }
        i = i + 1;
    }
    found
    }
}

@pure fn visited_count(v: i32) -> i32 {
    if visited_ok(v) == 0 { 0 } else { __arena_get(v) }
}

// Pick the highest-scoring neighbor from a list. neighbors_start points
// to n state ids in the arena; scoring_start[state_id] gives the score
// (so caller pre-builds a state -> score table indexed by state id).
// Returns the chosen neighbor id, or -1 if the list is empty.
@pure
fn hillclimb_step(neighbors_start: i32, n: i32, scoring_start: i32) -> i32 {
    if n <= 0 { 0 - 1 }
    else { if t1d_slice_ok(neighbors_start, n) == 0 { 0 - 1 }
    else {
        let mut max_state: i32 = 0;
        let mut scan: i32 = 0;
        let mut valid: i32 = 1;
        while scan < n {
            let st = __arena_get(neighbors_start + scan);
            if st < 0 { valid = 0; }
            else { if st > max_state { max_state = st; } }
            scan = scan + 1;
        }
        if valid == 0 { 0 - 1 }
        else { if t1d_slice_ok(scoring_start, max_state + 1) == 0 { 0 - 1 }
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
    }}}
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
@pure fn pq_magic() -> i32 { 6306301 }
@pure fn pq_slot_count() -> i32 { 2 + pq_capacity() * 2 }
@pure fn pq_footer() -> i32 { 0 - pq_magic() - pq_slot_count() - 1 }

fn pq_new() -> i32 {
    let cap = pq_capacity();
    __arena_push(pq_magic());
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
    __arena_push(pq_footer());
    start
}

@pure fn pq_ok(q: i32) -> i32 {
    let cap = pq_capacity();
    if q <= 0 { 0 }
    else { if __arena_get(q - 1) != pq_magic() { 0 }
    else { if q > 2147483647 - pq_slot_count() { 0 }
    else { if q + pq_slot_count() >= __arena_len() { 0 }
    else { if __arena_get(q + pq_slot_count()) != pq_footer() { 0 }
    else { if arena_span_in_tensor_payload(q - 1, pq_slot_count() + 2) != 0 { 0 }
    else {
        let cnt = __arena_get(q);
        let stored_cap = __arena_get(q + 1);
        if stored_cap != cap { 0 }
        else { if cnt < 0 { 0 }
        else { if cnt > cap { 0 } else { 1 } } }
    }}}}}}
}

@pure fn pq_size(q: i32) -> i32 {
    if pq_ok(q) == 0 { 0 } else { __arena_get(q) }
}

@pure fn pq_is_empty(q: i32) -> i32 {
    if pq_ok(q) == 0 { 1 }
    else { if __arena_get(q) == 0 { 1 } else { 0 } }
}

// Peek lowest-scoring entry without removing it. Returns -1 if empty.
// Pairs with pq_pop_min; the priority-queue layout keeps the lowest
// score at index 0, so this is a constant-time read.
@pure fn pq_peek_min(q: i32) -> i32 {
    if pq_ok(q) == 0 { 0 - 1 }
    else { if __arena_get(q) == 0 { 0 - 1 }
    else { __arena_get(q + 2) }
    }
}

// Insert: linear scan to insert in sorted order (simple, O(n); good for
// the small AGI problem sizes Phase 4 targets).
fn pq_insert(q: i32, state: i32, score: i32) -> i32 {
    if pq_ok(q) == 0 { 0 - 1 }
    else {
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
}

// Pop the lowest-scoring entry. Returns the state id, or -1 if empty.
fn pq_pop_min(q: i32) -> i32 {
    if pq_ok(q) == 0 { 0 - 1 }
    else {
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
}

// Beam search step: keep the top k highest-scoring entries from a
// candidate list. Returns count kept. The result lives at result_start
// in the same shape as candidates (state ids).
@pure
fn beam_top_k(candidates_start: i32, n: i32, scoring_start: i32,
              result_start: i32, k: i32) -> i32 {
    if n < 0 { 0 - 1 }
    else { if k < 0 { 0 - 1 }
    else { if n == 0 { 0 }
    else { if t1d_slice_ok(candidates_start, n) == 0 { 0 - 1 }
    else {
        let kept = if n < k { n } else { k };
        if t1d_slice_ok(result_start, kept) == 0 { 0 - 1 }
        else {
        let mut max_state: i32 = 0;
        let mut scan: i32 = 0;
        let mut valid: i32 = 1;
        while scan < n {
            let st = __arena_get(candidates_start + scan);
            if st < 0 { valid = 0; }
            else { if st > max_state { max_state = st; } }
            scan = scan + 1;
        }
        if valid == 0 { 0 - 1 }
        else { if t1d_slice_ok(scoring_start, max_state + 1) == 0 { 0 - 1 }
        else {
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
    }}}
    }
    }}}
}

// =========================================================================
// A* search: PQ ordered by f(n) = g(n) + h(n).
// =========================================================================
//
// Caller provides:
//   - g_table[state]: cost so far (caller updates during expansion)
//   - h_table[state]: heuristic estimate to goal
//   - successors_of(state): generated externally; A* loop just inserts/pops.
//
// A* itself is the priority-queue management + closed-set + path
// reconstruction. The actual graph traversal is the caller's loop.
// API:
//   astar_priority(g_start, h_start, state)    -> i32
//        f(n) = g(n) + h(n); use as the PQ score.
//   astar_path_set(came_from_start, child, parent) -> i32
//        Record came_from[child] = parent for path reconstruction.
//   astar_path_get(came_from_start, state)     -> i32
//        Read came_from[state]; -1 if unset.

@pure
fn astar_priority(g_start: i32, h_start: i32, state: i32) -> i32 {
    if state < 0 { 0 - 1 }
    else { if t1d_slice_ok(g_start, state + 1) == 0 { 0 - 1 }
    else { if t1d_slice_ok(h_start, state + 1) == 0 { 0 - 1 }
    else { __arena_get(g_start + state) + __arena_get(h_start + state) } } }
}

fn astar_path_set(came_from_start: i32, child: i32, parent: i32) -> i32 {
    if child < 0 { 0 - 1 }
    else { if t1d_slice_ok(came_from_start, child + 1) == 0 { 0 - 1 }
    else {
        __arena_set(came_from_start + child, parent);
        0
    } }
}

@pure
fn astar_path_get(came_from_start: i32, state: i32) -> i32 {
    if state < 0 { 0 - 1 }
    else { if t1d_slice_ok(came_from_start, state + 1) == 0 { 0 - 1 }
    else { __arena_get(came_from_start + state) } }
}

// Reconstruct path from start to goal by walking came_from backwards.
// Writes path into out_start in REVERSE order (goal first, start last).
// Stops on any of:
//   - cur < 0          (no parent recorded — broken came_from chain)
//   - prev == cur      (start node convention: came_from[start] = start)
//   - len >= max_len   (output buffer full)
// Writes a -1 terminator at out_start[len] if there's room for it, so
// callers that scan for sentinels can do so safely. Returns the actual
// number of path entries written (not counting the terminator).
//
// Pre-fix: this function had a subtle bug — on early exit it set
// len = max_len, then a second pass walked the buffer counting non-
// negative entries. Uninitialized buffer slots are arbitrary, so the
// returned count was unreliable unless the caller pre-zeroed (or pre-
// minused) the buffer. Now we just return the loop-tracked len.
fn astar_reconstruct(came_from_start: i32, goal: i32, out_start: i32,
                     max_len: i32) -> i32 {
    if max_len <= 0 { 0 }
    else { if t1d_slice_ok(out_start, max_len) == 0 { 0 - 1 }
    else {
    let mut cur = goal;
    let mut len: i32 = 0;
    let mut keep: i32 = 1;
    while keep == 1 {
        if cur < 0 { keep = 0; }
        else { if len >= max_len { keep = 0; }
        else { if t1d_slice_ok(came_from_start, cur + 1) == 0 { keep = 0; }
        else {
            __arena_set(out_start + len, cur);
            len = len + 1;
            let prev = __arena_get(came_from_start + cur);
            if prev == cur { keep = 0; }
            else { cur = prev; }
        }}};
    }
    // Write -1 terminator if the buffer has room.
    if len < max_len {
        __arena_set(out_start + len, 0 - 1);
    }
    len
    }}
}

// =========================================================================
// Attention with softmax on f32 (transformer-style scaled dot-product).
// =========================================================================
//
// Single query, n keys/values, dim d. f32 throughout (uses tf2d_get and
// __exp from transcendentals.hx). Output: 1 vector of dim d.
//
//   scores[k] = dot(q, keys[k]) / sqrt(d)         (for stability)
//   probs = softmax(scores)
//   out = sum_k probs[k] * values[k]

fn attention_softmax_f32(q_start: i32, keys_start: i32, vals_start: i32,
                          n: i32, d: i32, out_start: i32) -> i32 {
    if n <= 0 { t2d_error() }
    else { if d <= 0 { t2d_error() }
    else { if n > 2147483647 / d { t2d_error() }
    else {
    let total = n * d;
    if t1d_slice_ok(q_start, d) == 0 { t2d_error() }
    else { if t1d_slice_ok(keys_start, total) == 0 { t2d_error() }
    else { if t1d_slice_ok(vals_start, total) == 0 { t2d_error() }
    else { if t1d_slice_ok(out_start, d) == 0 { t2d_error() }
    else {
    // Step 1: scores[k] = dot(q, keys[k]).
    let scores = t1d_new(n);
    let inv_sqrt_d = 1.0_f32 / __sqrt((d as f32));
    let mut k: i32 = 0;
    while k < n {
        let mut dim: i32 = 0;
        let mut dot: f32 = 0.0_f32;
        while dim < d {
            let qv = __f32_from_bits(__arena_get(q_start + dim));
            let kv = __f32_from_bits(__arena_get(keys_start + k * d + dim));
            dot = dot + qv * kv;
            dim = dim + 1;
        }
        tf1d_set(scores, k, dot * inv_sqrt_d);
        k = k + 1;
    }
    // Step 2: softmax(scores) into probs.
    let probs = t1d_new(n);
    softmax_layer(scores, probs, n);
    // Step 3: out[d] = sum_k probs[k] * values[k][d].
    let mut dim2: i32 = 0;
    while dim2 < d {
        __arena_set(out_start + dim2, __bits_of_f32(0.0_f32));
        dim2 = dim2 + 1;
    }
    let mut k2: i32 = 0;
    while k2 < n {
        let p = __f32_from_bits(__arena_get(probs + k2));
        let mut d2: i32 = 0;
        while d2 < d {
            let cur = __f32_from_bits(__arena_get(out_start + d2));
            let v = __f32_from_bits(__arena_get(vals_start + k2 * d + d2));
            __arena_set(out_start + d2, __bits_of_f32(cur + p * v));
            d2 = d2 + 1;
        }
        k2 = k2 + 1;
    }
    0
    }}}}
    }}}
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
    if n <= 0 { t2d_error() }
    else { if d <= 0 { t2d_error() }
    else { if n > 2147483647 / d { t2d_error() }
    else {
    let total = n * d;
    if t1d_slice_ok(query_start, d) == 0 { t2d_error() }
    else { if t1d_slice_ok(keys_start, total) == 0 { t2d_error() }
    else { if t1d_slice_ok(values_start, total) == 0 { t2d_error() }
    else { if t1d_slice_ok(output_start, d) == 0 { t2d_error() }
    else {
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
    }}}}
    }}}
}
