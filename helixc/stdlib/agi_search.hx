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
