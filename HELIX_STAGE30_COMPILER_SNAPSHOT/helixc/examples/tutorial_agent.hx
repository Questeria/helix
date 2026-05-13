// helixc/examples/tutorial_agent.hx
//
// Tutorial AGI agent: a small grid-world solver that composes the Phase 4
// cognitive primitives end-to-end. NOT the Kovostov AI — that's the user's
// STOP point. This is a demonstration that primitives compose meaningfully.
//
// World: a 4x4 grid. The agent starts at cell 0 (top-left), goal at cell 15
// (bottom-right). Some cells are blocked. State id = row*4 + col (0..15).
// Actions: 0=up, 1=down, 2=left, 3=right.
//
// The agent uses:
//   - WORKING MEMORY  to track its current position
//   - EPISODIC MEMORY to log each action it tried
//   - WORLD MODEL     to predict next state from (state, action)
//   - PATTERN MATCH   (sequence_match) to detect loops in its action history
//   - SEARCH (BFS)    to find a path; reports path length
//
// Exit code = path length to goal in steps (Manhattan distance with no
// obstacles in this run = 6 steps from (0,0) to (3,3) via right-right-right
// then down-down-down).
//
// License: Apache 2.0

// Build a 4x4 grid world model (16 states, 4 actions).
// In open grid: action 0 (up) decreases row, 1 (down) increases row,
// 2 (left) decreases col, 3 (right) increases col. Block boundaries
// (going off the grid keeps you in place).
fn build_world_model() -> i32 {
    let wmt = wmt_new(16, 4);
    let mut s: i32 = 0;
    while s < 16 {
        let row = s / 4;
        let col = s % 4;
        // up
        let nu = if row > 0 { (row - 1) * 4 + col } else { s };
        wmt_set(wmt, s, 0, nu);
        // down
        let nd = if row < 3 { (row + 1) * 4 + col } else { s };
        wmt_set(wmt, s, 1, nd);
        // left
        let nl = if col > 0 { row * 4 + (col - 1) } else { s };
        wmt_set(wmt, s, 2, nl);
        // right
        let nr = if col < 3 { row * 4 + (col + 1) } else { s };
        wmt_set(wmt, s, 3, nr);
        s = s + 1;
    }
    wmt
}

// BFS from start to goal. Returns the shortest path length, or -1 if no path.
// Uses our bfs_queue + visited_set primitives.
fn bfs_path_length(wmt: i32, start: i32, goal: i32) -> i32 {
    let q = bfs_queue_new();
    let v = visited_new();
    // dist[state] = shortest steps from start; -1 if unreached.
    let dist = t1d_new(16);
    let mut i: i32 = 0;
    while i < 16 { ti1d_set(dist, i, 0 - 1); i = i + 1; }
    bfs_enqueue(q, start);
    visited_mark(v, start);
    ti1d_set(dist, start, 0);
    let mut found: i32 = 0 - 1;
    let mut keep: i32 = 1;
    while keep == 1 {
        let cur = bfs_dequeue(q);
        if cur < 0 { keep = 0; }
        else {
            if cur == goal {
                found = ti1d_get(dist, cur);
                keep = 0;
            } else {
                // Try each action.
                let mut a: i32 = 0;
                while a < 4 {
                    let nxt = wmt_predict(wmt, cur, a);
                    if nxt != cur {
                        if visited_has(v, nxt) == 0 {
                            visited_mark(v, nxt);
                            ti1d_set(dist, nxt, ti1d_get(dist, cur) + 1);
                            bfs_enqueue(q, nxt);
                        }
                    }
                    a = a + 1;
                }
            }
        }
    }
    found
}

// Simulate the agent for one step: read current state from WM,
// pick an action by hill-climbing toward the goal (using a simple
// |goal_row - cur_row| + |goal_col - cur_col| Manhattan score table),
// log it to episodic memory, advance via world model, write back.
fn agent_step(wm: i32, ep: i32, wmt: i32, mhd_table: i32, goal: i32) -> i32 {
    let cur = wm_load(wm, 0);   // key 0 = current state
    if cur == goal { 0 - 1 }
    else {
        // Generate the 4 candidate next-states; score by 1 / (1 + distance).
        // We'll pick the lowest-Manhattan-distance neighbor (highest score).
        let neighbors = t1d_new(4);
        let mut a: i32 = 0;
        while a < 4 {
            ti1d_set(neighbors, a, wmt_predict(wmt, cur, a));
            a = a + 1;
        }
        // Score table: scores[state] = 30 - manhattan(state, goal)
        // (higher = closer; pre-built by caller in mhd_table indexed by state)
        let best_state = hillclimb_step(neighbors, 4, mhd_table);
        // Find which action led to best_state (could be a loop if all four
        // neighbors equal cur, e.g., corners with blocked cells; we fall
        // through with a=0 in that case).
        let mut chosen_action: i32 = 0;
        let mut a2: i32 = 0;
        while a2 < 4 {
            if wmt_predict(wmt, cur, a2) == best_state {
                if chosen_action == 0 { chosen_action = a2; }
            }
            a2 = a2 + 1;
        }
        ep_record(ep, 1, chosen_action);     // log: kind=1 (action), payload=action
        wm_store(wm, 0, best_state);          // update WM
        chosen_action
    }
}

// A*-style path length using astar_priority + astar_path_set/get +
// astar_reconstruct. Same optimal answer as BFS on this uniform-cost
// world (every edge costs 1) but exercises a different primitive
// chain. The Manhattan heuristic admits the goal so A* explores
// fewer states than BFS would on a larger world.
fn astar_path_length(wmt: i32, start: i32, goal: i32) -> i32 {
    let g_table = t1d_new(16);   // g(state) = best-known cost from start
    let h_table = t1d_new(16);   // h(state) = Manhattan to goal
    let came_from = t1d_new(16);
    let mut i: i32 = 0;
    while i < 16 {
        ti1d_set(g_table, i, 999);
        let row = i / 4;
        let col = i % 4;
        let goal_row = goal / 4;
        let goal_col = goal % 4;
        let dr = if row > goal_row { row - goal_row } else { goal_row - row };
        let dc = if col > goal_col { col - goal_col } else { goal_col - col };
        ti1d_set(h_table, i, dr + dc);
        ti1d_set(came_from, i, 0 - 1);
        i = i + 1;
    }
    ti1d_set(g_table, start, 0);
    ti1d_set(came_from, start, start);   // start points to itself
    let pq = pq_new();
    pq_insert(pq, start, astar_priority(g_table, h_table, start));
    let v = visited_new();
    while pq_size(pq) > 0 {
        let cur = pq_pop_min(pq);
        if visited_has(v, cur) == 0 {
            visited_mark(v, cur);
            if cur == goal {
                // empty pq to break the loop
                while pq_size(pq) > 0 { pq_pop_min(pq); }
            } else {
                let mut a: i32 = 0;
                while a < 4 {
                    let nxt = wmt_predict(wmt, cur, a);
                    if nxt != cur {
                        let new_g = ti1d_get(g_table, cur) + 1;
                        if new_g < ti1d_get(g_table, nxt) {
                            ti1d_set(g_table, nxt, new_g);
                            astar_path_set(came_from, nxt, cur);
                            pq_insert(pq, nxt, astar_priority(g_table, h_table, nxt));
                        };
                    };
                    a = a + 1;
                }
            };
        };
    }
    // Reconstruct path: walk came_from back from goal. astar_reconstruct
    // returns NODE count (including start and goal); subtract 1 to match
    // BFS's EDGE count (number of steps).
    let path_buf = t1d_new(20);
    astar_reconstruct(came_from, goal, path_buf, 20) - 1
}

// PATTERN-MATCH detection: scan recent episodic actions for an A-B-A-B
// oscillation (which would mean the agent is stuck). Returns 1 if the
// last 4 actions form an oscillation, 0 otherwise.
//
// Uses sequence_match-style logic on the episodic memory's action log.
fn detect_oscillation(ep: i32) -> i32 {
    let cnt = ep_count(ep);
    if cnt < 4 { 0 }
    else {
        let a3 = ep_payload_at(ep, cnt - 1);
        let a2 = ep_payload_at(ep, cnt - 2);
        let a1 = ep_payload_at(ep, cnt - 3);
        let a0 = ep_payload_at(ep, cnt - 4);
        // Pattern A-B-A-B: a0 == a2 AND a1 == a3 AND a0 != a1.
        if a0 == a2 {
            if a1 == a3 {
                if a0 == a1 { 0 } else { 1 }
            } else { 0 }
        } else { 0 }
    }
}

// Build a Manhattan-distance score table from each state to goal.
// scores[state] = 30 - manhattan_distance(state, goal). Higher = closer.
fn build_mhd_score(goal: i32) -> i32 {
    let goal_row = goal / 4;
    let goal_col = goal % 4;
    let scores = t1d_new(16);
    let mut s: i32 = 0;
    while s < 16 {
        let row = s / 4;
        let col = s % 4;
        let dr = if row > goal_row { row - goal_row } else { goal_row - row };
        let dc = if col > goal_col { col - goal_col } else { goal_col - col };
        ti1d_set(scores, s, 30 - dr - dc);
        s = s + 1;
    }
    scores
}

// Run the full demo. Returns BFS-discovered path length, expected = 6.
fn main() -> i32 {
    let wmt = build_world_model();
    let goal: i32 = 15;
    // Verify BFS finds a path of length 6 from cell 0 to cell 15.
    let path_len = bfs_path_length(wmt, 0, goal);
    // A* should find the same optimal path length on this uniform-cost
    // world (Manhattan heuristic is admissible and consistent).
    let astar_len = astar_path_length(wmt, 0, goal);
    // Now do an agent rollout for 6 steps. Initially WM has cur=0, ep is empty.
    let wm = wm_new();
    wm_store(wm, 0, 0);
    let ep = ep_new();
    let scores = build_mhd_score(goal);
    let mut step: i32 = 0;
    while step < 6 {
        agent_step(wm, ep, wmt, scores, goal);
        step = step + 1;
    }
    // After 6 hill-climb steps the agent should be at the goal (cell 15).
    let final_state = wm_load(wm, 0);
    let actions_logged = ep_count(ep);
    // Pattern-match: a hill-climbing agent on an open grid shouldn't
    // oscillate, so detect_oscillation should return 0.
    let oscillating = detect_oscillation(ep);
    // Composite check: BFS-path-len + (astar_len==BFS-len ? 0 : 99) +
    // (final_state==goal ? 0 : 99) + (actions_logged==6 ? 0 : 99) +
    // (oscillating ? 99 : 0). Expected = 6.
    let astar_check = if astar_len == path_len { 0 } else { 99 };
    let final_check = if final_state == goal { 0 } else { 99 };
    let log_check = if actions_logged == 6 { 0 } else { 99 };
    let osc_check = if oscillating == 0 { 0 } else { 99 };
    path_len + astar_check + final_check + log_check + osc_check
}
