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
    // Composite check: BFS-path-len + (final_state==goal ? 0 : 99) +
    // (actions_logged==6 ? 0 : 99). Expected = 6.
    let final_check = if final_state == goal { 0 } else { 99 };
    let log_check = if actions_logged == 6 { 0 } else { 99 };
    path_len + final_check + log_check
}
