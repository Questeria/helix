// helixc/examples/dashboard_agent.hx
//
// Dashboard agent: 10x10 grid with 12 blocked cells, an agent that
// navigates from top-left to bottom-right via hill-climbing on
// Manhattan distance with obstacle penalties. Prints JSON per step
// for the live JavaScript dashboard to consume.
//
// JSON format (one line per step):
//   {"step":N,"pos":S,"action":A,"score":SC,"goal":35,"done":0}
//
// Final line:
//   {"step":N,"pos":S,"done":1,"reached_goal":1}
//
// Iterations are capped at 50; the dashboard plays back at user-
// controlled speed.
//
// License: Apache 2.0

@pure fn grid_n() -> i32 { 10 }
@pure fn grid_total() -> i32 { 100 }
@pure fn goal_id() -> i32 { 99 }   // bottom-right of 10x10
@pure fn max_iter() -> i32 { 50 }

// 12 obstacle cells. Tuned to create a maze-like detour.
@pure
fn is_obstacle(s: i32) -> i32 {
    if s == 12 { 1 } else {
    if s == 13 { 1 } else {
    if s == 14 { 1 } else {
    if s == 25 { 1 } else {
    if s == 35 { 1 } else {
    if s == 45 { 1 } else {
    if s == 47 { 1 } else {
    if s == 48 { 1 } else {
    if s == 56 { 1 } else {
    if s == 67 { 1 } else {
    if s == 78 { 1 } else {
    if s == 88 { 1 } else { 0 }}}}}}}}}}}}
}

// Build world model: 4 actions per state. Blocked-cell stepping reflects.
fn build_world() -> i32 {
    let n = grid_n();
    let wmt = wmt_new(n * n, 4);
    let mut s: i32 = 0;
    while s < n * n {
        let row = s / n;
        let col = s % n;
        let nu = if row > 0 { (row - 1) * n + col } else { s };
        let nd = if row < n - 1 { (row + 1) * n + col } else { s };
        let nl = if col > 0 { row * n + (col - 1) } else { s };
        let nr = if col < n - 1 { row * n + (col + 1) } else { s };
        let nu2 = if is_obstacle(nu) == 1 { s } else { nu };
        let nd2 = if is_obstacle(nd) == 1 { s } else { nd };
        let nl2 = if is_obstacle(nl) == 1 { s } else { nl };
        let nr2 = if is_obstacle(nr) == 1 { s } else { nr };
        wmt_set(wmt, s, 0, nu2);
        wmt_set(wmt, s, 1, nd2);
        wmt_set(wmt, s, 2, nl2);
        wmt_set(wmt, s, 3, nr2);
        s = s + 1;
    }
    wmt
}

// Score: higher = closer to goal (and not blocked).
fn build_scores() -> i32 {
    let n = grid_n();
    let goal = goal_id();
    let goal_row = goal / n;
    let goal_col = goal % n;
    let scores = t1d_new(n * n);
    let mut s: i32 = 0;
    while s < n * n {
        let row = s / n;
        let col = s % n;
        let dr = if row > goal_row { row - goal_row } else { goal_row - row };
        let dc = if col > goal_col { col - goal_col } else { goal_col - col };
        let base = 100 - dr - dc;
        let pen = if is_obstacle(s) == 1 { 0 - 1000 } else { 0 };
        ti1d_set(scores, s, base + pen);
        s = s + 1;
    }
    scores
}

// Pick best action via hillclimb. Returns chosen action.
fn pick_action(wmt: i32, scores: i32, cur: i32) -> i32 {
    let neighbors = t1d_new(4);
    let mut a: i32 = 0;
    while a < 4 {
        ti1d_set(neighbors, a, wmt_predict(wmt, cur, a));
        a = a + 1;
    }
    let best_state = hillclimb_step(neighbors, 4, scores);
    let mut chosen: i32 = 0;
    let mut found: i32 = 0;
    let mut a2: i32 = 0;
    while a2 < 4 {
        if found == 0 {
            if wmt_predict(wmt, cur, a2) == best_state {
                chosen = a2;
                found = 1;
            }
        }
        a2 = a2 + 1;
    }
    chosen
}

// Print one JSON line: {"step":N,"pos":S,"action":A,"score":SC}
fn print_step_json(step: i32, pos: i32, action: i32, score: i32) -> i32 {
    print_str("{\"step\":");
    print_int(step);
    print_str(",\"pos\":");
    print_int(pos);
    print_str(",\"action\":");
    print_int(action);
    print_str(",\"score\":");
    print_int(score);
    print_str(",\"done\":0}\n");
    0
}

// Print final marker JSON.
fn print_final_json(step: i32, pos: i32, reached: i32) -> i32 {
    print_str("{\"step\":");
    print_int(step);
    print_str(",\"pos\":");
    print_int(pos);
    print_str(",\"done\":1,\"reached_goal\":");
    print_int(reached);
    print_str("}\n");
    0
}

fn main() -> i32 {
    // Header: world description.
    print_str("{\"type\":\"init\",\"grid_n\":10,\"goal\":99,\"max_iter\":50,\"obstacles\":[12,13,14,25,35,45,47,48,56,67,78,88]}\n");
    let wmt = build_world();
    let scores = build_scores();
    let goal = goal_id();
    let mut pos: i32 = 0;
    // Print initial state.
    print_step_json(0, pos, 0 - 1, ti1d_get(scores, pos));
    let mut step: i32 = 1;
    let mut keep: i32 = 1;
    let mut reached: i32 = 0;
    let mut last_pos: i32 = pos;
    let mut stuck_count: i32 = 0;
    while keep == 1 {
        if pos == goal {
            reached = 1;
            keep = 0;
        } else { if step > max_iter() {
            keep = 0;
        }
        else {
            let action = pick_action(wmt, scores, pos);
            let next_pos = wmt_predict(wmt, pos, action);
            // Loop detection: if we're repeatedly returning to the same pos,
            // try a different action (random wandering would help; here we
            // just rotate through actions).
            if next_pos == last_pos {
                stuck_count = stuck_count + 1;
            } else {
                stuck_count = 0;
            }
            let final_action = if stuck_count > 1 { (action + 1) % 4 } else { action };
            let final_pos = wmt_predict(wmt, pos, final_action);
            last_pos = pos;
            pos = final_pos;
            print_step_json(step, pos, final_action, ti1d_get(scores, pos));
            step = step + 1;
        }};
    }
    print_final_json(step, pos, reached);
    if reached == 1 { 42 } else { 99 }
}
