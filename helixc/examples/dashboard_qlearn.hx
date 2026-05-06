// helixc/examples/dashboard_qlearn.hx
//
// Q-LEARNING agent that ACTUALLY IMPROVES across episodes.
//
// Same 10x10 grid + 12 obstacles as dashboard_agent.hx, but instead of
// deterministic hill-climbing, this agent uses tabular Q-learning:
//
//   Q(s, a) <- Q(s, a) + alpha [r + gamma * max_a' Q(s', a') - Q(s, a)]
//
// Action selection is eps-greedy: with probability eps pick random, else
// pick argmax_a Q(s, a). eps decays across episodes (more exploration
// early, more exploitation late).
//
// Reward shaping:
//   +1000 for reaching goal
//   -100  for stepping into an obstacle / wall (didn't move)
//   -1    per step (penalty for length)
//
// All Q-values stored as i32 scaled-by-100 so arithmetic stays integer.
//
// Output: same JSON-per-line format as dashboard_agent.hx, plus per-
// episode summary lines:
//   {"type":"episode","episode":N,"steps":S,"reached":1,"epsilon":E}
// and a final
//   {"type":"summary","episodes":N,"best_steps":S}
//
// License: Apache 2.0

@pure fn grid_n() -> i32 { 10 }
@pure fn grid_total() -> i32 { 100 }
@pure fn goal_id() -> i32 { 99 }
@pure fn n_actions() -> i32 { 4 }
@pure fn n_episodes() -> i32 { 20 }
@pure fn max_steps_per_ep() -> i32 { 80 }

// Manhattan distance from state to goal (10x10 grid, goal at 99).
@pure
fn dist_to_goal(s: i32) -> i32 {
    let row = s / 10;
    let col = s % 10;
    let dr = if row < 9 { 9 - row } else { row - 9 };
    let dc = if col < 9 { 9 - col } else { col - 9 };
    dr + dc
}

// Q-values are scaled by 100 throughout (so 1.0 -> 100, 0.5 -> 50).
@pure fn alpha_pct() -> i32 { 25 }   // alpha = 0.25
@pure fn gamma_pct() -> i32 { 90 }    // gamma = 0.90

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

// Q-table: 100 states x 4 actions = 400 entries on the arena.
fn q_new() -> i32 {
    let start = __arena_len();
    let total = grid_total() * n_actions();
    let mut i: i32 = 0;
    while i < total {
        __arena_push(0);
        i = i + 1;
    }
    start
}

@pure fn q_get(q: i32, state: i32, action: i32) -> i32 {
    __arena_get(q + state * n_actions() + action)
}

fn q_set(q: i32, state: i32, action: i32, value: i32) -> i32 {
    __arena_set(q + state * n_actions() + action, value);
    0
}

@pure
fn q_max(q: i32, state: i32) -> i32 {
    let mut best: i32 = q_get(q, state, 0);
    let mut a: i32 = 1;
    while a < n_actions() {
        let v = q_get(q, state, a);
        if v > best { best = v; }
        a = a + 1;
    }
    best
}

@pure
fn q_argmax(q: i32, state: i32) -> i32 {
    let mut best_a: i32 = 0;
    let mut best_v: i32 = q_get(q, state, 0);
    let mut a: i32 = 1;
    while a < n_actions() {
        let v = q_get(q, state, a);
        if v > best_v { best_v = v; best_a = a; }
        a = a + 1;
    }
    best_a
}

// Simple LCG for pseudo-randomness. Returns next seed.
@pure
fn lcg(seed: i32) -> i32 {
    // (seed * 1103515245 + 12345) mod 2^31, kept positive.
    let v = seed * 1103515245 + 12345;
    let m = (v % 2147483647 + 2147483647) % 2147483647;
    m
}

// eps-greedy action selection. epsilon_pct in [0..100].
// Returns (chosen_action, new_seed) packed: chosen * 1_000_000_000 + new_seed
// - but Phase-0 Helix lacks tuples cleanly. We instead pass seed by-cell
// using a single-slot scratch arena entry.
fn pick_action_eps(q: i32, state: i32, epsilon_pct: i32, seed_cell: i32) -> i32 {
    let s = __arena_get(seed_cell);
    let s2 = lcg(s);
    __arena_set(seed_cell, s2);
    let r_pct = ((s2 % 100) + 100) % 100;
    if r_pct < epsilon_pct {
        // Random action.
        let s3 = lcg(s2);
        __arena_set(seed_cell, s3);
        ((s3 % n_actions()) + n_actions()) % n_actions()
    } else {
        q_argmax(q, state)
    }
}

// Q-update step.
//   target = reward + gamma * max_a' Q(s', a')
//   Q(s, a) += alpha * (target - Q(s, a))
// All values scaled by 100; reward is already scaled.
fn q_update(q: i32, s: i32, a: i32, reward: i32, s_next: i32) -> i32 {
    let old_q = q_get(q, s, a);
    let max_next = q_max(q, s_next);
    // target = reward + gamma * max_next  (gamma = gamma_pct / 100)
    let target = reward + gamma_pct() * max_next / 100;
    // delta = alpha * (target - old_q)  (alpha = alpha_pct / 100)
    let delta = alpha_pct() * (target - old_q) / 100;
    q_set(q, s, a, old_q + delta);
    0
}

// Print step JSON.
fn print_step(ep: i32, step: i32, pos: i32, action: i32, reward: i32, qmax: i32) -> i32 {
    print_str("{\"type\":\"step\",\"ep\":");
    print_int(ep);
    print_str(",\"step\":");
    print_int(step);
    print_str(",\"pos\":");
    print_int(pos);
    print_str(",\"action\":");
    print_int(action);
    print_str(",\"reward\":");
    print_int(reward);
    print_str(",\"qmax\":");
    print_int(qmax);
    print_str(",\"done\":0}\n");
    0
}

fn print_episode_end(ep: i32, steps: i32, total_reward: i32, reached: i32, epsilon: i32) -> i32 {
    print_str("{\"type\":\"episode\",\"ep\":");
    print_int(ep);
    print_str(",\"steps\":");
    print_int(steps);
    print_str(",\"total_reward\":");
    print_int(total_reward);
    print_str(",\"reached\":");
    print_int(reached);
    print_str(",\"epsilon\":");
    print_int(epsilon);
    print_str("}\n");
    0
}

fn main() -> i32 {
    print_str("{\"type\":\"init\",\"grid_n\":10,\"goal\":99,\"n_episodes\":10,\"max_steps\":50,\"obstacles\":[12,13,14,25,35,45,47,48,56,67,78,88]}\n");
    let wmt = build_world();
    let q = q_new();
    let goal = goal_id();
    // Seed cell for LCG.
    let seed_cell = __arena_push(98765);
    let mut ep: i32 = 0;
    let mut best_steps: i32 = 999;
    while ep < n_episodes() {
        // eps decays linearly from 80% to 5% across episodes.
        let epsilon_pct = 80 - (ep * 75) / (n_episodes() - 1);
        let mut pos: i32 = 0;
        let mut step: i32 = 0;
        let mut total_reward: i32 = 0;
        let mut reached: i32 = 0;
        let mut keep: i32 = 1;
        while keep == 1 {
            if pos == goal {
                reached = 1;
                keep = 0;
            } else { if step >= max_steps_per_ep() {
                keep = 0;
            } else {
                let action = pick_action_eps(q, pos, epsilon_pct, seed_cell);
                let next_pos = wmt_predict(wmt, pos, action);
                let bumped = if next_pos == pos { 1 } else { 0 };
                // Potential-based reward: difference in Manhattan-to-goal.
                // Moving closer: +10. Same distance: -1. Moving away: -11.
                // Plus +1000 for actually reaching goal, -50 for bumping wall.
                let d_old = dist_to_goal(pos);
                let d_new = dist_to_goal(next_pos);
                let shaped = (d_old - d_new) * 10 - 1;
                let reward = if next_pos == goal { 1000 }
                             else { if bumped == 1 { 0 - 50 } else { shaped } };
                q_update(q, pos, action, reward, next_pos);
                total_reward = total_reward + reward;
                pos = next_pos;
                step = step + 1;
                print_step(ep, step, pos, action, reward, q_max(q, pos));
            }};
        }
        if reached == 1 {
            if step < best_steps { best_steps = step; }
        }
        print_episode_end(ep, step, total_reward, reached, epsilon_pct);
        ep = ep + 1;
    }
    print_str("{\"type\":\"summary\",\"episodes\":");
    print_int(n_episodes());
    print_str(",\"best_steps\":");
    print_int(best_steps);
    print_str("}\n");
    if best_steps < 999 { 42 } else { 99 }
}
