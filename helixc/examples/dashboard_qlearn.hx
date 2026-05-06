// helixc/examples/dashboard_qlearn.hx
//
// Q-LEARNING agent that ACTUALLY IMPROVES across episodes, with map
// randomization driven by a seed. Obstacles are placed pseudo-randomly
// each run (deterministic for a given seed).
//
// All Q-values stored as i32 scaled-by-100 so arithmetic stays integer.
// LCG pseudo-random; potential-based reward shaping for dense gradient.
//
// JSON output (one object per line):
//   {"type":"init","grid_n":10,"goal":99,"n_episodes":N,"max_steps":S,
//    "obstacles":[c1,c2,...],"seed":SEED}
//   {"type":"step","ep":N,"step":S,"pos":P,"action":A,"reward":R,"qmax":Q,"done":0}
//   {"type":"episode","ep":N,"steps":S,"total_reward":R,"reached":1,"epsilon":E}
//   {"type":"qmap","ep":N,"qmax":[100 ints, one per cell]}
//   {"type":"summary","episodes":N,"best_steps":S}
//
// License: Apache 2.0

@pure fn grid_n() -> i32 { 10 }
@pure fn grid_total() -> i32 { 100 }
@pure fn goal_id() -> i32 { 99 }
@pure fn n_actions() -> i32 { 4 }
@pure fn n_episodes() -> i32 { 50 }
@pure fn max_steps_per_ep() -> i32 { 150 }
@pure fn n_obstacles() -> i32 { 14 }
// Minimum epsilon — never drops below this so agent always keeps some
// exploration. Prevents getting stuck in a suboptimal corner of policy.
@pure fn epsilon_floor() -> i32 { 15 }

// Q-values are scaled by 100 throughout (so 1.0 -> 100, 0.5 -> 50).
@pure fn alpha_pct() -> i32 { 30 }
@pure fn gamma_pct() -> i32 { 92 }

// LCG pseudo-random.
@pure
fn lcg(seed: i32) -> i32 {
    let v = seed * 1103515245 + 12345;
    let m = (v % 2147483647 + 2147483647) % 2147483647;
    m
}

// Manhattan distance from state to goal (10x10 grid, goal at 99).
@pure
fn dist_to_goal(s: i32) -> i32 {
    let row = s / 10;
    let col = s % 10;
    let dr = if row < 9 { 9 - row } else { row - 9 };
    let dc = if col < 9 { 9 - col } else { col - 9 };
    dr + dc
}

// SEED_PLACEHOLDER — replaced by the server before compile.
@pure fn map_seed() -> i32 { 12345 }

// MAZE_PLACEHOLDER — set to 1 by server when "maze layout" toggle is on.
// 0 = random-scatter obstacles (default), 1 = wall-line maze layout.
@pure fn use_maze() -> i32 { 0 }

// Build obstacle list. Two modes:
//   use_maze() = 0 -> 14 random scattered cells (default, balanced)
//   use_maze() = 1 -> 4 random "wall lines" (3-5 cells each, h or v),
//                     forming corridors that force real planning.
fn build_obstacles() -> i32 {
    if use_maze() == 1 { build_maze_walls() }
    else { build_scatter_obstacles() }
}

fn build_scatter_obstacles() -> i32 {
    let arr = t1d_new(n_obstacles());
    let mut placed: i32 = 0;
    let mut s: i32 = map_seed();
    let mut tries: i32 = 0;
    while placed < n_obstacles() {
        if tries > 1000 { placed = n_obstacles(); }
        else {
            s = lcg(s);
            let cand = (((s % 90) + 90) % 90) + 5;
            if cand < 3 { tries = tries + 1; }
            else { if cand > 96 { tries = tries + 1; }
            else {
                let mut dup: i32 = 0;
                let mut i: i32 = 0;
                while i < placed {
                    if ti1d_get(arr, i) == cand { dup = 1; }
                    i = i + 1;
                }
                if dup == 0 {
                    ti1d_set(arr, placed, cand);
                    placed = placed + 1;
                }
                tries = tries + 1;
            }};
        }
    }
    arr
}

// Maze layout: 4 wall lines, each 3-5 cells, horizontal or vertical.
// Skips start (0) and goal (99). Pads remaining slots with -1 sentinel.
fn build_maze_walls() -> i32 {
    let arr = t1d_new(n_obstacles());
    let mut placed: i32 = 0;
    let mut s: i32 = map_seed();
    let mut wall: i32 = 0;
    while wall < 4 {
        s = lcg(s);
        let row = (((s % 8) + 8) % 8) + 1;
        s = lcg(s);
        let col = (((s % 8) + 8) % 8) + 1;
        s = lcg(s);
        let horiz = if s % 2 == 0 { 1 } else { 0 };
        s = lcg(s);
        let len = (((s % 3) + 3) % 3) + 3;
        let mut k: i32 = 0;
        while k < len {
            if placed >= n_obstacles() { k = len; }
            else {
                let cell = if horiz == 1 {
                    let c2 = col + k;
                    if c2 > 9 { 0 - 1 } else { row * 10 + c2 }
                } else {
                    let r2 = row + k;
                    if r2 > 9 { 0 - 1 } else { r2 * 10 + col }
                };
                if cell < 0 { k = len; }
                else { if cell == 0 { k = k + 1; }
                else { if cell == 99 { k = k + 1; }
                else {
                    let mut dup: i32 = 0;
                    let mut i: i32 = 0;
                    while i < placed {
                        if ti1d_get(arr, i) == cell { dup = 1; }
                        i = i + 1;
                    }
                    if dup == 0 {
                        ti1d_set(arr, placed, cell);
                        placed = placed + 1;
                    }
                    k = k + 1;
                }}};
            }
        }
        wall = wall + 1;
    }
    while placed < n_obstacles() {
        ti1d_set(arr, placed, 0 - 1);
        placed = placed + 1;
    }
    arr
}

@pure
fn is_obstacle(obs_arr: i32, s: i32) -> i32 {
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    while i < n_obstacles() {
        if __arena_get(obs_arr + i) == s { found = 1; }
        i = i + 1;
    }
    found
}

fn build_world(obs_arr: i32) -> i32 {
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
        let nu2 = if is_obstacle(obs_arr, nu) == 1 { s } else { nu };
        let nd2 = if is_obstacle(obs_arr, nd) == 1 { s } else { nd };
        let nl2 = if is_obstacle(obs_arr, nl) == 1 { s } else { nl };
        let nr2 = if is_obstacle(obs_arr, nr) == 1 { s } else { nr };
        wmt_set(wmt, s, 0, nu2);
        wmt_set(wmt, s, 1, nd2);
        wmt_set(wmt, s, 2, nl2);
        wmt_set(wmt, s, 3, nr2);
        s = s + 1;
    }
    wmt
}

// Q-table on arena.
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

// argmax with tiebreak by current state's seed cell — pick a randomized
// action when multiple actions tie. Avoids the "always-action-0" rut.
fn q_argmax(q: i32, state: i32, seed_cell: i32) -> i32 {
    let mut best_a: i32 = 0;
    let mut best_v: i32 = q_get(q, state, 0);
    let mut a: i32 = 1;
    while a < n_actions() {
        let v = q_get(q, state, a);
        if v > best_v { best_v = v; best_a = a; }
        else { if v == best_v {
            // Tiebreak randomly.
            let s = __arena_get(seed_cell);
            let s2 = lcg(s);
            __arena_set(seed_cell, s2);
            if s2 % 2 == 0 { best_a = a; }
        }};
        a = a + 1;
    }
    best_a
}

fn pick_action_eps(q: i32, state: i32, epsilon_pct: i32, seed_cell: i32) -> i32 {
    let s = __arena_get(seed_cell);
    let s2 = lcg(s);
    __arena_set(seed_cell, s2);
    let r_pct = ((s2 % 100) + 100) % 100;
    if r_pct < epsilon_pct {
        let s3 = lcg(s2);
        __arena_set(seed_cell, s3);
        ((s3 % n_actions()) + n_actions()) % n_actions()
    } else {
        q_argmax(q, state, seed_cell)
    }
}

fn q_update(q: i32, s: i32, a: i32, reward: i32, s_next: i32) -> i32 {
    let old_q = q_get(q, s, a);
    let max_next = q_max(q, s_next);
    let target = reward + gamma_pct() * max_next / 100;
    let delta = alpha_pct() * (target - old_q) / 100;
    q_set(q, s, a, old_q + delta);
    0
}

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

// Print policy: argmax-action per cell, all 100 in one line.
// Note: q_argmax has random tiebreak via seed_cell; for the policy
// snapshot we want a DETERMINISTIC argmax. So we just pick lowest-index
// action at ties.
fn print_policy_map(ep: i32, q: i32) -> i32 {
    print_str("{\"type\":\"policy\",\"ep\":");
    print_int(ep);
    print_str(",\"actions\":[");
    let mut i: i32 = 0;
    while i < grid_total() {
        if i > 0 { print_str(","); }
        // Deterministic argmax (first action with max Q).
        let mut best_a: i32 = 0;
        let mut best_v: i32 = q_get(q, i, 0);
        let mut a: i32 = 1;
        while a < n_actions() {
            let v = q_get(q, i, a);
            if v > best_v { best_v = v; best_a = a; }
            a = a + 1;
        }
        print_int(best_a);
        i = i + 1;
    }
    print_str("]}\n");
    0
}

// Print the Q-value heatmap snapshot: max-Q per cell, all 100 in one line.
fn print_qmap(ep: i32, q: i32) -> i32 {
    print_str("{\"type\":\"qmap\",\"ep\":");
    print_int(ep);
    print_str(",\"qmax\":[");
    let mut i: i32 = 0;
    while i < grid_total() {
        if i > 0 { print_str(","); }
        print_int(q_max(q, i));
        i = i + 1;
    }
    print_str("]}\n");
    0
}

fn print_init(obs_arr: i32) -> i32 {
    print_str("{\"type\":\"init\",\"grid_n\":10,\"goal\":99,\"n_episodes\":");
    print_int(n_episodes());
    print_str(",\"max_steps\":");
    print_int(max_steps_per_ep());
    print_str(",\"seed\":");
    print_int(map_seed());
    print_str(",\"obstacles\":[");
    let mut i: i32 = 0;
    while i < n_obstacles() {
        if i > 0 { print_str(","); }
        print_int(__arena_get(obs_arr + i));
        i = i + 1;
    }
    print_str("]}\n");
    0
}

fn main() -> i32 {
    let obs_arr = build_obstacles();
    print_init(obs_arr);
    let wmt = build_world(obs_arr);
    let q = q_new();
    let goal = goal_id();
    let seed_cell = __arena_push(map_seed() * 7919 + 31);
    let mut ep: i32 = 0;
    let mut best_steps: i32 = 9999;
    let mut last_failed: i32 = 0;
    while ep < n_episodes() {
        // Decay 80% -> 15% across episodes; never drop below floor.
        let raw = 80 - (ep * 65) / (n_episodes() - 1);
        let base_eps = if raw < epsilon_floor() { epsilon_floor() } else { raw };
        // Boost epsilon if last episode didn't reach goal (escape ruts).
        let epsilon_pct = if last_failed == 1 {
            let bumped = base_eps + 20;
            if bumped > 90 { 90 } else { bumped }
        } else { base_eps };
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
            last_failed = 0;
        } else {
            last_failed = 1;
        }
        print_episode_end(ep, step, total_reward, reached, epsilon_pct);
        print_qmap(ep, q);
        // Also print policy: argmax-action per cell.
        print_policy_map(ep, q);
        ep = ep + 1;
    }
    print_str("{\"type\":\"summary\",\"episodes\":");
    print_int(n_episodes());
    print_str(",\"best_steps\":");
    print_int(best_steps);
    print_str("}\n");
    if best_steps < 9999 { 42 } else { 99 }
}
