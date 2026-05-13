// helixc/examples/multi_goal.hx
//
// MULTI-GOAL agent — Q-learning on a 10x10 grid with THREE goal cells.
// The agent starts at cell 0 (top-left) and gets +1000 reward for
// reaching ANY of the three goals. Episode ends on first goal hit.
//
// Goals: cell 9 (top-right, 9 steps), cell 90 (bottom-left, 9 steps),
// cell 99 (bottom-right, 18 steps). The agent should learn to pick
// the closest goal — typically cell 9 or 90 since both are 9 steps,
// not the corner-diagonal 99.
//
// Reward shaping uses the MINIMUM Manhattan distance over all three
// goals: -delta(min_dist) per step. So the gradient always points
// toward whichever goal is currently closest, which can flip mid-
// trajectory if the agent veers.
//
// JSON output (one object per line):
//   {"type":"init","grid_n":10,"goals":[9,90,99],"seed":N}
//   {"type":"step","ep":N,"step":S,"pos":P,"action":A,"reward":R,"goal_picked":G}
//   {"type":"episode","ep":N,"steps":S,"reached":1,"goal_picked":G,"total_reward":R}
//   {"type":"summary","episodes":N,"best_steps":S,"goal_distribution":[c0,c1,c2]}
//
// License: Apache 2.0

@pure fn grid_n() -> i32 { 10 }
@pure fn grid_total() -> i32 { 100 }
@pure fn n_actions() -> i32 { 4 }
@pure fn n_goals() -> i32 { 3 }
@pure fn goal_0() -> i32 { 9 }       // top-right
@pure fn goal_1() -> i32 { 90 }      // bottom-left
@pure fn goal_2() -> i32 { 99 }      // bottom-right
@pure fn n_episodes() -> i32 { 100 }
@pure fn max_steps_per_ep() -> i32 { 200 }
@pure fn epsilon_floor() -> i32 { 10 }
@pure fn discovery_budget() -> i32 { 30000 }
@pure fn alpha_pct() -> i32 { 30 }
@pure fn gamma_pct() -> i32 { 92 }
@pure fn map_seed() -> i32 { 12345 }

@pure
fn lcg(seed: i32) -> i32 {
    let v = seed * 1103515245 + 12345;
    let m = (v % 2147483647 + 2147483647) % 2147483647;
    m
}

// HIGH bits to avoid the glibc-style LCG's low-bit bias (low bits cycle
// through only ~2-4 distinct values regardless of seed). See fog_of_war.hx
// for the empirical demo of why this matters. Even though shaped-reward
// demos like this one converge with the biased low-bit version, the
// random walks are sub-optimal — using high bits gives proper uniform
// distribution over [0, n) and exploration matches expectation.
@pure fn lcg_action(s: i32) -> i32 { ((s / 65536) % n_actions() + n_actions()) % n_actions() }
@pure fn lcg_pct(s: i32) -> i32 { ((s / 65536) % 100 + 100) % 100 }

@pure
fn manhattan(a: i32, b: i32) -> i32 {
    let n = grid_n();
    let ra = a / n; let ca = a % n;
    let rb = b / n; let cb = b % n;
    let dr = if ra < rb { rb - ra } else { ra - rb };
    let dc = if ca < cb { cb - ca } else { ca - cb };
    dr + dc
}

// Min Manhattan distance to ANY of the three goals.
@pure
fn min_dist_to_goal(pos: i32) -> i32 {
    let d0 = manhattan(pos, goal_0());
    let d1 = manhattan(pos, goal_1());
    let d2 = manhattan(pos, goal_2());
    let m01 = if d0 < d1 { d0 } else { d1 };
    if m01 < d2 { m01 } else { d2 }
}

// Identify which goal (0/1/2) the agent reached (or -1 if none).
@pure
fn goal_index(pos: i32) -> i32 {
    if pos == goal_0() { 0 }
    else { if pos == goal_1() { 1 }
    else { if pos == goal_2() { 2 }
    else { 0 - 1 }}}
}

@pure
fn step_pos(pos: i32, action: i32) -> i32 {
    let n = grid_n();
    let row = pos / n;
    let col = pos % n;
    if action == 0 {
        if row > 0 { (row - 1) * n + col } else { pos }
    } else { if action == 1 {
        if row < n - 1 { (row + 1) * n + col } else { pos }
    } else { if action == 2 {
        if col > 0 { row * n + (col - 1) } else { pos }
    } else {
        if col < n - 1 { row * n + (col + 1) } else { pos }
    }}}
}

fn alloc_q() -> i32 {
    let start = __arena_len();
    let mut i: i32 = 0;
    while i < grid_total() * n_actions() {
        __arena_push(0);
        i = i + 1;
    }
    start
}

@pure fn q_off(q: i32, state: i32, action: i32) -> i32 {
    q + state * n_actions() + action
}
@pure fn q_get(q: i32, state: i32, action: i32) -> i32 { __arena_get(q_off(q, state, action)) }
fn q_set(q: i32, state: i32, action: i32, v: i32) -> i32 { __arena_set(q_off(q, state, action), v); 0 }

@pure
fn argmax_q(q: i32, state: i32) -> i32 {
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

@pure
fn max_q(q: i32, state: i32) -> i32 {
    let mut best: i32 = q_get(q, state, 0);
    let mut a: i32 = 1;
    while a < n_actions() {
        let v = q_get(q, state, a);
        if v > best { best = v; }
        a = a + 1;
    }
    best
}

fn pick_action_eps(q: i32, state: i32, eps_pct: i32, seed_cell: i32) -> i32 {
    let s = __arena_get(seed_cell);
    let s2 = lcg(s);
    __arena_set(seed_cell, s2);
    let r_pct = lcg_pct(s2);
    if r_pct < eps_pct {
        let s3 = lcg(s2);
        __arena_set(seed_cell, s3);
        lcg_action(s3)
    } else {
        argmax_q(q, state)
    }
}

fn q_update(q: i32, state: i32, action: i32, reward: i32, next_state: i32, terminal: i32) -> i32 {
    let alpha = alpha_pct();
    let gamma = gamma_pct();
    let cur = q_get(q, state, action);
    let max_next = if terminal == 1 { 0 } else { max_q(q, next_state) };
    let target = reward * 100 + (gamma * max_next) / 100;
    let new_q = cur + (alpha * (target - cur)) / 100;
    q_set(q, state, action, new_q)
}

fn print_init() -> i32 {
    print_str("{\"type\":\"init\",\"grid_n\":10,\"goals\":[");
    print_int(goal_0()); print_str(",");
    print_int(goal_1()); print_str(",");
    print_int(goal_2()); print_str("],\"seed\":");
    print_int(map_seed());
    print_str("}\n");
    0
}

fn print_episode(ep: i32, steps: i32, reached: i32, goal_picked: i32, total_reward: i32) -> i32 {
    print_str("{\"type\":\"episode\",\"ep\":");
    print_int(ep);
    print_str(",\"steps\":");
    print_int(steps);
    print_str(",\"reached\":");
    print_int(reached);
    print_str(",\"goal_picked\":");
    print_int(goal_picked);
    print_str(",\"total_reward\":");
    print_int(total_reward);
    print_str("}\n");
    0
}

fn main() -> i32 {
    print_init();
    let q = alloc_q();
    let seed_cell = __arena_push(map_seed() * 7919 + 31);
    // Goal-distribution counters (one per goal).
    let dist_cell = __arena_push(0);
    __arena_push(0); __arena_push(0);
    // Discovery phase to populate Q with reward gradient.
    let mut disc_pos: i32 = 0;
    let mut disc_step: i32 = 0;
    let mut disc_done: i32 = 0;
    while disc_done == 0 {
        if disc_step >= discovery_budget() { disc_done = 1; }
        else {
            let s = __arena_get(seed_cell);
            let s2 = lcg(s);
            __arena_set(seed_cell, s2);
            let act = lcg_action(s2);
            let nxt = step_pos(disc_pos, act);
            let bumped = if nxt == disc_pos { 1 } else { 0 };
            let g_idx = goal_index(nxt);
            let reached = if g_idx >= 0 { 1 } else { 0 };
            let d_old = min_dist_to_goal(disc_pos);
            let d_new = min_dist_to_goal(nxt);
            let shaped = (d_old - d_new) - 1;
            let reward = if reached == 1 { 1000 }
                         else { if bumped == 1 { 0 - 50 } else { shaped } };
            q_update(q, disc_pos, act, reward, nxt, reached);
            disc_pos = nxt;
            disc_step = disc_step + 1;
            if reached == 1 { disc_pos = 0; }
        }
    }
    // Training phase.
    let mut ep: i32 = 0;
    let mut best_steps: i32 = 9999;
    while ep < n_episodes() {
        let raw = 80 - (ep * 60) / (n_episodes() - 1);
        let eps_pct = if raw < epsilon_floor() { epsilon_floor() } else { raw };
        let mut pos: i32 = 0;
        let mut step: i32 = 0;
        let mut total_reward: i32 = 0;
        let mut reached: i32 = 0;
        let mut goal_picked: i32 = 0 - 1;
        let mut keep: i32 = 1;
        while keep == 1 {
            if step >= max_steps_per_ep() { keep = 0; }
            else {
                let act = pick_action_eps(q, pos, eps_pct, seed_cell);
                let nxt = step_pos(pos, act);
                let bumped = if nxt == pos { 1 } else { 0 };
                let g_idx = goal_index(nxt);
                let hit = if g_idx >= 0 { 1 } else { 0 };
                let d_old = min_dist_to_goal(pos);
                let d_new = min_dist_to_goal(nxt);
                let shaped = (d_old - d_new) - 1;
                let reward = if hit == 1 { 1000 }
                             else { if bumped == 1 { 0 - 50 } else { shaped } };
                q_update(q, pos, act, reward, nxt, hit);
                total_reward = total_reward + reward;
                pos = nxt;
                step = step + 1;
                if hit == 1 {
                    reached = 1;
                    goal_picked = g_idx;
                    keep = 0;
                }
            }
        }
        if reached == 1 {
            if step < best_steps { best_steps = step; }
            // Increment goal distribution counter.
            let cur = __arena_get(dist_cell + goal_picked);
            __arena_set(dist_cell + goal_picked, cur + 1);
        }
        print_episode(ep, step, reached, goal_picked, total_reward);
        ep = ep + 1;
    }
    print_str("{\"type\":\"summary\",\"episodes\":");
    print_int(n_episodes());
    print_str(",\"best_steps\":");
    print_int(best_steps);
    print_str(",\"goal_distribution\":[");
    print_int(__arena_get(dist_cell)); print_str(",");
    print_int(__arena_get(dist_cell + 1)); print_str(",");
    print_int(__arena_get(dist_cell + 2));
    print_str("]}\n");
    if best_steps < 9999 { 42 } else { 99 }
}
