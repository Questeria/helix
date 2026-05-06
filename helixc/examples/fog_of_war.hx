// helixc/examples/fog_of_war.hx
//
// FOG-OF-WAR agent — Q-learning WITHOUT distance-based reward shaping.
// The agent doesn't know where the goal is until it stumbles on it.
//
// Curiosity-driven exploration: small intrinsic reward (+5) the FIRST
// time the agent visits each cell. After visiting, that cell yields
// 0 reward. The bonus drives exploration in the absence of any
// external gradient.
//
// External reward: only +1000 for reaching the goal at cell 77 (no
// shaping by distance).
//
// IMPORTANT: this demo uses HIGH bits of the LCG output for action
// selection because the standard glibc LCG (a=1103515245, c=12345)
// has known non-uniform low bits — `% 4` produces a biased cycle
// that makes pure random walk only visit ~1-2 cells. Existing demos
// (multi_goal, pickup_deliver) hide this because reward shaping +
// argmax provides directional pull. Fog-of-war exposes it: with no
// shaping, biased random walks fail to find the goal.
//
//   action = (s2 / 65536 % 4 + 4) % 4    [uses bits 16-17]
// vs
//   action = (s2 % 4 + 4) % 4             [BIASED — bits 0-1 cycle]
//
// JSON output (one object per line):
//   {"type":"init","grid_n":10,"goal":77,"seed":N,"mode":"fog"}
//   {"type":"episode","ep":N,"steps":S,"reached":1,"explored":E,
//                     "total_reward":R,"epsilon":E}
//   {"type":"summary","episodes":N,"best_steps":S,"reach_rate_pct":P}
//
// License: Apache 2.0

@pure fn grid_n() -> i32 { 10 }
@pure fn grid_total() -> i32 { 100 }
@pure fn n_actions() -> i32 { 4 }
@pure fn goal_pos() -> i32 { 77 }
@pure fn n_episodes() -> i32 { 200 }
@pure fn max_steps_per_ep() -> i32 { 300 }
@pure fn epsilon_floor() -> i32 { 15 }
@pure fn alpha_pct() -> i32 { 30 }
@pure fn gamma_pct() -> i32 { 92 }
@pure fn map_seed() -> i32 { 12345 }
@pure fn curiosity_bonus() -> i32 { 5 }

@pure
fn lcg(seed: i32) -> i32 {
    let v = seed * 1103515245 + 12345;
    let m = (v % 2147483647 + 2147483647) % 2147483647;
    m
}

// HIGH bits to avoid LCG low-bit bias. Bits 16-17 give a uniform [0..3].
@pure
fn lcg_action(s: i32) -> i32 { ((s / 65536) % 4 + 4) % 4 }

@pure
fn lcg_pct(s: i32) -> i32 { ((s / 65536) % 100 + 100) % 100 }

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

@pure fn q_off(q: i32, state: i32, action: i32) -> i32 { q + state * n_actions() + action }
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

fn alloc_visited() -> i32 {
    let start = __arena_len();
    let mut i: i32 = 0;
    while i < grid_total() {
        __arena_push(0);
        i = i + 1;
    }
    start
}

fn reset_visited(v: i32) -> i32 {
    let mut i: i32 = 0;
    while i < grid_total() {
        __arena_set(v + i, 0);
        i = i + 1;
    }
    0
}

@pure fn visited_get(v: i32, idx: i32) -> i32 { __arena_get(v + idx) }
fn visited_set(v: i32, idx: i32, val: i32) -> i32 { __arena_set(v + idx, val); 0 }

@pure
fn count_visited(v: i32) -> i32 {
    let mut total: i32 = 0;
    let mut i: i32 = 0;
    while i < grid_total() {
        if __arena_get(v + i) == 1 { total = total + 1; }
        i = i + 1;
    }
    total
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
    print_str("{\"type\":\"init\",\"grid_n\":10,\"goal\":");
    print_int(goal_pos());
    print_str(",\"seed\":");
    print_int(map_seed());
    print_str(",\"mode\":\"fog\"}\n");
    0
}

fn print_episode(ep: i32, steps: i32, reached: i32, explored: i32,
                 total_reward: i32, epsilon: i32) -> i32 {
    print_str("{\"type\":\"episode\",\"ep\":");
    print_int(ep);
    print_str(",\"steps\":");
    print_int(steps);
    print_str(",\"reached\":");
    print_int(reached);
    print_str(",\"explored\":");
    print_int(explored);
    print_str(",\"total_reward\":");
    print_int(total_reward);
    print_str(",\"epsilon\":");
    print_int(epsilon);
    print_str("}\n");
    0
}

fn main() -> i32 {
    print_init();
    let q = alloc_q();
    let visited = alloc_visited();
    let seed_cell = __arena_push(map_seed() * 7919 + 31);
    let mut ep: i32 = 0;
    let mut best_steps: i32 = 9999;
    let mut total_reached: i32 = 0;
    while ep < n_episodes() {
        let raw = 80 - (ep * 60) / (n_episodes() - 1);
        let eps_pct = if raw < epsilon_floor() { epsilon_floor() } else { raw };
        reset_visited(visited);
        let mut pos: i32 = 0;
        let mut step: i32 = 0;
        let mut total_reward: i32 = 0;
        let mut reached: i32 = 0;
        let mut keep: i32 = 1;
        while keep == 1 {
            if step >= max_steps_per_ep() { keep = 0; }
            else {
                let act = pick_action_eps(q, pos, eps_pct, seed_cell);
                let nxt = step_pos(pos, act);
                let bumped = if nxt == pos { 1 } else { 0 };
                let was_visited = visited_get(visited, nxt);
                let curiosity = if was_visited == 0 { curiosity_bonus() } else { 0 };
                visited_set(visited, nxt, 1);
                let goal_hit = if nxt == goal_pos() { 1 } else { 0 };
                let reward = if goal_hit == 1 { 1000 }
                             else { if bumped == 1 { 0 - 50 } else { curiosity }};
                q_update(q, pos, act, reward, nxt, goal_hit);
                total_reward = total_reward + reward;
                pos = nxt;
                step = step + 1;
                if goal_hit == 1 {
                    reached = 1;
                    keep = 0;
                }
            }
        }
        if reached == 1 {
            if step < best_steps { best_steps = step; }
            total_reached = total_reached + 1;
        }
        let explored = count_visited(visited);
        print_episode(ep, step, reached, explored, total_reward, eps_pct);
        ep = ep + 1;
    }
    print_str("{\"type\":\"summary\",\"episodes\":");
    print_int(n_episodes());
    print_str(",\"best_steps\":");
    print_int(best_steps);
    print_str(",\"reach_rate_pct\":");
    print_int((total_reached * 100) / n_episodes());
    print_str("}\n");
    if best_steps < 9999 { 42 } else { 99 }
}
