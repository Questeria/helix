// helixc/examples/pickup_deliver.hx
//
// PICKUP-AND-DELIVER agent — Q-learning on an extended state space.
//
// 10x10 grid. Item at cell 22 (row 2, col 2). Delivery point at cell 77
// (row 7, col 7). Agent starts at cell 0 (top-left, no item). Goal: pick
// up the item, then bring it to the delivery point.
//
// State: encoded as `pos * 2 + has_item` so the same physical position
// maps to TWO Q-table rows depending on whether the agent is carrying
// the item. This forces the agent to learn TWO sub-policies:
//   - has_item == 0:  "go to the item"
//   - has_item == 1:  "go to delivery"
//
// Action: 0=up 1=down 2=left 3=right (4 actions).
//
// Reward:
//   +500 on the step that picks up the item (transitions has_item 0 -> 1)
//   +1000 on the step that delivers (has_item 1, position == 77)
//   -50 if action would walk off the grid (bump)
//   shaped: -dist_to_subgoal + 1 (potential-based) so gradient is dense
//
// JSON output (one object per line):
//   {"type":"init","grid_n":10,"item":22,"delivery":77,"seed":N}
//   {"type":"step","ep":N,"step":S,"pos":P,"has":H,"action":A,"reward":R}
//   {"type":"episode","ep":N,"steps":S,"reached":1,"total_reward":R}
//   {"type":"summary","episodes":N,"best_steps":S}
//
// License: Apache 2.0

@pure fn grid_n() -> i32 { 10 }
@pure fn grid_total() -> i32 { 100 }
@pure fn n_actions() -> i32 { 4 }
@pure fn n_states() -> i32 { 200 }   // pos (0..99) * 2 + has_item (0..1)
@pure fn item_pos() -> i32 { 22 }
@pure fn delivery_pos() -> i32 { 77 }
@pure fn n_episodes() -> i32 { 200 }
@pure fn max_steps_per_ep() -> i32 { 300 }
@pure fn epsilon_floor() -> i32 { 10 }
@pure fn discovery_budget() -> i32 { 50000 }
@pure fn alpha_pct() -> i32 { 30 }
@pure fn gamma_pct() -> i32 { 92 }
@pure fn map_seed() -> i32 { 12345 }

@pure fn lcg(seed: i32) -> i32 {
    let v = seed * 1103515245 + 12345;
    let m = (v % 2147483647 + 2147483647) % 2147483647;
    m
}

// HIGH bits to avoid the glibc-style LCG's low-bit bias. See fog_of_war.hx
// for the empirical demo of why this matters.
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

// Subgoal: where the agent should head NEXT given has_item.
@pure
fn subgoal(has_item: i32) -> i32 {
    if has_item == 0 { item_pos() } else { delivery_pos() }
}

// Encode (pos, has_item) -> state id.
@pure fn enc(pos: i32, has_item: i32) -> i32 { pos * 2 + has_item }

// Step a (pos, action) and return new pos. Bumped (off-grid) returns
// the original pos.
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

// Allocate Q-table. n_states * n_actions = 200*4 = 800 entries, all 0.
fn alloc_q() -> i32 {
    let start = __arena_len();
    let mut i: i32 = 0;
    while i < n_states() * n_actions() {
        __arena_push(0);
        i = i + 1;
    }
    start
}

@pure fn q_off(q: i32, state: i32, action: i32) -> i32 {
    q + state * n_actions() + action
}

@pure fn q_get(q: i32, state: i32, action: i32) -> i32 {
    __arena_get(q_off(q, state, action))
}

fn q_set(q: i32, state: i32, action: i32, v: i32) -> i32 {
    __arena_set(q_off(q, state, action), v); 0
}

// Argmax over q[state, *].
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

// Max q value over q[state, *].
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

fn pick_action_eps(q: i32, state: i32, epsilon_pct: i32, seed_cell: i32) -> i32 {
    let s = __arena_get(seed_cell);
    let s2 = lcg(s);
    __arena_set(seed_cell, s2);
    let r_pct = lcg_pct(s2);
    if r_pct < epsilon_pct {
        let s3 = lcg(s2);
        __arena_set(seed_cell, s3);
        lcg_action(s3)
    } else {
        argmax_q(q, state)
    }
}

// Q-update: Q(s, a) <- (1 - alpha) * Q(s, a) + alpha * (r + gamma * max_a' Q(s', a'))
// All values are scaled-by-100 i32 so arithmetic stays integer.
fn q_update(q: i32, state: i32, action: i32, reward: i32, next_state: i32, terminal: i32) -> i32 {
    let alpha = alpha_pct();
    let gamma = gamma_pct();
    let cur = q_get(q, state, action);
    let max_next = if terminal == 1 { 0 } else { max_q(q, next_state) };
    // target = reward + gamma * max_next  (both scaled by 100)
    let target = reward * 100 + (gamma * max_next) / 100;
    // new = cur + alpha * (target - cur) / 100
    let new_q = cur + (alpha * (target - cur)) / 100;
    q_set(q, state, action, new_q)
}

fn print_init() -> i32 {
    print_str("{\"type\":\"init\",\"grid_n\":10,\"item\":");
    print_int(item_pos());
    print_str(",\"delivery\":");
    print_int(delivery_pos());
    print_str(",\"seed\":");
    print_int(map_seed());
    print_str("}\n");
    0
}

fn print_step(ep: i32, step: i32, pos: i32, has_item: i32, action: i32, reward: i32) -> i32 {
    print_str("{\"type\":\"step\",\"ep\":");
    print_int(ep);
    print_str(",\"step\":");
    print_int(step);
    print_str(",\"pos\":");
    print_int(pos);
    print_str(",\"has\":");
    print_int(has_item);
    print_str(",\"action\":");
    print_int(action);
    print_str(",\"reward\":");
    print_int(reward);
    print_str("}\n");
    0
}

fn print_episode(ep: i32, steps: i32, reached: i32, total_reward: i32) -> i32 {
    print_str("{\"type\":\"episode\",\"ep\":");
    print_int(ep);
    print_str(",\"steps\":");
    print_int(steps);
    print_str(",\"reached\":");
    print_int(reached);
    print_str(",\"total_reward\":");
    print_int(total_reward);
    print_str("}\n");
    0
}

fn main() -> i32 {
    print_init();
    let q = alloc_q();
    let seed_cell = __arena_push(map_seed() * 7919 + 31);
    // Discovery phase: pure random walk, populate Q-values with reward
    // signal so the agent has gradient before structured training.
    let mut disc_pos: i32 = 0;
    let mut disc_has: i32 = 0;
    let mut disc_step: i32 = 0;
    let mut disc_found: i32 = 0;
    while disc_found == 0 {
        if disc_step >= discovery_budget() { disc_found = 1; }
        else {
            let s = __arena_get(seed_cell);
            let s2 = lcg(s);
            __arena_set(seed_cell, s2);
            let action = lcg_action(s2);
            let next_pos = step_pos(disc_pos, action);
            let bumped = if next_pos == disc_pos { 1 } else { 0 };
            // Compute next has_item: pickup transition.
            let next_has = if disc_has == 0 {
                if next_pos == item_pos() { 1 } else { 0 }
            } else { 1 };
            // Reward.
            let pickup_event = if disc_has == 0 {
                if next_has == 1 { 1 } else { 0 }
            } else { 0 };
            let delivery_event = if disc_has == 1 {
                if next_pos == delivery_pos() { 1 } else { 0 }
            } else { 0 };
            let dist_old = manhattan(disc_pos, subgoal(disc_has));
            let dist_new = manhattan(next_pos, subgoal(next_has));
            let shaped = (dist_old - dist_new) - 1;
            let reward = if delivery_event == 1 { 1000 }
                         else { if pickup_event == 1 { 500 }
                         else { if bumped == 1 { 0 - 50 } else { shaped } }};
            let cur_state = enc(disc_pos, disc_has);
            let next_state = enc(next_pos, next_has);
            let terminal = if delivery_event == 1 { 1 } else { 0 };
            q_update(q, cur_state, action, reward, next_state, terminal);
            disc_pos = next_pos;
            disc_has = next_has;
            disc_step = disc_step + 1;
            if delivery_event == 1 {
                // Reset to start to keep collecting more delivery
                // experiences.
                disc_pos = 0;
                disc_has = 0;
            }
        }
    }
    // Training phase.
    let mut ep: i32 = 0;
    let mut best_steps: i32 = 9999;
    while ep < n_episodes() {
        let raw = 80 - (ep * 60) / (n_episodes() - 1);
        let epsilon_pct = if raw < epsilon_floor() { epsilon_floor() } else { raw };
        let mut pos: i32 = 0;
        let mut has_item: i32 = 0;
        let mut step: i32 = 0;
        let mut total_reward: i32 = 0;
        let mut reached: i32 = 0;
        let mut keep: i32 = 1;
        while keep == 1 {
            if step >= max_steps_per_ep() { keep = 0; }
            else {
                let cur_state = enc(pos, has_item);
                let action = pick_action_eps(q, cur_state, epsilon_pct, seed_cell);
                let next_pos = step_pos(pos, action);
                let bumped = if next_pos == pos { 1 } else { 0 };
                let next_has = if has_item == 0 {
                    if next_pos == item_pos() { 1 } else { 0 }
                } else { 1 };
                let pickup_event = if has_item == 0 {
                    if next_has == 1 { 1 } else { 0 }
                } else { 0 };
                let delivery_event = if has_item == 1 {
                    if next_pos == delivery_pos() { 1 } else { 0 }
                } else { 0 };
                let dist_old = manhattan(pos, subgoal(has_item));
                let dist_new = manhattan(next_pos, subgoal(next_has));
                let shaped = (dist_old - dist_new) - 1;
                let reward = if delivery_event == 1 { 1000 }
                             else { if pickup_event == 1 { 500 }
                             else { if bumped == 1 { 0 - 50 } else { shaped } }};
                let next_state = enc(next_pos, next_has);
                let terminal = if delivery_event == 1 { 1 } else { 0 };
                q_update(q, cur_state, action, reward, next_state, terminal);
                total_reward = total_reward + reward;
                pos = next_pos;
                has_item = next_has;
                step = step + 1;
                print_step(ep, step, pos, has_item, action, reward);
                if delivery_event == 1 {
                    reached = 1;
                    keep = 0;
                }
            }
        }
        if reached == 1 {
            if step < best_steps { best_steps = step; }
        }
        print_episode(ep, step, reached, total_reward);
        ep = ep + 1;
    }
    print_str("{\"type\":\"summary\",\"episodes\":");
    print_int(n_episodes());
    print_str(",\"best_steps\":");
    print_int(best_steps);
    print_str("}\n");
    if best_steps < 9999 { 42 } else { 99 }
}
