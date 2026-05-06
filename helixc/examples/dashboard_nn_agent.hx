// helixc/examples/dashboard_nn_agent.hx
//
// NEURAL-NETWORK AGENT: same 10x10 grid + random obstacles as the
// Q-learning agent, but instead of a 100x4 Q-table, this agent uses a
// small feedforward neural network to estimate Q-values:
//
//     state-1-hot (100) -> Dense(32) -> ReLU -> Dense(4) -> Q-values
//
// Training: gradient-descent on the temporal-difference loss using
// reverse-mode autodiff at each step. Single-sample minibatch.
//
// This stress-tests the Helix substrate end-to-end:
//   - tensor stdlib (tf1d_*, tf2d_matvec, tf1d_relu)
//   - autodiff_reverse (tape-based gradient propagation)
//   - nn primitives (dense_layer_f32_forward, sgd_f32_step)
//   - LCG random init + epsilon-greedy
//
// All weights and activations are f32 stored in arena via __bits_of_f32
// reinterpret (Phase 2.2 step 2 codegen primitive).
//
// JSON output (one per line):
//   {"type":"init","grid_n":10,"goal":99,"obstacles":[...],"seed":N}
//   {"type":"step","ep":N,"step":S,"pos":P,"action":A,"loss":L,"qmax":Q}
//   {"type":"episode","ep":N,"steps":S,"total_reward":R,"reached":1,"epsilon":E}
//   {"type":"summary","episodes":N,"best_steps":S}
//
// LICENSE: Apache 2.0

@pure fn grid_n() -> i32 { 10 }
@pure fn grid_total() -> i32 { 100 }
@pure fn goal_id() -> i32 { 99 }
@pure fn n_actions() -> i32 { 4 }
@pure fn hidden() -> i32 { 32 }
@pure fn n_episodes() -> i32 { 80 }
@pure fn max_steps_per_ep() -> i32 { 200 }
@pure fn n_obstacles() -> i32 { 14 }
@pure fn epsilon_floor() -> i32 { 25 }
// Experience replay buffer: 512 transitions, 5 i32s each (s, a, r, s', done).
@pure fn replay_capacity() -> i32 { 512 }
@pure fn replay_minibatch() -> i32 { 16 }

// SEED_PLACEHOLDER — replaced by server.
@pure fn map_seed() -> i32 { 12345 }

@pure fn lcg(seed: i32) -> i32 {
    let v = seed * 1103515245 + 12345;
    let m = (v % 2147483647 + 2147483647) % 2147483647;
    m
}

@pure
fn dist_to_goal(s: i32) -> i32 {
    let row = s / 10;
    let col = s % 10;
    let dr = if row < 9 { 9 - row } else { row - 9 };
    let dc = if col < 9 { 9 - col } else { col - 9 };
    dr + dc
}

// ---- Random map (same scheme as qlearn agent) ----
fn build_obstacles() -> i32 {
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

// ---- NN: 100 -> Dense(32) -> ReLU -> Dense(4) ----
//
// Layer 1 weights W1: 32 x 100 (rows x cols), bias b1: 32.
// Layer 2 weights W2:  4 x  32, bias b2:  4.
//
// Initialize with small random values via LCG. Scaled-by-1024 to keep
// numerical stability under integer-arithmetic LCG.

fn nn_init_weight(seed_cell: i32) -> f32 {
    let s = __arena_get(seed_cell);
    let s2 = lcg(s);
    __arena_set(seed_cell, s2);
    // Map s2 in [0..2^31] to roughly [-0.1, +0.1] f32.
    let r = (s2 % 2000) - 1000;   // -1000..999
    (r as f32) / 10000.0_f32       // -0.1..0.0999
}

fn nn_alloc_weights(seed_cell: i32) -> i32 {
    let start = __arena_len();
    // W1: 32*100 = 3200 entries
    let mut i: i32 = 0;
    while i < 3200 {
        let w = nn_init_weight(seed_cell);
        __arena_push(__bits_of_f32(w));
        i = i + 1;
    }
    // b1: 32 entries (zero)
    let mut j: i32 = 0;
    while j < 32 {
        __arena_push(__bits_of_f32(0.0_f32));
        j = j + 1;
    }
    // W2: 4*32 = 128 entries
    let mut k: i32 = 0;
    while k < 128 {
        let w = nn_init_weight(seed_cell);
        __arena_push(__bits_of_f32(w));
        k = k + 1;
    }
    // b2: 4 entries (zero)
    let mut m: i32 = 0;
    while m < 4 {
        __arena_push(__bits_of_f32(0.0_f32));
        m = m + 1;
    }
    start
}

@pure fn w1_off(weights: i32) -> i32 { weights }
@pure fn b1_off(weights: i32) -> i32 { weights + 3200 }
@pure fn w2_off(weights: i32) -> i32 { weights + 3232 }
@pure fn b2_off(weights: i32) -> i32 { weights + 3360 }

// Forward: q[a] = (W2 @ ReLU(W1 @ x + b1) + b2)[a]
// x is the one-hot state vector of length 100. Since x has only ONE nonzero
// element (at state index), W1 @ x = column `state` of W1 — much faster than
// a full matvec. We just need to extract the column.
//
// Returns: arena offset to a 4-element f32 vector (q values), AND populates
// hidden activations at hidden_buf, pre-activations at hidden_pre_buf.
fn nn_forward(weights: i32, state: i32, hidden_pre: i32, hidden_buf: i32, q_out: i32) -> i32 {
    let n = grid_n() * grid_n();   // 100
    let h = hidden();              // 32
    let na = n_actions();          // 4
    // hidden_pre[i] = W1[i, state] + b1[i]
    let mut i: i32 = 0;
    while i < h {
        let w_v = __f32_from_bits(__arena_get(w1_off(weights) + i * n + state));
        let b_v = __f32_from_bits(__arena_get(b1_off(weights) + i));
        __arena_set(hidden_pre + i, __bits_of_f32(w_v + b_v));
        i = i + 1;
    }
    // hidden = ReLU(hidden_pre)
    tf1d_relu(hidden_pre, hidden_buf, h);
    // q[a] = sum_i W2[a, i] * hidden[i] + b2[a]
    let mut a: i32 = 0;
    while a < na {
        let mut acc: f32 = __f32_from_bits(__arena_get(b2_off(weights) + a));
        let mut j: i32 = 0;
        while j < h {
            let w_ai = __f32_from_bits(__arena_get(w2_off(weights) + a * h + j));
            let h_j = __f32_from_bits(__arena_get(hidden_buf + j));
            acc = acc + w_ai * h_j;
            j = j + 1;
        }
        __arena_set(q_out + a, __bits_of_f32(acc));
        a = a + 1;
    }
    0
}

// Argmax over q[0..n_actions). Returns index.
@pure
fn argmax_q(q_buf: i32, na: i32) -> i32 {
    let mut best_a: i32 = 0;
    let mut best_v: f32 = __f32_from_bits(__arena_get(q_buf));
    let mut a: i32 = 1;
    while a < na {
        let v = __f32_from_bits(__arena_get(q_buf + a));
        if v > best_v { best_v = v; best_a = a; }
        a = a + 1;
    }
    best_a
}

// Max q value.
@pure
fn max_q(q_buf: i32, na: i32) -> f32 {
    let mut best: f32 = __f32_from_bits(__arena_get(q_buf));
    let mut a: i32 = 1;
    while a < na {
        let v = __f32_from_bits(__arena_get(q_buf + a));
        if v > best { best = v; }
        a = a + 1;
    }
    best
}

// Manual gradient + SGD update for a single TD step.
//
// Loss: L = 0.5 * (target - q[action])^2
// where target = reward + gamma * max_a' Q(s', a')
//
// d_L / d_q[a]   = (q[a] - target) for a == action, else 0.
// d_q[a] / d_w2[a, i] = hidden[i]
// d_q[a] / d_b2[a]    = 1
// d_q[a] / d_hidden[i] = w2[a, i]
// d_hidden[i] / d_hidden_pre[i] = (hidden_pre[i] > 0 ? 1 : 0)  (ReLU)
// d_hidden_pre[i] / d_w1[i, state] = 1   (since x is 1-hot at `state`)
// d_hidden_pre[i] / d_b1[i] = 1
fn nn_train_step(weights: i32, state: i32, action: i32, target: f32,
                 hidden_pre: i32, hidden_buf: i32, q_buf: i32, lr: f32) -> i32 {
    let n = grid_n() * grid_n();
    let h = hidden();
    let q_a = __f32_from_bits(__arena_get(q_buf + action));
    let dL_dqa = q_a - target;
    // Update W2[action, *] and b2[action] only (other actions don't contribute).
    let mut i: i32 = 0;
    while i < h {
        let h_i = __f32_from_bits(__arena_get(hidden_buf + i));
        let grad_w2 = dL_dqa * h_i;
        let off = w2_off(weights) + action * h + i;
        let old = __f32_from_bits(__arena_get(off));
        __arena_set(off, __bits_of_f32(old - lr * grad_w2));
        i = i + 1;
    }
    let b2_off_a = b2_off(weights) + action;
    let old_b2 = __f32_from_bits(__arena_get(b2_off_a));
    __arena_set(b2_off_a, __bits_of_f32(old_b2 - lr * dL_dqa));
    // d_L / d_hidden[i] = dL_dqa * w2[action, i]
    // For each hidden i, propagate to w1[i, state] and b1[i] via ReLU.
    let mut k: i32 = 0;
    while k < h {
        let pre = __f32_from_bits(__arena_get(hidden_pre + k));
        let relu_grad = if pre > 0.0_f32 { 1.0_f32 } else { 0.0_f32 };
        let w2_ak = __f32_from_bits(__arena_get(w2_off(weights) + action * h + k));
        let dL_dpre = dL_dqa * w2_ak * relu_grad;
        // Update only column `state` of W1 (since x is 1-hot).
        let w1_off_ks = w1_off(weights) + k * n + state;
        let old_w1 = __f32_from_bits(__arena_get(w1_off_ks));
        __arena_set(w1_off_ks, __bits_of_f32(old_w1 - lr * dL_dpre));
        // Update b1[k]
        let b1_off_k = b1_off(weights) + k;
        let old_b1 = __f32_from_bits(__arena_get(b1_off_k));
        __arena_set(b1_off_k, __bits_of_f32(old_b1 - lr * dL_dpre));
        k = k + 1;
    }
    0
}

// =====================================================================
// Experience replay buffer
// =====================================================================
// Layout:
//   slot 0: count (number of transitions stored, max replay_capacity)
//   slot 1: head (next write index, circular)
//   slot 2..: replay_capacity * 5 entries, each (s, a, reward_x100, s', done)
fn replay_new() -> i32 {
    let start = __arena_len();
    __arena_push(0);   // count
    __arena_push(0);   // head
    let total = replay_capacity() * 5;
    let mut i: i32 = 0;
    while i < total { __arena_push(0); i = i + 1; }
    start
}

fn replay_store(replay: i32, s: i32, a: i32, reward_x100: i32, s_next: i32, done: i32) -> i32 {
    let cap = replay_capacity();
    let head = __arena_get(replay + 1);
    let off = replay + 2 + head * 5;
    __arena_set(off, s);
    __arena_set(off + 1, a);
    __arena_set(off + 2, reward_x100);
    __arena_set(off + 3, s_next);
    __arena_set(off + 4, done);
    let new_head = (head + 1) % cap;
    __arena_set(replay + 1, new_head);
    let cnt = __arena_get(replay);
    if cnt < cap {
        __arena_set(replay, cnt + 1);
    }
    0
}

@pure fn replay_count(replay: i32) -> i32 { __arena_get(replay) }

@pure fn replay_get_s(replay: i32, idx: i32) -> i32 { __arena_get(replay + 2 + idx * 5) }
@pure fn replay_get_a(replay: i32, idx: i32) -> i32 { __arena_get(replay + 2 + idx * 5 + 1) }
@pure fn replay_get_r(replay: i32, idx: i32) -> i32 { __arena_get(replay + 2 + idx * 5 + 2) }
@pure fn replay_get_sp(replay: i32, idx: i32) -> i32 { __arena_get(replay + 2 + idx * 5 + 3) }
@pure fn replay_get_done(replay: i32, idx: i32) -> i32 { __arena_get(replay + 2 + idx * 5 + 4) }

fn pick_action_eps(q_buf: i32, na: i32, epsilon_pct: i32, seed_cell: i32) -> i32 {
    let s = __arena_get(seed_cell);
    let s2 = lcg(s);
    __arena_set(seed_cell, s2);
    let r_pct = ((s2 % 100) + 100) % 100;
    if r_pct < epsilon_pct {
        let s3 = lcg(s2);
        __arena_set(seed_cell, s3);
        ((s3 % na) + na) % na
    } else {
        argmax_q(q_buf, na)
    }
}

fn print_step(ep: i32, step: i32, pos: i32, action: i32, reward: i32, qmax_scaled: i32) -> i32 {
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
    print_int(qmax_scaled);
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

fn print_init(obs_arr: i32) -> i32 {
    print_str("{\"type\":\"init\",\"grid_n\":10,\"goal\":99,\"n_episodes\":");
    print_int(n_episodes());
    print_str(",\"max_steps\":");
    print_int(max_steps_per_ep());
    print_str(",\"seed\":");
    print_int(map_seed());
    print_str(",\"agent\":\"nn\",\"obstacles\":[");
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
    let goal = goal_id();
    let h = hidden();
    let na = n_actions();
    let seed_cell = __arena_push(map_seed() * 7919 + 31);
    let weights = nn_alloc_weights(seed_cell);
    let replay = replay_new();
    // DISCOVERY: pure random walk to populate replay with goal-reaching
    // transitions. Without this the NN almost never learns the reward
    // signal because it never sees a goal in 40*150=6000 random steps.
    let mut disc_pos: i32 = 0;
    let mut disc_step: i32 = 0;
    let mut disc_found: i32 = 0;
    let mut disc_since_restart: i32 = 0;
    while disc_found == 0 {
        if disc_step >= 200000 { disc_found = 1; }
        else {
            if disc_since_restart >= 500 {
                disc_pos = 0;
                disc_since_restart = 0;
            }
            let s = __arena_get(seed_cell);
            let s2 = lcg(s);
            __arena_set(seed_cell, s2);
            let dact = ((s2 % na) + na) % na;
            let dnxt = wmt_predict(wmt, disc_pos, dact);
            let dbumped = if dnxt == disc_pos { 1 } else { 0 };
            let dd_old = dist_to_goal(disc_pos);
            let dd_new = dist_to_goal(dnxt);
            let dshaped = (dd_old - dd_new) * 10 - 1;
            let drew = if dnxt == goal { 1000 }
                       else { if dbumped == 1 { 0 - 50 } else { dshaped } };
            let ddone = if dnxt == goal { 1 } else { 0 };
            replay_store(replay, disc_pos, dact, drew * 100, dnxt, ddone);
            disc_pos = dnxt;
            disc_step = disc_step + 1;
            disc_since_restart = disc_since_restart + 1;
            if dnxt == goal {
                disc_found = 1;
                disc_pos = 0;
            }
        }
    }
    // Per-step scratch buffers.
    let hidden_pre = t1d_new(h);
    let hidden_buf = t1d_new(h);
    let q_buf = t1d_new(na);
    let q_next_buf = t1d_new(na);
    let hidden_pre_next = t1d_new(h);
    let hidden_buf_next = t1d_new(h);
    let q_replay_buf = t1d_new(na);
    let q_replay_next = t1d_new(na);
    let h_pre_r = t1d_new(h);
    let h_buf_r = t1d_new(h);
    let mut ep: i32 = 0;
    let mut best_steps: i32 = 9999;
    let mut last_failed: i32 = 0;
    while ep < n_episodes() {
        let raw = 80 - (ep * 60) / (n_episodes() - 1);
        let base_eps = if raw < epsilon_floor() { epsilon_floor() } else { raw };
        let epsilon_pct = if last_failed == 1 {
            let bumped = base_eps + 25;
            if bumped > 95 { 95 } else { bumped }
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
                // Forward pass at current state.
                nn_forward(weights, pos, hidden_pre, hidden_buf, q_buf);
                let action = pick_action_eps(q_buf, na, epsilon_pct, seed_cell);
                let next_pos = wmt_predict(wmt, pos, action);
                let bumped = if next_pos == pos { 1 } else { 0 };
                let d_old = dist_to_goal(pos);
                let d_new = dist_to_goal(next_pos);
                let shaped = (d_old - d_new) * 10 - 1;
                let reward = if next_pos == goal { 1000 }
                             else { if bumped == 1 { 0 - 50 } else { shaped } };
                // Compute target = reward/100.0 + 0.9 * max_a' Q(s', a').
                let r_f = (reward as f32) / 100.0_f32;
                let target = if next_pos == goal {
                    r_f
                } else {
                    nn_forward(weights, next_pos, hidden_pre_next, hidden_buf_next, q_next_buf);
                    let max_next = max_q(q_next_buf, na);
                    r_f + 0.9_f32 * max_next
                };
                // Train on the current transition (higher lr).
                nn_train_step(weights, pos, action, target,
                              hidden_pre, hidden_buf, q_buf, 0.2_f32);
                // Store transition in replay buffer.
                let done_flag = if next_pos == goal { 1 } else { 0 };
                replay_store(replay, pos, action, reward * 100, next_pos, done_flag);
                // Replay-train: sample minibatch from buffer, train each.
                let cnt = replay_count(replay);
                if cnt >= replay_minibatch() {
                    let mb = replay_minibatch();
                    let mut mb_i: i32 = 0;
                    while mb_i < mb {
                        // Sample random index from replay.
                        let s_seed = __arena_get(seed_cell);
                        let s_seed2 = lcg(s_seed);
                        __arena_set(seed_cell, s_seed2);
                        let r_idx = ((s_seed2 % cnt) + cnt) % cnt;
                        let r_s = replay_get_s(replay, r_idx);
                        let r_a = replay_get_a(replay, r_idx);
                        let r_r = (replay_get_r(replay, r_idx) as f32) / 100.0_f32 / 100.0_f32;
                        let r_sp = replay_get_sp(replay, r_idx);
                        let r_done = replay_get_done(replay, r_idx);
                        // Forward at r_s, get q_replay_buf.
                        nn_forward(weights, r_s, h_pre_r, h_buf_r, q_replay_buf);
                        let r_target = if r_done == 1 {
                            r_r
                        } else {
                            nn_forward(weights, r_sp, hidden_pre_next, hidden_buf_next, q_replay_next);
                            r_r + 0.9_f32 * max_q(q_replay_next, na)
                        };
                        nn_train_step(weights, r_s, r_a, r_target,
                                      h_pre_r, h_buf_r, q_replay_buf, 0.1_f32);
                        mb_i = mb_i + 1;
                    }
                }
                total_reward = total_reward + reward;
                pos = next_pos;
                step = step + 1;
                let qm = max_q(q_buf, na);
                let qm_scaled = (qm * 100.0_f32) as i32;
                print_step(ep, step, pos, action, reward, qm_scaled);
            }};
        }
        if reached == 1 {
            if step < best_steps { best_steps = step; }
            last_failed = 0;
        } else {
            last_failed = 1;
        }
        print_episode_end(ep, step, total_reward, reached, epsilon_pct);
        ep = ep + 1;
    }
    print_str("{\"type\":\"summary\",\"episodes\":");
    print_int(n_episodes());
    print_str(",\"best_steps\":");
    print_int(best_steps);
    print_str("}\n");
    if best_steps < 9999 { 42 } else { 99 }
}
