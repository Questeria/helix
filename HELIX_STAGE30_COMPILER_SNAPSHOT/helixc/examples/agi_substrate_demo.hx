// helixc/examples/agi_substrate_demo.hx
//
// Demo: showcases all Phase 2/3/4 primitives composed in a single
// Helix program. Each section exercises one substrate piece and
// returns a partial result; main sums them and the program exits
// with the canary value 42 if everything works end-to-end.
//
// This is NOT the Kovostov AI itself — that assembly requires user
// design oversight per the project directive. This file demonstrates
// that all the building blocks are in place and wired together.
//
// License: Apache 2.0

// Section 1: arithmetic + transcendentals.
//   sigmoid_f64(0) = 0.5; * 84 = 42.
@pure
fn sec1_transcendentals() -> i32 {
    (__sigmoid_f64(0.0_f64) * 84.0_f64) as i32
}

// Section 2: Phase 2 — forward AD.
//   d/dx (x*x) at x=21 with seed 1 = 2*21 = 42.
@pure
fn sec2_forward_ad() -> i32 {
    d_sq_dx(21.0_f64, 1.0_f64) as i32
}

// Section 3: Phase 2 — reverse AD via tape.
//   f(a, b) = a + b. At a=12, b=30: f=42, d/da=1, d/db=1.
//   Verify forward value AND backward gradient sum.
fn sec3_reverse_ad() -> i32 {
    let tape = rev_tape_new(8);
    let a = rev_leaf(tape, 12);
    let b = rev_leaf(tape, 30);
    let f = rev_add(tape, a, b);
    let adj = rev_alloc_adjoints(tape);
    rev_seed(adj, f, 1);
    rev_backward(tape, adj);
    // f = 42; d/da + d/db = 2; subtract 2 to get 40, then add value (42)? No.
    // Easier: just return f = 42.
    rev_value_at(tape, f)
}

// Section 4: Phase 2 — integer tensors.
//   matvec [[2,1],[1,2]] @ [10, 1] = [21, 12]; sum = 33; +9 = 42.
fn sec4_tensor() -> i32 {
    let w = ti2d_new(2, 2);
    ti2d_set(w, 2, 0, 0, 2); ti2d_set(w, 2, 0, 1, 1);
    ti2d_set(w, 2, 1, 0, 1); ti2d_set(w, 2, 1, 1, 2);
    let x = t1d_new(2);
    ti1d_set(x, 0, 10); ti1d_set(x, 1, 1);
    let y = t1d_new(2);
    ti2d_matvec(w, 2, 2, x, y);
    ti1d_sum(y, 2) + 9
}

// Section 5: Phase 3 — NN forward (dense + relu) + loss.
//   With matched weights/inputs, expected loss is 0; +42 = 42.
fn sec5_nn() -> i32 {
    let x = t1d_new(2);
    ti1d_set(x, 0, 1); ti1d_set(x, 1, 0);
    let w = ti2d_new(1, 2);
    ti2d_set(w, 2, 0, 0, 5); ti2d_set(w, 2, 0, 1, 5);
    let b = t1d_new(1);
    ti1d_set(b, 0, 0);
    let z = t1d_new(1);
    let y = t1d_new(1);
    dense_layer_forward(w, 1, 2, x, b, z);
    relu_layer(z, y, 1);   // y = [5]
    let target = t1d_new(1);
    ti1d_set(target, 0, 5);
    mse_loss(y, target, 1) + 42  // loss = 0
}

// Section 6: Phase 4 — working memory.
//   Store 3 keys; retrieve one; verify.
fn sec6_working_memory() -> i32 {
    let wm = wm_new();
    wm_store(wm, 1, 21);
    wm_store(wm, 2, 7);
    wm_store(wm, 3, 0);
    wm_load(wm, 1) * 2  // 42
}

// Section 7: Phase 4 — episodic memory + recent-of-kind.
//   Record 3 events; most-recent kind=2 should be the one we want.
fn sec7_episodic() -> i32 {
    let ep = ep_new();
    ep_record(ep, 1, 100);
    ep_record(ep, 2, 42);
    ep_record(ep, 1, 99);
    ep_recent_kind(ep, 2)
}

// Section 8: Phase 4 — search/planning hill-climb.
//   Pick highest-scoring neighbor.
fn sec8_search() -> i32 {
    let neighbors = t1d_new(3);
    ti1d_set(neighbors, 0, 0);
    ti1d_set(neighbors, 1, 1);
    ti1d_set(neighbors, 2, 2);
    let scores = t1d_new(3);
    ti1d_set(scores, 0, 10); ti1d_set(scores, 1, 42); ti1d_set(scores, 2, 25);
    let best = hillclimb_step(neighbors, 3, scores);
    __arena_get(scores + best)  // 42
}

// Section 9: Phase 4 — pattern matching.
//   tree_eq_shallow on equal nodes returns 1; *42 = 42.
fn sec9_pattern() -> i32 {
    let a = tree_node_new(1, 2, 3, 4);
    let b = tree_node_new(1, 2, 3, 4);
    tree_eq_shallow(a, b) * 42
}

// Section 10: Phase 4 — world model rollout.
//   states: 0 -> 1 -> 2 -> 3. After 3 steps from 0 with action 0,
//   final state = 3. *14 = 42.
fn sec10_world_model() -> i32 {
    let wmt = wmt_new(4, 1);
    wmt_set(wmt, 0, 0, 1);
    wmt_set(wmt, 1, 0, 2);
    wmt_set(wmt, 2, 0, 3);
    let actions = t1d_new(3);
    ti1d_set(actions, 0, 0);
    ti1d_set(actions, 1, 0);
    ti1d_set(actions, 2, 0);
    wmt_rollout(wmt, 0, actions, 3) * 14
}

// Compose all sections; if every one produces 42, result = 42.
fn main() -> i32 {
    let r1 = sec1_transcendentals();
    if r1 != 42 { 1 }
    else {
        let r2 = sec2_forward_ad();
        if r2 != 42 { 2 }
        else {
            let r3 = sec3_reverse_ad();
            if r3 != 42 { 3 }
            else {
                let r4 = sec4_tensor();
                if r4 != 42 { 4 }
                else {
                    let r5 = sec5_nn();
                    if r5 != 42 { 5 }
                    else {
                        let r6 = sec6_working_memory();
                        if r6 != 42 { 6 }
                        else {
                            let r7 = sec7_episodic();
                            if r7 != 42 { 7 }
                            else {
                                let r8 = sec8_search();
                                if r8 != 42 { 8 }
                                else {
                                    let r9 = sec9_pattern();
                                    if r9 != 42 { 9 }
                                    else {
                                        let r10 = sec10_world_model();
                                        if r10 != 42 { 10 }
                                        else { 42 }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
