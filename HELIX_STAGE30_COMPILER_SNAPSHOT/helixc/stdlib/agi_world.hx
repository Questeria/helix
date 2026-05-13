// helixc/stdlib/agi_world.hx — world-model / state-transition primitives.
//
// Phase 4 step 5: a learned/static world model that predicts the next
// state given (current state, action). The AGI uses this for planning
// (simulate before acting), counterfactual reasoning ("what if I did X?"),
// and self-supervised learning (compare prediction vs actual outcome).
//
// Two implementations:
//   (a) wm_table_*  — explicit (state,action) -> next_state lookup table.
//                     Simple, exact, bounded by table size.
//   (b) wm_linear_* — linear model: next = w*state + b*action + c.
//                     Compact, generalises, but only for scalar dynamics.
//
// API (table-backed):
//   wmt_new(num_states, num_actions) -> i32      allocate table
//   wmt_set(wmt, state, action, next) -> i32    set transition
//   wmt_predict(wmt, state, action)   -> i32    lookup next state
//
// API (linear scalar):
//   wml_new(coef_state, coef_action, bias) -> i32  encode triple in arena
//   wml_predict(wml, state, action) -> i32       w_s*state + w_a*action + b
//
// API for self-supervised learning:
//   wm_prediction_error(predicted, actual) -> i32      |predicted - actual|
//   wm_prediction_error_sq(predicted, actual) -> i32   (predicted - actual)^2
//
// API extras (table-backed):
//   wmt_predict_or(wmt, state, action, default_v) -> i32  predict or default_v if unset
//   wmt_count_set(wmt) -> i32                             count of explicit transitions
//   wmt_is_self_loop(wmt, state, action) -> i32           1 if predict==state, else 0
//
// License: Apache 2.0

// ---- Table-backed world model ----------------------------------------

fn wmt_new(num_states: i32, num_actions: i32) -> i32 {
    let start = __arena_len();
    __arena_push(num_states);
    __arena_push(num_actions);
    let total = num_states * num_actions;
    let mut i: i32 = 0;
    while i < total {
        __arena_push(0 - 1);
        i = i + 1;
    }
    start
}

fn wmt_set(wmt: i32, state: i32, action: i32, next_state: i32) -> i32 {
    let num_actions = __arena_get(wmt + 1);
    let off = wmt + 2 + state * num_actions + action;
    __arena_set(off, next_state);
    0
}

@pure
fn wmt_predict(wmt: i32, state: i32, action: i32) -> i32 {
    let num_actions = __arena_get(wmt + 1);
    __arena_get(wmt + 2 + state * num_actions + action)
}

// ---- Linear scalar world model ---------------------------------------

fn wml_new(coef_state: i32, coef_action: i32, bias: i32) -> i32 {
    let start = __arena_len();
    __arena_push(coef_state);
    __arena_push(coef_action);
    __arena_push(bias);
    start
}

@pure
fn wml_predict(wml: i32, state: i32, action: i32) -> i32 {
    __arena_get(wml) * state + __arena_get(wml + 1) * action + __arena_get(wml + 2)
}

// ---- Self-supervised learning aid ------------------------------------

@pure
fn wm_prediction_error(predicted: i32, actual: i32) -> i32 {
    let d = predicted - actual;
    if d < 0 { 0 - d } else { d }
}

@pure
fn wm_prediction_error_sq(predicted: i32, actual: i32) -> i32 {
    let d = predicted - actual;
    d * d
}

// ---- Imagination rollout: simulate `steps` actions from a state ----

@pure
fn wmt_rollout(wmt: i32, start_state: i32, action_seq_start: i32, steps: i32) -> i32 {
    let num_actions = __arena_get(wmt + 1);
    let mut s: i32 = start_state;
    let mut i: i32 = 0;
    while i < steps {
        let a = __arena_get(action_seq_start + i);
        let off = wmt + 2 + s * num_actions + a;
        let nxt = __arena_get(off);
        s = if nxt < 0 { s } else { nxt };
        i = i + 1;
    }
    s
}

// ---- Table-backed accessors mirroring the option_*/result_* style ----

@pure
fn wmt_predict_or(wmt: i32, state: i32, action: i32, default_v: i32) -> i32 {
    let num_actions = __arena_get(wmt + 1);
    let nxt = __arena_get(wmt + 2 + state * num_actions + action);
    if nxt < 0 { default_v } else { nxt }
}

@pure
fn wmt_count_set(wmt: i32) -> i32 {
    let num_states = __arena_get(wmt);
    let num_actions = __arena_get(wmt + 1);
    let total = num_states * num_actions;
    let mut i: i32 = 0;
    let mut count: i32 = 0;
    while i < total {
        if __arena_get(wmt + 2 + i) >= 0 { count = count + 1; }
        i = i + 1;
    }
    count
}

@pure
fn wmt_is_self_loop(wmt: i32, state: i32, action: i32) -> i32 {
    let num_actions = __arena_get(wmt + 1);
    let nxt = __arena_get(wmt + 2 + state * num_actions + action);
    if nxt == state { 1 } else { 0 }
}
