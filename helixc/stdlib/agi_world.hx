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

@pure fn wmt_magic() -> i32 { 6006001 }

@pure fn wmt_footer(num_states: i32, num_actions: i32) -> i32 {
    0 - wmt_magic() - num_states - num_actions
}

@pure fn wmt_len(num_states: i32, num_actions: i32) -> i32 {
    if num_states <= 0 { 0 }
    else { if num_actions <= 0 { 0 }
    else { if num_states > 2147483647 / num_actions { 0 }
    else { num_states * num_actions } } }
}

fn wmt_new(num_states: i32, num_actions: i32) -> i32 {
    let total = wmt_len(num_states, num_actions);
    if total == 0 { 0 - 1 }
    else {
        __arena_push(wmt_magic());
        let start = __arena_len();
        __arena_push(num_states);
        __arena_push(num_actions);
        let mut i: i32 = 0;
        while i < total {
            __arena_push(0 - 1);
            i = i + 1;
        }
        __arena_push(wmt_footer(num_states, num_actions));
        start
    }
}

@pure fn wmt_ok(wmt: i32) -> i32 {
    if wmt <= 0 { 0 }
    else { if __arena_get(wmt - 1) != wmt_magic() { 0 }
    else {
        let num_states = __arena_get(wmt);
        let num_actions = __arena_get(wmt + 1);
        let total = wmt_len(num_states, num_actions);
        if total == 0 { 0 }
        else { if total > 2147483647 - wmt - 2 { 0 }
        else { if wmt + 2 + total >= __arena_len() { 0 }
        else { if __arena_get(wmt + 2 + total) != wmt_footer(num_states, num_actions) { 0 }
        else { 1 } } } }
    }}
}

@pure fn wmt_offset(wmt: i32, state: i32, action: i32) -> i32 {
    if wmt_ok(wmt) == 0 { 0 - 1 }
    else {
        let num_states = __arena_get(wmt);
        let num_actions = __arena_get(wmt + 1);
        if state < 0 { 0 - 1 }
        else { if state >= num_states { 0 - 1 }
        else { if action < 0 { 0 - 1 }
        else { if action >= num_actions { 0 - 1 }
        else { wmt + 2 + state * num_actions + action } } } }
    }
}

fn wmt_set(wmt: i32, state: i32, action: i32, next_state: i32) -> i32 {
    let off = wmt_offset(wmt, state, action);
    if off < 0 { 0 - 1 }
    else { if next_state < 0 { 0 - 1 }
    else { if next_state >= __arena_get(wmt) { 0 - 1 }
    else {
        __arena_set(off, next_state);
        0
    } } }
}

@pure
fn wmt_predict(wmt: i32, state: i32, action: i32) -> i32 {
    let off = wmt_offset(wmt, state, action);
    if off < 0 { 0 - 1 }
    else {
        let nxt = __arena_get(off);
        if nxt < 0 { 0 - 1 }
        else { if nxt >= __arena_get(wmt) { 0 - 1 } else { nxt } }
    }
}

// ---- Linear scalar world model ---------------------------------------

@pure fn wml_magic() -> i32 { 6007001 }

@pure fn wml_footer() -> i32 { 0 - wml_magic() - 3 }

fn wml_new(coef_state: i32, coef_action: i32, bias: i32) -> i32 {
    __arena_push(wml_magic());
    let start = __arena_len();
    __arena_push(coef_state);
    __arena_push(coef_action);
    __arena_push(bias);
    __arena_push(wml_footer());
    start
}

@pure
fn wml_ok(wml: i32) -> i32 {
    if wml <= 0 { 0 }
    else { if __arena_get(wml - 1) != wml_magic() { 0 }
    else { if wml + 3 >= __arena_len() { 0 }
    else { if __arena_get(wml + 3) != wml_footer() { 0 }
    else { 1 } } } }
}

@pure
fn wml_predict(wml: i32, state: i32, action: i32) -> i32 {
    if wml_ok(wml) == 0 { 0 - 1 }
    else { __arena_get(wml) * state + __arena_get(wml + 1) * action + __arena_get(wml + 2) }
}

// ---- Self-supervised learning aid ------------------------------------

@pure
fn wm_prediction_error(predicted: i32, actual: i32) -> i32 {
    if predicted >= actual {
        if actual < 0 {
            let limit = 2147483647 + actual;
            if predicted > limit { 2147483647 } else { predicted - actual }
        } else { predicted - actual }
    } else {
        if predicted < 0 {
            let limit = 2147483647 + predicted;
            if actual > limit { 2147483647 } else { actual - predicted }
        } else { actual - predicted }
    }
}

@pure
fn wm_prediction_error_sq(predicted: i32, actual: i32) -> i32 {
    let d = wm_prediction_error(predicted, actual);
    if d > 46340 { 2147483647 } else { d * d }
}

// ---- Imagination rollout: simulate `steps` actions from a state ----

@pure
fn wmt_rollout(wmt: i32, start_state: i32, action_seq_start: i32, steps: i32) -> i32 {
    if steps < 0 { 0 - 1 }
    else { if wmt_ok(wmt) == 0 { 0 - 1 }
    else { if t1d_slice_ok(action_seq_start, steps) == 0 { 0 - 1 }
    else { if start_state < 0 { 0 - 1 }
    else { if start_state >= __arena_get(wmt) { 0 - 1 }
    else {
        let mut s: i32 = start_state;
        let mut i: i32 = 0;
        while i < steps {
            let a = __arena_get(action_seq_start + i);
            let off = wmt_offset(wmt, s, a);
            if off < 0 {
                s = 0 - 1;
                i = steps;
            } else {
                let nxt = __arena_get(off);
                if nxt < 0 {
                    s = 0 - 1;
                    i = steps;
                } else {
                    if nxt >= __arena_get(wmt) {
                        s = 0 - 1;
                        i = steps;
                    } else {
                        s = nxt;
                        i = i + 1;
                    };
                };
            };
        }
        s
    } } }}}
}

// ---- Table-backed accessors mirroring the option_*/result_* style ----

@pure
fn wmt_predict_or(wmt: i32, state: i32, action: i32, default_v: i32) -> i32 {
    let off = wmt_offset(wmt, state, action);
    let nxt = if off < 0 { 0 - 1 } else { __arena_get(off) };
    if off < 0 { 0 - 1 }
    else { if nxt < 0 { default_v }
    else { if nxt >= __arena_get(wmt) { 0 - 1 } else { nxt } } }
}

@pure
fn wmt_count_set(wmt: i32) -> i32 {
    if wmt_ok(wmt) == 0 { 0 }
    else {
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
}

@pure
fn wmt_is_self_loop(wmt: i32, state: i32, action: i32) -> i32 {
    let off = wmt_offset(wmt, state, action);
    if off < 0 { 0 }
    else {
        let nxt = wmt_predict(wmt, state, action);
        if nxt == state { 1 } else { 0 }
    }
}
