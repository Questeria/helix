// helixc/stdlib/nn.hx — neural-net primitives in Helix.
//
// Phase 3: tiny NN library on top of the integer tensor primitives
// (helixc/stdlib/tensor.hx). Provides:
//   - dense_layer:    z = W @ x + b
//   - relu_layer:     y = relu(z)
//   - softmax_argmax: returns the index of the largest element
//                     (we don't have softmax probabilities yet because
//                     softmax needs exp + division on f32/f64 stored
//                     in arena, blocked on Phase 2.2 step 2).
//   - mse_loss:       (y - target)^2 summed; lower = better
//
// All over integer tensors for now; f32/f64 NN is Phase 3 step 2 once
// arena bit-reinterpret lands.
//
// License: Apache 2.0

// Forward: z = W @ x + b. y_start must already be allocated (rows
// slots) and gets z+b. Returns 0.
fn dense_layer_forward(w_start: i32, w_rows: i32, w_cols: i32,
                       x_start: i32, b_start: i32, y_start: i32) -> i32 {
    ti2d_matvec(w_start, w_rows, w_cols, x_start, y_start);
    let mut r: i32 = 0;
    while r < w_rows {
        let cur = __arena_get(y_start + r);
        __arena_set(y_start + r, cur + __arena_get(b_start + r));
        r = r + 1;
    }
    0
}

// Element-wise relu in-place: y[i] = max(0, x[i]). Returns 0.
fn relu_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    ti1d_relu(x_start, y_start, n)
}

// argmax: return the index of the largest element in x.
@pure
fn argmax(x_start: i32, n: i32) -> i32 {
    if n == 0 { 0 - 1 }
    else {
        let mut best_idx: i32 = 0;
        let mut best_val: i32 = __arena_get(x_start);
        let mut i: i32 = 1;
        while i < n {
            let v = __arena_get(x_start + i);
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
            i = i + 1;
        }
        best_idx
    }
}

// Sum of squared differences: sum((y[i] - t[i])^2). Lower = closer.
@pure
fn mse_loss(y_start: i32, t_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        let diff = __arena_get(y_start + i) - __arena_get(t_start + i);
        total = total + diff * diff;
        i = i + 1;
    }
    total
}

// Composed NN layers should be called by the user — Phase 0 SysV ABI
// caps at 6 int args, so multi-layer wrappers exceed that. To compose:
//   dense_layer_forward(w1, w1_rows, w1_cols, x, b1, h_pre);
//   relu_layer(h_pre, h, w1_rows);
//   dense_layer_forward(w2, w2_rows, w2_cols, h, b2, y);
//
// Once Phase 1.x lifts the arg limit (stack-spill for 7+ args), a
// multi-layer wrapper can land here.

// =========================================================================
// Phase 3 step 2: training (SGD / one optimizer step).
// =========================================================================

// SGD update for a single int parameter: w_new = w - lr * grad.
// Returns the new value. (Integer math; use lr=1 for "1 unit per
// gradient step" semantics. Float training pending Phase 2.2 step 2.)
@pure
fn sgd_step_scalar(w: i32, g: i32, lr: i32) -> i32 {
    w - lr * g
}

// SGD update for a 1D parameter array in-place.
//   w[i] = w[i] - lr * grad[i] for i in [0, n)
// Returns 0.
fn sgd_step_array(w_start: i32, g_start: i32, lr: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let w = __arena_get(w_start + i);
        let gi = __arena_get(g_start + i);
        __arena_set(w_start + i, w - lr * gi);
        i = i + 1;
    }
    0
}

// Linear-regression gradient w.r.t. weight w in y_pred = w*x + b:
//   loss = (w*x + b - target)^2
//   d_loss/d_w = 2 * (w*x + b - target) * x
// Useful for demo problems; real NN backprop computes per-layer
// gradients via reverse-mode AD (Phase 2.1 step 2).
@pure
fn lin_reg_grad_w(w: i32, b: i32, x: i32, target: i32) -> i32 {
    let pred = w * x + b;
    let err = pred - target;
    2 * err * x
}

// Linear-regression gradient w.r.t. bias b:
//   d_loss/d_b = 2 * (w*x + b - target)
@pure
fn lin_reg_grad_b(w: i32, b: i32, x: i32, target: i32) -> i32 {
    let pred = w * x + b;
    let err = pred - target;
    2 * err
}

// f32 SGD step over an array: w[i] = w[i] - lr * g[i].
fn sgd_f32_step(w_start: i32, g_start: i32, lr: f32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let w_i = __f32_from_bits(__arena_get(w_start + i));
        let g_i = __f32_from_bits(__arena_get(g_start + i));
        __arena_set(w_start + i, __bits_of_f32(w_i - lr * g_i));
        i = i + 1;
    }
    0
}

// MSE on f32 tensors.
@pure
fn mse_loss_f32(y_start: i32, t_start: i32, n: i32) -> f32 {
    if n == 0 { 0.0_f32 }
    else {
        let mut i: i32 = 0;
        let mut total: f32 = 0.0_f32;
        while i < n {
            let yv = __f32_from_bits(__arena_get(y_start + i));
            let tv = __f32_from_bits(__arena_get(t_start + i));
            let d = yv - tv;
            total = total + d * d;
            i = i + 1;
        }
        total / (n as f32)
    }
}

fn dense_layer_f32_forward(w_start: i32, w_rows: i32, w_cols: i32,
                           x_start: i32, b_start: i32, y_start: i32) -> i32 {
    tf2d_matvec(w_start, w_rows, w_cols, x_start, y_start);
    let mut r: i32 = 0;
    while r < w_rows {
        let cur = __f32_from_bits(__arena_get(y_start + r));
        let bv = __f32_from_bits(__arena_get(b_start + r));
        __arena_set(y_start + r, __bits_of_f32(cur + bv));
        r = r + 1;
    }
    0
}

// Leaky ReLU.
fn leaky_relu_layer(x_start: i32, alpha: f32, y_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        let v = if xi > 0.0_f32 { xi } else { alpha * xi };
        __arena_set(y_start + i, __bits_of_f32(v));
        i = i + 1;
    }
    0
}

// Momentum SGD.
fn momentum_step(w_start: i32, v_start: i32, g_start: i32,
                 beta: f32, lr: f32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let w_i = __f32_from_bits(__arena_get(w_start + i));
        let v_i = __f32_from_bits(__arena_get(v_start + i));
        let g_i = __f32_from_bits(__arena_get(g_start + i));
        let new_v = beta * v_i + g_i;
        __arena_set(v_start + i, __bits_of_f32(new_v));
        __arena_set(w_start + i, __bits_of_f32(w_i - lr * new_v));
        i = i + 1;
    }
    0
}

// tanh layer (uses __exp).
fn tanh_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        let e2x = __exp(2.0_f32 * xi);
        let v = (e2x - 1.0_f32) / (e2x + 1.0_f32);
        __arena_set(y_start + i, __bits_of_f32(v));
        i = i + 1;
    }
    0
}

// sigmoid layer (uses __sigmoid).
fn sigmoid_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        let v = __sigmoid(xi);
        __arena_set(y_start + i, __bits_of_f32(v));
        i = i + 1;
    }
    0
}

// Softmax (max-subtract, uses __exp + tf1d_max).
fn softmax_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    if n == 0 { 0 }
    else {
        let max_v = tf1d_max(x_start, n);
        let mut i: i32 = 0;
        let mut sum_e: f32 = 0.0_f32;
        while i < n {
            let xi = __f32_from_bits(__arena_get(x_start + i));
            let e = __exp(xi - max_v);
            __arena_set(y_start + i, __bits_of_f32(e));
            sum_e = sum_e + e;
            i = i + 1;
        }
        let mut j: i32 = 0;
        while j < n {
            let e = __f32_from_bits(__arena_get(y_start + j));
            __arena_set(y_start + j, __bits_of_f32(e / sum_e));
            j = j + 1;
        }
        0
    }
}

// BCE.
@pure
fn bce_loss_scalar(p: f32, t: f32) -> f32 {
    let eps = 0.0001_f32;
    let pc = __max(__min(p, 1.0_f32 - eps), eps);
    let inv = 1.0_f32 - t;
    let one_minus_pc = 1.0_f32 - pc;
    0.0_f32 - (t * __log(pc) + inv * __log(one_minus_pc))
}

// Cross-entropy.
@pure
fn ce_loss(p_start: i32, target_idx: i32) -> f32 {
    let p = __f32_from_bits(__arena_get(p_start + target_idx));
    let eps = 0.0001_f32;
    let pc = __max(p, eps);
    0.0_f32 - __log(pc)
}

// argmin: index of smallest element. Companion to argmax.
// Returns -1 on empty.
@pure
fn argmin(x_start: i32, n: i32) -> i32 {
    if n == 0 { 0 - 1 }
    else {
        let mut best_idx: i32 = 0;
        let mut best_val: i32 = __arena_get(x_start);
        let mut i: i32 = 1;
        while i < n {
            let v = __arena_get(x_start + i);
            if v < best_val {
                best_val = v;
                best_idx = i;
            }
            i = i + 1;
        }
        best_idx
    }
}

// MAE (sum of absolute differences) on integer tensors.
// Sibling of mse_loss; cheaper since no multiplication and no overflow risk.
@pure
fn mae_loss(y_start: i32, t_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut total: i32 = 0;
    while i < n {
        let d = __arena_get(y_start + i) - __arena_get(t_start + i);
        let ad = if d < 0 { 0 - d } else { d };
        total = total + ad;
        i = i + 1;
    }
    total
}

// MAE on f32 tensors (mean absolute error, returns 0.0 on empty).
@pure
fn mae_loss_f32(y_start: i32, t_start: i32, n: i32) -> f32 {
    if n == 0 { 0.0_f32 }
    else {
        let mut i: i32 = 0;
        let mut total: f32 = 0.0_f32;
        while i < n {
            let yv = __f32_from_bits(__arena_get(y_start + i));
            let tv = __f32_from_bits(__arena_get(t_start + i));
            total = total + __abs(yv - tv);
            i = i + 1;
        }
        total / (n as f32)
    }
}

// Count of positions where prediction matches target. Useful for batch
// classification accuracy: pred[i] is typically argmax of model output,
// target[i] is the integer class label.
@pure
fn count_correct(pred_start: i32, target_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    let mut hits: i32 = 0;
    while i < n {
        let p = __arena_get(pred_start + i);
        let t = __arena_get(target_start + i);
        if p == t { hits = hits + 1; }
        i = i + 1;
    }
    hits
}
