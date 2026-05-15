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

// Clip an f32 gradient vector in place if its L2 norm is above max_norm.
// Returns 0; g_start is mutated only when clipping is needed.
fn clip_grad_norm_f32(g_start: i32, max_norm: f32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else {
        let norm_sq = tf1d_l2_norm_sq(g_start, n);
        if norm_sq <= 0.0_f32 { 0 }
        else {
            let norm = __sqrt(norm_sq);
            let target = if max_norm < 0.0_f32 { 0.0_f32 } else { max_norm };
            if norm > target {
                let scale = target / norm;
                tf1d_scale_inplace(g_start, n, scale);
            };
            0
        }
    }
}

// Add L2 weight-decay contribution to an f32 gradient vector:
//   g[i] = g[i] + decay * w[i]
// This is the standard gradient contribution for 0.5 * decay * ||w||^2.
fn add_weight_decay_grad_f32(g_start: i32, w_start: i32,
                             decay: f32, n: i32) -> i32 {
    tf1d_axpby(w_start, g_start, decay, 1.0_f32, n)
}

// One stable f32 optimizer step:
//   1. add weight decay to gradient
//   2. clip gradient norm
//   3. apply SGD
fn sgd_f32_step_decay_clip(w_start: i32, g_start: i32, lr: f32,
                           decay: f32, max_norm: f32, n: i32) -> i32 {
    add_weight_decay_grad_f32(g_start, w_start, decay, n);
    clip_grad_norm_f32(g_start, max_norm, n);
    sgd_f32_step(w_start, g_start, lr, n)
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

fn softplus_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(y_start + i, __bits_of_f32(__softplus(xi)));
        i = i + 1;
    }
    0
}

fn silu_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(y_start + i, __bits_of_f32(__silu(xi)));
        i = i + 1;
    }
    0
}

fn gelu_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(y_start + i, __bits_of_f32(__gelu(xi)));
        i = i + 1;
    }
    0
}

// Layer normalization over one f32 vector.
// y[i] = (x[i] - mean(x)) / sqrt(variance(x) + eps)
fn layer_norm_f32(x_start: i32, y_start: i32, n: i32, eps: f32) -> i32 {
    if n == 0 { 0 }
    else {
        let mean = tf1d_mean(x_start, n);
        let mut i: i32 = 0;
        let mut var: f32 = 0.0_f32;
        while i < n {
            let x = __f32_from_bits(__arena_get(x_start + i));
            let d = x - mean;
            var = var + d * d;
            i = i + 1;
        }
        var = var / (n as f32);
        let inv_std = 1.0_f32 / __sqrt(var + eps);
        let mut j: i32 = 0;
        while j < n {
            let xj = __f32_from_bits(__arena_get(x_start + j));
            __arena_set(y_start + j, __bits_of_f32((xj - mean) * inv_std));
            j = j + 1;
        }
        0
    }
}

// Inverted dropout for f32 vectors. During training, each element is kept with
// probability keep_prob and scaled by 1/keep_prob; dropped elements become 0.
// Returns the final deterministic RNG state.
fn dropout_f32(x_start: i32, y_start: i32, n: i32,
               keep_prob: f32, seed: i32) -> i32 {
    if keep_prob <= 0.0_f32 {
        let mut i: i32 = 0;
        while i < n {
            __arena_set(y_start + i, __bits_of_f32(0.0_f32));
            i = i + 1;
        }
        seed
    }
    else {
        if keep_prob >= 1.0_f32 {
            let mut j: i32 = 0;
            while j < n {
                __arena_set(y_start + j, __arena_get(x_start + j));
                j = j + 1;
            }
            seed
        }
        else {
            let mut state: i32 = seed;
            let mut k: i32 = 0;
            while k < n {
                state = __rand_step(state);
                let r = __rand_float(state);
                if r < keep_prob {
                    let x = __f32_from_bits(__arena_get(x_start + k));
                    __arena_set(y_start + k, __bits_of_f32(x / keep_prob));
                }
                else {
                    __arena_set(y_start + k, __bits_of_f32(0.0_f32));
                };
                k = k + 1;
            }
            state
        }
    }
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

// For a row-major logits matrix (rows x cols), write each row's argmax class
// index to out_start[row].
fn argmax_rows_f32(logits_start: i32, rows: i32, cols: i32,
                   out_start: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else {
        let mut r: i32 = 0;
        while r < rows {
            let row_start = logits_start + r * cols;
            let mut best_idx: i32 = 0;
            let mut best_val: f32 = __f32_from_bits(__arena_get(row_start));
            let mut c: i32 = 1;
            while c < cols {
                let v = __f32_from_bits(__arena_get(row_start + c));
                if v > best_val {
                    best_val = v;
                    best_idx = c;
                };
                c = c + 1;
            }
            __arena_set(out_start + r, best_idx);
            r = r + 1;
        }
        0
    }}
}

@pure
fn accuracy_count_from_logits_f32(logits_start: i32, target_start: i32,
                                  rows: i32, cols: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else {
        let mut r: i32 = 0;
        let mut hits: i32 = 0;
        while r < rows {
            let row_start = logits_start + r * cols;
            let mut best_idx: i32 = 0;
            let mut best_val: f32 = __f32_from_bits(__arena_get(row_start));
            let mut c: i32 = 1;
            while c < cols {
                let v = __f32_from_bits(__arena_get(row_start + c));
                if v > best_val {
                    best_val = v;
                    best_idx = c;
                };
                c = c + 1;
            }
            if best_idx == __arena_get(target_start + r) {
                hits = hits + 1;
            };
            r = r + 1;
        }
        hits
    }}
}

@pure
fn ce_loss_batch_f32(probs_start: i32, target_start: i32,
                     rows: i32, cols: i32) -> f32 {
    if rows <= 0 { 0.0_f32 }
    else { if cols <= 0 { 0.0_f32 }
    else {
        let mut r: i32 = 0;
        let mut total: f32 = 0.0_f32;
        while r < rows {
            let row_start = probs_start + r * cols;
            let target = __arena_get(target_start + r);
            total = total + ce_loss(row_start, target);
            r = r + 1;
        }
        total / (rows as f32)
    }}
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
