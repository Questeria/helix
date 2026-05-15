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

// Adam-style optimizer step with uncorrected moving moments.
// m[i] = beta1*m[i] + (1-beta1)*g[i]
// v[i] = beta2*v[i] + (1-beta2)*g[i]^2
// w[i] = w[i] - lr*m[i]/(sqrt(v[i]) + eps)
fn adam_f32_step(w_start: i32, g_start: i32, m_start: i32, v_start: i32,
                 lr: f32, beta1: f32, beta2: f32, eps: f32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let w_i = __f32_from_bits(__arena_get(w_start + i));
        let g_i = __f32_from_bits(__arena_get(g_start + i));
        let m_i = __f32_from_bits(__arena_get(m_start + i));
        let v_i = __f32_from_bits(__arena_get(v_start + i));
        let next_m = beta1 * m_i + (1.0_f32 - beta1) * g_i;
        let next_v = beta2 * v_i + (1.0_f32 - beta2) * g_i * g_i;
        __arena_set(m_start + i, __bits_of_f32(next_m));
        __arena_set(v_start + i, __bits_of_f32(next_v));
        let raw_denom = __sqrt(next_v) + eps;
        if raw_denom <= 0.0_f32 {
            __arena_set(w_start + i, __bits_of_f32(w_i));
        }
        else {
            __arena_set(w_start + i,
                __bits_of_f32(w_i - lr * next_m / raw_denom));
        };
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

// Gradient of mean squared error with respect to y:
//   d/dy[i] mean((y - t)^2) = 2 * (y[i] - t[i]) / n
fn mse_loss_f32_grad(y_start: i32, t_start: i32,
                     dy_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else {
        let scale = 2.0_f32 / (n as f32);
        let mut i: i32 = 0;
        while i < n {
            let yv = __f32_from_bits(__arena_get(y_start + i));
            let tv = __f32_from_bits(__arena_get(t_start + i));
            __arena_set(dy_start + i, __bits_of_f32((yv - tv) * scale));
            i = i + 1;
        }
        0
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

// Dense layer backward helpers for y = W @ x + b.
// grad_w[r, c] = grad_y[r] * x[c]
fn dense_layer_f32_grad_w(dy_start: i32, x_start: i32,
                          grad_w_start: i32, rows: i32, cols: i32) -> i32 {
    let mut r: i32 = 0;
    while r < rows {
        let dy = __f32_from_bits(__arena_get(dy_start + r));
        let mut c: i32 = 0;
        while c < cols {
            let x = __f32_from_bits(__arena_get(x_start + c));
            __arena_set(grad_w_start + r * cols + c, __bits_of_f32(dy * x));
            c = c + 1;
        }
        r = r + 1;
    }
    0
}

fn dense_layer_f32_grad_b(dy_start: i32, grad_b_start: i32, rows: i32) -> i32 {
    let mut r: i32 = 0;
    while r < rows {
        __arena_set(grad_b_start + r, __arena_get(dy_start + r));
        r = r + 1;
    }
    0
}

// grad_x[c] = sum_r W[r, c] * grad_y[r]
fn dense_layer_f32_grad_x(w_start: i32, dy_start: i32,
                          grad_x_start: i32, rows: i32, cols: i32) -> i32 {
    let mut c: i32 = 0;
    while c < cols {
        let mut r: i32 = 0;
        let mut acc: f32 = 0.0_f32;
        while r < rows {
            let w = __f32_from_bits(__arena_get(w_start + r * cols + c));
            let dy = __f32_from_bits(__arena_get(dy_start + r));
            acc = acc + w * dy;
            r = r + 1;
        }
        __arena_set(grad_x_start + c, __bits_of_f32(acc));
        c = c + 1;
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

fn relu_layer_f32_backward(x_start: i32, dy_start: i32,
                           dx_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let x = __f32_from_bits(__arena_get(x_start + i));
        let dy = __f32_from_bits(__arena_get(dy_start + i));
        let dx = if x > 0.0_f32 { dy } else { 0.0_f32 };
        __arena_set(dx_start + i, __bits_of_f32(dx));
        i = i + 1;
    }
    0
}

fn sigmoid_layer_backward(y_start: i32, dy_start: i32,
                          dx_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let y = __f32_from_bits(__arena_get(y_start + i));
        let dy = __f32_from_bits(__arena_get(dy_start + i));
        __arena_set(dx_start + i, __bits_of_f32(dy * y * (1.0_f32 - y)));
        i = i + 1;
    }
    0
}

fn tanh_layer_backward(y_start: i32, dy_start: i32,
                       dx_start: i32, n: i32) -> i32 {
    let mut i: i32 = 0;
    while i < n {
        let y = __f32_from_bits(__arena_get(y_start + i));
        let dy = __f32_from_bits(__arena_get(dy_start + i));
        __arena_set(dx_start + i, __bits_of_f32(dy * (1.0_f32 - y * y)));
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

fn softmax_rows_f32(logits_start: i32, probs_start: i32,
                    rows: i32, cols: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else {
        let mut r: i32 = 0;
        while r < rows {
            let row_in = logits_start + r * cols;
            let row_out = probs_start + r * cols;
            softmax_layer(row_in, row_out, cols);
            r = r + 1;
        }
        0
    }}
}

fn softmax_ce_grad_f32(probs_start: i32, target_start: i32,
                       grad_start: i32, rows: i32, cols: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else {
        let mut r: i32 = 0;
        while r < rows {
            let target = __arena_get(target_start + r);
            if target < 0 {
                r = rows + 1;
            }
            else {
                if target >= cols {
                    r = rows + 1;
                };
            };
            r = r + 1;
        }
        if r > rows { 35001 }
        else {
            let scale = 1.0_f32 / (rows as f32);
            let mut rr: i32 = 0;
            while rr < rows {
                let target2 = __arena_get(target_start + rr);
                let row_start = probs_start + rr * cols;
                let grad_row = grad_start + rr * cols;
                let mut c: i32 = 0;
                while c < cols {
                    let p = __f32_from_bits(__arena_get(row_start + c));
                    let y = if c == target2 { 1.0_f32 } else { 0.0_f32 };
                    __arena_set(grad_row + c, __bits_of_f32((p - y) * scale));
                    c = c + 1;
                }
                rr = rr + 1;
            }
            0
        }
    }}
}

fn dense_classifier_sgd_step_f32(w_start: i32, b_start: i32, x_start: i32,
                                 target: i32, scratch_start: i32,
                                 shape_start: i32, lr: f32) -> i32 {
    let classes = __arena_get(shape_start);
    let in_dim = __arena_get(shape_start + 1);
    if classes <= 0 { 0 }
    else { if in_dim <= 0 { 0 }
    else {
        if target < 0 { 35001 }
        else { if target >= classes { 35001 }
        else {
            let mut max_score = __f32_from_bits(__arena_get(b_start));
            let mut j0: i32 = 0;
            while j0 < in_dim {
                let w0 = __f32_from_bits(__arena_get(w_start + j0));
                let x0 = __f32_from_bits(__arena_get(x_start + j0));
                max_score = max_score + w0 * x0;
                j0 = j0 + 1;
            }
            let mut mc: i32 = 1;
            while mc < classes {
                let mut score = __f32_from_bits(__arena_get(b_start + mc));
                let mut mj: i32 = 0;
                while mj < in_dim {
                    let wv = __f32_from_bits(__arena_get(w_start + mc * in_dim + mj));
                    let xv = __f32_from_bits(__arena_get(x_start + mj));
                    score = score + wv * xv;
                    mj = mj + 1;
                }
                if score > max_score { max_score = score; };
                mc = mc + 1;
            }
            let mut sum_e: f32 = 0.0_f32;
            let mut ec: i32 = 0;
            while ec < classes {
                let mut score2 = __f32_from_bits(__arena_get(b_start + ec));
                let mut ej: i32 = 0;
                while ej < in_dim {
                    let wv2 = __f32_from_bits(__arena_get(w_start + ec * in_dim + ej));
                    let xv2 = __f32_from_bits(__arena_get(x_start + ej));
                    score2 = score2 + wv2 * xv2;
                    ej = ej + 1;
                }
                sum_e = sum_e + __exp(score2 - max_score);
                ec = ec + 1;
            }
            let mut cls: i32 = 0;
            while cls < classes {
                let mut score3 = __f32_from_bits(__arena_get(b_start + cls));
                let mut sj: i32 = 0;
                while sj < in_dim {
                    let wv3 = __f32_from_bits(__arena_get(w_start + cls * in_dim + sj));
                    let xv3 = __f32_from_bits(__arena_get(x_start + sj));
                    score3 = score3 + wv3 * xv3;
                    sj = sj + 1;
                }
                let p = __exp(score3 - max_score) / sum_e;
                let y = if cls == target { 1.0_f32 } else { 0.0_f32 };
                let dy = p - y;
                let mut j: i32 = 0;
                while j < in_dim {
                    let w_idx = w_start + cls * in_dim + j;
                    let wv = __f32_from_bits(__arena_get(w_idx));
                    let xv = __f32_from_bits(__arena_get(x_start + j));
                    __arena_set(w_idx, __bits_of_f32(wv - lr * dy * xv));
                    j = j + 1;
                }
                let bv = __f32_from_bits(__arena_get(b_start + cls));
                __arena_set(b_start + cls, __bits_of_f32(bv - lr * dy));
                cls = cls + 1;
            }
            0
        }}
    }}
}

// BCE.
@pure
fn bce_loss_scalar(p: f32, t: f32) -> f32 {
    let eps = 0.0001_f32;
    let pc = __max(__min(p, 1.0_f32 - eps), eps);
    let inv = 1.0_f32 - t;
    let one_minus_pc = 1.0_f32 - pc;
    0.0_f32 - (t * __log_stable(pc) + inv * __log_stable(one_minus_pc))
}

// Cross-entropy.
@pure
fn ce_loss(p_start: i32, target_idx: i32) -> f32 {
    if target_idx < 0 { 1000000.0_f32 }
    else {
        let p = __f32_from_bits(__arena_get(p_start + target_idx));
        let eps = 0.0001_f32;
        let pc = __max(p, eps);
        0.0_f32 - __log_stable(pc)
    }
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
        let mut invalid: i32 = 0;
        while r < rows {
            let row_start = probs_start + r * cols;
            let target = __arena_get(target_start + r);
            if target < 0 {
                invalid = 1;
            }
            else {
                if target >= cols {
                    invalid = 1;
                }
                else {
                    total = total + ce_loss(row_start, target);
                };
            };
            r = r + 1;
        }
        if invalid == 1 { 1000000.0_f32 }
        else { total / (rows as f32) }
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
