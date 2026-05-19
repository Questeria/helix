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
    if w_rows <= 0 { 0 }
    else { if w_cols <= 0 { 0 }
    else { if t2d_len(w_rows, w_cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(w_start, w_rows, w_cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(x_start, w_cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(b_start, w_rows) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, w_rows) == 0 { t2d_error() }
    else {
    ti2d_matvec(w_start, w_rows, w_cols, x_start, y_start);
    // Restart 53 A6: i64 intermediate + INT32 saturation on the bias
    // add so the saturation guarantee from ti2d_matvec (restart 52 A1)
    // is preserved through the dense-layer output (otherwise an
    // INT32_MAX cur + positive bias silently wraps to negative).
    let mut r: i32 = 0;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    while r < w_rows {
        let cur: i64 = __arena_get(y_start + r) as i64;
        let bv: i64 = __arena_get(b_start + r) as i64;
        let mut v: i64 = cur + bv;
        if v > hi { v = hi; }
        else { if v < lo { v = lo; } };
        __arena_set(y_start + r, v as i32);
        r = r + 1;
    }
    0
    }}}}}}}
}

// Element-wise relu in-place: y[i] = max(0, x[i]). Returns 0.
fn relu_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    ti1d_relu(x_start, y_start, n)
}

// argmax: return the index of the largest element in x.
@pure
fn argmax(x_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 - 1 }
    else { if t1d_slice_ok(x_start, n) == 0 { 0 - 1 }
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
    }}
}

// Sum of squared differences: sum((y[i] - t[i])^2). Lower = closer.
// Restart 51 A1: i64 accumulator + INT32 saturation. A single element
// |diff| >= 46341 makes diff*diff exceed INT32_MAX; the prior i32 total
// silently wrapped, fooling any "lower-is-better" loss monitor. Matches
// the ti1d_prod (restart 50 A3) and hashmap_sum_values precedent.
@pure
fn mse_loss(y_start: i32, t_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(y_start, n) == 0 { 0 }
    else { if t1d_slice_ok(t_start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i64 = 0_i64;
    let hi: i64 = 2147483647_i64;
    while i < n {
        let diff = (__arena_get(y_start + i) as i64) - (__arena_get(t_start + i) as i64);
        total = total + diff * diff;
        if total > hi { total = hi; };
        i = i + 1;
    }
    total as i32
    }}}
}

// Cycle 2 Batch RT fix batch 17 (silent-failure HIGH-3):
// Pre-fix: mse_loss returned 0 on corruption (n<=0 or t1d_slice_ok
// failure). 0 collides with "perfect prediction, training converged."
// A loss-monitor watching mse_loss for "lower is better" saw
// converged-state when slices were actually corrupt. Training loops
// declared success despite broken data.
// Post-fix: mse_loss_strict returns INT32_MAX on corruption (max loss).
// Any "converged" check fails loudly; any "is decreasing" trips
// immediately. Original mse_loss preserved for backward compat.
@pure
fn mse_loss_strict(y_start: i32, t_start: i32, n: i32) -> i32 {
    if n <= 0 { 2147483647 }
    else { if t1d_slice_ok(y_start, n) == 0 { 2147483647 }
    else { if t1d_slice_ok(t_start, n) == 0 { 2147483647 }
    else { mse_loss(y_start, t_start, n) } } }
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
// Restart 54 A6: i64 intermediate + INT32 saturation. Sibling of
// sgd_step_array (restart 53 A7); the scalar mirror was missed.
@pure
fn sgd_step_scalar(w: i32, g: i32, lr: i32) -> i32 {
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    let mut v: i64 = (w as i64) - (lr as i64) * (g as i64);
    if v > hi { v = hi; }
    else { if v < lo { v = lo; } };
    v as i32
}

// SGD update for a 1D parameter array in-place.
//   w[i] = w[i] - lr * grad[i] for i in [0, n)
// Returns 0.
// Restart 53 A7: per-element i64 intermediate + INT32 saturation.
// Sibling of ti1d_axpy saturation; a single `lr * gi` term can wrap
// i32 (e.g. lr=1, gi=INT32_MAX/2 with negative w), corrupting the
// weight under hostile or uninitialized grads.
fn sgd_step_array(w_start: i32, g_start: i32, lr: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(w_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(g_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    let lr64: i64 = lr as i64;
    while i < n {
        let w: i64 = __arena_get(w_start + i) as i64;
        let gi: i64 = __arena_get(g_start + i) as i64;
        let mut v: i64 = w - lr64 * gi;
        if v > hi { v = hi; }
        else { if v < lo { v = lo; } };
        __arena_set(w_start + i, v as i32);
        i = i + 1;
    }
    0
    }}}
}

// Linear-regression gradient w.r.t. weight w in y_pred = w*x + b:
//   loss = (w*x + b - target)^2
//   d_loss/d_w = 2 * (w*x + b - target) * x
// Useful for demo problems; real NN backprop computes per-layer
// gradients via reverse-mode AD (Phase 2.1 step 2).
// Restart 54 A6: i64 intermediates + INT32 saturation. Every product
// in the chain (`w*x`, `2*err`, `2*err*x`) is a wrap candidate.
@pure
fn lin_reg_grad_w(w: i32, b: i32, x: i32, target: i32) -> i32 {
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    let mut pred: i64 = (w as i64) * (x as i64) + (b as i64);
    if pred > hi { pred = hi; }
    else { if pred < lo { pred = lo; } };
    let mut err: i64 = pred - (target as i64);
    if err > hi { err = hi; }
    else { if err < lo { err = lo; } };
    let mut r: i64 = 2_i64 * err * (x as i64);
    if r > hi { r = hi; }
    else { if r < lo { r = lo; } };
    r as i32
}

// Linear-regression gradient w.r.t. bias b:
//   d_loss/d_b = 2 * (w*x + b - target)
// Restart 54 A6: i64 intermediates + INT32 saturation.
@pure
fn lin_reg_grad_b(w: i32, b: i32, x: i32, target: i32) -> i32 {
    let hi: i64 = 2147483647_i64;
    let lo: i64 = (0_i64 - 2147483647_i64) - 1_i64;
    let mut pred: i64 = (w as i64) * (x as i64) + (b as i64);
    if pred > hi { pred = hi; }
    else { if pred < lo { pred = lo; } };
    let mut err: i64 = pred - (target as i64);
    if err > hi { err = hi; }
    else { if err < lo { err = lo; } };
    let mut r: i64 = 2_i64 * err;
    if r > hi { r = hi; }
    else { if r < lo { r = lo; } };
    r as i32
}

// f32 SGD step over an array: w[i] = w[i] - lr * g[i].
// Restart 62 A1: per-element NaN-fail-closed. Sibling of restart 50 A2
// adam_f32_step. Pre-fix, a NaN gradient slot (or NaN lr) would
// overwrite the corresponding weight with NaN, propagating into every
// subsequent forward pass. Now we leave w[i] untouched if the new
// value is NaN — matching the adam fail-closed convention.
fn sgd_f32_step(w_start: i32, g_start: i32, lr: f32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(w_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(g_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let w_i = __f32_from_bits(__arena_get(w_start + i));
        let g_i = __f32_from_bits(__arena_get(g_start + i));
        let new_w = w_i - lr * g_i;
        if new_w == new_w {
            __arena_set(w_start + i, __bits_of_f32(new_w));
        };
        i = i + 1;
    }
    0
    }}}
}

// Clip an f32 gradient vector in place if its L2 norm is above max_norm.
// Returns 0; g_start is mutated only when clipping is needed.
fn clip_grad_norm_f32(g_start: i32, max_norm: f32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(g_start, n) == 0 { t2d_error() }
    else {
        let norm_sq = tf1d_l2_norm_sq(g_start, n);
        if (norm_sq <= 0.0_f32) || (norm_sq != norm_sq) { 0 }
        else {
            let norm = __sqrt(norm_sq);
            let target = if max_norm < 0.0_f32 { 0.0_f32 } else { max_norm };
            if norm > target {
                let scale = target / norm;
                tf1d_scale_inplace(g_start, n, scale);
            };
            0
        }
    }}
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
    if n <= 0 { 0 }
    else { if t1d_slice_ok(w_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(g_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(m_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(v_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let w_i = __f32_from_bits(__arena_get(w_start + i));
        let g_i = __f32_from_bits(__arena_get(g_start + i));
        let m_i = __f32_from_bits(__arena_get(m_start + i));
        let v_i = __f32_from_bits(__arena_get(v_start + i));
        let next_m = beta1 * m_i + (1.0_f32 - beta1) * g_i;
        // Restart 47 A1: clamp next_v to >= 0 before __sqrt. A caller can
        // pass a negative v_i (uninitialized arena or hostile input). With a
        // negative next_v, __sqrt returns 0 (transcendentals fallback), then
        // raw_denom = eps; if eps is tiny (~1e-8), w_i - lr*m/eps explodes.
        // Matches the layer_norm_f32 negative-eps clamp precedent from
        // restart 46.
        let raw_next_v = beta2 * v_i + (1.0_f32 - beta2) * g_i * g_i;
        let next_v = if raw_next_v < 0.0_f32 { 0.0_f32 } else { raw_next_v };
        __arena_set(m_start + i, __bits_of_f32(next_m));
        __arena_set(v_start + i, __bits_of_f32(next_v));
        let raw_denom = __sqrt(next_v) + eps;
        // Restart 50 A2: also fail-closed on NaN (raw_denom != raw_denom),
        // matching the softmax_layer + dense_classifier_sgd_step_f32
        // idiom from restart 48 A2. A NaN eps from the caller would
        // otherwise poison every weight w[i] in the batch with NaN.
        if (raw_denom <= 0.0_f32) || (raw_denom != raw_denom) {
            __arena_set(w_start + i, __bits_of_f32(w_i));
        }
        else {
            __arena_set(w_start + i,
                __bits_of_f32(w_i - lr * next_m / raw_denom));
        };
        i = i + 1;
    }
    0
    }}}}}
}

// MSE on f32 tensors.
// Restart 58 A6 (Increment 77 catch-up sweep): NaN-skip on per-element
// squared error. One bad slot no longer poisons the entire batch loss.
// Divisor stays at `n` (matches the tf1d_sum convention; per-batch
// scaling preferred over dynamic non-NaN count for simplicity).
@pure
fn mse_loss_f32(y_start: i32, t_start: i32, n: i32) -> f32 {
    if n <= 0 { 0.0_f32 }
    else { if t1d_slice_ok(y_start, n) == 0 { 0.0_f32 }
    else { if t1d_slice_ok(t_start, n) == 0 { 0.0_f32 }
    else {
        let mut i: i32 = 0;
        let mut total: f32 = 0.0_f32;
        while i < n {
            let yv = __f32_from_bits(__arena_get(y_start + i));
            let tv = __f32_from_bits(__arena_get(t_start + i));
            let d = yv - tv;
            let pe = d * d;
            if pe == pe { total = total + pe; };
            i = i + 1;
        }
        total / (n as f32)
    }}}
}

// Gradient of mean squared error with respect to y:
//   d/dy[i] mean((y - t)^2) = 2 * (y[i] - t[i]) / n
fn mse_loss_f32_grad(y_start: i32, t_start: i32,
                     dy_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(t_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(dy_start, n) == 0 { t2d_error() }
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
    }}}}
}

fn dense_layer_f32_forward(w_start: i32, w_rows: i32, w_cols: i32,
                           x_start: i32, b_start: i32, y_start: i32) -> i32 {
    if w_rows <= 0 { 0 }
    else { if w_cols <= 0 { 0 }
    else { if t2d_len(w_rows, w_cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(w_start, w_rows, w_cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(x_start, w_cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(b_start, w_rows) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, w_rows) == 0 { t2d_error() }
    else {
    tf2d_matvec(w_start, w_rows, w_cols, x_start, y_start);
    let mut r: i32 = 0;
    while r < w_rows {
        let cur = __f32_from_bits(__arena_get(y_start + r));
        let bv = __f32_from_bits(__arena_get(b_start + r));
        __arena_set(y_start + r, __bits_of_f32(cur + bv));
        r = r + 1;
    }
    0
    }}}}}}}
}

// Dense layer backward helpers for y = W @ x + b.
// grad_w[r, c] = grad_y[r] * x[c]
fn dense_layer_f32_grad_w(dy_start: i32, x_start: i32,
                          grad_w_start: i32, rows: i32, cols: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if t2d_len(rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(grad_w_start, rows, cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(dy_start, rows) == 0 { t2d_error() }
    else { if t1d_slice_ok(x_start, cols) == 0 { t2d_error() }
    else {
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
    }}}}}}
}

fn dense_layer_f32_grad_b(dy_start: i32, grad_b_start: i32, rows: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if t1d_slice_ok(dy_start, rows) == 0 { t2d_error() }
    else { if t1d_slice_ok(grad_b_start, rows) == 0 { t2d_error() }
    else {
    let mut r: i32 = 0;
    while r < rows {
        __arena_set(grad_b_start + r, __arena_get(dy_start + r));
        r = r + 1;
    }
    0
    }}}
}

// grad_x[c] = sum_r W[r, c] * grad_y[r]
fn dense_layer_f32_grad_x(w_start: i32, dy_start: i32,
                          grad_x_start: i32, rows: i32, cols: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if t2d_len(rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(w_start, rows, cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(dy_start, rows) == 0 { t2d_error() }
    else { if t1d_slice_ok(grad_x_start, cols) == 0 { t2d_error() }
    else {
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
    }}}}}}
}

// Leaky ReLU.
fn leaky_relu_layer(x_start: i32, alpha: f32, y_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        let v = if xi > 0.0_f32 { xi } else { alpha * xi };
        __arena_set(y_start + i, __bits_of_f32(v));
        i = i + 1;
    }
    0
    }}}
}

// Momentum SGD.
// Restart 62 A2: per-element NaN-fail-closed. Sibling of restart 50 A2
// adam_f32_step and restart 62 A1 sgd_f32_step. Pre-fix, a NaN gradient
// poisoned both the velocity buffer and the weights silently — and
// because velocity carries forward across steps, a single NaN gradient
// permanently corrupted the optimizer state. Now we leave v[i] and w[i]
// untouched if either new value is NaN, matching the adam fail-closed
// convention.
fn momentum_step(w_start: i32, v_start: i32, g_start: i32,
                 beta: f32, lr: f32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(w_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(v_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(g_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let w_i = __f32_from_bits(__arena_get(w_start + i));
        let v_i = __f32_from_bits(__arena_get(v_start + i));
        let g_i = __f32_from_bits(__arena_get(g_start + i));
        let new_v = beta * v_i + g_i;
        let new_w = w_i - lr * new_v;
        if (new_v == new_v) && (new_w == new_w) {
            __arena_set(v_start + i, __bits_of_f32(new_v));
            __arena_set(w_start + i, __bits_of_f32(new_w));
        };
        i = i + 1;
    }
    0
    }}}}
}

// tanh layer (delegates to scalar __tanh).
// Restart 48 A3: delegate to __tanh (transcendentals.hx) instead of
// inlining __exp(2*xi). __tanh short-circuits at |x| > 20 to avoid
// saturating __exp's range, matching the precedent set by sigmoid_layer
// (delegates to __sigmoid), softplus_layer (delegates to __softplus),
// silu_layer/gelu_layer (delegate via __sigmoid/__tanh). The inline form
// could NaN at the saturation boundary.
fn tanh_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        let v = __tanh(xi);
        __arena_set(y_start + i, __bits_of_f32(v));
        i = i + 1;
    }
    0
    }}}
}

// sigmoid layer (uses __sigmoid).
fn sigmoid_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        let v = __sigmoid(xi);
        __arena_set(y_start + i, __bits_of_f32(v));
        i = i + 1;
    }
    0
    }}}
}

fn softplus_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(y_start + i, __bits_of_f32(__softplus(xi)));
        i = i + 1;
    }
    0
    }}}
}

fn silu_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(y_start + i, __bits_of_f32(__silu(xi)));
        i = i + 1;
    }
    0
    }}}
}

fn gelu_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let xi = __f32_from_bits(__arena_get(x_start + i));
        __arena_set(y_start + i, __bits_of_f32(__gelu(xi)));
        i = i + 1;
    }
    0
    }}}
}

fn relu_layer_f32_backward(x_start: i32, dy_start: i32,
                           dx_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(dy_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(dx_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let x = __f32_from_bits(__arena_get(x_start + i));
        let dy = __f32_from_bits(__arena_get(dy_start + i));
        let dx = if x > 0.0_f32 { dy } else { 0.0_f32 };
        __arena_set(dx_start + i, __bits_of_f32(dx));
        i = i + 1;
    }
    0
    }}}}
}

fn sigmoid_layer_backward(y_start: i32, dy_start: i32,
                          dx_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(dy_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(dx_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let y = __f32_from_bits(__arena_get(y_start + i));
        let dy = __f32_from_bits(__arena_get(dy_start + i));
        __arena_set(dx_start + i, __bits_of_f32(dy * y * (1.0_f32 - y)));
        i = i + 1;
    }
    0
    }}}}
}

fn tanh_layer_backward(y_start: i32, dy_start: i32,
                       dx_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(dy_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(dx_start, n) == 0 { t2d_error() }
    else {
    let mut i: i32 = 0;
    while i < n {
        let y = __f32_from_bits(__arena_get(y_start + i));
        let dy = __f32_from_bits(__arena_get(dy_start + i));
        __arena_set(dx_start + i, __bits_of_f32(dy * (1.0_f32 - y * y)));
        i = i + 1;
    }
    0
    }}}}
}

// Layer normalization over one f32 vector.
// y[i] = (x[i] - mean(x)) / sqrt(variance(x) + eps)
fn layer_norm_f32(x_start: i32, y_start: i32, n: i32, eps: f32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
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
        let safe_eps = if eps < 0.0_f32 { 0.0_f32 } else { eps };
        let denom = __sqrt(var + safe_eps);
        // Restart 47 A3: when both var == 0 (constant input) and safe_eps
        // == 0, denom == 0, then 1.0 / 0 = +Inf and (xj - mean) * Inf =
        // 0 * Inf = NaN. Fail-closed: write 0 to every output slot, which
        // is the mathematically correct centered output for a constant
        // input (x - mean = 0 for all j).
        // Restart 50 A2: also fail-closed on NaN (denom != denom). A NaN
        // input (eps or any x[i]) would otherwise poison every output
        // y[j] with NaN. Matches the softmax_layer + adam_f32_step NaN
        // discipline.
        if (denom <= 0.0_f32) || (denom != denom) {
            let mut j: i32 = 0;
            while j < n {
                __arena_set(y_start + j, __bits_of_f32(0.0_f32));
                j = j + 1;
            }
        }
        else {
            let inv_std = 1.0_f32 / denom;
            let mut j: i32 = 0;
            while j < n {
                let xj = __f32_from_bits(__arena_get(x_start + j));
                __arena_set(y_start + j, __bits_of_f32((xj - mean) * inv_std));
                j = j + 1;
            }
        };
    0
    }}}
}

// Inverted dropout for f32 vectors. During training, each element is kept with
// probability keep_prob and scaled by 1/keep_prob; dropped elements become 0.
// Returns the final deterministic RNG state.
fn dropout_f32(x_start: i32, y_start: i32, n: i32,
               keep_prob: f32, seed: i32) -> i32 {
    if n <= 0 { seed }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
    else { if keep_prob <= 0.0_f32 {
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
    }}}
}

// Softmax (max-subtract, uses __exp + tf1d_max).
fn softmax_layer(x_start: i32, y_start: i32, n: i32) -> i32 {
    if n < 0 { 35001 }
    else { if n == 0 { 0 }
    else { if t1d_slice_ok(x_start, n) == 0 { t2d_error() }
    else { if t1d_slice_ok(y_start, n) == 0 { t2d_error() }
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
        // Restart 48 A2: fail-closed when sum_e <= 0 or NaN. sum_e can be 0
        // if every __exp(xi - max_v) underflowed to 0 (extreme negative
        // inputs), and can be NaN if any xi was NaN (poisoned earlier layer
        // output). Either way, dividing by sum_e produces +Inf or NaN that
        // poisons every downstream consumer. Fail-closed writes the
        // maximum-entropy distribution (1/n to every slot), matching the
        // restart-47 layer_norm_f32 precedent (write zeros there, write the
        // canonical "no information" distribution here).
        let inv_n = if n > 0 { 1.0_f32 / (n as f32) } else { 0.0_f32 };
        if (sum_e <= 0.0_f32) || (sum_e != sum_e) {
            let mut k: i32 = 0;
            while k < n {
                __arena_set(y_start + k, __bits_of_f32(inv_n));
                k = k + 1;
            }
        }
        else {
            let mut j: i32 = 0;
            while j < n {
                let e = __f32_from_bits(__arena_get(y_start + j));
                __arena_set(y_start + j, __bits_of_f32(e / sum_e));
                j = j + 1;
            }
        };
        0
    }}}}
}

fn softmax_rows_f32(logits_start: i32, probs_start: i32,
                    rows: i32, cols: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if t2d_len(rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(logits_start, rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(probs_start, rows, cols) == 0 { t2d_error() }
    else {
        let mut r: i32 = 0;
        while r < rows {
            let row_in = logits_start + r * cols;
            let row_out = probs_start + r * cols;
            softmax_layer(row_in, row_out, cols);
            r = r + 1;
        }
        0
    }}}}}
}

fn softmax_ce_grad_f32(probs_start: i32, target_start: i32,
                       grad_start: i32, rows: i32, cols: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if t2d_len(rows, cols) == 0 { 35001 }
    else { if t2d_shape_ok(probs_start, rows, cols) == 0 { 35001 }
    else { if t2d_shape_ok(grad_start, rows, cols) == 0 { 35001 }
    else { if t1d_slice_ok(target_start, rows) == 0 { 35001 }
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
    }}}}}}
}

fn dense_classifier_sgd_step_f32(w_start: i32, b_start: i32, x_start: i32,
                                 target: i32, scratch_start: i32,
                                 shape_start: i32, lr: f32) -> i32 {
    if t1d_slice_ok(shape_start, 2) == 0 { 35001 }
    else {
    let classes = __arena_get(shape_start);
    let in_dim = __arena_get(shape_start + 1);
    if classes <= 0 { 35001 }
    else { if in_dim <= 0 { 35001 }
    else { if t2d_len(classes, in_dim) == 0 { 35001 }
    else { if t2d_shape_ok(w_start, classes, in_dim) == 0 { 35001 }
    else { if t1d_slice_ok(b_start, classes) == 0 { 35001 }
    else { if t1d_slice_ok(x_start, in_dim) == 0 { 35001 }
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
            // Restart 48 A2 (sibling): fail-closed no-op step when sum_e is
            // degenerate. Same precedent as softmax_layer's sum_e guard:
            // if every __exp(score - max) underflowed to 0 (sum_e == 0) or
            // any score is NaN (sum_e == NaN), the probabilities are
            // poisoned. A no-op step leaves weights untouched rather than
            // applying a divide-by-zero gradient that corrupts them.
            if (sum_e <= 0.0_f32) || (sum_e != sum_e) { 0 }
            else {
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
            }
        }}
    }}}}}}
    }
}

// BCE.
@pure
fn bce_loss_scalar(p: f32, t: f32) -> f32 {
    __bce(p, t)
}

// Cross-entropy.
@pure
fn ce_loss(p_start: i32, target_idx: i32, cols: i32) -> f32 {
    if target_idx < 0 { 1000000.0_f32 }
    else { if target_idx >= cols { 1000000.0_f32 }
    else { if t1d_slice_ok(p_start, cols) == 0 { 1000000.0_f32 }
    else {
        let p = __f32_from_bits(__arena_get(p_start + target_idx));
        let eps = 0.0001_f32;
        let pc = __max(__min(p, 1.0_f32 - eps), eps);
        0.0_f32 - __log_stable(pc)
    }}}
}

// For a row-major logits matrix (rows x cols), write each row's argmax class
// index to out_start[row].
// Restart 58 A7 (Increment 77 catch-up sweep): NaN-at-col-0 robustness.
// Same idiom as tf1d_argmax — adopt the first non-NaN slot.
fn argmax_rows_f32(logits_start: i32, rows: i32, cols: i32,
                   out_start: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if t2d_len(rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(logits_start, rows, cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(out_start, rows) == 0 { t2d_error() }
    else {
        let mut r: i32 = 0;
        while r < rows {
            let row_start = logits_start + r * cols;
            let mut best_idx: i32 = 0;
            let mut best_val: f32 = 0.0_f32;
            let mut seen: i32 = 0;
            let mut c: i32 = 0;
            while c < cols {
                let v = __f32_from_bits(__arena_get(row_start + c));
                if v == v {
                    if seen == 0 {
                        best_val = v;
                        best_idx = c;
                        seen = 1;
                    }
                    else { if v > best_val {
                        best_val = v;
                        best_idx = c;
                    }; };
                };
                c = c + 1;
            }
            __arena_set(out_start + r, best_idx);
            r = r + 1;
        }
        0
    }}}}}
}

// Restart 61 A2: NaN-at-col-0 robustness. Same idiom as tf1d_argmax /
// argmax_rows_f32 — adopt the first non-NaN slot with the `seen = 0`
// sentinel. Without this, a row whose col 0 is NaN would freeze
// best_val = NaN (per IEEE-754 `v > NaN` is always false), causing the
// row's argmax to silently stay at index 0 regardless of any later
// numeric maxima.
@pure
fn accuracy_count_from_logits_f32(logits_start: i32, target_start: i32,
                                  rows: i32, cols: i32) -> i32 {
    if rows <= 0 { 0 }
    else { if cols <= 0 { 0 }
    else { if t2d_len(rows, cols) == 0 { t2d_error() }
    else { if t2d_shape_ok(logits_start, rows, cols) == 0 { t2d_error() }
    else { if t1d_slice_ok(target_start, rows) == 0 { t2d_error() }
    else {
        let mut r: i32 = 0;
        let mut hits: i32 = 0;
        while r < rows {
            let row_start = logits_start + r * cols;
            let mut best_idx: i32 = 0;
            let mut best_val: f32 = 0.0_f32;
            let mut seen: i32 = 0;
            let mut c: i32 = 0;
            while c < cols {
                let v = __f32_from_bits(__arena_get(row_start + c));
                if v == v {
                    if seen == 0 {
                        best_val = v;
                        best_idx = c;
                        seen = 1;
                    }
                    else { if v > best_val {
                        best_val = v;
                        best_idx = c;
                    }; };
                };
                c = c + 1;
            }
            if best_idx == __arena_get(target_start + r) {
                hits = hits + 1;
            };
            r = r + 1;
        }
        hits
    }}}}}
}

@pure
fn ce_loss_batch_f32(probs_start: i32, target_start: i32,
                     rows: i32, cols: i32) -> f32 {
    if rows <= 0 { 0.0_f32 }
    else { if cols <= 0 { 0.0_f32 }
    else { if t2d_len(rows, cols) == 0 { 1000000.0_f32 }
    else { if t2d_shape_ok(probs_start, rows, cols) == 0 { 1000000.0_f32 }
    else { if t1d_slice_ok(target_start, rows) == 0 { 1000000.0_f32 }
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
                    total = total + ce_loss(row_start, target, cols);
                };
            };
            r = r + 1;
        }
        if invalid == 1 { 1000000.0_f32 }
        else { total / (rows as f32) }
    }}}}}
}

// argmin: index of smallest element. Companion to argmax.
// Returns -1 on empty.
@pure
fn argmin(x_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 - 1 }
    else { if t1d_slice_ok(x_start, n) == 0 { 0 - 1 }
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
    }}
}

// MAE (sum of absolute differences) on integer tensors.
// Sibling of mse_loss; cheaper since no multiplication and no overflow risk.
@pure
// Restart 51 A6: i64 accumulator + INT32 saturation. Both the inner
// `y[i] - t[i]` subtraction (i64-promoted to handle the boundary case
// y=INT32_MAX, t=-1) and the outer sum-of-abs are protected. Matches
// the mse_loss / ti1d_sum / ti1d_dot precedent.
fn mae_loss(y_start: i32, t_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(y_start, n) == 0 { 0 }
    else { if t1d_slice_ok(t_start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut total: i64 = 0_i64;
    let hi: i64 = 2147483647_i64;
    while i < n {
        let d: i64 = (__arena_get(y_start + i) as i64) - (__arena_get(t_start + i) as i64);
        let ad: i64 = if d < 0_i64 { 0_i64 - d } else { d };
        total = total + ad;
        if total > hi { total = hi; };
        i = i + 1;
    }
    total as i32
    }}}
}

// MAE on f32 tensors (mean absolute error, returns 0.0 on empty).
// Restart 58 A6 (Increment 77 catch-up sweep): NaN-skip on per-element
// abs error. Same divisor convention as mse_loss_f32.
@pure
fn mae_loss_f32(y_start: i32, t_start: i32, n: i32) -> f32 {
    if n <= 0 { 0.0_f32 }
    else { if t1d_slice_ok(y_start, n) == 0 { 0.0_f32 }
    else { if t1d_slice_ok(t_start, n) == 0 { 0.0_f32 }
    else {
        let mut i: i32 = 0;
        let mut total: f32 = 0.0_f32;
        while i < n {
            let yv = __f32_from_bits(__arena_get(y_start + i));
            let tv = __f32_from_bits(__arena_get(t_start + i));
            let ae = __abs(yv - tv);
            if ae == ae { total = total + ae; };
            i = i + 1;
        }
        total / (n as f32)
    }}}
}

// Count of positions where prediction matches target. Useful for batch
// classification accuracy: pred[i] is typically argmax of model output,
// target[i] is the integer class label.
@pure
fn count_correct(pred_start: i32, target_start: i32, n: i32) -> i32 {
    if n <= 0 { 0 }
    else { if t1d_slice_ok(pred_start, n) == 0 { 0 }
    else { if t1d_slice_ok(target_start, n) == 0 { 0 }
    else {
    let mut i: i32 = 0;
    let mut hits: i32 = 0;
    while i < n {
        let p = __arena_get(pred_start + i);
        let t = __arena_get(target_start + i);
        if p == t { hits = hits + 1; }
        i = i + 1;
    }
    hits
    }}}
}
