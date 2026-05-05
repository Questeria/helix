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
