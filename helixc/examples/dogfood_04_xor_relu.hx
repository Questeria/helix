// dogfood_04_xor_relu.hx — 2-layer ReLU net touching XOR.
//
// 2-input → 2-hidden → 1-output, no biases (6 weights, fits in 6 xmm regs).
//   h0 = relu(w0*x0 + w1*x1)
//   h1 = relu(w2*x0 + w3*x1)
//   y  = w4*h0 + w5*h1
//
// XOR is famously not solvable by a network without a nonlinear activation.
// This demo demonstrates that grad_rev composes through the language's
// stdlib __relu function — i.e. the analytic chain rule wired in
// autodiff_reverse.py fires correctly. We don't claim XOR is "solved"; we
// claim the program runs end-to-end with non-zero gradient through ReLU.
//
// Final exit code 42 if both forward and gradient evaluate as expected.

// loss_01: forward pass + squared error for input (0, 1) target 1.
// Inlined (no separate forward fn) because reverse-mode AD doesn't yet
// chain-rule across user-defined function calls — only across the
// stdlib transcendentals. Inlining keeps everything visible to grad_rev.
@pure fn loss_01(w0: f32, w1: f32, w2: f32, w3: f32,
                  w4: f32, w5: f32) -> f32 {
    let h0 = __relu(w0 * 0.0 + w1 * 1.0);
    let h1 = __relu(w2 * 0.0 + w3 * 1.0);
    let p = w4 * h0 + w5 * h1;
    let d = p - 1.0;
    d * d
}

@verifier
fn safe_w(handle: i32, val: f32) -> i32 {
    if val < -10.0 { 0 }
    else { if val > 10.0 { 0 } else { 1 } }
}

fn main() -> i32 {
    let cw0 = quote(0); let cw1 = quote(1); let cw2 = quote(2);
    let cw3 = quote(3); let cw4 = quote(4); let cw5 = quote(5);

    // Init: w0=1, w1=1 (h0 detects x0+x1>0), w2=1, w3=-1 (h1 detects x0>x1),
    // w4=1, w5=-2 (output combines them).
    // Non-zero, non-XOR-perfect initial weights so loss > 0.
    modify_f(cw0, 0.5, safe_w);
    modify_f(cw1, 0.3, safe_w);
    modify_f(cw2, 0.4, safe_w);
    modify_f(cw3, 0.0 - 0.2, safe_w);
    modify_f(cw4, 0.5, safe_w);
    modify_f(cw5, 0.6, safe_w);

    let w0 = splice_f(cw0); let w1 = splice_f(cw1); let w2 = splice_f(cw2);
    let w3 = splice_f(cw3); let w4 = splice_f(cw4); let w5 = splice_f(cw5);

    // Loss at the current weights for input (0, 1).
    let l = loss_01(w0, w1, w2, w3, w4, w5);
    // Synthesise a "prediction" too — same forward pass, just outside loss.
    let h0 = __relu(w0 * 0.0 + w1 * 1.0);
    let h1 = __relu(w2 * 0.0 + w3 * 1.0);
    let pred = w4 * h0 + w5 * h1;
    // Gradient w.r.t. w1 (input (0,1) ⇒ x1=1, so changes to w1 propagate
    // through to the loss; using w0 would give zero because x0=0 zeroes
    // its contribution).
    let g = grad_rev(loss_01, 1)(w0, w1, w2, w3, w4, w5);

    // Confirm the program ran end-to-end:
    //   - pred is finite (not 0 due to a crash)
    //   - l > 0 (we are not at the optimum)
    //   - gradient is non-zero (chain rule went through ReLU)
    let pred_nonzero = if pred > 0.0 { 1 } else { if pred < 0.0 { 1 } else { 0 } };
    let l_positive = if l > 0.0 { 1 } else { 0 };
    let g_nonzero = if g > 0.0 { 1 } else { if g < 0.0 { 1 } else { 0 } };

    pred_nonzero + l_positive + g_nonzero + 39
}
