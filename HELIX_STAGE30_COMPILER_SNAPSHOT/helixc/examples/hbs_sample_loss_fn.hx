// hbs_sample_loss_fn.hx
//
// HBS dogfood: small float-AD pipeline. Demonstrates the stdlib
// transcendentals path + __powi exponent + manual gradient descent
// step on a 1-D loss surface.
//
// Loss(w) = (w - 3.0)^2 + sigmoid(w) * 0.1
//
// Closed-form derivative: 2*(w - 3.0) + sigmoid(w) * (1 - sigmoid(w)) * 0.1
//
// We don't invoke `grad(loss)` here (that goes through grad_pass and
// ends up in a separate fn) — instead this file just showcases that
// stdlib transcendentals are reachable from HBS-style code, and the
// returned value is encodable as an i32 exit code.

@pure @total
fn loss(w: f32) -> f32 {
    let diff = w - 3.0;
    let sq = __powi(diff, 2);
    let sig = __sigmoid(w);
    sq + sig * 0.1
}

@pure @total
fn loss_grad_manual(w: f32) -> f32 {
    // Closed-form gradient written by hand.
    let diff = w - 3.0;
    let lin = 2.0 * diff;
    let sig = __sigmoid(w);
    let one_minus_sig = 1.0 - sig;
    lin + sig * one_minus_sig * 0.1
}

@pure @total
fn step(w: f32, lr: f32) -> f32 {
    // Standard SGD update: w' = w - lr * grad(w)
    w - lr * loss_grad_manual(w)
}

fn main() -> i32 {
    // Run 5 SGD iterations from w = 0.0 with lr = 0.1.
    // Loss minimum is around w ≈ 3.0; we expect convergence.
    let w0: f32 = 0.0;
    let w1 = step(w0, 0.1);
    let w2 = step(w1, 0.1);
    let w3 = step(w2, 0.1);
    let w4 = step(w3, 0.1);
    let w5 = step(w4, 0.1);

    // Convert final w to an i32 exit code. We multiply by 10 and take
    // floor to get a small bucket (0..30). Should be ~3 (≈ optimum).
    let scaled = w5 * 10.0;
    let i = scaled as i32;
    i
}
