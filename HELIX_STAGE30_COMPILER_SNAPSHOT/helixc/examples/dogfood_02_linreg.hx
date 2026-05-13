// dogfood_02_linreg.hx — linear regression with multiple training points.
//
// Learn slope w that fits y = w*x given 4 training pairs:
//   (1, 3), (2, 6), (3, 9), (4, 12)   (true w = 3)
//
// For a single example (x_i, y_i):
//   per_example_loss(w) = (w*x_i - y_i)^2
//   d/dw = 2*(w*x_i - y_i)*x_i
//
// Total loss is the sum over all examples; we'd ideally take gradient of
// the sum. Since Helix can't yet differentiate over arrays at compile
// time, we hand-write 4 per-example loss functions and add their gradients.
// (When tensor-AD lands later, this whole dogfood collapses to one line.)
//
// Verifier: bound the proposed weight to a sane range.
// Final exit code = round(w) + 39, expected w → 3 → exit 42.

@pure fn loss_pt1(w: f32) -> f32 { let d = w * 1.0 - 3.0;  d * d }
@pure fn loss_pt2(w: f32) -> f32 { let d = w * 2.0 - 6.0;  d * d }
@pure fn loss_pt3(w: f32) -> f32 { let d = w * 3.0 - 9.0;  d * d }
@pure fn loss_pt4(w: f32) -> f32 { let d = w * 4.0 - 12.0; d * d }

@verifier
fn safe_w(handle: i32, new_val: i32) -> i32 {
    if new_val < -10 { 0 }
    else { if new_val > 10 { 0 } else { 1 } }
}

fn main() -> i32 {
    let cell = quote(0);
    // Run several gradient-descent steps. Per step:
    //   total_grad = sum of grad of each per-example loss
    //   w_new = w - lr * total_grad / N
    // Total gradient at w=0 is -180; with lr=0.02 the first step lands
    // exactly at w=3 (the optimum), where all per-example gradients are
    // zero and subsequent steps are no-ops.
    let lr = 0.02;

    let mut step: i32 = 0;
    while step < 6 {
        let w = splice(cell) as f32;
        let g1 = grad_rev(loss_pt1)(w);
        let g2 = grad_rev(loss_pt2)(w);
        let g3 = grad_rev(loss_pt3)(w);
        let g4 = grad_rev(loss_pt4)(w);
        let total_g = g1 + g2 + g3 + g4;
        let w_new = w - lr * total_g;
        modify(cell, w_new as i32, safe_w);
        step = step + 1;
    }

    let final_w = splice(cell);
    final_w + 39
}
