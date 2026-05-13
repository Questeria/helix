// dogfood_01_one_param.hx — first dogfooding target.
//
// Learn a single parameter w that minimizes loss(w) = (w - 7)^2 by running
// actual gradient descent inside a Helix-emitted binary. Each step:
//   1. read current w from reflection cell via splice
//   2. compute g = grad_rev(loss)(w)
//   3. propose w_new = w - lr * g
//   4. submit to verifier-gated modify
// After enough steps the cell should hold ~7. Final exit code = round(w) + 35.
//
// This is the FIRST self-modifying program in Helix that uses live gradient
// information at runtime to decide its next state. Not just a static
// sequence of `modify(...)` calls.

@pure
fn loss(w: f32) -> f32 {
    let d = w - 7.0;
    d * d
}

@verifier
fn safe_step(handle: i32, new_val: i32) -> i32 {
    if new_val < 0 { 0 }
    else { if new_val > 20 { 0 } else { 1 } }
}

fn main() -> i32 {
    let cell = quote(0);
    let mut w_int: i32 = 0;
    let lr = 0.5;          // step size — lr=1 oscillates, 0.5 converges in 1 step

    // Five steps of gradient descent. Each step:
    //   - read current w from cell
    //   - compute grad of loss at w
    //   - step: w_new = w - lr * grad
    //   - commit via verifier-gated modify
    let mut step: i32 = 0;
    while step < 5 {
        let w_now = splice(cell);
        let g = grad_rev(loss)(w_now as f32);
        let w_new_f = (w_now as f32) - lr * g;
        let w_new_i = w_new_f as i32;
        modify(cell, w_new_i, safe_step);
        step = step + 1;
    }

    let final_w = splice(cell);
    final_w + 35
}
