// gradient_descent.hx — one step of gradient descent in Helix.
//
// We minimize loss(x) = (x - 3)^2  via gradient descent.
// d(loss)/dx = 2*(x - 3)
//
// Starting at x = 0, with learning rate 0.5:
//   gradient = 2*(0 - 3) = -6
//   step     = -0.5 * (-6) = 3
//   x_new    = 0 + 3 = 3   (= optimum, since loss is minimized at x=3)
//
// We verify: x_new should be 3. Add 39 for exit code 42.

fn loss(x: f32) -> f32 {
    let diff = x - 3.0;
    diff * diff
}

fn main() -> i32 {
    let x = 0.0;
    let lr = 0.5;
    // grad(loss)(x) = d(loss)/dx evaluated at x
    let g = grad(loss)(x);
    let step = lr * g;
    let x_new = x - step;
    // x_new should be 3.0
    (x_new + 39.0) as i32
}
