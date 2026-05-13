// Helix differentiating Helix: scalar regression learned by
// gradient descent. The loss function is written in Helix; Helix's
// own reverse-mode autodiff machinery (declared in
// helixc/frontend/autodiff_reverse.py) is invoked at compile time
// via the `grad_rev(f)` AST transform, which generates a fresh
// Helix function `f__rgrad` that computes dL/dw symbolically and
// then lowers to runtime x86-64. The gradient is then called inside
// a while-loop to drive a vanilla SGD update.
//
// Loss (single-parameter linear regression):
//   loss(w) = (w*4 - 12)^2     // target slope is 3 (so 3*4=12)
//
// Starting at w=0, gradient descent should drive w toward 3.
//
// `grad_rev(loss)(w)` is the magic line: at compile time the
// reverse-mode AD pass walks the AST of `loss`, builds the
// symbolic derivative as a NEW AST, registers it as `loss__rgrad`,
// then rewrites the call site. Most languages can't introspect
// their own derivative trees as runtime values; Helix can.
//
// We print w * 100 at the end so the exit code reflects how close
// we got to the optimum (target 300 -> exit byte 44 after mod 256).
//
// Exercises: @pure, grad_rev builtin transform, mutable lets,
// while loops, f32 arithmetic, AST-introspection of differentiation.

@pure
fn loss(w: f32) -> f32 {
    let d = w * 4.0 - 12.0;
    d * d
}

fn main() -> i32 {
    let mut w: f32 = 0.0;
    let lr: f32 = 0.05;
    let mut step: i32 = 0;
    while step < 50 {
        // Compile-time: grad_rev(loss) generates loss__rgrad. Runtime:
        // it's just a normal function call returning the gradient.
        let g = grad_rev(loss)(w);
        w = w - lr * g;
        step = step + 1;
    }
    // After 50 steps with lr=0.05, w should be very close to 3.
    // Return w * 100 as i32; with the loss surface and step size
    // chosen, w lands at exactly 3.0, so the exit byte is 300 mod 256 = 44.
    (w * 100.0) as i32
}
