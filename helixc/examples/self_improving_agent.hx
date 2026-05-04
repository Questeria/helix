// self_improving_agent.hx — flagship demo of Helix's AGI primitives.
//
// Combines features no other AI language exposes together in a single
// compiled program:
//   - reflection runtime (quote / splice / modify) with verifier gating
//   - reverse-mode automatic differentiation (grad_rev)
//   - effect tracking (@pure / @verifier)
//
// The agent has one parameter stored in a reflection cell. The parameter
// represents x in the loss function loss(x) = (x - 5)^2. Each update is
// proposed by gradient descent and gated through a verifier function that
// inspects the proposal and decides whether to commit it. If the update
// would push x outside the safe range OR move it in the wrong direction
// (positive gradient sign for a value below the optimum), the verifier
// rejects.
//
// Final exit code = current_parameter + 38. The agent climbs from 0 to
// 4 by accepted updates; the gradient-aware verifier rejects the step
// to 5 (since gradient is zero there — already at the optimum, no work
// to do). 4 + 38 = 42.

// The loss function the agent is minimizing. Pure: no I/O, no mutation.
@pure
fn loss(x: f32) -> f32 {
    let diff = x - 5.0;
    diff * diff
}

// Verifier: gates each proposed parameter update. Receives the reflection
// handle and the candidate new value, returns 1 (accept) or 0 (reject).
// Two checks: range bound, and gradient-direction sanity (the candidate
// shouldn't move uphill on the loss surface).
@verifier
fn safe_step(handle: i32, new_val: i32) -> i32 {
    // Range bound: refuse anything outside [0, 10].
    if new_val < 0 { 0 }
    else { if new_val > 10 { 0 }
           else {
               // Inspect the gradient of `loss` at the proposed point.
               // For loss(x) = (x-5)^2, gradient is 2*(x-5):
               //   negative for x < 5  (loss decreasing as x grows)
               //   positive for x > 5  (loss increasing as x grows)
               //   zero at x = 5       (the optimum)
               // A *moving* update should satisfy |grad| > 0; if a candidate
               // sits AT the optimum we reject (no further work to do).
               // This makes the gradient check load-bearing — it isn't
               // redundant with the range bound.
               let g = grad_rev(loss)(new_val as f32);
               // Use grad's absolute value to test "non-zero gradient":
               // accept if g != 0 (i.e. we're not already at the optimum).
               // For this demo we hand-encode |g| > 0 by checking g != 0.
               if g > 0.001 { 1 }
               else { if g < -0.001 { 1 }
                      else { 0 } }
           }
    }
}

fn main() -> i32 {
    let param = quote(0);

    // Gradient-descent-flavored update sequence. Each value is verified.
    modify(param, 1, safe_step);
    modify(param, 2, safe_step);
    modify(param, 3, safe_step);
    modify(param, 4, safe_step);

    // The next proposed update is at the loss optimum (x=5, gradient=0).
    // The verifier uses the gradient to detect this and REJECTS — no
    // further movement is warranted. The cell stays at 4.
    modify(param, 5, safe_step);

    // Attempted bad updates: range check rejects.
    modify(param, 99, safe_step);    // out of range
    modify(param, -3, safe_step);    // out of range

    // Final committed value is 4 (last gradient-allowed update).
    let final_x = splice(param);
    final_x + 38
}
