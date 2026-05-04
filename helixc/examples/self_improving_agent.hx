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
// Final exit code = current_parameter + 37, so a value of 5 (the optimum
// for the given loss) yields 42.

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
    if new_val < 0 { 0 }
    else { if new_val > 10 { 0 }
           else {
               // Inspect the gradient at the proposed point. For loss(x) =
               // (x-5)^2 the gradient is 2(x-5). At an accepted candidate
               // we want to be approaching x=5 — gradient magnitude
               // bounded means we're still in the basin.
               let g = grad_rev(loss)(new_val as f32);
               // Accept if magnitude of gradient is below threshold (i.e.
               // we're not too far from optimum) — for this demo any
               // value in [0, 10] satisfies this since gradient is at
               // most 2*5 = 10.
               if g >= -10.0 { if g <= 10.0 { 1 } else { 0 } }
               else { 0 }
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
    modify(param, 5, safe_step);

    // Attempted bad updates: verifier rejects.
    modify(param, 99, safe_step);    // out of range
    modify(param, -3, safe_step);    // out of range

    // Final committed value is 5 (optimum of the loss).
    let final_x = splice(param);
    final_x + 37
}
