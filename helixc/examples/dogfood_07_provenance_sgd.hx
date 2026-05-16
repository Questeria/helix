// dogfood_07_provenance_sgd.hx — Stage 36 Increment 7 dogfood.
//
// First Helix program that LEARNS A FUZZY-LOGIC RULE VIA SGD using
// gradients that flow through propositional logic operators. This
// is the end-to-end strategic demonstration of the Stage 36 Tier 3
// #10 differentiator: provenance-typed values + propositional
// algebra + autodiff, all composable in the same source.
//
// The task: learn a weight `w` (a Logic<f32> truth value) such that
//
//     fuzzy_and(prove(0.5, src=100), prove(w, src=200)) == 0.4
//
// where src=100 represents an input-feature source ("the fact has
// truth 0.5") and src=200 represents the learned-weight source.
// Because fuzzy_and uses product semantics, the closed-form
// solution is w = 0.8 (since 0.5 * 0.8 = 0.4).
//
// Algorithm:
//   loss(w) = (fuzzy_and(prove(0.5, 100), prove(w, 200)) - 0.4)^2
//   d/dw   = 2 * (0.5w - 0.4) * 0.5 = 0.5w - 0.4  [via the
//             product chain rule shipped in Increment 6]
//   step:   w := w - lr * grad_rev(loss)(w)
//
// With lr = 2.0 the step exactly cancels the gradient: w_1 = 0 + 2*0.4
// = 0.8 in a single iteration. Running 30 iterations confirms
// convergence is stable, not just lucky on step 1.
//
// What this dogfood proves end-to-end:
//   - prove() / unwrap_logic() compose into a real Helix expression
//   - fuzzy_and lowers to MUL and is differentiable
//   - grad_rev() flows gradients through the provenance wrappers
//   - mutable let + while loop drive an SGD inner loop
//   - the entire pipeline (parse -> typecheck -> grad_pass ->
//     lower -> opt -> codegen -> ELF -> WSL run) produces an exit
//     code that reflects a *learned* parameter value.
//
// Exit code 42 iff w converged to ~0.8 (after rounding w*100 to i32
// and subtracting 38). Anything else means the SGD didn't converge,
// the chain rule is wrong, or the provenance wrappers blocked
// gradient flow.

fn loss(w: f32) -> f32 {
    // Provenance-typed inputs: the constant 0.5 from "input feature
    // source 100", the learnable w from "weight source 200".
    let a: Logic<f32> = prove(0.5_f32, 100);
    let b: Logic<f32> = prove(w, 200);
    // Predicted value: product-semantics AND of the two truth values.
    let pred: f32 = unwrap_logic(fuzzy_and(a, b));
    // Target: 0.4. Squared error loss.
    let target: f32 = 0.4_f32;
    let diff: f32 = pred - target;
    diff * diff
}

fn main() -> i32 {
    let mut w: f32 = 0.0_f32;
    let lr: f32 = 2.0_f32;
    let mut i: i32 = 0;
    while i < 30 {
        let g: f32 = grad_rev(loss)(w);
        w = w - lr * g;
        i = i + 1;
    }
    // Round-to-nearest by adding 0.5 before truncating.
    let w_rounded: i32 = ((w * 100.0_f32) + 0.5_f32) as i32;
    // Exit 42 iff w ≈ 0.8 (so w_rounded = 80; 80 - 38 = 42).
    w_rounded - 38
}
