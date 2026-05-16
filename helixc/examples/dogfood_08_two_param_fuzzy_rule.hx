// dogfood_08_two_param_fuzzy_rule.hx — Stage 36 Increment 8 dogfood.
//
// Learns a TWO-PARAMETER fuzzy rule via SGD using gradients that
// flow through propositional logic. Extends dogfood_07's single-
// parameter SGD to a multi-parameter rule system with two examples.
//
// The rule:
//
//     hypothesis(a, b) = fuzzy_or(fuzzy_and(a, w1),
//                                  fuzzy_and(b, w2))
//
// Training data (two examples, each with its own loss):
//
//   loss1: example (a=1, b=0) — target output 0.9
//          hypothesis = fuzzy_or(fuzzy_and(1.0, w1), fuzzy_and(0.0, w2))
//                     = fuzzy_or(w1, 0)
//                     = w1 + 0 - w1*0
//                     = w1
//          loss1 = (w1 - 0.9)^2
//
//   loss2: example (a=0, b=1) — target output 0.7
//          hypothesis = fuzzy_or(0, w2) = w2
//          loss2 = (w2 - 0.7)^2
//
// Because the rule structure separates w1 and w2 cleanly (each
// loss only depends on one weight), SGD has a closed-form per-step
// update. With lr=0.5 the step is:
//     w_new = w - 0.5 * 2*(w - target) = target
// So w1 → 0.9 and w2 → 0.7 in one step, but 50 iterations confirm
// stability.
//
// AD machinery used: grad_rev(loss1, 0) differentiates loss1 w.r.t.
// the 0th param (w1); grad_rev(loss2, 1) differentiates loss2
// w.r.t. the 1st param (w2). This exercises the multi-argument
// reverse-mode AD path through fuzzy_or + fuzzy_and chains.
//
// Exit code 42 iff w1 ≈ 0.9 AND w2 ≈ 0.7 (so w1*100 + w2*100 ≈ 160,
// and 160 - 118 = 42).

fn loss1(w1: f32, w2: f32) -> f32 {
    // Example 1: a=1, b=0 — target 0.9
    let a: Logic<f32> = prove(1.0_f32, 100);
    let b: Logic<f32> = prove(0.0_f32, 200);
    let wl1: Logic<f32> = prove(w1, 300);
    let wl2: Logic<f32> = prove(w2, 400);
    let pred: f32 = unwrap_logic(
        fuzzy_or(fuzzy_and(a, wl1), fuzzy_and(b, wl2)));
    let target: f32 = 0.9_f32;
    let diff: f32 = pred - target;
    diff * diff
}

fn loss2(w1: f32, w2: f32) -> f32 {
    // Example 2: a=0, b=1 — target 0.7
    let a: Logic<f32> = prove(0.0_f32, 100);
    let b: Logic<f32> = prove(1.0_f32, 200);
    let wl1: Logic<f32> = prove(w1, 300);
    let wl2: Logic<f32> = prove(w2, 400);
    let pred: f32 = unwrap_logic(
        fuzzy_or(fuzzy_and(a, wl1), fuzzy_and(b, wl2)));
    let target: f32 = 0.7_f32;
    let diff: f32 = pred - target;
    diff * diff
}

fn main() -> i32 {
    let mut w1: f32 = 0.0_f32;
    let mut w2: f32 = 0.0_f32;
    let lr: f32 = 0.5_f32;
    let mut i: i32 = 0;
    while i < 50 {
        // grad_rev(fn, k) differentiates fn w.r.t. the k-th param.
        // loss1 depends only on w1 (example a=1, b=0). loss2 only on w2.
        let g_w1: f32 = grad_rev(loss1, 0)(w1, w2);
        let g_w2: f32 = grad_rev(loss2, 1)(w1, w2);
        w1 = w1 - lr * g_w1;
        w2 = w2 - lr * g_w2;
        i = i + 1;
    }
    // After convergence: w1 ≈ 0.9, w2 ≈ 0.7.
    let s: i32 = ((w1 * 100.0_f32) + (w2 * 100.0_f32) + 0.5_f32) as i32;
    // Expected sum = 90 + 70 = 160. Exit code 160 - 118 = 42.
    s - 118
}
