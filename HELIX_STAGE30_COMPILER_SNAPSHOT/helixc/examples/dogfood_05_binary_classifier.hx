// dogfood_05_binary_classifier.hx — sigmoid-based binary classifier.
//
// A 1D logistic regression: y_pred = sigmoid(w*x + b), trained via
// binary-cross-entropy loss with multi-output reverse-mode AD writing
// gradients into reflection cells in one pass.
//
// Demonstrates that the range-reduced __exp now handles realistic NN
// logits (|x| up to 30) — older Taylor-only __exp would have made
// sigmoid unusable here.
//
// Training data (4 points learning the threshold y = (x > 0.5)):
//   (0.0, 0)  (0.4, 0)  (0.6, 1)  (1.0, 1)
//
// Loss for one example:
//   pred = sigmoid(w * x + b)
//   bce  = -[y*log(pred) + (1-y)*log(1-pred)]    // __bce builtin
//
// Total loss = sum across 4 examples; grad_rev_all writes both
// gradients into cells [base, base+1] in one analysis pass.

@pure fn loss_total(w: f32, b: f32) -> f32 {
    let p1 = __sigmoid(w * 0.0 + b);
    let p2 = __sigmoid(w * 0.4 + b);
    let p3 = __sigmoid(w * 0.6 + b);
    let p4 = __sigmoid(w * 1.0 + b);
    __bce(p1, 0.0) + __bce(p2, 0.0) + __bce(p3, 1.0) + __bce(p4, 1.0)
}

@verifier
fn safe_param(handle: i32, val: f32) -> i32 {
    if val < 0.0 - 30.0 { 0 }
    else { if val > 30.0 { 0 } else { 1 } }
}

fn main() -> i32 {
    let cell_w = quote(0);
    let cell_b = quote(1);
    let cell_gw = quote(2);
    let cell_gb = quote(3);

    // Initial weights.
    modify_f(cell_w, 0.5, safe_param);
    modify_f(cell_b, 0.0 - 0.2, safe_param);

    print_str("=== Helix binary classifier ===\n");
    print_str("learning threshold y = (x > 0.5) via gradient descent\n");

    let lr = 2.0;
    let mut step: i32 = 0;
    while step < 30 {
        let w = splice_f(cell_w);
        let b = splice_f(cell_b);

        // grad_rev_all writes gradients into cells starting at quote(2).
        // Index 0 -> ∂L/∂w lands in cell_gw (= quote(2))
        // Index 1 -> ∂L/∂b lands in cell_gb (= quote(3))
        grad_rev_all(loss_total)(w, b, 2);
        let gw = splice_f(cell_gw);
        let gb = splice_f(cell_gb);

        // SGD step
        modify_f(cell_w, w - lr * gw, safe_param);
        modify_f(cell_b, b - lr * gb, safe_param);

        step = step + 1;
    }

    let final_w = splice_f(cell_w);
    let final_b = splice_f(cell_b);

    print_str("training done\n");

    // Check classifier on each training point. Convergence: w should be
    // moderately positive (so x increases logit), b should be near -w/2
    // so the decision boundary is around x = 0.5.
    let p_low = __sigmoid(final_w * 0.0 + final_b);   // expect < 0.5
    let p_high = __sigmoid(final_w * 1.0 + final_b);  // expect > 0.5

    let low_correct = if p_low < 0.5 { 1 } else { 0 };
    let high_correct = if p_high > 0.5 { 1 } else { 0 };

    if low_correct == 1 {
        print_str("low input correctly classified\n");
    }
    if high_correct == 1 {
        print_str("high input correctly classified\n");
    }

    // 1 + 1 + 40 = 42 if both classified correctly.
    low_correct + high_correct + 40
}
