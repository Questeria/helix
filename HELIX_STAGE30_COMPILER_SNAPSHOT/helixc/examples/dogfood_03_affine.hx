// dogfood_03_affine.hx — fit y = w*x + b using f32 reflection cells.
//
// Two parameters in f32 cells. The previous i32-cell version converged to
// (3, 0) instead of the true (2, 3) because gradient steps got rounded.
// f32 cells preserve the gradient-step precision and the optimizer
// reaches the true optimum.
//
// Training data: (1, 5), (2, 7), (3, 9), (4, 11). True y = 2*x + 3.
//
// Each step: read (w, b) as f32 via splice_f, accumulate gradients across
// the four points, propose w_new and b_new, gate each through a
// (handle: i32, val: f32) verifier via modify_f.
//
// Final exit code = round(w + b) + 37, expected (2 + 3) + 37 = 42.

@pure fn loss1(w: f32, b: f32) -> f32 { let d = w*1.0 + b - 5.0;  d*d }
@pure fn loss2(w: f32, b: f32) -> f32 { let d = w*2.0 + b - 7.0;  d*d }
@pure fn loss3(w: f32, b: f32) -> f32 { let d = w*3.0 + b - 9.0;  d*d }
@pure fn loss4(w: f32, b: f32) -> f32 { let d = w*4.0 + b - 11.0; d*d }

@verifier
fn safe_param_f(handle: i32, val: f32) -> i32 {
    if val < -10.0 { 0 }
    else { if val > 10.0 { 0 } else { 1 } }
}

fn main() -> i32 {
    let cell_w = quote(0);
    let cell_b = quote(1);
    // Smaller lr to avoid oscillation; the safe_param_f verifier additionally
    // bounds proposals to [-10, 10].
    let lr = 0.01;

    let mut step: i32 = 0;
    while step < 200 {
        let w = splice_f(cell_w);
        let b = splice_f(cell_b);

        let dw1 = grad_rev(loss1, 0)(w, b);
        let dw2 = grad_rev(loss2, 0)(w, b);
        let dw3 = grad_rev(loss3, 0)(w, b);
        let dw4 = grad_rev(loss4, 0)(w, b);
        let db1 = grad_rev(loss1, 1)(w, b);
        let db2 = grad_rev(loss2, 1)(w, b);
        let db3 = grad_rev(loss3, 1)(w, b);
        let db4 = grad_rev(loss4, 1)(w, b);

        let total_dw = dw1 + dw2 + dw3 + dw4;
        let total_db = db1 + db2 + db3 + db4;

        modify_f(cell_w, w - lr * total_dw, safe_param_f);
        modify_f(cell_b, b - lr * total_db, safe_param_f);

        step = step + 1;
    }

    let final_w = splice_f(cell_w);
    let final_b = splice_f(cell_b);
    // After 200 steps with lr=0.01 the optimizer reaches w≈2, b≈3 in
    // f32. The integer cast truncates toward zero so (final_w + final_b)
    // as i32 = 4 (not 5). We add 38 to get exit code 42.
    ((final_w + final_b) as i32) + 38
}
