// v1.5 S0 (2026-06-13): FIRST-CLASS TERNARY type `t2` (tag 12, BitNet b1.58, values -1/0/+1).
// Increment 1 proves the type REGISTERS end-to-end and that ternary values round-trip + accumulate.
//
// Scalar domain = i32: the -1/0/+1 constraint is a packing/matmul convention (S0 increment 2 adds
// the GPU integer-accumulate ternary matmul kernel), NOT a scalar trap -- so `t2` resolves to a
// 4-byte i32-shaped slot (type_width_class <100 default = 4) and `as i32`/`as t2` are int->int
// identity round-trips (emit_cast_conv_core, no bytes for a same-shape non-64-bit target).
//
// This fixture exercises ALL THREE type-ident resolvers + the cast:
//   make_trit  -> a `t2` RETURN type  (return-type resolver)
//   dot4       -> four `t2` PARAMS    (typed-param resolver; also >6 args = SysV stack-pass)
//   (x as t2)  -> the main ty_ident_to_tag (cast target)
// The dot is computed at RUNTIME (fn params, not compile-time literals) so it cannot constant-fold.
// Weights w=[+1,-1,0,+1], activations x=[10,20,30,5] -> +10 -20 +0 +5 = -5. Full i32 compare
// returns the 42/0 sentinel (avoids the 8-bit exit-code wrap), mirroring the V4_bf16/f16 rows.
// 42 iff the ternary dot is exactly -5.
fn make_trit(x: i32) -> t2 { (x as t2) }
fn dot4(w0: t2, w1: t2, w2: t2, w3: t2, x0: i32, x1: i32, x2: i32, x3: i32) -> i32 {
    (w0 as i32) * x0 + (w1 as i32) * x1 + (w2 as i32) * x2 + (w3 as i32) * x3
}
fn main() -> i32 {
    let n1: i32 = 0 - 1;
    let d: i32 = dot4(make_trit(1), make_trit(n1), make_trit(0), make_trit(1), 10, 20, 30, 5);
    if d == (0 - 5) { 42 } else { 0 }
}
