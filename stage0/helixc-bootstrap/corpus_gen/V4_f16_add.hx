// v1.3 f16 GAP FIX (charter §1 V4): f16 SAME-TYPE ADDITION now computes
// CORRECTLY via the F16C hardware path (vcvtph2ps -> addss -> vcvtps2ph RNE).
// Before this fix the `f16` type ident + the f16 literal never mapped to type
// tag 5, so is_f16_expr was always 0, emit_f16_binop (the F16C path) was
// UNREACHABLE DEAD CODE, and f16 arithmetic silently routed to the bf16/integer
// path -> a WRONG value (a half pattern misread as a bf16 top-16 = a tiny
// denormal that casts to ~0), with NO trap. The audit's repro was exactly
// 100.0_f16 + 28.0_f16 -> exit 0 (the silent-wrong outcome).
//
// 100.0_f16 and 28.0_f16 are BOTH exact in f16 (f16 has 10 mantissa bits ->
// integers up to 2048 are exact). The f16 sum is 128.0, exact. The OLD
// silent-wrong path yielded ~0 (-> exit 0, the audit's bug). The F16C path
// yields exactly 128. Comparing `c as i32` to 128 (a full i32 compare, not an
// exit-byte wrap) proves the F16C path is REACHED and computes the right value:
// exit 42 iff the sum is exactly 128, else 0.
fn main() -> i32 {
    let a: f16 = 100.0_f16;
    let b: f16 = 28.0_f16;
    let c: f16 = a + b;                      // F16C: widen->addss->narrow RNE = 128.0
    if (c as i32) == 128 { 42 } else { 0 }   // 42 iff the F16C add gives 128 (old silent-wrong path -> ~0 -> 0)
}
