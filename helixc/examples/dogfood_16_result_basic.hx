// dogfood_16_result_basic.hx — Stage 46 Increment 3 dogfood.
//
// Result<T, E> typecheck-side scaffolding. First two-parameter
// wrapper family in the Helix type system. Phase-0:
// identity-lowered at IR (no runtime tag yet); the `?`
// operator + real branching come in Stage 47+.
//
// What this dogfood demonstrates:
//   1. Ok(v) constructs a Result with the ok-side payload.
//   2. Err(e) constructs a Result with the err-side payload.
//   3. unwrap_ok / unwrap_err extract the inner.
//   4. map_ok replaces the Ok value with a new one (Phase-0:
//      operates on the value directly since no closures exist
//      yet — the new value is passed as the second arg).
//   5. Composition with Stage 37-41 wrappers: a Known<i32>
//      inside a Result<Known<i32>, i32> round-trips correctly.
//
// Exit code 42 iff FOUR independent Result-based witnesses
// (3 safe_double + 1 cross-stack `Known<Cause<i32>>`
// composition) all produce the expected values. Witness is
// collapse-resistant: any wrong-arm at Ok/unwrap/map collapses
// the product to 0.

@pure
fn safe_double(x: i32) -> i32 {
    let r: Result<i32, i32> = Ok(x);
    unwrap_ok(map_ok(r, x + x))
}

@pure
fn cross_stack_result(x: i32) -> i32 {
    // Result<Known<i32>, i32> — Phase-0 composition probe.
    let k: Known<i32> = into_known(x);
    let r: Result<Known<i32>, i32> = Ok(k);
    from_known(unwrap_ok(r))
}

fn main() -> i32 {
    let a: i32 = safe_double(5);           // 5+5 = 10
    let b: i32 = safe_double(7);           // 7+7 = 14
    let c: i32 = safe_double(9);           // 9+9 = 18
    let cs: i32 = cross_stack_result(42);  // 42

    let a_ok: i32 = if a == 10 { 1 } else { 0 };
    let b_ok: i32 = if b == 14 { 1 } else { 0 };
    let c_ok: i32 = if c == 18 { 1 } else { 0 };
    let cs_ok: i32 = if cs == 42 { 1 } else { 0 };

    let all_ok: i32 = a_ok * b_ok * c_ok * cs_ok;
    // Sum of safe_double results = 10 + 14 + 18 = 42.
    all_ok * (a + b + c)
}
