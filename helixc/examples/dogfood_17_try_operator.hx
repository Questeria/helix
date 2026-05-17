// dogfood_17_try_operator.hx — Stage 48 Increment 3 dogfood.
//
// `?` postfix propagation operator over Result<T, E>. Stage 48
// ships the syntax + typecheck guards + Phase-0 identity-lowering
// (no runtime tag yet, so every Result is shape-Ok at runtime).
// Real early-return semantics come with the runtime tag at
// Stage 49+; the source-level idiom is already correct today.
//
// What this dogfood demonstrates:
//   1. Result-returning helper (safe_div) constructed via
//      Ok(value) or Err(divisor) depending on input.
//   2. Caller (compute) chains TWO Result-returning calls via
//      `?`. Each `?` typechecks as
//      `__try(callee_returns_Result<i32, i32>)` and extracts
//      the Ok inner.
//   3. The enclosing function (compute) also returns
//      Result<i32, i32>, satisfying the Stage 48 typecheck
//      guard (3): `?` requires the enclosing fn to return
//      Result.
//   4. Err types are compatible: both safe_div's E and
//      compute's E are i32, satisfying Stage 48 guard (4).
//   5. The final unwrap_ok at main extracts the Ok payload as
//      an i32 exit code.
//
// Real-world parallel: this is the Rust / Swift idiom for
// non-throwing error propagation. `let n = parse(s)?;` is the
// canonical example.
//
// Exit code 42 iff the chained `?` operators correctly thread
// the Ok values: 20/4 = 5; 5/1 = 5; 5+37 = 42.

@pure
fn safe_div(a: i32, b: i32) -> Result<i32, i32> {
    if b == 0 { Err(b) } else { Ok(a / b) }
}

@pure
fn compute() -> Result<i32, i32> {
    // Step 1: 20 / 4 = 5. The `?` strips the Ok wrapper.
    let x: i32 = safe_div(20, 4)?;
    // Step 2: 5 / 1 = 5. Chained `?`.
    let y: i32 = safe_div(x, 1)?;
    // Step 3: rewrap as Ok with the computed result.
    Ok(y + 37)
}

fn main() -> i32 {
    // Unwrap the Ok and return as exit code.
    unwrap_ok(compute())
}
