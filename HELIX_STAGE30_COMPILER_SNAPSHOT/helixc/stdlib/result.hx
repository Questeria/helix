// helixc/stdlib/result.hx — Result<T, E> for fallible operations.
//
// Phase 1.9: Ok(value) | Err(code). Like Option<T> but with an
// explicit error payload. Use for fallible operations:
//
//   let r: Result = parse_int(s);
//   match r {
//       Result::Ok(x) => x,
//       Result::Err(code) => handle_error(code),
//   }
//
// Phase 1.9 specialisation: i32 value, i32 error code (most error
// codes fit in i32; for richer errors you can carry an arena offset
// or string-table index).
//
// License: Apache 2.0

@pure
enum Result {
    Ok(i32),
    Err(i32),
}

@pure
fn result_unwrap_or(r: Result, default_v: i32) -> i32 {
    match r {
        Result::Ok(x) => x,
        Result::Err(_) => default_v,
    }
}

@pure
fn result_is_ok(r: Result) -> i32 {
    match r {
        Result::Ok(_) => 1,
        Result::Err(_) => 0,
    }
}

@pure
fn result_is_err(r: Result) -> i32 {
    match r {
        Result::Ok(_) => 0,
        Result::Err(_) => 1,
    }
}

@pure
fn result_or_zero(r: Result) -> i32 {
    match r {
        Result::Ok(x) => x,
        Result::Err(_) => 0,
    }
}

@pure
fn result_or_neg(r: Result) -> i32 {
    match r {
        Result::Ok(x) => x,
        Result::Err(_) => 0 - 1,
    }
}

@pure
fn result_eq_ok(r: Result, target: i32) -> i32 {
    match r {
        Result::Ok(x) => if x == target { 1 } else { 0 },
        Result::Err(_) => 0,
    }
}

@pure
fn result_err_code_or(r: Result, default_v: i32) -> i32 {
    match r {
        Result::Ok(_) => default_v,
        Result::Err(c) => c,
    }
}
