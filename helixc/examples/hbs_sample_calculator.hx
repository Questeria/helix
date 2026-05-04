// hbs_sample_calculator.hx
//
// HBS dogfood: a small expression evaluator that exercises every Tier-A
// pattern-matching feature plus totality + stdlib reach. No reflection,
// no D<T>, no generics — pure HBS.
//
// Encoding: all expressions are flattened to a stream of i32 opcodes
// in a single array. The evaluator pops/pushes against a fixed-size
// stack, also as i32. The "AST" is a list-encoded integer tape.
//
// Opcodes:
//   0  CONST n      push n
//   1  ADD          x op y → x+y
//   2  SUB          x y → x-y
//   3  MUL          x y → x*y
//   4  NEG          x → -x
//   5  ABS          x → |x|
//   9  HALT         pop and return
//
// This is contrived — the goal is to exercise match arms, range patterns,
// or-patterns, guards, and totality, not to be a real VM.

@total
fn classify_op(code: i32) -> i32 {
    // Returns:
    //   0 if the opcode is a const-loader (pushes 1 value, consumes 0)
    //   1 if it is an arithmetic 2-input op (consumes 2, pushes 1)
    //   2 if it is a unary op (consumes 1, pushes 1)
    //   3 if it is HALT
    //   4 unknown opcode
    match code {
        0 => 0,
        1 | 2 | 3 => 1,
        4 | 5 => 2,
        9 => 3,
        _ => 4,
    }
}

@total
fn small_int_class(n: i32) -> i32 {
    // Bucket small integers by magnitude:  0, 1..=9, 10..=99, 100..=999, larger.
    match n {
        0 => 0,
        1..=9 => 1,
        10..=99 => 2,
        100..=999 => 3,
        _ if n < 0 => 5,    // negative
        _ => 4,             // big positive
    }
}

@total
fn factorial(n: i32) -> i32 {
    // strictly decreases on n - 1, accepted by totality
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

@total
fn fib(n: i32) -> i32 {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}

@total
fn sum_to_n(n: i32) -> i32 {
    if n <= 0 { 0 } else { n + sum_to_n(n - 1) }
}

@total
fn arith_op(opcode: i32, x: i32, y: i32) -> i32 {
    // Apply a binary opcode to two operands. Falls through to 0 for
    // unknown — the caller is responsible for not feeding that case.
    match opcode {
        1 => x + y,
        2 => x - y,
        3 => x * y,
        _ => 0,
    }
}

@total
fn unary_op(opcode: i32, x: i32) -> i32 {
    match opcode {
        4 => 0 - x,
        5 => if x < 0 { 0 - x } else { x },
        _ => x,
    }
}

@total
fn classify_score(n: i32) -> i32 {
    // Used to demonstrate guard precedence + or-patterns.
    match n {
        x if x > 100 => 5,
        50..=100 => 4,
        x if x > 10 => 3,
        1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 => 2,
        0 => 1,
        _ => 0,
    }
}

fn main() -> i32 {
    // Exercise classify_op + small_int_class + factorial + fib +
    // sum_to_n + classify_score in a single computation that
    // collapses to a single i32 the OS exit code can carry (0..=255).
    //
    // The goal is just: this compiles, runs, exits with a deterministic
    // value we can assert in a test.
    let f4 = factorial(4);             // 24
    let fib_5 = fib(5);                // 5
    let s10 = sum_to_n(5);             // 15
    let cls = small_int_class(42);     // 2
    let arith = arith_op(1, 10, 8);    // 18
    let unary = unary_op(5, 0 - 7);    // 7
    let score = classify_score(75);    // 4
    // 24 - 5 + 15 - 2 + 18 - 7 + 4 = 47
    f4 - fib_5 + s10 - cls + arith - unary + score
}
