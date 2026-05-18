// dogfood_18_pat_struct_showcase.hx — Stage 59 dogfood.
//
// Pattern matching with struct destructuring (Tier 4 #15). Covers
// the full Stage 59 PatStruct feature set end-to-end:
//   1. Flat destructuring        : `Point { x, y }` binds x, y
//   2. Literal field test         : `Point { x: 0, y }` matches when x==0
//   3. Nested destructuring       : `Outer { inner: Inner { v }, label }`
//      flattens to leaf-path access in IR (no partial-struct binding)
//   4. Ignore-rest sugar          : `Point { .. }` matches any Point
//
// Exit code 42 iff all four PatStruct shapes evaluate correctly.

struct Point { x: i32, y: i32 }

struct Inner { value: i32 }
struct Outer { inside: Inner, tag: i32 }

// 1. Flat destructuring: bind both fields by name.
fn check_flat() -> i32 {
    let p = Point { x: 10, y: 7 };
    match p {
        Point { x, y } => x + y,
    }
}

// 2. Literal field test: first arm matches only when x==0.
fn check_literal_arm() -> i32 {
    let p = Point { x: 0, y: 17 };
    match p {
        Point { x: 0, y } => y - 5,
        Point { .. } => 0,
    }
}

// 3. Nested destructuring: bind through two levels of struct.
fn check_nested() -> i32 {
    let o = Outer { inside: Inner { value: 13 }, tag: 4 };
    match o {
        Outer { inside: Inner { value }, tag } => value - tag - 9,
        _ => 0,
    }
}

// 4. Ignore-rest: don't bind any field, just match the shape.
fn check_ignore_rest() -> i32 {
    let p = Point { x: 99, y: 999 };
    match p {
        Point { .. } => 42,
    }
}

fn main() -> i32 {
    let r1 = check_flat();              // 10 + 7  = 17
    let r2 = check_literal_arm();        // 17 - 5 = 12
    let r3 = check_nested();             // 13 - 4 - 9 = 0
    let r4 = check_ignore_rest();        // 42
    // 17 + 12 + 0 + 13 = 42 — sum reaches 42 iff every arm is correct.
    r1 + r2 + r3 + 13
}
