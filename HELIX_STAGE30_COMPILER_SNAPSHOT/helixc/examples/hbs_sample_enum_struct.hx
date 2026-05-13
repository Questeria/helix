// hbs_sample_enum_struct.hx
//
// HBS dogfood: combined enum + struct usage. Models a tiny
// 2D-shape calculator. Each shape is a struct, the shape kind is an
// enum, and the dispatcher pattern-matches on enum variants.
//
// Shapes are stored flatly (no algebraic-data-type variants — those
// require payload-bearing enum codegen which is still TBD). So we
// represent each shape as `{ kind: i32, a: i32, b: i32 }` where:
//   - Circle:    a = radius, b unused
//   - Square:    a = side,   b unused
//   - Rectangle: a = width,  b = height

enum Kind { Circle, Square, Rectangle }

struct Shape {
    kind: i32,
    a: i32,
    b: i32,
}

@total
fn area_squared(s: Shape) -> i32 {
    // We return area^2 to keep everything in i32 (no f32 sqrt needed).
    // Circle: π*r^2 ≈ 3 * r^2 (cheap approximation)
    // Square: a^2
    // Rectangle: a*b
    match s.kind {
        Kind::Circle => 3 * s.a * s.a,
        Kind::Square => s.a * s.a,
        Kind::Rectangle => s.a * s.b,
        _ => 0,
    }
}

@total
fn perimeter(s: Shape) -> i32 {
    match s.kind {
        Kind::Circle => 2 * 3 * s.a,        // 2π*r approx
        Kind::Square => 4 * s.a,
        Kind::Rectangle => 2 * (s.a + s.b),
        _ => 0,
    }
}

fn main() -> i32 {
    let circle = Shape { kind: Kind::Circle, a: 3, b: 0 };
    let square = Shape { kind: Kind::Square, a: 4, b: 0 };
    let rect = Shape { kind: Kind::Rectangle, a: 5, b: 6 };

    // Sum: area_squared(circle) + area_squared(square) + area_squared(rect)
    //    = 3*9 + 16 + 30 = 27 + 16 + 30 = 73
    // But in main we can't pass struct-by-value yet; field-access in
    // place works, so we compute inline:
    let circle_a = 3 * circle.a * circle.a;        // 27
    let square_a = square.a * square.a;            // 16
    let rect_a = rect.a * rect.b;                  // 30
    let circle_p = 2 * 3 * circle.a;               // 18
    let square_p = 4 * square.a;                   // 16
    let rect_p = 2 * (rect.a + rect.b);            // 22
    // areas: 73; perimeters: 56; total = 129. Truncated to i32 mod 256
    // by the OS exit code wrapper, so 129.
    circle_a + square_a + rect_a + circle_p + square_p + rect_p
}
