// dogfood_19_pat_struct_guards.hx — Stage 59 follow-on.
//
// Demonstrates Tier 4 #15 pattern matching's FULL surface:
// PatStruct destructuring composed with guards. This exercises
// `_check_guard` + PatStruct binders being visible to the
// guard expression (the guard sees `x` and `y` bound by the
// pattern, so it can reference them).
//
// API showcase: a Point classifier that maps coordinates to a
// region tag based on the bound (x, y) values via guards.

struct Point { x: i32, y: i32 }

@pure
fn classify_point(p: Point) -> i32 {
    match p {
        // Origin → 1.
        Point { x: 0, y: 0 } => 1,
        // X-axis (non-origin) → 2. Guard distinguishes from origin.
        Point { x, y: 0 } if x != 0 => 2,
        // Y-axis (non-origin) → 3. Guard on y.
        Point { x: 0, y } if y != 0 => 3,
        // First quadrant (x>0 AND y>0) → 4. Guard tests bound fields.
        Point { x, y } if x > 0 => if y > 0 { 4 } else { 5 },
        // Anywhere else → 6.
        Point { .. } => 6,
    }
}

fn main() -> i32 {
    let p_origin = Point { x: 0, y: 0 };
    let p_xax = Point { x: 5, y: 0 };
    let p_yax = Point { x: 0, y: 7 };
    let p_q1 = Point { x: 3, y: 4 };
    let p_other = Point { x: 0 - 1, y: 2 };

    let c1 = classify_point(p_origin);     // 1
    let c2 = classify_point(p_xax);        // 2
    let c3 = classify_point(p_yax);        // 3
    let c4 = classify_point(p_q1);         // 4
    let c5 = classify_point(p_other);      // 6 (x is -1, y>0; neither first arm)

    // Sum: 1 + 2 + 3 + 4 + 6 = 16. Multiply by classifier-coverage
    // weight (6 arms covered, 5 inputs landed somewhere) → 16+6+5+15 = 42.
    // Actually simpler: 1+2+3+4+6 = 16; add 26 to reach 42.
    c1 + c2 + c3 + c4 + c5 + 26
}
