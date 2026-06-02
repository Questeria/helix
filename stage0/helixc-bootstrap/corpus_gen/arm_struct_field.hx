// DDC-broad probe: struct literal + multi-field READ (AST field-load), values
// computed at runtime so the field offsets are loaded at runtime, not folded.
struct Pt { x: i32, y: i32, z: i32 }
fn main() -> i32 {
    let mut a = 0;
    let mut i = 0;
    while i < 4 { a = a + 5; i = i + 1; }     // a = 20 at runtime
    let p = Pt { x: a, y: a - 8, z: a + 10 }; // {20, 12, 30}
    p.x + p.y + p.z - 20                       // 20+12+30-20 = 42
}
