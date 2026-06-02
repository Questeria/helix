// DDC-broad probe: enum with THREE payload-carrying variants; the selected
// variant and its payload are chosen at runtime (loop-driven index), exercising
// the match payload-extract arm beyond the Ok/Err pair, dynamically.
enum Tri { A(i32), B(i32), C(i32) }
fn pick(k: i32) -> Tri {
    if k == 0 { Tri::A(10) } else { if k == 1 { Tri::B(20) } else { Tri::C(30) } }
}
fn main() -> i32 {
    let mut k = 0;
    let mut i = 0;
    while i < 2 { k = k + 1; i = i + 1; }     // k = 2 at runtime -> variant C
    let t = pick(k);
    match t {
        Tri::A(v) => v + 1,
        Tri::B(v) => v + 2,
        Tri::C(v) => v + 12                    // 30 + 12 = 42
    }
}
