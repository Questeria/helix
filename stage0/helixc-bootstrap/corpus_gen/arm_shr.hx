// DDC-broad probe: AST_SHR (tag 33) right-shift on i32, dynamically executed.
// Shift amount and value are both runtime so the `shr eax, cl` path runs.
fn main() -> i32 {
    let mut v = 0;
    let mut i = 0;
    while i < 6 { v = v + 56; i = i + 1; }    // v = 336 at runtime
    let mut sh = 0;
    let mut j = 0;
    while j < 3 { sh = sh + 1; j = j + 1; }   // sh = 3 at runtime
    v >> sh                                    // 336 >> 3 = 42
}
