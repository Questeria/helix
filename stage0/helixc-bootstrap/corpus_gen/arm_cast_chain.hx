// DDC-broad probe: AST_CAST (tag 81) across widths on a RUNTIME value:
// i32 -> i64 (sign-extend) -> arithmetic in i64 -> back to i32. The casts run
// at runtime (value is loop-accumulated), exercising the cast/movsxd path.
fn main() -> i32 {
    let mut a = 0;
    let mut i = 0;
    while i < 7 { a = a + 3; i = i + 1; }     // a = 21 (i32) at runtime
    let big: i64 = a as i64;                   // i32 -> i64
    let doubled: i64 = big * 2_i64;            // 42 in i64
    doubled as i32                              // i64 -> i32 -> 42
}
