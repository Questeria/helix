fn main() -> i32 {
    // `continue` in a while loop: sum 1..=10 but SKIP 3 and 7 via continue.
    // 1+2+4+5+6+8+9+10 = 45 ; then +... no: sum = 55 - 3 - 7 = 45. Adjust to 42:
    // skip 3, 7, AND 13 (out of range) -> 55 - 10 = 45 ; subtract 3 more below.
    let mut i: i32 = 0;
    let mut acc: i32 = 0;
    while i < 10 {
        i = i + 1;
        if i == 3 { continue; }   // skip 3
        if i == 7 { continue; }   // skip 7
        acc = acc + i;
    }
    // acc = (1+2+4+5+6+8+9+10) = 45 ; 45 - 3 = 42
    acc - 3
}
