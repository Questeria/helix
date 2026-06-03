fn pick(k: i32) -> i32 {
    // early `return` from inside control flow (before the function's tail).
    if k == 1 { return 42; }
    if k == 2 { return 7; }
    let mut i: i32 = 0;
    while i < 100 {
        if i == 5 { return 99; }   // early return out of a loop
        i = i + 1;
    }
    0   // tail (unreached for k in {1,2})
}
fn main() -> i32 {
    // pick(1) takes the FIRST early return -> 42 ; pick(2) -> 7.
    pick(1) + (pick(2) - 7)   // 42 + 0 = 42
}
