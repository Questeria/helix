// v1.5 S0 ternary corpus row #2: t2 ROUND-TRIP + sign coverage. Each of {-1,0,+1}
// round-trips through t2 and back to i32 (identity, since t2's scalar domain is i32),
// via helpers that exercise BOTH the return-type resolver (to_t2) and the typed-param +
// return resolvers (from_t2) plus the (x as t2)/(w as i32) casts. A ternary-weighted sum
// over +/0/- weights covers all three trit values in arithmetic. Runtime n1 (0-1) so the
// negative case cannot be a trivial literal. 42 iff every round-trip is identity AND the
// weighted sum is exact.
fn to_t2(x: i32) -> t2 { (x as t2) }
fn from_t2(w: t2) -> i32 { (w as i32) }
fn main() -> i32 {
    let n1: i32 = 0 - 1;
    let rp: i32 = from_t2(to_t2(1));     // +1 -> +1
    let rz: i32 = from_t2(to_t2(0));     // 0 -> 0
    let rn: i32 = from_t2(to_t2(n1));    // -1 -> -1
    let w0: t2 = to_t2(1);
    let w1: t2 = to_t2(n1);
    let w2: t2 = to_t2(0);
    let s: i32 = (w0 as i32) * 10 + (w1 as i32) * 7 + (w2 as i32) * 99;  // 10 - 7 + 0 = 3
    if rp == 1 {
        if rz == 0 {
            if rn == (0 - 1) {
                if s == 3 { 42 } else { 0 }
            } else { 0 }
        } else { 0 }
    } else { 0 }
}
