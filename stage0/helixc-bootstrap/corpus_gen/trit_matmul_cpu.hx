// v1.5 S0 ternary corpus row #3: a 2x2 TERNARY matmul on the CPU (C = W . X), W ternary
// (t2), X i32. Exercises t2 values + casts in a matmul-shaped computation distinct from
// trit_dot (full 2x2 with mixed +/-/0 weights), compiled + run on the from-raw self-hosted
// toolchain. Runtime n1 (0-1) for the negative weight. 42 iff all four output cells are
// exactly right.
//   W = [[+1,-1],[0,+1]]   X = [[3,4],[5,6]]
//   C00 = +1*3 + -1*5 = -2   C01 = +1*4 + -1*6 = -2
//   C10 =  0*3 + +1*5 =  5   C11 =  0*4 + +1*6 =  6
fn t2v(x: i32) -> t2 { (x as t2) }
fn main() -> i32 {
    let n1: i32 = 0 - 1;
    let w00: t2 = t2v(1);  let w01: t2 = t2v(n1);
    let w10: t2 = t2v(0);  let w11: t2 = t2v(1);
    let x00: i32 = 3; let x01: i32 = 4;
    let x10: i32 = 5; let x11: i32 = 6;
    let c00: i32 = (w00 as i32) * x00 + (w01 as i32) * x10;
    let c01: i32 = (w00 as i32) * x01 + (w01 as i32) * x11;
    let c10: i32 = (w10 as i32) * x00 + (w11 as i32) * x10;
    let c11: i32 = (w10 as i32) * x01 + (w11 as i32) * x11;
    if c00 == (0 - 2) {
        if c01 == (0 - 2) {
            if c10 == 5 {
                if c11 == 6 { 42 } else { 0 }
            } else { 0 }
        } else { 0 }
    } else { 0 }
}
