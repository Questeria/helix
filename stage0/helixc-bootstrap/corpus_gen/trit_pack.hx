// v1.5 S0 ternary corpus row #4: PACKED ternary storage round-trip (the 2-bit packing the
// GPU packed_ternary_matmul kernel uses). Encode ternary t2 values -1/0/+1 as base-4 codes
// 2/0/1, pack into one i32 word (word = sum_j code_j*4^j), then UNPACK each field via the same
// repeated-DIVISION decode the device uses (code = w-(w/4)*4 = w%4; trit = code-3*(code/2)),
// and verify every field round-trips to its original t2 value. Uses t2 throughout (enc takes a
// t2, dec returns a t2; cast t2<->i32 is identity). 42 iff all four packed/unpacked trits match.
//   trits [-1, 0, +1, +1] -> codes [2,0,1,1] -> word = 2 + 0 + 16 + 64 = 82 -> unpack -> [-1,0,+1,+1]
fn enc(t: t2) -> i32 { let v: i32 = (t as i32); if v < 0 { 2 } else { if v > 0 { 1 } else { 0 } } }
fn dec(code: i32) -> t2 { ((code - 3 * (code / 2)) as t2) }
fn main() -> i32 {
    let n1: i32 = 0 - 1;
    let t0: t2 = (n1 as t2);
    let t1: t2 = (0 as t2);
    let tp2: t2 = (1 as t2);
    let t3: t2 = (1 as t2);
    let word: i32 = enc(t0) + enc(t1) * 4 + enc(tp2) * 16 + enc(t3) * 64;
    let mut w: i32 = word;
    let d0: t2 = dec(w - (w / 4) * 4); w = w / 4;
    let d1: t2 = dec(w - (w / 4) * 4); w = w / 4;
    let d2: t2 = dec(w - (w / 4) * 4); w = w / 4;
    let d3: t2 = dec(w - (w / 4) * 4);
    if (d0 as i32) == (t0 as i32) {
        if (d1 as i32) == (t1 as i32) {
            if (d2 as i32) == (tp2 as i32) {
                if (d3 as i32) == (t3 as i32) { 42 } else { 0 }
            } else { 0 }
        } else { 0 }
    } else { 0 }
}
