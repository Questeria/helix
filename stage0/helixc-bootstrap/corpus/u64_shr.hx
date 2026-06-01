fn shr_u64(x: u64) -> u64 { x >> 63_u64 }
fn main() -> i32 { let x: u64 = 1_u64 << 63_u64; shr_u64(x) as i32 }
