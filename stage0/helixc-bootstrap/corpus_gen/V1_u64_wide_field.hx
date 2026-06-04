// v1.3 V1 (P0): u64 WIDE STRUCT FIELD read full 64-bit. Same as the i64 case but
// the field type is u64 (type tag 9, also 8-byte). The value 5_000_000_000 > 2^32
// is built by COMPUTATION (u64 literals > 2^32 are lex-capped per L-2; the value is
// reachable by arithmetic). Pre-fix the field READ truncated to low-32 -> 7; the
// fix reads full-width -> 50. Uses unsigned divide (u64 operands). Runtime-derived.
struct UBig { v: u64 }
fn main() -> i32 {
    let mut k: u64 = 0_u64;
    let mut i = 0;
    while i < 5 { k = k + 1000000000_u64; i = i + 1; }   // k = 5_000_000_000 (> 2^32) at runtime
    let b = UBig { v: k };
    let g: u64 = 100000000_u64;                          // 1e8
    (b.v / g) as i32                                     // 5e9 / 1e8 = 50 (truncated would give 7)
}
