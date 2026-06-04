// v1.3 V1 (P0): i64 WIDE STRUCT FIELD read full 64-bit. A struct field holding
// an i64 value > 2^32 must read back EXACT (not low-32 truncated). Pre-fix the
// field READ emitted a 4-byte load -> 5_000_000_000 (0x1_2A05F200) read back as
// 705_032_704 (0x2A05F200, the low 32 bits) -> 705032704/1e8 = 7 (WRONG). The fix
// (decl-time 8-byte-scalar detection + a 64-bit field load + i64 result typing)
// makes the read full-width -> 5_000_000_000/1e8 = 50 (EXACT). Runtime-derived
// base (loop) so nothing constant-folds; the divide forces a real 64-bit value.
struct Big { v: i64 }
fn main() -> i32 {
    let mut k: i64 = 0_i64;
    let mut i = 0;
    while i < 5 { k = k + 1000000000_i64; i = i + 1; }   // k = 5_000_000_000 (> 2^32) at runtime
    let b = Big { v: k };
    let g: i64 = 100000000_i64;                          // 1e8
    (b.v / g) as i32                                     // 5e9 / 1e8 = 50 (truncated would give 7)
}
