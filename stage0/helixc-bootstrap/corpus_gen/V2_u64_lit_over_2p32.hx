// v1.3 V2 (charter §1 V2): a u64 LITERAL >= 2^32 parses + computes correctly.
// This SHIPS the v1.2 L-2 bound (which fail-closed any u64 literal > 2^32-1).
//
// 5_000_000_000_u64 (= 5e9 > 2^32 = 4_294_967_296) is written as a LITERAL --
// pre-V2 the lexer's i32 accumulator capped it (tag 40 -> compile error). The
// fix decodes the literal text into the full 64-bit value via the i64 limb
// path (UNSIGNED, no sign extension). Read it back EXACT through an unsigned
// divide: 5e9 / 1e8 = 50. A low-32 truncation would give 705032704/1e8 = 7.
// Uses u64 operands so DIV dispatches to the unsigned `xor rdx,rdx; div rcx`.
fn main() -> i32 {
    let big: u64 = 5_000_000_000_u64;   // u64 literal > 2^32 (was L-2-capped)
    let g: u64 = 100000000_u64;         // 1e8
    (big / g) as i32                    // 5e9 / 1e8 = 50 EXACT
}
