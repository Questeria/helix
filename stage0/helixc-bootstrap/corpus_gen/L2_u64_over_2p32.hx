// L-2 (charter §1.6) — DOCUMENT-AS-BOUND + negative test.
//
// A u64 LITERAL whose decimal magnitude exceeds 2^32-1 (= 4_294_967_295) is
// NOT supported in v1.2: the lexer's decimal accumulator is i32-wide, so such
// a literal would silently truncate mod 2^32. Rather than emit silent-wrong
// code, the lexer DELIBERATELY FAILS CLOSED — it tags the over-range literal
// as token 40 ("no parser arm", lexer.hx:580-617 + check_u64_10digit_overflow,
// lexer.hx:194), which the parser's unexpected-token catch-all turns into an
// AST_ERR, and the H-3 diagnostic path then prints a COMPILE-TIME
//   <path>:line:col: parse error: unexpected token
// and exits non-zero with NO output ELF.
//
// This is the HONEST bound: i64 values >= 2^32 work (via the i64 limb / hi32
// path — gated by i64_cmp `5_000_000_000_i64` and L2_i64_bigger ->50), and
// u64 values >= 2^32 are reachable by COMPUTATION (e.g. `1_u64 << 63`, gated
// by u64_shr), but a u64 *literal* >= 2^32 is a lex-side cap that fails loud.
// A lex-accumulator widening to carry the full 64-bit literal magnitude is a
// v-next item. This fixture is consumed by gate_kovc.sh's chk_err harness,
// which asserts the EXACT diagnostic (line 14, col 20 = the `5`) + non-zero
// exit + no ELF — proving the bound fails closed, never silent-wrong.
//
// (DO NOT "fix" this into a passing program: the feature it would test does
//  not exist; the loud compile-time rejection IS the documented behavior.)
fn main() -> i32 {
    let big: u64 = 5_000_000_000_u64;   // > 2^32 -> lex tag 40 -> compile error
    big as i32
}
