// M-2 (charter HELIX_COMPLETION.md §1.6 MED): compound assignment `op=`.
// Promotes the K1.U/K1.AN compound-assign desugar (parser.hx:7786) from
// [impl] to [proven]. The lexer has no `+=` token; the parser detects the
// (op, `=`) pair after an IDENT and desugars `x op= e` -> AST_ASSIGN(x,
// AST_BINOP(VAR(x), e)) for all 10 operators:
//   +=  -=  *=  /=  %=   (arith: AST_ADD/SUB/MUL/DIV/MOD)
//   &=  |=  ^=           (bitwise: AST_BAND/BOR/BXOR)
//   <<= >>=              (shift: AST_SHL/SHR)
// The self-host source never uses `op=` (it spells out `x = x + ...`), so
// this promotion keeps the fixpoint byte-identical; the desugar is only
// exercised here, through K2.
//
// Every operand below is a RUNTIME value (seeded via a while-loop so the
// accumulator cannot be const-folded), and each `op=` is applied in turn.
// A broken desugar for any single operator (e.g. `a += b` lowering to
// `a = b`, or `&=` mis-mapped) shifts the final value off 42.
fn main() -> i32 {
    // Seed a few runtime values that the optimizer cannot fold away.
    let mut base = 0;
    let mut i = 0;
    while i < 10 { base = base + 1; i = i + 1; }   // base = 10 at runtime
    let two = base - 8;                            // 2
    let three = base - 7;                          // 3

    let mut a = base;        // 10
    a += three;              // 13   (+=)
    a -= two;                // 11   (-=)
    a *= three;              // 33   (*=)
    a /= two;                // 16   (/=)  33/2 = 16 (integer)
    a %= base;               // 6    (%=)  16 % 10 = 6

    // Bitwise compound ops on a separate runtime accumulator.
    let mut b = base + two;  // 12   (0b1100)
    b &= base;               // 8    (12 & 10 = 0b1000)
    b |= three;              // 11   (8 | 3  = 0b1011)
    b ^= two;                // 9    (11 ^ 2 = 0b1001)

    // Shift compound ops.
    let mut c = three;       // 3
    c <<= two;               // 12   (3 << 2)
    c >>= base - 9;          // 6    (12 >> 1)

    // a=6, b=9, c=6, plus a literal 21 -> 6 + 9 + 6 + 21 = 42.
    a + b + c + 21
}
