// M-1 (charter HELIX_COMPLETION.md §1.6 MED): `for` loops.
// Promotes the parse_for desugar (parser.hx:16017) from [impl] to [proven].
// `for v in start..end { body }` desugars to AST_LET_MUT + AST_WHILE +
// AST_SEQ + AST_ASSIGN + AST_ADD + AST_LT (no new codegen tag); the
// inclusive `..=` form uses AST_LE. The self-host bootstrap source uses
// plain `while`, never `for`, so promoting this corpus row leaves the
// fixpoint byte-identical -- the feature is only exercised by THIS program
// compiled through K2.
//
// Three checks prove correct computation, all runtime-driven so nothing
// can be const-folded:
//   (a) exclusive range  for i in 0..9   -> sum 0+1+..+8        = 36
//   (b) inclusive range  for j in 1..=5  -> sum 1+2+3+4+5       = 15  (..= runs at j==5)
//   (c) bounds from vars  for k in lo..hi (lo=2, hi=6)          -> 2+3+4+5 = 14
// total = 36 - 15 + 14 = 35; +7 -> 42.  A broken `for` (wrong bound /
// missing increment / inclusive-as-exclusive) changes one of these sums
// and the exit code moves off 42.
fn main() -> i32 {
    let mut sum_a = 0;
    for i in 0..9 { sum_a = sum_a + i; }        // exclusive: 0..8 -> 36

    let mut sum_b = 0;
    for j in 1..=5 { sum_b = sum_b + j; }       // inclusive: 1..5 -> 15

    let lo = 2;
    let hi = 6;
    let mut sum_c = 0;
    for k in lo..hi { sum_c = sum_c + k; }      // var bounds: 2..5 -> 14

    sum_a - sum_b + sum_c + 7                    // 36 - 15 + 14 + 7 = 42
}
