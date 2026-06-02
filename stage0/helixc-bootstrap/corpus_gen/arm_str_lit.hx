// DDC-broad probe: AST_STR_LIT (tag 25). A &str literal is materialized (the
// str-lit codegen arm runs); __strlen folds its byte length on the inline
// literal (the literal-arg form the frontend requires). The folded length then
// drives a RUNTIME loop so the exit reflects the str-lit-derived value, and the
// str-lit arm provably participates (the program will not link/run without it).
fn main() -> i32 {
    let n = __strlen("helixhelix");   // 10 (inline-literal form)
    let mut acc = 0;
    let mut i = 0;
    while i < n { acc = acc + 4; i = i + 1; }   // 40, loop bound from str length
    acc + 2                                      // 42
}
