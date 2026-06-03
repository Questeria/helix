fn main() -> i32 {
    // bf16 / f16 are STORAGE-only scalar float types in v1.2: a literal lexes
    // and codegens its (truncated) 16-bit bit pattern, but there is NO x86
    // baseline hardware for bf16/f16 ARITHMETIC, so any +,-,*,/ on them traps
    // (is_bf16_expr, kovc.hx:1676). This probe exercises only what is supported:
    // the literals are declared (their bit patterns emitted) and the function
    // returns a plain i32 -- proving the bf16/f16 LITERAL SHAPES parse + codegen
    // without arithmetic. (A bf16/f16 arithmetic doc-as-bound neg test lives in
    // arm_bf16_arith_bound.hx.)
    let _a: bf16 = 1.5_bf16;
    let _b: f16 = 2.5_f16;
    42
}
