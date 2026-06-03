fn main() -> i32 {
    // DOC-AS-BOUND neg test for bf16/f16 ARITHMETIC (charter §1.6 L-7).
    // bf16/f16 are storage-only in v1.2: the literal codegens its truncated
    // 16-bit bit pattern (see arm_bf16_f16_decl.hx, which PASSES), but x86-64
    // baseline has no BF16/F16C arithmetic, so kovc deliberately TRAPS on any
    // bf16/f16 +,-,*,/ (is_bf16_expr, kovc.hx:1676 -- a ud2 with a bf16 trap-id).
    // This is a PERMANENT v1.2 bound (hardware-dictated), documented honestly.
    // The trap is a LOUD failure (SIGILL), not silent-wrong: this program runs a
    // bf16 ADD and is EXPECTED to die by SIGILL (exit 132), proving the arith
    // path fails closed rather than emitting wrong float math.
    let a: bf16 = 1.5_bf16;
    let b: bf16 = 2.5_bf16;
    let c: bf16 = a + b;        // <-- TRAPS (no bf16 hardware add); SIGILL 132
    c as i32
}
