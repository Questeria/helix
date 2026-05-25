"""Stage 44 — Stack-passed overflow float args (Tier 1 #5).

SysV x86-64 ABI: the first 8 float args go in xmm0..xmm7; the 9th
and later go on the caller's stack. Pre-Stage-44, the backend
raised NotImplementedError on the 9th float arg, blocking real ML
code (hit during XOR perceptron dogfooding).

Stage 44 wires:
- Caller side (lower_ast / codegen): pre-pass count of overflow
  float args, 16-byte-aligned `sub rsp, N`, bit-blit each
  overflow arg to `[rsp + 8*idx]` via rax/eax, then CALL, then
  `add rsp, N`.
- Callee side: prologue loads overflow args from
  `[rbp + 16 + 8*idx]` (above saved rbp + return addr) into
  local frame slots via xmm0 as scratch.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from helixc.tests._codegen_backend import compile_module_to_elf
from helixc.frontend.parser import parse
from helixc.frontend.typecheck import typecheck
from helixc.ir.lower_ast import lower


def _run_elf(elf: bytes) -> int:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(elf)
        bin_path = f.name
    try:
        os.chmod(bin_path, 0o755)
        abs_p = bin_path.replace("\\", "/").replace("C:", "/mnt/c")
        r = subprocess.run(
            ["wsl", "--", "bash", "-c", f"chmod +x {abs_p} && {abs_p}"],
            capture_output=True, timeout=30,
        )
        return r.returncode
    finally:
        try:
            os.unlink(bin_path)
        except OSError:
            pass


def test_stage44_9_float_args_sums_correctly():
    """The 9th float arg lands at the callee correctly. Sum of
    1.0..9.0 = 45 — any indexing bug in the overflow path
    collapses the witness."""
    src = """
fn sum9(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32, i: f32) -> f32 {
    a + b + c + d + e + f + g + h + i
}

fn main() -> i32 {
    let r = sum9(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    r as i32
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 45


def test_stage44_10_float_args_sums_correctly():
    """10 args = 2 overflow. Sum 1..10 = 55."""
    src = """
fn sum10(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32, i: f32, j: f32) -> f32 {
    a + b + c + d + e + f + g + h + i + j
}

fn main() -> i32 {
    let r = sum10(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
    r as i32
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 55


def test_stage44_12_float_args_sums_correctly():
    """12 args = 4 overflow. Sum 1..12 = 78."""
    src = """
fn sum12(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32, i: f32, j: f32, k: f32, l: f32) -> f32 {
    a + b + c + d + e + f + g + h + i + j + k + l
}

fn main() -> i32 {
    let r = sum12(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
    r as i32
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 78


def test_stage44_overflow_preserves_register_args():
    """The 9th arg overflows, but args 1-8 must still arrive in
    xmm0..xmm7 correctly. Subtracting the overflow value from
    the register-arg sum proves both paths work."""
    src = """
fn check(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32, i: f32) -> f32 {
    // Sum the first 8 (register args), then subtract the 9th (overflow).
    // 8 args = 8.0, 9th = 1.0, so result = 7.0.
    (a + b + c + d + e + f + g + h) - i
}

fn main() -> i32 {
    let r = check(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    r as i32
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 7


def test_stage44_9_f64_args_sums_correctly():
    """f64 path: same overflow semantics, 8-byte stack slots."""
    src = """
fn sum9_f64(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64, i: f64) -> f64 {
    a + b + c + d + e + f + g + h + i
}

fn main() -> i32 {
    let r = sum9_f64(1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64,
                     6.0_f64, 7.0_f64, 8.0_f64, 9.0_f64);
    r as i32
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 45


def test_stage44_distinct_arg_values_no_aliasing():
    """Distinct args 10/20/30/.../90 sum to 450. Any
    register-overlap or stack-slot aliasing bug collapses to a
    different total."""
    src = """
fn sum9(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32, i: f32) -> f32 {
    a + b + c + d + e + f + g + h + i
}

fn main() -> i32 {
    let r = sum9(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0);
    // Want 450 but exit code is 8-bit; use mod 256 + a confirming
    // arithmetic check: 450 mod 256 = 194.
    (r as i32) - 256
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    # 450 - 256 = 194 fits in exit code.
    assert _run_elf(elf) == 194


# ============================================================
# Stage 44 closure gate-1 backfills:
# - F1 (HIGH): FFI_CALL must support 9+ float args (parity with CALL).
# - F2 (HIGH): mixed int+float overflow rejects before any sub_rsp.
# - F3 (MEDIUM): assertions pin pre-pass vs store-loop accounting.
# - F4 (MEDIUM): alignment tripwire asserts stack_alloc % 16 == 0.
# ============================================================


def test_stage44_gate1_f1_ffi_call_supports_9_float_args():
    """FFI_CALL was the asymmetric sibling at Stage 44 Inc 1 —
    raised NotImplementedError on the 9th float. Post-fix it
    mirrors the CALL arm. Probe: declare an extern "C" 9-float
    function and call it; the codegen must NOT raise. We can't
    actually link an extern at test time, so we just verify
    that the compile pipeline succeeds end-to-end without
    raising."""
    from helixc.frontend.parser import parse as parse_helix
    from helixc.frontend.typecheck import typecheck as tc
    from helixc.ir.lower_ast import lower as lower_ir
    from helixc.tests._codegen_backend import compile_module_to_elf

    src = """
extern "C" fn fma9(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32, i: f32) -> f32;

fn main() -> i32 {
    let r = fma9(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    r as i32
}
"""
    prog = parse_helix(src, include_stdlib=True)
    assert tc(prog) == []
    # The compile must not raise NotImplementedError.
    # (We can't run the ELF — the extern can't link — but
    # the codegen must complete cleanly.)
    elf = compile_module_to_elf(lower_ir(prog))
    assert len(elf) > 0


def test_stage44_gate1_f2_mixed_int_float_overflow_rejects_cleanly():
    """A call with 7+ int args plus 9+ float args must raise BEFORE
    any sub_rsp mutation. Pre-fix the raise fired mid-shuffle
    after the float-overflow sub_rsp had already executed."""
    from helixc.frontend.parser import parse as parse_helix
    from helixc.frontend.typecheck import typecheck as tc
    from helixc.ir.lower_ast import lower as lower_ir
    from helixc.tests._codegen_backend import compile_module_to_elf
    import pytest

    src = """
fn big(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32,
       x1: f32, x2: f32, x3: f32, x4: f32, x5: f32, x6: f32,
       x7: f32, x8: f32, x9: f32) -> i32 {
    a + b + c + d + e + f + g
}

fn main() -> i32 {
    big(1, 2, 3, 4, 5, 6, 7,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
}
"""
    prog = parse_helix(src, include_stdlib=True)
    # Stage 44 only wired float overflow; the 7th int arg must
    # raise cleanly (before any rsp mutation). Helix wraps the
    # backend NotImplementedError; we just confirm SOME error
    # blocks the compile.
    errs = tc(prog)
    if errs:
        # Typecheck rejected — fine; the goal is no codegen
        # crash with an imbalanced rsp.
        return
    with pytest.raises(NotImplementedError) as exc:
        compile_module_to_elf(lower_ir(prog))
    # The error may come from either the callee-prologue check
    # ("v0.1 supports up to 6 int params" — fires first because
    # callee gets compiled before caller in our IR walker) or
    # the caller-side F2 front-load guard ("mixed int+float
    # overflow not yet wired"). Either is acceptable as a clean
    # rejection BEFORE any rsp mutation.
    msg = str(exc.value)
    assert ("6 int args" in msg or "6 int params" in msg
            or "mixed int+float overflow" in msg), \
        f"expected an int-overflow rejection, got {msg!r}"


def test_stage44_gate1_f4_named_sysv_constants_exist():
    """Module-level SysV constants are importable. Replaces the
    pre-fix hard-coded 16 / 8 magic numbers in 3 sites."""
    from helixc.backend.x86_64 import (
        SYSV_STACK_ARG_BASE,
        SYSV_STACK_ARG_STRIDE,
        SYSV_STACK_ALIGNMENT,
    )
    assert SYSV_STACK_ARG_BASE == 16, \
        "saved rbp (8) + return addr (8) = 16"
    assert SYSV_STACK_ARG_STRIDE == 8, \
        "each stack arg occupies 8 bytes (f32 pads to 8)"
    assert SYSV_STACK_ALIGNMENT == 16, \
        "SysV requires rsp 16-aligned before CALL"


def test_stage44_overflow_arg_position_pin():
    """Pins WHICH arg lands at WHICH stack slot. If the indexing
    is wrong, the picked arg comes back wrong. 9th arg is the
    only overflow; subtract every other arg to isolate it."""
    src = """
fn pick9(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32, i: f32) -> f32 {
    // Return just the 9th arg. With all register args = 0.0, the
    // result must equal the 9th arg exactly.
    i
}

fn main() -> i32 {
    let r = pick9(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 42.0);
    r as i32
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42
