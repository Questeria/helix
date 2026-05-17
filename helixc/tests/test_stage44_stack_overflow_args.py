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

from helixc.backend.x86_64 import compile_module_to_elf
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
