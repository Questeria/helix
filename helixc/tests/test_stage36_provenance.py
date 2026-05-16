"""Stage 36 Increment 1 — end-to-end runtime tests for provenance-typed
primitives.

The typecheck-level tests for `prove()` and `unwrap_logic()` live in
`test_provenance.py` next to the Stage 24 type-level scaffolding. This
file holds the **runtime** complement: programs are compiled to ELF
binaries, executed via WSL, and exit codes are checked.

Phase-0 invariant: `prove(v, src)` and `unwrap_logic(l)` lower to
identity in the IR — the Logic<T> wrapper has zero runtime overhead;
provenance lives purely at the type level. The end-to-end tests
here verify that runtime semantics match this invariant.
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
from helixc.frontend.typecheck import TypeChecker, typecheck
from helixc.ir.lower_ast import lower


def _run_elf(elf: bytes) -> int:
    """Compile + run via WSL, return exit code (low byte)."""
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


def test_stage36_prove_unwrap_roundtrip_runtime():
    """End-to-end: `unwrap_logic(prove(41, 99)) + 1` exits 42.
    Confirms Logic<T> has zero runtime overhead and that the
    round-trip preserves the value bit-for-bit."""
    src = """
fn main() -> i32 {
    let x: i32 = 41;
    let l: Logic<i32> = prove(x, 99);
    unwrap_logic(l) + 1
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs == [], f"unexpected errors: {[str(e) for e in errs]}"
    mod = lower(prog)
    elf = compile_module_to_elf(mod)
    rc = _run_elf(elf)
    assert rc == 42, f"expected exit 42, got {rc}"


def test_stage36_prove_inside_arithmetic_runtime():
    """End-to-end: prove() can be used inline inside arithmetic via
    unwrap_logic. `unwrap_logic(prove(10, 0)) + unwrap_logic(prove(32, 0))`
    exits 42."""
    src = """
fn main() -> i32 {
    unwrap_logic(prove(10, 0)) + unwrap_logic(prove(32, 0))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs == [], f"unexpected errors: {[str(e) for e in errs]}"
    mod = lower(prog)
    elf = compile_module_to_elf(mod)
    rc = _run_elf(elf)
    assert rc == 42, f"expected exit 42, got {rc}"


def test_stage36_prove_unwrap_is_builtin():
    """Registry check: `prove` and `unwrap_logic` are listed as
    builtins so the unbound-name diagnostic doesn't fire on them."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    assert "prove" in tc._BUILTIN_NAMES
    assert "unwrap_logic" in tc._BUILTIN_NAMES


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
