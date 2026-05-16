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


# Stage 36 Increment 2 — provenance-composing combinators.


def test_stage36_inc2_combinators_are_builtins():
    """derive, and_logic, or_logic, not_logic are registered."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    assert "derive" in tc._BUILTIN_NAMES
    assert "and_logic" in tc._BUILTIN_NAMES
    assert "or_logic" in tc._BUILTIN_NAMES
    assert "not_logic" in tc._BUILTIN_NAMES


def test_stage36_inc2_and_logic_truth_table():
    """and_logic on 0/1 truth values: 1 AND 1 = 1, 1 AND 0 = 0,
    0 AND 0 = 0."""
    for a, b, want in [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 0)]:
        src = f"""
fn main() -> i32 {{
    unwrap_logic(and_logic(prove({a}, 0), prove({b}, 0)))
}}
"""
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        rc = _run_elf(elf)
        assert rc == want, f"and_logic({a},{b}) -> {rc}, expected {want}"


def test_stage36_inc2_or_logic_truth_table():
    """or_logic on 0/1 truth values: covers all four input cases."""
    for a, b, want in [(1, 1, 1), (1, 0, 1), (0, 1, 1), (0, 0, 0)]:
        src = f"""
fn main() -> i32 {{
    unwrap_logic(or_logic(prove({a}, 0), prove({b}, 0)))
}}
"""
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        rc = _run_elf(elf)
        assert rc == want, f"or_logic({a},{b}) -> {rc}, expected {want}"


def test_stage36_inc2_not_logic_inverts():
    """not_logic: NOT 0 = 1, NOT 1 = 0."""
    for a, want in [(0, 1), (1, 0)]:
        src = f"fn main() -> i32 {{ unwrap_logic(not_logic(prove({a}, 0))) }}"
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        rc = _run_elf(elf)
        assert rc == want, f"not_logic({a}) -> {rc}, expected {want}"


def test_stage36_inc2_derive_keeps_first_parent_value():
    """derive(a, b) returns a's value in Phase-0 single-tag
    provenance. The lattice upgrade tracks both parents."""
    src = """
fn main() -> i32 {
    unwrap_logic(derive(prove(42, 1), prove(7, 2)))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


def test_stage36_inc2_compound_expression():
    """Compound: (1 AND 1) OR (NOT 0) = 1, times 42 = 42."""
    src = """
fn main() -> i32 {
    let t = prove(1, 0);
    let f = prove(0, 0);
    let r = or_logic(and_logic(t, t), not_logic(f));
    unwrap_logic(r) * 42
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


def test_stage36_inc2_and_logic_rejects_bare_value():
    """and_logic with a bare i32 (not Logic) fires trap 24100."""
    src = """
fn main() -> i32 {
    let a = prove(1, 0);
    unwrap_logic(and_logic(a, 1))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("24100" in str(e) for e in errs), \
        f"expected trap 24100, got {[str(e) for e in errs]}"


def test_stage36_inc2_derive_rejects_bare_first_arg():
    """derive(bare, Logic) fires trap 24100."""
    src = """
fn main() -> i32 {
    let b = prove(1, 0);
    unwrap_logic(derive(1, b))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("24100" in str(e) for e in errs), \
        f"expected trap 24100, got {[str(e) for e in errs]}"


# Stage 36 Increment 3 — boolean-algebra completeness.


def test_stage36_inc3_combinators_are_builtins():
    """xor_logic, implies_logic, eq_logic, if_logic, to_logic_bool
    are registered as builtins."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("xor_logic", "implies_logic", "eq_logic",
                 "if_logic", "to_logic_bool"):
        assert name in tc._BUILTIN_NAMES, \
            f"{name} not registered as builtin"


def test_stage36_inc3_xor_truth_table():
    """xor_logic on 0/1 truth values: 0 XOR 0 = 0, 0 XOR 1 = 1,
    1 XOR 0 = 1, 1 XOR 1 = 0."""
    for a, b, want in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        src = f"""
fn main() -> i32 {{
    unwrap_logic(xor_logic(prove({a}, 0), prove({b}, 0)))
}}
"""
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        rc = _run_elf(elf)
        assert rc == want, f"xor_logic({a},{b}) -> {rc}, expected {want}"


def test_stage36_inc3_implies_truth_table():
    """implies_logic: 0->0=1, 0->1=1, 1->0=0, 1->1=1."""
    for a, b, want in [(0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 1)]:
        src = f"""
fn main() -> i32 {{
    unwrap_logic(implies_logic(prove({a}, 0), prove({b}, 0)))
}}
"""
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        rc = _run_elf(elf)
        assert rc == want, f"implies({a},{b}) -> {rc}, expected {want}"


def test_stage36_inc3_eq_truth_table():
    """eq_logic: 0==0 = 1, 0==1 = 0, 1==0 = 0, 1==1 = 1."""
    for a, b, want in [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]:
        src = f"""
fn main() -> i32 {{
    unwrap_logic(eq_logic(prove({a}, 0), prove({b}, 0)))
}}
"""
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        rc = _run_elf(elf)
        assert rc == want, f"eq_logic({a},{b}) -> {rc}, expected {want}"


def test_stage36_inc3_if_logic_selects_then_branch():
    """if_logic(1, then_v, else_v) returns then_v."""
    src = """
fn main() -> i32 {
    unwrap_logic(if_logic(prove(1, 0), prove(42, 0), prove(7, 0)))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


def test_stage36_inc3_if_logic_selects_else_branch():
    """if_logic(0, then_v, else_v) returns else_v."""
    src = """
fn main() -> i32 {
    unwrap_logic(if_logic(prove(0, 0), prove(42, 0), prove(7, 0)))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 7


def test_stage36_inc3_to_logic_bool_lifts():
    """to_logic_bool(x) lifts a bare i32 into Logic<i32>."""
    src = "fn main() -> i32 { unwrap_logic(to_logic_bool(1)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 1


def test_stage36_inc3_de_morgan_law():
    """De Morgan's law: NOT(a AND b) == OR(NOT a, NOT b). If true,
    eq_logic(lhs, rhs) is 1, so output is 1*42 = 42. Verifies a
    real theorem of boolean algebra computed entirely over
    provenance-typed values."""
    src = """
fn main() -> i32 {
    let a = prove(1, 0);
    let b = prove(0, 0);
    let lhs = not_logic(and_logic(a, b));
    let rhs = or_logic(not_logic(a), not_logic(b));
    unwrap_logic(eq_logic(lhs, rhs)) * 42
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


def test_stage36_inc3_xor_rejects_bare_value():
    """xor_logic with a bare i32 fires trap 24100."""
    src = """
fn main() -> i32 {
    unwrap_logic(xor_logic(prove(1, 0), 0))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("24100" in str(e) for e in errs), \
        f"expected trap 24100, got {[str(e) for e in errs]}"


def test_stage36_inc3_if_logic_rejects_bare_cond():
    """if_logic with a bare i32 cond fires trap 24100."""
    src = """
fn main() -> i32 {
    unwrap_logic(if_logic(1, prove(42, 0), prove(7, 0)))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("24100" in str(e) for e in errs), \
        f"expected trap 24100, got {[str(e) for e in errs]}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
