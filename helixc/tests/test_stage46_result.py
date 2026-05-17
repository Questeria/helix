"""Stage 46 — Result<T, E> typecheck-side scaffolding (Tier 4 #14 Inc 1).

First two-parameter wrapper family in the Helix type system.
Phase-0: identity-lowered at IR (no runtime tag yet). The
`?` operator and real runtime tag are Stage 47+ work.
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


# ============================================================
# Inc 1 — Result<T, E> typecheck + IR identity-lowering
# ============================================================


def test_stage46_ok_unwrap_round_trip():
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(42);
    unwrap_ok(r)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage46_err_unwrap_round_trip():
    """Err(e) constructs a Result with err_ty=typeof(e); unwrap_err
    extracts that inner."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(13);
    unwrap_err(r)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 13


def test_stage46_gate1_f1_is_ok_rejects_in_phase_0():
    """CRITICAL gate-1 fix: pre-fix is_ok always returned 1
    silently miscompiling any `if is_err(r) { panic(...) }` —
    the user thought they had error handling; the compiled
    code ALWAYS took the else branch. Post-fix: typecheck
    rejects until Stage 48+ runtime tag lands."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    if is_ok(r) { 1 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("is_ok" in str(e)
               and ("no runtime semantics" in str(e)
                    or "statically" in str(e)
                    or "Phase-0" in str(e)
                    or "Stage 48" in str(e)) for e in errs), \
        f"is_ok must typecheck-reject in Phase-0, got {[str(e) for e in errs]}"


def test_stage46_gate1_f1_is_err_rejects_in_phase_0():
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    if is_err(r) { 1 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("is_err" in str(e) for e in errs)


def test_stage46_map_ok_replaces_inner():
    """map_ok(r, new_v) returns Result with new_v as the Ok side."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    unwrap_ok(map_ok(r, 99))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 99


def test_stage46_gate1_f2_map_err_rejects_in_phase_0():
    """HIGH gate-1 fix: pre-fix `map_err(r, 999)` silently
    returned the original Result, so `unwrap_err(map_err(r,
    999))` on Ok(5) returned 5 instead of 999. Post-fix:
    typecheck rejects until Stage 48+ runtime tag lands."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(5);
    unwrap_err(map_err(r, 999))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("map_err" in str(e) for e in errs)


def test_stage46_gate1_f4_unwrap_err_on_ok_inferred_rejects():
    """MEDIUM gate-1 fix: `let r = Ok(7); unwrap_err(r)` was
    silently accepted because Ok(v) sets err_ty to TyUnknown
    (universally compatible). Post-fix: detect the
    'Err inferred' provenance and reject."""
    src = """
fn main() -> i32 {
    let r = Ok(7);
    unwrap_err(r)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("constructed via Ok" in str(e) for e in errs), \
        f"unwrap_err on Ok-constructed must reject, got {[str(e) for e in errs]}"


def test_stage46_gate1_f4_unwrap_ok_on_err_inferred_rejects():
    src = """
fn main() -> i32 {
    let r = Err(13);
    unwrap_ok(r)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("constructed via Err" in str(e) for e in errs)


def test_stage46_unwrap_ok_rejects_non_result():
    """unwrap_ok requires Result<T, E>; a bare i32 must reject."""
    src = "fn main() -> i32 { unwrap_ok(42) }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Result<T, E>" in str(e) for e in errs), \
        f"expected Result-required diag, got {[str(e) for e in errs]}"


def test_stage46_result_arity_wrong_one_arg():
    """Result<T> (1 arg) must reject with arity diagnostic."""
    src = "fn foo() -> Result<i32> { panic(\"x\") } fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Result<T, E> takes 2 type arguments" in str(e)
               for e in errs), \
        f"expected arity diag, got {[str(e) for e in errs]}"


def test_stage46_result_arity_wrong_three_args():
    src = "fn foo() -> Result<i32, i32, i32> { panic(\"x\") } fn main() -> i32 { 0 }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Result<T, E> takes 2 type arguments" in str(e)
               for e in errs)


def test_stage46_result_composes_with_modal():
    """`Result<Known<i32>, i32>` — Phase-0 composition probe.
    Stage 40 Modal wrapped inside Stage 46 Result, both
    identity-lowered."""
    src = """
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    let r: Result<Known<i32>, i32> = Ok(k);
    from_known(unwrap_ok(r))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage46_builtins_registered():
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("Ok", "Err", "unwrap_ok", "unwrap_err",
                 "is_ok", "is_err", "map_ok", "map_err"):
        assert name in tc._BUILTIN_NAMES, \
            f"{name} not registered as builtin"


def test_stage46_ok_wrong_arity_diagnostic():
    src = "fn main() -> i32 { unwrap_ok(Ok(1, 2)) }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Ok() takes 1 argument" in str(e) for e in errs)


def test_stage46_map_ok_wrong_arity():
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    unwrap_ok(map_ok(r))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("map_ok() takes 2 arguments" in str(e) for e in errs)


def test_stage46_tyresult_dataclass_exists():
    """TyResult dataclass must be importable + have ok_ty/err_ty."""
    from helixc.frontend.typecheck import TyResult, TyPrim
    r = TyResult(ok_ty=TyPrim("i32"), err_ty=TyPrim("bool"))
    assert r.ok_ty == TyPrim("i32")
    assert r.err_ty == TyPrim("bool")


def test_stage46_compatible_rejects_swapped_ok_err():
    """Result<i32, str> vs Result<str, i32> must NOT be compatible.
    Both inners must agree."""
    src = """
fn foo() -> Result<i32, bool> { Ok(0) }
fn take(r: Result<bool, i32>) -> i32 { 0 }
fn main() -> i32 { take(foo()) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs, "swapped ok/err must reject"
