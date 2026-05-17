"""Stage 48 — `?` propagation operator (Tier 4 #14 Inc 2).

Parser desugars `expr?` to a reserved-builtin call `__try(expr)`.
Typecheck validates that:
  - the operand is a Result<T, E1>;
  - the enclosing function returns Result<U, E2>;
  - E1 is compatible with E2 (Err-type fits the propagation slot);
  - the expression's type is the operand's Ok inner.

IR lowering: identity-lowered (Phase-0). Without a runtime Ok/Err
tag, every Result is observationally Ok-shape, so `r?` reduces
to extracting the Ok inner — semantically identical to
`unwrap_ok(r)` until Stage 49 lands the runtime tag + real
conditional-branch IR.

Test coverage:
  - happy path: `Ok(7)?` returns 7;
  - parse-desugar check: AST shows `__try(...)` call;
  - typecheck rejections: non-Result operand; non-Result return
    type; mismatched Err type;
  - composition: chained `?` and `?` inside arithmetic.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from helixc.backend.x86_64 import compile_module_to_elf
from helixc.frontend import ast_nodes as A
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
# Inc 1 — parser desugar
# ============================================================


def _first_body_expr(prog, fn_name: str = "main"):
    fn = next(it for it in prog.items
              if isinstance(it, A.FnDecl) and it.name == fn_name)
    block = fn.body
    # The body is a Block; return its last expression (the implicit
    # tail) or — for tests where the `?` is in a let RHS — the let's
    # value. Simplest: walk through statements; the test source uses
    # a single-statement body shape.
    return block


def test_stage48_inc1_parse_desugars_question_to_try_call():
    """`expr?` parses as Call(callee=Name('__try'), args=[expr])."""
    src = """
fn helper() -> Result<i32, i32> {
    let r: Result<i32, i32> = Ok(7);
    let v: i32 = r?;
    Ok(v)
}
"""
    prog = parse(src, include_stdlib=True)
    fn = next(it for it in prog.items
              if isinstance(it, A.FnDecl) and it.name == "helper")
    # find the `let v: i32 = r?` statement
    let_v = next(s for s in fn.body.stmts
                 if isinstance(s, A.Let) and s.name == "v")
    rhs = let_v.value
    assert isinstance(rhs, A.Call), (
        f"expected Call from `?` desugar, got {type(rhs).__name__}")
    assert isinstance(rhs.callee, A.Name)
    assert rhs.callee.name == "__try", (
        f"expected callee __try, got {rhs.callee.name!r}")
    assert len(rhs.args) == 1
    assert isinstance(rhs.args[0], A.Name)
    assert rhs.args[0].name == "r"


def test_stage48_inc1_builtin_registered():
    """`__try` is in the BUILTIN_NAMES set."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    assert "__try" in tc._BUILTIN_NAMES, \
        "__try must be registered as a builtin name"


# ============================================================
# Inc 2 — typecheck happy path + rejections
# ============================================================


def test_stage48_happy_path_typecheck_clean():
    """Result-returning fn with a single `?` typechecks clean."""
    src = """
fn helper() -> Result<i32, i32> {
    let r: Result<i32, i32> = Ok(7);
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 {
    unwrap_ok(helper())
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs == [], f"expected clean typecheck, got {[str(e) for e in errs]}"


def test_stage48_rejects_non_result_operand():
    """`x?` on a non-Result operand must typecheck-reject."""
    src = """
fn helper() -> Result<i32, i32> {
    let x: i32 = 7;
    let v: i32 = x?;
    Ok(v)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("?" in str(e) and "Result" in str(e) for e in errs), \
        f"non-Result `?` must reject, got {[str(e) for e in errs]}"


def test_stage48_rejects_non_result_return_type():
    """`r?` in a fn whose return type is NOT Result must reject."""
    src = """
fn helper() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    let v: i32 = r?;
    v
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("?" in str(e)
               and "return type" in str(e)
               for e in errs), \
        f"`?` outside Result-returning fn must reject, " \
        f"got {[str(e) for e in errs]}"


def test_stage48_rejects_mismatched_err_type():
    """Operand Err type must be compatible with the function's
    Err type. Pre-fix: silent miscompile risk once Stage 49
    branching is live — the propagated Err would have the wrong
    type wrt the fn signature."""
    src = """
fn helper() -> Result<i32, bool> {
    let r: Result<i32, i32> = Ok(7);
    let v: i32 = r?;
    Ok(v)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("?" in str(e)
               and ("Err" in str(e) or "mismatch" in str(e))
               for e in errs), \
        f"Err-type mismatch on `?` must reject, " \
        f"got {[str(e) for e in errs]}"


def test_stage48_rejects_at_top_level_non_result_main():
    """`r?` directly in `fn main() -> i32` must reject — main's
    return type is i32, not Result."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    let v: i32 = r?;
    v
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("?" in str(e) for e in errs), \
        f"`?` in non-Result main must reject, got {[str(e) for e in errs]}"


# ============================================================
# Inc 3 — IR identity-lowering (Phase-0)
# ============================================================


def test_stage48_inc3_phase0_runtime_returns_ok_inner():
    """Phase-0 stance: every Result is shape-Ok at runtime
    (no tag yet), so `r?` extracts the Ok inner — identical
    to `unwrap_ok(r)`. Stage 49 will add the runtime tag and
    real propagation."""
    src = """
fn helper() -> Result<i32, i32> {
    let r: Result<i32, i32> = Ok(42);
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 {
    unwrap_ok(helper())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage48_inc3_chained_question_marks():
    """Two `?` in a row both lower as identity in Phase-0;
    the final unwrapped value is the deepest Ok inner."""
    src = """
fn inner() -> Result<i32, i32> {
    Ok(13)
}
fn middle() -> Result<i32, i32> {
    let v: i32 = inner()?;
    Ok(v)
}
fn main() -> i32 {
    unwrap_ok(middle())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 13


def test_stage48_inc3_arithmetic_around_question():
    """`r? + 1` uses the Ok inner of r as an operand. Phase-0
    identity lowering preserves the arithmetic."""
    src = """
fn helper() -> Result<i32, i32> {
    let r: Result<i32, i32> = Ok(40);
    let v: i32 = r? + 2;
    Ok(v)
}
fn main() -> i32 {
    unwrap_ok(helper())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


# ============================================================
# Constructor-provenance allowance (intentionally permissive)
# ============================================================


def test_stage48_closure_gate1_f2_err_constructed_question_rejects():
    """Gate-1 F2 fix (HIGH): `r?` on `let r: Result<i32, i32>
    = Err(99)` silently extracted the Err payload as Ok in
    Phase-0 (no runtime tag, identity-lowered). Stage 49+
    will add real propagation — but Phase-0 must REJECT to
    avoid silent miscompilation. Same defect class as Stage
    46 G2-F1's unwrap_ok-on-typed-Err. Mirrors the constructor-
    provenance check at unwrap_ok/unwrap_err."""
    src = """
fn helper() -> Result<i32, i32> {
    let r: Result<i32, i32> = Err(99);
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 {
    unwrap_ok(helper())
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    # Stage 48 gate-1 F2: must reject with a `?`-specific
    # diagnostic mentioning the constructor provenance.
    try_errs = [e for e in errs
                if "?" in str(e) and "constructed via Err" in str(e)]
    assert try_errs, \
        f"`Err(...)?` must reject with provenance diag, " \
        f"got {[str(e) for e in try_errs]}"


# ============================================================
# Stage 48 Inc 4 closure gate-2 silent-failure F1+M5 regressions
# ============================================================


def test_stage48_closure_gate2_f1_inner_block_shadow_no_provenance_leak():
    """Gate-2 F1 fix (HIGH): pre-fix, an inner-block `let r:
    Result<i32, i32> = Ok(5)` overwrote the OUTER `r='err'`
    provenance entry in the flat dict. After the inner block
    exited, the outer `r?` no longer saw 'err' provenance and
    silently extracted the Err payload as Ok (exit code 99
    verified end-to-end in the audit reproducer).

    Post-fix, _check_block snapshots the provenance map at
    entry and restores at exit. Inner-block shadows can't bleed
    outer provenance."""
    src = """
fn helper() -> Result<i32, i32> {
    let r: Result<i32, i32> = Err(99);
    let dummy: i32 = {
        let r: Result<i32, i32> = Ok(5);
        unwrap_ok(r)
    };
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 {
    unwrap_ok(helper())
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    # The outer `r?` MUST reject with the Err-provenance
    # diagnostic. Pre-fix this would typecheck clean and compile
    # to exit 99.
    try_errs = [e for e in errs
                if "?" in str(e) and "constructed via Err" in str(e)]
    assert try_errs, \
        f"inner-block shadow must not leak outer Err " \
        f"provenance, got errors: {[str(e) for e in errs]}"


def test_stage48_closure_gate2_m5_cross_fn_no_provenance_carry():
    """Gate-2 M5 fix (MEDIUM/false-reject): pre-fix, fn A's
    `let r = Ok(7)` set the provenance dict to {r: 'ok'} and
    the entry survived into fn B. fn B's parameter named `r`
    inherited the stale 'ok' provenance, so `unwrap_err(r)`
    on B's parameter FALSELY rejected as 'Ok-constructed'.

    Post-fix, _check_fn clears the provenance map at function
    entry. Per-fn locals must not leak across the fn boundary."""
    src = """
fn maker() -> Result<i32, i32> {
    let r: Result<i32, i32> = Ok(7);
    Ok(unwrap_ok(r))
}
fn taker(r: Result<i32, i32>) -> i32 {
    unwrap_err(r)
}
fn main() -> i32 {
    taker(Err(33))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    # taker(r) has a parameter `r` with NO statically-known
    # provenance. unwrap_err on it MUST typecheck cleanly
    # (the runtime panic for actual Ok at this position is a
    # Phase-0 F1 limitation, but the static check should not
    # false-reject on a parameter).
    err_strs = [str(e) for e in errs]
    assert not any("constructed via Ok" in s for s in err_strs), \
        f"cross-fn provenance carry must not false-reject " \
        f"parameter `r`, got: {err_strs}"


def test_stage48_closure_gate2_f5_member_access_documented_as_phase0_defect():
    """Gate-2 F5 (HIGH but DEFERRED): `let p: Pair = Pair {
    a: Err(99), b: 1 }; p.a?` typechecks clean and silently
    extracts the Err payload at runtime (exit code 99 verified).

    Same defect class as F1 dynamic-Err `?` from the gate-1
    audit: aggregate-field access is fundamentally a dynamic
    operand from the per-name provenance map's perspective.
    Stage 49+ runtime tag eliminates the entire class.

    This test asserts the current (Phase-0) behavior: typecheck
    PASSES. When Stage 49 lands the runtime tag, this test
    will need to be updated (the `?` will then early-return the
    Err naturally — no static rejection needed)."""
    src = """
struct Pair { a: Result<i32, i32>, b: i32 }
fn helper() -> Result<i32, i32> {
    let p: Pair = Pair { a: Err(99), b: 1 };
    let v: i32 = p.a?;
    Ok(v)
}
fn main() -> i32 {
    unwrap_ok(helper())
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    # Phase-0 limitation: typechecks clean. Documented as F5.
    assert errs == [], \
        f"F5 deferred: member-access operand to `?` should " \
        f"typecheck clean in Phase-0 (silent miscompile is a " \
        f"known Phase-0 defect — Stage 49+ runtime tag fixes " \
        f"the entire class). Got: {[str(e) for e in errs]}"
