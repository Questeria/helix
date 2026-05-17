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


def test_stage48_closure_gate1_f2_err_constructed_question_lifted_by_stage49_inc4():
    """Gate-1 F2 (HIGH, Stage 48): `r?` on
    `let r: Result<i32, i32> = Err(99)` was rejected at
    typecheck pre-Stage-49 because Phase-0 had no real `?`
    propagation. The reject avoided a silent Err-as-Ok
    miscompile (the operand's static provenance was 'err' but
    the identity-lowering would extract the Err payload as if
    it were Ok).

    Stage 49 Inc 4 LIFTED this reject. `?` now has real runtime
    semantics: COND_BR on RESULT_TAG, RETURN-the-packed-Err if
    tag == 1, fall-through-and-extract-payload if tag == 0.

    Post-Inc-4 the original source typechecks clean AND runs
    correctly: helper returns Err(99) up the call stack, main
    sees Err and exits with the unwrap_err payload."""
    src = """
fn helper() -> Result<i32, i32> {
    let r: Result<i32, i32> = Err(99);
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 {
    unwrap_err(helper())
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == [], \
        f"Err(...)? must typecheck post-Stage-49-Inc-4, got " \
        f"{[str(e) for e in typecheck(prog)]}"
    elf = compile_module_to_elf(lower(prog))
    # helper's `?` propagates Err(99); main extracts 99.
    assert _run_elf(elf) == 99


# ============================================================
# Stage 48 Inc 4 closure gate-2 silent-failure F1+M5 regressions
# ============================================================


def test_stage48_closure_gate2_f1_inner_block_shadow_no_provenance_leak():
    """Gate-2 F1 (HIGH, Stage 48): pre-fix, an inner-block `let r:
    Result<i32, i32> = Ok(5)` overwrote the OUTER `r='err'`
    provenance entry in the flat dict. After the inner block
    exited, the outer `r?` no longer saw 'err' provenance and
    silently extracted the Err payload as Ok (exit code 99
    verified end-to-end in the audit reproducer).

    The Stage 48 gate-2 fix snapshot-restored the provenance map
    across _check_block — preserved the static `err` claim and
    typecheck-rejected the post-block `r?`.

    Stage 49 Inc 4 LIFTED the original reject (real `?` is now
    sound), so the post-block `r?` now typechecks AND runs
    correctly: Inc 4 emits COND_BR → RETURN-Err. The scope-
    aware snapshot-restore from gate-2 still matters for the
    OTHER static-provenance consumers (unwrap_ok / unwrap_err)
    that still rely on the dict for wrong-arm detection."""
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
    unwrap_err(helper())
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    # Post-Stage-49-Inc-4: typecheck clean. The runtime `?`
    # propagates Err(99) up; main extracts 99.
    assert errs == [], \
        f"post-Inc-4 the source must typecheck clean, got " \
        f"{[str(e) for e in errs]}"
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 99


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
    # TODO(stage49): when the runtime Ok/Err tag lands, this
    # test's polarity FLIPS — change `assert errs == []` to
    # compile + run the ELF and `assert _run_elf(elf) == 99`
    # (the natural Err early-return through `?`). DO NOT delete
    # the test; the F5 reproducer is the canonical regression
    # anchor for aggregate-field provenance behaviour. Mirror
    # the TODO(stage49) markers already in lower_ast.py:866 and
    # lower_ast.py:2097.
    assert errs == [], \
        f"F5 deferred: member-access operand to `?` should " \
        f"typecheck clean in Phase-0 (silent miscompile is a " \
        f"known Phase-0 defect — Stage 49+ runtime tag fixes " \
        f"the entire class). Got: {[str(e) for e in errs]}"


# ============================================================
# Stage 48 Inc 4 closure gate-3 silent-failure G3-F1
# (inner-block ASSIGN to outer Result name + post-block `?`)
# ============================================================

# Order-sensitive note on M5 test above (per gate-3 code-review M3):
# the cross-fn carry test requires `maker` to be checked BEFORE
# `taker` so the stale provenance from maker pollutes taker's
# parameter check. The source order is intentional; do not
# reorder fn declarations.


def test_stage48_closure_gate3_g3f1a_inner_block_assign_no_stale_ok_claim():
    """Gate-3 G3-F1 fix: gate-2's snapshot-restore solved the
    inner-block LET shadow but introduced an inner-block ASSIGN
    mirror. `let mut r = Ok(7); { r = Err(99); } let v = r?;`
    pre-fix: in-block assign-arm correctly popped 'r' from the
    dict, then the restore put back the stale 'ok' provenance.
    Post-block `r?` then DID NOT REJECT (silent miscompile
    at runtime: exit 99).

    Post-fix: the restore detects outer-name mutation (current
    dict differs from saved snapshot) and DROPS the mutated
    name. Result: typecheck clean (no static provenance →
    F1-dynamic Phase-0 limitation territory), no false static
    'Ok-constructed' claim. Runtime exit 99 remains a Phase-0
    known defect (F6 deferred — joins F1-dynamic, fixed by
    Stage 49 runtime tag).

    The test asserts the post-fix typecheck behaviour: NO
    diagnostic claiming the operand is statically Ok-constructed
    (the pre-fix bug surfaced as the absence of any rejection,
    so we assert positively that the diagnostic that WOULD
    indicate the stale claim is not present)."""
    src = """
fn helper() -> Result<i32, i32> {
    let mut r: Result<i32, i32> = Ok(7);
    { r = Err(99); }
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 { unwrap_ok(helper()) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    err_strs = [str(e) for e in errs]
    # Post-fix: no static "Ok-constructed" claim on r.
    assert not any("constructed via Ok" in s for s in err_strs), \
        f"inner-block assign must drop stale 'ok' provenance, " \
        f"got: {err_strs}"


def test_stage48_closure_gate3_g3f1b_if_then_assign_no_stale_ok_claim():
    """Gate-3 G3-F1 mirror in if-then arm. Same defect class
    as G3-F1a, different scope vehicle. Pre-fix: silent miscompile
    via stale 'ok' restoration. Post-fix: F1-dynamic territory."""
    src = """
fn helper(cond: bool) -> Result<i32, i32> {
    let mut r: Result<i32, i32> = Ok(7);
    if cond { r = Err(99); } else { }
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 { unwrap_ok(helper(true)) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    err_strs = [str(e) for e in errs]
    assert not any("constructed via Ok" in s for s in err_strs), \
        f"if-then-arm assign must drop stale 'ok' provenance, " \
        f"got: {err_strs}"


def test_stage48_closure_gate3_g3f1c_match_arm_assign_no_stale_ok_claim():
    """Gate-3 G3-F1 mirror in match arm. Same defect class as
    G3-F1a + G3-F1b, third scope vehicle. Pre-fix: silent
    miscompile. Post-fix: F1-dynamic territory."""
    src = """
fn helper(b: bool) -> Result<i32, i32> {
    let mut r: Result<i32, i32> = Ok(7);
    match b {
        true => { r = Err(99); },
        false => { },
    }
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 { unwrap_ok(helper(true)) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    err_strs = [str(e) for e in errs]
    assert not any("constructed via Ok" in s for s in err_strs), \
        f"match-arm assign must drop stale 'ok' provenance, " \
        f"got: {err_strs}"


# ============================================================
# Stage 48 Inc 4 closure gate-3 code-review M2 polish:
# tighten the non-Result operand diagnostic regression test
# ============================================================


def test_stage48_question_diagnostic_names_operand():
    """Gate-3 code-review M2: the gate-2 M1 polish added the
    operand name to the non-Result-operand diagnostic. The pre-
    existing `test_stage48_rejects_non_result_operand` only
    asserted the strings '?' and 'Result' are present — a
    future refactor that strips the `f" on {expr.args[0].name!r}"`
    interpolation would silently revert the polish and that
    test would still pass. This test pins the operand-name
    inclusion."""
    src = """
fn helper() -> Result<i32, i32> {
    let x: i32 = 7;
    let v: i32 = x?;
    Ok(v)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs, "non-Result operand `?` must produce diagnostic"
    diag_str = " ".join(str(e) for e in errs)
    assert "on 'x'" in diag_str, \
        f"diagnostic must name the operand `x`, got: {diag_str}"


# ============================================================
# Stage 48 Inc 4 closure gate-5 fix sweep — verifies the
# fixes for G4-F1 (match-arm bare-Assign scope leak), G4-F2
# (ASSIGN-then-LET-shadow on same name), G4-H1 (Result<wrapper,
# _> typecheck-side rejection), G4-H2 (expression-form if-else
# arm scope leak), and G4-M3 (match-guard scope leak).
# ============================================================


def test_stage48_closure_gate5_g4f1_match_arm_bare_assign_no_stale_claim():
    """Gate-5 G4-F1: match-arm bare-expression bodies bypass
    _check_block. Pre-fix the Assign-arm mutated the provenance
    dict directly; the last arm's mutation would "win" silently
    and a post-match `?` accepted under stale provenance. Post-
    fix (the `_check_expr_in_block_scope` helper wrapping arm.body),
    the inner-block ASSIGN cases drop to F1-dynamic Phase-0
    territory — typecheck-clean (no false 'ok' claim), runtime
    exit 99 acknowledged as F1 deferral.

    The audit verified exit 99 on the reproducer; this test pins
    the typecheck-side discipline (no false static accept of a
    post-match `?` that follows arm-body Assign mutations)."""
    src = """
fn helper(x: i32) -> Result<i32, i32> {
    let mut r: Result<i32, i32> = Ok(7);
    match x {
        0 => r = Err(99),
        _ => r = Ok(7),
    };
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 { unwrap_ok(helper(0)) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    err_strs = [str(e) for e in errs]
    assert not any("constructed via Ok" in s for s in err_strs), \
        f"match-arm bare Assign must drop stale 'ok' provenance, " \
        f"got: {err_strs}"


def test_stage48_closure_gate5_g4f2_assign_then_let_shadow_no_stale_claim():
    """Gate-5 G4-F2: ASSIGN-then-LET-shadow on the same name.
    Pre-fix the let-shadow added the name to inner_lets, which
    then masked the prior Assign's mutation at restore — outer's
    stale 'ok' survived and a post-block `?` accepted under it,
    producing a silent runtime exit 99.

    Post-fix (parallel `_result_assigns_block_scopes` per-event
    mask): the assign event is recorded regardless of subsequent
    let-shadow. At restore, the saved-name-in-assigns branch fires
    first and drops the stale outer entry. Test pins the absence
    of the false static `constructed via Ok` claim."""
    src = """
fn helper() -> Result<i32, i32> {
    let mut r: Result<i32, i32> = Ok(11);
    {
        r = Err(99);
        let r: Result<i32, i32> = Ok(5);
        let _x: i32 = unwrap_ok(r);
    };
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 { unwrap_ok(helper()) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    err_strs = [str(e) for e in errs]
    assert not any("constructed via Ok" in s for s in err_strs), \
        f"ASSIGN-then-LET-shadow must drop stale 'ok' provenance, " \
        f"got: {err_strs}"


def test_stage48_closure_gate5_g4h1_result_of_wrapper_in_fn_signature_raises_at_ir():
    """Gate-5 G4-H1 (audit Option 3 — LIFTED at Stage 49 Inc 1):
    Result whose Ok or Err side is a Stage 37-41 wrapper-quintet
    in a FUNCTION SIGNATURE position used to raise
    NotImplementedError at IR lowering, because the pre-49
    `_lower_type` Result-arm identity-recurses into the wrapper
    which has no type-position arm.

    Stage 49 Inc 1 lifts this limitation: the Result-arm now
    short-circuits to a packed i64 (tag<<32 | payload) without
    recursing into the Ok/Err inner types. The wrapper inner
    (`Known<i32>`) only appears in EXPRESSION position
    (`into_known(42)`), where the existing identity-lowering arm
    handles it. The packed i32 payload then flows through
    RESULT_PACK / RESULT_PAYLOAD unchanged.

    Stage 49 Inc 4+ will revisit wider payload sizes (Known<i64>,
    etc.) and may reintroduce a limitation for wrappers around
    >32-bit inner types — but for the Phase-0 i32 payload constraint,
    this case now compiles AND links clean."""
    # Ok-side wrapper in fn-return-type position
    src_ok = """
fn ret_known() -> Result<Known<i32>, i32> {
    let k: Known<i32> = into_known(42);
    Ok(k)
}
fn main() -> i32 { 0 }
"""
    prog = parse(src_ok, include_stdlib=True)
    # Typecheck stays clean.
    assert typecheck(prog) == [], \
        "Result<Known<...>, ...> in fn-signature should still " \
        "typecheck clean post-Stage-49-Inc-1."
    # IR lowering now succeeds — the Result arm packs into i64 and
    # never recurses into the Known<i32> type-position.
    m = lower(prog)
    assert "ret_known" in m.functions, \
        "ret_known should lower successfully post-Stage-49-Inc-1"


def test_stage48_closure_gate5_g4h2_if_else_expr_form_no_scope_leak():
    """Gate-5 G4-H2 (medium-high): expression-form if/else arms
    bypass _check_block. Pre-fix an Assign inside an expression-
    form else branch permanently mutated the outer provenance
    dict, producing either a false-reject or a silent miscompile
    depending on which arm assigned. Post-fix (the helper wraps
    expression-form else), the if-expr is a scope vehicle and the
    outer dict is restored after the branches.

    This test exercises the cleaner shape: an if-then-else where
    one arm assigns to a sentinel and the other doesn't. The
    post-if `?` must NOT see a false 'ok' claim from the unmodified
    arm."""
    src = """
fn helper(b: bool) -> Result<i32, i32> {
    let mut r: Result<i32, i32> = Ok(13);
    if b { r = Err(99); } else { r = Ok(7) };
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 { unwrap_ok(helper(true)) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    err_strs = [str(e) for e in errs]
    assert not any("constructed via Ok" in s for s in err_strs), \
        f"if-else expression-form arms must not leave stale 'ok' " \
        f"claim, got: {err_strs}"


def test_stage48_closure_gate5_lifecycle_clears_at_check_entry():
    """Gate-5 G4-M1: `_result_let_block_scopes` and the new
    `_result_assigns_block_scopes` are cleared at `check()` entry
    (parallel to `_result_constructor_provenance`). Verifies the
    defense-in-depth: a TypeChecker instance whose prior check()
    invocation got an exception escape mid-_check_block would
    leak frames into the next check() without the explicit reset.

    Test pattern: re-run check() on the same TypeChecker instance
    twice after manually pre-poisoning the stacks. Post-fix the
    stacks come back empty."""
    src = """
fn main() -> i32 { 0 }
"""
    prog = parse(src, include_stdlib=True)
    tc = TypeChecker(prog)
    # Pre-poison
    tc._result_let_block_scopes = [{"poison"}, {"poison2"}]
    tc._result_assigns_block_scopes = [{"poison"}]
    tc._result_constructor_provenance = {"poison": "ok"}
    tc.check()
    assert tc._result_let_block_scopes == [], \
        f"let-block scopes must reset at check() entry, " \
        f"got: {tc._result_let_block_scopes}"
    assert tc._result_assigns_block_scopes == [], \
        f"assigns-block scopes must reset at check() entry, " \
        f"got: {tc._result_assigns_block_scopes}"
    # The prov dict also clears (already gate-2 M5):
    assert "poison" not in tc._result_constructor_provenance
