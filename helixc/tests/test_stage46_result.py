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


# ============================================================
# Stage 46 closure gate-2 silent-failure G2-F1 backfill: the
# typed-let path (`let r: Result<i32, i32> = Ok(7)`) was not
# covered by gate-1 F4 because the declared type annotation
# strips the TyUnknown hint that F4 keyed on. Gate-2 fix:
# `_result_constructor_provenance` map records the constructor
# side independently of the type-system metadata so typed-let
# wrong-arm calls are also caught.
# ============================================================


def test_stage46_gate2_unwrap_err_on_typed_ok_rejects():
    """G2-F1: typed-let `let r: Result<i32, i32> = Ok(7);
    unwrap_err(r)` — pre-gate-2 this silently returned 7 (the
    Ok-side payload). Post-fix: reject with name-bound
    constructor provenance."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    unwrap_err(r)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("'r'" in str(e) and "constructed via Ok" in str(e)
               for e in errs), \
        f"typed-let Ok unwrap_err must reject, got " \
        f"{[str(e) for e in errs]}"


def test_stage46_gate2_unwrap_ok_on_typed_err_rejects():
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(13);
    unwrap_ok(r)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("'r'" in str(e) and "constructed via Err" in str(e)
               for e in errs)


def test_stage46_gate2_correct_arm_on_typed_let_still_works():
    """Sanity: typed-let with correct unwrap_ok-after-Ok must
    NOT be rejected. Same surface as dogfood_16."""
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    unwrap_ok(r)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs == [], \
        f"correct-arm unwrap_ok on typed-let Ok must not error, " \
        f"got {[str(e) for e in errs]}"


def test_stage46_gate2_correct_arm_typed_let_err_unwrap_err_works():
    src = """
fn main() -> i32 {
    let r: Result<i32, i32> = Err(13);
    unwrap_err(r)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []


# ============================================================
# Stage 46 closure gate-3 silent-failure backfills:
# G3-F1: mutable reassignment must invalidate stale provenance.
# G3-F2: map_ok / map_err must propagate provenance from their
#        Name source argument.
# ============================================================


def test_stage46_gate3_f1_mutable_reassignment_clears_stale_provenance():
    """G3-F1: `let mut r: Result<i32, i32> = Ok(7); r = Err(99);
    unwrap_ok(r)` — pre-fix silently returned the Err value as
    if it were Ok. Post-fix: reassignment to Err(...) overwrites
    the provenance, so unwrap_ok(r) is now caught."""
    src = """
fn main() -> i32 {
    let mut r: Result<i32, i32> = Ok(7);
    r = Err(99);
    unwrap_ok(r)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("constructed via Err" in str(e) for e in errs), \
        f"reassign Ok->Err must invalidate provenance, got " \
        f"{[str(e) for e in errs]}"


def test_stage46_gate3_f1_mutable_reassignment_reverse_direction():
    """Reverse: Err -> Ok reassignment."""
    src = """
fn main() -> i32 {
    let mut r: Result<i32, i32> = Err(13);
    r = Ok(77);
    unwrap_err(r)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("constructed via Ok" in str(e) for e in errs)


def test_stage46_gate3_f1_correct_arm_after_reassign_works():
    """Sanity: a reassignment + correct-arm unwrap must NOT
    error. Confirms G3-F1 fix preserves valid usage."""
    src = """
fn main() -> i32 {
    let mut r: Result<i32, i32> = Ok(7);
    r = Err(99);
    unwrap_err(r)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []


def test_stage46_gate3_f1_non_constructor_reassign_pops_provenance():
    """`r = some_fn()` should POP the provenance map entry
    (since we can't statically know the new variant), letting
    subsequent unwrap_ok/unwrap_err pass through the TyResult
    check without the wrong-arm reject. Without the pop,
    the stale "ok" provenance from the original let would
    fire a false-positive reject."""
    src = """
fn opaque() -> Result<i32, i32> { Err(0) }
fn main() -> i32 {
    let mut r: Result<i32, i32> = Ok(7);
    r = opaque();
    unwrap_err(r)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    # The opaque function returns Err; unwrap_err is the right
    # arm. Pre-fix: false-positive reject from stale "ok"
    # provenance. Post-fix: no error.
    assert errs == [], \
        f"non-constructor reassign must clear stale provenance, " \
        f"got {[str(e) for e in errs]}"


def test_stage46_gate3_codereview_f1_no_cross_function_provenance_leak():
    """Gate-3 code-review CRITICAL G3-F1 follow-up: the
    `_result_constructor_provenance` map was process-global
    (keyed by bare name, no scope), so a `let r = Ok(7)` in
    fn A would leak the "ok" provenance into fn B's `let r =
    opaque_returning_err()`, falsely rejecting B's
    unwrap_err. Post-fix: any non-direct-constructor let
    pops the entry, so each function starts clean."""
    src = """
@pure
fn make_err() -> Result<i32, i32> { Err(13) }

@pure
fn first() -> i32 {
    let r: Result<i32, i32> = Ok(7);
    unwrap_ok(r)
}

fn main() -> i32 {
    let r: Result<i32, i32> = make_err();
    unwrap_err(r)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    # The make_err() RHS is a fn call (not direct Ok/Err),
    # so the prior "ok" provenance from first() must NOT leak
    # into main(). main's unwrap_err must be accepted.
    assert errs == [], \
        f"cross-function provenance leak must be cleared, " \
        f"got {[str(e) for e in errs]}"


def test_stage46_gate3_f2_map_ok_propagates_provenance():
    """G3-F2: `let r0 = Ok(7); let r = map_ok(r0, 999);
    unwrap_err(r)` — pre-fix the typecheck slipped through
    because the let-RHS matcher only handled direct Ok/Err.
    Post-fix: map_ok propagates the source arm's provenance
    so wrong-arm calls on the result are still caught."""
    src = """
fn main() -> i32 {
    let r0: Result<i32, i32> = Ok(7);
    let r: Result<i32, i32> = map_ok(r0, 999);
    unwrap_err(r)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("constructed via Ok" in str(e) for e in errs), \
        f"map_ok must propagate Ok provenance, got " \
        f"{[str(e) for e in errs]}"


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


def test_stage46_result_of_wrapper_in_let_binding_works_phase0():
    """`Result<Known<i32>, i32>` in a LET-BINDING position works
    in Phase-0 because the expression-lowerer arms handle the
    wrapper-quintet identity-unwrapping (the constructor calls
    `Ok(k)` and `unwrap_ok(r)` are intercepted by the identity-
    pass-through tuple).

    Earlier gate-5 G4-H1 attempted a broad typecheck rejection
    here, but that broke this established Stage 46 composition
    semantics (and the dogfood_16 cross_stack_result probe). The
    rejection was narrowed away. The Phase-0 limit applies only
    to fn-RETURN-type-position Result-of-wrapper, which is pinned
    separately by
    `test_stage48_closure_gate5_g4h1_result_of_wrapper_in_fn_signature_raises_at_ir`.

    Stage 49 will lift the fn-return-type limit too, at which
    point the test_stage48 pin will flip polarity. This Stage 46
    test stays as-is — let-binding Phase-0 path already works."""
    src = """
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    let r: Result<Known<i32>, i32> = Ok(k);
    from_known(unwrap_ok(r))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs == [], \
        f"Result<Known<...>, ...> in let-binding should typecheck " \
        f"clean (Phase-0 supported via expression-lowerer arms). " \
        f"Got: {[str(e) for e in errs]}"


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
