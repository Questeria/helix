"""Stage 38 Increment 1 — spatial-frame constructor + eliminator
end-to-end runtime tests.

Stage 38's first deliverable is spatial reference frames (World /
Robot / Camera). Inc 1 wires up the type-level scaffolding and IR
identity-lowering, mirroring the Stage 37 tier playbook.

Phase-0 invariant: TyFrame wrappers and cross-frame transitions
lower to identity at IR — the frame wrapper has zero runtime
overhead in Phase-0; the frame lives purely in the type system.
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


def test_stage38_inc1_into_world_returns_world_frame():
    src = """
fn main() -> i32 {
    let f: WorldFrame<i32> = into_world(42);
    from_world(f)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage38_inc1_into_robot_round_trip():
    src = "fn main() -> i32 { from_robot(into_robot(42)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage38_inc1_into_camera_round_trip():
    src = "fn main() -> i32 { from_camera(into_camera(42)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage38_inc1_from_world_rejects_robot_frame():
    """from_world on RobotFrame fires a typecheck diagnostic
    (cross-frame mistakes are caught at compile time)."""
    src = """
fn main() -> i32 {
    let r: RobotFrame<i32> = into_robot(42);
    from_world(r)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("WorldFrame" in str(e) for e in errs), \
        f"expected WorldFrame error, got {[str(e) for e in errs]}"


def test_stage38_inc1_from_robot_rejects_camera_frame():
    src = """
fn main() -> i32 {
    let c: CameraFrame<i32> = into_camera(42);
    from_robot(c)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("RobotFrame" in str(e) for e in errs)


def test_stage38_inc1_from_camera_rejects_world_frame():
    src = """
fn main() -> i32 {
    let w: WorldFrame<i32> = into_world(42);
    from_camera(w)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("CameraFrame" in str(e) for e in errs)


def test_stage38_inc1_builtins_registered():
    """All 6 new frame builtins are in _BUILTIN_NAMES."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("into_world", "into_robot", "into_camera",
                 "from_world", "from_robot", "from_camera"):
        assert name in tc._BUILTIN_NAMES, \
            f"{name} not registered as builtin"


def test_stage38_inc1_all_6_pair_combinations():
    """Cross-frame mismatch coverage: all 6 wrong-pair combinations
    (each from_X rejects 2 wrong intos). Symmetric to Stage 37 Inc 3
    tier-mismatch coverage but smaller (3 frames vs 4 tiers → 6
    wrong pairs vs 12)."""
    pairs = [
        ("from_world", "into_robot", "WorldFrame"),
        ("from_world", "into_camera", "WorldFrame"),
        ("from_robot", "into_world", "RobotFrame"),
        ("from_robot", "into_camera", "RobotFrame"),
        ("from_camera", "into_world", "CameraFrame"),
        ("from_camera", "into_robot", "CameraFrame"),
    ]
    for from_fn, into_fn, expected_want in pairs:
        src = f"fn main() -> i32 {{ {from_fn}({into_fn}(42)) }}"
        prog = parse(src, include_stdlib=True)
        errs = typecheck(prog)
        assert any(expected_want in str(e) for e in errs), \
            f"{from_fn}({into_fn}(42)) should reject with " \
            f"{expected_want!r}, got {[str(e) for e in errs]}"


# Stage 38 Inc 2 — cross-frame transforms (6 pairwise directions).
# All lower to identity at IR; the wrapper-shift tracks intent only.


def test_stage38_inc2_builtins_registered():
    """All 6 new cross-frame transform builtins are in _BUILTIN_NAMES."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("world_to_robot", "robot_to_world",
                 "robot_to_camera", "camera_to_robot",
                 "world_to_camera", "camera_to_world"):
        assert name in tc._BUILTIN_NAMES, \
            f"{name} not registered as builtin"


def test_stage38_inc2_world_to_robot_round_trip_runs():
    """world_to_robot then back via from_robot yields the original
    payload (Phase-0: identity-lowered)."""
    src = """
fn main() -> i32 {
    let w: WorldFrame<i32> = into_world(42);
    let r: RobotFrame<i32> = world_to_robot(w);
    from_robot(r)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == [], \
        f"unexpected errors: {[str(e) for e in typecheck(prog)]}"
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage38_inc2_world_camera_chain_round_trips():
    """Chain WorldFrame → CameraFrame → RobotFrame → WorldFrame
    via 3 cross-frame transforms; identity payload survives."""
    src = """
fn main() -> i32 {
    let w0: WorldFrame<i32> = into_world(42);
    let c: CameraFrame<i32> = world_to_camera(w0);
    let r: RobotFrame<i32> = camera_to_robot(c);
    let w1: WorldFrame<i32> = robot_to_world(r);
    from_world(w1)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage38_inc2_world_to_robot_rejects_robot_input():
    """world_to_robot requires WorldFrame input; RobotFrame must fail
    typecheck."""
    src = """
fn main() -> i32 {
    let r: RobotFrame<i32> = into_robot(42);
    let r2: RobotFrame<i32> = world_to_robot(r);
    from_robot(r2)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("WorldFrame" in str(e) and "world_to_robot" in str(e)
               for e in errs), \
        f"expected world_to_robot WorldFrame error, got {[str(e) for e in errs]}"


def test_stage38_inc2_all_6_transforms_reject_wrong_source():
    """Each cross-frame transform rejects the 2 wrong source frames."""
    cases = [
        ("world_to_robot",  "into_robot",  "WorldFrame"),
        ("world_to_robot",  "into_camera", "WorldFrame"),
        ("robot_to_world",  "into_world",  "RobotFrame"),
        ("robot_to_world",  "into_camera", "RobotFrame"),
        ("robot_to_camera", "into_world",  "RobotFrame"),
        ("robot_to_camera", "into_camera", "RobotFrame"),
        ("camera_to_robot", "into_world",  "CameraFrame"),
        ("camera_to_robot", "into_robot",  "CameraFrame"),
        ("world_to_camera", "into_robot",  "WorldFrame"),
        ("world_to_camera", "into_camera", "WorldFrame"),
        ("camera_to_world", "into_world",  "CameraFrame"),
        ("camera_to_world", "into_robot",  "CameraFrame"),
    ]
    for transform, wrong_into, expected_want in cases:
        src = f"fn main() -> i32 {{ from_world({transform}({wrong_into}(42))) }}"
        prog = parse(src, include_stdlib=True)
        errs = typecheck(prog)
        assert any(expected_want in str(e) and transform in str(e)
                   for e in errs), \
            f"{transform}({wrong_into}(42)) should reject with " \
            f"{expected_want!r} naming {transform}, got {[str(e) for e in errs]}"


# Stage 38 post-Inc-3 audit fix-sweep canaries.
# Three HIGH fixes (type-design H1+H2, silent-failure F1) + 1 f32
# coverage (code-review CR-002).


def test_stage38_postinc3_into_world_wrong_arity_zero_args_diagnoses():
    """Silent-failure F1 (HIGH, conf 95): wrong-arity calls must emit
    a diagnostic, not silently typecheck to TyUnknown and blow up at
    IR lowering. Zero args."""
    src = "fn main() -> i32 { from_world(into_world()) }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    msgs = [str(e) for e in errs]
    assert any("into_world" in m and "takes 1 argument" in m for m in msgs), \
        f"into_world() zero-args must diagnose, got {msgs}"


def test_stage38_postinc3_world_to_robot_wrong_arity_two_args_diagnoses():
    """Silent-failure F1 (HIGH, conf 95): cross-frame transforms also
    catch the wrong-arity case (originally fell through to TyUnknown)."""
    src = """
fn main() -> i32 {
    let w: WorldFrame<i32> = into_world(42);
    from_robot(world_to_robot(w, w))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    msgs = [str(e) for e in errs]
    assert any("world_to_robot" in m and "takes 1 argument" in m
               for m in msgs), \
        f"world_to_robot(w, w) must diagnose 2-arg call, got {msgs}"


def test_stage38_postinc3_from_camera_wrong_arity_zero_args_diagnoses():
    """Silent-failure F1 (HIGH, conf 95): eliminators too."""
    src = "fn main() -> i32 { from_camera() }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    msgs = [str(e) for e in errs]
    assert any("from_camera" in m and "takes 1 argument" in m
               for m in msgs), \
        f"from_camera() must diagnose, got {msgs}"


def test_stage38_postinc3_frame_param_rejects_bare_inner():
    """Type-design H1 (HIGH, conf 90): TyFrame in _compatible
    correctly rejects bare-i32 where WorldFrame<i32> expected."""
    src = """
fn takes_world(w: WorldFrame<i32>) -> i32 { from_world(w) }
fn main() -> i32 { takes_world(42) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs != [], \
        "takes_world(42) — bare i32 passed where WorldFrame<i32> expected — must fail typecheck"


def test_stage38_postinc3_frame_param_rejects_cross_frame():
    """Type-design H1 (HIGH, conf 90): TyFrame in _compatible
    correctly rejects RobotFrame<i32> where WorldFrame<i32>
    expected at the call boundary."""
    src = """
fn takes_world(w: WorldFrame<i32>) -> i32 { from_world(w) }
fn main() -> i32 {
    let r: RobotFrame<i32> = into_robot(42);
    takes_world(r)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs != [], \
        "takes_world(robot_frame) must fail typecheck (cross-frame at call boundary)"


def test_stage38_postinc3_frame_param_same_frame_compatible():
    """Type-design H1 (HIGH, conf 90): TyFrame in _compatible permits
    matching-frame matching-inner at the call boundary (positive
    control for the rejection tests above).

    Typecheck-only: compiling a `WorldFrame<i32>` function-parameter
    signature to ELF would require a TyGeneric arm in
    `_lower_type` — the same gap exists for Stage 37 tier params
    (no compile-to-ELF test exercises `fn(w: WorkingMem<i32>)`). That
    arm is Stage 39+ work, not Stage 38 closure-gate work."""
    src = """
fn takes_world(w: WorldFrame<i32>) -> i32 { from_world(w) }
fn main() -> i32 {
    let w: WorldFrame<i32> = into_world(42);
    takes_world(w)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == [], \
        f"matching WorldFrame<i32> must typecheck, got {[str(e) for e in typecheck(prog)]}"


def test_stage38_postinc3_is_refinement_container_includes_tyframe():
    """Type-design H2 (HIGH, conf 88): TyFrame in
    _is_refinement_container — refinements under a frame wrapper
    are now visible to the refinement-container predicate."""
    from helixc.frontend.typecheck import TyFrame, TyPrim
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    assert tc._is_refinement_container(TyFrame("world", TyPrim("i32"))), \
        "TyFrame must be a refinement container (H2 fix)"


def test_stage38_postinc3_f32_inner_propagates_through_transforms():
    """Code-review CR-002 (MEDIUM, conf 85): zero f32 coverage was
    flagged. This pins that WorldFrame<f32> round-trips through a
    cross-frame transform chain and back."""
    src = """
fn main() -> i32 {
    let wf: WorldFrame<f32> = into_world(1.5);
    let cf: CameraFrame<f32> = world_to_camera(wf);
    let rf: RobotFrame<f32> = camera_to_robot(cf);
    let wf2: WorldFrame<f32> = robot_to_world(rf);
    let v: f32 = from_world(wf2);
    let cmp: i32 = if v == 1.5 { 42 } else { 0 };
    cmp
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == [], \
        f"WorldFrame<f32> chain must typecheck, got {[str(e) for e in typecheck(prog)]}"
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


# Stage 38 post-Inc-3 additional canaries — F2 chain-rule + deeper H2.


def test_stage38_postinc3_forward_ad_through_frame_identity():
    """Silent-failure F2 (MEDIUM, conf 90): grad through frame ops must
    return the inner derivative, not raise opaque-call NotImplementedError.
    Pre-fix the call hit autodiff.py:1058 ('forward-mode AD does not
    support opaque call into_world')."""
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.autodiff import differentiate, fmt
    src = "fn _f(x: f32) -> f32 { from_world(into_world(x)) }"
    prog = parse(src, include_stdlib=True)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    body = fn.body.final_expr
    assert body is not None
    deriv = differentiate(body, "x")
    assert fmt(deriv) == "1", \
        f"d(from_world(into_world(x)))/dx must be 1, got {fmt(deriv)}"


def test_stage38_postinc3_forward_ad_through_cross_frame_chain():
    """F2: cross-frame transforms also identity-AD; constant survives."""
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.autodiff import differentiate, fmt
    src = ("fn _f(x: f32) -> f32 { "
           "from_robot(world_to_robot(into_world(x))) * 2.0 }")
    prog = parse(src, include_stdlib=True)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    body = fn.body.final_expr
    assert body is not None
    deriv = differentiate(body, "x")
    out = fmt(deriv)
    assert "2" in out, \
        f"d(2*from_robot(world_to_robot(into_world(x))))/dx must contain 2, got {out}"


def test_stage38_postinc3_reverse_ad_through_frame_identity():
    """F2 reverse arm: adjoint must flow through frame wrapper."""
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.autodiff import fmt
    from helixc.frontend.autodiff_reverse import differentiate_reverse
    src = "fn f(x: f32) -> f32 { from_world(into_world(x)) }"
    prog = parse(src, include_stdlib=True)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    grads = differentiate_reverse(fn.body, ["x"])
    assert fmt(grads["x"]) == "1", \
        f"grad_rev(from_world(into_world(x)))['x'] must be 1, got {fmt(grads['x'])}"


def test_stage38_postinc3_refined_inner_frame_param_typechecks():
    """Type-design H2 deeper coverage: a function parameter typed as
    WorldFrame<Probability> must accept a WorldFrame<Probability>
    argument. Pre-fix _refinement_shape_exact missed the TyFrame arm,
    so refined-inner frame parameters became uncallable."""
    src = """
type Probability = f64 where 0.0 <= self <= 1.0;

fn takes_world_prob(w: WorldFrame<Probability>) -> WorldFrame<Probability> { w }

fn main() -> i32 {
    let p: Probability = 0.5_f64;
    let w: WorldFrame<Probability> = into_world(p);
    let _ = takes_world_prob(w);
    0
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs == [], \
        f"refined-inner frame param must typecheck post-H2, got {[str(e) for e in errs]}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
