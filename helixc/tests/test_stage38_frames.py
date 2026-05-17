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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
