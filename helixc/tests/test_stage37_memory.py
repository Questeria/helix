"""Stage 37 Increment 1 — tiered memory constructor + eliminator
end-to-end runtime tests.

Stage 37's first deliverable is tiered memory (Working / Episodic /
Semantic / Procedural). Inc 1 wires up the type-level scaffolding
that Stage 24 already shipped (TyMemTier, consolidate, recall) into
runnable typecheck+lower+codegen path.

Phase-0 invariant: TyMemTier wrappers and cross-tier transitions
(into_*, unwrap_*, consolidate, recall) lower to identity at IR —
the tier wrapper has zero runtime overhead in Phase-0; tier lives
purely in the type system. Mirrors the Stage 36 Inc 1 Logic<T>
pattern.
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


def test_stage37_inc1_into_working_returns_working_mem():
    """into_working(v) wraps v in WorkingMem<T>."""
    src = """
fn main() -> i32 {
    let m: WorkingMem<i32> = into_working(42);
    unwrap_working(m)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs == [], f"unexpected errors: {[str(e) for e in errs]}"
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage37_inc1_into_episodic_round_trip():
    src = """
fn main() -> i32 {
    unwrap_episodic(into_episodic(42))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage37_inc1_into_semantic_round_trip():
    src = """
fn main() -> i32 {
    unwrap_semantic(into_semantic(42))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage37_inc1_into_procedural_round_trip():
    src = """
fn main() -> i32 {
    unwrap_procedural(into_procedural(42))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage37_inc1_consolidate_episodic_to_semantic():
    """The existing consolidate() builtin now runs end-to-end
    (was typecheck-only pre-Inc-1)."""
    src = """
fn main() -> i32 {
    let e: EpisodicMem<i32> = into_episodic(42);
    let s: SemanticMem<i32> = consolidate(e);
    unwrap_semantic(s)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage37_inc1_recall_semantic_to_working():
    src = """
fn main() -> i32 {
    let s: SemanticMem<i32> = into_semantic(42);
    let w: WorkingMem<i32> = recall(s);
    unwrap_working(w)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage37_inc1_full_lifecycle():
    """Working → Episodic → consolidate → Semantic → recall →
    Working. Full tier round-trip in one program."""
    src = """
fn main() -> i32 {
    let raw: i32 = 42;
    let e: EpisodicMem<i32> = into_episodic(raw);
    let s: SemanticMem<i32> = consolidate(e);
    let w: WorkingMem<i32> = recall(s);
    unwrap_working(w)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage37_inc1_unwrap_working_rejects_wrong_tier():
    """unwrap_working on EpisodicMem fires a trap-shaped diagnostic
    (the tier-mismatch boundary check from Stage 24 + Inc 1)."""
    src = """
fn main() -> i32 {
    let e: EpisodicMem<i32> = into_episodic(42);
    unwrap_working(e)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("WorkingMem" in str(e) for e in errs), \
        f"expected WorkingMem error, got {[str(e) for e in errs]}"


def test_stage37_inc1_builtins_registered():
    """All 8 new tier builtins are in _BUILTIN_NAMES so unbound-name
    diagnostics don't fire on them."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("into_working", "into_episodic", "into_semantic",
                 "into_procedural", "unwrap_working", "unwrap_episodic",
                 "unwrap_semantic", "unwrap_procedural"):
        assert name in tc._BUILTIN_NAMES, \
            f"{name} not registered as builtin"


# Stage 37 Increment 3 — cross-tier mismatch coverage.
#
# Inc 1 added an unwrap_working-on-EpisodicMem rejection test as the
# representative tier-mismatch case. Inc 3 expands to ALL 12 wrong-pair
# combinations so a tier-collapse regression (e.g., the tier_map dict
# collapsing two tiers to the same string) would be caught immediately
# rather than silently accepted.


def _assert_unwrap_rejects(unwrap_fn: str, into_fn: str,
                            expected_want: str) -> None:
    """Helper: assert unwrap_<X>(into_<Y>(42)) emits a typecheck
    error mentioning <X>Mem<T>."""
    src = f"""
fn main() -> i32 {{
    let m = {into_fn}(42);
    {unwrap_fn}(m)
}}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any(expected_want in str(e) for e in errs), \
        f"{unwrap_fn}({into_fn}(42)) should reject with {expected_want!r}, " \
        f"got {[str(e) for e in errs]}"


def test_stage37_inc3_unwrap_working_rejects_episodic():
    _assert_unwrap_rejects("unwrap_working", "into_episodic", "WorkingMem")


def test_stage37_inc3_unwrap_working_rejects_semantic():
    _assert_unwrap_rejects("unwrap_working", "into_semantic", "WorkingMem")


def test_stage37_inc3_unwrap_working_rejects_procedural():
    _assert_unwrap_rejects("unwrap_working", "into_procedural", "WorkingMem")


def test_stage37_inc3_unwrap_episodic_rejects_working():
    _assert_unwrap_rejects("unwrap_episodic", "into_working", "EpisodicMem")


def test_stage37_inc3_unwrap_episodic_rejects_semantic():
    _assert_unwrap_rejects("unwrap_episodic", "into_semantic", "EpisodicMem")


def test_stage37_inc3_unwrap_episodic_rejects_procedural():
    _assert_unwrap_rejects("unwrap_episodic", "into_procedural", "EpisodicMem")


def test_stage37_inc3_unwrap_semantic_rejects_working():
    _assert_unwrap_rejects("unwrap_semantic", "into_working", "SemanticMem")


def test_stage37_inc3_unwrap_semantic_rejects_episodic():
    _assert_unwrap_rejects("unwrap_semantic", "into_episodic", "SemanticMem")


def test_stage37_inc3_unwrap_semantic_rejects_procedural():
    _assert_unwrap_rejects("unwrap_semantic", "into_procedural", "SemanticMem")


def test_stage37_inc3_unwrap_procedural_rejects_working():
    _assert_unwrap_rejects("unwrap_procedural", "into_working", "ProceduralMem")


def test_stage37_inc3_unwrap_procedural_rejects_episodic():
    _assert_unwrap_rejects("unwrap_procedural", "into_episodic", "ProceduralMem")


def test_stage37_inc3_unwrap_procedural_rejects_semantic():
    _assert_unwrap_rejects("unwrap_procedural", "into_semantic", "ProceduralMem")


def test_stage37_inc3_consolidate_rejects_working():
    """consolidate requires EpisodicMem; WorkingMem is rejected."""
    src = """
fn main() -> i32 {
    let w = into_working(42);
    unwrap_semantic(consolidate(w))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("EpisodicMem" in str(e) for e in errs), \
        f"consolidate(WorkingMem) should reject, got {[str(e) for e in errs]}"


def test_stage37_inc3_recall_rejects_episodic():
    """recall requires SemanticMem; EpisodicMem is rejected."""
    src = """
fn main() -> i32 {
    let e = into_episodic(42);
    unwrap_working(recall(e))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("SemanticMem" in str(e) for e in errs), \
        f"recall(EpisodicMem) should reject, got {[str(e) for e in errs]}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
