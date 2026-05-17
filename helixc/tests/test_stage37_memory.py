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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
