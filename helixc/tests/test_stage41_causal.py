"""Stage 41 Increment 1+2 — causal/intent type constructors,
eliminators, and cross-causal transitions.

Stage 41 deliverable is causal kinds (Cause / Effect / Joint /
Independent). Mirrors Stage 37/38/39/40 playbooks exactly.
Phase-0 invariant: TyCausal wrappers and transitions lower to
identity at IR — zero runtime overhead; causal status lives in
the type system. Composes orthogonally with the 4-stack AGI
quartet completed at Stage 40.
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
# Inc 1 — causal constructors + eliminators
# ============================================================


def test_stage41_inc1_into_cause_round_trip():
    src = """
fn main() -> i32 {
    let c: Cause<i32> = into_cause(42);
    from_cause(c)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage41_inc1_into_effect_round_trip():
    src = "fn main() -> i32 { from_effect(into_effect(42)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage41_inc1_into_joint_round_trip():
    src = "fn main() -> i32 { from_joint(into_joint(42)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage41_inc1_into_independent_round_trip():
    src = "fn main() -> i32 { from_independent(into_independent(42)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage41_inc1_from_cause_rejects_effect():
    src = """
fn main() -> i32 {
    let e: Effect<i32> = into_effect(42);
    from_cause(e)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Cause" in str(e) for e in errs)


def test_stage41_inc1_all_12_wrong_kind_combinations():
    """4 kinds × 3 wrong intros = 12 combinations."""
    kinds = ["cause", "effect", "joint", "independent"]
    expected_label = {
        "cause":       "Cause",
        "effect":      "Effect",
        "joint":       "Joint",
        "independent": "Independent",
    }
    for elim_k in kinds:
        for intro_k in kinds:
            if elim_k == intro_k:
                continue
            from_fn = f"from_{elim_k}"
            into_fn = f"into_{intro_k}"
            want = expected_label[elim_k]
            src = f"fn main() -> i32 {{ {from_fn}({into_fn}(42)) }}"
            prog = parse(src, include_stdlib=True)
            errs = typecheck(prog)
            assert any(want in str(e) for e in errs), \
                f"{from_fn}({into_fn}(42)) should reject with " \
                f"{want!r}"


def test_stage41_inc1_builtins_registered():
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("into_cause", "into_effect", "into_joint",
                 "into_independent",
                 "from_cause", "from_effect", "from_joint",
                 "from_independent"):
        assert name in tc._BUILTIN_NAMES, f"{name} not registered"


# ============================================================
# Inc 2 — causal transitions
# ============================================================


def test_stage41_inc2_propagate_round_trip():
    """Cause -> Effect via propagate."""
    src = """
fn main() -> i32 {
    let c: Cause<i32> = into_cause(42);
    let e: Effect<i32> = propagate(c);
    from_effect(e)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage41_inc2_aggregate_round_trip():
    src = """
fn main() -> i32 {
    let e: Effect<i32> = into_effect(42);
    let j: Joint<i32> = aggregate(e);
    from_joint(j)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage41_inc2_isolate_round_trip():
    src = """
fn main() -> i32 {
    let j: Joint<i32> = into_joint(42);
    let i: Independent<i32> = isolate(j);
    from_independent(i)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage41_inc2_full_lifecycle_chain():
    """Cause -> Effect -> Joint -> Independent -> unwrap."""
    src = """
fn main() -> i32 {
    let c: Cause<i32> = into_cause(42);
    let e: Effect<i32> = propagate(c);
    let j: Joint<i32> = aggregate(e);
    let i: Independent<i32> = isolate(j);
    from_independent(i)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage41_inc2_propagate_rejects_effect_input():
    src = """
fn main() -> i32 {
    let e: Effect<i32> = into_effect(42);
    from_effect(propagate(e))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Cause" in str(e) and "propagate" in str(e)
               for e in errs)


def test_stage41_inc2_builtins_registered():
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("propagate", "aggregate", "isolate"):
        assert name in tc._BUILTIN_NAMES, f"{name} not registered"


# ============================================================
# F1 preemptive cross-causal laundering guard
# ============================================================


def test_stage41_f1_blocks_cross_causal_laundering():
    """Direct `into_X(from_Y(v))` with X != Y must reject."""
    src = """
fn main() -> i32 {
    let e: Effect<i32> = into_effect(42);
    from_cause(into_cause(from_effect(e)))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launders" in str(e) for e in errs)


def test_stage41_f1_allows_self_rewrap_all_4_kinds():
    for kind in ("cause", "effect", "joint", "independent"):
        src = f"""
fn main() -> i32 {{
    let x: {kind.capitalize()}<i32> = into_{kind}(42);
    from_{kind}(into_{kind}(from_{kind}(x)))
}}
"""
        prog = parse(src, include_stdlib=True)
        errs = typecheck(prog)
        assert errs == [], f"{kind}->{kind} self-rewrap must not error"


# ============================================================
# IR identity-lowering + AD-pure invariants
# ============================================================


def test_stage41_ir_identity_lowering_all_11():
    builtins_intro = [
        "into_cause", "into_effect", "into_joint", "into_independent",
    ]
    builtins_elim = [
        "from_cause", "from_effect", "from_joint", "from_independent",
    ]
    for intro, elim in zip(builtins_intro, builtins_elim):
        src = f"fn main() -> i32 {{ {elim}({intro}(7)) }}"
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        assert _run_elf(elf) == 7
    transitions = [
        ("propagate", "into_cause",  "from_effect"),
        ("aggregate", "into_effect", "from_joint"),
        ("isolate",   "into_joint",  "from_independent"),
    ]
    for trans, intro, elim in transitions:
        src = f"fn main() -> i32 {{ {elim}({trans}({intro}(11))) }}"
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        assert _run_elf(elf) == 11


def test_stage41_ad_pure_registration():
    from helixc.frontend.autodiff import AD_KNOWN_PURE_CALLS
    for name in ("into_cause", "into_effect", "into_joint",
                 "into_independent",
                 "from_cause", "from_effect", "from_joint",
                 "from_independent",
                 "propagate", "aggregate", "isolate"):
        assert name in AD_KNOWN_PURE_CALLS, f"{name} must be AD-pure"


def test_stage41_ad_identity_chain_rule_registration():
    from helixc.frontend.autodiff import _FRAME_IDENTITY_AD_NAMES
    for name in ("into_cause", "into_effect", "into_joint",
                 "into_independent",
                 "from_cause", "from_effect", "from_joint",
                 "from_independent",
                 "propagate", "aggregate", "isolate"):
        assert name in _FRAME_IDENTITY_AD_NAMES, \
            f"{name} must be identity-AD-chain-rule"


# ============================================================
# Type-system helper parity (preemptive H1/H3/F2 sweep)
# ============================================================


def test_stage41_h1_causal_compatible_rejects_raw_inner():
    src = """
fn unwrap(x: i32) -> i32 { x }
fn main() -> i32 {
    let c: Cause<i32> = into_cause(42);
    unwrap(c)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs, "Cause<i32> must not be _compatible with raw i32"


def test_stage41_h3_causal_in_refinement_container_set():
    from helixc.frontend.typecheck import (
        TypeChecker, TyCausal, TyPrim,
    )
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    assert tc._is_refinement_container(TyCausal("cause", TyPrim("i32")))


def test_stage41_f2_contains_unknown_walks_causal_wrapper():
    from helixc.frontend.typecheck import (
        TypeChecker, TyCausal, TyUnknown,
    )
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    wrapped = TyCausal("cause", TyUnknown(hint="probe"))
    assert tc._contains_unknown_type(wrapped), \
        "TyUnknown buried under TyCausal must be detected"


# ============================================================
# 5-stack quintet composition (memory + spatial + temporal + modal + causal)
# ============================================================


def test_stage41_causal_composes_with_modal():
    """`Known<Cause<i32>>` = "I directly observed that this was a cause"."""
    src = """
fn main() -> i32 {
    let c: Cause<i32> = into_cause(42);
    let kc: Known<Cause<i32>> = into_known(c);
    from_cause(from_known(kc))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42
