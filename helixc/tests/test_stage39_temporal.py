"""Stage 39 Increment 1+2 — temporal type constructors, eliminators,
and cross-temporal transitions.

Stage 39's first deliverable is temporal kinds (Past / Present /
Future / Eternal). Mirrors Stage 37 tier + Stage 38 frame playbooks
exactly. Phase-0 invariant: TyTemporal wrappers and transitions
lower to identity at IR — the kind wrapper has zero runtime overhead;
the temporal status lives purely in the type system.
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
# Inc 1 — temporal constructors + eliminators
# ============================================================


def test_stage39_inc1_into_past_round_trip():
    src = """
fn main() -> i32 {
    let p: Past<i32> = into_past(42);
    from_past(p)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage39_inc1_into_present_round_trip():
    src = "fn main() -> i32 { from_present(into_present(42)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage39_inc1_into_future_round_trip():
    src = "fn main() -> i32 { from_future(into_future(42)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage39_inc1_into_eternal_round_trip():
    src = "fn main() -> i32 { from_eternal(into_eternal(42)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage39_inc1_from_past_rejects_present():
    """Cross-kind eliminator mistakes fire a typecheck diagnostic."""
    src = """
fn main() -> i32 {
    let p: Present<i32> = into_present(42);
    from_past(p)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Past" in str(e) for e in errs), \
        f"expected Past error, got {[str(e) for e in errs]}"


def test_stage39_inc1_from_present_rejects_future():
    src = """
fn main() -> i32 {
    let f: Future<i32> = into_future(42);
    from_present(f)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Present" in str(e) for e in errs)


def test_stage39_inc1_from_future_rejects_eternal():
    src = """
fn main() -> i32 {
    let e: Eternal<i32> = into_eternal(42);
    from_future(e)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Future" in str(e) for e in errs)


def test_stage39_inc1_from_eternal_rejects_past():
    src = """
fn main() -> i32 {
    let p: Past<i32> = into_past(42);
    from_eternal(p)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Eternal" in str(e) for e in errs)


def test_stage39_inc1_all_12_wrong_kind_combinations():
    """For each (from_X, into_Y) where X != Y, the typechecker must
    raise. 4 kinds × 3 wrong intros = 12 combinations. Symmetric to
    Stage 37's 12-combo tier-mismatch coverage."""
    kinds = ["past", "present", "future", "eternal"]
    expected_label = {
        "past": "Past",
        "present": "Present",
        "future": "Future",
        "eternal": "Eternal",
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
                f"{want!r}, got {[str(e) for e in errs]}"


def test_stage39_inc1_builtins_registered():
    """All 8 new temporal builtins are in _BUILTIN_NAMES."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("into_past", "into_present", "into_future", "into_eternal",
                 "from_past", "from_present", "from_future", "from_eternal"):
        assert name in tc._BUILTIN_NAMES, \
            f"{name} not registered as builtin"


def test_stage39_inc1_wrong_arity_into_diagnostic():
    """Wrong-arity into_* call fires a diagnostic, not silent fall-through."""
    src = "fn main() -> i32 { from_past(into_past(1, 2)) }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("into_past" in str(e) and "1 argument" in str(e)
               for e in errs)


def test_stage39_inc1_wrong_arity_from_diagnostic():
    src = "fn main() -> i32 { from_present(into_present(1), 7) }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("from_present" in str(e) and "1 argument" in str(e)
               for e in errs)


# ============================================================
# Inc 2 — cross-temporal transitions
# ============================================================


def test_stage39_inc2_builtins_registered():
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("to_past", "forecast", "recall_past", "actualize"):
        assert name in tc._BUILTIN_NAMES, \
            f"{name} not registered as builtin"


def test_stage39_inc2_to_past_round_trip():
    """Present -> Past via to_past, then unwrap."""
    src = """
fn main() -> i32 {
    let now: Present<i32> = into_present(42);
    let was: Past<i32> = to_past(now);
    from_past(was)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage39_inc2_forecast_round_trip():
    """Present -> Future via forecast, then unwrap."""
    src = """
fn main() -> i32 {
    let now: Present<i32> = into_present(42);
    let pred: Future<i32> = forecast(now);
    from_future(pred)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage39_inc2_recall_past_round_trip():
    """Past -> Present via recall_past, then unwrap."""
    src = """
fn main() -> i32 {
    let memory: Past<i32> = into_past(42);
    let focus: Present<i32> = recall_past(memory);
    from_present(focus)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage39_inc2_actualize_round_trip():
    """Future -> Present via actualize, then unwrap."""
    src = """
fn main() -> i32 {
    let pred: Future<i32> = into_future(42);
    let realized: Present<i32> = actualize(pred);
    from_present(realized)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage39_inc2_lifecycle_chain_round_trips():
    """Chain Present -> Future -> Present -> Past -> i32 through
    forecast/actualize/to_past/from_past. Identity payload survives
    every Phase-0 transition."""
    src = """
fn main() -> i32 {
    let now: Present<i32> = into_present(42);
    let pred: Future<i32> = forecast(now);
    let realized: Present<i32> = actualize(pred);
    let was: Past<i32> = to_past(realized);
    from_past(was)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage39_inc2_to_past_rejects_past_input():
    """to_past requires Present; Past input must fail typecheck."""
    src = """
fn main() -> i32 {
    let p: Past<i32> = into_past(42);
    from_past(to_past(p))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Present" in str(e) and "to_past" in str(e)
               for e in errs)


def test_stage39_inc2_forecast_rejects_future_input():
    src = """
fn main() -> i32 {
    let f: Future<i32> = into_future(42);
    from_future(forecast(f))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Present" in str(e) and "forecast" in str(e)
               for e in errs)


def test_stage39_inc2_recall_past_rejects_eternal_input():
    src = """
fn main() -> i32 {
    let e: Eternal<i32> = into_eternal(42);
    from_present(recall_past(e))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Past" in str(e) and "recall_past" in str(e)
               for e in errs)


def test_stage39_inc2_actualize_rejects_past_input():
    src = """
fn main() -> i32 {
    let p: Past<i32> = into_past(42);
    from_present(actualize(p))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Future" in str(e) and "actualize" in str(e)
               for e in errs)


def test_stage39_inc2_eternal_never_transitions():
    """Eternal<T> isn't a source of any transition (timeless). Trying
    to recall_past / forecast / to_past / actualize an Eternal value
    must fail typecheck — fail-closed for Phase-0 (a later increment
    may add Eternal-aware transitions if needed)."""
    for fn in ("to_past", "forecast", "recall_past", "actualize"):
        src = f"""
fn main() -> i32 {{
    let e: Eternal<i32> = into_eternal(42);
    from_past({fn}(e))
}}
"""
        prog = parse(src, include_stdlib=True)
        errs = typecheck(prog)
        assert errs, f"{fn}(Eternal<i32>) should reject but didn't"


# ============================================================
# Inc 1 + 2 — IR identity-lowering invariant
# ============================================================


def test_stage39_ir_identity_lowering_all_12():
    """All 12 temporal builtins (8 intro/elim + 4 transitions) lower
    as identity at IR — same exit code as the raw value."""
    builtins_intro = [
        "into_past", "into_present", "into_future", "into_eternal",
    ]
    builtins_elim = [
        "from_past", "from_present", "from_future", "from_eternal",
    ]
    for intro, elim in zip(builtins_intro, builtins_elim):
        src = f"fn main() -> i32 {{ {elim}({intro}(7)) }}"
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        assert _run_elf(elf) == 7, \
            f"identity-lowering broken for {intro}/{elim}"
    transitions = [
        ("forecast",    "into_present", "from_future"),
        ("actualize",   "into_future",  "from_present"),
        ("to_past",     "into_present", "from_past"),
        ("recall_past", "into_past",    "from_present"),
    ]
    for trans, intro, elim in transitions:
        src = f"fn main() -> i32 {{ {elim}({trans}({intro}(11))) }}"
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        assert _run_elf(elf) == 11, \
            f"identity-lowering broken for {trans}"


def test_stage39_ad_pure_registration():
    """All 12 temporal builtins are in AD_KNOWN_PURE_CALLS — required
    for let-erasability inside grad/grad_rev bodies. Mirrors the
    Stage 37/38 AD-pure registrations."""
    from helixc.frontend.autodiff import AD_KNOWN_PURE_CALLS
    for name in ("into_past", "into_present", "into_future", "into_eternal",
                 "from_past", "from_present", "from_future", "from_eternal",
                 "to_past", "forecast", "recall_past", "actualize"):
        assert name in AD_KNOWN_PURE_CALLS, \
            f"{name} must be AD-pure for let-erasability"
