"""Stage 40 Increment 1+2 — modal/epistemic type constructors,
eliminators, and cross-modal transitions.

Stage 40's deliverable is modal kinds (Known / Believed / Goal /
Uncertain). Mirrors Stage 37 tier + Stage 38 frame + Stage 39
temporal playbooks exactly. Phase-0 invariant: TyModal wrappers
and transitions lower to identity at IR — the kind wrapper has
zero runtime overhead; the epistemic status lives purely in the
type system. Real-world AGI reasoning needs to track WHY a fact
is accepted; treating a goal as a known fact (a category mistake
at the heart of many AI safety failures) is caught at compile
time.
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
# Inc 1 — modal constructors + eliminators
# ============================================================


def test_stage40_inc1_into_known_round_trip():
    src = """
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    from_known(k)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage40_inc1_into_believed_round_trip():
    src = "fn main() -> i32 { from_believed(into_believed(42)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage40_inc1_into_goal_round_trip():
    src = "fn main() -> i32 { from_goal(into_goal(42)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage40_inc1_into_uncertain_round_trip():
    src = "fn main() -> i32 { from_uncertain(into_uncertain(42)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage40_inc1_from_known_rejects_believed():
    """Cross-kind eliminator mistakes fire a typecheck diagnostic.

    This is the core AI-safety invariant for Stage 40: a Believed
    value cannot be unwrapped as Known. Treating an inference as
    a directly-observed fact is the category mistake the type
    system is meant to catch.
    """
    src = """
fn main() -> i32 {
    let b: Believed<i32> = into_believed(42);
    from_known(b)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Known" in str(e) for e in errs), \
        f"expected Known error, got {[str(e) for e in errs]}"


def test_stage40_inc1_from_believed_rejects_goal():
    """Treating a goal as if it's a belief is also rejected
    (category mistake: hopeful thinking vs evidence)."""
    src = """
fn main() -> i32 {
    let g: Goal<i32> = into_goal(42);
    from_believed(g)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Believed" in str(e) for e in errs)


def test_stage40_inc1_from_goal_rejects_uncertain():
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(42);
    from_goal(u)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Goal" in str(e) for e in errs)


def test_stage40_inc1_from_uncertain_rejects_known():
    src = """
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    from_uncertain(k)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Uncertain" in str(e) for e in errs)


def test_stage40_inc1_all_12_wrong_kind_combinations():
    """For each (from_X, into_Y) where X != Y, the typechecker must
    raise. 4 kinds × 3 wrong intros = 12 combinations. Symmetric to
    Stage 37's tier / Stage 39's temporal coverage."""
    kinds = ["known", "believed", "goal", "uncertain"]
    expected_label = {
        "known":     "Known",
        "believed":  "Believed",
        "goal":      "Goal",
        "uncertain": "Uncertain",
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


def test_stage40_inc1_builtins_registered():
    """All 8 new modal builtins are in _BUILTIN_NAMES."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("into_known", "into_believed", "into_goal", "into_uncertain",
                 "from_known", "from_believed", "from_goal", "from_uncertain"):
        assert name in tc._BUILTIN_NAMES, \
            f"{name} not registered as builtin"


def test_stage40_inc1_wrong_arity_into_diagnostic():
    src = "fn main() -> i32 { from_known(into_known(1, 2)) }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("into_known" in str(e) and "1 argument" in str(e)
               for e in errs)


def test_stage40_inc1_wrong_arity_from_diagnostic():
    src = "fn main() -> i32 { from_believed(into_believed(1), 7) }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("from_believed" in str(e) and "1 argument" in str(e)
               for e in errs)


def test_stage40_inc1_zero_args_into_diagnostic():
    src = "fn main() -> i32 { from_goal(into_goal()) }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("into_goal" in str(e) and "1 argument" in str(e)
               for e in errs)


def test_stage40_inc1_zero_args_from_diagnostic():
    src = "fn main() -> i32 { from_uncertain() }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("from_uncertain" in str(e) and "1 argument" in str(e)
               for e in errs)


# ============================================================
# Inc 2 — cross-modal transitions (epistemic upgrades)
# ============================================================


def test_stage40_inc2_builtins_registered():
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("confirm", "act_on"):
        assert name in tc._BUILTIN_NAMES, \
            f"{name} not registered as builtin"


def test_stage40_inc2_confirm_round_trip():
    """Believed -> Known via confirm, then unwrap."""
    src = """
fn main() -> i32 {
    let b: Believed<i32> = into_believed(42);
    let k: Known<i32> = confirm(b);
    from_known(k)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage40_inc2_act_on_round_trip():
    """Goal -> Known via act_on, then unwrap."""
    src = """
fn main() -> i32 {
    let g: Goal<i32> = into_goal(42);
    let k: Known<i32> = act_on(g);
    from_known(k)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage40_inc2_confirm_rejects_known_input():
    """confirm requires Believed; Known input must fail typecheck."""
    src = """
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    from_known(confirm(k))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Believed" in str(e) and "confirm" in str(e)
               for e in errs)


def test_stage40_inc2_confirm_rejects_goal_input():
    src = """
fn main() -> i32 {
    let g: Goal<i32> = into_goal(42);
    from_known(confirm(g))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Believed" in str(e) and "confirm" in str(e)
               for e in errs)


def test_stage40_inc2_confirm_rejects_uncertain_input():
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(42);
    from_known(confirm(u))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Believed" in str(e) and "confirm" in str(e)
               for e in errs)


def test_stage40_inc2_act_on_rejects_believed_input():
    """act_on requires Goal; Believed input must fail typecheck."""
    src = """
fn main() -> i32 {
    let b: Believed<i32> = into_believed(42);
    from_known(act_on(b))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Goal" in str(e) and "act_on" in str(e)
               for e in errs)


def test_stage40_inc2_act_on_rejects_known_input():
    src = """
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    from_known(act_on(k))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Goal" in str(e) and "act_on" in str(e)
               for e in errs)


def test_stage40_inc2_act_on_rejects_uncertain_input():
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(42);
    from_known(act_on(u))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Goal" in str(e) and "act_on" in str(e)
               for e in errs)


def test_stage40_inc2_wrong_arity_transition_two_args_diagnostic():
    src = """
fn main() -> i32 {
    let b: Believed<i32> = into_believed(1);
    from_known(confirm(b, 7))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("confirm" in str(e) and "1 argument" in str(e)
               for e in errs)


def test_stage40_inc2_zero_arity_transition_diagnostic():
    src = "fn main() -> i32 { from_known(confirm()) }"
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("confirm" in str(e) and "1 argument" in str(e)
               for e in errs)


def test_stage40_inc2_lifecycle_chain_round_trip():
    """Realistic decision-loop chain: a Goal is acted on and becomes
    Known; independently a Believed value is confirmed and becomes
    Known; their sum survives unwrapping. Identity payload preserved
    through every Phase-0 transition."""
    src = """
fn main() -> i32 {
    let g: Goal<i32> = into_goal(20);
    let achieved: Known<i32> = act_on(g);
    let b: Believed<i32> = into_believed(22);
    let verified: Known<i32> = confirm(b);
    from_known(achieved) + from_known(verified)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


# ============================================================
# Stage 40 H1/H2/H3 backfills — TyModal parallels TyTemporal /
# TyFrame in refinement-traversal surfaces. Added preemptively so
# the audit gates don't flag the same lessons Stage 39 had to
# learn the hard way.
# ============================================================


def test_stage40_h1_modal_compatible_rejects_raw_inner():
    """`_compatible(Known<i32>, i32)` must reject — otherwise the
    eliminator's type-level intent is bypassed at call boundaries."""
    src = """
fn unwrap(x: i32) -> i32 { x }
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    unwrap(k)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs, "Known<i32> must not be _compatible with raw i32"


def test_stage40_h1_modal_compatible_rejects_cross_kind():
    """`_compatible(Known<i32>, Believed<i32>)` must reject."""
    src = """
fn take_believed(b: Believed<i32>) -> i32 { from_believed(b) }
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    take_believed(k)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs, "Known<i32> must not be _compatible with Believed<i32>"


def test_stage40_h3_modal_in_refinement_container_set():
    """TyModal must be in `_is_refinement_container` so the join
    logic at `_join_branch_types` correctly fires the refinement-
    shape check on modal-wrapped values."""
    from helixc.frontend.typecheck import (
        TypeChecker, TyModal, TyPrim,
    )
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    assert tc._is_refinement_container(TyModal("known", TyPrim("i32"))), \
        "TyModal must be in the refinement-container set"


# ============================================================
# Inc 1 + 2 — IR identity-lowering invariant
# ============================================================


def test_stage40_ir_identity_lowering_all_10():
    """All 10 modal builtins (8 intro/elim + 2 transitions) lower as
    identity at IR — same exit code as the raw value."""
    builtins_intro = [
        "into_known", "into_believed", "into_goal", "into_uncertain",
    ]
    builtins_elim = [
        "from_known", "from_believed", "from_goal", "from_uncertain",
    ]
    for intro, elim in zip(builtins_intro, builtins_elim):
        src = f"fn main() -> i32 {{ {elim}({intro}(7)) }}"
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        assert _run_elf(elf) == 7, \
            f"identity-lowering broken for {intro}/{elim}"
    transitions = [
        ("confirm", "into_believed", "from_known"),
        ("act_on",  "into_goal",     "from_known"),
    ]
    for trans, intro, elim in transitions:
        src = f"fn main() -> i32 {{ {elim}({trans}({intro}(11))) }}"
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        assert _run_elf(elf) == 11, \
            f"identity-lowering broken for {trans}"


def test_stage40_ad_pure_registration():
    """All 10 modal builtins are in AD_KNOWN_PURE_CALLS — required
    for let-erasability inside grad/grad_rev bodies. Mirrors the
    Stage 37/38/39 AD-pure registrations."""
    from helixc.frontend.autodiff import AD_KNOWN_PURE_CALLS
    for name in ("into_known", "into_believed", "into_goal", "into_uncertain",
                 "from_known", "from_believed", "from_goal", "from_uncertain",
                 "confirm", "act_on"):
        assert name in AD_KNOWN_PURE_CALLS, \
            f"{name} must be AD-pure for let-erasability"


def test_stage40_frame_identity_ad_registration():
    """All 10 modal builtins are in _FRAME_IDENTITY_AD_NAMES — the
    forward + reverse AD passes treat them as identity chain rules
    so `grad(into_known(u))/dx = du/dx`. Mirrors Stage 38/39
    preemptive registration; closes the Stage 38 post-Inc-3 F2
    lesson before audit time."""
    from helixc.frontend.autodiff import _FRAME_IDENTITY_AD_NAMES
    for name in ("into_known", "into_believed", "into_goal", "into_uncertain",
                 "from_known", "from_believed", "from_goal", "from_uncertain",
                 "confirm", "act_on"):
        assert name in _FRAME_IDENTITY_AD_NAMES, \
            f"{name} must be in _FRAME_IDENTITY_AD_NAMES"


# ============================================================
# F2/F6 wrapper-walk: TyUnknown buried under TyModal must be
# detected by `_contains_unknown_type`. Symmetric to Stage 39's
# F2 backfill for TyTemporal.
# ============================================================


def test_stage40_f2_contains_unknown_walks_modal_wrapper():
    from helixc.frontend.typecheck import (
        TypeChecker, TyModal, TyUnknown,
    )
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    wrapped = TyModal("known", TyUnknown(hint="probe"))
    assert tc._contains_unknown_type(wrapped), \
        "TyUnknown buried under TyModal must be detected"


def test_stage40_h3_erase_refinement_walks_modal():
    """`_erase_refinement(TyModal(known, TyRefined(...)))` walks
    into the inner and strips. Mirrors Stage 39's H3 backfill."""
    from helixc.frontend.typecheck import (
        TypeChecker, TyModal, TyRefined, TyPrim,
    )
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    refined = TyRefined("PosI32", TyPrim("i32"), ())
    wrapped = TyModal("known", refined)
    erased = tc._erase_refinement(wrapped)
    assert isinstance(erased, TyModal)
    assert isinstance(erased.inner, TyPrim), \
        f"erase should strip TyRefined under TyModal, got " \
        f"{type(erased.inner).__name__}"


# ============================================================
# Cross-stage composition: modal kinds compose with temporal
# kinds at the type level naturally. `Known<Past<i32>>` = "I
# directly observed this past fact" vs `Believed<Past<i32>>` =
# "I inferred this past fact". Both should typecheck and run.
# ============================================================


def test_stage40_compose_known_past_round_trip():
    """Known<Past<i32>> round-trips through both wrapper layers."""
    src = """
fn main() -> i32 {
    let p: Past<i32> = into_past(42);
    let kp: Known<Past<i32>> = into_known(p);
    let unwrapped_p: Past<i32> = from_known(kp);
    from_past(unwrapped_p)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage40_compose_believed_past_cross_kind_still_rejects():
    """A `Believed<Past<i32>>` cannot be unwrapped as Known<Past>.
    Stage 40's type discipline survives composition with Stage 39's
    temporal wrappers."""
    src = """
fn main() -> i32 {
    let p: Past<i32> = into_past(42);
    let bp: Believed<Past<i32>> = into_believed(p);
    let kp: Past<i32> = from_known(bp);
    from_past(kp)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Known" in str(e) for e in errs), \
        f"Believed<Past<i32>> unwrapped as Known should reject, " \
        f"got {[str(e) for e in errs]}"
