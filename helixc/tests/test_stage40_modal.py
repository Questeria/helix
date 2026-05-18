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
    """All 10 modal builtins are in _IDENTITY_AD_CHAIN_RULE_NAMES — the
    forward + reverse AD passes treat them as identity chain rules
    so `grad(into_known(u))/dx = du/dx`. Mirrors Stage 38/39
    preemptive registration; closes the Stage 38 post-Inc-3 F2
    lesson before audit time."""
    from helixc.frontend.autodiff import _IDENTITY_AD_CHAIN_RULE_NAMES
    for name in ("into_known", "into_believed", "into_goal", "into_uncertain",
                 "from_known", "from_believed", "from_goal", "from_uncertain",
                 "confirm", "act_on"):
        assert name in _IDENTITY_AD_CHAIN_RULE_NAMES, \
            f"{name} must be in _IDENTITY_AD_CHAIN_RULE_NAMES"


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


# ============================================================
# Stage 40 Inc 4 closure gate-1 fix-sweep regression tests.
# F1 (HIGH conf 90): `into_X(from_uncertain(...))` for any
# upgrade target X in {Known, Believed, Goal} must be rejected.
# Otherwise the entire epistemic-upgrade discipline is bypassed
# by a trivial unwrap-rewrap idiom and the AI-safety motivation
# for Stage 40 evaporates. F2 (MEDIUM conf 90): a user fn with
# the same name as a reserved builtin is silently dead-coded
# (typecheck dispatches builtin arm first) — Stage 40 makes
# this acute because `confirm` and `act_on` are generic verbs
# likely to collide with planner / state-machine code.
# ============================================================


def test_stage40_gate1_f1_rejects_uncertain_to_known_laundering():
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(42);
    let k: Known<i32> = into_known(from_uncertain(u));
    from_known(k)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launders" in str(e) and "Uncertain" in str(e)
               and "Known" in str(e) for e in errs), \
        f"Uncertain->Known laundering must be rejected, got " \
        f"{[str(e) for e in errs]}"


def test_stage40_gate1_f1_rejects_uncertain_to_believed_laundering():
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(42);
    let b: Believed<i32> = into_believed(from_uncertain(u));
    from_believed(b)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launders" in str(e) and "Believed" in str(e)
               for e in errs)


def test_stage40_gate1_f1_rejects_uncertain_to_goal_laundering():
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(42);
    let g: Goal<i32> = into_goal(from_uncertain(u));
    from_goal(g)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launders" in str(e) and "Goal" in str(e)
               for e in errs)


def test_stage40_gate1_f1_allows_uncertain_self_rewrap():
    """The F1 guard only triggers on upgrade-target rewraps; a
    `into_uncertain(from_uncertain(u))` (kind-preserving) round
    trip is benign and must remain allowed."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(42);
    let u2: Uncertain<i32> = into_uncertain(from_uncertain(u));
    from_uncertain(u2)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs == [], \
        f"Uncertain->Uncertain self-rewrap should be allowed, " \
        f"got {[str(e) for e in errs]}"


def test_stage40_gate1_f1_allows_known_self_rewrap():
    """Same-kind rewraps for the non-Uncertain kinds also stay
    allowed (the F1 guard is specifically about laundering AWAY
    from Uncertain)."""
    src = """
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    let k2: Known<i32> = into_known(from_known(k));
    from_known(k2)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []


def test_stage40_gate1_f2_rejects_user_fn_named_confirm():
    src = """
fn confirm(x: i32) -> i32 { x * 2 }
fn main() -> i32 { confirm(21) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("confirm" in str(e) and "shadows a reserved builtin"
               in str(e) for e in errs), \
        f"user fn 'confirm' must be rejected as builtin shadow, " \
        f"got {[str(e) for e in errs]}"


def test_stage40_gate1_f2_rejects_user_fn_named_act_on():
    src = """
fn act_on(x: i32) -> i32 { x + 1 }
fn main() -> i32 { act_on(41) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("act_on" in str(e) and "shadows a reserved builtin"
               in str(e) for e in errs)


def test_stage40_gate1_f2_rejects_user_fn_named_into_known():
    """The shadow guard covers all 10 Stage 40 modal builtins,
    not just the two short-named transition verbs."""
    src = """
fn into_known(x: i32) -> i32 { x }
fn main() -> i32 { into_known(42) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("into_known" in str(e) and "shadows a reserved builtin"
               in str(e) for e in errs)


def test_stage40_gate1_f2_allows_unrelated_user_fn_names():
    """The shadow guard must NOT mass-reject every user fn; only
    those whose name actually appears in _BUILTIN_NAMES."""
    src = """
fn double(x: i32) -> i32 { x * 2 }
fn make_known(raw: i32) -> Known<i32> { into_known(raw) }
fn main() -> i32 { from_known(make_known(double(21))) }
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []


def test_stage40_gate1_f2_shadow_diagnostic_includes_rename_hint():
    """The diagnostic must include a hint pointing the user at the
    reserved-name list so the rename is obvious. Without the hint,
    a user hitting this error has no fast path to resolution."""
    src = """
fn confirm(x: i32) -> i32 { x }
fn main() -> i32 { confirm(0) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    shadow_errs = [e for e in errs
                   if "shadows a reserved builtin" in str(e)]
    assert shadow_errs, "expected shadow diagnostic"
    err_text = str(shadow_errs[0])
    assert ("into_*" in err_text or "from_*" in err_text
            or "confirm" in err_text), \
        f"shadow diagnostic must hint at reserved-name pattern, " \
        f"got: {err_text}"


# ============================================================
# Stage 40 closure gate-2 F1 backfill: cross-modal laundering.
# Generalizes the gate-1 Uncertain-only guard to all
# `into_X(from_Y(...))` pairs where X != Y.
# ============================================================


def test_stage40_gate2_f1_blocks_believed_to_known_laundering():
    """`into_known(from_believed(b))` must reject — Phase-0 has
    `confirm` as the audited Believed -> Known transition."""
    src = """
fn main() -> i32 {
    let b: Believed<i32> = into_believed(42);
    from_known(into_known(from_believed(b)))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launders" in str(e) and "confirm" in str(e)
               for e in errs), \
        f"into_known(from_believed(b)) must reject with `confirm` " \
        f"hint, got {[str(e) for e in errs]}"


def test_stage40_gate2_f1_blocks_goal_to_known_laundering():
    """`into_known(from_goal(g))` must reject — Phase-0 has `act_on`
    as the audited Goal -> Known transition."""
    src = """
fn main() -> i32 {
    let g: Goal<i32> = into_goal(42);
    from_known(into_known(from_goal(g)))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launders" in str(e) and "act_on" in str(e)
               for e in errs)


def test_stage40_gate2_f1_blocks_known_to_believed_downgrade():
    """`into_believed(from_known(k))` is an epistemic downgrade.
    Phase-0 deliberately defers downgrades; typechecker enforces."""
    src = """
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    from_believed(into_believed(from_known(k)))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launders" in str(e) and "no Known -> Believed" in str(e)
               for e in errs)


def test_stage40_gate2_f1_blocks_known_to_uncertain_laundering():
    src = """
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    from_uncertain(into_uncertain(from_known(k)))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launders" in str(e) for e in errs)


def test_stage40_gate2_f1_allows_all_4_self_rewraps():
    """`into_X(from_X(v))` is benign (same kind in, same kind out);
    no laundering. All 4 kinds tested."""
    for kind in ("known", "believed", "goal", "uncertain"):
        src = f"""
fn main() -> i32 {{
    let x: {kind.capitalize()}<i32> = into_{kind}(42);
    from_{kind}(into_{kind}(from_{kind}(x)))
}}
"""
        prog = parse(src, include_stdlib=True)
        errs = typecheck(prog)
        assert errs == [], \
            f"{kind}->{kind} self-rewrap must not error, got " \
            f"{[str(e) for e in errs]}"


def test_stage40_gate2_f1_all_12_cross_modal_combinations_reject():
    """4 modal kinds × 3 wrong sources = 12 cross-modal laundering
    combinations — all must reject."""
    kinds = ["known", "believed", "goal", "uncertain"]
    for target in kinds:
        for source in kinds:
            if source == target:
                continue
            src = f"""
fn main() -> i32 {{
    let s: {source.capitalize()}<i32> = into_{source}(7);
    from_{target}(into_{target}(from_{source}(s)))
}}
"""
            prog = parse(src, include_stdlib=True)
            errs = typecheck(prog)
            assert any("launders" in str(e) for e in errs), \
                f"into_{target}(from_{source}(...)) must reject, " \
                f"got {[str(e) for e in errs]}"


# ============================================================
# Stage 40 closure gate-2 H2 + M1 + audit-trail backfills.
# ============================================================


def test_stage40_gate2_h2_shadowing_emits_one_diagnostic_not_three():
    """H2 backfill: pre-fix, a user fn that shadows a builtin
    fired 1 shadow error AT THE FN-DECL + 1 builtin per-call-site
    wrong-type error per call. Post-fix only the shadow error
    fires; call-sites fall through to user-fn dispatch."""
    src = """
fn confirm(x: i32) -> i32 { x * 2 }
fn main() -> i32 { confirm(21) }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    shadow_errs = [e for e in errs
                   if "shadows a reserved builtin" in str(e)]
    builtin_errs = [e for e in errs
                    if "requires Believed" in str(e)
                    or "requires Goal" in str(e)
                    or "requires Known" in str(e)
                    or "requires Uncertain" in str(e)]
    assert len(shadow_errs) == 1, \
        f"expected 1 shadow error, got {len(shadow_errs)}: " \
        f"{[str(e) for e in shadow_errs]}"
    assert len(builtin_errs) == 0, \
        f"expected 0 builtin per-call-site false-positives " \
        f"after H2 fix, got {len(builtin_errs)}: " \
        f"{[str(e) for e in builtin_errs]}"


def test_stage40_gate2_m1_no_false_laundering_when_inner_malformed():
    """M1 backfill: when the inner `from_X(...)` rejects its arg
    (wrong-kind input), F1 must NOT also fire the laundering
    diagnostic — no value was ever wrapped, so "launders" would
    mislead the user away from the real bug."""
    src = """
fn main() -> i32 {
    let k: Known<i32> = into_known(42);
    from_known(into_known(from_uncertain(k)))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    inner_errs = [e for e in errs
                  if "from_uncertain" in str(e)
                  and "Uncertain" in str(e)]
    launder_errs = [e for e in errs
                    if "launders" in str(e)]
    assert len(inner_errs) >= 1, \
        "inner from_uncertain(Known<i32>) must diagnose: got " \
        f"{[str(e) for e in errs]}"
    assert len(launder_errs) == 0, \
        f"M1 fix: F1 must not fire when inner is TyUnknown; got " \
        f"{[str(e) for e in launder_errs]}"


def test_stage40_f1_let_bypass_closed_by_stage52_taint_tracking():
    """Stage 40 closure gate-1 H1 had documented the F1 syntactic-
    guard limitation as "let-binding decomposes the inline pattern
    and bypasses the guard. Phase-0 known limitation; a Stage-41+
    taint-tracking pass would close this by propagating Uncertain-
    origin through bindings."

    Stage 52 Inc 1 CLOSED that limitation. The new
    `_modal_origin_provenance` map (parallel to Stage 46's
    `_result_constructor_provenance`) records when a var is bound
    to a `from_X(...)` call. The F1 launder guard at `into_Y(...)`
    sites consults the map at the Name operand branch. The
    let-binding bypass now produces a diagnostic naming the
    laundering pattern + the legitimate epistemic-upgrade hint
    (same structure as the inline-form diagnostic)."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(42);
    let raw: i32 = from_uncertain(u);
    let k: Known<i32> = into_known(raw);
    from_known(k)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    launder_errs = [
        e for e in errs
        if "launders" in str(e) and "Uncertain" in str(e)
        and "Known" in str(e)
    ]
    assert launder_errs, \
        f"Stage 52 Inc 1: let-binding bypass of F1 must now be " \
        f"caught with a launder diagnostic naming Uncertain → " \
        f"Known. Got: {[str(e) for e in errs]}"
    # Diagnostic should mention 'taint-tracking' to distinguish
    # from the inline-form (which has its own diagnostic shape).
    # Gate-2 code-review H1 fix: was 'let-binding bypass' which
    # mis-attributed the source for match-arm/Assign/while/if
    # paths. New wording covers all five entry points.
    assert any("taint-tracking" in str(e) for e in launder_errs), \
        f"diagnostic must distinguish taint-tracking path from " \
        f"inline form, got: {[str(e) for e in launder_errs]}"


def test_stage40_gate3_f1_shadow_suppression_includes_gpu_index_dispatch():
    """Gate-3 type-design F1 backfill: H2 dispatch suppression must
    apply uniformly across ALL early-fire builtin dispatch sites,
    not just the modal/temporal/frame/tier family. Pre-fix, the
    GPU-index dispatch (thread_idx / block_idx / block_dim and
    their _x/_y/_z variants) silently shadowed the user fn even
    when _register_fn had flagged it. Post-fix, the shadow
    diagnostic is the ONLY error (no "only allowed inside
    @kernel" cascade per call site)."""
    src = """
fn thread_idx() -> i32 { 7 }
fn main() -> i32 { thread_idx() }
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    shadow_errs = [e for e in errs
                   if "shadows a reserved builtin" in str(e)]
    kernel_cascade_errs = [
        e for e in errs
        if "only allowed inside @kernel" in str(e)
    ]
    assert len(shadow_errs) == 1, \
        f"expected 1 shadow error for fn thread_idx, got " \
        f"{len(shadow_errs)}: {[str(e) for e in shadow_errs]}"
    assert len(kernel_cascade_errs) == 0, \
        f"expected 0 kernel-cascade errors after gate-3 F1 fix, " \
        f"got {len(kernel_cascade_errs)}: " \
        f"{[str(e) for e in kernel_cascade_errs]}"


def test_stage40_gate2_medium1_typechecker_reentrancy_no_stale_shadows():
    """MEDIUM-1 backfill: re-running a TypeChecker instance must
    not carry stale `_shadowed_builtin_names` entries from a
    previous program. Pre-fix, lazy hasattr init left the set
    populated across check() invocations, false-suppressing
    builtin dispatch for non-shadowed callsites in the second
    program. Post-fix, __init__ + check() both clear the set."""
    from helixc.frontend.typecheck import TypeChecker

    # First program: shadows `confirm`.
    prog1 = parse(
        "fn confirm(x: i32) -> i32 { x }\n"
        "fn main() -> i32 { confirm(0) }",
        include_stdlib=False,
    )
    tc1 = TypeChecker(prog1)
    errs1 = tc1.check()
    assert any("shadows a reserved builtin" in str(e)
               and "confirm" in str(e) for e in errs1)
    assert "confirm" in tc1._shadowed_builtin_names

    # Second program with a fresh TypeChecker: does NOT shadow
    # confirm; the builtin should dispatch normally.
    prog2 = parse(
        "fn main() -> i32 {\n"
        "    let b: Believed<i32> = into_believed(7);\n"
        "    let k: Known<i32> = confirm(b);\n"
        "    from_known(k)\n"
        "}",
        include_stdlib=False,
    )
    tc2 = TypeChecker(prog2)
    errs2 = tc2.check()
    # A fresh TypeChecker starts with an empty shadow set.
    assert "confirm" not in tc2._shadowed_builtin_names
    # And the builtin `confirm` dispatches normally — no errors.
    assert errs2 == [], \
        f"fresh TypeChecker must not inherit shadow state, got " \
        f"{[str(e) for e in errs2]}"


# ============================================================
# Stage 52 closure gate-2 code-review M1 fix: regression pins
# for Inc 2+3 paths (each was only shell-probed during closure
# — now properly pinned so a future refactor catches breakage)
# ============================================================


def test_stage52_inc2_high3_while_loop_assign_caught():
    """HIGH-3: `while ... { r = from_uncertain(u); } into_known(r)`
    must fire the launder diagnostic. Inc 2 added Assign-arm
    POPULATE; this pins it."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let mut r: i32 = 0;
    let mut i: i32 = 0;
    while i < 1 { r = from_uncertain(u); i = i + 1; };
    let k: Known<i32> = into_known(r);
    from_known(k)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launder" in str(e) and "Uncertain" in str(e)
               and "Known" in str(e) for e in errs), \
        f"while-loop Assign launder must fire, got: " \
        f"{[str(e) for e in errs]}"


def test_stage52_inc2_f1e_inner_let_shadow_does_not_leak():
    """F1e: outer `let r: i32 = 0; { let r = from_uncertain(u); }
    let _ = into_known(r);` — outer r is i32, never tainted.
    Inner let-shadow installs taint that must NOT leak out."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let r: i32 = 5;
    let dummy: i32 = {
        let r: i32 = from_uncertain(u);
        r + 1
    };
    let k: Known<i32> = into_known(r);
    from_known(k) + dummy
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    # Outer `r` was never tainted; into_known(r) must NOT fire.
    assert not any("launder" in str(e) for e in errs), \
        f"outer untainted r must not trigger launder, got: " \
        f"{[str(e) for e in errs]}"


def test_stage52_inc3_match_arm_parallel_union_caught():
    """HIGH-1: match-arm Assign launder caught via parallel-union.
    Pre-Inc-3, arm 2's `r = 0` popped arm 1's installed taint
    (sequential override). Post-fix, parallel-union snapshots
    pre-match, restores between arms, unions arm-results."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let mut r: i32 = 0;
    match true { true => { r = from_uncertain(u); }, false => { r = 0; } };
    let k: Known<i32> = into_known(r);
    from_known(k)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launder" in str(e) and "Uncertain" in str(e)
               and "Known" in str(e) for e in errs), \
        f"match-arm Assign launder must fire, got: " \
        f"{[str(e) for e in errs]}"


def test_stage52_gate2_high_c_multi_kind_drop_on_conflict():
    """HIGH-C: when 2 arms install different modal kinds for the
    same name, drop the static claim (the static claim is
    invalidated; consult falls through to no-taint). Pre-fix,
    "first wins" silently picked one arm's kind, falsely passing
    into_X for the other kind.

    Post-fix, drop-on-conflict means into_X(r) doesn't fire on
    the conflict case (the safe-conservative behavior). True
    multi-kind diagnostic is deferred to Inc 4."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let kw: Known<i32> = into_known(1);
    let mut r: i32 = 0;
    match true { true => { r = from_uncertain(u); }, false => { r = from_known(kw); } };
    // r has conflicting taint (uncertain in true-arm, known in false-arm).
    // Post-fix: drop-on-conflict. into_known(r) must NOT fire false claim.
    let k2: Known<i32> = into_known(r);
    from_known(k2)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    # The "first wins" pre-fix would have silently passed (kind matched).
    # The drop-on-conflict post-fix correctly drops the static claim and
    # the consult silently passes too (no static claim, defers to runtime).
    # The KEY invariant: no FALSE Uncertain→Known diagnostic for what's
    # actually a Known→Known (in the false-arm).
    launder_errs = [e for e in errs if "launder" in str(e)]
    # Today no diagnostic fires (drop-on-conflict). When Inc 4 lands
    # multi-kind union, this assertion will flip to assert the launder
    # fires with "could be Uncertain OR Known" framing.
    assert launder_errs == [], \
        f"post-fix drop-on-conflict must not fire false launder, " \
        f"got: {[str(e) for e in errs]}"


def test_stage52_gate2_high_a_if_else_parallel_union_drop():
    """HIGH-A: if-else launder via different-kind branches must
    drop the static claim (same semantics as match-arm union)."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let kw: Known<i32> = into_known(1);
    let mut r: i32 = 0;
    if true { r = from_uncertain(u); } else { r = from_known(kw); };
    let k2: Known<i32> = into_known(r);
    from_known(k2)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    launder_errs = [e for e in errs if "launder" in str(e)]
    # Same drop-on-conflict semantics as match.
    assert launder_errs == [], \
        f"post-fix drop-on-conflict on if-else must not fire " \
        f"false launder, got: {[str(e) for e in errs]}"


def test_stage52_gate2_h1_diagnostic_says_taint_tracking_not_let_bypass():
    """H1: post-fix diagnostic says 'via taint-tracking' (covers
    let / Assign / match / if / while paths), not the misleading
    'via let-binding bypass'."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let mut r: i32 = 0;
    while true { r = from_uncertain(u); };
    let k: Known<i32> = into_known(r);
    from_known(k)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    launder_errs = [e for e in errs if "launder" in str(e)]
    assert launder_errs, \
        f"while-loop launder must fire, got: {[str(e) for e in errs]}"
    # H1 wording check: new format mentions "taint-tracking"
    assert any("taint-tracking" in str(e) for e in launder_errs), \
        f"diagnostic must say 'via taint-tracking' (not the old " \
        f"misleading 'via let-binding bypass'), got: " \
        f"{[str(e) for e in launder_errs]}"


def test_stage52_legitimate_same_kind_round_trip_does_not_fire():
    """Negative: legitimate `let r = from_known(k); into_known(r)`
    must NOT fire — source kind == target kind, no launder."""
    src = """
fn main() -> i32 {
    let kw: Known<i32> = into_known(42);
    let r: i32 = from_known(kw);
    let k2: Known<i32> = into_known(r);
    from_known(k2)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert not any("launder" in str(e) for e in errs), \
        f"same-kind round-trip must not fire launder, got: " \
        f"{[str(e) for e in errs]}"


def test_stage52_cross_fn_taint_does_not_leak():
    """gate-2 M5-equivalent: fn A's `let r = from_uncertain(u)`
    must not taint fn B's parameter `r`. _check_fn entry clear
    handles this; pin the regression."""
    src = """
fn maker() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    from_uncertain(u)
}
fn taker(r: i32) -> i32 {
    // r is a fresh parameter; cross-fn taint must NOT carry
    // over. into_known(r) here should NOT fire.
    let k: Known<i32> = into_known(r);
    from_known(k)
}
fn main() -> i32 {
    taker(maker())
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    # taker's `r` parameter is fresh — no taint leakage from maker.
    assert not any("launder" in str(e) for e in errs), \
        f"cross-fn parameter taint must not leak, got: " \
        f"{[str(e) for e in errs]}"


# ============================================================
# Stage 52 closure gate-3 regression pins (cascading-defect:
# gate-2 wave produced 5 new HIGHs). Each test below pins one
# fix path so a future refactor catches breakage.
# ============================================================


def test_stage52_gate3_new_high_1_inner_let_shadow_restores_outer_taint():
    """NEW-HIGH-1: outer `let r = from_uncertain(u)` taints r.
    An inner block with a non-tainted `let r: i32 = 7` shadow
    POPs r from the modal-origin dict (the inner let's Let-stmt
    POP path). At block exit, the OUTER taint must be RESTORED.

    Pre-fix: restore loop iterated current.keys(), but r was
    no longer in current.keys() (popped by inner-let). Outer
    taint was lost. Subsequent into_known(r) SILENTLY PASSED
    — a category-error launder went undetected.

    Post-fix: restore loop iterates inner_modal_lets (which
    has r), checks saved_modal_origin (which has r→uncertain),
    restores. into_known(r) now correctly fires."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let r: i32 = from_uncertain(u);
    let dummy: i32 = {
        let r: i32 = 7;
        r + 1
    };
    let k: Known<i32> = into_known(r);
    from_known(k) + dummy
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launder" in str(e) and "Uncertain" in str(e)
               and "Known" in str(e) for e in errs), \
        f"outer-taint-after-inner-shadow launder must fire " \
        f"(NEW-HIGH-1 silent miscompile regression), got: " \
        f"{[str(e) for e in errs]}"


def test_stage52_gate3_new_high_2_if_else_both_branches_clear_drops_claim():
    """NEW-HIGH-2: if-else where BOTH branches reassign without
    installing modal taint must DROP the pre-if static claim.

    Pre-fix: pre-if had r→uncertain. Both arms cleared (no
    'r' entry in arm-results). Pre-fix union saw only
    pre-if's uncertain in observed_kinds — single kind →
    propagated r→uncertain. Post-if into_known(r) FALSELY
    fired the launder diagnostic for a path that at runtime
    always reassigns r to a non-tainted value.

    Post-fix: branch_assigns tracks per-branch assignments;
    cleared_names drops names assigned without installing
    taint. r dropped from union → consult falls through to
    no-static-claim → no fire (Phase-0 dynamic territory)."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let r: i32 = from_uncertain(u);
    let mut r2: i32 = r;
    if true { r2 = 5; } else { r2 = 7; };
    let k: Known<i32> = into_known(r2);
    from_known(k)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    launder_errs = [e for e in errs if "launder" in str(e)]
    assert launder_errs == [], \
        f"both-arms-cleared if-else must DROP static claim " \
        f"(NEW-HIGH-2 false-positive regression), got: " \
        f"{[str(e) for e in errs]}"


def test_stage52_gate3_new_high_3_if_then_no_else_clear_drops_claim():
    """NEW-HIGH-3: if-then with no-else where the then-branch
    reassigns without modal taint must DROP the pre-if claim.

    Symmetric with NEW-HIGH-2 but exercises the no-else
    implicit-pre-match-snapshot path. Pre-fix's implicit
    else-branch copy of pre-if's r→uncertain caused observed_
    kinds={uncertain} → false-positive launder. Post-fix:
    branch_assigns[0] has 'r' (then assigned), branch_results
    [0] has no 'r' (then cleared) → cleared_names={r} → drop."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let r: i32 = from_uncertain(u);
    let mut r2: i32 = r;
    if true { r2 = 5; };
    let k: Known<i32> = into_known(r2);
    from_known(k)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    launder_errs = [e for e in errs if "launder" in str(e)]
    assert launder_errs == [], \
        f"no-else if-then cleared must DROP static claim " \
        f"(NEW-HIGH-3 false-positive regression), got: " \
        f"{[str(e) for e in errs]}"


def test_stage52_gate3_new_high_4_match_arm_clear_drops_claim():
    """NEW-HIGH-4: match-arm union must also DROP the pre-match
    static claim when any arm reassigns without installing
    modal taint. Symmetric with NEW-HIGH-2/3 for A.If.

    Pre-fix: pre-match r→uncertain. Arm 1 cleared (r=5),
    arm 2 kept (no Assign). Pre-fix union observed_kinds=
    {uncertain (pre-match), uncertain (arm 2 result)} →
    single kind → propagated r→uncertain → false-positive
    launder. Post-fix: arm_assigns[0] has 'r' (arm 1
    assigned), arm_results[0] has no 'r' (arm 1 cleared)
    → cleared_names_match={r} → drop."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let r: i32 = from_uncertain(u);
    let mut r2: i32 = r;
    match true { true => { r2 = 5; }, false => {} };
    let k: Known<i32> = into_known(r2);
    from_known(k)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    launder_errs = [e for e in errs if "launder" in str(e)]
    assert launder_errs == [], \
        f"match-arm cleared must DROP static claim " \
        f"(NEW-HIGH-4 false-positive regression), got: " \
        f"{[str(e) for e in errs]}"


def test_stage52_gate3_new_high_5_real_launder_still_fires_with_drop_path():
    """NEW-HIGH-5 (meta): after the cleared-branch drop fix,
    REAL launders must STILL fire. This is the dual of
    NEW-HIGH-2/3/4 — verify the drop semantics doesn't
    over-broadly suppress legitimate launder diagnostics.

    Setup: if-then-only where the then-branch INSTALLS taint
    (not clears). Pre-if's r is untainted; then's r=from_X(u)
    installs uncertain. Union: only then has r→uncertain →
    propagate. into_known(r) MUST fire (real launder).

    This pins the invariant that the cleared-branch drop only
    fires when an assignment lacks modal-installation — it
    must NOT drop when an arm installs taint."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let mut r: i32 = 0;
    if true { r = from_uncertain(u); };
    let k: Known<i32> = into_known(r);
    from_known(k)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launder" in str(e) and "Uncertain" in str(e)
               and "Known" in str(e) for e in errs), \
        f"real launder via if-then taint install MUST fire " \
        f"(NEW-HIGH-5 over-broad-drop regression), got: " \
        f"{[str(e) for e in errs]}"


# ============================================================
# Stage 52 closure gate-4 regression pins
# ============================================================


def test_stage52_gate4_high_1_match_patbind_propagates_scrutinee_taint():
    """Gate-4 HIGH-1: `let r = from_uncertain(u); match r { x =>
    into_known(x) }` was silent because PatBind only wrote to
    the value scope, never to _modal_origin_provenance. x had
    no entry; launder consult fell through to no-claim → silent
    miscompile. Direct AI-safety bypass via trivial bind.

    Post-fix: when scrutinee is a Name with tracked modal
    origin AND the pattern is a top-level PatBind, copy the
    taint to the bound name (placed INSIDE the snapshot region
    per gate-4 MEDIUM-2 placement constraint so the binding
    doesn't leak past the arm boundary)."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let r: i32 = from_uncertain(u);
    let kk: i32 = match r { x => from_known(into_known(x)) };
    kk
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launder" in str(e) and "Uncertain" in str(e)
               and "Known" in str(e) and "'x'" in str(e)
               for e in errs), \
        f"PatBind-bound x must inherit scrutinee taint " \
        f"(gate-4 HIGH-1 silent miscompile regression), got: " \
        f"{[str(e) for e in errs]}"


def test_stage52_gate4_critical_1_loop_body_phase0_limitation_documented():
    """Gate-4 CRITICAL-1 is the SAME Phase-0 limitation that
    if-no-else has (verified independently): when a loop body
    INSTALLS a different modal kind than the pre-loop value,
    the 0-iteration case at runtime leaves the original kind
    in place, but the static dict shows the new kind. into_X
    matching the new kind silently passes — but at runtime in
    the 0-iter case, into_X is applied to a value of the OLD
    kind (real launder, silent).

    This is NOT closeable by the audit-recommended union fix:
    union would drop on multi-kind divergence, also silent.
    The proper fix requires the deferred Inc 4 multi-kind
    diagnostic ("could be X or Y, neither matches target").

    This test PINS the current behavior so a refactor doesn't
    accidentally change it without flipping this test (and the
    Inc 4 fix will).

    Pre-loop: r→Uncertain. Body installs Known. into_known(r):
    - Current: dict[r]=Known (body's POPULATE), matches target
      → no fire (silent at 0-iter case, OK at 1+iter case).
    - Same defect: if-no-else with `r = from_known(kw)` in
      then-branch + no else. Pre-fix-gate-2 would have fired
      (over-conservative); post-gate-2 drops-on-conflict, also
      silent.

    Deferred to Inc 4 multi-kind diagnostic."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let kw: Known<i32> = into_known(1);
    let mut r: i32 = from_uncertain(u);
    let mut i: i32 = 0;
    while i < 1 { r = from_known(kw); i = i + 1; };
    let kk: Known<i32> = into_known(r);
    from_known(kk)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    launder_errs = [e for e in errs if "launder" in str(e)]
    # Phase-0 limitation: silent today. Inc 4 multi-kind
    # diagnostic should flip this to FIRE. When Inc 4 lands,
    # this assertion becomes the post-fix assertion.
    assert launder_errs == [], \
        f"Phase-0 limitation: loop-body-installs-Known on " \
        f"pre-loop-Uncertain is currently silent (same defect " \
        f"class as if-no-else). Inc 4 multi-kind diagnostic " \
        f"is the proper fix. If this test fires, update both " \
        f"this assertion and the parallel if-no-else case. " \
        f"Got: {[str(e) for e in errs]}"


def test_stage52_gate4_critical_1_parallel_if_no_else_same_limitation():
    """Companion to the loop-body test: pin the if-no-else
    variant of the same Phase-0 limitation. Same drop-on-
    conflict semantics. If Inc 4 multi-kind diagnostic lands,
    flip this AND the loop test in lockstep."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let kw: Known<i32> = into_known(1);
    let mut r: i32 = from_uncertain(u);
    if true { r = from_known(kw); };
    let kk: Known<i32> = into_known(r);
    from_known(kk)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    launder_errs = [e for e in errs if "launder" in str(e)]
    assert launder_errs == [], \
        f"Phase-0 limitation: if-no-else with then-installs- " \
        f"Known on pre-Uncertain is currently silent (drop-on- " \
        f"conflict). Inc 4 multi-kind diagnostic is the proper " \
        f"fix. Got: {[str(e) for e in errs]}"


# ============================================================
# Stage 52 closure gate-5 regression pins
# ============================================================


# ============================================================
# Stage 52 closure gate-6 regression pins (3 NEW CRITICAL +
# 1 bonus Assign-alias variant)
# ============================================================


def test_stage52_gate6_critical_1_call_form_scrutinee_propagates_taint():
    """Gate-6 CRITICAL-1: `match from_uncertain(u) { x => into_known(x) }`
    silently passed because PatBind taint copy gated on Name-only
    scrutinee. Direct AI-safety bypass via inline Call scrutinee
    — exactly the dogfood_13 modal-lifecycle pattern.
    Post-fix: unified _modal_origin_of_expr helper recognizes
    Call(from_X, ...) and copies the source kind to bound name."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let out: i32 = match from_uncertain(u) {
        x => from_known(into_known(x))
    };
    out
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launder" in str(e) and "Uncertain" in str(e)
               and "Known" in str(e) and "'x'" in str(e)
               for e in errs), \
        f"Call-form scrutinee taint must propagate to PatBind " \
        f"(gate-6 CRITICAL-1 regression), got: " \
        f"{[str(e) for e in errs]}"


def test_stage52_gate6_critical_2_let_alias_propagates_taint():
    """Gate-6 CRITICAL-2: `let r = from_X(v); let s = r;
    into_Y(s)` silently passed because Let-stmt populate only
    handled direct `Call(from_X, ...)` RHS — name-alias on the
    RHS pop'd the destination instead of copying taint.
    Trivial 2-line laundering vector; worse than gate-1 HIGH-2
    because no fancy match/if/while needed.
    Post-fix: Let-stmt RHS goes through unified
    _modal_origin_of_expr which handles A.Name lookup."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let r: i32 = from_uncertain(u);
    let s: i32 = r;
    from_known(into_known(s))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launder" in str(e) and "Uncertain" in str(e)
               and "Known" in str(e) and "'s'" in str(e)
               for e in errs), \
        f"let-alias must propagate taint (gate-6 CRITICAL-2 " \
        f"regression), got: {[str(e) for e in errs]}"


def test_stage52_gate6_critical_2_assign_alias_propagates_taint():
    """Gate-6 CRITICAL-2 variant: same defect via Assign instead
    of Let. `let mut s = 0; s = r;` where r is tainted must
    install taint on s. Post-fix: Assign RHS goes through
    unified _modal_origin_of_expr."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let r: i32 = from_uncertain(u);
    let mut s: i32 = 0;
    s = r;
    from_known(into_known(s))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launder" in str(e) and "Uncertain" in str(e)
               and "Known" in str(e) and "'s'" in str(e)
               for e in errs), \
        f"Assign-alias must propagate taint (gate-6 CRITICAL-2 " \
        f"variant regression), got: {[str(e) for e in errs]}"


def test_stage52_gate6_critical_3_pator_of_patbind_propagates_taint():
    """Gate-6 CRITICAL-3: PatOr arms where each alt is a PatBind
    of the SAME name (every alt aliases the entire scrutinee)
    must inherit scrutinee taint. Pre-fix the PatBind copy only
    handled top-level PatBind, missing PatOr-of-PatBind which is
    the natural fan-in pattern.
    Post-fix: detect PatOr where every alt is PatBind of one
    name; treat as top-level PatBind for taint-copy."""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let r: i32 = from_uncertain(u);
    let out: i32 = match r {
        x | x => from_known(into_known(x)),
        _ => 0
    };
    out
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launder" in str(e) and "Uncertain" in str(e)
               and "Known" in str(e) and "'x'" in str(e)
               for e in errs), \
        f"PatOr-of-same-PatBind must propagate taint " \
        f"(gate-6 CRITICAL-3 regression), got: " \
        f"{[str(e) for e in errs]}"


def test_stage52_gate6_real_let_alias_to_same_kind_does_not_fire():
    """Negative pin: legitimate same-kind name-alias must NOT
    fire (`let r = from_known(k); let s = r; into_known(s)` is
    a legal pass-through, not a launder). Mirrors the existing
    same-kind round-trip pin."""
    src = """
fn main() -> i32 {
    let kw: Known<i32> = into_known(42);
    let r: i32 = from_known(kw);
    let s: i32 = r;
    let k2: Known<i32> = into_known(s);
    from_known(k2)
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert not any("launder" in str(e) for e in errs), \
        f"same-kind alias must not fire launder " \
        f"(gate-6 negative regression), got: " \
        f"{[str(e) for e in errs]}"


def test_stage52_gate6_type_design_f1_no_residual_local_dict():
    """Gate-6 type-design F1: the launder-guard site at typecheck.py
    used to hold a 4th local copy of the elim→kind map under name
    `_modal_elim_kind`, contradicting the gate-2 F3 hoisting
    invariant. Post-fix: only one `_MODAL_ELIM_TO_KIND` reference
    exists (the module constant). This test pins the invariant by
    grepping the typecheck.py source for any residual local."""
    import re
    from pathlib import Path
    src = Path(
        "helixc/frontend/typecheck.py").read_text(encoding="utf-8")
    # Should have ONE _MODAL_ELIM_TO_KIND definition (the module
    # constant) and NO _modal_elim_kind = {...} local definitions.
    assert "_modal_elim_kind" not in src, \
        "Residual local `_modal_elim_kind` dict still present — " \
        "violates gate-2 F3 hoisting invariant (single source of " \
        "truth for elim→kind mapping). Use _MODAL_ELIM_TO_KIND."
    # Module constant present.
    assert "_MODAL_ELIM_TO_KIND" in src, \
        "Module-level _MODAL_ELIM_TO_KIND constant missing — " \
        "gate-2 F3 fix regressed."


def test_stage52_gate5_high_1_patbind_taint_propagation_in_guard():
    """Gate-5 HIGH-1: PatBind taint propagation MUST run BEFORE
    the guard expression, not after. Pre-gate-5 the order was:
      (1) _bind_pattern
      (2) guard check        ← guard reads dict BEFORE step 4 wrote taint
      (3) snapshot pre-match
      (4) PatBind taint copy
      (5) arm body

    A guard expression that calls a modal eliminator on the bound
    name (e.g. `match r { x if into_known(x) > 0 => 1, _ => 0 }`)
    consulted _modal_origin_provenance['x'] at step 2, found NO
    entry, fell through silently — direct AI-safety bypass via the
    guard slot.

    Post-fix order:
      (1) _bind_pattern
      (2) snapshot pre-match
      (3) PatBind taint copy
      (4) guard check        ← now sees taint
      (5) arm body"""
    src = """
fn main() -> i32 {
    let u: Uncertain<i32> = into_uncertain(1);
    let r: i32 = from_uncertain(u);
    let out: i32 = match r {
        x if from_known(into_known(x)) > 0 => 1,
        _ => 0
    };
    out
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("launder" in str(e) and "Uncertain" in str(e)
               and "Known" in str(e) and "'x'" in str(e)
               for e in errs), \
        f"PatBind taint must fire inside guard expression " \
        f"(gate-5 HIGH-1 silent miscompile regression), got: " \
        f"{[str(e) for e in errs]}"
