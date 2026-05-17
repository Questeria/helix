"""Stage 43 Increment 1 — deferred-items cleanup sweep regression tests.

Pins the 3 fixes that landed at Stage 43 closure:
- Item 2: `_FRAME_IDENTITY_AD_NAMES` rename to
  `_IDENTITY_AD_CHAIN_RULE_NAMES` with backwards-compat alias.
- Item 3: F5 `_resolve_type` arity arms across all 5 wrapper
  families (TyMemTier / TyFrame / TyTemporal / TyModal / TyCausal).
- Item 4: M1 intro double-wrap rejection across all 5 wrapper
  families.

Item 1 (aggregate -> composite diagnostic rename) was DEFERRED
to Stage 44+ — touches 6 existing test assertions; the risk
of breaking test surfaces for a cosmetic disambiguation
exceeds the reward.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.typecheck import typecheck


# ============================================================
# Item 2 — autodiff identity-AD chain rule set rename
# ============================================================


def test_stage43_item2_new_name_is_exported():
    """The renamed set must be importable under its new name."""
    from helixc.frontend.autodiff import _IDENTITY_AD_CHAIN_RULE_NAMES
    assert isinstance(_IDENTITY_AD_CHAIN_RULE_NAMES, frozenset)
    assert len(_IDENTITY_AD_CHAIN_RULE_NAMES) >= 45, \
        "5 wrapper families × ~9 builtins each = ~45 entries; got " \
        f"{len(_IDENTITY_AD_CHAIN_RULE_NAMES)}"


def test_stage47_old_name_alias_dropped():
    """Stage 47 drop: the backwards-compat alias
    `_FRAME_IDENTITY_AD_NAMES` was retained at Stage 43 LOW-3
    "drop at Stage 44 or beyond" for one stage of grace
    period. Three stages have elapsed (44, 45, 46); the alias
    is now dropped. This test inverts the gate-1 assertion
    to confirm the drop landed."""
    import helixc.frontend.autodiff as ad
    assert not hasattr(ad, "_FRAME_IDENTITY_AD_NAMES"), \
        "Stage 47 dropped the backwards-compat alias; if you " \
        "see this fail, the alias is unexpectedly back."


def test_stage43_item2_autodiff_reverse_imports_new_name():
    """autodiff_reverse.py must import the new name (not the old
    one) so the rename actually propagated."""
    import helixc.frontend.autodiff_reverse as ar
    # The new name should be available at module level via import.
    assert hasattr(ar, "_IDENTITY_AD_CHAIN_RULE_NAMES")


# ============================================================
# Item 3 — F5 _resolve_type arity arms (5 families)
# ============================================================


def test_stage43_item3_tier_zero_args_diagnoses():
    src = "fn main() -> i32 { 0 } fn f(x: WorkingMem<>) -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    assert any("WorkingMem<T> takes 1 type argument" in str(e)
               for e in errs), \
        f"WorkingMem<> must diagnose arity, got {[str(e) for e in errs]}"


def test_stage43_item3_tier_two_args_diagnoses():
    src = "fn main() -> i32 { 0 } fn f(x: WorkingMem<i32, i32>) -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    assert any("WorkingMem<T> takes 1 type argument" in str(e)
               for e in errs)


def test_stage43_item3_frame_two_args_diagnoses():
    src = "fn main() -> i32 { 0 } fn f(x: WorldFrame<i32, i32>) -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    assert any("WorldFrame<T> takes 1 type argument" in str(e)
               for e in errs)


def test_stage43_item3_temporal_two_args_diagnoses():
    src = "fn main() -> i32 { 0 } fn f(x: Past<i32, i32>) -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    assert any("Past<T> takes 1 type argument" in str(e)
               for e in errs)


def test_stage43_item3_modal_two_args_diagnoses():
    src = "fn main() -> i32 { 0 } fn f(x: Known<i32, i32>) -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    assert any("Known<T> takes 1 type argument" in str(e)
               for e in errs)


def test_stage43_item3_causal_two_args_diagnoses():
    src = "fn main() -> i32 { 0 } fn f(x: Cause<i32, i32>) -> i32 { 0 }"
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    assert any("Cause<T> takes 1 type argument" in str(e)
               for e in errs)


# ============================================================
# Item 4 — M1 intro double-wrap rejection (5 families)
# ============================================================


def test_stage43_item4_tier_double_wrap_rejected():
    src = """
fn main() -> i32 {
    let a: WorkingMem<i32> = into_working(7);
    unwrap_working(into_working(a));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    assert any("not idempotent" in str(e) and "into_working" in str(e)
               for e in errs)


def test_stage43_item4_frame_double_wrap_rejected():
    src = """
fn main() -> i32 {
    let a: WorldFrame<i32> = into_world(7);
    from_world(into_world(a));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    assert any("not idempotent" in str(e) and "into_world" in str(e)
               for e in errs)


def test_stage43_item4_temporal_double_wrap_rejected():
    src = """
fn main() -> i32 {
    let a: Past<i32> = into_past(7);
    from_past(into_past(a));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    assert any("not idempotent" in str(e) and "into_past" in str(e)
               for e in errs)


def test_stage43_item4_modal_double_wrap_rejected():
    src = """
fn main() -> i32 {
    let a: Known<i32> = into_known(7);
    from_known(into_known(a));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    assert any("not idempotent" in str(e) and "into_known" in str(e)
               for e in errs)


def test_stage43_item4_causal_double_wrap_rejected():
    src = """
fn main() -> i32 {
    let a: Cause<i32> = into_cause(7);
    from_cause(into_cause(a));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    assert any("not idempotent" in str(e) and "into_cause" in str(e)
               for e in errs)


def test_stage43_item4_self_wrap_to_unwrap_still_allowed():
    """Self-wrap-then-unwrap (the idempotent unit case) is not
    a double-wrap because no second wrapper is constructed.
    Sanity check: `from_X(into_X(v))` continues to typecheck
    clean. This is the bread-and-butter usage pattern."""
    src = """
fn main() -> i32 {
    from_known(into_known(42))
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    assert errs == [], \
        f"from_X(into_X(v)) is not double-wrap; got {[str(e) for e in errs]}"


# ============================================================
# Stage 43 closure gate-1 MEDIUM backfills:
# - Frame double-wrap hint must be direction-correct.
# - Tier double-wrap hint must name concrete transitions.
# ============================================================


def test_stage43_gate1_frame_double_wrap_hint_is_direction_correct():
    """`into_world(RobotFrame<i32>)` must suggest `robot_to_world`,
    not `world_to_robot` (the wrong-direction hard-coded example
    pre-fix)."""
    src = """
fn main() -> i32 {
    let r: RobotFrame<i32> = into_robot(7);
    from_world(into_world(r));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    matching = [str(e) for e in errs
                if "not idempotent" in str(e) and "into_world" in str(e)]
    assert matching, f"expected into_world double-wrap diag, got {[str(e) for e in errs]}"
    err_text = matching[0]
    assert "robot_to_world" in err_text, \
        f"frame double-wrap hint must name the direction-correct " \
        f"transition `robot_to_world` for (robot -> world); got " \
        f"{err_text!r}"


def test_stage43_gate1_frame_double_wrap_same_kind_hint_says_unwrap():
    """`into_world(WorldFrame<i32>)` (same source + target) has no
    legitimate transition — the hint must point at unwrap, not
    at a non-existent self-transform like `world_to_world`."""
    src = """
fn main() -> i32 {
    let w: WorldFrame<i32> = into_world(7);
    from_world(into_world(w));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    matching = [str(e) for e in errs
                if "not idempotent" in str(e) and "into_world" in str(e)]
    assert matching
    err_text = matching[0]
    assert "world_to_world" not in err_text, \
        f"must not suggest nonsense self-transition; got {err_text!r}"
    assert "from_world" in err_text or "unwrap" in err_text, \
        f"same-kind hint must point at unwrap; got {err_text!r}"


def test_stage43_gate1_tier_double_wrap_episodic_to_semantic_hint():
    """`into_semantic(EpisodicMem<i32>)` must suggest `consolidate`,
    the audited Episodic -> Semantic transition."""
    src = """
fn main() -> i32 {
    let e: EpisodicMem<i32> = into_episodic(7);
    unwrap_semantic(into_semantic(e));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    matching = [str(e) for e in errs
                if "not idempotent" in str(e) and "into_semantic" in str(e)]
    assert matching
    err_text = matching[0]
    assert "consolidate" in err_text, \
        f"tier double-wrap Episodic -> Semantic must suggest " \
        f"`consolidate`; got {err_text!r}"


def test_stage43_gate1_tier_double_wrap_semantic_to_working_hint():
    """`into_working(SemanticMem<i32>)` must suggest `recall`."""
    src = """
fn main() -> i32 {
    let s: SemanticMem<i32> = into_semantic(7);
    unwrap_working(into_working(s));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    matching = [str(e) for e in errs
                if "not idempotent" in str(e) and "into_working" in str(e)]
    assert matching
    err_text = matching[0]
    assert "recall" in err_text, \
        f"tier double-wrap Semantic -> Working must suggest " \
        f"`recall`; got {err_text!r}"


# ============================================================
# Stage 43 closure gate-2 MEDIUM backfills: direction-aware
# hints for the remaining 3 wrapper families (temporal / modal /
# causal). Gate-1 only direction-fixed frame + tier; gate-2
# audit flagged the asymmetry across all 5 families.
# ============================================================


def test_stage43_gate2_temporal_double_wrap_direction_aware_present_to_past():
    """`into_past(Present<i32>)` must suggest `to_past`."""
    src = """
fn main() -> i32 {
    let p: Present<i32> = into_present(7);
    from_past(into_past(p));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    matching = [str(e) for e in errs
                if "not idempotent" in str(e) and "into_past" in str(e)]
    assert matching
    err_text = matching[0]
    assert "to_past" in err_text, \
        f"temporal double-wrap Present -> Past must suggest " \
        f"`to_past`; got {err_text!r}"


def test_stage43_gate2_temporal_double_wrap_direction_aware_past_to_present():
    """`into_present(Past<i32>)` must suggest `recall_past`."""
    src = """
fn main() -> i32 {
    let p: Past<i32> = into_past(7);
    from_present(into_present(p));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    matching = [str(e) for e in errs
                if "not idempotent" in str(e) and "into_present" in str(e)]
    assert matching
    err_text = matching[0]
    assert "recall_past" in err_text, \
        f"temporal double-wrap Past -> Present must suggest " \
        f"`recall_past`; got {err_text!r}"


def test_stage43_gate2_modal_double_wrap_direction_aware_believed_to_known():
    """`into_known(Believed<i32>)` must suggest `confirm`."""
    src = """
fn main() -> i32 {
    let b: Believed<i32> = into_believed(7);
    from_known(into_known(b));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    matching = [str(e) for e in errs
                if "not idempotent" in str(e) and "into_known" in str(e)]
    assert matching
    err_text = matching[0]
    assert "confirm" in err_text, \
        f"modal double-wrap Believed -> Known must suggest " \
        f"`confirm`; got {err_text!r}"


def test_stage43_gate2_modal_double_wrap_direction_aware_goal_to_known():
    """`into_known(Goal<i32>)` must suggest `act_on`."""
    src = """
fn main() -> i32 {
    let g: Goal<i32> = into_goal(7);
    from_known(into_known(g));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    matching = [str(e) for e in errs
                if "not idempotent" in str(e) and "into_known" in str(e)]
    assert matching
    err_text = matching[0]
    assert "act_on" in err_text, \
        f"modal double-wrap Goal -> Known must suggest " \
        f"`act_on`; got {err_text!r}"


def test_stage43_gate2_causal_double_wrap_direction_aware_cause_to_effect():
    """`into_effect(Cause<i32>)` must suggest `propagate`."""
    src = """
fn main() -> i32 {
    let c: Cause<i32> = into_cause(7);
    from_effect(into_effect(c));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    matching = [str(e) for e in errs
                if "not idempotent" in str(e) and "into_effect" in str(e)]
    assert matching
    err_text = matching[0]
    assert "propagate" in err_text, \
        f"causal double-wrap Cause -> Effect must suggest " \
        f"`propagate`; got {err_text!r}"


def test_stage43_gate2_causal_double_wrap_direction_aware_effect_to_joint():
    """`into_joint(Effect<i32>)` must suggest `aggregate`."""
    src = """
fn main() -> i32 {
    let e: Effect<i32> = into_effect(7);
    from_joint(into_joint(e));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    matching = [str(e) for e in errs
                if "not idempotent" in str(e) and "into_joint" in str(e)]
    assert matching
    err_text = matching[0]
    assert "aggregate" in err_text, \
        f"causal double-wrap Effect -> Joint must suggest " \
        f"`aggregate`; got {err_text!r}"


def test_stage43_gate2_causal_double_wrap_direction_aware_joint_to_independent():
    """`into_independent(Joint<i32>)` must suggest `isolate`."""
    src = """
fn main() -> i32 {
    let j: Joint<i32> = into_joint(7);
    from_independent(into_independent(j));
    0
}
"""
    prog = parse(src, include_stdlib=False)
    errs = typecheck(prog)
    matching = [str(e) for e in errs
                if "not idempotent" in str(e) and "into_independent" in str(e)]
    assert matching
    err_text = matching[0]
    assert "isolate" in err_text, \
        f"causal double-wrap Joint -> Independent must suggest " \
        f"`isolate`; got {err_text!r}"
