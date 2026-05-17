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


def test_stage43_item2_old_name_still_aliased():
    """The old `_FRAME_IDENTITY_AD_NAMES` name must remain as a
    backwards-compat alias for one stage. Drop at Stage 44+."""
    from helixc.frontend.autodiff import (
        _FRAME_IDENTITY_AD_NAMES, _IDENTITY_AD_CHAIN_RULE_NAMES,
    )
    assert _FRAME_IDENTITY_AD_NAMES is _IDENTITY_AD_CHAIN_RULE_NAMES, \
        "old name must alias the new name (same object), not a copy"


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
