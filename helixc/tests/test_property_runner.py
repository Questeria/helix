"""Stage 86 tests — Stage 77 Inc 2 @property runner.

Two tests:
1. Fast discovery-only test against the safety.hx stdlib (no
   codegen, just parse + extract @property fn list). Validates
   the runner's discovery half.
2. Slow end-to-end test against a trivial single-property program.
   Compiles + runs via WSL once per input (7 inputs for i32 →
   ~20s total). Skipped if WSL isn't available.
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from helixc.runners.property_runner import (  # noqa: E402
    _discover_properties,
    _build_stdlib_only_src,
    _format_input_lit,
    _generate_runner_main,
    _strip_main,
    _INPUT_TABLE,
)


def test_stage86_discovery_finds_all_eleven_stdlib_properties():
    """Stage 86 + Stage 98 + Stage 104 — discovery half finds the 11
    @property fns shipped in safety.hx (one per Tier-S/A wrapper).
    History: 2 from Stage 78 (Conf, Taint), 3 from Stage 82 (Enclave,
    Cfact, Deadline), 1 from Stage 98 (Attribution closing the audit-
    identified gap), 5 from Stage 104 (DP, Quant, Domain, Robust,
    Energy closing the per-wrapper roundtrip-property gap)."""
    from helixc.frontend.parser import parse
    src = _build_stdlib_only_src()
    prog = parse(src, include_stdlib=True)
    props = _discover_properties(prog)
    names = {name for name, _args in props}
    expected = {
        "safety_conf_roundtrip_is_identity",          # Stage 78
        "safety_taint_roundtrip_is_identity",         # Stage 78
        "safety_enclave_roundtrip_is_identity",       # Stage 82
        "safety_cfact_roundtrip_is_identity",         # Stage 82
        "safety_deadline_roundtrip_is_identity",      # Stage 82
        "safety_attribution_roundtrip_is_identity",   # Stage 98
        "safety_dp_roundtrip_is_identity",            # Stage 104
        "safety_quant_roundtrip_is_identity",         # Stage 104
        "safety_domain_roundtrip_is_identity",        # Stage 104
        "safety_robust_roundtrip_is_identity",        # Stage 104
        "safety_energy_roundtrip_is_identity",        # Stage 104
    }
    missing = expected - names
    assert not missing, f"missing stdlib @property fns: {missing}"


def test_stage86_discovery_extracts_arg_types():
    """Stage 86 — discovery returns (name, [(arg_name, arg_ty_str)])
    pairs so the runner can pick the right input table."""
    from helixc.frontend.parser import parse
    src = _build_stdlib_only_src()
    prog = parse(src, include_stdlib=True)
    props = _discover_properties(prog)
    # All stdlib properties take a single f32 arg named x.
    for name, args in props:
        assert len(args) == 1, f"{name}: expected 1 arg, got {args}"
        arg_name, arg_ty = args[0]
        assert arg_name == "x", f"{name}: arg name was {arg_name!r}"
        assert arg_ty == "f32", f"{name}: arg ty was {arg_ty!r}"


def test_stage86_format_input_lit_for_each_type():
    """Stage 86 — `_format_input_lit` renders Python values as Helix
    literals with the right type suffix per primitive."""
    assert _format_input_lit(True, "bool") == "true"
    assert _format_input_lit(False, "bool") == "false"
    assert _format_input_lit(42, "i32") == "42_i32"
    assert _format_input_lit(-1, "i32") == "-1_i32"
    assert _format_input_lit(1.5, "f32") == "1.5_f32"
    # Edge: integer-valued floats keep the .0 via repr.
    assert "_f32" in _format_input_lit(1.0, "f32")


def test_stage86_generate_runner_main_calls_property_with_literal():
    """Stage 86 — generated main calls the named property with the
    formatted literal and returns 42/99."""
    src = _generate_runner_main("foo_prop", "i32", 42)
    assert "foo_prop(42_i32)" in src
    assert "fn main() -> i32" in src
    assert "{ 42 }" in src
    assert "{ 99 }" in src


def test_stage86_strip_main_removes_existing_main():
    """Stage 86 — `_strip_main` removes the user-supplied main so
    the runner can substitute its synthetic version."""
    src = "fn helper() -> i32 { 1 }\nfn main() -> i32 { 0 }\n"
    out = _strip_main(src)
    assert "fn main" not in out
    assert "fn helper" in out


def test_stage86_strip_main_no_op_when_main_absent():
    """Stage 86 — `_strip_main` returns input unchanged when there's
    no main (e.g., stdlib-only flows)."""
    src = "fn helper() -> i32 { 1 }\n"
    assert _strip_main(src) == src


def test_stage86_input_tables_cover_core_primitives():
    """Stage 86 — input table has entries for i32 / f32 / bool at
    minimum (the only primitives @property fns commonly take)."""
    assert "i32" in _INPUT_TABLE
    assert "f32" in _INPUT_TABLE
    assert "bool" in _INPUT_TABLE
    # Each table is non-empty + bounded.
    for ty, vals in _INPUT_TABLE.items():
        assert 2 <= len(vals) <= 10, f"{ty} table has {len(vals)} values"


def _wsl_available() -> bool:
    """Cheap probe: WSL exists + can `echo hi`."""
    return bool(shutil.which("wsl"))


@pytest.mark.skipif(
    not _wsl_available(),
    reason="WSL not available; runner needs WSL for code execution",
)
def test_stage86_runner_end_to_end_on_trivial_property():
    """Stage 86 — full runner: trivial always-true property must
    pass for every input in the i32 table (7 assertions).

    Slow: ~20s wall clock on a typical dev box (7 × compile + WSL).
    Bounded — only one property, only one input type."""
    from helixc.runners.property_runner import run_properties

    src = """
    @property
    @pure
    fn always_true(x: i32) -> bool { true }
    fn main() -> i32 { 0 }
    """
    # Stage 98 (Stage 93 audit MEDIUM fix) — include_stdlib=False
    # isolates the test's own property from the 6 stdlib safety.hx
    # @property fns that would otherwise also run (5 × 7 f32 inputs
    # = 35 extra passes, breaking the p==7 assertion).
    p, f, log = run_properties(src, verbose=False, include_stdlib=False)
    assert f == 0, f"trivial property should have 0 failures; log={log}"
    # The i32 input table has 7 entries; runner should hit all 7.
    assert p == 7, f"expected 7 passes (i32 table size); got {p}"
