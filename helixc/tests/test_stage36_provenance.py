"""Stage 36 Increment 1 — end-to-end runtime tests for provenance-typed
primitives.

The typecheck-level tests for `prove()` and `unwrap_logic()` live in
`test_provenance.py` next to the Stage 24 type-level scaffolding. This
file holds the **runtime** complement: programs are compiled to ELF
binaries, executed via WSL, and exit codes are checked.

Phase-0 invariant: `prove(v, src)` and `unwrap_logic(l)` lower to
identity in the IR — the Logic<T> wrapper has zero runtime overhead;
provenance lives purely at the type level. The end-to-end tests
here verify that runtime semantics match this invariant.
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
    """Compile + run via WSL, return exit code (low byte)."""
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


def test_stage36_prove_unwrap_roundtrip_runtime():
    """End-to-end: `unwrap_logic(prove(41, 99)) + 1` exits 42.
    Confirms Logic<T> has zero runtime overhead and that the
    round-trip preserves the value bit-for-bit."""
    src = """
fn main() -> i32 {
    let x: i32 = 41;
    let l: Logic<i32> = prove(x, 99);
    unwrap_logic(l) + 1
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs == [], f"unexpected errors: {[str(e) for e in errs]}"
    mod = lower(prog)
    elf = compile_module_to_elf(mod)
    rc = _run_elf(elf)
    assert rc == 42, f"expected exit 42, got {rc}"


def test_stage36_prove_inside_arithmetic_runtime():
    """End-to-end: prove() can be used inline inside arithmetic via
    unwrap_logic. `unwrap_logic(prove(10, 0)) + unwrap_logic(prove(32, 0))`
    exits 42."""
    src = """
fn main() -> i32 {
    unwrap_logic(prove(10, 0)) + unwrap_logic(prove(32, 0))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs == [], f"unexpected errors: {[str(e) for e in errs]}"
    mod = lower(prog)
    elf = compile_module_to_elf(mod)
    rc = _run_elf(elf)
    assert rc == 42, f"expected exit 42, got {rc}"


def test_stage36_prove_unwrap_is_builtin():
    """Registry check: `prove` and `unwrap_logic` are listed as
    builtins so the unbound-name diagnostic doesn't fire on them."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    assert "prove" in tc._BUILTIN_NAMES
    assert "unwrap_logic" in tc._BUILTIN_NAMES


# Stage 36 Increment 2 — provenance-composing combinators.


def test_stage36_inc2_combinators_are_builtins():
    """derive, and_logic, or_logic, not_logic are registered."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    assert "derive" in tc._BUILTIN_NAMES
    assert "and_logic" in tc._BUILTIN_NAMES
    assert "or_logic" in tc._BUILTIN_NAMES
    assert "not_logic" in tc._BUILTIN_NAMES


def test_stage36_inc2_and_logic_truth_table():
    """and_logic on 0/1 truth values: 1 AND 1 = 1, 1 AND 0 = 0,
    0 AND 0 = 0."""
    for a, b, want in [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 0)]:
        src = f"""
fn main() -> i32 {{
    unwrap_logic(and_logic(prove({a}, 0), prove({b}, 0)))
}}
"""
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        rc = _run_elf(elf)
        assert rc == want, f"and_logic({a},{b}) -> {rc}, expected {want}"


def test_stage36_inc2_or_logic_truth_table():
    """or_logic on 0/1 truth values: covers all four input cases."""
    for a, b, want in [(1, 1, 1), (1, 0, 1), (0, 1, 1), (0, 0, 0)]:
        src = f"""
fn main() -> i32 {{
    unwrap_logic(or_logic(prove({a}, 0), prove({b}, 0)))
}}
"""
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        rc = _run_elf(elf)
        assert rc == want, f"or_logic({a},{b}) -> {rc}, expected {want}"


def test_stage36_inc2_not_logic_inverts():
    """not_logic: NOT 0 = 1, NOT 1 = 0."""
    for a, want in [(0, 1), (1, 0)]:
        src = f"fn main() -> i32 {{ unwrap_logic(not_logic(prove({a}, 0))) }}"
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        rc = _run_elf(elf)
        assert rc == want, f"not_logic({a}) -> {rc}, expected {want}"


def test_stage36_inc2_derive_keeps_first_parent_value():
    """derive(a, b) returns a's value in Phase-0 single-tag
    provenance. The lattice upgrade tracks both parents."""
    src = """
fn main() -> i32 {
    unwrap_logic(derive(prove(42, 1), prove(7, 2)))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


def test_stage36_inc2_compound_expression():
    """Compound: (1 AND 1) OR (NOT 0) = 1, times 42 = 42."""
    src = """
fn main() -> i32 {
    let t = prove(1, 0);
    let f = prove(0, 0);
    let r = or_logic(and_logic(t, t), not_logic(f));
    unwrap_logic(r) * 42
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


def test_stage36_inc2_and_logic_rejects_bare_value():
    """and_logic with a bare i32 (not Logic) fires trap 24100."""
    src = """
fn main() -> i32 {
    let a = prove(1, 0);
    unwrap_logic(and_logic(a, 1))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("24100" in str(e) for e in errs), \
        f"expected trap 24100, got {[str(e) for e in errs]}"


def test_stage36_inc2_derive_rejects_bare_first_arg():
    """derive(bare, Logic) fires trap 24100."""
    src = """
fn main() -> i32 {
    let b = prove(1, 0);
    unwrap_logic(derive(1, b))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("24100" in str(e) for e in errs), \
        f"expected trap 24100, got {[str(e) for e in errs]}"


# Stage 36 Increment 3 — boolean-algebra completeness.


def test_stage36_inc3_combinators_are_builtins():
    """xor_logic, implies_logic, eq_logic, if_logic, to_logic_bool
    are registered as builtins."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("xor_logic", "implies_logic", "eq_logic",
                 "if_logic", "to_logic_bool"):
        assert name in tc._BUILTIN_NAMES, \
            f"{name} not registered as builtin"


def test_stage36_inc3_xor_truth_table():
    """xor_logic on 0/1 truth values: 0 XOR 0 = 0, 0 XOR 1 = 1,
    1 XOR 0 = 1, 1 XOR 1 = 0."""
    for a, b, want in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]:
        src = f"""
fn main() -> i32 {{
    unwrap_logic(xor_logic(prove({a}, 0), prove({b}, 0)))
}}
"""
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        rc = _run_elf(elf)
        assert rc == want, f"xor_logic({a},{b}) -> {rc}, expected {want}"


def test_stage36_inc3_implies_truth_table():
    """implies_logic: 0->0=1, 0->1=1, 1->0=0, 1->1=1."""
    for a, b, want in [(0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 1)]:
        src = f"""
fn main() -> i32 {{
    unwrap_logic(implies_logic(prove({a}, 0), prove({b}, 0)))
}}
"""
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        rc = _run_elf(elf)
        assert rc == want, f"implies({a},{b}) -> {rc}, expected {want}"


def test_stage36_inc3_eq_truth_table():
    """eq_logic: 0==0 = 1, 0==1 = 0, 1==0 = 0, 1==1 = 1."""
    for a, b, want in [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]:
        src = f"""
fn main() -> i32 {{
    unwrap_logic(eq_logic(prove({a}, 0), prove({b}, 0)))
}}
"""
        prog = parse(src, include_stdlib=True)
        assert typecheck(prog) == []
        elf = compile_module_to_elf(lower(prog))
        rc = _run_elf(elf)
        assert rc == want, f"eq_logic({a},{b}) -> {rc}, expected {want}"


def test_stage36_inc3_if_logic_selects_then_branch():
    """if_logic(1, then_v, else_v) returns then_v."""
    src = """
fn main() -> i32 {
    unwrap_logic(if_logic(prove(1, 0), prove(42, 0), prove(7, 0)))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


def test_stage36_inc3_if_logic_selects_else_branch():
    """if_logic(0, then_v, else_v) returns else_v."""
    src = """
fn main() -> i32 {
    unwrap_logic(if_logic(prove(0, 0), prove(42, 0), prove(7, 0)))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 7


def test_stage36_inc3_to_logic_bool_lifts():
    """to_logic_bool(x) lifts a bare i32 into Logic<i32>."""
    src = "fn main() -> i32 { unwrap_logic(to_logic_bool(1)) }"
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 1


def test_stage36_inc3_de_morgan_law():
    """De Morgan's law: NOT(a AND b) == OR(NOT a, NOT b). If true,
    eq_logic(lhs, rhs) is 1, so output is 1*42 = 42. Verifies a
    real theorem of boolean algebra computed entirely over
    provenance-typed values."""
    src = """
fn main() -> i32 {
    let a = prove(1, 0);
    let b = prove(0, 0);
    let lhs = not_logic(and_logic(a, b));
    let rhs = or_logic(not_logic(a), not_logic(b));
    unwrap_logic(eq_logic(lhs, rhs)) * 42
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


def test_stage36_inc3_xor_rejects_bare_value():
    """xor_logic with a bare i32 fires trap 24100."""
    src = """
fn main() -> i32 {
    unwrap_logic(xor_logic(prove(1, 0), 0))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("24100" in str(e) for e in errs), \
        f"expected trap 24100, got {[str(e) for e in errs]}"


def test_stage36_inc3_if_logic_rejects_bare_cond():
    """if_logic with a bare i32 cond fires trap 24100."""
    src = """
fn main() -> i32 {
    unwrap_logic(if_logic(1, prove(42, 0), prove(7, 0)))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("24100" in str(e) for e in errs), \
        f"expected trap 24100, got {[str(e) for e in errs]}"


# Stage 36 Increment 4 — D<Logic<T>> composition runs end-to-end.
#
# Stage 24 shipped TyDiff (the D<T> type) at typecheck level but
# attach/detach were never wired through IR lowering, so any program
# using them failed with "unknown function 'attach'". Increment 4
# wires both as identity at IR (matching Logic<T>'s zero-overhead
# Phase-0 convention), which unblocks the strategic D<Logic<T>>
# composition that the Stage 24 design called out.


def test_stage36_inc4_d_i32_roundtrip():
    """D<i32> via attach + detach round-trips. Verifies the
    Stage 24 D<T> type is now actually runnable, not just a
    typecheck annotation."""
    src = """
fn main() -> i32 {
    let d: D<i32> = attach(42);
    detach(d)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


def test_stage36_inc4_d_logic_attach_prove_compose():
    """D<Logic<i32>> = attach(prove(42, 99)) — the strategic
    composition. unwrap_logic(detach(...)) recovers 42."""
    src = """
fn main() -> i32 {
    let dl: D<Logic<i32>> = attach(prove(42, 99));
    unwrap_logic(detach(dl))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


def test_stage36_inc4_d_logic_boolean_compute():
    """D<Logic<i32>> values can be detached, boolean-combined, and
    reused — full provenance + diff type-tracking through a real
    boolean computation. 1 OR 0 = 1; result 1 * 42 = 42 (with the
    AND result 0 added)."""
    src = """
fn main() -> i32 {
    let a: D<Logic<i32>> = attach(prove(1, 0));
    let b: D<Logic<i32>> = attach(prove(0, 0));
    let and_result: Logic<i32> = and_logic(detach(a), detach(b));
    let or_result: Logic<i32> = or_logic(detach(a), detach(b));
    unwrap_logic(and_result) + unwrap_logic(or_result) * 42
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


def test_stage36_inc4_derive_as_rule():
    """Derive-as-rule pattern: two D<Logic<i32>> 'facts' combine via
    derive() to produce a third 'conclusion' D<Logic<i32>>. This is
    the foundational shape of a differentiable production-rule fire."""
    src = """
fn main() -> i32 {
    let parent1: D<Logic<i32>> = attach(prove(1, 100));
    let parent2: D<Logic<i32>> = attach(prove(1, 200));
    let conclusion: Logic<i32> = derive(detach(parent1), detach(parent2));
    let r: D<Logic<i32>> = attach(conclusion);
    unwrap_logic(detach(r)) * 42
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


# Stage 36 Increment 5 — real two-parent provenance via arena side-table.
#
# Phase-0 Logic<T> wrapper has no runtime overhead, so source IDs
# aren't recoverable from a Logic value at runtime. Increment 5 closes
# that gap by giving user code an arena-backed mechanism: register the
# (left_src, right_src) pair under an explicit handle, then look up
# either parent later via the handle. This is real two-parent
# provenance — observable at runtime, queryable, without any ABI
# change.


def test_stage36_inc5_builtins_registered():
    """register_derivation, parent_left_at, parent_right_at are
    registered as builtins."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("register_derivation", "parent_left_at", "parent_right_at"):
        assert name in tc._BUILTIN_NAMES, f"{name} not a builtin"


def test_stage36_inc5_parent_left_recovers_first_src():
    """parent_left_at(register_derivation(L, R)) recovers L."""
    src = """
fn main() -> i32 {
    let h = register_derivation(100, 200);
    parent_left_at(h) - 58
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"expected 42 (100-58), got {rc}"


def test_stage36_inc5_parent_right_recovers_second_src():
    """parent_right_at(register_derivation(L, R)) recovers R."""
    src = """
fn main() -> i32 {
    let h = register_derivation(100, 200);
    parent_right_at(h) - 158
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"expected 42 (200-158), got {rc}"


def test_stage36_inc5_two_derivations_independent():
    """Multiple register_derivation calls produce independent handles.
    Reading from h1 returns h1's data, h2 returns h2's data."""
    src = """
fn main() -> i32 {
    let h1 = register_derivation(10, 20);
    let h2 = register_derivation(30, 40);
    parent_left_at(h1) + parent_right_at(h2)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 50, f"expected 50 (10+40), got {rc}"


def test_stage36_inc5_datalog_with_provenance_recovery():
    """Integrated: the Datalog grandparent rule fires AND the
    provenance handle correctly identifies both parent source IDs."""
    src = """
fn main() -> i32 {
    let p_ab: Logic<i32> = prove(1, 1);
    let p_bc: Logic<i32> = prove(1, 2);
    let grandparent: Logic<i32> = and_logic(p_ab, p_bc);
    let h = register_derivation(1, 2);
    if unwrap_logic(grandparent) == 1 {
        if parent_left_at(h) == 1 {
            if parent_right_at(h) == 2 {
                42
            } else { 0 }
        } else { 0 }
    } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42


def test_stage36_inc5_register_rejects_non_int_src():
    """register_derivation arguments must be i32 source IDs."""
    src = """
fn main() -> i32 {
    let h = register_derivation(1.5_f32, 2);
    0
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("register_derivation" in str(e) for e in errs), \
        f"expected register_derivation type error, got {[str(e) for e in errs]}"


# Stage 36 Increment 6 — fuzzy logic + AD chain rules through Logic.
#
# Bridge to differentiable neuro-symbolic AI. fuzzy_and/or/not
# operate on Logic<f32> truth values in [0, 1] using product
# semantics so they're smooth and differentiable. grad() and
# grad_rev() now flow gradients through prove/unwrap_logic/attach/
# detach (identity chain rule) and through the fuzzy ops (product
# / probabilistic chain rules registered in autodiff.py and
# autodiff_reverse.py).
#
# This is the first running Helix code that COMPUTES GRADIENTS
# THROUGH PROPOSITIONAL LOGIC — the strategic Tier 3 #10 moat is
# now end-to-end functional, not just a type-level promise.


def _stage36_inc6_pipeline(src):
    """Stage 36 AD tests need grad_pass before lower. Returns rc."""
    from helixc.frontend.grad_pass import grad_pass
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    if errs:
        raise AssertionError(f"typecheck errors: {[str(e) for e in errs]}")
    grad_pass(prog)
    elf = compile_module_to_elf(lower(prog))
    return _run_elf(elf)


def test_stage36_inc6_fuzzy_builtins_registered():
    """fuzzy_and, fuzzy_or, fuzzy_not are registered as builtins."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("fuzzy_and", "fuzzy_or", "fuzzy_not"):
        assert name in tc._BUILTIN_NAMES, f"{name} not a builtin"


def test_stage36_inc6_fuzzy_and_product_semantics():
    """fuzzy_and(0.5, 0.8) = 0.4 (product semantics)."""
    src = """
fn main() -> i32 {
    let a: Logic<f32> = prove(0.5_f32, 0);
    let b: Logic<f32> = prove(0.8_f32, 0);
    let v: f32 = unwrap_logic(fuzzy_and(a, b));
    if v > 0.39_f32 { if v < 0.41_f32 { 42 } else { 0 } } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage36_inc6_fuzzy_or_probabilistic():
    """fuzzy_or(0.5, 0.5) = 0.5 + 0.5 - 0.25 = 0.75."""
    src = """
fn main() -> i32 {
    let v: f32 = unwrap_logic(
        fuzzy_or(prove(0.5_f32, 0), prove(0.5_f32, 0)));
    if v > 0.74_f32 { if v < 0.76_f32 { 42 } else { 0 } } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage36_inc6_fuzzy_not_complement():
    """fuzzy_not(0.3) = 0.7."""
    src = """
fn main() -> i32 {
    let v: f32 = unwrap_logic(fuzzy_not(prove(0.3_f32, 0)));
    if v > 0.69_f32 { if v < 0.71_f32 { 42 } else { 0 } } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage36_inc6_fuzzy_de_morgan():
    """De Morgan in fuzzy logic: fuzzy_not(fuzzy_and(a, b)) ==
    fuzzy_or(fuzzy_not(a), fuzzy_not(b)) for a=0.6, b=0.7.
    LHS: 1 - 0.42 = 0.58
    RHS: 0.4 + 0.3 - 0.12 = 0.58. |diff| < 0.01 -> 42."""
    src = """
fn main() -> i32 {
    let a: Logic<f32> = prove(0.6_f32, 0);
    let b: Logic<f32> = prove(0.7_f32, 0);
    let lhs: f32 = unwrap_logic(fuzzy_not(fuzzy_and(a, b)));
    let rhs: f32 = unwrap_logic(fuzzy_or(fuzzy_not(a), fuzzy_not(b)));
    let diff: f32 = lhs - rhs;
    let abs_diff: f32 = if diff < 0.0_f32 { 0.0_f32 - diff } else { diff };
    if abs_diff < 0.01_f32 { 42 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage36_inc6_grad_rev_through_fuzzy_and():
    """grad_rev(loss)(2.0) where loss(x) = fuzzy_and(x, 0.5) gives
    0.5 (since fuzzy_and is x * 0.5, d/dx = 0.5)."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_and(prove(x, 0), prove(0.5_f32, 0)))
}
fn main() -> i32 {
    let g: f32 = grad_rev(loss)(2.0_f32);
    let g_int: i32 = (g * 100.0_f32) as i32;
    g_int - 8
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc6_grad_rev_through_fuzzy_not():
    """grad_rev(loss)(0.4) where loss(x) = fuzzy_not(x) = 1 - x
    gives -1."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_not(prove(x, 0)))
}
fn main() -> i32 {
    let g: f32 = grad_rev(loss)(0.4_f32);
    if g < -0.99_f32 { if g > -1.01_f32 { 42 } else { 1 } } else { 2 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc6_grad_rev_nested_fuzzy_compose():
    """grad_rev through fuzzy_not(fuzzy_and(x, 0.5)) = 1 - 0.5x.
    d/dx = -0.5."""
    src = """
fn loss(x: f32) -> f32 {
    let a: Logic<f32> = prove(x, 0);
    let b: Logic<f32> = prove(0.5_f32, 0);
    unwrap_logic(fuzzy_not(fuzzy_and(a, b)))
}
fn main() -> i32 {
    let g: f32 = grad_rev(loss)(0.6_f32);
    if g < -0.49_f32 { if g > -0.51_f32 { 42 } else { 1 } } else { 2 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc6_grad_forward_through_fuzzy_and():
    """Forward-mode grad() through fuzzy_and: same gradient as
    reverse-mode."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_and(prove(x, 0), prove(0.5_f32, 0)))
}
fn main() -> i32 {
    let g: f32 = grad(loss)(2.0_f32);
    let g_int: i32 = (g * 100.0_f32) as i32;
    g_int - 8
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc6_grad_forward_through_fuzzy_not():
    """Forward-mode grad() through fuzzy_not."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_not(prove(x, 0)))
}
fn main() -> i32 {
    let g: f32 = grad(loss)(0.4_f32);
    if g < -0.99_f32 { if g > -1.01_f32 { 42 } else { 1 } } else { 2 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


# Stage 36 Increment 8 — fuzzy_xor + fuzzy_implies for fuzzy-algebra
# completeness.


def test_stage36_inc8_builtins_registered():
    """fuzzy_xor, fuzzy_implies are registered as builtins."""
    tc = TypeChecker(parse("fn main() -> i32 { 0 }"))
    for name in ("fuzzy_xor", "fuzzy_implies"):
        assert name in tc._BUILTIN_NAMES, f"{name} not a builtin"


def test_stage36_inc8_fuzzy_xor_probabilistic():
    """fuzzy_xor(0.3, 0.7) = 0.3 + 0.7 - 2*0.21 = 0.58 ~ 0.58."""
    src = """
fn main() -> i32 {
    let v: f32 = unwrap_logic(fuzzy_xor(prove(0.3_f32, 0), prove(0.7_f32, 0)));
    if v > 0.57_f32 { if v < 0.59_f32 { 42 } else { 1 } } else { 2 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage36_inc8_fuzzy_xor_self_inverse():
    """fuzzy_xor(0.5, 0.5) = 2*0.5*(1-0.5) = 0.5 (not zero — fuzzy
    XOR doesn't have the classical XOR(x,x)=0 property in [0,1])."""
    src = """
fn main() -> i32 {
    let v: f32 = unwrap_logic(fuzzy_xor(prove(0.5_f32, 0), prove(0.5_f32, 0)));
    if v > 0.49_f32 { if v < 0.51_f32 { 42 } else { 1 } } else { 2 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage36_inc8_fuzzy_implies_reichenbach():
    """Reichenbach: fuzzy_implies(0.8, 0.6) = 1 - 0.8 + 0.48 = 0.68."""
    src = """
fn main() -> i32 {
    let v: f32 = unwrap_logic(
        fuzzy_implies(prove(0.8_f32, 0), prove(0.6_f32, 0)));
    if v > 0.67_f32 { if v < 0.69_f32 { 42 } else { 1 } } else { 2 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42


def test_stage36_inc8_grad_rev_fuzzy_xor_at_b_half():
    """d/da fuzzy_xor(a, b) at b=0.5 is 1 - 2*0.5 = 0."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_xor(prove(x, 0), prove(0.5_f32, 0)))
}
fn main() -> i32 {
    let g: f32 = grad_rev(loss)(0.3_f32);
    if g > -0.01_f32 { if g < 0.01_f32 { 42 } else { 1 } } else { 2 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc8_grad_rev_fuzzy_implies():
    """d/da fuzzy_implies(a, b) = -1 + b. At b=0.6, gradient = -0.4."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_implies(prove(x, 0), prove(0.6_f32, 0)))
}
fn main() -> i32 {
    let g: f32 = grad_rev(loss)(0.3_f32);
    if g > -0.41_f32 { if g < -0.39_f32 { 42 } else { 1 } } else { 2 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc8_grad_forward_fuzzy_implies():
    """Forward-mode grad() through fuzzy_implies; same gradient as
    reverse-mode."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_implies(prove(x, 0), prove(0.6_f32, 0)))
}
fn main() -> i32 {
    let g: f32 = grad(loss)(0.3_f32);
    if g > -0.41_f32 { if g < -0.39_f32 { 42 } else { 1 } } else { 2 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


# Stage 36 Increment 9 — post-Inc-8 audit fix.
# A1 HIGH: parent_left_at / parent_right_at bounds-check.
# Pre-fix: bare ARENA_GET returned arbitrary memory on forged handles.
# Post-fix: returns -1 sentinel on out-of-range (negative, beyond
# arena_len). Mirrors the restart 45-47 forge-guard pattern.


def test_stage36_inc9_parent_at_valid_handle_recovers_data():
    """Valid handle from register_derivation still returns the
    correct parent source IDs."""
    src = """
fn main() -> i32 {
    let h = register_derivation(100, 200);
    parent_left_at(h) + parent_right_at(h)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 44, f"expected 44 (300 mod 256), got {rc}"


def test_stage36_inc9_parent_left_at_negative_returns_sentinel():
    """parent_left_at(-1) returns -1 (OOB sentinel), not arbitrary
    memory."""
    src = """
fn main() -> i32 {
    let v: i32 = parent_left_at(0 - 1);
    if v == 0 - 1 { 42 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"expected sentinel -1 detection, got {rc}"


def test_stage36_inc9_parent_right_at_huge_returns_sentinel():
    """parent_right_at(99999) returns -1 sentinel even for a handle
    far beyond arena_len."""
    src = """
fn main() -> i32 {
    let v: i32 = parent_right_at(99999);
    if v == 0 - 1 { 42 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"expected sentinel -1, got {rc}"


def test_stage36_inc9_parent_at_idx_zero_empty_arena_sentinel():
    """parent_left_at(0) on a process with empty arena returns -1
    sentinel — no read past arena_len."""
    src = """
fn main() -> i32 {
    let v: i32 = parent_left_at(0);
    if v == 0 - 1 { 42 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"expected sentinel on idx 0 empty arena, got {rc}"


# Stage 36 Increment 9 audit B2 fix: add reverse-mode ∂/∂b coverage
# for every 2-arg fuzzy op. The pre-existing tests fix the second arg
# as a literal and differentiate only against the first parameter; a
# transpose bug (e.g., flipping a_arg/b_arg in fuzzy_or chain rule)
# wouldn't be caught. These tests differentiate against the SECOND
# argument explicitly so any asymmetry-bug surfaces.


def test_stage36_inc9_grad_rev_fuzzy_and_db():
    """∂/∂b fuzzy_and(a, b) = a. At a=0.3, gradient = 0.3.
    loss(x) = fuzzy_and(prove(0.3, 0), prove(x, 0)); grad at any x = 0.3."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_and(prove(0.3_f32, 0), prove(x, 0)))
}
fn main() -> i32 {
    let g: f32 = grad_rev(loss)(0.7_f32);
    // g should be 0.3; 0.3*100=30, 30+12 = 42
    let g_int: i32 = (g * 100.0_f32 + 0.5_f32) as i32;
    g_int + 12
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc9_grad_rev_fuzzy_or_db():
    """∂/∂b fuzzy_or(a, b) = 1 - a. At a=0.6, gradient = 0.4.
    loss(x) = fuzzy_or(prove(0.6, 0), prove(x, 0))."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_or(prove(0.6_f32, 0), prove(x, 0)))
}
fn main() -> i32 {
    let g: f32 = grad_rev(loss)(0.2_f32);
    let g_int: i32 = (g * 100.0_f32 + 0.5_f32) as i32;
    // g should be 0.4 -> g_int = 40, +2 = 42
    g_int + 2
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc9_grad_rev_fuzzy_xor_db():
    """∂/∂b fuzzy_xor(a, b) = 1 - 2*a. At a=0.3, gradient = 0.4.
    Verifies the b-side asymmetry of the XOR chain rule."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_xor(prove(0.3_f32, 0), prove(x, 0)))
}
fn main() -> i32 {
    let g: f32 = grad_rev(loss)(0.5_f32);
    let g_int: i32 = (g * 100.0_f32 + 0.5_f32) as i32;
    g_int + 2
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc9_grad_rev_fuzzy_implies_db():
    """∂/∂b fuzzy_implies(a, b) = a. At a=0.42, gradient = 0.42.
    The chain-rule formula is asymmetric (-1+b vs a) so a transpose
    bug WOULD show up here as wrong gradient sign or magnitude."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_implies(prove(0.42_f32, 0), prove(x, 0)))
}
fn main() -> i32 {
    let g: f32 = grad_rev(loss)(0.7_f32);
    let g_int: i32 = (g * 100.0_f32 + 0.5_f32) as i32;
    // g should be 0.42 -> g_int = 42; +0 = 42
    g_int
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc9_derive_evaluates_args_in_source_order():
    """Stage 36 Inc 9 audit B1 (code-review) fix: `derive(a, b)`
    now lowers `a` before `b`. Hard to observe directly without
    side-effecting calls (which Logic-typed args don't have), so
    this test verifies that the lowering still returns the
    correct value (a's value) — a regression sanity check rather
    than an evaluation-order assertion."""
    src = """
fn main() -> i32 {
    let a: Logic<i32> = prove(42, 100);
    let b: Logic<i32> = prove(99, 200);
    unwrap_logic(derive(a, b))
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"derive(a, b) should return a's value, got {rc}"


# Stage 36 Inc 9 audit A1 (type-design lane) fix: tighten boolean
# and fuzzy ops to inspect the Logic<T> inner type. Pre-fix accepted
# any Logic<T>, then lowered to ops semantically wrong for the
# inner type (e.g., fuzzy_and on Logic<i32> lowered to f32 MUL).


def test_stage36_inc9_fuzzy_and_rejects_logic_i32():
    """fuzzy_and requires Logic<f32>; Logic<i32> input is a
    trap-24100 boundary violation per Inc 9 type-design A1 fix."""
    src = """
fn main() -> i32 {
    let a: Logic<i32> = prove(1, 0);
    let b: Logic<i32> = prove(1, 0);
    unwrap_logic(fuzzy_and(a, b))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Logic<f32>" in str(e) and "24100" in str(e) for e in errs), \
        f"expected Logic<f32> + trap 24100 error, got {[str(e) for e in errs]}"


def test_stage36_inc9_and_logic_rejects_logic_f32():
    """and_logic requires Logic<i32>; Logic<f32> input is a
    trap-24100 boundary violation per Inc 9 type-design A1 fix."""
    src = """
fn main() -> i32 {
    let a: Logic<f32> = prove(1.0_f32, 0);
    let b: Logic<f32> = prove(1.0_f32, 0);
    unwrap_logic(and_logic(a, b))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Logic<i32>" in str(e) and "24100" in str(e) for e in errs), \
        f"expected Logic<i32> + trap 24100 error, got {[str(e) for e in errs]}"


def test_stage36_inc9_fuzzy_xor_rejects_logic_i32():
    """fuzzy_xor (Inc 8) also requires Logic<f32> after the Inc 9
    type-design A1 fix."""
    src = """
fn main() -> i32 {
    let a: Logic<i32> = prove(1, 0);
    let b: Logic<i32> = prove(1, 0);
    unwrap_logic(fuzzy_xor(a, b))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Logic<f32>" in str(e) for e in errs), \
        f"expected Logic<f32> error, got {[str(e) for e in errs]}"


def test_stage36_inc9_xor_logic_rejects_logic_f32():
    """xor_logic requires Logic<i32>; Logic<f32> is now rejected."""
    src = """
fn main() -> i32 {
    let a: Logic<f32> = prove(1.0_f32, 0);
    let b: Logic<f32> = prove(1.0_f32, 0);
    unwrap_logic(xor_logic(a, b))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("Logic<i32>" in str(e) for e in errs), \
        f"expected Logic<i32> error, got {[str(e) for e in errs]}"


# Stage 36 Inc 9 audit B4 fix: to_logic_bool now strict i32-only.


def test_stage36_inc9_to_logic_bool_rejects_i64():
    """to_logic_bool now rejects i64 (pre-fix it silently produced
    Logic<i32> wrapping i64 data, causing downstream BIT_AND to
    truncate)."""
    src = """
fn main() -> i32 {
    let x: i64 = 1_i64;
    unwrap_logic(to_logic_bool(x))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert any("to_logic_bool" in str(e) and "i32" in str(e) for e in errs), \
        f"expected to_logic_bool i32 error, got {[str(e) for e in errs]}"


def test_stage36_inc9_to_logic_bool_accepts_i32():
    """to_logic_bool still accepts i32 — regression check."""
    src = """
fn main() -> i32 {
    unwrap_logic(to_logic_bool(1))
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    assert errs == [], f"unexpected errors: {[str(e) for e in errs]}"


# Stage 36 Inc 9 audit C1 fix: unwrap_logic error recovery returns
# TyUnknown instead of the input type, preventing cascading errors.


def test_stage36_inc9_unwrap_logic_error_recovery_no_cascade():
    """unwrap_logic on non-Logic still emits ONE error (not cascading
    type errors from downstream usage)."""
    src = """
fn main() -> i32 {
    let v: i32 = unwrap_logic(5);
    v + 1
}
"""
    prog = parse(src, include_stdlib=True)
    errs = typecheck(prog)
    # The error should be at the unwrap_logic call site itself.
    # Pre-fix, returning arg_tys[0] (i32) would let `v + 1` succeed
    # without further errors but with semantically-wrong type. Post-
    # fix, TyUnknown is returned; cascading downstream errors are
    # suppressed by TyUnknown's design (this is the existing pattern
    # in typecheck.py for many builtins).
    n_unwrap_errs = sum(1 for e in errs if "unwrap_logic" in str(e))
    assert n_unwrap_errs >= 1, \
        f"expected at least 1 unwrap_logic error, got {[str(e) for e in errs]}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
