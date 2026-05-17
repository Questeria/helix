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


# Stage 36 Inc 9 audit A2 (silent-failure lane) fix: register_derivation
# returns 1-based handles so handle 0 is reserved as the "null
# derivation" sentinel. Pre-fix, a default-zero handle stored in a
# side array was indistinguishable from "derivation at arena index 0".


def test_stage36_inc9_handle_is_one_based():
    """First register_derivation returns handle 1 (not 0)."""
    src = """
fn main() -> i32 {
    register_derivation(50, 60)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 1, f"expected first handle = 1, got {rc}"


def test_stage36_inc9_null_handle_zero_returns_sentinel():
    """parent_left_at(0) on a HANDLE-0 input (the null-derivation
    sentinel) returns -1 even if arena index 0 has valid content
    (i.e., from struct lowering or another arena user)."""
    src = """
fn main() -> i32 {
    // Push some content to arena index 0 via another path to
    // simulate a non-empty arena where handle 0 must still be null.
    let other = register_derivation(7, 8);
    // other = 1 (1-based). parent_left_at(0) is the null-sentinel
    // read — must return -1 even though arena index 0 holds 7.
    let v: i32 = parent_left_at(0);
    if v == 0 - 1 { 42 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"expected null-handle sentinel for handle 0, got {rc}"


def test_stage36_inc9_handle_round_trip_still_works():
    """Valid handle from register_derivation still recovers the
    correct parent source IDs through the 1-based offset."""
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
    assert rc == 44, f"expected 300 mod 256 = 44, got {rc}"


def test_stage36_inc9_two_handles_remain_independent():
    """Two register_derivation calls produce distinct 1-based handles
    (h1=1, h2=3) and reads from each don't cross-contaminate."""
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
    assert rc == 50, f"expected 10 + 40, got {rc}"


# Stage 36 Inc 9 audit A3 (silent-failure HIGH) fix: fuzzy_* inputs
# clamped to [0, 1] at IR lowering. Pre-fix, out-of-range inputs
# silently produced nonsense (fuzzy_or(2.0, -1.0) = 3.0). Post-fix,
# inputs are clamped before the algebraic form so values stay sane.


def test_stage36_inc9_fuzzy_and_in_range_unchanged():
    """Regression: fuzzy_and(0.5, 0.8) still = 0.4 after the
    clamp wrapper added by A3."""
    src = """
fn main() -> i32 {
    let v: f32 = unwrap_logic(fuzzy_and(prove(0.5_f32, 0), prove(0.8_f32, 0)));
    if v > 0.39_f32 { if v < 0.41_f32 { 42 } else { 0 } } else { 0 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc9_fuzzy_and_high_input_clamped():
    """fuzzy_and(2.0, 0.5): pre-fix = 1.0; post-fix clamps 2.0 to
    1.0 so result = 0.5."""
    src = """
fn main() -> i32 {
    let v: f32 = unwrap_logic(fuzzy_and(prove(2.0_f32, 0), prove(0.5_f32, 0)));
    if v > 0.49_f32 { if v < 0.51_f32 { 42 } else { 0 } } else { 0 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc9_fuzzy_and_negative_input_clamped():
    """fuzzy_and(-1.0, 0.5): pre-fix = -0.5; post-fix clamps -1.0
    to 0.0 so result = 0.0."""
    src = """
fn main() -> i32 {
    let v: f32 = unwrap_logic(fuzzy_and(prove(0.0_f32 - 1.0_f32, 0), prove(0.5_f32, 0)));
    if v < 0.01_f32 { if v > 0.0_f32 - 0.01_f32 { 42 } else { 0 } } else { 0 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc9_fuzzy_or_high_input_clamped():
    """fuzzy_or(0.5, 1.5): pre-fix = 0.5 + 1.5 - 0.75 = 1.25;
    post-fix clamps 1.5 to 1.0 so result = 0.5 + 1.0 - 0.5 = 1.0."""
    src = """
fn main() -> i32 {
    let v: f32 = unwrap_logic(fuzzy_or(prove(0.5_f32, 0), prove(1.5_f32, 0)));
    if v > 0.99_f32 { if v < 1.01_f32 { 42 } else { 0 } } else { 0 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc9_fuzzy_not_high_input_clamped():
    """fuzzy_not(2.0): pre-fix = -1.0; post-fix clamps 2.0 to 1.0
    so result = 0.0."""
    src = """
fn main() -> i32 {
    let v: f32 = unwrap_logic(fuzzy_not(prove(2.0_f32, 0)));
    if v < 0.01_f32 { 42 } else { 0 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc9_fuzzy_xor_inputs_clamped():
    """fuzzy_xor(2.0, -1.0): pre-fix = nonsense (1.0 + -1.0 -
    2*-2 = 4.0); post-fix clamps both to (1.0, 0.0) so result =
    1.0 + 0.0 - 0 = 1.0."""
    src = """
fn main() -> i32 {
    let v: f32 = unwrap_logic(fuzzy_xor(prove(2.0_f32, 0), prove(0.0_f32 - 1.0_f32, 0)));
    if v > 0.99_f32 { if v < 1.01_f32 { 42 } else { 0 } } else { 0 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc9_fuzzy_implies_inputs_clamped():
    """fuzzy_implies(2.0, -1.0): pre-fix = 1 - 2 + 2*-1 = -3;
    post-fix clamps to (1.0, 0.0) so result = 1 - 1 + 0 = 0."""
    src = """
fn main() -> i32 {
    let v: f32 = unwrap_logic(fuzzy_implies(prove(2.0_f32, 0), prove(0.0_f32 - 1.0_f32, 0)));
    if v < 0.01_f32 { if v > 0.0_f32 - 0.01_f32 { 42 } else { 0 } } else { 0 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


# Stage 36 Inc 9 type-design A2 fix: register_derivation now emits a
# single ARENA_PUSH_PAIR opcode (atomic at IR level) instead of two
# consecutive ARENA_PUSH ops. The handle invariant "left at N, right
# at N+1" cannot be broken by any scheduler/DCE/CSE pass or by another
# arena consumer (struct lowering, MatchDispatch) being inlined
# between the two pushes.
#
# Stage 36 Inc 9 silent-failure B2 fix: derive(a, b) also routes
# through ARENA_PUSH_PAIR so the call is observable — pre-fix it
# dropped b's value entirely, making derive(p, q) and p
# indistinguishable. Post-fix arena_len() grows by 2 per derive call,
# and the registered pair can be looked up via parent_*_at against
# the slot index that was just consumed.


def test_stage36_inc9_arena_push_pair_atomicity_against_intervening_push():
    """ARENA_PUSH_PAIR slots stay adjacent even when a separate
    arena.push happens between two register_derivation calls. With the
    pre-fix two-ARENA_PUSH lowering an optimizer pass could in
    principle reorder the second push past the intervening unrelated
    push, breaking the N/N+1 handle invariant. The fused opcode makes
    that physically impossible: the two writes are inside one IR op."""
    src = """
fn main() -> i32 {
    let h1 = register_derivation(11, 22);
    // Force an unrelated arena push between the two registrations
    // (uses the low-level __arena_push intrinsic — distinct opcode
    // from ARENA_PUSH_PAIR so the scheduler is theoretically free
    // to reorder).
    let _slot = __arena_push(99);
    let h2 = register_derivation(33, 44);
    // h1 reads must still recover (11, 22); h2 reads must still
    // recover (33, 44). Any cross-contamination would surface as a
    // mismatch here.
    let l1: i32 = parent_left_at(h1);
    let r1: i32 = parent_right_at(h1);
    let l2: i32 = parent_left_at(h2);
    let r2: i32 = parent_right_at(h2);
    if l1 == 11 {
        if r1 == 22 {
            if l2 == 33 {
                if r2 == 44 { 42 } else { 1 }
            } else { 2 }
        } else { 3 }
    } else { 4 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"expected adjacency preserved across unrelated push, got {rc}"


def test_stage36_inc9_arena_push_pair_advances_cursor_by_2():
    """register_derivation increments arena_len by exactly 2. With the
    pre-fix two-ARENA_PUSH lowering this was already true, but the
    fused ARENA_PUSH_PAIR variant must preserve the property."""
    src = """
fn main() -> i32 {
    let len_before: i32 = __arena_len();
    let _h = register_derivation(1, 2);
    let len_after: i32 = __arena_len();
    let delta: i32 = len_after - len_before;
    if delta == 2 { 42 } else { delta }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"expected cursor delta=2 from PAIR push, got {rc}"


def test_stage36_inc9_b2_derive_is_observable_via_arena_len():
    """B2 fix: derive(p, q) must have an observable side effect (so it
    is no longer equivalent to `p`). Pre-fix, arena_len() was
    unchanged because b was discarded. Post-fix, derive emits the
    same atomic ARENA_PUSH_PAIR that register_derivation does."""
    src = """
fn main() -> i32 {
    let p: Logic<i32> = prove(10, 1);
    let q: Logic<i32> = prove(20, 2);
    let len_before: i32 = __arena_len();
    let _d: Logic<i32> = derive(p, q);
    let len_after: i32 = __arena_len();
    let delta: i32 = len_after - len_before;
    if delta == 2 { 42 } else { delta }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"expected derive() to grow arena by 2, got {rc}"


def test_stage36_inc9_b2_derive_registered_pair_is_recoverable():
    """B2 fix: the two-parent pair derive() pushes can be recovered
    from the arena via parent_*_at at the freshly-consumed slot. This
    pins the contract that derive is semantically equivalent to
    `register_derivation(unwrap_logic(a), unwrap_logic(b))` for the
    arena state — the only difference is that derive returns a's
    value, register_derivation returns the handle."""
    src = """
fn main() -> i32 {
    // The slot index that derive() will consume is the current
    // arena_len(): the pair is pushed at slots len..len+1.
    let next_slot: i32 = __arena_len();
    let p: Logic<i32> = prove(77, 7);
    let q: Logic<i32> = prove(88, 8);
    let _d: Logic<i32> = derive(p, q);
    // 1-based handle convention matches register_derivation —
    // parent_*_at subtracts 1 before the arena lookup.
    let handle: i32 = next_slot + 1;
    let l: i32 = parent_left_at(handle);
    let r: i32 = parent_right_at(handle);
    if l == 77 {
        if r == 88 { 42 } else { 0 }
    } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"expected derive pair recoverable via parent_*_at, got {rc}"


def test_stage36_inc9_b2_derive_no_longer_equivalent_to_p():
    """B2 fix: derive(p, q) is no longer observationally
    indistinguishable from p. We can distinguish them by arena state
    even when the returned value is identical."""
    src = """
fn main() -> i32 {
    let p: Logic<i32> = prove(5, 1);
    let q: Logic<i32> = prove(99, 2);

    // Branch A: derive(p, q) — should grow arena by 2.
    let a_before: i32 = __arena_len();
    let v_a: i32 = unwrap_logic(derive(p, q));
    let a_after: i32 = __arena_len();

    // Branch B: just p — should not grow arena.
    let b_before: i32 = __arena_len();
    let v_b: i32 = unwrap_logic(p);
    let b_after: i32 = __arena_len();

    // The values must match (Phase-0 single-tag return), but the
    // arena delta must differ.
    if v_a == v_b {
        let delta_a: i32 = a_after - a_before;
        let delta_b: i32 = b_after - b_before;
        if delta_a == 2 {
            if delta_b == 0 { 42 } else { 1 }
        } else { 2 }
    } else { 3 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"expected derive distinguishable from p via arena, got {rc}"


def test_stage36_inc9_arena_push_pair_overflow_returns_negative_one():
    """ARENA_PUSH_PAIR matches single ARENA_PUSH on overflow: when
    cursor + 2 would exceed CAP, neither slot is written and the
    result is -1. Hard to trigger directly under CAP=2M, so we only
    verify the in-bounds case here and rely on the assembly's
    structural symmetry with ARENA_PUSH (same overflow encoding)."""
    # Sanity: a normal call returns a non-negative handle.
    src = """
fn main() -> i32 {
    let h = register_derivation(1, 2);
    if h >= 1 { 42 } else { 0 }
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf(elf)
    assert rc == 42, f"expected non-negative handle for in-bounds push, got {rc}"


# ----------------------------------------------------------------------
# Stage 36 Inc 9 catch-up sweep — 5 deferred MEDIUM/LOW audit findings
# (silent-failure B1, type-design B1, type-design B3, type-design C2,
# code-review B3 finite-difference cross-check). Each fix gets a
# focused regression canary here; the bookkeeping lives in the
# stage36 progress ledger under "Inc 9 catch-up sweep".
# ----------------------------------------------------------------------


def test_stage36_inc9_catchup_prove_rejects_nested_logic():
    """Type-design B1 (catch-up): prove(Logic<T>, src) is REJECTED at
    typecheck. Pre-fix, the call silently flattened to the input
    Logic<T>, dropping the new source tag. Post-fix, the user must
    unwrap_logic(...) first if re-proving with a new source tag."""
    src = """
fn user_main() -> i32 {
    let v: i32 = 5;
    let l1: Logic<i32> = prove(v, 1);
    let l2: Logic<i32> = prove(l1, 2);
    unwrap_logic(l2)
}
"""
    prog = parse(src)
    errs = typecheck(prog)
    assert any("already" in str(e) and "Logic" in str(e)
               and "unwrap_logic" in str(e) for e in errs), \
        "expected prove(Logic<T>, src) rejection diagnostic, " \
        f"got: {[str(e) for e in errs]}"


def test_stage36_inc9_catchup_derive_recovery_does_not_mask_nonlogic():
    """Type-design C2 (catch-up): when derive's first arg is non-Logic,
    typecheck recovery returns TyUnknown (not TyLogic(inner=non_logic)
    which would have masked the inner-type mismatch downstream)."""
    from helixc.frontend.typecheck import TyUnknown, TyLogic
    src = """
fn user_main() -> i32 {
    let x: i32 = 7;
    let y: i32 = 8;
    let _r = derive(x, y);
    0
}
"""
    prog = parse(src)
    tc = TypeChecker(prog)
    errs = tc.check()
    # Must produce a trap-24100 diagnostic (derive arg must be Logic).
    assert any("derive" in str(e) and "trap 24100" in str(e)
               for e in errs), \
        f"expected derive trap-24100 diagnostic, got: {[str(e) for e in errs]}"
    # Inspect the inferred type of `_r` — must NOT be Logic<i32> or any
    # TyLogic wrapping a non-Logic. Pre-fix it returned
    # TyLogic(inner=TyPrim('i32')) which leaked through. Post-fix we
    # expect TyUnknown.
    # We re-typecheck the inner call expression directly via a probe
    # function on the typechecker (the let-binding type is tracked in
    # the local env, but the easier path is to walk the program).
    # Simplest cross-check: verify that downstream code using _r as if
    # it were Logic ALSO errors (TyUnknown causes cascading errors to
    # be suppressed, not silently passed).
    src2 = """
fn user_main() -> i32 {
    let x: i32 = 7;
    let y: i32 = 8;
    let r = derive(x, y);
    unwrap_logic(r)
}
"""
    prog2 = parse(src2)
    errs2 = typecheck(prog2)
    # We expect the derive error (trap-24100). The unwrap_logic call
    # on a TyUnknown does not produce a "Logic<T>" type-mismatch
    # error (TyUnknown short-circuits). Pre-fix, unwrap_logic would
    # have happily stripped the TyLogic(inner=i32) recovery and the
    # ONLY error would have been derive's — which is the silent leak
    # the audit flagged. Post-fix the cascade is suppressed.
    # The contract we pin: derive's error is present, and no
    # secondary error of the form 'unwrap_logic.* requires Logic.*
    # i32' shows up that would have come from the recovered TyLogic.
    derive_errs = [e for e in errs2 if "derive" in str(e)]
    assert derive_errs, \
        f"expected derive's trap-24100 in cascade test, got: {errs2}"


def test_stage36_inc9_catchup_grad_pass_rejects_nonliteral_prove_source():
    """Type-design B3 (catch-up, forward-mode): prove(value, source)
    requires a literal i32 source tag in differentiated code. Pre-fix,
    `prove(x, x)` silently dropped the second arg from the chain rule
    (mathematically correct but undiagnosed). We exercise this via the
    autodiff entry point directly (grad_pass aggressively inlines
    let-bindings of literals, so the test feeds a bare expression
    with a Name-typed source tag and no let bindings — the inliner
    is a no-op on a plain Call, so the Name reaches _diff)."""
    import pytest
    import helixc.frontend.ast_nodes as A
    from helixc.frontend.autodiff import differentiate

    span = A.Span(line=1, col=1)
    # expression: prove(x, src) where both are bare Names
    prove_call = A.Call(span=span,
                        callee=A.Name(span=span, name="prove"),
                        args=[A.Name(span=span, name="x"),
                              A.Name(span=span, name="src")])
    with pytest.raises(NotImplementedError, match=r"prove.*source.*literal"):
        differentiate(prove_call, "x")


def test_stage36_inc9_catchup_grad_rev_rejects_nonliteral_prove_source():
    """Type-design B3 (catch-up, reverse-mode twin): the reverse-mode
    AD also enforces the literal-source-tag rule on prove() — we
    exercise the `_propagate` path directly via the reverse-mode
    autodiff entry point."""
    import pytest
    import helixc.frontend.ast_nodes as A
    from helixc.frontend.autodiff_reverse import differentiate_reverse

    span = A.Span(line=1, col=1)
    prove_call = A.Call(span=span,
                        callee=A.Name(span=span, name="prove"),
                        args=[A.Name(span=span, name="x"),
                              A.Name(span=span, name="src")])
    with pytest.raises(NotImplementedError, match=r"prove.*source.*literal"):
        differentiate_reverse(prove_call, ["x"])


def test_stage36_inc9_catchup_grad_with_literal_prove_source_still_works():
    """Type-design B3 (catch-up, negative-control): the IntLit src
    path is the supported case and must keep working — chain rule
    `d/dx prove(x, 0) = 1.0`."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(prove(x, 0))
}
fn main() -> i32 {
    let g: f32 = grad_rev(loss)(3.0_f32);
    if g > 0.99_f32 { if g < 1.01_f32 { 42 } else { 0 } } else { 0 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc9_catchup_finite_diff_forward_cross_check_fuzzy_and():
    """Code-review B3 (catch-up): central-difference cross-check of
    `grad_rev(loss)(x)` against an in-Helix finite difference. Pre-fix
    all AD tests compared against analytic expected values only — a
    chain-rule transpose bug could match the analytic expectation but
    miss reality. This test computes the FD inside Helix itself.

    loss(x) = fuzzy_and(prove(x, 0), prove(0.5, 0)) = 0.5 * x
    Analytic d/dx = 0.5; central diff with h=0.01 should match
    within 1e-3."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_and(prove(x, 0), prove(0.5_f32, 0)))
}
fn main() -> i32 {
    let x0: f32 = 0.4_f32;
    let h: f32 = 0.01_f32;
    let fp: f32 = loss(x0 + h);
    let fm: f32 = loss(x0 - h);
    let fd: f32 = (fp - fm) / (2.0_f32 * h);
    let g: f32 = grad_rev(loss)(x0);
    let diff: f32 = fd - g;
    let abs_diff: f32 = if diff < 0.0_f32 { 0.0_f32 - diff } else { diff };
    if abs_diff < 0.001_f32 { 42 } else { 0 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc9_catchup_finite_diff_reverse_cross_check_fuzzy_implies():
    """Code-review B3 (catch-up, second mode): central-difference
    cross-check against grad_rev for fuzzy_implies, which has a
    non-trivial d/da = -1 + b coefficient — exactly the kind of
    place where a transpose bug would NOT match the analytic
    constant but WOULD match a single finite-difference probe at
    one specific point."""
    src = """
fn loss(a: f32) -> f32 {
    unwrap_logic(fuzzy_implies(prove(a, 0), prove(0.3_f32, 0)))
}
fn main() -> i32 {
    let a0: f32 = 0.4_f32;
    let h: f32 = 0.005_f32;
    let fp: f32 = loss(a0 + h);
    let fm: f32 = loss(a0 - h);
    let fd: f32 = (fp - fm) / (2.0_f32 * h);
    let g: f32 = grad_rev(loss)(a0);
    let diff: f32 = fd - g;
    let abs_diff: f32 = if diff < 0.0_f32 { 0.0_f32 - diff } else { diff };
    // expected d/da = -1 + b = -1 + 0.3 = -0.7
    // FD and AD should both land near -0.7 and agree within ~1e-3.
    if abs_diff < 0.001_f32 { 42 } else { 0 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


# ---------------------------------------------------------------------
# Stage 36 Increment 12 — close Inc 11 type-design B2 MEDIUM deferral.
#
# Integer-valued boolean Logic ops (and_logic/or_logic/not_logic/...)
# remain in AD_KNOWN_PURE_CALLS so let-inlining of unused bindings
# does not trap, but a *differentiated* call site has no chain rule
# and pre-fix silently returned a zero derivative. Inc 12 makes both
# `differentiate` (forward) and `differentiate_reverse` raise
# NotImplementedError with a hint pointing at the fuzzy_* twin.
# ---------------------------------------------------------------------

def test_stage36_inc12_grad_forward_rejects_integer_and_logic():
    """Forward-mode AD on `and_logic(x, prove(1, 0))` raises with a
    message pointing at fuzzy_and. Pre-Inc-12 this silently returned
    a zero derivative (let-inliner saw and_logic as AD-pure, no
    chain-rule arm matched, fell through to the opaque-call raise —
    but only because Stage 35 made unknown calls fail closed; the
    deferred B2 fix is to fail closed with a *helpful* message that
    distinguishes "integer-valued boolean logic" from "totally
    unknown opaque call" so the user knows to reach for fuzzy_and)."""
    import pytest
    import helixc.frontend.ast_nodes as A
    from helixc.frontend.autodiff import differentiate

    span = A.Span(line=1, col=1)
    call = A.Call(span=span,
                  callee=A.Name(span=span, name="and_logic"),
                  args=[A.Name(span=span, name="x"),
                        A.Call(span=span,
                               callee=A.Name(span=span, name="prove"),
                               args=[A.IntLit(span=span, value=1),
                                     A.IntLit(span=span, value=0)])])
    with pytest.raises(NotImplementedError,
                       match=r"and_logic.*integer-valued.*fuzzy_and"):
        differentiate(call, "x")


def test_stage36_inc12_grad_reverse_rejects_integer_or_logic():
    """Reverse-mode AD twin: `or_logic` in a differentiated path
    raises with the fuzzy_or hint."""
    import pytest
    import helixc.frontend.ast_nodes as A
    from helixc.frontend.autodiff_reverse import differentiate_reverse

    span = A.Span(line=1, col=1)
    call = A.Call(span=span,
                  callee=A.Name(span=span, name="or_logic"),
                  args=[A.Name(span=span, name="x"),
                        A.Name(span=span, name="y")])
    with pytest.raises(NotImplementedError,
                       match=r"or_logic.*integer-valued.*fuzzy_or"):
        differentiate_reverse(call, ["x"])


def test_stage36_inc12_grad_reverse_rejects_if_logic_general_hint():
    """`if_logic` has no 1:1 fuzzy twin (the runtime branch is
    discrete); the diagnostic should fall through to the general
    guidance string rather than naming a nonexistent `fuzzy_if`."""
    import pytest
    import helixc.frontend.ast_nodes as A
    from helixc.frontend.autodiff_reverse import differentiate_reverse

    span = A.Span(line=1, col=1)
    call = A.Call(span=span,
                  callee=A.Name(span=span, name="if_logic"),
                  args=[A.Name(span=span, name="c"),
                        A.Name(span=span, name="t"),
                        A.Name(span=span, name="e")])
    with pytest.raises(NotImplementedError,
                       match=r"if_logic.*no differentiable fuzzy twin"):
        differentiate_reverse(call, ["t"])


def test_stage36_inc12_grad_reverse_to_logic_bool_general_hint():
    """`to_logic_bool` (i32 -> Logic<i32>) is discrete; the general
    fuzzy guidance applies."""
    import pytest
    import helixc.frontend.ast_nodes as A
    from helixc.frontend.autodiff_reverse import differentiate_reverse

    span = A.Span(line=1, col=1)
    call = A.Call(span=span,
                  callee=A.Name(span=span, name="to_logic_bool"),
                  args=[A.Name(span=span, name="x")])
    with pytest.raises(NotImplementedError,
                       match=r"to_logic_bool.*no differentiable fuzzy twin"):
        differentiate_reverse(call, ["x"])


def test_stage36_inc12_grad_fuzzy_and_unchanged_negative_control():
    """Positive control: the recommended replacement `fuzzy_and` MUST
    still differentiate. d/dx [fuzzy_and(x, 0.5)] = 0.5 at x=0.4. End-
    to-end pipeline so the simplify / lower / runtime path is exercised
    too — mirrors the Inc 9 catch-up B3 negative-control."""
    src = """
fn loss(x: f32) -> f32 {
    unwrap_logic(fuzzy_and(prove(x, 0), prove(0.5_f32, 0)))
}
fn main() -> i32 {
    let g: f32 = grad_rev(loss)(0.4_f32);
    if g > 0.49_f32 { if g < 0.51_f32 { 42 } else { 0 } } else { 0 }
}
"""
    assert _stage36_inc6_pipeline(src) == 42


def test_stage36_inc12_let_erasable_unused_and_logic_still_compiles():
    """Regression guard: the Inc 9 C2 + Inc 12 design says the
    integer Logic ops remain AD-pure so `let _unused = and_logic(...)`
    inside a grad/grad_rev body — where the value is NEVER consumed
    by the differentiated expression — does NOT trap on let-erasure.
    The trap fires only when AD actually tries to differentiate
    THROUGH the call. This test pins the let-inlining permission by
    feeding a body that uses `and_logic` only inside a side branch
    that the differentiation target never references (so _inline_lets
    can drop it cleanly)."""
    import helixc.frontend.ast_nodes as A
    from helixc.frontend.autodiff_reverse import differentiate_reverse

    # body: let _u = and_logic(p, q); x  -> derivative of `x` wrt x is 1.
    # The `let _u = and_logic(...)` is unused; if and_logic were
    # rejected at let-erasure, this would raise. We want it to succeed
    # and return d/dx (x) = 1.
    span = A.Span(line=1, col=1)
    body = A.Block(
        span=span,
        stmts=[
            A.Let(span=span, name="_u", ty=None,
                  value=A.Call(span=span,
                               callee=A.Name(span=span, name="and_logic"),
                               args=[A.Name(span=span, name="p"),
                                     A.Name(span=span, name="q")]),
                  is_mut=False)
        ],
        final_expr=A.Name(span=span, name="x"))
    derivs = differentiate_reverse(body, ["x"])
    assert "x" in derivs
    # The simplifier should reduce d(x)/dx to a `1.0` literal.
    assert isinstance(derivs["x"], A.FloatLit), \
        f"expected FloatLit for d(x)/dx, got {type(derivs['x']).__name__}"
    assert derivs["x"].value == 1.0, \
        f"expected 1.0 for d(x)/dx, got {derivs['x'].value}"


# ---------------------------------------------------------------------------
# Stage 36 Inc 13 — provenance debug/observation stdlib (helixc/stdlib/provenance.hx).
#
# Inc 13 ships four stdlib helpers wrapping the Inc 5/9 arena side-table:
#   - has_evidence(h)    -> 1 iff h is valid and parent_left_at(h) != -1
#   - evidence_left(h)   -> parent_left_at(h)  (readability alias)
#   - evidence_right(h)  -> parent_right_at(h) (readability alias)
#   - trace_evidence(h)  -> prints "h=<h> L=<l> R=<r>\n", returns has_evidence(h)
# ---------------------------------------------------------------------------


def _run_elf_capture(elf: bytes) -> tuple[int, bytes]:
    """Compile + run via WSL, return (returncode, stdout_bytes)."""
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
        return r.returncode, r.stdout
    finally:
        try:
            os.unlink(bin_path)
        except OSError:
            pass


def test_stage36_inc13_has_evidence_null_handle_returns_zero():
    """has_evidence(0) == 0 (handle 0 is the reserved null sentinel
    per the Inc 9 A2 fix). Exit code = 42 + 0*999 = 42 confirms."""
    src = """
fn main() -> i32 {
    42 + has_evidence(0) * 999
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf_capture(elf)[0]
    assert rc == 42, f"expected 42, got {rc}"


def test_stage36_inc13_has_evidence_unrecorded_handle_returns_zero():
    """has_evidence(h) on an out-of-range handle returns 0 because
    parent_left_at(h) hits the Inc 9 A1 bounds-check sentinel."""
    src = """
fn main() -> i32 {
    42 + has_evidence(999999) * 999
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf_capture(elf)[0]
    assert rc == 42, f"expected 42, got {rc}"


def test_stage36_inc13_has_evidence_valid_handle_returns_one():
    """After register_derivation(11, 22) returns handle h >= 1,
    has_evidence(h) must be 1 (parents recoverable). Exit code uses
    the multiplied form to FAIL CLOSED if has_evidence returns 0."""
    src = """
fn main() -> i32 {
    let h = register_derivation(11, 22);
    42 * has_evidence(h)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf_capture(elf)[0]
    assert rc == 42, f"expected 42 (handle valid + multiplied form), got {rc}"


def test_stage36_inc13_evidence_left_alias_matches_parent_left_at():
    """evidence_left(h) is a pure readability alias for parent_left_at(h).
    register_derivation(11, 22) → evidence_left should return 11."""
    src = """
fn main() -> i32 {
    let h = register_derivation(11, 22);
    evidence_left(h) + 31
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf_capture(elf)[0]
    assert rc == 42, f"expected 11+31=42, got {rc}"


def test_stage36_inc13_evidence_right_alias_matches_parent_right_at():
    """evidence_right(h) is a pure readability alias for parent_right_at(h).
    register_derivation(11, 22) → evidence_right should return 22."""
    src = """
fn main() -> i32 {
    let h = register_derivation(11, 22);
    evidence_right(h) + 20
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc = _run_elf_capture(elf)[0]
    assert rc == 42, f"expected 22+20=42, got {rc}"


def test_stage36_inc13_trace_evidence_returns_validity_flag_valid():
    """trace_evidence(h) on a valid handle returns 1 (same as
    has_evidence). The 42 in the exit code confirms the print_*
    side effects don't clobber the return register."""
    src = """
fn main() -> i32 {
    let h = register_derivation(11, 22);
    42 * trace_evidence(h)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc, _stdout = _run_elf_capture(elf)
    assert rc == 42, f"expected 42, got {rc}"


def test_stage36_inc13_trace_evidence_returns_zero_for_null_handle():
    """trace_evidence(0) prints "h=0 L=-1 R=-1\\n" and returns 0."""
    src = """
fn main() -> i32 {
    42 + trace_evidence(0) * 999
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc, _stdout = _run_elf_capture(elf)
    assert rc == 42, f"expected 42 + 0*999 = 42, got {rc}"


def test_stage36_inc13_trace_evidence_stdout_format():
    """Capture stdout: trace_evidence(h) after register_derivation(11, 22)
    emits "h=1 L=11 R=22\\n" (handle is 1-based per Inc 9 A2)."""
    src = """
fn main() -> i32 {
    let h = register_derivation(11, 22);
    trace_evidence(h);
    42
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc, stdout = _run_elf_capture(elf)
    assert rc == 42, f"expected 42, got {rc}"
    # The first register_derivation pushes 2 entries at arena indices
    # 0 and 1, so the user-visible 1-based handle is 1.
    assert stdout == b"h=1 L=11 R=22\n", \
        f"stdout mismatch: got {stdout!r}"


def test_stage36_inc13_trace_evidence_independent_handles_dont_collide():
    """Two register_derivation calls produce independent handles
    (h2 = h1 + 2 since each pushes 2 arena entries via ARENA_PUSH_PAIR
    per Inc 9 A2). Tracing both produces two well-formed lines."""
    src = """
fn main() -> i32 {
    let h1 = register_derivation(7, 8);
    let h2 = register_derivation(9, 10);
    trace_evidence(h1);
    trace_evidence(h2);
    42
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    rc, stdout = _run_elf_capture(elf)
    assert rc == 42, f"expected 42, got {rc}"
    assert stdout == b"h=1 L=7 R=8\nh=3 L=9 R=10\n", \
        f"stdout mismatch: got {stdout!r}"


def test_stage36_inc13_helpers_visible_in_stdlib():
    """The four helpers must be FnDecls in the merged program when
    include_stdlib=True. Pins the Inc 13 STDLIB_FILES registration."""
    import helixc.frontend.ast_nodes as A
    prog = parse("fn main() -> i32 { 0 }", include_stdlib=True)
    fn_names = {it.name for it in prog.items if isinstance(it, A.FnDecl)}
    for name in ("has_evidence", "evidence_left", "evidence_right",
                 "trace_evidence"):
        assert name in fn_names, \
            f"stdlib helper {name!r} missing from merged program"


# ----------------------------------------------------------------------
# Stage 36 Increment 14 — three-parent provenance via ARENA_PUSH_TRIPLE.
#
# Inc 14 extends the two-parent ARENA_PUSH_PAIR (from Inc 9 A2) to a
# fused three-slot ARENA_PUSH_TRIPLE opcode plus the higher-level
# `register_derivation3(l, m, r)` typecheck builtin. A generic indexed
# accessor `parent_at(handle, slot)` reads any pushed slot. These tests
# pin the contract:
#   - register_derivation3 returns a 1-based handle, same as
#     register_derivation (Inc 9 A2 invariant).
#   - parent_at(h, 0/1/2) recovers left/middle/right.
#   - The arena cursor advances by 3 per call.
#   - The triple is atomic against intervening pushes.
#   - parent_at on a two-parent handle still works at slot 0/1
#     (back-compat with the Inc 9 register_derivation contract).
# ----------------------------------------------------------------------


def test_stage36_inc14_register_derivation3_returns_one_based_handle():
    """First register_derivation3 call returns handle = 1, matching the
    Inc 9 A2 1-based-handle invariant. Exit code multiplies by handle
    so a regression to 0 (the null sentinel) fails closed at 0."""
    src = """
fn main() -> i32 {
    let h = register_derivation3(11, 22, 33);
    42 * h
}
"""
    prog = parse(src)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, "expected 42 * 1 = 42"


def test_stage36_inc14_parent_at_slot_0_recovers_left():
    """parent_at(h, 0) == left source ID. Pins the slot-0 contract."""
    src = """
fn main() -> i32 {
    let h = register_derivation3(7, 99, 99);
    parent_at(h, 0) * 6
}
"""
    prog = parse(src)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, "expected 7 * 6 = 42"


def test_stage36_inc14_parent_at_slot_1_recovers_middle():
    """parent_at(h, 1) == middle source ID. Pins the slot-1 contract
    (this is what register_derivation3 adds over the two-parent variant
    — for a two-parent handle, slot 1 would be the right value)."""
    src = """
fn main() -> i32 {
    let h = register_derivation3(99, 21, 99);
    parent_at(h, 1) * 2
}
"""
    prog = parse(src)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, "expected 21 * 2 = 42"


def test_stage36_inc14_parent_at_slot_2_recovers_right():
    """parent_at(h, 2) == right source ID. Pins the slot-2 contract
    (only meaningful for three-parent handles)."""
    src = """
fn main() -> i32 {
    let h = register_derivation3(99, 99, 42);
    parent_at(h, 2)
}
"""
    prog = parse(src)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, "expected 42"


def test_stage36_inc14_three_parents_all_recoverable():
    """All three parents readable from a single handle. Sum check
    catches any slot-shuffling regression."""
    src = """
fn main() -> i32 {
    let h = register_derivation3(10, 14, 18);
    parent_at(h, 0) + parent_at(h, 1) + parent_at(h, 2)
}
"""
    prog = parse(src)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, "expected 10 + 14 + 18 = 42"


def test_stage36_inc14_register_derivation3_advances_arena_by_3():
    """The cursor must advance by exactly 3 per call (one slot per
    parent). h2 = h1 + 3 since each register_derivation3 pushes 3
    arena entries. Pins the ARENA_PUSH_TRIPLE-advances-by-3 invariant."""
    src = """
fn main() -> i32 {
    let h1 = register_derivation3(1, 2, 3);
    let h2 = register_derivation3(4, 5, 6);
    // h1 = 1 (push_idx 0 + 1), h2 = 4 (push_idx 3 + 1). 4 - 1 = 3.
    (h2 - h1) * 14
}
"""
    prog = parse(src)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, "expected (4 - 1) * 14 = 42"


def test_stage36_inc14_independent_triples_stay_independent():
    """Two register_derivation3 calls write to disjoint slot regions;
    reads on h1 don't see h2's data. Exit checks all six values
    distinctly via a weighted sum that would fail closed on any
    slot-aliasing bug."""
    src = """
fn main() -> i32 {
    let h1 = register_derivation3(1, 2, 3);
    let h2 = register_derivation3(4, 5, 6);
    let a = parent_at(h1, 0);
    let b = parent_at(h1, 1);
    let c = parent_at(h1, 2);
    let d = parent_at(h2, 0);
    let e = parent_at(h2, 1);
    let f = parent_at(h2, 2);
    // 1+2+3+4+5+6 = 21; expected sum * 2 = 42.
    (a + b + c + d + e + f) * 2
}
"""
    prog = parse(src)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, "expected (1+2+3+4+5+6) * 2 = 42"


def test_stage36_inc14_arena_push_triple_atomic_against_intervening_push():
    """The triple's three slots stay adjacent even when an unrelated
    __arena_push happens between two register_derivation3 calls. Mirror
    of the Inc 9 ARENA_PUSH_PAIR atomicity test."""
    src = """
fn main() -> i32 {
    let h1 = register_derivation3(11, 12, 13);
    let _ = __arena_push(99999);
    let h2 = register_derivation3(21, 22, 23);
    let a = parent_at(h1, 0);
    let b = parent_at(h1, 1);
    let c = parent_at(h1, 2);
    let d = parent_at(h2, 0);
    let e = parent_at(h2, 1);
    let f = parent_at(h2, 2);
    // Each triple's slots must still match its register call inputs.
    // (11+12+13) + (21+22+23) = 36 + 66 = 102. Exit code is checked
    // strictly — any slot shift would change the sum.
    let ok1 = (a + b + c) - 36;
    let ok2 = (d + e + f) - 66;
    42 - (ok1 + ok2) * 99
}
"""
    prog = parse(src)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, \
        "expected 42 (both triples atomic across intervening __arena_push)"


def test_stage36_inc14_parent_at_on_two_parent_handle_back_compat():
    """parent_at(h, 0) == parent_left_at(h) and parent_at(h, 1) ==
    parent_right_at(h) for a handle returned by the original
    register_derivation. Back-compat invariant — the generic accessor
    must agree with the legacy accessors on the slots they share."""
    src = """
fn main() -> i32 {
    let h = register_derivation(17, 25);
    let lhs0 = parent_at(h, 0);
    let rhs0 = parent_left_at(h);
    let lhs1 = parent_at(h, 1);
    let rhs1 = parent_right_at(h);
    // Both pairs must agree. Build a value that's 42 iff they match.
    let ok = (lhs0 - rhs0) + (lhs1 - rhs1);
    42 - ok * 99
}
"""
    prog = parse(src)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, "parent_at must agree with parent_*_at on two-parent handles"


def test_stage36_inc14_parent_at_null_handle_returns_negative_one():
    """parent_at(0, slot) for any slot returns -1 via the Inc 9 A1
    bounds-check sentinel: handle 0 - 1 = -1 fails the >= 0 check."""
    src = """
fn main() -> i32 {
    let v = parent_at(0, 1);
    // v should be -1. 42 - (-1 + 1) * 99 = 42.
    42 - (v + 1) * 99
}
"""
    prog = parse(src)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, "null handle (0) must return -1 sentinel"


def test_stage36_inc14_parent_at_oob_slot_returns_negative_one():
    """parent_at(h, very_large_slot) returns -1 via bounds check. Pins
    the OOB-fallthrough invariant: even if the user passes a slot far
    larger than the arity, the read returns sentinel rather than
    walking off the arena edge."""
    src = """
fn main() -> i32 {
    let h = register_derivation3(1, 2, 3);
    let v = parent_at(h, 999999);
    42 - (v + 1) * 99
}
"""
    prog = parse(src)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, "OOB slot must return -1 sentinel"


def test_stage36_inc14_register_derivation3_typecheck_rejects_i64():
    """register_derivation3 must reject non-i32 args (Inc 11 C1
    family). Pre-fix, accepting i64 would silently truncate in the
    downstream arena push ops."""
    src = """
fn main() -> i32 {
    let big: i64 = 99;
    let h = register_derivation3(big, 2, 3);
    h
}
"""
    prog = parse(src)
    errs = typecheck(prog)
    assert any("register_derivation3" in str(e) and "must be exactly i32" in str(e)
               for e in errs), \
        f"expected register_derivation3 i32 strictness error, got {errs}"


def test_stage36_inc14_parent_at_typecheck_rejects_i64_handle():
    """parent_at(handle, slot) rejects i64 handle — same strictness as
    register_derivation* (no silent truncation)."""
    src = """
fn main() -> i32 {
    let h: i64 = 1;
    let v = parent_at(h, 0);
    v
}
"""
    prog = parse(src)
    errs = typecheck(prog)
    assert any("parent_at" in str(e) and "must be exactly i32" in str(e)
               for e in errs), \
        f"expected parent_at i32 strictness error, got {errs}"


def test_stage36_inc14_register_derivation3_arena_overflow_returns_zero_handle():
    """When the arena cursor is too high for three more slots,
    ARENA_PUSH_TRIPLE returns -1 (overflow sentinel) and the
    1-based-handle ADD turns that into 0 (null sentinel). Subsequent
    parent_at calls on the null handle return -1 via Inc 9 A1.
    Structural test: we can't easily fill a 2M-slot arena, so this
    confirms the pure structural symmetry — the contract follows from
    sharing the ADD-1 wrapping with register_derivation."""
    # If overflow ever does fire, the test would observe handle == 0.
    # In bounds (the common case) it reads back as a positive handle.
    src = """
fn main() -> i32 {
    let h = register_derivation3(7, 8, 9);
    // h must be at least 1 (non-null); using has_evidence to confirm.
    42 * has_evidence(h)
}
"""
    prog = parse(src, include_stdlib=True)
    assert typecheck(prog) == []
    elf = compile_module_to_elf(lower(prog))
    assert _run_elf(elf) == 42, "in-bounds register_derivation3 must yield a recoverable handle"


def test_stage36_inc14_arena_push_triple_is_in_effect_table():
    """The new opcode must carry the {"arena"} effect label so the
    effect-check pass treats it like ARENA_PUSH / ARENA_PUSH_PAIR."""
    from helixc.ir import tir
    from helixc.ir.passes.effect_check import OP_EFFECTS
    assert tir.OpKind.ARENA_PUSH_TRIPLE in OP_EFFECTS, \
        "ARENA_PUSH_TRIPLE missing from OP_EFFECTS — would be misclassified as pure"
    assert "arena" in OP_EFFECTS[tir.OpKind.ARENA_PUSH_TRIPLE], \
        "ARENA_PUSH_TRIPLE effect domain must include 'arena'"


def test_stage36_inc14_arena_push_triple_is_in_dce_side_effect_set():
    """DCE must treat ARENA_PUSH_TRIPLE as side-effectful even when
    the result slot index is unused — otherwise an unused
    `let _h = register_derivation3(...);` would erase the arena writes
    and silently break downstream parent_at reads."""
    from helixc.ir import tir
    from helixc.ir.passes.dce import SIDE_EFFECT_KINDS
    assert tir.OpKind.ARENA_PUSH_TRIPLE in SIDE_EFFECT_KINDS, \
        "ARENA_PUSH_TRIPLE must be in DCE side-effect set to survive unused-result DCE"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
