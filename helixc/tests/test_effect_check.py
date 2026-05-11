"""Tests for IR-level effect verification."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.passes.effect_check import (
    check_module, verify_module, EffectError, compute_closure,
    own_op_effects, declared_effects, is_pure_decl,
)
from helixc.ir import tir


def lower_only(src: str) -> tir.Module:
    return lower(parse(src))


def test_pure_function_with_no_effects_passes():
    src = """
    @pure fn add(a: i32, b: i32) -> i32 { a + b }
    fn main() -> i32 { add(20, 22) }
    """
    mod = lower_only(src)
    errs = check_module(mod)
    assert errs == [], f"unexpected: {errs}"


def test_pure_function_calling_pure_function_passes():
    src = """
    @pure fn double(x: i32) -> i32 { x + x }
    @pure fn quadruple(x: i32) -> i32 { double(double(x)) }
    fn main() -> i32 { quadruple(10) }
    """
    mod = lower_only(src)
    assert check_module(mod) == []


def test_pure_function_using_print_fails():
    # print_int is a PRINT op which has the "io" effect. A @pure fn can't have it.
    src = """
    @pure fn shout(x: i32) -> i32 {
        print_int(x);
        x
    }
    fn main() -> i32 { shout(5) }
    """
    mod = lower_only(src)
    errs = check_module(mod)
    assert any("@pure" in e and "shout" in e for e in errs), f"got {errs}"


def test_pure_function_calling_impure_transitively_fails():
    # @pure A → B → io. Should be caught: A's closure includes io.
    src = """
    fn impure_helper() -> i32 {
        print_int(7);
        7
    }
    @pure fn caller() -> i32 {
        impure_helper()
    }
    fn main() -> i32 { caller() }
    """
    mod = lower_only(src)
    errs = check_module(mod)
    assert any("@pure" in e and "caller" in e for e in errs), f"got {errs}"


def test_unknown_callee_treated_as_unknown_effect():
    # Build a module whose CALL targets an undeclared function name — the
    # checker should flag the caller as having effect "unknown" (which will
    # only be allowed if explicitly declared).
    mod = tir.Module()
    i32 = tir.TIRScalar("i32")
    blk = tir.Block(id=0)
    v = tir.Value(id=0, ty=i32)
    blk.ops = [
        tir.Op(kind=tir.OpKind.CALL, operands=[], results=[v],
               attrs={"target": "extern_unknown"}),
        tir.Op(kind=tir.OpKind.RETURN, operands=[v], results=[]),
    ]
    fn = tir.FnIR(name="caller", params=[], return_ty=i32, blocks=[blk],
                  attrs={"is_pure": True})
    mod.functions["caller"] = fn
    mod.next_value_id = 1
    mod.next_block_id = 1

    errs = check_module(mod)
    assert any("caller" in e and "unknown" in e for e in errs), f"got {errs}"


def test_verify_module_raises_on_violation():
    src = """
    @pure fn bad() -> i32 {
        print_int(1);
        0
    }
    fn main() -> i32 { bad() }
    """
    mod = lower_only(src)
    try:
        verify_module(mod)
    except EffectError:
        return
    raise AssertionError("expected EffectError")


def test_verify_module_passes_on_clean_module():
    src = """
    @pure fn add(a: i32, b: i32) -> i32 { a + b }
    fn main() -> i32 { add(1, 2) }
    """
    mod = lower_only(src)
    verify_module(mod)  # must not raise


def test_verifier_effects_propagate_to_caller_via_modify():
    # If a function uses modify(h, v, my_verifier) and my_verifier transitively
    # has effect "io", the caller's closure must include "io" — otherwise a
    # @pure caller could sneak I/O via the verifier.
    src = """
    fn shouts() -> i32 {
        print_int(7);
        1
    }
    fn my_verifier(h: i32, v: i32) -> i32 {
        shouts()
    }
    @pure fn caller() -> i32 {
        let h = quote(0);
        modify(h, 1, my_verifier);
        0
    }
    fn main() -> i32 { caller() }
    """
    mod = lower_only(src)
    errs = check_module(mod)
    # caller is @pure; its closure should now include the verifier's effects
    # (which include "unknown" from print_int being unresolved). Earlier this
    # was missed because callees() only inspected CALL ops.
    assert any("@pure" in e and "caller" in e for e in errs), f"got {errs}"


def test_recursive_pure_function_has_empty_closure():
    src = """
    @pure fn fact(n: i32) -> i32 {
        if n <= 1 { 1 } else { n * fact(n - 1) }
    }
    fn main() -> i32 { fact(5) }
    """
    mod = lower_only(src)
    closure = compute_closure(mod)
    assert closure["fact"] == frozenset(), f"got {closure['fact']}"


def test_effect_attribute_arg_captured_by_parser():
    """Bug H: parser used to drop the (io) part of @effect(io), so the
    typechecker couldn't tell which effects a function declared. Confirm
    the parsed AST now records each effect as `effect:<name>`."""
    from helixc.frontend.parser import parse
    src = """
    @effect(io)
    fn print_thing() -> i32 { 42 }
    @effect(io, rng)
    fn both() -> i32 { 1 }
    fn main() -> i32 { 0 }
    """
    p = parse(src)
    by_name = {item.name: item.attrs for item in p.items
               if hasattr(item, "attrs") and hasattr(item, "name")}
    assert "effect:io" in by_name["print_thing"], by_name["print_thing"]
    assert "effect:io" in by_name["both"], by_name["both"]
    assert "effect:rng" in by_name["both"], by_name["both"]


def test_pure_calling_effectful_fn_rejected_by_typecheck():
    """End-to-end: @pure caller calls @effect(io) callee → typecheck error."""
    from helixc.frontend.parser import parse
    from helixc.frontend.typecheck import typecheck
    src = """
    @effect(io)
    fn print_thing() -> i32 { 42 }
    @pure
    fn caller() -> i32 { print_thing() }
    fn main() -> i32 { caller() }
    """
    p = parse(src)
    errs = typecheck(p)
    msgs = [str(e) for e in errs]
    assert any("@pure" in m and "caller" in m and "print_thing" in m
               for m in msgs), f"got {msgs}"


def test_effectful_caller_missing_callee_effect_rejected():
    """@effect(io) caller cannot call an @effect(rng) function unless it
    also declares (rng) — effect inclusion must be verified."""
    from helixc.frontend.parser import parse
    from helixc.frontend.typecheck import typecheck
    src = """
    @effect(rng)
    fn rand_thing() -> i32 { 7 }
    @effect(io)
    fn caller() -> i32 { rand_thing() }
    fn main() -> i32 { 0 }
    """
    p = parse(src)
    errs = typecheck(p)
    msgs = [str(e) for e in errs]
    assert any("rng" in m and "caller" in m for m in msgs), f"got {msgs}"


def test_effect_inclusion_passes_when_caller_declares_superset():
    """An @effect(io, rng) caller may invoke an @effect(io) function."""
    from helixc.frontend.parser import parse
    from helixc.frontend.typecheck import typecheck
    src = """
    @effect(io)
    fn print_thing() -> i32 { 42 }
    @effect(io, rng)
    fn caller() -> i32 { print_thing() }
    fn main() -> i32 { 0 }
    """
    p = parse(src)
    errs = typecheck(p)
    # Errors related to caller's call to print_thing should not fire.
    msgs = [str(e) for e in errs]
    assert not any("caller" in m and "print_thing" in m for m in msgs), \
        f"unexpected: {msgs}"


# --- Stage 19 regression tests ---

def test_stage19_trap_19001_pure_calls_effectful_via_ir():
    """Stage 19 trap-id 19001: an under-declared function whose body has
    a side-effecting op is flagged at the IR level. Reports the trap-id
    in the error string so a downstream gate can grep for it."""
    # Use a direct PRINT op (avoids the typecheck path that already
    # blocks @pure-calls-effectful at the AST level).
    src = """
    @pure
    fn lies() -> i32 {
        print_int(7);
        7
    }
    fn main() -> i32 { lies() }
    """
    mod = lower_only(src)
    errs = check_module(mod)
    assert any("19001" in e and "lies" in e for e in errs), (
        f"expected trap 19001 in errors, got {errs}"
    )


def test_stage19_trap_19002_declared_but_unused_effect():
    """Stage 19 trap-id 19002: an @effect(...) declaration that the
    body's closure never actually exercises is flagged.

    `unused_decl` declares @effect(io) but never invokes a PRINT-style
    or known-io callee — the 19002 trap should surface."""
    src = """
    @effect(io)
    fn unused_decl() -> i32 { 42 }
    fn main() -> i32 { unused_decl() }
    """
    mod = lower_only(src)
    errs = check_module(mod)
    assert any("19002" in e and "unused_decl" in e for e in errs), (
        f"expected trap 19002 (unused declared effect), got {errs}"
    )


def test_stage19_trap_19002_does_not_fire_when_effect_is_used():
    """The 19002 trap MUST NOT fire when the declared effect is exercised
    via the body's closure (here `printer` actually prints)."""
    src = """
    @effect(io)
    fn printer() -> i32 { print_int(7); 7 }
    fn main() -> i32 { printer() }
    """
    mod = lower_only(src)
    errs = check_module(mod)
    # 19002 must NOT appear for `printer` (it really does have io).
    msgs_for_printer = [e for e in errs if "printer" in e]
    assert not any("19002" in e for e in msgs_for_printer), (
        f"19002 falsely fired for printer: {msgs_for_printer}"
    )


def test_stage19_19001_not_emitted_when_pure_is_actually_pure():
    """No trap when @pure does what it says."""
    src = """
    @pure
    fn truly_pure(x: i32) -> i32 { x + 1 }
    fn main() -> i32 { truly_pure(41) }
    """
    mod = lower_only(src)
    errs = check_module(mod)
    msgs_for_truly_pure = [e for e in errs if "truly_pure" in e]
    assert msgs_for_truly_pure == [], (
        f"unexpected error for clean @pure fn: {msgs_for_truly_pure}"
    )


def test_stage19_effect_check_runs_in_x86_64_driver_pipeline():
    """The x86_64 driver's __main__ wires effect_check_module after the
    optimization passes. Smoke check: the import works and the call
    site is present in the source. (Actually invoking subprocess on the
    driver is expensive; we keep this lightweight.)"""
    from helixc.backend import x86_64
    import inspect
    src = inspect.getsource(x86_64)
    assert "effect_check_module" in src, (
        "x86_64.py driver no longer imports effect_check — Stage 19 wiring lost"
    )
    assert "trap 19001" in src, (
        "x86_64.py driver no longer surfaces 19001 — Stage 19 wiring lost"
    )


def test_c22_1_ffi_call_is_a_side_effect():
    """Audit 28.8 cycle 23 C22-1 (HIGH): FFI_CALL must be in
    `OP_EFFECTS` so a `@pure` fn that calls an extern "C" function is
    rejected by the IR effect check. Pre-fix the DCE pass had FFI_CALL
    in SIDE_EFFECT_KINDS but the parallel effect_check pass omitted it
    — silent @pure violation for the entire FFI surface."""
    from helixc.ir import tir
    from helixc.ir.passes.effect_check import OP_EFFECTS
    assert tir.OpKind.FFI_CALL in OP_EFFECTS, (
        "FFI_CALL must be in OP_EFFECTS (C22-1)"
    )
    assert "ffi" in OP_EFFECTS[tir.OpKind.FFI_CALL]


def test_c22_3_arena_ops_are_side_effects():
    """Audit 28.8 cycle 23 C22-3 (HIGH): ARENA_PUSH / ARENA_SET mutate
    global state — must be in OP_EFFECTS. Pre-fix `@pure` could call
    `__arena_push` silently."""
    from helixc.ir import tir
    from helixc.ir.passes.effect_check import OP_EFFECTS
    assert tir.OpKind.ARENA_PUSH in OP_EFFECTS
    assert tir.OpKind.ARENA_SET in OP_EFFECTS
    assert "arena" in OP_EFFECTS[tir.OpKind.ARENA_PUSH]
    assert "arena" in OP_EFFECTS[tir.OpKind.ARENA_SET]


def test_c22_defense_in_depth_op_effects_complete():
    """Audit 28.8 cycle 23 C22-2/4/5 (LOW, defense in depth): gated-
    unreachable side-effect ops (QUOTE, REFLECT_HASH, TILE_INDEX_STORE,
    TRACE_ENTRY, TRACE_EXIT) must still be in OP_EFFECTS so any future
    gate-regression surfaces here, not at a downstream miscompile."""
    from helixc.ir import tir
    from helixc.ir.passes.effect_check import OP_EFFECTS
    expected_present = {
        tir.OpKind.QUOTE,
        tir.OpKind.REFLECT_HASH,
        tir.OpKind.TILE_INDEX_STORE,
        tir.OpKind.TRACE_ENTRY,
        tir.OpKind.TRACE_EXIT,
    }
    for kind in expected_present:
        assert kind in OP_EFFECTS, (
            f"{kind!r} must be in OP_EFFECTS as defense-in-depth (C22-*)"
        )


def test_c22_1_ffi_callee_appears_in_callees_set():
    """Audit 28.8 cycle 23 C22-1: `callees(fn)` must include FFI_CALL
    targets so the call-graph closure propagates the `ffi` effect to
    transitive callers. Pre-fix `callees()` only looked at CALL +
    MODIFY/SPLICE verifier targets."""
    from helixc.ir import tir
    from helixc.ir.passes.effect_check import callees
    span = (0, 0)
    blk = tir.Block(id=0, ops=[
        tir.Op(kind=tir.OpKind.FFI_CALL, operands=[], results=[],
               attrs={"target": "extern_puts"}, span=span),
        tir.Op(kind=tir.OpKind.RETURN, operands=[], results=[],
               attrs={}, span=span),
    ])
    fn = tir.FnIR(name="caller", params=[],
                  return_ty=tir.TIRScalar(name="i32"),
                  blocks=[blk], attrs={})
    out = callees(fn)
    assert "extern_puts" in out, (
        f"FFI_CALL target must appear in callees(); got {out}"
    )


# --- Stage 28.9 cycle 21 regression tests ---

def test_c20_t1_trace_pure_allowed():
    """C20-T1 regression: @trace on @pure fn is allowed per
    trace_pass.py:110-112 documented policy. Before the cycle-21 fix,
    TRACE_ENTRY/TRACE_EXIT ops contributed 'trace' to the closure and
    the @pure violation check (19001) fired."""
    src = "@trace @pure fn f(x: i32) -> i32 { x + 1 }"
    mod = lower_only(src)
    errs = check_module(mod)
    assert errs == [], (
        f"@trace @pure must be allowed (cycle 21 C20-T1); got {errs}"
    )


def test_c20_t1_quote_in_pure_allowed():
    """C20-T1 regression: quote { ... } in @pure fn is allowed —
    reflection returns an AST handle, not a value-effect. The cycle-22
    hardening that added QUOTE/REFLECT_HASH to OP_EFFECTS labeled them
    'reflect'; PURITY_OBSERVER_EFFECTS now exempts them."""
    src = "@pure fn f() -> i64 { quote { 1 + 2 } }"
    mod = lower_only(src)
    errs = check_module(mod)
    assert errs == [], (
        f"quote in @pure must be allowed (cycle 21 C20-T1); got {errs}"
    )


def test_c20_t1_print_in_pure_still_fails():
    """C20-T1 regression-the-other-way: a real @pure violation (PRINT
    via print_int / io call) must still flag 19001 after the fix.
    Otherwise the exemption logic over-corrected."""
    src = """
    @pure fn bad() -> i32 {
        print_int(42);
        0
    }
    """
    mod = lower_only(src)
    errs = check_module(mod)
    assert any("@pure" in e and "bad" in e for e in errs), (
        f"PRINT in @pure must still flag; got {errs}"
    )


def test_c20_t2_deprecated_attr_no_spurious_19002():
    """C20-T2 regression: @deprecated(\"msg\") emits 'deprecated' AND
    'deprecated:msg' attribute keys (parser.py:287-291). Before the fix
    these fell through declared_effects' bare-name fallback and tripped
    trap 19002 (declared unused effect)."""
    src = '@deprecated("old-fn") fn d(x: i32) -> i32 { x + 1 }'
    mod = lower_only(src)
    errs = check_module(mod)
    assert errs == [], (
        f"@deprecated must not be treated as a declared effect; got {errs}"
    )


def test_c20_t2_autotune_attr_no_spurious_19002():
    """C20-T2 regression: @autotune(KEY: [v1, v2]) emits 'autotune' AND
    'autotune:KEY=v1,v2' (parser.py:277-282). Both must be treated as
    META, not as declared effects."""
    src = "@autotune(TILE: [16, 32]) fn a(x: i32) -> i32 { x * 2 }"
    mod = lower_only(src)
    errs = check_module(mod)
    assert errs == [], (
        f"@autotune must not be treated as a declared effect; got {errs}"
    )


def test_c20_t2_since_attr_no_spurious_19002():
    """C20-T2 regression: @since(\"v0.3\") emits 'since' AND 'since:v0.3'."""
    src = '@since("v0.3") fn s(x: i32) -> i32 { x - 1 }'
    mod = lower_only(src)
    errs = check_module(mod)
    assert errs == [], (
        f"@since must not be treated as a declared effect; got {errs}"
    )


def test_c20_t2_trace_attr_alone_no_spurious_violation():
    """C20-T2 regression: @trace alone (without @pure) must not trip
    19001 from missing 'trace' declaration. Trace is observability,
    not effect — same PURITY_OBSERVER_EFFECTS exemption applies."""
    src = "@trace fn t(x: i32) -> i32 { x + 1 }"
    mod = lower_only(src)
    errs = check_module(mod)
    assert errs == [], (
        f"@trace alone must not trip 19001 missing-trace; got {errs}"
    )


def test_c20_t2_combo_attrs_no_spurious_19002():
    """C20-T2 regression: stacking @deprecated + @since + @autotune on
    the same fn (a plausible production scenario) must remain clean."""
    src = '@deprecated("d") @since("v1") @autotune(K: [4]) fn combo(x: i32) -> i32 { x }'
    mod = lower_only(src)
    errs = check_module(mod)
    assert errs == [], (
        f"@deprecated/@since/@autotune combo must remain clean; got {errs}"
    )


def test_c25_cr2_extern_fn_does_not_declare_ffi():
    """C25 audit-R cr2 / C23-4 invariant regression (conf 88): an
    extern fn's OWN FnIR has is_extern=True / extern_abi=... as
    structural tags, NOT as a declared 'ffi' effect. The "ffi" effect
    label (cycle 22 C22-1) is attributed to CALLERS via FFI_CALL
    ops, not to the extern declaration itself. META_ATTRS includes
    is_extern/extern_abi precisely so the 19002 (unused-effect) check
    does not produce a false positive on extern fn declarations.

    Without this regression test, a refactor of META_ATTRS or
    declared_effects that accidentally drops is_extern would silently
    reintroduce the false-positive 19002 on every extern fn — caught
    only at the test level. Cycle 26 closes this coverage gap."""
    # Construct a minimal extern fn FnIR directly. (Going through the
    # parser path would require a stdlib extern "C" decl; building IR
    # is cleaner for the unit-level invariant.)
    i32 = tir.TIRScalar("i32")
    fn = tir.FnIR(
        name="extern_puts",
        params=[],
        return_ty=i32,
        blocks=[],
        attrs={"is_extern": True, "extern_abi": "C"},
    )
    declared = declared_effects(fn)
    assert "ffi" not in declared, (
        f"extern fn declaration must NOT contribute 'ffi' to its own "
        f"declared_effects (the 'ffi' label is for CALLERS via FFI_CALL "
        f"ops, per cycle-22 C22-1 / cycle-24 C23-4). Got "
        f"declared_effects={declared}"
    )
    # Same fn declared as @pure must not pass either (would also be
    # an empty declared set, but the structural invariant is what
    # matters here).
    assert "is_extern" not in declared, (
        f"is_extern is META, must not appear as a declared effect; "
        f"got {declared}"
    )
    assert "extern_abi" not in declared, (
        f"extern_abi is META, must not appear as a declared effect; "
        f"got {declared}"
    )


def test_c20_t2_real_unused_effect_still_flags_19002():
    """C20-T2 regression-the-other-way: a genuinely-unused @effect(io)
    must still flag 19002. Otherwise the exemption logic over-corrected."""
    src = "@effect(io) fn doesnt_io(x: i32) -> i32 { x + 1 }"
    mod = lower_only(src)
    errs = check_module(mod)
    assert any("19002" in e and "doesnt_io" in e for e in errs), (
        f"genuinely unused @effect(io) must still flag 19002; got {errs}"
    )


def main():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
