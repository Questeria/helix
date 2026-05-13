"""Tests for the DCE (dead code elimination) IR pass."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.passes.const_fold import fold_module
from helixc.ir.passes.dce import dce_module
from helixc.ir import tir


def lower_fold_dce(src: str) -> tir.Module:
    mod = lower(parse(src))
    fold_module(mod)
    dce_module(mod)
    return mod


def count_ops(mod: tir.Module, kind: tir.OpKind) -> int:
    return sum(
        1 for fn in mod.functions.values()
        for blk in fn.blocks
        for op in blk.ops
        if op.kind == kind
    )


def total_ops(mod: tir.Module) -> int:
    return sum(
        len(blk.ops) for fn in mod.functions.values() for blk in fn.blocks
    )


def test_dce_removes_dead_constants_after_fold():
    # Before fold: const(1), const(2), add -> 3 ops + return = 4
    # After fold: const(3) returned; the original const(1)/const(2) are dead
    # After DCE: only const(3) + return remain
    mod = lower_fold_dce("fn f() -> i32 { 1 + 2 }")
    n_consts = count_ops(mod, tir.OpKind.CONST_INT)
    assert n_consts == 1, f"expected 1 const after dce, got {n_consts}"


def test_dce_preserves_used_values():
    # Function arg is live; can't be removed
    mod = lower_fold_dce("fn f(x: i32) -> i32 { x + 5 }")
    # add op should remain (x is runtime, not folded)
    adds = count_ops(mod, tir.OpKind.ADD)
    assert adds == 1


def test_dce_preserves_call():
    # CALL ops are side-effecting (might mutate); never removed even
    # if their result is unused
    src = """
    fn helper(x: i32) -> i32 { x }
    fn main() -> i32 {
        helper(42);
        7
    }
    """
    mod = lower_fold_dce(src)
    main = mod.functions["main"]
    calls = sum(1 for blk in main.blocks for op in blk.ops
                if op.kind == tir.OpKind.CALL)
    assert calls == 1


def test_dce_preserves_return():
    # Trivial: even an empty fn must keep its return
    mod = lower_fold_dce("fn f() {}")
    rets = count_ops(mod, tir.OpKind.RETURN)
    assert rets == 1


def test_dce_preserves_alloc_var():
    # let mut x = 0 emits ALLOC_VAR + STORE_VAR; both must stay even though
    # the alloc op has no result (or its result is unused)
    src = """
    fn f() -> i32 {
        let mut x = 7;
        x
    }
    """
    mod = lower_fold_dce(src)
    allocs = count_ops(mod, tir.OpKind.ALLOC_VAR)
    stores = count_ops(mod, tir.OpKind.STORE_VAR)
    loads = count_ops(mod, tir.OpKind.LOAD_VAR)
    assert allocs == 1
    assert stores == 1
    # LOAD_VAR result is used by RETURN; must remain
    assert loads >= 1


def test_dce_preserves_array_stores():
    src = """
    fn f() -> i32 {
        let xs = [1, 2, 3];
        xs[0]
    }
    """
    mod = lower_fold_dce(src)
    # STOREs into the array (initial population) must stay
    stores = count_ops(mod, tir.OpKind.STORE_ELEM)
    # LOAD for xs[0] must stay since it feeds into RETURN
    loads = count_ops(mod, tir.OpKind.LOAD_ELEM)
    assert stores >= 3
    assert loads >= 1


def test_c13_1_dce_preserves_trace_exit_operand():
    # Audit 28.8 cycle 13 C13-1 (HIGH): a @trace fn returning Unit
    # has a synthesized `const_int(0)` consumed only by TRACE_EXIT.
    # Pre-fix, DCE dropped the const but left TRACE_EXIT referencing
    # its dangling value-id; -O2 codegen then KeyError'd in the
    # x86_64 backend at slot lookup. After fix, TRACE_EXIT is in
    # SIDE_EFFECT_KINDS so its operands are seeded as live.
    src = "@trace\nfn foo() {\n    let x: i32 = 5;\n}\n"
    mod = lower_fold_dce(src)
    foo_fn = mod.functions["foo"]
    exits = [op for blk in foo_fn.blocks for op in blk.ops
             if op.kind == tir.OpKind.TRACE_EXIT]
    assert len(exits) == 1, f"expected 1 TRACE_EXIT, got {len(exits)}"
    alive_ids = {r.id for blk in foo_fn.blocks for op in blk.ops
                 for r in op.results} | {p.id for p in foo_fn.params}
    for op in exits:
        for operand in op.operands:
            assert operand.id in alive_ids, (
                f"TRACE_EXIT operand id={operand.id} was DCE'd "
                f"(producer dropped); backend will KeyError at -O2")


def test_c13_1_dce_preserves_trace_entry_in_kept_set():
    # Belt-and-suspenders sibling: TRACE_ENTRY should also survive
    # DCE even though it has no result (it never had operands, so
    # this is mostly future-proofing if a runtime helper-handle is
    # added to its operand list later).
    src = "@trace\nfn foo() {\n    let x: i32 = 5;\n}\n"
    mod = lower_fold_dce(src)
    foo_fn = mod.functions["foo"]
    entries = [op for blk in foo_fn.blocks for op in blk.ops
               if op.kind == tir.OpKind.TRACE_ENTRY]
    assert len(entries) == 1, f"expected 1 TRACE_ENTRY, got {len(entries)}"


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
