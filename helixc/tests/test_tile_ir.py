"""Tests for helixc.ir.tile_ir (Tile IR + Tensor IR lowering)."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
import pytest

from helixc.ir.tile_ir import lower_to_tile, TileOpKind, MemSpace


def lower_chain(src: str):
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    return tile_mod


def test_empty_function():
    mod = lower_chain("fn nothing() {}")
    assert "nothing" in mod.functions


def test_arith_passes_through():
    mod = lower_chain("fn add(a: i32, b: i32) -> i32 { a + b }")
    fn = mod.functions["add"]
    kinds = [op.kind for op in fn.entry.ops]
    assert TileOpKind.SCALAR_ADD in kinds
    assert TileOpKind.RETURN in kinds


def test_cmp_carries_attr():
    mod = lower_chain("fn f() -> bool { 1 < 2 }")
    fn = mod.functions["f"]
    cmp_ops = [op for op in fn.entry.ops if op.kind == TileOpKind.SCALAR_CMP]
    assert len(cmp_ops) == 1
    assert cmp_ops[0].attrs.get("cmp") == "cmp.lt"


def test_if_lowered_to_cfg_in_tile_ir():
    with pytest.raises(NotImplementedError, match="cond_br|br"):
        lower_chain("fn f(b: bool) -> i32 { if b { 1 } else { 2 } }")


def test_tile_ir_rejects_unmapped_scalar_div():
    with pytest.raises(NotImplementedError, match="elem.div"):
        lower_chain("fn f() -> i32 { 4 / 2 }")


def test_call_lowered():
    src = """
    fn double(x: i32) -> i32 { x + x }
    fn main() -> i32 { double(7) }
    """
    mod = lower_chain(src)
    main = mod.functions["main"]
    calls = [op for op in main.entry.ops if op.kind == TileOpKind.CALL]
    assert len(calls) >= 1


def test_value_id_consistency():
    mod = lower_chain("fn f() -> i32 { 1 + 2 + 3 }")
    fn = mod.functions["f"]
    ids = []
    for op in fn.entry.ops:
        for r in op.results:
            ids.append(r.id)
    assert len(ids) == len(set(ids)), "tile values must have unique ids"


def test_function_attrs_carried():
    mod = lower_chain("@kernel fn k() {}")
    fn = mod.functions["k"]
    assert fn.attrs.get("kernel") is True


# ============================================================================
# Stage 117-119 (v2.0 Phase B.3) substrate tests — tile-IR adjoint table
# ============================================================================
def test_stage117_tile_matmul_has_adjoint():
    """Stage 117 — TILE_MATMUL has a declared adjoint sequence (3-wmma
    reverse pattern: dA = dD @ Bt; dB = At @ dD; dC = dD)."""
    from helixc.ir.tile_ir import (
        TileOpKind, TILE_OP_ADJOINTS, has_adjoint, adjoint_outputs,
    )
    assert has_adjoint(TileOpKind.TILE_MATMUL)
    outs = adjoint_outputs(TileOpKind.TILE_MATMUL)
    assert outs == ("dA", "dB", "dC")
    entry = TILE_OP_ADJOINTS[TileOpKind.TILE_MATMUL]
    # 2 transposes + 2 matmuls (dC = dD is alias/copy, not a separate op).
    assert len(entry.ops) == 4
    # Verify the wmma-mirror sequence:
    op_kinds = [k for (k, _comment) in entry.ops]
    assert op_kinds.count(TileOpKind.TILE_TRANSPOSE) == 2
    assert op_kinds.count(TileOpKind.TILE_MATMUL) == 2


def test_stage118_tile_add_adjoint_is_identity():
    """Stage 118 — TILE_ADD adjoint is identity: dx = dz, dy = dz.
    No new ops emitted; gradient flows through unchanged. The empty
    `ops` is disambiguated from "pending impl" via dispatch=identity."""
    from helixc.ir.tile_ir import (
        TileOpKind, TILE_OP_ADJOINTS, has_adjoint, adjoint_outputs,
    )
    assert has_adjoint(TileOpKind.TILE_ADD)
    outs = adjoint_outputs(TileOpKind.TILE_ADD)
    assert outs == ("dx", "dy")
    entry = TILE_OP_ADJOINTS[TileOpKind.TILE_ADD]
    assert entry.ops == ()
    assert entry.dispatch == "identity"


def test_stage118_tile_sub_adjoint_negates_dy():
    """Stage 118 audit-fix — TILE_SUB adjoint must include SCALAR_NEG
    so dy = -dz. Without it, the gradient sign on the subtrahend is
    silently wrong."""
    from helixc.ir.tile_ir import (
        TileOpKind, TILE_OP_ADJOINTS, has_adjoint, adjoint_outputs,
    )
    assert has_adjoint(TileOpKind.TILE_SUB)
    assert adjoint_outputs(TileOpKind.TILE_SUB) == ("dx", "dy")
    entry = TILE_OP_ADJOINTS[TileOpKind.TILE_SUB]
    op_kinds = [k for (k, _comment) in entry.ops]
    assert TileOpKind.SCALAR_NEG in op_kinds
    assert len(entry.ops) == 1
    assert entry.dispatch == "explicit"


def test_stage118_tile_mul_adjoint_two_muls():
    """Stage 118 audit-fix — TILE_MUL adjoint must emit exactly two
    TILE_MULs (dx = dz*y, dy = dz*x). Swapping either to TILE_ADD or
    omitting one would silently corrupt elementwise-product grads."""
    from helixc.ir.tile_ir import (
        TileOpKind, TILE_OP_ADJOINTS,
    )
    entry = TILE_OP_ADJOINTS[TileOpKind.TILE_MUL]
    op_kinds = [k for (k, _comment) in entry.ops]
    assert op_kinds == [TileOpKind.TILE_MUL, TileOpKind.TILE_MUL]


def test_stage119_tile_reduce_has_adjoint():
    """Stage 119 — TILE_REDUCE has a declared adjoint (broadcast back
    along reduced axis for sum; scatter for max/min). The empty `ops`
    is disambiguated from "identity" via dispatch="reduce_kind", so a
    Stage 120 consumer cannot silently treat it as TILE_ADD-style."""
    from helixc.ir.tile_ir import (
        TileOpKind, TILE_OP_ADJOINTS, has_adjoint, adjoint_outputs,
    )
    assert has_adjoint(TileOpKind.TILE_REDUCE)
    assert adjoint_outputs(TileOpKind.TILE_REDUCE) == ("dx",)
    entry = TILE_OP_ADJOINTS[TileOpKind.TILE_REDUCE]
    assert entry.ops == ()
    assert entry.dispatch == "reduce_kind"


def test_stage117_tile_transpose_self_inverse():
    """Stage 117 — transpose is its own inverse for the gradient:
    dx = transpose(dz). Single TILE_TRANSPOSE in adjoint sequence."""
    from helixc.ir.tile_ir import (
        TileOpKind, TILE_OP_ADJOINTS, has_adjoint,
    )
    assert has_adjoint(TileOpKind.TILE_TRANSPOSE)
    entry = TILE_OP_ADJOINTS[TileOpKind.TILE_TRANSPOSE]
    assert len(entry.ops) == 1
    assert entry.ops[0][0] == TileOpKind.TILE_TRANSPOSE


def test_stage117_tile_reshape_has_adjoint():
    """Stage 117 audit-fix — TILE_RESHAPE is differentiable; gradient
    reshapes back. Missing this entry would have silently forced
    Stage 120 into a host-side autograd fallback for any MLP that
    flattens between layers."""
    from helixc.ir.tile_ir import (
        TileOpKind, TILE_OP_ADJOINTS, has_adjoint, adjoint_outputs,
    )
    assert has_adjoint(TileOpKind.TILE_RESHAPE)
    assert adjoint_outputs(TileOpKind.TILE_RESHAPE) == ("dx",)
    entry = TILE_OP_ADJOINTS[TileOpKind.TILE_RESHAPE]
    assert len(entry.ops) == 1
    assert entry.ops[0][0] == TileOpKind.TILE_RESHAPE


def test_stage117_ops_without_adjoint_return_empty():
    """Stage 117-119 — querying an op without a declared adjoint
    returns False / empty tuple. THREAD_IDX and barrier ops have no
    gradient sense — they are not differentiable."""
    from helixc.ir.tile_ir import (
        TileOpKind, has_adjoint, adjoint_outputs,
    )
    assert not has_adjoint(TileOpKind.THREAD_IDX)
    assert not has_adjoint(TileOpKind.BARRIER_WAIT)
    assert adjoint_outputs(TileOpKind.THREAD_IDX) == ()


def test_adjoint_helpers_reject_non_tileopkind():
    """Stage 117-119 audit-fix — has_adjoint / adjoint_outputs must
    raise TypeError on non-TileOpKind input. Silent membership tests
    would otherwise swallow typos and cross-IR `tir.OpKind` confusion
    and report "not differentiable" instead of failing loudly."""
    import pytest
    from helixc.ir.tile_ir import has_adjoint, adjoint_outputs
    for bad in ("tile.matmul", 42, None, object()):
        with pytest.raises(TypeError):
            has_adjoint(bad)
        with pytest.raises(TypeError):
            adjoint_outputs(bad)


def test_adjoint_table_covers_all_tile_op_kinds():
    """Stage 117-119 audit-fix — every TileOpKind must declare its
    diff/non-diff status (in TILE_OP_ADJOINTS or in
    TILE_OP_NON_DIFFERENTIABLE). Adding a new kind without doing so
    trips this test — silent non-differentiability is not allowed."""
    from helixc.ir.tile_ir import (
        TileOpKind, TILE_OP_ADJOINTS, TILE_OP_NON_DIFFERENTIABLE,
    )
    diff = set(TILE_OP_ADJOINTS.keys())
    non_diff = set(TILE_OP_NON_DIFFERENTIABLE)
    # Partitioning invariant: every kind is in exactly one set.
    assert diff.isdisjoint(non_diff), (
        f"TileOpKind(s) in both sets: {diff & non_diff!r}"
    )
    missing = set(TileOpKind) - diff - non_diff
    assert not missing, (
        f"TileOpKind(s) missing diff/non-diff declaration: {missing!r}"
    )


def test_adjoint_record_is_immutable():
    """Stage 117-119 audit-fix — AdjointRecord is frozen and the
    canonical table is wrapped in MappingProxyType. A consumer that
    tries to mutate either should fail loudly, not silently corrupt
    the canonical table for the rest of the process."""
    import pytest
    from helixc.ir.tile_ir import (
        TileOpKind, TILE_OP_ADJOINTS, AdjointRecord,
    )
    entry = TILE_OP_ADJOINTS[TileOpKind.TILE_ADD]
    # Frozen dataclass: cannot reassign fields.
    with pytest.raises((AttributeError, TypeError)):
        entry.ops = ((TileOpKind.SCALAR_NEG, "corrupted"),)
    # MappingProxy: cannot rebind canonical entries.
    with pytest.raises(TypeError):
        TILE_OP_ADJOINTS[TileOpKind.TILE_ADD] = AdjointRecord(
            inputs=(), outputs=(), ops=()
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
