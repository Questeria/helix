"""Tests for helixc.backend.ptx (PTX emission)."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.tile_ir import lower_to_tile
from helixc.backend.ptx import emit_ptx


def emit(src: str) -> str:
    return emit_ptx(lower_to_tile(lower(parse(src))))


def test_module_header():
    out = emit("@kernel fn k() {}")
    assert ".version" in out
    assert ".target sm_75" in out
    assert ".address_size 64" in out


def test_kernel_directive():
    out = emit("@kernel fn my_kernel() {}")
    assert ".visible .entry my_kernel" in out
    assert "{" in out and "}" in out


def test_kernel_has_register_declarations():
    out = emit("@kernel fn k() {}")
    assert ".reg .pred" in out
    assert ".reg .b32" in out
    assert ".reg .f32" in out


def test_kernel_ret():
    out = emit("@kernel fn k() {}")
    # Every kernel must end with ret;
    assert "ret;" in out


def test_scalar_const_int():
    src = "@kernel fn k() { let x = 42; }"
    out = emit(src)
    assert "mov.b32" in out
    assert "42" in out


def test_scalar_add():
    src = "@kernel fn k() { let x = 1; let y = 2; let z = x + y; }"
    out = emit(src)
    assert "add.s32" in out


def test_scalar_mul():
    src = "@kernel fn k() { let z = 3 * 4; }"
    out = emit(src)
    assert "mul.lo.s32" in out


def test_non_kernel_emits_func():
    src = """
    fn helper() -> i32 { 42 }
    @kernel fn k() {}
    """
    out = emit(src)
    assert ".func" in out
    assert ".visible .entry k" in out


# ============================================================================
# Stage 16 — GPU primitives end-to-end
# ============================================================================
def test_thread_idx_emits_tid_x():
    src = "@kernel fn k() { let i = thread_idx(); }"
    out = emit(src)
    assert "mov.u32" in out
    assert "%tid.x" in out


def test_thread_idx_outside_kernel_traps():
    # Trap-id 96001: thread_idx() outside @kernel.
    src = "fn main() -> i32 { let i = thread_idx(); 0 }"
    try:
        emit(src)
    except (SyntaxError, NotImplementedError) as e:
        assert "96001" in str(e) or "thread_idx" in str(e)
        return
    raise AssertionError("expected trap 96001 for thread_idx() outside kernel")


def test_hbm_tile_param_indexed_load_emits_ld_global_f32():
    src = """
    @kernel fn k(a: tile<f32, [256], HBM>) {
        let x = a[0];
    }
    """
    out = emit(src)
    assert "ld.param.u64" in out
    assert "cvta.to.global.u64" in out
    assert "ld.global.f32" in out


def test_hbm_tile_param_indexed_store_emits_st_global_f32():
    src = """
    @kernel fn k(a: tile<f32, [256], HBM>, b: tile<f32, [256], HBM>) {
        b[0] = a[0];
    }
    """
    out = emit(src)
    assert "ld.global.f32" in out
    assert "st.global.f32" in out


def test_vec_add_kernel_full_ptx():
    # The Stage 16 capstone: vec_add must produce a PTX kernel that:
    # - declares 3 .param .b64 entries
    # - reads %tid.x
    # - emits three ld.global.f32 sequences (a[i] + b[i] + result load for store)
    # - emits one add.f32
    # - emits one st.global.f32 to c
    src = """
    @kernel
    fn vec_add(a: tile<f32, [256], HBM>, b: tile<f32, [256], HBM>, c: tile<f32, [256], HBM>) {
        let i = thread_idx();
        c[i] = a[i] + b[i];
    }
    """
    out = emit(src)
    assert ".visible .entry vec_add" in out
    assert ".param .b64 param_0" in out
    assert ".param .b64 param_1" in out
    assert ".param .b64 param_2" in out
    assert "%tid.x" in out
    # Two HBM loads (a[i], b[i]) plus one HBM store (c[i] = ...).
    assert out.count("ld.global.f32") == 2
    assert out.count("st.global.f32") == 1
    assert "add.f32" in out
    # And the trapping `// TODO:` strings must not appear: every op was handled.
    assert "// TODO:" not in out


def test_per_prefix_register_counters():
    # %r and %f pools must be independent. Earlier shared `next_reg` would
    # produce stale labels like %r3 == %f3.
    src = """
    @kernel fn k(a: tile<f32, [16], HBM>) {
        let i = thread_idx();
        let x = a[i];
    }
    """
    out = emit(src)
    # %r0 reads tid; %f0 receives the ld.global.f32 result.
    assert "%r0" in out
    assert "%f0" in out


def test_thread_idx_y_and_z():
    src = """
    @kernel fn k() {
        let x = thread_idx();
        let y = thread_idx_y();
        let z = thread_idx_z();
    }
    """
    out = emit(src)
    assert "%tid.x" in out
    assert "%tid.y" in out
    assert "%tid.z" in out


def test_block_idx_and_block_dim():
    src = """
    @kernel fn k() {
        let bx = block_idx();
        let by = block_idx_y();
        let bdz = block_dim_z();
    }
    """
    out = emit(src)
    assert "%ctaid.x" in out
    assert "%ctaid.y" in out
    assert "%ntid.z" in out


def test_scalar_sub():
    out = emit("@kernel fn k() { let z = 10 - 3; }")
    assert "sub.s32" in out


def test_scalar_neg():
    out = emit("@kernel fn k() { let x = 5; let y = -x; }")
    assert "neg.s32" in out


def test_scalar_const_float():
    out = emit("@kernel fn k() { let x = 3.14; }")
    # Hex bit pattern of 3.14f rounded.
    assert "mov.f32" in out
    assert "0f" in out  # PTX hex-float prefix


def test_ptx_register_pool_overflow_raises():
    # Audit A3-MEDIUM-1 regression: per-prefix register pool overflow
    # used to silently emit references to undeclared registers (e.g.
    # %r33 when only %r<32> was declared). Now _new_reg raises
    # RuntimeError when the per-prefix counter exceeds _REG_POOL_CAP.
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    em.next_reg_by_prefix["r"] = PtxEmitter._REG_POOL_CAP
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="register pool overflow"):
        em._new_reg("r")


def test_ptx_register_pool_cap_in_kernel_decl():
    # Audit A3-MEDIUM-1: bumped pool from 32 to 256 in declarations.
    out = emit("@kernel fn k() {}")
    assert ".reg .b32   %r<256>;" in out
    assert ".reg .f32   %f<256>;" in out
    assert ".reg .pred  %p<256>;" in out
    assert ".reg .b64   %rd<256>;" in out


def test_hbm_subtract_uses_sub_f32():
    src = """
    @kernel fn k(a: tile<f32, [16], HBM>, b: tile<f32, [16], HBM>) {
        let i = thread_idx();
        b[i] = a[i] - a[i];
    }
    """
    out = emit(src)
    assert "sub.f32" in out


def test_c20_1_isize_usize_treated_as_64_bit_in_ptx():
    """Audit 28.8 cycle 21 C20-1 (HIGH): PTX backend width-keyed tables
    must treat isize/usize as 64-bit, matching typecheck.py canon.

    Pre-fix `_DTYPE_SIZE.get("isize", 4)` returned 4, `_ptx_type_str`
    returned `.b32`, and `_ld_reg_prefix("isize")` returned `"r"` (32-bit
    pool) — silently 32-bit-narrowing isize values in PTX output."""
    from helixc.backend.ptx import PtxEmitter
    from helixc.ir import tir
    # Probe class-level tables directly.
    assert PtxEmitter._DTYPE_SIZE["isize"] == 8
    assert PtxEmitter._DTYPE_SIZE["usize"] == 8
    assert PtxEmitter._DTYPE_SIZE["i64"] == 8
    assert PtxEmitter._DTYPE_PTX_LOAD["isize"] == "s64"
    assert PtxEmitter._DTYPE_PTX_LOAD["usize"] == "u64"
    # _ptx_type_str via instance.
    em = PtxEmitter.__new__(PtxEmitter)  # bare instance (no __init__ side-effects)
    isize_ty = tir.TIRScalar(name="isize")
    usize_ty = tir.TIRScalar(name="usize")
    assert em._ptx_type_str(isize_ty) == ".b64"
    assert em._ptx_type_str(usize_ty) == ".b64"
    # _ld_reg_prefix — isize/usize should pick the 64-bit `rd` pool.
    assert em._ld_reg_prefix("isize") == "rd"
    assert em._ld_reg_prefix("usize") == "rd"
    assert em._ld_reg_prefix("i64") == "rd"
    assert em._ld_reg_prefix("i32") == "r"


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
