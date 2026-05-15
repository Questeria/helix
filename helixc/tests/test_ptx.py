"""Tests for helixc.backend.ptx (PTX emission)."""

from __future__ import annotations
import os, sys
import subprocess
import tempfile
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir import tir, tile_ir as ti
from helixc.ir.tile_ir import lower_to_tile
from helixc.backend.ptx import emit_ptx


def emit(src: str) -> str:
    return emit_ptx(lower_to_tile(lower(parse(src))))


def run_ptx_cli(src: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    fd, path = tempfile.mkstemp(suffix=".hx", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(src)
        proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return subprocess.run(
            [sys.executable, "-m", "helixc.backend.ptx", path, *extra_args],
            cwd=proj_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_c118_direct_ptx_cli_aborts_on_type_errors():
    proc = run_ptx_cli("@kernel fn k() { let mut b: bool = true; b += false; }\n")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "operator '+' does not support operand type bool" in proc.stderr
    assert "add.s32" not in proc.stdout


def test_c118_hbm_tile_index_missing_ptx_register_fails_closed():
    a = ti.TileValue(0, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(256),), "HBM"
    ), name_hint="a")
    missing_index = ti.TileValue(1, tir.TIRScalar("i32"))
    out = ti.TileValue(2, tir.TIRScalar("f32"))
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.TILE_INDEX_LOAD_HBM, [missing_index], [out],
                  attrs={"name": "a", "dtype": "f32"})
    ])
    fn = ti.TileFn("k", [a], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="missing PTX register for HBM tile index"):
        emit_ptx(mod)


def test_c119_hbm_tile_missing_param_map_entry_fails_closed():
    idx = ti.TileValue(0, tir.TIRScalar("i32"))
    out = ti.TileValue(1, tir.TIRScalar("f32"))
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [idx], attrs={"value": 0}),
        ti.TileOp(ti.TileOpKind.TILE_INDEX_LOAD_HBM, [idx], [out],
                  attrs={"name": "missing", "dtype": "f32"}),
    ])
    fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="not in PTX param map"):
        emit_ptx(mod)


def test_c119_hbm_tile_index_rejects_address_register_class():
    from helixc.backend.ptx import PtxEmitter
    idx = ti.TileValue(0, tir.TIRScalar("i32"))
    out = ti.TileValue(1, tir.TIRScalar("f32"))
    op = ti.TileOp(ti.TileOpKind.TILE_INDEX_LOAD_HBM, [idx], [out],
                   attrs={"name": "a", "dtype": "f32"})
    em = PtxEmitter()
    em.hbm_param_map = {"a": (0, "f32")}
    em.reg_map = {idx.id: "%rd0"}
    with pytest.raises(RuntimeError, match="expected %r register class"):
        em.emit_op(op)


def test_c119_hbm_store_value_must_match_tile_dtype():
    idx = ti.TileValue(0, tir.TIRScalar("i32"))
    bad_value = ti.TileValue(1, tir.TIRScalar("bool"))
    param = ti.TileValue(10, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "HBM"
    ), name_hint="a")
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [idx], attrs={"value": 0}),
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [bad_value], attrs={"value": 1}),
        ti.TileOp(ti.TileOpKind.TILE_INDEX_STORE_HBM, [idx, bad_value], [],
                  attrs={"name": "a", "dtype": "f32"}),
    ])
    fn = ti.TileFn("k", [param], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="unsupported PTX HBM tile store value type bool"):
        emit_ptx(mod)


def test_c119_hbm_op_dtype_must_match_param_map_dtype():
    idx = ti.TileValue(0, tir.TIRScalar("i32"))
    out = ti.TileValue(1, tir.TIRScalar("f32"))
    param = ti.TileValue(10, tir.TIRTileTy(
        tir.TIRScalar("i32"), (tir.DimConst(4),), "HBM"
    ), name_hint="a")
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [idx], attrs={"value": 0}),
        ti.TileOp(ti.TileOpKind.TILE_INDEX_LOAD_HBM, [idx], [out],
                  attrs={"name": "a", "dtype": "f32"}),
    ])
    fn = ti.TileFn("k", [param], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        emit_ptx(mod)


def test_c119_hbm_param_shape_validated_in_direct_ptx():
    param = ti.TileValue(10, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4), tir.DimConst(4)), "HBM"
    ), name_hint="a")
    fn = ti.TileFn("k", [param], tir.TIRUnit(), [ti.TileBlock(0)],
                   attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="HBM tile parameters must be 1D"):
        emit_ptx(mod)


def test_c119_hbm_ops_require_dtype_attr():
    idx = ti.TileValue(0, tir.TIRScalar("i32"))
    out = ti.TileValue(1, tir.TIRScalar("f32"))
    param = ti.TileValue(10, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "HBM"
    ), name_hint="a")
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [idx], attrs={"value": 0}),
        ti.TileOp(ti.TileOpKind.TILE_INDEX_LOAD_HBM, [idx], [out],
                  attrs={"name": "a"}),
    ])
    fn = ti.TileFn("k", [param], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="missing PTX HBM tile dtype attr"):
        emit_ptx(mod)


def test_c119_thread_idx_requires_valid_attrs_in_direct_ptx():
    out = ti.TileValue(0, tir.TIRScalar("i32"))

    def mod_for(attrs):
        block = ti.TileBlock(0, ops=[
            ti.TileOp(ti.TileOpKind.THREAD_IDX, [], [out], attrs=attrs)
        ])
        fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
        return ti.TileModule(functions={"k": fn})

    with pytest.raises(RuntimeError, match="requires explicit dim and sreg"):
        emit_ptx(mod_for({}))
    with pytest.raises(RuntimeError, match="unsupported PTX THREAD_IDX dim"):
        emit_ptx(mod_for({"dim": "w", "sreg": "tid"}))
    with pytest.raises(RuntimeError, match="unsupported PTX THREAD_IDX sreg"):
        emit_ptx(mod_for({"dim": "x", "sreg": "foo"}))


def test_c119_thread_idx_requires_valid_op_shape_in_direct_ptx():
    good = ti.TileValue(0, tir.TIRScalar("i32"))
    bad = ti.TileValue(1, tir.TIRScalar("f32"))
    attrs = {"dim": "x", "sreg": "tid"}

    def mod_for(op):
        block = ti.TileBlock(0, ops=[op])
        fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
        return ti.TileModule(functions={"k": fn})

    with pytest.raises(RuntimeError, match="expects exactly 0 operand"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.THREAD_IDX, [good], [good],
                                   attrs=attrs)))
    with pytest.raises(RuntimeError, match="expects exactly 1 result"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.THREAD_IDX, [], [],
                                   attrs=attrs)))
    with pytest.raises(RuntimeError, match="unsupported PTX THREAD_IDX result type f32"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.THREAD_IDX, [], [bad],
                                   attrs=attrs)))


def test_c119_scalar_constants_require_value_attr_in_direct_ptx():
    int_out = ti.TileValue(0, tir.TIRScalar("i32"))
    float_out = ti.TileValue(1, tir.TIRScalar("f32"))

    def mod_for(op):
        block = ti.TileBlock(0, ops=[op])
        fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
        return ti.TileModule(functions={"k": fn})

    with pytest.raises(RuntimeError, match="SCALAR_CONST_INT requires value attr"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [int_out])))
    with pytest.raises(RuntimeError, match="SCALAR_CONST_FLOAT requires value attr"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CONST_FLOAT, [], [float_out])))


def test_c119_scalar_constant_values_are_not_coerced_in_direct_ptx():
    int_out = ti.TileValue(0, tir.TIRScalar("i32"))
    bool_out = ti.TileValue(1, tir.TIRScalar("bool"))
    float_out = ti.TileValue(2, tir.TIRScalar("f32"))

    def mod_for(op):
        block = ti.TileBlock(0, ops=[op])
        fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
        return ti.TileModule(functions={"k": fn})

    with pytest.raises(RuntimeError, match="i32 value must be an int"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [int_out],
                                   attrs={"value": 1.9})))
    with pytest.raises(RuntimeError, match="i32 value must be an int"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [int_out],
                                   attrs={"value": "7"})))
    with pytest.raises(RuntimeError, match="bool value must be true/false or 0/1"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [bool_out],
                                   attrs={"value": 2})))
    with pytest.raises(RuntimeError, match="value must be a float"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CONST_FLOAT, [], [float_out],
                                   attrs={"value": "1.25"})))


def test_c119_i32_scalar_constants_require_i32_range():
    int_out = ti.TileValue(0, tir.TIRScalar("i32"))
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [int_out],
                  attrs={"value": 2 ** 40})
    ])
    fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="i32 value out of range"):
        emit_ptx(mod)


def test_c119_direct_ptx_rejects_non_unit_kernel_returns():
    fn = ti.TileFn("k", [], tir.TIRScalar("i32"), [ti.TileBlock(0)],
                   attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="non-unit returns"):
        emit_ptx(mod)


def test_c119_direct_ptx_rejects_return_value_ops():
    value = ti.TileValue(0, tir.TIRScalar("i32"))
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [value], attrs={"value": 1}),
        ti.TileOp(ti.TileOpKind.RETURN, [value], []),
    ])
    fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="cannot return values"):
        emit_ptx(mod)


def test_c119_scalar_compare_requires_valid_cmp_attr_in_direct_ptx():
    a = ti.TileValue(0, tir.TIRScalar("i32"))
    b = ti.TileValue(1, tir.TIRScalar("i32"))
    out = ti.TileValue(2, tir.TIRScalar("bool"))

    def mod_for(attrs):
        block = ti.TileBlock(0, ops=[
            ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [a], attrs={"value": 1}),
            ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [b], attrs={"value": 2}),
            ti.TileOp(ti.TileOpKind.SCALAR_CMP, [a, b], [out], attrs=attrs),
        ])
        fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
        return ti.TileModule(functions={"k": fn})

    with pytest.raises(RuntimeError, match="SCALAR_CMP requires cmp attr"):
        emit_ptx(mod_for({}))
    with pytest.raises(RuntimeError, match="unsupported PTX scalar compare op"):
        emit_ptx(mod_for({"cmp": "cmp.nope"}))


def test_c119_ptx_ops_require_exact_operand_counts():
    a = ti.TileValue(0, tir.TIRScalar("i32"))
    b = ti.TileValue(1, tir.TIRScalar("i32"))
    c = ti.TileValue(2, tir.TIRScalar("i32"))
    out = ti.TileValue(3, tir.TIRScalar("i32"))
    param = ti.TileValue(10, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "HBM"
    ), name_hint="a")
    fval = ti.TileValue(11, tir.TIRScalar("f32"))

    def mod_for(op, params=None):
        block = ti.TileBlock(0, ops=[op])
        fn = ti.TileFn("k", params or [], tir.TIRUnit(), [block],
                       attrs={"kernel": True})
        return ti.TileModule(functions={"k": fn})

    with pytest.raises(RuntimeError, match="SCALAR_ADD expects exactly 2 operand"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_ADD, [a, b, c], [out])))
    with pytest.raises(RuntimeError, match="SCALAR_CMP expects exactly 2 operand"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CMP, [a, b, c], [out],
                                   attrs={"cmp": "cmp.lt"})))
    with pytest.raises(RuntimeError, match="TILE_INDEX_LOAD_HBM expects exactly 1 operand"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.TILE_INDEX_LOAD_HBM, [a, b], [fval],
                                   attrs={"name": "a", "dtype": "f32"}), [param]))
    with pytest.raises(RuntimeError, match="TILE_INDEX_STORE_HBM expects exactly 2 operand"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.TILE_INDEX_STORE_HBM, [a, fval, b], [],
                                   attrs={"name": "a", "dtype": "f32"}), [param]))


def test_c119_scalar_arithmetic_result_type_must_match_operands():
    a = ti.TileValue(0, tir.TIRScalar("i32"))
    b = ti.TileValue(1, tir.TIRScalar("i32"))
    bad_out = ti.TileValue(2, tir.TIRScalar("f32"))
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [a], attrs={"value": 1}),
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [b], attrs={"value": 2}),
        ti.TileOp(ti.TileOpKind.SCALAR_ADD, [a, b], [bad_out]),
    ])
    fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="unsupported PTX scalar add result type f32"):
        emit_ptx(mod)


def test_c119_direct_ptx_cli_rejects_kernel_helper_calls():
    src = """
    fn helper(x: i32) -> i32 { x + 1 }
    @kernel fn k() { let y = helper(41); }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "unsupported PTX op call" in proc.stderr
    assert "// TODO:" not in proc.stdout


def test_c119_direct_ptx_cli_rejects_scalar_kernel_params():
    proc = run_ptx_cli("@kernel fn k(x: i32) { let z = x + 2; }\n")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "PTX kernel parameter is not supported yet" in proc.stderr
    assert "add.s32" not in proc.stdout


def test_c119_direct_ptx_cli_rejects_modules_without_kernels():
    proc = run_ptx_cli("fn helper(x: i32) -> i32 { x + 1 }\n")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "PTX emission requires at least one @kernel function" in proc.stderr
    assert ".func" not in proc.stdout


def test_stage35_direct_ptx_cli_rejects_oversized_autotune():
    src = """
    @kernel
    @autotune(A: [1, 2, 3, 4, 5], B: [10, 20, 30, 40, 50])
    fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "trap 27001" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_stage35_direct_ptx_cli_ignores_host_helper_with_unsupported_tile_op():
    src = """
    fn host_helper(x: i32) -> i32 { x / 2 }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "elem.div" not in proc.stderr


def test_stage35_direct_ptx_cli_rejects_unwind_attr():
    proc = run_ptx_cli("@unwind @kernel fn k() { let i = thread_idx(); }\n")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "unwind" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_stage35_direct_ptx_cli_folds_kernel_before_tile_lowering():
    proc = run_ptx_cli("@kernel fn k() { let z = 4 / 2; }\n")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "elem.div" not in proc.stderr


def test_stage35_direct_ptx_cli_flattens_module_kernel():
    src = """
    mod m {
        @kernel fn k() { let i = thread_idx(); }
    }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry m__k" in proc.stdout


def test_stage35_direct_ptx_cli_rejects_duplicate_autotune_key():
    src = """
    @kernel
    @autotune(A: [1], A: [2])
    fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "duplicate parameter" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_stage35_direct_ptx_cli_strict_rejects_effect_violation():
    src = """
    @pure fn host() -> i32 {
        print_int(1);
        0
    }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src, "--strict", "--no-stdlib")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "--strict aborts" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_stage35_direct_ptx_cli_strict_rejects_totality_failure():
    src = """
    fn spin(n: i32) -> i32 { spin(n) }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src, "--strict")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "totality" in proc.stderr
    assert "--strict aborts" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_stage35_direct_ptx_cli_includes_stdlib_by_default():
    src = """
    fn host(x: f32) -> f32 { __relu(x) }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "unbound name '__relu'" not in proc.stderr


def test_stage35_direct_ptx_cli_strict_allows_clean_default_stdlib_kernel():
    proc = run_ptx_cli("@kernel fn k() { let i = thread_idx(); }\n", "--strict")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "effect-check warning" not in proc.stderr
    assert "vec_push" not in proc.stderr


def test_stage35_direct_ptx_cli_accepts_stdlib_compat_flag():
    src = """
    fn host(x: f32) -> f32 { __relu(x) }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src, "--stdlib")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "unknown flag --stdlib" not in proc.stderr


def test_stage35_direct_ptx_cli_reports_missing_file_without_traceback():
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    missing = os.path.join(proj_root, "__definitely_missing_stage35__.hx")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.ptx", missing],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "cannot read" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_direct_ptx_cli_bad_invocation_returns_two():
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.ptx", "--bogus"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "unknown flag --bogus" in proc.stderr

    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.ptx", "a.hx", "b.hx"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "expected at most one input path" in proc.stderr


def test_stage35_direct_ptx_cli_reports_parse_error_without_traceback():
    proc = run_ptx_cli("@kernel fn k( { let i = thread_idx(); }\n")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "PARSE ERROR" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_c119_direct_ptx_cli_rejects_unsupported_hbm_float_dtype():
    src = """
    @kernel fn k(a: tile<f16, [256], HBM>) {
        let x = a[0];
        let y = x < 1.0_f16;
    }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "@kernel HBM tile parameter dtype f16 is not supported" in proc.stderr
    assert "ld.global.f16" not in proc.stdout


def test_c119_direct_ptx_cli_rejects_unused_unsupported_hbm_dtype():
    proc = run_ptx_cli("@kernel fn k(a: tile<f16, [16], HBM>) { }\n")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "@kernel HBM tile parameter dtype f16 is not supported" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_c119_direct_ptx_cli_accepts_kernel_index_builtin():
    proc = run_ptx_cli("@kernel fn k() { let i = thread_idx(); }\n")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "%tid.x" in proc.stdout


def test_c119_direct_ptx_cli_rejects_extern_only_kernels():
    proc = run_ptx_cli('@kernel extern "C" fn k(a: tile<f32, [16], HBM>);\n')
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "PTX emission requires at least one @kernel function" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_c119_emit_ptx_rejects_unsupported_kernel_ops():
    with pytest.raises(NotImplementedError, match="elem.div"):
        emit("@kernel fn k() { let z = 4 / 2; }")
    with pytest.raises(NotImplementedError, match="bit.not"):
        emit("@kernel fn k() { let z = ~1; }")
    with pytest.raises(RuntimeError, match="unsupported PTX float constant type f64"):
        emit("@kernel fn k() { let z = 1.0_f64; }")


def test_c119_ptx_scalar_ops_require_mapped_operands():
    a = ti.TileValue(100, tir.TIRScalar("i32"))
    b = ti.TileValue(101, tir.TIRScalar("i32"))
    out = ti.TileValue(102, tir.TIRScalar("i32"))
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_ADD, [a, b], [out])
    ])
    fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="missing PTX register for scalar add lhs"):
        emit_ptx(mod)


def test_c119_ptx_scalar_ops_reject_address_register_operands():
    from helixc.backend.ptx import PtxEmitter
    a = ti.TileValue(100, tir.TIRScalar("i32"))
    b = ti.TileValue(101, tir.TIRScalar("i32"))
    out = ti.TileValue(102, tir.TIRScalar("i32"))
    op = ti.TileOp(ti.TileOpKind.SCALAR_ADD, [a, b], [out])
    em = PtxEmitter()
    em.reg_map = {a.id: "%rd0", b.id: "%rd1"}
    with pytest.raises(RuntimeError, match="expected %r register class"):
        em.emit_op(op)


def test_c119_ptx_float_compare_uses_f32_setp():
    src = """
    @kernel fn k(a: tile<f32, [256], HBM>) {
        let x = a[0];
        let y = x < 1.0_f32;
    }
    """
    out = emit(src)
    assert "setp.lt.f32" in out
    assert "setp.lt.s32" not in out


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


def test_non_kernel_functions_are_not_stubbed():
    src = """
    fn helper() -> i32 { 42 }
    @kernel fn k() {}
    """
    out = emit(src)
    assert ".func" not in out
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
