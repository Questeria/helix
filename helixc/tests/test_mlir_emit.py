"""Tests for helixc.ir.mlir.emit — v3.0 Phase E, Stage 212 chunk A:
the Helix-IR -> MLIR type bridge.

`emit.py` opens Stage 212 — the parallel MLIR translation path. Chunk A
is `render_mlir_type`: every Helix IR type (`tir.TIRType`) rendered as
MLIR type syntax — scalars to `i32` / `f32` / `i1`, tensors to
`tensor<...>`, tiles to `vector<...>`, tuples to `tuple<...>`, unit to
`none`. The translator FAILS CLOSED (`MLIRTranslationError`) on any
type it cannot faithfully render rather than emit a guessed type.

These tests pin: the scalar-dtype mapping; the fail-closed behaviour on
`char` / the quantized dtypes / unknown types / non-static tile dims;
shape-dimension rendering (static and dynamic); tensor / tile / tuple /
unit rendering; the two module-load coverage guards and their
non-vacuity; and — the mock-path rule — that the module never
`import mlir`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from helixc.ir import tile_ir, tir
from helixc.ir.mlir import emit
from helixc.ir.mlir.emit import (
    MLIRTranslationError, emit_mlir_module, render_dim, render_mlir_type,
)
from helixc.ir.mlir.validate import mock_validate_mlir

_S = tir.TIRScalar
_TK = tile_ir.TileOpKind


def _ret(*operands: tile_ir.TileValue) -> tile_ir.TileOp:
    """A Tile-IR `RETURN` op with the given operands."""
    return tile_ir.TileOp(_TK.RETURN, operands=list(operands))


def _fn(name: str, params: list[tile_ir.TileValue],
        return_ty: tir.TIRType,
        *ops: tile_ir.TileOp) -> tile_ir.TileFn:
    """A single-entry-block Tile-IR function carrying `ops`."""
    return tile_ir.TileFn(
        name, list(params), return_ty,
        [tile_ir.TileBlock(0, list(params), list(ops))])


# --------------------------------------------------------------------------
# MLIRTranslationError
# --------------------------------------------------------------------------
def test_mlir_translation_error_is_exception():
    """`MLIRTranslationError` is a plain `Exception` subclass — the
    fail-closed signal of the translator."""
    assert issubclass(MLIRTranslationError, Exception)


# --------------------------------------------------------------------------
# scalar types
# --------------------------------------------------------------------------
def test_render_scalar_types():
    """Helix scalar dtypes render to their MLIR spelling — integers are
    signless (`u32` and `i32` both `i32`), `isize`/`usize` are 64-bit,
    floats map 1:1, `bool` is `i1`."""
    cases = {
        "bool": "i1", "i8": "i8", "u8": "i8", "i16": "i16",
        "u16": "i16", "i32": "i32", "u32": "i32", "i64": "i64",
        "u64": "i64", "isize": "i64", "usize": "i64",
        "f16": "f16", "bf16": "bf16", "f32": "f32", "f64": "f64",
    }
    for dtype, mlir in cases.items():
        assert render_mlir_type(_S(dtype)) == mlir, dtype


def test_render_scalar_char_fails_closed():
    """`char` has no MLIR type — its bit width is not pinned (the LLVM
    path defers it the same way) — so it fails closed."""
    with pytest.raises(MLIRTranslationError, match="char"):
        render_mlir_type(_S("char"))


def test_render_scalar_quantized_fails_closed():
    """The front-end-only quantized dtypes have no backend codegen and
    no MLIR type — each fails closed, never a guessed width."""
    for dtype in ("fp8", "mxfp4", "nvfp4", "ternary"):
        with pytest.raises(MLIRTranslationError, match="no MLIR type"):
            render_mlir_type(_S(dtype))


def test_render_scalar_unknown_fails_closed():
    """An unrecognised dtype name fails closed."""
    with pytest.raises(MLIRTranslationError):
        render_mlir_type(_S("not_a_dtype"))


# --------------------------------------------------------------------------
# shape dimensions
# --------------------------------------------------------------------------
def test_render_dim():
    """A constant dimension renders as its integer; a dynamic,
    symbolic, or computed dimension renders as MLIR's dynamic `?`."""
    assert render_dim(tir.DimConst(4)) == "4"
    assert render_dim(tir.DimConst(0)) == "0"
    assert render_dim(tir.DimDyn()) == "?"
    assert render_dim(tir.DimVar("N")) == "?"
    assert render_dim(tir.DimExpr("+", (tir.DimConst(1),
                                       tir.DimVar("N")))) == "?"


def test_render_dim_negative_fails_closed():
    """A negative constant dimension is not a valid MLIR extent —
    fails closed."""
    with pytest.raises(MLIRTranslationError, match="negative"):
        render_dim(tir.DimConst(-1))


# --------------------------------------------------------------------------
# tensor / tile / tuple / unit types
# --------------------------------------------------------------------------
def test_render_tensor_type():
    """`TIRTensorTy` renders to `tensor<dims x dtype>` — a 0-d tensor
    has no dims, a non-constant size becomes a dynamic `?`."""
    assert render_mlir_type(
        tir.TIRTensorTy(_S("f32"),
                        (tir.DimConst(2), tir.DimConst(3)))
    ) == "tensor<2x3xf32>"
    assert render_mlir_type(
        tir.TIRTensorTy(_S("f32"), ())) == "tensor<f32>"
    assert render_mlir_type(
        tir.TIRTensorTy(_S("i32"), (tir.DimDyn(), tir.DimConst(4)))
    ) == "tensor<?x4xi32>"


def test_render_tensor_non_default_layout_fails_closed():
    """A non-default tensor layout (COL_MAJOR / BLOCKED) is
    correctness-relevant and has no slot in MLIR's builtin `tensor`
    type — the translator fails closed rather than silently erase it
    into a default row-major spelling."""
    for layout in (tir.Layout.COL_MAJOR, tir.Layout.BLOCKED):
        ty = tir.TIRTensorTy(_S("f32"), (tir.DimConst(2),),
                             layout=layout)
        with pytest.raises(MLIRTranslationError,
                           match="non-default layout"):
            render_mlir_type(ty)


def test_render_tensor_non_default_device_fails_closed():
    """A non-default tensor device likewise fails closed — it has no
    builtin-`tensor`-type slot and must not be silently erased."""
    ty = tir.TIRTensorTy(_S("f32"), (tir.DimConst(2),), device="cuda")
    with pytest.raises(MLIRTranslationError, match="non-default device"):
        render_mlir_type(ty)


def test_render_tile_type():
    """`TIRTileTy` renders to `vector<dims x dtype>` — MLIR's tile /
    SIMD type. `memspace` is not part of the spelling."""
    assert render_mlir_type(
        tir.TIRTileTy(_S("f32"),
                      (tir.DimConst(8), tir.DimConst(8)), "reg")
    ) == "vector<8x8xf32>"
    assert render_mlir_type(
        tir.TIRTileTy(_S("i32"), (tir.DimConst(16),), "smem")
    ) == "vector<16xi32>"


def test_render_tile_non_static_fails_closed():
    """An MLIR `vector` requires static dimensions — a tile with a
    dynamic / symbolic extent fails closed."""
    with pytest.raises(MLIRTranslationError, match="non-constant"):
        render_mlir_type(
            tir.TIRTileTy(_S("f32"), (tir.DimDyn(),), "reg"))


def test_render_tile_zero_d_fails_closed():
    """A 0-dimensional tile has no MLIR `vector` rendering — fails
    closed (an MLIR vector has at least one dimension)."""
    with pytest.raises(MLIRTranslationError, match="0-dimensional"):
        render_mlir_type(tir.TIRTileTy(_S("f32"), (), "reg"))


def test_render_tuple_type():
    """`TIRTuple` renders to `tuple<...>`, recursively over its
    elements."""
    assert render_mlir_type(
        tir.TIRTuple((_S("i32"), _S("f32")))) == "tuple<i32, f32>"
    # nested — `bool` is the Helix dtype name; it renders to MLIR `i1`
    assert render_mlir_type(
        tir.TIRTuple((_S("bool"),
                      tir.TIRTensorTy(_S("f32"), (tir.DimConst(2),))))
    ) == "tuple<i1, tensor<2xf32>>"


def test_render_empty_tuple_fails_closed():
    """An empty tuple has no MLIR `tuple` spelling — fails closed (a
    no-value result is `TIRUnit`, not an empty tuple)."""
    with pytest.raises(MLIRTranslationError, match="empty TIRTuple"):
        render_mlir_type(tir.TIRTuple(()))


def test_render_unit_type():
    """`TIRUnit` renders to MLIR's builtin unit type `none`."""
    assert render_mlir_type(tir.TIRUnit()) == "none"


def test_render_unknown_tir_type_fails_closed():
    """An object that is not a known `tir.TIRType` subclass fails
    closed — the translator never guesses a rendering."""
    with pytest.raises(MLIRTranslationError, match="unknown type"):
        render_mlir_type(object())          # type: ignore[arg-type]


# --------------------------------------------------------------------------
# module-load coverage guards
# --------------------------------------------------------------------------
def test_tir_type_coverage_guard():
    """`_check_tir_type_coverage` is callable and passes — `_TYPE_
    RENDERERS` covers exactly the concrete `tir.TIRType` subclasses."""
    emit._check_tir_type_coverage()  # must not raise
    assert {t.__name__ for t in emit._TYPE_RENDERERS} == {
        t.__name__ for t in tir.TIRType.__subclasses__()}


def test_dim_coverage_guard():
    """`_check_dim_coverage` is callable and passes — `_DIM_RENDERERS`
    covers exactly the concrete `tir.Dim` subclasses."""
    emit._check_dim_coverage()  # must not raise
    assert {t.__name__ for t in emit._DIM_RENDERERS} == {
        t.__name__ for t in tir.Dim.__subclasses__()}


def test_coverage_guards_are_not_vacuous(monkeypatch):
    """Each coverage guard genuinely catches an uncovered type — drop a
    renderer and confirm the guard raises, naming the now-unhandled
    class. A guard that always passed would not protect against drift."""
    short_types = dict(emit._TYPE_RENDERERS)
    del short_types[tir.TIRUnit]
    monkeypatch.setattr(emit, "_TYPE_RENDERERS", short_types)
    with pytest.raises(AssertionError, match=r"unhandled.*TIRUnit"):
        emit._check_tir_type_coverage()

    short_dims = dict(emit._DIM_RENDERERS)
    del short_dims[tir.DimExpr]
    monkeypatch.setattr(emit, "_DIM_RENDERERS", short_dims)
    with pytest.raises(AssertionError, match=r"unhandled.*DimExpr"):
        emit._check_dim_coverage()


# --------------------------------------------------------------------------
# emit_mlir_module — the module / function emitter (chunk B)
# --------------------------------------------------------------------------
def test_emit_empty_module():
    """An empty Tile-IR module emits an empty MLIR `module {}` — valid,
    if degenerate."""
    text = emit_mlir_module(tile_ir.TileModule())
    assert text == "module {\n}\n"


def test_emit_void_function():
    """A void function (a `TIRUnit` return) emits a `func.func` with no
    `->` result and a bare `func.return`."""
    text = emit_mlir_module(tile_ir.TileModule(
        functions={"f": _fn("f", [], tir.TIRUnit(), _ret())}))
    assert text == (
        "module {\n"
        "  func.func @f() {\n"
        "    func.return\n"
        "  }\n"
        "}\n")


def test_emit_identity_function():
    """An identity function `g(x: i32) -> i32` carries the parameter in
    the signature and returns it — params and results are `%v<id>`."""
    x = tile_ir.TileValue(0, _S("i32"))
    text = emit_mlir_module(tile_ir.TileModule(
        functions={"g": _fn("g", [x], _S("i32"), _ret(x))}))
    assert text == (
        "module {\n"
        "  func.func @g(%v0: i32) -> i32 {\n"
        "    func.return %v0 : i32\n"
        "  }\n"
        "}\n")


def test_emit_fails_closed_on_multi_block():
    """A multi-block function fails closed — chunk B emits only
    single-block functions; the `cf.br` / `^bb` CFG machinery is a
    later chunk, and emitting `^bb` blocks with no branch to reach
    them would be invalid MLIR."""
    fn = tile_ir.TileFn(
        "f", [], tir.TIRUnit(),
        [tile_ir.TileBlock(0, [], [_ret()]),
         tile_ir.TileBlock(1, [], [_ret()])])
    with pytest.raises(MLIRTranslationError, match="multi-block"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_fails_closed_on_return_arity_mismatch():
    """A `return` whose operand count is inconsistent with the declared
    result type fails closed — a unit function with an operand-carrying
    `return`, or a value function with a bare one."""
    x = tile_ir.TileValue(0, _S("i32"))
    unit_bad = _fn("f", [x], tir.TIRUnit(), _ret(x))    # unit + operand
    with pytest.raises(MLIRTranslationError, match="returns unit"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": unit_bad}))
    value_bad = _fn("g", [], _S("i32"), _ret())         # value + bare
    with pytest.raises(MLIRTranslationError, match="returns a value"):
        emit_mlir_module(tile_ir.TileModule(functions={"g": value_bad}))


def test_emit_fails_closed_on_return_type_mismatch():
    """A `return` whose operand type is not the function's declared
    result type fails closed — the translator never emits a
    type-mismatched `func.return`."""
    x = tile_ir.TileValue(0, _S("i32"))      # an i32 value
    fn = _fn("g", [x], _S("f32"), _ret(x))   # ...but the fn returns f32
    with pytest.raises(MLIRTranslationError,
                       match="does not match the declared result"):
        emit_mlir_module(tile_ir.TileModule(functions={"g": fn}))


def test_emit_fails_closed_on_entry_param_divergence():
    """A function whose entry block's parameters diverge from the
    signature parameters fails closed — emitting the signature from one
    list and the body against another would produce undefined SSA
    references."""
    sig_param = tile_ir.TileValue(0, _S("i32"))
    other_param = tile_ir.TileValue(7, _S("i32"))   # a different id
    fn = tile_ir.TileFn(
        "f", [sig_param], tir.TIRUnit(),
        [tile_ir.TileBlock(0, [other_param], [_ret()])])
    with pytest.raises(MLIRTranslationError, match="differ from the"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_multiple_functions():
    """A module with several functions emits a `func.func` for each."""
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "a": _fn("a", [], tir.TIRUnit(), _ret()),
        "b": _fn("b", [], tir.TIRUnit(), _ret()),
    }))
    assert "func.func @a()" in text and "func.func @b()" in text


def test_emit_output_passes_mock_validate():
    """The emitter's output is structurally well-formed — `mock_
    validate_mlir` returns DEFERRED (clean shape, real validity
    unverified), never FAILED."""
    x = tile_ir.TileValue(0, _S("f32"))
    text = emit_mlir_module(tile_ir.TileModule(
        functions={"g": _fn("g", [x], _S("f32"), _ret(x))}))
    result = mock_validate_mlir(text)
    assert result.deferred(), result.findings
    assert not result.failed()


def test_emit_fails_closed_on_unhandled_op():
    """An op kind with no emitter yet fails closed — the translator
    raises `MLIRTranslationError` naming the op, never emits a guess.
    Uses `TILE_MATMUL`, a representative not-yet-emitted op (a later
    chunk will add it, at which point this test picks another)."""
    fn = _fn("h", [], tir.TIRUnit(), tile_ir.TileOp(_TK.TILE_MATMUL))
    with pytest.raises(MLIRTranslationError, match="TILE_MATMUL"):
        emit_mlir_module(tile_ir.TileModule(functions={"h": fn}))


def test_emit_fails_closed_on_function_with_no_blocks():
    """A function with no blocks cannot form a `func.func` body —
    fails closed."""
    fn = tile_ir.TileFn("f", [], tir.TIRUnit(), [])
    with pytest.raises(MLIRTranslationError, match="no blocks"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_fails_closed_on_non_identifier_fn_name():
    """A function name with no plain MLIR symbol spelling fails
    closed."""
    fn = _fn("bad-name", [], tir.TIRUnit(), _ret())
    with pytest.raises(MLIRTranslationError, match="not an identifier"):
        emit_mlir_module(tile_ir.TileModule(functions={"bad-name": fn}))


# --------------------------------------------------------------------------
# the scalar `arith` op emitters (chunk C)
# --------------------------------------------------------------------------
def _const_int(v: tile_ir.TileValue, n: int) -> tile_ir.TileOp:
    return tile_ir.TileOp(_TK.SCALAR_CONST_INT, results=[v],
                          attrs={"value": n})


def test_emit_const_int():
    """`scalar.const_int` emits `%vR = arith.constant <n> : <iN>`."""
    c = tile_ir.TileValue(0, _S("i32"))
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [], _S("i32"), _const_int(c, 5), _ret(c))}))
    assert "%v0 = arith.constant 5 : i32" in text


def test_emit_const_float():
    """`scalar.const_float` emits `arith.constant` with a float
    literal; an integer-valued attr is rendered float-shaped (`4` ->
    `4.0`) so the literal matches its float type."""
    c = tile_ir.TileValue(0, _S("f32"))
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [], _S("f32"),
                 tile_ir.TileOp(_TK.SCALAR_CONST_FLOAT, results=[c],
                                attrs={"value": 2.5}), _ret(c))}))
    assert "%v0 = arith.constant 2.5 : f32" in text
    c2 = tile_ir.TileValue(0, _S("f32"))
    text2 = emit_mlir_module(tile_ir.TileModule(functions={
        "g": _fn("g", [], _S("f32"),
                 tile_ir.TileOp(_TK.SCALAR_CONST_FLOAT, results=[c2],
                                attrs={"value": 4}), _ret(c2))}))
    assert "arith.constant 4.0 : f32" in text2


def test_emit_const_float_fails_closed_on_non_finite():
    """`scalar.const_float` fails closed on infinity / NaN — MLIR's
    `arith.constant` has no plain float literal for them."""
    for bad in (float("inf"), float("-inf"), float("nan")):
        c = tile_ir.TileValue(0, _S("f32"))
        fn = _fn("f", [], _S("f32"),
                 tile_ir.TileOp(_TK.SCALAR_CONST_FLOAT, results=[c],
                                attrs={"value": bad}), _ret(c))
        with pytest.raises(MLIRTranslationError, match="not finite"):
            emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_const_float_scientific_notation():
    """A large-magnitude float constant emits a VALID MLIR literal — it
    carries a decimal point and a plain decimal exponent. Python's
    `repr(1e20)` is `'1e+20'`, which MLIR rejects (no `.`); the emitter
    renders `1.0e20`."""
    c = tile_ir.TileValue(0, _S("f32"))
    fn = _fn("f", [], _S("f32"),
             tile_ir.TileOp(_TK.SCALAR_CONST_FLOAT, results=[c],
                            attrs={"value": 1e20}), _ret(c))
    text = emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))
    assert "arith.constant 1.0e20 : f32" in text
    assert "e+" not in text          # no `+` in the exponent


def test_emit_scalar_add_int_and_float():
    """`scalar.add` picks the integer (`arith.addi`) or float
    (`arith.addf`) mnemonic by the operand type."""
    a = tile_ir.TileValue(0, _S("i32"))
    b = tile_ir.TileValue(1, _S("i32"))
    r = tile_ir.TileValue(2, _S("i32"))
    int_text = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [a, b], _S("i32"),
                 tile_ir.TileOp(_TK.SCALAR_ADD, operands=[a, b],
                                results=[r]), _ret(r))}))
    assert "%v2 = arith.addi %v0, %v1 : i32" in int_text
    x = tile_ir.TileValue(0, _S("f32"))
    y = tile_ir.TileValue(1, _S("f32"))
    z = tile_ir.TileValue(2, _S("f32"))
    flt_text = emit_mlir_module(tile_ir.TileModule(functions={
        "g": _fn("g", [x, y], _S("f32"),
                 tile_ir.TileOp(_TK.SCALAR_ADD, operands=[x, y],
                                results=[z]), _ret(z))}))
    assert "%v2 = arith.addf %v0, %v1 : f32" in flt_text


def test_emit_scalar_sub_and_mul():
    """`scalar.sub` / `scalar.mul` emit `arith.sub*` / `arith.mul*`."""
    a = tile_ir.TileValue(0, _S("i32"))
    b = tile_ir.TileValue(1, _S("i32"))
    r = tile_ir.TileValue(2, _S("i32"))
    sub = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [a, b], _S("i32"),
                 tile_ir.TileOp(_TK.SCALAR_SUB, operands=[a, b],
                                results=[r]), _ret(r))}))
    assert "arith.subi %v0, %v1 : i32" in sub
    mul = emit_mlir_module(tile_ir.TileModule(functions={
        "g": _fn("g", [a, b], _S("i32"),
                 tile_ir.TileOp(_TK.SCALAR_MUL, operands=[a, b],
                                results=[r]), _ret(r))}))
    assert "arith.muli %v0, %v1 : i32" in mul


def test_emit_arith_function_passes_mock_validate():
    """A whole arith function — constants + an add + a return — emits
    structurally well-formed MLIR (`mock_validate_mlir` -> DEFERRED)."""
    a = tile_ir.TileValue(0, _S("i32"))
    b = tile_ir.TileValue(1, _S("i32"))
    r = tile_ir.TileValue(2, _S("i32"))
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [], _S("i32"), _const_int(a, 2), _const_int(b, 3),
                 tile_ir.TileOp(_TK.SCALAR_ADD, operands=[a, b],
                                results=[r]), _ret(r))}))
    result = mock_validate_mlir(text)
    assert result.deferred(), result.findings


def test_emit_const_int_fails_closed_on_non_integer_value():
    """`scalar.const_int` fails closed on a non-integer `value`
    attribute — a float, or a bool (a bool is a distinct constant)."""
    for bad in (2.5, True):
        c = tile_ir.TileValue(0, _S("i32"))
        fn = _fn("f", [], _S("i32"),
                 tile_ir.TileOp(_TK.SCALAR_CONST_INT, results=[c],
                                attrs={"value": bad}), _ret(c))
        with pytest.raises(MLIRTranslationError, match="const_int"):
            emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_const_int_fails_closed_on_float_result_type():
    """`scalar.const_int` whose result type is a float fails closed."""
    c = tile_ir.TileValue(0, _S("f32"))
    fn = _fn("f", [], _S("f32"), _const_int(c, 5), _ret(c))
    with pytest.raises(MLIRTranslationError, match="non-integer result"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_scalar_binop_fails_closed_on_type_mismatch():
    """A scalar binop whose operand and result types are not all equal
    fails closed — a type-mismatched `arith` op is invalid MLIR."""
    a = tile_ir.TileValue(0, _S("f32"))      # f32 operand...
    b = tile_ir.TileValue(1, _S("i32"))
    r = tile_ir.TileValue(2, _S("i32"))      # ...i32 result
    fn = _fn("f", [a, b], _S("i32"),
             tile_ir.TileOp(_TK.SCALAR_ADD, operands=[a, b],
                            results=[r]), _ret(r))
    with pytest.raises(MLIRTranslationError, match="not all equal"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_scalar_binop_fails_closed_on_non_scalar():
    """A scalar binop on a non-scalar (tensor) type fails closed — the
    chunk-C `arith` emitters handle scalar ops only."""
    tens = tir.TIRTensorTy(_S("f32"), (tir.DimConst(4),))
    a = tile_ir.TileValue(0, tens)
    b = tile_ir.TileValue(1, tens)
    r = tile_ir.TileValue(2, tens)
    fn = _fn("f", [a, b], tens,
             tile_ir.TileOp(_TK.SCALAR_MUL, operands=[a, b],
                            results=[r]), _ret(r))
    with pytest.raises(MLIRTranslationError, match="not a scalar"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_scalar_binop_fails_closed_on_wrong_arity():
    """A scalar binop without exactly two operands fails closed."""
    a = tile_ir.TileValue(0, _S("i32"))
    r = tile_ir.TileValue(1, _S("i32"))
    fn = _fn("f", [a], _S("i32"),
             tile_ir.TileOp(_TK.SCALAR_ADD, operands=[a], results=[r]),
             _ret(r))
    with pytest.raises(MLIRTranslationError, match="expects 2 operands"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


# --------------------------------------------------------------------------
# the compare / select op emitters (chunk D)
# --------------------------------------------------------------------------
def _cmp(a: tile_ir.TileValue, b: tile_ir.TileValue,
         r: tile_ir.TileValue, kind: str) -> tile_ir.TileOp:
    return tile_ir.TileOp(_TK.SCALAR_CMP, operands=[a, b], results=[r],
                          attrs={"cmp": kind})


def test_emit_cmp_integer_signed_unsigned_and_eq():
    """`scalar.cmp` -> `arith.cmpi`: an ordered comparison is `slt` on a
    signed operand and `ult` on an unsigned one; `eq` is sign-agnostic."""
    for dtype, kind, pred in (("i32", "cmp.lt", "slt"),
                              ("u32", "cmp.lt", "ult"),
                              ("u32", "cmp.eq", "eq")):
        a = tile_ir.TileValue(0, _S(dtype))
        b = tile_ir.TileValue(1, _S(dtype))
        r = tile_ir.TileValue(2, _S("bool"))
        text = emit_mlir_module(tile_ir.TileModule(functions={
            "f": _fn("f", [a, b], _S("bool"), _cmp(a, b, r, kind),
                     _ret(r))}))
        assert f"arith.cmpi {pred}, %v0, %v1 : i32" in text, (dtype, kind)


def test_emit_cmp_float_predicates():
    """`scalar.cmp` on a float operand -> `arith.cmpf`. `==` and the
    relational comparisons are ORDERED (`o*`); `!=` is UNORDERED-not-
    equal (`une`, so `NaN != NaN` is true) — matching Helix's float
    `!=` semantics, the x86_64 backend's reference."""
    cases = {
        "cmp.eq": "oeq", "cmp.ne": "une", "cmp.lt": "olt",
        "cmp.le": "ole", "cmp.gt": "ogt", "cmp.ge": "oge",
    }
    for kind, pred in cases.items():
        a = tile_ir.TileValue(0, _S("f32"))
        b = tile_ir.TileValue(1, _S("f32"))
        r = tile_ir.TileValue(2, _S("bool"))
        text = emit_mlir_module(tile_ir.TileModule(functions={
            "f": _fn("f", [a, b], _S("bool"), _cmp(a, b, r, kind),
                     _ret(r))}))
        assert f"arith.cmpf {pred}, %v0, %v1 : f32" in text, kind


def test_emit_select():
    """`scalar.select` -> `arith.select %cond, %a, %b : <T>`."""
    c = tile_ir.TileValue(0, _S("bool"))
    x = tile_ir.TileValue(1, _S("i32"))
    y = tile_ir.TileValue(2, _S("i32"))
    r = tile_ir.TileValue(3, _S("i32"))
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [c, x, y], _S("i32"),
                 tile_ir.TileOp(_TK.SCALAR_SELECT, operands=[c, x, y],
                                results=[r]), _ret(r))}))
    assert "%v3 = arith.select %v0, %v1, %v2 : i32" in text


def test_emit_cmp_select_function_passes_mock_validate():
    """A function using a comparison and a select emits structurally
    well-formed MLIR (`mock_validate_mlir` -> DEFERRED)."""
    a = tile_ir.TileValue(0, _S("i32"))
    b = tile_ir.TileValue(1, _S("i32"))
    cond = tile_ir.TileValue(2, _S("bool"))
    r = tile_ir.TileValue(3, _S("i32"))
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [a, b], _S("i32"), _cmp(a, b, cond, "cmp.lt"),
                 tile_ir.TileOp(_TK.SCALAR_SELECT, operands=[cond, a, b],
                                results=[r]), _ret(r))}))
    assert mock_validate_mlir(text).deferred(), text


def test_emit_cmp_fails_closed_on_unknown_predicate():
    """`scalar.cmp` with an unrecognised / missing `cmp` attribute
    fails closed."""
    a = tile_ir.TileValue(0, _S("i32"))
    b = tile_ir.TileValue(1, _S("i32"))
    r = tile_ir.TileValue(2, _S("bool"))
    for bad in (_cmp(a, b, r, "cmp.bogus"),
                tile_ir.TileOp(_TK.SCALAR_CMP, operands=[a, b],
                               results=[r])):           # no `cmp` attr
        fn = _fn("f", [a, b], _S("bool"), bad, _ret(r))
        with pytest.raises(MLIRTranslationError,
                           match="recognised `cmp`"):
            emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_cmp_fails_closed_on_non_bool_result():
    """`scalar.cmp` whose result type is not i1 fails closed — a
    comparison produces a boolean."""
    a = tile_ir.TileValue(0, _S("i32"))
    b = tile_ir.TileValue(1, _S("i32"))
    r = tile_ir.TileValue(2, _S("i32"))      # i32, not bool
    fn = _fn("f", [a, b], _S("i32"), _cmp(a, b, r, "cmp.lt"), _ret(r))
    with pytest.raises(MLIRTranslationError, match="result type is not i1"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_cmp_fails_closed_on_operand_mismatch():
    """`scalar.cmp` with operands of different types fails closed."""
    a = tile_ir.TileValue(0, _S("i32"))
    b = tile_ir.TileValue(1, _S("f32"))      # mismatched
    r = tile_ir.TileValue(2, _S("bool"))
    fn = _fn("f", [a, b], _S("bool"), _cmp(a, b, r, "cmp.lt"), _ret(r))
    with pytest.raises(MLIRTranslationError, match="different types"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_select_fails_closed_on_non_bool_condition():
    """`scalar.select` whose condition is not i1 fails closed."""
    c = tile_ir.TileValue(0, _S("i32"))      # i32 condition, not bool
    x = tile_ir.TileValue(1, _S("i32"))
    y = tile_ir.TileValue(2, _S("i32"))
    r = tile_ir.TileValue(3, _S("i32"))
    fn = _fn("f", [c, x, y], _S("i32"),
             tile_ir.TileOp(_TK.SCALAR_SELECT, operands=[c, x, y],
                            results=[r]), _ret(r))
    with pytest.raises(MLIRTranslationError,
                       match="condition type is not i1"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_select_fails_closed_on_arm_type_mismatch():
    """`scalar.select` whose two arms / result are not all one type
    fails closed."""
    c = tile_ir.TileValue(0, _S("bool"))
    x = tile_ir.TileValue(1, _S("i32"))
    y = tile_ir.TileValue(2, _S("f32"))      # mismatched arm
    r = tile_ir.TileValue(3, _S("i32"))
    fn = _fn("f", [c, x, y], _S("i32"),
             tile_ir.TileOp(_TK.SCALAR_SELECT, operands=[c, x, y],
                            results=[r]), _ret(r))
    with pytest.raises(MLIRTranslationError,
                       match="share one type"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_cmp_predicate_tables_guard():
    """`_check_cmp_predicate_tables` is callable and passes — the
    `arith.cmpi` and `arith.cmpf` tables cover the same comparison
    kinds."""
    emit._check_cmp_predicate_tables()      # must not raise
    assert set(emit._CMPI_PREDICATES) == set(emit._CMPF_PREDICATES)


def test_cmp_predicate_tables_guard_is_not_vacuous(monkeypatch):
    """`_check_cmp_predicate_tables` genuinely catches a predicate
    table that has drifted from the Tensor-IR comparison kinds."""
    broken = dict(emit._CMPI_PREDICATES)
    del broken["cmp.lt"]
    monkeypatch.setattr(emit, "_CMPI_PREDICATES", broken)
    with pytest.raises(AssertionError, match="_CMPI_PREDICATES"):
        emit._check_cmp_predicate_tables()


# --------------------------------------------------------------------------
# the elementwise `vector` tile-op emitters (chunk E)
# --------------------------------------------------------------------------
def _tile(dtype: str, *dims: int) -> tir.TIRTileTy:
    """A Tile-IR tile type — renders to `vector<...x<dtype>>`."""
    return tir.TIRTileTy(_S(dtype),
                         tuple(tir.DimConst(d) for d in dims), "reg")


def test_emit_tile_add_int_and_float():
    """`tile.add` -> `arith.add{i,f}` on `vector<...>`-typed operands —
    the same mnemonic as the scalar core, the int / float choice the
    tile's element dtype."""
    for dtype, mnemonic in (("i32", "arith.addi"),
                            ("f32", "arith.addf")):
        t = _tile(dtype, 8, 8)
        a = tile_ir.TileValue(0, t)
        b = tile_ir.TileValue(1, t)
        r = tile_ir.TileValue(2, t)
        text = emit_mlir_module(tile_ir.TileModule(functions={
            "f": _fn("f", [a, b], t,
                     tile_ir.TileOp(_TK.TILE_ADD, operands=[a, b],
                                    results=[r]), _ret(r))}))
        assert f"{mnemonic} %v0, %v1 : vector<8x8x{dtype}>" in text


def test_emit_tile_sub_and_mul():
    """`tile.sub` / `tile.mul` -> `arith.sub*` / `arith.mul*` on
    vectors."""
    t = _tile("f32", 4)
    a = tile_ir.TileValue(0, t)
    b = tile_ir.TileValue(1, t)
    r = tile_ir.TileValue(2, t)
    sub = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [a, b], t,
                 tile_ir.TileOp(_TK.TILE_SUB, operands=[a, b],
                                results=[r]), _ret(r))}))
    assert "arith.subf %v0, %v1 : vector<4xf32>" in sub
    mul = emit_mlir_module(tile_ir.TileModule(functions={
        "g": _fn("g", [a, b], t,
                 tile_ir.TileOp(_TK.TILE_MUL, operands=[a, b],
                                results=[r]), _ret(r))}))
    assert "arith.mulf %v0, %v1 : vector<4xf32>" in mul


def test_emit_tile_zeros_int_and_float():
    """`tile.zeros` -> a `dense<0>` / `dense<0.0>` splat
    `arith.constant`."""
    ti = _tile("i32", 8, 8)
    zi = tile_ir.TileValue(0, ti)
    int_text = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [], ti,
                 tile_ir.TileOp(_TK.TILE_ZEROS, results=[zi]),
                 _ret(zi))}))
    assert "arith.constant dense<0> : vector<8x8xi32>" in int_text
    tf = _tile("f32", 8, 8)
    zf = tile_ir.TileValue(0, tf)
    flt_text = emit_mlir_module(tile_ir.TileModule(functions={
        "g": _fn("g", [], tf,
                 tile_ir.TileOp(_TK.TILE_ZEROS, results=[zf]),
                 _ret(zf))}))
    assert "arith.constant dense<0.0> : vector<8x8xf32>" in flt_text


def test_emit_tile_function_passes_mock_validate():
    """A tile function — a zero tile, a tile add, a return — emits
    structurally well-formed MLIR (`mock_validate_mlir` -> DEFERRED)."""
    t = _tile("f32", 8, 8)
    z = tile_ir.TileValue(0, t)
    a = tile_ir.TileValue(1, t)
    r = tile_ir.TileValue(2, t)
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [a], t,
                 tile_ir.TileOp(_TK.TILE_ZEROS, results=[z]),
                 tile_ir.TileOp(_TK.TILE_ADD, operands=[z, a],
                                results=[r]), _ret(r))}))
    assert mock_validate_mlir(text).deferred(), text


def test_emit_tile_binop_fails_closed_on_non_tile():
    """A tile binop on a non-tile (scalar) operand fails closed — the
    chunk-E `vector` emitters handle tile ops only."""
    a = tile_ir.TileValue(0, _S("i32"))      # scalar, not a tile
    b = tile_ir.TileValue(1, _S("i32"))
    r = tile_ir.TileValue(2, _S("i32"))
    fn = _fn("f", [a, b], _S("i32"),
             tile_ir.TileOp(_TK.TILE_ADD, operands=[a, b], results=[r]),
             _ret(r))
    with pytest.raises(MLIRTranslationError, match="not a tile"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_tile_binop_fails_closed_on_type_mismatch():
    """A tile binop whose operand / result tile types differ fails
    closed."""
    t8 = _tile("f32", 8, 8)
    t4 = _tile("f32", 4, 4)
    a = tile_ir.TileValue(0, t8)
    b = tile_ir.TileValue(1, t4)             # different tile shape
    r = tile_ir.TileValue(2, t8)
    fn = _fn("f", [a, b], t8,
             tile_ir.TileOp(_TK.TILE_MUL, operands=[a, b], results=[r]),
             _ret(r))
    with pytest.raises(MLIRTranslationError, match="not all equal"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_tile_zeros_fails_closed_on_operands():
    """`tile.zeros` takes no operands — one supplied fails closed."""
    t = _tile("f32", 8, 8)
    a = tile_ir.TileValue(0, t)
    z = tile_ir.TileValue(1, t)
    fn = _fn("f", [a], t,
             tile_ir.TileOp(_TK.TILE_ZEROS, operands=[a], results=[z]),
             _ret(z))
    with pytest.raises(MLIRTranslationError, match="takes no operands"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_tile_zeros_fails_closed_on_non_tile_result():
    """`tile.zeros` with a non-tile result type fails closed."""
    z = tile_ir.TileValue(0, _S("i32"))      # scalar, not a tile
    fn = _fn("f", [], _S("i32"),
             tile_ir.TileOp(_TK.TILE_ZEROS, results=[z]), _ret(z))
    with pytest.raises(MLIRTranslationError, match="not a tile"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


# --------------------------------------------------------------------------
# the layout-transform tile-op emitters (chunk F)
# --------------------------------------------------------------------------
def test_emit_tile_reshape():
    """`tile.reshape` -> `vector.shape_cast %src : <src> to <dst>`."""
    a = tile_ir.TileValue(0, _tile("f32", 4, 4))
    r = tile_ir.TileValue(1, _tile("f32", 16))
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [a], _tile("f32", 16),
                 tile_ir.TileOp(_TK.TILE_RESHAPE, operands=[a],
                                results=[r]), _ret(r))}))
    assert ("%v1 = vector.shape_cast %v0 : vector<4x4xf32> to "
            "vector<16xf32>") in text


def test_emit_tile_transpose():
    """`tile.transpose` -> `vector.transpose %src, [1, 0]` for a 2-D
    tile — the result is the operand's shape with the two dims
    swapped."""
    a = tile_ir.TileValue(0, _tile("f32", 2, 3))
    r = tile_ir.TileValue(1, _tile("f32", 3, 2))
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [a], _tile("f32", 3, 2),
                 tile_ir.TileOp(_TK.TILE_TRANSPOSE, operands=[a],
                                results=[r]), _ret(r))}))
    assert ("%v1 = vector.transpose %v0, [1, 0] : vector<2x3xf32> to "
            "vector<3x2xf32>") in text


def test_emit_tile_layout_function_passes_mock_validate():
    """A function using a tile reshape emits structurally well-formed
    MLIR (`mock_validate_mlir` -> DEFERRED)."""
    a = tile_ir.TileValue(0, _tile("i32", 8, 2))
    r = tile_ir.TileValue(1, _tile("i32", 16))
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "f": _fn("f", [a], _tile("i32", 16),
                 tile_ir.TileOp(_TK.TILE_RESHAPE, operands=[a],
                                results=[r]), _ret(r))}))
    assert mock_validate_mlir(text).deferred(), text


def test_emit_tile_reshape_fails_closed_on_element_count_mismatch():
    """`tile.reshape` whose source and result differ in total element
    count fails closed — a `shape_cast` preserves the count."""
    a = tile_ir.TileValue(0, _tile("f32", 4, 4))     # 16 elements
    r = tile_ir.TileValue(1, _tile("f32", 9))        # 9 elements
    fn = _fn("f", [a], _tile("f32", 9),
             tile_ir.TileOp(_TK.TILE_RESHAPE, operands=[a],
                            results=[r]), _ret(r))
    with pytest.raises(MLIRTranslationError,
                       match="total element count"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_tile_reshape_fails_closed_on_dtype_change():
    """`tile.reshape` that changes the element dtype fails closed."""
    a = tile_ir.TileValue(0, _tile("f32", 4, 4))
    r = tile_ir.TileValue(1, _tile("i32", 16))       # f32 -> i32
    fn = _fn("f", [a], _tile("i32", 16),
             tile_ir.TileOp(_TK.TILE_RESHAPE, operands=[a],
                            results=[r]), _ret(r))
    with pytest.raises(MLIRTranslationError, match="element dtype"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_tile_reshape_fails_closed_on_non_tile():
    """`tile.reshape` on a non-tile (scalar) operand fails closed."""
    a = tile_ir.TileValue(0, _S("i32"))
    r = tile_ir.TileValue(1, _S("i32"))
    fn = _fn("f", [a], _S("i32"),
             tile_ir.TileOp(_TK.TILE_RESHAPE, operands=[a],
                            results=[r]), _ret(r))
    with pytest.raises(MLIRTranslationError, match="must both be tiles"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_tile_transpose_fails_closed_on_non_2d():
    """`tile.transpose` of a non-2-D tile fails closed — chunk F
    handles the 2-D case only (an N-D permutation needs an explicit
    attribute)."""
    a = tile_ir.TileValue(0, _tile("f32", 2, 3, 4))    # 3-D
    r = tile_ir.TileValue(1, _tile("f32", 4, 3, 2))
    fn = _fn("f", [a], _tile("f32", 4, 3, 2),
             tile_ir.TileOp(_TK.TILE_TRANSPOSE, operands=[a],
                            results=[r]), _ret(r))
    with pytest.raises(MLIRTranslationError, match="2-D tile transpose"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_tile_transpose_fails_closed_on_dtype_change():
    """`tile.transpose` that changes the element dtype fails closed —
    a transpose permutes axes, it does not recast the element type."""
    a = tile_ir.TileValue(0, _tile("f32", 2, 3))
    r = tile_ir.TileValue(1, _tile("i32", 3, 2))       # f32 -> i32
    fn = _fn("f", [a], _tile("i32", 3, 2),
             tile_ir.TileOp(_TK.TILE_TRANSPOSE, operands=[a],
                            results=[r]), _ret(r))
    with pytest.raises(MLIRTranslationError, match="element dtype"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


def test_emit_tile_transpose_fails_closed_on_wrong_result_shape():
    """`tile.transpose` whose result is not the operand's shape
    transposed fails closed."""
    a = tile_ir.TileValue(0, _tile("f32", 2, 3))
    r = tile_ir.TileValue(1, _tile("f32", 2, 3))        # not swapped
    fn = _fn("f", [a], _tile("f32", 2, 3),
             tile_ir.TileOp(_TK.TILE_TRANSPOSE, operands=[a],
                            results=[r]), _ret(r))
    with pytest.raises(MLIRTranslationError,
                       match="not the operand's shape transposed"):
        emit_mlir_module(tile_ir.TileModule(functions={"f": fn}))


# --------------------------------------------------------------------------
# the function-call emitter (chunk G)
# --------------------------------------------------------------------------
def _call(target: object, operands: list[tile_ir.TileValue],
          results: list[tile_ir.TileValue]) -> tile_ir.TileOp:
    """A Tile-IR `CALL` op naming `target` as the callee."""
    return tile_ir.TileOp(_TK.CALL, operands=list(operands),
                          results=list(results), attrs={"target": target})


def test_emit_call_with_result():
    """`call` -> `%vR = func.call @callee(%args) : (argtypes) ->
    rettype` — a value-returning direct call; the callee name is the
    Tile-IR `target` attribute."""
    x = tile_ir.TileValue(0, _S("i32"))
    r = tile_ir.TileValue(1, _S("i32"))
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "caller": _fn("caller", [x], _S("i32"),
                      _call("callee", [x], [r]), _ret(r))}))
    assert "%v1 = func.call @callee(%v0) : (i32) -> i32" in text


def test_emit_call_void():
    """A call with no result — hand-built Tile-IR — emits `func.call
    @callee() : () -> ()` with no SSA result binding."""
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "caller": _fn("caller", [], tir.TIRUnit(),
                      _call("side_effect", [], []), _ret())}))
    assert "func.call @side_effect() : () -> ()" in text
    assert "= func.call" not in text          # no result binding


def test_emit_call_unit_result_is_void():
    """A call to a unit-returning callee — the shape the front end
    builds: a CALL carrying ONE `TIRUnit`-typed result — is a VOID call
    `func.call @callee() : () -> ()`. Unit is not a materialized MLIR
    value, so the result binds no `%v<id>` and the signature never ends
    `-> none` (a dangling SSA name no consumer could use)."""
    u = tile_ir.TileValue(0, tir.TIRUnit())
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "caller": _fn("caller", [], tir.TIRUnit(),
                      _call("side_effect", [], [u]), _ret())}))
    assert "func.call @side_effect() : () -> ()" in text
    assert "-> none" not in text               # never a dangling none
    assert "= func.call" not in text           # no result binding


def test_emit_call_nullary_with_result():
    """A no-argument value-returning call emits an empty argument list
    and an empty `()` signature — `%vR = func.call @f() : () -> T`."""
    r = tile_ir.TileValue(0, _S("f32"))
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "caller": _fn("caller", [], _S("f32"),
                      _call("make", [], [r]), _ret(r))}))
    assert "%v0 = func.call @make() : () -> f32" in text


def test_emit_call_multiple_args():
    """A call passes its operands and their types in order — the
    argument list and the `(argtypes)` signature line up."""
    a = tile_ir.TileValue(0, _S("i32"))
    b = tile_ir.TileValue(1, _S("f32"))
    r = tile_ir.TileValue(2, _S("i32"))
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "caller": _fn("caller", [a, b], _S("i32"),
                      _call("g", [a, b], [r]), _ret(r))}))
    assert "%v2 = func.call @g(%v0, %v1) : (i32, f32) -> i32" in text


def test_emit_call_function_passes_mock_validate():
    """A function that calls another emits structurally well-formed
    MLIR (`mock_validate_mlir` -> DEFERRED)."""
    x = tile_ir.TileValue(0, _S("i32"))
    r = tile_ir.TileValue(1, _S("i32"))
    text = emit_mlir_module(tile_ir.TileModule(functions={
        "caller": _fn("caller", [x], _S("i32"),
                      _call("callee", [x], [r]), _ret(r))}))
    assert mock_validate_mlir(text).deferred(), text


def test_emit_call_fails_closed_on_missing_target():
    """A `call` with no `target` attribute fails closed — a callee with
    no MLIR symbol name has no faithful `func.call`."""
    fn = _fn("caller", [], tir.TIRUnit(),
             tile_ir.TileOp(_TK.CALL), _ret())
    with pytest.raises(MLIRTranslationError, match="no identifier"):
        emit_mlir_module(tile_ir.TileModule(functions={"caller": fn}))


def test_emit_call_fails_closed_on_non_identifier_target():
    """A `call` whose `target` is not a plain identifier fails closed —
    a name with no MLIR symbol spelling, or a non-string."""
    for bad in ("bad-name", 123):
        fn = _fn("caller", [], tir.TIRUnit(),
                 _call(bad, [], []), _ret())
        with pytest.raises(MLIRTranslationError, match="no identifier"):
            emit_mlir_module(tile_ir.TileModule(functions={"caller": fn}))


def test_emit_call_fails_closed_on_multiple_results():
    """A `call` with more than one result fails closed — a Helix
    function returns one value or unit."""
    r1 = tile_ir.TileValue(0, _S("i32"))
    r2 = tile_ir.TileValue(1, _S("i32"))
    fn = _fn("caller", [], tir.TIRUnit(),
             _call("g", [], [r1, r2]), _ret())
    with pytest.raises(MLIRTranslationError,
                       match="returns one value or unit"):
        emit_mlir_module(tile_ir.TileModule(functions={"caller": fn}))


# --------------------------------------------------------------------------
# the mock-path rule — emit is pure text, never `import mlir`
# --------------------------------------------------------------------------
def test_emit_module_is_pure_text_no_mlir_import():
    """THE MOCK-PATH RULE (Stage 210 decision, section 3): `emit` is a
    pure-text translator — it NEVER `import mlir`, at module top level
    or anywhere. Parse the module's AST and confirm not one
    `import mlir` / `from mlir ...` statement."""
    tree = ast.parse(Path(emit.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("mlir"), a.name
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("mlir"), node.module
