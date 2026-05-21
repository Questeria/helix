"""
helixc/ir/mlir/emit.py — Helix IR -> MLIR text translation
(v3.0 Phase E, Stage 212).

Stage 212 builds the PARALLEL MLIR path: a translator that walks a
Helix IR module and emits MLIR textual IR, additive alongside the
home-grown tile-IR -> backends path (Stage 215 parity-gates the two).

Chunk A — the TYPE BRIDGE. Before any op can be emitted, every Helix
IR type must render as MLIR type syntax. `render_mlir_type` maps each
`tir.TIRType` to its MLIR spelling:

- `TIRScalar`   -> an MLIR scalar (`i32`, `f32`, `i1`, ...);
- `TIRTensorTy` -> `tensor<...>` (a non-constant size dimension
  becomes a dynamic `?`);
- `TIRTileTy`   -> `vector<...>` (MLIR's tile / SIMD type — which
  requires STATIC dimensions);
- `TIRTuple`    -> `tuple<...>`;
- `TIRUnit`     -> `none` (MLIR's unit type).

Chunk B — the MODULE / FUNCTION EMITTER. `emit_mlir_module` walks a
Tile-IR module (`tile_ir.TileModule`) and emits the MLIR
`module { func.func @name(...) -> ... { ... } }` text. The translator
works from the TILE IR — the plan (docs/V3_PLAN.md) names Stage 212
"tile-IR -> MLIR" and Stage 215 parity-gates "the MLIR path vs. the
home-grown tile-IR path", so the tile IR is the branch point. Chunk B
emits SINGLE-BLOCK functions and `func.return`; multi-block CFG (the
`cf.br` / `^bb` machinery) and the per-op bodies are later chunks —
the translator FAILS CLOSED on them, and on any function whose
signature is internally inconsistent.

Chunk C — the scalar `arith` op emitters. The first per-op chunk:
`scalar.const_int` / `const_float` -> `arith.constant`, and
`scalar.add` / `sub` / `mul` -> `arith.{add,sub,mul}{i,f}` (the
integer vs. float mnemonic chosen by the operand type). The emitters
are STATELESS — every Tile-IR value's SSA name is `%v<id>`, a pure
function of its id, so a result and its later uses name-match with no
per-function symbol table (MLIR emits `arith.constant` ops rather than
inlining constants, so a constant result has an ordinary `%v<id>`
name).

Chunk D — the compare / select op emitters. `scalar.cmp` ->
`arith.cmpi` / `arith.cmpf` (the predicate from the Tile-IR `cmp`
attribute; an integer ordered comparison is signed / unsigned by the
operand dtype), and `scalar.select` -> `arith.select`.

Chunk E — the elementwise `vector` tile-op emitters. `tile.add` /
`sub` / `mul` -> `arith.{add,sub,mul}{i,f}` on `vector<...>`-typed
operands (MLIR `arith` ops are elementwise-polymorphic over vectors —
the same mnemonics as the scalar core, the int / float choice being
the tile's element dtype); `tile.zeros` -> a `dense<0>`-splat
`arith.constant`. The non-elementwise tile ops (`tile.matmul` /
`reduce` / `transpose` / `reshape` / `const`), `memref` / `gpu`, the
`helix` dialect, and `scalar.neg` remain later chunks; `_emit_op`
FAILS CLOSED on them.

FAIL-CLOSED: a type with no faithful MLIR rendering — a width-unpinned
`char`, a front-end-only quantized dtype, a non-default tensor layout
or device, a non-static tile dimension, an unknown IR type — raises
`MLIRTranslationError`, never emits a guessed or lossy type. This is
the migration's hard rule: the translator never produces wrong MLIR;
an unsupported construct stops it loudly.

MOCK-PATH-FIRST: pure text — `render_mlir_type` builds strings, never
`import mlir`. The emitted text is shape-checked by
`validate.mock_validate_mlir`; real `mlir-opt` validation is a
binding-gated Stage-212+ concern.

License: Apache 2.0
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .. import tile_ir, tir


class MLIRTranslationError(Exception):
    """Raised when the Helix-IR -> MLIR translator hits a construct it
    cannot faithfully emit — an unsupported type, dtype, or (later)
    op. The translator FAILS CLOSED: it never emits a guessed or wrong
    MLIR fragment. The MLIR sibling of `llvm_ir.LLVMEmitError`."""


# Helix scalar dtype -> MLIR scalar type. MLIR integer types are
# SIGNLESS (like LLVM): `u32` and `i32` are both `i32`, signedness is
# per-op. `isize` / `usize` are 64-bit (the SysV target). MLIR's float
# spellings (`f16` / `bf16` / `f32` / `f64`) coincide with Helix's
# dtype names.
#
# Deliberately ABSENT, so `render_mlir_type` fails closed on them:
# - `char` — its bit width is not yet pinned (the LLVM path defers it
#   the same way, `llvm_ir._LLVM_INT_TYPES`);
# - the quantized dtypes `fp8` / `mxfp4` / `nvfp4` / `ternary` — they
#   are front-end-only, with no backend codegen or register model
#   (`regalloc_classes._RECOGNISED_SCALAR_DTYPES` omits them too).
_MLIR_SCALAR_TYPES: dict[str, str] = {
    "bool": "i1",
    "i8": "i8", "u8": "i8",
    "i16": "i16", "u16": "i16",
    "i32": "i32", "u32": "i32",
    "i64": "i64", "u64": "i64",
    "isize": "i64", "usize": "i64",
    "f16": "f16", "bf16": "bf16", "f32": "f32", "f64": "f64",
}


# --------------------------------------------------------------------------
# shape dimensions
# --------------------------------------------------------------------------
def _render_dim_const(d: tir.Dim) -> str:
    """A constant dimension renders as its integer value."""
    assert isinstance(d, tir.DimConst)
    if d.value < 0:
        raise MLIRTranslationError(
            f"tensor dimension is negative ({d.value}) — not a valid "
            f"MLIR shape extent")
    return str(d.value)


def _render_dim_dynamic(d: tir.Dim) -> str:
    """A runtime (`DimDyn`), symbolic (`DimVar`) or computed
    (`DimExpr`) dimension renders as MLIR's dynamic extent `?` — the
    builtin `tensor` type carries only static ints or `?`, so a
    non-constant size collapses to a dynamic dimension."""
    return "?"


# `tir.Dim` subclass -> its renderer. `_check_dim_coverage` asserts
# this covers `tir.Dim` exactly.
_DIM_RENDERERS: dict[type, Callable[[tir.Dim], str]] = {
    tir.DimConst: _render_dim_const,
    tir.DimDyn: _render_dim_dynamic,
    tir.DimVar: _render_dim_dynamic,
    tir.DimExpr: _render_dim_dynamic,
}


def render_dim(d: tir.Dim) -> str:
    """Render a tensor-shape dimension as an MLIR shape token — a
    static integer, or `?` for a dynamic / symbolic / computed size.
    Fails closed on an unknown `Dim` subclass."""
    renderer = _DIM_RENDERERS.get(type(d))
    if renderer is None:
        raise MLIRTranslationError(
            f"no MLIR shape rendering for dimension "
            f"{type(d).__name__} — the translator fails closed")
    return renderer(d)


# --------------------------------------------------------------------------
# types
# --------------------------------------------------------------------------
def _render_scalar(ty: tir.TIRType) -> str:
    """An MLIR scalar type — fails closed on a dtype with no MLIR
    spelling (`char`, the quantized dtypes, an unknown name)."""
    assert isinstance(ty, tir.TIRScalar)
    mlir = _MLIR_SCALAR_TYPES.get(ty.name)
    if mlir is None:
        raise MLIRTranslationError(
            f"scalar dtype {ty.name!r} has no MLIR type — it is "
            f"width-unpinned (`char`), a front-end-only quantized "
            f"dtype, or unknown; the translator fails closed rather "
            f"than guess a width")
    return mlir


def _render_tensor(ty: tir.TIRType) -> str:
    """`tensor<dims x dtype>` — the builtin MLIR tensor type, which
    carries shape and element type only.

    A non-default `layout` (anything but ROW_MAJOR) or `device`
    (anything but "cpu") is correctness-relevant and has NO slot in
    the builtin `tensor` type — it would be carried by an encoding
    attribute / memory space, which the translator does not yet emit
    (a Stage-213 concern). Rather than silently ERASE it — emitting a
    type that reads as the default — the translator FAILS CLOSED."""
    assert isinstance(ty, tir.TIRTensorTy)
    if ty.layout is not tir.Layout.ROW_MAJOR:
        raise MLIRTranslationError(
            f"tensor type has a non-default layout ({ty.layout.name}) "
            f"— MLIR's builtin `tensor` type cannot carry it "
            f"(encoding-attribute support is a Stage-213 concern); the "
            f"translator fails closed rather than silently erase the "
            f"layout")
    if ty.device != "cpu":
        raise MLIRTranslationError(
            f"tensor type has a non-default device ({ty.device!r}) — "
            f"MLIR's builtin `tensor` type cannot carry it; the "
            f"translator fails closed rather than silently erase it")
    elem = render_mlir_type(ty.dtype)
    dims = "".join(f"{render_dim(d)}x" for d in ty.shape)
    return f"tensor<{dims}{elem}>"


def _render_tile(ty: tir.TIRType) -> str:
    """`vector<dims x dtype>` — MLIR's tile / SIMD type. A `vector`
    requires STATIC dimensions, so a non-constant tile extent fails
    closed; `memspace` is not part of the type (a downstream concern,
    like a tensor's layout)."""
    assert isinstance(ty, tir.TIRTileTy)
    elem = render_mlir_type(ty.dtype)
    extents: list[str] = []
    for d in ty.shape:
        if not isinstance(d, tir.DimConst):
            raise MLIRTranslationError(
                f"tile type has a non-constant dimension "
                f"({type(d).__name__}) — an MLIR `vector` type requires "
                f"static dimensions; the translator fails closed")
        extents.append(_render_dim_const(d))
    if not extents:
        raise MLIRTranslationError(
            "0-dimensional tile type has no MLIR `vector` rendering — "
            "an MLIR vector has at least one dimension")
    return f"vector<{'x'.join(extents)}x{elem}>"


def _render_tuple(ty: tir.TIRType) -> str:
    """`tuple<elem, elem, ...>` — MLIR's builtin tuple type. An empty
    tuple has no MLIR tuple spelling and fails closed (a unit result
    is `TIRUnit` / `none`, not an empty tuple)."""
    assert isinstance(ty, tir.TIRTuple)
    if not ty.elems:
        raise MLIRTranslationError(
            "empty TIRTuple has no MLIR `tuple` type — a no-value "
            "result is `TIRUnit` (`none`), not an empty tuple")
    inner = ", ".join(render_mlir_type(e) for e in ty.elems)
    return f"tuple<{inner}>"


def _render_unit(ty: tir.TIRType) -> str:
    """`none` — MLIR's builtin unit type."""
    assert isinstance(ty, tir.TIRUnit)
    return "none"


# `tir.TIRType` subclass -> its renderer. Dispatch is by EXACT type
# (every `TIRType` subclass is a concrete leaf). `_check_tir_type_
# coverage` asserts this covers `tir.TIRType` exactly.
_TYPE_RENDERERS: dict[type, Callable[[tir.TIRType], str]] = {
    tir.TIRScalar: _render_scalar,
    tir.TIRTensorTy: _render_tensor,
    tir.TIRTileTy: _render_tile,
    tir.TIRTuple: _render_tuple,
    tir.TIRUnit: _render_unit,
}


def render_mlir_type(ty: tir.TIRType) -> str:
    """Render a Helix IR type as MLIR type syntax.

    Total over the concrete `tir.TIRType` subclasses — `_check_tir_
    type_coverage` (a module-load guard) pins that. Fails closed with
    `MLIRTranslationError` on any type, dtype, or dimension with no
    faithful MLIR rendering — it never emits a guessed type."""
    renderer = _TYPE_RENDERERS.get(type(ty))
    if renderer is None:
        raise MLIRTranslationError(
            f"no MLIR rendering for IR type {type(ty).__name__} — the "
            f"translator fails closed on an unknown type")
    return renderer(ty)


# --------------------------------------------------------------------------
# module-load drift guards
# --------------------------------------------------------------------------
def _check_tir_type_coverage() -> None:
    """Module-load guard: `_TYPE_RENDERERS` keys are EXACTLY the
    concrete `tir.TIRType` subclasses. A new IR type added to `tir.py`
    without a renderer fails loudly here — and `render_mlir_type`'s
    fail-closed lookup catches it at runtime too. Mirrors the
    Stage-211 module-load drift guards."""
    handled = {t.__name__ for t in _TYPE_RENDERERS}
    defined = {t.__name__ for t in tir.TIRType.__subclasses__()}
    if handled != defined:
        missing = sorted(defined - handled)
        stale = sorted(handled - defined)
        raise AssertionError(
            f"helixc.ir.mlir.emit: _TYPE_RENDERERS does not match "
            f"tir.TIRType's subclasses — unhandled type(s): "
            f"{missing or 'none'}; stale renderer(s): {stale or 'none'}")


def _check_dim_coverage() -> None:
    """Module-load guard: `_DIM_RENDERERS` keys are EXACTLY the
    concrete `tir.Dim` subclasses — a new shape-dimension kind without
    a renderer fails loudly here."""
    handled = {t.__name__ for t in _DIM_RENDERERS}
    defined = {t.__name__ for t in tir.Dim.__subclasses__()}
    if handled != defined:
        missing = sorted(defined - handled)
        stale = sorted(handled - defined)
        raise AssertionError(
            f"helixc.ir.mlir.emit: _DIM_RENDERERS does not match "
            f"tir.Dim's subclasses — unhandled dim(s): "
            f"{missing or 'none'}; stale renderer(s): {stale or 'none'}")


_check_tir_type_coverage()
_check_dim_coverage()


# --------------------------------------------------------------------------
# the module / function emitter (Stage 212 chunk B)
# --------------------------------------------------------------------------
def _value_ref(v: tile_ir.TileValue) -> str:
    """The MLIR SSA reference for a Tile-IR value — `%v<id>`.

    Every Tile-IR value — a parameter, a constant, a computed result —
    has a unique `TileValue.id`, so `%v<id>` is a uniform SSA name and
    the emitter stays STATELESS (no per-function symbol table). Unlike
    the LLVM backend, which INLINES constants and so holds no register
    for one, MLIR emits `arith.constant` ops — a constant result has
    an ordinary `%v<id>` name like any other value."""
    return f"%v{v.id}"


def _emit_return(op: tile_ir.TileOp) -> str:
    """`func.return` for a Helix function's unit (0-operand) or value
    (1-operand) return. `_check_fn_translatable` has already vetted the
    operand count against the function's declared result type, so this
    sees only 0 or 1 operand."""
    if not op.operands:
        return "func.return"
    assert len(op.operands) == 1, \
        "RETURN arity must be vetted by _check_fn_translatable"
    v = op.operands[0]
    return f"func.return {_value_ref(v)} : {render_mlir_type(v.ty)}"


# --- the scalar `arith` op emitters (Stage 212 chunk C) ---
def _single_result(op: tile_ir.TileOp, op_name: str) -> tile_ir.TileValue:
    """The one result of a single-result op — fails closed on any
    other result count."""
    if len(op.results) != 1:
        raise MLIRTranslationError(
            f"{op_name} expects exactly 1 result, got "
            f"{len(op.results)} — the translator fails closed")
    return op.results[0]


def _scalar_arith_type(ty: tir.TIRType, op_name: str) -> tuple[str, bool]:
    """`(mlir_type, is_integer)` for a scalar-arithmetic operand /
    result type. Fails closed on a non-scalar type — the chunk-C
    `arith` emitters handle scalar ops only. `is_integer` is True for
    every MLIR integer type (`i1`..`i64`), False for the floats."""
    if not isinstance(ty, tir.TIRScalar):
        raise MLIRTranslationError(
            f"{op_name}: type is {type(ty).__name__}, not a scalar — "
            f"the chunk-C `arith` emitters handle scalar ops only; the "
            f"translator fails closed")
    mlir = render_mlir_type(ty)        # fails closed on char / quantized
    return mlir, mlir.startswith("i")


def _tile_arith_type(ty: tir.TIRType, op_name: str) -> tuple[str, bool]:
    """`(mlir_type, is_integer)` for a tile-arithmetic operand /
    result type — a `TIRTileTy`, rendered to `vector<...>`. `arith`
    ops are elementwise-polymorphic over vectors, so a tile binop uses
    the same mnemonics as a scalar one; the int / float choice is the
    tile's ELEMENT dtype. Fails closed on a non-tile type."""
    if not isinstance(ty, tir.TIRTileTy):
        raise MLIRTranslationError(
            f"{op_name}: type is {type(ty).__name__}, not a tile — "
            f"the chunk-E `vector` emitters handle tile ops only; the "
            f"translator fails closed")
    mlir = render_mlir_type(ty)             # vector<...>
    element = render_mlir_type(ty.dtype)    # the element scalar type
    return mlir, element.startswith("i")


def _emit_const_int(op: tile_ir.TileOp) -> str:
    """`%vR = arith.constant <n> : <iN>` — an integer scalar constant."""
    r = _single_result(op, "scalar.const_int")
    if op.operands:
        raise MLIRTranslationError(
            "scalar.const_int takes no operands — the translator fails "
            "closed")
    value = op.attrs.get("value")
    # A Python `bool` is an `int` subclass — `type(...) is int` rejects
    # a stray bool, which would be malformed IR (a boolean constant is
    # a distinct op, not a `scalar.const_int`).
    if type(value) is not int:
        raise MLIRTranslationError(
            f"scalar.const_int has no integer `value` attribute (got "
            f"{value!r}) — the translator fails closed")
    mlir, is_int = _scalar_arith_type(r.ty, "scalar.const_int")
    if not is_int:
        raise MLIRTranslationError(
            f"scalar.const_int has a non-integer result type ({mlir}) "
            f"— the translator fails closed")
    return f"{_value_ref(r)} = arith.constant {value} : {mlir}"


def _float_literal(value: float) -> str:
    """A FINITE float as an MLIR float literal — round-trip-precise and
    always carrying a decimal point.

    MLIR's float-literal grammar requires a `.`; Python's `repr` omits
    it for round magnitudes in scientific notation (`repr(1e20)` is
    `'1e+20'` — no `.`). This inserts the `.0` and renders a plain
    decimal exponent (`int(...)` drops the `+` and any leading zeros),
    so `1e20` becomes the MLIR-valid `1.0e20`."""
    text = repr(value)
    if "e" not in text:
        return text if "." in text else text + ".0"
    mantissa, _, exponent = text.partition("e")
    if "." not in mantissa:
        mantissa += ".0"
    return f"{mantissa}e{int(exponent)}"


def _emit_const_float(op: tile_ir.TileOp) -> str:
    """`%vR = arith.constant <f> : <fN>` — a FINITE floating-point
    scalar constant. Fails closed on a non-finite value: infinity and
    NaN have no plain MLIR `arith.constant` float literal."""
    r = _single_result(op, "scalar.const_float")
    if op.operands:
        raise MLIRTranslationError(
            "scalar.const_float takes no operands — the translator "
            "fails closed")
    value = op.attrs.get("value")
    if type(value) not in (int, float):
        raise MLIRTranslationError(
            f"scalar.const_float has no numeric `value` attribute (got "
            f"{value!r}) — the translator fails closed")
    number = float(value)
    if not math.isfinite(number):
        raise MLIRTranslationError(
            f"scalar.const_float value {number!r} is not finite — "
            f"MLIR's `arith.constant` has no plain float literal for "
            f"infinity / NaN; the translator fails closed")
    mlir, is_int = _scalar_arith_type(r.ty, "scalar.const_float")
    if is_int:
        raise MLIRTranslationError(
            f"scalar.const_float has a non-float result type ({mlir}) "
            f"— the translator fails closed")
    return (f"{_value_ref(r)} = arith.constant "
            f"{_float_literal(number)} : {mlir}")


def _emit_arith_binop(
        op: tile_ir.TileOp, op_name: str, int_mnemonic: str,
        float_mnemonic: str,
        classify: Callable[[tir.TIRType, str], tuple[str, bool]],
        ) -> str:
    """A two-operand elementwise `arith` op — `%vR = <mnemonic> %vA,
    %vB : <T>`. `classify` (`_scalar_arith_type` for a scalar op,
    `_tile_arith_type` for a `vector` tile op) resolves the result
    type to its MLIR spelling and picks the integer / float mnemonic;
    `arith` ops are elementwise-polymorphic, so the same mnemonics
    serve both. The translator fails closed unless both operands and
    the result share that one type — a type-mismatched `arith` op
    would be invalid MLIR."""
    r = _single_result(op, op_name)
    if len(op.operands) != 2:
        raise MLIRTranslationError(
            f"{op_name} expects 2 operands, got {len(op.operands)} — "
            f"the translator fails closed")
    a, b = op.operands
    mlir, is_int = classify(r.ty, op_name)
    if a.ty != r.ty or b.ty != r.ty:
        raise MLIRTranslationError(
            f"{op_name}: operand and result types are not all equal — "
            f"a type-mismatched `arith` op is invalid MLIR; the "
            f"translator fails closed")
    mnemonic = int_mnemonic if is_int else float_mnemonic
    return (f"{_value_ref(r)} = {mnemonic} {_value_ref(a)}, "
            f"{_value_ref(b)} : {mlir}")


def _emit_add(op: tile_ir.TileOp) -> str:
    return _emit_arith_binop(op, "scalar.add", "arith.addi",
                             "arith.addf", _scalar_arith_type)


def _emit_sub(op: tile_ir.TileOp) -> str:
    return _emit_arith_binop(op, "scalar.sub", "arith.subi",
                             "arith.subf", _scalar_arith_type)


def _emit_mul(op: tile_ir.TileOp) -> str:
    return _emit_arith_binop(op, "scalar.mul", "arith.muli",
                             "arith.mulf", _scalar_arith_type)


# --- the compare / select op emitters (Stage 212 chunk D) ---
# An `arith.cmpi` predicate is sign-sensitive for the ORDERED
# comparisons: a Tensor-IR `cmp.lt` is `slt` on a signed operand,
# `ult` on an unsigned one. `cmp.eq` / `cmp.ne` are sign-agnostic.
_CMPI_PREDICATES: dict[str, tuple[str, str]] = {
    # `cmp` attr -> (signed predicate, unsigned predicate)
    "cmp.eq": ("eq", "eq"),
    "cmp.ne": ("ne", "ne"),
    "cmp.lt": ("slt", "ult"),
    "cmp.le": ("sle", "ule"),
    "cmp.gt": ("sgt", "ugt"),
    "cmp.ge": ("sge", "uge"),
}
# An `arith.cmpf` predicate. `==` and the relational comparisons are
# ORDERED (`o*` — false if either operand is NaN); `!=` is UNORDERED-
# not-equal (`une` — true when the operands are unordered, so
# `NaN != NaN` is true). This matches Helix's float surface semantics:
# the x86_64 backend's reference makes float `CMP_NE` "not-equal OR
# unordered" (it is the logical negation of the ordered `==`).
_CMPF_PREDICATES: dict[str, str] = {
    "cmp.eq": "oeq", "cmp.ne": "une",
    "cmp.lt": "olt", "cmp.le": "ole",
    "cmp.gt": "ogt", "cmp.ge": "oge",
}
# Helix unsigned integer dtypes — they select the unsigned `arith.cmpi`
# predicate. MLIR integer types are SIGNLESS, so signedness is read
# from the Helix dtype name, not the rendered MLIR type. Mirrors
# `llvm_ir._UNSIGNED_INT_DTYPES`.
_UNSIGNED_DTYPES: frozenset[str] = frozenset({
    "u8", "u16", "u32", "u64", "usize",
})


def _check_cmp_predicate_tables() -> None:
    """Module-load guard: the `arith.cmpi` / `arith.cmpf` predicate
    tables cover EXACTLY the six Tensor-IR comparison kinds — the
    `cmp.*` strings the Tile-IR lowerer tags `SCALAR_CMP` ops with. A
    new comparison op in `tir.py`, or a key present in one table but
    not the other, fails loudly here."""
    expected = {k.value for k in tir.OpKind if k.name.startswith("CMP_")}
    for name, keys in (("_CMPI_PREDICATES", set(_CMPI_PREDICATES)),
                       ("_CMPF_PREDICATES", set(_CMPF_PREDICATES))):
        if keys != expected:
            raise AssertionError(
                f"helixc.ir.mlir.emit: {name} keys {sorted(keys)} do "
                f"not match the Tensor-IR comparison kinds "
                f"{sorted(expected)}")


_check_cmp_predicate_tables()


def _emit_cmp(op: tile_ir.TileOp) -> str:
    """`%vR = arith.cmpi <pred>, %vA, %vB : <T>` — or `arith.cmpf` for
    a float operand. A scalar comparison; the result is an `i1`. The
    predicate comes from the Tile-IR `cmp` attribute; for an integer
    ordered comparison it is signed / unsigned by the operand dtype."""
    r = _single_result(op, "scalar.cmp")
    if len(op.operands) != 2:
        raise MLIRTranslationError(
            f"scalar.cmp expects 2 operands, got {len(op.operands)} — "
            f"the translator fails closed")
    a, b = op.operands
    if a.ty != b.ty:
        raise MLIRTranslationError(
            "scalar.cmp operands have different types — a "
            "type-mismatched comparison is invalid MLIR; the "
            "translator fails closed")
    if not isinstance(a.ty, tir.TIRScalar):
        raise MLIRTranslationError(
            f"scalar.cmp operand type is {type(a.ty).__name__}, not a "
            f"scalar — the translator fails closed")
    if render_mlir_type(r.ty) != "i1":
        raise MLIRTranslationError(
            "scalar.cmp result type is not i1 — a comparison produces "
            "a boolean; the translator fails closed")
    cmp = op.attrs.get("cmp")
    operand_mlir = render_mlir_type(a.ty)   # fails closed on char etc.
    # `_check_cmp_predicate_tables` guarantees both predicate tables
    # carry the same key set, so one membership check covers both.
    if cmp not in _CMPF_PREDICATES:
        raise MLIRTranslationError(
            f"scalar.cmp has no recognised `cmp` attribute (got "
            f"{cmp!r}) — the translator fails closed")
    if operand_mlir.startswith("i"):
        signed, unsigned = _CMPI_PREDICATES[cmp]
        predicate = unsigned if a.ty.name in _UNSIGNED_DTYPES else signed
        mnemonic = "arith.cmpi"
    else:
        predicate = _CMPF_PREDICATES[cmp]
        mnemonic = "arith.cmpf"
    return (f"{_value_ref(r)} = {mnemonic} {predicate}, "
            f"{_value_ref(a)}, {_value_ref(b)} : {operand_mlir}")


def _emit_select(op: tile_ir.TileOp) -> str:
    """`%vR = arith.select %vCond, %vA, %vB : <T>` — a ternary select.
    The condition is an `i1`; the two arms and the result share one
    type."""
    r = _single_result(op, "scalar.select")
    if len(op.operands) != 3:
        raise MLIRTranslationError(
            f"scalar.select expects 3 operands (cond, a, b), got "
            f"{len(op.operands)} — the translator fails closed")
    cond, a, b = op.operands
    if render_mlir_type(cond.ty) != "i1":
        raise MLIRTranslationError(
            "scalar.select condition type is not i1 — the condition "
            "must be a boolean; the translator fails closed")
    if a.ty != r.ty or b.ty != r.ty:
        raise MLIRTranslationError(
            "scalar.select: the two arms and the result do not all "
            "share one type — the translator fails closed")
    return (f"{_value_ref(r)} = arith.select {_value_ref(cond)}, "
            f"{_value_ref(a)}, {_value_ref(b)} : {render_mlir_type(r.ty)}")


# --- the elementwise `vector` tile-op emitters (Stage 212 chunk E) ---
def _emit_tile_add(op: tile_ir.TileOp) -> str:
    return _emit_arith_binop(op, "tile.add", "arith.addi",
                             "arith.addf", _tile_arith_type)


def _emit_tile_sub(op: tile_ir.TileOp) -> str:
    return _emit_arith_binop(op, "tile.sub", "arith.subi",
                             "arith.subf", _tile_arith_type)


def _emit_tile_mul(op: tile_ir.TileOp) -> str:
    return _emit_arith_binop(op, "tile.mul", "arith.muli",
                             "arith.mulf", _tile_arith_type)


def _emit_tile_zeros(op: tile_ir.TileOp) -> str:
    """`%vR = arith.constant dense<0> : vector<...>` — a zero-filled
    tile, a splat constant. Takes no operands; the element kind picks
    the `0` / `0.0` splat literal."""
    r = _single_result(op, "tile.zeros")
    if op.operands:
        raise MLIRTranslationError(
            "tile.zeros takes no operands — the translator fails "
            "closed")
    mlir, is_int = _tile_arith_type(r.ty, "tile.zeros")
    zero = "0" if is_int else "0.0"
    return f"{_value_ref(r)} = arith.constant dense<{zero}> : {mlir}"


# Tile-IR op kind -> its MLIR emitter. DELIBERATELY PARTIAL — chunks
# B-E emit the function terminator, the scalar `arith` core, compare /
# select, and the elementwise `vector` tile ops; the remaining per-op
# emitters (`scalar.neg`, the non-elementwise tile ops — matmul /
# reduce / transpose / reshape / const, `memref` / `gpu`, `helix`) are
# added in later chunks. `_emit_op` FAILS CLOSED on any op kind absent
# here, so the partial table is safe — never a silent miss. No
# completeness guard, deliberately: the table is MEANT to be
# incomplete until the per-op chunks land.
_OP_EMITTERS: dict[tile_ir.TileOpKind,
                   Callable[[tile_ir.TileOp], str]] = {
    tile_ir.TileOpKind.RETURN: _emit_return,
    tile_ir.TileOpKind.SCALAR_CONST_INT: _emit_const_int,
    tile_ir.TileOpKind.SCALAR_CONST_FLOAT: _emit_const_float,
    tile_ir.TileOpKind.SCALAR_ADD: _emit_add,
    tile_ir.TileOpKind.SCALAR_SUB: _emit_sub,
    tile_ir.TileOpKind.SCALAR_MUL: _emit_mul,
    tile_ir.TileOpKind.SCALAR_CMP: _emit_cmp,
    tile_ir.TileOpKind.SCALAR_SELECT: _emit_select,
    tile_ir.TileOpKind.TILE_ADD: _emit_tile_add,
    tile_ir.TileOpKind.TILE_SUB: _emit_tile_sub,
    tile_ir.TileOpKind.TILE_MUL: _emit_tile_mul,
    tile_ir.TileOpKind.TILE_ZEROS: _emit_tile_zeros,
}


def _emit_op(op: tile_ir.TileOp) -> str:
    """Emit one Tile-IR op as an MLIR op line. Fails closed on any op
    kind without an emitter yet — it never emits a guessed op."""
    emitter = _OP_EMITTERS.get(op.kind)
    if emitter is None:
        raise MLIRTranslationError(
            f"Stage 212 does not yet emit the Tile-IR op "
            f"{op.kind.name} ({op.kind.value!r}) — the per-op emitters "
            f"are added chunk by chunk; the translator fails closed "
            f"rather than emit a guessed op")
    return emitter(op)


def _check_fn_translatable(fn: tile_ir.TileFn) -> None:
    """Fail closed unless `fn` is a Tile-IR function chunk B can
    faithfully translate. Chunk B emits SINGLE-BLOCK functions with an
    internally consistent signature; anything else has no faithful
    chunk-B MLIR and raises rather than emit wrong text:

    - a name with no plain MLIR symbol spelling;
    - zero blocks (a `func.func` body needs one) or MULTI-block (the
      `cf.br` / `^bb` CFG machinery is a later chunk — emitting `^bb`
      blocks with no branch to reach them would be invalid MLIR);
    - an entry block whose parameters diverge from the signature
      parameters (would emit undefined SSA references);
    - a `return` inconsistent with the declared result type (wrong
      operand count, or an operand whose type is not the result type).
    """
    if not fn.name.isidentifier():
        raise MLIRTranslationError(
            f"function name {fn.name!r} is not an identifier — it has "
            f"no plain MLIR symbol spelling; the translator fails "
            f"closed")
    if not fn.blocks:
        raise MLIRTranslationError(
            f"function {fn.name!r} has no blocks — an MLIR `func.func` "
            f"body needs at least one block")
    if len(fn.blocks) > 1:
        raise MLIRTranslationError(
            f"function {fn.name!r} has {len(fn.blocks)} blocks — Stage "
            f"212 chunk B emits only single-block functions; "
            f"multi-block CFG (`cf.br` / `^bb` labels) is a later "
            f"chunk. The translator fails closed")
    entry = fn.entry
    sig_ids = [p.id for p in fn.params]
    entry_ids = [p.id for p in entry.params]
    if entry_ids != sig_ids:
        raise MLIRTranslationError(
            f"function {fn.name!r}: the entry block's parameter ids "
            f"{entry_ids} differ from the signature's {sig_ids} — the "
            f"entry block's arguments ARE the function parameters; the "
            f"translator fails closed rather than emit undefined SSA "
            f"references")
    returns_unit = isinstance(fn.return_ty, tir.TIRUnit)
    for op in entry.ops:
        if op.kind is not tile_ir.TileOpKind.RETURN:
            continue
        if returns_unit and op.operands:
            raise MLIRTranslationError(
                f"function {fn.name!r} returns unit but a `return` "
                f"carries {len(op.operands)} operand(s) — the "
                f"translator fails closed")
        if not returns_unit and len(op.operands) != 1:
            raise MLIRTranslationError(
                f"function {fn.name!r} returns a value but a `return` "
                f"carries {len(op.operands)} operand(s), not 1 — the "
                f"translator fails closed")
        if not returns_unit and op.operands[0].ty != fn.return_ty:
            raise MLIRTranslationError(
                f"function {fn.name!r}: a `return` operand's type does "
                f"not match the declared result type — the translator "
                f"fails closed rather than emit a type-mismatched "
                f"`func.return`")


def _emit_fn(fn: tile_ir.TileFn) -> list[str]:
    """Emit a Tile-IR function as MLIR `func.func` lines (unindented).
    `_check_fn_translatable` fail-closed-vets `fn` first, so this only
    ever emits a single-block, signature-consistent function."""
    _check_fn_translatable(fn)
    params = ", ".join(
        f"{_value_ref(p)}: {render_mlir_type(p.ty)}" for p in fn.params)
    # A `TIRUnit` return is a VOID function — MLIR writes no `->` at
    # all (not `-> none`).
    if isinstance(fn.return_ty, tir.TIRUnit):
        ret = ""
    else:
        ret = f" -> {render_mlir_type(fn.return_ty)}"
    lines = [f"func.func @{fn.name}({params}){ret} {{"]
    for op in fn.entry.ops:
        lines.append(f"  {_emit_op(op)}")
    lines.append("}")
    return lines


def emit_mlir_module(module: tile_ir.TileModule) -> str:
    """Translate a Tile-IR module to MLIR textual IR — a
    `module { ... }` string. The Stage-212 entry point.

    Walks `module.functions` and emits a `func.func` for each. Pure
    text — never `import mlir`. The emitted text is structurally
    checkable by `validate.mock_validate_mlir`; real `mlir-opt`
    validation is a binding-gated Stage-212+ concern.

    FAILS CLOSED (`MLIRTranslationError`) on any construct it cannot
    yet faithfully emit — it never produces a guessed or wrong
    module."""
    lines: list[str] = ["module {"]
    for fn in module.functions.values():
        for line in _emit_fn(fn):
            lines.append(f"  {line}")
    lines.append("}")
    return "\n".join(lines) + "\n"
