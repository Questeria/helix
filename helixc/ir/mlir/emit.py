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

    Chunk B references only function parameters, whose SSA name IS
    `%v<id>`. Chunk C (constants, computed results) will need a
    stateful name map — a folded constant has no `%v<id>` register —
    at which point the emitter becomes stateful (cf. the sibling
    `llvm_ir._FnEmitter`'s `operand` map). This pure id->name function
    is the chunk-B simplification, valid while only params are named."""
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


# Tile-IR op kind -> its MLIR emitter. DELIBERATELY PARTIAL — chunk B
# emits only the function terminator; the per-op emitters (arith /
# vector / memref / gpu / helix) are added in later chunks. `_emit_op`
# FAILS CLOSED on any op kind absent here, so the partial table is
# safe — never a silent miss. No completeness guard, deliberately: the
# table is MEANT to be incomplete until the per-op chunks land.
_OP_EMITTERS: dict[tile_ir.TileOpKind,
                   Callable[[tile_ir.TileOp], str]] = {
    tile_ir.TileOpKind.RETURN: _emit_return,
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
