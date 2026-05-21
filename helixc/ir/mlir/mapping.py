"""
helixc/ir/mlir/mapping.py — Helix-op -> MLIR-lowering mapping
(v3.0 Phase E, Stage 211 chunks B-C).

The mapping substrate for the MLIR migration: for every Helix IR
operation — both Tensor-IR `tir.OpKind` (chunk B) and Tile-IR
`tile_ir.TileOpKind` (chunk C) — which MLIR lowering target it belongs
to, per the ratified Stage 210 HYBRID decision
(docs/V3_STAGE210_MLIR_DECISION.md section 2):

- an UPSTREAM MLIR dialect (`arith` / `math` / `linalg` / `vector` /
  `tensor` / `memref` / `func` / `cf` / `gpu`) — the ~80-85% numerical
  / structural op core;
- the custom `helix` dialect — the Helix-specific ops with no faithful
  upstream home (the `grad`/`jvp`/`vmap` transforms, the `agi.*`
  metaprogramming ops, the atomic arena allocator);
- RESIDUAL — ops whose MLIR home the decision record explicitly
  deferred ("flag for review": the `Result` and quantize encodings,
  and the Tile-IR async memory ops — TMA / barrier — whose `nvgpu`-vs-
  `helix`-async representation is an open Stage-213 question), so the
  mapping records "not decided" honestly rather than asserting a
  placement the decision did not make.

This is the COARSE, type-independent classification — the indicative
lowering target per op-KIND. Two refinements are deliberately left to
Stage 212 (the tile-IR -> MLIR translation), not encoded here:
- Operand-type dependence. The scalar arithmetic / comparison /
  select ops (`ADD`, `CMP_*`, `SELECT`, ...) lower to `arith` on
  scalars but to `linalg` (an elementwise `linalg.generic` / named
  op) on tensors. The mapping records `ARITH` — the scalar/primary
  home — because the category (`UPSTREAM`) is correct either way; the
  exact dialect+op is a per-translation decision.
- The exact MLIR op mnemonic (`arith.addi` vs `arith.addf`, etc.).

MOCK-PATH-FIRST: this module is pure data — it imports the home-grown
IR (`helixc.ir.tir` and `helixc.ir.tile_ir`, no MLIR dependency) and
NEVER `import mlir`. It is fully usable on a machine with no MLIR
bindings.

License: Apache 2.0
"""

from __future__ import annotations

from enum import Enum

from .. import tile_ir, tir


class MLIRLowering(Enum):
    """The MLIR lowering target of a Helix IR operation — an upstream
    MLIR dialect, the custom `helix` dialect, or RESIDUAL (no target
    decided yet).

    The nine upstream members name the MLIR dialects the Stage 210
    hybrid decision maps the numerical / structural op core onto.
    `HELIX` is the small custom dialect for the Helix-specific ops.
    `RESIDUAL` is for ops the decision record explicitly flagged for
    later review — it is an honest "undecided", not a silent default.
    """
    # --- upstream MLIR dialects ---
    ARITH = "arith"       # scalar integer / float arithmetic, compare,
                          # select, cast, constants
    MATH = "math"         # transcendentals + elementwise activations
    LINALG = "linalg"     # matmul, conv, reduce, fill, transpose
    VECTOR = "vector"     # tile / SIMD compute — the Tile-IR op layer
    TENSOR = "tensor"     # shape ops: reshape, broadcast, slice, concat
    MEMREF = "memref"     # locals, stack arrays, tile memory movement
    FUNC = "func"         # calls, return, effectful runtime ops
    CF = "cf"             # unstructured control flow: br / cond_br
    GPU = "gpu"           # thread / tile-index ops
    # --- the custom Helix dialect ---
    HELIX = "helix"       # Helix-specific ops with no upstream home
    # --- not yet placed ---
    RESIDUAL = "residual"  # MLIR home deferred (decision-record review)


# The `MLIRLowering` members that name an UPSTREAM MLIR dialect — as
# opposed to the custom `helix` dialect or the deferred RESIDUAL
# bucket. `_check_lowering_partition` (below) asserts these three
# categories partition `MLIRLowering`.
_UPSTREAM_LOWERINGS: frozenset[MLIRLowering] = frozenset({
    MLIRLowering.ARITH, MLIRLowering.MATH, MLIRLowering.LINALG,
    MLIRLowering.VECTOR, MLIRLowering.TENSOR, MLIRLowering.MEMREF,
    MLIRLowering.FUNC, MLIRLowering.CF, MLIRLowering.GPU,
})


def _check_lowering_partition() -> None:
    """Module-load guard: every `MLIRLowering` member is exactly one of
    upstream / `HELIX` / `RESIDUAL`. A new member added without
    classifying it would silently default to non-upstream in
    `is_upstream` — fail loudly here instead. Mirrors
    `llvm_parity._check_parity_verdict_coverage`."""
    classified = _UPSTREAM_LOWERINGS | {MLIRLowering.HELIX,
                                        MLIRLowering.RESIDUAL}
    if classified != set(MLIRLowering):
        raise AssertionError(
            f"helixc.ir.mlir.mapping: the upstream / HELIX / RESIDUAL "
            f"split classifies {classified} but MLIRLowering has "
            f"{set(MLIRLowering)} — every member must be classified")
    if MLIRLowering.HELIX in _UPSTREAM_LOWERINGS or (
            MLIRLowering.RESIDUAL in _UPSTREAM_LOWERINGS):
        raise AssertionError(
            "helixc.ir.mlir.mapping: HELIX / RESIDUAL must not be in "
            "_UPSTREAM_LOWERINGS — they are not upstream dialects")


_check_lowering_partition()


# The per-`OpKind` MLIR lowering target. In Tensor-IR enum order so it
# reads alongside `tir.OpKind`; `_check_opkind_coverage` asserts it
# covers the enum exactly. See the module docstring for the
# type-dependence caveat on the `arith`-mapped arithmetic ops.
_OPKIND_LOWERING: dict[tir.OpKind, MLIRLowering] = {
    # constants
    tir.OpKind.CONST_INT: MLIRLowering.ARITH,
    tir.OpKind.CONST_FLOAT: MLIRLowering.ARITH,
    tir.OpKind.CONST_BOOL: MLIRLowering.ARITH,
    tir.OpKind.CONST_TENSOR: MLIRLowering.ARITH,
    # tensor creation
    tir.OpKind.TENSOR_ZEROS: MLIRLowering.LINALG,
    tir.OpKind.TENSOR_ONES: MLIRLowering.LINALG,
    tir.OpKind.TENSOR_FULL: MLIRLowering.LINALG,
    # `tensor.empty` + `linalg.fill` — the fill value comes from a
    # runtime RNG call (decision record section 2.2: TENSOR_ZEROS/ONES/
    # FULL/RAND share this row). The runtime call is a lowering detail;
    # the dominant structural op is `linalg.fill`.
    tir.OpKind.TENSOR_RAND: MLIRLowering.LINALG,
    # external host / file I/O — effectful, lower to runtime func.call
    tir.OpKind.TENSOR_LOAD: MLIRLowering.FUNC,
    tir.OpKind.TENSOR_STORE: MLIRLowering.FUNC,
    # scalar / elementwise arithmetic
    tir.OpKind.ADD: MLIRLowering.ARITH,
    tir.OpKind.SUB: MLIRLowering.ARITH,
    tir.OpKind.MUL: MLIRLowering.ARITH,
    tir.OpKind.DIV: MLIRLowering.ARITH,
    tir.OpKind.MOD: MLIRLowering.ARITH,
    tir.OpKind.MAXIMUM: MLIRLowering.ARITH,
    tir.OpKind.MINIMUM: MLIRLowering.ARITH,
    tir.OpKind.POW: MLIRLowering.MATH,
    # bitwise
    tir.OpKind.BIT_AND: MLIRLowering.ARITH,
    tir.OpKind.BIT_OR: MLIRLowering.ARITH,
    tir.OpKind.BIT_XOR: MLIRLowering.ARITH,
    tir.OpKind.SHL: MLIRLowering.ARITH,
    tir.OpKind.SHR: MLIRLowering.ARITH,
    tir.OpKind.BIT_NOT: MLIRLowering.ARITH,
    tir.OpKind.NEG: MLIRLowering.ARITH,
    tir.OpKind.ABS: MLIRLowering.ARITH,
    # transcendentals
    tir.OpKind.EXP: MLIRLowering.MATH,
    tir.OpKind.LOG: MLIRLowering.MATH,
    tir.OpKind.SQRT: MLIRLowering.MATH,
    tir.OpKind.RECIP: MLIRLowering.MATH,
    # activations — decompose into math + arith
    tir.OpKind.RELU: MLIRLowering.MATH,
    tir.OpKind.GELU: MLIRLowering.MATH,
    tir.OpKind.SILU: MLIRLowering.MATH,
    tir.OpKind.TANH: MLIRLowering.MATH,
    tir.OpKind.SIGMOID: MLIRLowering.MATH,
    # reductions
    tir.OpKind.REDUCE_SUM: MLIRLowering.LINALG,
    tir.OpKind.REDUCE_MEAN: MLIRLowering.LINALG,
    tir.OpKind.REDUCE_MAX: MLIRLowering.LINALG,
    tir.OpKind.REDUCE_MIN: MLIRLowering.LINALG,
    tir.OpKind.REDUCE_PROD: MLIRLowering.LINALG,
    # tensor contraction / convolution
    tir.OpKind.MATMUL: MLIRLowering.LINALG,
    tir.OpKind.CONV1D: MLIRLowering.LINALG,
    tir.OpKind.CONV2D: MLIRLowering.LINALG,
    # shape ops
    tir.OpKind.RESHAPE: MLIRLowering.TENSOR,
    tir.OpKind.TRANSPOSE: MLIRLowering.LINALG,   # linalg.transpose
    tir.OpKind.BROADCAST: MLIRLowering.TENSOR,
    tir.OpKind.SLICE: MLIRLowering.TENSOR,
    tir.OpKind.CONCAT: MLIRLowering.TENSOR,
    # casts
    tir.OpKind.CAST: MLIRLowering.ARITH,
    tir.OpKind.BITCAST: MLIRLowering.ARITH,
    # quantize — decision record section 2.4 flags these for review
    # (custom `helix` op vs. upstream `quant`); recorded RESIDUAL.
    tir.OpKind.QUANTIZE: MLIRLowering.RESIDUAL,
    tir.OpKind.DEQUANTIZE: MLIRLowering.RESIDUAL,
    # select / where — both lower to `arith.select` (decision record
    # section 2.2: "...SELECT, WHERE | ... arith.select"); on tensors it
    # is `arith.select` with tensor-typed operands.
    tir.OpKind.SELECT: MLIRLowering.ARITH,
    tir.OpKind.WHERE: MLIRLowering.ARITH,
    # comparisons
    tir.OpKind.CMP_EQ: MLIRLowering.ARITH,
    tir.OpKind.CMP_NE: MLIRLowering.ARITH,
    tir.OpKind.CMP_LT: MLIRLowering.ARITH,
    tir.OpKind.CMP_LE: MLIRLowering.ARITH,
    tir.OpKind.CMP_GT: MLIRLowering.ARITH,
    tir.OpKind.CMP_GE: MLIRLowering.ARITH,
    # compositional transforms — Helix-specific, the `helix` dialect
    tir.OpKind.GRAD: MLIRLowering.HELIX,
    tir.OpKind.JVP: MLIRLowering.HELIX,
    tir.OpKind.VMAP: MLIRLowering.HELIX,
    # control flow
    tir.OpKind.CALL: MLIRLowering.FUNC,
    tir.OpKind.BR: MLIRLowering.CF,
    tir.OpKind.COND_BR: MLIRLowering.CF,
    tir.OpKind.RETURN: MLIRLowering.FUNC,
    # mutable locals + stack arrays
    tir.OpKind.ALLOC_VAR: MLIRLowering.MEMREF,
    tir.OpKind.LOAD_VAR: MLIRLowering.MEMREF,
    tir.OpKind.STORE_VAR: MLIRLowering.MEMREF,
    tir.OpKind.ALLOC_ARRAY: MLIRLowering.MEMREF,
    tir.OpKind.LOAD_ELEM: MLIRLowering.MEMREF,
    tir.OpKind.STORE_ELEM: MLIRLowering.MEMREF,
    # AGI metaprogramming — Helix-specific, the `helix` dialect
    tir.OpKind.QUOTE: MLIRLowering.HELIX,
    tir.OpKind.SPLICE: MLIRLowering.HELIX,
    tir.OpKind.MODIFY: MLIRLowering.HELIX,
    tir.OpKind.REFLECT_HASH: MLIRLowering.HELIX,
    # the atomic arena allocator — Helix-specific, the `helix` dialect
    tir.OpKind.ARENA_PUSH: MLIRLowering.HELIX,
    tir.OpKind.ARENA_GET: MLIRLowering.HELIX,
    tir.OpKind.ARENA_SET: MLIRLowering.HELIX,
    tir.OpKind.ARENA_LEN: MLIRLowering.HELIX,
    tir.OpKind.ARENA_PUSH_PAIR: MLIRLowering.HELIX,
    tir.OpKind.ARENA_PUSH_TRIPLE: MLIRLowering.HELIX,
    # string-literal access + FFI — effectful runtime, func.call
    tir.OpKind.STR_BYTE: MLIRLowering.FUNC,
    tir.OpKind.STR_PTR: MLIRLowering.FUNC,
    tir.OpKind.FFI_CALL: MLIRLowering.FUNC,
    # GPU thread / tile index
    tir.OpKind.THREAD_IDX: MLIRLowering.GPU,
    tir.OpKind.TILE_INDEX_LOAD: MLIRLowering.GPU,
    tir.OpKind.TILE_INDEX_STORE: MLIRLowering.GPU,
    # effectful runtime ops — func.call of runtime symbols
    tir.OpKind.PRINT: MLIRLowering.FUNC,
    tir.OpKind.TRAP: MLIRLowering.FUNC,
    # Result<T,E> packed-tag encoding — decision record section 2.4
    # flags these for review (custom `helix` op vs. upstream `arith`
    # bit-twiddling); recorded RESIDUAL.
    tir.OpKind.RESULT_PACK: MLIRLowering.RESIDUAL,
    tir.OpKind.RESULT_TAG: MLIRLowering.RESIDUAL,
    tir.OpKind.RESULT_PAYLOAD: MLIRLowering.RESIDUAL,
    # `@trace` ring-buffer ops — effectful runtime, func.call
    tir.OpKind.TRACE_ENTRY: MLIRLowering.FUNC,
    tir.OpKind.TRACE_EXIT: MLIRLowering.FUNC,
}


def _check_mapping_coverage(table: dict, enum_cls: type[Enum],
                            table_name: str, enum_name: str) -> None:
    """Generic module-load drift guard: `table` maps EXACTLY the
    members of `enum_cls` — no member unmapped, no stale key. An
    op-kind added to the IR without a mapping entry would otherwise be
    silently invisible to the Stage-212 MLIR translation; fail loudly
    here instead. (Stale keys are reported via `getattr(.., 'name')`
    so a non-enum key yields a diagnostic, not an `AttributeError`.)"""
    mapped = set(table)
    members = set(enum_cls)
    if mapped != members:
        missing = sorted(m.name for m in members - mapped)
        stale = sorted(getattr(m, "name", repr(m))
                       for m in mapped - members)
        raise AssertionError(
            f"helixc.ir.mlir.mapping: {table_name} does not match "
            f"{enum_name} — unmapped op(s): {missing or 'none'}; stale "
            f"key(s): {stale or 'none'}")


def _check_opkind_coverage() -> None:
    """Module-load guard: `_OPKIND_LOWERING` maps `tir.OpKind`
    exactly."""
    _check_mapping_coverage(_OPKIND_LOWERING, tir.OpKind,
                            "_OPKIND_LOWERING", "tir.OpKind")


def _check_tileopkind_coverage() -> None:
    """Module-load guard: `_TILEOPKIND_LOWERING` maps
    `tile_ir.TileOpKind` exactly."""
    _check_mapping_coverage(_TILEOPKIND_LOWERING, tile_ir.TileOpKind,
                            "_TILEOPKIND_LOWERING",
                            "tile_ir.TileOpKind")


_check_opkind_coverage()


# The per-`TileOpKind` MLIR lowering target — the Tile IR op set (the
# mid-level tiled-GPU IR). In `tile_ir.TileOpKind` enum order;
# `_check_tileopkind_coverage` asserts it covers the enum exactly. Per
# the Stage 210 decision record section 2.2 Tile-IR table: tile
# compute / creation / layout ops are MLIR's `vector` dialect (its
# tile / SIMD layer); tile memory movement is `memref` (with `gpu`
# address spaces); the carried-through scalar ops are `arith` / `func`;
# the GPU primitives are `gpu`. The async memory ops (TMA / barrier)
# are RESIDUAL: decision-record section 3 and the section-5 checklist
# explicitly defer whether they target `nvgpu` (NVIDIA-only) or a
# cross-backend `helix` async abstraction — an open Stage-213 question.
_TILEOPKIND_LOWERING: dict[tile_ir.TileOpKind, MLIRLowering] = {
    # tile creation
    tile_ir.TileOpKind.TILE_ZEROS: MLIRLowering.VECTOR,
    tile_ir.TileOpKind.TILE_CONST: MLIRLowering.VECTOR,
    # tile memory movement — `memref` load/store across HBM / SMEM /
    # REG, the address space carried as a `gpu` memory-space attribute.
    tile_ir.TileOpKind.TILE_LOAD_GLOBAL: MLIRLowering.MEMREF,
    tile_ir.TileOpKind.TILE_STORE_GLOBAL: MLIRLowering.MEMREF,
    tile_ir.TileOpKind.TILE_LOAD_SHARED: MLIRLowering.MEMREF,
    tile_ir.TileOpKind.TILE_STORE_SHARED: MLIRLowering.MEMREF,
    # async memory — TMA / barrier. RESIDUAL: `nvgpu`-only vs. a
    # cross-backend `helix` async abstraction is a deferred Stage-213
    # decision (decision record section 3 / the section-5 checklist).
    tile_ir.TileOpKind.TMA_LOAD: MLIRLowering.RESIDUAL,
    tile_ir.TileOpKind.TMA_STORE: MLIRLowering.RESIDUAL,
    tile_ir.TileOpKind.BARRIER_WAIT: MLIRLowering.RESIDUAL,
    # tile compute — `vector` elementwise arithmetic
    tile_ir.TileOpKind.TILE_ADD: MLIRLowering.VECTOR,
    tile_ir.TileOpKind.TILE_SUB: MLIRLowering.VECTOR,
    tile_ir.TileOpKind.TILE_MUL: MLIRLowering.VECTOR,
    # accumulating tile matmul — `vector.contract`
    tile_ir.TileOpKind.TILE_MATMUL: MLIRLowering.VECTOR,
    # axis reduce within a tile — `vector.multi_reduction`
    tile_ir.TileOpKind.TILE_REDUCE: MLIRLowering.VECTOR,
    # tile layout transforms — `vector.transpose` / `vector.shape_cast`
    tile_ir.TileOpKind.TILE_TRANSPOSE: MLIRLowering.VECTOR,
    tile_ir.TileOpKind.TILE_RESHAPE: MLIRLowering.VECTOR,
    # scalar ops carried through from Tensor IR — `arith`
    tile_ir.TileOpKind.SCALAR_CONST_INT: MLIRLowering.ARITH,
    tile_ir.TileOpKind.SCALAR_CONST_FLOAT: MLIRLowering.ARITH,
    tile_ir.TileOpKind.SCALAR_ADD: MLIRLowering.ARITH,
    tile_ir.TileOpKind.SCALAR_SUB: MLIRLowering.ARITH,
    tile_ir.TileOpKind.SCALAR_MUL: MLIRLowering.ARITH,
    tile_ir.TileOpKind.SCALAR_NEG: MLIRLowering.ARITH,
    tile_ir.TileOpKind.SCALAR_CMP: MLIRLowering.ARITH,
    tile_ir.TileOpKind.SCALAR_SELECT: MLIRLowering.ARITH,
    # call / return — `func`
    tile_ir.TileOpKind.CALL: MLIRLowering.FUNC,
    tile_ir.TileOpKind.RETURN: MLIRLowering.FUNC,
    # GPU primitives — `gpu` thread / HBM tile-index ops
    tile_ir.TileOpKind.THREAD_IDX: MLIRLowering.GPU,
    tile_ir.TileOpKind.TILE_INDEX_LOAD_HBM: MLIRLowering.GPU,
    tile_ir.TileOpKind.TILE_INDEX_STORE_HBM: MLIRLowering.GPU,
}


_check_tileopkind_coverage()


def mlir_lowering_for(op: tir.OpKind) -> MLIRLowering:
    """The MLIR lowering target of a Tensor-IR op kind. Total over
    `tir.OpKind` — `_check_opkind_coverage` guarantees every member is
    mapped, so this never raises a `KeyError`."""
    return _OPKIND_LOWERING[op]


def mlir_lowering_for_tile(op: tile_ir.TileOpKind) -> MLIRLowering:
    """The MLIR lowering target of a Tile-IR op kind. Total over
    `tile_ir.TileOpKind` — `_check_tileopkind_coverage` guarantees
    every member is mapped, so this never raises a `KeyError`."""
    return _TILEOPKIND_LOWERING[op]


def is_upstream(lowering: MLIRLowering) -> bool:
    """True iff `lowering` names an upstream MLIR dialect — i.e. NOT
    the custom `helix` dialect and NOT the deferred RESIDUAL bucket."""
    return lowering in _UPSTREAM_LOWERINGS


def dialect_name(lowering: MLIRLowering) -> str:
    """The MLIR dialect mnemonic `lowering` names — the leading token of
    a dialect-qualified op (`arith` in `arith.addi`, `helix` in
    `helix.grad`).

    Defined for the eight upstream dialects and the custom `helix`
    dialect — nine real dialects. RESIDUAL names NO dialect: its MLIR
    home is an undecided, decision-record-deferred question (section
    2.4, "flag for review"). So `dialect_name` RAISES `ValueError` for
    RESIDUAL rather than returning the bare enum value `"residual"`,
    which reads like a dialect mnemonic but is not one. A Stage-212
    translation caller must branch on RESIDUAL explicitly and fail
    closed — never format it as a dialect."""
    if lowering is MLIRLowering.RESIDUAL:
        raise ValueError(
            "MLIRLowering.RESIDUAL names no MLIR dialect — its lowering "
            "home is undecided (decision record section 2.4, 'flag for "
            "review'); a caller must handle RESIDUAL explicitly, never "
            "format it as a dialect-qualified op")
    return lowering.value
