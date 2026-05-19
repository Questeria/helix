"""
helixc/ir/tile_ir.py — Tile IR (mid-level, explicit memory + async).

The Tile IR is the layer between the value-semantic Tensor IR and the raw
backend (x86-64 / PTX). Its job is to expose:
- explicit memory spaces (HBM / SMEM / REG / TMEM)
- explicit async memory ops (TMA load, async copy, barriers)
- explicit tile sizes (16x16, 64x64, etc.) selected by the autotuner
- explicit warp/CTA/cluster scheduling

Where Tensor IR talks about whole tensors, Tile IR talks about *tiles* of
those tensors and the kernels that operate on them.

v0.1 scope: data structures + a trivial Tensor IR -> Tile IR pass that just
copies scalar ops through. Real tensor-op tiling lands in v0.2 once the
matmul scheduling story is wired.

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping, Optional

from . import tir


# ============================================================================
# Memory spaces
# ============================================================================
class MemSpace(Enum):
    HBM = "hbm"        # GPU global memory (off-chip)
    SMEM = "smem"      # GPU shared memory (on-chip per-block)
    REG = "reg"        # registers
    TMEM = "tmem"      # Blackwell Tensor Memory
    CPU = "cpu"        # CPU memory (host)


# ============================================================================
# Tile types
# ============================================================================
@dataclass(frozen=True)
class TileType:
    """A typed tile: dtype + shape + memory space."""
    dtype: tir.TIRScalar
    shape: tuple[tir.Dim, ...]
    memspace: MemSpace


# ============================================================================
# Operations
# ============================================================================
class TileOpKind(Enum):
    # Tile creation
    TILE_ZEROS = "tile.zeros"
    TILE_CONST = "tile.const"

    # Memory movement
    TILE_LOAD_GLOBAL = "tile.load_global"      # HBM -> SMEM (or REG)
    TILE_STORE_GLOBAL = "tile.store_global"    # SMEM (or REG) -> HBM
    TILE_LOAD_SHARED = "tile.load_shared"      # SMEM -> REG
    TILE_STORE_SHARED = "tile.store_shared"    # REG -> SMEM

    # Async memory (Hopper TMA / Blackwell TMA)
    TMA_LOAD = "async.tma_load"
    TMA_STORE = "async.tma_store"
    BARRIER_WAIT = "async.barrier_wait"

    # Compute on tiles
    TILE_ADD = "tile.add"
    TILE_SUB = "tile.sub"
    TILE_MUL = "tile.mul"
    TILE_MATMUL = "tile.matmul"      # matrix multiply, accumulating into a tile
    TILE_REDUCE = "tile.reduce"      # axis-reduce within tile

    # Layout transforms
    TILE_TRANSPOSE = "tile.transpose"
    TILE_RESHAPE = "tile.reshape"

    # Scalar ops (passed through from Tensor IR for v0.1)
    SCALAR_CONST_INT = "scalar.const_int"
    SCALAR_CONST_FLOAT = "scalar.const_float"
    SCALAR_ADD = "scalar.add"
    SCALAR_SUB = "scalar.sub"
    SCALAR_MUL = "scalar.mul"
    SCALAR_NEG = "scalar.neg"
    SCALAR_CMP = "scalar.cmp"        # carries op via attr
    SCALAR_SELECT = "scalar.select"
    CALL = "call"
    RETURN = "return"

    # Stage 16 — GPU primitives carried through from Tensor IR. PTX backend
    # consumes these directly; x86 backend ignores them (kernel bodies emit
    # PTX text, not host code).
    THREAD_IDX = "gpu.thread_idx"                  # attrs: dim ("x"/"y"/"z")
    TILE_INDEX_LOAD_HBM = "tile.index_load_hbm"    # attrs: name, dtype, memspace
    TILE_INDEX_STORE_HBM = "tile.index_store_hbm"  # attrs: name, dtype, memspace


@dataclass
class TileValue:
    id: int
    ty: tir.TIRType        # may be a TileType wrapped in TIRType, or scalar
    name_hint: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TileValue) and self.id == other.id


@dataclass
class TileOp:
    kind: TileOpKind
    operands: list[TileValue] = field(default_factory=list)
    results: list[TileValue] = field(default_factory=list)
    attrs: dict[str, object] = field(default_factory=dict)
    span: Optional[tuple[int, int]] = None


@dataclass
class TileBlock:
    id: int
    params: list[TileValue] = field(default_factory=list)
    ops: list[TileOp] = field(default_factory=list)


@dataclass
class TileFn:
    name: str
    params: list[TileValue]
    return_ty: tir.TIRType
    blocks: list[TileBlock]
    attrs: dict[str, object] = field(default_factory=dict)

    @property
    def entry(self) -> TileBlock:
        return self.blocks[0]


@dataclass
class TileModule:
    functions: dict[str, TileFn] = field(default_factory=dict)


# ============================================================================
# Tensor IR -> Tile IR lowering (v0.1 trivial)
# ============================================================================
class TirToTileLowerer:
    """v0.1 lowering: supported scalar/GPU ops pass through unchanged.

    Unsupported Tensor IR ops fail closed so tile lowering cannot silently
    report a generic opaque call for behavior it does not yet represent.
    """

    SCALAR_OP_MAP = {
        tir.OpKind.CONST_INT: TileOpKind.SCALAR_CONST_INT,
        tir.OpKind.CONST_FLOAT: TileOpKind.SCALAR_CONST_FLOAT,
        tir.OpKind.ADD: TileOpKind.SCALAR_ADD,
        tir.OpKind.SUB: TileOpKind.SCALAR_SUB,
        tir.OpKind.MUL: TileOpKind.SCALAR_MUL,
        tir.OpKind.NEG: TileOpKind.SCALAR_NEG,
        tir.OpKind.CMP_EQ: TileOpKind.SCALAR_CMP,
        tir.OpKind.CMP_NE: TileOpKind.SCALAR_CMP,
        tir.OpKind.CMP_LT: TileOpKind.SCALAR_CMP,
        tir.OpKind.CMP_LE: TileOpKind.SCALAR_CMP,
        tir.OpKind.CMP_GT: TileOpKind.SCALAR_CMP,
        tir.OpKind.CMP_GE: TileOpKind.SCALAR_CMP,
        tir.OpKind.SELECT: TileOpKind.SCALAR_SELECT,
        tir.OpKind.CALL: TileOpKind.CALL,
        tir.OpKind.RETURN: TileOpKind.RETURN,
        # Stage 16 — GPU ops pass through to PTX backend untouched.
        tir.OpKind.THREAD_IDX: TileOpKind.THREAD_IDX,
        tir.OpKind.TILE_INDEX_LOAD: TileOpKind.TILE_INDEX_LOAD_HBM,
        tir.OpKind.TILE_INDEX_STORE: TileOpKind.TILE_INDEX_STORE_HBM,
    }

    def __init__(self, tir_module: tir.Module):
        self.tir_module = tir_module
        self.module = TileModule()
        self.next_id = 0
        self.value_map: dict[int, TileValue] = {}    # tir.Value.id -> TileValue

    def _new_value(self, ty: tir.TIRType, hint: Optional[str] = None) -> TileValue:
        v = TileValue(id=self.next_id, ty=ty, name_hint=hint)
        self.next_id += 1
        return v

    def _map_value(self, src: tir.Value) -> TileValue:
        if src.id in self.value_map:
            return self.value_map[src.id]
        v = self._new_value(src.ty, src.name_hint)
        self.value_map[src.id] = v
        return v

    def lower(self) -> TileModule:
        for name, fn in self.tir_module.functions.items():
            self._lower_fn(fn)
        return self.module

    def _lower_fn(self, fn: tir.FnIR) -> None:
        new_params = [self._map_value(p) for p in fn.params]
        new_blocks: list[TileBlock] = []
        for blk in fn.blocks:
            new_blk = TileBlock(id=blk.id,
                                params=[self._map_value(p) for p in blk.params])
            for op in blk.ops:
                new_blk.ops.append(self._lower_op(op))
            new_blocks.append(new_blk)
        new_fn = TileFn(name=fn.name, params=new_params,
                        return_ty=fn.return_ty, blocks=new_blocks,
                        attrs=dict(fn.attrs))
        self.module.functions[fn.name] = new_fn

    def _lower_op(self, op: tir.Op) -> TileOp:
        new_kind = self.SCALAR_OP_MAP.get(op.kind)
        if new_kind is None:
            raise NotImplementedError(
                f"Tile IR lowering does not support TIR op {op.kind.value}"
            )
        operands = [self._map_value(o) for o in op.operands]
        results = [self._map_value(r) for r in op.results]
        attrs = dict(op.attrs)
        if op.kind in (tir.OpKind.CMP_EQ, tir.OpKind.CMP_NE, tir.OpKind.CMP_LT,
                       tir.OpKind.CMP_LE, tir.OpKind.CMP_GT, tir.OpKind.CMP_GE):
            attrs["cmp"] = op.kind.value
        return TileOp(kind=new_kind, operands=operands, results=results,
                      attrs=attrs, span=op.span)


def lower_to_tile(tir_module: tir.Module) -> TileModule:
    return TirToTileLowerer(tir_module).lower()


# ============================================================================
# Stage 117-119 (v2.0 Phase B.3) substrate: tile-IR adjoint table.
#
# Maps each forward TileOpKind to the sequence of tile-IR ops that
# compute its reverse-mode gradient contribution. The full Stage
# 120 implementation will consume this table to generate adjoint
# kernels from forward kernels.
#
# Adjoint sequences are documented as Python tuples of (kind, comment)
# pairs — the actual code generation lives in a future stage, but
# the table itself is the type-system substrate that pins the design.
#
# Reverse-mode AD pattern: for forward op f(x, y) = z, the adjoint
# computes (dL/dx, dL/dy) given (x, y, dL/dz).
#
# Reference: v2.0 research Report 2 (AD through @kernel functions).
# Defensible claim: "first open tile-IR-native, tensor-core-aware
# source-level reverse-mode AD in a systems language with its own
# matmul lowering."
# ============================================================================

@dataclass(frozen=True)
class AdjointRecord:
    """Declared reverse-mode adjoint for one forward TileOpKind.

    Frozen + tuple-backed so a downstream consumer cannot mutate the
    canonical table by accident (e.g. `entry.ops.append(...)`).

    Fields:
        inputs:  forward-pass operand names the adjoint reads.
        outputs: gradient output names (one per differentiable input).
        ops:     (TileOpKind, comment) pairs sequenced to compute the
                 gradients. Substrate-only; operand wiring lands in
                 Stage 120's grad_pass.
        dispatch: disambiguator for empty `ops`:
            - "explicit": `ops` is the full computation (default).
            - "identity": gradient flows through unchanged (e.g. TILE_ADD).
            - any other string: name of an attr key on the forward op
              that the backend dispatches on (e.g. TILE_REDUCE →
              "reduce_kind" selects sum-broadcast vs max-scatter).
    """
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    ops: tuple[tuple[TileOpKind, str], ...]
    dispatch: str = "explicit"

    def __post_init__(self) -> None:
        # Stage 120 R1 audit-fix: catch malformed records at construction.
        # dispatch="explicit" with empty ops would silently emit no
        # backward steps and look indistinguishable from a successful
        # adjoint — guard the invariant here, not in the consumer.
        if self.dispatch == "explicit" and not self.ops:
            raise ValueError(
                "AdjointRecord: dispatch='explicit' requires at least one "
                "entry in ops. Use dispatch='identity' for pass-through "
                "gradients or a named dispatch-attr for runtime-keyed cases."
            )
        if self.dispatch == "identity" and self.ops:
            raise ValueError(
                "AdjointRecord: dispatch='identity' must have ops=(); the "
                f"recorded ops {self.ops!r} would never be emitted."
            )


# Stage 117-119 ships the substrate (table + lookup); Stage 120 wires
# this into grad_pass for end-to-end MLP forward→backward generation.
_TILE_OP_ADJOINTS_INNER: dict[TileOpKind, AdjointRecord] = {
    TileOpKind.TILE_ADD: AdjointRecord(
        # z = x + y    →    dL/dx = dL/dz,  dL/dy = dL/dz
        inputs=("x", "y"),
        outputs=("dx", "dy"),
        ops=(),
        dispatch="identity",
    ),
    TileOpKind.TILE_SUB: AdjointRecord(
        # z = x - y    →    dL/dx = dL/dz,  dL/dy = -dL/dz
        inputs=("x", "y"),
        outputs=("dx", "dy"),
        ops=(
            (TileOpKind.SCALAR_NEG, "negate upstream gradient for dy"),
        ),
    ),
    TileOpKind.TILE_MUL: AdjointRecord(
        # z = x * y    →    dL/dx = dL/dz * y,  dL/dy = dL/dz * x
        inputs=("x", "y"),
        outputs=("dx", "dy"),
        ops=(
            (TileOpKind.TILE_MUL, "dx = dz * y (elementwise)"),
            (TileOpKind.TILE_MUL, "dy = dz * x (elementwise)"),
        ),
    ),
    TileOpKind.TILE_MATMUL: AdjointRecord(
        # Forward: D = A @ B + C  (cuBLAS-style accumulating matmul).
        # Reverse: dA = dD @ B^T,  dB = A^T @ dD,  dC = dD (identity).
        # NOTE: C is assumed same-shape as D (no broadcast). If a future
        # bias-add lowering broadcasts C, Stage 120 must reduce dD over
        # the broadcast axes before binding to dC.
        inputs=("A", "B", "C"),
        outputs=("dA", "dB", "dC"),
        ops=(
            (TileOpKind.TILE_TRANSPOSE, "Bt = transpose(B)"),
            (TileOpKind.TILE_MATMUL, "dA = dD @ Bt"),
            (TileOpKind.TILE_TRANSPOSE, "At = transpose(A)"),
            (TileOpKind.TILE_MATMUL, "dB = At @ dD"),
        ),
    ),
    TileOpKind.TILE_REDUCE: AdjointRecord(
        # axis-reduce within tile. Exact gradient op depends on
        # attrs["reduce_kind"]:
        #   sum     → broadcast(dz, original shape)
        #   max/min → scatter dz to the argmax/argmin index
        # Stage 120 / backend resolves via the dispatch attr below.
        inputs=("x",),
        outputs=("dx",),
        ops=(),
        dispatch="reduce_kind",
    ),
    TileOpKind.TILE_TRANSPOSE: AdjointRecord(
        # transpose is its own inverse for the gradient.
        inputs=("x",),
        outputs=("dx",),
        ops=(
            (TileOpKind.TILE_TRANSPOSE, "dx = transpose(dz)"),
        ),
    ),
    TileOpKind.TILE_RESHAPE: AdjointRecord(
        # reshape is its own inverse: dx = reshape(dz, x.shape).
        # Needed by any MLP that flattens between layers.
        inputs=("x",),
        outputs=("dx",),
        ops=(
            (TileOpKind.TILE_RESHAPE, "dx = reshape(dz, x.shape)"),
        ),
    ),
}

# Read-only view of the canonical table. Wrapping in MappingProxyType
# blocks `TILE_OP_ADJOINTS[k] = ...` at the table level; the inner
# AdjointRecord is already frozen so per-entry mutation is also blocked.
TILE_OP_ADJOINTS: Final[Mapping[TileOpKind, AdjointRecord]] = MappingProxyType(
    _TILE_OP_ADJOINTS_INNER
)

# TileOpKinds that are intentionally non-differentiable. Adding a new
# TileOpKind without listing it here OR in TILE_OP_ADJOINTS will trip
# test_adjoint_table_covers_all_tile_op_kinds — that's the point:
# every new op must make a conscious diff / non-diff decision.
TILE_OP_NON_DIFFERENTIABLE: Final[frozenset[TileOpKind]] = frozenset({
    # Constants — zero gradient.
    TileOpKind.TILE_ZEROS,
    TileOpKind.TILE_CONST,
    # Memory boundary — Stage 120's param-grad path writes dW/db back
    # to HBM directly, it doesn't invoke a per-op adjoint here.
    TileOpKind.TILE_LOAD_GLOBAL,
    TileOpKind.TILE_STORE_GLOBAL,
    TileOpKind.TILE_LOAD_SHARED,
    TileOpKind.TILE_STORE_SHARED,
    TileOpKind.TMA_LOAD,
    TileOpKind.TMA_STORE,
    # Async / barrier — no value semantics.
    TileOpKind.BARRIER_WAIT,
    # Scalar ops live below the tile-IR autograd surface (Stage 120
    # differentiates tile-level ops; scalar arithmetic inside reduce
    # bodies and index math is handled by the per-op rule, not by
    # generic SCALAR_* adjoints).
    TileOpKind.SCALAR_CONST_INT,
    TileOpKind.SCALAR_CONST_FLOAT,
    TileOpKind.SCALAR_ADD,
    TileOpKind.SCALAR_SUB,
    TileOpKind.SCALAR_MUL,
    TileOpKind.SCALAR_NEG,
    TileOpKind.SCALAR_CMP,
    TileOpKind.SCALAR_SELECT,
    # Control flow — gradient flows through the call graph, not the op.
    TileOpKind.CALL,
    TileOpKind.RETURN,
    # GPU primitives — indices and thread IDs are not differentiable.
    TileOpKind.THREAD_IDX,
    TileOpKind.TILE_INDEX_LOAD_HBM,
    TileOpKind.TILE_INDEX_STORE_HBM,
})


def has_adjoint(kind: TileOpKind) -> bool:
    """Stage 117-119 substrate — query whether a TileOpKind has a
    declared adjoint sequence. Used by grad_pass (Stage 120) to
    decide whether to descend into a kernel body or treat it as
    opaque (call out to host-side gradient).

    Raises TypeError on non-TileOpKind input — silent membership tests
    on misspelled enums or cross-IR `tir.OpKind` values would otherwise
    fall through to "not differentiable" and corrupt gradient flow.
    """
    if not isinstance(kind, TileOpKind):
        raise TypeError(
            f"has_adjoint expects TileOpKind, got "
            f"{type(kind).__name__}: {kind!r}"
        )
    return kind in TILE_OP_ADJOINTS


def adjoint_outputs(kind: TileOpKind) -> tuple[str, ...]:
    """Stage 117-119 substrate — list the gradient outputs of a
    forward op (in operand order). Returns empty tuple for ops
    without a declared adjoint.

    Raises TypeError on non-TileOpKind input (see has_adjoint).
    """
    if not isinstance(kind, TileOpKind):
        raise TypeError(
            f"adjoint_outputs expects TileOpKind, got "
            f"{type(kind).__name__}: {kind!r}"
        )
    entry = TILE_OP_ADJOINTS.get(kind)
    if entry is None:
        return ()
    return entry.outputs


# ============================================================================
# Pretty print
# ============================================================================
def fmt_module(mod: TileModule) -> str:
    out = []
    for fn in mod.functions.values():
        ps = ", ".join(f"v{p.id}:{tir.fmt_type(p.ty)}" for p in fn.params)
        out.append(f"tile_fn {fn.name}({ps}) -> {tir.fmt_type(fn.return_ty)} {{")
        for blk in fn.blocks:
            out.append(f"  bb{blk.id}:")
            for op in blk.ops:
                rs = ", ".join(f"v{r.id}" for r in op.results)
                ops_ = ", ".join(f"v{o.id}" for o in op.operands)
                attrs = (" {" + ", ".join(f"{k}={v}" for k, v in op.attrs.items()) + "}"
                         if op.attrs else "")
                if rs:
                    out.append(f"    {rs} = {op.kind.value}({ops_}){attrs}")
                else:
                    out.append(f"    {op.kind.value}({ops_}){attrs}")
        out.append("}")
    return "\n".join(out)
