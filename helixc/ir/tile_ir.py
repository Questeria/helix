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
from typing import Optional

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

# Tile-IR ops with a known adjoint sequence. Each entry maps a forward
# TileOpKind to a tuple describing the adjoint computation:
#   - "inputs":   list of forward-pass operand kinds we read
#   - "outputs":  list of gradient outputs (one per differentiable input)
#   - "ops":      list of (TileOpKind, comment) sequenced to compute the gradients
#
# Stage 117-119 ships the substrate (table + lookup); Stage 120 wires
# this into grad_pass for end-to-end MLP forward→backward generation.
TILE_OP_ADJOINTS: dict[TileOpKind, dict] = {
    TileOpKind.TILE_ADD: {
        # z = x + y    →    dL/dx = dL/dz,  dL/dy = dL/dz
        # No new ops; gradient flows through identity.
        "inputs": ["x", "y"],
        "outputs": ["dx", "dy"],
        "ops": [
            # Both adjoints are the upstream gradient unchanged.
            # Backend may emit a single broadcast-copy.
        ],
    },
    TileOpKind.TILE_SUB: {
        # z = x - y    →    dL/dx = dL/dz,  dL/dy = -dL/dz
        "inputs": ["x", "y"],
        "outputs": ["dx", "dy"],
        "ops": [
            # dy = -dz (scalar-neg or sub-from-zero)
            (TileOpKind.SCALAR_NEG, "negate upstream gradient for dy"),
        ],
    },
    TileOpKind.TILE_MUL: {
        # z = x * y    →    dL/dx = dL/dz * y,  dL/dy = dL/dz * x
        # Requires saving x and y on the forward pass (or recomputing).
        "inputs": ["x", "y"],
        "outputs": ["dx", "dy"],
        "ops": [
            (TileOpKind.TILE_MUL, "dx = dz * y (elementwise)"),
            (TileOpKind.TILE_MUL, "dy = dz * x (elementwise)"),
        ],
    },
    TileOpKind.TILE_MATMUL: {
        # Forward: D = A @ B + C  (cuBLAS-style accumulating matmul)
        # Reverse: dA = dD @ B^T,  dB = A^T @ dD,  dC = dD
        # Three TILE_MATMUL calls + two TILE_TRANSPOSE.
        "inputs": ["A", "B", "C"],
        "outputs": ["dA", "dB", "dC"],
        "ops": [
            (TileOpKind.TILE_TRANSPOSE, "Bt = transpose(B)"),
            (TileOpKind.TILE_MATMUL, "dA = dD @ Bt"),
            (TileOpKind.TILE_TRANSPOSE, "At = transpose(A)"),
            (TileOpKind.TILE_MATMUL, "dB = At @ dD"),
            # dC = dD (identity); backend emits a copy or aliases.
        ],
    },
    TileOpKind.TILE_REDUCE: {
        # axis-reduce within tile  →  broadcast upstream gradient back
        # along the reduced axis.
        "inputs": ["x"],
        "outputs": ["dx"],
        "ops": [
            # The exact op depends on reduction kind (sum vs max/min).
            # For sum: broadcast(dz, original shape).
            # For max/min: scatter dz to the argmax/argmin index.
            # Backend dispatches on attrs["reduce_kind"].
        ],
    },
    TileOpKind.TILE_TRANSPOSE: {
        # transpose is its own inverse for the gradient.
        "inputs": ["x"],
        "outputs": ["dx"],
        "ops": [
            (TileOpKind.TILE_TRANSPOSE, "dx = transpose(dz)"),
        ],
    },
}


def has_adjoint(kind: TileOpKind) -> bool:
    """Stage 117-119 substrate — query whether a TileOpKind has a
    declared adjoint sequence. Used by grad_pass (Stage 120) to
    decide whether to descend into a kernel body or treat it as
    opaque (call out to host-side gradient).
    """
    return kind in TILE_OP_ADJOINTS


def adjoint_outputs(kind: TileOpKind) -> tuple[str, ...]:
    """Stage 117-119 substrate — list the gradient outputs of a
    forward op (in operand order). Returns empty tuple for ops
    without a declared adjoint.
    """
    entry = TILE_OP_ADJOINTS.get(kind)
    if entry is None:
        return ()
    return tuple(entry["outputs"])


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
