"""
kovc/ir/tir.py — Tensor IR (high-level, value-semantic).

The Tensor IR is the primary optimization layer:
- Whole-tensor operations (matmul, conv, elementwise, reduce, broadcast)
- Named axes via ShapeSpec
- Layout as type info (RowMajor / ColMajor / Blocked)
- Structured ops in the linalg sense (declared iteration space, indexing maps)

SSA with block parameters (Cranelift CLIF / Swift SIL pattern). No phi nodes.

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


# ============================================================================
# Types in Tensor IR
# ============================================================================
class Layout(Enum):
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"
    BLOCKED = "blocked"        # parameters via TIRType
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TIRType:
    """Base class for Tensor IR types."""
    pass


@dataclass(frozen=True)
class TIRScalar(TIRType):
    """A scalar element type."""
    name: str   # "f32", "bf16", "i32", "ternary", "fp8_e4m3", etc.


@dataclass(frozen=True)
class TIRTensorTy(TIRType):
    """tensor<dtype, shape, device, layout>"""
    dtype: TIRScalar
    shape: tuple["Dim", ...]
    device: str = "cpu"
    layout: Layout = Layout.ROW_MAJOR


@dataclass(frozen=True)
class TIRTileTy(TIRType):
    """Sized tile in a memory space."""
    dtype: TIRScalar
    shape: tuple["Dim", ...]
    memspace: str   # "hbm", "smem", "reg", "tmem"


@dataclass(frozen=True)
class TIRTuple(TIRType):
    elems: tuple[TIRType, ...]


@dataclass(frozen=True)
class TIRUnit(TIRType):
    pass


# ============================================================================
# Dims (size expressions in tensor shapes)
# ============================================================================
@dataclass(frozen=True)
class Dim:
    """Base class for shape dimensions."""
    pass


@dataclass(frozen=True)
class DimConst(Dim):
    value: int


@dataclass(frozen=True)
class DimVar(Dim):
    """A named size parameter from a function generic."""
    name: str


@dataclass(frozen=True)
class DimDyn(Dim):
    """Dynamic dimension, runtime-checked at boundary."""
    pass


@dataclass(frozen=True)
class DimExpr(Dim):
    """Arithmetic on dims: a + b, a * b, a / b, a % b."""
    op: str
    args: tuple[Dim, ...]


# ============================================================================
# SSA values
# ============================================================================
@dataclass
class Value:
    """An SSA value in Tensor IR. Identified by id; carries a type."""
    id: int
    ty: TIRType
    # Optional source for debug/inspection
    name_hint: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Value) and self.id == other.id


# ============================================================================
# Operations
# ============================================================================
class OpKind(Enum):
    # Constants
    CONST_INT = "const.int"
    CONST_FLOAT = "const.float"
    CONST_BOOL = "const.bool"
    CONST_TENSOR = "const.tensor"        # immediate tensor, e.g., zeros/ones/literal

    # Tensor creation / memory
    TENSOR_ZEROS = "tensor.zeros"        # shape -> tensor of zeros
    TENSOR_ONES = "tensor.ones"
    TENSOR_FULL = "tensor.full"
    TENSOR_RAND = "tensor.rand"
    TENSOR_LOAD = "tensor.load"          # external (file, host buffer)
    TENSOR_STORE = "tensor.store"

    # Elementwise (binary)
    ADD = "elem.add"
    SUB = "elem.sub"
    MUL = "elem.mul"
    DIV = "elem.div"
    MAXIMUM = "elem.maximum"
    MINIMUM = "elem.minimum"
    POW = "elem.pow"

    # Elementwise (unary)
    NEG = "elem.neg"
    ABS = "elem.abs"
    EXP = "elem.exp"
    LOG = "elem.log"
    SQRT = "elem.sqrt"
    RECIP = "elem.recip"
    RELU = "elem.relu"
    GELU = "elem.gelu"
    SILU = "elem.silu"
    TANH = "elem.tanh"
    SIGMOID = "elem.sigmoid"

    # Reductions
    REDUCE_SUM = "reduce.sum"
    REDUCE_MEAN = "reduce.mean"
    REDUCE_MAX = "reduce.max"
    REDUCE_MIN = "reduce.min"
    REDUCE_PROD = "reduce.prod"

    # Linear algebra
    MATMUL = "matmul"
    CONV1D = "conv1d"
    CONV2D = "conv2d"

    # Shape ops
    RESHAPE = "shape.reshape"
    TRANSPOSE = "shape.transpose"
    BROADCAST = "shape.broadcast"
    SLICE = "shape.slice"
    CONCAT = "shape.concat"

    # Cast / quantize
    CAST = "cast"
    QUANTIZE = "quantize"
    DEQUANTIZE = "dequantize"

    # Control flow primitives
    SELECT = "select"           # ternary if (cond, a, b) elementwise
    WHERE = "where"             # masked select

    # Compositional transforms (compiler-level, materialized via passes)
    GRAD = "transform.grad"
    JVP = "transform.jvp"
    VMAP = "transform.vmap"

    # Function-call (whole-program; resolved later)
    CALL = "call"

    # Block control (used in CFG)
    BR = "br"                    # unconditional branch with args
    COND_BR = "cond_br"          # conditional branch
    RETURN = "return"

    # Effectful ops (kept distinct so transforms can avoid them)
    PRINT = "io.print"


@dataclass
class Op:
    """A single operation in Tensor IR."""
    kind: OpKind
    operands: list[Value] = field(default_factory=list)
    results: list[Value] = field(default_factory=list)

    # Operation-specific attributes (axis, value, target block, etc.)
    attrs: dict[str, object] = field(default_factory=dict)

    # Source span for diagnostics (line, col)
    span: Optional[tuple[int, int]] = None

    def __repr__(self) -> str:
        result_str = ", ".join(f"v{r.id}" for r in self.results)
        operand_str = ", ".join(f"v{o.id}" for o in self.operands)
        attrs_str = ""
        if self.attrs:
            attrs_str = " {" + ", ".join(f"{k}={v}" for k, v in self.attrs.items()) + "}"
        if result_str:
            return f"{result_str} = {self.kind.value}({operand_str}){attrs_str}"
        return f"{self.kind.value}({operand_str}){attrs_str}"


# ============================================================================
# Blocks and functions
# ============================================================================
@dataclass
class Block:
    """A basic block. Block parameters replace phi nodes — values that flow in
    from predecessor branches are explicit."""
    id: int
    params: list[Value] = field(default_factory=list)
    ops: list[Op] = field(default_factory=list)


@dataclass
class FnIR:
    """A function in Tensor IR."""
    name: str
    params: list[Value]                           # SSA values for function args
    return_ty: TIRType
    blocks: list[Block]
    attrs: dict[str, object] = field(default_factory=dict)

    @property
    def entry(self) -> Block:
        return self.blocks[0]


@dataclass
class Module:
    """A whole Tensor IR module."""
    functions: dict[str, FnIR] = field(default_factory=dict)
    next_value_id: int = 0
    next_block_id: int = 0


# ============================================================================
# IR Builder
# ============================================================================
class IRBuilder:
    """Helper to construct Tensor IR programmatically."""
    def __init__(self, module: Module):
        self.module = module
        self.current_fn: Optional[FnIR] = None
        self.current_block: Optional[Block] = None

    # ---- value/block allocation ----
    def new_value(self, ty: TIRType, hint: Optional[str] = None) -> Value:
        v = Value(id=self.module.next_value_id, ty=ty, name_hint=hint)
        self.module.next_value_id += 1
        return v

    def new_block(self) -> Block:
        b = Block(id=self.module.next_block_id)
        self.module.next_block_id += 1
        return b

    # ---- function building ----
    def begin_function(self, name: str, params: list[tuple[str, TIRType]],
                       return_ty: TIRType, attrs: Optional[dict] = None) -> FnIR:
        param_values = [self.new_value(t, hint=n) for n, t in params]
        entry = self.new_block()
        fn = FnIR(name=name, params=param_values, return_ty=return_ty,
                  blocks=[entry], attrs=attrs or {})
        self.module.functions[name] = fn
        self.current_fn = fn
        self.current_block = entry
        return fn

    def end_function(self) -> None:
        self.current_fn = None
        self.current_block = None

    # ---- ops ----
    def emit(self, kind: OpKind, *operands: Value,
             result_ty: Optional[TIRType] = None,
             attrs: Optional[dict] = None,
             span: Optional[tuple[int, int]] = None) -> Optional[Value]:
        assert self.current_block is not None, "emit outside function"
        results: list[Value] = []
        if result_ty is not None:
            r = self.new_value(result_ty)
            results.append(r)
        op = Op(kind=kind, operands=list(operands), results=results,
                attrs=attrs or {}, span=span)
        self.current_block.ops.append(op)
        return results[0] if results else None

    def const_int(self, value: int, dtype: str = "i32") -> Value:
        return self.emit(OpKind.CONST_INT, result_ty=TIRScalar(dtype),
                         attrs={"value": value})

    def const_float(self, value: float, dtype: str = "f32") -> Value:
        return self.emit(OpKind.CONST_FLOAT, result_ty=TIRScalar(dtype),
                         attrs={"value": value})

    def add(self, a: Value, b: Value) -> Value:
        # Result type follows lhs (simplified)
        return self.emit(OpKind.ADD, a, b, result_ty=a.ty)

    def matmul(self, a: Value, b: Value, result_ty: TIRType) -> Value:
        return self.emit(OpKind.MATMUL, a, b, result_ty=result_ty)

    def ret(self, value: Optional[Value] = None) -> None:
        operands = [value] if value is not None else []
        self.emit(OpKind.RETURN, *operands)


# ============================================================================
# Pretty printing
# ============================================================================
def fmt_dim(d: Dim) -> str:
    if isinstance(d, DimConst): return str(d.value)
    if isinstance(d, DimVar): return d.name
    if isinstance(d, DimDyn): return "?"
    if isinstance(d, DimExpr):
        return f"({fmt_dim(d.args[0])} {d.op} {fmt_dim(d.args[1])})"
    return f"<{type(d).__name__}>"


def fmt_type(t: TIRType) -> str:
    if isinstance(t, TIRScalar): return t.name
    if isinstance(t, TIRTensorTy):
        shp = ", ".join(fmt_dim(d) for d in t.shape)
        return f"tensor<{t.dtype.name}, [{shp}], {t.device}, {t.layout.value}>"
    if isinstance(t, TIRTileTy):
        shp = ", ".join(fmt_dim(d) for d in t.shape)
        return f"tile<{t.dtype.name}, [{shp}], {t.memspace}>"
    if isinstance(t, TIRTuple):
        return "(" + ", ".join(fmt_type(e) for e in t.elems) + ")"
    if isinstance(t, TIRUnit): return "()"
    return f"<{type(t).__name__}>"


def fmt_value(v: Value) -> str:
    base = f"v{v.id}"
    return f"{base}:{fmt_type(v.ty)}" + (f"<{v.name_hint}>" if v.name_hint else "")


def fmt_op(op: Op, indent: str = "    ") -> str:
    rs = ", ".join(fmt_value(r) for r in op.results)
    os_ = ", ".join(f"v{o.id}" for o in op.operands)
    attrs = ""
    if op.attrs:
        attrs = " {" + ", ".join(f"{k}={v}" for k, v in op.attrs.items()) + "}"
    if rs:
        return f"{indent}{rs} = {op.kind.value}({os_}){attrs}"
    return f"{indent}{op.kind.value}({os_}){attrs}"


def fmt_block(blk: Block) -> str:
    out = []
    if blk.params:
        ps = ", ".join(fmt_value(p) for p in blk.params)
        out.append(f"  bb{blk.id}({ps}):")
    else:
        out.append(f"  bb{blk.id}:")
    for op in blk.ops:
        out.append(fmt_op(op))
    return "\n".join(out)


def fmt_function(fn: FnIR) -> str:
    out = []
    ps = ", ".join(fmt_value(p) for p in fn.params)
    attrs = ""
    if fn.attrs:
        attrs = " " + " ".join(f"@{k}" if v is True else f"@{k}({v})"
                              for k, v in fn.attrs.items())
    out.append(f"fn {fn.name}({ps}) -> {fmt_type(fn.return_ty)}{attrs} {{")
    for blk in fn.blocks:
        out.append(fmt_block(blk))
    out.append("}")
    return "\n".join(out)


def fmt_module(mod: Module) -> str:
    return "\n\n".join(fmt_function(fn) for fn in mod.functions.values())
