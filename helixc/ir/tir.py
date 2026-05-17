"""
helixc/ir/tir.py — Tensor IR (high-level, value-semantic).

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
    MOD = "elem.mod"
    MAXIMUM = "elem.maximum"
    MINIMUM = "elem.minimum"
    POW = "elem.pow"
    # Bitwise integer ops (i32). Surface syntax: & | ^.
    BIT_AND = "elem.bit_and"
    BIT_OR = "elem.bit_or"
    BIT_XOR = "elem.bit_xor"
    # Shifts. SHL is logical left shift; SHR is arithmetic right shift
    # (preserves sign — matches Rust's `>>` on signed types). Helix has
    # no unsigned int type yet, so logical right shift is unreachable.
    SHL = "elem.shl"
    SHR = "elem.shr"
    # Bitwise unary NOT (~): one's complement, flips every bit. Distinct
    # from logical NOT (`!`) which is lowered as CMP_EQ-against-0.
    BIT_NOT = "elem.bit_not"

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
    BITCAST = "bitcast"  # bit-level reinterpret (f32 <-> i32, f64 <-> i64)
    QUANTIZE = "quantize"
    DEQUANTIZE = "dequantize"

    # Control flow primitives
    SELECT = "select"           # ternary if (cond, a, b) elementwise
    WHERE = "where"             # masked select

    # Comparisons (produce bool/i1 results)
    CMP_EQ = "cmp.eq"
    CMP_NE = "cmp.ne"
    CMP_LT = "cmp.lt"
    CMP_LE = "cmp.le"
    CMP_GT = "cmp.gt"
    CMP_GE = "cmp.ge"

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

    # Mutable local variables (stack-resident cells)
    ALLOC_VAR = "var.alloc"      # attrs: name, dtype
    LOAD_VAR = "var.load"        # attrs: name
    STORE_VAR = "var.store"      # operand: value, attrs: name

    # Stack-allocated arrays
    ALLOC_ARRAY = "array.alloc"      # attrs: name, dtype, length
    LOAD_ELEM = "array.load_elem"    # operands: index_value; attrs: name
    STORE_ELEM = "array.store_elem"  # operands: index, value; attrs: name

    # AGI-specific
    QUOTE = "agi.quote"          # attrs: ast (the captured AST as a Python object)
    SPLICE = "agi.splice"        # operand: AstNode value
    MODIFY = "agi.modify"        # operands: target, transformation, verifier
    REFLECT_HASH = "agi.reflect_hash"  # placeholder: hash an AstNode for testing

    # Arena allocator — single shared bump-allocated i32 region. Used by
    # the eventual self-hosted compiler for AST/IR/symbol-table storage
    # without needing real malloc. Backed by a fixed-size R+W data section.
    ARENA_PUSH = "arena.push"    # operand: value (i32) → result: slot index
    ARENA_GET = "arena.get"      # operand: index (i32) → result: value
    ARENA_SET = "arena.set"      # operands: index, value → no result
    ARENA_LEN = "arena.len"      # no operand → result: current length
    # Stage 36 Inc 9 type-design A2 fix: atomic two-value push. Pushes
    # `left` at the current cursor slot and `right` at cursor+1 in a
    # single IR op so DCE / CSE / scheduler reordering cannot split the
    # pair. Bounds check verifies BOTH slots fit; on overflow neither is
    # written and the result is -1. Result: the slot index of `left`
    # (i.e. the old cursor). Used by register_derivation to keep the
    # "left at N, right at N+1" handle invariant intact under any
    # concurrent arena consumer (struct lowering, MatchDispatch, ...).
    ARENA_PUSH_PAIR = "arena.push_pair"  # operands: left, right → result: slot index of left
    # Stage 36 Inc 14: ternary fused-push extension. Parallels
    # ARENA_PUSH_PAIR but for three slots (left at N, middle at N+1,
    # right at N+2). Used by register_derivation3 to record a three-
    # parent derivation atomically. Bounds check requires room for all
    # three; on overflow none are written and the result is -1. Result:
    # the slot index of `left` (the old cursor).
    ARENA_PUSH_TRIPLE = "arena.push_triple"  # operands: left, middle, right → result: slot index of left

    # String operations on string literals. Self-host needs byte access to
    # source code; for v0.1 only literal strings are supported (no runtime
    # buffers yet). attrs: text=str literal.
    STR_BYTE = "str.byte"        # operand: index → result: byte at literal[i]

    # Stage 16.5 — FFI: a raw pointer (u64) to the bytes of a string literal.
    # attrs: text=str literal. Backend emits `lea rax, [rip + <symbol>]` and
    # stores the resulting i64 to the result slot.
    STR_PTR = "str.ptr"

    # Stage 16.5 — FFI: a call to an extern "C" function. attrs: target=name
    # (libc symbol), arg_types/ret_type as TIR types. Backend emits an
    # indirect call through the GOT entry resolved by the dynamic linker.
    FFI_CALL = "ffi.call"

    # Stage 16 — GPU kernel ops. Only emitted inside fns with @kernel attr.
    # THREAD_IDX: no operand, result is i32. attrs: dim="x"|"y"|"z" (default "x").
    # Lowers to `mov.u32 %r, %tid.<dim>` in PTX.
    THREAD_IDX = "gpu.thread_idx"
    # TILE_INDEX_LOAD: operands [idx], result is dtype scalar. attrs:
    # name=param-name (which kernel param this tile refers to),
    # dtype=str (e.g. "f32"), memspace=str ("hbm"). PTX backend
    # lowers to `ld.global.<dtype>` via the kernel's param pointer.
    TILE_INDEX_LOAD = "tile.index_load"
    # TILE_INDEX_STORE: operands [idx, value], no result. attrs as above.
    # PTX backend lowers to `st.global.<dtype>`.
    TILE_INDEX_STORE = "tile.index_store"

    # Effectful ops (kept distinct so transforms can avoid them)
    PRINT = "io.print"

    # Stage 28.5 — panic / abort. `panic("msg")` lowers to a TRAP op
    # carrying the message string in `attrs["text"]` and the trap id
    # in `attrs["trap_id"]` (default 28501). Backend writes the message
    # to stderr and then exits with a non-zero status (does NOT return).
    TRAP = "ctrl.trap"

    # Stage 49 Inc 1 — Result<T,E> packed-tag operations.
    # Phase-0 (Stages 46-48) lowered Result identity to its Ok inner with
    # no runtime tag. Stage 49 introduces a 2-slot packed representation:
    # a Result<T, E> at the IR level is a single i64 where the high 32
    # bits hold the tag (0 = Ok, 1 = Err) and the low 32 bits hold the
    # payload (Ok-inner OR Err-inner, both currently constrained to i32
    # for Inc 1). Wider payloads are deferred to Stage 50+.
    #
    # Convention (recorded so all three sites — lower_ast.py, x86_64.py,
    # tests — stay in sync):
    #   packed = (tag << 32) | (payload & 0xFFFFFFFF)
    #   RESULT_TAG     = packed >> 32   (logical, zero-extending — tag is
    #                                     small non-negative so shr == sar
    #                                     in observable behaviour, but we
    #                                     use shr/zero-extend for clarity)
    #   RESULT_PAYLOAD = packed & 0xFFFFFFFF   (low 32 bits, truncated)
    #
    # Why high-tag / low-payload: matches a natural Rust-style discriminated
    # union memory layout where the tag lives at the lowest address but we
    # store packed little-endian — so in register form the tag sits in the
    # high half of rax. Either ordering would work; this one keeps the
    # existing CALL/RETURN i64 path unchanged (rax full-width move) without
    # any byte-swap on the payload-extract fast path.
    #
    # All three ops are pure-functional, side-effect-free, and elidable by
    # DCE if their result is unused.
    #
    # Tag-value reservation policy (Stage 49 gate-1 type-design M2):
    # tag 0 = Ok, tag 1 = Err are reserved EXCLUSIVELY for Result<T, E>.
    # Future discriminated-union families (e.g. Option<T> in Stage 50+)
    # MUST get their own opcode family (e.g. OPTION_PACK / OPTION_TAG /
    # OPTION_PAYLOAD with their own tag-value convention) — DO NOT reuse
    # RESULT_TAG to query an Option discriminator, even if the natural
    # value (None = 0, Some = 1) happens to collide. Sharing the opcode
    # family would let is_ok-on-Option typecheck silently.
    RESULT_PACK = "result.pack"        # operands: (tag i32, payload i32) -> result: packed i64
    RESULT_TAG = "result.tag"          # operand: packed i64 -> result: tag i32
    RESULT_PAYLOAD = "result.payload"  # operand: packed i64 -> result: payload i32

    # Stage 25 / Audit 28.8 A7 — @trace fn prologue/epilogue events.
    # TRACE_ENTRY: emitted at the start of a `@trace`-attributed fn's
    # body. `attrs["fn_name"]` holds the name string the backend will
    # pass to the runtime (`__helix_trace_entry(name_ptr)`).
    # TRACE_EXIT: emitted at fn return. `attrs["fn_name"]` same as
    # entry; operand[0] is the return value (so the backend can pass
    # it to `__helix_trace_exit(name_ptr, ret_val)` for recording).
    #
    # Phase-0 wiring: the IR ops are emitted by `lower_ast.lower_fn`
    # when `is_traced(fn)` is True. Backend lowering to actual
    # `call` instructions is deferred until the runtime helpers
    # `__helix_trace_entry` / `__helix_trace_exit` are linked — at
    # which point the x86_64 emitter can resolve relocations against
    # them. Until the runtime exists, the backend emits these as
    # no-op stubs so the IR stays observable + tested.
    TRACE_ENTRY = "trace.entry"
    TRACE_EXIT = "trace.exit"


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

    def append_block(self) -> "Block":
        """Create a new block in the current function and return it (without
        switching to it)."""
        assert self.current_fn is not None
        b = self.new_block()
        self.current_fn.blocks.append(b)
        return b

    def switch_to(self, block: "Block") -> None:
        self.current_block = block

    def new_block_param(self, ty: TIRType, hint: Optional[str] = None) -> Value:
        assert self.current_block is not None
        v = self.new_value(ty, hint)
        self.current_block.params.append(v)
        return v

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
