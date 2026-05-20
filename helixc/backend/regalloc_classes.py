"""
helixc/backend/regalloc_classes.py — per-backend register-class models.

v2.4 item 15 (slice 4 of N): the per-backend dtype -> register-class
mappings that `regalloc.allocate_by_class` consumes.

`regalloc.py` is the backend-agnostic allocation engine — linear-scan,
liveness, multi-class partitioning. It does not know that PTX has
`%r` / `%rd` / `%f` / `%p` / `%h` register files or that AMDGCN has
VGPR / SGPR. This module supplies that knowledge: one `<backend>_
register_class(value)` classifier + one `<BACKEND>_REGISTER_POOLS`
pool-size table per backend.

Slice 4 ships the PTX model; slice 5 the ROCm/AMDGCN model.

NOTE — only PTX and ROCm need a register-class model. PTX is a
virtual ISA with explicit `.reg` declarations; AMDGCN is real
assembly with explicit VGPR/SGPR. Metal MSL and WebGPU WGSL are
HIGH-LEVEL shading languages — their downstream compilers
(xcrun-metal, naga) do register allocation. Helix's Metal/WebGPU
emitters emit named variables (`v3`, `v_smem`, ...), not registers,
so they need no Helix-side register-class model. The emitter-wiring
slice (threading the assignment into operand emission) therefore
targets PtxEmitter + HipEmitter only.

License: Apache 2.0
"""
from __future__ import annotations

from typing import Final, Literal, get_args

from ..ir import tile_ir as ti
from ..ir import tir
from .regalloc import MultiClassResult, allocate_by_class


# v2.5 polish (item-15 type-design audit Finding 5): closed-set
# register-class keys per backend. Typing the classifier RETURN as
# the Literal makes a typo'd `return "%rr"` a mypy error; the
# module-load checks below pin each pool dict's keys to its Literal
# so a pool/classifier key-set drift fails loudly at import.
PtxRegClass = Literal["%p", "%r", "%rd", "%f", "%h"]
RocmRegClass = Literal["vgpr", "sgpr"]


# ============================================================================
# PTX register-class model
# ============================================================================
# PtxEmitter declares five register files, each 256 deep
# (`_REG_POOL_CAP = 256`), via `.reg` directives in the kernel header:
#   %p  — .pred   (predicate / bool)
#   %r  — .b32    (32-bit integer)
#   %rd — .b64    (64-bit integer / pointer)
#   %f  — .f32    (32-bit float)
#   %h  — .b16    (16-bit: narrow int + f16/bf16)
# PTX has no register file narrower than 16 bits, so 8-bit dtypes
# (i8/u8/char) are register-allocated in the 16-bit %h file — the
# standard PTX practice (8-bit values are promoted on load).
PTX_REGISTER_POOLS: Final[dict[str, int]] = {
    "%p": 256,
    "%r": 256,
    "%rd": 256,
    "%f": 256,
    "%h": 256,
}

# v2.5 polish (Finding 5): pin PTX_REGISTER_POOLS' keys to the
# PtxRegClass Literal at module load — a key typo or a missing/extra
# pool entry fails loudly here, not as a vacuous allocation later.
if set(PTX_REGISTER_POOLS) != set(get_args(PtxRegClass)):
    raise AssertionError(
        f"helixc.backend.regalloc_classes: PTX_REGISTER_POOLS keys "
        f"{sorted(PTX_REGISTER_POOLS)} != PtxRegClass members "
        f"{sorted(get_args(PtxRegClass))}."
    )


# dtype -> PTX register-class key. Mirrors PtxEmitter._ptx_type_str's
# dtype set; the *class* is coarser than the type suffix (many dtypes
# share a register file).
_PTX_DTYPE_TO_CLASS: Final[dict[str, PtxRegClass]] = {
    "bool": "%p",
    # 16-bit file: narrow ints + half floats. 8-bit dtypes promote here.
    "i8": "%h", "u8": "%h", "char": "%h",
    "i16": "%h", "u16": "%h",
    "f16": "%h", "bf16": "%h",
    # 32-bit integer file.
    "i32": "%r", "u32": "%r",
    # 64-bit integer / pointer file.
    "i64": "%rd", "u64": "%rd", "isize": "%rd", "usize": "%rd",
    # 32-bit float file.
    "f32": "%f",
}


# v2.4 item 15 R1 audit-fix (silent-failure + type-design MEDIUM —
# two-auditor consensus): the shared scalar-dtype vocabulary both
# the PTX and ROCm register-class models accept. f64 is deliberately
# NOT in this set — PTX has no f64 register file (ptx_register_class
# raises NotImplementedError) and ROCm accepts f64 separately (vgpr
# pair). Before R1, `rocm_register_class` borrowed _PTX_DTYPE_TO_CLASS
# as its recognised-dtype gate — a directional coupling: a future
# PTX-only dtype added to that dict would be silently accepted by
# ROCm unreviewed. Both backends now reference this shared constant;
# the module-load check below pins _PTX_DTYPE_TO_CLASS to it so the
# two cannot drift.
#
# v2.4 end-of-cycle 5-clean-gate IR audit-fix: Helix's quantized
# dtypes ("fp8", "mxfp4", "nvfp4", "ternary") are deliberately ABSENT.
# They are parser/typecheck-only front-end types with no backend
# codegen, so no value of those dtypes ever reaches register
# allocation. If one ever did, ptx_register_class / rocm_register_class
# raise RuntimeError loudly rather than mis-filing it — pinned by
# test_regalloc_classes' *_classify_unknown_dtype_raises tests.
_RECOGNISED_SCALAR_DTYPES: Final[frozenset[str]] = frozenset({
    "bool", "i8", "u8", "char", "i16", "u16", "f16", "bf16",
    "i32", "u32", "i64", "u64", "isize", "usize", "f32",
})

if set(_PTX_DTYPE_TO_CLASS) != _RECOGNISED_SCALAR_DTYPES:
    raise AssertionError(
        f"helixc.backend.regalloc_classes: _PTX_DTYPE_TO_CLASS keys "
        f"{sorted(_PTX_DTYPE_TO_CLASS)} != _RECOGNISED_SCALAR_DTYPES "
        f"{sorted(_RECOGNISED_SCALAR_DTYPES)}. Adding a PTX dtype "
        f"requires a conscious decision about the shared vocabulary."
    )


def ptx_register_class(value: ti.TileValue) -> PtxRegClass:
    """v2.4 item 15 slice 4 — map a tile-IR scalar value to its PTX
    register-class key (one of `%p` / `%r` / `%rd` / `%f` / `%h`).

    Pass this as the `classify` argument of `regalloc.allocate_by_class`
    together with `PTX_REGISTER_POOLS`.

    NOTE — this maps purely by dtype, and bool maps to `%p`. That is
    correct for the bool-PREDICATE case (a `SCALAR_CMP` result) but
    NOT for a bool *constant* (`SCALAR_CONST_INT` materialises 0/1 in
    a `%r` b32 register via `mov.b32`). bool's PTX register class is
    op-dependent; `plan_ptx_registers` uses the op-aware
    `ptx_register_class_op_aware` instead. Call this dtype-only
    classifier only when a bool is known to be a predicate, or for
    non-bool values.

    Raises:
        ValueError: if `value.ty` is not a `TIRScalar`. Register
            allocation is for scalar values that occupy exactly one
            register; tile / tensor values are memory-resident (held
            across many registers or in shared memory) — a caller
            must filter those out before allocation.
        NotImplementedError: for `f64` — PtxEmitter declares no f64
            register file (%p/%r/%rd/%f/%h only). f64 GPU kernels
            need a dedicated f64 file; a later item-15 slice adds it.
        RuntimeError: for an unrecognised TIRScalar dtype — surfaces
            a missing mapping entry loudly rather than mis-filing the
            value (parity with PtxEmitter._ptx_type_str's KeyError
            -> raise discipline).
    """
    ty = value.ty
    if not isinstance(ty, tir.TIRScalar):
        raise ValueError(
            f"ptx_register_class: register allocation handles scalar "
            f"values only; vreg {value.id} has type "
            f"{type(ty).__name__} — tile/tensor values are memory-"
            f"resident, not single-register. Filter to scalar values "
            f"before allocate_by_class."
        )
    dtype = ty.name
    if dtype == "f64":
        raise NotImplementedError(
            "ptx_register_class: f64 has no PTX register class — "
            "PtxEmitter declares %p/%r/%rd/%f/%h but no .f64 (%fd) "
            "file. f64 GPU kernels need a dedicated f64 register "
            "file; tracked as a later v2.4 item-15 slice."
        )
    cls = _PTX_DTYPE_TO_CLASS.get(dtype)
    if cls is None:
        raise RuntimeError(
            f"ptx_register_class: unrecognised TIRScalar dtype "
            f"{dtype!r} — add it to _PTX_DTYPE_TO_CLASS."
        )
    return cls


# ============================================================================
# PTX register-allocation planning (v2.5 item 1 — emitter-wiring prep)
# ============================================================================
# bool is the one PTX dtype whose register class is OP-dependent, not
# dtype-dependent. PTX has a predicate file (%p) and a b32 integer file
# (%r); a bool lands in either depending on the op that produces it:
#   SCALAR_CMP       -> `setp` writes a real predicate    -> %p
#   SCALAR_CONST_INT -> `mov.b32` materialises 0/1 in b32 -> %r
# The dtype-only `ptx_register_class` maps bool -> %p unconditionally,
# which is wrong for a bool constant. `plan_ptx_registers` therefore
# classifies a bool op result by its DEFINING op via
# `ptx_register_class_op_aware` + this table. The table is the closed
# set of bool-producing ops the PTX emitter handles; each maps to the
# register file that op's `_result_reg(op, prefix)` call in PtxEmitter
# actually writes into.
_PTX_BOOL_OP_CLASS: Final[dict[ti.TileOpKind, PtxRegClass]] = {
    ti.TileOpKind.SCALAR_CMP: "%p",
    ti.TileOpKind.SCALAR_CONST_INT: "%r",
}


def _ptx_defining_ops(fn: ti.TileFn) -> dict[int, ti.TileOp]:
    """Map each op-result vreg id to the `TileOp` that defines it.

    Kernel / block params are NOT op results and are deliberately
    absent — a caller reads `.get(id)` and treats a missing entry
    (None) as "this value has no defining op".
    """
    defining: dict[int, ti.TileOp] = {}
    for blk in fn.blocks:
        for op in blk.ops:
            for v in op.results:
                defining[v.id] = op
    return defining


def ptx_register_class_op_aware(
    value: ti.TileValue, defining_op: ti.TileOp | None,
) -> PtxRegClass:
    """v2.5 item 1 — op-aware PTX register-class classifier.

    Identical to `ptx_register_class` for every dtype EXCEPT bool,
    whose PTX register class is op-dependent (see `_PTX_BOOL_OP_CLASS`).
    `defining_op` is the `TileOp` whose `results` contain `value`, or
    None when `value` is a kernel / block param (not an op result).

    `plan_ptx_registers` passes this — not the dtype-only
    `ptx_register_class` — as the `allocate_by_class` classifier, so a
    kernel mixing a bool `SCALAR_CMP` predicate (%p) with a bool
    `SCALAR_CONST_INT` constant (%r) allocates each into the file its
    PtxEmitter emit branch actually writes, instead of colliding both
    onto %p.

    Raises:
        RuntimeError: `value` is a bool with no defining op (a kernel /
            block param — bool's class needs the producing op;
            `plan_ptx_registers` skips those before classification, so
            this fires only on direct misuse), or a bool produced by an
            op absent from `_PTX_BOOL_OP_CLASS` (the PTX emitter has no
            register-class mapping for it).
        NotImplementedError / ValueError: via `ptx_register_class` for
            every non-bool value (f64, non-scalar, unknown dtype) —
            those are not op-dependent and delegate unchanged.
    """
    ty = value.ty
    if isinstance(ty, tir.TIRScalar) and ty.name == "bool":
        if defining_op is None:
            raise RuntimeError(
                f"ptx_register_class_op_aware: bool vreg {value.id} has "
                f"no defining op (a kernel / block param). bool's PTX "
                f"register class is op-dependent (SCALAR_CMP -> %p, "
                f"SCALAR_CONST_INT -> %r); a bool with no producing op "
                f"has no determinable class. plan_ptx_registers skips "
                f"such values before classifying — reaching here is a "
                f"direct-call misuse."
            )
        cls = _PTX_BOOL_OP_CLASS.get(defining_op.kind)
        if cls is None:
            raise RuntimeError(
                f"ptx_register_class_op_aware: bool vreg {value.id} is "
                f"defined by {defining_op.kind.name}, which has no PTX "
                f"register-class mapping. Bool-producing ops the PTX "
                f"emitter handles: "
                f"{sorted(k.name for k in _PTX_BOOL_OP_CLASS)}."
            )
        return cls
    return ptx_register_class(value)


def plan_ptx_registers(fn: ti.TileFn) -> MultiClassResult:
    """v2.5 item 1 — compute the PTX register allocation for one
    tile-IR kernel function.

    The thin composition the PtxEmitter operand-emission slice
    consumes: runs `regalloc.allocate_by_class` with the op-aware PTX
    register-class classifier (`ptx_register_class_op_aware`) + pool
    table (`PTX_REGISTER_POOLS`), passing a `skip` predicate that drops
    into `MultiClassResult.skipped` (so the classifier is never handed
    one, and PtxEmitter places them by its own bump allocator):
      - non-scalar (tile / tensor) values — memory-resident (shared
        memory / many registers), not single-register;
      - a `bool` value with no defining op (a kernel / block param) —
        bool's PTX register class is op-dependent, so a bool that is
        not an op result has no determinable class.
    Every other value — including a `bool` that IS an op result — is
    classified. bool's register class depends on its producing op
    (SCALAR_CMP -> %p predicate, SCALAR_CONST_INT -> %r b32);
    `_ptx_defining_ops` builds the vreg -> defining-op map the op-aware
    classifier consults. (An earlier slice excluded all bool from the
    plan as a quick unblock; this is the V2_PLAN.md "option (b)"
    op-aware classification that supersedes it — bool op results now
    join the linear-scan reuse like every other scalar.)

    This function is deliberately separate from operand emission: it
    is pure (no PtxEmitter state, emits no text), so it is unit-
    testable in isolation and the higher-risk emitter rewrite — which
    threads the returned assignment into every `_emit_op` operand —
    builds on a verified planning step rather than doing both at once.

    No-spill contract (v2.4 5-clean-gate IR LOW-2). The emitter slice
    trusts every register-allocated (non-skipped) scalar vreg has a
    `RegAssignment`. A spill would leave a vreg in neither
    `assignment` nor `skipped` — the emitter would then have no
    register to name it by. PTX declares 5 files * 256 registers; a
    Helix tile-IR kernel body (a few dozen scalars) cannot exhaust
    that, so a spill is a bug — a runaway live interval or a mis-sized
    pool — not an expected outcome. It is surfaced loudly here rather
    than trusted into a broken emit.

    Raises:
        NotImplementedError: via `ptx_register_class_op_aware` — a
            scalar value has dtype f64 (PTX declares no f64 register
            file).
        RuntimeError: via `ptx_register_class_op_aware` for an
            unrecognised scalar dtype, or for a bool op result whose
            defining op has no PTX register-class mapping; or here, if
            the allocation spilled (see the no-spill contract above).
        ValueError: via `allocate_by_class` — an empty pool table, or
            liveness/value-map drift (internal invariants).
    """
    defining_ops = _ptx_defining_ops(fn)

    def _skip(v: ti.TileValue) -> bool:
        ty = v.ty
        if not isinstance(ty, tir.TIRScalar):
            return True            # tile/tensor — memory-resident
        # A bool op result IS classified (op-aware). A bool with no
        # defining op (kernel / block param) has no determinable
        # class — skip it onto PtxEmitter's bump allocator.
        return ty.name == "bool" and v.id not in defining_ops

    result = allocate_by_class(
        fn,
        lambda v: ptx_register_class_op_aware(v, defining_ops.get(v.id)),
        PTX_REGISTER_POOLS,
        skip=_skip,
    )
    if result.spill_count != 0:
        raise RuntimeError(
            f"plan_ptx_registers: register allocation for kernel "
            f"{fn.name!r} spilled {result.spill_count} value(s) "
            f"(vregs {sorted(result.spilled)}). PTX declares 5 files "
            f"* 256 registers; a Helix tile-IR kernel body should "
            f"never exhaust that — a spill here is a bug (a runaway "
            f"live interval or a mis-sized pool), not an expected "
            f"outcome. Investigate before trusting the assignment."
        )
    return result


def ptx_register_names(result: MultiClassResult) -> dict[int, str]:
    """v2.5 item 1 — flatten a planned PTX allocation to the
    vreg-id -> register-name map `PtxEmitter.reg_map` consumes.

    `plan_ptx_registers` returns a `MultiClassResult` whose
    `assignment` payload is one `RegAssignment(reg_class, index)` per
    register-allocated scalar vreg. PTX register syntax is the class
    key concatenated with the index — and the `PtxRegClass` keys
    already carry the leading `%` (`%r` / `%rd` / `%f` / `%h` / `%p`),
    so `%r` + `3` -> `%r3`. The returned `dict[int, str]` is the exact
    shape `PtxEmitter.reg_map` (TileValue.id -> "%r3") already uses; the
    operand-emission slice assigns the result straight into `reg_map`.

    Only `assignment` is read. `skipped` vregs (tile / tensor values)
    are memory-resident — named by the emitter's own mechanism, not a
    single register — and a no-spill `MultiClassResult` (the
    `plan_ptx_registers` contract) carries an empty `spilled` set, so
    iterating `assignment` covers every register-allocated value
    exactly once.

    Each entry is checked against `PTX_REGISTER_POOLS`: a class absent
    from the pool table, or an index outside its file, is a
    register-model bug. PTX `.reg` directives declare a fixed pool per
    file; an out-of-pool register name passes silently through Helix
    and is rejected only by the CUDA PTX assembler far downstream. It
    is surfaced loudly here instead — the same name-construction
    boundary the emitter slice will trust.

    This function is pure (no `PtxEmitter` state, emits no text), so it
    is unit-testable in isolation — the higher-risk emitter rewrite
    threads a verified name map rather than building and trusting it
    in one step.

    Raises:
        ValueError: a `RegAssignment` names a register class absent
            from `PTX_REGISTER_POOLS`, or an index outside that
            class's declared pool [0, pool_size).
    """
    names: dict[int, str] = {}
    for vreg, ra in result.assignment.items():
        pool_size = PTX_REGISTER_POOLS.get(ra.reg_class)
        if pool_size is None:
            raise ValueError(
                f"ptx_register_names: vreg {vreg} assigned register "
                f"class {ra.reg_class!r}, which is not a PTX register "
                f"file ({sorted(PTX_REGISTER_POOLS)}). The allocation "
                f"is not consumable by PtxEmitter."
            )
        if not 0 <= ra.index < pool_size:
            raise ValueError(
                f"ptx_register_names: vreg {vreg} assigned "
                f"{ra.reg_class}{ra.index}, an index outside the "
                f"{ra.reg_class} pool [0, {pool_size}). PTX .reg "
                f"directives declare {pool_size} registers per file; an "
                f"out-of-pool name is rejected by ptxas downstream."
            )
        names[vreg] = f"{ra.reg_class}{ra.index}"
    return names


# ============================================================================
# ROCm / AMDGCN register-class model
# ============================================================================
# AMDGCN (gfx942 / MI300) exposes two register files HipEmitter
# allocates into:
#   vgpr — vector general-purpose registers (v0..), 32-bit, per-lane;
#          hold per-thread values — the bulk of a kernel's scalars.
#   sgpr — scalar general-purpose registers (s0..), 32-bit, uniform
#          across the wavefront; hold conditions / predicates / the
#          exec mask.
# Unlike PTX (a register file per width), AMDGCN VGPRs/SGPRs are all
# 32-bit: a 64-bit value occupies a register PAIR in the SAME file.
# Pair-aligned allocation is a register-allocation refinement (a
# later item-15 slice extends linear_scan); the register-CLASS model
# here just answers "which file", and 64-bit values answer "vgpr"
# (the file) — the pairing is orthogonal.
#
# gfx942: 256 VGPRs addressable per wave; ~104 usable SGPRs (the top
# few are reserved for vcc / exec / the wave's hardware state).
ROCM_REGISTER_POOLS: Final[dict[str, int]] = {
    "vgpr": 256,
    "sgpr": 104,
}

# v2.5 polish (Finding 5): pin ROCM_REGISTER_POOLS' keys to the
# RocmRegClass Literal at module load (parity with the PTX check).
if set(ROCM_REGISTER_POOLS) != set(get_args(RocmRegClass)):
    raise AssertionError(
        f"helixc.backend.regalloc_classes: ROCM_REGISTER_POOLS keys "
        f"{sorted(ROCM_REGISTER_POOLS)} != RocmRegClass members "
        f"{sorted(get_args(RocmRegClass))}."
    )

# A boolean / predicate is a wavefront condition — it lives in the
# scalar file (sgpr). Every other scalar dtype is a per-thread value
# in the vector file (vgpr); 64-bit dtypes are vgpr register pairs
# (pairing handled by a later slice — see module note above).
_ROCM_SGPR_DTYPES: Final[frozenset[str]] = frozenset({"bool"})


def rocm_register_class(value: ti.TileValue) -> RocmRegClass:
    """v2.4 item 15 slice 5 — map a tile-IR scalar value to its
    AMDGCN register-class key (`vgpr` or `sgpr`).

    Pass this as the `classify` argument of
    `regalloc.allocate_by_class` together with `ROCM_REGISTER_POOLS`.

    Raises:
        ValueError: if `value.ty` is not a `TIRScalar` — tile/tensor
            values are memory-resident (LDS / HBM), not single-
            register; a caller filters those out before allocation.
        RuntimeError: for an unrecognised TIRScalar dtype — surfaces
            a missing mapping entry loudly rather than mis-filing it.
    """
    ty = value.ty
    if not isinstance(ty, tir.TIRScalar):
        raise ValueError(
            f"rocm_register_class: register allocation handles scalar "
            f"values only; vreg {value.id} has type "
            f"{type(ty).__name__} — tile/tensor values are memory-"
            f"resident (LDS/HBM), not single-register. Filter to "
            f"scalar values before allocate_by_class."
        )
    dtype = ty.name
    # v2.4 item 15 R1 audit-fix: gate on the shared
    # _RECOGNISED_SCALAR_DTYPES constant (plus f64, which ROCm accepts
    # but PTX does not) — NOT on _PTX_DTYPE_TO_CLASS. This removes the
    # directional coupling the 3-clean-audit flagged: ROCm's dtype
    # vocabulary is now an independent, reviewed decision.
    if dtype != "f64" and dtype not in _RECOGNISED_SCALAR_DTYPES:
        raise RuntimeError(
            f"rocm_register_class: unrecognised TIRScalar dtype "
            f"{dtype!r}."
        )
    return "sgpr" if dtype in _ROCM_SGPR_DTYPES else "vgpr"
