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

from typing import Final

from ..ir import tile_ir as ti
from ..ir import tir


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

# dtype -> PTX register-class key. Mirrors PtxEmitter._ptx_type_str's
# dtype set; the *class* is coarser than the type suffix (many dtypes
# share a register file).
_PTX_DTYPE_TO_CLASS: Final[dict[str, str]] = {
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


def ptx_register_class(value: ti.TileValue) -> str:
    """v2.4 item 15 slice 4 — map a tile-IR scalar value to its PTX
    register-class key (one of `%p` / `%r` / `%rd` / `%f` / `%h`).

    Pass this as the `classify` argument of `regalloc.allocate_by_class`
    together with `PTX_REGISTER_POOLS`.

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

# A boolean / predicate is a wavefront condition — it lives in the
# scalar file (sgpr). Every other scalar dtype is a per-thread value
# in the vector file (vgpr); 64-bit dtypes are vgpr register pairs
# (pairing handled by a later slice — see module note above).
_ROCM_SGPR_DTYPES: Final[frozenset[str]] = frozenset({"bool"})


def rocm_register_class(value: ti.TileValue) -> str:
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
