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

Slice 4 ships the PTX model. ROCm / Metal / WebGPU models land in
subsequent slices; the emitter-wiring (threading the resulting
register assignment into operand emission) is the slice after that.

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
