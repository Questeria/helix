"""
helixc/backend/tile_ir_audit.py — Stage 130 (v2.0 Phase A.2).

Cross-backend tile-IR coverage audit. Walks every TileOpKind across
all 4 v2.0 backend lowering tables and produces a matrix report:

                   PTX   ROCm   Metal   WebGPU
  TILE_ZEROS      [ok]  [stub] [stub]  [stub]
  TILE_LOAD_GLOBAL [ok] [stub] [stub]  [stub]
  ...
  TMA_LOAD        [ok]  [skip] [skip]  [skip]
  TMEM            [ok]   N/A    N/A     N/A
  ...

Status conventions (matches the backend tables):
- supported: full codegen wired
- stub:      placeholder; needs implementation
- deferred:  blocked on Phase A.1 GPU CI / hardware test
- skipped:   no analog (NVIDIA-only); documented in backend
- missing:   NOT in the backend's table — surfaces as audit failure

Per v2.0 research Report 5 final synthesis: "Tile-IR audit per Report 5
confirming 40 TileOpKind ops decompose cleanly per backend; the 7
NVIDIA-specific ops (TMA, TMEM, etc.) need explicit fallback semantics
in tile-IR before any port begins."

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..ir import tile_ir as ti


# ============================================================================
# Backend table imports (lazy where possible to avoid hard coupling)
# ============================================================================
# The four lowering tables shipped in Stages 123/125/127 (PTX is the
# baseline; its "table" is implicit in ptx.py's _lower_op switch).
# Stage 130 explicitly enumerates PTX coverage via the same status
# vocabulary so the audit matrix is uniform across all four.
PTX_BASELINE_STATUS: dict[ti.TileOpKind, str] = {
    # PTX is the baseline; everything in tile_ir.py is by-definition
    # supported (or stubbed at the same level as the other backends).
    # Stages 1-108 shipped PTX-side codegen for these op kinds:
    ti.TileOpKind.TILE_ZEROS:           "stub",   # zero-init
    ti.TileOpKind.TILE_CONST:           "stub",
    ti.TileOpKind.TILE_LOAD_GLOBAL:     "stub",
    ti.TileOpKind.TILE_STORE_GLOBAL:    "stub",
    ti.TileOpKind.TILE_LOAD_SHARED:     "stub",
    ti.TileOpKind.TILE_STORE_SHARED:    "stub",
    ti.TileOpKind.TMA_LOAD:             "stub",   # Hopper TMA
    ti.TileOpKind.TMA_STORE:            "stub",
    ti.TileOpKind.BARRIER_WAIT:         "stub",
    ti.TileOpKind.TILE_ADD:             "stub",
    ti.TileOpKind.TILE_SUB:             "stub",
    ti.TileOpKind.TILE_MUL:             "stub",
    ti.TileOpKind.TILE_MATMUL:          "stub",   # wmma.*
    ti.TileOpKind.TILE_REDUCE:          "stub",
    ti.TileOpKind.TILE_TRANSPOSE:       "stub",
    ti.TileOpKind.TILE_RESHAPE:         "stub",
    ti.TileOpKind.SCALAR_CONST_INT:     "supported",
    ti.TileOpKind.SCALAR_CONST_FLOAT:   "supported",
    ti.TileOpKind.SCALAR_ADD:           "supported",
    ti.TileOpKind.SCALAR_SUB:           "supported",
    ti.TileOpKind.SCALAR_MUL:           "supported",
    ti.TileOpKind.SCALAR_NEG:           "supported",
    ti.TileOpKind.SCALAR_CMP:           "supported",
    ti.TileOpKind.SCALAR_SELECT:        "supported",
    ti.TileOpKind.CALL:                 "supported",
    ti.TileOpKind.RETURN:               "supported",
    ti.TileOpKind.THREAD_IDX:           "supported",  # Stage 16
    ti.TileOpKind.TILE_INDEX_LOAD_HBM:  "supported",  # Stage 16
    ti.TileOpKind.TILE_INDEX_STORE_HBM: "supported",  # Stage 16
}


@dataclass(frozen=True)
class AuditEntry:
    """One row of the cross-backend audit matrix."""
    op_kind: ti.TileOpKind
    ptx_status: str
    rocm_status: str
    metal_status: str
    webgpu_status: str

    def is_fully_covered(self) -> bool:
        """True if every backend has a recognized status (not 'missing').

        Note: 'skipped' still counts as covered — it means "documented
        as no-analog with rationale", not "forgot to handle"."""
        return all(s != "missing" for s in (
            self.ptx_status, self.rocm_status,
            self.metal_status, self.webgpu_status,
        ))


def _lookup_rocm(kind: ti.TileOpKind) -> str:
    from . import rocm
    entry = rocm.ROCM_OP_LOWERING.get(kind)
    return entry["status"] if entry else "missing"


def _lookup_metal(kind: ti.TileOpKind) -> str:
    from . import metal
    entry = metal.METAL_OP_LOWERING.get(kind)
    return entry["status"] if entry else "missing"


def _lookup_webgpu(kind: ti.TileOpKind) -> str:
    from . import webgpu
    entry = webgpu.WEBGPU_OP_LOWERING.get(kind)
    return entry["status"] if entry else "missing"


def audit_tile_ir_coverage() -> list[AuditEntry]:
    """Stage 130 — walk every TileOpKind and produce a cross-backend
    coverage row.

    Returns a list of AuditEntry, one per TileOpKind, sorted by op-kind
    enum order (deterministic for diff-based regression tests).
    """
    rows: list[AuditEntry] = []
    for kind in ti.TileOpKind:
        rows.append(AuditEntry(
            op_kind=kind,
            ptx_status=PTX_BASELINE_STATUS.get(kind, "missing"),
            rocm_status=_lookup_rocm(kind),
            metal_status=_lookup_metal(kind),
            webgpu_status=_lookup_webgpu(kind),
        ))
    return rows


def find_coverage_gaps() -> list[AuditEntry]:
    """Stage 130 — return only rows where at least one backend reports
    'missing'. Empty list = all 4 backends acknowledge every TileOpKind.
    """
    return [row for row in audit_tile_ir_coverage()
            if not row.is_fully_covered()]


def fmt_audit_matrix(rows: Optional[list[AuditEntry]] = None) -> str:
    """Stage 130 — produce a human-readable coverage matrix.

    Output:
        OP_KIND                  PTX        ROCm       Metal      WebGPU
        TILE_ZEROS               stub       stub       stub       stub
        TILE_MATMUL              stub       stub       stub       stub
        TMA_LOAD                 stub       skipped    skipped    skipped
        ...
    """
    if rows is None:
        rows = audit_tile_ir_coverage()
    out = []
    out.append(f"{'OP_KIND':<28} {'PTX':<10} {'ROCm':<10} "
               f"{'Metal':<10} {'WebGPU':<10}")
    out.append("-" * 78)
    for row in rows:
        out.append(
            f"{row.op_kind.name:<28} {row.ptx_status:<10} "
            f"{row.rocm_status:<10} {row.metal_status:<10} "
            f"{row.webgpu_status:<10}"
        )
    return "\n".join(out)


# ============================================================================
# Self-test at module load: surface drift between backend tables and
# tile-IR at import-time, not test-time. This is the same drift-detector
# pattern used in each backend module + tile_ir.py adjoint table.
# ============================================================================
def _check_audit_coverage_self_test() -> None:
    """Module-load: verify every TileOpKind has a row in PTX_BASELINE_STATUS.
    Backend tables are checked at their own module loads (drift detectors
    in rocm.py / metal.py / webgpu.py). Here we just guard the PTX row."""
    for kind in ti.TileOpKind:
        if kind not in PTX_BASELINE_STATUS:
            raise AssertionError(
                f"helixc.backend.tile_ir_audit: TileOpKind {kind.name} "
                f"is missing from PTX_BASELINE_STATUS. Update with the "
                f"current PTX backend's coverage."
            )


_check_audit_coverage_self_test()
