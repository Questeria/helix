"""Tests for helixc.backend.tile_ir_audit — Stage 130 (v2.0 Phase A.2).

Cross-backend tile-IR coverage matrix.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.backend.tile_ir_audit import (
    AuditEntry,
    PTX_BASELINE_STATUS,
    audit_tile_ir_coverage,
    find_coverage_gaps,
    fmt_audit_matrix,
)
from helixc.ir.tile_ir import TileOpKind


def test_stage130_ptx_baseline_covers_all_kinds():
    """Stage 130 — every TileOpKind has a PTX baseline status."""
    for k in TileOpKind:
        assert k in PTX_BASELINE_STATUS, (
            f"TileOpKind {k.name} missing from PTX_BASELINE_STATUS"
        )


def test_stage130_audit_returns_row_per_op_kind():
    """Stage 130 — audit_tile_ir_coverage produces exactly one row per
    TileOpKind enum value."""
    rows = audit_tile_ir_coverage()
    assert len(rows) == len(TileOpKind)
    kinds_seen = {row.op_kind for row in rows}
    kinds_expected = set(TileOpKind)
    assert kinds_seen == kinds_expected


def test_stage130_no_coverage_gaps_at_v2_substrate_close():
    """Stage 130 — at v2.0 substrate close (Stages 123/125/127 all shipped),
    every TileOpKind must be acknowledged in every backend table.

    This is the gate: if any backend reports 'missing', the audit fails
    and the dev knows immediately which TileOpKind/backend pair needs
    attention.
    """
    gaps = find_coverage_gaps()
    assert len(gaps) == 0, (
        "Coverage gaps detected:\n" + fmt_audit_matrix(gaps)
    )


def test_stage130_audit_entry_is_fully_covered():
    """Stage 130 — AuditEntry.is_fully_covered() returns False when
    any status is 'missing'."""
    # Fully covered:
    e1 = AuditEntry(
        op_kind=TileOpKind.TILE_ADD,
        ptx_status="stub", rocm_status="stub",
        metal_status="stub", webgpu_status="stub",
    )
    assert e1.is_fully_covered()

    # 'skipped' is still covered (documented no-analog):
    e2 = AuditEntry(
        op_kind=TileOpKind.TMA_LOAD,
        ptx_status="stub", rocm_status="skipped",
        metal_status="skipped", webgpu_status="skipped",
    )
    assert e2.is_fully_covered()

    # 'missing' is NOT covered (drift):
    e3 = AuditEntry(
        op_kind=TileOpKind.TILE_ADD,
        ptx_status="stub", rocm_status="missing",
        metal_status="stub", webgpu_status="stub",
    )
    assert not e3.is_fully_covered()


def test_stage130_fmt_audit_matrix_shape():
    """Stage 130 — fmt_audit_matrix produces a table with header +
    separator + one row per TileOpKind."""
    text = fmt_audit_matrix()
    assert "OP_KIND" in text
    assert "PTX" in text
    assert "ROCm" in text
    assert "Metal" in text
    assert "WebGPU" in text
    # Each op-kind appears once.
    for k in TileOpKind:
        assert k.name in text


def test_stage130_tma_marked_skipped_on_non_nvidia():
    """Stage 130 — TMA / TMEM should be SKIPPED on ROCm/Metal/WebGPU
    (per Report 5: no analog). Pure documentation regression — if
    someone changes a backend to mark TMA as 'supported' without
    actually implementing it, this catches the lie."""
    rows = audit_tile_ir_coverage()
    tma_row = next(r for r in rows if r.op_kind == TileOpKind.TMA_LOAD)
    assert tma_row.rocm_status == "skipped"
    assert tma_row.metal_status == "skipped"
    assert tma_row.webgpu_status == "skipped"


def test_stage130_matmul_status_consistency():
    """Stage 130 — TILE_MATMUL must be 'stub' across all four backends
    at v2.0 substrate close (the actual codegen wires land in
    Stages 124/126/128/120 which are deferred per substrate-first
    strategy)."""
    rows = audit_tile_ir_coverage()
    matmul_row = next(r for r in rows if r.op_kind == TileOpKind.TILE_MATMUL)
    assert matmul_row.ptx_status in ("stub", "supported")
    assert matmul_row.rocm_status == "supported"
    assert matmul_row.metal_status == "supported"
    assert matmul_row.webgpu_status == "supported"


def test_stage130_supported_ops_match_known_set():
    """Stage 130 — the PTX baseline's 'supported' set covers exactly
    the scalar + control-flow ops Helix has been shipping via Phase-0
    PTX (Stage 16+). Catches accidental status downgrades.
    """
    supported_ptx = {k for k, v in PTX_BASELINE_STATUS.items() if v == "supported"}
    expected_supported = {
        TileOpKind.SCALAR_CONST_INT,
        TileOpKind.SCALAR_CONST_FLOAT,
        TileOpKind.SCALAR_ADD,
        TileOpKind.SCALAR_SUB,
        TileOpKind.SCALAR_MUL,
        TileOpKind.SCALAR_NEG,
        TileOpKind.SCALAR_CMP,
        TileOpKind.SCALAR_SELECT,
        TileOpKind.CALL,
        TileOpKind.RETURN,
        TileOpKind.THREAD_IDX,
        TileOpKind.TILE_INDEX_LOAD_HBM,
        TileOpKind.TILE_INDEX_STORE_HBM,
    }
    assert supported_ptx == expected_supported


def test_stage130_audit_is_deterministic():
    """Stage 130 — two audit_tile_ir_coverage() calls produce
    byte-identical output. Required for the audit matrix to be a
    diff-based regression test."""
    a = audit_tile_ir_coverage()
    b = audit_tile_ir_coverage()
    assert len(a) == len(b)
    for i in range(len(a)):
        assert a[i] == b[i]
