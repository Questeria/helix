"""Tests for helixc.backend.gpu_ci — Stage 129 (v2.0 Phase A.1) GPU CI
scaffolding.

Mock-GPU validation infrastructure: validates emitted text shape
without requiring real hardware. Real-HW dispatch is Stage 130+.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.backend.gpu_ci import (
    BackendKind,
    ValidationResult,
    GPU_TOOLS,
    detect_tools,
    validate_emit,
    requires_backend_tool,
)
from helixc.backend.rocm import HipEmitter
from helixc.backend.metal import MslEmitter
from helixc.backend.webgpu import WgslEmitter
from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.tile_ir import lower_to_tile


# ============================================================================
# Module-level state
# ============================================================================
def test_stage129_backend_kind_covers_all_v2_backends():
    """Stage 129 — every backend Helix v2.0 ships has a BackendKind enum
    member. ptx + rocm_hip + metal_msl + webgpu_wgsl."""
    kinds = {b.value for b in BackendKind}
    assert kinds == {"ptx", "rocm_hip", "metal_msl", "webgpu_wgsl"}


def test_stage129_gpu_tools_table_complete():
    """Stage 129 — every BackendKind has at least one entry in GPU_TOOLS.
    Empty list would mean "no real-HW validation possible ever" which
    is wrong for all four targets."""
    for b in BackendKind:
        assert b in GPU_TOOLS, f"BackendKind {b.name} missing from GPU_TOOLS"
        assert len(GPU_TOOLS[b]) > 0, f"BackendKind {b.name} has empty tool list"


# ============================================================================
# Mock validation: ROCm/HIP
# ============================================================================
def test_stage129_validate_rocm_emit_passes():
    """Stage 129 — emit_module output from HipEmitter passes mock validation."""
    src = "@kernel fn k() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = HipEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.ROCM_HIP)
    assert result.mock_passed, result.mock_findings
    assert result.overall_passed()


def test_stage129_validate_rocm_catches_missing_endpgm():
    """Stage 129 — mock validator catches missing s_endpgm terminator."""
    broken = ".amdgcn_target \"amdgcn-amd-amdhsa--gfx942\"\n.text\n.globl k\nk:\n"
    result = validate_emit(broken, BackendKind.ROCM_HIP)
    assert not result.mock_passed
    assert any("s_endpgm" in f for f in result.mock_findings)


# ============================================================================
# Mock validation: Apple Metal MSL
# ============================================================================
def test_stage129_validate_metal_emit_passes():
    """Stage 129 — MslEmitter output passes mock validation."""
    src = "@kernel fn k() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = MslEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.METAL_MSL)
    assert result.mock_passed, result.mock_findings


def test_stage129_validate_metal_catches_missing_stdlib():
    """Stage 129 — mock validator catches missing #include <metal_stdlib>."""
    broken = "using namespace metal;\nkernel void k() {}\n"
    result = validate_emit(broken, BackendKind.METAL_MSL)
    assert not result.mock_passed
    assert any("metal_stdlib" in f for f in result.mock_findings)


# ============================================================================
# Mock validation: WebGPU WGSL
# ============================================================================
def test_stage129_validate_webgpu_emit_passes():
    """Stage 129 — WgslEmitter output passes mock validation."""
    src = "@kernel fn k() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = WgslEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.WEBGPU_WGSL)
    assert result.mock_passed, result.mock_findings


def test_stage129_validate_webgpu_catches_missing_compute():
    """Stage 129 — mock validator catches missing @compute attribute."""
    broken = "fn k() {}\n"
    result = validate_emit(broken, BackendKind.WEBGPU_WGSL)
    assert not result.mock_passed
    assert any("@compute" in f for f in result.mock_findings)


# ============================================================================
# Tool detection + real-HW indirection
# ============================================================================
def test_stage129_detect_tools_returns_list():
    """Stage 129 — detect_tools returns a list (possibly empty) of
    available tool names. Empty list is valid on CI without GPU
    toolchain — that's the v2.0 default."""
    for b in BackendKind:
        result = detect_tools(b)
        assert isinstance(result, list)
        for tool in result:
            assert isinstance(tool, str)


def test_stage129_requires_backend_tool_returns_bool():
    """Stage 129 — requires_backend_tool returns bool for marker
    indirection (pytest.mark.skipif(not requires_backend_tool(...), ...))."""
    for b in BackendKind:
        assert isinstance(requires_backend_tool(b), bool)


def test_stage129_real_hw_deferred_to_stage_130():
    """Stage 129 — real-HW dispatch is deferred to Stage 130+; validation
    result honestly reports `real_hw_passed=None` when tool detected
    but not actually invoked, NOT True (which would lie about coverage)."""
    src = "@kernel fn k() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = HipEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.ROCM_HIP, attempt_real_hw=True)
    if result.real_hw_attempted:
        # Tool was detected — real_hw_passed must be None (deferred), not True.
        assert result.real_hw_passed is None
        assert any("Stage 130" in f for f in result.real_hw_findings)
    else:
        # Tool not available — that's also a valid outcome.
        assert result.real_hw_passed is None


def test_stage129_validate_emit_rejects_unknown_backend():
    """Stage 129 — validate_emit raises ValueError on a backend we don't
    have a validator for. Defends against silent acceptance of garbage."""
    # Use a string that bypasses the enum membership check.
    class FakeBackend:
        value = "not_a_real_backend"
    with pytest.raises(ValueError):
        validate_emit("anything", FakeBackend())  # type: ignore[arg-type]


# ============================================================================
# ValidationResult shape
# ============================================================================
def test_stage129_validation_result_overall_passed_logic():
    """Stage 129 — overall_passed() returns True iff mock passed AND
    (real-HW not attempted OR real-HW passed). Stage 129 audit-fix:
    findings are now tuple[str, ...] (immutable)."""
    # Mock passed, no real-HW attempted → True.
    r = ValidationResult(
        backend=BackendKind.PTX, mock_passed=True, mock_findings=(),
        real_hw_attempted=False, real_hw_passed=None,
        real_hw_tool=None, real_hw_findings=(),
    )
    assert r.overall_passed()

    # Mock failed → False regardless of real-HW.
    r2 = ValidationResult(
        backend=BackendKind.PTX, mock_passed=False, mock_findings=("x",),
        real_hw_attempted=False, real_hw_passed=None,
        real_hw_tool=None, real_hw_findings=(),
    )
    assert not r2.overall_passed()

    # Mock passed, real-HW attempted + failed → False.
    r3 = ValidationResult(
        backend=BackendKind.PTX, mock_passed=True, mock_findings=(),
        real_hw_attempted=True, real_hw_passed=False,
        real_hw_tool="ptxas", real_hw_findings=("bad PTX",),
    )
    assert not r3.overall_passed()

    # Mock passed, real-HW deferred (passed=None) → True (Stage 129 substrate).
    r4 = ValidationResult(
        backend=BackendKind.PTX, mock_passed=True, mock_findings=(),
        real_hw_attempted=True, real_hw_passed=None,
        real_hw_tool="ptxas", real_hw_findings=("deferred",),
    )
    assert r4.overall_passed()


# ============================================================================
# Stage 129 code-reviewer follow-up: PTX test parity
# ============================================================================
def test_stage129_validate_ptx_emit_passes():
    """Stage 129 — PtxEmitter output passes mock validation.

    Closes the code-reviewer-flagged coverage gap (PTX had _validate_ptx
    with 4 directives but no positive-path emit-round-trip test, while
    ROCm/Metal/WebGPU each had one)."""
    from helixc.backend.ptx import PtxEmitter
    src = "@kernel fn k() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = PtxEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.PTX)
    assert result.mock_passed, result.mock_findings


def test_stage129_validate_ptx_catches_missing_version():
    """Stage 129 — mock validator catches missing .version directive."""
    broken = ".target sm_75\n.address_size 64\n.entry k\n"
    result = validate_emit(broken, BackendKind.PTX)
    assert not result.mock_passed
    assert any(".version" in f for f in result.mock_findings)


def test_stage129_validate_ptx_catches_missing_entry():
    """Stage 129 — mock validator catches missing .entry kernel."""
    broken = ".version 8.3\n.target sm_75\n.address_size 64\n"
    result = validate_emit(broken, BackendKind.PTX)
    assert not result.mock_passed
    assert any(".entry" in f or "kernel" in f.lower()
               for f in result.mock_findings)


def test_stage129_validation_result_frozen():
    """Stage 129 audit-fix — ValidationResult is frozen + tuple-backed,
    so mutation raises FrozenInstanceError (or similar)."""
    r = ValidationResult(
        backend=BackendKind.PTX, mock_passed=True, mock_findings=(),
        real_hw_attempted=False, real_hw_passed=None,
        real_hw_tool=None, real_hw_findings=(),
    )
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.mock_passed = False  # type: ignore[misc]


def test_stage129_validation_result_post_init_invariants():
    """Stage 129 audit-fix — __post_init__ rejects representable-but-
    illegal field combinations."""
    # mock_passed=True but mock_findings non-empty → illegal.
    with pytest.raises(ValueError, match="mock_passed"):
        ValidationResult(
            backend=BackendKind.PTX, mock_passed=True,
            mock_findings=("oops",),
            real_hw_attempted=False, real_hw_passed=None,
            real_hw_tool=None, real_hw_findings=(),
        )
    # mock_passed=False but mock_findings empty → also illegal.
    with pytest.raises(ValueError, match="mock_passed"):
        ValidationResult(
            backend=BackendKind.PTX, mock_passed=False,
            mock_findings=(),
            real_hw_attempted=False, real_hw_passed=None,
            real_hw_tool=None, real_hw_findings=(),
        )
    # real_hw_attempted=False but real_hw_tool != None → illegal.
    with pytest.raises(ValueError, match="real_hw_tool"):
        ValidationResult(
            backend=BackendKind.PTX, mock_passed=True, mock_findings=(),
            real_hw_attempted=False, real_hw_passed=None,
            real_hw_tool="ptxas", real_hw_findings=(),
        )
