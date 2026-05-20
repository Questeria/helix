"""Tests for helixc.backend.gpu_ci — Stage 129 (v2.0 Phase A.1) GPU CI
scaffolding.

Mock-GPU validation infrastructure: validates emitted text shape
without requiring real hardware. Real-HW dispatch is Stage 130+.
"""

from __future__ import annotations
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.backend.gpu_ci import (
    BackendKind,
    OverallStatus,
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


def test_v2x_reaudit_rocm_operand_less_kernel_flagged():
    """v2.x re-audit R1 regression (BE 5-clean-gate HIGH): a ROCm
    kernel whose ops emit operand-less substrate text (TILE_MATMUL et
    al.) must be flagged non-functional by validate_emit. Pre-fix
    rocm.py omitted the HELIX-STUB-OPERANDS marker its metal/webgpu
    siblings carry, so validate_emit reported mock_passed=True for a
    non-functional ROCm kernel."""
    from helixc.ir.tile_ir import (
        TileOp, TileBlock, TileFn, TileModule, TileOpKind)
    fn = TileFn(
        name="mm_k", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.TILE_MATMUL),
            TileOp(kind=TileOpKind.RETURN),
        ])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["mm_k"] = fn
    text = HipEmitter().emit_module(tile_mod)
    # The emit still carries the real mnemonic AND the stub marker.
    assert "v_mfma_f32_16x16x16_f16" in text
    assert "HELIX-STUB-OPERANDS" in text
    result = validate_emit(text, BackendKind.ROCM_HIP)
    assert not result.mock_passed, (
        "validate_emit must flag an operand-less ROCm kernel as "
        "non-functional")
    assert not result.overall_passed()
    assert any("HELIX-STUB" in f or "non-functional" in f
               for f in result.mock_findings), result.mock_findings


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


def test_v24_real_hw_deferred_on_non_canonical_tool(monkeypatch):
    """v2.4 item 13 R1 audit-fix (code-review MEDIUM-1) — when a
    real-HW tool IS detected but it is not the backend's canonical
    dispatch tool (e.g. PTX via nvcc rather than ptxas, WebGPU via
    wgpu rather than naga), validate_emit takes the DEFERRED path:
    real_hw_passed=None + a 'dispatch deferred' finding +
    overall_status()==DEFERRED — it must NOT claim a pass.

    This replaces the old test_stage129_real_hw_deferred_to_stage_130,
    which became self-masking once ROCm/Metal/WebGPU dispatch was
    wired in v2.4 item 13 — that test asserted ROCm was 'still
    deferred' and only passed because llvm-mc happened to be absent
    on CI; on a machine WITH llvm-mc it would have failed.

    Deterministic via monkeypatch — does not depend on which tools
    are actually on PATH."""
    import helixc.backend.gpu_ci as gc
    # Force detect_tools to report a non-canonical tool for PTX
    # (nvcc — a real GPU_TOOLS[PTX] entry, but not the canonical
    # ptxas the dispatch table is keyed on).
    monkeypatch.setattr(gc, "detect_tools", lambda backend: ["nvcc"])

    from helixc.backend.ptx import PtxEmitter
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = PtxEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.PTX, attempt_real_hw=True)
    assert result.real_hw_attempted is True
    assert result.real_hw_tool == "nvcc"
    # DEFERRED — real_hw_passed is None, not a lying True.
    assert result.real_hw_passed is None
    assert result.overall_status() == OverallStatus.DEFERRED
    assert any("dispatch deferred" in f for f in result.real_hw_findings)


def test_v24_dispatch_ptxas_tool_not_found():
    """v2.4 item 13 — _dispatch_ptxas surfaces a missing/non-executable
    tool loudly as (False, [diagnostic]) rather than swallowing the
    FileNotFoundError into a silent pass. Deterministic — exercises
    the error path with a tool name guaranteed not to exist."""
    from helixc.backend.gpu_ci import _dispatch_ptxas
    passed, findings = _dispatch_ptxas(
        ".version 8.3\n", "helix_no_such_ptxas_xyz123")
    assert passed is False
    assert len(findings) == 1
    # v2.4 item 13 R1 audit-fix: dispatchers now catch OSError (not
    # just FileNotFoundError) and report "unusable at invocation
    # time (<ExcType>: ...)" — covers tool-not-found + not-executable
    # + OS spawn-refusal uniformly.
    assert "unusable at invocation time" in findings[0]
    assert "FileNotFoundError" in findings[0]


def test_v24_validate_emit_ptx_real_hw_tool_absent_is_deterministic():
    """v2.4 item 13 — when ptxas is not on PATH, PTX real-HW dispatch
    does not run: validate_emit returns real_hw_attempted=False with
    the post-init invariants intact (tool None, findings empty,
    passed None). Regression-pin: wiring the dispatch must not change
    behavior on a tool-less machine."""
    from helixc.backend.ptx import PtxEmitter
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = PtxEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.PTX, attempt_real_hw=True)
    if not requires_backend_tool(BackendKind.PTX):
        # No ptxas/nvcc on PATH — dispatch must not run.
        assert result.real_hw_attempted is False
        assert result.real_hw_tool is None
        assert result.real_hw_passed is None
        assert result.real_hw_findings == ()
    else:
        # A toolchain IS present — dispatch (or deferred-for-nvcc) ran.
        assert result.real_hw_attempted is True


@pytest.mark.skipif(
    "ptxas" not in detect_tools(BackendKind.PTX),
    reason="ptxas not on PATH — real-HW dispatch path cannot be exercised",
)
def test_v24_validate_emit_ptx_real_hw_dispatch_runs_ptxas():
    """v2.4 item 13 — when ptxas IS available, PTX real-HW dispatch
    actually invokes it. The emitted substrate PTX may or may not
    assemble cleanly (operand-less mnemonics until item 15 RegAlloc),
    so we assert only that dispatch RAN: real_hw_attempted=True,
    real_hw_tool='ptxas', and real_hw_passed is a concrete bool
    (not None — None would mean still-deferred)."""
    from helixc.backend.ptx import PtxEmitter
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = PtxEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.PTX, attempt_real_hw=True)
    assert result.real_hw_attempted is True
    assert result.real_hw_tool == "ptxas"
    assert isinstance(result.real_hw_passed, bool)  # concrete, not deferred


def test_v24_dispatch_naga_tool_not_found():
    """v2.4 item 13 slice 2 — _dispatch_naga surfaces a missing/non-
    executable tool loudly as (False, [diagnostic]) rather than
    swallowing the FileNotFoundError into a silent pass. Deterministic
    — exercises the error path with a tool name guaranteed absent."""
    from helixc.backend.gpu_ci import _dispatch_naga
    passed, findings = _dispatch_naga(
        "@compute @workgroup_size(64) fn k() {}\n",
        "helix_no_such_naga_xyz123")
    assert passed is False
    assert len(findings) == 1
    # v2.4 item 13 R1 audit-fix: dispatchers now catch OSError (not
    # just FileNotFoundError) and report "unusable at invocation
    # time (<ExcType>: ...)" — covers tool-not-found + not-executable
    # + OS spawn-refusal uniformly.
    assert "unusable at invocation time" in findings[0]
    assert "FileNotFoundError" in findings[0]


def test_v24_validate_emit_webgpu_real_hw_tool_absent_is_deterministic():
    """v2.4 item 13 slice 2 — when naga is not on PATH, WebGPU real-HW
    dispatch does not run: validate_emit returns real_hw_attempted=
    False with the post-init invariants intact. Regression-pin:
    wiring the naga dispatch must not change behavior on a tool-less
    machine."""
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = WgslEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.WEBGPU_WGSL,
                           attempt_real_hw=True)
    if not requires_backend_tool(BackendKind.WEBGPU_WGSL):
        # No naga/wgpu/dawn_node on PATH — dispatch must not run.
        assert result.real_hw_attempted is False
        assert result.real_hw_tool is None
        assert result.real_hw_passed is None
        assert result.real_hw_findings == ()
    else:
        # A toolchain IS present — dispatch (naga) or deferred
        # (wgpu/dawn_node) ran.
        assert result.real_hw_attempted is True


@pytest.mark.skipif(
    "naga" not in detect_tools(BackendKind.WEBGPU_WGSL),
    reason="naga not on PATH — WGSL real-HW dispatch cannot be exercised",
)
def test_v24_validate_emit_webgpu_real_hw_dispatch_runs_naga():
    """v2.4 item 13 slice 2 — when naga IS available, WebGPU real-HW
    dispatch actually invokes it. The emitted substrate WGSL may or
    may not validate cleanly (HELIX-STUB-OPERANDS markers until item
    15), so we assert only that dispatch RAN: real_hw_attempted=True,
    real_hw_tool='naga', real_hw_passed is a concrete bool."""
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = WgslEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.WEBGPU_WGSL,
                           attempt_real_hw=True)
    assert result.real_hw_attempted is True
    assert result.real_hw_tool == "naga"
    assert isinstance(result.real_hw_passed, bool)  # concrete, not deferred


def test_v24_rocm_gpu_tools_lists_llvm_mc_first():
    """v2.4 item 13 slice 3 — GPU_TOOLS[ROCM_HIP] lists `llvm-mc`
    before `hipcc`. llvm-mc assembles AMDGCN assembly text (what
    helixc.backend.rocm emits); hipcc compiles HIP C++ source (wrong
    input format). detect_tools preserves list order, so validate_emit
    picks llvm-mc as the real-HW tool whenever it is present."""
    assert GPU_TOOLS[BackendKind.ROCM_HIP][0] == "llvm-mc"
    assert "hipcc" in GPU_TOOLS[BackendKind.ROCM_HIP]


def test_v24_dispatch_llvm_mc_tool_not_found():
    """v2.4 item 13 slice 3 — _dispatch_llvm_mc surfaces a missing/
    non-executable tool loudly as (False, [diagnostic]) rather than
    swallowing the FileNotFoundError into a silent pass."""
    from helixc.backend.gpu_ci import _dispatch_llvm_mc
    passed, findings = _dispatch_llvm_mc(
        ".amdgcn_target \"amdgcn-amd-amdhsa--gfx942\"\n",
        "helix_no_such_llvmmc_xyz123")
    assert passed is False
    assert len(findings) == 1
    # v2.4 item 13 R1 audit-fix: dispatchers now catch OSError (not
    # just FileNotFoundError) and report "unusable at invocation
    # time (<ExcType>: ...)" — covers tool-not-found + not-executable
    # + OS spawn-refusal uniformly.
    assert "unusable at invocation time" in findings[0]
    assert "FileNotFoundError" in findings[0]


def test_v24_validate_emit_rocm_real_hw_tool_absent_is_deterministic():
    """v2.4 item 13 slice 3 — when neither llvm-mc nor hipcc is on
    PATH, ROCm real-HW dispatch does not run: validate_emit returns
    real_hw_attempted=False with post-init invariants intact."""
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = HipEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.ROCM_HIP, attempt_real_hw=True)
    if not requires_backend_tool(BackendKind.ROCM_HIP):
        assert result.real_hw_attempted is False
        assert result.real_hw_tool is None
        assert result.real_hw_passed is None
        assert result.real_hw_findings == ()
    else:
        assert result.real_hw_attempted is True


@pytest.mark.skipif(
    "llvm-mc" not in detect_tools(BackendKind.ROCM_HIP),
    reason="llvm-mc not on PATH — ROCm real-HW dispatch cannot be exercised",
)
def test_v24_validate_emit_rocm_real_hw_dispatch_runs_llvm_mc():
    """v2.4 item 13 slice 3 — when llvm-mc IS available, ROCm real-HW
    dispatch actually invokes it. The emitted substrate AMDGCN may or
    may not assemble cleanly (operand-less mnemonics until item 15),
    so we assert only that dispatch RAN: real_hw_attempted=True,
    real_hw_tool='llvm-mc', real_hw_passed is a concrete bool."""
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = HipEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.ROCM_HIP, attempt_real_hw=True)
    assert result.real_hw_attempted is True
    assert result.real_hw_tool == "llvm-mc"
    assert isinstance(result.real_hw_passed, bool)  # concrete, not deferred


def test_v24_dispatch_xcrun_metal_tool_not_found():
    """v2.4 item 13 slice 4 — _dispatch_xcrun_metal surfaces a
    missing/non-executable tool loudly as (False, [diagnostic])
    rather than swallowing the FileNotFoundError into a silent pass.
    On non-macOS this is the only reachable path (xcrun is mac-only),
    so the test exercises it directly with a guaranteed-absent name."""
    from helixc.backend.gpu_ci import _dispatch_xcrun_metal
    passed, findings = _dispatch_xcrun_metal(
        "#include <metal_stdlib>\nkernel void k() {}\n",
        "helix_no_such_xcrun_xyz123")
    assert passed is False
    assert len(findings) == 1
    # v2.4 item 13 R1 audit-fix: dispatchers now catch OSError (not
    # just FileNotFoundError) and report "unusable at invocation
    # time (<ExcType>: ...)" — covers tool-not-found + not-executable
    # + OS spawn-refusal uniformly.
    assert "unusable at invocation time" in findings[0]
    assert "FileNotFoundError" in findings[0]


@pytest.mark.parametrize("dispatch_name", [
    "_dispatch_ptxas",
    "_dispatch_naga",
    "_dispatch_llvm_mc",
    "_dispatch_xcrun_metal",
])
def test_v25_dispatch_tempfile_write_oserror_is_a_finding(
        monkeypatch, dispatch_name):
    """v2.5 polish (end-of-v2.4 5-clean-gate BE LOW-1) — all 4 real-HW
    dispatchers write the emitted kernel to a temp file inside the
    outer `try`, which carries only a `finally`. Pre-fix, an OSError
    from open()/write() (disk full, quota, a read-only or vanished
    tmpdir) escaped uncaught — an unhandled traceback out of
    validate_emit instead of a structured real-HW finding. The fix
    wraps the write in its own `try/except OSError`, parity with the
    existing subprocess OSError catch.

    Deterministic: monkeypatch tempfile.mkdtemp to hand back a
    non-existent directory path, so the kernel-file open() raises
    FileNotFoundError (an OSError subclass) — no real toolchain or
    disk-fault injection needed. The dispatcher must return
    (False, [one structured finding]), never raise."""
    import helixc.backend.gpu_ci as gc
    dispatch_fn = getattr(gc, dispatch_name)

    # A directory path whose parent does not exist. mkdtemp normally
    # creates the dir; the fake skips that, so open() inside it fails.
    bogus_dir = os.path.join(
        tempfile.gettempdir(), "helix_be_low1_absent_parent_zzz",
        "kernel_dir")

    def _fake_mkdtemp(*args, **kwargs):
        return bogus_dir
    monkeypatch.setattr(gc.tempfile, "mkdtemp", _fake_mkdtemp)

    passed, findings = dispatch_fn("kernel text", "any_tool_name")
    assert passed is False
    assert len(findings) == 1
    assert "could not write kernel temp file" in findings[0]
    # The OSError subclass name is surfaced (FileNotFoundError here).
    assert "FileNotFoundError" in findings[0]


def test_v24_validate_emit_metal_real_hw_tool_absent_is_deterministic():
    """v2.4 item 13 slice 4 — when xcrun is not on PATH (any non-macOS
    machine), Metal real-HW dispatch does not run: validate_emit
    returns real_hw_attempted=False with post-init invariants intact."""
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = MslEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.METAL_MSL, attempt_real_hw=True)
    if not requires_backend_tool(BackendKind.METAL_MSL):
        assert result.real_hw_attempted is False
        assert result.real_hw_tool is None
        assert result.real_hw_passed is None
        assert result.real_hw_findings == ()
    else:
        assert result.real_hw_attempted is True


@pytest.mark.skipif(
    "xcrun" not in detect_tools(BackendKind.METAL_MSL),
    reason="xcrun not on PATH (non-macOS) — Metal real-HW dispatch "
           "cannot be exercised",
)
def test_v24_validate_emit_metal_real_hw_dispatch_runs_xcrun():
    """v2.4 item 13 slice 4 — when xcrun IS available (macOS), Metal
    real-HW dispatch actually invokes the metal compiler. Substrate
    MSL may or may not compile cleanly (HELIX-STUB-OPERANDS markers
    until item 15), so we assert only that dispatch RAN:
    real_hw_attempted=True, real_hw_tool='xcrun', real_hw_passed is
    a concrete bool."""
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = MslEmitter().emit_module(tile_mod)

    result = validate_emit(text, BackendKind.METAL_MSL, attempt_real_hw=True)
    assert result.real_hw_attempted is True
    assert result.real_hw_tool == "xcrun"
    assert isinstance(result.real_hw_passed, bool)  # concrete, not deferred


def test_stage129_validate_emit_rejects_unknown_backend():
    """Stage 129 — validate_emit raises ValueError on a backend we don't
    have a validator for. Defends against silent acceptance of garbage."""
    # Use a string that bypasses the enum membership check.
    class FakeBackend:
        value = "not_a_real_backend"
    # v2.4 5-clean-gate TEST LOW-2 audit-fix: anchored on the
    # unknown-backend diagnostic so a future ValueError from an
    # unrelated arg-validation path can't make this test pass.
    with pytest.raises(ValueError, match="unknown backend"):
        validate_emit("anything", FakeBackend())  # type: ignore[arg-type]


# ============================================================================
# ValidationResult shape
# ============================================================================
def test_stage129_validation_result_overall_passed_logic():
    """Stage 129 — overall_passed() returns True iff mock passed AND
    real-HW either not attempted OR explicitly returned True.

    v2.2 polish (end-of-v2.1 audit BE LOW-2): overall_passed() is now
    STRICT — it returns False when real-HW dispatch was attempted but
    deferred (real_hw_passed=None). Callers that want deferred-tolerant
    behavior should branch on overall_status() / overall_deferred()
    explicitly. See test_stage129_overall_status_tri_state below."""
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

    # Mock passed, real-HW deferred (passed=None) → False under v2.2
    # tightening (was True under Stage 129 substrate semantics, which
    # silently equated DEFERRED with PASSED). DEFERRED is its own state;
    # callers must opt into deferred-tolerance via overall_status().
    r4 = ValidationResult(
        backend=BackendKind.PTX, mock_passed=True, mock_findings=(),
        real_hw_attempted=True, real_hw_passed=None,
        real_hw_tool="ptxas", real_hw_findings=("deferred",),
    )
    assert not r4.overall_passed()
    assert r4.overall_deferred()
    assert r4.overall_status() == OverallStatus.DEFERRED


def test_stage129_overall_status_tri_state():
    """v2.2 polish (BE LOW-2): overall_status() is a tri-state
    PASSED / FAILED / DEFERRED, distinguishing real-HW-deferred from
    real-HW-actually-passed so callers cannot mistake "we didn't check"
    for "we checked and it's fine"."""
    # PASSED: mock OK, real-HW not attempted
    r_passed_no_hw = ValidationResult(
        backend=BackendKind.PTX, mock_passed=True, mock_findings=(),
        real_hw_attempted=False, real_hw_passed=None,
        real_hw_tool=None, real_hw_findings=(),
    )
    assert r_passed_no_hw.overall_status() == OverallStatus.PASSED
    assert r_passed_no_hw.overall_passed()
    assert not r_passed_no_hw.overall_deferred()

    # PASSED: mock OK, real-HW attempted + explicitly True
    r_passed_hw = ValidationResult(
        backend=BackendKind.PTX, mock_passed=True, mock_findings=(),
        real_hw_attempted=True, real_hw_passed=True,
        real_hw_tool="ptxas", real_hw_findings=(),
    )
    assert r_passed_hw.overall_status() == OverallStatus.PASSED
    assert r_passed_hw.overall_passed()

    # FAILED: mock failed
    r_failed_mock = ValidationResult(
        backend=BackendKind.PTX, mock_passed=False, mock_findings=("x",),
        real_hw_attempted=False, real_hw_passed=None,
        real_hw_tool=None, real_hw_findings=(),
    )
    assert r_failed_mock.overall_status() == OverallStatus.FAILED
    assert not r_failed_mock.overall_passed()
    assert not r_failed_mock.overall_deferred()

    # FAILED: mock OK, real-HW attempted + False
    r_failed_hw = ValidationResult(
        backend=BackendKind.PTX, mock_passed=True, mock_findings=(),
        real_hw_attempted=True, real_hw_passed=False,
        real_hw_tool="ptxas", real_hw_findings=("bad ptx",),
    )
    assert r_failed_hw.overall_status() == OverallStatus.FAILED
    assert not r_failed_hw.overall_passed()
    assert not r_failed_hw.overall_deferred()

    # DEFERRED: mock OK, real-HW attempted but real_hw_passed=None
    r_deferred = ValidationResult(
        backend=BackendKind.PTX, mock_passed=True, mock_findings=(),
        real_hw_attempted=True, real_hw_passed=None,
        real_hw_tool="ptxas", real_hw_findings=("Stage 130 deferred",),
    )
    assert r_deferred.overall_status() == OverallStatus.DEFERRED
    assert not r_deferred.overall_passed()
    assert r_deferred.overall_deferred()


def test_stage129_overall_status_enum_members():
    """v2.2 polish: OverallStatus is a closed enum with exactly three
    members so dispatch / match statements over it are exhaustive."""
    values = {s.value for s in OverallStatus}
    assert values == {"passed", "failed", "deferred"}


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


def test_v24_validate_emit_flags_helix_stub_directive():
    """v2.4 5-clean-gate BE MEDIUM-1 audit-fix — a kernel that is
    structurally well-formed (passes the header/attribute/terminator
    shape-check) but carries a `.error "HELIX-STUB..."` directive —
    a stub/deferred tile-IR op reached codegen — must NOT pass mock
    validation. Pre-fix the shape-only mock validators returned
    mock_passed=True for such non-functional kernels."""
    # Structurally-complete PTX (all 4 mock shape-tokens present) but
    # with a HELIX-STUB directive in the body.
    stub_ptx = (
        ".version 8.3\n"
        ".target sm_75\n"
        ".address_size 64\n"
        ".visible .entry k()\n"
        "{\n"
        '    .error "HELIX-STUB: TileOpKind.TILE_REDUCE status=\'stub\'"\n'
        "    ret;\n"
        "}\n"
    )
    result = validate_emit(stub_ptx, BackendKind.PTX)
    assert not result.mock_passed
    assert any("HELIX-STUB" in f or "non-functional" in f
               for f in result.mock_findings)


def test_v24_validate_emit_flags_helix_skipped_directive():
    """v2.4 5-clean-gate BE MEDIUM-1 audit-fix — same for a
    `HELIX-SKIPPED` directive (a skipped op — no analog on the
    target — routed to codegen)."""
    stub_ptx = (
        ".version 8.3\n.target sm_75\n.address_size 64\n"
        ".visible .entry k()\n{\n"
        '    .error "HELIX-SKIPPED: TileOpKind.TMA_LOAD"\n'
        "    ret;\n}\n"
    )
    result = validate_emit(stub_ptx, BackendKind.PTX)
    assert not result.mock_passed
    assert any("HELIX-SKIPPED" in f or "non-functional" in f
               for f in result.mock_findings)


def test_v24_validate_emit_clean_kernel_still_passes():
    """v2.4 5-clean-gate BE MEDIUM-1 audit-fix — regression-pin: a
    genuinely clean kernel (no HELIX markers) still passes mock
    validation. The stub-scan must not false-positive."""
    from helixc.backend.ptx import PtxEmitter
    src = "@kernel fn k() {}"
    prog = parse(src)
    tile_mod = lower_to_tile(lower(prog))
    text = PtxEmitter().emit_module(tile_mod)
    assert "HELIX-STUB" not in text and "HELIX-SKIPPED" not in text
    result = validate_emit(text, BackendKind.PTX)
    assert result.mock_passed, result.mock_findings


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


def test_v24_5clean_real_hw_failure_must_carry_a_diagnostic():
    """v2.4 end-of-cycle 5-clean-gate BE audit-fix — a real-HW
    FAILURE with no findings is a silent failure (a failure with no
    explanation). __post_init__ rejects it, mirroring the existing
    mock_passed/mock_findings invariant on the real-HW side."""
    # real_hw_attempted=True, real_hw_passed=False, but no findings → illegal.
    with pytest.raises(ValueError, match="real_hw_findings is empty"):
        ValidationResult(
            backend=BackendKind.PTX, mock_passed=True, mock_findings=(),
            real_hw_attempted=True, real_hw_passed=False,
            real_hw_tool="ptxas", real_hw_findings=(),
        )
    # A real-HW failure that DOES carry a diagnostic is legal.
    ok = ValidationResult(
        backend=BackendKind.PTX, mock_passed=True, mock_findings=(),
        real_hw_attempted=True, real_hw_passed=False,
        real_hw_tool="ptxas", real_hw_findings=("ptxas exit 1: bad op",),
    )
    assert ok.overall_status() == OverallStatus.FAILED
