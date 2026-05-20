"""
helixc/backend/gpu_ci.py — Stage 129 (v2.0 Phase A.1).

GPU CI scaffolding for v2.0 — mock-GPU validation infrastructure that
runs WITHOUT real GPU hardware. Validates emitted text is syntactically
plausible (header present, kernel attribute, terminator) and that
the per-backend op-mapping tables stay in sync with the tile-IR.

Real-HW validation (running emitted PTX/HIP/MSL/WGSL on actual GPUs)
is deferred to Stage 130+, which requires:
- nvcc/ptxas (NVIDIA)
- hipcc (AMD)
- xcrun metal (Apple, mac-only)
- A WebGPU runtime (deno-webgpu, dawn, browser playground)

The scaffolding shipped here is the harness that wraps mock and real
validation behind the same `validate_emit()` interface so v2.0
real-HW gates can be added in a later stage without test surface
churn.

Per v2.0 research Report 1 + Report 5: "GPU CI is the prerequisite
before any real-HW validation; mock validation catches 80% of
codegen drift without burning CI minutes on actual hardware."

License: Apache 2.0
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BackendKind(Enum):
    """Identifies which backend emitted a given text. Validators
    dispatch on this enum."""
    PTX = "ptx"           # NVIDIA — helixc/backend/ptx.py
    ROCM_HIP = "rocm_hip"  # AMD — helixc/backend/rocm.py
    METAL_MSL = "metal_msl"  # Apple — helixc/backend/metal.py
    WEBGPU_WGSL = "webgpu_wgsl"  # browser — helixc/backend/webgpu.py


class OverallStatus(Enum):
    """Tri-state outcome — v2.2 polish (end-of-v2.1 audit BE LOW-2):
    distinguishes 'real-HW dispatch returned PASS' from 'real-HW
    dispatch was deferred to Stage 130+ so we don't actually know'.

    Stage 129 originally collapsed DEFERRED into PASSED in
    `overall_passed()`, which the v2.1 5-clean-gate flagged as a
    silent-failure surface (a caller acting on overall_passed=True
    after `attempt_real_hw=True` could not tell whether real-HW
    actually validated the emit or whether it punted).
    """
    PASSED = "passed"
    FAILED = "failed"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a mock or real-HW validation pass.

    Stage 129 type-design audit-fix (2026-05-19): frozen + tuple-backed
    so results cannot be mutated post-construction (aliasing-bug class
    eliminated). `__post_init__` enforces cross-field invariants that
    were previously representable-but-illegal:
    - mock_passed must agree with len(mock_findings) == 0
    - real_hw_attempted=False must imply tool is None + findings empty
    """
    backend: BackendKind
    mock_passed: bool
    mock_findings: tuple[str, ...]
    real_hw_attempted: bool
    real_hw_passed: Optional[bool]
    real_hw_tool: Optional[str]
    real_hw_findings: tuple[str, ...]

    def __post_init__(self) -> None:
        # mock_passed must match the findings list emptiness.
        if self.mock_passed != (len(self.mock_findings) == 0):
            raise ValueError(
                f"ValidationResult: mock_passed={self.mock_passed} "
                f"disagrees with mock_findings emptiness "
                f"(len={len(self.mock_findings)}) — Stage 129 audit-fix"
            )
        # If real-HW not attempted, tool + findings must be absent.
        if not self.real_hw_attempted:
            if self.real_hw_tool is not None:
                raise ValueError(
                    f"ValidationResult: real_hw_attempted=False but "
                    f"real_hw_tool={self.real_hw_tool!r} — illegal "
                    f"combination"
                )
            if len(self.real_hw_findings) != 0:
                raise ValueError(
                    f"ValidationResult: real_hw_attempted=False but "
                    f"real_hw_findings has entries — illegal combination"
                )
            if self.real_hw_passed is not None:
                raise ValueError(
                    f"ValidationResult: real_hw_attempted=False but "
                    f"real_hw_passed={self.real_hw_passed!r} — illegal"
                )

    def overall_status(self) -> OverallStatus:
        """v2.2 polish: tri-state outcome distinguishing PASSED from
        DEFERRED (real-HW dispatch present but punted to Stage 130+).

        - FAILED if mock failed, or real-HW attempted + returned False
        - DEFERRED if mock passed + real-HW attempted + real_hw_passed is None
          (the Stage 129 substrate ships the harness but actual tool
          dispatch is wired in Stage 130+, so real-HW results are
          honestly None rather than spuriously True)
        - PASSED otherwise (mock passed AND either real-HW not
          attempted OR real-HW attempted + explicitly True)
        """
        if not self.mock_passed:
            return OverallStatus.FAILED
        if self.real_hw_attempted:
            if self.real_hw_passed is False:
                return OverallStatus.FAILED
            if self.real_hw_passed is None:
                return OverallStatus.DEFERRED
        return OverallStatus.PASSED

    def overall_passed(self) -> bool:
        """True ONLY if overall_status() is PASSED. v2.2 polish (end-of-
        v2.1 audit BE LOW-2): tightened from the Stage 129 substrate
        semantic, which silently equated DEFERRED with PASSED. Callers
        that explicitly want deferred-tolerant behavior should test
        `overall_status() != OverallStatus.FAILED` instead, making the
        deferred-acceptance choice visible at the call site."""
        return self.overall_status() == OverallStatus.PASSED

    def overall_deferred(self) -> bool:
        """True if mock passed but real-HW dispatch is deferred (Stage
        129 substrate behavior — real_hw_passed is None despite
        real_hw_attempted=True). v2.2 polish predicate so callers can
        branch on DEFERRED without re-deriving the condition."""
        return self.overall_status() == OverallStatus.DEFERRED


# ============================================================================
# Tool detection
# ============================================================================
# CI runners discover what's available; tests that require absent
# toolchains are skipped (via pytest marker indirection in test_gpu_ci.py).
GPU_TOOLS: dict[BackendKind, list[str]] = {
    BackendKind.PTX: ["ptxas", "nvcc"],
    BackendKind.ROCM_HIP: ["hipcc"],
    BackendKind.METAL_MSL: ["xcrun"],  # macOS-only; also need `xcrun -find metal`
    BackendKind.WEBGPU_WGSL: ["naga", "wgpu", "dawn_node"],  # any of these
}


def detect_tools(backend: BackendKind) -> list[str]:
    """Return the subset of GPU_TOOLS[backend] that are available on PATH.

    Empty list = no real-HW validation possible for this backend on the
    current machine (mock validation still runs).
    """
    return [tool for tool in GPU_TOOLS.get(backend, [])
            if shutil.which(tool) is not None]


# ============================================================================
# Mock validators (shape-check only — no real-HW dispatch)
# ============================================================================

def _validate_ptx(text: str) -> list[str]:
    """Mock PTX validation: look for .version / .target / .address_size
    headers + at least one kernel-shaped entry."""
    findings: list[str] = []
    if not re.search(r"\.version\s+\d", text):
        findings.append("missing .version directive")
    if not re.search(r"\.target\s+sm_", text):
        findings.append("missing .target directive (sm_NN expected)")
    if not re.search(r"\.address_size\s+(32|64)", text):
        findings.append("missing .address_size directive")
    if ".entry" not in text and ".visible .entry" not in text:
        findings.append("no kernel entry (.entry or .visible .entry) found")
    return findings


def _validate_rocm_hip(text: str) -> list[str]:
    """Mock ROCm/HIP validation: AMDGPU target triple + s_endpgm terminator."""
    findings: list[str] = []
    if ".amdgcn_target" not in text:
        findings.append("missing .amdgcn_target directive")
    if "amdgcn-amd-amdhsa" not in text:
        findings.append("missing AMDGPU object format triple")
    if "s_endpgm" not in text:
        findings.append("no s_endpgm kernel terminator found")
    return findings


def _validate_metal_msl(text: str) -> list[str]:
    """Mock MSL validation: metal_stdlib include + kernel attribute."""
    findings: list[str] = []
    if "metal_stdlib" not in text:
        findings.append("missing #include <metal_stdlib>")
    if "using namespace metal" not in text:
        findings.append("missing `using namespace metal`")
    if "kernel void" not in text:
        findings.append("no `kernel void` function declaration found")
    return findings


def _validate_webgpu_wgsl(text: str) -> list[str]:
    """Mock WGSL validation: compute attribute + workgroup_size + fn decl."""
    findings: list[str] = []
    if "@compute" not in text:
        findings.append("missing @compute attribute")
    if "@workgroup_size" not in text:
        findings.append("missing @workgroup_size attribute")
    if not re.search(r"\bfn\s+\w+\s*\(", text):
        findings.append("no WGSL `fn NAME(...)` declaration found")
    return findings


_MOCK_VALIDATORS = {
    BackendKind.PTX: _validate_ptx,
    BackendKind.ROCM_HIP: _validate_rocm_hip,
    BackendKind.METAL_MSL: _validate_metal_msl,
    BackendKind.WEBGPU_WGSL: _validate_webgpu_wgsl,
}


# ============================================================================
# Real-HW dispatch (v2.4 item 13)
# ============================================================================
# Default NVIDIA arch for ptxas. Matches helixc.backend.ptx.DEFAULT_TARGET
# (sm_75 Turing baseline). A caller targeting a newer GPU can override.
DEFAULT_PTXAS_ARCH = "sm_75"

# Hard cap on a single real-HW tool invocation. A hung assembler must
# not stall a CI run — TimeoutExpired is caught and surfaced as a
# finding, not a swallowed silent pass.
REAL_HW_TIMEOUT_S = 30


def _dispatch_ptxas(text: str, tool: str,
                    gpu_arch: str = DEFAULT_PTXAS_ARCH,
                    timeout_s: int = REAL_HW_TIMEOUT_S) -> tuple[bool, list[str]]:
    """v2.4 item 13 — real-HW dispatch for the PTX backend.

    Writes the emitted PTX to a temp file and runs `ptxas` on it.
    ptxas assembles PTX text directly to a cubin and exits non-zero
    with a stderr diagnostic on any assembly error — so this is a
    genuine syntax+semantics gate, not a mock string-check.

    Returns (passed, findings): passed=True iff ptxas exited 0;
    findings carries the ptxas diagnostic (truncated) on failure,
    the timeout message on a hang, or a tool-not-found message if
    `tool` vanished between detect_tools() and dispatch.

    NOTE: Helix's current PTX emit is substrate-level — operand-less
    mnemonics for several tile ops (real RegAlloc is v2.4 item 15).
    ptxas will legitimately reject such kernels; that is the honest,
    correct outcome and exactly what this gate exists to surface.
    Once item 15 lands, the same dispatch starts reporting passes.

    This slice wires PTX only; ROCm/Metal/WebGPU real-HW dispatch
    land in subsequent item-13 slices (their tool<->text-format
    matchups need llvm-mc / xcrun-metal / naga respectively).
    """
    tmpdir = tempfile.mkdtemp(prefix="helix_ptxas_")
    in_path = os.path.join(tmpdir, "kernel.ptx")
    out_path = os.path.join(tmpdir, "kernel.cubin")
    try:
        with open(in_path, "w", encoding="utf-8") as f:
            f.write(text)
        cmd = [tool, in_path, "-o", out_path, f"--gpu-name={gpu_arch}"]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, [
                f"ptxas dispatch timed out after {timeout_s}s "
                f"(tool={tool!r}, arch={gpu_arch})"
            ]
        except FileNotFoundError:
            # detect_tools() saw the tool on PATH but it vanished (or
            # is not executable) by dispatch time. Surface loudly —
            # do not silently fall back to a mock pass.
            return False, [
                f"ptxas dispatch: tool {tool!r} not found / not "
                f"executable at invocation time"
            ]
        if proc.returncode != 0:
            diag = (proc.stderr or proc.stdout or "").strip()
            return False, [
                f"ptxas exit {proc.returncode}: {diag[:500]}"
            ]
        return True, []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _dispatch_naga(text: str, tool: str,
                   timeout_s: int = REAL_HW_TIMEOUT_S) -> tuple[bool, list[str]]:
    """v2.4 item 13 (slice 2) — real-HW dispatch for the WebGPU backend.

    Writes the emitted WGSL to a temp file and runs `naga` on it.
    Given a single `.wgsl` input, naga parses + validates the module
    and exits non-zero with a stderr diagnostic on any parse or
    validation error — a genuine WGSL syntax+semantics gate, parity
    with `_dispatch_ptxas` for PTX.

    Returns (passed, findings): passed=True iff naga exited 0;
    findings carries the truncated naga diagnostic on failure, the
    timeout message on a hang, or a tool-not-found message if `tool`
    vanished between detect_tools() and dispatch.

    NOTE: Helix's WGSL emit is substrate-level — TILE_MATMUL and the
    memory ops carry `HELIX-STUB-OPERANDS` markers (a_tile/buf_in/...
    not bound; real binding is v2.4 item 15). naga will legitimately
    reject such kernels with "unresolved identifier"; that is the
    honest, correct outcome this gate exists to surface. An empty
    `@compute` kernel validates cleanly today.
    """
    tmpdir = tempfile.mkdtemp(prefix="helix_naga_")
    in_path = os.path.join(tmpdir, "kernel.wgsl")
    try:
        with open(in_path, "w", encoding="utf-8") as f:
            f.write(text)
        cmd = [tool, in_path]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, [
                f"naga dispatch timed out after {timeout_s}s "
                f"(tool={tool!r})"
            ]
        except FileNotFoundError:
            return False, [
                f"naga dispatch: tool {tool!r} not found / not "
                f"executable at invocation time"
            ]
        if proc.returncode != 0:
            diag = (proc.stderr or proc.stdout or "").strip()
            return False, [
                f"naga exit {proc.returncode}: {diag[:500]}"
            ]
        return True, []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# Stage 129 type-design audit-fix: module-load drift detector.
# Catches the case where a new BackendKind is added without updating
# GPU_TOOLS or _MOCK_VALIDATORS. Same pattern as the per-backend
# coverage checks in rocm/metal/webgpu/tile_ir_audit.
def _check_gpu_ci_drift() -> None:
    expected = set(BackendKind)
    if set(GPU_TOOLS) != expected:
        raise AssertionError(
            f"helixc.backend.gpu_ci: GPU_TOOLS keys "
            f"{set(GPU_TOOLS)} != BackendKind members {expected}"
        )
    if set(_MOCK_VALIDATORS) != expected:
        raise AssertionError(
            f"helixc.backend.gpu_ci: _MOCK_VALIDATORS keys "
            f"{set(_MOCK_VALIDATORS)} != BackendKind members {expected}"
        )


_check_gpu_ci_drift()


def validate_emit(text: str, backend: BackendKind,
                  attempt_real_hw: bool = False) -> ValidationResult:
    """Stage 129 — validate emitted text against the backend's expected
    shape. Two phases:

    1. Mock validation (always runs; no hardware required):
       - Backend-specific header / attribute / terminator checks
       - Returns a list of findings (empty = pass)

    2. Real-HW validation (only if attempt_real_hw=True AND tools detected):
       - Runs ptxas / hipcc / xcrun metal / naga on the text
       - Stage 129 ships the harness only; actual tool-invocation
         shells are wired in Stage 130+

    Returns a ValidationResult capturing both outcomes.
    """
    if backend not in _MOCK_VALIDATORS:
        raise ValueError(
            f"validate_emit: unknown backend {backend!r}; "
            f"expected one of {list(_MOCK_VALIDATORS.keys())}"
        )
    mock_findings = _MOCK_VALIDATORS[backend](text)
    mock_passed = (len(mock_findings) == 0)

    real_hw_attempted = False
    real_hw_passed: Optional[bool] = None
    real_hw_tool: Optional[str] = None
    real_hw_findings: list[str] = []

    if attempt_real_hw:
        available = detect_tools(backend)
        if available:
            real_hw_attempted = True
            real_hw_tool = available[0]
            # v2.4 item 13 — real-HW dispatch. PTX via ptxas (slice 1)
            # and WebGPU via naga (slice 2) are wired — both tools
            # validate their target text directly. ROCm/Metal remain
            # deferred until their slices land: ROCm AMDGCN assembly
            # needs llvm-mc (not the `hipcc` GPU_TOOLS entry, which
            # compiles HIP C++), and Metal MSL needs `xcrun -sdk
            # macosx metal` (mac-only).
            if backend is BackendKind.PTX and real_hw_tool == "ptxas":
                passed, dispatch_findings = _dispatch_ptxas(
                    text, real_hw_tool)
                real_hw_passed = passed
                real_hw_findings.extend(dispatch_findings)
            elif (backend is BackendKind.WEBGPU_WGSL
                  and real_hw_tool == "naga"):
                passed, dispatch_findings = _dispatch_naga(
                    text, real_hw_tool)
                real_hw_passed = passed
                real_hw_findings.extend(dispatch_findings)
            else:
                # Deferred: tool detected but dispatch not yet wired
                # for this backend. `real_hw_passed = None` (deferred)
                # rather than True so the result doesn't lie about
                # coverage — `overall_status()` maps this to DEFERRED.
                real_hw_passed = None
                real_hw_findings.append(
                    f"v2.4 item 13: real-HW tool '{real_hw_tool}' "
                    f"detected but dispatch for backend "
                    f"{backend.name} not yet wired"
                )

    return ValidationResult(
        backend=backend,
        mock_passed=mock_passed,
        mock_findings=tuple(mock_findings),
        real_hw_attempted=real_hw_attempted,
        real_hw_passed=real_hw_passed,
        real_hw_tool=real_hw_tool,
        real_hw_findings=tuple(real_hw_findings),
    )


# ============================================================================
# Pytest marker indirection (used by helixc/tests/test_gpu_ci.py)
# ============================================================================
def requires_backend_tool(backend: BackendKind) -> bool:
    """True if at least one tool for the backend is on PATH. Tests
    use this to skip real-HW assertions when CI lacks the toolchain.

    Stage 129 substrate: shipped as a predicate so test_gpu_ci.py can
    bind a `pytest.mark.skipif(not requires_backend_tool(...), ...)`
    decorator. Real per-tool detection lives in detect_tools().
    """
    return len(detect_tools(backend)) > 0
