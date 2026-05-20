"""
helixc/backend/gpu_ci.py — Stage 129 (v2.0 Phase A.1).

GPU CI scaffolding for v2.0 — GPU validation infrastructure with two
phases behind one `validate_emit()` interface:

1. Mock validation (always runs, no hardware/toolchain needed):
   shape-checks the emitted text (header, kernel attribute, terminator)
   and confirms the per-backend op-mapping tables stay in sync with
   the tile-IR.

2. Real-HW dispatch (v2.4 item 13 — wired for all 4 backends): runs
   the real toolchain on the emitted text —
   - ptxas       (PTX,    NVIDIA)
   - llvm-mc     (ROCm,   AMDGCN assembly)
   - xcrun metal (Metal,  Apple, mac-only)
   - naga        (WebGPU, WGSL)
   A non-zero tool exit becomes real_hw_passed=False + a finding.
   When the canonical tool is absent, mock validation still runs and
   real_hw_attempted=False.

Stage 129 shipped phase 1 + the harness; v2.4 item 13 wired phase 2.

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
from typing import Callable, Optional


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
    # v2.4 item 13 slice 3: `llvm-mc` listed first — it assembles the
    # AMDGCN *assembly text* helixc.backend.rocm emits. `hipcc`
    # compiles HIP *C++ source*, a different input format; it is kept
    # as a fallback-only entry (real-HW dispatch is wired for llvm-mc).
    BackendKind.ROCM_HIP: ["llvm-mc", "hipcc"],
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
# v2.4 item 13 R1 audit-fix (type-design Finding 1): the ptxas arch
# and AMDGCN mcpu were physical string copies of ptx.DEFAULT_TARGET /
# rocm.DEFAULT_TARGET, kept in sync only by a comment. If a backend
# bumped its baseline (ptx.py already anticipates sm_120) the CI
# dispatcher would silently assemble against a stale arch — a green
# gate not reflecting emitted code. R1 fix: import the backend
# constants so there is one source of truth. (ptx.py / rocm.py do
# not import gpu_ci — no circular-import risk.)
from .ptx import DEFAULT_TARGET as _PTX_DEFAULT_TARGET  # noqa: E402
from .rocm import DEFAULT_TARGET as _ROCM_DEFAULT_TARGET  # noqa: E402

# Default NVIDIA arch for ptxas — single source of truth: ptx.py.
DEFAULT_PTXAS_ARCH = _PTX_DEFAULT_TARGET

# Default AMDGCN target for llvm-mc — single source of truth: rocm.py.
DEFAULT_AMDGCN_MCPU = _ROCM_DEFAULT_TARGET

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

    One of four real-HW dispatchers (ptxas / naga / llvm-mc /
    xcrun-metal) — all wired as of v2.4 item 13.
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
        except OSError as e:
            # v2.4 item 13 R1 audit-fix (silent-failure MEDIUM): catch
            # OSError, not just FileNotFoundError. detect_tools() saw
            # the tool on PATH but it vanished, is not executable
            # (PermissionError), or the OS refused to spawn it
            # (ENOEXEC / E2BIG — all OSError) by dispatch time. All of
            # these must surface as a loud structured finding, not an
            # uncaught traceback out of validate_emit, and never a
            # silent fall-back to a mock pass.
            return False, [
                f"ptxas dispatch: tool {tool!r} unusable at "
                f"invocation time ({type(e).__name__}: {e})"
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
        except OSError as e:
            # v2.4 item 13 R1 audit-fix: catch OSError (covers
            # FileNotFoundError + PermissionError + spawn failures).
            return False, [
                f"naga dispatch: tool {tool!r} unusable at "
                f"invocation time ({type(e).__name__}: {e})"
            ]
        if proc.returncode != 0:
            diag = (proc.stderr or proc.stdout or "").strip()
            return False, [
                f"naga exit {proc.returncode}: {diag[:500]}"
            ]
        return True, []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _dispatch_llvm_mc(text: str, tool: str,
                      mcpu: str = DEFAULT_AMDGCN_MCPU,
                      timeout_s: int = REAL_HW_TIMEOUT_S) -> tuple[bool, list[str]]:
    """v2.4 item 13 (slice 3) — real-HW dispatch for the ROCm backend.

    Writes the emitted AMDGCN assembly to a temp file and runs
    `llvm-mc` on it with the amdgcn-amd-amdhsa triple. llvm-mc
    assembles the `.s` text to an object file and exits non-zero with
    a stderr diagnostic on any assembly error — a genuine AMDGCN
    syntax gate, parity with `_dispatch_ptxas` (PTX) and
    `_dispatch_naga` (WGSL).

    `hipcc` (the other ROCM_HIP GPU_TOOLS entry) is NOT used for
    dispatch: it compiles HIP C++ source, not raw AMDGCN assembly —
    the wrong input format for helixc.backend.rocm's text emit.

    Returns (passed, findings): passed=True iff llvm-mc exited 0;
    findings carries the truncated diagnostic on failure, the timeout
    message on a hang, or a tool-not-found message if `tool` vanished
    between detect_tools() and dispatch.

    NOTE: Helix's ROCm emit is substrate-level — operand-less MFMA /
    memory mnemonics until v2.4 item 15 RegAlloc. llvm-mc will
    legitimately reject such kernels; that honest outcome is exactly
    what this gate surfaces.
    """
    tmpdir = tempfile.mkdtemp(prefix="helix_llvmmc_")
    in_path = os.path.join(tmpdir, "kernel.s")
    out_path = os.path.join(tmpdir, "kernel.o")
    try:
        with open(in_path, "w", encoding="utf-8") as f:
            f.write(text)
        cmd = [
            tool,
            "-triple=amdgcn-amd-amdhsa",
            f"-mcpu={mcpu}",
            "-filetype=obj",
            in_path,
            "-o", out_path,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, [
                f"llvm-mc dispatch timed out after {timeout_s}s "
                f"(tool={tool!r}, mcpu={mcpu})"
            ]
        except OSError as e:
            # v2.4 item 13 R1 audit-fix: catch OSError (covers
            # FileNotFoundError + PermissionError + spawn failures).
            return False, [
                f"llvm-mc dispatch: tool {tool!r} unusable at "
                f"invocation time ({type(e).__name__}: {e})"
            ]
        if proc.returncode != 0:
            diag = (proc.stderr or proc.stdout or "").strip()
            return False, [
                f"llvm-mc exit {proc.returncode}: {diag[:500]}"
            ]
        return True, []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _dispatch_xcrun_metal(text: str, tool: str,
                          timeout_s: int = REAL_HW_TIMEOUT_S) -> tuple[bool, list[str]]:
    """v2.4 item 13 (slice 4) — real-HW dispatch for the Metal backend.

    Writes the emitted MSL to a temp `.metal` file and compiles it
    with `xcrun -sdk macosx metal -c`. The Metal front-end compiles
    MSL to AIR (Apple IR) and exits non-zero with a stderr diagnostic
    on any compile error — a genuine MSL syntax+semantics gate,
    parity with the ptxas/naga/llvm-mc dispatchers.

    macOS-only: `xcrun` exists only on macOS with Xcode installed.
    On any other platform detect_tools() returns [] and this is never
    called — so a Linux/Windows CI run sees the deterministic
    tool-absent path, not a failure.

    Returns (passed, findings): passed=True iff the metal compiler
    exited 0; findings carries the truncated diagnostic on failure,
    the timeout message on a hang, or a tool-not-found message if
    `tool` vanished between detect_tools() and dispatch.

    NOTE: Helix's MSL emit is substrate-level — TILE_MATMUL +
    memory ops carry `HELIX-STUB-OPERANDS` markers (matrix args /
    buffers not bound; real binding is v2.4 item 15 RegAlloc). The
    metal compiler will legitimately reject such kernels — the
    honest outcome this gate surfaces. An empty `kernel void`
    compiles cleanly.
    """
    tmpdir = tempfile.mkdtemp(prefix="helix_xcrun_metal_")
    in_path = os.path.join(tmpdir, "kernel.metal")
    out_path = os.path.join(tmpdir, "kernel.air")
    try:
        with open(in_path, "w", encoding="utf-8") as f:
            f.write(text)
        cmd = [tool, "-sdk", "macosx", "metal", "-c", in_path,
               "-o", out_path]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, [
                f"xcrun metal dispatch timed out after {timeout_s}s "
                f"(tool={tool!r})"
            ]
        except OSError as e:
            # v2.4 item 13 R1 audit-fix: catch OSError (covers
            # FileNotFoundError + PermissionError + spawn failures).
            return False, [
                f"xcrun metal dispatch: tool {tool!r} unusable at "
                f"invocation time ({type(e).__name__}: {e})"
            ]
        if proc.returncode != 0:
            diag = (proc.stderr or proc.stdout or "").strip()
            return False, [
                f"xcrun metal exit {proc.returncode}: {diag[:500]}"
            ]
        return True, []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# v2.4 item 13 R1 audit-fix (type-design Finding 2): real-HW dispatch
# table — restores parity with the _MOCK_VALIDATORS table. Maps each
# BackendKind to its (canonical_tool, dispatch_fn) pair. validate_emit
# looks this up instead of a hardcoded if/elif chain, and
# _check_gpu_ci_drift() asserts the keyset matches BackendKind — so a
# 5th backend that updates GPU_TOOLS + _MOCK_VALIDATORS but forgets
# real-HW dispatch now fails loudly at module load instead of always
# silently taking the deferred path.
#
# `canonical_tool` is the ONE tool name (of possibly several in
# GPU_TOOLS[backend]) for which real-HW dispatch is wired. A detected-
# but-non-canonical tool (PTX via nvcc, WebGPU via wgpu/dawn_node)
# still takes the deferred branch in validate_emit.
_DispatchFn = Callable[[str, str], "tuple[bool, list[str]]"]
_REAL_HW_DISPATCH: dict[BackendKind, tuple[str, _DispatchFn]] = {
    BackendKind.PTX:         ("ptxas",   _dispatch_ptxas),
    BackendKind.WEBGPU_WGSL: ("naga",    _dispatch_naga),
    BackendKind.ROCM_HIP:    ("llvm-mc", _dispatch_llvm_mc),
    BackendKind.METAL_MSL:   ("xcrun",   _dispatch_xcrun_metal),
}


# Stage 129 type-design audit-fix: module-load drift detector.
# Catches the case where a new BackendKind is added without updating
# GPU_TOOLS / _MOCK_VALIDATORS / _REAL_HW_DISPATCH. Same pattern as
# the per-backend coverage checks in rocm/metal/webgpu/tile_ir_audit.
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
    # v2.4 item 13 R1 audit-fix: real-HW dispatch table must also
    # cover every backend (type-design Finding 2).
    if set(_REAL_HW_DISPATCH) != expected:
        raise AssertionError(
            f"helixc.backend.gpu_ci: _REAL_HW_DISPATCH keys "
            f"{set(_REAL_HW_DISPATCH)} != BackendKind members {expected}"
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
       - v2.4 item 13: dispatch is wired for all 4 backends —
         ptxas (PTX), naga (WebGPU), llvm-mc (ROCm), xcrun metal
         (Metal). Each tool assembles/validates the emitted text and
         a non-zero exit becomes real_hw_passed=False + findings.
       - A detected-but-non-canonical tool (PTX via nvcc, WebGPU via
         wgpu/dawn_node) still takes the deferred path —
         real_hw_passed=None, overall_status()=DEFERRED.

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
            # v2.4 item 13 — real-HW dispatch via the _REAL_HW_DISPATCH
            # table (R1 audit-fix Finding 2: was a hardcoded if/elif
            # chain). All 4 backends wired: PTX/ptxas, WebGPU/naga,
            # ROCm/llvm-mc, Metal/xcrun. The deferred branch fires only
            # for a detected-but-non-canonical tool (PTX via nvcc,
            # WebGPU via wgpu/dawn_node) — `real_hw_passed=None` rather
            # than True so the result doesn't lie about coverage;
            # `overall_status()` maps it to DEFERRED.
            canonical_tool, dispatch_fn = _REAL_HW_DISPATCH[backend]
            if real_hw_tool == canonical_tool:
                passed, dispatch_findings = dispatch_fn(
                    text, real_hw_tool)
                real_hw_passed = passed
                real_hw_findings.extend(dispatch_findings)
            else:
                real_hw_passed = None
                real_hw_findings.append(
                    f"v2.4 item 13: real-HW tool '{real_hw_tool}' "
                    f"detected for backend {backend.name} but it is "
                    f"not the canonical dispatch tool "
                    f"'{canonical_tool}' — dispatch deferred"
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
