"""
helixc/backend/llvm_toolchain.py — LLVM toolchain detection + dispatch
(v3.0 Phase D, Stage 201).

Stage 200 stood up `helixc/backend/llvm_ir.py` — a textual-LLVM-IR
emitter with a toolchain-free `mock_validate_ll` shape check. Stage 201
adds the REAL validation path: detect the LLVM command-line tools
(`llvm-as`, `opt`, `llc`, `clang`) and, when they are present, assemble
emitted IR to an object so the IR is proven genuinely well-formed by
LLVM itself rather than only shape-checked.

This mirrors `helixc/backend/gpu_ci.py`'s real-HW dispatch discipline:
- detection via `shutil.which` — absent tools are normal, not an error;
- real dispatch is gated behind detection — a machine with no LLVM
  installed gets the mock path and a DEFERRED verdict, never a hard
  failure (CI on a tool-less runner stays green);
- subprocess calls carry a timeout and catch OSError;
- a 0 exit is necessary but not sufficient — the output artifact must
  exist and be non-empty (else the tool "passed" without producing
  anything);
- the result type makes "a failure with no diagnostic" unrepresentable.

License: Apache 2.0
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .llvm_ir import mock_validate_ll


# The LLVM command-line tools Stage 201 knows about. `llvm-as` is the
# load-bearing one — it assembles AND verifies textual IR, so it is the
# real well-formedness gate. `llc` lowers bitcode to a native object.
# `opt` (the optimizer) and `clang` are detected for the later Phase-D
# stages but are not invoked by Stage 201's dispatch.
LLVM_TOOLS: tuple[str, ...] = ("llvm-as", "opt", "llc", "clang")

# Wall-clock cap on any single tool invocation. Assembling one small
# `.ll` is sub-second; the cap only guards against a hung tool.
_LLVM_DISPATCH_TIMEOUT_S = 30


class LLVMDispatchStatus(Enum):
    """Tri-state outcome of validating emitted LLVM IR.

    Distinguishing DEFERRED from PASSED is deliberate (the same
    silent-failure concern `gpu_ci.ValidationResult` documents): a
    caller acting on a plain boolean cannot tell whether LLVM actually
    assembled the IR or whether the check was skipped because no
    toolchain was installed."""
    PASSED = "passed"      # mock passed AND real dispatch passed
    FAILED = "failed"      # mock failed, OR real dispatch returned False
    DEFERRED = "deferred"  # mock passed but no LLVM toolchain to dispatch on


@dataclass(frozen=True)
class LLVMDispatchResult:
    """Outcome of validating emitted LLVM IR — the toolchain-free
    `mock_validate_ll` shape check always, plus a real `llvm-as`
    (+ `llc` when present) dispatch when the LLVM toolchain is
    installed.

    Frozen + tuple-backed (the `gpu_ci.ValidationResult` pattern).
    `__post_init__` forbids the silent-failure / illegal shapes:
    - `mock_passed` must agree with `mock_findings` emptiness;
    - `real_attempted=False` must imply `real_passed` is None,
      `real_tool` is None, and `real_findings` empty;
    - `real_attempted=True` must imply `real_passed` is a concrete
      bool (not None) — the dispatch always reaches a verdict;
    - a real-dispatch FAILURE (`real_passed is False`) must carry at
      least one diagnostic — "fail without a reason" is unrepresentable.
    """
    mock_passed: bool
    mock_findings: tuple[str, ...]
    real_attempted: bool
    real_passed: Optional[bool]
    real_tool: Optional[str]
    real_findings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.mock_passed != (len(self.mock_findings) == 0):
            raise ValueError(
                f"LLVMDispatchResult: mock_passed={self.mock_passed} "
                f"disagrees with mock_findings emptiness "
                f"(len={len(self.mock_findings)})"
            )
        if not self.real_attempted:
            if self.real_passed is not None:
                raise ValueError(
                    f"LLVMDispatchResult: real_attempted=False but "
                    f"real_passed={self.real_passed!r} — illegal"
                )
            if self.real_tool is not None:
                raise ValueError(
                    f"LLVMDispatchResult: real_attempted=False but "
                    f"real_tool={self.real_tool!r} — illegal"
                )
            if self.real_findings:
                raise ValueError(
                    "LLVMDispatchResult: real_attempted=False but "
                    "real_findings has entries — illegal"
                )
        else:
            if self.real_passed is None:
                raise ValueError(
                    "LLVMDispatchResult: real_attempted=True but "
                    "real_passed is None — a dispatch must reach a "
                    "concrete verdict"
                )
            if self.real_passed is False and not self.real_findings:
                raise ValueError(
                    "LLVMDispatchResult: real_passed=False but "
                    "real_findings is empty — a real-dispatch failure "
                    "must carry a diagnostic"
                )

    def status(self) -> LLVMDispatchStatus:
        """Tri-state verdict. FAILED if the mock shape-check failed or
        the real dispatch returned False; DEFERRED if the mock passed
        but no toolchain was available to dispatch; PASSED only when
        both the mock check and a real LLVM dispatch passed."""
        if not self.mock_passed:
            return LLVMDispatchStatus.FAILED
        if not self.real_attempted:
            return LLVMDispatchStatus.DEFERRED
        if self.real_passed is False:
            return LLVMDispatchStatus.FAILED
        return LLVMDispatchStatus.PASSED

    def passed(self) -> bool:
        """True ONLY for a full PASS (mock + real both validated). A
        caller content with mock-only validation should instead test
        `status() is not LLVMDispatchStatus.FAILED`, making the
        deferred-acceptance choice visible at the call site."""
        return self.status() is LLVMDispatchStatus.PASSED


def detect_llvm_tools() -> dict[str, Optional[str]]:
    """Map each tool in `LLVM_TOOLS` to its resolved PATH location, or
    None when the tool is absent. An all-None result means no LLVM
    toolchain on this machine — `dispatch_validate_ll` then returns a
    DEFERRED (mock-only) result, never a hard failure. Mirrors
    `gpu_ci.detect_tools`."""
    return {tool: shutil.which(tool) for tool in LLVM_TOOLS}


def _run_tool(cmd: list[str], *, artifact: str,
              timeout_s: int = _LLVM_DISPATCH_TIMEOUT_S) -> list[str]:
    """Run one toolchain command. Returns a list of findings (empty ==
    success). `artifact` is the output path the tool must have written
    a non-empty file to.

    Applies the `gpu_ci` real-HW dispatch discipline: a `TimeoutExpired`
    or `OSError` (a tool that vanished or is non-executable between
    detection and invocation) becomes a structured finding, never an
    uncaught traceback; a non-zero exit carries the truncated tool
    diagnostic; and a 0 exit that produced no/empty artifact is treated
    as a failure (a tool exiting 0 without emitting its output is not a
    pass)."""
    label = os.path.basename(cmd[0])
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return [f"{label} dispatch timed out after {timeout_s}s"]
    except OSError as e:
        return [f"{label} dispatch: tool unusable at invocation "
                f"({type(e).__name__}: {e})"]
    if proc.returncode != 0:
        diag = (proc.stderr or proc.stdout or "").strip()
        return [f"{label} exit {proc.returncode}: {diag[:500]}"]
    try:
        size = os.path.getsize(artifact)
    except OSError:
        size = -1
    if size <= 0:
        return [f"{label} exited 0 but produced no output artifact at "
                f"{artifact!r} — a 0 exit with no artifact is not a pass"]
    return []


def dispatch_validate_ll(ll_text: str) -> LLVMDispatchResult:
    """Validate emitted LLVM IR text.

    Always runs the toolchain-free `mock_validate_ll` shape check.
    Additionally, when `llvm-as` is on PATH, assembles the IR for real
    (`llvm-as` -> bitcode; then `llc` -> native object when `llc` is
    also present) — `llvm-as` rejecting the IR is a genuine
    well-formedness failure.

    A machine with no `llvm-as` yields a DEFERRED result (mock-only),
    never FAILED — so CI on a runner with no LLVM installed stays
    green. The dispatch never raises for a tool/IO error: every such
    error is captured as a finding (see `_run_tool`)."""
    mock_findings = tuple(mock_validate_ll(ll_text))
    mock_passed = not mock_findings

    tools = detect_llvm_tools()
    llvm_as = tools.get("llvm-as")
    if llvm_as is None:
        # No toolchain — mock-only, DEFERRED.
        return LLVMDispatchResult(
            mock_passed=mock_passed, mock_findings=mock_findings,
            real_attempted=False, real_passed=None,
            real_tool=None, real_findings=())

    findings: list[str] = []
    tmpdir = tempfile.mkdtemp(prefix="helix_llvm_")
    try:
        ll_path = os.path.join(tmpdir, "module.ll")
        bc_path = os.path.join(tmpdir, "module.bc")
        try:
            with open(ll_path, "w", encoding="utf-8") as f:
                f.write(ll_text)
        except OSError as e:
            findings.append(
                f"could not write temp .ll {ll_path!r} "
                f"({type(e).__name__}: {e})")
        else:
            # llvm-as assembles + verifies the textual IR -> bitcode.
            findings += _run_tool(
                [llvm_as, ll_path, "-o", bc_path], artifact=bc_path)
            # When llvm-as succeeded and llc is present, lower the
            # bitcode to a native object too — the full "assemble
            # emitted IR to an object".
            llc = tools.get("llc")
            if not findings and llc is not None:
                obj_path = os.path.join(tmpdir, "module.o")
                findings += _run_tool(
                    [llc, "-filetype=obj", bc_path, "-o", obj_path],
                    artifact=obj_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    real_passed = not findings
    return LLVMDispatchResult(
        mock_passed=mock_passed, mock_findings=mock_findings,
        real_attempted=True, real_passed=real_passed,
        real_tool="llvm-as", real_findings=tuple(findings))
