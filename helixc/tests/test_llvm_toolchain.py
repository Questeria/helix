"""Tests for helixc.backend.llvm_toolchain — v3.0 Phase D Stage 201
(LLVM toolchain detection + dispatch).

Stage 201 adds the real validation path on top of Stage 200's
toolchain-free `mock_validate_ll`: detect `llvm-as` / `opt` / `llc` /
`clang`, and when present assemble emitted IR for real. It mirrors
`gpu_ci.py`'s real-HW dispatch — absent tools yield a DEFERRED verdict
(mock-only), never a hard failure.

The real-dispatch tests monkeypatch `subprocess.run` so the dispatch
ORCHESTRATION is verified deterministically on any machine; one
`skipif`-guarded test exercises a genuinely-installed `llvm-as`.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from helixc.ir import tir
from helixc.backend import llvm_ir
from helixc.backend import llvm_toolchain as lt


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _good_ll() -> str:
    """A valid emitted module — `fn main() -> i32 { 42 }`."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("main", [], tir.TIRScalar("i32"))
    b.ret(b.const_int(42))
    b.end_function()
    return llvm_ir.emit_module(mod)


_BROKEN_LL = (
    'target triple = "x86_64-unknown-linux-gnu"\n'
    "define i32 @main() {\n"
    "  %v0 = add i32 1, 2\n"
    "}\n"
)  # no `ret` terminator -> mock_validate_ll flags it


class _FakeProc:
    """Stand-in for subprocess.CompletedProcess."""
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_run(returncode: int = 0, *, write_artifact: bool = True,
              stderr: str = "", raises: BaseException | None = None):
    """Build a fake `subprocess.run`. On success it writes a non-empty
    file at the command's `-o` target so the artifact check passes."""
    def run(cmd, **kwargs):
        if raises is not None:
            raise raises
        if write_artifact and "-o" in cmd:
            out = cmd[cmd.index("-o") + 1]
            with open(out, "wb") as f:
                f.write(b"\x00fake-llvm-artifact\x00")
        return _FakeProc(returncode, stderr=stderr)
    return run


def _tools(**overrides):
    """A detect_llvm_tools() result; every tool absent unless given."""
    base = {t: None for t in lt.LLVM_TOOLS}
    base.update(overrides)
    return lambda: base


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------
def test_stage201_detect_llvm_tools_returns_all_keys():
    """detect_llvm_tools() reports every tool in LLVM_TOOLS, each value
    either a string path or None."""
    tools = lt.detect_llvm_tools()
    assert set(tools) == set(lt.LLVM_TOOLS)
    for name, path in tools.items():
        assert path is None or isinstance(path, str), (name, path)


# --------------------------------------------------------------------------
# dispatch — toolchain absent (DEFERRED) / mock-fail (FAILED)
# --------------------------------------------------------------------------
def test_stage201_dispatch_deferred_when_no_toolchain(monkeypatch):
    """With no `llvm-as` on PATH, a valid module yields DEFERRED — the
    mock check passed but real validation could not run. Never FAILED."""
    monkeypatch.setattr(lt, "detect_llvm_tools", _tools())
    res = lt.dispatch_validate_ll(_good_ll())
    assert res.status() is lt.LLVMDispatchStatus.DEFERRED
    assert res.mock_passed is True
    assert res.real_attempted is False
    assert res.passed() is False  # DEFERRED is not a full PASS


def test_stage201_dispatch_failed_when_mock_fails(monkeypatch):
    """A structurally-broken module is FAILED even with no toolchain —
    the mock check alone is decisive."""
    monkeypatch.setattr(lt, "detect_llvm_tools", _tools())
    res = lt.dispatch_validate_ll(_BROKEN_LL)
    assert res.status() is lt.LLVMDispatchStatus.FAILED
    assert res.mock_passed is False
    assert res.mock_findings  # carries the diagnostic


# --------------------------------------------------------------------------
# dispatch — real path (subprocess.run monkeypatched)
# --------------------------------------------------------------------------
def test_stage201_dispatch_passed_on_real_success(monkeypatch):
    """llvm-as present and exiting 0 with an artifact -> PASSED."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        _tools(**{"llvm-as": "/fake/llvm-as"}))
    monkeypatch.setattr(lt.subprocess, "run", _fake_run(0))
    res = lt.dispatch_validate_ll(_good_ll())
    assert res.status() is lt.LLVMDispatchStatus.PASSED
    assert res.real_attempted is True
    assert res.real_passed is True
    assert res.real_tool == "llvm-as"
    assert res.passed() is True


def test_stage201_dispatch_failed_on_nonzero_exit(monkeypatch):
    """llvm-as exiting non-zero -> FAILED, with the tool diagnostic."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        _tools(**{"llvm-as": "/fake/llvm-as"}))
    monkeypatch.setattr(lt.subprocess, "run",
                        _fake_run(1, write_artifact=False,
                                  stderr="bad token at line 3"))
    res = lt.dispatch_validate_ll(_good_ll())
    assert res.status() is lt.LLVMDispatchStatus.FAILED
    assert res.real_passed is False
    assert any("exit 1" in f and "bad token" in f
               for f in res.real_findings), res.real_findings


def test_stage201_dispatch_failed_on_zero_exit_no_artifact(monkeypatch):
    """A 0 exit that produced no output object is NOT a pass — the
    artifact check catches it (gpu_ci silent-failure discipline)."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        _tools(**{"llvm-as": "/fake/llvm-as"}))
    monkeypatch.setattr(lt.subprocess, "run",
                        _fake_run(0, write_artifact=False))
    res = lt.dispatch_validate_ll(_good_ll())
    assert res.status() is lt.LLVMDispatchStatus.FAILED
    assert any("no output artifact" in f for f in res.real_findings), \
        res.real_findings


def test_stage201_dispatch_failed_on_timeout(monkeypatch):
    """A hung tool surfaces as a finding, not an uncaught exception."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        _tools(**{"llvm-as": "/fake/llvm-as"}))
    monkeypatch.setattr(
        lt.subprocess, "run",
        _fake_run(raises=subprocess.TimeoutExpired(["llvm-as"], 30)))
    res = lt.dispatch_validate_ll(_good_ll())
    assert res.status() is lt.LLVMDispatchStatus.FAILED
    assert any("timed out" in f for f in res.real_findings), \
        res.real_findings


def test_stage201_dispatch_failed_on_os_error(monkeypatch):
    """A tool that vanished/became non-executable between detection and
    invocation surfaces as a finding, not an uncaught OSError."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        _tools(**{"llvm-as": "/fake/llvm-as"}))
    monkeypatch.setattr(lt.subprocess, "run",
                        _fake_run(raises=OSError("not executable")))
    res = lt.dispatch_validate_ll(_good_ll())
    assert res.status() is lt.LLVMDispatchStatus.FAILED
    assert any("unusable at invocation" in f
               for f in res.real_findings), res.real_findings


# --------------------------------------------------------------------------
# LLVMDispatchResult — invariant enforcement
# --------------------------------------------------------------------------
def test_stage201_result_rejects_mock_flag_findings_mismatch():
    with pytest.raises(ValueError, match="mock_passed"):
        lt.LLVMDispatchResult(
            mock_passed=True, mock_findings=("oops",),
            real_attempted=False, real_passed=None,
            real_tool=None, real_findings=())


def test_stage201_result_rejects_not_attempted_with_tool():
    with pytest.raises(ValueError, match="real_attempted=False"):
        lt.LLVMDispatchResult(
            mock_passed=True, mock_findings=(),
            real_attempted=False, real_passed=None,
            real_tool="llvm-as", real_findings=())


def test_stage201_result_rejects_attempted_without_verdict():
    with pytest.raises(ValueError, match="concrete verdict"):
        lt.LLVMDispatchResult(
            mock_passed=True, mock_findings=(),
            real_attempted=True, real_passed=None,
            real_tool="llvm-as", real_findings=())


def test_stage201_result_rejects_failure_without_diagnostic():
    with pytest.raises(ValueError, match="must carry a diagnostic"):
        lt.LLVMDispatchResult(
            mock_passed=True, mock_findings=(),
            real_attempted=True, real_passed=False,
            real_tool="llvm-as", real_findings=())


def test_stage201_result_status_tristate():
    """status() maps the field combinations to the three outcomes."""
    deferred = lt.LLVMDispatchResult(
        mock_passed=True, mock_findings=(), real_attempted=False,
        real_passed=None, real_tool=None, real_findings=())
    assert deferred.status() is lt.LLVMDispatchStatus.DEFERRED

    passed = lt.LLVMDispatchResult(
        mock_passed=True, mock_findings=(), real_attempted=True,
        real_passed=True, real_tool="llvm-as", real_findings=())
    assert passed.status() is lt.LLVMDispatchStatus.PASSED
    assert passed.passed() is True

    failed_real = lt.LLVMDispatchResult(
        mock_passed=True, mock_findings=(), real_attempted=True,
        real_passed=False, real_tool="llvm-as",
        real_findings=("llvm-as exit 1: boom",))
    assert failed_real.status() is lt.LLVMDispatchStatus.FAILED

    failed_mock = lt.LLVMDispatchResult(
        mock_passed=False, mock_findings=("no ret",),
        real_attempted=False, real_passed=None, real_tool=None,
        real_findings=())
    assert failed_mock.status() is lt.LLVMDispatchStatus.FAILED


# --------------------------------------------------------------------------
# real toolchain (only when llvm-as is genuinely installed)
# --------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("llvm-as") is None,
                    reason="llvm-as not installed on this machine")
def test_stage201_real_llvm_as_accepts_valid_ir():
    """When llvm-as is genuinely present, it assembles a valid emitted
    module without error."""
    res = lt.dispatch_validate_ll(_good_ll())
    assert res.status() is lt.LLVMDispatchStatus.PASSED, res.real_findings


@pytest.mark.skipif(shutil.which("llvm-as") is None,
                    reason="llvm-as not installed on this machine")
def test_stage201_real_llvm_as_rejects_malformed_ir():
    """When llvm-as is genuinely present, it rejects malformed IR — a
    `ret` of the wrong type for the function — as a real failure."""
    bad = (
        'target triple = "x86_64-unknown-linux-gnu"\n'
        "define i32 @main() {\n"
        "  ret i64 0\n"          # i64 value from an i32 function
        "}\n"
    )
    res = lt.dispatch_validate_ll(bad)
    assert res.status() is lt.LLVMDispatchStatus.FAILED, res
    assert res.real_findings
