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
    # Only llvm-as is detected -> exactly one _run_tool call -> exactly
    # one finding, and it is the artifact-check finding.
    assert len(res.real_findings) == 1, res.real_findings
    assert "no output artifact" in res.real_findings[0], res.real_findings


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


def test_stage201_dispatch_runs_llc_when_present(monkeypatch):
    """When both llvm-as and llc are detected, the dispatch runs both —
    llvm-as -> bitcode, then llc -> object — and `real_tool` reports
    the deepest tool reached (llc)."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        _tools(**{"llvm-as": "/fake/llvm-as",
                                  "llc": "/fake/llc"}))
    calls: list[list[str]] = []

    def run(cmd, **kwargs):
        calls.append(list(cmd))
        if "-o" in cmd:
            out = cmd[cmd.index("-o") + 1]
            with open(out, "wb") as f:
                f.write(b"\x00fake\x00")
        return _FakeProc(0)

    monkeypatch.setattr(lt.subprocess, "run", run)
    res = lt.dispatch_validate_ll(_good_ll())
    assert res.status() is lt.LLVMDispatchStatus.PASSED
    assert res.real_tool == "llc", res.real_tool
    assert len(calls) == 2, calls
    assert calls[0][0] == "/fake/llvm-as", calls[0]
    assert calls[1][0] == "/fake/llc", calls[1]
    assert calls[1][-1].endswith("module.o"), calls[1]


def test_stage201_dispatch_attributes_llc_failure_to_llc(monkeypatch):
    """v3.0 Stage 201 audit-fix — when llvm-as passes but llc fails,
    the failure is attributed to `llc`, not misattributed to llvm-as
    (the tool that actually passed)."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        _tools(**{"llvm-as": "/fake/llvm-as",
                                  "llc": "/fake/llc"}))
    n = [0]

    def run(cmd, **kwargs):
        n[0] += 1
        if n[0] == 1:  # llvm-as succeeds, writes the bitcode
            if "-o" in cmd:
                out = cmd[cmd.index("-o") + 1]
                with open(out, "wb") as f:
                    f.write(b"\x00bc\x00")
            return _FakeProc(0)
        return _FakeProc(1, stderr="llc: bad target triple")  # llc fails

    monkeypatch.setattr(lt.subprocess, "run", run)
    res = lt.dispatch_validate_ll(_good_ll())
    assert res.status() is lt.LLVMDispatchStatus.FAILED
    assert res.real_tool == "llc", res.real_tool  # NOT "llvm-as"
    assert any("llc" in f and "exit 1" in f
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


# ==========================================================================
# v3.1 — compile_module_to_elf_via_llvm: drop-in replacement for
# x86_64.compile_module_to_elf. Tri-state result type + thin
# bytes-returning wrapper that raises LLVMToolchainAbsent on missing
# clang/llc/llvm-as (so test_codegen.py can pytest.skip cleanly).
# ==========================================================================
def _trivial_module() -> tir.Module:
    """`fn main() -> i32 { 42 }` — minimal compilable module."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("main", [], tir.TIRScalar("i32"))
    b.ret(b.const_int(42))
    b.end_function()
    return mod


def test_stage_v31_compile_result_post_init_rejects_passed_with_empty_elf():
    """A PASSED status with no/empty ELF is malformed."""
    with pytest.raises(ValueError, match="elf_bytes is None or empty"):
        lt.LLVMCompileResult(
            status=lt.LLVMDispatchStatus.PASSED,
            elf_bytes=None, findings=(), real_tool="clang")
    with pytest.raises(ValueError, match="elf_bytes is None or empty"):
        lt.LLVMCompileResult(
            status=lt.LLVMDispatchStatus.PASSED,
            elf_bytes=b"", findings=(), real_tool="clang")


def test_stage_v31_compile_result_post_init_rejects_passed_with_findings():
    """A PASSED status with diagnostic findings is contradictory."""
    with pytest.raises(ValueError, match="PASS carries no diagnostics"):
        lt.LLVMCompileResult(
            status=lt.LLVMDispatchStatus.PASSED,
            elf_bytes=b"\x7fELF...", findings=("oops",),
            real_tool="clang")


def test_stage_v31_compile_result_post_init_rejects_passed_with_no_tool():
    with pytest.raises(ValueError, match="real_tool is None"):
        lt.LLVMCompileResult(
            status=lt.LLVMDispatchStatus.PASSED,
            elf_bytes=b"\x7fELF...", findings=(), real_tool=None)


def test_stage_v31_compile_result_post_init_rejects_deferred_with_elf():
    """A non-PASSED status carrying ELF bytes is contradictory."""
    with pytest.raises(ValueError, match="only PASSED carries"):
        lt.LLVMCompileResult(
            status=lt.LLVMDispatchStatus.DEFERRED,
            elf_bytes=b"\x7fELF...", findings=(), real_tool=None)


def test_stage_v31_compile_result_post_init_rejects_failed_no_findings():
    """A FAILED status with no diagnostics is contradictory."""
    with pytest.raises(ValueError,
                       match="FAILED but findings is empty"):
        lt.LLVMCompileResult(
            status=lt.LLVMDispatchStatus.FAILED,
            elf_bytes=None, findings=(), real_tool="clang")


def test_stage_v31_compile_deferred_when_toolchain_absent(monkeypatch):
    """Toolchain-absent (clang missing) -> DEFERRED, never FAILED.
    Mirrors `dispatch_validate_ll`'s DEFERRED semantics so a no-LLVM
    CI runner stays green."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        lambda: {"llvm-as": "/x/llvm-as",
                                 "opt": None, "llc": "/x/llc",
                                 "clang": None})
    result = lt.compile_module_to_elf_via_llvm_full(_trivial_module())
    assert result.status is lt.LLVMDispatchStatus.DEFERRED
    assert result.elf_bytes is None
    assert result.findings == ()
    assert result.real_tool is None


def test_stage_v31_compile_deferred_when_llc_absent(monkeypatch):
    """Any missing tool (here llc) triggers DEFERRED — all three of
    llvm-as / llc / clang are required to reach a runnable ELF."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        lambda: {"llvm-as": "/x/llvm-as",
                                 "opt": None, "llc": None,
                                 "clang": "/x/clang"})
    result = lt.compile_module_to_elf_via_llvm_full(_trivial_module())
    assert result.status is lt.LLVMDispatchStatus.DEFERRED


def test_stage_v31_compile_deferred_when_llvm_as_absent(monkeypatch):
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        lambda: {"llvm-as": None, "opt": None,
                                 "llc": "/x/llc", "clang": "/x/clang"})
    result = lt.compile_module_to_elf_via_llvm_full(_trivial_module())
    assert result.status is lt.LLVMDispatchStatus.DEFERRED


def test_stage_v31_compile_bytes_wrapper_raises_on_deferred(monkeypatch):
    """The thin bytes-returning wrapper raises `LLVMToolchainAbsent`
    on DEFERRED — so test code can `pytest.skip` cleanly without
    masking real codegen bugs."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        lambda: {"llvm-as": None, "opt": None,
                                 "llc": None, "clang": None})
    with pytest.raises(lt.LLVMToolchainAbsent,
                       match="not installed"):
        lt.compile_module_to_elf_via_llvm(_trivial_module())


def test_stage_v31_compile_bytes_wrapper_raises_on_failed(monkeypatch):
    """The wrapper raises `LLVMToolchainError` (carrying the
    diagnostic chain) when a tool fails — distinct exception type so
    callers can distinguish toolchain-absent from real codegen
    failure."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        lambda: {"llvm-as": "/x/llvm-as",
                                 "opt": None, "llc": "/x/llc",
                                 "clang": "/x/clang"})
    monkeypatch.setenv("HELIX_LLVM_CROSS", "1")  # bypass host-OS guard

    def fake_run(cmd, **kwargs):
        return _FakeProc(returncode=1, stderr="simulated llvm-as failure")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(lt.LLVMToolchainError) as exc_info:
        lt.compile_module_to_elf_via_llvm(_trivial_module())
    assert "simulated" in str(exc_info.value)
    # Diagnostic chain accessible on the exception.
    assert exc_info.value.findings


def test_stage_v31_compile_failed_carries_diagnostic_chain(monkeypatch):
    """The full result type records the chain of findings when a
    tool fails mid-pipeline. real_tool names the deepest tool
    reached."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        lambda: {"llvm-as": "/x/llvm-as",
                                 "opt": None, "llc": "/x/llc",
                                 "clang": "/x/clang"})
    monkeypatch.setenv("HELIX_LLVM_CROSS", "1")

    def fake_run(cmd, **kwargs):
        # llvm-as succeeds (writes a fake bitcode file), llc fails.
        tool = cmd[0]
        if "llvm-as" in tool:
            # Write the output artifact so the success path advances.
            out_idx = cmd.index("-o") + 1
            with open(cmd[out_idx], "wb") as f:
                f.write(b"fake-bitcode")
            return _FakeProc(returncode=0)
        # llc fails.
        return _FakeProc(returncode=2, stderr="simulated llc error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = lt.compile_module_to_elf_via_llvm_full(_trivial_module())
    assert result.status is lt.LLVMDispatchStatus.FAILED
    assert result.elf_bytes is None
    assert result.real_tool == "llc", result.real_tool
    assert any("simulated llc error" in f
               for f in result.findings), result.findings


def test_stage_v31_compile_passed_returns_elf_bytes(monkeypatch):
    """A simulated successful pipeline: every tool exits 0, the
    final ELF file contains valid bytes. Result is PASSED with
    elf_bytes populated."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        lambda: {"llvm-as": "/x/llvm-as",
                                 "opt": None, "llc": "/x/llc",
                                 "clang": "/x/clang"})
    monkeypatch.setenv("HELIX_LLVM_CROSS", "1")
    elf_magic = b"\x7fELF" + b"\x00" * 60

    def fake_run(cmd, **kwargs):
        out_idx = cmd.index("-o") + 1
        out_path = cmd[out_idx]
        # Final stage (clang) writes an ELF-shaped file; earlier
        # stages write placeholder bytes.
        with open(out_path, "wb") as f:
            f.write(elf_magic if cmd[0].endswith("clang")
                    else b"intermediate")
        return _FakeProc(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = lt.compile_module_to_elf_via_llvm_full(_trivial_module())
    assert result.status is lt.LLVMDispatchStatus.PASSED, (
        result.findings)
    assert result.real_tool == "clang"
    assert result.elf_bytes == elf_magic
    # Bytes-wrapper returns the same bytes.
    bytes_out = lt.compile_module_to_elf_via_llvm(_trivial_module())
    assert bytes_out == elf_magic


def test_stage_v31_compile_passed_rejects_empty_elf(monkeypatch):
    """If clang exits 0 but writes a 0-byte ELF, the result is
    FAILED (not PASSED): a clean exit without bytes is not a pass."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        lambda: {"llvm-as": "/x/llvm-as",
                                 "opt": None, "llc": "/x/llc",
                                 "clang": "/x/clang"})
    monkeypatch.setenv("HELIX_LLVM_CROSS", "1")

    def fake_run(cmd, **kwargs):
        out_idx = cmd.index("-o") + 1
        # Write empty bytes for every stage.
        with open(cmd[out_idx], "wb") as f:
            f.write(b"")
        return _FakeProc(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = lt.compile_module_to_elf_via_llvm_full(_trivial_module())
    assert result.status is lt.LLVMDispatchStatus.FAILED
    assert result.elf_bytes is None
    assert any("no output artifact" in f or "is empty" in f
               for f in result.findings), result.findings


def test_stage_v31_compile_rejects_missing_entry_fn():
    """Audit-fix HIGH-1: `entry_fn` matches `x86_64.compile_module_
    to_elf`'s contract — a typo'd entry is ValueErrored up front
    rather than producing an opaque linker error."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("not_main", [], tir.TIRScalar("i32"))
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises(ValueError, match="entry_fn 'main' not in"):
        lt.compile_module_to_elf_via_llvm_full(mod)
    with pytest.raises(ValueError, match="entry_fn 'main' not in"):
        lt.compile_module_to_elf_via_llvm(mod)


def test_stage_v31_compile_accepts_custom_entry_fn(monkeypatch):
    """`entry_fn` validation accepts any function present in the
    module (matches x86_64's tolerance)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("custom_entry", [], tir.TIRScalar("i32"))
    b.ret(b.const_int(0))
    b.end_function()
    # No real toolchain on dev — expect DEFERRED, not a raise.
    result = lt.compile_module_to_elf_via_llvm_full(
        mod, "custom_entry")
    assert result.status is lt.LLVMDispatchStatus.DEFERRED


def test_stage_v31_compile_deferred_on_non_linux_host(monkeypatch):
    """Audit-fix HIGH-2: on Windows / macOS without
    HELIX_LLVM_CROSS=1, clang-present-but-wrong-host returns
    DEFERRED with an explanatory finding (rather than blowing up
    in the linker with a libc-missing error)."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        lambda: {"llvm-as": "/x/llvm-as",
                                 "opt": None, "llc": "/x/llc",
                                 "clang": "/x/clang"})
    monkeypatch.setattr(lt.sys, "platform", "win32")
    monkeypatch.delenv("HELIX_LLVM_CROSS", raising=False)
    result = lt.compile_module_to_elf_via_llvm_full(_trivial_module())
    assert result.status is lt.LLVMDispatchStatus.DEFERRED
    assert any("cross-link" in f.lower() or "sysroot" in f.lower()
               for f in result.findings), result.findings


def test_stage_v31_compile_host_guard_bypassed_via_env(monkeypatch):
    """HELIX_LLVM_CROSS=1 bypasses the host-OS guard so users with
    a configured cross-link sysroot can opt in."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        lambda: {"llvm-as": "/x/llvm-as",
                                 "opt": None, "llc": "/x/llc",
                                 "clang": "/x/clang"})
    monkeypatch.setattr(lt.sys, "platform", "win32")
    monkeypatch.setenv("HELIX_LLVM_CROSS", "1")
    # Toolchain monkeypatched present but real subprocess will be
    # called; we don't care about the result, only that the host-
    # OS guard was bypassed (status is NOT DEFERRED with a
    # cross-link finding).
    def fake_run(cmd, **kwargs):
        out_idx = cmd.index("-o") + 1
        with open(cmd[out_idx], "wb") as f:
            f.write(b"placeholder")
        return _FakeProc(returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = lt.compile_module_to_elf_via_llvm_full(_trivial_module())
    # With the guard bypassed AND tools monkeypatched to "succeed",
    # the pipeline reaches PASSED.
    assert result.status is lt.LLVMDispatchStatus.PASSED, (
        result.findings)


def test_stage_v31_compile_clang_invocation_carries_explicit_target(
        monkeypatch):
    """Audit-fix HIGH-2 sibling: the clang command line includes
    `--target=x86_64-unknown-linux-gnu` so a cross-link clang on
    macOS/Windows knows which triple to target (matching the IR's
    emitted target triple)."""
    monkeypatch.setattr(lt, "detect_llvm_tools",
                        lambda: {"llvm-as": "/x/llvm-as",
                                 "opt": None, "llc": "/x/llc",
                                 "clang": "/x/clang"})
    monkeypatch.setenv("HELIX_LLVM_CROSS", "1")
    captured: list[list[str]] = []

    def capturing_run(cmd, **kwargs):
        captured.append(list(cmd))
        out_idx = cmd.index("-o") + 1
        with open(cmd[out_idx], "wb") as f:
            f.write(b"\x7fELF" + b"\x00" * 32)
        return _FakeProc(returncode=0)
    monkeypatch.setattr(subprocess, "run", capturing_run)
    lt.compile_module_to_elf_via_llvm_full(_trivial_module())
    # The clang invocation (the third tool call) carries --target.
    clang_calls = [c for c in captured if "clang" in c[0]]
    assert len(clang_calls) == 1, captured
    assert "--target=x86_64-unknown-linux-gnu" in clang_calls[0], (
        clang_calls[0])


@pytest.mark.skipif(
    shutil.which("llvm-as") is None
    or shutil.which("llc") is None
    or shutil.which("clang") is None,
    reason="full LLVM toolchain (llvm-as + llc + clang) not installed")
def test_stage_v31_compile_real_toolchain_produces_runnable_elf():
    """When the full LLVM toolchain is genuinely installed, compile
    a trivial module to a real ELF. The ELF starts with the ELF
    magic bytes; running it is out of scope for this test (the
    test_codegen migration exercises execution)."""
    elf = lt.compile_module_to_elf_via_llvm(_trivial_module())
    assert elf.startswith(b"\x7fELF"), elf[:16]
