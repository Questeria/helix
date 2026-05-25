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
import sys
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


# `dispatch_validate_ll` selects these tools from `detect_llvm_tools()`
# by literal name. A module-load guard keeps them coupled to
# LLVM_TOOLS, so a future rename of a tool in LLVM_TOOLS that misses
# the dispatch site cannot silently degrade every dispatch to a
# DEFERRED (mock-only) verdict. Mirrors gpu_ci's `_check_gpu_ci_drift`.
_DISPATCH_TOOLS: tuple[str, ...] = ("llvm-as", "llc")


def _check_llvm_toolchain_drift() -> None:
    missing = [t for t in _DISPATCH_TOOLS if t not in LLVM_TOOLS]
    if missing:
        raise AssertionError(
            f"helixc.backend.llvm_toolchain: dispatch_validate_ll "
            f"selects tool(s) {missing} absent from LLVM_TOOLS "
            f"{LLVM_TOOLS} — a rename would silently degrade every "
            f"dispatch to a DEFERRED (mock-only) verdict"
        )


_check_llvm_toolchain_drift()


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
    # `real_tool` reports the DEEPEST tool the dispatch reached: on a
    # failure that is the tool that failed (the pipeline stops there);
    # on success it is the last tool run. It starts at "llvm-as" and
    # advances to "llc" only once the llc leg actually begins — so a
    # caller never reads "llvm-as" for a failure llc produced, and a
    # bitcode-only run (no llc on PATH) is distinguishable from a full
    # lowering. v3.0 Stage 201 audit-fix.
    last_tool = "llvm-as"
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
            # When llvm-as succeeded (`not findings` — it is the only
            # finding source so far) and llc is present, lower the
            # bitcode to a native object too — the full "assemble
            # emitted IR to an object".
            llc = tools.get("llc")
            if not findings and llc is not None:
                last_tool = "llc"
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
        real_tool=last_tool, real_findings=tuple(findings))


# v3.1 — drop-in replacement for `x86_64.compile_module_to_elf`.
# Tri-state result type so callers that want to introspect the
# DEFERRED-vs-FAILED distinction can; a thin raise-on-non-PASS
# wrapper (`compile_module_to_elf_via_llvm`) gives the simpler
# bytes-returning surface that x86_64.py's callers use today.


class LLVMToolchainAbsent(Exception):
    """The LLVM toolchain isn't installed (or isn't complete enough
    to lower IR to a runnable ELF). Distinct from `LLVMEmitError` so
    test code can `pytest.skip` on toolchain-absent without
    masking real codegen bugs."""


class LLVMToolchainError(Exception):
    """The LLVM toolchain is installed but one of its tools reported
    a failure (non-zero exit, empty artifact, timeout, etc.).
    Carries the full chain of tool diagnostics on `.findings`."""

    def __init__(self, message: str, findings: tuple[str, ...]) -> None:
        super().__init__(message)
        self.findings = findings


@dataclass(frozen=True)
class LLVMCompileResult:
    """Result of compiling a `tir.Module` through the LLVM toolchain
    to a runnable ELF.

    Tri-state status mirrors `LLVMDispatchResult`:
      - PASSED: `elf_bytes` is the linked ELF.
      - FAILED: `findings` has the diagnostic chain; `elf_bytes` is
        None.
      - DEFERRED: toolchain absent (`clang` not on PATH — the linker
        is the last tool the pipeline needs); `elf_bytes` is None.
        Distinct from FAILED so a no-LLVM CI runner doesn't fail.

    `real_tool` reports the DEEPEST tool the pipeline reached
    (`llvm-as` / `llc` / `clang`) — on success the last tool run; on
    FAILED the tool that produced the failure.
    """
    status: LLVMDispatchStatus
    elf_bytes: Optional[bytes]
    findings: tuple[str, ...]
    real_tool: Optional[str]

    def __post_init__(self) -> None:
        if self.status is LLVMDispatchStatus.PASSED:
            if self.elf_bytes is None or len(self.elf_bytes) == 0:
                raise ValueError(
                    "LLVMCompileResult: status=PASSED but "
                    "elf_bytes is None or empty — a PASS must "
                    "produce a non-empty ELF")
            if self.findings:
                raise ValueError(
                    f"LLVMCompileResult: status=PASSED but "
                    f"findings has entries {self.findings!r} — "
                    f"a PASS carries no diagnostics")
            if self.real_tool is None:
                raise ValueError(
                    "LLVMCompileResult: status=PASSED but "
                    "real_tool is None — a PASS reached at least "
                    "one tool")
        else:
            if self.elf_bytes is not None:
                raise ValueError(
                    f"LLVMCompileResult: status={self.status} but "
                    f"elf_bytes is not None (len="
                    f"{len(self.elf_bytes)}) — only PASSED carries "
                    f"an ELF")
            if (self.status is LLVMDispatchStatus.FAILED
                    and not self.findings):
                raise ValueError(
                    "LLVMCompileResult: status=FAILED but findings "
                    "is empty — a failure must carry a diagnostic")


def compile_module_to_elf_via_llvm_full(
        module, entry_fn: str = "main", *,
        timeout_s: int = _LLVM_DISPATCH_TIMEOUT_S) -> LLVMCompileResult:
    """Compile a `tir.Module` through the LLVM toolchain to a
    runnable ELF, returning a tri-state result. Pipeline:

        llvm_ir.emit_module(mod) -> .ll text
        llvm-as -> .bc bitcode
        llc -filetype=obj -> .o object
        clang --target=x86_64-unknown-linux-gnu -o elf <object>
            -> ELF binary (links libc + crt)

    `clang` is the deepest tool because the emitted IR uses libc
    symbols (open / read / write / close / exit / llvm.trap) — those
    need to be resolved at link time. The explicit `--target` tells
    clang to cross-compile to the same Linux x86_64 triple
    `helixc/backend/llvm_ir.py` emits — without it, a Windows /
    macOS clang would default to its host triple and fail at link
    time with a confusing libc-missing error.

    DEFERRED conditions (all return without raising):
      - any of {llvm-as, llc, clang} not on PATH;
      - host platform is not Linux AND no `HELIX_LLVM_CROSS=1` env
        flag is set (cross-linking to x86_64-linux ELF requires a
        Linux sysroot which clang on Windows/macOS does not bundle).
        Findings carry a one-line explanation so a caller can show
        the user why DEFERRED fired (this is the one DEFERRED path
        that benefits from a finding — the toolchain-absent case
        is intentionally finding-free since the missing-tool log is
        more useful than a generic "tool missing" string).
    FAILED when any tool returns non-zero / empty artifact /
    timeout / OS error.
    PASSED only when every stage emits a non-empty artifact and
    the final ELF is readable.

    `entry_fn` mirrors `x86_64.compile_module_to_elf`'s parameter
    and is validated up front (the LLVM linker would otherwise fail
    with an opaque "undefined reference to `<entry>`" message).

    This is the v3.1 drop-in replacement for
    `x86_64.compile_module_to_elf` — see the thin wrapper
    `compile_module_to_elf_via_llvm` below for the bytes-returning
    surface that x86_64-style callers use today.
    """
    # entry_fn validation — mirror x86_64.compile_module_to_elf so
    # a typo'd entry produces the same diagnostic shape across
    # backends rather than an opaque linker error. Audit-fix HIGH-1.
    if entry_fn not in module.functions:
        # ValueError mirrors x86_64.compile_module_to_elf's contract
        # so callers can keep one `try/except ValueError` arm.
        raise ValueError(
            f"compile_module_to_elf_via_llvm: entry_fn "
            f"{entry_fn!r} not in module.functions "
            f"({sorted(module.functions)})")

    # Late import: llvm_ir's catchall + helper-registry validation
    # runs at module load and pulls in a lot of code; defer until
    # actually needed so a tool-detection-only caller doesn't pay
    # the import cost.
    from .llvm_ir import emit_module as llvm_emit_module
    ll_text = llvm_emit_module(module)

    tools = detect_llvm_tools()
    llvm_as = tools.get("llvm-as")
    llc = tools.get("llc")
    clang = tools.get("clang")
    if llvm_as is None or llc is None or clang is None:
        # DEFERRED — no diagnostic in findings (the absence is the
        # status; reporting it as a finding would conflate with
        # toolchain-present-but-failed).
        return LLVMCompileResult(
            status=LLVMDispatchStatus.DEFERRED,
            elf_bytes=None, findings=(), real_tool=None)

    # Host-OS guard (audit-fix HIGH-2): the emitted IR's target
    # triple is `x86_64-unknown-linux-gnu`. A clang on Windows /
    # macOS defaults to its host driver and would fail at link time
    # with a libc-missing error rather than a clean DEFERRED. The
    # `HELIX_LLVM_CROSS=1` env flag is the explicit escape hatch
    # for users who have configured a Linux sysroot for cross-link.
    if (sys.platform != "linux"
            and os.environ.get("HELIX_LLVM_CROSS") != "1"):
        return LLVMCompileResult(
            status=LLVMDispatchStatus.DEFERRED,
            elf_bytes=None,
            findings=(
                f"clang is installed but host platform is "
                f"{sys.platform!r}; cross-linking to "
                f"x86_64-unknown-linux-gnu requires a Linux "
                f"sysroot (set HELIX_LLVM_CROSS=1 to bypass this "
                f"guard once your sysroot is configured)",),
            real_tool=None)

    findings: list[str] = []
    last_tool = "llvm-as"
    elf_bytes: Optional[bytes] = None
    tmpdir = tempfile.mkdtemp(prefix="helix_llvm_compile_")
    try:
        ll_path = os.path.join(tmpdir, "module.ll")
        bc_path = os.path.join(tmpdir, "module.bc")
        obj_path = os.path.join(tmpdir, "module.o")
        elf_path = os.path.join(tmpdir, "module.elf")
        try:
            with open(ll_path, "w", encoding="utf-8") as f:
                f.write(ll_text)
        except OSError as e:
            findings.append(
                f"could not write temp .ll {ll_path!r} "
                f"({type(e).__name__}: {e})")
        else:
            findings += _run_tool(
                [llvm_as, ll_path, "-o", bc_path],
                artifact=bc_path, timeout_s=timeout_s)
            if not findings:
                last_tool = "llc"
                findings += _run_tool(
                    [llc, "-filetype=obj", bc_path, "-o", obj_path],
                    artifact=obj_path, timeout_s=timeout_s)
            if not findings:
                last_tool = "clang"
                # `clang` (not `ld` directly) so the host's libc +
                # crt are picked up via clang's driver search paths.
                # `--target=x86_64-unknown-linux-gnu` pins the link
                # to the same triple the IR carries — on a Linux
                # host this is a no-op; on cross-link platforms
                # (gated by HELIX_LLVM_CROSS above) it tells clang
                # to look for the Linux sysroot.
                findings += _run_tool(
                    [clang,
                     "--target=x86_64-unknown-linux-gnu",
                     obj_path, "-o", elf_path],
                    artifact=elf_path, timeout_s=timeout_s)
            if not findings:
                try:
                    with open(elf_path, "rb") as f:
                        elf_bytes = f.read()
                except OSError as e:
                    findings.append(
                        f"could not read linked ELF "
                        f"{elf_path!r} ({type(e).__name__}: {e})")
                else:
                    if not elf_bytes:
                        findings.append(
                            f"linked ELF {elf_path!r} is empty — "
                            f"a 0 exit with no bytes is not a pass")
                        elf_bytes = None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if findings:
        return LLVMCompileResult(
            status=LLVMDispatchStatus.FAILED,
            elf_bytes=None, findings=tuple(findings),
            real_tool=last_tool)
    return LLVMCompileResult(
        status=LLVMDispatchStatus.PASSED,
        elf_bytes=elf_bytes, findings=(), real_tool=last_tool)


def compile_module_to_elf_via_llvm(
        module, entry_fn: str = "main", *,
        timeout_s: int = _LLVM_DISPATCH_TIMEOUT_S) -> bytes:
    """Bytes-returning drop-in replacement for
    `x86_64.compile_module_to_elf(module, entry_fn="main") -> bytes`.

    Raises `LLVMToolchainAbsent` when the LLVM toolchain is not
    installed OR the host platform is not Linux without an explicit
    `HELIX_LLVM_CROSS=1` opt-in (so test code can
    `pytest.skip(...)` on toolchain-absent CI runners AND on
    Windows / macOS dev machines without masking real codegen bugs).
    `.findings` on the raised exception carries any explanatory
    text (currently the cross-link guard message; toolchain-absent
    raises with empty `.findings` since the missing-tool log is
    more useful than a generic string).
    Raises `LLVMToolchainError` carrying the diagnostic chain when
    a tool failed.
    Raises `ValueError` when `entry_fn` is not in `module.functions`
    (mirrors `x86_64.compile_module_to_elf`'s contract).

    Callers that want to introspect the DEFERRED-vs-FAILED-vs-PASS
    distinction (e.g. the Stage 207 parity gate) should use
    `compile_module_to_elf_via_llvm_full` directly.
    """
    result = compile_module_to_elf_via_llvm_full(
        module, entry_fn, timeout_s=timeout_s)
    if result.status is LLVMDispatchStatus.PASSED:
        assert result.elf_bytes is not None  # post_init invariant
        return result.elf_bytes
    if result.status is LLVMDispatchStatus.DEFERRED:
        reason = ("; ".join(result.findings) if result.findings
                  else ("LLVM toolchain not installed (llvm-as / "
                        "llc / clang all required for "
                        "compile_module_to_elf_via_llvm; detected "
                        "via shutil.which)"))
        exc = LLVMToolchainAbsent(reason)
        exc.findings = result.findings  # type: ignore[attr-defined]
        raise exc
    raise LLVMToolchainError(
        f"LLVM toolchain failed at {result.real_tool!r}: "
        f"{'; '.join(result.findings)}",
        findings=result.findings)
