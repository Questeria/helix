"""Tests for helixc.backend.llvm_parity — v3.0 Phase D, Stage 207
chunk A: the x86_64-vs-LLVM mock structural-parity harness.

The harness compiles a `tir.Module` through BOTH backends and
classifies the outcome (MATCH / UNCOVERED / MISMATCH / ERROR). It is
the chunk-A MOCK path — toolchain-free, always runs; it proves the
LLVM backend either emits structurally shaped IR or fails closed
LOUDLY on an op outside its covered subset, never silently
miscompiles. Real-execution (observable-behaviour) parity is chunk B.

These tests pin: the `ParityResult` type rejects every silent-failure
field shape; `verdict()` derives the classification correctly from the
facts; and `check_parity` classifies real modules (a covered program →
MATCH, a Stage 206-R op → UNCOVERED) and CAPTURES every backend
failure — a crash, a fail-closed, a degenerate input — into a verdict
rather than letting it escape, all without mutating the caller's
module.
"""
from __future__ import annotations

import copy

import pytest

from helixc.ir import tir
from helixc.backend import llvm_parity
from helixc.backend.llvm_parity import ParityResult, ParityVerdict


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _i32() -> tir.TIRScalar:
    return tir.TIRScalar("i32")


def _const_module() -> tir.Module:
    """`fn main() -> i32 { 42 }` — a fully covered program."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("main", [], _i32())
    b.ret(b.const_int(42))
    b.end_function()
    return mod


def _arith_module() -> tir.Module:
    """`fn main() -> i32 { 17 + 25 }` — a fully covered program."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("main", [], _i32())
    b.ret(b.add(b.const_int(17), b.const_int(25)))
    b.end_function()
    return mod


def _two_function_module() -> tir.Module:
    """A module with two functions — exercises the multi-function
    path of both backends. Both are fully covered."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("helper", [], _i32())
    b.ret(b.const_int(1))
    b.end_function()
    b.begin_function("main", [], _i32())
    b.ret(b.const_int(2))
    b.end_function()
    return mod


def _print_int_module() -> tir.Module:
    """`main` with a print_int PRINT — a Stage 206-R residual op the
    LLVM backend does not yet lower (x86_64 does). The canonical
    UNCOVERED case.

    NOTE: this depends on `print_int` remaining outside the LLVM
    backend's covered subset. If a future 206-R chunk lowers
    `print_int`, repoint this helper at a still-residual op (an ARENA
    op, a TRACE op, ...) so the UNCOVERED tests stay meaningful."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("main", [], _i32())
    n = b.const_int(7)
    r = b.emit(tir.OpKind.PRINT, n, result_ty=_i32(),
               attrs={"_kind": "print_int"})
    b.ret(r)
    b.end_function()
    return mod


# --------------------------------------------------------------------------
# ParityResult — __post_init__ rejects silent-failure field shapes
# --------------------------------------------------------------------------
def test_parity_result_rejects_empty_program():
    """Every result must name the program it classifies — a blank
    `program` would make a corpus walk's diagnostics unattributable."""
    with pytest.raises(ValueError, match="program is empty"):
        ParityResult(program="  ", x86_compiled=True, llvm_emitted=True,
                     llvm_failed_closed=False, llvm_mock_clean=True,
                     detail=())


def test_parity_result_rejects_emitted_and_failed_closed():
    """emit_module either returns IR or raises LLVMEmitError, never
    both — a result claiming both is illegal."""
    with pytest.raises(ValueError, match="both True"):
        ParityResult(program="p", x86_compiled=True, llvm_emitted=True,
                     llvm_failed_closed=True, llvm_mock_clean=False,
                     detail=("x",))


def test_parity_result_rejects_failed_closed_and_mock_clean():
    """A fail-closed run emitted no IR — it cannot also be 'mock
    clean'. The rule is stated explicitly, not left to transitive
    deduction."""
    with pytest.raises(ValueError, match="fail-closed run emitted no"):
        ParityResult(program="p", x86_compiled=True, llvm_emitted=False,
                     llvm_failed_closed=True, llvm_mock_clean=True,
                     detail=("x",))


def test_parity_result_rejects_mock_clean_without_emit():
    """`llvm_mock_clean` can only be True when IR was actually
    emitted — there is otherwise nothing to have validated."""
    with pytest.raises(ValueError, match="no emitted IR"):
        ParityResult(program="p", x86_compiled=True, llvm_emitted=False,
                     llvm_failed_closed=False, llvm_mock_clean=True,
                     detail=("x",))


def test_parity_result_rejects_match_with_detail():
    """A MATCH-shaped result must carry NO diagnostic — a clean match
    has nothing to explain."""
    with pytest.raises(ValueError, match="MATCH but detail"):
        ParityResult(program="p", x86_compiled=True, llvm_emitted=True,
                     llvm_failed_closed=False, llvm_mock_clean=True,
                     detail=("spurious",))


def test_parity_result_rejects_nonmatch_without_detail():
    """A non-MATCH verdict must carry a reason — a result that is a
    defect / uncovered with an empty detail is a silent failure."""
    with pytest.raises(ValueError, match="must carry a reason"):
        # x86 ok, emitted, NOT mock-clean -> MISMATCH, but no detail.
        ParityResult(program="p", x86_compiled=True, llvm_emitted=True,
                     llvm_failed_closed=False, llvm_mock_clean=False,
                     detail=())


def test_parity_result_rejects_blank_detail_entry():
    """A blank ('' / whitespace) detail entry is a reason-shaped object
    with no reason — it would satisfy 'non-MATCH must carry a detail'
    while saying nothing. Reject it."""
    with pytest.raises(ValueError, match="blank or non-str"):
        ParityResult(program="p", x86_compiled=True, llvm_emitted=True,
                     llvm_failed_closed=False, llvm_mock_clean=False,
                     detail=("   ",))


def test_parity_result_rejects_nonstr_detail_entry():
    """A non-str detail entry is rejected at construction — before it
    could crash a consumer at a `"; ".join(...)` site. The
    `tuple[str, ...]` annotation is only a hint; `__post_init__` is the
    enforcement."""
    with pytest.raises(ValueError, match="blank or non-str"):
        ParityResult(program="p", x86_compiled=True, llvm_emitted=True,
                     llvm_failed_closed=False, llvm_mock_clean=False,
                     detail=(None,))  # type: ignore[arg-type]


def test_parity_result_rejects_unattempted_real_with_verdict():
    """real_attempted=False must imply real_passed is None — a result
    cannot carry a real-run verdict it never ran (the gpu_ci
    invariant)."""
    with pytest.raises(ValueError, match="real_attempted=False"):
        ParityResult(program="p", x86_compiled=True, llvm_emitted=True,
                     llvm_failed_closed=False, llvm_mock_clean=True,
                     detail=(), real_attempted=False, real_passed=True)


def test_parity_result_rejects_unattempted_real_with_findings():
    """real_attempted=False must imply real_findings is empty."""
    with pytest.raises(ValueError, match="real_attempted=False"):
        ParityResult(program="p", x86_compiled=True, llvm_emitted=True,
                     llvm_failed_closed=False, llvm_mock_clean=True,
                     detail=(), real_attempted=False,
                     real_findings=("ran",))


def test_parity_result_rejects_attempted_real_without_verdict():
    """real_attempted=True must reach a concrete real_passed — a real
    run that returns None is the silent-failure shape forbidden."""
    with pytest.raises(ValueError, match="must reach a concrete"):
        ParityResult(program="p", x86_compiled=True, llvm_emitted=True,
                     llvm_failed_closed=False, llvm_mock_clean=True,
                     detail=(), real_attempted=True, real_passed=None)


def test_parity_result_rejects_real_failure_without_diagnostic():
    """A real-execution FAILURE must carry a diagnostic — 'fail with no
    reason' is unrepresentable (mirrors gpu_ci.ValidationResult)."""
    with pytest.raises(ValueError, match="must carry a"):
        ParityResult(program="p", x86_compiled=True, llvm_emitted=True,
                     llvm_failed_closed=False, llvm_mock_clean=True,
                     detail=(), real_attempted=True, real_passed=False,
                     real_findings=())


def test_parity_result_accepts_valid_real_attempted():
    """Well-formed real-attempted results construct cleanly — the
    forward-compatible fields chunk B fills in are usable now. A real
    PASS may still carry advisory findings; a real FAIL must."""
    ok = ParityResult(program="p", x86_compiled=True, llvm_emitted=True,
                      llvm_failed_closed=False, llvm_mock_clean=True,
                      detail=(), real_attempted=True, real_passed=True,
                      real_findings=())
    assert ok.real_attempted and ok.real_passed is True
    pass_with_note = ParityResult(
        program="p", x86_compiled=True, llvm_emitted=True,
        llvm_failed_closed=False, llvm_mock_clean=True, detail=(),
        real_attempted=True, real_passed=True,
        real_findings=("stderr differed in whitespace only",))
    assert pass_with_note.real_passed is True
    failed = ParityResult(
        program="p", x86_compiled=True, llvm_emitted=True,
        llvm_failed_closed=False, llvm_mock_clean=True, detail=(),
        real_attempted=True, real_passed=False,
        real_findings=("exit code 1 vs 0",))
    assert failed.real_passed is False and failed.real_findings


# --------------------------------------------------------------------------
# ParityResult.verdict() — derived classification
# --------------------------------------------------------------------------
def test_verdict_match():
    """x86 compiled + LLVM emitted well-shaped IR -> MATCH."""
    r = ParityResult(program="p", x86_compiled=True, llvm_emitted=True,
                     llvm_failed_closed=False, llvm_mock_clean=True,
                     detail=())
    assert r.verdict() is ParityVerdict.MATCH
    assert not r.is_parity_defect()


def test_verdict_uncovered():
    """x86 compiled + LLVM failed closed -> UNCOVERED (not a defect —
    the fail-closed guarantee working as designed)."""
    r = ParityResult(program="p", x86_compiled=True, llvm_emitted=False,
                     llvm_failed_closed=True, llvm_mock_clean=False,
                     detail=("uncovered op",))
    assert r.verdict() is ParityVerdict.UNCOVERED
    assert not r.is_parity_defect()


def test_verdict_mismatch():
    """x86 compiled + LLVM emitted MALFORMED IR -> MISMATCH (a real
    parity defect)."""
    r = ParityResult(program="p", x86_compiled=True, llvm_emitted=True,
                     llvm_failed_closed=False, llvm_mock_clean=False,
                     detail=("IR fails mock_validate_ll",))
    assert r.verdict() is ParityVerdict.MISMATCH
    assert r.is_parity_defect()


def test_verdict_error_x86_failed():
    """x86 (the full-coverage incumbent) failed to compile -> ERROR,
    regardless of what the LLVM backend did."""
    r = ParityResult(program="p", x86_compiled=False, llvm_emitted=True,
                     llvm_failed_closed=False, llvm_mock_clean=True,
                     detail=("x86 boom",))
    assert r.verdict() is ParityVerdict.ERROR
    assert r.is_parity_defect()


def test_verdict_error_llvm_crash():
    """x86 compiled, but LLVM neither emitted IR nor failed closed
    (a non-LLVMEmitError crash) -> ERROR."""
    r = ParityResult(program="p", x86_compiled=True, llvm_emitted=False,
                     llvm_failed_closed=False, llvm_mock_clean=False,
                     detail=("llvm crashed",))
    assert r.verdict() is ParityVerdict.ERROR
    assert r.is_parity_defect()


def test_verdict_partition_covers_every_verdict():
    """The defect / OK verdict sets partition ParityVerdict — every
    verdict is classified exactly once. The module-load drift guard
    already enforces this; pin it explicitly too."""
    defect = llvm_parity._PARITY_DEFECT_VERDICTS
    ok = llvm_parity._PARITY_OK_VERDICTS
    assert defect | ok == set(ParityVerdict)
    assert not (defect & ok)
    # The guard is callable and passes for the current partition.
    llvm_parity._check_parity_verdict_coverage()


# --------------------------------------------------------------------------
# check_parity — integration on real modules
# --------------------------------------------------------------------------
def test_check_parity_match_const_return():
    """`fn main() -> i32 { 42 }` is fully covered -> MATCH, no
    diagnostic."""
    r = llvm_parity.check_parity(_const_module(), "const42")
    assert r.verdict() is ParityVerdict.MATCH
    assert r.x86_compiled and r.llvm_emitted and r.llvm_mock_clean
    assert not r.llvm_failed_closed
    assert r.detail == ()


def test_check_parity_match_arithmetic():
    """An arithmetic program is fully covered -> MATCH."""
    r = llvm_parity.check_parity(_arith_module(), "add")
    assert r.verdict() is ParityVerdict.MATCH
    assert not r.is_parity_defect()


def test_check_parity_match_multi_function():
    """A module with two functions compiles through both backends ->
    MATCH."""
    r = llvm_parity.check_parity(_two_function_module(), "two_fns")
    assert r.verdict() is ParityVerdict.MATCH


def test_check_parity_uncovered_print_int():
    """A print_int PRINT (a Stage 206-R residual op) -> UNCOVERED: the
    x86_64 backend compiles it, the LLVM backend fails closed loudly.
    NOT a parity defect — the gate accepts it."""
    r = llvm_parity.check_parity(_print_int_module(), "print_int")
    assert r.verdict() is ParityVerdict.UNCOVERED
    assert r.x86_compiled
    assert not r.llvm_emitted and r.llvm_failed_closed
    assert not r.is_parity_defect()
    # the diagnostic names the uncovered op so the residual is visible
    assert any("print_int" in d for d in r.detail), r.detail


def test_check_parity_error_when_no_main():
    """A module with no `main` is not a runnable program — classified
    ERROR up front, before either backend is invoked, with an honest
    'no main' diagnostic (not inferred from an x86_64 exception)."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("notmain", [], _i32())
    b.ret(b.const_int(0))
    b.end_function()
    r = llvm_parity.check_parity(mod, "no_main")
    assert r.verdict() is ParityVerdict.ERROR
    assert not r.x86_compiled
    assert any("main" in d for d in r.detail), r.detail


def test_check_parity_error_when_empty_module():
    """An empty module (zero functions) has no `main` either — the
    same up-front ERROR guard catches it, rather than the LLVM backend
    emitting `define`-less IR that reads as a false MISMATCH."""
    r = llvm_parity.check_parity(tir.Module(), "empty")
    assert r.verdict() is ParityVerdict.ERROR
    assert not r.x86_compiled
    assert not r.llvm_emitted and not r.llvm_failed_closed
    assert any("main" in d for d in r.detail), r.detail


def test_check_parity_x86_crash_captured(monkeypatch):
    """A non-ValueError crash deep in the x86_64 backend is CAPTURED as
    an ERROR with the diagnostic — the broad `except` is
    capture-not-swallow, never an escaping traceback."""
    def boom(_mod):
        raise RuntimeError("x86 internal invariant blew up")
    monkeypatch.setattr(llvm_parity, "compile_module_to_elf", boom)
    r = llvm_parity.check_parity(_const_module(), "x86_crash")
    assert r.verdict() is ParityVerdict.ERROR
    assert not r.x86_compiled
    assert any("RuntimeError" in d for d in r.detail), r.detail


def test_check_parity_llvm_crash_is_error(monkeypatch):
    """A non-LLVMEmitError crash in emit_module is an ERROR (a real
    bug), distinct from the UNCOVERED clean fail-closed path."""
    def boom(_mod):
        raise RuntimeError("llvm emitter blew up")
    monkeypatch.setattr(llvm_parity, "emit_module", boom)
    r = llvm_parity.check_parity(_const_module(), "llvm_crash")
    assert r.verdict() is ParityVerdict.ERROR
    assert not r.llvm_emitted and not r.llvm_failed_closed
    assert any("crashed" in d and "RuntimeError" in d
               for d in r.detail), r.detail


def test_check_parity_mock_validate_crash_is_error(monkeypatch):
    """A crash inside the shape-checker is an ERROR, NOT a MISMATCH —
    a validator crash does not prove the emitted IR malformed, so it
    must not be reported as an LLVM emitter bug."""
    def boom(_text):
        raise RuntimeError("shape-checker blew up")
    monkeypatch.setattr(llvm_parity, "mock_validate_ll", boom)
    r = llvm_parity.check_parity(_const_module(), "validator_crash")
    assert r.verdict() is ParityVerdict.ERROR
    assert not r.llvm_emitted
    assert any("mock_validate_ll crashed" in d for d in r.detail), r.detail


def test_check_parity_mismatch_on_malformed_ir(monkeypatch):
    """If the LLVM backend emits IR that FAILS the mock_validate_ll
    shape check, the program is a MISMATCH — a real emitter defect the
    gate must fail on."""
    def malformed(_mod):
        # Valid-looking but with unbalanced braces — mock_validate_ll
        # flags it; the missing `}` is a structural defect.
        return ('; helixc\ntarget triple = "x86_64-unknown-linux-gnu"'
                '\n\ndefine i32 @main() {\n  ret i32 0\n')
    monkeypatch.setattr(llvm_parity, "emit_module", malformed)
    r = llvm_parity.check_parity(_const_module(), "malformed")
    assert r.verdict() is ParityVerdict.MISMATCH
    assert r.llvm_emitted and not r.llvm_mock_clean
    assert r.is_parity_defect()
    assert any("mock_validate_ll" in d for d in r.detail), r.detail


def test_check_parity_does_not_swallow_baseexception(monkeypatch):
    """The broad `except Exception` must NOT catch BaseException — a
    KeyboardInterrupt propagates out rather than being buried in an
    ERROR verdict (an un-interruptible corpus walk is the worst silent
    failure)."""
    def interrupt(_mod):
        raise KeyboardInterrupt
    monkeypatch.setattr(llvm_parity, "compile_module_to_elf", interrupt)
    with pytest.raises(KeyboardInterrupt):
        llvm_parity.check_parity(_const_module(), "interrupted")


def test_check_parity_does_not_mutate_caller_module():
    """The harness hands each backend its OWN deep copy — the caller's
    module is byte-identical before and after. A backend that mutates
    the module it is given cannot corrupt the caller's, nor make the
    two backends see different input."""
    mod = _arith_module()
    before = copy.deepcopy(mod)
    llvm_parity.check_parity(mod, "no_mutate")
    assert mod == before


def test_check_parity_leaves_real_fields_unattempted():
    """Chunk A is the MOCK path only — every result it produces leaves
    the real-execution fields not-attempted (chunk B fills them in)."""
    for mod, name in ((_const_module(), "c"),
                      (_print_int_module(), "p")):
        r = llvm_parity.check_parity(mod, name)
        assert r.real_attempted is False
        assert r.real_passed is None
        assert r.real_findings == ()


def test_check_parity_smoke_corpus():
    """A small smoke corpus through `check_parity`: no program is a
    parity defect, and the harness genuinely produces BOTH a MATCH
    (covered) and an UNCOVERED (a 206-R op) — proving it discriminates
    rather than rubber-stamping."""
    corpus = [
        ("const42", _const_module()),
        ("add", _arith_module()),
        ("two_fns", _two_function_module()),
        ("print_int", _print_int_module()),
    ]
    verdicts = set()
    for name, mod in corpus:
        r = llvm_parity.check_parity(mod, name)
        assert not r.is_parity_defect(), (name, r.detail)
        verdicts.add(r.verdict())
    assert ParityVerdict.MATCH in verdicts
    assert ParityVerdict.UNCOVERED in verdicts
