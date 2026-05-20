"""Tests for helixc.backend.llvm_parity — v3.0 Phase D, Stage 207
chunks A+B+C: the x86_64-vs-LLVM mock structural-parity harness, the
curated source-program corpus + parity gate, and the real-execution
result model + toolchain detection.

The harness compiles a `tir.Module` through BOTH backends and
classifies the outcome (MATCH / UNCOVERED / MISMATCH / ERROR). It is
the MOCK path — toolchain-free, always runs; it proves the LLVM
backend either emits structurally shaped IR or fails closed LOUDLY on
an op outside its covered subset, never silently miscompiles. The
real-execution (observable-behaviour) DISPATCH is chunk D.

These tests pin: (chunk A) the `ParityResult` type rejects every
silent-failure field shape, `verdict()` derives the classification
correctly from the facts, and `check_parity` classifies real modules
and CAPTURES every backend failure into a verdict rather than letting
it escape, without mutating the caller's module; (chunk B)
`check_parity_source` compiles a Helix source string through the
frontend pipeline, and the Stage 207 mock-path GATE asserts every
program in the curated corpus structurally MATCHes across both
backends; (chunk C) `real_status()` derives the 4-state real outcome
(NOT_RUN / DEFERRED / PASS / FAIL) and `detect_real_exec_support`
reports whether this machine can run the real comparison.
"""
from __future__ import annotations

import copy
import subprocess

import pytest

from helixc.ir import tir
from helixc.backend import llvm_parity
from helixc.backend.llvm_parity import (
    ParityResult, ParityVerdict, RealParityStatus)


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


def test_parity_result_rejects_deferred_real_without_reason():
    """A DEFERRED real run (real_attempted=True, real_passed=None) is
    legal — but it must record WHY it was deferred. real_passed=None
    with empty real_findings is the silent-failure shape forbidden."""
    with pytest.raises(ValueError, match="deferred real run must"):
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
    forward-compatible fields chunk D fills in are usable now. A real
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
    the real-execution fields not-attempted (chunk D fills them in)."""
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


# --------------------------------------------------------------------------
# check_parity_source + the curated parity corpus (chunk B)
# --------------------------------------------------------------------------
def test_check_parity_source_match():
    """A minimal integer Helix source program parity-checks to MATCH
    through the full frontend pipeline (parse -> lower -> both
    backends)."""
    r = llvm_parity.check_parity_source(
        "fn main() -> i32 { 17 + 25 }", "src_add")
    assert r.verdict() is ParityVerdict.MATCH
    assert not r.is_parity_defect()


def test_check_parity_source_uncovered_on_float():
    """A float-typed source -> UNCOVERED: the x86_64 backend handles
    f64, the LLVM backend fails closed (floats are outside its covered
    subset). check_parity_source carries the UNCOVERED verdict through
    from check_parity unchanged."""
    r = llvm_parity.check_parity_source(
        "fn main() -> f64 { 1.5 }", "src_float")
    assert r.verdict() is ParityVerdict.UNCOVERED
    assert not r.is_parity_defect()


def test_check_parity_source_error_on_bad_source():
    """A source the frontend cannot compile is CAPTURED as an ERROR
    (not re-raised) — a degenerate corpus entry, surfaced loudly with
    the pipeline diagnostic."""
    r = llvm_parity.check_parity_source("fn main((", "bad_src")
    assert r.verdict() is ParityVerdict.ERROR
    assert not r.x86_compiled
    assert any("frontend pipeline failed" in d for d in r.detail), r.detail


def test_check_parity_source_include_stdlib_uncovered():
    """`include_stdlib=True` pulls in the float-typed stdlib the LLVM
    backend does not yet cover, so even a trivial program builds to
    UNCOVERED — pinning the corpus's `include_stdlib=False` rationale
    and exercising the keyword flag."""
    r = llvm_parity.check_parity_source(
        "fn main() -> i32 { 42 }", "stdlib_on", include_stdlib=True)
    assert r.verdict() is ParityVerdict.UNCOVERED
    assert not r.is_parity_defect()


def test_parity_corpus_is_substantial_and_well_formed():
    """The curated corpus is non-trivial and every entry has a unique
    non-blank name and a non-blank source (the module-load guard
    enforces this; pin it explicitly too)."""
    assert len(llvm_parity.PARITY_CORPUS) >= 20
    names = [name for name, _ in llvm_parity.PARITY_CORPUS]
    assert len(names) == len(set(names)), "corpus names must be unique"
    for name, source in llvm_parity.PARITY_CORPUS:
        assert name.strip() and source.strip()
    llvm_parity._check_parity_corpus()


def test_run_parity_corpus_returns_one_result_per_entry():
    """`run_parity_corpus` returns one ParityResult per corpus entry,
    in corpus order, each naming its program."""
    results = llvm_parity.run_parity_corpus()
    assert len(results) == len(llvm_parity.PARITY_CORPUS)
    for result, (name, _src) in zip(results, llvm_parity.PARITY_CORPUS):
        assert result.program == name


def test_parity_corpus_gate():
    """THE STAGE 207 MOCK-PATH PARITY GATE. Every program in the
    curated corpus — real Helix source exercising the LLVM backend's
    covered op surface — must structurally MATCH across the x86_64 and
    LLVM backends. A covered op regressing to UNCOVERED, or the LLVM
    backend emitting malformed IR (MISMATCH) or crashing (ERROR),
    breaks this gate."""
    results = llvm_parity.run_parity_corpus()
    non_match = [(r.program, r.verdict().name, r.detail)
                 for r in results
                 if r.verdict() is not ParityVerdict.MATCH]
    assert not non_match, f"corpus programs not MATCH: {non_match}"


# --------------------------------------------------------------------------
# real-execution result model + toolchain detection (chunk C)
# --------------------------------------------------------------------------
def _real(attempted, passed, findings=()):
    """A MATCH-shaped ParityResult with the given real-* fields."""
    return ParityResult(
        program="p", x86_compiled=True, llvm_emitted=True,
        llvm_failed_closed=False, llvm_mock_clean=True, detail=(),
        real_attempted=attempted, real_passed=passed,
        real_findings=findings)


def test_real_status_derivation():
    """real_status() derives the 4-state real outcome from the real-*
    fields: NOT_RUN / DEFERRED / PASS / FAIL."""
    assert _real(False, None).real_status() is RealParityStatus.NOT_RUN
    assert _real(True, None, ("no toolchain",)).real_status() is \
        RealParityStatus.DEFERRED
    assert _real(True, True).real_status() is RealParityStatus.PASS
    assert _real(True, False, ("exit 1 vs 0",)).real_status() is \
        RealParityStatus.FAIL


def test_real_status_independent_of_verdict():
    """A structural MATCH can still carry a real FAIL — `real_status()`
    and `verdict()` are independent axes. That a covered, structurally
    matching program can still behave differently at runtime is exactly
    what the chunk-D real-execution path exists to catch."""
    r = _real(True, False, ("stdout differed",))
    assert r.verdict() is ParityVerdict.MATCH
    assert r.real_status() is RealParityStatus.FAIL
    assert not r.is_parity_defect()  # is_parity_defect is structural only


def test_parity_result_accepts_deferred_real():
    """A DEFERRED real run with a recorded reason constructs cleanly —
    it is a legal 4-state outcome, not a silent failure."""
    r = _real(True, None, ("no clang inside WSL — real run DEFERRED",))
    assert r.real_status() is RealParityStatus.DEFERRED
    assert r.real_passed is None and r.real_findings


def test_real_exec_support_rejects_clang_without_wsl():
    """`clang` inside WSL implies WSL itself is available — a support
    result claiming clang but no WSL is illegal."""
    with pytest.raises(ValueError, match="implies WSL"):
        llvm_parity.RealExecSupport(
            wsl_available=False, wsl_clang="/usr/bin/clang",
            detail=("x",))


def test_real_exec_support_rejects_empty_detail():
    """A support result must always explain what it found — an empty
    detail would make a DEFERRED silent about why."""
    with pytest.raises(ValueError, match="detail is empty"):
        llvm_parity.RealExecSupport(
            wsl_available=False, wsl_clang=None, detail=())


def test_detect_real_exec_support_is_consistent():
    """`detect_real_exec_support` returns a coherent RealExecSupport,
    whatever this machine actually has: non-empty self-explaining
    detail, and `can_run_real()` is exactly `wsl_available AND a clang
    was found`."""
    s = llvm_parity.detect_real_exec_support()
    assert isinstance(s, llvm_parity.RealExecSupport)
    assert s.detail  # always explains itself
    assert s.can_run_real() == (
        s.wsl_available and s.wsl_clang is not None)
    if s.wsl_clang is not None:
        assert s.wsl_available  # the __post_init__ invariant


def test_probe_wsl_clang_returns_path_on_success(monkeypatch):
    """A WSL probe exiting 0 with a path on stdout yields that path."""
    class _Proc:
        returncode = 0
        stdout = "/usr/bin/clang\n"
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    assert llvm_parity._probe_wsl_clang("wsl") == "/usr/bin/clang"


def test_probe_wsl_clang_none_on_nonzero_exit(monkeypatch):
    """A WSL probe exiting non-zero (clang absent) yields None."""
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "clang: not found"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    assert llvm_parity._probe_wsl_clang("wsl") is None


def test_probe_wsl_clang_handles_subprocess_failure(monkeypatch):
    """A timeout or OSError from the WSL probe yields None (clang not
    found) — never an escaping traceback; detection fails safe to
    DEFERRED."""
    def _oserror(*a, **k):
        raise OSError("wsl vanished between detection and probe")
    monkeypatch.setattr(subprocess, "run", _oserror)
    assert llvm_parity._probe_wsl_clang("wsl") is None

    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="wsl", timeout=20)
    monkeypatch.setattr(subprocess, "run", _timeout)
    assert llvm_parity._probe_wsl_clang("wsl") is None


def test_probe_wsl_clang_none_on_empty_stdout(monkeypatch):
    """A WSL probe exiting 0 but printing nothing usable (blank
    stdout) yields None — a 0 exit alone is not a found clang."""
    class _Proc:
        returncode = 0
        stdout = "   \n"
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    assert llvm_parity._probe_wsl_clang("wsl") is None


def test_probe_wsl_clang_takes_last_line(monkeypatch):
    """A login shell may print a profile banner before `command -v`'s
    output — the probe takes the FINAL non-blank stdout line as the
    path, not the whole multi-line blob."""
    class _Proc:
        returncode = 0
        stdout = "profile banner line\n/usr/bin/clang\n"
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    assert llvm_parity._probe_wsl_clang("wsl") == "/usr/bin/clang"


def test_probe_wsl_clang_none_on_non_path_token(monkeypatch):
    """A `command -v` result that is not an absolute path (a builtin /
    alias token) fails safe to None rather than being trusted as a
    compiler path."""
    class _Proc:
        returncode = 0
        stdout = "clang\n"
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    assert llvm_parity._probe_wsl_clang("wsl") is None
