"""Tests for helixc.backend.llvm_parity — v3.0 Phase D, Stage 207
chunks A-E: the x86_64-vs-LLVM mock structural-parity harness, the
curated source-program corpus + parity gate, the real-execution
result model + toolchain detection, the program-run substrate, and
the real-execution comparison + `attempt_real` wiring.

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
reports whether this machine can run the real comparison; (chunk D)
the `_ProgramRun` substrate builds and runs a backend's output under
WSL, capturing observable behaviour and never raising on failure;
(chunk E) `_compare_runs` decides observable-behaviour parity and
`check_parity(attempt_real=True)` fills the real-* fields (PASS / FAIL
where the toolchain is present, DEFERRED where it is absent).
"""
from __future__ import annotations

import copy
import os
import shutil
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
    """`main` with a QUOTE op — a Stage 206-R residual op the LLVM
    backend does not yet lower (x86_64 does). The canonical
    UNCOVERED case.

    HISTORY: this helper has been repointed several times as 206-R
    chunks land:
      - originally: print_int — lowered 2026-05-24 (commit c7b7cec)
        via the `@__helix_print_int` internal helper.
      - then: write_file — lowered (commit ac366c6) inline via libc
        open/write/close.
      - then: TRACE_ENTRY — lowered (this chunk) via the
        `@__helix_trace_event` void helper + ring buffer.
      - now: QUOTE — still residual; the AGI metaprogramming family
        (QUOTE / SPLICE / MODIFY / REFLECT_HASH) is the last 206-R
        group, gated on AST/cell infrastructure.

    The function name stays `_print_int_module` to avoid renaming
    every call site (the name names a ROLE — "the residual-op
    fixture" — not the specific op).

    NOTE: if a future chunk lowers QUOTE, repoint this helper at the
    next still-residual op (SPLICE / MODIFY / REFLECT_HASH) so the
    UNCOVERED tests stay meaningful."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("main", [], _i32())
    h = b.emit(tir.OpKind.QUOTE, result_ty=_i32(),
               attrs={"ast_handle": 0})
    b.ret(h)
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
    """A QUOTE op (a Stage 206-R residual op) -> UNCOVERED: the
    x86_64 backend compiles it, the LLVM backend fails closed
    loudly. NOT a parity defect — the gate accepts it.

    HISTORY: the canonical UNCOVERED case has migrated through
    print_int -> write_file -> TRACE_ENTRY -> QUOTE as each
    206-R chunk landed; the diagnostic check follows the fixture."""
    r = llvm_parity.check_parity(_print_int_module(), "print_int")
    assert r.verdict() is ParityVerdict.UNCOVERED
    assert r.x86_compiled
    assert not r.llvm_emitted and r.llvm_failed_closed
    assert not r.is_parity_defect()
    # the diagnostic names the uncovered op so the residual is visible
    assert any("agi.quote" in d.lower() for d in r.detail), r.detail


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


# --------------------------------------------------------------------------
# the program-run substrate (chunk D)
# --------------------------------------------------------------------------
_PR = llvm_parity._ProgramRun


def test_program_run_rejects_ran_without_exit_code():
    """A completed run (`ran=True`) must carry a concrete exit code."""
    with pytest.raises(ValueError, match="exit_code is None"):
        _PR(label="x86_64", ran=True, exit_code=None, stdout="",
            stderr="", findings=())


def test_program_run_rejects_ran_with_findings():
    """A completed run carries no failure diagnostic."""
    with pytest.raises(ValueError, match="findings is non-empty"):
        _PR(label="x86_64", ran=True, exit_code=0, stdout="",
            stderr="", findings=("spurious",))


def test_program_run_rejects_failed_with_exit_code():
    """A failed run (`ran=False`) has no exit code."""
    with pytest.raises(ValueError, match="ran=False but exit_code"):
        _PR(label="llvm", ran=False, exit_code=0, stdout="",
            stderr="", findings=("boom",))


def test_program_run_rejects_failed_without_findings():
    """A build/run failure must carry a diagnostic — silent failure
    forbidden."""
    with pytest.raises(ValueError, match="findings is empty"):
        _PR(label="llvm", ran=False, exit_code=None, stdout="",
            stderr="", findings=())


def test_program_run_rejects_failed_with_output():
    """A run that did not complete captured no output."""
    with pytest.raises(ValueError, match="stdout/stderr is"):
        _PR(label="llvm", ran=False, exit_code=None, stdout="leak",
            stderr="", findings=("boom",))


def test_program_run_rejects_blank_label():
    """Every run must name the backend it ran — a blank label is
    rejected, matching how every other string-identity field in the
    module is guarded."""
    with pytest.raises(ValueError, match="label is empty"):
        _PR(label="  ", ran=True, exit_code=0, stdout="", stderr="",
            findings=())


def test_program_run_classmethods():
    """`.failed` and `.completed` build the two legal shapes."""
    f = _PR.failed("llvm", "clang exploded")
    assert f.ran is False and f.exit_code is None
    assert f.findings == ("clang exploded",) and f.stdout == ""
    c = _PR.completed("x86_64", 42, "out", "err")
    assert c.ran is True and c.exit_code == 42
    assert c.stdout == "out" and c.stderr == "err" and c.findings == ()


def test_win_to_wsl():
    """A Windows path translates to its `/mnt/<drive>` WSL form, with
    any drive letter, lower-cased and forward-slashed."""
    assert llvm_parity._win_to_wsl("C:\\foo\\bar") == "/mnt/c/foo/bar"
    assert llvm_parity._win_to_wsl("D:\\x\\y.bin") == "/mnt/d/x/y.bin"
    # the result is always WSL-rooted and contains no backslashes.
    out = llvm_parity._win_to_wsl("C:\\a b\\c.bin")
    assert out.startswith("/mnt/c/") and "\\" not in out


def test_run_under_wsl_failed_when_no_wsl(monkeypatch):
    """No `wsl` on PATH -> a captured failure, not a raise."""
    monkeypatch.setattr(llvm_parity.shutil, "which", lambda n: None)
    r = llvm_parity._run_under_wsl("x86_64", "p", "C:\\t\\x.bin")
    assert r.ran is False
    assert any("wsl" in f for f in r.findings)


def test_run_under_wsl_failed_on_timeout(monkeypatch):
    """A run that times out is a captured failure, never a traceback."""
    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="wsl", timeout=30)
    monkeypatch.setattr(subprocess, "run", _timeout)
    r = llvm_parity._run_under_wsl("x86_64", "p", "C:\\t\\x.bin")
    assert r.ran is False
    assert any("timed out" in f for f in r.findings)


def test_run_under_wsl_failed_on_oserror(monkeypatch):
    """An OSError from `wsl` is a captured failure."""
    def _oserror(*a, **k):
        raise OSError("wsl vanished")
    monkeypatch.setattr(subprocess, "run", _oserror)
    r = llvm_parity._run_under_wsl("llvm", "p", "C:\\t\\x.bin")
    assert r.ran is False
    assert any("unusable" in f for f in r.findings)


def test_run_under_wsl_completed_on_success(monkeypatch):
    """A program that runs to completion yields a completed run with
    its captured exit code and output — chmod (a separate WSL call)
    succeeds, then the program runs."""
    calls = []

    class _Proc:
        def __init__(self, rc, out, err):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:          # chmod +x
            return _Proc(0, "", "")
        return _Proc(9, "the output", "")  # the program run

    monkeypatch.setattr(subprocess, "run", fake_run)
    r = llvm_parity._run_under_wsl("x86_64", "p", "C:\\t\\x.bin")
    assert r.ran is True
    assert r.exit_code == 9 and r.stdout == "the output"
    assert len(calls) == 2  # chmod, then run


def test_run_under_wsl_failed_when_chmod_fails(monkeypatch):
    """If `chmod +x` fails the program never runs — that is a captured
    failure, NOT chmod's exit code masquerading as the program's (the
    `ran=True` for a program that never ran the `&&`-chain would give).
    """
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "chmod: cannot access"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    r = llvm_parity._run_under_wsl("x86_64", "p", "C:\\t\\x.bin")
    assert r.ran is False
    assert any("chmod +x failed" in f for f in r.findings)


def test_run_under_wsl_failed_when_program_does_not_launch(monkeypatch):
    """A `wsl` exit outside 0-255 is the launcher's own error code (the
    WSL service down / no distro), not the program's — it is captured
    as a failure, never a `ran=True` with the plumbing artifact folded
    in as exit_code."""
    calls = []

    class _Proc:
        def __init__(self, rc, out, err):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:          # chmod +x succeeds
            return _Proc(0, "", "")
        return _Proc(4294967295, "", "no distribution")  # launcher fail

    monkeypatch.setattr(subprocess, "run", fake_run)
    r = llvm_parity._run_under_wsl("x86_64", "p", "C:\\t\\x.bin")
    assert r.ran is False
    assert any("did not run under WSL" in f for f in r.findings)


def test_run_under_wsl_failed_on_run_timeout(monkeypatch):
    """A timeout on the RUN step (after chmod succeeded) is captured —
    the step-2 timeout catch, distinct from the step-1 chmod catch."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:          # chmod +x succeeds
            class _P:
                returncode = 0
                stdout = ""
                stderr = ""
            return _P()
        raise subprocess.TimeoutExpired(cmd="wsl", timeout=30)

    monkeypatch.setattr(subprocess, "run", fake_run)
    r = llvm_parity._run_under_wsl("x86_64", "p", "C:\\t\\x.bin")
    assert r.ran is False
    assert any("timed out" in f for f in r.findings)


def test_run_under_wsl_completed_on_in_range_exit(monkeypatch):
    """A legitimate in-range exit code (127 — a real program may return
    it) passes through as a genuine completed run: the launcher-failure
    guard rejects only codes OUTSIDE 0-255, not real program exits."""
    calls = []

    class _Proc:
        def __init__(self, rc, out, err):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:          # chmod +x succeeds
            return _Proc(0, "", "")
        return _Proc(127, "", "")    # the program genuinely exits 127

    monkeypatch.setattr(subprocess, "run", fake_run)
    r = llvm_parity._run_under_wsl("x86_64", "p", "C:\\t\\x.bin")
    assert r.ran is True
    assert r.exit_code == 127


# --------------------------------------------------------------------------
# real-execution comparison + attempt_real wiring (chunk E)
# --------------------------------------------------------------------------
def _ok_support():
    """A RealExecSupport whose `can_run_real()` is True."""
    return llvm_parity.RealExecSupport(
        wsl_available=True, wsl_clang="/usr/bin/clang",
        detail=("WSL + clang available",))


def test_compare_runs_pass_on_identical():
    """Two completed runs with identical exit code / stdout / stderr
    are observable-behaviour PASS, carrying no findings."""
    a = _PR.completed("x86_64", 42, "out", "")
    b = _PR.completed("llvm", 42, "out", "")
    passed, findings = llvm_parity._compare_runs(a, b)
    assert passed is True and findings == ()


def test_compare_runs_fail_on_exit_code_diff():
    """A differing exit code is an observable-behaviour FAIL."""
    a = _PR.completed("x86_64", 42, "", "")
    b = _PR.completed("llvm", 7, "", "")
    passed, findings = llvm_parity._compare_runs(a, b)
    assert passed is False
    assert any("exit code differs" in f for f in findings)


def test_compare_runs_fail_on_stdout_diff():
    """Differing stdout is an observable-behaviour FAIL."""
    a = _PR.completed("x86_64", 0, "hello", "")
    b = _PR.completed("llvm", 0, "HELLO", "")
    passed, findings = llvm_parity._compare_runs(a, b)
    assert passed is False
    assert any("stdout differs" in f for f in findings)


def test_compare_runs_fail_when_llvm_did_not_run():
    """If the LLVM backend's output failed to build/run, there is
    nothing to compare — a FAIL naming the LLVM-side failure."""
    a = _PR.completed("x86_64", 0, "", "")
    b = _PR.failed("llvm", "clang rejected the IR")
    passed, findings = llvm_parity._compare_runs(a, b)
    assert passed is False
    assert any("LLVM backend's output did not build/run" in f
               for f in findings)


def test_compare_runs_fail_when_x86_did_not_run():
    """If the x86_64 baseline failed to build/run, parity cannot be
    established — a FAIL naming the baseline failure."""
    a = _PR.failed("x86_64", "ELF would not run")
    b = _PR.completed("llvm", 0, "", "")
    passed, findings = llvm_parity._compare_runs(a, b)
    assert passed is False
    assert any("x86_64 baseline did not build/run" in f
               for f in findings)


def test_check_parity_attempt_real_deferred_here():
    """`attempt_real=True` on a MATCH program: with no `clang` inside
    WSL on this dev machine, the real comparison is DEFERRED — the
    structural verdict stays MATCH, `real_status()` is DEFERRED, and
    it is not a parity defect."""
    r = llvm_parity.check_parity(
        _const_module(), "real_deferred", attempt_real=True)
    assert r.verdict() is ParityVerdict.MATCH
    assert r.real_status() is RealParityStatus.DEFERRED
    assert not r.is_parity_defect()
    assert r.real_findings  # the DEFERRED reason is recorded


def test_check_parity_attempt_real_skipped_for_non_match():
    """`attempt_real=True` runs the real comparison ONLY for a
    structural MATCH — a non-MATCH program (here UNCOVERED) has no
    pair of comparable executables, so real_status stays NOT_RUN."""
    r = llvm_parity.check_parity(
        _print_int_module(), "uncov", attempt_real=True)
    assert r.verdict() is ParityVerdict.UNCOVERED
    assert r.real_status() is RealParityStatus.NOT_RUN


def test_check_parity_attempt_real_pass(monkeypatch):
    """With the toolchain present and both backends producing
    identical observable behaviour, `attempt_real=True` -> real PASS."""
    monkeypatch.setattr(llvm_parity, "detect_real_exec_support",
                        _ok_support)
    monkeypatch.setattr(
        llvm_parity, "_run_x86_program",
        lambda m, p: _PR.completed("x86_64", 42, "", ""))
    monkeypatch.setattr(
        llvm_parity, "_run_llvm_program",
        lambda m, p, c: _PR.completed("llvm", 42, "", ""))
    r = llvm_parity.check_parity(
        _const_module(), "real_pass", attempt_real=True)
    assert r.verdict() is ParityVerdict.MATCH
    assert r.real_status() is RealParityStatus.PASS


def test_check_parity_attempt_real_fail(monkeypatch):
    """With the toolchain present but the two backends producing
    DIFFERENT observable behaviour, `attempt_real=True` -> real FAIL
    with a diagnostic — a real parity defect the structural mock path
    could not have caught."""
    monkeypatch.setattr(llvm_parity, "detect_real_exec_support",
                        _ok_support)
    monkeypatch.setattr(
        llvm_parity, "_run_x86_program",
        lambda m, p: _PR.completed("x86_64", 42, "", ""))
    monkeypatch.setattr(
        llvm_parity, "_run_llvm_program",
        lambda m, p, c: _PR.completed("llvm", 7, "", ""))
    r = llvm_parity.check_parity(
        _const_module(), "real_fail", attempt_real=True)
    assert r.verdict() is ParityVerdict.MATCH  # structural MATCH ...
    assert r.real_status() is RealParityStatus.FAIL  # ... real FAIL
    assert any("exit code differs" in f for f in r.real_findings)


def test_check_parity_source_attempt_real_deferred():
    """`attempt_real` passes through `check_parity_source` — a MATCH
    source program is DEFERRED here (no toolchain)."""
    r = llvm_parity.check_parity_source(
        "fn main() -> i32 { 42 }", "src_real", attempt_real=True)
    assert r.verdict() is ParityVerdict.MATCH
    assert r.real_status() is RealParityStatus.DEFERRED


def test_run_parity_corpus_attempt_real(monkeypatch):
    """`attempt_real` passes through `run_parity_corpus` to every
    program. With detection forced to no-toolchain, every result is a
    structural MATCH with a DEFERRED real comparison — no defects."""
    monkeypatch.setattr(
        llvm_parity, "detect_real_exec_support",
        lambda: llvm_parity.RealExecSupport(
            wsl_available=True, wsl_clang=None,
            detail=("no clang inside WSL — DEFERRED",)))
    results = llvm_parity.run_parity_corpus(attempt_real=True)
    assert len(results) == len(llvm_parity.PARITY_CORPUS)
    for r in results:
        assert r.verdict() is ParityVerdict.MATCH
        assert r.real_status() is RealParityStatus.DEFERRED
        assert not r.is_parity_defect()


@pytest.mark.skipif(
    not llvm_parity.detect_real_exec_support().can_run_real(),
    reason="real-execution parity needs WSL + clang inside it")
def test_check_parity_real_execution_end_to_end():
    """End-to-end: with the toolchain present (WSL + clang), a covered
    program's observable behaviour MATCHES across both backends — a
    real PASS. The LLVM-side mirror of `test_run_x86_program_real_
    execution`; skipped on a machine without WSL + clang (this dev
    machine has no clang in WSL, so it runs only on a tooled CI
    runner)."""
    r = llvm_parity.check_parity_source(
        "fn main() -> i32 { 42 }", "real_e2e", attempt_real=True)
    assert r.verdict() is ParityVerdict.MATCH
    assert r.real_status() is RealParityStatus.PASS, r.real_findings


def test_run_x86_program_failed_on_compile_error(monkeypatch):
    """An x86_64 backend compile crash is captured as a failed run."""
    def boom(_mod):
        raise RuntimeError("x86 backend exploded")
    monkeypatch.setattr(llvm_parity, "compile_module_to_elf", boom)
    r = llvm_parity._run_x86_program(_const_module(), "x86_crash")
    assert r.ran is False
    assert any("x86_64 backend failed" in f and "RuntimeError" in f
               for f in r.findings)


def test_run_x86_program_failed_on_empty_elf(monkeypatch):
    """An empty ELF from the x86_64 backend is a captured failure —
    there is nothing to run."""
    monkeypatch.setattr(llvm_parity, "compile_module_to_elf",
                        lambda m: b"")
    r = llvm_parity._run_x86_program(_const_module(), "empty_elf")
    assert r.ran is False
    assert any("empty ELF" in f for f in r.findings)


def test_run_x86_program_failed_on_mkdtemp_error(monkeypatch):
    """A temp-directory-creation OSError is captured, not raised — the
    substrate's no-raise contract holds even when the OS will not give
    a temp dir."""
    def boom(*a, **k):
        raise OSError("no space left on device")
    monkeypatch.setattr(llvm_parity.tempfile, "mkdtemp", boom)
    r = llvm_parity._run_x86_program(_const_module(), "no_tmp")
    assert r.ran is False
    assert any("temp directory" in f for f in r.findings)


@pytest.mark.skipif(shutil.which("wsl") is None,
                    reason="real x86_64 execution needs WSL")
def test_run_x86_program_real_execution():
    """End-to-end: the x86_64 backend's ELF for `fn main() { 42 }`
    builds and runs under WSL with exit code 42 — the real x86 leg of
    the parity comparison."""
    r = llvm_parity._run_x86_program(_const_module(), "const42")
    assert r.ran is True, r.findings
    assert r.exit_code == 42


def test_run_llvm_program_failed_on_emit_error(monkeypatch):
    """An LLVM backend emit crash is captured as a failed run."""
    def boom(_mod):
        raise RuntimeError("llvm emitter exploded")
    monkeypatch.setattr(llvm_parity, "emit_module", boom)
    r = llvm_parity._run_llvm_program(
        _const_module(), "llvm_crash", "/usr/bin/clang")
    assert r.ran is False
    assert any("LLVM backend failed to emit" in f for f in r.findings)


def test_run_llvm_program_failed_on_clang_nonzero(monkeypatch):
    """clang exiting non-zero on the emitted IR is a captured failure
    carrying clang's diagnostic."""
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "error: bad IR"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    r = llvm_parity._run_llvm_program(
        _const_module(), "bad_ir", "/usr/bin/clang")
    assert r.ran is False
    assert any("clang failed to compile" in f and "bad IR" in f
               for f in r.findings)


def test_run_llvm_program_failed_on_no_artifact(monkeypatch):
    """clang exiting 0 but producing no executable is a captured
    failure — a 0 exit with no artifact is not a successful build."""
    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    monkeypatch.setattr(os.path, "getsize", lambda p: 0)
    r = llvm_parity._run_llvm_program(
        _const_module(), "no_exe", "/usr/bin/clang")
    assert r.ran is False
    assert any("no executable" in f for f in r.findings)


def test_run_llvm_program_completed_on_success(monkeypatch):
    """Happy path: emit IR, clang compiles it (exit 0, artifact
    present), the executable runs — a completed run carrying the
    captured exit code and output."""
    calls = []

    class _Proc:
        def __init__(self, rc, out, err):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:          # the clang compile
            return _Proc(0, "", "")
        if len(calls) == 2:          # chmod +x
            return _Proc(0, "", "")
        return _Proc(42, "hello", "")  # the program run

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os.path, "getsize", lambda p: 4096)
    r = llvm_parity._run_llvm_program(
        _const_module(), "ok", "/usr/bin/clang")
    assert r.ran is True, r.findings
    assert r.exit_code == 42 and r.stdout == "hello"
    assert len(calls) == 3  # clang compile, chmod, run
