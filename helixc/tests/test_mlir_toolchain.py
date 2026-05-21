"""Tests for helixc.ir.mlir.toolchain — v3.0 Phase E, Stage 211: the
MLIR capability-detection substrate.

`detect_mlir_support()` probes the two real MLIR surfaces (the
in-process Python bindings and the `mlir-opt` CLI) and returns a
frozen `MLIRSupport`. Phase E is mock-path-first: a binding-less
machine must yield a clean "not available" result, never an
ImportError at compiler-import time.

These tests pin: the `MLIRSupport` type rejects every illegal field
shape; the predicates derive correctly; `detect_mlir_support` probes
each surface independently (a partial bindings install is not usable);
and — the load-bearing guard — the module imports with NO MLIR
bindings present (the Stage 210 decision's no-top-level-`import mlir`
rule).
"""
from __future__ import annotations

import importlib
import shutil

import pytest

from helixc.ir.mlir import toolchain
from helixc.ir.mlir.toolchain import MLIRSupport, detect_mlir_support


# --------------------------------------------------------------------------
# MLIRSupport — __post_init__ rejects illegal field shapes
# --------------------------------------------------------------------------
def test_mlir_support_rejects_dialects_without_bindings():
    """A dialect sub-module cannot import without the core package, so
    `dialects=True` with `bindings=False` is illegal."""
    with pytest.raises(ValueError, match="dialects=True but bindings"):
        MLIRSupport(bindings=False, dialects=True, mlir_opt=None,
                    detail=("x",))


def test_mlir_support_rejects_empty_detail():
    """A support result must always explain what it found — an empty
    detail would make a DEFERRED silent about why."""
    with pytest.raises(ValueError, match="detail is empty"):
        MLIRSupport(bindings=False, dialects=False, mlir_opt=None,
                    detail=())


def test_mlir_support_rejects_blank_detail_entry():
    """A blank / non-str detail entry is a reason-shaped object with no
    reason — rejected."""
    with pytest.raises(ValueError, match="blank or non-str"):
        MLIRSupport(bindings=False, dialects=False, mlir_opt=None,
                    detail=("   ",))


# --------------------------------------------------------------------------
# MLIRSupport — derived predicates
# --------------------------------------------------------------------------
def test_mlir_support_predicates():
    """`can_use_bindings` needs bindings AND dialects; `can_use_mlir_opt`
    needs a tool path; `is_available` is either surface."""
    # nothing available
    none = MLIRSupport(bindings=False, dialects=False, mlir_opt=None,
                       detail=("nothing",))
    assert not none.can_use_bindings()
    assert not none.can_use_mlir_opt()
    assert not none.is_available()
    # partial bindings install — core present, a dialect absent
    partial = MLIRSupport(bindings=True, dialects=False, mlir_opt=None,
                          detail=("partial",))
    assert not partial.can_use_bindings()  # partial is NOT usable
    assert not partial.is_available()
    # full bindings
    full = MLIRSupport(bindings=True, dialects=True, mlir_opt=None,
                       detail=("full bindings",))
    assert full.can_use_bindings()
    assert full.is_available()
    # mlir-opt only
    cli = MLIRSupport(bindings=False, dialects=False,
                      mlir_opt="/usr/bin/mlir-opt", detail=("cli",))
    assert cli.can_use_mlir_opt()
    assert cli.is_available()


# --------------------------------------------------------------------------
# detect_mlir_support — the probe
# --------------------------------------------------------------------------
def test_detect_mlir_support_on_this_machine():
    """`detect_mlir_support` returns a coherent MLIRSupport whatever
    this machine has: non-empty self-explaining detail, and the
    predicates derive consistently from the fields."""
    s = detect_mlir_support()
    assert isinstance(s, MLIRSupport)
    assert s.detail
    assert s.can_use_bindings() == (s.bindings and s.dialects)
    assert s.can_use_mlir_opt() == (s.mlir_opt is not None)
    assert s.is_available() == (
        s.can_use_bindings() or s.can_use_mlir_opt())
    # dialects set implies bindings set (the __post_init__ invariant)
    if s.dialects:
        assert s.bindings


def test_detect_mlir_support_full_bindings(monkeypatch):
    """When core `mlir.ir` and every required dialect sub-module
    import, the bindings surface is fully usable — and the probe
    genuinely imports EACH required dialect, not just a blanket
    success a deleted dialect loop would also pass."""
    probed: list[str] = []

    def _import(name):
        probed.append(name)
        return object()

    monkeypatch.setattr(importlib, "import_module", _import)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    s = detect_mlir_support()
    assert s.bindings is True and s.dialects is True
    assert s.can_use_bindings() is True
    # the probe imported core mlir.ir AND every required dialect.
    assert "mlir.ir" in probed
    for dia in toolchain._REQUIRED_MLIR_DIALECTS:
        assert f"mlir.dialects.{dia}" in probed, dia
    assert any("all" in d and "dialect" in d for d in s.detail), s.detail


def test_detect_mlir_support_captures_broken_build(monkeypatch):
    """A non-ImportError exception from a binding import (an
    installed-but-broken build, e.g. a native-lib mismatch) is
    CAPTURED, not raised — the broad `except Exception` exists for
    exactly this, and narrowing it to `except ImportError` would
    regress to an uncaught traceback at compiler-import time."""
    def _import(name):
        raise RuntimeError("broken MLIR build — native lib mismatch")
    monkeypatch.setattr(importlib, "import_module", _import)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    s = detect_mlir_support()
    assert s.bindings is False and s.dialects is False
    assert any("RuntimeError" in d for d in s.detail), s.detail


def test_detect_mlir_support_partial_install(monkeypatch):
    """A partial install — core `mlir.ir` present but a required
    dialect sub-module absent — is reported bindings-not-usable, not
    passed and then failed later."""
    def _import(name):
        if name == "mlir.dialects.gpu":
            raise ImportError("no gpu dialect in this build")
        return object()
    monkeypatch.setattr(importlib, "import_module", _import)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    s = detect_mlir_support()
    assert s.bindings is True       # core imported
    assert s.dialects is False      # ... but a dialect is missing
    assert not s.can_use_bindings()
    assert any("gpu" in d for d in s.detail), s.detail


def test_detect_mlir_support_bindings_absent(monkeypatch):
    """An absent core `mlir.ir` is captured (not raised) as
    bindings=False with a diagnostic."""
    def _import(name):
        raise ModuleNotFoundError("No module named 'mlir'")
    monkeypatch.setattr(importlib, "import_module", _import)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    s = detect_mlir_support()
    assert s.bindings is False and s.dialects is False
    assert any("bindings absent" in d for d in s.detail), s.detail


def test_detect_mlir_support_mlir_opt_present(monkeypatch):
    """The `mlir-opt` CLI is an INDEPENDENT surface — with the bindings
    forced absent, `mlir-opt` on PATH alone makes `is_available()`
    hold. (The bindings are mocked absent so the test is host-
    independent, not reliant on this machine lacking them.)"""
    def _no_bindings(name):
        raise ModuleNotFoundError("No module named 'mlir'")
    monkeypatch.setattr(importlib, "import_module", _no_bindings)
    monkeypatch.setattr(
        shutil, "which",
        lambda name: "/usr/bin/mlir-opt" if name == "mlir-opt"
        else None)
    s = detect_mlir_support()
    assert s.bindings is False and not s.can_use_bindings()
    assert s.can_use_mlir_opt() is True
    assert s.mlir_opt == "/usr/bin/mlir-opt"
    assert s.is_available() is True


def test_required_mlir_dialects_well_formed():
    """`_REQUIRED_MLIR_DIALECTS` is a non-empty tuple of unique,
    identifier-shaped names — the module-load guard
    `_check_mlir_dialects` enforces this (a typo / duplicate would
    silently skew what a 'full' bindings install means); pin it
    explicitly too."""
    dialects = toolchain._REQUIRED_MLIR_DIALECTS
    assert dialects, "must not be empty"
    assert len(dialects) == len(set(dialects)), "no duplicates"
    for dia in dialects:
        assert isinstance(dia, str) and dia.isidentifier(), dia
    # the guard is callable and passes for the current tuple.
    toolchain._check_mlir_dialects()


# --------------------------------------------------------------------------
# the no-top-level-`import mlir` guard
# --------------------------------------------------------------------------
def test_toolchain_module_imports_without_mlir_bindings():
    """THE LOAD-BEARING GUARD (Stage 210 decision, section 3.2): the
    module must import — and `detect_mlir_support` must be callable —
    on a machine with NO MLIR bindings. This test running at all proves
    `helixc.ir.mlir.toolchain` imported cleanly here (this dev machine
    has no `mlir`); a stray top-level `import mlir` would have made the
    import fail. Pin it explicitly."""
    assert toolchain.detect_mlir_support is detect_mlir_support
    # the probe runs and returns a result, bindings or not
    assert isinstance(detect_mlir_support(), MLIRSupport)
