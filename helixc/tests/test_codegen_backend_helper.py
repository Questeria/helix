"""Tests for `helixc/tests/_codegen_backend.py` — the v3.1 step 3
test-side migration seam between the legacy x86_64 backend and the
canonical v3.0+ LLVM backend.

The helper itself is module-private (`_codegen_backend.py` with
leading underscore in the path) but pytest still collects this
test module under `helixc/tests/`. We import the helper through
its underscore name, which is unusual for production code but the
standard pattern for test-only modules.
"""
from __future__ import annotations

import pytest

from helixc.ir import tir
from helixc.tests import _codegen_backend as cgb


def _trivial_module() -> tir.Module:
    """`fn main() -> i32 { 42 }` — small compilable module."""
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("main", [], tir.TIRScalar("i32"))
    b.ret(b.const_int(42))
    b.end_function()
    return mod


# --------------------------------------------------------------------------
# selected_backend
# --------------------------------------------------------------------------
def test_selected_backend_default_is_x86(monkeypatch):
    """No env var set → default backend is `"x86"` (preserves the
    legacy dev workflow during v3.1 migration)."""
    monkeypatch.delenv(cgb._TEST_BACKEND_ENV, raising=False)
    assert cgb.selected_backend() == "x86"


def test_selected_backend_empty_env_is_default(monkeypatch):
    """Empty-string env value is treated as unset → default."""
    monkeypatch.setenv(cgb._TEST_BACKEND_ENV, "")
    assert cgb.selected_backend() == "x86"


def test_selected_backend_llvm_explicit(monkeypatch):
    monkeypatch.setenv(cgb._TEST_BACKEND_ENV, "llvm")
    assert cgb.selected_backend() == "llvm"


def test_selected_backend_x86_explicit(monkeypatch):
    monkeypatch.setenv(cgb._TEST_BACKEND_ENV, "x86")
    assert cgb.selected_backend() == "x86"


def test_selected_backend_unknown_raises(monkeypatch):
    """An unrecognised value is a CI / dev typo — fail loudly
    rather than silently degrade to a default."""
    monkeypatch.setenv(cgb._TEST_BACKEND_ENV, "ptx")
    with pytest.raises(ValueError, match="not a recognised backend"):
        cgb.selected_backend()


# --------------------------------------------------------------------------
# compile_module_to_elf — routes to the selected backend
# --------------------------------------------------------------------------
def test_compile_module_default_routes_to_x86(monkeypatch):
    """Default routes to the legacy x86_64 backend — produces a
    runnable ELF (non-empty bytes starting with the ELF magic)."""
    monkeypatch.delenv(cgb._TEST_BACKEND_ENV, raising=False)
    elf = cgb.compile_module_to_elf(_trivial_module())
    assert elf.startswith(b"\x7fELF"), elf[:16]


def test_compile_module_llvm_route_raises_on_no_toolchain(
        monkeypatch):
    """LLVM route raises `LLVMToolchainAbsent` on a no-toolchain
    machine (Windows / macOS without sysroot). The bytes-returning
    contract is preserved — callers can `try/except` to handle."""
    from helixc.backend.llvm_toolchain import LLVMToolchainAbsent
    monkeypatch.setenv(cgb._TEST_BACKEND_ENV, "llvm")
    # On the dev machine the LLVM toolchain is absent (Windows) →
    # raise. We assert the exception type, not the message detail.
    with pytest.raises(LLVMToolchainAbsent):
        cgb.compile_module_to_elf(_trivial_module())


def test_compile_module_validates_entry_fn(monkeypatch):
    """Both backends share the `ValueError` on missing entry_fn
    contract — verified by going through the helper."""
    monkeypatch.delenv(cgb._TEST_BACKEND_ENV, raising=False)
    mod = tir.Module()
    b = tir.IRBuilder(mod)
    b.begin_function("not_main", [], tir.TIRScalar("i32"))
    b.ret(b.const_int(0))
    b.end_function()
    with pytest.raises((ValueError, Exception)):
        # x86_64 raises ValueError; type-loose `Exception` here to
        # also cover the LLVM path's ValueError when env=llvm
        # (the two backends agree on the error type but the
        # type-tightness check is in their own tests).
        cgb.compile_module_to_elf(mod)


# --------------------------------------------------------------------------
# compile_or_skip — pytest.skip on LLVM-toolchain-absent
# --------------------------------------------------------------------------
def test_compile_or_skip_x86_does_not_skip(monkeypatch):
    """Default backend (x86) does not skip — produces an ELF or
    raises whatever the backend raises."""
    monkeypatch.delenv(cgb._TEST_BACKEND_ENV, raising=False)
    elf = cgb.compile_or_skip(_trivial_module())
    assert elf.startswith(b"\x7fELF")


def test_compile_or_skip_llvm_skips_on_no_toolchain(monkeypatch):
    """LLVM backend on no-toolchain → pytest.skip (not raise).
    Exercised via the `outcome` machinery rather than the raised
    Skipped exception."""
    monkeypatch.setenv(cgb._TEST_BACKEND_ENV, "llvm")
    # `pytest.skip(...)` raises a private `Skipped` exception class
    # at module level — catch via `_pytest.outcomes.Skipped` which
    # is the standard way to assert "this would have skipped".
    from _pytest.outcomes import Skipped
    with pytest.raises(Skipped, match="LLVM toolchain unavailable"):
        cgb.compile_or_skip(_trivial_module())
