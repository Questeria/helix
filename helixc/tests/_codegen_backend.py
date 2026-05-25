"""
helixc/tests/_codegen_backend.py — backend-selection helper for the
test suite (v3.1 step 3: test_codegen.py migration seam).

Background: v3.0.0 made the LLVM IR backend the canonical Helix
backend, but ~1000 `test_codegen.py` compile-AND-RUN tests still
import `compile_module_to_elf` directly from `helixc.backend.x86_64`
to produce runnable Linux ELF binaries. The LLVM path
(`compile_module_to_elf_via_llvm`) needs three external tools
(`llvm-as` / `llc` / `clang`) and a Linux sysroot to produce
equivalent output — neither is universally available on dev
machines (Windows, macOS without WSL).

This helper centralizes the backend choice so:

  - The DEFAULT remains `helixc.backend.x86_64.compile_module_to_elf`
    — preserves the dev-loop behaviour Windows / macOS contributors
    have today.
  - Setting `HELIX_TEST_BACKEND=llvm` in the environment swaps every
    call site to `compile_module_to_elf_via_llvm` — exercises the
    LLVM path on Linux CI / sysroot-configured workstations.
  - A single seam (this module) means flipping the default later
    is one edit, not 100 grep-and-replaces across the test files.

The Stage 221 cutover (LLVM as canonical compiler backend) is
already shipped at the CLI level (`helixc check --emit-llvm-ir`).
This helper completes the test-side migration without forcing a
test-suite break on the current dev workflow.

Exit contract on toolchain-absent:
  - The LLVM bytes wrapper raises `LLVMToolchainAbsent` when
    `HELIX_TEST_BACKEND=llvm` is set but the toolchain (or the
    Linux host / `HELIX_LLVM_CROSS` opt-in) is missing.
  - This helper translates that into a `pytest.skip(...)` via a
    convenience `compile_or_skip` wrapper, so a test that wants
    LLVM coverage but runs on a no-toolchain machine skips cleanly
    rather than failing with a confusing OSError.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helixc.ir import tir


# The env var name. Keep in `_TEST_BACKEND_ENV` so a future
# rename is grep-friendly across the test suite.
_TEST_BACKEND_ENV = "HELIX_TEST_BACKEND"

# Recognised values. "x86" is the legacy backend (default).
# "llvm" exercises the new canonical path. Any other value is a
# typo / misconfiguration — fail loudly so a CI flag drift surfaces.
_KNOWN_BACKENDS = frozenset({"x86", "llvm"})

# The default. Stays at "x86" through v3.1.x; will flip to "llvm"
# during the v3.1 close-out chunk once test_codegen migration is
# complete (so CI runs on the canonical path) and the last
# fallbacks have been audited.
_DEFAULT_BACKEND = "x86"


def selected_backend() -> str:
    """Return the active backend name (`"x86"` or `"llvm"`).

    Reads `HELIX_TEST_BACKEND` from the environment; falls back to
    `_DEFAULT_BACKEND`. An unrecognised value raises ValueError
    rather than silently degrading.
    """
    raw = os.environ.get(_TEST_BACKEND_ENV)
    if raw is None or raw == "":
        return _DEFAULT_BACKEND
    if raw not in _KNOWN_BACKENDS:
        raise ValueError(
            f"{_TEST_BACKEND_ENV}={raw!r} is not a recognised "
            f"backend (known: {sorted(_KNOWN_BACKENDS)})")
    return raw


def compile_module_to_elf(module: "tir.Module",
                          entry_fn: str = "main") -> bytes:
    """Backend-routed `compile_module_to_elf`. The default routes
    to `helixc.backend.x86_64.compile_module_to_elf`; setting
    `HELIX_TEST_BACKEND=llvm` routes to
    `helixc.backend.llvm_toolchain.compile_module_to_elf_via_llvm`.

    Raises whatever the underlying backend raises:
      - x86_64: `ValueError` on missing entry_fn,
        `NotImplementedError` on unsupported op kinds, etc.
      - LLVM: `ValueError` on missing entry_fn,
        `LLVMToolchainAbsent` on missing toolchain / non-Linux host,
        `LLVMToolchainError` on tool failure.

    Callers that want the toolchain-absent case to skip cleanly
    instead of failing should use `compile_or_skip` below.
    """
    backend = selected_backend()
    if backend == "x86":
        from helixc.backend.x86_64 import (
            compile_module_to_elf as _x86_compile)
        return _x86_compile(module, entry_fn)
    if backend == "llvm":
        from helixc.backend.llvm_toolchain import (
            compile_module_to_elf_via_llvm as _llvm_compile)
        return _llvm_compile(module, entry_fn)
    # Unreachable — selected_backend() already validated the value.
    raise ValueError(
        f"unreachable: selected_backend() returned {backend!r}")


def compile_or_skip(module: "tir.Module",
                    entry_fn: str = "main") -> bytes:
    """Like `compile_module_to_elf`, but `pytest.skip(...)` when
    the active backend is LLVM and the toolchain is unavailable
    (instead of raising `LLVMToolchainAbsent`). Use this in test
    bodies that want LLVM coverage when available but should not
    fail on a no-toolchain dev machine.

    The x86_64 backend has no skip path: `pytest.skip` only fires
    on the LLVM-DEFERRED case.
    """
    backend = selected_backend()
    if backend == "x86":
        from helixc.backend.x86_64 import (
            compile_module_to_elf as _x86_compile)
        return _x86_compile(module, entry_fn)
    if backend == "llvm":
        import pytest
        from helixc.backend.llvm_toolchain import (
            compile_module_to_elf_via_llvm as _llvm_compile)
        from helixc.backend.llvm_toolchain import LLVMToolchainAbsent
        try:
            return _llvm_compile(module, entry_fn)
        except LLVMToolchainAbsent as e:
            pytest.skip(f"LLVM toolchain unavailable: {e}")
    raise ValueError(
        f"unreachable: selected_backend() returned {backend!r}")
