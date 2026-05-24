"""
helixc/ir/mlir/toolchain.py — MLIR capability detection
(v3.0 Phase E, Stage 211).

The first substrate of the MLIR migration: a capability probe that
reports whether — and via which surface — this machine can run real
MLIR work. Phase E is MOCK-PATH-FIRST (Stage 210 decision, section 3):
when no real MLIR surface is available the harness DEFERS, never
FAILS, so CI on a binding-less runner stays green and the home-grown
tile-IR path stays the reversible fallback until the Stage 221
cutover.

There are TWO independent real surfaces (Stage 210 decision, section
3.2):
- the in-process MLIR Python bindings (`import mlir`), and
- the `mlir-opt` command-line tool.

`detect_mlir_support()` probes each independently and returns a frozen
`MLIRSupport`. It realizes the decision record's `detect_mlir_python()`
together with the `mlir-opt` CLI detection in one result.

The MLIR import is LAZY — done with `importlib` inside the probe,
never at module top level — so importing THIS module never fails for
lack of the bindings. That is the hard rule of the Stage 210 decision
(section 3.2): a top-level `import mlir` anywhere in `helixc/` would
make the whole compiler unimportable on a binding-less machine.

Mirrors `helixc/backend/llvm_toolchain.py` (the Phase-D Stage 201
LLVM-toolchain probe) and the `gpu_ci.py` real-HW dispatch discipline.

License: Apache 2.0
"""

from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass
from typing import Optional


# The MLIR dialect sub-modules Phase E's hybrid strategy depends on —
# the upstream dialects the ~80-85% numerical / structural op core
# maps onto (docs/V3_STAGE210_MLIR_DECISION.md, section 2). The probe
# imports each INDEPENDENTLY: a partial install — the core `mlir`
# package present but a dialect sub-module absent — must degrade to
# "not usable", never pass the probe and then fail deep inside a
# dialect call (the Stage-210 architecture-review correction).
# `nvgpu` is deliberately NOT required here: it is NVIDIA-only and the
# decision record flags it as possibly replaced by a `helix` async op
# — its availability is a Stage-213 (GPU lowering) concern, not a
# Stage-211 capability requirement.
_REQUIRED_MLIR_DIALECTS: tuple[str, ...] = (
    "func", "arith", "math", "cf", "scf",
    "tensor", "memref", "linalg", "vector", "gpu",
)


def _check_mlir_dialects() -> None:
    """Module-load guard: `_REQUIRED_MLIR_DIALECTS` is a non-empty
    tuple of unique, identifier-shaped dialect names. A typo, a
    duplicate, or a blank entry would silently skew what
    `can_use_bindings()` means — a mistyped name would mark a perfectly
    good full install as a partial one (a false DEFERRED), the exact
    silent-degradation class this guard exists to prevent. Mirrors
    `gpu_ci._check_gpu_ci_drift` / `llvm_toolchain._check_llvm_
    toolchain_drift`."""
    if not _REQUIRED_MLIR_DIALECTS:
        raise AssertionError(
            "helixc.ir.mlir.toolchain: _REQUIRED_MLIR_DIALECTS is "
            "empty — the bindings probe would then vacuously accept "
            "any install as fully dialect-capable")
    seen: set[str] = set()
    for dia in _REQUIRED_MLIR_DIALECTS:
        if not isinstance(dia, str) or not dia.isidentifier():
            raise AssertionError(
                f"helixc.ir.mlir.toolchain: _REQUIRED_MLIR_DIALECTS "
                f"has a blank / non-identifier entry ({dia!r})")
        if dia in seen:
            raise AssertionError(
                f"helixc.ir.mlir.toolchain: _REQUIRED_MLIR_DIALECTS "
                f"has a duplicate entry {dia!r}")
        seen.add(dia)


_check_mlir_dialects()


@dataclass(frozen=True)
class MLIRSupport:
    """Whether this machine can run real MLIR work, and via which
    surface — the result of `detect_mlir_support()`.

    Three independent real surfaces (Stage 210 decision, section 3.2,
    expanded for Stage 214's translate-step plumbing):

    - The in-process MLIR Python bindings. `bindings` is True when core
      `mlir.ir` imports; `dialects` is True when, additionally, every
      required dialect sub-module imports. A partial install (core
      present, a dialect absent) is `bindings=True, dialects=False` —
      which `can_use_bindings()` treats as NOT usable.
    - The `mlir-opt` command-line tool: `mlir_opt` is its resolved
      PATH location, or None when absent.
    - The `mlir-translate` command-line tool: `mlir_translate` is its
      resolved PATH location, or None when absent. Stage 214 chains
      this after `mlir-opt` to convert MLIR dialect output into the
      raw target artifact downstream consumers read (LLVM IR / SPIR-V
      binary / etc.).

    Frozen + `__post_init__`-guarded — the structural sibling of
    `llvm_parity.RealExecSupport` (the Phase-D WSL/clang
    capability-detection result), built to the same house discipline
    as `gpu_ci.ValidationResult`:
    - `dialects` True implies `bindings` True — a dialect sub-module
      cannot import without the core package;
    - `detail` is always non-empty and every entry carries text, so a
      DEFERRED (no real surface) result is never silent about why.

    `bindings` / `dialects` record that the modules IMPORTED — a
    capability probe, not a deep validation; whether `mlir.ir`'s API
    is functionally sound is re-checked at the real-dispatch stage
    (Stage 212+), the same way `llvm_toolchain` defers real
    well-formedness to `llvm-as`.
    """
    bindings: bool
    dialects: bool
    mlir_opt: Optional[str]
    detail: tuple[str, ...]
    mlir_translate: Optional[str] = None

    def __post_init__(self) -> None:
        # A dialect sub-module cannot import without the core package.
        if self.dialects and not self.bindings:
            raise ValueError(
                "MLIRSupport: dialects=True but bindings=False — a "
                "dialect sub-module cannot import without core `mlir`")
        # The result must always explain what it found — a tool-less
        # machine especially must say WHY, so a DEFERRED is never
        # silent.
        if not self.detail:
            raise ValueError(
                "MLIRSupport: detail is empty — the result must "
                "explain what MLIR support is and is not available")
        for entry in self.detail:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    f"MLIRSupport: detail has a blank or non-str entry "
                    f"({entry!r}) — every line must carry text")
        if self.mlir_translate is not None:
            if not isinstance(self.mlir_translate, str) \
                    or not self.mlir_translate.strip() \
                    or self.mlir_translate != self.mlir_translate.strip():
                raise ValueError(
                    "MLIRSupport: mlir_translate must be a non-blank, "
                    f"whitespace-stripped path or None, got "
                    f"{self.mlir_translate!r}")

    def can_use_bindings(self) -> bool:
        """True iff the in-process MLIR Python bindings are FULLY
        usable — core `mlir` plus every required dialect sub-module
        imports. A partial install (`bindings` True but `dialects`
        False) is not usable."""
        return self.bindings and self.dialects

    def can_use_mlir_opt(self) -> bool:
        """True iff the `mlir-opt` command-line tool is on PATH."""
        return self.mlir_opt is not None

    def can_use_mlir_translate(self) -> bool:
        """True iff the `mlir-translate` CLI is on PATH. Stage 214
        chains this after `mlir-opt` to emit the raw target artifact
        downstream consumers read; without it, lowering stays
        DEFERRED for any target whose translator entry is wired."""
        return self.mlir_translate is not None

    def is_available(self) -> bool:
        """True iff at least one real MLIR surface — the in-process
        bindings or the `mlir-opt` CLI — is usable. When False, Phase E
        runs mock-path-only and every real-MLIR step is DEFERRED, never
        FAILED."""
        return self.can_use_bindings() or self.can_use_mlir_opt()


def _dialect_imports(dialect: str) -> bool:
    """True iff `mlir.dialects.<dialect>` imports cleanly.

    Any failure — an absent sub-module, or an installed-but-broken
    build that raises during import — is caught and reported as
    'absent': a probe that cannot answer fails SAFE to the mock path,
    never as an uncaught traceback."""
    try:
        importlib.import_module(f"mlir.dialects.{dialect}")
        return True
    except Exception:
        return False


def detect_mlir_support() -> MLIRSupport:
    """Probe this machine's MLIR capability and return a frozen
    `MLIRSupport`.

    The two real surfaces — the in-process Python bindings and the
    `mlir-opt` CLI — are probed INDEPENDENTLY: each absent surface
    degrades on its own, and a partial bindings install (core present,
    a required dialect sub-module absent) is reported as bindings-not-
    usable rather than passing and failing later.

    The MLIR import is LAZY (via `importlib`, here inside the probe) —
    never at module top level — so a binding-less machine yields a
    result with `bindings=False` rather than an `ImportError` at
    `helixc` import time. Realizes the Stage 210 decision's
    `detect_mlir_python()` together with the `mlir-opt` detection.

    Never raises for an absent/broken toolchain: every probe failure is
    captured into the returned `MLIRSupport`."""
    detail: list[str] = []

    # --- surface 1: the in-process MLIR Python bindings ---
    bindings = False
    dialects = False
    try:
        importlib.import_module("mlir.ir")
        bindings = True
    except Exception as exc:
        # `ModuleNotFoundError` when the bindings are simply absent; a
        # deeper exception if an installed-but-broken build fails
        # during import. Either way the bindings surface is unusable —
        # captured here, never raised.
        detail.append(
            f"MLIR Python bindings absent or broken — `import mlir.ir` "
            f"failed ({type(exc).__name__}: {exc})")
    if bindings:
        missing = [dia for dia in _REQUIRED_MLIR_DIALECTS
                   if not _dialect_imports(dia)]
        dialects = not missing
        if missing:
            detail.append(
                f"MLIR core present but {len(missing)} required dialect "
                f"sub-module(s) absent ({', '.join(missing)}) — a "
                f"partial install; the bindings surface is not usable")
        else:
            detail.append(
                f"MLIR Python bindings present — core `mlir.ir` and all "
                f"{len(_REQUIRED_MLIR_DIALECTS)} required dialect "
                f"sub-modules import cleanly")

    # --- surface 2: the `mlir-opt` command-line tool ---
    mlir_opt = shutil.which("mlir-opt")
    if mlir_opt is None:
        detail.append("`mlir-opt` is not on PATH")
    else:
        detail.append(f"`mlir-opt` is on PATH at {mlir_opt!r}")

    # --- surface 3: the `mlir-translate` command-line tool ---
    # Stage 214 chains this after `mlir-opt` to convert dialect-MLIR
    # output into the raw target artifact (LLVM IR, SPIR-V, etc.).
    mlir_translate = shutil.which("mlir-translate")
    if mlir_translate is None:
        detail.append("`mlir-translate` is not on PATH")
    else:
        detail.append(f"`mlir-translate` is on PATH at {mlir_translate!r}")

    # Every branch above appended at least one `detail` line, and
    # `dialects` is never set without `bindings` — so this construction
    # always satisfies MLIRSupport.__post_init__ and cannot raise.
    return MLIRSupport(bindings=bindings, dialects=dialects,
                       mlir_opt=mlir_opt, detail=tuple(detail),
                       mlir_translate=mlir_translate)
