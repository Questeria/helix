"""
helixc/ir/mlir/parity.py — Stage 215 MLIR-vs-tile-IR parity gate.

Stage 215 verifies the MLIR path matches the home-grown tile-IR path
across the five backend targets (LLVM_IR, PTX, ROCM_HIP, METAL_MSL,
WEBGPU_WGSL). Both paths start from the same `tile_ir.TileModule` —
the home-grown path lowers it through `helixc/backend/*.py` directly;
the MLIR path runs `emit_mlir_module` to produce MLIR text, then
`lower_mlir_to_backend` to produce the target artifact (via the
Stage 213 / 214 chain).

PARITY is the cross-path consistency check: for a given module and
target, the two paths must produce equivalent results (or both fail,
or both defer for the same reason).

Stage 215 chunk A — the harness skeleton. Defines the result type
`ParityResult` and the entry point `mlir_vs_tile_ir_parity_check`,
which:

- runs `emit_mlir_module` (the home-grown→MLIR translator) and
  catches `MLIRTranslationError` for unsupported constructs;
- runs `lower_mlir_to_backend` on the resulting MLIR text;
- returns a `ParityResult` describing the verdict.

Real cross-path equivalence checking (byte-equality of the target
artifact, semantic equivalence via re-execution, etc.) is the
chunk-B+ concern: a clean harness skeleton with documented hooks
lets the subsequent chunks fill in target-specific equivalence
checks without restructuring the API.

MOCK-PATH-FIRST: on a machine without MLIR / mlir-translate / etc.,
the MLIR side returns DEFERRED; the parity verdict is then
`PARITY_DEFERRED` ("the MLIR path is unverifiable on this machine —
cannot compare"). Fail-closed: any unexpected error in the MLIR
path bubbles into a `PARITY_FAILED` verdict with the named cause.

License: Apache 2.0
"""
from __future__ import annotations

import hashlib
import re
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .backends import (
    MLIRBackendResult,
    MLIRBackendStatus,
    MLIRBackendTarget,
    lower_mlir_to_backend,
)
from .emit import MLIRTranslationError, emit_mlir_module
from .toolchain import MLIRSupport
from .. import tile_ir


# Stage 215 chunk B — registry of home-grown path emitters per
# backend target. Each entry is a factory that builds a fresh
# emitter instance per call (the PTX / HIP / MSL / WGSL emitters
# carry per-module mutable state, so they cannot be cached). The
# `emit_module(TileModule) -> str` shape is the cross-backend
# `BackendEmitter` Protocol (helixc/backend/_lowering_schema.py).
#
# LLVM_IR is intentionally absent here: the home-grown LLVM path
# operates on `tir.Module`, not `tile_ir.TileModule`, so it cannot
# be invoked directly from the parity gate. Chunk C will document
# the LLVM_IR-specific parity contract (probably "MLIR side runs;
# home-grown side is structurally inaccessible from a TileModule;
# parity for LLVM_IR is verified by the Phase-D LLVM parity gate
# at Stage 207, not Stage 215").
_HOMEGROWN_EMITTERS: dict[MLIRBackendTarget,
                          Callable[[], object]] = {}


def _register_homegrown_emitters() -> None:
    """Lazy-import the per-backend emitter classes so this module
    imports cleanly even when one of the backends is unavailable.
    Builds the global `_HOMEGROWN_EMITTERS` registry; idempotent."""
    global _HOMEGROWN_EMITTERS
    if _HOMEGROWN_EMITTERS:
        return
    from ...backend.ptx import PtxEmitter
    from ...backend.rocm import HipEmitter
    from ...backend.metal import MslEmitter
    from ...backend.webgpu import WgslEmitter
    _HOMEGROWN_EMITTERS = {
        MLIRBackendTarget.PTX: PtxEmitter,
        MLIRBackendTarget.ROCM_HIP: HipEmitter,
        MLIRBackendTarget.METAL_MSL: MslEmitter,
        MLIRBackendTarget.WEBGPU_WGSL: WgslEmitter,
    }


def _run_tile_ir_path(
        module: tile_ir.TileModule,
        target: MLIRBackendTarget) -> tuple[Optional[str], tuple[str, ...]]:
    """Run the home-grown path: instantiate the per-target emitter
    and call `emit_module(module)`. Returns `(artifact_text, findings)`:
    on success `(text, ())`; on failure `(None, (finding,...))`.

    LLVM_IR returns a structurally-deferred finding because the
    home-grown LLVM path takes `tir.Module`, not `tile_ir.TileModule`,
    so it cannot be invoked from the parity gate's TileModule entry."""
    if target is MLIRBackendTarget.LLVM_IR:
        return None, (
            "home-grown LLVM_IR path operates on tir.Module, not "
            "tile_ir.TileModule; parity for LLVM_IR is handled by the "
            "Phase-D parity gate (Stage 207), not Stage 215",)
    _register_homegrown_emitters()
    factory = _HOMEGROWN_EMITTERS.get(target)
    if factory is None:
        return None, (
            f"home-grown path for {target.value} is not registered in "
            "_HOMEGROWN_EMITTERS — chunk-B+ must add the emitter "
            "factory",)
    try:
        emitter = factory()
        text = emitter.emit_module(module)  # type: ignore[attr-defined]
    except MemoryError:
        # Resource exhaustion is a host problem the caller must see,
        # not a parity-gate failure. Re-raise.
        raise
    except Exception as exc:
        # Stage 215 chunk D audit-fix MEDIUM-1: capture the full
        # traceback alongside the type+message so a debugger six
        # months from now can locate the offending op.
        tb_text = "".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__))
        return None, (
            f"home-grown path for {target.value} raised "
            f"{type(exc).__name__}: {exc}",
            f"traceback:\n{tb_text}",
        )
    if not isinstance(text, str) or not text.strip():
        return None, (
            f"home-grown path for {target.value} returned blank or "
            f"non-str output ({type(text).__name__})",)
    return text, ()


# Stage 215 chunk C — per-target line-comment markers. The
# normalization pass strips trailing line-comment text using this
# marker before whitespace folding so compiler-version-dependent
# inline annotations don't cause spurious parity-failure findings.
_TARGET_LINE_COMMENT_MARKER: dict[MLIRBackendTarget, str] = {
    MLIRBackendTarget.PTX: "//",        # PTX uses // (and /* */).
    MLIRBackendTarget.ROCM_HIP: "//",   # ROCm HIP (C++).
    MLIRBackendTarget.METAL_MSL: "//",  # MSL (C++-derived).
    MLIRBackendTarget.WEBGPU_WGSL: "//",
    MLIRBackendTarget.LLVM_IR: ";",     # LLVM IR uses ; for comments.
}


# Stage 215 chunk C — line-prefix markers that indicate a directive
# whose textual content can vary across compiler versions / build
# configurations but does not affect semantic equivalence. The
# normalization pass drops these lines outright. These are
# conservative — only the most clearly-cosmetic directives.
_TARGET_COSMETIC_LINE_PREFIXES: dict[MLIRBackendTarget,
                                     tuple[str, ...]] = {
    MLIRBackendTarget.PTX: (
        ".version",  # PTX assembler version differs per ptxas build.
        ".target",   # target sm version may be set per-deployment.
        ".address_size",
    ),
    MLIRBackendTarget.LLVM_IR: (
        "target datalayout",
        "target triple",
        "source_filename",
        "; ModuleID",
    ),
    MLIRBackendTarget.ROCM_HIP: (
        "#include",  # HIP / Metal / WGSL include-style headers.
    ),
    MLIRBackendTarget.METAL_MSL: (
        "#include",
        "using namespace",
    ),
    MLIRBackendTarget.WEBGPU_WGSL: (),
}


def _normalize_artifact_text(
        target: MLIRBackendTarget, text: str) -> str:
    """Normalize a target artifact for cross-path equivalence
    comparison. Strips line comments per target, drops cosmetic
    directive lines whose content varies across compiler builds,
    collapses whitespace runs, and discards blank lines.

    Conservative by design: only the rules listed in
    `_TARGET_LINE_COMMENT_MARKER` and
    `_TARGET_COSMETIC_LINE_PREFIXES` apply. More aggressive
    normalization (symbol reordering, dead-code elimination) belongs
    to Stage 216 / Phase F when concrete real-toolchain diffs
    inform the rules."""
    if not isinstance(text, str):
        raise ValueError(
            "_normalize_artifact_text: text must be str, got "
            f"{type(text).__name__}")
    # Stage 215 chunk D audit-fix MEDIUM-3: strip C-style block
    # comments before line iteration. The PTX home-grown emitter
    # does not emit `/* ... */` today, but `mlir-translate` /
    # `mlir-opt` may (e.g. via attribute serialization), and a
    # future LLVM upgrade could surface them only on the MLIR side.
    if target in (MLIRBackendTarget.PTX, MLIRBackendTarget.ROCM_HIP,
                  MLIRBackendTarget.METAL_MSL,
                  MLIRBackendTarget.WEBGPU_WGSL):
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    comment_marker = _TARGET_LINE_COMMENT_MARKER.get(target)
    cosmetic_prefixes = _TARGET_COSMETIC_LINE_PREFIXES.get(target, ())
    out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if comment_marker is not None:
            idx = line.find(comment_marker)
            if idx == 0:
                continue
            if idx > 0:
                line = line[:idx].rstrip()
                if not line:
                    continue
        if cosmetic_prefixes and any(
                line.startswith(prefix) for prefix in cosmetic_prefixes):
            continue
        line = " ".join(line.split())
        out.append(line)
    return "\n".join(out)


def _compare_artifacts(
        target: MLIRBackendTarget,
        tile_ir_text: str, mlir_text: str) -> tuple[bool, Optional[str]]:
    """Compare two artifacts after target-specific normalization.
    Returns `(match, summary)`: when the normalized SHA-256 digests
    match, returns `(True, None)`; otherwise `(False, summary)` with
    a short diff summary suitable for a `ParityResult.findings`
    entry.

    The summary names byte counts of both normalized forms and the
    first divergent line, so a downstream debugger can locate the
    mismatch without needing to recompute the diff."""
    if not isinstance(target, MLIRBackendTarget):
        raise ValueError(
            "_compare_artifacts: target must be MLIRBackendTarget")
    if not isinstance(tile_ir_text, str) \
            or not isinstance(mlir_text, str):
        raise ValueError(
            "_compare_artifacts: both texts must be str")
    tile_ir_norm = _normalize_artifact_text(target, tile_ir_text)
    mlir_norm = _normalize_artifact_text(target, mlir_text)
    # Stage 215 chunk D audit-fix HIGH-1: refuse to mint a parity
    # match on empty normalized forms — two artifacts that consist
    # entirely of cosmetic directives / comments would collide on
    # SHA-256 of "" and silently PARITY_HOLDS otherwise.
    if not tile_ir_norm or not mlir_norm:
        which = ("both" if (not tile_ir_norm and not mlir_norm)
                 else "tile-IR" if not tile_ir_norm else "MLIR")
        return False, (
            f"parity check refused: normalized {which} artifact is "
            "empty — likely an emitter producing only cosmetic "
            "directives or a normalization rule that ate all "
            "semantic content")
    tile_ir_digest = hashlib.sha256(
        tile_ir_norm.encode("utf-8")).hexdigest()
    mlir_digest = hashlib.sha256(mlir_norm.encode("utf-8")).hexdigest()
    if tile_ir_digest == mlir_digest:
        return True, None
    tile_ir_lines = tile_ir_norm.splitlines()
    mlir_lines = mlir_norm.splitlines()
    divergence_line: Optional[int] = None
    sample_tile_ir = ""
    sample_mlir = ""
    for index in range(max(len(tile_ir_lines), len(mlir_lines))):
        tile_ir_line = tile_ir_lines[index] if index < len(tile_ir_lines) \
            else "<eof>"
        mlir_line = mlir_lines[index] if index < len(mlir_lines) \
            else "<eof>"
        if tile_ir_line != mlir_line:
            divergence_line = index + 1
            sample_tile_ir = tile_ir_line[:80]
            sample_mlir = mlir_line[:80]
            break
    summary = (
        f"artifacts differ after normalization "
        f"(tile-IR side: {len(tile_ir_norm)} bytes, "
        f"{len(tile_ir_lines)} lines; "
        f"MLIR side: {len(mlir_norm)} bytes, "
        f"{len(mlir_lines)} lines)")
    if divergence_line is not None:
        summary += (
            f"; first divergence at line {divergence_line}: "
            f"tile-IR={sample_tile_ir!r}, MLIR={sample_mlir!r}")
    return False, summary


class ParityStatus(Enum):
    """Tri-state verdict for an MLIR-vs-tile-IR parity check.

    - PARITY_HOLDS: both paths produce equivalent results.
    - PARITY_FAILED: the two paths diverge (or the MLIR path raises
      an error the home-grown path would not).
    - PARITY_DEFERRED: parity cannot be verified on this machine —
      typically the MLIR path returned DEFERRED because a real
      toolchain is absent.
    """
    PARITY_HOLDS = "parity_holds"
    PARITY_FAILED = "parity_failed"
    PARITY_DEFERRED = "parity_deferred"


@dataclass(frozen=True, slots=True)
class ParityResult:
    """The result of an MLIR-vs-tile-IR parity check for one module
    and one backend target.

    Frozen + `__post_init__`-guarded — the house discipline of the
    Stage-213 result types (see `MLIRBackendResult`):
    - PARITY_FAILED or PARITY_DEFERRED MUST carry at least one
      finding explaining why;
    - PARITY_HOLDS MUST carry no findings;
    - the `mlir_result` attribute (when present) is the
      `MLIRBackendResult` from the MLIR path; the `tile_ir_output`
      attribute (when present) is the home-grown path's artifact
      text, so a caller can inspect both without re-running either.
    """
    target: MLIRBackendTarget
    status: ParityStatus
    findings: tuple[str, ...]
    mlir_result: Optional[MLIRBackendResult] = None
    tile_ir_output: Optional[str] = None

    def __init_subclass__(cls, **kwargs) -> None:
        raise TypeError(
            "ParityResult is final; subclassing could bypass parity-"
            "result invariants")

    def __post_init__(self) -> None:
        if not isinstance(self.target, MLIRBackendTarget):
            raise ValueError(
                "ParityResult: target must be MLIRBackendTarget, got "
                f"{self.target!r}")
        if not isinstance(self.status, ParityStatus):
            raise ValueError(
                "ParityResult: status must be ParityStatus, got "
                f"{self.status!r}")
        if not isinstance(self.findings, tuple):
            raise ValueError(
                "ParityResult: findings must be a tuple, got "
                f"{type(self.findings).__name__}")
        for entry in self.findings:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    f"ParityResult: findings has a blank or non-str "
                    f"entry ({entry!r}) — every finding carries text")
        if self.status in (ParityStatus.PARITY_FAILED,
                           ParityStatus.PARITY_DEFERRED) \
                and not self.findings:
            raise ValueError(
                f"ParityResult: a {self.status.name} result must "
                "carry at least one finding explaining why")
        if self.status is ParityStatus.PARITY_HOLDS and self.findings:
            raise ValueError(
                f"ParityResult: a PARITY_HOLDS result must carry NO "
                f"findings ({len(self.findings)} given) — a clean "
                "parity verdict has nothing to report")
        if self.mlir_result is not None \
                and type(self.mlir_result) is not MLIRBackendResult:
            raise ValueError(
                "ParityResult: mlir_result must be an MLIRBackendResult "
                "or None")
        if self.tile_ir_output is not None:
            if not isinstance(self.tile_ir_output, str) \
                    or not self.tile_ir_output.strip():
                raise ValueError(
                    "ParityResult: tile_ir_output must be non-blank "
                    "text or None")

    def holds(self) -> bool:
        return self.status is ParityStatus.PARITY_HOLDS

    def failed(self) -> bool:
        return self.status is ParityStatus.PARITY_FAILED

    def deferred(self) -> bool:
        return self.status is ParityStatus.PARITY_DEFERRED

    def is_positive_assertion(self) -> bool:
        """True iff this is a CHECKED parity claim — the parity gate
        ran both paths and the artifacts matched. PARITY_DEFERRED is
        explicitly NOT a positive assertion: it means "the parity
        gate could not verify on this machine", not "the artifacts
        match." Callers gating release / promotion decisions on the
        parity check MUST use `is_positive_assertion()`; using
        `not failed()` would silently ship LLVM_IR (which always
        defers — home-grown LLVM is structurally inaccessible from
        a TileModule) and any unverifiable target."""
        return self.status is ParityStatus.PARITY_HOLDS


def mlir_vs_tile_ir_parity_check(
        module: tile_ir.TileModule,
        target: MLIRBackendTarget,
        *,
        support: Optional[MLIRSupport] = None) -> ParityResult:
    """Run both paths of the parity gate for one Tile-IR module and
    one backend target.

    Stage 215 chunks A+B: this entry point runs BOTH the MLIR path
    (`emit_mlir_module` -> `lower_mlir_to_backend`) and the
    home-grown path (`_run_tile_ir_path` -> per-target emitter), and
    reports a `ParityResult` describing whether parity COULD be
    verified and (when both paths produced text) whether the
    artifacts match:

    - Home-grown raises -> PARITY_FAILED with the named cause.
    - MLIR translator raises -> PARITY_FAILED with the translator finding.
    - MLIR backend chain FAILED -> PARITY_FAILED with the backend finding.
    - MLIR backend chain DEFERRED (no toolchain) -> PARITY_DEFERRED
      with the deferral reason AND the home-grown output recorded
      for inspection.
    - MLIR backend chain PASSED + home-grown text present ->
      PARITY_HOLDS placeholder (chunk-C+ adds the actual artifact
      comparison; chunk B currently treats "both paths produced
      text" as parity-holding).

    For LLVM_IR specifically, the home-grown path cannot run from a
    TileModule (it consumes tir.Module instead); chunk B returns a
    structurally-deferred finding for LLVM_IR with a pointer to
    Stage 207's Phase-D parity gate.
    """
    if not isinstance(module, tile_ir.TileModule):
        raise ValueError(
            "mlir_vs_tile_ir_parity_check: module must be a "
            f"tile_ir.TileModule, got {type(module).__name__}")
    if not isinstance(target, MLIRBackendTarget):
        raise ValueError(
            "mlir_vs_tile_ir_parity_check: target must be "
            f"MLIRBackendTarget, got {target!r}")

    # Home-grown side first — if it raises, the parity gate fails
    # without bothering the MLIR side (the home-grown path is the
    # canonical compiler; an exception there is more severe than an
    # MLIR-side deferral).
    tile_ir_output, tile_ir_findings = _run_tile_ir_path(module, target)
    if tile_ir_findings and target is not MLIRBackendTarget.LLVM_IR:
        # Home-grown path failed; surface as PARITY_FAILED.
        return ParityResult(
            target=target,
            status=ParityStatus.PARITY_FAILED,
            findings=(
                f"home-grown {target.value} path failed: "
                + tile_ir_findings[0],
            ),
            mlir_result=None,
            tile_ir_output=None,
        )

    try:
        mlir_text = emit_mlir_module(module)
    except MLIRTranslationError as exc:
        return ParityResult(
            target=target,
            status=ParityStatus.PARITY_FAILED,
            findings=(
                f"MLIR translator failed for {target.value}: "
                f"{type(exc).__name__}: {exc}",),
            mlir_result=None,
            tile_ir_output=tile_ir_output,
        )

    mlir_result = lower_mlir_to_backend(
        mlir_text, target, support=support)
    backend_status = mlir_result.status()

    # LLVM_IR special case: home-grown is structurally inaccessible
    # from a TileModule, so the parity verdict is DEFERRED with both
    # the home-grown deferral note AND the MLIR side's status. The
    # status is prefixed `mlir_side_status=` so a machine-readable
    # caller can branch on it without parsing prose; callers must
    # still treat PARITY_DEFERRED as "unverifiable", not "ship-ok".
    if target is MLIRBackendTarget.LLVM_IR:
        mlir_summary = ("passed" if backend_status is MLIRBackendStatus.PASSED
                        else "failed" if backend_status
                        is MLIRBackendStatus.FAILED
                        else "deferred")
        return ParityResult(
            target=target,
            status=ParityStatus.PARITY_DEFERRED,
            findings=(
                tile_ir_findings[0],
                f"mlir_side_status={mlir_summary}; the Stage 207 "
                "Phase-D parity gate is the canonical LLVM_IR parity "
                "check (treat PARITY_DEFERRED as unverifiable, never "
                "as a positive assertion)",
            ),
            mlir_result=mlir_result,
            tile_ir_output=None,
        )

    if backend_status is MLIRBackendStatus.PASSED:
        # Chunk C — actual cross-path artifact comparison. Both
        # paths produced text; normalize each per target and compare
        # SHA-256 digests. PARITY_HOLDS only when they agree.
        if tile_ir_output is None or mlir_result.output_text is None:
            return ParityResult(
                target=target,
                status=ParityStatus.PARITY_FAILED,
                findings=(
                    f"parity check for {target.value} cannot compare: "
                    "one side produced no artifact text "
                    f"(tile_ir_output={'present' if tile_ir_output else 'None'}"
                    f", mlir_output_text="
                    f"{'present' if mlir_result.output_text else 'None'})",
                ),
                mlir_result=mlir_result,
                tile_ir_output=tile_ir_output,
            )
        match, summary = _compare_artifacts(
            target, tile_ir_output, mlir_result.output_text)
        if match:
            return ParityResult(
                target=target,
                status=ParityStatus.PARITY_HOLDS,
                findings=(),
                mlir_result=mlir_result,
                tile_ir_output=tile_ir_output,
            )
        return ParityResult(
            target=target,
            status=ParityStatus.PARITY_FAILED,
            findings=(
                f"parity check for {target.value} found a normalized "
                "artifact mismatch: "
                + (summary or "no diff summary emitted"),
            ),
            mlir_result=mlir_result,
            tile_ir_output=tile_ir_output,
        )
    if backend_status is MLIRBackendStatus.FAILED:
        return ParityResult(
            target=target,
            status=ParityStatus.PARITY_FAILED,
            findings=(
                f"MLIR backend chain for {target.value} failed: "
                + (mlir_result.lowering_findings[0]
                   if mlir_result.lowering_findings
                   else "no lowering finding emitted"),
            ),
            mlir_result=mlir_result,
            tile_ir_output=tile_ir_output,
        )
    # DEFERRED — usually no real toolchain on this machine.
    return ParityResult(
        target=target,
        status=ParityStatus.PARITY_DEFERRED,
        findings=(
            f"MLIR backend chain for {target.value} deferred — parity "
            "cannot be verified without a real toolchain: "
            + (mlir_result.lowering_findings[0]
               if mlir_result.lowering_findings
               else "no deferral reason emitted"),
        ),
        mlir_result=mlir_result,
        tile_ir_output=tile_ir_output,
    )
