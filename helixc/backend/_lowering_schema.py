"""
helixc/backend/_lowering_schema.py — shared type-design for backend
lowering tables.

v2.3 polish item 2 (from v2.1 + v2.2 5-clean-gate type-design audits).
All four backends (ptx, rocm, metal, webgpu) share an unwritten
contract: each exports `XXX_OP_LOWERING: dict[TileOpKind, dict]` with
inner schema `{"lowering": str, "status": ...}`, plus
`lowering_status(kind) -> str`. The cross-backend type-design
auditors flagged repeatedly:

1. The inner `dict` type erases the `lowering` / `status` keys at
   the type-system level — `mypy` sees `Mapping[TileOpKind, dict]`
   and gives up. The status-domain invariant lives in test files
   (e.g., test_metal.py:43-50) instead of in the type.

2. The `status` field is a free-form `str` — typos like `"Supported"`
   (capital S) only fail at test time, not at construction.

3. The four `XxxEmitter` classes share a near-identical surface
   (`emit_module(mod) -> str` + `lowering_status(kind) -> str`) but
   no `Protocol` lets downstream code be generic over target.

This module fixes all three:
  - `LoweringStatus`: `Literal["supported", "stub", "deferred", "skipped"]`
  - `OpLowering`: `TypedDict` with `lowering: str` + `status: LoweringStatus`
  - `BackendEmitter`: `Protocol` for the cross-backend emitter surface
  - `HELIX_STUB_TOKEN`: shared sentinel for stub-status emit markers

Per-backend tables retain their `Final[Mapping[TileOpKind, dict]]`
signatures for backwards compatibility (changing them to
`Mapping[TileOpKind, OpLowering]` is the v2.3 polish wave's wrapping
work; this module is the prerequisite).

License: Apache 2.0
"""
from __future__ import annotations

from typing import Final, Literal, Protocol, TypedDict, runtime_checkable

from ..ir import tile_ir as ti


# ============================================================================
# Status taxonomy
# ============================================================================
# Closed-set status values. Each per-backend table assigns one of these
# to every TileOpKind. Semantics:
#   "supported": the emitter has a concrete `_emit_op` branch producing
#                real target-language text (operand-binding may still
#                be HELIX-STUB-OPERANDS placeholder for the substrate).
#   "stub":      placeholder declared but no `_emit_op` branch yet;
#                forward guard at the top of `_emit_op` emits a loud
#                target-language directive (`.error` / `#error` /
#                `@@HELIX-STUB`) that aborts downstream assembly /
#                compilation.
#   "deferred":  blocked on a future stage (treated identically to
#                "stub" by the forward guard — same loud-fail emit).
#   "skipped":   no analog on this backend (e.g., TMA_LOAD on Apple).
#                Documented for completeness; forward guard emits
#                "HELIX-SKIPPED" directive on use.
LoweringStatus = Literal["supported", "stub", "deferred", "skipped"]

# Frozenset for runtime membership checks (per-backend tables can
# validate their inner dicts against this at module load via
# `entry["status"] in VALID_STATUSES`).
VALID_STATUSES: Final[frozenset[str]] = frozenset(
    {"supported", "stub", "deferred", "skipped"}
)


# ============================================================================
# Inner-dict schema
# ============================================================================
class OpLowering(TypedDict):
    """Schema for one entry in a per-backend `XXX_OP_LOWERING` table.

    Required keys:
      `lowering`: human-readable description of the emit pattern
                  (target-language flavor — e.g., "ld.global.b128" for
                  PTX, "device float*" for MSL).
      `status`:   one of LoweringStatus.

    Per-backend tables can still use `dict` for backwards-compat at
    v2.3.0; the TypedDict can be opted into per-backend in v2.4+ once
    the API surface stabilizes. Downstream code that wants strict
    typing reads through this TypedDict via `cast(OpLowering, entry)`.
    """
    lowering: str
    status: LoweringStatus


# ============================================================================
# Stub-token sentinel
# ============================================================================
# Each backend emits a loud-fail directive in its target language when
# a kind has status "stub" / "deferred" (HELIX-STUB) or "skipped" —
# no analog on this target — (HELIX-SKIPPED). The directives are:
#   PTX:    `.error "HELIX-STUB: ..."`  / `.error "HELIX-SKIPPED: ..."`
#   ROCm:   `.error "HELIX-STUB: ..."`  / `.error "HELIX-SKIPPED: ..."`
#   Metal:  `#error "HELIX-STUB: ..."`  / `#error "HELIX-SKIPPED: ..."`
#   WebGPU: `@@HELIX-STUB: ...`         / `@@HELIX-SKIPPED: ...`
#           (parse-breaking sigil)
# `HELIX_STUB_TOKEN` ("HELIX-STUB") is the substring every backend
# emits for the stub/deferred case; the "skipped" case uses the
# distinct "HELIX-SKIPPED" string (NOT a superstring of HELIX-STUB).
# Downstream tooling that wants to detect "this kernel is non-
# functional" must therefore check BOTH HELIX_STUB_TOKEN and the
# "HELIX-SKIPPED" literal (see gpu_ci.validate_emit).
HELIX_STUB_TOKEN: Final[str] = "HELIX-STUB"

# Operand-binding placeholder marker. When status="supported" but
# operand binding is not yet wired (real RegAlloc is v2.3 item 15),
# emit branches use this comment marker so the deferral is visible
# in the emitted source rather than hidden behind a docstring.
HELIX_STUB_OPERANDS_TOKEN: Final[str] = "HELIX-STUB-OPERANDS"


# ============================================================================
# Cross-backend Emitter Protocol
# ============================================================================
@runtime_checkable
class BackendEmitter(Protocol):
    """Structural type for all 4 backends' Emitter classes.

    Downstream pipeline code (CLI driver, autotuner, multi-target
    dispatchers) can be generic over backend via this Protocol. Each
    concrete emitter (PtxEmitter, HipEmitter, MslEmitter, WgslEmitter)
    satisfies this implicitly by exposing the same `emit_module`
    method — no `class Emitter(BackendEmitter)` declaration needed.

    The Protocol is `runtime_checkable` so `isinstance(em,
    BackendEmitter)` works at runtime (useful for plugin registration
    + factory dispatch).
    """
    def emit_module(self, mod: ti.TileModule) -> str: ...


# ============================================================================
# Helpers
# ============================================================================
def is_loud_stub_status(status: str) -> bool:
    """True iff the status should trigger a loud-stub forward guard
    at the top of `_emit_op` (i.e., emit a HELIX-STUB / HELIX-SKIPPED
    directive instead of silently emitting nothing).

    Maps:
      "stub", "deferred", "skipped"  → True (emit directive)
      "supported"                    → False (real codegen branch)
    """
    return status in ("stub", "deferred", "skipped")
