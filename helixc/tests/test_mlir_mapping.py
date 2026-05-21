"""Tests for helixc.ir.mlir.mapping — v3.0 Phase E, Stage 211 chunk B:
the Helix-op -> MLIR-lowering mapping.

`mapping.py` turns the ratified Stage 210 decision record's section-2.2
op-mapping table (docs/V3_STAGE210_MLIR_DECISION.md) into a code data
structure: for every Tensor-IR `tir.OpKind`, which MLIR lowering target
it belongs to — an upstream MLIR dialect, the custom `helix` dialect,
or RESIDUAL (a placement the decision record explicitly DEFERRED, "flag
for review").

These tests pin: the `MLIRLowering` enum and its upstream / HELIX /
RESIDUAL partition; that `_OPKIND_LOWERING` covers `tir.OpKind` EXACTLY
(the load-bearing drift guard — no op unmapped, no stale key); that the
two module-load guards are NOT vacuous (each genuinely catches the
corruption it exists to catch); the `mlir_lowering_for` accessor's
totality and the `is_upstream` predicate; the decision-record-anchored
classification of the Helix-specific and the deferred op sets; and —
the mock-path rule (Stage 210 decision, section 3.2) — that the module
is pure data: it never `import mlir`, at top level or anywhere.
"""
from __future__ import annotations

import ast
import enum
from pathlib import Path

import pytest

from helixc.ir import tir
from helixc.ir.mlir import mapping
from helixc.ir.mlir.mapping import (
    MLIRLowering, dialect_name, is_upstream, mlir_lowering_for,
)


# --------------------------------------------------------------------------
# MLIRLowering — the enum + the upstream / HELIX / RESIDUAL partition
# --------------------------------------------------------------------------
def test_mlir_lowering_members():
    """`MLIRLowering` names the eight upstream MLIR dialects plus HELIX
    and RESIDUAL — ten members, each carrying a distinct string value."""
    assert {m.name for m in MLIRLowering} == {
        "ARITH", "MATH", "LINALG", "TENSOR", "MEMREF", "FUNC", "CF",
        "GPU", "HELIX", "RESIDUAL"}
    values = [m.value for m in MLIRLowering]
    assert len(values) == len(set(values)), "values must be unique"
    assert all(isinstance(v, str) and v.strip() for v in values)


def test_upstream_lowerings_set():
    """`_UPSTREAM_LOWERINGS` is exactly the eight upstream MLIR dialect
    members — the custom `helix` dialect and the deferred RESIDUAL
    bucket are deliberately excluded (they are not upstream dialects)."""
    assert mapping._UPSTREAM_LOWERINGS == frozenset({
        MLIRLowering.ARITH, MLIRLowering.MATH, MLIRLowering.LINALG,
        MLIRLowering.TENSOR, MLIRLowering.MEMREF, MLIRLowering.FUNC,
        MLIRLowering.CF, MLIRLowering.GPU})
    assert MLIRLowering.HELIX not in mapping._UPSTREAM_LOWERINGS
    assert MLIRLowering.RESIDUAL not in mapping._UPSTREAM_LOWERINGS


def test_lowering_partition_guard_passes():
    """The module-load guard `_check_lowering_partition` is callable and
    passes for the current enum — upstream + {HELIX, RESIDUAL} is
    exactly `MLIRLowering`, so every member is classified."""
    mapping._check_lowering_partition()  # must not raise
    classified = (mapping._UPSTREAM_LOWERINGS
                  | {MLIRLowering.HELIX, MLIRLowering.RESIDUAL})
    assert classified == set(MLIRLowering)


def test_lowering_partition_guard_is_not_vacuous(monkeypatch):
    """`_check_lowering_partition` genuinely catches an unclassified
    member — a guard that always passed would not protect against a new
    `MLIRLowering` added without slotting it into the partition. Drop a
    member from `_UPSTREAM_LOWERINGS` and confirm it raises."""
    monkeypatch.setattr(
        mapping, "_UPSTREAM_LOWERINGS",
        mapping._UPSTREAM_LOWERINGS - {MLIRLowering.GPU})
    with pytest.raises(AssertionError, match="must be classified"):
        mapping._check_lowering_partition()


def test_lowering_partition_guard_catches_helix_in_upstream(monkeypatch):
    """`_check_lowering_partition`'s SECOND branch genuinely catches
    HELIX (or RESIDUAL) wrongly listed as an upstream dialect — they are
    the custom dialect and the deferred bucket, never upstream. Inject
    HELIX into `_UPSTREAM_LOWERINGS` (which keeps the first, coverage,
    branch passing) and confirm the disjointness branch still raises."""
    monkeypatch.setattr(
        mapping, "_UPSTREAM_LOWERINGS",
        mapping._UPSTREAM_LOWERINGS | {MLIRLowering.HELIX})
    with pytest.raises(AssertionError, match="must not be in"):
        mapping._check_lowering_partition()


# --------------------------------------------------------------------------
# _OPKIND_LOWERING — covers tir.OpKind EXACTLY (the drift guard)
# --------------------------------------------------------------------------
def test_opkind_lowering_covers_every_opkind():
    """`_OPKIND_LOWERING` maps EXACTLY the `tir.OpKind` enum — every op
    classified, no stale key. This is the load-bearing drift guard: an
    `OpKind` added without a mapping entry would otherwise be silently
    invisible to the Stage-212 MLIR translation."""
    assert set(mapping._OPKIND_LOWERING) == set(tir.OpKind)
    mapping._check_opkind_coverage()  # the guard itself must pass


def test_opkind_coverage_guard_catches_missing_op(monkeypatch):
    """`_check_opkind_coverage` genuinely catches an unmapped op — a
    guard that always passed would not detect drift. Drop an entry and
    confirm it raises, naming the now-unmapped op."""
    broken = dict(mapping._OPKIND_LOWERING)
    del broken[tir.OpKind.ADD]
    monkeypatch.setattr(mapping, "_OPKIND_LOWERING", broken)
    with pytest.raises(AssertionError, match=r"unmapped.*ADD"):
        mapping._check_opkind_coverage()


def test_opkind_coverage_guard_catches_stale_key(monkeypatch):
    """`_check_opkind_coverage` also catches a STALE key — an entry
    whose op is not (or no longer) a `tir.OpKind`. Mapped-but-unknown is
    as much a drift signal as unmapped."""
    ghost = enum.Enum("_GhostKind", ["GHOST"])
    broken = dict(mapping._OPKIND_LOWERING)
    broken[ghost.GHOST] = MLIRLowering.ARITH
    monkeypatch.setattr(mapping, "_OPKIND_LOWERING", broken)
    with pytest.raises(AssertionError, match=r"stale.*GHOST"):
        mapping._check_opkind_coverage()


# --------------------------------------------------------------------------
# the mapping itself — spot checks + accessors
# --------------------------------------------------------------------------
def test_opkind_lowering_spot_checks():
    """Anchor representative ops to their decision-record lowering so a
    silent reclassification is caught: scalar arithmetic / compare /
    select / cast -> arith; transcendentals -> math; matmul / reduce /
    conv / tensor-creation -> linalg; shape ops -> tensor; locals and
    arrays -> memref; calls and effectful runtime ops -> func; branches
    -> cf; GPU thread / tile index -> gpu; the Helix transforms and
    arena -> helix; the decision-record-deferred encodings -> residual."""
    cases = {
        tir.OpKind.ADD: MLIRLowering.ARITH,
        tir.OpKind.CMP_EQ: MLIRLowering.ARITH,
        tir.OpKind.SELECT: MLIRLowering.ARITH,
        tir.OpKind.WHERE: MLIRLowering.ARITH,
        tir.OpKind.CAST: MLIRLowering.ARITH,
        tir.OpKind.EXP: MLIRLowering.MATH,
        tir.OpKind.POW: MLIRLowering.MATH,
        tir.OpKind.MATMUL: MLIRLowering.LINALG,
        tir.OpKind.CONV2D: MLIRLowering.LINALG,
        tir.OpKind.REDUCE_SUM: MLIRLowering.LINALG,
        tir.OpKind.TENSOR_RAND: MLIRLowering.LINALG,
        tir.OpKind.RESHAPE: MLIRLowering.TENSOR,
        tir.OpKind.BROADCAST: MLIRLowering.TENSOR,
        tir.OpKind.ALLOC_VAR: MLIRLowering.MEMREF,
        tir.OpKind.LOAD_ELEM: MLIRLowering.MEMREF,
        tir.OpKind.CALL: MLIRLowering.FUNC,
        tir.OpKind.RETURN: MLIRLowering.FUNC,
        tir.OpKind.PRINT: MLIRLowering.FUNC,
        tir.OpKind.FFI_CALL: MLIRLowering.FUNC,
        tir.OpKind.BR: MLIRLowering.CF,
        tir.OpKind.COND_BR: MLIRLowering.CF,
        tir.OpKind.THREAD_IDX: MLIRLowering.GPU,
        tir.OpKind.TILE_INDEX_LOAD: MLIRLowering.GPU,
        tir.OpKind.GRAD: MLIRLowering.HELIX,
        tir.OpKind.ARENA_PUSH_PAIR: MLIRLowering.HELIX,
        tir.OpKind.QUANTIZE: MLIRLowering.RESIDUAL,
        tir.OpKind.RESULT_PACK: MLIRLowering.RESIDUAL,
    }
    for op, expected in cases.items():
        assert mlir_lowering_for(op) is expected, op


def test_mlir_lowering_for_is_total_and_consistent():
    """`mlir_lowering_for` is TOTAL over `tir.OpKind` — it returns an
    `MLIRLowering` for every op, never raising a `KeyError` — and
    `is_upstream` derives consistently from the result for every op."""
    for op in tir.OpKind:
        low = mlir_lowering_for(op)
        assert isinstance(low, MLIRLowering)
        assert is_upstream(low) == (low in mapping._UPSTREAM_LOWERINGS)


def test_is_upstream():
    """`is_upstream` is True for each of the eight upstream dialect
    members and False for the custom `helix` dialect and the deferred
    RESIDUAL bucket."""
    for m in (MLIRLowering.ARITH, MLIRLowering.MATH, MLIRLowering.LINALG,
              MLIRLowering.TENSOR, MLIRLowering.MEMREF, MLIRLowering.FUNC,
              MLIRLowering.CF, MLIRLowering.GPU):
        assert is_upstream(m) is True, m
    assert is_upstream(MLIRLowering.HELIX) is False
    assert is_upstream(MLIRLowering.RESIDUAL) is False


def test_dialect_name():
    """`dialect_name` returns the MLIR dialect mnemonic for each of the
    eight upstream dialects and the custom `helix` dialect — nine real
    dialects — and REFUSES RESIDUAL, which names no dialect, with a
    `ValueError`. The refusal is the point: it stops a caller silently
    formatting `residual.<op>` as if RESIDUAL were a real dialect."""
    for m in MLIRLowering:
        if m is MLIRLowering.RESIDUAL:
            continue
        assert dialect_name(m) == m.value
    assert dialect_name(MLIRLowering.HELIX) == "helix"
    with pytest.raises(ValueError, match="RESIDUAL names no MLIR dialect"):
        dialect_name(MLIRLowering.RESIDUAL)


# --------------------------------------------------------------------------
# decision-record-anchored classifications: HELIX and RESIDUAL
# --------------------------------------------------------------------------
def test_helix_dialect_ops_are_the_decision_record_set():
    """The custom `helix` dialect carries EXACTLY the three Helix-
    specific op families the decision record names as "Poor — Helix-
    specific" (section 2.2): the grad / jvp / vmap transforms, the
    `agi.*` metaprogramming ops, and the atomic arena allocator. No
    other op may slip into `helix` — that is the dialect's contract."""
    helix = {op for op, low in mapping._OPKIND_LOWERING.items()
             if low is MLIRLowering.HELIX}
    assert helix == {
        tir.OpKind.GRAD, tir.OpKind.JVP, tir.OpKind.VMAP,
        tir.OpKind.QUOTE, tir.OpKind.SPLICE, tir.OpKind.MODIFY,
        tir.OpKind.REFLECT_HASH,
        tir.OpKind.ARENA_PUSH, tir.OpKind.ARENA_GET,
        tir.OpKind.ARENA_SET, tir.OpKind.ARENA_LEN,
        tir.OpKind.ARENA_PUSH_PAIR, tir.OpKind.ARENA_PUSH_TRIPLE,
    }


def test_residual_ops_are_the_decision_record_deferred_set():
    """RESIDUAL carries EXACTLY the ops whose MLIR home the decision
    record explicitly DEFERRED ("flag for review", an open checklist
    item — section 2.4): the Result<T,E> packed-tag ops and the
    quantize / dequantize encodings. RESIDUAL is an honest "undecided",
    never a silent default for an op that simply lacks a mapping."""
    residual = {op for op, low in mapping._OPKIND_LOWERING.items()
                if low is MLIRLowering.RESIDUAL}
    assert residual == {
        tir.OpKind.QUANTIZE, tir.OpKind.DEQUANTIZE,
        tir.OpKind.RESULT_PACK, tir.OpKind.RESULT_TAG,
        tir.OpKind.RESULT_PAYLOAD,
    }


# --------------------------------------------------------------------------
# the mock-path rule — mapping is pure data, never `import mlir`
# --------------------------------------------------------------------------
def test_mapping_module_is_pure_data_no_mlir_import():
    """THE MOCK-PATH RULE (Stage 210 decision, section 3.2): `mapping`
    is pure data — it imports `helixc.ir.tir` (the home-grown IR) and
    NEVER `import mlir`, at module top level or anywhere. Parse the
    module's AST and confirm not one `import mlir` / `from mlir ...`
    statement — a host-independent structural pin (a string scan would
    false-match this very docstring; an import that merely failed on
    this binding-less machine would pass silently on an MLIR-equipped
    one).

    The scan covers THIS module's own source; `mapping`'s sole import,
    `helixc.ir.tir`, is the home-grown IR and carries no MLIR
    dependency of its own."""
    tree = ast.parse(Path(mapping.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("mlir"), a.name
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("mlir"), node.module
