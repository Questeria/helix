"""Tests for helixc.ir.mlir.helix_dialect — v3.0 Phase E, Stage 211
chunk D: the `helix` MLIR dialect op model.

`helix_dialect.py` is the pure-data op model of the custom `helix`
dialect — the small dialect the ratified Stage 210 decision record
(docs/V3_STAGE210_MLIR_DECISION.md section 2.4) defines for the
~15-20% of Helix ops with no faithful upstream-MLIR home: the
compositional transforms (grad / jvp / vmap), the AGI metaprogramming
ops (quote / splice / modify / reflect_hash), and the atomic bump
allocator (the six arena ops).

These tests pin: the `HelixOp` frozen record and its `__post_init__`
rejections; the `HelixOpCategory` enum; that the op model covers
EXACTLY `mapping.py`'s `MLIRLowering.HELIX` set (the load-bearing
cross-module drift guard) and that the guard is not vacuous; the
`helix_op_for` accessor's totality and self-diagnosing miss; the
un-splittable memory trait on the atomic arena pushes; the probe-gated
registration seam `helix_dialect_registrability`; and — the mock-path
rule — that the module is pure data and never `import mlir`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from helixc.ir import tir
from helixc.ir.mlir import helix_dialect, mapping
from helixc.ir.mlir.helix_dialect import (
    HELIX_DIALECT, HelixDialectRegistrability, HelixOp, HelixOpCategory,
    helix_dialect_ops, helix_dialect_registrability, helix_op_for,
)
from helixc.ir.mlir.toolchain import detect_mlir_support


def _helix_opkinds() -> set[tir.OpKind]:
    """The `tir.OpKind`s `mapping` classifies as `MLIRLowering.HELIX` —
    the set the op model must cover exactly."""
    return {op for op in tir.OpKind
            if mapping.mlir_lowering_for(op) is mapping.MLIRLowering.HELIX}


# --------------------------------------------------------------------------
# HELIX_DIALECT + HelixOpCategory
# --------------------------------------------------------------------------
def test_helix_dialect_namespace():
    """The dialect namespace is the literal `helix` — the leading token
    of every dialect-qualified op name."""
    assert HELIX_DIALECT == "helix"


def test_helix_op_category_members():
    """`HelixOpCategory` names the three op families the decision record
    section 2.4 enumerates — transforms, AGI metaprogramming, arena."""
    assert {c.name for c in HelixOpCategory} == {
        "TRANSFORM", "METAPROGRAMMING", "ARENA"}
    values = [c.value for c in HelixOpCategory]
    assert len(values) == len(set(values)), "values must be unique"


# --------------------------------------------------------------------------
# HelixOp — __post_init__ rejects illegal field shapes
# --------------------------------------------------------------------------
def test_helix_op_rejects_non_identifier_mnemonic():
    """A mnemonic must be a non-empty identifier so `helix.<mnemonic>`
    is a valid MLIR op name — a blank or hyphenated one is rejected."""
    with pytest.raises(ValueError, match="mnemonic must be"):
        HelixOp("", tir.OpKind.GRAD, HelixOpCategory.TRANSFORM, "x")
    with pytest.raises(ValueError, match="mnemonic must be"):
        HelixOp("bad-name", tir.OpKind.GRAD, HelixOpCategory.TRANSFORM,
                "x")


def test_helix_op_rejects_blank_summary():
    """Every helix op must carry a human summary — a blank one is a
    documentation hole and is rejected."""
    with pytest.raises(ValueError, match="summary must be non-blank"):
        HelixOp("grad", tir.OpKind.GRAD, HelixOpCategory.TRANSFORM,
                "   ")


def test_helix_op_rejects_unsplittable_outside_arena():
    """`unsplittable` is a multi-slot atomic memory-effect trait — it is
    meaningful only for the arena ops; setting it on a transform /
    metaprogramming op is rejected."""
    with pytest.raises(ValueError, match="unsplittable"):
        HelixOp("grad", tir.OpKind.GRAD, HelixOpCategory.TRANSFORM,
                "a transform", unsplittable=True)


def test_helix_op_qualified_name():
    """`qualified_name` is the dialect-qualified MLIR op name —
    `helix.<mnemonic>`."""
    op = HelixOp("arena_push", tir.OpKind.ARENA_PUSH,
                 HelixOpCategory.ARENA, "push one value")
    assert op.qualified_name == "helix.arena_push"
    # every modelled op's qualified name is `helix.`-prefixed
    for modelled in helix_dialect_ops():
        assert modelled.qualified_name == f"helix.{modelled.mnemonic}"


# --------------------------------------------------------------------------
# _HELIX_DIALECT_OPS — the op model itself
# --------------------------------------------------------------------------
def test_helix_dialect_ops_well_formed():
    """The op model is the 13 helix ops; mnemonics and source `OpKind`s
    are each unique (no two ops collide on a name or an IR op)."""
    ops = helix_dialect_ops()
    assert len(ops) == 13
    mnemonics = [op.mnemonic for op in ops]
    assert len(mnemonics) == len(set(mnemonics)), "mnemonics unique"
    opkinds = [op.source_opkind for op in ops]
    assert len(opkinds) == len(set(opkinds)), "source OpKinds unique"
    assert all(isinstance(op, HelixOp) for op in ops)


def test_helix_dialect_ops_categories():
    """The op model splits into the decision record's three families —
    3 transforms, 4 AGI metaprogramming ops, 6 arena ops."""
    by_cat: dict[HelixOpCategory, set[str]] = {}
    for op in helix_dialect_ops():
        by_cat.setdefault(op.category, set()).add(op.mnemonic)
    assert by_cat[HelixOpCategory.TRANSFORM] == {"grad", "jvp", "vmap"}
    assert by_cat[HelixOpCategory.METAPROGRAMMING] == {
        "quote", "splice", "modify", "reflect_hash"}
    assert by_cat[HelixOpCategory.ARENA] == {
        "arena_push", "arena_get", "arena_set", "arena_len",
        "arena_push_pair", "arena_push_triple"}


def test_unsplittable_ops_are_the_atomic_arena_pushes():
    """EXACTLY the two multi-slot arena pushes carry the `unsplittable`
    trait — the decision record (section 2.4) makes `arena_push_pair` /
    `arena_push_triple` un-splittable so DCE / CSE / scheduling cannot
    break the atomic-pair/triple invariant. No other op sets it."""
    unsplittable = {op.mnemonic for op in helix_dialect_ops()
                    if op.unsplittable}
    assert unsplittable == {"arena_push_pair", "arena_push_triple"}


# --------------------------------------------------------------------------
# the cross-module drift guard — model matches mapping.py's HELIX set
# --------------------------------------------------------------------------
def test_helix_dialect_model_matches_mapping_helix_set():
    """THE load-bearing cross-module guard: the op model covers EXACTLY
    the `OpKind`s `mapping.py` classifies as `MLIRLowering.HELIX` — the
    dialect and the lowering table cannot drift apart."""
    modeled = {op.source_opkind for op in helix_dialect_ops()}
    assert modeled == _helix_opkinds()
    helix_dialect._check_helix_dialect_model()  # the guard itself passes


def test_helix_dialect_model_guard_is_not_vacuous(monkeypatch):
    """`_check_helix_dialect_model` genuinely catches a model that has
    drifted from `mapping.py`'s HELIX set — drop an op and confirm it
    raises, naming the now-unmodelled `OpKind`. Order-robust: it drops
    whichever op is first and expects that op's name in the message, so
    a reordering of `_HELIX_DIALECT_OPS` cannot break it spuriously."""
    dropped = helix_dialect._HELIX_DIALECT_OPS[0]
    monkeypatch.setattr(helix_dialect, "_HELIX_DIALECT_OPS",
                        helix_dialect._HELIX_DIALECT_OPS[1:])
    with pytest.raises(
            AssertionError,
            match=rf"does not match.*{dropped.source_opkind.name}"):
        helix_dialect._check_helix_dialect_model()


def test_helix_dialect_model_guard_catches_duplicate_mnemonic(monkeypatch):
    """`_check_helix_dialect_model` also catches a duplicate op
    mnemonic — two `helix.<x>` ops with the same name. The injected
    clash uses a fresh, non-HELIX `source_opkind` (ADD) so ONLY the
    mnemonic collides — isolating the mnemonic branch of the guard."""
    ops = helix_dialect._HELIX_DIALECT_OPS
    clash = HelixOp(ops[0].mnemonic, tir.OpKind.ADD,
                    HelixOpCategory.TRANSFORM, "a name clash")
    monkeypatch.setattr(helix_dialect, "_HELIX_DIALECT_OPS",
                        ops + (clash,))
    with pytest.raises(AssertionError, match="duplicate op mnemonic"):
        helix_dialect._check_helix_dialect_model()


def test_helix_dialect_model_guard_catches_duplicate_opkind(monkeypatch):
    """`_check_helix_dialect_model` also catches two `HelixOp`s sharing
    a `source_opkind` — the injected clash has a FRESH mnemonic but
    reuses an existing `source_opkind`, isolating that guard branch."""
    ops = helix_dialect._HELIX_DIALECT_OPS
    clash = HelixOp("grad_again", ops[0].source_opkind,
                    HelixOpCategory.TRANSFORM, "a source_opkind clash")
    monkeypatch.setattr(helix_dialect, "_HELIX_DIALECT_OPS",
                        ops + (clash,))
    with pytest.raises(AssertionError, match="share a"):
        helix_dialect._check_helix_dialect_model()


# --------------------------------------------------------------------------
# helix_op_for — the accessor
# --------------------------------------------------------------------------
def test_helix_op_for():
    """`helix_op_for` returns the `HelixOp` modelling a helix `OpKind`,
    and for a non-helix `OpKind` raises `ValueError` naming the lowering
    the op actually has — never a bare `KeyError`."""
    grad = helix_op_for(tir.OpKind.GRAD)
    assert grad.mnemonic == "grad"
    assert grad.source_opkind is tir.OpKind.GRAD
    with pytest.raises(ValueError, match="not a helix-dialect op"):
        helix_op_for(tir.OpKind.ADD)        # ADD lowers to arith


def test_helix_op_for_is_total_over_helix_opkinds():
    """`helix_op_for` is TOTAL over the HELIX-mapped `OpKind`s — it
    returns a `HelixOp` with the matching `source_opkind` for every one,
    and there are exactly 13."""
    helix_opkinds = _helix_opkinds()
    assert len(helix_opkinds) == 13
    for opkind in helix_opkinds:
        op = helix_op_for(opkind)
        assert isinstance(op, HelixOp)
        assert op.source_opkind is opkind


# --------------------------------------------------------------------------
# the probe-gated registration seam + the mock-path rule
# --------------------------------------------------------------------------
def test_helix_dialect_registrability_is_probe_gated():
    """`helix_dialect_registrability` is the IRDL-registration seam: it
    derives from the Stage-211 capability probe — specifically
    `can_use_bindings()`, since registering an in-process dialect needs
    the bindings (the `mlir-opt` CLI cannot) — and CARRIES the probe's
    reasons so a cannot-register result is never silent. On this
    binding-less dev machine it cannot register; the op model stays
    usable regardless."""
    r = helix_dialect_registrability()
    assert isinstance(r, HelixDialectRegistrability)
    assert r.can_register == detect_mlir_support().can_use_bindings()
    assert r.can_register is False   # this dev machine has no bindings
    assert r.detail                  # a DEFERRED always explains itself
    assert any("MLIR" in line for line in r.detail)


def test_helix_dialect_module_is_pure_data_no_mlir_import():
    """THE MOCK-PATH RULE (Stage 210 decision, section 3.2):
    `helix_dialect` is pure data — it NEVER `import mlir`, at module
    top level or anywhere. Parse the module's AST and confirm not one
    `import mlir` / `from mlir ...` statement — a host-independent
    structural pin. (The probe-gated `helix_dialect_registrability`
    reaches the bindings only lazily, via `toolchain.detect_mlir_
    support`, never an import in this module.)"""
    tree = ast.parse(
        Path(helix_dialect.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("mlir"), a.name
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("mlir"), node.module
