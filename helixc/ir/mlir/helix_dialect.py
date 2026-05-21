"""
helixc/ir/mlir/helix_dialect.py — the `helix` MLIR dialect op model
(v3.0 Phase E, Stage 211 chunk D).

The structured op model of the custom `helix` dialect — the small
dialect the ratified Stage 210 HYBRID decision
(docs/V3_STAGE210_MLIR_DECISION.md section 2.4) defines for the
~15-20% of Helix ops with no faithful upstream-MLIR home. The decision
record names exactly three op families:

- the compositional transforms `helix.grad` / `helix.jvp` /
  `helix.vmap` — first-class so a Helix pass can pattern-match and
  materialize them before lowering the result into `linalg` / `vector`;
- the AGI metaprogramming ops `helix.quote` / `helix.splice` /
  `helix.modify` / `helix.reflect_hash` — no analogue anywhere in
  MLIR, core to the project's purpose;
- the atomic bump allocator `helix.arena_push` / `arena_get` /
  `arena_set` / `arena_len` / `arena_push_pair` / `arena_push_triple`
  — custom ops so the dialect verifier / op-trait system can ENFORCE
  the un-splittable atomic-pair/triple invariant instead of relying on
  source comments.

This module is the pure-data op model: which ops the dialect has,
their mnemonics, their source Tensor-IR `OpKind`, their category, and
the un-splittable memory trait. It is the single source of truth a
later IRDL registration step and the Stage-212 translation consult.

A module-load guard ties the model to `mapping.py`: the ops modelled
here are EXACTLY the `OpKind`s `mapping` classifies as
`MLIRLowering.HELIX` — the two cannot drift apart.

MOCK-PATH-FIRST: pure data, NEVER `import mlir`. Defining the dialect
as a LIVE MLIR dialect needs IRDL (the Stage 210 decision's no-C++/ODS
mechanism) and the in-process bindings — `helix_dialect_registrability()`
is the probe-gated seam for that, and the actual IRDL emission is built
in Stage 212, when an MLIR `Context` exists to register into. On this
binding-less machine the live dialect DEFERS; this op model does not.

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .. import tir
from . import mapping
from .toolchain import detect_mlir_support


# The dialect namespace — the leading token of every `helix` op's
# dialect-qualified name (`helix.grad`, `helix.arena_push`, ...).
HELIX_DIALECT: str = "helix"


class HelixOpCategory(Enum):
    """The three op families of the `helix` dialect — the exact
    grouping the Stage 210 decision record (section 2.4) names. Every
    `HelixOp` belongs to exactly one."""
    TRANSFORM = "transform"              # grad / jvp / vmap
    METAPROGRAMMING = "metaprogramming"  # quote / splice / modify / ...
    ARENA = "arena"                      # the atomic bump allocator


@dataclass(frozen=True)
class HelixOp:
    """One op of the custom `helix` MLIR dialect.

    Records the STABLE, mock-path-knowable facts about a helix op: its
    `mnemonic` (the dialect-qualified name is `helix.<mnemonic>`), the
    Tensor-IR `source_opkind` it models, its `category`, a human
    `summary`, and whether it carries the `unsplittable` memory-effect
    trait. The precise IRDL operand / result / attribute type spec is
    deliberately NOT recorded here — it is a per-op design pinned at
    the Stage-212 IRDL-registration step, which owns the MLIR type
    system; inventing SSA arity now (the transforms have no emit site
    in the front-end yet) would be fiction.

    Frozen + `__post_init__`-guarded, the house discipline of
    `toolchain.MLIRSupport`.
    """
    mnemonic: str
    source_opkind: tir.OpKind
    category: HelixOpCategory
    summary: str
    unsplittable: bool = False

    def __post_init__(self) -> None:
        if not (isinstance(self.mnemonic, str) and self.mnemonic
                and self.mnemonic.isidentifier()):
            raise ValueError(
                f"HelixOp: mnemonic must be a non-empty identifier so "
                f"`{HELIX_DIALECT}.<mnemonic>` is a valid MLIR op name "
                f"— got {self.mnemonic!r}")
        if not isinstance(self.source_opkind, tir.OpKind):
            raise ValueError(
                f"HelixOp {self.mnemonic!r}: source_opkind must be a "
                f"tir.OpKind — got {self.source_opkind!r}")
        if not isinstance(self.category, HelixOpCategory):
            raise ValueError(
                f"HelixOp {self.mnemonic!r}: category must be a "
                f"HelixOpCategory — got {self.category!r}")
        if not (isinstance(self.summary, str) and self.summary.strip()):
            raise ValueError(
                f"HelixOp {self.mnemonic!r}: summary must be non-blank "
                f"text — got {self.summary!r}")
        # `unsplittable` is a multi-slot atomic memory-effect trait; it
        # is meaningful only for the arena allocator ops.
        if self.unsplittable and self.category is not HelixOpCategory.ARENA:
            raise ValueError(
                f"HelixOp {self.mnemonic!r}: unsplittable is an arena "
                f"memory-effect trait — only ARENA-category ops may "
                f"set it")

    @property
    def qualified_name(self) -> str:
        """The dialect-qualified MLIR op name — e.g. `helix.grad`."""
        return f"{HELIX_DIALECT}.{self.mnemonic}"


# The op model of the `helix` dialect — every op, in decision-record
# section-2.4 order (the three families: transforms, AGI metaprogram-
# ming, arena). `_check_helix_dialect_model` asserts the `source_opkind`
# set here is EXACTLY `mapping.py`'s `MLIRLowering.HELIX` set.
_HELIX_DIALECT_OPS: tuple[HelixOp, ...] = (
    # --- compositional transforms ---
    HelixOp("grad", tir.OpKind.GRAD, HelixOpCategory.TRANSFORM,
            "Reverse-mode gradient transform — the adjoint of a "
            "differentiable function. Materialized by a Helix pass, "
            "then lowered into `linalg` / `vector`."),
    HelixOp("jvp", tir.OpKind.JVP, HelixOpCategory.TRANSFORM,
            "Forward-mode derivative transform (Jacobian-vector "
            "product)."),
    HelixOp("vmap", tir.OpKind.VMAP, HelixOpCategory.TRANSFORM,
            "Vectorizing map — lifts a function over a batch axis."),
    # --- AGI metaprogramming ---
    HelixOp("quote", tir.OpKind.QUOTE, HelixOpCategory.METAPROGRAMMING,
            "Capture an AST fragment as a first-class value."),
    HelixOp("splice", tir.OpKind.SPLICE, HelixOpCategory.METAPROGRAMMING,
            "Insert an AST-valued operand into a quoted fragment."),
    HelixOp("modify", tir.OpKind.MODIFY, HelixOpCategory.METAPROGRAMMING,
            "Apply a verified self-modification — transform `target` "
            "by `transformation`, gated by `verifier`."),
    HelixOp("reflect_hash", tir.OpKind.REFLECT_HASH,
            HelixOpCategory.METAPROGRAMMING,
            "Structural hash of an AST node — a reflective "
            "metaprogramming primitive."),
    # --- the atomic arena allocator ---
    HelixOp("arena_push", tir.OpKind.ARENA_PUSH, HelixOpCategory.ARENA,
            "Push one i32 value onto the shared bump arena; the result "
            "is the slot index."),
    HelixOp("arena_get", tir.OpKind.ARENA_GET, HelixOpCategory.ARENA,
            "Read the i32 value at an arena slot index."),
    HelixOp("arena_set", tir.OpKind.ARENA_SET, HelixOpCategory.ARENA,
            "Overwrite the i32 value at an arena slot index (no "
            "result)."),
    HelixOp("arena_len", tir.OpKind.ARENA_LEN, HelixOpCategory.ARENA,
            "The current arena length — the next free slot index."),
    HelixOp("arena_push_pair", tir.OpKind.ARENA_PUSH_PAIR,
            HelixOpCategory.ARENA,
            "Atomically push two i32 values into adjacent arena slots; "
            "un-splittable, so DCE / CSE / scheduling cannot break the "
            "pair. Result is the slot index of the first.",
            unsplittable=True),
    HelixOp("arena_push_triple", tir.OpKind.ARENA_PUSH_TRIPLE,
            HelixOpCategory.ARENA,
            "Atomically push three i32 values into adjacent arena "
            "slots; un-splittable. Result is the slot index of the "
            "first.",
            unsplittable=True),
)


def _check_helix_dialect_model() -> None:
    """Module-load guard: the `helix` op model is well-formed and
    matches `mapping.py`'s HELIX classification EXACTLY.

    Three drift classes fail loudly here:
    - a duplicate op mnemonic (two `helix.<x>` ops with the same name);
    - two `HelixOp`s sharing a `source_opkind`;
    - the modelled `source_opkind` set diverging from the `OpKind`s
      `mapping` classifies as `MLIRLowering.HELIX` — the load-bearing
      cross-module guard: an op reclassified into / out of HELIX in
      `mapping.py` without a matching op-model edit would otherwise
      leave the dialect silently wrong.

    The three checks run narrowest-first — duplicate mnemonic, then
    duplicate `source_opkind`, then divergence from `mapping.py`'s
    HELIX set — so the raised diagnostic names the most specific
    defect rather than a broad set mismatch."""
    mnemonics = [op.mnemonic for op in _HELIX_DIALECT_OPS]
    if len(mnemonics) != len(set(mnemonics)):
        raise AssertionError(
            f"helixc.ir.mlir.helix_dialect: duplicate op mnemonic in "
            f"_HELIX_DIALECT_OPS ({sorted(mnemonics)})")
    modeled = {op.source_opkind for op in _HELIX_DIALECT_OPS}
    if len(modeled) != len(_HELIX_DIALECT_OPS):
        raise AssertionError(
            "helixc.ir.mlir.helix_dialect: two HelixOps share a "
            "source_opkind in _HELIX_DIALECT_OPS")
    helix_opkinds = {
        op for op in tir.OpKind
        if mapping.mlir_lowering_for(op) is mapping.MLIRLowering.HELIX}
    if modeled != helix_opkinds:
        missing = sorted(o.name for o in helix_opkinds - modeled)
        extra = sorted(o.name for o in modeled - helix_opkinds)
        raise AssertionError(
            f"helixc.ir.mlir.helix_dialect: the op model does not "
            f"match mapping.py's HELIX set — OpKind(s) mapped HELIX "
            f"but not modelled: {missing or 'none'}; modelled but not "
            f"HELIX: {extra or 'none'}")


_check_helix_dialect_model()


# Source-`OpKind` -> `HelixOp` index. Built after the guard, which has
# already proven `source_opkind` is unique — so no entry is silently
# overwritten here.
_HELIX_OP_BY_OPKIND: dict[tir.OpKind, HelixOp] = {
    op.source_opkind: op for op in _HELIX_DIALECT_OPS
}


def helix_dialect_ops() -> tuple[HelixOp, ...]:
    """Every op of the `helix` dialect, in decision-record section-2.4
    order. The tuple and its `HelixOp`s are immutable — safe to share."""
    return _HELIX_DIALECT_OPS


def helix_op_for(opkind: tir.OpKind) -> HelixOp:
    """The `helix`-dialect op modelling a Tensor-IR `OpKind`.

    Defined for exactly the `OpKind`s `mapping.mlir_lowering_for`
    classifies as `MLIRLowering.HELIX`. For any other `OpKind` this
    raises `ValueError` naming the lowering the op actually has —
    rather than a bare `KeyError` — so a miscall is self-diagnosing."""
    op = _HELIX_OP_BY_OPKIND.get(opkind)
    if op is None:
        raise ValueError(
            f"{opkind} is not a helix-dialect op — `mapping` lowers it "
            f"to {mapping.mlir_lowering_for(opkind).name}, not HELIX")
    return op


@dataclass(frozen=True)
class HelixDialectRegistrability:
    """Whether the `helix` dialect can be registered as a live MLIR
    dialect on this machine — and, when it cannot, WHY.

    The registration-seam analogue of `toolchain.MLIRSupport`, built to
    the same mock-path discipline: a DEFERRED result is never silent
    about its reason, so `detail` is always non-empty. Frozen +
    `__post_init__`-guarded."""
    can_register: bool
    detail: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.detail:
            raise ValueError(
                "HelixDialectRegistrability: detail is empty — the "
                "result must explain whether the dialect can register")
        for entry in self.detail:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    f"HelixDialectRegistrability: detail has a blank "
                    f"or non-str entry ({entry!r}) — every line must "
                    f"carry text")


def helix_dialect_registrability() -> HelixDialectRegistrability:
    """Whether the `helix` dialect can be registered as a LIVE MLIR
    dialect on this machine — the IRDL-registration seam, gated behind
    the Stage-211 capability probe and CARRYING the probe's reasons so
    a DEFERRED is never silent about why.

    Registering an in-process dialect via IRDL needs the in-process
    MLIR Python bindings (an `mlir-opt` CLI cannot register one), so
    `can_register` is `detect_mlir_support().can_use_bindings()`. On a
    binding-less machine it is False and every live-dialect step
    DEFERS — the pure-data op model above stays fully usable
    regardless. The actual IRDL emission is built in Stage 212, when an
    MLIR `Context` exists to register the dialect into."""
    support = detect_mlir_support()
    can = support.can_use_bindings()
    if can:
        detail: tuple[str, ...] = (
            "the `helix` dialect can be registered — the in-process "
            "MLIR Python bindings are usable (IRDL emission is wired "
            "in Stage 212)",)
    else:
        detail = (
            "the `helix` dialect cannot be registered as a live MLIR "
            "dialect — the in-process MLIR bindings are not usable:",
            *support.detail)
    return HelixDialectRegistrability(can_register=can, detail=detail)
