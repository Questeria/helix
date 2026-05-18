"""
helixc/frontend/typecheck.py — Helix type checker with compile-time shape checking.

Phase 3-iv adds real Presburger-backed shape checking at function call sites.
When a function declares parameters with size-typed tensor shapes, the type
checker:
  1. Treats each `size` generic param as a Presburger variable
  2. Unifies the formal parameter's shape with the argument's actual shape,
     producing equality constraints between variables and concrete values
  3. Adds the function's `where` clauses to the constraint set
  4. Asks the solver: "is the call shape-consistent?" If the solver can prove
     a contradiction, the call is rejected with a diagnostic.

This catches matmul-style bugs (inner dims must match) at compile time.

License: Apache 2.0
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Literal, Optional

# Stage 52 closure gate-11 type-design F2 fix (was LOW, promoted to
# MEDIUM after Inc 7 expanded the surface to 5 consult sites + 3
# install sites + 4 module-level dicts). Single-line Literal alias
# gives mypy/pyright a chance to catch typos like "goal" vs "goals"
# at type-check time instead of silently bypassing the launder check.
# Zero runtime cost.
ModalKind = Literal["known", "believed", "goal", "uncertain"]

from . import ast_nodes as A
from . import presburger as P


# ============================================================================
# Type representations (separate from AST — these are "checked" types)
# ============================================================================
@dataclass(frozen=True)
class Type:
    """Base for resolved types."""
    pass


@dataclass(frozen=True)
class TyPrim(Type):
    """Primitive type by name: i32, f32, bool, etc."""
    name: str


@dataclass(frozen=True)
class TyRefined(Type):
    """Stage 31: named refinement alias erased to `base` after checks."""
    name: str
    base: Type
    predicates: tuple[A.Expr, ...]


@dataclass(frozen=True)
class ProofObligation:
    """Machine-readable Stage 31 proof obligation artifact."""
    kind: str
    context: str
    refinement: str
    predicate: str
    status: str
    line: int
    col: int
    value: str | None = None
    trap: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "kind": self.kind,
            "context": self.context,
            "refinement": self.refinement,
            "predicate": self.predicate,
            "status": self.status,
            "span": {"line": self.line, "col": self.col},
        }
        if self.value is not None:
            data["value"] = self.value
        if self.trap is not None:
            data["trap"] = self.trap
        return data


@dataclass(frozen=True)
class ProofCarry:
    """Machine-readable Stage 34 already-carried proof evidence."""
    kind: str
    context: str
    source_refinement: str
    target_refinement: str
    strategy: str
    line: int
    col: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "context": self.context,
            "source_refinement": self.source_refinement,
            "target_refinement": self.target_refinement,
            "strategy": self.strategy,
            "span": {"line": self.line, "col": self.col},
        }


@dataclass(frozen=True)
class TyVar(Type):
    """Generic type variable bound by a function signature."""
    name: str


@dataclass(frozen=True)
class TySize(Type):
    """A size value (compile-time integer). Tracked as an opaque symbol."""
    name: str


@dataclass(frozen=True)
class TyTensor(Type):
    dtype: Type
    shape: tuple[Type, ...]   # each element is TySize, TyPrim(int...), or computed expr-type
    device: Optional[str] = None
    layout: Optional[str] = None


@dataclass(frozen=True)
class TyTile(Type):
    dtype: Type
    shape: tuple[Type, ...]
    memspace: str


@dataclass(frozen=True)
class TyTuple(Type):
    elems: tuple[Type, ...]


@dataclass(frozen=True)
class TyArray(Type):
    elem: Type
    size: Type


@dataclass(frozen=True)
class TyRef(Type):
    inner: Type
    is_mut: bool


@dataclass(frozen=True)
class TyPtr(Type):
    """Stage 16.5: raw pointer *const T or *mut T (for FFI). Treated as a
    u64 at the ABI level."""
    inner: Type
    is_mut: bool


@dataclass(frozen=True)
class TyFn(Type):
    params: tuple[Type, ...]
    ret: Type


@dataclass(frozen=True)
class TyUnit(Type):
    pass


@dataclass(frozen=True)
class TyUnknown(Type):
    """Used during inference; should be resolved before checking completes."""
    hint: str = ""


@dataclass(frozen=True)
class TyStruct(Type):
    """A nominal struct type; field types resolved via TypeChecker._struct_decls."""
    name: str


@dataclass(frozen=True)
class TyEnum(Type):
    """A nominal enum type; variant payloads are tracked by _enum_decls."""
    name: str


@dataclass(frozen=True)
class TyDiff(Type):
    """D<T> — a value of type T that participates in gradient computation.
    Operations on TyDiff values propagate differentiability through results.
    Mixing TyDiff with non-Diff values is allowed (the result becomes
    TyDiff). The only way to extract the underlying T without gradient
    tracking is `detach(x)`."""
    inner: Type


@dataclass(frozen=True)
class TyLogic(Type):
    """Logic<T> — a relational / logical wrapper over T (Stage 24, Tier-3
    moat).

    Semantically, a `Logic<T>` value represents an atom of relational
    structure: a `Logic<Person>` is a relational entity, a
    `Logic<bool>` is a fuzzy truth value, a `Logic<Edge>` is a graph
    edge atom. Composing `D<Logic<T>>` then represents a *differen-
    tiable* relational/symbolic value — the core of provenance-typed
    neuro-symbolic computation.

    Phase-0: parse + type-level representation. Downstream:
      - Fuzzy semantics for logic ops (AND -> min, OR -> max,
        differentiable via straight-through or sigmoid relaxations)
      - Provenance lattice tracking which input atoms contributed to
        each derived value
      - Trap 24100 emitted if a non-Logic value is passed where a
        Logic-typed parameter is required, or vice versa, in a
        provenance-sensitive context (e.g. AD over a logic op).

      Trap-id history: the original reservation was 24001, but Audit
      28.8 A4 / Finding 4 found that kovc.hx:4220-4221 already emits
      24001 for `bf16 % bf16` (per the bootstrap's `AST_tag * 1000 +
      sub_id` convention, AST_MOD = 24 → 24001). Stage-level provenance
      reservations now use the 241xx prefix to keep the namespaces
      disjoint."""
    inner: Type
    # provenance: optional compile-time provenance tag. None means
    # "unconstrained"; a string like "infer_rule:parent" carries the
    # rule that produced this value. Phase-0 keeps it None; the field
    # exists so later passes can populate it without further schema
    # churn.
    provenance: Optional[str] = None


@dataclass(frozen=True)
class TyMemTier(Type):
    """A value tagged with a memory tier: Working / Episodic / Semantic /
    Procedural. Each tier has different consolidation, decay, and retrieval
    semantics. Cross-tier operations require explicit transitions
    (consolidate, recall, retrieve)."""
    tier: str        # "working", "episodic", "semantic", "procedural"
    inner: Type


@dataclass(frozen=True)
class TyFrame(Type):
    """Stage 38 — a value tagged with a spatial reference frame:
    WorldFrame / RobotFrame / CameraFrame. Real-world AGI workloads
    (robotics, vision, navigation) need to track WHICH frame a
    coordinate is expressed in — a camera's `(0.5, 0.3, 1.2)` means
    nothing without knowing it's CameraFrame vs WorldFrame vs
    RobotFrame. Cross-frame operations require explicit transforms
    (to_world, to_robot, to_camera — Stage 38 Inc 2)."""
    frame: str       # "world", "robot", "camera"
    inner: Type


@dataclass(frozen=True)
class TyTemporal(Type):
    """Stage 39 — a value tagged with a temporal kind: Past /
    Present / Future / Eternal. Real-world AGI reasoning needs to
    track WHEN a fact was true. 'The robot saw a cat at coordinate
    X' only matters if we know whether that was 5 seconds ago or
    5 days ago. Cross-temporal transitions (to_past, forecast,
    recall_past, actualize) move values between kinds — Stage 39
    Inc 2."""
    kind: str        # "past", "present", "future", "eternal"
    inner: Type


@dataclass(frozen=True)
class TyModal(Type):
    """Stage 40 — a value tagged with a modal/epistemic kind:
    Known / Believed / Goal / Uncertain. Real-world AGI reasoning
    needs to track WHY it accepts a proposition. Treating a goal
    as a known fact (category mistake at the heart of many AI
    safety failures) is caught at compile time. Cross-modal
    transitions (`confirm`: Believed -> Known when observed;
    `act_on`: Goal -> Known when achieved) — Stage 40 Inc 2.
    Composes with TyTemporal: `Known<Past<i32>>` = "I directly
    observed this past fact" vs `Believed<Past<i32>>` = "I
    inferred this past fact"."""
    kind: str        # "known", "believed", "goal", "uncertain"
    inner: Type


@dataclass(frozen=True)
class TyCausal(Type):
    """Stage 41 — a value tagged with a causal/intent kind:
    Cause / Effect / Joint / Independent. Real-world AGI reasoning
    needs to track WHY something is true beyond observation. The
    robot reaching position X is a Cause if it triggers a
    downstream plan revision, an Effect if it followed from some
    upstream decision, a Joint observation if multiple causes
    contributed, or Independent if causally isolated. AGI that
    mis-attributes causation makes systematically wrong decisions
    about which knob to turn next. Cross-causal transitions
    (`propagate`: Cause -> Effect when applied; `aggregate`:
    Effect -> Joint when multiple causes contribute; `isolate`:
    Joint -> Independent when no upstream actually matters) —
    Stage 41 Inc 2. Composes orthogonally with the 4-stack
    AGI quartet completed at Stage 40."""
    kind: str        # "cause", "effect", "joint", "independent"
    inner: Type


@dataclass(frozen=True)
class TyResult(Type):
    """Stage 46 — `Result<T, E>` two-parameter wrapper for
    error handling. Real programs need a way to say "this
    function either succeeds with a T, or fails with an E"
    without crashing on the failure case. Helix's first
    two-parameter wrapper family.

    Phase-0: identity-lowered at IR (the Ok/Err discriminant
    lives at the type system level only — no runtime tag).
    Stage 47+ will add the `?` operator (parser change) and
    Stage 48+ a real runtime tag once `?` early-return
    semantics need it.

    Built-in surface: `Ok(v)` / `Err(e)` constructors;
    `unwrap_ok` / `unwrap_err` accessors (panic on wrong
    variant); `is_ok` / `is_err` queries; `map_ok(r, f)` /
    `map_err(r, f)` combinators.

    Composes with the Stage 37-41 AGI semantic-type quintet
    naturally: `Result<Known<i32>, ParseError>` is a fact
    we either directly observed or failed to parse."""
    ok_ty: Type      # the success-variant inner type
    err_ty: Type     # the failure-variant inner type


@dataclass(frozen=True)
class TySkill(Type):
    """Skill<F> — a learned procedure with a known difficulty. Produced by
    `learn_to`; called like a function. The runtime maintains a registry
    of skills so the AGI can request "skills at difficulty X"."""
    inner: Type           # the function-like inner type
    task: str = ""        # task identifier (compile-time-known)


@dataclass(frozen=True)
class TyQuote(Type):
    """Audit 28.8 B10: type of `quote(...)` — a reified AST fragment.

    Phase-0 representation: the inner Type captures the type of the
    quoted body *as if* it were evaluated in the surrounding scope.
    Pattern matches Stage-11 reflection — `splice(q)` validates that
    `q` is `TyQuote`-typed before unwrapping back to the inner type."""
    inner: Type


# Audit 28.8 B13: numeric-type widening rank for TyDiff inner-type
# mixing. Higher rank dominates: f64 > f32 > bf16 > f16 > i64 > i32 ...
# Used by `_widen_diff_inner` to pick the result of D<T1> + D<T2>.
#
# Audit 28.8 cycle 2 B:C1: add fp8 / mxfp4 / nvfp4 / char so they
# don't fall through to rank -1 (which made an int "dominate" a
# quantized float — a float-to-int silent collapse). Quantized floats
# are ranked just below f16 — they're floats so they should beat any
# integer, but they have less precision than f16/bf16/f32/f64.
#
# Audit 28.8 cycle 2 B:C4: signed-vs-unsigned at the same width gets
# slightly different ranks so a tie doesn't silently left-win and
# drop the sign domain. Unsigned wins by +1 (matching C's promotion
# rule: in mixed signed/unsigned at the same width, the unsigned
# operand is the "wider" type for the operation). This means
# `u32 + i32` widens to u32 with a warning, and `i32 + u32` ALSO
# widens to u32 with a warning — symmetric.
# Audit 28.8 cycle 3 C3-2: pointer-width aliases on 64-bit targets.
# `isize` and `i64` (likewise `usize` and `u64`) sit at the same rank
# because they ARE the same machine width — there's no signedness or
# precision domain to drop when widening one to the other. The widen
# helper canonicalizes these before deciding whether a tie callback
# should fire, so `D<i64> + D<isize>` is silent.
# Audit 28.8 cycle 4 C4-1: trap-id constants promoted from literals
# embedded in diagnostic messages to real module-level identifiers so
# the registry in docs/lang/trap-ids.md cross-references actual symbols
# (per the "How to add a new trap ID" protocol).
TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO = 28802  # _resolve_size_expr
TRAP_CAST_MATRIX_RECURSION_DEPTH = 28803  # _check_cast_compat


_WIDEN_NAME_ALIASES: dict[str, str] = {
    "isize": "i64",
    "usize": "u64",
}


def _widen_canon_name(name: str) -> str:
    return _WIDEN_NAME_ALIASES.get(name, name)


_WIDEN_RANK: dict[str, int] = {
    "bool": 1,
    "char": 5,
    "i8": 10, "u8": 11,
    "i16": 20, "u16": 21,
    "i32": 30, "u32": 31,
    "i64": 40, "u64": 41, "isize": 40, "usize": 41,
    # Quantized floats live ABOVE every integer (so `D<fp8> + D<i64>`
    # picks fp8, not i64 — a float-to-int collapse pre-fix). They sit
    # below f16/bf16 because they have less precision; mxfp4/nvfp4
    # are 4-bit and lower-precision than fp8 (8-bit), so they sit
    # just below fp8 but above all integers. With this ordering,
    # any float-vs-int pair widens to the float (correct AD semantics
    # — gradient over an int tape is undefined).
    "mxfp4": 43, "nvfp4": 43,                   # quantized 4-bit floats
    "fp8":   45,                                # quantized 8-bit float
    "f16":   50, "bf16": 51,
    "f32":   60,
    "f64":   70,
}


def _widen_diff_inner(a: "Type", b: "Type",
                       _warn_cb=None,
                       _span=None) -> "Type":
    """Audit 28.8 B13: pick the wider of two D<>-inner types.

    Used when a TyDiff binop receives D<T1> + D<T2> with T1 != T2.
    Pre-fix this silently coerced T2 to T1; the new behavior widens
    + emits a warning (AD002 / trap 24200) so the user can fix the
    mix at source.

    Audit 28.8 cycle 2 B:C4: if the optional `_warn_cb` is provided
    AND the two types tie on rank (e.g. `u32` vs `i32` both at rank
    30/31 — they were at 30/30 pre-fix), emit a callback so the
    caller can issue AD002 with a "signedness flip" hint. With the
    new asymmetric ranks (B:C1+B:C4), exact ties now only occur
    when both sides are absent from the table (rank -1 fallback)
    or are mxfp4/nvfp4 same-name pairs.

    For TyPrim pairs, rank lookup picks the larger (or in same-rank
    case, picks `a` for backward compatibility but fires the warn
    callback). For pairs where one side isn't a TyPrim, the
    non-TyUnknown side wins.
    """
    if isinstance(a, TyPrim) and isinstance(b, TyPrim):
        ra = _WIDEN_RANK.get(a.name, -1)
        rb = _WIDEN_RANK.get(b.name, -1)
        # Same-rank-and-different-name tie (B:C4): callback if the
        # caller passed one. With the asymmetric ranks (B:C1+B:C4)
        # this should be rare in practice but the safety net stays.
        # Cycle 3 C3-2: pointer-width aliases (isize/i64, usize/u64)
        # canonicalize to the same name, so they're NOT a tie.
        ca = _widen_canon_name(a.name)
        cb = _widen_canon_name(b.name)
        if ra == rb and ca != cb and _warn_cb is not None:
            _warn_cb(a, b, _span)
        return a if ra >= rb else b
    if isinstance(a, TyUnknown):
        return b
    return a


# ============================================================================
# Type errors
# ============================================================================
class TypeError_(Exception):
    def __init__(self, msg: str, span: A.Span,
                 hint: Optional[str] = None):
        full = f"{span.line}:{span.col}: type error: {msg}"
        if hint:
            full += f"\n  hint: {hint}"
        super().__init__(full)
        self.span = span
        self.msg = msg
        self.hint = hint

    def render(self, source: Optional[str] = None,
               filename: str = "<input>",
               color: Optional[bool] = None) -> str:
        """Format with source-line context via the shared diagnostics
        module (Stage 22). Includes hint as a `= hint:` line when set.
        If `source` is None, falls back to the bare 'line:col: msg' form."""
        if source is None:
            return str(self)
        from .diagnostics import render_caret
        return render_caret(
            filename=filename,
            line=self.span.line,
            col=self.span.col,
            msg=self.msg,
            source=source,
            hint=self.hint,
            level="error",
            color=color,
        )


# ============================================================================
# Symbol table
# ============================================================================
PRIMITIVES = {
    "i8", "i16", "i32", "i64", "isize",
    "u8", "u16", "u32", "u64", "usize",
    "bool", "char",
    "bf16", "f16", "f32", "f64",
    "fp8", "mxfp4", "nvfp4", "ternary",
    # Stage 28.9 cycle-105 F105-1 fix (type-design HIGH, conf 90): "()" is
    # the unit type — it has its own canonical class TyUnit, not TyPrim.
    # Pre-fix the set contained "()", so TyName("()") resolved to
    # TyPrim("()") in source-typed positions (e.g. `fn foo() -> () {}`)
    # while implicit-unit paths produced TyUnit(); the dataclass __eq__
    # cascade rejected the cross-class pair and emitted a spurious
    # "type error: () does not match ()". _resolve_type now maps the
    # textual "()" name to TyUnit() directly, eliminating the duplicate
    # representation.
}


@dataclass
class Scope:
    parent: Optional["Scope"] = None
    locals: dict[str, Type] = field(default_factory=dict)
    mutables: set[str] = field(default_factory=set)

    def lookup(self, name: str) -> Optional[Type]:
        if name in self.locals:
            return self.locals[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None

    def lookup_mutable(self, name: str) -> bool:
        if name in self.locals:
            return name in self.mutables
        if self.parent is not None:
            return self.parent.lookup_mutable(name)
        return False

    def define(self, name: str, ty: Type, is_mut: bool = False) -> None:
        self.locals[name] = ty
        if is_mut:
            self.mutables.add(name)
        else:
            self.mutables.discard(name)


@dataclass
class FunctionSig:
    name: str
    generics: list[A.GenericParam]
    params: list[tuple[str, Type]]
    ret: Type
    # AGI-specific: effect/capability set
    is_pure: bool = False                    # @pure attribute
    effects: frozenset[str] = frozenset()    # @effect(...) declared capabilities


# ============================================================================
# Type checker
# Stage 28.9 cycle 93 audit-T F1 (extended cycle 95 F1): literal-suffix
# domain sets for the IntLit/FloatLit kind-coherence checks in
# `_check_expr`. Float-domain suffix on IntLit (or vice versa) is a
# silent cross-domain miscompile pre-fix.
#
# Cycle 95 expanded `_FLOAT_PRIM_NAMES` to include the quantized-float
# suffixes the lexer accepts (`fp8`, `mxfp4`, `nvfp4`) and the
# unclassified-low-precision `ternary` suffix. Cycle 94 audits found
# the cycle-93 set was incomplete vs lexer at lines 338-341 — `42_fp8`
# bypassed the kind-coherence check and reproduced the original
# defect (raw int bits in float slot). The sets here must be kept in
# sync with the lexer suffix whitelist.
_FLOAT_PRIM_NAMES = frozenset({
    "f16", "bf16", "f32", "f64",
    "fp8", "mxfp4", "nvfp4", "ternary",
})
_INT_PRIM_NAMES = frozenset({
    "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64",
    "isize", "usize",
})


# ============================================================================
# Stage 52 closure gate-2 type-design F3 fix: module-level constant
# for the modal-eliminator → kind mapping. Pre-fix this dict was
# duplicated at 3 callsites (Let-stmt populate, Assign-stmt populate,
# into_X consult-guard membership check). A future 5th modal kind
# (e.g. a hypothetical `Suspected<T>` between Believed and Uncertain)
# would have required touching all 3 sites. Hoisting kills the
# divergence risk.
_MODAL_ELIM_TO_KIND: dict[str, ModalKind] = {
    "from_known":     "known",
    "from_believed":  "believed",
    "from_goal":      "goal",
    "from_uncertain": "uncertain",
}

# Stage 53 Inc 1: hoist the modal upgrade hint table to module
# level so the launder check at the Stage 53 user-fn call site
# (helper-fn indirection) can share the same hint copy with the
# existing Stage 40 F1 into_X consult. Pre-Stage-53 this was a
# 3x-duplicated local dict — same gate-2 F3 hoisting pattern that
# kicked off _MODAL_ELIM_TO_KIND single-source-of-truth.
_MODAL_UPGRADE_HINT: dict[tuple[str, str], str] = {
    ("believed", "known"):
        "use `confirm(b)` — the audited "
        "Believed -> Known epistemic upgrade",
    ("goal", "known"):
        "use `act_on(g)` — the audited "
        "Goal -> Known epistemic upgrade",
    ("uncertain", "known"):
        "resolve uncertainty via outside "
        "observation BEFORE the value enters "
        "the type system as Known; an unwrap-"
        "rewrap is not an observation and "
        "cannot manufacture epistemic "
        "certainty",
    ("uncertain", "believed"):
        "form the belief via inference from "
        "non-Uncertain facts; Uncertain values "
        "gate info-gathering actions, they do "
        "not seed beliefs by themselves",
    ("uncertain", "goal"):
        "the planner sets a Goal independently "
        "of any Uncertain<T>; an unwrap-rewrap "
        "implies the goal came from uncertainty, "
        "which is a category mistake",
}


class TypeChecker:
    def __init__(self, prog: A.Program):
        self.prog = prog
        self.functions: dict[str, FunctionSig] = {}
        self.constraints: list[A.Expr] = []   # collected, not solved (v0.1)
        self.errors: list[TypeError_] = []
        # Effect-checking state: current function's pure/effect declaration
        self._current_pure: bool = False
        self._current_effects: frozenset[str] = frozenset()
        self._current_fn_name: str = ""
        self._current_is_kernel: bool = False
        self._current_hbm_tile_indexables: set[str] = set()
        # Cascade-suppression set for unbound-name diagnostics. Initialized
        # here so re-running check() on the same instance doesn't carry
        # stale entries that would silence real new errors.
        self._seen_unbound: set[str] = set()
        # Result-constructor provenance map (Stage 46 G2-F1
        # origin; Stage 48 gate-2 F1+M5 + gate-3 G3-F1 extensions).
        #
        # Invariant: this dict mirrors the names in scope inside
        # the CURRENT function's body, restored across block
        # boundaries (with scope-aware mutate-vs-shadow disambig)
        # and cleared across function boundaries. Used by
        # unwrap_ok / unwrap_err / __try to detect statically
        # determinable wrong-arm cases.
        #
        # Stewardship sites (search for `_result_constructor_provenance`):
        #   Mutation sites (WRITE to the dict — keep in lockstep):
        #     1. declaration (here)
        #     2. cleared at check() entry (per check() invocation)
        #     3. cleared at _check_fn entry (per fn — gate-2 M5 fix)
        #     4. snapshot + mutate-aware restore across _check_block
        #        (gate-2 F1 + gate-3 G3-F1 + gate-5 G4-F2 fixes;
        #        uses the parallel _result_let_block_scopes AND
        #        _result_assigns_block_scopes set-stacks to
        #        distinguish inner-let shadows from inner-assign
        #        outer mutations, including ASSIGN-then-LET-shadow
        #        on the same name).
        #     5. Let-stmt populates the prov dict AND records the
        #        name in the current block's let-set; pops on
        #        opaque RHS; propagates through map_ok / map_err
        #     6. Assign-stmt pops/overwrites AND records the name
        #        in the current block's assigns-set
        #
        #   Consumer (read) sites — these READ the dict to make
        #   typecheck decisions and must be migrated in lockstep
        #   when the dict's shape changes (gate-4 G4-M2 stewardship):
        #     C1. unwrap_ok / unwrap_err static-provenance reject
        #         (search `if bn in ("unwrap_ok", "unwrap_err"):`
        #         in _check_expr)
        #     C2. __try (`?`) static-provenance reject — gate-1 F2
        #         fix (search `bn == "__try"`)
        #     C3. Assign-arm consults before mutating
        #         (in the A.Assign arm of _check_expr)
        #
        #   Scope-restore helper:
        #     H1. _check_expr_in_block_scope wraps expression-form
        #         arm bodies (match-arm body, if-else expr branch,
        #         match-guard) that bypass _check_block — gate-5
        #         G4-F1/H2 fix.
        #
        # Lineage: see Stage 46 + Stage 48 closure ledgers in
        # docs/stage{46,48}-progress-2026-05-17.md for the
        # silent-miscompile patterns each rule guards.
        #
        # TODO(stage49): the runtime Ok/Err tag obsoletes most of
        # the Phase-0 static-provenance machinery (the snapshot/
        # restore, the assigns-stack, and the wrong-arm rejections
        # become a debug-only lint at most). Sites 2-6 + C1-C3 +
        # H1 all collapse when site 4's dict goes away.
        #
        # Stage 49 Inc 1.5 + closure gate-2 type-design G2-MH1
        # decision: the runtime tag-check on unwrap_ok / unwrap_err
        # (Inc 1.5, commit db26e1c) is now live and is the SOLE
        # soundness layer — `unwrap_<X>(Y-tagged)` panics at runtime
        # with deterministic TRAP_RESULT_WRONG_UNWRAP. The C1 static-
        # provenance reject at typecheck.py:4595-4625 is KEPT as a
        # defense-in-depth quality-of-life diagnostic: earlier
        # source-line diagnosis is friendlier than waiting for the
        # runtime panic. C2 (__try Err-provenance reject) WAS lifted
        # in Inc 4 (commit 47d8f66) since `?` is a propagator (not
        # an eliminator); the asymmetry between eliminator-side
        # (kept) and propagator-side (lifted) is intentional —
        # propagating an Err is never wrong, extracting from it is.
        # C3 (Assign-arm prov pop) is still useful for the few
        # static-fold paths. The full retirement of the mutation-
        # site stewardship (sites 2-6 collapsing to a flat-dict-
        # per-fn) is deferred to Stage 50+ when more Phase-0
        # surface lifts and the dict has more consumer sites to
        # collapse.
        self._result_constructor_provenance: dict[str, str] = {}
        # Gate-3 G3-F1 fix: parallel stack tracking which names
        # were INTRODUCED-via-let in each open block. Used at
        # _check_block exit to distinguish "inner-shadow let
        # changed the dict" (restore outer) from "inner assign
        # to outer name changed the dict" (drop outer's stale
        # entry — the value is now mutated and dynamic).
        # TODO(stage49): collapses when site-4 prov dict is removed.
        self._result_let_block_scopes: list[set[str]] = []
        # Gate-5 G4-F2 fix: parallel stack tracking which names
        # were ASSIGNED-TO in each open block. Closes the
        # ASSIGN-then-LET-shadow hole that gate-3's let-set
        # alone could not detect (the let-shadow added the name
        # to inner_lets, which then masked the prior assign's
        # mutation at restore — outer's stale 'ok' survived).
        # At restore, any saved name in this set is dropped from
        # the restored map regardless of let-set membership.
        # TODO(stage49): collapses when site-4 prov dict is removed.
        self._result_assigns_block_scopes: list[set[str]] = []
        # Stage 52 Inc 1 — modal-origin taint-tracking dict.
        # Maps: var_name → modal-kind string ('known'|'believed'|'goal'|
        # 'uncertain') when the var was bound to a `from_X(...)` call.
        # Consulted by the F1 cross-modal launder guard at the
        # `into_Y(...)` arm to catch the let-binding bypass:
        #   `let r = from_uncertain(u); into_known(r)`
        # — pre-fix this slipped the syntactic guard because
        # `into_known`'s arg was a Name, not a `Call(from_X, ...)`.
        # Post-fix the guard ALSO consults this map.
        #
        # Closes the Stage 40 closure gate-1 H1 documented limitation
        # ("let-binding bypass — Phase-1 task: taint-tracking pass").
        #
        # Lifecycle (mirrors _result_constructor_provenance):
        #   1. declaration (here)
        #   2. cleared at check() entry
        #   3. cleared at _check_fn entry
        #   4. populated at Let-stmt when value is `Call(from_X, ...)`
        #   5. popped at Let-stmt opaque RHS / Assign-stmt invalidation
        #
        # Stage 52 Inc 1 ships basic dict + per-fn clear. Stage 52
        # closure gate-1 silent-failure HIGH-1/3/5 + type-design F1e
        # forced Inc 2 to ship simultaneously (per gate-1 F8 — the
        # cascading-defect rhythm caught the deferred defects in
        # the same audit they were deferred from).
        #
        # Inc 2 lifts: (a) Assign-arm POPULATE on from_X(...) RHS
        # (HIGH-1+3); (b) block-scope snapshot/restore with let-set
        # parallel (F1e); (c) inner-let shadow vs inner-Assign
        # propagation distinguished via the let-set semantics
        # mirroring Stage 48 gate-3 — INVERTED for modal-origin:
        # inner-Assign to outer name PROPAGATES the new taint
        # (because the AI-safety invariant says any from_X
        # introduction must surface), whereas Result-provenance
        # DROPS on inner-Assign (because the assign invalidates
        # the static Ok/Err claim).
        self._modal_origin_provenance: dict[str, str] = {}
        # Stage 53 Inc 1: map user-defined function names → the modal
        # kind of their declared return type (e.g. 'known', 'uncertain').
        # Read-only after Pass 1 (_register_fn). Populated from sig.ret
        # for any user fn whose return is TyModal. Used by
        # _modal_origin_of_expr to propagate taint through helper-fn
        # calls — closes the LAST modal-launder bypass (helper-fn
        # indirection), which was the Stage 40 H1 "different defect
        # class" deferred from Stage 52.
        self._fn_modal_return_kind: dict[str, str] = {}
        # Stage 52 gate-1 F1e / Inc 2: parallel stack tracking
        # names introduced via let in each open block. Used at
        # block-exit restore to distinguish inner-shadow lets
        # (drop their entry, restore outer's if present) from
        # inner-Assign mutations of outer names (preserve the
        # mutation — taint propagates upward).
        self._modal_origin_let_block_scopes: list[set[str]] = []
        # Stage 52 closure gate-3 NEW-HIGH-2/3/4 fix: parallel
        # stack tracking names ASSIGNED-TO in each open block
        # (regardless of whether the Assign installed taint).
        # Mirrors `_result_assigns_block_scopes` (Stage 48 G4-F2).
        # Used by if-else/match union to detect "branch overwrote
        # the name with a non-modal value" — that should drop
        # the pre-state's taint claim (otherwise the union over-
        # claims taint that the runtime might never carry).
        # Pre-fix, `if cond { r = from_unc(u); } else { r = 7; }`
        # then `into_known(r)` falsely fired because the else
        # arm's `r = 7` didn't appear in observed_kinds at all
        # (only the then-arm's 'uncertain' did).
        self._modal_origin_assigns_block_scopes: list[set[str]] = []
        # Stage 52 closure gate-3 NEW-HIGH-2/3/4 fix: captures the
        # most-recently-popped modal-assigns set from _check_block
        # or _check_expr_in_block_scope. Used by if-else / match
        # union sites to detect "branch reassigned name without
        # installing modal taint" — those names drop from the
        # unioned static claim.
        self._last_modal_assigns_popped: set[str] = set()
        self._seen_unknown_type_names: set[str] = set()
        # Stage 40 closure gate-2 code-review MEDIUM-1 fix (conf
        # 88): explicit init so re-running check() on the same
        # TypeChecker instance (LSP / REPL / test harness reuse)
        # doesn't carry stale shadow names that would suppress
        # builtin dispatch for non-shadowed callsites in the
        # second run. Mirrors the cascade-suppression set
        # discipline at lines 542-550.
        self._shadowed_builtin_names: set[str] = set()
        self._resolving_type_aliases: set[str] = set()
        self._type_alias_cache: dict[str, Type] = {}
        self._const_scalar_values: dict[str, int | float] = {}
        self._unrepresentable_const_scalar_names: set[str] = set()
        self._invalid_const_names: set[str] = set()
        self._invalid_refined_return_functions: set[str] = set()
        self._unrepresentable_scalar_return_functions: set[str] = set()
        self._local_const_scalar_scopes: list[dict[str, int | float | None]] = []
        self._local_const_unrepresentable_scopes: list[set[str]] = []
        self._local_const_unrepresentable_base_scopes: list[dict[str, Type]] = []
        self._current_return_ty: Type = TyUnit()
        self.proof_obligations: list[ProofObligation] = []
        self.proof_carries: list[ProofCarry] = []
        # Audit 28.8 B3: unsafe-context depth counter. Incremented when
        # descending into an A.UnsafeBlock; consulted by the Cast
        # handler so raw-pointer casts (TyPtr targets from non-TyPtr
        # sources) outside any unsafe block emit trap 28603.
        self._in_unsafe_depth: int = 0

    # ---- entry point ----
    def check(self) -> list[TypeError_]:
        # Pass 0: index struct + enum decls *first* so that function
        # signatures referring to a nominal struct/enum resolve to
        # TyStruct/TyEnum (was: pass 1.5, which left struct-typed
        # params as TyUnknown until body-check).
        self._struct_decls: dict[str, A.StructDecl] = {}
        self._enum_decls: dict[str, A.EnumDecl] = {}
        self._type_alias_decls: dict[str, A.TypeAlias] = {}
        self._const_decls: dict[str, A.ConstDecl] = {}
        self._invalid_const_names = set()
        self._invalid_refined_return_functions = set()
        self._unrepresentable_scalar_return_functions = set()
        self._unrepresentable_const_scalar_names = set()
        # Stage 40 closure gate-2 MEDIUM-1: clear shadowed
        # builtin names on each check() so a second invocation
        # starts fresh.
        self._shadowed_builtin_names = set()
        # Stage 46 closure gate-2 G2-F1: clear Result constructor
        # provenance on each check(). Gate-5 G4-M1 parity: also
        # clear the parallel scope stacks so a check()-reuse from
        # an LSP/REPL/test harness can't leak stale frames if a
        # prior _check_block raised between push and pop.
        self._result_constructor_provenance = {}
        self._result_let_block_scopes = []
        self._result_assigns_block_scopes = []
        # Stage 52 Inc 1: clear modal-origin taint map per check().
        self._modal_origin_provenance = {}
        self._modal_origin_let_block_scopes = []
        self._modal_origin_assigns_block_scopes = []
        # Stage 52 closure gate-7 type-design HIGH-1: also clear
        # the covert-return-channel slot (defense-in-depth).
        self._last_modal_assigns_popped = set()
        # Stage 53 Inc 1: parallel clear for re-entrancy safety
        # (LSP/REPL). Repopulated by _register_fn during Pass 1.
        self._fn_modal_return_kind = {}
        self._recursive_enum_names: set[str] = set()
        self._type_alias_cache = {}
        self._local_const_scalar_scopes = []
        self._local_const_unrepresentable_scopes = []
        self._duplicate_type_alias_items: set[int] = set()
        self._duplicate_const_items: set[int] = set()
        self._type_namespace_names: dict[str, str] = {}
        for item in self.prog.items:
            if isinstance(item, A.StructDecl):
                if self._define_type_namespace_name(
                        item.name, "struct", item.span):
                    self._struct_decls[item.name] = item
            elif isinstance(item, A.EnumDecl):
                if self._define_type_namespace_name(
                        item.name, "enum", item.span):
                    self._enum_decls[item.name] = item
            elif isinstance(item, A.TypeAlias):
                if self._define_type_namespace_name(
                        item.name, "type alias", item.span):
                    self._type_alias_decls[item.name] = item
                else:
                    self._duplicate_type_alias_items.add(id(item))
            elif isinstance(item, A.ConstDecl):
                if item.name in self._const_decls:
                    self.errors.append(TypeError_(
                        f"duplicate const {item.name!r}", item.span,
                    ))
                    self._duplicate_const_items.add(id(item))
                else:
                    self._const_decls[item.name] = item

        self._recursive_enum_names = self._compute_recursive_enum_names()
        self._index_const_scalar_values()

        # Validate aliases even if unused. Otherwise bad refined aliases can
        # sit silently until a later edit happens to reference them.
        for item in self.prog.items:
            if isinstance(item, A.TypeAlias):
                if id(item) in self._duplicate_type_alias_items:
                    continue
                self._resolve_type_alias(item, Scope())

        # Pass 1: register function signatures (don't check bodies yet)
        for item in self.prog.items:
            if isinstance(item, A.FnDecl):
                try:
                    self._register_fn(item)
                except TypeError_ as e:
                    self.errors.append(e)

        # Pass 1.5: check top-level constants after function signatures are
        # registered, so a const initializer can still reference earlier
        # compiler-known functions if future Helix allows it.
        for item in self.prog.items:
            if isinstance(item, A.ConstDecl):
                if id(item) in self._duplicate_const_items:
                    continue
                try:
                    self._check_const_decl(item)
                except TypeError_ as e:
                    self.errors.append(e)

        # Pass 2: check function bodies. Refined-return functions that fail
        # must be known before later proof-carry artifacts are trusted, even
        # when callers are declared before the failed producer. Run a small
        # fixed point over function bodies, discarding intermediate
        # function-body diagnostics until the invalid-producer set stabilizes.
        function_error_start = len(self.errors)
        function_obligation_start = len(self.proof_obligations)
        function_carry_start = len(self.proof_carries)
        fn_items = [item for item in self.prog.items
                    if isinstance(item, A.FnDecl)]
        for _ in range(max(1, len(fn_items) + 1)):
            self.errors = self.errors[:function_error_start]
            self.proof_obligations = (
                self.proof_obligations[:function_obligation_start])
            self.proof_carries = self.proof_carries[:function_carry_start]
            self._seen_unbound = set()
            self._seen_unknown_type_names = set()
            invalid_before = set(self._invalid_refined_return_functions)
            scalar_invalid_before = set(
                self._unrepresentable_scalar_return_functions)
            for item in fn_items:
                try:
                    self._check_fn(item)
                except TypeError_ as e:
                    self.errors.append(e)
            if (self._invalid_refined_return_functions == invalid_before
                    and self._unrepresentable_scalar_return_functions
                    == scalar_invalid_before):
                break

        return self.errors

    def _define_type_namespace_name(
        self, name: str, kind: str, span: A.Span,
    ) -> bool:
        existing = self._type_namespace_names.get(name)
        if existing is not None:
            self.errors.append(TypeError_(
                f"duplicate type namespace name {name!r}: "
                f"{kind} conflicts with earlier {existing}",
                span,
            ))
            return False
        self._type_namespace_names[name] = kind
        return True

    def _index_const_scalar_values(self) -> None:
        consts = [
            item for item in self.prog.items
            if (isinstance(item, A.ConstDecl)
                and id(item) not in self._duplicate_const_items)
        ]
        for _ in range(len(consts)):
            progressed = False
            for decl in consts:
                if (decl.name in self._const_scalar_values
                        or decl.name in self._unrepresentable_const_scalar_names):
                    continue
                target = self._const_index_target_type(decl)
                if target is None:
                    continue
                if self._expr_has_unrepresentable_typed_const_scalar(
                        decl.value):
                    self._unrepresentable_const_scalar_names.add(decl.name)
                    progressed = True
                    continue
                value = self._eval_const_scalar_expr(
                    decl.value, None, honor_float_suffix=True,
                    numeric_base=target)
                represented = self._cast_const_scalar_to_type(value, target)
                if (isinstance(represented, (int, float))
                        and not isinstance(represented, bool)):
                    self._const_scalar_values[decl.name] = represented
                    progressed = True
            if not progressed:
                break

    def _const_index_target_type(self, decl: A.ConstDecl) -> Type | None:
        errors_before = len(self.errors)
        seen_unknown_before = set(self._seen_unknown_type_names)
        alias_cache_before = dict(self._type_alias_cache)
        resolving_aliases_before = set(self._resolving_type_aliases)
        try:
            declared = self._resolve_type(decl.ty, Scope())
        finally:
            self.errors = self.errors[:errors_before]
            self._seen_unknown_type_names = seen_unknown_before
            self._type_alias_cache = alias_cache_before
            self._resolving_type_aliases = resolving_aliases_before
        target = self._erase_refinement(declared)
        if isinstance(target, TyPrim) and (
                target.name in _INT_PRIM_NAMES or target.name in _FLOAT_PRIM_NAMES):
            return target
        return None

    def _push_local_const_scope(self) -> None:
        self._local_const_scalar_scopes.append({})
        self._local_const_unrepresentable_scopes.append(set())
        self._local_const_unrepresentable_base_scopes.append({})

    def _pop_local_const_scope(self) -> None:
        self._local_const_scalar_scopes.pop()
        self._local_const_unrepresentable_scopes.pop()
        self._local_const_unrepresentable_base_scopes.pop()

    def _define_local_const_scalar(
        self, name: str, value: int | float | None,
    ) -> None:
        if self._local_const_scalar_scopes:
            self._local_const_scalar_scopes[-1][name] = value

    def _mark_local_const_unrepresentable(
        self, name: str, base: Type | None = None,
    ) -> None:
        if self._local_const_unrepresentable_scopes:
            self._local_const_unrepresentable_scopes[-1].add(name)
            if base is not None:
                self._local_const_unrepresentable_base_scopes[-1][name] = base

    def _set_local_const_unrepresentable(
        self,
        name: str,
        unrepresentable: bool,
        base: Type | None = None,
        *,
        anchor_name: str | None = None,
    ) -> None:
        scope_name = anchor_name or name
        for idx in range(len(self._local_const_scalar_scopes) - 1, -1, -1):
            if scope_name not in self._local_const_scalar_scopes[idx]:
                continue
            self._local_const_scalar_scopes[idx].setdefault(name, None)
            if unrepresentable:
                self._local_const_unrepresentable_scopes[idx].add(name)
                if base is not None:
                    self._local_const_unrepresentable_base_scopes[idx][name] = (
                        base)
            else:
                self._local_const_unrepresentable_scopes[idx].discard(name)
                self._local_const_unrepresentable_base_scopes[idx].pop(
                    name, None)
            return
        if self._local_const_scalar_scopes:
            self._local_const_scalar_scopes[-1][name] = None
            if unrepresentable:
                self._local_const_unrepresentable_scopes[-1].add(name)
                if base is not None:
                    self._local_const_unrepresentable_base_scopes[-1][name] = (
                        base)

    def _local_const_index_key(self, name: str, index: int) -> str:
        return f"<index:{name}:{index}>"

    def _simple_local_const_index_key(
        self, expr: A.Index,
    ) -> tuple[str, str] | None:
        if (not isinstance(expr.callee, A.Name)
                or expr.callee.generics
                or len(expr.indices) != 1):
            return None
        index = expr.indices[0]
        if not isinstance(index, A.IntLit):
            return None
        return (
            expr.callee.name,
            self._local_const_index_key(expr.callee.name, index.value),
        )

    def _clear_local_const_index_unrepresentable(
        self, name: str,
    ) -> None:
        prefix = f"<index:{name}:"
        for idx in range(len(self._local_const_scalar_scopes) - 1, -1, -1):
            if name not in self._local_const_scalar_scopes[idx]:
                continue
            keys = [
                key for key in self._local_const_scalar_scopes[idx]
                if key.startswith(prefix)
            ]
            for key in keys:
                self._local_const_scalar_scopes[idx].pop(key, None)
                self._local_const_unrepresentable_scopes[idx].discard(key)
                self._local_const_unrepresentable_base_scopes[idx].pop(
                    key, None)
            return

    def _mark_array_literal_unrepresentable_elements(
        self, name: str, value: A.Expr,
    ) -> bool:
        if not isinstance(value, A.ArrayLit):
            return False
        marked = False
        for idx, elem in enumerate(value.elems):
            if not self._expr_has_unrepresentable_typed_const_scalar(elem):
                continue
            self._set_local_const_unrepresentable(
                self._local_const_index_key(name, idx),
                True,
                self._expr_unrepresentable_typed_const_scalar_base(elem),
                anchor_name=name,
            )
            marked = True
        return marked

    def _lookup_local_const_scalar(
        self, name: str,
    ) -> tuple[bool, int | float | None]:
        for scope in reversed(self._local_const_scalar_scopes):
            if name in scope:
                return True, scope[name]
        return False, None

    def _lookup_local_const_unrepresentable(
        self, name: str,
    ) -> tuple[bool, bool]:
        for idx in range(len(self._local_const_scalar_scopes) - 1, -1, -1):
            if name in self._local_const_scalar_scopes[idx]:
                return (
                    True,
                    name in self._local_const_unrepresentable_scopes[idx],
                )
        return False, False

    def _lookup_local_const_unrepresentable_base(
        self, name: str,
    ) -> Type | None:
        for idx in range(len(self._local_const_scalar_scopes) - 1, -1, -1):
            if name in self._local_const_scalar_scopes[idx]:
                return self._local_const_unrepresentable_base_scopes[idx].get(
                    name)
        return None

    # ---- registration ----
    def _check_const_decl(self, decl: A.ConstDecl) -> None:
        scope = Scope()
        error_start = len(self.errors)
        declared = self._resolve_type(decl.ty, scope)
        value_ty = self._check_expr(decl.value, scope)
        if not self._compatible(value_ty, declared):
            self.errors.append(TypeError_(
                f"const {decl.name!r}: declared {self._fmt(declared)} "
                f"but value is {self._fmt(value_ty)}",
                decl.span,
            ))
        elif len(self.errors) == error_start:
            self._check_refinement_contextual_value(
                decl.value, value_ty, declared, decl.span,
                f"const {decl.name!r}",
                scope,
            )
        if (self._contains_refinement(declared)
                and len(self.errors) != error_start):
            self._invalid_const_names.add(decl.name)
            self._const_scalar_values.pop(decl.name, None)
            return

    def _register_fn(self, fn: A.FnDecl) -> None:
        # Stage 40 closure gate-1 silent-failure F2 fix (MEDIUM
        # conf 90): refuse to silently shadow a reserved builtin
        # name. Pre-fix, `fn confirm(x: i32) -> i32 { x * 2 }` was
        # silently dead-coded — typecheck dispatched the builtin
        # arm before the user-fn lookup, so the user got
        # "confirm() requires Believed<T>, got i32" with no
        # shadowing hint. Stage 39 F3 (deferred at gate-1) becomes
        # urgent at Stage 40 because `confirm` and `act_on` are
        # extremely-generic names likely to collide with user
        # planning / state-machine code. Closes Stage 36-40 holes
        # in one arm; same fail-closed discipline as the other
        # gate-1 / gate-2 fixes.
        if fn.name in self._BUILTIN_NAMES:
            self.errors.append(TypeError_(
                f"function {fn.name!r} shadows a reserved builtin "
                f"name; rename the function to avoid silent "
                f"dispatch dead-coding (the typechecker resolves "
                f"the builtin first, so the user definition is "
                f"unreachable from any call site that uses the "
                f"bare name)",
                fn.span,
                hint=f"reserved builtins include modal/temporal/"
                f"frame/tier intro+elim+transition verbs (e.g. "
                f"into_*, from_*, confirm, act_on, forecast, "
                f"world_to_robot), Result accessors (Ok, Err, "
                f"unwrap_ok, etc.), and reserved internal "
                f"builtins with a double-underscore prefix "
                f"(e.g. __try, __arena_push); pick a different "
                f"name",
            ))
            # Stage 40 closure gate-2 H2 fix (HIGH conf 92):
            # the "diagnostic alone gates the typecheck pass"
            # claim in the original F2 comment was empirically
            # false — call sites still hit the builtin arm,
            # producing N additional false errors that
            # misrepresent the bug. Track shadowed names so the
            # call-dispatch path skips the builtin arms and falls
            # through to user-fn lookup. The fn-decl shadow
            # error is the ONLY one the user sees. The set is
            # initialized in __init__ + cleared in check() per
            # the cascade-suppression-set discipline at lines
            # 542-560 (gate-2 MEDIUM-1 re-entrancy fix).
            self._shadowed_builtin_names.add(fn.name)
            # Continue registration so downstream code doesn't
            # crash on a missing FunctionSig.
        # Build the generic-bindings scope
        gen_scope = Scope()
        for g in fn.generics:
            if g.kind == "size":
                gen_scope.define(g.name, TySize(g.name))
            elif g.kind == "type":
                gen_scope.define(g.name, TyVar(g.name))
            elif g.kind == "device":
                gen_scope.define(g.name, TyVar(g.name))
            else:
                gen_scope.define(g.name, TyVar(g.name))

        # Resolve param types
        params: list[tuple[str, Type]] = []
        for p in fn.params:
            t = self._resolve_type(p.ty, gen_scope)
            if fn.is_extern and self._contains_refinement(t):
                self.errors.append(TypeError_(
                    f"extern function {fn.name!r}: parameter {p.name!r} "
                    f"type {self._fmt(t)} cannot use refined types in "
                    f"Stage 31",
                    p.span,
                    hint="use raw FFI types and validate at a Helix boundary",
                ))
            params.append((p.name, t))

        # Resolve return type
        if fn.return_ty is not None:
            ret = self._resolve_type(fn.return_ty, gen_scope)
        else:
            ret = TyUnit()
        if self._is_unsupported_aggregate_return_type(ret):
            self.errors.append(TypeError_(
                f"function {fn.name!r}: aggregate return type "
                f"{self._fmt(ret)} is not supported by the Stage 31 "
                f"backend ABI",
                fn.return_ty.span if fn.return_ty is not None else fn.span,
                hint="return a scalar handle or pass an output aggregate "
                     "parameter until aggregate return ABI support lands",
            ))
        if fn.is_extern and self._contains_refinement(ret):
            self.errors.append(TypeError_(
                f"extern function {fn.name!r}: return type "
                f"{self._fmt(ret)} cannot use refined types in Stage 31",
                fn.return_ty.span if fn.return_ty is not None else fn.span,
                hint="return a raw FFI type and validate it at a Helix "
                     "boundary",
            ))

        # Record constraints (not yet solved)
        for w in fn.where_clauses:
            self.constraints.append(w.constraint)

        # Effect/capability inference from attributes
        is_pure = "pure" in fn.attrs
        effects: set[str] = set()
        for a in fn.attrs:
            # accept "effect" attribute that mentions a list of capabilities
            if a == "effect":
                effects.add("unknown_effect")
            elif a.startswith("effect:"):
                effects.add(a[len("effect:"):])
            elif a in ("io", "network", "modify_self", "rng", "time", "fs"):
                effects.add(a)
        if is_pure and effects:
            self.errors.append(TypeError_(
                f"function {fn.name!r}: cannot be both @pure and have @effect(...)",
                fn.span,
            ))

        sig = FunctionSig(
            name=fn.name, generics=fn.generics, params=params, ret=ret,
            is_pure=is_pure, effects=frozenset(effects),
        )
        if fn.name in self.functions:
            raise TypeError_(f"duplicate function {fn.name!r}", fn.span)
        self.functions[fn.name] = sig
        # Stage 53 Inc 1: if the declared return type is a modal
        # wrapper, record the kind so _modal_origin_of_expr can
        # propagate taint through user-defined helper functions
        # (closes the helper-fn indirection laundering vector).
        # Read-only after Pass 1; no per-fn clear needed.
        if isinstance(sig.ret, TyModal):
            self._fn_modal_return_kind[fn.name] = sig.ret.kind

    def _compute_recursive_enum_names(self) -> set[str]:
        recursive: set[str] = set()

        def refs_enum(
            ty: A.TyNode, target: str, visiting: frozenset[str],
            seen_aliases: frozenset[str],
        ) -> bool:
            if isinstance(ty, A.TyName):
                if ty.name == target:
                    return True
                alias = self._type_alias_decls.get(ty.name)
                if alias is not None and alias.name not in seen_aliases:
                    return refs_enum(
                        alias.target, target, visiting,
                        seen_aliases | {alias.name})
                enum_decl = self._enum_decls.get(ty.name)
                if enum_decl is not None and ty.name not in visiting:
                    return any(
                        refs_enum(payload_ty, target, visiting | {ty.name},
                                  seen_aliases)
                        for variant in enum_decl.variants
                        for payload_ty in variant.payload_tys
                    )
                return False
            if isinstance(ty, A.TyGeneric):
                return any(refs_enum(arg, target, visiting, seen_aliases)
                           for arg in ty.args)
            if isinstance(ty, A.TyTuple):
                return any(refs_enum(elem, target, visiting, seen_aliases)
                           for elem in ty.elems)
            if isinstance(ty, A.TyArray):
                return refs_enum(ty.elem, target, visiting, seen_aliases)
            if isinstance(ty, A.TyRef):
                return refs_enum(ty.inner, target, visiting, seen_aliases)
            if isinstance(ty, A.TyPtr):
                return refs_enum(ty.inner, target, visiting, seen_aliases)
            if isinstance(ty, A.TyFn):
                return (any(refs_enum(param, target, visiting, seen_aliases)
                            for param in ty.params)
                        or refs_enum(ty.ret, target, visiting, seen_aliases))
            if isinstance(ty, A.TyTensor):
                return refs_enum(ty.dtype, target, visiting, seen_aliases)
            if isinstance(ty, A.TyTile):
                return refs_enum(ty.dtype, target, visiting, seen_aliases)
            return False

        for name, decl in self._enum_decls.items():
            for variant in decl.variants:
                if any(refs_enum(payload_ty, name, frozenset({name}),
                                 frozenset())
                       for payload_ty in variant.payload_tys):
                    recursive.add(name)
                    break
        return recursive

    def _is_unsupported_aggregate_return_type(self, ty: Type) -> bool:
        if isinstance(ty, TyStruct):
            return True
        if isinstance(ty, TyEnum):
            return ty.name not in self._recursive_enum_names
        if isinstance(ty, TyTuple):
            return True
        if isinstance(ty, TyArray):
            return True
        return False

    # ---- type resolution ----
    def _resolve_type(self, ty: A.TyNode, scope: Scope) -> Type:
        if isinstance(ty, A.TyName):
            # Stage 28.9 cycle-105 F105-1 fix: normalize textual "()" to
            # TyUnit() so source-typed unit and implicit-unit converge to
            # a single representation. See PRIMITIVES comment for rationale.
            if ty.name == "()":
                return TyUnit()
            if ty.name in PRIMITIVES:
                return TyPrim(ty.name)
            looked = scope.lookup(ty.name)
            if looked is not None:
                return looked
            alias = getattr(self, "_type_alias_decls", {}).get(ty.name)
            if alias is not None:
                return self._resolve_type_alias(alias, scope)
            # Recognise nominal struct types so field-access on a struct-
            # typed field (e.g. nested struct: `inner: Inner`) gets a real
            # TyStruct instead of falling all the way to TyUnknown — that
            # was breaking chained field-type tracking and causing every
            # struct-field-typecheck to trivially pass.
            if ty.name in getattr(self, "_struct_decls", {}):
                return TyStruct(name=ty.name)
            if ty.name in getattr(self, "_enum_decls", {}):
                return TyEnum(name=ty.name)
            if ty.name not in self._seen_unknown_type_names:
                self._seen_unknown_type_names.add(ty.name)
                self.errors.append(TypeError_(
                    f"unknown type {ty.name!r}",
                    ty.span,
                    hint="declare this type or import it before use",
                ))
            return TyUnknown(hint=f"unknown name {ty.name}")
        if isinstance(ty, A.TyTuple):
            return TyTuple(tuple(self._resolve_type(e, scope) for e in ty.elems))
        if isinstance(ty, A.TyArray):
            elem = self._resolve_type(ty.elem, scope)
            size = self._resolve_size_expr(ty.size, scope)
            return TyArray(elem, size)
        if isinstance(ty, A.TyRef):
            return TyRef(self._resolve_type(ty.inner, scope), ty.is_mut)
        if isinstance(ty, A.TyPtr):
            # Stage 16.5: pointer types resolve to TyPtr (u64 at ABI level).
            return TyPtr(self._resolve_type(ty.inner, scope), ty.is_mut)
        if isinstance(ty, A.TyFn):
            return TyFn(
                tuple(self._resolve_type(p, scope) for p in ty.params),
                self._resolve_type(ty.ret, scope),
            )
        if isinstance(ty, A.TyTensor):
            dtype = self._resolve_type(ty.dtype, scope)
            shape = tuple(self._resolve_size_expr(s, scope) for s in ty.shape)
            device = self._stringify_marker(ty.device, scope) if ty.device else None
            layout = self._stringify_marker(ty.layout, scope) if ty.layout else None
            return TyTensor(dtype, shape, device, layout)
        if isinstance(ty, A.TyTile):
            dtype = self._resolve_type(ty.dtype, scope)
            shape = tuple(self._resolve_size_expr(s, scope) for s in ty.shape)
            memspace = self._stringify_marker(ty.memspace, scope) or "?"
            return TyTile(dtype, shape, memspace)
        if isinstance(ty, A.TyGeneric):
            # Differentiable wrapper: D<T>
            if ty.base == "D" and len(ty.args) == 1:
                return TyDiff(inner=self._resolve_type(ty.args[0], scope))
            # Stage 24: relational/logical wrapper: Logic<T>
            if ty.base == "Logic" and len(ty.args) == 1:
                return TyLogic(inner=self._resolve_type(ty.args[0], scope))
            # Audit 28.8 cycle 2 B:C3: reflection wrapper Quote<T>.
            # Pre-fix, `fn unbox(q: Quote<i32>)` resolved to TyUnknown
            # (no arm here), which accepted any value through the
            # parameter typecheck. The TyQuote variant existed but
            # only on the expression-typing side (Quote handler at
            # line ~1492). Now `Quote<T>` resolves to TyQuote(inner=T)
            # and is enforced by `_compatible`.
            if ty.base == "Quote" and len(ty.args) == 1:
                return TyQuote(inner=self._resolve_type(ty.args[0], scope))
            # Memory-tier wrappers: WorkingMem<T>, EpisodicMem<T>, etc.
            # Stage 43 Inc 1 F5 fix: explicit arity diagnostic so
            # `WorkingMem<>` / `WorkingMem<i32, i32>` emit "X<T>
            # takes 1 type argument, got N" instead of the
            # misleading "unknown type 'WorkingMem'" fall-through.
            tier_map = {
                "WorkingMem": "working",
                "EpisodicMem": "episodic",
                "SemanticMem": "semantic",
                "ProceduralMem": "procedural",
            }
            if ty.base in tier_map:
                if len(ty.args) != 1:
                    self.errors.append(TypeError_(
                        f"{ty.base}<T> takes 1 type argument, "
                        f"got {len(ty.args)}",
                        ty.span,
                    ))
                    return TyUnknown(hint=ty.base)
                return TyMemTier(tier=tier_map[ty.base],
                                 inner=self._resolve_type(ty.args[0], scope))
            # Stage 38 Inc 1 — spatial-frame wrappers. F5 arity arm.
            frame_map = {
                "WorldFrame": "world",
                "RobotFrame": "robot",
                "CameraFrame": "camera",
            }
            if ty.base in frame_map:
                if len(ty.args) != 1:
                    self.errors.append(TypeError_(
                        f"{ty.base}<T> takes 1 type argument, "
                        f"got {len(ty.args)}",
                        ty.span,
                    ))
                    return TyUnknown(hint=ty.base)
                return TyFrame(frame=frame_map[ty.base],
                               inner=self._resolve_type(ty.args[0], scope))
            # Stage 39 Inc 1 — temporal wrappers. F5 arity arm.
            temporal_map = {
                "Past": "past",
                "Present": "present",
                "Future": "future",
                "Eternal": "eternal",
            }
            if ty.base in temporal_map:
                if len(ty.args) != 1:
                    self.errors.append(TypeError_(
                        f"{ty.base}<T> takes 1 type argument, "
                        f"got {len(ty.args)}",
                        ty.span,
                    ))
                    return TyUnknown(hint=ty.base)
                return TyTemporal(kind=temporal_map[ty.base],
                                  inner=self._resolve_type(ty.args[0], scope))
            # Stage 40 Inc 1 — modal wrappers. F5 arity arm.
            modal_map = {
                "Known":     "known",
                "Believed":  "believed",
                "Goal":      "goal",
                "Uncertain": "uncertain",
            }
            if ty.base in modal_map:
                if len(ty.args) != 1:
                    self.errors.append(TypeError_(
                        f"{ty.base}<T> takes 1 type argument, "
                        f"got {len(ty.args)}",
                        ty.span,
                    ))
                    return TyUnknown(hint=ty.base)
                return TyModal(kind=modal_map[ty.base],
                               inner=self._resolve_type(ty.args[0], scope))
            # Stage 41 Inc 1 — causal wrappers. F5 arity arm.
            causal_map = {
                "Cause":       "cause",
                "Effect":      "effect",
                "Joint":       "joint",
                "Independent": "independent",
            }
            if ty.base in causal_map:
                if len(ty.args) != 1:
                    self.errors.append(TypeError_(
                        f"{ty.base}<T> takes 1 type argument, "
                        f"got {len(ty.args)}",
                        ty.span,
                    ))
                    return TyUnknown(hint=ty.base)
                return TyCausal(kind=causal_map[ty.base],
                                inner=self._resolve_type(ty.args[0], scope))
            # Stage 46 Inc 1 — Result<T, E>: first two-parameter
            # wrapper family. Tier 4 #14 ROADMAP item.
            if ty.base == "Result":
                if len(ty.args) != 2:
                    self.errors.append(TypeError_(
                        f"Result<T, E> takes 2 type arguments, "
                        f"got {len(ty.args)}",
                        ty.span,
                    ))
                    return TyUnknown(hint=ty.base)
                ok_ty = self._resolve_type(ty.args[0], scope)
                err_ty = self._resolve_type(ty.args[1], scope)
                # Stage 49 Inc 1 LIFT (commit a08f21a):
                # _lower_type's Result-arm no longer recurses into
                # Ok/Err inner types — it short-circuits to
                # TIRScalar("i64") for the packed-tag representation.
                # The Stage 48 G4-H1 asymmetry (Result<Known<...>, E>
                # in fn-return-type position raised NotImplementedError
                # at IR lowering) is therefore CLOSED at the IR layer;
                # the pin test
                # `test_stage48_closure_gate5_g4h1_..._lowers_clean_post_stage49_inc1`
                # asserts typecheck-clean + IR-lowered. Stage 49
                # closure gate-2 G2-H1 added a payload-width reject
                # (i32 only for Stage 49 — see _reject_non_i32_result_payload).
                # Result<wrapper<i32>, i32> still works because the
                # wrapper strips to i32 at the construction-site
                # check (Phase-0 identity-lowered wrappers).
                return TyResult(ok_ty=ok_ty, err_ty=err_ty)
            # Stage 28 — user-defined parametric struct (Audit 28.8 A3/B1).
            # If `ty.base` is a known generic struct AND the arity matches,
            # resolve `Pt<i32>` -> `TyStruct("Pt__i32")` so distinct
            # instantiations are non-unifiable at typecheck time. The
            # mangled name is shared with `struct_mono.mangle_struct`, so
            # the post-mono StructDecl for `Pt__i32` (added by
            # `monomorphize_structs`) resolves field-lookup via the same
            # `self._struct_decls` table.
            user_struct = getattr(self, "_struct_decls", {}).get(ty.base)
            if (user_struct is not None
                    and len(ty.args) == len(user_struct.generics)):
                resolved_args = [self._resolve_type(arg, scope)
                                 for arg in ty.args]
                if any(self._contains_unknown_type(arg_ty)
                       for arg_ty in resolved_args):
                    return TyUnknown(hint=f"generic {ty.base}")
                from .struct_mono import mangle_struct
                return TyStruct(name=mangle_struct(ty.base, ty.args))
            # User type with generic args, arity mismatch or unknown — v0.1
            # falls back to TyUnknown (existing behaviour preserved so
            # non-struct generic types don't regress).
            for arg in ty.args:
                self._resolve_type(arg, scope)
            if user_struct is not None:
                self.errors.append(TypeError_(
                    f"generic type {ty.base!r} expects "
                    f"{len(user_struct.generics)} arg(s), got "
                    f"{len(ty.args)}",
                    ty.span,
                ))
            elif ty.base in getattr(self, "_type_alias_decls", {}):
                self.errors.append(TypeError_(
                    f"type alias {ty.base!r} cannot be used with generic "
                    f"arguments in Stage 31",
                    ty.span,
                ))
            elif ty.base in PRIMITIVES:
                self.errors.append(TypeError_(
                    f"type {ty.base!r} is not generic",
                    ty.span,
                ))
            elif ty.base not in self._seen_unknown_type_names:
                self._seen_unknown_type_names.add(ty.base)
                self.errors.append(TypeError_(
                    f"unknown generic type {ty.base!r}",
                    ty.span,
                    hint="declare this generic type or import it before use",
                ))
            return TyUnknown(hint=f"generic {ty.base}")
        return TyUnknown(hint=f"unknown ty node {type(ty).__name__}")

    def _resolve_type_alias(self, alias: A.TypeAlias, scope: Scope) -> Type:
        cached = self._type_alias_cache.get(alias.name)
        if cached is not None:
            return cached
        if alias.generics:
            unknown = TyUnknown(hint=f"generic type alias {alias.name}")
            self.errors.append(TypeError_(
                f"type alias {alias.name!r}: generic aliases are not "
                f"supported in Stage 31",
                alias.span,
            ))
            self._type_alias_cache[alias.name] = unknown
            return unknown
        if alias.name in self._resolving_type_aliases:
            unknown = TyUnknown(hint=f"recursive alias {alias.name}")
            self.errors.append(TypeError_(
                f"type alias {alias.name!r} is recursive", alias.span,
            ))
            self._type_alias_cache[alias.name] = unknown
            return unknown
        self._resolving_type_aliases.add(alias.name)
        try:
            # Alias targets resolve at declaration/global scope, not at use
            # sites, so `type Alias = T` cannot capture a function generic T.
            base = self._resolve_type(alias.target, Scope())
        finally:
            self._resolving_type_aliases.discard(alias.name)
        if self._contains_unknown_type(base):
            self.errors.append(TypeError_(
                f"type alias {alias.name!r}: target type could not be "
                f"resolved ({self._fmt(base)})",
                alias.span,
            ))
            self._type_alias_cache[alias.name] = base
            return base
        if alias.where_clauses:
            self._validate_refinement_predicates(alias, base)
            resolved = TyRefined(
                name=alias.name,
                base=base,
                predicates=tuple(w.constraint for w in alias.where_clauses),
            )
            self._type_alias_cache[alias.name] = resolved
            return resolved
        self._type_alias_cache[alias.name] = base
        return base

    def _validate_refinement_predicates(
        self, alias: A.TypeAlias, base: Type,
    ) -> None:
        if not self._is_numeric_refinement_base(base):
            self.errors.append(TypeError_(
                f"type alias {alias.name!r}: refinement predicates in "
                f"Stage 31 require a numeric scalar base type, got "
                f"{self._fmt(base)}",
                alias.span,
                hint="refine integer or float aliases in this stage; "
                     "structured and boolean refinements need later proof "
                     "support",
            ))
            return
        for w in alias.where_clauses:
            if not self._refinement_predicate_shape_supported(w.constraint):
                self.errors.append(TypeError_(
                    f"type alias {alias.name!r}: refinement predicate "
                    f"{self._fmt_refinement_expr(w.constraint)} is not "
                    f"supported by Stage 31",
                    w.span,
                    hint="use a boolean constant, a comparison chain over "
                         "`self`, negate a supported predicate with `!`, "
                         "or combine supported predicates with `&&` / `||`",
                ))

    def _is_numeric_refinement_base(self, ty: Type) -> bool:
        if isinstance(ty, TyRefined):
            return self._is_numeric_refinement_base(ty.base)
        return isinstance(ty, TyPrim) and ty.name in (
            "i8", "i16", "i32", "i64", "isize",
            "u8", "u16", "u32", "u64", "usize",
            "bf16", "f16", "f32", "f64",
        )

    def _refinement_predicate_shape_supported(self, expr: A.Expr) -> bool:
        if isinstance(expr, A.BoolLit):
            return True
        if isinstance(expr, A.Unary) and expr.op == "!":
            return self._refinement_predicate_shape_supported(expr.operand)
        if isinstance(expr, A.Binary) and expr.op in ("&&", "||"):
            return (self._refinement_predicate_shape_supported(expr.left)
                    and self._refinement_predicate_shape_supported(expr.right))
        chain = self._flatten_relational_chain(expr)
        if chain is None:
            return False
        _ops, operands = chain
        operands_supported = all(
            self._refinement_scalar_expr_supported(op) for op in operands
        )
        if not operands_supported:
            return False
        if any(self._expr_mentions_self(op) for op in operands):
            return True
        return self._eval_refinement_predicate(expr, None) is not None

    def _refinement_scalar_expr_supported(self, expr: A.Expr) -> bool:
        if isinstance(expr, (A.IntLit, A.FloatLit)):
            return True
        if isinstance(expr, A.Name):
            if expr.generics:
                return False
            if expr.name == "self":
                return True
            const_value = self._const_scalar_values.get(expr.name)
            return (isinstance(const_value, (int, float))
                    and not isinstance(const_value, bool))
        if isinstance(expr, A.Unary) and expr.op == "-":
            return self._refinement_scalar_expr_supported(expr.operand)
        if isinstance(expr, A.Binary) and expr.op in ("+", "-", "*", "/", "%"):
            return (self._refinement_scalar_expr_supported(expr.left)
                    and self._refinement_scalar_expr_supported(expr.right))
        return False

    def _expr_mentions_self(self, expr: A.Expr) -> bool:
        if isinstance(expr, A.Name):
            return expr.name == "self" and not expr.generics
        if isinstance(expr, A.Unary):
            return self._expr_mentions_self(expr.operand)
        if isinstance(expr, A.Binary):
            return (self._expr_mentions_self(expr.left)
                    or self._expr_mentions_self(expr.right))
        return False

    def _enum_variant_for_expr(
        self, expr: A.Expr,
    ) -> Optional[tuple[str, A.EnumVariant]]:
        """Resolve `Enum::Variant` or flattened `mod__Enum__Variant`."""
        ename: Optional[str] = None
        vname: Optional[str] = None
        if isinstance(expr, A.Path) and len(expr.segments) == 2:
            ename, vname = expr.segments
        elif isinstance(expr, A.Name) and "__" in expr.name:
            parts = expr.name.split("__")
            if len(parts) >= 2:
                for i in range(len(parts) - 1, 0, -1):
                    candidate = "__".join(parts[:i])
                    if candidate in getattr(self, "_enum_decls", {}):
                        ename = candidate
                        vname = "__".join(parts[i:])
                        break
        if ename is None or vname is None:
            return None
        edecl = getattr(self, "_enum_decls", {}).get(ename)
        if edecl is None:
            return None
        for variant in edecl.variants:
            if variant.name == vname:
                return ename, variant
        return None

    def _resolve_size_expr(self, expr: A.Expr, scope: Scope) -> Type:
        """A size-expression is either a literal int, a name (size param), or
        an arithmetic expression. v0.1 represents complex exprs as TyUnknown
        with the source preserved by reference (not copied here).

        Audit 28.8 cycle 3 D3: literal size of 0 or negative now emits
        a typecheck error (trap 28802). Pre-fix `[T; -6]` or `[T; 0]`
        from a Binary fold flowed through as `TyPrim('size_-6')` /
        `TyPrim('size_0')`, and lower-ast.py silently used 0 as the
        length — a confusing zero-byte buffer.

        Audit 28.8 cycle 3 D5: `Unary(-, IntLit)` resolved as a single
        signed size literal so the validation above catches it. Plus a
        catch for source-level `[T; -5]` (parser may accept it).
        """
        if isinstance(expr, A.IntLit):
            if expr.value < 0:
                self.errors.append(TypeError_(
                    f"array size must be > 0, got {expr.value} "
                    f"(trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})",
                    expr.span,
                ))
            elif expr.value == 0:
                self.errors.append(TypeError_(
                    f"array size must be > 0, got 0 (trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})",
                    expr.span,
                ))
            return TyPrim(f"size_{expr.value}")
        if isinstance(expr, A.Unary) and expr.op == "-" \
                and isinstance(expr.operand, A.IntLit):
            # Source-level `[T; -N]` parses as Unary(-, IntLit(N)).
            v = -expr.operand.value
            if v < 0:
                self.errors.append(TypeError_(
                    f"array size must be > 0, got {v} (trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})",
                    expr.span,
                ))
            elif v == 0:
                self.errors.append(TypeError_(
                    f"array size must be > 0, got 0 (trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})",
                    expr.span,
                ))
            return TyPrim(f"size_{v}")
        if isinstance(expr, A.Name):
            found_local, local_const_value = self._lookup_local_const_scalar(
                expr.name)
            if found_local:
                if (isinstance(local_const_value, int)
                        and not isinstance(local_const_value, bool)):
                    if local_const_value <= 0:
                        self.errors.append(TypeError_(
                            f"array size must be > 0, got {local_const_value} "
                            f"(trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})",
                            expr.span,
                        ))
                    return TyPrim(f"size_{local_const_value}")
            looked = scope.lookup(expr.name)
            if looked is not None:
                return looked
            const_value = self._const_scalar_values.get(expr.name)
            if (isinstance(const_value, int)
                    and not isinstance(const_value, bool)):
                if const_value <= 0:
                    self.errors.append(TypeError_(
                        f"array size must be > 0, got {const_value} "
                        f"(trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})",
                        expr.span,
                    ))
                return TyPrim(f"size_{const_value}")
            return TyUnknown(hint=f"unbound size {expr.name}")
        if isinstance(expr, A.Binary) and expr.op in ("+", "-", "*", "/", "%"):
            # Symbolically compose; record as constraint material
            return TyUnknown(hint=f"size expr {expr.op}")
        return TyUnknown(hint=f"size expr {type(expr).__name__}")

    # ------------------------------------------------------------------
    # Convert a resolved size type to a Presburger LinExpr, for solver use.
    # Returns None if the type can't be represented (e.g., dynamic shapes).
    # ------------------------------------------------------------------
    def _size_type_to_lin(self, t: Type) -> Optional[P.LinExpr]:
        if isinstance(t, TySize):
            return P.var(t.name)
        if isinstance(t, TyPrim) and t.name.startswith("size_"):
            try:
                n = int(t.name[len("size_"):])
                return P.lit(n)
            except ValueError:
                return None
        if isinstance(t, TyVar):
            return P.var(t.name)
        return None

    def _size_expr_to_lin(self, expr: A.Expr, scope: Scope) -> Optional[P.LinExpr]:
        """Convert a size expression (in AST form) to a Presburger LinExpr,
        looking up generic parameters via scope."""
        if isinstance(expr, A.IntLit):
            return P.lit(expr.value)
        if isinstance(expr, A.Name):
            found_local, local_const_value = self._lookup_local_const_scalar(
                expr.name)
            if found_local and isinstance(local_const_value, int) \
                    and not isinstance(local_const_value, bool):
                return P.lit(local_const_value)
            looked = scope.lookup(expr.name)
            if looked is not None:
                return self._size_type_to_lin(looked)
            return P.var(expr.name)  # treat unbound as fresh var
        if isinstance(expr, A.Binary):
            l = self._size_expr_to_lin(expr.left, scope)
            r = self._size_expr_to_lin(expr.right, scope)
            if l is None or r is None:
                return None
            if expr.op == "+":
                return l + r
            if expr.op == "-":
                return l - r
            if expr.op == "*":
                # Linear-only: one side must be a constant
                if l.is_const():
                    return r * l.const
                if r.is_const():
                    return l * r.const
                return None  # nonlinear
            return None
        return None

    def _stringify_marker(self, expr: A.Expr | None, scope: Scope) -> Optional[str]:
        """Best-effort string for device/layout/memspace markers like
        `gpu(0)`, `cpu`, `smem`, etc."""
        if expr is None:
            return None
        if isinstance(expr, A.Name):
            return expr.name
        if isinstance(expr, A.Call) and isinstance(expr.callee, A.Name):
            args = ",".join(getattr(a, "value", "?").__repr__() if hasattr(a, "value") else "?"
                            for a in expr.args)
            return f"{expr.callee.name}({args})"
        return f"<{type(expr).__name__}>"

    # ------------------------------------------------------------------
    # Argument count + basic type checking for function calls
    # ------------------------------------------------------------------
    def _check_call_basic(self, call: A.Call, sig: FunctionSig,
                          arg_tys: list[Type], scope: Scope) -> None:
        """Check argument count and primitive type compatibility.
        Tensor-shape checking is in _check_call_shapes.

        Audit 28.8 cycle 2 B:C10: Logic-provenance diagnostics
        (trap 24100) are now batched. Pre-fix, each violating param
        produced an independent TypeError at the same `call.span` —
        so `f(a, b, c, d)` where all four Logic params got raw args
        produced four near-identical messages. Now we accumulate per
        call and emit a single grouped diagnostic when 2+ params
        violate, with the param-name list inline."""
        if len(arg_tys) != len(sig.params):
            self.errors.append(TypeError_(
                f"call to {sig.name!r}: expected {len(sig.params)} args, "
                f"got {len(arg_tys)}",
                call.span,
            ))
            return
        # Collect provenance violations across params for B:C10 batching.
        prov_violations: list[tuple[str, Type, Type, str]] = []
        for ((pname, pty), aty, arg_expr) in zip(sig.params, arg_tys, call.args):
            # For primitives, require an exact name match (i32 vs f32 etc.)
            if isinstance(pty, TyPrim) and isinstance(aty, TyPrim):
                if pty.name != aty.name:
                    # Treat 'size_N' (concrete sizes from shapes) loosely;
                    # they're not user-facing types.
                    if not (pty.name.startswith("size_")
                            or aty.name.startswith("size_")):
                        self.errors.append(TypeError_(
                            f"call to {sig.name!r}: arg {pname!r} expects "
                            f"{pty.name}, got {aty.name}",
                            call.span,
                        ))
            # Audit 28.8 cycle 3 D1: extend the call boundary check to
            # non-TyPrim parameter types. Pre-fix, only TyPrim-vs-TyPrim
            # was compared, so `fn use_q(q: Quote<i32>); use_q(42)` (i32
            # passed where Quote<i32> expected) silently typechecked
            # clean. Every other non-prim parameter type (TyDiff,
            # TyLogic, TyQuote, TyStruct, TyArray, TyRef, TyTile,
            # TyTensor, TyMemTier, TyFn, TyTuple, TyPtr) had the same
            # silent-acceptance hole. Now we fall through to
            # `_compatible` for any pair where neither side is TyVar /
            # TySize / TyUnknown (those defer to mono / cascade-safe).
            # The Logic-provenance path below still handles the
            # specialized TyLogic <-> non-Logic transition for the
            # better diagnostic; we skip the general check when that
            # specialized path will fire so the user sees one (not two)
            # messages.
            # Audit 28.8 cycle 5 C4-3: symmetric TyVar/TySize/TyUnknown
            # exclusion on aty side. Pre-fix, only pty was filtered for
            # TyVar/TySize, so the canonical generic-adapter pattern
            # `fn use_x[T](v: T) -> i32 { check_x(v) }` (T-typed arg
            # passed to i32-typed param) emitted a false-positive
            # "expects i32, got T" — but mono will bind T to a concrete
            # type at the call site of `use_x`, so the body-typecheck
            # should DEFER on TyVar at this call boundary. The same
            # cascade-safe rule already applies on the pty side; the
            # aty omission was asymmetric.
            elif (not isinstance(pty, (TyVar, TySize, TyUnknown))
                  and not isinstance(aty, (TyVar, TySize, TyUnknown))
                  and not (isinstance(pty, TyPrim)
                           and isinstance(aty, TyPrim))
                  and self._logic_provenance_violation_kind(pty, aty)
                      is None
                  and not self._compatible(pty, aty)):
                self.errors.append(TypeError_(
                    f"call to {sig.name!r}: arg {pname!r} expects "
                    f"{self._fmt(pty)}, got {self._fmt(aty)}",
                    call.span,
                ))
            # Audit 28.8 B2: provenance boundary check (trap 24100).
            # Collect (don't emit yet) so B:C10 batching can apply.
            kind = self._logic_provenance_violation_kind(pty, aty)
            if kind is not None:
                prov_violations.append((pname, pty, aty, kind))
            if ((self._contains_refinement(pty)
                    or self._contains_refinement(aty))
                    and self._compatible(aty, pty)
                    and not isinstance(pty, TyUnknown)):
                self._check_refinement_contextual_value(
                    arg_expr, aty, pty, call.span,
                    f"call to {sig.name!r}: arg {pname!r}",
                    scope,
                )
            elif (self._contains_refinement(sig.ret)
                  and (
                      self._compatible(aty, pty)
                      or isinstance(aty, (TyVar, TySize))
                      or isinstance(pty, (TyVar, TySize))
                  )
                  and not isinstance(pty, TyUnknown)):
                self._check_unrepresentable_scalar_context(
                    arg_expr,
                    pty,
                    arg_expr.span,
                    f"call to {sig.name!r}: arg {pname!r}",
                )
        # B:C10 — emit one grouped diagnostic if 2+ violations; else
        # the existing per-param path.
        if len(prov_violations) == 1:
            pname, pty, aty, kind = prov_violations[0]
            self._emit_logic_provenance_diagnostic(
                sig.name, pname, pty, aty, kind, call.span,
            )
        elif len(prov_violations) >= 2:
            self._emit_logic_provenance_grouped(
                sig.name, prov_violations, call.span,
            )

    # ------------------------------------------------------------------
    # Compile-time shape checking for function calls
    # ------------------------------------------------------------------
    def _check_call_shapes(self, call: A.Call, sig: FunctionSig,
                           arg_tys: list[Type], scope: Scope) -> None:
        """Build a Presburger constraint set from formal-vs-actual shape
        unification + where clauses, then check satisfiability.

        Each tensor parameter contributes per-axis equality constraints
        between the formal shape (in solver vars / consts) and the actual
        shape (in solver vars / consts). Where-clauses contribute
        additional Eq/Divides constraints.
        """
        solver = P.Solver()

        # Walk param/arg pairs and add shape-equality constraints.
        # We also track per-constraint readable labels for diagnostics.
        constraint_labels: list[str] = []
        for (pname, pty), aty in zip(sig.params, arg_tys):
            if isinstance(pty, TyTensor) and isinstance(aty, TyTensor):
                if len(pty.shape) != len(aty.shape):
                    self.errors.append(TypeError_(
                        f"call to {sig.name!r}: arg {pname!r} has rank "
                        f"{len(aty.shape)}, expected {len(pty.shape)}",
                        call.span,
                    ))
                    continue
                for axis, (pdim, adim) in enumerate(zip(pty.shape, aty.shape)):
                    p_lin = self._size_type_to_lin(pdim)
                    a_lin = self._size_type_to_lin(adim)
                    if p_lin is None or a_lin is None:
                        continue  # unknown shapes — skip (could warn)
                    diff = p_lin - a_lin
                    # Solver skips trivially-true (0 == 0) constraints. Only
                    # track a label if the constraint will actually be added.
                    if diff.is_zero():
                        continue
                    solver.add_eq_pair(p_lin, a_lin)
                    constraint_labels.append(
                        f"arg {pname!r} dim {axis}: expected {p_lin.pretty()}, "
                        f"got {a_lin.pretty()}"
                    )
            elif isinstance(pty, TyTile) and isinstance(aty, TyTile):
                # Audit 28.8 B8 (trap 16003): tile call-site shape +
                # memspace must agree. Pre-fix this branch was missing,
                # so `fn k(t: Tile<f32, [16,16], smem>)` called with
                # `Tile<f32, [32,32], smem>` (or `[16,16], hbm`) was
                # silently accepted — the kernel could overrun a 16x16
                # buffer or read from the wrong memory tier.
                if pty.memspace != aty.memspace:
                    self.errors.append(TypeError_(
                        f"call to {sig.name!r}: arg {pname!r} memspace "
                        f"mismatch — expected {pty.memspace!r}, got "
                        f"{aty.memspace!r} (trap 16003)",
                        call.span,
                    ))
                if len(pty.shape) != len(aty.shape):
                    self.errors.append(TypeError_(
                        f"call to {sig.name!r}: arg {pname!r} tile rank "
                        f"{len(aty.shape)}, expected {len(pty.shape)} "
                        f"(trap 16003)",
                        call.span,
                    ))
                    continue
                for axis, (pdim, adim) in enumerate(zip(pty.shape, aty.shape)):
                    p_lin = self._size_type_to_lin(pdim)
                    a_lin = self._size_type_to_lin(adim)
                    if p_lin is None or a_lin is None:
                        continue
                    diff = p_lin - a_lin
                    if diff.is_zero():
                        continue
                    solver.add_eq_pair(p_lin, a_lin)
                    constraint_labels.append(
                        f"arg {pname!r} tile-dim {axis}: expected "
                        f"{p_lin.pretty()}, got {a_lin.pretty()} (trap 16003)"
                    )

        # Add where-clause constraints (translate AST -> LinExpr).
        # We need a scope where the function's generic params are visible
        # as Presburger vars.
        where_scope = Scope()
        for g in sig.generics:
            if g.kind == "size":
                where_scope.define(g.name, TySize(g.name))
            else:
                where_scope.define(g.name, TyVar(g.name))
        for fn in self.prog.items:
            if isinstance(fn, A.FnDecl) and fn.name == sig.name:
                for w in fn.where_clauses:
                    self._add_where_constraint(solver, w.constraint, where_scope)

        # Verify each constraint is satisfied (i.e., solver does not refute it).
        # We focus on the explicit Eqs added above.
        contradictions = []
        for i, c in enumerate(solver.constraints):
            verdict = solver.implies(c)
            if verdict is False:
                # Pair the constraint with its label (if available)
                label = constraint_labels[i] if i < len(constraint_labels) else c.pretty()
                contradictions.append(label)

        if contradictions:
            details = "; ".join(contradictions[:3])
            self.errors.append(TypeError_(
                f"call to {sig.name!r}: shape constraint violated — {details}",
                call.span,
            ))

    def _logic_provenance_violation_kind(self, param_ty: Type,
                                          actual_ty: Type) -> Optional[str]:
        """Audit 28.8 cycle 2 B:C10: classify provenance mismatch
        without emitting. Returns "inject" if the actual lacks a
        Logic-wrap that the param requires, "strip" if the actual
        has a Logic-wrap that the param doesn't, or None if no
        violation."""
        if isinstance(param_ty, TyUnknown) or isinstance(actual_ty, TyUnknown):
            return None

        def _is_logic(t: Type) -> bool:
            if isinstance(t, TyLogic):
                return True
            if isinstance(t, TyDiff) and isinstance(t.inner, TyLogic):
                return True
            return False

        p_logic = _is_logic(param_ty)
        a_logic = _is_logic(actual_ty)
        if p_logic and not a_logic:
            return "inject"
        if a_logic and not p_logic:
            return "strip"
        return None

    def _emit_logic_provenance_diagnostic(self, fn_name: str, param_name: str,
                                          param_ty: Type, actual_ty: Type,
                                          kind: str, span: A.Span) -> None:
        """Single-param trap-24100 diagnostic — extracted from the
        old `_check_logic_provenance_boundary` for use by the new
        batching path in `_check_call_basic` (B:C10)."""
        if kind == "inject":
            self.errors.append(TypeError_(
                f"call to {fn_name!r}: arg {param_name!r} expects "
                f"Logic-wrapped value (provenance-typed); got "
                f"{self._fmt(actual_ty)} (trap 24100)",
                span,
                hint="wrap the value via a logic constructor to "
                     "preserve provenance",
            ))
        elif kind == "strip":
            self.errors.append(TypeError_(
                f"call to {fn_name!r}: arg {param_name!r} expects "
                f"{self._fmt(param_ty)}; got Logic-wrapped value — "
                f"would silently strip provenance (trap 24100)",
                span,
                hint="detach the Logic wrapper explicitly if you "
                     "really want a raw value",
            ))

    def _emit_logic_provenance_grouped(self, fn_name: str,
                                        violations: list,
                                        span: A.Span) -> None:
        """Audit 28.8 cycle 2 B:C10: emit a single grouped diagnostic
        when 2+ params violate the Logic-provenance boundary at the
        same call. Pre-fix, the per-param emission produced N copies
        of the same trap-24100 message; users saw a wall of
        near-identical text. Now we emit one diagnostic naming each
        violating param."""
        # Dedup by (param_name, kind) — defensive against any caller
        # that might double-feed the same param.
        seen: set[tuple[str, str]] = set()
        groups: dict[str, list[str]] = {"inject": [], "strip": []}
        for pname, pty, aty, kind in violations:
            key = (pname, kind)
            if key in seen:
                continue
            seen.add(key)
            groups[kind].append(pname)
        parts: list[str] = []
        if groups["inject"]:
            names = ", ".join(repr(n) for n in groups["inject"])
            parts.append(
                f"params {names} expect Logic-wrapped values "
                f"(provenance-typed); got raw values"
            )
        if groups["strip"]:
            names = ", ".join(repr(n) for n in groups["strip"])
            parts.append(
                f"params {names} expect raw values; got Logic-wrapped "
                f"(would silently strip provenance)"
            )
        msg = (f"call to {fn_name!r}: " + "; ".join(parts)
               + " (trap 24100)")
        self.errors.append(TypeError_(
            msg, span,
            hint="check each named param — wrap with logic_atom() to "
                 "inject, or detach the Logic wrapper to strip",
        ))

    def _check_logic_provenance_boundary(self, fn_name: str, param_name: str,
                                          param_ty: Type, actual_ty: Type,
                                          span: A.Span) -> None:
        """Audit 28.8 B2 (trap 24100): the provenance / Logic-wrapper
        type must agree at function-call boundaries.

        Phase-0 rules:
          * If the formal param is `Logic<T>` (possibly under TyDiff),
            and the actual is NOT Logic-wrapped, emit a diagnostic.
            Coercion is NOT silent — users must explicitly call a
            constructor (e.g., `logic_atom(x)`) to wrap a plain value
            into a Logic atom.
          * If the formal is plain T but actual is `Logic<T>`, also
            emit — passing a logic atom where a raw value is expected
            silently strips provenance.

        The check skips when either side is TyUnknown (inference still
        in progress) so it doesn't cascade off unrelated errors.

        Audit 28.8 cycle 2 B:C10: kept for backward compatibility with
        any callers outside `_check_call_basic`. Internally delegates
        to `_logic_provenance_violation_kind` +
        `_emit_logic_provenance_diagnostic` so behavior matches.
        """
        kind = self._logic_provenance_violation_kind(param_ty, actual_ty)
        if kind is not None:
            self._emit_logic_provenance_diagnostic(
                fn_name, param_name, param_ty, actual_ty, kind, span,
            )

    def _check_function_typed_call(
        self, call: A.Call, callee: TyFn, arg_tys: list[Type], scope: Scope,
    ) -> None:
        if len(arg_tys) != len(callee.params):
            self.errors.append(TypeError_(
                f"function-typed call: expected {len(callee.params)} args, "
                f"got {len(arg_tys)}",
                call.span,
            ))
            return
        for i, (arg_expr, arg_ty, param_ty) in enumerate(
                zip(call.args, arg_tys, callee.params)):
            if not self._compatible(arg_ty, param_ty):
                self.errors.append(TypeError_(
                    f"function-typed call arg {i}: expected "
                    f"{self._fmt(param_ty)}, got {self._fmt(arg_ty)}",
                    arg_expr.span,
                ))
                continue
            if ((self._contains_refinement(param_ty)
                 or self._contains_refinement(arg_ty))
                    and not isinstance(param_ty, TyUnknown)):
                self._check_refinement_contextual_value(
                    arg_expr, arg_ty, param_ty, arg_expr.span,
                    f"function-typed call arg {i}",
                    scope,
                )

    def _check_call_effects(self, call: A.Call, sig: FunctionSig) -> None:
        """Verify that calling a function with effects is permitted in the
        current calling context.

        Rules:
        - A @pure function may only call other functions whose declared
          effects are empty. Unannotated callees are allowed (their
          actual effect set is computed transitively by the IR-level
          effect_check pass — that's the soundness layer; this surface
          check only flags directly-declared effects).
        - A function with declared effects E may only call functions whose
          effects are a subset of E.
        - Calls to undeclared functions are not checked here (handled by
          shape-check or treated as opaque).
        """
        if self._current_pure and sig.effects:
            self.errors.append(TypeError_(
                f"@pure function {self._current_fn_name!r} cannot call "
                f"effectful {sig.name!r}",
                call.span,
            ))
            return
        # Check effect inclusion: callee's effects must subset caller's
        missing = sig.effects - self._current_effects
        if missing:
            missing_list = ", ".join(sorted(missing))
            self.errors.append(TypeError_(
                f"function {self._current_fn_name!r} calls {sig.name!r} "
                f"which requires effect(s) {{{missing_list}}}, "
                f"but caller does not declare them",
                call.span,
            ))

    # Names recognized as built-in operators by the typechecker — they're
    # only meaningful as Call callees and shouldn't fire "unbound" when
    # referenced bare.
    _BUILTIN_NAMES = frozenset({
        "detach", "attach",
        # Stage 36 Increment 1 — provenance-typed primitives.
        "prove", "unwrap_logic",
        # Stage 36 Increment 2 — provenance-composing combinators.
        "derive", "and_logic", "or_logic", "not_logic",
        # Stage 36 Increment 3 — boolean-algebra completeness.
        "xor_logic", "implies_logic", "eq_logic", "if_logic",
        "to_logic_bool",
        # Stage 36 Increment 5 — real two-parent provenance via arena
        # side-table.
        "register_derivation", "parent_left_at", "parent_right_at",
        # Stage 36 Increment 14 — three-parent provenance via atomic
        # ARENA_PUSH_TRIPLE + generic indexed accessor.
        "register_derivation3", "parent_at",
        # Stage 36 Increment 6 — fuzzy logic over Logic<f32> for AD.
        "fuzzy_and", "fuzzy_or", "fuzzy_not",
        # Stage 36 Increment 8 — fuzzy algebra completeness.
        "fuzzy_xor", "fuzzy_implies",
        "consolidate", "recall", "learn_to",
        # Stage 37 Inc 1 — tiered memory constructors + eliminators.
        "into_working", "into_episodic", "into_semantic", "into_procedural",
        "unwrap_working", "unwrap_episodic", "unwrap_semantic", "unwrap_procedural",
        # Stage 38 Inc 1 — spatial-frame constructors + eliminators.
        "into_world", "into_robot", "into_camera",
        "from_world", "from_robot", "from_camera",
        # Stage 38 Inc 2 — cross-frame transforms.
        "world_to_robot", "robot_to_world",
        "robot_to_camera", "camera_to_robot",
        "world_to_camera", "camera_to_world",
        # Stage 39 Inc 1 — temporal constructors + eliminators.
        "into_past", "into_present", "into_future", "into_eternal",
        "from_past", "from_present", "from_future", "from_eternal",
        # Stage 39 Inc 2 — temporal transitions.
        "to_past", "forecast", "recall_past", "actualize",
        # Stage 40 Inc 1 — modal constructors + eliminators.
        "into_known", "into_believed", "into_goal", "into_uncertain",
        "from_known", "from_believed", "from_goal", "from_uncertain",
        # Stage 40 Inc 2 — modal transitions (epistemic upgrades).
        "confirm", "act_on",
        # Stage 41 Inc 1 — causal constructors + eliminators.
        "into_cause", "into_effect", "into_joint", "into_independent",
        "from_cause", "from_effect", "from_joint", "from_independent",
        # Stage 41 Inc 2 — causal transitions.
        "propagate", "aggregate", "isolate",
        # Stage 46 Inc 1 — Result<T, E> constructors + accessors +
        # combinators. Two-parameter wrapper family; Phase-0
        # identity-lowered.
        "Ok", "Err",
        "unwrap_ok", "unwrap_err",
        "is_ok", "is_err",
        "map_ok", "map_err",
        # Stage 48 Inc 1 — `?` propagation operator. Parser desugars
        # `expr?` to `__try(expr)`. Reserved internal builtin (the
        # leading double-underscore is the convention for synthesizable-
        # only names — users cannot write `__try(...)` directly because
        # the lexer accepts it but typecheck additionally enforces
        # the enclosing-fn-return-type constraint, which has no
        # meaning at top level).
        "__try",
        "grad", "grad_rev", "grad_rev_all",
        "quote", "splice", "splice_f", "splice_f64",
        "modify", "modify_f", "modify_f64",
        "print_str", "print_int", "write_file", "read_file_int",
        "read_file_to_arena", "write_file_to_arena",
        "__arena_push", "__arena_get", "__arena_set", "__arena_len",
        "__strlen", "__strbyte", "__streq", "__strlit_to_arena",
        "__hash_i32",
        # Phase 2.2 step 2 — float-bit reinterpret intrinsics.
        "__bits_of_f32", "__f32_from_bits",
        "__bits_of_f64", "__f64_from_bits",
        # Stage 28.5 — panic/abort policy. `panic` is a builtin that
        # takes a single string-literal arg and emits a trap (id 28501).
        "panic",
        "thread_idx", "thread_idx_x", "thread_idx_y", "thread_idx_z",
        "block_idx", "block_idx_x", "block_idx_y", "block_idx_z",
        "block_dim", "block_dim_x", "block_dim_y", "block_dim_z",
    })
    _GPU_INDEX_BUILTINS = frozenset({
        "thread_idx", "thread_idx_x", "thread_idx_y", "thread_idx_z",
        "block_idx", "block_idx_x", "block_idx_y", "block_idx_z",
        "block_dim", "block_dim_x", "block_dim_y", "block_dim_z",
    })

    # Names of well-known stdlib functions that are surfaced as
    # did-you-mean candidates even when the stdlib hasn't been parsed in
    # (e.g. when a user invokes the typechecker on a fragment without
    # `include_stdlib=True`). Keep aligned with helixc/stdlib/transcendentals.hx.
    _STDLIB_HINTS = frozenset({
        "__exp", "__log", "__sin", "__cos", "__sqrt", "__powi",
        "__relu", "__sigmoid", "__tanh", "__softplus", "__silu",
        "__abs", "__gelu", "__floor", "__ceil",
        "__rand_step", "__momentum_step_v",
        "__min", "__max", "__clamp",
        "__min_i32", "__max_i32", "__clamp_i32",
        "__sgd_step", "__adam_step",
    })

    def _unbound_name_suggestion(self, name: str, span: A.Span,
                                  scope: "Scope") -> None:
        """Emit a typecheck error for an unbound name, with a Levenshtein
        'did you mean?' suggestion drawn from in-scope names + functions.
        Suppresses duplicate diagnostics for the same name and skips the
        small set of compiler-recognized builtins. The _seen_unbound set
        is initialised in __init__ so a re-checked instance starts clean."""
        if name in self._BUILTIN_NAMES:
            return
        from difflib import get_close_matches
        if name in self._seen_unbound:
            return
        self._seen_unbound.add(name)
        # Build candidate set from current scope + function names + builtins
        # + stdlib hints (helps when stdlib wasn't parsed in).
        candidates: list[str] = list(self.functions.keys())
        candidates.extend(self._BUILTIN_NAMES)
        candidates.extend(self._STDLIB_HINTS)
        s: "Scope | None" = scope
        while s is not None:
            candidates.extend(s.locals.keys())
            s = s.parent
        suggestions = get_close_matches(name, candidates, n=1, cutoff=0.6)
        hint = None
        if suggestions:
            hint = f"did you mean {suggestions[0]!r}?"
        self.errors.append(TypeError_(
            f"unbound name {name!r}", span, hint=hint
        ))

    def _add_where_constraint(self, solver: P.Solver, expr: A.Expr,
                              scope: Scope) -> None:
        """Translate a where-clause expression into Presburger constraints."""
        if isinstance(expr, A.Binary) and expr.op == "==":
            l = self._size_expr_to_lin(expr.left, scope)
            r = self._size_expr_to_lin(expr.right, scope)
            if l is not None and r is not None:
                solver.add_eq_pair(l, r)
        elif isinstance(expr, A.Binary) and expr.op == "%" and \
             isinstance(expr.right, A.IntLit):
            # `expr % k` — record as Divides
            l = self._size_expr_to_lin(expr.left, scope)
            if l is not None:
                solver.add_divides(l, expr.right.value)
        # Other forms: skip for v0.1

    # ---- function body checking ----

    def _modal_origin_of_expr(self, expr: A.Expr) -> Optional[str]:
        """Stage 52 closure gate-6 unified helper for the 3 CRITICAL
        silent-failure findings (Call-form scrutinee + name-alias
        let/Assign + PatOr binders). Returns the modal-origin kind
        ('known'/'believed'/'goal'/'uncertain') of an expression if
        it can be statically determined, else None.

        Cases handled:
        - Name lookup in `_modal_origin_provenance` (covers let-alias
          `let s = r;` where r is tainted — Stage 52 gate-6 CRITICAL-2).
        - Direct `Call(Name(from_X), ...)` for any modal eliminator
          (covers `match from_uncertain(u) { x => ... }` scrutinee
          — Stage 52 gate-6 CRITICAL-1, and reuses the same map as
          Let/Assign populate sites for invariant unity).
        - Stage 53 Inc 1: `Call(Name(user_fn), ...)` where user_fn's
          declared return type is a modal wrapper (TyModal). Closes
          the helper-fn indirection laundering vector — `fn launder(
          x: i32) -> Known<i32> { into_known(x) }` now propagates
          'known' at every call site, just like into_known() itself.

        Returning None means "no static modal-origin claim" — the
        F1 launder consult falls through to the Phase-0 dynamic
        territory (no fire), which is correct: the helper is the
        SINGLE source of truth that prior split logic missed.

        Used by (token-pinned, gate-7 code-review F3 fix —
        line numbers were drifting by 50-220 lines through
        the gate-3→6 closure rounds):
        - Let-stmt populate: the `let_rhs_kind` install in
          `_check_stmt`'s A.Let branch.
        - Assign-stmt populate: the `assign_rhs_kind` install in
          `_check_expr`'s A.Assign branch.
        - Match scrutinee for PatBind/PatOr taint propagation:
          the `scrut_kind` propagation in `_check_expr`'s A.Match
          branch.
        - Stage 52 Inc 7 / gate-10 HIGH-1 fix: the `source_kind`
          consult at the BUILTIN into_X launder check in
          `_check_expr`'s A.Call into_X arm (replaces the prior
          2 narrow syntactic guards — A.Call inner-from-X and
          A.Name taint-tracking).
        - Stage 53 Inc 1: the `arg_kind` consult at the user-fn
          launder check in `_check_expr`'s A.Call user-fn arm.

        Shadowed builtin safety: if a user defines a fn shadowing
        a builtin modal eliminator (e.g. `fn from_uncertain(...)`),
        `_MODAL_ELIM_TO_KIND` wins because it's checked FIRST. The
        shadow error fires separately at fn-decl site; this helper's
        order ensures no incidental modal-origin inconsistency.
        """
        if isinstance(expr, A.Name):
            return self._modal_origin_provenance.get(expr.name)
        if isinstance(expr, A.Call) and isinstance(expr.callee, A.Name):
            callee = expr.callee.name
            # Stage 52: builtin modal eliminators (from_known, etc.).
            if callee in _MODAL_ELIM_TO_KIND:
                return _MODAL_ELIM_TO_KIND[callee]
            # Stage 53 Inc 1: user-defined helpers whose declared
            # return type is a modal wrapper.
            if callee in self._fn_modal_return_kind:
                return self._fn_modal_return_kind[callee]
        # Stage 52 Inc 6 / gate-2 HIGH-2 (originally deferred) /
        # gate-9 silent-failure O1 fix: recursive yield-from-modal
        # detection. A.Block / A.If / A.Match arms whose terminal
        # expression yields a modal-origin value propagate that kind.
        # Reproducer (was silent):
        #   let v: i32 = match scrut { x => from_uncertain(u) };
        #   into_known(v);  // v inherits 'uncertain' from arm tail
        # All branches must agree on the kind for the recursion to
        # return a kind; mixed/missing → None (drop to dynamic
        # territory, same conservative semantics as gate-3 multi-
        # kind divergence drop).
        if isinstance(expr, A.Block):
            if expr.final_expr is not None:
                return self._modal_origin_of_expr(expr.final_expr)
            return None
        # Stage 52 Inc 8 / gate-11 silent-failure HIGH-1 fix:
        # `unsafe { ... }` wraps a Block; the inner Block's tail
        # is exactly the same modal source as Block tail. Without
        # this arm, `into_known(unsafe { from_uncertain(u) })`
        # silently passed — cascading-defect: gate-10 caught
        # Inc 6's missed wiring on builtin into_X; gate-11 catches
        # Inc 7's missed AST coverage on UnsafeBlock.
        if isinstance(expr, A.UnsafeBlock):
            return self._modal_origin_of_expr(expr.body)
        if isinstance(expr, A.If):
            then_kind = self._modal_origin_of_expr_block_tail(expr.then)
            if expr.else_ is None:
                return None  # no-else can't guarantee kind
            if isinstance(expr.else_, A.Block):
                else_kind = self._modal_origin_of_expr_block_tail(expr.else_)
            else:
                else_kind = self._modal_origin_of_expr(expr.else_)
            if then_kind is not None and then_kind == else_kind:
                return then_kind
            return None
        if isinstance(expr, A.Match):
            kinds: set[str] = set()
            for arm in expr.arms:
                if isinstance(arm.body, A.Block):
                    k = self._modal_origin_of_expr_block_tail(arm.body)
                else:
                    k = self._modal_origin_of_expr(arm.body)
                if k is None:
                    return None
                kinds.add(k)
            if len(kinds) == 1:
                return next(iter(kinds))
            return None
        return None

    def _modal_origin_of_expr_block_tail(
        self, block: A.Block
    ) -> Optional[str]:
        """Helper for the recursive yield-from-modal detection in
        `_modal_origin_of_expr`. Returns the modal kind of a block's
        tail expression if statically determinable."""
        if block.final_expr is not None:
            return self._modal_origin_of_expr(block.final_expr)
        return None

    def _check_fn(self, fn: A.FnDecl) -> None:
        sig = self.functions.get(fn.name)
        if sig is None:
            return
        # Stage 48 closure gate-2 silent-failure M5 fix: clear the
        # Result-constructor provenance map at function entry. Pre-
        # fix, a let-binding `let r = Ok(7)` in fn A left `r='ok'`
        # in the dict; fn B's parameter ALSO named `r` then
        # inherited the stale 'ok' provenance, falsely rejecting
        # `unwrap_err(r)` on B's parameter as "Ok-constructed".
        # Per-fn locals must not leak across the fn boundary.
        #
        # Gate-5 G4-M1 parity: also clear the parallel scope
        # stacks. Push/pop balance within a single fn body's
        # _check_block makes this defense-in-depth (a fn body is
        # always A.Block, so push/pop balance is preserved by the
        # try/finally), but a generic exception escaping
        # _check_block between push and pop would leak frames into
        # the next fn without the explicit reset.
        self._result_constructor_provenance = {}
        self._result_let_block_scopes = []
        self._result_assigns_block_scopes = []
        # Stage 52 Inc 1: clear modal-origin taint map per fn entry.
        # Same defect class as Stage 48 gate-2 M5 (cross-fn stale
        # provenance) — without this clear, fn A's `let r =
        # from_uncertain(u)` would taint `r` in fn B's parameter list.
        self._modal_origin_provenance = {}
        self._modal_origin_let_block_scopes = []
        self._modal_origin_assigns_block_scopes = []
        # Stage 52 closure gate-7 type-design HIGH-1 fix: also
        # clear the covert-return-channel slot. Currently masked
        # by the always-precedes-read ordering at union sites,
        # but defense-in-depth — a future edit reordering a union
        # site to read before the next _check_block call could
        # silently inherit fn A's last assigns-set.
        self._last_modal_assigns_popped = set()
        error_start = len(self.errors)
        completed = False
        kernel_hbm_indexables = (
            self._validate_kernel_hbm_params(fn, sig)
            if "kernel" in fn.attrs else set()
        )
        if "kernel" in fn.attrs and not isinstance(sig.ret, TyUnit):
            self.errors.append(TypeError_(
                "@kernel functions must return () for PTX emission",
                fn.span,
            ))
        # Stage 16.5: extern "C" declarations have no body to check,
        # but kernel ABI validation above still applies.
        if fn.is_extern:
            return
        # Set effect-checking context for this function
        prev_pure = self._current_pure
        prev_effects = self._current_effects
        prev_name = self._current_fn_name
        prev_is_kernel = self._current_is_kernel
        prev_hbm_tile_indexables = self._current_hbm_tile_indexables
        prev_return_ty = self._current_return_ty
        self._current_pure = sig.is_pure
        self._current_effects = sig.effects
        self._current_fn_name = sig.name
        self._current_is_kernel = "kernel" in fn.attrs
        self._current_hbm_tile_indexables = set(kernel_hbm_indexables)
        self._current_return_ty = sig.ret
        try:
            self._check_fn_body(fn, sig)
            completed = True
        finally:
            self._current_pure = prev_pure
            self._current_effects = prev_effects
            self._current_fn_name = prev_name
            self._current_is_kernel = prev_is_kernel
            self._current_hbm_tile_indexables = prev_hbm_tile_indexables
            self._current_return_ty = prev_return_ty
        if (completed
                and self._contains_refinement(sig.ret)
                and len(self.errors) != error_start):
            self._invalid_refined_return_functions.add(fn.name)

    def _validate_kernel_hbm_params(
        self, fn: A.FnDecl, sig: FunctionSig
    ) -> set[str]:
        indexables: set[str] = set()
        for param, (name, ty) in zip(fn.params, sig.params):
            if not (isinstance(ty, TyTile)
                    and ty.memspace.lower() == "hbm"):
                continue
            if (not isinstance(ty.dtype, TyPrim)
                    or ty.dtype.name not in self._HBM_TILE_PARAM_DTYPES):
                got = self._fmt(ty.dtype)
                allowed = ", ".join(sorted(self._HBM_TILE_PARAM_DTYPES))
                self.errors.append(TypeError_(
                    f"@kernel HBM tile parameter dtype {got} is not "
                    f"supported by PTX yet; expected one of {allowed}",
                    param.span,
                ))
                continue
            if len(ty.shape) != 1:
                self.errors.append(TypeError_(
                    "@kernel HBM tile parameters must be 1D for PTX "
                    f"emission; got {len(ty.shape)}D",
                    param.span,
                ))
                continue
            indexables.add(name)
        return indexables

    def _check_fn_body(self, fn: A.FnDecl, sig: FunctionSig) -> None:
        gen_scope = Scope()
        for g in fn.generics:
            if g.kind == "size":
                gen_scope.define(g.name, TySize(g.name))
            else:
                gen_scope.define(g.name, TyVar(g.name))
        body_scope = Scope(parent=gen_scope)
        for param, (name, t) in zip(fn.params, sig.params):
            body_scope.define(name, t, is_mut=param.is_mut)
        # Check body expression / block
        body_ty = self._check_block(
            fn.body,
            body_scope,
            expected_final_ty=sig.ret,
            final_context=f"return value of function {fn.name!r}",
        )
        # Compatibility check (simplified — strict equality on resolved types)
        if not self._compatible(body_ty, sig.ret):
            self.errors.append(TypeError_(
                f"function {fn.name!r}: body type {self._fmt(body_ty)} "
                f"does not match return type {self._fmt(sig.ret)}",
                fn.span,
            ))

    def _check_block(
        self,
        block: A.Block,
        scope: Scope,
        *,
        expected_final_ty: Type | None = None,
        final_context: str | None = None,
    ) -> Type:
        inner = Scope(parent=scope)
        self._push_local_const_scope()
        # Stage 48 closure gate-2 silent-failure F1 fix: snapshot the
        # Result-constructor provenance map at block entry and
        # restore at exit. Inner-block `let r: Result = Ok(5)`
        # shadow no longer overwrites the outer Err-constructed
        # `r`'s provenance.
        #
        # Stage 48 closure gate-3 silent-failure G3-F1 fix: at
        # restore, scope-disambiguate WHO caused any post-block
        # diff between saved and current. Inner-LET shadows
        # (introduced via let inside the block) are popped from
        # the restored map — the outer scope is unchanged. Names
        # that were in the saved map AND whose current value
        # differs from saved AND were NOT inner-let-introduced
        # were mutated by an inner ASSIGN (or map_ok/map_err)
        # to the outer name — drop them from the restored map
        # so the dynamic mutation is honoured (joins the F1-
        # dynamic Phase-0 limitation: typecheck-clean, runtime
        # still wrong without Stage 49 runtime tag, but no false
        # static 'Ok-constructed' claim).
        #
        # Stage 48 closure gate-5 silent-failure G4-F2 fix: gate-3's
        # let-set alone produced a per-name not per-event mask.
        # A block that ASSIGNS to outer `r` and THEN shadow-LETs
        # a new `r` would put `r` in inner_lets, masking the
        # prior assign's mutation at restore. The parallel
        # assigns-set populated by the Assign-arm makes the mask
        # per-event: any saved name in inner_assigns is treated
        # as mutated regardless of let-shadow membership.
        saved_provenance = dict(self._result_constructor_provenance)
        self._result_let_block_scopes.append(set())
        self._result_assigns_block_scopes.append(set())
        # Stage 52 closure gate-1 F1e + HIGH-1/3 fix: scope-aware
        # snapshot + let-set tracking for _modal_origin_provenance.
        # Distinguishes inner-LET shadows (drop their taint at
        # exit, restore outer's if present) from inner-Assign
        # mutations of outer names (preserve the new taint —
        # AI-safety semantics say from_X introductions must
        # surface, INVERTED from Result-provenance which DROPS
        # on inner-Assign).
        saved_modal_origin = dict(self._modal_origin_provenance)
        self._modal_origin_let_block_scopes.append(set())
        self._modal_origin_assigns_block_scopes.append(set())
        try:
            for stmt in block.stmts:
                self._check_stmt(stmt, inner)
            if block.final_expr is not None:
                final_ty = self._check_expr(block.final_expr, inner)
                if (expected_final_ty is not None
                        and final_context is not None
                        and self._compatible(final_ty, expected_final_ty)
                        and (self._contains_refinement(expected_final_ty)
                             or self._contains_refinement(final_ty))):
                    self._check_refinement_contextual_value(
                        block.final_expr,
                        final_ty,
                        expected_final_ty,
                        block.final_expr.span,
                        final_context,
                        inner,
                    )
                elif (expected_final_ty is not None
                      and final_context is not None
                      and self._compatible(final_ty, expected_final_ty)):
                    if self._check_unrepresentable_scalar_context(
                            block.final_expr,
                            expected_final_ty,
                            block.final_expr.span,
                            final_context,
                            report=False,
                    ):
                        self._unrepresentable_scalar_return_functions.add(
                            self._current_fn_name)
                return final_ty
            return TyUnit()
        finally:
            # Gate-3 code-review M1: pop-then-restore would skip
            # the provenance restore if `_pop_local_const_scope`
            # raised (push/pop imbalance). Nested try ensures the
            # provenance restore is always-must-run.
            try:
                self._pop_local_const_scope()
            finally:
                # Gate-3 G3-F1 + Gate-5 G4-F2 scope-aware restore:
                # the inner-let names recorded by this block are
                # inner-only shadows — they don't affect outer
                # state. The inner-assign names record the actual
                # mutation events (the per-event mask that closes
                # the G4-F2 ASSIGN-then-LET-shadow hole). Names
                # in the saved map are dropped if:
                #   (a) they were inner-assigned at any point
                #       (per-event mask), OR
                #   (b) their current value differs from saved
                #       AND they were NOT inner-let-introduced
                #       (the gate-3 G3-F1 detection path).
                # Both cases drop to F1-dynamic Phase-0 territory
                # (no false static claim).
                inner_lets = (self._result_let_block_scopes.pop()
                              if self._result_let_block_scopes
                              else set())
                inner_assigns = (self._result_assigns_block_scopes.pop()
                                 if self._result_assigns_block_scopes
                                 else set())
                mutated_outer_names = {
                    n for n in saved_provenance
                    if (n in inner_assigns
                        or (n not in inner_lets
                            and self._result_constructor_provenance.get(n)
                            != saved_provenance.get(n)))
                }
                self._result_constructor_provenance = saved_provenance
                for n in mutated_outer_names:
                    self._result_constructor_provenance.pop(n, None)
                # Stage 52 gate-1 F1e + HIGH-1/3 restore:
                # selective restore preserving inner-Assign
                # mutations while dropping inner-LET shadow taint.
                # For each name in current dict:
                #   - If introduced via inner-let: drop the inner
                #     entry. If the outer had a taint for that
                #     name, restore it from saved.
                #   - Else (untouched OR inner-Assign mutation):
                #     keep the current value (propagate taint).
                inner_modal_lets = (
                    self._modal_origin_let_block_scopes.pop()
                    if self._modal_origin_let_block_scopes
                    else set()
                )
                # Pop the assigns-set + stash for caller's union
                # site (NEW-HIGH-2/3/4 fix). _check_block doesn't
                # use the assigns-set directly; the if-else /
                # match union sites read self._last_modal_assigns_
                # popped after each branch returns.
                self._last_modal_assigns_popped = (
                    self._modal_origin_assigns_block_scopes.pop()
                    if self._modal_origin_assigns_block_scopes
                    else set()
                )
                # Stage 52 closure gate-3 NEW-HIGH-1 fix: iterate
                # the inner_modal_lets set, NOT the current dict's
                # keys. Pre-fix, an inner-let-shadow `let r: i32 =
                # 7;` triggered the Let-stmt POP (line ~2873) which
                # removed `r` from the dict — so at restore time
                # `r` was no longer in current.keys() and the
                # saved outer taint was never restored. Net effect:
                # `let r = from_uncertain(u); { let r: i32 = 7; };
                # into_known(r)` silently passed (outer r's taint
                # was lost when the inner shadow popped it).
                # Post-fix: iterate inner_modal_lets directly and
                # restore each from saved if present, else drop.
                for name in inner_modal_lets:
                    if name in saved_modal_origin:
                        self._modal_origin_provenance[name] = (
                            saved_modal_origin[name]
                        )
                    else:
                        self._modal_origin_provenance.pop(
                            name, None)

    def _check_loop_body_with_modal_union(
        self, body: A.Block, scope: Scope
    ) -> None:
        """Stage 52 closure gate-7 silent-failure HIGH-1+2 fix
        (Stage 52 Inc 5): apply union semantics to loop body
        modal-origin tracking. A loop body may execute 0 or N+
        times at runtime; pre-fix, the body's _check_block alone
        carried the body's mutation upward, silently dropping the
        pre-loop taint when the body opaque-cleared a tainted
        name. The 0-iter runtime path preserves the pre-loop
        taint, so into_X(name) after the loop should FIRE if the
        pre-loop kind mismatches the target.

        Semantic shape (mirror of A.If no-else union):
        - "executes" arm: post-body dict, body assigns
        - "0-iter" identity arm: pre-loop dict, empty assigns
        - Apply kept_somewhere / cleared semantics from A.If/A.Match
          (gate-7 conservative-fire: any preserved taint overrides
           branch-cleared signal).

        Verified clean cases:
        - body INSTALLS same kind as pre-loop: propagate (kind match)
        - body INSTALLS different kind: drop (multi-kind divergence,
          Phase-0 limit per gate-4 CRITICAL-1)
        - body CLEARS, pre-loop tainted: FIRE — 0-iter case may leak
        """
        modal_origin_pre_loop = dict(self._modal_origin_provenance)
        self._check_block(body, scope)
        body_assigns = set(self._last_modal_assigns_popped)
        body_result = dict(self._modal_origin_provenance)
        # Union: collect observed kinds across pre-loop + body.
        observed_kinds: dict[str, set[str]] = {}
        for name, kind in modal_origin_pre_loop.items():
            observed_kinds.setdefault(name, set()).add(kind)
        for name, kind in body_result.items():
            observed_kinds.setdefault(name, set()).add(kind)
        # kept_somewhere: names present in EITHER arm result
        # (pre-loop snapshot = 0-iter identity arm result).
        kept_somewhere: set[str] = set()
        kept_somewhere.update(modal_origin_pre_loop.keys())
        kept_somewhere.update(body_result.keys())
        # cleared: body assigned name AND body result has no entry
        # AND no other "arm" preserves it (i.e., pre-loop also doesn't
        # have it). With pre-loop as identity arm, a name cleared by
        # body but preserved by pre-loop should NOT drop — it should
        # propagate because the 0-iter case keeps the taint.
        cleared_names: set[str] = set()
        for name in body_assigns:
            if (name not in body_result
                    and name not in kept_somewhere):
                cleared_names.add(name)
        # Build the unioned dict.
        unioned_loop: dict[str, str] = {}
        for name, kinds in observed_kinds.items():
            if name in cleared_names:
                continue
            if len(kinds) == 1:
                unioned_loop[name] = next(iter(kinds))
            # Multi-kind divergence drops (Phase-0 limit; Inc 4 would
            # add the multi-kind diagnostic).
        self._modal_origin_provenance = unioned_loop
        # Reset _last_modal_assigns_popped so a subsequent union
        # site doesn't see this loop's body assigns. (Defense-in-
        # depth — the next _check_block call overwrites it anyway,
        # but ordering invariants benefit from explicit reset.)
        self._last_modal_assigns_popped = set()

    def _check_expr_in_block_scope(
        self, expr: A.Expr, scope: Scope
    ) -> Type:
        """Stage 48 closure gate-5 silent-failure G4-F1 / type-design
        G4-H2 fix: wrap _check_expr with the same provenance
        snapshot/restore that _check_block applies. Used at sites
        where an expression-form arm body bypasses _check_block:
        match-arm body, if/else expression-form arm, match-guard.

        Without this wrapper, an Assign inside such an arm
        permanently mutates the outer provenance dict — branching
        control flow is silently unmodeled. Block-form arms hit
        _check_block directly and get the same protection.

        Mirrors the gate-3 G3-F1 + gate-5 G4-F2 scope-aware
        restore semantics exactly: inner-let shadows leave outer
        provenance untouched; inner-assign mutations of outer
        names (including ASSIGN-then-LET-shadow on the same name)
        drop the outer's stale entry."""
        saved_provenance = dict(self._result_constructor_provenance)
        self._result_let_block_scopes.append(set())
        self._result_assigns_block_scopes.append(set())
        # Stage 52 gate-1 F1e + HIGH-1/3 parity at expression-arm.
        # Same scope-aware semantics as _check_block: inner-let
        # shadows pop (restore outer if present); inner-Assign
        # mutations propagate upward (AI-safety: taint MUST surface).
        saved_modal_origin = dict(self._modal_origin_provenance)
        self._modal_origin_let_block_scopes.append(set())
        self._modal_origin_assigns_block_scopes.append(set())
        try:
            return self._check_expr(expr, scope)
        finally:
            inner_lets = (self._result_let_block_scopes.pop()
                          if self._result_let_block_scopes
                          else set())
            inner_assigns = (self._result_assigns_block_scopes.pop()
                             if self._result_assigns_block_scopes
                             else set())
            mutated_outer_names = {
                n for n in saved_provenance
                if (n in inner_assigns
                    or (n not in inner_lets
                        and self._result_constructor_provenance.get(n)
                        != saved_provenance.get(n)))
            }
            self._result_constructor_provenance = saved_provenance
            for n in mutated_outer_names:
                self._result_constructor_provenance.pop(n, None)
            # Scope-aware modal-origin restore at expression-arm
            # site, mirroring the _check_block logic — gate-3
            # NEW-HIGH-1 fix: iterate inner_modal_lets, not the
            # current dict's keys (inner-let-shadow on same name
            # POPS from dict before this restore runs).
            inner_modal_lets = (
                self._modal_origin_let_block_scopes.pop()
                if self._modal_origin_let_block_scopes
                else set()
            )
            # Pop the assigns-set + stash for caller's union
            # site (NEW-HIGH-2/3/4 fix). Parallel to _check_block.
            # Stage 52 closure gate-6 latent-bug fix (surfaced by
            # gate-6 CRITICAL-2 let-alias fix): when the wrapped
            # expr is itself a Block (e.g. `arm => { r = ...; }`),
            # the inner _check_block already pushed AND popped its
            # own scope, leaving `_last_modal_assigns_popped` set
            # to the inner's contribution. Naive overwrite here
            # would CLOBBER that signal with the outer (empty)
            # scope. UNION semantics preserves the inner-block
            # assigns so the caller's match/if union loop sees
            # the full picture. Gate-3 NEW-HIGH-4's test never
            # exercised the drop path pre-gate-6 because the let-
            # alias silently broke; gate-6 surfaced this latent
            # double-pop bug.
            inner_assigns_signal = set(self._last_modal_assigns_popped)
            outer_assigns_scope = (
                self._modal_origin_assigns_block_scopes.pop()
                if self._modal_origin_assigns_block_scopes
                else set()
            )
            self._last_modal_assigns_popped = (
                outer_assigns_scope | inner_assigns_signal
            )
            for name in inner_modal_lets:
                if name in saved_modal_origin:
                    self._modal_origin_provenance[name] = (
                        saved_modal_origin[name]
                    )
                else:
                    self._modal_origin_provenance.pop(name, None)

    def _check_stmt(self, stmt: A.Stmt, scope: Scope) -> None:
        if isinstance(stmt, A.Let):
            value_ty: Type = TyUnit()
            stmt_error_start = len(self.errors)
            if stmt.value is not None:
                value_ty = self._check_expr(stmt.value, scope)
            if stmt.ty is not None:
                declared = self._resolve_type(stmt.ty, scope)
                if stmt.value is not None and not self._compatible(value_ty, declared):
                    self.errors.append(TypeError_(
                        f"let {stmt.name!r}: declared {self._fmt(declared)} "
                        f"but value is {self._fmt(value_ty)}",
                        stmt.span,
                    ))
                # Static overflow check: if the value is an IntLit and the
                # declared type is a fixed-width integer, verify the value
                # fits in that width (signed range for i*, unsigned for u*).
                if (stmt.value is not None
                        and isinstance(stmt.value, A.IntLit)
                        and isinstance(declared, TyPrim)):
                    self._check_int_lit_fits(stmt.value, declared)
                if (stmt.value is not None
                        and len(self.errors) == stmt_error_start):
                    self._check_refinement_contextual_value(
                        stmt.value, value_ty, declared, stmt.span,
                        f"let {stmt.name!r}",
                        scope,
                    )
                elif self._contains_refinement(declared):
                    self.errors.append(TypeError_(
                        f"let {stmt.name!r}: refined type "
                        f"{self._fmt(declared)} requires an initializer "
                        f"that proves its refinement in Stage 31",
                        stmt.span,
                        hint="initialize refined values with a proven value",
                    ))
                bind_ty = declared
                if (self._contains_refinement(declared)
                        and len(self.errors) != stmt_error_start):
                    bind_ty = self._erase_refinement(declared)
                scope.define(stmt.name, bind_ty, is_mut=stmt.is_mut)
            else:
                bind_ty = value_ty
                if (self._contains_refinement(value_ty)
                        and len(self.errors) != stmt_error_start):
                    bind_ty = self._erase_refinement(value_ty)
                scope.define(stmt.name, bind_ty, is_mut=stmt.is_mut)
            # Stage 48 closure gate-3 G3-F1 fix: record this let-
            # binding in the current block's let-set so the
            # _check_block exit can distinguish inner-let shadows
            # (don't touch outer provenance on restore) from
            # inner-assign mutations of outer names (drop the
            # stale outer provenance on restore).
            if self._result_let_block_scopes:
                self._result_let_block_scopes[-1].add(stmt.name)
            # Stage 46 closure gate-2 silent-failure G2-F1 fix:
            # if the let RHS is a direct Ok(...) / Err(...) call,
            # record the constructor provenance on the binding so
            # unwrap_ok/unwrap_err can detect the wrong-arm case
            # even when the declared type annotation strips the
            # TyUnknown hint that gate-1 F4 relied on. Only direct
            # constructor calls qualify; complex RHS expressions
            # (e.g. `map_ok(r, 99)`, a fn call returning Result)
            # don't get tracked — they'll have the existing F4
            # behavior or pass through.
            if (stmt.value is not None
                    and isinstance(stmt.value, A.Call)
                    and isinstance(stmt.value.callee, A.Name)
                    and stmt.value.callee.name in ("Ok", "Err")):
                self._result_constructor_provenance[stmt.name] = (
                    "ok" if stmt.value.callee.name == "Ok"
                    else "err"
                )
            # Stage 46 closure gate-3 silent-failure G3-F2 fix:
            # propagate provenance through map_ok / map_err
            # whose first arg is a Name with known provenance.
            # Pre-fix, `let r0 = Ok(7); let r = map_ok(r0, 999);
            # unwrap_err(r)` typechecked clean because the
            # let-RHS matcher only handled direct Ok/Err. map_ok
            # always returns an Ok-shape Result; map_err always
            # returns an Err-shape Result. Both preserve the
            # source arm's provenance (they only TRANSFORM the
            # matching inner value, not the variant tag).
            elif (stmt.value is not None
                    and isinstance(stmt.value, A.Call)
                    and isinstance(stmt.value.callee, A.Name)
                    and stmt.value.callee.name in ("map_ok",
                                                   "map_err")
                    and len(stmt.value.args) >= 1
                    and isinstance(stmt.value.args[0], A.Name)
                    and stmt.value.args[0].name
                        in self._result_constructor_provenance):
                # map_ok / map_err do not change the variant
                # tag — they only rebuild the value side. So
                # the new binding inherits the source's
                # provenance.
                self._result_constructor_provenance[stmt.name] = (
                    self._result_constructor_provenance[
                        stmt.value.args[0].name]
                )
            else:
                # Stage 46 closure gate-3 code-review CRITICAL
                # G3-F1 follow-up fix: any let with this name
                # whose RHS is NOT a direct Ok/Err or
                # map_ok/map_err call must POP any prior
                # provenance entry. Pre-fix, the map was keyed
                # by bare name with no scope qualification, so
                # a `let r` in fn B inherited stale "ok"
                # provenance from a `let r = Ok(7)` in fn A
                # (the entries are never cleared on
                # function-body entry/exit). Verified false-
                # positive: `fn first() { let r = Ok(7); ... }`
                # followed by `fn main() { let r =
                # opaque_returning_err(); unwrap_err(r) }`
                # incorrectly rejected the main's unwrap_err
                # as "constructed via Ok()". Post-fix, the
                # opaque-RHS let pops the stale entry. A
                # proper scope-stack lift is Stage 47+ work;
                # this dict-pop is the minimal sound fix.
                self._result_constructor_provenance.pop(
                    stmt.name, None)
            # Stage 52 Inc 1 — modal-origin taint tracking.
            # Populate when value is `Call(from_X, ...)` for any
            # of the 4 modal eliminators. Pop on any other RHS
            # (mirrors the Result-provenance discipline at gate-3).
            # Stage 52 closure gate-6 CRITICAL-2 fix: also populate
            # when RHS is a Name carrying tracked modal-origin
            # taint (`let s = r;` where r is already tainted).
            # Pre-fix this aliasing silently dropped the taint —
            # a 2-line laundering vector. Unified via
            # _modal_origin_of_expr to keep all 3 install sites
            # (Let/Assign/PatBind) in sync.
            let_rhs_kind = (
                self._modal_origin_of_expr(stmt.value)
                if stmt.value is not None else None
            )
            if let_rhs_kind is not None:
                self._modal_origin_provenance[stmt.name] = let_rhs_kind
            else:
                # Pop any stale modal-origin entry. Same defect
                # class avoidance as the Result-provenance pop
                # above: without this, fn A's `let r =
                # from_uncertain(u)` would taint `r` in any
                # subsequent fn's parameter list. _check_fn entry
                # clear handles the cross-fn case; this pop
                # handles intra-fn re-binding.
                self._modal_origin_provenance.pop(stmt.name, None)
            # Stage 52 gate-1 F1e: record this let-introduction in
            # the current block's modal-origin let-set so block-
            # exit restore can distinguish inner-shadow (drop) from
            # inner-Assign (propagate). Tracks ALL lets regardless
            # of whether they install taint — a non-tainting
            # `let r = 42` STILL shadows any outer taint of r and
            # must restore outer's taint after the block ends.
            if self._modal_origin_let_block_scopes:
                self._modal_origin_let_block_scopes[-1].add(stmt.name)
            self._define_local_const_scalar(stmt.name, None)
            if (stmt.value is not None
                    and self._expr_has_unrepresentable_typed_const_scalar(
                        stmt.value)):
                if not self._mark_array_literal_unrepresentable_elements(
                        stmt.name, stmt.value):
                    self._mark_local_const_unrepresentable(
                        stmt.name,
                        self._expr_unrepresentable_typed_const_scalar_base(
                            stmt.value),
                    )
            return
        if isinstance(stmt, A.ExprStmt):
            self._check_expr(stmt.expr, scope)
            return
        if isinstance(stmt, A.ConstStmt):
            stmt_error_start = len(self.errors)
            ty = self._resolve_type(stmt.ty, scope)
            value_ty = self._check_expr(stmt.value, scope)
            if not self._compatible(value_ty, ty):
                self.errors.append(TypeError_(
                    f"const {stmt.name!r}: declared {self._fmt(ty)} "
                    f"but value is {self._fmt(value_ty)}",
                    stmt.span,
                ))
            elif len(self.errors) == stmt_error_start:
                self._check_refinement_contextual_value(
                    stmt.value, value_ty, ty, stmt.span,
                    f"const {stmt.name!r}",
                    scope,
                )
            bind_ty = ty
            if (self._contains_refinement(ty)
                    and len(self.errors) != stmt_error_start):
                bind_ty = self._erase_refinement(ty)
            scope.define(stmt.name, bind_ty)
            source_unrepresentable = (
                self._expr_has_unrepresentable_typed_const_scalar(stmt.value)
            )
            const_value = self._eval_const_scalar_expr(
                stmt.value, None, use_local_consts=True,
                honor_float_suffix=True,
                numeric_base=self._erase_refinement(ty))
            const_value = self._cast_const_scalar_to_type(
                const_value, self._erase_refinement(ty))
            if source_unrepresentable:
                self._define_local_const_scalar(stmt.name, None)
                self._mark_local_const_unrepresentable(
                    stmt.name,
                    self._expr_unrepresentable_typed_const_scalar_base(
                        stmt.value),
                )
            elif (len(self.errors) == stmt_error_start
                    and isinstance(const_value, (int, float))
                    and not isinstance(const_value, bool)):
                self._define_local_const_scalar(stmt.name, const_value)
            else:
                self._define_local_const_scalar(stmt.name, None)
            return

    def _check_expr(self, expr: A.Expr, scope: Scope) -> Type:
        if isinstance(expr, A.IntLit):
            # Default integer type is i32 unless suffix specified.
            # Stage 28.9 cycle 93 audit-T F1 fix (HIGH conf 85):
            # reject IntLit with float-domain suffix. Pre-fix
            # `42_f32` lexed as IntLit(value=42, type_suffix="f32"),
            # passed typecheck as TyPrim("f32"), lowered via
            # IRBuilder.const_int to CONST_INT(result_ty=TIRScalar
            # ("f32")), and x86_64 stored the raw int bit-pattern
            # 0x2A into the f32 slot (≈5.88e-44, not 42.0).
            if expr.type_suffix in _FLOAT_PRIM_NAMES:
                self.errors.append(TypeError_(
                    f"integer literal '{expr.value}' has float-domain "
                    f"suffix '{expr.type_suffix}'; use a float literal "
                    f"(e.g. {expr.value}.0_{expr.type_suffix}) "
                    f"or change the suffix to an integer type",
                    expr.span,
                ))
                return TyPrim("i32")
            return TyPrim(expr.type_suffix or "i32")
        if isinstance(expr, A.FloatLit):
            # Stage 28.9 cycle 93 audit-T F1 fix: symmetric check —
            # reject FloatLit with integer-domain suffix.
            if expr.type_suffix in _INT_PRIM_NAMES:
                self.errors.append(TypeError_(
                    f"float literal '{expr.value}' has integer-domain "
                    f"suffix '{expr.type_suffix}'; use an integer literal "
                    f"or change the suffix to a float type",
                    expr.span,
                ))
                return TyPrim("f32")
            return TyPrim(expr.type_suffix or "f32")
        if isinstance(expr, A.StrLit):
            return TyRef(TyPrim("char"), is_mut=False)  # &str-ish
        if isinstance(expr, A.CharLit):
            return TyPrim("char")
        if isinstance(expr, A.BoolLit):
            return TyPrim("bool")
        if isinstance(expr, A.Name):
            looked = scope.lookup(expr.name)
            if looked is not None:
                return looked
            const_decl = getattr(self, "_const_decls", {}).get(expr.name)
            if const_decl is not None:
                const_ty = self._resolve_type(const_decl.ty, Scope())
                if expr.name in getattr(self, "_invalid_const_names", set()):
                    return self._erase_refinement(const_ty)
                return const_ty
            # Function reference?
            if expr.name in self.functions:
                sig = self.functions[expr.name]
                ret = sig.ret
                if expr.name in getattr(
                        self, "_invalid_refined_return_functions", set()):
                    ret = self._erase_refinement(ret)
                return TyFn(tuple(t for _, t in sig.params), ret)
            enum_variant = self._enum_variant_for_expr(expr)
            if enum_variant is not None:
                ename, variant = enum_variant
                if variant.payload_tys:
                    self.errors.append(TypeError_(
                        f"enum variant {ename}::{variant.name} has "
                        f"payload - call as a function instead",
                        expr.span,
                    ))
                return TyEnum(name=ename)
            if expr.name in self._GPU_INDEX_BUILTINS:
                self.errors.append(TypeError_(
                    f"{expr.name} must be called as {expr.name}()",
                    expr.span,
                ))
                return TyUnknown(hint=f"bare builtin {expr.name}")
            # Name truly unbound. Emit a soft diagnostic with a Levenshtein
            # "did you mean?" suggestion drawn from in-scope names + known
            # functions. Don't raise — return TyUnknown so cascade-style
            # downstream errors are suppressed.
            self._unbound_name_suggestion(expr.name, expr.span, scope)
            return TyUnknown(hint=f"unbound {expr.name}")
        if isinstance(expr, A.Path):
            # Check for `EnumName::VariantName` paths.
            enum_variant = self._enum_variant_for_expr(expr)
            if enum_variant is not None:
                ename, variant = enum_variant
                if variant.payload_tys:
                    self.errors.append(TypeError_(
                        f"enum variant {ename}::{variant.name} has "
                        f"payload - call as a function instead",
                        expr.span,
                    ))
                return TyEnum(name=ename)
            if len(expr.segments) == 2:
                ename, vname = expr.segments
                if ename in getattr(self, "_enum_decls", {}):
                    self.errors.append(TypeError_(
                        f"enum {ename!r} has no variant {vname!r}",
                        expr.span,
                    ))
            # 3+-segment paths aren't yet supported — emit a clear error
            # so users don't silently get TyUnknown propagation that masks
            # downstream type errors.
            if len(expr.segments) >= 3:
                # Allow well-known stdlib-shaped 3+-paths through as
                # opaque (e.g., tensor::ops::matmul) without erroring.
                # For now: only flag enum-like paths that start with a
                # known struct/enum name and exceed 2 segments.
                first = expr.segments[0]
                if (first in getattr(self, "_enum_decls", {})
                        or first in getattr(self, "_struct_decls", {})):
                    self.errors.append(TypeError_(
                        f"path {'::'.join(expr.segments)} has 3+ segments; "
                        f"only `EnumName::Variant` (2 segments) is supported "
                        f"in v0.1",
                        expr.span,
                    ))
            # v0.1: other paths are unresolved (e.g., tensor::zeros).
            return TyUnknown(hint=f"path {'::'.join(expr.segments)}")
        # Builtins: detach, attach for D<T>; consolidate for memory tiers
        # (Recognized by name in Call expressions below.)
        if isinstance(expr, A.Unary):
            inner = self._check_expr(expr.operand, scope)
            if expr.op == "-":
                if not isinstance(inner, (TyUnknown, TyVar, TySize)) \
                        and not (self._is_int_scalar(inner)
                                 or self._is_float_scalar(inner)):
                    self.errors.append(TypeError_(
                        f"operator '-' does not support operand type "
                        f"{self._fmt(inner)}",
                        expr.span,
                    ))
                return inner
            if expr.op == "~":
                if not isinstance(inner, (TyUnknown, TyVar, TySize)) \
                        and not self._is_int_scalar(inner):
                    self.errors.append(TypeError_(
                        f"operator '~' does not support operand type "
                        f"{self._fmt(inner)}",
                        expr.span,
                    ))
                return inner
            if expr.op == "!":
                if not (isinstance(inner, TyPrim) and inner.name == "bool"):
                    if not isinstance(inner, (TyUnknown, TyVar, TySize)):
                        self.errors.append(TypeError_(
                            f"operator '!' expects bool operand, got "
                            f"{self._fmt(inner)}",
                            expr.span,
                        ))
                return TyPrim("bool")
            if expr.op in ("&", "&mut"):
                # Stage 31 safety hardening: address-of has always parsed,
                # but previously fell through as the operand type. Keep the
                # type surface honest while emitting a diagnostic until real
                # reference storage/address semantics land in lowering.
                if not isinstance(expr.operand, A.Name):
                    self.errors.append(TypeError_(
                        f"operator {expr.op!r} requires an addressable "
                        "named binding in Stage 31",
                        expr.span,
                        hint="bind the value with `let` before taking a "
                             "reference",
                    ))
                elif scope.lookup(expr.operand.name) is None:
                    self.errors.append(TypeError_(
                        f"operator {expr.op!r} requires a local binding; "
                        f"{expr.operand.name!r} is not addressable in "
                        "Stage 31",
                        expr.span,
                        hint="only local `let` bindings have reference "
                             "storage in this stage",
                    ))
                elif (expr.op == "&mut"
                      and not scope.lookup_mutable(expr.operand.name)):
                    self.errors.append(TypeError_(
                        f"cannot take mutable reference to immutable binding "
                        f"{expr.operand.name!r}",
                        expr.span,
                        hint="declare the binding with `let mut`",
                    ))
                else:
                    self.errors.append(TypeError_(
                        f"operator {expr.op!r} is type-known but not "
                        "lowerable yet in Stage 31",
                        expr.span,
                        hint="compiled reference storage needs a real IR "
                             "operation before this can pass check-only",
                    ))
                return TyRef(inner=inner, is_mut=(expr.op == "&mut"))
            if expr.op == "*":
                if isinstance(inner, TyPtr):
                    if self._in_unsafe_depth == 0:
                        self.errors.append(TypeError_(
                            "raw-pointer dereference outside unsafe block "
                            "(trap 28601)",
                            expr.span,
                            hint="wrap the dereference in `unsafe { ... }`",
                        ))
                    else:
                        self.errors.append(TypeError_(
                            "raw-pointer dereference is type-known but not "
                            "lowerable yet in Stage 31",
                            expr.span,
                            hint="compiled pointer loads need a real IR "
                                 "operation before this can pass check-only",
                    ))
                    return inner.inner
                if isinstance(inner, TyRef):
                    self.errors.append(TypeError_(
                        "reference dereference is type-known but not "
                        "lowerable yet in Stage 31",
                        expr.span,
                        hint="compiled reference loads need a real IR "
                             "operation before this can pass check-only",
                    ))
                    return inner.inner
                if isinstance(inner, (TyUnknown, TyVar, TySize)):
                    self.errors.append(TypeError_(
                        f"operator '*' cannot dereference unresolved "
                        f"operand type {self._fmt(inner)} in Stage 31",
                        expr.span,
                        hint="use an explicit pointer or reference type once "
                             "deref lowering exists",
                    ))
                    return TyUnknown(hint="deref")
                self.errors.append(TypeError_(
                    f"operator '*' expects pointer or reference operand, "
                    f"got {self._fmt(inner)}",
                    expr.span,
                ))
                return TyUnknown(hint="deref")
            self.errors.append(TypeError_(
                f"unsupported unary operator {expr.op!r}",
                expr.span,
            ))
            return TyUnknown(hint=f"unary {expr.op}")
        if isinstance(expr, A.Binary):
            l = self._check_expr(expr.left, scope)
            r = self._check_expr(expr.right, scope)
            if expr.op in ("&&", "||"):
                if not (isinstance(l, TyPrim) and l.name == "bool"):
                    self.errors.append(TypeError_(
                        f"operator {expr.op!r} expects bool left operand, "
                        f"got {self._fmt(l)}",
                        expr.span,
                    ))
                if not (isinstance(r, TyPrim) and r.name == "bool"):
                    self.errors.append(TypeError_(
                        f"operator {expr.op!r} expects bool right operand, "
                        f"got {self._fmt(r)}",
                        expr.span,
                    ))
                return TyPrim("bool")
            if expr.op in ("==", "!=", "<", "<=", ">", ">="):
                self._check_plain_binary_scalar_compat(l, r, expr.op, expr.span)
                return TyPrim("bool")
            # Differentiability + Logic provenance propagation.
            #
            # Audit 28.8 B2: the binop handler previously had an arm for
            # TyDiff but NOT for TyLogic — so `Logic<T> + T`, `T + Logic<T>`,
            # and `D<Logic<T>> + Logic<T>` silently stripped the Logic
            # wrapper. The docstring at line 132-148 documents
            # `D<Logic<T>>` as a *differentiable relational value* — the
            # Tier-3 moat for neuro-symbolic AGI — but without
            # propagation through arithmetic, every Logic value lost
            # its wrapper at first arithmetic touch.
            #
            # Compositional rule: D wraps Logic (per TyLogic docstring).
            # So if either operand is TyLogic AND the operand pair is
            # also TyDiff-bearing, the result is TyDiff(TyLogic(inner)).
            # If only one side is Logic-wrapped, propagate Logic.
            l_is_diff = isinstance(l, TyDiff)
            r_is_diff = isinstance(r, TyDiff)

            # Extract the innermost type beneath any TyDiff/TyLogic
            # wrappers, treating TyDiff(TyLogic(T)) and TyLogic(T) and
            # T uniformly.
            def _unwrap(t: Type) -> Type:
                if isinstance(t, TyDiff):
                    return _unwrap(t.inner)
                if isinstance(t, TyLogic):
                    return _unwrap(t.inner)
                return t

            l_is_logic = isinstance(l, TyLogic) or (
                isinstance(l, TyDiff) and isinstance(l.inner, TyLogic)
            )
            r_is_logic = isinstance(r, TyLogic) or (
                isinstance(r, TyDiff) and isinstance(r.inner, TyLogic)
            )

            if l_is_logic or r_is_logic or l_is_diff or r_is_diff:
                # Audit 28.8 B13 (trap AD002 / 24200): TyDiff binop
                # with mixed inner types previously silently coerced
                # the right operand to the left's inner type. The
                # docstring acknowledged "real compiler would unify
                # innerness" but no warning surfaced. Widen-then-warn
                # contract: result inner is the wider of the two
                # (float dominates int, larger width dominates),
                # emit a warning so the silent loss path is visible.
                #
                # Audit 28.8 cycle 2 B:C4: same-rank ties (e.g.
                # u32 vs i32) also emit AD002, with a hint about the
                # sign-domain transition.
                #
                # Audit 28.8 cycle 2 B:C6: asymmetric D<T> + bareT
                # also warns. Pre-fix the gate was `l_is_diff AND
                # r_is_diff`, so `D<f64> + i32` (one D-wrapped, one
                # raw) silently promoted i32 to f64. Now the gate is
                # `(l_is_diff OR r_is_diff) AND inner mismatch`.
                l_inner = _unwrap(l)
                r_inner = _unwrap(r)
                if not self._check_wrapped_binary_operator_domain(
                        l_inner, r_inner, expr.op, expr.span):
                    return TyUnknown(hint="wrapped binary")

                # Cycle 3 C3-2: dedup AD002 emission. The tie callback
                # fires the same-rank-tie message; the outer call would
                # otherwise fire a second, less-specific mismatch warn
                # for the same span. Track and skip the outer when the
                # callback already spoke.
                #
                # Audit 28.8 cycle 5 C4-8 / LOW: the tie-callback fires
                # the warn without knowing whether the binop is happening
                # in D-domain or Logic-domain. Pre-fix the Logic-domain
                # tie case dropped the `[Logic-domain]` suffix because
                # the callback emits and then sets tie_fired=True,
                # suppressing the outer Logic-tagged emit. We now pass
                # a mutable flag `logic_domain_active` that the Logic
                # branch sets to True so the callback can append the
                # `[Logic-domain]` suffix when the tie fires inside that
                # branch.
                tie_fired = [False]
                logic_domain_active = [False]

                def _tie_cb(a, b, span):
                    tie_fired[0] = True
                    suffix = ""
                    if logic_domain_active[0]:
                        suffix = " [Logic-domain]"
                    self._ad_warn_mixed_inner(
                        span or expr.span, a, b, a,
                        extra=" (same-rank tie; sign or quant domain "
                              "silently dropped without this warning)"
                              + suffix,
                    )

                # Cycle 3 C3-2: treat pointer-width aliases as same inner.
                inner_mismatch = (
                    l_inner != r_inner
                    and not isinstance(l_inner, TyUnknown)
                    and not isinstance(r_inner, TyUnknown)
                    and not (
                        isinstance(l_inner, TyPrim)
                        and isinstance(r_inner, TyPrim)
                        and _widen_canon_name(l_inner.name)
                            == _widen_canon_name(r_inner.name)
                    )
                )
                if (l_is_diff or r_is_diff) and (
                        inner_mismatch
                        or (l_is_diff != r_is_diff)
                ):
                    # Audit 28.8 cycle 6 F2: D<T> + T same-inner asymmetric
                    # wrap also warns. Pre-fix the cycle-2 B:C6 D-vs-bare
                    # gate required `inner_mismatch`; same-inner pair
                    # `D<f64> + f64` silently produced `D<f64>` with no
                    # diagnostic. Now fires whenever exactly one side
                    # carries D — symmetric with cycle-4 E2 for Logic.
                    inner = _widen_diff_inner(
                        l_inner, r_inner,
                        _warn_cb=_tie_cb, _span=expr.span,
                    )
                    if not tie_fired[0]:
                        extra = ""
                        if not (l_is_diff and r_is_diff):
                            # B:C6: name the asymmetric (D-wrap + raw) case
                            # so the user can tell it apart from D-D mixing.
                            # Cycle 7 G1: if the other side is Logic-wrapped,
                            # say "Logic-wrapped" not "bare" so the
                            # diagnostic doesn't mis-identify the pair.
                            other_is_logic = (
                                (l_is_diff and r_is_logic)
                                or (r_is_diff and l_is_logic)
                            )
                            if other_is_logic:
                                extra = " (one side D-wrapped, other Logic-wrapped)"
                            else:
                                extra = " (one side D-wrapped, other bare)"
                        self._ad_warn_mixed_inner(
                            expr.span, l_inner, r_inner, inner, extra=extra,
                        )
                elif (l_is_logic or r_is_logic) and (
                        inner_mismatch
                        or (l_is_logic != r_is_logic)
                ):
                    # Audit 28.8 cycle 3 D4: pure Logic<T1> + Logic<T2>
                    # (neither side TyDiff) — previously silent
                    # left-wins. The TyLogic docstring explicitly says
                    # Logic carries provenance; silently dropping the
                    # right-side domain defeats that contract.
                    #
                    # Audit 28.8 cycle 4 E2: the wrap-asymmetric case
                    # `Logic<T> + T` (one side Logic, other bare, SAME
                    # inner) was missed by D4 because the gate required
                    # inner_mismatch. Now we also fire when l_is_logic
                    # != r_is_logic regardless of inner — symmetric
                    # with the cycle-2 B:C6 D-vs-bare fix.
                    #
                    # Audit 28.8 cycle 5 C4-8 / LOW: thread the
                    # Logic-domain marker through the tie-callback so
                    # the tie-warn (when it fires) also carries the
                    # `[Logic-domain]` suffix. Pre-fix this suffix was
                    # dropped because the callback fired first.
                    logic_domain_active[0] = True
                    inner = _widen_diff_inner(
                        l_inner, r_inner,
                        _warn_cb=_tie_cb, _span=expr.span,
                    )
                    logic_domain_active[0] = False
                    if not tie_fired[0]:
                        extra = ""
                        if not (l_is_logic and r_is_logic):
                            extra = " (one side Logic-wrapped, other bare)"
                        self._ad_warn_mixed_inner(
                            expr.span, l_inner, r_inner, inner,
                            extra=" [Logic-domain]" + extra,
                        )
                else:
                    inner = l_inner if not isinstance(l_inner, TyUnknown) \
                        else r_inner
                # Build the wrapping: Logic first (if any side carries
                # Logic), then D on the outside (if any side carries D).
                wrapped: Type = inner
                if l_is_logic or r_is_logic:
                    wrapped = TyLogic(inner=wrapped)
                if l_is_diff or r_is_diff:
                    wrapped = TyDiff(inner=wrapped)
                return wrapped
            # Arithmetic: take the left type (simplified)
            self._check_plain_binary_scalar_compat(l, r, expr.op, expr.span)
            return self._erase_refinement(l)
        if isinstance(expr, A.Call):
            # Stage 16.5: "literal".as_ptr() — type is *const u8 (TyPtr(u8, mut=False)).
            if (isinstance(expr.callee, A.Field)
                    and expr.callee.name == "as_ptr"
                    and isinstance(expr.callee.obj, A.StrLit)
                    and len(expr.args) == 0):
                return TyPtr(inner=TyPrim("u8"), is_mut=False)
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name in self._GPU_INDEX_BUILTINS
                    # Stage 40 closure gate-3 type-design F1 fix
                    # (HIGH conf 88): mirror the H2 dispatch
                    # suppression at the modal/temporal/frame/tier
                    # site (line 2848). Without this check, the
                    # early GPU-index dispatch silently shadowed
                    # the user fn even when _register_fn had
                    # already flagged it — pre-fix, `fn thread_idx`
                    # produced 1 shadow error + N noisy "only
                    # allowed inside @kernel" errors per call. H2
                    # invariant applied uniformly: if shadowed,
                    # skip builtin dispatch and fall through to
                    # user-fn lookup.
                    and expr.callee.name
                        not in self._shadowed_builtin_names):
                bn = expr.callee.name
                arg_tys = [self._check_expr(a, scope) for a in expr.args]
                if arg_tys:
                    self.errors.append(TypeError_(
                        f"{bn}() expects 0 args, got {len(arg_tys)}",
                        expr.span,
                    ))
                if not self._current_is_kernel:
                    self.errors.append(TypeError_(
                        f"{bn}() is only allowed inside @kernel functions",
                        expr.span,
                    ))
                return TyPrim("i32")
            # Payload-bearing enum constructor: `Maybe::Some(42)`.
            enum_variant = self._enum_variant_for_expr(expr.callee)
            if enum_variant is not None:
                ename, variant = enum_variant
                # Type-check args against payload_tys.
                arg_tys = [self._check_expr(a, scope) for a in expr.args]
                if len(arg_tys) != len(variant.payload_tys):
                    self.errors.append(TypeError_(
                        f"enum variant {ename}::{variant.name} expects "
                        f"{len(variant.payload_tys)} payload arg(s), "
                        f"got {len(arg_tys)}",
                        expr.span,
                    ))
                else:
                    for i, (at, pt) in enumerate(
                            zip(arg_tys, variant.payload_tys)):
                        expected = self._resolve_type(pt, scope)
                        if not self._compatible(at, expected):
                            self.errors.append(TypeError_(
                                f"enum {ename}::{variant.name} arg {i}: "
                                f"expected {self._fmt(expected)}, "
                                f"got {self._fmt(at)}",
                                expr.span,
                            ))
                        else:
                            self._check_refinement_contextual_value(
                                expr.args[i], at, expected,
                                expr.args[i].span,
                                f"enum {ename}::{variant.name} arg {i}",
                                scope,
                            )
                return TyEnum(name=ename)
            callee = self._check_expr(expr.callee, scope)
            arg_tys = [self._check_expr(a, scope) for a in expr.args]
            # Built-in functions for type-level transitions
            if isinstance(expr.callee, A.Name):
                bn = expr.callee.name
                # Stage 40 closure gate-2 H2 fix: if the bare name
                # has been shadowed by a user fn, suppress builtin
                # dispatch entirely so the call site falls through
                # to user-fn lookup. The user sees the ONE shadow
                # diagnostic at the fn-decl site (not 1 shadow + N
                # noisy false-positive builtin errors per call).
                if bn in self._shadowed_builtin_names:
                    bn = "<<shadowed_builtin_skip>>"
                if bn == "detach" and len(arg_tys) == 1:
                    if isinstance(arg_tys[0], TyDiff):
                        return arg_tys[0].inner
                    return arg_tys[0]
                if bn == "attach" and len(arg_tys) == 1:
                    if isinstance(arg_tys[0], TyDiff):
                        return arg_tys[0]
                    return TyDiff(inner=arg_tys[0])
                # Stage 36 Increment 1: provenance-typed primitives.
                # prove(value: T, source: i32) -> Logic<T> — wraps a
                # bare value with a provenance tag (an i32 identifier
                # into a user-managed source table). The Logic<T>
                # wrapper is representationally identical to T at the
                # IR level in Phase-0; provenance lives purely in the
                # type system (lattice/semiring upgrade is reserved
                # for later increments).
                if bn == "prove" and len(arg_tys) == 2:
                    # Stage 36 Inc 11 post-Inc-10 audit C2 LOW fix:
                    # tighten source-tag arg from `_is_int_scalar`
                    # (which accepts i32/i64/u32/u64) to strict i32 —
                    # same family as the Inc 9 B4 fix on `to_logic_bool`.
                    # Pre-fix, prove(v, x_i64) silently passed; the
                    # downstream BIT_AND in source-tag handling would
                    # drop the upper 32 bits.
                    if not (isinstance(arg_tys[1], TyPrim)
                            and arg_tys[1].name == "i32"):
                        self.errors.append(TypeError_(
                            f"prove(value, source): source must be exactly "
                            f"i32, got {self._fmt(arg_tys[1])} (pre-Inc-11 "
                            f"also accepted i64/u32/u64 but those silently "
                            f"truncated in downstream BIT_AND ops)",
                            expr.span,
                        ))
                    inner = arg_tys[0]
                    # Stage 36 Inc 9 catch-up — type-design B1 fix:
                    # reject prove(Logic<T>, src) instead of silently
                    # flattening to the input Logic<T>. The pre-fix
                    # `if isinstance(inner, TyLogic): return inner`
                    # dropped the new source tag — a programmer who
                    # wrapped twice to record additional evidence lost
                    # it. Phase-0 representation is single-tag, so
                    # wrap-and-keep-both would require an ABI change;
                    # the conservative Phase-0 fix is to reject so the
                    # user knows to unwrap_logic(...) first.
                    if isinstance(inner, TyLogic):
                        self.errors.append(TypeError_(
                            f"prove(value, source): value is already "
                            f"Logic<...> ({self._fmt(inner)}); call "
                            f"unwrap_logic(...) first if re-proving "
                            f"with a new source tag (Phase-0 single-"
                            f"tag provenance cannot stack)",
                            expr.span,
                        ))
                        return TyUnknown(hint="prove")
                    return TyLogic(inner=inner)
                # unwrap_logic(l: Logic<T>) -> T — strips the Logic
                # wrapper. Provenance information is discarded; this
                # is the only legal way to escape the Logic<T> trap-
                # 24100 boundary check, so callers explicitly
                # acknowledge they are abandoning the evidence trail.
                if bn == "unwrap_logic" and len(arg_tys) == 1:
                    if isinstance(arg_tys[0], TyLogic):
                        return arg_tys[0].inner
                    self.errors.append(TypeError_(
                        f"unwrap_logic() requires Logic<T>, got "
                        f"{self._fmt(arg_tys[0])}",
                        expr.span,
                    ))
                    # Stage 36 Inc 9 audit C1 (type-design lane) fix:
                    # error-recovery returned arg_tys[0] (the
                    # non-Logic input) which cascaded misleading
                    # downstream errors. Return TyUnknown so the
                    # error stays local to the call site.
                    return TyUnknown(hint="unwrap_logic")
                # Stage 36 Increment 2: provenance-composing combinators.
                # derive(a: Logic<T>, b: Logic<U>) -> Logic<T> — propagates
                # provenance through a binary derivation step. The result
                # carries the value of `a` (Phase-0 single-tag provenance;
                # the lattice/semiring upgrade tracks BOTH parents). Both
                # inputs must already be Logic-wrapped — passing a bare T
                # is a trap-24100 boundary violation.
                if bn == "derive" and len(arg_tys) == 2:
                    if not isinstance(arg_tys[0], TyLogic):
                        self.errors.append(TypeError_(
                            f"derive(a, b): arg a must be Logic<T>, got "
                            f"{self._fmt(arg_tys[0])} [trap 24100]",
                            expr.span,
                        ))
                    if not isinstance(arg_tys[1], TyLogic):
                        self.errors.append(TypeError_(
                            f"derive(a, b): arg b must be Logic<T>, got "
                            f"{self._fmt(arg_tys[1])} [trap 24100]",
                            expr.span,
                        ))
                    if isinstance(arg_tys[0], TyLogic):
                        return arg_tys[0]
                    # Stage 36 Inc 9 catch-up — type-design C2 fix:
                    # pre-fix recovery returned TyLogic(inner=arg_tys[0])
                    # which wrapped a non-Logic input into Logic<NonLogic>,
                    # masking the inner-type mismatch in chained calls.
                    # TyUnknown keeps the error local to derive's call
                    # site (matches the C1 fix on unwrap_logic).
                    return TyUnknown(hint="derive")
                # Stage 36 Inc 9 audit A1 HIGH (type-design lane) fix:
                # tighten boolean ops to require Logic<i32> inner. The
                # pre-fix isinstance(t, TyLogic) check accepted any
                # Logic<T>, then unconditionally lowered to BIT_AND/OR
                # which is wrong for non-i32 operands. The fuzzy-op
                # block below got the symmetric fix (require Logic<f32>).
                def _is_logic_of(ty, prim_name):
                    return (isinstance(ty, TyLogic)
                            and isinstance(ty.inner, TyPrim)
                            and ty.inner.name == prim_name)

                # and_logic(a: Logic<i32>, b: Logic<i32>) -> Logic<i32>
                # — boolean AND on provenance-tagged truth values.
                if bn == "and_logic" and len(arg_tys) == 2:
                    for i, t in enumerate(arg_tys):
                        if not _is_logic_of(t, "i32"):
                            self.errors.append(TypeError_(
                                f"and_logic(a, b): arg {'ab'[i]} must be "
                                f"Logic<i32>, got {self._fmt(t)} [trap 24100]",
                                expr.span,
                            ))
                    if (_is_logic_of(arg_tys[0], "i32")
                            and _is_logic_of(arg_tys[1], "i32")):
                        return arg_tys[0]
                    return TyLogic(inner=TyPrim("i32"))
                # or_logic(a: Logic<i32>, b: Logic<i32>) -> Logic<i32>
                if bn == "or_logic" and len(arg_tys) == 2:
                    for i, t in enumerate(arg_tys):
                        if not _is_logic_of(t, "i32"):
                            self.errors.append(TypeError_(
                                f"or_logic(a, b): arg {'ab'[i]} must be "
                                f"Logic<i32>, got {self._fmt(t)} [trap 24100]",
                                expr.span,
                            ))
                    if (_is_logic_of(arg_tys[0], "i32")
                            and _is_logic_of(arg_tys[1], "i32")):
                        return arg_tys[0]
                    return TyLogic(inner=TyPrim("i32"))
                # not_logic(a: Logic<i32>) -> Logic<i32>
                if bn == "not_logic" and len(arg_tys) == 1:
                    if not _is_logic_of(arg_tys[0], "i32"):
                        self.errors.append(TypeError_(
                            f"not_logic(a): arg must be Logic<i32>, got "
                            f"{self._fmt(arg_tys[0])} [trap 24100]",
                            expr.span,
                        ))
                    if _is_logic_of(arg_tys[0], "i32"):
                        return arg_tys[0]
                    return TyLogic(inner=TyPrim("i32"))
                # Stage 36 Increment 3: boolean-algebra completeness.
                # xor_logic / implies_logic / eq_logic all require
                # Logic<i32> (consistent with and/or/not above).
                if bn in ("xor_logic", "implies_logic", "eq_logic") \
                        and len(arg_tys) == 2:
                    for i, t in enumerate(arg_tys):
                        if not _is_logic_of(t, "i32"):
                            self.errors.append(TypeError_(
                                f"{bn}(a, b): arg {'ab'[i]} must be "
                                f"Logic<i32>, got {self._fmt(t)} [trap 24100]",
                                expr.span,
                            ))
                    if (_is_logic_of(arg_tys[0], "i32")
                            and _is_logic_of(arg_tys[1], "i32")):
                        return arg_tys[0]
                    return TyLogic(inner=TyPrim("i32"))
                # if_logic(cond: Logic<i32>, then_val: Logic<T>,
                #          else_val: Logic<T>) -> Logic<T> — provenance-
                # typed ternary. Returns then_val when cond's value is
                # nonzero, else else_val. All three inputs must be
                # Logic-wrapped; cond must be Logic<i32>; then_val and
                # else_val must share the same inner type (Stage 36
                # Inc 11 post-Inc-10 audit B1 fix — pre-fix accepted
                # mismatched inner types and silently picked then_val's
                # type, type-punning the result).
                if bn == "if_logic" and len(arg_tys) == 3:
                    for i, t in enumerate(arg_tys):
                        if not isinstance(t, TyLogic):
                            self.errors.append(TypeError_(
                                f"if_logic(cond, then_v, else_v): arg "
                                f"{['cond', 'then_v', 'else_v'][i]} must "
                                f"be Logic<...>, got {self._fmt(t)} "
                                f"[trap 24100]",
                                expr.span,
                            ))
                    if not _is_logic_of(arg_tys[0], "i32"):
                        self.errors.append(TypeError_(
                            f"if_logic(cond, then_v, else_v): cond must "
                            f"be Logic<i32>, got {self._fmt(arg_tys[0])} "
                            f"[trap 24100]",
                            expr.span,
                        ))
                    if (isinstance(arg_tys[1], TyLogic)
                            and isinstance(arg_tys[2], TyLogic)
                            and arg_tys[1].inner is not None
                            and arg_tys[2].inner is not None
                            and self._fmt(arg_tys[1].inner)
                                != self._fmt(arg_tys[2].inner)):
                        self.errors.append(TypeError_(
                            f"if_logic(cond, then_v, else_v): then and "
                            f"else inner types must match, got "
                            f"{self._fmt(arg_tys[1])} vs "
                            f"{self._fmt(arg_tys[2])} [trap 24100]",
                            expr.span,
                        ))
                    if isinstance(arg_tys[1], TyLogic):
                        return arg_tys[1]
                    if isinstance(arg_tys[2], TyLogic):
                        return arg_tys[2]
                    return TyLogic(inner=TyPrim("i32"))
                # to_logic_bool(x: i32) -> Logic<i32> — convenience: lift
                # a bare 0/1 truth value into Logic<i32> with provenance
                # tag 0 (anonymous). Equivalent to `prove(x, 0)`; named
                # for clarity at boolean-algebra entry points.
                #
                # Stage 36 Inc 9 audit B4 (type-design lane) fix:
                # tighten from `_is_int_scalar` (which accepts i32/i64/
                # u32/u64) to strict i32. Pre-fix, passing i64 would
                # silently produce Logic<i32> wrapping i64 data, and
                # downstream BIT_AND would drop the upper 32 bits.
                if bn == "to_logic_bool" and len(arg_tys) == 1:
                    if not (isinstance(arg_tys[0], TyPrim)
                            and arg_tys[0].name == "i32"):
                        hint = self._strict_i32_truncation_hint(
                            arg_tys[0], "pre-Inc-9", "BIT_AND ops")
                        self.errors.append(TypeError_(
                            f"to_logic_bool(x): arg must be exactly i32, got "
                            f"{self._fmt(arg_tys[0])}{hint}",
                            expr.span,
                        ))
                    return TyLogic(inner=TyPrim("i32"))
                # Stage 36 Increment 5: real two-parent provenance via
                # arena side-table. register_derivation(left_src,
                # right_src) writes the pair to the global arena and
                # returns the index where `left_src` was written. The
                # user keeps this index as the "derivation handle" and
                # later queries `parent_left_at(idx)` / `parent_right_at
                # (idx)` to recover the source IDs. This is genuine
                # two-parent tracking without an ABI change.
                if bn == "register_derivation" and len(arg_tys) == 2:
                    # Stage 36 Inc 11 post-Inc-10 audit C1 LOW fix:
                    # tighten both source-id args from `_is_int_scalar`
                    # to strict i32 — same family as the C2 prove() fix
                    # above and the Inc 9 B4 fix on `to_logic_bool`.
                    for i, t in enumerate(arg_tys):
                        if not (isinstance(t, TyPrim) and t.name == "i32"):
                            hint = self._strict_i32_truncation_hint(
                                t, "pre-Inc-11", "arena push ops")
                            self.errors.append(TypeError_(
                                f"register_derivation(left, right): arg "
                                f"{'12'[i]} must be exactly i32 source id, "
                                f"got {self._fmt(t)}{hint}",
                                expr.span,
                            ))
                    return TyPrim("i32")
                if bn in ("parent_left_at", "parent_right_at") \
                        and len(arg_tys) == 1:
                    # Stage 36 Inc 15 (type-design M1): tighten to strict
                    # i32 to match register_derivation (Inc 11 C1),
                    # register_derivation3 + parent_at (Inc 14). Pre-fix
                    # the loose _is_int_scalar accepted i64/u32/u64 which
                    # then silently truncated in the downstream arena
                    # read (same bug class as the Inc 11 C1 register-side
                    # fix). Family is now uniformly strict-i32.
                    if not (isinstance(arg_tys[0], TyPrim)
                            and arg_tys[0].name == "i32"):
                        hint = self._strict_i32_truncation_hint(
                            arg_tys[0], "pre-Inc-15", "arena read")
                        self.errors.append(TypeError_(
                            f"{bn}(idx): arg must be exactly i32 "
                            f"derivation handle, got "
                            f"{self._fmt(arg_tys[0])}{hint}",
                            expr.span,
                        ))
                    return TyPrim("i32")
                # Stage 36 Inc 14: three-parent provenance.
                # register_derivation3(left, middle, right: i32) -> i32
                # writes the triple atomically via ARENA_PUSH_TRIPLE and
                # returns a 1-based handle (Inc 9 A2 invariant). The
                # left slot is the handle's base; middle lives at slot+1,
                # right at slot+2.
                if bn == "register_derivation3" and len(arg_tys) == 3:
                    for i, t in enumerate(arg_tys):
                        if not (isinstance(t, TyPrim) and t.name == "i32"):
                            hint = self._strict_i32_truncation_hint(
                                t, "pre-Inc-14", "arena push ops")
                            self.errors.append(TypeError_(
                                f"register_derivation3(left, middle, right): "
                                f"arg {'123'[i]} must be exactly i32 source "
                                f"id, got {self._fmt(t)}{hint}",
                                expr.span,
                            ))
                    return TyPrim("i32")
                # Stage 36 Inc 14: generic indexed parent accessor.
                # parent_at(handle: i32, slot: i32) -> i32 reads the
                # arena slot at (handle - 1 + slot), with the same Inc 9
                # A1 bounds-check sentinel (-1 on OOB).
                #
                # Stage 36 Inc 15 (silent-failure H1, partial closure):
                # statically reject literal `slot < 0` or `slot > 2`.
                # Max arity is 3 (register_derivation3); a literal
                # outside [0, 2] is provably unreachable for any current
                # register_derivation* call. Dynamic slots still flow
                # through; the runtime guard at lower_ast.py covers
                # `handle <= 0` + dynamic `slot < 0` paths.
                # TODO(stage36-inc16-arity-in-handle): the remaining
                # cross-record hazard (literal slot=2 on a 2-parent
                # handle silently reads the next record's left value)
                # requires a per-record arity word in the arena layout
                # — too large for this audit-fix increment. See audit
                # docs/audit-stage36-postinc14-silent-failures.md#H1.
                if bn == "parent_at" and len(arg_tys) == 2:
                    # Stage 37 post-closure M2 fix (gate-3 type-design
                    # audit, conf 70): parent_at was the only member of
                    # the strict-i32 family without the family-standard
                    # "pre-Inc-N also accepted i64/u32/u64 but silently
                    # truncated" remediation hint. Use the same gated
                    # helper as the rest of the family.
                    #
                    # Stage 38 post-Inc-3 CR-003 fix (LOW, conf 82):
                    # parent_at was introduced in Inc 14 using the loose
                    # `_is_int_scalar` predicate, then tightened to
                    # strict i32 in Inc 15. So the era during which it
                    # silently truncated wider integers is exactly Inc
                    # 14, and the strict-enforcement boundary is Inc
                    # 15 — `pre-Inc-15` is the accurate hint label.
                    arg_types_ok = True
                    for i, t in enumerate(arg_tys):
                        if not (isinstance(t, TyPrim) and t.name == "i32"):
                            arg_types_ok = False
                            hint = self._strict_i32_truncation_hint(
                                t, "pre-Inc-15", "arena read")
                            self.errors.append(TypeError_(
                                f"parent_at(handle, slot): arg "
                                f"{'12'[i]} must be exactly i32, "
                                f"got {self._fmt(t)}{hint}",
                                expr.span,
                            ))
                    # Inc 15 static slot-literal bounds check.
                    # Stage 37 post-closure M2 fix (gate-3 type-design
                    # audit secondary smell): skip the slot-bounds error
                    # when arg types already failed — otherwise a caller
                    # who passes both wrong-typed args AND an out-of-
                    # range literal slot gets 3 errors when 2 suffice
                    # (the slot value is moot when the call can't
                    # typecheck). Gate on arg_types_ok.
                    if arg_types_ok:
                        slot_node = expr.args[1]
                        slot_literal_value: Optional[int] = None
                        if isinstance(slot_node, A.IntLit):
                            slot_literal_value = slot_node.value
                        elif (isinstance(slot_node, A.Unary)
                                and slot_node.op == "-"
                                and isinstance(slot_node.operand, A.IntLit)):
                            slot_literal_value = -slot_node.operand.value
                        if (slot_literal_value is not None
                                and not (0 <= slot_literal_value <= 2)):
                            self.errors.append(TypeError_(
                                f"parent_at(handle, slot): literal slot "
                                f"{slot_literal_value} is out of range "
                                f"[0, 2] (max arity is 3 from "
                                f"register_derivation3). Inc 15 "
                                f"silent-failure H1 closure.",
                                expr.span,
                            ))
                    return TyPrim("i32")
                # Stage 36 Increment 6: fuzzy logic operators over
                # Logic<f32>. Truth values live in [0, 1]; operators
                # use product semantics so they're smooth and
                # differentiable: fuzzy_and = a*b, fuzzy_or = a+b-a*b,
                # fuzzy_not = 1-a. Because they lower to MUL/ADD/SUB
                # (which already have AD chain rules), grad() flows
                # through them automatically — the bridge to neuro-
                # symbolic AD without overhauling the AD passes.
                if bn in ("fuzzy_and", "fuzzy_or") and len(arg_tys) == 2:
                    for i, t in enumerate(arg_tys):
                        if not _is_logic_of(t, "f32"):
                            self.errors.append(TypeError_(
                                f"{bn}(a, b): arg {'ab'[i]} must be "
                                f"Logic<f32>, got {self._fmt(t)} "
                                f"[trap 24100]",
                                expr.span,
                            ))
                    if (_is_logic_of(arg_tys[0], "f32")
                            and _is_logic_of(arg_tys[1], "f32")):
                        return arg_tys[0]
                    return TyLogic(inner=TyPrim("f32"))
                if bn == "fuzzy_not" and len(arg_tys) == 1:
                    if not _is_logic_of(arg_tys[0], "f32"):
                        self.errors.append(TypeError_(
                            f"fuzzy_not(a): arg must be Logic<f32>, got "
                            f"{self._fmt(arg_tys[0])} [trap 24100]",
                            expr.span,
                        ))
                    if _is_logic_of(arg_tys[0], "f32"):
                        return arg_tys[0]
                    return TyLogic(inner=TyPrim("f32"))
                # Stage 36 Increment 8: round out the fuzzy algebra.
                # fuzzy_xor(a, b) = a + b - 2*a*b  (probabilistic XOR)
                # fuzzy_implies(a, b) = 1 - a + a*b (Reichenbach implication)
                # Both compose to MUL/ADD/SUB and are auto-differentiable
                # via the chain rules added in Inc 8 (autodiff.py +
                # autodiff_reverse.py).
                if bn in ("fuzzy_xor", "fuzzy_implies") \
                        and len(arg_tys) == 2:
                    for i, t in enumerate(arg_tys):
                        if not _is_logic_of(t, "f32"):
                            self.errors.append(TypeError_(
                                f"{bn}(a, b): arg {'ab'[i]} must be "
                                f"Logic<f32>, got {self._fmt(t)} "
                                f"[trap 24100]",
                                expr.span,
                            ))
                    if (_is_logic_of(arg_tys[0], "f32")
                            and _is_logic_of(arg_tys[1], "f32")):
                        return arg_tys[0]
                    return TyLogic(inner=TyPrim("f32"))
                # Stage 37 Inc 1 — tiered memory constructors + eliminators.
                # 4 memory tiers (working/episodic/semantic/procedural)
                # each get an `into_*` constructor (T -> TierMem<T>) and
                # an `unwrap_*` eliminator (TierMem<T> -> T). All lower
                # to identity at IR (Phase-0: zero runtime overhead,
                # tier lives purely in the type system — mirrors the
                # Stage 36 Inc 1 Logic<T>/prove pattern). The existing
                # consolidate/recall cross-tier transitions stay
                # unchanged.
                _tier_intro_elim = {
                    "into_working": "working",
                    "into_episodic": "episodic",
                    "into_semantic": "semantic",
                    "into_procedural": "procedural",
                }
                if bn in _tier_intro_elim and len(arg_tys) == 1:
                    # Stage 43 Inc 1 M1 fix: reject already-wrapped
                    # tier value. `into_working(WorkingMem<i32>)` ->
                    # `WorkingMem<WorkingMem<i32>>` was silently
                    # accepted (gate-1 M1 across all 5 wrapper
                    # families). Closes the symmetric pattern.
                    # Stage 43 closure gate-1 MEDIUM fix: name the
                    # tier transitions explicitly (consolidate /
                    # recall / learn_to) to give tier users parity
                    # with the frame/temporal/modal/causal hints.
                    if isinstance(arg_tys[0], TyMemTier):
                        target_tier = _tier_intro_elim[bn]
                        source_tier = arg_tys[0].tier
                        if source_tier == "episodic" and target_tier == "semantic":
                            transition_hint = (
                                "use `consolidate(m)` — the audited "
                                "Episodic -> Semantic transition"
                            )
                        elif source_tier == "semantic" and target_tier == "working":
                            transition_hint = (
                                "use `recall(m)` — the audited "
                                "Semantic -> Working transition"
                            )
                        else:
                            transition_hint = (
                                "Phase-0 tier transitions: "
                                "`consolidate(EpisodicMem<T>) "
                                "-> SemanticMem<T>` and "
                                "`recall(SemanticMem<T>) -> "
                                "WorkingMem<T>`; for other "
                                "directions, unwrap with "
                                f"unwrap_{source_tier} first"
                            )
                        self.errors.append(TypeError_(
                            f"{bn}() received an already-wrapped "
                            f"{self._fmt(arg_tys[0])}; intro "
                            f"builtins are not idempotent — "
                            f"{transition_hint}.",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    return TyMemTier(tier=_tier_intro_elim[bn],
                                     inner=arg_tys[0])
                _tier_unwrap = {
                    "unwrap_working": "working",
                    "unwrap_episodic": "episodic",
                    "unwrap_semantic": "semantic",
                    "unwrap_procedural": "procedural",
                }
                if bn in _tier_unwrap and len(arg_tys) == 1:
                    want = _tier_unwrap[bn]
                    if (isinstance(arg_tys[0], TyMemTier)
                            and arg_tys[0].tier == want):
                        return arg_tys[0].inner
                    self.errors.append(TypeError_(
                        f"{bn}() requires "
                        f"{want.capitalize()}Mem<T>, got "
                        f"{self._fmt(arg_tys[0])}",
                        expr.span,
                    ))
                    return TyUnknown(hint=bn)
                # Stage 38 Inc 1 — spatial-frame constructors + eliminators.
                # 3 reference frames (world/robot/camera) each get an
                # into_* constructor (T -> FrameName<T>) and a from_*
                # eliminator (FrameName<T> -> T). Mirrors the Stage 37
                # tier pattern. All lower to identity at IR (Phase-0:
                # frame lives at type level, zero runtime overhead).
                _frame_intro = {
                    "into_world": "world",
                    "into_robot": "robot",
                    "into_camera": "camera",
                }
                # Stage 38 post-Inc-3 silent-failure F1 fix (HIGH, conf 95):
                # gate on name FIRST, then arity, so wrong-arity calls emit
                # a diagnostic instead of falling through to TyUnknown.
                if bn in _frame_intro:
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    # Stage 43 Inc 1 M1 fix: reject already-wrapped
                    # frame value. `into_world(WorldFrame<i32>)` ->
                    # `WorldFrame<WorldFrame<i32>>` silently accepted
                    # pre-fix.
                    # Stage 43 closure gate-1 MEDIUM fix: compute the
                    # direction-correct transition for the actual
                    # (source, target) frame pair rather than hard-
                    # coding `world_to_robot` (which goes the wrong
                    # direction half the time).
                    if isinstance(arg_tys[0], TyFrame):
                        target_frame = _frame_intro[bn]
                        source_frame = arg_tys[0].frame
                        if source_frame == target_frame:
                            transition_hint = (
                                f"unwrap with from_{source_frame} "
                                f"first if you really want to "
                                f"re-tag the value"
                            )
                        else:
                            transition_hint = (
                                f"use `{source_frame}_to_"
                                f"{target_frame}` — the "
                                f"direction-correct cross-frame "
                                f"transform for this pair"
                            )
                        self.errors.append(TypeError_(
                            f"{bn}() received an already-wrapped "
                            f"{self._fmt(arg_tys[0])}; intro "
                            f"builtins are not idempotent — "
                            f"{transition_hint}.",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    return TyFrame(frame=_frame_intro[bn],
                                   inner=arg_tys[0])
                _frame_elim = {
                    "from_world": "world",
                    "from_robot": "robot",
                    "from_camera": "camera",
                }
                if bn in _frame_elim:
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    want = _frame_elim[bn]
                    if (isinstance(arg_tys[0], TyFrame)
                            and arg_tys[0].frame == want):
                        return arg_tys[0].inner
                    self.errors.append(TypeError_(
                        f"{bn}() requires "
                        f"{want.capitalize()}Frame<T>, got "
                        f"{self._fmt(arg_tys[0])}",
                        expr.span,
                    ))
                    return TyUnknown(hint=bn)
                # Stage 38 Inc 2 — cross-frame transform builtins. All 6
                # pairwise directions (3 frames × 2 directions per pair).
                # Lower as identity at IR (Phase-0: actual transformation
                # math is Phase-1+; the wrapper-shift tracks intent only).
                # The typechecker enforces the input is in the SOURCE
                # frame and the output is in the TARGET frame so cross-
                # frame mistakes are caught at compile time.
                _frame_transforms = {
                    "world_to_robot":   ("world",  "robot"),
                    "robot_to_world":   ("robot",  "world"),
                    "robot_to_camera":  ("robot",  "camera"),
                    "camera_to_robot":  ("camera", "robot"),
                    "world_to_camera":  ("world",  "camera"),
                    "camera_to_world":  ("camera", "world"),
                }
                if bn in _frame_transforms:
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    src_frame, dst_frame = _frame_transforms[bn]
                    if (isinstance(arg_tys[0], TyFrame)
                            and arg_tys[0].frame == src_frame):
                        return TyFrame(frame=dst_frame,
                                       inner=arg_tys[0].inner)
                    self.errors.append(TypeError_(
                        f"{bn}() requires "
                        f"{src_frame.capitalize()}Frame<T>, got "
                        f"{self._fmt(arg_tys[0])}",
                        expr.span,
                    ))
                    return TyUnknown(hint=bn)
                # Stage 39 Inc 1 — temporal constructors + eliminators.
                # 4 temporal kinds (past/present/future/eternal) each get
                # an into_* constructor (T -> KindName<T>) and a from_*
                # eliminator (KindName<T> -> T). Mirrors Stage 37/38
                # tier+frame pattern. All lower to identity at IR (Phase-0:
                # temporal kind lives at type level, zero runtime overhead).
                _temporal_intro = {
                    "into_past":    "past",
                    "into_present": "present",
                    "into_future":  "future",
                    "into_eternal": "eternal",
                }
                if bn in _temporal_intro:
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    # Stage 43 Inc 1 M1 fix + Stage 43 closure gate-2
                    # MEDIUM: direction-aware temporal transition
                    # hint. Same pattern as frame/tier arms post-gate-1.
                    if isinstance(arg_tys[0], TyTemporal):
                        target_kind = _temporal_intro[bn]
                        source_kind = arg_tys[0].kind
                        # Audited temporal transitions (Stage 39 Inc 2):
                        _temp_transitions_by_pair = {
                            ("present", "past"):     "to_past",
                            ("present", "future"):   "forecast",
                            ("past",    "present"):  "recall_past",
                            ("future",  "present"):  "actualize",
                        }
                        if source_kind == target_kind:
                            transition_hint = (
                                f"unwrap with from_{source_kind} "
                                f"first if you really want to "
                                f"re-tag the value"
                            )
                        elif (source_kind, target_kind) in _temp_transitions_by_pair:
                            tname = _temp_transitions_by_pair[
                                (source_kind, target_kind)]
                            transition_hint = (
                                f"use `{tname}(...)` — the audited "
                                f"{source_kind.capitalize()} -> "
                                f"{target_kind.capitalize()} "
                                f"temporal transition"
                            )
                        else:
                            transition_hint = (
                                f"Phase-0 has no "
                                f"{source_kind.capitalize()} -> "
                                f"{target_kind.capitalize()} "
                                f"temporal transition; unwrap with "
                                f"from_{source_kind} first if you "
                                f"need to re-tag"
                            )
                        self.errors.append(TypeError_(
                            f"{bn}() received an already-wrapped "
                            f"{self._fmt(arg_tys[0])}; intro "
                            f"builtins are not idempotent — "
                            f"{transition_hint}.",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    return TyTemporal(kind=_temporal_intro[bn],
                                      inner=arg_tys[0])
                _temporal_elim = {
                    "from_past":    "past",
                    "from_present": "present",
                    "from_future":  "future",
                    "from_eternal": "eternal",
                }
                if bn in _temporal_elim:
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    want = _temporal_elim[bn]
                    if (isinstance(arg_tys[0], TyTemporal)
                            and arg_tys[0].kind == want):
                        return arg_tys[0].inner
                    self.errors.append(TypeError_(
                        f"{bn}() requires "
                        f"{want.capitalize()}<T>, got "
                        f"{self._fmt(arg_tys[0])}",
                        expr.span,
                    ))
                    return TyUnknown(hint=bn)
                # Stage 39 Inc 2 — cross-temporal transitions. 4 directions:
                # to_past (present->past), forecast (present->future),
                # recall_past (past->present), actualize (future->present).
                # Eternal doesn't transition (it's timeless). All lower as
                # identity at IR — Phase-0 transitions track intent only.
                _temporal_transitions = {
                    "to_past":     ("present", "past"),
                    "forecast":    ("present", "future"),
                    "recall_past": ("past",    "present"),
                    "actualize":   ("future",  "present"),
                }
                if bn in _temporal_transitions:
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    src_kind, dst_kind = _temporal_transitions[bn]
                    if (isinstance(arg_tys[0], TyTemporal)
                            and arg_tys[0].kind == src_kind):
                        return TyTemporal(kind=dst_kind,
                                          inner=arg_tys[0].inner)
                    self.errors.append(TypeError_(
                        f"{bn}() requires "
                        f"{src_kind.capitalize()}<T>, got "
                        f"{self._fmt(arg_tys[0])}",
                        expr.span,
                    ))
                    return TyUnknown(hint=bn)
                # Stage 40 Inc 1 — modal constructors + eliminators.
                # 4 modal kinds (known/believed/goal/uncertain) each
                # get an into_* constructor (T -> KindName<T>) and a
                # from_* eliminator (KindName<T> -> T). Mirrors Stage
                # 37/38/39 patterns. All lower to identity at IR
                # (Phase-0: modal kind lives at the type system level
                # — zero runtime overhead).
                _modal_intro = {
                    "into_known":     "known",
                    "into_believed":  "believed",
                    "into_goal":      "goal",
                    "into_uncertain": "uncertain",
                }
                if bn in _modal_intro:
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    # Stage 43 Inc 1 M1 fix + gate-2 MEDIUM:
                    # direction-aware modal transition hint.
                    if isinstance(arg_tys[0], TyModal):
                        target_kind = _modal_intro[bn]
                        source_kind = arg_tys[0].kind
                        # Audited modal transitions (Stage 40 Inc 2):
                        _modal_transitions_by_pair = {
                            ("believed", "known"): "confirm",
                            ("goal",     "known"): "act_on",
                        }
                        if source_kind == target_kind:
                            mt_hint = (
                                f"unwrap with from_{source_kind} "
                                f"first if you really want to "
                                f"re-tag"
                            )
                        elif (source_kind, target_kind) in _modal_transitions_by_pair:
                            tname = _modal_transitions_by_pair[
                                (source_kind, target_kind)]
                            mt_hint = (
                                f"use `{tname}(...)` — the audited "
                                f"{source_kind.capitalize()} -> "
                                f"{target_kind.capitalize()} "
                                f"epistemic upgrade"
                            )
                        else:
                            mt_hint = (
                                f"Phase-0 has no audited "
                                f"{source_kind.capitalize()} -> "
                                f"{target_kind.capitalize()} "
                                f"modal transition; downgrades and "
                                f"sideways shifts are semantically "
                                f"incoherent or deferred (see "
                                f"stage40 progress doc)"
                            )
                        self.errors.append(TypeError_(
                            f"{bn}() received an already-wrapped "
                            f"{self._fmt(arg_tys[0])}; intro "
                            f"builtins are not idempotent — "
                            f"{mt_hint}.",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    # Stage 40 closure gate-1 silent-failure F1 fix
                    # (HIGH): block the direct syntactic Uncertain-
                    # laundering pattern `into_X(from_uncertain(u))`.
                    # An agent could unwrap-rewrap an Uncertain value
                    # into a Known fact with no diagnostic, vacating
                    # Stage 40's "category mistake at the heart of
                    # many AI safety failures" claim. Closes the
                    # direct form.
                    #
                    # PARTIAL CLOSURE (Stage 52 Inc 1, commit c274059):
                    # let-binding bypass is closed by the new
                    # _modal_origin_provenance consult at the Name-
                    # operand branch below. The inline form (this
                    # arm) catches `into_X(from_Y(v))` directly;
                    # the let-binding form is now caught via the
                    # taint-tracking dict.
                    #
                    # STILL DEFERRED to Stage 52 Inc 2+ /
                    # Stage 53: helper-fn indirection
                    # (`fn launder(x: i32) -> Known<i32> { into_known(x) }`
                    # called with a from_X result) requires inter-
                    # procedural taint propagation, a different
                    # defect class than let-bypass.
                    # Defensive: explicit args bounds check
                    # (gate-1 M1, conf 82) — pre-fix this block
                    # accessed expr.args[0] without re-asserting
                    # the structural precondition.
                    target_kind = _modal_intro[bn]
                    # Stage 40 closure gate-2 type-design F1 fix
                    # (HIGH conf 90): extend gate-1's Uncertain-
                    # only guard to ALL cross-modal direct
                    # laundering. The audit found `into_known(
                    # from_believed(b))` and `into_known(
                    # from_goal(g))` were still uncatchable post-
                    # gate-1 — but Phase-0 has only `confirm`
                    # (Believed -> Known) and `act_on` (Goal ->
                    # Known) as audited upgrade paths. The
                    # asymmetry left the "category mistake at
                    # compile time" thesis materially incomplete.
                    # Generalize: any `into_X(from_Y(v))` where
                    # X != Y is rejected with a kind-specific
                    # hint pointing at the legitimate transition
                    # (or noting the deferral when none exists).
                    # KNOWN LIMITATION (carried from gate-1 H1):
                    # this guard is syntactic. Let-binding
                    # bypass (`let r = from_Y(v); into_X(r)`)
                    # and helper-fn indirection are documented
                    # as Phase-0 known limits requiring future
                    # taint-tracking spec.
                    # Stage 52 closure gate-6 type-design F1 fix:
                    # eliminate the residual local dict that contradicts
                    # the gate-2 F3 hoisting invariant. Pre-fix, this
                    # site held a 4th identical copy of the elim→kind
                    # map under a different name (`_MODAL_ELIM_TO_KIND`),
                    # recreating exactly the divergence-risk class the
                    # F3 fix was meant to prevent. Use the module-level
                    # _MODAL_ELIM_TO_KIND (the gate-2 F3 single source
                    # of truth) instead.
                    # Stage 53 Inc 1 hoist: use the module-level
                    # _MODAL_UPGRADE_HINT instead of duplicating the
                    # dict locally. The hint table is now shared with
                    # the Stage 53 helper-fn-indirection launder check
                    # (gate-2 F3 single-source-of-truth pattern).
                    # Stage 40 closure gate-2 M1 fix (MEDIUM conf
                    # 85): only fire the laundering diagnostic when
                    # the inner from_X(...) actually returned a
                    # successfully-typed value. If the inner already
                    # produced its own diagnostic (TyUnknown), the
                    # F1 "launders" message would be semantically
                    # false (no value was ever wrapped) and would
                    # mislead the user away from the real bug.
                    #
                    # Stage 40 closure gate-3 HIGH cross-confirmed
                    # fix (type-design H1 conf 86 + code-review
                    # MEDIUM-1 conf 82): the F1 guard inspects the
                    # INNER call's syntactic name without checking
                    # `_shadowed_builtin_names`. When a user shadows
                    # `from_X`, the H2 shadow diagnostic fires AT
                    # the fn-decl site AND the launder guard fires
                    # on top — violating H2's "1 + 0 noise"
                    # invariant. Skip the launder check when the
                    # inner callee name has been shadowed (the H2
                    # cascade-suppression already fires for the
                    # bare-name dispatch path; the launder guard
                    # has to mirror that discipline).
                    # Stage 52 Inc 7 / gate-10 HIGH-1 fix: unified
                    # source-kind consult via `_modal_origin_of_expr`.
                    # Pre-fix, the F1 launder check had TWO narrow
                    # syntactic guards: one for `Call(from_X, ...)`
                    # (Stage 40 F1) and one for `Name with tracked
                    # taint` (Stage 52 Inc 1). Inline forms like
                    # `into_known(match scrut { x => from_X(...) })`
                    # bypassed both (arg is A.Match, not A.Call or
                    # A.Name). Inc 6 added recursive helper support
                    # for Block/If/Match yield-from-modal detection
                    # but wired it ONLY into the Stage 53 user-fn
                    # launder check — leaving builtin into_X with the
                    # narrow guards, producing asymmetric coverage.
                    # Post-fix: single helper call replaces both
                    # syntactic checks — coverage symmetric across
                    # all consult sites (builtin into_X + user-fn
                    # call + Let-RHS + Assign-RHS + match-scrutinee).
                    #
                    # Shadowed builtin safety: helper checks
                    # `_MODAL_ELIM_TO_KIND` first, so a user fn
                    # shadowing `from_X` returns the builtin kind
                    # rather than the user-fn kind. The dedicated
                    # `_shadowed_builtin_names` skip is no longer
                    # needed for THIS check (the H2 cascade-suppression
                    # at the bare-name dispatch path still applies).
                    if (len(expr.args) >= 1
                            and not isinstance(arg_tys[0], TyUnknown)):
                        source_kind = self._modal_origin_of_expr(
                            expr.args[0])
                        if (source_kind is not None
                                and source_kind != target_kind):
                            upgrade_hint = _MODAL_UPGRADE_HINT.get(
                                (source_kind, target_kind))
                            if upgrade_hint:
                                hint = upgrade_hint
                            else:
                                hint = (
                                    "Phase-0 has no "
                                    f"{source_kind.capitalize()} "
                                    f"-> {target_kind.capitalize()} "
                                    "transition; if this direction "
                                    "is semantically meaningful, "
                                    "request a future-stage spec "
                                    "and keep the value in its "
                                    "current modal kind until then"
                                )
                            # Diagnostic form: name the arg if A.Name
                            # (preserves "via taint-tracking" framing
                            # for the Inc 1 path), else show the form.
                            if isinstance(expr.args[0], A.Name):
                                arg_repr = f"'{expr.args[0].name}'"
                                form = (
                                    f"via taint-tracking — "
                                    f"{arg_repr} carries a tracked "
                                    f"from_{source_kind}(...) origin "
                                    f"from a let-binding, Assign-stmt, "
                                    f"match-arm, if-branch, while-body, "
                                    f"or yielded modal expression."
                                )
                            elif (isinstance(expr.args[0], A.Call)
                                  and isinstance(expr.args[0].callee, A.Name)
                                  and expr.args[0].callee.name
                                      in _MODAL_ELIM_TO_KIND):
                                arg_repr = (
                                    f"from_{source_kind}(...)"
                                )
                                form = "with no epistemic-upgrade audit."
                            else:
                                arg_repr = "..."
                                form = (
                                    f"via yielded modal expression "
                                    f"(a match/if/block tail with "
                                    f"from_{source_kind}(...))."
                                )
                            self.errors.append(TypeError_(
                                f"{bn}({arg_repr}) launders a "
                                f"{source_kind.capitalize()}<T> "
                                f"into "
                                f"{target_kind.capitalize()}<T> "
                                f"{form}",
                                expr.span,
                                hint=hint,
                            ))
                            return TyUnknown(hint=bn)
                    return TyModal(kind=_modal_intro[bn],
                                   inner=arg_tys[0])
                _modal_elim = {
                    "from_known":     "known",
                    "from_believed":  "believed",
                    "from_goal":      "goal",
                    "from_uncertain": "uncertain",
                }
                if bn in _modal_elim:
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    want = _modal_elim[bn]
                    if (isinstance(arg_tys[0], TyModal)
                            and arg_tys[0].kind == want):
                        return arg_tys[0].inner
                    self.errors.append(TypeError_(
                        f"{bn}() requires "
                        f"{want.capitalize()}<T>, got "
                        f"{self._fmt(arg_tys[0])}",
                        expr.span,
                    ))
                    return TyUnknown(hint=bn)
                # Stage 40 Inc 2 — modal transitions (epistemic
                # upgrades). 2 deliberate directions:
                #   confirm: Believed -> Known (an inferred belief
                #     becomes a known fact when directly observed).
                #   act_on:  Goal -> Known (the agent achieves a
                #     goal; what was desired is now observed-true).
                # Downgrades + Goal->Believed + Uncertain->any are
                # deferred (see stage40 progress doc rationale).
                _modal_transitions = {
                    "confirm": ("believed", "known"),
                    "act_on":  ("goal",     "known"),
                }
                if bn in _modal_transitions:
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    src_kind, dst_kind = _modal_transitions[bn]
                    if (isinstance(arg_tys[0], TyModal)
                            and arg_tys[0].kind == src_kind):
                        return TyModal(kind=dst_kind,
                                       inner=arg_tys[0].inner)
                    self.errors.append(TypeError_(
                        f"{bn}() requires "
                        f"{src_kind.capitalize()}<T>, got "
                        f"{self._fmt(arg_tys[0])}",
                        expr.span,
                    ))
                    return TyUnknown(hint=bn)
                # Stage 41 Inc 1 — causal constructors + eliminators.
                _causal_intro = {
                    "into_cause":       "cause",
                    "into_effect":      "effect",
                    "into_joint":       "joint",
                    "into_independent": "independent",
                }
                _causal_elim_kind = {
                    "from_cause":       "cause",
                    "from_effect":      "effect",
                    "from_joint":       "joint",
                    "from_independent": "independent",
                }
                _causal_upgrade_hint = {
                    ("cause", "effect"):
                        "use `propagate(c)` — the audited "
                        "Cause -> Effect causal transition",
                    ("effect", "joint"):
                        "use `aggregate(e)` — the audited "
                        "Effect -> Joint causal aggregation",
                    ("joint", "independent"):
                        "use `isolate(j)` — the audited "
                        "Joint -> Independent causal collapse",
                    # Stage 41 closure gate-1 LOW fix: safety-
                    # anchored framing for the obviously-
                    # incoherent reverse directions (an effect
                    # does not retroactively become its own
                    # cause). Stage 40 gate-3 LOW lesson applied:
                    # generic "Phase-0 has no transition" framing
                    # mis-suggests a future feature when the
                    # direction is semantically nonsensical.
                    ("effect", "cause"):
                        "an effect does not retroactively become "
                        "its own cause; if you mean to identify "
                        "the upstream cause, recover it from the "
                        "same provenance source rather than "
                        "unwrap-rewrap the downstream value",
                    ("joint", "cause"):
                        "a joint observation is downstream of "
                        "multiple causes; promoting it back to "
                        "Cause<T> conflates aggregation with "
                        "origination — re-derive the cause from "
                        "the original provenance",
                    ("independent", "cause"):
                        "an Independent<T> value has been shown "
                        "to have NO upstream; treating it as a "
                        "Cause<T> contradicts that experimental "
                        "finding",
                    ("independent", "joint"):
                        "Independent<T> means the experiment "
                        "collapsed the multi-cause dependency; "
                        "re-promoting to Joint<T> would require "
                        "fresh evidence of dependency, not "
                        "unwrap-rewrap",
                    ("independent", "effect"):
                        "an Independent<T> value's upstream is "
                        "by construction empty; calling it an "
                        "Effect<T> claims a downstream-of-"
                        "something relationship that was just "
                        "experimentally falsified",
                }
                if bn in _causal_intro:
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    # Stage 43 Inc 1 M1 fix + gate-2 MEDIUM:
                    # direction-aware causal transition hint.
                    if isinstance(arg_tys[0], TyCausal):
                        target_kind = _causal_intro[bn]
                        source_kind = arg_tys[0].kind
                        # Audited causal transitions (Stage 41 Inc 2):
                        _causal_transitions_by_pair = {
                            ("cause",  "effect"):      "propagate",
                            ("effect", "joint"):       "aggregate",
                            ("joint",  "independent"): "isolate",
                        }
                        if source_kind == target_kind:
                            ct_hint = (
                                f"unwrap with from_{source_kind} "
                                f"first if you really want to "
                                f"re-tag"
                            )
                        elif (source_kind, target_kind) in _causal_transitions_by_pair:
                            tname = _causal_transitions_by_pair[
                                (source_kind, target_kind)]
                            ct_hint = (
                                f"use `{tname}(...)` — the audited "
                                f"{source_kind.capitalize()} -> "
                                f"{target_kind.capitalize()} "
                                f"causal transition"
                            )
                        else:
                            ct_hint = (
                                f"Phase-0 has no audited "
                                f"{source_kind.capitalize()} -> "
                                f"{target_kind.capitalize()} "
                                f"causal transition; reverse and "
                                f"skip-step directions are semantically "
                                f"incoherent or deferred (see "
                                f"stage41 progress doc)"
                            )
                        self.errors.append(TypeError_(
                            f"{bn}() received an already-wrapped "
                            f"{self._fmt(arg_tys[0])}; intro "
                            f"builtins are not idempotent — "
                            f"{ct_hint}.",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    target_kind = _causal_intro[bn]
                    # Stage 40 F1 lesson applied preemptively:
                    # reject direct cross-causal laundering
                    # (`into_X(from_Y(v))` with X != Y), with
                    # kind-specific hint pointing at the audited
                    # transition or noting Phase-0 deferral.
                    # Stage 41 closure gate-1 type-design F1
                    # PARITY FIX: mirror the Stage 40 closure
                    # gate-3 H1 amendment to the cross-modal
                    # guard. The `inner_is_shadowed` cascade-
                    # suppression check must also fire here so
                    # `into_effect(from_cause(v))` where `from_cause`
                    # is user-shadowed doesn't produce 1 shadow +
                    # 1 launder noise. Mirror the modal guard
                    # at line 3643-3656 verbatim for the causal
                    # surface.
                    inner_is_shadowed = (
                        len(expr.args) >= 1
                        and isinstance(expr.args[0], A.Call)
                        and isinstance(expr.args[0].callee, A.Name)
                        and expr.args[0].callee.name
                            in self._shadowed_builtin_names
                    )
                    if (len(expr.args) >= 1
                            and isinstance(expr.args[0], A.Call)
                            and isinstance(expr.args[0].callee, A.Name)
                            and expr.args[0].callee.name
                                in _causal_elim_kind
                            and not isinstance(arg_tys[0], TyUnknown)
                            and not inner_is_shadowed):
                        source_kind = _causal_elim_kind[
                            expr.args[0].callee.name]
                        if source_kind != target_kind:
                            upgrade_hint = _causal_upgrade_hint.get(
                                (source_kind, target_kind))
                            if upgrade_hint:
                                hint = upgrade_hint
                            else:
                                hint = (
                                    "Phase-0 has no "
                                    f"{source_kind.capitalize()} "
                                    f"-> {target_kind.capitalize()} "
                                    "transition; if this direction "
                                    "is semantically meaningful, "
                                    "request a future-stage spec "
                                    "and keep the value in its "
                                    "current causal kind until then"
                                )
                            self.errors.append(TypeError_(
                                f"{bn}(from_{source_kind}(...)) "
                                f"launders a "
                                f"{source_kind.capitalize()}<T> "
                                f"into "
                                f"{target_kind.capitalize()}<T> "
                                f"with no causal-transition "
                                f"audit.",
                                expr.span,
                                hint=hint,
                            ))
                            return TyUnknown(hint=bn)
                    return TyCausal(kind=target_kind,
                                    inner=arg_tys[0])
                if bn in _causal_elim_kind:
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    want = _causal_elim_kind[bn]
                    if (isinstance(arg_tys[0], TyCausal)
                            and arg_tys[0].kind == want):
                        return arg_tys[0].inner
                    self.errors.append(TypeError_(
                        f"{bn}() requires "
                        f"{want.capitalize()}<T>, got "
                        f"{self._fmt(arg_tys[0])}",
                        expr.span,
                    ))
                    return TyUnknown(hint=bn)
                # Stage 41 Inc 2 — causal transitions.
                _causal_transitions = {
                    "propagate": ("cause",  "effect"),
                    "aggregate": ("effect", "joint"),
                    "isolate":   ("joint",  "independent"),
                }
                if bn in _causal_transitions:
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    src_kind, dst_kind = _causal_transitions[bn]
                    if (isinstance(arg_tys[0], TyCausal)
                            and arg_tys[0].kind == src_kind):
                        return TyCausal(kind=dst_kind,
                                        inner=arg_tys[0].inner)
                    self.errors.append(TypeError_(
                        f"{bn}() requires "
                        f"{src_kind.capitalize()}<T>, got "
                        f"{self._fmt(arg_tys[0])}",
                        expr.span,
                    ))
                    return TyUnknown(hint=bn)
                # Stage 46 Inc 1 — Result<T, E> constructors,
                # accessors, queries, combinators. Phase-0:
                # identity-lowered at IR. The Ok/Err discriminant
                # lives at the type system level only. Real
                # runtime tag is Stage 48+ work when `?` early-
                # return semantics need it.
                #
                # Ok(v) and Err(e) need the OTHER variant's type
                # to be inferred from context. Phase-0 inference
                # is shallow: we use TyUnknown for the unspecified
                # side, letting downstream usage constrain it. A
                # full bidirectional inference pass is future work.
                if bn == "Ok":
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"Ok() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    # Stage 49 closure gate-2 type-design G2-H1 fix:
                    # reject non-i32 payload. The Stage 49 Inc 1
                    # packed-i64 representation uses a 32-bit
                    # payload slot; wider types (i64, f32, f64,
                    # struct) would silently truncate at lowering.
                    # Stage 50+ will widen the representation
                    # (per docs/stage49-plan-2026-05-17.md:164-171).
                    self._reject_non_i32_result_payload(
                        arg_tys[0], expr.span, side="Ok")
                    return TyResult(
                        ok_ty=arg_tys[0],
                        err_ty=TyUnknown(hint="Err inferred"),
                    )
                if bn == "Err":
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"Err() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    # Stage 49 closure gate-2 G2-H1 (Err side).
                    self._reject_non_i32_result_payload(
                        arg_tys[0], expr.span, side="Err")
                    return TyResult(
                        ok_ty=TyUnknown(hint="Ok inferred"),
                        err_ty=arg_tys[0],
                    )
                if bn in ("unwrap_ok", "unwrap_err"):
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    if not isinstance(arg_tys[0], TyResult):
                        self.errors.append(TypeError_(
                            f"{bn}() requires Result<T, E>, got "
                            f"{self._fmt(arg_tys[0])}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    # Stage 46 closure gate-1 silent-failure
                    # Finding 4 fix (MEDIUM): if the operand is an
                    # inference-determined Ok-constructed value
                    # (err_ty is TyUnknown with hint "Err inferred"),
                    # `unwrap_err` is a known runtime panic. Same
                    # for `unwrap_ok` on an Err-constructed value.
                    # Reject at typecheck rather than letting the
                    # TyUnknown-universally-compatible cascade
                    # silently accept the misuse.
                    #
                    # Stage 46 closure gate-2 silent-failure G2-F1
                    # fix: the F4 check covers the inference path
                    # (`let r = Ok(7)`) but NOT the typed-let path
                    # (`let r: Result<i32, i32> = Ok(7)`) because
                    # the declared type annotation overrides the
                    # TyUnknown hint at bind time. The
                    # `_result_constructor_provenance` map records
                    # the original constructor side independently
                    # so typed-let wrong-arm calls are also caught.
                    inner = arg_tys[0]
                    # First check: explicit name-bound provenance.
                    if (isinstance(expr.args[0], A.Name)
                            and expr.args[0].name
                                in self._result_constructor_provenance):
                        prov = self._result_constructor_provenance[
                            expr.args[0].name]
                        if bn == "unwrap_err" and prov == "ok":
                            self.errors.append(TypeError_(
                                f"unwrap_err() called on "
                                f"{expr.args[0].name!r}, which was "
                                f"constructed via Ok() — the Err "
                                f"side was never set, so this is "
                                f"an unconditional runtime panic.",
                                expr.span,
                                hint="use is_ok / unwrap_ok / "
                                "or remove the unwrap_err",
                            ))
                            return TyUnknown(hint=bn)
                        if bn == "unwrap_ok" and prov == "err":
                            self.errors.append(TypeError_(
                                f"unwrap_ok() called on "
                                f"{expr.args[0].name!r}, which was "
                                f"constructed via Err() — the Ok "
                                f"side was never set, so this is "
                                f"an unconditional runtime panic.",
                                expr.span,
                                hint="use is_err / unwrap_err / "
                                "or remove the unwrap_ok",
                            ))
                            return TyUnknown(hint=bn)
                    if bn == "unwrap_err":
                        err = inner.err_ty
                        if (isinstance(err, TyUnknown)
                                and err.hint == "Err inferred"):
                            self.errors.append(TypeError_(
                                "unwrap_err() called on a Result "
                                "constructed via Ok() — the Err "
                                "side was never set, so this is "
                                "an unconditional runtime panic.",
                                expr.span,
                                hint="use is_ok / unwrap_ok / "
                                "or remove the unwrap_err",
                            ))
                            return TyUnknown(hint=bn)
                        return inner.err_ty
                    else:  # unwrap_ok
                        ok = inner.ok_ty
                        if (isinstance(ok, TyUnknown)
                                and ok.hint == "Ok inferred"):
                            self.errors.append(TypeError_(
                                "unwrap_ok() called on a Result "
                                "constructed via Err() — the Ok "
                                "side was never set, so this is "
                                "an unconditional runtime panic.",
                                expr.span,
                                hint="use is_err / unwrap_err / "
                                "or remove the unwrap_ok",
                            ))
                            return TyUnknown(hint=bn)
                        return inner.ok_ty
                if bn == "__try":
                    # Stage 48 Inc 2 — `?` propagation operator (parser
                    # desugars `expr?` to `__try(expr)`). Validation:
                    #
                    # 1. Arity: exactly one operand.
                    # 2. Operand must be Result<T, E1>.
                    # 3. Enclosing fn return type must be Result<U, E2>
                    #    — `?` cannot propagate up to a non-Result
                    #    return type. (This is the spec-defining
                    #    constraint for the operator.)
                    # 4. E1 must be compatible with E2 (the Err type
                    #    that `?` propagates must fit the function's
                    #    own Err slot). This is the silent-miscompile
                    #    failure mode if the audit lane skipped here:
                    #    `expr?` would compile, look like it works,
                    #    and at runtime (Stage 49+ once branching is
                    #    live) the propagated Err would have the
                    #    WRONG TYPE wrt the function's signature.
                    # 5. Constructor-provenance: `Ok(7)?` is
                    #    benign (Phase-0 identity); `Err(7)?` is
                    #    REJECTED via the gate-1 F2 diagnostic
                    #    when the operand is an A.Name with known
                    #    "err" provenance (see code at the
                    #    `_result_constructor_provenance` consult
                    #    below — search "gate-1 silent-failure F2
                    #    fix"). The non-Name and dynamic-Err cases
                    #    remain F1-class deferred (Stage 49 runtime
                    #    tag fixes the whole class).
                    # 6. Result type = the operand's Ok inner.
                    #    Caveat: a freshly-constructed `Err(7)?`
                    #    yields a Result whose ok_ty is the
                    #    Stage-46 placeholder TyUnknown(hint="Ok
                    #    inferred"). That propagates through the
                    #    rest of typecheck as universally
                    #    compatible (Stage 46 inference policy),
                    #    which is the correct Phase-0 behaviour
                    #    pre-runtime-tag.
                    if len(arg_tys) != 1:
                        # Stage 49 closure gate-1 code-review L1 polish:
                        # vocab aligned with is_ok/is_err/map_* arity
                        # diagnostics ("argument" not "operand") for
                        # consistency. Pre-fix `?` said "1 operand"
                        # while sibling builtins said "1 argument" —
                        # confusing for users since the underlying
                        # arity-mismatch class is identical.
                        self.errors.append(TypeError_(
                            f"`?` takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint="try")
                    operand_ty = arg_tys[0]
                    if not isinstance(operand_ty, TyResult):
                        # Stage 48 closure gate-2 code-review M1
                        # polish: name the operand when it's an
                        # A.Name so the user gets `x?` not just
                        # the position in the source.
                        operand_label = (
                            f" on {expr.args[0].name!r}"
                            if isinstance(expr.args[0], A.Name)
                            else ""
                        )
                        self.errors.append(TypeError_(
                            f"`?`{operand_label} requires a "
                            f"Result<T, E> operand, got "
                            f"{self._fmt(operand_ty)}",
                            expr.span,
                            hint="`?` propagates the Err arm of a "
                            "Result; the operand must itself be "
                            "Result-typed",
                        ))
                        return TyUnknown(hint="try")
                    ret_ty = self._current_return_ty
                    if not isinstance(ret_ty, TyResult):
                        self.errors.append(TypeError_(
                            f"`?` used in function "
                            f"{self._current_fn_name!r} whose return "
                            f"type is {self._fmt(ret_ty)}, not "
                            f"Result<T, E>",
                            expr.span,
                            hint="change the function's return type "
                            "to Result<T, E>, or extract the Ok "
                            "value with unwrap_ok() instead of `?`",
                        ))
                        return TyUnknown(hint="try")
                    # Err-compat: the operand's Err must fit the
                    # function's Err slot. TyUnknown on either side
                    # (e.g. `Err(7)` with no annotation produces
                    # err_ty inferred-from-arg, ok_ty TyUnknown) is
                    # universally compatible per Stage 46 inference
                    # policy.
                    # Stage 48 closure gate-1 LOW: `_compatible` is
                    # symmetric in Phase-0; the argument-order here
                    # would become meaningful once subtyping lands.
                    # Today both directions yield the same answer.
                    if not self._compatible(operand_ty.err_ty,
                                            ret_ty.err_ty):
                        self.errors.append(TypeError_(
                            f"`?` Err-type mismatch: operand has "
                            f"Err={self._fmt(operand_ty.err_ty)}, "
                            f"function {self._current_fn_name!r} "
                            f"returns Result with "
                            f"Err={self._fmt(ret_ty.err_ty)}",
                            expr.span,
                            hint="the Err type propagated by `?` "
                            "must match the function's own Err type",
                        ))
                        return TyUnknown(hint="try")
                    # Stage 48 closure gate-1 silent-failure F2 fix
                    # (HIGH): the `?` arm must consult
                    # `_result_constructor_provenance` exactly as
                    # `unwrap_ok` / `unwrap_err` do. Pre-fix,
                    # `let r: Result<i32, i32> = Err(99); r?`
                    # silently extracted the Err payload as if it
                    # were Ok (no runtime tag yet, identity-lowered).
                    # Post-fix: when the operand is a Name with
                    # known "err" provenance, reject — this is a
                    # statically-determinable wrong-arm case.
                    # `Ok` provenance is benign: `?` on a known-Ok
                    # value is identity, which is the correct
                    # Phase-0 behavior.
                    # Stage 49 Inc 4 LIFTED this gate-1 F2 reject:
                    # `?` on a statically-Err-constructed Result is
                    # now sound at runtime — the COND_BR + RETURN
                    # propagates the Err from the enclosing fn. The
                    # previously-required Phase-0 reject is removed.
                    # (Static-provenance bookkeeping at this site
                    # would only be needed if we wanted to const-
                    # fold the `?` to an unconditional return, which
                    # is a future optimizer pass concern.)
                    # Stage 48 closure gate-1 silent-failure F1
                    # acknowledgement (HIGH, partial fix): for
                    # operands whose Result variant is NOT
                    # statically determinable (fn-call returns,
                    # if-branches), Phase-0 identity-lowering can
                    # silently extract an Err payload as Ok if the
                    # call returned Err at runtime. This is a
                    # known Phase-0 limitation that the Stage 49+
                    # runtime tag will eliminate. We don't reject
                    # these cases (would block legitimate `?`
                    # usage); we document them inline. A future
                    # static-analysis pass could flag function-
                    # call operands with "may-return-Err" return
                    # types as a soft warning. Stage 49+ work.
                    return operand_ty.ok_ty
                if bn in ("is_ok", "is_err"):
                    # Stage 49 Inc 2: is_ok / is_err now have real
                    # runtime semantics via the RESULT_TAG opcode
                    # introduced in Inc 1. They consult the high-32
                    # bits of the packed-i64 Result representation:
                    # is_ok(r) iff tag == 0, is_err(r) iff tag == 1.
                    # Pre-Stage-49 they were typecheck-rejected per
                    # Stage 46 closure gate-1 F1 fix (silent wrong-
                    # branch miscompilation risk without a runtime
                    # tag). Inc 2 lifts the rejection.
                    if len(arg_tys) != 1:
                        self.errors.append(TypeError_(
                            f"{bn}() takes 1 argument, got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    if not isinstance(arg_tys[0], TyResult):
                        self.errors.append(TypeError_(
                            f"{bn}() requires Result<T, E>, got "
                            f"{self._fmt(arg_tys[0])}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    return TyPrim("bool")
                if bn == "map_ok":
                    if len(arg_tys) != 2:
                        self.errors.append(TypeError_(
                            f"map_ok() takes 2 arguments (Result, "
                            f"new_value), got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    if not isinstance(arg_tys[0], TyResult):
                        self.errors.append(TypeError_(
                            f"map_ok() requires first arg "
                            f"Result<T, E>, got "
                            f"{self._fmt(arg_tys[0])}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    # Stage 49 closure gate-3 G3-H1 fix: map_ok
                    # constructs a fresh TyResult whose Ok side
                    # is the caller-provided new_value's type.
                    # Without this check, `map_ok(r, 9999_i64)`
                    # typecheck-passed but the new_value silently
                    # truncated to i32 at IR lowering. Same defect
                    # class as G2-H1 at Ok/Err constructors.
                    self._reject_non_i32_result_payload(
                        arg_tys[1], expr.span, side="map_ok new_value")
                    return TyResult(
                        ok_ty=arg_tys[1],
                        err_ty=arg_tys[0].err_ty,
                    )
                if bn == "map_err":
                    # Stage 49 Inc 3: map_err(r, new_err) now has
                    # real runtime semantics. Lowers to a SELECT on
                    # the tag: if tag==1 (Err), return RESULT_PACK(
                    # 1, new_err); else pass r through unchanged.
                    # Pre-Stage-49 it was typecheck-rejected (Stage
                    # 46 closure gate-1 F2 fix — pre-fix map_err
                    # silently discarded the intent because there
                    # was no runtime Err side to replace). Inc 3
                    # lifts the rejection.
                    if len(arg_tys) != 2:
                        self.errors.append(TypeError_(
                            f"map_err() takes 2 arguments (Result, "
                            f"new_value), got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    if not isinstance(arg_tys[0], TyResult):
                        self.errors.append(TypeError_(
                            f"map_err() requires first arg "
                            f"Result<T, E>, got "
                            f"{self._fmt(arg_tys[0])}",
                            expr.span,
                        ))
                        return TyUnknown(hint=bn)
                    # Stage 49 closure gate-3 G3-H1 fix (Err side):
                    # symmetric companion to map_ok above. Without
                    # this check, `map_err(r, 9999_i64)` typecheck-
                    # passed but silently truncated the new_err
                    # to i32 at IR lowering.
                    self._reject_non_i32_result_payload(
                        arg_tys[1], expr.span, side="map_err new_value")
                    return TyResult(
                        ok_ty=arg_tys[0].ok_ty,
                        err_ty=arg_tys[1],
                    )
                if bn == "consolidate" and len(arg_tys) == 1:
                    # Episodic -> Semantic
                    if isinstance(arg_tys[0], TyMemTier) and arg_tys[0].tier == "episodic":
                        return TyMemTier(tier="semantic", inner=arg_tys[0].inner)
                    self.errors.append(TypeError_(
                        f"consolidate() requires EpisodicMem<T>, got "
                        f"{self._fmt(arg_tys[0])}",
                        expr.span,
                    ))
                    return arg_tys[0]
                if bn == "recall" and len(arg_tys) == 1:
                    # Semantic -> Working (retrieve into working memory)
                    if isinstance(arg_tys[0], TyMemTier) and arg_tys[0].tier == "semantic":
                        return TyMemTier(tier="working", inner=arg_tys[0].inner)
                    self.errors.append(TypeError_(
                        f"recall() requires SemanticMem<T>, got "
                        f"{self._fmt(arg_tys[0])}",
                        expr.span,
                    ))
                    return arg_tys[0]
                if bn == "learn_to":
                    # learn_to(task: &str, difficulty: f32, budget: i32) -> Skill<...>
                    if len(arg_tys) != 3:
                        self.errors.append(TypeError_(
                            f"learn_to() requires 3 args (task, difficulty, "
                            f"budget); got {len(arg_tys)}",
                            expr.span,
                        ))
                        return TySkill(inner=TyUnknown(hint="learn_to"))
                    # Extract task name if it's a string literal
                    task_name = ""
                    if (isinstance(expr.args[0], A.StrLit)):
                        task_name = expr.args[0].value
                    return TySkill(inner=TyUnknown(hint=task_name), task=task_name)
                if bn in self._GPU_INDEX_BUILTINS:
                    if arg_tys:
                        self.errors.append(TypeError_(
                            f"{bn}() expects 0 args, got {len(arg_tys)}",
                            expr.span,
                        ))
                    if not self._current_is_kernel:
                        self.errors.append(TypeError_(
                            f"{bn}() is only allowed inside @kernel functions",
                            expr.span,
                        ))
                    return TyPrim("i32")
            # If callee is a known function (by name), do checks
            if isinstance(expr.callee, A.Name) and expr.callee.name in self.functions:
                sig = self.functions[expr.callee.name]
                self._check_call_basic(expr, sig, arg_tys, scope)
                self._check_call_shapes(expr, sig, arg_tys, scope)
                self._check_call_effects(expr, sig)
                # Stage 53 Inc 1: helper-fn indirection launder check.
                # If the user-fn returns a modal kind AND any arg is a
                # Name carrying a DIFFERENT tracked modal-origin, the
                # call is effectively `into_RETKIND(from_ARGKIND(...))`
                # — a category-error launder. Mirrors the F1 launder
                # check pattern for builtin into_X calls.
                #
                # Reproducer (was silent pre-Stage-53):
                #   fn launder(x: i32) -> Known<i32> { into_known(x) }
                #   let r = from_uncertain(u);
                #   let k = launder(r);  // <-- fires here
                #
                # This is the LAST modal-laundering bypass — closes
                # the Stage 40 H1 "different defect class" deferred
                # from Stage 52.
                fn_ret_kind = self._fn_modal_return_kind.get(
                    expr.callee.name)
                if fn_ret_kind is not None:
                    for arg_expr in expr.args:
                        arg_kind = self._modal_origin_of_expr(arg_expr)
                        if (arg_kind is not None
                                and arg_kind != fn_ret_kind):
                            arg_name_repr = (
                                f"'{arg_expr.name}'"
                                if isinstance(arg_expr, A.Name)
                                else "argument"
                            )
                            upgrade_hint = _MODAL_UPGRADE_HINT.get(
                                (arg_kind, fn_ret_kind))
                            if upgrade_hint:
                                hint = upgrade_hint
                            else:
                                hint = (
                                    "Phase-0 has no "
                                    f"{arg_kind.capitalize()} -> "
                                    f"{fn_ret_kind.capitalize()} "
                                    "transition; if this direction "
                                    "is semantically meaningful, "
                                    "request a future-stage spec "
                                    "and keep the value in its "
                                    "current modal kind until then"
                                )
                            self.errors.append(TypeError_(
                                f"{expr.callee.name}({arg_name_repr}) "
                                f"launders a "
                                f"{arg_kind.capitalize()}<T> "
                                f"into "
                                f"{fn_ret_kind.capitalize()}<T> "
                                f"via helper-fn indirection — "
                                f"the helper's declared return "
                                f"type asserts {fn_ret_kind.capitalize()}, "
                                f"but the argument carries a "
                                f"tracked from_{arg_kind}(...) "
                                f"origin. Same launder semantics "
                                f"as `into_{fn_ret_kind}(from_"
                                f"{arg_kind}(v))`.",
                                expr.span,
                                hint=hint,
                            ))
                if expr.callee.name in getattr(
                        self, "_invalid_refined_return_functions", set()):
                    return self._erase_refinement(sig.ret)
                return sig.ret
            if isinstance(callee, TyFn):
                self._check_function_typed_call(expr, callee, arg_tys, scope)
                self.errors.append(TypeError_(
                    "function-typed calls are not supported by the Stage 31 "
                    "backend",
                    expr.span,
                    hint="call a named function directly until indirect-call "
                         "lowering lands",
                ))
                return callee.ret
            return TyUnknown(hint="call")
        if isinstance(expr, A.Index):
            callee_ty = self._check_expr(expr.callee, scope)
            for i in expr.indices:
                idx_ty = self._check_expr(i, scope)
                if not isinstance(idx_ty, (TyUnknown, TyVar, TySize)) \
                        and not self._is_int_scalar(idx_ty):
                    self.errors.append(TypeError_(
                        f"array index must be an integer, got "
                        f"{self._fmt(idx_ty)}",
                        i.span,
                    ))
            if isinstance(callee_ty, TyArray):
                if len(expr.indices) != 1:
                    self.errors.append(TypeError_(
                        f"array index expects 1 index, got "
                        f"{len(expr.indices)}",
                        expr.span,
                    ))
                return callee_ty.elem
            if isinstance(callee_ty, TyTensor):
                self.errors.append(TypeError_(
                    "tensor indexing is not supported until tensor index "
                    "lowering is implemented",
                    expr.span,
                ))
                return TyUnknown(hint="tensor index")
            if isinstance(callee_ty, TyTile):
                if (isinstance(expr.callee, A.Name)
                        and expr.callee.name in self._current_hbm_tile_indexables
                        and len(expr.indices) == 1):
                    return callee_ty.dtype
                self.errors.append(TypeError_(
                    "tile indexing currently supports only @kernel HBM tile "
                    "parameters with exactly 1 index",
                    expr.span,
                ))
                return TyUnknown(hint="tile index")
            if not isinstance(callee_ty, (TyUnknown, TyVar, TySize)):
                self.errors.append(TypeError_(
                    f"type {self._fmt(callee_ty)} is not indexable",
                    expr.span,
                ))
            return TyUnknown(hint="index")
        if isinstance(expr, A.Field):
            obj_ty = self._check_expr(expr.obj, scope)
            if isinstance(obj_ty, TyStruct):
                decl = getattr(self, "_struct_decls", {}).get(obj_ty.name)
                if decl is not None:
                    for p in decl.fields:
                        if p.name == expr.name:
                            return self._resolve_type(p.ty, scope)
                    self.errors.append(TypeError_(
                        f"struct {obj_ty.name!r} has no field {expr.name!r}",
                        expr.span,
                    ))
                else:
                    self.errors.append(TypeError_(
                        f"unknown struct type {obj_ty.name!r} for field "
                        f"access {expr.name!r}",
                        expr.span,
                    ))
            # Tuple field access: `t.0`, `t.1`. The field "name" is a
            # stringified integer (per parser convention).
            if isinstance(obj_ty, TyTuple) and expr.name.isdigit():
                idx = int(expr.name)
                if 0 <= idx < len(obj_ty.elems):
                    return obj_ty.elems[idx]
                self.errors.append(TypeError_(
                    f"tuple index {idx} out of range (tuple has "
                    f"{len(obj_ty.elems)} elems)", expr.span,
                ))
            return TyUnknown(hint=f"field .{expr.name}")
        if isinstance(expr, A.Block):
            return self._check_block(expr, scope)
        if isinstance(expr, A.If):
            self._check_expr(expr.cond, scope)
            # Stage 52 closure gate-2 silent-failure HIGH-A fix:
            # mirror match-arm parallel-union for the if/else
            # branches. Pre-fix, `if cond { r = from_uncertain(u);
            # } else { r = from_known(k); }; into_known(r)`
            # silently passed because the then-branch mutated
            # the dict, then the else-branch's Assign-arm
            # POPULATE overwrote with 'known' (last write wins),
            # so the post-if dict claimed 'known' and into_known
            # consult found a matching kind.
            #
            # Same algorithm as match-arm (line 5275+): snapshot
            # pre-if, restore between branches, union arm-results
            # via the multi-kind drop semantics (HIGH-C fix).
            modal_origin_pre_if = dict(self._modal_origin_provenance)
            branch_results: list[dict[str, str]] = []
            branch_assigns: list[set[str]] = []
            t = self._check_block(expr.then, scope)
            branch_results.append(dict(self._modal_origin_provenance))
            branch_assigns.append(set(self._last_modal_assigns_popped))
            branch_tys = [t]
            if expr.else_ is not None:
                # Restore pre-if state before else.
                self._modal_origin_provenance = dict(
                    modal_origin_pre_if)
                if isinstance(expr.else_, A.Block):
                    e = self._check_block(expr.else_, scope)
                else:
                    # Gate-5 G4-F1/H2 fix: expression-form else
                    # arm bypasses _check_block. Wrap with the
                    # snapshot/restore helper so an Assign inside
                    # cannot leak provenance mutations past the
                    # if-expr boundary.
                    e = self._check_expr_in_block_scope(expr.else_, scope)
                branch_results.append(dict(self._modal_origin_provenance))
                branch_assigns.append(set(self._last_modal_assigns_popped))
                branch_tys.append(e)
                if not self._compatible(t, e):
                    self.errors.append(TypeError_(
                        f"if/else branches differ: {self._fmt(t)} vs {self._fmt(e)}",
                        expr.span,
                    ))
            else:
                # No-else branch: implicit no-op arm result == pre-if,
                # with no assigns (the branch didn't execute).
                branch_results.append(dict(modal_origin_pre_if))
                branch_assigns.append(set())
            # Stage 52 closure gate-3 NEW-HIGH-2/3/4 fix: union with
            # the "branch reassigned without modal taint" drop. For
            # each name observed across pre-if + arm-results,
            # collect its observed kinds. Then for any name that
            # ANY branch reassigned (in branch_assigns[i]) but
            # which is NOT in that branch's arm_result (i.e. the
            # branch's reassignment didn't install modal taint),
            # drop the name from the union. Else apply the
            # multi-kind-drop rule.
            observed_kinds: dict[str, set[str]] = {}
            for name, kind in modal_origin_pre_if.items():
                observed_kinds.setdefault(name, set()).add(kind)
            for arm_result in branch_results:
                for name, kind in arm_result.items():
                    observed_kinds.setdefault(name, set()).add(kind)
            # Stage 52 closure gate-6 latent-bug post-fix: only
            # mark a name as cleared if NO branch INSTALLED taint
            # for it. If some branch installs and others clear,
            # the installing branch may run at runtime → propagate
            # conservatively (FIRE rather than DROP). Stage 52
            # closure gate-7 silent-failure HIGH-3 post-fix: ALSO
            # check `kept_somewhere` — any name preserved in any
            # arm's result (e.g. the no-else implicit identity arm
            # preserves pre-if state) should override the cleared
            # signal. Without this, `let r = from_X(u); if cond {
            # r = 5; }; into_X(r)` silently passed because the
            # then-arm cleared and the no-else arm's preservation
            # didn't count toward "kept" — pre-this-fix, dropping
            # the static claim silently missed the cond=false
            # runtime path where r is still tainted.
            #
            # Semantic shift: NEW-HIGH-3 and NEW-HIGH-4 prior tests
            # asserted DROP (drop-on-conflict design); the gate-7
            # audit correctly identified those as false-positive-
            # leaning tests that miss real-runtime launders. The
            # stage's stated AI-safety property is "category-error
            # launders MUST be caught" — missing one is worse than
            # a false positive, so FIRE is the correct choice when
            # an identity arm preserves the taint.
            # Stage 52 closure gate-8 type-design MEDIUM-1 polish:
            # the prior code carried both `installed_names` and
            # `kept_somewhere` checks; installed_names is a strict
            # subset of kept_somewhere (any name in branch_results[i]
            # for some i is in the union of branch_results.keys()),
            # so the `installed_names` check was redundant. Dropped
            # to make the actual invariant single-source-of-truth:
            # "a name is cleared iff every branch that touched it
            # either erased it AND no other branch preserved/installed
            # it". The kept_somewhere check carries both conditions.
            kept_somewhere: set[str] = set()
            for arm_result in branch_results:
                kept_somewhere.update(arm_result.keys())
            cleared_names: set[str] = set()
            for i, assigns in enumerate(branch_assigns):
                for name in assigns:
                    if (name not in branch_results[i]
                            and name not in kept_somewhere):
                        cleared_names.add(name)
            unioned_if: dict[str, str] = {}
            for name, kinds in observed_kinds.items():
                if name in cleared_names:
                    continue  # branch cleared → drop static claim
                if len(kinds) == 1:
                    unioned_if[name] = next(iter(kinds))
            self._modal_origin_provenance = unioned_if
            return self._join_branch_types(branch_tys, expr.span)
        if isinstance(expr, A.Match):
            scrut_ty = self._check_expr(expr.scrutinee, scope)
            arm_tys: list[Type] = []
            # Stage 52 Inc 3 — match-arm modal-origin UNION
            # semantics (gate-1 silent-failure HIGH-1 fix). Each
            # arm body may install OR clear modal-origin taint via
            # the Assign-arm; sequential processing (Inc 2's
            # default) lets arm N+1 pop arm N's installed taint,
            # producing silent launders. Correct semantics is
            # PARALLEL UNION: at compile time we don't know which
            # arm runs, so any name that ANY arm taints is
            # conservatively post-match-tainted (taint surfaces).
            #
            # Implementation: snapshot the modal-origin dict before
            # each arm. After each arm, collect that arm's resulting
            # modal-origin state. Restore to pre-arm snapshot before
            # next arm. After all arms: union all arm-result dicts
            # into the post-match modal-origin state (any taint in
            # any arm propagates).
            modal_origin_pre_match = dict(self._modal_origin_provenance)
            modal_origin_arm_results: list[dict[str, str]] = []
            arm_assigns: list[set[str]] = []
            for arm in expr.arms:
                inner = Scope(parent=scope)
                self._bind_pattern(arm.pattern, scrut_ty, inner)
                # Stage 52 Inc 3 — snapshot modal-origin before arm
                # so each arm starts from the pre-match state.
                # Stage 52 closure gate-5 HIGH-1 fix: hoisted ABOVE
                # the guard check so the guard sees the pre-match
                # snapshot (not the previous arm's mutated dict).
                self._modal_origin_provenance = dict(
                    modal_origin_pre_match)
                # Stage 52 closure gate-4 HIGH-1 fix: propagate
                # scrutinee modal-origin taint to the pattern's
                # binding name. Pre-fix, `let r = from_uncertain(u);
                # match r { x => into_known(x) }` silently passed
                # because `x` was bound via `_bind_pattern` (which
                # only writes to the value scope) and never received
                # r's taint. Stage 52 closure gate-5 HIGH-1 fix:
                # hoisted ABOVE the guard check (so guards see the
                # taint). Stage 52 closure gate-6 CRITICAL-1 + 3
                # fix: unified taint-source via _modal_origin_of_expr
                # (handles Call-form scrutinee like `match
                # from_uncertain(u) { x => ... }`), AND PatOr-of-
                # PatBind support (handles `match r { (x | x) =>
                # ...}` and `E::A(x) | E::B(x)` enum fan-in
                # patterns).
                #
                # Top-level PatBind: copy scrutinee kind to bound
                # name. PatOr where every alt is a PatBind with the
                # SAME name: same copy (the name is bound in every
                # alt to the whole scrutinee). PatVariant payload
                # binds intentionally skipped — Phase-0 has no
                # modal-typed enum/tuple field; revisit in Inc-N
                # when that arrives.
                scrut_kind = self._modal_origin_of_expr(expr.scrutinee)
                if scrut_kind is not None:
                    bind_names_to_taint: set[str] = set()
                    if isinstance(arm.pattern, A.PatBind):
                        bind_names_to_taint.add(arm.pattern.name)
                    elif isinstance(arm.pattern, A.PatOr):
                        # PatOr-of-PatBind: every alt must be a
                        # PatBind of the same name to safely
                        # propagate (otherwise some alts decompose
                        # — defer those).
                        all_binds = [
                            alt for alt in arm.pattern.alts
                            if isinstance(alt, A.PatBind)
                        ]
                        if (len(all_binds) == len(arm.pattern.alts)
                                and len({b.name for b in all_binds}) == 1):
                            bind_names_to_taint.add(all_binds[0].name)
                    for bname in bind_names_to_taint:
                        self._modal_origin_provenance[bname] = scrut_kind
                if arm.guard is not None:
                    # Gate-5 G4-M3 fix: guard expression bypasses
                    # _check_block. Wrap to prevent any Assign
                    # inside the guard from leaking provenance
                    # mutations into the surrounding scope.
                    g_ty = self._check_expr_in_block_scope(arm.guard, inner)
                    if not (isinstance(g_ty, TyPrim) and g_ty.name == "bool") \
                            and not isinstance(g_ty, TyUnknown):
                        self.errors.append(TypeError_(
                            f"match guard must be bool, got {self._fmt(g_ty)}",
                            arm.span,
                        ))
                # Gate-5 G4-F1/H2 fix: bare-expression arm bodies
                # (e.g. `pat => r = Err(99)`) bypass _check_block.
                # The Assign-arm mutates the provenance dict in
                # _check_expr directly; without snapshot/restore
                # the last arm's mutation "wins" silently and a
                # post-match `?` accepts under stale provenance,
                # producing a silent runtime miscompile (gate-4
                # G4-F1 reproducer exit 99).
                arm_tys.append(
                    self._check_expr_in_block_scope(arm.body, inner))
                # Save this arm's resulting modal-origin state.
                modal_origin_arm_results.append(
                    dict(self._modal_origin_provenance))
                # Stage 52 closure gate-3 NEW-HIGH-4 fix: capture
                # the assigns-set the arm popped (names this arm
                # reassigned, whether or not the Assign installed
                # modal kind). Symmetric with the A.If fix.
                arm_assigns.append(set(self._last_modal_assigns_popped))
            # Stage 52 Inc 3 — UNION arm results. For each name,
            # if ANY arm installed taint of kind K, the post-match
            # value is conservatively tainted with K. If two arms
            # install different kinds (e.g. one Uncertain, one
            # Known), the post-match value could be either —
            # conservatively take the FIRST arm's kind (matches
            # "any taint propagates" semantics; refining to
            # "multi-kind sum" needs a richer dict value).
            # Stage 52 closure gate-2 silent-failure HIGH-C fix
            # (silent-launder when arm overwrites pre-match kind):
            # "first wins" silently drops the conflicting kind
            # information. Pre-fix, `let r = from_known(k); match
            # _ => { r = from_uncertain(u); } end; into_known(r)`
            # passed because the arm's 'uncertain' was discarded
            # in favor of pre-match's 'known'. Post-fix: any name
            # whose kind differs across arms (or differs between
            # any arm and pre-match) DROPS from the unioned dict
            # — the static claim is invalidated; the consult at
            # into_X falls back to no-static-claim (joins the
            # dynamic/no-taint path which Stage 53 helper-fn taint
            # will cover).
            # Stage 52 closure gate-3 NEW-HIGH-4 fix: also drop
            # any name an arm REASSIGNED without installing taint
            # (the assignment cleared the pre-match kind). Mirrors
            # A.If's branch-cleared drop. Without this, `let r =
            # from_uncertain(u); match cond { true => {} false =>
            # r = 0 }; into_known(r)` silently passes because the
            # union sees pre-match 'uncertain' + arm-true 'uncer-
            # tain' = single kind 'uncertain', but arm-false clear-
            # ed it; the union should drop because at runtime
            # arm-false's path makes r untainted.
            unioned_modal_origin: dict[str, str] = {}
            # Collect all (name → set of observed kinds) across
            # pre-match + each arm-result. Names with a single
            # consistent kind across all observations keep that
            # kind; names with multi-kind divergence drop out.
            observed_kinds: dict[str, set[str]] = {}
            for name, kind in modal_origin_pre_match.items():
                observed_kinds.setdefault(name, set()).add(kind)
            for arm_result in modal_origin_arm_results:
                for name, kind in arm_result.items():
                    observed_kinds.setdefault(name, set()).add(kind)
            # Stage 52 closure gate-6 latent-bug post-fix: only
            # mark a name as cleared if NO arm INSTALLED taint
            # for it. Symmetric with A.If's installed_names guard.
            # Stage 52 closure gate-7 HIGH-3 post-fix: ALSO check
            # `kept_somewhere_match` — any name preserved in any
            # arm's result (e.g. an empty arm `false => {}` that
            # preserves pre-match state) overrides cleared. Mirror
            # of A.If kept_somewhere. Closes gate-7 silent-failure
            # for the match case (the if-no-else analogue).
            # Stage 52 closure gate-8 type-design MEDIUM-1 polish:
            # dropped redundant installed_names_match check (strict
            # subset of kept_somewhere_match). Mirror of A.If polish.
            kept_somewhere_match: set[str] = set()
            for arm_result in modal_origin_arm_results:
                kept_somewhere_match.update(arm_result.keys())
            cleared_names_match: set[str] = set()
            for i, assigns in enumerate(arm_assigns):
                for name in assigns:
                    if (name not in modal_origin_arm_results[i]
                            and name not in kept_somewhere_match):
                        cleared_names_match.add(name)
            for name, kinds in observed_kinds.items():
                if name in cleared_names_match:
                    continue  # arm cleared → drop static claim
                if len(kinds) == 1:
                    unioned_modal_origin[name] = next(iter(kinds))
                # else: multi-kind divergence → drop (no static
                # claim). TODO(stage52-inc4): consider a richer
                # `Union[str, frozenset[str]]` dict shape that
                # carries the conflict for a "could be any of
                # {U, K}" diagnostic. For now the safe behaviour
                # is to drop the static claim.
            self._modal_origin_provenance = unioned_modal_origin
            self._check_match_exhaustive(expr, scrut_ty)
            if not arm_tys:
                return TyUnit()
            first = arm_tys[0]
            for i, t in enumerate(arm_tys[1:], start=1):
                if not self._compatible(first, t):
                    self.errors.append(TypeError_(
                        f"match arm {i} body type {self._fmt(t)} incompatible "
                        f"with arm 0 type {self._fmt(first)}",
                        expr.arms[i].span,
                    ))
            return self._join_branch_types(arm_tys, expr.span)
        if isinstance(expr, A.For):
            iter_ty = self._check_expr(expr.iter_expr, scope)
            inner = Scope(parent=scope)
            # Loop variable inherits the iterator's element type. For a
            # range expression `0..n` the iter_expr is currently typed
            # as i32 (or whatever the operands are); fall back to i64
            # only when we can't determine a concrete element type.
            loop_var_ty = iter_ty if iter_ty is not None else TyPrim("i64")
            inner.define(expr.var_name, loop_var_ty)
            self._check_loop_body_with_modal_union(expr.body, inner)
            return TyUnit()
        if isinstance(expr, A.While):
            self._check_expr(expr.cond, scope)
            self._check_loop_body_with_modal_union(expr.body, scope)
            return TyUnit()
        if isinstance(expr, A.Loop):
            self._check_loop_body_with_modal_union(expr.body, scope)
            return TyUnit()
        if isinstance(expr, A.Range):
            if expr.start is not None:
                self._check_expr(expr.start, scope)
            if expr.end is not None:
                self._check_expr(expr.end, scope)
            return TyUnknown(hint="range")
        if isinstance(expr, A.Assign):
            r = self._check_expr(expr.value, scope)
            target_ty = self._check_expr(expr.target, scope)
            self._check_assignment_target(expr, scope)
            # Stage 46 closure gate-3 silent-failure G3-F1 fix:
            # invalidate Result constructor provenance on
            # mutable reassignment. Pre-fix, `let mut r:
            # Result<i32, i32> = Ok(7); r = Err(99);
            # unwrap_ok(r)` typechecked clean because the
            # provenance map still held "ok" from the original
            # let. Same defect class as the gate-2 typed-let
            # case; this one was exposed by the gate-3 mutable-
            # reassignment probe. The fix: when the assign
            # target is a bare Name and the RHS is a direct
            # Ok(...)/Err(...) call, overwrite the provenance;
            # otherwise pop the entry so a non-constructor
            # reassignment clears the stale provenance rather
            # than silently keeping it.
            # Gate-5 G4-F2 fix: record the assign event regardless
            # of whether the target name is currently tracked in
            # the prov dict. The restore filter at _check_block /
            # _check_expr_in_block_scope intersects with the saved
            # snapshot, so non-Result names are harmless. The
            # per-event mask closes the ASSIGN-then-LET-shadow
            # hole gate-3's let-set alone could not detect.
            if (expr.op == "="
                    and isinstance(expr.target, A.Name)
                    and self._result_assigns_block_scopes):
                self._result_assigns_block_scopes[-1].add(
                    expr.target.name)
            if (expr.op == "="
                    and isinstance(expr.target, A.Name)
                    and expr.target.name
                        in self._result_constructor_provenance):
                if (isinstance(expr.value, A.Call)
                        and isinstance(expr.value.callee, A.Name)
                        and expr.value.callee.name in ("Ok", "Err")):
                    self._result_constructor_provenance[
                        expr.target.name] = (
                        "ok" if expr.value.callee.name == "Ok"
                        else "err"
                    )
                else:
                    self._result_constructor_provenance.pop(
                        expr.target.name, None)
            # Stage 52 closure gate-1 CRITICAL C1 + silent-failure
            # HIGH-1 + HIGH-3 fix: Assign-arm symmetric POP+POPULATE
            # for _modal_origin_provenance.
            #
            # C1 (POP on opaque RHS): `let mut r = from_uncertain(u);
            #   r = some_known_call(); into_known(r)` — clear stale
            #   taint on Assign-to-non-from_X.
            # HIGH-1+3 (POPULATE on from_X(...) RHS): `let mut r:
            #   i32 = 0; match b { true => { r = from_uncertain(u); }
            #   ... }; into_known(r)` — Assign-to-from_X must INSTALL
            #   taint, not just clear existing. Pre-HIGH-1 fix the
            #   populate only fired when the name was ALREADY tainted
            #   (Stage 46 G3-F1 pattern). For modal-origin, fresh
            #   taint via Assign IS a real launder vector and must
            #   install the entry.
            #
            # TODO(stage52-inc4): the populate is structural-
            # syntactic (must be a direct `Call(Name(from_X), ...)`).
            # RHS of `A.If` / `A.Block` / `A.Match` that YIELDS a
            # from_X value is gate-1 silent-failure HIGH-2 —
            # requires a `_yields_from_call(expr) -> str|None`
            # recursive helper that walks all terminal branches.
            # (Renamed from stage52-inc2 per gate-2 type-design F8.)
            if (expr.op == "="
                    and isinstance(expr.target, A.Name)):
                # Stage 52 closure gate-3 NEW-HIGH-2/3/4 fix:
                # record this Assign in the assigns-set
                # regardless of whether the RHS installs taint.
                # Union sites (if-else, match) use this to detect
                # "branch overwrote name with non-modal value"
                # which should drop the pre-state's static claim
                # (else the union over-claims taint that the
                # runtime might never carry).
                if self._modal_origin_assigns_block_scopes:
                    self._modal_origin_assigns_block_scopes[-1].add(
                        expr.target.name)
                # Stage 52 closure gate-6 CRITICAL-2 fix: unified
                # taint-source lookup via _modal_origin_of_expr.
                # Handles both direct from_X(...) RHS (Inc 2 path)
                # AND name-alias RHS (gate-6 fix: `r = s;` where
                # s is tainted now installs taint on r). Pop on
                # opaque RHS unchanged.
                assign_rhs_kind = self._modal_origin_of_expr(expr.value)
                if assign_rhs_kind is not None:
                    # POPULATE (or overwrite existing) — taint
                    # installs unconditionally on from_X(...) RHS
                    # or name-alias of a tainted source.
                    self._modal_origin_provenance[
                        expr.target.name] = assign_rhs_kind
                elif expr.target.name in self._modal_origin_provenance:
                    # POP on opaque RHS — clear stale taint when the
                    # assigned value's modal origin is no longer
                    # statically determinable. Only fires when there
                    # WAS a prior taint to clear (avoids polluting
                    # the dict with names that were never tainted).
                    self._modal_origin_provenance.pop(
                        expr.target.name, None)
            if expr.op != "=":
                op = {
                    "+=": "+",
                    "-=": "-",
                    "*=": "*",
                    "/=": "/",
                    "%=": "%",
                }.get(expr.op)
                if op is not None:
                    self._check_plain_binary_scalar_compat(
                        target_ty, r, op, expr.span)
            if not self._compatible(r, target_ty):
                self.errors.append(TypeError_(
                    f"assignment target type {self._fmt(target_ty)} "
                    f"incompatible with value type {self._fmt(r)}",
                    expr.span,
                    hint="use an explicit cast on the assigned value",
                ))
            elif expr.op == "=":
                self._check_refinement_contextual_value(
                    expr.value, r, target_ty, expr.span, "assignment",
                    scope,
                )
            elif self._contains_refinement(target_ty):
                self.errors.append(TypeError_(
                    f"compound assignment to refined type "
                    f"{self._fmt(target_ty)} requires proof support beyond "
                    f"Stage 31 constants",
                    expr.span,
                    hint="assign an explicitly proven refined value instead",
                ))
            if expr.op == "=" and isinstance(expr.target, A.Name):
                self._clear_local_const_index_unrepresentable(
                    expr.target.name)
                assigned_unrepresentable = (
                    self._expr_has_unrepresentable_typed_const_scalar(
                        expr.value)
                )
                if (assigned_unrepresentable
                        and self._mark_array_literal_unrepresentable_elements(
                            expr.target.name, expr.value)):
                    self._set_local_const_unrepresentable(
                        expr.target.name, False)
                else:
                    self._set_local_const_unrepresentable(
                        expr.target.name,
                        assigned_unrepresentable,
                        self._expr_unrepresentable_typed_const_scalar_base(
                            expr.value)
                        if assigned_unrepresentable else None,
                    )
            elif (expr.op == "="
                  and isinstance(expr.target, A.Index)
                  and isinstance(expr.target.callee, A.Name)):
                assigned_unrepresentable = (
                    self._expr_has_unrepresentable_typed_const_scalar(
                        expr.value)
                )
                indexed_key = self._simple_local_const_index_key(expr.target)
                if indexed_key is not None:
                    aggregate_name, key = indexed_key
                    self._set_local_const_unrepresentable(
                        key,
                        assigned_unrepresentable,
                        self._expr_unrepresentable_typed_const_scalar_base(
                            expr.value)
                        if assigned_unrepresentable else None,
                        anchor_name=aggregate_name,
                    )
                elif assigned_unrepresentable:
                    self._set_local_const_unrepresentable(
                        expr.target.callee.name,
                        True,
                        self._expr_unrepresentable_typed_const_scalar_base(
                            expr.value),
                    )
            return TyUnit()
        if isinstance(expr, A.TupleLit):
            return TyTuple(tuple(self._check_expr(e, scope) for e in expr.elems))
        if isinstance(expr, A.ArrayLit):
            ts = [self._check_expr(e, scope) for e in expr.elems]
            elem = ts[0] if ts else TyUnknown(hint="empty array")
            for t in ts[1:]:
                if not self._compatible(t, elem):
                    self.errors.append(TypeError_(
                        f"array literal element type {self._fmt(t)} "
                        f"incompatible with first element type "
                        f"{self._fmt(elem)}",
                        expr.span,
                        hint="use an explicit cast so every array element "
                             "has the same type",
                    ))
                elif ((self._contains_refined_function(elem)
                       or self._contains_refined_function(t))
                      and not self._refinement_shape_exact(elem, t)):
                    self.errors.append(TypeError_(
                        f"array literal function element types "
                        f"{self._fmt(elem)} and {self._fmt(t)} differ in "
                        f"refined parameter or return requirements in "
                        f"Stage 31",
                        expr.span,
                        hint="use function elements with exactly matching "
                             "refinements",
                    ))
                    elem = self._erase_refinement(elem)
                elif (self._contains_refinement(elem)
                      and not self._refinement_proof_carried(t, elem)):
                    elem = self._erase_refinement(elem)
            return TyArray(elem, TyPrim(f"size_{len(ts)}"))
        if isinstance(expr, A.StructLit):
            decl = getattr(self, "_struct_decls", {}).get(expr.name)
            if decl is None:
                self.errors.append(TypeError_(
                    f"unknown struct {expr.name!r}", expr.span,
                ))
                # Still type-check the field values for downstream errors.
                for _, v in expr.fields:
                    self._check_expr(v, scope)
                return TyUnknown(hint=f"struct {expr.name}")
            # Verify every required field is present and no extras.
            decl_fields = {p.name for p in decl.fields}
            given_fields = {fname for fname, _ in expr.fields}
            missing = decl_fields - given_fields
            extra = given_fields - decl_fields
            if missing:
                self.errors.append(TypeError_(
                    f"struct {expr.name!r}: missing field(s) "
                    f"{sorted(missing)}", expr.span,
                ))
            if extra:
                self.errors.append(TypeError_(
                    f"struct {expr.name!r}: unknown field(s) "
                    f"{sorted(extra)}", expr.span,
                ))
            # Type-check each given field's value against the declared type.
            field_decl_by_name = {p.name: p for p in decl.fields}
            for fname, fval in expr.fields:
                v_ty = self._check_expr(fval, scope)
                p = field_decl_by_name.get(fname)
                if p is not None:
                    expected = self._resolve_type(p.ty, scope)
                    if not self._compatible(v_ty, expected):
                        self.errors.append(TypeError_(
                            f"struct {expr.name!r}.{fname}: expected "
                            f"{self._fmt(expected)}, got {self._fmt(v_ty)}",
                            fval.span,
                        ))
                    else:
                        self._check_refinement_contextual_value(
                            fval, v_ty, expected, fval.span,
                            f"struct {expr.name!r}.{fname}",
                            scope,
                        )
            return TyStruct(name=expr.name)
        if isinstance(expr, (A.Break, A.Continue)):
            return TyUnit()
        if isinstance(expr, A.Return):
            if expr.value is not None:
                value_ty = self._check_expr(expr.value, scope)
                if not self._compatible(value_ty, self._current_return_ty):
                    self.errors.append(TypeError_(
                        f"return value of function "
                        f"{self._current_fn_name!r}: expected "
                        f"{self._fmt(self._current_return_ty)}, got "
                        f"{self._fmt(value_ty)}",
                        expr.span,
                    ))
                elif ((self._contains_refinement(self._current_return_ty)
                       or self._contains_refinement(value_ty))
                      and not isinstance(self._current_return_ty, TyUnknown)):
                    self._check_refinement_contextual_value(
                        expr.value, value_ty, self._current_return_ty,
                        expr.span,
                        f"return value of function {self._current_fn_name!r}",
                        scope,
                    )
                elif self._check_unrepresentable_scalar_context(
                    expr.value,
                    self._current_return_ty,
                    expr.span,
                    f"return value of function {self._current_fn_name!r}",
                    report=False,
                ):
                    self._unrepresentable_scalar_return_functions.add(
                        self._current_fn_name)
                if self._current_is_kernel:
                    self.errors.append(TypeError_(
                        "@kernel functions cannot return a value in PTX",
                        expr.span,
                    ))
            return TyUnit()
        if isinstance(expr, A.UnsafeBlock):
            # Audit 28.8 B3: track unsafe-context depth so Cast checks
            # below know whether they're inside an unsafe region.
            # Stage 28.6's outer pass (unsafe_pass.check_unsafe_ops)
            # handles `*ptr` deref outside unsafe; the type-level
            # gate (Cast int→ptr) lives here so the checks compose.
            self._in_unsafe_depth += 1
            try:
                body_ty = self._check_block(expr.body, scope) \
                    if isinstance(expr.body, A.Block) \
                    else self._check_expr(expr.body, scope)
            finally:
                self._in_unsafe_depth -= 1
            return body_ty
        if isinstance(expr, A.Cast):
            src_ty = self._check_expr(expr.value, scope)
            tgt_ty = self._resolve_type(expr.target_ty, scope)
            if isinstance(tgt_ty, TyRefined):
                before_cast_errors = len(self.errors)
                self._check_cast_compat(
                    self._erase_refinement(src_ty),
                    self._erase_refinement(tgt_ty),
                    expr.span,
                )
                if len(self.errors) != before_cast_errors:
                    return TyUnknown(hint="invalid refined cast")
                if self._refinement_proof_carried(src_ty, tgt_ty):
                    self._record_refinement_proof_carries_for_type(
                        f"cast to refined type {self._fmt(tgt_ty)}",
                        src_ty,
                        tgt_ty,
                        expr.span,
                    )
                    return tgt_ty
                if not self._check_refinement_cast_value(
                    expr.value, src_ty, tgt_ty, expr.span,
                    f"cast to refined type {self._fmt(tgt_ty)}",
                    scope,
                ):
                    return TyUnknown(hint="failed refined cast")
                return tgt_ty
            if self._contains_refinement(tgt_ty):
                self.errors.append(TypeError_(
                    f"cast to {self._fmt(tgt_ty)} would change refined "
                    f"parameter or return requirements in Stage 31",
                    expr.span,
                    hint="construct refined composite values explicitly so "
                         "the checker can verify their proofs",
                ))
                return TyUnknown(hint="invalid refined composite cast")
            # Audit 28.8 B3 (trap 28603): raw-pointer casts must be in
            # an unsafe block. `int as *mut T` outside unsafe is a
            # forged pointer; `float as *T` is dubious even inside
            # unsafe (no defined coercion). The pre-fix Cast handler
            # accepted both silently and the unsafe-pass walker only
            # matched syntactic Unary deref — so a cast-formed pointer
            # could escape every gate.
            if isinstance(tgt_ty, TyPtr):
                src_is_ptr_like = isinstance(src_ty, (TyPtr, TyRef))
                src_is_float = (isinstance(src_ty, TyPrim)
                                and src_ty.name in ("bf16", "f16", "f32",
                                                    "f64", "fp8"))
                if src_is_float:
                    # Float→ptr blocked even inside unsafe.
                    self.errors.append(TypeError_(
                        f"cast from {self._fmt(src_ty)} to "
                        f"{self._fmt(tgt_ty)}: float→pointer is not a "
                        f"valid coercion (trap 28603)",
                        expr.span,
                        hint="use a bitcast through a u64 intermediate "
                             "if you really mean to read the bit pattern",
                    ))
                elif (not src_is_ptr_like
                      and self._in_unsafe_depth == 0):
                    self.errors.append(TypeError_(
                        f"raw-pointer cast from {self._fmt(src_ty)} to "
                        f"{self._fmt(tgt_ty)} outside unsafe block "
                        f"(trap 28603)",
                        expr.span,
                        hint="wrap this cast in `unsafe { ... }` to "
                             "acknowledge the capability requirement",
                    ))
            else:
                # Audit 28.8 B14 (trap 28604): allowed-cast matrix.
                # B3 covers ptr-targeted casts above; this branch
                # covers non-ptr targets — int/float/etc. Pre-fix the
                # Cast handler accepted *anything*: tuple-as-i32,
                # struct-as-f64, unit-as-pointer all silently went
                # through and codegen produced garbage. The matrix
                # enforces:
                #   int <-> int   (any widths)
                #   int <-> float (any widths)
                #   float <-> float (any widths)
                #   bool -> int   (1/0)
                #   int -> bool   (truthiness)
                #   char <-> int  (codepoint)
                #   *T -> integer (usize/u64) [inside unsafe; handled by B3]
                # Anything else: trap 28604.
                self._check_cast_compat(src_ty, tgt_ty, expr.span)
            return tgt_ty
        # Audit 28.8 B10: typecheck Quote/Splice/Modify arms. Pre-fix
        # they fell through to `TyUnknown(hint='unhandled Quote/...')`,
        # which compatible-with-everything per `_compatible` — silent
        # type-pun at every let-binding site. Now:
        #   Quote(inner) -> TyQuote(typeof(inner))
        #   Splice(inner) -> typeof(inner) when inner is TyQuote;
        #                    diagnostic 11001 otherwise
        #   Modify(target, transformation, verifier) -> typeof(target)
        if isinstance(expr, A.Quote):
            inner_ty = self._check_expr(expr.inner, scope)
            return TyQuote(inner=inner_ty)
        if isinstance(expr, A.Splice):
            inner_ty = self._check_expr(expr.inner, scope)
            if isinstance(inner_ty, TyQuote):
                return inner_ty.inner
            if isinstance(inner_ty, TyUnknown):
                # Cascade-safe: don't fire a second diagnostic if the
                # inner was already unbound or had an upstream error.
                return inner_ty
            self.errors.append(TypeError_(
                f"splice() requires a Quote value; got {self._fmt(inner_ty)} "
                f"(trap 11001)",
                expr.span,
                hint="wrap the argument in `quote(...)` first",
            ))
            return TyUnknown(hint="splice of non-Quote")
        if isinstance(expr, A.Modify):
            # Audit 28.8 B10: typecheck Modify's three sub-exprs (so any
            # internal errors surface), then return i32 — the runtime
            # semantics (verifier-gated cell write) yields 1 on apply,
            # 0 on reject. Pre-fix this fell through to TyUnknown.
            self._check_expr(expr.target, scope)
            self._check_expr(expr.transformation, scope)
            self._check_expr(expr.verifier, scope)
            return TyPrim("i32")
        return TyUnknown(hint=f"unhandled {type(expr).__name__}")

    # ---- bounds checking ----
    _INT_BOUNDS = {
        "i8":  (-(1 << 7),  (1 << 7) - 1),
        "i16": (-(1 << 15), (1 << 15) - 1),
        "i32": (-(1 << 31), (1 << 31) - 1),
        "i64": (-(1 << 63), (1 << 63) - 1),
        "u8":  (0, (1 << 8) - 1),
        "u16": (0, (1 << 16) - 1),
        "u32": (0, (1 << 32) - 1),
        "u64": (0, (1 << 64) - 1),
        "isize": (-(1 << 63), (1 << 63) - 1),
        "usize": (0, (1 << 64) - 1),
    }

    def _check_int_lit_fits(self, lit: A.IntLit, ty: "TyPrim") -> None:
        """Static check: the literal value must fit in the declared width.
        On overflow, surface a typecheck error with did-you-mean for a
        wider type. The literal's own type_suffix takes precedence over
        the contextual `ty` (audit-10 #2: `let x: i32 = 5_000_000_000_i64`
        should be checked against i64, not i32)."""
        # If the literal has a suffix, use that as the actual type; the
        # contextual `ty` only applies to suffix-less literals.
        eff_name = lit.type_suffix or ty.name
        bounds = self._INT_BOUNDS.get(eff_name)
        if bounds is None:
            return
        # Refit the rest of the function against eff_name.
        ty = TyPrim(eff_name)
        bounds = self._INT_BOUNDS.get(ty.name)
        if bounds is None:
            return
        lo, hi = bounds
        v = lit.value
        if v < lo or v > hi:
            wider = self._suggest_wider_int(v, ty.name)
            hint = (f"use `{wider}` instead" if wider else None)
            self.errors.append(TypeError_(
                f"value {v} does not fit in {ty.name} (range "
                f"{lo}..={hi})", lit.span, hint=hint,
            ))

    def _check_refinement_const_value(
        self, value_expr: A.Expr, value_ty: Type, refined: "TyRefined",
        span: A.Span, context: str, scope: Scope,
    ) -> None:
        source_base = self._const_eval_numeric_base(value_ty)
        value, source_unrepresentable = (
            self._eval_refinement_source_scalar(value_expr, source_base)
        )
        target_base = self._erase_refinement(refined)
        represented = (
            None if source_unrepresentable
            else self._cast_const_scalar_to_type(value, target_base)
        )
        if self._expr_uses_invalid_refined_return(value_expr):
            self._record_unproven_refinement_obligations(
                context, refined, span)
            self.errors.append(TypeError_(
                f"{context}: refinement {refined.name} depends on a "
                f"failed refined-return producer in Stage 34",
                span,
                hint="repair the producer's refined return proof before "
                     "carrying its value into another refinement",
            ))
            return
        if value is None and not source_unrepresentable:
            pending = self._check_self_independent_refinement(
                refined, span, context,
            )
            if pending:
                for pred in pending:
                    self._record_refinement_obligation(
                        context, refined, pred, "unproven", span, None,
                    )
                self.errors.append(TypeError_(
                    f"{context}: refinement {refined.name} requires a "
                    f"compile-time-proven value in Stage 31; could not prove "
                    f"{' and '.join(self._fmt_refinement_expr(p) for p in pending)}",
                    span,
                    hint="use a literal that satisfies the refinement for now; "
                         "SMT/runtime proof support is a later Stage 31 step",
                ))
            if isinstance(refined.base, TyRefined):
                self._check_refinement_const_value(
                    value_expr, value_ty, refined.base, span, context, scope,
                )
            return
        if represented is None:
            pending = self._check_self_independent_refinement(
                refined, span, context,
            )
            for pred in pending:
                self._record_refinement_obligation(
                    context, refined, pred, "unproven", span, None,
                )
            detail = (
                f"; could not prove "
                f"{' and '.join(self._fmt_refinement_expr(p) for p in pending)}"
                if pending else ""
            )
            self.errors.append(TypeError_(
                f"{context}: refinement {refined.name} requires a "
                f"representable target value in Stage 34{detail} for "
                f"target base {self._fmt(target_base)}",
                span,
                hint="refined values must be representable by their erased "
                     "base type before they can become refined values",
            ))
            if isinstance(refined.base, TyRefined):
                self._check_refinement_const_value(
                    value_expr, value_ty, refined.base, span, context, scope,
                )
            return
        for pred in refined.predicates:
            ok = self._eval_refinement_predicate(
                pred, represented, numeric_base=target_base)
            if ok is None:
                self._record_refinement_obligation(
                    context, refined, pred, "unsupported", span, represented,
                )
                self.errors.append(TypeError_(
                    f"{context}: refinement {refined.name} predicate "
                    f"{self._fmt_refinement_expr(pred)} is not supported by "
                    f"the Stage 31 constant checker",
                    span,
                ))
                continue
            if not ok:
                self._record_refinement_obligation(
                    context, refined, pred, "failed", span, represented,
                    trap="31001",
                )
                self.errors.append(TypeError_(
                    f"{context}: refinement {refined.name} violated: "
                    f"value {self._fmt_scalar_value(represented)} does not "
                    f"satisfy {self._fmt_refinement_expr(pred)} "
                    f"(trap 31001)",
                    span,
                    hint="refined values must satisfy their `where` predicate",
                ))
            else:
                self._record_refinement_obligation(
                    context, refined, pred, "proved", span, represented,
                )
        if isinstance(refined.base, TyRefined):
            self._check_refinement_const_value(
                value_expr, value_ty, refined.base, span, context, scope,
            )

    def _check_refinement_cast_value(
        self, value_expr: A.Expr, src_ty: Type, refined: "TyRefined",
        span: A.Span, context: str, scope: Scope,
    ) -> bool:
        source_base = self._const_eval_numeric_base(src_ty)
        value, source_unrepresentable = (
            self._eval_refinement_source_scalar(value_expr, source_base)
        )
        target_base = self._erase_refinement(refined)
        converted = (
            None if source_unrepresentable
            else self._cast_const_scalar_to_type(value, target_base)
        )
        invalid_source = self._expr_uses_invalid_refined_return(value_expr)
        proved = True
        if converted is None:
            pending = [] if invalid_source else (
                self._check_self_independent_refinement(
                    refined, span, context,
                )
            )
            if value is not None or source_unrepresentable or pending or invalid_source:
                proved = False
                for pred in pending:
                    self._record_refinement_obligation(
                        context, refined, pred, "unproven", span, None,
                    )
                if invalid_source:
                    detail = "source depends on a failed refined-return producer"
                    if pending:
                        detail += (
                            "; could not prove "
                            f"{' and '.join(self._fmt_refinement_expr(p) for p in pending)}"
                        )
                elif value is not None or source_unrepresentable:
                    detail = "value is not representable"
                    if pending:
                        detail += (
                            "; could not prove "
                            f"{' and '.join(self._fmt_refinement_expr(p) for p in pending)}"
                        )
                else:
                    detail = (
                        "could not prove "
                        f"{' and '.join(self._fmt_refinement_expr(p) for p in pending)}"
                    )
                self.errors.append(TypeError_(
                    f"{context}: refinement {refined.name} requires a "
                    f"compile-time-proven target value in Stage 34; "
                    f"{detail} after casting {self._fmt(src_ty)} to "
                    f"{self._fmt(target_base)}",
                    span,
                    hint="cast to the refined type only when the target "
                         "value can be represented and proven to satisfy "
                         "the predicate",
                ))
            if isinstance(refined.base, TyRefined):
                proved = self._check_refinement_cast_value(
                    value_expr, src_ty, refined.base, span, context, scope,
                ) and proved
            return proved
        for pred in refined.predicates:
            ok = self._eval_refinement_predicate(
                pred, converted, numeric_base=target_base)
            if ok is None:
                proved = False
                self._record_refinement_obligation(
                    context, refined, pred, "unsupported", span, converted,
                )
                self.errors.append(TypeError_(
                    f"{context}: refinement {refined.name} predicate "
                    f"{self._fmt_refinement_expr(pred)} is not supported by "
                    f"the Stage 34 cast checker",
                    span,
                ))
                continue
            if not ok:
                proved = False
                self._record_refinement_obligation(
                    context, refined, pred, "failed", span, converted,
                    trap="31001",
                )
                self.errors.append(TypeError_(
                    f"{context}: refinement {refined.name} violated: "
                    f"target value {self._fmt_scalar_value(converted)} "
                    f"does not satisfy {self._fmt_refinement_expr(pred)} "
                    f"(trap 31001)",
                    span,
                    hint="refined casts must satisfy their `where` "
                         "predicate after target conversion",
                ))
            else:
                self._record_refinement_obligation(
                    context, refined, pred, "proved", span, converted,
                )
        if isinstance(refined.base, TyRefined):
            proved = self._check_refinement_cast_value(
                value_expr, src_ty, refined.base, span, context, scope,
            ) and proved
        return proved

    def _expr_uses_invalid_refined_return(self, expr: A.Expr) -> bool:
        invalid = getattr(self, "_invalid_refined_return_functions", set())
        if isinstance(expr, A.Call):
            if isinstance(expr.callee, A.Name) and expr.callee.name in invalid:
                return True
            return (self._expr_uses_invalid_refined_return(expr.callee)
                    or any(self._expr_uses_invalid_refined_return(arg)
                           for arg in expr.args))
        if isinstance(expr, A.Cast):
            return self._expr_uses_invalid_refined_return(expr.value)
        if isinstance(expr, A.Unary):
            return self._expr_uses_invalid_refined_return(expr.operand)
        if isinstance(expr, A.Binary):
            return (self._expr_uses_invalid_refined_return(expr.left)
                    or self._expr_uses_invalid_refined_return(expr.right))
        if isinstance(expr, A.TupleLit):
            return any(self._expr_uses_invalid_refined_return(e)
                       for e in expr.elems)
        if isinstance(expr, A.ArrayLit):
            return any(self._expr_uses_invalid_refined_return(e)
                       for e in expr.elems)
        if isinstance(expr, A.Field):
            return self._expr_uses_invalid_refined_return(expr.obj)
        if isinstance(expr, A.Index):
            return (self._expr_uses_invalid_refined_return(expr.callee)
                    or any(self._expr_uses_invalid_refined_return(i)
                           for i in expr.indices))
        if isinstance(expr, A.StructLit):
            return any(self._expr_uses_invalid_refined_return(v)
                       for _, v in expr.fields)
        if isinstance(expr, A.Assign):
            return (self._expr_uses_invalid_refined_return(expr.target)
                    or self._expr_uses_invalid_refined_return(expr.value))
        return False

    def _const_eval_numeric_base(self, ty: Type) -> Type | None:
        base = self._erase_refinement(ty)
        if isinstance(base, TyPrim) and (
                base.name in _INT_PRIM_NAMES or base.name in _FLOAT_PRIM_NAMES):
            return base
        return None

    def _const_eval_type_node_base(self, ty: A.TyNode) -> Type | None:
        if isinstance(ty, A.TyName) and (
                ty.name in _INT_PRIM_NAMES
                or ty.name in _FLOAT_PRIM_NAMES
                or ty.name == "bool"):
            return TyPrim(ty.name)
        return None

    def _eval_refinement_source_scalar(
        self, expr: A.Expr, source_base: Type | None,
    ) -> tuple[int | float | bool | None, bool]:
        value = self._eval_const_scalar_expr(
            expr, None, use_local_consts=True,
            honor_float_suffix=True, numeric_base=source_base)
        if value is not None:
            return value, False

        source_unrepresentable = (
            self._expr_has_unrepresentable_typed_const_scalar(expr)
        )
        raw_value = self._eval_raw_const_scalar_fallback(expr)
        if raw_value is None:
            return None, source_unrepresentable

        if source_unrepresentable:
            return raw_value, True

        if source_base is None:
            return raw_value, False

        source_represented = self._cast_const_scalar_to_type(
            raw_value, source_base)
        if source_represented is None:
            return raw_value, True
        return None, False

    def _infer_const_expr_numeric_base(self, expr: A.Expr) -> Type | None:
        if isinstance(expr, A.IntLit):
            suffix = expr.type_suffix or "i32"
            if suffix in _INT_PRIM_NAMES:
                return TyPrim(suffix)
            return None
        if isinstance(expr, A.FloatLit):
            suffix = expr.type_suffix or "f32"
            if suffix in _FLOAT_PRIM_NAMES:
                return TyPrim(suffix)
            return None
        if isinstance(expr, A.BoolLit):
            return TyPrim("bool")
        if isinstance(expr, A.Cast):
            return self._const_eval_type_node_base(expr.target_ty)
        if isinstance(expr, A.Unary):
            return self._infer_const_expr_numeric_base(expr.operand)
        if isinstance(expr, A.Binary):
            return self._infer_const_expr_numeric_base(expr.left)
        return None

    def _expr_has_unrepresentable_typed_const_scalar(
        self, expr: A.Expr,
    ) -> bool:
        if isinstance(expr, A.Name) and not expr.generics:
            found_local, local_unrepresentable = (
                self._lookup_local_const_unrepresentable(expr.name)
            )
            if found_local:
                return local_unrepresentable
            return expr.name in self._unrepresentable_const_scalar_names
        base = self._infer_const_expr_numeric_base(expr)
        if base is not None:
            typed_value = self._eval_const_scalar_expr(
                expr, None, use_local_consts=True,
                honor_float_suffix=True, numeric_base=base)
            if typed_value is None:
                raw_value = self._eval_raw_const_scalar_fallback(expr)
                if (raw_value is not None
                        and self._cast_const_scalar_to_type(
                            raw_value, base) is None):
                    return True
        if isinstance(expr, A.Cast):
            return self._expr_has_unrepresentable_typed_const_scalar(
                expr.value)
        if isinstance(expr, A.Unary):
            return self._expr_has_unrepresentable_typed_const_scalar(
                expr.operand)
        if isinstance(expr, A.Binary):
            return (
                self._expr_has_unrepresentable_typed_const_scalar(expr.left)
                or self._expr_has_unrepresentable_typed_const_scalar(expr.right)
            )
        if isinstance(expr, A.If):
            branches = [expr.then]
            if expr.else_ is not None:
                branches.append(expr.else_)
            return any(
                self._expr_has_unrepresentable_typed_const_scalar(branch)
                for branch in branches
            )
        if isinstance(expr, A.Match):
            return any(
                self._expr_has_unrepresentable_typed_const_scalar(arm.body)
                for arm in expr.arms
            )
        if isinstance(expr, A.Block):
            return (
                any(
                    self._stmt_has_unrepresentable_typed_const_scalar(stmt)
                    for stmt in expr.stmts
                )
                or (
                    expr.final_expr is not None
                    and self._expr_has_unrepresentable_typed_const_scalar(
                        expr.final_expr)
                )
            )
        if isinstance(expr, A.TupleLit):
            return any(
                self._expr_has_unrepresentable_typed_const_scalar(elem)
                for elem in expr.elems
            )
        if isinstance(expr, A.ArrayLit):
            return any(
                self._expr_has_unrepresentable_typed_const_scalar(elem)
                for elem in expr.elems
            )
        if isinstance(expr, A.StructLit):
            return any(
                self._expr_has_unrepresentable_typed_const_scalar(value)
                for _, value in expr.fields
            )
        if isinstance(expr, A.Field):
            return self._expr_has_unrepresentable_typed_const_scalar(expr.obj)
        if isinstance(expr, A.Index):
            indexed_key = self._simple_local_const_index_key(expr)
            if indexed_key is not None:
                _, key = indexed_key
                found_local, local_unrepresentable = (
                    self._lookup_local_const_unrepresentable(key)
                )
                if found_local and local_unrepresentable:
                    return True
            return (
                self._expr_has_unrepresentable_typed_const_scalar(expr.callee)
                or any(
                    self._expr_has_unrepresentable_typed_const_scalar(index)
                    for index in expr.indices
                )
            )
        if isinstance(expr, A.Call):
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name in getattr(
                        self,
                        "_unrepresentable_scalar_return_functions",
                        set(),
                    )):
                return True
            return (
                self._expr_has_unrepresentable_typed_const_scalar(expr.callee)
                or any(
                    self._expr_has_unrepresentable_typed_const_scalar(arg)
                    for arg in expr.args
                )
            )
        if isinstance(expr, A.Assign):
            return (
                self._expr_has_unrepresentable_typed_const_scalar(expr.target)
                or self._expr_has_unrepresentable_typed_const_scalar(expr.value)
            )
        return False

    def _expr_unrepresentable_typed_const_scalar_base(
        self, expr: A.Expr,
    ) -> Type | None:
        if isinstance(expr, A.Name) and not expr.generics:
            local_base = self._lookup_local_const_unrepresentable_base(
                expr.name)
            if local_base is not None:
                return local_base
            if expr.name in self._unrepresentable_const_scalar_names:
                decl = self._const_decls.get(expr.name)
                if decl is not None:
                    return self._const_index_target_type(decl)
            return None
        base = self._infer_const_expr_numeric_base(expr)
        if base is not None:
            typed_value = self._eval_const_scalar_expr(
                expr, None, use_local_consts=True,
                honor_float_suffix=True, numeric_base=base)
            if typed_value is None:
                raw_value = self._eval_raw_const_scalar_fallback(expr)
                if (raw_value is not None
                        and self._cast_const_scalar_to_type(
                            raw_value, base) is None):
                    return base
        if isinstance(expr, A.Cast):
            return self._expr_unrepresentable_typed_const_scalar_base(
                expr.value)
        if isinstance(expr, A.Unary):
            return self._expr_unrepresentable_typed_const_scalar_base(
                expr.operand)
        if isinstance(expr, A.Binary):
            return (
                self._expr_unrepresentable_typed_const_scalar_base(expr.left)
                or self._expr_unrepresentable_typed_const_scalar_base(
                    expr.right)
            )
        if isinstance(expr, A.If):
            branches = [expr.then]
            if expr.else_ is not None:
                branches.append(expr.else_)
            for branch in branches:
                base = self._expr_unrepresentable_typed_const_scalar_base(
                    branch)
                if base is not None:
                    return base
            return None
        if isinstance(expr, A.Match):
            for arm in expr.arms:
                base = self._expr_unrepresentable_typed_const_scalar_base(
                    arm.body)
                if base is not None:
                    return base
            return None
        if isinstance(expr, A.Block):
            for stmt in expr.stmts:
                base = self._stmt_unrepresentable_typed_const_scalar_base(stmt)
                if base is not None:
                    return base
            if expr.final_expr is not None:
                return self._expr_unrepresentable_typed_const_scalar_base(
                    expr.final_expr)
            return None
        if isinstance(expr, A.TupleLit):
            for elem in expr.elems:
                base = self._expr_unrepresentable_typed_const_scalar_base(
                    elem)
                if base is not None:
                    return base
            return None
        if isinstance(expr, A.ArrayLit):
            for elem in expr.elems:
                base = self._expr_unrepresentable_typed_const_scalar_base(
                    elem)
                if base is not None:
                    return base
            return None
        if isinstance(expr, A.StructLit):
            for _, value in expr.fields:
                base = self._expr_unrepresentable_typed_const_scalar_base(
                    value)
                if base is not None:
                    return base
            return None
        if isinstance(expr, A.Field):
            return self._expr_unrepresentable_typed_const_scalar_base(expr.obj)
        if isinstance(expr, A.Index):
            indexed_key = self._simple_local_const_index_key(expr)
            if indexed_key is not None:
                _, key = indexed_key
                base = self._lookup_local_const_unrepresentable_base(key)
                if base is not None:
                    return base
            base = self._expr_unrepresentable_typed_const_scalar_base(
                expr.callee)
            if base is not None:
                return base
            for index in expr.indices:
                base = self._expr_unrepresentable_typed_const_scalar_base(
                    index)
                if base is not None:
                    return base
            return None
        if isinstance(expr, A.Call):
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name in getattr(
                        self,
                        "_unrepresentable_scalar_return_functions",
                        set(),
                    )):
                sig = self.functions.get(expr.callee.name)
                if sig is not None:
                    return self._const_eval_numeric_base(sig.ret)
            base = self._expr_unrepresentable_typed_const_scalar_base(
                expr.callee)
            if base is not None:
                return base
            for arg in expr.args:
                base = self._expr_unrepresentable_typed_const_scalar_base(arg)
                if base is not None:
                    return base
            return None
        if isinstance(expr, A.Assign):
            return (
                self._expr_unrepresentable_typed_const_scalar_base(expr.target)
                or self._expr_unrepresentable_typed_const_scalar_base(
                    expr.value)
            )
        return None

    def _stmt_has_unrepresentable_typed_const_scalar(
        self, stmt: A.Stmt,
    ) -> bool:
        if isinstance(stmt, A.Let):
            return (
                stmt.value is not None
                and self._expr_has_unrepresentable_typed_const_scalar(
                    stmt.value)
            )
        if isinstance(stmt, A.ConstStmt):
            return self._expr_has_unrepresentable_typed_const_scalar(
                stmt.value)
        if isinstance(stmt, A.ExprStmt):
            return self._expr_has_unrepresentable_typed_const_scalar(
                stmt.expr)
        return False

    def _stmt_unrepresentable_typed_const_scalar_base(
        self, stmt: A.Stmt,
    ) -> Type | None:
        if isinstance(stmt, A.Let) and stmt.value is not None:
            return self._expr_unrepresentable_typed_const_scalar_base(
                stmt.value)
        if isinstance(stmt, A.ConstStmt):
            return self._expr_unrepresentable_typed_const_scalar_base(
                stmt.value)
        if isinstance(stmt, A.ExprStmt):
            return self._expr_unrepresentable_typed_const_scalar_base(
                stmt.expr)
        return None

    def _eval_raw_const_scalar_fallback(
        self, expr: A.Expr,
    ) -> int | float | bool | None:
        return self._eval_raw_const_scalar_expr(
            expr, None, use_local_consts=True)

    def _eval_raw_const_scalar_expr(
        self, expr: A.Expr, self_value: int | float | bool | None,
        *, use_local_consts: bool = False,
    ) -> int | float | bool | None:
        if isinstance(expr, A.IntLit):
            return expr.value
        if isinstance(expr, A.FloatLit):
            return expr.value
        if isinstance(expr, A.BoolLit):
            return expr.value
        if isinstance(expr, A.Name) and expr.generics:
            return None
        if isinstance(expr, A.Name) and expr.name == "self":
            return self_value
        if isinstance(expr, A.Name):
            if use_local_consts:
                found_local, local_value = self._lookup_local_const_scalar(
                    expr.name)
                if found_local:
                    return local_value
            return self._const_scalar_values.get(expr.name)
        if isinstance(expr, A.Cast):
            inner_value = self._eval_raw_const_scalar_expr(
                expr.value, self_value, use_local_consts=use_local_consts)
            target_base = self._const_eval_type_node_base(expr.target_ty)
            if target_base is None:
                return None
            converted = self._cast_const_scalar_to_type(
                inner_value, target_base)
            if converted is None:
                return inner_value
            return converted
        if isinstance(expr, A.Unary) and expr.op == "-":
            inner = self._eval_raw_const_scalar_expr(
                expr.operand, self_value, use_local_consts=use_local_consts)
            if isinstance(inner, (int, float)) and not isinstance(inner, bool):
                return -inner
            return None
        if isinstance(expr, A.Binary):
            left = self._eval_raw_const_scalar_expr(
                expr.left, self_value, use_local_consts=use_local_consts)
            right = self._eval_raw_const_scalar_expr(
                expr.right, self_value, use_local_consts=use_local_consts)
            if not (isinstance(left, (int, float))
                    and isinstance(right, (int, float))
                    and not isinstance(left, bool)
                    and not isinstance(right, bool)):
                return None
            try:
                if expr.op == "+":
                    return left + right
                if expr.op == "-":
                    return left - right
                if expr.op == "*":
                    return left * right
                if expr.op == "/" and right != 0:
                    return left / right
                if expr.op == "%" and right != 0:
                    return left % right
            except (OverflowError, ValueError):
                return None
        return None

    def _cast_const_scalar_to_type(
        self, value: int | float | bool | None, target: Type,
    ) -> int | float | bool | None:
        if value is None:
            return None
        if not isinstance(target, TyPrim):
            return value
        if target.name in _INT_PRIM_NAMES:
            if not (isinstance(value, (int, float))
                    and not isinstance(value, bool)):
                return None
            if isinstance(value, float) and not math.isfinite(value):
                return None
            try:
                converted = int(value)
            except (OverflowError, ValueError):
                return None
            bounds = self._INT_BOUNDS.get(target.name)
            if bounds is not None:
                lo, hi = bounds
                if converted < lo or converted > hi:
                    return None
            return converted
        if target.name in ("bf16", "f16"):
            return None
        if target.name == "f32":
            if not (isinstance(value, (int, float))
                    and not isinstance(value, bool)):
                return None
            return self._round_const_scalar_to_f32(float(value))
        if target.name == "f64":
            if not (isinstance(value, (int, float))
                    and not isinstance(value, bool)):
                return None
            try:
                converted = float(value)
            except OverflowError:
                return None
            if not math.isfinite(converted):
                return None
            return converted
        if target.name == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
        return value

    def _round_const_scalar_to_f32(self, value: float) -> float | None:
        if not math.isfinite(value):
            return None
        try:
            rounded = struct.unpack("<f", struct.pack("<f", value))[0]
        except OverflowError:
            return None
        except (ValueError, struct.error):
            return None
        if not math.isfinite(rounded):
            return None
        return rounded

    def _check_self_independent_refinement(
        self, refined: "TyRefined", span: A.Span, context: str,
    ) -> list[A.Expr]:
        """Check predicates that do not depend on `self`.

        These predicates can be proved or failed even when the assigned value is
        not a compile-time scalar. Return predicates that still need the value.
        """
        pending: list[A.Expr] = []
        for pred in refined.predicates:
            ok = self._eval_refinement_predicate(
                pred, None, numeric_base=self._erase_refinement(refined))
            if ok is None and self._expr_mentions_self(pred):
                pending.append(pred)
                continue
            if ok is None:
                self._record_refinement_obligation(
                    context, refined, pred, "unsupported", span, None,
                )
                self.errors.append(TypeError_(
                    f"{context}: refinement {refined.name} predicate "
                    f"{self._fmt_refinement_expr(pred)} is not supported by "
                    f"the Stage 31 constant checker",
                    span,
                ))
                continue
            if not ok:
                self._record_refinement_obligation(
                    context, refined, pred, "failed", span, None,
                    trap="31001",
                )
                self.errors.append(TypeError_(
                    f"{context}: refinement {refined.name} violated: "
                    f"predicate {self._fmt_refinement_expr(pred)} is always "
                    f"false (trap 31001)",
                    span,
                    hint="refined values must satisfy their `where` predicate",
                ))
            else:
                self._record_refinement_obligation(
                    context, refined, pred, "proved", span, None,
                )
        return pending

    def _record_unproven_refinement_obligations(
        self, context: str, refined: "TyRefined", span: A.Span,
    ) -> None:
        for pred in refined.predicates:
            self._record_refinement_obligation(
                context, refined, pred, "unproven", span, None,
            )
        if isinstance(refined.base, TyRefined):
            self._record_unproven_refinement_obligations(
                context, refined.base, span)

    def _record_refinement_obligation(
        self, context: str, refined: "TyRefined", predicate: A.Expr,
        status: str, span: A.Span, value: int | float | bool | None,
        *, trap: str | None = None,
    ) -> None:
        self.proof_obligations.append(ProofObligation(
            kind="refinement",
            context=context,
            refinement=refined.name,
            predicate=self._fmt_refinement_expr(predicate),
            status=status,
            line=span.line,
            col=span.col,
            value=(None if value is None else self._fmt_scalar_value(value)),
            trap=trap,
        ))

    def _record_refinement_proof_carry(
        self, context: str, value_ty: "TyRefined", target: "TyRefined",
        strategy: str, span: A.Span,
    ) -> None:
        self.proof_carries.append(ProofCarry(
            kind="refinement-proof-carry",
            context=context,
            source_refinement=value_ty.name,
            target_refinement=target.name,
            strategy=strategy,
            line=span.line,
            col=span.col,
        ))

    def _record_refinement_proof_carries_for_type(
        self, context: str, value_ty: Type, target_ty: Type, span: A.Span,
    ) -> None:
        if isinstance(target_ty, TyRefined) and isinstance(value_ty, TyRefined):
            strategy = self._refinement_proof_carry_strategy(
                value_ty, target_ty)
            if strategy is not None:
                self._record_refinement_proof_carry(
                    context, value_ty, target_ty, strategy, span)
            return
        if isinstance(target_ty, TyArray) and isinstance(value_ty, TyArray):
            self._record_refinement_proof_carries_for_type(
                f"{context}: array element",
                value_ty.elem,
                target_ty.elem,
                span,
            )
            return
        if isinstance(target_ty, TyTuple) and isinstance(value_ty, TyTuple):
            for idx, (value_elem, target_elem) in enumerate(
                    zip(value_ty.elems, target_ty.elems)):
                self._record_refinement_proof_carries_for_type(
                    f"{context}: tuple element {idx}",
                    value_elem,
                    target_elem,
                    span,
                )
            return

    def _reject_non_i32_result_payload(
        self, ty: Type, span, *, side: str,
    ) -> None:
        """Stage 49 closure gate-2 type-design G2-H1 fix: the
        Inc 1 packed-i64 Result representation uses a 32-bit
        payload slot. Non-i32 payloads (i64, f32, f64, struct,
        nested Result, etc.) silently truncate at IR lowering.
        Until Stage 50+ widens the representation, reject at
        typecheck with a clear diagnostic naming the side and
        the offending type.

        Stage 49 closure gate-4 silent-failure SF4-C1+C2 fix
        (commit b4c8434): the prior version of this docstring
        claimed TyResult was permitted "via the existing
        identity-lowering." That was WRONG — `_lower_type(Result)`
        returns TIRScalar('i64'), but RESULT_PACK's payload
        operand is still a 32-bit `mov ecx`, so nested Result
        silently dropped the inner Result's tag bit at position
        32. The TyResult whitelist was deleted; nested Result
        now correctly falls through to the rejection path.

        Permits: TyPrim('i32') and TyUnknown only (TyUnknown
        comes from the sibling side's inferred-constructor
        provenance, kept by the gate-1 G2-F1 mechanism).
        """
        # Permissive: TyUnknown comes from the sibling side's
        # constructor (Err inferred / Ok inferred) — let it pass.
        if isinstance(ty, TyUnknown):
            return
        # The Stage 37-41 wrapper-quintet types currently lower
        # as identity in expression position (the constructor
        # call e.g. into_known() routes through the identity
        # arm in lower_ast). The inner type is what matters
        # for the packed payload — recurse one level.
        from . import typecheck as _self_mod  # avoid re-import noise
        # Strip wrapper layers; Phase-0 they're identity-lowered.
        stripped = ty
        for _ in range(8):  # guard against pathological nesting
            if isinstance(stripped, TyModal):
                stripped = stripped.inner
            elif isinstance(stripped, TyTemporal):
                stripped = stripped.inner
            elif isinstance(stripped, TyCausal):
                stripped = stripped.inner
            elif isinstance(stripped, TyFrame):
                stripped = stripped.inner
            elif isinstance(stripped, TyMemTier):
                stripped = stripped.inner
            elif isinstance(stripped, TyLogic):
                stripped = stripped.inner
            else:
                break
        if isinstance(stripped, TyUnknown):
            return
        # Accept i32 (the supported payload width).
        if isinstance(stripped, TyPrim) and stripped.name == "i32":
            return
        # Stage 49 closure gate-4 fix (CRITICAL SF4-C1/SF4-C2 +
        # TD1-C2/TD4-C3 — cross-lane convergence on one root):
        # the gate-2 helper originally whitelisted TyResult here
        # under the reasoning "identity-recurses to i64 packed".
        # That reasoning was wrong: `_lower_type(Result<T,E>)`
        # returns `TIRScalar("i64")`, but the RESULT_PACK
        # opcode's payload operand is a 32-bit read at the
        # backend (helixc/backend/x86_64.py:2200). So
        # `Ok(Err(99))` would silently truncate the inner
        # Result's high-32 bits — destroying its tag — and
        # `map_ok(r, make_inner())` where make_inner returns
        # Result inherited the same bug via G3-H1's wider-
        # payload reject helper. Gate-4 silent-failure + type-
        # design lanes independently reproduced exit 200 vs
        # expected 100 on `Ok(Err(99))` and exit 88 vs 77 on
        # `map_ok(Ok(1), Err(99))`. Removing the whitelist
        # closes both miscompiles in one stroke and re-routes
        # nested Result through the canonical diagnostic.
        # Reject everything else (including nested TyResult)
        # with a clear diagnostic.
        self.errors.append(TypeError_(
            f"{side}() payload type {self._fmt(ty)} is not "
            f"supported by the Stage 49 packed-i64 Result "
            f"representation; only i32 payloads work today",
            span,
            hint="Stage 50+ widens the payload representation; "
            "for now, use Result<i32, i32> or wrap a wider "
            "value in a small i32 handle. See "
            "docs/stage49-plan-2026-05-17.md:164-171.",
        ))

    def _contains_unknown_type(self, ty: Type) -> bool:
        if isinstance(ty, TyUnknown):
            return True
        if isinstance(ty, TyRefined):
            return self._contains_unknown_type(ty.base)
        if isinstance(ty, TyArray):
            return (self._contains_unknown_type(ty.elem)
                    or self._contains_unknown_type(ty.size))
        if isinstance(ty, TyTuple):
            return any(self._contains_unknown_type(e) for e in ty.elems)
        if isinstance(ty, TyRef):
            return self._contains_unknown_type(ty.inner)
        if isinstance(ty, TyPtr):
            return self._contains_unknown_type(ty.inner)
        if isinstance(ty, TyFn):
            return (any(self._contains_unknown_type(p) for p in ty.params)
                    or self._contains_unknown_type(ty.ret))
        if isinstance(ty, TyTensor):
            return (self._contains_unknown_type(ty.dtype)
                    or any(self._contains_unknown_type(s)
                           for s in ty.shape))
        if isinstance(ty, TyTile):
            return (self._contains_unknown_type(ty.dtype)
                    or any(self._contains_unknown_type(s)
                           for s in ty.shape))
        # Stage 39 closure gate-1 silent-failure F2 fix + gate-2 F6
        # extension + Stage 40 Inc 1 preemptive TyModal: ALL single-
        # inner wrapper types must walk to their inner so a TyUnknown
        # buried under any wrapper still short-circuits downstream
        # struct monomorphization. Pre-F2 the wrapped case silently
        # returned False — a Stage-37/38 hole Stage 39 would have
        # inherited and widened. Gate 2 F6: the F2 sweep stopped short
        # of TyDiff / TyLogic / TyQuote; folded in. Stage 40 Inc 1:
        # TyModal added preemptively to close the H1/F2/F6 lesson
        # before audit time.
        if isinstance(ty, (
                TyMemTier, TyFrame, TyTemporal, TyModal, TyCausal,
                TyDiff, TyLogic, TyQuote,
        )):
            return self._contains_unknown_type(ty.inner)
        # Stage 46 Inc 1 — TyResult walks BOTH inners. First
        # two-parameter wrapper family. The Ok and Err sides each
        # carry an independent type that could buried-contain a
        # TyUnknown.
        if isinstance(ty, TyResult):
            return (self._contains_unknown_type(ty.ok_ty)
                    or self._contains_unknown_type(ty.err_ty))
        return False

    def _refinement_proof_carried(
        self, value_ty: Type, target: Type,
    ) -> bool:
        """Whether `value_ty` already carries the target refinement proof.

        Stage 31's first checker can prove literals only, but a variable that
        already has the same refined type should carry its proof forward
        through lets, calls, and returns.
        """
        if isinstance(target, TyArray) and isinstance(value_ty, TyArray):
            return self._refinement_proof_carried(value_ty.elem, target.elem)
        if isinstance(target, TyTuple) and isinstance(value_ty, TyTuple):
            return (len(value_ty.elems) == len(target.elems)
                    and all(self._refinement_proof_carried(v, t)
                            for v, t in zip(value_ty.elems, target.elems)))
        if isinstance(target, TyFn) and isinstance(value_ty, TyFn):
            return self._function_refinement_shape_exact(value_ty, target)
        if isinstance(target, TyRef) and isinstance(value_ty, TyRef):
            if target.is_mut != value_ty.is_mut:
                return False
            if (self._contains_refinement(value_ty.inner)
                    or self._contains_refinement(target.inner)):
                return self._refinement_shape_exact(
                    value_ty.inner, target.inner)
            return True
        if isinstance(target, TyPtr) and isinstance(value_ty, TyPtr):
            if target.is_mut != value_ty.is_mut:
                return False
            if (self._contains_refinement(value_ty.inner)
                    or self._contains_refinement(target.inner)):
                return self._refinement_shape_exact(
                    value_ty.inner, target.inner)
            return True
        if isinstance(target, TyDiff) and isinstance(value_ty, TyDiff):
            return self._refinement_shape_exact(value_ty.inner, target.inner)
        if isinstance(target, TyLogic) and isinstance(value_ty, TyLogic):
            return (target.provenance == value_ty.provenance
                    and self._refinement_shape_exact(
                        value_ty.inner, target.inner))
        if isinstance(target, TyQuote) and isinstance(value_ty, TyQuote):
            return self._refinement_shape_exact(value_ty.inner, target.inner)
        if isinstance(target, TyMemTier) and isinstance(value_ty, TyMemTier):
            return (target.tier == value_ty.tier
                    and self._refinement_shape_exact(
                        value_ty.inner, target.inner))
        # Stage 38 post-Inc-3 type-design H2 fix (HIGH, conf 88):
        # TyFrame must walk to its inner type so refinements under a
        # frame wrapper are visible to the refinement-shape check.
        if isinstance(target, TyFrame) and isinstance(value_ty, TyFrame):
            return (target.frame == value_ty.frame
                    and self._refinement_shape_exact(
                        value_ty.inner, target.inner))
        # Stage 39 closure gate-1 type-design H2 fix: TyTemporal needs
        # parallel walk so refinements under a temporal wrapper are
        # visible to the refinement-shape check. Mirrors TyFrame.
        if isinstance(target, TyTemporal) and isinstance(value_ty, TyTemporal):
            return (target.kind == value_ty.kind
                    and self._refinement_shape_exact(
                        value_ty.inner, target.inner))
        # Stage 40 Inc 1: TyModal preemptive parallel arm
        # (also covers _refinement_proof_carried arm — both call
        # sites share the same recursive pattern).
        if isinstance(target, TyModal) and isinstance(value_ty, TyModal):
            return (target.kind == value_ty.kind
                    and self._refinement_shape_exact(
                        value_ty.inner, target.inner))
        # Stage 41 Inc 1: TyCausal preemptive parallel arm.
        if isinstance(target, TyCausal) and isinstance(value_ty, TyCausal):
            return (target.kind == value_ty.kind
                    and self._refinement_shape_exact(
                        value_ty.inner, target.inner))
        # Stage 46 Inc 1: TyResult two-inner parallel arm.
        if isinstance(target, TyResult) and isinstance(value_ty, TyResult):
            return (self._refinement_shape_exact(
                        value_ty.ok_ty, target.ok_ty)
                    and self._refinement_shape_exact(
                        value_ty.err_ty, target.err_ty))
        if isinstance(target, TyTensor) and isinstance(value_ty, TyTensor):
            return (len(target.shape) == len(value_ty.shape)
                    and self._refinement_shape_exact(
                        value_ty.dtype, target.dtype)
                    and all(self._size_compatible(vs, ts)
                            for vs, ts in zip(value_ty.shape, target.shape))
                    and value_ty.device == target.device
                    and value_ty.layout == target.layout)
        if isinstance(target, TyTile) and isinstance(value_ty, TyTile):
            return (len(target.shape) == len(value_ty.shape)
                    and self._refinement_shape_exact(
                        value_ty.dtype, target.dtype)
                    and all(self._size_compatible(vs, ts)
                            for vs, ts in zip(value_ty.shape, target.shape))
                    and value_ty.memspace == target.memspace)
        if not isinstance(target, TyRefined):
            return not self._contains_refinement(target)
        if not isinstance(value_ty, TyRefined):
            return False
        if self._refinement_proof_carry_strategy(value_ty, target) is not None:
            return True
        return self._refinement_proof_carried(value_ty.base, target)

    def _refinement_proof_carry_strategy(
        self, value_ty: "TyRefined", target: "TyRefined",
    ) -> str | None:
        if (value_ty.name == target.name
                and self._compatible(value_ty.base, target.base)):
            return "same-refinement"
        if self._refinement_predicates_exact_cover(value_ty, target):
            return "exact-predicate-subset"
        if self._refinement_numeric_bounds_cover(value_ty, target):
            return "numeric-bound-implication"
        if isinstance(value_ty.base, TyRefined):
            return self._refinement_proof_carry_strategy(value_ty.base, target)
        return None

    def _refinement_predicates_cover(
        self, value_ty: Type, target: "TyRefined",
    ) -> bool:
        """Whether value_ty proves every predicate required by target.

        Stage 31 reused already-carried proofs for alias-equivalent
        refinements and exact predicate subsets. Stage 34 adds a small,
        fail-closed implication step for simple numeric bounds such as
        `self >= 1.0` proving `self >= 0.0`.
        """
        if not isinstance(value_ty, TyRefined):
            return False
        if self._erase_refinement(value_ty) != self._erase_refinement(target):
            return False
        if self._refinement_predicates_exact_cover(value_ty, target):
            return True
        return self._refinement_numeric_bounds_cover(value_ty, target)

    def _refinement_predicates_exact_cover(
        self, value_ty: Type, target: "TyRefined",
    ) -> bool:
        if not isinstance(value_ty, TyRefined):
            return False
        if self._erase_refinement(value_ty) != self._erase_refinement(target):
            return False
        value_preds = self._refinement_predicate_keys(value_ty)
        target_preds = self._refinement_predicate_keys(target)
        if value_preds is None or target_preds is None:
            return False
        return bool(target_preds) and target_preds.issubset(value_preds)

    def _refinement_numeric_bounds_cover(
        self, value_ty: "TyRefined", target: "TyRefined",
    ) -> bool:
        if self._erase_refinement(value_ty) != self._erase_refinement(target):
            return False
        value_bounds = self._refinement_numeric_bounds(
            value_ty, self._erase_refinement(value_ty))
        target_reqs = self._refinement_numeric_requirements(
            target, self._erase_refinement(target))
        if value_bounds is None or target_reqs is None:
            return False
        return all(
            self._numeric_bounds_imply(value_bounds, req)
            for req in target_reqs
        )

    def _refinement_numeric_bounds(
        self, ty: Type, numeric_base: Type | None = None,
    ) -> Optional[dict[str, tuple[int | float, bool]]]:
        lower: tuple[int | float, bool] | None = None
        upper: tuple[int | float, bool] | None = None
        for pred in self._refinement_predicate_exprs(ty):
            bounds = self._refinement_predicate_bounds(pred, numeric_base)
            if bounds is None:
                return None
            for kind, value, inclusive in bounds:
                if kind == "lower":
                    if (lower is None
                            or value > lower[0]
                            or (value == lower[0]
                                and lower[1] and not inclusive)):
                        lower = (value, inclusive)
                elif kind == "upper":
                    if (upper is None
                            or value < upper[0]
                            or (value == upper[0]
                                and upper[1] and not inclusive)):
                        upper = (value, inclusive)
                else:
                    return None
        out: dict[str, tuple[int | float, bool]] = {}
        if lower is not None:
            out["lower"] = lower
        if upper is not None:
            out["upper"] = upper
        return out

    def _refinement_numeric_requirements(
        self, ty: Type, numeric_base: Type | None = None,
    ) -> Optional[list[tuple[str, int | float, bool]]]:
        out: list[tuple[str, int | float, bool]] = []
        for pred in self._refinement_predicate_exprs(ty):
            bounds = self._refinement_predicate_bounds(pred, numeric_base)
            if bounds is None:
                return None
            out.extend(bounds)
        return out

    def _refinement_predicate_exprs(self, ty: Type) -> list[A.Expr]:
        if not isinstance(ty, TyRefined):
            return []
        out = list(ty.predicates)
        out.extend(self._refinement_predicate_exprs(ty.base))
        return out

    def _refinement_predicate_bounds(
        self, expr: A.Expr, numeric_base: Type | None = None,
    ) -> Optional[list[tuple[str, int | float, bool]]]:
        if isinstance(expr, A.BoolLit):
            return [] if expr.value else None
        if isinstance(expr, A.Unary) and expr.op == "!":
            return self._negated_refinement_predicate_bounds(
                expr.operand, numeric_base)
        if isinstance(expr, A.Binary) and expr.op == "&&":
            left = self._refinement_predicate_bounds(
                expr.left, numeric_base)
            right = self._refinement_predicate_bounds(
                expr.right, numeric_base)
            if left is None or right is None:
                return None
            return left + right
        if isinstance(expr, A.Binary) and expr.op == "||":
            return None
        chain = self._flatten_relational_chain(expr)
        if chain is None:
            return None
        ops, operands = chain
        out: list[tuple[str, int | float, bool]] = []
        for left, op, right in zip(operands, ops, operands[1:]):
            bounds = self._refinement_binary_bounds(
                left, op, right, numeric_base)
            if bounds is None:
                ok = self._eval_refinement_predicate(
                    A.Binary(left=left, op=op, right=right, span=expr.span),
                    None,
                    numeric_base=numeric_base,
                )
                if ok is True:
                    continue
                return None
            out.extend(bounds)
        return out

    def _negated_refinement_predicate_bounds(
        self, expr: A.Expr, numeric_base: Type | None = None,
    ) -> Optional[list[tuple[str, int | float, bool]]]:
        if isinstance(expr, A.BoolLit):
            return [] if not expr.value else None
        if isinstance(expr, A.Unary) and expr.op == "!":
            return self._refinement_predicate_bounds(
                expr.operand, numeric_base)
        if isinstance(expr, A.Binary) and expr.op in ("&&", "||"):
            return None
        chain = self._flatten_relational_chain(expr)
        if chain is None:
            return None
        ops, operands = chain
        if len(ops) != 1:
            return None
        negated_op = self._negate_comparison_op(ops[0])
        if negated_op is None:
            return None
        return self._refinement_binary_bounds(
            operands[0], negated_op, operands[1], numeric_base)

    def _negate_comparison_op(self, op: str) -> str | None:
        return {
            "<": ">=",
            "<=": ">",
            ">": "<=",
            ">=": "<",
            "==": "!=",
            "!=": "==",
        }.get(op)

    def _refinement_binary_bounds(
        self, left: A.Expr, op: str, right: A.Expr,
        numeric_base: Type | None = None,
    ) -> Optional[list[tuple[str, int | float, bool]]]:
        affine = None
        if not self._numeric_base_is_fixed_width_number(numeric_base):
            affine = self._refinement_affine_binary_bounds(
                left, op, right, numeric_base)
        if affine is not None:
            return affine
        left_is_self = self._expr_is_plain_self(left)
        right_is_self = self._expr_is_plain_self(right)
        if left_is_self == right_is_self:
            return None
        if left_is_self:
            value = self._eval_const_scalar_expr(
                right, None, honor_float_suffix=True,
                numeric_base=numeric_base)
            return self._bound_from_self_compare(op, value)
        value = self._eval_const_scalar_expr(
            left, None, honor_float_suffix=True,
            numeric_base=numeric_base)
        return self._bound_from_const_compare(op, value)

    def _refinement_affine_binary_bounds(
        self, left: A.Expr, op: str, right: A.Expr,
        numeric_base: Type | None = None,
    ) -> Optional[list[tuple[str, int | float, bool]]]:
        left_affine = self._refinement_affine_expr(left, numeric_base)
        right_affine = self._refinement_affine_expr(right, numeric_base)
        if left_affine is None or right_affine is None:
            return None
        left_coeff, left_const = left_affine
        right_coeff, right_const = right_affine
        coeff = left_coeff - right_coeff
        const = left_const - right_const
        if coeff == 0:
            return None
        bound_value = -const / coeff
        bound_op = op if coeff > 0 else self._flip_comparison_op(op)
        if bound_op is None:
            return None
        return self._bound_from_self_compare(bound_op, bound_value)

    def _refinement_affine_expr(
        self, expr: A.Expr, numeric_base: Type | None = None,
    ) -> Optional[tuple[int | float, int | float]]:
        if self._expr_is_plain_self(expr):
            return (1, 0)
        if not self._expr_mentions_self(expr):
            value = self._eval_const_scalar_expr(
                expr, None, honor_float_suffix=True,
                numeric_base=numeric_base)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return (0, value)
            return None
        if isinstance(expr, A.Unary) and expr.op == "-":
            inner = self._refinement_affine_expr(expr.operand, numeric_base)
            if inner is None:
                return None
            coeff, const = inner
            return (-coeff, -const)
        if isinstance(expr, A.Binary) and expr.op in ("+", "-"):
            left = self._refinement_affine_expr(expr.left, numeric_base)
            right = self._refinement_affine_expr(expr.right, numeric_base)
            if left is None or right is None:
                return None
            left_coeff, left_const = left
            right_coeff, right_const = right
            if expr.op == "+":
                return (left_coeff + right_coeff, left_const + right_const)
            return (left_coeff - right_coeff, left_const - right_const)
        if isinstance(expr, A.Binary) and expr.op == "*":
            left = self._refinement_affine_expr(expr.left, numeric_base)
            right = self._refinement_affine_expr(expr.right, numeric_base)
            if left is None or right is None:
                return None
            if left[0] == 0:
                return (right[0] * left[1], right[1] * left[1])
            if right[0] == 0:
                return (left[0] * right[1], left[1] * right[1])
            return None
        if isinstance(expr, A.Binary) and expr.op == "/":
            if self._numeric_base_is_int(numeric_base):
                return None
            left = self._refinement_affine_expr(expr.left, numeric_base)
            right = self._refinement_affine_expr(expr.right, numeric_base)
            if left is None or right is None or right[0] != 0 or right[1] == 0:
                return None
            return (left[0] / right[1], left[1] / right[1])
        return None

    def _flip_comparison_op(self, op: str) -> str | None:
        return {
            "<": ">",
            "<=": ">=",
            ">": "<",
            ">=": "<=",
            "==": "==",
            "!=": "!=",
        }.get(op)

    def _bound_from_self_compare(
        self, op: str, value: int | float | bool | None,
    ) -> Optional[list[tuple[str, int | float, bool]]]:
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
            return None
        if op == ">":
            return [("lower", value, False)]
        if op == ">=":
            return [("lower", value, True)]
        if op == "<":
            return [("upper", value, False)]
        if op == "<=":
            return [("upper", value, True)]
        if op == "==":
            return [("lower", value, True), ("upper", value, True)]
        if op == "!=":
            return None
        return None

    def _bound_from_const_compare(
        self, op: str, value: int | float | bool | None,
    ) -> Optional[list[tuple[str, int | float, bool]]]:
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
            return None
        if op == "<":
            return [("lower", value, False)]
        if op == "<=":
            return [("lower", value, True)]
        if op == ">":
            return [("upper", value, False)]
        if op == ">=":
            return [("upper", value, True)]
        if op == "==":
            return [("lower", value, True), ("upper", value, True)]
        if op == "!=":
            return None
        return None

    def _numeric_bounds_imply(
        self,
        value_bounds: dict[str, tuple[int | float, bool]],
        req: tuple[str, int | float, bool],
    ) -> bool:
        kind, req_value, req_inclusive = req
        value_bound = value_bounds.get(kind)
        if value_bound is None:
            return False
        value, inclusive = value_bound
        if kind == "lower":
            if value > req_value:
                return True
            if value < req_value:
                return False
        elif kind == "upper":
            if value < req_value:
                return True
            if value > req_value:
                return False
        else:
            return False
        if req_inclusive:
            return True
        return not inclusive

    def _expr_is_plain_self(self, expr: A.Expr) -> bool:
        return (isinstance(expr, A.Name)
                and expr.name == "self"
                and not expr.generics)

    def _refinement_predicate_keys(
        self, ty: Type,
    ) -> Optional[set[tuple[object, ...]]]:
        if not isinstance(ty, TyRefined):
            return set()
        out: set[tuple[object, ...]] = set()
        for pred in ty.predicates:
            key = self._refinement_predicate_key(pred)
            if key is None:
                return None
            out.add(key)
        base_keys = self._refinement_predicate_keys(ty.base)
        if base_keys is None:
            return None
        out.update(base_keys)
        return out

    def _refinement_predicate_key(
        self, expr: A.Expr,
    ) -> Optional[tuple[object, ...]]:
        if isinstance(expr, A.BoolLit):
            return ("bool", expr.value)
        if isinstance(expr, A.Unary) and expr.op == "!":
            operand = self._refinement_predicate_key(expr.operand)
            if operand is None:
                return None
            return ("not", operand)
        if isinstance(expr, A.Binary) and expr.op in ("&&", "||"):
            left = self._refinement_predicate_key(expr.left)
            right = self._refinement_predicate_key(expr.right)
            if left is None or right is None:
                return None
            return ("logical", expr.op, left, right)
        chain = self._flatten_relational_chain(expr)
        if chain is None:
            return None
        ops, operands = chain
        operand_keys = tuple(
            self._refinement_scalar_expr_key(operand)
            for operand in operands
        )
        if any(key is None for key in operand_keys):
            return None
        return ("rel", tuple(ops), operand_keys)

    def _refinement_scalar_expr_key(
        self, expr: A.Expr,
    ) -> Optional[tuple[object, ...]]:
        if isinstance(expr, A.IntLit):
            return ("int", expr.value, expr.type_suffix)
        if isinstance(expr, A.FloatLit):
            return ("float", expr.value, expr.type_suffix)
        if isinstance(expr, A.Name) and not expr.generics:
            return ("name", expr.name)
        if isinstance(expr, A.Unary) and expr.op == "-":
            operand = self._refinement_scalar_expr_key(expr.operand)
            if operand is None:
                return None
            return ("unary", expr.op, operand)
        if isinstance(expr, A.Binary) and expr.op in ("+", "-", "*", "/", "%"):
            left = self._refinement_scalar_expr_key(expr.left)
            right = self._refinement_scalar_expr_key(expr.right)
            if left is None or right is None:
                return None
            return ("arith", expr.op, left, right)
        return None

    def _check_refinement_contextual_value(
        self, value_expr: A.Expr, value_ty: Type, target_ty: Type,
        span: A.Span, context: str, scope: Scope,
    ) -> None:
        if isinstance(target_ty, TyRefined):
            if self._refinement_proof_carried(value_ty, target_ty):
                self._record_refinement_proof_carries_for_type(
                    context, value_ty, target_ty, span)
            else:
                self._check_refinement_const_value(
                    value_expr, value_ty, target_ty, span, context, scope,
                )
            return
        if isinstance(target_ty, TyArray) and isinstance(value_expr, A.ArrayLit):
            for elem_expr in value_expr.elems:
                elem_ty = self._check_expr(elem_expr, scope)
                self._check_refinement_contextual_value(
                    elem_expr, elem_ty, target_ty.elem, elem_expr.span,
                    f"{context}: array element",
                    scope,
                )
            return
        if isinstance(target_ty, TyArray):
            if (self._contains_refinement(target_ty)
                    or self._contains_refinement(value_ty)):
                if self._refinement_proof_carried(value_ty, target_ty):
                    self._record_refinement_proof_carries_for_type(
                        context, value_ty, target_ty, span)
                    return
                if self._contains_refinement(target_ty):
                    self.errors.append(TypeError_(
                        f"{context}: refined array type "
                        f"{self._fmt(target_ty)} requires an array literal "
                        f"or already-proven refined array in Stage 31",
                        span,
                        hint="use an array literal with proven refined "
                             "elements for now",
                    ))
                else:
                    self.errors.append(TypeError_(
                        f"{context}: array type conversion from "
                        f"{self._fmt(value_ty)} to {self._fmt(target_ty)} "
                        f"would change refined parameter or return "
                        f"requirements in Stage 31",
                        span,
                        hint="use an array type with exactly matching "
                             "refinements",
                    ))
            return
        if isinstance(target_ty, TyTuple) and isinstance(value_expr, A.TupleLit):
            for elem_expr, elem_target in zip(value_expr.elems, target_ty.elems):
                elem_ty = self._check_expr(elem_expr, scope)
                self._check_refinement_contextual_value(
                    elem_expr, elem_ty, elem_target, elem_expr.span,
                    f"{context}: tuple element",
                    scope,
                )
            return
        if isinstance(target_ty, TyTuple):
            if (self._contains_refinement(target_ty)
                    or self._contains_refinement(value_ty)):
                if self._refinement_proof_carried(value_ty, target_ty):
                    self._record_refinement_proof_carries_for_type(
                        context, value_ty, target_ty, span)
                    return
                if self._contains_refinement(target_ty):
                    self.errors.append(TypeError_(
                        f"{context}: refined tuple type "
                        f"{self._fmt(target_ty)} requires a tuple literal "
                        f"or already-proven refined tuple in Stage 31",
                        span,
                        hint="use a tuple literal with proven refined "
                             "elements for now",
                    ))
                else:
                    self.errors.append(TypeError_(
                        f"{context}: tuple type conversion from "
                        f"{self._fmt(value_ty)} to {self._fmt(target_ty)} "
                        f"would change refined parameter or return "
                        f"requirements in Stage 31",
                        span,
                        hint="use a tuple type with exactly matching "
                             "refinements",
                    ))
            return
        if isinstance(target_ty, TyFn):
            if (self._contains_refinement(target_ty)
                    or self._contains_refinement(value_ty)):
                if not self._function_refinement_shape_exact(
                        value_ty, target_ty):
                    self.errors.append(TypeError_(
                        f"{context}: function type conversion from "
                        f"{self._fmt(value_ty)} to {self._fmt(target_ty)} "
                        f"would change refined parameter or return "
                        f"requirements in Stage 31",
                        span,
                        hint="use a function type with exactly matching "
                             "refinements",
                    ))
            return
        if isinstance(target_ty, TyRef):
            if (self._contains_refinement(target_ty)
                    or self._contains_refinement(value_ty)):
                if not self._refinement_proof_carried(value_ty, target_ty):
                    self.errors.append(TypeError_(
                        f"{context}: reference type conversion from "
                        f"{self._fmt(value_ty)} to {self._fmt(target_ty)} "
                        f"would change refined parameter or return "
                        f"requirements in Stage 31",
                        span,
                        hint="use a reference type with exactly matching "
                             "refinements",
                    ))
            return
        if isinstance(target_ty, TyPtr):
            if (self._contains_refinement(target_ty)
                    or self._contains_refinement(value_ty)):
                if not self._refinement_proof_carried(value_ty, target_ty):
                    self.errors.append(TypeError_(
                        f"{context}: pointer type conversion from "
                        f"{self._fmt(value_ty)} to {self._fmt(target_ty)} "
                        f"would change refined parameter or return "
                        f"requirements in Stage 31",
                        span,
                        hint="use a pointer type with exactly matching "
                             "refinements",
                    ))
            return
        if (self._contains_refinement(value_ty)
                or self._contains_refinement(target_ty)):
            if self._is_refinement_container(value_ty) \
                    or self._is_refinement_container(target_ty):
                if not self._refinement_proof_carried(value_ty, target_ty):
                    self.errors.append(TypeError_(
                        f"{context}: type conversion from "
                        f"{self._fmt(value_ty)} to {self._fmt(target_ty)} "
                        f"would change refined parameter or return "
                        f"requirements in Stage 31",
                        span,
                        hint="use a type with exactly matching refinements",
                    ))
                return

    def _check_unrepresentable_scalar_context(
        self,
        value_expr: A.Expr,
        target_ty: Type,
        span: A.Span,
        context: str,
        *,
        report: bool = True,
    ) -> bool:
        if self._contains_refinement(target_ty):
            return False
        target_base = self._const_eval_numeric_base(target_ty)
        if target_base is None:
            target_base = self._expr_unrepresentable_typed_const_scalar_base(
                value_expr)
            if target_base is None:
                return False
        if not self._expr_has_unrepresentable_typed_const_scalar(value_expr):
            return False
        if not report:
            return True
        self.errors.append(TypeError_(
            f"{context}: value requires a representable target value in "
            f"Stage 34 for target base {self._fmt(target_base)}",
            span,
            hint="values must be representable by their erased target type "
                 "before they can be used as proof sources",
        ))
        return True

    def _function_refinement_shape_exact(
        self, value_ty: Type, target_ty: Type,
    ) -> bool:
        if not isinstance(value_ty, TyFn) or not isinstance(target_ty, TyFn):
            return False
        if len(value_ty.params) != len(target_ty.params):
            return False
        return (all(self._refinement_shape_exact(v, t)
                    for v, t in zip(value_ty.params, target_ty.params))
                and self._refinement_shape_exact(value_ty.ret, target_ty.ret))

    def _refinement_shape_exact(self, a: Type, b: Type) -> bool:
        if isinstance(a, TyRefined) or isinstance(b, TyRefined):
            return (isinstance(a, TyRefined)
                    and isinstance(b, TyRefined)
                    and a.name == b.name
                    and self._refinement_shape_exact(a.base, b.base))
        if isinstance(a, TyArray) and isinstance(b, TyArray):
            return self._refinement_shape_exact(a.elem, b.elem)
        if isinstance(a, TyTuple) and isinstance(b, TyTuple):
            return (len(a.elems) == len(b.elems)
                    and all(self._refinement_shape_exact(x, y)
                            for x, y in zip(a.elems, b.elems)))
        if isinstance(a, TyFn) and isinstance(b, TyFn):
            return self._function_refinement_shape_exact(a, b)
        if isinstance(a, TyRef) and isinstance(b, TyRef):
            return (a.is_mut == b.is_mut
                    and self._refinement_shape_exact(a.inner, b.inner))
        if isinstance(a, TyPtr) and isinstance(b, TyPtr):
            return (a.is_mut == b.is_mut
                    and self._refinement_shape_exact(a.inner, b.inner))
        if isinstance(a, TyDiff) and isinstance(b, TyDiff):
            return self._refinement_shape_exact(a.inner, b.inner)
        if isinstance(a, TyLogic) and isinstance(b, TyLogic):
            return (a.provenance == b.provenance
                    and self._refinement_shape_exact(a.inner, b.inner))
        if isinstance(a, TyQuote) and isinstance(b, TyQuote):
            return self._refinement_shape_exact(a.inner, b.inner)
        if isinstance(a, TyMemTier) and isinstance(b, TyMemTier):
            return (a.tier == b.tier
                    and self._refinement_shape_exact(a.inner, b.inner))
        if isinstance(a, TyFrame) and isinstance(b, TyFrame):
            return (a.frame == b.frame
                    and self._refinement_shape_exact(a.inner, b.inner))
        # Stage 39 closure gate-1 type-design H2 fix: parallel arm for
        # TyTemporal — refinements under temporal wrappers must be
        # shape-visible at branch boundaries / join sites.
        if isinstance(a, TyTemporal) and isinstance(b, TyTemporal):
            return (a.kind == b.kind
                    and self._refinement_shape_exact(a.inner, b.inner))
        # Stage 40 Inc 1: TyModal preemptive parallel arm.
        if isinstance(a, TyModal) and isinstance(b, TyModal):
            return (a.kind == b.kind
                    and self._refinement_shape_exact(a.inner, b.inner))
        # Stage 41 Inc 1: TyCausal preemptive parallel arm.
        if isinstance(a, TyCausal) and isinstance(b, TyCausal):
            return (a.kind == b.kind
                    and self._refinement_shape_exact(a.inner, b.inner))
        # Stage 46 Inc 1: TyResult two-inner parallel arm.
        if isinstance(a, TyResult) and isinstance(b, TyResult):
            return (self._refinement_shape_exact(a.ok_ty, b.ok_ty)
                    and self._refinement_shape_exact(
                        a.err_ty, b.err_ty))
        if isinstance(a, TyTensor) and isinstance(b, TyTensor):
            return (len(a.shape) == len(b.shape)
                    and self._refinement_shape_exact(a.dtype, b.dtype)
                    and all(self._size_compatible(x, y)
                            for x, y in zip(a.shape, b.shape))
                    and a.device == b.device
                    and a.layout == b.layout)
        if isinstance(a, TyTile) and isinstance(b, TyTile):
            return (len(a.shape) == len(b.shape)
                    and self._refinement_shape_exact(a.dtype, b.dtype)
                    and all(self._size_compatible(x, y)
                            for x, y in zip(a.shape, b.shape))
                    and a.memspace == b.memspace)
        return (not self._contains_refinement(a)
                and not self._contains_refinement(b))

    def _refinements_equivalent(self, a: Type, b: Type) -> bool:
        return (isinstance(a, TyRefined)
                and isinstance(b, TyRefined)
                and a.name == b.name
                and self._compatible(a.base, b.base)
                and self._compatible(b.base, a.base))

    def _erase_refinement(self, ty: Type) -> Type:
        if isinstance(ty, TyRefined):
            return self._erase_refinement(ty.base)
        if isinstance(ty, TyArray):
            return TyArray(self._erase_refinement(ty.elem), ty.size)
        if isinstance(ty, TyTuple):
            return TyTuple(tuple(self._erase_refinement(e) for e in ty.elems))
        if isinstance(ty, TyRef):
            return TyRef(self._erase_refinement(ty.inner), ty.is_mut)
        if isinstance(ty, TyPtr):
            return TyPtr(self._erase_refinement(ty.inner), ty.is_mut)
        if isinstance(ty, TyDiff):
            return TyDiff(self._erase_refinement(ty.inner))
        if isinstance(ty, TyLogic):
            return TyLogic(self._erase_refinement(ty.inner), ty.provenance)
        if isinstance(ty, TyQuote):
            return TyQuote(self._erase_refinement(ty.inner))
        if isinstance(ty, TyMemTier):
            return TyMemTier(ty.tier, self._erase_refinement(ty.inner))
        if isinstance(ty, TyFrame):
            return TyFrame(ty.frame, self._erase_refinement(ty.inner))
        # Stage 39 closure gate-1 type-design H3: parallel TyTemporal
        # arm so erase walks into the inner type (otherwise refined
        # inners survive erasure and produce inconsistent diagnostics).
        if isinstance(ty, TyTemporal):
            return TyTemporal(ty.kind, self._erase_refinement(ty.inner))
        # Stage 40 Inc 1: TyModal preemptive parallel arm.
        if isinstance(ty, TyModal):
            return TyModal(ty.kind, self._erase_refinement(ty.inner))
        # Stage 41 Inc 1: TyCausal preemptive parallel arm.
        if isinstance(ty, TyCausal):
            return TyCausal(ty.kind, self._erase_refinement(ty.inner))
        # Stage 46 Inc 1: TyResult two-inner parallel arm.
        if isinstance(ty, TyResult):
            return TyResult(
                self._erase_refinement(ty.ok_ty),
                self._erase_refinement(ty.err_ty),
            )
        if isinstance(ty, TyTensor):
            return TyTensor(
                self._erase_refinement(ty.dtype), ty.shape, ty.device,
                ty.layout)
        if isinstance(ty, TyTile):
            return TyTile(
                self._erase_refinement(ty.dtype), ty.shape, ty.memspace)
        if isinstance(ty, TyFn):
            return TyFn(
                tuple(self._erase_refinement(p) for p in ty.params),
                self._erase_refinement(ty.ret),
            )
        return ty

    def _join_branch_types(
        self, tys: list[Type], span: Optional[A.Span] = None,
    ) -> Type:
        if not tys:
            return TyUnit()
        joined = tys[0]
        for t in tys[1:]:
            if ((self._contains_refined_function(joined)
                 or self._contains_refined_function(t))
                    and not self._refinement_shape_exact(joined, t)):
                self.errors.append(TypeError_(
                    f"branch function types {self._fmt(joined)} and "
                    f"{self._fmt(t)} differ in refined parameter or "
                    f"return requirements in Stage 31",
                    span or A.Span(0, 0),
                    hint="make each branch return a function with exactly "
                    "matching refinements",
                ))
                joined = self._erase_refinement(joined)
                continue
            if ((self._is_refinement_container(joined)
                 or self._is_refinement_container(t))
                    and (self._contains_refinement(joined)
                         or self._contains_refinement(t))
                    and not (self._refinement_proof_carried(joined, t)
                             and self._refinement_proof_carried(t, joined))):
                self.errors.append(TypeError_(
                    f"branch types {self._fmt(joined)} and "
                    f"{self._fmt(t)} differ in refined parameter or "
                    f"return requirements in Stage 31",
                    span or A.Span(0, 0),
                    hint="make each branch return a type with exactly "
                    "matching refinements",
                ))
                joined = self._erase_refinement(joined)
                continue
            if (self._contains_refinement(joined)
                    or self._contains_refinement(t)):
                if (self._refinement_proof_carried(joined, t)
                        and self._refinement_proof_carried(t, joined)):
                    continue
                joined = self._erase_refinement(joined)
        return joined

    def _contains_refinement(
        self, ty: Type, _seen_structs: Optional[set[str]] = None,
    ) -> bool:
        if _seen_structs is None:
            _seen_structs = set()
        if isinstance(ty, TyRefined):
            return True
        if isinstance(ty, TyArray):
            return self._contains_refinement(ty.elem, _seen_structs)
        if isinstance(ty, TyTuple):
            return any(self._contains_refinement(e, _seen_structs)
                       for e in ty.elems)
        if isinstance(ty, TyStruct):
            if ty.name in _seen_structs:
                return False
            decl = getattr(self, "_struct_decls", {}).get(ty.name)
            if decl is None:
                return False
            next_seen = set(_seen_structs)
            next_seen.add(ty.name)
            return any(
                self._contains_refinement(
                    self._resolve_type(field.ty, Scope()), next_seen)
                for field in decl.fields
            )
        if isinstance(ty, TyEnum):
            key = f"enum:{ty.name}"
            if key in _seen_structs:
                return False
            decl = getattr(self, "_enum_decls", {}).get(ty.name)
            if decl is None:
                return False
            next_seen = set(_seen_structs)
            next_seen.add(key)
            return any(
                self._contains_refinement(
                    self._resolve_type(payload_ty, Scope()), next_seen)
                for variant in decl.variants
                for payload_ty in variant.payload_tys
            )
        if isinstance(ty, TyRef):
            return self._contains_refinement(ty.inner, _seen_structs)
        if isinstance(ty, TyPtr):
            return self._contains_refinement(ty.inner, _seen_structs)
        if isinstance(ty, TyDiff):
            return self._contains_refinement(ty.inner, _seen_structs)
        if isinstance(ty, TyLogic):
            return self._contains_refinement(ty.inner, _seen_structs)
        if isinstance(ty, TyQuote):
            return self._contains_refinement(ty.inner, _seen_structs)
        if isinstance(ty, TyMemTier):
            return self._contains_refinement(ty.inner, _seen_structs)
        if isinstance(ty, TyFrame):
            return self._contains_refinement(ty.inner, _seen_structs)
        # Stage 39 closure gate-1 type-design H3: TyTemporal wrappers
        # must be transparent to refinement detection (otherwise
        # `Past<{x: f32 | x.is_finite()}>` reports False and
        # `_join_branch_types` silently drops the refinement).
        if isinstance(ty, TyTemporal):
            return self._contains_refinement(ty.inner, _seen_structs)
        # Stage 40 Inc 1: TyModal preemptive parallel arm.
        if isinstance(ty, TyModal):
            return self._contains_refinement(ty.inner, _seen_structs)
        # Stage 41 Inc 1: TyCausal preemptive parallel arm.
        if isinstance(ty, TyCausal):
            return self._contains_refinement(ty.inner, _seen_structs)
        # Stage 46 Inc 1: TyResult — refinement in either side
        # counts (the inner Ok side and the Err side are
        # symmetrically reachable).
        if isinstance(ty, TyResult):
            return (self._contains_refinement(ty.ok_ty, _seen_structs)
                    or self._contains_refinement(
                        ty.err_ty, _seen_structs))
        if isinstance(ty, TyTensor):
            return (self._contains_refinement(ty.dtype, _seen_structs)
                    or any(self._contains_refinement(s, _seen_structs)
                           for s in ty.shape))
        if isinstance(ty, TyTile):
            return (self._contains_refinement(ty.dtype, _seen_structs)
                    or any(self._contains_refinement(s, _seen_structs)
                           for s in ty.shape))
        if isinstance(ty, TyFn):
            return (any(self._contains_refinement(p, _seen_structs)
                        for p in ty.params)
                    or self._contains_refinement(ty.ret, _seen_structs))
        return False

    def _is_refinement_container(self, ty: Type) -> bool:
        # Stage 39 closure gate-1 type-design H3: include TyTemporal so
        # `_join_branch_types` correctly fires the refinement-shape
        # check for temporal-wrapped values across branches.
        # Stage 40 Inc 1: TyModal added preemptively (same rationale).
        # Stage 41 Inc 1: TyCausal added preemptively (same rationale).
        # Stage 46 Inc 1: TyResult added — both inners are refinement
        # containers (Ok and Err sides each carry a refineable type).
        return isinstance(ty, (
            TyArray, TyTuple, TyRef, TyPtr, TyFn, TyDiff, TyLogic, TyQuote,
            TyMemTier, TyFrame, TyTemporal, TyModal, TyCausal, TyResult,
            TyTensor, TyTile,
        ))

    def _contains_refined_function(self, ty: Type) -> bool:
        if isinstance(ty, TyFn):
            return self._contains_refinement(ty)
        if isinstance(ty, TyArray):
            return self._contains_refined_function(ty.elem)
        if isinstance(ty, TyTuple):
            return any(self._contains_refined_function(e) for e in ty.elems)
        if isinstance(ty, TyRef):
            return self._contains_refined_function(ty.inner)
        if isinstance(ty, TyPtr):
            return self._contains_refined_function(ty.inner)
        if isinstance(ty, TyDiff):
            return self._contains_refined_function(ty.inner)
        if isinstance(ty, TyLogic):
            return self._contains_refined_function(ty.inner)
        if isinstance(ty, TyQuote):
            return self._contains_refined_function(ty.inner)
        if isinstance(ty, TyMemTier):
            return self._contains_refined_function(ty.inner)
        if isinstance(ty, TyFrame):
            return self._contains_refined_function(ty.inner)
        # Stage 39 closure gate-1 type-design H3: TyTemporal walks
        # to inner for the refined-function check too.
        if isinstance(ty, TyTemporal):
            return self._contains_refined_function(ty.inner)
        # Stage 40 Inc 1: TyModal preemptive parallel arm.
        if isinstance(ty, TyModal):
            return self._contains_refined_function(ty.inner)
        # Stage 41 Inc 1: TyCausal preemptive parallel arm.
        if isinstance(ty, TyCausal):
            return self._contains_refined_function(ty.inner)
        # Stage 46 Inc 1: TyResult two-inner arm.
        if isinstance(ty, TyResult):
            return (self._contains_refined_function(ty.ok_ty)
                    or self._contains_refined_function(ty.err_ty))
        if isinstance(ty, TyTensor):
            return (self._contains_refined_function(ty.dtype)
                    or any(self._contains_refined_function(s)
                           for s in ty.shape))
        if isinstance(ty, TyTile):
            return (self._contains_refined_function(ty.dtype)
                    or any(self._contains_refined_function(s)
                           for s in ty.shape))
        return False

    def _eval_refinement_predicate(
        self, expr: A.Expr, self_value: int | float | bool | None,
        numeric_base: Type | None = None,
    ) -> Optional[bool]:
        if isinstance(expr, A.BoolLit):
            return expr.value
        if isinstance(expr, A.Unary) and expr.op == "!":
            inner = self._eval_refinement_predicate(
                expr.operand, self_value, numeric_base=numeric_base)
            if inner is None:
                return None
            return not inner
        if isinstance(expr, A.Binary) and expr.op in ("&&", "||"):
            left = self._eval_refinement_predicate(
                expr.left, self_value, numeric_base=numeric_base)
            right = self._eval_refinement_predicate(
                expr.right, self_value, numeric_base=numeric_base)
            if expr.op == "&&":
                if left is False or right is False:
                    return False
                if left is True and right is True:
                    return True
                return None
            if expr.op == "||":
                if left is True or right is True:
                    return True
                if left is False and right is False:
                    return False
                return None
            if left is None or right is None:
                return None
        chain = self._flatten_relational_chain(expr)
        if chain is not None:
            ops, operands = chain
            values = [self._eval_const_scalar_expr(
                e, self_value, honor_float_suffix=True,
                numeric_base=numeric_base)
                      for e in operands]
            if any(v is None for v in values):
                return None
            return all(self._compare_scalar(values[i], ops[i], values[i + 1])
                       for i in range(len(ops)))
        return None

    def _flatten_relational_chain(
        self, expr: A.Expr,
    ) -> Optional[tuple[list[str], list[A.Expr]]]:
        ops: list[str] = []
        rights: list[A.Expr] = []
        cur = expr
        while isinstance(cur, A.Binary) and cur.op in ("<", "<=", ">", ">=",
                                                       "==", "!="):
            ops.append(cur.op)
            rights.append(cur.right)
            cur = cur.left
        if not ops:
            return None
        return list(reversed(ops)), [cur] + list(reversed(rights))

    def _eval_const_scalar_expr(
        self, expr: A.Expr, self_value: int | float | bool | None,
        *, use_local_consts: bool = False, honor_float_suffix: bool = False,
        numeric_base: Type | None = None,
    ) -> Optional[int | float | bool]:
        if isinstance(expr, A.IntLit):
            return self._eval_int_lit_scalar(expr, numeric_base)
        if isinstance(expr, A.FloatLit):
            if honor_float_suffix:
                return self._eval_float_lit_scalar(expr)
            return expr.value
        if isinstance(expr, A.BoolLit):
            return expr.value
        if isinstance(expr, A.Name) and expr.generics:
            return None
        if isinstance(expr, A.Name) and expr.name == "self":
            return self_value
        if isinstance(expr, A.Name):
            if use_local_consts:
                found_local, local_value = self._lookup_local_const_scalar(
                    expr.name)
                if found_local:
                    return local_value
            return self._const_scalar_values.get(expr.name)
        if isinstance(expr, A.Cast):
            target_base = self._const_eval_type_node_base(expr.target_ty)
            if target_base is None:
                return None
            inner_base = (
                self._infer_const_expr_numeric_base(expr.value)
                or numeric_base
            )
            value = self._eval_const_scalar_expr(
                expr.value, self_value,
                use_local_consts=use_local_consts,
                honor_float_suffix=honor_float_suffix,
                numeric_base=inner_base,
            )
            return self._cast_const_scalar_to_type(value, target_base)
        if isinstance(expr, A.Unary) and expr.op == "-":
            inner = self._eval_const_scalar_expr(
                expr.operand, self_value,
                use_local_consts=use_local_consts,
                honor_float_suffix=honor_float_suffix,
                numeric_base=numeric_base,
            )
            if isinstance(inner, (int, float)) and not isinstance(inner, bool):
                return self._const_scalar_arithmetic_result(
                    -inner, numeric_base)
            return None
        if isinstance(expr, A.Binary):
            if expr.op in ("<", "<=", ">", ">=", "==", "!=", "&&", "||"):
                return self._eval_refinement_predicate(
                    expr, self_value, numeric_base=numeric_base)
            left = self._eval_const_scalar_expr(
                expr.left, self_value, use_local_consts=use_local_consts,
                honor_float_suffix=honor_float_suffix,
                numeric_base=numeric_base)
            right = self._eval_const_scalar_expr(
                expr.right, self_value, use_local_consts=use_local_consts,
                honor_float_suffix=honor_float_suffix,
                numeric_base=numeric_base)
            if not (isinstance(left, (int, float))
                    and isinstance(right, (int, float))
                    and not isinstance(left, bool)
                    and not isinstance(right, bool)):
                return None
            try:
                if expr.op == "+":
                    return self._const_scalar_arithmetic_result(
                        left + right, numeric_base)
                if expr.op == "-":
                    return self._const_scalar_arithmetic_result(
                        left - right, numeric_base)
                if expr.op == "*":
                    return self._const_scalar_arithmetic_result(
                        left * right, numeric_base)
                if expr.op == "/" and right != 0:
                    if self._numeric_base_is_int(numeric_base):
                        if isinstance(left, int) and isinstance(right, int):
                            return self._const_scalar_arithmetic_result(
                                self._trunc_div_int(left, right),
                                numeric_base)
                        return None
                    return self._const_scalar_arithmetic_result(
                        left / right, numeric_base)
                if expr.op == "%" and right != 0:
                    if self._numeric_base_is_int(numeric_base):
                        if isinstance(left, int) and isinstance(right, int):
                            return self._const_scalar_arithmetic_result(
                                self._trunc_mod_int(left, right),
                                numeric_base)
                        return None
                    return self._const_scalar_arithmetic_result(
                        left % right, numeric_base)
            except (OverflowError, ValueError):
                return None
        return None

    def _const_scalar_arithmetic_result(
        self, value: int | float, numeric_base: Type | None = None,
    ) -> int | float | None:
        if self._numeric_base_is_f32(numeric_base):
            if not (isinstance(value, (int, float))
                    and not isinstance(value, bool)):
                return None
            return self._round_const_scalar_to_f32(float(value))
        if self._numeric_base_is_int(numeric_base):
            return self._representable_int_arithmetic_result(
                value, numeric_base)
        return self._finite_const_scalar_result(value)

    def _finite_const_scalar_result(
        self, value: int | float,
    ) -> int | float | None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _numeric_base_is_f32(self, numeric_base: Type | None) -> bool:
        return isinstance(numeric_base, TyPrim) and numeric_base.name == "f32"

    def _numeric_base_is_int(self, numeric_base: Type | None) -> bool:
        return (
            isinstance(numeric_base, TyPrim)
            and numeric_base.name in _INT_PRIM_NAMES
        )

    def _numeric_base_is_float(self, numeric_base: Type | None) -> bool:
        return (
            isinstance(numeric_base, TyPrim)
            and numeric_base.name in _FLOAT_PRIM_NAMES
        )

    def _numeric_base_is_fixed_width_number(
        self, numeric_base: Type | None,
    ) -> bool:
        return (
            self._numeric_base_is_int(numeric_base)
            or self._numeric_base_is_float(numeric_base)
        )

    def _representable_int_arithmetic_result(
        self, value: int | float, numeric_base: Type | None,
    ) -> int | None:
        if (not isinstance(value, int)
                or isinstance(value, bool)
                or not isinstance(numeric_base, TyPrim)):
            return None
        bounds = self._INT_BOUNDS.get(numeric_base.name)
        if bounds is None:
            return None
        lo, hi = bounds
        if value < lo or value > hi:
            return None
        return value

    def _trunc_div_int(self, left: int, right: int) -> int:
        quotient = abs(left) // abs(right)
        if (left < 0) != (right < 0):
            quotient = -quotient
        return quotient

    def _trunc_mod_int(self, left: int, right: int) -> int:
        return left - self._trunc_div_int(left, right) * right

    def _eval_int_lit_scalar(
        self, expr: A.IntLit, numeric_base: Type | None,
    ) -> int | None:
        suffix = expr.type_suffix
        target: Type | None = None
        if suffix in _INT_PRIM_NAMES:
            target = TyPrim(suffix)
        elif suffix is not None:
            return None
        elif self._numeric_base_is_int(numeric_base):
            target = numeric_base
        if target is None:
            return expr.value
        represented = self._cast_const_scalar_to_type(expr.value, target)
        if isinstance(represented, int) and not isinstance(represented, bool):
            return represented
        return None

    def _eval_float_lit_scalar(self, expr: A.FloatLit) -> float | None:
        suffix = expr.type_suffix
        if suffix is None or suffix == "f32":
            return self._round_const_scalar_to_f32(float(expr.value))
        if suffix == "f64":
            value = float(expr.value)
            if not math.isfinite(value):
                return None
            return value
        if suffix in ("bf16", "f16", "fp8"):
            return None
        return expr.value

    def _compare_scalar(
        self, left: int | float | bool, op: str, right: int | float | bool,
    ) -> bool:
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        return False

    def _fmt_scalar_value(self, value: int | float | bool) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _fmt_refinement_expr(self, expr: A.Expr) -> str:
        chain = self._flatten_relational_chain(expr)
        if chain is not None:
            ops, operands = chain
            out = self._fmt_refinement_atom(operands[0])
            for op, operand in zip(ops, operands[1:]):
                out += f" {op} {self._fmt_refinement_atom(operand)}"
            return out
        return self._fmt_refinement_atom(expr)

    def _fmt_refinement_atom(self, expr: A.Expr) -> str:
        if isinstance(expr, A.IntLit):
            return str(expr.value)
        if isinstance(expr, A.FloatLit):
            return str(expr.value)
        if isinstance(expr, A.BoolLit):
            return "true" if expr.value else "false"
        if isinstance(expr, A.Name):
            if expr.generics:
                args = ", ".join(
                    self._fmt_refinement_ty_arg(arg)
                    for arg in expr.generics
                )
                return f"{expr.name}::<{args}>"
            return expr.name
        if isinstance(expr, A.Unary):
            return f"{expr.op}{self._fmt_refinement_atom(expr.operand)}"
        if isinstance(expr, A.Binary):
            return (f"({self._fmt_refinement_expr(expr.left)} {expr.op} "
                    f"{self._fmt_refinement_expr(expr.right)})")
        return type(expr).__name__

    def _fmt_refinement_ty_arg(self, ty: A.TyNode) -> str:
        if isinstance(ty, A.TyName):
            return ty.name
        if isinstance(ty, A.TyGeneric):
            args = ", ".join(
                self._fmt_refinement_ty_arg(arg) for arg in ty.args
            )
            return f"{ty.base}<{args}>"
        if isinstance(ty, A.TyTuple):
            return "(" + ", ".join(
                self._fmt_refinement_ty_arg(elem) for elem in ty.elems
            ) + ")"
        if isinstance(ty, A.TyArray):
            return (
                f"[{self._fmt_refinement_ty_arg(ty.elem)}; "
                f"{self._fmt_refinement_expr(ty.size)}]"
            )
        return type(ty).__name__

    @staticmethod
    def _suggest_wider_int(value: int, current: str) -> "Optional[str]":
        for cand in ("i32", "i64"):
            if cand == current:
                continue
            lo, hi = TypeChecker._INT_BOUNDS.get(cand, (0, 0))
            if lo <= value <= hi:
                return cand
        return None

    # ---- pattern binders ----
    def _bind_pattern(self, pat: A.Pattern, scrut_ty: Type, scope: Scope) -> None:
        """Recursively walk a match pattern and define any variable binders
        in `scope` with the appropriate inferred type from `scrut_ty`."""
        if isinstance(pat, A.PatWildcard):
            return
        if isinstance(pat, A.PatLit):
            self._check_expr(pat.value, scope)
            if isinstance(pat.value, A.Path):
                enum_variant = self._enum_variant_for_expr(pat.value)
                if enum_variant is not None and isinstance(scrut_ty, TyEnum):
                    ename, _variant = enum_variant
                    if ename != scrut_ty.name:
                        self.errors.append(TypeError_(
                            f"pattern {ename}::{_variant.name} cannot match "
                            f"scrutinee type {scrut_ty.name}",
                            pat.span,
                        ))
            return
        if isinstance(pat, A.PatBind):
            scope.define(pat.name, scrut_ty)
            return
        if isinstance(pat, A.PatTuple):
            if isinstance(scrut_ty, TyTuple) and len(scrut_ty.elems) == len(pat.elems):
                for sub_pat, sub_ty in zip(pat.elems, scrut_ty.elems):
                    self._bind_pattern(sub_pat, sub_ty, scope)
            else:
                for sub_pat in pat.elems:
                    self._bind_pattern(sub_pat, TyUnknown(hint="tuple-pat"), scope)
            return
        if isinstance(pat, A.PatOr):
            # Or-pattern semantics: a name is bound in the arm body iff
            # it is bound by EVERY alternative (the intersection). If a
            # name is bound by some alternatives but not all, accepting
            # it would be a type-safety hole — references in the body
            # could read uninitialized scope. Compute each alternative's
            # binder set in a throwaway scope, then define only the
            # intersection in the outer scope.
            alt_scopes: list[Scope] = []
            for alt in pat.alts:
                alt_scope = Scope(parent=None)
                self._bind_pattern(alt, scrut_ty, alt_scope)
                alt_scopes.append(alt_scope)
            if alt_scopes:
                common = set(alt_scopes[0].locals.keys())
                for s in alt_scopes[1:]:
                    common &= set(s.locals.keys())
                for name in common:
                    # Use the type from the first alternative's scope.
                    scope.define(name, alt_scopes[0].locals[name])
            return
        if isinstance(pat, A.PatRange):
            # Type-check both endpoints in the surrounding scope.
            self._check_expr(pat.lo, scope)
            self._check_expr(pat.hi, scope)
            return
        if isinstance(pat, A.PatVariant):
            # Look up the enum + variant; bind each sub-pattern against
            # its corresponding payload type.
            if len(pat.path.segments) == 2:
                ename, vname = pat.path.segments
                if isinstance(scrut_ty, TyEnum) and ename != scrut_ty.name:
                    self.errors.append(TypeError_(
                        f"pattern {ename}::{vname} cannot match scrutinee "
                        f"type {scrut_ty.name}",
                        pat.span,
                    ))
                    return
                edecl = getattr(self, "_enum_decls", {}).get(ename)
                if edecl is not None:
                    for v in edecl.variants:
                        if v.name == vname:
                            if len(pat.sub_patterns) != len(v.payload_tys):
                                self.errors.append(TypeError_(
                                    f"variant {ename}::{vname} expects "
                                    f"{len(v.payload_tys)} payload(s), got "
                                    f"{len(pat.sub_patterns)}",
                                    pat.span,
                                ))
                            for sub, pty in zip(pat.sub_patterns,
                                                v.payload_tys):
                                self._bind_pattern(
                                    sub, self._resolve_type(pty, scope),
                                    scope)
                            return
                    self.errors.append(TypeError_(
                        f"enum {ename!r} has no variant {vname!r}",
                        pat.span,
                    ))
            return

    def _pattern_covers(self, pat: A.Pattern, value) -> bool:
        """Does `pat` definitely match the given concrete `value`?
        Used by exhaustiveness for finite enumerable types."""
        if isinstance(pat, (A.PatWildcard, A.PatBind)):
            return True
        if isinstance(pat, A.PatLit):
            v = pat.value
            if isinstance(v, A.BoolLit) and isinstance(value, bool):
                return v.value == value
            return False
        if isinstance(pat, A.PatOr):
            return any(self._pattern_covers(a, value) for a in pat.alts)
        if isinstance(pat, A.PatTuple) and isinstance(value, tuple) \
                and len(value) == 0 and len(pat.elems) == 0:
            return True
        return False

    def _arm_variant_name(self, pat: A.Pattern,
                            expected_enum: str) -> "Optional[str]":
        """First-match helper used by callers that want a single variant
        name (e.g. error reporting). Use _arm_variant_names_all for
        exhaustiveness counting since PatOr arms cover multiple variants."""
        names = self._arm_variant_names_all(pat, expected_enum)
        return names[0] if names else None

    def _arm_variant_names_all(self, pat: A.Pattern,
                                expected_enum: str) -> list[str]:
        """Collect EVERY variant name an arm pattern covers from
        expected_enum. PatOr alts each contribute their own variant; this
        is what exhaustiveness counting needs (rather than first-match)."""
        out: list[str] = []
        if isinstance(pat, A.PatLit) and isinstance(pat.value, A.Path):
            segs = pat.value.segments
            if len(segs) == 2 and segs[0] == expected_enum:
                out.append(segs[1])
        elif isinstance(pat, A.PatVariant):
            segs = pat.path.segments
            if len(segs) == 2 and segs[0] == expected_enum:
                out.append(segs[1])
        elif isinstance(pat, A.PatOr):
            for alt in pat.alts:
                out.extend(self._arm_variant_names_all(alt, expected_enum))
        return out

    def _infer_enum_name_from_arms(self,
                                    arms: list[A.MatchArm]) -> "Optional[str]":
        """Heuristic: if every non-wildcard arm uses a Path / PatVariant
        rooted at the same EnumName, return that name. Otherwise None."""
        seen: "Optional[str]" = None
        any_path = False
        for arm in arms:
            pat = arm.pattern
            if isinstance(pat, (A.PatWildcard, A.PatBind)):
                continue
            n = self._arm_path_root_enum(pat)
            if n is None:
                # Non-enum pattern (e.g. PatLit of a number) — bail out.
                if isinstance(pat, (A.PatLit, A.PatRange, A.PatTuple)):
                    return None
                continue
            any_path = True
            if seen is None:
                seen = n
            elif seen != n:
                return None
        return seen if any_path else None

    def _arm_path_root_enum(self, pat: A.Pattern) -> "Optional[str]":
        """Get the enum name from a path-shaped arm pattern, if any."""
        if isinstance(pat, A.PatLit) and isinstance(pat.value, A.Path):
            segs = pat.value.segments
            if len(segs) == 2:
                return segs[0]
        if isinstance(pat, A.PatVariant):
            segs = pat.path.segments
            if len(segs) == 2:
                return segs[0]
        if isinstance(pat, A.PatOr):
            for alt in pat.alts:
                n = self._arm_path_root_enum(alt)
                if n is not None:
                    return n
        return None

    def _check_match_exhaustive(self, expr: A.Match, scrut_ty: Type) -> None:
        """Cheap exhaustiveness for finite types: bool ({true,false}) and
        unit (only ()). Anything else needs a wildcard or PatBind to be
        considered exhaustive."""
        # Any arm with no guard and a wildcard / bare-name binder is total.
        # Also: an or-pattern containing a wildcard or binder is total
        # (matches anything). Audit-10 finding #4 — `E::A | _` was
        # falsely flagged non-exhaustive.
        def _arm_is_total(p: A.Pattern) -> bool:
            if isinstance(p, (A.PatWildcard, A.PatBind)):
                return True
            if isinstance(p, A.PatOr):
                return any(_arm_is_total(a) for a in p.alts)
            return False
        for arm in expr.arms:
            if arm.guard is None and _arm_is_total(arm.pattern):
                return
        # Enum-shaped match: prefer the actual scrutinee enum. Inferring only
        # from arm roots lets `match A { B::X => ... }` confuse equal tag
        # numbers across different enums.
        enum_name = (scrut_ty.name if isinstance(scrut_ty, TyEnum)
                     else self._infer_enum_name_from_arms(expr.arms))
        if enum_name is not None:
            edecl = getattr(self, "_enum_decls", {}).get(enum_name)
            if edecl is not None:
                covered: set[str] = set()
                for arm in expr.arms:
                    if arm.guard is not None:
                        continue
                    # Collect ALL variants this arm covers — PatOr alts
                    # each contribute one; without this fix `E::A | E::C`
                    # would count only E::A and falsely flag E::C missing.
                    for name in self._arm_variant_names_all(
                            arm.pattern, enum_name):
                        covered.add(name)
                missing = [v.name for v in edecl.variants
                           if v.name not in covered]
                if missing:
                    self.errors.append(TypeError_(
                        f"non-exhaustive match on enum {enum_name!r}: "
                        f"missing variant(s) {missing}",
                        expr.span,
                    ))
                return
        if isinstance(scrut_ty, TyPrim) and scrut_ty.name == "bool":
            covers_true = any(arm.guard is None and self._pattern_covers(arm.pattern, True)
                              for arm in expr.arms)
            covers_false = any(arm.guard is None and self._pattern_covers(arm.pattern, False)
                               for arm in expr.arms)
            missing = []
            if not covers_true:  missing.append("true")
            if not covers_false: missing.append("false")
            if missing:
                self.errors.append(TypeError_(
                    f"non-exhaustive match on bool: missing {', '.join(missing)}",
                    expr.span,
                ))
            return
        if isinstance(scrut_ty, TyUnit):
            covers_unit = any(arm.guard is None and self._pattern_covers(arm.pattern, ())
                              for arm in expr.arms)
            if not covers_unit:
                self.errors.append(TypeError_(
                    "non-exhaustive match on unit: missing ()",
                    expr.span,
                ))
            return
        # For any other type (i32, f32, tuples with content, etc.) we need
        # a wildcard / binder arm to be exhaustive — the early-return above
        # handles this. Without one, emit a diagnostic.
        self.errors.append(TypeError_(
            f"non-exhaustive match on {self._fmt(scrut_ty)}: add a `_` arm",
            expr.span,
        ))

    # ---- compatibility (simplified) ----
    # ----- Allowed-cast matrix helpers (Audit 28.8 B14) -----
    _NUMERIC_INT_PRIMS = frozenset({
        "i8", "i16", "i32", "i64", "isize",
        "u8", "u16", "u32", "u64", "usize",
    })
    _NUMERIC_FLOAT_PRIMS = frozenset({
        "f16", "bf16", "f32", "f64", "fp8", "mxfp4", "nvfp4",
    })
    _NUMERIC_BOOL_PRIMS = frozenset({"bool", "char"})
    _HBM_TILE_PARAM_DTYPES = frozenset({"f32", "i32"})
    _HBM_TILE_INDEX_DTYPES = _HBM_TILE_PARAM_DTYPES

    def _is_numeric_scalar(self, t: Type) -> bool:
        t = self._erase_refinement(t)
        if not isinstance(t, TyPrim):
            return False
        return (t.name in self._NUMERIC_INT_PRIMS
                or t.name in self._NUMERIC_FLOAT_PRIMS
                or t.name in self._NUMERIC_BOOL_PRIMS)

    def _is_int_scalar(self, t: Type) -> bool:
        t = self._erase_refinement(t)
        return isinstance(t, TyPrim) and t.name in self._NUMERIC_INT_PRIMS

    def _strict_i32_truncation_hint(
        self, t: "Type", pre_inc_label: str, downstream_op: str,
    ) -> str:
        # Stage 37 post-closure L1 fix (Stage 36 gate-3 type-design audit,
        # conf 75): the family-standard "pre-Inc-N also accepted i64/u32/
        # u64 but those silently truncated" parenthetical is only TRUE
        # when the rejected type is itself a wider integer. For non-int
        # categories (Logic<i32>, struct, function, etc.) pre-fix would
        # have rejected the call too — claiming a truncation history
        # misleads the user. Gate the hint on _is_int_scalar so non-int
        # rejections get the bare "must be exactly i32" message.
        if not self._is_int_scalar(t):
            return ""
        return (f" ({pre_inc_label} also accepted i64/u32/u64 but those "
                f"silently truncated in downstream {downstream_op})")

    def _is_float_scalar(self, t: Type) -> bool:
        t = self._erase_refinement(t)
        return isinstance(t, TyPrim) and t.name in self._NUMERIC_FLOAT_PRIMS

    def _check_assignment_target(self, expr: A.Assign, scope: Scope) -> None:
        target = expr.target
        if isinstance(target, A.Name):
            if scope.lookup(target.name) is not None \
                    and not scope.lookup_mutable(target.name):
                self.errors.append(TypeError_(
                    f"cannot assign to immutable binding {target.name!r}",
                    target.span,
                    hint="declare the binding with `let mut`",
                ))
            return
        if isinstance(target, A.Index):
            if not isinstance(target.callee, A.Name):
                self.errors.append(TypeError_(
                    "invalid assignment target; indexed assignments require "
                    "a named array or tile binding",
                    target.span,
                ))
                return
            if (expr.op != "="
                    and isinstance(target.callee, A.Name)
                    and target.callee.name in self._current_hbm_tile_indexables):
                self.errors.append(TypeError_(
                    "compound assignment to HBM tile indices is not "
                    "supported; use load + arithmetic + store",
                    expr.span,
                ))
            return
        self.errors.append(TypeError_(
            "invalid assignment target; expected a mutable variable or "
            "index expression",
            target.span,
        ))

    def _check_wrapped_binary_operator_domain(self, left: Type, right: Type,
                                              op: str, span: A.Span) -> bool:
        if isinstance(left, (TyUnknown, TyVar, TySize)) \
                or isinstance(right, (TyUnknown, TyVar, TySize)):
            return True
        left = self._erase_refinement(left)
        right = self._erase_refinement(right)
        if not (isinstance(left, TyPrim) and isinstance(right, TyPrim)):
            self.errors.append(TypeError_(
                f"operator {op!r} does not support operand types "
                f"{self._fmt(left)} and {self._fmt(right)}",
                span,
                hint="wrapper arithmetic needs scalar inner types",
            ))
            return False
        left_is_int = self._is_int_scalar(left)
        right_is_int = self._is_int_scalar(right)
        left_is_float = self._is_float_scalar(left)
        right_is_float = self._is_float_scalar(right)
        float_arith = op in ("+", "-", "*", "/")
        int_only = op in ("%", "&", "|", "^", "<<", ">>")
        if float_arith:
            if (left_is_int or left_is_float) \
                    and (right_is_int or right_is_float):
                return True
        elif int_only:
            if left_is_int and right_is_int:
                return True
        else:
            self._check_plain_binary_scalar_compat(left, right, op, span)
            return False
        if left.name == right.name:
            self.errors.append(TypeError_(
                f"operator {op!r} does not support operand type "
                f"{self._fmt(left)}",
                span,
                hint="use an explicit supported inner scalar type for "
                     "wrapped arithmetic",
            ))
            return False
        self.errors.append(TypeError_(
            f"operator {op!r} has incompatible operand types "
            f"{self._fmt(left)} and {self._fmt(right)}",
            span,
            hint="use an explicit cast or a supported operator for these "
                 "wrapped inner types",
        ))
        return False

    def _check_plain_binary_scalar_compat(self, left: Type, right: Type,
                                          op: str, span: A.Span) -> None:
        if isinstance(left, (TyUnknown, TyVar, TySize)) \
                or isinstance(right, (TyUnknown, TyVar, TySize)):
            return
        left = self._erase_refinement(left)
        right = self._erase_refinement(right)
        if not (isinstance(left, TyPrim) and isinstance(right, TyPrim)):
            self.errors.append(TypeError_(
                f"operator {op!r} does not support operand types "
                f"{self._fmt(left)} and {self._fmt(right)}",
                span,
                hint="use an explicit scalar value or implement this "
                     "operator for the aggregate type",
            ))
            return
        is_eq = op in ("==", "!=")
        is_order = op in ("<", "<=", ">", ">=")
        left_is_int = self._is_int_scalar(left)
        right_is_int = self._is_int_scalar(right)
        left_is_float = self._is_float_scalar(left)
        right_is_float = self._is_float_scalar(right)
        left_is_bool = left.name == "bool"
        right_is_bool = right.name == "bool"
        left_is_char = left.name == "char"
        right_is_char = right.name == "char"
        float_arith = op in ("+", "-", "*", "/")
        int_only = op in ("%", "&", "|", "^", "<<", ">>")
        if is_eq:
            if left.name == right.name and (
                    left_is_int or left_is_float or left_is_bool
                    or left_is_char):
                return
            if left_is_int and right_is_int:
                return
        elif is_order:
            if left_is_int and right_is_int:
                return
            if left_is_float and right_is_float and left.name == right.name:
                return
        elif float_arith:
            if left_is_int and right_is_int:
                return
            if left_is_float and right_is_float and left.name == right.name:
                return
        elif int_only:
            if left_is_int and right_is_int:
                return
        if left.name == right.name:
            self.errors.append(TypeError_(
                f"operator {op!r} does not support operand type "
                f"{self._fmt(left)}",
                span,
                hint="use an explicit cast or a supported operator for "
                     "this type",
            ))
            return
        self.errors.append(TypeError_(
            f"operator {op!r} has incompatible operand types "
            f"{self._fmt(left)} and {self._fmt(right)}",
            span,
            hint="use an explicit cast so lowering and codegen agree "
                 "on the operand representation",
        ))

    def _check_cast_compat(self, src: Type, tgt: Type, span: A.Span,
                            _depth: int = 0,
                            _outer_src: Type | None = None,
                            _outer_tgt: Type | None = None) -> None:
        """Audit 28.8 B14 (trap 28604): reject scalar casts whose
        source/target pair isn't in the allowed matrix.

        Allowed:
          - numeric scalar <-> numeric scalar (int, float, bool, char)
          - TyUnknown on either side (cascade-safe)
          - struct <-> same struct (identity cast)
          - generic / unknown TyVar (defer to mono)
          - tuple/array/struct/unit -> identical type (no-op cast)
        Otherwise: error 28604.

        Audit 28.8 cycle 3 D7: peel matching ref-pair wrappers
        iteratively before recursing so deeply-nested ref casts can't
        blow the Python recursion stack. Trap 28803 fires if the
        cast-matrix is invoked at a nesting depth beyond the guard
        (8) — a defense in depth against malicious / autogenerated
        sources, since real Phase-0 syntax has no way to write more
        than a few nested refs.

        Audit 28.8 cycle 5 C4-7 / F6: track the OUTER (pre-peel) src
        and tgt so the trap-28604 diagnostic renders `&Foo cannot
        convert to &Bar` (with the `&` prefix preserved) rather than
        the peeled inners. The recursive call after ref-peel passes
        the same outer types so the inner failure still surfaces with
        the original source-level pretty-print."""
        # Audit 28.8 cycle 5 C4-7 / F6: remember the user-visible
        # outer types so the diagnostic preserves `&` prefix.
        if _outer_src is None:
            _outer_src = src
        if _outer_tgt is None:
            _outer_tgt = tgt
        # Cascade-safe: unknown / generic types skip the check.
        if isinstance(src, (TyUnknown, TyVar, TySize)) \
                or isinstance(tgt, (TyUnknown, TyVar, TySize)):
            return
        # Audit 28.8 cycle 3 D7: peel matching TyRef wrappers
        # iteratively (in lockstep on both sides) so we don't burn
        # recursion budget on each layer. After the loop, fall through
        # to the rest of the matrix on the unwrapped inner pair.
        peeled = 0
        while isinstance(src, TyRef) and isinstance(tgt, TyRef) \
                and src.is_mut == tgt.is_mut:
            src = src.inner
            tgt = tgt.inner
            peeled += 1
            if peeled > 8:
                # Hard limit on ref-peel depth — emit a structured
                # error rather than risk RecursionError on the rest
                # of the matrix (trap 28803).
                self.errors.append(TypeError_(
                    f"invalid cast: ref-nesting depth exceeds "
                    f"8 levels (trap {TRAP_CAST_MATRIX_RECURSION_DEPTH})",
                    span,
                    hint="Phase-0 caps recursive cast-matrix depth "
                         "to keep the typechecker stack-bounded",
                ))
                return
        if not (peeled == 0 and (
                isinstance(src, TyRef) or isinstance(tgt, TyRef))):
            # Identity / no-op cast is always OK. Keep this after the
            # TyRef peel so dataclass equality cannot recurse through a
            # user-authored tower of references before the depth cap fires.
            if src == tgt:
                return
        # Numeric-scalar <-> numeric-scalar in either direction.
        if self._is_numeric_scalar(src) and self._is_numeric_scalar(tgt):
            return
        # After peeling, src/tgt may be a scalar (and the numeric
        # arm above runs again via the recursive call). Use a depth
        # guard regardless so any future non-Ref recursive arm is
        # also bounded.
        if _depth > 8:
            self.errors.append(TypeError_(
                f"invalid cast: cast-matrix recursion exceeds "
                f"8 levels (trap {TRAP_CAST_MATRIX_RECURSION_DEPTH})",
                span,
                hint="Phase-0 caps recursive cast-matrix depth "
                     "to keep the typechecker stack-bounded",
            ))
            return
        # Re-check the unwrapped pair through the (now scalar-or-
        # nominal) matrix at one bumped depth level. Pass outer types
        # through so the failure diagnostic prints the user-visible
        # source/target (with `&` prefix preserved).
        if peeled > 0:
            self._check_cast_compat(src, tgt, span,
                                    _depth=_depth + 1,
                                    _outer_src=_outer_src,
                                    _outer_tgt=_outer_tgt)
            return
        # All other source/target pairs are invalid scalar casts.
        # Common slipups: tuple-as-i32, struct-as-f64, unit-as-Pt,
        # and (post B:C5 fix) &Foo as &Bar.
        # Cycle 5 C4-7 / F6: render the OUTER types so `&Foo` prints
        # as `&Foo`, not the peeled inner `Foo`.
        self.errors.append(TypeError_(
            f"invalid cast: source {self._fmt(_outer_src)} cannot "
            f"convert to {self._fmt(_outer_tgt)} (trap 28604)",
            span,
            hint="numeric scalar casts (int<->int, int<->float, "
                 "float<->float, bool/char<->int) are allowed; other "
                 "shapes need explicit construction",
        ))

    def _ad_warn_mixed_inner(self, span: A.Span, l: Type, r: Type,
                              chosen: Type, extra: str = "") -> None:
        """Audit 28.8 B13 (trap AD002 / 24200): record a warning that
        a TyDiff binop received mixed inner types and we widened to
        the dominant one. Funneled into the autodiff warning channel
        so check.py's existing -Wad=error policy applies.

        Audit 28.8 cycle 2 B:C4 / B:C6: an optional `extra` suffix
        names the sub-case (same-rank tie / asymmetric D-wrap) so
        the user can tell apart the now-three classes of widening:
          * D<T1> + D<T2> with T1, T2 of different rank
          * D<T> + bareT  (one D-wrapped, one raw)
          * same-rank tie (sign or quantization-domain transition)
        """
        from . import autodiff as _ad
        _ad._DIFF_WARNINGS.append(
            f"{span.line}:{span.col}: AD: D-binop with mixed inner "
            f"types {self._fmt(l)} vs {self._fmt(r)} — widened to "
            f"{self._fmt(chosen)} (trap 24200/AD002)" + extra
        )

    def _size_compatible(self, a: Type, b: Type) -> bool:
        """Audit 28.8 cycle 7 C6-1: shape-position-only cascade for
        TyVar / TySize. Used inside TyArray / TyTensor / TyTile size
        compares — the cycle-5 audit's option (b). The body-of-function-
        position uses the full `_compatible` (no auto-cascade for
        TyVar at the top), so `fn g[T]() -> T { 42 }` correctly emits
        the "body type i32 does not match return type T" error that
        the cycle-6 top-level F1 cascade had silently swallowed."""
        if isinstance(a, (TyVar, TySize)) or isinstance(b, (TyVar, TySize)):
            return True
        if isinstance(a, TyUnknown) or isinstance(b, TyUnknown):
            return True
        if a == b:
            return True
        return self._compatible(a, b)

    def _compatible(self, a: Type, b: Type) -> bool:
        if isinstance(a, TyUnknown) or isinstance(b, TyUnknown):
            return True
        if isinstance(a, TyRefined) and isinstance(b, TyRefined):
            return self._compatible(a.base, b.base)
        if isinstance(a, TyRefined):
            return self._compatible(a.base, b)
        if isinstance(b, TyRefined):
            return self._compatible(a, b.base)
        if isinstance(a, TyEnum) and isinstance(b, TyEnum):
            return a.name == b.name
        if isinstance(a, TyEnum) or isinstance(b, TyEnum):
            return False
        # Memory-tier types are incompatible across tiers (must explicitly
        # consolidate / recall to convert).
        #
        # Audit 28.8 cycle 8 C7-1: dropped cycle-7 G2's TyMemTier × (TyVar
        # | TySize) carve-out. The carve-out was placed at top-level
        # `_compatible` and leaked silent-acceptance to body / let / if-
        # else / match-arm value-position callsites — the same over-broad
        # cascade pattern that cycle-7 narrowed for F1 via
        # `_size_compatible`. TyMemTier × TySize is a genuine kind
        # mismatch (a size can't be a memory-tier); TyMemTier × TyVar at
        # value position is rare enough that a hard error is preferable
        # to silent acceptance. If a generic-over-MemTier pattern emerges
        # later, re-introduce the carve-out only at the call boundary
        # (`_check_call_basic`) rather than in the structural matcher.
        #
        # Audit 28.8 cycle 5 F4 / MEDIUM: tier compare uses raw string
        # equality (`a.tier == b.tier`). This does NOT recognize tier
        # subsumption — conceptually HBM ⊆ DDR for read-only accesses,
        # so a HBM-stored value could pass to a DDR-typed param. Phase-0
        # limitation: strict equality only. When a tier-subsumption
        # matrix is specced (Phase-1+), this arm needs a subsumption
        # check rather than equality. Deferred enhancement.
        if isinstance(a, TyMemTier) and isinstance(b, TyMemTier):
            return a.tier == b.tier and self._compatible(a.inner, b.inner)
        if isinstance(a, TyMemTier) or isinstance(b, TyMemTier):
            return False
        # Stage 38 post-Inc-3 type-design H1 fix (HIGH, conf 90):
        # TyFrame wrapper arm. Cross-frame inputs are caught by the
        # dataclass-equality fallthrough, but refined/generic/shape-
        # symbolic inners need the explicit recursive `_compatible`
        # delegation that every other wrapper has. Mirrors TyMemTier.
        if isinstance(a, TyFrame) and isinstance(b, TyFrame):
            return a.frame == b.frame and self._compatible(a.inner, b.inner)
        if isinstance(a, TyFrame) or isinstance(b, TyFrame):
            return False
        # Stage 39 closure gate-1 type-design H1: TyTemporal needs the
        # parallel arm — same `a == b` dataclass-equality risk for
        # refined / generic / shape-symbolic inners. `Past<i32>` vs
        # raw `i32` must be rejected (no silent unwrap on the
        # `_compatible` path the way `from_past` enforces it).
        if isinstance(a, TyTemporal) and isinstance(b, TyTemporal):
            return a.kind == b.kind and self._compatible(a.inner, b.inner)
        if isinstance(a, TyTemporal) or isinstance(b, TyTemporal):
            return False
        # Stage 40 Inc 1: TyModal preemptive parallel arm (same H1
        # rationale — `Known<i32>` must not be silently compatible
        # with raw i32 / cross-kind Modal).
        if isinstance(a, TyModal) and isinstance(b, TyModal):
            return a.kind == b.kind and self._compatible(a.inner, b.inner)
        if isinstance(a, TyModal) or isinstance(b, TyModal):
            return False
        # Stage 41 Inc 1: TyCausal preemptive parallel arm.
        if isinstance(a, TyCausal) and isinstance(b, TyCausal):
            return a.kind == b.kind and self._compatible(a.inner, b.inner)
        if isinstance(a, TyCausal) or isinstance(b, TyCausal):
            return False
        # Stage 46 Inc 1: TyResult two-inner arm. Both Ok and Err
        # sides must be compatible; reject mixed wrapper pairs.
        if isinstance(a, TyResult) and isinstance(b, TyResult):
            return (self._compatible(a.ok_ty, b.ok_ty)
                    and self._compatible(a.err_ty, b.err_ty))
        if isinstance(a, TyResult) or isinstance(b, TyResult):
            return False
        # Audit 28.8 cycle 2 B:C3: Quote<T> ~ Quote<U> iff T ~ U.
        # Reject Quote<T> ~ T (raw value passed where Quote expected)
        # — that was the silent acceptance path pre-fix.
        if isinstance(a, TyQuote) and isinstance(b, TyQuote):
            return self._compatible(a.inner, b.inner)
        if isinstance(a, TyQuote) or isinstance(b, TyQuote):
            return False
        # Audit 28.8 cycle 3 D1: wrapper types must agree by kind AND
        # by inner. TyDiff(T) ~ TyDiff(U) iff T ~ U; TyDiff(T) ~ U is
        # rejected (raw passed where D expected) — symmetric to the
        # Quote arm above. Same for TyLogic. This closes the call
        # boundary silent-acceptance hole exposed by D1.
        #
        # Audit 28.8 cycle 5 F2 / MEDIUM: TyDiff currently has no
        # sub-domain metadata (smooth / non-smooth / jacobian variants).
        # When that metadata is specced (Phase-1+), this arm needs to
        # also compare the diff-domain markers. Phase-0 limitation:
        # `D<T>` is treated as a single domain. Documented for cycle-6+
        # follow-up once the sub-domain spec lands.
        if isinstance(a, TyDiff) and isinstance(b, TyDiff):
            return self._compatible(a.inner, b.inner)
        if isinstance(a, TyDiff) or isinstance(b, TyDiff):
            return False
        # Audit 28.8 cycle 5 F3 / MEDIUM: TyLogic has provenance
        # metadata (the `provenance` field) but `_compatible` only
        # checks the inner type. Phase-0 limitation: provenance
        # matching is handled separately via
        # `_logic_provenance_violation_kind` at the call boundary
        # (trap 24100), not in the structural-equality check. When a
        # comprehensive sub-domain matrix lands (Phase-1+), this arm
        # may need to also compare provenance tier markers. Documented
        # for cycle-6+ follow-up.
        if isinstance(a, TyLogic) and isinstance(b, TyLogic):
            return self._compatible(a.inner, b.inner)
        if isinstance(a, TyLogic) or isinstance(b, TyLogic):
            return False
        # Structural arms for the remaining composite types so D1's
        # `_compatible` fall-through at the call boundary recognizes
        # both TyTuple, TyArray, TyRef, TyPtr, TyFn pairs by their
        # inner structure rather than identity-comparison only.
        if isinstance(a, TyTuple) and isinstance(b, TyTuple):
            if len(a.elems) != len(b.elems):
                return False
            return all(self._compatible(x, y)
                       for x, y in zip(a.elems, b.elems))
        if isinstance(a, TyTuple) or isinstance(b, TyTuple):
            return False
        if isinstance(a, TyArray) and isinstance(b, TyArray):
            # Audit 28.8 cycle 4 E1: size compare uses `_size_compatible`
            # (cycle 7 narrowing of cycle 6 F1) — TyVar/TySize defer
            # only at shape positions, not at value-type positions.
            return (self._compatible(a.elem, b.elem)
                    and (a.size == b.size
                         or self._size_compatible(a.size, b.size)))
        if isinstance(a, TyArray) or isinstance(b, TyArray):
            return False
        if isinstance(a, TyRef) and isinstance(b, TyRef):
            return (a.is_mut == b.is_mut
                    and self._compatible(a.inner, b.inner))
        if isinstance(a, TyRef) or isinstance(b, TyRef):
            return False
        if isinstance(a, TyPtr) and isinstance(b, TyPtr):
            return (a.is_mut == b.is_mut
                    and self._compatible(a.inner, b.inner))
        if isinstance(a, TyPtr) or isinstance(b, TyPtr):
            return False
        if isinstance(a, TyFn) and isinstance(b, TyFn):
            if len(a.params) != len(b.params):
                return False
            return (all(self._compatible(x, y)
                        for x, y in zip(a.params, b.params))
                    and self._compatible(a.ret, b.ret))
        if isinstance(a, TyFn) or isinstance(b, TyFn):
            return False
        # Audit 28.8 cycle 4 C4-4: TyTile/TyTensor arms. D1's commit
        # message named these as silent-acceptance holes to close but
        # the patch omitted them. Tensor/tile pairs are equal iff dtype
        # and shape (positionally) agree; device/layout/memspace are
        # markers compared nominally.
        if isinstance(a, TyTensor) and isinstance(b, TyTensor):
            if len(a.shape) != len(b.shape):
                return False
            # Cycle 7 C6-1: shape elements use _size_compatible (narrow
            # cascade), dtype uses _compatible (full).
            return (self._compatible(a.dtype, b.dtype)
                    and all(self._size_compatible(x, y)
                            for x, y in zip(a.shape, b.shape))
                    and a.device == b.device
                    and a.layout == b.layout)
        if isinstance(a, TyTensor) or isinstance(b, TyTensor):
            return False
        if isinstance(a, TyTile) and isinstance(b, TyTile):
            if len(a.shape) != len(b.shape):
                return False
            return (self._compatible(a.dtype, b.dtype)
                    and all(self._size_compatible(x, y)
                            for x, y in zip(a.shape, b.shape))
                    and a.memspace == b.memspace)
        if isinstance(a, TyTile) or isinstance(b, TyTile):
            return False
        return a == b

    def _fmt_size(self, t: Type) -> str:
        """Audit 28.8 cycle 5 F7 / F8 / LOW: render a size-typed value
        cleanly in user diagnostics. Pre-fix `_fmt(TyPrim('size_3'))`
        printed `size_3` (with `size_` prefix); `_fmt(TySize('N'))`
        printed `size:N`. Diagnostics like `expected tensor<f32, [N]>,
        got tensor<f32, [3]>` were ambiguous. Now: concrete sizes
        print as their integer value (3); symbolic sizes print as
        their generic-param name (N) without the `size:` prefix."""
        if isinstance(t, TyPrim) and t.name.startswith("size_"):
            return t.name[len("size_"):]
        if isinstance(t, TySize):
            return t.name
        return self._fmt(t)

    def _fmt(self, t: Type) -> str:
        if isinstance(t, TyPrim): return t.name
        if isinstance(t, TyRefined): return t.name
        # Audit 28.8 cycle 3 D8: print TyStruct as its declared name
        # (e.g. `Foo`) instead of falling through to repr (which gave
        # `TyStruct(name='Foo')` in user-facing diagnostics).
        if isinstance(t, TyStruct): return t.name
        if isinstance(t, TyEnum): return t.name
        if isinstance(t, TyVar): return t.name
        if isinstance(t, TySize): return f"size:{t.name}"
        if isinstance(t, TyTensor):
            # Audit 28.8 cycle 5 F7: use _fmt_size for shape elements
            # so `tensor<f32, [3]>` prints as `tensor<f32, [3]>` (not
            # `tensor<f32, [size_3]>`). Symbolic sizes still print
            # their generic-param name (N) without `size:` prefix.
            shp = ",".join(self._fmt_size(s) for s in t.shape)
            return f"tensor<{self._fmt(t.dtype)}, [{shp}]" + (f", {t.device}" if t.device else "") + ">"
        if isinstance(t, TyTile):
            # Audit 28.8 cycle 5 F7: ditto for tile shape elements.
            shp = ",".join(self._fmt_size(s) for s in t.shape)
            return f"tile<{self._fmt(t.dtype)}, [{shp}], {t.memspace}>"
        if isinstance(t, TyTuple):
            return "(" + ", ".join(self._fmt(e) for e in t.elems) + ")"
        if isinstance(t, TyArray):
            # Audit 28.8 cycle 5 F9: TyArray's `_fmt` already includes
            # the size via the existing call. Cycle 5 F7: render size
            # via `_fmt_size` so `[i32; 4]` prints clean (not `[i32;
            # size_4]`).
            return f"[{self._fmt(t.elem)}; {self._fmt_size(t.size)}]"
        if isinstance(t, TyRef):
            return ("&mut " if t.is_mut else "&") + self._fmt(t.inner)
        if isinstance(t, TyPtr):
            return ("*mut " if t.is_mut else "*const ") + self._fmt(t.inner)
        if isinstance(t, TyFn):
            return f"fn({', '.join(self._fmt(p) for p in t.params)}) -> {self._fmt(t.ret)}"
        if isinstance(t, TyUnit): return "()"
        if isinstance(t, TyDiff): return f"D<{self._fmt(t.inner)}>"
        if isinstance(t, TyLogic):
            base = f"Logic<{self._fmt(t.inner)}>"
            if t.provenance:
                return f"{base}@{t.provenance}"
            return base
        if isinstance(t, TyMemTier):
            cap = {"working": "WorkingMem", "episodic": "EpisodicMem",
                   "semantic": "SemanticMem", "procedural": "ProceduralMem"}
            return f"{cap.get(t.tier, t.tier)}<{self._fmt(t.inner)}>"
        if isinstance(t, TyFrame):
            cap = {"world": "WorldFrame", "robot": "RobotFrame",
                   "camera": "CameraFrame"}
            return f"{cap.get(t.frame, t.frame)}<{self._fmt(t.inner)}>"
        if isinstance(t, TyTemporal):
            cap = {"past": "Past", "present": "Present",
                   "future": "Future", "eternal": "Eternal"}
            return f"{cap.get(t.kind, t.kind)}<{self._fmt(t.inner)}>"
        if isinstance(t, TyModal):
            cap = {"known": "Known", "believed": "Believed",
                   "goal": "Goal", "uncertain": "Uncertain"}
            return f"{cap.get(t.kind, t.kind)}<{self._fmt(t.inner)}>"
        if isinstance(t, TyCausal):
            cap = {"cause": "Cause", "effect": "Effect",
                   "joint": "Joint", "independent": "Independent"}
            return f"{cap.get(t.kind, t.kind)}<{self._fmt(t.inner)}>"
        if isinstance(t, TyResult):
            return (f"Result<{self._fmt(t.ok_ty)}, "
                    f"{self._fmt(t.err_ty)}>")
        if isinstance(t, TySkill):
            tag = f' "{t.task}"' if t.task else ""
            return f"Skill<{self._fmt(t.inner)}{tag}>"
        if isinstance(t, TyQuote):
            return f"Quote<{self._fmt(t.inner)}>"
        if isinstance(t, TyUnknown): return f"?{{{t.hint}}}"
        return repr(t)


def typecheck(prog: A.Program) -> list[TypeError_]:
    return TypeChecker(prog).check()


def typecheck_with_obligations(
    prog: A.Program,
) -> tuple[list[TypeError_], list[ProofObligation]]:
    errors, obligations, _carries = typecheck_with_proof_artifacts(prog)
    return errors, obligations


def typecheck_with_proof_artifacts(
    prog: A.Program,
) -> tuple[list[TypeError_], list[ProofObligation], list[ProofCarry]]:
    checker = TypeChecker(prog)
    errors = checker.check()
    return errors, checker.proof_obligations, checker.proof_carries


if __name__ == "__main__":
    import sys
    from .parser import parse
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        with open(filename) as f:
            src = f.read()
    else:
        filename = "<stdin>"
        src = sys.stdin.read()
    prog = parse(src)
    errors = typecheck(prog)
    for e in errors:
        print(e.render(source=src, filename=filename), file=sys.stderr)
        print(file=sys.stderr)
    sys.exit(1 if errors else 0)
