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
from typing import Optional

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
        self._seen_unknown_type_names: set[str] = set()
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
            tier_map = {
                "WorkingMem": "working",
                "EpisodicMem": "episodic",
                "SemanticMem": "semantic",
                "ProceduralMem": "procedural",
            }
            if ty.base in tier_map and len(ty.args) == 1:
                return TyMemTier(tier=tier_map[ty.base],
                                 inner=self._resolve_type(ty.args[0], scope))
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
        "consolidate", "recall", "learn_to",
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
    def _check_fn(self, fn: A.FnDecl) -> None:
        sig = self.functions.get(fn.name)
        if sig is None:
            return
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
            self._pop_local_const_scope()

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
                    and expr.callee.name in self._GPU_INDEX_BUILTINS):
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
                    if not self._is_int_scalar(arg_tys[1]):
                        self.errors.append(TypeError_(
                            f"prove(value, source): source must be i32, "
                            f"got {self._fmt(arg_tys[1])}",
                            expr.span,
                        ))
                    inner = arg_tys[0]
                    if isinstance(inner, TyLogic):
                        return inner
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
                    return arg_tys[0]
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
            t = self._check_block(expr.then, scope)
            branch_tys = [t]
            if expr.else_ is not None:
                if isinstance(expr.else_, A.Block):
                    e = self._check_block(expr.else_, scope)
                else:
                    e = self._check_expr(expr.else_, scope)
                branch_tys.append(e)
                if not self._compatible(t, e):
                    self.errors.append(TypeError_(
                        f"if/else branches differ: {self._fmt(t)} vs {self._fmt(e)}",
                        expr.span,
                    ))
            return self._join_branch_types(branch_tys, expr.span)
        if isinstance(expr, A.Match):
            scrut_ty = self._check_expr(expr.scrutinee, scope)
            arm_tys: list[Type] = []
            for arm in expr.arms:
                inner = Scope(parent=scope)
                self._bind_pattern(arm.pattern, scrut_ty, inner)
                if arm.guard is not None:
                    g_ty = self._check_expr(arm.guard, inner)
                    if not (isinstance(g_ty, TyPrim) and g_ty.name == "bool") \
                            and not isinstance(g_ty, TyUnknown):
                        self.errors.append(TypeError_(
                            f"match guard must be bool, got {self._fmt(g_ty)}",
                            arm.span,
                        ))
                arm_tys.append(self._check_expr(arm.body, inner))
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
            self._check_block(expr.body, inner)
            return TyUnit()
        if isinstance(expr, A.While):
            self._check_expr(expr.cond, scope)
            self._check_block(expr.body, scope)
            return TyUnit()
        if isinstance(expr, A.Loop):
            self._check_block(expr.body, scope)
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
        return isinstance(ty, (
            TyArray, TyTuple, TyRef, TyPtr, TyFn, TyDiff, TyLogic, TyQuote,
            TyMemTier, TyTensor, TyTile,
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
