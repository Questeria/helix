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
      - Trap 24001 emitted if a non-Logic value is passed where a
        Logic-typed parameter is required, or vice versa, in a
        provenance-sensitive context (e.g. AD over a logic op)."""
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
    "()",
}


@dataclass
class Scope:
    parent: Optional["Scope"] = None
    locals: dict[str, Type] = field(default_factory=dict)

    def lookup(self, name: str) -> Optional[Type]:
        if name in self.locals:
            return self.locals[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None

    def define(self, name: str, ty: Type) -> None:
        self.locals[name] = ty


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
        # Cascade-suppression set for unbound-name diagnostics. Initialized
        # here so re-running check() on the same instance doesn't carry
        # stale entries that would silence real new errors.
        self._seen_unbound: set[str] = set()

    # ---- entry point ----
    def check(self) -> list[TypeError_]:
        # Pass 0: index struct + enum decls *first* so that function
        # signatures referring to a nominal struct/enum resolve to
        # TyStruct/TyEnum (was: pass 1.5, which left struct-typed
        # params as TyUnknown until body-check).
        self._struct_decls: dict[str, A.StructDecl] = {}
        self._enum_decls: dict[str, A.EnumDecl] = {}
        for item in self.prog.items:
            if isinstance(item, A.StructDecl):
                self._struct_decls[item.name] = item
            elif isinstance(item, A.EnumDecl):
                self._enum_decls[item.name] = item

        # Pass 1: register function signatures (don't check bodies yet)
        for item in self.prog.items:
            if isinstance(item, A.FnDecl):
                try:
                    self._register_fn(item)
                except TypeError_ as e:
                    self.errors.append(e)

        # Pass 2: check function bodies
        for item in self.prog.items:
            if isinstance(item, A.FnDecl):
                try:
                    self._check_fn(item)
                except TypeError_ as e:
                    self.errors.append(e)

        return self.errors

    # ---- registration ----
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
            params.append((p.name, t))

        # Resolve return type
        if fn.return_ty is not None:
            ret = self._resolve_type(fn.return_ty, gen_scope)
        else:
            ret = TyUnit()

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

    # ---- type resolution ----
    def _resolve_type(self, ty: A.TyNode, scope: Scope) -> Type:
        if isinstance(ty, A.TyName):
            if ty.name in PRIMITIVES:
                return TyPrim(ty.name)
            looked = scope.lookup(ty.name)
            if looked is not None:
                return looked
            # Recognise nominal struct types so field-access on a struct-
            # typed field (e.g. nested struct: `inner: Inner`) gets a real
            # TyStruct instead of falling all the way to TyUnknown — that
            # was breaking chained field-type tracking and causing every
            # struct-field-typecheck to trivially pass.
            if ty.name in getattr(self, "_struct_decls", {}):
                return TyStruct(name=ty.name)
            # Unresolved: treat as unknown user type for v0.1
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
            # User type with generic args — v0.1 unknown
            return TyUnknown(hint=f"generic {ty.base}")
        return TyUnknown(hint=f"unknown ty node {type(ty).__name__}")

    def _resolve_size_expr(self, expr: A.Expr, scope: Scope) -> Type:
        """A size-expression is either a literal int, a name (size param), or
        an arithmetic expression. v0.1 represents complex exprs as TyUnknown
        with the source preserved by reference (not copied here)."""
        if isinstance(expr, A.IntLit):
            return TyPrim(f"size_{expr.value}")
        if isinstance(expr, A.Name):
            looked = scope.lookup(expr.name)
            if looked is not None:
                return looked
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
                          arg_tys: list[Type]) -> None:
        """Check argument count and primitive type compatibility.
        Tensor-shape checking is in _check_call_shapes."""
        if len(arg_tys) != len(sig.params):
            self.errors.append(TypeError_(
                f"call to {sig.name!r}: expected {len(sig.params)} args, "
                f"got {len(arg_tys)}",
                call.span,
            ))
            return
        for (pname, pty), aty in zip(sig.params, arg_tys):
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
        "detach", "attach", "consolidate", "recall", "learn_to",
        "grad", "grad_rev", "grad_rev_all",
        "quote", "splice", "splice_f", "modify", "modify_f",
        "print_str", "print_int", "write_file", "read_file_int",
        "read_file_to_arena", "write_file_to_arena",
        "__arena_push", "__arena_get", "__arena_set", "__arena_len",
        "__strlen", "__strbyte", "__streq", "__strlit_to_arena",
        "__hash_i32",
        # Phase 2.2 step 2 — float-bit reinterpret intrinsics.
        "__bits_of_f32", "__f32_from_bits",
        "__bits_of_f64", "__f64_from_bits",
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
        # Stage 16.5: extern "C" declarations have no body to check.
        if fn.is_extern:
            return
        sig = self.functions.get(fn.name)
        if sig is None:
            return
        # Set effect-checking context for this function
        prev_pure = self._current_pure
        prev_effects = self._current_effects
        prev_name = self._current_fn_name
        self._current_pure = sig.is_pure
        self._current_effects = sig.effects
        self._current_fn_name = sig.name
        try:
            self._check_fn_body(fn, sig)
        finally:
            self._current_pure = prev_pure
            self._current_effects = prev_effects
            self._current_fn_name = prev_name

    def _check_fn_body(self, fn: A.FnDecl, sig: FunctionSig) -> None:
        gen_scope = Scope()
        for g in fn.generics:
            if g.kind == "size":
                gen_scope.define(g.name, TySize(g.name))
            else:
                gen_scope.define(g.name, TyVar(g.name))
        body_scope = Scope(parent=gen_scope)
        for name, t in sig.params:
            body_scope.define(name, t)
        # Check body expression / block
        body_ty = self._check_block(fn.body, body_scope)
        # Compatibility check (simplified — strict equality on resolved types)
        if not self._compatible(body_ty, sig.ret):
            self.errors.append(TypeError_(
                f"function {fn.name!r}: body type {self._fmt(body_ty)} "
                f"does not match return type {self._fmt(sig.ret)}",
                fn.span,
            ))

    def _check_block(self, block: A.Block, scope: Scope) -> Type:
        inner = Scope(parent=scope)
        for stmt in block.stmts:
            self._check_stmt(stmt, inner)
        if block.final_expr is not None:
            return self._check_expr(block.final_expr, inner)
        return TyUnit()

    def _check_stmt(self, stmt: A.Stmt, scope: Scope) -> None:
        if isinstance(stmt, A.Let):
            value_ty: Type = TyUnit()
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
                scope.define(stmt.name, declared)
            else:
                scope.define(stmt.name, value_ty)
            return
        if isinstance(stmt, A.ExprStmt):
            self._check_expr(stmt.expr, scope)
            return
        if isinstance(stmt, A.ConstStmt):
            ty = self._resolve_type(stmt.ty, scope)
            scope.define(stmt.name, ty)
            return

    def _check_expr(self, expr: A.Expr, scope: Scope) -> Type:
        if isinstance(expr, A.IntLit):
            # Default integer type is i32 unless suffix specified
            return TyPrim(expr.type_suffix or "i32")
        if isinstance(expr, A.FloatLit):
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
            # Function reference?
            if expr.name in self.functions:
                sig = self.functions[expr.name]
                return TyFn(tuple(t for _, t in sig.params), sig.ret)
            # Name truly unbound. Emit a soft diagnostic with a Levenshtein
            # "did you mean?" suggestion drawn from in-scope names + known
            # functions. Don't raise — return TyUnknown so cascade-style
            # downstream errors are suppressed.
            self._unbound_name_suggestion(expr.name, expr.span, scope)
            return TyUnknown(hint=f"unbound {expr.name}")
        if isinstance(expr, A.Path):
            # Check for `EnumName::VariantName` paths.
            if len(expr.segments) == 2:
                ename, vname = expr.segments
                edecl = getattr(self, "_enum_decls", {}).get(ename)
                if edecl is not None:
                    for v in edecl.variants:
                        if v.name == vname:
                            # Payload-bearing variants need constructor-call
                            # form; bare path on those is an error.
                            if v.payload_tys:
                                self.errors.append(TypeError_(
                                    f"enum variant {ename}::{vname} has "
                                    f"payload — call as a function instead",
                                    expr.span,
                                ))
                            return TyPrim("i32")  # tag-only variant
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
            return inner
        if isinstance(expr, A.Binary):
            l = self._check_expr(expr.left, scope)
            r = self._check_expr(expr.right, scope)
            if expr.op in ("==", "!=", "<", "<=", ">", ">=", "&&", "||"):
                return TyPrim("bool")
            # Differentiability propagation: if either operand is D<T>,
            # the result is D<T>. Mixing D<T1> with D<T2>: result is D<T1>
            # (simplified; real compiler would unify innerness).
            l_is_diff = isinstance(l, TyDiff)
            r_is_diff = isinstance(r, TyDiff)
            if l_is_diff or r_is_diff:
                inner = l.inner if l_is_diff else r.inner if r_is_diff else l
                return TyDiff(inner=inner)
            # Arithmetic: take the left type (simplified)
            return l
        if isinstance(expr, A.Call):
            # Stage 16.5: "literal".as_ptr() — type is *const u8 (TyPtr(u8, mut=False)).
            if (isinstance(expr.callee, A.Field)
                    and expr.callee.name == "as_ptr"
                    and isinstance(expr.callee.obj, A.StrLit)
                    and len(expr.args) == 0):
                return TyPtr(inner=TyPrim("u8"), is_mut=False)
            # Payload-bearing enum constructor: `Maybe::Some(42)`.
            if (isinstance(expr.callee, A.Path)
                    and len(expr.callee.segments) == 2):
                ename, vname = expr.callee.segments
                edecl = getattr(self, "_enum_decls", {}).get(ename)
                if edecl is not None:
                    for v in edecl.variants:
                        if v.name == vname:
                            # Type-check args against payload_tys.
                            arg_tys = [self._check_expr(a, scope) for a in expr.args]
                            if len(arg_tys) != len(v.payload_tys):
                                self.errors.append(TypeError_(
                                    f"enum variant {ename}::{vname} expects "
                                    f"{len(v.payload_tys)} payload arg(s), "
                                    f"got {len(arg_tys)}",
                                    expr.span,
                                ))
                            else:
                                for i, (at, pt) in enumerate(zip(arg_tys, v.payload_tys)):
                                    expected = self._resolve_type(pt, scope)
                                    if not self._compatible(at, expected):
                                        self.errors.append(TypeError_(
                                            f"enum {ename}::{vname} arg {i}: "
                                            f"expected {self._fmt(expected)}, "
                                            f"got {self._fmt(at)}",
                                            expr.span,
                                        ))
                            return TyPrim("i32")  # tagged value backed by [tag, ...payload]
                    self.errors.append(TypeError_(
                        f"enum {ename!r} has no variant {vname!r}",
                        expr.span,
                    ))
                    for a in expr.args:
                        self._check_expr(a, scope)
                    return TyUnknown(hint=f"enum {ename}")
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
            # If callee is a known function (by name), do checks
            if isinstance(expr.callee, A.Name) and expr.callee.name in self.functions:
                sig = self.functions[expr.callee.name]
                self._check_call_basic(expr, sig, arg_tys)
                self._check_call_shapes(expr, sig, arg_tys, scope)
                self._check_call_effects(expr, sig)
                return sig.ret
            if isinstance(callee, TyFn):
                return callee.ret
            return TyUnknown(hint="call")
        if isinstance(expr, A.Index):
            self._check_expr(expr.callee, scope)
            for i in expr.indices:
                self._check_expr(i, scope)
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
            if expr.else_ is not None:
                if isinstance(expr.else_, A.Block):
                    e = self._check_block(expr.else_, scope)
                else:
                    e = self._check_expr(expr.else_, scope)
                if not self._compatible(t, e):
                    self.errors.append(TypeError_(
                        f"if/else branches differ: {self._fmt(t)} vs {self._fmt(e)}",
                        expr.span,
                    ))
            return t
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
            return first
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
            self._check_expr(expr.target, scope)
            return TyUnit()
        if isinstance(expr, A.TupleLit):
            return TyTuple(tuple(self._check_expr(e, scope) for e in expr.elems))
        if isinstance(expr, A.ArrayLit):
            ts = [self._check_expr(e, scope) for e in expr.elems]
            elem = ts[0] if ts else TyUnknown(hint="empty array")
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
            return TyStruct(name=expr.name)
        if isinstance(expr, (A.Break, A.Continue, A.Return)):
            return TyUnit()
        if isinstance(expr, A.Cast):
            self._check_expr(expr.value, scope)
            return self._resolve_type(expr.target_ty, scope)
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
        # Enum-shaped match: if every non-trivial arm uses a Path / PatVariant
        # rooted at the same enum, check coverage of that enum's variants.
        enum_name = self._infer_enum_name_from_arms(expr.arms)
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
    def _compatible(self, a: Type, b: Type) -> bool:
        if isinstance(a, TyUnknown) or isinstance(b, TyUnknown):
            return True
        # Memory-tier types are incompatible across tiers (must explicitly
        # consolidate / recall to convert)
        if isinstance(a, TyMemTier) and isinstance(b, TyMemTier):
            return a.tier == b.tier and self._compatible(a.inner, b.inner)
        if isinstance(a, TyMemTier) or isinstance(b, TyMemTier):
            return False
        return a == b

    def _fmt(self, t: Type) -> str:
        if isinstance(t, TyPrim): return t.name
        if isinstance(t, TyVar): return t.name
        if isinstance(t, TySize): return f"size:{t.name}"
        if isinstance(t, TyTensor):
            shp = ",".join(self._fmt(s) for s in t.shape)
            return f"tensor<{self._fmt(t.dtype)}, [{shp}]" + (f", {t.device}" if t.device else "") + ">"
        if isinstance(t, TyTile):
            shp = ",".join(self._fmt(s) for s in t.shape)
            return f"tile<{self._fmt(t.dtype)}, [{shp}], {t.memspace}>"
        if isinstance(t, TyTuple):
            return "(" + ", ".join(self._fmt(e) for e in t.elems) + ")"
        if isinstance(t, TyArray):
            return f"[{self._fmt(t.elem)}; {self._fmt(t.size)}]"
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
        if isinstance(t, TyUnknown): return f"?{{{t.hint}}}"
        return repr(t)


def typecheck(prog: A.Program) -> list[TypeError_]:
    return TypeChecker(prog).check()


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
