"""
kovc/frontend/typecheck.py — Kov type checker (v0.1 scaffold).

Scope of v0.1:
- Resolve names (functions, generic params, locals)
- Check basic primitive type compatibility
- Track size-parameters as opaque integer-valued symbols
- Defer the real Presburger constraint solver to v0.2

Out of scope for v0.1:
- Full Presburger arithmetic on size constraints (records constraints; doesn't solve)
- Trait resolution
- Effect inference
- Linear/affine borrow checking

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import ast as A


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


# ============================================================================
# Type errors
# ============================================================================
class TypeError_(Exception):
    def __init__(self, msg: str, span: A.Span):
        super().__init__(f"{span.line}:{span.col}: type error: {msg}")
        self.span = span


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


# ============================================================================
# Type checker
# ============================================================================
class TypeChecker:
    def __init__(self, prog: A.Program):
        self.prog = prog
        self.functions: dict[str, FunctionSig] = {}
        self.constraints: list[A.Expr] = []   # collected, not solved (v0.1)
        self.errors: list[TypeError_] = []

    # ---- entry point ----
    def check(self) -> list[TypeError_]:
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

        sig = FunctionSig(name=fn.name, generics=fn.generics, params=params, ret=ret)
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

    # ---- function body checking ----
    def _check_fn(self, fn: A.FnDecl) -> None:
        sig = self.functions.get(fn.name)
        if sig is None:
            return
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
            return TyUnknown(hint=f"unbound {expr.name}")
        if isinstance(expr, A.Path):
            # v0.1: paths are unresolved (e.g., tensor::zeros, tile::matmul)
            return TyUnknown(hint=f"path {'::'.join(expr.segments)}")
        if isinstance(expr, A.Unary):
            inner = self._check_expr(expr.operand, scope)
            return inner
        if isinstance(expr, A.Binary):
            l = self._check_expr(expr.left, scope)
            r = self._check_expr(expr.right, scope)
            if expr.op in ("==", "!=", "<", "<=", ">", ">=", "&&", "||"):
                return TyPrim("bool")
            # Arithmetic: take the left type (simplified)
            return l
        if isinstance(expr, A.Call):
            callee = self._check_expr(expr.callee, scope)
            for a in expr.args:
                self._check_expr(a, scope)
            if isinstance(callee, TyFn):
                return callee.ret
            return TyUnknown(hint="call")
        if isinstance(expr, A.Index):
            self._check_expr(expr.callee, scope)
            for i in expr.indices:
                self._check_expr(i, scope)
            return TyUnknown(hint="index")
        if isinstance(expr, A.Field):
            self._check_expr(expr.obj, scope)
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
            self._check_expr(expr.scrutinee, scope)
            ts = [self._check_expr(arm.body, scope) for arm in expr.arms]
            if not ts:
                return TyUnit()
            return ts[0]
        if isinstance(expr, A.For):
            self._check_expr(expr.iter_expr, scope)
            inner = Scope(parent=scope)
            inner.define(expr.var_name, TyPrim("i64"))   # default loop var
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
        if isinstance(expr, (A.Break, A.Continue, A.Return)):
            return TyUnit()
        if isinstance(expr, A.Cast):
            self._check_expr(expr.value, scope)
            return self._resolve_type(expr.target_ty, scope)
        return TyUnknown(hint=f"unhandled {type(expr).__name__}")

    # ---- compatibility (simplified) ----
    def _compatible(self, a: Type, b: Type) -> bool:
        if isinstance(a, TyUnknown) or isinstance(b, TyUnknown):
            return True
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
        if isinstance(t, TyFn):
            return f"fn({', '.join(self._fmt(p) for p in t.params)}) -> {self._fmt(t.ret)}"
        if isinstance(t, TyUnit): return "()"
        if isinstance(t, TyUnknown): return f"?{{{t.hint}}}"
        return repr(t)


def typecheck(prog: A.Program) -> list[TypeError_]:
    return TypeChecker(prog).check()


if __name__ == "__main__":
    import sys
    from .parser import parse
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            src = f.read()
    else:
        src = sys.stdin.read()
    prog = parse(src)
    errors = typecheck(prog)
    for e in errors:
        print(e, file=sys.stderr)
    sys.exit(1 if errors else 0)
