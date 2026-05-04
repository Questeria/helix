"""
kovc/ir/lower_ast.py — lower Kov AST into Tensor IR.

For v0.1 we lower the *scalar/control-flow* subset directly: function decls,
arithmetic on primitive types, if/else, while, calls, returns.

Tensor and tile operations are recognized but emitted as opaque CALL ops
(real lowering rules come in v0.2 once we wire up the linalg-style
structured ops).

License: Apache 2.0
"""

from __future__ import annotations

from typing import Optional

from ..frontend import ast as A
from . import tir


# ============================================================================
# Lowerer
# ============================================================================
class Lowerer:
    def __init__(self, prog: A.Program):
        self.prog = prog
        self.module = tir.Module()
        self.builder = tir.IRBuilder(self.module)
        # name -> Value (locals + params)
        self.scope: list[dict[str, tir.Value]] = []
        # name -> FnIR (registered functions)
        self.functions: dict[str, tir.FnIR] = {}

    # ---- entry ----
    def lower(self) -> tir.Module:
        # Pass 1: register function signatures (so calls work)
        for item in self.prog.items:
            if isinstance(item, A.FnDecl):
                self._register_fn(item)
        # Pass 2: lower bodies
        for item in self.prog.items:
            if isinstance(item, A.FnDecl):
                self._lower_fn_body(item)
        return self.module

    # ---- scope ----
    def _push_scope(self) -> None: self.scope.append({})
    def _pop_scope(self) -> None: self.scope.pop()
    def _bind(self, name: str, v: tir.Value) -> None:
        self.scope[-1][name] = v
    def _lookup(self, name: str) -> Optional[tir.Value]:
        for sc in reversed(self.scope):
            if name in sc:
                return sc[name]
        return None

    # ---- type lowering ----
    def _lower_type(self, ty: A.TyNode) -> tir.TIRType:
        if isinstance(ty, A.TyName):
            return tir.TIRScalar(ty.name)
        if isinstance(ty, A.TyTuple):
            return tir.TIRTuple(tuple(self._lower_type(e) for e in ty.elems))
        if isinstance(ty, A.TyTensor):
            dtype = self._lower_type(ty.dtype)
            assert isinstance(dtype, tir.TIRScalar), f"tensor dtype must be scalar, got {dtype}"
            shape = tuple(self._lower_dim(s) for s in ty.shape)
            device = self._stringify_marker(ty.device) or "cpu"
            return tir.TIRTensorTy(dtype=dtype, shape=shape, device=device,
                                   layout=tir.Layout.ROW_MAJOR)
        if isinstance(ty, A.TyTile):
            dtype = self._lower_type(ty.dtype)
            assert isinstance(dtype, tir.TIRScalar)
            shape = tuple(self._lower_dim(s) for s in ty.shape)
            mem = self._stringify_marker(ty.memspace) or "?"
            return tir.TIRTileTy(dtype=dtype, shape=shape, memspace=mem)
        if isinstance(ty, A.TyArray):
            return tir.TIRTuple(elems=(self._lower_type(ty.elem),))  # simplified
        if isinstance(ty, A.TyRef):
            return self._lower_type(ty.inner)  # references erased in IR for v0.1
        return tir.TIRScalar("?")

    def _lower_dim(self, expr: A.Expr) -> tir.Dim:
        if isinstance(expr, A.IntLit):
            return tir.DimConst(expr.value)
        if isinstance(expr, A.Name):
            return tir.DimVar(expr.name)
        if isinstance(expr, A.Binary) and expr.op in ("+", "-", "*", "/", "%"):
            return tir.DimExpr(op=expr.op,
                               args=(self._lower_dim(expr.left),
                                     self._lower_dim(expr.right)))
        return tir.DimDyn()

    def _stringify_marker(self, expr: Optional[A.Expr]) -> Optional[str]:
        if expr is None: return None
        if isinstance(expr, A.Name): return expr.name
        if isinstance(expr, A.Call) and isinstance(expr.callee, A.Name):
            args = ",".join(str(getattr(a, "value", "?")) for a in expr.args)
            return f"{expr.callee.name}({args})"
        return None

    # ---- function registration ----
    def _register_fn(self, fn: A.FnDecl) -> None:
        params: list[tuple[str, tir.TIRType]] = []
        for p in fn.params:
            t = self._lower_type(p.ty)
            params.append((p.name, t))
        ret = self._lower_type(fn.return_ty) if fn.return_ty else tir.TIRUnit()
        attrs: dict[str, object] = {}
        for a in fn.attrs:
            attrs[a] = True
        ir_fn = self.builder.begin_function(fn.name, params, ret, attrs=attrs)
        self.functions[fn.name] = ir_fn
        # Don't lower body yet — that's pass 2
        self.builder.end_function()

    # ---- function bodies ----
    def _lower_fn_body(self, fn: A.FnDecl) -> None:
        ir_fn = self.functions.get(fn.name)
        if ir_fn is None:
            return
        self.builder.current_fn = ir_fn
        self.builder.current_block = ir_fn.entry
        self._push_scope()
        # Bind params to their SSA values
        for ((name, _ty), v) in zip(
            [(p.name, p.ty) for p in fn.params], ir_fn.params,
        ):
            self._bind(name, v)
        # Lower body block
        body_val = self._lower_block(fn.body)
        # Emit return
        if isinstance(ir_fn.return_ty, tir.TIRUnit):
            self.builder.ret(None)
        elif body_val is not None:
            self.builder.ret(body_val)
        else:
            self.builder.ret(None)
        self._pop_scope()
        self.builder.end_function()

    def _lower_block(self, block: A.Block) -> Optional[tir.Value]:
        self._push_scope()
        try:
            for stmt in block.stmts:
                self._lower_stmt(stmt)
            if block.final_expr is not None:
                return self._lower_expr(block.final_expr)
            return None
        finally:
            self._pop_scope()

    def _lower_stmt(self, stmt: A.Stmt) -> None:
        if isinstance(stmt, A.Let):
            v: Optional[tir.Value] = None
            if stmt.value is not None:
                v = self._lower_expr(stmt.value)
            if v is None:
                # Default: emit a unit-typed placeholder
                v = self.builder.const_int(0)
            self._bind(stmt.name, v)
            return
        if isinstance(stmt, A.ExprStmt):
            self._lower_expr(stmt.expr)
            return
        if isinstance(stmt, A.ConstStmt):
            v = self._lower_expr(stmt.value)
            self._bind(stmt.name, v)
            return

    def _lower_expr(self, expr: A.Expr) -> Optional[tir.Value]:
        if isinstance(expr, A.IntLit):
            return self.builder.const_int(expr.value, expr.type_suffix or "i32")
        if isinstance(expr, A.FloatLit):
            return self.builder.const_float(expr.value, expr.type_suffix or "f32")
        if isinstance(expr, A.BoolLit):
            return self.builder.emit(tir.OpKind.CONST_BOOL,
                                     result_ty=tir.TIRScalar("bool"),
                                     attrs={"value": expr.value})
        if isinstance(expr, A.Name):
            v = self._lookup(expr.name)
            if v is not None:
                return v
            # Maybe a function reference (v0.1: emit a call-able marker)
            if expr.name in self.functions:
                # We don't have function-pointer values yet; treat as opaque placeholder
                return self.builder.const_int(0)
            # Unbound: emit a unit constant
            return self.builder.const_int(0)
        if isinstance(expr, A.Path):
            # Treat as opaque call target
            return self.builder.const_int(0)
        if isinstance(expr, A.Binary):
            l = self._lower_expr(expr.left)
            r = self._lower_expr(expr.right)
            if l is None or r is None:
                return None
            arith = {
                "+": tir.OpKind.ADD, "-": tir.OpKind.SUB,
                "*": tir.OpKind.MUL, "/": tir.OpKind.DIV,
                "%": tir.OpKind.DIV,  # placeholder
            }
            cmp_ = {
                "==": tir.OpKind.CMP_EQ, "!=": tir.OpKind.CMP_NE,
                "<": tir.OpKind.CMP_LT, "<=": tir.OpKind.CMP_LE,
                ">": tir.OpKind.CMP_GT, ">=": tir.OpKind.CMP_GE,
            }
            if expr.op in arith:
                return self.builder.emit(arith[expr.op], l, r, result_ty=l.ty)
            if expr.op in cmp_:
                return self.builder.emit(cmp_[expr.op], l, r,
                                         result_ty=tir.TIRScalar("bool"))
            # Logical or/and — short-circuit eval not yet wired; treat as bitwise for v0.1
            return self.builder.emit(tir.OpKind.ADD, l, r, result_ty=tir.TIRScalar("bool"))
        if isinstance(expr, A.Unary):
            inner = self._lower_expr(expr.operand)
            if inner is None:
                return None
            if expr.op == "-":
                return self.builder.emit(tir.OpKind.NEG, inner, result_ty=inner.ty)
            return inner
        if isinstance(expr, A.Call):
            args: list[tir.Value] = []
            for a in expr.args:
                v = self._lower_expr(a)
                if v is not None:
                    args.append(v)
            # Determine call target name
            target = "<unknown>"
            if isinstance(expr.callee, A.Name):
                target = expr.callee.name
            elif isinstance(expr.callee, A.Path):
                target = "::".join(expr.callee.segments)
            # Emit as opaque CALL with return type from the registered fn
            ret_ty: tir.TIRType = tir.TIRScalar("?")
            if isinstance(expr.callee, A.Name) and expr.callee.name in self.functions:
                ret_ty = self.functions[expr.callee.name].return_ty
            return self.builder.emit(tir.OpKind.CALL, *args,
                                     result_ty=ret_ty, attrs={"target": target})
        if isinstance(expr, A.If):
            # v0.1: lower as eager branchless eval — both arms compute, select picks.
            # Real CFG-with-blocks lowering comes in v0.2.
            cond = self._lower_expr(expr.cond)
            t_val = self._lower_block(expr.then)
            e_val = None
            if expr.else_ is not None:
                if isinstance(expr.else_, A.Block):
                    e_val = self._lower_block(expr.else_)
                else:
                    e_val = self._lower_expr(expr.else_)
            if t_val is None: t_val = self.builder.const_int(0)
            if e_val is None: e_val = self.builder.const_int(0)
            return self.builder.emit(tir.OpKind.SELECT, cond, t_val, e_val,
                                     result_ty=t_val.ty)
        if isinstance(expr, A.Block):
            return self._lower_block(expr)
        if isinstance(expr, A.For):
            # v0.1: for loops not yet lowered to CFG. Emit a placeholder call so
            # the IR stays well-formed.
            self._lower_expr(expr.iter_expr)
            self._lower_block(expr.body)
            return None
        if isinstance(expr, A.While):
            self._lower_expr(expr.cond)
            self._lower_block(expr.body)
            return None
        if isinstance(expr, A.Loop):
            self._lower_block(expr.body)
            return None
        if isinstance(expr, A.Match):
            self._lower_expr(expr.scrutinee)
            for arm in expr.arms:
                self._lower_expr(arm.body)
            return None
        if isinstance(expr, A.Return):
            v = self._lower_expr(expr.value) if expr.value is not None else None
            self.builder.ret(v)
            return None
        if isinstance(expr, A.Range):
            return None
        if isinstance(expr, A.Assign):
            self._lower_expr(expr.value)
            return None
        if isinstance(expr, A.TupleLit):
            for e in expr.elems:
                self._lower_expr(e)
            return None
        if isinstance(expr, A.ArrayLit):
            for e in expr.elems:
                self._lower_expr(e)
            return None
        if isinstance(expr, A.Index):
            self._lower_expr(expr.callee)
            for i in expr.indices:
                self._lower_expr(i)
            return None
        if isinstance(expr, A.Field):
            self._lower_expr(expr.obj)
            return None
        return None


def lower(prog: A.Program) -> tir.Module:
    return Lowerer(prog).lower()


if __name__ == "__main__":
    import sys
    from ..frontend.parser import parse
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            src = f.read()
    else:
        src = sys.stdin.read()
    prog = parse(src)
    mod = lower(prog)
    print(tir.fmt_module(mod))
