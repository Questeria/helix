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
        # name -> Value (locals + params, immutable)
        self.scope: list[dict[str, tir.Value]] = []
        # Mutable variables: a separate set of names that are stored as cells
        # (LOAD_VAR/STORE_VAR ops). Their type comes from initial binding.
        self.mut_scope: list[dict[str, tir.TIRType]] = []
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
    def _push_scope(self) -> None:
        self.scope.append({})
        self.mut_scope.append({})
    def _pop_scope(self) -> None:
        self.scope.pop()
        self.mut_scope.pop()
    def _bind(self, name: str, v: tir.Value) -> None:
        self.scope[-1][name] = v
    def _bind_mut(self, name: str, ty: tir.TIRType) -> None:
        self.mut_scope[-1][name] = ty
    def _lookup(self, name: str) -> Optional[tir.Value]:
        for sc in reversed(self.scope):
            if name in sc:
                return sc[name]
        return None
    def _lookup_mut(self, name: str) -> Optional[tir.TIRType]:
        for sc in reversed(self.mut_scope):
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
                v = self.builder.const_int(0)
            if stmt.is_mut:
                # Allocate a mutable cell, store the initial value
                self.builder.emit(tir.OpKind.ALLOC_VAR,
                                  attrs={"name": stmt.name, "dtype": v.ty})
                self.builder.emit(tir.OpKind.STORE_VAR, v,
                                  attrs={"name": stmt.name})
                self._bind_mut(stmt.name, v.ty)
            else:
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
            # Mutable variable -> emit LOAD_VAR
            mut_ty = self._lookup_mut(expr.name)
            if mut_ty is not None:
                return self.builder.emit(tir.OpKind.LOAD_VAR,
                                         result_ty=mut_ty,
                                         attrs={"name": expr.name})
            # Maybe a function reference (v0.1: emit a call-able marker)
            if expr.name in self.functions:
                return self.builder.const_int(0)
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
                "%": tir.OpKind.MOD,
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
            # Real CFG-based if/else: cond_br -> then_block | else_block,
            # both branches end with br merge_block(value). Result is the
            # merge block's parameter.
            cond = self._lower_expr(expr.cond)
            if cond is None:
                cond = self.builder.const_int(0)
            then_blk = self.builder.append_block()
            else_blk = self.builder.append_block()
            merge_blk = self.builder.append_block()

            # Emit conditional branch: if cond, go to then_blk; else, else_blk
            self.builder.emit(tir.OpKind.COND_BR, cond,
                              attrs={"true_block": then_blk.id,
                                     "false_block": else_blk.id})

            # Then arm
            self.builder.switch_to(then_blk)
            t_val = self._lower_block(expr.then)
            if t_val is None:
                t_val = self.builder.const_int(0)
            self.builder.emit(tir.OpKind.BR, t_val,
                              attrs={"target_block": merge_blk.id})

            # Else arm
            self.builder.switch_to(else_blk)
            if expr.else_ is None:
                e_val = self.builder.const_int(0)
            elif isinstance(expr.else_, A.Block):
                e_val = self._lower_block(expr.else_) or self.builder.const_int(0)
            else:
                e_val = self._lower_expr(expr.else_) or self.builder.const_int(0)
            self.builder.emit(tir.OpKind.BR, e_val,
                              attrs={"target_block": merge_blk.id})

            # Merge: the if's value is the merge block's single param
            self.builder.switch_to(merge_blk)
            result = self.builder.new_block_param(t_val.ty, hint="if_result")
            return result
        if isinstance(expr, A.Block):
            return self._lower_block(expr)
        if isinstance(expr, A.For):
            # Desugar `for i in start..end { body }` into:
            #   let mut __for_i = start
            #   let __for_end = end       (immutable cache)
            #   while __for_i < __for_end {
            #       <body with i bound to LOAD_VAR(__for_i)>
            #       __for_i += 1
            #   }
            # We generate unique var names by prefixing with __for_<linenum>_
            if not isinstance(expr.iter_expr, A.Range) or expr.iter_expr.start is None or expr.iter_expr.end is None:
                # Non-range for: not supported in v0.1
                self._lower_expr(expr.iter_expr)
                self._lower_block(expr.body)
                return None

            tag = f"__for_{expr.span.line}_{expr.span.col}_"
            iter_var = tag + expr.var_name
            end_var = tag + "end"

            start_v = self._lower_expr(expr.iter_expr.start) or self.builder.const_int(0)
            self.builder.emit(tir.OpKind.ALLOC_VAR,
                              attrs={"name": iter_var, "dtype": start_v.ty})
            self.builder.emit(tir.OpKind.STORE_VAR, start_v,
                              attrs={"name": iter_var})

            end_v = self._lower_expr(expr.iter_expr.end) or self.builder.const_int(0)
            self.builder.emit(tir.OpKind.ALLOC_VAR,
                              attrs={"name": end_var, "dtype": end_v.ty})
            self.builder.emit(tir.OpKind.STORE_VAR, end_v,
                              attrs={"name": end_var})

            self._bind_mut(iter_var, start_v.ty)
            self._bind_mut(end_var, end_v.ty)

            # Loop CFG
            header_blk = self.builder.append_block()
            body_blk = self.builder.append_block()
            exit_blk = self.builder.append_block()
            self.builder.emit(tir.OpKind.BR,
                              attrs={"target_block": header_blk.id})

            # Header: cond = iter < end
            self.builder.switch_to(header_blk)
            i_val = self.builder.emit(tir.OpKind.LOAD_VAR, result_ty=start_v.ty,
                                      attrs={"name": iter_var})
            e_val = self.builder.emit(tir.OpKind.LOAD_VAR, result_ty=end_v.ty,
                                      attrs={"name": end_var})
            cond = self.builder.emit(tir.OpKind.CMP_LT, i_val, e_val,
                                     result_ty=tir.TIRScalar("bool"))
            self.builder.emit(tir.OpKind.COND_BR, cond,
                              attrs={"true_block": body_blk.id,
                                     "false_block": exit_blk.id})

            # Body: bind expr.var_name to a load of iter_var, lower body, then i += 1
            self.builder.switch_to(body_blk)
            self._push_scope()
            try:
                # Each body iteration loads the current i value
                cur = self.builder.emit(tir.OpKind.LOAD_VAR, result_ty=start_v.ty,
                                        attrs={"name": iter_var})
                self._bind(expr.var_name, cur)
                self._lower_block(expr.body)
            finally:
                self._pop_scope()
            # Increment i
            cur_i = self.builder.emit(tir.OpKind.LOAD_VAR, result_ty=start_v.ty,
                                      attrs={"name": iter_var})
            one = self.builder.const_int(1)
            new_i = self.builder.emit(tir.OpKind.ADD, cur_i, one, result_ty=start_v.ty)
            self.builder.emit(tir.OpKind.STORE_VAR, new_i,
                              attrs={"name": iter_var})
            self.builder.emit(tir.OpKind.BR,
                              attrs={"target_block": header_blk.id})

            self.builder.switch_to(exit_blk)
            return None
        if isinstance(expr, A.While):
            # Real CFG-based loop:
            #   br header_block
            # header_block:
            #   cond = lower(expr.cond)
            #   cond_br cond, body_block, exit_block
            # body_block:
            #   lower(expr.body)
            #   br header_block
            # exit_block:
            #   (continues with following code)
            header_blk = self.builder.append_block()
            body_blk = self.builder.append_block()
            exit_blk = self.builder.append_block()
            # Branch from current block to header
            self.builder.emit(tir.OpKind.BR,
                              attrs={"target_block": header_blk.id})
            # Header
            self.builder.switch_to(header_blk)
            cond = self._lower_expr(expr.cond)
            if cond is None:
                cond = self.builder.const_int(0)
            self.builder.emit(tir.OpKind.COND_BR, cond,
                              attrs={"true_block": body_blk.id,
                                     "false_block": exit_blk.id})
            # Body
            self.builder.switch_to(body_blk)
            self._lower_block(expr.body)
            self.builder.emit(tir.OpKind.BR,
                              attrs={"target_block": header_blk.id})
            # Exit
            self.builder.switch_to(exit_blk)
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
            v = self._lower_expr(expr.value)
            if v is None:
                v = self.builder.const_int(0)
            # If target is a mutable variable name, emit STORE_VAR.
            # Compound assignments (+=, etc.) need a load+op+store.
            if isinstance(expr.target, A.Name) and self._lookup_mut(expr.target.name):
                if expr.op == "=":
                    self.builder.emit(tir.OpKind.STORE_VAR, v,
                                      attrs={"name": expr.target.name})
                else:
                    # Compound: load, op, store
                    op_map = {
                        "+=": tir.OpKind.ADD, "-=": tir.OpKind.SUB,
                        "*=": tir.OpKind.MUL, "/=": tir.OpKind.DIV,
                        "%=": tir.OpKind.MOD,
                    }
                    cur = self.builder.emit(tir.OpKind.LOAD_VAR,
                                            result_ty=v.ty,
                                            attrs={"name": expr.target.name})
                    new = self.builder.emit(op_map[expr.op], cur, v,
                                            result_ty=v.ty)
                    self.builder.emit(tir.OpKind.STORE_VAR, new,
                                      attrs={"name": expr.target.name})
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
