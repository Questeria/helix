"""
helixc/ir/lower_ast.py — lower Helix AST into Tensor IR.

For v0.1 we lower the *scalar/control-flow* subset directly: function decls,
arithmetic on primitive types, if/else, while, calls, returns.

Tensor and tile operations are recognized but emitted as opaque CALL ops
(real lowering rules come in v0.2 once we wire up the linalg-style
structured ops).

License: Apache 2.0
"""

from __future__ import annotations

from typing import Optional

from ..frontend import ast_nodes as A
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
        # (LOAD_VAR/STORE_VAR ops). Source name -> (IR name, type). The IR
        # name is mangled (e.g. "x__1") when a shadowing inner `let mut`
        # would otherwise collide with an outer binding's stack slot in
        # the codegen var_slots table.
        self.mut_scope: list[dict[str, tuple[str, tir.TIRType]]] = []
        # Monotonic counter per source name for shadow-disambiguation.
        self._mut_shadow_counter: dict[str, int] = {}
        # Arrays: name -> (elem_ty, length)
        self.array_scope: list[dict[str, tuple[tir.TIRType, int]]] = []
        # Structs: binding-name -> struct-decl-name (so we can resolve
        # `p.field_x` to a LOAD_ELEM at the correct field index).
        self.struct_scope: list[dict[str, str]] = []
        # Recursive enums: binding-name -> enum-decl-name. The binding
        # holds a scalar i32 (arena index); Index(Name, k) on it emits
        # ARENA_GET(arena_index + k) — the dispatch primitive for
        # recursive-enum match.
        self.rec_enum_scope: list[dict[str, str]] = []
        # struct-decl-name -> ordered list of field names (declaration order).
        # Built from prog.items at lower-time. Used for the flat (single-
        # level) struct case where every field is i32.
        self._struct_fields: dict[str, list[str]] = {}
        # Same struct decls but with nested-struct fields recursively
        # flattened into dot-paths. For `Outer { count: i32, inner: Inner }`
        # with `Inner { value: i32 }`, this becomes:
        #   {"Outer": [("count",), ("inner", "value")], "Inner": [("value",)]}.
        # Used for nested StructLit + chained Field access.
        self._struct_flat_paths: dict[str, list[tuple[str, ...]]] = {}
        # enum-decl-name -> {variant-name: index}.
        self._enum_variants: dict[str, dict[str, int]] = {}
        # name -> FnIR (registered functions)
        self.functions: dict[str, tir.FnIR] = {}
        # quote-handle assignment table: maps AST pretty-form -> unique cell
        # index in [0, HELIX_NUM_CELLS). Sequential allocation prevents
        # silent hash collisions; two quote() sites with the same AST share
        # a cell, but two distinct ASTs always get distinct cells (until
        # we run out and raise).
        self._quote_handle_table: dict[str, int] = {}

    # ---- entry ----
    def lower(self) -> tir.Module:
        # Pass 0: index struct + enum decls so StructLit / Field / Path
        # can resolve.
        for item in self.prog.items:
            if isinstance(item, A.StructDecl):
                self._struct_fields[item.name] = [p.name for p in item.fields]
            elif isinstance(item, A.EnumDecl):
                self._enum_variants[item.name] = {
                    v.name: i for i, v in enumerate(item.variants)
                }
        # Build flat paths for each struct. Requires all referenced sub-
        # struct decls already indexed (above).  Iterate to fixpoint to
        # handle forward references.
        struct_decls = {it.name: it for it in self.prog.items
                        if isinstance(it, A.StructDecl)}

        def _ty_struct_name(ty) -> Optional[str]:
            """Return the struct-decl name a type refers to, or None."""
            if isinstance(ty, A.TyName) and ty.name in struct_decls:
                return ty.name
            return None

        def _flat_paths_for(name: str, visiting: frozenset[str]) -> list[tuple[str, ...]]:
            if name in visiting:
                # Recursive struct without indirection: skip emitting any
                # leaf for this back-edge so the parent's slot count stays
                # accurate. Returning [(name,)] would inject a bogus path
                # like ("inner", "Outer") and corrupt the array sizing.
                return []
            decl = struct_decls.get(name)
            if decl is None:
                return [()]
            paths: list[tuple[str, ...]] = []
            for p in decl.fields:
                ty = p.ty
                # Detect a nested struct field by checking if the type
                # resolves to another known struct decl name.
                sub_name = _ty_struct_name(ty)
                if sub_name is not None and sub_name in struct_decls:
                    sub_paths = _flat_paths_for(sub_name, visiting | {name})
                    for sp in sub_paths:
                        paths.append((p.name,) + sp)
                else:
                    paths.append((p.name,))
            return paths

        for name in struct_decls:
            self._struct_flat_paths[name] = _flat_paths_for(
                name, frozenset())

        # Detect recursive enum types: any enum where a variant payload
        # references the enum itself (directly or transitively). Recursive
        # enums use arena-indirection storage (a value is the i32 arena
        # index of [tag, payload0, payload1, ...]) instead of flat-array
        # storage. This sidesteps the unbounded slot-count problem.
        enum_decls = {it.name: it for it in self.prog.items
                      if isinstance(it, A.EnumDecl)}

        def _enum_references(name: str, target: str,
                             visiting: frozenset[str]) -> bool:
            if name == target and visiting:
                return True
            if name in visiting:
                return False
            decl = enum_decls.get(name)
            if decl is None:
                return False
            for v in decl.variants:
                for pty in v.payload_tys:
                    if isinstance(pty, A.TyName) and pty.name in enum_decls:
                        if _enum_references(pty.name, target,
                                            visiting | {name}):
                            return True
            return False

        self._recursive_enums: set[str] = set()
        for ename in enum_decls:
            if _enum_references(ename, ename, frozenset()):
                self._recursive_enums.add(ename)

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
        self.array_scope.append({})
        self.struct_scope.append({})
        self.rec_enum_scope.append({})
    def _pop_scope(self) -> None:
        self.scope.pop()
        self.mut_scope.pop()
        self.array_scope.pop()
        self.struct_scope.pop()
        self.rec_enum_scope.pop()
    def _bind_rec_enum(self, name: str, enum_name: str) -> None:
        self.rec_enum_scope[-1][name] = enum_name
    def _lookup_rec_enum(self, name: str) -> Optional[str]:
        for sc in reversed(self.rec_enum_scope):
            if name in sc:
                return sc[name]
        return None
    def _bind(self, name: str, v: tir.Value) -> None:
        self.scope[-1][name] = v
    def _bind_mut(self, name: str, ty: tir.TIRType) -> str:
        """Bind `name` as mutable; return the unique IR-level name to use
        in ALLOC_VAR/STORE_VAR/LOAD_VAR attrs. If any outer scope already
        has a mut binding with this source name, mangle to a fresh name
        so the codegen's name->slot table doesn't alias them."""
        already_bound = any(name in sc for sc in self.mut_scope)
        if already_bound:
            self._mut_shadow_counter[name] = self._mut_shadow_counter.get(name, 0) + 1
            ir_name = f"{name}__{self._mut_shadow_counter[name]}"
        else:
            ir_name = name
        self.mut_scope[-1][name] = (ir_name, ty)
        return ir_name
    def _bind_array(self, name: str, elem_ty: tir.TIRType, length: int) -> None:
        self.array_scope[-1][name] = (elem_ty, length)
    def _bind_struct(self, binding_name: str, struct_name: str) -> None:
        self.struct_scope[-1][binding_name] = struct_name
    def _lookup_struct(self, name: str) -> Optional[str]:
        for sc in reversed(self.struct_scope):
            if name in sc:
                return sc[name]
        return None

    def _resolve_path_value(self, slit: "A.StructLit",
                              path: tuple[str, ...]) -> "Optional[tir.Value]":
        """Resolve a flat path through a (possibly-nested) StructLit, with
        a fallback to LOAD_ELEM from any Name binding the walk encounters
        mid-path. Returns the lowered value or None if the path can't be
        resolved."""
        cur: A.Expr = slit
        consumed = 0
        for seg in path:
            if isinstance(cur, A.StructLit):
                found = None
                for fname, fexpr in cur.fields:
                    if fname == seg:
                        found = fexpr
                        break
                if found is None:
                    return None
                cur = found
                consumed += 1
                continue
            if isinstance(cur, A.Name):
                # Remaining path segments must be resolved via the Name's
                # struct binding's flat-path table.
                struct_name = self._lookup_struct(cur.name)
                if struct_name is None:
                    return None
                src_paths = self._struct_flat_paths.get(struct_name, [])
                remaining = path[consumed:]
                try:
                    idx_int = src_paths.index(remaining)
                except ValueError:
                    return None
                arr = self._lookup_array(cur.name)
                if arr is None:
                    return None
                elem_ty, _ = arr
                idx_v = self.builder.const_int(idx_int)
                return self.builder.emit(
                    tir.OpKind.LOAD_ELEM, idx_v,
                    result_ty=elem_ty,
                    attrs={"name": cur.name})
            return None
        # Reached the leaf via path traversal of nested StructLits.
        return self._lower_expr(cur)

    def _aggregate_slot_count(self, ty: "A.TyNode") -> Optional[int]:
        """If `ty` is a struct or enum decl name, return the number of
        i32 slots it occupies (struct: flat field count; enum: max
        payload count + 1 for tag, OR 1 if recursive — arena index).
        None for non-aggregate types."""
        if not isinstance(ty, A.TyName):
            return None
        # Struct: flat-path length already encodes nested-struct flattening.
        flat = self._struct_flat_paths.get(ty.name)
        if flat is not None:
            return len(flat)
        # Recursive enum: the value is a single i32 arena index.
        if ty.name in getattr(self, "_recursive_enums", set()):
            return 1
        # Non-recursive enum: tag (1) + max payload arity across variants.
        if ty.name in self._enum_variants:
            decl = next(
                (it for it in self.prog.items
                 if isinstance(it, A.EnumDecl) and it.name == ty.name),
                None,
            )
            if decl is None:
                return 1
            max_payload = max(
                (len(v.payload_tys) for v in decl.variants),
                default=0,
            )
            return 1 + max_payload
        return None
    def _lookup(self, name: str) -> Optional[tir.Value]:
        for sc in reversed(self.scope):
            if name in sc:
                return sc[name]
        return None
    def _lookup_mut(self, name: str) -> Optional[tir.TIRType]:
        """Return the type of the closest enclosing mut binding, or None."""
        for sc in reversed(self.mut_scope):
            if name in sc:
                return sc[name][1]
        return None

    def _lookup_mut_ir_name(self, name: str) -> Optional[str]:
        """Return the IR-level name for the closest enclosing mut binding."""
        for sc in reversed(self.mut_scope):
            if name in sc:
                return sc[name][0]
        return None
    def _lookup_array(self, name: str):
        for sc in reversed(self.array_scope):
            if name in sc:
                return sc[name]
        return None

    # ---- type lowering ----
    # Built-in primitive type names — anything else under TyName is
    # either a struct, enum, or a generic type parameter. Generics
    # silently lower to TIRScalar("T") with i32-sized ABI today; this
    # is correct for i32 type args and silently wrong for i64+.
    # Documented as HBS limitation (audit-2-deep-research bug G).
    _PRIMITIVE_TYPE_NAMES = frozenset({
        "i8", "i16", "i32", "i64", "isize",
        "u8", "u16", "u32", "u64", "usize",
        "bool", "char",
        "bf16", "f16", "f32", "f64",
        "unit",
    })

    def _lower_type(self, ty: A.TyNode) -> tir.TIRType:
        if isinstance(ty, A.TyName):
            # Recognize struct / enum / primitive names. Anything else is
            # likely a generic type parameter (e.g. T in `fn id[T](x: T)`).
            # Generic type params lower to TIRScalar(name) which defaults
            # to i32-sized ABI — works for i32 type args, silently wrong
            # otherwise. Documented HBS limitation.
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
        # Build the IR-level param list. AGGREGATE-typed AST params (struct
        # or enum) expand to N i32 IR params, where N is the slot count.
        # The callee uses param-name suffixed with "__slot{i}" to
        # distinguish; reassembly into an array binding happens in
        # _lower_fn_body.
        params: list[tuple[str, tir.TIRType]] = []
        for p in fn.params:
            # Recursive enum: single i32 arena index (no slot expansion).
            if (isinstance(p.ty, A.TyName)
                    and p.ty.name in self._recursive_enums):
                params.append((p.name, tir.TIRScalar("i32")))
                continue
            n_slots = self._aggregate_slot_count(p.ty)
            # Aggregate-typed params always go through the multi-slot
            # path, even single-field ones — the body uses field-access
            # syntax which expects an array binding.
            if n_slots is not None and n_slots >= 1:
                for i in range(n_slots):
                    params.append((f"{p.name}__slot{i}",
                                   tir.TIRScalar("i32")))
            else:
                t = self._lower_type(p.ty)
                params.append((p.name, t))
        ret = self._lower_type(fn.return_ty) if fn.return_ty else tir.TIRUnit()
        attrs: dict[str, object] = {}
        for a in fn.attrs:
            attrs[a] = True
        if fn.is_pub:
            attrs["is_pub"] = True
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
        # Bind params to their SSA values. AGGREGATE-typed params were
        # expanded to N consecutive IR params in _register_fn — reassemble
        # them into an array binding here so the body can use field/index
        # access transparently.
        ir_param_idx = 0
        for p in fn.params:
            # Recursive-enum-typed param: scalar i32 arena index. Bind as
            # scalar + register in rec_enum scope so Index access emits
            # ARENA_GET against it.
            if (isinstance(p.ty, A.TyName)
                    and p.ty.name in self._recursive_enums):
                v = ir_fn.params[ir_param_idx]
                ir_param_idx += 1
                self._bind(p.name, v)
                self._bind_rec_enum(p.name, p.ty.name)
                continue
            n_slots = self._aggregate_slot_count(p.ty)
            if n_slots is not None and n_slots >= 1:
                # Take next n_slots IR params, allocate array, store each.
                slot_vals = list(ir_fn.params[ir_param_idx:
                                              ir_param_idx + n_slots])
                ir_param_idx += n_slots
                elem_ty = tir.TIRScalar("i32")
                self.builder.emit(tir.OpKind.ALLOC_ARRAY,
                                  attrs={"name": p.name,
                                         "dtype": elem_ty,
                                         "length": n_slots})
                for i, sv in enumerate(slot_vals):
                    idx = self.builder.const_int(i)
                    self.builder.emit(tir.OpKind.STORE_ELEM, idx, sv,
                                      attrs={"name": p.name})
                self._bind_array(p.name, elem_ty, n_slots)
                if isinstance(p.ty, A.TyName) \
                        and p.ty.name in self._struct_flat_paths:
                    self._bind_struct(p.name, p.ty.name)
            else:
                v = ir_fn.params[ir_param_idx]
                ir_param_idx += 1
                self._bind(p.name, v)
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
            # Phase 0: f64/f16/bf16 as a scalar `let` binding silently
            # demote to f32 in the IR (FloatLit defaults to f32) and the
            # x86_64 backend can't emit movsd/F16C. Reject here so users
            # get a clear error rather than silent corruption. Tile
            # dtypes (`tile<bf16, ...>`) are still allowed because the
            # ptx backend handles them.
            if (stmt.ty is not None
                    and isinstance(stmt.ty, A.TyName)
                    and stmt.ty.name in ("f16", "bf16")):
                raise NotImplementedError(
                    f"scalar float type '{stmt.ty.name}' is not supported "
                    f"yet — f32 and f64 are implemented in the x86_64 "
                    f"backend; f16/bf16 need the F16C / AVX-512 codegen "
                    f"path."
                )
            # Special case: payload-bearing enum constructor.
            #     let m = Maybe::Some(42)
            # Allocates a [tag, payload, ...] array. The tag is the
            # variant's positional index in the enum decl. The payload
            # slots are the call args, lowered in order. For tag-only
            # variants (no args), this still allocates a 1-slot array
            # holding just the tag.
            # NOTE: tag-only enum let-values (e.g. `let m = Maybe::None;`)
            # remain bound as scalars (Path lowers to const_int(tag)).
            # When passed to a function expecting an enum-typed param,
            # the call-site multi-slot expansion handles them via the
            # generic-scalar-padding path.
            #
            # EXCEPTION: tag-only path of a RECURSIVE enum needs to be
            # arena-allocated (the binding's value is the arena index,
            # not the tag itself).
            if (stmt.value is not None
                    and isinstance(stmt.value, A.Path)
                    and len(stmt.value.segments) == 2):
                ename, vname = stmt.value.segments
                variants = self._enum_variants.get(ename)
                if (variants is not None and vname in variants
                        and ename in self._recursive_enums):
                    tag_v = self.builder.const_int(variants[vname])
                    pushed = self.builder.emit(
                        tir.OpKind.ARENA_PUSH, tag_v,
                        result_ty=tir.TIRScalar("i32"))
                    self._bind(stmt.name, pushed)
                    self._bind_rec_enum(stmt.name, ename)
                    return
            if (stmt.value is not None
                    and isinstance(stmt.value, A.Call)
                    and isinstance(stmt.value.callee, A.Path)
                    and len(stmt.value.callee.segments) == 2):
                ename, vname = stmt.value.callee.segments
                variants = self._enum_variants.get(ename)
                if variants is not None and vname in variants:
                    tag_v = self.builder.const_int(variants[vname])
                    arg_vals = []
                    for a in stmt.value.args:
                        v = self._lower_expr(a)
                        if v is None:
                            v = self.builder.const_int(0)
                        arg_vals.append(v)
                    slots = [tag_v] + arg_vals
                    # Recursive enum: arena-indirected. Push slots into
                    # the arena, bind name to the start index (scalar
                    # i32). Match dispatch will use ARENA_GET against
                    # the index.
                    if ename in self._recursive_enums:
                        start_idx = None
                        for ev in slots:
                            pushed = self.builder.emit(
                                tir.OpKind.ARENA_PUSH, ev,
                                result_ty=tir.TIRScalar("i32"))
                            if start_idx is None:
                                start_idx = pushed
                        self._bind(stmt.name, start_idx)
                        self._bind_rec_enum(stmt.name, ename)
                        return
                    elem_ty = slots[0].ty
                    n = len(slots)
                    self.builder.emit(tir.OpKind.ALLOC_ARRAY,
                                      attrs={"name": stmt.name,
                                             "dtype": elem_ty,
                                             "length": n})
                    for i, ev in enumerate(slots):
                        idx = self.builder.const_int(i)
                        self.builder.emit(tir.OpKind.STORE_ELEM, idx, ev,
                                          attrs={"name": stmt.name})
                    self._bind_array(stmt.name, elem_ty, n)
                    return
            # Special case: struct literal initializer -> back it with a
            # fixed-length stack array indexed by flat-path order.
            if stmt.value is not None and isinstance(stmt.value, A.StructLit):
                slit = stmt.value
                flat_paths = self._struct_flat_paths.get(slit.name)
                if flat_paths is None:
                    # Unknown struct (typecheck would have flagged) — fall
                    # through to default-value binding to avoid crashing.
                    self._bind(stmt.name, self.builder.const_int(0))
                    return
                # For each flat path, walk the (possibly nested) StructLit
                # value to resolve the leaf expression. If the walk hits a
                # Name (existing struct binding) before consuming the full
                # path, dereference the remaining path against the Name's
                # array binding via LOAD_ELEM. Missing fields fall back
                # to const_int(0).
                elem_vals = []
                for path in flat_paths:
                    v = self._resolve_path_value(slit, path)
                    if v is None:
                        v = self.builder.const_int(0)
                    elem_vals.append(v)
                if not elem_vals:
                    self._bind(stmt.name, self.builder.const_int(0))
                    return
                elem_ty = elem_vals[0].ty
                n = len(elem_vals)
                self.builder.emit(tir.OpKind.ALLOC_ARRAY,
                                  attrs={"name": stmt.name, "dtype": elem_ty,
                                         "length": n})
                for i, ev in enumerate(elem_vals):
                    idx = self.builder.const_int(i)
                    self.builder.emit(tir.OpKind.STORE_ELEM, idx, ev,
                                      attrs={"name": stmt.name})
                self._bind_array(stmt.name, elem_ty, n)
                self._bind_struct(stmt.name, slit.name)
                return
            # Special case: tuple literal initializer -> back it with an
            # array (positional access via `t.0`, `t.1`, ...).
            if stmt.value is not None and isinstance(stmt.value, A.TupleLit):
                elems = stmt.value.elems
                if not elems:
                    self._bind(stmt.name, self.builder.const_int(0))
                    return
                elem_vals = []
                for e in elems:
                    v = self._lower_expr(e)
                    if v is None:
                        v = self.builder.const_int(0)
                    elem_vals.append(v)
                elem_ty = elem_vals[0].ty
                n = len(elem_vals)
                self.builder.emit(tir.OpKind.ALLOC_ARRAY,
                                  attrs={"name": stmt.name, "dtype": elem_ty,
                                         "length": n})
                for i, ev in enumerate(elem_vals):
                    idx = self.builder.const_int(i)
                    self.builder.emit(tir.OpKind.STORE_ELEM, idx, ev,
                                      attrs={"name": stmt.name})
                self._bind_array(stmt.name, elem_ty, n)
                return
            # Special case: array literal initializer -> allocate stack array
            if stmt.value is not None and isinstance(stmt.value, A.ArrayLit):
                elems = stmt.value.elems
                if not elems:
                    self._bind(stmt.name, self.builder.const_int(0))
                    return
                # Lower each element (collect SSA values)
                elem_vals = []
                for e in elems:
                    v = self._lower_expr(e)
                    if v is None:
                        v = self.builder.const_int(0)
                    elem_vals.append(v)
                elem_ty = elem_vals[0].ty
                n = len(elem_vals)
                self.builder.emit(tir.OpKind.ALLOC_ARRAY,
                                  attrs={"name": stmt.name, "dtype": elem_ty,
                                         "length": n})
                for i, ev in enumerate(elem_vals):
                    idx = self.builder.const_int(i)
                    self.builder.emit(tir.OpKind.STORE_ELEM, idx, ev,
                                      attrs={"name": stmt.name})
                self._bind_array(stmt.name, elem_ty, n)
                return

            # Aliasing case: `let new = old_array_binding;` should make
            # `new` refer to the same backing array (so subsequent
            # `new[k]`, `new.field` etc. work). This is what makes match-
            # lowering's `let __scrut_N = m;` correctly preserve m's
            # tagged-value structure when m is an enum/tuple/struct.
            if (stmt.value is not None and isinstance(stmt.value, A.Name)
                    and not stmt.is_mut):
                src_name = stmt.value.name
                # Recursive-enum aliasing: just copy the scalar arena
                # index; both bindings refer to the same arena slot.
                src_rec = self._lookup_rec_enum(src_name)
                if src_rec is not None:
                    src_v = self._lookup(src_name)
                    if src_v is not None:
                        self._bind(stmt.name, src_v)
                        self._bind_rec_enum(stmt.name, src_rec)
                        return
                src_arr = self._lookup_array(src_name)
                if src_arr is not None:
                    elem_ty, length = src_arr
                    self._bind_array(stmt.name, elem_ty, length)
                    src_struct = self._lookup_struct(src_name)
                    if src_struct is not None:
                        self._bind_struct(stmt.name, src_struct)
                    # Also alias the storage at the IR level: rebind the
                    # old array name to the new name via STORE_ELEM/LOAD_
                    # ELEM aliasing. The simplest correct way is to copy
                    # element-by-element into a fresh ALLOC_ARRAY.
                    self.builder.emit(tir.OpKind.ALLOC_ARRAY,
                                      attrs={"name": stmt.name,
                                             "dtype": elem_ty,
                                             "length": length})
                    for i in range(length):
                        idx = self.builder.const_int(i)
                        loaded = self.builder.emit(
                            tir.OpKind.LOAD_ELEM, idx,
                            result_ty=elem_ty,
                            attrs={"name": src_name})
                        self.builder.emit(
                            tir.OpKind.STORE_ELEM, idx, loaded,
                            attrs={"name": stmt.name})
                    return

            v: Optional[tir.Value] = None
            if stmt.value is not None:
                v = self._lower_expr(stmt.value)
            if v is None:
                v = self.builder.const_int(0)
            if stmt.is_mut:
                # Bind first so we get the unique IR name (mangled if
                # this shadows an outer mut binding), then allocate +
                # store using that name.
                ir_name = self._bind_mut(stmt.name, v.ty)
                self.builder.emit(tir.OpKind.ALLOC_VAR,
                                  attrs={"name": ir_name, "dtype": v.ty})
                self.builder.emit(tir.OpKind.STORE_VAR, v,
                                  attrs={"name": ir_name})
            else:
                self._bind(stmt.name, v)
            return
        if isinstance(stmt, A.ExprStmt):
            self._lower_expr(stmt.expr)
            return
        if isinstance(stmt, A.ConstStmt):
            v = self._lower_expr(stmt.value)
            if v is None:
                v = self.builder.const_int(0)
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
            # Mutable variable -> emit LOAD_VAR (use mangled IR name)
            mut_ty = self._lookup_mut(expr.name)
            if mut_ty is not None:
                ir_name = self._lookup_mut_ir_name(expr.name) or expr.name
                return self.builder.emit(tir.OpKind.LOAD_VAR,
                                         result_ty=mut_ty,
                                         attrs={"name": ir_name})
            # Maybe a function reference (v0.1: emit a call-able marker)
            if expr.name in self.functions:
                return self.builder.const_int(0)
            return self.builder.const_int(0)
        if isinstance(expr, A.Path):
            # Lower `EnumName::VariantName` to const_int(variant_index).
            # Tag-only variants only — payload variants need separate
            # constructor-call lowering (TBD).
            # NOTE: a bare Path as a value-position EXPRESSION in a
            # recursive-enum context (e.g. `match ... { _ => List::Nil }`)
            # is currently lowered as the tag i32 — the let-stmt path
            # is the only place that arena-allocates. This means a bare
            # Path in expr position for a recursive enum produces just
            # the tag (not an arena index), which breaks downstream
            # match. Workaround: bind via a let first.
            segs = list(expr.segments)
            # `crate::EnumName::Variant` is a Phase-0 alias for
            # `EnumName::Variant`. Strip a leading "crate" segment so
            # the resolution below works without a real module system.
            if len(segs) >= 3 and segs[0] == "crate":
                segs = segs[1:]
            if len(segs) == 2:
                ename, vname = segs
                variants = self._enum_variants.get(ename)
                if variants is not None and vname in variants:
                    return self.builder.const_int(variants[vname])
            # 3+-segment paths that aren't an enum variant: raise rather
            # than silently lower to 0 (which used to make pattern arms
            # collide and match the wrong variant). 2-segment paths that
            # don't resolve to an enum still fall through to opaque(0).
            if len(segs) >= 3:
                raise NotImplementedError(
                    f"3+-segment path {'::'.join(expr.segments)} is not "
                    f"supported in Phase 0 (no module system). Use "
                    f"`EnumName::Variant` or `crate::EnumName::Variant`."
                )
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
            # Logical or/and — no short-circuit yet, but we DO normalize
            # the result to a strict bool (0 or 1) so downstream uses can
            # safely CMP_EQ against 1 / treat the value as a typed bool.
            #   &&: MUL (true & true = 1*1 = 1, otherwise 0)
            #   ||: ADD-then-CMP_NE-against-zero (avoids `1 || 1 = 2` polluting
            #       compares like `(a||b) == 1`)
            if expr.op == "&&":
                return self.builder.emit(tir.OpKind.MUL, l, r,
                                         result_ty=tir.TIRScalar("bool"))
            # ||: emit (l + r) != 0
            sum_ = self.builder.emit(tir.OpKind.ADD, l, r,
                                     result_ty=tir.TIRScalar("bool"))
            zero = self.builder.const_int(0)
            return self.builder.emit(tir.OpKind.CMP_NE, sum_, zero,
                                     result_ty=tir.TIRScalar("bool"))
        if isinstance(expr, A.Unary):
            inner = self._lower_expr(expr.operand)
            if inner is None:
                return None
            if expr.op == "-":
                return self.builder.emit(tir.OpKind.NEG, inner, result_ty=inner.ty)
            return inner
        if isinstance(expr, A.Call):
            # Recursive enum constructor as a value expression (i.e. NOT
            # a fn arg, but appearing as the result of a match arm body
            # or a let value in expression position). Push slots into
            # the arena and return the start index as the value.
            if (isinstance(expr.callee, A.Path)
                    and len(expr.callee.segments) == 2):
                ename, vname = expr.callee.segments
                variants = self._enum_variants.get(ename)
                if (variants is not None and vname in variants
                        and ename in self._recursive_enums):
                    tag_v = self.builder.const_int(variants[vname])
                    arg_vals = []
                    for a in expr.args:
                        v = self._lower_expr(a) or self.builder.const_int(0)
                        arg_vals.append(v)
                    start_idx = None
                    for ev in [tag_v] + arg_vals:
                        pushed = self.builder.emit(
                            tir.OpKind.ARENA_PUSH, ev,
                            result_ty=tir.TIRScalar("i32"))
                        if start_idx is None:
                            start_idx = pushed
                    return start_idx
            # Intercept print_str(string_literal) — emits a PRINT op whose
            # attr carries the literal bytes; backend writes them to stdout
            # via a write(1, ptr, len) syscall.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "print_str"
                    and len(expr.args) == 1
                    and isinstance(expr.args[0], A.StrLit)):
                s = expr.args[0].value
                return self.builder.emit(tir.OpKind.PRINT,
                                          result_ty=tir.TIRScalar("i32"),
                                          attrs={"text": s})
            # Intercept print_int(i32) — formats the value as decimal on
            # stdout. Carries the value as an SSA operand so a runtime
            # int can be printed (unlike print_str which is literal-only).
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "print_int"
                    and len(expr.args) == 1):
                v = self._lower_expr(expr.args[0])
                if v is None:
                    v = self.builder.const_int(0)
                return self.builder.emit(tir.OpKind.PRINT, v,
                                          result_ty=tir.TIRScalar("i32"),
                                          attrs={"_kind": "print_int"})
            # Arena allocator builtins — bump-alloc i32 region in the
            # binary's data section. Foundation for self-hosted compiler
            # storage of AST/IR/symbol-table.
            if isinstance(expr.callee, A.Name):
                bn = expr.callee.name
                if bn == "__arena_push" and len(expr.args) == 1:
                    v = self._lower_expr(expr.args[0]) \
                        or self.builder.const_int(0)
                    return self.builder.emit(
                        tir.OpKind.ARENA_PUSH, v,
                        result_ty=tir.TIRScalar("i32"))
                if bn == "__arena_get" and len(expr.args) == 1:
                    i = self._lower_expr(expr.args[0]) \
                        or self.builder.const_int(0)
                    return self.builder.emit(
                        tir.OpKind.ARENA_GET, i,
                        result_ty=tir.TIRScalar("i32"))
                if bn == "__arena_set" and len(expr.args) == 2:
                    i = self._lower_expr(expr.args[0]) \
                        or self.builder.const_int(0)
                    v = self._lower_expr(expr.args[1]) \
                        or self.builder.const_int(0)
                    return self.builder.emit(
                        tir.OpKind.ARENA_SET, i, v,
                        result_ty=tir.TIRScalar("i32"))
                if bn == "__arena_len" and len(expr.args) == 0:
                    return self.builder.emit(
                        tir.OpKind.ARENA_LEN,
                        result_ty=tir.TIRScalar("i32"))
                # __hash_i32(x) — FNV-1a-style hash on a single i32.
                # Used for symbol-table bucketing. Lowers to inline
                # arithmetic — pure operation, no IR op needed.
                if bn == "__hash_i32" and len(expr.args) == 1:
                    x = self._lower_expr(expr.args[0]) \
                        or self.builder.const_int(0)
                    # Quadratic mixer: h = x*x*c1 + x*c2 + c3 (mod 2^32 via
                    # signed wraparound). Without bitwise XOR/SHR in TIR,
                    # we can't do a real murmur3 finalizer; the previous
                    # `h = x*c1 + c2` was linear, so adjacent integers
                    # produced hashes differing by a fixed constant —
                    # maximally collision-prone for sequential symbol
                    # IDs (the primary use case). The quadratic form
                    # makes the difference between h(x+1) and h(x)
                    # depend on x, breaking linearity.
                    c1 = self.builder.const_int(0x05EBCA6B)
                    c2 = self.builder.const_int(0x27D4EB2F)
                    c3 = self.builder.const_int(0x165667B1)
                    i32 = tir.TIRScalar("i32")
                    x_sq = self.builder.emit(tir.OpKind.MUL, x, x, result_ty=i32)
                    quad = self.builder.emit(tir.OpKind.MUL, x_sq, c1, result_ty=i32)
                    lin = self.builder.emit(tir.OpKind.MUL, x, c2, result_ty=i32)
                    sum1 = self.builder.emit(tir.OpKind.ADD, quad, lin, result_ty=i32)
                    out = self.builder.emit(tir.OpKind.ADD, sum1, c3, result_ty=i32)
                    return out
                # String builtins on literals.
                # __strlen("literal") → compile-time const_int(len).
                if (bn == "__strlen" and len(expr.args) == 1
                        and isinstance(expr.args[0], A.StrLit)):
                    s = expr.args[0].value
                    return self.builder.const_int(len(s.encode("utf-8")))
                # __strbyte("literal", i) → runtime byte at index i.
                if (bn == "__strbyte" and len(expr.args) == 2
                        and isinstance(expr.args[0], A.StrLit)):
                    s = expr.args[0].value
                    i = self._lower_expr(expr.args[1]) \
                        or self.builder.const_int(0)
                    return self.builder.emit(
                        tir.OpKind.STR_BYTE, i,
                        result_ty=tir.TIRScalar("i32"),
                        attrs={"text": s})
                # __streq("a", "b") → compile-time const 0/1.
                if (bn == "__streq" and len(expr.args) == 2
                        and isinstance(expr.args[0], A.StrLit)
                        and isinstance(expr.args[1], A.StrLit)):
                    eq = 1 if expr.args[0].value == expr.args[1].value else 0
                    return self.builder.const_int(eq)
                # __strlit_to_arena("text") — push each byte of the literal
                # into the arena (one byte per i32 slot). Returns the start
                # slot index. Use __strlen("text") for the count.
                # Self-host lexer's source-buffer load primitive.
                if (bn == "__strlit_to_arena" and len(expr.args) == 1
                        and isinstance(expr.args[0], A.StrLit)):
                    s = expr.args[0].value
                    data = s.encode("utf-8")
                    if not data:
                        return self.builder.emit(
                            tir.OpKind.ARENA_LEN,
                            result_ty=tir.TIRScalar("i32"))
                    start_idx = None
                    for byte in data:
                        b_v = self.builder.const_int(byte)
                        pushed = self.builder.emit(
                            tir.OpKind.ARENA_PUSH, b_v,
                            result_ty=tir.TIRScalar("i32"))
                        if start_idx is None:
                            start_idx = pushed
                    return start_idx
            # Intercept write_file(path_literal, content_literal) — emits
            # a sequence of open/write/close syscalls. Returns 0 on
            # success, the negative errno on failure.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "write_file"
                    and len(expr.args) == 2
                    and isinstance(expr.args[0], A.StrLit)
                    and isinstance(expr.args[1], A.StrLit)):
                return self.builder.emit(tir.OpKind.PRINT,
                                          result_ty=tir.TIRScalar("i32"),
                                          attrs={"_kind": "write_file",
                                                  "path": expr.args[0].value,
                                                  "content": expr.args[1].value})
            # NOTE: read_file_to_arena is a builtin entry point but its
            # backend implementation is currently a stub returning the
            # file's byte count without pushing bytes into the arena.
            # The stack-allocation + byte-loop codegen has a bug that
            # caused the read syscall to silently fail. Tracked as a
            # foundation TODO (deep-research bug #4 / lexer blocker).
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "read_file_to_arena"
                    and len(expr.args) == 1
                    and isinstance(expr.args[0], A.StrLit)):
                return self.builder.emit(
                    tir.OpKind.PRINT,
                    result_ty=tir.TIRScalar("i32"),
                    attrs={"_kind": "read_file_to_arena",
                           "path": expr.args[0].value})
            # Intercept write_file_to_arena(path_literal, arena_start, n_bytes)
            # — opens the file (O_WRONLY|O_CREAT|O_TRUNC, mode 0644), writes
            # n_bytes whose values are read from arena slots
            # [arena_start .. arena_start+n_bytes) (low byte of each i32),
            # closes the fd. Returns the count of bytes successfully
            # written. Symmetric to read_file_to_arena. Required for
            # the bootstrap-stage-3 codegen to emit ELF binaries to disk.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "write_file_to_arena"
                    and len(expr.args) == 3
                    and isinstance(expr.args[0], A.StrLit)):
                arena_start = self._lower_expr(expr.args[1]) \
                    or self.builder.const_int(0)
                n_bytes = self._lower_expr(expr.args[2]) \
                    or self.builder.const_int(0)
                return self.builder.emit(
                    tir.OpKind.PRINT, arena_start, n_bytes,
                    result_ty=tir.TIRScalar("i32"),
                    attrs={"_kind": "write_file_to_arena",
                           "path": expr.args[0].value})
            # Intercept read_file_int(path_literal) — opens the file,
            # reads the first 4 bytes interpreted as i32 little-endian,
            # closes the fd. Returns the i32 value on success or 0 on
            # any error / short read.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "read_file_int"
                    and len(expr.args) == 1
                    and isinstance(expr.args[0], A.StrLit)):
                return self.builder.emit(tir.OpKind.PRINT,
                                          result_ty=tir.TIRScalar("i32"),
                                          attrs={"_kind": "read_file_int",
                                                  "path": expr.args[0].value})
            # Intercept built-in float-cell reflection ops before treating as
            # an ordinary function call.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "splice_f"
                    and len(expr.args) == 1):
                handle = self._lower_expr(expr.args[0]) or self.builder.const_int(0)
                # Emit a SPLICE op tagged with f32 result type. Codegen looks
                # at the result type to decide whether to use mov vs movss
                # for the cell read; the bit pattern in the cell is
                # interpreted as f32.
                return self.builder.emit(tir.OpKind.SPLICE, handle,
                                         result_ty=tir.TIRScalar("f32"),
                                         attrs={"value_kind": "f32"})
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "modify_f"
                    and len(expr.args) == 3):
                target_v = self._lower_expr(expr.args[0]) or self.builder.const_int(0)
                xform_v = self._lower_expr(expr.args[1]) or self.builder.const_int(0)
                attrs: dict[str, object] = {"value_kind": "f32"}
                v_arg = expr.args[2]
                if (isinstance(v_arg, A.Name)
                        and v_arg.name in self.functions
                        and self._lookup(v_arg.name) is None
                        and self._lookup_mut(v_arg.name) is None
                        and self._verifier_abi_matches_f(v_arg.name)):
                    attrs["verifier_fn"] = v_arg.name
                    placeholder = self.builder.const_int(0)
                    return self.builder.emit(tir.OpKind.MODIFY,
                                             target_v, xform_v, placeholder,
                                             result_ty=tir.TIRScalar("i32"),
                                             attrs=attrs)
                # No matching verifier — fallback to runtime-value form.
                # Drop the f32 hint so the backend doesn't emit movss from an
                # operand slot whose actual type wasn't validated as float.
                attrs.pop("value_kind", None)
                vrt = self._lower_expr(v_arg) or self.builder.const_int(0)
                return self.builder.emit(tir.OpKind.MODIFY,
                                         target_v, xform_v, vrt,
                                         result_ty=tir.TIRScalar("i32"),
                                         attrs=attrs)

            # Look up the callee's AST decl to know which params are
            # aggregate-typed (struct/enum) — those receive multi-slot
            # expansion at the call site.
            callee_ast = None
            if isinstance(expr.callee, A.Name):
                for it in self.prog.items:
                    if isinstance(it, A.FnDecl) and it.name == expr.callee.name:
                        callee_ast = it
                        break
            args: list[tir.Value] = []
            for i, a in enumerate(expr.args):
                # If the corresponding param is aggregate-typed, expand
                # into N i32 args so the callee's reassembled-array
                # binding sees the full layout. Three sub-cases:
                #   (a) arg is a Name pointing to an array binding →
                #       emit N LOAD_ELEMs (pad with 0 if source is shorter)
                #   (b) arg is a tag-only enum path (Maybe::None) →
                #       emit [const_int(tag), 0, 0, ...]
                #   (c) anything else → lower normally, pad to N slots
                expanded = False
                if callee_ast is not None and i < len(callee_ast.params):
                    p_ty = callee_ast.params[i].ty
                    # Recursive-enum-typed param: callee expects a SINGLE
                    # i32 (the arena index). For inline constructors, we
                    # must arena-push and pass the resulting index — NOT
                    # expand into flat slots like the non-recursive case.
                    if (isinstance(p_ty, A.TyName)
                            and p_ty.name in self._recursive_enums):
                        if (isinstance(a, A.Call)
                                and isinstance(a.callee, A.Path)
                                and len(a.callee.segments) == 2):
                            ename2, vname2 = a.callee.segments
                            variants2 = self._enum_variants.get(ename2)
                            if variants2 is not None and vname2 in variants2:
                                tag_v = self.builder.const_int(
                                    variants2[vname2])
                                arg_vals = [
                                    self._lower_expr(x)
                                    or self.builder.const_int(0)
                                    for x in a.args]
                                start_idx = None
                                for ev in [tag_v] + arg_vals:
                                    pushed = self.builder.emit(
                                        tir.OpKind.ARENA_PUSH, ev,
                                        result_ty=tir.TIRScalar("i32"))
                                    if start_idx is None:
                                        start_idx = pushed
                                args.append(start_idx)
                                expanded = True
                        elif (isinstance(a, A.Path)
                              and len(a.segments) == 2):
                            ename2, vname2 = a.segments
                            variants2 = self._enum_variants.get(ename2)
                            if variants2 is not None and vname2 in variants2:
                                tag_v = self.builder.const_int(
                                    variants2[vname2])
                                pushed = self.builder.emit(
                                    tir.OpKind.ARENA_PUSH, tag_v,
                                    result_ty=tir.TIRScalar("i32"))
                                args.append(pushed)
                                expanded = True
                        if not expanded:
                            # Existing rec-enum binding: pass the scalar
                            # arena index directly via _lower_expr.
                            v = self._lower_expr(a)
                            args.append(v or self.builder.const_int(0))
                            expanded = True
                        continue
                    n_slots = self._aggregate_slot_count(p_ty)
                    if n_slots is not None and n_slots >= 1:
                        # Inline enum constructor as fn arg, e.g.
                        # `f(Maybe::Some(42))`. Recognize the pattern and
                        # emit [tag, payload, ...] directly without an
                        # intermediate let-bind.
                        if (isinstance(a, A.Call)
                                and isinstance(a.callee, A.Path)
                                and len(a.callee.segments) == 2):
                            ename, vname = a.callee.segments
                            variants = self._enum_variants.get(ename)
                            if variants is not None and vname in variants:
                                args.append(self.builder.const_int(
                                    variants[vname]))
                                for arg_expr in a.args:
                                    av = self._lower_expr(arg_expr)
                                    if av is None:
                                        av = self.builder.const_int(0)
                                    args.append(av)
                                # Pad to n_slots if variant has fewer args.
                                payload_count = len(a.args)
                                for _ in range(n_slots - 1 - payload_count):
                                    args.append(self.builder.const_int(0))
                                expanded = True
                        # Inline tag-only path as fn arg, e.g. `f(Maybe::None)`.
                        if (not expanded and isinstance(a, A.Path)
                                and len(a.segments) == 2):
                            ename, vname = a.segments
                            variants = self._enum_variants.get(ename)
                            if variants is not None and vname in variants:
                                args.append(self.builder.const_int(
                                    variants[vname]))
                                for _ in range(n_slots - 1):
                                    args.append(self.builder.const_int(0))
                                expanded = True
                        if not expanded and isinstance(a, A.Name):
                            arr = self._lookup_array(a.name)
                            if arr is not None:
                                elem_ty, length = arr
                                for j in range(n_slots):
                                    if j < length:
                                        idx_v = self.builder.const_int(j)
                                        loaded = self.builder.emit(
                                            tir.OpKind.LOAD_ELEM, idx_v,
                                            result_ty=elem_ty,
                                            attrs={"name": a.name})
                                        args.append(loaded)
                                    else:
                                        args.append(self.builder.const_int(0))
                                expanded = True
                        # Field chain: passing a sub-struct of a struct.
                        # E.g. `mul_tokens(e.lhs, e.rhs)` where e.lhs is a
                        # Token (sub-struct) of e (BinExpr). Locate the
                        # field's flat-path prefix in the parent's layout
                        # and emit n_slots LOAD_ELEMs starting at that
                        # offset.
                        if not expanded and isinstance(a, A.Field):
                            base, segs = _walk_field_chain(a)
                            if base is not None:
                                struct_name = self._lookup_struct(base)
                                if struct_name is not None:
                                    parent_paths = self._struct_flat_paths.get(
                                        struct_name, [])
                                    prefix = tuple(segs)
                                    # Find indices of paths whose prefix
                                    # matches `segs`. Their suffixes form
                                    # the sub-struct's flat layout.
                                    base_idx = None
                                    for idx_int, p in enumerate(parent_paths):
                                        if p[:len(prefix)] == prefix:
                                            if base_idx is None:
                                                base_idx = idx_int
                                    if base_idx is not None:
                                        arr = self._lookup_array(base)
                                        if arr is not None:
                                            elem_ty, _ = arr
                                            for j in range(n_slots):
                                                idx_v = self.builder.const_int(
                                                    base_idx + j)
                                                loaded = self.builder.emit(
                                                    tir.OpKind.LOAD_ELEM,
                                                    idx_v,
                                                    result_ty=elem_ty,
                                                    attrs={"name": base})
                                                args.append(loaded)
                                            expanded = True
                        if not expanded:
                            # Generic case: lower the arg as a scalar,
                            # treat it as the tag (slot 0), pad the rest
                            # with zero. Covers tag-only enum paths
                            # (Maybe::None) and accidental scalar args.
                            v = self._lower_expr(a)
                            if v is None:
                                v = self.builder.const_int(0)
                            args.append(v)
                            for _ in range(n_slots - 1):
                                args.append(self.builder.const_int(0))
                            expanded = True
                if not expanded:
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
            # `loop { body }` — same skeleton as While but with no exit
            # condition (caller expected to break, which we don't yet
            # support, so this becomes effectively infinite). Without a
            # header→body→header back-edge, the body would just fall
            # through into whatever follows, which was the prior bug.
            header_blk = self.builder.new_block()
            body_blk = self.builder.new_block()
            self.builder.emit(tir.OpKind.BR, attrs={"target_block": header_blk.id})
            self.builder.switch_to(header_blk)
            self.builder.emit(tir.OpKind.BR, attrs={"target_block": body_blk.id})
            self.builder.switch_to(body_blk)
            self._lower_block(expr.body)
            self.builder.emit(tir.OpKind.BR, attrs={"target_block": header_blk.id})
            return None
        if isinstance(expr, A.Match):
            # All Match nodes should have been desugared by match_lower
            # before reaching the IR lowerer. If we hit one here, it means
            # `lower_matches` missed a position (currently it walks only
            # FnDecl bodies — Match nodes inside ConstStmt or other items
            # would slip through). Loud failure beats silent miscompile.
            raise AssertionError(
                "A.Match should not reach _lower_expr — match_lower must "
                "rewrite it to if/let chains first. Got at "
                f"{expr.span.line}:{expr.span.col}. If you've added a new "
                "AST item type that holds expressions, extend lower_matches.")
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
            # Array element assignment: arr[i] = e (or compound)
            if isinstance(expr.target, A.Index) and isinstance(expr.target.callee, A.Name):
                arr_name = expr.target.callee.name
                arr = self._lookup_array(arr_name)
                if arr is not None and len(expr.target.indices) == 1:
                    elem_ty, _ = arr
                    idx_v = self._lower_expr(expr.target.indices[0])
                    if idx_v is None:
                        idx_v = self.builder.const_int(0)
                    if expr.op == "=":
                        self.builder.emit(tir.OpKind.STORE_ELEM, idx_v, v,
                                          attrs={"name": arr_name})
                    else:
                        op_map = {
                            "+=": tir.OpKind.ADD, "-=": tir.OpKind.SUB,
                            "*=": tir.OpKind.MUL, "/=": tir.OpKind.DIV,
                            "%=": tir.OpKind.MOD,
                        }
                        cur = self.builder.emit(tir.OpKind.LOAD_ELEM, idx_v,
                                                result_ty=elem_ty,
                                                attrs={"name": arr_name})
                        new = self.builder.emit(op_map[expr.op], cur, v,
                                                result_ty=elem_ty)
                        self.builder.emit(tir.OpKind.STORE_ELEM, idx_v, new,
                                          attrs={"name": arr_name})
                    return None
            # If target is a mutable variable name, emit STORE_VAR.
            # Compound assignments (+=, etc.) need a load+op+store.
            if isinstance(expr.target, A.Name) and self._lookup_mut(expr.target.name):
                ir_name = self._lookup_mut_ir_name(expr.target.name) or expr.target.name
                if expr.op == "=":
                    self.builder.emit(tir.OpKind.STORE_VAR, v,
                                      attrs={"name": ir_name})
                else:
                    # Compound: load, op, store
                    op_map = {
                        "+=": tir.OpKind.ADD, "-=": tir.OpKind.SUB,
                        "*=": tir.OpKind.MUL, "/=": tir.OpKind.DIV,
                        "%=": tir.OpKind.MOD,
                    }
                    cur = self.builder.emit(tir.OpKind.LOAD_VAR,
                                            result_ty=v.ty,
                                            attrs={"name": ir_name})
                    new = self.builder.emit(op_map[expr.op], cur, v,
                                            result_ty=v.ty)
                    self.builder.emit(tir.OpKind.STORE_VAR, new,
                                      attrs={"name": ir_name})
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
            # If callee is a Name pointing to an array, emit LOAD_ELEM.
            if isinstance(expr.callee, A.Name):
                # Recursive-enum binding: scalar value is the arena
                # index. Index(name, k) lowers to ARENA_GET(idx + k).
                rec_enum = self._lookup_rec_enum(expr.callee.name)
                if rec_enum is not None and len(expr.indices) == 1:
                    base_idx = self._lookup(expr.callee.name)
                    if base_idx is not None:
                        offset_v = self._lower_expr(expr.indices[0])
                        if offset_v is None:
                            offset_v = self.builder.const_int(0)
                        # Compute idx + offset.
                        if (isinstance(expr.indices[0], A.IntLit)
                                and expr.indices[0].value == 0):
                            full_idx = base_idx
                        else:
                            full_idx = self.builder.emit(
                                tir.OpKind.ADD, base_idx, offset_v,
                                result_ty=tir.TIRScalar("i32"))
                        return self.builder.emit(
                            tir.OpKind.ARENA_GET, full_idx,
                            result_ty=tir.TIRScalar("i32"))
                arr = self._lookup_array(expr.callee.name)
                if arr is not None and len(expr.indices) == 1:
                    elem_ty, _length = arr
                    idx_v = self._lower_expr(expr.indices[0])
                    if idx_v is None:
                        idx_v = self.builder.const_int(0)
                    return self.builder.emit(tir.OpKind.LOAD_ELEM, idx_v,
                                             result_ty=elem_ty,
                                             attrs={"name": expr.callee.name})
                # Fallback for tag-only enum scrutinees: if the binding is
                # scalar (not array) and the requested index is 0, return
                # the scalar directly — for a tag-only enum binding the
                # whole value IS the tag. This makes match_lower's
                # `__scrut[0] == ...` test work uniformly.
                scalar_v = self._lookup(expr.callee.name)
                if (scalar_v is not None and len(expr.indices) == 1
                        and isinstance(expr.indices[0], A.IntLit)
                        and expr.indices[0].value == 0):
                    return scalar_v
            # Fallback: opaque
            self._lower_expr(expr.callee)
            for i in expr.indices:
                self._lower_expr(i)
            return None
        if isinstance(expr, A.Field):
            # Inline tuple/struct field access: `(1, 2, 3).1` — no Name
            # base, so _walk_field_chain wouldn't resolve. Lower the
            # tuple's elements directly and pick the indexed value.
            if (isinstance(expr.obj, A.TupleLit)
                    and expr.name.isdigit()):
                idx = int(expr.name)
                if 0 <= idx < len(expr.obj.elems):
                    return self._lower_expr(expr.obj.elems[idx])
            # Struct field access. May be a chain: o.inner.value. Walk the
            # Field-of-Field chain to find the base Name, accumulate path
            # segments, then look up the path in the flat-path table.
            base_name, path_segs = _walk_field_chain(expr)
            if base_name is not None:
                struct_name = self._lookup_struct(base_name)
                if struct_name is not None:
                    flat_paths = self._struct_flat_paths.get(struct_name, [])
                    target = tuple(path_segs)
                    try:
                        idx_int = flat_paths.index(target)
                    except ValueError:
                        idx_int = -1
                    if idx_int >= 0:
                        arr = self._lookup_array(base_name)
                        if arr is not None:
                            elem_ty, _ = arr
                            idx_v = self.builder.const_int(idx_int)
                            return self.builder.emit(
                                tir.OpKind.LOAD_ELEM, idx_v,
                                result_ty=elem_ty,
                                attrs={"name": base_name})
                # Tuple field access: `t.0`, `t.1`. Single-segment digit
                # name on an array binding lowers directly to LOAD_ELEM.
                if (len(path_segs) == 1 and path_segs[0].isdigit()
                        and struct_name is None):
                    arr = self._lookup_array(base_name)
                    if arr is not None:
                        elem_ty, length = arr
                        idx_int = int(path_segs[0])
                        if 0 <= idx_int < length:
                            idx_v = self.builder.const_int(idx_int)
                            return self.builder.emit(
                                tir.OpKind.LOAD_ELEM, idx_v,
                                result_ty=elem_ty,
                                attrs={"name": base_name})
            self._lower_expr(expr.obj)
            return None
        if isinstance(expr, A.Cast):
            inner = self._lower_expr(expr.value)
            if inner is None:
                inner = self.builder.const_int(0)
            target = self._lower_type(expr.target_ty)
            return self.builder.emit(tir.OpKind.CAST, inner,
                                     result_ty=target,
                                     attrs={"from_ty": inner.ty,
                                            "to_ty": target})

        # AGI primitives
        if isinstance(expr, A.Quote):
            # quote { ... } captures the inner AST as a constant value of
            # type AstNode. The handle is a stable cell index assigned by
            # this lowerer — alpha-equivalent ASTs share a cell (via
            # structural_hash), distinct shapes get distinct cells (no
            # collision aliasing). Backed by HELIX_NUM_CELLS mutable cells
            # in the binary.
            from ..backend.x86_64 import HELIX_NUM_CELLS
            from ..frontend.ast_hash import structural_hash
            try:
                key = structural_hash(expr.inner)
            except Exception:
                # Fall back to pretty-string if hashing fails for any reason.
                key = _pretty(expr.inner)
            if key not in self._quote_handle_table:
                idx = len(self._quote_handle_table)
                if idx >= HELIX_NUM_CELLS:
                    raise ValueError(
                        f"too many distinct quote() expressions: limit is "
                        f"{HELIX_NUM_CELLS}; this AST is the {idx + 1}-th "
                        f"distinct one"
                    )
                self._quote_handle_table[key] = idx
            ast_handle = self._quote_handle_table[key]
            return self.builder.emit(tir.OpKind.QUOTE,
                                     result_ty=tir.TIRScalar("i64"),
                                     attrs={"ast_handle": ast_handle,
                                            "ast_pretty": _pretty(expr.inner)})
        if isinstance(expr, A.Splice):
            inner = self._lower_expr(expr.inner)
            if inner is None:
                inner = self.builder.const_int(0)
            return self.builder.emit(tir.OpKind.SPLICE, inner,
                                     result_ty=tir.TIRScalar("i64"))
        if isinstance(expr, A.Modify):
            target = self._lower_expr(expr.target) or self.builder.const_int(0)
            xform = self._lower_expr(expr.transformation) or self.builder.const_int(0)
            attrs: dict[str, object] = {}
            # If the verifier expression is a Name pointing at a known function
            # (NOT a local variable), resolve it to a compile-time call: the
            # backend emits a direct call to the verifier before applying the
            # modification. Otherwise fall back to the legacy "is the value
            # nonzero?" form so existing dynamic-verifier tests keep working.
            if (isinstance(expr.verifier, A.Name)
                    and expr.verifier.name in self.functions
                    and self._lookup(expr.verifier.name) is None
                    and self._lookup_mut(expr.verifier.name) is None
                    and self._verifier_abi_matches(expr.verifier.name)):
                attrs["verifier_fn"] = expr.verifier.name
                verifier_v = self.builder.const_int(0)
                return self.builder.emit(tir.OpKind.MODIFY, target, xform,
                                         verifier_v,
                                         result_ty=tir.TIRScalar("i32"),
                                         attrs=attrs)
            verifier = self._lower_expr(expr.verifier) or self.builder.const_int(0)
            return self.builder.emit(tir.OpKind.MODIFY, target, xform, verifier,
                                     result_ty=tir.TIRScalar("i32"))
        return None

    def _hash_ast(self, node: A.Expr) -> int:
        """Compute a stable hash over an AST structure for QUOTE handles.
        For v0.1 we use Python's hash of a stringified form."""
        return abs(hash(_pretty(node))) & 0x7FFFFFFF

    def _verifier_abi_matches_f(self, fn_name: str) -> bool:
        """A modify_f verifier takes (handle: i32, val: f32) and returns
        i32/bool. System V passes the f32 in xmm0 and the int in edi."""
        ir_fn = self.functions.get(fn_name)
        if ir_fn is None or len(ir_fn.params) != 2:
            return False
        p0, p1 = ir_fn.params
        if not (isinstance(p0.ty, tir.TIRScalar) and p0.ty.name == "i32"):
            return False
        if not (isinstance(p1.ty, tir.TIRScalar) and p1.ty.name == "f32"):
            return False
        if not (isinstance(ir_fn.return_ty, tir.TIRScalar)
                and ir_fn.return_ty.name in ("i32", "bool")):
            raise ValueError(
                f"verifier function {fn_name!r} has parameters (i32, f32) "
                f"but returns {ir_fn.return_ty!r} — verifiers must return "
                f"i32 or bool"
            )
        return True

    def _verifier_abi_matches(self, fn_name: str) -> bool:
        """A verifier function must take exactly two i32 params and return
        an integer. Otherwise the System V int-register call convention
        used by MODIFY's call to the verifier doesn't apply (e.g. f32 args
        would land in xmm0/xmm1 instead of edi/esi).

        When the ABI looks "almost right" (function exists, has 2 i32
        params, but the return type is wrong — e.g. unit/void), we raise
        an explicit compile error rather than silently routing to the
        legacy fallback (which would always reject every modify, with no
        diagnostic). When the ABI is clearly different (wrong arity or
        non-i32 params), we silently fall back; that case is plausibly
        the user passing a runtime expression that happens to have the
        same name as a function.
        """
        ir_fn = self.functions.get(fn_name)
        if ir_fn is None or len(ir_fn.params) != 2:
            return False
        for p in ir_fn.params:
            if not (isinstance(p.ty, tir.TIRScalar) and p.ty.name == "i32"):
                return False
        # Two i32 params — looks like a verifier. Now require an integer-
        # like return type or raise a clear error.
        if not (isinstance(ir_fn.return_ty, tir.TIRScalar)
                and ir_fn.return_ty.name in ("i32", "bool")):
            ret_str = (ir_fn.return_ty.name
                       if isinstance(ir_fn.return_ty, tir.TIRScalar)
                       else type(ir_fn.return_ty).__name__)
            raise ValueError(
                f"verifier function {fn_name!r} has parameters (i32, i32) "
                f"but returns {ret_str!r} — verifiers must return i32 or "
                f"bool (1=accept, 0=reject)"
            )
        return True


def _pretty(node: A.Expr | A.Block) -> str:
    """Best-effort textual form of an AST node — used for stable hashing of
    `quote { ... }` capture. Recursive but avoids cycles by class+attr scan."""
    if isinstance(node, A.IntLit):
        return f"int({node.value})"
    if isinstance(node, A.FloatLit):
        return f"float({node.value})"
    if isinstance(node, A.BoolLit):
        return f"bool({node.value})"
    if isinstance(node, A.Name):
        return f"name({node.name})"
    if isinstance(node, A.Binary):
        return f"binary({node.op},{_pretty(node.left)},{_pretty(node.right)})"
    if isinstance(node, A.Unary):
        return f"unary({node.op},{_pretty(node.operand)})"
    if isinstance(node, A.Call):
        args = ",".join(_pretty(a) for a in node.args)
        return f"call({_pretty(node.callee)},[{args}])"
    if isinstance(node, A.If):
        return f"if({_pretty(node.cond)},{_pretty(node.then)})"
    if isinstance(node, A.Block):
        stmts = ";".join(_pretty(s.expr) if isinstance(s, A.ExprStmt) else "<stmt>"
                         for s in node.stmts)
        last = _pretty(node.final_expr) if node.final_expr else ""
        return f"block({stmts};{last})"
    return f"<{type(node).__name__}>"


def _walk_field_chain(field: A.Field) -> tuple[Optional[str], list[str]]:
    """Walk a chain like `Name.f1.f2.f3` and return (base_name, [f1, f2, f3]).
    Returns (None, []) if the chain isn't rooted at a Name."""
    segs: list[str] = []
    cur: A.Expr = field
    while isinstance(cur, A.Field):
        segs.append(cur.name)
        cur = cur.obj
    if isinstance(cur, A.Name):
        segs.reverse()
        return cur.name, segs
    return None, []


def _resolve_struct_leaf(slit: A.StructLit, path: tuple[str, ...]) -> Optional[A.Expr]:
    """Walk a (possibly-nested) StructLit by a dot-path, return the leaf
    expr at that path or None if the path doesn't fully resolve."""
    cur: A.Expr = slit
    for seg in path:
        if not isinstance(cur, A.StructLit):
            return None
        found = None
        for fname, fexpr in cur.fields:
            if fname == seg:
                found = fexpr
                break
        if found is None:
            return None
        cur = found
    return cur


def lower(prog: A.Program) -> tir.Module:
    # Pre-pass: rewrite `match` expressions into nested if/let chains so
    # the rest of the pipeline (IR lowering, autodiff, x86 backend) is
    # match-agnostic.
    from ..frontend.match_lower import lower_matches
    lower_matches(prog)
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
