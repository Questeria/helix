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
    _GPU_INDEX_BUILTINS = frozenset({
        "thread_idx", "thread_idx_x", "thread_idx_y", "thread_idx_z",
        "block_idx", "block_idx_x", "block_idx_y", "block_idx_z",
        "block_dim", "block_dim_x", "block_dim_y", "block_dim_z",
    })

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
        # Heterogeneous aggregates: name -> typed slot SSA values. Homogeneous
        # structs/enums still use array_scope for the older backend path.
        self.aggregate_scope: list[dict[str, list[tir.Value]]] = []
        # Structs: binding-name -> struct-decl-name (so we can resolve
        # `p.field_x` to a LOAD_ELEM at the correct field index).
        self.struct_scope: list[dict[str, str]] = []
        self.enum_scope: list[dict[str, str]] = []
        # Recursive enums: binding-name -> enum-decl-name. The binding
        # holds a scalar i32 (arena index); Index(Name, k) on it emits
        # ARENA_GET(arena_index + k) — the dispatch primitive for
        # recursive-enum match.
        self.rec_enum_scope: list[dict[str, str]] = []
        # Stage 15 — Tiles: binding-name -> (rows, cols). Stack-allocated as
        # ALLOC_ARRAY of rows*cols f32 elements. tile_matmul produces a new
        # tile binding with computed shape; .get(row, col) lowers to
        # LOAD_ELEM at index row*cols + col.
        self.tile_scope: list[dict[str, tuple[int, int]]] = []
        # Stage 16 — HBM tile kernel params: binding-name -> (dtype-name, length,
        # param_idx). These are kernel fn parameters typed `tile<f32, [N], HBM>`.
        # `a[i]` and `a[i] = v` on them lower to TILE_INDEX_LOAD/STORE TIR ops,
        # which the PTX backend turns into `ld.global.f32` / `st.global.f32`.
        # x86 backend never sees these (kernel bodies are PTX-only).
        self.hbm_tile_scope: list[dict[str, tuple[str, int, int]]] = []
        # Stage 16 — inside-kernel flag. Set in _lower_fn_body for fns with
        # @kernel attribute, so the thread_idx() builtin only resolves there.
        self._in_kernel: bool = False
        # Audit 28.8 cycle 2 C2-2 — inside-traced-fn flag. Set in
        # _lower_fn_body for fns with @trace attribute. Read by the
        # A.Return arm of _lower_expr so an explicit early `return X`
        # emits TRACE_EXIT before the IR `ret` op, mirroring the
        # fall-through-return path. Without this, traced fns with
        # explicit returns produced TRACE_ENTRY-without-EXIT pairs
        # on the early-return path — invisible at Phase-0 because the
        # backend stubs the ops as nops, but the moment a runtime exists
        # the trace stream would be permanently corrupted.
        self._is_fn_traced: bool = False
        # Name of the currently-being-lowered fn (used as TRACE_EXIT
        # attr). None when not inside a fn body.
        self._current_fn_name: str | None = None
        self._current_expected_rec_enum: str | None = None
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
        self._struct_flat_slot_types: dict[str, list[tir.TIRType]] = {}
        # enum-decl-name -> {variant-name: index}.
        self._enum_variants: dict[str, dict[str, int]] = {}
        # Stage 31: type aliases, including refined aliases, erase to their
        # target type before IR so backend scalar names stay primitive.
        self._type_aliases: dict[str, A.TypeAlias] = {}
        self._lowering_type_aliases: set[str] = set()
        self._const_scalar_values: dict[str, int | float | bool] = {}
        self._const_scalar_types: dict[str, tir.TIRType] = {}
        self._const_fn_aliases: dict[str, str] = {}
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
            elif isinstance(item, A.TypeAlias):
                self._type_aliases[item.name] = item
        self._index_const_values()
        # Build flat paths for each struct. Requires all referenced sub-
        # struct decls already indexed (above).  Iterate to fixpoint to
        # handle forward references.
        struct_decls = {it.name: it for it in self.prog.items
                        if isinstance(it, A.StructDecl)}

        def _ty_struct_name(ty) -> Optional[str]:
            """Return the struct-decl name a type refers to, or None."""
            ty = self._resolve_type_alias_node(ty)
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

        def _field_type_for_path(name: str, path: tuple[str, ...]) -> A.TyNode:
            cur_name = name
            leaf_ty: A.TyNode = A.TyName(A.Span(0, 0), "i32")
            for seg in path:
                decl = struct_decls.get(cur_name)
                if decl is None:
                    break
                field = next((p for p in decl.fields if p.name == seg), None)
                if field is None:
                    break
                leaf_ty = field.ty
                sub_name = _ty_struct_name(field.ty)
                if sub_name is not None:
                    cur_name = sub_name
            return leaf_ty

        for name in struct_decls:
            self._struct_flat_paths[name] = _flat_paths_for(
                name, frozenset())
            self._struct_flat_slot_types[name] = [
                self._lower_type(_field_type_for_path(name, path))
                for path in self._struct_flat_paths[name]
            ]

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
                    pty = self._resolve_type_alias_node(pty)
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
        self.aggregate_scope.append({})
        self.struct_scope.append({})
        self.enum_scope.append({})
        self.rec_enum_scope.append({})
        self.tile_scope.append({})
        self.hbm_tile_scope.append({})
    def _pop_scope(self) -> None:
        self.scope.pop()
        self.mut_scope.pop()
        self.array_scope.pop()
        self.aggregate_scope.pop()
        self.struct_scope.pop()
        self.enum_scope.pop()
        self.rec_enum_scope.pop()
        self.tile_scope.pop()
        self.hbm_tile_scope.pop()
    def _bind_hbm_tile(self, name: str, dtype: str, length: int, param_idx: int) -> None:
        self.hbm_tile_scope[-1][name] = (dtype, length, param_idx)
    def _lookup_hbm_tile(self, name: str):
        for sc in reversed(self.hbm_tile_scope):
            if name in sc:
                return sc[name]
        return None
    def _bind_tile(self, name: str, rows: int, cols: int) -> None:
        self.tile_scope[-1][name] = (rows, cols)
    def _lookup_tile(self, name: str):
        for sc in reversed(self.tile_scope):
            if name in sc:
                return sc[name]
        return None
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
    def _bind_aggregate(self, name: str, values: list[tir.Value]) -> None:
        self.aggregate_scope[-1][name] = values
    def _lookup_aggregate(self, name: str) -> Optional[list[tir.Value]]:
        for sc in reversed(self.aggregate_scope):
            if name in sc:
                return sc[name]
        return None
    def _bind_struct(self, binding_name: str, struct_name: str) -> None:
        self.struct_scope[-1][binding_name] = struct_name
    def _bind_enum(self, binding_name: str, enum_name: str) -> None:
        self.enum_scope[-1][binding_name] = enum_name
    def _lookup_enum(self, name: str) -> Optional[str]:
        for sc in reversed(self.enum_scope):
            if name in sc:
                return sc[name]
        return None
    def _lookup_struct(self, name: str) -> Optional[str]:
        for sc in reversed(self.struct_scope):
            if name in sc:
                return sc[name]
        return None

    def _enum_variant_for_expr(
        self, expr: A.Expr,
    ) -> Optional[tuple[str, str, int]]:
        """Resolve `Enum::Variant` and flattened `mod__Enum__Variant`."""
        ename: Optional[str] = None
        vname: Optional[str] = None
        if isinstance(expr, A.Path):
            segs = list(expr.segments)
            if len(segs) >= 3 and segs[0] == "crate":
                segs = segs[1:]
            if len(segs) == 2:
                ename, vname = segs
        elif isinstance(expr, A.Name) and "__" in expr.name:
            parts = expr.name.split("__")
            for i in range(len(parts) - 1, 0, -1):
                candidate = "__".join(parts[:i])
                variants = self._enum_variants.get(candidate)
                if variants is None:
                    continue
                tail = "__".join(parts[i:])
                if tail in variants:
                    return candidate, tail, variants[tail]
        if ename is None or vname is None:
            return None
        variants = self._enum_variants.get(ename)
        if variants is None or vname not in variants:
            return None
        return ename, vname, variants[vname]

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
                agg = self._lookup_aggregate(cur.name)
                if agg is not None:
                    if 0 <= idx_int < len(agg):
                        return agg[idx_int]
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
        ABI slots it occupies (struct: flat field count; enum: max
        payload count + 1 for tag, OR 1 if recursive — arena index).
        None for non-aggregate types."""
        slot_types = self._aggregate_slot_types(ty)
        return len(slot_types) if slot_types is not None else None

    def _aggregate_slot_types(
        self, ty: "A.TyNode",
    ) -> Optional[list[tir.TIRType]]:
        ty = self._resolve_type_alias_node(ty)
        if not isinstance(ty, A.TyName):
            return None
        # Struct: flat-path length already encodes nested-struct flattening.
        slots = self._struct_flat_slot_types.get(ty.name)
        if slots is not None:
            return slots
        # Recursive enum: the value is a single i32 arena index.
        if ty.name in getattr(self, "_recursive_enums", set()):
            return [tir.TIRScalar("i32")]
        # Non-recursive enum: tag (1) + max payload arity across variants.
        if ty.name in self._enum_variants:
            return self._enum_slot_types(ty.name)
        return None

    def _enum_slot_types(self, enum_name: str) -> list[tir.TIRType]:
        decl = next(
            (it for it in self.prog.items
             if isinstance(it, A.EnumDecl) and it.name == enum_name),
            None,
        )
        if decl is None:
            return [tir.TIRScalar("i32")]
        max_payload = max(
            (len(v.payload_tys) for v in decl.variants),
            default=0,
        )
        slot_types: list[tir.TIRType] = [tir.TIRScalar("i32")]
        for payload_idx in range(max_payload):
            seen: Optional[tir.TIRType] = None
            for variant in decl.variants:
                if payload_idx >= len(variant.payload_tys):
                    continue
                payload_ty = self._lower_type(variant.payload_tys[payload_idx])
                if seen is None:
                    seen = payload_ty
                elif payload_ty != seen:
                    raise NotImplementedError(
                        f"enum {enum_name} payload slot {payload_idx} has "
                        f"mixed types {tir.fmt_type(seen)} and "
                        f"{tir.fmt_type(payload_ty)}; mixed-type enum "
                        f"payload positions need a tagged-union ABI"
                    )
            slot_types.append(seen or tir.TIRScalar("i32"))
        return slot_types

    def _enum_variant_decl(
        self, enum_name: str, variant_name: str,
    ) -> Optional[A.EnumVariant]:
        decl = next(
            (it for it in self.prog.items
             if isinstance(it, A.EnumDecl) and it.name == enum_name),
            None,
        )
        if decl is None:
            return None
        return next(
            (variant for variant in decl.variants
             if variant.name == variant_name),
            None,
        )

    def _require_tag_only_enum_variant(
        self, enum_name: str, variant_name: str,
    ) -> None:
        variant = self._enum_variant_decl(enum_name, variant_name)
        if variant is not None and variant.payload_tys:
            raise NotImplementedError(
                f"enum constructor {enum_name}::{variant_name} expects "
                f"{len(variant.payload_tys)} payload arg(s); call it with "
                f"payload values before IR lowering"
            )

    def _lower_enum_payload_args(
        self, enum_name: str, variant_name: str, args: list[A.Expr],
    ) -> list[tir.Value]:
        variant = self._enum_variant_decl(enum_name, variant_name)
        if variant is None:
            raise NotImplementedError(
                f"enum constructor {enum_name}::{variant_name} reached IR "
                f"lowering but no matching variant exists; run typecheck first"
            )
        if len(args) != len(variant.payload_tys):
            raise NotImplementedError(
                f"enum constructor {enum_name}::{variant_name} expects "
                f"{len(variant.payload_tys)} payload arg(s), got "
                f"{len(args)}; run typecheck first"
            )
        values: list[tir.Value] = []
        for idx, (arg_expr, payload_ty) in enumerate(
                zip(args, variant.payload_tys)):
            resolved_payload_ty = self._resolve_type_alias_node(payload_ty)
            if (isinstance(resolved_payload_ty, A.TyName)
                    and resolved_payload_ty.name in self._recursive_enums):
                expected = tir.TIRScalar("i32")
                value = self._lower_recursive_enum_payload_arg(
                    arg_expr, resolved_payload_ty.name,
                    f"{enum_name}::{variant_name} arg {idx}")
            else:
                expected = self._lower_type(payload_ty)
                value = self._lower_expr(arg_expr)
                if value is None:
                    value = self.builder.const_int(0)
            if value.ty != expected:
                raise NotImplementedError(
                    f"enum constructor {enum_name}::{variant_name} arg "
                    f"{idx}: expected {tir.fmt_type(expected)}, got "
                    f"{tir.fmt_type(value.ty)}; run typecheck first"
                )
            values.append(value)
        return values

    def _lower_recursive_enum_payload_arg(
        self, expr: A.Expr, enum_name: str, context: str,
    ) -> tir.Value:
        if isinstance(expr, A.Call):
            enum_variant = self._enum_variant_for_expr(expr.callee)
            if enum_variant is not None:
                ename, vname, tag = enum_variant
                if ename != enum_name:
                    raise NotImplementedError(
                        f"{context}: expected {enum_name}, got inline enum "
                        f"constructor {ename}; run typecheck first"
                    )
                tag_v = self.builder.const_int(tag)
                payload_vals = self._lower_enum_payload_args(
                    ename, vname, expr.args)
                return self._arena_push_slots([tag_v] + payload_vals)
            call_rec = self._recursive_enum_name_for_expr(expr)
            if call_rec is not None:
                if call_rec != enum_name:
                    raise NotImplementedError(
                        f"{context}: expected {enum_name}, got function "
                        f"returning {call_rec}; run typecheck first"
                    )
                value = self._lower_expr(expr, expected_rec_enum=enum_name)
                if value is not None:
                    return value
        enum_variant = self._enum_variant_for_expr(expr)
        if enum_variant is not None:
            ename, vname, tag = enum_variant
            if ename != enum_name:
                raise NotImplementedError(
                    f"{context}: expected {enum_name}, got inline enum "
                    f"constructor {ename}; run typecheck first"
                )
            self._require_tag_only_enum_variant(ename, vname)
            return self._arena_push_slots([self.builder.const_int(tag)])
        if isinstance(expr, A.Name) and self._lookup_rec_enum(expr.name) == enum_name:
            value = self._lower_expr(expr)
            if value is not None:
                return value
        got = type(expr).__name__
        if isinstance(expr, A.Name):
            got = self._lookup_rec_enum(expr.name) or f"scalar/name '{expr.name}'"
        raise NotImplementedError(
            f"{context}: expected {enum_name}, got {got}; run typecheck first"
        )

    def _arena_push_slots(self, slots: list[tir.Value]) -> tir.Value:
        start_idx: Optional[tir.Value] = None
        for value in slots:
            pushed = self.builder.emit(
                tir.OpKind.ARENA_PUSH, value,
                result_ty=tir.TIRScalar("i32"))
            if start_idx is None:
                start_idx = pushed
        return start_idx or self.builder.const_int(0)

    def _recursive_enum_name_for_type_node(
        self, ty: Optional["A.TyNode"],
    ) -> Optional[str]:
        if ty is None:
            return None
        resolved = self._resolve_type_alias_node(ty)
        if (isinstance(resolved, A.TyName)
                and resolved.name in self._recursive_enums):
            return resolved.name
        return None

    def _recursive_enum_name_for_expr(
        self, expr: Optional[A.Expr],
    ) -> Optional[str]:
        if expr is None:
            return None
        if isinstance(expr, A.Name):
            rec_enum = self._lookup_rec_enum(expr.name)
            if rec_enum is not None:
                return rec_enum
        enum_variant = self._enum_variant_for_expr(expr)
        if enum_variant is not None:
            ename, _vname, _tag = enum_variant
            if ename in self._recursive_enums:
                return ename
        if isinstance(expr, A.Call):
            enum_ctor = self._enum_variant_for_expr(expr.callee)
            if enum_ctor is not None:
                ename, _vname, _tag = enum_ctor
                if ename in self._recursive_enums:
                    return ename
            if isinstance(expr.callee, A.Name):
                for item in self.prog.items:
                    if (isinstance(item, A.FnDecl)
                            and item.name == expr.callee.name):
                        return self._recursive_enum_name_for_type_node(
                            item.return_ty)
        return None

    def _recursive_enum_name_for_value(
        self, ty: Optional["A.TyNode"], expr: Optional[A.Expr],
        value: tir.Value,
    ) -> Optional[str]:
        rec_enum = self._recursive_enum_name_for_type_node(ty)
        if rec_enum is not None:
            return rec_enum
        rec_enum = self._recursive_enum_name_for_expr(expr)
        if rec_enum is not None:
            return rec_enum
        if (isinstance(value.ty, tir.TIRScalar)
                and value.ty.name in self._recursive_enums):
            return value.ty.name
        return None

    def _resolve_type_alias_node(self, ty: "A.TyNode") -> "A.TyNode":
        """Erase type-alias nodes before ABI-shape decisions."""
        seen: set[str] = set()
        while isinstance(ty, A.TyName):
            alias = self._type_aliases.get(ty.name)
            if alias is None or alias.name in seen:
                return self._resolve_monomorphized_struct_type(ty)
            if alias.generics:
                raise NotImplementedError(
                    f"generic type alias '{alias.name}' reached IR "
                    f"lowering; generic aliases are not supported in Stage 31"
                )
            seen.add(alias.name)
            ty = alias.target
        return self._resolve_monomorphized_struct_type(ty)

    def _resolve_monomorphized_struct_type(
        self, ty: "A.TyNode",
    ) -> "A.TyNode":
        if isinstance(ty, A.TyGeneric):
            # Restart 47 B1: narrow exception scope. mangle_struct ->
            # _mangle_ty explicitly raises NotImplementedError as a loud-fail
            # discipline ("Promote to loud-fail so future additions force
            # explicit dispatch here"). A bare `except Exception` here
            # defeated that discipline: a future TyNode subclass (refinement,
            # confidence, tiered memory) would silently fall through to the
            # unresolved TyGeneric instead of forcing the dispatch.
            # Narrow to (KeyError, AttributeError) which are the
            # mangle_struct-internal lookup failures that legitimately mean
            # "this isn't a known monomorphized struct, return unresolved".
            try:
                from ..frontend.struct_mono import mangle_struct
                mangled = mangle_struct(ty.base, list(ty.args))
            except (KeyError, AttributeError):
                return ty
            if mangled in self._struct_fields:
                return A.TyName(span=ty.span, name=mangled)
        return ty

    def _zero_for_type(self, ty: tir.TIRType) -> tir.Value:
        if isinstance(ty, tir.TIRScalar) and ty.name in {
                "bf16", "f16", "f32", "f64"}:
            return self.builder.const_float(0.0, dtype=ty.name)
        if isinstance(ty, tir.TIRScalar):
            return self.builder.const_int(0, dtype=ty.name)
        return self.builder.const_int(0)

    @staticmethod
    def _homogeneous_slot_type(
        slot_types: list[tir.TIRType],
    ) -> tir.TIRType:
        first = slot_types[0]
        if any(slot_ty != first for slot_ty in slot_types[1:]):
            raise NotImplementedError(
                "heterogeneous aggregate ABI reassembly is not supported yet")
        return first

    @staticmethod
    def _is_homogeneous_slot_list(slot_types: list[tir.TIRType]) -> bool:
        first = slot_types[0]
        return all(slot_ty == first for slot_ty in slot_types[1:])

    def _index_const_values(self) -> None:
        consts = [item for item in self.prog.items
                  if isinstance(item, A.ConstDecl)]
        fn_names = {item.name for item in self.prog.items
                    if isinstance(item, A.FnDecl)}
        for _ in range(len(consts)):
            progressed = False
            for decl in consts:
                if decl.name not in self._const_scalar_values:
                    value = self._eval_const_scalar_expr(decl.value)
                    if value is not None:
                        self._const_scalar_values[decl.name] = value
                        self._const_scalar_types[decl.name] = self._lower_type(
                            decl.ty)
                        progressed = True
                        continue
                if decl.name not in self._const_fn_aliases:
                    fn_alias = self._eval_const_fn_alias_expr(
                        decl.value, fn_names)
                    if fn_alias is not None:
                        self._const_fn_aliases[decl.name] = fn_alias
                        progressed = True
            if not progressed:
                break

    def _eval_const_scalar_expr(
        self, expr: A.Expr,
    ) -> Optional[int | float | bool]:
        if isinstance(expr, A.IntLit):
            return expr.value
        if isinstance(expr, A.FloatLit):
            return expr.value
        if isinstance(expr, A.BoolLit):
            return expr.value
        if isinstance(expr, A.Name):
            return self._const_scalar_values.get(expr.name)
        if isinstance(expr, A.Unary) and expr.op == "-":
            inner = self._eval_const_scalar_expr(expr.operand)
            if (isinstance(inner, (int, float))
                    and not isinstance(inner, bool)):
                return -inner
        if isinstance(expr, A.Binary):
            left = self._eval_const_scalar_expr(expr.left)
            right = self._eval_const_scalar_expr(expr.right)
            if not (isinstance(left, (int, float))
                    and isinstance(right, (int, float))
                    and not isinstance(left, bool)
                    and not isinstance(right, bool)):
                return None
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
        return None

    def _eval_const_fn_alias_expr(
        self, expr: A.Expr, fn_names: set[str],
    ) -> Optional[str]:
        if isinstance(expr, A.Name):
            if expr.name in fn_names:
                return expr.name
            return self._const_fn_aliases.get(expr.name)
        return None

    def _match_payload_rec_enum_name(self, stmt: A.Let) -> Optional[str]:
        if not isinstance(stmt.value, A.Index):
            return None
        payload_ty = getattr(stmt.value, "_match_payload_ty", None)
        if payload_ty is None:
            return None
        payload_ty = self._resolve_type_alias_node(payload_ty)
        if (isinstance(payload_ty, A.TyName)
                and payload_ty.name in self._recursive_enums):
            return payload_ty.name
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
        ty = self._resolve_type_alias_node(ty)
        if isinstance(ty, A.TyName):
            if ty.name in getattr(self, "_recursive_enums", set()):
                return tir.TIRScalar("i32")
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
        if isinstance(ty, A.TyPtr):
            # Stage 16.5: raw pointer lowers to u64 in IR. The pointee type
            # is preserved on the AST for diagnostics but unused at IR level.
            return tir.TIRScalar("u64")
        if isinstance(ty, A.TyGeneric):
            # Stage 48 — Result<T, E> in a position that reaches IR
            # (fn return type, let-binding type) lowers to T (the
            # Ok inner). Phase-0 has no runtime Ok/Err tag, so a
            # Result is observationally identical to its Ok inner.
            # Stage 49 will add the runtime tag and supersede this
            # rule with a 2-slot aggregate (tag word + payload)
            # lowering.
            #
            # Same identity-lowering treatment as the constructors /
            # accessors / `__try` in the expression lowerer above.
            #
            # Only Result is handled here; the Stage 37-41 wrapper
            # quintet (Known/Past/WorldFrame/Cause/WorkingMem etc.)
            # is explicitly *not* extended in this stage — those
            # families have historically been handled by struct-mono
            # turning them into named structs before IR, and only
            # surface in let-RHS expression positions where the
            # expression-lowerer arms handle them. Result is the
            # first family that NEEDS this type-position rule
            # because `?` only makes sense in a Result-returning
            # function, which forces Result into the fn signature.
            if ty.base == "Result" and len(ty.args) == 2:
                # Stage 49 Inc 1: Result<T, E> lowers to a single packed
                # i64 (tag in high 32 bits, payload in low 32 bits). See
                # the convention block on OpKind.RESULT_PACK in tir.py.
                #
                # Stages 46-48 identity-lowered to T (the Ok inner). The
                # transition is safe because:
                #   - Ok(v) constructor now emits RESULT_PACK(0, v) (high
                #     bits = 0), so the low 32 bits == the old i32 value.
                #   - unwrap_ok extracts payload back as i32. Round-trip
                #     value-identical for the static-Ok pathway.
                #   - SysV ABI returns i64 in rax (vs i32 in eax). The
                #     existing _is_64bit_int_type path in CALL/RETURN
                #     already handles this — no caller change needed.
                # Wider payloads (Result<i64, ...>, Result<f64, ...>)
                # remain out of scope until Stage 50+; the i32 payload
                # constraint is enforced via constructor/accessor type
                # arms below (and still by the typecheck arms that
                # require T and E to be i32 for this stage).
                return tir.TIRScalar("i64")
            # Stage 48 closure gate-1 LOW: future 2-parameter
            # wrapper families needing the same type-position
            # identity rule should be added here. The loud-fail
            # raise below is the right discipline for families
            # that escape struct-mono without an explicit arm.
            raise NotImplementedError(
                f"unresolved generic type {ty.base}<...> reached IR "
                f"lowering; run struct monomorphization first")
        if isinstance(ty, A.TyFn):
            return tir.TIRScalar("u64")
        # Restart 54 B2: loud-fail on unknown TyNode subclass instead of
        # returning a TIRScalar("?") sentinel. Sibling of restart 47 B1
        # which installed the same loud-fail discipline on
        # _resolve_monomorphized_struct_type — that comment cites this
        # site as the next target. A future TyNode subclass would
        # otherwise silently lower to "?" and codegen sizing routines
        # would silently fall past every branch.
        raise NotImplementedError(
            f"unsupported TyNode subclass {type(ty).__name__} in IR "
            f"lowering: {ty!r}")

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
        # or enum) expand to N typed IR params, where N is the slot count.
        # The callee uses param-name suffixed with "__slot{i}" to
        # distinguish; reassembly into an array binding happens in
        # _lower_fn_body.
        params: list[tuple[str, tir.TIRType]] = []
        for p in fn.params:
            p_ty = self._resolve_type_alias_node(p.ty)
            # Recursive enum: single i32 arena index (no slot expansion).
            if (isinstance(p_ty, A.TyName)
                    and p_ty.name in self._recursive_enums):
                params.append((p.name, tir.TIRScalar("i32")))
                continue
            slot_types = self._aggregate_slot_types(p_ty)
            # Aggregate-typed params always go through the multi-slot
            # path, even single-field ones — the body uses field-access
            # syntax which expects an array binding.
            if slot_types is not None and len(slot_types) >= 1:
                for i, slot_ty in enumerate(slot_types):
                    params.append((f"{p.name}__slot{i}", slot_ty))
            else:
                t = self._lower_type(p_ty)
                params.append((p.name, t))
        ret = self._lower_type(fn.return_ty) if fn.return_ty else tir.TIRUnit()
        attrs: dict[str, object] = {}
        for a in fn.attrs:
            attrs[a] = True
        if fn.is_pub:
            attrs["is_pub"] = True
        # Stage 16.5 — record extern fns so the call lowering can route to
        # FFI_CALL instead of CALL, and the backend can skip emitting a body
        # symbol (resolved by the dynamic linker at runtime instead).
        if fn.is_extern:
            attrs["is_extern"] = True
            attrs["extern_abi"] = fn.extern_abi or "C"
        ir_fn = self.builder.begin_function(fn.name, params, ret, attrs=attrs)
        self.functions[fn.name] = ir_fn
        # Don't lower body yet — that's pass 2
        self.builder.end_function()

    # ---- function bodies ----
    def _lower_fn_body(self, fn: A.FnDecl) -> None:
        # Stage 16.5: extern "C" fn declarations have no body to lower.
        # The fn entry exists for type-check / call-site resolution only;
        # the backend never emits a body symbol for it.
        if fn.is_extern:
            return
        ir_fn = self.functions.get(fn.name)
        if ir_fn is None:
            return
        self.builder.current_fn = ir_fn
        self.builder.current_block = ir_fn.entry
        self._push_scope()
        # Audit 28.8 A7 — Stage 25 @trace prologue. Emit a TRACE_ENTRY
        # op carrying the fn name so the backend (when the runtime is
        # linked) can call __helix_trace_entry(name_ptr). Phase-0:
        # backend emits this as a no-op stub until the runtime exists.
        # The IR-level op is observable for tests + IR dumps so the
        # wiring is validated regardless.
        is_fn_traced = "trace" in fn.attrs
        if is_fn_traced:
            self.builder.emit(tir.OpKind.TRACE_ENTRY,
                              attrs={"fn_name": fn.name})
        # Audit 28.8 cycle 2 C2-2 — track @trace context on the lowerer
        # so A.Return's lowering arm can emit TRACE_EXIT before the
        # `ret` op on early-return paths (mirroring the fall-through
        # epilogue at the end of this function).
        prev_is_fn_traced = self._is_fn_traced
        prev_current_fn_name = self._current_fn_name
        prev_expected_rec_enum = self._current_expected_rec_enum
        self._is_fn_traced = is_fn_traced
        self._current_fn_name = fn.name
        self._current_expected_rec_enum = (
            self._recursive_enum_name_for_type_node(fn.return_ty))
        # Stage 16 — track kernel context. thread_idx() and indexed-tile ops
        # are only valid inside @kernel fns.
        prev_in_kernel = self._in_kernel
        self._in_kernel = "kernel" in fn.attrs
        # Bind params to their SSA values. AGGREGATE-typed params were
        # expanded to N consecutive IR params in _register_fn — reassemble
        # them into an array binding here so the body can use field/index
        # access transparently.
        ir_param_idx = 0
        # Stage 16 — track kernel HBM tile params positionally; needed by
        # the PTX backend to know which `.param .u64 param_N` slot to load.
        kernel_hbm_param_pos = 0
        for p in fn.params:
            p_ty = self._resolve_type_alias_node(p.ty)
            # Stage 16 — kernel param typed `tile<dtype, [N], HBM>`. Register
            # for indexed-load/store lowering. Skip the multi-slot expansion
            # below (kernel tile params are opaque pointers at this level).
            if (self._in_kernel
                    and isinstance(p_ty, A.TyTile)
                    and self._stringify_marker(p_ty.memspace) in ("HBM", "hbm")):
                # Validate dtype + shape constraints. Phase-0: 1D, dtype known.
                dtype_node = p_ty.dtype
                if not (isinstance(dtype_node, A.TyName)
                        and dtype_node.name in ("f32", "i32")):
                    raise NotImplementedError(
                        "Stage 16 HBM tile param dtype must be f32/i32; "
                        f"got {dtype_node}")
                if len(p_ty.shape) != 1:
                    raise NotImplementedError(
                        "Stage 16 HBM tile param shape must be 1D; "
                        f"got {len(p_ty.shape)}D")
                length_expr = p_ty.shape[0]
                length = (length_expr.value
                          if isinstance(length_expr, A.IntLit) else 0)
                # Skip IR param consumption — kernel param ptrs are not bound
                # as SSA values (they're addressed in PTX via param_N slots,
                # not the host calling convention).
                v = ir_fn.params[ir_param_idx]
                ir_param_idx += 1
                self._bind(p.name, v)
                self._bind_hbm_tile(p.name, dtype_node.name, length,
                                    kernel_hbm_param_pos)
                kernel_hbm_param_pos += 1
                continue
            # Recursive-enum-typed param: scalar i32 arena index. Bind as
            # scalar + register in rec_enum scope so Index access emits
            # ARENA_GET against it.
            if (isinstance(p_ty, A.TyName)
                    and p_ty.name in self._recursive_enums):
                v = ir_fn.params[ir_param_idx]
                ir_param_idx += 1
                self._bind(p.name, v)
                self._bind_rec_enum(p.name, p_ty.name)
                continue
            slot_types = self._aggregate_slot_types(p_ty)
            if slot_types is not None and len(slot_types) >= 1:
                n_slots = len(slot_types)
                # Take next n_slots IR params. Homogeneous aggregates keep the
                # older array-backed path; heterogeneous aggregates bind typed
                # slots directly so field access preserves each slot type.
                slot_vals = list(ir_fn.params[ir_param_idx:
                                              ir_param_idx + n_slots])
                ir_param_idx += n_slots
                if self._is_homogeneous_slot_list(slot_types):
                    elem_ty = slot_types[0]
                    self.builder.emit(tir.OpKind.ALLOC_ARRAY,
                                      attrs={"name": p.name,
                                             "dtype": elem_ty,
                                             "length": n_slots})
                    for i, sv in enumerate(slot_vals):
                        idx = self.builder.const_int(i)
                        self.builder.emit(tir.OpKind.STORE_ELEM, idx, sv,
                                          attrs={"name": p.name})
                    self._bind_array(p.name, elem_ty, n_slots)
                else:
                    self._bind_aggregate(p.name, slot_vals)
                if isinstance(p_ty, A.TyName) \
                        and p_ty.name in self._struct_flat_paths:
                    self._bind_struct(p.name, p_ty.name)
                if (isinstance(p_ty, A.TyName)
                        and p_ty.name in self._enum_variants):
                    self._bind_enum(p.name, p_ty.name)
            else:
                v = ir_fn.params[ir_param_idx]
                ir_param_idx += 1
                if p.is_mut:
                    ir_name = self._bind_mut(p.name, v.ty)
                    self.builder.emit(tir.OpKind.ALLOC_VAR,
                                      attrs={"name": ir_name, "dtype": v.ty})
                    self.builder.emit(tir.OpKind.STORE_VAR, v,
                                      attrs={"name": ir_name})
                else:
                    self._bind(p.name, v)
                if (isinstance(p_ty, A.TyName)
                        and p_ty.name in self._enum_variants):
                    self._bind_enum(p.name, p_ty.name)
        # Lower body block
        body_val = self._lower_block(
            fn.body,
            expected_rec_enum=self._current_expected_rec_enum)
        # Audit 28.8 A7 — Stage 25 @trace epilogue. Emit TRACE_EXIT
        # with the return value (so the runtime can record it) before
        # the actual return instruction. If the fn returns Unit, pass
        # a synthesized 0 sentinel.
        if is_fn_traced:
            ret_operand = body_val
            if isinstance(ir_fn.return_ty, tir.TIRUnit) or ret_operand is None:
                ret_operand = self.builder.const_int(0)
            self.builder.emit(tir.OpKind.TRACE_EXIT, ret_operand,
                              attrs={"fn_name": fn.name})
        # Emit return
        if isinstance(ir_fn.return_ty, tir.TIRUnit):
            self.builder.ret(None)
        elif body_val is not None:
            self.builder.ret(body_val)
        else:
            self.builder.ret(None)
        self._pop_scope()
        # Stage 16 — restore the in-kernel flag (lower_fn_body is called per
        # top-level fn so this is technically always False after, but the
        # explicit restore matches the scope-pop discipline).
        self._in_kernel = prev_in_kernel
        # Audit 28.8 cycle 2 C2-2 — restore traced-fn state.
        self._is_fn_traced = prev_is_fn_traced
        self._current_fn_name = prev_current_fn_name
        self._current_expected_rec_enum = prev_expected_rec_enum
        self.builder.end_function()

    def _lower_block(
        self, block: A.Block, expected_rec_enum: Optional[str] = None,
    ) -> Optional[tir.Value]:
        self._push_scope()
        try:
            for stmt in block.stmts:
                self._lower_stmt(stmt)
            if block.final_expr is not None:
                return self._lower_expr(
                    block.final_expr,
                    expected_rec_enum=expected_rec_enum)
            return None
        finally:
            self._pop_scope()

    # ---- Stage 15 Tile helpers ----
    def _tile_shape_dims(self, lit: "A.TileLit") -> tuple[int, int]:
        """Resolve a TileLit's shape to (rows, cols). Phase-0 requires both
        dims to be IntLit (compile-time constants). Raises ValueError on
        non-literal dims."""
        if len(lit.shape) != 2:
            raise NotImplementedError(
                f"tile<>:: requires a 2D shape; got {len(lit.shape)}D"
            )
        dims = []
        for d in lit.shape:
            if isinstance(d, A.IntLit):
                dims.append(d.value)
            else:
                raise NotImplementedError(
                    "tile<> shape must be IntLit constants in Phase 0"
                )
        return (dims[0], dims[1])

    def _tile_dtype_check(self, dtype: "A.TyNode") -> None:
        """Phase-0: only f32 tile dtype supported."""
        if not (isinstance(dtype, A.TyName) and dtype.name == "f32"):
            raise NotImplementedError(
                f"tile<> dtype must be f32 in Phase 0; got {dtype}"
            )

    def _tile_cap_check(self, rows: int, cols: int) -> None:
        """Phase-0 cap shape at 8x8 (N*M <= 64). Trap-id 91001 at codegen."""
        if rows * cols > 64:
            raise NotImplementedError(
                f"tile<> shape {rows}x{cols} exceeds Phase 0 cap of 64 elems"
            )

    def _lower_tile_lit_let(self, stmt: "A.Let") -> None:
        """Stage 15: `let X = tile<f32, [N, M], REG>::zeros()/ones()`.

        Allocates an N*M f32 array on the stack, stores the init value
        (0.0 or 1.0) into every slot, and records the tile shape so
        tile_matmul + .get can use it.
        """
        lit = stmt.value
        assert isinstance(lit, A.TileLit)
        self._tile_dtype_check(lit.dtype)
        rows, cols = self._tile_shape_dims(lit)
        self._tile_cap_check(rows, cols)
        elem_ty = tir.TIRScalar("f32")
        n = rows * cols
        self.builder.emit(tir.OpKind.ALLOC_ARRAY,
                          attrs={"name": stmt.name, "dtype": elem_ty,
                                 "length": n})
        # Build the init value once.
        init_val = 0.0 if lit.init == "zeros" else 1.0
        init_v = self.builder.emit(tir.OpKind.CONST_FLOAT,
                                   result_ty=elem_ty,
                                   attrs={"value": init_val,
                                          "dtype": "f32"})
        # Store into every slot. Loop unrolled because N*M <= 64 is small.
        for i in range(n):
            idx = self.builder.const_int(i)
            self.builder.emit(tir.OpKind.STORE_ELEM, idx, init_v,
                              attrs={"name": stmt.name})
        self._bind_array(stmt.name, elem_ty, n)
        self._bind_tile(stmt.name, rows, cols)

    def _lower_tile_matmul_let(self, stmt: "A.Let") -> None:
        """Stage 15: `let C = tile_matmul(A, B)`.

        Naive triple-loop matmul on f32 tiles, fully unrolled at compile
        time. Phase-0: A is N×K, B is K×M, C is N×M. Both A and B must be
        existing tile bindings registered via tile_scope.

        For each (i, k) with i in 0..N, k in 0..M:
            acc = 0.0
            for j in 0..K:
                acc += A[i*K + j] * B[j*M + k]
            C[i*M + k] = acc
        """
        call = stmt.value
        assert isinstance(call, A.Call)
        if len(call.args) != 2:
            raise NotImplementedError(
                "tile_matmul takes exactly 2 args (a, b)"
            )
        a_arg, b_arg = call.args
        if not (isinstance(a_arg, A.Name) and isinstance(b_arg, A.Name)):
            raise NotImplementedError(
                "tile_matmul args must be tile bindings (Name) in Phase 0"
            )
        a_shape = self._lookup_tile(a_arg.name)
        b_shape = self._lookup_tile(b_arg.name)
        if a_shape is None or b_shape is None:
            raise NotImplementedError(
                "tile_matmul args must be tile bindings registered via "
                "tile<>:: literals"
            )
        n, k_a = a_shape
        k_b, m = b_shape
        if k_a != k_b:
            raise NotImplementedError(
                f"tile_matmul shape mismatch: A is {n}x{k_a}, B is {k_b}x{m}"
            )
        k = k_a
        out_n = n * m
        self._tile_cap_check(n, m)
        elem_ty = tir.TIRScalar("f32")
        # Allocate the output tile.
        self.builder.emit(tir.OpKind.ALLOC_ARRAY,
                          attrs={"name": stmt.name, "dtype": elem_ty,
                                 "length": out_n})
        # Naive triple-loop, fully unrolled.
        zero_v = self.builder.emit(tir.OpKind.CONST_FLOAT,
                                   result_ty=elem_ty,
                                   attrs={"value": 0.0, "dtype": "f32"})
        for i in range(n):
            for kk in range(m):
                acc = zero_v
                for j in range(k):
                    a_idx = self.builder.const_int(i * k + j)
                    a_v = self.builder.emit(tir.OpKind.LOAD_ELEM, a_idx,
                                            result_ty=elem_ty,
                                            attrs={"name": a_arg.name})
                    b_idx = self.builder.const_int(j * m + kk)
                    b_v = self.builder.emit(tir.OpKind.LOAD_ELEM, b_idx,
                                            result_ty=elem_ty,
                                            attrs={"name": b_arg.name})
                    prod = self.builder.emit(tir.OpKind.MUL, a_v, b_v,
                                             result_ty=elem_ty)
                    acc = self.builder.emit(tir.OpKind.ADD, acc, prod,
                                            result_ty=elem_ty)
                out_idx = self.builder.const_int(i * m + kk)
                self.builder.emit(tir.OpKind.STORE_ELEM, out_idx, acc,
                                  attrs={"name": stmt.name})
        self._bind_array(stmt.name, elem_ty, out_n)
        self._bind_tile(stmt.name, n, m)

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
            # Stage 15: tile literal (`tile<f32, [N, M], REG>::zeros()` /
            # `::ones()`). Lowers to ALLOC_ARRAY of N*M f32 elements + a
            # STORE_ELEM for each slot. The binding tracks the tile shape
            # in self.tile_scope so tile_matmul / .get can resolve dims.
            # Phase-0: f32 dtype + REG memspace + cap shape at 8x8 (N*M <= 64).
            if stmt.value is not None and isinstance(stmt.value, A.TileLit):
                self._lower_tile_lit_let(stmt)
                return
            # Stage 15: tile_matmul(a, b) result binding. Special-case here
            # because the call-site lowering needs to know shape of the
            # result tile up-front (so we can ALLOC_ARRAY at the binding).
            if (stmt.value is not None
                    and isinstance(stmt.value, A.Call)
                    and isinstance(stmt.value.callee, A.Name)
                    and stmt.value.callee.name == "tile_matmul"):
                self._lower_tile_matmul_let(stmt)
                return
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
            if stmt.value is not None:
                enum_variant = self._enum_variant_for_expr(stmt.value)
            else:
                enum_variant = None
            if enum_variant is not None:
                ename, vname, tag = enum_variant
                self._require_tag_only_enum_variant(ename, vname)
                if ename in self._recursive_enums:
                    tag_v = self.builder.const_int(tag)
                    pushed = self.builder.emit(
                        tir.OpKind.ARENA_PUSH, tag_v,
                        result_ty=tir.TIRScalar("i32"))
                    self._bind(stmt.name, pushed)
                    self._bind_rec_enum(stmt.name, ename)
                    return
                tag_v = self.builder.const_int(tag)
                self._bind(stmt.name, tag_v)
                self._bind_enum(stmt.name, ename)
                return
            if (stmt.value is not None
                    and isinstance(stmt.value, A.Call)):
                enum_variant = self._enum_variant_for_expr(stmt.value.callee)
                if enum_variant is not None:
                    ename, vname, tag = enum_variant
                    tag_v = self.builder.const_int(tag)
                    arg_vals = self._lower_enum_payload_args(
                        ename, vname, stmt.value.args)
                    slots = [tag_v] + arg_vals
                    # Recursive enum: arena-indirected. Push slots into
                    # the arena, bind name to the start index (scalar
                    # i32). Match dispatch will use ARENA_GET against
                    # the index.
                    if ename in self._recursive_enums:
                        start_idx = self._arena_push_slots(slots)
                        self._bind(stmt.name, start_idx)
                        self._bind_rec_enum(stmt.name, ename)
                        return
                    n = len(slots)
                    if self._is_homogeneous_slot_list(
                            [slot.ty for slot in slots]):
                        elem_ty = slots[0].ty
                        self.builder.emit(tir.OpKind.ALLOC_ARRAY,
                                          attrs={"name": stmt.name,
                                                 "dtype": elem_ty,
                                                 "length": n})
                        for i, ev in enumerate(slots):
                            idx = self.builder.const_int(i)
                            self.builder.emit(tir.OpKind.STORE_ELEM, idx, ev,
                                              attrs={"name": stmt.name})
                        self._bind_array(stmt.name, elem_ty, n)
                    else:
                        self._bind_aggregate(stmt.name, slots)
                    self._bind_enum(stmt.name, ename)
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
                n = len(elem_vals)
                if all(ev.ty == elem_vals[0].ty for ev in elem_vals[1:]):
                    elem_ty = elem_vals[0].ty
                    self.builder.emit(tir.OpKind.ALLOC_ARRAY,
                                      attrs={"name": stmt.name,
                                             "dtype": elem_ty,
                                             "length": n})
                    for i, ev in enumerate(elem_vals):
                        idx = self.builder.const_int(i)
                        self.builder.emit(tir.OpKind.STORE_ELEM, idx, ev,
                                          attrs={"name": stmt.name})
                    self._bind_array(stmt.name, elem_ty, n)
                else:
                    self._bind_aggregate(stmt.name, elem_vals)
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
                        # Audit 28.8 cycle 19 C18-1/audit-C (B18-1 from
                        # cycle-18 codereview): pre-fix the silent
                        # `const_int(0)` fallback decayed nested
                        # aggregates (`(a, b, [c, d])` etc.) to zero
                        # at runtime — typecheck didn't catch it; lower
                        # silently inserted 0; STORE_ELEM emitted i32
                        # zero. Now we trap loudly with the audit
                        # stamp so the user sees a clear migration
                        # hint instead of a wrong-answer.
                        raise NotImplementedError(
                            f"x86_64 backend does not yet support tuple "
                            f"element kind '{type(e).__name__}' (would "
                            f"silently lower to 0 — see audit-stage28-8 "
                            f"cycle 19 C18-1/B18-1). Phase-0 supports "
                            f"scalar tuple elements only; nested "
                            f"aggregate literals (ArrayLit, StructLit, "
                            f"TupleLit) are not yet lowered."
                        )
                    elem_vals.append(v)
                elem_ty = elem_vals[0].ty
                for ev in elem_vals[1:]:
                    if ev.ty != elem_ty:
                        raise TypeError(
                            f"array literal element type mismatch after "
                            f"typecheck: first element is "
                            f"{tir.fmt_type(elem_ty)}, later element is "
                            f"{tir.fmt_type(ev.ty)}"
                        )
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
                        # Audit 28.8 cycle 19 C18-1 (cycle-18 codereview
                        # audit): pre-fix the silent `const_int(0)`
                        # fallback decayed nested aggregate elements
                        # (`[[10, 20], [30, 40]]`, `[Pt{x:1,y:2},
                        # Pt{x:3,y:4}]`, etc.) to zero at runtime —
                        # surface code returned `xs[0]` = 0 instead
                        # of the inner aggregate. Now we trap loudly
                        # so the user sees a clear migration hint
                        # instead of a wrong-answer.
                        raise NotImplementedError(
                            f"x86_64 backend does not yet support array "
                            f"element kind '{type(e).__name__}' (would "
                            f"silently lower to 0 — see audit-stage28-8 "
                            f"cycle 19 C18-1). Phase-0 supports scalar "
                            f"array elements only; nested aggregate "
                            f"literals (ArrayLit, StructLit, TupleLit) "
                            f"are not yet lowered."
                        )
                    elem_vals.append(v)
                elem_ty = elem_vals[0].ty
                for ev in elem_vals[1:]:
                    if ev.ty != elem_ty:
                        raise TypeError(
                            f"array literal element type mismatch after "
                            f"typecheck: first element is "
                            f"{tir.fmt_type(elem_ty)}, later element is "
                            f"{tir.fmt_type(ev.ty)}"
                        )
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
                src_enum = self._lookup_enum(src_name)
                src_arr = self._lookup_array(src_name)
                if src_arr is not None:
                    elem_ty, length = src_arr
                    self._bind_array(stmt.name, elem_ty, length)
                    src_struct = self._lookup_struct(src_name)
                    if src_struct is not None:
                        self._bind_struct(stmt.name, src_struct)
                    if src_enum is not None:
                        self._bind_enum(stmt.name, src_enum)
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
                src_agg = self._lookup_aggregate(src_name)
                if src_agg is not None:
                    self._bind_aggregate(stmt.name, list(src_agg))
                    src_struct = self._lookup_struct(src_name)
                    if src_struct is not None:
                        self._bind_struct(stmt.name, src_struct)
                    if src_enum is not None:
                        self._bind_enum(stmt.name, src_enum)
                    return
                if src_enum is not None:
                    src_v = self._lookup(src_name)
                    if src_v is not None:
                        self._bind(stmt.name, src_v)
                        self._bind_enum(stmt.name, src_enum)
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
                rec_enum = self._recursive_enum_name_for_value(
                    stmt.ty, stmt.value, v)
                if rec_enum is not None:
                    self._bind_rec_enum(stmt.name, rec_enum)
                else:
                    payload_rec = self._match_payload_rec_enum_name(stmt)
                    if payload_rec is not None:
                        self._bind_rec_enum(stmt.name, payload_rec)
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

    def _lower_expr(
        self, expr: A.Expr, expected_rec_enum: Optional[str] = None,
    ) -> Optional[tir.Value]:
        if isinstance(expr, A.IntLit):
            return self.builder.const_int(expr.value, expr.type_suffix or "i32")
        if isinstance(expr, A.FloatLit):
            return self.builder.const_float(expr.value, expr.type_suffix or "f32")
        if isinstance(expr, A.BoolLit):
            return self.builder.emit(tir.OpKind.CONST_BOOL,
                                     result_ty=tir.TIRScalar("bool"),
                                     attrs={"value": expr.value})
        # Stage 28.9 cycle 108 audit-S C107-F8 fix (HIGH conf 82):
        # explicit loud-fail arms for A.CharLit / A.StructLit (in expr
        # position) / A.TileLit (in expr position). Pre-fix the bottom-
        # of-_lower_expr `return None` silently dropped these three
        # subclasses; the caller-side `or self.builder.const_int(0)`
        # then substituted 0 for the lost value. Typical surface:
        # `let c = 'A';` lowered to const_int(0), making `c == 'A'`
        # silently return `0 == 0 -> true` for the wrong reason.
        # Cycle-106 added explicit arms for A.Break / A.Continue with
        # the same loud-fail pattern; this cycle extends it to the
        # remaining subclasses the parser accepts but lowering has no
        # arm for. The bottom `return None` at line ~2245 is preserved
        # for A.StrLit (cycle-101 F1 deferred-known) — the StrLit case
        # is not yet ported to IR and the user has accepted its silent
        # miscompile under the deferred-known contract; converting the
        # catch-all would close it without a real implementation.
        if isinstance(expr, A.CharLit):
            raise NotImplementedError(
                f"char literal not yet supported in IR lowering at "
                f"{expr.span.line}:{expr.span.col}")
        if isinstance(expr, A.StructLit):
            # Handled in LetStmt at line 848 by short-circuit; reaching
            # _lower_expr means the StructLit appears in an arg / if-arm
            # / return / assign rhs position the let-stmt path can't see.
            raise NotImplementedError(
                f"struct literal in expression position not yet supported "
                f"in IR lowering at {expr.span.line}:{expr.span.col}")
        if isinstance(expr, A.TileLit):
            # Handled in LetStmt at line 762 by short-circuit; sibling
            # of StructLit-in-expr-position above.
            raise NotImplementedError(
                f"tile literal in expression position not yet supported "
                f"in IR lowering at {expr.span.line}:{expr.span.col}")
        if isinstance(expr, A.Name):
            v = self._lookup(expr.name)
            if v is not None:
                return v
            const_value = self._const_scalar_values.get(expr.name)
            if isinstance(const_value, bool):
                return self.builder.emit(tir.OpKind.CONST_BOOL,
                                         result_ty=tir.TIRScalar("bool"),
                                         attrs={"value": const_value})
            if isinstance(const_value, int):
                const_ty = self._const_scalar_types.get(expr.name)
                if isinstance(const_ty, tir.TIRScalar):
                    return self.builder.const_int(
                        const_value, dtype=const_ty.name)
                return self.builder.const_int(const_value)
            if isinstance(const_value, float):
                const_ty = self._const_scalar_types.get(expr.name)
                dtype = const_ty.name if isinstance(const_ty, tir.TIRScalar) else "f64"
                return self.builder.const_float(const_value, dtype=dtype)
            # Mutable variable -> emit LOAD_VAR (use mangled IR name)
            mut_ty = self._lookup_mut(expr.name)
            if mut_ty is not None:
                ir_name = self._lookup_mut_ir_name(expr.name) or expr.name
                return self.builder.emit(tir.OpKind.LOAD_VAR,
                                         result_ty=mut_ty,
                                         attrs={"name": ir_name})
            if expr.name in self._GPU_INDEX_BUILTINS:
                raise NotImplementedError(
                    f"GPU builtin {expr.name} must be called as {expr.name}()"
                )
            # Maybe a function reference (v0.1: emit a call-able marker)
            if expr.name in self.functions:
                return self.builder.const_int(0)
            fn_alias = self._const_fn_aliases.get(expr.name)
            if fn_alias in self.functions:
                return self.builder.const_int(0)
            enum_variant = self._enum_variant_for_expr(expr)
            if enum_variant is not None:
                ename, vname, tag = enum_variant
                self._require_tag_only_enum_variant(ename, vname)
                if expected_rec_enum is not None:
                    if ename != expected_rec_enum:
                        raise NotImplementedError(
                            f"expected recursive enum {expected_rec_enum}, "
                            f"got enum constructor {ename}; run typecheck first"
                        )
                    if ename in self._recursive_enums:
                        return self._arena_push_slots(
                            [self.builder.const_int(tag)])
                return self.builder.const_int(tag)
            raise NotImplementedError(
                f"unresolved value name '{expr.name}' in IR lowering at "
                f"{expr.span.line}:{expr.span.col}; run typecheck first"
            )
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
            enum_variant = self._enum_variant_for_expr(expr)
            if enum_variant is not None:
                ename, vname, tag = enum_variant
                self._require_tag_only_enum_variant(ename, vname)
                if expected_rec_enum is not None:
                    if ename != expected_rec_enum:
                        raise NotImplementedError(
                            f"expected recursive enum {expected_rec_enum}, "
                            f"got enum constructor {ename}; run typecheck first"
                        )
                    if ename in self._recursive_enums:
                        return self._arena_push_slots(
                            [self.builder.const_int(tag)])
                return self.builder.const_int(tag)
            segs = list(expr.segments)
            if len(segs) >= 3 and segs[0] == "crate":
                segs = segs[1:]
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
            raise NotImplementedError(
                f"unresolved path {'::'.join(expr.segments)} cannot be "
                f"lowered"
            )
        if isinstance(expr, A.Binary):
            if expr.op in ("==", "!="):
                # match_lower desugars enum-pattern dispatch to
                # `scrut[0] == Enum::Variant`. Payload-bearing variants are
                # valid as tag constants only in that generated index-compare
                # shape; ordinary value-position lowering still fails closed.
                right_variant = self._enum_variant_for_expr(expr.right)
                left_variant = self._enum_variant_for_expr(expr.left)
                if right_variant is not None and isinstance(expr.left, A.Index):
                    l = self._lower_expr(expr.left)
                    r = self.builder.const_int(right_variant[2])
                elif left_variant is not None and isinstance(expr.right, A.Index):
                    l = self.builder.const_int(left_variant[2])
                    r = self._lower_expr(expr.right)
                else:
                    l = self._lower_expr(expr.left)
                    r = self._lower_expr(expr.right)
            else:
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
            # Bitwise integer ops. Pre-fix: & / | / ^ used to fall through to
            # the `||` lowering (`(l + r) != 0`), so `5 & 3` returned 1
            # because `5+3 != 0` — silently wrong for any non-zero operand.
            bitwise = {
                "&": tir.OpKind.BIT_AND,
                "|": tir.OpKind.BIT_OR,
                "^": tir.OpKind.BIT_XOR,
                "<<": tir.OpKind.SHL,
                ">>": tir.OpKind.SHR,
            }
            if expr.op in arith:
                return self.builder.emit(arith[expr.op], l, r, result_ty=l.ty)
            if expr.op in cmp_:
                return self.builder.emit(cmp_[expr.op], l, r,
                                         result_ty=tir.TIRScalar("bool"))
            if expr.op in bitwise:
                return self.builder.emit(bitwise[expr.op], l, r, result_ty=l.ty)
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
            # Bitwise NOT (~). Pre-fix: `~` fell through to `return inner`
            # so `~5` returned 5 unchanged.
            if expr.op == "~":
                return self.builder.emit(tir.OpKind.BIT_NOT, inner, result_ty=inner.ty)
            # Logical NOT (!). Pre-fix: `!` also fell through to `return inner`
            # so `!1` returned 1 instead of 0. Lower as `inner == 0`.
            if expr.op == "!":
                zero = self.builder.const_int(0)
                return self.builder.emit(tir.OpKind.CMP_EQ, inner, zero,
                                         result_ty=tir.TIRScalar("bool"))
            if expr.op in ("&", "&mut"):
                raise NotImplementedError(
                    f"unary {expr.op} address-of lowering is not implemented "
                    "yet; check-only typing is available, but compiled "
                    "reference storage needs a real IR operation"
                )
            if expr.op == "*":
                raise NotImplementedError(
                    "unary * dereference lowering is not implemented yet; "
                    "check-only typing is available, but compiled pointer "
                    "loads need a real IR operation"
                )
            raise NotImplementedError(
                f"unsupported unary operator {expr.op!r} reached lowering"
            )
        if isinstance(expr, A.Call):
            # Stage 36 Increment 4: D<T> wrapper made runnable.
            # Stage 24 defined attach(T) -> D<T> and detach(D<T>) -> T as
            # typecheck-only — programs using them failed to lower with
            # "unknown function 'attach'". The D<T> wrapper is
            # representationally identical to T at IR level (Phase-0:
            # same single-tag convention as Logic<T>), so wiring both
            # as identity unblocks D<Logic<T>> compositions for free.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "attach"
                    and len(expr.args) == 1):
                return self._lower_expr(expr.args[0])
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "detach"
                    and len(expr.args) == 1):
                return self._lower_expr(expr.args[0])
            # Stage 36 Increment 1: provenance-typed primitives.
            # prove(value, source) lowers to value (Phase-0: the Logic<T>
            # wrapper has zero runtime overhead; provenance lives purely
            # at type level). unwrap_logic(l) lowers to l (identity).
            # Both calls are typecheck-recognized boundary markers; once
            # typecheck passes, the IR-level value is just the inner T.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "prove"
                    and len(expr.args) == 2):
                return self._lower_expr(expr.args[0])
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "unwrap_logic"
                    and len(expr.args) == 1):
                return self._lower_expr(expr.args[0])
            # Stage 36 Increment 2: provenance-composing combinators.
            # derive(a, b) returns a's value but ALSO registers the
            # two-parent relationship in the arena side-table so the
            # call is observable (audit B2 fix). and_logic / or_logic
            # lower to bitwise i32 min/max on 0/1 truth values.
            # not_logic flips 0<->1.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "derive"
                    and len(expr.args) == 2):
                # Stage 36 Inc 9 audit B1 (code-review) fix: evaluate
                # args in source order (a then b). Helix expressions
                # can have observable side effects (io::println etc.),
                # so `derive(log("a"), log("b"))` must print "a"
                # before "b".
                #
                # Stage 36 Inc 9 silent-failure B2 fix: pre-fix the
                # lowering was `_lower(a); _lower(b); return a` — b's
                # value was dropped entirely, making derive(p, q) and
                # p observationally indistinguishable (the combinator
                # was dead weight that violated its own typecheck
                # contract). Now derive routes its two parents through
                # the atomic ARENA_PUSH_PAIR side-table the same way
                # register_derivation does, so the call has an
                # observable effect (arena_len() grows by 2, and the
                # parent_*_at lookups at the freshly returned slot
                # index recover both parents). The user-visible return
                # value remains a's value (Phase-0 single-tag value
                # propagation); the registration handle is dropped on
                # the floor because derive's contract returns Logic<T>,
                # not a registry handle. Code that wants the handle
                # should call register_derivation directly.
                a_v = self._lower_expr(expr.args[0])
                b_v = self._lower_expr(expr.args[1])
                if a_v is not None and b_v is not None:
                    self.builder.emit(
                        tir.OpKind.ARENA_PUSH_PAIR, a_v, b_v,
                        result_ty=tir.TIRScalar("i32"))
                return a_v
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "and_logic"
                    and len(expr.args) == 2):
                a = self._lower_expr(expr.args[0])
                b = self._lower_expr(expr.args[1])
                if a is None or b is None:
                    return None
                # AND on 0/1 truth values: bitwise AND (preserves
                # 0/1 semantics correctly for boolean inputs).
                return self.builder.emit(
                    tir.OpKind.BIT_AND, a, b,
                    result_ty=tir.TIRScalar("i32"))
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "or_logic"
                    and len(expr.args) == 2):
                a = self._lower_expr(expr.args[0])
                b = self._lower_expr(expr.args[1])
                if a is None or b is None:
                    return None
                # OR on 0/1 truth values: bitwise OR (preserves 0/1
                # semantics correctly for boolean inputs).
                return self.builder.emit(
                    tir.OpKind.BIT_OR, a, b,
                    result_ty=tir.TIRScalar("i32"))
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "not_logic"
                    and len(expr.args) == 1):
                a = self._lower_expr(expr.args[0])
                if a is None:
                    return a
                # NOT on a 0/1 truth value: 1 - a.
                one = self.builder.const_int(1)
                return self.builder.emit(
                    tir.OpKind.SUB, one, a,
                    result_ty=tir.TIRScalar("i32"))
            # Stage 36 Increment 3: boolean-algebra completeness.
            # xor_logic(a, b) → BIT_XOR(a, b).
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "xor_logic"
                    and len(expr.args) == 2):
                a = self._lower_expr(expr.args[0])
                b = self._lower_expr(expr.args[1])
                if a is None or b is None:
                    return None
                return self.builder.emit(
                    tir.OpKind.BIT_XOR, a, b,
                    result_ty=tir.TIRScalar("i32"))
            # implies_logic(a, b) = OR(NOT a, b) → BIT_OR(1-a, b).
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "implies_logic"
                    and len(expr.args) == 2):
                a = self._lower_expr(expr.args[0])
                b = self._lower_expr(expr.args[1])
                if a is None or b is None:
                    return None
                one = self.builder.const_int(1)
                not_a = self.builder.emit(
                    tir.OpKind.SUB, one, a,
                    result_ty=tir.TIRScalar("i32"))
                return self.builder.emit(
                    tir.OpKind.BIT_OR, not_a, b,
                    result_ty=tir.TIRScalar("i32"))
            # eq_logic(a, b) = NOT XOR(a, b) → 1 - BIT_XOR(a, b).
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "eq_logic"
                    and len(expr.args) == 2):
                a = self._lower_expr(expr.args[0])
                b = self._lower_expr(expr.args[1])
                if a is None or b is None:
                    return None
                xor = self.builder.emit(
                    tir.OpKind.BIT_XOR, a, b,
                    result_ty=tir.TIRScalar("i32"))
                one = self.builder.const_int(1)
                return self.builder.emit(
                    tir.OpKind.SUB, one, xor,
                    result_ty=tir.TIRScalar("i32"))
            # if_logic(cond, then_v, else_v) → SELECT(cond_nonzero,
            # then_v, else_v). For Phase-0 we use CMP_NE against 0
            # then SELECT.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "if_logic"
                    and len(expr.args) == 3):
                c = self._lower_expr(expr.args[0])
                t = self._lower_expr(expr.args[1])
                e = self._lower_expr(expr.args[2])
                if c is None or t is None or e is None:
                    return None
                zero = self.builder.const_int(0)
                cond_nz = self.builder.emit(
                    tir.OpKind.CMP_NE, c, zero,
                    result_ty=tir.TIRScalar("i32"))
                return self.builder.emit(
                    tir.OpKind.SELECT, cond_nz, t, e,
                    result_ty=tir.TIRScalar("i32"))
            # to_logic_bool(x) → x (identity; Logic<T> wrapper has no
            # runtime representation in Phase-0).
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "to_logic_bool"
                    and len(expr.args) == 1):
                return self._lower_expr(expr.args[0])
            # Stage 37 Inc 1: tiered memory constructors + eliminators
            # + cross-tier transitions. All lower as identity (Phase-0:
            # TyMemTier wrapper has no runtime representation; tier
            # lives purely at the type system level — mirrors the
            # Stage 36 Logic<T> attach/detach pattern). Phase-1+ work
            # will add tier-id arena side-tables for runtime tracking.
            # Stage 38 Inc 1 adds spatial-frame constructors +
            # eliminators to this identity-lowering arm. Same Phase-0
            # zero-overhead pattern as Stage 37 tier ops.
            #
            # Stage 49 Inc 1 — Result<T,E> constructors and accessors
            # split out of the identity tuple and emit real packed-tag
            # IR (RESULT_PACK / RESULT_TAG / RESULT_PAYLOAD). See the
            # convention block on OpKind.RESULT_PACK in tir.py.
            #
            # Ok(v)        -> RESULT_PACK(const_int(0), v)            : i64
            # Err(e)       -> RESULT_PACK(const_int(1), e)            : i64
            # unwrap_ok(r) -> RESULT_PAYLOAD(r)                       : i32
            # unwrap_err(r)-> RESULT_PAYLOAD(r)                       : i32
            # __try(r)     -> RESULT_PAYLOAD(r)  (Inc 1 placeholder;
            #                 the real conditional-branch IR ships in
            #                 Inc 4 — for now `__try` extracts the Ok
            #                 inner same as unwrap_ok, preserving the
            #                 Phase-0 dogfood_17 exit-42 invariant on
            #                 the static-Ok pathway.)
            #
            # Note: unwrap_ok and unwrap_err currently emit the same IR.
            # The Stage 46 typecheck guards (constructor-provenance
            # check) catch wrong-arm calls on STATICALLY-known Results
            # at compile time. The runtime tag check that distinguishes
            # them on DYNAMIC Results (call-returns) is deferred to
            # Inc 1.5 / Inc 2 to keep this increment small. Until then,
            # unwrap_err on a runtime-Ok Result extracts the i32 payload
            # silently — a known semantic gap that the typecheck arm
            # currently blocks at the source level.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "Ok"
                    and len(expr.args) == 1):
                payload = self._lower_expr(expr.args[0])
                if payload is None:
                    return None
                tag = self.builder.const_int(0, "i32")
                return self.builder.emit(
                    tir.OpKind.RESULT_PACK, tag, payload,
                    result_ty=tir.TIRScalar("i64"))
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "Err"
                    and len(expr.args) == 1):
                payload = self._lower_expr(expr.args[0])
                if payload is None:
                    return None
                tag = self.builder.const_int(1, "i32")
                return self.builder.emit(
                    tir.OpKind.RESULT_PACK, tag, payload,
                    result_ty=tir.TIRScalar("i64"))
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name in ("unwrap_ok", "unwrap_err")
                    and len(expr.args) == 1):
                packed = self._lower_expr(expr.args[0])
                if packed is None:
                    return None
                # Inc 1: payload extract only — runtime tag-check on
                # wrong-arm (Stage 49 Inc 1.5 / later) still pending,
                # so unwrap_ok / unwrap_err currently extract the
                # low-32 payload without verifying the tag matches.
                # Static-provenance checks at typecheck already block
                # the most common wrong-arm cases.
                return self.builder.emit(
                    tir.OpKind.RESULT_PAYLOAD, packed,
                    result_ty=tir.TIRScalar("i32"))
            # Stage 49 Inc 4: `?` (parsed as __try) becomes a real
            # conditional early-return. If the operand is Err
            # (tag == 1), return the entire packed Result up the
            # call stack — the enclosing fn must return Result<U, E2>
            # with E2 compatible with the operand's E1 (already
            # validated by the Stage 48 typecheck arm). If the
            # operand is Ok (tag == 0), extract the payload and
            # continue with the user's code in the ok-fall-through
            # block.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "__try"
                    and len(expr.args) == 1):
                packed = self._lower_expr(expr.args[0])
                if packed is None:
                    return None
                tag = self.builder.emit(
                    tir.OpKind.RESULT_TAG, packed,
                    result_ty=tir.TIRScalar("i32"))
                one = self.builder.const_int(1, "i32")
                is_err_cond = self.builder.emit(
                    tir.OpKind.CMP_EQ, tag, one,
                    result_ty=tir.TIRScalar("bool"))
                err_blk = self.builder.append_block()
                ok_blk = self.builder.append_block()
                self.builder.emit(
                    tir.OpKind.COND_BR, is_err_cond,
                    attrs={"true_block": err_blk.id,
                           "false_block": ok_blk.id})
                # Err arm: return the packed Result from the
                # enclosing fn. The return value matches the fn's
                # signature (i64 packed Result).
                self.builder.switch_to(err_blk)
                self.builder.emit(tir.OpKind.RETURN, packed)
                # Ok arm: extract the payload (i32). Subsequent
                # lowering for code following `r?` runs in this
                # block, which is the natural fall-through.
                self.builder.switch_to(ok_blk)
                return self.builder.emit(
                    tir.OpKind.RESULT_PAYLOAD, packed,
                    result_ty=tir.TIRScalar("i32"))
            # Stage 49 Inc 2 — is_ok / is_err lower to a tag
            # extract + compare-equal against the expected tag
            # constant. is_ok(r) iff RESULT_TAG(r) == 0;
            # is_err(r) iff RESULT_TAG(r) == 1. Returns a bool
            # (lowered as i32 0/1 in TIR per Helix's existing
            # bool convention).
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name in ("is_ok", "is_err")
                    and len(expr.args) == 1):
                packed = self._lower_expr(expr.args[0])
                if packed is None:
                    return None
                tag = self.builder.emit(
                    tir.OpKind.RESULT_TAG, packed,
                    result_ty=tir.TIRScalar("i32"))
                expected_tag = self.builder.const_int(
                    0 if expr.callee.name == "is_ok" else 1, "i32")
                return self.builder.emit(
                    tir.OpKind.CMP_EQ, tag, expected_tag,
                    result_ty=tir.TIRScalar("bool"))
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name in (
                        "into_working", "into_episodic",
                        "into_semantic", "into_procedural",
                        "unwrap_working", "unwrap_episodic",
                        "unwrap_semantic", "unwrap_procedural",
                        "consolidate", "recall",
                        "into_world", "into_robot", "into_camera",
                        "from_world", "from_robot", "from_camera",
                        # Stage 38 Inc 2 — cross-frame transforms
                        # also lower as identity (Phase-0: actual
                        # transform math is Phase-1+).
                        "world_to_robot", "robot_to_world",
                        "robot_to_camera", "camera_to_robot",
                        "world_to_camera", "camera_to_world",
                        # Stage 39 Inc 1 — temporal constructors +
                        # eliminators lower as identity. Phase-0:
                        # temporal kind lives at the type system level
                        # — zero runtime overhead. Mirrors Stage 37/38.
                        "into_past", "into_present",
                        "into_future", "into_eternal",
                        "from_past", "from_present",
                        "from_future", "from_eternal",
                        # Stage 39 Inc 2 — cross-temporal transitions.
                        # Also identity at Phase-0; intent-only.
                        "to_past", "forecast",
                        "recall_past", "actualize",
                        # Stage 40 Inc 1 — modal/epistemic
                        # constructors + eliminators lower as
                        # identity (Phase-0: modal kind lives at
                        # the type system level only).
                        "into_known", "into_believed",
                        "into_goal", "into_uncertain",
                        "from_known", "from_believed",
                        "from_goal", "from_uncertain",
                        # Stage 40 Inc 2 — modal transitions
                        # (epistemic upgrades).
                        "confirm", "act_on",
                        # Stage 41 Inc 1 — causal/intent
                        # constructors + eliminators lower as
                        # identity (Phase-0: causal kind lives
                        # at the type system level only).
                        "into_cause", "into_effect",
                        "into_joint", "into_independent",
                        "from_cause", "from_effect",
                        "from_joint", "from_independent",
                        # Stage 41 Inc 2 — causal transitions.
                        "propagate", "aggregate", "isolate",
                        # Stage 46 Inc 1 — Result<T,E> constructors
                        # + value-preserving accessors USED to live
                        # here as identity-lowered ops. Stage 49 Inc 1
                        # split them into their own arm above that
                        # emits real RESULT_PACK / RESULT_PAYLOAD IR
                        # with packed-i64 representation. `Ok`, `Err`,
                        # `unwrap_ok`, `unwrap_err`, and `__try` are
                        # all handled by that arm now. `is_ok` /
                        # `is_err` / `map_err` remain typecheck-
                        # rejected (Stage 46 F1/F2) until Inc 2/3
                        # of Stage 49 wire their lowering.
                        )
                    and len(expr.args) == 1):
                return self._lower_expr(expr.args[0])
            # Stage 49 Inc 2 lifted the is_ok / is_err typecheck
            # reject (handled in the dedicated arm above). Inc 3
            # lifts the map_err reject and ALSO upgrades map_ok
            # from a Phase-0 thread-through to a proper packed-i64
            # Result transform.
            #
            # map_ok(r, new_v):
            #   if Ok(r):  Ok(new_v)  = RESULT_PACK(0, new_v)
            #   else:      r          unchanged (Err passes through)
            # map_err(r, new_e):
            #   if Err(r): Err(new_e) = RESULT_PACK(1, new_e)
            #   else:      r          unchanged (Ok passes through)
            #
            # Both use SELECT on the tag-equality comparison; the
            # SELECT then chooses between the freshly-packed
            # replacement and the original packed i64.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name in ("map_ok", "map_err")
                    and len(expr.args) == 2):
                r_packed = self._lower_expr(expr.args[0])
                new_val = self._lower_expr(expr.args[1])
                if r_packed is None or new_val is None:
                    return None
                tag = self.builder.emit(
                    tir.OpKind.RESULT_TAG, r_packed,
                    result_ty=tir.TIRScalar("i32"))
                # For map_ok: replace when tag == 0 (Ok side).
                # For map_err: replace when tag == 1 (Err side).
                # Build the replacement packed-i64 with the matching
                # tag value (0 for Ok-side, 1 for Err-side).
                replace_tag_value = (
                    0 if expr.callee.name == "map_ok" else 1)
                replace_tag = self.builder.const_int(
                    replace_tag_value, "i32")
                cond = self.builder.emit(
                    tir.OpKind.CMP_EQ, tag, replace_tag,
                    result_ty=tir.TIRScalar("bool"))
                new_packed = self.builder.emit(
                    tir.OpKind.RESULT_PACK, replace_tag, new_val,
                    result_ty=tir.TIRScalar("i64"))
                return self.builder.emit(
                    tir.OpKind.SELECT, cond, new_packed, r_packed,
                    result_ty=tir.TIRScalar("i64"))
            # Stage 36 Increment 5: real two-parent provenance via
            # arena side-table.
            #
            # Stage 36 Inc 9 audit A2 (silent-failure lane) fix:
            # register_derivation now returns (arena_index + 1) so
            # handle 0 is reserved as the "null derivation" sentinel.
            # Pre-fix, if a user stored handles in a side array
            # initialized to 0, reading back 0 was indistinguishable
            # from "derivation that happens to live at arena index 0"
            # — silent corruption. Post-fix, handle 0 always means
            # "no derivation" (parent_left_at(0) returns -1 via the
            # existing bounds-check fix from A1, since 0 - 1 = -1
            # fails the >= 0 check).
            #
            # Stage 36 Inc 9 type-design A2 fix: the two pushes are
            # now emitted as a single ARENA_PUSH_PAIR op. Prior
            # implementation used two consecutive ARENA_PUSH ops with
            # no data dependency between them — any IR pass that
            # reorders side-effectful ops, or any other arena consumer
            # scheduled in between (struct lowering, MatchDispatch,
            # inlined-arg ARENA_PUSH), would have broken the
            # "left at N, right at N+1" handle invariant. The fused
            # opcode is atomic at IR level: DCE/CSE/scheduler cannot
            # split it.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "register_derivation"
                    and len(expr.args) == 2):
                l = self._lower_expr(expr.args[0])
                r = self._lower_expr(expr.args[1])
                if l is None or r is None:
                    return None
                push_idx = self.builder.emit(
                    tir.OpKind.ARENA_PUSH_PAIR, l, r,
                    result_ty=tir.TIRScalar("i32"))
                # Return push_idx + 1 so handles are 1-based; 0 means
                # "null". parent_*_at subtracts 1 before lookup.
                one = self.builder.const_int(1)
                return self.builder.emit(
                    tir.OpKind.ADD, push_idx, one,
                    result_ty=tir.TIRScalar("i32"))
            # Stage 36 Inc 14: three-parent provenance. Same shape as
            # register_derivation but lowers to the atomic three-slot
            # ARENA_PUSH_TRIPLE. The handle is the (1-based) slot index
            # of the left value; middle lives at handle+1-1+1=handle+1
            # (i.e., the slot reachable via parent_at(handle, 1)).
            # ARENA_PUSH_TRIPLE returns -1 on arena overflow; in that
            # case push_idx + 1 = 0 = the null-handle sentinel — same
            # fail-closed contract that ARENA_PUSH_PAIR already gives.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "register_derivation3"
                    and len(expr.args) == 3):
                l = self._lower_expr(expr.args[0])
                m = self._lower_expr(expr.args[1])
                r = self._lower_expr(expr.args[2])
                if l is None or m is None or r is None:
                    return None
                push_idx = self.builder.emit(
                    tir.OpKind.ARENA_PUSH_TRIPLE, l, m, r,
                    result_ty=tir.TIRScalar("i32"))
                one = self.builder.const_int(1)
                return self.builder.emit(
                    tir.OpKind.ADD, push_idx, one,
                    result_ty=tir.TIRScalar("i32"))
            # Stage 36 Increment 9 post-Inc-8 audit A1 HIGH fix:
            # parent_left_at(idx) and parent_right_at(idx) previously
            # lowered to bare ARENA_GET with no bounds check. A user
            # passing an arbitrary i32 (negative, zero before any
            # register_derivation, or > arena_len) silently returned
            # whatever bit pattern the arena held — the exact forged-
            # handle pattern swept clean in restart 45-47 for AGI
            # typed handles. Now: read returns -1 sentinel on
            # out-of-range, with the underlying ARENA_GET using a
            # clamped index so no out-of-bounds memory access occurs
            # under any user input.
            def _safe_arena_get(idx_v, offset: int):
                """Emit a bounds-checked ARENA_GET. Returns -1 if
                (idx + offset) is outside [0, arena_len). Uses only
                CMP_* + SELECT + arithmetic — MINIMUM/MAXIMUM are
                tensor-shaped ops that don't work as i32 scalars."""
                arena_len = self.builder.emit(
                    tir.OpKind.ARENA_LEN,
                    result_ty=tir.TIRScalar("i32"))
                if offset != 0:
                    off_v = self.builder.const_int(offset)
                    eff_idx = self.builder.emit(
                        tir.OpKind.ADD, idx_v, off_v,
                        result_ty=tir.TIRScalar("i32"))
                else:
                    eff_idx = idx_v
                zero = self.builder.const_int(0)
                one = self.builder.const_int(1)
                # in_bounds = (eff_idx >= 0) AND (eff_idx < arena_len)
                ge_zero = self.builder.emit(
                    tir.OpKind.CMP_GE, eff_idx, zero,
                    result_ty=tir.TIRScalar("i32"))
                lt_len = self.builder.emit(
                    tir.OpKind.CMP_LT, eff_idx, arena_len,
                    result_ty=tir.TIRScalar("i32"))
                in_bounds = self.builder.emit(
                    tir.OpKind.BIT_AND, ge_zero, lt_len,
                    result_ty=tir.TIRScalar("i32"))
                # safe_idx = clamp(eff_idx, 0, max(arena_len - 1, 0))
                # via SELECT: the speculative ARENA_GET reads from a
                # never-OOB index; SELECT then gates the result on
                # in_bounds and returns -1 if out of range.
                len_minus_1 = self.builder.emit(
                    tir.OpKind.SUB, arena_len, one,
                    result_ty=tir.TIRScalar("i32"))
                len_pos = self.builder.emit(
                    tir.OpKind.CMP_GE, len_minus_1, zero,
                    result_ty=tir.TIRScalar("i32"))
                hi_clamp = self.builder.emit(
                    tir.OpKind.SELECT, len_pos, len_minus_1, zero,
                    result_ty=tir.TIRScalar("i32"))
                # clamp eff_idx down to hi_clamp if too high
                idx_le_hi = self.builder.emit(
                    tir.OpKind.CMP_LE, eff_idx, hi_clamp,
                    result_ty=tir.TIRScalar("i32"))
                clamped_hi = self.builder.emit(
                    tir.OpKind.SELECT, idx_le_hi, eff_idx, hi_clamp,
                    result_ty=tir.TIRScalar("i32"))
                # clamp up to 0 if negative
                clamped_ge_zero = self.builder.emit(
                    tir.OpKind.CMP_GE, clamped_hi, zero,
                    result_ty=tir.TIRScalar("i32"))
                safe_idx = self.builder.emit(
                    tir.OpKind.SELECT, clamped_ge_zero, clamped_hi, zero,
                    result_ty=tir.TIRScalar("i32"))
                val = self.builder.emit(
                    tir.OpKind.ARENA_GET, safe_idx,
                    result_ty=tir.TIRScalar("i32"))
                # Sentinel -1 on out-of-range; otherwise the value.
                neg_one = self.builder.const_int(-1)
                return self.builder.emit(
                    tir.OpKind.SELECT, in_bounds, val, neg_one,
                    result_ty=tir.TIRScalar("i32"))

            # Stage 36 Inc 9 audit A2 fix: parent_*_at subtract 1
            # from the user-visible 1-based handle before arena lookup.
            # Handle 0 (null sentinel) effectively becomes arena index
            # -1 which falls through to the -1 OOB sentinel via the
            # bounds-check from A1.
            # Stage 37 post-closure correction (retroactive Stage 36
            # closure gate-3 fix; "Inc 4" name collides with the
            # concurrent Stage 37 closure commit, so named explicitly):
            # the closure gate-3 type-design audit (H1, conf 95) found that
            # parent_right_at(0) silently leaked arena[0] because the
            # _safe_arena_get bounds check fires on eff_idx (= 0-1+1 = 0,
            # in-bounds) rather than on the original handle. Inc 15's
            # uniform `handle <= 0 → -1` guard was applied only to
            # parent_at; parent_right_at perpetuated the silent-leak.
            # Apply the same SELECT-on-invalid guard here, and to
            # parent_left_at for explicit family symmetry (its current
            # safety is accidental — SUB-1 with offset 0 lands on
            # negative eff_idx that _safe_arena_get clamps to -1).
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "parent_left_at"
                    and len(expr.args) == 1):
                idx = self._lower_expr(expr.args[0])
                if idx is None:
                    return None
                zero = self.builder.const_int(0)
                one = self.builder.const_int(1)
                neg_one = self.builder.const_int(-1)
                handle_valid = self.builder.emit(
                    tir.OpKind.CMP_GT, idx, zero,
                    result_ty=tir.TIRScalar("i32"))
                base_idx = self.builder.emit(
                    tir.OpKind.SUB, idx, one,
                    result_ty=tir.TIRScalar("i32"))
                raw_read = _safe_arena_get(base_idx, 0)
                return self.builder.emit(
                    tir.OpKind.SELECT, handle_valid, raw_read, neg_one,
                    result_ty=tir.TIRScalar("i32"))
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "parent_right_at"
                    and len(expr.args) == 1):
                idx = self._lower_expr(expr.args[0])
                if idx is None:
                    return None
                zero = self.builder.const_int(0)
                one = self.builder.const_int(1)
                neg_one = self.builder.const_int(-1)
                handle_valid = self.builder.emit(
                    tir.OpKind.CMP_GT, idx, zero,
                    result_ty=tir.TIRScalar("i32"))
                base_idx = self.builder.emit(
                    tir.OpKind.SUB, idx, one,
                    result_ty=tir.TIRScalar("i32"))
                raw_read = _safe_arena_get(base_idx, 1)
                return self.builder.emit(
                    tir.OpKind.SELECT, handle_valid, raw_read, neg_one,
                    result_ty=tir.TIRScalar("i32"))
            # Stage 36 Inc 14: generic indexed parent accessor.
            # parent_at(handle, slot) reads arena slot (handle-1+slot)
            # with the same bounds-check sentinel as parent_*_at. The
            # `slot` operand is dynamic (an i32 value, not a literal),
            # so the +offset is computed via ADD rather than baked into
            # the displacement. parent_at(h, 0) ≡ parent_left_at(h);
            # parent_at(h, 1) ≡ parent_right_at(h). parent_at(h, 2) is
            # only meaningful for handles registered via
            # register_derivation3 (the three-parent variant); for
            # two-parent handles it reads into whatever happens to live
            # at slot N+2, which may be another derivation's slot or
            # the OOB sentinel.
            #
            # Stage 36 Inc 15 (silent-failure H1 partial closure):
            # - Static reject literal `slot < 0` or `slot > 2` lives at
            #   typecheck (typecheck.py:parent_at clause).
            # - Runtime guard for `handle <= 0` (null sentinel) returns
            #   -1 directly, defeating the audit's hidden-error #3
            #   (parent_at(0, 1) silently reading arena[0]).
            # - Runtime guard for dynamic `slot < 0` returns -1 directly,
            #   defeating the audit's hidden-error #2 (negative slot
            #   shifting eff_idx back into a previous record).
            # TODO(stage36-inc16-arity-in-handle): the remaining
            # cross-record hazard (slot >= arity-of(handle)) requires
            # a per-record arity word in the arena layout. See audit
            # docs/audit-stage36-postinc14-silent-failures.md#H1.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "parent_at"
                    and len(expr.args) == 2):
                handle = self._lower_expr(expr.args[0])
                slot = self._lower_expr(expr.args[1])
                if handle is None or slot is None:
                    return None
                zero = self.builder.const_int(0)
                one = self.builder.const_int(1)
                neg_one = self.builder.const_int(-1)
                # Inc 15 guard 1: handle <= 0 → -1.
                handle_valid = self.builder.emit(
                    tir.OpKind.CMP_GT, handle, zero,
                    result_ty=tir.TIRScalar("i32"))
                # Inc 15 guard 2: slot < 0 → -1.
                slot_valid = self.builder.emit(
                    tir.OpKind.CMP_GE, slot, zero,
                    result_ty=tir.TIRScalar("i32"))
                # Stage 36 clean-gate-1 audit A1 fix (HIGH): mirror the
                # static-typecheck literal-slot upper bound at runtime.
                # Pre-fix, dynamic `slot > 2` silently read sibling-
                # record data via the unchecked-upper path. The typecheck
                # rejects literal slot >= 3; this runtime guard catches
                # the dynamic case. (The full cross-record fix needs
                # per-handle arity word — deferred to Inc 16.)
                three = self.builder.const_int(3)
                slot_lt_3 = self.builder.emit(
                    tir.OpKind.CMP_LT, slot, three,
                    result_ty=tir.TIRScalar("i32"))
                slot_ok = self.builder.emit(
                    tir.OpKind.BIT_AND, slot_valid, slot_lt_3,
                    result_ty=tir.TIRScalar("i32"))
                guards_pass = self.builder.emit(
                    tir.OpKind.BIT_AND, handle_valid, slot_ok,
                    result_ty=tir.TIRScalar("i32"))
                base_idx = self.builder.emit(
                    tir.OpKind.SUB, handle, one,
                    result_ty=tir.TIRScalar("i32"))
                eff_idx = self.builder.emit(
                    tir.OpKind.ADD, base_idx, slot,
                    result_ty=tir.TIRScalar("i32"))
                raw_read = _safe_arena_get(eff_idx, 0)
                # If guards fail, return -1; else return the (already
                # bounds-checked) arena read.
                return self.builder.emit(
                    tir.OpKind.SELECT, guards_pass, raw_read, neg_one,
                    result_ty=tir.TIRScalar("i32"))
            # Stage 36 Inc 9 A3 (silent-failure HIGH) fix: clamp
            # fuzzy_* inputs to [0, 1] at IR lowering. Pre-fix, an
            # out-of-range input (e.g., a=2.0 from optimizer drift)
            # silently produced fuzzy_or = 3.0 with gradient 2.0 — no
            # diagnostic, optimizer diverges silently. Post-fix, each
            # input is clamped to [0, 1] before the algebraic form via
            # SELECT + CMP_GE/CMP_LE.
            # The AD chain rules still operate at the AST level and
            # see the unclamped formula — for in-range inputs (the
            # common case) gradients are unchanged; for out-of-range
            # inputs the chain rule gives the unclamped derivative,
            # which is mathematically useful for SGD to steer inputs
            # back into [0, 1]. (Strict mathematical-fuzzy semantics
            # would emit a 0 gradient inside the clamp region; the
            # Phase-0 tradeoff prefers the "useful for SGD recovery"
            # behaviour.)
            def _clamp_unit_f32(x):
                """Clamp x to [0.0, 1.0] using SELECT + CMP."""
                zero_f = self.builder.const_float(0.0, dtype="f32")
                one_f = self.builder.const_float(1.0, dtype="f32")
                ge_zero = self.builder.emit(
                    tir.OpKind.CMP_GE, x, zero_f,
                    result_ty=tir.TIRScalar("i32"))
                low_clamped = self.builder.emit(
                    tir.OpKind.SELECT, ge_zero, x, zero_f,
                    result_ty=tir.TIRScalar("f32"))
                le_one = self.builder.emit(
                    tir.OpKind.CMP_LE, low_clamped, one_f,
                    result_ty=tir.TIRScalar("i32"))
                return self.builder.emit(
                    tir.OpKind.SELECT, le_one, low_clamped, one_f,
                    result_ty=tir.TIRScalar("f32"))

            # Stage 36 Increment 6: fuzzy logic operators over
            # Logic<f32>. Product semantics for AND, probabilistic OR.
            # All three lower to MUL/ADD/SUB so the existing AD chain
            # rules apply — grad() flows through them automatically.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "fuzzy_and"
                    and len(expr.args) == 2):
                a = self._lower_expr(expr.args[0])
                b = self._lower_expr(expr.args[1])
                if a is None or b is None:
                    return None
                a = _clamp_unit_f32(a)
                b = _clamp_unit_f32(b)
                # fuzzy_and(a, b) = a * b
                return self.builder.emit(
                    tir.OpKind.MUL, a, b,
                    result_ty=tir.TIRScalar("f32"))
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "fuzzy_or"
                    and len(expr.args) == 2):
                a = self._lower_expr(expr.args[0])
                b = self._lower_expr(expr.args[1])
                if a is None or b is None:
                    return None
                a = _clamp_unit_f32(a)
                b = _clamp_unit_f32(b)
                # fuzzy_or(a, b) = a + b - a*b (probabilistic sum)
                sum_ab = self.builder.emit(
                    tir.OpKind.ADD, a, b,
                    result_ty=tir.TIRScalar("f32"))
                prod_ab = self.builder.emit(
                    tir.OpKind.MUL, a, b,
                    result_ty=tir.TIRScalar("f32"))
                return self.builder.emit(
                    tir.OpKind.SUB, sum_ab, prod_ab,
                    result_ty=tir.TIRScalar("f32"))
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "fuzzy_not"
                    and len(expr.args) == 1):
                a = self._lower_expr(expr.args[0])
                if a is None:
                    # Stage 36 clean-gate-1 audit B1 fix (LOW): grep
                    # symmetry with the Inc 13 return-None convention
                    # used by parent_*_at and the other 2-arg builtin
                    # arms. Functionally identical (a is None already).
                    return None
                a = _clamp_unit_f32(a)
                # fuzzy_not(a) = 1.0 - a
                one = self.builder.const_float(1.0, dtype="f32")
                return self.builder.emit(
                    tir.OpKind.SUB, one, a,
                    result_ty=tir.TIRScalar("f32"))
            # Stage 36 Increment 8: fuzzy_xor + fuzzy_implies.
            # fuzzy_xor(a, b) = a + b - 2*a*b
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "fuzzy_xor"
                    and len(expr.args) == 2):
                a = self._lower_expr(expr.args[0])
                b = self._lower_expr(expr.args[1])
                if a is None or b is None:
                    return None
                a = _clamp_unit_f32(a)
                b = _clamp_unit_f32(b)
                sum_ab = self.builder.emit(
                    tir.OpKind.ADD, a, b,
                    result_ty=tir.TIRScalar("f32"))
                prod_ab = self.builder.emit(
                    tir.OpKind.MUL, a, b,
                    result_ty=tir.TIRScalar("f32"))
                two = self.builder.const_float(2.0, dtype="f32")
                two_prod = self.builder.emit(
                    tir.OpKind.MUL, two, prod_ab,
                    result_ty=tir.TIRScalar("f32"))
                return self.builder.emit(
                    tir.OpKind.SUB, sum_ab, two_prod,
                    result_ty=tir.TIRScalar("f32"))
            # fuzzy_implies(a, b) = 1 - a + a*b
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "fuzzy_implies"
                    and len(expr.args) == 2):
                a = self._lower_expr(expr.args[0])
                b = self._lower_expr(expr.args[1])
                if a is None or b is None:
                    return None
                a = _clamp_unit_f32(a)
                b = _clamp_unit_f32(b)
                one = self.builder.const_float(1.0, dtype="f32")
                one_minus_a = self.builder.emit(
                    tir.OpKind.SUB, one, a,
                    result_ty=tir.TIRScalar("f32"))
                prod_ab = self.builder.emit(
                    tir.OpKind.MUL, a, b,
                    result_ty=tir.TIRScalar("f32"))
                return self.builder.emit(
                    tir.OpKind.ADD, one_minus_a, prod_ab,
                    result_ty=tir.TIRScalar("f32"))
            # Stage 16.5: "literal".as_ptr() — emit STR_PTR op that resolves
            # to a `lea rax, [rip + sym]` of the literal's bytes. The result
            # is a u64 raw pointer suitable for FFI calls.
            if (isinstance(expr.callee, A.Field)
                    and expr.callee.name == "as_ptr"
                    and isinstance(expr.callee.obj, A.StrLit)
                    and len(expr.args) == 0):
                s = expr.callee.obj.value
                return self.builder.emit(
                    tir.OpKind.STR_PTR,
                    result_ty=tir.TIRScalar("u64"),
                    attrs={"text": s})
            # Stage 15: tile.get(row, col). Lowers to LOAD_ELEM at
            # row * cols + col. Requires the tile binding to be in scope
            # (registered via tile<>:: literal or tile_matmul).
            if (isinstance(expr.callee, A.Field)
                    and expr.callee.name == "get"
                    and isinstance(expr.callee.obj, A.Name)):
                tile_name = expr.callee.obj.name
                shape = self._lookup_tile(tile_name)
                if shape is not None:
                    rows, cols = shape
                    if len(expr.args) != 2:
                        raise NotImplementedError(
                            "tile.get() takes exactly 2 args (row, col)"
                        )
                    # Resolve the row*cols+col offset. Phase-0: prefer the
                    # const-fold path when both args are IntLit.
                    a0, a1 = expr.args
                    if isinstance(a0, A.IntLit) and isinstance(a1, A.IntLit):
                        idx_val = a0.value * cols + a1.value
                        idx_v = self.builder.const_int(idx_val)
                    else:
                        row_v = self._lower_expr(a0) or self.builder.const_int(0)
                        col_v = self._lower_expr(a1) or self.builder.const_int(0)
                        cols_v = self.builder.const_int(cols)
                        prod = self.builder.emit(tir.OpKind.MUL, row_v, cols_v,
                                                 result_ty=tir.TIRScalar("i32"))
                        idx_v = self.builder.emit(tir.OpKind.ADD, prod, col_v,
                                                  result_ty=tir.TIRScalar("i32"))
                    arr = self._lookup_array(tile_name)
                    elem_ty = arr[0] if arr is not None else tir.TIRScalar("f32")
                    return self.builder.emit(tir.OpKind.LOAD_ELEM, idx_v,
                                             result_ty=elem_ty,
                                             attrs={"name": tile_name})
            # Recursive enum constructor as a value expression (i.e. NOT
            # a fn arg, but appearing as the result of a match arm body
            # or a let value in expression position). Push slots into
            # the arena and return the start index as the value.
            enum_variant = self._enum_variant_for_expr(expr.callee)
            if enum_variant is not None:
                ename, vname, tag = enum_variant
                if ename in self._recursive_enums:
                    tag_v = self.builder.const_int(tag)
                    arg_vals = self._lower_enum_payload_args(
                        ename, vname, expr.args)
                    return self._arena_push_slots([tag_v] + arg_vals)
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
            # Stage 28.5 — `panic("msg")` lowers to a TRAP op (kind
            # ctrl.trap) carrying the message string and trap id 28501.
            # Backend writes the message to stderr and exits non-zero.
            # The op produces an i32 result for SSA bookkeeping, but the
            # value is never observed — execution aborts before any
            # subsequent op runs.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "panic"
                    and len(expr.args) == 1
                    and isinstance(expr.args[0], A.StrLit)):
                from ..frontend.panic_pass import TRAP_PANIC_INVOKED
                s = expr.args[0].value
                return self.builder.emit(
                    tir.OpKind.TRAP,
                    result_ty=tir.TIRScalar("i32"),
                    attrs={"text": s, "trap_id": TRAP_PANIC_INVOKED})
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
            # Stage 16 — GPU kernel builtins. Only legal inside @kernel fns.
            # `thread_idx()` returns the thread's x-dim index (i32). Lowers to
            # a THREAD_IDX TIR op which PTX backend maps to `mov.u32 %r, %tid.x`.
            # `thread_idx_y()` / `thread_idx_z()` are the y/z analogues.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name in ("thread_idx", "thread_idx_x",
                                              "thread_idx_y", "thread_idx_z")
                    and len(expr.args) == 0):
                if not self._in_kernel:
                    # Trap-id 96001: thread_idx() outside @kernel.
                    raise SyntaxError(
                        f"trap 96001: {expr.callee.name}() only valid inside "
                        "@kernel fn")
                dim = "x" if expr.callee.name in ("thread_idx", "thread_idx_x") \
                      else ("y" if expr.callee.name == "thread_idx_y" else "z")
                return self.builder.emit(
                    tir.OpKind.THREAD_IDX,
                    result_ty=tir.TIRScalar("i32"),
                    attrs={"dim": dim, "sreg": "tid"})
            # Stage 16 — `block_idx()` / `block_dim()` companions to thread_idx.
            # These return %ctaid.x and %ntid.x respectively. Same x/y/z variants
            # available.
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name in ("block_idx", "block_idx_x",
                                              "block_idx_y", "block_idx_z",
                                              "block_dim", "block_dim_x",
                                              "block_dim_y", "block_dim_z")
                    and len(expr.args) == 0):
                if not self._in_kernel:
                    raise SyntaxError(
                        f"trap 96001: {expr.callee.name}() only valid inside "
                        "@kernel fn")
                # Pick PTX special-reg by builtin name.
                name = expr.callee.name
                if name.startswith("block_idx"):
                    sreg = "ctaid"
                else:
                    sreg = "ntid"
                if name.endswith("_y"):
                    dim = "y"
                elif name.endswith("_z"):
                    dim = "z"
                else:
                    dim = "x"
                return self.builder.emit(
                    tir.OpKind.THREAD_IDX,
                    result_ty=tir.TIRScalar("i32"),
                    attrs={"dim": dim, "sreg": sreg})
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
                # f32/f64 bit-reinterpret. Used by Helix-side stdlib to
                # store float bit patterns in the arena (which is i32-typed).
                # bits_of: f32->i32 / f64->i64 (just relabel the same 4/8 bytes)
                # from_bits: i32->f32 / i64->f64
                if bn == "__bits_of_f32" and len(expr.args) == 1:
                    v = self._lower_expr(expr.args[0])
                    return self.builder.emit(
                        tir.OpKind.BITCAST, v,
                        result_ty=tir.TIRScalar("i32"))
                if bn == "__f32_from_bits" and len(expr.args) == 1:
                    v = self._lower_expr(expr.args[0])
                    return self.builder.emit(
                        tir.OpKind.BITCAST, v,
                        result_ty=tir.TIRScalar("f32"))
                if bn == "__bits_of_f64" and len(expr.args) == 1:
                    v = self._lower_expr(expr.args[0])
                    return self.builder.emit(
                        tir.OpKind.BITCAST, v,
                        result_ty=tir.TIRScalar("i64"))
                if bn == "__f64_from_bits" and len(expr.args) == 1:
                    v = self._lower_expr(expr.args[0])
                    return self.builder.emit(
                        tir.OpKind.BITCAST, v,
                        result_ty=tir.TIRScalar("f64"))
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
            # read_file_to_arena: opens path, reads up to 1 MB, pushes
            # each byte to the arena (one slot per byte). Returns count of
            # bytes pushed. Implementation in x86_64 backend is full; the
            # bootstrap pipeline test exercises it.
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
                    and expr.callee.name == "splice_f64"
                    and len(expr.args) == 1):
                handle = self._lower_expr(expr.args[0]) or self.builder.const_int(0)
                return self.builder.emit(tir.OpKind.SPLICE, handle,
                                         result_ty=tir.TIRScalar("f64"),
                                         attrs={"value_kind": "f64"})
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
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name == "modify_f64"
                    and len(expr.args) == 3):
                target_v = self._lower_expr(expr.args[0]) or self.builder.const_int(0)
                xform_v = self._lower_expr(expr.args[1]) or self.builder.const_int(0)
                attrs: dict[str, object] = {"value_kind": "f64"}
                v_arg = expr.args[2]
                if (isinstance(v_arg, A.Name)
                        and v_arg.name in self.functions
                        and self._lookup(v_arg.name) is None
                        and self._lookup_mut(v_arg.name) is None
                        and self._verifier_abi_matches_f64(v_arg.name)):
                    attrs["verifier_fn"] = v_arg.name
                    placeholder = self.builder.const_int(0)
                    return self.builder.emit(tir.OpKind.MODIFY,
                                             target_v, xform_v, placeholder,
                                             result_ty=tir.TIRScalar("i32"),
                                             attrs=attrs)
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
                    p_ty = self._resolve_type_alias_node(
                        callee_ast.params[i].ty)
                    # Recursive-enum-typed param: callee expects a SINGLE
                    # i32 (the arena index). For inline constructors, we
                    # must arena-push and pass the resulting index — NOT
                    # expand into flat slots like the non-recursive case.
                    if (isinstance(p_ty, A.TyName)
                            and p_ty.name in self._recursive_enums):
                        if isinstance(a, A.Call):
                            enum_variant = self._enum_variant_for_expr(
                                a.callee)
                            if enum_variant is not None:
                                ename2, vname2, tag = enum_variant
                                if ename2 != p_ty.name:
                                    raise NotImplementedError(
                                        f"aggregate argument for parameter "
                                        f"'{callee_ast.params[i].name}' "
                                        f"expects {p_ty.name}, got inline "
                                        f"enum constructor {ename2}; run "
                                        f"typecheck first"
                                    )
                                tag_v = self.builder.const_int(tag)
                                arg_vals = self._lower_enum_payload_args(
                                    ename2, vname2, a.args)
                                args.append(
                                    self._arena_push_slots([tag_v] + arg_vals))
                                expanded = True
                            if not expanded:
                                call_rec = self._recursive_enum_name_for_expr(a)
                                if call_rec is not None:
                                    if call_rec != p_ty.name:
                                        raise NotImplementedError(
                                            f"aggregate argument for parameter "
                                            f"'{callee_ast.params[i].name}' "
                                            f"expects {p_ty.name}, got "
                                            f"function returning {call_rec}; "
                                            f"run typecheck first"
                                        )
                                    v = self._lower_expr(
                                        a, expected_rec_enum=p_ty.name)
                                    args.append(v or self.builder.const_int(0))
                                    expanded = True
                        else:
                            name_is_bound = (
                                isinstance(a, A.Name)
                                and (
                                    self._lookup(a.name) is not None
                                    or self._lookup_mut(a.name) is not None
                                    or self._lookup_array(a.name) is not None
                                    or self._lookup_aggregate(a.name)
                                    is not None
                                )
                            )
                            enum_variant = self._enum_variant_for_expr(a)
                            if enum_variant is not None and not name_is_bound:
                                ename2, vname2, tag = enum_variant
                                if ename2 != p_ty.name:
                                    raise NotImplementedError(
                                        f"aggregate argument for parameter "
                                        f"'{callee_ast.params[i].name}' "
                                        f"expects {p_ty.name}, got inline "
                                        f"enum constructor {ename2}; run "
                                        f"typecheck first"
                                    )
                                self._require_tag_only_enum_variant(
                                    ename2, vname2)
                                args.append(
                                    self._arena_push_slots(
                                        [self.builder.const_int(tag)]))
                                expanded = True
                        if not expanded:
                            # Existing rec-enum binding: pass the scalar
                            # arena index directly via _lower_expr.
                            if (isinstance(a, A.Name)
                                    and self._lookup_rec_enum(a.name)
                                    == p_ty.name):
                                v = self._lower_expr(a)
                                args.append(v or self.builder.const_int(0))
                                expanded = True
                            else:
                                got = type(a).__name__
                                if isinstance(a, A.Name):
                                    bound_enum = self._lookup_rec_enum(a.name)
                                    got = bound_enum or f"scalar/name '{a.name}'"
                                raise NotImplementedError(
                                    f"aggregate argument for parameter "
                                    f"'{callee_ast.params[i].name}' expects "
                                    f"{p_ty.name}, got {got}; run "
                                    f"typecheck first"
                                )
                        if expanded:
                            continue
                    slot_types = self._aggregate_slot_types(p_ty)
                    if slot_types is not None and len(slot_types) >= 1:
                        is_struct_param = (
                            isinstance(p_ty, A.TyName)
                            and p_ty.name in self._struct_flat_paths
                        )
                        is_enum_param = (
                            isinstance(p_ty, A.TyName)
                            and p_ty.name in self._enum_variants
                        )
                        n_slots = len(slot_types)
                        # Inline enum constructor as fn arg, e.g.
                        # `f(Maybe::Some(42))`. Recognize the pattern and
                        # emit [tag, payload, ...] directly without an
                        # intermediate let-bind.
                        if isinstance(a, A.Call):
                            enum_variant = self._enum_variant_for_expr(
                                a.callee)
                            if enum_variant is not None:
                                ename, vname, tag = enum_variant
                                if not is_enum_param:
                                    raise NotImplementedError(
                                        f"aggregate argument for parameter "
                                        f"'{callee_ast.params[i].name}' "
                                        f"expects {getattr(p_ty, 'name', p_ty)}, "
                                        f"got inline enum constructor "
                                        f"{ename}; run typecheck first"
                                    )
                                if ename != p_ty.name:
                                    raise NotImplementedError(
                                        f"aggregate argument for parameter "
                                        f"'{callee_ast.params[i].name}' "
                                        f"expects {p_ty.name}, got inline "
                                        f"enum constructor {ename}; run "
                                        f"typecheck first"
                                    )
                                args.append(self.builder.const_int(tag))
                                for av in self._lower_enum_payload_args(
                                        ename, vname, a.args):
                                    args.append(av)
                                # Pad to n_slots if variant has fewer args.
                                payload_count = len(a.args)
                                for slot_ty in slot_types[
                                        1 + payload_count:]:
                                    args.append(self._zero_for_type(slot_ty))
                                expanded = True
                        # Inline tag-only path as fn arg, e.g. `f(Maybe::None)`.
                        if not expanded:
                            name_is_bound = (
                                isinstance(a, A.Name)
                                and (
                                    self._lookup(a.name) is not None
                                    or self._lookup_mut(a.name) is not None
                                    or self._lookup_array(a.name) is not None
                                    or self._lookup_aggregate(a.name)
                                    is not None
                                )
                            )
                            enum_variant = self._enum_variant_for_expr(a)
                            if enum_variant is not None and not name_is_bound:
                                ename, vname, tag = enum_variant
                                if not is_enum_param:
                                    raise NotImplementedError(
                                        f"aggregate argument for parameter "
                                        f"'{callee_ast.params[i].name}' "
                                        f"expects {getattr(p_ty, 'name', p_ty)}, "
                                        f"got inline enum constructor "
                                        f"{ename}; run typecheck first"
                                    )
                                if ename != p_ty.name:
                                    raise NotImplementedError(
                                        f"aggregate argument for parameter "
                                        f"'{callee_ast.params[i].name}' "
                                        f"expects {p_ty.name}, got inline "
                                        f"enum constructor {ename}; run "
                                        f"typecheck first"
                                    )
                                self._require_tag_only_enum_variant(
                                    ename, vname)
                                args.append(self.builder.const_int(tag))
                                for slot_ty in slot_types[1:]:
                                    args.append(self._zero_for_type(slot_ty))
                                expanded = True
                        if (not expanded and is_struct_param
                                and isinstance(a, A.StructLit)):
                            if a.name != p_ty.name:
                                raise NotImplementedError(
                                    f"aggregate argument for parameter "
                                    f"'{callee_ast.params[i].name}' expects "
                                    f"{p_ty.name}, got {a.name}"
                                )
                            flat_paths = self._struct_flat_paths.get(
                                p_ty.name, [])
                            for path in flat_paths:
                                av = self._resolve_path_value(a, path)
                                if av is None:
                                    raise NotImplementedError(
                                        f"aggregate argument for parameter "
                                        f"'{callee_ast.params[i].name}' "
                                        f"could not lower field "
                                        f"{'.'.join(path)}"
                                    )
                                args.append(av)
                            expanded = True
                        if not expanded and isinstance(a, A.Name):
                            if (is_struct_param
                                    and self._lookup_struct(a.name)
                                    != p_ty.name):
                                raise NotImplementedError(
                                    f"aggregate argument for parameter "
                                    f"'{callee_ast.params[i].name}' expects "
                                    f"{p_ty.name}, got scalar/name "
                                    f"'{a.name}'"
                                )
                            if (is_enum_param
                                    and self._lookup_enum(a.name)
                                    != p_ty.name):
                                raise NotImplementedError(
                                    f"aggregate argument for parameter "
                                    f"'{callee_ast.params[i].name}' expects "
                                    f"{p_ty.name}, got scalar/name "
                                    f"'{a.name}'"
                                )
                            agg = self._lookup_aggregate(a.name)
                            if ((is_struct_param or is_enum_param)
                                    and agg is not None):
                                for j in range(n_slots):
                                    if j < len(agg):
                                        args.append(agg[j])
                                    else:
                                        args.append(
                                            self._zero_for_type(slot_types[j]))
                                expanded = True
                            arr = self._lookup_array(a.name)
                            if not expanded and arr is not None:
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
                                        args.append(
                                            self._zero_for_type(slot_types[j]))
                                expanded = True
                            if not expanded and is_enum_param:
                                scalar_v = self._lookup(a.name)
                                if scalar_v is not None:
                                    args.append(scalar_v)
                                    for slot_ty in slot_types[1:]:
                                        args.append(
                                            self._zero_for_type(slot_ty))
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
                                        agg = self._lookup_aggregate(base)
                                        if agg is not None:
                                            for j in range(n_slots):
                                                src_idx = base_idx + j
                                                if src_idx < len(agg):
                                                    args.append(agg[src_idx])
                                                else:
                                                    args.append(
                                                        self._zero_for_type(
                                                            slot_types[j]))
                                            expanded = True
                                        arr = self._lookup_array(base)
                                        if not expanded and arr is not None:
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
                            raise NotImplementedError(
                                f"aggregate argument for parameter "
                                f"'{callee_ast.params[i].name}' expects "
                                f"{getattr(p_ty, 'name', repr(p_ty))}, got "
                                f"{type(a).__name__}; run typecheck first"
                            )
                if not expanded:
                    v = self._lower_expr(a)
                    if v is not None:
                        args.append(v)
            # Determine call target name
            if (isinstance(expr.callee, A.Name)
                    and expr.callee.name not in self.functions
                    and (self._lookup(expr.callee.name) is not None
                         or self._lookup_mut(expr.callee.name) is not None)):
                raise NotImplementedError(
                    "function-typed calls are not supported by the Stage 31 "
                    "backend"
                )
            target = "<unknown>"
            if isinstance(expr.callee, A.Name):
                target = expr.callee.name
            elif isinstance(expr.callee, A.Path):
                target = "::".join(expr.callee.segments)
            if target not in self.functions:
                if target in self._const_fn_aliases:
                    raise NotImplementedError(
                        "function-typed calls are not supported by the Stage "
                        "31 backend"
                    )
                raise NotImplementedError(
                    f"unknown function '{target}' in IR lowering at "
                    f"{expr.span.line}:{expr.span.col}; run typecheck first"
                )
            # Emit as opaque CALL with return type from the registered fn
            ret_ty: tir.TIRType = tir.TIRScalar("?")
            is_extern_target = False
            callee_ir = self.functions[target]
            ret_ty = callee_ir.return_ty
            if callee_ir.attrs.get("is_extern"):
                is_extern_target = True
            # Stage 16.5: route extern "C" calls through FFI_CALL so the
            # backend emits a GOT-indirect call resolved by the dynamic
            # linker, not a relative call to a user-fn body.
            if is_extern_target:
                return self.builder.emit(tir.OpKind.FFI_CALL, *args,
                                         result_ty=ret_ty,
                                         attrs={"target": target})
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
            t_val = self._lower_block(
                expr.then, expected_rec_enum=expected_rec_enum)
            if t_val is None:
                t_val = self.builder.const_int(0)
            self.builder.emit(tir.OpKind.BR, t_val,
                              attrs={"target_block": merge_blk.id})

            # Else arm
            self.builder.switch_to(else_blk)
            if expr.else_ is None:
                e_val = self.builder.const_int(0)
            elif isinstance(expr.else_, A.Block):
                e_val = self._lower_block(
                    expr.else_, expected_rec_enum=expected_rec_enum,
                ) or self.builder.const_int(0)
            else:
                e_val = self._lower_expr(
                    expr.else_, expected_rec_enum=expected_rec_enum,
                ) or self.builder.const_int(0)
            self.builder.emit(tir.OpKind.BR, e_val,
                              attrs={"target_block": merge_blk.id})

            # Merge: the if's value is the merge block's single param
            self.builder.switch_to(merge_blk)
            result = self.builder.new_block_param(t_val.ty, hint="if_result")
            return result
        if isinstance(expr, A.UnsafeBlock):
            # Stage 28.6 — `unsafe { ... }` is a capability boundary at
            # the source level. At lowering time we treat it as a normal
            # Block: the body is lowered identically to a non-unsafe
            # block; raw-pointer ops inside are permitted, raw-pointer
            # ops OUTSIDE any UnsafeBlock are diagnosed by
            # `helixc.frontend.unsafe_pass.check_unsafe_ops` (wired into
            # helixc/check.py). The lowering pass intentionally does
            # NOT replicate the syntactic gate — there's nothing to
            # add to the IR for a permitted op.
            return self._lower_block(
                expr.body, expected_rec_enum=expected_rec_enum)
        if isinstance(expr, A.Block):
            return self._lower_block(expr, expected_rec_enum=expected_rec_enum)
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
                # Stage 28.9 cycle-105 F2 fix (silent-failure HIGH, conf 80):
                # pre-fix silently lowered body ONCE with the iter-var
                # unbound, so `for x in xs { ... }` (non-Range iter) returned
                # 0 / garbage instead of iterating. Same defect class as
                # cycle-101 F1 (StrLit silent fallthrough): typecheck admits
                # what lower cannot handle. Loud trap beats silent miscompile.
                raise NotImplementedError(
                    f"for-loop with non-Range iter not yet supported "
                    f"at {expr.span.line}:{expr.span.col} "
                    f"(iter expr: {type(expr.iter_expr).__name__})")

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
            # Stage 28.9 cycle 77 audit-T F1 fix (HIGH conf 78): emit the
            # constant `1` in the same dtype as the iterator. Pre-fix
            # const_int(1) defaulted to i32 even when start_v.ty was i64,
            # producing ADD(i64, i32, result_ty=i64) — the x86_64 backend
            # dispatches ADD by result type only and issued an 8-byte read
            # of the i32 slot, leaking 4 bytes of uninitialized stack into
            # every `for i in 0i64..N` loop increment.
            inc_dtype = start_v.ty.name if isinstance(start_v.ty, tir.TIRScalar) else "i32"
            one = self.builder.const_int(1, dtype=inc_dtype)
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
        if isinstance(expr, A.Break):
            # Stage 28.9 cycle-105 F1 fix (silent-failure CRITICAL, conf 95):
            # pre-fix the catch-all `return None` at the bottom of
            # _lower_expr silently dropped A.Break/A.Continue. The parser
            # accepts `break;` / `continue;` inside loops and typecheck
            # passes them without scope validation; with no lower arm,
            # `loop { ...; if c { break; } }` silently emitted an infinite
            # loop. Loud trap beats silent miscompile. Real CFG support
            # requires a loop-break-block stack threaded through the
            # While/Loop/For arms — deferred until used by bootstrap.
            raise NotImplementedError(
                f"break not yet supported at "
                f"{expr.span.line}:{expr.span.col}")
        if isinstance(expr, A.Continue):
            # Stage 28.9 cycle-105 F1 fix (companion to A.Break).
            raise NotImplementedError(
                f"continue not yet supported at "
                f"{expr.span.line}:{expr.span.col}")
        if isinstance(expr, A.Loop):
            # `loop { body }` — same skeleton as While but with no exit
            # condition (caller expected to break, which we don't yet
            # support, so this becomes effectively infinite). Without a
            # header→body→header back-edge, the body would just fall
            # through into whatever follows, which was the prior bug.
            # Stage 28.9 cycle 97 audit-T C96-1 fix (HIGH conf 90):
            # pre-fix used `new_block()` which creates the Block but
            # does NOT append it to `current_fn.blocks` — orphaned
            # blocks were unreachable from the function-level slot/
            # label/BR enumeration, so the x86_64 backend aborted at
            # "BR to unknown block <id>" for any `loop { body }`.
            # Same idiom as For/While arms at lines 1813/1873 which
            # correctly use `append_block`.
            header_blk = self.builder.append_block()
            body_blk = self.builder.append_block()
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
            v = (self._lower_expr(
                    expr.value,
                    expected_rec_enum=self._current_expected_rec_enum)
                 if expr.value is not None else None)
            # Audit 28.8 cycle 2 C2-2 — emit TRACE_EXIT before the
            # `ret` op when the enclosing fn is @trace'd. Pre-fix, only
            # the fall-through return at the end of _lower_fn_body
            # emitted TRACE_EXIT, so an explicit early `return X` in a
            # traced fn produced an unbalanced ENTRY-without-EXIT pair
            # in the trace stream (invisible at Phase-0 because the
            # backend stubs both ops; would corrupt the buffer the
            # moment Stage 30 runtime exists).
            if self._is_fn_traced:
                ret_operand = v
                if ret_operand is None:
                    ret_operand = self.builder.const_int(0)
                self.builder.emit(
                    tir.OpKind.TRACE_EXIT, ret_operand,
                    attrs={"fn_name": self._current_fn_name or "<unknown>"},
                )
            self.builder.ret(v)
            return None
        if isinstance(expr, A.Range):
            # Stage 28.9 cycle 110 audit-S F2 fix (HIGH conf 92): pre-fix
            # A.Range silently returned None — the caller's `or const_int
            # (0)` then substituted 0 for whatever the user meant. A
            # range-in-expr-position is currently only meaningful as the
            # `iter_expr` of a `for` (handled at line 1820 directly on
            # the A.For arm). Every other position is a misuse the
            # parser admits but the lower can't represent. Loud-fail so
            # the gap surfaces at lower time rather than as `=0`
            # miscompile. Sibling of cycle-108 F8 (CharLit/StructLit/
            # TileLit) loud-fail arms.
            raise NotImplementedError(
                f"range expression in non-For-iter position not yet "
                f"supported in IR lowering at "
                f"{expr.span.line}:{expr.span.col}")
        if isinstance(expr, A.Assign):
            v = self._lower_expr(expr.value)
            if v is None:
                v = self.builder.const_int(0)
            # Stage 16 — HBM tile param indexed store: `name[i] = expr` where
            # `name` is a kernel tile<dtype, [N], HBM> param. Lowers to
            # TILE_INDEX_STORE which PTX backend turns into `st.global.<dtype>`.
            if (isinstance(expr.target, A.Index)
                    and isinstance(expr.target.callee, A.Name)
                    and len(expr.target.indices) == 1):
                hbm = self._lookup_hbm_tile(expr.target.callee.name)
                if hbm is not None:
                    dtype_name, _length, _param_pos = hbm
                    idx_v = self._lower_expr(expr.target.indices[0]) \
                            or self.builder.const_int(0)
                    if expr.op != "=":
                        # Phase-0: only plain assign on HBM tiles.
                        raise NotImplementedError(
                            "Stage 16 HBM tile compound assign (e.g. +=) not "
                            "supported; use load + arith + store")
                    self.builder.emit(
                        tir.OpKind.TILE_INDEX_STORE, idx_v, v,
                        attrs={"name": expr.target.callee.name,
                               "dtype": dtype_name,
                               "memspace": "hbm"})
                    return None
            if isinstance(expr.target, A.Index) and isinstance(expr.target.callee, A.Name):
                base_v = self._lookup(expr.target.callee.name)
                if base_v is not None and isinstance(
                        base_v.ty, (tir.TIRTensorTy, tir.TIRTileTy)):
                    raise TypeError(
                        "unsupported tensor/tile indexing reached lowering; "
                        "run typecheck first or add matching TIR lowering"
                    )
            if isinstance(expr.target, A.Index) and not isinstance(
                    expr.target.callee, A.Name):
                base_v = self._lower_expr(expr.target.callee)
                if base_v is not None and isinstance(
                        base_v.ty, (tir.TIRTensorTy, tir.TIRTileTy)):
                    raise TypeError(
                        "unsupported tensor/tile indexing reached lowering; "
                        "run typecheck first or add matching TIR lowering"
                    )
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
            # Stage 16 — HBM tile param indexed load: `name[i]` where `name`
            # is a kernel tile<dtype, [N], HBM> param. Lowers to
            # TILE_INDEX_LOAD which PTX backend turns into `ld.global.<dtype>`.
            if (isinstance(expr.callee, A.Name)
                    and len(expr.indices) == 1):
                hbm = self._lookup_hbm_tile(expr.callee.name)
                if hbm is not None:
                    dtype_name, _length, _param_pos = hbm
                    idx_v = self._lower_expr(expr.indices[0]) \
                            or self.builder.const_int(0)
                    return self.builder.emit(
                        tir.OpKind.TILE_INDEX_LOAD, idx_v,
                        result_ty=tir.TIRScalar(dtype_name),
                        attrs={"name": expr.callee.name,
                               "dtype": dtype_name,
                               "memspace": "hbm"})
            # If callee is a Name pointing to an array, emit LOAD_ELEM.
            if isinstance(expr.callee, A.Name):
                base_v = self._lookup(expr.callee.name)
                if base_v is not None and isinstance(
                        base_v.ty, (tir.TIRTensorTy, tir.TIRTileTy)):
                    raise TypeError(
                        "unsupported tensor/tile indexing reached lowering; "
                        "run typecheck first or add matching TIR lowering"
                    )
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
                agg = self._lookup_aggregate(expr.callee.name)
                if (agg is not None and len(expr.indices) == 1
                        and isinstance(expr.indices[0], A.IntLit)):
                    idx_int = expr.indices[0].value
                    if 0 <= idx_int < len(agg):
                        return agg[idx_int]
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
            callee_v = self._lower_expr(expr.callee)
            if callee_v is not None and isinstance(
                    callee_v.ty, (tir.TIRTensorTy, tir.TIRTileTy)):
                raise TypeError(
                    "unsupported tensor/tile indexing reached lowering; "
                    "run typecheck first or add matching TIR lowering"
                )
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
                        agg = self._lookup_aggregate(base_name)
                        if agg is not None and 0 <= idx_int < len(agg):
                            return agg[idx_int]
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
            # Restart 49 B4: narrow exception scope. structural_hash
            # raises NotImplementedError (per ast_hash._hash_into's
            # cycle-14/15 loud-fail discipline) for unhandled AST
            # subclasses. The previous wide `except Exception` swallowed
            # the loud-fail and aliased two distinct quote() bodies of
            # the new AST type to the same _pretty fallback string,
            # silently miscompiling quote-cell lookup. Mirrors the
            # restart-47 B1 narrowing in
            # _resolve_monomorphized_struct_type and the autodiff.py
            # sibling pattern. Catch only the lookup-style errors that
            # legitimately mean "this expression isn't hashable by name";
            # let loud-fail signals propagate so the new AST subclass
            # forces explicit dispatch.
            try:
                key = structural_hash(expr.inner)
            except (KeyError, AttributeError, TypeError, ValueError):
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

    def _verifier_abi_matches_f64(self, fn_name: str) -> bool:
        """A modify_f64 verifier takes (handle: i32, val: f64) and returns
        i32/bool. System V passes the f64 in xmm0 and the int in edi."""
        ir_fn = self.functions.get(fn_name)
        if ir_fn is None or len(ir_fn.params) != 2:
            return False
        p0, p1 = ir_fn.params
        if not (isinstance(p0.ty, tir.TIRScalar) and p0.ty.name == "i32"):
            return False
        if not (isinstance(p1.ty, tir.TIRScalar) and p1.ty.name == "f64"):
            return False
        if not (isinstance(ir_fn.return_ty, tir.TIRScalar)
                and ir_fn.return_ty.name in ("i32", "bool")):
            raise ValueError(
                f"verifier function {fn_name!r} has parameters (i32, f64) "
                f"but returns {ir_fn.return_ty!r} - verifiers must return "
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
