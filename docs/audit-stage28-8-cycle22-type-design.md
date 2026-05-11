# Stage 28.8 pre-29 audit gate — Cycle 22 (Audit B: type-design soundness)

**Date:** 2026-05-11
**HEAD:** `bee36e6` ("Audit 28.8 cycle 21 fix-sweep: close C20-1 (HIGH,
PTX backend isize/usize silent 32-bit)")
**Lens:** type-design soundness (Audit B)
**Streak counter at start:** 1/5 (cycle 21 first clean of streak; cycle
22-A passed under same HEAD)
**Bar:** ZERO findings of ANY severity at confidence >= 75. Re-flagging
prior-cycle findings is forbidden; manufacturing findings is forbidden.

---

## Scope

Type-design soundness re-audit of the **newly added** Stage 28.8.1 +
28.8.2 surface and the autonomous cycles 17-21 isize/usize fix-sweeps,
under the strict criterion. Aligned with the user's cycle-22 task brief:

1. `helixc/frontend/ast_walker.py` — does `dataclasses.fields()`
   introspection respect type-field annotations safely?
2. `_op_suffix(op)` table semantics — do two distinct ops with the same
   value-tuple collide in the symbol map?
3. isize/usize fixes (commits c6136d4, 0803902, 5a1e406, bee36e6) —
   consistent handling of `TyPrim` pointer-width aliases across
   `const_fold`, PTX, x86_64.
4. Deferred `grad_pass._rewrite_in_expr` / `_resolve_in_expr` rewriter
   case — verify no type-soundness gap is introduced by NOT refactoring.

---

## Target 1 — `helixc/frontend/ast_walker.py` (Stage 28.8.2)

### Field-introspection contract

`_iter_child_nodes` enumerates dataclass fields via
`dataclasses.fields(node)` and yields each value that passes
`_is_ast_node`. The skip set is partitioned into two frozensets at
module scope:

- `_TYPE_FIELD_NAMES = {"ty", "target_ty", "return_ty", "dtype"}` —
  matches every `TyNode`-typed field in `ast_nodes.py`:
  - `Let.ty`, `ConstStmt.ty`, `ConstDecl.ty` → `ty`
  - `Cast.target_ty` → `target_ty`
  - `FnDecl.return_ty` → `return_ty`
  - `TileLit.dtype`, `TyTensor.dtype`, `TyTile.dtype` → `dtype`

  Exhaustive — I cross-referenced `Grep` for `:\s+"?TyNode"?|:\s+Optional\["?TyNode"?\]` across
  `ast_nodes.py` and every Ty-typed Expr/Stmt/Item field name in that
  output appears in the set. No drift.

- `_NON_NODE_FIELD_NAMES = {"span", "module", "op", "type_suffix",
  "is_mut", "is_pub", "is_extern", "extern_abi", "attrs", "var_name",
  "generics", "where_clauses", "trait_name"}` — all primitive- or
  metadata-typed fields. Two require justification:

  - `generics`: on `Name` it is `list[TyNode]` (correctly type-only;
    skip preserves the documented "type fields are not walked by
    default" contract). On `FnDecl`/`StructDecl`/`EnumDecl`/`TypeAlias`
    it is `list[GenericParam]` — but `GenericParam` carries only
    `name: str` + `kind: str` (no Expr children), so skipping does
    not lose Expr reachability.
  - `where_clauses`: `list[WhereClause]` where `WhereClause` carries
    `constraint: Expr`. Walkers in this codebase all dispatch at
    `it.body` (a `Block`) and never start at `FnDecl` itself, so the
    `where_clauses` skip is unreachable for every existing pass. (Pre-
    Stage 28.8.2 walkers also did not walk `where_clauses`; behavior
    preserved.) If a future visitor starts at `FnDecl` and needs
    `where_clauses`, the documented escape hatch is to override
    `visit_FnDecl` and walk explicitly.

### Type-field annotation safety

The walker never relies on Python type **annotations** at runtime — it
relies on `dataclasses.fields()` (which returns `Field` objects keyed by
name + default) and a **value-shape check** via `_is_ast_node`:

```python
if isinstance(obj, (str, int, bool, float, tuple)): return False
if isinstance(obj, A.Span): return False
if isinstance(obj, A.TyNode): return False
return dataclasses.is_dataclass(obj) and not isinstance(obj, type)
```

This is safe under three observations:

1. Strings, ints, bools, floats are rejected before the dataclass check
   so primitive `value` fields (`IntLit.value: int`, `BoolLit.value: bool`,
   `StrLit.value: str`, `CharLit.value: str`, `FloatLit.value: float`)
   are filtered at iteration time. Adding a primitive-typed field to a
   future AST class does not break the walker.
2. Every dataclass that is NOT `Span` or a `TyNode` is treated as
   walkable — including `MatchArm`, `Pattern`, `PatVariant`, `FnParam`,
   `GenericParam`, `EnumVariant`, `AgentMethod`, `WhereClause`. Each is
   reached only when its parent field is not in either skip set. For
   the existing passes, the reachable set is:
   - `MatchArm` (via `Match.arms`) — walked, then its `pattern` /
     `guard` / `body` walked.
   - `Pattern` subclasses (via `MatchArm.pattern`) — walked, then
     `PatLit.value`, `PatRange.lo/hi`, `PatVariant.path/sub_patterns`,
     `PatTuple.elems`, `PatOr.alts` walked.

   All match `test_ast_walker.py:test_walker_visits_match_arm_guard`
   plus the manual `Modify` test.
3. `tuple` values are filtered as non-nodes at the top level
   (`_is_ast_node` returns False on tuple) so a bare `Span` field's
   internals are not walked; but `_iter_child_nodes` also descends into
   `tuple` values element-by-element to handle `StructLit.fields:
   list[tuple[str, Expr]]`. The element-level `_is_ast_node` rejects
   the string and walks the Expr — exactly matching cycle-2 fix
   coverage.

### Skip-marker semantics

`visit()` calls the override if present, then calls `generic_visit(node)`
**unless** the override returned the literal `False`. Per-pass usage:

- `panic_pass._PanicCollector` / `_PanicArgsValidator`: returns `None`
  (post-descent fires) — correct for accumulating visitors.
- `unsafe_pass._RawPtrOpVisitor`: returns `False` from
  `visit_UnsafeBlock` and explicitly drives descent inside the
  `try/finally` so the unsafe-depth counter stays balanced.
- `grad_pass._GradCallFinder`: returns `False` from `visit_Call` once
  the grad call is found AND overrides `visit()` to short-circuit on
  `self.found`. Net effect: stops descent into discovered grad call's
  args (which is fine — we only need the existence bit) and into
  every subsequent node.
- `struct_mono._BodyVisitor`: returns `None` (post-descent fires). Type
  fields walked explicitly via `visit_ty(...)` in the override before
  fall-through.

All four patterns are documented in `ast_walker.py`'s class docstring;
all four behaviors verified against `test_ast_walker.py`. No anomaly.

### TileLit re-coverage observation (not a finding)

The new `_GradCallFinder` predicate, by virtue of dataclass
introspection, now walks `TileLit.shape` (`list[Expr]`) + `TileLit.memspace`
(`Expr`). The pre-Stage-28.8.2 hand-rolled `_expr_has_grad` did NOT
have a `TileLit` arm (verified at parent `git show 9436810^`). The
behavior change is strictly more permissive (no false-negatives
introduced; only the predicate runs the slower rewriter more often).
TileLit shape/memspace must be const-int per typecheck, so a grad call
nested there is rejected upstream — no soundness consequence.

**Conclusion for Target 1: no finding.**

---

## Target 2 — `helixc/backend/x86_64.py:_op_suffix` (Stage 28.8.1)

The user's question (re-phrased): "do two distinct ops with the same
value-tuple collide in the symbol map?"

This rests on a misread of the implementation. The suffix is **not**
keyed by value-tuple. From `FnCompiler.__init__` at lines 830-840:

```python
self._op_index: dict[int, int] = {}
_idx = 0
for _blk in fn.blocks:
    for _op in _blk.ops:
        self._op_index[id(_op)] = _idx
        _idx += 1
```

- Key = `id(op)` (Python object identity).
- Value = sequential integer.
- Suffix = `f"{self.fn_index}_{idx}"`.

**Two distinct `tir.Op` instances ALWAYS have distinct `id()`s** (CPython
guarantees `id` uniqueness for the lifetime of each object). Distinct
ids get distinct sequential `op_index` values via the enumeration. The
only way to produce a collision is for the SAME `Op` object to appear
in two block-ops lists — which the IR builder never does (every pass
constructs fresh `Op(...)` dataclass instances; verified across
`helixc/ir/passes/const_fold.py`, `dce.py`, `tir.py`, `tile_ir.py`).

Even in the hypothetical shared-identity case the result would be **two
references to the same op emitting the same symbol** — i.e. correct
dedupe, not a name clash. The `_pending_strings` list is a `list` (not a
set), but the assembler's relocation table is keyed by symbol name; two
appends of `(__helix_strptr_0_7, b"...")` reduce to one symbol entry
(the second emission overwrites the same byte range). Idempotent.

The C20-2 forward debt (cycle 20 silent-failures, also in cycle 21 doc):
the `_op_suffix` fallback at line 875 still embeds `id(op):x` when
the op isn't in the pre-walk index. That fallback fires only if a later
codegen pass synthesizes a new `Op` after `FnCompiler.__init__`, which
no current pass does. The escape is intentionally LOUD via the symbol
name (the byte-identical regression test would diff). This is a
documented forward note, not a current-cycle defect.

**Conclusion for Target 2: no finding.**

---

## Target 3 — isize/usize fixes consistency (commits c6136d4, 0803902, 5a1e406, bee36e6)

Cross-pass canon check at HEAD `bee36e6`:

| Site | File:Line | isize handling | usize handling |
|---|---|---|---|
| typecheck `_widen_canon_name` | `typecheck.py:225-228` | aliased to `i64` | aliased to `u64` |
| typecheck `_WIDEN_RANK` | `typecheck.py:241` | rank 40 (= i64) | rank 41 (= u64) |
| typecheck `_PRIMITIVE_TYPE_NAMES` | `typecheck.py:337-338, 2091-2092` | listed | listed |
| typecheck IntLit range | `typecheck.py:1816-1817` | full 64-bit signed | full 64-bit unsigned |
| lower_ast `_PRIMITIVE_TYPE_NAMES` | `lower_ast.py:357-358` | listed (lowers to `TIRScalar("isize")`) | listed (lowers to `TIRScalar("usize")`) |
| const_fold `_INT_BITS` | `const_fold.py:53` | `64` | `64` |
| x86_64 `_is_i64_type` | `x86_64.py:1011` | accepts | (separate predicate) |
| x86_64 `_is_u64_type` | `x86_64.py:1017` | (separate predicate) | accepts |
| x86_64 `_check_array_elem_size_supported` | `x86_64.py:1042` | wide-trap-loud | wide-trap-loud |
| PTX `_ptx_type_str` mapping | `ptx.py:169` | `.b64` | `.b64` |
| PTX `_DTYPE_SIZE` | `ptx.py:342` | `8` | `8` |
| PTX `_DTYPE_PTX_LOAD` | `ptx.py:347` | `s64` | `u64` |
| PTX `_ld_reg_prefix` | `ptx.py:361` | `rd` (64-bit pool) | `rd` (64-bit pool) |

Every width-keyed table that influences emitted bytes routes
isize→i64-semantics and usize→u64-semantics. The chain is closed:

- Source `let x: isize = ...` → AST `TyName("isize")` → typecheck
  resolves to `TyPrim("isize")` with widening rank 40 (= i64) and
  IntLit range matching signed-64-bit → lower_ast emits
  `TIRScalar("isize")` → x86_64 classifier returns True for
  `_is_i64_type` → CONST_INT emits 64-bit `mov rax, imm64`; fn-param
  spill emits 64-bit; LOAD_ELEM/STORE_ELEM trap loudly with the
  C16-1 marker.
- Same source → emit-ptx path → `_ptx_type_str` returns `.b64`;
  `_DTYPE_SIZE` returns 8; `_ptx_load_suffix` returns `s64`;
  `_ld_reg_prefix` returns `rd`.
- Same source → const-fold path → `_wrap_int_to_type` uses
  `_INT_BITS["isize"] = 64`, masking at the 64-bit boundary that
  matches both backends.

No remaining isize/usize site falls back to 32-bit silently. The
3 `.get(..., default)` defaults that remain (`_ptx_type_str` →
`.b32`, `_dtype_size` → `4`, `_ptx_load_suffix` → `u32`, plus
`_INT_BITS.get(..., 32)`) all sit beneath the explicit isize/usize
entries; only a future scalar type NOT yet in the tables would
silently fall back. That is the documented Stage-29 forward-debt
(centralized `_scalar_width_bits` predicate); explicitly out of
cycle-22 scope and not a new finding.

**Conclusion for Target 3: no finding.**

---

## Target 4 — Deferred `grad_pass` rewriter case

`_rewrite_in_expr` + `_resolve_in_expr` remain hand-rolled per the
Stage 28.8.2 commit-body decision. The commit explicitly justifies the
deferral: a generic rewriter cannot reuse `ASTVisitor`'s read-only walk
contract because each Expr subtype has different rewrite semantics
(some return new nodes, some mutate in place, some need a per-arm
recursion into Match arms).

Coverage comparison vs. AST Expr subtypes at HEAD `bee36e6`:

| Expr subtype | `_rewrite_in_expr` | `_resolve_in_expr` |
|---|---|---|
| Call | yes | yes |
| Binary | yes | yes |
| Unary | yes | yes |
| Block | yes (via `_rewrite_in_block`) | yes (via `_resolve_let_aliases`) |
| If | yes (incl. else-if chain — cycle-3 C3-1 fix) | yes (incl. else-if) |
| Match | yes (incl. arm bodies) | yes (incl. guard + body) |
| For | yes | yes |
| While | yes | yes |
| Loop | yes | yes |
| Break | yes | yes |
| Return | yes | yes |
| Cast | yes | yes |
| Assign | yes | yes |
| Index | yes | yes |
| Field | yes | yes |
| Range | yes (start/end) | yes (start/end) |
| TupleLit | yes | yes |
| ArrayLit | yes | yes |
| StructLit | yes (field exprs) | yes (field exprs) |
| UnsafeBlock | yes | yes |
| Quote / Splice | yes (inner) | yes (inner) |
| Modify | yes (target/transformation/verifier) | yes (target/transformation/verifier) |
| IntLit / FloatLit / StrLit / CharLit / BoolLit | leaf (no children) | leaf |
| Name / Path / Continue | leaf | leaf |
| TileLit | NOT walked | NOT walked |

The single asymmetry vs. the new `_GradCallFinder` predicate is
`TileLit`. The predicate (via dataclass introspection) walks
`TileLit.shape`/`memspace`; the rewriter does not. This is **not a new
gap** introduced by Stage 28.8.2 — the pre-Stage-28.8.2 hand-rolled
`_expr_has_grad` predicate also did not walk `TileLit` (verified at
`git show 9436810^:helixc/frontend/grad_pass.py`). So the deferral is
behavior-preserving for the rewriter side.

Is the asymmetry a soundness gap? No:

- A `grad(f)` call nested inside `TileLit.shape` would require the
  shape position to accept a non-const-int Call expression. Typecheck
  rejects this at the tile-shape resolution step (TileLit shape must
  be a compile-time-resolvable const-int per Stage 15; `Call`
  expressions fail the resolution and raise a type error). The
  rewriter therefore cannot encounter unrewritten `grad(...)` at
  codegen — the upstream type-check is the soundness barrier, not
  the rewriter's exhaustiveness.
- `TileLit.memspace` is parsed as a `Name` (e.g., `REG`) per the Stage-
  15 surface contract; a `Call` there is also rejected upstream.

Net: the rewriter being non-exhaustive over TileLit is reachable only
via type-rejected source — no silent miscompile path exists. The
behavior matches the pre-refactor rewriter exactly, so Stage 28.8.2
introduces zero net soundness regression. The deferral remains the
documented Phase-A forward note: when rewriter semantics are unified
(Stage 29 candidate), the same dataclass-introspection contract should
extend to it.

**Conclusion for Target 4: no finding.**

---

## Cross-target summary

| Target | Confidence finding exists | Outcome |
|---|---|---|
| 1. `ast_walker.py` field-introspection safety | < 5 | CLEAN |
| 2. `_op_suffix` collision potential | < 5 | CLEAN |
| 3. isize/usize cross-pass consistency | < 5 | CLEAN |
| 4. Deferred rewriter type-soundness gap | < 5 | CLEAN |

No finding at any severity meets the 75-confidence bar.

---

## Streak verdict

Cycle 22, Audit B (type-design): **CLEAN** under the strict criterion.

Combined with cycle 22, Audit A (silent failures — already CLEAN at
the same HEAD per `audit-stage28-8-cycle22-silent-failures.md`), the
fresh-clean streak counter advances:

- Cycle 21: 1/5 (first clean)
- Cycle 22 (A clean + B clean): **2/5**

Three more clean cycles needed before Stage 29 gate opens.
