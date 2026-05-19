# Cycle 1 Batch IR — Silent-Failure Hunter (Depth, Auditor 4)

Audit date: 2026-05-18
Focus: exhaustiveness + fall-through + effect-gap depth pass
Auditor: pr-review-toolkit:silent-failure-hunter (auditor #4 of 5)

Scope:
- `helixc/ir/tir.py`
- `helixc/ir/tile_ir.py`
- `helixc/ir/lower_ast.py`
- `helixc/ir/passes/const_fold.py`
- `helixc/ir/passes/cse.py`
- `helixc/ir/passes/dce.py`
- `helixc/ir/passes/effect_check.py`
- `helixc/ir/passes/fdce.py`
- `helixc/ir/passes/tile_opt.py`

## Verdict

**NOT_CLEAN — 4 HIGH + 5 MEDIUM + 3 LOW**

The surface pass cleaned up the previously-known broad `except Exception` and
explicit `return None` catch-alls. This depth pass found a second tier of
silent failures that the surface pass did not surface: **shape-expression
dispatch is non-exhaustive and quietly degrades to `DimDyn()` / `"?"`**,
**`_lower_stmt` has no catch-all loud-fail** so any new `Stmt` subclass would
silently no-op, **DCE's `SIDE_EFFECT_KINDS` is an opt-in inclusion list that
will silently treat `TENSOR_STORE` / `TENSOR_RAND` as pure if they're ever
emitted**, and **`tile_opt.redundant_zero_coalesce` falls back to `""` for
missing memspace** — meaning two `TILE_ZEROS` whose result type is not a
`TileType`-shaped object collide on the same key and get falsely coalesced.

The pattern across all 4 HIGH findings matches the Stage-105+ FE pattern the
brief warned about: *exhaustiveness encoded as "this is what we know how to
handle, everything else is a benign default"* rather than *"everything else
is a loud failure"*. Each one is reachable today via a future-safe code
path (new AST subclass, new TIR op, new tile constructor).

---

## HIGH findings

### HIGH-1: `_lower_dim` silently lowers unknown shape expressions to `DimDyn()`
**File**: `helixc/ir/lower_ast.py:938-947`

```python
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
```

**Pattern**: fall-through default. Any `Expr` subclass that is not
`IntLit` / `Name` / arithmetic-`Binary` silently becomes
`DimDyn()`, the "runtime-checked at boundary" sentinel.

**Demonstration**: A `tensor<f32, [N, M+1, foo(K)]>` annotation where
`foo(K)` is a function call evaluates to a runtime-dynamic dim, with no
diagnostic at IR-lowering. Same for `tensor<f32, [N, -K]>` (a `Unary`),
`tensor<f32, [N, K as i32]>` (a `Cast`), `tensor<f32, [N, K & MASK]>`
(`Binary` with op `&` not in the listed arith set), `tensor<f32, [N, true ? a : b]>`
(`If`). All silently degrade to `DimDyn()`.

**Why it matters**: `DimDyn()` disables compile-time shape checking
*for that dimension* — downstream `reshape`, `matmul`, `broadcast`
become unchecked at the IR level (runtime-only). A user-authored
shape expression that is fully compile-time-decidable but happens to
use an operator outside `+-*/%` is silently downgraded to a dynamic
dim, losing both performance (no static shape specialization) and
safety (no compile-time mismatch diagnostic). The bitwise-and case is
particularly insidious because `K & 7` is a common "round down to
power-of-2 boundary" idiom in shape arithmetic.

**Fix**: replace the `return tir.DimDyn()` catch-all with explicit arms
plus an explicit loud-fail:

```python
def _lower_dim(self, expr: A.Expr) -> tir.Dim:
    if isinstance(expr, A.IntLit):
        return tir.DimConst(expr.value)
    if isinstance(expr, A.Name):
        return tir.DimVar(expr.name)
    if isinstance(expr, A.Binary) and expr.op in ("+", "-", "*", "/", "%"):
        return tir.DimExpr(op=expr.op, args=(...,))
    # Only explicit "?" annotation should produce DimDyn — recognise the
    # parser's sentinel name for it and require everything else to either
    # match an explicit arm or raise.
    if isinstance(expr, A.Name) and expr.name == "?":
        return tir.DimDyn()
    raise NotImplementedError(
        f"lower_dim: unsupported shape expression {type(expr).__name__} "
        f"at {getattr(expr, 'span', '?')!r}; add an explicit dispatch arm "
        f"or use a `?` dim for explicit runtime-checked dynamic"
    )
```

---

### HIGH-2: `_stringify_marker` silently substitutes `"?"` for unknown memspace / device
**File**: `helixc/ir/lower_ast.py:949-955`, used at `lower_ast.py:826` and `lower_ast.py:833`

```python
def _stringify_marker(self, expr: Optional[A.Expr]) -> Optional[str]:
    if expr is None: return None
    if isinstance(expr, A.Name): return expr.name
    if isinstance(expr, A.Call) and isinstance(expr.callee, A.Name):
        args = ",".join(str(getattr(a, "value", "?")) for a in expr.args)
        return f"{expr.callee.name}({args})"
    return None
```

And the call sites:

```python
# lower_ast.py:826
device = self._stringify_marker(ty.device) or "cpu"
# lower_ast.py:833
mem = self._stringify_marker(ty.memspace) or "?"
```

**Pattern**: optional-chain silent-default. `_stringify_marker` returns
`None` for any expr that isn't `Name` or simple `Call(Name, ...)`. The
caller then substitutes a default (`"cpu"` for device, `"?"` for
memspace). Three failure modes silently collapse to the same sentinel:
no annotation at all, unparseable annotation, unhandled-shape annotation.

**Demonstration**: `tensor<f32, [N], some_module.gpu>` (an `A.Index` /
`A.Field` callee) silently becomes `device="cpu"`. Worse: `tile<f32, [4, 4], expr_call(SMEM)>` where the user wrote a builder-style call with a non-Name callee silently becomes `memspace="?"`.

**Why it matters**: The cycle-22 audit added `tile_io` and `arena` effects
to differentiate memspace-sensitive ops, and the `tile_opt.redundant_zero_coalesce`
pass coalesces tiles by `memspace`. A tile whose memspace silently
degraded to `"?"` collides in the coalescer's hash key with any other
silently-degraded tile — see HIGH-4. Two `tile<f32, [4, 4], SMEM>` and
`tile<f32, [4, 4], REG>` annotations, both written with an unhandled
callee shape, would coalesce as if they shared the same memspace.

**Fix**: make `_stringify_marker` either return the marker or raise
`NotImplementedError` with the span; the `or "cpu"` / `or "?"` fallback
only triggers when the annotation is genuinely absent (`None`), not
when it failed to parse:

```python
def _stringify_marker(self, expr: Optional[A.Expr]) -> Optional[str]:
    if expr is None:
        return None
    if isinstance(expr, A.Name):
        return expr.name
    if isinstance(expr, A.Call) and isinstance(expr.callee, A.Name):
        args = ",".join(str(getattr(a, "value", "?")) for a in expr.args)
        return f"{expr.callee.name}({args})"
    raise NotImplementedError(
        f"_stringify_marker: unsupported marker expr "
        f"{type(expr).__name__} at {getattr(expr, 'span', '?')!r} — "
        f"only Name and Call(Name, ...) shapes are recognised"
    )
```

---

### HIGH-3: `dce.SIDE_EFFECT_KINDS` is opt-in; `TENSOR_STORE` / `TENSOR_RAND` silently treated as pure
**File**: `helixc/ir/passes/dce.py:32-89` (set definition); cross-ref `helixc/ir/tir.py:138, 136`

The DCE side-effect set lists every op whose result-less-or-result-unused
form must still be preserved. `tir.OpKind` declares:

- `TENSOR_STORE = "tensor.store"` (tir.py:138) — not in `SIDE_EFFECT_KINDS`
- `TENSOR_RAND = "tensor.rand"` (tir.py:136) — not in `SIDE_EFFECT_KINDS`
  AND not in `cse.PURE_KINDS` (so CSE won't dedup it — that part is fine)
- `TENSOR_LOAD = "tensor.load"` (tir.py:137) — not in `SIDE_EFFECT_KINDS`;
  documented as "external (file, host buffer)", arguably observable

**Pattern**: opt-in inclusion list with implicit "everything else is
pure" semantics. DCE's `dce_function` drops any op (a) whose results
are all dead AND (b) whose `op.kind not in SIDE_EFFECT_KINDS`. A
side-effecting op that nobody remembered to add to the set is silently
droppable.

**Demonstration**: today no pass emits `TENSOR_STORE` (grep confirms:
the OpKind is declared but never produced anywhere in `helixc/`). So
this is a *dormant* miscompile rather than an actively-firing one. But
the moment a future tensor-store lowering arm lands and emits
`TENSOR_STORE` for a write whose return value isn't used (e.g. `let _ = tensor.store(buf, t);` or a void-returning store), DCE will silently
drop the store and the file/host buffer is never written.

The same gap applies to `TENSOR_RAND`: it's not in `OP_EFFECTS` for
`effect_check.py` either, so `@pure fn f() -> tensor { tensor.rand(...) }`
would silently pass effect-check despite producing nondeterministic
results.

**Why it matters**: DCE's contract is "preserve all observable
behavior." An opt-in side-effect set inverts that contract: it
preserves only what we remembered to enumerate. The cycle-22 C22-1
fix that added `FFI_CALL` (and the comment "DCE was silently dropping
void-return extern calls because their results were never live")
documents the exact pattern at one site; that same pattern is still
latent at `TENSOR_STORE` / `TENSOR_RAND`.

**Fix**: invert the policy. Maintain a `PURE_KINDS` set instead (or
in addition), and assert at module-load time that
`SIDE_EFFECT_KINDS ∪ PURE_KINDS` covers every `tir.OpKind`. Any new
op added without classification raises `RuntimeError` at import time.

```python
# at module load:
_classified = SIDE_EFFECT_KINDS | _PURE_KINDS
_unclassified = set(tir.OpKind) - _classified
if _unclassified:
    raise RuntimeError(
        f"dce.py: unclassified TIR op kinds — must be added to "
        f"either SIDE_EFFECT_KINDS (preserved) or _PURE_KINDS "
        f"(eligible for dead-code drop): "
        f"{sorted(k.name for k in _unclassified)}"
    )
```

The same hardening should be applied to `cse.PURE_KINDS`,
`effect_check.OP_EFFECTS`, and `tile_opt._SIDE_EFFECT_KINDS` (MEDIUM-3).

---

### HIGH-4: `tile_opt.redundant_zero_coalesce` falls back to `""` for missing memspace, collapsing distinct tiles
**File**: `helixc/ir/passes/tile_opt.py:148-162`

```python
if op.kind == ti.TileOpKind.TILE_ZEROS and op.results:
    result = op.results[0]
    if hasattr(result.ty, "dtype") and hasattr(result.ty, "shape"):
        dtype_name = result.ty.dtype.name
        shape_tuple = tuple(
            d.value if hasattr(d, "value") else repr(d)
            for d in result.ty.shape
        )
        memspace = getattr(result.ty, "memspace", "")
        key = (dtype_name, shape_tuple, memspace)
        if key in seen:
            # Drop this op; remap its result id.
            remap[result.id] = seen[key]
            continue
        seen[key] = result
```

**Pattern**: silent default substitution. `getattr(result.ty, "memspace", "")`
quietly returns `""` if the tile result's type has no `memspace`
attribute. Two `TILE_ZEROS` whose result types both lack `memspace`
collide on the key `(dtype, shape, "")` and the second is silently
remapped to the first.

**Demonstration**: a `TILE_ZEROS` whose `result.ty` is
`tir.TIRTensorTy` (which has `dtype` and `shape` but **no** `memspace` —
see `tir.py:44-50`) silently becomes `memspace=""`. Two
`TILE_ZEROS` of the same shape/dtype with `result.ty = TIRTensorTy`
(rather than `TIRTileTy` / `TileType`) coalesce regardless of any
device or layout differences. Worse: any future tile result type that
adds `dtype` + `shape` but uses a different memspace attribute name
(e.g., `memory_space`, `space`) also degrades to `""`.

A more concrete failure trajectory: the comment at `tir.py:8-9`
declares "Layout as type info (RowMajor / ColMajor / Blocked)" — the
`TIRTensorTy` carries a `layout: Layout = Layout.ROW_MAJOR` field but
no `memspace`. If `TILE_ZEROS` ever lowers to a tile-IR op whose
`result.ty` is a `TIRTensorTy` (e.g., a tensor literal materialized
on the boundary), coalescing across two RowMajor / two BLOCKED zeros
silently collapses regardless of layout differences.

**Why it matters**: Coalescing two TILE_ZEROS with different
memspaces is a silent miscompile — the PTX backend allocates SMEM
registers for one and the merged kept-op may have been the REG one;
the dropped one's downstream consumers get the wrong allocation.

The defensive `hasattr(result.ty, "dtype") and hasattr(result.ty, "shape")`
gate at line 150 only protects the dtype/shape fields, but the
memspace `getattr(..., "")` fallback at line 156 silently degrades
inside the guarded branch — the same gate should also require
`hasattr(result.ty, "memspace")` or refuse to coalesce.

**Fix**:

```python
if op.kind == ti.TileOpKind.TILE_ZEROS and op.results:
    result = op.results[0]
    # All three attrs are required — if any are missing, the result
    # type is not a tile-shaped type and coalescing isn't safe.
    if (hasattr(result.ty, "dtype")
            and hasattr(result.ty, "shape")
            and hasattr(result.ty, "memspace")):
        dtype_name = result.ty.dtype.name
        shape_tuple = tuple(
            d.value if hasattr(d, "value") else repr(d)
            for d in result.ty.shape
        )
        memspace = result.ty.memspace
        # Normalise enum-vs-string memspace to canonical form so
        # MemSpace.SMEM and "smem" don't get treated as distinct.
        if hasattr(memspace, "value"):
            memspace = memspace.value
        key = (dtype_name, shape_tuple, memspace)
        ...
```

Additionally, the existing `if hasattr(result.ty, ...)` short-circuit
silently *skips* coalescing for non-tile-typed `TILE_ZEROS` results
— a future-safe form would assert the type is a known tile-shaped
class instead of `hasattr`-sniffing.

---

## MEDIUM findings

### MEDIUM-1: `_lower_stmt` has no catch-all — new `Stmt` subclass would silently no-op
**File**: `helixc/ir/lower_ast.py:1316-1654`

`_lower_stmt` dispatches on `A.Let` / `A.ExprStmt` / `A.ConstStmt` and has
no terminal `else: raise NotImplementedError(...)` clause. Today the three
arms are exhaustive over `ast_nodes.Stmt`'s subclasses (confirmed: only
`Let`, `ExprStmt`, `ConstStmt` extend `Stmt`), so no live miscompile —
but the cycle-108 fix on `_lower_expr` (line 4723-4728) added exactly the
same loud-fail there for exactly this reason. The sibling pattern was not
extended to `_lower_stmt`.

**Why it matters**: defense-in-depth. If a future stage adds `A.AssertStmt`
or `A.LoopStmt` or any other `Stmt` subclass and forgets to extend
`_lower_stmt`, the statement is silently dropped from IR — no diagnostic,
no error, just missing emission. The user sees runtime divergence with
no compile trail back to the missed dispatch.

**Fix**: add a terminal `raise NotImplementedError(f"lower_ast: unhandled stmt type {type(stmt).__name__} at {getattr(stmt, 'span', '?')!r}")` after the `ConstStmt` arm, matching the cycle-108 closure of `_lower_expr`'s catch-all.

---

### MEDIUM-2: `_lower_type` `A.TyName` arm silently lowers unknown type names to `TIRScalar(name)`
**File**: `helixc/ir/lower_ast.py:811-819`

```python
if isinstance(ty, A.TyName):
    if ty.name in getattr(self, "_recursive_enums", set()):
        return tir.TIRScalar("i32")
    # Recognize struct / enum / primitive names. Anything else is
    # likely a generic type parameter (e.g. T in `fn id[T](x: T)`).
    # Generic type params lower to TIRScalar(name) which defaults
    # to i32-sized ABI — works for i32 type args, silently wrong
    # otherwise. Documented HBS limitation.
    return tir.TIRScalar(ty.name)
```

**Pattern**: documented-known silent miscompile. The comment acknowledges
the limitation explicitly: any `TyName` that isn't a known recursive
enum becomes a `TIRScalar` carrying the raw name. If the backend's
sizing routine matches by name (`i32`, `f32`, etc.) it works for the
common case, but a `TyName("MyCustomType")` slips through as a `TIRScalar`
that the backend will silently size as i32.

**Why it matters**: the comment says "works for i32 type args, silently
wrong otherwise." That is the textbook silent-failure pattern. The
documented-known status mitigates the urgency but not the severity. A
generic `fn id[T](x: T) -> T` instantiated with `T = f64` silently
demotes to i32 ABI inside the body's `TIRScalar("T")`.

**Fix**: at minimum, raise NotImplementedError when `ty.name` doesn't
match any of the recognised primitive scalar names AND isn't a known
recursive enum AND isn't in `_struct_fields`. The fall-through to
`TIRScalar(name)` for monomorphization-survivors should be replaced
by an explicit hash-set check for known generic type-parameter names
emitted by struct_mono. Anything else is a loud failure.

---

### MEDIUM-3: `tile_opt._SIDE_EFFECT_KINDS` opt-in same pattern as DCE — `TMA_LOAD` ambiguous
**File**: `helixc/ir/passes/tile_opt.py:54-62`

`tile_opt.dead_tile_elim` uses the same opt-in side-effect set pattern
as `dce.py` (HIGH-3). Each new `TileOpKind` must be classified.
Today's `TileOpKind` set has 24 kinds; `_SIDE_EFFECT_KINDS` lists 7.
The 17 omitted are mostly clearly pure (TILE_ZEROS, TILE_ADD, ...)
but `TMA_LOAD` is borderline — it's an async load that completes
asynchronously and is paired with a `BARRIER_WAIT`. If the load's
result tile is never consumed (a prefetch-only pattern), `dead_tile_elim`
drops the TMA_LOAD and the paired BARRIER_WAIT then waits on a
barrier that was never posted.

**Why it matters**: prefetch / decoupled-access kernels are a
real pattern for HBM-bandwidth-limited workloads. The dropped
TMA_LOAD is silent — no warning, the kernel just hangs at
the barrier (or barely-passes when SMEM happens to contain
old/zero data).

**Fix**: same as HIGH-3 — invert to a `_PURE_KINDS` whitelist
and assert union covers all `TileOpKind` at module load. Place
`TMA_LOAD` explicitly in either `_SIDE_EFFECT_KINDS` (treat
prefetch as a side effect) or document via a new
`_BARRIER_PAIRED_KINDS` set that liveness rolls up from the
paired `BARRIER_WAIT`.

---

### MEDIUM-4: CSE mutates `op.operands` in place; some passes are immutable, some aren't
**File**: `helixc/ir/passes/cse.py:122`, `helixc/ir/passes/const_fold.py:334`

```python
# cse.py:122 — mutates op.operands in place
op.operands[i] = rewrites[o.id]

# const_fold.py:334 — same pattern
op.operands = [subst.get(o.id, o) for o in op.operands]
```

vs. `tile_opt.py` which uses `dataclasses.replace` to build fresh
ops/blocks/fns immutably.

**Pattern**: inconsistent invariant. The module docstring at
`tir.py:10` says "SSA with block parameters (Cranelift CLIF / Swift
SIL pattern). No phi nodes." — implying functional / immutable IR.
But TIR `Op` is `@dataclass` (not frozen), so it's mutable, and CSE +
const_fold take advantage of this. The cycle-18 audit-C C18-C1 fix at
`cse.py:147` (`seen[key] = list(op.results)`) already documents the
aliasing risk and copies defensively at one site, but the mutation at
line 122 is still live.

**Why it matters**: if any downstream code aliases the same `Op`
across two blocks (which shouldn't happen today but isn't enforced),
the in-place mutation of `operands` propagates the rewrite to the
unrelated alias. Also, this prevents the IR from being shared across
parallel pass invocations.

**Fix**: either make `tir.Op` `frozen=True` (forcing immutable
rewrites everywhere — significant refactor) or document the mutation
policy explicitly at the top of `cse.py` / `const_fold.py` and add
an `assert` that the function being processed hasn't been frozen by
a prior pass (e.g., a `_frozen` attribute on `FnIR`).

---

### MEDIUM-5: `fdce.live_function_names` doesn't track MODIFY's `xform` SSA operand
**File**: `helixc/ir/passes/fdce.py:53-56`, cross-ref `helixc/ir/lower_ast.py:4681, 4700`

```python
# fdce.py:53-56 — only verifier_fn attr is tracked
elif op.kind == tir.OpKind.MODIFY:
    vfn = op.attrs.get("verifier_fn")
    if isinstance(vfn, str):
        called.add(vfn)
```

vs. `lower_ast.py:4681`:

```python
xform = self._lower_expr(expr.transformation) or self.builder.const_int(0)
```

Function references at `lower_ast.py:1731-1732` silently lower to
`const_int(0)`:

```python
if expr.name in self.functions:
    return self.builder.const_int(0)
```

**Pattern**: lost function reference. MODIFY's `xform` operand is
supposed to identify a transformation function; the lowerer turns
function-name expressions into `const_int(0)`, losing the function
identity. FDCE then has no way to mark the xform function live, so
it's silently DCE'd from the module — and the runtime sees a 0 where
a function reference was expected.

**Why it matters**: two MODIFY calls with different `xform` functions
both lower to operand `const_int(0)`. The backend (`x86_64.py:4413+`)
only uses the `verifier_fn` attr, so today this is dormant — but the
moment any MODIFY consumer actually reads the `xform` operand, the
function identity has been lost. FDCE then strips the xform fns as
dead.

**Fix**: lower function references to a proper `OpKind.FN_REF` op
(new kind needed) that carries the fn name in `attrs["target"]`. FDCE's
call-graph builder then walks `FN_REF` ops and adds the targets to
`called`. Until that lands, document the current behavior loudly in
the `lower_ast.py:1731-1732` comment (currently no comment, just a
silent `return const_int(0)`).

---

## LOW findings

### LOW-1: `_lower_stmt` Let-with-unknown-struct silently falls back to `const_int(0)`
**File**: `helixc/ir/lower_ast.py:1422-1429`

```python
if stmt.value is not None and isinstance(stmt.value, A.StructLit):
    slit = stmt.value
    flat_paths = self._struct_flat_paths.get(slit.name)
    if flat_paths is None:
        # Unknown struct (typecheck would have flagged) — fall
        # through to default-value binding to avoid crashing.
        self._bind(stmt.name, self.builder.const_int(0))
        return
```

The "typecheck would have flagged" defense is real for normal
compilations, but if `typecheck` is bypassed (e.g., `--no-typecheck`
mode, IR fuzzing, an externally-constructed AST passed straight to
`lower_ast.lower`), the silent `const_int(0)` produces wrong-answer
output with no diagnostic. Replace with an explicit raise — typecheck
holes should surface as IR-lowering failures, not silent zeros.

### LOW-2: `_lower_block` returns `None` when `final_expr` lowers to `None`
**File**: `helixc/ir/lower_ast.py:1173-1177`

```python
if block.final_expr is not None:
    return self._lower_expr(
        block.final_expr,
        expected_rec_enum=expected_rec_enum)
return None
```

If `_lower_expr(block.final_expr)` returns `None` (e.g., the final
expr is a `Return` / `Break` / `Continue` with no value), the caller
(typically `_lower_expr`'s `A.Block` arm or `_lower_stmt`'s Let
arm at line 1624) substitutes `const_int(0)`. That's a benign
shape-coercion today, but a `Block { ...; return 5 }` whose `Return`
expr semantically diverges should NOT be lowered as `const_int(0)` —
that's a value where the language semantics says "unreachable."

Fix: differentiate "no final expr" (legitimate void block) from
"final expr lowered to None" (early-exit / unreachable) and raise on
the latter.

### LOW-3: `tile_ir.fmt_module` formats unknown TileOp kinds via `op.kind.value` without coverage check
**File**: `helixc/ir/tile_ir.py:248-256`

Pretty-printer falls through to `op.kind.value` for any TileOp kind.
Not a miscompile (just diagnostic output), but a future TileOp added
without a TIR-side equivalent will still format cleanly here while the
real reachable miscompile site (e.g., a TILE_IR -> PTX emitter arm
that's missing) goes unmarked. Mirror the
`tir.fmt_type` / `tir.fmt_dim` fall-through pattern: emit
`<{type(op).__name__}: UNHANDLED kind={op.kind.name}>` so it's at
least scannable.

---

## Cross-cutting observations

1. **The opt-in classification pattern recurs**: `dce.SIDE_EFFECT_KINDS`,
   `cse.PURE_KINDS`, `effect_check.OP_EFFECTS`, `tile_opt._SIDE_EFFECT_KINDS`
   are all opt-in lists with implicit "everything else is in the
   complement bucket" semantics. Each is a future-drift trap. A single
   `OP_CLASSIFICATION` table mapping every `tir.OpKind` to one of
   `{PURE, SIDE_EFFECT_LOCAL, SIDE_EFFECT_OBSERVABLE, NONDETERMINISTIC,
   CONTROL, NEEDS_REVIEW}` — asserted complete at module load — would
   eliminate all four classes at once. Each pass would query the
   classification table rather than maintaining its own set.

2. **`hasattr` / `getattr(..., default)` is silent-failure machinery**.
   HIGH-4 uses both. `getattr(x, "name", default)` is fine when the
   default is genuinely indistinguishable from a real value (a missing
   `name_hint` defaulting to `None`), but a memspace of `""` is NOT
   indistinguishable from `MemSpace.SMEM`, `MemSpace.REG`, etc. — it's
   a sentinel that should fail loudly.

3. **The pretty-printer fall-throughs at `tir.py:533, 547`** (`return f"<{type(d).__name__}>"`) are diagnostic-only and acceptable, but they conceal which `Dim` / `TIRType` subclasses lack arms.
   Add a `# noqa: SILENT-DIAG-FALL-THROUGH` marker or run a lint check
   that fails CI if a new subclass is added without an `fmt_*` arm.

4. **Consistency between sibling passes**: DCE's `SIDE_EFFECT_KINDS`
   and effect_check's `OP_EFFECTS` overlap but aren't enforced to
   be consistent. `TRAP` is in both (good); `FFI_CALL` is in both
   (cycle 22 fix); but `TENSOR_STORE` is in neither. A future audit
   should add a cross-table consistency check at module load:
   `assert OP_EFFECTS.keys() ⊆ SIDE_EFFECT_KINDS` (anything with an
   effect must be preserved by DCE).
