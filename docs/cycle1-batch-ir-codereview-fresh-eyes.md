# Cycle 1 Batch IR — Fresh-Eyes Code Review (Auditor 5)

Audit date: 2026-05-18
Scope: helixc/ir/ (tir.py, tile_ir.py, lower_ast.py, passes/*.py)
Auditor: feature-dev:code-reviewer

## Verdict
NOT_CLEAN — 2 HIGH + 4 MEDIUM

## HIGH findings

### HIGH-1: DCE unconditionally keeps every zero-result op, including pure ones
**File**: helixc/ir/passes/dce.py:145-147
**Problem**: In `dce_function`, the removal sweep contains:
```python
if not op.results:
    new_ops.append(op)
    continue
```
This keeps every op with an empty `results` list regardless of membership in `SIDE_EFFECT_KINDS`. The intent was to preserve side-effecting result-less ops (STORE_VAR, STORE_ELEM, RETURN, COND_BR, etc.), but those are already handled by the preceding `if op.kind in SIDE_EFFECT_KINDS` guard. The second guard is therefore both redundant for side-effecting ops and incorrectly permissive for any pure zero-result op. Contrast with `tile_opt.dead_tile_elim` (tile_opt.py:105-108), which has the correct inverse: it explicitly drops zero-result ops that are NOT side-effecting. The two passes are asymmetric.
**Why it matters**: A future pass or new opcode that emits a pure zero-result op (e.g., a fence/annotation node without an SSA result slot) will be silently retained by DCE forever with no diagnostic. The mismatch with `dead_tile_elim` is a latent divergence that will manifest as IR pollution when the tile and scalar DCE paths are compared on the same program. The bug is invisible in existing tests precisely because no current codepath emits a reachable pure zero-result op.
**Fix**:
```python
if not op.results:
    if op.kind in SIDE_EFFECT_KINDS:
        new_ops.append(op)
    else:
        removed_total += 1
        changed = True
    continue
```
This matches the `dead_tile_elim` pattern and closes the asymmetry.

### HIGH-2: `_arena_push_slots` silently returns `const_int(0)` when called with an empty slot list
**File**: helixc/ir/lower_ast.py:563-571
**Problem**: If `slots` is empty the loop body never executes, `start_idx` stays `None`, and the function returns `self.builder.const_int(0)` — a real SSA constant with integer value 0, emitted into the IR block. This is then bound as the recursive enum's arena index. Arena index 0 is a valid live slot once any other push has occurred, so the caller's recursive enum binding silently aliases whatever already lives at position 0.

All current callers always prepend `tag_v` before calling (`[tag_v] + payload_vals`), so the empty-list path is unreachable today. However there is no assertion enforcing this contract, and the fallback silently injects `const_int(0)` into the IR rather than failing loudly. A future caller that assembles `slots = payload_vals` (omitting the tag slot) will produce an incorrect recursive enum arena index with no diagnostic.
**Why it matters**: Silent structural corruption of recursive enum arena indices. `const_int(0)` is a valid SSA value, so DCE and the backend will not flag it. The program silently reads wrong-enum-variant data at runtime via `ARENA_GET(0)`.
**Fix**:
```python
def _arena_push_slots(self, slots: list[tir.Value]) -> tir.Value:
    if not slots:
        raise AssertionError(
            "_arena_push_slots called with empty slot list; "
            "callers must always include at least the tag slot")
    ...
```

## MEDIUM findings

### MEDIUM-1: CAST folding is documented in the module docstring but has no implementation
**File**: helixc/ir/passes/const_fold.py:14, _try_fold_op (entire function)
**Problem**: The module docstring at line 14 explicitly lists `CAST (between numeric scalars)` as a folded operation. There is no CAST arm anywhere in `_try_fold_op`. A `CAST(CONST_INT(5), to_ty=f32)` produces an unfolded CAST op even after const-fold runs. CSE correctly includes `tir.OpKind.CAST` in `PURE_KINDS` for deduplication, so redundant CASTs are deduplicated but constant CASTs are never folded to scalar literals.
**Why it matters**: The docstring is an implicit contract that downstream passes rely on. Every const-to-float CAST in an otherwise fully-constant expression chain escapes the folder and becomes a runtime instruction. CAST-chains will carry unnecessary ops and will not benefit from algebraic identity forwarding.
**Fix**: Add a CAST arm to `_try_fold_op`. For `CAST(CONST_INT(v), to_ty=TIRScalar("f32"))` emit `CONST_FLOAT(float(v))` with the NaN guard from the existing float-arith path. For `CAST(CONST_FLOAT(v), to_ty=TIRScalar("i32"))` emit `CONST_INT(int(v))` wrapped through `_wrap_int_to_type`. Restrict to scalar-to-scalar numeric casts only.

**Update (post-audit)**: Cycle 1 fix batch 8 commit 4fc7bb2 addressed this by correcting the docstring to say CAST is NOT YET folded (Stage 110+ improvement) rather than implementing the fold. This is an acceptable resolution.

### MEDIUM-2: `_try_algebraic_identity` reuses the source op's result Value object in CONST_INT nodes, inheriting its source-op type
**File**: helixc/ir/passes/const_fold.py:197-200, 213-215, 221-223, 239-242
**Problem**: Every folding path in `_try_algebraic_identity` returns:
```python
return tir.Op(kind=tir.OpKind.CONST_INT, results=[res], attrs={"value": 0})
```
where `res = op.results[0]`. For a comparison self-compare fold that returns 1, `res.ty` is `TIRScalar("bool")`, so the emitted `CONST_INT` carries type `bool`. Numerically harmless today (`_INT_BITS["bool"] = 32`), but structurally: a `CONST_INT` op whose sole result has type `TIRScalar("bool")` is a type-system anomaly. Any downstream IR validator enforcing that `CONST_INT.results[0].ty` must be an integer scalar name will false-fire on these folds. The pattern is pervasive — all algebraic-identity folds in `_try_algebraic_identity` share this shape.
**Why it matters**: Type-precision debt that scales with the number of algebraic-identity folds. A future IR validator (a natural addition to the audit pipeline) will break on these nodes.
**Fix**: Allocate a fresh `Value(ty=TIRScalar("i32"))` for the replacement result inside `_try_algebraic_identity`. The caller's `defs` substitution map already handles remapping old result ids to new ones during the next fold iteration.

### MEDIUM-3: `fdce.py` imports `re` inside `live_function_names` on every call
**File**: helixc/ir/passes/fdce.py:41
**Problem**: `import re` and `_ID_RE = re.compile(...)` appear inside the function body. Python's import cache makes this cheap at runtime, but it violates the uniform top-of-file import convention used by every other file in the IR subsystem (`tir.py`, `tile_ir.py`, `const_fold.py`, `cse.py`, `dce.py`, `effect_check.py`). A project-wide import linter will not see `re` as a dependency of `fdce.py`, and the regex is logically re-created on every call.
**Why it matters**: Convention violation; obscured dependency; misleading style for maintainers.
**Fix**: Move `import re` to module top-level and compile `_ID_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")` once at module scope.

### MEDIUM-4: `tile_ir.py` SCALAR_OP_MAP missing `CONST_BOOL` — a reachable TIR opcode has no tile-lowering path
**File**: helixc/ir/tile_ir.py:159-179
**Problem**: `TirToTileLowerer.SCALAR_OP_MAP` covers 19 of ~80 `OpKind` members; all others raise `NotImplementedError` (correct fail-closed design). However `CONST_BOOL` is absent. `lower_ast._lower_expr` emits `CONST_BOOL` for every `BoolLit` and for `@kernel`-tagged functions that compare booleans. A kernel body containing `let flag: bool = true;` produces a `CONST_BOOL` TIR op; when `lower_to_tile` is called on that module it raises `NotImplementedError: Tile IR lowering does not support TIR op const.bool`. No existing kernel test uses a boolean literal, so the gap is untested.
**Why it matters**: Any kernel body using a boolean literal will fail at tile-lowering with no actionable diagnostic. Because the tile-lowering path is only exercised during PTX emission, the failure surfaces late and at a confusing IR layer. The fix is trivial and closes a gap in a first-class opcode.
**Fix**: Add `tir.OpKind.CONST_BOOL: TileOpKind.SCALAR_CONST_INT` to `SCALAR_OP_MAP`. Boolean is represented as i32 (0/1) in TIR, so `SCALAR_CONST_INT` is the correct mapping. Additionally add a comment block listing intentionally-unsupported ops (MATMUL, CONV*, REDUCE_*, GRAD, VMAP, etc.) as "deferred to v0.2" so future op additions get explicit guidance.

## LOW findings

### LOW-1: `tir.py:fmt_dim` accesses `DimExpr.args` without checking length
**File**: helixc/ir/tir.py:531-532
`fmt_dim` does `d.args[0]` and `d.args[1]` unconditionally. `DimExpr.args` is typed `tuple[Dim, ...]` (variable-length). A `DimExpr` with arity != 2 raises `IndexError` in the pretty-printer instead of a useful diagnostic. All current producers use binary forms, but the struct type does not enforce it.
**Fix**: Add `assert len(d.args) == 2` in `fmt_dim`, or change `DimExpr` to carry `left: Dim` and `right: Dim` explicit fields.

### LOW-2: `cse.py:_find_value_by_id` is defined but never called
**File**: helixc/ir/passes/cse.py:152-164
`_find_value_by_id` is a private module-level helper not referenced anywhere in `cse.py` or any other IR pass file. It appears to be scaffolding for a deferred cross-block CSE (explicitly deferred in cse.py:106) that was never implemented and never cleaned up.
**Fix**: Remove the dead function. If cross-block CSE is planned, note it in a TODO comment at the planned call site rather than leaving unreachable code.

### LOW-3: `const_fold.py:_is_int_const` and `_is_float_const` are dead and their `consts` parameter is unused
**File**: helixc/ir/passes/const_fold.py:247-251
Both helpers ignore their `consts: dict` parameter entirely and are never called anywhere in the file. The actual const-identification logic is inlined at every site in `_try_fold_op`. These are dead scaffolding.
**Fix**: Remove both functions.

### LOW-4: `lower_ast.py:_lower_tile_matmul_let` docstring uses wrong variable names
**File**: helixc/ir/lower_ast.py:1252-1255
The docstring says "For each (i, k) with i in 0..N, k in 0..M" but the code uses `kk` for the output-column iterator. `k` in the code refers to the shared contraction dimension K, not the output column M. The comment "k in 0..M" inverts the meaning.
**Fix**: Correct to "For each (i, kk) with i in 0..N (output rows), kk in 0..M (output cols); for j in 0..K (contraction dim)".

## Cross-check vs concurrent session

The concurrent session reported (from audit-sweep-progress.md):
- type-design: 3H + 5M (1H FP, 2H->MED, 5M)
- silent-failure: 3H + 3M (all REAL)

My findings have no direct item-level overlap. The concurrent auditors focused on the frontend (typecheck.py, autodiff.py) and type-wrapper completeness — outside the IR subsystem files audited here. My HIGH-1 (DCE zero-result logic gap) and HIGH-2 (_arena_push_slots empty-list) are structural logic bugs not captured by type-design or exception-swallowing scans. My MEDIUM-2 has conceptual kinship with type-design findings but targets a different site (_try_algebraic_identity result Value reuse). I flag HIGH-1 and HIGH-2 as must-fix before the Batch IR cycle counter can advance.

## Post-audit reconciliation (added by parent session)

Cycle 1 fix batch 8 (commit 4fc7bb2) was authored by the concurrent session in parallel with this audit. It addressed 4 HIGHs from its own independent 3-auditor IR pass:
- lower_ast.py silent-failure HIGH-1 (`_lower_expr` catch-all)
- lower_ast.py silent-failure HIGH-3 (Field/Index silent-None)
- tile_opt.py silent-failure HIGH-2 (Stage 64 ops blindness — inverted to allowlist)
- const_fold.py fresh-eyes IR-1 (CAST docstring drift)

My HIGH-1 (DCE zero-result) and HIGH-2 (_arena_push_slots empty-list) remain OPEN and are NEW findings the concurrent session did not surface. Recommend incorporation into fix batch 9 or 10.
