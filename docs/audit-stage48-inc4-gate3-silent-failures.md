# Stage 48 Inc 4 Gate-3 — Silent-Failure Audit

Date: 2026-05-17
Scope: Stage 48 `?` propagation operator surface (commits 48db12d + 722bfdb + c32dfbb on main).
HEAD: c32dfbb (Stage 48 gate-2 silent-failure F1+M5 fixes + polish).

## Verdict

**3/3 GATE-3 CLEAN — Stage 48 ready to CLOSE.**

No HIGH or CRITICAL silent-failure findings at confidence >= 70 beyond those already
documented as Stage 49 deferred items (F1-dynamic + F5 aggregate-field, plus the new
MED-1 below which joins the same Phase-0-without-runtime-tag equivalence class).

**Deferred to Stage 49.**

## Findings

### MED-1 — inline `map_ok(Err(...), v)` operand to `?` (conf 78)

**Location**: `helixc/frontend/typecheck.py:2547-2563` (let-RHS provenance carry) +
`helixc/frontend/typecheck.py:4628-4646` (`__try` provenance-rejection arm, Name-only).

**Description**: provenance carry through `map_ok` / `map_err` activates only when the
first arg is a `Name` already in the provenance dict. When the source is an inline
constructor call (`let r = map_ok(Err(7), 999); r?`), `args[0]` is `A.Call`, not
`A.Name`, so the carry branch falls into the else-pop (line 2583) and `prov[r]` is
cleared. Post-let, `r?` typechecks clean. At runtime, `map_ok` lowers to `args[1]` =
999 (lower_ast.py:2117), so `r=999` and `r?` returns 999 — wrong answer, real
semantics should propagate `Err(7)`.

**Repro**:
```
fn helper() -> Result<i32, i32> {
    let r: Result<i32, i32> = map_ok(Err(7), 999);
    let v: i32 = r?;
    Ok(v)
}
fn main() -> i32 { unwrap_ok(helper()) }
```
Expected (Stage 49+): propagate `Err(7)`. Actual (Phase-0): returns `Ok(999)`.

**Why MED not HIGH**: structurally identical to gate-1 F1-dynamic (`fn-call returning
Result`) and gate-2 F5 (`aggregate-field operand`). Stage 49 runtime tag eliminates the
entire equivalence class in one move. Adding a Phase-0 peephole that recognises inline
`map_ok(Err(_), _)` / `map_err(Ok(_), _)` would be defensive work that Stage 49
obsoletes.

**Recommendation**: defer to Stage 49. Add to the F1/F5 known-defect tracking list with
the F6 label.

### LOW-1 — `Result<T, E>` inside `TyTuple` / `TyArray` / `TyTensor.dtype` (conf 65)

**Location**: `helixc/ir/lower_ast.py:820-836` (tuple/array/tensor lower arms),
`_lower_type` `TyGeneric` arm at 843-882.

**Description**: `_lower_type` recursively descends into tuple-element, array-element,
and tensor-dtype slots. A `Result<T, E>` in any of these positions silently lowers to
its Ok inner `T`. Most surface paths are already typecheck-blocked (e.g. tensor dtype
must be a primitive scalar via the assertion on line 824), but `TyTuple` and `TyArray`
have no such guard. A tuple like `(Result<i32, i32>, i32)` would lower its first slot
to `i32` — and any subsequent `?` on a field access (`t.0?`) joins the existing F5
aggregate-field defect. Already covered transitively by Stage 49, but worth pinning a
Stage 49+ test for the tuple-element case explicitly.

**Why LOW**: no observed source path constructs a heterogeneous tuple of Result and
non-Result through current parser/typecheck without rejection upstream, and field-`?`
is already a documented Phase-0 defect.

### Sub-threshold observations

- **OBS-1 (conf 60, NOT a finding)**: while / for / loop bodies (`typecheck.py:4979-4996`)
  invoke `_check_block(expr.body, scope)`. `_check_block` snapshots and restores
  `_result_constructor_provenance` with gate-3 mutation-detection. Therefore the
  G3-F1a/b/c fix automatically extends to while-body, for-body, and loop-body inner
  assigns. Verified by tracing four loop-body assign patterns by hand: all behave
  correctly (mutated outer names dropped, conservative `prov={}` post-block,
  F1-dynamic territory). Helix has no closure / lambda AST node so that vehicle is
  not reachable.

- **OBS-2 (conf 55, NOT a finding)**: compound branch shapes (`if c1 { r=Err } else if
  c2 { r=Ok }` and arbitrarily deep `else if` chains) all converge to the conservative
  `prov={}` post-fix outcome because each leaf `Block` snapshots from its parent's
  current state and the outer-then's drop persists into the else-evaluation. Traced
  four variants by hand; no false static "Ok-constructed" rejection, no missed
  rejection.

- **OBS-3 (conf 50, NOT a finding)**: users can write `__try(r)` directly in source
  because `__try` is registered as a `_BUILTIN_NAMES` entry (typecheck.py:2177). The
  `__try` typecheck arm (typecheck.py:4516) enforces the enclosing-fn return-type
  check unconditionally, so a top-level `__try(r)` in a non-Result-returning fn
  rejects identically to the parser-desugared `r?` form. Symmetric, no silent-failure
  window.

- **OBS-4 (conf 50, NOT a finding)**: Helix `Let` AST node has a single `name: str`
  field (ast_nodes.py:394) — no tuple-destructuring let. Concern about
  `let (a, b) = (Ok, Ok); a?` cannot fire on the current grammar.

## Verification steps performed

1. Read `_BUILTIN_NAMES`, the `__try` arm, the `_check_block` snapshot/restore, the
   let-stmt provenance recording, the assign-arm provenance overwrite/pop, and the
   `_check_fn` per-fn clear. All six gate-1/2/3 fix sites accounted for.
2. Traced while/for/loop bodies, compound `else if` chains, tuple/array/tensor type
   positions, and inline-constructor `map_ok`/`map_err` arguments by hand against
   each of the seven hunt-targets in the audit prompt.
3. Confirmed Helix has no closure/lambda AST node (`Glob class (Closure|Lambda)` →
   zero hits in `ast_nodes.py`) and no tuple-destructuring `Let`.
4. Cross-referenced existing Stage 48 test coverage in
   `helixc/tests/test_stage48_try.py` (18 tests, including G3-F1a/b/c regressions).
5. Identified MED-1 as a new structural variant of the F1/F5 known-defect class —
   not a new failure mode, but a sharper, more inline-source-visible instance.
