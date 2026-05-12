# Audit Stage 28.9 cycle 76 — Type design

**Scope:** `HEAD 92ffc5a` (cycle-75 audit-clean commit; identical to working
HEAD `b4e793c` for in-scope source — the only diff is
`helixc/tests/test_deprecated.py`).

**Areas audited (narrow, per cycle-76 prompt):**

1. `helixc/ir/lower_ast.py` — Op result-type vs operand-type invariants
   (the prompt also names `helixc/ir/builder.py`; no such file exists.
   The IR builder is `tir.IRBuilder` in `helixc/ir/tir.py`, which is
   thin and was inspected in passing).
2. `helixc/frontend/struct_mono.py` — `mangle_struct` injectivity for
   `TyGeneric` with multiple args.
3. `helixc/frontend/typecheck.py` — limited internal type-environment
   correctness (Scope shadowing, `_check_block` / `_check_stmt` /
   `_register_fn` / `_check_fn_body` / `_compatible` /
   `_size_compatible`; pre-flatten pipeline-ordering questions
   intentionally deferred per scope).

**Deferred-known items NOT re-flagged** (per scope):

- `mangle_struct` / `_mangle_ty(TyGeneric)` `_`-separator non-injectivity
  on underscore-bearing names — cycle-65 sub-75 observation (conf ~65 /
  ~55). Same hazard still latent; the only mitigation since cycle-65 is
  the `existing`-set check at `struct_mono.py:376`, which simply drops
  the second instantiation silently rather than aliasing. Not re-flagged.
- `_lower_type` lowering generic type params to `TIRScalar(name)` with
  i32-sized ABI (audit-2-deep-research bug G, documented in source at
  `lower_ast.py:350-362`).
- `&&` / `||` typecheck accepting non-bool operands and returning
  `TyPrim("bool")` without operand-type validation, plus the
  `lower_ast.py:1132,1135` `MUL`/`ADD` lowerings emitting
  `result_ty=bool` over potentially non-bool operand types. Design
  intent ambiguous (C-style truthiness vs strict-bool) and no prior
  audit cycle re-flagged it; my confidence ~55–70 — sub-75. Recorded
  here as a sub-75 observation, not flagged.
- `TyMemTier` tier-subsumption (cycle-5 F4, deferred Phase-1+);
  `TyDiff` sub-domain metadata (cycle-5 F2, deferred); `TyLogic`
  provenance handled separately (cycle-5 F3, deferred). All
  acknowledged in source comments.

## Findings ≥ 75 % confidence

### F1 — for-range loop increment uses i32 `one` regardless of iterator width (conf 78)

**Site:** `helixc/ir/lower_ast.py:1843-1847`

```python
cur_i = self.builder.emit(tir.OpKind.LOAD_VAR, result_ty=start_v.ty,
                          attrs={"name": iter_var})
one = self.builder.const_int(1)              # default dtype = "i32"
new_i = self.builder.emit(tir.OpKind.ADD, cur_i, one,
                          result_ty=start_v.ty)
```

**Op result-type vs operand-type invariant violated.** When the range
start expression has an integer suffix wider than i32 — e.g.
`for i in 0i64..100i64 { ... }`, lawful Helix syntax that
`frontend/typecheck.py:1645-1650` accepts without restriction —
`start_v.ty` is `TIRScalar("i64")`. The `ADD` op therefore has:

- `cur_i.ty = i64`
- `one.ty  = i32`  (from `IRBuilder.const_int(value, dtype="i32")`,
  `tir.py:432-434`)
- `result_ty = i64`

The two operands' types differ from each other and from the result
type. The backend's `ADD` dispatch at `backend/x86_64.py:1265-1289`
selects the i64 branch by **result type only** (line 1279
`elif self._is_i64_type(op.results[0].ty)`), then issues
`mov_rax_mem_rbp(r_slot)` — an **8-byte read** of `one`'s slot.

Slots are 8-byte aligned (`backend/x86_64.py:899` `self.next_slot -= 8`)
but `CONST_INT` for an i32 result writes only the low 4 bytes
(`backend/x86_64.py:1151-1153` `mov_eax_imm32` + `mov_mem_rbp_eax`).
The function prologue at `backend/x86_64.py:944-948` does **not**
zero the frame — `sub_rsp_imm32(frame_size)` only reserves space. The
upper 4 bytes of `one`'s slot therefore hold whatever was in that
stack address before the frame was entered: nondeterministic data
left over from caller frames, prior stack usage, kernel return path,
etc.

Net effect: a Helix `for i in 0i64..N_i64` loop increments `i` by
`1 + (garbage << 32)` on every iteration — silently corrupting the
counter, the loop bound test, and any body that reads `i`.

The same pattern would apply to i16/i8/u8/u16-typed range bounds
(narrower-than-i32) — though for those the backend's `else` branch
on lines 1284-1288 reads 4 bytes and truncates, which is the
"normal" narrowing. The bug is sharpest for widths > i32 because
those over-read.

**Why ≥ 75 conf:**

- The IR-level invariant violation is mechanically present in the
  emit — two operands at different widths, neither matching the
  result, with no preceding cast / zero-extend op.
- The backend's reliance on result-type dispatch is documented in the
  ADD switch and is the same shape for SUB / MUL / DIV / MOD /
  CMP_LT / etc., so the over-read pattern is not specific to ADD.
- Reachability requires only `for i in <expr>i64..<expr>i64 { ... }`
  in user source. No typecheck rule forbids it
  (`typecheck.py:1645-1650` returns `TyUnknown` regardless of start /
  end inner type, with no width or same-type constraint).
- The absence of test coverage is consistent with the bug being
  latent: I checked `helixc/tests/` for `range.*i64` / `for.*i64`
  patterns and found none. So 585-test green does not falsify the
  finding.

**Why not 100 conf:** the upper-4-bytes garbage *might* happen to be
zero on first frame entry for some platforms / configurations,
which would mask the bug in casual smoke tests. The invariant
violation is real either way (correct programs should not depend on
caller-frame contents); whether a given run observes corruption is
a separate question from whether the IR has the invariant.

**Suggested fix shape:** dispatch the constant `one` by `start_v.ty`:
`one = self.builder.const_int(1, start_v.ty.name)` (mirroring the
`IntLit` lowering at `lower_ast.py:1037-1038`), conditional on
`start_v.ty` being a `TIRScalar`. The same pattern should be audited
at `lower_ast.py:1196-1197` (the tile.get index `cols * row + col`
where `cols_v` is `const_int(cols)` defaulting to i32) and the
panic-args slot writes around `lower_ast.py:1311`-`1591`, though
those usually involve only i32 result types.

## Sub-75 % observations (not flagged)

- **`&&` / `||` lowering emits result_ty=bool over non-bool operand
  types (conf ~55).** `lower_ast.py:1132-1133` lowers
  `&&` as `MUL(l, r, result_ty=bool)` and `:1135-1138` lowers `||` as
  `ADD(l, r, result_ty=bool) -> CMP_NE 0`. If `l.ty` / `r.ty` are
  e.g. i64 (allowed because `typecheck.py:1273-1274` accepts any
  operand types and returns `bool`), the MUL/ADD operand-vs-result
  invariant is broken (i64 operands, i32-or-smaller result). The
  backend's MUL switch at `x86_64.py:1315-1338` dispatches by result
  type, taking the i32 path, and `mov_eax_mem_rbp` (4-byte) truncates
  the i64 operand silently. Below 75 because the design intent of
  the existing `&&`/`||` lowering is documented as a C-style truthiness
  normalization (`lower_ast.py:1125-1130`) and the rule "operands of
  `&&` must be bool" is not stated anywhere in the source — this is
  a typecheck-laxness issue downstream of which the lowering is
  internally consistent. If typecheck were tightened to require bool
  operands, the lowering would be correct.

- **`for i in <i32>..<i64>` (mixed-width range bounds) (conf ~50).**
  `lower_ast.py:1797-1810` binds `iter_var` with `start_v.ty` and
  `end_var` with `end_v.ty`. The CMP_LT at line 1825 uses
  `result_ty=bool` and operands `i_val` (start width) and `e_val`
  (end width). For mixed widths, the comparison silently truncates
  one side (same pattern as F1, but the comparison rather than the
  increment). Bracketed because typecheck does not reject mixed-width
  Range and reachability is the same narrow case as F1; rolled into
  F1's fix-suggestion above rather than separately flagged.

- **`mangle_struct` for multi-arg TyGeneric (conf ~65).** Already
  deferred per cycle-65 sub-75 observations; not re-flagged per
  cycle-76 scope.

- **`struct_mono._ty_key` falls through to `("?", type(t).__name__)`
  at line 304 (conf ~30).** The line-298 `if not isinstance(t,
  A.TyNode)` guard raises before the fallthrough, so the fallthrough
  is only reachable for TyNode subclasses not in the enumerated arms.
  The current enumerated arms (TyName, TyGeneric, TyTuple, TyArray,
  TyRef, TyPtr, TyFn, TyTensor, TyTile) are exhaustive over the 9
  TyNode subclasses in `ast_nodes.py:32-93`. Latent drift hazard for
  future TyNode additions; matches the `_mangle_ty` catchall pattern
  that cycle-71 promoted to loud-fail. Below 75 because no defect on
  current inputs.

- **Scope shadowing between `_register_fn`'s `gen_scope` and
  `_check_fn_body`'s rebuilt `gen_scope` (conf ~15, not a finding).**
  Both scopes define generic-param names to `TyVar` / `TySize`
  instances. `TyVar` is `@dataclass(frozen=True)` with `name: str`
  (`typecheck.py:43-46`), so the two scopes produce equal-by-value
  instances. The `_check_fn_body` body-scope uses params from
  `sig.params` (the register-scope's `TyVar`s) but bindings flow
  through `_compatible` which uses `frozen=True` equality. No
  identity drift. Correctness preserved.

- **`_compatible` catchall `return a == b` at line 2377 (conf ~25,
  not a finding).** The structural arms above (TyMemTier, TyQuote,
  TyDiff, TyLogic, TyTuple, TyArray, TyRef, TyPtr, TyFn, TyTensor,
  TyTile) all return explicitly. The remaining types (TyPrim,
  TyVar, TySize, TyStruct, TyEnum, TyUnit, TySkill) are all
  `frozen=True` dataclasses where `==` is structural. Correct.

### Stability

The cycle-65 deferred items remain deferred. F1 is a new finding
(not previously flagged in cycles C1–C75 nor in any prior deferred
list — confirmed by grep across `docs/audit-stage28-9-*.md` for
`const_int(1)`, `for.*loop.*i64`, `range.*i64`, `loop increment`).

### Verdict

**FAIL** — 1 finding at confidence ≥ 75.
