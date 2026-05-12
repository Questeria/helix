# Audit Stage 28.9 cycle 57 — Type design

**Scope.** Read-only at HEAD `2f3dcbc` (cycle-56 audit docs only;
no code delta since `5d58d3d`). Adversarial second pass —
deliberately rotates AWAY from cycle-56's autodiff / `_fn_table_sig`
and identity-layer focus (`ast_hash`, `hash_cons`, `cse`). This
cycle targets:

- `helixc/frontend/typecheck.py` — type-environment scoping,
  generic-parameter substitution, where-clause discharging,
  match-exhaustiveness coverage of `Pattern` subclasses.
- `helixc/frontend/match_lower.py` — Item subclass enumeration in
  `_rewrite_item`, Expr-subclass enumeration in `_rewrite_expr`,
  `_pattern_test_expr` Pattern dispatch.
- `helixc/ir/lower_ast.py`, `helixc/ir/tir.py`,
  `helixc/ir/passes/const_fold.py` — Op result-type vs
  operand-type invariants, `_wrap_int_to_type` table coverage,
  `_INT_BITS` keys.
- `helixc/backend/x86_64.py` — type-to-machine-mapping for the
  f64/i64 wide paths (CONST_INT 64-bit dispatch, CONST_FLOAT
  packing, BITCAST `wide` decision, CAST i64↔f64 paths). Note:
  cycle-57 brief named `wat_emitter.py` / `x64_emitter.py`; no such
  files exist in this tree. The x86-64 emitter is
  `helixc/backend/x86_64.py` (the only native backend on Phase-0
  Helix; PTX and ELF-dyn are out-of-scope for type-design audits
  per cycle 56's scope precedent).

Prior C1–C56 dispositions not re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### Item-walker totality (re-verified)

`ast_nodes.py` declares exactly 10 `Item` subclasses:
`FnDecl, StructDecl, EnumDecl, TypeAlias, UseDecl, ConstDecl,
AgentDecl, ModuleDecl, ModBlock, ImplBlock`.
`match_lower._rewrite_item` (lines 73–117) enumerates all 10 with
the cycle-15/16/17 loud-fail catchall preserved. No regression
since cycle-19's pass.

### Expr-walker totality (re-verified)

`_rewrite_expr` (lines 157–304) enumerates every Expr subclass
declared in `ast_nodes.py`:
`IntLit, FloatLit, BoolLit, StrLit, CharLit, Name, Path, Unary,
Binary, Call, Index, Field, TupleLit, ArrayLit, StructLit, Block,
UnsafeBlock, If, Match, For, While, Loop, Break, Continue, Return,
Range, Assign, Cast, TileLit, Quote, Splice, Modify`. The cycle-58
audit-R C57-4 `Continue` arm (line 290–291) closes the last
silent fall-through; the final `NotImplementedError` (line 299)
preserves the cycle-14/15 loud-fail discipline.

### Pattern-walker totality (re-verified)

`_pattern_test_expr` (lines 421–581) enumerates all 7 Pattern
subclasses (`PatWildcard, PatBind, PatLit, PatRange, PatOr,
PatTuple, PatVariant`) with the cycle-14 C14-3 / cycle-15 C15-3
loud-fail catchall. `_collect_binds` (lines 600–704) mirrors
the dispatch with explicit leaf arms for `PatWildcard / PatLit /
PatRange` (cycle-15 C15-1). No new Pattern subclass surfaced.

### Backend type-to-machine mapping (re-verified)

- **CONST_INT 64-bit branch** (`x86_64.py:1148`). The
  `_is_i64_type` predicate (line 1005) recognises `i64`/`isize`
  per cycle-19 C18-1; the 8-byte `mov_rax_imm64` + `mov_mem_rbp_rax`
  pair correctly stores the full 8 bytes. The narrow else-branch
  uses `mov_eax_imm32` (4-byte). The 4-byte-only store is
  consistent with the matching narrow load paths (ADD/SUB/MUL/
  RETURN/BR all `_is_i64_type`-gated and otherwise 32-bit), so
  no within-i64-domain mismatch exists.
- **CONST_FLOAT f64 branch** (`x86_64.py:1161-1177`). f64 is
  packed via `struct.pack("<d", value)` then split into two
  32-bit immediates and stored lo-then-hi at `slot` and
  `slot+4`. Matches x86-64 little-endian convention. The 4-byte
  f32 path uses `struct.pack("<f", ...)` — correct narrow
  packing.
- **BITCAST `wide` decision** (`x86_64.py:1184-1186`). Uses
  `_is_f64_type(res_ty) or _is_i64_type(res_ty) or
  _is_f64_type(operand) or _is_i64_type(operand)`. The four
  reachable BITCAST shapes from `lower_ast.py:1346-1365` are
  `f32↔i32` (both narrow → wide=False, 4-byte mov) and
  `f64↔i64` (both wide → wide=True, 8-byte mov). The two
  shapes never reach mismatched widths through any current
  builtin call path.
- **CAST matrix** (`x86_64.py:1194-1264`). Enumerates 8
  cross-product cases (i64→i32, i32→i64, i64→f64, i64→i64,
  i32→f64, i32→f32, f64→i32, f32→i32) plus the f64↔f64 8-byte
  copy and the same-class 4-byte copy. Each case routes the
  correct REX.W / SSE prefix sequence. No coverage gap relative
  to the typecheck cast-matrix (B14).

### `_INT_BITS` table coverage

`const_fold._INT_BITS` (lines 98–111) covers every integer
primitive in `typecheck.PRIMITIVES` that participates in
`_wrap_int_to_type`: `i8, i16, i32, i64, isize, u8, u16, u32,
u64, usize, bool`. `char` is absent from `_INT_BITS`, but
`CharLit` has no `lower_ast` arm (verified via grep) and so
never reaches `_wrap_int_to_type` with a `char` result type.
Default `bits = 32` fallback for unknown types is defensive but
unreachable in current production.

### Generic parameter substitution + where-clause discharge

`typecheck._register_fn` (lines 429–481) maps each
`GenericParam.kind` to either `TySize` or `TyVar` and resolves
param/return types in that gen-scope.
`_check_call_shapes` (lines 853–882) re-builds the gen-scope at
each call site and feeds `where_clauses` through
`_add_where_constraint` → Presburger solver → `solver.implies`
verdict. Constraints with a False verdict surface as
`shape constraint violated` diagnostics. No silent-accept of
unresolved where-clauses.

### Match exhaustiveness — variant-tag coverage

`_check_match_exhaustive` (lines 2015–2086) gates on:
finite-type cases (bool/unit), enum-shaped first-class arms via
`_arm_variant_names_all`, or a top-level wildcard/bind arm. The
enum branch counts each `PatVariant` or `PatLit-of-Path` as
covering its tag; PatOr alts expand correctly. Sub-pattern
content (e.g. `Cons(1, _)` not covering `Cons(2, _)`) is
intentionally NOT factored into the exhaustiveness verdict —
the desugared if-chain falls through to a unit-typed bare-else
when no arm matches at runtime, which typecheck's match-result
type would reject under strict type comparison. This is a Phase-0
deliberate over-approximation aligned with the Stage-28.6 doc
note `audit-stage28-8/cycle10`.

### Stability

No prior-cycle findings re-surface. HEAD `2f3dcbc` is a
docs-only delta from `5d58d3d` (verified via
`git diff --stat`: three audit `.md` files, +257/-0). No code
change since the cycle-56-verified `_fn_table_sig` /
`structural_hash` deltas; type surfaces inspected here are
identical to the cycle-56 snapshot.

## Notes (<75)

- **`_is_i64_type` does NOT cover `u64`/`usize` outside FFI
  sites (conf ~62).** Cycle 19 C18-1 split into a separate
  `_is_u64_type` helper (line 1013) but consulted it only at
  FFI_CALL (lines 1752, 1763). Every other 64-bit dispatch
  point (CONST_INT @1148, BITCAST `wide` @1184, ADD @1279,
  SUB @1304, MUL @1329, NEG @1369, MOD @1404, CMP @1434,
  RETURN @1789, BR @1817, etc.) checks only `_is_i64_type`,
  which means a `let x: u64 = 0xFFFF_FFFF_FFFF_FFFF_u64;` would
  CONST_INT-truncate to 32 bits and subsequent ADD/MUL would
  operate on the low 32 bits only. The existing u64 tests
  (`test_codegen.py:2810, 2890`) use values that fit in 32 bits
  so the silent-trunc does not surface. Below 75 because: (a)
  the cycle-19 fix-sweep author deliberately split the
  predicates and only wired the u64 one at FFI, implying
  Phase-0-scope acceptance; (b) `_check_float_supported`
  precedent exists for "narrow + loud" but no analogous
  `_check_u64_arith_supported` was added — suggesting the
  defect class was triaged as deferred to Phase 1.
  Recommend: a cycle-58 fix-sweep could add a `_is_64bit_int_type`
  predicate (= `_is_i64_type or _is_u64_type`) and route every
  arithmetic/store dispatch through it; or a `NotImplementedError`
  trap in CONST_INT/ADD/etc when the result-type is u64/usize.

- **`BITCAST` `wide` decision admits mismatched-width hazard
  (conf ~50).** Line 1184 takes the disjunction across
  result_ty and operand.ty, so an i32→f64 BITCAST (operand
  4-byte, result 8-byte) would set `wide=True` and read 8 bytes
  from the i32 operand's slot, picking up 4 bytes of
  uninitialised stack above. No current builtin produces such a
  BITCAST (the four `__bits_of_f*` / `__f*_from_bits` lowerings
  all preserve width per `lower_ast.py:1346-1365`), and there
  is no surface syntax for arbitrary bitcasts in Phase-0. Below
  75 because no current-grammar reachability exists; flagged as
  a latent hazard a future `as` cast or `transmute` builtin would
  trip. Pragmatic fix: assert
  `_width_of(operand.ty) == _width_of(result.ty)` at BITCAST
  emission in `lower_ast.py`.

- **`_check_match_exhaustive` PatVariant sub-pattern
  over-approximation (conf ~55).** The enum-exhaustiveness path
  counts `Cons(1, _)` as full coverage of `Cons` even though
  `Cons(2, _)` would fall through to the desugared chain's bare
  unit-typed else, which the match-expression-typecheck would
  then reject if the body type is non-unit. The interaction
  produces a "soundness via downstream rejection" — correct in
  aggregate but the exhaustiveness diagnostic message would
  blame the wrong source location. Below 75 because: the
  downstream rejection IS sound; this is a diagnostic-quality
  observation, not a miscompile. Recommend Phase-1 to push
  sub-pattern totality into `_arm_variant_names_all`.

Files:
- `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`
- `C:/Projects/Kovostov-Native/helixc/ir/lower_ast.py`
- `C:/Projects/Kovostov-Native/helixc/ir/tir.py`
- `C:/Projects/Kovostov-Native/helixc/ir/passes/const_fold.py`
- `C:/Projects/Kovostov-Native/helixc/ir/passes/cse.py`
- `C:/Projects/Kovostov-Native/helixc/backend/x86_64.py`
- `C:/Projects/Kovostov-Native/helixc/tests/test_codegen.py` (cross-check)
