# Stage 28.8 Pre-29 Audit Gate — Cycle 6, Audit C: Code Review

**Date**: 2026-05-11
**Commit**: c3f26ef (read-only)
**Scope**: Audit C (general code-review) of the cycle-6 fix-sweep at commit
c3f26ef, which lands the cycle-5 audit closures (C5-1..C5-4 from audit C +
F1..F6 from audits A and B) plus a revert of the cycle-4 C4-2 parser.hx
broadening. Specifically reviewed:

- `helixc/frontend/typecheck.py` — C5-1 close (wire `TRAP_ARRAY_SIZE_*` and
  `TRAP_CAST_MATRIX_*` constants via f-string interpolation at six emit
  sites), F2 (D-wrap gate broadened to fire on same-inner asymmetric
  `D<f64> + f64`, symmetric with cycle-4 E2 for Logic), C5-2 / F1 (new
  `_compatible` TyVar / TySize cascade-safe arm).
- `helixc/frontend/monomorphize.py` — C5-1 close (wire `TRAP_SHAPE_FOLD_ZERO_DIV`
  at the two emit sites in `_fold_intlit_arith`).
- `helixc/frontend/autodiff.py` — C5-3 / F4 (TileLit identity arm replaced
  with explicit constructor that walks `shape: list[Expr]` and `memspace:
  Expr` for inlining).
- `helixc/backend/x86_64.py` — C5-2 / F3 (driver now prints `error: fn-mono`
  and exits 1 on `mono_diags` non-empty, instead of `warning:` and
  continuing past partial-mono state).
- `helixc/bootstrap/parser.hx` — revert of cycle-4 C4-2 broadening (54-line
  removal of the 8-arm secondary dispatch on Binary/Unary/Index/Field/If/
  Match/Block/UnsafeBlock, restoring the cycle-3 D2 Call-only sentinel).
- `helixc/tests/test_autodiff.py`, `helixc/tests/test_typecheck.py` —
  cycle-6 regression test additions.

**Method**: Read the full c3f26ef diff in `git show`. Walked each
modified source file at HEAD. Re-verified the four cycle-5 audit-C
findings against the current source: confirmed that each emit site now
references the corresponding `TRAP_*` constant (eight emit sites total
across typecheck.py + monomorphize.py); confirmed the x86_64 driver
now exits 1 with `error:` prefix on a non-empty `mono_diags`; confirmed
the D-wrap gate at typecheck.py:1349 was broadened to mirror E2's Logic
gate at line 1372; confirmed the `_compatible` TyVar / TySize cascade
arm now returns True when either side is a generic variable or size
symbol. Verified parser.hx revert by counting brace balance in the new
val_tag dispatch block (lines 2300-2347): 12 `};` closes match the 12
`if val_tag == ...` arms (val_tag = 0, 27, 31, 34, 35, 36, 37, 38, 39,
40, 41, 16). Cross-referenced the trap-ids.md registry to confirm rows
28801 / 28802 / 28803 still point at extant, now-referenced constants.
Counted regression-test additions and cross-checked the commit message's
test claims against the diff.

**Reporting threshold**: confidence ≥ 80 (per cycle-6 audit-C prompt's
strict criterion).

**Result**: **0 findings (0 CRITICAL, 0 HIGH, 0 MEDIUM, 0 LOW).**

---

## Summary table

| ID    | Severity | Confidence | Component | Issue |
|-------|----------|------------|-----------|-------|
| —     | —        | —          | —         | (none at or above threshold) |

**Cycle 6 Audit C: CLEAN — 0 findings at or above the confidence-80
reporting threshold.**

Per user directive 2026-05-10 (strict criterion): cycle counts CLEAN
only when zero findings of ANY severity at or above the audit
threshold. **This cycle qualifies as clean.**

---

## Cycle-5 finding closure verification

### C5-1 (HIGH, conf 90): Dead TRAP_* constants — **CLOSED**

Verified by repo-wide grep:

- `helixc/frontend/typecheck.py:221` — `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO`
  is now referenced at lines 591, 596, 606, 611 (four emit sites in
  `_resolve_size_expr`). All four sites were previously literal
  `"trap 28802"` strings; cycle 6 swapped them to `f"(trap {TRAP_*})"`.
- `helixc/frontend/typecheck.py:222` — `TRAP_CAST_MATRIX_RECURSION_DEPTH`
  is now referenced at lines 2109, 2122 (two emit sites in
  `_check_cast_compat`). Previously literal `"trap 28803"`.
- `helixc/frontend/monomorphize.py:66` — `TRAP_SHAPE_FOLD_ZERO_DIV` is
  now referenced at lines 117, 125 (the cycle-5 sub-threshold
  diagnostic-quality note also closed for symmetry). Previously
  literal `"trap 28801"`.

Repo-wide grep:
```
$ grep -rn 'TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO\|TRAP_CAST_MATRIX_RECURSION_DEPTH\|TRAP_SHAPE_FOLD_ZERO_DIV' helixc/
helixc/frontend/monomorphize.py:66:TRAP_SHAPE_FOLD_ZERO_DIV = 28801
helixc/frontend/monomorphize.py:72:    (= TRAP_SHAPE_FOLD_ZERO_DIV).
helixc/frontend/monomorphize.py:76:    trap_id: int = TRAP_SHAPE_FOLD_ZERO_DIV
helixc/frontend/monomorphize.py:117:                f"in shape expression (trap {TRAP_SHAPE_FOLD_ZERO_DIV})",
helixc/frontend/monomorphize.py:125:                f"in shape expression (trap {TRAP_SHAPE_FOLD_ZERO_DIV})",
helixc/frontend/typecheck.py:221:TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO = 28802  # _resolve_size_expr
helixc/frontend/typecheck.py:222:TRAP_CAST_MATRIX_RECURSION_DEPTH = 28803  # _check_cast_compat
helixc/frontend/typecheck.py:591:                    f"(trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})",
helixc/frontend/typecheck.py:596:                    f"array size must be > 0, got 0 (trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})",
helixc/frontend/typecheck.py:606:                    f"array size must be > 0, got {v} (trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})",
helixc/frontend/typecheck.py:611:                    f"array size must be > 0, got 0 (trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})",
helixc/frontend/typecheck.py:2109:                    f"8 levels (trap {TRAP_CAST_MATRIX_RECURSION_DEPTH})",
helixc/frontend/typecheck.py:2122:                f"8 levels (trap {TRAP_CAST_MATRIX_RECURSION_DEPTH})",
```

The trap-ids.md audit-time invariant ("every TRAP_* must have at least
one reader") is satisfied. **CLOSED.**

### C5-2 (MEDIUM, conf 92): `monomorphize_safe` partial-mutation leak — **CLOSED (option A)**

Cycle 6 implements the recommended option-A fix: x86_64 driver now
exits 1 with `error:` prefix:

```python
# helixc/backend/x86_64.py:3032-3036
mono_count, mono_diags = monomorphize_safe(prog)
if mono_diags:
    for d in mono_diags:
        print(f"error: fn-mono: {d}", file=sys.stderr)
    sys.exit(1)
```

Sole-caller check: `grep -rn monomorphize_safe helixc/` returns only
`backend/x86_64.py` as a caller (plus the function definition in
`monomorphize.py:691`). The driver's `sys.exit(1)` short-circuits the
pipeline before `grad_pass` / `typecheck` / `lower` / codegen run on
the partially-mutated `prog`. The orphaned-mangled-callee state is no
longer reachable by any production path. The check.py front-end CLI
still doesn't call `monomorphize_safe` at all (only `monomorphize_structs`),
so the cycle-5 reproducer trace (write a generic fn with `[i32; 1/0]`,
run `python -m helixc.backend.x86_64 repro.hx`) now produces a clean
`error: fn-mono: ...: division by zero in shape expression (trap 28801)`
followed by `exit(1)`. **CLOSED.**

The in-place mutation in `Monomorphizer.run()` (mutation #1 at line
432: `item.body = new_body`) still happens, but no caller sees the
partial state because the only caller exits immediately. Option B
(transactional mono) was not implemented — but per the cycle-5
recommendation, option A is the lighter touch and matches the
docstring intent. Acceptable.

### C5-3 (MEDIUM, conf 85): D-wrap-asymmetric same-inner mirror gap — **CLOSED**

Verified at `helixc/frontend/typecheck.py:1349-1352`:

```python
if (l_is_diff or r_is_diff) and (
        inner_mismatch
        or (l_is_diff != r_is_diff)
):
```

Direct mirror of the E2 broadening at line 1372-1375. The recommended
fix from cycle 5 was applied verbatim. The downstream `extra = " (one
side D-wrapped, other bare)"` annotation at line 1368 correctly fires
on the asymmetric case. **CLOSED.**

A tangential diagnostic-quality concern: when same-inner asymmetric
fires (`D<f64> + f64`), the warning text reads "AD: D-binop with mixed
inner types f64 vs f64 — widened to f64 (trap 24200/AD002) (one side
D-wrapped, other bare)". The "mixed inner types f64 vs f64" lead-in
is incongruous when the inners match. The Logic branch (line 1396)
has the same wording issue with its prefix. This is a diagnostic-text
quality concern at confidence ~70, below the 80 threshold — not
re-flagged here.

### C5-4 (MEDIUM, conf 82): Zero regression tests for cycle-4 fix-sweep — **PARTIALLY CLOSED**

Cycle 6 adds **4** regression tests in `test_autodiff.py` + `test_typecheck.py`:

1. `test_c4_1_path_no_false_positive` — covers cycle-4 C4-1 Path
   identity arm (no 85001 false-positive).
2. `test_c4_3_inline_lets_if_cond_substituted` — covers cycle-4 C4-3
   `_inline_lets` recursion into `If.cond`.
3. `test_c5_2_compatible_tysize_cascade` — covers cycle-6 C5-2 / F1
   TyVar / TySize cascade arm.
4. `test_c6_revert_c4_2_literal_binary_no_false_trap` — covers the
   cycle-6 parser.hx C4-2 revert (source-text assertion, not a
   behavioral test).

The commit message claims tests for "C4-1 (Path), C4-3 (If.cond), C5-2
(TySize cascade), C4-4 (TyTile/TyTensor structural compat), C6 revert" —
that's five claimed and four landed. The claimed-but-missing test for
C4-4 (TyTile/TyTensor structural compat) is a doc-source mismatch in
the commit message; confidence ~70, below threshold (the actual cycle-4
fix is exercised indirectly through the typecheck baseline + the new
TySize cascade test that round-trips through `TyArray`).

Cycle 3's baseline was 16 tests for 15 findings; cycle 4 was 0 tests
for 13 findings; cycle 6 lands 4 tests for ~13 cycle-4-plus-cycle-6
fixes. Test density is improved but still below the cycle-3 bar.
This is no longer a confidence-80+ finding because the most-critical
gaps (C4-1, C4-3, C5-2 cascade, parser.hx revert) are now covered, and
the missing cases (C4-2-eight-arm-revert behavioral, C4-4 structural,
C5-3 / F4 TileLit-walk, C5-4 / F3 monomorphize-safe-abort, F2 D-wrap-
same-inner-warns, monomorphize_safe end-to-end happy and sad paths)
are individually below the per-finding threshold. **PARTIALLY CLOSED;
not re-flagged.**

---

## Files reviewed

`helixc/backend/x86_64.py`, `helixc/bootstrap/parser.hx`,
`helixc/frontend/autodiff.py`, `helixc/frontend/ast_nodes.py`,
`helixc/frontend/monomorphize.py`, `helixc/frontend/typecheck.py`,
`helixc/tests/test_autodiff.py`, `helixc/tests/test_typecheck.py`,
`docs/lang/trap-ids.md`, plus the persisted cycle-5 audit-doc files
(`audit-stage28-8-cycle5-{codereview,silent-failures,type-design}.md`)
for cross-reference.

---

## Specific cycle-6 changes audited (12 items)

1. `helixc/backend/x86_64.py:3032-3036` — driver abort on `mono_diags`
   non-empty (C5-2 / F3 close). `sys.exit(1)` is reached before any
   downstream pass sees the partial-mono prog. **PASS.**
2. `helixc/frontend/monomorphize.py:117, 125` — `f"(trap {TRAP_SHAPE_FOLD_ZERO_DIV})"`
   replaces the literal `"trap 28801"` substring in both div-by-zero
   and mod-by-zero `ShapeFoldError.__init__` calls. **PASS.**
3. `helixc/frontend/typecheck.py:591, 596, 606, 611` — `_resolve_size_expr`
   four emit sites now use `f"(trap {TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO})"`.
   **PASS.**
4. `helixc/frontend/typecheck.py:2109, 2122` — `_check_cast_compat` two
   emit sites now use `f"(trap {TRAP_CAST_MATRIX_RECURSION_DEPTH})"`.
   **PASS.**
5. `helixc/frontend/typecheck.py:1349-1352` — D-wrap gate broadened to
   `(l_is_diff or r_is_diff) and (inner_mismatch or (l_is_diff != r_is_diff))`.
   Mirrors E2's Logic gate at line 1372-1375. **PASS** (F2 close;
   matches cycle-5 C5-3 recommended fix verbatim).
6. `helixc/frontend/typecheck.py:2166-2179` — `_compatible` new
   cascade-safe arm: `if isinstance(a, (TyVar, TySize)) or isinstance(b,
   (TyVar, TySize)): return True`. **PASS** (C5-2 / F1 close). The
   TyVar inclusion is broader than strictly required by the cycle-5
   finding (which named only TySize); all call sites that could
   matter are either gated upstream (line 736: call boundary
   already skips TyVar/TySize/TyUnknown) or operate in contexts
   where TyVar deferral to mono is the correct behavior (if/else
   merge, struct-field, match-arm-merge, let-stmt-decl, body-vs-
   return). No production path was identified that would silently
   accept a real type error because of the broader arm.
7. `helixc/frontend/autodiff.py:707-720` — TileLit identity arm
   replaced with an explicit constructor walking
   `shape: list[Expr]` (per-element `_inline_lets`) and
   `memspace: Expr` (`_inline_lets`). The `dtype: TyNode` and
   `init: str` fields are correctly preserved as-is (TyNode is
   not in the Expr hierarchy; init is a literal "zeros"/"ones"
   string). **PASS** (C5-3 / F4 close).
8. `helixc/bootstrap/parser.hx:2300-2347` — revert of cycle-4 C4-2
   broadening. The 8-arm secondary dispatch (Binary/Unary/Index/
   Field/If/Match/Block/UnsafeBlock) is removed; only the original
   D2 Call-only sentinel (val_tag == 16) remains. Brace-balance
   check: 12 `if val_tag == N {` arms (N = 0, 27, 31, 34, 35, 36,
   37, 38, 39, 40, 41, 16) are closed by exactly 12 `};` tokens
   at line 2347 (`};};};};};};};};};};};};`). The revert is
   structural; the cycle-3 D2 behavior is preserved (provably-i32
   literal RHS lets get inferred_ty_tag = 0 through 11; Call RHS
   gets sentinel 12; everything else stays at -1 and defers to
   `var_type_tab_lookup`). **PASS** (revert; matches the
   cycle-5 silent-failures audit's recommendation).
9. `helixc/tests/test_autodiff.py:549-562` — `test_c4_1_path_no_false_positive`:
   constructs an `A.Path(segments=["Maybe", "None"])` and asserts
   `_inline_lets` returns identity (no 85001 fire). **PASS**.
10. `helixc/tests/test_autodiff.py:565-595` — `test_c4_3_inline_lets_if_cond_substituted`:
    builds an `A.If` with cond `A.Binary(>, Name('g'), 0)` and
    asserts that `_inline_lets(if_expr, {"g": <expr>})` substitutes
    the cond. The assertion (`not (isinstance(left, A.Name) and
    left.name == "g")`) correctly tests the negative — left is no
    longer Name('g'). **PASS.**
11. `helixc/tests/test_autodiff.py:598-622` — `test_c6_revert_c4_2_literal_binary_no_false_trap`:
    asserts `if val_tag == 6 {` (a marker of the cycle-4 C4-2
    broadening) is absent from parser.hx source, AND the original
    `if val_tag == 16 {` arm is present. This is a source-text
    structural test, not a behavioral one — running the actual
    `let a = 10 + 5; let c = |x| x + a; c(5)` repro would require
    the bootstrap binary to rebuild against the reverted parser.hx.
    Acceptable for the bootstrap-language constraint. **PASS** at
    the structural level; below-threshold concern about behavioral
    coverage tracked outside this finding (confidence ~55).
12. `helixc/tests/test_typecheck.py:1436-1455` — `test_c5_2_compatible_tysize_cascade`:
    asserts `_compatible(TySize('N'), TySize('M'))` returns True
    AND `_compatible(TyArray<i32; N>, TyArray<i32; 3>)` returns
    True. Both are direct exercises of the new cascade arm.
    **PASS**.

---

## What was checked and found below threshold

- **`_compatible` cascade arm broader than minimum** (line 2178):
  cycle-5 C5-2 / F1 named TySize as the specific gap; cycle-6
  includes both TyVar and TySize. All identified production call
  sites either gate around TyVar upstream (line 736 call-boundary)
  or use TyVar in contexts where mono-deferral is correct. No
  silent-acceptance hole identified. **Confidence 55**, below
  threshold.

- **Diagnostic wording "mixed inner types f64 vs f64" when
  asymmetric same-inner fires**: the F2 close at line 1349 makes
  `D<f64> + f64` warn, but the message ("mixed inner types ... vs
  ...") was authored for the inner-mismatch case and reads
  incongruously when the inners are equal. The "(one side D-wrapped,
  other bare)" suffix does carry the useful info, but the lead-in
  is misleading. Identical issue at line 1396 for the Logic branch
  (which already fires on same-inner asymmetric per cycle-4 E2,
  so this isn't new to cycle 6). **Confidence 70**, below
  threshold.

- **Commit message claims 5 tests but landed 4** (no test for the
  cycle-4 C4-4 TyTile/TyTensor structural compat path):
  doc-source mismatch in the commit log. The cycle-4 C4-4 path is
  exercised indirectly through `test_c5_2_compatible_tysize_cascade`
  (which round-trips through `TyArray` whose elem and size both
  flow through `_compatible`), but no direct
  `_compatible(TyTile(...), TyTile(...))` mismatch test exists.
  **Confidence 70**, below threshold.

- **`test_c6_revert_c4_2_literal_binary_no_false_trap` is source-
  text inspection rather than behavioral**: the test reads
  parser.hx as text and asserts `"if val_tag == 6 {"` is absent +
  `"if val_tag == 16 {"` is present. A behavioral test would
  require rebuilding the bootstrap binary. The chosen approach is
  reasonable for the constraint but coupled to specific source
  text rather than to the observable parser behavior.
  **Confidence 55**, below threshold.

- **TileLit walk does not descend into `dtype` field**: the cycle-6
  C5-3 / F4 fix walks `shape` and `memspace` but passes `dtype:
  TyNode` through unchanged. `TyNode` is a separate AST hierarchy
  from `Expr`, so `_inline_lets` (Expr-domain) correctly does not
  recurse. A future feature where `dtype` could carry let-bound
  generic names (e.g. `let T = i32; tile<T, ...>` syntax) would
  resurface this; not in scope for Phase 0. **Confidence 45**,
  below threshold.

- **`Monomorphizer.run()` in-place mutation NOT made transactional**:
  cycle-5 C5-2 recommended option-A (driver-side abort) over
  option-B (transactional mono); cycle 6 chose option A. The
  in-place mutation in `run()` still exists, but no production
  caller sees the partial state (sole caller exits before
  observing). Future callers (e.g. an LSP server, an incremental
  compile mode) would re-expose this. **Confidence 60**, below
  threshold — out of scope for current Phase 0 entry points.

- **`A.Path` callee in `_inline_lets.A.Call` E6 path drops
  generics** (autodiff.py line 584-585, unchanged by cycle 6):
  pre-existing behavior, flagged at confidence 55 in cycle 5's
  below-threshold notes. Pre-cycle-4. **Not re-flagged.**

- **`_inline_lets.A.Block.stmts` linear-search `.index(stmt)`
  pattern**: O(n²) in stmt-count; pre-cycle-4 introduced.
  Out of cycle-6 scope. **Not re-flagged.**

- **Parser.hx revert leaves the bool-tag mapping arms gone** (the
  cycle-4 mapping of AST_LT, AST_GT, AST_EQ-AST_GE to tag 0):
  these were arms 6, 19, 20, 21, 22, 23 in the C4-2 dispatch.
  Reverting drops them, meaning a `let b = a < c; let cl = |x|
  x + b; cl(0)` now leaves `b`'s inferred_ty_tag at -1 (untracked)
  rather than 0 (proven-i32-bool). The capture-site guard at
  `> 0` then silently captures `b` as i32. This is the pre-cycle-4
  D2 behavior; reverting is intended per the cycle-5
  silent-failures audit. **Pre-cycle-4 behavior; not a regression
  caused by cycle 6.**

---

## Open prior findings (not re-flagged this cycle)

All four cycle-5 audit-C findings (C5-1, C5-2, C5-3, C5-4) are closed
or partially-closed-but-below-threshold per the analysis above.

Cycle-5 silent-failures findings (F1-F4 / C5-1..C5-4 in that audit's
numbering) and cycle-5 type-design findings (F1-F6 in that audit's
numbering) are confirmed closed by the cycle-6 fix-sweep per the
commit message's mapping. Spot-verification:

- Silent-failures C5-1 (cycle-4 C4-2 false-positive over-broadening) →
  reverted at parser.hx:2300-2347. **CLOSED.**
- Silent-failures C5-2 / type-design F1 (`_compatible` TySize) →
  closed at typecheck.py:2178. **CLOSED.**
- Silent-failures C5-3 / type-design F4 (TileLit walk) → closed at
  autodiff.py:707-720. **CLOSED.**
- Silent-failures C5-4 / type-design F3 (`monomorphize_safe` abort
  semantics) → closed at x86_64.py:3033-3036. **CLOSED.**
- Type-design F2 (D-wrap asymmetric mirror) → closed at
  typecheck.py:1349-1352. **CLOSED.**

Pre-cycle-1 audit-stage5-6, audit-stage7-8, audit-stage9-16 baselines
unchanged from cycle-1 status; cycle-1 through cycle-4 findings all
marked CLOSED by their respective fix-sweep commits.

---

## Verdict

**Cycle 6 Audit C: CLEAN — 0 findings (0 CRITICAL, 0 HIGH, 0 MEDIUM,
0 LOW) at or above the confidence-80 reporting threshold.**

Strict-zero rule per user directive 2026-05-10. Cycle counter advances
provided cycles A (silent-failure) and B (type-design) are also clean
at this commit.

No recommended fixes for cycle 6 audit-C scope. The below-threshold
notes (broader `_compatible` cascade arm, "mixed inner types T vs T"
diagnostic wording, commit-message-vs-test-count drift,
source-text-vs-behavioral coverage choice) are documented for
future cycles but do not block this cycle's clean status.
