# Helix Foreground Work Queue

Generated 2026-05-04 by long-horizon planner. Each ticket is sized for a single 5–15 min foreground tick. Pop the **top** unchecked item; mark it done and append a one-line note when finished. Re-prioritize only if a ticket is blocked.

Sizing: **S** ≤ 50 lines / 30 min, **M** ≤ 200 lines / 2 h, **L** ≤ 600 lines / 1 day. Anything **L** is split into S/M sub-tickets.

Project paths (Windows): all under `C:\Projects\Kovostov-Native\`.

---

## Tier A — Pattern matching with guards (Tier 1, broken into 7 sub-tickets)

Parser + AST already support `match`, `if`-guards, `PatLit | PatBind | PatWildcard | PatTuple`. What's missing: pattern-binder scoping in typecheck, exhaustiveness, codegen, enum variant patterns, range patterns, or-patterns, tests.

### 1. Bind pattern variables into match-arm body scope (S) [done 28f252a]
- **File:** `helixc/frontend/typecheck.py` lines ~824-829 (the `isinstance(expr, A.Match)` arm).
- **What:** Walk each arm's pattern; for every `PatBind`/`PatTuple` element, push a new `Scope(parent=scope)` and `define()` the binder with the inferred type of the scrutinee (or tuple element). Type-check `arm.guard` (if present, must be `bool`) and `arm.body` in that inner scope. Unify all arm-body types and return the lub.
- **Test:** add `helixc/tests/test_match.py::test_match_binds_pattern_var` — `match x { y => y + 1, _ => 0 }` should typecheck and the `y` reference must resolve.

### 2. Exhaustiveness checker — boolean and unit (S) [done b82e215]
- **File:** `helixc/frontend/typecheck.py` (new helper `_check_match_exhaustive`).
- **What:** For scrutinees of type `bool`, require `{true, false}` or a wildcard. For `unit`, require `()` or wildcard. Emit a typecheck error with did-you-mean for missing arm.
- **Test:** `test_match.py::test_non_exhaustive_bool_errors`.

### 3. Guard typecheck must be bool, body types must unify (S) [done b82e215]
- **File:** `helixc/frontend/typecheck.py` (same arm).
- **What:** Assert `_check_expr(arm.guard, inner_scope)` returns `TyPrim("bool")`. Assert all arm-body types are equal (or both numeric → numeric lub). Add error path with citation to first-mismatched arm.
- **Test:** `test_match.py::test_guard_must_be_bool`, `::test_arm_body_type_mismatch_errors`.

### 4. Or-patterns `a | b | c` in parser + AST (S) [done b82e215]
- **File:** `helixc/frontend/ast_nodes.py` add `class PatOr(Pattern): alts: list[Pattern]`. `helixc/frontend/parser.py::_parse_pattern` lines 952-975 — after parsing one pattern, while `_match(T.PIPE)` loop and accumulate.
- **Test:** `helixc/tests/test_parser.py::test_or_pattern_parses`.

### 5. Range patterns `0..10` and `0..=10` (S) [done 4d16b0d]
- **File:** `ast_nodes.py` add `class PatRange(Pattern): lo, hi, inclusive`. Parser lines 952-975, integrate into `_parse_pattern`.
- **Test:** `test_parser.py::test_range_pattern_parses`.

### 6. Codegen Match → chained `if`-let lowering (M) [done 07c4dd2]
- **File:** `helixc/frontend/grad_pass.py` or a new `helixc/frontend/match_lower.py` — desugar `Match` into nested `If` + temporary `Let` bindings before IR. Hook into the existing IR pipeline so x86 backend sees no Match.
- **Test:** `helixc/tests/test_match.py::test_match_int_literal_runs` — emit, run, assert exit code matches selected arm.

### 7. Match in autodiff: passthrough scrutinee, propagate gradients to selected arm (M) [done ea75ff6]
- **File:** `helixc/frontend/autodiff.py` and `autodiff_reverse.py` — currently neither handles `A.Match`. Forward mode: differentiate the chosen arm; reverse mode: route the cotangent into the arm body.
- **Test:** `helixc/tests/test_autodiff.py::test_grad_through_match` — `f(x) = match cond { true => 2*x, false => 3*x }` should yield `2` or `3` depending on cond.

---

## Tier B — Hash-consing follow-through (already started in `ast_hash.py`)

### 8. Memoize differentiate() by body hash (S) [done d25d969]
- **File:** `helixc/frontend/autodiff.py` — add module-level `_DIFF_CACHE: dict[str, Expr] = {}` keyed on `structural_hash(body)`. On reentry, return cached deriv. Invalidate on per-call binder shadowing only if hash includes free names (it does — see `FreeName` branch).
- **Test:** `helixc/tests/test_autodiff.py::test_diff_memo_hits` — instrument cache and verify two structurally-equal calls share an entry.

### 9. Replace Quote handle (currently Python-`hash() mod 64`) with content-addressed slot allocation (M) [done ff97c45]
- **File:** `helixc/frontend/grad_pass.py` (or wherever Quote handles get assigned — grep `% 64`). Use `structural_hash()[:n]` mod 64 with collision-aware fallback.
- **Test:** `helixc/tests/test_reflection.py::test_quote_handles_stable_across_runs`.

### 10. Add `ast_hash` round-trip test for every node kind (S) [done 9b79c76]
- **File:** `helixc/tests/test_ast_hash.py` — extend with one test per remaining AST type (Match, MatchArm, Range, Quote, Splice, Modify) verifying alpha-equivalence collapses and structural difference separates.

### 11. Hash-keyed CSE (M)
- **File:** `helixc/frontend/cse.py` (or wherever CSE lives in IR — grep). Replace whatever expression-equality check is in use today with `structural_hash()`. Pure refactor; expect 0 behavior change but test count constant.
- **Test:** `helixc/tests/test_cse.py::test_cse_uses_structural_hash` (reuse existing test corpus, just confirm parity).

---

## Tier C — Stdlib + AD gaps surfaced by audit cadence

### 12. Add `__pow(x, n)` integer exponent builtin + AD chain rule (S) [done 70e0d9f]
- **File:** `helixc/stdlib/` (find existing `__exp` / `__sin` and mirror), `autodiff.py` and `autodiff_reverse.py` rule tables.
- **Test:** `helixc/tests/test_autodiff_parity.py::test_pow_int_parity` — forward vs reverse on `x^3, x^4` at 4 inputs.

### 13. Add `__atan2(y, x)` with two-arg AD rule (S)
- **Files:** stdlib + autodiff rule tables (Binary case).
- **Test:** `test_autodiff_parity.py::test_atan2_parity`.

### 14. Add `__abs` with AD subgradient at 0 documented (S) [done 128fb31]
- **Files:** stdlib + autodiff. At 0 return 0 for the subgradient; document choice in `docs/lang/agi-features.md`.
- **Test:** `test_autodiff.py::test_abs_subgrad_at_zero_is_zero`.

### 15. Const-fold pass: extend algebraic identities to `x*1, 1*x, x/1, x+0, 0+x` (S) [done 0249bce]
- **File:** `helixc/frontend/grad_pass.py` or wherever the const-fold table lives — grep `x\*0`.
- **Test:** `helixc/tests/test_const_fold.py::test_x_times_one_folds`.

### 16. Did-you-mean for misspelled stdlib calls (S) [done b9f92b9]
- **File:** `helixc/frontend/typecheck.py` — extend the existing did-you-mean (used for unbound names) to also fire when an `A.Call` callee resolves to a name not in the function table; suggest from `BUILTINS` list.
- **Test:** `test_typecheck.py::test_call_did_you_mean_suggests_builtin`.

---

## Tier D — Total-by-default groundwork (Tier 1 in roadmap)

### 17. Parse `@partial` and `@total` attribute on `fn` decls (S) [done 128fb31]
- **File:** `helixc/frontend/parser.py` — extend the existing `attrs` recognizer (it already accepts `@pure`, `@effect(...)`).
- **Test:** `test_parser.py::test_partial_attribute_parses`.

### 18. Static structural-recursion checker stub (M) [done 128fb31]
- **File:** new `helixc/frontend/totality.py` — for fns without `@partial`, walk body; flag any recursion that does not strictly decrease a syntactic measure on at least one arg. Conservative — accept obvious tail cases, reject unknown.
- **Test:** new `helixc/tests/test_totality.py::test_factorial_accepted, ::test_collatz_rejected_without_partial`.

---

## Tier E — Ergonomics / dev velocity

### 19. Better Match-related error spans (S) [done fad8e03]
- **File:** `helixc/frontend/parser.py` lines 931-950 — currently the arm span uses `arm_start` token; ensure `_parse_pattern` returns a span covering the full pattern including `|`/`..` extensions.
- **Test:** `test_parser.py::test_match_arm_span_covers_pattern`.

### 20. `--dump-ast-hashes` CLI flag (S) [done fad8e03]
- **File:** `helixc/frontend/autodiff_cli.py` (or whichever CLI entry helixc uses; grep `argparse`). Add a flag that prints `name : 12-char-hash` for every top-level fn.
- **Test:** `helixc/tests/test_codegen.py::test_dump_ast_hashes_flag` — invoke CLI, assert stable hash for a fixed source file across two runs.

---

## Foreground protocol

1. Read this file. Pick the topmost unchecked ticket.
2. If sized **S**: implement + write the listed test + run `python -m pytest helixc/tests/ -q`. If green, commit (one ticket = one commit). Append `[done <commit-hash>]` to the bullet.
3. If sized **M**: implement against the listed test only; defer broader regression to a follow-up ticket; commit. Append `[done <commit-hash>]`.
4. If a ticket is blocked (missing infrastructure, ambiguous spec), append `[blocked: <reason>]` and skip to the next.
5. After every 5 done tickets, run the full suite once and update `README.md` test count.

## Backlog seeds (out-of-tick, do not pick yet)

- HBS spec freeze (Tier 2 in WAVE1 — needs design doc first)
- E-graph layer (Tier 2 — needs hash-consing fully landed first)
- Refinement-reflected verifiers (Tier 2 — needs SMT bridge)
- AST-as-Helix-value (Tier 2 — needs HBS)
- Algebraic effect handlers (Tier 3)
- Provenance-typed `D<S, T>` (Tier 3)

When Tier A through Tier D above are mostly green, regenerate this file from the next research wave.
