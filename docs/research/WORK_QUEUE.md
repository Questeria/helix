# Helix Foreground Work Queue

> Historical snapshot: this file records a 2026-05-04 foreground queue. It is
> not current Stage 35 gate evidence and should not be used as the live work
> selector. Use `docs/ROADMAP.md` and `docs/stage35-progress-2026-05-15.md`
> for current status.

Generated 2026-05-04 by long-horizon planner. Each ticket was sized for a
single 5-15 min foreground tick in that historical queue.

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
- **File:** `helixc/frontend/autodiff.py` and `autodiff_reverse.py` — at the time of this historical ticket, neither handled `A.Match`. Forward mode: differentiate the chosen arm; reverse mode: route the cotangent into the arm body.
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
5. At the time of this snapshot, workers were expected to run the full suite
   after every 5 done tickets and update the then-current test count. Do not
   use those snapshot counts as current Stage 35 evidence.

## Backlog seeds (out-of-tick, do not pick yet)

- HBS spec freeze (Tier 2 in WAVE1 — needs design doc first)
- E-graph layer (Tier 2 — needs hash-consing fully landed first)
- Refinement-reflected verifiers (Tier 2 — needs SMT bridge)
- AST-as-Helix-value (Tier 2 — needs HBS)
- Algebraic effect handlers (Tier 3)
- Provenance-typed `D<S, T>` (Tier 3)

When Tier A through Tier D above are mostly green, regenerate this file from the next research wave.

---

## Tier F — Wave 2 work (post-2026-05-04)

Generated 2026-05-04 evening after the enum-payload + struct-flatten + tuple-field epic landed (commits `a39a9aa`..`3a83a38`). This historical wave focused on (1) **closing the self-host gap** — payload pattern-extraction and struct-by-value were the two features then blocking a real `helixc-bootstrap` in HBS — (2) **paying down latent bugs** surfaced when struct/enum codegen landed, and (3) **tightening UX** so dogfood programs stopped hitting "unsupported" papercuts. Its old green-test growth target was a 2026-05-04 projection, not a current test count.

### 21. Payload pattern extraction in match arms (M) [done f5fa4a7]
At the time of this historical ticket, `Maybe::Some(x) => body` parsed but the parser **discarded the payload binders** (parser.py:1040-1060: "Skip the payload args for now") and lowered the whole arm as a tag-only `PatLit`. Users had to write `match m[0] { 1 => m[1], _ => 0 }`, defeating the abstraction. The planned fix touched three places:
- **Files:** `helixc/frontend/ast_nodes.py` add `class PatEnum(Pattern): enum_name: str, variant: str, binders: list[Pattern]`. `helixc/frontend/parser.py::_parse_pattern_atom` (~line 1040) — replace the "skip payload" branch with a real `_parse_pattern()` loop that collects sub-patterns. `helixc/frontend/match_lower.py::_pattern_test` add a `PatEnum` arm: emit `__scrut[0] == variant_idx` test and per-binder `let bi = __scrut[i+1]` lets injected before the body.
- **Test:** `helixc/tests/test_match.py::test_match_extracts_enum_payload` — `enum Maybe { None, Some(i32) } fn main() -> i32 { let m = Maybe::Some(42); match m { Maybe::Some(x) => x, Maybe::None => 0 } }` should exit 42, not require manual `m[1]`.

### 22. Struct pass-by-value flattens slot range into call args (M)
`test_struct_passed_to_helper` is currently a `try/except` that accepts both 42 and 0 because `lower_ast.py::_lower_expr(A.Call)` passes the struct's first slot only (line 559: `v = self._lower_expr(a)` returns one tir.Value). Real fix: at call sites, look up `_struct_flat_paths[struct_name]` for any arg whose typecheck'd type is `TyStruct`, emit one `LOAD_ELEM` per flat slot, and append all slot values to the args list. Callee's parameter binding in `_lower_fn` must also expand the struct param into N consecutive slots.
- **Files:** `helixc/ir/lower_ast.py` — patch `Call` arg-lowering loop (~line 557-561) and `_lower_fn` parameter setup (around line 220 `self._bind(name, ...)` for params). May need a new helper `_lower_struct_arg(name, slit_or_arr) -> list[tir.Value]`.
- **Test:** `helixc/tests/test_codegen.py::test_struct_passed_to_helper_returns_value` — copy the existing test but assert `code == 42` strictly (drop the `try`/0-fallback).

### 23. Make `(1, 2, 3).1` work without an intermediate `let` (S)
`test_tuple_field_access_e2e` only works because the tuple is bound to `t` first. A literal-tuple-with-immediate-field `(1, 2, 3).1` falls through `_lower_expr(A.Field)` because there's no base Name — `_walk_field_chain` returns `base_name=None`. Allocate an anonymous array for inline TupleLit (mirror the `let stmt` special case) and load_elem from it.
- **File:** `helixc/ir/lower_ast.py::_lower_expr(A.Field)` (~line 826-864) — when `expr.obj` is a TupleLit and `expr.name.isdigit()`, lower elements, allocate a temp array, store, load_elem with the indexed slot. Same for direct `(1,2,3).0` via temp-binding-on-the-fly.
- **Test:** `helixc/tests/test_codegen.py::test_inline_tuple_field_access` — `fn main() -> i32 { (10, 32, 0).0 + (10, 32, 0).1 }` should exit 42.

### 24. `print_int(i32)` builtin for diagnostic output (S)
At the time of this historical ticket, `print_str` only emitted a literal and runtime i32 printing was missing from Helix code. The planned fix added `print_int(n: i32)` that emits a `PRINT` op tagged `_kind="print_int"`, with backend decimal formatting via repeated divide-by-10 into a 12-byte buffer + write(1, buf, len).
- **Files:** `helixc/frontend/typecheck.py` — add `"print_int"` to `_BUILTIN_NAMES` plus a Call handler returning `i32`. `helixc/ir/lower_ast.py::_lower_expr(A.Call)` add an intercept similar to `print_str`. `helixc/backend/x86_64.py` — emit a small int→ASCII routine inline (or a single emitted helper labelled `__print_int`).
- **Test:** `helixc/tests/test_strings_io.py::test_print_int_decimal_output` — capture stdout from `fn main() -> i32 { print_int(2026); 0 }`, assert it contains `b"2026"`.

### 25. Parser progress guard for `_parse_pattern_atom` payload-skip block (S)
The current "skip payload" branch (parser.py:1052-1060) advances `self.i` based on paren depth but does not check for unmatched `{`/`[` or recursive calls into other parsers. Audit-pass-3 fixed a similar issue elsewhere; harden this one too by replacing the manual depth-counter with a real `_parse_pattern()` recursion (becomes a no-op once #21 lands; do this first as a defensive cleanup).
- **File:** `helixc/frontend/parser.py` lines 1049-1060.
- **Test:** `helixc/tests/test_parser.py::test_pattern_payload_handles_nested_parens` — `Foo::Bar((1, 2), 3)` should parse to a PatEnum (after #21) or skip cleanly (before #21) without dropping or eating the `=>`.

### 26. Static int-literal overflow check (S)
At the time of this historical ticket, `let x: i32 = 5_000_000_000;` silently truncated because `IntLit.value` was a Python int and lowering did `const_int(value & 0xFFFFFFFF)` (or similar). The planned fix added a typecheck-time bounds check: for any `IntLit` whose contextual type is fixed-width (`i8/i16/i32/i64/u8/...`), assert the value fits in that width's signed/unsigned range, else error with did-you-mean ("use `i64`?").
- **File:** `helixc/frontend/typecheck.py` — add `_check_int_lit_fits(lit, ty)`. Call it from `Let` (typed), `FnDecl` parameter defaults if any, and `Cast`.
- **Test:** `helixc/tests/test_typecheck.py::test_int_literal_overflow_errors` — `let x: i32 = 5000000000;` errors with "value 5000000000 does not fit in i32".

### 27. Const-fold for division/modulo by literal one (S)
At the time of this historical ticket, `x / 1 = x` and `x % 1 = 0` were not folded. The const-fold pass already handled `x*1`, `x+0`, etc. (Tier C #15). The planned fix mirrored those rules.
- **File:** find the const-fold table (grep `"x \\* 1"` in `helixc/ir/`); add `(DIV, x, 1) -> x` and `(MOD, x, 1) -> 0`.
- **Test:** `helixc/tests/test_const_fold.py::test_x_div_one_folds`, `::test_x_mod_one_folds_to_zero`.

### 28. `helixc check` shows source-with-caret on errors (S)
`TypeError_.format_with_source(src, filename)` exists at typecheck.py:160 and renders the Rust-style `^` caret display, but `helixc/check.py:64` only does `print(f"     {e}")` — bare `__str__`. Wire `format_with_source` through.
- **File:** `helixc/check.py` lines 60-68.
- **Test:** `helixc/tests/test_codegen.py::test_check_cli_error_has_caret_display` — invoke check.py on a known-bad file, assert stdout contains `^` and the offending source line.

### 29. `--emit-ir` flag on `helixc check` for IR inspection (S)
Long-horizon: when self-hosting begins, the bootstrap compiler will need to dump IR for parity comparison against the Python pipeline. A `--emit-ir <file.hx>` flag that runs lower_ast + the canonical pass pipeline and prints `tir.OpKind.NAME args -> result` for every op gives us the parity oracle. Reuse the format from existing `tir.Op.__repr__` if any.
- **File:** `helixc/check.py` — add `--emit-ir` flag handling. Lower with `lower_ast.lower_program(prog)`, then iterate functions, then ops.
- **Test:** `helixc/tests/test_codegen.py::test_check_cli_emit_ir_flag` — invoke on `fn main() -> i32 { 1 + 2 }`, assert stdout contains `ADD` and `RET`.

### 30. `enum_variants[name]` exhaustive check in match (M)
Tier A #2 implemented exhaustiveness for `bool` and `unit`. Now the lowerer indexes enum variants (`_enum_variants` in lower_ast.py:51) — surface that to typecheck so `match op { Op::Add => ..., Op::Sub => ... }` errors when `Op` has 3 variants. Hook into the existing `_check_match_exhaustive`. After #21 lands, also recurse into PatEnum sub-patterns; before #21, only a flat enumeration of `EnumName::VariantName` PatLit-of-Path tags is required.
- **File:** `helixc/frontend/typecheck.py` — extend `_check_match_exhaustive` with a TyEnum-or-by-name case; need a small `_enum_decls` map populated in pass 0 alongside `_struct_decls`. Diagnose missing variants by name.
- **Test:** `helixc/tests/test_match.py::test_non_exhaustive_enum_errors` — `enum Op { Add, Sub, Mul } fn f(o: Op) -> i32 { match o { Op::Add => 0, Op::Sub => 1 } }` errors with "missing variant: `Op::Mul`".

---

### Foreground priority for Tier F

Prioritize 21 → 22 → 23 (the three self-host blockers) before 24-30. After #21+#22+#23 land, real pass-by-struct compiler IR can be expressed in HBS, which unlocks the `kovc-bootstrap` direction in stage4-m2 / kovc/. Tickets 25-27 are quick latent-bug paydowns; 24/28/29 are dev-velocity wins; 30 is a polish item that should land last because it'll grow once #21 reshapes the pattern AST.
