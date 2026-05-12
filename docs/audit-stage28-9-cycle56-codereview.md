# Audit Stage 28.9 cycle 56 — Code review

**Scope.** Read-only HEAD `5d58d3d` (cycle-55 fix-sweep: `_fn_table_sig` body-hash extension + NIE guard). Prior C1–C54 not re-flagged.
**Criterion.** 0 findings at conf >=75%.

## Result: PASS (0 findings)

| Severity | Count |
|---|---|
| CRITICAL (>=90) | 0 |
| HIGH (80-89)    | 0 |
| MEDIUM (75-79)  | 0 |
| **Total >=75**  | **0** |

The cycle-55 delta (3 fixes to `helixc/frontend/autodiff.py::_fn_table_sig` + outer `differentiate()` cache-key guard + 3 regression tests in `helixc/tests/test_autodiff.py`) is correctly implemented and internally consistent. Spot-scan of recently-touched frontend files (`autodiff.py`, `hash_cons.py`, `ast_hash.py`, `monomorphize.py`, `match_lower.py`) surfaced no >=75 issues outside the C1–C54 prior set.

## Verification highlights

1. **`_fn_table_sig` new key format `{name}/{arity}/{sorted_attrs}/{body_hash}` joined by `|`.** Separator chars `/` and `|` cannot appear in Helix identifiers (lexer: `[A-Za-z_][A-Za-z0-9_]*`), nor in any attr string produced by `parser._parse_attributes` (which emits `<ident>`, `<ident>:<arg>`, `autotune:KEY=v1,v2`, `effect:<arg>` — none containing `/` or `|`). Collision risk negligible.
2. **`sorted(fn.attrs)`** — `fn.attrs: list[str]` per `ast_nodes.py:452`; Python `sorted` on strings is total + stable. Deterministic.
3. **NIE catch on line 146** (`structural_hash` body-hash) — verified `structural_hash(SyntheticUnknown())` raises `NotImplementedError` from `ast_hash._hash_into`; without the new arm it would propagate; with the new arm the `<unhashable:id>` sentinel is emitted.
4. **NIE catch on line 184** (outer `differentiate` key tuple) — same rationale, now covers `structural_hash(expr)` raising NIE for a novel top-level `expr` subclass.
5. **`test_c54_ad3_cache_layer_catches_not_implemented_error`** — exercises the `_fn_table_sig` site (line 146). Test passes (`pytest -k c54_ad3 -x` → 1 passed). Test does NOT exercise the outer `differentiate()` site directly, but the outer catch is structurally identical (same except-tuple, same sentinel-style fallback to `key=None`) and the cycle-55 commit message does not claim a second regression test — coverage gap is acknowledged, not silent.
6. **Comment accuracy** — the cycle-55 header comment in `_fn_table_sig` lists C54-AD1/AD2/AD3 with accurate descriptions of `_inline_user_calls`'s three dimensions (attrs / arity / body). The "line ~365" cross-reference uses `~` (approximate); actual reads at 361/362 (`"pure" not in fn.attrs`) and 364 (`len(fn.params) != len(new_args)`). Approximation is honest.
7. **Hash_cons.py / ast_hash.py / monomorphize.py / match_lower.py** — fix-ID comments scanned for the ~10-cycle window (C36, C38, C39, C44, C46, C47, C48, C50, C52). Each ID resolved to an in-file rationale block tied to an actually-implemented arm. No orphan fix-IDs (referenced-but-not-implemented), no stale `# TODO cycle N` markers.

## Notes (<75)

- B56-1 (conf 60): `test_c54_ad3` covers only the inner `_fn_table_sig` NIE-catch (line 146); the outer `differentiate()` NIE-catch (line 184) for novel top-level `expr` subclasses lacks a direct regression test. Symmetric to the inner site so risk is low, but a one-line test (call `differentiate(SyntheticUnknown(), "x")` and assert no raise + cache-bypass warning emitted) would close the gap.
- B56-2 (conf 45): Comment at `autodiff.py:126` says "`_inline_user_calls` at line ~365" — actual @pure check is at line 361, arity check at line 364. The `~` prefix is acceptable but a precise line number would survive future edits better. Cosmetic.
- B56-3 (conf 35): `_fn_table_sig` sig grammar `{name}/{arity}/{attrs}/{body_hash}` uses `,` to join sorted attrs. A pathological autotune attr `"autotune:K=1,2"` plus a hypothetical attr literally named `"2"` would join to the same string as `"autotune:K=1"` + `"2"` — but the parser doesn't produce attrs named `"2"` (must be ident-shaped per `parser.py:270`), so the ambiguity is unreachable. Worth documenting as an assumption if attr-grammar evolves.
- B56-4 (conf 25): `_DIFF_CACHE` is module-global with no per-test reset in `test_c54_ad1` / `test_c54_ad2`; both call `_fn_table_sig` directly (not `differentiate`), so cache state is irrelevant — but `test_c52_ad1` (sibling, prior cycle) does the right thing with `clear_diff_cache()`. Pre-existing pattern.

**Cycle 56 code-review: PASS.** 0 findings at conf >=75. Counter advances toward 5-clean if sibling audits (silent-failures, type-design) also pass.
